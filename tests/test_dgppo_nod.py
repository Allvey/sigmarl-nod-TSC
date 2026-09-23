"""Offline checks for local NOD + fixed-alpha DGPPO. No environment rollout."""
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, SafeProbabilisticTensorDictSequential, TanhNormal

from utilities.helper_training import Parameters
from utilities.mappo_cavs import BoundedNormalParamExtractor, _load_policy_checkpoint, _load_nod_if_available, _bind_frozen_nod
from utilities.nod_marl.interaction import build_directed_interactions, nod_observation_inputs
from utilities.nod_marl.policy import NODActorInputModule, NOD_ACTOR_OBSERVATION_KEY
from utilities.nod_marl.trainer import NODOpinionManager
from utilities.nod_marl.safety_value import SafetyValueManager
from utilities.nod_marl.dgppo import prepare_dgppo_advantage


def parameters(finetune=False):
    suffix = '_finetune' if finetune else ''
    return Parameters.from_json(
        f'configs/archive/nod_history/config_dgppo_nod_fixed{suffix}.json'
    )


@pytest.mark.parametrize('finetune', [False, True])
def test_configuration_roundtrip_and_fixed_alpha(finetune):
    p = parameters(finetune)
    restored = Parameters.from_dict(p.to_dict())
    assert restored.to_dict() == p.to_dict()
    assert restored.nod_observation_mode == 'local_kinematics'
    assert restored.is_using_nod_actor and restored.is_using_nod_opinion
    assert restored.dgppo_alpha == 10 and restored.dgppo_task_mode == 'gated'
    assert restored.safety_control_mode == 'dgppo'
    old = Parameters.from_json('configs/archive/dgppo_history/config_ppo_original_dgppo.json')
    assert not old.is_using_nod_actor and old.nod_observation_mode == 'legacy_paths'


@pytest.mark.parametrize('change', [dict(nod_observation_mode='legacy_paths'),
    dict(is_using_nod_actor=False), dict(is_using_nod_opinion=False),
    dict(is_observe_ref_path_other_agents=True), dict(nod_observation_mode='unknown')])
def test_invalid_local_combinations(change):
    with pytest.raises(ValueError):
        Parameters.from_dict(dict(parameters().to_dict(), **change))


def interactions(paths):
    return build_directed_interactions(
        torch.tensor([[[0., 0.], [.3, 0.], [2., 0.]]]),
        torch.tensor([[[.1, 0.], [-.1, 0.], [0., 0.]]]), torch.zeros(1, 3), paths, 0,
        sensing_range=.8, interaction_distance=.48, ttc_limit=2.,
        conflict_radius=.08, max_speed=1., observation_mode='local_kinematics')


def test_local_geometry_never_reads_paths_or_route_masks():
    first = interactions(torch.randn(1, 3, 3, 2))
    second = interactions(None)  # No path access, even without a path tensor.
    third = interactions(torch.full((1, 3, 3, 2), float('nan')))
    for key in first:
        torch.testing.assert_close(first[key], second[key])
        torch.testing.assert_close(first[key], third[key])
    assert first['edge_mask'].tolist() == [[True, False]]
    assert first['features'][..., 9:16].count_nonzero() == 0
    assert first['features'][0, 1].count_nonzero() == 0


def input_data():
    pair = torch.zeros(1, 2, 1, 20)
    pair[..., 6] = .4; pair[..., 8] = .5; pair[..., 17] = 1.
    return TensorDict({('agents', 'observation'): torch.randn(1, 2, 5),
        ('agents', 'info', 'nod_pair_features'): pair,
        ('agents', 'info', 'nod_edge_mask'): torch.ones(1, 2, 1, dtype=torch.bool),
        ('agents', 'info', 'nod_neighbor_indices'): torch.tensor([[[1], [0]]]),
        ('agents', 'info', 'nod_ego_generation'): torch.zeros(1, 2, dtype=torch.long),
        ('agents', 'info', 'nod_neighbor_generation'): torch.zeros(1, 2, 1, dtype=torch.long),
        ('agents', 'info', 'act_vel'): torch.randn(1, 2),
        ('agents', 'info', 'act_steer'): torch.randn(1, 2)}, [1])


def test_local_opinion_sequence_is_independent_of_hidden_path_fields():
    p = parameters(); p.n_agents = 2
    left, right = NODOpinionManager(p), NODOpinionManager(p)
    right.model.load_state_dict(left.model.state_dict())
    for _ in range(4):
        a = input_data(); b = a.clone()
        b['agents', 'info', 'nod_pair_features'][..., 9:16] = 999.
        b['agents', 'info', 'nod_edge_mask'].zero_()
        x, y = left.online_step(a), right.online_step(b)
        for key in x: torch.testing.assert_close(x[key], y[key])
        # The same sanitizer accepts ordered [batch,time,agent,neighbor,feature] input.
        pa, ma = nod_observation_inputs(a['agents', 'info', 'nod_pair_features'].unsqueeze(1),
                                      a['agents', 'info', 'nod_edge_mask'].unsqueeze(1), p.nod_observation_mode)
        pb, mb = nod_observation_inputs(b['agents', 'info', 'nod_pair_features'].unsqueeze(1),
                                      b['agents', 'info', 'nod_edge_mask'].unsqueeze(1), p.nod_observation_mode)
        torch.testing.assert_close(pa, pb); assert torch.equal(ma, mb)


def make_policy(p, with_nod):
    manager = NODOpinionManager(p)
    module = NODActorInputModule(observation_key=('agents', 'observation'),
        base_observation_dim=5, nod_manager=manager, action_dim=2,
        message_dim=4, message_hidden_dim=8)
    net = torch.nn.Sequential(MultiAgentMLP(n_agent_inputs=11 if with_nod else 5,
        n_agent_outputs=4, n_agents=2, centralised=False, share_params=True,
        device='cpu', depth=2, num_cells=16, activation_class=torch.nn.Tanh), BoundedNormalParamExtractor())
    actor = ProbabilisticActor(TensorDictModule(net,
        in_keys=[NOD_ACTOR_OBSERVATION_KEY if with_nod else ('agents', 'observation')],
        out_keys=[('agents', 'loc'), ('agents', 'scale')]),
        in_keys=[('agents', 'loc'), ('agents', 'scale')], out_keys=[('agents', 'action')],
        distribution_class=TanhNormal, distribution_kwargs={'min': -1., 'max': 1.},
        return_log_prob=True, log_prob_key=('agents', 'sample_log_prob'))
    return (SafeProbabilisticTensorDictSequential(module, *actor.module) if with_nod else actor), manager


def test_base_actor_migration_preserves_distribution_and_new_inputs_learn(tmp_path):
    p = parameters(); p.n_agents = 2
    old, _ = make_policy(p, False); new, nod = make_policy(p, True)
    path = tmp_path / 'policy.pth'; torch.save(old.state_dict(), path)
    assert _load_policy_checkpoint(str(path), new, p, actor_base_observation_dim=5, use_nod_actor=True) == 'migrated'
    data = input_data(); base_data = data.clone()
    d0, d1 = old.get_dist(base_data), new.get_dist(data)
    for key in ('loc', 'scale'):
        torch.testing.assert_close(data['agents', key], base_data['agents', key])
    torch.testing.assert_close(d0.mode, d1.mode)
    torch.testing.assert_close(d0.log_prob(d0.mode), d1.log_prob(d0.mode))
    # Real PPO log-prob gradient can reach newly added columns, while cached
    # opinion context stays detached from NOD's recurrent model.
    action = d1.sample().detach()
    (-new.get_dist(data).log_prob(action).mean()).backward()
    first = next(v for k, v in new.named_parameters() if 'agent_networks' in k and v.ndim == 2 and v.shape[-1] == 11)
    assert first.grad[:, 5:].abs().sum() > 0
    assert all(v.grad is None for v in nod.model.parameters())
    torch.save(new.state_dict(), path)
    assert _load_policy_checkpoint(str(path), new, p, actor_base_observation_dim=5, use_nod_actor=True) == 'loaded'


def test_nod_contract_and_explicit_fresh_initialization(tmp_path):
    p = parameters(); local = NODOpinionManager(p)
    legacy_p = Parameters.from_json('configs/archive/staged_history/config.json'); legacy = NODOpinionManager(legacy_p)
    old = legacy.checkpoint_state(); old.pop('observation_mode')
    assert legacy.load_checkpoint(old)  # Existing path-based snapshots still work.
    assert not local.load_checkpoint(old)
    assert not legacy.load_checkpoint(local.checkpoint_state())
    path = str(tmp_path / 'nod.pth')
    with pytest.raises(FileNotFoundError): _load_nod_if_available(path, local, p, load_optimizer=False)
    assert not _load_nod_if_available(path, local, p, load_optimizer=False, allow_fresh=True)
    p.nod_freeze_training = True
    with pytest.raises(ValueError): _bind_frozen_nod(local, SimpleNamespace(barrier_contract={}), p)
    torch.save(old, path)
    with pytest.raises(ValueError): _load_nod_if_available(path, local, p, load_optimizer=False)
    torch.save(local.checkpoint_state(), path)
    assert _load_nod_if_available(path, local, p, load_optimizer=True)
    _bind_frozen_nod(local, SimpleNamespace(barrier_contract={}), p)
    assert not any(v.requires_grad for v in local.model.parameters())


def test_value_weights_reused_but_policy_change_restarts_warmup(tmp_path):
    old = SafetyValueManager(Parameters.from_json('configs/archive/dgppo_history/config_ppo_original_dgppo.json'), 5, ('agents', 'observation'))
    new = SafetyValueManager(parameters(), 5, ('agents', 'observation'))
    assert old.contract == new.contract and old.loss_contract == new.loss_contract
    old.barrier_fit_batches = 99
    path = tmp_path / 'value.pth'; torch.save(old.checkpoint_state(), path)
    assert new.load_if_available(path, load_optimizer=True)
    assert new.barrier_fit_batches == 0
    for a, b in zip(old.model.parameters(), new.model.parameters()): torch.testing.assert_close(a, b)


def test_fixed_alpha_dgppo_does_not_consume_opinions():
    p = parameters()
    g = -torch.ones(1, 3, 2, 4)
    state = dict(g=g, valid=torch.ones_like(g, dtype=torch.bool),
                 ego_gen=torch.zeros_like(g, dtype=torch.long), other_gen=torch.zeros_like(g, dtype=torch.long))
    manager = SimpleNamespace(parameters=p, enabled=True, barrier_fit_batches=99,
                              rollouts=1, state=lambda _: state, model=lambda s: s['g'])
    td = TensorDict({'advantage': torch.randn(1, 3, 2, 1),
        'next': TensorDict({'done': torch.zeros(1, 3, 1, dtype=torch.bool)}, [1, 3])}, [1, 3])
    other = td.clone()
    for batch, z in [(td, -1.), (other, 1.)]:
        batch.set(('agents', 'info', 'nod_actor_edge_context'), torch.full((1, 3, 2, 1, 72), z))
        prepare_dgppo_advantage(manager, batch, 'advantage')
    torch.testing.assert_close(td['advantage'], other['advantage'])
    assert torch.equal(td['agents', 'barrier_violation_mask'], other['agents', 'barrier_violation_mask'])
