"""Offline PPO gate gradients, causal replay, masks and checkpoint isolation.

No environment, simulator or training job is run by this test module.
"""
import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import (
    MultiAgentMLP, ProbabilisticActor, SafeProbabilisticTensorDictSequential, TanhNormal,
)
from torchrl.objectives import ClipPPOLoss

from utilities.helper_training import Parameters
from utilities.mappo_cavs import BoundedNormalParamExtractor, _load_policy_checkpoint
from utilities.nod_marl.opinion import kl_proximal_update
from utilities.nod_marl.policy import (
    NODActorInputModule, NODEvidenceGate, NOD_ACTOR_CONTEXT_READY_KEY,
    NOD_ACTOR_EDGE_CONTEXT_KEY, NOD_ACTOR_EDGE_MASK_KEY,
    NOD_ACTOR_MESSAGE_KEY, NOD_ACTOR_OBSERVATION_KEY,
    NOD_EVIDENCE_GATE_CONTEXT_KEY, NOD_EVIDENCE_WEIGHT_KEY,
)
from utilities.nod_marl.trainer import NODOpinionManager


CONFIG = 'configs/ablations/config_dgppo_nod_future_evidence_ppo_gate_control.json'


def make_actor(mode='ppo', sequence_length=1, gate_only_iters=0, target='evidence'):
    p = Parameters.from_json(CONFIG)
    p.n_agents = 2
    p.nod_hidden_dim = 8
    p.nod_evidence_gate_mode = mode
    p.nod_evidence_gate_sequence_length = sequence_length
    p.nod_evidence_gate_only_iters = gate_only_iters
    p.nod_evidence_gate_target = target
    manager = NODOpinionManager(p)
    manager.model.requires_grad_(False)
    with torch.no_grad():
        manager.model.evidence_head[-1].bias.fill_(1.0)
    module = NODActorInputModule(
        observation_key=('agents', 'observation'), base_observation_dim=5,
        nod_manager=manager, action_dim=2, message_dim=4, message_hidden_dim=8,
        evidence_gate_mode=mode,
        evidence_gate_sequence_length=sequence_length,
        evidence_gate_only_iters=gate_only_iters,
        evidence_gate_target=target,
    )
    net = torch.nn.Sequential(
        MultiAgentMLP(n_agent_inputs=11, n_agent_outputs=4, n_agents=2,
                      centralised=False, share_params=True, device='cpu',
                      depth=1, num_cells=16, activation_class=torch.nn.Tanh),
        BoundedNormalParamExtractor(),
    )
    actor = ProbabilisticActor(
        TensorDictModule(net, in_keys=[NOD_ACTOR_OBSERVATION_KEY],
                         out_keys=[('agents', 'loc'), ('agents', 'scale')]),
        in_keys=[('agents', 'loc'), ('agents', 'scale')],
        out_keys=[('agents', 'action')], distribution_class=TanhNormal,
        distribution_kwargs={'min': -1., 'max': 1.}, return_log_prob=True,
        log_prob_key=('agents', 'sample_log_prob'),
    )
    return SafeProbabilisticTensorDictSequential(module, *actor.module), module, manager, p


def cached_data(manager, module):
    pair = torch.zeros(1, 5, 2, 1, 20)
    mask = torch.ones(1, 5, 2, 1, dtype=torch.bool)
    mask[:, 3, 0] = False
    ego = torch.ones(1, 5, 2, dtype=torch.long)
    neighbor = torch.ones_like(mask, dtype=torch.long)
    neighbor[:, 4, 1] = 2  # Identity replacement must remain neutral.
    with torch.no_grad():
        out, _ = manager.model.forward_sequence(
            pair, torch.ones_like(mask), ego, neighbor,
            opinion_mask=mask, evidence_override=torch.ones_like(mask, dtype=torch.float32),
            evidence_gate=module.evidence_gate,
        )
    gate = out['gate_context'].squeeze(0)
    edge = torch.cat([gate[..., :-5], out['attention'].squeeze(0).unsqueeze(-1),
                      out['z'].squeeze(0).unsqueeze(-1)], -1)
    data = TensorDict({
        ('agents', 'observation'): torch.randn(5, 2, 5),
        NOD_ACTOR_EDGE_CONTEXT_KEY: edge,
        NOD_ACTOR_EDGE_MASK_KEY: torch.ones(5, 2, 1, dtype=torch.bool),
        NOD_ACTOR_CONTEXT_READY_KEY: torch.ones(5, 2, 1, dtype=torch.bool),
        NOD_EVIDENCE_GATE_CONTEXT_KEY: gate,
    }, batch_size=[5])
    return data, out


def test_split_solver_preserves_old_update_and_has_correct_beta_gradient():
    args = [torch.tensor([x], dtype=torch.float64, requires_grad=True)
            for x in (.1, .6, .8, 0., 1., 1.)]
    kwargs = dict(bifurcation_gain=2., observation_weight=1., kl_weight=5.,
                  max_iterations=64, tolerance=0.)
    old = kl_proximal_update(*args, **kwargs)
    explicit = kl_proximal_update(*args, evidence_attention=args[1], **kwargs)
    torch.testing.assert_close(old, explicit, rtol=0, atol=0)
    for lhs, rhs in zip(torch.autograd.grad(old.sum(), args, retain_graph=True),
                        torch.autograd.grad(explicit.sum(), args)):
        torch.testing.assert_close(lhs, rhs)
    beta = torch.tensor([.4], dtype=torch.float64, requires_grad=True)
    root = kl_proximal_update(*args, evidence_attention=beta, **kwargs)
    gradient = torch.autograd.grad(root.sum(), beta)[0]
    delta = 1e-5
    numeric = (kl_proximal_update(*args, evidence_attention=beta.detach()+delta, **kwargs)
               - kl_proximal_update(*args, evidence_attention=beta.detach()-delta, **kwargs)) / (2*delta)
    torch.testing.assert_close(gradient, numeric, rtol=1e-5, atol=1e-7)
    assert gradient.item() > 0


def test_gate_starts_at_risk_attention_and_masks_absent_interactions():
    gate = NODEvidenceGate(context_dim=19, hidden_dim=8, max_logit_shift=2.)
    context = torch.randn(3, 19)
    context[:, -5] = torch.tensor([.2, .5, .8])
    context[:, -1] = torch.tensor([1., 1., 0.])
    torch.testing.assert_close(gate(context), torch.tensor([.2, .5, 0.]))
    for bias in (-100., 100.):
        with torch.no_grad():
            gate.net[-1].bias.fill_(bias)
        beta = gate(context)
        assert ((beta >= 0.) & (beta <= 1.)).all()
        assert beta[-1] == 0
        shift = torch.logit(beta[:2]) - torch.logit(context[:2, -5])
        assert shift.abs().max() <= 2. + 1e-5


def test_replay_matches_collection_and_can_shuffle_without_advancing_history():
    _, module, manager, _ = make_actor()
    data, out = cached_data(manager, module)
    before = data[NOD_ACTOR_EDGE_CONTEXT_KEY].clone()
    expected_message, _ = module.aggregator(before, data[NOD_ACTOR_EDGE_MASK_KEY])
    module(data)
    torch.testing.assert_close(data[NOD_ACTOR_MESSAGE_KEY], expected_message, atol=1e-6, rtol=1e-6)
    reference = data[NOD_ACTOR_MESSAGE_KEY].detach().clone()
    permutation = torch.tensor([4, 0, 2, 1, 3])
    shuffled = data[permutation].clone()
    module(shuffled)
    torch.testing.assert_close(shuffled[NOD_ACTOR_MESSAGE_KEY], reference[permutation])
    torch.testing.assert_close(data[NOD_ACTOR_EDGE_CONTEXT_KEY], before, rtol=0, atol=0)
    assert manager.online_state is None
    assert out['z'][0, 0].eq(0.).all()
    assert out['z'][0, 3, 0].eq(0.).all()
    assert out['z'][0, 4, 1].eq(0.).all()
    with torch.no_grad():
        module.evidence_gate.net[-1].bias.fill_(1.)
    module(data)
    assert not torch.allclose(reference[1:3], data[NOD_ACTOR_MESSAGE_KEY][1:3])
    torch.testing.assert_close(data[NOD_ACTOR_EDGE_CONTEXT_KEY], before, rtol=0, atol=0)
    assert data[NOD_EVIDENCE_WEIGHT_KEY][0].eq(0.).all()


@pytest.mark.parametrize('target', ['evidence', 'attention'])
def test_online_collection_and_cached_replay_use_the_same_opinion_step(target):
    _, module, manager, _ = make_actor(target=target)
    with torch.no_grad():
        module.evidence_gate.net[-1].bias.fill_(.7)
    pair = torch.zeros(1, 2, 1, 20)
    pair[..., 17] = 1.0
    # Two approaching vehicles provide an active local conflict reference.
    fields = {
        ('agents', 'observation'): torch.randn(1, 2, 5),
        ('agents', 'info', 'nod_pair_features'): pair,
        ('agents', 'info', 'nod_edge_mask'): torch.ones(1, 2, 1, dtype=torch.bool),
        ('agents', 'info', 'nod_neighbor_indices'): torch.tensor([[[1], [0]]]),
        ('agents', 'info', 'nod_ego_generation'): torch.ones(1, 2, dtype=torch.long),
        ('agents', 'info', 'nod_neighbor_generation'): torch.ones(1, 2, 1, dtype=torch.long),
        ('agents', 'info', 'nod_world_pos'): torch.tensor([[[0., 0.], [.3, 0.]]]),
        ('agents', 'info', 'nod_world_vel'): torch.tensor([[[.1, 0.], [-.1, 0.]]]),
    }
    for _ in range(3):
        data = TensorDict(fields, batch_size=[1]).clone()
        with torch.no_grad():
            module(data)
        message = data[NOD_ACTOR_MESSAGE_KEY].clone()
        state = {key: value.clone() for key, value in manager.online_state.items()}
        module(data)
        torch.testing.assert_close(message, data[NOD_ACTOR_MESSAGE_KEY], atol=1e-6, rtol=1e-6)
        for key, value in manager.online_state.items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    assert data[NOD_EVIDENCE_GATE_CONTEXT_KEY][..., -1].bool().any()


def test_real_clip_ppo_updates_gate_and_leaves_predictor_and_cache_frozen():
    torch.manual_seed(42)
    actor, module, manager, _ = make_actor()
    data, _ = cached_data(manager, module)
    frozen = {k: v.clone() for k, v in manager.model.state_dict().items()}
    context = data[NOD_EVIDENCE_GATE_CONTEXT_KEY].clone()
    with torch.no_grad():
        actor(data)
    critic = TensorDictModule(torch.nn.Linear(5, 1),
                             in_keys=[('agents', 'observation')],
                             out_keys=[('agents', 'state_value')])
    loss = ClipPPOLoss(actor=actor, critic=critic, normalize_advantage=False)
    loss.set_keys(action=('agents', 'action'), sample_log_prob=('agents', 'sample_log_prob'),
                  value=('agents', 'state_value'), advantage=('agents', 'advantage'),
                  value_target=('agents', 'value_target'))
    data.set(('agents', 'advantage'), torch.randn(5, 2, 1))
    data.set(('agents', 'value_target'), torch.zeros(5, 2, 1))
    optim = torch.optim.Adam(loss.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in module.evidence_gate.parameters()]
    losses = loss(data)
    losses['loss_objective'].backward()
    assert module.evidence_gate.net[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in manager.model.parameters())
    optim.step()
    assert any(not torch.equal(p.detach(), b) for p, b in zip(module.evidence_gate.parameters(), before))
    for key, value in manager.model.state_dict().items():
        torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
    torch.testing.assert_close(data[NOD_EVIDENCE_GATE_CONTEXT_KEY], context, rtol=0, atol=0)


def test_gate_checkpoint_introduction_is_explicit_and_resume_is_strict(tmp_path):
    old, _, _, _ = make_actor('fixed')
    new, module, _, p = make_actor()
    p.training_init_checkpoint = str(tmp_path / 'source')
    path = p.training_init_checkpoint + '_policy.pth'
    torch.save(old.state_dict(), path)
    assert _load_policy_checkpoint(path, new, p, actor_base_observation_dim=5, use_nod_actor=True) == 'gate_initialized'
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
    with torch.no_grad():
        module.evidence_gate.net[-1].bias.fill_(.4)
    torch.save(new.state_dict(), path)
    restored, _, _, _ = make_actor()
    p.is_load_model = True
    assert _load_policy_checkpoint(path, restored, p, actor_base_observation_dim=5, use_nod_actor=True) == 'loaded'
    for key, value in new.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        _load_policy_checkpoint(path, old, p, actor_base_observation_dim=5, use_nod_actor=True)
    torch.save(old.state_dict(), path)
    with pytest.raises(RuntimeError):
        _load_policy_checkpoint(path, restored, p, actor_base_observation_dim=5, use_nod_actor=True)


def test_configuration_requires_frozen_future_evidence_and_matches_fixed_control():
    p = Parameters.from_json(CONFIG)
    fixed = Parameters.from_json('configs/ablations/config_dgppo_nod_future_evidence_control.json')
    for key in ('training_init_checkpoint', 'seed', 'n_iters', 'frames_per_batch',
                'num_epochs', 'lr', 'dgppo_alpha_gain', 'nod_fixed_variance_value'):
        assert getattr(p, key) == getattr(fixed, key)
    assert Parameters.from_dict(p.to_dict()).to_dict() == p.to_dict()
    for change in (dict(nod_freeze_training=False), dict(nod_evidence_target='geometric'),
                   dict(nod_training_mode='candidate_only'), dict(nod_actor_opinion_mode='neutral'),
                   dict(nod_evidence_gate_lr=0), dict(nod_evidence_gate_mode='unknown')):
        with pytest.raises(ValueError):
            Parameters.from_dict(dict(p.to_dict(), **change))


def test_optional_gate_does_not_advance_the_actor_initialization_rng():
    torch.manual_seed(123)
    fixed_actor, _, _, _ = make_actor('fixed')
    fixed_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(123)
    gated_actor, _, _, _ = make_actor()
    assert torch.equal(torch.random.get_rng_state(), fixed_rng)
    for key, value in fixed_actor.state_dict().items():
        torch.testing.assert_close(value, gated_actor.state_dict()[key], rtol=0, atol=0)
