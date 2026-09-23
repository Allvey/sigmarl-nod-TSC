"""Stage 9 identity-aligned opinions, local effects and frozen semantics."""
import copy
import json

import pytest
import torch
from tensordict import TensorDict

from utilities.helper_training import Parameters
from utilities.nod_marl.barrier import aligned_opinion_kappa, fixed_barrier_advantage


def opinion_states():
    n = 3
    ids = torch.tensor([[[[1, 2], [0, 2], [0, 1]]]])
    generation = torch.tensor([2, 4, 6])
    context = torch.zeros(1, 1, n, 2, 72)
    context[..., -1] = torch.tensor([[-1., 1.], [0., .5], [-.5, 0.]])
    info = TensorDict(dict(nod_actor_edge_context=context,
        nod_actor_edge_mask=torch.ones_like(ids, dtype=torch.bool),
        nod_edge_mask=torch.ones_like(ids, dtype=torch.bool),
        nod_actor_context_ready=torch.ones(1, 1, n, 1, dtype=torch.bool),
        nod_neighbor_indices=ids, nod_neighbor_generation=generation[ids]), [1, 1, n])
    g = -torch.ones(1, 1, n, n+2)
    ego = generation.view(1, 1, n, 1).expand_as(g)
    other = torch.cat([generation.view(1, 1, 1, n).expand(1, 1, n, n),
                       generation.view(1, 1, n, 1).expand(1, 1, n, 2)], -1)
    valid = torch.ones_like(g, dtype=torch.bool)
    valid[..., :n] &= ~torch.eye(n, dtype=torch.bool)
    state = dict(g=g, ego_gen=ego, other_gen=other, valid=valid)
    td = TensorDict(dict(agents=TensorDict(dict(info=info), [1, 1, n])), [1, 1])
    return td, state


def mapped(td, state):
    return aligned_opinion_kappa(td, state, minimum=.04, maximum=.06)


def test_cached_opinion_is_world_aligned_and_permutation_invariant():
    td, state = opinion_states()
    kappa, valid, z = mapped(td, state)
    assert valid.sum() == 6
    torch.testing.assert_close(kappa[0, 0, 0], torch.tensor([.04, .04, .06]))
    permuted = td.clone()
    info = permuted['agents', 'info']
    for key in ['nod_neighbor_indices', 'nod_neighbor_generation', 'nod_actor_edge_mask', 'nod_edge_mask']:
        info[key] = info[key].flip(-1)
    info['nod_actor_edge_context'] = info['nod_actor_edge_context'].flip(-2)
    for a, b in zip(mapped(permuted, state), (kappa, valid, z)):
        torch.testing.assert_close(a, b)
    # The next frame's opinions are irrelevant to the current action constraint.
    td['next'] = permuted.clone()
    td['next', 'agents', 'info', 'nod_actor_edge_context'][..., -1] = 999.
    torch.testing.assert_close(mapped(td, state)[0], kappa)


@pytest.mark.parametrize('missing', ['ready', 'generation', 'nan', 'mask', 'duplicate', 'absent'])
def test_missing_opinion_is_conservative_not_neutral(missing):
    td, state = opinion_states()
    info = td['agents', 'info']
    if missing == 'ready': info['nod_actor_context_ready'][..., 0, :] = False
    elif missing == 'generation': info['nod_neighbor_generation'][..., 0, :] += 1
    elif missing == 'nan': info['nod_actor_edge_context'][..., 0, :, -1] = float('nan')
    elif missing == 'mask': info['nod_actor_edge_mask'][..., 0, :] = False
    elif missing == 'duplicate': info['nod_neighbor_indices'][..., 0, :] = 1
    else: info.del_('nod_actor_edge_context')
    kappa, available, _ = mapped(td, state)
    assert not available[..., 0, :].any()
    torch.testing.assert_close(kappa[..., 0, :], torch.full((1, 1, 3), .04))


def test_z_changes_only_its_safe_pair_and_neutral_matches_fixed():
    td, state = opinion_states()
    task = torch.ones(1, 1, 3, 1, requires_grad=True)
    value, nxt = state['g'].clone().requires_grad_(), state['g'].clone()
    nxt[..., 0, 1] = -.95  # At the fixed-kappa decision boundary.
    def run(k):
        return fixed_barrier_advantage(task, state, state, value, nxt,
            kappa=.05, road_kappa=.05, nu=1., strength=.1, pair_kappa=k)
    td['agents', 'info', 'nod_actor_edge_context'][..., -1] = 0
    neutral, *_ = mapped(td, state)
    torch.testing.assert_close(run(neutral)[0], run(None)[0])
    td['agents', 'info', 'nod_actor_edge_context'][..., 0, 0, -1] = -1
    low, li = run(mapped(td, state)[0])
    td['agents', 'info', 'nod_actor_edge_context'][..., 0, 0, -1] = 1
    high, hi = run(mapped(td, state)[0])
    assert hi['delta'][..., 0, 1] < li['delta'][..., 0, 1]
    assert high[..., 0, :] > low[..., 0, :]
    assert not high.requires_grad
    difference = hi['delta'] != li['delta']
    assert difference.sum() == 1 and difference[..., 0, 1].all()
    for v in [0., .2]:
        with torch.no_grad(): value[..., 0, 1] = v
        torch.testing.assert_close(run(neutral)[1]['delta'], run(mapped(td, state)[0])[1]['delta'])


@pytest.mark.parametrize('kwargs', [dict(is_using_nod_opinion=False),
    dict(safety_barrier_kappa_min=.01), dict(safety_barrier_kappa_max=1.),
    dict(is_using_nod_actor=False)])
def test_opinion_configuration_rejects_uncontrolled_setups(kwargs):
    options=dict(is_using_safety_value_shadow=True, safety_control_mode='barrier_opinion', nod_freeze_training=True)
    options.update(kwargs)
    with pytest.raises(ValueError, match='Stage-9'):
        Parameters(**options)


def test_from_scratch_trains_nod_and_opinion_barrier_without_loading(tmp_path, monkeypatch):
    from utilities import mappo_cavs as training
    from utilities.nod_marl.trainer import NODOpinionManager
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    def forbid_load(*args, **kwargs):
        raise AssertionError('From-scratch training must not load checkpoints')
    monkeypatch.setattr(training, '_load_policy_checkpoint', forbid_load)
    monkeypatch.setattr(training, '_load_nod_if_available', forbid_load)
    original = NODOpinionManager.train_on_rollout
    changed = []
    def track_nod(manager, td):
        before = {k: v.clone() for k, v in manager.model.state_dict().items()}
        metrics = original(manager, td)
        changed.append(any(not torch.equal(v, manager.model.state_dict()[k]) for k, v in before.items()))
        return metrics
    monkeypatch.setattr(NODOpinionManager, 'train_on_rollout', track_nod)
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        p = Parameters.from_json('configs/archive/staged_history/config.json')
        assert p.safety_control_mode == 'barrier_opinion' and not p.nod_freeze_training
        assert p.training_init_checkpoint is None and not p.is_load_model and not p.is_continue_train
        p.seed = 571; p.n_iters = 2; p.num_epochs = 1
        p.frames_per_batch = 64; p.total_frames = 128; p.minibatch_size = 32
        p.num_vmas_envs = 4; p.max_steps = 16; p.nod_sequence_length = 8
        p.nod_num_epochs = p.safety_num_epochs = p.safety_value_num_epochs = 1
        p.safety_value_num_envs = 2; p.safety_value_rollout_steps = 16
        p.safety_barrier_warmup_batches = 1
        p.where_to_save = str(tmp_path) + '/'
        env, *_ = training.mappo_cavs(p)
        assert env.scenario.safety_value_manager.rollouts == 2
        env.close()
        assert changed and any(changed)
        d = json.loads(next(tmp_path.glob('reward*_data.json')).read_text())
        m = d['safety_value_metrics_list']
        assert m[0]['barrier_ready'] == 0
        assert m[1]['barrier_active'] and m[1]['actor_updates'] > 0
        assert m[1]['opinion_valid_count'] > 0 and m[1]['opinion_advantage_changed_count'] > 0
        assert m[1]['actor_probe_mode_delta_abs'] > 0
        assert all(not row['training_frozen'] for row in d['nod_metrics_list'])
        assert sum(row['optimizer_updates'] for row in d['nod_metrics_list']) > 0
    finally:
        torch.set_num_threads(threads)


def test_frozen_opinion_training_from_same_snapshot(tmp_path, monkeypatch):
    from utilities.mappo_cavs import mappo_cavs
    from utilities.nod_marl.trainer import NODOpinionManager
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        p = Parameters.from_json('configs/archive/staged_history/config.json')
        p.seed = 571; p.n_iters = 2; p.num_epochs = 1
        p.frames_per_batch = 64; p.total_frames = 128; p.minibatch_size = 32
        p.num_vmas_envs = 4; p.max_steps = 16; p.nod_sequence_length = 8
        p.nod_num_epochs = p.safety_num_epochs = p.safety_value_num_epochs = 1
        p.safety_value_num_envs = 2; p.safety_value_rollout_steps = 16
        p.safety_barrier_warmup_batches = 1
        p.nod_freeze_training = False; p.training_init_checkpoint = None
        p.safety_control_mode = 'barrier_fixed'; p.is_load_model = p.is_continue_train = False
        p.where_to_save = str(tmp_path / 'start') + '/'
        env, *_ = mappo_cavs(p); env.close()
        initial = torch.load(tmp_path / 'start/final_nod.pth')['model']
        original_train = NODOpinionManager.train_on_rollout
        def no_frozen_fit(manager, td):
            assert not manager.parameters.nod_freeze_training
            return original_train(manager, td)
        monkeypatch.setattr(NODOpinionManager, 'train_on_rollout', no_frozen_fit)
        for mode in ['barrier_fixed', 'barrier_opinion']:
            p.training_init_checkpoint = str(tmp_path / 'start/final')
            p.nod_freeze_training = True; p.safety_control_mode = mode
            p.where_to_save = str(tmp_path / mode) + '/'
            env, *_ = mappo_cavs(p)
            assert all(not w.requires_grad for w in env.scenario.nod_manager.model.parameters())
            env.close()
            frozen = torch.load(tmp_path / mode / 'final_nod.pth')['model']
            assert all(torch.equal(initial[k], frozen[k]) for k in initial)
        fixed = torch.load(tmp_path / 'barrier_fixed/final_policy.pth')
        opinion = torch.load(tmp_path / 'barrier_opinion/final_policy.pth')
        assert any(not torch.equal(fixed[k], opinion[k]) for k in fixed)
        d=json.loads(next((tmp_path/'barrier_opinion').glob('reward*_data.json')).read_text())
        metrics=d['safety_value_metrics_list'][-1]
        assert metrics['opinion_valid_count'] > 0
        assert metrics['opinion_advantage_changed_count'] > 0
        assert metrics['actor_probe_mode_delta_abs'] > 0
        assert all(m['optimizer_updates'] == 0 and m['training_frozen'] for m in d['nod_metrics_list'])
        # Ordinary resume overrides initialization prefix and preserves the warmup contract.
        p.training_init_checkpoint = '/not/a/checkpoint'
        p.is_load_model = p.is_load_final_model = p.is_continue_train = True
        p.n_iters = 1; p.total_frames = 64
        env, *_ = mappo_cavs(p)
        assert env.scenario.safety_value_manager.barrier_fit_batches == 3
        env.close()
    finally:
        torch.set_num_threads(threads)
