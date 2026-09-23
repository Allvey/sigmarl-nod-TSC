"""Fixed barrier signs, PPO integration, isolation and checkpoint warmup."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from utilities.helper_training import Parameters
from utilities.nod_marl.barrier import fixed_barrier_advantage, prepare_barrier_advantage
from utilities.nod_marl.safety_value import SafetyValueManager
from utilities.nod_marl.safety import SafetyCriticManager


def constraints(values, successors, g=None):
    value = torch.tensor(values, dtype=torch.float).view(-1, 1, 3).requires_grad_()
    nxt = torch.tensor(successors, dtype=torch.float).view_as(value).requires_grad_()
    state = dict(g=-torch.ones_like(value) if g is None else torch.tensor(g).view_as(value),
                 valid=torch.ones_like(value, dtype=torch.bool),
                 ego_gen=torch.zeros_like(value, dtype=torch.long),
                 other_gen=torch.zeros_like(value, dtype=torch.long))
    return state, copy.deepcopy(state), value, nxt


def apply(task, state, following, value, nxt, strength=1.):
    return fixed_barrier_advantage(task, state, following, value, nxt,
                                   kappa=.05, road_kappa=.1, nu=1., strength=strength)


def test_barrier_signs_boundary_recovery_and_soft_start():
    s, sn, v, vn = constraints([[-1., -1., -1.], [0., -1., -1.], [.5, -1., -1.]],
                               [[-.9, -.95, -1.], [.2, -1., -1.], [.4, -1., -1.]])
    task = torch.tensor([2., -2., 3.]).view(3, 1, 1).requires_grad_()
    out, info = apply(task, s, sn, v, vn)
    torch.testing.assert_close(out.flatten(), torch.tensor([-.05, -.2, 3.]))
    assert not out.requires_grad
    assert info['recovery'][2, 0, 0]
    soft, _ = apply(task, s, sn, v, vn, .1)
    torch.testing.assert_close(soft.flatten(), torch.tensor([1.795, -1.82, 3.]))
    zero, _ = apply(task, s, sn, v, vn, 0.)
    torch.testing.assert_close(zero, task)
    # PPO decreases probability of the violated action under the full rule.
    log_ratio = torch.zeros_like(out, requires_grad=True)
    (-torch.exp(log_ratio) * out).mean().backward()
    assert log_ratio.grad[0] > 0 and log_ratio.grad[1] > 0 and log_ratio.grad[2] < 0
    assert task.grad is None and v.grad is None and vn.grad is None


@pytest.mark.parametrize('reason', ['ego', 'neighbor', 'disappeared', 'nonfinite'])
def test_unknown_constraints_fall_back_for_whole_agent(reason):
    s, sn, v, vn = constraints([-1.] * 3, [.5] * 3)
    if reason == 'ego':
        sn['ego_gen'][..., 0] += 1
    elif reason == 'neighbor':
        sn['other_gen'][..., 0] += 1
    elif reason == 'disappeared':
        sn['valid'][..., 0] = False
    else:
        with torch.no_grad():
            vn[..., 0] = float('nan')
    task = torch.ones(1, 1, 1)
    out, info = apply(task, s, sn, v, vn)
    assert not info['eligible'].any()
    torch.testing.assert_close(out, task)
    assert torch.isfinite(out).all()


def test_masked_edges_do_not_change_maximum_and_physical_unsafe_uses_recovery():
    s, sn, v, vn = constraints([-1., -1., -1.], [999., -.96, -1.])
    s['valid'][..., 0] = False
    task = torch.ones(1, 1, 1)
    out, info = apply(task, s, sn, v, vn)
    torch.testing.assert_close(out, task)  # Road kappa=.1: -.96 + .9 < 0.
    s['g'][..., 1] = .1
    out, info = apply(task, s, sn, v, vn)
    torch.testing.assert_close(out, torch.full_like(task, -.04))
    assert info['recovery'][..., 1].all()


@pytest.mark.parametrize('terminal', [False, True])
def test_rollout_cut_bootstraps_but_terminal_uses_observed_physics(terminal):
    s, sn, v, vn = constraints([-1.] * 3, [1.] * 3)
    s = {k: x.unsqueeze(0) for k, x in s.items()}
    sn = {k: x.unsqueeze(0) for k, x in sn.items()}
    s['prediction'], sn['prediction'] = v.unsqueeze(0), vn.unsqueeze(0)
    p = Parameters(is_using_safety_value_shadow=True, is_using_safety_constraint=True,
                   safety_control_mode='barrier_fixed', safety_barrier_strength=1.,
                   safety_barrier_warmup_batches=0)
    td = TensorDict(dict(advantage=torch.ones(1, 1, 1, 1), next=TensorDict(
        dict(done=torch.full((1, 1, 1), terminal, dtype=torch.bool)), (1, 1))), (1, 1))
    manager = SimpleNamespace(parameters=p, enabled=True, updates=1, barrier_fit_batches=1,
                              state=lambda d: s if 'next' in d.keys() else sn,
                              model=lambda state: state['prediction'])
    metrics = prepare_barrier_advantage(manager, td, 'advantage')
    expected = 1. if terminal else -1.95
    torch.testing.assert_close(td['advantage'], torch.full((1, 1, 1, 1), expected))
    assert metrics['barrier_active'] == float(not terminal)


def test_modes_and_barrier_checkpoint_migration(tmp_path):
    assert Parameters.from_dict({}).safety_control_mode == 'legacy_q'
    p = Parameters(is_using_safety_value_shadow=True, is_using_safety_constraint=True,
                   safety_control_mode='barrier_fixed', safety_barrier_warmup_batches=1)
    assert not SafetyCriticManager(p, 4, 2, 2, ('agents', 'observation')).constraint_enabled
    manager = SafetyValueManager(p, 4, ('agents', 'observation'))
    manager.barrier_fit_batches = manager.updates = 4
    path = tmp_path / 'value.pth'
    torch.save(manager.checkpoint_state(), path)
    restored = SafetyValueManager(p, 4, ('agents', 'observation'))
    assert restored.load_if_available(path, load_optimizer=True)
    assert restored.barrier_fit_batches == 4
    p.safety_barrier_strength = .2
    changed = SafetyValueManager(p, 4, ('agents', 'observation'))
    assert changed.load_if_available(path, load_optimizer=True)
    assert changed.barrier_fit_batches == 0 and 'fresh warmup' in changed.last_load_info
    for key, weight in manager.model.state_dict().items():
        torch.testing.assert_close(weight, changed.model.state_dict()[key])
    old = manager.checkpoint_state()
    del old['barrier_contract']; del old['barrier_fit_batches']
    torch.save(old, path)
    assert restored.load_if_available(path, load_optimizer=True)
    assert restored.barrier_fit_batches == 0


@pytest.mark.parametrize('kwargs', [dict(safety_control_mode='unknown'),
    dict(safety_barrier_kappa=0), dict(safety_barrier_road_kappa=1),
    dict(safety_barrier_nu=float('nan')), dict(safety_barrier_strength=1.1),
    dict(safety_barrier_warmup_batches=-1), dict(safety_control_mode='barrier_fixed')])
def test_invalid_barrier_configuration(kwargs):
    with pytest.raises(ValueError, match='fixed barrier'):
        Parameters(**kwargs)


@pytest.mark.parametrize('prioritized_replay', [False, True])
def test_fixed_barrier_changes_actor_preserves_targets_and_loads(tmp_path, monkeypatch, prioritized_replay):
    from utilities import mappo_cavs as training
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    actual_prepare = training.prepare_barrier_advantage
    calls = []

    def checked_prepare(manager, td, key):
        protected = ['value_target', ('agents', 'action'), ('agents', 'sample_log_prob'),
                     ('next', 'agents', 'reward')]
        before = {k: td.get(k).clone() for k in protected}
        task = td.get(key).clone()
        metrics = actual_prepare(manager, td, key)
        for k, tensor in before.items():
            torch.testing.assert_close(td.get(k), tensor)
        if metrics['barrier_active']:
            assert not torch.equal(td.get(key), task)
            assert not td.get(key).requires_grad
            calls.append(True)
        return metrics

    monkeypatch.setattr(training, 'prepare_barrier_advantage', checked_prepare)
    try:
        for mode in ['off', 'zero', 'fixed']:
            p = Parameters.from_json('configs/archive/staged_history/config.json')
            p.nod_freeze_training = False; p.training_init_checkpoint = None
            p.seed = 571; p.n_iters = 2; p.num_epochs = 1
            p.frames_per_batch = 32; p.total_frames = 64; p.minibatch_size = 16
            p.num_vmas_envs = 2; p.max_steps = 16; p.nod_sequence_length = 8
            p.nod_num_epochs = p.safety_num_epochs = p.safety_value_num_epochs = 1
            p.safety_value_num_envs = 2; p.safety_value_rollout_steps = 16
            p.safety_value_minibatch_size = 16; p.safety_barrier_warmup_batches = 1
            p.safety_control_mode = 'off' if mode == 'off' else 'barrier_fixed'
            p.safety_barrier_strength = 0. if mode == 'zero' else .1
            p.is_prb = prioritized_replay
            p.is_load_model = p.is_continue_train = False
            p.where_to_save = str(tmp_path / mode) + '/'
            env, *_ = training.mappo_cavs(p)
            assert not env.scenario.safety_manager.constraint_enabled
            env.close()
        assert calls
        off = torch.load(tmp_path / 'off/final_policy.pth')
        zero = torch.load(tmp_path / 'zero/final_policy.pth')
        fixed = torch.load(tmp_path / 'fixed/final_policy.pth')
        assert all(torch.equal(off[k], zero[k]) for k in off)
        assert any(not torch.equal(off[k], fixed[k]) for k in off)
        d = json.loads(next((tmp_path / 'fixed').glob('reward*_data.json')).read_text())
        m = d['safety_value_metrics_list']
        assert m[0]['barrier_ready'] == m[0]['actor_updates'] == 0
        assert m[1]['barrier_active'] == 1 and m[1]['actor_updates'] > 0
        assert all(x['actor_constraint_active'] == 0 for x in d['safety_metrics_list'])
        for final in [False, True]:
            p.is_load_model = True; p.is_load_final_model = final
            env, *_ = training.mappo_cavs(p)
            env.rollout(2, break_when_any_done=False)
            env.close()
        p.is_continue_train = True; p.n_iters = 1; p.total_frames = 32
        env, *_ = training.mappo_cavs(p)
        assert env.scenario.safety_value_manager.barrier_fit_batches == 3
        env.close()
    finally:
        torch.set_num_threads(threads)
