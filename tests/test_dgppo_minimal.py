"""Numerical DGPPO parity, local inputs, terminal labels and real PPO integration."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from utilities.helper_training import Parameters
from utilities.nod_marl.dgppo import dgppo_targets, dgppo_advantage, prepare_dgppo_advantage
from utilities.nod_marl.safety_value import SafetyValueManager, value_state


def states(g, gn=None):
    g = torch.as_tensor(g, dtype=torch.float64)
    a = dict(g=g, valid=torch.ones_like(g, dtype=torch.bool),
             ego_gen=torch.zeros_like(g, dtype=torch.long), other_gen=torch.zeros_like(g, dtype=torch.long))
    b = copy.deepcopy(a)
    if gn is not None: b['g'] = torch.as_tensor(gn, dtype=g.dtype)
    return a, b


def official_numpy(hs, values, gamma, lam):
    # Copyright (c) 2025 REALM; see utilities/nod_marl/DGPPO_LICENSE.txt.
    # Direct NumPy transcription of MIT-REALM/dgppo algo/utils.py's DP rows
    # and GAE coefficients; independent of the horizon implementation under test.
    T, agents, heads = hs.shape
    rows = np.zeros((T + 1, agents, heads)); rows[0] = values[-1]
    coeff = np.zeros(T + 1); coeff[0] = 1
    outputs = []
    for ii, t in enumerate(reversed(range(T))):
        h = hs[t]
        new = np.maximum(h, (1 - gamma) * h.max(-1)[None, :, None] + gamma * rows)
        new[np.arange(T + 1) >= ii + 1] = 0
        outputs.append(np.einsum('t,tah->ah', coeff, new))
        new[ii + 1] = values[t]
        rows = new
        coeff = np.roll(coeff, 1)
        coeff[0], coeff[1] = lam ** (ii + 1), lam ** ii * (1 - lam)
    return np.stack(outputs[::-1])


@pytest.mark.parametrize('lam', [0., .4, .95, 1.])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_targets_match_official_dp(lam, dtype):
    rng = np.random.default_rng(190)
    h = rng.uniform(-1, 1, (2, 7, 3, 4))
    v = rng.normal(size=(2, 8, 3, 4))
    a, b = states(h)
    a['g'] = a['g'].to(dtype); b['g'] = b['g'].to(dtype)
    y, valid, *_ = dgppo_targets(a, b, torch.tensor(v[:, 1:], dtype=dtype), torch.zeros(2, 7), .99, lam)
    expected = np.stack([official_numpy(h[i], v[i], .99, lam) for i in range(2)])
    np.testing.assert_allclose(y.numpy(), expected, rtol=1e-6 if dtype == torch.float32 else 1e-12,
                               atol=1e-6 if dtype == torch.float32 else 1e-12)
    assert valid.all() and not y.requires_grad


def test_terminal_censor_and_invalid_heads():
    a, b = states([[[[-.8, -.5]], [[-.7, -.3]], [[-.6, -.2]]]])
    b['g'][0, 1, 0, 0] = 1.
    a['valid'][..., 1] = b['valid'][..., 1] = False
    a['g'][..., 1] = 999.  # Invalid padded heads cannot contaminate cross-head max.
    v = torch.full_like(a['g'], -1.)
    done = torch.tensor([[False, True, False]])
    y, valid, *_ = dgppo_targets(a, b, v, done, .99, 1.)
    assert y[0, 1, 0, 0] > .9 and y[0, 0, 0, 0] > .8
    assert (y[..., 1] == 0).all()
    # Reset breaks the old pair: no bootstrap into a new occupant.
    b['other_gen'][0, 1, 0, 0] = 1
    y, valid, *_ = dgppo_targets(a, b, v, done, .99, 1.)
    assert not valid[0, 1, 0, 0]
    assert y[0, 0, 0, 0] == -.8


def test_barrier_recovery_margin_unknown_and_actor_gradient():
    a, b = states([[[.4, -.5]], [[-.5, -.5]], [[-.5, -.5]]])
    v = a['g'].clone().requires_grad_()
    vn = torch.tensor([[[.4, -.5]], [[-.45, -.5]], [[-.5, -.5]]], dtype=v.dtype)
    task = torch.tensor([[[100.]], [[2.]], [[3.]]], dtype=v.dtype)
    out, info = dgppo_advantage(task, a, b, v, vn, dt=.1, alpha=1., eps=.01, weight=1.)
    torch.testing.assert_close(out.flatten(), torch.tensor([-.41, 1.99, 3.], dtype=v.dtype))
    assert info['violation'][0] and not info['violation'][2]
    # Probability of the sampled unsafe action decreases under the PPO objective.
    logits = torch.zeros(2, requires_grad=True)
    (-torch.log_softmax(logits, 0)[0] * out[0].squeeze()).backward()
    assert logits.grad[0] > 0 and v.grad is None
    b['valid'][0, 0, 1] = False
    out, info = dgppo_advantage(task, a, b, v, vn, dt=.1, alpha=1., eps=.01, weight=1.)
    assert info['eligible'][0] and out[0] < 0
    b['valid'][2, 0, 1] = False
    out, info = dgppo_advantage(task, a, b, v, vn, dt=.1, alpha=1., eps=.01, weight=0.)
    torch.testing.assert_close(out, task)
    assert not info['eligible'][2]


def test_task_scaling_is_identical_in_control_warmup_and_safe_updates():
    p = Parameters.from_json('config_dgppo_minimal.json')
    p.dgppo_schedule = False
    current, _ = states(-torch.ones(1, 3, 1, 3))
    manager = SimpleNamespace(parameters=p, enabled=True, barrier_fit_batches=0,
                              rollouts=0, state=lambda td: current, model=lambda s: s['g'])
    raw = torch.tensor([2., 4., 6.]).reshape(1, 3, 1, 1)
    expected = (raw - raw.mean(1, keepdim=True)) / (raw.std(1, unbiased=False, keepdim=True) + 1e-8)
    for weight, warmup, fits in [(0., 0, 0), (1., 5, 0), (1., 5, 5)]:
        p.dgppo_weight = weight
        p.safety_barrier_warmup_batches = warmup
        manager.barrier_fit_batches = fits
        td = TensorDict({'advantage': raw.clone(),
                         'next': TensorDict({'done': torch.zeros(1, 3, 1, dtype=torch.bool)}, [1, 3])}, [1, 3])
        prepare_dgppo_advantage(manager, td, 'advantage')
        torch.testing.assert_close(td['advantage'], expected.to(td['advantage'].dtype))
        torch.testing.assert_close(td['agents', 'barrier_task_advantage'], raw)
        assert not td['agents', 'barrier_violation_mask'].any()


def make_info():
    ids = torch.tensor([[[1], [0]]])
    pair = torch.rand(1, 2, 1, 20); pair[..., 17] = 1.; pair[..., 6] = .4
    return TensorDict({'agents': TensorDict({'observation': torch.rand(1, 2, 4), 'info': TensorDict(dict(
        nod_pair_features=pair, nod_neighbor_indices=ids, nod_edge_mask=torch.ones_like(ids, dtype=torch.bool),
        nod_world_pos=torch.rand(1, 2, 2), nod_ego_generation=torch.zeros(1, 2, dtype=torch.long),
        safety_margins=-torch.ones(1, 2, 3)), batch_size=[1, 2])}, batch_size=[1, 2])}, batch_size=[1])


def test_no_neighbor_path_or_hidden_world_input_and_checkpoint(tmp_path):
    p = Parameters.from_json('config_dgppo_minimal.json')
    manager = SafetyValueManager(p, 4, ('agents', 'observation'))
    td = make_info(); first = manager.state(td)
    td['agents', 'info', 'nod_pair_features'][..., 9:16] = 900
    td['agents', 'info', 'nod_edge_mask'].zero_()
    td['agents', 'info', 'nod_world_pos'].mul_(100)
    second = manager.state(td)
    for k in first: torch.testing.assert_close(first[k], second[k])
    assert manager.target is None
    path = tmp_path / 'value.pth'; torch.save(manager.checkpoint_state(), path)
    other = SafetyValueManager(p, 4, ('agents', 'observation'))
    assert other.load_if_available(path, load_optimizer=True)
    legacy = Parameters(safety_control_mode='barrier_fixed', is_using_safety_value_shadow=True)
    old = SafetyValueManager(legacy, 4, ('agents', 'observation'))
    assert not old.load_if_available(path)


@pytest.mark.parametrize('changes', [dict(dgppo_lambda=2), dict(dgppo_alpha=20),
    dict(dgppo_weight=-1), dict(is_using_nod_opinion=True), dict(safety_value_loss_mode='balanced'),
    dict(is_observe_ref_path_other_agents=True)])
def test_configuration_rejects_invalid_combinations(changes):
    opts = json.loads(Path('config_dgppo_minimal.json').read_text()); opts.update(changes)
    with pytest.raises(ValueError): Parameters(**opts)


def test_task_only_configuration_is_a_matched_control():
    revised = json.loads(Path('config_dgppo_minimal.json').read_text())
    control = json.loads(Path('config_dgppo_task_only.json').read_text())
    assert {k for k in revised if revised[k] != control[k]} == {'dgppo_weight', 'where_to_save'}
    p = Parameters.from_json('config_dgppo_minimal.json')
    assert p.num_epochs == 15 and p.safety_barrier_warmup_batches == 20
    assert not p.dgppo_schedule
    assert p.where_to_save == 'outputs/dgppo_minimal_v2/'
    assert Parameters.from_json('config_dgppo_task_only.json').dgppo_weight == 0


def test_physical_terminal_collision_survives_and_trains(monkeypatch):
    from torchrl.envs.libs.vmas import VmasEnv
    from scenarios.road_traffic import ScenarioRoadTraffic
    p = Parameters.from_json('config_dgppo_minimal.json')
    p.scenario_type = 'on_ramp_1'; p.n_agents = 4; p.max_steps = 16
    scenario = ScenarioRoadTraffic(); scenario.parameters = p
    env = VmasEnv(scenario=scenario, num_envs=1, continuous_actions=True,
                  max_steps=16, device='cpu', n_agents=4)
    try:
        td = env.reset()
        a, b = scenario.world.agents[:2]
        a.set_pos(b.state.pos + torch.tensor([[.02, .005]]), batch_index=None)
        a.set_rot(b.state.rot + torch.pi / 2, batch_index=None)
        monkeypatch.setattr(scenario.world, 'step', lambda: None)
        td = env.rand_action(td); td['agents', 'action'].zero_()
        transition = env.step(td).unsqueeze(1)
        assert transition['next', 'done'].all()
        manager = SafetyValueManager(p, td['agents', 'observation'].shape[-1], ('agents', 'observation'))
        current = manager.state(transition); following = manager.state(transition['next'])
        assert (following['g'][..., -1] > 0).any()
        y, valid, *_ = dgppo_targets(current, following, -torch.ones_like(following['g']),
                                      transition['next', 'done'], .99, .95)
        collision = following['g'][..., -1] > 0
        assert valid[..., -1][collision].all() and (y[..., -1][collision] > 0).all()
        metrics = manager.train_on_rollout(transition)
        assert metrics['optimizer_updates'] > 0 and metrics['collision_observed_unsafe'] > 0
    finally:
        env.close()


@pytest.mark.parametrize('warmup', [0, 1])
def test_real_short_training_save_load_and_gradient_isolation(tmp_path, monkeypatch, warmup):
    from utilities import mappo_cavs as training
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    original = training.prepare_barrier_advantage
    checks = []
    def checked(manager, td, key):
        protected = ['value_target', ('agents', 'action'), ('agents', 'sample_log_prob'), ('next', 'agents', 'reward')]
        saved = {k: td.get(k).clone() for k in protected}
        metrics = original(manager, td, key)
        for k in protected: torch.testing.assert_close(td.get(k), saved[k])
        if metrics['barrier_ready']:
            mask = td['agents', 'barrier_violation_mask']
            if mask.any():
                assert (td.get(key)[mask] < 0).all(); checks.append(True)
            assert not td.get(key).requires_grad
        return metrics
    monkeypatch.setattr(training, 'prepare_barrier_advantage', checked)
    p = Parameters.from_json('config_dgppo_minimal.json')
    p.safety_barrier_warmup_batches = warmup
    p.num_epochs = 1
    p.n_iters = 3; p.frames_per_batch = 64; p.total_frames = 192
    p.minibatch_size = 32; p.num_vmas_envs = 4; p.max_steps = 16
    p.safety_value_num_envs = 4; p.safety_value_rollout_steps = 16; p.safety_value_minibatch_size = 32
    p.where_to_save = str(tmp_path) + '/'
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        env, policy, *_ = training.mappo_cavs(p)
        assert not env.scenario.nod_manager.enabled and not env.scenario.safety_manager.enabled
        env.close()
        assert checks
        d = json.loads(next(tmp_path.glob('reward*_data.json')).read_text())
        ms = d['safety_value_metrics_list']
        assert all(x['dgppo_target'] == 1 for x in ms)
        assert ms[0]['barrier_ready'] == float(warmup == 0)
        assert ms[-1]['barrier_ready'] == 1
        assert all(x['barrier_unsafe_positive_advantage_rate'] == 0 for x in ms if x['barrier_ready'])
        assert any(x['actor_updates'] > 0 and x['actor_probe_mode_delta_abs'] > 0 for x in ms)
        p.is_load_model = p.is_load_final_model = True
        env, *_ = training.mappo_cavs(p)
        assert env.scenario.safety_value_manager.last_load_info.startswith('loaded')
        env.rollout(2, break_when_any_done=False); env.close()
    finally:
        torch.set_num_threads(threads)
