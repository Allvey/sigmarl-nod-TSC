"""State-Value semantics, identity censoring and shadow-training isolation."""
import copy
import json
import random

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from utilities.helper_training import Parameters
from utilities.nod_marl.safety_value import (
    PairSafetyValue, SafetyValueManager, discounted_max_targets, isolated_rng, value_state,
    balanced_value_loss, positive_class_weights, value_head_metrics,
    SafetyStartBuffer, DeterministicSafetySampler,
)


def target_states(g, gn):
    current = dict(g=torch.tensor(g).view(1, -1, 1, 1))
    following = dict(g=torch.tensor(gn).view(1, -1, 1, 1))
    for state in (current, following):
        state.update(valid=torch.ones_like(state["g"], dtype=torch.bool),
                     ego_gen=torch.zeros_like(state["g"], dtype=torch.long),
                     other_gen=torch.zeros_like(state["g"], dtype=torch.long))
    return current, following


def test_discounted_max_terminal_bootstrap_and_negative_values():
    a, b = target_states([-.8, -.4], [-.4, .6])
    next_value = torch.ones_like(a["g"], requires_grad=True)
    y, valid, observed, count = discounted_max_targets(a, b, next_value, torch.tensor([[False, True]]), .5)
    torch.testing.assert_close(y.flatten(), torch.tensor([-.35, .1]))
    torch.testing.assert_close(observed.flatten(), torch.tensor([.6, .6]))
    assert valid.all() and not y.requires_grad
    assert count.flatten().tolist() == [3, 2]
    cut, *_ = discounted_max_targets(a, b, next_value, torch.zeros(1, 2).bool(), .5)
    torch.testing.assert_close(cut.flatten(), torch.tensor([-.25, .3]))
    a, b = target_states([-1., -1.], [-1., -1.])
    y, *_ = discounted_max_targets(a, b, torch.zeros_like(a["g"]), torch.zeros(1, 2).bool(), .5)
    torch.testing.assert_close(y.flatten(), torch.tensor([-.75, -.5]))


@pytest.mark.parametrize("broken", ["ego_gen", "other_gen", "visibility"])
def test_reset_or_disappearance_censors_suffix_without_safe_zero(broken):
    a, b = target_states([-.8, -.4], [-.4, 1.])
    if broken == "visibility":
        b["valid"][:, 1] = False
    else:
        b[broken][:, 1] += 1
    y, valid, observed, _ = discounted_max_targets(
        a, b, torch.zeros_like(a["g"]), torch.zeros(1, 2).bool(), .5)
    assert valid.flatten().tolist() == [True, False]
    # First transition uses a valid successor bootstrap, not the invalid future.
    assert y.flatten()[0] == pytest.approx(-.4)
    assert observed.flatten()[0] == pytest.approx(-.4)


def physical_td(n=3):
    ids = torch.arange(n).view(1, 1, 1, n).expand(1, 2, n, n).clone()
    pos = torch.zeros(1, 2, n, 2)
    pos[..., 0] = torch.arange(n) * .2
    return TensorDict({"agents": TensorDict({
        "observation": torch.zeros(1, 2, n, 4),
        "info": TensorDict({
            "nod_pair_features": torch.arange(1 * 2 * n * n * 20).float().reshape(1, 2, n, n, 20) / 100,
            "nod_neighbor_indices": ids,
            "nod_edge_mask": (~torch.eye(n, dtype=torch.bool)).expand(1, 2, n, n).clone(),
            "nod_world_pos": pos,
            "nod_ego_generation": torch.zeros(1, 2, n, dtype=torch.long),
            "safety_margins": torch.full((1, 2, n, 3), -.5),
        }, [1, 2, n]),
    }, [1, 2, n])}, [1, 2])


def test_neighbor_permutation_mask_action_independence_and_variable_count():
    td = physical_td()
    original = value_state(td, ("agents", "observation"), .25)
    permuted = td.clone()
    info = permuted["agents", "info"]
    for name in ("nod_pair_features", "nod_neighbor_indices", "nod_edge_mask"):
        x = info[name]
        info[name] = x.flip(-2 if name == "nod_pair_features" else -1)
    permuted["agents", "action"] = torch.full((1, 2, 3, 2), 999.)
    after = value_state(permuted, ("agents", "observation"), .25)
    for key in original:
        torch.testing.assert_close(original[key], after[key])
    assert not original["mask"].diagonal(dim1=-2, dim2=-1).any()
    assert original["g"][0, 0, 0, 1] == pytest.approx(.2)
    model = PairSafetyValue(4, 8)
    torch.testing.assert_close(model(original), model(after))
    # Hidden/non-local edge features do not alter valid pair or local outputs.
    changed = {k: v.clone() for k, v in original.items()}
    changed["pair"][~changed["mask"]] = 1000.
    torch.testing.assert_close(model(changed)[changed["valid"]], model(original)[original["valid"]])
    for n in (1, 4, 8):
        state = value_state(physical_td(n), ("agents", "observation"), .25)
        assert model(state).shape == (1, 2, n, n + 2)


def test_rng_restored_on_exception_and_stream_continues():
    torch.manual_seed(94); random.seed(94); np.random.seed(94)
    expected_t = torch.get_rng_state().clone()
    expected_p, expected_n = random.getstate(), np.random.get_state()
    private = {"seed": 12}
    with pytest.raises(RuntimeError), isolated_rng(private):
        torch.rand(3); random.random(); np.random.rand()
        raise RuntimeError("sampling failed")
    assert torch.equal(torch.get_rng_state(), expected_t)
    assert random.getstate() == expected_p
    assert np.array_equal(np.random.get_state()[1], expected_n[1])
    with isolated_rng(private):
        draw = torch.rand(3)
    with isolated_rng({"seed": 12}):
        torch.rand(3)
        torch.testing.assert_close(draw, torch.rand(3))


def test_shadow_sidecar_contract_and_old_q_rejection(tmp_path):
    p = Parameters(seed=4, is_using_safety_value_shadow=True)
    manager = SafetyValueManager(p, 4, ("agents", "observation"))
    path = tmp_path / "value.pth"
    torch.save(manager.checkpoint_state(), path)
    other = SafetyValueManager(p, 4, ("agents", "observation"))
    assert other.load_if_available(path, load_optimizer=True)
    p2 = copy.deepcopy(p); p2.safety_value_gamma = .8
    assert not SafetyValueManager(p2, 4, ("agents", "observation")).load_if_available(path)
    torch.save({"model": manager.model.state_dict()}, path)
    assert not other.load_if_available(path)
    assert not other.load_if_available(tmp_path / "missing.pth")
    assert Parameters.from_dict({}).is_using_safety_value_shadow is False


def test_head_loss_is_independent_of_pair_channel_count():
    for pairs in (1, 8):
        target = torch.tensor([1.] * pairs + [2., 3.]).view(1, 1, -1)
        prediction = torch.zeros_like(target, requires_grad=True)
        valid = torch.ones_like(target, dtype=torch.bool)
        weights = positive_class_weights(target, valid, 4.)
        loss = balanced_value_loss(prediction, target, valid, weights, 2.)
        # Smooth L1: .5, 1.5, 2.5, equally averaged across three heads.
        assert loss.item() == pytest.approx(1.5)
        loss.backward()
        assert torch.isfinite(prediction.grad).all()


def test_positive_gradient_is_bounded_and_targets_are_detached():
    target = -torch.ones(32, 1, 3)
    target[0, 0, 0] = 1.
    target.requires_grad_()
    pred = torch.zeros_like(target, requires_grad=True)
    valid = torch.zeros_like(target, dtype=torch.bool)
    valid[..., 0] = True
    weights = positive_class_weights(target, valid, 4.)
    assert weights == [4., 1., 1.]
    loss = balanced_value_loss(pred, target, valid, weights, 2.)
    loss.backward()
    assert abs(pred.grad[0, 0, 0] / pred.grad[1, 0, 0]) == pytest.approx(8.)
    assert not pred.grad[..., 1:].any()
    assert target.grad is None


def test_no_positive_or_valid_samples_does_not_invent_targets():
    pred = torch.zeros(2, 1, 3, requires_grad=True)
    target = -torch.ones_like(pred)
    valid = torch.ones_like(pred, dtype=torch.bool)
    weights = positive_class_weights(target, valid, 4.)
    assert weights == [1., 1., 1.]
    loss = balanced_value_loss(pred, target, valid, weights, 2.)
    assert loss.item() == pytest.approx(.5)
    loss.backward()
    assert (pred.grad > 0).all()  # Gradient descent still learns safe negatives.
    pred.grad = None
    loss = balanced_value_loss(pred, target, ~valid, weights, 2.)
    loss.backward()
    assert loss.item() == 0 and not pred.grad.any()


def test_metrics_distinguish_targets_physics_and_early_warning():
    def pair(values):
        x = torch.zeros(len(values), 1, 3)
        x[:, 0, 0] = torch.tensor(values)
        return x
    pred = pair([.2, -.2, .3, -.4, 999.])
    target = pair([.5, -.1, -.3, -.5, 999.])
    observed = pair([.5, .4, -.2, .7, 999.])
    current = pair([-.1, -.2, -.3, .7, 999.])
    valid = torch.zeros_like(pred, dtype=torch.bool)
    valid[:4, :, 0] = True
    m = value_head_metrics(pred, target, observed, current, valid)
    assert m['pair_samples'] == 4
    assert m['pair_target_positive_count'] == 1
    assert m['pair_target_nonpositive_count'] == 3
    assert m['pair_target_positive_mae'] == pytest.approx(.3)
    assert m['pair_observed_unsafe_target_nonpositive_count'] == 2
    assert m['pair_observed_unsafe_target_nonpositive_ratio'] == pytest.approx(2/3)
    assert m['pair_early_warning_count'] == 2  # Current collision excluded.
    assert m['pair_early_warning_recall'] == .5
    assert m['pair_early_warning_missed_count'] == 1
    assert m['pair_observed_safe_count'] == 1
    assert m['pair_observed_safe_positive_rate'] == 1
    assert m['road_samples'] == 0 and m['road_early_warning_recall'] == 0


def test_loss_contract_migration_keeps_values_and_resets_adam(tmp_path):
    p = Parameters(seed=4, is_using_safety_value_shadow=True)
    manager = SafetyValueManager(p, 4, ("agents", "observation"))
    manager.model(value_state(physical_td(), ("agents", "observation"), .25)).sum().backward()
    manager.optimizer.step()
    manager.optimizer.zero_grad(set_to_none=True)
    checkpoint = copy.deepcopy(manager.checkpoint_state())
    path = tmp_path / 'value.pth'
    torch.save(checkpoint, path)
    restored = SafetyValueManager(p, 4, ("agents", "observation"))
    assert restored.load_if_available(path, load_optimizer=True)
    assert restored.optimizer.state
    # v1 sidecars have no loss contract. Preserve model, reset legacy moments.
    del checkpoint['loss_contract']
    torch.save(checkpoint, path)
    assert restored.load_if_available(path, load_optimizer=True)
    assert not restored.optimizer.state
    assert 'optimizer reset' in restored.last_load_info
    for k, tensor in manager.model.state_dict().items():
        torch.testing.assert_close(tensor, restored.model.state_dict()[k])
    # A caller explicitly continuing the legacy loss can retain old moments.
    old = Parameters(seed=4, is_using_safety_value_shadow=True, safety_value_loss_mode='legacy')
    legacy = SafetyValueManager(old, 4, ("agents", "observation"))
    assert legacy.load_if_available(path, load_optimizer=True)
    assert legacy.optimizer.state


@pytest.mark.parametrize('kwargs', [dict(safety_value_loss_mode='invalid'),
    dict(safety_value_positive_weight_cap=.5), dict(safety_value_underestimate_weight=float('nan'))])
def test_invalid_loss_configuration(kwargs):
    with pytest.raises(ValueError, match='loss configuration'):
        Parameters(**kwargs)


def start_rollout():
    shape = (2, 5, 2)
    state = torch.arange(2 * 5 * 2 * 9).float().reshape(*shape, 9)
    info = TensorDict(dict(safety_reset_state=state,
                           nod_ego_generation=torch.zeros(*shape, 1, dtype=torch.long),
                           safety_margins=-torch.ones(*shape, 3)), shape)
    nxt = info.clone()
    nxt['safety_margins'][0, 3:, :, 2] = 1.  # Repeated collision flags: one start.
    nxt['safety_margins'][1, 3:, :, 1] = .2  # Road is separately represented.
    return TensorDict(dict(agents=TensorDict(dict(info=info), shape),
        next=TensorDict(dict(agents=TensorDict(dict(info=nxt), shape),
                             done=torch.zeros(2, 5, 1, dtype=torch.bool)), (2, 5))), (2, 5))


def test_start_mining_is_pre_danger_identity_safe_and_balanced(tmp_path):
    p = Parameters(n_agents=2, safety_value_challenging_fraction=.25, safety_value_start_lookback=3)
    starts = SafetyStartBuffer(p)
    td = start_rollout()
    m = starts.update(td)
    assert m['collision_starts_added'] == m['road_starts_added'] == 1
    for env, name in enumerate(('collision', 'road')):
        torch.testing.assert_close(starts.pools[name].buffer[0],
                                   td['agents', 'info', 'safety_reset_state'][env, 1])
    for env in range(3):
        assert starts.sample(env) is None and not starts.active[env]
    assert starts.sample(3) is not None and starts.active[3]
    # Buffer snapshots must not alias live storage, and incompatible maps are rejected.
    saved = starts.state_dict()
    path = tmp_path / 'starts.pth'
    torch.save(saved, path)
    other = SafetyStartBuffer(p)
    assert other.load_state_dict(torch.load(path))
    starts.pools['collision'].buffer.zero_()
    assert other.pools['collision'].buffer[0].abs().sum() > 0
    p.n_agents = 8
    assert not SafetyStartBuffer(p).load_state_dict(saved)


@pytest.mark.parametrize('broken', ['reset', 'done', 'unsafe', 'nonfinite'])
def test_start_mining_rejects_invalid_history(broken):
    p = Parameters(n_agents=2, safety_value_challenging_fraction=.25, safety_value_start_lookback=3)
    starts, td = SafetyStartBuffer(p), start_rollout()
    if broken == 'reset':
        td['next', 'agents', 'info', 'nod_ego_generation'][:, 3:] += 1
    elif broken == 'done':
        td['next', 'done'][:, 2:4] = True
    elif broken == 'unsafe':
        td['agents', 'info', 'safety_margins'][:] = .1
    else:
        td['agents', 'info', 'safety_reset_state'][:] = float('nan')
    m = starts.update(td)
    assert m['collision_starts_added'] == m['road_starts_added'] == 0


@pytest.mark.parametrize('kwargs', [dict(safety_value_challenging_fraction=1.),
    dict(safety_value_challenging_fraction=float('nan')), dict(safety_value_start_lookback=0),
    dict(safety_value_challenging_fraction=.25, safety_value_num_envs=1)])
def test_invalid_start_configuration(kwargs):
    with pytest.raises(ValueError, match='challenging start'):
        Parameters(**kwargs)
    assert Parameters.from_dict({}).safety_value_challenging_fraction == 0.


def test_replayed_mixed_scenario_uses_recorded_map_and_partial_resets():
    from torchrl.envs.libs.vmas import VmasEnv
    from scenarios.road_traffic import ScenarioRoadTraffic
    p = Parameters(scenario_type='CPM_mixed', n_agents=2, safety_value_num_envs=2, safety_value_challenging_fraction=.5,
                   cpm_scenario_probabilities=[0., 1., 0.],
                   is_challenging_initial_state_buffer=False, is_using_deadlock_critic=False)
    scenario = ScenarioRoadTraffic()
    scenario.parameters = p
    scenario.safety_start_buffer = starts = SafetyStartBuffer(p)
    env = VmasEnv(scenario=scenario, num_envs=2, continuous_actions=True,
                  max_steps=16, device='cpu', n_agents=p.n_agents)
    try:
        td = env.reset()
        recorded = td['agents', 'info', 'safety_reset_state'][0].clone()
        reference = scenario.ref_paths_agent_related.long_term[0].clone()
        assert (recorded[:, 5] == 2).all()
        starts.pools['road'].add(recorded)
        # Normal starts now choose intersection; replay must still select merge-in paths.
        p.cpm_scenario_probabilities = [1., 0., 0.]
        generation = scenario.nod_agent_generation.clone()
        td = env.reset()
        torch.testing.assert_close(td['agents', 'info', 'safety_reset_state'][1], recorded)
        torch.testing.assert_close(scenario.ref_paths_agent_related.long_term[1], reference)
        assert (scenario.ref_paths_agent_related.scenario_id[0] == 1).all()
        assert (scenario.nod_agent_generation > generation).all()
        scenario.reset_world_at(env_index=0)
        assert not starts.active[0] and starts.active[1]
        torch.testing.assert_close(scenario.ref_paths_agent_related.long_term[1], reference)
    finally:
        env.close()


def test_training_shadow_is_bitwise_isolated_and_loadable(tmp_path, monkeypatch):
    from utilities.mappo_cavs import mappo_cavs
    monkeypatch.setenv("WANDB_MODE", "disabled")
    original_collect = DeterministicSafetySampler.collect
    replay_checks = []

    def collect_with_forced_start(sampler):
        td, seconds = original_collect(sampler)
        expected = getattr(sampler, '_test_start', None)
        if expected is not None:
            info = td['agents', 'info']
            env = sampler.starts.normal_envs
            torch.testing.assert_close(info['safety_reset_state'][env, 0], expected)
            assert info['safety_challenging_start'][env, 0].all()
            assert not info['safety_challenging_start'][:env].any()
            assert not info['act_vel'][:, 0].any() and not info['act_steer'][:, 0].any()
            for name, weight in sampler.source_nod.model.state_dict().items():
                torch.testing.assert_close(sampler.nod.model.state_dict()[name], weight)
            replay_checks.append(True)
        # A known physical start forces the replay path even in this tiny smoke.
        sampler._test_start = td['agents', 'info', 'safety_reset_state'][0, 0].clone()
        for pool in sampler.starts.pools.values():
            pool.reset()
        sampler.starts.pools['collision'].add(sampler._test_start)
        sampler.manager.start_buffer_state = sampler.starts.state_dict()
        return td, seconds

    monkeypatch.setattr(DeterministicSafetySampler, 'collect', collect_with_forced_start)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for enabled in (False, True):
            p = Parameters.from_json("config.json")
            p.safety_control_mode = 'legacy_q'
            p.seed = 571; p.n_iters = 2; p.num_epochs = 1
            p.frames_per_batch = 16; p.minibatch_size = 8
            p.num_vmas_envs = 2; p.max_steps = 8; p.total_frames = 32
            p.nod_num_epochs = 1; p.nod_sequence_length = 8
            p.safety_num_epochs = 1; p.is_using_safety_value_shadow = enabled
            p.safety_value_num_envs = 2; p.safety_value_rollout_steps = 8
            p.safety_value_num_epochs = 1; p.safety_value_minibatch_size = 4
            p.is_load_model = False; p.is_continue_train = False
            p.where_to_save = str(tmp_path / str(enabled)) + "/"
            env, _, _, _ = mappo_cavs(p)
            env.close()
        for name in ("policy", "critic", "nod", "safety_critic"):
            a = torch.load(tmp_path / f"False/final_{name}.pth")
            b = torch.load(tmp_path / f"True/final_{name}.pth")
            if name in ("nod", "safety_critic"):
                a, b = a["model"], b["model"]
            assert all(torch.equal(a[k], b[k]) for k in a), name
        assert not list((tmp_path / "False").glob("*safety_value.pth"))
        data = json.loads(next((tmp_path / "True").glob("reward*_data.json")).read_text())
        metrics = data["safety_value_metrics_list"]
        assert len(metrics) == 2 and all(m["optimizer_updates"] > 0 for m in metrics)
        assert all(m["actor_updates"] == 0 and m["valid_samples"] > 0 for m in metrics)
        assert sum(m['pair_samples'] for m in metrics) > 0
        assert all(m['balanced_loss'] == 1 and m['pair_positive_class_weight'] <= 4 for m in metrics)
        assert replay_checks
        assert all(m['normal_env_count'] == 1 and m['challenge_env_count'] == 1 for m in metrics)
        assert metrics[1]['challenging_start_frame_ratio'] > 0
        for m in metrics:
            for head in ('pair', 'road', 'collision'):
                assert m['normal_' + head + '_samples'] + m['challenge_' + head + '_samples'] == m[head + '_samples']
                assert m[head + '_target_positive_count'] + m[head + '_target_nonpositive_count'] == m[head + '_samples']
                assert m[head + '_early_warning_count'] <= m[head + '_observed_unsafe']
        for final in (False, True):
            p.is_load_model = True; p.is_load_final_model = final
            env, _, _, _ = mappo_cavs(p)
            assert env.scenario.safety_value_manager.updates > 0
            assert env.scenario.safety_value_manager.last_load_info.startswith("loaded")
            env.close()
        old_updates = torch.load(tmp_path / "True/final_safety_value.pth")["updates"]
        p.is_continue_train = True; p.n_iters = 1; p.total_frames = 16
        env, _, _, _ = mappo_cavs(p)
        assert env.scenario.safety_value_manager.updates > old_updates
        env.close()
        # Existing policy/NOD checkpoints remain usable without the new sidecar.
        p.is_continue_train = False; p.where_to_save = str(tmp_path / "False") + "/"
        env, _, _, _ = mappo_cavs(p)
        assert env.scenario.safety_value_manager.updates == 0
        env.close()
    finally:
        torch.set_num_threads(threads)
