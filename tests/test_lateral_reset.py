"""Reset augmentation geometry, train/eval separation and respawn consistency."""
import json
from pathlib import Path

import pytest
import torch
from torchrl.envs.libs.vmas import VmasEnv

from scenarios.road_traffic import ScenarioRoadTraffic
from utilities.helper_training import Parameters, TransformedEnvCustom
from utilities.reset_augmentation import safe_lateral_position


def candidate(offset, **overrides):
    args = dict(position=torch.zeros(2), yaw=torch.tensor(0.), offset=offset,
                width=.08, length=.16,
                left=torch.tensor([[-1., .1], [1., .1]]),
                right=torch.tensor([[-1., -.1], [1., -.1]]),
                is_loop=False, other_positions=torch.empty(0, 2),
                min_distance=.22, clearance=.002)
    args.update(overrides)
    return safe_lateral_position(**args)


def test_swept_body_rejects_crossing_touching_and_neighbor_overlap():
    torch.testing.assert_close(candidate(.02), torch.tensor([0., .02]))
    torch.testing.assert_close(candidate(-.02), torch.tensor([0., -.02]))
    assert candidate(.061) is None  # Body crosses boundary, center stays inside.
    assert candidate(.06, clearance=0.) is None  # Parallel tangency.
    assert candidate(.3) is None  # Cannot jump completely over a boundary.
    assert candidate(.02, position=torch.tensor([.95, 0.])) is None  # End cap.
    assert candidate(.02, other_positions=torch.tensor([[0., .23]])) is None
    assert candidate(-.02, other_positions=torch.tensor([[0., .23]])) is not None
    # A short boundary segment entirely inside the box must also be rejected.
    assert candidate(.02, left=torch.tensor([[-.01, .01], [.01, .01]])) is None


def make_env(probability, testing=False):
    s = ScenarioRoadTraffic()
    s.parameters = Parameters(
        scenario_type='CPM_mixed', n_agents=4, max_steps=16,
        is_testing_mode=testing, is_apply_mask=False, is_add_noise=False,
        is_using_nod_actor=False, is_using_nod_opinion=False,
        is_using_deadlock_critic=False, is_challenging_initial_state_buffer=False,
        refresh_respawn_observations=True, fix_respawn_training=True,
        training_lateral_reset_probability=probability)
    env = TransformedEnvCustom(VmasEnv(
        scenario=s, num_envs=4, continuous_actions=True,
        max_steps=16, device='cpu', n_agents=4))
    return env, s


def test_evaluation_ignores_augmentation_and_preserves_rng():
    env, s = make_env(0., testing=True)
    try:
        env.set_seed(23)
        baseline = env.reset()
        baseline_rng = torch.get_rng_state().clone()
        s.parameters.training_lateral_reset_probability = 1.
        env.set_seed(23)
        actual = env.reset()
        assert torch.equal(baseline['agents', 'observation'], actual['agents', 'observation'])
        assert torch.equal(baseline_rng, torch.get_rng_state())
        assert not hasattr(s, 'lateral_reset_stats')
    finally:
        env.close()


def test_training_shifts_only_position_and_updates_reset_buffers(monkeypatch):
    env, s = make_env(.2)
    records = []
    original = s._apply_training_lateral_reset

    def record(e, i, single, ref, agents):
        agent = agents[i]
        before = agent.state.pos[e].clone()
        rot, vel = agent.state.rot[e].clone(), agent.state.vel[e].clone()
        original(e, i, single, ref, agents)
        delta = agent.state.pos[e] - before
        tangent = torch.cat((rot.cos(), rot.sin()))
        assert abs(float(delta @ tangent)) < 1e-6
        assert delta.norm() <= .020001
        assert torch.equal(rot, agent.state.rot[e])
        assert torch.equal(vel, agent.state.vel[e])
        if delta.norm() > 1e-6:
            # Validate the final rectangle independently with the environment's
            # existing collision geometry after a full reset below.
            records.append((e, int(i), float(delta.norm())))

    monkeypatch.setattr(s, '_apply_training_lateral_reset', record)
    try:
        env.set_seed(23)
        s.lateral_reset_stats = dict(attempted=0, applied=0, rejected=0)
        for _ in range(20):
            env.reset()
            expected = torch.stack([a.state.pos for a in s.world.agents], 1)
            torch.testing.assert_close(s.state_buffer.get_latest()[..., :2], expected)
            s._update_state_before_rewarding(s.world.agents[0], 0)
            assert not s.collisions.with_lanelets.any()
            assert not s.collisions.with_agents.any()
        stats = s.lateral_reset_stats
        assert 32 < stats['attempted'] < 96  # 20% of 320 resets, fixed seed.
        assert stats['applied'] == len(records) > 0
        assert stats['attempted'] == stats['applied'] + stats['rejected']
        # Exercise the actual training exit-respawn path and refreshed decision.
        s.parameters.training_lateral_reset_probability = 1.
        td = env.reset()
        old_reward = s.reward
        forced = False

        def reward(agent):
            nonlocal forced
            value = old_reward(agent)
            if agent is s.world.agents[-1] and not forced:
                s.collisions.with_exit_segments[0, 0] = True
                forced = True
            return value

        monkeypatch.setattr(s, 'reward', reward)
        td.set(('agents', 'action'), torch.zeros(4, 4, 2))
        attempted = s.lateral_reset_stats['attempted']
        transition, decision = env.step_and_maybe_reset(td)
        assert s.lateral_reset_stats['attempted'] > attempted
        assert transition['next', 'agents', 'info', 'task_respawn'][0, 0]
        pos = s.world.agents[0].state.pos[0]
        torch.testing.assert_close(s.state_buffer.get_latest()[0, 0, :2], pos)
        torch.testing.assert_close(decision['agents', 'info', 'nod_world_pos'][0, 0], pos)
    finally:
        env.close()


def test_configs_have_only_one_experimental_difference_and_roundtrip():
    root = Path(__file__).resolve().parents[1]
    control = json.loads((root / 'config_dgppo_lateral_reset_control.json').read_text())
    treatment = json.loads((root / 'config_dgppo_lateral_reset.json').read_text())
    assert {k for k in control if control[k] != treatment[k]} == {
        'where_to_save', 'training_lateral_reset_probability'}
    p = Parameters.from_dict(treatment)
    assert Parameters.from_dict(p.to_dict()).training_lateral_reset_probability == .2
    assert Parameters.from_dict({}).training_lateral_reset_probability == 0.
    with pytest.raises(ValueError):
        Parameters(training_lateral_reset_probability=1.1)
    with pytest.raises(ValueError):
        Parameters(training_lateral_reset_max_m=float('nan'))
