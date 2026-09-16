import json
from pathlib import Path

import pytest
import torch
from torchrl.envs.libs.vmas import VmasEnv

from scenarios.road_traffic import ScenarioRoadTraffic
from utilities.helper_training import Parameters, TransformedEnvCustom
from utilities.navigation_boundary import navigation_task_penalty
from utilities.nod_marl.safety_value import SafetyValueManager


def test_penalty_uses_full_violation_and_does_not_stack_with_road_collision():
    nav = torch.tensor([False, True, True, False])
    road = torch.tensor([False, False, True, True])
    actual = navigation_task_penalty(nav, road, torch.tensor(-1.), .1)
    torch.testing.assert_close(actual, torch.tensor([0., -.1, 0., 0.]))


def make_env():
    s = ScenarioRoadTraffic()
    s.parameters = Parameters.from_json('config_dgppo_navigation_task.json')
    s.parameters.is_testing_mode = True
    s.parameters.max_steps = 16
    env = TransformedEnvCustom(VmasEnv(scenario=s, num_envs=2, continuous_actions=True,
                                      max_steps=16, device='cpu', n_agents=4))
    env.set_seed(23)
    return env, s


def move_virtual_corridor(s, outside):
    a = s.world.agents[0]
    for e in range(2):
        yaw = a.state.rot[e, 0]
        tangent = torch.stack((yaw.cos(), yaw.sin()))
        normal = torch.stack((-yaw.sin(), yaw.cos()))
        center = a.state.pos[e] - (.3 if outside else 0.) * normal
        s.navigation_routes[e, 0] = dict(
            left_boundary=torch.stack((center - tangent + .1*normal, center + tangent + .1*normal)),
            right_boundary=torch.stack((center - tangent - .1*normal, center + tangent - .1*normal)))


def test_penalty_follows_current_frame_persists_and_stops_after_reentry():
    env, s = make_env()
    try:
        td = env.reset()
        # The previous observation is inside. The next reward must see the
        # newly assigned, shifted corridor before any observation refresh.
        move_virtual_corridor(s, False)
        s._update_navigation_geometry()
        assert not s.navigation_violation[:, 0].any()
        for outside in (True, True, False):
            move_virtual_corridor(s, outside)
            td.set(('agents', 'action'), torch.zeros(2, 4, 2))
            transition, td = env.step_and_maybe_reset(td)
            info = transition['next', 'agents', 'info']
            assert (info['navigation_violation'][:, 0].bool() == outside).all()
            expected = torch.full_like(info['navigation_task_penalty'][:, 0], -.1 if outside else 0.)
            torch.testing.assert_close(info['navigation_task_penalty'][:, 0], expected)
            assert not info['physical_road_collision'].any()
            assert (info['safety_margins'][:, 0, 1:] <= 0).all()
            assert not transition['next', 'done'].any()
    finally:
        env.close()


def test_same_actions_only_change_reward_not_observation_safety_or_physics():
    env, s = make_env()
    try:
        results = []
        for ratio in (0., .1):
            s.parameters.navigation_penalty_ratio = ratio
            env.set_seed(23)
            td = env.reset()
            move_virtual_corridor(s, True)
            td.set(('agents', 'action'), torch.zeros(2, 4, 2))
            step, _ = env.step_and_maybe_reset(td)
            results.append(step['next'].clone())
        before, after = results
        torch.testing.assert_close(after['agents', 'reward'][:, 0], before['agents', 'reward'][:, 0] - .1)
        torch.testing.assert_close(after['agents', 'reward'][:, 1:], before['agents', 'reward'][:, 1:])
        for key in [('agents', 'observation'), ('agents', 'info', 'safety_margins'),
                    ('agents', 'info', 'nod_world_pos'), ('done',)]:
            torch.testing.assert_close(before[key], after[key])
        # Safety-only and task-only modes see identical actor boundary inputs.
        s.parameters.navigation_boundary_mode = 'safety'
        obs_safety = torch.stack([s.observation(a) for a in s.world.agents], 1)
        s.parameters.navigation_boundary_mode = 'task'
        obs_task = torch.stack([s.observation(a) for a in s.world.agents], 1)
        torch.testing.assert_close(obs_safety, obs_task)
    finally:
        env.close()


def test_modes_have_distinct_value_contracts_and_old_configs_keep_safety(tmp_path):
    p = Parameters.from_json('config_dgppo_navigation.json')
    assert p.navigation_boundary_mode == 'safety'
    old = SafetyValueManager(p, 32, ('agents', 'observation'))
    p.navigation_boundary_mode = 'task'
    new = SafetyValueManager(p, 32, ('agents', 'observation'))
    assert old.contract != new.contract
    checkpoint = tmp_path / 'old_value.pth'
    torch.save({'contract': old.contract}, checkpoint)
    assert not new.load_if_available(str(checkpoint))
    for suffix in ('', '_scratch'):
        base = json.loads(Path(f'config_dgppo_navigation{suffix}.json').read_text())
        task = json.loads(Path(f'config_dgppo_navigation_task{suffix}.json').read_text())
        changes = {k for k in base.keys() | task.keys() if base.get(k) != task.get(k)}
        expected = {'where_to_save', 'navigation_boundary_mode', 'navigation_penalty_ratio'}
        if 'observe_navigation_boundary' in task:
            expected.add('observe_navigation_boundary')
        assert changes == expected
        assert Parameters.from_dict(task).navigation_penalty_ratio == .1
    with pytest.raises(ValueError):
        Parameters(navigation_boundary_mode='both')
    with pytest.raises(ValueError):
        Parameters(navigation_penalty_ratio=float('nan'))


def test_shared_observation_is_v2_equivalent_while_navigation_penalty_remains():
    env, s = make_env()
    try:
        s.parameters.record_navigation_metrics = True
        s.parameters.observe_navigation_boundary = False
        records = []
        for enabled in (False, True):
            s.parameters.use_navigation_boundary = enabled
            env.set_seed(23)
            td = env.reset()
            move_virtual_corridor(s, True)
            td.set(('agents', 'action'), torch.zeros(2, 4, 2))
            step, _ = env.step_and_maybe_reset(td)
            records.append(step['next'].clone())
        baseline, task = records
        for key in [('agents', 'observation'), ('agents', 'info', 'safety_margins'),
                    ('agents', 'info', 'nod_world_pos'), ('done',)]:
            torch.testing.assert_close(baseline[key], task[key], atol=0, rtol=0)
        assert task['agents', 'info', 'navigation_violation'][:, 0].all()
        torch.testing.assert_close(task['agents', 'reward'][:, 0], baseline['agents', 'reward'][:, 0] - .1)
        torch.testing.assert_close(task['agents', 'info', 'navigation_task_penalty'][:, 0],
                                   torch.full((2, 1), -.1))
    finally:
        env.close()


def test_observation_switch_defaults_serialization_and_value_identity():
    base = json.loads(Path('config_dgppo_navigation.json').read_text())
    base.pop('observe_navigation_boundary', None)
    assert Parameters.from_dict(base).observe_navigation_boundary is True
    base['use_navigation_boundary'] = False
    assert Parameters.from_dict(base).observe_navigation_boundary is False
    p = Parameters.from_json('config_dgppo_navigation_task_scratch.json')
    assert p.use_navigation_boundary and not p.observe_navigation_boundary
    assert Parameters.from_dict(p.to_dict()).observe_navigation_boundary is False
    task = SafetyValueManager(p, 32, ('agents', 'observation'))
    p.use_navigation_boundary = False
    v2 = SafetyValueManager(p, 32, ('agents', 'observation'))
    assert task.contract == v2.contract  # Same input semantics and safety labels.
    p.use_navigation_boundary = p.observe_navigation_boundary = True
    old_navigation_observation = SafetyValueManager(p, 32, ('agents', 'observation'))
    assert task.contract != old_navigation_observation.contract
    with pytest.raises(ValueError):
        Parameters(observe_navigation_boundary='false')
