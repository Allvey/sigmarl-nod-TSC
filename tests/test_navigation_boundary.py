import json
from pathlib import Path

import pytest
import torch
from torchrl.envs.libs.vmas import VmasEnv

from scenarios.road_traffic import ScenarioRoadTraffic
from utilities.helper_scenario import get_rectangle_vertices
from utilities.helper_training import Parameters, TransformedEnvCustom
from utilities.navigation_boundary import corridor_geometry
from utilities.nod_marl.safety_value import SafetyValueManager


def test_event_rate_reduces_agents_before_time_with_optional_scalar_axis():
    from utilities.evaluate_navigation_boundary import environment_event_rates
    flags = torch.zeros(2, 4, 3, dtype=torch.bool)
    flags[0, 0, :2] = True  # Two cars, but only one collision frame.
    flags[0, 2, 2] = True
    flags[1, 3, 0] = True
    expected = torch.tensor([50., 25.])
    torch.testing.assert_close(environment_event_rates(flags), expected)
    torch.testing.assert_close(environment_event_rates(flags.unsqueeze(-1)), expected)


def test_footprint_crossing_fully_outside_and_rotation():
    left = torch.tensor([[-1., .1], [1., .1]])
    right = torch.tensor([[-1., -.1], [1., -.1]])
    centers = torch.tensor([[0., 0.], [0., .08], [0., .3], [0., .05]])
    yaw = torch.tensor([[0.], [0.], [0.], [torch.pi/4]])
    vertices = get_rectangle_vertices(centers, yaw, .08, .16)
    distances, violation = corridor_geometry(vertices, left, right)
    torch.testing.assert_close(distances[0], torch.tensor([.06, .06]))
    assert violation.tolist() == [False, True, True, True]
    assert distances[2].min() > .01  # Positive distance cannot imply containment.
    rotation = torch.tensor([[0., -1.], [1., 0.]])
    d, v = corridor_geometry(vertices @ rotation, left @ rotation, right @ rotation)
    torch.testing.assert_close(d, distances)
    assert torch.equal(v, violation)


def test_closed_loop_has_no_false_wall_at_seam():
    angle = torch.linspace(0, 2*torch.pi, 257)
    unit = torch.stack((angle.cos(), angle.sin()), -1)
    vertices = get_rectangle_vertices(torch.tensor([[1., 0.], [0., 0.], [1.5, 0.]]),
                                      torch.full((3, 1), torch.pi/2), .08, .16)
    _, v = corridor_geometry(vertices, .9*unit, 1.1*unit, is_loop=True)
    assert v.tolist() == [False, True, True]


def test_crossing_route_exit_is_not_lateral_violation():
    vertices = get_rectangle_vertices(torch.tensor([[1., 0.], [-1., 0.]]),
                                      torch.zeros(2, 1), .08, .16)
    left = torch.tensor([[-1., .1], [1., .1]])
    right = torch.tensor([[-1., -.1], [1., -.1]])
    d, v = corridor_geometry(vertices, left, right)
    assert not v.any()
    torch.testing.assert_close(d, torch.full((2, 2), .06))


def test_cache_reuses_static_data_but_not_vehicle_state():
    from utilities.navigation_boundary import NavigationBoundaryCache
    route = dict(left_boundary=torch.tensor([[-1., .1], [1., .1]]),
                 right_boundary=torch.tensor([[-1., -.1], [1., -.1]]))
    cache = NavigationBoundaryCache(1, 4, .32, 'cpu')
    for i in range(4):
        cache[0, i] = route
    for positions in [torch.tensor([[0., 0.], [0., .06], [0., .3], [1., 0.]]),
                      torch.tensor([[0., .3], [0., 0.], [-1., 0.], [0., -.06]])]:
        vertices = get_rectangle_vertices(positions, torch.zeros(4, 1), .08, .16)
        d, v = cache.evaluate(vertices.reshape(1, 4, 5, 2))
        expected_d, expected_v = corridor_geometry(vertices, route['left_boundary'], route['right_boundary'])
        torch.testing.assert_close(d[0], expected_d, atol=1e-6, rtol=1e-5)
        assert torch.equal(v[0], expected_v)
    assert len(cache.prepared) == len(cache.groups) == 1


@pytest.mark.parametrize('scenario', ['CPM_mixed', 'intersection_2', 'on_ramp_1', 'roundabout_1'])
def test_cached_geometry_matches_reference_for_real_routes_and_off_lane_poses(scenario):
    from utilities.navigation_boundary import NavigationBoundaryCache
    env, s = make_env(True, scenario=scenario)
    try:
        routes = {}
        for values in vars(s.ref_paths_map_related).values():
            if isinstance(values, list):
                for route in values:
                    if isinstance(route, dict) and 'center_line' in route:
                        routes[id(route)] = route
        generator = torch.Generator().manual_seed(711)
        for route in routes.values():
            count = min(len(route['center_line']), len(route['center_line_yaw']))
            ids = torch.randint(count, (24,), generator=generator)
            pos = route['center_line'][ids] + .15 * torch.randn(24, 2, generator=generator)
            yaw = route['center_line_yaw'][ids].reshape(24, 1) + torch.randn(24, 1, generator=generator)
            vertices = get_rectangle_vertices(pos, yaw, .08, .16)
            cache = NavigationBoundaryCache(6, 4, .32, 'cpu')
            for e in range(6):
                for i in range(4):
                    cache[e, i] = route
            actual = cache.evaluate(vertices.reshape(6, 4, 5, 2))
            expected = corridor_geometry(vertices, route['left_boundary'], route['right_boundary'], route['is_loop'])
            torch.testing.assert_close(actual[0].reshape(24, 2), expected[0], atol=1e-6, rtol=1e-5)
            assert torch.equal(actual[1].reshape(24), expected[1])
            groups = cache.groups
            state = torch.get_rng_state().clone()
            cache.evaluate(vertices.reshape(6, 4, 5, 2))
            assert cache.groups is groups and len(groups) == 1
            assert torch.equal(state, torch.get_rng_state())
        # Resetting selected slots must invalidate grouping and refresh routes.
        env.reset()
        s.reset_world_at(env_index=0, agent_index=torch.tensor(0))
        s._update_navigation_geometry()
        for e in range(2):
            for i, a in enumerate(s.world.agents):
                route = s.navigation_routes[e, i]
                vertices = get_rectangle_vertices(a.state.pos[e:e+1], a.state.rot[e:e+1], .08, .16)
                d, v = corridor_geometry(vertices, route['left_boundary'], route['right_boundary'], route['is_loop'])
                torch.testing.assert_close(s.navigation_distances[e, i], d[0], atol=1e-6, rtol=1e-5)
                assert s.navigation_violation[e, i] == v[0]
    finally:
        env.close()


def make_env(enabled=False, record=True, scenario='CPM_mixed'):
    s = ScenarioRoadTraffic()
    s.parameters = Parameters(scenario_type=scenario, n_agents=4, max_steps=16,
        is_testing_mode=True, is_apply_mask=False, is_add_noise=False,
        is_using_nod_actor=False, is_using_nod_opinion=False, is_using_deadlock_critic=False,
        is_using_safety_value_shadow=True, is_challenging_initial_state_buffer=False,
        use_navigation_boundary=enabled, record_navigation_metrics=record)
    env = TransformedEnvCustom(VmasEnv(scenario=s, num_envs=2, continuous_actions=True,
                                      max_steps=16, device='cpu', n_agents=4))
    env.set_seed(23)
    return env, s


def test_virtual_violation_changes_safety_only_and_keeps_observation_width():
    env, s = make_env()
    try:
        td = env.reset()
        obs_before = td['agents', 'observation'].clone()
        physical = s.collisions.with_lanelets.clone()
        # Move only the virtual corridor, keeping the physical map/state intact.
        a = s.world.agents[0]
        for e in range(2):
            yaw = a.state.rot[e, 0]
            t = torch.stack((yaw.cos(), yaw.sin()))
            n = torch.stack((-yaw.sin(), yaw.cos()))
            center = a.state.pos[e] - .3*n
            s.navigation_routes[e, 0] = dict(
                left_boundary=torch.stack((center-t+.1*n, center+t+.1*n)),
                right_boundary=torch.stack((center-t-.1*n, center+t-.1*n)))
        before = s.info(a)['safety_margins'].clone()
        s.parameters.use_navigation_boundary = True
        obs = torch.stack([s.observation(a) for a in s.world.agents], 1)
        info = s.info(a)
        assert obs.shape == obs_before.shape
        assert info['navigation_violation'].all()
        assert (info['safety_margins'][:, 1:] > 0).all()
        assert (before[:, 1:] <= 0).all()
        assert torch.equal(physical, s.collisions.with_lanelets)
        assert not s.done().any()  # No new termination/respawn rule.
    finally:
        env.close()


def test_same_actions_keep_rewards_resets_and_physics_unchanged():
    envs = [make_env(enabled)[0] for enabled in (False, True)]
    try:
        outputs = []
        for env in envs:
            env.set_seed(23)
            td = env.reset()
            rows = []
            for _ in range(3):
                td.set(('agents', 'action'), torch.zeros(2, 4, 2))
                step, td = env.step_and_maybe_reset(td)
                rows.append(step['next'].clone())
            outputs.append(rows)
        for a, b in zip(*outputs):
            for key in [('agents', 'reward'), ('agents', 'info', 'nod_world_pos'),
                        ('agents', 'info', 'physical_road_collision'), ('done',)]:
                torch.testing.assert_close(a[key], b[key])
    finally:
        for env in envs:
            env.close()


def test_contract_prevents_loading_old_boundary_value(tmp_path):
    p = Parameters.from_json(str(Path(__file__).resolve().parents[1] / 'config_dgppo_minimal.json'))
    old = SafetyValueManager(p, 32, ('agents', 'observation'))
    path = tmp_path / 'value.pth'
    torch.save({'contract': old.contract}, path)
    p.use_navigation_boundary = True
    new = SafetyValueManager(p, 32, ('agents', 'observation'))
    assert not new.load_if_available(str(path))
    assert new.barrier_fit_batches == 0
    root = Path(__file__).resolve().parents[1]
    a, b = [json.loads((root / f'config_{name}.json').read_text())
            for name in ('dgppo_navigation_control', 'dgppo_navigation')]
    # The user may extend either training budget independently; compare the
    # model/environment settings rather than the selected run duration.
    assert {k for k in a if k != 'n_iters' and a[k] != b[k]} == {'where_to_save', 'use_navigation_boundary'}


@pytest.mark.parametrize('config', ['config_dgppo_navigation_control.json', 'config_dgppo_navigation.json'])
def test_pinned_initialization_starts_with_fresh_value_without_training(tmp_path, monkeypatch, config):
    import importlib
    module = importlib.import_module('utilities.mappo_cavs')
    root = Path(__file__).resolve().parents[1]
    prefix = root / 'outputs/dgppo_minimal_v2/reward6.38'
    if not Path(str(prefix) + '_policy.pth').exists():
        pytest.skip('Local v2 checkpoint required for initialization smoke test')
    p = Parameters.from_json(str(root / config))
    p.training_init_checkpoint = str(prefix)
    p.where_to_save = str(tmp_path) + '/'
    p.num_vmas_envs = 2
    p.max_steps = 16
    checked = []

    class StopBeforeTraining(Exception):
        pass

    def collector(env, *args, **kwargs):
        try:
            manager = env.base_env.scenario.safety_value_manager
            assert manager.barrier_fit_batches == manager.updates == manager.frames == 0
            assert ('boundary_semantics' in manager.contract) is p.use_navigation_boundary
            checked.append(True)
        finally:
            env.close()
        raise StopBeforeTraining

    def forbidden_load(*args, **kwargs):
        raise AssertionError('Old boundary Value must not be loaded into this experiment')

    monkeypatch.setattr(module, 'SyncDataCollectorCustom', collector)
    monkeypatch.setattr(SafetyValueManager, 'load_if_available', forbidden_load)
    with pytest.raises(StopBeforeTraining):
        module.mappo_cavs(p)
    assert checked
