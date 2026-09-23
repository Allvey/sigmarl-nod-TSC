import torch

from utilities.collision_counter import CollisionCounter


def test_contact_events_and_independent_environment_resets():
    counter = CollisionCounter(2, 3, 'cpu')
    pairs = torch.zeros(2, 3, 3, dtype=torch.bool)
    road = torch.zeros(2, 3, dtype=torch.bool)
    pairs[0, 0, 1] = pairs[0, 1, 0] = True
    pairs[0, 1, 2] = pairs[0, 2, 1] = True
    pairs[0, 0, 0] = True  # Self entries never count.
    road[1, :2] = True
    counter.update(pairs, road)
    counter.update(pairs, road)  # Persistent contacts do not count twice.
    assert counter.vehicle.tolist() == [2, 0]
    assert counter.road.tolist() == [0, 2]

    # Respawning car 0 preserves totals and leaves the 1--2 contact continuous.
    counter.reset(0, 0)
    counter.update(pairs, road)
    assert counter.vehicle.tolist() == [3, 0]
    counter.update(torch.zeros_like(pairs), torch.zeros_like(road))
    counter.update(pairs, road)  # Separation followed by contact is a new event.
    assert counter.vehicle.tolist() == [5, 0]
    assert counter.road.tolist() == [0, 4]
    counter.reset(0)
    assert counter.vehicle.tolist() == [0, 0]
    assert counter.road.tolist() == [0, 4]


def test_physical_collision_is_counted_before_testing_respawn(monkeypatch):
    from torchrl.envs.libs.vmas import VmasEnv
    from utilities.helper_training import Parameters
    from scenarios.road_traffic import ScenarioRoadTraffic

    scenario = ScenarioRoadTraffic()
    scenario.parameters = Parameters(
        scenario_type='on_ramp_1', n_agents=8, is_testing_mode=True,
        is_using_deadlock_critic=False, is_apply_mask=False,
        is_use_mtv_distance=False, max_steps=16,
    )
    env = VmasEnv(scenario=scenario, num_envs=1, continuous_actions=True,
                  max_steps=16, device='cpu', n_agents=8)
    try:
        td = env.reset()
        a, b = scenario.world.agents[:2]
        a.set_pos(b.state.pos + torch.tensor([[.02, .005]]), batch_index=None)
        a.set_rot(b.state.rot + torch.pi / 2, batch_index=None)
        # Hold the intersecting rectangles in place for collision detection.
        monkeypatch.setattr(scenario.world, 'step', lambda: None)
        generations = scenario.nod_agent_generation.clone()
        td = env.rand_action(td)
        td['agents', 'action'].zero_()
        transition = env.step(td)
        assert scenario.collision_counter.vehicle[0] >= 1
        assert transition['next', 'agents', 'info',
                          'testing_vehicle_collision_events'].max() >= 1
        assert (scenario.nod_agent_generation != generations).any()
        # done() has cleared collision flags, but totals survive single-car resets.
        assert not scenario.collisions.with_agents.any()
        env.reset()
        assert not scenario.collision_counter.vehicle.any()
        assert not scenario.collision_counter.road.any()
    finally:
        env.close()
