"""Inspect isolated map adapters without constructing an env or running a policy."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from baselines.original_sigmarl import runtime as original
from baselines.tsc import runtime as tsc
from baselines.scenario_assets import ASSET_SCENARIOS, install_scene_assets, reset_from_entry_slots, scene_asset_record
from utilities.constants import AGENTS, SCENARIOS
from utilities.parse_osm import ParseOSM


@pytest.mark.parametrize("backend", ["original", "tsc"])
@pytest.mark.parametrize("name", ASSET_SCENARIOS)
def test_child_parser_matches_current_coordinates_spawns_and_physical_curbs(tmp_path, backend, name):
    target = tmp_path/"inspection.json"
    if backend == "original":
        original.launch_worker(["inspect", "--scenario", name, "--output", str(target)])
    else:
        tsc.launch_worker(["--inspect", "--scenario", name, "--output", str(target)])
    record = json.loads(target.read_text())
    parser = ParseOSM(name, "cpu")
    assert record["scene_assets"] == scene_asset_record(name)
    assert record["lanelet_count"] == len(parser.lanelets_all)
    assert record["nominal_body_road_contacts"] == 0
    assert len(record["routes"]) == len(parser.reference_paths)
    for child, current in zip(record["routes"], parser.reference_paths):
        np.testing.assert_allclose(child["center_line"], current["center_line"], atol=1e-6)
        assert child["spawn_point_indices"] == current["spawn_point_indices"]
    # Loading the shared maps must not import the NOD trainer into a worker.
    expected = original.UPSTREAM if backend == "original" else tsc.DEFAULT_TSC_PROJECT
    assert Path(record["trainer"]) == expected/"utilities/mappo_cavs.py"


def test_legacy_scenarios_do_not_install_any_adapter():
    class Original:
        pass
    for name in ("CPM_entire", "intersection_2", "on_ramp_1", "roundabout_1"):
        assert install_scene_assets(Original, name) is Original
        assert scene_asset_record(name) is None
    original.verify_upstream()


def fixture_scenario(name):
    class Agent:
        max_speed = AGENTS["max_speed"]

        def __init__(self):
            self.state = SimpleNamespace(pos=torch.zeros(1, 2), rot=torch.zeros(1, 1), vel=torch.zeros(1, 2))

        def set_pos(self, value, batch_index):
            self.state.pos[batch_index] = value

        def set_rot(self, value, batch_index):
            self.state.rot[batch_index] = value

        def set_vel(self, value, batch_index):
            self.state.vel[batch_index] = value

    count = SCENARIOS[name]["n_agents"]
    return SimpleNamespace(
        world=SimpleNamespace(agents=[Agent() for _ in range(count)], device="cpu"), n_agents=count,
        parameters=SimpleNamespace(scenario_type=name, is_testing_mode=True),
        constants=SimpleNamespace(reset_agent_min_distance=1.2*math.hypot(AGENTS["width"], AGENTS["length"])),
        ref_paths_agent_related=SimpleNamespace(path_id=torch.zeros(1, count, dtype=torch.long),
                                              point_id=torch.zeros(1, count, dtype=torch.long)),
    )


@pytest.mark.parametrize("name", ASSET_SCENARIOS)
def test_isolated_entry_reset_matches_current_same_seed_without_environment(name):
    from scenarios.road_traffic import ScenarioRoadTraffic
    paths = ParseOSM(name, "cpu").reference_paths
    outputs = []
    for reset in (ScenarioRoadTraffic._reset_init_state, reset_from_entry_slots):
        torch.manual_seed(101)
        scenario = fixture_scenario(name)
        agents = scenario.world.agents
        for i in range(scenario.n_agents):
            reset(scenario, 0, i, False, False, None, paths, agents)
        # Also compare a single-agent respawn against the currently occupied map.
        reset(scenario, 0, torch.tensor(3), True, False, None, paths, agents)
        outputs.append((scenario.ref_paths_agent_related.path_id.clone(),
                        scenario.ref_paths_agent_related.point_id.clone(),
                        torch.cat([torch.cat((a.state.pos, a.state.rot, a.state.vel), -1) for a in agents])))
    for current, child in zip(*outputs):
        torch.testing.assert_close(current, child, rtol=0, atol=0)


def test_overcrowded_entry_fails_with_a_bound_instead_of_hanging():
    scenario = fixture_scenario("bottleneck_1")
    agents = scenario.world.agents
    agents[0].state.pos[:] = torch.tensor([1., 1.])
    path = dict(center_line=torch.ones(10, 2), spawn_point_indices=[4], center_line_yaw=torch.zeros(9))
    with pytest.raises(RuntimeError, match="after 2000 attempts"):
        reset_from_entry_slots(scenario, 0, 1, False, False, None, [path], agents)


def test_physical_contacts_replace_lane_contacts_before_reward(monkeypatch):
    from utilities import constants, parse_osm, map_manager
    # The installer intentionally changes module globals in an isolated worker;
    # restore them after exercising the class with an in-memory stub here.
    monkeypatch.setitem(constants.SCENARIOS, "bottleneck_1", constants.SCENARIOS["bottleneck_1"])
    monkeypatch.setattr(parse_osm, "ParseOSM", parse_osm.ParseOSM)
    monkeypatch.setattr(map_manager, "ParseOSM", map_manager.ParseOSM)

    class Original:
        def _init_params(self, batch_dim, device, **kwargs):
            self.normalizers = SimpleNamespace()

        def _update_state_before_rewarding(self, agent, agent_index):
            if agent_index == 0:
                self.collisions.with_lanelets[:] = True

        def reward(self, agent):
            self._update_state_before_rewarding(agent, 0)
            return -self.collisions.with_lanelets.float()

    scenario = install_scene_assets(Original, "bottleneck_1")()
    scenario._init_params(1, "cpu")
    scene = constants.SCENARIOS["bottleneck_1"]
    cy = scene["world_y_dim"]/2

    def body(y):
        return [[.9, y-.04], [1.06, y-.04], [1.06, y+.04], [.9, y+.04], [.9, y-.04]]

    # One car straddles the white lane divider, the other sits on the median.
    scenario.vertices = torch.tensor([[body(cy-.04-.18), body(cy)]])
    scenario.collisions = SimpleNamespace(with_lanelets=torch.zeros(1, 2, dtype=torch.bool))
    assert scenario.reward(None).tolist() == [[0., -1.]]
    assert scenario.world_x_dim == scene["world_x_dim"]
    assert scenario.normalizers.pos_world.tolist() == pytest.approx([scene["world_x_dim"], scene["world_y_dim"]])
