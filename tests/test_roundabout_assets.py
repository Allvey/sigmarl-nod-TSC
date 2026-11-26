"""Map topology, artwork alignment and reset integration; no simulation steps."""
import math
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import cv2

from scripts.build_roundabout_assets import (
    NAME, WIDTH, RADII, ISLAND_RADIUS, build_geometry, write_assets,
    footprint_corners, clearance_values, world, pixels,
)
from scripts.roundabout_width_design import (
    TRACE, REFERENCE, PIXEL_SCALE, RASTER_SCALE,
    design_layers, carriageways, ring_frame, frames,
)
from scripts.roundabout_vectorization import contour_groups, trapezoids
from utilities.constants import AGENTS, SCENARIOS
from utilities.map_asset_rendering import static_map_geometries, overlay_text_rows
from utilities.parse_osm import ParseOSM


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def parsed():
    return ParseOSM(NAME, "cpu")


def test_checked_in_assets_are_reproducible(tmp_path):
    write_assets(tmp_path)
    for suffix in (".osm", ".json", ".render.json", ".boundaries.json", ".svg"):
        assert (tmp_path/(NAME+suffix)).read_bytes() == (ROOT/"assets/maps"/(NAME+suffix)).read_bytes()


def test_two_ccw_lanes_and_all_origin_destination_routes(parsed):
    ways, desc, routes, info, neighbors, center, dimensions = build_geometry()
    assert len(ways) == 32 and len(routes) == len(parsed.reference_paths) == 24
    assert len([d for d in desc.values() if d["kind"] == "ring"]) == 16
    assert set(d["lane"] for d in desc.values()) == {0, 1}
    assert {(m["origin"], m["destination"], m["lane"]) for m in info} == {
        (a, b, lane) for a in range(4) for b in range(4) for lane in range(2) if a != b}
    for key, points in ways.items():
        assert key in neighbors[key]
        assert all(key in neighbors[n] for n in neighbors[key])
        if desc[key]["kind"] == "ring":
            xy = points-center
            # A true parallel offset from the same ellipse, rather than two
            # independently chosen ellipses with variable lane spacing.
            assert np.all(np.diff(np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))) > 0)
    # Preserve a small green island with a broad truck apron.
    assert .40 < ISLAND_RADIUS/(RADII[-1]+WIDTH/2) < .48
    assert RADII[0]-WIDTH/2-ISLAND_RADIUS > WIDTH
    for route, parsed_route in zip(routes, parsed.reference_paths):
        assert desc[route[0]]["kind"] == "entry" and desc[route[-1]]["kind"] == "exit"
        for a, b in zip(route, route[1:]):
            assert np.allclose(ways[a][-1], ways[b][0], atol=1e-8)
        expected = np.vstack([ways[route[0]]]+[ways[key][1:] for key in route[1:]])
        np.testing.assert_allclose(parsed_route["center_line"], expected, atol=1e-6)
        assert np.all(expected > 0) and np.all(expected < dimensions)


def test_exported_road_boundary_matches_width_constrained_design():
    mask, _ = design_layers()
    reconstructed = np.zeros_like(mask)
    # The convex render tiles must retain concavities and every large hole.
    artifact = json.loads((ROOT/"assets/maps/roundabout_realistic.render.json").read_text())
    for item in artifact["shapes"]:
        if item["role"] == "road":
            quad = np.rint(pixels(item["points"])*RASTER_SCALE).astype(np.int32)
            cv2.fillConvexPoly(reconstructed, quad, 255)
    union = np.count_nonzero((reconstructed > 0) | (mask > 0))
    intersection = np.count_nonzero((reconstructed > 0) & (mask > 0))
    assert intersection/union > .99
    # Central apron and green island must not become drivable after tessellation.
    cx, cy = np.array(TRACE["ring_center"])*RASTER_SCALE
    assert reconstructed[cy, cx] == 0
    assert reconstructed[cy-90*RASTER_SCALE, cx] == 0
    meta = json.loads((ROOT/"assets/maps/roundabout_realistic.boundaries.json").read_text())
    assert meta["reference_sha256"] == hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    assert meta["geometry_source"] == "analytic_width_constrained_spines"
    assert meta["lane_width"] == WIDTH
    assert TRACE["vehicle_size"] == [AGENTS["length"], AGENTS["width"]]
    points = [[0, 0], TRACE["image_size"], TRACE["ring_center"]]
    np.testing.assert_allclose(pixels(world(points)), points, atol=1e-8)


def test_nominal_vehicle_footprints_have_pavement_and_own_lane_clearance(parsed):
    mask, _ = design_layers()
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    for route in parsed.reference_paths:
        points = route["center_line"].numpy()
        t, n = frames(points)
        # All body edges, not only its four corners; test the parsed forward
        # yaw actually used by the simulator as well as the nominal tangent.
        yaw = route["center_line_yaw"].numpy()
        t = np.column_stack((np.cos(yaw), np.sin(yaw)))
        t = np.vstack((t, t[-1]))
        n = np.column_stack((-t[:, 1], t[:, 0]))
        body = np.stack([points+s*.08*t+r*.04*n
                         for s in np.linspace(-1, 1, 5)
                         for r in np.linspace(-1, 1, 3)])
        assert np.min(clearance_values(body, distance))*PIXEL_SCALE >= .025
        # Distance to the route's own polyline must fit inside its half width,
        # with 2 cm remaining. Exclude road-terminal longitudinal overhangs.
        q = body[:, 3:-3].reshape(-1, 2)
        v = np.diff(points, axis=0)
        delta = q[:, None]-points[:-1]
        fraction = np.clip((delta*v).sum(-1)/(v*v).sum(-1), 0., 1.)
        dist = np.linalg.norm(delta-fraction[..., None]*v, axis=-1).min(-1)
        assert dist.max() <= WIDTH/2-.02


def test_adjacent_lanes_fit_two_vehicles_and_never_fold():
    sections = list(carriageways().values())
    sections.append(ring_frame(np.linspace(0, 2*np.pi, 1001)))
    assert WIDTH >= 2*AGENTS["width"]
    for p, t, n in sections:
        inner, outer = p+WIDTH/2*n, p-WIDTH/2*n
        # Global nearest distance also detects curves doubling back onto the
        # other lane; checking only matched-point separation would miss that.
        separation = np.linalg.norm(inner[::4, None]-outer[None, ::4], axis=-1).min()
        assert separation >= WIDTH-.002
        curvature = np.linalg.norm(np.gradient(t, axis=0), axis=1)/np.linalg.norm(np.gradient(p, axis=0), axis=1)
        assert np.max(curvature)*WIDTH < .5  # Offset map has no cusp/fold.
        for j in range(20, len(p)-20, 40):
            rectangles = []
            for c in (inner[j], outer[j]):
                rectangles.append(np.array([c+s*.08*t[j]+r*.04*n[j]
                                             for s, r in ((-1,-1),(1,-1),(1,1),(-1,1))], np.float32))
            area, _ = cv2.intersectConvexConvex(*rectangles)
            assert area == 0.


def test_actual_map_white_dividers_are_passable_but_curbs_and_islands_are_not():
    from utilities.road_collision import RoadBoundaryCollision

    collision = RoadBoundaryCollision.from_asset(SCENARIOS[NAME]["geometry_asset_path"])

    def rectangle(c, t, n):
        return np.array([c+s*.08*t+r*.04*n
                         for s, r in ((-1,-1),(1,-1),(1,1),(-1,1),(-1,-1))])

    # Centres exactly on the white divider of all eight carriageways and the
    # ring: the full-size vehicle is in both lanes, without leaving pavement.
    samples = [rectangle(p[300], t[300], n[300]) for p, t, n in carriageways().values()]
    p, t, n = ring_frame(np.linspace(0, 2*np.pi, 1001))
    samples += [rectangle(p[j], t[j], n[j]) for j in (120, 370, 620, 870)]
    assert not collision(torch.tensor(np.array(samples), dtype=torch.float32)).any()

    p, t, n = carriageways()[0, "entry"]
    off_road = rectangle(p[80]-(WIDTH+.05)*n[80], t[80], n[80])
    island = rectangle(world(TRACE["ring_center"]), np.array([1., 0.]), np.array([0., 1.]))
    assert collision(torch.tensor(np.array([off_road, island]), dtype=torch.float32)).all()


def test_paths_are_finite_smooth_and_have_only_entry_spawn_points(parsed):
    limit = math.tan(math.radians(AGENTS["max_steering"]))/ (AGENTS["l_f"]+AGENTS["l_r"])
    for path in parsed.reference_paths:
        for key in ("center_line", "center_line_yaw", "left_boundary", "right_boundary"):
            assert torch.isfinite(path[key]).all()
        p = path["center_line"].numpy()
        vec = np.diff(p, axis=0)
        lengths = np.linalg.norm(vec, axis=1)
        assert lengths.min() > .02 and lengths.max() < .05
        yaw = np.unwrap(np.arctan2(vec[:, 1], vec[:, 0]))
        curvature = np.abs(np.diff(yaw))/((lengths[1:]+lengths[:-1])/2)
        assert curvature.max() < .6*limit  # Curves leave steering capacity for tracking.
        np.testing.assert_allclose(torch.norm(path["left_boundary"]-path["right_boundary"], dim=1), WIDTH, atol=2e-6)
        entry = parsed.lanelets_all[parsed._lanelet_id_to_index[path["lanelet_IDs"][0]]]["center_line"]
        assert min(path["spawn_point_indices"]) > 0
        assert max(path["spawn_point_indices"]) < len(entry)-5
        np.testing.assert_allclose(p[path["spawn_point_indices"]], entry[path["spawn_point_indices"]], atol=1e-6)


def test_artwork_is_finite_inside_viewport_and_cached(parsed):
    scene = SCENARIOS[NAME]
    for item in parsed.render_asset["shapes"]:
        p = np.asarray(item["points"])
        assert np.isfinite(p).all() and np.all(p >= 0)
        assert np.all(p <= np.array([scene["world_x_dim"], scene["world_y_dim"]])+1e-6)
        assert item["kind"] in ("polygon", "line")

    class Geom:
        def set_color(self, *args):
            pass

        def set_linewidth(self, width):
            pass

    class Rendering:
        calls = 0

        @classmethod
        def make_polygon(cls, **kwargs):
            cls.calls += 1
            return Geom()

        @classmethod
        def PolyLine(cls, **kwargs):
            cls.calls += 1
            return Geom()

    fake = SimpleNamespace(render_asset=parsed.render_asset)
    geoms = static_map_geometries(fake, Rendering)
    assert static_map_geometries(fake, Rendering) is geoms
    assert Rendering.calls == len(geoms) == len(parsed.render_asset["shapes"])
    assert static_map_geometries(SimpleNamespace(), Rendering) is None


def test_compact_window_keeps_full_map_and_readable_vehicle_scale():
    scene = SCENARIOS[NAME]
    assert scene["viewer_size"] == [800, 800]
    assert scene["compact_hud"]
    # VMAS squares viewer_zoom for the minimum shared-camera half extent.
    half_extent = scene["viewer_zoom"] ** 2
    assert 2 * half_extent >= max(scene["world_x_dim"], scene["world_y_dim"])
    vehicle_length = AGENTS["length"]
    assert vehicle_length / (2 * half_extent) > 1 / 36
    rows = overlay_text_rows([
        "A1 -> A12  z=-0.905  alpha=5.47  [non-safe: fixed]  e=-0.91  beta=0.98  A=0.7",
        "Decision t=12.35s (before displayed step)",
    ], width=588, font_size=9)
    assert len(rows) == 3
    assert all(len(row) * 9 <= 588 for row in rows)
    assert "beta=0.98" in " ".join(rows)


def test_environment_reset_preserves_checkpoint_observation_dimensions():
    """Instantiate/reset only: the user executes actual policy rollouts."""
    from torchrl.envs.libs.vmas import VmasEnv
    from utilities.helper_training import Parameters
    from scenarios.road_traffic import ScenarioRoadTraffic

    p = Parameters.from_json(ROOT/"configs/current/error_scale_repeat/02_control.json")
    torch.manual_seed(101)
    p.scenario_type, p.n_agents = NAME, SCENARIOS[NAME]["n_agents"]
    p.is_testing_mode, p.is_add_noise = True, False
    p.is_challenging_initial_state_buffer = False
    p.num_vmas_envs, p.max_steps, p.device = 1, 16, "cpu"
    scenario = ScenarioRoadTraffic()
    scenario.parameters = p
    env = VmasEnv(scenario=scenario, num_envs=1, continuous_actions=True,
                  max_steps=16, device="cpu", n_agents=p.n_agents)
    try:
        td = env.reset()
        assert scenario.world_x_dim == SCENARIOS[NAME]["world_x_dim"]
        assert scenario.world_y_dim == SCENARIOS[NAME]["world_y_dim"]
        assert scenario.viewer_size == (800, 800)
        assert scenario.compact_hud
        assert scenario.lane_collision_margin == .02
        assert scenario.road_collision_mode == "road_edges"
        assert scenario.road_boundary_collision is not None
        assert not scenario.road_boundary_collision(scenario.vertices).any()
        assert scenario.nod_agent_generation.shape == (1, p.n_agents)
        observation_shape = scenario.observation(scenario.world.agents[0]).shape
        for agent in scenario.world.agents:
            assert torch.isfinite(scenario.observation(agent)).all()
        for agent_id in range(p.n_agents):
            path_id = int(scenario.ref_paths_agent_related.path_id[0, agent_id])
            point_id = int(scenario.ref_paths_agent_related.point_id[0, agent_id])
            assert point_id in scenario.map.parser.reference_paths[path_id]["spawn_point_indices"]
        assert not scenario.collisions.with_agents.any()
        assert not scenario.collisions.with_lanelets.any()
        assert not td["done"].any()
    finally:
        env.close()

    # Compare features with an existing map under the same model configuration.
    baseline = ScenarioRoadTraffic()
    torch.manual_seed(101)
    baseline.parameters = Parameters.from_json(ROOT/"configs/current/error_scale_repeat/02_control.json")
    baseline.parameters.scenario_type = "intersection_2"
    baseline.parameters.n_agents = SCENARIOS["intersection_2"]["n_agents"]
    baseline.parameters.is_testing_mode = True
    baseline.parameters.is_add_noise = False
    baseline.parameters.num_vmas_envs = 1
    baseline.parameters.max_steps = 16
    original = VmasEnv(scenario=baseline, num_envs=1, continuous_actions=True,
                       max_steps=16, device="cpu", n_agents=baseline.parameters.n_agents)
    try:
        original.reset()
        assert baseline.observation(baseline.world.agents[0]).shape == observation_shape
        # Legacy OSM maps retain their existing CPM-default viewport.
        assert baseline.world_x_dim == SCENARIOS["CPM_mixed"]["world_x_dim"]
        assert not baseline.compact_hud
        assert baseline.lane_collision_margin == 0.
        assert baseline.road_collision_mode == "reference_lane"
        assert baseline.road_boundary_collision is None
    finally:
        original.close()


def test_full_entry_spawn_region_and_impossible_insertion_stops(capsys):
    from scenarios.road_traffic import ScenarioRoadTraffic

    ways, descriptions, _, metadata, _, _, _ = build_geometry()
    for route in metadata:
        entry = next(key for key, desc in descriptions.items()
                     if desc["kind"] == "entry" and desc["lane"] == route["lane"]
                     and desc["arm"] == route["origin"])
        assert route["spawn_end_index"] == len(ways[entry]) - 6

    class OccupiedAgent:
        def __init__(self):
            self.state = SimpleNamespace(pos=torch.zeros(1, 2))

        def set_pos(self, position, batch_index):
            self.state.pos[batch_index] = position

    # A blocked respawn must stop without moving any other car or accepting
    # an overlapping start. This does not step the simulation.
    scenario = SimpleNamespace()
    agents = [OccupiedAgent(), OccupiedAgent()]
    scenario.parameters = SimpleNamespace(is_testing_mode=True, scenario_type=NAME)
    scenario.n_agents = 2
    scenario.world = SimpleNamespace(agents=agents)
    scenario.constants = SimpleNamespace(reset_agent_min_distance=torch.tensor(.215))
    scenario.ref_paths_agent_related = SimpleNamespace(
        path_id=torch.zeros(1, 2, dtype=torch.long),
        point_id=torch.zeros(1, 2, dtype=torch.long))
    path = {"center_line": torch.zeros(10, 2), "spawn_point_indices": [4]}
    with pytest.raises(RuntimeError, match="No collision-free entry spawn for agent 2"):
        ScenarioRoadTraffic._reset_init_state(scenario, 0, 1, True, False, None, [path], agents)
    assert "random_count = 2000" not in capsys.readouterr().out
