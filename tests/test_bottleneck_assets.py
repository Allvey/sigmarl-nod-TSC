"""Check exported geometry, physical curbs and CLI integration, without rollouts."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.build_bottleneck_assets import (
    NAME, body_samples, build_geometry, load_design, road_sections, stations, write_assets,
)
from utilities.constants import AGENTS, SCENARIOS
from utilities.parse_osm import ParseOSM
from utilities.road_collision import RoadBoundaryCollision


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def parsed():
    return ParseOSM(NAME, "cpu")


def rectangle(x, y):
    return [[x-.08, y-.04], [x+.08, y-.04], [x+.08, y+.04], [x-.08, y+.04]]


def test_checked_in_assets_are_reproducible(tmp_path):
    write_assets(tmp_path)
    for suffix in (".osm", ".json", ".render.json", ".boundaries.json", ".svg"):
        assert (tmp_path/(NAME+suffix)).read_bytes() == (ROOT/"assets/maps"/(NAME+suffix)).read_bytes()


def test_shared_throat_and_connected_routes_in_both_directions(parsed):
    design = load_design()
    ways, desc, routes, metadata, neighbors, dims = build_geometry(design)
    assert len(ways) == 18 and len(routes) == len(parsed.reference_paths) == 8
    assert design["vehicle_size"] == [AGENTS["length"], AGENTS["width"]]
    for direction in ("eastbound", "westbound"):
        directional = [route for route, item in zip(routes, metadata) if item["direction"] == direction]
        assert len({r[0] for r in directional}) == len({r[-1] for r in directional}) == 2
        assert len({r[2] for r in directional}) == 1  # one actual shared throat lane
        assert {(item["entry_lane"], item["exit_lane"]) for item in metadata
                if item["direction"] == direction} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for route, path in zip(routes, parsed.reference_paths):
        for first, second in zip(route, route[1:]):
            np.testing.assert_allclose(ways[first][-1], ways[second][0], atol=1e-9)
        expected = np.vstack([ways[route[0]]] + [ways[key][1:] for key in route[1:]])
        np.testing.assert_allclose(path["center_line"], expected, atol=1e-6)
        assert np.all(expected > 0) and np.all(expected < dims)
    for key, adjacent in neighbors.items():
        assert key in adjacent
        for other in adjacent:
            assert key in neighbors[other]
            assert desc[key]["direction"] == desc[other]["direction"]
    # Both approach lanes see each other, before occupying the common throat.
    for first, second in (("1", "2"), ("10", "11")):
        assert second in neighbors[first]


def test_full_width_throat_and_real_vehicle_clearance(parsed):
    design = load_design()
    section = road_sections(design)[0]
    x = stations(design)
    mid = (x[2]+x[3])/2
    assert np.interp(mid, section["x"], section["top"]-section["bottom"]) == pytest.approx(.18)
    assert section["top"][0]-section["bottom"][0] == pytest.approx(.36)
    collision = RoadBoundaryCollision.from_asset(SCENARIOS[NAME]["geometry_asset_path"])
    for path in parsed.reference_paths:
        p = path["center_line"].numpy()
        delta = np.diff(p, axis=0)
        segment_lengths = np.linalg.norm(delta, axis=1)
        assert segment_lengths.min() > .025 and segment_lengths.max() < .036
        yaw = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
        curvature = np.abs(np.diff(yaw))/((segment_lengths[1:]+segment_lengths[:-1])/2)
        assert curvature.max() < .5*math.tan(math.radians(AGENTS["max_steering"]))/(
            AGENTS["l_f"]+AGENTS["l_r"])
        body = body_samples(p, design["vehicle_size"])
        # Actual exported polygons and parser coordinates, not only analytic
        # builder checks. Include both tapered sections and terminal overhangs.
        corners = torch.tensor(body[[0, 12, 14, 2]].transpose(1, 0, 2), dtype=torch.float32)
        assert not collision(corners).any()


def test_only_curbs_and_median_are_collision_boundaries():
    design = load_design()
    x, cy, width = stations(design), design["world_height"]/2, design["lane_width"]
    collision = RoadBoundaryCollision.from_asset(SCENARIOS[NAME]["geometry_asset_path"])
    divider_y = cy-design["median_width"]/2-width
    samples = [rectangle(x[0]+.7, divider_y),  # straddles a white dash
               rectangle(x[0]+.7, cy),          # central median
               rectangle(x[0]+.7, cy-.43),     # outer curb
               rectangle((x[2]+x[3])/2, divider_y),  # former outer lane is now grass
               rectangle((x[2]+x[3])/2, cy-design["median_width"]/2-width/2)]
    assert collision(torch.tensor(samples)).tolist() == [False, True, True, True, False]


def test_spawn_regions_stay_upstream_and_camera_keeps_map_visible(parsed):
    design = load_design()
    meta = json.loads((ROOT/"assets/maps"/(NAME+".json")).read_text())
    scene = SCENARIOS[NAME]
    for path, item in zip(parsed.reference_paths, meta["scene"]["route_metadata"]):
        entry = parsed.lanelets_all[parsed._lanelet_id_to_index[path["lanelet_IDs"][0]]]["center_line"]
        indices = path["spawn_point_indices"]
        assert 0 < min(indices) < max(indices) < len(entry)-5
        np.testing.assert_allclose(path["center_line"][indices], entry[indices], atol=1e-6)
        assert meta["lanelets"][path["lanelet_IDs"][0]]["kind"] == "entry"
        assert indices == item["spawn_point_indices"]
    spawn_points = torch.unique(torch.cat([p["center_line"][p["spawn_point_indices"]]
                                          for p in parsed.reference_paths]), dim=0)
    distance = torch.cdist(spawn_points, spawn_points)
    distance.fill_diagonal_(float("inf"))
    assert distance.min() > 1.2*math.hypot(AGENTS["width"], AGENTS["length"])
    assert len(spawn_points) >= scene["n_agents"]
    aspect = scene["viewer_size"][0]/scene["viewer_size"][1]
    half_height = scene["viewer_zoom"]**2
    assert 2*half_height >= scene["world_y_dim"]
    assert 2*half_height*aspect >= scene["world_x_dim"]
    assert AGENTS["length"]*scene["viewer_size"][1]/(2*half_height) >= 24.
    assert scene["n_agents"] == 20 and scene["compact_hud"]
    assert scene["road_collision_mode"] == "road_edges" and scene["lane_collision_margin"] == 0.


def test_invalid_spawn_density_is_rejected_before_writing(tmp_path):
    design = load_design()
    design["n_agents"] = 50
    filename = tmp_path/"dense.design.json"
    filename.write_text(json.dumps(design))
    output = tmp_path/"assets"
    with pytest.raises(ValueError, match="Entry spawn slots"):
        write_assets(output, design_path=filename)
    assert not output.exists()
