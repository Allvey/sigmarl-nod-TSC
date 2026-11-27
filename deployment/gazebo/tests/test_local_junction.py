"""Geometry and route contracts for the local two-lane experiment."""
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configuration import load_config
from cpm_map import sample_route, write_mesh
from generate_world import tag_cells, validate
from local_junction import inside_road, prepare, project_route, route_length, select_route_indices
from tag_detection import detect_tags


class LocalJunctionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.data = prepare(load_config(ROOT / "config/jetauto_local_junction.json"))

    def test_one_lane_each_direction_and_connected_routes(self):
        validate(self.config)
        lanes = self.data["lanelets"]
        self.assertEqual(sum(lane["kind"] == "incoming" for lane in lanes), 4)
        self.assertEqual(sum(lane["kind"] == "outgoing" for lane in lanes), 4)
        self.assertEqual(sum(lane["kind"] == "connector" for lane in lanes), 12)
        by_id = {lane["id"]: lane for lane in lanes}
        routes = self.data["routes"]["intersection"]
        self.assertEqual(len(routes), 12)
        for route in routes:
            a, b, c = [by_id[i] for i in route["lanelet_ids"]]
            np.testing.assert_allclose(a["center_line"][-1], b["center_line"][0], atol=1e-12)
            np.testing.assert_allclose(b["center_line"][-1], c["center_line"][0], atol=1e-12)
            self.assertIn(c["id"], b["successor"])

    def test_routes_stay_on_local_road_and_project_correctly(self):
        for route in self.data["routes"]["intersection"]:
            points = route["center_line"]
            for distance in np.linspace(0, route_length(points), 201):
                x, y, _ = sample_route(points, distance)
                self.assertTrue(inside_road((x, y), 0.45, 2.5 + 1e-10, 0.35))
                self.assertAlmostEqual(project_route(np.array([x, y]), points), distance, places=7)

    def test_spawn_tag_camera_and_activity_bounds(self):
        self.assertEqual(self.config["map"]["activity_bounds"], [-2.5, 2.5, -2.5, 2.5])
        positions = np.array([[car["x"], car["y"]] for car in self.config["cars"]])
        distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
        self.assertEqual(len(positions), 3)
        self.assertGreater(distances[np.triu_indices(len(positions), 1)].min(), 2.5)
        half_view = self.config["camera"]["height"] * np.tan(self.config["camera"]["horizontal_fov"] / 2)
        for tag in self.config["ground_tags"]:
            self.assertLess(max(abs(tag["x"]), abs(tag["y"])) + tag["size"], half_view)

    def test_mixed_routes_and_staggered_spawns_are_preserved(self):
        cars, routes = self.config["cars"], self.data["routes"]["intersection"]
        self.assertEqual(select_route_indices(cars, routes), [1, 5, 6])
        self.assertEqual([car["maneuver"] for car in cars], ["left", "right", "straight"])
        self.assertEqual(len({car["spawn_distance"] for car in cars}), 3)
        np.testing.assert_allclose([max(abs(car["x"]), abs(car["y"])) for car in cars], [2.1, 1.75, 1.95])
        for car in cars:
            x, y, heading = sample_route(routes[car["route_index"]]["center_line"], car["spawn_distance"])
            np.testing.assert_allclose([car["x"], car["y"], car["yaw"]], [x, y, heading])

    def test_uniform_maneuver_requires_explicit_override(self):
        cars, routes = self.config["cars"], self.data["routes"]["intersection"]
        for maneuver, expected in (("straight", [0, 3, 6]), ("left", [1, 4, 7]), ("right", [2, 5, 8])):
            indices = select_route_indices(cars, routes, maneuver)
            self.assertEqual(indices, expected)
            for car, index in zip(cars, indices):
                self.assertEqual(routes[index]["approach"], car["approach"])
        self.assertEqual([car["route_index"] for car in cars], [1, 5, 6])

    def test_small_rotated_vehicle_and_ground_tags_decode(self):
        canvas = np.full((350, 700, 3), 110, dtype=np.uint8)
        expected = [tag["id"] for tag in self.config["ground_tags"]] + [car["tag_id"] for car in self.config["cars"]]
        for i, tag_id in enumerate(expected):
            bitmap = np.rot90(np.pad(tag_cells(tag_id), 1, constant_values=255), i % 4)
            patch = cv2.resize(bitmap, (70, 70), interpolation=cv2.INTER_NEAREST)
            x, y = (i % 4) * 175 + 40, (i // 4) * 175 + 40
            canvas[y:y + 70, x:x + 70] = patch[:, :, None]
        _, ids = detect_tags(canvas)
        self.assertEqual(ids.flatten().tolist(), sorted(expected))

    def test_vehicle_footprints_clear_all_turns(self):
        half_length = self.config["vehicle"]["length"] / 2
        half_width = self.config["vehicle"]["width"] / 2
        for route in self.data["routes"]["intersection"]:
            points = route["center_line"]
            for distance in np.linspace(0.45, route_length(points) - 0.4, 101):
                x, y, angle = sample_route(points, distance)
                for dx in (-half_length, half_length):
                    for dy in (-half_width, half_width):
                        point = (x + dx * np.cos(angle) - dy * np.sin(angle),
                                 y + dx * np.sin(angle) + dy * np.cos(angle))
                        self.assertTrue(inside_road(point, 0.45, 2.5, 0.35))

    def test_paint_does_not_draw_virtual_connector_boundaries(self):
        marks = self.data["markings"]
        self.assertEqual(sum(mark["style"] == "dashed" for mark in marks), 4)
        for mark in marks:
            for point in mark["points"]:
                self.assertGreaterEqual(max(abs(c) for c in point), 0.45)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "road.dae"
            counts = write_mesh(self.data, self.config["map"], path)
            self.assertTrue(all(value > 0 for value in counts.values()))
            ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
            mesh = ET.parse(path).getroot().find("c:library_geometries/c:geometry/c:mesh", ns)
            positions = np.fromstring(mesh.find("c:source[@id='positions']/c:float_array", ns).text, sep=" ").reshape(-1, 3)
            for group in mesh.findall("c:triangles", ns):
                indices = np.fromstring(group.find("c:p", ns).text, sep=" ", dtype=int).reshape(-1, 2)[:, 0].reshape(-1, 3)
                triangles = positions[indices]
                self.assertTrue((np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])[:, 2] > 0).all())


if __name__ == "__main__":
    unittest.main()
