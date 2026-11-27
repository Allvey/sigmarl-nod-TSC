"""Verify the imported CPM map against canonical source coordinates and topology."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configuration import load_config
from cpm_map import dashed_segments, prepare, project_root, read_map, write_mesh
from generate_world import validate


class CPMTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.data = prepare(load_config(ROOT / "config/jetauto_cpm_mixed.json"))

    def test_native_coordinates_scale_without_geometry_changes(self):
        self.assertEqual(len(self.data["lanelets"]), 104)
        original = ET.parse(project_root() / "assets/maps/cpm.xml").getroot()
        for source, lane in zip(original.findall("lanelet"), self.data["lanelets"]):
            self.assertEqual(int(source.get("id")), lane["id"])
            self.assertEqual(source.findtext("laneletType"), lane["type"])
            for side in ("left", "right"):
                points = np.array([[float(p.findtext("x")), float(p.findtext("y"))]
                                   for p in source.findall(side + "Bound/point")])
                np.testing.assert_allclose(lane[side + "_boundary"], 3 * (points - [2.25, 2.0]))
            self.assertEqual(lane["successor"], [int(x.get("ref")) for x in source.findall("successor")])
        self.assertEqual(self.data["source_sha256"], hashlib.sha256((project_root() / "assets/maps/cpm.xml").read_bytes()).hexdigest())

    def test_four_vehicle_profile_camera_and_corner_tag_clearance(self):
        validate(self.config)
        self.assertEqual(self.config["arena_size"], [15.0, 13.5])
        self.assertAlmostEqual(self.config["lane_width"], 0.45)
        self.assertEqual(len(self.config["cars"]), 4)
        self.assertEqual([car["tag_id"] for car in self.config["cars"]], [10, 11, 12, 13])
        positions = np.array([[car["x"], car["y"]] for car in self.config["cars"]])
        distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
        self.assertGreater(distances[np.triu_indices(4, 1)].min(), 4)
        for tag in self.config["ground_tags"]:
            self.assertGreater(abs(tag["x"]) - tag["size"] * 1.25 / 2, 6.75)
            self.assertGreater(abs(tag["y"]) - tag["size"] * 1.25 / 2, 6.0)

    def test_reference_route_groups_and_spawn_arc_distance(self):
        self.assertEqual({group: len(routes) for group, routes in self.data["routes"].items()},
                         {"intersection": 24, "merge_in": 4, "merge_out": 4, "loop": 40})
        for car in self.config["cars"]:
            route = self.data["routes"][car["route_group"]][car["route_index"]]
            points = np.array(route["center_line"])
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            index = np.searchsorted(np.cumsum(lengths), car["spawn_distance"])
            before = lengths[:index].sum()
            vector = points[index + 1] - points[index]
            expected = points[index] + (car["spawn_distance"] - before) * vector / lengths[index]
            np.testing.assert_allclose([car["x"], car["y"]], expected)
            self.assertAlmostEqual(car["yaw"], np.arctan2(vector[1], vector[0]))

    def test_mesh_faces_are_upward_and_all_marking_types_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "road.dae"
            counts = write_mesh(self.data, self.config["map"], path)
            self.assertTrue(all(count > 0 for count in counts.values()))
            ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
            mesh = ET.parse(path).getroot().find("c:library_geometries/c:geometry/c:mesh", ns)
            positions = np.fromstring(mesh.find("c:source[@id='positions']/c:float_array", ns).text, sep=" ").reshape(-1, 3)
            for group in mesh.findall("c:triangles", ns):
                indices = np.fromstring(group.find("c:p", ns).text, sep=" ", dtype=int).reshape(-1, 2)[:, 0].reshape(-1, 3)
                triangles = positions[indices]
                normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
                self.assertTrue((normals[:, 2] > 0).all())

    def test_dashes_keep_phase_across_polyline_samples(self):
        segments = list(dashed_segments([[0, 0], [0.3, 0], [0.6, 0], [1, 0]], 0.18, 0.12))
        self.assertAlmostEqual(sum(np.linalg.norm(b - a) for a, b in segments), 0.64)

    def test_profile_inheritance_cycle_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cycle.json"
            path.write_text(json.dumps({"extends": "cycle.json"}))
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
