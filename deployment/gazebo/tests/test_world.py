"""Offline checks: generated road geometry, camera coverage and decodable fiducials."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("gazebo_world", ROOT / "generate_world.py")
WORLD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORLD)


class WorldTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config/arena.json").read_text())

    def test_six_actual_apriltags_decode(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        ids = [tag["id"] for tag in self.config["ground_tags"]] + [car["tag_id"] for car in self.config["cars"]]
        for tag_id in ids:
            bitmap = np.pad(WORLD.tag_cells(tag_id), 1, constant_values=255)
            image = cv2.resize(bitmap, (400, 400), interpolation=cv2.INTER_NEAREST)
            if hasattr(cv2.aruco, "ArucoDetector"):
                _, detected, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
            else:
                _, detected, _ = cv2.aruco.detectMarkers(image, dictionary)
            self.assertIsNotNone(detected)
            self.assertEqual(detected.flatten().tolist(), [tag_id])

    def test_generated_physical_models_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = WORLD.generate(self.config, Path(directory))
            world = ET.parse(path).getroot().find("world")
            self.assertFalse(world.findall("include"), "World must not need model downloads")
            for car in self.config["cars"]:
                model = world.find(f"model[@name='{car['name']}']")
                self.assertEqual(len(model.findall("joint")), 6)
                self.assertEqual(len(model.findall("link/collision")), 5)
                self.assertEqual(model.find("plugin").get("filename"), "libsigmarl_drive.so")
            sensor = world.find("model[@name='overhead_camera']/link/sensor")
            self.assertEqual(sensor.find("camera/image/width").text, "2048")
            metadata = json.loads((Path(directory) / "arena_metadata.json").read_text())
            self.assertEqual(metadata["camera_optical_pose"]["quaternion_xyzw"], [1, 0, 0, 0])

    def test_duplicate_tag_rejected(self):
        self.config["cars"][0]["tag_id"] = 0
        with self.assertRaises(ValueError):
            WORLD.validate(self.config)

    def test_incomplete_camera_coverage_rejected(self):
        self.config["camera"]["height"] = 1
        with self.assertRaises(ValueError):
            WORLD.validate(self.config)
        other = copy.deepcopy(self.config)
        other["camera"]["height"] = 3
        other["camera"]["height_pixels"] = 512
        with self.assertRaises(ValueError):
            WORLD.validate(other)

    def test_jetauto_hardware_profile_and_road_clearance(self):
        config = json.loads((ROOT / "config/jetauto_a1.json").read_text())
        WORLD.validate(config)
        self.assertEqual(config["hardware"]["lidar"], "SLAMTEC A1")
        self.assertEqual(config["hardware"]["ros_distro"], "humble")
        self.assertGreater(config["lane_width"], config["vehicle"]["width"] + 0.1)
        config["ros_version"] = 1
        with self.assertRaises(ValueError):
            WORLD.validate(config)

    def test_urdf_lidar_transform_includes_root_offset_and_mount_rotation(self):
        spec = importlib.util.spec_from_file_location("jetauto_model", ROOT / "jetauto_model.py")
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
        source = ET.fromstring('''<robot name="example">
          <link name="base_footprint"/><link name="base_link"/><link name="lidar_frame"/>
          <joint name="base" type="fixed"><parent link="base_footprint"/><child link="base_link"/><origin xyz="0 0 0.005"/></joint>
          <joint name="laser" type="fixed"><parent link="base_link"/><child link="lidar_frame"/><origin xyz="0.1 0 0.13" rpy="0 0 3.141592653589793"/></joint>
        </robot>''')
        pose = model.link_transforms(source)["lidar_frame"]
        np.testing.assert_allclose(pose[:3, 3], [0.1, 0, 0.135])
        np.testing.assert_allclose(pose[:3, 0], [-1, 0, 0], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
