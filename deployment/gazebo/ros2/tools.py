#!/usr/bin/python3
"""Camera/lidar checks and bounded manual control for the isolated ROS2 arena."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_srvs.srv import Empty, SetBool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tag_detection import detect_tags


def yaw(odometry):
    q = odometry.pose.pose.orientation
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def timestamp(message):
    return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9


class Arena(Node):
    def __init__(self):
        super().__init__("sigmarl_jetauto_tools")
        self.config = json.loads((ROOT / "generated/ros2/arena_metadata.json").read_text())
        self.image = self.info = None
        self.states, self.scans = {}, {}
        self.create_subscription(Image, "/overhead_camera/image_raw", self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/overhead_camera/camera_info", self.on_info, qos_profile_sensor_data)
        self.publishers_by_car = {}
        interfaces = self.config["interfaces"]
        for definition in self.config["cars"]:
            car = definition["name"]
            self.create_subscription(Odometry, f"/{car}/ground_truth", lambda msg, car=car: self.states.update({car: msg}), qos_profile_sensor_data)
            self.create_subscription(LaserScan, f"/{car}/{interfaces['scan_topic']}", lambda msg, car=car: self.scans.update({car: msg}), qos_profile_sensor_data)
            self.publishers_by_car[car] = self.create_publisher(Twist, f"/{car}/{interfaces['command_topic']}", 1)

    def on_image(self, message):
        self.image = message

    def on_info(self, message):
        self.info = message

    def until(self, predicate, timeout=60):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not predicate():
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for the Gazebo camera, lidar or simulation clock")
            rclpy.spin_once(self, timeout_sec=0.05)

    def check(self):
        self.until(lambda: self.image is not None and self.info is not None and
                   len(self.scans) == len(self.config["cars"]) and len(self.states) == len(self.config["cars"]))
        metadata = self.config
        expected = {t["id"] for t in metadata["ground_tags"]} | {c["tag_id"] for c in metadata["cars"]}
        source = self.image
        if source.encoding not in ("rgb8", "bgr8"):
            raise RuntimeError(f"Unexpected image encoding: {source.encoding}")
        image = np.frombuffer(source.data, np.uint8).reshape(source.height, source.step)
        image = image[:, :source.width * 3].reshape(source.height, source.width, 3).copy()
        if source.encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        corners, ids = detect_tags(image)
        detected = set() if ids is None else set(ids.flatten().tolist())
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
        output = ROOT / "runtime/ros2/camera_tags.png"
        if not cv2.imwrite(str(output), image):
            raise RuntimeError("Could not save camera preview")
        if source.header.frame_id != "overhead_camera_optical" or self.info.header.frame_id != source.header.frame_id:
            raise RuntimeError("Incorrect camera optical frame")
        if (source.width, source.height) != (self.info.width, self.info.height):
            raise RuntimeError("Camera intrinsics and image dimensions disagree")
        expected_focal = source.width / (2 * math.tan(metadata["camera"]["horizontal_fov"] / 2))
        if not np.isfinite(self.info.k).all() or abs(self.info.k[0] - expected_focal) > 1:
            raise RuntimeError("Camera focal length disagrees with configured field of view")
        result = {"scenario": metadata.get("scenario", "intersection"), "vehicles": sorted(self.states),
                  "preview": str(output), "resolution": [source.width, source.height],
                  "detected_ids": sorted(detected), "missing_ids": sorted(expected - detected),
                  "lidars": {name: {"frame": msg.header.frame_id, "samples": len(msg.ranges),
                                    "range_min": msg.range_min, "range_max": msg.range_max}
                             for name, msg in self.scans.items()}}
        print(json.dumps(result, indent=2))
        if expected - detected:
            raise RuntimeError("Not all configured tags are visible; inspect the preview")
        for car, scan in self.scans.items():
            if scan.header.frame_id != car + "/lidar_frame" or len(scan.ranges) != metadata["lidar"]["samples"]:
                raise RuntimeError("A1 scan frame or sample count is incorrect")
        (ROOT / "runtime/ros2/scene_check.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    def command(self, car, x=0, y=0, angular=0):
        message = Twist()
        message.linear.x, message.linear.y, message.angular.z = float(x), float(y), float(angular)
        self.publishers_by_car[car].publish(message)

    def set_demo(self, enabled):
        if "traffic" not in self.config:
            if enabled:
                raise RuntimeError("This scene has no local traffic demo")
            return
        client = self.create_client(SetBool, "/local_traffic/set_enabled")
        if not client.wait_for_service(timeout_sec=10):
            raise RuntimeError("Local traffic service is unavailable; inspect runtime/ros2/traffic.log")
        request = SetBool.Request()
        request.data = enabled
        future = client.call_async(request)
        self.until(future.done, timeout=10)
        response = future.result()
        if not response.success:
            raise RuntimeError(response.message)
        if not enabled:
            self.until(lambda: len(self.states) == len(self.config["cars"]) and all(
                math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y) < 0.025 and
                abs(msg.twist.twist.angular.z) < 0.025 for msg in self.states.values()))
        print(response.message)

    def stop_policy(self):
        client = self.create_client(SetBool, "/policy_bridge/set_enabled")
        if not client.wait_for_service(timeout_sec=1.5):
            return
        request = SetBool.Request()
        request.data = False
        future = client.call_async(request)
        self.until(future.done, timeout=5)
        if not future.result().success:
            raise RuntimeError(future.result().message)
        self.until(lambda: len(self.states) == len(self.config["cars"]) and all(
            math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y) < 0.025 and
            abs(msg.twist.twist.angular.z) < 0.025 for msg in self.states.values()))
        print(future.result().message)

    def phase(self, car, duration, command=None):
        self.until(lambda: car in self.states and self.publishers_by_car[car].get_subscription_count() > 0)
        initial = timestamp(self.states[car])
        deadline = time.monotonic() + 90
        while rclpy.ok() and timestamp(self.states[car]) - initial < duration:
            if time.monotonic() >= deadline:
                raise RuntimeError("Simulation clock stopped or simulation is too slow")
            if command is not None:
                self.command(car, *command)
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.02)
        return self.states[car]

    def test(self, car):
        try:
            initial = self.phase(car, 0.5, (0, 0, 0))
            forward = self.phase(car, 2, (0.15, 0, 0))
            heading = yaw(initial)
            dx = forward.pose.pose.position.x - initial.pose.pose.position.x
            dy = forward.pose.pose.position.y - initial.pose.pose.position.y
            distance = math.cos(heading) * dx + math.sin(heading) * dy
            stopped = self.phase(car, 1.5)  # Stop publishing to verify the watchdog.
            lateral_start = self.phase(car, 0.5, (0, 0, 0))
            sideways = self.phase(car, 2, (0, 0.12, 0))
            dx = sideways.pose.pose.position.x - lateral_start.pose.pose.position.x
            dy = sideways.pose.pose.position.y - lateral_start.pose.pose.position.y
            lateral_distance = -math.sin(heading) * dx + math.cos(heading) * dy
            turning_start = self.phase(car, 0.5, (0, 0, 0))
            turning_end = self.phase(car, 2, (0, 0, 0.4))
            rotation = math.atan2(math.sin(yaw(turning_end) - yaw(turning_start)), math.cos(yaw(turning_end) - yaw(turning_start)))
            rotation_drift = math.hypot(turning_end.pose.pose.position.x - turning_start.pose.pose.position.x,
                                        turning_end.pose.pose.position.y - turning_start.pose.pose.position.y)
            self.phase(car, 0.8, (0, 0, 0))
            result = {"forward_m": distance, "lateral_m": lateral_distance,
                      "rotation_rad": rotation, "watchdog_speed_mps": abs(stopped.twist.twist.linear.x),
                      "rotation_reported_speed_mps": math.hypot(turning_end.twist.twist.linear.x, turning_end.twist.twist.linear.y),
                      "rotation_drift_m": rotation_drift, "root_height_m": turning_end.pose.pose.position.z}
            print(json.dumps(result, indent=2))
            if distance < 0.18 or lateral_distance < 0.12 or rotation < 0.4:
                raise RuntimeError("JetAuto forward/lateral/rotation motion check failed")
            if result["watchdog_speed_mps"] > 0.03 or abs(result["root_height_m"]) > 0.05:
                raise RuntimeError("Watchdog or ground contact check failed")
            if rotation_drift > 0.03:
                raise RuntimeError("In-place rotation moved the base reference point")
            if result["rotation_reported_speed_mps"] > 0.02:
                raise RuntimeError("Odometry velocity does not describe the base reference point")
            (ROOT / "runtime/ros2/motion_check.json").write_text(json.dumps(result, indent=2) + "\n")
            print("ROS2 JetAuto motion checks passed")
        finally:
            for _ in range(5):
                self.command(car)
                rclpy.spin_once(self, timeout_sec=0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "drive", "test", "reset", "demo-start", "demo-stop", "policy-stop", "policy-status"))
    configured_cars = json.loads((ROOT / "generated/ros2/arena_metadata.json").read_text())["cars"]
    parser.add_argument("--car", choices=[car["name"] for car in configured_cars], default=configured_cars[0]["name"])
    parser.add_argument("--vx", type=float, default=0.15)
    parser.add_argument("--vy", type=float, default=0)
    parser.add_argument("--wz", type=float, default=0)
    parser.add_argument("--duration", type=float, default=2, help="Simulation seconds")
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in (args.vx, args.vy, args.wz, args.duration)) or args.duration <= 0:
        parser.error("Commands must be finite and duration positive")
    rclpy.init()
    node = Arena()
    try:
        if args.action in ("drive", "test", "reset", "demo-start", "policy-stop"):
            node.stop_policy()
        if args.action in ("drive", "test", "reset"):
            node.set_demo(False)
        if args.action == "policy-status":
            print((ROOT / "runtime/ros2/policy_status.json").read_text())
        elif args.action == "policy-stop":
            print("Policy stop requested")
        elif args.action in ("demo-start", "demo-stop"):
            node.set_demo(args.action == "demo-start")
        elif args.action == "reset":
            client = node.create_client(Empty, "/reset_simulation")
            if not client.wait_for_service(timeout_sec=10):
                raise RuntimeError("Gazebo reset_simulation service is unavailable")
            future = client.call_async(Empty.Request())
            node.until(future.done, timeout=10)
            future.result()
            initial = node.config["cars"]
            node.until(lambda: all(car["name"] in node.states and
                       math.hypot(node.states[car["name"]].pose.pose.position.x - car["x"],
                                  node.states[car["name"]].pose.pose.position.y - car["y"]) < 0.01
                       for car in initial))
            print("JetAuto arena reset to initial positions")
        elif args.action == "check":
            node.check()
        elif args.action == "test":
            node.test(args.car)
        else:
            try:
                node.phase(args.car, args.duration, (args.vx, args.vy, args.wz))
                node.phase(args.car, 0.5, (0, 0, 0))
            finally:
                for _ in range(5):
                    node.command(args.car)
                    time.sleep(0.05)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
