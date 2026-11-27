#!/usr/bin/python3
"""Ground-truth route-following demo with exclusive junction reservations."""
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import SetBool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cpm_map import sample_route
from local_junction import inside_road, project_route, route_length


class LocalTraffic(Node):
    def __init__(self):
        super().__init__("local_traffic")
        self.config = json.loads((ROOT / "generated/ros2/arena_metadata.json").read_text())
        self.routes = json.loads((ROOT / "generated/ros2/cpm_map.json").read_text())["routes"]["intersection"]
        self.settings = self.config["traffic"]
        self.enabled = self.settings["enabled"]
        self.states, self.received, self.publishers_by_car, self.cars = {}, {}, {}, {}
        self.queue, self.owner, self.fault = [], None, None
        self.max_coordinate, self.min_separation, self.last_status = 0.0, float("inf"), 0.0
        self.last_sim_time = None
        self.reset_client = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.service = self.create_service(SetBool, "/local_traffic/set_enabled", self.set_enabled)
        for definition in self.config["cars"]:
            name = definition["name"]
            self.cars[name] = {"definition": definition, "route": definition["route_index"], "phase": "active",
                               "progress": definition["spawn_distance"], "completed": 0, "future": None}
            self.publishers_by_car[name] = self.create_publisher(Twist, f"/{name}/{self.config['interfaces']['command_topic']}", 1)
            self.create_subscription(Odometry, f"/{name}/ground_truth", lambda msg, name=name: self.receive(name, msg), qos_profile_sensor_data)
        self.timer = self.create_timer(0.05, self.tick)

    def receive(self, name, message):
        self.states[name] = message
        self.received[name] = time.monotonic()

    def stop(self, name):
        self.publishers_by_car[name].publish(Twist())

    def set_enabled(self, request, response):
        self.enabled = request.data
        self.queue, self.owner, self.fault = [], None, None
        if request.data:
            self.last_sim_time = None
        for name, car in self.cars.items():
            self.stop(name)
            if request.data:
                car["phase"], car["future"] = "resetting", None
                car["route"] = car["definition"]["route_index"]
        response.success = True
        response.message = "Local traffic reset/resume requested" if request.data else "Local traffic paused"
        return response

    def reset_car(self, name, car):
        if not self.reset_client.service_is_ready():
            return
        future = car["future"]
        if future is None:
            definition = car["definition"]
            x, y, heading = sample_route(self.routes[car["route"]]["center_line"], definition["spawn_distance"])
            request = SetEntityState.Request()
            request.state.name, request.state.reference_frame = name, "world"
            request.state.pose.position.x, request.state.pose.position.y, request.state.pose.position.z = x, y, 0.002
            request.state.pose.orientation.z, request.state.pose.orientation.w = math.sin(heading / 2), math.cos(heading / 2)
            car["future"] = self.reset_client.call_async(request)
            return
        if not future.done():
            return
        if future.exception() or not future.result().success:
            self.fault = f"Failed to respawn {name}"
            return
        x, y, _ = sample_route(self.routes[car["route"]]["center_line"], car["definition"]["spawn_distance"])
        position = self.states[name].pose.pose.position
        if math.hypot(position.x - x, position.y - y) < 0.04:
            car["phase"], car["future"] = "active", None
            car["progress"] = car["definition"]["spawn_distance"]

    def status(self):
        now = time.monotonic()
        if now - self.last_status < 0.5:
            return
        self.last_status = now
        data = {"controller": "rule-based demo using simulator ground truth", "enabled": self.enabled,
                "owner": self.owner, "queue": self.queue, "fault": self.fault,
                "max_abs_coordinate_m": self.max_coordinate,
                "min_pair_separation_m": self.min_separation if math.isfinite(self.min_separation) else None,
                "vehicles": {name: {"phase": car["phase"], "route": car["route"], "completed": car["completed"],
                                    "progress_m": car["progress"]} for name, car in self.cars.items()}}
        path = ROOT / "runtime/ros2/traffic_status.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(path)

    def tick(self):
        self.status()
        if not self.enabled:
            return
        if self.fault or len(self.states) != len(self.cars):
            for name in self.cars:
                self.stop(name)
            return
        if any(time.monotonic() - self.received[name] > 1 for name in self.cars):
            for name in self.cars:
                self.stop(name)
            return
        sim_time = min(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 for msg in self.states.values())
        if self.last_sim_time is not None and sim_time < self.last_sim_time - 0.1:
            self.queue, self.owner = [], None
            for car in self.cars.values():
                car["phase"], car["future"], car["route"] = "active", None, car["definition"]["route_index"]
                car["progress"] = car["definition"]["spawn_distance"]
        self.last_sim_time = sim_time
        positions = np.array([[self.states[name].pose.pose.position.x, self.states[name].pose.pose.position.y] for name in self.cars])
        self.max_coordinate = max(self.max_coordinate, float(np.abs(positions).max()))
        separations = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
        self.min_separation = min(self.min_separation, float(separations[np.triu_indices(len(positions), 1)].min()))
        extent, radius, width = self.config["map"]["approach_extent"], self.config["map"]["corner_radius"], self.config["lane_width"]
        hold_s = extent - self.settings["holding_distance"]
        for name, car in self.cars.items():
            position = self.states[name].pose.pose.position
            if car["phase"] != "resetting" and not inside_road((position.x, position.y), width + 0.03, extent, radius):
                self.fault = f"{name} left the local road activity region"
                for target in self.cars:
                    self.stop(target)
                return
            if car["phase"] == "active":
                points = self.routes[car["route"]]["center_line"]
                car["progress"] = max(car["progress"], project_route(np.array([position.x, position.y]), points))
                if car["progress"] >= hold_s - 0.025 and name != self.owner and name not in self.queue and car["progress"] < route_length(points) - hold_s:
                    self.queue.append(name)
                if name == self.owner and car["progress"] > route_length(points) - hold_s:
                    self.owner = None
        if self.owner is None and self.queue:
            self.owner = self.queue.pop(0)
        for name, car in self.cars.items():
            state = self.states[name]
            if car["phase"] == "resetting":
                self.stop(name)
                self.reset_car(name, car)
                continue
            if car["phase"] == "braking":
                self.stop(name)
                if math.hypot(state.twist.twist.linear.x, state.twist.twist.linear.y) < 0.025:
                    car["completed"] += 1
                    if self.settings.get("cycle_maneuvers"):
                        car["route"] = car["route"] // 3 * 3 + (car["route"] + 1) % 3
                    car["phase"] = "resetting"
                continue
            points = self.routes[car["route"]]["center_line"]
            length = route_length(points)
            exit_s = length - self.settings["exit_margin"]
            if car["progress"] >= exit_s - 0.025:
                car["phase"] = "braking"
                self.stop(name)
                continue
            limit = exit_s if name == self.owner or car["progress"] > length - hold_s else hold_s
            target_s = min(car["progress"] + 0.15, limit)
            if target_s < car["progress"] - 0.03:
                self.stop(name)
                continue
            x, y, heading = sample_route(points, max(0, target_s))
            delta = np.array([x - state.pose.pose.position.x, y - state.pose.pose.position.y])
            distance = float(np.linalg.norm(delta))
            velocity = delta * min(self.settings["speed"], 1.5 * distance) / max(distance, 1e-9)
            q = state.pose.pose.orientation
            current_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
            c, s = math.cos(current_yaw), math.sin(current_yaw)
            message = Twist()
            message.linear.x, message.linear.y = float(c * velocity[0] + s * velocity[1]), float(-s * velocity[0] + c * velocity[1])
            error = math.atan2(math.sin(heading - current_yaw), math.cos(heading - current_yaw))
            message.angular.z = max(-0.8, min(0.8, 2 * error))
            self.publishers_by_car[name].publish(message)


def main():
    rclpy.init()
    node = LocalTraffic()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for name in node.cars:
            node.stop(name)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
