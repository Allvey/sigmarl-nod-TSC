#!/usr/bin/python3
"""Bounded ROS2 actuator endpoint for the frozen host Actor/NOD worker."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
import time
import uuid

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_junction import inside_road, project_route, route_length
from policy_contract import (PROTOCOL, atomic_json, bicycle_twist, footprint,
                             receive_packet, rectangles_overlap, send_packet)


class PolicyBridge(Node):
    def __init__(self, args):
        super().__init__("sigmarl_policy_bridge")
        self.args = args
        metadata_path = ROOT / "generated/ros2/arena_metadata.json"
        self.config = json.loads(metadata_path.read_text())
        self.manifest = json.loads((ROOT / "runtime/ros2/policy_manifest.json").read_text())
        if hashlib.sha256(metadata_path.read_bytes()).hexdigest() != self.manifest["scene_sha256"]:
            raise ValueError("Inference worker and ROS2 scene configurations disagree")
        geometry = json.loads((ROOT / "generated/ros2/cpm_map.json").read_text())
        self.routes = geometry["routes"]["intersection"]
        self.names = [c["name"] for c in self.config["cars"]]
        self.controlled = set(args.cars or self.names)
        if not self.controlled <= set(self.names):
            raise ValueError("Unknown controlled car")
        self.session = uuid.uuid4().hex
        self.states, self.stamps, self.received, self.publishers_by_car = {}, {}, {}, {}
        self.commands = {name: [0.0, 0.0, 0.0] for name in self.names}
        self.completed, self.progress = set(), {}
        self.seq, self.frames, self.future, self.pending = 0, 0, None, None
        self.started_wall, self.start_stamp, self.last_request_stamp = time.monotonic(), None, None
        self.last_reply_wall = None
        self.fault, self.reason, self.exit_requested = None, None, False
        self.guard_details = None
        self.last_status, self.max_coordinate, self.min_separation = 0.0, 0.0, None
        self.max_command_speed = 0.0
        self.last_response = None
        self.io_pool = ThreadPoolExecutor(max_workers=1)
        self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.connection.settimeout(0.8)
        self.connection.connect(str(ROOT / "runtime/ros2/policy.sock"))
        self.stream = self.connection.makefile("rwb")
        self.trace = (ROOT / "runtime/ros2/policy_trace.jsonl").open("w", buffering=1)
        self.create_service(SetBool, "/policy_bridge/set_enabled", self.set_enabled)
        for name in self.names:
            self.create_subscription(Odometry, f"/{name}/ground_truth", lambda msg, name=name: self.receive(name, msg), qos_profile_sensor_data)
            self.publishers_by_car[name] = self.create_publisher(Twist, f"/{name}/{self.config['interfaces']['command_topic']}", 1)
        self.create_timer(0.05, self.tick)

    def receive(self, name, message):
        p, q, v = message.pose.pose.position, message.pose.pose.orientation, message.twist.twist
        heading = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        state = [p.x, p.y, heading, v.linear.x, v.linear.y, v.angular.z]
        if message.header.frame_id != "map" or message.child_frame_id != name + "/base_footprint" or not all(math.isfinite(value) for value in state):
            self.fail("Invalid odometry frame or nonfinite state: " + name)
            return
        self.states[name] = state
        self.stamps[name] = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        self.received[name] = time.monotonic()

    def publish(self, commands):
        if self.args.dry_run:
            return
        for name, command in commands.items():
            message = Twist()
            message.linear.x, message.linear.y, message.angular.z = map(float, command)
            self.publishers_by_car[name].publish(message)

    def stop(self):
        self.commands = {name: [0.0, 0.0, 0.0] for name in self.names}
        self.publish(self.commands)

    def fail(self, reason):
        self.fault = self.fault or reason
        self.reason, self.exit_requested = "fault", True
        self.stop()

    def set_enabled(self, request, response):
        response.success = not request.data
        response.message = "Policy stopped" if not request.data else "Start a new policy run with launch_jetauto.sh policy"
        if not request.data:
            self.reason, self.exit_requested = "operator_stop", True
            self.stop()
        return response

    def exchange(self, request):
        send_packet(self.stream, request)
        return receive_packet(self.stream)

    def guard(self):
        vehicle, layout = self.config["vehicle"], self.config["map"]
        for name, state in self.states.items():
            for point in footprint(state, vehicle["length"], vehicle["width"]):
                if not inside_road(point, self.config["lane_width"] + 0.025,
                                   layout["approach_extent"] + 0.025, layout["corner_radius"]):
                    self.guard_details = {"kind": "road_exit", "car": name,
                                          "position_m": state[:2], "yaw_rad": state[2],
                                          "violating_corner_m": list(point),
                                          "road_half_width_m": self.config["lane_width"], "tolerance_m": 0.025,
                                          "stamp": self.stamps[name]}
                    return name + " footprint left the local road"
        # Account for the existing acceleration-limited stopping transient.
        # This is a simulation stop guard, not a learned safety filter.
        horizon = max(math.hypot(*state[3:5]) / vehicle["max_acceleration"] + 0.1 for state in self.states.values())
        horizon = min(1.2, max(0.3, horizon))
        for step in range(7):
            elapsed = step * horizon / 6
            shapes = {}
            for name, state in self.states.items():
                command = self.commands[name]
                velocity = command if math.hypot(*command[:2]) > math.hypot(*state[3:5]) else state[3:6]
                x, y, angle = state[:3]
                c, s = math.cos(angle), math.sin(angle)
                predicted = [x + (c * velocity[0] - s * velocity[1]) * elapsed,
                             y + (s * velocity[0] + c * velocity[1]) * elapsed,
                             angle + velocity[2] * elapsed]
                shapes[name] = footprint(predicted, vehicle["length"], vehicle["width"], padding=0.01)
            for i, a in enumerate(self.names):
                for b in self.names[i + 1:]:
                    if rectangles_overlap(shapes[a], shapes[b]):
                        self.guard_details = {"kind": "predicted_contact", "cars": [a, b], "lookahead_s": elapsed}
                        return "Predicted vehicle contact: " + a + "/" + b
        return None

    def status(self, force=False):
        if not force and time.monotonic() - self.last_status < 0.25:
            return
        self.last_status = time.monotonic()
        data = {"enabled": not self.exit_requested, "dry_run": self.args.dry_run,
                "controller": "frozen Actor/NOD", "checkpoint": self.manifest["checkpoint"],
                "n_agents": len(self.names),
                "state_source": "Gazebo ground_truth", "frames": self.frames, "fault": self.fault,
                "guard_details": self.guard_details,
                "stop_reason": self.reason, "completed": sorted(self.completed),
                "controlled": sorted(self.controlled), "route_indices": self.manifest["route_indices"],
                "vehicle_routes": self.manifest["vehicle_routes"],
                "space_scale": self.manifest["space_scale"], "time_scale": self.manifest["time_scale"],
                "control_interval_sim_s": self.manifest["control_interval_sim_s"],
                "commands": self.commands, "states": self.states, "progress_m": self.progress,
                "max_abs_coordinate_m": self.max_coordinate, "min_pair_separation_m": self.min_separation,
                "max_command_speed_mps": self.max_command_speed,
                "stop_verified": len(self.states) == len(self.names) and all(
                    time.monotonic() - self.received[name] < 0.5 and
                    math.hypot(*s[3:5]) < 0.01 and abs(s[5]) < 0.02 for name, s in self.states.items()),
                "last_response": self.last_response}
        atomic_json(ROOT / "runtime/ros2/policy_status.json", data)

    def tick(self):
        self.status()
        if self.exit_requested:
            return
        now = time.monotonic()
        if len(self.states) != len(self.names):
            if now - self.started_wall > 10:
                self.fail("Timed out waiting for all configured odometry streams")
            return
        if any(now - self.received[name] > 0.5 for name in self.names):
            self.fail("Odometry is stale or Gazebo is paused")
            return
        stamp = min(self.stamps.values())
        if max(self.stamps.values()) - stamp > 0.11:
            self.fail("Odometry streams are not synchronized")
            return
        if self.start_stamp is None:
            self.start_stamp = stamp
        if self.last_request_stamp is not None and stamp < self.last_request_stamp - 0.1:
            self.fail("Simulation reset during policy run; start a new run")
            return
        if stamp - self.start_stamp >= self.args.duration:
            self.reason, self.exit_requested = "duration", True
            self.stop()
            return
        if now - self.started_wall > self.args.duration * 5 + 30:
            self.fail("Simulation is too slow to finish the bounded policy run")
            return
        positions = [self.states[name][:2] for name in self.names]
        self.max_coordinate = max(self.max_coordinate, max(abs(v) for point in positions for v in point))
        separation = min(math.dist(a, b) for i, a in enumerate(positions) for b in positions[i + 1:])
        self.min_separation = separation if self.min_separation is None else min(self.min_separation, separation)
        for name in self.controlled:
            route = self.routes[self.manifest["route_indices"][name]]["center_line"]
            self.progress[name] = project_route(self.states[name][:2], route)
            if self.progress[name] >= route_length(route) - self.config["traffic"]["exit_margin"] - 0.025:
                self.completed.add(name)
                self.commands[name] = [0.0, 0.0, 0.0]
        if self.future is not None and self.future.done():
            try:
                result = self.future.result()
                if (result["session"], result["seq"], result["stamp"]) != (self.session, self.pending["seq"], self.pending["stamp"]):
                    raise ValueError("Response does not match the pending state snapshot")
                if set(result["actions"]) != set(self.names) or now - self.pending["sent_wall"] > 0.6:
                    raise ValueError("Incomplete or stale inference response")
                if result["checkpoint"] != self.manifest["checkpoint"]:
                    raise ValueError("Inference checkpoint changed")
                for name, action in result["actions"].items():
                    if len(action) != 2 or not all(math.isfinite(value) for value in action) or abs(action[0]) > 1.0001 or abs(action[1]) > math.radians(35) + 1e-5:
                        raise ValueError("Invalid policy action")
                    self.commands[name] = (bicycle_twist(*action, self.manifest["space_scale"], self.manifest["time_scale"], self.config["vehicle"], self.args.speed_limit)
                                           if name in self.controlled - self.completed else [0.0, 0.0, 0.0])
                self.frames += 1
                self.last_reply_wall = now
                self.last_response = result
                self.trace.write(json.dumps({"request": self.pending, "response": result, "commands": self.commands}, allow_nan=False) + "\n")
                self.max_command_speed = max(self.max_command_speed, max(math.hypot(*v[:2]) for v in self.commands.values()))
            except Exception as error:
                self.fail("Inference failed: " + str(error))
                return
            self.future = None
        interval = self.manifest["control_interval_sim_s"]
        if self.last_reply_wall is not None and now - self.last_reply_wall > max(2.0, interval * 4):
            self.fail("Policy response watchdog expired")
            return
        reason = self.guard()
        if reason:
            self.fail(reason)
            return
        if self.completed == self.controlled:
            self.reason, self.exit_requested = "all_routes_complete", True
            self.stop()
            return
        if self.future is None and (self.last_request_stamp is None or stamp - self.last_request_stamp >= interval):
            self.seq += 1
            self.pending = {"protocol": PROTOCOL, "session": self.session, "seq": self.seq, "stamp": stamp,
                            "states": {name: state[:] for name, state in self.states.items()},
                            "previous_commands": {name: command[:] for name, command in self.commands.items()}, "sent_wall": now}
            self.last_request_stamp = stamp
            self.future = self.io_pool.submit(self.exchange, self.pending)
        if self.last_reply_wall is not None:
            self.publish(self.commands)

    def finish(self):
        self.exit_requested = True
        self.reason = self.reason or "process_exit"
        started, last_publish = time.monotonic(), 0.0
        while time.monotonic() - started < 10:
            now = time.monotonic()
            if now - last_publish >= 0.05:
                self.stop()
                last_publish = now
            rclpy.spin_once(self, timeout_sec=0.01)
            if now - started > 0.5 and len(self.states) == len(self.names) and all(
                    math.hypot(*s[3:5]) < 0.01 and abs(s[5]) < 0.02 for s in self.states.values()):
                break
        self.status(force=True)
        self.connection.close()
        self.io_pool.shutdown(wait=True)
        self.stream.close()
        self.trace.close()
        self.destroy_node()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60, help="Bounded simulation seconds")
    parser.add_argument("--speed-limit", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cars", nargs="+")
    args = parser.parse_args()
    if not (0 < args.duration <= 600 and 0 < args.speed_limit <= 0.3):
        parser.error("Duration must be in (0,600], speed limit in (0,0.3]")
    rclpy.init()
    node = PolicyBridge(args)
    try:
        while rclpy.ok() and not node.exit_requested:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish()
        if rclpy.ok():
            rclpy.shutdown()
    if node.fault:
        raise SystemExit(node.fault)


if __name__ == "__main__":
    main()
