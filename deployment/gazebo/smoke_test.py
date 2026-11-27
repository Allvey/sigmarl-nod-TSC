#!/usr/bin/python3
"""Exercise physical driving, left steering and the command-loss watchdog."""
import json
import math
import threading
import time

import rospy
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry


def yaw(message):
    q = message.pose.pose.orientation
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def main():
    rospy.init_node("sigmarl_smoke_test", anonymous=True)
    lock, ready = threading.Lock(), threading.Event()
    state = [None]

    def receive(message):
        with lock:
            state[0] = message
        ready.set()

    subscriber = rospy.Subscriber("/car_0/ground_truth", Odometry, receive, queue_size=1)
    pub = rospy.Publisher("/car_0/drive", Vector3Stamped, queue_size=1)
    if not ready.wait(30):
        raise RuntimeError("No ground truth; start an unpaused arena")

    def snapshot():
        with lock:
            return state[0]

    def phase(seconds, speed=None, steering=0):
        start, deadline = snapshot().header.stamp.to_sec(), time.monotonic() + 45
        while not rospy.is_shutdown():
            current = snapshot()
            if current.header.stamp.to_sec() - start >= seconds:
                return current
            if time.monotonic() > deadline:
                raise RuntimeError("Simulation clock stopped or simulation is too slow")
            if speed is not None:
                message = Vector3Stamped()
                message.header.stamp = rospy.Time.now()
                message.vector.x, message.vector.y = speed, steering
                pub.publish(message)
            time.sleep(0.025)
        raise RuntimeError("ROS shut down during smoke test")

    try:
        initial = phase(0.5, 0)
        straight = phase(2, 0.15)
        dx = straight.pose.pose.position.x - initial.pose.pose.position.x
        dy = straight.pose.pose.position.y - initial.pose.pose.position.y
        heading = yaw(initial)
        forward = dx * math.cos(heading) + dy * math.sin(heading)
        lateral = -dx * math.sin(heading) + dy * math.cos(heading)
        stopped = phase(1.5)  # Deliberately stop publishing: exercise the watchdog.
        stopped_speed = abs(stopped.twist.twist.linear.x)
        turned = phase(2, 0.15, 0.3)
        angle = math.atan2(math.sin(yaw(turned) - yaw(stopped)), math.cos(yaw(turned) - yaw(stopped)))
        phase(0.8, 0)
        result = {"straight_forward_m": forward, "straight_lateral_m": lateral,
                  "watchdog_stopped_speed_mps": stopped_speed, "left_turn_radians": angle,
                  "vehicle_height_m": turned.pose.pose.position.z}
        print(json.dumps(result, indent=2))
        failures = []
        if forward < 0.15 or abs(lateral) > 0.06:
            failures.append("straight wheel-driven motion")
        if stopped_speed > 0.025:
            failures.append("command-loss stop")
        if angle < 0.15:
            failures.append("positive steering turns left")
        if abs(turned.pose.pose.position.z) > 0.03:
            failures.append("stable ground contact")
        if failures:
            raise SystemExit("Failed: " + ", ".join(failures))
        print("Physical drive smoke test passed")
    finally:
        message = Vector3Stamped()
        for _ in range(5):
            pub.publish(message)
            time.sleep(0.05)
        subscriber.unregister()


if __name__ == "__main__":
    main()
