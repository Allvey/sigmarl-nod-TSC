#!/usr/bin/python3
"""Send bounded speed/steering commands; always send stop on exit."""
import argparse
import math
import time

import rospy
from geometry_msgs.msg import Vector3Stamped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--car", default="car_0", choices=("car_0", "car_1"))
    parser.add_argument("--speed", type=float, default=0.15, help="m/s, signed")
    parser.add_argument("--steering", type=float, default=0, help="radians, left positive")
    parser.add_argument("--duration", type=float, default=2, help="wall-clock seconds")
    args = parser.parse_args()
    if not all(math.isfinite(x) for x in (args.speed, args.steering, args.duration)) or args.duration <= 0:
        parser.error("Arguments must be finite and duration positive")
    rospy.init_node("sigmarl_manual_drive", anonymous=True)
    pub = rospy.Publisher(f"/{args.car}/drive", Vector3Stamped, queue_size=1)
    deadline = time.monotonic() + 5
    while not rospy.is_shutdown() and pub.get_num_connections() == 0:
        if time.monotonic() > deadline:
            raise RuntimeError("Vehicle controller not connected; start the arena first")
        time.sleep(0.05)
    message = Vector3Stamped()
    message.header.frame_id = args.car + "/base_link"
    message.vector.x = args.speed
    message.vector.y = args.steering
    try:
        deadline = time.monotonic() + args.duration
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            message.header.stamp = rospy.Time.now()
            pub.publish(message)
            time.sleep(0.05)
    finally:
        message.vector.x = message.vector.y = 0
        for _ in range(5):
            if rospy.is_shutdown():
                break
            message.header.stamp = rospy.Time.now()
            pub.publish(message)
            time.sleep(0.05)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
