#!/usr/bin/python3
"""Check a rendered ROS camera frame for all configured tags and save a preview."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from tag_detection import detect_tags

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=40)
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/camera_tags.png")
    args = parser.parse_args()
    metadata = json.loads((ROOT / "generated/arena_metadata.json").read_text())
    expected = {tag["id"] for tag in metadata["ground_tags"]} | {car["tag_id"] for car in metadata["cars"]}
    rospy.init_node("sigmarl_camera_check", anonymous=True)
    info = rospy.wait_for_message("/overhead_camera/camera_info", CameraInfo, timeout=args.timeout)
    message = rospy.wait_for_message("/overhead_camera/image_raw", Image, timeout=args.timeout)
    image = CvBridge().imgmsg_to_cv2(message, desired_encoding="bgr8")
    corners, ids = detect_tags(image)
    detected = set() if ids is None else set(ids.flatten().tolist())
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), image):
        raise RuntimeError("Could not save camera image")
    if message.header.frame_id != "overhead_camera_optical" or info.header.frame_id != message.header.frame_id:
        raise RuntimeError("Unexpected camera frame")
    if (info.width, info.height) != (message.width, message.height) or not np.isfinite(info.K).all():
        raise RuntimeError("Invalid camera intrinsics")
    print(json.dumps({"image": str(args.output), "resolution": [message.width, message.height],
                      "detected_ids": sorted(detected), "missing_ids": sorted(expected - detected),
                      "camera_matrix": list(info.K)}, indent=2))
    if expected - detected:
        raise SystemExit("Some tags were not detected; inspect the saved frame")


if __name__ == "__main__":
    main()
