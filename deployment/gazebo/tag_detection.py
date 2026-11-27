"""ROS-independent multi-scale detection for ground and vehicle AprilTags."""
import cv2
import numpy as np


def detect_tags(image):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = (cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters")
                  else cv2.aruco.DetectorParameters_create())
    # Use AprilTag's quad extraction for small roof markers next to dark CAD parts.
    # OpenCV's default ArUco candidate extraction misses some otherwise legible tags.
    if hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
        # Legacy OpenCV's parallel AprilTag extraction can deadlock on small
        # downscaled images. The check tool processes one frame at a time.
        cv2.setNumThreads(1)
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    found = {}
    for scale in (1.0, 0.5):
        scaled = image if scale == 1 else cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if hasattr(cv2.aruco, "ArucoDetector"):
            corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(scaled)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(scaled, dictionary, parameters=parameters)
        if ids is not None:
            for marker, tag_id in zip(corners, ids.flatten()):
                found.setdefault(int(tag_id), marker / scale)
    ids = np.array(sorted(found), dtype=np.int32).reshape(-1, 1)
    return [found[int(tag)] for tag in ids.flatten()], ids if len(ids) else None
