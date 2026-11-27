"""Unit/geometry contracts for the ROS2 policy actuator path."""
import io
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_contract import (PROTOCOL, bicycle_twist, footprint, previous_action,
                             receive_packet, rectangles_overlap, send_packet)

LIMITS = {"max_speed": 0.4, "max_lateral_speed": 0.3, "max_yaw_rate": 1.2}


class PolicyContractTest(unittest.TestCase):
    def test_bicycle_cg_derivative_and_spatial_time_similarity(self):
        for speed in (-1, 0, 0.7, 1):
            for delta in (-0.61, 0, 0.61):
                vx, vy, wz = bicycle_twist(speed, delta, 3, 15, LIMITS, 0.2)
                beta = math.atan(0.5 * math.tan(delta))
                self.assertAlmostEqual(vx, 0.2 * speed * math.cos(beta))
                self.assertAlmostEqual(vy, 0.2 * speed * math.sin(beta))
                self.assertAlmostEqual(wz, speed * math.cos(beta) * math.tan(delta) / (0.16 * 15))
                actual_speed, actual_delta = previous_action([vx, vy, wz], 3, 15)
                self.assertAlmostEqual(actual_speed, speed)
                if speed:
                    self.assertAlmostEqual(actual_delta, delta)

    def test_limits_scale_the_whole_twist_to_preserve_curvature(self):
        large = bicycle_twist(1, 0.61, 3, 1, LIMITS, 0.2)
        reference = bicycle_twist(1, 0.61, 3, 15, LIMITS, 0.2)
        self.assertLessEqual(math.hypot(*large[:2]), 0.2 + 1e-12)
        self.assertAlmostEqual(large[1] / large[0], reference[1] / reference[0])
        self.assertAlmostEqual(large[2] / large[0], reference[2] / reference[0])
        for action in ((float('nan'), 0), (1, float('inf'))):
            with self.assertRaises(ValueError):
                bicycle_twist(*action, 3, 15, LIMITS, 0.2)

    def test_rotated_rectangle_contacts_and_adjacent_lanes(self):
        a = footprint([0, 0, 0], 0.302, 0.26)
        self.assertTrue(rectangles_overlap(a, footprint([0.1, 0, math.pi / 4], 0.302, 0.26)))
        self.assertFalse(rectangles_overlap(a, footprint([0, 0.45, math.pi], 0.302, 0.26)))
        self.assertFalse(rectangles_overlap(a, footprint([0.5, 0.5, math.pi / 4], 0.302, 0.26)))

    def test_wire_protocol_rejects_stale_schema_nan_and_truncation(self):
        stream = io.BytesIO()
        packet = {"protocol": PROTOCOL, "seq": 2, "states": {"car_0": [1, 2, 3]}}
        send_packet(stream, packet)
        stream.seek(0)
        self.assertEqual(receive_packet(stream), packet)
        for data in (b'{"protocol":0}\n', b'{"protocol":1,"x":NaN}\n', b'{"protocol":1}'):
            with self.assertRaises((ValueError, ConnectionError)):
                receive_packet(io.BytesIO(data))


if __name__ == "__main__":
    unittest.main()
