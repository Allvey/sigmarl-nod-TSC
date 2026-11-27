"""ROS/Torch independent units, wire format and simulation guard checks."""
import json
import math

PROTOCOL = 1
MAX_PACKET = 1024 * 1024


def send_packet(stream, packet):
    data = (json.dumps(packet, allow_nan=False, separators=(",", ":")) + "\n").encode()
    if len(data) > MAX_PACKET:
        raise ValueError("Policy packet too large")
    stream.write(data)
    stream.flush()


def receive_packet(stream):
    data = stream.readline(MAX_PACKET + 1)
    if not data or len(data) > MAX_PACKET or not data.endswith(b"\n"):
        raise ConnectionError("Policy connection closed or packet exceeds limit")
    packet = json.loads(data, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(packet, dict) or packet.get("protocol") != PROTOCOL:
        raise ValueError("Policy protocol mismatch")
    return packet


def bicycle_twist(speed, steering, space_scale, time_scale, limits, speed_limit, lf=0.08, lr=0.08):
    """Training action at CG -> base Twist; preserve beta and curvature at limits."""
    if not all(math.isfinite(v) for v in (speed, steering, space_scale, time_scale, speed_limit)):
        raise ValueError("Nonfinite action or scaling")
    if min(space_scale, time_scale, speed_limit, lf, lr) <= 0:
        raise ValueError("Scales, speed limit and virtual wheelbase must be positive")
    beta = math.atan(lr / (lf + lr) * math.tan(steering))
    velocity = speed * space_scale / time_scale
    command = [velocity * math.cos(beta), velocity * math.sin(beta),
               speed * math.cos(beta) * math.tan(steering) / ((lf + lr) * time_scale)]
    factor = min(1.0, speed_limit / max(abs(velocity), 1e-12))
    for value, key in zip(command, ("max_speed", "max_lateral_speed", "max_yaw_rate")):
        factor = min(factor, limits[key] / max(abs(value), 1e-12))
    return [value * factor for value in command]


def previous_action(command, space_scale, time_scale, lf=0.08, lr=0.08):
    vx, vy, _ = command
    speed = math.copysign(math.hypot(vx, vy), vx) * time_scale / space_scale
    steering = math.atan((lf + lr) / lr * vy / vx) if abs(vx) > 1e-8 else 0.0
    return [speed, steering]


def footprint(state, length, width, padding=0):
    c, s = math.cos(state[2]), math.sin(state[2])
    return [(state[0] + c * dx - s * dy, state[1] + s * dx + c * dy)
            for dx, dy in ((length / 2 + padding, width / 2 + padding),
                           (-length / 2 - padding, width / 2 + padding),
                           (-length / 2 - padding, -width / 2 - padding),
                           (length / 2 + padding, -width / 2 - padding))]


def rectangles_overlap(a, b):
    for points in (a, b):
        for p, q in zip(points[:2], points[1:3]):
            axis = (p[1] - q[1], q[0] - p[0])
            pa = [x * axis[0] + y * axis[1] for x, y in a]
            pb = [x * axis[0] + y * axis[1] for x, y in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)
