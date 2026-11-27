"""A local right-hand-traffic junction: one incoming/outgoing lane per approach."""
import copy
import math

import numpy as np

from cpm_map import sample_route


def rotate(points, angle):
    c, s = math.cos(angle), math.sin(angle)
    return (np.asarray(points) @ np.array([[c, s], [-s, c]])).tolist()


def reference_routes(width, extent, radius):
    join = width + radius + 0.1
    routes = []
    for approach, angle in (("east", math.pi), ("north", -math.pi / 2), ("west", 0), ("south", math.pi / 2)):
        for maneuver in ("straight", "left", "right"):
            incoming = [[-extent, -width / 2], [-join, -width / 2]]
            if maneuver == "straight":
                connector = [[-join, -width / 2], [join, -width / 2]]
                outgoing = [[join, -width / 2], [extent, -width / 2]]
            elif maneuver == "left":
                r = join + width / 2
                theta = np.linspace(-math.pi / 2, 0, 49)
                connector = np.column_stack((-join + r * np.cos(theta), join + r * np.sin(theta))).tolist()
                outgoing = [[width / 2, join], [width / 2, extent]]
            else:
                r = join - width / 2
                theta = np.linspace(math.pi / 2, 0, 49)
                connector = np.column_stack((-join + r * np.cos(theta), -join + r * np.sin(theta))).tolist()
                outgoing = [[-width / 2, -join], [-width / 2, -extent]]
            points = rotate(incoming + connector[1:] + outgoing[1:], angle)
            routes.append({"index": len(routes), "approach": approach, "maneuver": maneuver,
                           "center_line": points, "lanelet_ids": []})
    return routes


def inside_road(point, width, extent, radius):
    x, y = abs(point[0]), abs(point[1])
    if x > extent or y > extent:
        return False
    if min(x, y) <= width:
        return True
    corner = width + radius
    return x <= corner and y <= corner and math.hypot(x - corner, y - corner) >= radius - 1e-9


def project_route(point, points):
    points = np.asarray(points)
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    fractions = np.clip(np.sum((point - points[:-1]) * vectors, axis=1) / np.maximum(lengths**2, 1e-12), 0, 1)
    projected = points[:-1] + vectors * fractions[:, None]
    i = int(np.argmin(np.sum((projected - point)**2, axis=1)))
    return float(lengths[:i].sum() + fractions[i] * lengths[i])


def route_length(points):
    return float(np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1).sum())


def select_route_indices(cars, routes, maneuver="configured"):
    """Keep each car's route unless a uniform maneuver override is explicit."""
    if maneuver not in ("configured", "straight", "left", "right"):
        raise ValueError("Unknown maneuver selection: " + str(maneuver))
    by_index = {route["index"]: route for route in routes}
    selected = []
    for car in cars:
        index = car["route_index"]
        if type(index) is not int or index not in by_index:
            raise ValueError("Invalid local route index for " + car["name"])
        route = by_index[index]
        if maneuver != "configured":
            route = next(candidate for candidate in routes
                         if candidate["approach"] == route["approach"] and candidate["maneuver"] == maneuver)
        selected.append(route["index"])
    return selected


def prepare(config):
    config = copy.deepcopy(config)
    width, extent, radius = config["lane_width"], config["map"]["approach_extent"], config["map"]["corner_radius"]
    if width <= config["vehicle"]["width"] or radius <= 0 or extent <= width + radius + 0.5:
        raise ValueError("Invalid two-lane junction dimensions")
    if min(config["arena_size"]) < 2 * extent + 0.5:
        raise ValueError("The local junction needs a border outside its approaches")
    traffic = config.get("traffic", {})
    hold = traffic.get("holding_distance", 1.35)
    if not width + radius + 0.3 <= hold < extent - 0.4:
        raise ValueError("Holding line must keep the whole vehicle outside the conflict area")
    routes = reference_routes(width, extent, radius)
    lanes = []
    for approach in range(4):
        for kind in ("incoming", "outgoing"):
            route = routes[approach * 3]["center_line"]
            line = route[:2] if kind == "incoming" else route[-2:]
            direction = np.diff(np.asarray(line), axis=0)[0]
            normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction) * width / 2
            lanes.append({"id": len(lanes) + 1, "kind": kind, "approach": routes[approach * 3]["approach"],
                          "center_line": line, "left_boundary": (np.array(line) + normal).tolist(),
                          "right_boundary": (np.array(line) - normal).tolist(), "left_marking": "dashed", "right_marking": "solid"})
    for route in routes:
        line = np.asarray(route["center_line"])[1:-1]
        tangent = np.gradient(line, axis=0)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        normal = normal / np.linalg.norm(normal, axis=1)[:, None] * width / 2
        connector_id = len(lanes) + 1
        lanes.append({"id": connector_id, "kind": "connector", "approach": route["approach"],
                      "center_line": line.tolist(), "left_boundary": (line + normal).tolist(),
                      "right_boundary": (line - normal).tolist(), "left_marking": None, "right_marking": None})
        approach = route["index"] // 3
        outgoing_index = min(range(4), key=lambda j: np.linalg.norm(np.array(lanes[j * 2 + 1]["center_line"][-1]) - route["center_line"][-1]))
        route["lanelet_ids"] = [approach * 2 + 1, connector_id, outgoing_index * 2 + 2]
        route["length"] = route_length(route["center_line"])
    by_id = {lane["id"]: lane for lane in lanes}
    for lane in lanes:
        lane.update({"successor": [], "predecessor": []})
    for route in routes:
        for a, b in zip(route["lanelet_ids"], route["lanelet_ids"][1:]):
            if b not in by_id[a]["successor"]:
                by_id[a]["successor"].append(b)
                by_id[b]["predecessor"].append(a)
    # Paint is independent of the virtual lanelets used for path planning.
    triangles, markings = [], []
    for rectangle in (((-extent, -width), (extent, -width), (extent, width), (-extent, width)),
                      ((-width, -extent), (width, -extent), (width, extent), (-width, extent))):
        a, b, c, d = rectangle
        triangles.extend([[a, b, c], [a, c, d]])
    for sx in (-1, 1):
        for sy in (-1, 1):
            theta = np.linspace(math.pi, 1.5 * math.pi, 33)
            arc = np.column_stack((width + radius + radius * np.cos(theta), width + radius + radius * np.sin(theta))) * [sx, sy]
            anchor = np.array([sx * width, sy * width])
            triangles.extend([[anchor.tolist(), a.tolist(), b.tolist()] for a, b in zip(arc, arc[1:])])
            markings.append({"style": "solid", "points": arc.tolist()})
            markings.append({"style": "solid", "points": [[sx * extent, sy * width], [sx * (width + radius), sy * width]]})
            markings.append({"style": "solid", "points": [[sx * width, sy * extent], [sx * width, sy * (width + radius)]]})
    for axis in (0, 1):
        for sign in (-1, 1):
            line = [[sign * (width + radius + 0.1), 0], [sign * extent, 0]]
            if axis:
                line = [p[::-1] for p in line]
            markings.append({"style": "dashed", "points": line})
    config["map"].update({"layout": "local_two_lane_junction", "lanes_per_direction": 1,
                           "derived_from": "CPM_mixed intersection experiment", "activity_bounds": [-extent, extent, -extent, extent],
                           "map_size": [2 * extent, 2 * extent]})
    for car in config["cars"]:
        route = routes[car["route_index"]]
        car["approach"], car["maneuver"] = route["approach"], route["maneuver"]
        car["route_group"] = "intersection"
        car["route_lanelet_ids"] = route["lanelet_ids"]
        car["x"], car["y"], car["yaw"] = sample_route(route["center_line"], car["spawn_distance"])
    data = {"scenario": config["scenario"], "geometry_kind": "simplified_local_junction", "lanes_per_direction": 1,
            "map_size": [2 * extent, 2 * extent], "lanelets": lanes, "routes": {"intersection": routes},
            "road_triangles": triangles, "markings": markings, "activity_bounds": config["map"]["activity_bounds"]}
    return config, data
