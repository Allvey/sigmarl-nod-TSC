"""Read canonical CPM geometry/routes without importing the training runtime."""
import ast
import copy
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def project_root():
    local = Path(__file__).resolve().parent
    for candidate in (local, *local.parents):
        if (candidate / "assets/maps/cpm.xml").is_file():
            return candidate
    if Path("/project/assets/maps/cpm.xml").is_file():
        return Path("/project")
    raise FileNotFoundError("Mount the repository's assets/maps and utilities into /project")


def definitions(root):
    constants = ast.parse((root / "utilities/constants.py").read_text())
    scenario = None
    for node in constants.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SCENARIOS" for t in node.targets):
            for key, value in zip(node.value.keys, node.value.values):
                if ast.literal_eval(key) == "CPM_mixed":
                    scenario = ast.literal_eval(value)
    routes = {}
    for node in ast.walk(ast.parse((root / "utilities/parse_xml.py").read_text())):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else None
                if name in ("path_intersection", "path_merge_in", "path_merge_out", "path_to_loop", "_reference_paths_ids"):
                    routes[name] = ast.literal_eval(node.value)
    if scenario is None or len(routes) != 5:
        raise ValueError("Canonical CPM scenario or route definitions changed; review the importer")
    return scenario, routes


def read_map(scale=3.0, root=None):
    root = project_root() if root is None else Path(root)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("CPM scale must be positive")
    scenario, routing = definitions(root)
    center = np.array([(scenario["x_dim_min"] + scenario["x_dim_max"]) / 2,
                       (scenario["y_dim_min"] + scenario["y_dim_max"]) / 2])
    source = root / scenario["map_path"]
    lanes = []
    for lane in ET.parse(source).getroot().findall("lanelet"):
        record = {"id": int(lane.get("id")), "type": lane.findtext("laneletType") or lane.findtext("lanelet_type"),
                  "predecessor": [int(n.get("ref")) for n in lane.findall("predecessor")],
                  "successor": [int(n.get("ref")) for n in lane.findall("successor")]}
        for side in ("left", "right"):
            bound = lane.find(side + "Bound")
            points = np.array([[float(p.findtext("x")), float(p.findtext("y"))] for p in bound.findall("point")])
            record[side + "_boundary"] = ((points - center) * scale).tolist()
            record[side + "_marking"] = bound.findtext("lineMarking")
        if len(record["left_boundary"]) != len(record["right_boundary"]):
            raise ValueError("CPM boundary sample counts do not match")
        record["center_line"] = ((np.array(record["left_boundary"]) + np.array(record["right_boundary"])) / 2).tolist()
        for side in ("Left", "Right"):
            adjacent = lane.find("adjacent" + side)
            record["adjacent_" + side.lower()] = None if adjacent is None else {
                "id": int(adjacent.get("ref")), "driving_direction": adjacent.get("drivingDir")}
        lanes.append(record)
    by_id = {lane["id"]: lane for lane in lanes}

    def route(ids, index):
        points = []
        for lane_id in ids:
            line = by_id[lane_id]["center_line"]
            if points and np.linalg.norm(np.array(points[-1]) - line[0]) < 1e-4 * scale:
                points.extend(line[1:])
            else:
                points.extend(line)
        return {"index": index, "lanelet_ids": ids, "center_line": points}

    groups = {group: [route(ids, i) for i, ids in enumerate(routing[variable])]
              for group, variable in (("intersection", "path_intersection"), ("merge_in", "path_merge_in"), ("merge_out", "path_merge_out"))}
    groups["loop"] = []
    for index, (loop_id, start_lane) in sorted(routing["path_to_loop"].items()):
        ids = routing["_reference_paths_ids"][loop_id - 1]
        start = ids.index(start_lane)
        groups["loop"].append(route(ids[start:] + ids[:start], index - 1))
    return {"scenario": "CPM_mixed", "source": scenario["map_path"], "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_lane_width": scenario["lane_width"], "source_n_agents": scenario["n_agents"],
            "scale": scale, "source_center": center.tolist(),
            "source_to_map": [[scale, 0, -scale * center[0]], [0, scale, -scale * center[1]], [0, 0, 1]],
            "map_size": [(scenario["x_dim_max"] - scenario["x_dim_min"]) * scale,
                         (scenario["y_dim_max"] - scenario["y_dim_min"]) * scale],
            "lanelets": lanes, "routes": groups}


def sample_route(points, distance):
    if distance < 0:
        raise ValueError("Spawn distance must be non-negative")
    for first, second in zip(points, points[1:]):
        vector = np.array(second) - first
        length = float(np.linalg.norm(vector))
        if length < 1e-9:
            continue
        if distance <= length:
            point = np.array(first) + vector * distance / length
            return float(point[0]), float(point[1]), math.atan2(vector[1], vector[0])
        distance -= length
    raise ValueError("Spawn distance exceeds assigned reference route length")


def prepare(config):
    config = copy.deepcopy(config)
    data = read_map(config["map"]["scale"])
    padding = config["map"]["padding"]
    if padding < 0.5:
        raise ValueError("CPM corner tags require at least 0.5 m of padding")
    config["arena_size"] = [size + 2 * padding for size in data["map_size"]]
    config["lane_width"] = data["source_lane_width"] * data["scale"]
    config["map"].update({"source": data["source"], "source_sha256": data["source_sha256"],
                          "map_size": data["map_size"], "source_to_map": data["source_to_map"]})
    tx, ty = [size / 2 - padding / 2 for size in config["arena_size"]]
    config["ground_tags"] = [{"id": i, "x": x, "y": y, "size": 0.3}
                             for i, (x, y) in enumerate(((-tx, -ty), (tx, -ty), (tx, ty), (-tx, ty)))]
    if len(config["cars"]) != data["source_n_agents"]:
        raise ValueError("CPM_mixed must use the configured canonical agent count")
    for car in config["cars"]:
        route = data["routes"][car["route_group"]][car["route_index"]]
        car["x"], car["y"], car["yaw"] = sample_route(route["center_line"], car["spawn_distance"])
        car["route_lanelet_ids"] = route["lanelet_ids"]
    return config, data


def dashed_segments(points, dash, gap):
    position = 0.0
    period = dash + gap
    for a, b in zip(points, points[1:]):
        a, b = np.array(a), np.array(b)
        length = float(np.linalg.norm(b - a))
        if length < 1e-9:
            continue
        offset = 0.0
        while offset < length - 1e-9:
            phase = (position + offset) % period
            if phase < 1e-8 or period - phase < 1e-8:
                phase = 0.0
            elif abs(phase - dash) < 1e-8:
                phase = dash
            active = phase < dash - 1e-9
            remaining = (dash if active else period) - phase
            if remaining < 1e-9:
                remaining = gap if not active else dash
            end = min(length, offset + remaining)
            if active:
                yield a + (b - a) * offset / length, a + (b - a) * end / length
            offset = end
        position += length


def write_mesh(data, options, path):
    """One coloured Collada mesh for exact lane ribbons and line markings."""
    vertices, groups = [], {"road": [], "solid": [], "dashed": []}

    def triangle(a, b, c, z, group):
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cross) < 1e-12:
            return
        if cross < 0:
            b, c = c, b
        for p in (a, b, c):
            groups[group].append(len(vertices))
            vertices.append((float(p[0]), float(p[1]), z))

    def stripe(a, b, group):
        delta = np.array(b) - a
        length = np.linalg.norm(delta)
        if length < 1e-9:
            return
        normal = np.array([-delta[1], delta[0]]) / length * options["marking_width"] / 2
        a, b = np.array(a), np.array(b)
        triangle(a + normal, a - normal, b - normal, 0.002, group)
        triangle(a + normal, b - normal, b + normal, 0.002, group)

    markings = data.get("markings")
    if "road_triangles" in data:
        for a, b, c in data["road_triangles"]:
            triangle(a, b, c, 0.001, "road")
    else:
        for lane in data["lanelets"]:
            left, right = lane["left_boundary"], lane["right_boundary"]
            for i in range(len(left) - 1):
                triangle(left[i], right[i], right[i + 1], 0.001, "road")
                triangle(left[i], right[i + 1], left[i + 1], 0.001, "road")
    if markings is None:
        markings, seen = [], set()
        for lane in data["lanelets"]:
            for side in ("left", "right"):
                points, marking = lane[side + "_boundary"], lane[side + "_marking"]
                key = (marking, tuple(tuple(round(c, 7) for c in p) for p in points))
                if key in seen or (marking, key[1][::-1]) in seen:
                    continue
                seen.add(key)
                markings.append({"points": points, "style": marking})
    for marking in markings:
        points = marking["points"]
        if marking["style"] == "solid":
            for a, b in zip(points, points[1:]):
                stripe(a, b, "solid")
        elif marking["style"] == "dashed":
            for a, b in dashed_segments(points, options["dash_length"], options["dash_gap"]):
                stripe(a, b, "dashed")
    collada = ET.Element("COLLADA", xmlns="http://www.collada.org/2005/11/COLLADASchema", version="1.4.1")
    asset = ET.SubElement(collada, "asset")
    ET.SubElement(asset, "unit", name="meter", meter="1")
    ET.SubElement(asset, "up_axis").text = "Z_UP"
    effects, materials = ET.SubElement(collada, "library_effects"), ET.SubElement(collada, "library_materials")
    colors = {"road": "0.18 0.18 0.20 1", "solid": "0.95 0.95 0.95 1", "dashed": "1 0.85 0.05 1"}
    for name, color in colors.items():
        effect = ET.SubElement(effects, "effect", id=name + "_effect")
        technique = ET.SubElement(ET.SubElement(effect, "profile_COMMON"), "technique", sid="common")
        ET.SubElement(ET.SubElement(ET.SubElement(technique, "lambert"), "diffuse"), "color").text = color
        ET.SubElement(ET.SubElement(materials, "material", id=name), "instance_effect", url="#" + name + "_effect")
    mesh = ET.SubElement(ET.SubElement(ET.SubElement(collada, "library_geometries"), "geometry", id="cpm_geometry"), "mesh")
    for name, values, count, params in (("positions", [x for v in vertices for x in v], len(vertices), ("X", "Y", "Z")),
                                        ("normals", [0, 0, 1], 1, ("X", "Y", "Z"))):
        source = ET.SubElement(mesh, "source", id=name)
        ET.SubElement(source, "float_array", id=name + "_array", count=str(len(values))).text = " ".join(f"{v:.10g}" for v in values)
        accessor = ET.SubElement(ET.SubElement(source, "technique_common"), "accessor", source="#" + name + "_array", count=str(count), stride="3")
        for param in params:
            ET.SubElement(accessor, "param", name=param, type="float")
    ET.SubElement(ET.SubElement(mesh, "vertices", id="vertices"), "input", semantic="POSITION", source="#positions")
    for name, indices in groups.items():
        triangles = ET.SubElement(mesh, "triangles", material=name, count=str(len(indices) // 3))
        ET.SubElement(triangles, "input", semantic="VERTEX", source="#vertices", offset="0")
        ET.SubElement(triangles, "input", semantic="NORMAL", source="#normals", offset="1")
        ET.SubElement(triangles, "p").text = " ".join(f"{i} 0" for i in indices)
    scene = ET.SubElement(ET.SubElement(collada, "library_visual_scenes"), "visual_scene", id="cpm_scene")
    instance = ET.SubElement(ET.SubElement(scene, "node", id="cpm_node"), "instance_geometry", url="#cpm_geometry")
    bindings = ET.SubElement(ET.SubElement(instance, "bind_material"), "technique_common")
    for name in groups:
        ET.SubElement(bindings, "instance_material", symbol=name, target="#" + name)
    ET.SubElement(ET.SubElement(collada, "scene"), "instance_visual_scene", url="#cpm_scene")
    ET.ElementTree(collada).write(str(path), encoding="utf-8", xml_declaration=True)
    return {name: len(indices) // 3 for name, indices in groups.items()}
