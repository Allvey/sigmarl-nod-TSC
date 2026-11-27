#!/usr/bin/python3
"""Generate a self-contained Gazebo Classic world, including real AprilTag cells."""
import argparse
import copy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
from configuration import load_config

ROOT = Path(__file__).resolve().parent


def element(parent, tag, value=None, **attributes):
    node = ET.SubElement(parent, tag, attributes)
    if value is not None:
        node.text = str(value)
    return node


def numbers(values):
    return " ".join(f"{v:.10g}" for v in values)


def box(link, name, size, pose, color, collision=False):
    visual = element(link, "visual", name=name)
    element(visual, "pose", numbers(pose))
    element(element(element(visual, "geometry"), "box"), "size", numbers(size))
    material = element(visual, "material")
    element(material, "ambient", numbers(color))
    element(material, "diffuse", numbers(color))
    element(visual, "cast_shadows", "false")
    if collision:
        contact = element(link, "collision", name=name)
        element(contact, "pose", numbers(pose))
        element(element(element(contact, "geometry"), "box"), "size", numbers(size))
    return visual


def inertia(link, mass, diagonal, z=0):
    inertial = element(link, "inertial")
    element(inertial, "pose", f"0 0 {z} 0 0 0")
    element(inertial, "mass", mass)
    tensor = element(inertial, "inertia")
    for key, value in zip(("ixx", "iyy", "izz"), diagonal):
        element(tensor, key, value)
    for key in ("ixy", "ixz", "iyz"):
        element(tensor, key, 0)


def static_model(world, name):
    model = element(world, "model", name=name)
    element(model, "static", "true")
    return element(model, "link", name="base_link")


def tag_cells(tag_id):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, tag_id, 8)
    return cv2.aruco.drawMarker(dictionary, tag_id, 8)


def add_tag(link, tag_id, x, y, z, size):
    # size measures the BLACK border, excluding the one-cell white quiet zone.
    # Bitmap top points toward local +Y, bitmap right toward local +X.
    cells = tag_cells(tag_id)
    cell = size / len(cells)
    box(link, f"tag_{tag_id}_white", (size + 2 * cell, size + 2 * cell, 0.0002),
        (x, y, z - 0.0001, 0, 0, 0), (1, 1, 1, 1))
    for row, pixels in enumerate(cells):
        col = 0
        while col < len(pixels):
            if pixels[col] != 0:
                col += 1
                continue
            start = col
            while col < len(pixels) and pixels[col] == 0:
                col += 1
            box(link, f"tag_{tag_id}_{row}_{start}", ((col - start) * cell, cell, 0.0001),
                (x + ((start + col) / 2 - 4) * cell,
                 y + (3.5 - row) * cell, z + 0.00005, 0, 0, 0), (0, 0, 0, 1))


def add_joint(model, name, parent, child, axis, limit=None):
    joint = element(model, "joint", name=name, type="revolute")
    element(joint, "parent", parent)
    element(joint, "child", child)
    joint_axis = element(joint, "axis")
    element(joint_axis, "xyz", numbers(axis))
    element(joint_axis, "use_parent_model_frame", "true")
    bounds = element(joint_axis, "limit")
    element(bounds, "lower", -limit if limit else -1e16)
    element(bounds, "upper", limit if limit else 1e16)
    element(bounds, "effort", 0.3)
    element(bounds, "velocity", 100)
    element(element(joint_axis, "dynamics"), "damping", 0.00001)


def add_car(world, car, vehicle):
    model = element(world, "model", name=car["name"])
    element(model, "pose", numbers((car["x"], car["y"], 0.002, 0, 0, car["yaw"])))
    link = element(model, "link", name="base_link")
    length, width = vehicle["length"], vehicle["width"]
    mass, body_height, body_z = 0.25, 0.035, 0.0525
    inertia(link, mass, (mass * (width**2 + body_height**2) / 12,
                         mass * (length**2 + body_height**2) / 12,
                         mass * (length**2 + width**2) / 12), body_z)
    box(link, "body", (length, width, body_height), (0, 0, body_z, 0, 0, 0), car["color"], True)
    add_tag(link, car["tag_id"], 0, 0, vehicle["tag_height"], vehicle["tag_size"])
    for front in (False, True):
        for left in (False, True):
            label = ("front" if front else "rear") + ("_left" if left else "_right")
            wheel_name = label + "_wheel"
            x = vehicle["wheelbase"] / 2 * (1 if front else -1)
            y = vehicle["track"] / 2 * (1 if left else -1)
            pose = numbers((x, y, vehicle["wheel_radius"], 0, 0, 0))
            parent = "base_link"
            if front:
                parent = label + "_knuckle"
                knuckle = element(model, "link", name=parent)
                element(knuckle, "pose", pose)
                inertia(knuckle, 0.003, (1e-6, 1e-6, 1e-6))
                add_joint(model, label + "_steer", "base_link", parent, (0, 0, 1), 0.95)
            wheel = element(model, "link", name=wheel_name)
            element(wheel, "pose", pose)
            radius, wheel_width, wheel_mass = vehicle["wheel_radius"], vehicle["wheel_width"], 0.015
            i_axial = wheel_mass * radius**2 / 2
            i_transverse = wheel_mass * (3 * radius**2 + wheel_width**2) / 12
            inertia(wheel, wheel_mass, (i_transverse, i_axial, i_transverse))
            for kind in ("visual", "collision"):
                shape = element(wheel, kind, name=wheel_name)
                element(shape, "pose", f"0 0 0 {math.pi / 2} 0 0")
                cylinder = element(element(shape, "geometry"), "cylinder")
                element(cylinder, "radius", radius)
                element(cylinder, "length", wheel_width)
                if kind == "visual":
                    material = element(shape, "material")
                    element(material, "ambient", "0.04 0.04 0.04 1")
                    element(material, "diffuse", "0.04 0.04 0.04 1")
                else:
                    ode = element(element(element(shape, "surface"), "friction"), "ode")
                    element(ode, "mu", 1.0)
                    element(ode, "mu2", 1.0)
            add_joint(model, label + "_axle", parent, wheel_name, (0, 1, 0))
    plugin = element(model, "plugin", name="sigmarl_drive", filename="libsigmarl_drive.so")
    element(plugin, "namespace", car["name"])
    for key in ("wheelbase", "track", "wheel_radius", "max_speed", "max_steering",
                "max_acceleration", "max_steering_rate", "command_timeout"):
        element(plugin, key, vehicle[key])


def validate(config):
    width, height = config["arena_size"]
    camera, vehicle = config["camera"], config["vehicle"]
    if min(width, height, config["lane_width"], camera["height"], camera["width"],
           camera["height_pixels"], camera["fps"], *vehicle.values()) <= 0:
        raise ValueError("Dimensions, limits and camera settings must be positive")
    if not 0 < camera["horizontal_fov"] < math.pi:
        raise ValueError("Camera field of view must be between 0 and pi")
    footprint_x = 2 * camera["height"] * math.tan(camera["horizontal_fov"] / 2)
    footprint_y = footprint_x * camera["height_pixels"] / camera["width"]
    if footprint_x < width or footprint_y < height:
        raise ValueError("Overhead camera does not cover the whole arena")
    tags = [t["id"] for t in config["ground_tags"]] + [c["tag_id"] for c in config["cars"]]
    if len(tags) != len(set(tags)) or any(not 0 <= tag < 587 for tag in tags):
        raise ValueError("AprilTag 36h11 IDs must be unique and in [0, 586]")
    if len({car["name"] for car in config["cars"]}) != len(config["cars"]):
        raise ValueError("Vehicle names must be unique")
    if config["lane_width"] <= vehicle["width"]:
        raise ValueError("Lane width must exceed the vehicle footprint width")
    if config.get("robot_type") == "jetauto":
        if config.get("ros_version") != 2:
            raise ValueError("The JetAuto profile requires the ROS2 launcher")
        lidar = config["lidar"]
        if lidar["samples"] < 2 or lidar["fps"] <= 0 or not 0 < lidar["range_min"] < lidar["range_max"]:
            raise ValueError("Invalid A1 simulation scan settings")


def add_crossroad(floor, sx, sy, lane):
    for name, size in (("east_west", (sx, lane * 2, 0.0002)), ("north_south", (lane * 2, sy, 0.0002))):
        box(floor, name, size, (0, 0, 0.0001, 0, 0, 0), (0.18, 0.18, 0.20, 1))
    for axis, span in ((0, sx), (1, sy)):
        segment = span / 2 - lane
        for end in (-1, 1):
            center = end * (lane + segment / 2)
            for side in (-1, 1):
                pose = (center, side * lane, 0.0004, 0, 0, 0) if axis == 0 else (side * lane, center, 0.0004, 0, 0, 0)
                size = (segment, 0.012, 0.0002) if axis == 0 else (0.012, segment, 0.0002)
                box(floor, f"edge_{axis}_{end}_{side}", size, pose, (0.95, 0.95, 0.95, 1))
            cursor, index = lane + 0.06, 0
            while cursor + 0.08 < span / 2:
                pose = (end * cursor, 0, 0.0004, 0, 0, 0) if axis == 0 else (0, end * cursor, 0.0004, 0, 0, 0)
                size = (0.10, 0.012, 0.0002) if axis == 0 else (0.012, 0.10, 0.0002)
                box(floor, f"dash_{axis}_{end}_{index}", size, pose, (1, 0.85, 0.05, 1))
                cursor, index = cursor + 0.18, index + 1


def generate(config, destination):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cpm = None
    if config.get("scenario") == "CPM_mixed":
        from cpm_map import prepare
        config, cpm = prepare(config)
    elif config.get("scenario") == "CPM_mixed_local_2lane":
        from local_junction import prepare
        config, cpm = prepare(config)
    validate(config)
    sdf = ET.Element("sdf", version="1.6")
    world = element(sdf, "world", name="sigmarl_arena")
    if "traffic" in config:
        state = element(world, "plugin", name="gazebo_state", filename="libgazebo_ros_state.so")
        element(element(state, "ros"), "namespace", "/gazebo")
        element(state, "update_rate", 20)
    element(world, "gravity", "0 0 -9.81")
    physics = element(world, "physics", name="ode", type="ode")
    element(physics, "max_step_size", 0.001)
    element(physics, "real_time_update_rate", 1000)
    solver = element(element(physics, "ode"), "solver")
    element(solver, "type", "quick")
    element(solver, "iters", 80)
    scene = element(world, "scene")
    element(scene, "ambient", "0.8 0.8 0.8 1")
    element(scene, "background", "0.22 0.25 0.28 1")
    element(scene, "shadows", "false")
    sun = element(world, "light", name="sun", type="directional")
    element(sun, "diffuse", "0.7 0.7 0.7 1")
    element(sun, "direction", "0 0 -1")
    gui_camera = element(element(world, "gui"), "camera", name="user_camera")
    sx, sy = config["arena_size"]
    element(gui_camera, "pose", numbers((sx * 0.6, -sy * 0.7, sx * 0.85, 0, 0.75, 2.25)) if cpm else "2.7 -3.4 3.6 0 0.68 2.25")
    floor = static_model(world, "arena")
    sx, sy = config["arena_size"]
    lane = config["lane_width"]
    box(floor, "floor", (sx, sy, 0.04), (0, 0, -0.02, 0, 0, 0), (0.14, 0.24, 0.18, 1), True)
    if cpm:
        from cpm_map import write_mesh
        mesh_path = destination / "cpm_road.dae"
        cpm["mesh_triangles"] = write_mesh(cpm, config["map"], mesh_path)
        visual = element(floor, "visual", name="cpm_road")
        element(element(element(visual, "geometry"), "mesh"), "uri", mesh_path.as_uri())
        element(visual, "cast_shadows", "false")
        (destination / "cpm_map.json").write_text(json.dumps(cpm, indent=2) + "\n")
    else:
        add_crossroad(floor, sx, sy, lane)
    for tag in config["ground_tags"]:
        add_tag(floor, tag["id"], tag["x"], tag["y"], 0.0008, tag["size"])
    if config.get("robot_type") == "jetauto":
        from jetauto_model import add_jetauto, template
        destination.mkdir(parents=True, exist_ok=True)
        source_model, poses = template(destination)
        for car in config["cars"]:
            add_jetauto(world, car, config, source_model, poses)
    else:
        for car in config["cars"]:
            add_car(world, car, config["vehicle"])
    camera = config["camera"]
    link = static_model(world, "overhead_camera")
    sensor = element(link, "sensor", name="overhead", type="camera")
    element(sensor, "pose", f"0 0 {camera['height']} 0 {math.pi / 2} {math.pi / 2}")
    element(sensor, "always_on", "true")
    element(sensor, "update_rate", camera["fps"])
    optics = element(sensor, "camera")
    element(optics, "horizontal_fov", camera["horizontal_fov"])
    image = element(optics, "image")
    for key, value in (("width", camera["width"]), ("height", camera["height_pixels"]), ("format", "R8G8B8")):
        element(image, key, value)
    clip = element(optics, "clip")
    element(clip, "near", 0.05)
    element(clip, "far", max(10, camera["height"] * 3))
    if config.get("ros_version") == 2:
        plugin = element(sensor, "plugin", name="overhead_ros2", filename="libgazebo_ros_camera.so")
        ros = element(plugin, "ros")
        element(ros, "namespace", "/")
        element(plugin, "camera_name", "overhead_camera")
        element(plugin, "frame_name", "overhead_camera_optical")
    else:
        element(sensor, "plugin", name="sigmarl_camera", filename="libsigmarl_camera.so")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "arena.world"
    ET.ElementTree(sdf).write(str(path), encoding="utf-8", xml_declaration=True)
    metadata = copy.deepcopy(config)
    if cpm:
        metadata["map"]["geometry_file"] = "cpm_map.json"
        metadata["map"]["lanelet_count"] = len(cpm["lanelets"])
        metadata["map"]["route_counts"] = {name: len(routes) for name, routes in cpm["routes"].items()}
    metadata.update({"tag_dictionary": "DICT_APRILTAG_36h11", "frame": "map",
                     "camera_optical_pose": {"xyz": [0, 0, camera["height"]], "quaternion_xyzw": [1, 0, 0, 0]},
                     "tag_size_definition": "Black border side in metres; white quiet zone excluded",
                     "tag_bitmap_axes": "right = local +X; top = local +Y",
                     "vehicle_reference": "Official base_footprint" if config.get("robot_type") == "jetauto" else "Midpoint between axles, projected onto ground; +X forward, +Y left"})
    (destination / "arena_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/arena.json")
    parser.add_argument("--output", type=Path, default=ROOT / "generated")
    args = parser.parse_args()
    print(generate(load_config(args.config), args.output))
