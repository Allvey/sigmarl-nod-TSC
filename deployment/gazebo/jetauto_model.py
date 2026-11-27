"""Adapt pinned official JetAuto geometry to our ROS2 experiment world."""
import copy
import math
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parent


def transform(xyz, rpy):
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    result = np.eye(4)
    result[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                       [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                       [-sp, cp * sr, cp * cr]]
    result[:3, 3] = xyz
    return result


def link_transforms(urdf):
    children = {joint.find("child").get("link") for joint in urdf.findall("joint")}
    roots = {link.get("name") for link in urdf.findall("link")} - children
    if len(roots) != 1:
        raise ValueError("Official URDF must have one root link")
    poses = {roots.pop(): np.eye(4)}
    pending = list(urdf.findall("joint"))
    while pending:
        progress = False
        for joint in pending[:]:
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in poses:
                continue
            origin = joint.find("origin")
            xyz = [0, 0, 0] if origin is None else [float(v) for v in origin.get("xyz", "0 0 0").split()]
            rpy = [0, 0, 0] if origin is None else [float(v) for v in origin.get("rpy", "0 0 0").split()]
            poses[child] = poses[parent] @ transform(xyz, rpy)
            pending.remove(joint)
            progress = True
        if not progress:
            raise ValueError("Unresolved URDF joint tree")
    return poses


def template(destination):
    import xacro
    vendor = ROOT / "generated/vendor/jetauto_description"
    if not (vendor / "provenance.json").exists():
        raise RuntimeError("Run /usr/bin/python3 deployment/gazebo/fetch_jetauto.py first")
    expanded = destination / "expanded_description"
    (expanded / "urdf").mkdir(parents=True, exist_ok=True)
    for source in (vendor / "urdf").glob("*.xacro"):
        text = source.read_text().replace("$(find jetauto_description)", str(expanded))
        text = text.replace("package://jetauto_description/", vendor.as_uri() + "/")
        (expanded / "urdf" / source.name).write_text(text)
    # These select ordinary JetAuto and A1 geometry; runtime sensors are added below.
    names = {"MACHINE_TYPE": "JetAuto", "LIDAR_TYPE": "A1", "DEPTH_CAMERA_TYPE": "None"}
    previous = {key: os.environ.get(key) for key in names}
    try:
        os.environ.update(names)
        document = xacro.process_file(str(expanded / "urdf/jetauto.xacro"))
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    urdf_path = destination / "jetauto.urdf"
    urdf_path.write_text(document.toxml())
    # Use Gazebo's own URDF importer so official inertias, meshes and joints survive.
    converted = subprocess.run(["gz", "sdf", "-p", str(urdf_path)], check=True, capture_output=True, text=True)
    model = ET.fromstring(converted.stdout).find("model")
    if model is None or model.find("link") is None:
        raise RuntimeError("Gazebo could not convert the official JetAuto model")
    return model, link_transforms(ET.fromstring(document.toxml()))


def add_jetauto(world, car, config, source_model, poses):
    from generate_world import add_tag, box, element, inertia, numbers
    model = copy.deepcopy(source_model)
    model.set("name", car["name"])
    element(model, "pose", numbers((car["x"], car["y"], 0.002, 0, 0, car["yaw"])))
    # The imported CAD wheels are fixed geometry. Remove ordinary-wheel tangential
    # friction for the holonomic approximation; retain normal collision response.
    for collision in model.findall("link/collision"):
        surface = collision.find("surface")
        if surface is None:
            surface = element(collision, "surface")
        friction = surface.find("friction")
        if friction is not None:
            surface.remove(friction)
        ode = element(element(surface, "friction"), "ode")
        element(ode, "mu", 0)
        element(ode, "mu2", 0)
    # URDF conversion drops the manufacturer's separate Gazebo colour tags.
    for visual in model.findall("link/visual"):
        name = visual.get("name", "")
        color = (0.22, 0.70, 0.28, 1) if "__base_link_visual" in name else (0.06, 0.07, 0.08, 1)
        old = visual.find("material")
        if old is not None:
            visual.remove(old)
        material = element(visual, "material")
        element(material, "ambient", numbers(color))
        element(material, "diffuse", numbers(color))
    main = model.find("link")
    parent = main.get("name")
    # Keep official meshes and add the elevated, measured tag plate to the model.
    mount = element(model, "link", name="tag_mount")
    inertia(mount, 0.005, (1e-5, 1e-5, 1e-5), config["vehicle"]["tag_height"])
    tag_height = config["vehicle"]["tag_height"]
    box(mount, "tag_support", (0.004, 0.004, tag_height - 0.12),
        (0, 0, (tag_height + 0.12) / 2, 0, 0, 0), car["color"])
    add_tag(mount, car["tag_id"], 0, 0, tag_height, config["vehicle"]["tag_size"])
    joint = element(model, "joint", name="tag_mount_joint", type="fixed")
    element(joint, "parent", parent)
    element(joint, "child", "tag_mount")
    # URDF root and SDF model frame agree; lidar_frame includes the A1 mounting yaw.
    lidar_pose = poses["lidar_frame"]
    yaw = math.atan2(lidar_pose[1, 0], lidar_pose[0, 0])
    lidar = element(model, "link", name="a1_sensor")
    element(lidar, "pose", numbers((*lidar_pose[:3, 3], 0, 0, yaw)))
    inertia(lidar, 0.001, (1e-6, 1e-6, 1e-6))
    fixed = element(model, "joint", name="a1_sensor_joint", type="fixed")
    element(fixed, "parent", parent)
    element(fixed, "child", "a1_sensor")
    sensor = element(lidar, "sensor", name="a1", type="ray")
    element(sensor, "always_on", "true")
    element(sensor, "update_rate", config["lidar"]["fps"])
    horizontal = element(element(element(sensor, "ray"), "scan"), "horizontal")
    for name, value in (("samples", config["lidar"]["samples"]), ("resolution", 1), ("min_angle", -math.pi), ("max_angle", math.pi)):
        element(horizontal, name, value)
    ray = sensor.find("ray")
    bounds = element(ray, "range")
    for name, value in (("min", config["lidar"]["range_min"]), ("max", config["lidar"]["range_max"]), ("resolution", 0.01)):
        element(bounds, name, value)
    plugin = element(sensor, "plugin", name="a1_ros2", filename="libgazebo_ros_ray_sensor.so")
    ros = element(plugin, "ros")
    element(ros, "namespace", "/" + car["name"])
    element(ros, "remapping", "~/out:=" + config["interfaces"]["scan_topic"])
    element(plugin, "output_type", "sensor_msgs/LaserScan")
    element(plugin, "frame_name", car["name"] + "/lidar_frame")
    controller = element(model, "plugin", name="sigmarl_jetauto", filename="libsigmarl_jetauto_ros2.so")
    element(element(controller, "ros"), "namespace", "/" + car["name"])
    for name in ("max_speed", "max_lateral_speed", "max_yaw_rate", "max_acceleration", "max_yaw_acceleration", "command_timeout"):
        element(controller, name, config["vehicle"][name])
    element(controller, "base_frame", car["name"] + "/base_footprint")
    element(controller, "command_topic", config["interfaces"]["command_topic"])
    element(controller, "lidar_frame", car["name"] + "/lidar_frame")
    element(controller, "lidar_xyz", numbers(lidar_pose[:3, 3]))
    element(controller, "lidar_yaw", yaw)
    world.append(model)
