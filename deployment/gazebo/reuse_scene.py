#!/usr/bin/python3
"""Reuse this workspace's existing Gazebo container without resetting its world."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CONTAINER = "sigmarl-jetauto-humble"
NEW_CONTAINER = 20


def reuse_scene(action, profile):
    try:
        inspected = subprocess.run(["docker", "inspect", "--type", "container", CONTAINER],
                                   capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SystemExit("无法检查 Gazebo 容器：" + str(error)) from None
    if inspected.returncode:
        if any(message in inspected.stderr.lower() for message in ("no such object", "no such container")):
            return NEW_CONTAINER
        raise SystemExit("无法访问 Docker：\n" + inspected.stderr.strip())
    container = json.loads(inspected.stdout)[0]
    mounts = {mount["Destination"]: mount["Source"] for mount in container.get("Mounts", [])}
    if mounts.get("/workspace") != str(ROOT):
        raise SystemExit("同名容器属于其他工作目录，请先确认它的用途；未修改现有容器。")
    environment = dict(value.split("=", 1) for value in container["Config"].get("Env", []) if "=" in value)
    if environment.get("SIGMARL_ARENA_CONFIG") != "/workspace/config/" + profile:
        raise SystemExit("已有容器使用不同的场景配置。切换场景前先执行：\n"
                         "  bash deployment/gazebo/launch_jetauto.sh stop\n"
                         "然后重新执行启动命令。")
    if action == "build":
        raise SystemExit("已有仿真容器。构建前请先停止它，避免覆盖正在使用的场景文件。")
    state = container["State"]
    if state.get("Paused"):
        raise SystemExit("Gazebo 容器已暂停，请先执行：docker unpause " + CONTAINER)
    if not state.get("Running"):
        raise SystemExit("已有容器已停止，可恢复运行：\n  docker start -a " + CONTAINER)
    print("复用已运行的 Gazebo 场景（" + profile + "）。", flush=True)
    if action == "gui":
        window = subprocess.run(["docker", "exec", CONTAINER, "pgrep", "-x", "gzclient"],
                                capture_output=True, text=True, timeout=10)
        if window.returncode == 0:
            print("Gazebo 窗口已经打开，无需重复启动。")
        elif window.returncode == 1:
            display = environment.get("DISPLAY")
            if not display or "/tmp/.X11-unix" not in mounts:
                raise SystemExit("当前容器以无窗口模式启动。要切换到图形界面，请先执行 stop，再执行 gui。")
            authority = subprocess.run(["xauth", "nlist", display], capture_output=True, text=True, timeout=10)
            if authority.returncode or not authority.stdout.strip():
                raise SystemExit("无法读取现有显示器的 X11 授权，请在本机桌面终端执行 gui。")
            auth_path = ROOT / "runtime/ros2/xauthority"
            auth_path.touch(mode=0o600, exist_ok=True)
            auth_path.chmod(0o600)
            entries = "\n".join("ffff" + line[4:] for line in authority.stdout.splitlines()) + "\n"
            subprocess.run(["xauth", "-f", str(auth_path), "nmerge", "-"], input=entries,
                           text=True, check=True, timeout=10)
            print("正在重新打开 Gazebo 窗口；现有仿真继续运行。", flush=True)
            os.execvp("docker", ["docker", "exec", "-e", "GAZEBO_MASTER_URI=http://127.0.0.1:11365",
                                 "-e", "GAZEBO_MODEL_DATABASE_URI=http://127.0.0.1:9",
                                 "-e", "ALSOFT_DRIVERS=null", CONTAINER, "gzclient", "--verbose"])
        else:
            raise SystemExit("无法检查 Gazebo 窗口：\n" + window.stderr.strip())
    print("执行模型策略：bash deployment/gazebo/launch_jetauto.sh policy --duration 10")
    return 0


if __name__ == "__main__":
    raise SystemExit(reuse_scene(sys.argv[1], sys.argv[2]))
