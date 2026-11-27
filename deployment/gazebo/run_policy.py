#!/usr/bin/python3
"""Launch frozen host inference and its isolated ROS2 endpoint as one bounded run."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
CONTAINER = "sigmarl-jetauto-humble"


def require_running_gazebo():
    """Report missing simulation separately from Docker access failures."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "--type", "container", "-f", "{{.State.Running}}", CONTAINER],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError:
        raise SystemExit("未找到 docker 命令，请先安装并启动 Docker。") from None
    except subprocess.TimeoutExpired:
        raise SystemExit("检查 Gazebo 容器超时，请先确认 Docker 服务可用。") from None
    if result.returncode:
        error = result.stderr.strip()
        if "no such object" not in error.lower() and "no such container" not in error.lower():
            raise SystemExit("无法检查 Gazebo 容器，请先解决 Docker 访问问题：\n" + error)
        start_command = "bash deployment/gazebo/launch_jetauto.sh gui"
        reason = "Gazebo 容器尚未启动，或已退出并被自动删除。"
    elif result.stdout.strip() != "true":
        start_command = "docker start -a " + CONTAINER
        reason = "Gazebo 容器存在，但当前没有运行。"
    else:
        return
    raise SystemExit(
        reason + "\n\n请在项目根目录依次操作：\n"
        "终端 1：启动 Gazebo，并保持该终端运行：\n  " + start_command +
        "\n终端 2：场景加载完成后执行模型策略：\n"
        "  bash deployment/gazebo/launch_jetauto.sh policy --duration 10"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=PROJECT / "checkpoints/nod_error_scale_repeat/seed5080868027432654403/02_control")
    parser.add_argument("--python", default=os.environ.get("SIGMARL_POLICY_PYTHON"))
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--space-scale", type=float, default=3)
    parser.add_argument("--speed-limit", type=float, default=0.2)
    parser.add_argument("--maneuver", choices=("configured", "straight", "left", "right"), default="configured",
                        help="Default follows each car's configured route; other values override all cars")
    parser.add_argument("--cars", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not (0 < args.duration <= 600 and 0 < args.speed_limit <= 0.3 and args.space_scale > 0):
        parser.error("Invalid duration, speed limit or scale")
    if not args.model_path.is_absolute():
        args.model_path = PROJECT / args.model_path
    if not list(args.model_path.glob("reward*_data.json")):
        parser.error("No scored checkpoint metadata at " + str(args.model_path))
    if not args.python:
        candidates = [Path.home() / "anaconda3/envs/sigmarl/bin/python", Path.home() / "miniconda3/envs/sigmarl/bin/python"]
        args.python = str(next((p for p in candidates if p.is_file()), shutil.which("python") or "/usr/bin/python3"))
    runtime = ROOT / "runtime/ros2"
    runtime.mkdir(parents=True, exist_ok=True)
    lock = (runtime / "policy.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("A policy run is already active; use policy-stop first")
    require_running_gazebo()
    metadata = json.loads((ROOT / "generated/ros2/arena_metadata.json").read_text())
    if metadata.get("scenario") != "CPM_mixed_local_2lane":
        parser.error("The running scene must use jetauto_local_junction.json")
    if args.cars and not set(args.cars) <= {c["name"] for c in metadata["cars"]}:
        parser.error("Unknown car name")
    subprocess.run(["bash", str(ROOT / "launch_jetauto.sh"), "reset"], check=True, cwd=PROJECT, timeout=90)
    manifest = runtime / "policy_manifest.json"
    manifest.unlink(missing_ok=True)
    worker = endpoint = None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with (runtime / "policy_worker.log").open("w") as worker_log, (runtime / "policy_bridge.log").open("w") as bridge_log:
        try:
            worker = subprocess.Popen([args.python, "-u", str(ROOT / "policy_worker.py"), "--model-path", str(args.model_path),
                                       "--space-scale", str(args.space_scale), "--speed-limit", str(args.speed_limit),
                                       "--maneuver", args.maneuver], cwd=PROJECT, stdout=worker_log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 45
            while not manifest.is_file() or not (runtime / "policy.sock").exists():
                if worker.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Inference startup failed; see " + str(runtime / "policy_worker.log"))
                time.sleep(0.1)
            contract = json.loads(manifest.read_text())
            print(f"Policy {contract['checkpoint']}: {args.maneuver}; measured Gazebo states; "
                  f"spatial scale={args.space_scale}, time scale={contract['time_scale']}; dry-run={args.dry_run}", flush=True)
            print("Routes: " + ", ".join(name + "=" + route["maneuver"]
                                           for name, route in contract["vehicle_routes"].items()), flush=True)
            command = ["docker", "exec", CONTAINER, "/bin/bash", "/workspace/ros2/run.sh", "policy",
                       "--duration", str(args.duration), "--speed-limit", str(args.speed_limit)]
            if args.dry_run:
                command.append("--dry-run")
            if args.cars:
                command += ["--cars", *args.cars]
            (runtime / "policy_status.json").unlink(missing_ok=True)
            endpoint = subprocess.Popen(command, stdout=bridge_log, stderr=subprocess.STDOUT)
            while endpoint.poll() is None:
                time.sleep(0.2)
            if not (runtime / "policy_status.json").is_file():
                raise RuntimeError("ROS2 endpoint failed at startup; see " + str(runtime / "policy_bridge.log"))
            summary = json.loads((runtime / "policy_status.json").read_text())
            print(json.dumps({key: summary[key] for key in ("checkpoint", "n_agents", "frames", "dry_run", "stop_reason", "fault", "guard_details", "stop_verified", "completed")}, indent=2))
            if endpoint.returncode:
                raise SystemExit("Policy stopped: " + str(summary["fault"]) + "; see " + str(runtime / "policy_bridge.log"))
        except KeyboardInterrupt:
            print("Stopping policy run", flush=True)
        finally:
            if endpoint and endpoint.poll() is None:
                try:
                    subprocess.run(["bash", str(ROOT / "launch_jetauto.sh"), "policy-stop"], cwd=PROJECT, timeout=6, check=False)
                except subprocess.TimeoutExpired:
                    pass
            if worker and worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
            if endpoint and endpoint.poll() is None:
                try:
                    endpoint.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    endpoint.terminate()
                    endpoint.wait(timeout=3)
            (runtime / "policy.sock").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
