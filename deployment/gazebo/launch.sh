#!/usr/bin/env bash
set -eo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
gui=true
paused=false
build_only=false
config="$task_root/config/arena.json"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --headless) gui=false; shift ;;
    --paused) paused=true; shift ;;
    --build-only) build_only=true; shift ;;
    --config) config="${2:?--config requires a JSON file}"; shift 2 ;;
    --ros-port) export ROS_MASTER_URI="http://127.0.0.1:${2:?port required}"; shift 2 ;;
    --gazebo-port) export GAZEBO_MASTER_URI="http://127.0.0.1:${2:?port required}"; shift 2 ;;
    --help|-h)
      echo "Usage: bash deployment/gazebo/launch.sh [--headless] [--paused] [--build-only] [--config FILE] [--ros-port PORT] [--gazebo-port PORT]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ ! -f /opt/ros/noetic/setup.bash ]] || ! command -v gzserver >/dev/null; then
  echo "Requires ROS Noetic and Gazebo Classic 11 (see docs/current/GAZEBO_ENVIRONMENT.md)." >&2
  exit 1
fi
# Use the ROS/system Python and libraries, independently of the training Conda environment.
unset PYTHONHOME
export PATH="/opt/ros/noetic/bin:/usr/bin:/bin"
export CMAKE_PREFIX_PATH=/opt/ros/noetic
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages
export LD_LIBRARY_PATH=/opt/ros/noetic/lib:/usr/lib/x86_64-linux-gnu
source /opt/ros/noetic/setup.bash
source /usr/share/gazebo-11/setup.sh
export ROS_LOG_DIR="$task_root/runtime/ros_logs"
export ROS_HOME="$task_root/runtime/ros_home"
export GAZEBO_LOG_PATH="$task_root/runtime/gazebo_logs"
export GAZEBO_MODEL_DATABASE_URI=""
mkdir -p "$ROS_LOG_DIR" "$ROS_HOME" "$GAZEBO_LOG_PATH"
/usr/bin/python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); sys.exit("Use launch_jetauto.sh for the ROS2 JetAuto profile" if c.get("ros_version",1)!=1 else 0)' "$config"
/usr/bin/python3 "$task_root/generate_world.py" --config "$config"
/usr/bin/cmake -S "$task_root" -B "$task_root/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPYTHON_EXECUTABLE=/usr/bin/python3
/usr/bin/cmake --build "$task_root/build" --parallel 2
export GAZEBO_PLUGIN_PATH="$task_root/build:$GAZEBO_PLUGIN_PATH"
if [[ "$build_only" == true ]]; then exit 0; fi
camera_height="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["camera"]["height"])' "$config")"
echo "Starting arena. Drive: /car_0/drive and /car_1/drive; image: /overhead_camera/image_raw"
exec /usr/bin/python3 /opt/ros/noetic/bin/roslaunch "$task_root/arena.launch" \
  "world:=$task_root/generated/arena.world" "gui:=$gui" "paused:=$paused" "camera_height:=$camera_height"
