#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export ROS_LOG_DIR="$task_root/runtime/ros2/ros_logs"
export GAZEBO_LOG_PATH="$task_root/runtime/ros2/gazebo_logs"
export GAZEBO_MODEL_DATABASE_URI=http://127.0.0.1:9
export GAZEBO_MASTER_URI=http://127.0.0.1:11365
export ALSOFT_DRIVERS=null
mkdir -p "$ROS_LOG_DIR" "$GAZEBO_LOG_PATH"
action="${1:-run}"
shift || true
if [[ "$action" == tools ]]; then
  exec /usr/bin/python3 "$task_root/ros2/tools.py" "$@"
fi
if [[ "$action" == policy ]]; then
  exec /usr/bin/python3 "$task_root/ros2/policy_bridge.py" "$@"
fi
if [[ "$action" != run && "$action" != gui && "$action" != build ]]; then
  echo "Expected run, gui, build or tools" >&2
  exit 2
fi
config="${SIGMARL_ARENA_CONFIG:-$task_root/config/jetauto_local_junction.json}"
/usr/bin/python3 "$task_root/generate_world.py" --config "$config" --output "$task_root/generated/ros2"
cmake -S "$task_root/ros2" -B "$task_root/build/ros2" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$task_root/build/ros2" --parallel 2
export GAZEBO_PLUGIN_PATH="$task_root/build/ros2:/opt/ros/humble/lib:$GAZEBO_PLUGIN_PATH"
if [[ "$action" == build ]]; then exit 0; fi
children=()
cleanup() {
  trap - EXIT INT TERM
  for child in "${children[@]}"; do kill -TERM "$child" 2>/dev/null || true; done
  wait || true
}
trap cleanup EXIT INT TERM
if [[ "$action" != gui ]]; then
  export DISPLAY=:99
  Xvfb "$DISPLAY" -screen 0 2048x2048x24 -nolisten tcp > "$task_root/runtime/ros2/xvfb.log" 2>&1 &
  children+=("$!")
  sleep 1
fi
gzserver --verbose -s libgazebo_ros_init.so -s libgazebo_ros_factory.so "$task_root/generated/ros2/arena.world" &
server=$!
children+=("$server")
camera_height="$(/usr/bin/python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["camera"]["height"])' "$task_root/generated/ros2/arena_metadata.json")"
ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z "$camera_height" --qx 1 --qy 0 --qz 0 --qw 0 \
  --frame-id map --child-frame-id overhead_camera_optical > "$task_root/runtime/ros2/camera_tf.log" 2>&1 &
children+=("$!")
if /usr/bin/python3 -c 'import json,sys;sys.exit(0 if "traffic" in json.load(open(sys.argv[1])) else 1)' "$task_root/generated/ros2/arena_metadata.json"; then
  /usr/bin/python3 "$task_root/ros2/local_traffic.py" > "$task_root/runtime/ros2/traffic.log" 2>&1 &
  children+=("$!")
fi
if [[ "$action" == gui ]]; then
  gzclient --verbose &
  children+=("$!")
fi
wait "$server"
