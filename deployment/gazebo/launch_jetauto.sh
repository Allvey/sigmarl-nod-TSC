#!/usr/bin/env bash
set -eo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$task_root/../.." && pwd)"
action="${1:-run}"
if [[ "$action" == --config ]]; then
  action=run
elif [[ $# -gt 0 ]]; then
  shift
fi
image_name=sigmarl-jetauto:humble
container_name=sigmarl-jetauto-humble
case "$action" in
  policy) exec /usr/bin/python3 "$task_root/run_policy.py" "$@" ;;
  stop) exec docker stop "$container_name" ;;
  check|test|drive|reset|demo-start|demo-stop|policy-stop|policy-status)
    exec docker exec "$container_name" /bin/bash /workspace/ros2/run.sh tools "$action" "$@" ;;
  run|gui|build) ;;
  --help|-h) echo "Usage: bash deployment/gazebo/launch_jetauto.sh [run|gui|build] [--config NAME] | check|test|drive|reset|demo-start|demo-stop|policy|policy-stop|policy-status|stop"; exit 0 ;;
  *) echo "Usage: bash deployment/gazebo/launch_jetauto.sh [run|gui|build] [--config NAME] | check|test|drive|reset|demo-start|demo-stop|policy|policy-stop|policy-status|stop" >&2; exit 2 ;;
esac
profile=jetauto_local_junction.json
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) profile="${2:?--config requires a profile filename}"; shift 2 ;;
    *) echo "Unknown launch argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$profile" != "${profile##*/}" || ! -f "$task_root/config/$profile" ]]; then
  echo "Profile must name an existing file under deployment/gazebo/config" >&2
  exit 2
fi
if /usr/bin/python3 "$task_root/reuse_scene.py" "$action" "$profile"; then
  exit 0
else
  reuse_result=$?
  if [[ "$reuse_result" != 20 ]]; then exit "$reuse_result"; fi
fi
if ! docker image inspect "$image_name" >/dev/null 2>&1; then
  docker build --build-arg "USER_ID=$(id -u)" --build-arg "GROUP_ID=$(id -g)" -t "$image_name" "$task_root/ros2"
fi
if [[ ! -f "$task_root/generated/vendor/jetauto_description/provenance.json" ]]; then
  /usr/bin/python3 "$task_root/fetch_jetauto.py"
fi
mkdir -p "$task_root/runtime/ros2"
options=(--rm --init --name "$container_name" --shm-size 512m
  --mount "type=bind,src=$task_root,dst=/workspace"
  --mount "type=bind,src=$project_root/assets/maps,dst=/project/assets/maps,readonly"
  --mount "type=bind,src=$project_root/utilities,dst=/project/utilities,readonly"
  --env "SIGMARL_ARENA_CONFIG=/workspace/config/$profile" --env ROS_DOMAIN_ID=57 --env ROS_LOCALHOST_ONLY=1)
if [[ "$action" == gui ]]; then
  if [[ -z "${DISPLAY:-}" ]]; then echo "No DISPLAY; use the default headless renderer instead" >&2; exit 1; fi
  auth="$task_root/runtime/ros2/xauthority"
  (umask 077; touch "$auth")
  xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$auth" nmerge -
  options+=(--env "DISPLAY=$DISPLAY" --env XAUTHORITY=/workspace/runtime/ros2/xauthority
    --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix,readonly)
fi
echo "JetAuto Orin Nano Super 4GB + A1 / ROS2 Humble. Container: $container_name"
echo "Scene profile: $profile"
echo "In another terminal: bash deployment/gazebo/launch_jetauto.sh check"
exec docker run "${options[@]}" "$image_name" "$action" "$@"
