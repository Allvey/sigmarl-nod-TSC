FROM ros:humble-ros-base-jammy@sha256:a18afd4a42ad603579d03b9012385dfeaf2e848423f0a6c41699766e7849c86c
SHELL ["/bin/bash", "-c"]
COPY download_dependencies.py /tmp/download_dependencies.py
RUN apt-get update
RUN packages="gazebo libgazebo-dev ros-humble-gazebo-ros-pkgs ros-humble-xacro ros-humble-tf2-ros python3-opencv python3-numpy cmake g++ xvfb xauth libgl1-mesa-dri"; \
    python3 /tmp/download_dependencies.py $packages && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $packages && \
    rm -rf /var/lib/apt/lists/* /tmp/download_dependencies.py
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd --gid ${GROUP_ID} jetauto && useradd --uid ${USER_ID} --gid ${GROUP_ID} --create-home jetauto
USER jetauto
WORKDIR /workspace
ENV ROS_DOMAIN_ID=57 ROS_LOCALHOST_ONLY=1 LIBGL_ALWAYS_SOFTWARE=1
ENTRYPOINT ["/bin/bash", "/workspace/ros2/run.sh"]
