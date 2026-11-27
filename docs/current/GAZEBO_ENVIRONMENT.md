# Gazebo 俯拍定位实验环境

> 本文记录最初的 ROS1 前轮转向小车原型。已确认实车为 JetAuto Orin Nano
> Super 4GB＋A1，专用 ROS2 环境与启动命令见 [JETAUTO_GAZEBO.md](JETAUTO_GAZEBO.md)。

本环境使用机器已有的 **Ubuntu 20.04、ROS Noetic、Gazebo Classic 11**。
在独立目录 `deployment/gazebo/` 内构建，无需修改训练 Conda 环境，
无需下载外部车辆模型，也不需要安装缺失的 `gazebo_plugins` 相机插件。
Gazebo Classic 与新版 Gazebo 的插件不兼容；这里的启动命令适用于 Classic 11。

## 启动

在项目根目录执行：

```bash
bash deployment/gazebo/launch.sh
```

脚本使用系统 Python `/usr/bin/python3` 和 ROS Noetic 的库，生成场景、编译两个
C++ 插件、启动 ROS master（如需要）、Gazebo server 和图形界面。首次编译较慢。
关闭启动终端中的进程使用 `Ctrl+C`。

```bash
# 只生成场景和编译插件
bash deployment/gazebo/launch.sh --build-only

# 不打开图形窗口；相机渲染仍需要有效的 DISPLAY/OpenGL
bash deployment/gazebo/launch.sh --headless

# 暂停启动，之后在 Gazebo 中点击播放
bash deployment/gazebo/launch.sh --paused

# 与其他 ROS/Gazebo 实验隔离端口
bash deployment/gazebo/launch.sh --ros-port 11321 --gazebo-port 11355
```

使用自定义 ROS 端口时，其他终端也要设置，例如：

```bash
export ROS_MASTER_URI=http://127.0.0.1:11321
```

只有连接 Gazebo 客户端时需要匹配 `GAZEBO_MASTER_URI`，例如：

```bash
export GAZEBO_MASTER_URI=http://127.0.0.1:11355
```

不要同时启动两个使用同一 ROS/Gazebo master 的本场景实例。不同配置实例也应使用
各自的项目副本，避免覆盖同一 `generated/` 目录。

## 默认场景

配置来源：[`arena.json`](../../deployment/gazebo/config/arena.json)。修改后重启即可。

| 项目 | 默认值 |
| --- | --- |
| 场地 | 3×3 米，原点在中心，X 向右、Y 向上、Z 向上 |
| 道路 | 双向交叉口，单车道宽 0.25 米 |
| 标线 | 白色道路边界、黄色虚线中心线；路口内部无标线 |
| 地面 Tag | AprilTag 36h11，ID 0、1、2、3，黑色外边框边长 0.14 米 |
| 车辆 Tag | ID 10、11，黑色外边框边长 0.07 米，安装高度 0.071 米 |
| 车辆 | `car_0`、`car_1`，车身 0.16×0.08 米，轴距 0.16 米 |
| 相机 | 高度 3 米，2048×2048，目标 20 帧/秒，水平视场角 1.04 rad |
| 动力学 | 四轮物理接触，后轮驱动、前轮 Ackermann 转向 |
| 控制限制 | ±1 m/s、±35°、加速度 0.8 m/s²、转向速率 1.5 rad/s |
| 命令超时 | 0.35 秒墙钟时间未收到新命令，目标速度和转角归零，按速率限制减速 |

车辆默认静止。道路外有实体地板，无实体围栏；车辆可以离开道路。
相机实际帧率取决于 GPU 和实时因子，配置的 20 Hz 不能视为实测保证。

车体参考点是两轴中点在地面的投影，车体 +X 指向车头、+Y 指向左侧。
Tag 图案右方对应车体 +X，上方对应车体 +Y；其位置、车体参考点、
Tag 标准 PnP 坐标轴之间的变换需要在后续定位节点中显式处理。

Tag 按真实码字生成黑白几何块，包含一格白色留边，不依赖贴图下载。
参数中的 `tag_size` 不包含白色留边。

## ROS 接口

| 话题 | 消息类型 | 含义 |
| --- | --- | --- |
| `/overhead_camera/image_raw` | `sensor_msgs/Image` | RGB8 俯拍图像 |
| `/overhead_camera/camera_info` | `sensor_msgs/CameraInfo` | 无畸变针孔相机内参 |
| `/car_0/drive`、`/car_1/drive` | `geometry_msgs/Vector3Stamped` | `vector.x` 目标速度 m/s，`vector.y` 前轮等效转角 rad，`vector.z` 保留 |
| `/car_0/ground_truth`、`/car_1/ground_truth` | `nav_msgs/Odometry` | 20 Hz 左右的仿真真值，供调试和评估 |
| `/clock` | `rosgraph_msgs/Clock` | 仿真时间 |
| `/tf_static` | `tf2_msgs/TFMessage` | `map → overhead_camera_optical` 的固定变换 |

`drive` 是本项目的临时适配接口，**不是** `Twist` 或标准 Ackermann 消息。
速度允许倒车；正转角表示向左。连续以约 20 Hz 发命令。
命令时间戳供记录，超时目前按接收时墙钟计算，不会拒绝时间戳过旧的消息。
该接口适合本地仿真，后续实车接入应补充消息时效检查。

Odometry 的位姿在 `map` 下，速度按 ROS 约定表达在车辆自身坐标系中。
相机 optical 坐标为右、下、前，固定变换四元数为 `[1,0,0,0]`（xyzw）。
图像右方对应世界 +X，图像上方对应世界 +Y。

生成的 `deployment/gazebo/generated/arena_metadata.json` 包含所有 Tag 的位置、
尺寸、高度、车辆初始位姿和相机外参。车顶 Tag 不能直接套用地面单应矩阵。

## 手动控制与检查

另开终端，在项目根目录运行。使用自定义 ROS 端口时先设置 `ROS_MASTER_URI`。

```bash
# car_0 以 0.15 m/s 直行 2 秒，然后发送停车命令
/usr/bin/python3 deployment/gazebo/drive.py --car car_0 --speed 0.15 --duration 2

# 前轮左转 0.3 rad（约 17°）
/usr/bin/python3 deployment/gazebo/drive.py --car car_0 --speed 0.15 --steering 0.3 --duration 2

# 检查相机内参、图像与六个 Tag，并保存带标注图片
/usr/bin/python3 deployment/gazebo/check_camera.py

# 验证直行、左转、地面接触和断流后停车；会移动车辆
/usr/bin/python3 deployment/gazebo/smoke_test.py

# 不启动 Gazebo 的离线检查
/usr/bin/python3 -m unittest discover -s deployment/gazebo/tests -v
```

`drive.py --duration` 使用墙钟时间；实时因子较低时，仿真内的行驶距离会缩短。
`smoke_test.py` 按仿真时间运行，建议在刚启动且车辆处于默认位置时使用。
相机检查采用原图和半分辨率两个尺度，以覆盖大小不同的地面及车顶 Tag。

输出都位于被 Git 忽略的目录中：

```text
deployment/gazebo/build/       编译产物
deployment/gazebo/generated/   arena.world、arena_metadata.json
deployment/gazebo/runtime/     相机预览、ROS 和渲染日志
```

## 依赖与常见问题

当前机器已验证有 `gazebo11`、Gazebo 开发头文件、ROS Noetic 的
`gazebo_ros`、`roscpp`、`geometry_msgs`、`nav_msgs`、`sensor_msgs`、`tf2_ros`、
Python `rospy`、`cv_bridge`、NumPy、带 ArUco 的系统 OpenCV，以及 CMake/G++。
脚本不会执行 apt 安装或更改全局环境。

- **Conda 中缺少 cv2 / ROS 包**：执行上面的 `/usr/bin/python3` 命令，不使用训练环境的 Python。
- **没有图像、OpenGL 或 DISPLAY 错误**：相机需要渲染，即使 `--headless` 也需要
  X/OpenGL。请在有图形桌面的会话运行，或单独配置虚拟显示和软件渲染。
- **收不到话题**：检查 `ROS_MASTER_URI` 与启动端口一致，并确认场景未暂停。
- **图像慢**：降低配置中的分辨率或帧率；降低分辨率后必须重新验证小车 Tag 检出率。
- **相机检查缺少某个 Tag**：检查预览，确认车辆没有驶出视野，地面 Tag 未被遮挡。

## 当前边界与下一步

本次完成场景、物理小车、俯拍相机、控制接口和 Tag 可检出性验证。
尚未接入训练好的 Actor/NOD，没有自动驾驶、视觉定位滤波或车道建图节点。
道路几何是独立的最小实验场景，车道宽度与现有 CPM 训练地图不同，
不能把本场景运行成功视为策略迁移成功。

下一步可依次接入：相机标定与车辆定位、道路地图/路线定义、训练一致的观测构造、
NOD/Actor 推理。仿真真值在切换到视觉闭环后仅用于评估。

实现参考：[Gazebo Classic 模型插件](https://classic.gazebosim.org/tutorials?cat=write_plugin&tut=plugins_model)、
[Gazebo ROS 相机接口](https://classic.gazebosim.org/tutorials?cat=connect_ros&tut=ros_gzplugins)。
