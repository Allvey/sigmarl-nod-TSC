# JetAuto Orin Nano Super 4GB＋A1 俯拍实验场景

已确认的实车配置是 **Hiwonder JetAuto、Jetson Orin Nano Super（4GB）、SLAMTEC A1 雷达**。
按厂家手册，Orin Nano 的软件路线为 Ubuntu 22.04 / ROS2 Humble。
实车是否改过镜像，后续连接时仍应以设备上的实际环境为准。

本目录新增 ROS2 专用环境，运行在开发电脑的 Docker 容器里，不在 4GB Jetson 上运行
Gazebo，不改动开发电脑已有的 Noetic 安装。
最初的 [ROS1 小车原型](GAZEBO_ENVIRONMENT.md) 保留作参考。
当前默认场景为 **局部双车道路口＋3 台 JetAuto**，地图、相机、规则演示与路线说明见
[LOCAL_JUNCTION_GAZEBO.md](LOCAL_JUNCTION_GAZEBO.md)。

## 启动与操作

从项目根目录执行；需要 Docker daemon 可用，并能联网获取首次构建依赖。

```bash
# 默认：启动隔离的 ROS2/Gazebo 环境，通过容器内虚拟显示渲染俯拍图像
bash deployment/gazebo/launch_jetauto.sh

# 在有 X11 桌面的会话中打开 Gazebo 图形窗口
bash deployment/gazebo/launch_jetauto.sh gui

# 只准备镜像、生成模型和编译插件
bash deployment/gazebo/launch_jetauto.sh build

# 原来的两车简单交叉口配置
bash deployment/gazebo/launch_jetauto.sh gui --config jetauto_a1.json
```

首次启动会构建 `sigmarl-jetauto:humble` 镜像，并获取约 4.5 MB 官方模型描述资源。
容器名为 `sigmarl-jetauto-humble`，同一时间运行一个实例。重复启动相同配置会复用现有实例；
再次执行 `gui` 可重新打开已关闭的窗口。首次启动为无窗口模式时，需先 `stop` 再用 `gui` 启动。
脚本使用官方 ROS Humble 基础镜像的固定 digest，依赖包在构建时从 apt 源解析，
后续重建的依赖版本可能变化。Ubuntu 包并行下载后会按 apt 提供的摘要校验。

另开终端运行检查或控制命令，所有 ROS2 工具都在该容器中执行：

```bash
# 按当前场景检查所有车辆、Tag 和 A1 的 LaserScan
bash deployment/gazebo/launch_jetauto.sh check

# 暂停规则演示；恢复时用 demo-start
bash deployment/gazebo/launch_jetauto.sh demo-stop

# 前进 2 秒（仿真时间），随后停车
bash deployment/gazebo/launch_jetauto.sh drive --vx 0.15 --duration 2

# 横向平移
bash deployment/gazebo/launch_jetauto.sh drive --vx 0 --vy 0.12 --duration 2

# 原地逆时针旋转
bash deployment/gazebo/launch_jetauto.sh drive --vx 0 --wz 0.4 --duration 2

# 实际移动车辆，检查直行、横移、旋转、失联停车和地面接触
bash deployment/gazebo/launch_jetauto.sh test

# 恢复初始位置与仿真时钟
bash deployment/gazebo/launch_jetauto.sh reset

# 停止容器；关闭启动终端也可以用 Ctrl+C
bash deployment/gazebo/launch_jetauto.sh stop
```

运动检查建议在车辆处于初始位置时执行。`drive/test/reset` 自动暂停规则演示；
执行 `demo-start` 会重置当前配置的全部车辆并恢复演示。重新启动会恢复默认场景。
默认局部路口支持 `--car car_0`、`car_1`、`car_2`；完整 CPM 可选配置另有 `car_3`。
脚本中的速度单位为 m/s，角速度单位为 rad/s；
角速度参数不表示车轮转角。

## 模型来源与适配范围

原始车体、轮子、屏幕和 A1 安装变换取自
[Hiwonder/JetAuto](https://github.com/Hiwonder/JetAuto/tree/Jetson_nano_ros1/src/jetauto_simulations/jetauto_description)，
固定提交为 `569b77d3ed5ae4a581cdd8e341f29eb8476284da`。
`fetch_jetauto.py` 按该提交的 Git blob 校验文件完整性，下载到忽略目录；
不修改或提交上游网格资源。生成目录保存 `provenance.json`。

这里复用的是官方 ROS1 分支的机械描述，并新增本项目的 ROS2 运行插件。
它不是对厂家完整 ROS2 课程资源包的原样安装。对应的官方 ROS2 仿真课程见
[MoveIt & Gazebo Simulation](https://wiki.hiwonder.com/projects/JetAuto/en/jetauto-orin-nano/docs/10.Movelt_Gazebo_simulation.html)。

- 官方 Xacro 经展开后由 Gazebo 自己的 URDF 导入器转成 SDF，保留网格、碰撞及惯量。
- 使用普通 JetAuto 形态；暂不加载机械臂或机载深度相机，顶部增加 Tag 安装板。
- A1 扫描器的安装位置及绕 Z 轴方向从官方 URDF 的关节树计算，包含 `base_footprint` 偏移。
- 底盘采用厂家 ROS1 模型所用的同类平面运动学近似，并增加速度限幅、加速度限制和命令超时停车。
- 官方导入模型的车轮为固定几何。本适配关闭其普通车轮切向摩擦，保留法向碰撞响应，
  以免用普通轮子的接触约束阻碍全向运动；速度按各连杆质心偏移换算，保持 `base_footprint` 的指令语义。
- 不模拟麦轮滚子、电机闭环和滑移的细节；当前结果不能用于验证轮地接触或真实刹车距离。
- A1 使用普通 ray 传感器，输出 360 点、10 Hz、0.15～12 米的合成扫描。
  这些是本实验配置，不是对真实 A1 扫描模式、时序和精度的完整复刻。

## 场景与接口

下表记录 [`jetauto_a1.json`](../../deployment/gazebo/config/jetauto_a1.json) 的两车基础交叉口配置；
默认局部路口的尺寸与相机配置见 [局部路口说明](LOCAL_JUNCTION_GAZEBO.md)。

| 配置项 | 默认值 |
| --- | --- |
| 场地 | 3×3 米双向交叉口，单车道宽 0.5 米 |
| 碰撞尺寸参考 | 长 0.302 米、宽 0.26 米；接实车前按外廓实测复核 |
| 地面 Tag | 36h11，ID 0～3，黑色边框边长 0.14 米 |
| 车顶 Tag | ID 10、11，黑色边框边长 0.12 米，模型安装高度 0.35 米 |
| 俯拍相机 | 3 米高，2048×2048，目标 10 Hz |
| 速度限制 | 前后 ±0.4 m/s、横向 ±0.3 m/s、旋转 ±1.2 rad/s |
| 命令超时 | 0.35 秒墙钟时间，随后按加速度限制减速 |

相机默认软件渲染，实际帧率与实时因子取决于电脑性能；没有保证达到目标帧率。
Tag 高度、尺寸是仿真实验安装配置，不能当作真实 Tag 安装测量结果。

| 话题 | 类型与语义 |
| --- | --- |
| `/car_0/controller/cmd_vel` | `geometry_msgs/msg/Twist`：`linear.x=vx`、`linear.y=vy`、`angular.z=ω` |
| `/car_1/controller/cmd_vel` | 第二台机器人，同上 |
| `/car_2/controller/cmd_vel` | 默认局部路口的第三台机器人，同上；完整 CPM 另有 `/car_3/controller/cmd_vel` |
| `/car_0/scan_raw`、`/car_1/scan_raw` | `sensor_msgs/msg/LaserScan`，A1 合成扫描 |
| `/car_0/ground_truth`、`/car_1/ground_truth` | `nav_msgs/msg/Odometry`，仿真真值 |
| `/overhead_camera/image_raw` | `sensor_msgs/msg/Image`，俯拍图像 |
| `/overhead_camera/camera_info` | `sensor_msgs/msg/CameraInfo`，相机内参 |
| `/tf`、`/tf_static` | 场地、车体、A1 与相机光学坐标变换 |
| `/clock` | 仿真时钟 |

所有配置车辆均提供相同的 `scan_raw`、`ground_truth` 与坐标变换。

车体速度在本车坐标系表达，+X 前、+Y 左，正角速度逆时针。
话题后缀对应厂家的 ROS2 `controller/cmd_vel` 与 `scan_raw`，增加车辆命名空间用于多车。
参考[官方 ROS2 运动控制课程](https://wiki.hiwonder.com/projects/JetAuto/en/jetauto-orin-nano/docs/2.motion_control.html)。
位姿在 `map` 下表达，车辆参考点为官方 `base_footprint`。
命令接口使用 Twist 无时间戳，超时按接收时墙钟计算；实车接入时应额外处理上游消息时效。
A1 与相机数据可以用于后续算法，但本版控制尚未使用这些数据自动驾驶。

容器默认使用私有网络、`ROS_DOMAIN_ID=57` 和 `ROS_LOCALHOST_ONLY=1`，
当前配置用于本地仿真。后续连接实车前，应单独设置网络、DDS 与话题映射，
不能将仿真工具直接视为实车控制程序。

## 输出与当前边界

```text
deployment/gazebo/generated/vendor/  固定版本的官方机械描述
deployment/gazebo/generated/ros2/    展开后的 URDF、场景和元数据
deployment/gazebo/build/ros2/        本项目 ROS2 底盘插件
deployment/gazebo/runtime/ros2/      图像、运动检查结果和日志
```

当前已有 Gazebo 真值驱动的 [Actor/NOD ROS2 桥接](POLICY_ROS2_BRIDGE.md)，
将 `[v, δ]` 按训练自行车模型的参考点和空间、时间尺度转换为车体 `[vx, vy, ω]`。
尚未接入相机定位滤波、道路视觉建图或实车；零样本策略在简化双车道路口的通行性能需单独验证。

离线检查命令仍是：

```bash
/usr/bin/python3 -m unittest discover -s deployment/gazebo/tests -v
```

2026-11-26 在两车基础交叉口配置上完成镜像构建、ROS2 插件编译与 GUI 运行验证：
六个 Tag 全部检出，两台 A1 都输出 360 点 LaserScan；一次运动检查中前进约 0.249 米、
横移约 0.208 米、原地旋转约 0.691 rad，旋转参考点漂移约 0.092 毫米，
命令断流后速度接近零。该结果仅验证上述理想平面仿真接口，不代表实车精度。
六项离线检查通过。运行结果和带标注图像保存在 `runtime/ros2/`。
