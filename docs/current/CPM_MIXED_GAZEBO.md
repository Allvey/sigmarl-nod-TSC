# CPM_mixed：三倍地图＋四台 JetAuto

本可选场景为项目的完整 `CPM_mixed`，按选定比例放大 3 倍，使用
JetAuto Orin Nano Super 4GB＋A1 配置与 ROS2 Humble。仿真在开发电脑的 Docker 中运行。
当前默认已改为[局部双车道路口](LOCAL_JUNCTION_GAZEBO.md)。

## 启动

在项目根目录执行：

```bash
# 图形界面
bash deployment/gazebo/launch_jetauto.sh gui --config jetauto_cpm_mixed.json

# 无图形窗口；仍通过容器内虚拟显示渲染俯拍相机
bash deployment/gazebo/launch_jetauto.sh --config jetauto_cpm_mixed.json
```

一个时间点只运行一个 `sigmarl-jetauto-humble` 容器。切换场景前先停止当前实例：

```bash
bash deployment/gazebo/launch_jetauto.sh stop

# 显式选择完整 CPM_mixed
bash deployment/gazebo/launch_jetauto.sh gui --config jetauto_cpm_mixed.json

# 原来的两车简单交叉口
bash deployment/gazebo/launch_jetauto.sh gui --config jetauto_a1.json
```

控制、检查和复位自动读取运行场景的元数据，无需再指定配置：

```bash
# 检查 4 台车、4 台雷达、4 个地面 Tag＋4 个车顶 Tag
bash deployment/gazebo/launch_jetauto.sh check

# 分别控制任一台车；速度以本车坐标系表达
bash deployment/gazebo/launch_jetauto.sh drive --car car_3 --vx 0.15 --duration 2

# 验证指定车辆的运动接口；此检查不会自动沿车道行驶
bash deployment/gazebo/launch_jetauto.sh test --car car_3

# 恢复四车初始位置和仿真时钟
bash deployment/gazebo/launch_jetauto.sh reset
```

## 地图与坐标

配置入口是 [`jetauto_cpm_mixed.json`](../../deployment/gazebo/config/jetauto_cpm_mixed.json)，
通过 `extends` 继承 [`jetauto_a1.json`](../../deployment/gazebo/config/jetauto_a1.json) 的车辆和 ROS2 接口参数。

地图直接解析 [`assets/maps/cpm.xml`](../../assets/maps/cpm.xml)。场地边界、原始车道宽度和车辆数
取自 [`SCENARIOS["CPM_mixed"]`](../../utilities/constants.py)，路线定义直接读取
[`ParseXML`](../../utilities/parse_xml.py) 中的字面量。导入过程不执行训练模块，也不依赖 Torch。
Docker 以只读方式挂载这两处源目录。

| 项目 | 原始 CPM_mixed | 本 Gazebo 场景 |
| --- | --- | --- |
| 地图坐标范围 | X=0～4.5，Y=0～4 米 | X=−6.75～6.75，Y=−6～6 米 |
| 地图范围尺寸 | 4.5×4 米 | 13.5×12 米 |
| 标称单车道宽 | 0.15 米 | 0.45 米 |
| 含四角 Tag 留白的地板 | — | 15×13.5 米 |
| lanelet 数 | 104 | 104 |
| JetAuto 数 | 配置为 4 辆智能体 | 4 台真实尺寸模型，未放大车体 |

原始坐标到 Gazebo `map` 的变换为：

```text
x_map = 3 × (x_CPM − 2.25)
y_map = 3 × (y_CPM − 2.00)
yaw_map = yaw_CPM
```

导出保留每个 lanelet 的左右边界、中心线、前驱、后继、邻接关系、道路类型和线型，
并记录源 XML 的 SHA256 与上述坐标变换。曲线按原始采样点生成道路带状网格，
使用单个 Collada 网格降低场景对象数量。白色实线和黄色虚线取自原始边界线型。
路口显示的交叉曲线包括原始 lanelet 的路径边界。

该版本的道路和标线是地板上的视觉几何，没有增加实体路沿。雷达不能检测颜色标线；
后续车道识别使用俯拍图像，道路越界判定使用导出的几何。

## 车辆与路线

原始路线完整导出：24 条交叉口路线、4 条汇入路线、4 条分流路线，
以及由 7 个基础环路得到的 40 条环行参考路线。

四台车默认分布在中心交叉口的四个来向，其位置沿参考路线弧长采样，车头与路径切线一致：

| 车辆 | 车顶 Tag | 分配的 lanelet 路线 |
| --- | --- | --- |
| `car_0` | 10 | 11 → 26 → 52 → 37 |
| `car_1` | 11 | 39 → 50 → 102 → 91 |
| `car_2` | 12 | 89 → 104 → 78 → 63 |
| `car_3` | 13 | 65 → 76 → 24 → 13 |

`route_index` 使用从 0 开始的索引；`spawn_distance` 为从整条路线起点前进的弧长，单位为放大后的米。
初始状态是可复现的固定布置，不复刻训练环境的随机采样。分配路线当前仅用于生成位置和导出地图；
车辆启动后保持静止，尚未接入路线跟踪器、Actor 或 NOD。

每车保留独立的接口：

```text
/car_N/controller/cmd_vel    geometry_msgs/msg/Twist
/car_N/ground_truth          nav_msgs/msg/Odometry
/car_N/scan_raw              sensor_msgs/msg/LaserScan
```

`N` 为 0～3。车辆动力学近似、限幅、命令超时停车与基础环境一致，详见
[JetAuto Gazebo 运行层](JETAUTO_GAZEBO.md)。

## 俯拍相机与 Tag

- 相机高度 10.5 米，分辨率 3072×2816，水平视场角 1.25 rad，目标更新率 5 Hz。
- 四个地面 Tag 的 ID 为 0～3，位于道路范围外的四角留白，黑色边框边长 0.30 米。
- 四个车顶 Tag 的 ID 为 10～13，黑色边框边长 0.20 米，带留白总宽约 0.25 米。
- Tag 家族为 AprilTag 36h11，车顶安装高度参数为 0.35 米。

这些为全场单相机仿真配置。实际场地的相机安装高度、分辨率和更新率需要实测确认。
软件渲染的实际帧率可能低于目标更新率。

## 输出与验证

```text
deployment/gazebo/generated/ros2/arena.world          Gazebo 场景
deployment/gazebo/generated/ros2/cpm_road.dae         原始几何生成的道路网格
deployment/gazebo/generated/ros2/cpm_map.json         lanelet、路线与坐标变换
deployment/gazebo/generated/ros2/arena_metadata.json  车辆初始状态、相机、Tag 与配置
deployment/gazebo/runtime/ros2/camera_tags.png        Gazebo 实际渲染的识别预览
deployment/gazebo/runtime/ros2/scene_check.json       车辆、雷达与 Tag 检查结果
```

离线验证：

```bash
/usr/bin/python3 -m unittest discover -s deployment/gazebo/tests -v
```

测试将全部左右边界与源 XML 的 3 倍坐标逐点比较，核对路线、出生位姿、相机覆盖、Tag 留白
和网格三角形朝向。完整地图版本的 12 项离线检查通过。Gazebo 运行中已验证 8 个 Tag 全部检出，
4 台 JetAuto 与 4 台 A1 话题正常，并在 `car_3` 上完成运动、旋转与断流停车检查。

本次未改变 VMAS 训练地图、车辆常量或模型权重。地图放大、车辆外形和控制动力学发生变化后，
策略接入仍需单独处理观测、动作和时间尺度，不能将场景运行成功视为权重已经迁移成功。
