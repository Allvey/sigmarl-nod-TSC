# CPM 局部双车道路口：三台 JetAuto

当前 Gazebo 默认场景为 `CPM_mixed_local_2lane`：保留 CPM_mixed 的路口交互实验形式，
道路改为双向共两车道（每个方向一条），四个方向只保留短引道。三台 JetAuto 的生成、
通行和退出都发生在这个局部区域，没有外围环路。

这是按实车尺寸重新构建的简化路口，使用解析几何生成，**不是原 CPM XML 的等比例裁剪**。
原始训练地图、`utilities/constants.py` 和模型权重没有改动。
需要原始几何时仍可选择[完整三倍 CPM 场景](CPM_MIXED_GAZEBO.md)。

## 启动与演示

在项目根目录执行；同一时间运行一个仿真容器：

```bash
# 如果旧场景正在运行，先停止
bash deployment/gazebo/launch_jetauto.sh stop

# 默认局部双车道路口，打开 Gazebo 窗口
bash deployment/gazebo/launch_jetauto.sh gui

# 或者无窗口运行，仍渲染俯拍图像
bash deployment/gazebo/launch_jetauto.sh
```

默认启动一个**规则通行演示**，使用仿真真值跟踪路线：车辆接近路口后排队，
中央冲突区域一次允许一辆车通过；驶至短引道终点后停车并重新放回原来向的入口。
各车从配置的初始路线开始，循环直行、左转、右转，方便持续检查地图和传感器。
这里的入口重置是规则演示的仿真车流回收。要改由已有 Actor/NOD 权重控制，使用
[策略 ROS2 桥接](POLICY_ROS2_BRIDGE.md)；两者均尚未使用相机定位闭环。

另开终端操作：

```bash
# 检查三车、三台 A1、七个 Tag，并保存俯拍预览
bash deployment/gazebo/launch_jetauto.sh check

# 暂停演示并停车，随后可由外部节点发布 cmd_vel
bash deployment/gazebo/launch_jetauto.sh demo-stop

# 将三车放回初始入口，重新开始规则演示
bash deployment/gazebo/launch_jetauto.sh demo-start

# 以下命令自动暂停演示；不会自动恢复
bash deployment/gazebo/launch_jetauto.sh reset
bash deployment/gazebo/launch_jetauto.sh test --car car_2
bash deployment/gazebo/launch_jetauto.sh drive --car car_2 --vx 0.15 --duration 2
```

接入策略前先执行 `demo-stop`，避免多个控制节点同时发布速度。手动 `drive/test` 是底盘接口检查，
不执行车道跟踪。执行 `reset` 后需要用 `demo-start` 恢复自动演示。

## 尺寸、标线与路线

配置文件为 [`jetauto_local_junction.json`](../../deployment/gazebo/config/jetauto_local_junction.json)，
继承 JetAuto A1 的机械模型、雷达和 ROS2 接口参数。

| 配置项 | 数值 |
| --- | --- |
| 地板范围 | 6×6 米，中心为 `map` 原点 |
| 道路及车辆活动范围 | X、Y 均为 −2.5～2.5 米，十字形道路 |
| 车道数 | 每条道路双向共 2 条，每方向 1 条 |
| 单车道宽 / 总路宽 | 0.45 / 0.90 米 |
| 路缘圆角半径 | 0.35 米 |
| 演示目标速度 | 0.20 m/s |
| 待行位置 | 距中心 1.35 米 |
| 出生位置 | 三车分别距中心 2.10、1.75、1.95 米，见下表 |
| 退出位置 | 对应引道纵向距中心约 2.10 米 |
| 俯拍相机 | 高 5.5 米，2048×2048，水平视场角 1.1 rad，目标 10 Hz |
| 地面 Tag | ID 0～3，位于 (±2.7, ±2.7)，边长 0.20 米 |
| 车顶 Tag | ID 10～12，边长 0.16 米，安装高度参数 0.35 米 |

道路为灰色，外侧边界为白色实线，双向分隔为黄色虚线。路口中央留空；
规划使用的转弯边界不会画到地面上，避免出现截图中的交叉线网。
颜色标线是视觉几何，A1 不能通过雷达识别颜色。

地图导出 4 条入道、4 条出道、12 条虚拟连接道，以及每个来向各直行/左转/右转的 12 条完整路线。
`car_0` 从东侧进入，`car_1` 从北侧进入，`car_2` 从西侧进入，均为右侧通行；南侧保留道路，没有第四台车。
初始路线和出生位置如下，启动模型策略时默认保留各车的路线：

| 车辆 | 来向 | 动作 | 路线索引 | 起点 (X, Y)，米 | 沿引道距中心 |
| --- | --- | --- | --- | --- | --- |
| `car_0` | 东 | 左转，驶向南 | 1 | (2.10, 0.225) | 2.10 米 |
| `car_1` | 北 | 右转，驶向西 | 5 | (−0.225, 1.75) | 1.75 米 |
| `car_2` | 西 | 直行，驶向东 | 6 | (−1.95, −0.225) | 1.95 米 |

`spawn_distance` 是沿整条路线从场地边缘起点前进的弧长，三车分别为 0.40、0.75、0.55 米。
采用固定的不对称布置，便于重复比较模型表现；每次 `reset` 或启动 `policy` 都恢复这组位置。
修改配置中的路线或出生距离后，需要先 `stop` 再 `gui`，让 Gazebo 重新生成场景。

活动范围约束由演示控制器检查；暂停演示后，外部控制器需要自行执行越界和冲突处理。
底盘仍采用理想平面全向运动近似，机械及 ROS2 接口说明见 [JETAUTO_GAZEBO.md](JETAUTO_GAZEBO.md)。

## 输出和检查

```text
deployment/gazebo/generated/ros2/arena.world          当前场景
deployment/gazebo/generated/ros2/cpm_road.dae         简化路口与实际标线
deployment/gazebo/generated/ros2/cpm_map.json         20 条 lanelet、12 条路线及活动范围
deployment/gazebo/generated/ros2/arena_metadata.json  当前有效配置及三车出生位姿
deployment/gazebo/runtime/ros2/camera_tags.png        实际 Gazebo 渲染的 Tag 识别预览
deployment/gazebo/runtime/ros2/scene_check.json       相机、三车和雷达检查
deployment/gazebo/runtime/ros2/traffic_status.json    演示开关、路权、队列、完成次数与故障
deployment/gazebo/runtime/ros2/traffic_check.json     历史四车规则演示运行记录
deployment/gazebo/runtime/ros2/traffic.log            演示日志
```

离线验证：

```bash
/usr/bin/python3 -m unittest discover -s deployment/gazebo/tests -v
```

其中包含双向单车道拓扑、十二条路线的连续性和道路范围、路线弧长投影、出生间距、
相机覆盖、路口中央无路径标线以及道路网格三角形朝向检查。
相机目标帧率和 Gazebo 实时因子受开发电脑渲染性能影响。

2026-11-26 历史四车对称布置验证记录：主机与 ROS2 容器内的 18 项离线检查均通过；实际渲染帧同时检出
0～3、10～13 共八个 Tag，四台 A1 各输出 360 点扫描。四车分别完成直行、左转、右转及入口回收，
该轮运行中车中心坐标绝对值最大约 2.081 米，最小两车中心距约 0.450 米，无演示范围故障。
另验证暂停、四车复位和 `car_3` 手动前进、横移、旋转及断流停车。
