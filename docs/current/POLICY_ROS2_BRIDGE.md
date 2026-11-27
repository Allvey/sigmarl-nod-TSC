# reward7.38 策略接入 ROS2 / Gazebo

桥接将已下载的误差尺度 NOD-DGPPO 控制检查点接到局部双车道路口。
输入为三台 JetAuto 的 Gazebo `ground_truth`，输出为各车的 `controller/cmd_vel`。
Actor、消息网络和 NOD 使用保存权重进行确定性推理，不执行训练，也不使用规则路权队列。

## 运行

在项目根目录执行。Gazebo 使用已配置的 ROS2 Humble 容器；策略使用主机已有的 `sigmarl`
Python 环境，两者通过仅本机可用、权限为 0600 的 Unix socket 交换 JSON。
无需在 Gazebo 镜像中安装 PyTorch。

```bash
# 先启动局部路口；已有仿真运行时跳过这一步
bash deployment/gazebo/launch_jetauto.sh gui
```

重复执行 `gui` 会复用本项目已经运行的同配置容器：窗口已打开时直接提示，窗口已关闭时
重新连接现有仿真，不重置车辆。不同配置会提示先停止旧场景；无窗口模式的容器需要先停止，
再用 `gui` 启动，才能获得 X11 显示挂载。

保持这个终端运行，等待场景加载后，**另开第二个终端**进入项目根目录执行：

```bash
# 首先执行真实模型控制：仿真真值输入，三车运行 10 秒仿真时间
bash deployment/gazebo/launch_jetauto.sh policy --duration 10

# 继续观察较长运行，最多 60 秒；风险保护触发时会提前停止
bash deployment/gazebo/launch_jetauto.sh policy

# 检查当前运行记录 / 停止正在运行的策略
bash deployment/gazebo/launch_jetauto.sh policy-status
bash deployment/gazebo/launch_jetauto.sh policy-stop
```

当前主线是 `Gazebo 真值 → 观测与 NOD 历史 → reward7.38 Actor → ROS2 Twist → JetAuto`。
运行策略不依赖相机 Tag 定位，也不需要先启动训练。相机与雷达可继续显示，但不会作为本版策略输入。
若只想排查通信而不发车辆指令，可选用 `policy --dry-run --duration 3`。

默认 `--maneuver configured` 保留每车配置，当前分别为 `car_0` 左转、`car_1` 右转、
`car_2` 直行。三车出生位置沿引道距中心分别为 2.10、1.75、1.95 米，
详见[局部场景配置](LOCAL_JUNCTION_GAZEBO.md)。这些路线同时用于策略观测、进度和终点判断，
可在启动输出、`policy_manifest.json` 和 `policy_status.json` 的 `vehicle_routes` 中核对。

`policy` 连接已经运行的 Gazebo 容器。若遇到 `No such object: sigmarl-jetauto-humble`，
说明需要先在第一个终端执行上述 `gui` 命令。仿真容器带有 `--rm`，停止后会被自动删除；
之后再次运行策略需要重新启动场景，不需要重建已有镜像或重新下载权重。

默认模型目录为
`checkpoints/nod_error_scale_repeat/seed5080868027432654403/02_control/`。
加载最高训练奖励的检查点及同名前缀的 JSON、NOD、Safety Value；本次为 `reward7.38`。
策略工厂还会校验所选前缀与元数据一致。Safety Value 是训练用的影子网络，不参与在线动作筛选。

可选参数：

```bash
# 指定推理解释器；也可设置 SIGMARL_POLICY_PYTHON
bash deployment/gazebo/launch_jetauto.sh policy --python /path/to/sigmarl/bin/python

# 先只让一台车运动，其余两台保持停车并继续参与观测
bash deployment/gazebo/launch_jetauto.sh policy --cars car_0 --duration 45

# 显式覆盖所有车为左转；straight/right 同样会统一覆盖，不是混合模式
bash deployment/gazebo/launch_jetauto.sh policy --maneuver left --duration 60

# 降低最高平移速度，并相应调整时间尺度
bash deployment/gazebo/launch_jetauto.sh policy --speed-limit 0.15 --duration 60
```

启动会停止已有策略、暂停规则演示，并通过 `/reset_simulation` 复位三车和仿真时间。
同一时间只允许一个策略启动器。`drive/test/reset/demo-start` 会先停止策略，避免竞争发布速度。
退出、到达全部目标或到时后停车；不会自动恢复规则演示。需要时另行执行 `demo-start`。
本版本到短引道出口后停车，不自动生成新一轮车流。

## 状态、观测与历史

ROS2 节点订阅 `/car_N/ground_truth`，要求父坐标系为 `map`，车辆参考点为
`car_N/base_footprint`；线速度按 Odometry 的子坐标系语义解释为车体速度。
节点检查有限数值、各车时间同步、状态新鲜度，以及请求和响应的序号、会话及时间戳。

主机推理器构造一个**外部状态镜像**：每帧覆盖三车位姿和测得速度，使用生成的局部路线、
实际 JetAuto 外廓和训练时相同的归一化、近邻选择、32 维观测及 NOD 物理输入。
路线在训练单位中按原 CPM 地图典型采样间隔重采样。规划用的虚拟左右边界供策略计算距离，
不会画成 Gazebo 路口内的交叉标线。

调用原 `ScenarioRoadTraffic` 的观测函数和原 Actor/NOD 在线路径，避免重新定义特征顺序。
推理过程中不调用 VMAS 的 `step`，不使用预测状态代替 ROS2 测量值。
NOD 的循环状态跨帧保留；新会话或时间倒退会清空历史。上一动作来自实际发布的速度指令，
经逆映射还原为训练动作单位。模型参数和训练地图文件保持不变。

虽然检查点由四车训练得到，Actor 与消息网络是共享参数的，三车仍各观察两个近邻，
每车输入保持 32 维。加载时按三车构造批次和 NOD 身份图，并严格校验现有权重；
不添加虚拟第四台车。车辆数量来自部署配置，不改训练用 `utilities/constants.py`。

## 空间、时间与动作单位

单车道宽 0.45 米对应训练中的 0.15 米，默认空间比例 `S=3`。
最高训练动作速度为 1 m/s，默认 Gazebo 平移上限为 0.20 m/s，采用时间比例
`T=S×1/0.20=15`：

```text
p_model = p_gazebo / S
v_model = v_gazebo × T / S
ω_model = ω_gazebo × T
Δt_gazebo = Δt_model × T = 0.05 × 15 = 0.75 s
```

策略约每 0.75 秒仿真时间更新一次；ROS2 节点按 20 Hz 墙钟发送当前指令，维持底盘看门狗。
Gazebo 变慢时不把墙钟当作模型时间。控制频率与发布频率不同。

Actor 的 `[v, δ]` 是训练自行车模型**质心处**的目标速度和虚拟转向角，单位分别为 m/s、rad。
复用 `utilities/kinematic_bicycle.py` 的 `l_f=l_r=0.08 m`：

```text
β  = atan(l_r/(l_f+l_r) × tan(δ))
vx = S/T × v × cos(β)
vy = S/T × v × sin(β)
ω  = v × cos(β) × tan(δ) / ((l_f+l_r) × T)
```

三个分量映射到 JetAuto `base_footprint` 的 Twist；限幅时整体缩放，保留曲率与横移比例。
虚拟轴距用于复现策略的运动学语义，不表示 JetAuto 的实测轴距。
传入观测的实际车长宽除以 S；它与训练小车的长宽比例仍有差别，新道路也属于简化几何，
因此这里验证的是闭环迁移链路，通行性能需要在新场景中单独测量。

## 停车条件与输出

节点对状态超时、推理超时、非法动作、运行中复位、车辆外廓出路面，以及短时预测车车接触
执行锁存停车并结束本轮。原底盘还保留 0.35 秒命令超时和加速度限制。
接触检查包含当前速度的制动过渡，采用矩形外廓及短时速度外推；它是额外的仿真停车保护，
不是完整碰撞避免控制器。保护触发会记录为 `fault`，不能计为策略自主通行成功。
出路面时 `guard_details` 保存触发时的位置、航向和越界角点；车辆中心在路内时，旋转后的角点仍可能越界。
推理响应还保存实际 32 维 `observations` 和 `reference_points_model`，可与动作和运动轨迹一起核对。

```text
deployment/gazebo/runtime/ros2/policy_manifest.json  检查点 SHA256、场景校验、观测维度和尺度
deployment/gazebo/runtime/ros2/policy_status.json    活动状态、指令、测量、完成车辆、停车原因
deployment/gazebo/runtime/ros2/policy_trace.jsonl    每帧请求、原始策略动作、NOD 诊断与最终速度
deployment/gazebo/runtime/ros2/policy_worker.log     主机推理日志
deployment/gazebo/runtime/ros2/policy_bridge.log     ROS2 接口日志
```

这些运行文件会被下一次策略运行替换，需要保留实验记录时先复制完整目录。
当前输入明确为仿真真值；俯拍 Tag 定位、延迟和测量噪声尚未接入这条桥接。

## 检查

几何、传输协议和动作单位检查使用系统 Python；检查点集成使用主机推理环境：

```bash
/usr/bin/python3 -m unittest discover -s deployment/gazebo/tests -v
/home/allvey/anaconda3/envs/sigmarl/bin/python -m unittest discover \
  -s deployment/gazebo/tests -p 'test_policy*.py' -v
```

检查点集成测试覆盖实际权重加载、有限观测、NOD 在线交互、历史重置、参数冻结，
并阻止推理期间调用 VMAS 物理步进。

2026-11-26 最初对称直行布置的运行验证：

- 3 秒 dry-run：四帧推理完成，没有发布速度指令。
- 5 秒四车闭环：七帧推理，四台车各移动约 0.85 米，最高指令速度约 0.196 m/s，
  推理计算最长约 17 ms；到时停止且四车测量速度满足停稳阈值。
- 强制终止本轮推理子进程：ROS2 检测到连接断开，锁存故障并确认四车停稳。
- `policy-stop`：四车停稳，主机与 ROS2 策略进程正常退出。
- 较长的四车测试约在第十八帧触发 `car_2/car_3` 预测接触保护；单车测试约在第二十一帧
  触发外廓出路面保护，停稳验证通过。**这些不是路口通行成功结果**，表明旧权重在本简化场景
  仍需适配或微调；不能因桥接已运行而认为策略已经完成仿真迁移。

各轮记录分别保存在 `runtime/ros2/policy_validation/` 的 `four_car_short`、`worker_disconnect`、
`operator_stop`、`four_car_guard` 和 `single_car_guard` 子目录中。

历史四车布置改为混合路线和非对称起点后，已核对 Gazebo 复位位置、策略路径及各车终点一致。
本轮四车实际运行完成 9 次推理后，`car_0` 触发出路面保护，四车停稳；记录位于
`runtime/ros2/policy_validation/mixed_asymmetric/`。这轮验证证明配置已生效，不表示模型完成通行。

按实际硬件数量改为三车后，已严格加载同一 `reward7.38` 权重，并验证观测为 `1×3×32`。
三车 10 秒运行完成 14 次推理、无故障、到时停稳，记录在 `policy_validation/three_car_short/`。
较长运行在第 19 次推理后触发 `car_2` 出路面保护，记录在 `policy_validation/three_car_probe/`，
包含触发角点与指令跟踪误差。该结果说明三车接入完成，长时间通行的策略越界仍未解决。
