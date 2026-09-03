# SigmaRL Traffic 强化学习环境设计说明

本文档基于当前仓库代码与根目录 `config.json`，梳理该项目中的强化学习智能体状态、动作、奖励机制，以及关键环境参数、训练参数、车辆参数和拓扑/对手建模相关参数。

核心代码位置：

- 环境定义：`scenarios/road_traffic.py`
- 训练入口：`main_training.py`
- MAPPO/PPO 训练逻辑：`utilities/mappo_cavs.py`
- 参数类：`utilities/helper_training.py`
- 车辆运动学模型：`utilities/kinematic_bicycle.py`
- 地图与车辆常量：`utilities/constants.py`

## 1. 项目整体定位

该项目是一个面向联网自动驾驶车辆（CAVs）的多智能体强化学习交通运动规划环境。仿真基于 VMAS，训练基于 TorchRL，策略使用连续动作空间。

当前训练框架可以概括为：

- 多智能体环境：每辆车是一个 agent。
- 策略 Actor：分散式执行，每个 agent 只使用自己的观测。
- Critic：MAPPO 风格中心化 critic。
- 动作空间：连续二维动作，速度命令 + 转向角命令。
- 车辆动力学：kinematic bicycle model。
- 支持模块：拓扑关系学习、邻居动作预测、对手建模、优先级 MARL。

当前 `config.json` 中：

- `scenario_type = "CPM_mixed"`
- `is_partial_observation = true`
- `n_nearing_agents_observed = 3`
- `is_using_opponent_modeling = true`
- `use_topology_neighbor_selection = true`
- `is_using_prioritized_marl = false`

注意：`config.json` 中写了 `n_agents = 6`，但 `CPM_mixed` 在 `utilities/constants.py` 中定义的 agent 数是 4。环境初始化时会根据 `scenario_type` 使用场景常量覆盖 agent 数，因此当前实际 agent 数应按 4 理解。

## 2. 智能体状态 / 观测设计

观测由 `ScenarioRoadTraffic.observation()` 生成。每个 agent 的观测由两部分组成：

1. 自车观测
2. 周围车辆观测

如果启用对手建模，还会在观测尾部追加邻居动作占位。

### 2.1 坐标系设计

项目支持两种观测坐标系：

| 参数 | 含义 |
|---|---|
| `is_ego_view = true` | 使用 ego-view，即以自车为坐标原点和朝向基准 |
| `is_ego_view = false` | 使用 bird-view/global-view，即全局坐标 |

当前配置为 `is_ego_view = true`。

在 ego-view 下：

- 自车位置与朝向不再显式进入观测，因为自车局部坐标中自车始终位于原点。
- 其他车辆位置、顶点、速度、参考路径等会转换到自车局部坐标。
- 自车速度只取前向速度分量。

### 2.2 自车观测

自车观测由 `_observe_self()` 生成。

当前配置下，自车观测包含：

| 项 | 维度 | 说明 |
|---|---:|---|
| 自车速度 | 1 | ego-view 下只取前向速度 |
| 短期参考路径 | `n_points_short_term * 2 = 6` | 当前为 3 个未来参考点，每点 `(x, y)` |
| 到中心线距离 | 1 | `is_observe_distance_to_center_line = true` |
| 到左边界距离 | 1 | `is_observe_distance_to_boundaries = true` |
| 到右边界距离 | 1 | `is_observe_distance_to_boundaries = true` |

因此当前自车观测维度为：

```text
1 + 3 * 2 + 1 + 1 + 1 = 10
```

如果 `is_ego_view = false`，自车位置和朝向也会进入观测。

如果 `is_observe_distance_to_boundaries = false`，观测不再使用边界距离，而是使用近邻边界点坐标。

### 2.3 周围车辆观测

周围车辆观测由 `_observe_other_agents()` 生成。

当前配置：

- `is_partial_observation = true`
- `n_nearing_agents_observed = 3`
- `is_observe_vertices = true`
- `is_observe_distance_to_agents = true`
- `is_observe_ref_path_other_agents = false`

每个邻居包含：

| 项 | 维度 | 说明 |
|---|---:|---|
| 邻居车辆矩形 4 个顶点 | 8 | 每个顶点 `(x, y)` |
| 邻居速度 | 2 | ego-view 下的相对朝向速度表示 |
| 到邻居距离 | 1 | 当前使用 c2c 距离 |

每个邻居维度为：

```text
8 + 2 + 1 = 11
```

当前观测 3 个邻居，因此邻居观测维度为：

```text
3 * 11 = 33
```

### 2.4 部分观测与邻居选择

项目有两种邻居选择机制。

默认机制：

- 按 agent 间距离排序。
- 选最近的 `n_nearing_agents_observed` 个 agent。
- 如果 `is_apply_mask = true`，还会按距离阈值和 lanelet 邻接关系进行 mask。

拓扑选择机制：

- 由 `use_topology_neighbor_selection` 控制。
- 当前配置为 `true`。
- 如果环境中存在 `topology_learner`，会先取距离最近的候选集，再由拓扑模型输出边概率。
- 按拓扑概率筛选，再按距离稳定排序，最终选择策略观测用的 Top-K 邻居。
- 当前 `topology_selection_threshold = 0.0`，意味着只要模型可用，通常会尽量选满 K 个邻居。

当前配置中：

```text
n_nearing_agents_observed = 3
n_topology_nearing_agents_observed = 5
```

但 `CPM_mixed` 实际 agent 数为 4，因此可用邻居最多是 `n_agents - 1 = 3`。所以 topology 候选数最终也会被限制为 3。

### 2.5 对手建模观测尾部

当前配置：

```text
is_using_opponent_modeling = true
```

因此环境原始观测会在尾部追加：

```text
n_nearing_agents_observed * AGENTS["n_actions"]
= 3 * 2
= 6
```

这 6 维是邻居动作占位，每个邻居 2 维动作。

训练时：

- actor 的观测尾部会被置零。
- critic 的 `critic_observation` 尾部会填入预测的邻居动作。
- 如果存在 topology action predictor，则使用拓扑动作预测器预测邻居动作。
- 如果预测器不可用，则 critic 尾部也可能退化为零。

### 2.6 当前 actor 输入维度

基于当前配置：

| 部分 | 维度 |
|---|---:|
| 自车观测 | 10 |
| 3 个邻居观测 | 33 |
| 对手建模动作占位 | 6 |

总维度：

```text
10 + 33 + 6 = 49
```

因此当前每个 agent 的 actor 输入约为 49 维。

### 2.7 观测归一化

环境会将观测归一化后送入网络。

| 变量 | 归一化尺度 |
|---|---|
| ego-view 位置 / 参考点 | `[agent_length * 10, agent_length * 10] = [1.6, 1.6]` |
| bird-view 位置 | `[world_x_dim, world_y_dim]` |
| 速度 | `max_speed = 1.0` |
| 朝向 | `2π` |
| 转角动作 | `max_steering_angle` |
| 速度动作 | `max_speed` |
| lanelet/参考线距离 | `lane_width * 3` |
| agent 间距离 | `agent_length * 10` |

## 3. 动作设计

每个 agent 的动作空间是连续二维：

```text
action = [v_command, steering_command]
```

| 动作分量 | 含义 | 范围 |
|---|---|---|
| `action[0]` | 速度命令 | `[-max_speed, max_speed]` |
| `action[1]` | 转向角命令 | `[-max_steering_angle, max_steering_angle]` |

当前车辆参数：

```text
max_speed = 1.0 m/s
max_steering = 35 deg = 0.6109 rad
```

环境创建 agent 时设置：

```python
u_range = [self.max_speed, self.max_steering_angle]
u_multiplier = [1, 1]
```

策略网络输出动作分布：

- 网络输出 `2 * action_dim = 4` 维。
- 使用 `NormalParamExtractor` 拆成 `loc` 和 `scale`。
- 使用 `TanhNormal` 分布采样动作。
- `TanhNormal` 的上下界来自环境 action spec。

## 4. 车辆动力学模型

车辆使用 kinematic bicycle model。

车辆状态核心变量：

```text
state = [x, y, theta]
```

动作：

```text
v = v_command
delta = steering_command
```

滑移角：

```text
beta = atan(tan(delta) * l_r / (l_f + l_r))
```

状态导数：

```text
dx     = v * cos(theta + beta)
dy     = v * sin(theta + beta)
dtheta = v / (l_f + l_r) * cos(beta) * tan(delta)
```

当前默认积分方法：

```text
integration = "rk4"
```

动作处理流程：

1. 从 agent action 中取速度命令和转向命令。
2. 将速度 clamp 到 `[-max_speed, max_speed]`。
3. 将转向 clamp 到 `[-max_steering_angle, max_steering_angle]`。
4. 用 kinematic bicycle 模型计算状态变化。
5. 转换成 VMAS 需要的 force 和 torque。

## 5. 奖励设计

奖励函数在 `ScenarioRoadTraffic.reward()` 中实现。每个 agent 独立计算 reward，最后将 reward clamp 到：

```text
[-1, 1]
```

奖励项包括：

1. 沿参考路径前进奖励
2. 高速度奖励
3. 到达目标奖励
4. 靠近车道边界惩罚
5. 靠近其他车辆惩罚
6. 偏离参考路径惩罚
7. 转向变化过快惩罚
8. 与其他车辆碰撞惩罚
9. 与车道边界碰撞惩罚
10. 时间/速度方向项

### 5.1 奖励与惩罚系数

代码中使用：

```text
r_p_normalizer = 100
```

默认奖励系数如下：

| 项 | 原始值 | 实际系数 |
|---|---:|---:|
| `reward_progress` | 10 | 0.10 |
| `reward_vel` | 5 | 0.05 |
| `reward_reach_goal` | 0 | 0 |
| `penalty_deviate_from_ref_path` | -2 | -0.02 |
| `penalty_near_boundary` | -20 | -0.20 |
| `penalty_near_other_agents` | -20 | -0.20 |
| `penalty_collide_with_agents` | -100 | -1.00 |
| `penalty_collide_with_boundaries` | -100 | -1.00 |
| `penalty_change_steering` | -2 | -0.02 |
| `penalty_time` | 5 | 0.05 |

### 5.2 前进奖励

前进奖励衡量车辆当前位置相对上一时刻位置的位移，在短期参考路径方向上的投影。

计算逻辑：

```text
move_vec = current_pos - previous_pos
ref_points_vecs = short_term_ref_points - previous_pos
move_projected = dot(move_vec, ref_points_vecs)
move_projected_weighted = weighted_sum(move_projected)
reward_movement = move_projected_weighted / (max_speed * dt) * reward_progress
```

短期参考点越近，权重越大。当前 `n_points_short_term = 3`，权重从 1 到 0.2 线性递减后归一化。

### 5.3 高速度奖励

速度奖励基于当前速度在参考路径方向上的投影。

```text
v_proj = mean(dot(agent_vel, ref_points_vecs))
reward_vel = factor * v_proj / max_speed * reward_vel_coef
```

如果 `v_proj > 0`，说明沿参考路径正向运动。

如果 `v_proj <= 0`，说明逆向或无正向运动，代码中使用更大的系数让该项变成更强的负向反馈。

### 5.4 到达目标奖励

如果 agent 与 exit segment 相交，会触发到达目标奖励：

```text
reward_goal = collision_with_exit_segment * reward_reach_goal
```

当前默认 `reward_reach_goal = 0`，所以该项目前不产生实际奖励。

### 5.5 靠近车道边界惩罚

车道边界惩罚基于车辆到左右边界的最小距离。车辆距离边界越近，惩罚越大。

惩罚函数：

```text
penalty_close_to_lanelets =
    exponential_decreasing_fcn(distance_to_boundary, low, high)
    * penalty_near_boundary
```

`exponential_decreasing_fcn` 会在 `[low, high]` 区间内从 1 指数下降到 0。

### 5.6 靠近其他车辆惩罚

对 ego agent 到所有其他 agent 的距离分别计算指数惩罚，然后求和。

```text
penalty_close_to_agents =
    sum(exponential_decreasing_fcn(distance_to_agents, low, high))
    * penalty_near_other_agents
```

当前 `is_use_mtv_distance = false`，因此使用中心点距离 `c2c`。

如果 `is_use_mtv_distance = true`，则使用 MTV-based distance。

### 5.7 偏离参考路径惩罚

偏离参考路径惩罚是线性的：

```text
penalty_deviation =
    distance_to_ref_path / threshold_deviate_from_ref_path
    * penalty_deviate_from_ref_path
```

偏离越大，惩罚越大。

### 5.8 转向变化过快惩罚

该项惩罚相邻时刻转向命令变化过大：

```text
steering_change =
    abs(current_steering - previous_steering) * max_steering_angle
    - threshold_change_steering
```

只有当变化量超过阈值后才惩罚。

当前默认：

```text
threshold_change_steering = 10 deg
```

### 5.9 碰撞惩罚

碰撞检测不是依赖 VMAS 的物理碰撞响应。环境中 `Agent(collide=False)`，但代码手动检测碰撞并用于 reward 和 done。

车辆间碰撞：

- 将车辆视为矩形。
- 计算车辆四个顶点。
- 使用矩形边界相交或 MTV 距离判断碰撞。
- 任意车辆碰撞给 `penalty_collide_with_agents = -1.0`。

车道边界碰撞：

- 检测车辆矩形边界是否与 left/right lane boundary 相交。
- 撞边界给 `penalty_collide_with_boundaries = -1.0`。

### 5.10 时间/速度方向项

该项代码注释称为 time reward/penalty，但实际机制是：

```text
time_reward =
    sign(v_proj) * speed_norm / max_speed * penalty_time
```

如果车辆沿参考路径方向运动，该项为正。

如果车辆反向运动，该项为负。

## 6. 关键阈值

当前 `CPM_mixed` 场景参数：

```text
lane_width = 0.15 m
agent_width = 0.08 m
agent_length = 0.16 m
```

主要阈值：

| 阈值 | 计算方式 | 当前值 |
|---|---|---:|
| 偏离参考路径阈值 | `(lane_width - agent_width) / 2` | 0.035 m |
| 到达目标阈值 | `agent_width / 2` | 0.04 m |
| 转向变化阈值 | 10 deg | 0.1745 rad |
| 靠近边界 low | 0 | 0 m |
| 靠近边界 high | `(lane_width - agent_width) / 2 * 0.9` | 0.0315 m |
| 靠近其他车 c2c low | `(agent_length + agent_width) / 2` | 0.12 m |
| 靠近其他车 c2c high | `agent_length + agent_width` | 0.24 m |
| 距离 mask 阈值 | `agent_length * 5` | 0.8 m |
| reset 最小初始距离 | `1.2 * sqrt(length^2 + width^2)` | 约 0.2147 m |

## 7. Episode 终止与 reset 机制

`done()` 中定义了终止逻辑。

非测试模式下，整个环境 done 的条件：

- 达到最大步数。
- 任意 agent 与其他 agent 碰撞。
- 任意 agent 与 lanelet 边界碰撞。

对于非 `CPM_entire` 场景：

- 如果 agent 碰到 entry segment 或 exit segment，可以单独 reset 该 agent。
- 如果发生车辆间碰撞或边界碰撞，则整个 env done。

测试模式下：

- 只有达到最大步数才整体 done。
- 碰撞或离开入口/出口的 agent 会被单独 reset。

当前训练配置：

```text
is_testing_mode = false
max_steps = 128
```

## 8. 初始状态设计

reset 时每个 agent 会随机初始化：

1. 随机选择一条参考路径。
2. 随机选择参考路径上的一个点。
3. 设置位置为该参考点。
4. 设置朝向为参考路径中心线 yaw。
5. 随机设置初始速度，速度大小在 `[0, max_speed]`，方向沿参考路径 yaw。
6. 检查与其他 agent 的初始距离，保证大于 reset 最小安全距离。

对于 `CPM_mixed`：

- 随机点不会从路径起点附近选。
- 会偏向使用路径中前段，使 agent 更容易发生交互。

如果 `is_challenging_initial_state_buffer = true`：

- 碰撞前的困难状态可能被记录。
- reset 时有概率从困难状态 buffer 中采样。

当前配置：

```text
is_challenging_initial_state_buffer = false
```

## 9. 车辆参数

车辆参数定义在 `utilities/constants.py` 的 `AGENTS`。

| 参数 | 当前值 | 说明 |
|---|---:|---|
| `width` | 0.08 m | 车宽 |
| `length` | 0.16 m | 车长 |
| `l_f` | 0.08 m | 前轴到质心距离 |
| `l_r` | 0.08 m | 后轴到质心距离 |
| `max_speed` | 1.0 m/s | 最大速度命令 |
| `max_speed_achievable` | 0.82 m/s | 记录的可达最大速度 |
| `max_steering` | 35 deg | 最大转向角 |
| `n_actions` | 2 | 动作维度 |

## 10. 当前地图 / 场景参数

当前 `scenario_type = "CPM_mixed"`。

`CPM_mixed` 参数：

| 参数 | 值 |
|---|---|
| `map_path` | `assets/maps/cpm.xml` |
| `n_agents` | 4 |
| `name` | `CPM Map` |
| `x_dim_min` | 0 |
| `x_dim_max` | 4.5 |
| `y_dim_min` | 0 |
| `y_dim_max` | 4.0 |
| `world_x_dim` | 4.5 |
| `world_y_dim` | 4.0 |
| `figsize_x` | 3 |
| `viewer_zoom` | 1.44 |
| `lane_width` | 0.15 m |
| `scale` | 1.0 |

`CPM_mixed` 内部包含三类子场景：

| `scenario_id` | 子场景 |
|---:|---|
| 1 | intersection |
| 2 | merge-in |
| 3 | merge-out |

当前概率：

```json
"cpm_scenario_probabilities": [1.0, 0.0, 0.0]
```

因此当前只采样 intersection 子场景。

## 11. 训练参数

当前 `config.json` 中训练参数：

| 参数 | 当前值 | 说明 |
|---|---:|---|
| `n_iters` | 250 | 训练迭代数 |
| `frames_per_batch` | 4096 | 每轮采样 frame 数 |
| `num_epochs` | 60 | 每 batch 优化 epoch 数 |
| `minibatch_size` | 512 | minibatch 大小 |
| `lr` | 2e-4 | 主 PPO 学习率 |
| `lr_action_predictor` | 3e-4 | 动作预测器学习率 |
| `lr_min` | 1e-5 | 学习率衰减下限 |
| `max_grad_norm` | 1.0 | 梯度裁剪上限 |
| `clip_epsilon` | 0.2 | PPO clip 系数 |
| `gamma` | 0.99 | 折扣因子 |
| `lmbda` | 0.9 | GAE lambda |
| `entropy_eps` | 1e-4 | 熵正则系数 |
| `topology_loss_weight` | 0.5 | 拓扑 BCE loss 权重 |
| `max_steps` | 128 | 每个 episode 最大步数 |

派生参数：

```text
num_vmas_envs = frames_per_batch // max_steps
              = 4096 // 128
              = 32
```

## 12. 观测相关参数

| 参数 | 当前值 | 说明 |
|---|---:|---|
| `is_partial_observation` | true | 是否只观测近邻 |
| `n_steps_stored` | 10 | 状态 buffer 存储步数 |
| `n_points_short_term` | 3 | 短期参考路径点数 |
| `is_append_current_pos_to_short_refs_for_topology` | true | topology 参考路径中是否拼接当前位置 |
| `n_nearing_agents_observed` | 3 | 策略观测近邻数 |
| `n_topology_nearing_agents_observed` | 5 | topology 候选近邻数 |
| `is_ego_view` | true | 是否使用 ego-view |
| `is_observe_vertices` | true | 是否观测邻车顶点 |
| `is_observe_distance_to_agents` | true | 是否观测到邻车距离 |
| `is_observe_distance_to_boundaries` | true | 是否观测到左右边界距离 |
| `is_observe_distance_to_center_line` | true | 是否观测到中心线距离 |
| `is_apply_mask` | false | 是否 mask 远距离或非相关 lanelet agent |
| `is_add_noise` | false | 是否添加观测噪声 |
| `is_observe_ref_path_other_agents` | false | 是否观测其他 agent 参考路径 |
| `is_use_mtv_distance` | false | 是否使用 MTV 距离 |

## 13. 对手建模与拓扑参数

| 参数 | 当前值 | 说明 |
|---|---:|---|
| `is_using_opponent_modeling` | true | 是否启用对手建模 |
| `is_using_prioritized_marl` | false | 是否启用优先级 MARL |
| `prioritization_method` | `soft_label` | 优先级方法配置 |
| `use_topology_neighbor_selection` | true | 是否用 topology 模型筛选策略邻居 |
| `topology_selection_threshold` | 0.0 | topology 选择阈值 |
| `topology_loss_weight` | 0.5 | topology loss 权重 |
| `lr_action_predictor` | 3e-4 | topology action predictor 学习率 |

对手建模流程：

1. 从环境 `info` 中读取 ego observation、neighbors observation、relative features。
2. 如果 topology action predictor 可用，预测每个邻居的动作。
3. actor 观测尾部保持 0。
4. critic 观测尾部填入预测动作。
5. 同时生成 soft-label priority ordering 和 random priority ordering，供优先级传播或分析使用。

## 14. 保存、测试与评估参数

| 参数 | 当前值 | 说明 |
|---|---:|---|
| `is_save_intermediate_model` | true | 是否保存中间最优模型 |
| `episode_reward_mean_current` | 0.0 | 当前平均 episode reward |
| `episode_reward_intermediate` | -1000 | 已保存最优 reward 初值 |
| `where_to_save` | `outputs/Top-K-3-6R/seed2/` | 输出路径 |
| `is_load_model` | false | 是否加载模型 |
| `is_load_final_model` | false | 是否加载最终模型 |
| `is_continue_train` | false | 是否继续训练 |
| `model_name` | `""` | 模型名 |
| `is_testing_mode` | false | 是否测试模式 |
| `is_visualize_short_term_path` | true | 是否可视化短期参考路径 |
| `is_save_eval_results` | true | 是否保存评估结果 |
| `is_prb` | false | 是否启用 prioritized replay buffer |

## 15. 当前配置下的核心设计总结

当前训练环境可以简化理解为：

```text
每个 CAV agent:
    输入:
        自车局部速度
        自车未来 3 个参考路径点
        自车到参考线和左右边界距离
        3 个近邻车辆的顶点、速度、距离
        对手建模动作占位

    输出:
        速度命令
        转向角命令

    奖励:
        鼓励沿参考路径正向快速前进
        惩罚偏离参考线
        惩罚靠近车道边界
        惩罚靠近其他车辆
        强惩罚碰撞车辆或边界
        惩罚转向变化过快
```

当前最重要的实现注意点：

1. `config.json` 中 `n_agents = 6`，但 `CPM_mixed` 实际使用 4 个 agent。
2. `n_topology_nearing_agents_observed = 5`，但实际最多只能观测 3 个其他 agent。
3. `is_using_opponent_modeling = true` 时，actor 和 critic 的观测并不完全相同。
4. `Agent(collide=False)` 不代表没有碰撞惩罚；碰撞由项目代码手动检测。
5. 当前 `is_use_mtv_distance = false`，车间距离和近邻选择主要基于中心距。
6. 当前 `cpm_scenario_probabilities = [1.0, 0.0, 0.0]`，所以 `CPM_mixed` 实际只采样 intersection。

