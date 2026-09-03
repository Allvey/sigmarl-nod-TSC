# SigmaRL Traffic 网络结构与维度说明

本文档基于当前仓库源码和根目录 `config.json`，整理项目中各个神经网络模块的结构、隐藏层、输入输出维度。

主要参考文件：

- 主训练网络：`utilities/mappo_cavs.py`
- 优先级网络：`utilities/helper_training.py`
- 拓扑网络：`utilities/topology_module.py`
- 环境观测：`scenarios/road_traffic.py`
- 当前配置：`config.json`

说明：本地直接实例化环境读取 spec 时遇到 Python 依赖问题：

```text
ImportError: cannot import name 'TypeIs' from 'typing_extensions'
```

因此本文档中的维度为按源码和当前配置静态推导所得。

## 1. 当前配置下的关键维度

当前 `config.json` 中的关键参数：

```text
n_agents = 6
n_nearing_agents_observed = 3
n_topology_nearing_agents_observed = 5
n_points_short_term = 3
is_ego_view = true
is_observe_vertices = true
is_observe_distance_to_agents = true
is_observe_distance_to_boundaries = true
is_observe_distance_to_center_line = true
is_observe_ref_path_other_agents = false
is_using_opponent_modeling = true
is_using_prioritized_marl = false
```

记号：

| 记号 | 含义 | 当前值 |
|---|---|---:|
| `N` | agent 数量 | 6 |
| `A` | 动作维度 | 2 |
| `K_policy` | 策略观测近邻数 | 3 |
| `K_topo` | 拓扑候选近邻数 | 5 |
| `D_ego` | 拓扑 ego observation 维度 | 10 |
| `D_nei` | 单个邻居 observation 维度 | 11 |
| `D_rel` | 单个邻居 relative feature 维度 | 4 |
| `D_obs` | 主 actor 每 agent 观测维度 | 49 |

当前每个 agent 的 actor 输入维度：

```text
D_obs = 自车观测 + 策略邻居观测 + 对手建模动作占位
      = 10 + 3 * 11 + 3 * 2
      = 49
```

自车观测 10 维：

```text
自车前向速度                         1
自车短期参考路径 3 个点，每点 x,y      3 * 2 = 6
到参考中心线距离                     1
到左边界距离                         1
到右边界距离                         1
----------------------------------------
合计                                10
```

单个邻居观测 11 维：

```text
邻车 4 个矩形顶点，每点 x,y            4 * 2 = 8
邻车速度                              2
到邻车距离                            1
----------------------------------------
合计                                11
```

拓扑相对特征 4 维：

```text
dx, dy, d_yaw, d_speed
```

## 2. 主 Actor 网络

代码位置：`utilities/mappo_cavs.py`

功能：分散式策略网络。每个 agent 根据自己的观测输出连续动作分布。

### 2.1 输入

```text
key   = ("agents", "observation")
shape = [B, N, D_obs]
      = [B, 6, 49]
```

其中 `B` 表示 batch 维度，可以来自并行环境、rollout 时间步和 minibatch 的组合。

### 2.2 网络结构

源码配置：

```python
MultiAgentMLP(
    n_agent_inputs=49,
    n_agent_outputs=2 * action_dim,
    n_agents=6,
    centralised=False,
    share_params=True,
    depth=2,
    num_cells=256,
    activation_class=torch.nn.Tanh,
)
NormalParamExtractor()
ProbabilisticActor(TanhNormal)
```

等效结构：

```text
每个 agent:

49
 -> Linear / MLP hidden 256
 -> Tanh
 -> Linear / MLP hidden 256
 -> Tanh
 -> Linear output 4
 -> NormalParamExtractor
 -> loc[2], scale[2]
 -> TanhNormal
 -> action[2]
```

### 2.3 输出

```text
("agents", "loc")              [B, 6, 2]
("agents", "scale")            [B, 6, 2]
("agents", "action")           [B, 6, 2]
("agents", "sample_log_prob")  [B, 6]
```

动作含义：

```text
action[0] = v_command
action[1] = steering_command
```

## 3. 主 Critic 网络

代码位置：`utilities/mappo_cavs.py`

功能：MAPPO 中心化价值网络，输出每个 agent 的状态价值。

当前 `is_using_opponent_modeling = true`，因此 critic 输入键为：

```text
("agents", "info", "critic_observation")
```

该输入维度仍为 49，但含义不同：观测尾部 6 维会填入预测的邻居动作；actor 侧尾部通常置零。

### 3.1 输入

```text
shape = [B, N, D_obs]
      = [B, 6, 49]
```

由于 `centralised=True`，critic 在计算每个 agent value 时可使用全体 agent 信息。

中心化等效输入维度：

```text
N * D_obs = 6 * 49 = 294
```

### 3.2 网络结构

源码配置：

```python
MultiAgentMLP(
    n_agent_inputs=49,
    n_agent_outputs=1,
    n_agents=6,
    centralised=True,
    share_params=True,
    depth=2,
    num_cells=256,
    activation_class=torch.nn.Tanh,
)
```

等效结构：

```text
每个 agent value head:

294
 -> Linear / MLP hidden 256
 -> Tanh
 -> Linear / MLP hidden 256
 -> Tanh
 -> Linear output 1
```

### 3.3 输出

```text
("agents", "state_value") [B, 6, 1]
```

## 4. TopologyLearner 拓扑关系网络

代码位置：`utilities/topology_module.py`

功能：判断 ego agent 与每个候选邻居之间是否存在拓扑交互边。

该模块由三部分组成：

1. `TopoDecoder`
2. `TopoDecoderLayer`
3. `TopologyHead`

当前固定超参数：

```text
num_layers = 2
d_latent = 128
```

### 4.1 输入

训练时会将时间、环境、agent 等维度展平为 `B_total`。

```text
ego_observation:
  [B_total, D_ego]
  [B_total, 10]

neighbors_observation:
  [B_total, K_topo, D_nei]
  [B_total, 5, 11]

relative_features:
  [B_total, K_topo, D_rel]
  [B_total, 5, 4]
```

### 4.2 TopoDecoder 初始映射

```text
relative_features: [B_total, 5, 4]

Linear:
  4 -> 128

输出:
  q_R: [B_total, 5, 128]
```

### 4.3 TopoDecoderLayer

每一层都会拼接四类特征：

```text
q_ego broadcast:      [B_total, 5, 10]
s_neighbors:         [B_total, 5, 11]
r_relative:          [B_total, 5, 4]
q_R_in:              [B_total, 5, 128]
```

拼接后维度：

```text
10 + 11 + 4 + 128 = 153
```

每层 MLP：

```text
153 -> 128 -> 128
activation = ReLU
```

残差连接：

```text
q_R_out = q_R_in + q_R_update
```

当前有 2 层：

```text
TopoDecoderLayer x 2
```

输出：

```text
q_R_final: [B_total, 5, 128]
```

### 4.4 TopologyHead

```text
输入:
  q_R_final [B_total, 5, 128]

MLP:
  128 -> 64 -> 1
  activation = ReLU

输出:
  edge_logits [B_total, 5, 1]
```

训练时通常会 squeeze：

```text
edge_logits.squeeze(-1): [B_total, 5]
```

再与拓扑标签做 BCE：

```text
binary_cross_entropy_with_logits(edge_logits, e_labels)
```

其中：

```text
e_labels: [B_total, 5]
```

### 4.5 输出含义

```text
edge_logits[b, k, 0]
```

表示第 `b` 个样本中，ego agent 与第 `k` 个候选邻居之间的拓扑交互边 logit。

经过 sigmoid 后：

```text
edge_probs = sigmoid(edge_logits)
```

得到拓扑边概率：

```text
edge_probs: [B_total, 5, 1]
```

## 5. TopologyActionPredictor 邻居动作预测网络

代码位置：`utilities/topology_module.py`

功能：基于拓扑关系表征预测候选邻居动作，用于对手建模填充 critic observation 的尾部。

当前配置：

```text
share_decoder = false
action_dim = 2
d_latent = 128
hidden_ratio = 0.5
hidden_size = 64
```

### 5.1 输入

与 `TopologyLearner` 相同：

```text
ego_observation:        [B_total, 10]
neighbors_observation:  [B_total, 5, 11]
relative_features:      [B_total, 5, 4]
```

### 5.2 网络结构

因为 `share_decoder = false`，动作预测器会构造一个独立的 `TopoDecoder`。

独立 decoder：

```text
Initial mapper:
  4 -> 128

TopoDecoderLayer x 2:
  每层输入 153
  153 -> 128 -> 128
  activation = ReLU
  residual update
```

动作预测头 `NeighborActionHead`：

```text
输入:
  q_R_final [B_total, 5, 128]

MLP:
  128 -> 64 -> 2
  activation = ReLU

输出:
  action_pred [B_total, 5, 2]
```

### 5.3 输出

```text
predicted_neighbor_actions: [B_total, 5, 2]
```

每个候选邻居动作：

```text
[v_command, steering_command]
```

如果调用时 `return_edges = true`，还会额外通过原拓扑网络返回：

```text
edge_logits: [B_total, 5, 1]
```

## 6. PriorityModule 优先级网络

代码位置：`utilities/helper_training.py`

当前配置：

```text
is_using_prioritized_marl = false
```

因此该模块当前训练不会启用。但代码中定义了完整结构。

PriorityModule 包含：

1. Priority Actor
2. Priority Critic

## 6.1 Priority Actor

功能：输出每个 agent 的优先级 score 分布。

输入键：

```text
("agents", "info", "priority_observation")
```

维度：

```text
[B, N, D_obs] = [B, 6, 49]
```

网络结构：

```python
MultiAgentMLP(
    n_agent_inputs=49,
    n_agent_outputs=2 * 1,
    n_agents=6,
    centralised=False,
    share_params=True,
    depth=2,
    num_cells=256,
    activation_class=torch.nn.Tanh,
)
NormalParamExtractor()
ProbabilisticActor(TanhNormal)
```

等效结构：

```text
49
 -> 256
 -> Tanh
 -> 256
 -> Tanh
 -> 2
 -> loc[1], scale[1]
 -> TanhNormal
 -> priority_score[1]
```

输出：

```text
priority loc:             [B, 6, 1]
priority scale:           [B, 6, 1]
priority scores:          [B, 6, 1]
priority sample_log_prob: [B, 6]
```

## 6.2 Priority Critic

功能：为优先级策略提供 value。

输入：

```text
[B, 6, 49]
```

由于 `mappo = true`，priority critic 也是中心化：

```text
等效输入维度 = 6 * 49 = 294
```

结构：

```text
294 -> 256 -> 256 -> 1
activation = Tanh
```

输出：

```text
priority state_value: [B, 6, 1]
```

## 7. 网络结构总表

| 网络 | 输入 | 隐藏层 | 输出 | 当前启用 |
|---|---|---|---|---|
| 主 Actor | `[B, 6, 49]` | `256, 256`, Tanh | action `[B, 6, 2]` | 是 |
| 主 Critic | `[B, 6, 49]`，中心化等效 `294` | `256, 256`, Tanh | value `[B, 6, 1]` | 是 |
| TopologyLearner | ego `[B,10]`, nei `[B,5,11]`, rel `[B,5,4]` | latent `128`, decoder 2 层 | logits `[B,5,1]` | 是 |
| TopologyActionPredictor | ego `[B,10]`, nei `[B,5,11]`, rel `[B,5,4]` | latent `128`, head `64` | action pred `[B,5,2]` | 是 |
| Priority Actor | `[B,6,49]` | `256,256`, Tanh | score `[B,6,1]` | 否 |
| Priority Critic | `[B,6,49]`，中心化等效 `294` | `256,256`, Tanh | value `[B,6,1]` | 否 |

## 8. 关键信息总结

当前主策略网络输入是：

```text
49 维 / agent
```

主策略输出是：

```text
2 维连续动作 / agent
```

拓扑网络不直接吃完整 49 维策略观测，而是吃结构化输入：

```text
ego:      10 维
neighbor: 11 维 / 候选邻居
relative: 4 维 / 候选邻居
```

拓扑网络对 5 个候选邻居分别输出边 logit：

```text
[B_total, 5, 1]
```

动作预测头对 5 个候选邻居分别输出预测动作：

```text
[B_total, 5, 2]
```

