# SigmaRL 模型结构概览

- 目标：用于联网自动驾驶（CAVs）的分散式多智能体强化学习框架，集成优先级学习、拓扑关系学习与对手建模。
- 基础库：TorchRL（多智能体模块、PPO、GAE、ProbabilisticActor）、VMAS（可微仿真）、TensorDict。
- 核心组件：策略 Actor、价值 Critic、优先级模块、拓扑学习网络、邻居动作预测、对手建模与优先化动作传播。

## 组件总览
- 主策略 Actor
  - 使用 `MultiAgentMLP` 构建分散式策略网络，输出高斯分布的参数（均值与尺度）。
  - 分布经 `TanhNormal` 采样，输出动作与 `log_prob`。
- 价值 Critic
  - 使用 `MultiAgentMLP` 构建中心化价值网络（MAPPO），输出每智能体状态价值。
- 优先级模块（PriorityModule）
  - 独立的优先级 Actor/Critic（结构同主网），学习“优先级得分”，供排序与动作传播使用。
- 拓扑学习（TopologyLearner）
  - 解码器堆栈编码智能体之间关系，MLP 头输出边关系 logits。
- 邻居动作预测（TopologyActionPredictor）
  - 将拓扑关系潜在向量映射为每邻居动作预测，用于对手建模的动作填充。

## 数据流与关键键
- 观测键
  - 基础观测：`("agents","observation")`
  - 优先化传播期间的基础观测：`("agents","info","base_observation")`
  - 对手建模的评估观测（critic）：`("agents","info","critic_observation")`
- 拓扑相关键
  - 结构化输入：`("agents","info","ego_observation")`、`("agents","info","neighbors_observation_flat")`、`("agents","info","relative_features")`
  - 参考路径：`("agents","info","ref_local")`、`("agents","info","ref_neighbors_local")`
  - 邻居索引与距离：`("agents","info","neighbors_indices")`、`("agents","info","neighbors_distance")`
- 优先级排序输出（对手建模）
  - 软标签排序：`("agents","info","soft_label_priority_ordering")`
  - 随机排序：`("agents","info","random_priority_ordering")`

## 主策略（Actor）
- 网络结构
  - `MultiAgentMLP(depth=2, num_cells=256, activation=tanh, share_params=True, centralised=False)`
  - 输出维度：`2 * action_dim`（高斯 `loc/scale`），经 `NormalParamExtractor` 拆分。
  - 分布：`TanhNormal`（边界由环境动作空间设定），返回 `log_prob`。
- 代码位置
  - `utilities/mappo_cavs.py:120-135`（Actor MLP + NormalParamExtractor）
  - `utilities/mappo_cavs.py:139-146`（TensorDictModule 封装）
  - `utilities/mappo_cavs.py:148-165`（ProbabilisticActor 配置）

## 价值网络（Critic）
- 网络结构
  - `MultiAgentMLP(depth=2, num_cells=256, activation=tanh, share_params=True, centralised=True)`
  - 输出维度：每智能体 1 个状态价值。
- 输入选择
  - 对手建模开启时使用 `("agents","info","critic_observation")`，否则与 Actor 相同观测。
- 代码位置
  - `utilities/mappo_cavs.py:168-180`（Critic MLP）
  - `utilities/mappo_cavs.py:182-186`（TensorDictModule 封装）
  - `utilities/mappo_cavs.py:113-118`（critic 观测键选择）

## 优先级模块（PriorityModule）
- 优先级 Actor
  - `MultiAgentMLP(depth=2, num_cells=256, activation=tanh, share_params=True, centralised=False)`
  - 输出维度：`2 * 1`（优先级得分分布的 `loc/scale`）
  - 通过 `ProbabilisticActor(TanhNormal)` 采样得分，返回 `sample_log_prob`
- 优先级 Critic
  - 结构同主 Critic，输出每智能体状态价值。
- 损失与 GAE
  - `ClipPPOLoss`（独立键集） + `make_value_estimator(GAE)`
- 代码位置
  - `utilities/helper_training.py:1029-1042`（优先级 Actor MLP + NormalParamExtractor）
  - `utilities/helper_training.py:1044-1059`（ProbabilisticActor 配置）
  - `utilities/helper_training.py:1061-1073`（优先级 Critic）
  - `utilities/helper_training.py:1086-1114`（PPO Loss 与 GAE）

## 拓扑学习网络（TopologyLearner）
- 解码器层（TopoDecoderLayer）
  - 输入拼接：`q_ego`、`s_neighbors`、`r_relative`、`q_R_in`，两层线性 + `ReLU`，残差连接输出。
- 解码器堆栈（TopoDecoder）
  - 初始映射：`nn.Linear(d_rel, d_latent)`，随后堆叠若干 `TopoDecoderLayer`。
- 拓扑头（TopologyHead）
  - `MLP: d_latent → d_latent/2 → 1`，输出边关系 logits。
- 代码位置
  - `utilities/topology_module.py:21-44`（TopoDecoderLayer）
  - `utilities/topology_module.py:50-63`（TopoDecoder）
  - `utilities/topology_module.py:65-75`（TopologyHead）

## 邻居动作预测（TopologyActionPredictor）
- 轻量动作头（NeighborActionHead）
  - `MLP: d_latent → hidden → action_dim`，根据关系潜在向量预测邻居动作。
- 参数共享/独立模式
  - 支持复用拓扑解码器参数或构建同结构的独立解码器（默认独立，避免相互干扰）。
- 管理器初始化
  - `TopologyManager.ensure_initialized()` 构建 `TopologyLearner(d_latent=128, num_layers=2)` 与 `TopologyActionPredictor(action_dim=2)` 并设置优化器。
- 代码位置
  - `utilities/topology_module.py:104-111`（NeighborActionHead）
  - `utilities/topology_module.py:123-209`（TopologyActionPredictor）
  - `utilities/topology_module.py:224-251`（TopologyManager.ensure_initialized）

## 对手建模（opponent_modeling）
- 作用
  - 若提供拓扑动作预测器，则用邻居动作预测填充智能体观测尾部；否则退化为策略一次前向的近邻动作填充。
- 软标签优先级与随机优先级并行输出
  - 生成软标签全序 `soft_label_priority_ordering` 与随机全序 `random_priority_ordering` 写入 `tensordict`。
  - 控制台打印一组样例，便于对比。
- 代码位置
  - `utilities/helper_training.py:1443-1573`（对手建模主流程）
  - `utilities/helper_training.py:1496-1527`（软标签优先级拓扑排序）
  - `utilities/helper_training.py:1531-1537`（双排序写入与打印）

## 优先化动作传播（prioritized_ap_policy）
- 流程
  - 生成智能体优先顺序（来源可选：优先级模块、随机、软标签）。
  - 依次处理当前轮到的智能体，将已产生的邻居动作拼接到观测尾部，调用策略生成当前动作并回填。
  - 完成全体智能体动作后，写回组合动作与观测。
- 代码位置
  - `utilities/helper_training.py:1588-1768`（函数主体）
  - 软标签排序分支：`utilities/helper_training.py:1649-1689`

## 训练损失与优化
- 主策略
  - `ClipPPOLoss(actor, critic, clip_epsilon, entropy_coef, normalize_advantage=False)`：`utilities/mappo_cavs.py:381-388`
  - 键设置 `reward/action/log_prob/value/done/terminated`：`utilities/mappo_cavs.py:389-397`
  - `make_value_estimator(ValueEstimators.GAE, gamma, lmbda)`：`utilities/mappo_cavs.py:399-404`
  - 优化器：`Adam(loss_module.parameters(), lr)`：`utilities/mappo_cavs.py:404`
- 优先级策略
  - 独立 `ClipPPOLoss` 与 `GAE`，键与学习率配置见：`utilities/helper_training.py:1086-1119`
- 拓扑分支
  - BCE：`binary_cross_entropy_with_logits(edge_logits, e_labels)`：`utilities/topology_module.py:300-308`
  - 权重融合与训练循环集成：`utilities/mappo_cavs.py:542-559`

## 参考位置索引
- 主策略 Actor/Critic
  - `utilities/mappo_cavs.py:120-165`（Actor 组网与分布）
  - `utilities/mappo_cavs.py:168-186`（Critic 组网）
  - `utilities/mappo_cavs.py:381-404`（PPO + GAE + 优化器）
- 优先级模块
  - `utilities/helper_training.py:1008-1121`（优先级 Actor/Critic/PPO/GAE）
- 拓扑学习与动作预测
  - `utilities/topology_module.py:21-75`（解码器层、堆栈与头）
  - `utilities/topology_module.py:123-209`（动作预测器）
  - `utilities/topology_module.py:224-251`（初始化）
- 对手建模与优先化传播
  - `utilities/helper_training.py:1443-1573`（对手建模）
  - `utilities/helper_training.py:1588-1768`（优先化动作传播）
