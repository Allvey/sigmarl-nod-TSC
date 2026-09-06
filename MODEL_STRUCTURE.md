# 最初版本与当前版本网络结构对比

本文以根目录 `config.json` 的当前设置为准。训练和测试入口保持不变：

```bash
python main_training.py
python main_testing.py
```

## 1. 最初版本（任何 NOD 修改之前）

```text
环境状态
  │
  ├─ 基础 observation：32 维
  │    └─ 对手建模动作占位：2 个邻居 × 2 维动作
  │
  ├─ TopologyLearner
  │    ├─ ego observation
  │    ├─ neighbor observation
  │    ├─ relative features
  │    └─ 输出有向边概率，参与 BCE 训练和邻居选择
  │
  ├─ TopologyActionPredictor
  │    └─ 复用拓扑表示，预测邻居动作并填充 Critic 观测尾部
  │
  └─ Actor / Critic
       ├─ Actor：局部观测 → 256 → 256 → loc/scale → 二维动作
       └─ Critic：带预测动作的集中式观测 → 256 → 256 → value
```

这条路径同时维护策略、拓扑分类和动作预测三个学习目标，Topology 输出还会改变策略所看到的邻居集合。

## 2. 当前简化版本

```text
环境状态
  │
  ├─ 原始局部 observation：32 维 ─────────────────────────────┐
  │                                                           │
  ├─ 每条有向交互边的物理特征：20 维                           │
  │    ├─ 相对位置、速度、朝向、距离                            │
  │    ├─ TTC、冲突点、ETA 差、接近程度、重叠风险等             │
  │    └─ 固定 agent identity + generation mask               │
  │          │                                                 │
  │          ▼                                                 │
  │       GRUCell：20 → 64（`nod_history_mode="none"` 可关闭历史）
  │          │
  │          ├─ 6 维单调物理风险量
  │          ├─ 1 维风险注意力
  │          └─ 1 维有界意见 z
  │          │
  │          ▼
  │       每边 context：64 + 6 + 1 + 1 = 72 维
  │          │
  │          ▼
  │       LayerNorm → 72 → 64 → 32 → masked attention sum
  │          │
  │          └─ 32 维消息，逐元素限制在 [-0.1, 0.1]
  │                                                           │
  └──────────────────────┬────────────────────────────────────┘
                         ▼
       Actor 输入：[原始 observation 32，NOD 消息 32，上一步动作 2]
                  = 66 维
                         │
                         ▼
       分散式共享 Actor：66 → 256 → 256 → 4
                         │                └─ loc 2 + scale 2
                         ▼
                  TanhNormal → 二维车辆动作

       集中式 Critic：每车原始 observation 32 → 256 → 256 → value

       独立 Safety Critic：全局观测/物理信息 + 实际联合动作
                         168 → 128 → 128 → 4个安全视野输出
```

当前有效网络中不存在 TopologyLearner、动作预测器和对手建模分支；策略邻居仍由原项目的最近邻观测逻辑产生，NOD 则使用独立的稳定有向物理边。

## 3. 当前训练关系

```text
PPO task loss
  ├─ 更新 Actor MLP
  ├─ 更新 32 维消息聚合器（独立较小学习率 5e-5）
  └─ 更新集中式 Critic

NOD auxiliary loss（默认每 10 个 rollout 更新一次）
  ├─ 时序似然 NLL
  ├─ 反事实风险标定 BCE
  └─ 更新物理特征 GRU、似然头和有界风险权重

Safety auxiliary loss（每个 rollout，独立优化4轮）
  ├─ 当前状态开始的有限视野最大违反量，无折扣
  └─ 只更新 Safety Critic，不进入 Actor loss
```

NOD 的循环状态在采集时按时间顺序推进，并以 detached context 存入 rollout。PPO 打乱 minibatch 时只重新计算无状态消息聚合器，不会按乱序重放 GRU。NOD 参数和 PPO 参数不共享梯度。

## 4. 为缓解此前策略坍塌加入的约束

- PPO 每批训练轮数为 60，Actor标准差保留原下限并限制上限为1.0。
- NOD 默认每 10 个 rollout 更新一次，降低表示非平稳性。
- 消息聚合器使用单独的 `5e-5` 学习率。
- 32 维消息逐元素限制到 `[-0.1, 0.1]`。
- 六个风险权重被限制在 `[0, 1]`，避免注意力权重持续无界增长。
- Actor 始终直接保留原始 32 维物理观测，不依赖 NOD 消息才能行动。

这些约束提高了训练稳定性的可控性，但最终性能仍需要用新的完整训练曲线判断。

## 5. Checkpoint 兼容性

- 原有训练、测试命令和 policy/critic 文件命名不变。
- 旧 Actor checkpoint 会自动迁移可复用的 MLP 权重；新消息聚合器保持新初始化。
- 旧版 NOD checkpoint 的输入合同不同，因此会被明确忽略并重新初始化。
- 已存在的旧 Topology/动作预测 checkpoint 不会删除，但当前代码不再读取或写入它们。
- Safety Critic 另存为 `rewardX.XX_safety_critic.pth` / `final_safety_critic.pth`；旧模型缺少该文件时独立初始化，不影响策略加载。安全输入维度、视野或margin配置不兼容时明确提示并重新初始化。

## 6. 阶段五：独立安全评价

本阶段暂不加入 Safety Actor loss、Deadlock Critic 或对偶变量。`n_iters=250`、现有PPO/NOD参数和训练测试命令保持不变。

安全量使用环境的物理单位，先归一化再取最大值：

- 车辆间距：`(0.25 - 最近车辆中心距离) / 0.25`；0.25米可通过 `safety_safe_distance` 调整。这是中心距离margin，不是车辆轮廓间距。
- 边界：`(0.01 - 车辆边界净距) / 0.01`；净距复用环境考虑车身尺寸后的边界距离，0.01米由 `safety_boundary_margin` 设置。
- 碰撞：碰撞为+1，否则为-1；安全侧margin截到-1，正违反量不截断。

每车得到3个margin，全局 `g_s` 为所有车辆所有margin的最大值。非正为满足，正为违反。首版不加入TTC约束。

安全网络输入每车的原始32维观测、归一化位置/速度/航向5维、3个当前margin及2维实际动作，4车合计168维。只读取当前信息，标签使用后续轨迹。它拟合当前采样策略下的未来最大违反量，不构成严格安全保证。

`safety_horizons=[1,4,8,16]` 中的h表示从当前状态起共h个状态：`max(g_t,...,g_{t+h-1})`。h=1只评价当前状态。终止时包含终止状态违反量，之后不跨入下一回合；单车身份重置导致的跨段样本和批次尾部不足视野的样本会被屏蔽，而不是标为安全。批次末尾若有真实后继状态，仍可监督h=2。

训练使用独立MLP、Adam与随机数生成器，输入和标签均detach；安全网络初始化及小批次打乱不推进PPO的随机数流。Smooth L1对低估样本赋予2倍权重。每轮先记录新采集数据上的预测误差，再训练安全网络，避免把训练后的拟合误差当作新数据表现。

诊断写入原训练JSON的 `safety_metrics_list` 和W&B的 `safety/*`：

- `prediction_mae`、各视野 `h*/mae` / `h*/target_mean`；
- `unsafe_recall`、`unsafe_target_count`、`missed_unsafe_count`；无危险样本时recall记0，必须结合计数解释；
- `underestimate_rate`（低估超过0.05）、`worst_underestimate`；
- `instant_violation_rate` / `instant_violation_count` / `transition_count`（全局环境状态口径）；
- `valid_target_ratio`、各视野 `h*/valid_count`、`training_loss`、`optimizer_updates`。

验证时首先关注漏判、低估及有效样本量。本阶段不应期待安全网络本身提高Actor奖励；它为后续约束策略提供经过检验的风险评价。
