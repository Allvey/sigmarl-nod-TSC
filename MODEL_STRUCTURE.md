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

       独立 Deadlock Critic：全局观测/物理信息 + 时序状态 + 实际联合动作
                           204 → 128 → 128 → 4个死锁视野输出
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

Deadlock auxiliary loss（每个 rollout，独立优化4轮）
  ├─ 持续停滞车组的有限视野最大违反量，无折扣
  └─ 只更新 Deadlock Critic，不进入 Actor / Safety / NOD loss
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

阶段五不加入 Safety Actor loss 或对偶变量；Deadlock Critic 由下述阶段六独立实现。`n_iters=250`、现有PPO/NOD参数和训练测试命令保持不变。

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

## 7. 阶段六：时序死锁标签与独立评价

实现位于 `utilities/nod_marl/deadlock.py`。检测器属于环境，每个物理时间步只更新一次；重复读取info不推进计时，rollout换批次不清空历史。全环境或单车重置时清除对应历史和相关车组的持续计时。

首版将以下条件的交集定义为“疑似策略性死锁”：

1. 车辆中心距离不超过0.6米，且前方参考路径走廊相交或接近；据此形成至少2辆车的连通车组。
2. 车组中所有车辆持续低速（不超过0.03米/秒），最近1秒沿参考路线的净进展不超过0.01米；需要完整历史窗口。单车静止、邻车还在正常前进或不同通道不会仅因停车被标为死锁。
3. 至少一辆车被规则允许前进，且存在通过检查的局部前进动作。当前地图未建模信号灯，许可默认true；信号/规则控制器可在物理步前设置 `scenario.deadlock_forward_allowed[B,N]`，该车辆重置时许可恢复默认。禁止全组前进时不累计死锁时间。
4. 局部动作取零转向、0.1米/秒向前移动0.02米：沿参考方向进展至少0.01米；其他车辆保持当前速度时，整个动作期间的连续最近中心距离至少0.25米；当前车身边界净距至少“移动距离+0.01米”。已有碰撞状态不标为死锁。
5. 以上条件连续满足超过2秒，才令 `g_d>0`。计时过程中 `g_d=eligible_seconds/2-1`，截到[-1,1]；条件不满足则为-1。车组成员/车辆身份变化会清除持续计时。

该局部动作只是可执行微小前进的保守代理，不证明整组必然能完全解锁；未找到此动作也不代表所有可能的转向、倒车或协调动作均不可行。TTC、信号和优先权的完整规则建模不在当前首版中。

每车12维增广状态依次为：归一化低速时间、等待时间、窗口路线进展、历史就绪、本人通行许可、本人局部动作可行、车组大小、全组停滞、全组存在通行机会、条件持续时间、是否属于冲突车组、归一化速度。当前位置/速度/航向5维、原始观测32维和实际动作2维一并输入独立Critic，4车合计204维。

视野仍为 `[1,4,8,16]` 个状态，复用阶段五的终止状态与跨重置/批次截断规则。模型、Adam、随机数流、梯度均独立；Safety复用的只有训练代码，其参数、目标、指标和检查点合同不变。现有Actor标准差上限及全部PPO/NOD/Safety训练参数保持不变。

新增 `rewardX.XX_deadlock_critic.pth` 和 `final_deadlock_critic.pth`，支持独立加载/继续训练。旧模型缺少死锁文件时初始化新分支；时序阈值、视野、输入维度或安全margin不兼容时提示后重新初始化。环境历史不写入模型检查点，重新启动的物理环境从中性状态开始。

训练JSON新增 `deadlock_metrics_list`，W&B新增 `deadlock/*`。重点检查：

- `deadlock_target_count`、`missed_deadlock_count`、`deadlock_recall`、`precision`、`false_positive_count`及各视野的正样本/漏判/误报计数；
- `waiting_agent_ratio`、`conflict_group_agent_ratio`、`escape_available_agent_ratio`、`eligible_agent_ratio`、`max_eligible_seconds`；
- `onset_transition_count`（后继状态出现新死锁的环境时间步数，每个车组成员不重复计数）；
- `prediction_mae`、`underestimate_rate`、`valid_target_ratio`、各视野MAE和有效样本量。

所有预测误差在对新rollout拟合之前记录。无正样本或无正预测时，对应recall/precision记0并配套计数；若全程缺少死锁正样本，不能因loss很低而认定Critic已经学会死锁。阶段六仍不使用这些预测约束Actor，需结合人工检查的死锁片段验证后再进入下一阶段。
