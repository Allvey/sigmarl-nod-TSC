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
```

NOD 的循环状态在采集时按时间顺序推进，并以 detached context 存入 rollout。PPO 打乱 minibatch 时只重新计算无状态消息聚合器，不会按乱序重放 GRU。NOD 参数和 PPO 参数不共享梯度。

## 4. 为缓解此前策略坍塌加入的约束

- PPO 每批训练轮数由 60 降为 15。
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
