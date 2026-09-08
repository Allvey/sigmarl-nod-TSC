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

Safety Actor penalty（预热后加入 PPO）
  ├─ 冻结 Safety 权重，评价当前 Actor 的名义联合动作
  └─ 通过动作梯度更新 Actor 和消息聚合器，标量约束权重每批更新一次

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

所有预测误差在对新rollout拟合之前记录。无正样本或无正预测时，对应recall/precision记0并配套计数；若全程缺少死锁正样本，不能因loss很低而认定Critic已经学会死锁。2026-09-07按用户决定暂停死锁工作：根配置关闭 `is_using_deadlock_critic`，测试入口也显式关闭以覆盖旧训练JSON中的true。已有实现与文件兼容保留，但当前入口不进行死锁监督训练、加载或Actor约束；其验证不再阻塞安全主线。

## 8. 阶段七：先接入 Safety Actor 约束

根配置启用 `is_using_safety_constraint=true`。为保持旧 JSON 行为，Parameters 中该开关默认 false；关闭后恢复仅独立训练 Safety Critic 的路径。死锁预测仍不参与 Actor 更新。

实现复用 PPO、现有 Safety 网络和安全指标列表，不新增网络。Safety 至少完成10个有效 rollout 后，还必须通过可靠性门控；第11批只是最早可能启用的时间。每个 PPO minibatch 使用同一套可训练 Actor 参数及缓存的 NOD context，重新生成确定性名义动作（TanhNormal.mode）；不使用缓存动作计算策略梯度，也不改写采样动作或旧 log-prob。

门控采用最近5批新rollout在拟合前的预测，选择“当前安全且所有h>1目标有效”的状态，分别对预测和真实目标取未来视野最大值。累计至少64个危险状态目标、危险召回率不低于0.90、低估超过0.05的比例不高于0.15，才允许约束更新。统计按状态目标计数，不是64个独立碰撞事件；不混入h=1的容易样本，不使用拟合后误差。窗口逐批滚动，可靠性变差可以重新关闭。当前批PPO使用此前已完成批次的门控证据，当前新证据在本批Safety训练时记录，供下一批使用；这是在线经验门控，不是独立测试集上的安全证书。

状态特征和 Safety 权重 detach，仅保留动作梯度。取 h>1 的预测最大值加0.05安全余量作为 q；h=1只描述当前状态，不作为动作约束。由于目标包含当前违反量，已违反状态无法通过当前动作使目标满足，因此本版仅在当前所有 safety margin 非正的状态上计算 `lambda * mean(relu(q))`，用于从安全状态避免进入危险状态，不提供已危险状态的恢复控制。

约束权重初值0.01，每个 rollout 的 PPO 更新完成后执行 `lambda = clip(lambda + 0.01 * (mean(relu(q)) - risk_budget), 0, 0.1)`。均值的分母是所有eligible状态，负风险记0，不能再抵消正违反量。默认 `safety_constraint_risk_budget=0`：没有违反时保持权重，有违反时增加至上限；即使加载权重为0，也能在门控就绪且存在正风险时恢复。显式设置正预算时，低于预算才允许下降。门控未就绪或没有eligible样本时不更新权重。PPO更新期间权重不变；Safety在其后用真实采样动作继续监督训练，并清空拟合梯度以验证Actor反传隔离。这个有上限的标量惩罚仍是工程简化，只约束名义动作，随机采样动作仍可能偏离。

指标沿用 `safety_metrics_list` 和 `safety/*`。`actor_constraint_ready`表示门控就绪；`actor_constraint_active`要求本批实际应用正权重且有正惩罚。分别记录 `actor_constraint_signed_risk`（旧名risk保留）、`actor_constraint_positive_risk`、`actor_constraint_dual_signal`、`actor_constraint_violation_rate`、`actor_constraint_sample_count`、`actor_constraint_eligible_fraction`、权重/下一权重/损失及 `actor_constraint_weight_at_cap`。门控关闭时动作风险没有评估，sample_count为0；此时risk为0不能解释为没有危险。

`constraint_gate_reason`：0=就绪，1=开关关闭，2=预热，3=窗口内危险目标不足，4=召回不足，5=低估过多。配套 `constraint_gate_valid_count/unsafe_count/recall/underestimate_rate` 是本批Actor实际使用的门控证据，口径与原有跨视野全体样本的 `unsafe_recall` 不同。门控长期不通过时改善数据和Critic，不放宽阈值伪装启用；权重长期顶上限但碰撞未改善，也需单独分析。

Safety sidecar 增存有效训练批数、权重、最近门控统计和版本2约束配置合同。继续训练时仅在合同一致且状态齐全时恢复；旧sidecar缺少状态或门控/对偶配置变化时，保留兼容的Safety模型与优化器，重置门控历史、预热计数和初始权重。Actor输入输出不变，测试时不会额外调用Safety过滤动作。4车Safety仍不能直接加载到8车测试；当前阶段仍需完整250轮训练验证碰撞与通行效果，未实施阶段8的模型保守化或阶段9的场景扩展。


## 9. 阶段8A：成对安全State Value影子分支（2026-09-08）

根配置已开启`is_using_safety_value_shadow`，不改变阶段7的Actor训练目标。旧配置缺失此字段时默认关闭。

```text
独立环境：4环境 × 64步 / PPO批，Actor mode动作
  ├─ policy/NOD权重从当前训练模型同步，在线历史和随机数完全隔离
  └─ 当前物理数据 → 成对State Value
       ├─ 每边20维物理特征 → 64维Tanh编码 → masked邻居均值
       ├─ 原始局部observation + road/collision margin → 64维Tanh编码
       ├─ 每边编码 + 本车编码 + 邻居聚合 → 64 → 1（成对距离Value）
       └─ 本车编码 + 邻居聚合 → 64 → 2（道路、实际碰撞Value）

真实确定性轨迹 → 按约束独立折扣max目标 → 仅更新新State Value
```

网络不读取当前动作或意见，不新增循环网络；Safety当前局部观测是工程近似，不能假设覆盖Actor全部隐藏历史。车辆对固定按world slot对齐，并使用双方generation与可见性mask；输出为`[..., N, N+2]`，参数与车辆数无关。离开观测或重置导致的无效后继被屏蔽，不标成安全。

安全目标γ为0.99，采用倒序多步max递推及末端target自举；真实终止保留终止违反量，不连接重置后的车辆。target网络在每批监督后按tau=0.05软更新。每批独立优化4轮，输入和target均detach，不向Actor、旧Safety或NOD传梯度。

模型另存`*_safety_value.pth`；旧`*_safety_critic.pth`仍为阶段7动作Q，不能互换。新文件包含target、优化器、私有采样状态与版本合同。推理入口可加载新Value，但它不直接参与动作选择；8B已通过训练优势影响Actor，阶段9仍待实施。

日志为JSON的`safety_value_metrics_list`及W&B的`safety_value/*`：拟合前pair/road/collision危险计数、recall、低估、零值比例、目标MAE、相邻Value变化、有效样本和额外采样步数/耗时。47项测试通过，短程开关对照的Actor、任务Critic、NOD及旧Safety最终参数逐项一致；完整250轮训练由用户执行。


### 9.1 8A危险识别改进：分头损失与诊断

首版两seed显示普通回归loss降低不代表危险识别可靠。新默认`balanced`模式对pair/road/collision分别计算加权Smooth L1，再等权平均有效分支；`legacy`保留原损失作为对照。正训练目标按rollout内非正/正数量比例加权、上限4，预测低估正目标超过0.05时额外乘2，总权重最高8；各分支以权重和归一化。目标、模型结构、采样预算和Actor目标不变。

新增每头正/非正目标数量与误差、观测危险对应非正训练目标的比例、当前该项约束安全但未来危险的提前预警召回，以及已观测安全窗口的正预测比例。所有误差来自拟合前，空类别同时报告计数；已观测安全不是完整未来安全真值。

保存独立`loss_contract`；继续训练时损失合同变化仅重置Adam、保留兼容Value/target和采样状态。旧sidecar缺失该字段视为legacy。reward7.09完整训练的pair危险召回改善至73.9%，但不同seed不能确认因果；碰撞召回仍仅1.85%，未通过可靠性验收。

本轮新增8项损失、诊断和迁移测试，全套55项通过；完整训练的危险召回与误报取舍仍待验证。


### 9.2 8A危险前起点覆盖

根配置`safety_value_challenging_fraction=0.25`，旁路4个环境中固定3个使用普通随机起点、1个从危险前起点池采样，仍采集4×64步。旧JSON缺失该项默认0；池为空时普通初始化。collision与road分池各128个，非空类型等概率抽取，避免一类挤占另一类。

完整旁路轨迹发生碰撞或道路违反后，从此前最多10步挖掘当前各margin非正、车辆身份连续且未跨reset的起点。只保存物理状态和路径ID（含角速度），不复用旧动作/标签/隐藏历史。reset恢复对应地图和物理状态，NOD按新episode初始化，使用当前同步policy/NOD的mode动作重新采集与生成目标。状态池随sidecar保存；旧文件从空池开始，兼容模型仍可加载。

保留总诊断，新增`normal_*`/`challenge_*`分组诊断及实际起点使用比例、两类起点新增数/池大小。普通环境不会在自动reset时混入缓冲起点，但仍参与拟合，不能称为独立留出集。应优先检验普通组的危险提前预警与已观测安全窗口正预测比例。Actor、网络、损失及折扣目标未改；道路目标符号问题仍待单独处理。

混合采样版本全套65项测试通过，覆盖起点筛选、跨场景物理/路径恢复、旁路隔离和保存加载；真实短程续训已观察到25%的危险前起点帧。完整训练收益尚未验证。


## 10. 阶段8B：固定κ的PPO安全优势（2026-09-08）

根配置`safety_control_mode="barrier_fixed"`，总开关仍为`is_using_safety_constraint`。`legacy_q`保留阶段7旧Q惩罚及lambda，`off`关闭Actor安全约束。旧JSON缺失mode默认legacy_q；新模式不叠加旧Q损失。8A的`is_using_safety_value_shadow`字段名保持兼容，在8B表示启用独立Value训练，不代表Actor仍处于纯旁路模式。

```text
确定性独立混合起点轨迹 → 继续监督Value
真实随机PPO转移 → 同一在线Value预测V、V_next
  → 固定κ屏障C → 最大正违反P → 冻结的安全优势 → PPO更新Actor/消息聚合器
```

当前V≤0且物理g≤0时，`C=V_next-(1-κ)V`；已危险状态使用固定恢复规则`C=V_next-V`。默认成对/碰撞κ=0.05，道路κ=0.05（dt=0.05时alpha=1/s）。终止后继使用物理g，collector截断使用网络预测；若当前需要的约束后继不可见、车辆generation变化或Value非有限，该车该步回退原任务优势。

`A_used=(1-β*1[P>0])*A_task-β*ν*P`，默认β=0.1、ν=1。β=1为原计划的完整任务屏蔽公式；β=0等价关闭。默认违反样本保留90%任务优势，属于低强度实验，未保证违反动作的总优势为负。沿用原始任务GAE口径、不额外归一化，保留原任务value_target/动作/log-prob。PPO epochs前一次计算并detach，优先回放更新TD误差也不重写冻结安全优势。

至少10个新合同下Value优化批之后启用，召回作为诊断，不再是实验接线的硬前置条件。旧Value或变更κ/ν/β/模式后续训保留兼容权重、重做预热；同合同恢复计数。Value与NOD不收到该安全优势梯度，NOD原有独立训练继续。测试部署仍只运行Actor，没有额外动作过滤器。阶段9的z→κ尚未实现。

新增barrier诊断和实际Actor安全更新minibatch数量写入现有Value指标列表，分清预测质量与真实策略收益。完整训练由用户执行。

8B新增18项测试，全套83 passed；β=0与关闭约束的Actor权重逐项一致，β>0产生实际Actor参数差异，任务目标保护、梯度隔离及加载续训通过。
