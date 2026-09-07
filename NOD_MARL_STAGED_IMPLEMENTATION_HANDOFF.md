# NOD-MARL 安全优先实施方案与新 Session 交接

> 更新时间：2026-09-07  
> 用途：新开发 session 的首要阅读材料。本文以当前工作区和实际训练结果为准，区分“已经完成”“当前未提交”“后续计划”，避免重复实现或按旧方案误改。

> 本轮范围调整：用户决定暂缓 Deadlock Critic，优先安全。近期只推进任务 Critic + Safety Critic + NOD/Actor；死锁标签补充、Deadlock ensemble、死锁 Actor 约束及死锁对偶网络均移出实施范围。本文的新顺序替代旧交接计划中的双约束路线；原方法文档保留为长期研究参考。
> 本轮只更新实施方案，不修改运行代码或配置。当前 `config.json` 的 `is_using_deadlock_critic` 仍为 true，实际停用列入阶段7收尾；不能把“计划暂停”写成“代码已停用”。

## 0. 当前状态与近期顺序

| 阶段 | 状态 | 本轮决定 |
|---|---|---|
| 1—4：NOD 与 Actor 接入、稳定性修复 | 主体实现完成，有完整训练记录 | 保留现有结构和标准差上限 |
| 5：Safety Critic | 实现完成，低估问题仍待改善 | 继续使用并验证可靠性 |
| 6：Deadlock Critic | 实现完成，但已有多seed训练没有正样本 | 暂停，不再作为安全工作的前置条件 |
| 7：Safety Actor 约束 | 原型可运行，权重会归零，未验收 | 首先收尾 |
| 8：Safety Critic 可靠性 | 未实施 | 第二优先级，逐项改善数据和预测 |
| 9：安全泛化训练与验收 | 未实施 | 替代旧阶段9“死锁约束接入” |
| 10—12：状态相关安全约束、动作一致性、联合微调 | 未实施 | 后续按证据逐项推进，不作为近期验收前置条件 |

近期完成标准是：安全约束持续有效、碰撞和风险低估改善、目标场景仍有合理通行能力；不是必须完成全部网络扩展。以下所有“后续修改”均为计划，未实施时不得当作当前能力。

## 1. 不可改变的项目约束

后续所有阶段都必须遵守：

1. 训练入口保持为 `python main_training.py`。
2. 测试入口保持为 `python main_testing.py`。
3. 配置仍从根目录 `config.json` 读取，不另建一套 CLI。
4. `n_iters` 保持为 `250`。
5. 每个阶段结束时都必须是可以从头训练、保存 checkpoint、再用原测试入口加载的完整版本，不能提交只有半条数据流的中间状态。
6. 不恢复已经删除的 TopologyLearner、TopologyActionPredictor 和 opponent modeling。
7. 不采用此前效果不好的残差式意见接入。
8. 不要求每阶段性能必然高于 base model，但不得用“后面还没实现”解释运行错误、NaN、无法保存或无法测试。
9. 工作区可能包含用户或其他 session 的未提交修改；禁止 `git reset --hard`、`git checkout --` 或覆盖无关文件。
10. 本轮优先级为安全优先、同时报告通行效率；不新增死锁学习目标，不以死锁正样本验收阻塞安全主线。

## 2. 当前仓库快照

当前 `HEAD`：

```text
389024a 修复 testing 问题
e0e7ed6 阶段6 3 critic
6d10c8c 阶段5 加入 safety critic
95a88b0 阶段4 效果好了很多
9a2a08d 阶段4
b253aee 阶段4，存在策略坍塌
ae4c159 阶段1-3调整
c0e078a 阶段1-3
```

当前已提交到 `HEAD` 的能力：

- 简化后的 NOD 意见模型及独立训练；
- NOD 消息接入分布式 Actor；
- Actor 方差上限修复；
- 独立 Safety Critic；
- 独立 Deadlock Critic；
- 新旧模型的训练/测试加载修复。

当前工作区存在未提交的“阶段 7：Safety 约束进入 Actor”实现，主要涉及：

```text
MODEL_STRUCTURE.md
config.json
utilities/helper_training.py
utilities/mappo_cavs.py
utilities/nod_marl/safety.py
tests/test_safety_training.py
tests/test_safety_constraint.py（未跟踪）
```

W&B 日志文件也是运行时改动，不应当作为核心源代码处理。

截至本文生成时：

```text
git diff --check：通过
pytest：35 passed
```

测试使用的解释器为：

```bash
/Users/zhangxiaotong/anaconda3/envs/sigmarl-nod/bin/python -m pytest -q
```

用户正常训练和测试仍使用原命令，不需要改成上述绝对路径。

## 3. 当前真正生效的网络结构

### 3.1 NOD 与 Actor

```text
每条有向交互边的 20 维物理特征
  └─ 相对位置、相对速度、朝向、距离、TTC、冲突点、ETA差、重叠风险等
        │
        ▼
共享 GRUCell：20 → 64
        │
        ├─ 6维单调物理风险量
        ├─ 1维风险注意力
        └─ 1维有界意见 z
        │
        ▼
每边 72维 context
        │
        ▼
LayerNorm → 72 → 64 → 32 → 邻居维 masked attention
        │
        ▼
32维 NOD 消息，每维限制在 [-0.1, 0.1]

Actor 输入：
32维原始局部 observation
+ 32维 NOD 消息
+ 2维上一时刻动作
= 66维

共享分布式 Actor：66 → 256 → 256 → 4
                              ├─ loc[2]
                              └─ scale[2]，当前限制 scale ≤ 1.0
                                      │
                                      ▼
                                  TanhNormal
                                      │
                                      ▼
                              二维车辆连续动作
```

Actor 在执行时只使用本地观测、本车可见邻居形成的 NOD 消息和上一动作。集中式 Critic 不参与部署动作计算。

### 3.2 当前代码中的三类 Critic，以及计划保留的主线

```text
任务 Critic
  └─ MAPPO 集中式 State Value，使用折扣加法 GAE/PPO target

Safety Critic
  └─ 全局观测/物理状态 + 当前联合动作
  └─ 预测 h=[1,4,8,16] 内最大安全违反量
  └─ target 不折扣

Deadlock Critic
  └─ 全局观测/物理状态 + 12维时序死锁增广状态 + 当前联合动作
  └─ 预测 h=[1,4,8,16] 内最大死锁违反量
  └─ target 不折扣
```

Safety 与 Deadlock 网络、优化器、随机数流和 checkpoint 相互独立，也不与 NOD/PPO 共享参数。

下一次代码修改将关闭 Deadlock 分支，主线只保留任务与 Safety Critic。保留现有死锁实现和旧 sidecar 的可选兼容，不进行大范围删除。停用后必须验证没有死锁监督更新或 Actor 损失；旧文件可以存在，不要求继续生成死锁 checkpoint。

## 4. 已完成阶段

### 阶段 1—3：NOD 语义链路和行为隔离

已完成的实质内容：

- 从原始物理状态构造稳定的有向交互边和 20 维成对特征；
- 维护 agent identity、generation 和边生命周期，避免重置后串接旧历史；
- 使用 GRU 表达成对历史；
- 构造风险注意力、证据似然、自由能和一维 KL 意见更新；
- 生成反事实软标签，训练 NLL 与校准 BCE；
- 验证意见有界性、求根残差、曲率和边重置行为；
- NOD 参数与 PPO 参数、随机数流和梯度隔离；
- 采集时按时间顺序推进 NOD，PPO replay 只读取 detached context，不乱序重放 GRU。

这部分当前仍在工作，不应重新写一套 topology 或动作预测网络。

### 阶段 4：NOD 消息进入 Actor，并修复方差坍缩

已完成的实质内容：

- 删除旧 topology/action predictor/opponent modeling 训练链路；
- 将 72 维边 context 通过排列不变注意力聚合为 32 维消息；
- Actor 输入改为 `[observation, NOD message, previous action]`；
- 消息聚合器使用独立较小学习率 `5e-5`；
- NOD 默认每 10 个 rollout 更新一次；
- 风险权重限制在 `[0,1]`；
- Actor 的 `scale` 增加固定上限 `1.0`。

曾确认原始状态相关 scale 会从约 `0.3` 膨胀到数百，使随机动作大量饱和到边界并导致车道碰撞。实际提交的修复是 `BoundedNormalParamExtractor`：保留当前状态相关方差结构，只在输出处执行 `scale.clamp_max(1.0)`。

历史上 `NOD_MARL_Actor_Stability_Modification_Plan.md` 记录过“改成状态无关全局方差”的备选方案。该文件在本轮磁盘检查中未找到；该方案没有落地。新 session 应以当前 scale 上限实现为准，不按旧 IDE 标签或备选方案重写 Actor。

### 阶段 5：独立 Safety Critic

已完成的实质内容：

- 环境输出车辆间距、道路边界和碰撞的 signed safety margins；
- `g_s≤0` 表示满足，`g_s>0` 表示违反；
- Safety Critic 输入当前状态和真实执行联合动作；
- 监督目标为有限视野内从当前状态开始的最大违反量；
- 终止、agent generation 变化和 rollout 截断均有有效性 mask；
- 危险低估样本在 Smooth L1 中获得更高权重；
- 在新 rollout 上先记录预测误差，再训练 Critic，避免把拟合后误差当作泛化误差；
- 保存独立 `*_safety_critic.pth`。

阶段 5 不改变 Actor 行为。

### 阶段 6：独立 Deadlock Critic（实现保留，后续工作暂停）

已完成的实质内容：

- 建立持续低速、路线进展、冲突车组、规则许可和局部可解除动作共同定义的时序死锁状态；
- 正常等待、无冲突单车停车和物理不可解除状态不会仅凭低速被标记为策略性死锁；
- 维护 12 维 deadlock augmented state；
- Deadlock Critic 使用独立有限视野最大值 target；
- 保存独立 `*_deadlock_critic.pth`；
- 输出 recall、precision、漏判、误报、持续时间、有效样本等指标。

阶段 6 同样不改变 Actor 行为。

已有多seed训练中，死锁正样本和有效计时均为0，不能认为该网络已学会识别死锁。按本轮用户决定，不再投入死锁场景构造、标签调整、训练或 Actor 接入；这项未验收事项不会阻塞阶段7—9。仍通过速度、停车时长和通行率观察策略是否过度保守，无需为此训练 Deadlock Critic。

## 5. 当前可靠的性能参照

### 5.1 阶段 4—6、未加入 Safety Actor 约束

文件：`outputs/testing_random/reward7.43_data.json`

```text
迭代数：250
最佳奖励：7.427，第157轮
最终奖励：6.087
最后20轮平均奖励：6.044
最后20轮平均总碰撞率：0.00285
```

这证明 `scale≤1.0` 后，当前 NOD→Actor 结构能够完整训练，不再出现早期版本的方差型策略坍缩。后续阶段应以此作为主要行为参照，而不是已经坍缩的 `reward4.84`。

### 5.2 阶段 7 的历史中途快照

历史文件名：`outputs/testing_random/reward5.86_data.json`。最佳奖励文件可能随训练更新而替换；不能假定该路径仍存在，也不能由这个快照推断训练已中断。新 session 应扫描当前 `reward*_data.json`，按配置中的 seed 和 `is_using_safety_constraint` 找到对应运行。

旧版交接文档记录到 `101/250` 轮：

```text
最佳奖励：5.857，第99轮
最近20轮平均奖励：约5.18
```

这是不完整训练快照，不能据此判断最终效果。其权重归零问题在后续读取的123轮快照中仍存在；进一步分析应以实际可读的最新JSON为准。

## 6. 当前未完成阶段 7：Safety 约束进入 Actor

### 6.1 已经实现但未提交的内容

当前工作区已经实现：

- `is_using_safety_constraint` 开关；
- Safety Critic 完成 10 个有效 rollout 后启用约束；
- 每个 PPO minibatch 用当前 Actor 重新计算确定性名义联合动作；
- Safety Critic 参数冻结，只保留 `risk → action → Actor` 梯度；
- h=1 不参与未来动作约束，只使用 h>1；
- 当前已经发生安全违反的状态被排除，首版只学习“从安全状态避免进入危险”；
- 全局标量约束权重及上下界、对偶更新、checkpoint 恢复；
- `actor_constraint_*` 指标；
- 单元测试和最小功能 PPO 测试。

当前损失近似为：

```text
L = L_PPO + lambda_s * mean(relu(Q_s + safety_margin))
```

这是工程化标量约束，不是方法文档最终要求的状态相关对偶变量，也不是严格安全保证。

### 6.2 当前必须先解决的问题

在阶段 7 的 101 轮训练中：

```text
lambda_s 在第11轮开始生效
lambda_s 在第16轮达到约0.0326
lambda_s 在第50轮下降到0
第50轮后约束损失持续为0
最近阶段预测违反率仍约0.32
Safety Critic 最近20轮 unsafe recall 约0.836
Safety Critic 最近20轮 underestimate rate 约0.237
worst underestimate 最近仍约1.5
```

直接原因是当前对偶权重使用所有 eligible 状态的“有正有负的平均 signed risk”更新。大量负风险样本会抵消危险尾部，因此即使约30%的状态预测违反，平均 risk 仍为负，最终把全局权重推到0。

因此阶段 7 目前只能称为“可运行的原型”，不能宣布完成。Deadlock Actor 约束已移出近期范围。

### 6.3 阶段 7 的收尾修改

下一 session 应首先在现有未提交代码上完成以下实质修改：

0. **停用 Deadlock 分支**：根配置设为 `is_using_deadlock_critic=false`，保留源码及旧 checkpoint 兼容；核对使用旧训练JSON的测试入口不会无意重新启用该分支。关闭它不代表 Safety 自动可靠，也不改变已有Actor权重。
1. **可靠性门控**：不能只按 `warmup_batches=10` 启用约束；同时要求 Safety Critic 已见到足够危险样本，并满足配置化的 recall/underestimate 条件。门控不满足时继续训练 Critic，但约束 Actor 保持关闭。
2. **危险尾部对偶信号**：标量权重更新不能继续使用全部 signed risk 的普通均值。先采用正违反量 `mean(relu(q))` 与显式风险预算比较的最小方案，预算、上下限和无违反时的更新规则必须写清楚；不同时引入多种尾部风险算法。若后续需要CVaR，作为独立对照再改。避免危险样本被负风险抵消，也监控权重长期顶到上限但碰撞未改善的情况。
3. **区分三种量并分别记录**：普通 signed mean、positive/tail risk、violation ratio。Actor penalty 和对偶更新使用哪一种必须在代码和指标名中明确。
4. **显式梯度隔离断言**：约束反传后 Safety 参数梯度必须为空，Actor（包括消息聚合器）必须能收到非零梯度。
5. **继续保持 scale 上限**：Safety 约束不能替换或绕过 `BoundedNormalParamExtractor`。
6. **checkpoint 兼容**：旧 Safety sidecar 缺少新门控/对偶状态时重新执行可靠性预热，不影响 policy/critic/NOD 加载。
7. **有效状态与启用状态可解释**：分别记录门控就绪、权重是否为正、eligible样本数量和实际非零约束损失。使用新数据上拟合前的统计或独立验证数据做门控，避免用刚拟合的训练误差决定可靠性；门控长期不通过时报告原因并转入阶段8改进Critic，不放宽指标掩盖问题。

阶段 7 收尾后仍使用原命令完成一次 250 轮训练。验收重点：

- 无 NaN/Inf、无动作方差坍缩；
- 约束启用条件和关闭原因可从指标解释；
- 当危险尾部持续为正时，`lambda_s` 不应长期为0；
- Safety 低估率、碰撞率和通行奖励同时报告；
- policy、任务/Safety Critic 和 NOD checkpoint 能正常保存、兼容加载；缺少或停用死锁 sidecar 不阻塞入口。Safety 与车辆数不兼容时必须明确报告，不能将随机初始化评价当成安全控制能力。
- 通行率、停车时间和碰撞暴露量同时报告，不能通过全体停车获得表面上的安全改善。

## 7. 后续分阶段实施顺序

每阶段只改变一个主要机制，并保持独立训练、保存和测试能力。阶段7—9是近期交付主线；阶段10—12是按验收结果选择的后续扩展，不为凑齐网络而必做。

### 阶段 8：Safety Critic 可靠性与保守评价

目的：降低危险漏判和乐观低估，让阶段7能够获得可信的动作风险信号。只修改 Safety，不扩展 Deadlock。

按改动量递增分两步执行，各自做对照：

1. 先改善危险、近碰撞和低估样本的覆盖及分层采样；保留现有有限视野目标、重置mask和独立优化器。独立统计各视野的召回、误报、低估幅度及样本量。
2. 若单模型低估仍妨碍验收，再增加第二个独立初始化、独立样本顺序的 Safety 成员，用逐视野最大值作保守聚合；增加成员分歧和聚合前后低估指标，并实现明确的单模型checkpoint迁移。

验收：在相同验证场景和样本口径下低估改善，误报和停车比例没有明显恶化；多个模型取最大值仍是经验保守近似，不是安全证书。若门控长期关闭，应优先改善数据和模型，不靠降低可靠性要求强行启用。

### 阶段 9：安全场景覆盖与多seed泛化验收（替代旧死锁接入阶段）

目的：解决当前4车交叉口训练、8车匝道测试之间的场景差异。现有 `cpm_scenario_probabilities=[1,0,0]` 实际只覆盖交叉口；`CPM_mixed` 名称不表示训练过汇入。

实施顺序：

1. 阶段7修复先保持原4车配置，固定seed比较安全约束开/关，避免把换场景的效果归因于约束。
2. 再给CPM训练加入汇入样本，比例单独配置和记录；保留交叉口回归测试。CPM汇入不等于 `on_ramp_1`，目标地图适配训练与未见地图测试必须分开标明。
3. 在 `on_ramp_1` 先验证4车，再以6车、8车逐级增加密度；若在目标地图训练，保留独立初始条件和其他地图用于泛化验证。车辆数变化时Safety输入维度不兼容，必须重新训练相应Safety或在另一个独立改动中支持可变车辆数，不能直接沿用不兼容的安全模型。
4. 至少使用3个固定训练seed及共同的评估seed集；分别报告默认确定性测试和训练随机动作下的结果，不混用口径。最好模型和最终模型均评估，不能只用最高训练奖励作结论。

验收指标：车辆碰撞事件、越线事件、对应暴露时间或车辆步数、通行率、平均速度、长时间停车比例、Safety低估及约束权重；跨seed报告离散程度。区分碰撞重置和出口正常重置。优先比较相同场景/密度/动作模式下安全约束开关的差异，不能把4车与8车的原始事件数直接当作模型优劣比较。

### 阶段 10：状态相关安全约束（条件性扩展）

仅当标量权重仍无法兼顾危险状态和正常通行，或需要进一步对齐方法目标时实施。先完成阶段7—9，确认问题并非数据覆盖或Critic低估导致。

- 只新增安全乘子网络 `nu_s(y)`，不新增 `nu_d(y)`；输出非负且有界，只在集中式训练时使用。
- Actor更新时冻结Safety和乘子参数；乘子更新时冻结Actor与Safety，保持独立优化器和checkpoint。
- 保留阶段7标量模式作对照，检查输出全零、全顶上限及高风险状态的权重分布。
- 状态相关乘子本身不保证严格安全优先，仍需用碰撞、低估和通行结果验收。

### 阶段 11：动作变化率、实际动作与危险状态处理

当评估发现动作抖动、随机动作风险或危险状态恢复缺口时，分别处理，不能一次混改。当前previous action只是输入；当前Safety约束仅评价名义动作，并且排除了已违反状态。

- 动作幅值/变化率限制需采用正确参数化和相应log-prob；不能采样后平滑或裁剪，却继续用修改前的PPO概率。
- 明确训练随机动作与测试确定性动作的风险差异；收集、Critic监督和环境执行必须使用一致的实际动作。
- 对已危险或约束不可行状态先补诊断和测试，再单独设计最小风险恢复/后备机制。若引入干预，记录介入次数并分别评估Actor与完整系统，不能用测试时过滤掩盖训练策略问题。
- 保持分散执行边界：当前集中式Safety不能未经设计直接作为本地执行过滤器。新增机制也不得宣称未经验证的安全保证。

### 阶段 12：有控制的 NOD 联合微调（最后考虑）

仅在安全主线稳定、且有证据表明固定NOD语义限制性能时实施。当前循环NOD对PPO是detached，只有消息聚合器接受任务/安全梯度。

- 从有序原始rollout重建历史，用独立低学习率步骤微调；禁止在打乱PPO minibatch中乱序训练GRU。
- 保留NLL、校准BCE、单调性、曲率和KL求根检查；只允许任务与安全目标的梯度进入意见层，冻结相应Critic参数。
- 保存Actor/NOD快照；校准或安全指标退化时回滚，保留不联合微调的对照模式。此阶段不恢复死锁目标。

## 8. 后续暂时不要做的事情

近期范围内不恢复死锁开发；阶段7—9尚未验收前，也不要：

- 直接把 Deadlock loss 接入 Actor；
- 直接实现状态相关 dual；
- 删除原始 32 维 observation；
- 因为 Actor 输入是 66 维就压缩到很小；
- 恢复 topology/action predictor；
- 移除当前 scale 上限；
- 用测试时安全投影掩盖训练策略问题；
- 在执行动作后做未计入 log-prob 的额外平滑或裁剪；
- 把任务与安全 target 合并成一个普通折扣回报。

## 9. 关键文件导航

| 文件 | 当前职责 |
|---|---|
| `NOD_MARL_Methodology_Plan.md` | 完整方法的长期参考；死锁及双约束部分按本计划暂缓 |
| `MODEL_STRUCTURE.md` | 当前实际网络结构和已实现阶段说明 |
| `utilities/nod_marl/interaction.py` | 20维成对物理特征和有向边 |
| `utilities/nod_marl/opinion.py` | NOD 历史、似然、风险注意力和意见求解 |
| `utilities/nod_marl/trainer.py` | NOD 在线状态、序列训练、checkpoint 和指标 |
| `utilities/nod_marl/policy.py` | NOD 消息聚合与 66维 Actor 输入 |
| `utilities/nod_marl/safety.py` | Safety Critic、有限视野 target、当前阶段7约束原型 |
| `utilities/nod_marl/deadlock.py` | 已有死锁实现，计划停用并保留兼容，不继续扩展 |
| `scenarios/road_traffic.py` | 环境 observation、NOD/Safety/Deadlock 物理字段 |
| `utilities/mappo_cavs.py` | Actor/Critic 构造、PPO、三类辅助训练和保存主流程 |
| `utilities/helper_training.py` | Parameters、SaveData、collector、checkpoint 辅助逻辑 |
| `config.json` | 保持原接口的全部开关和超参数 |
| `tests/test_safety_constraint.py` | 当前未提交的阶段7核心测试 |

## 10. 新 Session 开始工作的建议顺序

1. 先读本文的新范围与顺序，再看 `MODEL_STRUCTURE.md` 的当前实现；原方法第9—16节中的死锁和双约束目标暂缓。
2. 执行 `git status --short`，确认并保护阶段 7 的未提交修改。
3. 阅读 `utilities/nod_marl/safety.py` 中 `actor_loss()`、`finish_actor_update()` 和 checkpoint 逻辑。
4. 从当前 `outputs/testing_random/reward*_data.json` 中按seed和安全约束开关定位最新记录，核对 `lambda_s→0` 与违反比例，不依赖可能已被替换的历史奖励文件名。
5. 先按第6.3节停用Deadlock并收尾阶段7，再按阶段8改善Safety、阶段9验证场景泛化。若门控暴露Critic不可靠，则进入数据/模型改进，不宣布约束已有效。
6. 修改后运行完整 pytest；当前基线是 `35 passed`。
7. 用户随后会手动执行完整训练。不要自行改变 `main_training.py`/`main_testing.py` 的使用方式，也不要改变 `n_iters=250`。

## 11. 交接定位

当前项目已经不是“阶段 4 尚未解决坍缩”的状态：Actor scale 上限已经消除了主要方差坍缩，阶段 4—6 的完整运行达到最佳奖励 `7.43`。当前真正的工作点是阶段 7 Safety Actor 约束原型：代码和测试已存在，但全局对偶权重会被平均安全样本推到0，且 Safety Critic 的危险低估仍偏高。

新的近期推进顺序为：

```text
停用 Deadlock（保留已有实现）
  → 收尾 Safety Actor 约束
  → 改善 Safety 数据与风险低估
  → 汇入场景覆盖、车辆密度与多seed安全验收

后续按证据选择：
状态相关安全乘子 / 动作一致性及危险状态处理 / 可回滚的 NOD 联合微调
```

每一步都保持现有训练/测试入口、`n_iters=250` 和完整 checkpoint 数据流不变。
