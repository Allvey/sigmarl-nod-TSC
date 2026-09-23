# 第二步：合作意见调节 DGPPO 成对 alpha

本步骤将已有 NOD 的合作判断直接接入 Safety Value 的约束残差和 PPO 安全优势，不增加网络。Actor、NOD、Safety Value 的结构和物理安全目标不变。

## 计算规则

新增配置：

```json
"dgppo_opinion_alpha": true,
"dgppo_alpha": 10.0,
"dgppo_alpha_span": 5.0
```

对本车 i 与邻车 j，使用采集动作时缓存的 `z_ij`：

```text
当前物理 g_ij <= 0、预测 V_ij < 0，且意见有效：alpha_ij = 10 + 5*z_ij
其他情况：alpha_ij = 10
C_ij = (V_ij(next) - V_ij(current))/dt + alpha_ij*V_ij(current)
```

`z` 是对邻车会采取风险缓解行为的判断。预测安全区内，正意见增大 alpha、适度放宽成对约束；负意见减小 alpha、收紧约束。中性意见精确恢复固定值。道路边界和独立实际碰撞头始终使用固定 alpha，物理安全距离不改变。这不是双方距离预算的严格分配，也不能保证无碰撞。

意见按邻车编号和重生代次映射至 Safety Value 的车辆槽位。缓存缺失、未就绪、不可见、非有限、越界或身份不符时回退固定 alpha；重复邻车编号不采信。跨重生转移仍由原 DGPPO 有效性检查排除。只读取当前动作的缓存，不重新运行 NOD、不读取下一时刻的意见。

保留现有 gated 任务优势与最坏约束惩罚。意见和 Value 不接收 PPO 梯度，PPO 通过动作 log-prob 更新 Actor 及消息聚合器。新增模式默认关闭；旧 JSON 和固定 alpha 路径保持原行为。开关开启时校验整个 alpha 区间满足 `0 < alpha*dt < 1`。`dgppo_alpha_span=0` 可精确退回固定计算。

## 从第一步模型继续

两份配置从同一完整检查点初始化：

```text
outputs/archive/dgppo_nod_fixed_finetune/reward7.33
```

加载 Actor、任务 critic、NOD 和 Safety Value。已在本机检查四个文件存在，并离线验证 NOD/Safety Value 的加载兼容性。两份配置冻结 NOD 权重，仍正常进行在线意见推理；Actor 的消息聚合器会在预热结束后随 PPO 学习。

两份配置除意见开关和输出目录外完全相同，保留70批训练、前20个成功 Value 拟合批次冻结 Actor、微调学习率5e-5和KL阈值0.01。新实验重新预热 Safety Value，使用新优化器；两组均不覆盖第一步输出。

先手动运行短程集成检查：

```bash
python -m pytest -q -s tests/test_dgppo_opinion_smoke.py
```

此测试只读取第一步检查点，在 pytest 临时目录分别执行两组3批小规模微调，检查冻结 NOD、Actor 更新、保存和加载推理。依赖第一步检查点，缺失时会跳过；短程样本不保证触发活跃的成对安全约束。

运行意见调节实验：

```bash
python main_training.py --config configs/archive/nod_history/config_dgppo_nod_opinion_finetune.json
```

运行相同起点、相同预算的固定 alpha 对照：

```bash
python main_training.py --config configs/archive/nod_history/config_dgppo_nod_fixed_control_finetune.json
```

输出分别为 `outputs/archive/dgppo_nod_opinion_finetune/` 和 `outputs/archive/dgppo_nod_fixed_control_finetune/`。这组对照比直接与第一步模型比较更能区分意见调节效果和额外训练效果。

## 训练时检查

在 `safety_value_metrics_list` 中新增以下指标；每批对同一任务优势、同一 Value 预测同时计算固定与意见 alpha，不增加网络前向计算。

| 指标 | 含义 |
| --- | --- |
| `opinion_alpha_enabled` | 配置开关；预热期间也为1，不代表已经产生影响 |
| `opinion_pair_count` | 有有效转移且处于可调节安全区的成对约束数 |
| `opinion_valid_count` / `opinion_missing_count` | 上述约束中意见有效/不可用的数量 |
| `opinion_z_mean` | 有效意见均值 |
| `opinion_alpha_mean/min/max` | 可调节约束上实际 alpha，包含缺失意见时的固定值回退 |
| `opinion_c_delta_abs` | 成对约束残差相对固定 alpha 的平均绝对变化 |
| `opinion_pair_flip_rate` | 成对约束违反判断翻转比例 |
| `opinion_agent_flip_rate` | 智能体整体违反判断翻转比例 |
| `opinion_advantage_delta_abs` | 最终 PPO 优势平均绝对变化，按全部样本统计 |
| `opinion_advantage_changed_count/rate` | 最终优势改变的数量/占有效智能体转移比例 |

预热期间影响指标为0；没有可用样本时均值/极值记0。若残差有变化、最终优势变化为0，可能是道路/碰撞头占主导，或成对残差仍未触发惩罚。不能仅凭 alpha 或残差变化声称 Actor 已受到意见约束。

Safety Value 的漏判能力和最终碰撞、通行、停滞情况仍需评估。推理时延续当前直接 Actor 出动作的方式；本步骤没有增加在线安全过滤器。

## 验证记录

新增离线测试 `tests/test_dgppo_opinion.py`，覆盖方向与边界、零意见/零幅度等价、私有缓存对齐与排列、缺失/异常回退、重生、道路头隔离、预热、配置往返、检查点兼容，以及构造临界交互下最终优势和 PPO 梯度的变化。

本轮第一组离线/回归检查74项通过、5项环境测试排除；补充检查59项通过（两组存在重复测试）。补充检查包含旧 `test_safety_barrier.py` 自带的两项临时环境训练测试，均通过。新意见分支的短程集成测试、完整训练与评估尚未执行。输出中的 Matplotlib/pyparsing 弃用警告未导致检查失败。

本轮未修改 `main_testing.py`；训练后可手动将其模型路径切换到对应输出目录。

## 固定车辆的实时意见/alpha 显示

后续新增测试可视化：`main_testing.py` 默认开启 `parameters.is_visualize_nod_alpha = True`，右上角显示固定车辆对可见 NOD 邻居的意见和 alpha，并随视频保存。

```python
parameters.visualize_observed_neighbors_agent_index = 0  # 0 对应画面车辆 1
parameters.is_visualize_nod_alpha = True
```

例如 `A1 -> A3  z=+0.150  alpha=10.00  [non-safe: fixed]` 表示车辆1对车辆3持正意见，但当前物理/预测安全条件不允许调节，实际训练规则仍使用固定 alpha。`opinion` 表示使用意见映射；`missing: fixed` 表示意见无效而回退；`fixed mode` 表示固定 alpha 对照。没有可见邻居时明确显示空状态；缺少兼容的 Safety Value 时显示 `alpha=N/A`，不使用随机 Value 冒充有效结果。

此 alpha 是用已加载 Safety Value 对动作执行前的缓存状态计算的训练规则诊断值，不是运行时安全干预。画面显示动作执行后的车辆位置，面板的 `Decision t=...` 标明对应的执行前时刻。显示不会再次推进 NOD、采样动作或改变策略；为判断是否回退，会额外进行一次所显示环境的 Value 前向计算。

## 可配置的意见映射增益

新增 `dgppo_alpha_gain`，默认1，旧配置和模型保持原行为。映射改为：

```text
alpha_ij = dgppo_alpha + dgppo_alpha_span * clip(dgppo_alpha_gain * z_ij, -1, 1)
```

仍仅用于原来的有效、安全车辆对。原始意见 z、NOD/Actor 输入、道路与碰撞头及所有固定值回退规则不变。增益2使 `z=-0.072` 对应的 alpha 从9.64变为9.28，但范围仍为[5,15]；中性意见保持10。可视化读取模型保存参数并使用同一映射，显示的 z 仍为原始意见。

从已完成的意见模型 `outputs/archive/dgppo_nod_opinion_finetune/reward6.98` 继续微调：

```bash
python main_training.py --config configs/archive/nod_history/config_dgppo_nod_opinion_gain2_finetune.json
```

新配置设 `dgppo_alpha_gain=2.0`，输出至 `outputs/archive/dgppo_nod_opinion_gain2_finetune/`。保留现有意见配置的其他超参数（包括当前 `nod_safe_distance=0.20`），冻结 NOD，重新进行20批 Value 预热，训练总计70批。增益变更记入约束合同，兼容的 Value 权重保留；增益1保持旧约束合同以支持原模式续训。源模型及原输出目录不被覆盖。此配置用于继续训练，不是与旧实验相同训练预算的对照。

新增指标 `opinion_alpha_gain`、`opinion_mapped_z_mean`、`opinion_alpha_saturation_rate`，分别记录增益、映射后意见均值和有效可调意见达到上下界的比例。放大映射不代表合作判断更可靠。

本次执行58项离线检查：57项通过；既有 `test_config_roundtrip_and_matched_control` 因目前意见配置与固定对照配置的 `nod_safe_distance` 分别为0.20和0.25而失败。这一配置差异保留，原先文档关于“两组仅开关与目录不同”的描述已不适用于当前文件。新增增益及可视化一致性检查全部通过，也已离线确认 reward6.98 的 NOD/Value 加载兼容。未启动训练或仿真。

训练完成后，将 `main_testing.py` 的 `path` 改为新输出目录，才能加载增益2训练后的策略和显示规则。直接测试旧模型仍显示增益1的结果。
