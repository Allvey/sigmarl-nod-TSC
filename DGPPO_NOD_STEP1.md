# 第一步：本地 NOD 与固定 alpha 的 DGPPO 共存

本步骤复用现有 NOD、消息聚合器、Actor 和 Safety Value。NOD 通过消息输入影响 Actor；DGPPO 继续使用固定 `dgppo_alpha=10` 和 `gated` 更新。本步骤不实现意见调节 alpha、不分配双边责任，也不增加网络或动力学损失。

## 配置与数据流

```json
"is_using_nod_opinion": true,
"is_using_nod_actor": true,
"nod_observation_mode": "local_kinematics",
"safety_control_mode": "dgppo",
"dgppo_task_mode": "gated",
"dgppo_alpha": 10.0
```

新增 `nod_observation_mode`：

- `legacy_paths`：默认值，保留旧 NOD 的路径特征与交互掩码，旧配置继续可用。
- `local_kinematics`：只使用本车坐标系中的相对位置、真实相对速度、相对航向、距离、闭合速度、TTC、可见性和速度大小。路径相关的7个通道保留零占位；邻居掩码仅按感知距离确定，不读取邻车参考路径，也不使用对方内部意见。不可见槽位的特征置零。

环境生成、NOD 在线推理及有序序列训练使用同一输入规则。NOD 仍使用既有反事实监督标签；未来轨迹仅用于训练标签，不进入部署输入。原始 Actor 观测仍保留本车导航信息。

已有消息网络与 Actor 输入维度不变：32维基础观测 + 32维消息 + 2维上一步动作。PPO 复用采集时的 detached context，不乱序更新 GRU；NOD 仍独立监督训练。Safety Value 的特征接口、网络、目标和 DGPPO 优势公式保持原样。

## 从现有策略初始化

优先使用独立微调配置：

```bash
python main_training.py --config config_dgppo_nod_fixed_finetune.json
```

- 初始化来源：`outputs/ppo_original_task_only/reward7.36`。
- 输出：`outputs/dgppo_nod_fixed_finetune/`。
- 保留原微调设置：70批、20个成功 Safety Value 拟合批次预热、学习率5e-5、KL阈值0.01。
- Actor 从基础策略扩展输入时，新增消息和上一动作的输入列置零，初始 `loc/scale` 与原策略一致。之后这些列正常接受 PPO 梯度；消息网络的梯度在输入列开始非零后才能进入。
- 基础模型没有 NOD sidecar 时，新实验允许 NOD 随机初始化并独立训练，`nod_freeze_training=false`。前20批 Actor 和消息网络冻结，NOD 仍按既有间隔10批更新。预热不是 NOD 已校准的证明，需查看其指标。
- 兼容的 Safety Value 权重保留，重新计算预热计数。

可选的从零训练配置：

```bash
python main_training.py --config config_dgppo_nod_fixed.json
```

使用250批，输出至 `outputs/dgppo_nod_fixed/`，不加载旧检查点。现有默认 `config.json` 未切换。

## 检查点兼容

NOD sidecar 新增 `observation_mode`。缺少该字段的历史模型视为 `legacy_paths`；不能把它当作兼容的本地运动模型。新实验从无 NOD 的基础 Actor 迁移时允许重新初始化 NOD；测试、续训或加载已有本地 NOD Actor 时，缺少/不兼容的 NOD sidecar 会明确报错，防止随机意见改变策略行为。

`nod_freeze_training=true` 仍要求加载兼容的 NOD 检查点。已经带 NOD 的 Actor 不允许在丢失其 NOD sidecar 后悄悄重新初始化。输入和消息形状改变造成的不完整 Actor 迁移也会报错。

另外修复了原 `Parameters` 丢失显式 `model_name` 的问题，使保存参数往返转换一致。

## 验证

离线检查（不启动环境或训练任务）：

```bash
python -m pytest -q tests/test_dgppo_nod.py
```

覆盖：配置往返、局部输入隔离、在线意见对私有路径不敏感、基础 Actor 迁移后的动作分布、PPO梯度隔离、NOD sidecar兼容、Safety Value复用与预热重置，以及固定alpha的安全优势不读取意见。

手动短程集成检查（会运行两次小规模训练并测试加载，只写 pytest 临时目录）：

```bash
python -m pytest -q tests/test_dgppo_nod_smoke.py
```

先生成一个无 NOD 的基础模型，再微调为本地 NOD + DGPPO，检查冻结预热、实际PPO更新、NOD拟合、context缓存、保存与最终模型加载。

本轮已执行47项离线检查并通过；短程环境集成检查、完整训练及评估留给用户手动执行。功能检查不代表性能提升。

## 训练后检查

- `nod_metrics_list`：`enabled`、`actor_context_ready_ratio`、`actor_message_l2_mean`、`optimizer_updates`、`ece`、`z_counterfactual_gap_correlation`。
- `safety_value_metrics_list`：`finetune_actor_frozen`、`finetune_ppo_updates`、`barrier_ready`、`barrier_active`、`actor_probe_mode_delta_abs`。
- 评估四个场景的道路碰撞、车间碰撞、速度与等待情况。消息非零或动作发生变化都不等同于性能提升。

使用原 `main_testing.py`，将模型目录切到对应输出；脚本按保存配置恢复本地 NOD。测试入口本次未修改。

本步骤得到固定 alpha 的 NOD 基线。第二步再单独加入本车意见调节成对 alpha，并用相同结构、初始化和预算作对照。
