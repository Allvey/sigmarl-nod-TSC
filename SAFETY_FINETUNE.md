# 固定策略预热与安全微调

## 完整任务优势对照实验

```bash
python main_training.py --config config_ppo_original_dgppo_additive.json
```

新增 `dgppo_task_mode`：默认 `gated` 保持违规样本任务优势置零；`additive` 使用完整任务优势减去原有安全惩罚，正负任务优势都保留。这是加性安全惩罚 PPO 变体，不再严格使用 DGPPO 的任务门控，也不保证安全提升。缺失实体、无效后继回退以及预热、零权重行为保持一致。

新配置与下述微调配置仅模式和输出目录不同：同样从 reward7.36 初始化，70 批、前 20 次 Value 拟合冻结 PPO、安全权重 1、KL 阈值 0.01。输出到 `outputs/ppo_original_dgppo_additive/`。评估时将模型和评估输出路径切换到此目录，沿用相同场景、步数和随机种子；对照原微调和源 task-only 的道路碰撞、车间碰撞及速度。

启动日志显示 `task mode=additive`，训练指标 `barrier_task_additive=1`。有效安全更新样本的 `barrier_task_retention_mean=1`；违规样本仍可能获得正优势，这是保留完整任务评价的预期行为。模式保存到配置和屏障元数据，Safety Value 拟合目标不变。

可手动执行 `python -m pytest -q tests/test_ppo_training_profile.py tests/test_dgppo_minimal.py tests/test_safety_finetune.py` 检查默认行为、两种模式的优势计算和配置兼容性。

## 原门控微调对照

```bash
python main_training.py --config config_ppo_original_dgppo_finetune.json
```

加载 `outputs/ppo_original_task_only/reward7.36` 的 Actor、任务 Critic 和 Safety Value，输出到独立目录 `outputs/ppo_original_dgppo_finetune/`。源权重不会覆盖。不要求不存在的 NOD 检查点；意见模块仍关闭。

新增参数：`safety_training_mode` 默认为 `scratch`，保持原训练行为；`finetune` 启用本流程。`safety_finetune_lr=0.00005`、`safety_finetune_target_kl=0.01` 为本次配置的实验值。

1. 前 20 个成功拟合 Safety Value 的批次不进行任何 PPO 优化，Actor 和任务 Critic 权重保持不变。Safety Value 从源权重开始继续拟合，使用新优化器和新的阶段计数。仍采集常规 PPO 数据用于日志，但不会用它更新 Actor。冻结期不参与最佳微调模型选择。
2. 第 21 批起（若前面拟合成功），使用原 DGPPO 安全优势公式和原始任务 GAE，最多 60 epochs，以较小学习率更新。由于现有 PPO 共用优化器，Actor 和任务 Critic 的学习率一起减小；Safety Value 学习率仍为 0.001。微调学习率优先于 `ppo_training_profile=original` 的 0.0002 设置。
3. 每次 PPO 优化后，用该 minibatch 的旧策略动作对数概率与当前策略对数概率估计 KL(old || new)。超过 0.01，停止该批剩余 PPO 更新，Safety Value 照常拟合。估计按每个智能体的联合动作平均。它是采样近似，不是精确 KL；不回滚刚完成的更新，也不限制相对于最初 reward7.36 的累计漂移。

默认总计 70 批，预计 20 批冻结预热、50 批安全微调。PPO 学习率在实际开始更新后才衰减，预热不消耗其衰减进度。本模式不加入在线安全过滤，不修改道路余量、边界、网络结构或安全优势公式。

## 用户手动验证

训练和仿真由用户执行。本轮仅完成静态语法、配置和源文件存在性检查，没有执行训练或下列测试。

```bash
python -m pytest -q tests/test_safety_finetune.py
```

训练日志中查看：

- 预热时 `finetune_actor_frozen=1`、`finetune_ppo_updates=0`、Actor probe 变化应为 0。
- 开始更新后 `finetune_actor_frozen=0`；`finetune_kl_last/max` 和 `finetune_kl_stopped` 反映提前停止。
- `finetune_ppo_updates` 是实际优化次数；旧的 `actor_updates` 仅统计受安全优势影响的次数，含义不同。

将测试脚本模型路径及评估输出路径切换到新目录，沿用 1200 steps、8 次仿真、多车场景，与源 reward7.36 比较边界碰撞、车间碰撞及速度。最佳奖励不是安全最佳的保证，建议同时检查 final 模型。冻结阶段结束前不会写出 reward 最佳微调模型，不要在此时直接用默认最佳模型加载进行评估。

当前默认配置是一个新建的短程微调实验，不是恢复原 250 批训练。模式写入保存参数，更新规则写入 Safety Value 的屏障元数据，避免跨模式恢复时错误跳过预热。该方法是否能实际改善安全尚待验证。
