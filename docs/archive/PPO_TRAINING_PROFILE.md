# 原版任务 PPO 设置开关

参数 `ppo_training_profile`：

- `current`（默认）：保留 JSON 的 PPO 参数；DGPPO 分支仍按每环境、每辆车沿时间归一化任务优势。旧配置无需修改。
- `original`：参数初始化时统一覆盖为 `num_epochs=60`、`lr=0.0002`、`lmbda=0.9`、`clip_epsilon=0.2`；DGPPO 分支使用原始任务 GAE，不做新增的时间归一化。四项数值对应初次提交 `6426af6`。

`original` 的四项设置优先于 JSON 中的同名字段，最终生效值会在启动时打印，并保存到模型配套 JSON。gamma、熵系数、采样规模、学习率下限等保持配置中的值。安全 Value 的学习率、训练次数、lambda 目标与 DGPPO 安全权重不由该开关改变。

## 用户手动运行

先运行任务学习基线：

```bash
python main_training.py --config configs/archive/dgppo_history/config_ppo_original_task_only.json
```

输出 `outputs/archive/ppo_original_task_only/`。`dgppo_weight=0`，Safety Value 仍旁路采样/拟合，但不影响 Actor；任务优势保留原始值。

在相同 PPO 设置下启用 DGPPO：

```bash
python main_training.py --config configs/archive/dgppo_history/config_ppo_original_dgppo.json
```

输出 `outputs/archive/ppo_original_dgppo/`。与上一组只差 `dgppo_weight=1` 及输出目录。两组均从零训练，使用相同 seed、1 cm 道路余量、20 批安全预热、共享边界和原安全违规任务优势清零机制。

每批 PPO 更新由 15 epochs 增到 60 epochs，PPO 优化工作量约为原来的四倍；总运行时间不一定四倍。训练和仿真由用户手动执行，本次未运行。

训练完成后，将 `main_testing.py` 的 `path`、`utilities/evaluation_tase26.py` 的 `model_paths` 及图表/日志输出路径指向相应新目录。沿用原多车评估设置（1200 steps、8 次仿真），比较边界碰撞、车间碰撞与速度。请使用模型目录中的实际 reward JSON 和权重配套加载。

## 实验边界

此开关恢复四项 PPO 超参数和任务优势处理，不等同于完整回退初始代码，也不保证恢复原模型性能。Actor/任务 Critic 网络、观测与奖励仍使用当前实现，不启用 NOD；Safety Value 在 task_only 中仍会采样和拟合。

关闭优势归一化会改变任务项相对于安全惩罚的尺度。两组统一使用原始任务优势，便于隔离安全更新的影响；不能把 `original + DGPPO` 视为原论文的完全复现。

物理 Safety Value 目标不变，因此其模型契约不变；原版 profile 的屏障更新元数据单独记录原始 GAE 和 PPO 设置，跨更新规则恢复训练时沿用已有机制重新开始安全预热。默认 current 的旧屏障契约保持不变。

已准备只用合成 TensorDict 的单元检查，手动执行：

```bash
python -m pytest -q tests/test_ppo_training_profile.py
```

覆盖默认兼容、参数覆盖和序列化、两组配置差异、检查点元数据，以及预热/关闭安全/启用安全时的优势处理。本轮仅进行静态语法和配置检查，未执行该测试脚本。
