# DGPPO 风格最小安全分支

2026-09-10。运行入口：

```bash
python main_training.py --config configs/archive/dgppo_history/config_dgppo_minimal.json
```

修订版从零训练，输出 `outputs/archive/dgppo_minimal_v2/`，保留原失败实验 `outputs/dgppo_minimal/`。默认 `configs/archive/staged_history/config.json` 及阶段 7/8A/8B/9 的算法分支保持不变。完整训练由用户执行。训练完成后，将 `main_testing.py` 的 `path` 或评估脚本的模型目录设为该输出目录；评估日志目录也需相应设置。本次没有改变用户的测试/评估路径。

无安全更新的对照使用 `python main_training.py --config configs/archive/dgppo_history/config_dgppo_task_only.json`，输出 `outputs/archive/dgppo_task_only/`。相对修订版仅将 dgppo_weight 设为 0 并更换输出目录，Safety Value 仍正常采样/训练，但不影响 Actor。两组使用相同任务优势归一化，不能把对照组称为完全未修改的原 MAPPO。

## 实现范围

保留 MAPPO Actor、任务 Critic、20 维成对输入和 PairSafetyValue 网络。新增 `safety_control_mode="dgppo"`，关闭意见、旧动作 Q 安全惩罚和 Deadlock；不增加测试时动作过滤。

- Critic 目标：官方 `compute_dec_ocp_gae` 风格的折扣最大值 n 步目标与精确 lambda 混合；默认跨有效约束取当前最大违反量作为折扣偏置。不是把 lambda 混合放到非线性的 max 内部。
- bootstrap 使用本批更新前网络预测；不使用 EMA 目标网络。损失为有效输出上的 `0.5 * mean((V-target)^2)`，无类别/低估加权。
- 每批先使用同一更新前策略采集随机及确定性轨迹。安全优势在 PPO 前固定，确定性轨迹用于独立拟合安全 Value。任务 value_target、动作和采样 log-prob 不变。
- 安全导数 `C=(V_next-V)/dt+alpha*V`；每辆车全部有效约束满足 C<=0 时保留任务优势，否则清零。惩罚为 `weight * max(relu(C+eps))`，即使 C 在 [-eps,0] 内也有小惩罚。危险状态仍要求 Value 下降。
- 任务优势按每个环境、每辆车的时间维归一化，然后组合安全项；混合优势不再整体中心化。关闭安全、weight=0 和预热期间也使用相同归一化，防止安全启用时改变任务学习尺度。ClipPPOLoss 保持 normalize_advantage=False。
- 使用本车观测及邻居可观测运动学。旧 NOD 输入中依赖邻居参考路径的第 9–15 维在新分支置零，邻居 mask 只由感知范围确定，距离标签来自本地相对距离。网络输入维度不变。新配置禁止观测邻居参考路径。
- 碰撞终止使用 VMAS 重置前 info 中的物理违反量；断开跨车辆重生的目标。邻居退出观测会截断其目标时域，使用前一有效转移自己的 bootstrap。
- 对部分后继信息缺失的 PPO 样本，已知违规仍可惩罚；仅有未知项时回退到归一化后的任务优势，并不标记为已证安全。这是针对本项目的适配。

## 修订版配置

| 参数 | 值 |
|---|---:|
| seed | 5080868027432654403 |
| 计划批数 | 250 |
| 随机 PPO 采样 | 4096 环境步/批 |
| 确定性安全采样 | 32 环境 × 128 步/批 |
| PPO epoch | 15 |
| 安全 Value epoch | 1 |
| minibatch | 512 |
| Actor/任务 Critic lr | 0.0003 |
| Safety Value lr | 0.001 |
| gamma / lambda | 0.99 / 0.95 |
| dgppo_alpha / dt | 10 / 0.05 |
| dgppo_eps | 0.01 |
| dgppo_weight | 1 |
| dgppo_schedule | false：固定安全权重，不自动翻倍 |
| safety_barrier_warmup_batches | 20 |

alpha*dt=0.5，不能将 alpha=10 与旧 kappa=0.05 直接比较。原 safety_barrier_strength、kappa、nu 不控制此分支；其控制项为 dgppo_alpha/eps/weight/schedule。weight=0 或关闭安全约束时只使用归一化任务优势。前20个成功拟合安全Value的批次只训练任务策略和旁路安全Value，第21批开始安全更新；20批预热是本项目的适配选择，不是可靠性保证，也不是经过完整调参得到的最优值。

安全采样量由原阶段 9 的 256 步提高到 4096 步。首版每批仅1次PPO更新、无预热且安全权重自动翻倍，完整训练最高奖励约-0.18。修订版采用15次PPO更新、20批预热及固定安全权重；它是项目适配实验，不是对阶段9的单变量消融，也不是官方默认超参数复现。风险目标和屏障公式不变。

## 与完整 DGPPO 的区别

修订依据：`outputs/dgppo_diagnostics/20260910_123203/` 下完成同seed的两组24批无安全更新短测。学习率按原250批进度运行，不因提前结束而加快衰减；每组98,304步随机采样及等量确定性采样，使用相同归一化，仅PPO epoch不同。

| 指标 | 每批1次更新 | 每批15次更新 |
|---|---:|---:|
| 最高训练奖励 | 0.852 | 2.719 |
| 最后8批平均奖励 | 0.375 | 1.866 |
| 末策略确定性短测平均速度 | 0.594 | 0.702 |
| 末策略确定性短测平均单步奖励 | 0.0260 | 0.0398 |

这支持增加更新次数，但并不能证明20批预热后的安全更新一定有效。权重为0时1次更新也能获得正奖励，说明首版失败不能完全归因于更新预算。每组目录中保留输入快照、模型、训练历史和 `diagnostic_summary.json`；预热时长及固定安全权重仍需修订版完整训练验证。

这不是完整 DGPPO 复现：保留原前馈/均值聚合网络，没有 Graph Transformer 或 GRU；保留项目的每车任务 GAE、物理约束、车辆重置语义和 TorchRL PPO 采样方式。现有风险头为每个邻居的距离加道路/碰撞，数量及定义也不同于官方环境。已对齐安全目标和更新公式，不移植论文的形式安全保证。

新检查点记录目标递推、输入语义、lambda、损失、归一化及安全更新合同。旧 Safety Value 检查点不能在新语义下直接复用。支持保存/加载新模型；修订版尚未运行完整训练，没有碰撞率改善结论。

## 验证

```bash
python -m pytest -q tests/test_dgppo_minimal.py tests/test_safety_value.py tests/test_safety_barrier.py tests/test_safety_opinion_barrier.py tests/test_safety_constraint.py
```

首版84项测试通过（19项新分支、65项旧分支）。本轮补充任务归一化一致性、真实预热切换及对照配置一致性覆盖，新分支22项测试通过。包括 float32/float64 和多种 lambda 下的官方 DP NumPy 对照、终止/身份截断、填充约束屏蔽、正风险恢复、违规动作梯度方向、无邻居路径依赖、配置与检查点隔离、真实 VMAS 碰撞标签、三批短程 PPO 更新及保存加载。数值对照实现保留上游 MIT 许可。

正式训练后检查实际碰撞、提前预警召回、违规优势正值比例、动作分布变化以及无条件速度/停车；短程测试只验证实现链路，不替代这些性能评估。

## 来源

- https://github.com/MIT-REALM/dgppo/blob/main/dgppo/algo/utils.py
- https://github.com/MIT-REALM/dgppo/blob/main/dgppo/algo/dgppo.py
- 上游 Copyright (c) 2025 REALM，MIT 许可见 `utilities/nod_marl/DGPPO_LICENSE.txt`。
