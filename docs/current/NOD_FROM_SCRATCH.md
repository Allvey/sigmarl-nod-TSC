# 未来证据与 PPO β：两阶段从零训练

本页对应学习未来证据 e 的分支。几何证据加学习误差尺度的 250+70 轮路线见
[ERROR_SCALE_FROM_SCRATCH.md](ERROR_SCALE_FROM_SCRATCH.md)。

本流程从随机初始化开始，不读取既有实验的 Policy、Critic、NOD 或 Safety Value。
第一阶段将原来的行为预训练、独立候选 NOD、固定采纳控制训练合并为同一轮联合学习；
第二阶段只加载本轮第一阶段的新权重，冻结语义预测器，再联合学习 PPO β 与 Actor。

这是新增的两阶段训练方案，不是原 reward7.16 训练过程的原样重放；效果需要以本轮
训练和评估结果确认。现有默认微调配置与旧的 `scratch_candidate` 配置仍依赖旧权重，
不能用于声称整个系统从零训练。

## 执行顺序

在项目根目录、`sigmarl-nod` 环境中按顺序执行，第一条完成后再执行第二条：

```bash
python main_training.py --method nod --config configs/current/from_scratch/01_joint.json
python main_training.py --method nod --config configs/current/from_scratch/02_ppo_beta.json
```

| 阶段 | 轮数 | 训练内容 |
| --- | ---: | --- |
| 01_joint | 250 | 所有网络随机初始化；同一批在线轨迹用于 Actor/任务 Critic 的 PPO、NOD 的未来证据监督及 Safety Value 风险学习；固定 β=A |
| 02_ppo_beta | 70 | 加载第一阶段 final 权重；冻结 NOD 历史编码、未来证据 e、A 和固定尺度；新增 PPO β，与 Actor/消息和任务 Critic 联合更新；Safety Value 独立拟合风险 |

第一阶段的关键设置：

- `training_init_checkpoint=null`、`is_load_model=false`、`is_continue_train=false`。
- `nod_training_mode=joint`、`nod_freeze_training=false`，每批更新 NOD。
- 从第一轮直接采用 `nod_evidence_mode=learned`、`nod_evidence_target=future`，
  固定方差为 1。e 使用未来缓解标签监督；A 没有单独标签，通过意见目标间接学习。
- `nod_evidence_gate_mode=fixed`，证据项使用 β=A；意见以 gain=1、死区 0.1 调节 α。
- `safety_training_mode=scratch`，Actor 从随机初始化直接学习。PPO 与监督 NOD 使用
  各自优化器，不是让 PPO 改写未来证据语义。在线预测只使用当前及历史信息；未来
  轨迹只用于采样结束后的监督标签。

第二阶段保持未来证据预测器冻结，采用既有 8 步有序回放学习 β。70 轮包含
20 轮 Safety Value warmup，以及其后 50 轮 Actor/消息/β 联合控制学习。
`nod_evidence_gate_only_iters=0`，不额外安排 β 单独训练阶段。β 初始采用第一阶段的
A 先验，之后由 PPO 决定采纳强度。warmup 不更新 Actor 或 β，NOD 全程冻结。

第一阶段的行为策略与 NOD 同时变化，不再具有四阶段方案中冻结行为模型的实验
隔离条件；这缩短了训练流程，但不保证与四阶段同样稳定或获得同样的性能。

## 参数与完整预算

两个阶段均为四车 `CPM_mixed`、概率 `[1,0,0]`，时间步长 0.05 秒、4096 帧/批、
128 步/段、32 并行环境、minibatch 512、最多 60 epochs，关闭观测噪声，种子为 0。

第一阶段 Actor/消息初始学习率为 `2e-4`，Actor 最低学习率 `1e-5`；NOD 监督学习率
为 `1e-3`，每批 4 个监督 epochs。第二阶段沿用 β 主线控制配置：
Actor/消息 `1e-5`、Actor 最低 `2e-6`、β `5e-5`、KL 阈值 0.01。

合计 320 批、1,310,720 个团队环境帧，包含 Safety warmup。相比此前四阶段的
420 批减少 100 批；仍高于两个基线各自的 250 批、1,024,000 帧，论文应报告完整预算。

## 权重与评估

新结果使用独立目录，保留四阶段方案和既有模型的全部输出：

```text
checkpoints/nod_from_scratch/two_stage/seed0/01_joint/
checkpoints/nod_from_scratch/two_stage/seed0/02_ppo_beta/
```

第二阶段自动从 `01_joint/final` 初始化，无需人工填写最高奖励前缀。
每次重新从零训练应同步修改两个配置的输出目录和第二阶段初始化前缀，避免与本轮
结果混合。旧四阶段配置已归档至 `configs/archive/nod_from_scratch_four_stage/`，
不属于当前执行流程。

现有评估脚本的默认本方法仍指向已评估的旧论文候选，暂不切换。完成本轮两个阶段
后，显式指定最终模型目录评估三个方法：

```bash
python utilities/evaluation_tase26.py \
  --model-paths checkpoints/nod_from_scratch/two_stage/seed0/02_ppo_beta/ \
    checkpoints/sigmarl_original/config_v1_2_0/ \
    checkpoints/xp_marl_original/config_v1_2_0/ \
  --output-root outputs/benchmarks/nod_from_scratch/two_stage/seed0/
```

评估选择各目录最高奖励检查点，默认四场景、五项指标保持不变；
仅评估交叉口可追加 `--scenarios intersection_2`。
