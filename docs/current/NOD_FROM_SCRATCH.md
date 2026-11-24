# 本方法完整从零训练

本流程从随机初始化开始，不读取既有实验的 Policy、Critic、NOD 或 Safety Value。
后续阶段只使用本轮前一阶段新产生的 `final_*` 权重，保留未来证据监督、冻结语义
预测器与 PPO β 控制学习的分工。它是一套新增的从零复现配置，不是既有 reward7.16
训练过程的原样重放，也没有承诺获得同样的奖励。

现有主线微调配置依赖旧检查点；名字含 `scratch_candidate` 的旧配置仅随机初始化
候选 NOD，仍依赖预训练行为模型，不能用于声称整个系统从零训练。

## 执行顺序

在项目根目录、`sigmarl-nod` 环境中按顺序执行，每条完成后再执行下一条：

```bash
python main_training.py --method nod --config configs/current/from_scratch/01_bootstrap.json
python main_training.py --method nod --config configs/current/from_scratch/02_future_candidate.json
python main_training.py --method nod --config configs/current/from_scratch/03_fixed_control.json
python main_training.py --method nod --config configs/current/from_scratch/04_ppo_beta.json
```

| 阶段 | 轮数 | 初始化与学习内容 |
| --- | ---: | --- |
| 01_bootstrap | 250 | 所有模型随机初始化；以几何责任证据训练行为 NOD，训练 Actor、消息、任务 Critic 和 Safety Value；意见不调节 α，β 为固定模式 |
| 02_future_candidate | 30 | 冻结本轮行为模型；独立候选 NOD 随机初始化，监督学习未来证据和意见，不改变采集策略 |
| 03_fixed_control | 70 | 加载本轮候选 NOD 并冻结；重新 warmup Safety Value，训练固定采纳控制器，启用意见 α |
| 04_ppo_beta | 70 | 加载本轮固定采纳控制器；冻结未来预测器，新增 PPO β，沿用短序列回放和分阶段控制学习 |

训练场景均为四车 `CPM_mixed`、概率 `[1,0,0]`，时间步长 0.05 秒、4096 帧/批、
128 步/段、32 并行环境、minibatch 512、最多 60 epochs，关闭观测噪声，种子为 0。
首阶段采用原始 PPO 学习率 `2e-4 → 1e-5`；后续使用现有未来证据候选及控制主线
各自的学习率、warmup 和 KL 配置。首阶段 `safety_training_mode=scratch`，不冻结
随机 Actor 等待微调 warmup；后续微调阶段保持原有 warmup 规则。

四阶段共 420 批，即 1,720,320 个团队环境帧，其中包含候选采集和 Safety warmup。
这与两个基线各 250 批、1,024,000 帧的预算不同；论文需报告完整成本，不能只将最后
70 轮视为本方法的全部训练成本。

## 权重与评估

所有新结果位于：

```text
checkpoints/nod_from_scratch/seed0/01_bootstrap/
checkpoints/nod_from_scratch/seed0/02_future_candidate/
checkpoints/nod_from_scratch/seed0/03_fixed_control/
checkpoints/nod_from_scratch/seed0/04_ppo_beta/
```

配置均不续训、不加载旧模型，首阶段 `training_init_checkpoint=null`。后续阶段从
前一阶段的 `final` 前缀初始化，避免人工填写最高奖励前缀；三个后续阶段属于本轮
流水线内的分阶段训练。正式评估仍按最终目录中最高奖励检查点选择。
已有旧权重不移动、不覆盖。若要重复整条流程，应先为四个配置同步指定新的输出
目录与初始化前缀，以免与本轮结果混合。

现有评估脚本的默认本方法仍指向旧论文候选。要评估本次从零训练结果，完成全部
阶段后显式指定三个模型目录：

```bash
python utilities/evaluation_tase26.py \
  --model-paths checkpoints/nod_from_scratch/seed0/04_ppo_beta/ \
    checkpoints/sigmarl_original/config_v1_2_0/ \
    checkpoints/xp_marl_original/config_v1_2_0/ \
  --output-root outputs/benchmarks/nod_from_scratch/seed0/
```

默认四个测试场景和五项指标不变；只测交叉口可追加 `--scenarios intersection_2`。
