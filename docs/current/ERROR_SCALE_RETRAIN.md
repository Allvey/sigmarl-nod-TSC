# 重训 reward7.38 误差尺度分支

若要从随机初始化开始，采用新的[250+70 轮误差尺度联合训练流程](ERROR_SCALE_FROM_SCRATCH.md)。
下文保留复用既有行为初始化的独立候选重训流程。

本流程复用当时的 `reward7.21` 行为初始化，重新执行 NOD 候选监督和控制微调，
不加载旧 `reward7.38` 权重。它不是整个系统从随机初始化训练，也不是未来证据 e
加 PPO β 分支。保留原流程有助于比较重训表现；相同种子不保证跨代码、依赖版本
或硬件逐位复现奖励。

## 固定初始化和种子

- 初始化前缀：`outputs/dependencies/dgppo_nod_responsibility_finetune/reward7.21`。
- 两阶段随机种子均为 **5080868027432654403**，与原训练记录完全相同。
- 新配置与原训练保存参数逐项核对；只改变输出目录和第二阶段的候选来源路径。
  旧奖励记录和模型名重置为新实验的初始值。
- 四车 `CPM_mixed`，场景概率 `[1,0,0]`，关闭观测噪声；4096 帧/批、128 步/段、
  32 个并行环境、minibatch 512、最多 60 epochs。

## 第一阶段：独立训练误差尺度 NOD，30 轮

```bash
python main_training.py --method nod --config configs/current/error_scale_repeat/01_candidate.json
```

Policy、消息网络、任务 Critic、Safety Value 和行为 NOD 保持冻结，独立 learner
NOD 使用采集轨迹监督学习。e 仍为几何计算；误差尺度根据几何证据相对未来缓解
标签的误差训练，A 通过意见监督间接学习。NOD 学习率 `1e-3`、4 个监督 epochs、
32 步序列、小批最多 8 个环境；不训练 PPO β。

完成后保存于：

```text
checkpoints/nod_error_scale_repeat/seed5080868027432654403/01_candidate/
```

其中 `final_policy.pth`、`final_critic.pth`、`final_safety_value.pth` 延续冻结的
行为模型，`final_nod.pth` 是本轮新训练候选。第二阶段需要同一目录的这四个文件。

## 第二阶段：冻结 NOD，控制微调，70 轮

第一条命令正常结束后执行：

```bash
python main_training.py --method nod --config configs/current/error_scale_repeat/02_control.json
```

自动读取第一阶段的 `final` 前缀。NOD 参数全程冻结，意见在线演化；前 20 轮只做
Safety Value warmup，后 50 轮更新 Actor、消息、任务 Critic 和 Safety Value。
Actor/消息初始学习率 `5e-5`，Actor 下限 `1e-5`，KL 阈值 0.01。
沿用 gated 安全优势，意见 α gain=1、死区 0.1。原训练最高回报约 7.3798，
出现在控制阶段第 65 轮；这是历史记录，不是本次预设的目标值。

结果保存于：

```text
checkpoints/nod_error_scale_repeat/seed5080868027432654403/02_control/
```

选择本轮最高训练奖励检查点评估，不预先假定其文件名前缀仍为 `reward7.38`。
本次两阶段额外采样共 409,600 个团队环境帧，不含复用的 reward7.21 及其上游训练。
完整系统训练成本不能只按本次 100 轮报告。

## 正式评估

完成两阶段后，评估本次重训模型与原版 SigmaRL、XP-MARL：

```bash
python utilities/evaluation_tase26.py \
  --model-paths checkpoints/nod_error_scale_repeat/seed5080868027432654403/02_control/ \
    checkpoints/sigmarl_original/config_v1_2_0/ \
    checkpoints/xp_marl_original/config_v1_2_0/ \
  --output-root outputs/benchmarks/nod_error_scale_repeat/seed5080868027432654403/
```

默认四场景、五项指标不变；仅评估交叉口可追加 `--scenarios intersection_2`。
训练种子沿用原记录；此评估命令默认测试种子仍为 101，以保持现有测试协议。

旧实验、新的两阶段从零方案及评估默认模型均保留。本配置用于一次独立重训，
若需再次运行，应更换输出目录并同步第二阶段初始化路径，避免混合不同运行结果。
