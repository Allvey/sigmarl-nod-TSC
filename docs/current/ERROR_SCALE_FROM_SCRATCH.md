# 几何证据与误差尺度：两阶段从零训练

本流程直接从随机初始化训练完整系统，将误差尺度 NOD 训练合并进最初的 250 轮，
随后冻结 NOD，进行 70 轮控制微调。无需 reward7.21 初始化，也无需单独的 30 轮
候选 NOD 训练。随机种子固定为 **5080868027432654403**，沿用历史误差尺度实验
的种子；这是一条新训练路线，性能需由本次结果确认。

## 执行命令

在项目根目录、`sigmarl-nod` 环境中，依次执行；第一条正常结束后再执行第二条：

```bash
python main_training.py --method nod --config configs/current/error_scale_scratch/01_joint.json
python main_training.py --method nod --config configs/current/error_scale_scratch/02_control.json
```

| 阶段 | 轮数 | 更新内容 | 初始化 |
| --- | ---: | --- | --- |
| 01_joint | 250 | Actor、消息网络、任务 Critic、Safety Value 与监督 NOD | 所有网络随机初始化 |
| 02_control | 70 | NOD 全程冻结；前 20 轮 Safety Value warmup，后 50 轮更新控制器与 Safety Value | 本次第一阶段的 final 权重 |

第一阶段每批先按已有流程更新 PPO，再使用采集轨迹更新监督 NOD；它们使用各自
优化器，不是把 NOD 的监督目标替换成 PPO。该阶段 Actor 直接开始学习，不采用
第二阶段的 Actor 冻结 warmup。第二阶段冻结 NOD 参数，但意见状态仍随观测演化，
Actor 的 NOD 消息网络继续参与控制学习。

## 本分支学习什么

- 证据 e 由几何与已观测运动计算，不使用学习型未来证据预测头。
- 保持 b=0、s=1；A 通过意见输出与未来缓解标签之间的监督损失间接学习。
- 学习方差作为几何证据相对未来缓解目标的误差尺度，而非相对已吸收证据的意见
  的 posterior 残差尺度。令未来标签为 y=q，训练误差为
  `r = stopgrad(e_geometry) - stopgrad(2*y-1)`，采用高斯负对数似然
  `0.5 * (r²/variance + log(variance) + log(2*pi))`。
- 方差损失只使用有效、完整的未来窗口；未来轨迹仅用于采样后的训练标签。
  在线动作只使用当前与历史观测。意见监督和方差监督权重均为 1。
- `nod_variance_supervision=evidence_error` 从第一轮就启用；不额外训练 PPO β。
  保留既有意见动力学和 gated 安全优势，意见调节 α 的 gain=1、死区=0.1。

该误差尺度描述训练分布下证据的预测误差，不等同于经过覆盖率校准的风险下界。

## 训练参数与输出

四车 `CPM_mixed`、场景概率 `[1,0,0]`，关闭观测噪声。每轮 4096 团队环境帧，
128 步/段、32 并行环境、minibatch 512、最多 60 PPO epochs。
两阶段合计 **1,310,720 团队环境帧**，包含第二阶段 warmup。

第一阶段 Actor/消息学习率 `2e-4`，Actor 下限 `1e-5`；NOD 每批更新，学习率
`1e-3`、4 个监督 epochs、32 步序列、小批最多 8 个环境。第二阶段 Actor/消息
学习率 `5e-5`，Actor 下限 `1e-5`、KL 阈值 0.01。

```text
checkpoints/nod_error_scale_scratch/seed5080868027432654403/
├── 01_joint/
└── 02_control/
```

第二阶段自动读取 `01_joint/final` 前缀的 Policy、任务 Critic、NOD 与 Safety Value。
请保留第一阶段目录中的完整检查点及参数文件。最终模型在 `02_control/`，其中
保留最高训练奖励中间检查点和最终检查点；文件名取决于实际奖励。
再次独立运行时应更换输出目录，并同步第二阶段初始化路径。

## 训练后评估

```bash
python utilities/evaluation_tase26.py \
  --model-paths checkpoints/nod_error_scale_scratch/seed5080868027432654403/02_control/ \
    checkpoints/sigmarl_original/config_v1_2_0/ \
    checkpoints/xp_marl_original/config_v1_2_0/ \
  --output-root outputs/benchmarks/nod_error_scale_scratch/seed5080868027432654403/
```

默认测试种子 101，评估四场景的碰撞率（含车车与道路边界分类）、吞吐量、jerk、
平均速度和 smoothness。只评估交叉口可追加 `--scenarios intersection_2`。

此路线与[复用 reward7.21 的历史重训](ERROR_SCALE_RETRAIN.md)、
[学习未来证据并训练 PPO β 的从零路线](NOD_FROM_SCRATCH.md)各用独立配置和输出目录。
