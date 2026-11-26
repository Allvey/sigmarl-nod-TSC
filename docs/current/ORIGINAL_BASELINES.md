# 本项目内的原始 SigmaRL / XP-MARL

> 当前评估入口以[六场景四方法说明](TSC_BASELINE.md)为准：默认使用固定的 NOD、
> 原版重训 SigmaRL、ICRA25 发布 XP-MARL M1 和同级项目 TSC seed5。
> 两个新场景通过评估适配器共享本项目资产，原版训练源码仍保持字节一致。
> 下文的三方法、旧模型和输出目录说明保留为历史记录，不定义当前默认命令。

## 实现边界

原版代码位于 `baselines/original_sigmarl/upstream/`，来自官方 SigmaRL 标签
`1.2.0`，提交 `5fe715bdfba4ff3e33d901d69dfa220f1222c060`。
35 个官方文件按字节保留，SHA256 清单见同级 `provenance.json`；原版 MIT
许可证包含在源码目录中。运行不依赖本项目之外的基线工程或临时文件。

两种方法使用同一份官方实现，通过各自的配置选择行为：

| 项目 | SigmaRL | XP-MARL |
| --- | --- | --- |
| 训练配置来源 | 官方 1.2.0 根目录 `config.json` | 同一份 `config.json`，仅开启学习型优先级 |
| Actor | 参数共享的局部 Actor | 参数共享的局部 Actor，附邻车动作槽 |
| Critic | 集中式 | 集中式 |
| 动作分布 | 原始 NormalParamExtractor + TanhNormal | 同左 |
| 辅助优先级网络 | 无 | 原始 PriorityModule |
| 顺序动作传播 | 无 | 原始 prioritized_ap_policy |
| `is_using_prioritized_marl` | `false` | `true` |
| `prioritization_method` | `"random"`（优先级关闭，不生效） | `"marl"` |
| 训练观测噪声 | 关闭 | 关闭 |
| NOD / Safety Value / DGPPO | 不加载 | 不加载 |

两种配置均为 4 辆车、250 次迭代、4096 帧/批、60 epochs、minibatch 512、
初始学习率 2e-4，沿用官方固定种子逻辑。**训练配置以该标签根目录的
`config.json` 为唯一默认来源，不从已训练权重的元数据生成配置。** SigmaRL
只修改 `where_to_save`；XP-MARL 在同一份配置上额外修改上述两个优先级开关。
因此除保存目录和这两个开关外，两种配置的键和值完全相同。
总帧数 1,024,000、并行环境数 32 由原版 `Parameters` 根据采样参数自动计算，
不额外覆盖。上游 `config.json` 与其余官方源码保持原样。

此前直接保存在 `checkpoints/sigmarl_original/`、`checkpoints/xp_marl_original/`
根目录的训练结果属于旧的发布参数方案，保留为历史记录；特别是旧 SigmaRL
开启了训练噪声。修改当前配置不会改变这些旧权重或元数据，也不能将旧 SigmaRL
结果改称为本节关闭噪声的新训练结果。新训练使用下列独立子目录。

训练通过独立 Python 子进程导入官方 `utilities` / `scenarios`，以避免与
当前 NOD 工程的同名模块相互覆盖。训练时使用原版 ScenarioRoadTraffic，
不加载评估记录子类；观测、奖励、动力学、重置、PPO/GAE、优先级学习和
动作传播代码均来自原始文件。启动前检查完整性，变更官方文件会直接报错。

## 从零训练

在本项目根目录、现有 `sigmarl-nod` 环境中分别执行：

```bash
python main_training.py --method sigmarl
python main_training.py --method xp_marl
```

也可使用独立入口，效果相同：

```bash
python -m scripts.train_original_baseline --method sigmarl
python -m scripts.train_original_baseline --method xp_marl
```

权重分别保存至：

```text
checkpoints/sigmarl_original/config_v1_2_0/
checkpoints/xp_marl_original/config_v1_2_0/
```

两个目录含原版格式的 `reward*_data.json`、Actor/Critic 权重；XP-MARL
另有对应的 priority_policy / priority_critic。`baseline_provenance.json`
记录源码版本和启动参数。已有 `checkpoints/sigmarl/` 与 `checkpoints/xp-marl/`
中的历史权重不参与上述训练；若新目录已经有策略权重，启动器会拒绝覆盖，
可通过另一个配置的 `where_to_save` 指定新目录。

配置文件为：

- `configs/baselines/sigmarl_original.json`
- `configs/baselines/xp_marl_original.json`

`main_training.py` 不指定方法时仍按当前 NOD 配置运行；使用这两个配置文件
的路径也会自动选择相应的官方训练入口。自定义配置用 `--method` 显式选择。

## 与本方法统一评估

默认评估本方法和刚训练的两个原版基线（不包含 MAPPO）：

```bash
python utilities/evaluation_tase26.py
```

默认三方法顺序为本方法、SigmaRL、XP-MARL。本方法使用
`outputs/current/ablations/dgppo_nod_future_evidence_ppo_gate_sequence_control/`，
SigmaRL / XP-MARL 从各自训练配置的 `where_to_save` 读取上述新训练目录，
不会回退加载父目录的旧结果。每个目录选取最高奖励检查点；正式比较在两者
完成训练后执行。`--baseline-checkpoints retrained` 与默认命令等价。
默认覆盖 `CPM_entire`、`intersection_2`、`on_ramp_1`、`roundabout_1` 四个场景；
仅评估交叉口时使用 `--scenarios intersection_2`。

历史发布权重评估仍可单独执行（仅用于旧结果参考，不用于确定本节训练配置）：

```bash
python utilities/evaluation_tase26.py --baseline-checkpoints released
```

这时 SigmaRL 使用 `checkpoints/itsc24/M0 (our)/`，XP-MARL 使用
`checkpoints/icra25/M1 (XP-MARL)/`。两个命令都会自动选择原版运行代码，
不再通过修改后的 `utilities/mappo_cavs.py` 执行基线。只有本方法走 NOD
运行代码。`--model-paths`、`--legends` 可以指定其他模型组合；必要时用
`--model-backends` 显式指定每一项，选项为 `current`、`auto`、
`sigmarl_1_2_0`、`xpmarl_1_2_0`。

自动选择依据原版格式元数据；带 NOD 字段的历史重训练权重仍属于当前工程
运行代码，即使 NOD 开关为关闭也不会被当作原版。强制交给原版的修改工程
权重会被拒绝，避免错误地给它们冠以“官方原版”的名称。

## 测量与可比性

五项指标与现有图表保持一致：碰撞率（区分车车、道路和总计）、吞吐量、
世界坐标 jerk RMS、相对平均速度与动作变化 smoothness。指标统计与绘图
仍统一由 `utilities/evaluation_base.py` 完成。

评估时在原版环境上附加只读记录子类，读取车辆代际、物理位置/速度和重置
之前的出口/碰撞标记，以计算吞吐量和 jerk。它调用原版 reward/done/reset，
保持原返回值和控制行为，只在 info 中增加记录字段；这些字段不进入 Actor、
Critic 或优先级网络。VMAS 的 reward → observation → info → done 顺序
下每步都会刷新记录，避免沿用上一步快照。训练不使用该子类。

统一评估沿用当前试验条件：同场景车辆数、种子、32 并行环境、1200 个名义
步长、确定性动作、关闭测试观测噪声；`--stochastic-actions` 和
`--checkpoint-observation-noise` 可显式更换后两项。在模型构建结束、正式
rollout 之前重新设置环境种子，避免原版内部
`check_env_specs(seed=0)` 或网络初始化消耗随机数使请求的测试种子失效。
这一设置同时应用于本方法。原版算法实现与统一评估
协议应分别报告：例如本项目 CPM 全图为 20 辆车、intersection_2 为 8 辆车，
并非原版默认的 15/6 辆车。实际轨迹步数沿用现有入口的 `steps - 1`，
吞吐量分母采用实际记录时长。循环路线的吞吐量仍为 N/A。

原版 `PriorityModule` 与 `prioritized_ap_policy` 均保留，XP-MARL 会加载
对应的优先级策略权重并按顺序传播动作；不会退化成并行普通 Actor 测试。

结果目录：

```text
outputs/benchmarks/paper_metrics_original_baselines/
  released/three_methods/<场景>/steps_1200_envs_32_seed_101/
  retrained/config_v1_2_0/three_methods/<场景>/steps_1200_envs_32_seed_101/
```

每组包含 PNG/PDF、CSV、JSON 和日志。`metrics_runs.csv` 的
`runtime_backend` 列以及 `metrics_summary.json` 的 runtime 记录会区分
实际执行代码，并记录官方提交、测试噪声、车辆数与实际步数。
早期三方法/四方法结果目录不会被这些默认命令覆盖。

本接入针对正式 `evaluation_tase26.py`；历史 `run_sota_benchmark` 个体种子
诊断仍走旧诊断流程，不应用它替代上述原版运行代码的五指标比较。
