# TSC 外部项目评估

TSC 保留在同级 `sigmarl-traffic` 项目中，使用其原有 Policy、拓扑网络、动作预测器、
观测处理和 rollout 实现。当前项目通过独立子进程采集轨迹，再使用同一个
`Evaluation` 计算指标与绘图。未向本项目迁移 TSC 算法或权重，也不改写原项目源码。

## 当前四个模型

| 方法 | 模型目录 |
| --- | --- |
| NOD-DGPPO | `checkpoints/nod_error_scale_repeat/seed5080868027432654403/02_control/` |
| SigmaRL | `checkpoints/sigmarl_original/config_v1_2_0/` |
| XP-MARL | `checkpoints/icra25/M1 (XP-MARL)/` |
| TSC | `../sigmarl-traffic/outputs/TSC/seed5/` |

默认仍分别选择各目录最高训练奖励检查点。接入时 TSC 对应 `reward7.02`；需要同前缀
的 `_data.json`、`_policy.pth`、`_topology.pth` 和 `_action_predictor.pth`。
不将文件夹名 seed5 解释为保存参数中的真实随机种子。

## 执行

在当前 `sigmarl-nod-TSC` 项目根目录运行：

```bash
python utilities/evaluation_tase26.py
```

默认六场景：`CPM_entire`、`intersection_2`、`on_ramp_1`、`roundabout_1`、
`roundabout_realistic`、`bottleneck_1`。每个方法 32 个并行环境、1200 名义步、测试种子 101，确定性动作、
关闭观测噪声。沿用现有 rollout 协议，实际记录 1199 个转移，吞吐量分母使用实际
记录时长。不读取或写入旧轨迹缓存，不启动训练。

只测试交叉口：

```bash
python utilities/evaluation_tase26.py --scenarios intersection_2
```

结果都保存在本项目：

```text
outputs/benchmarks/paper_metrics_with_tsc/seed5/mixed/four_methods/six_scenarios/
  <scenario>/steps_1200_envs_32_seed_101/
```

每场景输出 PNG/PDF、`metrics_runs.csv`、`metrics_summary.json` 和日志，包含碰撞率
（总计、车车和道路边界）、吞吐量、世界坐标速度 jerk RMS、平均速度、动作变化
smoothness。仍使用现有分布图，四方法图宽适当增大。环形路线没有可比的驶出吞吐量，
沿用现有协议记为不可用。

只运行原先固定的三个方法：

```bash
python utilities/evaluation_tase26.py --without-tsc
```

该选项保留三方法输出目录
`outputs/benchmarks/paper_metrics_original_baselines/mixed/three_methods/`。
`--baseline-checkpoints retrained` 可显式改用两份重训基线，`released` 可改用两份发布
基线；默认 `mixed` 对应上表中的固定三组权重。`--model-paths` 显式指定目录时，不
自动追加 TSC；需自行列出完整目录与 `--legends`。

其他 TSC 路径可用：

```bash
python utilities/evaluation_tase26.py \
  --tsc-project ../sigmarl-traffic \
  --tsc-model-path outputs/TSC/seed5/
```

## 隔离与测量口径

### 新环岛和瓶颈的共同环境资产

两个新增地图在原版仓库中不存在。评估 worker 通过
`baselines/scenario_assets.py` 在子进程内注册本项目的同一份 OSM/路线/邻接，
对齐地图坐标原点、世界范围、入口生成点和实际路面碰撞轮廓；无需复制文件到原版
仓库或修改 TSC 项目的源码。SigmaRL、XP-MARL 的训练代码和上游文件哈希不变，
TSC 的策略、拓扑筛选和动作预测也继续使用原项目。

这两个场景采用与本方法相同的 `road_edges` 规则：白色虚线不判碰撞，实际路缘、
隔离带和岛体参与判定；环岛边缘容差 0.02 m，瓶颈为 0。
碰撞规则在奖励和重置发生前生效，不是事后仅重新统计指标。
仅从资产规定的入口区间生成车辆，并采用同一随机选择、最小初始间距和 2000 次
重试上限。保持各方法既有观测结构、归一化默认值和动作分布，不增加 NOD 功能给基线。

| 场景 | 每个方法车辆数 | 道路碰撞规则 |
| --- | ---: | --- |
| CPM_entire | 20 | 原有参考车道边界 |
| intersection_2 | 8 | 原有参考车道边界 |
| on_ramp_1 | 8 | 原有参考车道边界 |
| roundabout_1 | 8 | 原有参考车道边界 |
| roundabout_realistic | 16 | 共同资产的实际路缘和岛体 |
| bottleneck_1 | 20 | 共同资产的实际路缘和中央隔离带 |

车辆数取 `SCENARIOS` 的当前定义；修改资产需重建并重启评估。新结果使用
`six_scenarios/` 子目录，与此前四场景结果分开；不删除旧结果，不复用轨迹缓存。
`metrics_summary.json` 记录共同资产、适配代码和碰撞代码的 SHA256，便于核对
四方法是否使用同一场景版本。原有四个场景不安装新地图适配器。

仅测试两个新场景：

```bash
python utilities/evaluation_tase26.py --scenarios roundabout_realistic bottleneck_1
```

### 原版运行时

- `baselines/tsc/runtime.py` 将子进程 cwd 和 PYTHONPATH 设置到 TSC 原项目；worker
  验证实际导入的 trainer、helper、scenario 都属于该项目，避免同名模块混用。
- 测量钩子与现有原版 SigmaRL/XP-MARL 评估相同，仅记录世界速度、车辆重置代次和
  重置前的驶出/碰撞事件，不改变动作、奖励、原观测字段或重置决策。
- seed5 检查点策略观察 1 个邻居、拓扑网络观察 3 个邻居；保持训练时结构，不强改
  为其他方法的邻居数。任务场景、车辆数、步长、测试种子及动作采样方式统一。
- 加载后检查拓扑网络和动作预测器与指定检查点完全一致；原加载器即使仅打印警告，
  桥接程序也会在权重缺失或不一致时停止，避免将不完整网络当作 TSC 测评。
- 导出的 runtime 元数据包含原项目位置、源文件/地图和三份模型权重的 SHA256，
  以及测试协议、实际邻居数。源码/权重仍属于原项目，修改后会产生不同记录。
- 接入时核对 `cpm.xml`、`intersection_2.osm`、`on_ramp_1.osm`、`roundabout_1.osm`
  及 `kinematic_bicycle.py` 与本项目逐字节相同；这是基础地图与运动学一致性检查，
  不代表两个项目的全部场景实现相同。
- 可选推理计时仅统计 Policy forward，与已有接口一致，不代表含拓扑筛选、环境
  步进在内的完整决策耗时。

## 已做检查

测试覆盖子进程导入来源、原三模型保持、六场景及测试参数传递、禁用缓存、拓扑/
预测器完整加载检查，以及对统一指标管线的分发。另检查新增地图在两种隔离运行时
中的坐标、生成点、实际车身道路包含关系与本项目一致；用内存状态验证相同种子的
入口重置一致，以及实际路缘判定先于奖励。检查不运行真实环境仿真，完整
六场景四方法评估由用户执行后确认。
