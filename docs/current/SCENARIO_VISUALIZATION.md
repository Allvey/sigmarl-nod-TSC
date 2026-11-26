# 本方法新增场景可视化

新增直线双向隔离瓶颈：每方向独立 `2 → 1 → 2`、默认 20 辆车，
详见[瓶颈场景资产说明](BOTTLENECK_SCENARIO.md)：

```bash
python main_testing.py --scenario bottleneck_1 --no-speed-print
```

航拍参考的弯曲四臂双车道环岛已接入，详见
[环岛资产说明](ROUNDABOUT_REALISTIC.md)：

```bash
python main_testing.py --scenario roundabout_realistic --no-speed-print
```

在当前项目根目录、`sigmarl-nod` 环境中执行：

```bash
python main_testing.py \
  --scenarios intersection_3 intersection_4 on_ramp_2_multilane roundabout_2 \
  --no-speed-print
```

依次打开四个独立可视化测试进程。默认使用固定模型目录：

```text
checkpoints/nod_error_scale_repeat/seed5080868027432654403/02_control/
```

自动选择最高奖励 Policy 对应的参数文件，并由既有加载器加载匹配权重。默认全部车辆
由 Actor 控制（`--rule-fraction 0`），测试种子 101、关闭观测噪声、确定性动作、
1200 名义步（1199 个转移）。不训练，不读取旧轨迹缓存。保留车辆编号、轨迹以及
车辆 1 对邻居的 z/alpha 显示。

| 场景 | 默认车辆数 | 重点观察 |
| --- | ---: | --- |
| intersection_3 | 8 | 新路口结构中的通过、避让及边界接触 |
| intersection_4 | 8 | 四方向与多通行路径相互冲突时的协调 |
| on_ramp_2_multilane | 10 | 多车道汇入、连续交互及排队 |
| roundabout_2 | 10 | 多入口/驶出路径上的持续让行及拥堵 |
| roundabout_realistic | 16 | 弯曲分隔出入口、双车道环行及连续汇入/驶出 |
| bottleneck_1 | 20 | 同向双车道合流、单车道瓶颈排队和驶出分流；双向物理隔离 |

只测试一个场景：

```bash
python main_testing.py --scenario intersection_4 --no-speed-print
```

可以调整 `--env-seed 202`、`--max-steps 600`；`--no-realtime` 关闭实时显示节奏，
`--no-video` 禁用视频，`--stochastic-actions` 改为随机采样动作。

## 输出

全部 Actor 模式按场景、环境种子、检查点分别保存，例如：

```text
checkpoints/nod_error_scale_repeat/seed5080868027432654403/02_control/
  visualization/
    intersection_3/envseed101_reward7.38/video.mp4
    intersection_4/envseed101_reward7.38/video.mp4
    on_ramp_2_multilane/envseed101_reward7.38/video.mp4
    roundabout_2/envseed101_reward7.38/video.mp4
```

每个目录同时保存 `agent_speeds.csv` 和 `test_setup.json`。检查点前缀取决于实际
目录中最高奖励模型；随机动作模式目录附加 `_stochastic`。相同场景、种子及模型
重复运行时覆盖该目录的同名输出。

## 规则车辆模式

之前的混合车辆可视化可以显式启用：

```bash
python main_testing.py --scenario intersection_2 --rule-fraction 0.5 --stochastic-actions
```

规则控制输出仍使用原 `rule_vehicle_visualization/` 命名方式。多种子规则评估脚本已
显式传入 0.5 比例与随机动作，保留其原测试条件。

本页描述可视化入口。正式四方法评估已接入新环岛和瓶颈的共同地图、入口生成区和
实际路缘碰撞规则：运行 `python utilities/evaluation_tase26.py`，详见
[六场景评估说明](TSC_BASELINE.md)。其他新增可视化地图不在默认评估列表中。
