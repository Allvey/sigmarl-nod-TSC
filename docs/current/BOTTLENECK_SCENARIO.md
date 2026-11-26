# 双向隔离瓶颈场景

`bottleneck_1` 对应确认过的直线交通简图：每个方向独立完成
**2 车道 → 1 车道 → 2 车道**。中央隔离带贯穿全场景，两方向不共用瓶颈车道。
主通行车道保持直线，外侧车道通过平滑过渡汇入，驶出后重新分为两条车道。
这是依据简图构建的局部交通场景，不是某一实景道路的测绘复刻。

## 可视化

在项目根目录、`sigmarl-nod` 环境中执行：

```bash
python main_testing.py --scenario bottleneck_1 --no-speed-print
```

默认加载固定 NOD 模型目录
`checkpoints/nod_error_scale_repeat/seed5080868027432654403/02_control/` 中最高奖励模型。
20 辆车全部由 Actor 控制，环境种子 101、确定性动作、无观测噪声；保留车号、
行驶轨迹和 A1 的意见面板。窗口为 1280 × 640，车辆使用实际的 0.16 × 0.08 m 尺寸。

SigmaRL 使用同一资产和碰撞规则：

```bash
python main_testing.py --method sigmarl --scenario bottleneck_1 --no-speed-print
```

视频和日志保存到：

```text
<model>/visualization/bottleneck_1/envseed101_<checkpoint>/
  video.mp4
  agent_speeds.csv
  test_setup.json
```

具体前缀由模型目录决定，运行时 `[Test output]` 会打印完整路径。
相同模型、场景、种子重复执行会覆盖同名输出。可追加 `--env-seed 202`、
`--max-steps 600`；`--rule-fraction 0.5` 则启用已有混合规则控制入口。

## 几何与通行路径

| 参数 | 默认值 |
| --- | ---: |
| 单车道宽度 | 0.18 m |
| 中央隔离带宽度 | 0.08 m |
| 每方向入口直线段 | 2.0 m |
| 收窄 / 扩宽过渡段 | 各 1.05 m |
| 单车道瓶颈段 | 1.0 m |
| 每方向出口直线段 | 2.0 m |
| 世界范围 | 7.7 × 3.8 m |
| 参考中心线采样间距 | 不超过 0.035 m |

每方向有两条入口、两条汇入段、一条共享瓶颈段、两条分流段和两条出口，合计
18 个有向车道段、8 条完整路线。每个入口车道均可通向同方向的任一出口车道。
外侧中心线采用五次平滑过渡，两端的一阶、二阶导数均为零，避免连接处突然转向。
这些是预设导航参考路径，不额外加入自动变道规划器。

同向两车道在汇入前互为邻接车道；被隔离带分开的对向车道不设为邻接车道。
车辆只在各自入口直线段生成和重置，两端留出车身和合流余量，不在瓶颈内部随机插车。
两条车道的生成点沿纵向交错排列；当前提供 26 个唯一入口位置，任意两点间距均大于
现有的最小初始间距（车身对角线的 1.2 倍，约 0.215 m），避免随机停车式采样把
剩余空间切碎。重置仍检查实际占用情况，不会将新车硬插到已有车辆上。
生成器会拒绝超过入口位置数的默认车辆数；增加密度时应同步延长入口。

## 碰撞定义

使用既有 `road_edges` 模式：车身与**实际路面外缘或中央隔离带**接触判为道路碰撞。
白色虚线、方向箭头以及参考路径之间的重叠均不参与道路碰撞。
本场景路缘容差为 0，没有额外放宽路缘。车车碰撞仍采用已有车身判定。

参考路线及其名义左右边界仍参与导航观测、连续边界奖励和 Safety Value 输入。
因此“白色虚线不判碰撞”并不表示所有训练中的边界信号都被删除。
共享瓶颈会产生同向合流冲突；几何可通行不能保证现有策略零碰撞或零拥堵。

## 资产及重建

| 文件 | 用途 |
| --- | --- |
| `assets/maps/bottleneck_1.design.json` | 可编辑的尺寸、车辆数和视窗参数 |
| `scripts/build_bottleneck_assets.py` | 从同一解析几何生成全部资产，不识别图片中的线条 |
| `assets/maps/bottleneck_1.osm` | 执行用有向参考中心线 |
| `assets/maps/bottleneck_1.json` | 场景注册、路线、邻接、生成区间和几何检查摘要 |
| `assets/maps/bottleneck_1.boundaries.json` | 两个独立道路走廊的真实碰撞轮廓 |
| `assets/maps/bottleneck_1.render.json` | 运行时渲染的凸多边形图层 |
| `assets/maps/bottleneck_1.svg` | 可编辑的道路矢量简图 |
| `outputs/designs/bottleneck_1/asset_preview.png` | 带真实尺寸静态车辆的设计预览，不是仿真结果 |

修改 `.design.json` 后重建，然后重新启动测试进程：

```bash
python -m scripts.build_bottleneck_assets \
  --preview outputs/designs/bottleneck_1/asset_preview.png
```

生成器在写资产前检查路线连接、曲率、终端车身余量和实际车身的路面包含关系。
静态检查命令（不运行策略）：

```bash
python -m pytest tests/test_bottleneck_assets.py tests/test_main_testing_scenarios.py -q
```

四方法统一评估已将本场景和新环岛加入默认六场景列表，详见
[评估说明](TSC_BASELINE.md)。现有模型和训练流程不变，训练和动态测试由用户执行。
