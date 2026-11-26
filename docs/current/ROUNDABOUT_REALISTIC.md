# 弯曲四臂双车道环岛

`roundabout_realistic` 按原始航拍图的四臂方向、弯曲接入和中央岛布局重新规划。
当前版本优先满足车辆通行尺寸：外边界、同向车道分隔线、参考路线从同一套
等距曲线生成，不再把生成图片中的蓝线、黄线直接当作可执行道路几何。
这是依据原图规整的仿真道路，不是实测地图的逐像素复刻。

## 可视化

在项目根目录、`sigmarl-nod` 环境运行：

```bash
python main_testing.py --scenario roundabout_realistic --no-speed-print
```

默认使用现有固定模型目录：

```text
checkpoints/nod_error_scale_repeat/seed5080868027432654403/02_control/
```

全部由 Actor 控制，16 辆车、环境种子 101、确定性动作、关闭观测噪声。
窗口为 **800 × 800**；世界范围约 **5.06 × 5.15 m**。车辆保持原有
**0.16 × 0.08 m**，没有通过夸大渲染车身来改变显示比例。
保留车辆编号、轨迹和 A1 的 z/alpha 面板，长文字换行。

输出沿用：

```text
<model>/visualization/roundabout_realistic/envseed101_<checkpoint>/
  video.mp4
  agent_speeds.csv
  test_setup.json
```

重复执行同一模型、种子会覆盖该目录下的上一轮结果。实际输出路径由运行时
打印的 `[Test output]` 给出。可追加 `--env-seed 202`、`--max-steps 600`。

## 本次车道重新规划

- 单车道宽度 **0.18 m**，同向两车道总宽 **0.36 m**；标称车宽 0.08 m，
  在直线车道居中时每侧有 0.05 m 的几何余量。
- 环道为两条逆时针车道。先定义一条椭圆基准曲线，再沿法线分别偏移
  ±0.09 m 得到车道中心，偏移 ±0.18 m 得到路面边缘。没有把两条独立椭圆
  当作等宽车道，因此弯道内侧不会因独立拟合而突然收窄。
- 每个出入口采用有共同切向的三次曲线，同向两条车道由同一条曲线平行偏移。
  右上方的进出道路也按这一规则重建，外缘与车道中心一起计算。
- 道路汇接处取路面走廊的并集，并圆滑凹角。入口与出口分隔线在汇接区前
  停止，避免它们与环道标线叠加成无法容纳车辆的狭长“车道”。
- 横道及箭头相对基准曲线定位；箭头指示各自进出方向。同向车道之间使用
  白色虚线，移除原先缺少一致几何依据的黄线。
- 参考路线终端外另延伸 0.12 m 路面，避免车心到达端点时车身超出路面截面。

### 碰撞边界：只检查实际路缘，白色虚线不算碰撞

本场景使用 `road_collision_mode: "road_edges"`。道路碰撞以路面并集的外轮廓
及内部孔洞边缘为准，不再检查车辆自己那条参考路线的左右边界。跨越白色
虚线或在未画线的汇接区偏离原车道，不会仅因此增加 Road contacts 或触发
车道碰撞惩罚、重置；车辆越过实际外缘、中央岛铺装边缘或分流岛边缘仍会判碰撞。

`lane_collision_margin: 0.02` 在这个模式下表示实际路缘的米制容差：离线将
可通行路面向外膨胀约 0.02 m，按扩展后的轮廓与完整车身做包含和相交判定。
内部岛体孔洞保留，仅边缘同样给予容差。白色标线和显示车道宽度不变。

碰撞轮廓存储在 `.boundaries.json` 的 `collision_regions`，并记录
`collision_margin_m`。运行时不依赖 OpenCV；通过多边形包含、车身边缘相交
和岛体被车身包围检查检测碰撞，避免只查四角漏掉窄岛或凹角。

车车碰撞保持原判定。参考路线仍用于导航、车道观测、接近边界的连续惩罚和
Safety Value 的边界距离输入；因此允许跨虚线不代表现有策略一定会主动变道。
其他场景保持 `reference_lane` 模式和零容差。模式及容差写入可视化输出的
`test_setup.json`，无需修改模型权重。调节 `.trace.json` 后须重新生成资产。

仍为 32 个有向车道段、24 条完整路线：每个入口车道可去其他三个出口，
不包含原路掉头和额外的显式变道路线。车辆只从入口段生成。双车道环岛的汇入、驶出仍
存在交通冲突；这里检查的是尺寸可通行，策略的实际避让由运行测试确认。

20 辆车的入口生成区间覆盖入口段索引 `[4, len(entry)-6)`，两端分别留出
车身及汇入余量；不再把所有车辆挤在入口前约三分之一区间。车心最小间距仍为
车辆对角线的 1.2 倍（约 0.215 m）。带显式生成区间的地图最多尝试 2000 次，
找不到空位会报出车辆编号和密度信息，避免无限重试。

## SigmaRL 可视化对比

```bash
python main_testing.py --method sigmarl --scenario roundabout_realistic --no-speed-print
```

默认从 `checkpoints/sigmarl_original/config_v1_2_0/` 选择最高奖励的 Actor，
关闭 NOD、Safety 扩展及意见 HUD，并使用原版的动作分布尺度映射。
无需重新训练；原版共享 Actor 可以用于当前场景的 16 辆车。
这里复用原版训练权重，环境使用本项目的统一可视化入口，因此环岛几何和上述
实际路缘碰撞规则与 NOD 测试相同，不是单独运行原版仓库的环境。

视频与速度日志写入该模型目录下的
`visualization/roundabout_realistic/envseed101_<checkpoint>/`，分别为
`video.mp4` 和 `agent_speeds.csv`；`test_setup.json` 记录模型、方法、种子及边界模式。
默认使用确定性动作、无观测噪声、无规则车辆。省略 `--method sigmarl` 仍测试 NOD。

## 资产与生成输入

| 文件 | 用途 |
| --- | --- |
| `assets/maps/roundabout_realistic.aerial.png` | 用户提供的原始航拍参考，记录来源哈希；不作运行时背景 |
| `assets/maps/roundabout_realistic.trace.json` | 可编辑道路设计：尺度、车道宽度、环道基准、门户和曲线控制参数 |
| `scripts/roundabout_width_design.py` | 等距车道、路面并集、标线和构建前几何检查 |
| `scripts/roundabout_vectorization.py` | 将设计掩码转为带孔轮廓及凸多边形；不再从生成图片识别车道 |
| `scripts/build_roundabout_assets.py` | 同步导出 OSM、各 JSON、SVG 和静态预览 |
| `assets/maps/roundabout_realistic.osm` | 执行用参考中心线 |
| `assets/maps/roundabout_realistic.json` | 路线、邻接、入口生成区间、视窗、离线几何检查摘要 |
| `assets/maps/roundabout_realistic.boundaries.json` | 规整后的道路轮廓、坐标变换、来源哈希和几何来源说明 |
| `assets/maps/roundabout_realistic.render.json` | 运行时凸多边形图层，与道路设计共源 |
| `assets/maps/roundabout_realistic.svg` | 可编辑分层交通简图 |

旧的 `.reference.png` 是此前生成的简图，保留供追溯，**不再参与构建**。

## 离线重建与检查

```bash
python -m scripts.build_roundabout_assets \
  --preview outputs/designs/roundabout_realistic/replanned_asset_preview.png \
  --route-preview outputs/designs/roundabout_realistic/replanned_routes_preview.png \
  --vehicle-preview outputs/designs/roundabout_realistic/replanned_vehicle_preview.png

python -m pytest tests/test_roundabout_assets.py tests/test_main_testing_scenarios.py -q
python -m pytest tests/test_road_collision_margin.py -q
```

构建前检查 24 条完整路线的有限坐标、连接、转弯曲率以及按车辆实际尺寸采样的
车身路面余量。检查失败会在覆盖可执行资产之前报错。测试还覆盖：

- 弯道平行偏移没有折叠，邻道间距足够；相同进度的两辆车可并排放置而不重叠；
- OSM 解析后的参考走廊宽度确为 0.18 m；车身在自身走廊内有剩余空间；
- 路面矢量孔洞、渲染范围、资产重复构建一致性；
- 环境初始化/重置、入口生成、观测维度和旧场景兼容性。

构建摘要位于 `.json` 的 `geometry_checks`。它描述参考几何，不是实际行驶
无碰撞保证。仿真的道路碰撞采用实际路面轮廓与岛体孔洞；横道和车道虚线
不参与碰撞。保留既有动力学、观测和控制模型，原有四个
正式评估的原有四场景不变，本场景与瓶颈一并加入默认六场景四方法评估列表，
详见[评估说明](TSC_BASELINE.md)。离线检查不执行策略行驶。
