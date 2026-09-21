# 测试用规则车辆集中协调

`main_testing.py` 使用 `coordinated_rules_v4`。`rule_fraction`、类型比例和巡航速度仍在该文件开头设置。规则车之间可以共享位置、路线和动作计划；这些信息不添加到 Actor 或 NOD 的观测中。训练不使用这个控制器。

## 速度设置

`main_testing.py` 中的 `rule_lateral_accel_limit = 1.8` 控制弯道限速，路径跟踪器和预约预测器共用这个值。旧值是 `0.6`。限速大致按 `sqrt(参数 / 曲率)` 计算，之后仍受巡航速度、路径误差和交通预约约束。该参数作用于速度命令；VMAS 中的实际速度还受阻尼影响，不能将它当成实际横向加速度的严格保证。

直道主要受 `rule_cruise_speed = 1.0` 控制。只提高巡航速度，无法突破弯道限速。需要恢复旧弯道速度时，把 `rule_lateral_accel_limit` 设回 `0.6` 即可，不需要再修改预测器内的常数。

无交通干扰、实际 VMAS 动力学下，当前四条弯道平均速度从约 0.37–0.42 m/s 提高到 0.61–0.68 m/s，整条路线通过时间缩短约 19%–28%。六条路线的车身均未接触左右道路边界。对照记录见 `outputs/rule_coordination_checks/free_route_speed_comparison.json`。

## 控制过程

1. 原有路径跟踪器生成期望动作。规则车对 Actor 的反应仍由 yielding / moderate / non_yielding 类型决定。
2. 测试端从道路参考路线的实际交点识别冲突区域，为每个区域分配持续到车辆驶离的通行权。入口按等待时间排队，后车不能抢占前车需要的通行权，出口被规则车占据时暂停新放行。车辆只申请当前最先需要通过的区域，避免上游未通过时提前占用下游通行权。
3. `RuleCoordinator` 为规则车预约 1.2 秒轨迹，包含末端停车段。用自行车动力学和 VMAS 阻尼预测实际运动，检查整个积分区间内车身扫过的区域。汇流后或同车道行驶时前车先规划，后车根据前车轨迹减速跟随。尚未获得本轮规划结果的车辆以可实现的制动轨迹参与检查，后车的旧追赶轨迹不会反向阻塞前车。上一帧已批准的轨迹继续执行前，也要满足当前入口等待线的约束。
4. 三种类型均服从集中避碰。`non_yielding` 不再无视其他规则车。
5. 规则车在冲突区域外静止重生，旧一代的预约、等待记录失效。单车重生传入的 Tensor 车辆编号会先转为整数，保证规则车辆字典判断有效。超过插入尝试上限会明确报错，不会无限循环或隐瞒碰撞。

几何通行权目前用于 `intersection_*` 地图。其他地图仍使用联合轨迹检查，但本次长时验收仅覆盖 `intersection_2`。预测器要求单子步、当前项目的无车身碰撞力配置；改变仿真动力学后需要重新校验。

## 可视化与诊断

- `reserved-go`：按批准轨迹前进。
- `reservation-wait`：被集中协调限制速度或要求等待。
- `keep-reservation`：新提议未获批准，继续原预约轨迹。
- `rule_contact`：仅统计规则车与规则车的接触；视频左上角的总碰撞数还包含 Actor。
- `reservation_infeasible`：最终联合轨迹存在冲突，需要排查，不是被忽略的正常状态。
- `reservation_wait`：调度使用的等待积分，区别于实际连续停车时间。
- `curvature` / `cruise_limit`：前方路线曲率、施加交通约束前的本车速度上限，用于区分弯道限速和排队等待。

视频和日志保存到原检查点下的 `rule_vehicle_visualization/`，运行目录带有 `coordinated_rules_v4` 后缀。脚本会打印完整输出目录。

## 回归命令

在 sigmarl-nod 环境、项目根目录执行：

```bash
python -m pytest tests/test_testing_rule_coordination.py tests/test_testing_rule_policy.py tests/test_nod_visualization.py -q
python -m scripts.validate_rule_coordination --fraction 1 --seeds 123 456 789 --steps 1200
python -m scripts.validate_rule_coordination --fraction 0.5 --seeds 123 456 789 --steps 1200
# 同一控制器使用旧弯道参数做对照（输出目录会区分参数值）
python -m scripts.validate_rule_coordination --fraction 0.5 --seeds 123 --steps 1200 --lateral-accel-limit 0.6
```

结果位于 `outputs/rule_coordination_checks/`。验证脚本在规则车接触、最终预约不可行或持续低速超过阈值时返回失败。默认持续低速报警阈值是 15 秒、速度阈值是 0.03 m/s；它用于发现堵塞，不能作为数学上的无死锁证明。

当前目标是规则车辆之间的安全通行。Actor 仍可能造成碰撞或堵塞，有限随机种子测试也不等于任意密度、任意地图下的无碰撞和无死锁保证。

本轮最终版本的调度与单车重生修复尚未完成运行验证，由用户自行运行上述命令。前面的无交通弯道速度对照不能替代最终多车验证。
