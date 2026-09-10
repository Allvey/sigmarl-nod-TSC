# 道路安全诊断

独立加载已训练的 Actor 和匹配的 Safety Value，执行确定性评估；不训练、不修正动作、不修改 v2 配置。

```bash
python utilities/diagnose_road_safety.py --data outputs/dgppo_minimal_v2/reward6.38_data.json
```

默认 intersection_2、8 个环境、1199 步，沿用训练 JSON 的 seed。可用 `--seed`、`--scenario`、`--envs`、`--steps`、`--seconds` 调整。`--data` 必须指向与要测试的最佳奖励检查点匹配的 JSON；本工具不选择 final 检查点。

结果保存到模型目录下新建的 `road_diagnostics/<时间戳>/`：

- `summary.json`：碰撞次数、路线分布、每次碰撞的预警提前量和预警步数。零碰撞时仍输出汇总。
- `collision_windows.csv`：每次车道碰撞的最近 20 个转移（包含碰撞转移）；不足 20 步或中途重生则截断。记录路径编号、身份、位置、速度、朝向、Actor 动作、道路几何风险、Value、屏障残差、安全惩罚及任务优势是否会被屏蔽。

路径编号从 0 开始，对应 `utilities/constants.py` 中该场景的 `reference_paths_ids`。intersection_2 的路径 1 对应 `[1,2,6,11]`，路径 5 对应 `[8,7,10]`。

位置、速度、朝向、净空和 `g_next` 对应动作执行后的状态；`g`、`value` 和动作对应执行前的状态。`value_next` 是下一状态的网络预测。`c_raw` 全部使用网络预测；`c_training` 在环境内任一车辆碰撞时以物理下一状态风险代替下一 Value，模拟训练时整环境终止的规则。实际评估仍按原逻辑逐车重生。该规则模拟不是完整训练复现。

`road_penalty` 仅对应道路头；`all_head_penalty` 为所有有效安全头取最大后的惩罚；`task_masked` 表示按现有训练规则是否会屏蔽任务优势。这里没有实际执行 Actor 更新。`valid=false` 的转移不用于预警统计。车辆碰撞使用环境重生前的逐车标志，而非广播到所有车的 info 标志。

先检查主故障路线是否在碰撞前出现持续的正屏障残差。仅 Value 为负不能证明未预警，单次残差为正也不能证明预警可靠。再结合几何阈值、预警时间和训练样本分布决定修改 Critic 拟合还是 Actor 更新。测试场景中的离线违规不能直接证明训练时曾收到同类安全梯度。

## v2 首次完整诊断

输出：`outputs/dgppo_minimal_v2/road_diagnostics/20260910_150036_521027/`。

使用 reward6.38、原 seed、8 环境、1199 步，复现 336 次逐车边界碰撞，碰撞时间步占比为 3.4404%。所有事件在可用窗口内至少出现过一次正道路屏障残差，但首次预警提前量中位数仅为 0.15 秒；路径 1 为 0.15 秒，路径 5 为 0.10 秒。所有有效碰撞前转移中，约 22.1% 道路屏障残差为正；纯网络残差对应比例约 22.2%。

这支持优先检查预警提前量，不能依据 Value 正值召回率直接认定 Critic 完全没有预警，也不能据此断言 Actor 曾在训练中忽略相同风险。首次预警时间受到 20 步窗口截断影响，只描述该窗口内的情况。本次没有更改训练配置或安全损失。

## 下一轮：道路余量单变量实验

新增 `config_dgppo_road_margin.json`，只将 v2 的 `safety_boundary_margin` 从 0.01 m 改为 0.02 m，并使用独立输出目录 `outputs/dgppo_road_margin/`。现有几何标签已经读取此参数，不需要修改网络或 DGPPO 更新公式。Actor 15 epochs、Value 1 epoch、20 批预热、安全权重及 seed 均保持一致。

道路风险为 `max(-1, (margin - clearance) / margin)`。例如净空为 1.5 cm 时，旧标签为 -0.5，新标签为 0.25；当车靠近边界时，新标签更早进入正值区域。余量也是归一化分母，因此同时改变风险数值尺度；这不是安全惩罚权重翻倍，也不意味着预警秒数翻倍。2 cm 是待验证的实验值，需检查窄路是否过度保守。

由用户手动从零训练：

```bash
python main_training.py --config config_dgppo_road_margin.json
```

旧 Safety Value 的检查点契约包含 `boundary_margin`，不能把 1 cm 模型直接当作 2 cm 模型继续使用。新配置已关闭模型加载和继续训练。

训练后，将 `main_testing.py` 的 `path` 设为 `outputs/dgppo_road_margin/`，手动运行 `python main_testing.py`。定量评估时，把 `evaluation_tase26.py` 的 `model_paths`、`where_to_save_eva_results` 和 `where_to_save_logging` 改到新目录，然后运行 `python utilities/evaluation_tase26.py`，以免覆盖 v2 评估结果。

再用新目录中实际生成的 `reward*_data.json` 执行诊断，例如将以下占位路径替换为真实文件名：

```bash
python utilities/diagnose_road_safety.py --data outputs/dgppo_road_margin/rewardXXX_data.json
```

比较相同场景、seed、环境数和仿真长度下的碰撞率、速度与预警提前量，重点关注路径 1 和 5，并复查 CPM_entire、on_ramp_1、roundabout_1。不同模型的碰撞样本会变化，因此不能仅凭“剩余碰撞的预警中位数”判断整体改善。本轮不同时增加 Value epochs，以免无法归因。训练、仿真测试均由用户执行。
