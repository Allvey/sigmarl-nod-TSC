# NOD 交互参照标签微调

本次只修改训练标签，不修改在线证据、NOD 网络或 Actor 的输入结构。
旧配置默认 `nod_label_mode="instantaneous"`，保留原标签及检查点加载方式。

## 新标签

`nod_label_mode="interaction"` 使用按时间顺序生成的车辆对参照：

1. 根据当前相对位置、速度预测最近接近距离。预测窗口内正在接近且最近距离不大于 `1.5 * nod_safe_distance`，邻车速度大于 0.05 m/s 时，记录邻车速度。选择参照不读取未来轨迹。
2. 此次交互中保留该速度，包括邻车随后停车的阶段。每个标签从邻车**当前位置**按保留速度预测未来，比较“继续等待”与“恢复此前运动趋势”，不从历史位置生成已经穿过路口的虚拟车辆。
3. 本车未来轨迹在真实与对照两种计算中相同。沿用窗口内最大距离风险之差作为改善量 `gap`。
4. 对称中性区间为 `signed_gap = sign(gap) * max(abs(gap) - margin, 0)`，标签为 `sigmoid(label_slope * signed_gap)`。区间内标签为 0.5；无活动参照时归因改善量为零。
5. 参照预测不再有前方冲突、失去观测、邻居编号改变、任一车辆重生或超过最长时间时失效。邻车有明显速度时，方向相对参照改变超过 30 度也会使参照失效；停车不视为转向。超时或转向后需冲突解除才能重新建立参照。

默认最长参照时间 `nod_reference_seconds=2.0` 秒。新配置标签窗口为 20 步，在 dt=0.05 时为 1 秒。参照缓存只覆盖当前训练 rollout，不跨批次保存；批次起点已经停车的车辆不会被凭空赋予避让贡献。

这是短期匀速代理评价，不能保证识别所有弯道避让，也不是合作意图真值。标签修正不会直接改变旧模型的视频输出。在线证据仍使用原定义，其与新标签的一致性属于下一步工作。

## 使用

从已有 reward7.28 检查点开启 NOD 更新：

```bash
python main_training.py --config configs/archive/nod_history/config_dgppo_nod_interaction_finetune.json
```

输出：`outputs/archive/dgppo_nod_interaction_finetune/`。
该配置复用现有联合微调流程，Actor 和 Safety Value 仍按原流程训练，并非仅训练 NOD。没有新增单独的 NOD 训练入口。
训练指标新增 `label_positive_ratio`、`label_neutral_ratio`、`label_reference_active_ratio`，用于观察标签分布与活动参照比例。

训练后将 main_testing.py 的模型目录切换至新输出再可视化。无需调整 alpha 增益来启用新标签。

## 待用户运行的回归

```bash
python -m pytest tests/test_nod_counterfactual.py -q
```

覆盖制动后停车贡献保留、未参与冲突的静止车辆中性、匀速中性、增加风险为负、通行后释放、超时、转向、重生、可见性中断，以及参照选择的因果性。
本次未运行测试或训练，不能视为已验证行为效果。
