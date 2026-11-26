# 三阶段误差尺度训练

在项目根目录依次执行，各阶段正常结束后再开始下一阶段：

```bash
python main_training.py --method nod --config configs/current/error_scale_three_stage/01_bootstrap.json
python main_training.py --method nod --config configs/current/error_scale_three_stage/02_candidate.json
python main_training.py --method nod --config configs/current/error_scale_three_stage/03_control.json
```

1. 250 轮：所有网络随机初始化，联合学习基础控制与 responsibility NOD，几何 e、
   posterior 方差监督；暂不通过意见调节 alpha。
2. 30 轮：加载本轮第一阶段 final；冻结行为模型，用独立 learner NOD 学习几何证据
   相对未来目标的误差尺度，并监督意见。保留历史/A 初始化，新尺度输出初始化为方差 1。
3. 70 轮：加载本轮候选阶段 final；冻结 NOD，20 轮 Safety Value warmup 后进行
   50 轮控制更新。启用意见 alpha，gain=1、死区 0.1，不训练 PPO beta。

种子均为 5080868027432654403。输出在
`checkpoints/nod_error_scale_three_stage/seed5080868027432654403/` 的
`01_bootstrap/`、`02_candidate/`、`03_control/` 中。

此流程不依赖旧 reward7.21，复用的是三阶段方法，并非历史 reward7.38 全训练轨迹的
精确复现。总计 350 轮、1,433,600 团队环境帧。已检查配置衔接、NOD 版本迁移与冻结
兼容性；训练和性能确认由实际运行完成。

候选阶段优先读取源 `final_nod.pth` 内嵌的方差范围，不要求 `final_data.json`。
只有不含该字段的旧 NOD 权重才需要同前缀的 `_data.json`；不使用其他奖励快照代替。
