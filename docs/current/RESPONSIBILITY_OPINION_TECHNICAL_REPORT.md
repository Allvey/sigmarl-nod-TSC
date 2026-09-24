# 无通信多车责任意见与安全学习：技术报告

> 状态：当前选定方法的技术说明稿，依据 `configs/current/config_dgppo_nod_gain1_control_finetune.json` 及现有代码整理。本文描述问题、方法、网络实现、优化目标、训练流程和能够成立的理论性质；不包含实验设计、实验结果或投稿结论。历史规划与当前实现不一致时，以当前配置和代码为准。

## 摘要

在没有车辆间显式通信的交互中，本车无法直接获得邻车的避让意图或未来控制动作。固定、对称的避让规则可能过于保守；直接将邻车减速解释为合作，又可能错误地把本车自行避让造成的风险下降归因给邻车。本项目首先构造有方向的反事实责任指标：固定本车在冲突开始时的速度基准，只改变邻车的观测运动，估计邻车已提供的风险缓解占解除冲突所需缓解的比例。随后，以该指标作为在线证据及未来监督目标，通过带 Bernoulli KL 惯性的非线性意见动力学形成有界意见。最后，意见在训练期间调节成对 Safety Value 的离散屏障系数，并作为局部消息输入 Actor。意见不改变物理安全距离，也不构成部署时的动作级安全过滤器；神经网络近似和 PPO 更新同样不提供无碰撞保证。

## 1. 问题背景与研究目标

考虑 \(N\) 辆在交叉口、汇入区等结构化道路上交互的车辆。本车 \(i\) 可使用自身状态与感知范围内邻车的位置、速度及由此计算的局部几何特征，却不接收邻车的内部策略、计划动作或私有路径。部署决策只能依赖时刻 \(t\) 及此前的信息。

本文关注的问题不是识别邻车是否具有永久不变的“合作性格”，而是估计当前一场潜在冲突中，邻车 \(j\) **相对于其冲突开始时的运动基准，实际承担了多少风险缓解**。这个判断具有方向性：\(i\) 对 \(j\) 的意见 \(z_{ij}\) 与 \(j\) 对 \(i\) 的意见 \(z_{ji}\) 分别由各自可得的历史形成，二者不要求相同，也不要求对应的责任比例相加为一。

研究目标可以表述为：在保持物理安全约束定义不变的条件下，把局部历史形成的责任意见用于学习不同交互对象对应的避让压力，并分析该机制何时有效、何时不能推出安全保证。与已有工作的位置关系如下：AVOCADO 已经研究由传感观测驱动的避碰意见动力学；责任感知 CBF 已研究场景相关的责任分配；DGPPO 已研究学习型离散屏障与多智能体策略优化。本项目的具体组合是**反事实责任监督、具有固定语义零点的意见演化，以及责任调节的成对安全优势更新**，不主张首次提出意见、责任分配或 CBF。[AVOCADO](https://doi.org/10.1109/TRO.2025.3552350)；[责任感知 CBF](https://arxiv.org/abs/2303.03504)；[DGPPO](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b0916c29bc32d57e2c86a63e8f2df9f2-Abstract-Conference.html)。

## 2. 系统、信息与符号

令 \(x_{i,t}\) 为车辆 \(i\) 的物理状态，\(u_{i,t}\) 为加速度、转向等控制量，离散动力学写作

\[
x_{i,t+1}=F_i(x_t,u_{1,t},\ldots,u_{N,t},\xi_t),
\]

其中 \(\xi_t\) 表示可能的扰动。动力学在此仅用于定义问题；本文当前实现不假设已获得一个可用于严格安全证明的精确 \(F_i\)。本车观测为 \(o_{i,t}\)，其可见邻车集合为 \(\mathcal N_i(t)\)。部署策略根据本车观测、邻居意见消息 \(m_{i,t}\) 和上一动作形成 \(u_{i,t}\sim\pi_\theta(\cdot\mid o_{i,t},m_{i,t},u_{i,t-1})\)。相同世界编号与车辆代际编号共同标识一条持续的有向边 \(i\leftarrow j\)，避免车辆重生或邻居槽位变化时将旧意见误用于新对象。

当前 `local_kinematics` 模式只将可见邻车的局部运动学特征提供给 NOD；路径相关特征被清零。可见性掩码用于维持历史，**交互掩码**用于判断责任意见是否有效。无预测冲突时，程序输出数值上的 \(z_{ij}=0\)，同时将交互标记为无效；这一零值表示接口的中性占位，而不是“证实邻车承担了恰好一半责任”。[局部输入契约](../../utilities/nod_marl/interaction.py)；[意见状态](../../utilities/nod_marl/opinion.py)。

本文使用两种不同的距离尺度。NOD 责任风险使用 \(d_{\mathrm{nod}}\)，成对物理约束使用 \(d_{\mathrm{safe}}\)。当前选定配置分别为 0.20 m 与 0.25 m；它们服务于不同计算，不应合并为同一个“共享安全距离”。[当前配置](../../configs/current/config_dgppo_nod_gain1_control_finetune.json)。

| 记号 | 含义 |
| --- | --- |
| \(q_{ij,t}\in[0,1]\) | 邻车相对于本车承担的估计风险缓解比例 |
| \(e_{ij,t}=2q_{ij,t}-1\) | 有符号在线责任证据 |
| \(z_{ij,t}\in(-1,1)\) | 本车对邻车的动态责任意见 |
| \(\rho_{ij,t}=(1+z_{ij,t})/2\) | 与软责任标签对齐的意见表示 |
| \(A_{ij,t}\in(0,1)\) | 由可观测风险特征计算的关注强度 |
| \(\sigma_{ij,t}^{2}>0\) | 模型学习的证据方差 |
| \(g_{ij,t}\) | 成对距离的物理约束值，非正表示距离侧安全 |
| \(V_{ij,t}\) | 学习得到的成对 Safety Value |
| \(\alpha_{ij,t}\) | 训练期离散屏障的成对系数 |

## 3. 反事实责任证据与监督目标

### 3.1 冲突与速度锚点

对车辆对 \((i,j)\)，记当前相对位置 \(r_t=p_{j,t}-p_{i,t}\)，相对速度 \(u_t=v_{j,t}-v_{i,t}\)。给定前瞻时间 \(H\)，匀速最近点时间和风险定义为

\[
t^*(r,u)=\operatorname{clip}\!\left(-\frac{r^\top u}{\max(\|u\|^2,\varepsilon)},0,H\right),
\qquad
R(r,u)=\exp\!\left(-\frac{\|r+t^*(r,u)u\|}{d_{\mathrm{nod}}}\right).
\]

当车辆正在接近，且最近点风险高于 \(R_s=e^{-1}\) 时，认为存在潜在冲突。在首次激活的冲突中，缓存双方当时的速度 \(v_i^0,v_j^0\)。后续风险比较均从**当前相对位置**出发，而非将两辆“幽灵车”继续从旧位置外推。交互身份丢失、基准冲突消失或锚点超时会结束本次责任窗口；锁定状态防止同一持续冲突立刻重新设定有利的基准。[在线责任计算](../../utilities/nod_marl/counterfactual.py)。

### 3.2 在线责任比例

固定本车速度锚点，分别计算基准风险和使用邻车当前观测速度的风险：

\[
R^0_{ij,t}=R(r_t,v_j^0-v_i^0),\qquad
R^{\mathrm{obs}}_{ij,t}=R(r_t,v_{j,t}-v_i^0).
\]

仅对有效冲突定义

\[
q^{\mathrm{on}}_{ij,t}=
\operatorname{clip}\!\left(
\frac{R^0_{ij,t}-R^{\mathrm{obs}}_{ij,t}}
     {\max(R^0_{ij,t}-R_s,\varepsilon)},0,1\right),
\qquad e_{ij,t}=2q^{\mathrm{on}}_{ij,t}-1.
\]

\(q=0\) 表示按此风险模型邻车尚未提供所需缓解；\(q=0.5\) 表示提供一半；\(q=1\) 表示邻车提供的缓解已经达到模型定义的需求。数值截断使指标有界，但也丢弃了“风险变得更坏”和“缓解超出需求”的幅度。该比例是*风险缓解比例*，不是以米为单位的避让距离份额。

把 \(v_i^0\) 固定在两种风险计算中，使本车后续减速不会**直接**进入分子。因此，算法不会仅因本车停车就给邻车记正面责任。然而邻车的真实动作仍可能是对本车行为的响应；这个计算约束消除了一个明显的算术混淆，不等同于对交互因果关系的完整识别。

### 3.3 使用未来轨迹的训练标签

训练数据允许读取后续 \(h=1,\ldots,H_y\) 帧。令本车固定锚定轨迹为 \(\hat p_i(h)=p_{i,t}+h\Delta t\,v_i^0\)，邻车匀速反事实为 \(\hat p_j^0(h)=p_{j,t}+h\Delta t\,v_j^0\)，真实邻车位置为 \(p_{j,t+h}\)。于是

\[
R^{\mathrm{cf}}_{ij,t}=\max_{1\le h\le H_y}
\exp\!\left(-\frac{\|\hat p_i(h)-\hat p_j^0(h)\|}{d_{\mathrm{nod}}}\right),
\]

\[
R^{\mathrm{real}}_{ij,t}=\max_{1\le h\le H_y}
\exp\!\left(-\frac{\|\hat p_i(h)-p_{j,t+h}\|}{d_{\mathrm{nod}}}\right),
\]

\[
q^*_{ij,t}=\operatorname{clip}\!\left(
\frac{R^{\mathrm{cf}}_{ij,t}-R^{\mathrm{real}}_{ij,t}}
{\max(R^{\mathrm{cf}}_{ij,t}-R_s,\varepsilon)},0,1\right).
\]

该标签只在冲突有效、整个预测区间内车辆身份未更换且未来数据齐全时参与监督。未来邻车轨迹只用于**训练标签**；在线推理只使用已经观测到的位置和速度。在线 \(q^{\mathrm{on}}\) 使用匀速最近点近似，\(q^*\) 使用未来轨迹上的最大风险，两者定义一致但数值不必相等。[未来标签](../../utilities/nod_marl/counterfactual.py)。

## 4. 责任意见动力学

### 4.1 风险关注、历史与固定证据映射

对局部成对特征 \(\phi_{ij,t}\)，模型先形成历史 \(h_{ij,t}=\operatorname{GRU}(\phi_{ij,t},h_{ij,t-1})\)。另由非负权重加权的风险特征生成

\[
A_{ij,t}=\operatorname{sigmoid}\!\left(T\left[\sum_k w_k r_k(\phi_{ij,t})-\vartheta\right]\right),
\qquad w_k\ge0.
\]

历史网络输出有界的 \(\sigma_{ij,t}^2\)。当前责任模式固定观测均值映射为 \(\mu(z)=b+sz=z\)，即 \(b=0,s=1\)。这避免模型单靠可学习截距移动意见零点，或通过改变斜率反转/重标度证据；但它**不自动保证**所有真实场景中的意见已经校准。风险关注权重、历史编码与方差仍由数据学习。[风险与参数](../../utilities/nod_marl/opinion.py)。

### 4.2 KL 近端更新

令 \(p(z)=(1+z)/2\)，并记 \(\lambda=\tau/\Delta t\)、\(a\) 为非线性增益、\(w\) 为证据权重。每个有效时间步求解有界标量优化

\[
z_t=\arg\min_{|z|<1}\;J_t(z),
\]

\[
J_t(z)=\frac{z^2}{2}-\frac{A_t}{a}\log\cosh(az)
+\frac{wA_t}{2}\left[\frac{(e_t-z)^2}{\sigma_t^2}+\log\sigma_t^2\right]
+\lambda D_{\mathrm{KL}}\!\left(
\operatorname{Bern}(p(z))\,\|\,\operatorname{Bern}(p(z_{t-1}))\right).
\]

省略车辆对下标后，内部解的一阶条件为

\[
\lambda\bigl[\operatorname{atanh}(z_t)-\operatorname{atanh}(z_{t-1})\bigr]
=-z_t+A_t\tanh(az_t)+\frac{wA_t}{\sigma_t^2}(e_t-z_t).
\]

KL 项提供状态记忆，\(-z\) 使意见倾向中性，非线性项允许风险相关的意见持续，最后一项按关注程度与方差吸收新证据。实现对有界区间求根，并以隐式微分训练相关参数；新出现的边首先以 \(z=0\) 初始化，后续有效帧才演化。无冲突时关闭意见掩码，并将输出归零；身份更换时重置历史。[更新实现](../../utilities/nod_marl/opinion.py)。

### 4.3 监督目标

令 \(\rho_t=(1+z_t)/2\)。训练损失的主要部分为

\[
\mathcal L_{\mathrm{NOD}}
=\lambda_{\mathrm{nll}}\,\mathbb E\!\left[
\tfrac12\left(\frac{(e_t-z_t)^2}{\sigma_t^2}
+\log\sigma_t^2+\log(2\pi)\right)\right]
+\lambda_{\mathrm{cal}}\,\mathbb E\!\left[
\operatorname{BCE}(\rho_t,q^*_t)\right].
\]

期望只取各自有效的交互样本。\(q^*\) 是 \([0,1]\) 内的软责任目标；BCE 使 \(\rho\) 接近该目标，不能仅凭这个损失就宣称 \(\rho\) 是经过独立验证的事件概率。标签没有直接监督某个单独参数 \(A_t\) 或 \(\sigma_t\)，而是通过隐式更新和输出损失间接塑造它们。[训练损失](../../utilities/nod_marl/trainer.py)。

### 4.4 可以证明的局部性质

固定当步 \(A_t\) 和 \(\sigma_t^2\)，目标在内部的曲率为

\[
J_t''(z)=\frac{\lambda}{1-z^2}+1-A_ta\operatorname{sech}^2(az)
+\frac{wA_t}{\sigma_t^2}.
\]

若 \(0<A_t<1\)、\(w\ge0\) 且 \(\lambda+1>a\)，则 \(J_t''(z)>0\)，内部最优解唯一。当前配置给出 \(\lambda=0.25/0.05=5\)、\(a=2\)，满足该充分条件。再由隐式函数定理，在 \(w>0\)、其他量固定且内部解成立时，

\[
\frac{\partial z_t}{\partial e_t}
=\frac{wA_t/\sigma_t^2}{J_t''(z_t)}>0.
\]

因此，对**同一历史与相同模型状态**，更正面的证据使更新后的意见不降低；\(z_{t-1}=e_t=0\) 时零意见也是唯一解。这个结论不等于在整个真实闭环中 \(z\) 必然随观察到的邻车减速单调增加：车辆位置、风险关注、方差和过去意见也同时在变化。

## 5. 意见调节的安全学习

### 5.1 物理约束与 Safety Value

车辆对的即时物理约束写为

\[
g_{ij}(s_t)=\operatorname{clip}_{[-1,\infty)}
\left(\frac{d_{\mathrm{safe}}-d_{ij,t}}{d_{\mathrm{safe}}}\right),
\qquad g_{ij}\le0\Longleftrightarrow d_{ij,t}\ge d_{\mathrm{safe}}
\quad\text{（在有效车辆对上）。}
\]

道路边界与实际碰撞使用另外的约束头，邻车意见只可能修改成对头。共享的成对编码器和本地聚合网络得到 \(V_{ij}(s_t)\)。训练目标基于观测轨迹的多步、折扣最大风险备份；其单步结构可概括为

\[
\mathcal T V_{c,t}=\max\{g_{c,t},(1-\gamma)\max_k g_{k,t}+\gamma \widetilde V_{c,t+1}\},
\]

再对不同备份长度做 \(\lambda\) 混合。\(c\) 可表示成对、道路或碰撞头，故成对 Value 的目标也会受同一车辆其他约束头影响。它是*学习得到的风险价值近似*，并非经过形式验证的真实约束上界或无穷时域最坏风险。[状态与约束头](../../utilities/nod_marl/safety_value.py)；[备份目标](../../utilities/nod_marl/dgppo.py)。

### 5.2 成对屏障系数与伪优势

只有当车辆身份对齐、意见有效、\(g_{ij,t}\le0\)、\(V_{ij,t}<0\)，且 \(|z_{ij,t}|>z_{\rm dead}\) 时，才应用

\[
\alpha_{ij,t}=\alpha_0+\Delta\alpha\,
\operatorname{clip}(kz_{ij,t},-1,1).
\]

其他情况使用基础值 \(\alpha_0\)；道路和实际碰撞约束头不接受邻车意见。当前配置为 \(\alpha_0=10\)、\(\Delta\alpha=5\)、\(k=1\)、\(z_{\rm dead}=0.1\)，可应用时 \(\alpha\in[5,15]\)。正意见提高 \(\alpha\)，负意见降低 \(\alpha\)；意见缺失不会被解释为负意见。正在危险侧的成对 Value 使用基础恢复系数，避免由正意见放宽恢复要求。[系数映射](../../utilities/nod_marl/dgppo.py)；[身份对齐](../../utilities/nod_marl/barrier.py)。

每个有效约束的训练期离散屏障量定义为

\[
\delta_{i,c,t}
=\frac{V_{i,c,t+1}-V_{i,c,t}}{\Delta t}
+\alpha_{i,c,t}V_{i,c,t}.
\]

若任一有效头 \(\delta_{i,c,t}>0\)，当前 `gated` 模式删除对应样本的任务优势；同时用 \(\beta\max_c[\delta_{i,c,t}+\epsilon]_+\) 扣减优势，其中当前 \(\beta=1\)、\(\epsilon=0.01\)。设 \(\widehat A^{\rm task}_{i,t}\) 为原任务优势，可写成

\[
\widehat A^{\rm used}_{i,t}
=\mathbf 1\{\text{无有效违规}\}\widehat A^{\rm task}_{i,t}
-\beta\max_c[\delta_{i,c,t}+\epsilon]_+,
\]

式中最大值仅覆盖有效约束头。前提是该样本具备完整有效约束，或至少已有明确违规；未知后继且无已知违规时回退到任务优势。这个优势被 `detach` 后供 PPO 的截断策略目标更新；若记新旧策略概率比为 \(r_\theta\)，其标准形式为

\[
\max_\theta\;\mathbb E\!\left[\min\!\left(
r_\theta\widehat A^{\rm used},
\operatorname{clip}(r_\theta,1-\epsilon_{\rm PPO},1+\epsilon_{\rm PPO})
\widehat A^{\rm used}\right)\right].
\]

因此这里不对 NOD 或 Safety Value 做穿透式策略梯度。屏障中的 \(\epsilon\) 会让非常接近零、但尚未为正的 \(\delta\) 也产生轻微惩罚，它与 PPO 截断参数 \(\epsilon_{\rm PPO}\) 不同。[优势构造](../../utilities/nod_marl/dgppo.py)。

在安全侧固定 \(V_{ij,t}<0\) 及其他输入时，\(\partial\delta_{ij,t}/\partial z_{ij,t}=\Delta\alpha\,kV_{ij,t}<0\)（未饱和、未落入死区）。正意见使同一物理转移更不容易触发该成对头的违规；负意见反之。不过总体 PPO 优势还经过其他约束头的最大值与门控，不能仅由这个局部导数推出所有状态下 Actor 动作对意见单调。

### 5.3 训练路径与执行路径

两条路径必须分开描述：

```text
训练：局部历史 → z → 成对 α → 安全违规量 → PPO 优势 → Actor 参数更新
执行：局部历史 → z 与其他 NOD 上下文 → 消息聚合 → Actor 动作
```

Actor 消息包含 NOD 历史状态、局部风险特征、风险关注和意见，经过按邻居聚合的消息网络后，与本车观测及上一动作拼接。该路径允许 Actor 使用意见，但**仅有输入通道并不能证明 Actor 已经实质使用意见**。Safety Value/\(\alpha\) 路径只在训练期改变学习信号；当前部署策略不通过它逐动作求解安全投影。当前配置中的 `is_using_safety_critic=false` 指旧的独立 Safety Critic 分支未启用；本文所讨论的安全机制是独立训练的 `PairSafetyValue` 与 DGPPO 风格优势更新。[Actor 消息](../../utilities/nod_marl/policy.py)；[配置](../../configs/current/config_dgppo_nod_gain1_control_finetune.json)。

### 5.4 条件安全侧命题与结论边界

假设某成对 Value 当前 \(V_t\le0\)，对真实执行转移**精确**满足 \(\delta_t\le0\)，并且 \(0<\alpha_t\Delta t\le1\)。由定义得到

\[
V_{t+1}\le(1-\alpha_t\Delta t)V_t\le0.
\]

这说明理想屏障条件保持的是 \(V\le0\) 的边界，调节 \(\alpha\) 改变允许接近边界的速度，而不是修改安全距离。当前 \(\Delta t=0.05\)、\(\alpha\in[5,15]\)，故在被意见调节的安全侧有 \(\alpha\Delta t\in[0.25,0.75]\)。

这**不是当前系统的物理无碰撞定理**。其前提在代码中并未被逐动作强制：\(V\) 是神经网络近似，\(V\le0\) 未被证明蕴含 \(g\le0\)，PPO 更新不保证每个未来转移满足 \(\delta\le0\)，扰动、未观测车辆与策略共同适应也会改变轨迹。论文应将上述结果限定为“理想屏障条件下的安全侧不变性”，并将实际碰撞表现视为经验问题。

## 6. 网络结构、损失函数与参数更新

本节中的 \(D_o\) 表示环境在运行时给出的单车原始观测维度；动作维度为 2。网络共享权重不代表车辆共享观测或通信：执行时每辆车仍分别用自己的观测、邻居历史和同一套 Actor 参数计算动作。当前配置训练 4 辆车，但以下成对模块不把邻车数量固定为网络层宽度。[主网络构建](../../utilities/mappo_cavs.py)；[当前配置](../../configs/current/config_dgppo_nod_gain1_control_finetune.json)。

### 6.1 从车辆对输入到 NOD 状态

每条有向可见边 \(i\leftarrow j\) 生成 20 维成对向量 \(\phi_{ij,t}\)。它包括相对位置 2 维、相对速度 2 维、相对航向的正余弦、距离、闭合速度、TTC、若干路径/冲突相关槽位、接近置信度、可见标记及双方速度。当前 `local_kinematics` 模式将第 9—15 号路径/冲突槽位置零，保留张量维度以兼容网络和检查点；因此“20 维输入”不表示 20 维都有非零观测信息。邻居槽位的有效性由可见掩码和车辆身份共同决定。[特征索引](../../utilities/nod_marl/interaction.py)。

NOD 对每条边独立维护 64 维 GRU 隐状态。具体数据流为：

```text
当前成对特征 20 + 上一隐状态 64
  → GRUCell(20, 64) → h_ij,t ∈ R^64
  ├─ Linear(64, 64) → Tanh → Linear(64, 3)
  │    └─ 当前责任模式仅使用第 3 个输出计算 log σ；b=0、s=1 固定
  ├─ 6 个风险分量 → 6 个非负可学习权重 → sigmoid → A_ij,t
  └─ 在线证据 e_ij,t、上一意见 z_ij,t−1、A_ij,t、σ²_ij,t
       → 一维隐式 KL 求根 → z_ij,t
```

方差由 \(\log\sigma=\ell_{\min}+(\ell_{\max}-\ell_{\min})\operatorname{sigmoid}(r_3)\) 得到，当前默认 \(\ell_{\min}=-2.5,\ell_{\max}=0\)。风险权重以 sigmoid 参数化在 \((0,1)\)；风险温度和阈值当前分别为 2 与 1.25。在局部运动学模式下，六个风险分量中的冲突有效性、ETA 紧迫度和路径重叠量因对应槽位清零而不提供有效变化，主要变化来自距离、接近程度与 TTC。求根最多进行 48 次迭代，意见在 \((-1+10^{-4},1-10^{-4})\) 内。前两路似然头输出虽然保留形状，责任模式下不参与 \(b,s\) 的计算。[NOD 网络与风险量](../../utilities/nod_marl/opinion.py)。

监督学习时使用上一节的 \(\mathcal L_{\mathrm{NOD}}=\mathcal L_{\mathrm{NLL}}+\mathcal L_{\mathrm{BCE}}\)，两项权重当前均为 1。NLL 只在同一边已形成可延续意见、交互有效的时间步计算；BCE 还要求未来责任标签有效。代码记录 Brier 误差作为诊断量，**它不进入反向传播的损失**。优化器为独立的 Adam，当前学习率 \(10^{-3}\)。训练保持时间顺序：每个环境的轨迹被切成 32 步序列，按最多 8 个环境组成小批量，重复 4 个 epoch；序列间传递但截断隐状态梯度。[NOD 损失与序列训练](../../utilities/nod_marl/trainer.py)。

### 6.2 消息聚合、Actor 与任务 Critic

每条可见边的 Actor 上下文由 `[GRU hidden 64, risk components 6, A 1, z 1]` 拼接而成，共 72 维。聚合器先执行无仿射参数的 `LayerNorm(72)`，再经 `Linear(72,64) → Tanh → Linear(64,32) → Tanh` 得到边消息。另一个 `Linear(32,1)` 给出邻居打分；对有效边做 masked softmax 和加权求和，最后以 \(0.1\tanh(\cdot)\) 限制聚合消息的每个分量。没有有效邻居时消息为零。即使该邻居当前没有责任意见，只要可见，其物理历史上下文仍可进入 Actor；交互掩码只控制 \(z\) 的有效性。[消息聚合](../../utilities/nod_marl/policy.py)；[在线上下文](../../utilities/nod_marl/trainer.py)。

Actor 的实际输入为 `原始单车观测 D_o + NOD 消息 32 + 上一动作 2`，即 \(D_o+34\) 维。共享、分散式 `MultiAgentMLP` 设置为两层 256 单元、Tanh 激活，输出 4 个分布参数（二维动作各有位置参数和尺度参数）；`BoundedNormalParamExtractor` 与 `TanhNormal` 将其变成受动作范围约束的随机策略。邻居消息聚合器和 Actor MLP 一起由 PPO 损失更新，但在线 GRU/NOD 上下文被 `detach`，PPO 梯度不会进入 NOD。[Actor 构建](../../utilities/mappo_cavs.py)；[输入组装](../../utilities/nod_marl/policy.py)。

任务 Critic 与 Safety Value 是不同网络。任务 Critic 使用原始观测和 `centralised=True` 的两层 256 单元 `MultiAgentMLP`，输出每车一维任务价值；它不以意见消息为输入。这个集中式 Critic 仅在训练时为 GAE/PPO 提供任务价值；执行动作仍由分散式 Actor 决定。[任务 Critic](../../utilities/mappo_cavs.py)。

令任务价值为 \(V^{\mathrm{task}}_\psi\)，标准 GAE 可概括为

\[
\delta^r_t=r_t+\gamma V^{\mathrm{task}}_\psi(s_{t+1})-V^{\mathrm{task}}_\psi(s_t),
\qquad
\widehat A^{\mathrm{task}}_t=\sum_{\ell\ge0}(\gamma\lambda_{\mathrm{GAE}})^\ell\delta^r_{t+\ell}.
\]

当前 \(\gamma=0.99,\lambda_{\mathrm{GAE}}=0.9\)。第 5 节的屏障规则只替换供 Actor 使用的优势，不修改任务 GAE 的价值目标。实际 PPO 总损失是 TorchRL `ClipPPOLoss` 的 `loss_objective + loss_critic + loss_entropy`：分别对应负的截断策略目标、任务 Critic 对 GAE 价值目标的 Smooth L1 误差，以及系数 \(10^{-4}\) 的负熵项。截断系数为 0.2。Actor/任务 Critic 与消息聚合器的 Adam 参数组分开管理，当前两组初始学习率都是 \(5\times10^{-5}\)。[PPO 实际更新](../../utilities/mappo_cavs.py)。

### 6.3 Safety Value 网络和独立回归损失

`PairSafetyValue` 不复用任务 Critic。它把按世界车辆编号排列的 20 维成对特征输入 `Linear(20,64) → Tanh`，把 `[本车原始观测 D_o, 道路余量, 碰撞余量]` 输入 `Linear(D_o+2,64) → Tanh`。对本车有效邻居的 64 维边嵌入取 masked mean，再与本车嵌入组成 128 维上下文。每个成对头采用 `Linear(192,64) → Tanh → Linear(64,1)`；道路和实际碰撞两个本地头共用 `Linear(128,64) → Tanh → Linear(64,2)`。因此每车输出为 `N 个按世界编号索引的成对头 + 1 个道路头 + 1 个碰撞头`，而不是把多个邻居压缩为一个安全数值。[Safety Value 网络](../../utilities/nod_marl/safety_value.py)。

当前 Safety Value 对有效的、前后代际一致的样本使用第 5 节所述多步最大风险 \(\lambda\) 目标 \(y^V_{i,c,t}\)。现行损失为

\[
\mathcal L_V(\varphi)
=\frac{1}{2|\mathcal M|}
\sum_{(i,c,t)\in\mathcal M}
\left(V_{\varphi,i,c}(s_t)-y^V_{i,c,t}\right)^2,
\]

其中 \(\mathcal M\) 是有效身份与有限值掩码。当前 \(\gamma_V=0.99\)、多步混合系数 \(\lambda_V=0.95\)；真实终止时使用观察到的物理约束值，采样片段截断时才引入模型 bootstrap。目标在更新前由 Safety Value 快照计算并 `detach`，再用独立 Adam（学习率 \(10^{-3}\)）训练 1 个 epoch、小批量 512。配置选择的是普通 MSE，不是文件中保留的旧式类别加权或低估惩罚损失。[目标和损失实现](../../utilities/nod_marl/safety_value.py)；[多步目标](../../utilities/nod_marl/dgppo.py)。

Safety Value 的数据来自额外的确定性策略影子采样器，而不是直接复用随机 PPO 采样批。当前每批用 32 个独立环境、每环境 128 步；采样前同步 Actor/NOD 权重，以隔离随机数流收集轨迹。Safety Value 在当前批的 PPO 更新**之后**拟合，因此构造本批 Actor 优势时使用的 Value 权重不会在该批 PPO epoch 中改变。[影子采样器](../../utilities/nod_marl/safety_value.py)；[更新顺序](../../utilities/mappo_cavs.py)。

### 6.4 两阶段训练与每批更新时间线

当前训练分两阶段，以避免正在学习的意见实时改变用于收集自身训练数据的行为策略。

1. **候选意见阶段：**冻结 Actor、Safety Value 及用于采样的 behavior NOD；另一份参数独立的 learner NOD 使用已完成的轨迹更新。未来责任标签仅在这一阶段的监督计算中出现，不进入在线 Actor 决策。候选模型通过配置 `nod_training_mode="candidate_only"` 实现。[训练入口](../../utilities/mappo_cavs.py)；[NOD 管理器](../../utilities/nod_marl/trainer.py)。
2. **控制微调阶段：**加载候选 NOD 并冻结其权重，同时允许其在线隐状态与意见继续随观察演化；训练 Actor、任务 Critic 和 Safety Value，开启意见调节的成对屏障优势。配置位于 [`configs/current/config_dgppo_nod_gain1_control_finetune.json`](../../configs/current/config_dgppo_nod_gain1_control_finetune.json)。

这两阶段有助于明确哪些参数在何时更新，却不构成对意见因果贡献的数学证明。尤其不能在已训练 Actor 上仅切换测试时的 `dgppo_opinion_alpha`，因为 \(\alpha\) 的作用发生在训练优势构造阶段。[现行方法说明](DGPPO_NOD_RESPONSIBILITY.md)。

控制微调阶段的单批顺序为：① 用当前 Actor 与冻结 NOD 收集 4096 帧随机策略数据，同时从独立影子环境收集 Safety Value 数据；② 计算任务 GAE，用**尚未更新的** Safety Value 和动作前缓存意见产生屏障优势；③ 在前 20 个 Value 拟合批期间冻结 Actor，之后对随机策略批做最多 60 个 PPO epoch、每个小批量 512，并使用目标 KL 0.01 触发提前停止；④ 在 PPO 之后用影子数据拟合 Safety Value；⑤ 为下一批同步策略权重。控制阶段 NOD 权重始终冻结，而 GRU 隐状态及 \(z\) 在每次实际采样时继续演化。旧式独立 Safety Critic 和 Deadlock Critic 在本配置中均关闭。[微调门控](../../utilities/nod_marl/finetune.py)；[训练主循环](../../utilities/mappo_cavs.py)。

### 6.5 损失与被更新参数的对应关系

| 阶段与目标 | 实际更新的模块 | 不更新的模块或作用边界 |
| --- | --- | --- |
| 候选阶段 \(\mathcal L_{\mathrm{NLL}}+\mathcal L_{\mathrm{BCE}}\) | learner NOD 的 GRU、有效方差头和风险权重 | behavior NOD、Actor、任务 Critic、Safety Value 固定；Brier 只记录 |
| 控制阶段 PPO 截断目标 + 任务价值 Smooth L1 + 熵项 | Actor MLP、消息聚合器、任务 Critic | 冻结 NOD；屏障量只改变已缓存的 Actor 优势 |
| 控制阶段 \(\mathcal L_V\) | 独立的 `PairSafetyValue` | 不与 PPO 共用优化器；不向 Actor 反传 |
| 意见映射 \(z\mapsto\alpha\) 与屏障门控 | 无独立网络或损失 | 当前 \(\alpha_0,\Delta\alpha,k,z_{\rm dead}\) 为配置超参数 |

当前 `ppo_training_profile="original"` 保留原始任务 GAE 的尺度；构造意见屏障优势后也不再对混合优势重新标准化。这样的梯度隔离使责任监督、Safety Value 回归与 PPO 更新有清楚的参数归属，但整体仍是交替优化，而非单一端到端可微目标。[优势构造](../../utilities/nod_marl/dgppo.py)；[优化器配置](../../utilities/mappo_cavs.py)。

## 7. 可证明的性质与证明思路

下述命题针对第 3—5 节定义的数学对象。分析意见或系数的单调性时，未被显式改变的状态、其他车辆动作、模型参数和有效掩码均保持固定。精确优化解与代码中有限次二分求根的数值近似也应区分。

若以后提炼为论文，最能支撑方法主线的是**责任指标的可解释次序、KL 更新的唯一性与证据单调性，以及责任证据到成对训练违规量的局部方向链**。风险关注单调性、邻居重排不变性属于有用的结构性质；理想屏障定理则必须与其额外假设同时陈述。

### 7.1 责任指标的边界、方向和敏感性

**命题 1（固定冲突基准下的责任次序）。** 设当前冲突有效，\(R^0>R_s\)，记 \(D=R^0-R_s>0\)。对固定的 \(R^0\)，

\[
q(R^{\mathrm{obs}})=\operatorname{clip}\left(
\frac{R^0-R^{\mathrm{obs}}}{D},0,1\right)
\]

满足 \(0\le q\le1\)，并随邻车观测风险 \(R^{\mathrm{obs}}\) 单调不增。若 \(R^{\mathrm{obs}}\ge R^0\)，则 \(q=0\)；若 \(R^{\mathrm{obs}}\le R_s\)，则 \(q=1\)。若还要求 \(D\ge m>0\)，对两次风险估计 \(R_1,R_2\) 有

\[
|q(R_1)-q(R_2)|\le \frac{|R_1-R_2|}{m}.
\]

**证明思路：**分式对 \(R^{\mathrm{obs}}\) 的斜率为 \(-1/D\)，截断到 \([0,1]\) 是单调且 1-Lipschitz 的操作。该界揭示一个实际弱点：代码仅用 \(10^{-6}\) 防止分母为零，没有强制具有意义的冲突余量 \(m\)。临近风险阈值时，上式可能给出非常大的误差放大系数；论文不能声称 \(q\) 对几何测量误差天然鲁棒。[责任计算](../../utilities/nod_marl/counterfactual.py)。

**命题 2（固定本车反事实下的直接归因隔离）。** 给定当前状态、两车速度锚点和邻车未来位置，未来标签 \(q^*_{ij,t}\) 不依赖本车的真实未来位置 \(p_{i,t+h}\)：标签的两种风险均使用 \(\hat p_i(h)=p_{i,t}+h\Delta t\,v_i^0\)。因此，在这些量固定时，改变本车真实未来刹车轨迹不会直接改变 \(q^*_{ij,t}\)。

这是一条**公式依赖关系**，不是独立识别邻车行为原因的因果定理。本车在冲突开始前的动作仍会影响锚点和当前相对位置，本车后续行为也可能间接诱发邻车改变动作。[未来标签](../../utilities/nod_marl/counterfactual.py)。

### 7.2 意见更新的唯一性、零点和证据响应

**命题 3（标量意见更新的唯一性）。** 对固定的 \(z_{t-1},A_t,e_t,\sigma_t^2\)，假设 \(0<A_t<1\)、\(w\ge0\)、\(\sigma_t^2>0\)、\(a>0\) 且 \(\lambda+1>a\)。则第 4 节的目标 \(J_t(z)\) 在 \((-1,1)\) 上严格凸，在实现采用的闭合裁剪区间上存在唯一最小点，因为

\[
J_t''(z)
=\frac{\lambda}{1-z^2}+1-A_ta\operatorname{sech}^2(az)
+\frac{wA_t}{\sigma_t^2}
\ge\lambda+1-a>0.
\]

当前 \(\lambda=5,a=2\)，所以该曲率下界为 4。求根代码在区间边界可直接选择边界解；48 次二分迭代给出数值近似，严格凸性本身并不等于浮点求解零误差。[目标与求根](../../utilities/nod_marl/opinion.py)。

**命题 4（证据次序与零点）。** 在命题 3 的条件下进一步令 \(w>0\)。对内部解及固定的 \(A_t,\sigma_t^2,z_{t-1}\)，

\[
\frac{\partial z_t}{\partial e_t}
=\frac{wA_t/\sigma_t^2}{J_t''(z_t)}>0.
\]

由此，同一前态下更正面的责任证据对应更大的意见。若 \(z_{t-1}=0,e_t=0\)，唯一解为 \(z_t=0\)。若 \(z_{t-1}\ge0,e_t\ge0\)，则 \(z_t\ge0\)；两者均非正时相应地 \(z_t\le0\)。后一同号性质可由一阶条件左边在 \(z=0\) 的符号及其严格单调性得到。它表示**持续同号证据不会凭空产生相反意见**，并不表示遇到一次反向证据就立即翻转已有意见。

此外，若 \(E=wA_t/\sigma_t^2\)，内部解的单步证据灵敏度满足

\[
0<\frac{\partial z_t}{\partial e_t}
\le\frac{E}{\lambda+1-a+E}<1.
\]

这是固定 \(A_t,\sigma_t^2\) 的**局部证据响应界**，不是对整个交互闭环的指数稳定性结论。真实运动同时改变证据、关注强度、方差和历史；新出现的边还会先按实现重置为 \(z=0\)，随后才使用该更新式。[意见状态机](../../utilities/nod_marl/opinion.py)。

### 7.3 风险关注与邻居消息的结构性质

**命题 5（风险分量的单调关注）。** 风险温度 \(T>0\)、权重 \(w_k\ge0\) 时，固定其他分量，

\[
\frac{\partial A}{\partial r_k}=T\,A(1-A)w_k\ge0.
\]

因此将某个已定义的风险分量增大，不会降低风险关注 \(A\)。这只描述网络的参数化，不保证所有原始几何量变化都使全部风险分量同向变化；当前局部运动学模式还有三个路径相关分量恒为零。[风险关注实现](../../utilities/nod_marl/opinion.py)。

**命题 6（邻居重排不改变聚合消息）。** 设每条有效边的上下文为 \(c_j\)，消息为 \(f(c_j)\)，得分为 \(s(c_j)\)，聚合器输出

\[
m=0.1\tanh\left(
\sum_{j\in\mathcal N_i}
\frac{\exp s(c_j)}{\sum_{\ell\in\mathcal N_i}\exp s(c_\ell)}
f(c_j)\right).
\]

同时对上下文和有效掩码做任意同一置换，只会置换 softmax 权重与各边消息，求和后的 \(m\) 不变；无有效边时 \(m=0\)。此外，每个消息坐标均满足 \(|m_k|\le0.1\)。这是**邻居槽位顺序的不变性**，并不能推出 Actor 动作对责任意见单调，也不能在身份缓存错误时保证语义正确。[消息聚合实现](../../utilities/nod_marl/policy.py)。

### 7.4 意见调节方向及无交互回退

**命题 7（固定安全状态下的屏障次序）。** 固定当前和后继 Value、物理约束与有效掩码，并设该成对头满足 \(g_{ij,t}\le0,V_{ij,t}<0\)。当前配置 \(k>0,\Delta\alpha>0\) 下，含死区和饱和的 \(\alpha_{ij}(z)\) 随 \(z\) 非递减，所以

\[
z_1\le z_2\quad\Longrightarrow\quad
\delta_{ij}(z_1)\ge\delta_{ij}(z_2).
\]

在死区外、未饱和的内部区域，\(\partial\delta_{ij}/\partial z=\Delta\alpha\,kV_{ij,t}<0\)。证明只需注意当前增益 \(k>0\) 下的死区与饱和映射 \(\operatorname{clip}(kz,-1,1)\mathbf 1\{|z|>z_{\rm dead}\}\) 非递减，且 \(\delta\) 对 \(\alpha\) 的斜率为负的 \(V_{ij,t}\)。当意见无效、不足死区、当前 \(g>0\) 或 \(V\ge0\) 时，代码使用基础 \(\alpha_0\)。道路和实际碰撞头不接受成对意见，因此**不会因某个邻车正意见直接放宽这些头**。[系数门控](../../utilities/nod_marl/dgppo.py)；[意见身份对齐](../../utilities/nod_marl/barrier.py)。

**推论（责任证据到训练违规量的局部次序）。** 进一步固定 \(R^0,A_t,\sigma_t^2,V_t,V_{t+1}\)，并假设责任比例、意见求解、增益映射均处于内部，\(D=R^0-R_s>0\)。把邻车风险 \(R^{\mathrm{obs}}\) 当作此局部比较中唯一变化的量，则链式法则给出

\[
\frac{\partial\delta_{ij}}{\partial R^{\mathrm{obs}}}
=-\frac{2\Delta\alpha\,kV_{ij,t}}{D}
\cdot\frac{wA_t/\sigma_t^2}{J_t''(z_t)}>0.
\]

所以，在这一严格限定的比较中，邻车降低其预测接近风险会降低本车该成对头的训练违规量。该推论把 \(R^{\mathrm{obs}}\to q\to e\to z\to\alpha\to\delta\) 连成完整方向链，但**不证明**改变真实邻车动作时所有中间量仍可视为固定，也不证明最终策略动作或碰撞率随之单调。

在同一固定转移上，成对违规指示与该头的正部惩罚也随正意见非递增；若任务优势非负，则当前 gated 规则形成的混合优势非递减。不过这一结论不推及任务优势为负的样本、参数更新后的新转移，也不推及最终 Actor 动作的方向。其他约束头的最大惩罚可能完全遮蔽该成对头的变化。[混合优势](../../utilities/nod_marl/dgppo.py)。

### 7.5 理想屏障下的条件安全定理

**命题 8（附加假设下的物理约束保持）。** 假设所有可达状态和有效约束头满足 \(g_c(s)\le V_c(s)\)，初始 \(V_c(s_0)\le0\)，执行策略在每个真实转移上**精确**满足

\[
\frac{V_c(s_{t+1})-V_c(s_t)}{\Delta t}
+\alpha_{c,t}V_c(s_t)\le0,
\qquad 0<\alpha_{c,t}\Delta t\le1.
\]

则对所有这样的转移，有 \(V_c(s_t)\le0\)，进而 \(g_c(s_t)\le0\)。证明由 \(V_{t+1}\le(1-\alpha_t\Delta t)V_t\le0\) 做归纳，再用上界关系 \(g_t\le V_t\)。在这些额外假设下，改变 \(\alpha\) 只改变允许接近 \(V=0\) 的速率，不移动物理约束 \(g=0\) 的边界。

**适用边界必须紧跟定理。** 当前 Safety Value 未被证明始终满足 \(g\le V\)；PPO 也没有对执行动作逐次强制屏障不等式。因此命题 8 是设计动机的条件定理，**不是当前神经策略的无碰撞保证**。原始 DGPPO 研究的是离散图 CBF 与策略共同学习；本项目使用的是本地成对 Safety Value 和 DGPPO 风格的优势更新，应明确区别。[DGPPO 原论文](https://arxiv.org/abs/2502.03640)；[当前适配实现](../../utilities/nod_marl/dgppo.py)。

### 7.6 信息依赖与证明边界

在线证据只访问当前已观测运动和缓存锚点；意见状态仅由历史与当步输入递推；未来轨迹只用于离线标签。故在“相关位置与速度可由本地传感获得，且可见性掩码正确”的前提下，可对时间步归纳证明执行动作不依赖未来轨迹或邻车发送的内部计划。代码在仿真中使用世界坐标张量进行计算，这种**信息流性质**不自动证明真实传感系统具备相同可观测性。[在线证据](../../utilities/nod_marl/counterfactual.py)；[在线策略输入](../../utilities/nod_marl/policy.py)。

以下问题不能由现有公式直接推出，应在后续理论完善或独立验证中处理：

- **反事实模型误差：**匀速锚点只是比较基准，转向、加减速和邻车对本车的反应均可能使 \(q\) 偏离实际责任。由于分母 \(R^0-R_s\) 在冲突阈值附近很小，\(q\) 对风险估计误差会特别敏感；若希望给出稳健界，需要对该分母设正的冲突余量并分析轨迹预测误差。
- **在线证据与未来标签的差异：**两者采用不同的时间聚合和运动近似，不能在没有附加假设时证明 \(q^{\mathrm{on}}=q^*\) 或意见一定无偏。
- **责任不守恒：**\(q_{ij}\) 是从本车视角估计的邻车风险缓解，不要求 \(q_{ij}+q_{ji}=1\)。本方法尚未实现双方协商的精确距离或控制责任分摊。
- **Value 与真实安全之间的缺口：**Safety Value 使用有限采样、折扣目标和函数近似，未证明是物理约束的保守上界；学得的训练优势也不强制运行时每一步满足屏障不等式。
- **在线动作响应：**训练期 \(z\to\alpha\) 的方向可解析，但部署动作对 \(z\) 的灵敏度、单调性和收益取决于 Actor 学习结果，而非由屏障公式保证。
- **通行活性：**局部责任意见和当前屏障优势都未提供有限时间到达或全局无死锁证明；当前选定配置也关闭了独立 Deadlock Critic。

因此，本文适宜将自身定位为**责任意见调节的分布式安全策略学习方法**，而非具有严格碰撞避免证明的责任分配控制器。
