# Decouple What Travels：PaRD-MAG 新架构深度研究与实验路线

> 面向当前 MAP 项目的多模态属性图节点分类；研究日期：2026-09-08  
> 建议模型名：**PaRD-MAG**（**P**ropagation-**a**ware **R**esidual **D**ecoupling for Multimodal Attributed Graphs）  
> 建议论文标题：**Decouple What Travels: Propagation-Aware Residual Decoupling for Multimodal Attributed Graphs**

## 1. 直接结论

最值得做的不是“把 DecAlign 的 OT、GMM、MMD 和 Transformer 搬到 MAP-MAG”，也不是再造一个 v3 式双模态传播器，而是提出一个更贴合现有证据的问题：

> **跨模态共享性不等于图可传播性；模态私有信息也不等于噪声。模型应先识别什么信息适合沿图传播，再分别处理可传播共识与不可传播创新，最后依据节点级的反事实传播收益重组。**

PaRD-MAG 由三部分构成：

1. **Stable Diffusion Anchor（稳定扩散锚）**：完整保留已经被严格消融验证的 v2 `semantic_uniform` 主干；
2. **Propagation-aware Residual Decoupling（传播感知残差解耦）**：将每种模态拆成 shared/common 与 private/innovation，只有 shared 分量进入图扩散，private 分量留在节点本地；
3. **Counterfactual Utility Recomposition（反事实效用重组）**：比较“稳定锚、选择性传播、本地属性”三个候选在训练节点上的反事实损失，用显式目标监督节点路由，而不是让 gate 仅靠最终 CE 自发学会可靠性。

这个设计的关键不是模块数量，而是顺序和职责清楚：

```text
投影 -> 解耦“可传播共识 / 本地私有创新”
     -> 共识做两跳 restart low-pass，私有只做本地变换
     -> 用反事实传播效用选择修正幅度
     -> residual MLP + LayerNorm 判别整形
```

预期收益点非常具体：Movies/Toys/Grocery/Reddit-S 由 v2 锚路径兜底；ele-fashion 通过保留文本主导的本地私有信息与传播 bypass，修复“所有度数段 low-pass 都掉点”的反例。

### 研究范围与证据口径

本报告核查了项目内 benchmark、v2 重训练消融/冻结反事实/表示诊断、v2/v3 实现与统一 NC trainer；外部检索截至 2026-09-08，优先使用论文原文、会议页面和官方代码。检索覆盖 DecAlign、shared/private 表示学习、restart diffusion，以及 2026 年最接近的 MAG 方法 CoMAG、NSG-MoE、GraphMNL 和 DiP。研究在核心机制、主要新颖性威胁和工程约束均有一手证据，且继续扩展关键词只返回同类方法时停止。架构性能仍属于待实验验证假设，文中没有把预期提升写成既成事实。

## 2. 从现有实验真正应该继承什么

### 2.1 已证实的成功经验

当前五数据集主表中，`map_mag_v2` 的跨数据集平均为 **80.5091 Acc / 74.0399 Macro-F1**，v3 为 **80.3260 / 73.9737**；二者非常接近。更重要的是，v2 的受控诊断显示：

- `lowpass_uniform - core_no_graph` 平均为 **+3.21 Acc / +4.49 F1**，是唯一大幅且稳定的独立增益；
- `semantic_uniform` 达到 **80.70 / 74.45**，高于默认 core 的 80.65/74.06，是当前最好的“最小充分结构”；
- reliability 的独立贡献为 **+0.03/-0.06**，与 semantic edge 同时使用时还有负交互；
- prototype residual 的实际修正范数仅为 `z_low` 的约 **0.8%–1.0%**；
- v1 gamma、self residual、high-pass residual 都呈强数据集条件性，且 gate 经常贴近上界，退化成固定比例修正；
- 输出端 `residual MLP + LayerNorm` 对后验线性可分性贡献很大，暂时不能删除。

因此新架构的性能底座应固定为：

```text
双模态投影 -> uniform fusion -> semantic edge
-> 2-hop restart diffusion -> residual output MLP + LayerNorm
```

本地证据见 [NC benchmark](./nc_benchmark_results.md) 与 [v2 performance diagnosis](./map_mag_v2_performance_diagnosis.md)。

### 2.2 必须解释的反例

ele-fashion 的 attribute-only 已达 87.89 Acc；纯 low-pass 反而降低 0.64 Acc / 1.13 F1，而且每个度数分箱都下降。它说明：

- 高 homophily 不自动等于传播有益；
- degree 也不是充分的 route 信号；
- 一个全局固定的 low/high 或 self residual 比例不能解决问题；
- “是否传播”应成为可监督、可校准的节点级决策。

这正是 PaRD-MAG 的问题起点，而不是事后为某个数据集增加特例。

## 3. DecAlign 带来的原则，以及不应照搬的部分

DecAlign 先用 shared encoder 与 modality-specific encoder 将表示拆成 common/unique 两部分，再对 unique 特征使用 GMM prototype 与多边际 OT，对 common 特征使用分布矩、PDE 和 MMD 对齐，最后再融合。其最有价值的启发是：**先解耦，再针对不同语义成分采用不同处理规则**。[DecAlign 原文](https://arxiv.org/html/2503.11892v3) 和 [官方代码](https://github.com/taco-group/DecAlign) 均支持这一点。

但直接移植不合适：

- DecAlign 面向带时间 token 的视频情感任务，不包含真实节点拓扑，也没有回答“哪一部分应该沿图传播”；
- 其 GMM prototype 数设为类别数，OT 与 sample-to-prototype 校准建立在 mini-batch 表示上；当前 MAP-MAG 的 prototype 路径已被实验证明几乎没有形成有效修正；
- 官方实现的核 MMD 构造样本两两距离。对 ele-fashion 的 97,766 个节点做全图二次复杂度 MMD 不现实；
- DecAlign 自己的敏感性实验也显示对齐权重过大会明显损害性能，说明“越对齐越好”不成立。

所以应继承它的**职责分离原则**，但将具体机制改造成图任务专用的“传播感知解耦 + 线性稀疏处理”。

## 4. PaRD-MAG 的核心假设

对节点 $i$ 的模态 $m\in\{t,v\}$，投影表示 $h_i^m$ 同时包含三类东西：

1. 跨模态共同且图邻域可复用的类别语义；
2. 模态私有但对类别有用的本地创新；
3. 模态噪声或与当前拓扑不匹配的信息。

v2 将三者先融合再统一低通；v3 将每个模态的三者分别传播后再融合。两者都没有在传播前显式区分“应该传播的成分”和“应该留在节点本地的成分”。PaRD-MAG 的新假设是：

> 图扩散的对象不应是完整模态表示，而应是经过跨模态一致性与信息保真约束提取的 propagation-ready common factor；私有残差只在其反事实效用为正时注入最终表示。

这是论文最应坚持的窄而清晰的 claim。不要写成泛泛的“首次 shared-private multimodal graph learning”。截至 2026-09，CoMAG 已经包含传播后 shared/private 解耦，NSG-MoE 已经显式拆分模态节点；宽泛 claim 很容易被审稿人否定。

## 5. 模型架构

### 5.1 Modality projection

完全沿用 v2 的双投影：

\[
h_i^t=P_t(x_i^t),\qquad h_i^v=P_v(x_i^v),\qquad
h_i^0=\tfrac12(h_i^t+h_i^v).
\]

不在这里引入 reliability gate。现有实验已经表明，从头训练时 projector 足以吸收数据集级模态比例，节点 reliability 若无独立目标只是重参数化。

### 5.2 Stable Diffusion Anchor

锚路径严格复现 v2 的 `semantic_uniform`：

\[
s_{ij}^{0}=\tfrac12[\cos(h_i^t,h_j^t)+\cos(h_i^v,h_j^v)],
\]

\[
w_{ij}^{0}=0.1+0.9\,\sigma(s_{ij}^{0}/2),
\]

\[
H^{(0)}=h^0,\qquad
H^{(k+1)}=0.85\hat A_{w^0}H^{(k)}+0.15h^0,\qquad k=0,1,
\]

并记 $z_i^0=H_i^{(2)}$。它不是“普通 baseline”，而是新模型中的稳定锚。PaRD-MAG 的新增部分只学习相对它的残差修正。

### 5.3 Propagation-aware shared/private decoupler

对每个模态建立 common encoder $C_m$、private encoder $U_m$ 与轻量 decoder $D_m$：

\[
c_i^m=C_m(h_i^m),\qquad p_i^m=U_m(h_i^m),
\]

\[
\hat h_i^m=D_m([c_i^m\Vert p_i^m]).
\]

设计上采用残差初始化：$C_m$ 初始化为近似恒等映射，$U_m$ 的末层零初始化。因此训练开始时 common 近似原投影、private 近似 0，不会一开始就破坏 v2 表示。

解耦由三种约束承担：

\[
\mathcal L_{com}=\operatorname{VICReg}(c^t,c^v),
\]

\[
\mathcal L_{rec}=\sum_m\|\hat h^m-h^m\|_2^2,
\]

\[
\mathcal L_{orth}=\sum_m\|\operatorname{Cov}(c^m,p^m)\|_F^2.
\]

这里使用 VICReg 的 invariance/variance/covariance 三项，而不是单独 cosine/MSE。原因是只拉近 paired common features 容易坍塌；VICReg 明确用方差和协方差正则避免常数解。[VICReg](https://openreview.net/forum?id=xm6YD62D1Ub) 提供了这一依据。

工程上不应在 97k 节点上每轮计算完整 $d\times d$ 统计。建议每轮固定随机抽取 4,096–8,192 个节点，或先投影到 64 维再计算统计；抽样只用于无监督辅助损失，主 CE 与图传播仍为全图。

### 5.4 Selective common diffusion and local private refinement

common 共识为：

\[
c_i=\tfrac12(c_i^t+c_i^v).
\]

仅在原始稀疏边上由 common 表示产生边权 $w^c$，并沿用 v2 的两跳 restart diffusion：

\[
z^c=\operatorname{RDiff}_2(c,A,w^c;\alpha=0.15).
\]

private 信息不做邻域低通，而经本地创新网络：

\[
p_i=F_p([p_i^t\Vert p_i^v\Vert |p_i^t-p_i^v|\Vert p_i^t\odot p_i^v]).
\]

得到“只传播 common、保留 local private”的选择性传播候选，以及完全不传播的属性候选：

\[
z_i^g=z_i^c+p_i,\qquad z_i^a=c_i+p_i.
\]

这一步是“解耦后再处理”的图版本：common 被当作可传播语义，private 被当作本地创新；二者不走相同算子。

边权增强可以作为第二阶段可选项，而不应进入首个 core。若要修复 v2 边权 CV 仅 1.9%–2.7% 的问题，可使用均值保持的残差锐化：

\[
w_{ij}^c=w_{ij}^{v2}\exp(\kappa\,\operatorname{clip}((s_{ij}^c-\mu_s)/\sigma_s,-3,3)),
\]

其中 $\kappa$ 从 0 初始化。这样 $\kappa=0$ 时退化为保守 v2 权重，只有验证集支持时才扩大动态范围。不要一开始 hard-prune 或全图 KNN rewiring；Movies 的低同配图仍能从原拓扑低通获益，贸然删边可能破坏这部分结构统计。

### 5.5 Counterfactual Utility Recomposition

最终表示不是无监督 sigmoid gate 随意混合，而是在三个有明确语义的候选之间重组。令 router 产生 $\tilde\pi_i=\operatorname{softmax}(g(r_i))$，训练阶段用 continuation 系数 $\alpha_t$ 控制离开 anchor 的速度：

\[
\pi_i=(1-\alpha_t)[1,0,0]+\alpha_t\tilde\pi_i,
\]

\[
z_i^{mix}=\pi_{i,0}z_i^0+\pi_{i,g}z_i^g+\pi_{i,a}z_i^a.
\]

三个候选分别表示：稳定 v2 扩散、仅 common 扩散并保留 private、完整本地属性。$\alpha_t=0$ 时模型严格等于 anchor；随后再逐步开放两个新候选。

router 只读取推理时可获得的证据：

\[
r_i=[\cos(c_i^t,c_i^v),\|p_i^t\|,\|p_i^v\|,
s_i^t,s_i^v,\log(1+d_i),
\|z_i^g-z_i^0\|/\|z_i^0\|,
\|z_i^a-z_i^0\|/\|z_i^0\|].
\]

其中 $s_i^m$ 是节点与一阶邻域的模态内平均 cosine。

关键是为它建立独立的反事实效用目标。用同一个 EMA classifier 分别评估三个候选，在训练节点上计算：

\[
q_i=\operatorname{softmax}\left(
[-\ell_i^0,-\ell_i^g,-\ell_i^a]/\tau_r
\right),
\]

\[
\mathcal L_{route}=\operatorname{KL}
(\operatorname{stopgrad}(q_i)\Vert\tilde\pi_i),
\]

其中 $\ell_i^b=\operatorname{CE}(y_i,\bar f(z_i^b))$，$b\in\{0,g,a\}$。哪个候选的 CE 更小，$q_i$ 就向哪个候选分配更多质量。EMA、stop-gradient 与 warm-up 防止 router 和 classifier 互相投机。标签只在训练节点上产生 route target，验证/测试时 router 仅使用上述无标签统计。

这与旧 gate 的根本区别是：旧 gate 只收到最终 CE 的间接梯度，容易找到“所有节点近似同一比例”的共适应解；新 gate 被直接要求预测“对该节点，离开稳定扩散锚是否真的更好”。

### 5.6 Output shaping

沿用已被诊断证明重要的尾部：

\[
z_i=\operatorname{LayerNorm}(F_{out}(z_i^{mix})+z_i^{mix}).
\]

统一 NC linear head 不变，保证主表差异来自 encoder。

## 6. 完整目标与训练日程

\[
\mathcal L=
\mathcal L_{CE}
+\lambda_{com}\mathcal L_{com}
+\lambda_{rec}\mathcal L_{rec}
+\lambda_{orth}\mathcal L_{orth}
+\lambda_{route}\mathcal L_{route}
+\lambda_{anchor}(t)\mathcal L_{KD}(z,z^0).
\]

建议不是一次端到端硬训，而是 anchored continuation：

1. **Anchor stage**：训练 `semantic_uniform` 到收敛，保存同 seed 最佳 checkpoint；
2. **Decoupling warm-up**：载入 anchor，冻结 anchor 与任务 head，训练 common/private 的 `L_com+L_rec+L_orth`，此时强制 $\alpha_t=0$；
3. **Route calibration**：开放候选 head，先让三个候选都能分类，再基于 EMA counterfactual loss 训练 router；
4. **Joint adaptation**：5–20 epoch 内把 $\alpha_t$ 从 0 线性升到 1，小学习率联合微调；`lambda_anchor` 同期从小值衰减到 0；
5. **Checkpoint pool**：最终选择范围必须包含 Stage 1 anchor checkpoint。因此验证指标至少有一个与 anchor 完全相同的候选，降低新分支导致整体退化的风险。

这不能数学保证 test 指标不降，但它提供了很强的工程非退化机制：模型函数族显式包含稳定 v2 锚，优化过程也保留锚 checkpoint，而不是寄希望于复杂模型自己学会退化回 v2。

首轮建议权重只搜小范围：

| 参数 | 建议候选 |
|---|---|
| `lambda_com` | 0.005, 0.01, 0.02 |
| `lambda_rec` | 0.01, 0.05 |
| `lambda_orth` | 0.001, 0.005 |
| `lambda_route` | 0.05, 0.1 |
| `tau_route` | 0.5, 1.0 |
| private dim | 32, 64（不要直接 256） |
| corrector ramp | 10, 20 epochs |

不要同时搜索 edge sharpening、private dim、全部 loss weight 和传播深度；否则无法知道增益来源。

## 7. 与最近工作的创新边界

| 方法 | 何时解耦 | 图处理对象 | private 是否服务 NC | 路由监督 | 与 PaRD-MAG 的本质区别 |
|---|---|---|---|---|---|
| MAP-MAG v2 | 不显式解耦 | 早期融合表示 | 隐式 | 无 | PaRD 在其稳定主干上增加“传播对象分解” |
| MAP-MAG v3 | 不显式 shared/private | 完整文本、完整视觉各自传播 | 隐式 | 无 | v3 是“模态先传播后融合”；PaRD 是“语义成分先解耦再选择性传播” |
| DecAlign | 传播概念不存在；先 common/unique 解耦 | 无真实图拓扑 | 是 | 无 | PaRD 把解耦轴定义为 propagation utility，并以稀疏扩散处理 common |
| CoMAG | 模态多跳轨迹之后解耦 | task-adaptive context 上的完整模态轨迹 | graph output 只用 shared，private 主要服务模态任务 | task gate，无节点反事实收益目标 | PaRD 在传播前解耦，private 可经校准修正直接服务 NC |
| NSG-MoE | 按模态拆成异构子节点 | rewired typed graph + experts | 是 | MoE gate | PaRD 不复制节点、不建关系类型、不依赖 MoE |
| GraphMNL | 不做表示 shared/private 解耦 | 多预测分支 | 是 | 分支可靠性与负类迁移 | PaRD 预测的是“传播处理的反事实增益”，在表示层重组，不做教师负类蒸馏 |

相关原始工作：[CoMAG](https://arxiv.org/html/2606.14172)、[NSG-MoE](https://arxiv.org/html/2602.00067)、[GraphMNL](https://arxiv.org/html/2606.12863)、[DiP](https://arxiv.org/abs/2603.09258)、[MISA](https://arxiv.org/abs/2005.03545)、[Disentangled Multiplex Graph Representation Learning](https://proceedings.mlr.press/v202/mo23a.html)。

最安全的 novelty 话术是：

> Existing MAG models decide **where or how modalities propagate**; PaRD-MAG instead decides **what semantic component is allowed to propagate**, and calibrates its post-propagation recomposition with node-wise counterfactual utility.

不要声称“首次多模态图解耦”“首次 private/shared”或“首次自适应传播”。

## 8. 论文故事线

### 8.1 Introduction 的四段逻辑

1. MAG 同时包含 attribute knowledge 与 topology knowledge；现有方法主要研究融合和消息传递。[MAGB](https://arxiv.org/abs/2410.09132) 给出了这一任务背景。
2. 现有方法通常把“模态表示”作为最小传播单位：early fusion 后统一传播，或每个 modality 整体传播。但一个模态内部仍混合 shared、private 与 topology-incompatible 成分。
3. 现有 MAP-MAG 诊断提供反直觉证据：浅层 restart diffusion 是绝大多数性能来源，但 ele-fashion 在所有度数段都被传播伤害；复杂无监督 gate 又退化为近常数。这说明问题不是“要不要图”，而是“什么应传播、何时应采用处理结果”。
4. 提出 PaRD-MAG：稳定锚、传播感知解耦、反事实效用重组。强调它既保留图低通的正则化/邻域收益，又保存模态私有判别信息。

### 8.2 三条贡献建议

1. **Problem reframing**：首次将 MAG 的多模态融合问题表述为 component-wise propagation utility，而非 modality-wise routing；
2. **Method**：提出 anchored propagation-aware residual decoupling，将 shared/common 与 private/innovation 分别交给 restart diffusion 与 local refinement；
3. **Optimization and evidence**：提出 counterfactual utility target 校准节点重组，并用跨数据集、度数、噪声/缺失模态与 gate calibration 分析验证“传播对象选择”假设。

第一条中的“首次”仍需在投稿前再做一次系统检索；当前检索只能支持“具有明显差异化”，不能保证不存在未公开或刚发布的相似工作。

## 9. 最关键的消融设计

主表之外，必须按因果链做消融，而不是只删模块：

| ID | 模型 | 回答的问题 |
|---|---|---|
| A0 | v2 default core | 与已发表/现有结果对齐 |
| A1 | v2 `semantic_uniform` anchor | 最小充分稳定底座 |
| A2 | A1 + 参数量匹配 MLP | 排除单纯容量增益 |
| A3 | decouple，但 common/private 都 low-pass | 证明收益来自“异质处理”，不是仅解耦 |
| A4 | decouple，common low-pass、private local，固定只用 $z^g$ | 选择性传播本身的效果 |
| A5 | A4 + 普通 CE-only 三路 gate | 复现旧 gate 共适应问题 |
| A6 | A4 + counterfactual route | 校准目标的独立贡献 |
| A7 | A6 去 `L_com` / 去 `L_rec` / 去 `L_orth` | 解耦约束分别做什么 |
| A8 | A6 使用 MMD 或 cosine-only alignment | 证明 VICReg/防坍塌选择合理 |
| A9 | A6 + edge sharpening | 边动态范围是否还有额外价值 |

还应加入顺序反事实：

- **Decouple → Propagate → Recompose**（PaRD）；
- Propagate each modality → Decouple（CoMAG-style simplified）；
- Propagate each modality → Fuse（v3）；
- Fuse → Propagate（v2）。

这张顺序表会比堆很多 loss ablation 更能支撑论文主张。

## 10. 诊断指标与成功判据

### 10.1 下游非退化标准

正式 3 seeds 前先跑 Movies 与 ele-fashion，它们分别代表“强传播收益”和“传播反例”。建议 go/no-go：

- Movies 相对 A1 不下降超过 0.2 Acc / 0.3 F1；
- ele-fashion 至少回收 A1 相对 attribute-only 的损失，并争取达到 88.0+ Acc、77.5+ F1；
- 五数据集宏平均不低于 A1，且至少 3/5 数据集方向为正；
- 最终目标至少达到或超过当前 v2 的 80.51/74.04，并以 A1 的 80.70/74.45 作为更严格目标。

### 10.2 机制必须“真的工作”

- 三路 $\pi$ 的节点内标准差不能再次接近 0；
- $\pi_g-\pi_0$、$\pi_a-\pi_0$ 应分别与 held-out train-node 的对应 counterfactual gain 显著正相关；
- 按 $\pi_a$ 分位数统计时，高分位组应确实更偏向 attribute candidate，且 ele-fashion 的质量应高于 Movies；
- private correction norm / anchor norm 建议落在 5%–20% 的可见区间，避免 prototype 路径的 <1% 无效状态，也避免旧 self residual 的 20%–40% 固定强注入；
- common 的跨模态 CKA/cosine 提升时，private 的有效秩和单模态 label probe 不应坍塌；
- edge sharpening 若启用，应同时报告 edge CV、weighted homophily、same/different-edge AUC，以及 shuffle/invert 反事实敏感性。

### 10.3 鲁棒性实验

DecAlign 式解耦如果只在完整干净模态上有用，故事不够强。建议补：

- test-time text/image mask；
- 10%、30%、50% 节点模态缺失；
- 高斯噪声、feature shuffle；
- 10%、30% edge deletion/addition；
- 类别分层与 degree 分层。

这也为 common/private 提供了独立验证：shared 应在缺失模态下稳定，private 应在完整模态时补充细粒度类别信息。

## 11. 最小可行实现路线

第一轮不要实现完整论文版。按以下顺序最省实验预算：

1. 从 `map_mag_v2.py` 复制出 `pard_mag.py`，固定 anchor 为 `semantic_uniform`；
2. 加两个 common residual MLP、两个低维 private MLP、两个 reconstruction decoder；
3. common 使用第二套两跳 diffusion；private 只过 local MLP；
4. 先用全局三路 simplex 权重，验证 selective processing 与 attribute bypass 是否有信号；
5. 再改成节点 router，并在 trainer 中加入 detached counterfactual target；
6. 先跑 Movies/ele-fashion seed 42；通过后跑 Grocery/Reddit-S，再补 Toys；
7. 最后才加入 calibrated edge sharpening 和 modality corruption。

需要改动的主要位置预计为：

- `src/models/pard_mag.py`：encoder、anchor/corrector、aux exports；
- `configs/model/pard_mag.yaml`；
- `src/tasks/nc.py`：候选表示的 counterfactual route loss 与 EMA head；
- `src/tasks/common.py`、`src/tasks/analysis.py`：新增标量与节点诊断；
- `tests/test_pard_mag.py`：退化等价、空图、数值稳定、反向、inference equivalence。

最重要的单元测试是：关闭 corrector 或令 $\alpha_t=0$ 时，PaRD 的 encoder 输出与 `map_mag_v2 + uniform fusion` 在相同 state 下数值等价。

## 12. 风险与预案

### 风险 1：common alignment 伤害强单模态信号

ele-fashion 的文本明显强于视觉，硬对齐可能把文本拉向弱视觉。预案是低维 private、弱 `lambda_com`、modality dropout、VICReg 防坍塌，并让 anchor 始终可回退。必要时 common loss 只对两模态均高置信节点生效。

### 风险 2：router 再次饱和

饱和本身不是问题；无语义依据地全体饱和才是问题。反事实 KL、EMA teacher、候选辅助 CE、按分位数校准曲线共同判断。若仍退化，先用全局三路 simplex 验证分支价值，再检查 route target 是否噪声过大。

### 风险 3：模型只靠额外参数获益

必须有 A2 参数量匹配 MLP，并报告 FLOPs、峰值显存、训练时间。PaRD 是两套稀疏两跳扩散，复杂度仍为 $O(K|E|d)$，但常数约为 v2 的两倍；不能只说“线性复杂度”而回避常数。

### 风险 4：与 CoMAG 叙事重合

全文始终强调三个差别：**pre-propagation component decoupling、private-for-NC、counterfactual recomposition**。不要以“reliable context”“hop-token alignment”“task-adaptive graph”作为主卖点。

### 风险 5：只在 ele-fashion 提升

若结果是 Movies/Toys/Grocery 小降、ele-fashion 大涨，不应硬包装通用 SOTA。可缩小 claim 为 propagation-negative-transfer mitigation，并以 anchor checkpoint/route calibration 争取其他数据集零退化。

## 13. 最终建议

优先实现 **PaRD-MAG core = semantic_uniform anchor + shared/private decoupler + common-only restart diffusion + local private refinement + counterfactual router**。首版不要上 OT、GMM、prototype、MMD、MoE、KNN rewiring 或 conflict channel。

这套方案与现有 MAP-MAG 的成功经验是连续的：它没有否定 low-pass，而是把 low-pass 从“处理全部多模态信息”升级为“只处理适合传播的共识信息”；它也没有再次把复杂 gate 当作黑盒，而是用可观测的反事实收益定义 gate 应该学习什么。因而它同时具备三点：

- **性能可守**：显式锚定当前最强的 v2 最小结构；
- **创新可讲**：从 modality-wise propagation 转向 component-wise propagation utility；
- **机制可证**：每个模块都有对应的受控消融、校准指标和失败判据。

这比“DecAlign + MAP-MAG 的模块拼装”更像一篇完整论文：问题来自现有实验反例，方法逐项对应问题，训练目标修复已知失败模式，实验又能直接验证核心假设。

## 参考来源

- Chengxuan Qian et al. [DecAlign: Hierarchical Cross-Modal Alignment for Decoupled Multimodal Representation Learning](https://arxiv.org/html/2503.11892v3), ICLR 2026.
- Sirui Zhang et al. [Context-aware Modality-Topology Co-Alignment for Multimodal Attributed Graphs](https://arxiv.org/html/2606.14172), arXiv 2026.
- Yihan Zhang and Ercan E. Kuruoglu. [Modality as Heterogeneity: Node Splitting and Graph Rewiring for Multimodal Graph Learning](https://arxiv.org/html/2602.00067), 2026.
- Zhengyu Wu et al. [Multimodal Graph Negative Learning](https://arxiv.org/html/2606.12863), 2026.
- Xiaobin Hong et al. [Multimodal Graph Representation Learning with Dynamic Information Pathways](https://arxiv.org/abs/2603.09258), AAAI 2026.
- Hao Yan et al. [When Graph Meets Multimodal: Benchmarking on Multimodal Attributed Graphs Learning](https://arxiv.org/abs/2410.09132), 2024.
- Devamanyu Hazarika et al. [MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis](https://arxiv.org/abs/2005.03545), ACM MM 2020.
- Yujie Mo et al. [Disentangled Multiplex Graph Representation Learning](https://proceedings.mlr.press/v202/mo23a.html), ICML 2023.
- Johannes Gasteiger et al. [Predict then Propagate: Graph Neural Networks Meet Personalized PageRank](https://openreview.net/forum?id=H1gL-2A9Ym), ICLR 2019.
- Adrien Bardes et al. [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://openreview.net/forum?id=xm6YD62D1Ub), ICLR 2022.
