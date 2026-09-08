# MAP-MAG v2 性能来源分析与设计经验

## 1. 结论先行

本次结果已经可以较明确地回答“v2 的性能主要从哪里来”：

1. **决定性增益来自带 restart 的两跳 low-pass 图传播主干。** 在保持模态均匀融合、关闭语义边权的严格对照下，`lowpass_uniform - core_no_graph` 的五数据集平均增益为 **+3.21 Acc / +4.49 Macro-F1 个百分点**。默认 `core` 相对 `core_no_graph` 的总增益为 +3.31/+4.45，因此按宏平均的描述性分解，low-pass 主干解释了约 **97% 的 Acc 总增益和几乎全部 F1 总增益**。
2. **默认 v2 的高性能并不依赖复杂模块堆叠。** `semantic_uniform`（均匀模态融合 + 语义边权 + low-pass）反而得到所有变体中最高的宏平均：**80.70 Acc / 74.45 F1**；默认 `core` 为 80.65/74.06；甚至最简单的 `lowpass_uniform` 也有 80.55/74.10。三者非常接近。
3. **语义边权有方向正确但幅度很小的增益。** 单独加入语义边权平均为 **+0.14 Acc / +0.35 F1**，4/5 个数据集方向为正。学习后的边权判别同类边的 AUC 达 0.654–0.935，但边权 CV 只有 1.90%–2.67%，加权同配率仅提高 0.17–0.53 个百分点；当前温度与下界把有用的排序信号压得过平。
4. **reliability gate 没有被证明具有独立性能增益。** 在统一边图上加入 reliability，平均仅 **+0.03 Acc / -0.06 F1**；在语义边图上加入则为 **-0.05/-0.39**。它学到了合理的数据集级模态偏好，但均匀融合模型可以通过投影层自行适配，说明当前 gate 更多是重参数化/共适应，而不是新增了不可替代的信息。
5. **single reliability 的真正价值更像是“避免 double reliability 的伤害”。** `double_reliability - core` 在所有五个数据集 Acc 都不增，平均 **-0.59 Acc / -0.53 F1**。不要对弱模态既用比例归一化、又再次乘可靠性。
6. **high-pass residual、v1 gamma、自残差都不是通用增益项。** 它们在 Movies/Toys/Grocery 多数退化，在 ele-fashion/Reddit-S 多数改善，属于强数据集条件机制。更重要的是，它们的门控大面积贴近上限，近似“固定残差比例”，没有表现出预期的节点级精细路由。
7. **prototype residual 基本没有进入有效计算。** 其修正范数只有 `z_low` 的约 0.8%–1.0%；平均 Acc -0.13，F1 +0.29，没有稳定收益，可继续默认关闭。
8. **ele-fashion 是关键反例。** 它的原始文本特征最有判别力、图较稀疏；low-pass 在每一个度数区间都轻微伤害性能。今后的传播设计应允许属性强节点/数据集绕过图传播，而不能仅根据全局 homophily 决定是否平滑。

一句话概括：**v2 的有效核心是“分别投影两种模态 → 简单融合 → 带自环和 restart 的浅层全图低通 → 残差 MLP + LayerNorm 输出整形”；复杂 gate、细粒度边权和额外路径目前都不是主要性能来源。**

---

## 2. 分析范围与口径

- 数据集：Movies、Toys、Grocery、ele-fashion、Reddit-S。
- 变体：13 个。
- 每个变体/数据集：3 个 seed（42、43、44）。
- 共 65 个训练 job、195 个 checkpoint、195 份诊断；全部成功完成。
- 训练配置没有额外 override，均使用 v2/NC 默认正式配置：`hidden_dim=256`、`num_hops=2`、`restart=0.15`、full-graph training、以 validation accuracy 选择 checkpoint。
- 表中 `均值±标准差` 是 3 seeds 的均值与总体标准差（`ddof=0`）；“宏平均”是五个数据集均值的等权平均。
- 差值全部为百分点，写作 `Acc/F1`。

原始结果：

- `outputs/map_v2_analysis/training_summary.json`
- `outputs/map_v2_analysis/diagnostics_summary.json`
- `outputs/map_v2_analysis/diagnostics_table.csv`
- `outputs/map_v2_analysis/counterfactual_table.csv`
- `outputs/map_v2_analysis/diagnostics/<dataset>/<experiment>/run*/{nodes,edges}.csv`

### 2.1 三类证据必须分开

1. **重训练消融**：模块开关后重新训练，最接近模块的独立增益，是本文主要结论依据。
2. **冻结模型反事实**：在训练完成后临时替换边权/融合/图传播，只说明模型已经依赖什么，不能替代重训练消融。
3. **表示诊断**：线性探针、有效秩、门控分布等用于解释机制，不单独构成因果结论。

---

## 3. 数据集结构与原始模态信号

| 数据集 | 节点数 | 有向边数 | 平均度 | 边同配率 | 文本同/异类边 cosine gap | 视觉 gap | 文本边 AUC | 视觉边 AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 16,672 | 160,802 | 9.65 | 0.447 | 0.0057 | 0.0307 | 0.580 | 0.577 |
| Toys | 20,695 | 113,402 | 5.48 | 0.724 | 0.0027 | 0.0264 | 0.548 | 0.562 |
| Grocery | 17,074 | 142,262 | 8.33 | 0.681 | 0.0043 | 0.0431 | 0.573 | 0.601 |
| ele-fashion | 97,766 | 399,172 | 4.08 | 0.767 | 0.1058 | 0.0524 | 0.676 | 0.612 |
| Reddit-S | 15,894 | 283,080 | 17.81 | 0.959 | 0.0064 | 0.1693 | 0.634 | 0.825 |

这张表解释了后续的两个现象：

- Reddit-S 的图和视觉模态都非常强，传播最安全，视觉路径应占主导。
- ele-fashion 虽然同配率高，但图稀疏且原始文本语义已经很强；“高 homophily”并不自动等价于“继续低通一定有益”。

---

## 4. 实验设计与各变体的含义

### 4.1 总体设计思路

这组实验不是把 13 个名称相近的模型简单横向排名，而是围绕默认 v2 建立三层受控分解：

1. **核心机制分解（前 5 个变体）**：回答图传播、reliability fusion、semantic edge 各自贡献多少，以及两种 gate 是否有交互。
2. **同一机制内部替换（4 个变体）**：回答 double reliability 是否必要，以及语义边权究竟应由文本、视觉还是 reliability-aware 相似度控制。
3. **附加路径诊断（4 个变体）**：回答 high-pass、自属性和 prototype 是否应重新放回低通主干。

“运行全部变体”包含前 5 个核心实验，并在其基础上增加 8 个机制细化实验；不是另一套重复实验。

每一个变体都继续使用同一个 `MAPMAGV2` 类、相同 NC trainer、数据划分、hidden dimension、优化器、训练轮数上限、early stopping、输出 head 和 checkpoint 选择规则。实验只通过 Hydra override 改变目标计算路径。这种设计尽量让性能差异来自机制本身，而不是训练协议或 head 容量。

### 4.2 默认 core 的完整计算链

先把输入拆成文本和视觉特征并分别投影：

```text
h_t = TextProj(x_t)
h_v = VisualProj(x_v)
```

对每个节点计算本模态表示与一阶邻居均值的组合，得到可靠性：

```text
r_t = 0.1 + 0.9 * sigmoid(RelMLP_t([h_t, mean_t, h_t-mean_t, h_t*mean_t]))
r_v = 0.1 + 0.9 * sigmoid(RelMLP_v([h_v, mean_v, h_v-mean_v, h_v*mean_v]))
lambda_t = r_t / (r_t + r_v)
lambda_v = r_v / (r_t + r_v)
h0 = lambda_t*h_t + lambda_v*h_v                 # single reliability
```

在原始拓扑的每条边上计算投影后模态 cosine，默认取均值生成共享语义边权：

```text
sim_ij = (cos(h_ti,h_tj) + cos(h_vi,h_vj)) / 2
w_ij = 0.1 + 0.9 * sigmoid(sim_ij / 2.0)
```

然后用带自环、GCN normalization 和 restart 的两跳传播：

```text
H(0) = h0
H(k+1) = 0.85 * A_sem_norm * H(k) + 0.15 * h0,  k=0,1
z_low = H(2)
```

默认 core 不启用 high-pass/self/prototype，最后输出：

```text
z_final = LayerNorm(OutputMLP(z_low) + z_low)
```

所以默认 core 可以拆成五个功能块：双模态投影、reliability fusion、semantic edge、restart low-pass、输出残差整形。此次 13 个变体主要拆开了中间三个模块和可选路径；投影与输出整形在所有变体中保持不变。

### 4.3 五个核心实验：一个带交互项的阶梯/因子设计

| 变体 | Reliability | 图传播 | 边权 | 实际计算含义 | 主要回答的问题 |
|---|---|---|---|---|---|
| `core_no_graph` | 关闭，`lambda_t=lambda_v=0.5` | 关闭，`num_hops=0` | 关闭 | 双投影均匀融合后，直接进入输出残差 MLP | v2 脚手架内不使用图时能达到多少；它是 attribute-only 参照 |
| `lowpass_uniform` | 关闭，均匀融合 | 两跳 restart low-pass | 全部为 1 | 在未经语义重加权的原始图上传播 | 相对 `core_no_graph` 严格隔离 low-pass bundle 的贡献 |
| `reliability_uniform` | single | 两跳 restart low-pass | 全部为 1 | reliability 融合后在原始图传播 | 相对 `lowpass_uniform` 隔离 reliability 的独立作用 |
| `semantic_uniform` | 关闭，均匀融合 | 两跳 restart low-pass | 默认 `avg_cos` | 均匀模态融合，但用语义边权传播 | 相对 `lowpass_uniform` 隔离 semantic edge 的独立作用 |
| `core` | single | 两跳 restart low-pass | 默认 `avg_cos` | 默认完整 v2 core | 检验 reliability 与 semantic edge 同时存在后的总效果和交互 |

这里的 **uniform graph** 不是完整图、随机图或把拓扑打乱，而是保留原始 `edge_index`，仅令所有原始边 `w_ij=1`。因此 `lowpass_uniform` 测到的是“原始拓扑传播”的价值。

五个核心实验应按下面的成对关系解释：

```text
core_no_graph ──(+ low-pass)──────────────> lowpass_uniform
                                               │
                         + reliability         │ + semantic edge
                                               │
                     reliability_uniform   semantic_uniform
                               └──────── core ────────┘
```

- `lowpass_uniform - core_no_graph`：low-pass 图传播的净增益。
- `reliability_uniform - lowpass_uniform`：在 uniform graph 条件下 reliability 的净增益。
- `semantic_uniform - lowpass_uniform`：在 uniform fusion 条件下 semantic edge 的净增益。
- `core - semantic_uniform`：已有 semantic edge 时，再加 reliability 的条件增益。
- `core - reliability_uniform`：已有 reliability 时，再加 semantic edge 的条件增益。
- `core - reliability_uniform - semantic_uniform + lowpass_uniform`：两种机制的交互项。宏平均交互约为 **-0.07 Acc / -0.33 F1**，表明二者的信息存在一定重叠或优化干扰。

不能只做 `core - core_no_graph` 就宣称差值来自某一个模块，因为这个差值同时包含图传播、reliability 和 semantic edge。

### 4.4 reliability 与边权内部替换实验

| 变体 | 相对 core 的唯一主要改变 | 具体公式/含义 | 应与谁比较 | 诊断目标 |
|---|---|---|---|---|
| `double_reliability` | `reliability_mode=double` | `h0=lambda_t*(r_t*h_t)+lambda_v*(r_v*h_v)` | `core` | v1 风格的第二次可靠性缩放是否有益 |
| `edge_text_only` | `edge_weight_mode=text_only` | `sim_ij=cos_t_ij` | `core` | 边筛选信号是否主要来自文本 |
| `edge_visual_only` | `edge_weight_mode=visual_only` | `sim_ij=cos_v_ij` | `core` | 边筛选信号是否主要来自视觉 |
| `edge_reliability_aware` | `edge_weight_mode=reliability_aware` | 用边两端的 `r_t/r_v` 加权 `cos_t/cos_v` | `core` | 节点可靠性是否也应参与边权构造 |

三个 edge 变体都保留 single reliability、两跳 low-pass 和其余 core 设置，只替换 `sim_ij` 的来源。因此它们回答的是**边权内部哪种相似度更好**，不能用于判断“要不要图传播”；是否需要 semantic edge 应看 `semantic_uniform - lowpass_uniform` 或 `core - reliability_uniform`。

`double_reliability` 也不是“两个 reliability 网络”：它仍是文本/视觉各一个 gate，区别只是可靠性既决定 `lambda`，又再次乘到特征幅值上。

### 4.5 附加结构路径实验

| 变体 | 相对 core 的改变 | 计算形式 | 控制细节 | 诊断目标 |
|---|---|---|---|---|
| `lowpass_residual` | 加有界 high-pass correction | `z=z_low+eta*MLP(h0-z_low)` | `0≤eta≤0.2` | 纯低通是否丢失了需要的小幅高频信息 |
| `original_v1_gamma` | 换成 v1 风格 low/high 竞争 | `z=gamma*z_low+(1-gamma)*(h0-z_low)` | `0.05≤gamma≤0.95` | v1 对称频率混合是否优于 v2 纯低通；仅作退化诊断 |
| `self_residual` | 增加未传播属性路径 | `z_input=z_low+alpha_self*SelfMLP([h_t,h_v])` | `0≤alpha_self≤0.2` | 图传播后是否还需显式保留节点自身属性 |
| `prototype_residual` | 增加共享原型残差 | `z=z_low+0.2*z_proto` | 16 个 prototype，`lambda_proto=0` | 共享全局语义原型是否补充局部图信息 |

四个实验都直接与 `core` 比较，但它们不是一个逐步累加链：每次只打开一种可选路径，避免把多个增强项的交互混进结果。

`prototype_residual` 特意保持 `lambda_proto=0`，即不增加 prototype diversity loss。这样测的是前向原型路径本身，而不是“路径 + 新辅助损失”的混合效果。

### 4.6 训练与统计控制

- 每个 dataset/variant 在一个 job 中连续运行 3 次，实际 seed 为 42、43、44。
- 每次都完整重新初始化并训练模型；不存在用 core checkpoint 直接测试某个消融作为主结果的情况。
- 训练只使用 train split，early stopping 和最佳 checkpoint 选择只看 validation accuracy；test 只在最终评估/诊断时读取。
- 所有变体使用相同 `hidden_dim=256`、dropout、LayerNorm、AdamW、学习率、weight decay、梯度裁剪、最多 300 epochs、patience 30。
- 所有 NC 实验使用 full-graph forward，避免不同变体因邻居采样随机性或采样覆盖范围不同而产生额外混杂。
- 对比以相同 dataset/seed 成对计算，再对数据集取等权宏平均；不能把 15 个结果简单当作独立同分布样本做过强显著性结论。
- 单卡两个 job 并行只改变墙钟时间和显存占用，不改变随机种子、计算图或实验定义；ele-fashion 独占 GPU 是调度约束，不是另一种训练设置。

当前类会实例化大部分可选模块，即使某条路径关闭，对应参数也可能仍出现在参数统计中，但不会获得该路径的梯度。因此大多数变体参数统计相同，有利于控制“参数规模”混杂；它不代表关闭模块后的部署计算量不能进一步降低。

### 4.7 诊断实验是怎么设计的

训练后的诊断分成六组，每组回答不同问题：

| 诊断 | 计算对象 | 指标/导出 | 回答的问题 |
|---|---|---|---|
| 原始数据审计 | 未训练的 `x_t/x_v` 和原图 | 节点/边/度、homophily、同异类 cosine gap、边 AUC | 数据集在训练前具有什么结构和模态先验 |
| 下游与泛化 | 每个最佳 checkpoint | train/val/test Acc、Macro-F1、train-test gap | 增益是否出现在 test，是否伴随过拟合变化 |
| 表示链路 | `h_t,h_v,h0,z_low,z_struct,z_final` | 同设置后验线性探针 | 类别可分性在哪一步产生或损失 |
| 图与边权 | 每条原始边和传播前后表示 | 边权均值/CV/AUC、加权同配率、有害消息质量、Dirichlet energy、有效秩 | semantic edge 是否真的改变消息分配，low-pass 是否真的改变表示 |
| 节点分层 | 每个节点 | degree、正确性、`r/lambda`、局部相似度、表示变化 | 收益集中在哪类节点，gate 是否按节点自适应 |
| 冻结反事实 | 固定已训练参数，仅改推理计算 | no-graph、uniform/shuffle/invert/text/visual edge、uniform fusion | 已训练模型对某机制有多强依赖，以及是否存在共适应 |

反事实结果必须和重训练消融联合解释。例如 core 在推理时突然换成 uniform fusion 会掉点，不代表 uniform fusion 从头训练也差；投影层与 head 已经适应了原来的 gate。本文所有“模块带来多少净增益”的结论都优先采用重训练结果。

诊断中的边 AUC、homophily 和正确性分层使用标签，但只在 checkpoint 训练完成后离线计算；它们不进入前向、loss 或 early stopping，因此没有标签泄漏。

---

## 5. 全部变体的下游结果

### 5.1 Test Accuracy（%）

| 变体 | Movies | Toys | Grocery | ele-fashion | Reddit-S | 宏平均 |
|---|---:|---:|---:|---:|---:|---:|
| core_no_graph | 52.51±0.27 | 74.44±0.21 | 78.68±0.34 | 87.89±0.21 | 93.17±0.09 | 77.34 |
| lowpass_uniform | 56.95±0.70 | 79.59±0.18 | 83.35±0.46 | 87.25±0.13 | 95.63±0.12 | 80.55 |
| reliability_uniform | 56.90±0.07 | 79.40±0.06 | **83.48±0.40** | 87.50±0.08 | 95.61±0.09 | 80.58 |
| semantic_uniform | 56.83±0.53 | **79.66±0.18** | 83.47±0.26 | 87.57±0.12 | 95.96±0.08 | **80.70** |
| core | **57.07±0.29** | 79.41±0.26 | 83.27±0.10 | 87.60±0.12 | 95.90±0.17 | 80.65 |
| double_reliability | 55.78±0.74 | 78.59±0.51 | 82.71±0.91 | 87.60±0.09 | 95.62±0.26 | 80.06 |
| edge_text_only | 57.00±0.20 | 79.35±0.12 | 83.16±0.20 | 87.66±0.12 | 95.90±0.15 | 80.62 |
| edge_visual_only | 56.78±0.64 | 79.45±0.20 | 83.28±0.31 | 87.63±0.06 | 95.98±0.17 | 80.62 |
| edge_reliability_aware | **57.08±0.29** | 79.37±0.05 | 83.28±0.38 | 87.72±0.06 | 95.96±0.16 | 80.68 |
| lowpass_residual | 55.14±0.35 | 78.91±0.40 | 82.36±0.72 | 87.91±0.02 | 96.23±0.07 | 80.11 |
| original_v1_gamma | 55.94±0.69 | 78.94±0.18 | 82.98±0.53 | **88.31±0.15** | 96.26±0.13 | 80.49 |
| self_residual | 56.56±0.36 | 79.13±0.45 | 83.10±0.17 | 87.97±0.25 | **96.32±0.07** | 80.62 |
| prototype_residual | 57.07±0.34 | 79.24±0.04 | 82.95±0.34 | 87.47±0.21 | 95.88±0.07 | 80.52 |

粗体表示同一数据集中的最好值或并列近似最好值；它仅用于定位，不代表显著性。

### 5.2 Test Macro-F1（%）

| 变体 | Movies | Toys | Grocery | ele-fashion | Reddit-S | 宏平均 |
|---|---:|---:|---:|---:|---:|---:|
| core_no_graph | 42.29±1.23 | 71.25±0.32 | 69.89±0.48 | 76.95±1.18 | 87.67±0.41 | 69.61 |
| lowpass_uniform | **51.86±0.17** | 77.04±0.25 | 74.72±0.54 | 75.83±0.15 | 91.04±0.03 | 74.10 |
| reliability_uniform | 50.63±0.24 | 76.75±0.48 | 75.36±0.36 | 76.31±0.27 | 91.14±0.17 | 74.04 |
| semantic_uniform | 51.27±0.32 | **77.35±0.54** | **75.92±0.86** | 76.20±0.46 | 91.49±0.13 | **74.45** |
| core | 50.66±0.58 | 76.81±0.52 | 74.68±0.63 | 76.71±0.52 | 91.44±0.37 | 74.06 |
| double_reliability | 49.18±0.80 | 75.94±0.44 | 74.84±0.83 | 76.59±0.49 | 91.09±0.40 | 73.53 |
| edge_text_only | 51.28±0.28 | 76.57±0.66 | 75.39±0.79 | 76.51±0.38 | 91.58±0.24 | 74.27 |
| edge_visual_only | 50.56±0.16 | 76.83±0.81 | 75.30±0.06 | 76.41±0.35 | 91.77±0.19 | 74.17 |
| edge_reliability_aware | 50.47±0.34 | 76.89±0.29 | 75.07±0.70 | 76.59±0.75 | 91.70±0.17 | 74.14 |
| lowpass_residual | 47.31±0.29 | 75.76±0.59 | 73.02±0.98 | 77.58±0.61 | 92.05±0.19 | 73.14 |
| original_v1_gamma | 48.75±0.27 | 76.07±0.34 | 75.26±0.71 | **77.60±0.17** | 92.18±0.26 | 73.97 |
| self_residual | 49.48±0.93 | 76.67±0.71 | 74.57±0.24 | 77.07±1.22 | **92.26±0.14** | 74.01 |
| prototype_residual | 51.29±0.57 | 76.58±0.22 | 75.21±0.56 | 76.90±0.34 | 91.75±0.19 | 74.35 |

---

## 6. 重训练消融：每个机制到底贡献多少

下表是相同 dataset/seed 的成对差值。单元格为 `ΔAcc/ΔF1`；最后一列的胜场是 15 个 dataset-seed 配对中差值大于 0 的次数。

| 对比（A - B） | Movies | Toys | Grocery | ele-fashion | Reddit-S | 五数据集平均 | Acc/F1 胜场 |
|---|---:|---:|---:|---:|---:|---:|---:|
| low-pass：lowpass_uniform - core_no_graph | +4.44/+9.57 | +5.15/+5.79 | +4.67/+4.82 | -0.64/-1.13 | +2.45/+3.37 | **+3.21/+4.49** | 12/15，13/15 |
| reliability：reliability_uniform - lowpass_uniform | -0.05/-1.23 | -0.19/-0.30 | +0.14/+0.65 | +0.25/+0.49 | -0.02/+0.10 | +0.03/-0.06 | 8/15，7/15 |
| semantic edge：semantic_uniform - lowpass_uniform | -0.12/-0.59 | +0.07/+0.31 | +0.12/+1.20 | +0.32/+0.37 | +0.34/+0.45 | **+0.14/+0.35** | 11/15，10/15 |
| reliability on semantic：core - semantic_uniform | +0.24/-0.61 | -0.26/-0.54 | -0.20/-1.24 | +0.03/+0.51 | -0.06/-0.06 | -0.05/-0.39 | 8/15，5/15 |
| semantic on reliability：core - reliability_uniform | +0.17/+0.03 | +0.01/+0.06 | -0.21/-0.68 | +0.10/+0.39 | +0.29/+0.30 | +0.07/+0.02 | 10/15，8/15 |
| double reliability - core | -1.29/-1.49 | -0.82/-0.87 | -0.56/+0.16 | -0.00/-0.11 | -0.28/-0.35 | **-0.59/-0.53** | 5/15，3/15 |
| text-only edge - core | -0.07/+0.62 | -0.06/-0.24 | -0.11/+0.71 | +0.06/-0.19 | +0.00/+0.14 | -0.03/+0.21 | 7/15，8/15 |
| visual-only edge - core | -0.29/-0.10 | +0.04/+0.02 | +0.01/+0.61 | +0.02/-0.30 | +0.08/+0.33 | -0.03/+0.11 | 10/15，9/15 |
| reliability-aware edge - core | +0.01/-0.19 | -0.04/+0.08 | +0.01/+0.39 | +0.12/-0.12 | +0.06/+0.26 | +0.03/+0.08 | 8/15，7/15 |
| lowpass residual - core | -1.93/-3.35 | -0.50/-1.05 | -0.91/-1.66 | +0.30/+0.87 | +0.33/+0.61 | **-0.54/-0.92** | 6/15，5/15 |
| v1 gamma - core | -1.13/-1.91 | -0.47/-0.74 | -0.29/+0.58 | +0.71/+0.90 | +0.36/+0.74 | -0.17/-0.09 | 7/15，9/15 |
| self residual - core | -0.51/-1.18 | -0.27/-0.14 | -0.17/-0.11 | +0.37/+0.36 | +0.42/+0.82 | -0.03/-0.05 | 9/15，8/15 |
| prototype residual - core | +0.00/+0.62 | -0.17/-0.23 | -0.32/+0.53 | -0.14/+0.20 | -0.02/+0.31 | -0.13/+0.29 | 4/15，8/15 |

### 6.1 最小充分结构

三个关键模型的宏平均是：

| 模型 | 组成 | Acc | F1 |
|---|---|---:|---:|
| lowpass_uniform | 均匀模态融合 + uniform graph low-pass | 80.55 | 74.10 |
| semantic_uniform | 均匀模态融合 + semantic graph low-pass | **80.70** | **74.45** |
| core | reliability fusion + semantic graph low-pass | 80.65 | 74.06 |

因此，当前五个 NC 数据集上的“最小充分版本”其实是 `semantic_uniform`，甚至 `lowpass_uniform` 已经保留了绝大多数性能。默认 core 的 reliability gate 并没有形成额外的平均收益。

需要注意，各模块存在交互，不能把这些差值当作严格可加的 Shapley 分解；但主次关系远大于 seed 波动，结论足够明确。

---

## 7. 为什么 low-pass 主干有效

### 7.1 冻结 core 去图会大幅掉点

对训练完成的 core 临时改为 `no_graph`，性能平均下降 **5.45 Acc / 5.46 F1**；这比重训练的 low-pass 增益更大，因为模型训练期间已经与图传播共适应。

| core 冻结反事实 | Movies | Toys | Grocery | ele-fashion | Reddit-S | 平均 |
|---|---:|---:|---:|---:|---:|---:|
| no_graph | -7.18/-8.09 | -6.93/-6.74 | -8.77/-7.94 | -1.15/-0.82 | -3.20/-3.71 | **-5.45/-5.46** |
| uniform_edges | +0.00/-0.28 | +0.27/+0.18 | +0.32/+0.09 | -0.31/-0.79 | -0.30/-0.44 | -0.00/-0.25 |
| shuffled_edges | -0.08/-0.30 | +0.02/+0.02 | +0.00/-0.03 | -0.05/-0.09 | -0.05/-0.05 | -0.03/-0.09 |
| inverted_edges | -0.07/-0.12 | -0.13/-0.18 | -0.29/-0.18 | +0.05/+0.19 | +0.04/+0.09 | -0.08/-0.04 |
| uniform modality fusion | -5.71/-12.62 | -0.39/-0.42 | -0.15/-0.27 | -1.31/-2.08 | -0.27/-0.43 | -1.56/-3.16 |

解释：模型强依赖“是否传播”，却对“当前这组细微边权如何排列”不敏感。这与重训练消融完全一致：主要价值来自拓扑上的浅层消息传递，而非精细边权。

### 7.2 不同节点度数上的增益

以下为 `core - core_no_graph` 的 test accuracy 差值。括号内为该分箱 test 节点数；节点数在 3 个 run 间相同。

| 数据集 | degree 0–1 | 2–3 | 4–7 | 8–15 | ≥16 |
|---|---:|---:|---:|---:|---:|
| Movies | +0.54 (677) | +4.10 (683) | +5.67 (699) | +8.41 (674) | +3.99 (602) |
| Toys | +4.03 (1,132) | +4.72 (1,194) | +6.08 (971) | +4.50 (548) | +6.80 (294) |
| Grocery | +2.69 (868) | +6.86 (816) | +4.30 (721) | +6.27 (537) | +2.68 (473) |
| ele-fashion | -0.16 (13,579) | -0.43 (8,757) | -0.22 (4,143) | -0.51 (1,624) | -0.62 (1,227) |
| Reddit-S | +3.57 (961) | +4.42 (716) | +1.46 (525) | +2.59 (348) | +0.64 (629) |

关键诊断：

- 在四个正收益数据集上，增益覆盖多个度数区间，不是少量 hub 节点造成的平均数假象。
- ele-fashion 在所有度数区间都退化，所以其问题不能仅归因于低度节点；即使高阶节点也没有从当前传播规则受益。
- Movies 的最大收益在 8–15 度，Reddit-S 的最大收益反而在低度区间。单一“度越高越该传播”的规则不成立。

### 7.3 表示变化

core 中 `z_low` 相对 `h0` 的平均相对变化为 0.287–0.445；有效秩降至 `h0` 的 49%–62%。这说明两跳传播确实完成了明显的低通/压缩，而不是近似恒等映射。

| 数据集 | `z_low` 相对变化 | rank(`z_low`)/rank(`h0`) | 冻结去图 ΔAcc/F1 |
|---|---:|---:|---:|
| Movies | 0.445 | 0.545 | -7.18/-8.09 |
| Toys | 0.363 | 0.570 | -6.93/-6.74 |
| Grocery | 0.421 | 0.489 | -8.77/-7.94 |
| ele-fashion | 0.345 | 0.545 | -1.15/-0.82 |
| Reddit-S | 0.287 | 0.622 | -3.20/-3.71 |

低秩化本身不是目的：ele-fashion 同样发生低秩化，却没有性能收益。有效的是“标签一致邻域带来的有用压缩”，而非越平滑越好。

### 7.4 low-pass 同时具有明显的正则化效应

在各模型自己的最佳 validation checkpoint 上，五数据集平均的 `train Acc - test Acc` 为：

| 变体 | 平均 train-test gap |
|---|---:|
| core_no_graph | 13.33 pp |
| lowpass_uniform | **8.68 pp** |
| core | 9.01 pp |

low-pass 将平均训练—测试间隙缩小约 4.65 个百分点；Toys/Grocery 分别缩小约 10.58/8.62 个点，Reddit-S 也缩小 2.13 个点。Movies 的 gap 只缩小 0.43 个点但测试性能仍显著提高。由此看，图传播的收益至少包含两部分：邻域标签相关信息的注入，以及对属性投影网络过拟合的平滑正则化。

---

## 8. 输出 MLP/残差归一化承担了很强的判别整形作用

对 core 各阶段做相同设置的后验线性探针，结果如下（`Acc/F1`）：

| 数据集 | `h_t` | `h_v` | reliability 后 `h0` | low-pass 后 `z_low` | 输出后 `z_final` |
|---|---:|---:|---:|---:|---:|
| Movies | 35.07/4.26 | 34.17/3.79 | 33.38/2.99 | 33.57/3.17 | 50.63/17.19 |
| Toys | 51.35/35.49 | 58.27/42.65 | 62.61/44.76 | 65.85/45.90 | 77.28/69.92 |
| Grocery | 48.90/28.64 | 50.21/32.92 | 53.77/33.95 | 56.22/35.46 | 80.80/65.65 |
| ele-fashion | 76.80/37.10 | 64.99/23.61 | 72.78/30.72 | 72.03/28.93 | 85.29/59.44 |
| Reddit-S | 46.64/34.30 | 86.43/74.55 | 86.55/71.82 | 88.53/74.02 | 94.79/88.00 |

五数据集平均：

- `h0 → z_low`：线性探针 +1.42 Acc / +0.65 F1；ele-fashion 在该阶段退化。
- `z_low → z_final`：+14.52 Acc / +22.54 F1。

这说明 `output_norm(output_mlp(z_input) + z_struct)` 并非无关紧要的尾部层，而是把传播后的压缩表征重新展开为类别可分空间的重要环节。不过当前实验没有对 output MLP、残差连接和 LayerNorm 分别重训练消融，因此只能确认“输出整形阶段很重要”，还不能判断三者各自贡献。

---

## 9. reliability fusion：学到了偏好，但没有增加独立性能

### 9.1 学到的模态方向基本合理

| 数据集 | 平均 `lambda_text` | 节点内 SD | P10–P90 | 跨 seed 节点排序相关 | 重训练 reliability 增益 | core 冻结改均匀融合 |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 0.220 | 0.067 | 0.166–0.319 | 0.537 | -0.05/-1.23 | -5.71/-12.62 |
| Toys | 0.415 | 0.096 | 0.293–0.545 | 0.525 | -0.19/-0.30 | -0.39/-0.42 |
| Grocery | 0.395 | 0.123 | 0.246–0.563 | 0.630 | +0.14/+0.65 | -0.15/-0.27 |
| ele-fashion | 0.592 | 0.151 | 0.381–0.781 | 0.411 | +0.25/+0.49 | -1.31/-2.08 |
| Reddit-S | 0.404 | 0.062 | 0.327–0.490 | 0.790 | -0.02/+0.10 | -0.27/-0.43 |

Toys/Grocery/Reddit-S 偏视觉，ele-fashion 偏文本，与单模态线性探针和原始模态统计的强弱方向一致。gate 也并非完全常数，且节点排序具有中等跨 seed 稳定性。

但是：

- reliability 独立重训练几乎没有平均增益；
- `core - semantic_uniform` 甚至为 -0.05 Acc / -0.39 F1；
- test 节点中正确与错误预测的平均 `lambda_text` 很接近，gate 没有表现出明显的“困难节点可靠性校准”。

冻结 core 后突然改成均匀融合会掉点，尤其 Movies，但这只说明投影层、输出层和 head 已与 gate 共适应。均匀融合从头训练仍可达到相同甚至更好的性能，因此不能把冻结掉点解释成 gate 的独立贡献。

### 9.2 为什么 double reliability 会伤害

single 模式为：

```text
h0 = lambda_t * h_t + lambda_v * h_v
lambda_t = r_t / (r_t + r_v)
```

double 模式又在特征上乘一次可靠性：

```text
h0 = lambda_t * (r_t * h_t) + lambda_v * (r_v * h_v)
```

这会二次压低弱模态，同时改变融合表示的整体幅值和优化条件。五个数据集的 Acc 全部不增，说明 v2 从 v1 的 double 改为 single 是正确的简化；它的价值是避免伤害，而不是说明 single gate 本身优于无 gate。

---

## 10. semantic edge：判别排序很好，但实际权重过平

| 数据集 | 原图同配率 | 投影文本 AUC | 投影视觉 AUC | 最终边权 AUC | 边权 CV | 加权同配率提升 | 有害消息质量变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | 0.447 | 0.612 | 0.644 | 0.654 | 2.67% | +0.35 pp | -0.09 pp |
| Toys | 0.724 | 0.639 | 0.657 | 0.680 | 2.34% | +0.31 pp | -0.07 pp |
| Grocery | 0.681 | 0.710 | 0.730 | 0.763 | 2.62% | +0.50 pp | -0.15 pp |
| ele-fashion | 0.767 | 0.766 | 0.698 | 0.758 | 2.63% | +0.53 pp | -0.09 pp |
| Reddit-S | 0.959 | 0.613 | 0.956 | 0.935 | 1.90% | +0.17 pp | -0.04 pp |

当前公式是：

```text
w = 0.1 + 0.9 * sigmoid(sim / 2.0)
```

即使 `sim` 覆盖完整的 `[-1, 1]`，理论边权也只约落在 `[0.440, 0.660]`；实际均值约 0.615–0.635，离散程度更小。随后 `gcn_norm` 还会消除共同尺度，真正起作用的只有非常有限的相对差异。

这解释了三个相互一致的观察：

1. 边权 AUC 很好，表明语义相似度排序包含真实标签关系；
2. 加权同配率只提高不到 0.6 个百分点，说明这些信息没有被强力用于传播；
3. 冻结模型中 uniform/shuffle/invert 边权只引起约 0.0–0.1 个点的平均变化。

不同 edge mode 间差异也都很小，`avg_cos` 没有不可替代性。现状不是“semantic edge 思路无效”，而是“当前映射函数非常保守，只产生小修正”。

---

## 11. 额外结构路径为什么没有形成通用增益

### 11.1 性能模式

- `lowpass_residual`：平均 -0.54 Acc / -0.92 F1；Movies 最差（-1.93/-3.35），ele-fashion 和 Reddit-S 小幅受益。
- `original_v1_gamma`：平均 -0.17/-0.09；ele-fashion +0.71/+0.90，Reddit-S +0.36/+0.74，但 Movies/Toys/Grocery 的 Acc 均下降。
- `self_residual`：平均基本持平；ele-fashion +0.37/+0.36，Reddit-S +0.42/+0.82，另外三个数据集下降。
- `prototype_residual`：平均 -0.13/+0.29，没有任何数据集 Acc 稳定提高。

### 11.2 门控与实际修正幅度

| 机制 | 门值范围（五数据集均值） | 门的节点内 SD 范围 | 修正范数 / `z_low` 范围 | 诊断 |
|---|---:|---:|---:|---|
| lowpass residual (`eta_max=0.2`) | 0.166–0.198 | 0.0004–0.0038 | 19.1%–28.0% | 多数接近上限，几乎不是节点级 gate |
| self residual (`alpha_max=0.2`) | 0.136–0.199 | 0.0001–0.0247 | 23.0%–40.4% | 除 Movies 外高度饱和，修正不算“小” |
| v1 gamma | 0.661–0.842 | 0.0005–0.0566 | 相对纯 low-pass 改动 14.2%–27.9% | 多数数据集近似常数混合 |
| prototype residual | 固定系数 0.2 | — | **0.76%–1.01%** | 实际路径幅度太小，基本被忽略 |

因此额外路径失败的直接原因不是“模型学会在无用节点把门关掉”，恰恰相反：有界 sigmoid gate 大量饱和在上界附近，把本应自适应的机制退化成近似固定残差。该固定修正对 ele-fashion/Reddit-S 有利，对另外三个数据集有害。

五个数据点上，`lowpass_residual - core` 的增益与 homophily 的 Spearman 相关为 1.0，self/gamma 也呈强正相关；但由于 ele-fashion 的纯 low-pass 本身是反例，不能据此得出“homophily 越高越应加 high-pass”。更合理的解释是：额外属性/残差通道是否有用取决于图传播相对原始模态的净收益，homophily 只是其中一个因素。

---

## 12. 分数据集诊断

### Movies

- 图同配率最低，但 low-pass 仍带来 +4.44 Acc / +9.57 F1，说明即使边中异类比例高，两跳归一化传播仍能提供有用的局部统计。
- semantic edge 没有带来重训练收益，原因很可能是边权动态范围太小，不足以真正抑制其 56% 左右的节点平均有害消息质量。
- reliability 强烈偏视觉，但其独立 F1 贡献为 -1.23；冻结改均匀融合却大跌，说明共适应尤其强。
- 所有较强残差路径都明显伤害。Movies 应使用干净的低通主干，不应增加固定比例高频/自路径。

### Toys

- low-pass 是最主要增益：+5.15/+5.79。
- 视觉模态更强，gate 方向正确，但 reliability 不增益。
- semantic edge 有很小的正 F1；各种额外残差均下降。
- 最适合简单的 uniform/semantic low-pass。

### Grocery

- low-pass +4.67/+4.82，是主因。
- semantic edge 的 F1 增益最大（+1.20），边权 AUC 0.763、加权同配率 +0.50 pp，说明这里最有继续强化边过滤的潜力。
- reliability 单独略有增益，但与 semantic 组合后出现负交互；应避免同时用两个弱 gate 对同一信号重复调节。

### ele-fashion

- `core_no_graph` 已达到 87.89 Acc，纯 low-pass 降低 0.64/1.13；所有度数段都下降。
- 文本单模态明显强于视觉，reliability/semantic 能回收一部分损失，但 core 仍未超过 no-graph Acc。
- v1 gamma 达到最高 Acc/F1（88.31/77.60），表明该数据集需要更强的属性/非低频保留。
- 后续应优先研究按节点 bypass，而不是继续增加传播深度。

### Reddit-S

- homophily 0.959，视觉边 AUC 0.825；low-pass 稳定增加 +2.45/+3.37。
- semantic edge 再增加 +0.34/+0.45，但原图已很纯，提升空间有限。
- self residual 获得最高 96.32/92.26；gamma/lowpass residual 也改善。这说明在非常可靠的图低频之外，保留少量节点自身判别信息仍有价值。

---

## 13. 对后续模型设计最有价值的经验

### 13.1 应保留

1. **浅层 restart low-pass 作为默认结构主干。** 这是唯一跨数据集大幅、稳定、被严格对照确认的模块。
2. **single 而不是 double reliability。** 如果保留 reliability，只让它决定比例，不要再次缩放模态特征。
3. **投影后的输出残差 MLP + LayerNorm。** 表示探针显示输出阶段承担强判别整形；在进一步拆分实验前不应删除。
4. **全图一致训练/评估、相同 checkpoint 规则与成对 seed。** 这些工程设计让图传播不受采样邻域变化干扰，并使机制差值可解释。其性能贡献尚未独立消融，但实验稳定性价值明确。

### 13.2 可以简化或默认关闭

1. reliability gate：当前不是独立增益来源。可先替换为均匀融合或更廉价的全局可学习标量。
2. prototype residual：实际幅度不足 1%，继续默认关闭。
3. lowpass residual/self residual/v1 gamma：不要作为所有数据集默认开启项，只能作为条件模块。
4. 多种 edge mode：当前差异太小，不值得增加选择复杂度；`avg_cos` 或 uniform 都足够作为基线。

### 13.3 值得重新设计，而不是简单删除

1. **semantic edge 的映射强度。** 保留 cosine 排序信号，但重新设计权重动态范围；把 edge CV、加权同配率提升和冻结 shuffle/invert 敏感性作为联合诊断指标。
2. **传播 bypass。** 学习 `z = beta*z_low + (1-beta)*h0` 时应零初始化或从低通端初始化，并防止 gate 贴边；输入可包含度、邻域一致性、模态冲突和局部结构不确定性。
3. **reliability 的训练目标。** 如果想让它真表示可靠性，需要模态 dropout/corruption、缺失模态或独立校准约束；仅靠最终分类损失，投影层与 gate 存在严重可替代性。
4. **数据集/节点条件化而非全局固定增强。** ele-fashion 和 Reddit-S 说明额外自信息有时有用，但最优强度不同；当前门饱和导致其近似固定系数。

### 13.4 下一轮最优先的最小实验

当前实验仍把“low-pass bundle”和“输出整形 bundle”视为整体。若目标是继续深挖性能来源，优先级应是：

1. `num_hops = 0/1/2/3/4`，确认两跳是否真是拐点；
2. `restart = 0/0.05/0.15/0.3/0.5`；
3. 自环开/关；
4. `output_mlp` 开/关、残差开/关、LayerNorm 开/关的 2×2×2 小因子实验；
5. semantic edge 的 `temperature × edge_weight_min` 网格，并要求 edge CV 至少形成可见跨度；
6. uniform fusion、全局可学习模态标量、节点 gate 三者对照，再加入模态 corruption 测试；
7. 只对 ele-fashion/Reddit-S 测试零初始化的 adaptive bypass，观察 gate 是否仍贴边。

这七项比继续添加新路径更有信息价值，因为它们直接拆解目前已确认的主干和失败 gate。

---

## 14. 工程与诊断注意事项

1. 除 prototype 外，当前所有变体报告的参数量都为 1,226,652，因为模型即使关闭模块也仍实例化对应参数；prototype 为 1,230,748。这个设计很好地控制了消融中的容量差异，但参数量不代表部署时的真实活动计算量。若最终简化模型，应再做物理删模块后的速度/显存测试。
2. 当前以 validation accuracy 选 checkpoint，而本文同时关心 Macro-F1。Movies 等类别不平衡数据上的 F1 波动明显；如果后续目标是最大化 F1，需要单独比较 val-Acc 与 val-F1 checkpoint selection，不能把选择规则效应误认为架构效应。
3. `counterfactual_table.csv` 中的 `double_reliability_input` 实际是恒等反事实：诊断函数再次调用当前模型的 `_reliability_fusion`，core 仍按 single 模式执行。因此该行不能使用；本文关于 double reliability 的结论只来自有效的重训练变体。
4. self-residual 变体的通用 counterfactual 重构没有重新注入 `z_input` 中的 self path，因此本文没有使用该变体的冻结反事实，只使用重训练结果与直接重算的门/修正幅度。
5. 反事实去图的掉点大于重训练消融增益是正常的训练共适应现象，不代表图传播真实独立贡献有 5.45 个点；独立贡献应优先看 +3.21/+4.49 的重训练对照。
6. 边 AUC、同配率和“有害消息”使用标签只做训练后诊断，模型生成边权时没有读取标签，不构成标签泄漏。
7. 只有 5 个数据集、每项 3 seeds。除 low-pass 外多数差异与 seed 波动同量级，应视作设计方向而非稳定普适规律。

---

## 15. 最终判断

如果现在要基于这些结果设计下一版模型，最合理的出发点不是继续保留 v2 的所有组件，而是：

```text
双模态投影
  → 简单/低成本融合
  → 两跳 restart low-pass（默认核心）
  → 可学习但真正可关闭的 attribute bypass
  → residual output MLP + LayerNorm
```

semantic edge 可以作为下一阶段重点优化项，因为它已经证明“排序正确但作用太弱”；reliability 只有在引入模态缺失/噪声训练后才值得保留为节点级机制；prototype 和当前饱和式残差 gate 不应进入默认主干。

因此，v2 的主要设计经验不是“复杂多模态路由有效”，而是：**删除并列路径竞争后，让一个稳定、浅层、带 restart 的结构低通成为主干，再用强输出整形恢复判别空间，已经足以获得绝大多数下游性能。**
