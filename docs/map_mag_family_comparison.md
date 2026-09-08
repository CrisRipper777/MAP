# MAP-MAG 系列实现与版本差异

本文以当前 `/hdd1/DataInHere/YHF/MAP/MAP` 中的实际代码为准，说明
`map_mag`、`map_mag_v1`、`map_mag_v2` 和 `map_mag_v3` 的继承关系、结构差异、
当前训练协议以及公平比较时应使用的配置。

## 1. 先澄清版本命名

`map_mag` 和 `map_mag_v1` 不是两个完全不同的模型世代。

- `map_mag` 是最早的可变实现入口，代码位于 `src/models/map_mag.py`。
- `map_mag_v1` 是在此基础上冻结的 v1 快照，代码位于
  `src/models/map_mag_v1.py`。它保留原始三路径架构，并把
  `reliability_mode`、`fusion_type` 等实验接口显式化。
- `map_mag_v2` 才是第一次明显改变主干：从“三条并列路径竞争”改成
  “先融合模态、再在一个语义加权图上低通传播”。
- `map_mag_v3` 再次改变主干：文本和视觉先在各自的语义加权图上传播，之后
  才进行节点级模态融合。

因此更准确的谱系是：

```text
map_mag (原始工作实现)
  └─ map_mag_v1 (冻结并扩展消融接口，核心架构不变)
       └─ map_mag_v2 (单一语义图、低通主干)
            └─ map_mag_v3 (双模态语义图、先传播后融合)
```

## 2. 四个实现的共同点

四个版本都满足当前 trainer 的同一 encoder 契约：

```python
z, None, None, aux_loss, aux_info = model(x, edge_index)
```

共同设计包括：

- 按 `text_dim` 和 `visual_dim` 将冻结 joint feature 拆为文本与视觉特征；
- 两种模态分别经过投影 MLP，映射到统一 `hidden_dim`；
- 使用节点的一阶邻居均值估计文本/视觉可靠性；
- 所有图计算都基于稀疏 `edge_index`，不构造 `N x N` dense adjacency；
- 图扩散均采用带 restart 的 GCN 归一化传播；
- 输出均为节点表示，NC/LP task head 位于模型外部；
- 均支持 `full` 推理；当前所谓 `layerwise` 对这些全局模型仍是 exact
  full-graph fallback，并不真正降低峰值显存。

## 3. 默认结构总览

| 维度 | `map_mag` | `map_mag_v1` | `map_mag_v2` | `map_mag_v3` core |
|---|---|---|---|---|
| 默认 hop | 2 | 3 | 2 | 2 |
| reliability 默认 | 固定等效 double | double，可切 single | single，可切 double | single，可切 double/none |
| 模态何时融合 | 传播前 | 传播前 | 传播前 | 传播后 |
| message graph | 原图、统一无权边 | 原图、统一无权边 | 一个共享语义加权图 | 文本/视觉两个语义加权图 |
| 传播主干 | low/high 混合 | low/high 混合 | low-pass | 双路 low-pass |
| 主 router | self/structure/prototype 三路 | self/structure/prototype 三路 | 默认无三路竞争 | text/visual 两模态 router |
| self 路径 | 默认开启、参与竞争 | 默认开启、参与竞争 | 可控小残差，默认关闭 | 可控小残差，默认关闭 |
| prototype | 一组共享 prototype，默认开启 | 一组共享 prototype，默认开启 | 一组共享残差，默认关闭 | text/visual/common 三组，默认关闭 |
| 语义边权 | 无 | 无 | 有，但所有模态共享一个权重 | 每种模态独立权重 |
| conflict 通道 | 无 | 无 | 仅 high-pass residual 诊断 | 可选互补/冲突传播，core 关闭 |
| 默认辅助损失 | prototype + router balance | prototype + router balance | 全关 | 全关 |

这里的“双图”是指在同一原始拓扑上使用两组不同边权，不会额外添加不存在的边。

## 4. `map_mag`：原始三路径模型

主流程为：

```text
x_t,x_v -> h_t,h_v
          -> r_t,r_v
          -> reliability-weighted h0
          -> self / structure / shared prototype
          -> 三路径 router
          -> residual MLP + LayerNorm
```

它先计算 `h_t_rel=r_t*h_t`、`h_v_rel=r_v*h_v`，再用归一化可靠性
`lambda_t/lambda_v` 得到 `h0`。这相当于可靠性既进入模态占比，又乘到模态特征
上，即后续所称的 `double reliability`。

结构路径使用原图的统一无权归一化邻接：

```text
z_low  = RestartDiffuse(h0, A)
z_high = h0 - z_low
z_struct = gamma*z_low + (1-gamma)*z_high
```

最后由节点级三路 router 融合 `z_self/z_struct/z_proto`。prototype 是一组全局共享
原型，prototype diversity loss 和 router balance loss 默认开启。

## 5. `map_mag_v1`：冻结版三路径模型

v1 与 `map_mag` 的默认计算图高度一致，最重要的价值是冻结版本并提供可复现实验
接口，而不是提出一条全新主干。

相对 `map_mag` 的实质变化：

- 增加 `reliability_mode=double|single`。`double` 与原始实现一致；`single`
  让 structure 的 `h0` 不再额外乘一次可靠性，但 self/router 仍使用可靠性缩放特征。
- 增加 `fusion_type=weighted_sum|concat_mlp`；默认仍是原始加权求和。
- 默认 `num_hops` 从 2 改为 3。
- 增加版本字段和 prototype 诊断导出配置。
- norm helper 从文件内实现改为共享实现只属于代码组织差异，不改变数学行为。

所以按默认配置直接比较 `map_mag` 与 `map_mag_v1` 时，主要差别是传播深度 2 vs 3；
若统一 `num_hops=2`、保持 `reliability_mode=double` 和
`fusion_type=weighted_sum`，两者应当非常接近。

## 6. `map_mag_v2`：共享语义图上的低通主干

v2 删除了 v1 默认的三条并列路径竞争，将主线收敛为：

```text
h_t,h_v -> r_t,r_v -> h0
         -> shared semantic edge weight w_ij
         -> restart low-pass diffusion
         -> optional residual modules
```

默认使用 single reliability：

```text
h0 = lambda_t*h_t + lambda_v*h_v
```

然后在每条原图边上根据文本和视觉 cosine 生成一个共享权重。默认
`edge_weight_mode=avg_cos`：

```text
w_ij = w_min + (1-w_min)*sigmoid((cos_t+cos_v)/(2*tau))
```

核心默认只保留 `z_low`。v1 的对称 low/high 竞争不再是主线，仅保留：

- `lowpass_residual`：受小门控约束的 high-frequency correction；
- `original_v1_gamma`：用于退化诊断；
- self residual 和 shared prototype residual：默认均关闭。

因此 v2 的核心假设是：先得到统一多模态表示，再利用一个结构—语义一致的共享图
做平滑传播。

## 7. `map_mag_v3`：模态偏好引导的双语义传播

v3 的核心计算顺序为：

```text
x_t,x_v -> h_t,h_v -> r_t,r_v
          -> w_text, w_visual
          -> z_text, z_visual（分别传播）
          -> alpha_text, alpha_visual（节点级模态 router）
          -> z_struct
          -> optional self/prototype residual
          -> residual MLP + LayerNorm
```

核心区别不是“边权公式更复杂”，而是把模态融合从传播前推迟到传播后：

- 文本边权只由文本语义关系控制；
- 视觉边权只由视觉语义关系控制；
- 两种模态分别传播，避免一个模态的不可靠邻接关系直接污染另一个模态；
- 最后用每节点 `alpha_text/alpha_visual` 决定两路传播结果的占比。

core 默认使用 `separate_cos + modality_specific + learned_router`。提供的受控退化项：

- `propagation_mode=shared_fused` 和 `edge_weight_mode=shared_avg_cos`：接近 v2
  的“先融合、后单图传播”；
- `modality_fusion_mode=reliability`：去掉学习型模态 router；
- `modality_fusion_mode=uniform`：固定两模态各 0.5；
- `reliability_residual`：以 reliability 比例为先验，学习零初始化的有界修正。

可选增强模块：

- conflict channel：使用 `1-w_m` 构造互补关系，通过最大值受限的 `eta_m`
  注入与一致传播的差分；
- controlled self residual：通过 `beta_self<=self_residual_max` 注入属性路径；
- modality-specific prototypes：分别维护 `P_text/P_visual/P_common`，支持随机或
  一次性 K-means 初始化；
- modality balance、prototype diversity 和 edge regularization，默认权重均为 0。

三个配置的定位：

- `map_mag_v3`：稳定核心，conflict/self/prototype 全部关闭；
- `map_mag_v3_full`：开启 conflict、self 和三组 prototype；
- `map_mag_v3_lp`：开启 conflict 与 self，prototype 关闭。它是 encoder 消融
  preset，不会绕过统一 LP decoder。

## 8. 当前项目中的训练与评估协议

迁移没有为 v3 创建独立 trainer。四个版本均走当前统一协议：

### NC

- 使用固定 NC split；
- 全图训练、全图确定性验证与测试；
- 以 validation accuracy 选 checkpoint；
- 相同的任务 head、early stopping 和汇报规则。

四个 MAP-MAG 默认配置都声明 `full_graph_training=true`，与统一 NC 协议一致。

### LP

- 所有图 encoder 都使用统一 `LinkNeighborLoader` sampled training；
- fanout `[5,5]`、batch size 2048、每个正样本一个全局过滤负样本；
- batch 正监督边的两个方向通过 `global_eid` 后端从 message graph 删除；
- validation/test 使用完整 train-only message graph 和官方固定 negatives；
- 所有 encoder 输出统一投影到 128 维，使用相同 Hadamard MLP decoder。

参考目录中的 v3 实现曾为 `map_mag_v3_full/map_mag_v3_lp` 使用
`concat_product_abs=[z_i,z_j,z_i*z_j,|z_i-z_j|]` decoder。当前项目没有启用它，
因为这会同时改变 encoder 和 LP head 容量，破坏与其他模型的统一比较。若未来研究
decoder，应作为独立 ablation，不进入统一协议主表。

虽然 v2/v3 配置声明偏好 full-graph training，LP runner 会统一覆盖为 sampled
adaptation；这与当前协议文档保持一致。

## 9. 如何做公平的版本比较

建议主表使用：

```text
map_mag_v1
map_mag_v2
map_mag_v3
```

`map_mag` 作为 legacy 复现项单列，因为它与 v1 大部分计算重合。若四者都进入表格，
必须明确结果是“各版本默认配置比较”，其中 `map_mag_v1` 默认 3 hops，而其他版本
默认 2 hops。

若目标是隔离架构演进，至少统一：

- `hidden_dim`；
- `num_hops`；
- dropout 和 norm；
- NC/LP split、seed、训练轮数和 early stopping；
- LP sampler、负样本、正边删除和共享 decoder；
- 是否开启额外 prototype/self/conflict 模块及其 auxiliary loss。

`map_mag_v3_full` 和 `map_mag_v3_lp` 应放在 v3 ablation 表，不应拿来替代 core v3
与 v1/v2 主干直接比较，否则观察到的差异同时包含多个新增模块。

需要注意：当前统一协议暂未统一每个模型的学习率搜索，因此主表仍属于冻结配置下的
协议公平，而不是“每个模型获得同等超参数搜索预算”的最强公平性。

## 10. 迁移清单与验证

当前迁移包括：

- `src/models/map_mag_v3.py`：完整 v3 encoder；
- `configs/model/map_mag_v3*.yaml`：core/full/LP 三个 preset；
- `src/tasks/nc.py`、`src/tasks/lp.py`：训练前一次性 K-means prototype 初始化；
- `src/tasks/common.py`：v3 scalar `aux_info` 汇总键；
- `src/tasks/analysis.py`：动态 node aux 字段和双模态 edge-weight 导出；
- `tests/test_map_mag_v3.py`：factory、模式开关、数值稳定性、反向、K-means、
  推理等价和导出测试。

迁移后应使用以下命令验证：

```bash
PYTHONPATH=. conda run --no-capture-output -n yhf_env \
  python -m pytest tests/test_map_mag_v3.py -q

PYTHONPATH=. conda run --no-capture-output -n yhf_env \
  python -m pytest -q
```

正式运行示例：

```bash
# v3 core NC
python -m src.main dataset=Movies task=nc model=map_mag_v3 num_runs=3

# v3 core LP；仍自动采用统一 sampled LP
python -m src.main dataset=sports-copurchase task=lp model=map_mag_v3 num_runs=3

# 增强模块只作为独立消融
python -m src.main dataset=Movies task=nc model=map_mag_v3_full num_runs=3
python -m src.main dataset=sports-copurchase task=lp model=map_mag_v3_lp num_runs=3
```
