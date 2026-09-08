# MAP-MAG v3 Implementation Report

> 当前 MAP 项目迁移说明（2026-09-08）：本文最初记录的是
> `/hdd1/DataInHere/YHF/idea/MAG_baseline` 中的参考实现。v3 encoder、三个
> preset、K-means 初始化和 node/edge 诊断现已迁移到当前项目。为遵守
> `unified_sampled_lp_v1`，当前项目没有迁移参考实现的
> `concat_product_abs` 专用 LP decoder；所有模型继续使用共享 128 维投影和
> Hadamard MLP decoder。当前实现状态及四代模型的准确比较见
> [`map_mag_family_comparison.md`](map_mag_family_comparison.md)。本文第 11、12
> 节中的测试数量和 sanity 指标属于参考项目的历史记录，不代表当前项目验证结果。

当前 MAP 迁移验证（2026-09-08）：

- `map_mag_v3.py` 与参考实现 SHA-256 一致：
  `b31b25a6ac43cc4c168cdde96ca6df4ca5c45fb6bac7c3b43c459eca16ad9e16`；
- v3 专项测试：`30 passed`；完整项目测试：`97 passed`；
- `map_mag_v3`、`map_mag_v3_full`、`map_mag_v3_lp` 均通过 Hydra 解析；
- Movies-NC 的 v3-full/K-means 单 epoch CPU smoke 成功；
- Movies-LP 的 v3-LP sampled/global-eid 单 batch CPU smoke 成功，删除 100 条
  sampled message edges（仅作链路验证，不作为性能结果）。

## 1. 新增与修改文件

新增：

- `src/models/map_mag_v3.py`：MAP-MAG v3 完整 encoder。
- `src/models/predictor.py`：当前 MAP 迁移中未修改，继续使用统一 Hadamard predictor。
- `configs/model/map_mag_v3.yaml`：稳定 core 配置。
- `configs/model/map_mag_v3_full.yaml`：开启 conflict、self residual 和三组 prototype。
- `configs/model/map_mag_v3_lp.yaml`：开启 conflict 与 self residual，仍使用统一 LP head。
- `tests/test_map_mag_v3.py`：v3 factory、ablation、推理和导出测试。
- `docs/map_mag_v3_implementation_report.md`：本报告。

最小修改：

- `src/tasks/common.py`：注册 v3 数值型 `aux_info` keys。
- `src/tasks/analysis.py`：动态导出 v3 node aux 和 NC/LP 双模态 edge weight；v1/v2 表头不变。
- `src/tasks/nc.py`、`src/tasks/lp.py`：训练前执行可选的 K-means prototype
  初始化，并接入可选 edge aux 导出；LP 不接受模型私有 pair operator。
- `configs/task/lp.yaml`：不因 v3 改变，继续执行统一 sampled LP 与共享 decoder。
- `README.md`：增加 v3 core/full/LP 运行说明。

未修改 split、dataloader、negative sampling、LP message graph 过滤逻辑和
model factory。现有 factory 会根据 `model.name` 动态导入
`src.models.map_mag_v3`。

## 2. 总体架构

v3 的主题是 **Modality-Preference Guided Dual Semantic Propagation**：

```text
x_t, x_v
  -> h_t, h_v
  -> r_t, r_v
  -> text semantic graph, visual semantic graph
  -> z_t, z_v (independent propagation)
  -> alpha_t, alpha_v (node-level modality router)
  -> z_struct
  -> optional self/prototype residuals
  -> residual MLP + LayerNorm
```

所有图操作只在稀疏 `edge_index` 上进行，没有 dense adjacency 或
`N x N` attention。输出保持 trainer 契约：

```python
z, None, None, aux_loss, aux_info = model(x, edge_index)
```

## 3. 与 v1/v2 的区别

- v1：self/structure/prototype 三路径并列，通过三路 router 融合。
- v2：先融合文本和视觉，再在单个语义重加权图上使用低通传播。
- v3：文本和视觉分别构图、分别传播，之后才用节点级模态偏好 router 融合。
- v3 的 conflict、self residual、modality-specific prototypes 属于统一的受控增强模块，core 默认全部关闭。
- v3 提供 `shared_fused`，可退化为接近 v2 的“先融合、后传播”路径，便于公平消融。

## 4. Reliability Estimation

文本和视觉分别使用独立 reliability gate：

```text
c_i^m = [h_i^m, mean_N(h^m), h_i^m-mean_N(h^m), h_i^m*mean_N(h^m)]
r_i^m = r_min + (1-r_min) * sigmoid(MLP_m(c_i^m))
lambda_i^t = r_i^t / (r_i^t+r_i^v)
lambda_i^v = r_i^v / (r_i^t+r_i^v)
```

支持：

- `single`：reliability 用于融合、router 和可选边权调制，不再次乘特征。
- `double`：传播输入额外乘 `r_m`，保留 v1 风格的二次调制诊断。
- `none`：`r_t=r_v=1`。

core 默认 `single`。

## 5. Modality-specific Edge Construction

每条原图边 `(i,j)` 上计算：

```text
cos_t(i,j) = cosine(h_t[i], h_t[j])
cos_v(i,j) = cosine(h_v[i], h_v[j])
w(sim) = w_min + (1-w_min) * sigmoid(sim / tau_e)
```

支持五种 `edge_weight_mode`：

- `separate_cos`：`w_t=w(cos_t)`，`w_v=w(cos_v)`；core 默认。
- `reliability_scaled`：端点平均可靠性分别调制 `cos_t/cos_v`。
- `learnable_scalar`：文本和视觉使用独立可学习 scale/bias。
- `shared_avg_cos`：两种模态共享 `w((cos_t+cos_v)/2)`，作为 v2 式退化。
- `raw_uniform`：两套边权均为 1。

所有 cosine 使用 `nan_to_num`，边权被限制在
`[edge_weight_min, 1]`；空边图返回空边权及全零统计。

## 6. Dual-channel Propagation

一致通道使用 v2 restart low-pass：

```text
H_m^0 = h_m
H_m^{k+1} = (1-restart) * A_m_norm H_m^k + restart * H_m^0
```

`propagation_mode=modality_specific` 时分别计算 `z_t_pos` 和 `z_v_pos`。
`shared_fused` 时先用 reliability 融合得到 `h0`，再在
`w_shared=(w_t+w_v)/2` 上传播。

可选 conflict/complementary 通道：

```text
w_m_conflict = conflict_min + conflict_scale * (1-w_m)
eta_m = eta_max * sigmoid(MLP_m([h_m, s_m, r_m, log_degree]))
z_m = z_m_pos + eta_m * MLP_m(z_m_conflict-z_m_pos)
```

core 默认 `use_conflict_channel=false`；full 和 LP preset 开启。

## 7. Modality Preference Router

router 输入：

```text
[z_t, z_v, |z_t-z_v|, z_t*z_v, r_t, r_v, s_t, s_v, log_degree]
```

learned router：

```text
[alpha_t, alpha_v] = softmax(router(input) / tau_router)
z_struct = alpha_t*z_t + alpha_v*z_v
```

支持：

- `learned_router`：core 默认。
- `reliability_residual`：从 `alpha=lambda` 出发，学习有界的节点级
  logit 修正；输出层零初始化，因此初始表示严格等于 reliability fusion。
- `reliability`：`alpha=lambda`。
- `uniform`：固定 `0.5/0.5`。

`lambda_modality_balance` 可对全图平均 alpha 添加均衡 KL，默认 0。

## 8. Self Residual、Prototype 与 LP Decoder

### Controlled self residual

```text
z_self = MLP_self([h_t,h_v])
beta = beta_max * sigmoid(MLP_gate([
  h_t,h_v,|h_t-h_v|,h_t*h_v,r_t,r_v,s_t,s_v,log_degree]))
z = z_struct + beta*z_self
```

core 默认关闭；full 和 LP preset 开启，默认最大系数 0.1。

### Modality-specific prototypes

实现了 `P_t/P_v/P_c`：

```text
z_proto_t = Attn(h_t,P_t)
z_proto_v = Attn(h_v,P_v)
z_proto_c = Attn((h_t+h_v)/2,P_c)
z_proto = alpha_t*z_proto_t + alpha_v*z_proto_v + z_proto_c
z = z + prototype_residual_max*z_proto
```

三组 prototype diversity loss 已实现。core/LP 默认关闭，full preset 开启。

支持两种初始化：

- `prototype_init=random`：Xavier 随机初始化，保持默认行为。
- `prototype_init=kmeans`：训练前对冻结输入特征进行采样，经过当前 text/visual
  projector 后，分别初始化 `P_t/P_v/P_c`。采样规模、迭代次数和投影 batch
  分别由 `prototype_init_max_samples`、`prototype_init_iters` 和
  `prototype_init_batch_size` 控制。

K-means 只执行一次，初始化完成标志进入 checkpoint；若绕过 trainer 直接调用
尚未初始化的模型，会显式报错而不会静默退回随机 prototype。

### LP decoder 的当前迁移策略

参考实现提供过：

```text
concat_product_abs = [z_i || z_j || z_i*z_j || |z_i-z_j|]
```

当前 MAP 项目没有迁移这一模型私有 decoder。`map_mag_v3`、`map_mag_v3_full` 和
`map_mag_v3_lp` 都与其他 encoder 一样，先经过统一 128 维投影，再使用
`pair=z_i*z_j` 的共享 Hadamard predictor。这样正式比较只改变 encoder，不同时
增加 v3 的 LP head 容量。`concat_product_abs` 只能在未来作为独立 decoder ablation
研究，不能与统一协议主表混用。

## 9. Auxiliary Loss 与 aux_info

```text
L_aux = lambda_modality_balance * L_balance
      + lambda_proto_diversity * L_proto
      + lambda_edge_reg * L_edge
```

core 三项默认均为 0。基础 `aux_info`：

```text
mean_r_text, mean_r_visual
mean_lambda_text, mean_lambda_visual
mean_edge_weight_text, std_edge_weight_text
mean_edge_weight_visual, std_edge_weight_visual
mean_edge_weight, std_edge_weight, min_edge_weight, max_edge_weight
mean_cos_text, mean_cos_visual
mean_alpha_text, mean_alpha_visual
mean_degree
modality_balance_loss, proto_loss, edge_reg_loss
```

按模块条件增加：

- conflict：`mean_conflict_weight_text/visual`、`mean_eta_conflict_text/visual`
- self residual：`mean_beta_self`
- prototypes：`mean_proto_residual`
- shared propagation：`mean_shared_edge_weight`

所有数值均为 scalar、finite，并在 `aux_info` 中 detach。

## 10. Node / Edge Aux Export

`node_aux_stats()` 提供：

```text
degree, s_text, s_visual, r_text, r_visual
lambda_text, lambda_visual
alpha_text, alpha_visual
beta_self, eta_conflict_text, eta_conflict_visual
text_visual_cosine
```

为兼容既有通用 CSV，还提供 `p_self/p_struct/p_proto/gamma` 兼容列。

NC/LP `edge_aux` 在原 v2 列基础上动态增加：

```text
edge_weight_text, edge_weight_visual, edge_weight_shared
```

同时保留 `src/dst`、两种 cosine、标签、同类标记和端点 degree。v1/v2 不返回
这些可选字段，因此旧 CSV 表头不变。

LP 在 `final_epoch` 和 `best_val` 自动导出（仍由 `export_edge_aux` 开关控制），
并增加 `positive_split`。MM-Graph LP 没有 node label 时，label 相关列留空。

## 11. 参考项目历史测试结果（非当前 MAP）

执行：

```bash
python -m py_compile src/models/map_mag_v3.py
conda run -n yhf_env python -c \
  'import src.models.map_mag_v3 as m; print(m.Model.__name__)'
conda run -n yhf_env python -m pytest tests/test_map_mag_v3.py -q
conda run -n yhf_env python -m pytest tests/test_inference_equivalence.py -q
conda run -n yhf_env python -m pytest -q
```

结果：

- import：`MAPMAGV3`
- v3 专项：`27 passed`
- LP sampling / decoder 专项：`5 passed`
- inference equivalence：`37 passed`
- 完整测试集：`73 passed`
- full/LP preset 均通过 Hydra 实际解析和数据加载检查

覆盖 factory、forward contract、2 种 propagation、5 种 edge mode、3 种
fusion、3 种 reliability、conflict/self/prototype 开关、辅助损失反向、空图、
full/layerwise 等价、K-means 一次性初始化、两种 LP pair operator 及 NC/LP
node/edge export。

## 12. 参考项目历史 Sanity 与已知问题（非当前 MAP）

Movies-NC，CPU，`hidden_dim=16,num_hops=1,epochs=1`：

- Val Acc：`1.11`
- Test Acc：`0.93`
- Test Macro-F1：`0.52`

Movies-LP，CPU，单 batch/epoch，256 条训练正边：

- Val MRR：`3.14`
- Test MRR：`3.34`
- Test Hits@10：`5.27`

以上仅用于链路 smoke，不代表模型质量。LP 日志确认继续使用现有 full-graph
positive label edge removal 分支。

完整化后的额外 CPU smoke：

- Movies-NC，`map_mag_v3_full`、K-means prototype、`hidden_dim=8`、
  单 epoch：Val Acc `7.17`，Test Acc `7.02`。
- Movies-LP，`map_mag_v3_lp`、`concat_product_abs`、`hidden_dim=8`、
  单 epoch/64 条正边：Val MRR `9.39`，Test MRR `9.54`。

同样只验证配置、初始化、训练、checkpoint 和评测全链路，不作为正式结果。

已知问题：

- `layerwise` 与 v1/v2 一样是 exact full-graph fallback，数值等价但不节省显存。
- full-graph LP 每个 edge-label batch 都要重新编码移除对应正边后的 message graph，
  在大图上成本较高；可用 `full_graph_training=false` 做采样近似。
- 当前运行环境无法连接 NVIDIA driver，因此附件中的三个 GPU 单 seed 正式实验未执行。

## 13. 参考项目原始实验建议

第一阶段先验证核心贡献：

```bash
# Core v3
python -m src.main dataset=Movies task=nc model=map_mag_v3 num_runs=1 seed=42 device=cuda:0

# 退化为 v2 风格共享传播
python -m src.main dataset=Movies task=nc model=map_mag_v3 num_runs=1 seed=42 device=cuda:0 \
  model.propagation_mode=shared_fused model.edge_weight_mode=shared_avg_cos

# 双图但固定 reliability 融合
python -m src.main dataset=Movies task=nc model=map_mag_v3 num_runs=1 seed=42 device=cuda:0 \
  model.modality_fusion_mode=reliability
```

第二阶段比较受控增强：

```bash
python -m src.main dataset=ele-fashion task=nc model=map_mag_v3_full num_runs=1 seed=42 device=cuda:0
python -m src.main dataset=sports-copurchase task=lp model=map_mag_v3_lp num_runs=1 seed=42 device=cuda:0
```

LP 应同时报告 `model.lp_pair_operator=hadamard` 与
`model.lp_pair_operator=concat_product_abs`，避免把 encoder 增益和 decoder
容量增益混在一起。Prototype 应比较 random/K-means；若仍无稳定增益，主论文保持
core 默认关闭，只作为完整消融模块。
