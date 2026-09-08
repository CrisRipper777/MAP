# MAP-MAG v2 Implementation Report

## 1. 新增/修改文件

- `src/models/map_mag_v2.py`
  - 新增实验性 `MAPMAGV2` encoder，实现 Single Reliability Fusion、Structure-Semantic Edge Reweighting、Low-pass Backbone、可选 High-pass Residual Correction、自残差和 shared prototype residual。
- `configs/model/map_mag_v2.yaml`
  - 新增 Hydra 模型配置，`name: map_mag_v2` 可直接通过现有 factory 动态导入。
- `src/tasks/common.py`
  - 扩展 `AUX_INFO_KEYS`，让 v2 的边权、模态融合、残差门等数值诊断能进入现有 trainer 日志。
- `tests/test_inference_equivalence.py`
  - 新增 v2 工厂构建、full/layerwise 推理一致、边权模式、结构模式、可选路径和空边图测试。
- `docs/map_mag_v2_implementation_report.md`
  - 本实现报告。

## 2. v2 架构说明

MAP-MAG v2 仍是任务无关的节点 encoder，兼容现有 NC/LP trainer 的输出格式：

```python
z, None, None, aux_loss, aux_info = model(x, edge_index)
```

整体流程：

1. 将 joint feature 拆成 `x_t, x_v`，并分别投影为 `h_t, h_v`。
2. 用邻居均值估计文本/图像可靠性 `r_t, r_v`。
3. 默认使用 single reliability fusion 得到基础表示 `h0`。
4. 在 `edge_index` 上计算语义边权 `w_ij`，不构造 dense adjacency。
5. 将 `w_ij` 传入 `gcn_norm(..., edge_weight=w_ij)`，进行 restart low-pass diffusion。
6. 可选启用 `lowpass_residual`，以小幅 high-pass residual correction 修正低通主干。
7. 可选启用 self residual 和 shared prototype residual。
8. 输出 `output_norm(output_mlp(z_input) + z_struct)`。

## 3. 与 v1 的关键区别

- v1 是 self/structure/prototype 三路径并列，再用 router 做路径融合；v2 默认取消并列路径竞争，以低通结构传播为主干。
- v1 默认 `reliability_mode=double`；v2 默认 `single`，避免弱模态被二次压制。
- v1 的 high/low 由 `gamma * z_low + (1 - gamma) * z_high` 对称竞争；v2 默认只用 `lowpass`，高频只在 `lowpass_residual` 中作为小幅残差校正。
- v1 prototype path 默认参与融合；v2 prototype 默认关闭，只保留 shared prototype residual 的可消融实现。
- v2 新增基于文本/图像语义一致性的 edge reweighting，并把边权直接交给 diffusion。

## 4. 新增配置项

核心默认配置位于 `configs/model/map_mag_v2.yaml`：

- `reliability_mode`: `single | double`
- `use_semantic_edge_weight`: 是否启用语义边权
- `edge_weight_mode`: `avg_cos | text_only | visual_only | reliability_aware | learnable_scalar`
- `edge_weight_min`: 边权下界
- `edge_weight_temperature`: sigmoid 温度
- `structure_mode`: `lowpass | lowpass_residual | original_v1_gamma`
- `residual_eta_max`: high-pass residual correction 最大系数
- `use_self_residual`: 是否启用 self residual
- `self_residual_max`: self residual 最大系数
- `use_prototype_path`: 是否启用 shared prototype residual
- `lambda_edge_reg`: 轻量边权正则，默认 0
- `lambda_proto`: prototype diversity loss 权重，默认 0
- `export_aux_stats`: v2 诊断导出总开关，映射到现有 `export_node_aux`

## 5. Semantic Edge Weight 公式

对每条边 `(i, j)` 只在稀疏 `edge_index` 上计算：

```text
cos_t_ij = cosine(h_t[i], h_t[j])
cos_v_ij = cosine(h_v[i], h_v[j])
w_ij = edge_weight_min + (1 - edge_weight_min) * sigmoid(sim_ij / edge_weight_temperature)
```

支持的 `sim_ij`：

- `avg_cos`: `(cos_t_ij + cos_v_ij) / 2`
- `text_only`: `cos_t_ij`
- `visual_only`: `cos_v_ij`
- `reliability_aware`: 以端点平均可靠性加权文本/图像相似性
- `learnable_scalar`: `a_t * cos_t_ij + a_v * cos_v_ij + b`

实现中对 cosine 和 edge weight 使用 `torch.nan_to_num`，并将边权 clamp 到 `[edge_weight_min, 1.0]`。

## 6. Low-pass Backbone 与 Residual Correction

Low-pass 主干：

```text
H^0 = h0
for k in range(num_hops):
    propagated = A_sem_norm H
    H = (1 - restart) * propagated + restart * h0
z_low = H
```

`structure_mode=lowpass` 时：

```text
z_struct = z_low
```

`structure_mode=lowpass_residual` 时：

```text
z_res = h0 - z_low
eta = residual_eta_max * sigmoid(MLP_eta([h0, s_t, s_v, log_degree, r_t, r_v]))
z_struct = z_low + eta * MLP_res(z_res)
```

`structure_mode=original_v1_gamma` 只用于诊断：

```text
z_struct = gamma * z_low + (1 - gamma) * (h0 - z_low)
```

## 7. Sanity Check 结果

运行环境使用项目 README 中的 `yhf_env`：

```bash
conda run -n yhf_env python -m py_compile src/models/map_mag_v2.py src/tasks/common.py
conda run -n yhf_env python -c "import src.models.map_mag_v2 as m; print(m.Model.__name__)"
conda run -n yhf_env python -m pytest tests/test_inference_equivalence.py -k 'map_mag_v2' -q
conda run -n yhf_env python -m pytest tests/test_inference_equivalence.py -q
conda run -n yhf_env python -m pytest -q
```

结果：

- `py_compile`: passed
- import: `MAPMAGV2`
- v2 targeted tests: `16 passed, 18 deselected`
- inference equivalence file: `34 passed`
- full test suite: `40 passed`

覆盖项：

- `model.name=map_mag_v2` factory 构建正常
- forward 输出 `z.shape == (N, hidden_dim)`
- `aux_loss` 是 scalar tensor
- `aux_info` 数值项无 NaN
- `use_semantic_edge_weight=true/false` 均可跑
- 5 种 `edge_weight_mode` 均可跑
- 3 种 `structure_mode` 均可跑
- `use_self_residual=true/false` 均可跑
- `use_prototype_path=true/false` 均可跑
- 空 `edge_index` 的边权统计不会产生 NaN

## 8. 最小实验结果

以下只是链路 smoke，不是正式性能对比。设置为 `hidden_dim=16, num_hops=1, num_runs=1, epochs=1, device=cpu`。

NC:

```bash
conda run -n yhf_env python -m src.main dataset=Movies task=nc model=map_mag_v2 \
  num_runs=1 seed=42 device=cpu task.epochs=1 task.patience=1 \
  model.hidden_dim=16 model.num_hops=1 model.num_layers=1 \
  model.full_graph_training=true model.export_aux_stats=false \
  hydra.run.dir=outputs/map_mag_v2_sanity_nc
```

Result:

- Val Acc: `31.43`
- Test Acc: `31.36`
- Test Macro-F1: `2.54`

LP:

```bash
conda run -n yhf_env python -m src.main dataset=Movies task=lp model=map_mag_v2 \
  num_runs=1 seed=42 device=cpu task.epochs=1 task.patience=1 \
  task.max_train_batches=1 task.train_pos_per_epoch=256 task.eval_edge_batch_size=4096 \
  model.hidden_dim=16 model.num_hops=1 model.num_layers=1 \
  model.full_graph_training=true model.export_aux_stats=false \
  hydra.run.dir=outputs/map_mag_v2_sanity_lp
```

Result:

- Val MRR: `5.37`
- Test MRR: `5.56`
- Test Hits@1: `0.93`
- Test Hits@3: `3.18`
- Test Hits@10: `11.78`

LP 仍复用现有 trainer：full-graph edge-label batches 会调用当前 `_exclude_positive_label_edges_from_message_graph`，不会把当前正标签边放入 message graph。

## 9. 已知问题

- v2 的 `inference()` 当前是 exact full-graph inference API fallback，会忽略 `batch_size`；这保证 full/layerwise 模式兼容，但不是省内存的 layerwise 实现。
- `use_prototype_path=true` 当前实现为 shared prototype residual 和 diversity loss；没有实现 modality-specific prototype。
- 现有 `prototype_aux` 导出逻辑是 v1 router/prototype 诊断专用，v2 默认不启用 `export_prototype_aux`。
- `export_aux_stats=true` 会走现有 node aux CSV 兼容列；v2 新增的 `lambda_text/lambda_visual/eta_residual/alpha_self` 已在 `node_aux_stats()` 中提供，但 CSV 导出仍是旧列集合。
- `full_graph_training=true` 是 v2 默认值，正式跑大图时需要注意 CPU/GPU 显存和 LP full-graph per-batch 成本。

## 10. 下一步建议

- 先跑小规模 ablation：`edge_weight_mode`、`edge_weight_min`、`edge_weight_temperature`、`structure_mode`。
- 优先比较 `lowpass` vs `lowpass_residual`，确认 residual correction 是否真的补充有效高频。
- 在 MAPB 四个数据集上先用 `hidden_dim=256,num_hops=2` 做 NC/LP 单 seed 验证，再扩展到 3 seeds。
- 如果 v2 在 LP 上 full-graph 成本过高，再为 v2 单独实现真正的 layerwise semantic-edge inference。
- 若 prototype residual 仍无收益，可以保持默认关闭并从论文主线中弱化。
