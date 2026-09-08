# ORED Evidence Ledger

本 ledger 是 ORED-MAG 的证据边界，不是新模型设计文档。内容只记录当前 MAP
仓库已有报告和 ORED-0 reference run 能支持的结论；不把失败实验重新包装为正面
证据，也不在本阶段实现任何 ORED 模块。

## SUPPORTED / LOCKED

1. **MAP-v2 shallow restart diffusion 是 strong performance source。** 现有
   `docs/map_mag_v2_performance_diagnosis.md` 的受控重训练对照显示，两跳、带
   restart 的 low-pass 主干是跨数据集最稳定的性能来源。ORED-0 的 Movies
   reference 也确认该主干可在冻结 NC 协议下正常运行。
2. **MAP-v2 semantic edge 有小幅真实信号。** 既有诊断中 semantic-uniform 相对
   lowpass-uniform 的方向总体为正但幅度较小；ORED-0 保留该变体作为 reference，
   不把它升级为已证明的主导机制。
3. **P0 Semantic Ownership 已在 `exp/oft-mag` 验证。** 来源为
   `CrisRipper777/exp` 的 `oft-mag` 分支，核心文件为
   `src/models/biaxis_p0.py` 与 `src/models/biaxis_components.py`。其已验证的
   因子为 `C`, `P_t`, `P_v`，并包含 common、orthogonality、reconstruction
   auxiliary losses。
4. **O1/O1.5 已支持 ownership states 可作为 graph states。** 该结论来自已有
   `exp/oft-mag` 研究记录；ORED-0 只登记其作为后续输入证据，不实现对应传播。

## CLOSED / NOT TO REINTRODUCE

以下方向在 ORED-MAG 主线中关闭，不能作为默认核心重新引入：

1. K4 topology relation
2. static cross-ownership transfer
3. node-specific dynamic cross routing
4. source-specific cross routing
5. operator bank on cross states
6. prototype path as default core
7. high-pass path as default core
8. complex modality router as default core

## OPEN

后续可研究但尚未被 ORED-0 证明的方向：

1. Joint-RD vs Ownership-RD
2. Ownership-conditioned Composition
3. Ownership-conditioned Exposure
4. Exposure × Composition
5. same-node cross-factor conditioning without cross-factor transport

## Guardrails

- ORED-0 没有实现 evidence scorer、Semantic Ownership propagation、Exposure、
  Composition、restart factor propagation 或新 loss。
- `docs/pard_mag_deep_research_proposal.md` 保持原样；PaRD-MAG 不属于本阶段实现。
- Ledger 中的 OPEN 项不代表已获准进入默认模型；进入 ORED-1 前必须保持当前
  NC protocol、split 和 checkpoint selection 不变。
