# ORED-F1 Dual-Granularity Collaborative Diffusion Prototype

## 1. Architecture motivation

ORED-F1 tests whether a strong ownership-preserving graph parent benefits from a second, joint relational context. The ownership branch keeps semantic ownership explicit; the joint branch provides one graph-context stream that can be projected back into each ownership state.

The central design is deliberately small: ownership diffusion, joint diffusion, ownership-specific projections, a shared compatibility gate, and a single scalar bridge strength. No new task loss is introduced.

## 2. Relation to ORED-2 findings

ORED-2 established uniform two-hop restart diffusion on `C`, `Pt`, and `Pv` as a viable strong parent. F1 therefore retains `num_hops=2`, `restart=0.15`, self-loops, the existing SemanticFactorizer, the existing P0 common/orth/reconstruction losses, and the existing late fusion/output refinement block.

## 3. Why Composition was closed

ORED-3A learned edge composition and ownership-specific edge weighting but did not obtain a stable downstream accuracy gain. Composition is closed for this stage and is not used by any F1 variant. F1 studies same-node dual-granularity context, not semantic edge reweighting or cross-factor transport.

## 4. Three matched variants

- `f1_owner`: ownership diffusion only; effective bridge strength is exactly zero.
- `f1_dual_direct`: ownership diffusion plus the projected joint stream with all gates fixed to one.
- `f1_dual_ocb`: ownership diffusion plus the projected joint stream with the learned ownership-conditioned compatibility gate.

All three instantiate the complete module graph, including the joint branch, projections, ownership embedding, gate MLP, bridge parameter, factor bridge norms, late fusion, output MLP and output norm. The variant only controls which paths affect the output.

## 5. Dual-granularity data flow

```text
C0, Pt0, Pv0 ── ownership restart diffusion ──> CG, PtG, PvG
       │
       └─ concat ─> JointFusion ─> Joint restart diffusion ─> JG
                                                        │
                                  W_C, W_Pt, W_Pv ─────┘
                                                        │
                         compatibility gates ─> bridge ─┘
                                                        │
                 C*, Pt*, Pv* ─> LateFusion ─> OutputMLP/Norm ─> z
```

Both graph streams use the original uniform GCN-normalized graph. Ownership payloads never move between factors.

## 6. OCB formula

For ownership factor `b`, let `H_b^G` be the ownership branch state and `G_b=W_b(J^G)` the projected joint state. The shared gate MLP receives:

```text
[H_b^G || G_b || |H_b^G-G_b| || H_b^G*G_b || ownership_embedding[b]]
```

and returns `g_b=sigmoid(MLP(...))`. The bridge is:

```text
alpha = 0.5 * tanh(bridge_raw)
H_b* = H_b^G + alpha * g_b * BridgeNorm_b(G_b)
```

`f1_dual_direct` fixes `g_b=1`; `f1_owner` fixes `alpha_effective=0`.

## 7. Parent-preserving initialization

`bridge_raw=0` at initialization, hence `alpha=0` and all three variants produce the same representation under identical weights in evaluation mode. The update is multiplied by alpha before it is added to the ownership state, so the bridge norms remain bypassable at initialization.

## 8. Parameter parity

| Dataset | Model params | Head params | Total params | Parity |
|---|---|---|---|---|
| Movies | 1436594 | 5140 | 1441734 | True |
| Toys | 1436594 | 4626 | 1441220 | True |
| Grocery | 1436594 | 5140 | 1441734 | True |

Construction order is matched and the complete module/state-dict manifest is checked by the test suite. The classifier head varies only with dataset class count and is also matched across variants.

## 9. Tests

`tests/test_ored_mag.py` covers the previous ORED-1/ORED-2/ORED-3A tests plus F1 parameter/state-dict parity, exact zero-bridge parity, owner joint/bridge isolation, direct gate isolation, OCB gate effect, graph sensitivity, factor-local ownership diffusion, projection shapes, gate/alpha bounds, finite forward/backward, bridge and gate gradients, and inference parity.

## 10. Movies smoke

The smoke is Val-only, `task.evaluate_test=false`, seed 42, and is not used for tuning. It is stored under `outputs/ored/f1/smoke/` and summarized here:

| Variant | Val Acc % | Val F1 % | alpha | gate C/Pt/Pv | joint update | bridge C/Pt/Pv |
|---|---|---|---|---|---|---|
| f1_owner | 57.47 | 48.46 | 0.0000 | 1.0000/1.0000/1.0000 | 0.4900 | 0.0000/0.0000/0.0000 |
| f1_dual_direct | 57.35 | 49.40 | 0.0139 | 1.0000/1.0000/1.0000 | 0.5245 | 0.0476/0.0881/0.0567 |
| f1_dual_ocb | 57.86 | 49.83 | 0.0185 | 0.6043/0.6073/0.5428 | 0.5198 | 0.0399/0.0720/0.0417 |

## 11. 3 datasets × 3 seeds results

All values are validation percentages, mean±population SD over seeds 42/43/44. No test labels or metrics were used.

| Dataset | Variant | Val Acc % | Val Macro-F1 % |
|---|---|---|---|
| Movies | f1_owner | 57.55±0.25 | 49.80±1.08 |
| Movies | f1_dual_direct | 57.51±0.38 | 51.27±0.62 |
| Movies | f1_dual_ocb | 57.58±0.05 | 50.28±0.89 |
| Toys | f1_owner | 80.12±0.17 | 77.29±0.54 |
| Toys | f1_dual_direct | 80.23±0.21 | 77.64±0.24 |
| Toys | f1_dual_ocb | 80.17±0.20 | 77.54±0.21 |
| Grocery | f1_owner | 82.88±0.20 | 74.89±0.47 |
| Grocery | f1_dual_direct | 82.96±0.32 | 74.88±0.76 |
| Grocery | f1_dual_ocb | 83.01±0.26 | 74.95±0.83 |

## 12. Contrasts A/B/C

| Contrast | ΔAcc pp | ΔF1 pp | + Acc seeds/9 | + F1 seeds/9 |
|---|---|---|---|---|
| A_dual_direct_minus_owner | 0.05±0.14 | 0.60±1.19 | 6 | 6 |
| B_dual_ocb_minus_direct | 0.02±0.25 | -0.34±1.08 | 3 | 3 |
| C_dual_ocb_minus_owner | 0.07±0.22 | 0.26±0.86 | 4 | 6 |

- A (`dual_direct - owner`) measures the value of dual-granularity relational context without compatibility gating.
- B (`dual_ocb - dual_direct`) measures the value of the ownership-constrained bridge relative to direct collaboration.
- C (`dual_ocb - owner`) measures the complete F1 prototype relative to the strong ownership parent.

Paired seed rows are in `experiments/ored/f1/f1_paired_contrasts.csv`; no significance test is performed.

## 13. Alpha/gate diagnostics

Alpha, gate means and gate standard deviations are in `f1_bridge_diagnostics.json`. The full formal diagnostic table is:

| Run | alpha | gate means C/Pt/Pv | gate SD C/Pt/Pv | joint update | bridge update C/Pt/Pv |
|---|---|---|---|---|---|
| Movies/seed42/f1_owner | 0.0000 | 1.000/1.000/1.000 | 0.000/0.000/0.000 | 0.4909 | 0.0000/0.0000/0.0000 |
| Movies/seed42/f1_dual_direct | 0.0140 | 1.000/1.000/1.000 | 0.000/0.000/0.000 | 0.5219 | 0.0495/0.0906/0.0564 |
| Movies/seed42/f1_dual_ocb | 0.0201 | 0.585/0.615/0.530 | 0.160/0.091/0.170 | 0.5287 | 0.0435/0.0836/0.0456 |
| Movies/seed43/f1_owner | 0.0000 | 1.000/1.000/1.000 | 0.000/0.000/0.000 | 0.5031 | 0.0000/0.0000/0.0000 |
| Movies/seed43/f1_dual_direct | 0.0185 | 1.000/1.000/1.000 | 0.000/0.000/0.000 | 0.5323 | 0.0762/0.1270/0.0815 |
| Movies/seed43/f1_dual_ocb | 0.0230 | 0.727/0.694/0.595 | 0.114/0.087/0.151 | 0.5376 | 0.0695/0.1120/0.0624 |
| Movies/seed44/f1_owner | 0.0000 | 1.000/1.000/1.000 | 0.000/0.000/0.000 | 0.4803 | 0.0000/0.0000/0.0000 |
| Movies/seed44/f1_dual_direct | 0.0133 | 1.000/1.000/1.000 | 0.000/0.000/0.000 | 0.5051 | 0.0482/0.0817/0.0501 |
| Movies/seed44/f1_dual_ocb | 0.0128 | 0.666/0.619/0.623 | 0.034/0.042/0.050 | 0.5022 | 0.0300/0.0474/0.0301 |
| Toys/seed42/f1_owner | 0.0000 | 1.000/1.000/1.000 | 0.000/0.000/0.000 | 0.4242 | 0.0000/0.0000/0.0000 |
| Toys/seed42/f1_dual_direct | -0.0147 | 1.000/1.000/1.000 | 0.000/0.000/0.000 | 0.4359 | 0.0450/0.0635/0.0529 |
| Toys/seed42/f1_dual_ocb | -0.0041 | 0.593/0.554/0.474 | 0.030/0.027/0.057 | 0.4078 | 0.0074/0.0094/0.0066 |
| Toys/seed43/f1_owner | 0.0000 | 1.000/1.000/1.000 | 0.000/0.000/0.000 | 0.4144 | 0.0000/0.0000/0.0000 |
| Toys/seed43/f1_dual_direct | -0.0097 | 1.000/1.000/1.000 | 0.000/0.000/0.000 | 0.4299 | 0.0301/0.0388/0.0333 |
| Toys/seed43/f1_dual_ocb | -0.0096 | 0.713/0.611/0.618 | 0.069/0.050/0.126 | 0.4268 | 0.0212/0.0230/0.0204 |
| Toys/seed44/f1_owner | 0.0000 | 1.000/1.000/1.000 | 0.000/0.000/0.000 | 0.4198 | 0.0000/0.0000/0.0000 |
| Toys/seed44/f1_dual_direct | 0.0215 | 1.000/1.000/1.000 | 0.000/0.000/0.000 | 0.4411 | 0.0663/0.0956/0.0774 |
| Toys/seed44/f1_dual_ocb | 0.0286 | 1.000/0.999/0.999 | 0.000/0.002/0.002 | 0.4477 | 0.0888/0.1298/0.1048 |
| Grocery/seed42/f1_owner | 0.0000 | 1.000/1.000/1.000 | 0.000/0.000/0.000 | 0.4783 | 0.0000/0.0000/0.0000 |
| Grocery/seed42/f1_dual_direct | 0.0212 | 1.000/1.000/1.000 | 0.000/0.000/0.000 | 0.4936 | 0.0769/0.1005/0.0931 |
| Grocery/seed42/f1_dual_ocb | 0.0259 | 1.000/0.996/0.987 | 0.001/0.006/0.024 | 0.4952 | 0.0955/0.1193/0.1106 |
| Grocery/seed43/f1_owner | 0.0000 | 1.000/1.000/1.000 | 0.000/0.000/0.000 | 0.4869 | 0.0000/0.0000/0.0000 |
| Grocery/seed43/f1_dual_direct | -0.0234 | 1.000/1.000/1.000 | 0.000/0.000/0.000 | 0.5017 | 0.0927/0.1204/0.1028 |
| Grocery/seed43/f1_dual_ocb | -0.0187 | 0.991/0.975/0.920 | 0.009/0.023/0.110 | 0.4984 | 0.0730/0.0917/0.0739 |
| Grocery/seed44/f1_owner | 0.0000 | 1.000/1.000/1.000 | 0.000/0.000/0.000 | 0.4854 | 0.0000/0.0000/0.0000 |
| Grocery/seed44/f1_dual_direct | 0.0318 | 1.000/1.000/1.000 | 0.000/0.000/0.000 | 0.5051 | 0.1148/0.1753/0.1503 |
| Grocery/seed44/f1_dual_ocb | 0.0337 | 1.000/0.999/0.999 | 0.000/0.001/0.001 | 0.5042 | 0.1206/0.1815/0.1539 |

## 14. Joint/bridge update magnitude

The joint update ratio is `mean(||JG-J0||/(||J0||+eps))`. Each bridge update ratio is `mean(||H_b*-H_b^G||/(||H_b^G||+eps))`. These are descriptive representation diagnostics and use no labels.

## 15. Performance-vs-complexity

All F1 variants are matched in parameter count by construction. Relative to the owner control, direct and OCB activate an additional joint diffusion and three hidden-to-factor projections, plus compatibility gating in OCB. The parameter manifest records model/head/total counts; runtime and peak allocated GPU memory are recorded per run in `f1_bridge_diagnostics.json`.

## 16. GO/HOLD

The registered promotion rule is applied to OCB relative to owner: macro ΔAcc at least -0.20pp, no operationalized systematic Macro-F1 decline, at least one dataset with ΔAcc at least +0.30pp or ΔF1 at least +0.50pp, stable training, nonzero bridge, and nontrivial joint update. The automated result is **HOLD_REVIEW**.

Detailed rule fields are in `f1_summary.json`. `STRONG_GO` additionally requires positive macro Acc and Macro-F1 and positive Acc means on at least two of three datasets. `HOLD_BRIDGE` is reserved for macro ΔAcc below -0.50pp with at most one positive dataset mean.

## 17. Recommended F2 final architecture

If the verdict is `STRONG_GO` or `GO_TO_F2`, carry the OCB prototype forward as the F2 candidate while keeping the parent-preserving scalar bridge, uniform ownership diffusion, and no new auxiliary loss. If the verdict is `HOLD_REVIEW` or `HOLD_BRIDGE`, retain `f1_owner` as the safe parent and do not claim that the bridge is supported. Here OCB is slightly positive relative to owner but below the headroom rule, while OCB versus direct has a small positive Acc and negative Macro-F1 contrast; both are reported for researcher review rather than triggering an automatic structural rewrite.

## 18. Exact commands

```bash
# Movies seed42 Val-only smoke
PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python scripts/run_ored_f1.py --gpu 0 --shard 0 --shards 2 --smoke
PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python scripts/run_ored_f1.py --gpu 1 --shard 1 --shards 2 --smoke

# Formal 3 datasets × 3 seeds × 3 variants, Val-only
PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python scripts/run_ored_f1.py --gpu 0 --shard 0 --shards 2
PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python scripts/run_ored_f1.py --gpu 1 --shard 1 --shards 2

# Aggregate results and diagnostics
PYTHONPATH=. /home/m3/miniconda3/envs/yhf_env/bin/python scripts/summarize_ored_f1.py --device cuda:0
```

## 19. Scope

`NO TEST` · `NO EXPOSURE` · `NO COMPOSITION` · `NO TUNING` · `NO LP` · `NO CROSS-FACTOR MESSAGE PASSING` · `NO NEW LOSS`.
