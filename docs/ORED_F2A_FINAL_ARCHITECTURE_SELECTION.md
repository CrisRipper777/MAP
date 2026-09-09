# ORED-F2A Final Architecture Selection

## Scope and frozen protocol

ORED-F2A completes the five-dataset validation-only architecture selection between
`f1_owner` and `f1_dual_direct`. It adds only `ele-fashion` and `Reddit-S`; the
Movies, Toys, and Grocery owner/direct results are reused unchanged from ORED-F1.

`NO TEST` · `NO TUNING` · `NO NEW MODULE` · `NO OCB FORMAL RERUN`

All formal runs use `task=nc`, `task.training_mode=full_graph`, protocol
`unified_full_graph_nc_v1`, seeds 42/43/44, validation-accuracy checkpoint
selection, and `task.evaluate_test=false`. No test labels or test metrics were
read or used for the decision.

## 1. Why Dual-Direct was selected from F1

F1 established `f1_dual_direct` as the provisional candidate: relative to the
ownership parent it gave `+0.047 pp` paired Acc and `+0.601 pp` paired Macro-F1
over the 9 matched seeds, with 6/9 positive paired Acc and 6/9 positive paired
Macro-F1 contrasts. It retains explicit semantic ownership while adding a joint
collaborative restart-diffusion context and ownership-specific projections.

## 2. Why OCB was demoted

The ownership-constrained compatibility bridge (`f1_dual_ocb`) was not carried
into F2A formal runs. F1 showed only `+0.021 pp` Acc and `-0.338 pp` Macro-F1
relative to Direct, so it did not provide a stable incremental benefit. It remains
secondary ablation evidence only.

## 3. Final provisional architecture

The selected architecture is **Ownership-Projected Residual Collaboration**,
consisting of an ownership-preserving diffusion branch, a joint collaborative
restart-diffusion branch, ownership-specific projections, and a bounded residual
bridge. The five-dataset selection below freezes it as the final ORED core.

## 4. New ele-fashion / Reddit-S results

The 12 new formal runs are validation-only:

| Dataset | Variant | Val Acc % | Val Macro-F1 % |
|---|---|---:|---:|
| ele-fashion | Owner | 87.24±0.20 | 74.32±0.34 |
| ele-fashion | Dual-Direct | 87.19±0.22 | 73.82±0.45 |
| Reddit-S | Owner | 95.82±0.12 | 91.67±0.22 |
| Reddit-S | Dual-Direct | 95.83±0.12 | 91.92±0.18 |

## 5. Merged five-dataset results

Values are validation percentages, mean±population SD over seeds 42/43/44.

| Dataset | Variant | Val Acc % | Val Macro-F1 % |
|---|---|---:|---:|
| Movies | Owner | 57.55±0.25 | 49.80±1.08 |
| Movies | Dual-Direct | 57.51±0.38 | 51.27±0.62 |
| Toys | Owner | 80.12±0.17 | 77.29±0.54 |
| Toys | Dual-Direct | 80.23±0.21 | 77.64±0.24 |
| Grocery | Owner | 82.88±0.20 | 74.89±0.47 |
| Grocery | Dual-Direct | 82.96±0.32 | 74.88±0.76 |
| ele-fashion | Owner | 87.24±0.20 | 74.32±0.34 |
| ele-fashion | Dual-Direct | 87.19±0.22 | 73.82±0.45 |
| Reddit-S | Owner | 95.82±0.12 | 91.67±0.22 |
| Reddit-S | Dual-Direct | 95.83±0.12 | 91.92±0.18 |

## 6. Dual vs Owner paired analysis

| Dataset | ΔAcc pp | ΔMacro-F1 pp | Positive Acc seeds | Positive F1 seeds |
|---|---:|---:|---:|---:|
| Movies | -0.04±0.16 | +1.47±1.31 | 1/3 | 3/3 |
| Toys | +0.10±0.05 | +0.35±0.40 | 3/3 | 2/3 |
| Grocery | +0.08±0.15 | -0.01±1.10 | 2/3 | 1/3 |
| ele-fashion | -0.05±0.03 | -0.50±0.12 | 0/3 | 0/3 |
| Reddit-S | +0.01±0.01 | +0.25±0.21 | 1/3 | 2/3 |

Across all 15 paired seed comparisons, Dual-Direct has
`+0.020±0.117 pp` Acc and `+0.312±1.025 pp` Macro-F1, with 7/15 positive Acc
seeds and 8/15 positive Macro-F1 seeds. Positive dataset means are 3/5 for Acc
and 3/5 for Macro-F1.

The small ele-fashion Macro-F1 decline is not systematic across datasets, and its
Acc contrast remains far above the severe-regression threshold.

## 7. MAP/DiP-family validation positioning

The following comparison reads only the Val Accuracy column from
`docs/nc_benchmark_results.md`.

| Dataset | Dual Val Acc % | Best family model | Best family Val Acc % | Dual gap pp |
|---|---:|---|---:|---:|
| Movies | 57.51 | map_mag_v3 | 57.13 | +0.38 |
| Toys | 80.23 | dip | 80.69 | -0.46 |
| Grocery | 82.96 | dip | 83.70 | -0.74 |
| ele-fashion | 87.19 | map_mag | 88.20 | -1.01 |
| Reddit-S | 95.83 | dip | 96.21 | -0.39 |

Five-dataset macro Val Accuracy:

| Reference | Mean Val Acc % | Dual minus reference pp |
|---|---:|---:|
| Dual-Direct | 80.7425 | — |
| dip | 80.9594 | -0.2169 |
| map_mag | 80.4363 | +0.3062 |
| map_mag_v2 | 80.8712 | -0.1287 |
| map_mag_v3 | 80.8441 | -0.1015 |

These are performance-position diagnostics, not test-based architecture claims.

## 8. Stability

All 30 merged formal runs produced finite validation metrics, validation
checkpoints, the expected protocol/configuration, and no NaN/Inf/OOM/traceback
marker in the captured training logs: 30/30 stable. Owner and Dual preserve
parameter parity on every dataset.

## 9. Final architecture decision

The registered rule evaluates five-dataset macro Acc, Macro-F1 decline, severe
dataset-level Acc regressions, and stability:

- macro ΔAcc must be at least `-0.20 pp`;
- Macro-F1 must not show a systematic material decline; here this is operationalized
  as macro ΔF1 ≥ `-0.20 pp` and no majority of dataset means being negative;
- fewer than two datasets may have mean ΔAcc ≤ `-0.80 pp`;
- all formal training must be stable and finite.

Measured macro changes are `+0.0204 pp` Acc and `+0.3122 pp` Macro-F1. Zero
datasets have mean ΔAcc ≤ `-0.80 pp`. The decision is **FREEZE_DUAL_FINAL**.

## 10. Frozen modules

Freeze **ORED-MAG** with:

1. Semantic Ownership Decomposition
2. Dual-Granularity Relational Diffusion
3. Ownership-Projected Residual Collaboration

## 11. Rejected/non-core modules

- OCB compatibility gate: **NOT SELECTED**; no stable incremental benefit.
- Composition: **CLOSED**.
- Exposure: **NOT PURSUED**.
- No new loss, prototype, high-pass path, vector gate, cross-factor transport, or
  additional router is part of this freeze.

## 12. Next formal-Test step

The next step is a separately authorized formal Test evaluation of the frozen
ORED-MAG configuration. F2A itself opened no Test data and ran no OCB rerun.

## Artifacts

- `experiments/ored/f2a/f2a_new_runs.csv`
- `experiments/ored/f2a/f2a_five_dataset_summary.csv`
- `experiments/ored/f2a/f2a_paired_contrasts.csv`
- `experiments/ored/f2a/f2a_benchmark_val_comparison.csv`
- `experiments/ored/f2a/f2a_summary.json`
