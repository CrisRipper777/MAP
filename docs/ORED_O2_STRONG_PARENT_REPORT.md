# ORED-MAG Stage ORED-2: Strong Parent Construction

## 1. Stage objective

ORED-2 isolates one question: after the ORED-1 Semantic Ownership factorization
`C, P_t, P_v`, should MAP-v2 shallow restart diffusion be applied after fusion
(`JOINT-RD`) or factor-wise before late fusion (`OWNERSHIP-RD`)?

The four registered variants were:

1. `p0`
2. `p0_refine`
3. `joint_rd`
4. `ownership_rd`

The only graph operator is uniform two-hop restart diffusion with `restart=0.15`
and self-loops. No Test labels or metrics were read.

## 2. Starting SHA/tag

- Repository: `CrisRipper777/MAP/MAP`
- Branch: `ORED`
- Starting SHA: `82f1a66`
- Starting tag: `ored-o1-semantic-ownership`
- Starting worktree: preserved the pre-existing unrelated deletion of
  `docs/pard_mag_deep_research_proposal.md` and untracked
  `docs/lp_benchmark_results.md`.

## 3. Files changed

- `src/models/ored_components.py`: MAP-v2-equivalent `RestartDiffusion`, one-time
  normalized adjacency helper, and factor-wise application.
- `src/models/ored_mag.py`: four variants, shared matched-control refinement,
  exact graph inference, and `encode_ored_states()` diagnostics API.
- `configs/model/ored_mag.yaml`: variant selector and fixed diffusion settings.
- `src/tasks/nc.py`: records Val Macro-F1, best epoch, and early-stop epoch in the
  existing result artifact; training and checkpoint selection are unchanged.
- `tests/test_ored_mag.py`: 20 ORED tests covering the ORED-1 baseline and ORED-2.
- `scripts/run_ored_o2.py`: fixed 36-run paired replication grid.
- `scripts/summarize_ored_o2.py`: artifacts, contrasts, parameter manifest, gates.
- `scripts/ored_o2_diagnostics.py`: Movies/seed42 representation diagnostics.
- `docs/ORED_EVIDENCE_LEDGER.md`: ORED-1 lock and ORED-2 evidence updates.
- `docs/ORED_O2_STRONG_PARENT_REPORT.md`: this report.

## 4. Four matched variants

`p0` is unchanged: factorizer → original P0 fusion → `z_local`.

`p0_refine` adds only:

```text
u = Fusion([C || Pt || Pv])
z = LayerNorm(u + OutputMLP(u))
```

where `OutputMLP` is exactly `Linear → LayerNorm → ReLU → Dropout → Linear`.
It ignores `edge_index`.

`joint_rd` is:

```text
C, Pt, Pv → P0 Fusion → joint0
joint2 = RestartDiffusion(joint0, edge_index, hops=2, restart=.15)
z = LayerNorm(joint2 + OutputMLP(joint2))
```

`ownership_rd` is:

```text
[C, Pt, Pv] → one normalized adjacency → factor-wise diffusion
[C2 || Pt2 || Pv2] → same Fusion → u
z = LayerNorm(u + OutputMLP(u))
```

The ownership propagation uses `H=[C,Pt,Pv]` with shape `[N,3,128]`; scatter is
only over the node dimension. There are no factor-specific projections, edge
weights, restart values, cross-factor messages, or learnable diffusion parameters.

## 5. Exact computation and MAP-v2 diffusion parity

The helper uses the same call as `MAPMAGV2._diffuse()`:

```text
gcn_norm(edge_index,
         edge_weight=None,
         num_nodes=N,
         improved=False,
         add_self_loops=True,
         flow="source_to_target",
         dtype=h.dtype)
```

Each hop is:

```text
propagated = scatter(h[src] * norm_weight, dst, reduce="sum")
h = (1 - .15) * propagated + .15 * h0
```

`test_restart_diffusion_matches_map_v2_diffuse` passed with `allclose`; zero hops
returns identity and an empty graph with self-loops returns identity.

## 6. Parameter parity

The three matched controls have identical model parameters, head parameters, and
total parameters within every dataset. Diffusion contributes zero parameters.

| variant | model params | head params, Movies/Grocery | total, Movies/Grocery | head params, Toys | total, Toys |
|---|---:|---:|---:|---:|---:|
| p0 | 1,037,568 | 5,140 | 1,042,708 | 4,626 | 1,042,194 |
| p0_refine | 1,170,176 | 5,140 | 1,175,316 | 4,626 | 1,174,802 |
| joint_rd | 1,170,176 | 5,140 | 1,175,316 | 4,626 | 1,174,802 |
| ownership_rd | 1,170,176 | 5,140 | 1,175,316 | 4,626 | 1,174,802 |

Manifest: `experiments/ored/o2/o2_parameter_parity.json`.

## 7. Unit tests

```text
conda run --no-capture-output -n yhf_env python -m pytest tests/ -q
117 passed, 1 warning
```

The ORED-specific file passes `20/20`. Coverage includes all existing P0 checks,
MAP-v2 diffusion parity, no-graph equivalence, graph sensitivity, no cross-factor
transport, no learnable graph parameters, finite forward/aux/backward, and exact
graph inference parity.

## 8. ORED-2A Movies seed42 smoke

All four runs were Val-only with the fixed unified NC protocol. The smoke run used
the best checkpoint selected by validation Accuracy.

| variant | Val Acc | Val Macro-F1 | best epoch | early-stop epoch | total params |
|---|---:|---:|---:|---:|---:|
| p0 | 51.83% | 40.00% | 71 | 101 | 1,042,708 |
| p0_refine | 51.50% | 40.28% | 53 | 83 | 1,175,316 |
| joint_rd | 56.78% | 49.98% | 123 | 153 | 1,175,316 |
| ownership_rd | 57.35% | 50.14% | 93 | 123 | 1,175,316 |

P0 matches the ORED-1 Movies/seed42 reference (`51.83%` Acc, `40.00%` F1), so
there is no P0 regression above `0.3pp` Acc. The small difference between smoke and
formal same-seed graph results is consistent with the existing CUDA scatter atomic
reduction caveat; it is not treated as bitwise determinism.

## 9. ORED-2A representation/update diagnostics

Diagnostics are from the O2A Movies/seed42 best checkpoints and contain no Test
labels or metrics.

| state/update | mean norm or ratio | cosine | effective rank before → after |
|---|---:|---:|---:|
| P0-REFINE `z_local` → `z_final` | 10.6775 → 16.2174; relative update 1.0092 | — | 22.2516 → 18.5385 |
| JOINT `joint0` → `joint2` | ratio 0.5129 | 0.8912 | 26.9617 → 11.0976 |
| Ownership `C0` → `C2` | ratio 0.3703 | 0.9566 | 5.1990 → 2.7715 |
| Ownership `Pt0` → `Pt2` | ratio 0.5359 | 0.8756 | 7.9267 → 6.9427 |
| Ownership `Pv0` → `Pv2` | ratio 0.5579 | 0.8717 | 23.6305 → 17.6175 |

All states are finite and all graph updates are nonzero. No new rank-1 collapse is
present; the lowest post-diffusion effective rank is `2.77` for `C2`.

Artifact: `experiments/ored/o2/o2_movies_seed42_diagnostics.json`.

## 10. ORED-2B formal replication

The fixed grid completed `3 datasets × 3 seeds × 4 variants = 36 runs`:

| dataset | p0 Acc/F1 | p0-refine Acc/F1 | joint-rd Acc/F1 | ownership-rd Acc/F1 |
|---|---:|---:|---:|---:|
| Movies | 52.23±0.37 / 38.46±1.34 | 52.05±0.43 / 38.81±2.75 | 56.84±0.21 / 48.10±0.88 | **57.19±0.20 / 50.20±0.40** |
| Toys | 74.20±0.16 / 70.76±0.38 | 74.40±0.25 / 70.79±0.60 | **80.27±0.19 / 77.18±0.49** | 80.22±0.05 / **77.56±0.37** |
| Grocery | 78.26±0.15 / 69.57±0.28 | 77.41±0.16 / 67.44±0.80 | **82.72±0.06 / 75.10±1.04** | 82.62±0.13 / 73.95±0.83 |

Values are mean±population SD over seeds 42/43/44, in percent. Formal artifacts:
`o2_all_runs.csv` and `o2_dataset_summary.csv`.

## 11. Paired contrasts A–D

Each cell is `mean ΔAcc pp ± SD / mean ΔF1 pp ± SD`, followed by positive-seed
counts for Acc/F1.

| contrast | Movies | Toys | Grocery | macro mean ΔAcc / ΔF1 |
|---|---:|---:|---:|---:|
| A `p0_refine - p0` | -0.18±0.12 / +0.36±1.78; 0/3,2/3 | +0.19±0.29 / +0.03±0.53; 2/3,2/3 | -0.85±0.19 / -2.13±0.52; 0/3,0/3 | **-0.28 / -0.58** |
| B `joint_rd - p0_refine` | +4.79±0.49 / +9.29±3.41; 3/3,3/3 | +5.87±0.09 / +6.39±0.43; 3/3,3/3 | +5.31±0.17 / +7.66±1.01; 3/3,3/3 | **+5.32 / +7.78** |
| C `ownership_rd - p0_refine` | +5.14±0.46 / +11.39±2.44; 3/3,3/3 | +5.82±0.20 / +6.77±0.48; 3/3,3/3 | +5.20±0.12 / +6.51±0.04; 3/3,3/3 | **+5.39 / +8.22** |
| D `ownership_rd - joint_rd` | +0.35±0.03 / +2.10±1.27; 3/3,3/3 | -0.05±0.15 / +0.38±0.12; 1/3,3/3 | -0.11±0.09 / -1.15±1.00; 0/3,0/3 | **+0.06 / +0.44** |

Paired per-seed rows and paired mean/SD are in `o2_paired_contrasts.csv` and
`o2_summary.json`. No n=3 significance test was performed.

## 12. Joint-RD vs Ownership-RD interpretation

Ownership-RD is performance-viable: its macro mean Acc is slightly higher than
Joint-RD and its macro mean Macro-F1 is also higher. The advantage is concentrated
in Movies; Toys is essentially neutral and Grocery is slightly worse, especially in
Macro-F1.

The honest interpretation is **ownership-preserving factor-wise contextualization
versus pre-fused joint contextualization**. Uniform diffusion is a linear node
mixing operation, but Fusion is nonlinear and appears before diffusion in Joint-RD
and after diffusion in Ownership-RD. This experiment does not show that merely
splitting an identical linear operator into three tensors creates a new function.

## 13. Available MAP strong-reference comparison

The only comparable saved Val reference is ORED-0's MAP-v2 `lowpass_uniform`
Movies/seed42 run: `58.0684%` Val Acc and `49.86%` Val Macro-F1.

Formal Ownership-RD Movies/seed42 is `57.0486%` Acc and `50.6721%` Macro-F1:

- Acc gap: `-1.02pp` (a 1.02pp shortfall; conditional band).
- Macro-F1 gap: `+0.81pp`.

For Toys and Grocery, the existing MAP documents do not contain a comparable
`lowpass_uniform` Val reference. Existing Test tables and non-matched MAP-v2
references were not used.

## 14. Gate 1 verdict — graph computation

**GO_GRAPH.**

- Joint-RD − P0-REFINE: macro mean `+5.32pp` Acc; all 3 dataset means positive.
- Ownership-RD − P0-REFINE: macro mean `+5.39pp` Acc; all 3 dataset means positive.

Both exceed the preregistered `+1.5pp` threshold. Diagnostics and unit tests also
confirm nonzero finite graph updates.

## 15. Gate 2 verdict — ownership-preserving processing

**GO / VIABLE, not STRONG_GO.**

Ownership-RD − Joint-RD is macro mean `+0.06pp` Acc and `+0.44pp` Macro-F1.
Only 1/3 dataset mean Acc contrasts are positive, so the `+0.30pp` and 2/3
positive-dataset STRONG_GO rule is not met. However, the Acc delta is above
`-0.30pp`, no 2/3 dataset has a stable `≤-0.5pp` disadvantage, and Macro-F1 has
no systematic macro decline. Ownership-preserving propagation is therefore viable
as a later evidence-calibration parent, without claiming superiority.

## 16. Gate 3 verdict — absolute parent quality

**CONDITIONAL.** The available Movies reference shows a 1.02pp Val-Acc shortfall,
which is above the `1.0pp` acceptable band but within the `1.5pp` conditional band.
There is no comparable Val reference for Toys or Grocery. This stage does not add
an unregistered MAP rerun to fill that gap.

## 17. Failure analysis

- Refinement alone is not a universal gain: macro mean A is `-0.28pp` Acc and
  `-0.58pp` Macro-F1, with Grocery notably negative.
- Graph computation is effective in both placements; the ORED scaffold did not
  lose the MAP-v2 graph benefit.
- Ownership-RD is not uniformly better than Joint-RD: Grocery loses `0.11pp` Acc
  and `1.15pp` Macro-F1 on average, while Movies gains `0.35pp` Acc.
- Ownership factor diffusion changes all three streams substantially but remains
  finite and does not collapse them to rank one.
- CUDA scatter reductions are not claimed bitwise deterministic; all comparisons
  use the same MAP protocol and paired dataset/seed design.

## 18. Evidence Ledger update

`docs/ORED_EVIDENCE_LEDGER.md` now records:

- P0 Semantic Ownership migration into MAP = **LOCKED FOR ORED DEVELOPMENT**,
  including architecture/state_dict parity, exact topology invariance, factor
  health, Acc parity, and the protocol-qualified Macro-F1 decision.
- Stable restart diffusion on Semantic Ownership representation = **SUPPORTED IN
  ORED**.
- Ownership-preserving factor-wise contextualization is **performance-viable**, but
  superiority over pre-fused Joint-RD is **NOT YET established**.

The ledger explicitly does not claim bitwise training reproduction.

## 19. GO/HOLD summary

| gate | verdict |
|---|---|
| Gate 1 graph computation | **GO_GRAPH** |
| Gate 2 ownership-preserving processing | **GO / VIABLE** |
| Gate 3 absolute parent quality | **CONDITIONAL** |
| ORED-2 final | **GO_VIABLE_CONDITIONAL_PARENT_QUALITY** |

This is a viable parent result, not a STRONG_GO and not a license to start ORED-3
inside this stage. The work stops here as preregistered.

## 20. Exact commands

Unit tests:

```bash
conda run --no-capture-output -n yhf_env python -m pytest tests/ -q
```

ORED-2A smoke commands used the fixed overrides below for each of the four
variants, with `hydra.run.dir=outputs/ored/o2a/movies_seed42_<variant>` and a
variant-specific checkpoint path:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. conda run --no-capture-output -n yhf_env python -m src.main \
  dataset=Movies task=nc model=ored_mag model.variant=<variant> num_runs=1 seed=42 device=cuda:0 \
  task.evaluate_test=false model.hidden_dim=256 model.factor_dim=128 model.dropout=0.2 \
  model.num_hops=2 model.restart=0.15 model.diffusion_add_self_loops=true
```

Formal ORED-2B replication:

```bash
PYTHONPATH=. conda run --no-capture-output -n yhf_env python scripts/run_ored_o2.py --gpu 0 --shard 0 --shards 2
PYTHONPATH=. conda run --no-capture-output -n yhf_env python scripts/run_ored_o2.py --gpu 1 --shard 1 --shards 2
conda run --no-capture-output -n yhf_env python scripts/summarize_ored_o2.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. conda run --no-capture-output -n yhf_env python scripts/ored_o2_diagnostics.py --device cuda:0
```

All formal commands set `task.evaluate_test=false`; no Test command was run.

## 21. Scope statement

```text
NO TEST
NO SEMANTIC EDGE
NO COMPOSITION
NO EXPOSURE
NO CROSS-FACTOR TRANSPORT
NO TUNING
NO NEW AUXILIARY LOSS
NO ORED-3
```

The only comparison is `p0`, `p0_refine`, `joint_rd`, and `ownership_rd` under the
MAP unified NC protocol with the frozen split, optimizer, early stopping, and
validation-Accuracy checkpoint rule.
