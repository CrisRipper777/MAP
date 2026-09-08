# MAG_baseline

`MAG_baseline` is a PyTorch / PyG / Hydra baseline framework for multimodal attributed graph experiments.

It only supports:

- Node Classification (NC)
- Link Prediction (LP)

It uses frozen features from `../data` and does not train BERT, ViT, CLIP, Qwen-VL, or other large encoders.

The frozen NC/LP experiment contract is documented in
[`docs/unified_training_evaluation_protocol.md`](docs/unified_training_evaluation_protocol.md).

Implemented models: `mlp`, `gcn`, `sage`, `mmgcn`, `mgat`, `dip`, `dgf`, `dmgc`, `lgmrec`, `map_mag`, `map_mag_v1`, `map_mag_v2`, `map_mag_v3`.

## Datasets

MAGB:

- `Movies`: NC + LP
- `Toys`: NC + LP
- `Grocery`: NC + LP
- `Reddit-S`: NC + LP

MM-Graph:

- `sports-copurchase`: LP
- `cloth-copurchase`: LP
- `books-lp`: LP
- `ele-fashion`: NC
- `books-nc`: NC

## Run

Use the requested environment:

```bash
conda activate yhf_env
cd /hdd1/DataInHere/YHF/MAG_baseline
```

Run one NC experiment:

```bash
python -m src.main dataset=Movies task=nc model=mlp num_runs=3
```

Run one LP experiment:

```bash
python -m src.main dataset=sports-copurchase task=lp model=sage num_runs=3
```

Run DiP:

```bash
python -m src.main dataset=ele-fashion task=nc model=dip num_runs=3
python -m src.main dataset=sports-copurchase task=lp model=dip num_runs=3
```

Run the 0901/RPTA baseline models:

```bash
python -m src.main dataset=Movies task=nc model=dgf num_runs=3
python -m src.main dataset=Movies task=nc model=dmgc num_runs=3
python -m src.main dataset=Movies task=nc model=lgmrec num_runs=3
python -m src.main dataset=sports-copurchase task=lp model=dgf num_runs=3
python -m src.main dataset=sports-copurchase task=lp model=dmgc num_runs=3
python -m src.main dataset=sports-copurchase task=lp model=lgmrec num_runs=3
```

Run frozen MAP-MAG v1:

```bash
python -m src.main dataset=Movies task=nc model=map_mag_v1 num_runs=3
python -m src.main dataset=sports-copurchase task=lp model=map_mag_v1 num_runs=3
```

Run MAP-MAG v3 core and controlled variants:

```bash
python -m src.main dataset=Movies task=nc model=map_mag_v3 num_runs=3
python -m src.main dataset=Movies task=nc model=map_mag_v3_full num_runs=3
python -m src.main dataset=sports-copurchase task=lp model=map_mag_v3_lp num_runs=3
```

`map_mag_v3` is the stable dual-semantic-graph core. `map_mag_v3_full` also
enables conflict propagation, self residuals, and modality-specific
prototypes. `map_mag_v3_lp` enables conflict propagation and the self
residual, but still uses the shared Hadamard LP decoder required by the
unified protocol. See
[`docs/map_mag_family_comparison.md`](docs/map_mag_family_comparison.md).

Run the complete main benchmark (all implemented models, 5 NC datasets with
three seeds, and sports-copurchase LP with one seed):

```bash
python scripts/run_full_benchmark.py
```

The launcher uses one process per device, so the default `cuda:0 cuda:1`
setting runs two jobs in parallel without sharing a GPU. Use
`--devices cuda:0` for single-card execution. Every job has a unique output
directory under `outputs/full_benchmark/`; completed jobs are skipped on a
rerun, and `benchmark_manifest.json` / `benchmark_summary.json` record the
plan and status. Inspect commands first with `--dry-run`. The optional
`--include-v3-full` adds the controlled v3 ablation; it is not part of the
main-model list.

For a short smoke run, append Hydra overrides after `--`, for example:

```bash
python scripts/run_full_benchmark.py --only nc --models map_mag_v2 map_mag_v3 \
  --nc-datasets Movies --devices cuda:0 --output-root outputs/benchmark_smoke \
  -- task.epochs=1 task.patience=1 model.export_node_aux=false
```

Run the MAP-MAG v1 core ablations on Movies-NC, Toys-NC, sports-copurchase-LP, and ele-fashion-NC:

```bash
python scripts/run_map_mag_v1_core_ablation.py --device cuda:0
```

This script uses `seed=42` and `num_runs=3`, so each job runs seeds 42, 43, and 44.
Use `--dry-run` to inspect commands, `--datasets ...` / `--ablations ...` to run a subset,
and append Hydra overrides after `--`.

Analyze MAP-MAG v1 path preferences from exported node aux CSVs:

```bash
python scripts/analyze_map_mag_v1_paths.py \
  --input-root outputs/map_mag_v1_core_ablation \
  --ablations full \
  --checkpoint best_val
```

The analysis script reuses existing training outputs and writes distribution summaries, histogram/boxplot SVGs,
degree-group summaries, consistency-group summaries, and NC class summaries under `outputs/map_mag_v1_path_analysis/`.

Run MAP-MAG v1 frequency-gamma diagnostics on Movies-NC, Toys-NC, and ele-fashion-NC:

```bash
python scripts/run_map_mag_v1_frequency_gamma.py --device cuda:0
```

This runs learned gamma plus fixed gamma values 0.05, 0.50, 0.75, and 0.95 with seeds 42, 43, and 44.
Previously completed sports-copurchase-LP fixed-gamma logs can be copied into the same output tree with:

```bash
python scripts/archive_sports_fixed_gamma_logs.py
```

Summarize all frequency-gamma `results.json` files:

```bash
python scripts/summarize_map_mag_frequency_gamma.py
```

Smoke test with a short run:

```bash
python -m src.main dataset=Movies task=nc model=mlp num_runs=1 task.epochs=1 task.max_train_batches=2 device=cpu
```

## Notes

- MAGB `*Graph.pt` files are DGL graphs, so the loader converts DGL graphs to PyG `edge_index`.
- MAGB NC/LP splits are generated once under `../data/MAGB_split` when missing.
- MM-Graph NC/LP tasks use the official split files shipped in each dataset directory.
- LP evaluation ranks one positive target against fixed negative targets and reports MRR / Hits@1 / Hits@3 / Hits@10.
- NC uses the unified full-graph training protocol by default: one graph forward per epoch, CE on train nodes, validation-accuracy checkpoint selection, and one final test evaluation.
- LP uses the unified sampled protocol: graph encoders use bidirectional two-hop `LinkNeighborLoader` sampling with one filtered negative per positive; the shared LP projection dimension is 128 and equal-score negatives use pessimistic ranking.
- Test metrics are computed once after training, by reloading the best validation checkpoint.
- Dataset graphs do not add self-loops by default; models own their self-loop policy, e.g. GCN adds them internally.
- `model=mlp` does not use graph sampling: NC uses node mini-batches and LP uses edge mini-batches.
- GNN models such as `gcn` and `sage` use PyG `NeighborLoader` / `LinkNeighborLoader`.
- NC always uses full-graph training. LP always uses the unified sampled protocol, so global encoders such as `dip` and `map_mag_v2` are explicitly evaluated as sampled adaptations even if their model config advertises a native full-graph preference.
- `model=dip` implements the exact DiP global pseudo-node recurrence; its `layerwise` inference API currently falls back to the exact full-graph DiP pass because pseudo-node states depend on all nodes at each recurrent step.
- `model=map_mag_v1` exports node-level analysis CSVs by default under each Hydra run directory: `node_aux/run_XX_best_val_node_aux.csv` and `node_aux/run_XX_final_epoch_node_aux.csv`. Disable with `model.export_node_aux=false`.
- `model=dgf`, `model=dmgc`, and `model=lgmrec` are ported from `/hdd1/DataInHere/YHF/0901/0901` and use the same supervised CE/BCE task runners as the other encoders; their model-specific optimizer presets live in `configs/model/`.
