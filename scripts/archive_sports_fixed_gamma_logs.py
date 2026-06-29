from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SportsLog:
    gamma_key: str
    fixed_gamma: float
    source: Path


SPORTS_LOGS = (
    SportsLog("fixed_gamma_005", 0.05, Path("outputs/2026-06-21/16-21-03")),
    SportsLog("fixed_gamma_025", 0.25, Path("outputs/2026-06-21/19-05-37")),
    SportsLog("fixed_gamma_050", 0.50, Path("outputs/2026-06-21/21-46-03")),
    SportsLog("fixed_gamma_075", 0.75, Path("outputs/2026-06-22/00-19-48")),
    SportsLog("fixed_gamma_095", 0.95, Path("outputs/2026-06-22/02-53-08")),
)


def _read_results(path: Path) -> dict:
    results_path = path / "results.json"
    if not results_path.exists():
        return {}
    return json.loads(results_path.read_text(encoding="utf-8"))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / "outputs/map_mag_v1_frequency_gamma/sports-copurchase-LP"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    for item in SPORTS_LOGS:
        source = project_root / item.source
        if not source.exists():
            raise FileNotFoundError(f"Missing sports fixed-gamma log directory: {source}")

        target = output_root / item.gamma_key / "legacy_seed42_runs1" / source.parent.name / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)

        results = _read_results(source)
        manifest_rows.append(
            {
                "dataset": "sports-copurchase-LP",
                "gamma_key": item.gamma_key,
                "fixed_gamma": item.fixed_gamma,
                "source_path": str(source),
                "archived_path": str(target),
                "model": "map_mag",
                "num_runs": 1,
                "seed": 42,
                "note": "legacy sports fixed-gamma run copied from dated Hydra output",
                "val_mrr": results.get("val_mrr", {}).get("mean"),
                "test_mrr": results.get("test_mrr", {}).get("mean"),
                "test_hits@1": results.get("test_hits@1", {}).get("mean"),
                "test_hits@3": results.get("test_hits@3", {}).get("mean"),
                "test_hits@10": results.get("test_hits@10", {}).get("mean"),
            }
        )
        print(f"Archived {source} -> {target}", flush=True)

    manifest_path = output_root / "legacy_sports_fixed_gamma_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Saved manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
