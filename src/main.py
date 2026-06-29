from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.data import load_mag_data
from src.tasks import run_lp, run_nc
from src.utils.device import get_device
from src.utils.logging import setup_logger


def _log_data_info(logger, data, model_name: str) -> None:
    logger.info("Dataset: %s | Source: %s | Task: %s", data.name, data.source, data.task)
    logger.info("Model: %s", model_name)
    logger.info("X: %s | dtype=%s", tuple(data.x.shape), data.x.dtype)
    if data.x_i is not None or data.x_t is not None:
        x_i_shape = tuple(data.x_i.shape) if data.x_i is not None else None
        x_t_shape = tuple(data.x_t.shape) if data.x_t is not None else None
        logger.info("X_i: %s | X_t: %s", x_i_shape, x_t_shape)
    logger.info("Graph edge_index: %s | num_nodes=%d | num_edges=%d", tuple(data.edge_index.shape), data.num_nodes, data.num_edges)
    if data.y is not None:
        logger.info("Labels: shape=%s | num_classes=%s", tuple(data.y.shape), data.num_classes)
    if data.train_idx is not None:
        logger.info(
            "NC split: train=%d | val=%d | test=%d",
            int(data.train_idx.numel()),
            int(data.val_idx.numel()),
            int(data.test_idx.numel()),
        )
    if data.edge_split is not None:
        logger.info(
            "LP split: train=%d | valid=%d x %d neg | test=%d x %d neg",
            int(data.edge_split.train["source_node"].numel()),
            int(data.edge_split.valid["source_node"].numel()),
            int(data.edge_split.valid["target_node_neg"].size(1)),
            int(data.edge_split.test["source_node"].numel()),
            int(data.edge_split.test["target_node_neg"].size(1)),
        )
    for key, value in data.info.items():
        logger.info("%s: %s", key, value)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger = setup_logger(output_dir, cfg.logging.level)
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    device = get_device(str(cfg.device))
    logger.info("Device: %s", device)

    data = load_mag_data(cfg, str(cfg.task.name), int(cfg.seed))
    _log_data_info(logger, data, str(cfg.model.name))

    if int(cfg.task.epochs) <= 0:
        logger.info("task.epochs <= 0, stopping after data loading/split preparation")
        results = {}
    elif str(cfg.task.name) == "nc":
        results = run_nc(cfg, data, device, logger, output_dir)
    elif str(cfg.task.name) == "lp":
        results = run_lp(cfg, data, device, logger, output_dir)
    else:
        raise ValueError(f"Unsupported task: {cfg.task.name}")

    serializable = {key: {"mean": value[0], "std": value[1]} for key, value in results.items()}
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Saved results: %s", output_dir / "results.json")


if __name__ == "__main__":
    main()
