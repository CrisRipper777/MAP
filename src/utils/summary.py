from __future__ import annotations

import numpy as np


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    # 0901/RPTA reporting convention: standard deviation over the observed
    # runs is the population std (ddof=0), not the unbiased sample estimate.
    return float(arr.mean()), float(arr.std(ddof=0))


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
