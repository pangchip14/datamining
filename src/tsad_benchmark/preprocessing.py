from __future__ import annotations

import numpy as np
import pandas as pd


def clean_values(values: np.ndarray) -> np.ndarray:
    """Convert to finite float values and interpolate missing entries."""
    series = pd.Series(np.asarray(values, dtype=float))
    series = series.replace([np.inf, -np.inf], np.nan)
    if series.isna().all():
        raise ValueError("time series contains no finite values")
    series = series.interpolate(limit_direction="both")
    series = series.fillna(series.median())
    return series.to_numpy(dtype=float)


def z_normalize(values: np.ndarray) -> np.ndarray:
    """Return z-normalized values, falling back to centered values if std is zero."""
    values = clean_values(values)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-12:
        return values - mean
    return (values - mean) / std


def sliding_windows(values: np.ndarray, window: int) -> np.ndarray:
    """Create all consecutive subsequences with shape (n-window+1, window)."""
    values = np.asarray(values, dtype=float)
    if window < 1:
        raise ValueError("window must be >= 1")
    if len(values) < window:
        return values.reshape(1, -1)
    shape = (len(values) - window + 1, window)
    strides = (values.strides[0], values.strides[0])
    return np.lib.stride_tricks.as_strided(values, shape=shape, strides=strides).copy()


def align_window_scores(
    window_scores: np.ndarray,
    n_points: int,
    window: int,
    mode: str = "max",
) -> np.ndarray:
    """Map window-level scores to point-level scores.

    The default follows the final plan: each point receives the maximum score
    among all windows covering that point.
    """
    scores = np.asarray(window_scores, dtype=float)
    if n_points <= 0:
        return np.array([], dtype=float)
    if scores.size == 0:
        return np.zeros(n_points, dtype=float)

    window = max(1, min(window, n_points))
    point_scores = np.full(n_points, -np.inf if mode == "max" else 0.0, dtype=float)
    counts = np.zeros(n_points, dtype=float)
    for start, score in enumerate(scores):
        end = min(n_points, start + window)
        if mode == "max":
            point_scores[start:end] = np.maximum(point_scores[start:end], score)
        elif mode == "mean":
            point_scores[start:end] += score
            counts[start:end] += 1.0
        elif mode == "center":
            center = min(n_points - 1, start + window // 2)
            point_scores[center] = score
            counts[center] = 1.0
        else:
            raise ValueError(f"unknown alignment mode: {mode}")

    if mode == "mean":
        counts[counts == 0] = 1.0
        point_scores = point_scores / counts
    elif mode == "center":
        valid = counts > 0
        if valid.any():
            point_scores = pd.Series(point_scores).replace([np.inf, -np.inf], np.nan)
            point_scores = point_scores.interpolate(limit_direction="both").fillna(0.0).to_numpy()
        else:
            point_scores = np.zeros(n_points, dtype=float)
    else:
        finite = np.isfinite(point_scores)
        fill = float(np.nanmin(point_scores[finite])) if finite.any() else 0.0
        point_scores[~finite] = fill
    return point_scores


def minmax_scale(scores: np.ndarray) -> np.ndarray:
    """Scale scores into [0, 1] for plotting only."""
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    if not finite.any():
        return np.zeros_like(scores, dtype=float)
    lo = float(np.min(scores[finite]))
    hi = float(np.max(scores[finite]))
    if hi - lo < 1e-12:
        return np.zeros_like(scores, dtype=float)
    out = (scores - lo) / (hi - lo)
    out[~finite] = 0.0
    return out
