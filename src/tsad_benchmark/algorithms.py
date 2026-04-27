from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from .config import RANDOM_SEED
from .preprocessing import align_window_scores, sliding_windows, z_normalize


@dataclass
class AlgorithmResult:
    name: str
    scores: np.ndarray
    runtime_sec: float
    metadata: dict[str, float | int | str]


def rolling_zscore_scores(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    window = max(3, min(window, len(values)))
    scores = np.zeros(len(values), dtype=float)
    half = window // 2
    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        local = values[start:end]
        mean = float(np.mean(local))
        std = float(np.std(local))
        scores[i] = 0.0 if std < 1e-12 else abs((values[i] - mean) / std)
    return scores


def window_knn_scores(windows: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    if len(windows) < 2:
        return np.zeros(len(windows), dtype=float)
    k = min(max(1, n_neighbors), len(windows) - 1)
    model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    model.fit(windows)
    distances, _ = model.kneighbors(windows)
    return distances[:, -1]


def window_lof_scores(windows: np.ndarray, n_neighbors: int = 20) -> np.ndarray:
    if len(windows) < 2:
        return np.zeros(len(windows), dtype=float)
    k = min(max(1, n_neighbors), len(windows) - 1)
    model = LocalOutlierFactor(n_neighbors=k, metric="euclidean")
    model.fit_predict(windows)
    return -model.negative_outlier_factor_


def window_isolation_forest_scores(
    windows: np.ndarray,
    n_estimators: int = 200,
    max_samples: str | int | float = "auto",
) -> np.ndarray:
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination="auto",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(windows)
    return -model.decision_function(windows)


def window_ik_density_scores(
    windows: np.ndarray,
    n_estimators: int = 200,
    max_samples: str | int | float = "auto",
    method: str = "inne",
) -> np.ndarray:
    if len(windows) < 2:
        return np.zeros(len(windows), dtype=float)
    try:
        from ikpykit.anomaly import IDKD
    except Exception as exc:
        raise ImportError(
            "ikpykit is required for the official IDKD implementation. "
            "Install it with `python -m pip install ikpykit`."
        ) from exc

    windows32 = windows.astype(np.float32, copy=False)
    model = IDKD(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination="auto",
        method=method,
        random_state=RANDOM_SEED,
    )
    model.fit(windows32)
    return -model.score_samples(windows32)


def _matrix_profile_with_stumpy(values: np.ndarray, window: int) -> np.ndarray | None:
    try:
        import stumpy  # type: ignore
    except Exception:
        return None
    try:
        profile = stumpy.stump(np.asarray(values, dtype=float), m=window)[:, 0].astype(float)
        return profile
    except Exception:
        return None


def _matrix_profile_fallback(values: np.ndarray, window: int, max_windows: int = 2500) -> np.ndarray:
    windows = sliding_windows(z_normalize(values), window)
    n_windows = len(windows)
    if n_windows == 1:
        return np.zeros(1, dtype=float)

    if n_windows > max_windows:
        sampled_idx = np.linspace(0, n_windows - 1, max_windows, dtype=int)
        sampled = windows[sampled_idx]
        distances = pairwise_distances(windows, sampled, metric="euclidean")
        exclusion = max(1, window // 2)
        for row in range(n_windows):
            bad = np.abs(sampled_idx - row) <= exclusion
            distances[row, bad] = np.inf
        scores = np.min(distances, axis=1)
    else:
        distances = pairwise_distances(windows, windows, metric="euclidean")
        exclusion = max(1, window // 2)
        for row in range(n_windows):
            lo = max(0, row - exclusion)
            hi = min(n_windows, row + exclusion + 1)
            distances[row, lo:hi] = np.inf
        scores = np.min(distances, axis=1)

    finite = np.isfinite(scores)
    if finite.any():
        fill = float(np.max(scores[finite]))
        scores[~finite] = fill
    else:
        scores[:] = 0.0
    return scores


def matrix_profile_scores(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 4:
        return np.zeros(len(values), dtype=float)
    window = max(4, min(window, len(values)))
    if len(values) - window + 1 < 2:
        return np.zeros(len(values), dtype=float)
    profile = _matrix_profile_with_stumpy(values, window)
    if profile is None:
        profile = _matrix_profile_fallback(values, window)
    finite = np.isfinite(profile)
    if finite.any():
        fill = float(np.max(profile[finite]))
        profile[~finite] = fill
    else:
        profile[:] = 0.0
    return align_window_scores(profile, len(values), window, mode="max")


def run_algorithm(
    name: str,
    values: np.ndarray,
    window: int,
    normalize: bool = True,
) -> AlgorithmResult:
    """Run one detector and return point-level anomaly scores."""
    raw_values = np.asarray(values, dtype=float)
    model_values = z_normalize(raw_values) if normalize else raw_values.astype(float)
    start = time.perf_counter()

    if name == "rolling_zscore":
        point_scores = rolling_zscore_scores(model_values, window)
    elif name == "matrix_profile":
        point_scores = matrix_profile_scores(model_values, window)
    else:
        windows = sliding_windows(model_values, window)
        if name == "knn":
            window_scores = window_knn_scores(windows)
        elif name == "lof":
            window_scores = window_lof_scores(windows)
        elif name == "isolation_forest":
            window_scores = window_isolation_forest_scores(windows)
        elif name == "ik_density":
            window_scores = window_ik_density_scores(windows)
        else:
            raise ValueError(f"unknown algorithm: {name}")
        point_scores = align_window_scores(window_scores, len(model_values), window, mode="max")

    runtime = time.perf_counter() - start
    return AlgorithmResult(
        name=name,
        scores=np.asarray(point_scores, dtype=float),
        runtime_sec=runtime,
        metadata={"window": int(window), "normalize": str(normalize)},
    )
