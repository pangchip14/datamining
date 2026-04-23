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


class IKDensityDetector:
    """Lightweight isolation-kernel-family density detector.

    For each random sample, points are assigned to the nearest sampled center.
    Points repeatedly assigned to large cells are treated as more normal; low
    average cell mass becomes a higher anomaly score.
    """

    def __init__(self, n_estimators: int = 100, psi: int = 32, random_state: int = RANDOM_SEED):
        self.n_estimators = n_estimators
        self.psi = psi
        self.random_state = random_state

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = x.shape[0]
        if n == 0:
            return np.array([], dtype=float)
        rng = np.random.default_rng(self.random_state)
        psi = min(self.psi, n)
        density = np.zeros(n, dtype=float)
        for _ in range(self.n_estimators):
            centers_idx = rng.choice(n, size=psi, replace=False)
            centers = x[centers_idx]
            nearest = pairwise_distances(x, centers, metric="euclidean").argmin(axis=1)
            counts = np.bincount(nearest, minlength=psi).astype(float)
            density += counts[nearest] / n
        density /= max(1, self.n_estimators)
        return -density


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
    k = min(max(1, n_neighbors), max(1, len(windows) - 1))
    model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    model.fit(windows)
    distances, _ = model.kneighbors(windows)
    return distances[:, -1]


def window_lof_scores(windows: np.ndarray, n_neighbors: int = 20) -> np.ndarray:
    k = min(max(2, n_neighbors), max(2, len(windows) - 1))
    model = LocalOutlierFactor(n_neighbors=k, metric="euclidean")
    model.fit_predict(windows)
    return -model.negative_outlier_factor_


def window_isolation_forest_scores(windows: np.ndarray) -> np.ndarray:
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(windows)
    return -model.decision_function(windows)


def window_ik_density_scores(windows: np.ndarray) -> np.ndarray:
    model = IKDensityDetector(n_estimators=100, psi=32, random_state=RANDOM_SEED)
    return model.score_samples(windows)


def _matrix_profile_with_stumpy(values: np.ndarray, window: int) -> np.ndarray | None:
    try:
        import stumpy  # type: ignore
    except Exception:
        return None
    profile = stumpy.stump(np.asarray(values, dtype=float), m=window)[:, 0].astype(float)
    return profile


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
    window = max(4, min(window, len(values)))
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
