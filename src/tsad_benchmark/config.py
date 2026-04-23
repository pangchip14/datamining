from __future__ import annotations

RANDOM_SEED = 20260423
MAIN_ALGORITHMS = (
    "rolling_zscore",
    "knn",
    "lof",
    "isolation_forest",
    "ik_density",
    "matrix_profile",
)


def default_window_size(n: int) -> int:
    """Default subsequence length used across window-based detectors."""
    if n <= 1:
        return 1
    return max(16, min(128, int(0.02 * n)))
