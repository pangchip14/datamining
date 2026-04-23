from __future__ import annotations

import numpy as np

from .data import TimeSeriesRecord
from .config import RANDOM_SEED


def _base_signal(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n)
    return (
        np.sin(2 * np.pi * t / 80)
        + 0.35 * np.sin(2 * np.pi * t / 23)
        + rng.normal(0, 0.18, size=n)
    )


def make_synthetic_suite(n: int = 1200, seed: int = RANDOM_SEED) -> list[TimeSeriesRecord]:
    rng = np.random.default_rng(seed)
    records: list[TimeSeriesRecord] = []

    values = _base_signal(n, rng)
    labels = np.zeros(n, dtype=int)
    idx = np.array([180, 360, 690, 930])
    values[idx] += np.array([4.5, -4.0, 5.0, -4.7])
    labels[idx] = 1
    records.append(TimeSeriesRecord("synthetic_point_spikes", "synthetic", "", values, labels))

    values = _base_signal(n, rng)
    labels = np.zeros(n, dtype=int)
    for start in [260, 780]:
        end = start + 70
        values[start:end] += 2.2
        values[start:end] += np.linspace(0, 1.5, end - start)
        labels[start:end] = 1
    records.append(TimeSeriesRecord("synthetic_collective_shift", "synthetic", "", values, labels))

    values = _base_signal(n, rng)
    labels = np.zeros(n, dtype=int)
    for start in [420, 880]:
        end = start + 90
        t = np.arange(end - start)
        values[start:end] = 0.5 * np.sin(2 * np.pi * t / 80) + rng.normal(0, 0.1, end - start)
        labels[start:end] = 1
    records.append(TimeSeriesRecord("synthetic_contextual_phase", "synthetic", "", values, labels))

    values = _base_signal(n, rng) + rng.normal(0, 0.35, size=n)
    labels = np.zeros(n, dtype=int)
    for start in [170, 610, 980]:
        end = start + 45
        values[start:end] = values[start:end][::-1] + rng.normal(0, 0.15, end - start)
        labels[start:end] = 1
    records.append(TimeSeriesRecord("synthetic_noisy_subsequence", "synthetic", "", values, labels))

    return records
