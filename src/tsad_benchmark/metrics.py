from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from .config import DEFAULT_VUS_THRESHOLDS, DEFAULT_VUS_WINDOW
from .vus import official_vus


def _valid_binary(labels: np.ndarray) -> bool:
    labels = np.asarray(labels, dtype=int)
    return labels.size > 0 and len(np.unique(labels)) == 2


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if not _valid_binary(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    if not _valid_binary(labels):
        return float("nan")
    return float(average_precision_score(labels, scores))


def best_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    if not _valid_binary(labels):
        return float("nan")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return float("nan")
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(np.nanmax(f1))


def contamination_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    if positives <= 0 or positives >= len(labels):
        return float("nan")
    order = np.argsort(scores)[::-1]
    pred = np.zeros_like(labels)
    pred[order[:positives]] = 1
    return float(f1_score(labels, pred))


def dilate_labels(labels: np.ndarray, radius: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if radius <= 0 or labels.size == 0:
        return labels.copy()
    positives = np.where(labels > 0)[0]
    out = labels.copy()
    for idx in positives:
        lo = max(0, idx - radius)
        hi = min(len(labels), idx + radius + 1)
        out[lo:hi] = 1
    return out


def vus_auc_approx(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: str,
    max_radius: int | None = None,
    steps: int = 20,
) -> float:
    """Range-dilated VUS-style approximation.

    This averages AUROC or AUPRC over increasing label-dilation radii. It is
    useful for early experiments; strict final reporting should use the
    range-aware VUS implementation exposed as ``vus_pr`` / ``vus_roc``.
    """
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        return float("nan")
    if max_radius is None:
        max_radius = min(100, max(1, int(0.02 * len(labels))))
    radii = np.unique(np.linspace(0, max_radius, steps, dtype=int))
    values: list[float] = []
    for radius in radii:
        dilated = dilate_labels(labels, int(radius))
        if not _valid_binary(dilated):
            continue
        if metric == "roc":
            values.append(safe_auroc(dilated, scores))
        elif metric == "pr":
            values.append(safe_auprc(dilated, scores))
        else:
            raise ValueError("metric must be 'roc' or 'pr'")
    return float(np.nanmean(values)) if values else float("nan")


def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    vus_window: int = DEFAULT_VUS_WINDOW,
    vus_thresholds: int = DEFAULT_VUS_THRESHOLDS,
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    if not finite.all():
        fill = float(np.nanmedian(scores[finite])) if finite.any() else 0.0
        scores = scores.copy()
        scores[~finite] = fill
    vus_roc, vus_pr = official_vus(labels, scores, sliding_window=vus_window, thresholds=vus_thresholds)
    return {
        "auroc": safe_auroc(labels, scores),
        "auprc": safe_auprc(labels, scores),
        "vus_roc": vus_roc,
        "vus_pr": vus_pr,
        "vus_roc_approx": vus_auc_approx(labels, scores, metric="roc"),
        "vus_pr_approx": vus_auc_approx(labels, scores, metric="pr"),
        "best_f1": best_f1(labels, scores),
        "contamination_f1": contamination_f1(labels, scores),
    }
