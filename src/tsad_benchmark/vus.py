from __future__ import annotations

import numpy as np


def _segments(labels: np.ndarray) -> list[tuple[int, int]]:
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        return []
    starts = np.where(np.diff(labels) == 1)[0] + 1
    ends = np.where(np.diff(labels) == -1)[0]
    if labels[0] > 0:
        starts = np.concatenate([[0], starts])
    if labels[-1] > 0:
        ends = np.concatenate([ends, [len(labels) - 1]])
    return list(zip(starts.astype(int), ends.astype(int)))


def _mask_from_segments(n: int, segments: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for start, end in segments:
        mask[start : end + 1] = True
    return mask


def _expanded_segments(n: int, original: list[tuple[int, int]], window: int) -> list[tuple[int, int]]:
    if not original:
        return []
    start = max(original[0][0] - window // 2, 0)
    out: list[tuple[int, int]] = []
    for current, nxt in zip(original, original[1:]):
        current_end = current[1] + window // 2
        next_start = nxt[0] - window // 2
        if current_end < next_start:
            out.append((start, min(current_end, n - 1)))
            start = max(next_start, 0)
    out.append((start, min(original[-1][1] + window // 2, n - 1)))
    return out


def _soft_extend_labels(labels: np.ndarray, segments: list[tuple[int, int]], window: int) -> np.ndarray:
    out = labels.astype(float).copy()
    if window <= 0:
        return out
    n = len(out)
    for start, end in segments:
        right = np.arange(end + 1, min(end + window // 2 + 1, n))
        if right.size:
            out[right] += np.sqrt(1 - (right - end) / window)
        left = np.arange(max(start - window // 2, 0), start)
        if left.size:
            out[left] += np.sqrt(1 - (start - left) / window)
    return np.minimum(1.0, out)


def official_vus(
    labels: np.ndarray,
    scores: np.ndarray,
    sliding_window: int = 100,
    thresholds: int = 250,
) -> tuple[float, float]:
    """Compute official-style VUS-ROC and VUS-PR.

    This follows the public TSB-UAD/TSB-AD VUS algorithm from Paparrizos et al.
    It integrates range-aware ROC/PR curves over buffer windows 0..sliding_window.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0 or labels.size != scores.size or len(np.unique(labels)) != 2:
        return float("nan"), float("nan")

    n = len(labels)
    sliding_window = max(0, int(sliding_window))
    thresholds = max(1, int(thresholds))
    original_segments = _segments(labels)
    if not original_segments:
        return float("nan"), float("nan")

    score_sorted = -np.sort(-scores)
    threshold_idx = np.linspace(0, n - 1, thresholds).astype(int)
    threshold_values = score_sorted[threshold_idx]
    pred_matrix = scores[:, None] >= threshold_values[None, :]
    n_pred = pred_matrix.sum(axis=0).astype(float)
    n_pred = np.maximum(n_pred, 1e-15)

    original_mask = _mask_from_segments(n, original_segments)
    max_window_segments = _expanded_segments(n, original_segments, sliding_window)
    max_window_mask = _mask_from_segments(n, max_window_segments)
    p_original = float(labels.sum())

    auc_values: list[float] = []
    ap_values: list[float] = []
    for window in range(sliding_window + 1):
        labels_extended = _soft_extend_labels(labels, original_segments, window)
        current_segments = _expanded_segments(n, original_segments, window)
        if not current_segments:
            continue

        weights_for_tp = labels_extended.copy()
        weights_for_tp[original_mask] = 1.0
        active_mask = max_window_mask
        non_original_active = active_mask & ~original_mask

        tp = (weights_for_tp[active_mask, None] * pred_matrix[active_mask]).sum(axis=0)
        n_labels = float(original_mask[active_mask].sum())
        if non_original_active.any():
            n_labels = n_labels + (
                labels_extended[non_original_active, None] * pred_matrix[non_original_active]
            ).sum(axis=0)

        existence = np.zeros(thresholds, dtype=float)
        for start, end in current_segments:
            existence += pred_matrix[start : end + 1].any(axis=0)
        existence_ratio = existence / len(current_segments)

        p_new = (p_original + n_labels) / 2.0
        recall = np.minimum(tp / np.maximum(p_new, 1e-15), 1.0)
        tpr = recall * existence_ratio

        fp = n_pred - tp
        n_new = np.maximum(n - p_new, 1e-15)
        fpr = fp / n_new
        precision = tp / n_pred

        fpr_curve = np.concatenate([[0.0], fpr, [1.0]])
        tpr_curve = np.concatenate([[0.0], tpr, [1.0]])
        auc = np.sum((fpr_curve[1:] - fpr_curve[:-1]) * (tpr_curve[1:] + tpr_curve[:-1]) / 2.0)

        tpr_for_pr = np.concatenate([[0.0], tpr])
        ap = np.sum((tpr_for_pr[1:] - tpr_for_pr[:-1]) * precision)

        auc_values.append(float(auc))
        ap_values.append(float(ap))

    if not auc_values:
        return float("nan"), float("nan")
    return float(np.mean(auc_values)), float(np.mean(ap_values))
