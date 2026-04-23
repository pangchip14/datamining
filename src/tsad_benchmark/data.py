from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import clean_values


SUPPORTED_SUFFIXES = {".out", ".csv", ".txt", ".tsv"}


@dataclass(frozen=True)
class TimeSeriesRecord:
    name: str
    source: str
    path: str
    values: np.ndarray
    labels: np.ndarray


def _read_univariate_value_label(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a univariate TSB-UAD-style value/label pair.

    Accepted formats:
    - headered CSV with Data and Label columns
    - headered CSV with exactly one non-label value column and Label
    - headerless two-column file

    Multivariate files with multiple non-label columns are intentionally
    rejected because the study scope is univariate time-series anomaly
    detection.
    """
    sep = "\t" if path.suffix.lower() == ".tsv" else None
    header_df = pd.read_csv(path, sep=sep, engine="python").dropna(how="all")
    columns = [str(col) for col in header_df.columns]
    lower_to_col = {col.lower(): col for col in columns}
    if "label" in lower_to_col:
        label_col = lower_to_col["label"]
        if "data" in lower_to_col:
            value_col = lower_to_col["data"]
        else:
            value_candidates = [col for col in columns if col != label_col]
            if len(value_candidates) != 1:
                raise ValueError(
                    f"{path} appears multivariate ({len(value_candidates)} value columns); "
                    "skipping for univariate study"
                )
            value_col = value_candidates[0]
        return header_df[value_col].to_numpy(dtype=float), header_df[label_col].to_numpy()

    raw_df = pd.read_csv(path, header=None, sep=sep, engine="python").dropna(how="all")
    if raw_df.shape[1] < 2:
        raise ValueError(f"{path} has fewer than two columns")
    if raw_df.shape[1] > 2:
        raise ValueError(f"{path} appears multivariate without a Label header")
    return raw_df.iloc[:, 0].to_numpy(dtype=float), raw_df.iloc[:, 1].to_numpy()


def load_tsb_file(path: str | Path) -> TimeSeriesRecord:
    """Load a univariate TSB-UAD-style file."""
    path = Path(path)
    raw_values, raw_labels = _read_univariate_value_label(path)
    values = clean_values(raw_values)
    labels = raw_labels
    labels = np.asarray(labels, dtype=float)
    labels = np.where(labels > 0, 1, 0).astype(int)
    if values.shape[0] != labels.shape[0]:
        raise ValueError(f"value/label length mismatch in {path}")
    return TimeSeriesRecord(
        name=path.stem,
        source=path.parent.name,
        path=str(path),
        values=values,
        labels=labels,
    )


def discover_series_files(root: str | Path) -> list[Path]:
    """Recursively discover candidate TSB-UAD files."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files)


def anomaly_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive-exclusive anomaly segments."""
    labels = np.asarray(labels, dtype=int)
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for i, label in enumerate(labels):
        if label and start is None:
            start = i
        elif not label and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(labels)))
    return segments


def seasonality_proxy(values: np.ndarray, max_lag: int = 200) -> float:
    """Estimate seasonality strength with maximum absolute autocorrelation."""
    values = clean_values(values)
    if len(values) < 8:
        return 0.0
    centered = values - np.mean(values)
    denom = float(np.dot(centered, centered))
    if denom < 1e-12:
        return 0.0
    max_lag = min(max_lag, len(values) // 2)
    if max_lag < 2:
        return 0.0
    cors = []
    for lag in range(2, max_lag + 1):
        left = centered[:-lag]
        right = centered[lag:]
        local_denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if local_denom > 1e-12:
            cors.append(abs(float(np.dot(left, right) / local_denom)))
    return float(max(cors)) if cors else 0.0


def noise_proxy(values: np.ndarray) -> float:
    """Estimate high-frequency noise with median absolute first difference."""
    values = clean_values(values)
    if len(values) < 2:
        return 0.0
    diffs = np.diff(values)
    return float(np.median(np.abs(diffs - np.median(diffs))))


def describe_record(record: TimeSeriesRecord) -> dict[str, float | int | str]:
    labels = np.asarray(record.labels, dtype=int)
    segments = anomaly_segments(labels)
    durations = [end - start for start, end in segments]
    return {
        "name": record.name,
        "source": record.source,
        "path": record.path,
        "length": int(len(record.values)),
        "anomaly_ratio": float(np.mean(labels)) if len(labels) else 0.0,
        "anomaly_segments": int(len(segments)),
        "mean_anomaly_duration": float(np.mean(durations)) if durations else 0.0,
        "max_anomaly_duration": int(max(durations)) if durations else 0,
        "seasonality_proxy": seasonality_proxy(record.values),
        "noise_proxy": noise_proxy(record.values),
    }


def crop_record(record: TimeSeriesRecord, max_length: int | None) -> TimeSeriesRecord:
    """Crop very long series to a deterministic segment that preserves anomalies.

    If anomalies exist, the crop is centered around the anomaly span. Otherwise,
    it uses the beginning of the series. This is a compute-safety feature for
    pilot runs; full experiments can disable it with a large max_length.
    """
    if max_length is None or max_length <= 0 or len(record.values) <= max_length:
        return record

    labels = np.asarray(record.labels, dtype=int)
    positives = np.where(labels > 0)[0]
    if positives.size:
        center = int((positives[0] + positives[-1]) // 2)
        start = max(0, center - max_length // 2)
        end = min(len(labels), start + max_length)
        start = max(0, end - max_length)
    else:
        start = 0
        end = max_length

    return TimeSeriesRecord(
        name=f"{record.name}__crop_{start}_{end}",
        source=record.source,
        path=record.path,
        values=record.values[start:end],
        labels=record.labels[start:end],
    )


def select_series(paths: list[Path], limit: int = 60) -> list[Path]:
    """Deterministically select files while preserving source-level diversity."""
    by_source: dict[str, list[Path]] = {}
    for path in paths:
        by_source.setdefault(path.parent.name, []).append(path)
    for source_paths in by_source.values():
        source_paths.sort()

    selected: list[Path] = []
    sources = sorted(by_source)
    round_idx = 0
    while len(selected) < limit:
        added = False
        for source in sources:
            source_paths = by_source[source]
            if round_idx < len(source_paths):
                selected.append(source_paths[round_idx])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        round_idx += 1
    return selected
