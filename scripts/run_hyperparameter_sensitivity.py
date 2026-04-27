from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tsad_benchmark.algorithms import (
    window_ik_density_scores,
    window_isolation_forest_scores,
    window_knn_scores,
    window_lof_scores,
)
from tsad_benchmark.config import DEFAULT_VUS_THRESHOLDS, DEFAULT_VUS_WINDOW, default_window_size
from tsad_benchmark.data import crop_record, describe_record, discover_series_files, load_tsb_file, select_series
from tsad_benchmark.metrics import evaluate_scores
from tsad_benchmark.preprocessing import align_window_scores, sliding_windows, z_normalize


def is_evaluable(record) -> bool:
    positives = int(record.labels.sum())
    return 0 < positives < len(record.labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-data hyperparameter sensitivity experiments.")
    parser.add_argument("--data-root", default="data/raw/TSB-UAD-Public-v2")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=12000)
    parser.add_argument("--output", default="results/real_hyperparameter_results.csv")
    parser.add_argument("--vus-window", type=int, default=DEFAULT_VUS_WINDOW)
    parser.add_argument("--vus-thresholds", type=int, default=DEFAULT_VUS_THRESHOLDS)
    return parser.parse_args()


def load_records(data_root: str, limit: int, max_length: int):
    records = []
    skipped = 0
    for path in select_series(discover_series_files(data_root), limit=10_000):
        if len(records) >= limit:
            break
        try:
            record = crop_record(load_tsb_file(path), max_length=max_length)
            if not is_evaluable(record):
                skipped += 1
                print(f"skip {path}: labels are single-class after cropping")
                continue
            records.append(record)
        except Exception as exc:
            skipped += 1
            print(f"skip {path}: {exc}")
    print(f"loaded {len(records)} records, skipped {skipped} candidates")
    return records


def config_grid() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k in (3, 5, 10, 20):
        rows.append({"algorithm": "knn", "config_id": f"k={k}", "n_neighbors": k})
    for k in (10, 20, 35, 50):
        rows.append({"algorithm": "lof", "config_id": f"k={k}", "n_neighbors": k})
    for n_estimators in (100, 200):
        for max_samples in ("auto", 256, 512):
            rows.append(
                {
                    "algorithm": "isolation_forest",
                    "config_id": f"n={n_estimators},max_samples={max_samples}",
                    "n_estimators": n_estimators,
                    "max_samples": max_samples,
                }
            )
    for n_estimators in (100, 200):
        for max_samples in (8, 16, 32):
            rows.append(
                {
                    "algorithm": "ik_density",
                    "config_id": f"n={n_estimators},max_samples={max_samples}",
                    "n_estimators": n_estimators,
                    "max_samples": max_samples,
                    "method": "inne",
                }
            )
    return rows


def run_config(windows, config: dict[str, object]):
    algorithm = config["algorithm"]
    if algorithm == "knn":
        return window_knn_scores(windows, n_neighbors=int(config["n_neighbors"]))
    if algorithm == "lof":
        return window_lof_scores(windows, n_neighbors=int(config["n_neighbors"]))
    if algorithm == "isolation_forest":
        return window_isolation_forest_scores(
            windows,
            n_estimators=int(config["n_estimators"]),
            max_samples=config["max_samples"],
        )
    if algorithm == "ik_density":
        return window_ik_density_scores(
            windows,
            n_estimators=int(config["n_estimators"]),
            max_samples=config["max_samples"],
            method=str(config["method"]),
        )
    raise ValueError(f"unknown algorithm: {algorithm}")


def main() -> None:
    args = parse_args()
    records = load_records(args.data_root, args.limit, args.max_length)
    if not records:
        raise SystemExit("no records available")

    configs = config_grid()
    rows = []
    for record in tqdm(records, desc="hyperparameter records"):
        meta = describe_record(record)
        window = default_window_size(len(record.values))
        model_values = z_normalize(record.values)
        windows = sliding_windows(model_values, window)
        for config in configs:
            start = time.perf_counter()
            window_scores = run_config(windows, config)
            point_scores = align_window_scores(window_scores, len(record.values), window, mode="max")
            runtime = time.perf_counter() - start
            metrics = evaluate_scores(
                record.labels,
                point_scores,
                vus_window=args.vus_window,
                vus_thresholds=args.vus_thresholds,
            )
            rows.append(
                {
                    **meta,
                    **config,
                    "window": window,
                    "normalize": True,
                    "vus_window": args.vus_window,
                    "vus_thresholds": args.vus_thresholds,
                    "runtime_sec": runtime,
                    **metrics,
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
