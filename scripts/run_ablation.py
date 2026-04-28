from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tsad_benchmark.config import DEFAULT_VUS_THRESHOLDS, DEFAULT_VUS_WINDOW, MAIN_ALGORITHMS, default_window_size
from tsad_benchmark.data import crop_record, discover_series_files, load_tsb_file, select_series
from tsad_benchmark.runner import run_record
from tsad_benchmark.synthetic import make_synthetic_suite


def is_evaluable(record) -> bool:
    positives = int(record.labels.sum())
    return 0 < positives < len(record.labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run core ablations for the assignment.")
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--data-root", default="data/raw/TSB-UAD-Public-v2")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", default="results/ablation_results.csv")
    parser.add_argument("--max-length", type=int, default=12000)
    parser.add_argument(
        "--crop-strategy",
        choices=["middle", "head", "tail", "label_centered"],
        default="middle",
    )
    parser.add_argument("--vus-window", type=int, default=DEFAULT_VUS_WINDOW)
    parser.add_argument("--vus-thresholds", type=int, default=DEFAULT_VUS_THRESHOLDS)
    return parser.parse_args()


def load_records(mode: str, data_root: str, limit: int, max_length: int, crop_strategy: str):
    if mode == "synthetic":
        return make_synthetic_suite()
    records = []
    skipped = 0
    for path in select_series(discover_series_files(data_root), limit=10_000):
        if len(records) >= limit:
            break
        try:
            record = crop_record(load_tsb_file(path), max_length=max_length, strategy=crop_strategy)
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


def main() -> None:
    args = parse_args()
    records = load_records(args.mode, args.data_root, args.limit, args.max_length, args.crop_strategy)
    if not records:
        raise SystemExit("no records available")

    rows = []
    for record in tqdm(records, desc="ablation records"):
        base_window = default_window_size(len(record.values))
        window_grid = [
            ("small_1pct", max(4, min(len(record.values), int(0.01 * len(record.values))))),
            ("default_2pct", base_window),
            ("large_5pct", max(4, min(len(record.values), int(0.05 * len(record.values))))),
        ]
        seen_windows = set()
        for window_label, window in window_grid:
            if window in seen_windows:
                continue
            seen_windows.add(window)
            ablation_rows = run_record(
                record,
                algorithms=MAIN_ALGORITHMS,
                window=window,
                normalize=True,
                save_scores_dir=None,
                vus_window=args.vus_window,
                vus_thresholds=args.vus_thresholds,
            )
            for row in ablation_rows:
                row["ablation"] = "window_size"
                row["ablation_value"] = window_label
                row["window"] = int(window)
            rows.extend(ablation_rows)

        for normalize in (False, True):
            ablation_rows = run_record(
                record,
                algorithms=MAIN_ALGORITHMS,
                window=base_window,
                normalize=normalize,
                save_scores_dir=None,
                vus_window=args.vus_window,
                vus_thresholds=args.vus_thresholds,
            )
            for row in ablation_rows:
                row["ablation"] = "normalization"
                row["ablation_value"] = "z_normalized" if normalize else "raw"
            rows.extend(ablation_rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
