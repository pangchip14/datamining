from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tsad_benchmark.data import crop_record, describe_record, discover_series_files, load_tsb_file, select_series
from tsad_benchmark.runner import run_records


def is_evaluable(record) -> bool:
    labels = record.labels
    positives = int(labels.sum())
    return 0 < positives < len(labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TSB-UAD-style benchmark.")
    parser.add_argument("--data-root", default="data/raw/TSB-UAD-Public-v2", help="Root directory containing TSB-UAD files.")
    parser.add_argument("--limit", type=int, default=60, help="Maximum number of time series to run.")
    parser.add_argument("--output", default="results/benchmark_results.csv", help="Result CSV path.")
    parser.add_argument("--manifest", default="results/benchmark_manifest.csv", help="Selected dataset manifest path.")
    parser.add_argument("--scores-dir", default="results/benchmark_scores", help="Directory for point-level scores.")
    parser.add_argument("--max-length", type=int, default=20000, help="Crop long series to this many points; <=0 disables cropping.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_series_files(args.data_root)
    candidate_files = select_series(files, limit=len(files))
    if not candidate_files:
        raise SystemExit(f"No supported files found under {args.data_root}")

    records = []
    skipped = 0
    for path in candidate_files:
        if len(records) >= args.limit:
            break
        try:
            record = load_tsb_file(path)
            record = crop_record(record, max_length=args.max_length)
            if not is_evaluable(record):
                skipped += 1
                print(f"skip {path}: labels are single-class after cropping")
                continue
            records.append(record)
        except Exception as exc:
            skipped += 1
            print(f"skip {path}: {exc}")

    if not records:
        raise SystemExit("No loadable two-column time-series files found.")
    print(f"loaded {len(records)} records, skipped {skipped} candidates")

    manifest = pd.DataFrame([describe_record(record) for record in records])
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print(f"wrote {manifest_path}")

    results = run_records(records, save_scores_dir=args.scores_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
