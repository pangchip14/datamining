from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tsad_benchmark.plots import save_case_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create representative case-study overlays.")
    parser.add_argument("--results", default="results/real_main_results.csv")
    parser.add_argument("--scores-dir", default="results/real_main_scores")
    parser.add_argument("--output-dir", default="figures/real_cases")
    parser.add_argument("--metric", default="vus_pr")
    parser.add_argument("--cases", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_col = "series_id" if "series_id" in results.columns else "name"
    best_rows = results.loc[results.groupby(group_col)[args.metric].idxmax()]
    selected = []
    for algorithm in best_rows["algorithm"].value_counts().index:
        row = best_rows[best_rows["algorithm"] == algorithm].sort_values(args.metric, ascending=False).iloc[0]
        selected.append(row)
        if len(selected) >= args.cases:
            break

    for row in selected:
        score_file = row["score_file"] if "score_file" in row and pd.notna(row["score_file"]) else f"{str(row['name']).replace('/', '_').replace(' ', '_')}_scores.csv"
        score_csv = Path(args.scores_dir) / score_file
        if not score_csv.exists():
            print(f"missing {score_csv}")
            continue
        case_rows = results[results[group_col] == row[group_col]].sort_values(args.metric, ascending=False)
        algorithms = []
        for algorithm in list(case_rows["algorithm"]) + ["rolling_zscore"]:
            if algorithm not in algorithms:
                algorithms.append(algorithm)
            if len(algorithms) >= 4:
                break
        save_case_overlay(score_csv, output_dir / f"{score_csv.stem}.png", algorithms=algorithms)
        print(f"wrote {output_dir / f'{score_csv.stem}.png'}")


if __name__ == "__main__":
    main()
