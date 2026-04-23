from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tsad_benchmark.plots import save_metric_correlation_heatmap, save_runtime_tradeoff
from tsad_benchmark.runner import average_ranks, metric_ranking_correlation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize benchmark results.")
    parser.add_argument("--input", default="results/synthetic_results.csv", help="Input result CSV.")
    parser.add_argument("--output-dir", default="figures/summary", help="Directory for figures and tables.")
    parser.add_argument("--metric", default="vus_pr_approx", help="Main metric for ranking.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(args.input)
    ranking = average_ranks(results, metric=args.metric)
    ranking_path = output_dir / "average_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    print(f"wrote {ranking_path}")

    corr = metric_ranking_correlation(results)
    corr_path = output_dir / "metric_ranking_correlation.csv"
    corr.to_csv(corr_path)
    print(f"wrote {corr_path}")

    save_metric_correlation_heatmap(corr, output_dir / "metric_ranking_correlation.png")
    save_runtime_tradeoff(results, output_dir / "runtime_tradeoff.png", metric=args.metric)
    print(f"wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
