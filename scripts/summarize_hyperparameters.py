from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize hyperparameter sensitivity results.")
    parser.add_argument("--input", default="results/real_hyperparameter_results.csv")
    parser.add_argument("--output-dir", default="figures/real_hyperparameters")
    parser.add_argument("--metric", default="vus_pr")
    return parser.parse_args()


def save_algorithm_plot(summary: pd.DataFrame, algorithm: str, metric: str, output: Path) -> None:
    subset = summary[summary["algorithm"] == algorithm].sort_values(metric, ascending=False)
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(subset["config_id"], subset[metric])
    ax.set_title(f"{algorithm} parameter sensitivity")
    ax.set_ylabel(f"Mean {metric}")
    ax.set_xlabel("configuration")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_runtime_tradeoff(summary: pd.DataFrame, metric: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for algorithm, group in summary.groupby("algorithm"):
        ax.scatter(group["runtime_sec"], group[metric], label=algorithm, s=70)
        best = group.sort_values(metric, ascending=False).iloc[0]
        ax.annotate(best["config_id"], (best["runtime_sec"], best[metric]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean runtime per series (seconds)")
    ax.set_ylabel(f"Mean {metric}")
    ax.set_title("Hyperparameter accuracy-efficiency trade-off")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(args.input)
    group_col = "series_id" if "series_id" in results.columns else "name"
    ranked = results.copy()
    ranked["config_rank_within_algorithm"] = ranked.groupby([group_col, "algorithm"])[args.metric].rank(
        ascending=False,
        method="average",
    )

    summary = (
        ranked.groupby(["algorithm", "config_id"], as_index=False)
        .agg(
            auroc=("auroc", "mean"),
            auprc=("auprc", "mean"),
            vus_roc=("vus_roc", "mean"),
            vus_pr=("vus_pr", "mean"),
            vus_roc_approx=("vus_roc_approx", "mean"),
            vus_pr_approx=("vus_pr_approx", "mean"),
            runtime_sec=("runtime_sec", "mean"),
            average_config_rank=("config_rank_within_algorithm", "mean"),
            n=(group_col, "nunique"),
        )
        .sort_values(["algorithm", "average_config_rank", args.metric], ascending=[True, True, False])
    )
    summary_path = output_dir / "hyperparameter_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")

    best = summary.loc[summary.groupby("algorithm")[args.metric].idxmax()].sort_values(args.metric, ascending=False)
    best_path = output_dir / "best_by_algorithm.csv"
    best.to_csv(best_path, index=False)
    print(f"wrote {best_path}")

    for algorithm in sorted(summary["algorithm"].unique()):
        save_algorithm_plot(summary, algorithm, args.metric, output_dir / f"{algorithm}_sensitivity.png")
    save_runtime_tradeoff(summary, args.metric, output_dir / "hyperparameter_runtime_tradeoff.png")
    print(f"wrote hyperparameter figures to {output_dir}")


if __name__ == "__main__":
    main()
