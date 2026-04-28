from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ablation result CSV.")
    parser.add_argument("--input", default="results/synthetic_ablation_results.csv")
    parser.add_argument("--output-dir", default="figures/ablation")
    parser.add_argument("--metric", default="vus_pr")
    return parser.parse_args()


def save_line_plot(summary: pd.DataFrame, ablation: str, metric: str, output: Path) -> None:
    subset = summary[summary["ablation"] == ablation].copy()
    if subset.empty:
        return
    if ablation == "window_size":
        order = {"small_1pct": 0, "default_2pct": 1, "large_5pct": 2}
    elif ablation == "normalization":
        order = {"raw": 0, "z_normalized": 1}
    else:
        order = {}
    subset["_order"] = subset["ablation_value"].map(order).fillna(999)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algorithm, group in subset.groupby("algorithm"):
        group = group.sort_values(["_order", "ablation_value"])
        ax.plot(group["ablation_value"], group[metric], marker="o", label=algorithm)
    ax.set_xlabel(ablation)
    ax.set_ylabel(f"Mean {metric}")
    ax.set_title(f"{ablation} effect")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(args.input)
    summary = (
        results.groupby(["ablation", "ablation_value", "algorithm"], as_index=False)
        .agg(
            auroc=("auroc", "mean"),
            auprc=("auprc", "mean"),
            vus_roc=("vus_roc", "mean"),
            vus_pr=("vus_pr", "mean"),
            vus_roc_approx=("vus_roc_approx", "mean"),
            vus_pr_approx=("vus_pr_approx", "mean"),
            runtime_sec=("runtime_sec", "mean"),
        )
    )
    summary_path = output_dir / "ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")

    save_line_plot(summary, "window_size", args.metric, output_dir / "window_size_effect.png")
    save_line_plot(summary, "normalization", args.metric, output_dir / "normalization_effect.png")
    print(f"wrote ablation figures to {output_dir}")


if __name__ == "__main__":
    main()
