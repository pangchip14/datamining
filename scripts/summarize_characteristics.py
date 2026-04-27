from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FEATURES = ("anomaly_ratio", "length", "mean_anomaly_duration", "seasonality_proxy", "noise_proxy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize performance by dataset characteristics.")
    parser.add_argument("--input", default="results/real_main_results.csv")
    parser.add_argument("--output-dir", default="figures/real_characteristics")
    parser.add_argument("--metric", default="vus_pr")
    return parser.parse_args()


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for feature in FEATURES:
        if feature not in out:
            continue
        values = out[feature]
        unique = values.nunique(dropna=True)
        if unique < 3:
            out[f"{feature}_bin"] = "all"
            continue
        try:
            out[f"{feature}_bin"] = pd.qcut(values, q=3, labels=["low", "medium", "high"], duplicates="drop")
        except ValueError:
            out[f"{feature}_bin"] = "all"
    return out


def save_feature_plot(summary: pd.DataFrame, feature: str, metric: str, output: Path) -> None:
    bin_col = f"{feature}_bin"
    subset = summary[summary["feature"] == feature]
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    pivot = subset.pivot(index=bin_col, columns="algorithm", values=metric)
    pivot = pivot.reindex([idx for idx in ["low", "medium", "high", "all"] if idx in pivot.index])
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel(f"Mean {metric}")
    ax.set_title(f"Performance by {feature}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = add_bins(pd.read_csv(args.input))

    rows = []
    for feature in FEATURES:
        bin_col = f"{feature}_bin"
        if bin_col not in results:
            continue
        grouped = (
            results.groupby([bin_col, "algorithm"], observed=True)
            .agg(
                auroc=("auroc", "mean"),
                auprc=("auprc", "mean"),
                vus_roc=("vus_roc", "mean"),
                vus_pr=("vus_pr", "mean"),
                vus_roc_approx=("vus_roc_approx", "mean"),
                vus_pr_approx=("vus_pr_approx", "mean"),
                runtime_sec=("runtime_sec", "mean"),
                n=("name", "nunique"),
            )
            .reset_index()
            .rename(columns={bin_col: f"{feature}_bin"})
        )
        grouped["feature"] = feature
        rows.append(grouped)

    summary = pd.concat(rows, ignore_index=True)
    summary_path = output_dir / "characteristic_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")
    for feature in FEATURES:
        save_feature_plot(summary, feature, args.metric, output_dir / f"{feature}_effect.png")
    print(f"wrote characteristic figures to {output_dir}")


if __name__ == "__main__":
    main()
