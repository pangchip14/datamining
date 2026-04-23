from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .preprocessing import minmax_scale


def save_runtime_tradeoff(results: pd.DataFrame, output: str | Path, metric: str = "vus_pr_approx") -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = (
        results.groupby("algorithm", as_index=False)
        .agg(metric=(metric, "mean"), runtime_sec=("runtime_sec", "mean"))
        .sort_values("metric", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(summary["runtime_sec"], summary["metric"], s=80)
    for _, row in summary.iterrows():
        ax.annotate(row["algorithm"], (row["runtime_sec"], row["metric"]), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean runtime per series (seconds)")
    ax.set_ylabel(f"Mean {metric}")
    ax.set_title("Accuracy-efficiency trade-off")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_metric_correlation_heatmap(corr: pd.DataFrame, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Metric ranking correlation")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_case_overlay(score_csv: str | Path, output: str | Path, algorithms: list[str] | None = None) -> None:
    score_csv = Path(score_csv)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(score_csv)
    if algorithms is None:
        algorithms = [col.removeprefix("score_") for col in df.columns if col.startswith("score_")][:3]

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    values = minmax_scale(df["value"].to_numpy())
    ax.plot(x, values, label="value (scaled)", color="black", linewidth=1.0)
    labels = df["label"].to_numpy(dtype=int)
    anomaly_idx = np.where(labels > 0)[0]
    if anomaly_idx.size:
        ax.fill_between(x, 0, 1, where=labels > 0, color="tab:red", alpha=0.18, label="label")
    for algorithm in algorithms:
        col = f"score_{algorithm}"
        if col in df:
            ax.plot(x, minmax_scale(df[col].to_numpy()), label=algorithm, alpha=0.85)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(score_csv.stem.replace("_scores", ""))
    ax.set_xlabel("time")
    ax.set_ylabel("scaled value / score")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
