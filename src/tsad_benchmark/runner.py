from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .algorithms import run_algorithm
from .config import MAIN_ALGORITHMS, default_window_size
from .data import TimeSeriesRecord, describe_record
from .metrics import evaluate_scores


def run_record(
    record: TimeSeriesRecord,
    algorithms: tuple[str, ...] = MAIN_ALGORITHMS,
    window: int | None = None,
    normalize: bool = True,
    save_scores_dir: str | Path | None = None,
) -> list[dict[str, object]]:
    if window is None:
        window = default_window_size(len(record.values))
    rows: list[dict[str, object]] = []
    meta = describe_record(record)

    score_frame = pd.DataFrame({"label": record.labels, "value": record.values})
    for algorithm in algorithms:
        result = run_algorithm(algorithm, record.values, window=window, normalize=normalize)
        metrics = evaluate_scores(record.labels, result.scores)
        row: dict[str, object] = {
            **meta,
            "algorithm": algorithm,
            "runtime_sec": result.runtime_sec,
            "window": window,
            "normalize": normalize,
            **metrics,
        }
        rows.append(row)
        score_frame[f"score_{algorithm}"] = result.scores

    if save_scores_dir is not None:
        out_dir = Path(save_scores_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = record.name.replace("/", "_").replace(" ", "_")
        score_frame.to_csv(out_dir / f"{safe_name}_scores.csv", index=False)
    return rows


def run_records(
    records: list[TimeSeriesRecord],
    algorithms: tuple[str, ...] = MAIN_ALGORITHMS,
    normalize: bool = True,
    save_scores_dir: str | Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in tqdm(records, desc="records"):
        rows.extend(
            run_record(
                record,
                algorithms=algorithms,
                normalize=normalize,
                save_scores_dir=save_scores_dir,
            )
        )
    return pd.DataFrame(rows)


def average_ranks(results: pd.DataFrame, metric: str = "vus_pr_approx") -> pd.DataFrame:
    ranks = results.copy()
    ranks["rank"] = ranks.groupby("name")[metric].rank(ascending=False, method="average")
    return (
        ranks.groupby("algorithm", as_index=False)
        .agg(
            average_rank=("rank", "mean"),
            mean_metric=(metric, "mean"),
            median_metric=(metric, "median"),
            mean_runtime_sec=("runtime_sec", "mean"),
        )
        .sort_values(["average_rank", "mean_metric"], ascending=[True, False])
    )


def metric_ranking_correlation(
    results: pd.DataFrame,
    metrics: tuple[str, ...] = ("auroc", "auprc", "vus_roc_approx", "vus_pr_approx"),
) -> pd.DataFrame:
    rank_table = {}
    for metric in metrics:
        summary = average_ranks(results, metric=metric)
        rank_table[metric] = summary.set_index("algorithm")["average_rank"]
    frame = pd.DataFrame(rank_table)
    return frame.corr(method="spearman")
