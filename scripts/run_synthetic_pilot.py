from __future__ import annotations

from pathlib import Path

from tsad_benchmark.config import DEFAULT_VUS_THRESHOLDS, DEFAULT_VUS_WINDOW
from tsad_benchmark.plots import save_case_overlay
from tsad_benchmark.runner import run_records
from tsad_benchmark.synthetic import make_synthetic_suite


def main() -> None:
    results_dir = Path("results")
    score_dir = results_dir / "synthetic_scores"
    figure_dir = Path("figures") / "synthetic_cases"
    results_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    records = make_synthetic_suite()
    results = run_records(
        records,
        save_scores_dir=score_dir,
        vus_window=DEFAULT_VUS_WINDOW,
        vus_thresholds=DEFAULT_VUS_THRESHOLDS,
    )
    output = results_dir / "synthetic_results.csv"
    results.to_csv(output, index=False)
    print(f"wrote {output}")

    for csv_path in sorted(score_dir.glob("*_scores.csv")):
        save_case_overlay(csv_path, figure_dir / f"{csv_path.stem}.png")
    print(f"wrote case figures to {figure_dir}")


if __name__ == "__main__":
    main()
