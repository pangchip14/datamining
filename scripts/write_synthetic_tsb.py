from __future__ import annotations

from pathlib import Path

import pandas as pd

from tsad_benchmark.synthetic import make_synthetic_suite


def main() -> None:
    root = Path("data/raw/synthetic_tsb")
    for record in make_synthetic_suite():
        source = root / record.source
        source.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"value": record.values, "label": record.labels})
        df.to_csv(source / f"{record.name}.out", index=False, header=False)
    print(f"wrote synthetic TSB-style files to {root}")


if __name__ == "__main__":
    main()
