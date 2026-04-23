# Data Mining Assignment: Time-Series Anomaly Detection

本工程用于实现课程 assignment 的实验部分：比较并解释多种 unsupervised anomaly detection algorithms（无监督异常检测算法）在 univariate time-series data（单变量时间序列）上的 accuracy（准确性）、robustness（鲁棒性）和 runtime（运行时间）。

## Current Status

- `最终方案.md`：最终研究方案存档。
- `src/tsad_benchmark/`：实验 pipeline 代码。
- `scripts/run_synthetic_pilot.py`：先运行 synthetic mechanism test（合成机制测试），验证算法和评价流程。
- `scripts/run_benchmark.py`：用于 TSB-UAD 格式真实数据集的批量实验。
- `scripts/summarize_results.py`：汇总实验结果并生成核心图表。

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python scripts/run_synthetic_pilot.py
PYTHONPATH=src python scripts/summarize_results.py --input results/synthetic_results.csv --output-dir figures/synthetic
PYTHONPATH=src python scripts/run_ablation.py --mode synthetic --output results/synthetic_ablation_results.csv
PYTHONPATH=src python scripts/summarize_ablation.py --input results/synthetic_ablation_results.csv --output-dir figures/synthetic_ablation
PYTHONPATH=src python scripts/write_synthetic_tsb.py
PYTHONPATH=src python scripts/run_benchmark.py --data-root data/raw/synthetic_tsb --limit 4 --output results/smoke_benchmark_results.csv --manifest results/smoke_manifest.csv --scores-dir results/smoke_scores
```

## Real Dataset Setup

TSB-UAD 格式文件通常是两列、无表头：

- 第 1 列：time-series value（时间序列值）
- 第 2 列：anomaly label（异常标签，0/1）

下载 TSB-UAD Public-v2 后，将解压目录放到：

```text
data/raw/TSB-UAD-Public-v2/
```

然后运行：

```bash
bash scripts/download_tsb_uad_public_v2.sh
PYTHONPATH=src python scripts/run_benchmark.py --data-root data/raw/TSB-UAD-Public-v2 --limit 60 --max-length 20000
PYTHONPATH=src python scripts/summarize_results.py --input results/real_main_results.csv --output-dir figures/real_main_summary
PYTHONPATH=src python scripts/summarize_characteristics.py --input results/real_main_results.csv --output-dir figures/real_characteristics
PYTHONPATH=src python scripts/make_case_figures.py --results results/real_main_results.csv --scores-dir results/real_main_scores --output-dir figures/real_cases
PYTHONPATH=src python scripts/run_ablation.py --mode real --data-root data/raw/TSB-UAD-Public-v2 --limit 12 --max-length 12000 --output results/real_ablation_results.csv
PYTHONPATH=src python scripts/summarize_ablation.py --input results/real_ablation_results.csv --output-dir figures/real_ablation
```

如果下载中断，重新运行同一个脚本即可断点续传。

## Notes

- 本实现中的 `vus_roc_approx` 和 `vus_pr_approx` 是 range-dilated VUS-style approximation（基于标签膨胀的近似 VUS 指标），用于项目初期快速验证。最终报告若需要严格对齐 TSB-UAD，应替换或核对官方 VUS 实现。
- Matrix Profile 优先使用 `stumpy`；如果环境没有安装，会回退到 pairwise subsequence distance fallback（成对窗口距离回退实现），长序列上会较慢。
- `IKDensityDetector` 是轻量级 Isolation Kernel family detector（隔离核家族检测器），用于课程相关方法的可执行 baseline；最终报告应准确描述实现而不夸大为官方完整 IDK 实现。
