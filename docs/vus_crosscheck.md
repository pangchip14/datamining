# VUS 交叉核对记录

日期：2026-04-28

## 核对目标

确认项目中的 `src/tsad_benchmark/vus.py` 是否与 TSB-UAD / VUS 公开实现的核心 VUS 逻辑一致。

## 官方参考

- `vus.metrics.get_metrics` 调用 `generate_curve` 计算 `VUS_ROC` 和 `VUS_PR`：<https://github.com/TheDatumOrg/VUS/blob/main/vus/metrics.py>
- `generate_curve` 调用 `metricor().RangeAUC_volume_opt(...)`：<https://github.com/TheDatumOrg/VUS/blob/main/vus/analysis/robustness_eval.py>
- `RangeAUC_volume_opt` 的核心实现位于：<https://github.com/TheDatumOrg/VUS/blob/main/vus/utils/metrics.py>

## 核对方式

本地没有把 PyPI `vus` 包加入主依赖，因为该包会拉取 TensorFlow 等重依赖，不适合本课程项目的轻量复现实验环境。核对时直接读取官方 GitHub 源码，按 `RangeAUC_volume_opt` 的核心逻辑写出独立 direct transcription，然后与项目中的向量化 `official_vus` 在多组二值标签和连续分数上比较。

## 核对结果

使用 `thre=50`、不同序列长度和 `slidingWindow` 的三组样例，最大绝对差为浮点误差量级：

| n | slidingWindow | Max absolute difference |
|---:|---:|---:|
| 80 | 10 | 0.0 |
| 120 | 20 | 1.11e-16 |
| 250 | 35 | 2.22e-16 |

因此当前实现可以视为与官方核心 VUS 计算一致的 range-aware VUS 实现。报告中仍避免把早期 `vus_pr_approx` 作为主指标；`vus_pr_approx` 只保留为 sanity-check 对照。

