# Dataset Setup

## TSB-UAD Public-v2

主实验计划使用 TSB-UAD Public-v2。数据下载后建议解压到：

```text
data/raw/TSB-UAD-Public-v2/
```

TSB-UAD formatted file（TSB-UAD 格式文件）通常为两列、无表头：

```text
value,label
```

代码会递归搜索 `.out`、`.csv`、`.txt` 文件，并尝试读取前两列作为 value 和 label。

## Sampling Rule

为了满足 assignment 中的 many datasets（多数据集）要求，同时避免 cherry-picking（选择性挑选），真实数据实验采用 deterministic sampling（确定性抽样）：

- 至少覆盖多个 source datasets（来源数据集），source 默认由父目录名识别。
- 每个 source 优先抽取若干条序列，直到达到 `--limit 60`。
- 记录每条序列的 length、anomaly_ratio、mean_anomaly_duration、seasonality_proxy、noise_proxy。

## Synthetic Mechanism Test

Synthetic test（合成测试）不是替代真实数据，而是用来支持 RQ4 的 mechanism explanation（机制解释）。当前包含：

- point_spikes：点异常。
- collective_shift：连续区间形态/均值异常。
- contextual_phase：上下文异常。
- noisy_subsequence：高噪声下的局部形态异常。
