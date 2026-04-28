# 上台分享专用分析报告

日期：2026-04-28  
项目：Unsupervised Anomaly Detection for Univariate Time Series  
主线：从“谁最好”推进到“为什么在什么数据上最好”

---

## 0. 开场版本

这次项目研究的是单变量时间序列异常检测。我们不是只做一个算法排行榜，而是做一个解释型实证研究：在统一的数据、统一的预处理、统一的窗口嵌入、统一的 range-aware VUS-PR 指标下，比较六类无监督异常检测算法，并进一步分析数据特征、算法机制、运行时间、窗口大小、标准化和参数配置如何影响结果。

最终结论可以用三句话概括：

1. 在 60 条真实 TSB-UAD Public-v2 时间序列上，kNN 是整体表现最好的方法，平均 VUS-PR 为 0.4929，平均排名为 2.45。
2. Isolation Forest 不是最高分，但在效果和效率之间最均衡，平均 VUS-PR 为 0.4577，平均运行时间约 0.26 秒。
3. 官方 IDKD 方法与 Matrix Profile 都有 specialist 价值，但 IDKD 的平均运行时间明显更高，约 3.10 秒每条序列，因此它更适合作为机制分析对象，而不是当前设置下最经济的默认选择。

---

## 1. 作业要求如何转化为我们的研究问题

Assignment 要求我们围绕一种复杂数据类型做 empirical study，并且不能只是堆算法。它强调 aim statement、related work、methodology、many datasets、multiple algorithms、in-depth analysis。

因此我们的设计不是“找一个算法跑一下”，而是从一开始就确定为解释型实证研究：

> The aim of this study is to evaluate and explain the accuracy, robustness, and computational efficiency of representative unsupervised anomaly detection algorithms for univariate time-series data, with a focus on how dataset characteristics and algorithm mechanisms jointly affect detection performance.

这个 aim 对应四个研究问题：

- RQ1：哪些无监督算法在多数据集上整体表现最好？
- RQ2：异常比例、异常持续时间、周期性和噪声是否影响算法表现？
- RQ3：准确性和运行时间之间是否存在 trade-off？
- RQ4：算法失败是否能由自身机制解释？

上台时可以强调：我们的项目优势不在于“算法最多”，而在于每个实验都服务于这四个问题。

---

## 2. 数据与实验范围

我们聚焦单变量时间序列异常检测。主数据源是 TSB-UAD Public-v2。

真实主实验设置：

- 60 条可评估单变量时间序列
- 覆盖 23 个 source datasets
- 每条序列同时包含 normal points 和 anomaly points
- 多变量文件跳过
- 过长序列做 label-neutral deterministic middle crop；裁剪只看序列长度，不看 anomaly label

为什么不做 multivariate 或 deep learning：

- 课程要求更重视清楚的 empirical study 和解释，而不是模型复杂度。
- 两人小组、有限时间内，单变量无监督设置更容易保证公平比较。
- Deep learning 如果加入主实验，会带来训练集划分、超参数、硬件、训练稳定性等额外变量，不利于月底前给出可信结论。

---

## 3. 统一 Pipeline

所有算法都放进同一个离线无监督 pipeline：

1. Data loading：读取 value 和 label。
2. Cleaning：处理缺失值、无穷值，插值或填补。
3. Normalization：默认 z-normalization。
4. Sliding-window embedding：默认窗口 `w = max(16, min(128, floor(0.02n)))`。
5. Algorithm scoring：每个算法输出 anomaly score。
6. Window-to-point alignment：窗口分数映射回点分数，点分数取覆盖该点的窗口分数最大值。
7. Evaluation：label 只用于评价，不参与训练或调参。

这一点很重要，因为它保证六个算法的结果是在同一把尺子下比较的。

---

## 4. 算法家族与机制

我们比较六类方法。

### Rolling Z-Score

这是统计基线。它看每个点和局部均值相差多少个标准差。机制简单、速度快，但只能很好地处理幅度突变，对形状变化、上下文异常和复杂模式不敏感。

### kNN

kNN 是距离型方法。它把每个窗口看成一个向量，如果一个窗口离它最近的邻居也很远，就认为它异常。它适合捕捉“这个形状和大多数窗口都不像”的异常。

### LOF

LOF 是局部密度方法。它不是只看距离远不远，而是看一个点所在区域是否比附近区域更稀疏。如果某个窗口处在低密度区域，它会被认为更异常。它的优点是能处理局部结构，弱点是对重复值和邻居数比较敏感。

### Isolation Forest

Isolation Forest 是隔离型方法。直觉是异常点更容易被随机切分隔离出来。它通常速度较快，鲁棒性不错，但对时间序列形状信息的利用不如专门的 subsequence 方法。

### IDKD

现在的 `ik_density` 使用官方 `ikpykit.anomaly.IDKD`，也就是 Isolation Distributional Kernel Detector。它属于 Isolation Kernel family。IDKD 原生分数是“越低越异常”，我们在 pipeline 中取负号，让所有算法都保持“分数越高越异常”的统一约定。

它的价值是课程相关性强，机制上和普通距离或树模型不同。但完整官方实现的运行成本明显高于轻量 baseline。

### Matrix Profile

Matrix Profile 是时间序列专用方法。它寻找每个 subsequence 最近的相似 subsequence，如果最近邻距离很大，就说明这段形状很罕见。它在某些形状异常或 motif 结构明显的数据上有价值，但不是所有数据上都稳定。

---

## 5. 指标选择

我们最终使用 range-aware VUS-PR 作为主指标。

原因：

- AUROC 衡量整体排序能力，但在异常比例很低时可能过于乐观。
- AUPRC 更关注异常点这一正类，更适合 anomaly detection。
- VUS-ROC 和 VUS-PR 会考虑时间序列异常的范围容忍度，不要求每个异常点都在完全相同位置被命中。
- TSB-UAD / TSB-AD 官方评价实现支持 VUS，因此它更贴近当前 benchmark 语境。

技术上，我们已经从早期近似 VUS 切换到 range-aware VUS，并用 TSB-UAD / VUS 公开代码中的 `generate_curve` / `RangeAUC_volume_opt` 逻辑做了逐样例交叉核对。现在结果表里：

- `vus_pr` 是主指标
- `vus_roc` 是辅助指标
- `vus_pr_approx` 和 `vus_roc_approx` 只作为早期 sanity-check 对照，不用于主结论

---

## 6. 主实验结果

60 条真实时间序列上的平均表现如下：

| Algorithm | AUROC | AUPRC | VUS-ROC | VUS-PR | Runtime |
|---|---:|---:|---:|---:|---:|
| kNN | 0.7568 | 0.3012 | 0.8034 | 0.4929 | 0.41s |
| Isolation Forest | 0.7115 | 0.3026 | 0.7605 | 0.4577 | 0.26s |
| LOF | 0.6950 | 0.2506 | 0.7641 | 0.4416 | 0.43s |
| IDKD | 0.6494 | 0.2242 | 0.7156 | 0.3986 | 3.10s |
| Matrix Profile | 0.5721 | 0.1848 | 0.6548 | 0.3599 | 0.45s |
| Rolling Z-Score | 0.5897 | 0.2610 | 0.6628 | 0.2768 | 0.14s |

平均排名：

| Algorithm | Average Rank | Mean VUS-PR |
|---|---:|---:|
| kNN | 2.45 | 0.4929 |
| Isolation Forest | 3.11 | 0.4577 |
| LOF | 3.17 | 0.4416 |
| IDKD | 3.75 | 0.3986 |
| Matrix Profile | 4.18 | 0.3599 |
| Rolling Z-Score | 4.34 | 0.2768 |

注意一个细节：Isolation Forest 的 mean VUS-PR 和平均排名都略优于 LOF，且运行时间更短。这使它成为更强的 practical trade-off。

每条序列拿第一的次数：

- kNN：18 次
- LOF：11 次
- Rolling Z-Score：10 次
- Isolation Forest：9 次
- IDKD：7 次
- Matrix Profile：5 次

每条序列垫底的次数：

- Rolling Z-Score：21 次
- Matrix Profile：18 次
- IDKD：10 次
- Isolation Forest：7 次
- LOF：3 次
- kNN：1 次

这些结果支持三个结论：

1. kNN 是整体 winner。
2. Isolation Forest 是最均衡的 practical baseline。
3. Matrix Profile 和 IDKD 有特定场景价值，但不能作为整体最优。

---

## 7. RQ2：数据特征如何影响表现

我们按照 anomaly ratio、mean anomaly duration、seasonality proxy、noise proxy 分组。

### 异常比例

- Low anomaly ratio：kNN 最好，VUS-PR 约 0.3868。
- Medium anomaly ratio：Isolation Forest 最好，约 0.6658。
- High anomaly ratio：Isolation Forest 仍最好，约 0.4569。

解释：低异常比例时，距离型方法能比较清楚地找出少数离群窗口。异常比例变高后，异常窗口不再那么稀少，kNN 的邻居结构会被污染，Isolation Forest 的隔离机制更稳。

### 异常持续时间

- 短异常段：kNN 最好，约 0.5250。
- 中等异常段：kNN 最好，约 0.5142。
- 长异常段：Isolation Forest 最好，约 0.5271。

解释：短到中等异常段往往表现为局部窗口形状不同，kNN 能捕捉这种差异。长异常段会形成新的连续模式，很多窗口彼此相似，距离型方法优势下降。

### 周期性

- Low seasonality：kNN 最好，约 0.5069。
- Medium seasonality：kNN 最好，约 0.5206。
- High seasonality：LOF 最好，约 0.4699。

解释：周期性强时，局部密度结构更明显，LOF 的 density-based 假设更容易发挥作用。

### 噪声

- Low noise：kNN 最好，约 0.4060。
- Medium noise：kNN 最好，约 0.6190。
- High noise：Isolation Forest 最好，约 0.4645。

解释：高噪声会破坏距离结构，使 kNN 和 LOF 的邻近关系不稳定。Isolation Forest 更依赖随机隔离，对噪声更稳一些。

---

## 8. RQ3：准确率与效率的 trade-off

如果只看效果，kNN 最好。

如果同时看运行时间：

- Rolling Z-Score 最快，但效果最低。
- Isolation Forest 比 kNN 略低，但运行时间更短。
- IDKD 用官方完整实现后，平均运行时间约 3.10 秒，是主算法中最慢的，但 VUS-PR 只排第四。

因此我们可以说：

> kNN gives the best overall detection performance, while Isolation Forest provides the most attractive accuracy-efficiency trade-off.

报告里不要说 IDKD 不好。更准确的说法是：

> IDKD provides a principled isolation-kernel baseline, but under our full-series sliding-window setting it is computationally expensive and does not dominate simpler baselines.

---

## 9. RQ4：消融与失败模式

### 标准化影响

12 条真实数据的 normalization ablation 显示：

| Algorithm | Raw VUS-PR | Z-normalized VUS-PR |
|---|---:|---:|
| kNN | 0.3681 | 0.3709 |
| IDKD | 0.2151 | 0.3265 |
| Isolation Forest | 0.3383 | 0.3382 |
| LOF | 0.3746 | 0.3745 |
| Matrix Profile | 0.2109 | 0.2112 |
| Rolling Z-Score | 0.1086 | 0.1083 |

重点解释：

- IDKD 对标准化明显敏感，因为它依赖 kernel embedding；kNN 在这组中性裁剪的 12 条序列上变化较小。
- Isolation Forest 基本不受影响。
- Matrix Profile 在我们当前实现下变化很小，因为它本身偏向形状比较。

### 窗口大小影响

窗口大小对所有 window-based 方法都重要。本轮消融把窗口明确分成 `small_1pct`、`default_2pct`、`large_5pct` 三档，而不是按每条序列的实际窗口长度混合聚合。结果显示 kNN、LOF、IDKD 在默认 2% 窗口附近表现最好，Matrix Profile 在这批 12 条序列上更偏向小窗口。

这里应强调：我们没有用窗口消融去为每个数据集挑最优窗口，而是观察整体趋势，用来支持“算法对窗口参数敏感”这个结论。

### 参数敏感性

我们额外跑了 12 条真实序列、20 个参数配置，共 240 条结果。最优配置如下：

| Algorithm | Best Config | Mean VUS-PR | Runtime |
|---|---|---:|---:|
| LOF | k=10 | 0.4039 | 0.21s |
| kNN | k=10 | 0.3797 | 0.21s |
| Isolation Forest | n=200, max_samples=512 | 0.3452 | 0.23s |
| IDKD | n=200, max_samples=16 | 0.3265 | 2.49s |

重要解释：

- kNN 默认 `k=5` 已不错，但 `k=10` 在这批参数实验里更好。
- LOF 对 k 很敏感，`k=10` 在中性裁剪调参子集里均值最好。
- Isolation Forest 的最优配置转为 `n=200,max_samples=512`，但主实验默认配置仍更适合作为统一公平 baseline。
- IDKD 对 `max_samples` 很敏感，本轮中 `max_samples=16` 最好，增大到 32 后下降。

---

## 10. Case Study 如何讲

当前 VUS-PR 下自动选择的 4 张真实 case 图在：

- `figures/real_cases/NEK_10_scores.png`
- `figures/real_cases/14046_col_1__middlecrop_5404400_5424400_scores.png`
- `figures/real_cases/tao_pointg_10000_0.05_1_col_1_scores.png`
- `figures/real_cases/NEK_0_scores.png`

上台不建议四张都细讲。建议只讲两张：

1. 成功案例：展示 kNN 或 Isolation Forest 的分数峰值和 label 对齐较好。
2. 失败或差异案例：展示某些算法在异常区没有明显抬高，说明机制假设不匹配。

讲 case 图时不要只说“红色是异常”。要围绕机制解释：

- 如果 kNN 成功：异常窗口在嵌入空间中离多数窗口远。
- 如果 Isolation Forest 成功：异常段更容易被随机切分隔离。
- 如果 Matrix Profile 失败：异常并没有表现为罕见 subsequence，或者窗口长度不匹配。
- 如果 Rolling Z-Score 失败：异常不是简单幅度尖峰。

---

## 11. Limitations

需要主动说明限制，反而会让报告更可信。

1. 我们使用的是离线 full-series scoring，不是在线检测。
2. 主实验使用 60 条真实序列，足够满足 many datasets，但不是 TSB-UAD 全量 3427 条。
3. 多变量数据被跳过，因为项目 scope 是 univariate。
4. IDKD 使用官方实现，但放在我们统一的 sliding-window embedding 上，与原始时间序列专用 IDK 变体不同。
5. 参数敏感性实验只在 12 条真实序列上做，用于分析 robustness，不用于替代主实验结论。
6. 默认窗口和 point-level score alignment 是研究设计选择，可能影响 Matrix Profile 和 window-based 方法。
7. 为避免 label-aware cropping，过长序列使用中性中心裁剪；因此有些 source 的候选序列因为裁剪后单类标签而被跳过，最终覆盖 23 个 source datasets。

---

## 12. 最终结论

完整结论可以这样讲：

> Across 60 real univariate time series from TSB-UAD Public-v2, selected with label-neutral cropping, kNN achieves the best overall detection performance under range-aware VUS-PR, suggesting that distance-based subsequence comparison is a strong baseline for offline unsupervised time-series anomaly detection. However, Isolation Forest provides a more favorable accuracy-efficiency trade-off and becomes more robust under higher anomaly ratios, longer anomaly durations, and noisier series. LOF and Matrix Profile show specialist behavior under certain structural conditions, while Rolling Z-Score remains a useful but weak statistical baseline. Official IDKD improves methodological rigor and course relevance, but under our sliding-window setup it is computationally expensive and sensitive to max_samples.

中文版本：

> 在 60 条真实单变量时间序列上，使用不依赖标签的中性裁剪后，kNN 在 VUS-PR 下整体最好，说明基于窗口距离的子序列比较是一个强 baseline。但如果考虑运行效率和稳定性，Isolation Forest 更均衡，尤其在异常比例较高、异常持续时间较长或噪声较强的数据上更稳。LOF、Matrix Profile 和 IDKD 都有特定场景价值，但不是整体最优。Rolling Z-Score 很快，但只能作为弱统计基线。

---

## 13. 推荐上台结构

建议 12 分钟 presentation 分配：

1. 1 分钟：任务背景和 aim。
2. 1 分钟：为什么选择 univariate time-series anomaly detection。
3. 2 分钟：pipeline、数据和指标，强调 range-aware VUS-PR。
4. 2 分钟：六个算法机制。
5. 2 分钟：主实验结果。
6. 1.5 分钟：数据特征分析。
7. 1.5 分钟：消融和参数敏感性。
8. 1 分钟：case study 和 limitations。
9. 结尾 30 秒：最终 takeaways。

核心 takeaway 放最后一页：

- kNN is the best overall detector.
- Isolation Forest is the best accuracy-efficiency compromise.
- Algorithm performance depends strongly on data characteristics and parameters.
- Range-aware VUS-PR, label-neutral cropping, and mechanism-based analysis make the study more than a simple leaderboard.
