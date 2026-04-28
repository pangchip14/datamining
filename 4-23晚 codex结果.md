下面是对第一版真实实验结果的完整解读。先说总判断：**这轮结果已经能形成一条清楚的报告主线：kNN 整体最强，Isolation Forest 最均衡，LOF 和 IK-density 有一定竞争力，Matrix Profile 是“特定场景有效”的 specialist，Rolling Z-Score 主要作为弱基线。**

结果文件主要在：

- [real_main_results.csv](/Users/Pang_chip/Documents/数据挖掘/results/real_main_results.csv)
- [real_main_manifest.csv](/Users/Pang_chip/Documents/数据挖掘/results/real_main_manifest.csv)
- [average_ranking.csv](/Users/Pang_chip/Documents/数据挖掘/figures/real_main_summary/average_ranking.csv)
- [characteristic_summary.csv](/Users/Pang_chip/Documents/数据挖掘/figures/real_characteristics/characteristic_summary.csv)
- [real_ablation_results.csv](/Users/Pang_chip/Documents/数据挖掘/results/real_ablation_results.csv)

**1. 这次真实实验到底做了什么**

我们从 TSB-UAD Public-v2 中筛选了 `60` 条可评估的 **univariate time series（单变量时间序列）**，来自 `26` 个 source datasets（来源数据集）。

每条序列都跑了 6 个 **unsupervised anomaly detection algorithms（无监督异常检测算法）**：

| Algorithm        | 中文理解                         | 算法类型                             |
| ---------------- | -------------------------------- | ------------------------------------ |
| Rolling Z-Score  | 滑动窗口标准分                   | statistical baseline（统计基线）     |
| kNN              | 最近邻异常检测                   | distance-based（基于距离）           |
| LOF              | Local Outlier Factor             | density-based（基于密度）            |
| Isolation Forest | 孤立森林                         | isolation/tree-based（基于隔离/树）  |
| IK-density       | Isolation Kernel family detector | isolation-kernel-based（隔离核家族） |
| Matrix Profile   | 矩阵轮廓                         | subsequence-based（基于子序列）      |

我们得到的是 `60 datasets × 6 algorithms = 360` 条主实验结果。

**2. 先理解评价指标**

异常检测里不能只看 accuracy（准确率），因为异常点通常很少。如果一个数据集只有 1% 异常，模型全预测正常也有 99% accuracy，但毫无意义。

这次主要看这些指标：

| Metric         | 中文解释                    | 怎么理解                                                 |
| -------------- | --------------------------- | -------------------------------------------------------- |
| AUROC          | ROC 曲线下面积              | 看模型能否把异常排在正常前面，但对类别不平衡不够敏感     |
| AUPRC          | Precision-Recall 曲线下面积 | 更适合异常很少的情况，越高越说明检测到的高分点真的是异常 |
| VUS-ROC approx | 时间序列版 ROC 近似         | 考虑异常区间附近的容忍度                                 |
| VUS-PR approx  | 时间序列版 PR 近似          | 本轮最重要指标，更适合时间序列异常片段                   |
| Runtime        | 运行时间                    | 看算法效率                                               |
| Average Rank   | 平均排名                    | 每个数据集内排序后再平均，更能看稳定性                   |

注意：目前的 `VUS-ROC/VUS-PR` 是 approximation（近似实现），最终报告前最好和 TSB-UAD 官方 VUS 实现核对。

**3. 主结果排名**

按 `VUS-PR approx` 看，平均结果如下：

| Algorithm        | Mean VUS-PR | Average Rank | Mean Runtime |
| ---------------- | ----------: | -----------: | -----------: |
| kNN              |      0.4885 |        2.375 |      0.4455s |
| LOF              |      0.4257 |        3.333 |      0.4691s |
| Isolation Forest |      0.4210 |        3.158 |      0.3012s |
| IK-density       |      0.4005 |        3.533 |      0.4266s |
| Matrix Profile   |      0.3788 |        3.933 |      0.4639s |
| Rolling Z-Score  |      0.2173 |        4.667 |      0.1528s |

这里有一个细节：LOF 的 mean VUS-PR 略高于 Isolation Forest，但 Isolation Forest 的 average rank 更好。意思是：LOF 在某些数据集上分数不错，但稳定性略差；Isolation Forest 更均衡。

每个数据集上谁拿第一的次数：

| Algorithm        | Best Count |
| ---------------- | ---------: |
| kNN              |         18 |
| Isolation Forest |         10 |
| IK-density       |          9 |
| Matrix Profile   |          9 |
| LOF              |          8 |
| Rolling Z-Score  |          6 |

谁最常垫底：

| Algorithm        | Worst Count |
| ---------------- | ----------: |
| Rolling Z-Score  |          20 |
| Matrix Profile   |          15 |
| Isolation Forest |           8 |
| LOF              |           7 |
| IK-density       |           7 |
| kNN              |           3 |

这说明 kNN 不只是平均高，而且很少彻底失败；Rolling Z-Score 最常失败；Matrix Profile 有时赢、有时输，波动较大。

**4. 每个算法到底在做什么，为什么会有这些结果**

**Rolling Z-Score（滑动标准分）**

它的逻辑很简单：看某个点和周围窗口的均值差多少个 standard deviations（标准差）。如果突然冒出一个极大值或极小值，它容易抓到。

它的问题也明显：如果异常不是单点幅值突变，而是一段形状变化、上下文异常或周期结构变化，它不一定能识别。所以它最快，但总体最弱。它在本轮 mean VUS-PR 只有 `0.2173`，主要作用是 baseline（基线），用来证明复杂算法确实有收益。

**kNN（k-nearest neighbors，最近邻）**

我们把时间序列切成 sliding windows（滑动窗口），每个窗口变成一个向量。kNN 的想法是：如果一个窗口和其他窗口都不像，它离 nearest neighbors（最近邻）很远，就可能是异常。

它本轮表现最好，mean VUS-PR 是 `0.4885`，best count 是 `18/60`。这说明很多 TSB-UAD 序列里的异常窗口确实在特征空间中“离正常窗口比较远”。

它的弱点是 runtime 和 window size sensitivity（窗口大小敏感性）。窗口太小可能看不到形态变化，窗口太大又可能稀释短异常信号。

**LOF（Local Outlier Factor，局部异常因子）**

LOF 也是基于邻居，但它不是只看“离邻居远不远”，而是看 local density（局部密度）是否比周围低。如果一个点在稀疏区域，它更可能是异常。

它本轮 mean VUS-PR 是 `0.4257`，和 Isolation Forest 接近，但 average rank 稍差。实验中多次出现 duplicate values warning（重复值警告），说明一些真实序列里重复值很多，这会影响 LOF 的局部密度估计。

所以 LOF 的解释可以写成：当正常模式形成清晰密度结构时，LOF 有效；当时间序列有大量重复值、平台段或密度变化本身很大时，LOF 容易不稳定。

**Isolation Forest（孤立森林）**

Isolation Forest 的想法是：异常点通常更容易被 random partition（随机划分）隔离出来。它不直接依赖距离或密度，而是用很多随机树来判断一个点是否容易孤立。

它本轮 mean VUS-PR 是 `0.4210`，average rank 是第二，runtime 只有 `0.3012s`，比 kNN/LOF 更快。它不是最高分，但非常均衡。

这在报告里很有价值：如果只看最高分，kNN 更好；如果看 accuracy-efficiency trade-off（准确性-效率权衡），Isolation Forest 是更 practical（实用）的选择。

**IK-density（Isolation Kernel family，隔离核家族方法）**

这是我们为了贴合课程背景加入的 Isolation Kernel family detector。它大致思想是：用随机采样构造 isolation-based representation（基于隔离的表示），再估计某个窗口是否处在稀疏或不典型区域。

它本轮 mean VUS-PR 是 `0.4005`，best count 是 `9`，说明它不是最强，但并不是无效。它在部分数据集上能赢，说明 isolation-kernel-style methods（隔离核风格方法）确实有潜力。

写报告时要注意：我们现在实现的是 lightweight IK-based density detector（轻量 IK 密度检测器），不要夸大成完整官方 IDK 实现。

**Matrix Profile（矩阵轮廓）**

Matrix Profile 是时间序列专用方法。它寻找 discord subsequence（不相似子序列）：如果某段子序列找不到相似片段，它可能是异常。

它的平均分不高，mean VUS-PR 是 `0.3788`，但 best count 有 `9`。这非常有解释价值：Matrix Profile 不是 general-purpose winner（通用赢家），而是 specialist（专门型方法）。

它适合 shape anomaly（形态异常）或 collective anomaly（群体/片段异常），但对单点异常、标签很细的异常或窗口设定不合适的数据，容易失败。

**5. RQ1：总体谁最好**

答案：**kNN 是当前第一版真实结果中的整体最佳算法。**

证据有三点：

- kNN 的 mean VUS-PR 最高：`0.4885`。
- kNN 的 average rank 最好：`2.375`。
- kNN 在 60 个数据集中拿第一最多：`18` 次，而且只垫底 `3` 次。

但不能写成“kNN 永远最好”。更准确的说法是：

> kNN achieved the best overall ranking in this benchmark, suggesting that many anomalies in the selected univariate time-series datasets can be detected as windows that are distant from their nearest normal subsequences.

中文意思：kNN 的整体排名最好，说明这些数据中的很多异常可以被理解为“和正常窗口距离较远的异常窗口”。

**6. RQ2：数据特征如何影响算法表现**

我们按 anomaly ratio（异常比例）、mean anomaly duration（平均异常持续长度）、seasonality（季节性）、noise level（噪声水平）做了分组。

**Anomaly Ratio（异常比例）**

低异常比例下，kNN 最强：

| Anomaly Ratio | 最强算法                    |          VUS-PR |
| ------------- | --------------------------- | --------------: |
| low           | kNN                         |          0.4358 |
| medium        | Isolation Forest / kNN 接近 | 0.5266 / 0.5260 |
| high          | Isolation Forest            |          0.5292 |

解释：低异常比例时，异常窗口更稀有，kNN 容易把它们识别为“远离正常窗口”的点。异常比例变高后，异常窗口不再那么孤立，Isolation Forest 反而更稳定。

**Anomaly Duration（异常持续时间）**

短/中等持续时间下，kNN 很强：

| Duration | kNN VUS-PR | 最强算法                  |
| -------- | ---------: | ------------------------- |
| low      |     0.5846 | kNN                       |
| medium   |     0.5825 | kNN                       |
| high     |     0.2982 | Isolation Forest 相对更好 |

解释：短到中等异常更像“局部不寻常窗口”，kNN 容易抓。很长的异常段可能形成另一个稳定模式，不再像孤立异常，所有算法都变难。

这点要谨慎写，因为 duration 和 source dataset 可能有 confounding（混杂因素），不能说绝对因果。

**Seasonality（季节性）**

kNN 在 low / medium seasonality 下更强；high seasonality 下 LOF、kNN、Isolation Forest、Matrix Profile 差距变小。

解释：强周期数据中，正常模式本身结构更清晰，密度型、隔离型、子序列型方法都有机会利用这种结构。但如果周期形态复杂，简单 Z-Score 仍然不够。

**Noise Level（噪声水平）**

中等噪声下 kNN、LOF、Matrix Profile 都表现较好；高噪声下 Isolation Forest 更稳。

解释：一点噪声可以让异常窗口和正常窗口差异更明显，但噪声太高会破坏距离和子序列匹配。Isolation Forest 不完全依赖精确距离，所以在高噪声下更稳。

**7. RQ3：准确性和效率是否有 trade-off**

有，但不是简单的“越慢越准”。

| Algorithm        | Mean VUS-PR | Mean Runtime |
| ---------------- | ----------: | -----------: |
| kNN              |      0.4885 |      0.4455s |
| Isolation Forest |      0.4210 |      0.3012s |
| LOF              |      0.4257 |      0.4691s |
| IK-density       |      0.4005 |      0.4266s |
| Matrix Profile   |      0.3788 |      0.4639s |
| Rolling Z-Score  |      0.2173 |      0.1528s |

Rolling Z-Score 最快，但效果最差。kNN 效果最好，但不是最快。Isolation Forest 的分数略低于 kNN/LOF，但速度更好，所以它是 accuracy-efficiency trade-off（准确性-效率权衡）最好的候选之一。

报告里可以这样写：

> kNN provides the strongest detection accuracy, while Isolation Forest offers a more balanced trade-off between accuracy and computational efficiency.

**8. RQ4：失败模式能否用算法机制解释**

可以，第一版结果已经有可解释线索。

kNN 的失败可能来自 high-dimensional window representation（高维窗口表示）。窗口太大时，每个窗口维度变高，距离会变得不稳定，这就是 curse of dimensionality（维度灾难）。

LOF 的失败可能来自 duplicate values（重复值）和 variable density（密度变化）。我们实验中确实看到多个 LOF duplicate warning，这支持“密度估计不稳定”的解释。

Isolation Forest 的失败可能出现在异常和正常窗口不容易被随机划分隔离时。它通常稳，但不一定拿最高分。

Matrix Profile 的失败多半和 window size（窗口大小）有关。如果窗口长度和真实异常片段长度不匹配，它找不到合适的 discord subsequence（不相似子序列），分数就会差。

Rolling Z-Score 的失败最容易解释：它只看幅值偏离，不能理解形状异常、上下文异常或长片段异常。

**9. 消融实验怎么看**

我们做了两类核心消融：normalization effect（标准化影响）和 window size sensitivity（窗口大小敏感性）。

**Normalization Effect（标准化影响）**

在 12 条真实序列子集上：

| Algorithm        | Raw VUS-PR | Z-normalized VUS-PR |
| ---------------- | ---------: | ------------------: |
| kNN              |     0.3992 |              0.4554 |
| Isolation Forest |     0.5105 |              0.5105 |
| LOF              |     0.3628 |              0.3615 |
| Matrix Profile   |     0.2958 |              0.2959 |
| IK-density       |     0.3692 |              0.3689 |

最明显的是 kNN：标准化后从 `0.3992` 提升到 `0.4554`。

解释：kNN 直接依赖 distance（距离），如果不同数据尺度差异很大，距离会被数值范围主导。z-normalization（标准化）能让距离更关注形状差异，而不是原始数值大小。

**Window Size Sensitivity（窗口大小敏感性）**

按 1%、default、5% 三种设置重新聚合后：

| Window Setting |   kNN | Isolation Forest |   LOF | Matrix Profile |
| -------------- | ----: | ---------------: | ----: | -------------: |
| small 1%       | 0.434 |            0.476 | 0.358 |          0.294 |
| default        | 0.455 |            0.511 | 0.361 |          0.296 |
| large 5%       | 0.325 |            0.356 | 0.371 |          0.274 |

default 通常较稳。large 5% 会让 kNN 和 Isolation Forest 明显下降，说明窗口过大可能把局部异常信号稀释掉。LOF 在 large 5% 下略升，可能因为更长窗口让局部密度结构更清楚，但这个结论还需要更多数据确认。

**10. 可以写进报告的核心故事线**

这轮结果最适合写成下面这条逻辑：

1. 我们不是只做 leaderboard（排行榜），而是比较不同 algorithm families（算法家族）在不同数据特征下的表现。
2. kNN 是整体最强方法，说明许多异常可被表示为“远离正常窗口”的模式。
3. Isolation Forest 虽不是最高分，但有最好的 practical trade-off（实用权衡）：效果较稳、速度较好。
4. LOF 与 kNN 相关但更依赖局部密度，因此在重复值或密度变化大的数据上更不稳定。
5. Matrix Profile 是 specialist，不适合作为总体最强模型，但在特定 shape/subsequence anomaly 上有价值。
6. Rolling Z-Score 是必要 baseline，证明简单幅值偏离不足以处理复杂时间序列异常。
7. normalization 和 window size 不是小细节，而是会实际改变结果的重要 experimental settings（实验设置）。

**11. 现在还不能过度下的结论**

不要写：

> kNN is always the best anomaly detector.

应该写：

> In the first real-data experiment, kNN achieved the best overall ranking across 60 selected univariate time series, but the best method still depends on anomaly characteristics and computational constraints.

不要写：

> Matrix Profile is bad.

应该写：

> Matrix Profile is not the strongest overall method in this benchmark, but its wins on several datasets suggest that subsequence-based detection remains useful for specific shape or contextual anomalies.

不要写：

> Window size 大一定更差.

应该写：

> Window size has a clear effect, and overly large windows may dilute local anomaly signals for distance-based and isolation-based methods in this subset.

**12. 下一步最该做什么**

最重要的下一步不是继续盲目跑更多算法，而是把这轮结果变成报告里的 insight（洞察）。

建议优先做三件事：

1. 用官方 TSB-UAD VUS 实现核对 `VUS-PR`，确保最终指标严谨。
2. 从 [real_cases](/Users/Pang_chip/Documents/数据挖掘/figures/real_cases) 里选 2-3 张最清楚的图，写 case study。
3. 把 RQ1-RQ4 各对应一张核心图，避免报告超过 10 页后变成堆表格。

当前这版结果已经足够支撑 preparation presentation，也足够作为 final report 的实验基础。接下来关键是“解释得漂亮”，不是“再跑一堆没分析的结果”。