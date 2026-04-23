# 对你方案的评估

整体来看，这是一份成熟度相当高的方案——结构完整、细节充分、对 scope 有主动控制，超出一般学生提案的平均水平。但也正因为它已经相对完整，一些问题就更值得提前处理，否则到了报告阶段再改代价会大。下面分成"可取之处"、"主要风险"、"具体改进建议"三部分谈。

## 一、可取之处

**Aim statement 的措辞与作业范例高度对齐。** 作业 PDF 给的正面范例之一就是 "The aim of this study is to evaluate the detection accuracy of different anomaly detectors in time series"，而反面范例中被批评的（比如"提出改进 IForest"那条）问题在于看起来像"提方法"而不是"做评测"。你的 aim 是评测+解释路线，没踩反面坑。加入 "explain" 这个动词让研究目标比单纯排名更有层次感，也给 analysis 部分留了发挥空间。

**研究问题分层合理。** RQ1（谁最好）→ RQ2（在什么数据上好）→ RQ3（trade-off）→ RQ4（失败模式），从 ranking 到 sensitivity 再到 failure analysis，正好对应作业对 A 档要求的"in-depth analysis"。这种分层比单纯列 3-4 个并列问题要好。

**方法流程写得足够具体。** 从 preprocessing 到 score alignment 到 evaluation 的 7 步流程清晰，window size 还给了公式 (w = \max(16, \min(128, \lfloor 0.02 n \rfloor)))，这种细节在 presentation 阶段老师一看就知道你"真的想过怎么跑"，而不是纸上谈兵。

**消融设计是这份方案最突出的亮点。** 三个必做消融（window size、normalization、metric choice）刚好打在时间序列异常检测这个子领域最容易出问题的点上，尤其 "Metric Choice Effect" 是很多学生方案里想不到的——能说明"换个指标排名就变"这件事本身就是有价值的发现。

**scope 控制主动而非被动。** 显式声明不做 deep learning、说明理由、列 assumptions、给两人组定位，这是很专业的做法。

## 二、主要风险

**风险一：与 TSB-UAD 原始论文的区分度需要更具体地讲清楚。** 这是我认为最需要警惕的一点。作业明确写了 "Your study must not be similar to the one which has already been done and reported in existing work"。TSB-UAD 本身就是"多算法在多数据集上的 benchmark"，VUS 指标也是 TSB-UAD 团队提的。你现在的区分论点是"我们做解释型而非排行榜型"，这个说法在 presentation 里可能被追问："那你的解释框架和 TSB-UAD 论文 Section 5/6 的分析有什么本质不同？"建议现在就准备好一个硬区分点，比如：(a) 明确选一个 TSB-UAD 没有重点分析的 dataset 维度（比如 anomaly duration 的精细分档）、(b) 明确你要检验的一个具体假设（比如"在 anomaly ratio < 1% 时 density-based 方法一定优于 distance-based"）、或 (c) 引入 TSB-UAD 原文没比较过的算法。

**风险二：算法列表里缺 IDK，这在本课程语境下是显著的策略问题。** 作业材料里专门列了 Isolation Kernel based methods 的 GitHub 链接（`https://github.com/IsolationKernel/`），"Potential projects" 部分也反复提到 IDK。课程/作者的学术背景和 IDK 直接相关。你的方案有 Isolation Forest 但没 IDK 或 IK-based detector。这不会让你不及格，但会让报告在"算法覆盖面"上看起来不够贴合课程重点。建议至少加入 IDK 作为第 7 个算法，放在 "kernel/isolation 家族"里和 Isolation Forest、One-Class SVM 形成对照。

**风险三：排除 deep learning 的理由需要再打磨。** 作业 PDF 明确说 "may include deep learning methods"——这是建议而非要求，所以你不做是合法的。但 A 档标准里有 "explored a wide range of different algorithms"。你现在给的理由是"时间成本+公平调参难"，这个理由成立，但更安全的表述是在报告里把它转成"主动取舍"：比如"我们的 aim 是解释型比较，需要每个算法的机制足够清晰；深度模型的 anomaly score 机制高度依赖架构和训练细节，不适合本研究的解释框架。"这样就把缺席从"做不完"包装成"不适合 aim"，专业度完全不同。

**风险四："many datasets" 的门槛要守住。** 作业两处都强调 "use many datasets"。20-40 条序列听起来不少，但 TSB-UAD Public-v2 是按 source dataset 组织的，如果这 20-40 条都来自同一两个 source，marker 会觉得你只用了 2 个 dataset。建议明确写成："从 TSB-UAD 的 X 个 source dataset 中各抽取 Y 条序列，共 N 条"，让"多样性"是 source-level 的而不是 series-level 的。

**风险五：Aim 里 "explain" 这个动词开了一个不好兑现的支票。** "解释原因"听起来很好，但在无监督异常检测里，把"算法 A 在数据 B 上差"归因到"因为 A 的机制是 C 而 B 的特征是 D"本质上是相关性不是因果。如果分析写得不够严谨会变成事后讲故事。建议 RQ4 的分析配合一个**小型 controlled synthetic experiment**：自己生成几条控制了 anomaly ratio / duration / shape 的合成序列，在上面跑所有算法，用合成实验支持你在真实数据上的解释。这样 "explain" 才有实证底气。

**风险六：presentation 和 report 的"工作分工与时间规划"完全缺失。** Presentation 要求里明确有 "Planning: how you are going to split the work & timelines" 这一条，你的方案里没有。两人组尤其要把这个写清楚，否则汇报会被问。

## 三、具体改进建议

按优先级排：

**优先级 1（影响等级判定的）**：把 IDK 加入主算法列表；为"与 TSB-UAD 的区别"写一段硬核区分论证（1-2 句话能打动 marker 的那种）；给 deep learning 缺席换一个基于 aim 的说辞。

**优先级 2（提升分析深度的）**：加一个小规模合成数据实验支撑 RQ4 的解释；明确数据集抽样规则中"source dataset 多样性"的保证；在 presentation 里加一页明确的分工+时间线。

**优先级 3（锦上添花的）**：metric choice ablation 可以做成一张 ranking correlation heatmap（Kendall's tau 或 Spearman），比单纯列表更有冲击力；case study 那几张图可以直接作为 report 的"insight"卖点，建议提前规划好要呈现哪 2-3 条失败/成功案例。

---

一句话总结：方案的骨架和逻辑没有问题，最大的风险不在实验设计而在**定位**——你需要在 presentation 的前两分钟就把"为什么这不是又一个 TSB-UAD"讲清楚。把这件事处理好，加上 IDK，这份方案冲 A 档是有基础的。
