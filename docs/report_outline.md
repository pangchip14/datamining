# Final Report Outline

正文不超过 10 页，建议按以下版面压缩：

1. Front page：标题、组员、abstract，不计入 10 页。
2. Introduction 与 Aim：约 1 页，突出不是 leaderboard reproduction，而是 mechanism-oriented explanatory study。
3. Related Work：约 1 页，说明 TSB-UAD / TimeEval / VUS / Isolation Kernel family 与本研究的关系和区别。
4. Methodology：约 1.5 页，描述 preprocessing、window embedding、score alignment、algorithms。
5. Experimental Settings：约 1 页，描述 sampling rule、metrics、参数和硬件环境。
6. Results for RQ1-RQ3：约 2 页，放 overall ranking、metric correlation、runtime trade-off、dataset characteristic grouping。
7. Ablation and Failure Analysis for RQ4：约 2 页，放 window size、normalization、synthetic mechanism test、case studies。
8. Discussion 与 Conclusion：约 1.5 页，总结 insight、limitations、lessons learned。
9. References：不计入 10 页。

## Figure Priority

- Overall performance ranking table。
- Metric ranking correlation heatmap。
- Runtime vs VUS-PR scatter。
- Performance by anomaly ratio / duration / length。
- Window size sensitivity plot。
- Normalization effect plot。
- 2-3 张 case study overlay。

## Related Work Gap

报告中必须明确说明：TSB-UAD / TimeEval 的价值在于 benchmark infrastructure（基准评测基础设施）和大规模数据，而本研究的差异是 mechanism-oriented explanation（机制导向解释）。主线不是覆盖最多算法，而是回答 data characteristics（数据特征）如何影响不同 algorithm families（算法家族）的表现。
