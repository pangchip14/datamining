# 反馈修正记录

日期：2026-04-28

这份记录对应收到的反馈文件 `/Users/Pang_chip/Desktop/反馈.md`，用于说明哪些质疑已经修正，哪些改为报告中的明确限制。

## 已修正

1. Label-aware cropping

   旧版 `crop_record` 会根据 anomaly label 把长序列裁剪窗口居中到异常区间。新版默认改为 `middle` crop，只根据序列长度取中心窗口，不读取标签。`label_centered` 仅保留为显式 legacy/sanity-check 选项，不再是主实验默认。

2. Window-size ablation 汇总

   旧版把每条序列的实际窗口长度作为 `ablation_value`，例如 22、40、113、600，跨序列聚合后不易解释。新版改为三档标签：

   - `small_1pct`
   - `default_2pct`
   - `large_5pct`

3. 结果版本统一

   真实主实验、真实消融、真实调参敏感性、合成消融，以及对应图表和报告，已经用 2026-04-28 版本重新生成。报告不再沿用 4-23 或 label-centered crop 旧数值。

4. VUS 表述收紧

   `src/tsad_benchmark/vus.py` 的向量化实现已经与 TSB-UAD / VUS 公开代码中的 `generate_curve` -> `RangeAUC_volume_opt` 核心逻辑做逐样例交叉核对。报告中统一表述为 range-aware VUS-PR，并说明经过官方核心逻辑核对。

## 报告中明确说明

1. Sampling

   当前主实验是 source-balanced deterministic selection，不再声称严格 stratified sampling。数据特征分层用于结果分析，而不是用于抽样声明。

2. Neutral crop 的代价

   由于不再用标签决定裁剪窗口，部分长序列中性裁剪后只有单类标签，因此被跳过。新版真实主实验仍为 60 条可评估单变量序列，但 source 覆盖为 23 个，而不是旧版 26 个。

3. IDKD

   `ik_density` 使用 `ikpykit.anomaly.IDKD` 官方完整实现；在当前统一 sliding-window setting 下，IDKD 有课程和机制价值，但不是准确率或效率最优方法。

