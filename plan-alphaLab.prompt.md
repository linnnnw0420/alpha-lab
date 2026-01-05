# Plan: 让策略不再满仓换仓

## 目标
- 定位满仓换仓根因（权重生成与调仓执行链路）。
- 用最小增量加入分散、稳定、约束机制，降低换手与集中度。
- 用指标量化改动是否有效（换手、回撤、成本拖累等）。

## 目录分支结构（新增/修改）
```text
alpha_lab/
|-- backtest/
|   |-- engine.py (修改)
|   |   - 追踪权重链路并确认满仓换仓根因
|   |   - 接入 max_turnover / rebalance_threshold
|   |   - 加入 execution_delay_days 与交易成本
|   |   - 输出交易明细供 metrics 统计
|   `-- experiments.py (新增)
|       - 分段回测 / 走动窗口 / 参数扫描
|-- config/
|   `-- backtest.py (修改)
|       - 新增 max_turnover / rebalance_threshold / execution_delay_days / 成本参数
|-- data/
|   |-- metadata.py (新增, 推荐) 或 loader.py (修改, 备选)
|   |   - 提供 load_classification
|-- factors/
|   |-- momentum.py (修改)
|   |   - 多周期动量
|   `-- transform.py (修改)
|       - EWMA/rolling mean 平滑
|-- metrics/
|   |-- summary.py (修改)
|   |   - 换手 / 持仓数 / HHI / 成本拖累 / 回撤
|   |-- performance.py (修改)
|   |   - 与 summary 对齐输出口径
|   |-- factor_diagnostics.py (新增)
|   |   - compute_ic_series / compute_ic_decay / compute_quantile_returns
|   `-- rolling.py (新增)
|       - 滚动 Sharpe / 回撤
|-- portfolio/
|   |-- weighting.py (修改)
|   |   - hysteresis/no-trade band
|   |   - 约束投影接入
|   |-- rebalance.py (修改)
|   |   - apply_turnover_limit
|   |-- constraints.py (新增)
|   |   - 单票/行业/持仓数上限
|   `-- weighting_alt.py (新增)
|       - rank-weight / inverse-vol 等
data/
`-- raw/
    `-- sector_map.csv (新增)
        - ticker -> sector/industry
notebooks/
`-- 99_sanity_check.ipynb (修改)
    - 对比单周期 vs 多周期+平滑的换手/回撤
```

## 执行顺序建议
1. backtest/engine.py + portfolio/rebalance.py + config/backtest.py
2. portfolio/weighting.py + factors/* (稳定信号 + 换仓缓冲)
3. metrics/* + engine 明细输出
4. factor_diagnostics + experiments + rolling
5. data/raw + data/metadata (行业约束依赖)

## 前置确认
- 目标偏好：低回撤稳健 vs 高收益高换手
- universe 规模 + 调仓频率（决定 k_pct / buffer / max_turnover）
- 是否允许持有现金/空仓（决定是否加降杠杆风控）

## 详细子任务清单（新增/修改）
### alpha_lab/backtest/engine.py（修改）
- 追踪权重链路：记录目标权重与实际成交权重，确认是否“每次精确打到目标权重”导致满仓换仓。
- 接入换手控制：在“计算 turnover 之后、执行交易之前”应用 `rebalance_threshold` 与 `max_turnover`（低于阈值不调仓，超上限做部分调仓）。
- 加入延迟成交：支持 `execution_delay_days`（信号日与成交日错开）。
- 成本模型统一化：撮合处读取 `BacktestConfig` 的成本参数（比例费/固定费/最小费用等）。
- 输出交易明细：提供订单/成交日志供换手、成本、回撤统计。

### alpha_lab/portfolio/rebalance.py（修改）
- 实现 `apply_turnover_limit(current_weights, target_weights, max_turnover)`。
- 返回“被裁剪后的目标权重”，确保总权重与现金处理符合配置。

### alpha_lab/config/backtest.py（修改）
- 新增 `max_turnover`、`rebalance_threshold`、`execution_delay_days`。
- 新增成本参数：`commission_rate`、`fixed_commission`、`min_commission`（或现有命名体系的等价字段）。
- 可选：加入 `trade_at` / `mark_at` 以区分成交价与估值价口径。

### alpha_lab/portfolio/weighting.py（修改）
- 加入 hysteresis/no-trade band：仅当新候选超过阈值或旧持仓跌出 top-(k+buffer) 才换。
- 在权重生成后接入约束投影（单票/行业/持仓数限制）。

### alpha_lab/factors/momentum.py（修改）
- 扩展为多周期动量（例如短/中/长周期组合），参数化窗口长度与权重。

### alpha_lab/factors/transform.py（修改）
- 增加 EWMA 或 rolling mean 平滑函数，支持可配置平滑参数。

### alpha_lab/metrics/summary.py（修改）
- 输出换手、持仓数、集中度 HHI、成本拖累、最大回撤等指标。

### alpha_lab/metrics/performance.py（修改）
- 与 summary 指标口径一致，支持新增统计项与报告输出。

### alpha_lab/metrics/factor_diagnostics.py（新增）
- `compute_ic_series`：支持 Spearman/Pearson。
- `compute_ic_decay`：IC 衰减曲线。
- `compute_quantile_returns`：分层收益（等权/市值权可选）。

### alpha_lab/metrics/rolling.py（新增）
- 计算滚动 Sharpe/回撤等序列指标，供实验与可视化。

### alpha_lab/portfolio/constraints.py（新增）
- `enforce_max_holdings`、`enforce_single_name_cap`、`enforce_sector_caps`。
- 处理超限后的权重再分配策略（按比例缩放或现金留空）。

### alpha_lab/portfolio/weighting_alt.py（新增）
- 提供替代权重方法：rank-weight、inverse-vol 等，参数化可切换。

### alpha_lab/backtest/experiments.py（新增）
- `run_segmented_backtest`：分段回测。
- `run_walk_forward`：滚动窗口训练/测试。
- `run_param_sweep`：参数敏感性扫描与汇总。

### alpha_lab/data/metadata.py（新增）或 alpha_lab/data/loader.py（修改）
- 提供 `load_classification`，读取行业/单票元数据。

### data/raw/sector_map.csv（新增）
- 字段：`ticker,sector,industry`（或与你现有数据对齐）。

### notebooks/99_sanity_check.ipynb（修改）
- 对比单周期 vs 多周期+平滑的换手/回撤。
- 输出关键指标与曲线对照。


* **交易摩擦**：`commission_bps`、`slippage_bps` 都有，成本在回测里会扣（`backtest/engine.py`里 Step F）。
* **执行延迟**：`execution_delay_days` 会把权重按交易日历平移（信号日 -> 成交日），不是同一天“先知成交”。
* **换手控制**：`max_turnover` 和 `rebalance_threshold` 都实现了（低于阈值不动，超过上限做 partial rebalance）。
* **交易记录**：能输出 trades，positions 也有 daily matrix，后面做归因/诊断的地基算是有了。

这些已经覆盖了我之前提的“成本、延迟、换手”三大块，只是你**默认参数**基本还在“放飞自我”模式（下面说）。

---

## 你还缺的，是我之前提过但你框架里没真正落地的部分

### 1) 数据导入有一个很现实的坑：你的 CSV 列名对不上代码

`csv_source.py` 里写死要求列名必须有 **`date`**，但你 data/raw 里的文件第一列是 **`Date`**（大写）。
如果你是靠 loader 去读 CSV，这在严格意义上应该会直接报错。你能跑起来，多半是你绕过 loader 直接传了 prices，或本地文件列名不一样。

**该补的**：date 列名大小写兼容，或统一读入后强制 rename。

---

### 2) 你现在的 Universe 是静态的，基本等价于默认吃“幸存者偏差”

`config/universe.py` 的 `get_universe(..., as_of=...)` 现在完全不根据日期过滤，注释里也写了“future versions”。
这意味着：如果你的 200 个 ticker 是“现在还活着的”，你在历史上回测就是在作弊，只是你不知道你在作弊。

**该补的最小版本**：

* 每个日期的可交易列表（至少考虑 IPO/退市/停牌）
* 或者更简单：**价格缺失就不可交易**（但你数据现在几乎没缺失，所以还得改数据结构才能触发这个保护）

---

### 3) 价格对齐用的是无限 `ffill`，会把停牌/退市变成“永远不动的神票”

`data/loader.py` 里 `_align_to_calendar()` 默认 `forward_fill_limit=None`，也就是无限前向填充。
这会导致两个很典型的回测幻觉：

* 退市后价格被无限延续，组合像拿着一张永不归零的彩票
* 长期停牌被当成稳定资产，回撤被“抹平”

**该补的**：把 `forward_fill_limit` 设成一个默认安全值（比如 5 或 10 天），并且对“最后有效价格后”直接置 NaN（模拟退市/数据断档）。

---

### 4) 你缺少“因子是否真的有效”的研究输出（IC/分层/衰减）

你现在有 `momentum` 和 transform（winsorize/zscore），但没有一个模块告诉你：

* 因子对未来收益到底有没有预测力（IC / RankIC）
* 分层收益是不是单调（Top 组是否稳定赢 Bottom）
* 有效期多长（衰减曲线决定你该周调还是月调）

你现在讨论“买在顶点/错过浪潮”，本质上就是因为**没有这些诊断**，只能靠感觉。

**该补的**：一个 `research/diagnostics.py`（或 metrics 子模块），输出 IC、分层、衰减和简单统计。

---

### 5) 你缺“稳健性框架”：walk-forward / 参数敏感性 / 分段回测

你 config 里 `ml.py` 甚至已经写了 split/walk_forward 的结构，但项目里没有把它接到策略评估上。
没有这块，你做的所有“优化”都很可能是在拟合噪声。

**该补的最小版本**：

* 按年份拆分报告（年度收益、回撤、换手）
* 简单 walk-forward：训练窗口选参数，下一段验证
* 参数敏感性表：lookback、rebalance_freq、topK 改了以后曲线是否崩

---

### 6) 风控与约束还停留在“可以写，但没写”

你现在的权重生成（top-k 等权/按分数比例）能用，但缺少实际的组合约束：

* 单票权重上限（max weight）
* 行业中性/暴露控制（你 config 里写了 neutralize_industry，但没有行业数据也没实现）
* 波动率目标/风险平价（让组合风险更可控）

**该补的最小版本**：先做“单票上限 + 持仓数固定 + 波动率倒数加权”就够你学很多。

---

### 7) Benchmark 字段存在但没真正用起来

`BacktestConfig` 有 `benchmark`，但回测结果里没有加载 benchmark、算 alpha/beta、information ratio 这些“你到底赚的是什么”的指标。

**该补的**：最简版 benchmark 对比 + alpha/beta 分解。

---

### 8) 没有 tests（这会让你未来非常痛苦）

现在 tests 文件夹是空的。框架越长，越需要用最小单测锁住关键逻辑：

* 权重 shift 是否正确
* turnover 计算是否正确
* 成本扣减是否正确
* rebalance schedule 是否符合预期

---

## 你的数据：200个 ticker 近十年 “能不能用”？

### 用来入门：可以

200 个 ticker、10 年数据，足够你把整条链路跑明白：数据 -> 因子 -> 权重 -> 调仓 -> 成本 -> 绩效。学习目标完全够用。

### 用来得出“策略很强”的结论：不太够，甚至很危险

我看你 zip 里的 `close.csv`：

* **222 个 ticker**
* **3457 行交易日**
* 时间从 **2009-09-29 到 2025-12-16**
* **几乎没有缺失值**

“没有缺失值”这件事在真实股票数据里非常不自然，通常意味着：

* 你选的是“完整存活样本”（幸存者偏差）
* 或者你/数据源做了大规模填充（回测会变得更乐观）

另外，你现在只有 close/adj_close 这类价格文件，看不到 volume，就很难做流动性约束。

**结论**：这份数据适合作为“教学样本/框架验证”，不适合作为“策略有效性证明”。

---

## 你下一步最该改的 5 件事（按收益/成本比排序）

1. **修 CSV 的 Date/date 兼容**（不然你框架表面能跑，实际上数据入口很脆）
2. **把 forward_fill_limit 设默认值 + 退市后置 NaN**（让数据更像现实）
3. **把默认 backtest 参数变现实**：`execution_delay_days=1`，`max_turnover=0.3`，`rebalance_threshold>0`，slippage 提高一点
4. **加因子诊断模块**：IC + 分层 + 衰减（你讨论策略优劣就会从“感觉”升级成“证据”）
5. **加稳健性评估**：年度拆分 + 简单 walk-forward + 参数敏感性表

做完这五个，你的框架就从“能跑的玩具”升级到“至少不会自己骗自己”的研究平台。
