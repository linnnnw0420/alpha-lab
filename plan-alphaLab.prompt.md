## Plan: 让策略不再满仓换仓

把“满仓买入/满仓卖出”的根因定位到权重与调仓执行链路，然后用最小增量加入三类机制：分散（更多持仓/连续权重）、稳定（信号平滑+换仓缓冲）、约束（turnover 上限/部分调仓/交易成本），让策略从“押注单一篮子”变成“可控风险的组合轮动”，并用指标把改动是否有效量化出来。

### Steps 5
1. 定位满仓换仓来源：检查权重生成与调仓执行在 `alpha_lab/portfolio/weighting.py` → `alpha_lab/portfolio/rebalance.py` → `alpha_lab/backtest/engine.py` 的数据流，确认是否是 `top_k_*`/单一标的导致权重稀疏（0/1）以及引擎是否“每次精确打到目标权重”。
2. 加入“部分调仓/turnover 上限”：在 `alpha_lab/portfolio/rebalance.py` 实现 `apply_turnover_limit`（或等价函数），新增配置项（放在 `alpha_lab/config/backtest.py` 或新增轻量 portfolio config），让每次从当前权重向目标权重移动不超过 `max_turnover`。
3. 加入“换仓缓冲（hysteresis/no-trade band）”：在权重生成前后增加“保留旧持仓”的规则（例如：持仓跌出 top-(k+buffer) 才卖，或要求新候选分数超过阈值才换），落点建议在 `alpha_lab/portfolio/weighting.py` 或作为权重后处理函数。
4. 做“更稳定的信号”：扩展因子为多周期动量 + 时间平滑（EWMA/rolling mean），改动集中在 `alpha_lab/factors/momentum.py` 与 `alpha_lab/factors/transform.py`，并在 `notebooks/99_sanity_check.ipynb` 对比单周期 vs 多周期+平滑在换手与回撤上的变化。
5. 把“是否更稳妥”量化：补齐/扩展指标与记录（换手、持仓数、集中度HHI、成本拖累、净值回撤），落点在 `alpha_lab/metrics/summary.py` / `alpha_lab/metrics/performance.py`，并确保引擎输出交易明细可用于成本与换手统计（见 `alpha_lab/backtest/engine.py`）。

### Further Considerations 3
1. 你的目标更偏向：A 低回撤稳健（优先 turnover 上限+风控）还是 B 更高收益（允许更高换手但要成本模型）？
2. 当前 universe 规模（标的数）与调仓频率（周/月）是多少？这决定 `k_pct`、`buffer`、`max_turnover` 的合理区间。
3. 是否允许持有现金/空仓：若允许，可加“回撤/波动率阈值降杠杆”作为最小风控覆盖。
