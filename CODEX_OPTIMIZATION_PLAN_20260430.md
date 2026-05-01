# WorldQuant Alpha Factory — Optimization Plan Prompt

基于当前 workflow 状态（2026-04-30），目标：提升 high-quality、low-self-correlation alpha 的产出率。

## 背景约束（必须在所有任务中遵守）
- 不绕过任何 self-correlation gate 或 submit guard
- 不修改 AUTO_SUBMIT_MAX_P_SELF_CORR=0.20（正常提交上限）
- 不删除 data/ / logs/ / state/ / secrets/ / reports/
- 不因 raw Sharpe/Fitness/Turnover 强就提交
- 所有新增逻辑必须与现有 SQLite (data/backtests.sqlite3) 和 state/*.json 兼容
- 每个任务完成后输出变更摘要到对应 reports/*.md
- 如果当前有 `run_review_correct` / `alpha_factory` / active DB rows，先不要改代码；等 active run 完成后再执行。

---

## TASK 1：打破 Exploration 死锁（P0）
### 目标
当前 exploration 机制因 p_self_corr_block=1.000 的候选占据排队，导致每轮 2 个 exploration slots 实际从未被使用。需新增一个「强制多样化探索通道（Forced Diversity Exploration, FDE）」，与主 exploration gate 并行运行，专门为 lineage-orthogonal 候选分配 detail-check 配额。

### 修改范围
- `scripts/auto_submit.py`
- `scripts/ml_candidate_scorer.py`
- `state/auto_submit_state.json`（新增字段，兼容旧格式）

### 具体实现要求

#### 1.1 在 ml_candidate_scorer.py 中新增 FDE 标记逻辑
新增布尔字段 `fde_candidate`，满足以下全部条件时标记为 True：
- `lineage.theme NOT IN ['analyst_earnings', 'mixed']`
  - 即 under-sampled lineage：`fundamental_valuation / market_size / price_volume / liquidity_volatility`
- `pass_quality >= 1` 或等价 boolean true
- `repair_depth <= 1`
  - `repair_depth` 定义：该候选 parent_ids 中经历 `repairsc_` / `repairsc2_` / `repairsc3_` 的累计次数
- `p_self_corr_block` 可以是任意值（FDE 通道不受此约束限制）

输出追加到 scorer summary：
```yaml
fde_candidates: N
fde_by_lineage:
  fundamental_valuation: N
  market_size: N
  price_volume: N
  liquidity_volatility: N
```

#### 1.2 在 auto_submit.py 中新增 FDE 通道逻辑
新增环境变量读取（默认值）：
```bash
AUTO_SUBMIT_FDE_CHECKS_PER_ROUND=2
AUTO_SUBMIT_FDE_ENABLED=1
```

FDE 通道规则：
- 每轮最多触发 FDE_CHECKS_PER_ROUND 次 detail-check（与 exploration 预算独立计数）
- 候选必须标记 `fde_candidate=True`
- 按 `quality_score` 降序选取
- detail-check 结果正常写入 truth table 能解析的日志/状态来源（clear / blocked / pending）
- 即使 detail-check 返回 clear，仍需通过正常 submit gate 才能提交
  - FDE 只用于标签采集，不影响 submit 决策

在 `auto_submit_state.json` 中新增追踪字段：
```json
{
  "fde_checks_this_round": 0,
  "fde_total_checks": 0,
  "fde_clear_count": 0,
  "fde_blocked_count": 0
}
```

---

## TASK 2：修复 ML Scorer 的正样本缺失（P0）
### 目标
当前 `known_clear_rate=0.0`，scorer 退化为最悲观状态。需在不依赖历史 clear 标签的情况下，引入结构特征降低 prior 悲观偏差。

### 修改范围
- `scripts/ml_candidate_scorer.py`

### 具体实现要求

#### 2.1 新增 expression-level 结构特征
从 alpha expression 字段提取以下特征，追加到 scoring feature vector：

- `operator_entropy`
  - 计算 expression 中所有 operator token 的 Shannon entropy
  - operator 列表包含：`rank, zscore, ts_rank, decay, delta, corr, residual, neutralize, median, log, sign, abs, power, min, max, if_else`
  - entropy 越高 → 结构越多样 → prior clear 概率应更高

- `field_diversity_index`
  - 计算 expression 中使用的 data field 种类数量（去重）
  - 参考 `lineage.datasets_or_fields` 字段
  - `> 3` 种 field → `diversity_high`

- `uses_residual_or_neutralize`
  - 布尔值，expression 中是否包含 residual / neutralize 结构
  - True → 结构上更可能降低 market factor loading → prior clear 概率提升

#### 2.2 新增 under-sampled lineage prior 提升
对以下 lineage theme 的候选，在计算 `p_self_corr_block` 时，从当前 heuristic 基础上乘以 `lineage_prior_discount`：

```text
fundamental_valuation: 0.65
market_size: 0.70
liquidity_volatility: 0.75
price_volume: 0.80
analyst_earnings: 1.00
mixed: 1.00
```

`lineage_prior_discount` 应记录在每条候选 metadata 中。

#### 2.3 Scorer summary 新增输出字段
在 `reports/ml_candidate_scorer_report.md` 中新增：
- `avg_operator_entropy`
- `avg_field_diversity_index`
- `p_self_corr_block` 分布直方图：`0-0.2`, `0.2-0.4`, `0.4-0.6`, `0.6-0.8`, `0.8-1.0`
- `lineage_prior_discount` 应用次数

---

## TASK 3：生成侧 Lineage Quota + Repair Depth Retirement（P1）
### 目标
当前 repair_candidates.py 在生成侧没有 lineage 配额约束，导致 analyst_earnings 候选数量持续累积，同时 repairsc3_ 对同一父信号反复修复，边际收益递减。

### 修改范围
- `scripts/repair_candidates.py`
- `scripts/adaptive_sampler.py`
- `scripts/self_corr_truth_table.py`

### 具体实现要求

#### 3.1 在 self_corr_truth_table.py 中新增 repair_depth 字段
对每条 alpha 记录计算 `repair_depth`：

```text
repair_depth = 沿 parent_ids 链追溯，经过 repairsc_ / repairsc2_ / repairsc3_ 前缀的节点数量
```

例：原始 → repairsc_ → repairsc2_ → repairsc3_，repair_depth = 3。

写入 truth table 每条记录，并在 `reports/self_corr_truth_table.md` 新增：

```yaml
repair_depth_distribution:
  0: N
  1: N
  2: N
  3+: N
```

#### 3.2 在 repair_candidates.py 中实现 Retirement Gate
生成新 repair 候选前，检查父候选 repair_depth：

```text
if repair_depth >= 3 AND self_corr_status in ['blocked', 'predicted_blocked']:
  skip parent
  log: RETIRED [alpha_id] repair_depth=N reason=self_corr_blocked
```

Retirement 状态写入 truth table 可解析的状态来源；truth table 支持 `self_corr_status='retired'`。

#### 3.3 在 repair_candidates.py 中实现 Lineage Quota
每轮生成前，从 truth table 统计当前各 lineage theme 的候选占比：

```text
lineage_quota_max = 0.25
lineage share = current lineage candidate count / total candidate count
```

若 `analyst_earnings` 占比 > 0.25：
- 该轮 analyst_earnings repair 生成数量上限 = `max(1, normal_count * 0.3)`
- 释放的 budget 按 under-sampled lineage 当前 arm 得分比例分配

在 `reports/repair_candidates_latest.md` 新增 `lineage_quota_status` 摘要。

#### 3.4 在 adaptive_sampler.py 中引入 Diversity Bonus
在 arm reward 计算中新增：

```text
if arm.lineage_theme NOT IN ['analyst_earnings', 'mixed']:
  arm.reward += lineage_diversity_bonus
```

默认：

```bash
ADAPTIVE_LINEAGE_DIVERSITY_BONUS=3.0
```

---

## TASK 4：D1 专属 Truth Table 通道（P2）
### 目标
当前 `d1_ready=0`，D1 候选虽有 pass_quality，但无法通过 submit gate。需为 D1 类候选建立独立 relaxed exploration 通道，专门积累 D1 lineage 的 self-correlation 标签。

### 修改范围
- `scripts/auto_submit.py`
- `scripts/d1_generator.py`
- 新增 `state/d1_truth_table.json`

### 具体实现要求

#### 4.1 新增 D1 Relaxed Exploration Gate
新增环境变量：

```bash
AUTO_SUBMIT_D1_EXPLORATION_ENABLED=1
AUTO_SUBMIT_D1_MAX_P_SELF_CORR=0.35
AUTO_SUBMIT_D1_CHECKS_PER_ROUND=2
```

D1 通道规则：
- 候选前缀为 `d1_ / d1v2_ / d1v23_`
- `pass_quality >= 1`
- `p_self_corr_block <= 0.35`
- 每轮最多 2 次 detail-check
- 结果写入 `state/d1_truth_table.json` 和主 truth table 可解析来源
- 即使 detail-check clear，仍需 `p_self_corr_block <= 0.20` 才能正常 submit

#### 4.2 d1_truth_table.json 结构
```json
{
  "updated_at": "...",
  "total": 0,
  "clear": 0,
  "blocked": 0,
  "pending": 0,
  "clear_rate": 0.0,
  "entries": [
    {
      "alpha_id": "...",
      "family": "d1v23_",
      "p_self_corr_block": 0.0,
      "detail_check_result": "clear/blocked/pending",
      "checked_at": "..."
    }
  ]
}
```

#### 4.3 在 d1_generator.py 中反哺生成模板
读取 `d1_truth_table.json` 中 clear 的候选，提取 expression 结构特征：
- operator 组合
- field 组合
- neutralization 位置

作为下一轮 D1 generation 的 template bias，并在 `reports/d1_generator_latest.md` 记录 `template_bias_source`。

---

## TASK 5：SuperAlpha 跨 Lineage 合成约束（P2）
### 目标
当前 superalpha_builder.py 可能将同一 lineage 的多个父 alpha 合成，产生表面不同、实质相关的 super_ 候选，浪费 detail-check 预算。

### 修改范围
- `scripts/superalpha_builder.py`

### 具体实现要求

#### 5.1 新增父候选 Lineage Diversity 检查
合成 super_ 之前检查所有父候选的 lineage.theme：

```text
parent alpha 数量 >= 2 时，至少 2 个不同 lineage theme
违反时：跳过组合，记录 SKIP reason=same_lineage_parents
```

#### 5.2 新增 Lineage Pair Blacklist
对已经产生过 `self_corr_status=blocked` 的父候选 lineage pair 组合加入 blacklist：

```json
{
  "analyst_earnings+analyst_earnings": N_blocked
}
```

写入 `state/superalpha_state.json` 的 `lineage_pair_blacklist` 字段。

#### 5.3 Super_ 合成优先级重排
优先级从高到低：

1. 两个 under-sampled lineage（均不是 analyst_earnings/mixed）
2. 一个 under-sampled + 一个 analyst_earnings/mixed
3. 两个 analyst_earnings/mixed（最低优先级）

---

## 输出与验收标准

TASK 1（FDE）：
- 下一轮运行后 `fde_checks_this_round > 0`
- `state/auto_submit_state.json` 包含 `fde_*` 字段

TASK 2（Scorer）：
- scorer summary 包含 `p_self_corr_block` 分布直方图
- under-sampled lineage 候选的平均 `p_self_corr_block < 0.80`

TASK 3（Quota + Retirement）：
- truth table 中出现 `self_corr_status=retired` 记录（如果存在满足 retirement 的输入）
- repair_candidates report 包含 `lineage_quota_status`
- analyst_earnings 在新生成候选中占比 <= 35%（允许一个缓冲期）

TASK 4（D1 通道）：
- `state/d1_truth_table.json` 创建并写入至少 1 条记录（如果存在满足 D1 relaxed gate 的候选）
- auto_submit log 显示 D1 exploration detail-check 触发

TASK 5（SuperAlpha）：
- `superalpha_state.json` 包含 `lineage_pair_blacklist` 字段
- super_ 新生成候选中，单 lineage pair 的候选占比 <= 50%

---

## 执行顺序建议

1. TASK 2（Scorer 修复）
2. TASK 1（FDE）
3. TASK 3（Quota + Retirement）
4. TASK 4（D1 通道）
5. TASK 5（SuperAlpha）

---

## Codex 执行要求

1. 先输出一份执行计划。
2. 然后实现以上任务，优先保证兼容性和安全。
3. 不删除历史数据或敏感文件。
4. 不运行 live submit；如果需要验证 auto_submit，只做 dry-run 或单元测试/函数级测试。
5. 添加或更新轻量测试。
6. 运行：
   - `./.venv/bin/python -m py_compile` 覆盖修改脚本
   - 现有 unittest/新增 tests
   - scorer/truth-table/report generation 的非破坏性命令
7. 最终输出：
   - files changed
   - tests/commands run
   - blockers
   - expected effect
