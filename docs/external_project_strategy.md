# External Project Strategy for Alpha Digging

This document distills implementation lessons from six external alpha-mining projects and maps them onto the current `alpha_digger` workflow.

The current workflow already includes strong first-generation defenses against high self-correlation:

- submitted-alpha reference fetching (`scripts/fetch_submitted_alpha_library.py`)
- submitted-expression similarity scoring (`scripts/submitted_similarity.py`, `scripts/submitted_alpha_features.py`)
- ML-style candidate scoring (`scripts/ml_candidate_scorer.py`)
- repair/self-correlation generation branches (`repairsc_`, `repairsc2_`, `repairsc3_`, `supersc_`)
- D1 and multi-dataset exploration
- strict autosubmit guards (`p_self_corr_block`, D1 readiness, unsafe checks, cooldowns, daily limits)

These mechanisms are necessary, but they are still mostly heuristic. The external projects point toward a stronger direction: move from text/metadata screening toward local semantic evaluation, canonical expression analysis, and better candidate triage before spending WorldQuant Brain resources.

## Executive Summary

| Project | Main strength | What to borrow | Current gap addressed |
| --- | --- | --- | --- |
| KunQuant | Fast vectorized/JIT factor evaluation | Local high-throughput pre-screening on sample panels | Avoid sending obviously weak/duplicative candidates to Brain |
| Alphalens | Factor diagnostics: IC, quantiles, turnover, stability | Better quality/risk diagnostics beyond Sharpe/Fitness/Turnover | Distinguish true signal from platform-metric artifacts |
| AlphaForge | AST/symbolic expression generation and mutation | Expression-tree mutation, canonicalization, structural novelty scoring | Replace superficial string mutations with semantic structure changes |
| WorldQuant Alpha101 code | Proven alpha motifs and expression templates | Motif library for generation and lineage labeling | Generate from validated structures instead of only local templates |
| worldquant-brain-simulator | Local approximation of Brain operators/backtests | Lightweight simulator and proxy self-correlation tests | Catch likely self-correlation collisions before online checks |
| worldquant-miner | Long-running queue, retry, rate-limit, mining ops | More durable task orchestration and adaptive throttling | Keep workflow productive through auth/rate-limit/backlog states |

## What Is Already Covered

`alpha_digger` already has several important capabilities inspired by these ideas:

1. **Submitted-alpha screening**
   - The submitted-alpha library gives the workflow local visibility into the user's own accepted/submitted expressions.
   - Similarity checks already block obvious near-duplicates before candidates enter the main task pool.

2. **Heuristic self-correlation prediction**
   - `ml_candidate_scorer.py` estimates self-correlation risk and D1 readiness before autosubmit.
   - Autosubmit remains conservative and blocks high-risk candidates.

3. **Structural repair families**
   - `repairsc_`, `repairsc2_`, `repairsc3_`, and `supersc_` are designed to preserve signal while changing expression structure.
   - These branches already use field/operator/window/neutralization changes to reduce obvious collisions.

4. **Family supervision**
   - Repeatedly unproductive or high-collision families can be downweighted/paused.
   - This prevents endless spending on crowded lineages.

5. **Strict submit safety**
   - Passing Sharpe/Fitness/Turnover is not treated as submit-clear.
   - Self-correlation, D1 readiness, unsafe checks, cooldowns, and daily limits remain hard gates.

## Remaining Gaps

The current filters are useful but incomplete. The main gaps are:

1. **Similarity is still mostly feature/text/proxy based**
   - Two expressions can look different in text but be mathematically equivalent or highly correlated.
   - Example: `divide(x, y)` and `multiply(x, inverse(y))` may evade shallow string similarity while producing the same signal.

2. **No robust local semantic execution layer yet**
   - Current screening does not fully evaluate candidate time series locally before Brain simulation.
   - This means high-correlation candidates may still reach remote checks and trigger cooldowns.

3. **Generation is not yet fully AST/DAG-native**
   - Many mutations remain template/string driven.
   - Deeper tree-level transformations would produce more meaningful structural novelty.

4. **Quality diagnostics are platform-metric heavy**
   - Sharpe/Fitness/Turnover are necessary but not enough.
   - The workflow would benefit from local IC decay, quantile monotonicity, turnover stability, and regime diagnostics when data is available.

5. **Queue orchestration can become blocked by platform state**
   - Auth expiry, rate limits, pending self-correlation checks, and detail-check budgets should be first-class scheduling inputs.

## Lessons by Project

### 1. KunQuant

**Useful idea:** treat factor evaluation as a high-throughput compilation problem.

Borrow:

- vectorized local expression evaluation
- expression normalization for cache reuse
- fast batch screening before remote backtests
- operator-level performance profiling

Apply to `alpha_digger`:

- Add a local `fast_eval` layer for a subset of supported Brain operators.
- Use synthetic or historical sample panels to reject unstable expressions early.
- Cache canonical expression outputs so repeated repair variants do not waste evaluation time.

Priority: **near-term**, after expression canonicalization exists.

### 2. Alphalens

**Useful idea:** factor quality should be measured with diagnostics, not just final score.

Borrow:

- information coefficient (IC) and rank IC
- IC decay by horizon
- quantile return monotonicity
- factor turnover and stability analysis
- group/sector neutral diagnostics

Apply to `alpha_digger`:

- Add optional local diagnostics for candidate families when enough panel data or returned Brain detail is available.
- Use diagnostics to separate candidates that are merely overfit to Brain metrics from candidates with robust signal shape.
- Feed diagnostic features into `ml_candidate_scorer.py` as additional ranking/risk inputs.

Priority: **near-term** for diagnostic reports; **later** for scorer integration.

### 3. AlphaForge

**Useful idea:** alpha expressions should be parsed and mutated as syntax trees, not strings.

Borrow:

- AST/DAG expression parser
- canonicalization rules
- subtree mutation and crossover
- expression depth/complexity controls
- structural diversity scoring

Apply to `alpha_digger`:

- Upgrade `alpha_factory/expr_parser.py` into a canonical expression representation that can round-trip Brain expressions.
- Add structural fingerprints: operator multiset, field lineage, window profile, neutralization placement, tree depth, and canonical hash.
- Use tree-edit or DAG-distance features inside `scripts/submitted_similarity.py`.
- Let `repair_candidates.py` and `superalpha_builder.py` perform controlled subtree rewrites instead of only surface-level replacements.

Priority: **immediate to near-term**. This is the cleanest way to improve self-correlation avoidance without loosening gates.

### 4. WorldQuant Alpha101 Code

**Useful idea:** classic alpha formulas are reusable mathematical motifs.

Borrow:

- motif library of proven expression shapes
- rank/correlation/delay/decay/topology templates
- motif labels for lineage tracking
- template diversity constraints

Apply to `alpha_digger`:

- Represent Alpha101-style formulas as abstract motifs with placeholder fields.
- Fill motifs with modern fields from discovered datasets.
- Track motif lineage so overused motifs can be downweighted before they cause crowding.
- Use motifs as D1/multi-dataset seeds, not as direct copies.

Priority: **immediate** for documentation/labels; **near-term** for generator integration.

### 5. worldquant-brain-simulator

**Useful idea:** approximate Brain locally before spending remote simulation/check budget.

Borrow:

- local implementation of Brain operators
- lightweight backtest proxy
- local self-correlation proxy against submitted library
- simulator-vs-Brain calibration

Apply to `alpha_digger`:

- Start with a small subset of operators used by top candidates.
- Evaluate candidates and submitted alphas on the same local sample panel.
- Compute Pearson/Spearman correlation between candidate outputs and submitted-library outputs.
- Use this as a pre-submit warning, not as a replacement for Brain checks.
- Calibrate proxy scores against known Brain outcomes before making hard decisions.

Priority: **immediate design**, **near-term prototype**, **later calibration**.

### 6. worldquant-miner

**Useful idea:** long-running alpha mining is an operations problem as much as a modeling problem.

Borrow:

- durable task queue semantics
- retry/backoff policies
- auth-aware scheduling
- rate-limit and cooldown awareness
- separation of generate / simulate / score / submit phases

Apply to `alpha_digger`:

- Treat pending self-correlation checks, detail-check budgets, and auth state as scheduler constraints.
- Prevent duplicate controllers and stale lock confusion.
- Prefer resumable state transitions over ad hoc long shell loops.
- Keep real submit as a narrowly gated final stage.

Priority: **ongoing**. This is already partly implemented, but can be made more explicit and durable.

## Prioritized Roadmap

### Immediate: Documentation and Low-Risk Scoring Improvements

1. Add structural-fingerprint fields to submitted-similarity reports:
   - operator family
   - field family
   - window profile
   - neutralization placement
   - expression depth
   - canonical text hash

2. Add Alpha101-style motif labels for generated candidates.

3. Extend reports to explain whether a candidate was rejected because of:
   - submitted-expression collision
   - crowded motif lineage
   - high predicted self-correlation
   - poor D1 readiness
   - repeated family-level failure

4. Keep all changes in dry-run/reporting mode first.

### Near-Term: AST-Based Similarity and Repair

1. Build a canonical expression tree layer on top of `alpha_factory/expr_parser.py`.

2. Add canonicalization rules such as:
   - normalize commutative operator argument order where safe
   - normalize equivalent operator forms where safe
   - standardize nested rank/zscore/scale wrappers
   - extract window constants and operator families

3. Upgrade submitted-similarity scoring with AST/DAG features.

4. Add controlled AST rewrites to `repair_candidates.py` and `superalpha_builder.py`:
   - subtree replacement
   - wrapper migration
   - neutralization relocation
   - field-family substitution
   - motif-preserving window changes

5. Run only local tests and dry-run generation first.

### Later: Local Simulator and Diagnostic Lab

1. Implement a small Brain-operator simulator for the most common operator subset.

2. Evaluate candidates and submitted-library expressions on a shared local panel.

3. Compute local correlation proxies and use them as an additional warning feature.

4. Add Alphalens-style diagnostics where suitable data exists:
   - IC / rank IC
   - IC decay
   - quantile spread
   - turnover stability
   - group exposure

5. Calibrate local proxy outputs against real Brain results before using them as hard filters.

## Safety Constraints

These constraints should remain non-negotiable:

- **No real submit from new screening or simulator code.** New modules must default to dry-run/reporting behavior.
- **No loosening of submit gates.** `AUTO_SUBMIT_MAX_P_SELF_CORR=0.20`, D1 readiness, unsafe checks, cooldowns, and daily limits remain intact unless the user explicitly approves a change.
- **Submitted-alpha library is read-only.** Fetching and analysis may read submitted expressions, but should not modify platform-side records.
- **No credential exposure.** Scripts and reports must never print cookies, passwords, API keys, or auth headers.
- **Local validation first.** Any new generator/filter should pass unit tests, compile checks, and dry-run reporting before remote backtests.
- **High raw quality is not submit clearance.** Sharpe/Fitness/Turnover passing candidates still require self-correlation and submitted-lineage checks.

## Recommended Next Implementation Step

The best next step is **AST-based expression canonicalization and similarity**.

Reason:

- It directly improves the current self-correlation bottleneck.
- It is lower risk than a full simulator because it does not require market data or remote execution.
- It improves both screening and generation.
- It can be unit-tested locally with known equivalent expression pairs.

Suggested first deliverable:

1. Add canonical expression fingerprints to submitted-similarity reports.
2. Add unit tests for equivalent-looking expressions.
3. Keep the first version report-only.
4. After validation, wire the score into pre-write filtering and family supervision.
