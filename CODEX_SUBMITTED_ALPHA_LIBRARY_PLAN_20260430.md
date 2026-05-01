# Codex Plan: Submitted Alpha Reference Library + Self-Correlation Avoidance

You are implementing a guarded WorldQuant Alpha Factory refactor in this repo.

## Context

The factory currently produces many candidates with strong raw metrics (Sharpe/Fitness/Turnover), but submit/self-correlation clearance remains poor. A new read-only WorldQuant API discovery confirmed that the user's submitted/active alphas can be listed from:

`GET /users/self/alphas?order=-dateSubmitted&limit=...&offset=...`

Important: submitted alphas appear as records with `dateSubmitted != null`, often `status=ACTIVE`, `stage=OS`, not necessarily literal `status=SUBMITTED`. Endpoints/filters like `/users/self/submissions` and `status=SUBMITTED` may return empty.

The next refactor should build a local submitted-alpha reference library and integrate submitted-alpha structural similarity into scorer/generator/autosubmit safeguards, so new candidates avoid high similarity to already submitted alphas.

## Hard safety constraints

- Do NOT submit anything to WorldQuant.
- Do NOT PATCH/DELETE/PUT any WorldQuant remote resource.
- API access must be read-only GET only.
- Do NOT expose cookie/token/credential values.
- Do NOT loosen `AUTO_SUBMIT_MAX_P_SELF_CORR=0.20` or equivalent strict submit thresholds.
- Do NOT bypass D1 readiness, unsafe check gates, cooldown, daily submit caps, or existing autosubmit safety logic.
- Exploration/detail-check changes must remain label-collection only and must not force unsafe submit.
- Do NOT modify `secrets/`, `.venv/`, historical logs, historical reports, or `data/backtests.sqlite3`.
- New local generated artifacts are allowed only for this feature, e.g. `data/submitted_alpha_library.jsonl`, `state/submitted_alpha_features.json`, and `reports/submitted_alpha_library.md`.
- Use deterministic, testable code. Prefer small functions.
- If a workflow is currently active, do not start WorldQuant live runs; validation should be local compile/tests/previews/dry-run only.

## Implementation goals

### 1. Add read-only submitted alpha fetcher

Create `scripts/fetch_submitted_alpha_library.py`.

Behavior:
- Use existing auth/client style, preferably `alpha_factory.brain_client.BrainClient`.
- GET `/users/self/alphas?order=-dateSubmitted&limit=N&offset=K` with pagination.
- Keep only records where `dateSubmitted` is not null and expression exists (`regular.code`). Include status/stage broadly; do not rely on `status=SUBMITTED`.
- Respect rate limits and `Retry-After` where practical. Add a small delay between pages by default.
- CLI args: `--limit`, `--page-size`, `--output`, `--max-pages`, `--sleep-seconds`, maybe `--dry-run`.
- Output JSONL to `data/submitted_alpha_library.jsonl` by default.
- Record safe fields only: alpha_id, status, stage, date_submitted, settings subset, metrics subset, expression, fetched_at.
- Never print full expressions unless an explicit debug flag is used; normal output should summarize counts only.

### 2. Add submitted alpha feature extraction

Create `scripts/submitted_alpha_features.py`.

Behavior:
- Read `data/submitted_alpha_library.jsonl` by default.
- Extract structural features per alpha:
  - fields used in expression.
  - operator set/counts.
  - numeric window set.
  - normalized expression skeleton/fingerprints.
  - economic lineage tags: analyst_momentum, valuation, price_return, volume_liquidity, size, volatility, fundamental_quality, composite_factor, correlation, unknown.
  - settings summary.
- Output `state/submitted_alpha_features.json`.
- Write human-readable `reports/submitted_alpha_library.md` with aggregate counts, top fields/operators/lineages, but avoid dumping full expressions.
- CLI args for input/output/report.

### 3. Add similarity module

Create `scripts/submitted_similarity.py`.

Functions should be importable by other scripts and tested:
- `extract_expression_features(expression: str) -> dict`
- `load_submitted_features(path=...) -> list[dict]`
- `score_against_submitted(expression: str, settings: dict, submitted_features: list[dict]) -> dict`

Similarity should combine:
- field Jaccard.
- operator Jaccard.
- window overlap.
- lineage overlap.
- normalized/skeleton similarity.
- settings similarity.

Suggested weights:
- 0.30 field_jaccard
- 0.20 operator_jaccard
- 0.15 window_overlap
- 0.20 lineage_overlap
- 0.10 normalized/skeleton similarity
- 0.05 settings_similarity

Return fields:
- max_similarity
- mean_top5_similarity
- nearest_submitted_alpha
- field_jaccard
- operator_jaccard
- window_overlap
- lineage_overlap
- settings_similarity
- collision_level: high if >=0.80, medium if >=0.65, weak if >=0.45, low otherwise
- top_matches (small safe list: id, similarity, lineages; no full expression)

### 4. Integrate into `scripts/ml_candidate_scorer.py`

Add submitted-reference features to scored candidates where expression is available:
- submitted_similarity_max
- submitted_similarity_top5_mean
- submitted_nearest_alpha_id
- submitted_lineage_overlap
- submitted_collision_level
- submitted_similarity_penalty

Adjust self-corr prior conservatively:
- high collision (>=0.80): `p_self_corr_block = max(p_self_corr_block, 0.90)`
- medium collision (>=0.65): `p_self_corr_block = max(p_self_corr_block, 0.60)`
- weak collision (>=0.45): add small penalty, capped <=1

Penalize submit_score for submitted similarity. Do not reduce safety thresholds.

Make it robust if library/features are missing: warn/report and continue with neutral values.

Update scorer state/report to summarize submitted collision counts and top collision lineages.

### 5. Integrate into generation

Modify carefully and minimally:

#### `scripts/repair_candidates.py`
- Avoid shallow repair from parents with high submitted collision.
- Permit only deeper structural migration or deprioritize/retire high-collision parents.
- Keep existing functionality if submitted library is missing.

#### `scripts/d1_generator.py`
- Prefer low submitted-similarity candidates/fields for D1 exploration when feature data exists.
- Treat D1 as new-lineage exploration: favor distance from submitted lineages while maintaining raw-quality priors.

#### `scripts/superalpha_builder.py`
- Avoid pairing two parents that collide with the same nearest submitted alpha.
- Penalize pairs whose lineages both overlap heavily with the same submitted lineage.
- Favor cross-lineage combinations that lower submitted-similarity.

### 6. Integrate into `scripts/auto_submit.py`

Add a pre-detail-check submitted-reference guard:
- if submitted_similarity_max >= 0.85, skip detail check with reason like `submitted-reference high collision`.
- medium collision may still be eligible only for existing controlled exploration label collection if all existing cooldown/budget/quality gates allow it.
- Preserve strict final submit behavior and all existing gates.

### 7. Integrate into `scripts/self_corr_truth_table.py`

Record submitted-reference collision as a prior/evidence source, not as a real WQ self-correlation label:
- e.g. `submitted_reference_high_collision`, `submitted_reference_medium_collision`, `submitted_reference_low_collision`.
- This should help calibrate whether submitted similarity predicts actual future self-corr outcomes.

### 8. Tests

Add tests:
- `tests/test_submitted_similarity.py`
- optionally `tests/test_submitted_alpha_library.py`

Test at least:
- feature extraction finds fields/operators/windows.
- similar expressions score higher than unrelated expressions.
- high similarity maps to high collision level.
- scorer/autosubmit helper behavior does not crash when library missing.
- superalpha pair helper, if introduced, blocks same nearest submitted alpha pairs.

Update existing tests if necessary, but do not remove safety checks.

## Validation commands to run

Use project venv:

```bash
./.venv/bin/python -m py_compile \
  scripts/fetch_submitted_alpha_library.py \
  scripts/submitted_alpha_features.py \
  scripts/submitted_similarity.py \
  scripts/ml_candidate_scorer.py \
  scripts/repair_candidates.py \
  scripts/d1_generator.py \
  scripts/superalpha_builder.py \
  scripts/auto_submit.py \
  scripts/self_corr_truth_table.py

./.venv/bin/python -m unittest tests.test_self_corr_refactor tests.test_submitted_similarity

./.venv/bin/python scripts/fetch_submitted_alpha_library.py --limit 200 --page-size 50
./.venv/bin/python scripts/submitted_alpha_features.py
./.venv/bin/python scripts/ml_candidate_scorer.py --limit 80
./.venv/bin/python scripts/auto_submit.py --dry-run
```

If network/auth/API fails during fetch, do not fake success. Make code robust and report the failure.

## Final answer expected from Codex

Summarize:
- files changed/created.
- key behavior implemented.
- validation commands run and results.
- whether submitted library was fetched and how many submitted alphas were indexed.
- dry-run/autosubmit result summary.
- any blockers.
