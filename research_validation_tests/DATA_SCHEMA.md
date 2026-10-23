# NCAAF combined historical brains — shadow validation V1

## Purpose

Validate full historical **STAT, Big Al, Pathi, System Miner, and PT** against
real game results before attaching them to **CORE**. The independent system
universe is **2013–2025**, not CORE's narrower retrospective forecast coverage.
Use 2026 games only in a **separate prospective/settled audit**; never use 2026
results to discover a system, select a pair, or alter historical thresholds.

This package is an external research harness. **It has not been merged into your
live repository, deployed to Cloud Run, or run on BigQuery.** It does not change
NFL, Production V1, the weekly score job, or authority in the live dashboard.

## Data contract — no invented historical evidence

Provide seven CSV or Parquet inputs using the exact column names below. Archive
and source evidence must come from the original stored records / a reproducible
replay of the existing individual brain rules. Do not fill missing inputs with
false, zero, simulated recommendations, or today's current expert ratings.

| Input | Required columns | Notes |
|---|---|---|
| `games` | `season,game_id,kickoff_utc,actual_home_margin` | One **physical game** per season/game ID, 2013–2025. Optional `home_spread`, `close_home_spread`. Home spread is signed from home team's perspective. |
| `core` | `season,game_id,model_home_margin,prediction_at_utc` | Existing V5.5 historical replay schema. May have fewer years; that must **not** trim the system universe. |
| `signals` | `season,game_id,brain,system_id,side,feature_asof_utc,evidence_family` | **One firing event per system/game/side.** Brains are `STAT`, `BIG_AL`, `PATHI`, `SYSTEM_MINER`, `PT`; side HOME or AWAY. Add `rule_fit_max_season` for prior-only proof. |
| `coverage` | `season,brain,status,eligible_games,feature_rows,source_hash,point_in_time_certified` | **Every brain × year 2013–2025**. `status` is COMPLETE, PARTIAL, or UNAVAILABLE. `source_hash` is the 64-character SHA-256 of the independently sourced snapshot. Source times and correctness must be auditable. |
| `eligibility` | `season,game_id,brain,evaluation_status,feature_asof_utc` | Every brain × canonical game. EVALUATED, INELIGIBLE, or UNKNOWN. `EVALUATED` requires pre-kickoff source time. This records **non-trigger evaluations**, not just fired systems. |
| `rule_evaluations` | `season,game_id,brain,system_id,evaluation_status,feature_asof_utc,rule_fit_max_season,rule_version_sha256` | Every rule × brain-eligible game, including a record when a rule did **not** fire. `evaluation_status` is EVALUATED, INELIGIBLE, or UNKNOWN. This gives the **individual system**, not just brain, a verified denominator. `rule_version_sha256` is a 64-character hash of the rule recipe as frozen for that season. |
| `universe_manifest` | `season,expected_games,game_ids_sha256,source,source_snapshot_utc` | One row per year, independently computed from CFBD raw canonical games, **not** from the model's candidate table. SHA-256 of `\n`-joined lexicographically sorted string game IDs. |

Additional `games` price evidence required for certified ATS comparison:
`quote_at_utc`, `quote_source`, `quote_snapshot_id`, `quote_vintage_certified`.
These must describe genuine archival original pregame quotes. The old 2026-10-23
export marked original quote times **unavailable**; such rows remain exploratory.
A missing opener may be graded with `close_home_spread`, but this is
`CLOSE_FALLBACK_DESCRIPTIVE_ONLY` and never a certified pregame CORE edge.
No return-on-investment claim is calculated without genuine odds and bet time.

### Evidence independence

Each signal has an `evidence_family` assigned to the *underlying evidence*, e.g.
`TEAM_FORM`, `SCHEDULE`, `MARKET`, `EXPERT_PICKS`, `PT_CONSENSUS`.
Two Miner rules using TEAM_FORM and an overlapping STAT signal using TEAM_FORM
count as **one** family, not multiple independent approvals. Families must be
grounded in the actual input features: don't assign unique families just to
increase votes.

### Coverage and no-fire records

A complete year-level `coverage` row does not establish that a brain evaluated
a particular game. `eligibility` records are therefore required to certify
non-trigger states. In addition, `rule_evaluations` must cover each rule on
every brain-eligible game before the **per-rule** non-fire denominator can be
certified. The script explicitly outputs `MISSING_OR_PARTIAL` otherwise.
Do not treat an absent manual pick history as a losing or nonfiring signal.

## Period design

* **2013–2021:** Long-horizon historical discovery pool; keep every original
  season and historical feature even when CORE predictions are absent.
* **2022–2023:** Chronologically later system/pair selection. Multiple-testing
  correction (Holm) reported for systems and pairs.
* **2024–2025:** Independent historical confirmation. Pair definitions must
  have been selected without these outcomes. Verify source snapshots and that
  `rule_fit_max_season < game season`.
* **2026:** Current-season prospective outcomes isolated from research selection.
  Not supplied to this historical research CLI at all.

These are **chronologically partitioned research tests**, not claims that
2024–2025 were true contemporaneous prospective wagers. Since the V5.5 recipe
and some brain rules were reconstructed later, any missing historical source
vintage or rule timing blocks independent wagering authority.

## Run (research machine / Cloud Run, NOT Streamlit startup)

```bash
python -m pip install pandas numpy scipy pyarrow
python ncaaf_brain_full_history.py \
  --games exports/games.parquet \
  --core exports/core_replay.csv \
  --signals exports/brain_events.parquet \
  --coverage exports/brain_year_coverage.csv \
  --eligibility exports/brain_game_evaluations.parquet \
  --universe-manifest exports/cfbd_canonical_game_manifest.csv \
  --rule-evaluations exports/per_rule_game_replay.parquet \
  --out /tmp/ncaaf_brain_full_history
```

Without eligibility, independent universe manifest, or rule evaluations the
research is still runnable for diagnostics, **but certification stays false**.

Source discovery in your **authorized GCP runtime**:

```bash
python audit_bigquery_brain_sources.py --project sharplogger --dataset sharp_data --out /tmp/source_schema_audit.json
```

Audit script only enumerates existing BigQuery column metadata. It **cannot**
prove timestamps or generate historical picks without actual underlying source
and rule code. Review identified tables, export the six datasets, and keep the
CFBD source ID mapping consistent with V55.

## Output

* `status.json` — counts, coverage holds, artifact SHA-256 hashes, authority=0.
* `universe_audit.csv` — independent canonical game-ID completeness by season.
* `coverage_matrix.csv` — 65 brain-year coverage cells (13 × 5).
* `coverage_by_brain.csv` — each brain's historical-years-covered status.
* `brain_game_eligibility.csv` — proof a silent brain was actually evaluated.
* `rule_evaluations.csv`, `rule_year_audit.csv` — per-rule nonfiring denominators,
  completeness and point-in-time evaluation proof.
* `signals_ledger.csv` — pregame-vintage checks for every reconstructed event.
* `systems_by_period.csv`, `systems_by_year.csv`, `holdout_review.csv` — all-history
  system records, chronological phases, and confirmation diagnostics.
* `core_matched.csv`, `combined_cohorts.csv`, `combined_splits.csv` — CORE-conditioned
  support, opposition, vetoes, by season/edge/role.
* `brain_pairs.csv` — evidence-family pairs found in 2022–2023, with 2024–2025
  confirmation views; unselected pairs are **not** valid held-out discoveries.

Every output has `authority=0` or top-level `bet_authority=0`. No hidden
promotion of live bet thresholds occurs. Bet sizes, bankroll, and odds are not
used to choose systems. Showing a win rate does not establish profitability.

## Integration point

After placing this package in the repo, call it in the existing
`MARKET=ncaaf_v55_heavy` **research-only** route after the individual brain
historical replays finish and after the frozen rolling CORE replay is available:

```python
from ncaaf_brain_full_history import report, write_output, RunConfig

research = report(
    games=historical_game_frame,      # physical CFBD games 2013-2025
    signals=historical_brain_events,  # complete source-time-preserving events
    coverage=brain_year_manifest,     # every brain x season
    core=v55_rolling_predictions,     # V5.5 replay subset
    cfg=RunConfig(),
    eligibility=brain_game_evaluations,
    universe_manifest=cfbd_independent_game_manifest,
    rule_evaluations=individual_rule_game_evaluations,
)
write_output(research, '/tmp/combined_brain_research')
# Separately, publish under a versioned immutable research GCS prefix.
# DO NOT call live action selection, Production V1, or overwrite weekly bets.
```

The seven frame variables above are explicit integration inputs, **not** current
symbols claimed to exist in your repository. Wiring them requires the up-to-date
source ZIP and actual BigQuery table schemas. The original heavy log shows
`verified_historical_brain_events=0`, so this cannot honestly be marked as a
completed real-data backtest yet.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests use **synthetic fixture records** solely to verify chronology, 13-season
coverage, CORE subset separation, duplicate rejection, unknown eligibility,
quote timing, source vintage, vetoes and zero betting authority.
