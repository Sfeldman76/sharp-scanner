# Sharp Scanner — NCAAF V5.5 Heavy source-handoff repair (2026-10-24)

## Root cause of the user's 2026-10-24 HOLD

Heavy ran its legacy system pipeline and V5.5 retrospective CORE replay, but `ncaaf_heavy_brain_bridge.py` asked for **seven pre-exported files/tables** while no `NCAAF_BRAIN_HISTORY_INPUT_PREFIX` or all-seven `NCAAF_BRAIN_BQ_*_TABLE` settings were present. As a result, `missing_inputs` listed all seven even though historical games and V5.5 predictions were already produced in other stages of the same job.

The upstream V5.5 primary result was `COMPLETE_V55_RETROSPECTIVE_MARGINS__ATS_SOURCE_VINTAGE_HOLD` and cannot certify time-stamped historical sportsbook quotes or historically frozen CORE recipe selection.

## Changes in this complete repository ZIP

- `ncaaf_brain_source_recovery.py` **NEW**: recover one physical historical game row per season/game ID (home-side only) from `legacy_cache['games']`; recover the V5.5 `game_predictions.csv` from the exact immutable GCS replay prefix in `v55_primary['model_replay_artifacts']` after validating its recorded SHA-256 digest.
- `ncaaf_heavy_brain_bridge.py` **UPDATED**: only when no explicit seven-file GCS/BQ input sources are configured, reuse the two existing sources as **uncertified diagnostic views**, publish them under `recovered_uncertified/`, and enumerate the actual remaining source gaps in `status.json`.
- `research_validation_tests/tests/test_brain_source_recovery.py` **NEW**: eight tests for duplicate games, SHA mismatch, stale/model-selection provenance, explicit-source noninterference, and correct remaining HOLD.
- No changes to Production V1, NFL, dashboard, `utils.py`, V5.5 CORE training, or actual betting authority.

## What the next Heavy run should report

When the existing cache and V5.5 replay are readable, the combined-brain status will say:

```
status: HOLD
reason: HISTORICAL_BRAIN_EVENT_AND_COVERAGE_ARCHIVES_MISSING
source_recovery.recovered_inputs: ["core", "games"]
missing_inputs: ["signals", "coverage", "eligibility", "rule_evaluations", "universe_manifest"]
bet_authority: 0
```

This is **not** a promise that the cloud run will pass: if cache schema or replay bytes differ, the `recovery_failures` field will show why. Existing authentic configured GCS or BQ input tables are not mixed with the reconstructed views.

GCS output directory:

`gs://sharp-models/research/ncaaf/v55/combined_brains_shadow/runs/<run-id>/`

Read `status.json`, `input_receipts.json`, and `recovered_uncertified/games.csv` / `core.csv` (if present). The recovered CSVs are intentionally **NOT** independently certified full-universe or as-of betting records.

## Five remaining genuine evidence sources

1. `signals` — actual historical game/brain/system/direction events and original feature-as-of timestamps.
2. `coverage` — season/brain eligibility, feature counts, source SHA-256, and point-in-time certification.
3. `eligibility` — explicit status for each brain on each game: EVALUATED / INELIGIBLE / UNKNOWN.
4. `rule_evaluations` — an independent archived version per rule, prior-year training freeze, and per-game firing/non-firing denominator with source timestamps.
5. `universe_manifest` — season-level physical CFBD game count/identity hash from the raw CFBD archive independently of the research/model cache.

See `DATA_SCHEMA.md` for the exact contract. BigQuery schema discovery helper: `python audit_bigquery_brain_sources.py --project sharplogger --dataset sharp_data` in the authenticated research container. This only lists candidate source tables, not certifies them.

**Do not** use the old aggregate system-library results to construct fabricated per-game signals, invent snapshots on kickoff days, declare missing PT systems a no-trigger, or promote retrospective V5.5 recipe selection to independent holdout evidence. The existing historical score validation can continue without opening markets, but claims of ATS bet edge require verified pregame market quotes and independently available signals.

## Release validation

- 40 research validation tests pass locally (32 prior + 8 source recovery).
- Python files compile.
- Full repository retained, not a patch-only ZIP.
- Live Cloud Run / private BigQuery real-data backtest **not executed in this environment**.
