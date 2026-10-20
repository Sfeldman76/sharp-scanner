# NCAAF V5.3 — Single-run complete V5 research + original brains

**Implementation status:** full orchestrator built, 15 local regression tests passed (including a simulated complete Cloud Run workflow and simulated failed stage that publishes its detailed audit). **The real GCP/Cloud Run full job has NOT been executed from this environment**. Results, original feature-vintage certification, and live V5 bet authority remain unknown pending the actual run. Do not mistake a Cloud Run exit-code 0 for a promoted betting model.

## Scope and source of truth

Rather than guessing BigQuery mappings from schema names, `ncaaf_v53_full_research.py` calls the **original** NCAAF code in the deployed repository. It audits `ncaaf_research_v2.py` and `utils.py` constants, the dashboard/Utils build ID, and actual BigQuery and GCS metadata. A mismatched original path or mixed dashboard build fails visibly before research starts.

### Real legacy files/functions and storage source mapping

| Family | Original module/function | Exact legacy storage | In V5.3 |
|---|---|---|---|
| Big Al observations | `ncaaf_research_v2._load_big_al_observation_ledger` | `gs://sharp-models/research/expert_observations/big_al/observations.jsonl` | Full legacy Miner hypotheses and observation count; grades only with proper original evidence |
| Big Al systems | `sharp_line_dashboard.fit_historical_ncaaf_core_expert` → `_hc_build_system_history_stats` → `_V143_SYSTEM_HISTORY_CACHE`; `ncaaf_research_v2._attach_exact_expert_flags_to_miner` | `sharplogger.sharp_data.scores_with_features`, then fallback `sharp_scores_full`; historical core view | Reuses directional occurrence records, matches V4 game IDs and side exactly using the original canonical team-alias resolver, audits mismatches |
| Pathi systems | `utils.fetch_pathi_bigal_history_backend`, `build_pathi_bigal_team_game_state`, `add_pathi_bigal_rule_flags`; dashboard occurrence cache | Same existing scored historical view/table, plus `ncaaf_historical_game_side_context` | Reuses directional occurrences; no new substitute rules |
| Prediction Tracker | `ncaaf_research_v2.refresh_prediction_tracker_external` and `_pt_load_season`/`_pt_load_live_current` | `research/ncaaf/external/prediction_tracker/raw`, `/normalized`, `/snapshots`, `feeder_manifest.json`, `header_manifest.json`, `current.csv` | Original identity/alias resolution; retains sparse components. Missing Pigskin Index is **not a new V5 gate** |
| System Miner | `ncaaf_research_v2.run_system_miner_v3`, `_extended_atoms`, confirmed mechanism registry and existing incumbent library | `gs://sharp-models/research/ncaaf/v2/current_report.json`, `current_bundle.pkl`; existing historical feature sources | Replay actual representative-condition masks on older Miner game frame, map selected side to V4 orientation |
| State/bounce-back | `_attach_historical_state_context_to_miner`, role-transition mechanism flags | `ncaaf_historical_game_side_context`, `ncaaf_cfbd_miner_features_v1`, historical caches | Incorporated from original Miner; do not misclassify missing state signals as opposition |
| CFBD football & travel | `ncaaf_core_challenger_v2._attach_cfbd_core_context`, `_attach_cfbd_context_to_miner`, `_attach_h2h_rivalry_travel_context` | Existing `ncaaf_cfbd_archive_research_v1`, `ncaaf_game_travel_reference`, associated source views | Full V4/V4.2 refit & legacy full research, 2026 excluded from discovery |

**Important qualification:** the dashboard's graded occurrence cache is a *retrospective historical reconstruction*, not an unchanged archived pregame bet ticket. V5.3 marks all such Pathi/Big Al observations `RETROSPECTIVE_GRADED_OCCURRENCE__NOT_PREGAME_ARCHIVE`. Confirmed Miner mechanisms applied to historical games are marked `CHRONOLOGICAL_MINER_RETROSPECTIVE__INSPECTED_CONFIRMATION`. Historic 2024–25 have been repeatedly examined. The V5.2 original-GCS-proof verifier runs separately and its count never silently grants votes into the V5.3 shadow score. During this V5 run, any automatic promotion call from the older Prediction Tracker controller is explicitly intercepted and logged `LEGACY_PT_AUTO_PROMOTION_HOLD`; original controller behavior is restored afterward.

## One dropdown, one complete V5 research run

The dashboard NCAAF Model Maintenance choices are:

- **NCAAF — Weekly Update (existing frozen Production V1)** — restored. This is the original existing weekly route `ncaaf_production_weekly`, *not* a V5 weekly inference/retrain. It settles existing V1 locks and scores the weekly slate; no automatic V5 policy promotion.
- **NCAAF V5 — FULL Research + All Legacy Brains (audited)** — new comprehensive route `ncaaf_v53_full_research`; no manual stage sequencing or BigQuery source-map handoff.
- **NCAAF Legacy — Heavy Challenger Search** — preserved, separate old research route for reproducibility.

The V5.2 route remains accessible to scripts using its exact older market name, but no longer clutters the operator dropdown.

### Full research scope, with log stages

One execution carries out ALL stages (no short diagnostic-only run):

`LEGACY_MODULE_CONTRACT` → `MATCHED_DASHBOARD_UTILS_BUILD` → `LEGACY_SOURCE_CONSTANTS` → `EXACT_LEGACY_SOURCE_MANIFEST` → `BUILD_HISTORICAL_EXPERT_CACHE` → `BUILD_STAT_CACHE` → `FULL_V4_FOOTBALL_MARGIN_RESEARCH` → `FULL_V42_ALL_FIVE_FEATURE_FAMILIES` → `LEGACY_FULL_CORE_SPECIALIST_RESEARCH` → `FULL_LEGACY_PATHI_BIGAL_PT_MINER_STATE_RESEARCH` → `FULL_V5_MARGIN_TO_MARKET_RESEARCH` → `FULL_V51_POINT_EDGE_ORIENTATION_REPLAY` → `EXISTING_NCAAF_CANONICAL_TEAM_ALIAS_RESOLVER` → `LEGACY_PATHI_BIGAL_OCCURRENCE_JOIN` → `MINER_CONFIRMED_MECHANISM_FIRING_TO_V4` → `MATCHED_LEGACY_CONDITIONAL_ATTRIBUTION` → `V5_SHADOW_AUTHORITY_POLICY` → `V52_VERIFIED_ARCHIVE_SOURCE_BRIDGE` → `V53_FINAL_AUTHORITY`.

Every phase is logged as `[NCAAF-V53-AUDIT]` with timestamp, `START`/`PASS`/`FAIL`, elapsed seconds, exact named sources and joining coverage, error and exception traceback on failure. The job publishes an **immutable run report and audit JSONL even when a fatal research stage fails** and then returns a failed execution rather than concealing the error.

### Cloud Run full-run command

Deploy the two updated entrypoints and add new research modules to the **training image** (`COPY . /app` or explicitly enumerate all scripts in your Dockerfile). Keep your actual original `utils.py`, `ncaaf_research_v2.py`, core challenger, `pt_promotion_v1.py`, CFBD loaders and helpers. The reference SHA file records the older files used for this audit; don't replace newer good production files merely to match a checksum.

```bash
gcloud run jobs execute sharp-train-job \
 --project=sharplogger --region=us-east4 \
 --update-env-vars=SPORT=NCAAF,MARKET=ncaaf_v53_full_research,HEADLESS=1,NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED=1 \
 --task-timeout=14400s --wait
```

The 4-hour timeout permits a genuine long full-history research run; it is **not** a 15-minute forced stop. Check your Cloud Run account quota and job resources; this flag does not bypass regional CPU/memory/duration limits.

**Report:** `gs://sharp-models/research/ncaaf/v53_full_research/latest_report.json`.

Every run also stores `runs/<run-id>/report.json`, `audit_events.jsonl`, `source_manifest.json` and (on successful completion) `legacy_occurrences.csv`, `system_attribution.csv`, `family_attribution.csv`, `v5_shadow_edge_candidates.csv`.

## Model & authority contract

- **CORE:** full V4 and V4.2 football-only research; V4 identity and model-game ATS orientation independently checked by V5.1.
- **STAT:** the actual legacy Big Al/Pathi/Miner/PT/State functions and rules, not newly invented heuristics.
- **MARKET:** verified historical opener comparison in *points*, not false 80–90% cover-probability betting confidence. True historical executable prices/ROI remain unavailable.
- **EDGE/authority:** family-collapsed confirmation/opposition/conflict in shadow scorecards; `CORE_EDGE_ONLY`, `ONE_FAMILY_RETROSPECTIVE`, `MULTIFAMILY_RETROSPECTIVE`, `CONFLICT_REVIEW`. These are hypothesis labels, **not bets**.
- **Gates:** no live V4/A production inference pipeline or prospective bookmaker price-lock contract is certified yet; archived pregame evidence has not been established for historical systems; original source-vintage hold remains. Therefore `bet_authority=0`, every row has `V5_Bet_Decision=NO_BET`, and no Production V1 or current live betting authority is changed by Full Research.
- Failed read, wrong source/build, missing V4/V4.2 parity, bad side mapping, unexpected return status, or fatal research exception is a visible Cloud Run failure with an immutable audit. Optional V5.2 archive proof failure is separately recorded as `DEGRADED` and does not create fake signals.

**Not implemented or claimed:** an authorized deployed V5 live betting model, any profitable historic ROI, a new uninspected holdout, or full point-in-time source certification. The outputs identify the remaining blockers rather than silently promoting 2024–2025 retrospective patterns.

## Local verification

`pytest -q test_v53.py` — 15 tests passed with the current package; direct GCP data reads and end-to-end heavy V4+legacy training have not been executed here. `python -m py_compile` passes for the replacement scripts and included modules. A normal real-world execution is required to catch Cloud Run/BigQuery/data-contract failures, which are instrumented in the audit JSONL.
