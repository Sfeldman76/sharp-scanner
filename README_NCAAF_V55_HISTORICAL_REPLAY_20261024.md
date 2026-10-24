# Sharp Scanner NCAAF V5.5 — Source-aware historical brain replay

## Deliverable

This is the **entire Sharp Scanner repository**, built from the previous source-recovery release. Deploy the repository as a whole. **Do not merge an overlay ZIP.** Production V1, NFL, dashboard, `utils.py`, and the existing V5.5 CORE trainers are unchanged.

## What is now implemented

`ncaaf_historical_brain_replay_v1.py` is a live-wired, fail-closed **retrospective diagnostic** replay integrated into the existing `MARKET=ncaaf_v55_heavy` Heavy Research route through `ncaaf_heavy_brain_bridge.py`.

It executes *after* existing Heavy CORE/legacy research, where the previous strict bridge returned a 7-input HOLD:

1. Reads 2013–2025 CFBD archive research and historical STAT/context records **directly from BigQuery** using fixed read-only queries, the configured Cloud Run service account, and season bounds. Uses CFBD physical game IDs to generate an independent year-by-year universe manifest; scores/lines come from separately matched context.
2. Joins on exact `(season, game_id, home_team, away_team)`, then only on unambiguous exact `(season, date, home_team, away_team)`. Does not fabricate matches or erase unmatched CFBD games. Includes explicit game-side and join coverage diagnostics.
3. Evaluates a deliberately modest catalog of **10 explicit research hypotheses** over 2013–2025, recording FIRE / NO_FIRE / UNKNOWN per rule/game. These are **not** representations of the legacy 191 confirmed mechanisms; the source labels say POST_HOC_RETROSPECTIVE_HYPOTHESIS, and these hypotheses must not be called existing historical bets. STAT: 4; CFBD-based Miner: 3; Pathi opening-key descriptive conditions: 3.
4. Loads 2022–2025 **raw PT GCS CSV archives** using the existing strict header parser. Matches only unambiguous team/date identities and marks missing archives and source publication time as unknown. PT average-vs-open direction is numeric exploratory **only**: source value orientation and publication timing are not certified.
5. Writes explicit unknown eligibility for Big Al (warehouse context is not a historic pick ledger) and PT wherever genuine dated picks cannot be verified. Pathi opening-key tests are **non-directional context**, never support votes; line-movement tests remain excluded because historical original quote timing is unknown.
6. Compares the recovered V5.5 CORE replay with rule-family support/conflict on exact, matched game IDs. Reports per-brain and joint descriptive ATS rates, grade pushes separately, and count distinct brain families (not mechanism count). Only matched games participate in CORE comparisons; broader independent CFBD rule discovery history is retained separately.
7. Does **not** modify the strict `ncaaf_brain_full_history.py` bet-certification contract, cannot turn unverified data into vintage-certified bets, cannot use 2026 outcomes for discovery, and never mutates production.

## Historical integrity limitations

- These hypotheses and their thresholds were defined **after** 2024–2025 results. Therefore 2024–2025 is only a descriptive chronological slice here, **not a genuinely independent holdout**. Split labels are DISCOVERY (2013–21), SELECTION (2022–23), and CONFIRMATION (2024–25) for organizing data only. Prospective validation is needed for any causal / actionable model selection.
- The V5.5 CORE replay recipe was retrospectively selected. Its historical results are not a certified historic point-in-time predictive record.
- Bookmaker opens, PT archives, and historical feature construction lack original verified as-of publication times. Reported ATS outcomes are **exploratory reconstructed grades**, not validated historic betting return.
- Big Al's actual historical selections, full existing Miner rule-version histories, and full historical PT archive before 2022 are not present in the audited sources. These lanes are not fabricated.
- No `BET` or `STRONG BET` promotion is possible from these reports; bet authority = 0 and Production V1 is frozen.

## Deploy and inspect

Deploy this **complete repository** through the existing Sharp Scanner image build process, keeping existing Cloud Run job environment `SPORT=NCAAF` and `MARKET=ncaaf_v55_heavy`. The new replay is enabled by default; `NCAAF_BRAIN_RETROSPECTIVE_REPLAY_ENABLED=0` opts it out. It uses `GOOGLE_CLOUD_PROJECT` (default `sharplogger`) and `NCAAF_HISTORY_DATASET` (default `sharp_data`). The service account requires BigQuery read access to the two existing tables and GCS read/write access to the existing research bucket.

After the job finishes, see:

```
gs://sharp-models/research/ncaaf/v55/combined_brains_shadow/latest.json

gs://sharp-models/research/ncaaf/v55/combined_brains_shadow/runs/<run-id>/status.json

gs://sharp-models/research/ncaaf/v55/combined_brains_shadow/runs/<run-id>/retrospective_replay/audit.json

gs://sharp-models/research/ncaaf/v55/combined_brains_shadow/runs/<run-id>/retrospective_replay/cfbd_universe_manifest.csv

gs://sharp-models/research/ncaaf/v55/combined_brains_shadow/runs/<run-id>/retrospective_replay/rule_evaluations_retrospective.csv

gs://sharp-models/research/ncaaf/v55/combined_brains_shadow/runs/<run-id>/retrospective_replay/rule_descriptive_ats_by_year.csv

gs://sharp-models/research/ncaaf/v55/combined_brains_shadow/runs/<run-id>/retrospective_replay/core_independent_brain_diagnostics.csv

gs://sharp-models/research/ncaaf/v55/combined_brains_shadow/runs/<run-id>/retrospective_replay/core_incremental_exploratory.csv
```

Look for `[NCAAF-HIST-REPLAY]` in the Heavy logs. A successful replay reports `RETROSPECTIVE_DIAGNOSTICS_COMPLETE_NO_CERTIFIED_BET_AUTHORITY` under `status.json.retrospective_replay`, while the **strict combined-brain certification status appropriately remains HOLD** until independently archived rule, signal, eligibility and original-quote vintages can be verified.

### Handling a replay HOLD

`retrospective_replay/audit.json` gives the actual error type/reason (BigQuery permission, missing source column, identity collision, PT parser, etc.). If the strict layer still reports five inputs missing, that is **expected** and is not a reason to rebuild fake input files.

## Release checks

- All **50** local research validation tests pass.
- Python compilation passes.
- Clean ZIP extraction and retest required before handoff.
- No access to live BigQuery or Cloud Run was used for this release; real-data performance and coverage remain unverified until first deployed execution.
