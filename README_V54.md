# NCAAF V5.4 — CFBD Historical Source of Truth + Full Research

## Contract

- **CFBD authoritative historical football**: BigQuery `sharplogger.sharp_data.ncaaf_cfbd_api_archive` `/games` immutable snapshots provide game IDs, opponents, kickoff and verified completed final scores. `sharplogger.sharp_data.ncaaf_cfbd_archive_research_v1` provides prior-only CFBD football feature candidates from 2013–2025.
- **Market data separately**: Only allowlisted historical open/close spreads, totals, moneylines cross from `sharplogger.sharp_data.ncaaf_historical_game_side_raw`, filtered to `Source_Name='CollegeFootballData'` and joined to the CFBD game ID and team. No historical market-table football results or context statistics are used. Actual open/close quote timestamps/executability remain uncertified.
- **External intelligence separate**: Prediction Tracker GCS snapshots and Big Al observation files are separate *hypothesis/prospective* evidence, not the source of game results or CFBD football statistics.
- **Legacy research functions reused**: the original `utils.add_pathi_bigal_rule_flags`, `ncaaf_research_v2.run_system_miner_v3`, `_extended_atoms`, and original team alias resolver are still used. The original dashboard's historical cache builders, STAT fit, legacy historical CORE challenger and old `run_ncaaf_research_v2()` are intentionally bypassed for V5.4 to prevent importing non-CFBD football facts. They remain available for older separate workflows. This may reduce the number of historical systems that fire; missing flags must be shown as not evaluable, not filled with legacy data.
- **No automatic bets/promotions**: all research remains `SHADOW_ONLY`. 2026 outcomes are excluded from historical research. Publication vintage for prior CFBD features must still be certified before use as historical proven signals.

## Included files

- `ncaaf_v54_cfbd_truth.py` — canonical CFBD game, score, feature, team-side state and market-only join with strict integrity checks
- `ncaaf_v54_full_research.py` — full one-shot research and audit stages, CFBD-only V4/V4.2/Miner, source manifests, Big Al/Pathi rule flags, PT external refresh, V5/V5.1 edge attribution and shadow authority
- `ncaaf_margin_tournament_v4.py`, `ncaaf_v42_conditional_features.py` — direct canonical CFBD path when the source marker is present (avoids old second-stage legacy bridge)
- `ncaaf_v53_full_research.py` — preserved older V5.3 path, used for shared audit/helper functions
- `train_job.py`, `sharp_line_dashboard.py` — adds V5.4 full research; retains existing NCAAF weekly frozen V1 and heavy search options
- Existing V5/V5.1/V5.2 research modules and tests
- `test_v54_cfbd_truth.py`, `test_v54_full_integration.py`, `test_v53.py`

**Dependencies from existing deployment that must remain in the repository/container:** `utils.py`, `ncaaf_research_v2.py`, `ncaaf_core_challenger_v2.py`, `ncaaf_v43_validation.py`, `ncaaf_v52_brain_bridge.py` and any imports the current working app already needs. Do not delete these. This archive includes research route files, **not the complete repo**.

## Deployment

Replace/add files from the ZIP into the existing `sharp-scanner` repository. Rebuild and redeploy `sharp-train-job` and the dashboard image from the same source revision, including `ncaaf_v54_cfbd_truth.py`. The job's service account requires read permissions on the CFBD API archive and research feature tables and write access to the research artifacts GCS bucket.

The NCAAF dropdown keeps: weekly Production V1, V5 full CFBD-history research, and legacy heavy challenger. Weekly Production V1 remains its **own route**, not a V5 weekly refit.

Cloud Shell direct run (after deploying the new training image):

```bash
gcloud run jobs execute sharp-train-job \
  --project=sharplogger --region=us-east4 \
  --update-env-vars=SPORT=NCAAF,MARKET=ncaaf_v54_full_research,HEADLESS=1,NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED=1 \
  --task-timeout=14400s --wait
```

Read final report:

```bash
gcloud storage cp gs://sharp-models/research/ncaaf/v54_full_research/latest_report.json ~/ncaaf_v54_report.json
```

## Audit stages / fail-closed behavior

The Cloud Run log produces `[NCAAF-V53-AUDIT]` stage records under the V5.4 route (shared logger retained for compatibility) and a final `[NCAAF-V54]` status line. Each start/pass/fail includes timestamps, duration and error traces. The GCS report saves the immutable run-specific audit and exposes `report_uri` / `audit_uri`.

The checks include: archive opt-in, module/function/build matching, GCS/BQ manifest, CFBD source hash verification, CFBD score and game-ID matching, two-sided score reciprocity, CFBD home/away orientation and historical season coverage, market-side price joins and contradictions, no 2026 labels, no non-CFBD football source, V4/V4.2 full run, Big Al/Pathi CFBD-based reconstruction, full System Miner per market, V5 historical edge replay, family attribution and no automatic betting authority.

An incomplete CFBD API archive, conflicting final scores, insufficient coverage or mismatched team identifiers **fails the run**. Legacy historical score tables are never a fallback. Failed runs still publish a report and exit nonzero.

## Known live-validation limits

Local unit and mocked integration tests demonstrate program flow and safety contracts only. **No real Cloud Run V5.4 execution has been observed.** Its first run may reveal BigQuery field coverage differences, genuinely missing 2013–2025 CFBD API `/games` snapshots, feature completeness, unrecognized team keys or missing prospective pregame archives. Inspect the exact audit rather than relaxing source integrity gates.

Existing V5.3 log from 2026-10-20: successful end-to-end run with 4,113 edge games, 5,106 matched retrospective Big Al/Pathi occurrences, 20,705 matched retrospective Miner system/game rows; evidence remains **shadow**. The V5.3 run mixed preexisting historical caches. Its historical system counts must not be treated as CFBD-only V5.4 results.

The V4 ability to predict a game margin is not equivalent to verified ATS profitability. Odds/ROI and pregame source-vintage proof remain open.
