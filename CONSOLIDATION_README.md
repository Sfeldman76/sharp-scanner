# Sharp Scanner 2026-10-23 — audited first-stage runtime consolidation

## Result
- Original source snapshot: **87** top-level Python modules; original ZIP also carries a large `.git/` history.
- Deployment profile: **66** retained Python modules; **21** scripts are outside the identified runtime dependency graph and are preserved in your original full ZIP.
- This is **NOT** a nine-file final merge. The remaining files are referenced by the current dashboard, legacy job dispatcher, exact-local source preflight, NFL engine, or maintenance entrypoints. Removing them without refactoring would break supported workflows.
- The V5.5 rolling replay and ATS evaluator are **already consolidated** in `ncaaf_v552_live.py` and `ncaaf_v55_primary_research.py`. Duplicate standalone `v55_ats_evaluator.py` and obsolete `ncaaf_v55_weekly.py` are not deployed.

## Actual source fixes
1. **Heavy Research route repaired**: The live UI submits `ncaaf_v55_heavy`, and `train_job.py` now accepts it as equivalent to `ncaaf_research_heavy`, invoking the existing V5.5 ATS-primary/legacy-secondary workflow.
2. **On-demand replay viewer** added to the NCAAF V5.5 Heavy Research controls. It verifies immutable GCS run prefix, model, audit status, output digests, and displays game-level predictions and season-level MAE/RMSE. No startup query or initial render cost.
3. **Security correction**: Removed a plaintext API key from the dashboard. Configure the `ODDS_API_KEY` secret/environment variable **before deploying**; the previous embedded key should be rotated. Do not deploy this copy without checking the credential binding.
4. `.dockerignore` excludes `.git/`, temporary files, and legacy archive directories from Docker build context.

## Training/research mechanics and limitations
- The actual V5.5 `NONLINEAR_HGB` model calls the V4 dynamic feature/state and HGB fit utilities; those dependencies are intentionally retained, along with old system Miner components.
- The V5.5 replay fits only before the tested year and shows retrospective margin prediction accuracy. Live BigQuery/GCS access and historical source-data checks require Cloud Run; we did not query your private datasets here.
- Historical V5.5 model recipe selection and historical feature vintages are **not independently certified**; no validated 2024–25 held-out betting win-rate claims.
- The primary research currently reports ATS `HOLD` until genuine timestamped historical bookmaker spread/brain inputs pass the separate vintage/source rules. Showing a historical margin forecast does not make it an ATS backtest.
- The research job never authorizes or auto-promotes betting thresholds.

## Deploy
1. Retain an untouched backup of the original source ZIP / Git commit.
2. Place this deployment tree at repository root or deploy from this directory. Do not include the original `.git` data in Docker.
3. Ensure Cloud Run has `ODDS_API_KEY` configured (Secret Manager preferred). Rotate the exposed prior key.
4. Rebuild both dashboard and job images; check the job menu calls `MARKET=ncaaf_v55_heavy`.
5. Run Heavy Research, open the on-demand rolling replay viewer and inspect `latest_status.json`, fold metrics and research holds.
6. Run the normal weekly/market refresh smoke checks separately. NFL and existing model authority are intentionally unchanged.

## Next stage: real consolidation, not hiding modules
- Untangle the 19 exact-file-loaded legacy components from the job-wide import/preflight. Those functions currently block collapsing the runtime to ~9 files.
- Move legacy NCAAF research behind an optional separate compatibility runner after parity tests; keep present qualified systems and retrospective reports.
- Consolidate V4 model feature utilities and CFBD truth into a dedicated, tested V5.5 model foundation.
- Consolidate three legacy source paths only after comparing outputs on fixed historical games and verifying old UI/system report access.
- Preserve NFL's internally bundled components and current public import contract.

## Coverage
- Python syntax compilation of all deployed modules.
- Import graph closure plus `train_job`'s `_load_exact_local_module` preflight and standalone CLI/server roots.
- Static dashboard-to-trainer V5.5 Heavy Research route reconciliation.
- Historical replay publishing path and model provenance inspection.
- This is **static/local validation**, not live Cloud Run/GCS smoke-test proof.
