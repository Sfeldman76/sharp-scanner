NFL Engine V3.0.1 — Weekly Update Memory Fix
Date: 2026-10-03

Symptom addressed
-----------------
Weekly production reaches [NFL-PROD-V1-PROMOTION-CLOCK] and then the Cloud Run
container is terminated by signal 9 before the model-authority market phase logs.

Root cause in V3.0
------------------
The canonical Utils market adapter called read_recent_sharp_moves(hours=336)
twice with SELECT * and no server-side Sport/Game_Start restriction.  Utils then
materialized up to 14 days of every sport from sharp_moves_master and
moves_with_features_merged into pandas before the adapter filtered to NFL.
That creates an unnecessary high-memory transient on the weekly production path.

Fix
---
1. utils.read_recent_sharp_moves remains backward compatible and adds optional:
   sport, game_start_after, game_start_before, use_bq_storage.
2. The NFL market backend now REQUIRES server-side Sport='NFL' plus the upcoming
   8-day Game_Start window before either market source is materialized.
3. The weekly NFL market snapshot disables the BigQuery Storage API dataframe
   path for these reads to reduce peak memory further.
4. New phase logs identify AUTHORITATIVE_MARKET and SHADOW_EVIDENCE start/done.
5. nfl_engine source tag is now:
   nfl-engine-v3.0.1-weekly-memory-safe-20261003

No model/research policy changes
--------------------------------
Production V1 artifacts, frozen model-authority contract, thresholds, ledgers,
prospective predictions, research contracts and promotion gates are unchanged.
Do NOT rerun Historical Validation solely for this hotfix.

Deploy
------
Replace ONLY these repo-root files:
  nfl_engine.py
  utils.py

Then redeploy the Cloud Run job/service using the same process you normally use.
Run:
  python nfl_engine.py --self-test
Expected: status PASS and weekly_memory_guard SERVER_SIDE_NFL_PLUS_GAME_WINDOW_FILTER.

Then rerun dropdown:
  NFL Production — Weekly Update

Expected new logs after promotion clock:
  [NFL-PROD-V1-LIVE-PHASE] phase=AUTHORITATIVE_MARKET start=TRUE
  Utils load messages showing sport=NFL, game_start_window=bounded, bq_storage=False
  [NFL-PROD-V1-LIVE-PHASE] {"phase":"AUTHORITATIVE_MARKET","status":"DONE",...}
  [NFL-PROD-V1-LIVE-PHASE] phase=SHADOW_EVIDENCE start=TRUE
