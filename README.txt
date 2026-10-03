NFL Production Betting V2 — frozen edge-selector promotion
Generated 2026-10-03

REPLACE ONLY
1) nfl_engine.py
2) sharp_line_dashboard.py

NO CHANGES REQUIRED
- train_job.py
- utils.py
- ncaaf_research_v2.py
- NCAAF production files

PRODUCTION POLICY
SPREADS
- Prediction/fair-value champion remains frozen.
- A production BET is authorized only when BOTH approved independent Spread mechanism families trigger the SAME side:
  * NFL_SPREAD_HOME_FAVORITE_OFF_SU_LOSS_FADE
  * NFL_SPREAD_ROLE_FLIP_DOG_TO_FAVORITE_FADE
- One family only => PASS.
- Family conflict => PASS.
- Execution price must be -110 or better; otherwise EDGE — NO EXEC QUOTE.

TOTALS
- Production authority is limited to:
  * NFL_TOTAL_EARLY_DIVISION_UNDER
- Trigger direction must be UNDER.
- Execution price must be -110 or better; otherwise EDGE — NO EXEC QUOTE.

H2H
- MODEL ONLY. No production betting edge is approved.

FREEZE / SAFETY
- Approved Edge V2 contract SHA is frozen to:
  6a1e2e770071725081e29855d47f09aaa797831d62b9b2a8a13968661f8dc20c
- A future Heavy Research run cannot silently change production. If Edge V2 publishes a different contract SHA, Spread/Totals hold MODEL ONLY until explicit promotion review.
- Other CORE / STAT / PBP / MARKET / SYSTEM research remains shadow unless explicitly promoted.
- No automatic bet execution.
- No automatic model or policy promotion.
- 2026 remains prospective evidence; this patch does not retune rules on 2026.

LIVE REFRESH
- NFL Production — Weekly Update: run once per week after prior games settle / new slate is ready.
- Background scanner: automatically reapplies the frozen selector + current execution-price gate to line/price changes. No weekly rerun is needed for market moves.
- NFL Research — Heavy Challenger Search remains the research workflow and cannot silently overwrite this frozen production contract.

AFTER DEPLOY
1) python nfl_engine.py --self-test
2) Run NFL Production — Weekly Update ONCE.
3) Confirm logs include [NFL-PROD-BET-V2-LIVE].
4) Thereafter normal background NFL scans should refresh current recommendations.

EXPECTED SELF TEST
- engine_source_tag: nfl-engine-v3.2-production-betting-v2-20261003
- status: PASS
- 31/31 bundled components compile

V3.2.1 APPROVAL PATCH
- Explicitly approves the Edge V2 contract published by the 2026-10-03 heavy research run.
- Approved contract SHA: 6a1e2e770071725081e29855d47f09aaa797831d62b9b2a8a13968661f8dc20c
- No betting mechanism definitions changed: Spread requires both approved independent families; Totals uses the approved early-division Under family; H2H remains model only.
