# NCAAF V2.14.3 — Expert Occurrence Key Normalization Fix

This is a narrow research-only hotfix over V2.14.2.

Changes:
- Fixes the expert occurrence-ledger key mismatch that caused 15 Pathi + 5 Big Al columns to project zero fires into the Miner.
- Uses the same compact lowercase alphanumeric team-token contract as the dashboard historical W/L ledger.
- Rebuilds occurrence keys from season/date/team/opponent fields instead of trusting a differently-normalized external key string.
- Adds a hard diagnostic: if validated historical systems have nonzero fired counts but the Miner projection is zero, the occurrence bridge fails and falls back instead of falsely reporting PASS.
- Retains strict non-null boolean masks before Miner aggregation.
- Prediction Tracker anti-bot challenge rejection remains unchanged; PT still fails closed when transport is unavailable.
- Production authority remains 0 for expert/external research bridges.

Deploy from V2.14.2:
1. Replace `ncaaf_research_v2.py` and `train_job.py`.
2. Redeploy.
3. Run **NCAAF Research — Heavy Challenger Search**.
4. Do not run Production Publish.

Expected log markers:
- `[NCAAF-RV2143-DEPLOY-PREFLIGHT] PASS`
- `[NCAAF-RV2143-EXPERT-OCCURRENCE-BRIDGE] status=PASS ... home_pathi_fires=>0 and/or road_pathi_fires=>0 ... home_bigal_fires=>0 and/or road_bigal_fires=>0`
- `[NCAAF-RV25-ATOM-BRIDGE] market=spreads ... pathi=>0 bigal=>0 ...`
- Prediction Tracker will either show verified data or fail closed. Anti-bot challenge pages are never cached as PT data.
