# NCAAF V2.14.2 — PT Challenge Rejection + Exact Expert Occurrence Bridge

This is a narrow research-only hotfix over V2.14.1.

Changes:
- Reject Cloudflare / anti-bot challenge pages before parse or GCS cache.
- Try both HTTPS-origin and HTTP-origin Jina relay variants.
- Try both `predncaa.php` and static `predncaa.html` for the live named table.
- Use the already-validated dashboard historical system occurrence ledger as the primary source for Pathi/Big Al Miner atoms.
- Preserve HOME- and ROAD-side expert occurrences separately.
- Convert expert masks to strict non-null booleans before Miner aggregation; fixes `ValueError: cannot convert NA to integer`.
- Production authority remains 0 for research bridge/external ratings.

Deploy from V2.14.1:
1. Replace `ncaaf_research_v2.py` and `train_job.py`.
2. Redeploy.
3. Run **NCAAF Research — Heavy Challenger Search**.
4. Do not run Production Publish.

Expected log markers:
- `[NCAAF-RV2142-DEPLOY-PREFLIGHT] PASS`
- `[NCAAF-RV2142-EXPERT-OCCURRENCE-BRIDGE] status=PASS ... pathi_cols=>0 bigal_cols=>0 ...`
- `[NCAAF-RV25-ATOM-BRIDGE] ... pathi=>0 bigal=>0 ...`
- Prediction Tracker will either show verified data, or fail closed. Challenge pages must never appear as `Home/Road` CSV and must never be cached as valid PT data.
