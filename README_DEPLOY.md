# NCAAF Engine V2.12 — PT Session/HTML Fallback + Expert Atom Bridge Fix

This is a narrow research-integration repair on top of V2.11.1. Production V1 remains frozen.

## Why this patch exists
The 2026-10-06 Heavy run completed successfully after the nullable-mask hotfix, but exposed two separate integration failures:

1. Prediction Tracker archive and live CSV requests returned HTTP 403 from Cloud Run, so no external-rating rows were attached.
2. The historical Pathi/Big Al W/L cache was populated, but the lean Miner frame did not contain the named deterministic flags, so the Expert/Model Atom Bridge reported Pathi=0 and BigAl=0.

## V2.12 fixes

### Prediction Tracker retrieval
- Browser-like `requests.Session` fetch.
- Same-origin parent-page warm-up/cookie session.
- Correct Referer for archive and current-week CSV requests.
- Archive page: `ncaaarchive.html` -> `ncaaYYYY.csv`.
- Live page: `predncaa.php` -> `ncaapredictions.csv`.
- If the live CSV is still blocked, parse the individual-system table directly from `predncaa.php`.
- Existing GCS cache remains the final fail-safe.

Historical archives remain fail-closed: no synthetic ratings and no substitution with the site's generic prediction average.

### Pathi / Big Al Miner bridge
The existing Miner frame now materializes deterministic Pathi/Big Al flags from the pregame fields before atom construction. This connects the already-working historical expert systems to the existing Miner rather than creating a second miner.

### Individual external ratings retained
In addition to the fixed published `META_MARGIN`, the research frame now keeps each component separately:
- ESPN FPI
- Pi-Ratings Bias
- Dokter
- Keeper
- Pigskin Index

Research-only Miner atoms now include:
- individual component edge vs market,
- individual component agreement/conflict with OOF CORE,
- 4-of-5 and 5-of-5 external consensus,
- low/high five-system dispersion,
- META vs CORE agreement/conflict.

All external component atoms remain one correlated external evidence family. They are not five Bet Authority votes.

## Authority contract
- Production CORE: unchanged.
- Production ledger: unchanged.
- Utils/backend: unchanged.
- External ratings: authority=0.
- OOF CORE/specialist bridge atoms: authority=0.
- 2026 outcomes: never used for discovery/confirmation.
- No automatic promotion.

## Deploy
If V2.11.1 is already deployed, replace only:

`ncaaf_research_v2.py`

Redeploy the training job, then run:

**NCAAF Research — Heavy Challenger Search**

Do not run Production Publish.

## Lines to inspect in the next Heavy log
- `[NCAAF-PT-SEASON]` — completed-season archive retrieval/cache status
- `[NCAAF-PT-LIVE]` — live CSV or HTML fallback status
- `[NCAAF-PT-CURRENT-MERGE]` — archive + current-week merge
- `[NCAAF-PT-MATCH]` — historical game-match coverage
- `[NCAAF-PT-CONTRACT]` — overall external-rating status
- `[NCAAF-RV25-ATOM-BRIDGE]` — Pathi/BigAl/CORE/specialist/external atom counts
- `[NCAAF-RV25-CONTRACT]` — confirmed bridge mechanisms
- `[NCAAF-HEAVY-RUN] status=PASS`
