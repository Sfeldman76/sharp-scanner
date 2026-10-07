# NCAAF Research V2.17 — Sparse Prediction Tracker / Manual GCS

## Why this patch exists
Prediction Tracker does not cover every NCAA game, and a listed game can have fewer than all five benchmark systems. V2.17 treats Prediction Tracker as **sparse optional external intelligence** instead of a complete game universe.

## Behavior
- A game absent from Prediction Tracker stays in the NCAAF model normally. It simply has **no PT external signal**.
- If a PT game has 1–4 of the named systems, those individual component ratings are preserved for research.
- `META_MARGIN` is still created **only with all five exact named systems**. No imputation, renormalization, or guessed rating is allowed.
- Five-system consensus/dispersion atoms require full 5-of-5 coverage.
- PT remains one correlated external family and has **zero Production/Bet Authority** unless separately validated under existing gates.
- 2026 remains sealed from discovery/confirmation.
- No paid proxy / Oxylabs / web-unblocker is required. Heavy and Weekly read validated GCS uploads.
- Current YTD archive freshness defaults to 168 hours; live/current files to 72 hours.

## Replace
Replace these files in the `sharp-train-job` source:

- `ncaaf_research_v2.py`
- `train_job.py`

`prediction_tracker_cloud_feeder.py` is no longer imported or required by V2.17.

## Deploy
Redeploy the existing `sharp-train-job` exactly as you normally do from its source directory.

## Run
After deployment, run:

**NCAAF Research — Heavy Challenger Search**

Do not run Production Publish for this research change.

## Expected log markers

```
[NCAAF-RV217-DEPLOY-PREFLIGHT] PASS source_tag=ncaaf-research-v2.17-sparse-pt-manual-gcs-20261007
[NCAAF-PT-GCS-MANUAL] mode=GCS_ONLY ... paid_proxy_required=FALSE
[NCAAF-PT-MATCH] ... partial=... full_five=... no_pt=... sparse=TRUE authority=0
[NCAAF-PT-CONTRACT] ... missing_games_expected=TRUE sparse=TRUE ...
```

The Heavy run should still finish with production mutation disabled.
