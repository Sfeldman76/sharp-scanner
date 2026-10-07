# NCAAF Engine V2.15 — GCS-first Prediction Tracker Feeder + Expert Occurrence Reconciliation

## Cloud deployment

From V2.14.3, replace only:

- `ncaaf_research_v2.py`
- `train_job.py`

Do **not** replace Production V1, CORE challenger, dashboard, production ledger, or `utils.py`.

Redeploy `sharp-train-job` and run **NCAAF Research — Heavy Challenger Search** after the feeder has populated GCS.

Expected deploy marker:

`[NCAAF-RV215-DEPLOY-PREFLIGHT] PASS`

## Prediction Tracker architecture

V2.15 makes GCS the model-side contract. Heavy and Weekly now read fresh raw Prediction Tracker artifacts from GCS **before** attempting any direct/relay web access.

Current feeder freshness default: 24 hours (`PT_FEEDER_CURRENT_MAX_AGE_HOURS`).

The local/residential feeder is included:

- `prediction_tracker_feeder.py`
- `requirements_prediction_tracker_feeder.txt`
- `run_prediction_tracker_feeder.ps1`
- `README_PREDICTION_TRACKER_FEEDER.md`

Run the feeder on a normal residential Windows machine/network. It validates the content first, rejects Cloudflare/challenge pages, preserves exact source bytes, saves immutable timestamped snapshots, and does not overwrite the last good raw blob on a failed fetch.

## Expert occurrence reconciliation

The prior 19-count difference was not missing mapped occurrences. The historical W/L occurrence ledger stores **graded** occurrences, while the system summary `fired` count also includes fired rows lacking a grade/ATS target.

V2.15 logs both explicitly and reconciles against the occurrence ledger:

`[NCAAF-RV215-EXPERT-OCCURRENCE-RECON] ... fired=... graded=... occurrence_records=... projected=... ungraded=... projection_delta=0 ...`

The bridge fails closed if `projected != occurrence_records`.

## Expected successful research markers

- `[NCAAF-RV215-EXPERT-OCCURRENCE-RECON] status=PASS ... projection_delta=0`
- `[NCAAF-RV215-EXPERT-OCCURRENCE-BRIDGE] status=PASS ...`
- `[NCAAF-PT-FEEDER-MANIFEST] status=READY ...`
- `[NCAAF-PT-GCS] ... status=PASS ...`
- `[NCAAF-PT-HEADER-VERIFY] ...`
- `[NCAAF-PT-CONTRACT] status=PASS ... matched_rows=>0`
- `[NCAAF-RV25-ATOM-BRIDGE] ... pathi=>0 bigal=>0 external=>0`

The external ratings remain research-only and have zero direct Production/Bet Authority.
