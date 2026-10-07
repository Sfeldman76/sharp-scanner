# Prediction Tracker Residential Feeder — V2.15

This feeder runs **outside Cloud Run** on a normal residential network and writes validated raw Prediction Tracker artifacts to GCS. The NCAAF Heavy and Weekly jobs now read these GCS blobs first.

## One-time Windows setup

1. Install Python 3.11+ and Google Cloud CLI.
2. Authenticate Application Default Credentials:

   `gcloud auth application-default login`

3. Install feeder dependencies from this folder:

   `python -m pip install -r requirements_prediction_tracker_feeder.txt`

4. Install Chromium for the fallback browser path:

   `python -m playwright install chromium`

5. Test locally without uploading:

   `python prediction_tracker_feeder.py --self-test`

6. Run the feeder:

   `python prediction_tracker_feeder.py --bucket sharp-models`

The script tries normal HTTP first. If the site challenges that request, it falls back to Chromium on the residential machine and reuses the browser cookie jar.

## What is uploaded

Current canonical raw blobs:

- `gs://sharp-models/research/ncaaf/external/prediction_tracker/raw/ncaa2022.csv`
- `.../ncaa2023.csv`
- `.../ncaa2024.csv`
- `.../ncaa2025.csv`
- `.../ncaa2026.csv`
- `.../ncaapredictions.csv`
- `.../predncaa_live_page.txt`

Each successful current run is also preserved under:

- `gs://sharp-models/research/ncaaf/external/prediction_tracker/snapshots/<UTC timestamp>/`

The feeder manifest is:

- `gs://sharp-models/research/ncaaf/external/prediction_tracker/feeder_manifest.json`

The feeder validates content **before overwriting** the last good raw blob. Cloudflare/challenge pages are rejected.

## Scheduling

Run the PowerShell wrapper `run_prediction_tracker_feeder.ps1` every 6 hours during the football season using Windows Task Scheduler. Historical 2022–2025 files are downloaded only when missing unless `--refresh-history` is specified. The 2026 archive, current-week CSV, and named live page refresh every run.

The Cloud-side model rejects current feeder data older than 24 hours by default. Override only if necessary with `PT_FEEDER_CURRENT_MAX_AGE_HOURS`.
