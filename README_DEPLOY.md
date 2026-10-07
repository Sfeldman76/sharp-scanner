# NCAAF V2.16 — VERIFIED REBUILD

This is a clean rebuild of V2.16. The three runtime files were recompiled and self-tested together before packaging. **If the deployed Heavy log does not show `[NCAAF-RV216-DEPLOY-PREFLIGHT] PASS`, Cloud Run is not running this package.**

# NCAAF V2.16 — Cloud-only Prediction Tracker ingestion

This replaces the V2.15 residential-PC feeder. Nothing runs on the operator's computer.

## Replace / add in the repository

Replace:
- `ncaaf_research_v2.py`
- `train_job.py`

Add:
- `prediction_tracker_cloud_feeder.py`

No dashboard change is required. The feeder now runs automatically inside:
- `NCAAF Production — Weekly Update` (current data only)
- `NCAAF Research — Heavy Challenger Search` (history + current)

## Required cloud credential

Prediction Tracker blocks ordinary Cloud Run/datacenter egress, so a web-unblocker/residential-proxy credential is required for a cloud-only design.

Supported configuration:

### Oxylabs Web Unblocker
Set Cloud Run secrets/env vars:
- `OXYLABS_USERNAME`
- `OXYLABS_PASSWORD`

Optional:
- `OXYLABS_ENDPOINT=unblock.oxylabs.io:60000`
- `PT_UNBLOCKER_GEO=United States`
- `PT_UNBLOCKER_TIMEOUT=90`

### Generic compatible proxy
Alternatively set:
- `PT_UNBLOCKER_PROXY_URL=http://user:password@host:port`

## Secret Manager example from Google Cloud Shell

```bash
printf '%s' 'YOUR_OXYLABS_USERNAME' | gcloud secrets create pt-oxylabs-username --data-file=- --replication-policy=automatic
printf '%s' 'YOUR_OXYLABS_PASSWORD' | gcloud secrets create pt-oxylabs-password --data-file=- --replication-policy=automatic

gcloud secrets add-iam-policy-binding pt-oxylabs-username \
  --member='serviceAccount:sharp-train-sa@sharplogger.iam.gserviceaccount.com' \
  --role='roles/secretmanager.secretAccessor'

gcloud secrets add-iam-policy-binding pt-oxylabs-password \
  --member='serviceAccount:sharp-train-sa@sharplogger.iam.gserviceaccount.com' \
  --role='roles/secretmanager.secretAccessor'

gcloud run jobs update sharp-train-job \
  --project=sharplogger \
  --region=us-east4 \
  --set-secrets='OXYLABS_USERNAME=pt-oxylabs-username:latest,OXYLABS_PASSWORD=pt-oxylabs-password:latest'
```

If the secrets already exist, add a new version instead of recreating them.

## Expected log flow

```text
[NCAAF-PT-CLOUD-FEEDER-PREFLIGHT] PASS ...
[NCAAF-PT-CLOUD-FEEDER] source=ncaa2022 status=PASS|SKIPPED_VALID_EXISTING ...
...
[NCAAF-PT-CLOUD-FEEDER] source=ncaapredictions status=PASS ...
[NCAAF-PT-CLOUD-FEEDER] source=predncaa_live_page status=PASS ...
[NCAAF-PT-CLOUD-FEEDER] status=PASS ...
[NCAAF-PT-FEEDER-MANIFEST] status=READY ...
[NCAAF-PT-CONTRACT] status=PASS history_attached=True ... matched_rows>0 ...
[NCAAF-RV25-ATOM-BRIDGE] market=spreads ... external>0 ...
```

If credentials are not configured, the NCAAF job continues safely with:

```text
[NCAAF-PT-CLOUD-FEEDER] status=CONFIG_MISSING ...
```

Production authority remains zero for this external family.
