# NCAAF V2.17 — GitHub root build fix

## Root cause

The GitHub/Cloud Build trigger is invoking the repository's default `Dockerfile`.
The current root `Dockerfile` is the NCAAF weekly-uploader Dockerfile, so the build
tries to copy `requirements_weekly_uploader.txt` and fails.

The V2.17 Python files are not the cause of that failure.

## Replace these files in the GitHub repository root

1. Replace `Dockerfile` with the `Dockerfile` in this package.
2. Replace `train_job.py`.
3. Replace `ncaaf_research_v2.py`.

The correct root Dockerfile is:

```dockerfile
FROM python:3.11
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train_job.py"]
```

`Dockerfile.job` is included as the same training-job Dockerfile for explicit builds.

## Keep the uploader separate

The old uploader Dockerfile is included only as `Dockerfile.uploader.reference`.
Do not leave that file named `Dockerfile` at the repository root if this GitHub
trigger is the sharp-train-job build.

The weekly uploader is already deployed separately through its Cloud Run overlay.

## Prediction Tracker note

For V2.17, `prediction_tracker_cloud_feeder.py` and
`README_PREDICTION_TRACKER_CLOUD.md` are old V2.16-era proxy/unblocker files.
V2.17 uses manual/GCS-first sparse Prediction Tracker data and does not require
Oxylabs or the cloud feeder in the normal path.

They can remain archived, but they are not required by V2.17 and should not be
treated as the active deployment contract.

## Expected GitHub build

After committing the three root replacements, the build should begin with:

```text
Step 1/... : FROM python:3.11
...
COPY requirements.txt .
...
CMD ["python", "train_job.py"]
```

It should NOT contain:

```text
COPY requirements_weekly_uploader.txt .
```

After deployment, the NCAAF Heavy Research log should contain:

```text
[NCAAF-RV217-DEPLOY-PREFLIGHT] PASS
```
