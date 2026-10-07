# NCAAF V2.17 — deploy-ready overlay

This package is for the **sharp-train-job repository**, not the NCAAF weekly uploader.

## Replace in the existing training repository

Replace:
- `ncaaf_research_v2.py`
- `train_job.py`

Use the included `Dockerfile.job` for the training image. It is the user's existing training-job Dockerfile:

```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "train_job.py"]
```

Do not use the uploader Dockerfile containing `requirements_weekly_uploader.txt`.

## Cloud Shell deployment

From the existing **sharp-train-job repo root**, place/copy these package files into the repo root and run:

```bash
chmod +x deploy_v217_from_repo_root.sh
bash deploy_v217_from_repo_root.sh
```

The script deliberately builds with `Dockerfile.job`, even if another file named `Dockerfile` exists. It updates **only the image** on the existing `sharp-train-job`, preserving the job's existing environment, service account, resources, timeout, and other configuration.

It does not execute the training job automatically.

After deployment, use the existing UI/dropdown and run:

`NCAAF Research — Heavy Challenger Search`

Expected first V2.17 marker:

`[NCAAF-RV217-DEPLOY-PREFLIGHT] PASS`
