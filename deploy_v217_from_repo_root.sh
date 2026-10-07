#!/usr/bin/env bash
set -euo pipefail

PROJECT="sharplogger"
REGION="us-east4"
JOB="sharp-train-job"

printf '\n=== NCAAF V2.17 TRAIN JOB DEPLOY ===\n'
gcloud config set project "$PROJECT"

# This script MUST run from the existing sharp-train-job repository root.
for f in requirements.txt Dockerfile.job train_job.py ncaaf_research_v2.py; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f in $(pwd)"
    echo "Run this from the existing sharp-train-job repository root, not the NCAAF uploader directory."
    exit 1
  fi
done

if [[ -f requirements_weekly_uploader.txt || -f ncaaf_weekly_uploader.py ]]; then
  echo "ERROR: this looks like the NCAAF weekly uploader directory."
  echo "Do not deploy V2.17 from here."
  exit 1
fi

if ! grep -q 'ncaaf-research-v2.17-sparse-pt-manual-gcs-20261007' ncaaf_research_v2.py; then
  echo "ERROR: ncaaf_research_v2.py is not the V2.17 file."
  exit 1
fi
if ! grep -q 'NCAAF-RV217-DEPLOY-PREFLIGHT' train_job.py; then
  echo "ERROR: train_job.py is not the V2.17 file."
  exit 1
fi

python -m py_compile train_job.py ncaaf_research_v2.py

echo "Preflight PASS: correct V2.17 files and Dockerfile.job build context."

PREVIOUS_IMAGE="$(gcloud run jobs describe "$JOB" \
  --project="$PROJECT" \
  --region="$REGION" \
  --format='value(spec.template.spec.template.spec.containers[0].image)' 2>/dev/null || true)"

if [[ -z "$PREVIOUS_IMAGE" ]]; then
  PREVIOUS_IMAGE="$(gcloud run jobs describe "$JOB" \
    --project="$PROJECT" \
    --region="$REGION" \
    --format='value(template.template.containers[0].image)' 2>/dev/null || true)"
fi

echo "Current job image: ${PREVIOUS_IMAGE:-<could not resolve>}"
if [[ -n "$PREVIOUS_IMAGE" ]]; then
  printf '%s\n' "$PREVIOUS_IMAGE" > v217_previous_image.txt
fi

TAG="v2-17-$(date -u +%Y%m%d-%H%M%S)"
IMAGE="gcr.io/${PROJECT}/sharp-train-job:${TAG}"

echo "New image: $IMAGE"
echo
echo "=== CLOUD BUILD USING Dockerfile.job ==="

gcloud builds submit . \
  --project="$PROJECT" \
  --config=cloudbuild_v217.yaml \
  --substitutions="_IMAGE=$IMAGE"

echo
echo "=== UPDATE CLOUD RUN JOB IMAGE ONLY ==="
gcloud run jobs update "$JOB" \
  --project="$PROJECT" \
  --region="$REGION" \
  --image="$IMAGE"

echo
echo "=== VERIFY ==="
gcloud run jobs describe "$JOB" \
  --project="$PROJECT" \
  --region="$REGION" \
  --format='yaml(metadata.name,spec.template.spec.template.spec.containers[0].image,spec.template.spec.template.spec.serviceAccountName,spec.template.spec.template.spec.timeoutSeconds)'

echo
echo "DEPLOY COMPLETE."
echo "Do not run Production Publish. In the app, run: NCAAF Research — Heavy Challenger Search"
echo "Expected first marker: [NCAAF-RV217-DEPLOY-PREFLIGHT] PASS"
