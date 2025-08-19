import os
import json
from datetime import datetime
from fastapi import FastAPI, Request, Response, HTTPException
from google.cloud import tasks_v2

app = FastAPI()

# ====== Config via env ======
PROJECT_ID = os.environ.get("GCP_PROJECT") or os.environ.get("PROJECT_ID", "sharplogger")
QUEUE_ID   = os.environ.get("TASK_QUEUE_ID", "sharp-detection-queue")
LOCATION   = os.environ.get("TASK_QUEUE_LOCATION", "us-east4")
# Public HTTPS URL for your worker endpoint (this same Cloud Run service):
TARGET_URL = os.environ.get("TASK_TARGET_URL", "https://sharp-detection-trigger-723770381669.us-east4.run.app/tasks/sharp-detection")
# Service account used to sign OIDC token for auth to TARGET_URL
SA_EMAIL   = os.environ.get("TASK_INVOKER_SA", "scheduler-invoker@sharplogger.iam.gserviceaccount.com")

# ====== 1) Lightweight trigger: enqueue task and return quickly ======
@app.api_route("/run-sharp-detection", methods=["GET", "POST"])
def run_sharp_detection(_: Request):
    try:
        client = tasks_v2.CloudTasksClient()
        parent = client.queue_path(PROJECT_ID, LOCATION, QUEUE_ID)

        payload = {
            "trigger": "scheduler",
            "requested_at": datetime.utcnow().isoformat() + "Z"
        }
        # Build HTTP request for the task (POST JSON to worker endpoint)
        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": TARGET_URL,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(payload).encode("utf-8"),

                # Use OIDC for authenticated call into Cloud Run
                "oidc_token": {
                    "service_account_email": SA_EMAIL,
                    # Optional: set audience to your Cloud Run URL base if needed
                    # "audience": "https://sharp-detection-trigger-723770381669.us-east4.run.app"
                },
            },
        }

        _ = client.create_task(request={"parent": parent, "task": task})
        return {"status": "queued", "message": "Sharp detection task enqueued ✅"}
    except Exception as e:
        # Do NOT run heavy work here — just report enqueue errors
        return {"status": "error", "message": f"Failed to enqueue task: {e}"}

# ====== 2) Worker endpoint: does the heavy lifting (called by Cloud Tasks) ======
@app.post("/tasks/sharp-detection")
def sharp_detection_worker(request: Request):
    try:
        # (optional) Verify caller if you want stricter checks (e.g., X-Cloud-Tasks-QueueName)
        # headers = request.headers

        from detect_utils import detect_and_save_all_sports
        detect_and_save_all_sports()  # heavy work happens here

        return {"status": "success", "message": "Sharp detection completed ✅"}
    except Exception as e:
        import logging
        logging.exception("❌ Error in sharp_detection_worker")
        # Return 5xx to make Cloud Tasks retry based on queue settings
        raise HTTPException(status_code=500, detail=str(e))
