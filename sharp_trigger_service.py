from google.cloud import tasks_v2
import os, json
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException

app = FastAPI()

PROJECT_ID = os.environ.get("GCP_PROJECT") or os.environ.get("PROJECT_ID", "sharplogger")
QUEUE_ID   = os.environ.get("TASK_QUEUE_ID", "sharp-detection-queue")
LOCATION   = os.environ.get("TASK_QUEUE_LOCATION", "us-east4")
TARGET_URL = os.environ.get("TASK_TARGET_URL", "https://sharp-detection-trigger-723770381669.us-east4.run.app/tasks/sharp-detection")
SA_EMAIL   = os.environ.get("TASK_INVOKER_SA", "sharplogger@appspot.gserviceaccount.com")
AUDIENCE   = os.environ.get("TASK_AUDIENCE", "https://sharp-detection-trigger-723770381669.us-east4.run.app")

@app.api_route("/run-sharp-detection", methods=["GET", "POST"])
def run_sharp_detc(_: Request):
    try:
        client = tasks_v2.CloudTasksClient()
        parent = client.queue_path(PROJECT_ID, LOCATION, QUEUE_ID)

        payload = {"trigger": "scheduler", "requested_at": datetime.utcnow().isoformat() + "Z"}
        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": TARGET_URL,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(payload).encode("utf-8"),
                "oidc_token": {
                    "service_account_email": SA_EMAIL,
                    "audience": AUDIENCE,
                },
            }
        }
        created = client.create_task(request={"parent": parent, "task": task})
        print(f"[ENQUEUE] Task created: {created.name}")  # <-- shows up in Cloud Run logs
        return {"status": "queued", "task": created.name}
    except Exception as e:
        import logging
        logging.exception("Failed to enqueue task")
        return {"status": "error", "message": f"Failed to enqueue task: {e}"}

@app.api_route("/tasks/sharp-detection", methods=["POST", "GET"])  # keep GET temporarily for debugging
async def sharp_detection_worker(request: Request):
    if request.method == "GET":
        return {"status": "ok", "note": "Worker expects POST; received GET."}
    try:
        _ = await request.body()  # optional: force-read
        from detect_utils import detect_and_save_all_sports
        detect_and_save_all_sports()
        return {"status": "success", "message": "Sharp detection completed ✅"}
    except Exception as e:
        import logging
        logging.exception("❌ Error in sharp_detection_worker")
        raise HTTPException(status_code=500, detail=str(e))
