import os, json, logging
from datetime import datetime, timezone

from fastapi import FastAPI, Request, HTTPException
from google.cloud import tasks_v2
from google.api_core.exceptions import AlreadyExists
# ✅ Configure logging once, right after imports

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

app = FastAPI()

PROJECT_ID = os.environ.get("GCP_PROJECT") or os.environ.get("PROJECT_ID", "sharplogger")
QUEUE_ID   = os.environ.get("TASK_QUEUE_ID", "sharp-detection-queue")
LOCATION   = os.environ.get("TASK_QUEUE_LOCATION", "us-east4")
TARGET_URL = os.environ.get("TASK_TARGET_URL", "https://sharp-detection-trigger-723770381669.us-east4.run.app/tasks/sharp-detection")
SA_EMAIL   = os.environ.get("TASK_INVOKER_SA", "sharplogger@appspot.gserviceaccount.com")
AUDIENCE   = os.environ.get("TASK_AUDIENCE", "https://sharp-detection-trigger-723770381669.us-east4.run.app")

def enqueue_detection_task(client: tasks_v2.CloudTasksClient, parent: str, base_task: dict) -> str:
    # Idempotency: one task per 10-minute window (customize as you like)
    window = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")[:-1]  # e.g. 20250819T1623 -> 20250819T162
    task_id = f"sharp-detection-{window}"
    base_task["name"] = client.task_path(PROJECT_ID, LOCATION, QUEUE_ID, task_id)
    try:
        created = client.create_task(request={"parent": parent, "task": base_task})
        print(f"[ENQUEUE] Task created: {created.name}")
        return created.name
    except AlreadyExists:
        print(f"[ENQUEUE] Task {task_id} already exists — skipping duplicate enqueue")
        return f"existing:{task_id}"

@app.api_route("/run-sharp-detection", methods=["GET", "POST"])
def run_sharp_detection(_: Request):
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

        # ✅ use the idempotent helper
        task_name = enqueue_detection_task(client, parent, task)
        return {"status": "queued", "task": task_name}
    except Exception as e:
        logging.exception("Failed to enqueue task")
        return {"status": "error", "message": f"Failed to enqueue task: {e}"}

@app.post("/tasks/sharp-detection")
async def sharp_detection_worker(request: Request):
    retry = request.headers.get("X-Cloud-Tasks-TaskRetryCount", "0")
    tname = request.headers.get("X-Cloud-Tasks-TaskName", "")
    logging.info("[WORKER] task=%s retry=%s", tname, retry)

    try:
        from detect_utils import detect_and_save_all_sports
        detect_and_save_all_sports()
        return {"status": "success"}
    except Exception as e:
        logging.exception("❌ Error in sharp_detection_worker")
        # 500 => Cloud Tasks will retry per queue policy
        raise HTTPException(status_code=500, detail=str(e))

