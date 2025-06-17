from fastapi import FastAPI
from detect_utils import detect_and_save_all_sports

app = FastAPI()

@app.get("/")
def health_check():
    """Health check endpoint for Cloud Run readiness probe."""
    return {"status": "ok"}

@app.post("/run-sharp-detection")
def run_detection():
    """
    Main endpoint to trigger sharp detection job.
    Should be invoked via Cloud Scheduler (HTTP POST).
    """
    try:
        detect_and_save_all_sports()
        return {"status": "✅ Sharp detection completed"}
    except Exception as e:
        return {"status": "❌ Sharp detection failed", "error": str(e)}
