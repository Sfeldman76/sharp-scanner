from fastapi import FastAPI
from detect_utils import detect_and_save_all_sports

app = FastAPI()

@app.get("/run-sharp-detection")
def trigger_run():
    try:
        detect_and_save_all_sports()
        return {"status": "success", "message": "Sharp detection completed"}
    except Exception as e:
        import logging
        logging.exception("❌ Sharp detection failed.")
        return {"status": "error", "message": str(e)}
