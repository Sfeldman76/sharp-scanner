from fastapi import FastAPI
from detect_utils import detect_and_save_all_sports

app = FastAPI()

@app.post("/run-sharp-detection")
def run_detection():
    detect_and_save_all_sports()
    return {"status": "✅ Sharp detection completed"}
