from fastapi import FastAPI
import uvicorn
import os
from sharp_line_dashboard import detect_and_save_all_sports  # assumes you extract your detection logic here

app = FastAPI()

@app.post("/run-sharp-detection")
def run_sharp_detection():
    detect_and_save_all_sports()
    return {"status": "Sharp detection triggered successfully"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("sharp_trigger_service:app", host="0.0.0.0", port=port)
