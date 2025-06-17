from fastapi import FastAPI, Request

app = FastAPI()

@app.api_route("/run-sharp-detection", methods=["GET", "POST"])
def run_sharp_detection(request: Request):
    try:
        from detect_utils import detect_and_save_all_sports
        detect_and_save_all_sports()
        return {"status": "success", "message": "Sharp detection completed ✅"}
    except Exception as e:
        import logging
        logging.exception("❌ Sharp detection failed.")
        return {"status": "error", "message": str(e)}
