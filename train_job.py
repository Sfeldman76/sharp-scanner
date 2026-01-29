# train_job.py
import os
from google.cloud import storage
from progress import ProgressWriter

def main():
    # Provided via env vars when you execute the job
    progress_uri = os.environ["PROGRESS_URI"]  # e.g. gs://sharp-models/progress/EXEC_ID.jsonl
    sport = os.environ.get("SPORT", "NBA")
    market = os.environ.get("MARKET", "All")

    gcs = storage.Client()
    pw = ProgressWriter(progress_uri, gcs)

    pw.emit("start", f"Training start sport={sport} market={market}", pct=0.0)

    # Example: timing model
    pw.emit("timing", "Training timing opportunity model...", pct=0.05)
    train_timing_opportunity_model(...)

    mkts = ("h2h", "spreads", "totals") if market == "All" else (market,)
    for i, mkt in enumerate(mkts, start=1):
        pw.emit("train", f"Training sharp model market={mkt}", pct=0.10 + 0.80*(i-1)/max(1,len(mkts)))
        train_with_champion_wrapper(..., market=mkt)

    pw.emit("done", "Training complete ✅", pct=1.0)

if __name__ == "__main__":
    main()
