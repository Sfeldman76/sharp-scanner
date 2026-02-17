import os
from google.cloud import storage
from progress import ProgressWriter

from train_sharp_model_from_bq_extracted import (
    train_sharp_model_for_market,
    train_timing_model_for_market,
)

def main():
    progress_uri = os.environ["PROGRESS_URI"]
    sport = os.environ.get("SPORT", "NBA")
    market = os.environ.get("MARKET", "All")
    bucket = os.environ.get("MODEL_BUCKET", "sharp-models")

    gcs = storage.Client()
    pw = ProgressWriter(progress_uri, gcs)

    pw.emit("start", f"Training start sport={sport} market={market}", pct=0.0)

    if market == "All":
        pw.emit("timing", "Training timing model...", pct=0.05)
        train_timing_model_for_market(sport)

        mkts = ("h2h", "spreads", "totals")
    else:
        mkts = (market,)

    for i, mkt in enumerate(mkts, start=1):
        pw.emit("train", f"Training sharp model {mkt}", pct=0.10 + 0.80*(i-1)/len(mkts))
        train_sharp_model_for_market(sport, mkt, bucket)

    pw.emit("done", "Training complete", pct=1.0)

if __name__ == "__main__":
    main()
