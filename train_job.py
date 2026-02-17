# train_job.py
import os
import uuid
import traceback

from google.cloud import storage
from progress import ProgressWriter

from train_sharp_model_from_bq_extracted import (
    train_sharp_model_for_market,
    train_timing_model_for_market,
)


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def main():
    # Optional run id (never hard-fail)
    run_id = os.environ.get("TRAIN_RUN_ID") or str(uuid.uuid4())[:8]
    

    # Required for progress + UI
    progress_uri = _require_env("PROGRESS_URI")

    # Training selectors
    sport = os.environ.get("SPORT", "NBA")
    market = os.environ.get("MARKET", "All")

    # Where models get written (your training code should use this)
    bucket = os.environ.get("MODEL_BUCKET", "sharp-models")

    gcs = storage.Client()
    pw = ProgressWriter(progress_uri, gcs)

    # Make a log function compatible with training code that expects log_func(str)
    log_func = lambda msg: pw.emit("log", str(msg))

    pw.emit("start", f"Training start run_id={run_id} sport={sport} market={market}", pct=0.0)

    try:
        if market == "All":
            pw.emit("timing", f"[{sport}] Training timing model...", pct=0.05)
            # If your timing trainer supports bucket/log_func, pass them; otherwise keep simple
            try:
                train_timing_model_for_market(sport=sport, bucket_name=bucket, log_func=log_func)
            except TypeError:
                train_timing_model_for_market(sport=sport)

            mkts = ("h2h", "spreads", "totals")
        else:
            mkts = (market,)

        n = len(mkts)
        for i, mkt in enumerate(mkts, start=1):
            pct = 0.10 + 0.80 * (i - 1) / max(1, n)
            pw.emit("train", f"[{sport}] Training sharp model market={mkt}", pct=pct)

            # Try passing bucket/log_func; fallback if your wrapper signature is simpler
            try:
                train_sharp_model_for_market(sport=sport, market=mkt, bucket_name=bucket, log_func=log_func)
            except TypeError:
                train_sharp_model_for_market(sport=sport, market=mkt, bucket_name=bucket)

        pw.emit("done", "Training complete ✅", pct=1.0)

    except Exception as e:
        tb = traceback.format_exc()
        pw.emit("error", f"{e}\n{tb}", pct=1.0)
        raise  # make the job fail visibly in Cloud Run


if __name__ == "__main__":
    main()
