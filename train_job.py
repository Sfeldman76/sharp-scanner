# train_job.py
import os
from google.cloud import storage
from progress import ProgressWriter

from train_sharp_model_from_bq_extracted import (
    train_sharp_model_for_market,      # <-- you create/rename these
    train_timing_model_for_market,     # <-- in the extracted file
)

def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

def main():
    progress_uri = _require_env("PROGRESS_URI")     # gs://.../EXEC_ID.jsonl
    sports_csv = os.environ.get("SPORTS", "NBA")    # "NBA" or "NBA,NCAAB"
    market = os.environ.get("MARKET", "All")        # All|spreads|totals|h2h

    sports = [s.strip() for s in sports_csv.split(",") if s.strip()]
    mkts = ("h2h", "spreads", "totals") if market == "All" else (market,)

    gcs = storage.Client()
    pw = ProgressWriter(progress_uri, gcs)

    pw.emit("start", f"Training start sports={sports} market={market}", pct=0.0)

    total_steps = len(sports) * (1 + len(mkts))  # timing + each market
    step = 0

    for sport in sports:
        # timing model once per sport (or per market if you prefer)
        step += 1
        pw.emit("timing", f"[{sport}] Training timing model...", pct=step/total_steps)
        train_timing_model_for_market(sport=sport)

        for mkt in mkts:
            step += 1
            pw.emit("train", f"[{sport}] Training sharp model market={mkt}...", pct=step/total_steps)
            train_sharp_model_for_market(sport=sport, market=mkt)

    pw.emit("done", "Training complete ✅", pct=1.0)

if __name__ == "__main__":
    main()
