# train_job.py
import os, sys, uuid, traceback
from types import SimpleNamespace



def _install_streamlit_shim(log_func=print):
    def _log(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        try:
            log_func(msg)
        except Exception:
            print(msg, flush=True)

    def _noop(*a, **k): return None

    def _decorator(func=None, **dkwargs):
        if callable(func):
            return func
        def wrap(fn):
            return fn
        return wrap

    st = SimpleNamespace(
        set_page_config=_noop,
        title=_log,
        subheader=_log,
        header=_log,
        write=_log,
        info=_log,
        warning=_log,
        error=_log,
        success=_log,
        markdown=_log,
        json=_log,
        text=_log,
        caption=_log,

        cache_data=_decorator,
        cache_resource=_decorator,

        session_state={},

        # common things dashboards reference at import-time
        sidebar=SimpleNamespace(
            write=_log, info=_log, warning=_log, error=_log,
            selectbox=lambda *a, **k: k.get("index", 0),
            radio=lambda *a, **k: k.get("index", 0),
            checkbox=lambda *a, **k: k.get("value", False),
        ),

        columns=lambda n, **k: [SimpleNamespace() for _ in range(n)],
        progress=_noop,
        status=lambda *a, **k: SimpleNamespace(__enter__=lambda s: s, __exit__=lambda *x: None, write=_log),

        # prevent accidental hard-fails
        button=lambda *a, **k: False,
        selectbox=lambda *a, **k: (k.get("options") or [None])[0],
        radio=lambda *a, **k: (k.get("options") or [None])[0],
        checkbox=lambda *a, **k: k.get("value", False),
    )

    sys.modules["streamlit"] = st

if os.getenv("HEADLESS", "1") == "1":
    _install_streamlit_shim(print)
# Install shim BEFORE any other imports that might import streamlit

from google.cloud import storage
from progress import ProgressWriter

from train_sharp_model_from_bq_extracted import (
    train_sharp_model_for_market,
    train_timing_model_for_market,
)

def main():
    run_id = os.environ.get("TRAIN_RUN_ID") or str(uuid.uuid4())[:8]
    sport = os.environ.get("SPORT", "NBA")
    market = os.environ.get("MARKET", "All")
    bucket = os.environ.get("MODEL_BUCKET", "sharp-models")

    progress_uri = os.environ.get("PROGRESS_URI")
    if not progress_uri:
        progress_uri = f"gs://{bucket}/train-progress/{sport}/{market}/{run_id}.json"
        os.environ["PROGRESS_URI"] = progress_uri

    gcs = storage.Client()
    pw = ProgressWriter(progress_uri, gcs)

    log_func = lambda msg: pw.emit("log", str(msg))
    pw.emit("start", f"Training start run_id={run_id} sport={sport} market={market}", pct=0.0)

    try:
        if market == "All":
            pw.emit("timing", f"[{sport}] Training timing model...", pct=0.05)
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

            try:
                train_sharp_model_for_market(sport=sport, market=mkt, bucket_name=bucket, log_func=log_func)
            except TypeError:
                train_sharp_model_for_market(sport=sport, market=mkt, bucket_name=bucket)

        pw.emit("done", "Training complete ✅", pct=1.0)

    except Exception as e:
        pw.emit("error", f"{e}\n{traceback.format_exc()}", pct=1.0)
        raise

if __name__ == "__main__":
    main()
