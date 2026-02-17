# train_job.py
import os
import sys
import uuid
import traceback


# --- HEADLESS STREAMLIT SHIM (must be installed BEFORE importing sharp_line_dashboard) ---
import os, sys
from types import SimpleNamespace

class _Ctx:
    """A no-op context manager that also swallows common Streamlit 'status' methods."""
    def __init__(self, log_func=print):
        self._log = log_func

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    # Streamlit status-like API
    def write(self, *a, **k): return None
    def markdown(self, *a, **k): return None
    def code(self, *a, **k): return None
    def json(self, *a, **k): return None
    def dataframe(self, *a, **k): return None
    def table(self, *a, **k): return None
    def success(self, *a, **k): return None
    def info(self, *a, **k): return None
    def warning(self, *a, **k): return None
    def error(self, *a, **k): return None
    def update(self, *a, **k): return None

class _NullStreamlit:
    """
    Object that absorbs any Streamlit calls in headless mode.
    - Any function call returns None (or a _Ctx for context manager functions).
    - Any attribute access returns another callable absorber.
    """
    def __init__(self, log_func=print):
        self._log = log_func
        self.session_state = {}  # behave like Streamlit session_state
        self.sidebar = self      # sidebar.* should also work

        # decorator shims
        self.cache_data = self._decorator
        self.cache_resource = self._decorator

    def _decorator(self, func=None, **kwargs):
        # Supports both @st.cache_data and @st.cache_data(...)
        if callable(func):
            return func
        def wrap(fn):
            return fn
        return wrap

    def __getattr__(self, name):
        # Context manager style APIs
        if name in ("spinner", "expander", "container", "form"):
            return lambda *a, **k: _Ctx(self._log)

        # st.status() returns an object with .write/.update etc.
        if name == "status":
            return lambda *a, **k: _Ctx(self._log)

        # Most layout builders return context-ish objects
        if name == "tabs":
            return lambda labels, **k: [_Ctx(self._log) for _ in range(len(labels or []))]
        if name == "columns":
            return lambda n, **k: [_Ctx(self._log) for _ in range(int(n or 0))]

        # Any other attribute becomes a callable no-op
        return lambda *a, **k: None

    # allow st(...) calls (rare)
    def __call__(self, *a, **k):
        return None

def install_streamlit_shim(log_func=print):
    st = _NullStreamlit(log_func=log_func)

    # Make "import streamlit as st" work
    shim_module = SimpleNamespace(**{"__dict__": {}, **st.__dict__})

    # But easiest: set sys.modules["streamlit"] to an object with attributes
    sys.modules["streamlit"] = st
    return st


# Install shim BEFORE importing anything that imports streamlit (sharp_line_dashboard)
HEADLESS = os.getenv("HEADLESS", "0") == "1"
if HEADLESS:
    install_streamlit_shim(print)

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
