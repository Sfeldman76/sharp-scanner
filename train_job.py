# train_job.py
import os
import sys
import uuid
import traceback
import warnings
import numpy as np

# Silence noisy numpy warnings in training jobs

# --- HEADLESS STREAMLIT SHIM (must be installed BEFORE importing sharp_line_dashboard) ---
import os
import warnings
import numpy as np
import logging

HEADLESS = os.getenv("HEADLESS", "0") == "1"

if HEADLESS:
    # Silence noisy numpy warnings in training jobs
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    np.seterr(all="ignore")
    logging.getLogger("numpy").setLevel(logging.ERROR)
from types import SimpleNamespace

# --- HEADLESS STREAMLIT SHIM (install BEFORE importing sharp_line_dashboard) ---
import os, sys, types

import os, sys, types

def install_streamlit_shim(log_func=print):
    import types

    class _Ctx:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False

        # common status/spinner methods
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

    class _Null:
        """Absorb any chained streamlit calls: st.x.y().z ..."""
        def __init__(self, prefix="st"):
            self._prefix = prefix

        def __call__(self, *a, **k):
            return None

        def __getattr__(self, name):
            # context manager style APIs
            if name in ("spinner", "status", "expander", "container", "form"):
                return lambda *a, **k: _Ctx()
            if name == "empty":
                return lambda *a, **k: _Null(prefix=f"{self._prefix}.empty")
            if name == "tabs":
                return lambda labels, **k: [_Null(prefix=f"{self._prefix}.tabs[{i}]") for i in range(len(labels or []))]
            if name == "columns":
                return lambda n, **k: [_Null(prefix=f"{self._prefix}.columns[{i}]") for i in range(int(n or 0))]

            return _Null(prefix=f"{self._prefix}.{name}")

        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False

    # decorator shim: supports @st.cache_data and @st.cache_data(...)
    def _decorator(fn=None, **kwargs):
        if callable(fn):
            return fn
        def wrap(f):
            return f
        return wrap

    # create a module-like object (important: some libs expect a module)
    st = types.ModuleType("streamlit")

    # common outputs
    st.write = lambda *a, **k: None
    st.markdown = lambda *a, **k: None
    st.text = lambda *a, **k: None
    st.caption = lambda *a, **k: None
    st.code = lambda *a, **k: None
    st.json = lambda *a, **k: None
    st.dataframe = lambda *a, **k: None
    st.table = lambda *a, **k: None
    st.info = lambda *a, **k: None
    st.warning = lambda *a, **k: None
    st.error = lambda *a, **k: None
    st.success = lambda *a, **k: None
    st.title = lambda *a, **k: None
    st.header = lambda *a, **k: None
    st.subheader = lambda *a, **k: None
    st.set_page_config = lambda *a, **k: None

    # layout
    st.tabs = lambda labels, **k: [_Null(prefix=f"st.tabs[{i}]") for i in range(len(labels or []))]
    st.columns = lambda n, **k: [_Null(prefix=f"st.columns[{i}]") for i in range(int(n or 0))]
    st.container = lambda **k: _Ctx()
    st.expander = lambda *a, **k: _Ctx()
    st.form = lambda *a, **k: _Ctx()
    st.empty = lambda: _Null(prefix="st.empty")

    # status/spinner
    st.status = lambda *a, **k: _Ctx()
    st.spinner = lambda *a, **k: _Ctx()

    # widgets (safe defaults)
    st.button = lambda *a, **k: False
    st.checkbox = lambda *a, **k: k.get("value", False)
    st.selectbox = lambda *a, **k: (k.get("options") or [None])[0]
    st.radio = lambda *a, **k: (k.get("options") or [None])[0]
    st.slider = lambda *a, **k: k.get("value", 0)
    st.text_input = lambda *a, **k: k.get("value", "")
    st.number_input = lambda *a, **k: k.get("value", 0)
    st.date_input = lambda *a, **k: k.get("value", None)
    st.metric = lambda *a, **k: None
    st.plotly_chart = lambda *a, **k: None

    # caching
    st.cache_data = _decorator
    st.cache_resource = _decorator

    # session state + sidebar (IMPORTANT: sidebar must NOT call back into st init)
    st.session_state = {}
    st.sidebar = _Null(prefix="st.sidebar")

    # catch-all: if code calls st.something_unexpected, return a _Null chain
    def __getattr__(name):
        return getattr(st, name, _Null(prefix=f"st.{name}"))
    st.__getattr__ = __getattr__  # type: ignore[attr-defined]

    sys.modules["streamlit"] = st
    return st

HEADLESS = os.getenv("HEADLESS", "0") == "1"
if HEADLESS:
    install_streamlit_shim()

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
