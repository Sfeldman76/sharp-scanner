# train_job.py
import os
import sys
import uuid
import traceback
import warnings
import numpy as np

# Silence noisy numpy warnings in training jobs
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")

# --- HEADLESS STREAMLIT SHIM (must be installed BEFORE importing sharp_line_dashboard) ---
import os, sys
from types import SimpleNamespace

# --- HEADLESS STREAMLIT SHIM (install BEFORE importing sharp_line_dashboard) ---
import os, sys, types

class _Ctx:
    """No-op context manager that swallows common Streamlit status/spinner APIs."""
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False

    # status-like API
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

def _decorator(func=None, **kwargs):
    # Supports @st.cache_data and @st.cache_data(...)
    if callable(func):
        return func
    def wrap(fn):
        return fn
    return wrap

class _NullStreamlit:
    """
    Absorbs any Streamlit calls in headless mode.
    - Unknown attrs return callables that don't crash.
    - Context-managers return _Ctx.
    - Widgets return safe defaults.
    """
    def __init__(self):
        self.session_state = {}
        self.cache_data = _decorator
        self.cache_resource = _decorator

        # sidebar is its own proxy, but shares session_state
        self.sidebar = _NullSidebar(self.session_state)

    # --- Explicit common APIs (no-op) ---
    def set_page_config(self, *a, **k): return None
    def stop(self): return None
    def rerun(self): return None

    def write(self, *a, **k): return None
    def markdown(self, *a, **k): return None
    def code(self, *a, **k): return None
    def json(self, *a, **k): return None
    def dataframe(self, *a, **k): return None
    def table(self, *a, **k): return None
    def metric(self, *a, **k): return None
    def plotly_chart(self, *a, **k): return None
    def pyplot(self, *a, **k): return None
    def altair_chart(self, *a, **k): return None
    def progress(self, *a, **k): return None

    def title(self, *a, **k): return None
    def header(self, *a, **k): return None
    def subheader(self, *a, **k): return None
    def info(self, *a, **k): return None
    def warning(self, *a, **k): return None
    def error(self, *a, **k): return None
    def success(self, *a, **k): return None
    def caption(self, *a, **k): return None
    def text(self, *a, **k): return None

    # --- Context managers ---
    def spinner(self, *a, **k): return _Ctx()
    def status(self, *a, **k): return _Ctx()
    def container(self, *a, **k): return _Ctx()
    def expander(self, *a, **k): return _Ctx()
    def form(self, *a, **k): return _Ctx()

    # --- Layout builders ---
    def columns(self, n, **k):
        try:
            n = int(n)
        except Exception:
            n = 0
        return [_Ctx() for _ in range(n)]

    def tabs(self, labels, **k):
        labels = labels or []
        return [_Ctx() for _ in range(len(labels))]

    def empty(self):
        return _Ctx()

    # --- Widgets: safe defaults ---
    def button(self, *a, **k): return False
    def checkbox(self, *a, **k): return k.get("value", False)

    def selectbox(self, *a, **k):
        opts = k.get("options") or (a[1] if len(a) > 1 else None) or []
        return opts[0] if opts else None

    def radio(self, *a, **k):
        opts = k.get("options") or (a[1] if len(a) > 1 else None) or []
        return opts[0] if opts else None

    def slider(self, *a, **k): return k.get("value", 0)
    def text_input(self, *a, **k): return k.get("value", "")
    def number_input(self, *a, **k): return k.get("value", 0)
    def date_input(self, *a, **k): return k.get("value", None)

    def __getattr__(self, name):
        # Anything else: return a callable no-op so st.<new_api>() never crashes
        def _noop(*a, **k): return None
        return _noop

class _NullSidebar(_NullStreamlit):
    def __init__(self, shared_session_state):
        super().__init__()
        self.session_state = shared_session_state  # share same dict

def install_streamlit_shim():
    st_obj = _NullStreamlit()

    # Install as a real module
    mod = types.ModuleType("streamlit")
    for attr in dir(st_obj):
        if not attr.startswith("__"):
            setattr(mod, attr, getattr(st_obj, attr))
    # Also expose session_state + sidebar (important)
    mod.session_state = st_obj.session_state
    mod.sidebar = st_obj.sidebar

    sys.modules["streamlit"] = mod
    return mod

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
