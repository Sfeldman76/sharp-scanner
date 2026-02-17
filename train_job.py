# train_job.py
import os
import sys
import uuid
import traceback
import warnings
import logging

HEADLESS = os.getenv("HEADLESS", "0") == "1"

# -----------------------------------------------------------------------------
# Headless warning / numeric noise control (ONLY in Cloud Run Jobs / headless)
# -----------------------------------------------------------------------------
if HEADLESS:
    # Keep this narrow so we don't hide real numerical bugs
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Degrees of freedom <= 0", category=RuntimeWarning)

    # Optional: if you still get spam from other libs, you can broaden *slightly*:
    # warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"numpy\.lib\.nanfunctions")

    # Numpy doesn't log via logging; this is mostly harmless, but doesn't hurt.
    logging.getLogger("numpy").setLevel(logging.ERROR)

# -----------------------------------------------------------------------------
# HEADLESS STREAMLIT SHIM (MUST be installed BEFORE importing sharp_line_dashboard)
# -----------------------------------------------------------------------------
def install_streamlit_shim(log_func=print):
    import types

    class _Ctx:
        """No-op context manager for st.status/st.spinner/st.expander/etc."""
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False

        # Common "status" methods
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
            # Return another absorber for any attribute access
            return _Null(prefix=f"{self._prefix}.{name}")

        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False

    class _Progress:
        """
        Streamlit progress bar object.
        Streamlit supports:
          pb = st.progress(0)
          pb.progress(50)
          pb.empty()
        """
        def progress(self, *a, **k): return None
        def update(self, *a, **k): return None
        def empty(self): return None

    # decorator shim: supports @st.cache_data and @st.cache_data(...)
    def _decorator(fn=None, **kwargs):
        if callable(fn):
            return fn
        def wrap(f):
            return f
        return wrap

    # Create a real module object (some libs check types.ModuleType)
    st = types.ModuleType("streamlit")

    # Common outputs
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

    # Progress (IMPORTANT: avoid crashes at st.progress(0))
    st.progress = lambda *a, **k: _Progress()

    # Layout
    st.tabs = lambda labels, **k: [_Null(prefix=f"st.tabs[{i}]") for i in range(len(labels or []))]
    st.columns = lambda n, **k: [_Null(prefix=f"st.columns[{i}]") for i in range(int(n or 0))]
    st.container = lambda **k: _Ctx()
    st.expander = lambda *a, **k: _Ctx()
    st.form = lambda *a, **k: _Ctx()
    st.empty = lambda: _Null(prefix="st.empty")

    # Status / spinner
    st.status = lambda *a, **k: _Ctx()
    st.spinner = lambda *a, **k: _Ctx()

    # Widgets (safe defaults)
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

    # Caching
    st.cache_data = _decorator
    st.cache_resource = _decorator

    # Session state + sidebar
    st.session_state = {}
    st.sidebar = _Null(prefix="st.sidebar")

    # SAFE module-level __getattr__ fallback (DO NOT call getattr(st, ...) here -> recursion)
    def _module_getattr(name):
        d = st.__dict__
        if name in d:
            return d[name]
        return _Null(prefix=f"st.{name}")

    st.__getattr__ = _module_getattr  # type: ignore[attr-defined]

    # Make `import streamlit as st` resolve to this shim
    sys.modules["streamlit"] = st
    return st

if HEADLESS:
    install_streamlit_shim()

# -----------------------------------------------------------------------------
# Normal imports (after shim)
# -----------------------------------------------------------------------------
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
