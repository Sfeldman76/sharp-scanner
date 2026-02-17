import os, sys
from types import SimpleNamespace
import os, sys, uuid, traceback
from types import SimpleNamespace

class _Null:
    def __init__(self, log_func=print, prefix=""):
        self._log_func = log_func
        self._prefix = prefix

    def __call__(self, *args, **kwargs):
        # For calls like st.markdown("x")
        return None

    def __getattr__(self, name):
        # For chained things like st.sidebar.markdown(...)
        if name in ("__enter__", "__exit__"):
            return getattr(self, name)
        # Return another _Null so chains keep working
        return _Null(self._log_func, prefix=f"{self._prefix}.{name}" if self._prefix else name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

def _decorator(func=None, **kwargs):
    # Supports @st.cache_data and @st.cache_data(...)
    if callable(func):
        return func
    def wrap(fn):
        return fn
    return wrap

def install_streamlit_shim(log_func=print):
    st = _Null(log_func, "st")

    # Provide real decorator attributes (otherwise they become _Null and break decoration)
    st.cache_data = _decorator
    st.cache_resource = _decorator

    # Provide a sidebar object (also Null)
    st.sidebar = _Null(log_func, "st.sidebar")

    # Common structural helpers Streamlit returns lists for
    st.columns = lambda n, **k: [_Null(log_func, f"st.columns[{i}]") for i in range(n)]
    st.tabs = lambda labels, **k: [_Null(log_func, f"st.tabs[{i}]") for i in range(len(labels or []))]
    st.container = lambda **k: _Null(log_func, "st.container")
    st.expander = lambda *a, **k: _Null(log_func, "st.expander")
    st.form = lambda *a, **k: _Null(log_func, "st.form")

    # Widgets: return defaults so code paths don’t explode
    st.button = lambda *a, **k: False
    st.checkbox = lambda *a, **k: k.get("value", False)
    st.selectbox = lambda *a, **k: (k.get("options") or [None])[0]
    st.radio = lambda *a, **k: (k.get("options") or [None])[0]
    st.slider = lambda *a, **k: k.get("value", 0)
    st.text_input = lambda *a, **k: k.get("value", "")
    st.number_input = lambda *a, **k: k.get("value", 0)

    # Session state
    st.session_state = {}

    sys.modules["streamlit"] = st

if os.getenv("HEADLESS", "1") == "1":
    install_streamlit_shim(print) = st

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
