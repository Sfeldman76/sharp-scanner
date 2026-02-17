# train_job.py
import os
import sys
import uuid
import traceback

class _Null:
    def __init__(self, log_func=print, prefix=""):
        self._log_func = log_func
        self._prefix = prefix

    def __call__(self, *args, **kwargs):
        # Allows calls like st.markdown("x") without crashing
        return None

    def __getattr__(self, name):
        # Allows chained calls like st.sidebar.markdown(...)
        return _Null(self._log_func, prefix=f"{self._prefix}.{name}" if self._prefix else name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

def _decorator(func=None, **kwargs):
    # Supports both:
    #   @st.cache_data
    # and:
    #   @st.cache_data(ttl=3600)
    if callable(func):
        return func
    def wrap(fn):
        return fn
    return wrap



import sys
from types import SimpleNamespace

def install_streamlit_shim(log_func=print):
    def _log(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        try:
            log_func(msg)
        except Exception:
            print(msg, flush=True)

    def _noop(*a, **k):
        return None

    # Supports both @st.cache_data and @st.cache_data(...)
    def _decorator(fn=None, **kwargs):
        if callable(fn):
            return fn
        def wrap(f):
            return f
        return wrap

    class _Ctx:
        """Context manager returned by st.status / st.spinner in headless mode."""
        def __init__(self, label=""):
            self.label = label

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, *a, **k): _log(*a)
        def markdown(self, *a, **k): _log(*a)
        def update(self, *a, **k): return None
        def success(self, *a, **k): _log(*a)
        def warning(self, *a, **k): _log(*a)
        def error(self, *a, **k): _log(*a)

    class _Null:
        """Null object to absorb arbitrary streamlit calls/chains."""
        def __init__(self, prefix="st"):
            self._prefix = prefix

        def __call__(self, *args, **kwargs):
            # swallow calls like st.markdown("x")
            return None

        def __getattr__(self, name):
            # allow st.sidebar.markdown(...) chains
            return _Null(prefix=f"{self._prefix}.{name}")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    st = SimpleNamespace()

    # Page + common outputs
    st.set_page_config = _noop
    st.title = _log
    st.header = _log
    st.subheader = _log
    st.write = _log
    st.info = _log
    st.warning = _log
    st.error = _log
    st.success = _log
    st.markdown = _log
    st.text = _log
    st.caption = _log
    st.json = _log
    st.code = _log

    # Sidebar
    st.sidebar = _Null(prefix="st.sidebar")
    st.sidebar.markdown = _log
    st.sidebar.write = _log
    st.sidebar.info = _log
    st.sidebar.warning = _log
    st.sidebar.error = _log
    st.sidebar.success = _log

    # Layout helpers
    st.columns = lambda n, **k: [_Null(prefix=f"st.columns[{i}]") for i in range(n)]
    st.tabs = lambda labels, **k: [_Null(prefix=f"st.tabs[{i}]") for i in range(len(labels or []))]
    st.container = lambda **k: _Null(prefix="st.container")
    st.expander = lambda *a, **k: _Null(prefix="st.expander")
    st.form = lambda *a, **k: _Null(prefix="st.form")
    st.empty = lambda: _Null(prefix="st.empty")

    # Widgets (return defaults)
    st.button = lambda *a, **k: False
    st.checkbox = lambda *a, **k: k.get("value", False)
    st.selectbox = lambda *a, **k: (k.get("options") or [None])[0]
    st.radio = lambda *a, **k: (k.get("options") or [None])[0]
    st.slider = lambda *a, **k: k.get("value", 0)
    st.text_input = lambda *a, **k: k.get("value", "")
    st.number_input = lambda *a, **k: k.get("value", 0)
    st.date_input = lambda *a, **k: k.get("value", None)

    # Progress/status/spinner
    st.progress = _noop
    st.status = lambda *a, **k: _Ctx(label=str(a[0]) if a else "")
    st.spinner = lambda *a, **k: _Ctx(label=str(a[0]) if a else "")

    # Cache decorators
    st.cache_data = _decorator
    st.cache_resource = _decorator

    # Session state
    st.session_state = {}

    sys.modules["streamlit"] = st
    return st

if os.getenv("HEADLESS", "1") == "1":
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
