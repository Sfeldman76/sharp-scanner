# train_job.py
import os
import sys
import uuid
import traceback
import warnings
import logging
import threading
import time

HEADLESS = os.getenv("HEADLESS", "0") == "1"

# -----------------------------------------------------------------------------
# Headless warning / numeric noise control (ONLY in Cloud Run Jobs / headless)
# -----------------------------------------------------------------------------
if HEADLESS:
    # Keep narrow: only suppress the specific spam you showed
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Degrees of freedom <= 0", category=RuntimeWarning)

    # Optional: reduce other noisy libs (doesn't affect numpy warnings)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# HEADLESS STREAMLIT SHIM (MUST be installed BEFORE importing sharp_line_dashboard)
# -----------------------------------------------------------------------------
def install_streamlit_shim(log_func=print):
    """
    Install/replace a 'streamlit' module shim that:
      - never recurses in __getattr__
      - converts common st.* calls into log_func(...) so you see step-by-step logs
      - supports st.status/st.spinner context patterns
      - supports st.progress(0) returning an object with .progress()/.empty()
    """
    import types

    def _log(*a, **k):
        msg = " ".join(str(x) for x in a).strip()
        if not msg:
            return
        try:
            log_func(msg)
        except Exception:
            print(msg, flush=True)

    class _Ctx:
        def __init__(self, label=""):
            if label:
                _log(label)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        # status-like methods
        def write(self, *a, **k): _log(*a)
        def markdown(self, *a, **k): _log(*a)
        def info(self, *a, **k): _log(*a)
        def warning(self, *a, **k): _log(*a)
        def error(self, *a, **k): _log(*a)
        def success(self, *a, **k): _log(*a)

        def update(self, *a, **k):
            # streamlit status.update(label="...")
            label = k.get("label")
            if label:
                _log(label)
            return None

    class _Null:
        """Absorb chained calls: st.x.y().z ..."""
        def __init__(self, prefix="st"):
            self._prefix = prefix

        def __call__(self, *a, **k):
            return None

        def __getattr__(self, name):
            return _Null(prefix=f"{self._prefix}.{name}")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Progress:
        def __init__(self):
            self._last = None

        def progress(self, value=None):
            # value can be 0..100 or 0..1 depending on code; just log occasionally
            try:
                if value is not None and value != self._last:
                    self._last = value
                    _log(f"[progress] {value}")
            except Exception:
                pass
            return None

        def update(self, value=None):
            return self.progress(value)

        def empty(self):
            return None

    def _decorator(fn=None, **kwargs):
        if callable(fn):
            return fn
        def wrap(f):
            return f
        return wrap

    st = types.ModuleType("streamlit")

    # Common outputs -> log
    st.write = _log
    st.markdown = _log
    st.text = _log
    st.caption = _log
    st.code = lambda *a, **k: _log(*a)
    st.json = lambda *a, **k: _log(*a)
    st.dataframe = lambda *a, **k: None
    st.table = lambda *a, **k: None
    st.info = _log
    st.warning = _log
    st.error = _log
    st.success = _log
    st.title = _log
    st.header = _log
    st.subheader = _log
    st.set_page_config = lambda *a, **k: None

    # Progress
    st.progress = lambda *a, **k: _Progress()

    # Layout
    st.tabs = lambda labels, **k: [_Null(prefix=f"st.tabs[{i}]") for i in range(len(labels or []))]
    st.columns = lambda n, **k: [_Null(prefix=f"st.columns[{i}]") for i in range(int(n or 0))]
    st.container = lambda **k: _Ctx()
    st.expander = lambda *a, **k: _Ctx()
    st.form = lambda *a, **k: _Ctx()
    st.empty = lambda: _Null(prefix="st.empty")

    # Status/spinner return context managers that log their label (if any)
    st.status = lambda label=None, **k: _Ctx(label=str(label) if label else "")
    st.spinner = lambda text=None, **k: _Ctx(label=str(text) if text else "")

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

    # SAFE module-level __getattr__ fallback (no recursion)
    def _module_getattr(name):
        d = st.__dict__
        if name in d:
            return d[name]
        return _Null(prefix=f"st.{name}")

    st.__getattr__ = _module_getattr  # type: ignore[attr-defined]

    sys.modules["streamlit"] = st
    return st


# Install an early shim (print-based) BEFORE importing modules that import streamlit
if HEADLESS:
    install_streamlit_shim(print)

# -----------------------------------------------------------------------------
# Normal imports (after shim)
# -----------------------------------------------------------------------------
from google.cloud import storage
from progress import ProgressWriter
from train_sharp_model_from_bq_extracted import (
    train_sharp_model_for_market,
    train_timing_model_for_market,
)

def _force_unbuffered_streams():
    # Helps Cloud Run stream logs continuously
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

def _start_heartbeat(pw, label, every=45):
    stop_evt = threading.Event()

    def _hb():
        mins = 0
        while not stop_evt.wait(every):
            mins += every / 60.0
            # If your ProgressWriter requires pct, remove pct entirely or use a stable value
            try:
                pw.emit("hb", f"{label} ... still running ({mins:.1f}m)")
            except TypeError:
                pw.emit("hb", f"{label} ... still running ({mins:.1f}m)", pct=0.0)

    t = threading.Thread(target=_hb, daemon=True)
    t.start()
    return stop_evt

def main():
    _force_unbuffered_streams()

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

    # Re-install shim so st.write/status/progress go into ProgressWriter logs
    if HEADLESS:
        install_streamlit_shim(log_func)

    pw.emit("start", f"Training start run_id={run_id} sport={sport} market={market}", pct=0.0)

    try:
        if market == "All":
            pw.emit("timing", f"[{sport}] Training timing model...", pct=0.05)
            hb_stop = _start_heartbeat(pw, f"[{sport}] timing")
            try:
                try:
                    train_timing_model_for_market(sport=sport, bucket_name=bucket, log_func=log_func)
                except TypeError:
                    train_timing_model_for_market(sport=sport)
            finally:
                hb_stop.set()

            mkts = ("h2h", "spreads", "totals")
        else:
            mkts = (market,)

        n = len(mkts)
        for i, mkt in enumerate(mkts, start=1):
            pct = 0.10 + 0.80 * (i - 1) / max(1, n)
            pw.emit("train", f"[{sport}] Training sharp model market={mkt}", pct=pct)
            pw.emit("log", f"ENTER train_sharp_model_for_market sport={sport} mkt={mkt}", pct=pct)

            hb_stop = _start_heartbeat(pw, f"[{sport}] market={mkt}")
            try:
                try:
                    train_sharp_model_for_market(sport=sport, market=mkt, bucket_name=bucket, log_func=log_func)
                except TypeError:
                    train_sharp_model_for_market(sport=sport, market=mkt, bucket_name=bucket)
            finally:
                hb_stop.set()

            pw.emit("log", f"EXIT train_sharp_model_for_market sport={sport} mkt={mkt}", pct=pct)

        pw.emit("done", "Training complete ✅", pct=1.0)

    except Exception as e:
        pw.emit("error", f"{e}\n{traceback.format_exc()}", pct=1.0)
        raise

if __name__ == "__main__":
    main()
