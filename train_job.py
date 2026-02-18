# train_job.py
"""
Cloud Run Job / headless training entrypoint.

Goals:
- Run training in HEADLESS mode (no Streamlit UI) without crashing.
- Emit rich progress + model "Streamlit style" logs to:
  1) Google Cloud Logging (stdout/stderr), AND
  2) your ProgressWriter JSON in GCS (pw.emit)
- Avoid the recursion bug you hit (module __getattr__ calling getattr on itself).
- Keep output unbuffered so logs appear live.

Set env vars in Cloud Run Job:
  HEADLESS=1
  PYTHONUNBUFFERED=1
Optionally:
  SPORT=NBA
  MARKET=All|spreads|totals|h2h
  MODEL_BUCKET=sharp-models
  PROGRESS_URI=gs://... (optional; auto-derived if missing)
"""

import os
import sys
import uuid
import time
import threading
import traceback
import warnings
import logging


# -----------------------------------------------------------------------------
# Environment + basic noise control
# -----------------------------------------------------------------------------
HEADLESS = os.getenv("HEADLESS", "0") == "1"


def configure_cloud_logging():
    """
    Send python logging + warnings to stdout so Cloud Run captures them.
    Make stdout line-buffered when possible.
    """
    # stdout line buffering (Python 3.7+ may support reconfigure)
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    # Root logger -> stdout
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Route warnings through logging
    logging.captureWarnings(True)
    warnings.simplefilter("default")  # "once" if too noisy

    # Narrow suppression for known numpy spam (optional)
    if HEADLESS:
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0", category=RuntimeWarning)


# -----------------------------------------------------------------------------
# Headless Streamlit shim
# MUST install BEFORE importing sharp_line_dashboard (or anything that imports streamlit)
# -----------------------------------------------------------------------------
def install_streamlit_shim(log_func=print):
    """
    Provides a minimal `streamlit` module replacement that:
      - prevents crashes in headless execution
      - emits "Streamlit style" messages (write/markdown/status/progress/etc) via log_func
      - avoids recursion in __getattr__
    """
    import types

    def _format_arg(x, max_len=1400):
        try:
            import pandas as pd  # optional
            if isinstance(x, pd.DataFrame):
                head = x.head(12)
                return head.to_string() + f"\n[{x.shape[0]} rows x {x.shape[1]} columns]"
            if isinstance(x, pd.Series):
                return x.head(20).to_string()

            s = str(x)
            return s if len(s) <= max_len else (s[:max_len] + "…(truncated)")
        except Exception:
            s = str(x)
            return s if len(s) <= max_len else (s[:max_len] + "…(truncated)")

    def _log(*a, **k):
        msg = " ".join(_format_arg(v) for v in a).strip()
        if not msg:
            return None
        try:
            log_func(msg)
        except Exception:
            print(msg, flush=True)
        return None

    class _Ctx:
        """
        Context manager used for st.status(), st.spinner(), st.container(), st.expander(), st.form()
        Also supports .update(label="...") calls.
        """
        def __init__(self, label: str = ""):
            if label:
                _log(label)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        # common UI-ish methods
        def write(self, *a, **k): return _log(*a)
        def markdown(self, *a, **k): return _log(*a)
        def code(self, *a, **k): return _log(*a)
        def json(self, *a, **k): return _log(*a)
        def dataframe(self, *a, **k): return _log(*a)
        def table(self, *a, **k): return _log(*a)
        def info(self, *a, **k): return _log(*a)
        def warning(self, *a, **k): return _log(*a)
        def error(self, *a, **k): return _log(*a)
        def success(self, *a, **k): return _log(*a)

        def update(self, *a, **k):
            # Streamlit status.update(label="...") common pattern
            label = k.get("label") or (a[0] if a else "")
            if label:
                _log(label)
            return None

    class _Null:
        """Absorb any chained streamlit calls: st.x.y().z ..."""
        def __init__(self, prefix="st"):
            self._prefix = prefix

        def __call__(self, *a, **k):
            # Some code does st.sidebar.xxx(...) -> just log if it looks like a message
            if a:
                try:
                    _log(*a)
                except Exception:
                    pass
            return None

        def __getattr__(self, name):
            return _Null(prefix=f"{self._prefix}.{name}")

        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False

    class _Progress:
        """
        Streamlit progress bar object:
          pb = st.progress(0)
          pb.progress(50)
          pb.empty()
        """
        def __init__(self):
            self._last = None

        def progress(self, value=None, *a, **k):
            try:
                v = float(value) if value is not None else None
            except Exception:
                v = None
            if v is not None and v != self._last:
                self._last = v
                _log(f"[progress] {v}")
            return None

        def update(self, *a, **k):
            # Sometimes called with value=... or label=...
            if "label" in k and k["label"]:
                _log(k["label"])
            if "value" in k:
                return self.progress(k["value"])
            return None

        def empty(self):
            _log("[progress] cleared")
            return None

    def _decorator(fn=None, **kwargs):
        # supports @st.cache_data and @st.cache_data(...)
        if callable(fn):
            return fn

        def wrap(f):
            return f

        return wrap

    # Create module
    st = types.ModuleType("streamlit")

    # Common outputs -> log
    st.write = _log
    st.markdown = _log
    st.text = _log
    st.caption = _log
    st.code = _log
    st.json = _log
    st.dataframe = lambda df=None, *a, **k: _log(df) if df is not None else None
    st.table = lambda df=None, *a, **k: _log(df) if df is not None else None
    st.info = _log
    st.warning = _log
    st.error = _log
    st.success = _log
    st.title = _log
    st.header = _log
    st.subheader = _log
    st.set_page_config = lambda *a, **k: None

    # Progress (IMPORTANT: avoid crashes at st.progress(0))
    st.progress = lambda *a, **k: _Progress()

    # Layout
    st.tabs = lambda labels, **k: [_Null(prefix=f"st.tabs[{i}]") for i in range(len(labels or []))]
    st.columns = lambda n, **k: [_Null(prefix=f"st.columns[{i}]") for i in range(int(n or 0))]
    st.container = lambda **k: _Ctx()
    st.expander = lambda *a, **k: _Ctx(label=str(a[0]) if a else "")
    st.form = lambda *a, **k: _Ctx(label=str(a[0]) if a else "")
    st.empty = lambda: _Null(prefix="st.empty")

    # Status / spinner
    st.status = lambda *a, **k: _Ctx(label=str(a[0]) if a else (k.get("label") or ""))
    st.spinner = lambda *a, **k: _Ctx(label=str(a[0]) if a else (k.get("text") or ""))

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

    # SAFE module-level __getattr__ fallback (NO recursion!)
    def _module_getattr(name):
        d = st.__dict__
        if name in d:
            return d[name]
        return _Null(prefix=f"st.{name}")

    st.__getattr__ = _module_getattr  # type: ignore[attr-defined]

    # Register shim
    sys.modules["streamlit"] = st
    return st


# Install a minimal shim early so imports never crash in headless mode.
# We'll reinstall inside main() with pw.emit logger after pw exists.
if HEADLESS:
    configure_cloud_logging()
    install_streamlit_shim(print)


# -----------------------------------------------------------------------------
# Imports AFTER shim
# -----------------------------------------------------------------------------
from google.cloud import storage
from progress import ProgressWriter

from train_sharp_model_from_bq_extracted import (
    train_sharp_model_for_market,
    train_timing_model_for_market,
)


# -----------------------------------------------------------------------------
# Heartbeat thread (keeps logs alive during long CPU work)
# -----------------------------------------------------------------------------
def start_heartbeat(pw: ProgressWriter, *, label: str, every_sec: int = 45):
    stop_evt = threading.Event()

    def _hb():
        tick = 0
        while not stop_evt.is_set():
            # pct=None means "do not overwrite pct" (assuming your ProgressWriter supports it)
            try:
                pw.emit("hb", f"{label} ... still running ({tick})", pct=None)
            except TypeError:
                # fallback if pct=None not supported
                pw.emit("hb", f"{label} ... still running ({tick})")
            tick += 1
            stop_evt.wait(every_sec)

    t = threading.Thread(target=_hb, daemon=True)
    t.start()
    return stop_evt


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    configure_cloud_logging()

    run_id = os.environ.get("TRAIN_RUN_ID") or str(uuid.uuid4())[:8]
    sport = os.environ.get("SPORT", "NBA")
    market = os.environ.get("MARKET", "All")
    bucket = os.environ.get("MODEL_BUCKET", "sharp-models")

    progress_uri = os.environ.get("PROGRESS_URI")
    if not progress_uri:
        progress_uri = f"gs://{bucket}/train-progress/{sport}/{market}/{run_id}.json"
        os.environ["PROGRESS_URI"] = progress_uri

    # Create progress writer
    gcs = storage.Client()
    pw = ProgressWriter(progress_uri, gcs)

    # All logs go both to pw + stdout (cloud logs)
    def log_func(msg: str):
        s = str(msg)
        # Cloud logs
        print(s, flush=True)
        # GCS progress JSON
        pw.emit("log", s)

    # Reinstall streamlit shim with the real log_func,
    # so st.write/st.markdown inside the model training land in cloud logs too.
    if HEADLESS:
        install_streamlit_shim(log_func)

    pw.emit("start", f"Training start run_id={run_id} sport={sport} market={market}", pct=0.0)
    log_func(f"[boot] progress_uri={progress_uri}")

    # Decide markets
    if market == "All":
        mkts = ("h2h", "spreads", "totals")
    else:
        mkts = (market,)

    # Heartbeat for the overall job
    hb_stop = start_heartbeat(pw, label=f"[{sport}] market={market}", every_sec=45)

    try:
        # Optional timing model first
        if market == "All":
            pw.emit("timing", f"[{sport}] Training timing model...", pct=0.05)
            log_func(f"[timing] ENTER sport={sport}")
            try:
                train_timing_model_for_market(sport=sport, bucket_name=bucket, log_func=log_func)
            except TypeError:
                train_timing_model_for_market(sport=sport)
            log_func(f"[timing] EXIT sport={sport}")

        # Train sharp models per market
        n = len(mkts)
        for i, mkt in enumerate(mkts, start=1):
            pct = 0.10 + 0.80 * (i - 1) / max(1, n)
            pw.emit("train", f"[{sport}] Training sharp model market={mkt}", pct=pct)
            log_func(f"[sharp] ENTER sport={sport} market={mkt}")

            # Heartbeat per market (useful when a single market takes forever)
            hb_mkt_stop = start_heartbeat(pw, label=f"[{sport}] market={mkt}", every_sec=45)
            try:
                try:
                    train_sharp_model_for_market(
                        sport=sport,
                        market=mkt,
                        bucket_name=bucket,
                        log_func=log_func,
                    )
                except TypeError:
                    # Backward-compatible signature
                    train_sharp_model_for_market(sport=sport, market=mkt, bucket_name=bucket)
            finally:
                hb_mkt_stop.set()

            log_func(f"[sharp] EXIT sport={sport} market={mkt}")

        pw.emit("done", "Training complete ✅", pct=1.0)
        log_func("[done] Training complete ✅")

    except Exception as e:
        err = f"{e}\n{traceback.format_exc()}"
        pw.emit("error", err, pct=1.0)
        print(err, flush=True)
        raise

    finally:
        hb_stop.set()


if __name__ == "__main__":
    main()
