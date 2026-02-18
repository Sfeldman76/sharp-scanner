
# train_job.py
import os
import sys
import uuid
import traceback
import warnings
import logging
import time
import threading

from google.cloud import storage
from progress import ProgressWriter

HEADLESS = os.getenv("HEADLESS", "0") == "1"


# -----------------------------
# 1) Tee stdout/stderr -> pw.emit("log", ...)
# -----------------------------
class _StreamTee:
    """
    Mirrors writes to the original stream AND forwards full lines to a callback.
    """
    def __init__(self, stream, on_line):
        self._stream = stream
        self._on_line = on_line
        self._buf = ""

    def write(self, s):
        self._stream.write(s)
        self._stream.flush()

        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip()
            if line:
                try:
                    self._on_line(line)
                except Exception:
                    pass

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass


def install_streamlit_shim(log_func):
    """
    Must be called BEFORE importing sharp_line_dashboard / training modules.
    Wires st.write/st.info/etc to log_func.
    """
    import types

    def _log(*a, **k):
        msg = " ".join(str(x) for x in a).strip()
        if msg:
            log_func(msg)
        return None

    class _Ctx:
        def __init__(self, label=""):
            if label:
                _log(label)
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False
        def write(self, *a, **k): return _log(*a)
        def markdown(self, *a, **k): return _log(*a)
        def update(self, *a, **k):
            lab = k.get("label") or (a[0] if a else "")
            if lab:
                _log(lab)
            return None
        def success(self, *a, **k): return _log(*a)
        def warning(self, *a, **k): return _log(*a)
        def error(self, *a, **k): return _log(*a)

    class _Null:
        def __init__(self, prefix="st"):
            self._prefix = prefix
        def __call__(self, *a, **k):
            # if someone does st.something("text"), capture it
            if a:
                _log(*a)
            return None
        def __getattr__(self, name):
            return _Null(prefix=f"{self._prefix}.{name}")
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False

    class _Progress:
        def progress(self, v=None, *a, **k):
            if v is not None:
                _log(f"[progress] {v}")
            return None
        def update(self, *a, **k):
            if "label" in k and k["label"]:
                _log(k["label"])
            if "value" in k:
                return self.progress(k["value"])
            return None
        def empty(self):
            _log("[progress] cleared")
            return None

    def _decorator(fn=None, **kwargs):
        if callable(fn):
            return fn
        def wrap(f):
            return f
        return wrap

    st = types.ModuleType("streamlit")

    # outputs -> log_func
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

    # progress/layout/context
    st.progress = lambda *a, **k: _Progress()
    st.tabs = lambda labels, **k: [_Null(prefix=f"st.tabs[{i}]") for i in range(len(labels or []))]
    st.columns = lambda n, **k: [_Null(prefix=f"st.columns[{i}]") for i in range(int(n or 0))]
    st.container = lambda **k: _Ctx()
    st.expander = lambda *a, **k: _Ctx(label=str(a[0]) if a else "")
    st.form = lambda *a, **k: _Ctx(label=str(a[0]) if a else "")
    st.empty = lambda: _Null(prefix="st.empty")
    st.status = lambda *a, **k: _Ctx(label=str(a[0]) if a else (k.get("label") or ""))
    st.spinner = lambda *a, **k: _Ctx(label=str(a[0]) if a else (k.get("text") or ""))

    # widgets
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

    # state/sidebar
    st.session_state = {}
    st.sidebar = _Null(prefix="st.sidebar")

    # IMPORTANT: no recursion
    def _module_getattr(name):
        d = st.__dict__
        if name in d:
            return d[name]
        return _Null(prefix=f"st.{name}")

    st.__getattr__ = _module_getattr  # type: ignore[attr-defined]

    sys.modules["streamlit"] = st
    return st


def start_heartbeat(pw, label, every_sec=45):
    stop_evt = threading.Event()

    def _hb():
        i = 0
        while not stop_evt.is_set():
            pw.emit("hb", f"{label} ... still running ({i})", pct=None)
            i += 1
            stop_evt.wait(every_sec)

    t = threading.Thread(target=_hb, daemon=True)
    t.start()
    return stop_evt


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

    # ONE unified logger for everything
    def log_line(line: str):
        line = str(line).rstrip()
        if not line:
            return
        # Cloud Run logs
        print(line, flush=True)
        # Progress JSON feed
        pw.emit("log", line)

    # tee stdout/stderr so any print() inside model shows up in pw.emit("log", ...)
    sys.stdout = _StreamTee(sys.__stdout__, log_line)
    sys.stderr = _StreamTee(sys.__stderr__, log_line)

    # install shim *after* we have log_line but *before* importing training modules
    if HEADLESS:
        install_streamlit_shim(log_line)

    # NOW import anything that uses streamlit / prints
    from train_sharp_model_from_bq_extracted import (
        train_sharp_model_for_market,
        train_timing_model_for_market,
    )

    pw.emit("start", f"Training start run_id={run_id} sport={sport} market={market}", pct=0.0)
    log_line(f"[boot] HEADLESS={HEADLESS} bucket={bucket} progress_uri={progress_uri}")

    hb_stop = start_heartbeat(pw, f"[{sport}] market={market}", 45)

    try:
        if market == "All":
            pw.emit("timing", f"[{sport}] Training timing model...", pct=0.05)
            log_line(f"[timing] ENTER sport={sport}")
            try:
                train_timing_model_for_market(sport=sport, bucket_name=bucket, log_func=log_line)
            except TypeError:
                train_timing_model_for_market(sport=sport)
            log_line(f"[timing] EXIT sport={sport}")

            mkts = ("h2h", "spreads", "totals")
        else:
            mkts = (market,)

        n = len(mkts)
        for i, mkt in enumerate(mkts, start=1):
            pct = 0.10 + 0.80 * (i - 1) / max(1, n)
            pw.emit("train", f"[{sport}] Training sharp model market={mkt}", pct=pct)
            log_line(f"[sharp] ENTER sport={sport} market={mkt}")

            hb_mkt_stop = start_heartbeat(pw, f"[{sport}] market={mkt}", 45)
            try:
                try:
                    train_sharp_model_for_market(
                        sport=sport,
                        market=mkt,
                        bucket_name=bucket,
                        log_func=log_line,
                    )
                except TypeError:
                    train_sharp_model_for_market(sport=sport, market=mkt, bucket_name=bucket)
            finally:
                hb_mkt_stop.set()

            log_line(f"[sharp] EXIT sport={sport} market={mkt}")

        pw.emit("done", "Training complete ✅", pct=1.0)
        log_line("[done] Training complete ✅")

    except Exception as e:
        err = f"{e}\n{traceback.format_exc()}"
        pw.emit("error", err, pct=1.0)
        log_line(err)
        raise

    finally:
        hb_stop.set()


if __name__ == "__main__":
    main()
