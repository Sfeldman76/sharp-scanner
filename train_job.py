# train_job.py
import os
import sys
import uuid
import traceback
import warnings
import logging
import threading

from google.cloud import storage
from progress import ProgressWriter

HEADLESS = os.getenv("HEADLESS", "0") == "1"


# -----------------------------------------------------------------------------
# Headless warning / numeric noise control
# -----------------------------------------------------------------------------
if HEADLESS:
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Degrees of freedom <= 0", category=RuntimeWarning)
    logging.getLogger("numpy").setLevel(logging.ERROR)


def install_streamlit_shim(log_func):
    """
    Wire streamlit calls to log_func, but do NOT capture all stdout/stderr.
    Must be installed BEFORE importing sharp_line_dashboard / training modules.
    """
    import types

    def _log(*a, **k):
        msg = " ".join(str(x) for x in a).strip()
        if msg:
            try:
                log_func(msg)
            except Exception:
                pass
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
            lab = k.get("label") or ""
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
            # capture if someone does st.something("text")
            if a:
                _log(*a)
            return None
        def __getattr__(self, name):
            return _Null(prefix=f"{self._prefix}.{name}")
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): return False

    class _Progress:
        def progress(self, v=None, *a, **k):
            # optional: only log when value is meaningful
            if v is not None:
                _log(f"[progress] {v}")
            return None
        def update(self, *a, **k):
            if "label" in k and k["label"]:
                _log(k["label"])
            if "value" in k:
                return self.progress(k["value"])
            return None
        def empty(self): return None

    def _decorator(fn=None, **kwargs):
        if callable(fn):
            return fn
        def wrap(f): return f
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

    # caching
    st.cache_data = _decorator
    st.cache_resource = _decorator

    # state/sidebar
    st.session_state = {}
    st.sidebar = _Null(prefix="st.sidebar")

    # SAFE getattr
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
        progress_uri = f"gs://{bucket}/train-progress/{sport}/{market}/{run_id}.jsonl"
        os.environ["PROGRESS_URI"] = progress_uri

    gcs = storage.Client()
    pw = ProgressWriter(progress_uri, gcs)

    # This is the ONLY log stream you want:
    def log_func(msg: str):
        pw.emit("log", str(msg))
        # optional: also show in Cloud Run logs, but only for log_func messages
        print(str(msg), flush=True)

    # Install shim before importing training modules
    if HEADLESS:
        install_streamlit_shim(log_func)

    # V13.5.0 deployment-path lock. Load the three training modules from
    # the exact directory containing this train_job.py, rather than allowing an
    # older copy elsewhere on PYTHONPATH or in a retained module cache to win.
    # This does not relax mixed-version protection: a genuinely stale /app file
    # still fails the version/tag check below.
    import hashlib
    import importlib
    import importlib.util
    from pathlib import Path

    _app_dir = Path(__file__).resolve().parent

    def _load_exact_local_module(_name):
        _path = (_app_dir / f"{_name}.py").resolve()
        if not _path.exists():
            raise RuntimeError(
                f"[V13.5.7.2-DEPLOY-PREFLIGHT] LOCAL_SOURCE_MISSING module={_name} path={_path}"
            )
        importlib.invalidate_caches()
        sys.modules.pop(_name, None)
        _spec = importlib.util.spec_from_file_location(_name, str(_path))
        if _spec is None or _spec.loader is None:
            raise RuntimeError(
                f"[V13.5.7.2-DEPLOY-PREFLIGHT] LOCAL_IMPORT_SPEC_FAILED module={_name} path={_path}"
            )
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_name] = _mod
        _spec.loader.exec_module(_mod)
        return _mod, _path, hashlib.sha256(_path.read_bytes()).hexdigest()

    # utils first, then dashboard, then wrapper. The wrapper's dashboard import
    # therefore resolves to the exact local dashboard object already registered.
    _utils, _utils_path, _utils_sha = _load_exact_local_module("utils")
    _sld, _dashboard_path, _dashboard_sha = _load_exact_local_module("sharp_line_dashboard")
    _wrapper, _wrapper_path, _wrapper_sha = _load_exact_local_module("train_sharp_model_from_bq_extracted")
    _v143, _v143_path, _v143_sha = _load_exact_local_module("v14_stat_reliability")
    _scv1, _scv1_path, _scv1_sha = _load_exact_local_module("stat_combination_v1")

    train_sharp_model_for_market = _wrapper.train_sharp_model_for_market
    train_timing_model_for_market = _wrapper.train_timing_model_for_market

    _expected_build = getattr(_sld, "V133_DEPLOY_BUILD_ID", None)
    _utils_build = getattr(_utils, "V133_DEPLOY_BUILD_ID", None)
    _dashboard_tag = getattr(_sld, "V1337_SOURCE_TAG", None)
    _utils_tag = getattr(_utils, "V1337_SOURCE_TAG", None)
    _wrapper_tag = getattr(_wrapper, "V1337_WRAPPER_SOURCE_TAG", None)
    _required_utils = [
        "build_ncaaf_core_feature_frame",
        "ncaaf_core_feature_recipe",
        "predict_ncaaf_v13_production",
        "_v1332_core_v2_logit_prob",
        "_v1332_enforce_canonical_spread_complements",
        "_v133_core_v2_runtime_score",
        "_v13310_runtime_stat_live_point_edge",
        "_v13310_runtime_brain_input_audit",
    ]
    _missing_utils = [n for n in _required_utils if not hasattr(_utils, n)]
    _required_dashboard = ["_v1357_system_miner_v2","_v1355_match_live_systems"]
    _missing_dashboard = [n for n in _required_dashboard if not hasattr(_sld, n)]
    if (not _expected_build) or (_expected_build != _utils_build) or _missing_utils or _missing_dashboard or _dashboard_tag != "dashboard-v13.5.7.2-walk-forward-diagnostics" or _utils_tag != "utils-v13.5.7.2-walk-forward-diagnostics" or _wrapper_tag != "wrapper-v13.5.7.2-walk-forward-diagnostics":
        raise RuntimeError(
            "[V13.5.7.2-DEPLOY-PREFLIGHT] MIXED_OR_STALE_DEPLOYMENT "
            f"dashboard_build={_expected_build!r} utils_build={_utils_build!r} "
            f"dashboard_tag={_dashboard_tag!r} utils_tag={_utils_tag!r} wrapper_tag={_wrapper_tag!r} missing_utils={_missing_utils} missing_dashboard={_missing_dashboard} "
            f"dashboard_path={str(_dashboard_path)!r} dashboard_sha={_dashboard_sha[:16]} "
            f"utils_path={str(_utils_path)!r} utils_sha={_utils_sha[:16]} "
            f"wrapper_path={str(_wrapper_path)!r} wrapper_sha={_wrapper_sha[:16]}. "
            "The running container itself contains a mixed source set; rebuild/redeploy the job image from this exact V13.5.7 bundle."
        )
    log_func(
        f"[V13.5.7.2-DEPLOY-PREFLIGHT] PASS build={_expected_build} "
        f"dashboard_tag={_dashboard_tag} utils_tag={_utils_tag} wrapper_tag={_wrapper_tag} "
        f"dashboard_path={_dashboard_path} dashboard_sha={_dashboard_sha[:16]} "
        f"utils_path={_utils_path} utils_sha={_utils_sha[:16]} wrapper_path={_wrapper_path} wrapper_sha={_wrapper_sha[:16]}"
    )
    _v143_tag = getattr(_v143, "V143_SOURCE_TAG", None)
    if _v143_tag != "v14.3-stat-reliability-hardening":
        raise RuntimeError(
            f"[V14.3-DEPLOY-PREFLIGHT] STALE_OR_MISSING source_tag={_v143_tag!r} "
            f"path={str(_v143_path)!r} sha={_v143_sha[:16]}"
        )
    log_func(
        f"[V14.3-DEPLOY-PREFLIGHT] PASS source_tag={_v143_tag} "
        f"path={_v143_path} sha={_v143_sha[:16]} v14_1_runtime=REMOVED production_authority=0"
    )

    pw.emit("start", f"Training start run_id={run_id} sport={sport} market={market}", pct=0.0)

    hb_stop = start_heartbeat(pw, f"[{sport}] market={market}", 45)

    try:
        if market == "All":
            pw.emit("timing", f"[{sport}] Training timing model...", pct=0.05)
            # Call exactly once. A TypeError raised inside training is a real
            # training failure and must propagate; retrying the entire model can
            # duplicate work and mutate production state twice.
            train_timing_model_for_market(
                sport=sport, bucket_name=bucket, log_func=log_func
            )

            mkts = ("h2h", "spreads", "totals")
        else:
            mkts = (market,)

        n = len(mkts)
        _trained_market_keys = set()
        for i, mkt in enumerate(mkts, start=1):
            _market_key = (str(sport).upper().strip(), str(mkt).lower().strip())
            if _market_key in _trained_market_keys:
                raise RuntimeError(
                    f"Duplicate market training blocked in one job: {_market_key}"
                )
            _trained_market_keys.add(_market_key)
            pct = 0.10 + 0.80 * (i - 1) / max(1, n)
            pw.emit("train", f"[{sport}] Training sharp model market={mkt}", pct=pct)

            hb_mkt_stop = start_heartbeat(pw, f"[{sport}] market={mkt}", 45)
            try:
                # EXACTLY ONE sharp-model invocation per market. Do not catch
                # TypeError here: a TypeError can occur late after artifact
                # evaluation/promotion, and the old fallback silently retrained
                # the entire market a second time.
                train_sharp_model_for_market(
                    sport=sport, market=mkt, bucket_name=bucket, log_func=log_func
                )
            finally:
                hb_mkt_stop.set()

        # V14.3: V13 STAT remains frozen. Retired replacement/corrector and weekly
        # adaptive-refit experiments no longer execute in the normal job. Reliability
        # research now asks FOLLOW / FADE / SUPPRESS on season-forward OOF history.
        if str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() in ("all", "spreads"):
            log_func("[V14.3-RETIREMENT] path=V14_DIRECT_ATS_MODELS state=ARCHIVED runtime_call=REMOVED")
            log_func("[V14.3-RETIREMENT] path=V14.1_RESIDUAL_CORRECTORS state=ARCHIVED runtime_call=REMOVED")
            log_func("[V14.3-RETIREMENT] path=WEEKLY_STAT_REFIT state=ARCHIVED runtime_call=REMOVED")
            log_func("[V14.3-RETIREMENT] path=SPREAD_RESIDUAL_STACK_V1 state=ARCHIVED runtime_call=REMOVED")
            log_func("[V14.3-RETIREMENT] path=TOTAL_SCORE_V2 state=ARCHIVED runtime_call=REMOVED")
            _v143.run_v14_stat_reliability(
                dashboard_module=_sld, log_func=log_func, hard_fail=True
            )
            _scv1_tag = getattr(_scv1, "SCV1_SOURCE_TAG", None)
            if _scv1_tag != "stat-combination-v1-market-error-small-ensemble":
                raise RuntimeError(
                    f"[STAT-COMBO-DEPLOY-PREFLIGHT] STALE_OR_MISSING source_tag={_scv1_tag!r} "
                    f"path={str(_scv1_path)!r} sha={_scv1_sha[:16]}"
                )
            log_func(
                f"[STAT-COMBO-DEPLOY-PREFLIGHT] PASS source_tag={_scv1_tag} path={_scv1_path} "
                f"sha={_scv1_sha[:16]} production_authority=0"
            )
            _scv1.run_stat_combination_v1(
                dashboard_module=_sld, log_func=log_func, hard_fail=True
            )
            log_func(
                "[CODE-LIFECYCLE-AUDIT] status=PASS active_production=V13_STAT_CURRENT_BASELINE "
                "active_research=V14.3_RELIABILITY,STAT_COMBINATION_V1,SYSTEM_MINER,H2H_SIBLINGS "
                "retired_runtime=V14_DIRECT_ATS,V14.1_CORRECTORS,WEEKLY_STAT_REFIT,SPREAD_RESIDUAL_STACK,TOTAL_SCORE_V2 "
                "retired_runtime_calls=0 v13_role=BENCHMARK_NOT_PROTECTED production_contract=PASS"
            )

        pw.emit("done", "Training complete ✅", pct=1.0)

    except Exception as e:
        pw.emit("error", f"{e}\n{traceback.format_exc()}", pct=1.0)
        raise

    finally:
        hb_stop.set()


if __name__ == "__main__":
    main()
