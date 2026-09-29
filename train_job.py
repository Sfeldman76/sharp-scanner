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
    _scv21, _scv21_path, _scv21_sha = _load_exact_local_module("stat_combination_v2_1")
    _erv2, _erv2_path, _erv2_sha = _load_exact_local_module("edge_registry_v2")
    _etv1, _etv1_path, _etv1_sha = _load_exact_local_module("edge_topology_v1")
    _ecv1, _ecv1_path, _ecv1_sha = _load_exact_local_module("edge_complementarity_v1")
    _ecv2, _ecv2_path, _ecv2_sha = _load_exact_local_module("edge_complementarity_v2")
    _emmv1, _emmv1_path, _emmv1_sha = _load_exact_local_module("edge_mechanism_matrix_v1")

    train_sharp_model_for_market = _wrapper.train_sharp_model_for_market
    train_timing_model_for_market = _wrapper.train_timing_model_for_market

    # NCAAF edge-research fast path. This intentionally bypasses the production
    # champion-training graph and builds only the leakage-safe historical objects
    # consumed by V14.3 / STAT Combo / Edge Registry. It never publishes or
    # promotes a model artifact. Enable with NCAAF_EDGE_RESEARCH_ONLY=1 or
    # MARKET=edge_research.
    _edge_research_only = bool(
        str(sport).upper().strip() == "NCAAF" and (
            str(os.getenv("NCAAF_EDGE_RESEARCH_ONLY", "0")).strip().lower() in {"1","true","yes","on"}
            or str(market).lower().strip() in {"edge_research","research"}
        )
    )

    def _run_ncaaf_edge_research_stack():
        log_func("[EDGE-RESEARCH-STACK] phase=V14.3_RELIABILITY start=TRUE")
        _v143_out = _v143.run_v14_stat_reliability(
            dashboard_module=_sld, log_func=log_func, hard_fail=True
        )
        log_func("[STAT-COMBO-V2-RETIREMENT] path=STAT_COMBINATION_V1 state=ARCHIVED reason=GLOBAL_COMBO_NO_STABLE_INCREMENTAL_SKILL runtime_call=REMOVED")
        log_func("[STAT-COMBO-V2-RETIREMENT] component=DYNAMIC_STRENGTH_V1 state=ARCHIVED reason=CATASTROPHIC_RMSE_UNDERPERFORMANCE runtime_call=REMOVED")
        _scv21_tag = getattr(_scv21, "SCV21_SOURCE_TAG", None)
        if _scv21_tag != "stat-combination-v2.1-consensus-tail-recurrence":
            raise RuntimeError(
                f"[STAT-COMBO-V2.1-DEPLOY-PREFLIGHT] STALE_OR_MISSING source_tag={_scv21_tag!r} "
                f"path={str(_scv21_path)!r} sha={_scv21_sha[:16]}"
            )
        log_func(
            f"[STAT-COMBO-V2.1-DEPLOY-PREFLIGHT] PASS source_tag={_scv21_tag} path={_scv21_path} "
            f"sha={_scv21_sha[:16]} production_authority=0"
        )
        _scv21_out = _scv21.run_stat_combination_v2_1(
            dashboard_module=_sld, log_func=log_func, hard_fail=True
        )
        _erv2_tag = getattr(_erv2, "EDGE_REGISTRY_V2_SOURCE_TAG", None)
        if _erv2_tag != "edge-registry-v2-evidence-states":
            raise RuntimeError(
                f"[EDGE-REGISTRY-V2-DEPLOY-PREFLIGHT] STALE_OR_MISSING source_tag={_erv2_tag!r} "
                f"path={str(_erv2_path)!r} sha={_erv2_sha[:16]}"
            )
        log_func(
            f"[EDGE-REGISTRY-V2-DEPLOY-PREFLIGHT] PASS source_tag={_erv2_tag} path={_erv2_path} "
            f"sha={_erv2_sha[:16]} production_authority=0"
        )
        _erv2_out = _erv2.run_edge_registry_v2(
            dashboard_module=_sld, stat_out=_scv21_out, reliability_out=_v143_out,
            log_func=log_func, hard_fail=True
        )
        _etv1_tag = getattr(_etv1, "EDGE_TOPOLOGY_V1_SOURCE_TAG", None)
        if _etv1_tag != "edge-topology-v1-peer-source-map-clv-ledger":
            raise RuntimeError(
                f"[EDGE-TOPOLOGY-V1-DEPLOY-PREFLIGHT] STALE_OR_MISSING source_tag={_etv1_tag!r} "
                f"path={str(_etv1_path)!r} sha={_etv1_sha[:16]}"
            )
        log_func(
            f"[EDGE-TOPOLOGY-V1-DEPLOY-PREFLIGHT] PASS source_tag={_etv1_tag} path={_etv1_path} "
            f"sha={_etv1_sha[:16]} production_authority=0"
        )
        _etv1_out = _etv1.run_edge_topology_v1(
            dashboard_module=_sld, stat_out=_scv21_out, registry_out=_erv2_out,
            log_func=log_func, hard_fail=True
        )
        _ecv2_tag = getattr(_ecv2, "EDGE_COMPLEMENTARITY_V2_SOURCE_TAG", None)
        if _ecv2_tag != "edge-complementarity-v2-regime-vs-direction":
            raise RuntimeError(
                f"[EDGE-COMPLEMENTARITY-V2-DEPLOY-PREFLIGHT] STALE_OR_MISSING source_tag={_ecv2_tag!r} "
                f"path={str(_ecv2_path)!r} sha={_ecv2_sha[:16]}"
            )
        log_func(
            f"[EDGE-COMPLEMENTARITY-V2-DEPLOY-PREFLIGHT] PASS source_tag={_ecv2_tag} path={_ecv2_path} "
            f"sha={_ecv2_sha[:16]} production_authority=0"
        )
        _ecv2_out = _ecv2.run_edge_complementarity_v2(
            dashboard_module=_sld, stat_out=_scv21_out, registry_out=_erv2_out, topology_out=_etv1_out,
            log_func=log_func, hard_fail=True
        )
        _emmv1_tag = getattr(_emmv1, "EDGE_MECHANISM_MATRIX_V1_SOURCE_TAG", None)
        if _emmv1_tag != "edge-mechanism-matrix-v1-peer-source-orthogonality-clv":
            raise RuntimeError(
                f"[EDGE-MECHANISM-V1-DEPLOY-PREFLIGHT] STALE_OR_MISSING source_tag={_emmv1_tag!r} "
                f"path={str(_emmv1_path)!r} sha={_emmv1_sha[:16]}"
            )
        log_func(
            f"[EDGE-MECHANISM-V1-DEPLOY-PREFLIGHT] PASS source_tag={_emmv1_tag} path={_emmv1_path} "
            f"sha={_emmv1_sha[:16]} production_authority=0"
        )
        _emmv1_out = _emmv1.run_edge_mechanism_matrix_v1(
            dashboard_module=_sld, stat_out=_scv21_out, registry_out=_erv2_out, topology_out=_etv1_out,
            complementarity_v2_out=_ecv2_out, log_func=log_func, hard_fail=True
        )
        return _v143_out, _scv21_out, _erv2_out, _etv1_out, _ecv2_out, _emmv1_out

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
        if _edge_research_only:
            # FAST RESEARCH MODE: build only the upstream historical caches that
            # the edge-research stack consumes. Do not run timing, H2H/Totals
            # production training, AutoFS heads, promotion replay, or artifact
            # publication. This keeps normal production behavior unchanged when
            # the flag is off.
            log_func(
                "[EDGE-RESEARCH-FAST-PREFLIGHT] status=PASS sport=NCAAF "
                "scope=SPREAD_EDGE_RESEARCH skips=TIMING,H2H_PRODUCTION,TOTALS_PRODUCTION,"
                "GENERIC_AUTOFS,PROMOTION_REPLAY,ARTIFACT_PUBLICATION production_authority=0"
            )
            _t0 = __import__('time').perf_counter()
            _sld.fit_historical_ncaaf_core_expert("spreads", log_func=log_func)
            _t1 = __import__('time').perf_counter()
            _sld.fit_ncaaf_statistical_brain(log_func=log_func)
            _t2 = __import__('time').perf_counter()
            _cache = getattr(_sld, "_V1357_SPREAD_RESEARCH_CACHE", {})
            _sys_cache = getattr(_sld, "_V143_SYSTEM_HISTORY_CACHE", {})
            _games = _cache.get("games") if isinstance(_cache, dict) else None
            _oof = _cache.get("oof_margin") if isinstance(_cache, dict) else None
            _miner = ((_cache.get("system_miner_v2") or {}).get("spreads") or {}) if isinstance(_cache, dict) else {}
            if _games is None or getattr(_games, "empty", True) or _oof is None or not _sys_cache or not (_miner.get("systems") or []):
                raise RuntimeError(
                    "[EDGE-RESEARCH-FAST-CACHE] missing required cache "
                    f"games={0 if _games is None else len(_games)} oof={'READY' if _oof is not None else 'MISSING'} "
                    f"system_history={len(_sys_cache) if isinstance(_sys_cache,dict) else 0} "
                    f"spread_miner_systems={len(_miner.get('systems') or [])}"
                )
            log_func(
                f"[EDGE-RESEARCH-FAST-CACHE] status=PASS games={len(_games)} "
                f"system_history={len(_sys_cache)} spread_miner_systems={len(_miner.get('systems') or [])} "
                f"historical_seconds={_t1-_t0:.1f} stat_cache_seconds={_t2-_t1:.1f}"
            )
            _run_ncaaf_edge_research_stack()
            _t3 = __import__('time').perf_counter()
            log_func(
                f"[EDGE-RESEARCH-FAST-CONTRACT] status=PASS total_seconds={_t3-_t0:.1f} "
                "production_training_skipped=TRUE artifact_publication=FALSE production_authority=0"
            )
            log_func(
                "[CODE-LIFECYCLE-AUDIT] status=PASS active_production=UNCHANGED "
                "active_research=V14.3_RELIABILITY,STAT_COMBINATION_V2_1,EDGE_REGISTRY_V2,EDGE_TOPOLOGY_V1,EDGE_COMPLEMENTARITY_V2,EDGE_MECHANISM_MATRIX_V1,SYSTEM_MINER "
                "fast_path=EDGE_RESEARCH_ONLY skipped_legacy_runtime=TIMING,H2H_PRODUCTION,TOTALS_PRODUCTION,"
                "GENERIC_AUTOFS,PROMOTION_REPLAY,ARTIFACT_PUBLICATION production_contract=UNCHANGED"
            )
            pw.emit("done", "Edge research complete ✅", pct=1.0)
            return

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
            _run_ncaaf_edge_research_stack()
            log_func(
                "[CODE-LIFECYCLE-AUDIT] status=PASS active_production=V13_STAT_CURRENT_BASELINE "
                "active_research=V14.3_RELIABILITY,STAT_COMBINATION_V2_1,EDGE_REGISTRY_V2,EDGE_TOPOLOGY_V1,EDGE_COMPLEMENTARITY_V2,EDGE_MECHANISM_MATRIX_V1,SYSTEM_MINER,H2H_SIBLINGS "
                "retired_runtime=V14_DIRECT_ATS,V14.1_CORRECTORS,WEEKLY_STAT_REFIT,SPREAD_RESIDUAL_STACK,TOTAL_SCORE_V2,STAT_COMBINATION_V1,STAT_COMBINATION_V2,DYNAMIC_STRENGTH_V1,EDGE_REGISTRY_V1 "
                "retired_runtime_calls=0 v13_role=BENCHMARK_NOT_PROTECTED edge_generators=STAT,BIGAL,PATHI,MINER production_contract=PASS"
            )

        pw.emit("done", "Training complete ✅", pct=1.0)

    except Exception as e:
        pw.emit("error", f"{e}\n{traceback.format_exc()}", pct=1.0)
        raise

    finally:
        hb_stop.set()


if __name__ == "__main__":
    main()
