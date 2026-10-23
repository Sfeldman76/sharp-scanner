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
    import json  # initialize function-local binding before any route uses it
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

    # NCAAF_FULL_HISTORY_V3_ISOLATED_ROUTE_20261019 — separate research job mode.
    # No Production V1 or shared heavy-research path is executed by this branch.
    if str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() == "ncaaf_full_history_v3":
        if str(os.getenv("NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED", "0")).strip().lower() not in {"1","true","yes","on"}:
            raise RuntimeError("[NCAAF-FULL-HISTORY-V3] archive opt-in required; set NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED=1")
        import sharp_line_dashboard as _v3_dashboard
        import ncaaf_full_history_challenger_v3 as _v3
        log_func("[NCAAF-FULL-HISTORY-V3] phase=HISTORICAL_CACHE source=ARCHIVE_2013_2025 year_2026_queried=FALSE production_mutation=FALSE")
        pw.emit("cache", "NCAAF Full-History V3: prepare historical game cache", pct=0.15)
        _v3_dashboard.fit_historical_ncaaf_core_expert("spreads", log_func=log_func)
        _v3_dashboard.fit_ncaaf_statistical_brain(log_func=log_func)
        pw.emit("v3", "NCAAF Full-History V3: fit expanding-origin football models", pct=0.55)
        _v3_result=_v3.run_full_history_challenger_v3(
            dashboard_module=_v3_dashboard,storage_client=gcs,
            bucket_name=bucket,log_func=log_func)
        if _v3_result.get("status") != "RESEARCH_COMPLETE_NO_PROMOTION":
            raise RuntimeError("[NCAAF-FULL-HISTORY-V3] discovery incomplete: "+str(_v3_result.get("status")))
        pw.emit("done", "NCAAF Full-History V3 complete (research only)", pct=1.0)
        return

    # NCAAF_V4_FOOTBALL_TOURNAMENT_20261020 — research-only, no production route.
    if str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() == "ncaaf_margin_tournament_v4":
        if str(os.getenv("NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED", "0")).strip().lower() not in {"1","true","yes","on"}:
            raise RuntimeError("[NCAAF-V4] explicit archive research opt-in required")
        import sharp_line_dashboard as _v4_dashboard
        import ncaaf_margin_tournament_v4 as _v4
        log_func("[NCAAF-V4] phase=CACHE_PREP legacy_dashboard_cache_only=TRUE authority=0")
        pw.emit("cache", "NCAAF V4: preparing historical game + CFBD archive context", pct=0.12)
        _v4_dashboard.fit_historical_ncaaf_core_expert("spreads", log_func=log_func)
        _v4_dashboard.fit_ncaaf_statistical_brain(log_func=log_func)
        pw.emit("research", "NCAAF V4: chronological football-model tournament", pct=0.50)
        _v4_result = _v4.run_cloud_research(
            dashboard_module=_v4_dashboard, storage_client=gcs,
            bucket_name=bucket, log_func=log_func)
        if _v4_result.get("status") != "RESEARCH_COMPLETE_NO_PROMOTION":
            raise RuntimeError("[NCAAF-V4] incomplete: " + str(_v4_result.get("status")))
        pw.emit("done", "NCAAF V4 football tournament complete (research-only)", pct=1.0)
        return

    # NCAAF V4.2 conditional football feature research; ALL FIVE families in one
    # isolated, no-market, no-promotion historical tournament.
    if str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() == "ncaaf_conditional_features_v42":
        if str(os.getenv("NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED", "0")).strip().lower() not in {"1","true","yes","on"}:
            raise RuntimeError("[NCAAF-V42] explicit archive research opt-in required")
        import sharp_line_dashboard as _v42_dashboard
        import ncaaf_v42_conditional_features as _v42
        log_func("[NCAAF-V42] phase=HISTORICAL_CACHE_PREP source=2013_2025 legacy_cache_only=TRUE production_authority=0")
        pw.emit("cache", "V4.2: preparing verified historical game and CFBD archive join", pct=0.10)
        _v42_dashboard.fit_historical_ncaaf_core_expert("spreads", log_func=log_func)
        _v42_dashboard.fit_ncaaf_statistical_brain(log_func=log_func)
        pw.emit("research", "V4.2: five conditional feature families plus predeclared combinations", pct=0.50)
        result = _v42.run_cloud_research(
            dashboard_module=_v42_dashboard,storage_client=gcs,
            bucket_name=bucket,log_func=log_func)
        if result.get("status") != "EXPLORATORY_RESEARCH__SOURCE_VINTAGE_HOLD":
            raise RuntimeError("[NCAAF-V42] unexpected status: " + str(result.get("status")))
        pw.emit("done", "V4.2 football feature tournament complete: research only", pct=1.0)
        return

    # NCAAF V4.1 retrospective artifact audit, zero historical cache or model refit.
    # Input is the existing V4 published report + game predictions in research GCS.
    if str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() == "ncaaf_v41_integrity":
        import ncaaf_v41_integrity as _v41
        log_func("[NCAAF-V41] phase=READ_PRIOR_V4_ARTIFACTS legacy_cache=SKIPPED bq_queries=NONE authority=0")
        pw.emit("audit", "V4.1: checking published V4 per-game forecasts and pregame integrity", pct=0.15)
        _v41_result = _v41.run_cloud_audit(
            storage_client=gcs, bucket_name=bucket, log_func=log_func)
        if _v41_result.get("status") != "RETROSPECTIVE_AUDITED__SOURCE_VINTAGE_HOLD":
            raise RuntimeError("[NCAAF-V41] unexpected audit status " + str(_v41_result.get("status")))
        pw.emit("done", "V4.1 audit complete; source-vintage certification remains gated", pct=1.0)
        return

    # NCAAF Production V1 historical fixed-backbone replay (RESEARCH ONLY).
    # This is not a re-scoring of the fitted 2026 artifact, which was trained
    # through 2025. Each validation year fits the original fixed head recipes
    # using strictly earlier seasons. There are no historical bet-ROI claims.
    if str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() == "ncaaf_prod_v1_historical_replay":
        import sharp_line_dashboard as _v1hist_dashboard
        import ncaaf_prod_v1_historical_replay as _v1hist
        pw.emit("audit", "NCAAF Production V1: expanding-season replay of frozen head recipes (research only)", pct=0.10)
        result = _v1hist.run_cloud_research(
            dashboard_module=_v1hist_dashboard, storage_client=gcs,
            bucket_name=bucket, log_func=log_func)
        if result.get("status") != "RESEARCH_FIXED_BACKBONE_REPLAY__HISTORICAL_SELECTION_AND_SOURCE_VINTAGE_HOLD":
            raise RuntimeError("[NCAAF-PROD-V1-BACKTEST] unexpected status="+str(result.get("status")))
        pw.emit("done", "NCAAF Production V1 historical model replay complete; original bet ROI unverified", pct=1.0)
        return

    # NCAAF V5.3 — one comprehensive football + legacy-brain research run.
    # Logs every source, stage, integrity gate and partial failure. No V1 refit,
    # no silent fallbacks or promotion of previously inspected historical results.
    if str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() == "ncaaf_v53_full_research":
        import ncaaf_v53_full_research as _v53
        pw.emit("research", "V5 full research: V4 football, V4.2, legacy Big Al/Pathi/PT/Miner, V5 point edges and shadow authority", pct=0.05)
        _result=_v53.run_full(storage_client=gcs,bucket_name=bucket,log_func=log_func)
        if _result.get("status")!="COMPLETE_FULL_RESEARCH__SHADOW_AUTHORITY__VINTAGE_HOLD" or _result.get("bet_authority")!=0:
            raise RuntimeError("[NCAAF-V53] incomplete or unsafe authority contract: "+str(_result.get("status")))
        pw.emit("done", "NCAAF V5 full research finished; see detailed audited phases and candidate policy", pct=1.0)
        return

    # NCAAF V5.5: fast COMBINED brain/market-rich research from a completed
    # immutable CFBD V5.4.9 archive. No CORE re-fit or production file writes.
    if str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() == "ncaaf_v55_combined_brain":
        import ncaaf_v55_combined_brain as _v55
        from google.cloud import bigquery as _v55_bq
        pw.emit("research", "V5.5: selected V4 CORE + STAT availability + Big Al/Pathi/Miner + price-only historical Market Rich", pct=0.10)
        _result=_v55.run_cloud(storage_client=gcs,bucket_name=bucket,
            bq_client=_v55_bq.Client(project="sharplogger"), log_func=log_func)
        if (_result.get("status")!="COMPLETE_COMBINED_BRAIN_RESEARCH__PROVISIONAL_ONLY"
                or _result.get("bet_authority")!=0 or _result.get("production_modified") is not False):
            raise RuntimeError("[NCAAF-V55] incomplete or unsafe production authority contract")
        pw.emit("done", "NCAAF V5.5 combined-brain research completed; see immutable GCS result directory", pct=1.0)
        return

    # V5.5.4 targeted 2026 CFBD /games backfill — independent maintenance.
    # Preserves all existing market, features, research, production and model data.
    if str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() == "ncaaf_v55_cfbd_2026_games_repair":
        import ncaaf_v554_cfbd_games_backfill as _v554
        from google.cloud import bigquery as _v554_bq
        pw.emit("audit", "V5.5: audit 2026 CFBD game archive and backfill /games ONLY", pct=0.10)
        _v554_result=_v554.run_backfill(
            bq_client=_v554_bq.Client(project="sharplogger"),
            storage_client=gcs,bucket_name=bucket,log_func=log_func)
        if (_v554_result.get("status") not in ("PASS", "PASS_ALREADY_COMPLETE")
            or _v554_result.get("production_modified") is not False
            or _v554_result.get("bet_authority") != 0):
            raise RuntimeError("[NCAAF-V554] CFBD 2026 games archive repair is not verified")
        pw.emit("done", "2026 CFBD /games archive verified; next run V5.5 Weekly CORE", pct=1.0)
        return

    # V5.5 WEEKLY: own CFBD NONLINEAR_HGB forecast, independently priced
    # current-game slate and research brains. The frozen Production V1 scorer,
    # pick selector, contract, and bet ledgers are NOT used or modified.
    if str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() == "ncaaf_v55_weekly":
        import ncaaf_v552_live as _v552
        from google.cloud import bigquery as _v552_bq
        pw.emit("model", "V5.5: CFBD-only 2013-25 champion training + 2026 pregame scoring; independent selection", pct=0.10)
        _v552_result=_v552.run_weekly(storage_client=gcs,bucket_name=bucket,
            bq_client=_v552_bq.Client(project="sharplogger"),log_func=log_func)
        if (_v552_result.get("status")!="COMPLETE_V55_INDEPENDENT_WEEKLY_SHADOW"
            or _v552_result.get("production_mutation") is not False
            or _v552_result.get("production_v1_probabilities_used") is not False
            or _v552_result.get("v55_bet_authority")!=0):
            raise RuntimeError("[NCAAF-V552] independent weekly model did not publish safely")
        pw.emit("done", "NCAAF V5.5 independent weekly CORE and brains published", pct=1.0)
        return

    # V5.5 market-only repricing: reuse the LAST VERIFIED V5.5 weekly CFBD
    # forecast; no model fit or Production V1 probabilities may be substituted.
    if str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() == "ncaaf_v55_market_refresh":
        import ncaaf_v552_live as _v552
        pw.emit("market", "V5.5 market-only quote update against the latest V5.5 CORE projections", pct=0.30)
        _result=_v552.refresh_market_only(storage_client=gcs,bucket_name=bucket,log_func=log_func)
        if (_result.get("status")!="COMPLETE_V55_INDEPENDENT_WEEKLY_SHADOW"
                or _result.get("production_v1_probabilities_used") is not False
                or _result.get("core_refit") is not False):
            raise RuntimeError("[NCAAF-V552] unsafe market-only refresh")
        pw.emit("done", "V5.5 CORE retained; market-price snapshot refreshed", pct=1.0)
        return

    # NCAAF V5.4: CFBD authoritative historical CORE and system mining.
    # Original weekly Production V1 and legacy research remain untouched.
    if str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() == "ncaaf_v54_full_research":
        import ncaaf_v54_full_research as _v54
        pw.emit("research", "V5 full research with CFBD-only historical football facts, separate rich market and original expert rules", pct=0.05)
        _result=_v54.run_full(storage_client=gcs,bucket_name=bucket,log_func=log_func)
        if _result.get("status")!="COMPLETE_FULL_RESEARCH__SHADOW_AUTHORITY__VINTAGE_HOLD" or _result.get("bet_authority")!=0:
            raise RuntimeError("[NCAAF-V54] incomplete or unsafe authority contract: "+str(_result.get("status")))
        pw.emit("done", "NCAAF V5.4 CFBD-authoritative research finished; inspect source and full-stage audits", pct=1.0)
        return

    # NCAAF V5.2: source discovery and independently verified brain attribution.
    # No V4 refit, production changes, inferred odds or unverified historical votes.
    if str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() == "ncaaf_v52_brain_attribution":
        import ncaaf_v52_brain_research as _v52
        pw.emit("research", "V5.2: inventory historical Big Al/Pathi/PT/Miner sources and gate original pregame brain evidence", pct=0.20)
        _result = _v52.cloud_run(storage_client=gcs, bucket_name=bucket, log_func=log_func)
        if _result.get("bet_authority") != 0 or _result.get("production_modified") is not False:
            raise RuntimeError("[NCAAF-V52] unexpected authority contract")
        pw.emit("done", f"V5.2 completed with status={_result['status']}; source inventory and evidence gates published", pct=1.0)
        return

    # NCAAF V5.1 — Read-only point-edge research: NO refit, no live bet votes.
    if str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() == "ncaaf_v51_edge_research":
        import ncaaf_v51_edge_research as _v51
        pw.emit("research", "V5.1: deduplicate V5 signals, verify frozen V4 outcome/orientation, score conditional point edges and source-vintage gated brains", pct=0.20)
        _result = _v51.cloud_run(storage_client=gcs, bucket_name=bucket, log_func=log_func)
        if _result.get("status") != "HISTORICAL_EDGE_ATTRIBUTION__SOURCE_VINTAGE_HOLD":
            raise RuntimeError("[NCAAF-V51] unexpected status: " + str(_result.get("status")))
        pw.emit("done", "V5.1 point-edge attribution published; no bet authority, no historical ROI", pct=1.0)
        return

    # NCAAF V5 — Independent football margins -> market-value research.
    # Consumes immutable historical OOF research artifacts only. The optional
    # BIG_AL/PATHI/PT/MINER/STATE snapshot bridge is timestamp-vintage gated.
    # No Production V1 files, weekly retraining, dashboards or bet votes run.
    if str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() == "ncaaf_v5_market_value":
        import ncaaf_v5_betting_research as _v5
        pw.emit("research", "V5: verify V4/A forecasts, calibrate ATS margin probabilities and audit independent brains", pct=0.18)
        _result = _v5.cloud_run(storage_client=gcs, bucket_name=bucket, log_func=log_func)
        if _result.get("status") not in {
                "HISTORICAL_DIRECTIONAL_RESEARCH__SOURCE_VINTAGE_HOLD",
                "NO_VERIFIED_HISTORICAL_OPEN_MARKET_GAMES__RESEARCH_BLOCKED"}:
            raise RuntimeError("[NCAAF-V5] unexpected status: " + str(_result.get("status")))
        pw.emit("done", "V5 research report published; betting authority remains zero", pct=1.0)
        return

    # NCAAF V4.3: read-only focused research validation and production robustness.
    # This route does NOT import dashboard history trainers, refit challengers,
    # touch the frozen production model, or give any betting authority.
    if str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() == "ncaaf_v43_focused_validation":
        import ncaaf_v43_validation as _v43
        import ncaaf_prod_v1_robustness as _prod_audit
        pw.emit("validation", "V4.3: comparing frozen V4 with A, AB, AC on immutable V4.2 predictions", pct=0.20)
        result = _v43.run_cloud_audit(storage_client=gcs,bucket_name=bucket,log_func=log_func)
        if result.get("status") != "EXPLORATORY_FOCUSED_VALIDATION__SOURCE_VINTAGE_HOLD":
            raise RuntimeError("[NCAAF-V43] unexpected status: " + str(result.get("status")))
        pw.emit("robustness", "V4.3: auditing prospective Production V1 ledger read-only", pct=0.70)
        _prod_audit.run_cloud_audit(storage_client=gcs,bucket_name=bucket,log_func=log_func)
        pw.emit("done", "V4.3 validation + Production V1 robustness audit complete, research only", pct=1.0)
        return

    # NFL Heavy Research / Engine V3.17 — persistent Miner library + Big Al current-card fingerprint + protected PT lanes.
    # Research-only: frozen CORE remains the fair-value anchor. STAT/PBP/context learn
    # residual corrections; SYSTEM/STAT/PBP become selective trust lanes; a partially pooled
    # distribution is a challenger; actual price is required for live EV. 2026 stays sealed.
    if str(sport).upper().strip() == "NFL" and str(market).lower().strip() == "nfl_research_heavy":
        from google.cloud import bigquery

        def _load_nfl_heavy_exact(_name, _tag):
            import nfl_engine as _nfl_engine
            return _nfl_engine.load_component(_name, _tag)

        _audit = _load_nfl_heavy_exact("nfl_audit_v1", "nfl-audit-v1.3-prior-feature-provenance-20260930")
        _rcontract = _load_nfl_heavy_exact("nfl_research_contract_v1", "nfl-research-contract-v1.9.2-protected-architecture-20261001")
        _syscontract = _load_nfl_heavy_exact("nfl_research_v2_contract", "nfl-research-v2.0-foundation-expansion-20261001")
        _ledger3 = _load_nfl_heavy_exact("nfl_prospective_ledger_v3", "nfl-prospective-ledger-v3-v1.9.4-edge-gate-shadow-20261001")
        _research = _load_nfl_heavy_exact("nfl_research_engine_v2", "nfl-research-engine-v1.9.4-edge-gate-manager-20261001")
        _systems = _load_nfl_heavy_exact("nfl_system_lab_v3", "nfl-system-lab-v3.17.1-bigal-fingerprint-detail-logging-20261011")
        _ptinc = _load_nfl_heavy_exact("nfl_pt_incremental_v1", "nfl-pt-incremental-v1.1-encompassing-control-residual-20261007")
        _pbpdiag = _load_nfl_heavy_exact("nfl_pbp_attribution_v1", "nfl-pbp-attribution-v1-research-v2.0.2-frozen-core2-20261001")
        _heavy = _load_nfl_heavy_exact("nfl_heavy_research_v31", "nfl-heavy-research-v2.11.2-source-confluence-attribution-20261004")

        pw.emit("audit", f"[NFL-HEAVY-V317] Start run={run_id}; frozen V29 Spread 0.575 benchmark + selective trust + rules index + Big Al 6 coverage + price-aware EV; production mutation forbidden; 2026 sealed", pct=0.03)
        try:
            _bq = bigquery.Client(project="sharplogger")
            _rcontract.assert_contract()
            _syscontract.assert_contract()
            _audit_report = _audit.run_nfl_audit_v1(storage_client=gcs, bucket_name=bucket, log_func=log_func)
            if _audit_report.get("status") != "READY_FOR_OFFLINE_CHALLENGER_SANDBOX":
                raise RuntimeError("[NFL-HEAVY-V317-HOLD] PRECEDING_AUDIT_NOT_GREEN "+str(_audit_report.get("status")))
            _health = _ledger3.ledger_health_check(_bq, service_identity_hint="sharp-train-sa@sharplogger.iam.gserviceaccount.com")

            pw.emit("core_stat", "Audit PASS; run protected CORE/STAT/residual challenger research", pct=0.16)
            _research_report = _research.run_nfl_research_engine_v2(
                bq_client=_bq, storage_client=gcs, bucket_name=bucket,
                audit_report=_audit_report, ledger_health=_health, log_func=log_func,
            )

            pw.emit("systems", "CORE/STAT complete; run System Miner V3.17 with persistent exact-incumbent revalidation + Big Al pregame fingerprint + separate PT external-family lane", pct=0.34)
            _system_report = _systems.run_nfl_system_lab_v3(
                bq_client=_bq, storage_client=gcs, bucket_name=bucket,
                audit_report=_audit_report, log_func=log_func,
            )

            # V3.13: formal incremental-information test. This is separate from the
            # PT System Miner lane: frozen Production V1 is the control; simple blend,
            # residual, and conditional-residual challengers are selected pre-2023 and
            # validated on 2023-2025. No 2026 query and no automatic promotion.
            pw.emit("pt_incremental", "Systems complete; test PT incremental value vs frozen Production V1", pct=0.48)
            _pt_incremental = _ptinc.run_nfl_pt_incremental_v1(
                bq_client=_bq, storage_client=gcs, bucket_name=bucket,
                system_report=_system_report, log_func=log_func,
            )
            if not isinstance(_pt_incremental,dict) or int(_pt_incremental.get("production_authority") or 0) != 0 or bool(_pt_incremental.get("year_2026_queried")):
                raise RuntimeError("[NFL-HEAVY-V317-HOLD] PT_INCREMENTAL_CONTRACT_FAIL")
            _system_report["pt_incremental_value"]=_pt_incremental

            pw.emit("pbp", "PT incremental test complete; run frozen PBP attribution/confirmation diagnostic", pct=0.58)
            try:
                _pbp_report = _pbpdiag.run_nfl_pbp_attribution_v1(
                    bq_client=_bq, storage_client=gcs, bucket_name=bucket,
                    audit_report=_audit_report, log_func=log_func,
                )
            except Exception as _pbp_exc:
                _pbp_report = {"status":"PBP_ADVANCED_DIAGNOSTIC_UNAVAILABLE","error":f"{type(_pbp_exc).__name__}:{_pbp_exc}","production_authority":0}
                log_func("[NFL-HEAVY-V211-PBP-HOLD] "+json.dumps(_pbp_report, sort_keys=True, default=str))

            pw.emit("fusion", "Run frozen 0.575 Spread benchmark + selective-trust challengers + partially pooled distribution + price-aware EV + H2H freeze fix", pct=0.72)
            _result = _heavy.run_nfl_heavy_research(
                bq_client=_bq, storage_client=gcs, bucket_name=bucket,
                audit_report=_audit_report, research_report=_research_report,
                system_report=_system_report, pbp_report=_pbp_report, log_func=log_func,
            )
            if _result.get("production_mutated") is not False or int(_result.get("production_authority") or 0) != 0:
                raise RuntimeError("[NFL-HEAVY-V317-HOLD] RESEARCH_ROUTE_ATTEMPTED_PRODUCTION_AUTHORITY")
            log_func("[NFL-HEAVY-V317-OPERATOR-CONTRACT] "+json.dumps({
                "status":_result.get("status"),
                "research_contract_sha256":_result.get("research_contract_sha256"),
                "production_mutated":False,
                "production_authority":0,
                "automatic_promotion":False,
                "year_2026_queried":False,
                "spread_residual_model":(((_result.get("markets") or {}).get("SPREADS") or {}).get("selected_residual_model")),
                "spread_policy":(((_result.get("markets") or {}).get("SPREADS") or {}).get("betting_policy") or {}).get("status"),
                "spread_threshold":(((_result.get("markets") or {}).get("SPREADS") or {}).get("betting_policy") or {}).get("selected_threshold"),
                "spread_frozen_base_threshold":(((_result.get("markets") or {}).get("SPREADS") or {}).get("frozen_base_policy") or {}).get("threshold"),
                "spread_frozen_base_reproduced":(((_result.get("markets") or {}).get("SPREADS") or {}).get("frozen_base_policy") or {}).get("reproduced_in_current_run"),
                "spread_partial_pool_threshold":(((_result.get("markets") or {}).get("SPREADS") or {}).get("partial_pooling_policy") or {}).get("selected_threshold"),
                "spread_selective_trust_enhancers":(((_result.get("markets") or {}).get("SPREADS") or {}).get("selective_trust") or {}).get("frozen_enhancer_challengers"),
                "spread_veto_watchlist":(((_result.get("markets") or {}).get("SPREADS") or {}).get("selective_trust") or {}).get("veto_watchlist"),
                "totals_residual_model":(((_result.get("markets") or {}).get("TOTALS") or {}).get("selected_residual_model")),
                "totals_policy":(((_result.get("markets") or {}).get("TOTALS") or {}).get("betting_policy") or {}).get("status"),
                "totals_threshold":(((_result.get("markets") or {}).get("TOTALS") or {}).get("betting_policy") or {}).get("selected_threshold"),
                "totals_partial_pool_threshold":(((_result.get("markets") or {}).get("TOTALS") or {}).get("partial_pooling_policy") or {}).get("selected_threshold"),
                "h2h_policy":((_result.get("h2h") or {}).get("betting_policy") or {}).get("status"),
                "h2h_core_weight":(_result.get("h2h") or {}).get("selected_core_weight"),
                "h2h_min_ev":((_result.get("h2h") or {}).get("betting_policy") or {}).get("selected_min_ev"),
                "system_rules_index_uri":(((_system_report.get("artifacts") or {}).get("rules_index") or {}).get("uri")),
                "system_rules_pointer_uri":(((_system_report.get("artifacts") or {}).get("rules_pointer") or {}).get("uri")),
                "system_coverage_status":((_system_report.get("coverage_audit") or {}).get("status")),
                "system_registry_sha256":((_system_report.get("registry") or {}).get("registry_sha256")),
                "system_family_registry_sha256":((_system_report.get("registry") or {}).get("family_registry_sha256")),
                "prediction_tracker_status":((_system_report.get("prediction_tracker_external") or {}).get("status")),
                "prediction_tracker_spread_predictors":((_system_report.get("prediction_tracker_external") or {}).get("spread_predictor_count")),
                "prediction_tracker_total_predictors":((_system_report.get("prediction_tracker_external") or {}).get("total_predictor_count")),
                "prediction_tracker_external_mechanisms":((_system_report.get("prediction_tracker_external") or {}).get("external_mechanism_count")),
                "pt_incremental_status":(_pt_incremental or {}).get("status"),
                "pt_incremental_spread_selected":(((_pt_incremental or {}).get("spread") or {}).get("discovery_selected_challenger")),
                "pt_incremental_spread_signal":(((_pt_incremental or {}).get("spread") or {}).get("strict_incremental_signal")),
                "pt_incremental_totals_selected":(((_pt_incremental or {}).get("totals") or {}).get("discovery_selected_challenger")),
                "pt_incremental_totals_signal":(((_pt_incremental or {}).get("totals") or {}).get("strict_incremental_signal")),
                "prediction_tracker_production_authority":0,
                "spread_dormant_historical":len((((_system_report.get("miner") or {}).get("spreads") or {}).get("dormant_historical_rules") or [])),
                "totals_dormant_historical":len((((_system_report.get("miner") or {}).get("totals") or {}).get("dormant_historical_rules") or [])),
                "bigal6_present":"BA-NFL6" in (((_system_report.get("bigal") or {}).get("documented_close_reference") or {})),
                "deep_context_v316_status":((_system_report.get("deep_objective_context_v316") or {}).get("status")),
                "deep_result_quality_ready":((_system_report.get("deep_objective_context_v316") or {}).get("result_quality_ready")),
                "deep_recent3_ready":((_system_report.get("deep_objective_context_v316") or {}).get("recent3_ready")),
                "deep_resume_ready":((_system_report.get("deep_objective_context_v316") or {}).get("resume_ready")),
                "deep_matchup_ready":((_system_report.get("deep_objective_context_v316") or {}).get("matchup_ready")),
                "market_snapshot_ready":((_system_report.get("deep_objective_context_v316") or {}).get("market_snapshot_ready")),
                "market_snapshot_contract":((_system_report.get("deep_objective_context_v316") or {}).get("market_snapshot_contract")),
                "market_movement_authority":0,
                "bigal_weekly_observations":((_system_report.get("bigal_weekly_observation_ledger_v316") or {}).get("observations")),
                "bigal_line_tolerance_observations":((_system_report.get("bigal_weekly_observation_ledger_v316") or {}).get("line_tolerance_observations")),
                "bigal_same_game_multi_release_groups":((_system_report.get("bigal_weekly_observation_ledger_v316") or {}).get("same_game_multi_release_groups")),
                "persistent_library_status":((_system_report.get("persistent_system_library_v317") or {}).get("status")),
                "persistent_prior_qualified_families":((_system_report.get("persistent_system_library_v317") or {}).get("prior_qualified_families")),
                "persistent_recovered_exact_legit":((_system_report.get("persistent_system_library_v317") or {}).get("recovered_exact_legit")),
                "persistent_demoted_after_exact_revalidation":((_system_report.get("persistent_system_library_v317") or {}).get("demoted_after_exact_revalidation")),
                "persistent_legacy_contract_holds":((_system_report.get("persistent_system_library_v317") or {}).get("legacy_contract_holds")),
                "persistent_library_uri":((_system_report.get("persistent_library_persist_v317") or {}).get("uri")),
                "bigal_fingerprint_status":((_system_report.get("bigal_current_card_fingerprint_v317") or {}).get("status")),
                "bigal_fingerprinted_observations":((_system_report.get("bigal_current_card_fingerprint_v317") or {}).get("fingerprinted_observations")),
                "bigal_fingerprint_matched_games":((_system_report.get("bigal_current_card_fingerprint_v317") or {}).get("matched_games")),
                "bigal_fingerprint_atom_vocabulary_count":((_system_report.get("bigal_current_card_fingerprint_v317") or {}).get("atom_vocabulary_count")),
                "bigal_fingerprint_uri":((_system_report.get("bigal_current_card_fingerprint_v317") or {}).get("artifact_uri")),
                "bigal_2026_trigger_context_queried":bool(((_system_report.get("bigal_current_card_fingerprint_v317") or {}).get("year_2026_trigger_context_queried"))),
                "bigal_2026_qualification_queried":bool(((_system_report.get("bigal_current_card_fingerprint_v317") or {}).get("year_2026_qualification_queried"))),
            }, sort_keys=True, default=str))
            pw.emit("done", "NFL Heavy Research V3.17 complete: "+str(_result.get("status")), pct=1.0)
        except Exception as exc:
            pw.emit("error", "NFL Heavy Research V3.17 failed: "+str(exc)+"\n"+traceback.format_exc(), pct=1.0)
            raise
        return

    # NFL Challenger Research — permanent protected research/challenger route.
    # Uses the latest V1.9.4 structured research + residual Miner + fair-line/edge
    # scorecards. 2026 is never queried by this route and production authority stays zero.
    if str(sport).upper().strip() == "NFL" and str(market).lower().strip() == "nfl_research_engine":
        import json
        import importlib.util
        from pathlib import Path
        from google.cloud import bigquery
        _dir = Path(__file__).resolve().parent

        def _load_nfl_v194_exact(_name, _tag):
            import nfl_engine as _nfl_engine
            return _nfl_engine.load_component(_name, _tag)

        _audit = _load_nfl_v194_exact(
            "nfl_audit_v1",
            "nfl-audit-v1.3-prior-feature-provenance-20260930",
        )
        _contract = _load_nfl_v194_exact(
            "nfl_research_contract_v1",
            "nfl-research-contract-v1.9.2-protected-architecture-20261001",
        )
        _structured = _load_nfl_v194_exact(
            "nfl_structured_research_v2",
            "nfl-structured-research-v1.9.4-null-safe-season-forward-20261001",
        )
        _miner = _load_nfl_v194_exact(
            "nfl_residual_miner_v2",
            "nfl-residual-miner-v2.0-market-error-fdr-20261001",
        )
        _edge = _load_nfl_v194_exact(
            "nfl_edge_gate_v1",
            "nfl-edge-gate-v1.9.4-season-forward-dual-scorecard-20261001",
        )
        _ledger3 = _load_nfl_v194_exact(
            "nfl_prospective_ledger_v3",
            "nfl-prospective-ledger-v3-v1.9.4-edge-gate-shadow-20261001",
        )
        _engine2 = _load_nfl_v194_exact(
            "nfl_research_engine_v2",
            "nfl-research-engine-v1.9.4-edge-gate-manager-20261001",
        )
        pw.emit("audit", f"[NFL-CHALLENGER-RESEARCH] Protected challenger research start run={run_id}; history through 2025 only", pct=0.05)
        try:
            _contract.assert_contract()
            _audit_report = _audit.run_nfl_audit_v1(storage_client=gcs, bucket_name=bucket, log_func=log_func)
            if _audit_report.get("status") != "READY_FOR_OFFLINE_CHALLENGER_SANDBOX":
                raise RuntimeError("[NFL-V1.9.4-HOLD] PRECEDING_AUDIT_NOT_GREEN "+str(_audit_report.get("status")))
            _health = _ledger3.ledger_health_check(
                bigquery.Client(project="sharplogger"),
                service_identity_hint="sharp-train-sa@sharplogger.iam.gserviceaccount.com",
            )
            pw.emit("research", "[NFL-CHALLENGER-RESEARCH] Run null-safe structured research + Miner V2 + fair-line/edge dual scorecards + season-forward edge gates", pct=0.25)
            _result = _engine2.run_nfl_research_engine_v2(
                bq_client=bigquery.Client(project="sharplogger"),
                storage_client=gcs,
                bucket_name=bucket,
                audit_report=_audit_report,
                ledger_health=_health,
                log_func=log_func,
            )
            pw.emit("done", "NFL Challenger Research complete and frozen for prospective comparison: "+_result["status"], pct=1.0)
        except Exception as exc:
            pw.emit("error", "NFL Challenger Research failed: "+str(exc)+"\n"+traceback.format_exc(), pct=1.0)
            raise
        return

    # NFL Engine V3 / Production V2.6.1 — unified weekly update with frozen production-model betting authority.
    # One operator action: parity -> challenger refresh/reuse -> paired live score
    # -> model-authority decisions/settlement -> performance/promotion clocks.
    if str(sport).upper().strip() == "NFL" and str(market).lower().strip() == "nfl_production_weekly":
        import importlib.util
        from pathlib import Path
        from google.cloud import bigquery
        _dir = Path(__file__).resolve().parent

        def _load_nfl_weekly_exact(_name, _tag):
            import nfl_engine as _nfl_engine
            return _nfl_engine.load_component(_name, _tag)

        _parity = _load_nfl_weekly_exact("nfl_live_feature_parity_v1", "nfl-production-v1-live-feature-parity-v1.0.5-frozen-local-feature-contract-20261002")
        _prod = _load_nfl_weekly_exact("nfl_production_v1", "nfl-production-v1.1.1-publish-receipt-normalization-20261002")
        # Production MODEL -> ACTION must not be blocked by a research-only module.
        # nfl_production_live_v1 loads V2.5 shadow evidence opportunistically and
        # fails that lane open-to-diagnostics / closed-to-authority if unavailable.
        _market_backend = _load_nfl_weekly_exact("nfl_market_backend_v261", "nfl-market-backend-v2.6.1-utils-canonical-20261003")
        _modelauth = _load_nfl_weekly_exact("nfl_model_authority_v26", "nfl-model-authority-v2.6.2-v316-deep-live-parity-20261011")
        _live = _load_nfl_weekly_exact("nfl_production_live_v1", "nfl-production-v2.6.1-utils-market-backend-20261003")
        pw.emit("audit", f"[NFL-EDGE-V2] Weekly production update start run={run_id}", pct=0.05)
        try:
            _bq=bigquery.Client(project="sharplogger")
            _market_preflight=_market_backend.preflight(bq_client=_bq)
            log_func("[NFL-MARKET-BACKEND-V261-PREFLIGHT] "+json.dumps(_market_preflight,sort_keys=True,default=str))
            _parity_result=_parity.run_nfl_live_feature_parity_v1(bq_client=_bq,log_func=log_func)
            if _parity_result.get("status") != "NFL_PRODUCTION_V1_LIVE_FEATURE_PARITY_PASS":
                raise RuntimeError("[NFL-PROD-V1-WEEKLY-HOLD] LIVE_FEATURE_PARITY_NOT_GREEN "+str(_parity_result.get("status")))
            pw.emit("train", "Parity PASS; refresh/reuse weekly challenger", pct=0.30)
            _refresh=_prod.run_nfl_production_refresh(bq_client=_bq,storage_client=gcs,bucket_name=bucket,log_func=log_func)
            pw.emit("score", "Challenger ready; score paired ledger + frozen production-model betting authority and shadow evidence", pct=0.65)
            _score=_live.run_nfl_production_live_score(bq_client=_bq,storage_client=gcs,bucket_name=bucket,log_func=log_func)
            _result={"status":"NFL_PRODUCTION_V1_WEEKLY_UPDATE_PASS","refresh":_refresh,"score":_score}
            log_func("[NFL-PROD-V1-WEEKLY-CONTRACT] "+json.dumps({
                "status":_result["status"],
                "challenger_reused_existing":bool(_refresh.get("challenger_reused_existing",False)),
                "live_status":_score.get("status"),
                "market_backend":_market_preflight,
                "model_authority_v26":_score.get("model_authority_v26",{}),
                "edge_authority_v2":_score.get("edge_authority_v2",{}),
                "promotion_clock":_score.get("promotion_clock",{}),
                "automatic_promotion":False,
            },sort_keys=True,default=str))
            pw.emit("done", "NFL Production weekly update complete", pct=1.0)
        except Exception as exc:
            pw.emit("error", "NFL Production weekly update failed: "+str(exc)+"\n"+traceback.format_exc(), pct=1.0)
            raise
        return

    # NFL V2.6 historical validation — preserve V2.5 research, then freeze model-only betting policy.
    # One operator action now runs the complete protected research stack first:
    # audit -> structured CORE/STAT research -> residual Miner -> edge gates ->
    # direct System Miner V3/mechanism registry -> production replay -> authority.
    # 2026 remains sealed from all historical selection/confirmation work.
    if str(sport).upper().strip() == "NFL" and str(market).lower().strip() == "nfl_production_replay":
        import importlib.util
        from pathlib import Path
        from google.cloud import bigquery
        _dir = Path(__file__).resolve().parent

        def _load_nfl_prod_replay_exact(_name, _tag):
            import nfl_engine as _nfl_engine
            return _nfl_engine.load_component(_name, _tag)

        # Exact protected-research dependencies.  These checks deliberately make
        # a mixed deployment fail closed instead of silently shortening the run.
        _audit = _load_nfl_prod_replay_exact("nfl_audit_v1", "nfl-audit-v1.3-prior-feature-provenance-20260930")
        _rcontract = _load_nfl_prod_replay_exact("nfl_research_contract_v1", "nfl-research-contract-v1.9.2-protected-architecture-20261001")
        _structured = _load_nfl_prod_replay_exact("nfl_structured_research_v2", "nfl-structured-research-v1.9.4-null-safe-season-forward-20261001")
        _resid = _load_nfl_prod_replay_exact("nfl_residual_miner_v2", "nfl-residual-miner-v2.0-market-error-fdr-20261001")
        _egate = _load_nfl_prod_replay_exact("nfl_edge_gate_v1", "nfl-edge-gate-v1.9.4-season-forward-dual-scorecard-20261001")
        _ledger3 = _load_nfl_prod_replay_exact("nfl_prospective_ledger_v3", "nfl-prospective-ledger-v3-v1.9.4-edge-gate-shadow-20261001")
        _research = _load_nfl_prod_replay_exact("nfl_research_engine_v2", "nfl-research-engine-v1.9.4-edge-gate-manager-20261001")
        _syscontract = _load_nfl_prod_replay_exact("nfl_research_v2_contract", "nfl-research-v2.0-foundation-expansion-20261001")
        _systems = _load_nfl_prod_replay_exact("nfl_system_lab_v3", "nfl-system-lab-v3.17.1-bigal-fingerprint-detail-logging-20261011")
        _prod = _load_nfl_prod_replay_exact("nfl_production_v1", "nfl-production-v1.1.1-publish-receipt-normalization-20261002")
        _bet = _load_nfl_prod_replay_exact("nfl_betting_engine_v1", "nfl-betting-engine-v1.0-unified-decision-20261002")
        _shared = _load_nfl_prod_replay_exact("sports_edge_authority_v1", "sports-edge-authority-v1.1-dependency-aware-cross-sport-standard-20261002")
        _stat23 = _load_nfl_prod_replay_exact("nfl_stat_selector_v23", "nfl-stat-selector-v2.3-ncaaf-style-market-reliability-20261002")
        _stat24 = _load_nfl_prod_replay_exact("nfl_advanced_stat_research_v24", "nfl-advanced-stat-v2.4-orthogonal-hidden-signal-research-20261003")
        _multi25 = _load_nfl_prod_replay_exact("nfl_multidimensional_edge_v25", "nfl-multidimensional-edge-v2.5-four-lane-research-20261003")
        _pbpdiag = _load_nfl_prod_replay_exact("nfl_pbp_attribution_v1", "nfl-pbp-attribution-v1-research-v2.0.2-frozen-core2-20261001")
        _edge = _load_nfl_prod_replay_exact("nfl_edge_authority_v2", "nfl-edge-authority-v2.5-four-lane-multidimensional-research-20261003")
        _modelauth = _load_nfl_prod_replay_exact("nfl_model_authority_v26", "nfl-model-authority-v2.6.2-v316-deep-live-parity-20261011")
        _replay = _load_nfl_prod_replay_exact("nfl_production_replay_v1", "nfl-production-v2.6.1-historical-model-authority-freeze-20261003")

        pw.emit("audit", f"[NFL-MODEL-AUTH-V2.6] Historical validation start run={run_id}; preserve full V2.5 research then freeze MODEL -> ACTION policy; 2026 untouched", pct=0.03)
        try:
            _bq=bigquery.Client(project="sharplogger")
            _rcontract.assert_contract()
            _syscontract.assert_contract()
            _audit_report=_audit.run_nfl_audit_v1(storage_client=gcs,bucket_name=bucket,log_func=log_func)
            if _audit_report.get("status") != "READY_FOR_OFFLINE_CHALLENGER_SANDBOX":
                raise RuntimeError("[NFL-MODEL-AUTH-V2.6-HOLD] PRECEDING_AUDIT_NOT_GREEN "+str(_audit_report.get("status")))
            _health=_ledger3.ledger_health_check(_bq,service_identity_hint="sharp-train-sa@sharplogger.iam.gserviceaccount.com")

            pw.emit("research", "Audit PASS; run full CORE / STAT / residual / edge-gate research", pct=0.18)
            _research_report=_research.run_nfl_research_engine_v2(
                bq_client=_bq,storage_client=gcs,bucket_name=bucket,
                audit_report=_audit_report,ledger_health=_health,log_func=log_func,
            )

            pw.emit("systems", "CORE/STAT research complete; run full direct System Miner V3 and freeze mechanism-family registry", pct=0.48)
            _system_report=_systems.run_nfl_system_lab_v3(
                bq_client=_bq,storage_client=gcs,bucket_name=bucket,
                audit_report=_audit_report,log_func=log_func,
            )

            pw.emit("advanced_stats", "System registry frozen; run frozen PBP/EPA confirmation-veto diagnostic before orthogonal STAT research", pct=0.62)
            try:
                _pbp_report=_pbpdiag.run_nfl_pbp_attribution_v1(
                    bq_client=_bq,storage_client=gcs,bucket_name=bucket,
                    audit_report=_audit_report,log_func=log_func,
                )
            except Exception as _pbp_exc:
                # The frozen PBP artifact is an advanced research sidecar, not a
                # production dependency. Missing/stale PBP cannot block CORE/system replay.
                _pbp_report={"status":"PBP_ADVANCED_DIAGNOSTIC_UNAVAILABLE","error":f"{type(_pbp_exc).__name__}:{_pbp_exc}","production_authority":0}
                log_func("[NFL-STAT-V24-PBP-HOLD] "+json.dumps(_pbp_report,sort_keys=True,default=str))
            _research_report=dict(_research_report or {})
            _research_report["pbp_advanced_diagnostic_v24"]=_pbp_report

            pw.emit("replay", "PBP sidecar checked; run leak-safe replay + V2.5 research + freeze model-only betting thresholds on discovery/confirmation", pct=0.72)
            _result = _replay.run_nfl_production_historical_replay(
                bq_client=_bq,storage_client=gcs,bucket_name=bucket,log_func=log_func,
                research_report=_research_report,system_report=_system_report,
            )
            pw.emit("done", "NFL Model Authority V2.6.1 historical validation complete: "+_result["status"], pct=1.0)
        except Exception as exc:
            pw.emit("error", "NFL Model Authority V2.6.1 historical validation failed: "+str(exc)+"\n"+traceback.format_exc(), pct=1.0)
            raise
        return

    # NFL Research V2.0 — independent situational System Lab.
    # Reuses the NCAAF System Miner V3 research discipline but never imports
    # CORE/model predictions into system discovery.  Big Al, Pathi translations,
    # academic replications and mined/domain systems remain separately labelled.
    if str(sport).upper().strip() == "NFL" and str(market).lower().strip() == "nfl_system_lab":
        import importlib.util
        from pathlib import Path
        from google.cloud import bigquery
        _dir = Path(__file__).resolve().parent

        def _load_nfl_v2_system_exact(_name, _tag):
            import nfl_engine as _nfl_engine
            return _nfl_engine.load_component(_name, _tag)

        _contract_v2 = _load_nfl_v2_system_exact(
            "nfl_research_v2_contract",
            "nfl-research-v2.0-foundation-expansion-20261001",
        )
        _systems = _load_nfl_v2_system_exact(
            "nfl_system_lab_v3",
            "nfl-system-lab-v3.17.1-bigal-fingerprint-detail-logging-20261011",
        )
        _audit = _load_nfl_v2_system_exact(
            "nfl_audit_v1",
            "nfl-audit-v1.3-prior-feature-provenance-20260930",
        )
        pw.emit("research", f"[NFL-RESEARCH-V2-SYSTEM-V3] Start run={run_id}; direct ATS/OU system discovery=2017-2022; frozen validation=2023-2025; 2026 sealed", pct=0.05)
        try:
            _contract_v2.assert_contract()
            _audit_report = _audit.run_nfl_audit_v1(storage_client=gcs, bucket_name=bucket, log_func=log_func)
            if _audit_report.get("status") != "READY_FOR_OFFLINE_CHALLENGER_SANDBOX":
                raise RuntimeError("[NFL-RESEARCH-V2-SYSTEM-HOLD] PRECEDING_AUDIT_NOT_GREEN "+str(_audit_report.get("status")))
            pw.emit("research", "[NFL-RESEARCH-V2-SYSTEM-V310] Run Advanced System Miner V3.10.2 with horizon-symmetric one/two/three-game sequences, magnitude, opponent symmetry and team-memory lanes; retain established 2017-2022 discovery evidence even when later periods weaken, collapse correlated variants, add Big Al 6, publish human rules index/coverage audit, freeze prospective family registry", pct=0.20)
            _result = _systems.run_nfl_system_lab_v3(
                bq_client=bigquery.Client(project="sharplogger"),
                storage_client=gcs,
                bucket_name=bucket,
                audit_report=_audit_report,
                log_func=log_func,
            )
            pw.emit("done", "NFL System Research / Miner V3.2 complete: "+_result["status"], pct=1.0)
        except Exception as exc:
            pw.emit("error", "NFL Research V2 System Lab failed: "+str(exc)+"\n"+traceback.format_exc(), pct=1.0)
            raise
        return

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
    _aegv1, _aegv1_path, _aegv1_sha = _load_exact_local_module("atomic_edge_graph_v1")
    _arrv1, _arrv1_path, _arrv1_sha = _load_exact_local_module("atomic_rule_refinement_v1")
    _fsepv1, _fsepv1_path, _fsepv1_sha = _load_exact_local_module("frozen_spread_edge_policy_v1")
    _smev1, _smev1_path, _smev1_sha = _load_exact_local_module("sibling_market_edge_research_v1")
    _tarv1, _tarv1_path, _tarv1_sha = _load_exact_local_module("totals_atomic_refinement_v1")
    _rcv1, _rcv1_path, _rcv1_sha = _load_exact_local_module("refit_cadence_test_v1")
    _npv1, _npv1_path, _npv1_sha = _load_exact_local_module("ncaaf_production_v1")
    _nrv22, _nrv22_path, _nrv22_sha = _load_exact_local_module("ncaaf_research_v2")
    _nccv2, _nccv2_path, _nccv2_sha = _load_exact_local_module("ncaaf_core_challenger_v2")

    train_sharp_model_for_market = _wrapper.train_sharp_model_for_market
    train_timing_model_for_market = _wrapper.train_timing_model_for_market

    # NCAAF edge fast path. Research mode remains zero-authority.  The explicit
    # production-promotion mode runs the same leakage-safe evidence stack once,
    # then publishes only the frozen NCAAF Production V1 edge contract.  It does
    # NOT revive timing, generic AutoFS, multi-head training, or legacy artifact
    # publication.
    # NCAAF operator workflows mirror NFL: one weekly production refresh and one
    # protected heavy research run.  Legacy granular routes remain backend-only
    # aliases for recovery/testing, but are intentionally hidden from the UI.
    _ncaaf_weekly_run = bool(
        str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() == "ncaaf_production_weekly"
    )
    _ncaaf_research_heavy_run = bool(
        str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() in {"ncaaf_research_heavy", "ncaaf_v55_heavy"}
    )
    _ncaaf_rv22_run = bool(
        str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() == "ncaaf_research_v2"
    )
    _ncaaf_core_challenger_run = bool(
        str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() == "ncaaf_core_challenger"
    )
    _ncaaf_prod_promote = bool(
        str(sport).upper().strip() == "NCAAF" and (
            str(os.getenv("NCAAF_PROMOTE_EDGE_V1", "0")).strip().lower() in {"1","true","yes","on"}
            or str(market).lower().strip() in {"ncaaf_production","production_v1"}
        )
    )
    _edge_research_only = bool(
        str(sport).upper().strip() == "NCAAF" and (
            _ncaaf_prod_promote
            or str(os.getenv("NCAAF_EDGE_RESEARCH_ONLY", "0")).strip().lower() in {"1","true","yes","on"}
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
        if _emmv1_tag != "edge-mechanism-matrix-v1.1-peer-source-canonical-joinfix":
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
        _aegv1_tag = getattr(_aegv1, "ATOMIC_EDGE_GRAPH_V1_SOURCE_TAG", None)
        if _aegv1_tag != "atomic-edge-graph-v1-individual-rule-mechanism-stacks":
            raise RuntimeError(
                f"[ATOMIC-EDGE-V1-DEPLOY-PREFLIGHT] STALE_OR_MISSING source_tag={_aegv1_tag!r} "
                f"path={str(_aegv1_path)!r} sha={_aegv1_sha[:16]}"
            )
        log_func(
            f"[ATOMIC-EDGE-V1-DEPLOY-PREFLIGHT] PASS source_tag={_aegv1_tag} path={_aegv1_path} "
            f"sha={_aegv1_sha[:16]} production_authority=0"
        )
        _aegv1_out = _aegv1.run_atomic_edge_graph_v1(
            dashboard_module=_sld, stat_out=_scv21_out, registry_out=_erv2_out, mechanism_matrix_out=_emmv1_out,
            log_func=log_func, hard_fail=True
        )
        _arrv1_tag = getattr(_arrv1, "ATOMIC_RULE_REFINEMENT_V1_SOURCE_TAG", None)
        if _arrv1_tag != "atomic-rule-refinement-v1-play-fade-lineage-independence":
            raise RuntimeError(
                f"[RULE-REFINE-V1-DEPLOY-PREFLIGHT] STALE_OR_MISSING source_tag={_arrv1_tag!r} "
                f"path={str(_arrv1_path)!r} sha={_arrv1_sha[:16]}"
            )
        log_func(
            f"[RULE-REFINE-V1-DEPLOY-PREFLIGHT] PASS source_tag={_arrv1_tag} path={_arrv1_path} "
            f"sha={_arrv1_sha[:16]} production_authority=0"
        )
        _arrv1_out = _arrv1.run_atomic_rule_refinement_v1(
            dashboard_module=_sld, stat_out=_scv21_out, registry_out=_erv2_out, atomic_graph_out=_aegv1_out,
            log_func=log_func, hard_fail=True
        )
        _fsepv1_tag = getattr(_fsepv1, "FROZEN_SPREAD_EDGE_POLICY_V1_SOURCE_TAG", None)
        if _fsepv1_tag != "frozen-spread-edge-policy-v1-20260929":
            raise RuntimeError(
                f"[FROZEN-SPREAD-V1-DEPLOY-PREFLIGHT] STALE_OR_MISSING source_tag={_fsepv1_tag!r} "
                f"path={str(_fsepv1_path)!r} sha={_fsepv1_sha[:16]}"
            )
        log_func(
            f"[FROZEN-SPREAD-V1-DEPLOY-PREFLIGHT] PASS source_tag={_fsepv1_tag} path={_fsepv1_path} "
            f"sha={_fsepv1_sha[:16]} production_authority=0"
        )
        _fsepv1_out = _fsepv1.run_frozen_spread_edge_policy_v1(
            dashboard_module=_sld, stat_out=_scv21_out, registry_out=_erv2_out, refinement_out=_arrv1_out,
            log_func=log_func, hard_fail=True
        )
        _smev1_tag = getattr(_smev1, "SIBLING_MARKET_EDGE_RESEARCH_V1_SOURCE_TAG", None)
        if _smev1_tag != "sibling-market-edge-research-v1-h2h-totals-independent":
            raise RuntimeError(
                f"[SIBLING-EDGE-V1-DEPLOY-PREFLIGHT] STALE_OR_MISSING source_tag={_smev1_tag!r} "
                f"path={str(_smev1_path)!r} sha={_smev1_sha[:16]}"
            )
        log_func(
            f"[SIBLING-EDGE-V1-DEPLOY-PREFLIGHT] PASS source_tag={_smev1_tag} path={_smev1_path} "
            f"sha={_smev1_sha[:16]} production_authority=0"
        )
        _smev1_out = _smev1.run_sibling_market_edge_research_v1(
            dashboard_module=_sld, log_func=log_func, hard_fail=True
        )
        _tarv1_tag = getattr(_tarv1, "TOTALS_ATOMIC_REFINEMENT_V1_SOURCE_TAG", None)
        if _tarv1_tag != "totals-atomic-refinement-v1-play-fade-lineage-family-collapse":
            raise RuntimeError(
                f"[TOTALS-REFINE-V1-DEPLOY-PREFLIGHT] STALE_OR_MISSING source_tag={_tarv1_tag!r} "
                f"path={str(_tarv1_path)!r} sha={_tarv1_sha[:16]}"
            )
        log_func(
            f"[TOTALS-REFINE-V1-DEPLOY-PREFLIGHT] PASS source_tag={_tarv1_tag} path={_tarv1_path} "
            f"sha={_tarv1_sha[:16]} production_authority=0"
        )
        _tarv1_out = _tarv1.run_totals_atomic_refinement_v1(
            dashboard_module=_sld, sibling_out=_smev1_out, sibling_module=_smev1, log_func=log_func, hard_fail=True
        )
        _rcv1_tag = getattr(_rcv1, "REFIT_CADENCE_TEST_V1_SOURCE_TAG", None)
        if _rcv1_tag != "refit-cadence-test-v1-fixed-backbones-asof":
            raise RuntimeError(
                f"[CADENCE-V1-DEPLOY-PREFLIGHT] STALE_OR_MISSING source_tag={_rcv1_tag!r} "
                f"path={str(_rcv1_path)!r} sha={_rcv1_sha[:16]}"
            )
        log_func(
            f"[CADENCE-V1-DEPLOY-PREFLIGHT] PASS source_tag={_rcv1_tag} path={_rcv1_path} "
            f"sha={_rcv1_sha[:16]} production_authority=0"
        )
        _rcv1_out = _rcv1.run_refit_cadence_test_v1(
            dashboard_module=_sld, log_func=log_func, hard_fail=True
        )
        return _v143_out, _scv21_out, _erv2_out, _etv1_out, _ecv2_out, _emmv1_out, _aegv1_out, _arrv1_out, _fsepv1_out, _smev1_out, _tarv1_out, _rcv1_out

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
    _npv1_tag = getattr(_npv1, "NCAAF_PRODUCTION_V1_SOURCE_TAG", None)
    if _npv1_tag != "ncaaf-production-v1-fixed-backbones-edge-authority-20260929":
        raise RuntimeError(
            f"[NCAAF-PROD-V1-DEPLOY-PREFLIGHT] STALE_OR_MISSING source_tag={_npv1_tag!r} "
            f"path={str(_npv1_path)!r} sha={_npv1_sha[:16]}"
        )
    log_func(
        f"[NCAAF-PROD-V1-DEPLOY-PREFLIGHT] PASS source_tag={_npv1_tag} "
        f"path={_npv1_path} sha={_npv1_sha[:16]} promotion_requested={_ncaaf_prod_promote}"
    )
    _nrv22_tag = getattr(_nrv22, "NCAAF_RESEARCH_V2_SOURCE_TAG", None)
    if _nrv22_tag != "ncaaf-research-v2.33.0-feature-aware-max-history-miner-20261015":
        raise RuntimeError(
            f"[NCAAF-RV2330-DEPLOY-PREFLIGHT] STALE_OR_MISSING source_tag={_nrv22_tag!r} "
            f"path={str(_nrv22_path)!r} sha={_nrv22_sha[:16]}"
        )
    log_func(
        f"[NCAAF-RV2330-DEPLOY-PREFLIGHT] PASS source_tag={_nrv22_tag} "
        f"path={_nrv22_path} sha={_nrv22_sha[:16]} production_authority=0"
    )
    log_func(
        "[NCAAF-PT-GCS-MANUAL] mode=GCS_CSV_ONLY live_html_required=FALSE sparse_source=TRUE "
        "missing_game_policy=NO_EXTERNAL_SIGNAL partial_component_policy=PRESERVE_AVAILABLE "
        "meta_policy=EXACT_5_OF_5_ONLY current_external_consensus=META_SOURCE_CLUSTER_OUT_CLUSTER_BALANCED_MEDIAN_MIN8 full_index_policy=ALL_SOURCE_NATIVE_LINE_PREDICTORS_ONE_CORRELATED_FAMILY miner_policy=LEGACY_PLUS_SEPARATE_EXTERNAL_BEHAVIOR_LANE paid_proxy_required=FALSE production_authority=0"
    )
    _nccv2_tag = getattr(_nccv2, "SOURCE_TAG", None)
    if _nccv2_tag != "ncaaf-core-challenger-v2.4-expert-model-atom-bridge-20261006":
        raise RuntimeError(
            f"[NCAAF-CORE-CHALLENGER-V2-DEPLOY-PREFLIGHT] STALE_OR_MISSING source_tag={_nccv2_tag!r} "
            f"path={str(_nccv2_path)!r} sha={_nccv2_sha[:16]}"
        )
    log_func(
        f"[NCAAF-CORE-CHALLENGER-V2-DEPLOY-PREFLIGHT] PASS source_tag={_nccv2_tag} "
        f"path={_nccv2_path} sha={_nccv2_sha[:16]} production_authority=0 automatic_promotion=FALSE"
    )

    pw.emit("start", f"Training start run_id={run_id} sport={sport} market={market}", pct=0.0)

    hb_stop = start_heartbeat(pw, f"[{sport}] market={market}", 45)

    try:
        if _ncaaf_weekly_run:
            # Normal NCAAF production workflow.  Reuse the explicitly published
            # frozen Production V1 probability contract; do not refit, research,
            # republish, retune, or promote anything.  Settle prior immutable
            # ledger locks, then refresh the current slate from Utils/Move Master
            # through the exact production scorer and current bounded overlays.
            from google.cloud import bigquery as _bq
            _bq_client=_bq.Client(project="sharplogger")
            _contract=_npv1.load_production_contract(bucket_name=bucket,storage_client=gcs)
            if not isinstance(_contract,dict) or not _contract.get("_artifact_sha256"):
                raise RuntimeError(
                    "[NCAAF-WEEKLY-HOLD] Production V1 contract is missing. "
                    "Weekly Update cannot create or promote a production contract."
                )
            pw.emit("settlement","NCAAF Weekly Update: settle prior immutable production locks",pct=0.20)
            _settle=_utils.settle_ncaaf_production_v1(client=_bq_client)
            log_func(
                f"[NCAAF-WEEKLY-SETTLE] status={_settle.get('status')} "
                f"settled={int(_settle.get('settled',0) or 0)} artifact={str(_contract.get('_artifact_sha256'))[:16]}"
            )
            # Refresh the current validated manual GCS PT snapshot BEFORE the
            # betting vote pass. This remains read-only historical research context;
            # no source is scraped here and no CORE probability is modified.
            try:
                _pt=_nrv22.refresh_prediction_tracker_external(
                    dashboard_module=None,storage_client=gcs,bucket_name=bucket,
                    include_history=False,include_current=True,force=True,log_func=log_func)
                log_func("[NCAAF-WEEKLY-PT-REFRESH-PRE-SCORE] "+json.dumps({
                    "status":_pt.get("status"),"current":_pt.get("current"),
                    "production_mutation":False,"authority":0},default=str,sort_keys=True))
            except Exception as _pt_pre_exc:
                log_func(f"[NCAAF-WEEKLY-PT-REFRESH-PRE-SCORE] status=UNAVAILABLE error={type(_pt_pre_exc).__name__}:{_pt_pre_exc} authority=0")
            pw.emit("market","NCAAF Weekly Update: load current pregame market and refresh frozen production scoring",pct=0.50)
            _moves=_utils.read_recent_sharp_moves(
                hours=240,table=getattr(_utils,"DEFAULT_MOVES_VIEW","sharp_data.moves_with_features_merged"),
                pregame_only=True,sport="NCAAF",use_bq_storage=False,
            )
            _report=None
            # Load/read-only research evidence BEFORE Utils/Move Master scores
            # the slate; the production module consumes this exact report
            # through its shared decision function. Frozen CORE is not refit.
            try:
                import json as _ncaaf_live_json
                _report,_report_diag=_nrv22.load_current_report_diagnostic(
                    bucket_name=bucket,storage_client=gcs)
                _registered=_npv1.set_current_live_research_report(
                    _report,storage_client=gcs,bq_client=_bq_client,bucket_name=bucket)
                log_func("[NCAAF-WEEKLY-LIVE-REGISTRY-PRE-SCORE] "+_ncaaf_live_json.dumps({
                    **_report_diag,"installed":_registered,"production_mutation":False,
                    "core_frozen":True},sort_keys=True,default=str))
            except Exception as _ncaaf_live_err:
                _npv1.set_current_live_research_report(
                    None,storage_client=gcs,bq_client=_bq_client,bucket_name=bucket)
                log_func(f"[NCAAF-WEEKLY-LIVE-REGISTRY-PRE-SCORE] status=ERROR type={type(_ncaaf_live_err).__name__} detail={_ncaaf_live_err} core_frozen=TRUE")
            if _moves is None or getattr(_moves,"empty",True):
                _refresh={"status":"NO_CURRENT_MARKET","attempted":0,"inserted":0,"scored_rows":0,"selected_markets":0}
            else:
                _refresh=_utils.score_and_record_ncaaf_production_v1(
                    _moves,client=_bq_client,contract=_contract,
                    research_report=_report if isinstance(_report,dict) else None,
                    storage_client=gcs,return_picks=True)
            log_func(
                f"[NCAAF-WEEKLY-REFRESH] status={_refresh.get('status')} rows={0 if _moves is None else len(_moves)} "
                f"scored_rows={int(_refresh.get('scored_rows',0) or 0)} selected_markets={int(_refresh.get('selected_markets',0) or 0)} "
                f"attempted={int(_refresh.get('attempted',0) or 0)} inserted={int(_refresh.get('inserted',0) or 0)} "
                f"contract_source={_refresh.get('contract_source','NA')} artifact={_refresh.get('artifact',str(_contract.get('_artifact_sha256'))[:16])} "
                "probability_refit=FALSE research_run=FALSE production_publish=FALSE automatic_promotion=FALSE"
            )
            # Never mirror V1's already-selected picks into the V5.5 main
            # weekly artifact. That would overwrite V5.5's own CORE forecast.
            log_func('[NCAAF-V552-PRODUCTION-ISOLATION] frozen_v1_weekly=UNCHANGED '
                     'v55_board_not_overwritten=TRUE separate_models=TRUE')

            # The canonical selector already captured PT observations from the
            # exact selected spread rows. Here we verify coverage and settle, but
            # DO NOT run a second score_live_rows() or quote selection.
            try:
                _canonical=_refresh.pop("picks_frame",None)
                if _canonical is not None and not getattr(_canonical,"empty",True):
                    _sp=_canonical.loc[_canonical.get("Market").astype(str).str.lower().eq("spreads")]
                    log_func("[NCAAF-PT-CANONICAL-SELECTION] "+json.dumps({
                        "status":"PASS","all_market_rows":len(_canonical),
                        "spread_rows":len(_sp),
                        "core_spread_candidates":int(_sp.get("_core_qualifies").fillna(False).astype(bool).sum()) if "_core_qualifies" in _sp else 0,
                        "matches_ledger_selector":True,"duplicate_scoring":False,
                        "capture_occurs_in_shared_selector":True,
                    },sort_keys=True))
                else:
                    log_func("[NCAAF-PT-CANONICAL-SELECTION] status=NO_PICKS duplicate_scoring=FALSE")
                _pt_review=_nrv22.pt_promotion_settle_and_review(
                    storage_client=gcs,bucket_name=bucket,log_func=log_func)
                log_func("[NCAAF-PT-CANDIDATE-FORWARD-REVIEW] "+json.dumps(_pt_review,default=str,sort_keys=True))
            except Exception as _pt_live_exc:
                log_func(f"[NCAAF-PT-CANDIDATE-FORWARD-HOLD] {type(_pt_live_exc).__name__}:{_pt_live_exc}")

            # Surface registry availability because weekly scoring may use only
            # already-qualified live Miner definitions; it never changes them.
            try:
                _rr=_nrv22.load_current_report(bucket_name=bucket,storage_client=gcs)
                _miners=(_rr or {}).get("system_miner_v3") or {}
                _confirmed=sum(int((v or {}).get("confirmed_mechanism_count",0) or 0) for v in _miners.values())
                log_func(f"[NCAAF-WEEKLY-RESEARCH-REGISTRY] status={'READY' if isinstance(_rr,dict) else 'UNAVAILABLE'} confirmed_mechanisms={_confirmed} mutation=FALSE")
            except Exception as _reg_exc:
                log_func(f"[NCAAF-WEEKLY-RESEARCH-REGISTRY] status=UNAVAILABLE error={type(_reg_exc).__name__}:{_reg_exc} mutation=FALSE")
            # Prediction Tracker is now a validated manual-GCS sparse source. No paid
            # proxy/unblocker is required. Missing source games remain ordinary NCAAF games
            # with no external-rating signal.
            log_func("[NCAAF-PT-GCS-MANUAL] reason=WEEKLY refresh_network=FALSE missing_games_expected=TRUE authority=0")

            pw.emit("done","NCAAF Weekly Production Update complete ✅",pct=1.0)
            return

        if _ncaaf_research_heavy_run:
            # V5.5 ATS-FIRST HEAVY RESEARCH — PERMANENT PRIMARY CONTRACT.
            # A genuine chronological NONLINEAR_HGB replay and pregame book
            # quotes MUST validate before the legacy challenger / system lab
            # may run as an ancillary comparison.  The legacy CORE cache and
            # its old OOF margins are NEVER acceptable V5.5 stand-ins.
            import ncaaf_v55_primary_research as _v55_ats_primary
            pw.emit("v55_ats", "NCAAF Heavy: V5.5 chronological OOS point-edge / ATS / independent-brain study (primary)", pct=0.04)
            _v55_primary = _v55_ats_primary.run_cloud(
                storage_client=gcs, bucket_name=bucket, log_func=log_func,
            )
            _v55_primary_pass = bool(
                _v55_primary.get("status") == "COMPLETE_V55_ATS_PRIMARY_RESEARCH"
                and _v55_primary.get("production_modified") is False
                and _v55_primary.get("bet_authority") == 0
            )
            if not _v55_primary_pass:
                log_func("[NCAAF-HEAVY-V55-PRIMARY-HOLD] V5.5 ATS research is NOT "
                         "complete; legacy research will continue as a secondary lane "
                         "without using its OOF predictions as V5.5 evidence: "
                         + str(_v55_primary.get("status")) + " "
                         + str(_v55_primary.get("reason", "")))
            else:
                log_func("[NCAAF-HEAVY-V55-PRIMARY] status=PASS legacy_pipeline=SECONDARY "
                         "policy_promotion=FALSE production_mutation=FALSE")
            # One protected research workflow, analogous to NFL Heavy Research.
            # Build the leakage-safe historical cache once, then run both the
            # current STAT/System Miner research and the compact CORE challenger.
            # 2026 remains sealed from discovery/confirmation and no production
            # artifact or Bet Authority policy is mutated automatically.
            log_func("[NCAAF-HEAVY-RUN] phase=HISTORICAL_CACHE start=TRUE production_mutation=FALSE year_2026_selection=FALSE")
            _t0=__import__('time').perf_counter()
            pw.emit("cache","NCAAF Heavy secondary legacy diagnostics: historical/OOF cache",pct=0.45)
            _sld.fit_historical_ncaaf_core_expert("spreads",log_func=log_func)
            _sld.fit_ncaaf_statistical_brain(log_func=log_func)
            _cache=getattr(_sld,"_V1357_SPREAD_RESEARCH_CACHE",{}) or {}
            _games=_cache.get("games") if isinstance(_cache,dict) else None
            _features=_cache.get("candidate_feature_cols") if isinstance(_cache,dict) else None
            _miner_games=_cache.get("miner_games") if isinstance(_cache,dict) else None
            if _games is None or getattr(_games,"empty",True) or not _features:
                raise RuntimeError("[NCAAF-HEAVY-CACHE] historical game frame/candidate features missing")
            if _miner_games is None or getattr(_miner_games,"empty",True):
                raise RuntimeError("[NCAAF-HEAVY-CACHE] miner_games cache missing")
            log_func(f"[NCAAF-HEAVY-CACHE] status=PASS games={len(_games)} miner_games={len(_miner_games)} candidate_features={len(_features)}")

            # V2.11 ordering is intentional: CORE challenger first publishes leakage-safe
            # season-forward incumbent/specialist state into the research-only Miner frame.
            # The same existing Miner then tests those states together with Pathi/Big Al.
            pw.emit("core","NCAAF Heavy secondary legacy CORE challenger (not V5.5)",pct=0.64)
            _core=_nccv2.run_ncaaf_core_challenger_v2(
                dashboard_module=_sld,bucket_name=bucket,storage_client=gcs,log_func=log_func,hard_fail=True
            )
            if not isinstance(_core,dict) or _core.get("status")!="NCAAF_CORE_CHALLENGER_V2_COMPLETE":
                raise RuntimeError(f"[NCAAF-HEAVY-RUN] core challenger failed status={getattr(_core,'get',lambda *_:None)('status')}")

            pw.emit("systems","NCAAF Heavy ancillary persistent Miner, Big Al, Pathi and PT research",pct=0.80)
            # Prediction Tracker is read only from validated GCS uploads. It is intentionally
            # sparse: absence from PT never removes a game and partial named-system coverage
            # remains usable as component research while META still requires exact 5-of-5.
            log_func("[NCAAF-PT-GCS-MANUAL] reason=HEAVY refresh_network=FALSE missing_games_expected=TRUE sparse=TRUE authority=0")
            _research=_nrv22.run_ncaaf_research_v2(
                dashboard_module=_sld,utils_module=_utils,bucket_name=bucket,
                storage_client=gcs,production_module=_npv1,log_func=log_func,hard_fail=True
            )
            if not isinstance(_research,dict) or _research.get("status")!="NCAAF_RESEARCH_V2_COMPLETE":
                raise RuntimeError(f"[NCAAF-HEAVY-RUN] research report failed status={getattr(_research,'get',lambda *_:None)('status')}")
            # NCAAF_FULL_HISTORY_COMBINED_BRAIN_SHADOW_HOOK_V1
            # Research-only: 2013-2025 all-brain history, never live bet authority.
            # Missing archival events/eligibility remain HOLD, not inferred non-triggers.
            try:
                from ncaaf_heavy_brain_bridge import run_heavy_brain_shadow as _run_full_history_brains
                _historical_brains = _run_full_history_brains(
                    storage_client=gcs, bucket_name=bucket, log_func=log_func,
                    legacy_cache=_cache, legacy_research=_research,
                    v55_primary_result=_v55_primary,
                )
                log_func("[NCAAF-HEAVY-COMBINED-BRAINS] status="
                         + str(_historical_brains.get("status", "HOLD"))
                         + " reason=" + str(_historical_brains.get("reason", "UNKNOWN"))
                         + " bet_authority=0 production_mutation=FALSE")
            except Exception as _brain_shadow_exc:
                log_func("[NCAAF-HEAVY-COMBINED-BRAINS] status=HOLD type="
                         + type(_brain_shadow_exc).__name__
                         + " reason=" + str(_brain_shadow_exc)[:250]
                         + " bet_authority=0 production_mutation=FALSE")
            _miners=_research.get("system_miner_v3") or {}
            _confirmed=sum(int((v or {}).get("confirmed_mechanism_count",0) or 0) for v in _miners.values())
            _best=_core.get("best_challenger") or {}
            _attr=_core.get("conditional_specialist_attribution") or {}
            _attr_q=len(_attr.get("qualified_shadow_regimes") or []) if isinstance(_attr,dict) else 0
            _ptinc_ncaaf=_research.get("pt_incremental_value") or {}
            _t1=__import__('time').perf_counter()
            log_func(
                f"[NCAAF-HEAVY-RUN] status=PASS seconds={_t1-_t0:.1f} confirmed_mechanisms={_confirmed} "
                f"best_core={_best.get('name')} core_state={_best.get('state')} core_recommendation={_core.get('recommendation')} "
                f"conditional_specialist_shadow_qualified={_attr_q} "
                f"pt_incremental_status={_ptinc_ncaaf.get('status')} pt_selected={_ptinc_ncaaf.get('discovery_selected_challenger')} "
                f"pt_strict_incremental_signal={_ptinc_ncaaf.get('strict_incremental_signal')} "
                "2026_selection_influence=0 production_mutation=FALSE automatic_promotion=FALSE"
            )
            if _v55_primary_pass:
                pw.emit("done", "NCAAF V5.5 ATS Primary + Legacy Systems Research complete", pct=1.0)
            else:
                log_func("[NCAAF-HEAVY-OVERALL] status=V55_PRIMARY_HOLD "
                         "legacy_systems=COMPLETE secondary_only=TRUE "
                         "v55_ats_results=UNAVAILABLE betting_policy_promotion=FALSE")
                pw.emit("done", "NCAAF Heavy: legacy research complete, V5.5 ATS PRIMARY HELD pending genuine replay", pct=1.0)
            return

        if _ncaaf_core_challenger_run:
            # Protected NCAAF CORE challenger.  Build the same leakage-safe game frame
            # used by Production V1, then hand only <=2025 rows to the challenger.
            # Production V1 is never mutated and 2026 is sealed from selection/confirmation.
            log_func("[NCAAF-CORE-CHALLENGER-RUN] phase=HISTORICAL_CACHE start=TRUE production_mutation=FALSE year_2026_selection=FALSE")
            _t0=__import__('time').perf_counter()
            _sld.fit_historical_ncaaf_core_expert("spreads",log_func=log_func)
            _sld.fit_ncaaf_statistical_brain(log_func=log_func)
            _cache=getattr(_sld,"_V1357_SPREAD_RESEARCH_CACHE",{}) or {}
            _games=_cache.get("games") if isinstance(_cache,dict) else None
            _features=_cache.get("candidate_feature_cols") if isinstance(_cache,dict) else None
            if _games is None or getattr(_games,"empty",True) or not _features:
                raise RuntimeError("[NCAAF-CORE-CHALLENGER-CACHE] historical game frame/candidate features missing")
            log_func(f"[NCAAF-CORE-CHALLENGER-CACHE] status=PASS games={len(_games)} candidate_features={len(_features)}")
            _report=_nccv2.run_ncaaf_core_challenger_v2(
                dashboard_module=_sld,bucket_name=bucket,storage_client=gcs,log_func=log_func,hard_fail=True
            )
            if not isinstance(_report,dict) or _report.get("status")!="NCAAF_CORE_CHALLENGER_V2_COMPLETE":
                raise RuntimeError(f"[NCAAF-CORE-CHALLENGER-RUN] report failed status={getattr(_report,'get',lambda *_:None)('status')}")
            _t1=__import__('time').perf_counter()
            _best=_report.get("best_challenger") or {}
            log_func(
                f"[NCAAF-CORE-CHALLENGER-RUN] status=PASS seconds={_t1-_t0:.1f} best={_best.get('name')} "
                f"state={_best.get('state')} recommendation={_report.get('recommendation')} "
                "year_2026_queried=FALSE production_mutation=FALSE automatic_promotion=FALSE"
            )
            pw.emit("done","NCAAF CORE Expert/Specialist Challenger Search complete ✅",pct=1.0)
            return

        if _ncaaf_rv22_run:
            # Dedicated protected NCAAF Research V2.3 route. Build only the
            # historical/OOF caches required by the challenger; never publish or
            # mutate the frozen Production V1 probability/edge artifact.
            log_func("[NCAAF-RV22-RUN] phase=HISTORICAL_CACHE start=TRUE production_mutation=FALSE")
            _t0=__import__('time').perf_counter()
            _sld.fit_historical_ncaaf_core_expert("spreads",log_func=log_func)
            _sld.fit_ncaaf_statistical_brain(log_func=log_func)
            _cache=getattr(_sld,"_V1357_SPREAD_RESEARCH_CACHE",{}) or {}
            _games=_cache.get("games") if isinstance(_cache,dict) else None
            _miner_games=_cache.get("miner_games") if isinstance(_cache,dict) else None
            if _games is None or getattr(_games,"empty",True):
                raise RuntimeError("[NCAAF-RV22-CACHE] historical research games cache missing")
            if _miner_games is None or getattr(_miner_games,"empty",True):
                raise RuntimeError("[NCAAF-RV22-CACHE] miner_games cache missing")
            log_func(f"[NCAAF-RV22-CACHE] status=PASS games={len(_games)} miner_games={len(_miner_games)}")
            _report=_nrv22.run_ncaaf_research_v2(
                dashboard_module=_sld,utils_module=_utils,bucket_name=bucket,
                storage_client=gcs,production_module=_npv1,log_func=log_func,hard_fail=True
            )
            if not isinstance(_report,dict) or _report.get("status")!="NCAAF_RESEARCH_V2_COMPLETE":
                raise RuntimeError(f"[NCAAF-RV22-RUN] report failed status={getattr(_report,'get',lambda *_:None)('status')}")
            _t1=__import__('time').perf_counter()
            _miners=_report.get("system_miner_v3") or {}
            _confirmed=sum(int((v or {}).get("confirmed_mechanism_count",0) or 0) for v in _miners.values())
            log_func(
                f"[NCAAF-RV22-RUN] status=PASS seconds={_t1-_t0:.1f} confirmed_mechanisms={_confirmed} "
                "miner=V4_HORIZON_SYMMETRIC live_bridge=PUBLISHED 2026_selection_influence=0 production_mutation=FALSE"
            )
            pw.emit("done","NCAAF Research V2.3 complete ✅",pct=1.0)
            return

        if _edge_research_only:
            # FAST RESEARCH MODE: build only the upstream historical caches that
            # the edge-research stack consumes. Do not run timing, H2H/Totals
            # production training, AutoFS heads, promotion replay, or artifact
            # publication. This keeps normal production behavior unchanged when
            # the flag is off.
            log_func(
                "[EDGE-RESEARCH-FAST-PREFLIGHT] status=PASS sport=NCAAF "
                f"scope={'NCAAF_PRODUCTION_V1_PROMOTION' if _ncaaf_prod_promote else 'MULTI_MARKET_EDGE_RESEARCH'} "
                "spread=FROZEN_POLICY h2h=MODEL_ONLY totals=FAMILY_COLLAPSED skips=TIMING,H2H_PRODUCTION,TOTALS_PRODUCTION,"
                f"GENERIC_AUTOFS,PROMOTION_REPLAY artifact_publication={'EDGE_CONTRACT_ONLY' if _ncaaf_prod_promote else 'FALSE'} "
                f"production_authority={1 if _ncaaf_prod_promote else 0}"
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
            _edge_outputs = _run_ncaaf_edge_research_stack()
            if _ncaaf_prod_promote:
                (_v143_out, _scv21_out, _erv2_out, _etv1_out, _ecv2_out, _emmv1_out,
                 _aegv1_out, _arrv1_out, _fsepv1_out, _smev1_out, _tarv1_out, _rcv1_out) = _edge_outputs
                _npv1.build_production_contract(
                    dashboard_module=_sld, stat_module=_scv21, stat_out=_scv21_out,
                    frozen_spread_out=_fsepv1_out, totals_out=_tarv1_out,
                    cadence_module=_rcv1, cadence_out=_rcv1_out,
                    bucket_name=bucket, storage_client=gcs, log_func=log_func
                )
                log_func(
                    "[NCAAF-PROD-V1-CONTRACT] status=PASS probability_models=FROZEN_AND_SEPARATE "
                    "probability_runtime=FIXED_FEATURE_CONTRACT spread_features=3 totals_features=2 h2h_features=8 "
                    "spread_edge_authority=PROMOTED totals_edge_authority=PROMOTED_SINGLE_FAMILY "
                    "h2h_edge_authority=CLOSED cadence=FROZEN generic_autofs_runtime=RETIRED multi_head_runtime=RETIRED production_authority=1"
                )
            _t3 = __import__('time').perf_counter()
            log_func(
                f"[EDGE-RESEARCH-FAST-CONTRACT] status=PASS total_seconds={_t3-_t0:.1f} "
                f"legacy_production_training_skipped=TRUE compact_fixed_backbones={'FIT_AND_PUBLISHED' if _ncaaf_prod_promote else 'NOT_PUBLISHED'} artifact_publication={'NCAAF_PRODUCTION_V1' if _ncaaf_prod_promote else 'FALSE'} "
                f"production_authority={1 if _ncaaf_prod_promote else 0}"
            )
            log_func(
                f"[CODE-LIFECYCLE-AUDIT] status=PASS active_production={'NCAAF_PRODUCTION_V1' if _ncaaf_prod_promote else 'UNCHANGED'} "
                "active_research=V14.3_RELIABILITY,STAT_COMBINATION_V2_1,EDGE_REGISTRY_V2,EDGE_TOPOLOGY_V1,EDGE_COMPLEMENTARITY_V2,EDGE_MECHANISM_MATRIX_V1,ATOMIC_EDGE_GRAPH_V1,ATOMIC_RULE_REFINEMENT_V1,FROZEN_SPREAD_EDGE_POLICY_V1,SIBLING_MARKET_EDGE_RESEARCH_V1,TOTALS_ATOMIC_REFINEMENT_V1,SYSTEM_MINER "
                "fast_path=EDGE_RESEARCH_ONLY skipped_legacy_runtime=TIMING,H2H_PRODUCTION,TOTALS_PRODUCTION,"
                f"GENERIC_AUTOFS,PROMOTION_REPLAY legacy_artifact_publication=SKIPPED production_contract={'NCAAF_V1_PROMOTED' if _ncaaf_prod_promote else 'UNCHANGED'}"
            )
            pw.emit("done", "NCAAF Production V1 contract published ✅" if _ncaaf_prod_promote else "Edge research complete ✅", pct=1.0)
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
                "active_research=V14.3_RELIABILITY,STAT_COMBINATION_V2_1,EDGE_REGISTRY_V2,EDGE_TOPOLOGY_V1,EDGE_COMPLEMENTARITY_V2,EDGE_MECHANISM_MATRIX_V1,ATOMIC_EDGE_GRAPH_V1,ATOMIC_RULE_REFINEMENT_V1,FROZEN_SPREAD_EDGE_POLICY_V1,SIBLING_MARKET_EDGE_RESEARCH_V1,SYSTEM_MINER,H2H_SIBLINGS "
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
