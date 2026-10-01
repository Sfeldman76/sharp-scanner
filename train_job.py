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

    # NFL AUDIT V1: isolate read-only inventory from heavyweight NCAAF research
    # imports and every legacy NFL training / model publication path.
    if str(sport).upper().strip() == "NFL" and str(market).lower().strip() == "nfl_audit":
        import importlib.util
        from pathlib import Path
        _audit_path = Path(__file__).resolve().parent / "nfl_audit_v1.py"
        if not _audit_path.is_file():
            raise RuntimeError(f"[NFL-AUDIT-V1-DEPLOY-PREFLIGHT] MISSING {_audit_path}")
        _spec = importlib.util.spec_from_file_location("nfl_audit_v1", _audit_path)
        _audit = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_audit)
        if getattr(_audit, "SOURCE_TAG", "") != "nfl-audit-v1.3-prior-feature-provenance-20260930":
            raise RuntimeError("[NFL-AUDIT-V1-DEPLOY-PREFLIGHT] STALE_OR_MIXED_SOURCE")
        _feature_path = Path(__file__).resolve().parent / "nfl_feature_audit_v1.py"
        if not _feature_path.is_file():
            raise RuntimeError(f"[NFL-FEATURE-V1-DEPLOY-PREFLIGHT] MISSING {_feature_path}")
        # Load and register this exact local module for the audit's import.
        # Use the module-level sys import: importing sys inside main() creates
        # a local binding that breaks the earlier NFL challenger nested loader.
        _feature_spec = importlib.util.spec_from_file_location("nfl_feature_audit_v1", _feature_path)
        _feature = importlib.util.module_from_spec(_feature_spec)
        sys.modules["nfl_feature_audit_v1"] = _feature
        _feature_spec.loader.exec_module(_feature)
        if getattr(_feature, "SOURCE_TAG", "") != "nfl-feature-audit-v1.3-prior-only-20260930":
            raise RuntimeError("[NFL-FEATURE-V1-DEPLOY-PREFLIGHT] STALE_OR_MIXED_SOURCE")
        pw.emit("audit", f"[NFL-AUDIT-V1] Read-only NFL inventory start run={run_id}", pct=0.1)
        try:
            result = _audit.run_nfl_audit_v1(storage_client=gcs, bucket_name=bucket, log_func=log_func)
            pw.emit("done", f"NFL audit complete: {result['status']} (no model published)", pct=1.0)
        except Exception as exc:
            pw.emit("error", f"NFL audit failed: {exc}\n{traceback.format_exc()}", pct=1.0)
            raise
        return

    # NFL Challenger V1.5 — specialized score-domain sandbox; audit reruns in SAME job.
    # It never writes production artifacts, alters NCAAF, or enters legacy trainer.
    if str(sport).upper().strip() == "NFL" and str(market).lower().strip() == "nfl_challenger":
        import importlib.util
        from pathlib import Path
        from google.cloud import bigquery
        _dir = Path(__file__).resolve().parent

        def _load_nfl_exact(_name, _tag):
            _path = _dir / (_name + ".py")
            if not _path.is_file():
                raise RuntimeError(f"[NFL-CHALLENGER-V1-DEPLOY-PREFLIGHT] MISSING {_path}")
            _spec = importlib.util.spec_from_file_location(_name, _path)
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules[_name] = _mod
            _spec.loader.exec_module(_mod)
            if getattr(_mod, "SOURCE_TAG", "") != _tag:
                raise RuntimeError("[NFL-CHALLENGER-V1-DEPLOY-PREFLIGHT] STALE_OR_MIXED_"+_name)
            return _mod

        _feature = _load_nfl_exact("nfl_feature_audit_v1", "nfl-feature-audit-v1.3-prior-only-20260930")
        _audit = _load_nfl_exact("nfl_audit_v1", "nfl-audit-v1.3-prior-feature-provenance-20260930")
        _challenge = _load_nfl_exact("nfl_challenger_v1", "nfl-challenger-v1.4-season-forward-three-market-no-publish-20260930")
        _special = _load_nfl_exact("nfl_specialized_v1", "nfl-specialized-v1.5-score-domain-h2h-stack-20260930")
        pw.emit("audit", f"[NFL-CHALLENGER-V1] Recheck historical and prior-only audits run={run_id}", pct=0.05)
        try:
            _audit_report = _audit.run_nfl_audit_v1(storage_client=gcs, bucket_name=bucket, log_func=log_func)
            if _audit_report.get("status") != "READY_FOR_OFFLINE_CHALLENGER_SANDBOX":
                raise RuntimeError("[NFL-CHALLENGER-V1-HOLD] PRECEDING_AUDIT_NOT_GREEN "+str(_audit_report.get("status")))
            pw.emit("sandbox", "[NFL-SPECIALIZED-V1] H2H blend + Spread margin + Totals points; 2026 sealed", pct=0.37)
            _result = _special.run_nfl_specialized_v1(bq_client=bigquery.Client(project="sharplogger"),
                        audit_report=_audit_report, log_func=log_func)
            pw.emit("done", "NFL specialized modeling complete: "+_result["status"]+" (no model published)", pct=1.0)
        except Exception as exc:
            pw.emit("error", "NFL challenger failed: "+str(exc)+"\n"+traceback.format_exc(), pct=1.0)
            raise
        return

    # NFL Score Engine V1.6 — team offense + opponent defense -> projected score.
    # Research only: audit reruns in the same job; 2026 remains sealed.
    if str(sport).upper().strip() == "NFL" and str(market).lower().strip() == "nfl_score_engine":
        import importlib.util
        from pathlib import Path
        from google.cloud import bigquery
        _dir = Path(__file__).resolve().parent

        def _load_nfl_score_exact(_name, _tag):
            _path = _dir / (_name + ".py")
            if not _path.is_file():
                raise RuntimeError(f"[NFL-SCORE-V1-DEPLOY-PREFLIGHT] MISSING {_path}")
            _spec = importlib.util.spec_from_file_location(_name, _path)
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules[_name] = _mod
            _spec.loader.exec_module(_mod)
            if getattr(_mod, "SOURCE_TAG", "") != _tag:
                raise RuntimeError("[NFL-SCORE-V1-DEPLOY-PREFLIGHT] STALE_OR_MIXED_"+_name)
            return _mod

        _feature = _load_nfl_score_exact("nfl_feature_audit_v1", "nfl-feature-audit-v1.3-prior-only-20260930")
        _audit = _load_nfl_score_exact("nfl_audit_v1", "nfl-audit-v1.3-prior-feature-provenance-20260930")
        _challenge = _load_nfl_score_exact("nfl_challenger_v1", "nfl-challenger-v1.4-season-forward-three-market-no-publish-20260930")
        _special = _load_nfl_score_exact("nfl_specialized_v1", "nfl-specialized-v1.5-score-domain-h2h-stack-20260930")
        _score = _load_nfl_score_exact("nfl_score_engine_v1", "nfl-score-engine-v1.6-team-offense-defense-20260930")
        pw.emit("audit", f"[NFL-SCORE-V1] Recheck historical and prior-only audits run={run_id}", pct=0.05)
        try:
            _audit_report = _audit.run_nfl_audit_v1(storage_client=gcs, bucket_name=bucket, log_func=log_func)
            if _audit_report.get("status") != "READY_FOR_OFFLINE_CHALLENGER_SANDBOX":
                raise RuntimeError("[NFL-SCORE-V1-HOLD] PRECEDING_AUDIT_NOT_GREEN "+str(_audit_report.get("status")))
            pw.emit("sandbox", "[NFL-SCORE-V1] Team offense + opponent defense score engine; 2026 sealed", pct=0.37)
            _result = _score.run_nfl_score_engine_v1(
                bq_client=bigquery.Client(project="sharplogger"),
                audit_report=_audit_report, log_func=log_func)
            pw.emit("done", "NFL score-engine modeling complete: "+_result["status"]+" (no model published)", pct=1.0)
        except Exception as exc:
            pw.emit("error", "NFL score engine failed: "+str(exc)+"\n"+traceback.format_exc(), pct=1.0)
            raise
        return

    # NFL Intelligence V1.8 — residual/error research + market arbitration +
    # stricter system miner + within-family reconciliation. Research only.
    if str(sport).upper().strip() == "NFL" and str(market).lower().strip() == "nfl_intelligence":
        import importlib.util
        from pathlib import Path
        from google.cloud import bigquery
        _dir = Path(__file__).resolve().parent

        def _load_nfl_intel_exact(_name, _tag):
            _path = _dir / (_name + ".py")
            if not _path.is_file():
                raise RuntimeError(f"[NFL-INTEL-V1-DEPLOY-PREFLIGHT] MISSING {_path}")
            _spec = importlib.util.spec_from_file_location(_name, _path)
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules[_name] = _mod
            _spec.loader.exec_module(_mod)
            if getattr(_mod, "SOURCE_TAG", "") != _tag:
                raise RuntimeError("[NFL-INTEL-V1-DEPLOY-PREFLIGHT] STALE_OR_MIXED_"+_name)
            return _mod

        _feature = _load_nfl_intel_exact("nfl_feature_audit_v1", "nfl-feature-audit-v1.3-prior-only-20260930")
        _audit = _load_nfl_intel_exact("nfl_audit_v1", "nfl-audit-v1.3-prior-feature-provenance-20260930")
        _challenge = _load_nfl_intel_exact("nfl_challenger_v1", "nfl-challenger-v1.4-season-forward-three-market-no-publish-20260930")
        _special = _load_nfl_intel_exact("nfl_specialized_v1", "nfl-specialized-v1.5-score-domain-h2h-stack-20260930")
        _score = _load_nfl_intel_exact("nfl_score_engine_v1", "nfl-score-engine-v1.6-team-offense-defense-20260930")
        _intel = _load_nfl_intel_exact("nfl_intelligence_v1", "nfl-intelligence-v1.8-residual-arbitration-reconciliation-20260930")
        pw.emit("audit", f"[NFL-INTEL-V1.8] Recheck historical and prior-only audits run={run_id}", pct=0.05)
        try:
            _audit_report = _audit.run_nfl_audit_v1(storage_client=gcs, bucket_name=bucket, log_func=log_func)
            if _audit_report.get("status") != "READY_FOR_OFFLINE_CHALLENGER_SANDBOX":
                raise RuntimeError("[NFL-INTEL-V1-HOLD] PRECEDING_AUDIT_NOT_GREEN "+str(_audit_report.get("status")))
            pw.emit("sandbox", "[NFL-INTEL-V1.8] CORE residuals + market arbitration + Big Al + Pathi + stricter Miner; 2026 sealed", pct=0.37)
            _result = _intel.run_nfl_intelligence_v1(
                bq_client=bigquery.Client(project="sharplogger"),
                audit_report=_audit_report, log_func=log_func)
            pw.emit("done", "NFL intelligence research complete: "+_result["status"]+" (no model/system published)", pct=1.0)
        except Exception as exc:
            pw.emit("error", "NFL intelligence research failed: "+str(exc)+"\n"+traceback.format_exc(), pct=1.0)
            raise
        return

    # NFL V1.9.1 — expanded stats-only frozen 2026 holdout confirmation. The registry
    # is defined and hashed before 2026 is queried. No 2026 row can tune anything.
    if str(sport).upper().strip() == "NFL" and str(market).lower().strip() == "nfl_frozen_confirmation":
        import importlib.util
        from pathlib import Path
        from google.cloud import bigquery
        _dir = Path(__file__).resolve().parent

        def _load_nfl_v19_exact(_name, _tag):
            _path = _dir / (_name + ".py")
            if not _path.is_file():
                raise RuntimeError(f"[NFL-V1.9.1-DEPLOY-PREFLIGHT] MISSING {_path}")
            _spec = importlib.util.spec_from_file_location(_name, _path)
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules[_name] = _mod
            _spec.loader.exec_module(_mod)
            if getattr(_mod, "SOURCE_TAG", "") != _tag:
                raise RuntimeError("[NFL-V1.9.1-DEPLOY-PREFLIGHT] STALE_OR_MIXED_"+_name)
            return _mod

        _feature = _load_nfl_v19_exact("nfl_feature_audit_v1", "nfl-feature-audit-v1.3-prior-only-20260930")
        _audit = _load_nfl_v19_exact("nfl_audit_v1", "nfl-audit-v1.3-prior-feature-provenance-20260930")
        _challenge = _load_nfl_v19_exact("nfl_challenger_v1", "nfl-challenger-v1.4-season-forward-three-market-no-publish-20260930")
        _special = _load_nfl_v19_exact("nfl_specialized_v1", "nfl-specialized-v1.5-score-domain-h2h-stack-20260930")
        _score = _load_nfl_v19_exact("nfl_score_engine_v1", "nfl-score-engine-v1.6-team-offense-defense-20260930")
        _intel = _load_nfl_v19_exact("nfl_intelligence_v1", "nfl-intelligence-v1.8-residual-arbitration-reconciliation-20260930")
        _ledger = _load_nfl_v19_exact("nfl_prospective_ledger_v1", "nfl-prospective-ledger-v1.9.2-sharp-research-20261001")
        _stats = _load_nfl_v19_exact("nfl_stats_context_v1", "nfl-stats-context-v1.9.1-existing-dataset-only-20261001")
        _v19 = _load_nfl_v19_exact("nfl_frozen_confirmation_v1", "nfl-frozen-confirmation-v1.9.1-expanded-stats-2026-holdout-20261001")
        pw.emit("audit", f"[NFL-V1.9.1] Recheck audits before opening expanded frozen 2026 holdout run={run_id}", pct=0.05)
        try:
            _audit_report = _audit.run_nfl_audit_v1(storage_client=gcs, bucket_name=bucket, log_func=log_func)
            if _audit_report.get("status") != "READY_FOR_OFFLINE_CHALLENGER_SANDBOX":
                raise RuntimeError("[NFL-V1.9.1-HOLD] PRECEDING_AUDIT_NOT_GREEN "+str(_audit_report.get("status")))
            pw.emit("holdout", "[NFL-V1.9.1] Freeze expanded stats registry first; fit only through 2025; score 2026 once; initialize append-only research ledger", pct=0.37)
            _result = _v19.run_nfl_frozen_confirmation_v1(
                bq_client=bigquery.Client(project="sharplogger"),
                audit_report=_audit_report, log_func=log_func, ensure_ledger=True)
            pw.emit("done", "NFL V1.9.1 expanded stats frozen confirmation complete: "+_result["status"]+" (zero production authority)", pct=1.0)
        except Exception as exc:
            pw.emit("error", "NFL V1.9.1 expanded stats frozen confirmation failed: "+str(exc)+"\n"+traceback.format_exc(), pct=1.0)
            raise
        return

    # NFL V1.9.2 — protected research architecture + isolated prospective ledger.
    # Infrastructure/contract validation only: no production publication and no
    # retuning of the already-observed 2026 V1.9.1 holdout.
    if str(sport).upper().strip() == "NFL" and str(market).lower().strip() == "nfl_research_foundation":
        import json
        import importlib.util
        from pathlib import Path
        from google.cloud import bigquery
        _dir = Path(__file__).resolve().parent

        def _load_nfl_v192_exact(_name, _tag):
            _path = _dir / (_name + ".py")
            if not _path.is_file():
                raise RuntimeError(f"[NFL-V1.9.2-DEPLOY-PREFLIGHT] MISSING {_path}")
            _spec = importlib.util.spec_from_file_location(_name, _path)
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules[_name] = _mod
            _spec.loader.exec_module(_mod)
            if getattr(_mod, "SOURCE_TAG", "") != _tag:
                raise RuntimeError("[NFL-V1.9.2-DEPLOY-PREFLIGHT] STALE_OR_MIXED_"+_name)
            return _mod

        _contract = _load_nfl_v192_exact(
            "nfl_research_contract_v1",
            "nfl-research-contract-v1.9.2-protected-architecture-20261001",
        )
        _attribution = _load_nfl_v192_exact(
            "nfl_edge_attribution_v1",
            "nfl-edge-attribution-v1.9.2-independent-margin-20261001",
        )
        _ledger = _load_nfl_v192_exact(
            "nfl_prospective_ledger_v1",
            "nfl-prospective-ledger-v1.9.2-sharp-research-20261001",
        )
        _settlement = _load_nfl_v192_exact(
            "nfl_prospective_settlement_v1",
            "nfl-prospective-settlement-v1.9.2-append-only-20261001",
        )
        pw.emit("audit", f"[NFL-V1.9.2] Validate protected architecture and sharp_research ledger run={run_id}", pct=0.10)
        try:
            _contract_report = _contract.assert_contract()
            pw.emit("contract", "[NFL-V1.9.2] Protected CORE + MARKET + STAT + systems/miner contract PASS", pct=0.35)
            _health = _ledger.ledger_health_check(
                bigquery.Client(project="sharplogger"),
                service_identity_hint="sharp-train-sa@sharplogger.iam.gserviceaccount.com",
            )
            _result = {
                "status": "NFL_V1_9_2_RESEARCH_FOUNDATION_READY",
                "contract": _contract_report,
                "ledger_health": _health,
                "edge_attribution_source_tag": _attribution.SOURCE_TAG,
                "settlement_source_tag": _settlement.SOURCE_TAG,
                "production_authority": 0,
                "ncaaf": "UNCHANGED",
                "legacy_nfl": "UNCHANGED",
            }
            log_func("[NFL-V1.9.2-CONTRACT] "+json.dumps(_result, sort_keys=True, default=str))
            pw.emit("done", "NFL V1.9.2 research foundation ready (zero production authority)", pct=1.0)
        except Exception as exc:
            pw.emit("error", "NFL V1.9.2 research foundation failed: "+str(exc)+"\n"+traceback.format_exc(), pct=1.0)
            raise
        return

    # NFL V1.9.3 — protected challenger research engine.
    # Uses history only through 2025. 2026 is not queried by this route; the
    # resulting registry starts a new prospective shadow clock after deployment.
    if str(sport).upper().strip() == "NFL" and str(market).lower().strip() == "nfl_research_engine":
        import json
        import importlib.util
        from pathlib import Path
        from google.cloud import bigquery
        _dir = Path(__file__).resolve().parent

        def _load_nfl_v193_exact(_name, _tag):
            _path = _dir / (_name + ".py")
            if not _path.is_file():
                raise RuntimeError(f"[NFL-V1.9.3-DEPLOY-PREFLIGHT] MISSING {_path}")
            _spec = importlib.util.spec_from_file_location(_name, _path)
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules[_name] = _mod
            _spec.loader.exec_module(_mod)
            if getattr(_mod, "SOURCE_TAG", "") != _tag:
                raise RuntimeError("[NFL-V1.9.3-DEPLOY-PREFLIGHT] STALE_OR_MIXED_"+_name)
            return _mod

        _audit = _load_nfl_v193_exact(
            "nfl_audit_v1",
            "nfl-audit-v1.3-prior-feature-provenance-20260930",
        )
        _contract = _load_nfl_v193_exact(
            "nfl_research_contract_v1",
            "nfl-research-contract-v1.9.2-protected-architecture-20261001",
        )
        _structured = _load_nfl_v193_exact(
            "nfl_structured_research_v1",
            "nfl-structured-research-v1.9.3-nested-season-forward-20261001",
        )
        _miner = _load_nfl_v193_exact(
            "nfl_residual_miner_v2",
            "nfl-residual-miner-v2.0-market-error-fdr-20261001",
        )
        _ledger2 = _load_nfl_v193_exact(
            "nfl_prospective_ledger_v2",
            "nfl-prospective-ledger-v2-v1.9.3-enhanced-shadow-20261001",
        )
        _engine = _load_nfl_v193_exact(
            "nfl_research_engine_v1",
            "nfl-research-engine-v1.9.3-protected-challengers-20261001",
        )
        pw.emit("audit", f"[NFL-V1.9.3] Protected challenger research start run={run_id}; history through 2025 only", pct=0.05)
        try:
            _contract.assert_contract()
            _audit_report = _audit.run_nfl_audit_v1(storage_client=gcs, bucket_name=bucket, log_func=log_func)
            if _audit_report.get("status") != "READY_FOR_OFFLINE_CHALLENGER_SANDBOX":
                raise RuntimeError("[NFL-V1.9.3-HOLD] PRECEDING_AUDIT_NOT_GREEN "+str(_audit_report.get("status")))
            _health = _ledger2.ledger_health_check(
                bigquery.Client(project="sharplogger"),
                service_identity_hint="sharp-train-sa@sharplogger.iam.gserviceaccount.com",
            )
            pw.emit("research", "[NFL-V1.9.3] Run structured residual + independent CORE challengers + residual Miner V2", pct=0.30)
            _result = _engine.run_nfl_research_engine_v1(
                bq_client=bigquery.Client(project="sharplogger"),
                storage_client=gcs,
                bucket_name=bucket,
                audit_report=_audit_report,
                ledger_health=_health,
                log_func=log_func,
            )
            pw.emit("done", "NFL V1.9.3 research engine complete and frozen for prospective shadow: "+_result["status"], pct=1.0)
        except Exception as exc:
            pw.emit("error", "NFL V1.9.3 research engine failed: "+str(exc)+"\n"+traceback.format_exc(), pct=1.0)
            raise
        return

    # NFL V1.9.4 — dual-scorecard edge-gate research manager.
    # Re-runs the protected through-2025 research stack, then trains season-forward
    # edge-probability gates. 2026 is never queried and production authority stays zero.
    if str(sport).upper().strip() == "NFL" and str(market).lower().strip() == "nfl_edge_gate":
        import json
        import importlib.util
        from pathlib import Path
        from google.cloud import bigquery
        _dir = Path(__file__).resolve().parent

        def _load_nfl_v194_exact(_name, _tag):
            _path = _dir / (_name + ".py")
            if not _path.is_file():
                raise RuntimeError(f"[NFL-V1.9.4-DEPLOY-PREFLIGHT] MISSING {_path}")
            _spec = importlib.util.spec_from_file_location(_name, _path)
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules[_name] = _mod
            _spec.loader.exec_module(_mod)
            if getattr(_mod, "SOURCE_TAG", "") != _tag:
                raise RuntimeError("[NFL-V1.9.4-DEPLOY-PREFLIGHT] STALE_OR_MIXED_"+_name)
            return _mod

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
        pw.emit("audit", f"[NFL-V1.9.4] Edge-gate research start run={run_id}; protected history through 2025 only", pct=0.05)
        try:
            _contract.assert_contract()
            _audit_report = _audit.run_nfl_audit_v1(storage_client=gcs, bucket_name=bucket, log_func=log_func)
            if _audit_report.get("status") != "READY_FOR_OFFLINE_CHALLENGER_SANDBOX":
                raise RuntimeError("[NFL-V1.9.4-HOLD] PRECEDING_AUDIT_NOT_GREEN "+str(_audit_report.get("status")))
            _health = _ledger3.ledger_health_check(
                bigquery.Client(project="sharplogger"),
                service_identity_hint="sharp-train-sa@sharplogger.iam.gserviceaccount.com",
            )
            pw.emit("research", "[NFL-V1.9.4] Run null-safe structured research + Miner V2 + fair-line/edge dual scorecards + season-forward edge gates", pct=0.25)
            _result = _engine2.run_nfl_research_engine_v2(
                bq_client=bigquery.Client(project="sharplogger"),
                storage_client=gcs,
                bucket_name=bucket,
                audit_report=_audit_report,
                ledger_health=_health,
                log_func=log_func,
            )
            pw.emit("done", "NFL V1.9.4 edge-gate research complete and frozen for prospective shadow: "+_result["status"], pct=1.0)
        except Exception as exc:
            pw.emit("error", "NFL V1.9.4 edge-gate research failed: "+str(exc)+"\n"+traceback.format_exc(), pct=1.0)
            raise
        return

    # NFL V1.9.5 — prospective new-information shadow collector.
    # This route does NOT retrain historical models.  It establishes/continues an
    # append-only post-deployment market clock, captures timestamped book quotes,
    # builds fixed T-120/T-60/T-30/current market states, and settles prior states.
    # Pre-clock quotes are excluded from V1.9.5 evidence by contract.
    if str(sport).upper().strip() == "NFL" and str(market).lower().strip() == "nfl_prospective_shadow":
        import json
        import importlib.util
        from pathlib import Path
        from google.cloud import bigquery
        _dir = Path(__file__).resolve().parent

        def _load_nfl_v195_exact(_name, _tag):
            _path = _dir / (_name + ".py")
            if not _path.is_file():
                raise RuntimeError(f"[NFL-V1.9.5-DEPLOY-PREFLIGHT] MISSING {_path}")
            _spec = importlib.util.spec_from_file_location(_name, _path)
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules[_name] = _mod
            _spec.loader.exec_module(_mod)
            if getattr(_mod, "SOURCE_TAG", "") != _tag:
                raise RuntimeError("[NFL-V1.9.5-DEPLOY-PREFLIGHT] STALE_OR_MIXED_"+_name)
            return _mod

        _ledger4 = _load_nfl_v195_exact(
            "nfl_prospective_ledger_v4",
            "nfl-prospective-ledger-v4-v1.9.5-market-microstructure-20261001",
        )
        _market_shadow = _load_nfl_v195_exact(
            "nfl_market_shadow_v1",
            "nfl-market-shadow-v1.9.5-prospective-microstructure-20261001",
        )
        _system_shadow = _load_nfl_v195_exact(
            "nfl_system_shadow_v1",
            "nfl-system-shadow-v1.9.5.1-role-flip-family-20261001",
        )
        _shadow = _load_nfl_v195_exact(
            "nfl_prospective_shadow_v1",
            "nfl-prospective-shadow-v1.9.5-new-information-clock-20261001",
        )
        pw.emit("research", f"[NFL-V1.9.5] Prospective new-information shadow start run={run_id}; pre-clock quotes forbidden", pct=0.10)
        try:
            _result = _shadow.run_nfl_prospective_shadow_v1(
                bq_client=bigquery.Client(project="sharplogger"),
                storage_client=gcs,
                bucket_name=bucket,
                log_func=log_func,
            )
            pw.emit("done", "NFL V1.9.5 prospective market shadow active: "+_result["status"], pct=1.0)
        except Exception as exc:
            pw.emit("error", "NFL V1.9.5 prospective market shadow failed: "+str(exc)+"\n"+traceback.format_exc(), pct=1.0)
            raise
        return

    # NFL Research V2.0 — genuinely new market-blind play-by-play information.
    # This route keeps incumbent CORE protected and tests one fixed Ridge
    # challenger built only from prior-game PBP football efficiency.  2026 is
    # sealed and PBP final scores are never authoritative labels.
    if str(sport).upper().strip() == "NFL" and str(market).lower().strip() == "nfl_pbp_foundation":
        import importlib.util
        from pathlib import Path
        from google.cloud import bigquery
        _dir = Path(__file__).resolve().parent

        def _load_nfl_v2_exact(_name, _tag):
            _path = _dir / (_name + ".py")
            if not _path.is_file():
                raise RuntimeError(f"[NFL-RESEARCH-V2-DEPLOY-PREFLIGHT] MISSING {_path}")
            _spec = importlib.util.spec_from_file_location(_name, _path)
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules[_name] = _mod
            _spec.loader.exec_module(_mod)
            if getattr(_mod, "SOURCE_TAG", "") != _tag:
                raise RuntimeError("[NFL-RESEARCH-V2-DEPLOY-PREFLIGHT] STALE_OR_MIXED_"+_name)
            return _mod

        _contract_v2 = _load_nfl_v2_exact(
            "nfl_research_v2_contract",
            "nfl-research-v2.0-foundation-expansion-20261001",
        )
        _pbp = _load_nfl_v2_exact(
            "nfl_pbp_foundation_v1",
            "nfl-pbp-foundation-v1-research-v2.0-20261001",
        )
        _audit = _load_nfl_v2_exact(
            "nfl_audit_v1",
            "nfl-audit-v1.3-prior-feature-provenance-20260930",
        )
        pw.emit("research", f"[NFL-RESEARCH-V2-PBP] Start run={run_id}; download/aggregate nflverse 2017-2025 only; 2026 sealed", pct=0.05)
        try:
            _contract_v2.assert_contract()
            _audit_report = _audit.run_nfl_audit_v1(storage_client=gcs, bucket_name=bucket, log_func=log_func)
            if _audit_report.get("status") != "READY_FOR_OFFLINE_CHALLENGER_SANDBOX":
                raise RuntimeError("[NFL-RESEARCH-V2-PBP-HOLD] PRECEDING_AUDIT_NOT_GREEN "+str(_audit_report.get("status")))
            pw.emit("research", "[NFL-RESEARCH-V2-PBP] Build prior-only PBP efficiency/QB-history context and season-forward market-blind CORE2 challenger", pct=0.20)
            _result = _pbp.run_nfl_pbp_foundation_v1(
                bq_client=bigquery.Client(project="sharplogger"),
                storage_client=gcs,
                bucket_name=bucket,
                audit_report=_audit_report,
                log_func=log_func,
            )
            pw.emit("done", "NFL Research V2 PBP foundation complete: "+_result["status"], pct=1.0)
        except Exception as exc:
            pw.emit("error", "NFL Research V2 PBP foundation failed: "+str(exc)+"\n"+traceback.format_exc(), pct=1.0)
            raise
        return

    # NFL Research V2.0.2 — attribution diagnostic for the exact frozen PBP CORE2.
    # This route does not retrain/tune CORE2. It reuses the frozen OOF/context/model
    # artifacts and tests disagreement, residual, confirmation, regime and system
    # interaction value against the protected incumbent CORE. 2026 remains sealed.
    if str(sport).upper().strip() == "NFL" and str(market).lower().strip() == "nfl_pbp_diagnostic":
        import importlib.util
        from pathlib import Path
        from google.cloud import bigquery
        _dir = Path(__file__).resolve().parent

        def _load_nfl_v202_diag_exact(_name, _tag):
            _path = _dir / (_name + ".py")
            if not _path.is_file():
                raise RuntimeError(f"[NFL-RESEARCH-V2-PBP-DIAG-PREFLIGHT] MISSING {_path}")
            _spec = importlib.util.spec_from_file_location(_name, _path)
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules[_name] = _mod
            _spec.loader.exec_module(_mod)
            if getattr(_mod, "SOURCE_TAG", "") != _tag:
                raise RuntimeError("[NFL-RESEARCH-V2-PBP-DIAG-PREFLIGHT] STALE_OR_MIXED_"+_name)
            return _mod

        _contract_v2 = _load_nfl_v202_diag_exact(
            "nfl_research_v2_contract",
            "nfl-research-v2.0-foundation-expansion-20261001",
        )
        _diag = _load_nfl_v202_diag_exact(
            "nfl_pbp_attribution_v1",
            "nfl-pbp-attribution-v1-research-v2.0.2-frozen-core2-20261001",
        )
        _audit = _load_nfl_v202_diag_exact(
            "nfl_audit_v1",
            "nfl-audit-v1.3-prior-feature-provenance-20260930",
        )
        pw.emit("research", f"[NFL-RESEARCH-V2-PBP-DIAG] Start run={run_id}; frozen CORE2 attribution only; no PBP refit; 2026 sealed", pct=0.05)
        try:
            _contract_v2.assert_contract()
            _audit_report = _audit.run_nfl_audit_v1(storage_client=gcs, bucket_name=bucket, log_func=log_func)
            if _audit_report.get("status") != "READY_FOR_OFFLINE_CHALLENGER_SANDBOX":
                raise RuntimeError("[NFL-RESEARCH-V2-PBP-DIAG-HOLD] PRECEDING_AUDIT_NOT_GREEN "+str(_audit_report.get("status")))
            pw.emit("research", "[NFL-RESEARCH-V2-PBP-DIAG] Verify SHA 741474dc..., reuse frozen OOF/context/bundle, test independent Spread/Total/H2H/system attribution", pct=0.20)
            _result = _diag.run_nfl_pbp_attribution_v1(
                bq_client=bigquery.Client(project="sharplogger"),
                storage_client=gcs,
                bucket_name=bucket,
                audit_report=_audit_report,
                log_func=log_func,
            )
            pw.emit("done", "NFL Research V2 PBP attribution complete: "+_result["status"], pct=1.0)
        except Exception as exc:
            pw.emit("error", "NFL Research V2 PBP attribution failed: "+str(exc)+"\n"+traceback.format_exc(), pct=1.0)
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
            _path = _dir / (_name + ".py")
            if not _path.is_file():
                raise RuntimeError(f"[NFL-RESEARCH-V2-DEPLOY-PREFLIGHT] MISSING {_path}")
            _spec = importlib.util.spec_from_file_location(_name, _path)
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules[_name] = _mod
            _spec.loader.exec_module(_mod)
            if getattr(_mod, "SOURCE_TAG", "") != _tag:
                raise RuntimeError("[NFL-RESEARCH-V2-DEPLOY-PREFLIGHT] STALE_OR_MIXED_"+_name)
            return _mod

        _contract_v2 = _load_nfl_v2_system_exact(
            "nfl_research_v2_contract",
            "nfl-research-v2.0-foundation-expansion-20261001",
        )
        _systems = _load_nfl_v2_system_exact(
            "nfl_system_lab_v1",
            "nfl-system-lab-v1-research-v2.0-ncaaf-miner-v3-methodology-20261001",
        )
        _audit = _load_nfl_v2_system_exact(
            "nfl_audit_v1",
            "nfl-audit-v1.3-prior-feature-provenance-20260930",
        )
        pw.emit("research", f"[NFL-RESEARCH-V2-SYSTEM] Start run={run_id}; discovery=2017-2022 shadow=2023 confirm=2024 final=2025; 2026 sealed", pct=0.05)
        try:
            _contract_v2.assert_contract()
            _audit_report = _audit.run_nfl_audit_v1(storage_client=gcs, bucket_name=bucket, log_func=log_func)
            if _audit_report.get("status") != "READY_FOR_OFFLINE_CHALLENGER_SANDBOX":
                raise RuntimeError("[NFL-RESEARCH-V2-SYSTEM-HOLD] PRECEDING_AUDIT_NOT_GREEN "+str(_audit_report.get("status")))
            pw.emit("research", "[NFL-RESEARCH-V2-SYSTEM] Replicate documented systems + academic hypotheses; run independent domain Miner with V3-style robustness controls", pct=0.20)
            _result = _systems.run_nfl_system_lab_v1(
                bq_client=bigquery.Client(project="sharplogger"),
                storage_client=gcs,
                bucket_name=bucket,
                audit_report=_audit_report,
                log_func=log_func,
            )
            pw.emit("done", "NFL Research V2 System Lab complete: "+_result["status"], pct=1.0)
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

    train_sharp_model_for_market = _wrapper.train_sharp_model_for_market
    train_timing_model_for_market = _wrapper.train_timing_model_for_market

    # NCAAF edge fast path. Research mode remains zero-authority.  The explicit
    # production-promotion mode runs the same leakage-safe evidence stack once,
    # then publishes only the frozen NCAAF Production V1 edge contract.  It does
    # NOT revive timing, generic AutoFS, multi-head training, or legacy artifact
    # publication.
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
