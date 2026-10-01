"""
NFL Research Execution V1.9.3
=============================

Purpose
-------
Add the execution/prospective layer on top of the validated V1.9.2 protected
research contract without changing production NFL, NCAAF, or any protected
research family.

This module intentionally:
  * keeps production_authority = 0
  * does not overwrite legacy NFL model artifacts
  * does not alter NCAAF
  * runs existing NFL research modules rather than reimplementing them
  * keeps CORE / MARKET / STAT / BIG_AL / PATHI / MINER independent
  * forces PATHI to SHADOW_ONLY
  * forbids retroactive 2026 prediction backfill
  * writes append-only rows only after filtering to the already-created
    BigQuery table schema
  * does not infer CLV where timestamped executable historical quotes are absent

Expected existing V1.9.2 tables
-------------------------------
sharplogger.sharp_research.nfl_research_v1_predictions
sharplogger.sharp_research.nfl_research_v1_results
sharplogger.sharp_research.nfl_research_v1_health

Expected existing research modules from the September 30 stack
---------------------------------------------------------------
nfl_audit_v1.py
nfl_score_engine_v1.py
nfl_intelligence_v1.py

The executor is deliberately tolerant of function naming differences but NOT
of missing protected research modules. Missing required modules/functions fail
closed.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from google.cloud import bigquery


V193_SOURCE_TAG = "nfl-research-execution-v1.9.3-prospective-plumbing-20261001"
V192_CONTRACT_VERSION = "nfl-research-v1.9.2-protected-architecture-20261001"
V192_LEDGER_VERSION = "nfl-research-v1.9.2-prospective-20261001"

DEFAULT_PROJECT = os.getenv("BQ_PROJECT", "sharplogger")
DEFAULT_RESEARCH_DATASET = os.getenv("NFL_RESEARCH_DATASET", "sharp_research")

PREDICTION_TABLE = "nfl_research_v1_predictions"
RESULT_TABLE = "nfl_research_v1_results"
HEALTH_TABLE = "nfl_research_v1_health"

PRODUCTION_AUTHORITY = 0
PUBLICATION = False

PROTECTED_VOTE_FAMILIES = (
    "BIG_AL",
    "CORE",
    "MARKET",
    "MINER",
    "PATHI",
    "STAT",
)

PROTECTED_COMPONENTS = (
    "BIG_AL_DOCUMENTED_SYSTEMS",
    "CONTEXT_FAMILIES",
    "CORE_DIRECT_MARGIN",
    "CORE_DIRECT_TOTAL",
    "CORE_H2H_PROBABILITY",
    "CORE_SCORE_DERIVED_MARGIN",
    "CORE_SCORE_DERIVED_TOTAL",
    "MARKET_BASELINE",
    "PROSPECTIVE_LEDGER",
    "SOURCE_RECONCILIATION",
    "STATS_MARKET_RESIDUAL",
    "UNCERTAINTY",
)

REQUIRED_EXECUTION_MODULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("nfl_audit_v1", ("run_nfl_audit_v1", "run_nfl_audit", "run")),
    ("nfl_score_engine_v1", ("run_nfl_score_engine_v1", "run_nfl_score_engine", "run")),
    ("nfl_intelligence_v1", ("run_nfl_intelligence_v1", "run_nfl_intelligence", "run")),
)


class NFLResearchExecutionError(RuntimeError):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: Optional[datetime] = None) -> str:
    dt = dt or _utcnow()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _emit(log_func, marker: str, payload: Mapping[str, Any]) -> None:
    msg = f"{marker} {json.dumps(dict(payload), sort_keys=True, default=str)}"
    if log_func is not None:
        log_func(msg)
    else:
        print(msg, flush=True)


def _table_id(project: str, dataset: str, table: str) -> str:
    return f"{project}.{dataset}.{table}"


def _schema_names(client: bigquery.Client, table_id: str) -> List[str]:
    table = client.get_table(table_id)
    return [f.name for f in table.schema]


def _coerce_scalar(v: Any) -> Any:
    # Keep values JSON/BigQuery safe.
    if v is None:
        return None
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, datetime):
        return _iso_utc(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        return float(v)
    if isinstance(v, (str, int, bool)):
        return v
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, sort_keys=True, default=str)
    return str(v)


def _filter_row_to_schema(row: Mapping[str, Any], schema_names: Sequence[str]) -> Dict[str, Any]:
    allowed = set(schema_names)
    return {k: _coerce_scalar(v) for k, v in row.items() if k in allowed}


def _insert_append_only(
    client: bigquery.Client,
    table_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    log_func=None,
    marker="[NFL-V1.9.3-LEDGER-WRITE]",
) -> Dict[str, Any]:
    if not rows:
        return {"status": "NO_ROWS", "table": table_id, "rows": 0}

    schema = _schema_names(client, table_id)
    payload = [_filter_row_to_schema(r, schema) for r in rows]
    payload = [r for r in payload if r]

    if not payload:
        raise NFLResearchExecutionError(
            f"No supplied fields match the existing schema for {table_id}."
        )

    errors = client.insert_rows_json(table_id, payload)
    if errors:
        raise NFLResearchExecutionError(
            f"Append-only insert failed for {table_id}: {errors[:5]}"
        )

    out = {"status": "APPENDED", "table": table_id, "rows": len(payload)}
    _emit(log_func, marker, out)
    return out


def _latest_v192_health(
    client: bigquery.Client,
    project: str,
    dataset: str,
) -> Dict[str, Any]:
    table_id = _table_id(project, dataset, HEALTH_TABLE)
    fields = set(_schema_names(client, table_id))

    contract_col = next(
        (
            c
            for c in (
                "research_contract_version",
                "contract_version",
                "research_version",
            )
            if c in fields
        ),
        None,
    )
    ledger_col = next(
        (c for c in ("ledger_version", "prospective_ledger_version") if c in fields),
        None,
    )
    ts_col = next(
        (
            c
            for c in (
                "created_at",
                "event_ts",
                "event_timestamp",
                "timestamp",
                "created_utc",
            )
            if c in fields
        ),
        None,
    )

    select_cols = [c for c in (contract_col, ledger_col, ts_col, "status", "production_authority") if c and c in fields]
    if not select_cols:
        raise NFLResearchExecutionError(
            f"V1.9.2 health table {table_id} has no recognizable contract columns."
        )

    where = ""
    params: List[bigquery.ScalarQueryParameter] = []
    if contract_col:
        where = f"WHERE `{contract_col}` = @contract_version"
        params.append(bigquery.ScalarQueryParameter("contract_version", "STRING", V192_CONTRACT_VERSION))

    order = f"ORDER BY `{ts_col}` DESC" if ts_col else ""
    sql = f"""
        SELECT {", ".join(f"`{c}`" for c in select_cols)}
        FROM `{table_id}`
        {where}
        {order}
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    rows = list(client.query(sql, job_config=job_config).result())
    if not rows:
        raise NFLResearchExecutionError(
            "V1.9.2 protected contract/ledger health row was not found. "
            "Run the V1.9.2 bootstrap first."
        )

    d = dict(rows[0].items())
    if contract_col and d.get(contract_col) != V192_CONTRACT_VERSION:
        raise NFLResearchExecutionError(
            f"Unexpected research contract: {d.get(contract_col)!r}"
        )
    if ledger_col and d.get(ledger_col) not in (None, "", V192_LEDGER_VERSION):
        raise NFLResearchExecutionError(
            f"Unexpected prospective ledger version: {d.get(ledger_col)!r}"
        )
    if "production_authority" in d and int(d.get("production_authority") or 0) != 0:
        raise NFLResearchExecutionError(
            "V1.9.2 health row unexpectedly has production authority."
        )
    return d


def _resolve_callable(module_name: str, candidate_functions: Sequence[str]):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise NFLResearchExecutionError(
            f"Required protected NFL research module {module_name!r} could not be imported: {exc}"
        ) from exc

    for name in candidate_functions:
        fn = getattr(module, name, None)
        if callable(fn):
            return module, fn, name

    raise NFLResearchExecutionError(
        f"Module {module_name!r} loaded but no expected runner was found: "
        f"{list(candidate_functions)}"
    )


def _call_runner(fn, *, dashboard_module=None, log_func=None):
    """
    Call an existing runner without guessing positional arguments.
    Only known safe plumbing arguments are supplied when accepted.
    """
    sig = inspect.signature(fn)
    kwargs: Dict[str, Any] = {}
    if "dashboard_module" in sig.parameters:
        kwargs["dashboard_module"] = dashboard_module
    if "log_func" in sig.parameters:
        kwargs["log_func"] = log_func
    if "hard_fail" in sig.parameters:
        kwargs["hard_fail"] = True
    if "production_authority" in sig.parameters:
        kwargs["production_authority"] = 0
    if "publication" in sig.parameters:
        kwargs["publication"] = False
    return fn(**kwargs)


def _extract_market_summaries(obj: Any) -> List[Dict[str, Any]]:
    """
    Recursively capture research-state summaries without changing them.
    Useful for MINER H2H/TOTALS/SPREADS and other existing modules.
    """
    out: List[Dict[str, Any]] = []
    seen = set()

    def walk(x: Any, depth: int = 0):
        if depth > 8:
            return
        if isinstance(x, Mapping):
            keys = set(x.keys())
            if keys.intersection({"market", "status", "promising_count", "tested_hypotheses"}):
                row = {
                    k: _coerce_scalar(x.get(k))
                    for k in (
                        "market",
                        "status",
                        "promising_count",
                        "tested_hypotheses",
                        "production_authority",
                        "research_state",
                        "source_tag",
                    )
                    if k in x
                }
                sig = json.dumps(row, sort_keys=True, default=str)
                if row and sig not in seen:
                    seen.add(sig)
                    out.append(row)
            for v in x.values():
                walk(v, depth + 1)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v, depth + 1)

    walk(obj)
    return out


def _research_health_row(
    run_id: str,
    status: str,
    *,
    module_runs: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    now = _iso_utc()
    return {
        "health_event_id": hashlib.sha256(
            f"{V193_SOURCE_TAG}|{run_id}|{status}".encode("utf-8")
        ).hexdigest(),
        "created_at": now,
        "event_timestamp": now,
        "status": status,
        "source_tag": V193_SOURCE_TAG,
        "research_contract_version": V192_CONTRACT_VERSION,
        "ledger_version": V192_LEDGER_VERSION,
        "production_authority": 0,
        "publication": False,
        "legacy_nfl": "UNCHANGED",
        "ncaaf": "UNCHANGED",
        "protected_vote_families": list(PROTECTED_VOTE_FAMILIES),
        "protected_components": list(PROTECTED_COMPONENTS),
        "module_runs": list(module_runs),
        "research_summaries": list(summaries),
    }


def run_nfl_research_execution_v1_9_3(
    *,
    dashboard_module=None,
    log_func=None,
    bq_client: Optional[bigquery.Client] = None,
    project: str = DEFAULT_PROJECT,
    research_dataset: str = DEFAULT_RESEARCH_DATASET,
) -> Dict[str, Any]:
    """
    Execute the already-built NFL research stack after V1.9.2 bootstrap.

    Important:
      * This does NOT train/publish a replacement production model.
      * This does NOT change any authority.
      * This does NOT write retroactive predictions.
      * Existing research outputs remain independent.
    """
    client = bq_client or bigquery.Client(project=project)
    run_id = str(uuid.uuid4())

    _emit(
        log_func,
        "[NFL-V1.9.3-PREFLIGHT]",
        {
            "status": "START",
            "source_tag": V193_SOURCE_TAG,
            "run_id": run_id,
            "production_authority": 0,
            "publication": False,
            "legacy_nfl": "UNCHANGED",
            "ncaaf": "UNCHANGED",
        },
    )

    health = _latest_v192_health(client, project, research_dataset)
    _emit(
        log_func,
        "[NFL-V1.9.3-V1.9.2-CONTRACT]",
        {
            "status": "PASS",
            "research_contract_version": health.get("research_contract_version")
            or health.get("contract_version"),
            "ledger_version": health.get("ledger_version"),
            "production_authority": 0,
        },
    )

    module_runs: List[Dict[str, Any]] = []
    outputs: Dict[str, Any] = {}
    summaries: List[Dict[str, Any]] = []

    for module_name, candidates in REQUIRED_EXECUTION_MODULES:
        module, fn, fn_name = _resolve_callable(module_name, candidates)
        source_tag = next(
            (
                getattr(module, attr)
                for attr in (
                    "SOURCE_TAG",
                    "NFL_SOURCE_TAG",
                    "NFL_AUDIT_V1_SOURCE_TAG",
                    "NFL_SCORE_V1_SOURCE_TAG",
                    "NFL_INTEL_V1_SOURCE_TAG",
                )
                if hasattr(module, attr)
            ),
            None,
        )

        _emit(
            log_func,
            "[NFL-V1.9.3-MODULE]",
            {
                "module": module_name,
                "runner": fn_name,
                "source_tag": source_tag,
                "status": "START",
            },
        )

        result = _call_runner(
            fn,
            dashboard_module=dashboard_module,
            log_func=log_func,
        )
        outputs[module_name] = result

        ms = _extract_market_summaries(result)
        summaries.extend(ms)
        module_runs.append(
            {
                "module": module_name,
                "runner": fn_name,
                "source_tag": source_tag,
                "status": "PASS",
                "summary_count": len(ms),
            }
        )
        _emit(
            log_func,
            "[NFL-V1.9.3-MODULE]",
            {
                "module": module_name,
                "runner": fn_name,
                "status": "PASS",
                "summary_count": len(ms),
            },
        )

    # Do not manufacture prospective rows from historical research outputs.
    # Live/scanner code must explicitly call record_prospective_predictions(...)
    # with true pre-kickoff rows.
    health_row = _research_health_row(
        run_id,
        "RESEARCH_EXECUTION_COMPLETE",
        module_runs=module_runs,
        summaries=summaries,
    )
    health_write = _insert_append_only(
        client,
        _table_id(project, research_dataset, HEALTH_TABLE),
        [health_row],
        log_func=log_func,
        marker="[NFL-V1.9.3-HEALTH]",
    )

    final = {
        "status": "RESEARCH_EXECUTION_COMPLETE",
        "source_tag": V193_SOURCE_TAG,
        "run_id": run_id,
        "production_authority": 0,
        "publication": False,
        "legacy_nfl": "UNCHANGED",
        "ncaaf": "UNCHANGED",
        "module_runs": module_runs,
        "research_summaries": summaries,
        "prospective_rows_written": 0,
        "prospective_note": (
            "No historical or 2026 backfill. Live scanner must call "
            "record_prospective_predictions before kickoff."
        ),
        "health_write": health_write,
        "outputs": outputs,
    }
    _emit(log_func, "[NFL-V1.9.3-FINAL]", {k: v for k, v in final.items() if k != "outputs"})
    return final


def _parse_dt(v: Any) -> datetime:
    if isinstance(v, datetime):
        dt = v
    else:
        s = str(v).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _family(row: Mapping[str, Any]) -> str:
    return str(
        row.get("vote_family")
        or row.get("family")
        or row.get("component_family")
        or ""
    ).upper().strip()


def _game_key(row: Mapping[str, Any]) -> str:
    return str(
        row.get("game_key")
        or row.get("Game_Key")
        or row.get("event_id")
        or row.get("Source_Game_ID")
        or ""
    ).strip()


def _market(row: Mapping[str, Any]) -> str:
    return str(row.get("market") or row.get("Market") or "").upper().strip()


def _prediction_id(row: Mapping[str, Any]) -> str:
    key = "|".join(
        [
            V192_CONTRACT_VERSION,
            _game_key(row),
            _market(row),
            _family(row),
            str(row.get("asof_utc") or row.get("prediction_timestamp") or row.get("created_at") or ""),
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def record_prospective_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    log_func=None,
    bq_client: Optional[bigquery.Client] = None,
    project: str = DEFAULT_PROJECT,
    research_dataset: str = DEFAULT_RESEARCH_DATASET,
    now_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Append true prospective, pre-kickoff research predictions.

    Required semantic fields per row:
      game_key/event_id
      market
      family/vote_family
      kickoff_utc
      asof_utc or prediction_timestamp

    Other family-specific attribution fields are retained when they already
    exist in the V1.9.2 prediction-table schema.
    """
    client = bq_client or bigquery.Client(project=project)
    _latest_v192_health(client, project, research_dataset)

    now = now_utc or _utcnow()
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for src in rows:
        r = dict(src)
        fam = _family(r)
        game = _game_key(r)
        market = _market(r)

        if fam not in PROTECTED_VOTE_FAMILIES:
            rejected.append({"game_key": game, "market": market, "family": fam, "reason": "UNKNOWN_FAMILY"})
            continue
        if not game or not market:
            rejected.append({"game_key": game, "market": market, "family": fam, "reason": "MISSING_IDENTITY"})
            continue

        kickoff_raw = r.get("kickoff_utc") or r.get("game_start_utc") or r.get("Game_Start")
        asof_raw = r.get("asof_utc") or r.get("prediction_timestamp") or r.get("created_at")
        if not kickoff_raw or not asof_raw:
            rejected.append({"game_key": game, "market": market, "family": fam, "reason": "MISSING_TIME"})
            continue

        try:
            kickoff = _parse_dt(kickoff_raw)
            asof = _parse_dt(asof_raw)
        except Exception:
            rejected.append({"game_key": game, "market": market, "family": fam, "reason": "BAD_TIME"})
            continue

        if asof >= kickoff:
            rejected.append({"game_key": game, "market": market, "family": fam, "reason": "NOT_PREGAME"})
            continue

        # No retroactive backfill: prediction rows must be generated now, not
        # reconstructed after games have begun.
        if kickoff <= now:
            rejected.append({"game_key": game, "market": market, "family": fam, "reason": "RETROACTIVE_BACKFILL_BLOCKED"})
            continue

        r.update(
            {
                "prediction_id": _prediction_id(r),
                "source_tag": V193_SOURCE_TAG,
                "research_contract_version": V192_CONTRACT_VERSION,
                "ledger_version": V192_LEDGER_VERSION,
                "production_authority": 0,
                "publication": False,
                "vote_family": fam,
                "family": fam,
                "market": market,
                "game_key": game,
                "kickoff_utc": _iso_utc(kickoff),
                "asof_utc": _iso_utc(asof),
                "research_state": "SHADOW_ONLY" if fam == "PATHI" else str(r.get("research_state") or "PROSPECTIVE"),
                "clv": None,
                "clv_status": "NOT_VERIFIED",
            }
        )
        accepted.append(r)

    table_id = _table_id(project, research_dataset, PREDICTION_TABLE)

    # Deduplicate against existing prediction IDs when the table exposes that key.
    schema = set(_schema_names(client, table_id))
    if accepted and "prediction_id" in schema:
        ids = [r["prediction_id"] for r in accepted]
        q = f"SELECT prediction_id FROM `{table_id}` WHERE prediction_id IN UNNEST(@ids)"
        cfg = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("ids", "STRING", ids)]
        )
        existing = {str(x["prediction_id"]) for x in client.query(q, job_config=cfg).result()}
        accepted = [r for r in accepted if r["prediction_id"] not in existing]

    write = _insert_append_only(
        client,
        table_id,
        accepted,
        log_func=log_func,
        marker="[NFL-V1.9.3-PROSPECTIVE-WRITE]",
    )

    out = {
        "status": "PASS",
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejection_sample": rejected[:20],
        "production_authority": 0,
        "publication": False,
        "write": write,
    }
    _emit(log_func, "[NFL-V1.9.3-PROSPECTIVE]", out)
    return out


def record_settlements(
    rows: Sequence[Mapping[str, Any]],
    *,
    log_func=None,
    bq_client: Optional[bigquery.Client] = None,
    project: str = DEFAULT_PROJECT,
    research_dataset: str = DEFAULT_RESEARCH_DATASET,
) -> Dict[str, Any]:
    """
    Append settlement rows produced by the existing trusted result/reconciliation
    path. This function does not infer final scores or closing lines itself.

    This separation is intentional: it prevents the research executor from
    inventing a join between live scanner event IDs and historical source IDs.
    """
    client = bq_client or bigquery.Client(project=project)
    _latest_v192_health(client, project, research_dataset)

    payload: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for src in rows:
        r = dict(src)
        game = _game_key(r)
        market = _market(r)
        fam = _family(r)
        if not game or not market or fam not in PROTECTED_VOTE_FAMILIES:
            rejected.append(
                {"game_key": game, "market": market, "family": fam, "reason": "BAD_IDENTITY"}
            )
            continue

        # Settlement must point to a prospective prediction identifier.
        pred_id = str(r.get("prediction_id") or "").strip()
        if not pred_id:
            rejected.append(
                {"game_key": game, "market": market, "family": fam, "reason": "MISSING_PREDICTION_ID"}
            )
            continue

        r.update(
            {
                "result_id": hashlib.sha256(
                    f"{pred_id}|SETTLEMENT|{V193_SOURCE_TAG}".encode("utf-8")
                ).hexdigest(),
                "source_tag": V193_SOURCE_TAG,
                "research_contract_version": V192_CONTRACT_VERSION,
                "ledger_version": V192_LEDGER_VERSION,
                "production_authority": 0,
                "publication": False,
                "game_key": game,
                "market": market,
                "vote_family": fam,
                "family": fam,
                "settled_at": _iso_utc(),
                # CLV stays null until an audited timestamped executable quote
                # contract exists.
                "clv": None,
                "clv_status": "NOT_VERIFIED",
            }
        )
        payload.append(r)

    write = _insert_append_only(
        client,
        _table_id(project, research_dataset, RESULT_TABLE),
        payload,
        log_func=log_func,
        marker="[NFL-V1.9.3-SETTLEMENT-WRITE]",
    )
    out = {
        "status": "PASS",
        "accepted": len(payload),
        "rejected": len(rejected),
        "rejection_sample": rejected[:20],
        "production_authority": 0,
        "publication": False,
        "write": write,
    }
    _emit(log_func, "[NFL-V1.9.3-SETTLEMENT]", out)
    return out

def run_after_v192_foundation(*, dashboard_module=None, log_func=None):
    """Canonical train_job entrypoint for V1.9.3 after V1.9.2 bootstrap."""
    _emit(
        log_func,
        "[NFL-V1.9.3-ORCHESTRATION]",
        {
            "status": "ENTER",
            "source_tag": V193_SOURCE_TAG,
            "production_authority": 0,
            "legacy_nfl": "UNCHANGED",
            "ncaaf": "UNCHANGED",
        },
    )
    return run_nfl_research_execution_v1_9_3(
        dashboard_module=dashboard_module,
        log_func=log_func,
    )

