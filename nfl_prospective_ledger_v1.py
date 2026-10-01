"""Append-only NFL research prediction ledger for V1.9.2+.

Research-only.  Default destination is the isolated ``sharp_research`` dataset,
not production ``sharp_data``.  The ledger preserves independent CORE forecasts,
market-residual forecasts, systems/miner/context diagnostics and uncertainty in
one immutable pregame record.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
import time
from typing import Iterable

import pandas as pd

from nfl_research_contract_v1 import CONTRACT_VERSION, COMPONENTS, contract_hash

SOURCE_TAG = "nfl-prospective-ledger-v1.9.2-sharp-research-20261001"
PROJECT_ID = os.getenv("GCP_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT", "sharplogger"))
DATASET_ID = os.getenv("NFL_RESEARCH_DATASET", "sharp_research")
PRED_TABLE = f"{PROJECT_ID}.{DATASET_ID}.nfl_research_v1_predictions"
RESULT_TABLE = f"{PROJECT_ID}.{DATASET_ID}.nfl_research_v1_results"
HEALTH_TABLE = f"{PROJECT_ID}.{DATASET_ID}.nfl_research_v1_health"
LEDGER_VERSION = "nfl-research-v1.9.2-prospective-20261001"
PRODUCTION_AUTHORITY = 0


def _schema_predictions():
    from google.cloud import bigquery as b
    S = b.SchemaField
    return [
        S("prediction_event_id", "STRING", mode="REQUIRED"),
        S("canonical_snapshot_key", "STRING"),
        S("ledger_version", "STRING", mode="REQUIRED"),
        S("research_contract_version", "STRING"),
        S("research_contract_sha256", "STRING"),
        S("model_version", "STRING"),
        S("registry_sha256", "STRING"),
        S("physical_game_id", "STRING", mode="REQUIRED"),
        S("game_start", "TIMESTAMP"),
        S("home_team", "STRING"),
        S("away_team", "STRING"),
        S("market", "STRING"),
        S("side", "STRING"),
        S("line", "FLOAT64"),
        S("odds", "FLOAT64"),
        S("market_margin_reference", "FLOAT64"),
        S("market_total_reference", "FLOAT64"),
        S("market_h2h_probability", "FLOAT64"),
        S("quote_timestamp", "TIMESTAMP"),
        S("prediction_timestamp", "TIMESTAMP"),
        S("snapshot_type", "STRING"),
        S("as_of_bucket", "STRING"),
        S("canonical_evaluation", "BOOL"),
        # Independent CORE family.  Direct + score are preserved but remain one family.
        S("core_direct_margin_pred", "FLOAT64"),
        S("core_score_margin_pred", "FLOAT64"),
        S("core_margin_pred", "FLOAT64"),
        S("core_direct_total_pred", "FLOAT64"),
        S("core_score_total_pred", "FLOAT64"),
        S("core_total_pred", "FLOAT64"),
        S("h2h_probability", "FLOAT64"),
        S("core_margin_internal_gap", "FLOAT64"),
        S("core_total_internal_gap", "FLOAT64"),
        S("core_market_margin_gap", "FLOAT64"),
        S("core_market_total_gap", "FLOAT64"),
        # Market-residual/stat family remains separate from CORE.
        S("stats_spread_correction", "FLOAT64"),
        S("stats_corrected_margin_pred", "FLOAT64"),
        S("stats_total_correction", "FLOAT64"),
        S("stats_corrected_total_pred", "FLOAT64"),
        S("stats_market_margin_gap", "FLOAT64"),
        S("stats_market_total_gap", "FLOAT64"),
        S("spread_core_stat_same_direction", "BOOL"),
        S("total_core_stat_same_direction", "BOOL"),
        # Existing V1.8/V1.9 diagnostics remain available.
        S("spread_consensus_edge", "FLOAT64"),
        S("total_consensus_edge", "FLOAT64"),
        S("spread_model_gap", "FLOAT64"),
        S("total_model_gap", "FLOAT64"),
        S("uncertainty_q80", "FLOAT64"),
        S("uncertainty_q90", "FLOAT64"),
        S("bigal_active", "STRING"),
        S("pathi_family_opinion", "STRING"),
        S("miner_active", "STRING"),
        S("resolved_family_opinions", "STRING"),
        S("context_flags_json", "STRING"),
        S("edge_attribution_json", "STRING"),
        S("component_registry_json", "STRING"),
        S("snapshot_payload_json", "STRING"),
        S("is_prospective", "BOOL"),
        S("production_authority", "INT64"),
    ]


def _schema_results():
    from google.cloud import bigquery as b
    S = b.SchemaField
    return [
        S("result_event_id", "STRING", mode="REQUIRED"),
        S("prediction_event_id", "STRING", mode="REQUIRED"),
        S("physical_game_id", "STRING"),
        S("market", "STRING"),
        S("settled_at", "TIMESTAMP"),
        S("home_score", "FLOAT64"),
        S("away_score", "FLOAT64"),
        S("actual_margin", "FLOAT64"),
        S("actual_total", "FLOAT64"),
        S("result", "STRING"),
        S("market_error", "FLOAT64"),
        S("core_direct_error", "FLOAT64"),
        S("core_score_error", "FLOAT64"),
        S("core_family_error", "FLOAT64"),
        S("stats_corrected_error", "FLOAT64"),
        S("brier", "FLOAT64"),
        S("log_loss", "FLOAT64"),
        # CLV stays nullable until a canonical matched closing-quote contract is proven.
        S("clv_points", "FLOAT64"),
        S("closing_line", "FLOAT64"),
        S("closing_odds", "FLOAT64"),
        S("closing_quote_timestamp", "TIMESTAMP"),
        S("production_authority", "INT64"),
    ]


def _schema_health():
    from google.cloud import bigquery as b
    S = b.SchemaField
    return [
        S("health_event_id", "STRING", mode="REQUIRED"),
        S("checked_at", "TIMESTAMP"),
        S("ledger_version", "STRING"),
        S("service_identity_hint", "STRING"),
        S("status", "STRING"),
    ]


def _dataset_ref():
    return f"{PROJECT_ID}.{DATASET_ID}"


def _ensure_table(client, table_id: str, schema: list, *, partition_field: str | None = None, cluster_fields: list[str] | None = None):
    from google.cloud import bigquery as b
    from google.api_core.exceptions import NotFound

    created = False
    added = []
    try:
        table = client.get_table(table_id)
    except NotFound:
        table = b.Table(table_id, schema=schema)
        table.description = "Append-only NFL research ledger. Zero production betting authority."
        if partition_field:
            table.time_partitioning = b.TimePartitioning(type_=b.TimePartitioningType.DAY, field=partition_field)
        if cluster_fields:
            table.clustering_fields = list(cluster_fields)
        table = client.create_table(table)
        created = True
    else:
        existing = {f.name for f in table.schema}
        missing = [f for f in schema if f.name not in existing]
        if missing:
            # BigQuery allows appending NULLABLE fields to existing schemas.  All V1.9.2
            # additions are nullable; required identity columns predate this revision.
            bad = [f.name for f in missing if f.mode == "REQUIRED"]
            if bad:
                raise RuntimeError(f"NFL_LEDGER_REQUIRED_SCHEMA_MIGRATION_NEEDED table={table_id} fields={bad}")
            table.schema = list(table.schema) + missing
            table = client.update_table(table, ["schema"])
            added = [f.name for f in missing]
    return table, created, added


def ensure_tables(client=None):
    """Create/validate the isolated research tables; never create the dataset itself."""
    from google.cloud import bigquery as b
    from google.api_core.exceptions import Forbidden, NotFound

    c = client or b.Client(project=PROJECT_ID)
    try:
        c.get_dataset(_dataset_ref())
    except NotFound as exc:
        raise RuntimeError(
            f"NFL_RESEARCH_DATASET_MISSING {_dataset_ref()} -- create it explicitly before running V1.9.2"
        ) from exc
    except Forbidden as exc:
        raise PermissionError(f"NFL_RESEARCH_DATASET_FORBIDDEN {_dataset_ref()}: {exc}") from exc

    made, migrated = [], {}
    for table, schema, part, cluster in (
        (PRED_TABLE, _schema_predictions(), "prediction_timestamp", ["market", "physical_game_id", "snapshot_type"]),
        (RESULT_TABLE, _schema_results(), "settled_at", ["market", "physical_game_id", "prediction_event_id"]),
        (HEALTH_TABLE, _schema_health(), "checked_at", ["status"]),
    ):
        try:
            _, created, added = _ensure_table(c, table, schema, partition_field=part, cluster_fields=cluster)
        except Forbidden as exc:
            raise PermissionError(f"NFL_RESEARCH_TABLE_FORBIDDEN table={table}: {exc}") from exc
        if created:
            made.append(table)
        if added:
            migrated[table] = added

    return {
        "status": "READY",
        "dataset": _dataset_ref(),
        "prediction_table": PRED_TABLE,
        "result_table": RESULT_TABLE,
        "health_table": HEALTH_TABLE,
        "created": made,
        "schema_fields_added": migrated,
        "ledger_version": LEDGER_VERSION,
        "research_contract_version": CONTRACT_VERSION,
        "research_contract_sha256": contract_hash(),
        "production_authority": 0,
    }


def ledger_health_check(client=None, *, service_identity_hint: str | None = None) -> dict:
    """Prove table create/read/write/query permission without polluting prediction data."""
    from google.cloud import bigquery as b

    c = client or b.Client(project=PROJECT_ID)
    ready = ensure_tables(c)
    eid = hashlib.sha256(f"{LEDGER_VERSION}|{uuid.uuid4()}".encode()).hexdigest()
    now = pd.Timestamp.now(tz="UTC")
    row = {
        "health_event_id": eid,
        "checked_at": now.isoformat(),
        "ledger_version": LEDGER_VERSION,
        "service_identity_hint": service_identity_hint or "runtime-service-account",
        "status": "PASS",
    }
    errors = c.insert_rows_json(HEALTH_TABLE, [row])
    if errors:
        raise RuntimeError("NFL_LEDGER_HEALTH_WRITE_FAILED " + str(errors[:3]))
    cfg = b.QueryJobConfig(query_parameters=[b.ScalarQueryParameter("eid", "STRING", eid)])
    got = pd.DataFrame()
    for _ in range(4):
        got = c.query(
            f"SELECT health_event_id,status FROM `{HEALTH_TABLE}` WHERE health_event_id=@eid LIMIT 1",
            job_config=cfg,
        ).to_dataframe(create_bqstorage_client=False)
        if not got.empty and str(got.iloc[0].get("status")) == "PASS":
            break
        time.sleep(1.0)
    if got.empty or str(got.iloc[0].get("status")) != "PASS":
        raise RuntimeError("NFL_LEDGER_HEALTH_READBACK_FAILED")
    return {
        **ready,
        "TABLES_EXIST": True,
        "WRITE_PERMISSION_PASS": True,
        "READ_PERMISSION_PASS": True,
        "QUERY_PERMISSION_PASS": True,
        "APPEND_ONLY_CONTRACT_PASS": True,
        "health_event_id": eid,
        "status": "HEALTHY",
    }


def _flt(x):
    try:
        z = float(x)
        return z if math.isfinite(z) else None
    except Exception:
        return None


def _bool(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "yes", "y"}:
            return True
        if s in {"0", "false", "no", "n"}:
            return False
    return bool(x)


def _txt(row, *names, default=""):
    for name in names:
        if name in row and pd.notna(row.get(name)):
            return str(row.get(name))
    return default


def _json_payload(obj) -> str:
    return json.dumps(obj, default=str, sort_keys=True, separators=(",", ":"))


def _payload_value(v):
    if isinstance(v, (dict, list, tuple, set)):
        return v
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    try:
        missing = pd.isna(v)
        if isinstance(missing, bool) and missing:
            return None
    except Exception:
        pass
    return v


def _time_bucket(game_start: pd.Timestamp, quote_ts: pd.Timestamp) -> str:
    mins = (game_start - quote_ts).total_seconds() / 60.0
    if mins < 0:
        return "POST_START"
    if mins <= 30:
        return "T_MINUS_0_30M"
    if mins <= 90:
        return "T_MINUS_30_90M"
    if mins <= 240:
        return "T_MINUS_90M_4H"
    if mins <= 1440:
        return "T_MINUS_4H_24H"
    return "T_MINUS_GT_24H"


def prepare_prediction_rows(rows: pd.DataFrame, *, model_version: str, registry_sha256: str, now=None):
    """Prepare immutable pregame events.

    No completed game may be backfilled.  Snapshot type defaults to EVENT_STREAM;
    a caller may explicitly mark a fixed prospective snapshot as canonical_evaluation.
    """
    if rows is None or rows.empty:
        return pd.DataFrame(), {"status": "NO_ROWS"}
    n = pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now, utc=True)
    out, rejected = [], {}

    def rej(k):
        rejected[k] = rejected.get(k, 0) + 1

    registry_json = _json_payload(COMPONENTS)
    contract_sha = contract_hash()
    for _, r in rows.iterrows():
        gs = pd.to_datetime(r.get("game_start", r.get("Game_Start")), errors="coerce", utc=True)
        qt = pd.to_datetime(r.get("quote_timestamp", r.get("Snapshot_Timestamp")), errors="coerce", utc=True)
        if pd.isna(gs) or pd.isna(qt) or not (qt < n + pd.Timedelta(minutes=1) and n < gs and qt < gs):
            rej("NOT_PROSPECTIVE")
            continue
        gid = _txt(r, "physical_game_id", "Physical_Game_ID", "Merge_Key_Short").strip().lower()
        market = _txt(r, "market", "Market").strip().lower()
        side = _txt(r, "side", "Outcome").strip()
        if not gid or market not in {"spreads", "totals", "h2h"}:
            rej("IDENTITY_OR_MARKET")
            continue
        snapshot_type = _txt(r, "snapshot_type", default="EVENT_STREAM").strip().upper() or "EVENT_STREAM"
        canonical_eval = _bool(r.get("canonical_evaluation"))
        canonical_eval = bool(canonical_eval) if canonical_eval is not None else False
        as_of_bucket = _txt(r, "as_of_bucket", default="").strip() or _time_bucket(gs, qt)
        canonical_key = hashlib.sha256(
            f"{LEDGER_VERSION}|{gid}|{market}|{side}|{snapshot_type}".encode()
        ).hexdigest()
        key = f"{LEDGER_VERSION}|{registry_sha256}|{gid}|{market}|{side}|{snapshot_type}|{qt.isoformat()}"
        eid = hashlib.sha256(key.encode()).hexdigest()
        payload = {k: _payload_value(v) for k, v in r.to_dict().items()}

        row = {
            "prediction_event_id": eid,
            "canonical_snapshot_key": canonical_key,
            "ledger_version": LEDGER_VERSION,
            "research_contract_version": CONTRACT_VERSION,
            "research_contract_sha256": contract_sha,
            "model_version": model_version,
            "registry_sha256": registry_sha256,
            "physical_game_id": gid,
            "game_start": gs,
            "home_team": _txt(r, "home_team", "Home_Team"),
            "away_team": _txt(r, "away_team", "Away_Team"),
            "market": market,
            "side": side,
            "line": _flt(r.get("line", r.get("Value"))),
            "odds": _flt(r.get("odds", r.get("Odds_Price"))),
            "market_margin_reference": _flt(r.get("market_margin_reference", r.get("market_fair_margin"))),
            "market_total_reference": _flt(r.get("market_total_reference", r.get("market_fair_total"))),
            "market_h2h_probability": _flt(r.get("market_h2h_probability")),
            "quote_timestamp": qt,
            "prediction_timestamp": n,
            "snapshot_type": snapshot_type,
            "as_of_bucket": as_of_bucket,
            "canonical_evaluation": canonical_eval,
            "core_direct_margin_pred": _flt(r.get("core_direct_margin_pred", r.get("direct_margin_pred"))),
            "core_score_margin_pred": _flt(r.get("core_score_margin_pred", r.get("score_margin_pred"))),
            "core_margin_pred": _flt(r.get("core_margin_pred", r.get("core_margin_family_pred"))),
            "core_direct_total_pred": _flt(r.get("core_direct_total_pred", r.get("direct_total_pred"))),
            "core_score_total_pred": _flt(r.get("core_score_total_pred", r.get("score_total_pred"))),
            "core_total_pred": _flt(r.get("core_total_pred", r.get("core_total_family_pred"))),
            "h2h_probability": _flt(r.get("h2h_probability", r.get("h2h_prob"))),
            "core_margin_internal_gap": _flt(r.get("core_margin_internal_gap")),
            "core_total_internal_gap": _flt(r.get("core_total_internal_gap")),
            "core_market_margin_gap": _flt(r.get("core_market_margin_gap")),
            "core_market_total_gap": _flt(r.get("core_market_total_gap")),
            "stats_spread_correction": _flt(r.get("stats_spread_correction", r.get("spread_market_correction"))),
            "stats_corrected_margin_pred": _flt(r.get("stats_corrected_margin_pred")),
            "stats_total_correction": _flt(r.get("stats_total_correction", r.get("total_market_correction"))),
            "stats_corrected_total_pred": _flt(r.get("stats_corrected_total_pred")),
            "stats_market_margin_gap": _flt(r.get("stats_market_margin_gap")),
            "stats_market_total_gap": _flt(r.get("stats_market_total_gap")),
            "spread_core_stat_same_direction": _bool(r.get("spread_core_stat_same_direction")),
            "total_core_stat_same_direction": _bool(r.get("total_core_stat_same_direction")),
            "spread_consensus_edge": _flt(r.get("spread_consensus_edge")),
            "total_consensus_edge": _flt(r.get("total_consensus_edge")),
            "spread_model_gap": _flt(r.get("spread_model_gap")),
            "total_model_gap": _flt(r.get("total_model_gap")),
            "uncertainty_q80": _flt(r.get("uncertainty_q80")),
            "uncertainty_q90": _flt(r.get("uncertainty_q90")),
            "bigal_active": _txt(r, "bigal_active"),
            "pathi_family_opinion": _txt(r, "pathi_family_opinion"),
            "miner_active": _txt(r, "miner_active"),
            "resolved_family_opinions": _txt(r, "resolved_family_opinions"),
            "context_flags_json": _txt(r, "context_flags_json", default="{}"),
            "edge_attribution_json": _txt(r, "edge_attribution_json", default="{}"),
            "component_registry_json": registry_json,
            "snapshot_payload_json": _json_payload(payload),
            "is_prospective": True,
            "production_authority": 0,
        }
        out.append(row)
    return pd.DataFrame(out), {
        "status": "PREPARED" if out else "NO_ELIGIBLE_ROWS",
        "prepared": len(out),
        "rejected": rejected,
        "ledger_version": LEDGER_VERSION,
        "production_authority": 0,
    }


def append_prediction_rows(rows: pd.DataFrame, *, client=None):
    if rows is None or rows.empty:
        return {"status": "NO_ROWS", "inserted": 0}
    from google.cloud import bigquery as b

    c = client or b.Client(project=PROJECT_ID)
    ensure_tables(c)
    ids = rows.prediction_event_id.astype(str).tolist()
    cfg = b.QueryJobConfig(query_parameters=[b.ArrayQueryParameter("ids", "STRING", ids)])
    got = c.query(
        f"SELECT prediction_event_id FROM `{PRED_TABLE}` WHERE prediction_event_id IN UNNEST(@ids)",
        job_config=cfg,
    ).to_dataframe(create_bqstorage_client=False)
    existing = set(got.prediction_event_id.astype(str)) if not got.empty else set()
    new = rows.loc[~rows.prediction_event_id.astype(str).isin(existing)].copy()
    if new.empty:
        return {"status": "NO_NEW_ROWS", "inserted": 0, "existing": len(existing), "production_authority": 0}
    recs = []
    for rec in new.where(pd.notna(new), None).to_dict("records"):
        for k in ("game_start", "quote_timestamp", "prediction_timestamp"):
            v = rec.get(k)
            if isinstance(v, pd.Timestamp):
                rec[k] = v.isoformat()
        recs.append(rec)
    errors = c.insert_rows_json(PRED_TABLE, recs)
    if errors:
        raise RuntimeError("NFL_LEDGER_INSERT_FAILED " + str(errors[:3]))
    return {
        "status": "INSERTED",
        "inserted": int(len(new)),
        "existing": int(len(existing)),
        "ledger_version": LEDGER_VERSION,
        "production_authority": 0,
    }
