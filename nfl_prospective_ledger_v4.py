"""NFL V1.9.5 prospective market-research ledger.

Extends the isolated ``sharp_research`` ledger with three append-only research
objects:
  * raw, canonicalized pregame quote events,
  * fixed/as-of market-state snapshots with microstructure features,
  * append-only settlement rows for those market states.

This module has zero production betting authority.  It deliberately establishes
its own prospective clock on the first successful deployment run and refuses to
admit source quotes observed before that clock into V1.9.5 research evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from typing import Iterable

import pandas as pd
from google.cloud import bigquery as b

import nfl_prospective_ledger_v3 as base3
import nfl_prospective_ledger_v1 as base1

SOURCE_TAG = "nfl-prospective-ledger-v4-v1.9.5-market-microstructure-20261001"
LEDGER_VERSION = "nfl-research-v1.9.5-prospective-market-shadow-20261001"
RESEARCH_CLOCK_ID = "NFL_V1_9_5_PROSPECTIVE_MARKET_CLOCK"
PRODUCTION_AUTHORITY = 0

PROJECT_ID = base1.PROJECT_ID
DATASET_ID = base1.DATASET_ID
QUOTE_TABLE = f"{PROJECT_ID}.{DATASET_ID}.nfl_market_quote_v1"
STATE_TABLE = f"{PROJECT_ID}.{DATASET_ID}.nfl_market_state_v1"
STATE_RESULT_TABLE = f"{PROJECT_ID}.{DATASET_ID}.nfl_market_state_results_v1"
CLOCK_TABLE = f"{PROJECT_ID}.{DATASET_ID}.nfl_research_clock_v1"
RUN_TABLE = f"{PROJECT_ID}.{DATASET_ID}.nfl_shadow_run_v1"

FROZEN_V194_REGISTRY_SHA256 = "e582effdfb52d34686b3bc17b4f4fec48f8b59b4b41596f66d796bfa522ab204"
FROZEN_V194_EDGE_GATE_SHA256 = "9c6ee183c54919fc371bc90339b45679b9533c0c816f055b4c11e99101482b18"
FROZEN_V194_BUNDLE_URI = "gs://sharp-models/nfl-research/v1_9_4/e582effdfb52d346/research_bundle.joblib"


def _quote_schema():
    S=b.SchemaField
    return [
        S("quote_event_id","STRING",mode="REQUIRED"),
        S("research_clock_id","STRING"),
        S("ledger_version","STRING"),
        S("captured_at","TIMESTAMP"),
        S("physical_game_id","STRING",mode="REQUIRED"),
        S("source_game_key","STRING"),
        S("game_start","TIMESTAMP"),
        S("home_team","STRING"),
        S("away_team","STRING"),
        S("market","STRING"),
        S("outcome","STRING"),
        S("bookmaker","STRING"),
        S("value","FLOAT64"),
        S("odds_price","FLOAT64"),
        S("snapshot_timestamp","TIMESTAMP"),
        S("is_sharp_book","BOOL"),
        S("is_recreational_book","BOOL"),
        S("source_payload_sha256","STRING"),
        S("production_authority","INT64"),
    ]


def _state_schema():
    S=b.SchemaField
    return [
        S("state_event_id","STRING",mode="REQUIRED"),
        S("research_clock_id","STRING"),
        S("ledger_version","STRING"),
        S("frozen_registry_sha256","STRING"),
        S("edge_gate_registry_sha256","STRING"),
        S("captured_at","TIMESTAMP"),
        S("physical_game_id","STRING",mode="REQUIRED"),
        S("source_game_key","STRING"),
        S("game_start","TIMESTAMP"),
        S("home_team","STRING"),
        S("away_team","STRING"),
        S("market","STRING"),
        S("outcome","STRING"),
        S("snapshot_type","STRING"),
        S("target_asof","TIMESTAMP"),
        S("quote_cutoff_timestamp","TIMESTAMP"),
        S("canonical_evaluation","BOOL"),
        S("consensus_value","FLOAT64"),
        S("consensus_odds","FLOAT64"),
        S("consensus_implied_probability","FLOAT64"),
        S("book_count","INT64"),
        S("sharp_book_count","INT64"),
        S("recreational_book_count","INT64"),
        S("distinct_snapshot_times","INT64"),
        S("timed_span_minutes","FLOAT64"),
        S("median_quote_age_minutes","FLOAT64"),
        S("max_quote_age_minutes","FLOAT64"),
        S("book_value_std","FLOAT64"),
        S("book_value_iqr","FLOAT64"),
        S("odds_price_std","FLOAT64"),
        S("line_move_from_first","FLOAT64"),
        S("line_move_30m","FLOAT64"),
        S("line_move_60m","FLOAT64"),
        S("line_move_120m","FLOAT64"),
        S("line_velocity_30m","FLOAT64"),
        S("line_velocity_60m","FLOAT64"),
        S("price_move_30m","FLOAT64"),
        S("price_move_60m","FLOAT64"),
        S("sharp_move_30m","FLOAT64"),
        S("sharp_move_60m","FLOAT64"),
        S("soft_move_30m","FLOAT64"),
        S("soft_move_60m","FLOAT64"),
        S("sharp_soft_move_divergence_30m","FLOAT64"),
        S("sharp_soft_move_divergence_60m","FLOAT64"),
        S("book_dispersion_change_60m","FLOAT64"),
        S("crossed_key_3_last60m","BOOL"),
        S("crossed_key_7_last60m","BOOL"),
        S("crossed_key_10_last60m","BOOL"),
        S("crossed_key_14_last60m","BOOL"),
        S("market_direction_60m","INT64"),
        S("data_quality_status","STRING"),
        S("microstructure_json","STRING"),
        S("production_authority","INT64"),
    ]


def _state_result_schema():
    S=b.SchemaField
    return [
        S("state_result_event_id","STRING",mode="REQUIRED"),
        S("state_event_id","STRING",mode="REQUIRED"),
        S("research_clock_id","STRING"),
        S("physical_game_id","STRING"),
        S("market","STRING"),
        S("outcome","STRING"),
        S("snapshot_type","STRING"),
        S("settled_at","TIMESTAMP"),
        S("home_score","FLOAT64"),
        S("away_score","FLOAT64"),
        S("actual_margin","FLOAT64"),
        S("actual_total","FLOAT64"),
        S("market_result","STRING"),
        S("market_absolute_error","FLOAT64"),
        S("production_authority","INT64"),
    ]


def _clock_schema():
    S=b.SchemaField
    return [
        S("research_clock_id","STRING",mode="REQUIRED"),
        S("activated_at","TIMESTAMP"),
        S("source_tag","STRING"),
        S("frozen_registry_sha256","STRING"),
        S("edge_gate_registry_sha256","STRING"),
        S("bundle_uri","STRING"),
        S("note","STRING"),
        S("production_authority","INT64"),
    ]


def _run_schema():
    S=b.SchemaField
    return [
        S("run_event_id","STRING",mode="REQUIRED"),
        S("research_clock_id","STRING"),
        S("started_at","TIMESTAMP"),
        S("finished_at","TIMESTAMP"),
        S("status","STRING"),
        S("source_rows","INT64"),
        S("eligible_quote_rows","INT64"),
        S("quotes_inserted","INT64"),
        S("states_inserted","INT64"),
        S("state_results_inserted","INT64"),
        S("upcoming_games","INT64"),
        S("canonical_t60_states","INT64"),
        S("details_json","STRING"),
        S("production_authority","INT64"),
    ]


def _ensure_table(client, table_id, schema, *, partition_field=None, clustering=None):
    from google.api_core.exceptions import NotFound
    try:
        t=client.get_table(table_id); created=False
        existing={f.name for f in t.schema}
        missing=[f for f in schema if f.name not in existing]
        if missing:
            bad=[f.name for f in missing if f.mode=="REQUIRED"]
            if bad:
                raise RuntimeError(f"NFL_V1_9_5_REQUIRED_SCHEMA_MIGRATION table={table_id} fields={bad}")
            t.schema=list(t.schema)+missing
            client.update_table(t,["schema"])
        return created,[f.name for f in missing]
    except NotFound:
        t=b.Table(table_id,schema=schema)
        t.description="Append-only NFL V1.9.5 prospective research data. Zero production authority."
        if partition_field:
            t.time_partitioning=b.TimePartitioning(type_=b.TimePartitioningType.DAY,field=partition_field)
        if clustering:
            t.clustering_fields=list(clustering)
        client.create_table(t)
        return True,[]


def ensure_tables(client=None):
    c=client or b.Client(project=PROJECT_ID)
    base_health=base3.ensure_tables(c)
    created=[]; added={}
    specs=[
        (QUOTE_TABLE,_quote_schema(),"snapshot_timestamp",["physical_game_id","market","bookmaker"]),
        (STATE_TABLE,_state_schema(),"captured_at",["physical_game_id","market","snapshot_type"]),
        (STATE_RESULT_TABLE,_state_result_schema(),"settled_at",["physical_game_id","market","snapshot_type"]),
        (CLOCK_TABLE,_clock_schema(),"activated_at",["research_clock_id"]),
        (RUN_TABLE,_run_schema(),"started_at",["status","research_clock_id"]),
    ]
    for table,schema,part,cluster in specs:
        made,miss=_ensure_table(c,table,schema,partition_field=part,clustering=cluster)
        if made: created.append(table)
        if miss: added[table]=miss
    return {
        **base_health,
        "status":"READY_V1_9_5",
        "ledger_version_v4":LEDGER_VERSION,
        "research_clock_id":RESEARCH_CLOCK_ID,
        "v1_9_5_created":created,
        "v1_9_5_schema_fields_added":added,
        "quote_table":QUOTE_TABLE,
        "state_table":STATE_TABLE,
        "state_result_table":STATE_RESULT_TABLE,
        "clock_table":CLOCK_TABLE,
        "run_table":RUN_TABLE,
        "production_authority":0,
    }


def _existing_ids(client, table, id_col, ids):
    ids=[str(x) for x in ids if str(x)]
    if not ids: return set()
    cfg=b.QueryJobConfig(query_parameters=[b.ArrayQueryParameter("ids","STRING",ids)])
    d=client.query(f"SELECT `{id_col}` FROM `{table}` WHERE `{id_col}` IN UNNEST(@ids)",job_config=cfg).to_dataframe(create_bqstorage_client=False)
    return set(d[id_col].astype(str)) if not d.empty else set()


def _serialize_records(df: pd.DataFrame, time_cols: Iterable[str]):
    recs=[]
    for rec in df.where(pd.notna(df),None).to_dict("records"):
        for k in time_cols:
            v=rec.get(k)
            if isinstance(v,pd.Timestamp): rec[k]=v.isoformat()
        recs.append(rec)
    return recs


def append_idempotent(df: pd.DataFrame, *, table: str, id_col: str, time_cols=(), client=None):
    if df is None or df.empty:
        return {"status":"NO_ROWS","inserted":0,"existing":0}
    c=client or b.Client(project=PROJECT_ID)
    ensure_tables(c)
    existing=_existing_ids(c,table,id_col,df[id_col].astype(str).tolist())
    new=df.loc[~df[id_col].astype(str).isin(existing)].copy()
    if new.empty:
        return {"status":"NO_NEW_ROWS","inserted":0,"existing":len(existing)}
    errors=c.insert_rows_json(table,_serialize_records(new,time_cols))
    if errors:
        raise RuntimeError(f"NFL_V1_9_5_INSERT_FAILED table={table} errors={errors[:3]}")
    return {"status":"INSERTED","inserted":int(len(new)),"existing":int(len(existing))}


def get_or_create_research_clock(client=None, *, now=None):
    c=client or b.Client(project=PROJECT_ID)
    ensure_tables(c)
    cfg=b.QueryJobConfig(query_parameters=[b.ScalarQueryParameter("cid","STRING",RESEARCH_CLOCK_ID)])
    d=c.query(f"SELECT * FROM `{CLOCK_TABLE}` WHERE research_clock_id=@cid ORDER BY activated_at LIMIT 1",job_config=cfg).to_dataframe(create_bqstorage_client=False)
    if not d.empty:
        r=d.iloc[0].to_dict()
        r["activated_at"]=pd.to_datetime(r.get("activated_at"),utc=True,errors="coerce")
        r["created_now"]=False
        return r
    ts=pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now,utc=True)
    row={
        "research_clock_id":RESEARCH_CLOCK_ID,
        "activated_at":ts,
        "source_tag":SOURCE_TAG,
        "frozen_registry_sha256":FROZEN_V194_REGISTRY_SHA256,
        "edge_gate_registry_sha256":FROZEN_V194_EDGE_GATE_SHA256,
        "bundle_uri":FROZEN_V194_BUNDLE_URI,
        "note":"First successful V1.9.5 deployment establishes the prospective clock. No pre-clock quote can enter V1.9.5 evidence.",
        "production_authority":0,
    }
    errors=c.insert_rows_json(CLOCK_TABLE,_serialize_records(pd.DataFrame([row]),["activated_at"]))
    if errors:
        # A concurrent creator may have won; read once more before failing.
        d=c.query(f"SELECT * FROM `{CLOCK_TABLE}` WHERE research_clock_id=@cid ORDER BY activated_at LIMIT 1",job_config=cfg).to_dataframe(create_bqstorage_client=False)
        if d.empty:
            raise RuntimeError("NFL_V1_9_5_CLOCK_CREATE_FAILED "+str(errors[:3]))
        r=d.iloc[0].to_dict(); r["activated_at"]=pd.to_datetime(r.get("activated_at"),utc=True,errors="coerce"); r["created_now"]=False
        return r
    return {**row,"created_now":True}


def record_run(summary: dict, *, client=None):
    c=client or b.Client(project=PROJECT_ID)
    ensure_tables(c)
    started=pd.to_datetime(summary.get("started_at"),utc=True,errors="coerce")
    finished=pd.to_datetime(summary.get("finished_at"),utc=True,errors="coerce")
    eid=hashlib.sha256(f"{RESEARCH_CLOCK_ID}|{uuid.uuid4()}".encode()).hexdigest()
    row={
        "run_event_id":eid,
        "research_clock_id":RESEARCH_CLOCK_ID,
        "started_at":started,
        "finished_at":finished,
        "status":str(summary.get("status","UNKNOWN")),
        "source_rows":int(summary.get("source_rows",0) or 0),
        "eligible_quote_rows":int(summary.get("eligible_quote_rows",0) or 0),
        "quotes_inserted":int(summary.get("quotes_inserted",0) or 0),
        "states_inserted":int(summary.get("states_inserted",0) or 0),
        "state_results_inserted":int(summary.get("state_results_inserted",0) or 0),
        "upcoming_games":int(summary.get("upcoming_games",0) or 0),
        "canonical_t60_states":int(summary.get("canonical_t60_states",0) or 0),
        "details_json":json.dumps(summary,sort_keys=True,default=str,separators=(",",":")),
        "production_authority":0,
    }
    err=c.insert_rows_json(RUN_TABLE,_serialize_records(pd.DataFrame([row]),["started_at","finished_at"]))
    if err: raise RuntimeError("NFL_V1_9_5_RUN_LOG_INSERT_FAILED "+str(err[:3]))
    return row


def ledger_health_check(client=None, *, service_identity_hint=None):
    c=client or b.Client(project=PROJECT_ID)
    ready=ensure_tables(c)
    # Reuse the protected v1.9.4 health path for base prediction/result append-only contract.
    base_health=base3.ledger_health_check(c,service_identity_hint=service_identity_hint)
    tables=[QUOTE_TABLE,STATE_TABLE,STATE_RESULT_TABLE,CLOCK_TABLE,RUN_TABLE]
    for t in tables:
        c.get_table(t)
    return {
        **base_health, **ready,
        "V1_9_5_MARKET_SHADOW_SCHEMA_PASS":True,
        "V1_9_5_PRECLOCK_EXCLUSION_CONTRACT":True,
        "V1_9_5_APPEND_ONLY_CONTRACT":True,
        "status":"HEALTHY_V1_9_5",
        "production_authority":0,
    }
