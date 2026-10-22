"""NCAAF V5.5 ONLY: backend market refresh and atomic BigQuery board publication.

This module never calls the Production V1 model, selector, or ledgers. It does
not recalculate probabilities, make synthetic prices, or promote research bets.
The installed ncaaf_v552_live.py is the sole source of V5.5 forecasts and
market-join logic. Its return contract is verified before any publication.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone

TABLE = os.getenv("NCAAF_V55_BOARD_TABLE", "sharplogger.sharp_data.ncaaf_v55_live_board")
SOURCE_STATUS = "COMPLETE_V55_INDEPENDENT_WEEKLY_SHADOW"


def _utcnow():
    return datetime.now(timezone.utc)


def _json_default(obj):
    try:
        import numpy as np
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
    except ImportError:
        pass
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def _verify_audit(audit):
    if not isinstance(audit, dict):
        raise RuntimeError("V55_AUDIT_NOT_DICTIONARY")
    if audit.get("status") != SOURCE_STATUS:
        raise RuntimeError("V55_WEEKLY_NOT_VERIFIED:" + str(audit.get("status")))
    if audit.get("production_v1_probabilities_used") is not False:
        raise RuntimeError("V55_PROBABILITY_PROVENANCE_UNVERIFIED")
    if audit.get("v55_bet_authority", 0) != 0:
        raise RuntimeError("V55_UNREVIEWED_BET_AUTHORITY_NONZERO")
    return True


def _ensure_table(client):
    from google.cloud import bigquery
    from google.api_core.exceptions import NotFound
    try:
        existing = client.get_table(TABLE)
        cols = {f.name for f in existing.schema}
        expected = {"snapshot_id", "published_at", "record_kind", "payload_json", "audit_json", "row_count", "payload_sha256", "publication_type"}
        if not expected.issubset(cols):
            raise RuntimeError("V55_BQ_TABLE_SCHEMA_MISMATCH:" + ",".join(sorted(expected-cols)))
        return
    except NotFound:
        pass
    fields = [
        bigquery.SchemaField("snapshot_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("published_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("record_kind", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("payload_json", "STRING"),
        bigquery.SchemaField("audit_json", "STRING"),
        bigquery.SchemaField("row_count", "INT64"),
        bigquery.SchemaField("payload_sha256", "STRING"),
        bigquery.SchemaField("publication_type", "STRING"),
    ]
    client.create_table(bigquery.Table(TABLE, schema=fields), exists_ok=True)


def publish_dataframe(weekly, audit, *, bq_client=None, publication_type="WEEKLY"):
    """Publish a whole V5.5 forecast only after its source assertions pass.

    A DATA row contains the exact V5.5 DataFrame. A COMPLETE record is inserted
    only after DATA succeeded. Readers require both, row count and SHA-256 parity.
    This is a display journal, not a substitute for immutable bet/settlement logs.
    """
    import pandas as pd
    from google.cloud import bigquery
    _verify_audit(audit)
    if not isinstance(weekly, pd.DataFrame):
        raise RuntimeError("V55_FORECAST_NOT_DATAFRAME")
    if len(weekly) > 10000:
        raise RuntimeError("V55_BOARD_UNEXPECTED_ROW_VOLUME")
    publication_type = str(publication_type).upper()
    if publication_type not in ("WEEKLY", "MARKET_REFRESH"):
        raise ValueError("V55_INVALID_PUBLICATION_TYPE")
    safe = weekly.copy().replace([float("inf"), -float("inf")], None)
    # to_json converts datetimes/NaN to transport-safe values; do not synthesize
    # missing price fields or normalize data away from the model's own results.
    payload = safe.to_json(orient="records", date_format="iso", date_unit="ms")
    if len(payload.encode("utf-8")) > 8_000_000:
        raise RuntimeError("V55_BOARD_PAYLOAD_OVER_8MB")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    snapshot_id = uuid.uuid4().hex
    now = _utcnow().isoformat()
    audit_json = json.dumps(audit, default=_json_default, sort_keys=True, allow_nan=False)
    if len(audit_json.encode("utf-8")) > 400000:
        raise RuntimeError("V55_AUDIT_PAYLOAD_TOO_LARGE")
    base = {"snapshot_id":snapshot_id, "published_at":now,
            "row_count":int(len(weekly)), "payload_sha256":digest,
            "publication_type":publication_type}
    client = bq_client or bigquery.Client(project="sharplogger")
    _ensure_table(client)
    data = {**base, "record_kind":"DATA", "payload_json":payload, "audit_json":None}
    try:
        errors = client.insert_rows_json(TABLE,[data],row_ids=[snapshot_id+":DATA"])
    except Exception as exc:
        # No silent fallback to V1, no dashboard cache masquerading as a publish.
        if "bigquery.tables.updateData" in str(exc) or "Access Denied: Table" in str(exc):
            raise RuntimeError(
                "V55_BQ_WRITE_PERMISSION_DENIED: table="+TABLE+
                "; the Cloud Run execution service account needs BigQuery write "
                "access (bigquery.tables.updateData), normally via "
                "roles/bigquery.dataEditor on dataset sharplogger.sharp_data. "
                "Check that the table exists and its schema is correct. "
                "The saved V5.5 forecast can be republished without a CORE refit."
            ) from exc
        raise
    if errors:
        raise RuntimeError("V55_BQ_DATA_WRITE_FAILED:" + str(errors)[:600])
    # Verify the staged DATA is query-visible before exposing a COMPLETE marker.
    q = f"SELECT payload_sha256,row_count FROM `{TABLE}` WHERE snapshot_id=@sid AND record_kind='DATA'"
    cfg=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("sid","STRING",snapshot_id)])
    matches=[]
    for attempt in range(4):
        matches=list(client.query(q,job_config=cfg).result())
        if len(matches)==1 and matches[0].payload_sha256==digest and int(matches[0].row_count)==len(weekly):
            break
        if attempt < 3:
            import time
            time.sleep(1.0)
    if len(matches)!=1 or matches[0].payload_sha256!=digest or int(matches[0].row_count)!=len(weekly):
        raise RuntimeError("V55_BQ_DATA_NOT_QUERY_VISIBLE")
    commit = {**base, "record_kind":"COMPLETE", "payload_json":None,"audit_json":audit_json}
    try:
        errors=client.insert_rows_json(TABLE,[commit],row_ids=[snapshot_id+":COMPLETE"])
    except Exception as exc:
        if "bigquery.tables.updateData" in str(exc) or "Access Denied: Table" in str(exc):
            raise RuntimeError(
                "V55_BQ_COMMIT_WRITE_PERMISSION_DENIED: table="+TABLE+
                "; service account needs bigquery.tables.updateData on dataset sharplogger.sharp_data. "
                "Only COMPLETE snapshots are displayed; the partial DATA row cannot appear."
            ) from exc
        raise
    if errors:
        raise RuntimeError("V55_BQ_COMMIT_FAILED:" + str(errors)[:600])
    result={"status":"PASS","snapshot_id":snapshot_id,"rows":int(len(weekly)),
            "priced_games":int(audit.get("priced_games",0) or 0),"type":publication_type,
            "published_at":now,"v55_bet_authority":0}
    logging.info("[NCAAF-V55-BOARD-PUBLISH] %s",json.dumps(result,sort_keys=True))
    return result


def publish_weekly_latest(*, storage_client=None, bq_client=None, bucket_name=None, publication_type="WEEKLY"):
    from google.cloud import storage
    import ncaaf_v552_live as v55
    gcs = storage_client or storage.Client()
    weekly, audit = v55.read_weekly_latest(gcs,bucket_name=bucket_name or os.getenv("MODEL_BUCKET","sharp-models"))
    return publish_dataframe(weekly,audit,bq_client=bq_client,publication_type=publication_type)


def market_refresh_and_publish(*, storage_client=None, bq_client=None, bucket_name=None, log_func=None):
    """Backend-only V5.5 market repricing. No CORE refit or V1 fallback."""
    from google.cloud import storage
    import ncaaf_v552_live as v55
    log = log_func or logging.info
    gcs = storage_client or storage.Client()
    bucket = bucket_name or os.getenv("MODEL_BUCKET","sharp-models")
    refresh = v55.refresh_market_only(storage_client=gcs,bucket_name=bucket,log_func=log)
    if not isinstance(refresh,dict) or refresh.get("status")!=SOURCE_STATUS:
        raise RuntimeError("V55_MARKET_REFRESH_INCOMPLETE:"+str(getattr(refresh,"get",lambda *_:None)("status")))
    if refresh.get("core_refit") is not False or refresh.get("production_v1_probabilities_used") is not False:
        raise RuntimeError("V55_MARKET_REFRESH_FROZEN_CORE_CONTRACT_FAIL")
    if refresh.get("v55_bet_authority",0) != 0:
        raise RuntimeError("V55_MARKET_REFRESH_UNREVIEWED_BET_AUTHORITY")
    pub = publish_weekly_latest(storage_client=gcs,bq_client=bq_client,bucket_name=bucket,publication_type="MARKET_REFRESH")
    log("[NCAAF-V55-BACKGROUND-REFRESH] "+json.dumps(pub,sort_keys=True))
    return pub


def read_published_latest(*, bq_client=None, max_market_age_minutes=180):
    """Read newest COMPLETE V5.5 publication from BigQuery; never infer bets."""
    from google.cloud import bigquery
    import pandas as pd
    client=bq_client or bigquery.Client(project="sharplogger")
    q=f"""SELECT snapshot_id,published_at,record_kind,payload_json,audit_json,
              row_count,payload_sha256,publication_type
        FROM `{TABLE}`
        WHERE snapshot_id=(SELECT snapshot_id FROM `{TABLE}`
                           WHERE record_kind='COMPLETE'
                           ORDER BY published_at DESC LIMIT 1)"""
    rows=list(client.query(q).result())
    if len(rows)!=2 or {str(r.record_kind) for r in rows}!={"DATA","COMPLETE"}:
        raise RuntimeError("V55_BQ_PUBLICATION_INCOMPLETE_OR_MISSING")
    data=next(r for r in rows if r.record_kind=="DATA")
    commit=next(r for r in rows if r.record_kind=="COMPLETE")
    if str(data.snapshot_id)!=str(commit.snapshot_id) or data.payload_sha256!=commit.payload_sha256 or data.row_count!=commit.row_count:
        raise RuntimeError("V55_BQ_SNAPSHOT_IDENTITY_MISMATCH")
    if hashlib.sha256(data.payload_json.encode("utf-8")).hexdigest()!=commit.payload_sha256:
        raise RuntimeError("V55_BQ_SNAPSHOT_HASH_MISMATCH")
    audit=json.loads(commit.audit_json)
    _verify_audit(audit)
    vals=json.loads(data.payload_json)
    if not isinstance(vals,list) or len(vals)!=int(commit.row_count):
        raise RuntimeError("V55_BQ_ROW_COUNT_MISMATCH")
    frame=pd.DataFrame(vals)
    published=pd.to_datetime(commit.published_at,utc=True,errors="coerce")
    if pd.isna(published):
        raise RuntimeError("V55_BQ_PUBLISHED_AT_INVALID")
    age=(_utcnow()-published.to_pydatetime()).total_seconds()/60.0
    stale=age>max_market_age_minutes or age< -5
    if stale:
        # Model-only slate may remain visible; current price/EV/action must not.
        price_fields={"Current_Spread","Odds_American","CORE_Cover_Probability","CORE_Edge","CORE_EV","CORE_Project_Cover_Points","V55_Bet_Decision","Market_Rich_Sharp_Move","Market_Rich_Limit","Quote_State"}
        for col in price_fields.intersection(frame.columns):
            frame[col]=None
        frame["Quote_State"]="STALE_BACKEND_SNAPSHOT"
        frame["V55_Bet_Decision"]="NO CURRENT QUOTE — STALE"
    meta={**audit,"publication_status":"STALE_MARKET" if stale else "PASS",
          "publication_type":str(commit.publication_type),
          "published_at_utc":published.isoformat(),"published_age_minutes":round(age,1),
          "published_rows":len(frame),"snapshot_id":str(commit.snapshot_id),
          "v55_bet_authority":0}
    return frame,meta
