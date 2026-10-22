"""NCAAF V5.5 backend-only model publication and scanner-quote handoff.

The installed ncaaf_v552_live.py supplies frozen CORE forecasts. The backend
quote bridge replaces its unverified market join with fresh, independently
validated, existing scanner quotes before publication. No V1 fallback.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import unicodedata
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
    from google.cloud import bigquery
    client = bq_client or bigquery.Client(project="sharplogger")
    # Always reattach the current stored sportsbook quotes on the backend;
    # weekly and market-only workflows share exactly the same price authority.
    weekly, audit = bridge_weekly_prices(weekly,audit,bq_client=client)
    return publish_dataframe(weekly,audit,bq_client=client,publication_type=publication_type)


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


# ----- Backend scanner quote adapter (single-module V5.5 deployment) -----


import math
import re
import unicodedata

MARKET_TABLE = "sharplogger.sharp_data.sharp_moves_master"
QUOTE_MAX_AGE_MINUTES = 180
KICKOFF_TOLERANCE_MINUTES = 60


def _norm_team(value):
    if value is None:
        return ""
    val = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    val = val.lower().replace("&", "and")
    val = re.sub(r"[^a-z0-9]+", " ", val)
    return " ".join(val.split())


def _matchup_teams(row):
    """Return (home,away), orientation_known. Reject ambiguous string formats."""
    home = _norm_team(row.get("Home_Team") or row.get("Home_Team_Norm"))
    away = _norm_team(row.get("Away_Team") or row.get("Away_Team_Norm"))
    if home and away and home != away:
        return (home, away), True
    value = str(row.get("Matchup") or "").strip()
    # 'away @ home'/'away at home' are unambiguous.  'vs' may differ by source;
    # do not assume a home side for point-edge calculations based on it.
    m = re.match(r"^(.+?)\s+(?:@|at)\s+(.+)$", value, flags=re.I)
    if m:
        return (_norm_team(m.group(2)), _norm_team(m.group(1))), True
    m = re.match(r"^(.+?)\s+vs\.?\s+(.+)$", value, flags=re.I)
    if m:
        return (_norm_team(m.group(1)), _norm_team(m.group(2))), False
    return ("", ""), False


def _timestamp(value):
    import pandas as pd
    return pd.to_datetime(value, errors="coerce", utc=True)


def _price(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _valid_american(value):
    price = _price(value)
    return price is not None and (price <= -100 or price >= 100) and abs(price) <= 100000


def read_scanner_quotes(bq_client, *, minutes=QUOTE_MAX_AGE_MINUTES, horizon_days=14):
    """Fetch ONLY stored scanner odds. An API call must never happen here."""
    from google.cloud import bigquery
    if not 5 <= minutes <= 1440:
        raise ValueError("quote window must be 5..1440 minutes")
    query = f"""
    SELECT Game_Start,Home_Team_Norm,Away_Team_Norm,Outcome,Outcome_Norm,
           Value,Odds_Price,Bookmaker,Book,Snapshot_Timestamp,Game_Key
    FROM `{MARKET_TABLE}`
    WHERE UPPER(TRIM(CAST(Sport AS STRING)))='NCAAF'
      AND LOWER(TRIM(CAST(Market AS STRING)))='spreads'
      AND Game_Start > CURRENT_TIMESTAMP()
      AND Game_Start <= TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL @horizon DAY)
      AND Snapshot_Timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @minutes MINUTE)
      AND Snapshot_Timestamp <= CURRENT_TIMESTAMP()
      AND Snapshot_Timestamp < Game_Start
      AND SAFE_CAST(Value AS FLOAT64) IS NOT NULL
      AND SAFE_CAST(Odds_Price AS FLOAT64) IS NOT NULL
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("minutes", "INT64", minutes),
        bigquery.ScalarQueryParameter("horizon", "INT64", horizon_days),
    ])
    return bq_client.query(query, job_config=cfg).to_dataframe(create_bqstorage_client=False)


def attach_verified_backend_quotes(forecasts, quotes, *, now=None, max_age_minutes=QUOTE_MAX_AGE_MINUTES):
    """Return V5.5 forecast with backend quotes and detailed provenance diagnostics.

    Joins on physical game participants + near-identical kickoff + actual outcome.
    Multiple ambiguous matches fail closed. Does NOT calculate bet recommendations.
    """
    import pandas as pd
    if not isinstance(forecasts, pd.DataFrame):
        raise TypeError("forecasts must be a DataFrame")
    out = forecasts.copy()
    report = {
        "backend_price_source": MARKET_TABLE,
        "backend_quotes_read": len(quotes) if isinstance(quotes, pd.DataFrame) else 0,
        "core_forecast_rows": len(out), "upcoming_forecast_rows": 0,
        "matched_quote_rows": 0, "priced_games": 0,
        "unmatched_forecast_rows": 0, "ambiguous_quote_rows": 0,
        "unresolved_forecast_identity": 0, "no_matching_game": 0,
        "no_selected_side_quote": 0, "stale_or_invalid_quote_rows": 0,
        "unmatched_examples": [], "status": "NO_UPCOMING_FORECASTS",
    }
    if out.empty:
        return out, report
    # Existing V5.5 model-derived pricing is not reliable if the backend prices
    # against which it was calculated differ. The bridge invalidates derived
    # probabilities/EV and never repurposes a legacy V1 recommendation.
    for col in ("Current_Spread", "Odds_American", "CORE_Cover_Probability", "CORE_Edge",
                "CORE_EV", "CORE_Project_Cover_Points", "Backend_Quote_Book",
                "Backend_Quote_Timestamp", "Backend_Quote_Game_Key"):
        out[col] = None
    out["Quote_State"] = "NO_VERIFIED_BACKEND_QUOTE"
    out["V55_Bet_Decision"] = "RESEARCH_ONLY_NO_AUTHORITY"

    now = _timestamp(now if now is not None else datetime.now(timezone.utc))
    if pd.isna(now):
        raise ValueError("invalid now")
    if "Game_Start_UTC" not in out.columns:
        report["status"] = "V55_MISSING_KICKOFF_FIELD"
        return out, report
    kickoff = pd.to_datetime(out["Game_Start_UTC"], errors="coerce", utc=True)
    upcoming = kickoff.gt(now) & kickoff.le(now + pd.Timedelta(days=14))
    report["upcoming_forecast_rows"] = int(upcoming.sum())
    if not upcoming.any():
        return out, report

    required = {"Game_Start", "Home_Team_Norm", "Away_Team_Norm", "Outcome", "Value",
                "Odds_Price", "Snapshot_Timestamp"}
    if not isinstance(quotes, pd.DataFrame) or not required.issubset(quotes.columns):
        report["status"] = "BACKEND_QUOTE_SCHEMA_MISSING"
        report["missing_backend_fields"] = sorted(required - set(getattr(quotes, "columns", [])))
        report["unmatched_forecast_rows"] = report["upcoming_forecast_rows"]
        return out, report
    q = quotes.copy()
    q["_kick"] = pd.to_datetime(q["Game_Start"], errors="coerce", utc=True)
    q["_snap"] = pd.to_datetime(q["Snapshot_Timestamp"], errors="coerce", utc=True)
    q["_home"] = q["Home_Team_Norm"].map(_norm_team)
    q["_away"] = q["Away_Team_Norm"].map(_norm_team)
    q["_selected"] = q["Outcome"].map(_norm_team)
    q["_line"] = pd.to_numeric(q["Value"],errors="coerce")
    q["_american"] = pd.to_numeric(q["Odds_Price"],errors="coerce")
    if "Bookmaker" not in q.columns:
        q["Bookmaker"] = q.get("Book", "")
    q["_book"] = q["Bookmaker"].fillna("").astype(str).str.lower().str.strip()
    valid = (q["_kick"].gt(now) & q["_kick"].le(now + pd.Timedelta(days=14)) &
             q["_snap"].le(now) & q["_snap"].lt(q["_kick"]) &
             q["_snap"].ge(now - pd.Timedelta(minutes=max_age_minutes)) &
             q["_home"].ne("") & q["_away"].ne("") & q["_home"].ne(q["_away"]) &
             q["_selected"].isin(pd.concat([q["_home"],q["_away"]]).unique()) &
             q["_line"].between(-100,100) &
             q["_american"].map(_valid_american) & q["_book"].ne(""))
    # Verify each outcome is one of ITS OWN matchup's participants.
    valid &= (q["_selected"].eq(q["_home"]) | q["_selected"].eq(q["_away"]))
    report["stale_or_invalid_quote_rows"] = int((~valid).sum())
    q = q.loc[valid].copy()
    if q.empty:
        report["status"] = "NO_VALID_FRESH_BACKEND_QUOTES"
        report["unmatched_forecast_rows"] = report["upcoming_forecast_rows"]
        return out, report
    # Scanner may write historical snapshots in addition to latest lines;
    # keep latest per book + game + selected outcome, rather than a stale opener.
    q = (q.sort_values("_snap",ascending=False)
           .drop_duplicates(["_home", "_away", "_kick", "_book", "_selected"], keep="first"))
    q["_team_pair"] = q.apply(lambda r: tuple(sorted((r["_home"],r["_away"]))),axis=1)
    matches = 0
    unique_games = set()
    def _note_unmatched(idx, reason):
        if len(report["unmatched_examples"]) < 8:
            report["unmatched_examples"].append({
                "matchup": str(out.at[idx,"Matchup"])[:100] if "Matchup" in out.columns else "",
                "selected": str(out.at[idx,"Selected_Team"])[:80] if "Selected_Team" in out.columns else "",
                "kickoff_utc": str(kickoff.loc[idx]), "reason": reason,
            })
    for idx in out.index[upcoming]:
        row = out.loc[idx]
        (home,away), oriented = _matchup_teams(row)
        selected = _norm_team(row.get("Selected_Team"))
        if not home or not away or home == away or selected not in (home,away):
            report["unresolved_forecast_identity"] += 1
            out.at[idx,"Quote_State"] = "UNRESOLVED_GAME_OR_SELECTED_SIDE"
            _note_unmatched(idx, "UNRESOLVED_GAME_OR_SELECTED_SIDE")
            continue
        game = q[q["_team_pair"].map(lambda p: p == tuple(sorted((home,away))))]
        if game.empty:
            report["no_matching_game"] += 1
            _note_unmatched(idx, "NO_MATCHING_PARTICIPANTS")
            continue
        # Guard against one team-pair playing on multiple different dates.
        delta = (game["_kick"] - kickoff.loc[idx]).abs()
        game = game[delta.le(pd.Timedelta(minutes=KICKOFF_TOLERANCE_MINUTES))]
        if game.empty:
            report["no_matching_game"] += 1
            _note_unmatched(idx, "KICKOFF_OUTSIDE_60_MINUTES")
            continue
        # Match actual sportsbook outcome; never infer side from spread sign.
        sides = game[game["_selected"] == selected]
        if sides.empty:
            report["no_selected_side_quote"] += 1
            _note_unmatched(idx, "NO_SPREAD_QUOTE_FOR_SELECTED_SIDE")
            continue
        candidates = sides[["_home","_away","_kick"]].drop_duplicates()
        if len(candidates) != 1:
            report["ambiguous_quote_rows"] += 1
            out.at[idx,"Quote_State"] = "AMBIGUOUS_BACKEND_GAME"
            _note_unmatched(idx, "AMBIGUOUS_BACKEND_GAME")
            continue
        # Different books have different current lines. For display choose the
        # most recent quote, not a synthetic consensus or a best-price bet.
        chosen = sides.sort_values(["_snap","_book"],ascending=[False,True]).iloc[0]
        if oriented and (home != chosen["_home"] or away != chosen["_away"]):
            report["unresolved_forecast_identity"] += 1
            out.at[idx,"Quote_State"] = "HOME_AWAY_ORIENTATION_CONFLICT"
            _note_unmatched(idx, "HOME_AWAY_ORIENTATION_CONFLICT")
            continue
        out.at[idx,"Current_Spread"] = float(chosen["_line"])
        out.at[idx,"Odds_American"] = float(chosen["_american"])
        out.at[idx,"Backend_Quote_Book"] = chosen["_book"]
        out.at[idx,"Backend_Quote_Timestamp"] = chosen["_snap"].isoformat()
        out.at[idx,"Backend_Quote_Game_Key"] = str(chosen.get("Game_Key") or "")
        out.at[idx,"Quote_State"] = "FRESH_VERIFIED_BACKEND_SPREAD"
        # Signed margin arithmetic is safe only when home/away is known.
        home_margin = _price(row.get("CORE_Projected_Home_Margin"))
        if oriented and home_margin is not None:
            selected_margin = home_margin if selected==home else -home_margin
            out.at[idx,"CORE_Project_Cover_Points"] = selected_margin + float(chosen["_line"])
        matches += 1
        unique_games.add((tuple(sorted((home,away))), chosen["_kick"].isoformat()))
    report["matched_quote_rows"] = matches
    report["priced_games"] = len(unique_games)
    report["unmatched_forecast_rows"] = report["upcoming_forecast_rows"] - matches
    report["status"] = "PASS" if matches else "NO_MATCHING_UPCOMING_MODEL_AND_BACKEND_PRICES"
    return out, report


def bridge_weekly_prices(weekly, audit, *, bq_client, log_func=None):
    """Enforce one backend price authority for both weekly and market refresh."""
    import json
    import pandas as pd
    log = log_func or logging.info
    try:
        market = read_scanner_quotes(bq_client)
    except Exception as exc:
        # Failure is visible, and the output is model-only. No old prices survive.
        market = pd.DataFrame()
        error = f"{type(exc).__name__}:{str(exc)[:180]}"
    else:
        error = None
    out, diag = attach_verified_backend_quotes(weekly, market)
    if error:
        diag["backend_read_error"] = error
        diag["status"] = "BACKEND_QUERY_ERROR_FAIL_CLOSED"
    updated = dict(audit)
    updated.update(diag)
    updated["v55_bet_authority"] = 0
    updated["price_authority"] = "SHARP_SCANNER_BIGQUERY_ONLY"
    updated["price_join_semantics"] = "FRESH_QUOTE_MATCH_ONLY_NO_V55_PROBABILITY_RECOMPUTE"
    log("[NCAAF-V55-BACKEND-PRICE-BRIDGE] " + json.dumps(diag,sort_keys=True))
    return out, updated
