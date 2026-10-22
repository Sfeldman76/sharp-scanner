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
    # Verify immutable source BEFORE a quote join can update the audit payload.
    _verify_audit(audit)
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
        price_fields={"Current_Spread","Odds_American","CORE_Cover_Probability","CORE_Edge","CORE_EV","CORE_Project_Cover_Points","V55_Bet_Decision","Market_Rich_Sharp_Move","Market_Rich_Limit","Quote_State",
             "Selected_Team","Backend_Home_Spread","Backend_Away_Spread",
             "Backend_Home_Odds_American","Backend_Away_Odds_American",
             "Backend_Quote_Book","Backend_Quote_Timestamp",
             "V55_Raw_Cover_Prob","V55_Raw_EV","V55_Raw_Edge",
             "V55_Research_Selected_Team","V55_Research_Lean_Team",
             "V55_Research_Decision","V55_Probability_State",
             "V55_Research_Selected_Side","V55_Home_Raw_Prob",
             "V55_Away_Raw_Prob","V55_Home_Raw_EV","V55_Away_Raw_EV",
             "V55_Trigger_Context","V55_Trigger_Context_Authority",
             "Backend_Model_Lean_Side","Backend_Model_Lean_Points",
             "Backend_Projected_Home_Cover_Points","Backend_Projected_Away_Cover_Points",
             "CORE_EV","CORE_Edge","CORE_Cover_Probability","Book"}
        for col in price_fields.intersection(frame.columns):
            frame[col]=None
        frame["Quote_State"]="STALE_BACKEND_SNAPSHOT"
        frame["V55_Probability_State"]="STALE_BACKEND_SNAPSHOT"
        frame["V55_Research_Decision"]="NO_CURRENT_QUOTE__STALE"
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


# Explicit equivalents and generic mascot stripping are intentionally separate.
# No fuzzy matching: a verified matchup needs BOTH teams, home/away orientation,
# an unambiguous sportsbook identity and compatible kickoff time.
SCHOOL_EQUIVALENTS = {
    "app state": "appalachian state",
    "appalachian st": "appalachian state",
    "uconn": "connecticut",
    "umass": "massachusetts",
    "massachusetts amherst": "massachusetts",
    "nc state": "north carolina state",
    "n c state": "north carolina state",
    "north carolina st": "north carolina state",
    "ole miss": "mississippi",
    "southern miss": "southern mississippi",
    "usm": "southern mississippi",
    "miami oh": "miami ohio",
    "miami of ohio": "miami ohio",
    "miami florida": "miami",
    "miami fl": "miami",
    "usc": "southern california",
    "south carolina": "south carolina",
    "smu": "southern methodist",
    "tcu": "texas christian",
    "ucf": "central florida",
    "fau": "florida atlantic",
    "fiu": "florida international",
    "utsa": "texas san antonio",
    "utep": "texas el paso",
    "unlv": "nevada las vegas",
    "byu": "brigham young",
    "lsu": "louisiana state",
    "ul monroe": "louisiana monroe",
    "ulm": "louisiana monroe",
    "louisiana lafayette": "louisiana",
    "ul lafayette": "louisiana",
    "ul louisiana": "louisiana",
    "middle tennessee state": "middle tennessee",
    "mtsu": "middle tennessee",
    "western kentucky": "western kentucky",
    "wku": "western kentucky",
    "boise st": "boise state",
    "san jose st": "san jose state",
    "fresno st": "fresno state",
    "florida st": "florida state",
    "penn st": "penn state",
    "iowa st": "iowa state",
    "kansas st": "kansas state",
    "michigan st": "michigan state",
    "mississippi st": "mississippi state",
    "ohio st": "ohio state",
    "oklahoma st": "oklahoma state",
    "oregon st": "oregon state",
    "washington st": "washington state",
    "utah st": "utah state",
    "texas st": "texas state",
    "texas am": "texas a and m",
    "texas a m": "texas a and m",
    "texas a and m": "texas a and m",
    "hawaii": "hawaii",
    "hawai i": "hawaii",
    "army west point": "army",
    "navy midshipmen": "navy",
}

# Mascots are suffixes, never entire-school substrings. Longer first protects
# "golden eagles" from the shorter "eagles" suffix, etc.
NCAAF_MASCOTS = tuple(sorted({m.strip() for m in """
    black knights, blue devils, golden hurricane, fighting irish, fighting illini,
    fighting hawks, rainbow warriors, golden eagles, golden flashes, golden gophers,
    green wave, red wolves, red raiders, redhawks, mean green, blue raiders,
    blue hens, blue jays, scarlet knights, crimson tide, fighting camels,
    raging cajuns, ragin cajuns, horned frogs, fighting hawks, fighting scots,
    mountaineers, minutemen, monarchs, bulldogs, wildcats, tigers, lions, cougars,
    spartans, buckeyes, wolverines, broncos, mustangs, knights, owls, pirates,
    flames, dukes, huskies, cavaliers, cardinals, eagles, falcons, bears,
    bruins, boilermakers, panthers, terrapins, hoosiers, badgers, gators,
    seminoles, volunteers, hurricanes, cowboys, cowgirls, roadrunners, aggies,
    hornets, bobcats, beavers, buffaloes, bulls, lobos, warriors, ramblers,
    rams, tar heels, hilltoppers, razorbacks, rebels, commodores, trojans,
    gamecocks, yellow jackets, demons, demon deacons, jayhawks, longhorns,
    redbirds, bulldogs, redhawks, thundering herd, sun devils, greenies,
    vandals, chanticleers, rockets, zips, retrievers, warhawks, shockers,
    orange, wolfpack, wolf pack, wolfpack, bluejays, midshipmen, nittany lions,
    49ers, miners, golden bears, fighting saints, green knights, golden knights,
    fighting broncos, aztecs, rainbow warriors, golden panthers,
""".split(",") if m.strip()}, key=len, reverse=True))


def _basic_team(value):
    if value is None:
        return ""
    val = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    val = val.lower().replace("&", " and ")
    val = re.sub(r"[^a-z0-9]+", " ", val)
    return " ".join(val.split())


def _norm_team(value):
    school = _basic_team(value)
    if not school or school in ("none", "nan", "null"):
        return ""
    school = SCHOOL_EQUIVALENTS.get(school, school)
    for mascot in NCAAF_MASCOTS:
        suffix = " " + mascot.strip()
        if school.endswith(suffix):
            school = school[:-len(suffix)].strip()
            break
    return SCHOOL_EQUIVALENTS.get(school, school)


def _matchup_teams(row):
    """Return (home, away), orientation_known. Prefer CFBD Home/Away fields."""
    home = _norm_team(row.get("Home") or row.get("Home_Team") or row.get("Home_Team_Norm"))
    away = _norm_team(row.get("Away") or row.get("Away_Team") or row.get("Away_Team_Norm"))
    if home and away and home != away:
        return (home, away), True
    value = str(row.get("Matchup") or "").strip()
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
    """Read the existing BigQuery sportsbook feed, never an independent API."""
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
    # Lightweight optional as-of trigger context. These are NOT qualified OOS
    # confirmations and are never used as training inputs or probability shifts.
    optional=("BigAl_Side_Support_Count","BigAl_Side_Conflict_Count",
              "Pathi_System_Count","Pathi_Active_Text","NCAAF_Stat_Active",
              "System_Side_Consensus_Count","System_Side_Conflict_Count")
    try:
        present={f.name for f in bq_client.get_table(MARKET_TABLE).schema}
        extra=[c for c in optional if c in present]
    except Exception:
        extra=[]
    if extra:
        query=query.replace("SELECT Game_Start,", "SELECT "+",".join(extra)+",Game_Start,",1)
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("minutes", "INT64", minutes),
        bigquery.ScalarQueryParameter("horizon", "INT64", horizon_days),
    ])
    return bq_client.query(query, job_config=cfg).to_dataframe(create_bqstorage_client=False)


def _paired_spread(book_quotes, home, away):
    """Best fresh *paired* market per bookmaker and kickoff, not mixed books.

    Treat unmatched/mismatched 2-sided markets as not executable. Ensure the
    home and away spread signs are consistent and timestamps are near each other.
    """
    import pandas as pd
    choices=[]
    for (book,kick), grp in book_quotes.groupby(["_book", "_kick"], dropna=True):
        homes = grp.loc[grp["_selected"].eq(home)].sort_values("_snap",ascending=False)
        aways = grp.loc[grp["_selected"].eq(away)].sort_values("_snap",ascending=False)
        if homes.empty or aways.empty:
            continue
        # Select the freshest consistent pair, not a mixture of different
        # alternate lines or timestamps from a different round of quoting.
        for _, h in homes.head(4).iterrows():
            valid_away = aways[(aways["_line"].add(float(h["_line"]))).abs().le(0.01)]
            valid_away = valid_away[(valid_away["_snap"]-h["_snap"]).abs().le(pd.Timedelta(minutes=15))]
            if valid_away.empty:
                continue
            a=valid_away.iloc[0]
            newest=max(h["_snap"],a["_snap"])
            choices.append((newest,str(book),h,a))
            break
    if not choices:
        return None
    choices.sort(key=lambda p:(p[0],p[1]),reverse=True)
    return choices[0][2], choices[0][3]


def attach_verified_backend_quotes(forecasts, quotes, *, now=None, max_age_minutes=QUOTE_MAX_AGE_MINUTES):
    """Attach backend market evidence without inventing a V5.5 betting pick.

    Missing Selected_Team is normal for the independent CORE-only forecast. A
    two-sided sportsbook matchup can be identified and priced independently.
    This stage does not authorize bets or synthesize probability/EV estimates.
    """
    import pandas as pd
    if not isinstance(forecasts, pd.DataFrame):
        raise TypeError("forecasts must be a DataFrame")
    out = forecasts.copy()
    report = {
        "backend_price_source": MARKET_TABLE,
        "backend_quotes_read": len(quotes) if isinstance(quotes,pd.DataFrame) else 0,
        "core_forecast_rows":len(out), "upcoming_forecast_rows":0,
        "matched_quote_rows":0, "priced_games":0, "paired_two_sided_games":0,
        "unmatched_forecast_rows":0,"ambiguous_quote_rows":0,
        "unresolved_forecast_identity":0,"no_matching_game":0,
        "no_selected_side_quote":0,"no_two_sided_spread":0,
        "stale_or_invalid_quote_rows":0,"unmatched_examples":[],
        "identity_examples":[], "status":"NO_UPCOMING_FORECASTS",
    }
    if out.empty:
        return out,report
    # Any previous model price or unverified EV must not survive a backend
    # refresh. Keep model-only home-margin and standard deviation unchanged.
    for col in (
        "Current_Spread","Odds_American","CORE_Cover_Probability","CORE_Edge",
        "CORE_EV","CORE_Project_Cover_Points","Backend_Quote_Book",
        "Backend_Quote_Timestamp","Backend_Quote_Game_Key",
        "Backend_Home_Spread","Backend_Away_Spread","Backend_Home_Odds_American",
        "Backend_Away_Odds_American","Backend_Home_Book","Backend_Away_Book",
        "Backend_Home_Quote_UTC","Backend_Away_Quote_UTC",
        "Backend_Projected_Home_Cover_Points","Backend_Projected_Away_Cover_Points",
        "Backend_Model_Lean_Side","Backend_Model_Lean_Points",
        "V55_Trigger_Context","V55_Trigger_Context_Authority",
    ):
        out[col]=None
    out["Quote_State"]="NO_VERIFIED_BACKEND_QUOTE"
    out["V55_Bet_Decision"]="RESEARCH_ONLY_NO_AUTHORITY"
    out["V55_Bet_Authority"]=0
    now=_timestamp(now if now is not None else datetime.now(timezone.utc))
    if pd.isna(now):
        raise ValueError("invalid now")
    if "Game_Start_UTC" not in out.columns:
        report["status"]="V55_MISSING_KICKOFF_FIELD"
        return out,report
    kickoff=pd.to_datetime(out["Game_Start_UTC"],errors="coerce",utc=True)
    upcoming=kickoff.gt(now) & kickoff.le(now+pd.Timedelta(days=14))
    report["upcoming_forecast_rows"]=int(upcoming.sum())
    if not upcoming.any():
        return out,report
    required={"Game_Start","Home_Team_Norm","Away_Team_Norm","Outcome",
              "Value","Odds_Price","Snapshot_Timestamp"}
    if not isinstance(quotes,pd.DataFrame) or not required.issubset(quotes.columns):
        report["status"]="BACKEND_QUOTE_SCHEMA_MISSING"
        report["missing_backend_fields"]=sorted(required-set(getattr(quotes,"columns",[])))
        report["unmatched_forecast_rows"]=report["upcoming_forecast_rows"]
        return out,report
    q=quotes.copy()
    q["_kick"]=pd.to_datetime(q["Game_Start"],errors="coerce",utc=True)
    q["_snap"]=pd.to_datetime(q["Snapshot_Timestamp"],errors="coerce",utc=True)
    q["_home"]=q["Home_Team_Norm"].map(_norm_team)
    q["_away"]=q["Away_Team_Norm"].map(_norm_team)
    q["_selected"]=q["Outcome"].map(_norm_team)
    if "Outcome_Norm" in q.columns:
        missing=q["_selected"].eq("")
        q.loc[missing,"_selected"]=q.loc[missing,"Outcome_Norm"].map(_norm_team)
    q["_line"]=pd.to_numeric(q["Value"],errors="coerce")
    q["_american"]=pd.to_numeric(q["Odds_Price"],errors="coerce")
    b1=q["Bookmaker"] if "Bookmaker" in q.columns else pd.Series("",index=q.index)
    b2=q["Book"] if "Book" in q.columns else pd.Series("",index=q.index)
    q["_book"]=b1.fillna("").astype(str).str.lower().str.strip()
    q.loc[q["_book"].isin(["", "nan","none"]),"_book"]=b2.fillna("").astype(str).str.lower().str.strip()
    valid=(q["_kick"].gt(now)&q["_kick"].le(now+pd.Timedelta(days=14))&
           q["_snap"].le(now)&q["_snap"].lt(q["_kick"])&
           q["_snap"].ge(now-pd.Timedelta(minutes=max_age_minutes))&
           q["_home"].ne("")&q["_away"].ne("")&q["_home"].ne(q["_away"])&
           (q["_selected"].eq(q["_home"])|q["_selected"].eq(q["_away"]))&
           q["_line"].between(-100,100)&
           q["_american"].map(_valid_american)&q["_book"].ne(""))
    report["stale_or_invalid_quote_rows"]=int((~valid).sum())
    q=q.loc[valid].copy()
    if q.empty:
        report["status"]="NO_VALID_FRESH_BACKEND_QUOTES"
        report["unmatched_forecast_rows"]=report["upcoming_forecast_rows"]
        return out,report
    q=(q.sort_values("_snap",ascending=False)
       .drop_duplicates(["_home","_away","_kick","_book","_selected","_line"],keep="first"))
    q["_team_pair"]=q.apply(lambda r:tuple(sorted((r["_home"],r["_away"]))),axis=1)
    counted=set()
    def note(idx,reason):
        if len(report["unmatched_examples"])<10:
            report["unmatched_examples"].append({
                "matchup":str(out.at[idx,"Matchup"])[:100] if "Matchup" in out.columns else "",
                "selected":str(out.at[idx,"Selected_Team"])[:60] if "Selected_Team" in out.columns else "",
                "kickoff_utc":str(kickoff.loc[idx]),"reason":reason,
            })
    for idx in out.index[upcoming]:
        row=out.loc[idx]
        (home,away), oriented=_matchup_teams(row)
        selected=_norm_team(row.get("Selected_Team"))
        if not home or not away or home==away:
            report["unresolved_forecast_identity"]+=1
            out.at[idx,"Quote_State"]="UNRESOLVED_V55_GAME_IDENTITY"
            note(idx,"UNRESOLVED_GAME_IDENTITY")
            continue
        cand=q.loc[q["_team_pair"].map(lambda p:p==tuple(sorted((home,away))))]
        if cand.empty:
            report["no_matching_game"]+=1
            note(idx,"NO_MATCHING_PARTICIPANTS")
            continue
        cand=cand.loc[(cand["_kick"]-kickoff.loc[idx]).abs().le(pd.Timedelta(minutes=KICKOFF_TOLERANCE_MINUTES))]
        if cand.empty:
            report["no_matching_game"]+=1
            note(idx,"KICKOFF_OUTSIDE_TOLERANCE")
            continue
        if oriented:
            cand=cand.loc[cand["_home"].eq(home)&cand["_away"].eq(away)]
            if cand.empty:
                report["unresolved_forecast_identity"]+=1
                out.at[idx,"Quote_State"]="HOME_AWAY_ORIENTATION_CONFLICT"
                note(idx,"HOME_AWAY_ORIENTATION_CONFLICT")
                continue
        unique_games=cand[["_home","_away","_kick"]].drop_duplicates()
        if len(unique_games)!=1:
            report["ambiguous_quote_rows"]+=1
            out.at[idx,"Quote_State"]="AMBIGUOUS_BACKEND_GAME"
            note(idx,"AMBIGUOUS_GAME_OR_KICKOFF")
            continue
        actual_home=str(unique_games.iloc[0]["_home"])
        actual_away=str(unique_games.iloc[0]["_away"])
        pair=_paired_spread(cand,actual_home,actual_away)
        # Unselected CORE forecast: matching paired game prices is possible and
        # desirable, but a model betting side must NOT be guessed or invented.
        if not selected:
            if pair is None:
                report["no_two_sided_spread"]+=1
                out.at[idx,"Quote_State"]="MATCHED_GAME_NO_VERIFIED_TWO_SIDED_SPREAD"
                note(idx,"NO_SAME_BOOK_TWO_SIDED_SPREAD")
                continue
            h,a=pair
            for side,label in ((h,"Home"),(a,"Away")):
                out.at[idx,"Backend_"+label+"_Spread"]=float(side["_line"])
                out.at[idx,"Backend_"+label+"_Odds_American"]=float(side["_american"])
                out.at[idx,"Backend_"+label+"_Book"]=side["_book"]
                out.at[idx,"Backend_"+label+"_Quote_UTC"]=side["_snap"].isoformat()
            out.at[idx,"Backend_Quote_Book"]=h["_book"]
            out.at[idx,"Backend_Quote_Timestamp"]=max(h["_snap"],a["_snap"]).isoformat()
            out.at[idx,"Backend_Quote_Game_Key"]=str(h.get("Game_Key") or "")
            trigger_context={}
            for t in ("BigAl_Side_Support_Count","BigAl_Side_Conflict_Count",
                      "Pathi_System_Count","Pathi_Active_Text","NCAAF_Stat_Active",
                      "System_Side_Consensus_Count","System_Side_Conflict_Count"):
                val=h.get(t)
                if val is not None and str(val).lower() not in ("nan","none","null",""):
                    trigger_context[t]=_json_default(val)
            if trigger_context:
                out.at[idx,"V55_Trigger_Context"]=json.dumps(trigger_context,sort_keys=True)
            out.at[idx,"V55_Trigger_Context_Authority"]="UNQUALIFIED_PRE_GAME_CONTEXT_ONLY"
            out.at[idx,"Quote_State"]="FRESH_TWO_SIDED_BACKEND_PRICE_NO_SELECTED_TEAM"
            # Quotes are available; V5.5 has not selected a betting side or
            # calculated approved probabilities/EV. Never label this NO_QUOTE.
            if "V55_Research_Priority" in out.columns:
                out.at[idx,"V55_Research_Priority"]="VERIFIED_PAIRED_QUOTE__NO_BET_AUTHORITY"
            report["paired_two_sided_games"]+=1
            margin=_price(row.get("CORE_Projected_Home_Margin"))
            if oriented and margin is not None:
                he=margin+float(h["_line"])
                ae=-margin+float(a["_line"])
                out.at[idx,"Backend_Projected_Home_Cover_Points"]=he
                out.at[idx,"Backend_Projected_Away_Cover_Points"]=ae
                out.at[idx,"Backend_Model_Lean_Side"]=home if he>=ae else away
                out.at[idx,"Backend_Model_Lean_Points"]=abs(he)
        else:
            if selected not in (home,away):
                report["unresolved_forecast_identity"]+=1
                out.at[idx,"Quote_State"]="SELECTED_SIDE_NOT_PARTICIPANT"
                note(idx,"SELECTED_SIDE_NOT_PARTICIPANT")
                continue
            s=cand.loc[cand["_selected"].eq(selected)].sort_values("_snap",ascending=False)
            if s.empty:
                report["no_selected_side_quote"]+=1
                note(idx,"NO_QUOTE_FOR_SELECTED_SIDE")
                continue
            chosen=s.iloc[0]
            out.at[idx,"Current_Spread"]=float(chosen["_line"])
            out.at[idx,"Odds_American"]=float(chosen["_american"])
            out.at[idx,"Backend_Quote_Book"]=chosen["_book"]
            out.at[idx,"Backend_Quote_Timestamp"]=chosen["_snap"].isoformat()
            out.at[idx,"Backend_Quote_Game_Key"]=str(chosen.get("Game_Key") or "")
            out.at[idx,"Quote_State"]="FRESH_VERIFIED_SELECTED_SIDE_BACKEND_SPREAD"
            if "V55_Research_Priority" in out.columns:
                out.at[idx,"V55_Research_Priority"]="VERIFIED_SELECTED_SIDE_QUOTE__NO_BET_AUTHORITY"
            margin=_price(row.get("CORE_Projected_Home_Margin"))
            if oriented and margin is not None:
                points=(margin if selected==home else -margin)+float(chosen["_line"])
                out.at[idx,"CORE_Project_Cover_Points"]=points
            if pair is not None:
                report["paired_two_sided_games"]+=1
        report["matched_quote_rows"]+=1
        counted.add((tuple(sorted((actual_home,actual_away))),str(unique_games.iloc[0]["_kick"])))
        if len(report["identity_examples"])<6:
            report["identity_examples"].append({"forecast":str(row.get("Matchup")),
                "market_home":str(cand.iloc[0]["Home_Team_Norm"]),
                "market_away":str(cand.iloc[0]["Away_Team_Norm"]),
                "selected_side_present":bool(selected)})
    report["priced_games"]=len(counted)
    report["unmatched_forecast_rows"]=report["upcoming_forecast_rows"]-report["matched_quote_rows"]
    report["status"]="PASS" if counted else "NO_MATCHING_UPCOMING_MODEL_AND_BACKEND_PRICES"
    return out,report


def bridge_weekly_prices(weekly, audit, *, bq_client, log_func=None):
    """Enforce one backend price authority for both weekly and market refresh."""
    import json
    import pandas as pd
    log = log_func or logging.info
    # Never rewrite a rejected model's provenance into a seeming clean audit.
    if audit.get("production_v1_probabilities_used") is True or audit.get("v55_bet_authority",0)!=0:
        raise RuntimeError("V55_UNTRUSTED_SOURCE_PROVENANCE_OR_BET_AUTHORITY")
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
    # Preserve the model provenance contract. The join stage has its OWN status;
    # replacing audit['status'] with PASS would block publication entirely.
    updated.update({k: v for k, v in diag.items() if k != "status"})
    updated["market_price_join_status"] = diag["status"]
    # Legacy model audit no_market_match was computed before the verified
    # backend bridge. Publish a consistent, horizon-scoped market coverage count.
    updated["no_market_match"] = int(diag["unmatched_forecast_rows"])
    updated["outside_pricing_window_forecast_rows"] = max(
        0,int(diag["core_forecast_rows"])-int(diag["upcoming_forecast_rows"]))
    updated["status"] = audit.get("status")
    updated["v55_bet_authority"] = 0
    updated["price_authority"] = "SHARP_SCANNER_BIGQUERY_ONLY"
    updated["price_join_semantics"] = "FRESH_QUOTE_MATCH_PLUS_INDEPENDENT_RESEARCH_ONLY"
    # Model is unchanged. Calculate both spread sides entirely on the backend
    # and attach qualified evidence only when independently sourced and vetted.
    calibration, cal_problem = read_v55_oos_calibration()
    out, research = evaluate_v55_spread_research(out,calibration=calibration)
    if cal_problem:
        research["calibration_unavailable_reason"] = cal_problem
    evidence, evidence_problem = read_v55_qualified_evidence(bq_client)
    out, brain = attach_v55_qualified_evidence(out,evidence)
    if evidence_problem:
        brain["evidence_unavailable_reason"] = evidence_problem
    updated.update({k:v for k,v in research.items() if k!="status"})
    updated["research_decision_status"]=research["status"]
    updated.update(brain)
    # The original model source audit is immutable: price/research statuses
    # must never replace it with a generic PASS marker.
    updated["status"]=audit["status"]
    updated["v55_bet_authority"] = 0
    updated["price_join_semantics"] = "FRESH_QUOTES_INDEPENDENT_RESEARCH_ONLY"
    log("[NCAAF-V55-BACKEND-PRICE-BRIDGE] " + json.dumps(diag,sort_keys=True))
    log("[NCAAF-V55-RESEARCH-SIDE-EV] " + json.dumps(research,sort_keys=True))
    log("[NCAAF-V55-INDEPENDENT-BRAIN-EVIDENCE] " + json.dumps(brain,sort_keys=True))
    return out, updated


# ----- V5.5 independent spread research; NEVER betting authority -----

VALID_BRAIN_FAMILIES = ("STAT", "BIG_AL", "PATHI", "SYSTEM_MINER", "PREDICTION_TRACKER")


def _normal_cdf(z):
    """Normal approximation from the V5.5 *own* residual SD, not V1."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _american_net(decimal_american):
    v=_price(decimal_american)
    if v is None or not _valid_american(v):
        return None
    return v/100.0 if v>=100 else 100.0/abs(v)


def _validate_oos_calibration(bundle):
    """Fail closed: a residual SD fitted on 2013-25 is NOT OOS calibration.

    Only a separately certified chronological, held-out V5.5 validation bundle
    may be used in the calibrated fields. No fitting occurs during repricing.
    """
    if not isinstance(bundle,dict):
        return None
    if bundle.get("status")!="PASS_OOS_LOCKED" or bundle.get("core_model")!="NONLINEAR_HGB":
        return None
    if bundle.get("validation_method")!="CHRONOLOGICAL_OUT_OF_FOLD":
        return None
    if bundle.get("model_source")!="CFBD_ONLY_V55":
        return None
    if bundle.get("production_v1_probabilities_used") is not False:
        return None
    n=_price(bundle.get("validated_games"))
    if n is None or n<500 or n!=int(n):
        return None
    if not re.fullmatch(r"[0-9a-f]{64}",str(bundle.get("oof_predictions_sha256") or "")):
        return None
    if not re.fullmatch(r"[0-9a-f]{64}",str(bundle.get("artifact_sha256") or "")):
        return None
    brier=_price(bundle.get("brier_score"))
    ece=_price(bundle.get("ece"))
    if brier is None or not 0<brier<0.25 or ece is None or not 0<=ece<=0.05:
        return None
    scale=_price(bundle.get("residual_scale"))
    bias=_price(bundle.get("margin_bias"))
    if scale is None or not 5<=scale<=35 or bias is None or abs(bias)>10:
        return None
    train_cutoff=_timestamp(bundle.get("calibration_qualified_at_utc"))
    if str(train_cutoff) == 'NaT' or train_cutoff.year>2025:
        return None
    return {"residual_scale":scale,"margin_bias":bias,
            "calibration_id":str(bundle.get("calibration_id") or "")[:128]}


def read_v55_oos_calibration():
    """Optional immutable GCS calibration *artifact*; never learn from live bets.

    The canonical 2013-25 research artifact is not present in this supplied
    source tree. No artifact is silently generated or treated as validated.
    """
    uri=os.getenv("NCAAF_V55_OOS_CALIBRATION_GCS_URI","").strip()
    if not uri:
        return None,"NO_INDEPENDENT_OOS_CALIBRATION_ARTIFACT_CONFIGURED"
    if not uri.startswith("gs://") or "/" not in uri[5:]:
        return None,"INVALID_CALIBRATION_GCS_URI"
    try:
        from google.cloud import storage
        bucket,key=uri[5:].split("/",1)
        raw=storage.Client().bucket(bucket).blob(key).download_as_bytes()
        if len(raw)>200_000:
            return None,"OVERSIZED_CALIBRATION_ARTIFACT"
        b=json.loads(raw)
        candidate=_validate_oos_calibration(b)
        return (candidate,None) if candidate else (None,"OOS_CALIBRATION_CONTRACT_REJECTED")
    except Exception as exc:
        return None,"CALIBRATION_ACCESS_ERROR:"+type(exc).__name__


def evaluate_v55_spread_research(frame, *, calibration=None):
    """Backend-only paired spread comparison, shadow-only with explicit provenance.

    V55_Raw_* is a NORMAL RESIDUAL APPROXIMATION, NOT a calibrated probability.
    CORE_Cover_Probability/CORE_EV are published ONLY with vetted OOS calibration.
    No decision or model field ever changes bet authority or production V1.
    """
    import pandas as pd
    out=frame.copy()
    fields=("V55_Research_Lean_Team","V55_Research_Selected_Team",
            "V55_Research_Selected_Side","V55_Research_Decision",
            "V55_Raw_Cover_Prob","V55_Raw_EV","V55_Raw_Edge",
            "V55_Home_Raw_Prob","V55_Away_Raw_Prob",
            "V55_Home_Raw_EV","V55_Away_Raw_EV",
            "V55_Probability_State","V55_Calibration_Id")
    for key in fields:
        out[key]=None
    # Never carry earlier model's unverified probability or selected wager into
    # this backend-only market refresh. Every row is recomputed or withheld.
    for key in ("CORE_Cover_Probability","CORE_Edge","CORE_EV","Selected_Team",
                "Current_Spread","Odds_American","CORE_Project_Cover_Points"):
        out[key]=None
    audited_calibration=_validate_oos_calibration(calibration)
    stats={"status":"PASS_RESEARCH_SHADOW","paired_quotes":0,
           "raw_side_estimates":0,"positive_raw_ev_leans":0,
           "selected_research_sides":0,"oos_calibrated":0,
           "calibration_state":"LOCKED_OOS" if audited_calibration else "UNAVAILABLE_UNCALIBRATED",
           "core_refit":False,"v55_bet_authority":0,"production_v1_probabilities_used":False}
    for i,r in out.iterrows():
        if not str(r.get("Quote_State") or "").startswith("FRESH_"):
            out.at[i,"V55_Research_Decision"]="NO_VERIFIED_QUOTE"
            out.at[i,"V55_Probability_State"]="NOT_EVALUABLE_NO_QUOTE"
            continue
        home_line=_price(r.get("Backend_Home_Spread"))
        away_line=_price(r.get("Backend_Away_Spread"))
        home_odds=_american_net(r.get("Backend_Home_Odds_American"))
        away_odds=_american_net(r.get("Backend_Away_Odds_American"))
        if any(v is None for v in (home_line,away_line,home_odds,away_odds)):
            out.at[i,"V55_Research_Decision"]="NO_VERIFIED_TWO_SIDED_QUOTE"
            continue
        if abs(home_line+away_line)>0.01:
            out.at[i,"V55_Research_Decision"]="INCONSISTENT_SIDES"
            continue
        stats["paired_quotes"]+=1
        mu=_price(r.get("CORE_Projected_Home_Margin"))
        sigma=_price(r.get("CORE_Residual_SD"))
        hname=str(r.get("Home") or "").strip()
        aname=str(r.get("Away") or "").strip()
        if not hname or not aname:
            matchup=str(r.get("Matchup") or "")
            m=re.match(r"^(.+?)\s+@\s+(.+)$",matchup)
            if m: aname,hname=m.group(1),m.group(2)
        if mu is None or sigma is None or not 4<=sigma<=50 or not hname or not aname:
            out.at[i,"V55_Research_Decision"]="CORE_DISTRIBUTION_OR_IDENTITY_NOT_VERIFIED"
            out.at[i,"V55_Probability_State"]="NOT_EVALUABLE_MISSING_SD"
            continue
        if str(r.get("CORE_Model"))!="NONLINEAR_HGB":
            out.at[i,"V55_Research_Decision"]="UNVERIFIED_CORE_MODEL"
            continue
        # Continuous normal assumption: push probability is not estimated.
        hp=_normal_cdf((mu+home_line)/sigma)
        ap=1-hp
        hev=hp*home_odds-(1-hp)
        aev=ap*away_odds-(1-ap)
        out.at[i,"V55_Home_Raw_Prob"]=hp
        out.at[i,"V55_Away_Raw_Prob"]=ap
        out.at[i,"V55_Home_Raw_EV"]=hev
        out.at[i,"V55_Away_Raw_EV"]=aev
        out.at[i,"V55_Probability_State"]="UNVALIDATED_NORMAL_RESIDUAL_APPROXIMATION"
        choose_home=(hev>=aev)
        side=hname if choose_home else aname
        p=hp if choose_home else ap
        ev=hev if choose_home else aev
        spread=home_line if choose_home else away_line
        odds=_price(r.get("Backend_Home_Odds_American" if choose_home else "Backend_Away_Odds_American"))
        out.at[i,"V55_Research_Lean_Team"]=side
        out.at[i,"V55_Raw_Cover_Prob"]=p
        out.at[i,"V55_Raw_EV"]=ev
        out.at[i,"V55_Raw_Edge"]=p-(1/(1+(home_odds if choose_home else away_odds)))
        stats["raw_side_estimates"]+=1
        if ev>0:
            stats["positive_raw_ev_leans"]+=1
        # These side/price fields are research decisions, not actionable bets.
        if ev<=0:
            out.at[i,"V55_Research_Decision"]="MODEL_ONLY__NO_POSITIVE_RAW_EV"
            out.at[i,"V55_Research_Priority"]="MATCHED_QUOTE__MODEL_ONLY"
            continue
        out.at[i,"Selected_Team"]=side
        out.at[i,"V55_Research_Selected_Team"]=side
        out.at[i,"Quote_State"]="FRESH_TWO_SIDED_BACKEND_PRICE_RESEARCH_SIDE"
        out.at[i,"V55_Research_Selected_Side"]="HOME" if choose_home else "AWAY"
        out.at[i,"Current_Spread"]=spread
        out.at[i,"Odds_American"]=odds
        out.at[i,"Book"]=r.get("Backend_Quote_Book")
        out.at[i,"CORE_Project_Cover_Points"]=(mu+home_line if choose_home else -mu+away_line)
        out.at[i,"V55_Research_Decision"]="UNVALIDATED_NORMAL_MODEL_LEAN_ONLY"
        out.at[i,"V55_Research_Priority"]="RAW_MODEL_RESEARCH_LEAN__NO_BET_AUTHORITY"
        stats["selected_research_sides"]+=1
        # Independent OOS validation is a required separate input. The weekly
        # run's retrospective residual SD is NOT sufficient to populate CORE_*.
        if audited_calibration:
            z=((mu-audited_calibration["margin_bias"])+home_line)/audited_calibration["residual_scale"]
            cal_p=_normal_cdf(z)
            if not choose_home: cal_p=1-cal_p
            net=home_odds if choose_home else away_odds
            out.at[i,"CORE_Cover_Probability"]=cal_p
            out.at[i,"CORE_Edge"]=cal_p-1/(1+net)
            out.at[i,"CORE_EV"]=cal_p*net-(1-cal_p)
            out.at[i,"V55_Probability_State"]="INDEPENDENT_CHRONOLOGICAL_OOS_RESIDUAL_CALIBRATION"
            out.at[i,"V55_Calibration_Id"]=audited_calibration["calibration_id"]
            out.at[i,"V55_Research_Decision"]="CALIBRATED_RESEARCH_ONLY_NO_BET_AUTHORITY"
            stats["oos_calibrated"]+=1
    out["V55_Bet_Authority"]=0
    out["V55_Bet_Decision"]="RESEARCH_ONLY_NO_AUTHORITY"
    return out,stats


def read_v55_qualified_evidence(bq_client):
    """Optional V5.5-specific, pregame/OOS-vetted events. No V1 vote reuse.

    Explicitly opt in using a preexisting validated evidence table; never
    auto-create or infer schema for unvetted 2026 triggers.
    """
    table=os.getenv("NCAAF_V55_QUALIFIED_EVIDENCE_TABLE", "").strip()
    if not table:
        return None,"NO_V55_QUALIFIED_EVIDENCE_TABLE_CONFIGURED"
    if not re.fullmatch(r"[a-zA-Z0-9_-]+\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+",table):
        return None,"BAD_EVIDENCE_TABLE_ID"
    try:
        df=bq_client.query(f"SELECT * FROM `{table}` WHERE Game_Start_UTC > CURRENT_TIMESTAMP() AND Game_Start_UTC < TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)").to_dataframe(create_bqstorage_client=False)
        return df,None
    except Exception as exc:
        return None,"EVIDENCE_READ_ERROR:"+type(exc).__name__


def attach_v55_qualified_evidence(frame,events):
    """Only independently qualified sources may confirm or oppose V5.5 CORE.

    Exact CFBD game ID, kickoff, directional team, historical qualification
    prior to 2026, independently timestamped observation <= quote time,
    verified pregame provenance, and unique source family are mandatory.
    """
    import pandas as pd
    out=frame.copy()
    # Quarantine every unvetted legacy/live source. Votes in saved weekly GCS
    # payload have no authority unless they pass the exact V5.5 event contract.
    for key in ("Confirm_Count","Oppose_Count"):
        out[key]=0
    for key in ("Confirm_Families","Oppose_Families"):
        out[key]=None
    for family in VALID_BRAIN_FAMILIES:
        out[family]="NOT_EVALUABLE"
    report={"evidence_state":"NOT_EVALUABLE_NO_VERIFIED_EVENTS",
            "rows_read":len(events) if isinstance(events,pd.DataFrame) else 0,
            "qualified_events_used":0,"multi_family":0,"conflict_games":0,
            "source_family_cap":1,"v55_bet_authority":0}
    required={"Game_ID","Game_Start_UTC","Source_Family","Signal_Family","Signal_Side",
              "Observed_At_UTC","Qualified_At_UTC","Qualification_Status",
              "OOS_Status","Direction_Verified","Provenance_Status"}
    if not isinstance(events,pd.DataFrame) or not required.issubset(events.columns):
        return out,report
    ev=events.copy()
    ev["_obs"]=pd.to_datetime(ev["Observed_At_UTC"],utc=True,errors="coerce")
    ev["_qualified"]=pd.to_datetime(ev["Qualified_At_UTC"],utc=True,errors="coerce")
    ev["_kick"]=pd.to_datetime(ev["Game_Start_UTC"],utc=True,errors="coerce")
    ev["_family"]=ev["Source_Family"].astype(str).str.upper().str.strip()
    ev["_game_id"]=ev["Game_ID"].astype(str)
    def check(v): return v is True or str(v).lower() in ("true","1")
    valid=(ev["_family"].isin(VALID_BRAIN_FAMILIES)
           &ev["Qualification_Status"].eq("QUALIFIED_OOS")
           &ev["OOS_Status"].eq("PASS")
           &ev["Provenance_Status"].eq("PRE_GAME_VERIFIED")
           &ev["Direction_Verified"].map(check)
           &ev["_qualified"].lt(pd.Timestamp("2026-01-01T00:00:00Z"))
           &ev["_obs"].lt(ev["_kick"])
           &ev["_obs"].ge(ev["_qualified"]))
    ev=ev.loc[valid]
    if ev.empty:
        report["evidence_state"]="NO_VALID_V55_OOS_EVENTS"
        return out,report
    report["evidence_state"]="EVALUATED_V55_OOS_EVENTS"
    for i,r in out.iterrows():
        selected=_norm_team(r.get("Selected_Team"))
        gid=str(r.get("Game_ID") or "")
        quote_time=_timestamp(r.get("Backend_Quote_Timestamp"))
        kick=_timestamp(r.get("Game_Start_UTC"))
        (home,away),_= _matchup_teams(r)
        if not selected or not gid or pd.isna(quote_time) or pd.isna(kick):
            continue
        relevant=ev.loc[(ev["_game_id"]==gid)&ev["_kick"].eq(kick)&ev["_obs"].le(quote_time)]
        if relevant.empty:
            continue
        # Only one vote per independent source. If independent signals from
        # the same source oppose each other, that source ABSTAINS.
        support=set(); opposition=set()
        for family,g in relevant.groupby("_family"):
            dirs=set(_norm_team(v) for v in g["Signal_Side"].tolist())
            dirs.discard("")
            if len(dirs)!=1:
                continue
            direction=next(iter(dirs))
            if direction not in (home,away):
                continue
            if direction==selected: support.add(family)
            else: opposition.add(family)
        for family in VALID_BRAIN_FAMILIES:
            if family in support:
                out.at[i,family]="QUALIFIED_SUPPORT"
            elif family in opposition:
                out.at[i,family]="QUALIFIED_OPPOSE"
        report["qualified_events_used"]+=len(support)+len(opposition)
        out.at[i,"Confirm_Count"]=len(support)
        out.at[i,"Oppose_Count"]=len(opposition)
        out.at[i,"Confirm_Families"]="|".join(sorted(support)) if support else None
        out.at[i,"Oppose_Families"]="|".join(sorted(opposition)) if opposition else None
        if opposition:
            report["conflict_games"]+=1
            out.at[i,"V55_Research_Decision"]="RESEARCH_CONFLICT__VETO"
        elif len(support)>=2:
            report["multi_family"]+=1
            out.at[i,"V55_Research_Decision"]="MULTI_FAMILY_RESEARCH_SUPPORT__NO_BET_AUTHORITY"
    out["V55_Bet_Authority"]=0
    out["V55_Bet_Decision"]="RESEARCH_ONLY_NO_AUTHORITY"
    return out,report
