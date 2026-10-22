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
# Explicit project-owner decision: V5.5 replaces V1 as the *betting selection*
# authority. This is intentionally not represented as independently calibrated.
V55_POLICY_ID = "V55-CORE-PLUS-INDEPENDENT-CONFIRMATION-20261022-R2"
V55_POLICY_STATUS = "QUALIFIED_CONFIRMATION_REQUIRED_PROVISIONAL"
V55_POLICY_MAX_BETS = 3
V55_POLICY_MIN_EV = 0.18
V55_POLICY_MIN_MARGIN_POINTS = 5.0
V55_POLICY_MIN_CONDITIONAL_PROB = 0.60
V55_POLICY_MAX_QUOTE_AGE_MINUTES = 90



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
    authority = audit.get("v55_bet_authority", 0)
    if authority not in (0, 1):
        raise RuntimeError("V55_AUTHORITY_INVALID_VALUE")
    if authority == 1:
        if (audit.get("v55_authority_policy_id") != V55_POLICY_ID
                or audit.get("v55_authority_policy_status") != V55_POLICY_STATUS
                or audit.get("v55_authority_probability_basis") != "UNVALIDATED_CFBD_V55_NORMAL_MARGIN"
                or int(audit.get("v55_authorized_bet_rows") or 0) < 1
                or int(audit.get("v55_authorized_bet_rows") or 0) > V55_POLICY_MAX_BETS):
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
    # The publication itself is the authority journal. Check agreement between
    # row-level actionable decisions and the audit before atomic publication.
    if "V55_Bet_Authority" in weekly.columns:
        bet_rows = weekly.loc[weekly["V55_Bet_Authority"].eq(1)]
        if len(bet_rows) != int(audit.get("v55_authorized_bet_rows", 0) or 0):
            raise RuntimeError("V55_AUTHORIZED_ROW_COUNT_DISAGREEMENT")
        if bool(len(bet_rows)) != bool(audit.get("v55_bet_authority",0)):
            raise RuntimeError("V55_AUTHORITY_AUDIT_FLAG_DISAGREEMENT")
        if len(bet_rows) > V55_POLICY_MAX_BETS:
            raise RuntimeError("V55_AUTHORIZED_ROW_CAP_EXCEEDED")
        if len(bet_rows) and bet_rows["V55_Bet_Key"].nunique()!=len(bet_rows):
            raise RuntimeError("V55_DUPLICATED_AUTHORIZED_GAME_SIDE")
        for _, bet in bet_rows.iterrows():
            if (str(bet.get("V55_Bet_Decision")) not in ("BET", "STRONG BET")
                    or bet.get("V55_Authority_Policy_Id") != V55_POLICY_ID
                    or bet.get("V55_Authority_State") != V55_POLICY_STATUS
                    or not str(bet.get("Quote_State") or "").startswith("FRESH_")
                    or _price(bet.get("V55_Policy_EV")) is None
                    or _price(bet.get("V55_Policy_EV")) < V55_POLICY_MIN_EV
                    or _price(bet.get("Current_Spread")) is None
                    or not _valid_american(bet.get("Odds_American"))
                    or not str(bet.get("Selected_Team") or "").strip()):
                raise RuntimeError("V55_AUTHORIZED_BET_ROW_INVALID")
            confirmed=int(_price(bet.get('Confirm_Count')) or 0)
            opposed=int(_price(bet.get('Oppose_Count')) or 0)
            if (confirmed < (2 if bet.get('V55_Bet_Decision')=='STRONG BET' else 1)
                    or opposed>0
                    or not str(bet.get('V55_Qualified_Evidence_Detail_JSON') or '').startswith('[')
                    or not str(bet.get('V55_Independent_Support_Keys') or '').strip()):
                raise RuntimeError('V55_BET_WITHOUT_AUDITABLE_INDEPENDENT_CONFIRMATION')
    elif int(audit.get("v55_bet_authority") or 0):
        raise RuntimeError("V55_MISSING_AUTHORIZED_ROW_SCHEMA")
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
            "published_at":now,"v55_bet_authority":int(audit.get("v55_bet_authority",0) or 0),
            "authorized_bets":int(audit.get("v55_authorized_bet_rows",0) or 0)}
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
    weekly, audit = bridge_weekly_prices(weekly,audit,bq_client=client,storage_client=gcs,bucket_name=bucket_name or os.getenv('MODEL_BUCKET','sharp-models'))
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
            "V55_Authority_State","V55_Bet_Key","V55_Policy_EV",
            "V55_Policy_Win_Prob","V55_Policy_Push_Prob",
            "V55_Policy_Lose_Prob","V55_Policy_Edge","V55_Policy_Stake_Units",
            "V55_Authority_Policy_Id",
             "V55_Research_Selected_Side","V55_Home_Raw_Prob",
             "V55_Away_Raw_Prob","V55_Home_Raw_EV","V55_Away_Raw_EV",
             "V55_Trigger_Context","V55_Trigger_Context_Authority",
             "V55_Qualified_Evidence_Detail_JSON",
             "V55_Active_System_Names","V55_Independent_Support_Keys",
             "V55_Independent_Conflict_Keys",
             "Backend_Model_Lean_Side","Backend_Model_Lean_Points",
             "Backend_Projected_Home_Cover_Points","Backend_Projected_Away_Cover_Points",
             "CORE_EV","CORE_Edge","CORE_Cover_Probability","Book"}
        for col in price_fields.intersection(frame.columns):
            frame[col]=None
        frame["Quote_State"]="STALE_BACKEND_SNAPSHOT"
        # Never label a prior publication's time-dependent raw overlays as live.
        for raw_col in V55_RAW_OVERLAY_COLUMNS:
            if raw_col in frame.columns:
                frame[raw_col]=None
        frame["V55_Raw_Overlay_State"]="STALE_BACKEND_SNAPSHOT"
        frame["V55_Probability_State"]="STALE_BACKEND_SNAPSHOT"
        frame["V55_Research_Decision"]="NO_CURRENT_QUOTE__STALE"
        frame["V55_Bet_Decision"]="NO CURRENT QUOTE — STALE"
        frame["V55_Bet_Authority"]=0
        # Expired publications cannot retain seemingly active qualified votes.
        for col in ("Confirm_Count","Oppose_Count"):
            if col in frame.columns: frame[col]=0
        for col in ("Confirm_Families","Oppose_Families"):
            if col in frame.columns: frame[col]=None
        for col in VALID_BRAIN_FAMILIES:
            if col in frame.columns: frame[col]="STALE_BACKEND_SNAPSHOT"
    # The board may still be fresh (<180m) after an individual bet quote has
    # expired (our tighter 90m policy). Redact authority on READ in the backend;
    # never make the dashboard itself infer bet permission from time fields.
    if not stale and "V55_Bet_Authority" in frame.columns:
        quoted=pd.to_datetime(frame.get("Backend_Quote_Timestamp",
                    pd.Series(index=frame.index,dtype="object")),utc=True,errors="coerce")
        kicks=pd.to_datetime(frame.get("Game_Start_UTC",
                    pd.Series(index=frame.index,dtype="object")),utc=True,errors="coerce")
        current=pd.Timestamp(_utcnow())
        expired=frame["V55_Bet_Authority"].eq(1) & (
            quoted.isna() | kicks.isna() | (quoted>current) |
            (current-quoted>pd.Timedelta(minutes=V55_POLICY_MAX_QUOTE_AGE_MINUTES)) |
            (kicks<=current))
        if expired.any():
            frame.loc[expired,"V55_Bet_Authority"]=0
            frame.loc[expired,"V55_Bet_Decision"]="BET_EXPIRED_RECHECK_QUOTE"
            frame.loc[expired,"V55_Authority_State"]="EXPIRED_MARKET_AUTHORITY"
            frame.loc[expired,"V55_Policy_Stake_Units"]=None
            for col in ('V55_Qualified_Evidence_Detail_JSON','V55_Active_System_Names',
                        'V55_Independent_Support_Keys','V55_Independent_Conflict_Keys',
                        'Confirm_Families','Oppose_Families'):
                if col in frame.columns:frame.loc[expired,col]=None
            for col in VALID_BRAIN_FAMILIES:
                if col in frame.columns:frame.loc[expired,col]='QUOTE_EXPIRED_REEVALUATE'
    active_count=int(frame["V55_Bet_Authority"].eq(1).sum()) if "V55_Bet_Authority" in frame else 0
    strong_count=int((frame.get("V55_Bet_Decision",pd.Series(dtype="object"))=="STRONG BET").sum()) if active_count else 0
    meta={**audit,"publication_status":"STALE_MARKET" if stale else "PASS",
          "publication_type":str(commit.publication_type),
          "published_at_utc":published.isoformat(),"published_age_minutes":round(age,1),
          "published_rows":len(frame),"snapshot_id":str(commit.snapshot_id),
          "v55_bet_authority":1 if active_count else 0,
          "v55_authorized_bet_rows":active_count,
          "v55_strong_bet_rows":strong_count}
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
    "ohio university": "ohio",
    "ohio u": "ohio",
    "texas a m": "texas a and m",
    "texas a and m": "texas a and m",
    "texas a and m university": "texas a and m",
    "texas a m university": "texas a and m",
    "cal": "california",
    "cal berkeley": "california",
    "pitt": "pittsburgh",
    "pittsburgh university": "pittsburgh",
    "western michigan university": "western michigan",
    "ucla": "ucla",
    "university of connecticut": "connecticut",
    "army west point black knights": "army",
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
    cardinal, hawkeyes, ducks, utes, cyclones, sooners, cornhuskers,
    hokies, bearcats, gophers, jackrabbits, bison, spiders, blazers,
    huskers, chippewas, broncs, bobcats, owls, red hawks, blue raiders,
    lumberjacks, terriers, penguins, penguins, purple aces, leathernecks,
    skyhawks, sycamores, salukis, mocs, governors, colonels, racers,
    paladins, buccaneers, seahawks, grizzlies, grizz, red foxes,
    nittany lions, minutewomen, privateers, ragin cajuns, rainbow wahine,
    bulls, blue hose, horned frogs, crusaders, bantams, pioneers,
    vandals, grizzlies, demon deacons, blazers, lancers, terrapins,
    greyhounds, matadors, seawolves, red storm, big green, samurai,
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


def _cfbd_midnight_placeholder(kickoff):
    """Detect the CFBD unknown-time midnight ET sentinel, not arbitrary mismatches."""
    import pandas as pd
    k=pd.to_datetime(kickoff, utc=True, errors='coerce')
    return bool(pd.notna(k) and k.tz_convert('America/New_York').hour == 0
                and k.tz_convert('America/New_York').minute == 0)


def _verified_market_placeholder_repair(candidates, forecast_kick, home, away):
    """Use sportsbook kickoff only for an unequivocal, future, two-sided matchup.

    The CFBD midnight placeholder must never be used to price an unrelated
    same-date game. Require the exact oriented teams and a unique market kickoff.
    """
    import pandas as pd
    if not _cfbd_midnight_placeholder(forecast_kick) or candidates.empty:
        return None
    near=candidates.loc[(candidates['_kick']-forecast_kick).abs().le(pd.Timedelta(hours=26))
                       &candidates['_home'].eq(home)&candidates['_away'].eq(away)]
    market_kicks=near['_kick'].dropna().drop_duplicates()
    if len(market_kicks) != 1:
        return None
    market_kick=market_kicks.iloc[0]
    same_event=near.loc[near['_kick'].eq(market_kick)]
    if _paired_spread(same_event,home,away) is None:
        return None
    return market_kick


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
        "unique_available_market_games":0,"no_market_participant_pairs":0,
        "partial_team_identity_matches":0,"no_valid_kickoff_match":0,
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
    out['V55_Original_CFBD_Kickoff_UTC']=None
    out['V55_Kickoff_Verification']='UNCHANGED_CFBD_KICKOFF'
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
    # A default CFBD midnight kickoff is not necessarily a real kickoff.
    # Keep its forecast eligible briefly after midnight; ONLY a future paired
    # sportsbook market can replace the placeholder with the actual event time.
    placeholders=kickoff.map(_cfbd_midnight_placeholder)
    upcoming=((kickoff.gt(now) & kickoff.le(now+pd.Timedelta(days=14))) |
              (placeholders & kickoff.ge(now-pd.Timedelta(hours=26))
               &kickoff.le(now+pd.Timedelta(days=14))))
    report['cfbd_midnight_placeholder_games']=int((upcoming & placeholders).sum())
    report['verified_market_kickoff_repairs']=0
    report['alias_normalization_version']='V55_EXPLICIT_SCHOOL_MASCOT_20261022'
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
    report['unique_available_market_games']=len(q[['_home','_away','_kick']].drop_duplicates())
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
            report['no_market_participant_pairs']+=1
            near_home=(q['_home'].eq(home) | q['_away'].eq(home)).any()
            near_away=(q['_home'].eq(away) | q['_away'].eq(away)).any()
            if near_home or near_away:
                report['partial_team_identity_matches']+=1
                out.at[idx,'Quote_State']='NO_EXACT_TWO_TEAM_MARKET_PAIR__CHECK_ALIAS_OR_FIXTURE'
                note(idx,'PARTIAL_NAME_MATCH__VERIFY_ALIAS_OR_MARKET_FIXTURE')
            else:
                out.at[idx,'Quote_State']='NO_CURRENT_MARKET_FOR_THIS_TEAM_PAIR'
                note(idx,'NO_MARKET_TEAM_PAIR_IN_CURRENT_SPORTSBOOK_FEED')
            continue
        exact=cand.loc[(cand["_kick"]-kickoff.loc[idx]).abs().le(pd.Timedelta(minutes=KICKOFF_TOLERANCE_MINUTES))]
        kickoff_repaired=False
        if exact.empty and oriented:
            repaired=_verified_market_placeholder_repair(cand,kickoff.loc[idx],home,away)
            if repaired is not None and pd.notna(repaired) and repaired>now:
                exact=cand.loc[cand['_kick'].eq(repaired)]
                kickoff_repaired=True
        cand=exact
        if cand.empty:
            report["no_matching_game"]+=1
            report['no_valid_kickoff_match']+=1
            out.at[idx,'Quote_State']='NO_VERIFIED_KICKOFF_MATCH__CFBD_TIME_OR_DIFFERENT_MARKET_EVENT'
            note(idx,"KICKOFF_OUTSIDE_TOLERANCE" if not _cfbd_midnight_placeholder(kickoff.loc[idx])
                 else "CFBD_PLACEHOLDER_WITHOUT_UNIQUE_PAIRED_FUTURE_MARKET")
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
        if kickoff_repaired:
            # Source correction is observable and used by subsequent raw-overlay,
            # policy and game-side identity matching. The original is preserved.
            out.at[idx,'V55_Original_CFBD_Kickoff_UTC']=kickoff.loc[idx].isoformat()
            out.at[idx,'Game_Start_UTC']=unique_games.iloc[0]['_kick'].isoformat()
            out.at[idx,'V55_Kickoff_Verification']='VERIFIED_TWO_SIDED_SPORTSBOOK_REPLACED_CFBD_MIDNIGHT_PLACEHOLDER'
            report['verified_market_kickoff_repairs']+=1
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


# V5.5 raw pregame evidence is DISPLAY CONTEXT, not qualified evidence or betting authority.
# Its source is the existing scanner-enriched BigQuery market view, never V1
# probabilities, an independent odds API, or a dashboard computation.
V55_RAW_OVERLAY_TABLE = os.getenv('NCAAF_V55_RAW_OVERLAY_TABLE', 'sharplogger.sharp_data.moves_with_features_merged')
V55_RAW_OVERLAY_FIELDS = (
    'BigAl_Active_Text', 'BigAl_Side_Support_Count', 'BigAl_Side_Conflict_Count',
    'Pathi_Active_Text', 'Pathi_System_Count', 'System_Signals_Text',
    'NCAAF_RV2_System_Summary', 'NCAAF_RV2_Research_System_Summary',
    'NCAAF_RV22_Miner_Research_Votes', 'Miner_System_Summary', 'Miner_System_Count',
    'System_Side_Consensus_Count', 'System_Side_Conflict_Count',
    'NCAAF_Stat_Active', 'NCAAF_Stat_Trust',
    'PT_Consensus_Side', 'PT_Consensus', 'Prediction_Tracker_Consensus',
    'Open_Value', 'Line_Move_From_Open', 'Line_Move_30m', 'Line_Move_60m',
    'Sharp_Book_Move_60m', 'Sharp_Soft_Divergence',
    'Sharp_Consensus_Direction', 'Crossed_Key_3_Last60m',
    'Crossed_Key_7_Last60m', 'Crossed_Key_10_Last60m',
    'Crossed_Key_14_Last60m', 'Key_Cross_Confirmed_By_Sharp_Books',
    'Opening_Limit', 'Current_vs_Best_Line',
    # Structured directional, historically qualified Miner votes, if published.
    'NCAAF_RV22_Miner_Votes', 'NCAAF_Miner_Live_Authority_Policy',
    'Value', 'Odds_Price',
)
V55_RAW_OVERLAY_COLUMNS = (
    'V55_Raw_BigAl', 'V55_Raw_Pathi', 'V55_Raw_Miner',
    'V55_Raw_STAT', 'V55_Raw_PT', 'V55_Raw_Market_Rich',
    'V55_Raw_Overlay_State', 'V55_Raw_Overlay_Source',
    'V55_Raw_Overlay_AsOf_UTC', 'V55_Raw_Overlay_Detail_JSON',
)


def _v55_raw_clean(value):
    """Serialize pregame evidence safely without turning NaN into signals."""
    if value is None:
        return None
    try:
        import pandas as pd
        if not isinstance(value, (list, dict)) and bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, dict)):
        return json.dumps(value, default=_json_default, ensure_ascii=False)[:1200]
    s = str(value).strip()
    return s[:1200] if s and s.lower() not in ('nan', 'none', 'null', '—', '-') else None


def _v55_raw_named(row, field, count_field=None):
    if field in row.index:
        name = _v55_raw_clean(row.get(field))
        if name:
            return name[:240]
        count = _price(row.get(count_field)) if count_field in row.index else None
        return (f'{int(count)} unnamed triggers' if count and count > 0 else
                'NONE_REPORTED' if count is not None else 'NONE_REPORTED')
    count = _price(row.get(count_field)) if count_field in row.index else None
    if count is not None:
        return f'{int(count)} unnamed triggers' if count > 0 else '0 (no names in feed)'
    return 'NOT_PUBLISHED'


def _v55_raw_rich(row):
    names = (
        ('Line_Move_From_Open', 'Open→Now'), ('Line_Move_30m', '30m'),
        ('Line_Move_60m', '60m'), ('Sharp_Book_Move_60m', 'Sharp60'),
        ('Sharp_Soft_Divergence', 'Sharp/Soft'),
    )
    vals = []
    for col, label in names:
        if col in row.index:
            v = _price(row.get(col))
            if v is not None:
                vals.append(f'{label} {v:+.1f}')
    for col, label in (
        ('Crossed_Key_3_Last60m', 'Crossed 3'),
        ('Crossed_Key_7_Last60m', 'Crossed 7'),
        ('Crossed_Key_10_Last60m', 'Crossed 10'),
        ('Crossed_Key_14_Last60m', 'Crossed 14'),
    ):
        if col in row.index and (_price(row.get(col)) or 0) > 0:
            vals.append(label)
    return ' | '.join(vals[:6]) if vals else 'NOT_PUBLISHED'


def _v55_read_enriched_pregame_overlays(bq_client, *, horizon_days=14, lookback_minutes=720,
                                          qualified_brain_registry=None):
    """Recover legacy system overlays from the actual enriched scoring source.

    The unscored market view may have prices but not the legacy system columns.
    Try the saved scored feed as well, and retain source provenance. Each source
    is schema-inspected and bounded per game/side/book, not a 25k global bias.
    """
    import pandas as pd
    base_fields={'Game_Start','Home_Team_Norm','Away_Team_Norm',
                 'Outcome','Snapshot_Timestamp','Sport','Market'}
    qualified_fields=set()
    if isinstance(qualified_brain_registry,dict):
        for rule in (qualified_brain_registry.get('rules') or {}).values():
            qualified_fields.add(rule['signal'])
            if rule.get('ready_field'):qualified_fields.add(rule['ready_field'])
    sources=[V55_RAW_OVERLAY_TABLE]
    for candidate in ('sharplogger.sharp_data.sharp_scores_with_features',
                      'sharplogger.sharp_data.moves_with_features_merged'):
        if candidate not in sources:sources.append(candidate)
    pieces=[];statuses=[];failures=[]
    for source in sources:
        if not re.fullmatch(r'[A-Za-z_][\w-]*\.[A-Za-z_][\w-]*\.[A-Za-z_][\w-]*',source):
            failures.append('BAD_TABLE_ID');continue
        try:
            schema={f.name for f in bq_client.get_table(source).schema}
        except Exception as exc:
            failures.append(source.rsplit('.',1)[-1]+':SCHEMA_'+type(exc).__name__);continue
        if not base_fields.issubset(schema):
            failures.append(source.rsplit('.',1)[-1]+':MISSING_IDENTITY');continue
        optional=list(dict.fromkeys(x for x in (*V55_RAW_OVERLAY_FIELDS,'Bookmaker','Book',
                         *sorted(qualified_fields)) if x in schema and x not in base_fields))
        if not optional:
            failures.append(source.rsplit('.',1)[-1]+':NO_OVERLAY_COLUMNS');continue
        if 'Bookmaker' in schema and 'Book' in schema:
            book_expr="COALESCE(NULLIF(TRIM(CAST(Bookmaker AS STRING)), ''), NULLIF(TRIM(CAST(Book AS STRING)), ''), '')"
        elif 'Bookmaker' in schema:
            book_expr="COALESCE(CAST(Bookmaker AS STRING), '')"
        elif 'Book' in schema:
            book_expr="COALESCE(CAST(Book AS STRING), '')"
        else:book_expr="''"
        fields=list(sorted(base_fields))+optional
        sql=f"""SELECT {','.join('`'+x+'`' for x in fields)}
          FROM `{source}`
          WHERE UPPER(TRIM(CAST(Sport AS STRING)))='NCAAF'
            AND LOWER(TRIM(CAST(Market AS STRING)))='spreads'
            AND Game_Start > CURRENT_TIMESTAMP()
            AND Game_Start <= TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL {int(horizon_days)} DAY)
            AND Snapshot_Timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {int(lookback_minutes)} MINUTE)
            AND Snapshot_Timestamp <= CURRENT_TIMESTAMP()
            AND Snapshot_Timestamp < Game_Start
          QUALIFY ROW_NUMBER() OVER (
             PARTITION BY Game_Start, Home_Team_Norm, Away_Team_Norm, Outcome, {book_expr}
             ORDER BY Snapshot_Timestamp DESC) <= 12
          ORDER BY Game_Start, Snapshot_Timestamp DESC LIMIT 25000"""
        try:
            frame=bq_client.query(sql).to_dataframe(create_bqstorage_client=False)
        except Exception as exc:
            logging.warning('[NCAAF-V55-RAW-OVERLAY-QUERY-ERROR] table=%s %s: %s',
                            source,type(exc).__name__,str(exc)[:350])
            failures.append(source.rsplit('.',1)[-1]+':QUERY_'+type(exc).__name__);continue
        if frame.empty:
            statuses.append(source.rsplit('.',1)[-1]+':EMPTY');continue
        frame['V55_Overlay_Actual_Source']=source
        pieces.append(frame)
        statuses.append(source.rsplit('.',1)[-1]+':'+str(len(frame))+'rows')
    if not pieces:
        return pd.DataFrame(),'NO_LIVE_OVERLAY_ROWS__'+','.join((failures+statuses)[:5])
    merged=pd.concat(pieces,ignore_index=True,sort=False)
    return merged,'PASS_MULTI_SOURCE__'+';'.join(statuses)+'__FAILURES:'+','.join(failures[:3])


def attach_v55_raw_pregame_overlays(board, enrich, *, source_status='PASS', now=None):
    """Display actual named research overlays without granting them bet authority.

    Unlike betting decisions, display can show an explicitly marked cross-book
    or home-side context record. Market Rich is ALWAYS sportsbook-and-side
    specific; verified voting continues to use independent exact join functions.
    Past timestamps are never inferred from the current clock.
    """
    import pandas as pd
    out=board.copy()
    for col in V55_RAW_OVERLAY_COLUMNS:
        out[col]=None
    out['V55_Raw_Overlay_State']='NO_LIVE_SYSTEM_CONTEXT'
    report={'status':source_status,'priced_sides':0,'unpriced_games':0,
        'matched_raw_overlays':0,'matched_same_book_side':0,
        'cross_book_display_only':0,'home_side_display_only':0,
        'unpriced_display_only':0,'source_rows':0,'matched_game_context':0,
        'source_has_structured_miner_votes':False,
        'source_has_named_miner_text':False,'structured_miner_vote_rows':0,
        'quote_count_context_rows':0,'named_big_al_rows':0,
        'named_pathi_rows':0,'named_miner_rows':0,'rich_rows':0,
        'no_game_context':0,'no_usable_context_at_quote':0}
    if out.empty:return out,report
    now=_timestamp(now or datetime.now(timezone.utc))
    has_source=isinstance(enrich,pd.DataFrame) and not enrich.empty
    if has_source:
        required={'Game_Start','Home_Team_Norm','Away_Team_Norm','Outcome','Snapshot_Timestamp'}
        if not required.issubset(enrich.columns):
            has_source=False;report['status']='SOURCE_REQUIRED_IDENTITY_MISSING'
        else:
            e=enrich.copy()
            e['_kick']=pd.to_datetime(e['Game_Start'],utc=True,errors='coerce')
            e['_snap']=pd.to_datetime(e['Snapshot_Timestamp'],utc=True,errors='coerce')
            e['_home']=e['Home_Team_Norm'].map(_norm_team)
            e['_away']=e['Away_Team_Norm'].map(_norm_team)
            e['_side']=e['Outcome'].map(_norm_team)
            e=e.loc[e['_home'].ne('') & e['_away'].ne('') & e['_side'].ne('')
                   & e['_snap'].notna() & e['_kick'].notna()
                   & e['_snap'].lt(e['_kick']) & e['_snap'].le(now)].copy()
            report['source_rows']=len(e)
            report['source_has_structured_miner_votes']='NCAAF_RV22_Miner_Votes' in e.columns
            report['source_has_named_miner_text']=bool({'NCAAF_RV2_System_Summary','Miner_System_Summary','System_Signals_Text'}&set(e.columns))
            if 'Bookmaker' in e.columns or 'Book' in e.columns:
                a=e['Bookmaker'] if 'Bookmaker' in e.columns else pd.Series('',index=e.index)
                b=e['Book'] if 'Book' in e.columns else pd.Series('',index=e.index)
                e['_book']=a.fillna('').astype(str).str.lower().str.strip()
                e['_book']=e['_book'].mask(e['_book'].isin(('', 'nan','none')),
                                         b.fillna('').astype(str).str.lower().str.strip())
            else:e['_book']=''
    for idx,r in out.iterrows():
        quote=_timestamp(r.get('Backend_Quote_Timestamp'))
        kick=_timestamp(r.get('Game_Start_UTC'))
        if pd.isna(kick) or not now<kick:continue
        priced=not pd.isna(quote) and quote<=now and quote<kick
        if priced:report['priced_sides']+=1
        else:report['unpriced_games']+=1
        prior=_v55_raw_clean(r.get('V55_Trigger_Context'))
        context={}
        if priced and prior:
            try:context=json.loads(prior)
            except (TypeError,ValueError):pass
        if context:
            report['quote_count_context_rows']+=1
            ba1=context.get('BigAl_Side_Support_Count');ba2=context.get('BigAl_Side_Conflict_Count')
            if _price(ba1) is not None or _price(ba2) is not None:
                out.at[idx,'V55_Raw_BigAl']=f'HOME quote counts: support {ba1 or 0}, opposition {ba2 or 0} (unnamed)'
            pa=context.get('Pathi_System_Count')
            if _price(pa) is not None:out.at[idx,'V55_Raw_Pathi']=f'HOME quote systems: {pa} (unnamed)'
            sy1=context.get('System_Side_Consensus_Count');sy2=context.get('System_Side_Conflict_Count')
            if _price(sy1) is not None or _price(sy2) is not None:
                out.at[idx,'V55_Raw_Miner']=f'HOME quote system counts: support {sy1 or 0}, opposition {sy2 or 0} (not Miner IDs)'
            out.at[idx,'V55_Raw_Overlay_State']='QUOTE_HOME_SIDE_COUNTS_ONLY'
            out.at[idx,'V55_Raw_Overlay_Source']=MARKET_TABLE
        if not has_source:continue
        (home,away),oriented=_matchup_teams(r)
        selected=_norm_team(r.get('V55_Research_Lean_Team') or r.get('Selected_Team'))
        # Pure CORE-only, two-sided priced games may not have a selected lean.
        # Show a canonical HOME-side system evaluation for completeness only;
        # no recommendation or qualifying vote is inferred from this fallback.
        no_core_side=selected not in (home,away)
        if not oriented:continue
        if no_core_side:selected=home
        exact_market_kick=(e['_kick']-kick).abs().le(pd.Timedelta(minutes=KICKOFF_TOLERANCE_MINUTES))
        # Miner research snapshots can still contain the original CFBD midnight
        # sentinel. Permit this ONLY for display context, NEVER qualified votes.
        original_kick=_timestamp(r.get('V55_Original_CFBD_Kickoff_UTC'))
        if pd.notna(original_kick) and 'V55_Source_Type' in e.columns:
            original_snapshot=(e['V55_Source_Type'].fillna('').astype(str)
                .eq('HEADLESS_MINER_SNAPSHOT_DISPLAY_ONLY'))
            exact_market_kick |= (original_snapshot &
                (e['_kick']-original_kick).abs().le(pd.Timedelta(minutes=KICKOFF_TOLERANCE_MINUTES)))
        v=e.loc[e['_home'].eq(home)&e['_away'].eq(away)&exact_market_kick]
        if v.empty:report['no_game_context']+=1;continue
        report['matched_game_context']+=1
        # A priced bet can only display information known AT THAT PRICE.
        anchor=quote if priced else now
        direct=(v.get('V55_Live_Computed_On_Current_Quote',pd.Series(False,index=v.index)).fillna(False).eq(True)
                &v['_snap'].ge(anchor)&v['_snap'].le(now)
                &(now-v['_snap']).le(pd.Timedelta(minutes=5)))
        v=v.loc[(v['_snap'].le(anchor)
            &(anchor-v['_snap']).le(pd.Timedelta(minutes=720 if not priced else 240))) | direct]
        if v.empty:report['no_usable_context_at_quote']+=1;continue
        book=str(r.get('Backend_Quote_Book') or '').lower().strip()
        same_side=v.loc[v['_side'].eq(selected)]
        same_book=same_side.loc[same_side['_book'].eq(book)] if book else same_side.iloc[0:0]
        # Prefer an exact displayed quote, then cross-book selected side, then
        # canonically oriented HOME context explicitly labeled as such.
        if not same_book.empty:chosen=same_book;status='MATCHED_PRE_GAME_ENRICHED_SIDE__UNQUALIFIED';report['matched_same_book_side']+=1
        elif not same_side.empty:chosen=same_side;status='CROSS_BOOK_GAME_SYSTEM_CONTEXT_DISPLAY_ONLY';report['cross_book_display_only']+=1
        else:
            chosen=v.loc[v['_side'].eq(home)]
            if chosen.empty:continue
            status='HOME_SIDE_CONTEXT_NOT_SELECTED_BET_SIDE';report['home_side_display_only']+=1
        if no_core_side:
            status='MODEL_ONLY_HOME_CONTEXT_NO_SELECTED_BET_SIDE'
        if not priced:
            status='UNPRICED_GAME_SYSTEM_CONTEXT_DISPLAY_ONLY';report['unpriced_display_only']+=1
        elif not no_core_side and bool(chosen.get('V55_Live_Computed_On_Current_Quote',pd.Series(False,index=chosen.index)).fillna(False).eq(True).any()):
            status='LIVE_SYSTEM_RECOMPUTED_AT_REFRESH__QUOTE_EXECUTION_RECHECK_REQUIRED'
        # Prefer the most recent available named observation, rather than the
        # most recent EMPTY odds refresh that wiped the research fields.
        def richness(rec):
            present=0
            for field in ('NCAAF_RV2_System_Summary','NCAAF_RV2_Research_System_Summary','Miner_System_Summary',
                          'BigAl_Active_Text','Pathi_Active_Text','System_Signals_Text',
                          'NCAAF_RV22_Miner_Votes'):
                z=_v55_raw_clean(rec.get(field))
                if z and z not in ('—','[]','{}'):
                    present+=1
            return present
        candidates=chosen.copy()
        candidates['_richness']=candidates.apply(richness,axis=1)
        candidates['_live_engine']=candidates.get('V55_Live_Computed_On_Current_Quote',pd.Series(False,index=candidates.index)).fillna(False).eq(True).astype(int)
        # Named context up to 240m old is preferable to fresh empty rows,
        # but NEVER preferable to quote-exact qualified event matching.
        picked=candidates.sort_values(['_live_engine','_richness','_snap'],ascending=False).iloc[0]
        out.at[idx,'V55_Raw_BigAl']=_v55_raw_named(picked,'BigAl_Active_Text','BigAl_Side_Support_Count')
        out.at[idx,'V55_Raw_Pathi']=_v55_raw_named(picked,'Pathi_Active_Text','Pathi_System_Count')
        summary=_v55_raw_clean(picked.get('NCAAF_RV2_System_Summary'))
        if not summary or summary in ('—','NONE_REPORTED'):
            summary=_v55_raw_clean(picked.get('Miner_System_Summary'))
        # Structured original miner events are DISPLAYED as named research
        # events, but are NOT automatically qualified by this display bridge.
        votes=_v55_parse_structured_votes(picked.get('NCAAF_RV22_Miner_Votes'))
        report['structured_miner_vote_rows']+=len(votes)
        if votes:
            names=[f"{x.get('family_id')} → {x.get('target')}"
                   for x in votes if isinstance(x,dict) and x.get('family_id') and x.get('target')]
            if names:summary=' | '.join(names[:8])
        if not summary:
            shadow=_v55_raw_clean(picked.get('NCAAF_RV2_Research_System_Summary'))
            if shadow and shadow not in ('—','NONE_REPORTED'):
                summary='[RESEARCH/SHADOW DISPLAY ONLY] '+shadow
        if not summary:
            system_text=_v55_raw_clean(picked.get('System_Signals_Text'))
            if system_text:
                names=[x.strip() for x in system_text.split('|')
                       if x.strip().upper().startswith(('MINER ','MINER:','NCAAF_RV2','NCAAF_RV22'))]
                if names:summary=' | '.join(names[:8])
        out.at[idx,'V55_Raw_Miner']=summary[:500] if summary else _v55_raw_named(picked,'Miner_System_Summary','Miner_System_Count')
        # A legacy exact-book odds refresh may contain zero research columns.
        # Preserve the separate, original Miner research snapshot as a
        # secondary *display-only* source (never as quote-matched authority).
        miner_value=str(out.at[idx,'V55_Raw_Miner'] or '')
        if miner_value in ('NOT_PUBLISHED','NONE_REPORTED','—','0 (no names in feed)') or 'unnamed' in miner_value:
            if 'V55_Source_Type' in v.columns:
                source_type=v['V55_Source_Type'].fillna('').astype(str)
                research_only=v.loc[source_type.eq('HEADLESS_MINER_SNAPSHOT_DISPLAY_ONLY')
                    &v['_snap'].le(anchor)&v['_snap'].lt(kick)]
                if not research_only.empty:
                    rr=research_only.sort_values('_snap',ascending=False).iloc[0]
                    names=_v55_raw_clean(rr.get('NCAAF_RV2_System_Summary'))
                    votes=_v55_parse_structured_votes(rr.get('NCAAF_RV22_Miner_Votes'))
                    if not names or names=='—':
                        names=_v55_raw_clean(rr.get('NCAAF_RV2_Research_System_Summary'))
                        if names and names != '—':
                            names='[SHADOW] '+names
                    if not names or names=='—':
                        names=' | '.join(f"{q.get('family_id')} → {q.get('target')}"
                            for q in votes if isinstance(q,dict) and q.get('family_id') and q.get('target'))
                    if names:
                        out.at[idx,'V55_Raw_Miner']='[RESEARCH GAME CONTEXT ONLY] '+names[:450]
                        report['structured_miner_vote_rows']+=len(votes)
        system_text=_v55_raw_clean(picked.get('System_Signals_Text'))
        if system_text:
            bits=[x.strip() for x in system_text.split('|') if x.strip()]
            for col,prefix in (('V55_Raw_BigAl','BIG AL'),('V55_Raw_Pathi','PATHI')):
                val=str(out.at[idx,col] or '')
                if val in ('NOT_PUBLISHED','NONE_REPORTED') or 'unnamed' in val:
                    named=[x for x in bits if x.upper().startswith(prefix+' ')]
                    if named:out.at[idx,col]=' | '.join(named[:8])[:240]
        active=_price(picked.get('NCAAF_Stat_Active'))
        trust=_price(picked.get('NCAAF_Stat_Trust'))
        if active is not None:
            out.at[idx,'V55_Raw_STAT']=('ACTIVE'+(f' trust={trust:.2f}' if trust is not None else '') if active>0 else 'NOT_ACTIVE')
        for pc in ('PT_Consensus_Side','PT_Consensus','Prediction_Tracker_Consensus'):
            z=_v55_raw_clean(picked.get(pc))
            if z:out.at[idx,'V55_Raw_PT']=z[:240];break
        # Only match movement to actual bookmaker AND execution side;
        # nothing from cross-book or home-only rows can look executable.
        if priced and not same_book.empty:
            richrow=same_book.sort_values('_snap',ascending=False).iloc[0]
            out.at[idx,'V55_Raw_Market_Rich']=_v55_raw_rich(richrow)
        else:out.at[idx,'V55_Raw_Market_Rich']='NOT_VERIFIED_FOR_EXECUTION_BOOK'
        raw={k:_v55_raw_clean(picked.get(k)) for k in V55_RAW_OVERLAY_FIELDS if k in picked.index}
        raw={k:v for k,v in raw.items() if v is not None}
        raw.update({'display_side':str(picked.get('Outcome')),
                    'display_book':str(picked.get('_book')),
                    'display_only':True,'selected_team':str(r.get('Selected_Team')),
                    'verified_quote_book':book if priced else None})
        out.at[idx,'V55_Raw_Overlay_Detail_JSON']=json.dumps(raw,ensure_ascii=False,sort_keys=True)
        out.at[idx,'V55_Raw_Overlay_AsOf_UTC']=picked['_snap'].isoformat()
        out.at[idx,'V55_Raw_Overlay_Source']=str(picked.get('V55_Overlay_Actual_Source') or V55_RAW_OVERLAY_TABLE)
        out.at[idx,'V55_Raw_Overlay_State']=status
        report['matched_raw_overlays']+=1
        for col,key in [('V55_Raw_BigAl','named_big_al_rows'),('V55_Raw_Pathi','named_pathi_rows'),
                        ('V55_Raw_Miner','named_miner_rows'),('V55_Raw_Market_Rich','rich_rows')]:
            value=str(out.at[idx,col] or '')
            if value and value not in ('NOT_PUBLISHED','NONE_REPORTED','NOT_VERIFIED_FOR_EXECUTION_BOOK') and 'unnamed' not in value and not value.startswith('0 ('):
                report[key]+=1
    return out,report



def _v55_read_headless_miner_system_snapshot(*,storage_client=None,bucket_name=None,now=None):
    """Read original live Miner rows; verify book/line/timing before authority.

    Legacy snapshots remain display-only. The 2026 original-source format can
    qualify ONLY after the separate frozen OOS rule+exact-quote checks.
    """
    import pandas as pd
    blob='research/ncaaf/v55/live_miner_system_context/latest.json'
    try:
        if storage_client is None:
            from google.cloud import storage
            storage_client=storage.Client()
        bucket=bucket_name or os.getenv('MODEL_BUCKET','sharp-models')
        raw=storage_client.bucket(bucket).blob(blob).download_as_bytes()
        if len(raw)>6_000_000:return pd.DataFrame(),'MINER_SNAPSHOT_TOO_LARGE'
        doc=json.loads(raw)
        if not isinstance(doc,dict) or doc.get('source') not in ('NCAAF_RV22_HEADLESS_MINER_OUTPUT_DISPLAY_ONLY','NCAAF_RV22_ORIGINAL_LIVE_SYSTEMS') or doc.get('authority') is not False:
            return pd.DataFrame(),'MINER_SNAPSHOT_PROVENANCE_REJECTED'
        t=_timestamp(doc.get('generated_at_utc'));now=_timestamp(now or datetime.now(timezone.utc))
        if pd.isna(t) or pd.isna(now) or t>now or (now-t).total_seconds()>12*3600:
            return pd.DataFrame(),'MINER_SNAPSHOT_STALE_OR_FUTURE'
        rows=doc.get('rows')
        if not isinstance(rows,list) or len(rows)>1800:return pd.DataFrame(),'MINER_SNAPSHOT_INVALID_ROWS'
        source=pd.DataFrame(rows)
        required={'Game_Start','Home_Team_Norm','Away_Team_Norm','Outcome',
                  'Snapshot_Timestamp','NCAAF_RV22_Miner_Votes'}
        if not required.issubset(source.columns):return pd.DataFrame(),'MINER_SNAPSHOT_REQUIRED_FIELDS_MISSING'
        # Old snapshots carried only names; newer snapshots preserve the
        # original quote identity. Require the signed producer schema AND the
        # original timestamp/book/price before sending to authority verification.
        trusted=(doc.get('source')=='NCAAF_RV22_ORIGINAL_LIVE_SYSTEMS' and
                 str(doc.get('source_module') or '').startswith('ncaaf-research-v2.'))
        if not trusted:
            source=source.drop(columns=[x for x in ('Bookmaker','Book','Value','Odds_Price') if x in source.columns])
            source['V55_Original_Quote_Provenance_Verified']=False
        else:
            marked=source.get('V55_Original_Quote_Provenance_Verified',pd.Series(False,index=source.index))
            mask=marked.map(lambda x:x is True or str(x).lower()=='true')
            for c in ('Value','Snapshot_Timestamp','Outcome'):
                if c not in source:mask &=False
            if not {'Book','Bookmaker'}.intersection(source):mask &=False
            source['V55_Original_Quote_Provenance_Verified']=mask
            if 'Value' in source:source.loc[~mask,'Value']=None
            for bookcol in ('Book','Bookmaker'):
                if bookcol in source:source.loc[~mask,bookcol]=None
        source['V55_Source_Type']='HEADLESS_MINER_SNAPSHOT_DISPLAY_ONLY'
        source['V55_Overlay_Actual_Source']='gs://'+bucket+'/'+blob
        return source,('PASS_MARKET_PROVENANCE_CHECKED' if trusted else 'PASS_DISPLAY_ONLY')
    except Exception as exc:
        return pd.DataFrame(),'MINER_SNAPSHOT_UNAVAILABLE:'+type(exc).__name__



# V55 DIRECT LIVE BRAINS — run the legacy *feature evaluators*, not legacy V1
# scoring. The earlier V5.5 bridge only queried saved features from market
# views; those views are not populated by the independent CORE execution path.
def _v55_direct_current_quote_rows(board, market, *, now=None):
    """Select ONLY this board's verified sportsbook/book/side/line quote pairs.

    Do not evaluate unrelated stale book quotes, synthesize prices, or use a
    hypothetical CFBD scheduled kickoff in place of the verified market event.
    """
    import pandas as pd
    at = _timestamp(now or _utcnow())
    if not isinstance(board,pd.DataFrame) or not isinstance(market,pd.DataFrame) or market.empty:
        return pd.DataFrame(), {'status':'NO_VERIFIED_MARKET_FRAME','games':0,'rows':0}
    req={'Home_Team_Norm','Away_Team_Norm','Game_Start','Outcome','Value','Snapshot_Timestamp'}
    if not req.issubset(market.columns):
        return pd.DataFrame(),{'status':'MISSING_QUOTE_COLUMNS','missing':sorted(req-set(market.columns)),'games':0,'rows':0}
    q=market.copy()
    q['_home']=q['Home_Team_Norm'].map(_norm_team)
    q['_away']=q['Away_Team_Norm'].map(_norm_team)
    q['_side']=q['Outcome'].map(_norm_team)
    q['_kick']=pd.to_datetime(q['Game_Start'],utc=True,errors='coerce')
    q['_obs']=pd.to_datetime(q['Snapshot_Timestamp'],utc=True,errors='coerce')
    bk=q.get('Bookmaker',pd.Series('',index=q.index)).fillna('').astype(str).str.lower().str.strip()
    alt=q.get('Book',pd.Series('',index=q.index)).fillna('').astype(str).str.lower().str.strip()
    q['_book']=bk.mask(bk.isin(('','nan','none')),alt)
    q['_spread']=pd.to_numeric(q['Value'],errors='coerce')
    found=[]; games=0
    for _,r in board.iterrows():
        if not str(r.get('Quote_State') or '').startswith('FRESH_'):
            continue
        (home,away),oriented=_matchup_teams(r)
        if not oriented:continue
        kick=_timestamp(r.get('Game_Start_UTC'))
        anchor=_timestamp(r.get('Backend_Quote_Timestamp'))
        book=str(r.get('Backend_Quote_Book') or '').lower().strip()
        hv=_price(r.get('Backend_Home_Spread'))
        av=_price(r.get('Backend_Away_Spread'))
        if not (book and pd.notna(kick) and pd.notna(anchor) and anchor<=at<kick
                and hv is not None and av is not None and abs(hv+av)<.01):
            continue
        parts=q.loc[q['_home'].eq(home)&q['_away'].eq(away)&q['_book'].eq(book)
            &(q['_kick']-kick).abs().le(pd.Timedelta(minutes=KICKOFF_TOLERANCE_MINUTES))
            &q['_obs'].le(anchor)&(anchor-q['_obs']).le(pd.Timedelta(minutes=15))]
        if parts.empty:continue
        chosen=[]
        for side,spread in ((home,hv),(away,av)):
            opt=parts.loc[parts['_side'].eq(side)&parts['_spread'].sub(spread).abs().le(.001)].sort_values('_obs',ascending=False)
            if opt.empty:chosen=[];break
            chosen.append(opt.iloc[0].copy())
        if len(chosen)!=2:continue
        if abs((chosen[0]['_obs']-chosen[1]['_obs']).total_seconds())>900:continue
        games+=1
        found.extend(chosen)
    if not found:return pd.DataFrame(),{'status':'NO_EXACT_BOOK_TWO_SIDED_PAIRS','games':0,'rows':0}
    rows=pd.DataFrame(found).drop_duplicates(subset=['_home','_away','_kick','_side','_book','_spread'],keep='last')
    # These are original odds, not model/V1 features. The live evaluation runs
    # against completed-game history and cannot infer any current-game result.
    rows['Sport']='NCAAF';rows['Market']='spreads'
    rows['Game']=rows['Away_Team_Norm'].astype(str)+' @ '+rows['Home_Team_Norm'].astype(str)
    rows['V55_Original_Quote_Timestamp']=rows['Snapshot_Timestamp']
    rows['V55_Original_Quote_Provenance_Verified']=True
    rows['V55_Source_Type']='LIVE_ENGINE_RECALCULATED'
    return rows.drop(columns=[c for c in rows.columns if c.startswith('_')],errors='ignore'),{
        'status':'PASS','games':games,'rows':len(rows),'source':'ORIGINAL_VERIFIED_SCANNER_BOOK_QUOTES'}


def _v55_evaluate_existing_live_systems(board, market, *, bq_client,
                                         storage_client=None, bucket_name=None, now=None,
                                         utils_module=None, miner_module=None):
    """Restore former NCAAF/NFL-style live system evaluation on V5.5 quotes.

    Models are not loaded/refit here. Big Al/Pathi and Miner are evaluated by
    their existing implementations against CURRENT verified sportsbook sides;
    final qualification happens later via independent frozen research registries.
    """
    import pandas as pd
    from types import SimpleNamespace
    import time
    t0=time.monotonic()
    quote_rows,audit=_v55_direct_current_quote_rows(board,market,now=now)
    audit.update({'big_al_pathi':'NOT_RUN','miner':'NOT_RUN','named_big_al_rows':0,
                  'named_pathi_rows':0,'named_miner_rows':0,'miner_vote_rows':0,
                  'miner_report_status':'NOT_LOADED'})
    if quote_rows.empty:return quote_rows,audit
    try:
        if utils_module is None:import utils as utils_module
        # Two prior seasons are enough to derive live form/role/revenge inputs.
        # The frozen historical qualification still uses the FULL 2013-25 set.
        updated=utils_module.attach_pathi_bigal_backend_features(quote_rows,'NCAAF',days_back=730)
        if not isinstance(updated,pd.DataFrame) or len(updated)!=len(quote_rows):
            raise RuntimeError('LEGACY_LIVE_ENRICHMENT_ROW_COUNT_CHANGED')
        quote_rows=updated
        audit['big_al_pathi']='PASS' if any(c in quote_rows for c in ('BigAl_Active_Text','Pathi_Active_Text','System_Signals_Text')) else 'NO_SYSTEM_FIELDS_PRODUCED'
        # Football key-number concepts existed in the older Pathi feature
        # engine but did not appear in Pathi_Active_Text. Show real computed
        # flags separately; these are descriptive context, NOT qualified votes.
        fb_flags=tuple(c for c in ('Pathi_FB_Dog_Below_Key_3','Pathi_FB_Dog_Below_Key_7',
            'Pathi_FB_Dog_10_Plus','Pathi_FB_Dog_Above_7','Pathi_FB_Dog_Hook_Above_3',
            'Pathi_FB_Dog_Hook_Above_7','Pathi_FB_Dog_Hook_Above_10',
            'Pathi_FB_Favorite_Below_Key_3','Pathi_FB_Favorite_Below_Key_7',
            'Pathi_FB_Favorite_Below_Key_10','Pathi_FB_Moved_Through_Key') if c in quote_rows)
        audit['pathi_football_flag_columns']=len(fb_flags)
        audit['pathi_football_context_rows']=0
        if fb_flags:
            for _ix,_r in quote_rows.iterrows():
                _flags=[c.removeprefix('Pathi_FB_').replace('_',' ') for c in fb_flags
                    if _price(_r.get(c))==1]
                if not _flags:continue
                audit['pathi_football_context_rows']+=1
                _old=str(_r.get('Pathi_Active_Text') or '—').strip()
                if _old in ('—','','nan','None'):
                    quote_rows.at[_ix,'Pathi_Active_Text']='[FOOTBALL RESEARCH CONTEXT ONLY] '+ ' | '.join(_flags[:5])
        exact=quote_rows.get('System_Signals_Text',pd.Series('',index=quote_rows.index)).fillna('').astype(str)
        audit['big_al_exact_firings']=int(exact.str.contains('BIG AL ',regex=False).sum())
        audit['pathi_exact_firings']=int(exact.str.contains('PATHI ',regex=False).sum())
    except Exception as exc:
        audit['big_al_pathi']='ERROR:'+type(exc).__name__+':'+str(exc)[:130]
        logging.exception('[NCAAF-V55-LIVE-BRAIN-CALC] stage=BIGAL_PATHI')
    # Try original confirmed Miner evaluator directly — no separate headless
    # job needs to publish an orphaned snapshot for V5.5 to see its triggers.
    try:
        if miner_module is None:import ncaaf_research_v2 as miner_module
        report, rep_status = miner_module.load_current_report_diagnostic(
            storage_client=storage_client,
            bucket_name=bucket_name or os.getenv('MODEL_BUCKET','sharp-models'))
        audit['miner_report_status']=str(rep_status.get('status') or 'UNKNOWN')
        cross_version_shadow=False
        if report is None and audit['miner_report_status']=='SOURCE_TAG_MISMATCH':
            # A valid prior V2.x library may still evaluate original named
            # RESEARCH triggers. Do not pretend its version is the current
            # frozen authority: zero all qualified vote fields after scoring.
            try:
                from google.cloud import storage as _storage
                _sc=storage_client or _storage.Client()
                _bucket=bucket_name or os.getenv('MODEL_BUCKET','sharp-models')
                _raw=_sc.bucket(_bucket).blob(miner_module.REPORT_CURRENT_BLOB).download_as_bytes()
                _old=json.loads(_raw)
                _created=_timestamp(_old.get('created_utc'))
                if (isinstance(_old,dict)
                        and str(_old.get('source_tag') or '').startswith('ncaaf-research-v2.')
                        and isinstance(_old.get('system_miner_v3'),dict)
                        and pd.notna(_created) and _created<=_timestamp(now or _utcnow())):
                    report=_old
                    cross_version_shadow=True
                    audit['miner_report_status']='OLD_V2_VERSION_RESEARCH_DISPLAY_ONLY'
            except Exception as exc:
                audit['cross_version_read_error']=type(exc).__name__
        if isinstance(report,dict):
            ctx=SimpleNamespace(bq_client=bq_client,
                HISTORICAL_NCAAF_CORE_VIEW='sharplogger.sharp_data.ncaaf_historical_core_training_vw',
                gcs_client=storage_client,GCS_BUCKET=bucket_name or os.getenv('MODEL_BUCKET','sharp-models'))
            evaluated=miner_module.attach_live_miner_votes(quote_rows,report,dashboard_module=ctx)
            if not isinstance(evaluated,pd.DataFrame) or len(evaluated)!=len(quote_rows):
                raise RuntimeError('LIVE_MINER_EVALUATOR_CHANGED_ROW_COUNT')
            quote_rows=evaluated
            if cross_version_shadow:
                quote_rows['NCAAF_RV22_Miner_Votes']=[[] for _ in range(len(quote_rows))]
                quote_rows['NCAAF_RV2_System_Count']=0
                quote_rows['NCAAF_RV2_System_Summary']='—'
            audit['miner']='PASS_SHADOW_CROSS_VERSION' if cross_version_shadow else 'PASS'
        else:audit['miner']='REPORT_NOT_READY:'+audit['miner_report_status']
    except Exception as exc:
        audit['miner']='ERROR:'+type(exc).__name__+':'+str(exc)[:130]
        logging.exception('[NCAAF-V55-LIVE-BRAIN-CALC] stage=MINER')
    # We KNOW the time when these signals were evaluated. Do not backdate their
    # observation to the bookmaker's earlier quote timestamp.
    evaluated_at=_timestamp(now or _utcnow())
    quote_rows['Snapshot_Timestamp']=evaluated_at
    quote_rows['V55_Live_Computed_On_Current_Quote']=True
    for c,k in (('BigAl_Active_Text','named_big_al_rows'),
                ('Pathi_Active_Text','named_pathi_rows'),
                ('NCAAF_RV2_Research_System_Summary','named_miner_rows')):
        if c in quote_rows:
            v=quote_rows[c].fillna('').astype(str).str.strip()
            audit[k]=int((~v.isin(('', '—','nan','None','NOT_PUBLISHED'))).sum())
    if 'NCAAF_RV22_Miner_Votes' in quote_rows:
        audit['miner_vote_rows']=sum(len(_v55_parse_structured_votes(v)) for v in quote_rows['NCAAF_RV22_Miner_Votes'])
    if 'NCAAF_RV22_Miner_Research_Votes' in quote_rows:
        audit['miner_shadow_vote_rows']=sum(len(_v55_parse_structured_votes(v)) for v in quote_rows['NCAAF_RV22_Miner_Research_Votes'])
    audit['elapsed_seconds']=round(time.monotonic()-t0,2)
    audit['status']='PASS_LIVE_SYSTEM_EVALUATION' if audit['big_al_pathi']=='PASS' or audit['miner'].startswith('PASS') else 'NO_EVALUATOR_COMPLETED'
    return quote_rows,audit


def _v55_live_quote_time_eligible(live_rows, quote_time, now):
    """Allow post-quote live evaluation ONLY if its original price is recent.

    Original pre-quote archived signals continue under the legacy as-of rule.
    This guard prevents re-timestamping an old quote to invent betting authority.
    """
    import pandas as pd
    obs=pd.to_datetime(live_rows['Snapshot_Timestamp'],utc=True,errors='coerce')
    original=pd.to_datetime(live_rows.get('V55_Original_Quote_Timestamp',pd.Series(index=live_rows.index,dtype='object')),utc=True,errors='coerce')
    marked=live_rows.get('V55_Live_Computed_On_Current_Quote',pd.Series(False,index=live_rows.index)).fillna(False).eq(True)
    return (marked & original.notna() & original.le(quote_time)
            &(quote_time-original).le(pd.Timedelta(minutes=15))
            &((now-quote_time)<=pd.Timedelta(minutes=15))
            &obs.ge(quote_time)&obs.le(now)
            &(now-obs).le(pd.Timedelta(minutes=5)))

def bridge_weekly_prices(weekly, audit, *, bq_client, log_func=None, storage_client=None, bucket_name=None):
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
    live_system_rows,live_system_audit=_v55_evaluate_existing_live_systems(
        out,market,bq_client=bq_client,storage_client=storage_client,
        bucket_name=bucket_name)
    updated['v55_live_system_evaluation']=live_system_audit
    # Shared enriched feed: raw overlays are display-only; ONLY a separately
    # authenticated, research-registry-matched directional vote can count.
    other_registry, other_registry_problem = read_v55_long_history_brain_registry(
        storage_client=storage_client,bucket_name=bucket_name)
    overlay_frame, overlay_source_status = _v55_read_enriched_pregame_overlays(
        bq_client,qualified_brain_registry=other_registry)
    # Independently published live Miner identifiers are merged for RAW display
    # only. They are never used by the qualified vote extractor, which continues
    # to require real quote/price/provenance and the frozen research registry.
    miner_display_frame, miner_display_status = _v55_read_headless_miner_system_snapshot(
        storage_client=storage_client,bucket_name=bucket_name)
    display_frames=[x for x in (live_system_rows,overlay_frame,miner_display_frame)
                    if isinstance(x,pd.DataFrame) and not x.empty]
    raw_display_frame=(pd.concat(display_frames,ignore_index=True,sort=False)
                       if display_frames else pd.DataFrame())
    out, raw_overlays = attach_v55_raw_pregame_overlays(
        out,raw_display_frame,source_status=overlay_source_status)
    raw_overlays['headless_miner_snapshot_status']=miner_display_status
    raw_overlays['headless_miner_snapshot_rows']=len(miner_display_frame)
    updated["v55_raw_overlay_audit"] = raw_overlays
    explicit_events, evidence_problem = read_v55_qualified_evidence(bq_client)
    registry, registry_problem = read_v55_research_miner_registry(
        storage_client=storage_client,bucket_name=bucket_name)
    # Original legacy Miner evaluator already produces the correct validated
    # mechanism IDs. Its GCS handoff is admitted only when ORIGINAL book, line,
    # observation timestamp, and pregame provenance are intact.
    miner_authority_rows=(pd.concat([overlay_frame,live_system_rows],ignore_index=True,sort=False)
        if not live_system_rows.empty else overlay_frame)
    if isinstance(miner_display_frame,pd.DataFrame) and not miner_display_frame.empty:
        checked=miner_display_frame.loc[
            miner_display_frame.get('V55_Original_Quote_Provenance_Verified',
                pd.Series(False,index=miner_display_frame.index)).eq(True)]
        if not checked.empty:
            miner_authority_rows=pd.concat([miner_authority_rows,checked],ignore_index=True,sort=False)
    miner_events, miner_audit = derive_v55_miner_qualified_events(
        out,miner_authority_rows,registry)
    miner_audit['direct_runtime_rows']=len(live_system_rows)
    miner_audit['original_snapshot_provenance_rows']=max(0,len(miner_authority_rows)-len(overlay_frame)-len(live_system_rows))
    other_authority_rows=(pd.concat([overlay_frame,live_system_rows],ignore_index=True,sort=False)
        if not live_system_rows.empty else overlay_frame)
    other_events, other_audit = derive_v55_other_qualified_events(
        out,other_authority_rows,other_registry)
    # Externally ingested evidence rows are admissible only if backed by a
    # matching frozen rule. String statuses in a live table are not a certificate.
    vetted_explicit, explicit_gate = validate_v55_external_qualified_events(
        explicit_events,other_registry)
    event_frames=[x for x in (vetted_explicit,miner_events,other_events)
                  if isinstance(x,pd.DataFrame) and not x.empty]
    combined_events=pd.concat(event_frames,ignore_index=True) if event_frames else None
    out, brain = attach_v55_qualified_evidence(out,combined_events)
    brain['miner_registry_status']=registry_problem or 'PASS'
    brain['miner_live_bridge']=miner_audit
    brain['other_brain_registry_status']=other_registry_problem or 'PASS'
    brain['long_history_other_brain_bridge']=other_audit
    brain['external_evidence_gate']=explicit_gate
    brain['long_history_reference']='2013-2025; pre-2026 independent evidence only'
    if evidence_problem:
        brain["additional_evidence_table_status"] = evidence_problem
    updated.update({k:v for k,v in research.items() if k!="status"})
    updated["research_decision_status"]=research["status"]
    updated.update(brain)
    # The original model source audit is immutable: price/research statuses
    # must never replace it with a generic PASS marker.
    # Governance handoff is independent of the model's immutable, shadow-only
    # training audit. Only the published V5.5 decision policy can grant bets.
    out, policy = apply_v55_owner_betting_policy(out)
    updated.update(policy)
    updated["status"]=audit["status"]
    updated["price_join_semantics"] = "FRESH_QUOTES_V55_OWNER_DIRECTED_DECISION"
    log("[NCAAF-V55-LIVE-SYSTEM-EVALUATION] " + json.dumps(live_system_audit,sort_keys=True))
    log("[NCAAF-V55-BACKEND-PRICE-BRIDGE] " + json.dumps(diag,sort_keys=True))
    log("[NCAAF-V55-RESEARCH-SIDE-EV] " + json.dumps(research,sort_keys=True))
    log("[NCAAF-V55-INDEPENDENT-BRAIN-EVIDENCE] " + json.dumps(brain,sort_keys=True))
    log("[NCAAF-V55-RAW-OVERLAYS] " + json.dumps(raw_overlays,sort_keys=True))
    log("[NCAAF-V55-OWNER-AUTHORITY] " + json.dumps(policy,sort_keys=True))
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



# ----- Historical frozen-system -> timestamped V5.5 game-side evidence bridge -----
V55_RESEARCH_MINER_REPORT_BLOB = "research/ncaaf/v2/current_report.json"
V55_RESEARCH_MINER_POLICY = "NCAAF_MINER_LIVE_AUTHORITY_V2_20_SOURCE_NEUTRAL_STRONG_VALIDATED_20261007"


def read_v55_research_miner_registry(*, storage_client=None, bucket_name=None):
    """Read the immutable pre-2026 qualification *criteria* from the research report.

    The report may have been published in 2026; its discovery/confirmation
    seasons must be <=2025. No 2026 ATS/game outcome is permitted to qualify a
    system. This bridge never executes researcher pickle payloads.
    """
    try:
        if storage_client is None:
            from google.cloud import storage
            storage_client=storage.Client()
        client=storage_client
        bucket=bucket_name or os.getenv('MODEL_BUCKET','sharp-models')
        blob=client.bucket(bucket).blob(V55_RESEARCH_MINER_REPORT_BLOB)
        raw=blob.download_as_bytes()
        report=json.loads(raw)
        if not isinstance(report,dict):return None,'RESEARCH_REPORT_NOT_OBJECT'
        if report.get('status')!='NCAAF_RESEARCH_V2_COMPLETE':return None,'RESEARCH_REPORT_NOT_COMPLETE'
        if not str(report.get('source_tag') or '').startswith('ncaaf-research-v2.'):
            return None,'RESEARCH_REPORT_PROVENANCE_UNVERIFIED'
        if int(report.get('discovery_max_season',9999))>2023:return None,'DISCOVERY_NOT_FROZEN'
        if sorted(int(x) for x in report.get('confirmation_seasons',[]))!=[2024,2025]:return None,'CONFIRMATION_YEARS_NOT_FROZEN'
        if (report.get('miner_live_authority_policy') or {}).get('policy')!=V55_RESEARCH_MINER_POLICY:return None,'RESEARCH_POLICY_MISMATCH'
        if (report.get('miner_live_authority_policy') or {}).get('uses_2026_selection') is not False:return None,'2026_SELECTION_UNVERIFIED'
        created=_timestamp(report.get('created_utc'))
        if created is None or str(created)=='NaT':return None,'RESEARCH_REPORT_TIMESTAMP_MISSING'
        mechanisms={}
        for rec in (report.get('system_miner_v3') or {}).get('spreads',{}).get('mechanism_families',[]):
            mid=str(rec.get('mechanism_id') or '')
            cond=[str(c) for c in rec.get('representative_conditions') or []]
            if not mid or not cond or mid in mechanisms:continue
            if any(x.startswith(('CORE_OOF_','SPEC_','CFBD_WEATHER_')) for x in cond):continue
            if rec.get('incumbent_revalidation_status') in ('HOLD_NOT_EVALUABLE','LEGACY_CONTRACT_HOLD'):continue
            if rec.get('confirmation_pass') is not True or rec.get('live_authority_eligible') is not True:continue
            if rec.get('split_audit',{}).get('disjoint') is not True:continue
            if int(rec.get('confirmation_n') or 0)<60 or float(rec.get('confirmation_rate') or 0)<0.56:continue
            if rec.get('meta_constituent_overlap_guard','PASS')!='PASS':continue
            if rec.get('market')!='spreads':continue
            mechanisms[mid]={
                'target':rec.get('direction'),
                'evidence_family_key':str(rec.get('evidence_family_key') or mid),
                'rule':' AND '.join(cond),
                'n':int(rec.get('confirmation_n')),
                'rate':float(rec.get('confirmation_rate')),
            }
        return {'mechanisms':mechanisms,'created_utc':created,
                'report_sha256':hashlib.sha256(raw).hexdigest(),
                'uri':f'gs://{bucket}/{V55_RESEARCH_MINER_REPORT_BLOB}'},None
    except Exception as exc:
        return None,'RESEARCH_REGISTRY_READ_FAILED:'+type(exc).__name__


def _v55_parse_structured_votes(value):
    """Parse JSON/repeated-record vote fields without inferring votes from counts."""
    if isinstance(value,str):
        try:value=json.loads(value)
        except (TypeError,ValueError):return []
    if isinstance(value,tuple):value=list(value)
    if not isinstance(value,list) and hasattr(value,'tolist'):
        value=value.tolist()
    if not isinstance(value,list):return []
    cleaned=[]
    for v in value:
        if isinstance(v,dict):cleaned.append(v)
        elif hasattr(v,'as_py') and isinstance(v.as_py(),dict):cleaned.append(v.as_py())
    return cleaned


def derive_v55_miner_qualified_events(board, live_rows, registry, *, now=None):
    """Join frozen system IDs to live *directional* pregame triggers.

    The upstream live Miner must publish NCAAF_RV22_Miner_Votes. Never create
    confirmations from raw named text, Big Al ratings, counts, or historical
    system inventory alone. A trigger and its certified historical rule must
    both exist, and the report must predate the observed pregame signal.
    """
    import pandas as pd
    cols=('Game_ID','Game_Start_UTC','Source_Family','Signal_Family',
          'Evidence_Family_Key','Signal_Side','Observed_At_UTC',
          'Qualified_At_UTC','Qualification_Status','OOS_Status',
          'Direction_Verified','Provenance_Status')
    audit={'status':'NO_MINER_EVENTS','report_mechanisms':0,'live_source_rows':0,
           'structured_vote_rows':0,'matched_game_sides':0,'qualified_events':0,
           'source':'sharplogger.sharp_data.moves_with_features_merged',
           'distinct_games':0,'distinct_mechanisms':0}
    empty=pd.DataFrame(columns=cols)
    if not isinstance(registry,dict):
        audit['status']='RESEARCH_REGISTRY_UNAVAILABLE_FAIL_CLOSED';return empty,audit
    rules=registry.get('mechanisms') or {}
    audit['report_mechanisms']=len(rules)
    audit['research_registry_sha256']=registry.get('report_sha256')
    audit['research_registry_uri']=registry.get('uri')
    if not rules:
        audit['status']='NO_HISTORICALLY_QUALIFIED_SPREAD_MECHANISMS';return empty,audit
    if not isinstance(live_rows,pd.DataFrame) or live_rows.empty:
        audit['status']='NO_LIVE_SOURCE_ROWS';return empty,audit
    needed={'NCAAF_RV22_Miner_Votes','Game_Start','Home_Team_Norm',
            'Away_Team_Norm','Snapshot_Timestamp','Outcome','Value'}
    if 'NCAAF_RV22_Miner_Votes' not in live_rows.columns:
        audit['status']='STRUCTURED_LIVE_MINER_VOTES_NOT_PUBLISHED';return empty,audit
    if not needed.issubset(live_rows.columns):
        audit['status']='LIVE_MARKET_IDENTITY_OR_SPREAD_UNAVAILABLE';return empty,audit
    audit['live_source_rows']=len(live_rows)
    live=live_rows.copy()
    live['_kick']=pd.to_datetime(live['Game_Start'],utc=True,errors='coerce')
    live['_obs']=pd.to_datetime(live['Snapshot_Timestamp'],utc=True,errors='coerce')
    live['_home']=live['Home_Team_Norm'].map(_norm_team)
    live['_away']=live['Away_Team_Norm'].map(_norm_team)
    live['_side']=live['Outcome'].map(_norm_team)
    emit=[]
    report_time=_timestamp(registry['created_utc'])
    now=_timestamp(now or datetime.now(timezone.utc))
    if pd.isna(report_time) or report_time>now:
        audit['status']='REGISTRY_TIME_INVALID_OR_FUTURE';return empty,audit
    # One event per (game, mechanism, direction), retaining latest valid sighting.
    dedup={}
    for _,r in board.iterrows():
        selected=_norm_team(r.get('Selected_Team'))
        if not selected:continue
        (home,away),oriented=_matchup_teams(r)
        if not oriented or selected not in (home,away):continue
        quote=_timestamp(r.get('Backend_Quote_Timestamp'))
        kick=_timestamp(r.get('Game_Start_UTC'))
        if pd.isna(quote) or pd.isna(kick) or not quote<kick or not quote<=now:continue
        gid=str(r.get('Game_ID') or '')
        selected_spread=_price(r.get('Current_Spread'))
        if not gid or selected_spread is None:continue
        hits=live.loc[(live['_home']==home)&(live['_away']==away)
              &(live['_side']==selected)
              &(live['_kick']-kick).abs().le(pd.Timedelta(minutes=KICKOFF_TOLERANCE_MINUTES))
              &live['_obs'].le(now)&live['_obs'].lt(kick)&live['_obs'].ge(report_time)
              &((live['_obs'].le(quote)&(quote-live['_obs']).le(pd.Timedelta(minutes=90)))
                |_v55_live_quote_time_eligible(live,quote,now))]
        # A researched role/line trigger must apply at the *execution* spread.
        hits=hits.loc[pd.to_numeric(hits['Value'],errors='coerce').sub(selected_spread).abs().le(.001)]
        # The derived Miner rule may depend on market spread or role. Use the
        # EXACT sportsbook attached to the V5.5 selection, not another book.
        book=str(r.get('Backend_Quote_Book') or '').lower().strip()
        if book and ('Bookmaker' in hits or 'Book' in hits):
            b1=hits['Bookmaker'] if 'Bookmaker' in hits else pd.Series('',index=hits.index)
            b2=hits['Book'] if 'Book' in hits else pd.Series('',index=hits.index)
            hb=b1.fillna('').astype(str).str.lower().str.strip()
            hb=hb.mask(hb.isin(('','nan','none')),b2.fillna('').astype(str).str.lower().str.strip())
            hits=hits.loc[hb.eq(book)]
        if hits.empty:continue
        audit['matched_game_sides']+=1
        for _,x in hits.iterrows():
            votes=_v55_parse_structured_votes(x['NCAAF_RV22_Miner_Votes'])
            audit['structured_vote_rows']+=len(votes)
            for v in votes:
                mid=str(v.get('family_id') or '')
                rule=rules.get(mid)
                if rule is None:continue
                if str(v.get('source_type') or '').upper()!='MINER':continue
                if str(v.get('market') or '').lower()!='spreads':continue
                if v.get('live_authority_eligible') is not True:continue
                if v.get('live_authority_policy')!=V55_RESEARCH_MINER_POLICY:continue
                if v.get('evidence_level')!='STRONG_VALIDATED':continue
                if str(v.get('evidence_family_key') or '')!=rule['evidence_family_key']:continue
                if str(v.get('rule') or '')!=rule['rule']:continue
                try:
                    vote_n=int(v.get('confirmation_n') or 0)
                    vote_rate=float(v.get('confirmation_rate') or 0)
                except (TypeError,ValueError,OverflowError):continue
                if vote_n!=rule['n']:continue
                if not math.isfinite(vote_rate) or abs(vote_rate-rule['rate'])>1e-8:continue
                target=_norm_team(v.get('target'))
                if target not in (home,away):continue
                expected_target=home if rule['target']=='PLAY_ON' else away if rule['target']=='FADE' else None
                if expected_target is None or target!=expected_target:continue
                obs=x['_obs'];key=(gid,mid,target)
                source_family=('PREDICTION_TRACKER' if v.get('uses_external_ratings_family') is True
                               else 'SYSTEM_MINER')
                direct=x.get('V55_Live_Computed_On_Current_Quote') is True
                source_status='PRE_GAME_VERIFIED_LIVE_RECALCULATED' if direct else 'PRE_GAME_VERIFIED'
                payload=dict(zip(cols,(gid,kick.isoformat(),source_family,mid,
                    rule['evidence_family_key'],target,obs.isoformat(),
                    report_time.isoformat(),'QUALIFIED_OOS','PASS',
                    True,source_status)))
                if direct:payload['Observed_Quote_UTC']=str(x.get('V55_Original_Quote_Timestamp'))
                if key not in dedup or obs>_timestamp(dedup[key]['Observed_At_UTC']):dedup[key]=payload
    emit=list(dedup.values())
    audit['qualified_events']=len(emit)
    audit['distinct_games']=len({x['Game_ID'] for x in emit})
    audit['distinct_mechanisms']=len({x['Signal_Family'] for x in emit})
    audit['status']='PASS' if emit else ('MATCHED_LIVE_ROWS_NO_QUALIFIED_TRIGGER' if audit['matched_game_sides'] else 'NO_VERIFIED_SAME_SIDE_ASOF_LIVE_ROWS')
    return pd.DataFrame(emit).reindex(columns=list(cols)+['Observed_Quote_UTC'] if any('Observed_Quote_UTC' in r for r in emit) else list(cols)),audit



# Historical independent brain governance: 2013-2023 discovery, 2024-2025
# confirmation, 2026 sealed for prospective monitoring. The historical model
# does NOT create events: a PRE-GAME, book/line-aligned, named live trigger must
# also be observed. These rules must be exported by audited Heavy Research,
# not invented from non-directional overlay counts.
V55_LONG_HISTORY_BRAIN_BLOB = 'research/ncaaf/v55/qualified_independent_brains_2013_2025.json'
V55_LONG_HISTORY_POLICY = 'V55-FROZEN-INDEPENDENT-BRAINS-2013-2025-R1'
V55_NONMINER_FAMILIES = ('BIG_AL','PATHI','STAT','PREDICTION_TRACKER')


def _v55_validate_long_history_brain_registry(doc, *, raw_sha='', now=None):
    """Validate a research-generated frozen independent system qualification.

    Each entry needs an auditable 2024/2025 out-of-sample result and the
    exact pregame flag/column contract. In particular, descriptive Pathi/BigAl
    trust scores and the 2026 season are NOT qualification certificates.
    """
    if not isinstance(doc,dict) or doc.get('status')!='FROZEN_OOS_QUALIFIED':
        return None,'BRAIN_REGISTRY_NOT_FROZEN_OOS'
    if doc.get('policy')!=V55_LONG_HISTORY_POLICY or doc.get('operator_approved') is not True:
        return None,'BRAIN_REGISTRY_POLICY_OR_APPROVAL_MISSING'
    try:
        first=int(doc['history_first_season']);last=int(doc['discovery_last_season'])
        confirmations=sorted(int(v) for v in doc['confirmation_seasons'])
    except (TypeError,KeyError,ValueError):
        return None,'BRAIN_REGISTRY_SEASONS_MISSING'
    if first>2013 or last>2023 or confirmations!=[2024,2025] or doc.get('uses_2026_outcomes') is not False:
        return None,'BRAIN_REGISTRY_LEAKAGE_OR_HISTORY_GAP'
    try:
        observed=sorted(int(v) for v in (doc.get('source_audit') or {})['observed_seasons'])
    except (KeyError,TypeError,ValueError):
        return None,'BRAIN_REGISTRY_HISTORICAL_SEASON_COVERAGE_MISSING'
    if observed!=list(range(2013,2026)):
        return None,'BRAIN_REGISTRY_2013_2025_SEASON_COVERAGE_INCOMPLETE'
    created=_timestamp(doc.get('created_utc'))
    at=_timestamp(now or datetime.now(timezone.utc))
    if created is None or str(created)=='NaT' or created>at:
        return None,'BRAIN_REGISTRY_FUTURE_OR_MISSING_DATE'
    rules={};rejects={}
    for entry in doc.get('systems',[]):
        if not isinstance(entry,dict):continue
        fam=str(entry.get('family') or '').upper()
        sid=str(entry.get('system_id') or '')
        key=(fam,sid)
        if not sid or fam not in V55_NONMINER_FAMILIES or key in rules:continue
        try:
            n=int(entry.get('confirmation_n') or 0)
            rate=float(entry.get('confirmation_rate') or 0)
        except (TypeError,ValueError,OverflowError):continue
        oos=entry.get('by_season') or {}
        try:
            y2024=oos.get('2024') or {};y2025=oos.get('2025') or {}
            y1=int(y2024['n']); y2=int(y2025['n'])
            rate1=float(y2024['rate']);rate2=float(y2025['rate'])
        except (KeyError,TypeError,ValueError):continue
        # Independent holdout certification AND source/vintage audit.
        if not (entry.get('oos_pass') is True and entry.get('split_disjoint') is True
                and entry.get('pregame_feature_vintage_pass') is True
                and entry.get('source_neutral') is True
                and entry.get('role')=='directional'
                and entry.get('uses_2026_outcomes') is False
                and entry.get('overlap_audit')=='PASS'
                and entry.get('multiple_testing_audit')=='PASS'
                and n==y1+y2 and n>=60 and rate>=0.56
                and y1>=15 and y2>=15
                and 0<=rate1<=1 and 0<=rate2<=1
                and abs((rate1*y1+rate2*y2)/n-rate)<0.001):
            rejects[fam]=rejects.get(fam,0)+1
            continue
        field=str(entry.get('live_signal_field') or '')
        typ=str(entry.get('live_signal_type') or 'FLAG').upper()
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,127}',field) or typ not in ('FLAG','SIDE'):
            continue
        if typ=='FLAG' and not field.startswith(('BigAl_','Pathi_','NCAAF_Stat_')):
            continue
        ready=str(entry.get('data_ready_field') or '')
        if ready and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,127}',ready):
            continue
        if fam=='BIG_AL' and not field.startswith('BigAl_'):continue
        if fam=='PATHI' and not field.startswith('Pathi_'):continue
        if fam=='STAT' and not field.startswith('NCAAF_Stat_'):continue
        if fam=='PREDICTION_TRACKER' and typ!='SIDE':continue
        if str(entry.get('direction')) not in ('PLAY_ON','FADE'):continue
        rules[key]={'family':fam,'id':sid,'signal':field,'signal_type':typ,
                    'direction':entry['direction'], 'n':n,'rate':rate,
                    'key':str(entry.get('evidence_family_key') or fam),
                    'ready_field':ready,
                    'qualified_at':created, 'report_sha256':raw_sha}
    return {'rules':rules,'created_utc':created,'report_sha256':raw_sha,
            'source_seasons':'2013-2025','rejected_by_family':rejects},None


def read_v55_long_history_brain_registry(*, storage_client=None,bucket_name=None):
    try:
        if storage_client is None:
            from google.cloud import storage
            storage_client=storage.Client()
        bucket=bucket_name or os.getenv('MODEL_BUCKET','sharp-models')
        raw=storage_client.bucket(bucket).blob(V55_LONG_HISTORY_BRAIN_BLOB).download_as_bytes()
        obj=json.loads(raw)
        result,problem=_v55_validate_long_history_brain_registry(
            obj,raw_sha=hashlib.sha256(raw).hexdigest())
        if result:result['uri']=f'gs://{bucket}/{V55_LONG_HISTORY_BRAIN_BLOB}'
        return result,problem
    except Exception as exc:
        return None,'BRAIN_REGISTRY_UNAVAILABLE:'+type(exc).__name__


def derive_v55_other_qualified_events(board, live_rows, registry, *, now=None):
    """Re-evaluate Big Al, Pathi, STAT, PT on the CURRENT booked spread.

    No V1 model probabilities, historical 2026 outcomes, raw unnamed counts, or
    context-only rules can create a confirmation. Missing named source columns
    or historical OOS registry => zero events and explicit audit status.
    """
    import pandas as pd
    columns=('Game_ID','Game_Start_UTC','Source_Family','Signal_Family',
       'Evidence_Family_Key','Signal_Side','Observed_At_UTC','Qualified_At_UTC',
       'Qualification_Status','OOS_Status','Direction_Verified',
       'Provenance_Status','Observed_Book','Observed_Spread','Registry_SHA256')
    empty=pd.DataFrame(columns=columns)
    report={'status':'NO_QUALIFIED_NONMINER_EVENTS','registry_systems':0,
        'live_signal_columns':0,'matched_games':0,'qualified_events':0,
        'qualified_by_brain':{},'source_seasons':'2013-2025',
        'raw_display_does_not_grant_authority':True}
    if not isinstance(registry,dict):
        report['status']='HISTORICAL_BRAIN_REGISTRY_UNAVAILABLE';return empty,report
    rules=registry.get('rules') or {}
    report['registry_systems']=len(rules)
    if not rules:
        report['status']='NO_FROZEN_OOS_BRAIN_SYSTEMS';return empty,report
    if not isinstance(live_rows,pd.DataFrame) or live_rows.empty:
        report['status']='NO_LIVE_GAME_SIDE_SIGNAL_SOURCE';return empty,report
    needed={'Game_Start','Home_Team_Norm','Away_Team_Norm','Outcome',
            'Snapshot_Timestamp','Value'}
    if not needed.issubset(live_rows.columns) or not {'Bookmaker','Book'}.intersection(live_rows.columns):
        report['status']='MISSING_LIVE_IDENTITY_BOOK_OR_SPREAD';return empty,report
    fields={v['signal'] for v in rules.values()}
    report['live_signal_columns']=len(fields & set(live_rows.columns))
    if not fields.intersection(live_rows.columns):
        report['status']='QUALIFIED_RULE_FIELDS_NOT_PUBLISHED';return empty,report
    live=live_rows.copy()
    if 'Sport' in live.columns:
        live=live.loc[live['Sport'].astype(str).str.upper().eq('NCAAF')]
    if 'Market' in live.columns:
        live=live.loc[live['Market'].astype(str).str.lower().eq('spreads')]
    if live.empty:
        report['status']='NO_NCAAF_SPREAD_SOURCE_ROWS';return empty,report
    live['_kick']=pd.to_datetime(live['Game_Start'],utc=True,errors='coerce')
    live['_obs']=pd.to_datetime(live['Snapshot_Timestamp'],utc=True,errors='coerce')
    live['_home']=live['Home_Team_Norm'].map(_norm_team)
    live['_away']=live['Away_Team_Norm'].map(_norm_team)
    live['_side']=live['Outcome'].map(_norm_team)
    if 'Bookmaker' not in live.columns:
        live['Bookmaker']=live['Book']
    live['_book']=live['Bookmaker'].fillna('').astype(str).str.lower().str.strip()
    if 'Book' in live:
        fallback=live['Book'].fillna('').astype(str).str.lower().str.strip()
        live['_book']=live['_book'].mask(live['_book'].isin(('', 'nan','none')),fallback)
    live['_spread']=pd.to_numeric(live['Value'],errors='coerce')
    at=_timestamp(now or datetime.now(timezone.utc))
    dedup={}
    for _,r in board.iterrows():
        selected=_norm_team(r.get('Selected_Team'))
        (home,away),oriented=_matchup_teams(r)
        qt=_timestamp(r.get('Backend_Quote_Timestamp'))
        kick=_timestamp(r.get('Game_Start_UTC'))
        chosen_spread=_price(r.get('Current_Spread'))
        book=str(r.get('Backend_Quote_Book') or '').lower().strip()
        gid=str(r.get('Game_ID') or '')
        if not (oriented and selected in (home,away) and gid and book and chosen_spread is not None):continue
        if pd.isna(qt) or pd.isna(kick) or pd.isna(at) or not qt<=at<kick:continue
        hits=live.loc[(live['_home']==home)&(live['_away']==away)
          &(live['_kick']-kick).abs().le(pd.Timedelta(minutes=KICKOFF_TOLERANCE_MINUTES))
          &live['_book'].eq(book)&live['_obs'].lt(kick)
          &((live['_obs'].le(qt)&(qt-live['_obs']).le(pd.Timedelta(minutes=90)))
            |_v55_live_quote_time_eligible(live,qt,at))]
        if hits.empty:continue
        report['matched_games']+=1
        for rule in rules.values():
            field=rule['signal']
            if field not in hits.columns:continue
            src=hits.loc[hits['_obs'].ge(rule['qualified_at'])]
            if src.empty:continue
            if rule['ready_field']:
                if rule['ready_field'] not in src.columns:continue
                src=src.loc[pd.to_numeric(src[rule['ready_field']],errors='coerce').eq(1)]
            if rule['signal_type']=='FLAG':
                src=src.loc[pd.to_numeric(src[field],errors='coerce').eq(1)]
            elif rule['signal_type']=='SIDE':
                src=src.loc[src[field].map(_norm_team).eq(src['_side'])]
            if src.empty:continue
            for _,hit in src.iterrows():
                side=hit['_side'];direction=side if rule['direction']=='PLAY_ON' else (away if side==home else home)
                needed_spread=chosen_spread if side==selected else -chosen_spread
                if side not in (home,away) or abs(hit['_spread']-needed_spread)>0.001:continue
                if direction not in (home,away):continue
                key=(gid,rule['family'],rule['id'],direction)
                direct=hit.get('V55_Live_Computed_On_Current_Quote') is True
                source_status='PRE_GAME_VERIFIED_LIVE_RECALCULATED' if direct else 'PRE_GAME_VERIFIED'
                record=dict(zip(columns,(gid,kick.isoformat(),rule['family'],rule['id'],
                    rule['key'],direction,hit['_obs'].isoformat(),rule['qualified_at'].isoformat(),
                    'QUALIFIED_OOS','PASS',True,source_status,book,
                    float(hit['_spread']),rule['report_sha256'])))
                if direct:record['Observed_Quote_UTC']=str(hit.get('V55_Original_Quote_Timestamp'))
                if key not in dedup or hit['_obs']>_timestamp(dedup[key]['Observed_At_UTC']):dedup[key]=record
    emit=list(dedup.values())
    report['qualified_events']=len(emit)
    for x in emit:
        fam=x['Source_Family'];report['qualified_by_brain'][fam]=report['qualified_by_brain'].get(fam,0)+1
    report['status']='PASS' if emit else 'NO_MATCHING_QUALIFIED_PREGAME_RULE_FIRINGS'
    return pd.DataFrame(emit).reindex(columns=list(columns)+['Observed_Quote_UTC'] if any('Observed_Quote_UTC' in r for r in emit) else list(columns)),report


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


def validate_v55_external_qualified_events(events,registry):
    """Prevent an arbitrary BigQuery row from self-authorizing a named system."""
    import pandas as pd
    audit={'status':'NONE','read':len(events) if isinstance(events,pd.DataFrame) else 0,'verified':0}
    if not isinstance(events,pd.DataFrame) or events.empty:
        return None,audit
    if not isinstance(registry,dict):
        audit['status']='NO_FROZEN_BRAIN_REGISTRY';return None,audit
    required={'Source_Family','Signal_Family','Evidence_Family_Key',
              'Registry_SHA256','Observed_Book','Observed_Spread',
              'Game_ID','Game_Start_UTC','Signal_Side','Observed_At_UTC',
              'Qualified_At_UTC','Qualification_Status','OOS_Status',
              'Direction_Verified','Provenance_Status'}
    if not required.issubset(events.columns):
        audit['status']='EVENT_CONTRACT_MISSING';return None,audit
    rules=registry.get('rules') or {}
    rows=[]
    for _,row in events.iterrows():
        fam=str(row.get('Source_Family') or '').upper()
        sid=str(row.get('Signal_Family') or '')
        rule=rules.get((fam,sid))
        if not rule:continue
        qt=_timestamp(row.get('Qualified_At_UTC'))
        if (str(row.get('Evidence_Family_Key'))!=rule['key']
          or str(row.get('Registry_SHA256'))!=rule['report_sha256']
          or pd.isna(qt) or qt!=rule['qualified_at']):continue
        rows.append(row.to_dict())
    audit['verified']=len(rows)
    audit['status']='PASS' if rows else 'NO_FROZEN_REGISTRY_MATCH'
    return pd.DataFrame(rows) if rows else None,audit


def attach_v55_qualified_evidence(frame,events, *, now=None):
    """Only independently qualified sources may confirm or oppose V5.5 CORE.

    Exact CFBD game ID, kickoff, directional team, qualification using
    historically frozen 2013–2025 evidence, observation <= quote time,
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
    out['V55_Independent_Support_Keys']=None
    out['V55_Independent_Conflict_Keys']=None
    out['V55_Active_System_Names']=None
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
    ev['_original_quote']=pd.to_datetime(ev.get('Observed_Quote_UTC',pd.Series(index=ev.index,dtype='object')),utc=True,errors='coerce')
    observed_now=_timestamp(now or _utcnow())
    def check(v): return v is True or str(v).lower() in ("true","1")
    valid=(ev["_family"].isin(VALID_BRAIN_FAMILIES)
           &ev["Qualification_Status"].eq("QUALIFIED_OOS")
           &ev["OOS_Status"].eq("PASS")
           &ev["Provenance_Status"].isin(("PRE_GAME_VERIFIED","PRE_GAME_VERIFIED_LIVE_RECALCULATED"))
           &ev["Direction_Verified"].map(check)
           &ev["_qualified"].lt(ev["_kick"])
           &ev["_obs"].lt(ev["_kick"])
           &ev["_obs"].ge(ev["_qualified"]))
    ev=ev.loc[valid]
    if ev.empty:
        report["evidence_state"]="NO_VALID_V55_OOS_EVENTS"
        return out,report
    report["evidence_state"]="EVALUATED_V55_OOS_EVENTS"
    out["V55_Qualified_Evidence_Detail_JSON"]=None
    report["qualified_event_rows_validated"]=int(len(ev))
    for i,r in out.iterrows():
        selected=_norm_team(r.get("Selected_Team"))
        gid=str(r.get("Game_ID") or "")
        quote_time=_timestamp(r.get("Backend_Quote_Timestamp"))
        kick=_timestamp(r.get("Game_Start_UTC"))
        (home,away),_= _matchup_teams(r)
        if not selected or not gid or pd.isna(quote_time) or pd.isna(kick):
            continue
        archived=(ev['Provenance_Status'].eq('PRE_GAME_VERIFIED')&ev['_obs'].le(quote_time))
        # A live evaluator cannot backdate itself to the quote snapshot.
        # Admit it only with an unchanged recent sportsbook quote and a recent
        # 2026 PREGAME evaluation, then maintain exact family/line constraints.
        live=(ev['Provenance_Status'].eq('PRE_GAME_VERIFIED_LIVE_RECALCULATED')
              &ev['_original_quote'].notna()
              &ev['_original_quote'].le(quote_time)
              &(quote_time-ev['_original_quote']).le(pd.Timedelta(minutes=15))
              &ev['_obs'].ge(quote_time)
              &(ev['_obs']-quote_time).le(pd.Timedelta(minutes=15))
              &ev['_obs'].le(observed_now)
              &(observed_now-ev['_obs']).le(pd.Timedelta(minutes=5))
              &((observed_now-quote_time)<=pd.Timedelta(minutes=15)))
        relevant=ev.loc[(ev["_game_id"]==gid)&ev["_kick"].eq(kick)&(archived|live)].copy()
        # Explicit Big Al/Pathi/STAT/PT event must be sourced to the same
        # sportsbook and spread; the legacy Miner producer performs these
        # checks before emission. A quoted event cannot self-verify.
        if not relevant.empty:
            source_book=str(r.get('Backend_Quote_Book') or '').lower().strip()
            value=_price(r.get('Current_Spread'))
            keep=[]
            for _,evrow in relevant.iterrows():
                family=str(evrow.get('_family') or '')
                if family=='SYSTEM_MINER':
                    keep.append(True);continue
                book=str(evrow.get('Observed_Book') or '').lower().strip()
                spread=_price(evrow.get('Observed_Spread'))
                # The source market row is allowed to be the opposite team for
                # a FADE system; explicit evidence must be on the chosen side.
                keep.append(bool(source_book and book==source_book and
                                 spread is not None and value is not None and
                                 abs(abs(spread)-abs(value))<=0.001))
            relevant=relevant.loc[keep]
        if relevant.empty:
            continue
        # Independent-voter components: connect events that share a source brain
        # OR an underlying evidence family. This prevents duplicate Miner votes,
        # PT/Miner external ratings double counting, and mixed-direction signals
        # from a single brain being treated as independent confirmations.
        relevant['_dependency']=relevant.apply(
            lambda z:str(z.get('Evidence_Family_Key') or z.get('_family') or '').strip().upper(),axis=1)
        records=relevant.to_dict('records')
        parents=list(range(len(records)))
        def root(i):
            while parents[i]!=i:
                parents[i]=parents[parents[i]];i=parents[i]
            return i
        def union(i,j):
            a=root(i);b=root(j)
            if a!=b:parents[b]=a
        seen_source={};seen_dep={}
        for ix,evr in enumerate(records):
            f=evr['_family'];d=evr['_dependency']
            if f in seen_source:union(ix,seen_source[f])
            if d in seen_dep:union(ix,seen_dep[d])
            seen_source[f]=ix;seen_dep[d]=ix
        components={}
        for ix,evr in enumerate(records):components.setdefault(root(ix),[]).append(evr)
        support=set();opposition=set();conflicts=set()
        independent_support=set();independent_oppose=set()
        for component in components.values():
            dirs={_norm_team(v['Signal_Side']) for v in component}
            dirs.discard('')
            if not dirs or not dirs.issubset({home,away}):continue
            brains={v['_family'] for v in component}
            label='+'.join(sorted(brains))
            if len(dirs)>1:
                independent_oppose.add(label)
                opposition.update(brains);conflicts.update(brains)
            elif next(iter(dirs))==selected:
                independent_support.add(label);support.update(brains)
            else:
                independent_oppose.add(label);opposition.update(brains)
        support.difference_update(conflicts)
        # Source-specific provenance, signal ID and timing remain inspectable.
        evidence_details=[]
        for _,ee in relevant.iterrows():
            evidence_details.append({"source":str(ee.get("Source_Family")),
                "system_id":str(ee.get("Signal_Family")),
                "direction":str(ee.get("Signal_Side")),
                "observed_utc":str(ee.get("Observed_At_UTC")),
                "qualified_utc":str(ee.get("Qualified_At_UTC")),
                "evidence_family_key":str(ee.get("Evidence_Family_Key") or "") if pd.notna(ee.get("Evidence_Family_Key")) else ""})
        out.at[i,"V55_Qualified_Evidence_Detail_JSON"]=json.dumps(evidence_details[:20],sort_keys=True)
        for family in VALID_BRAIN_FAMILIES:
            if family in support:
                out.at[i,family]="QUALIFIED_SUPPORT"
            elif family in opposition:
                out.at[i,family]="QUALIFIED_INTERNAL_CONFLICT_VETO" if family in conflicts else "QUALIFIED_OPPOSE"
        report["qualified_events_used"]+=len(independent_support)+len(independent_oppose)
        out.at[i,"Confirm_Count"]=len(independent_support-independent_oppose)
        out.at[i,"Oppose_Count"]=len(independent_oppose)
        out.at[i,"V55_Independent_Support_Keys"]='|'.join(sorted(independent_support-independent_oppose))
        out.at[i,"V55_Independent_Conflict_Keys"]='|'.join(sorted(independent_oppose))
        out.at[i,"V55_Active_System_Names"]=' | '.join(sorted({
            str(v) for v in relevant['Signal_Family'].tolist() if str(v).strip()}))[:700]
        out.at[i,"Confirm_Families"]="|".join(sorted(support)) if support else None
        out.at[i,"Oppose_Families"]="|".join(sorted(opposition)) if opposition else None
        if independent_oppose:
            report["conflict_games"]+=1
            out.at[i,"V55_Research_Decision"]="RESEARCH_CONFLICT__VETO"
        elif len(independent_support)>=2:
            report["multi_family"]+=1
            out.at[i,"V55_Research_Decision"]="MULTI_FAMILY_RESEARCH_SUPPORT__NO_BET_AUTHORITY"
    out["V55_Bet_Authority"]=0
    out["V55_Bet_Decision"]="RESEARCH_ONLY_NO_AUTHORITY"
    return out,report


# ----- Explicit V5.5 betting-selection promotion: owner-directed, provisional -----

def _v55_integer_margin_win_push_loss(mu, sigma, line):
    """Continuity-corrected discrete football margin approximation.

    Computes probability of winning, pushing and losing the selected spread.
    Retrospective residual scale is not a calibrated probability distribution.
    Integer-margin push mass is modeled explicitly rather than treated as loss.
    """
    if not all(math.isfinite(float(v)) for v in (mu,sigma,line)) or sigma<=0:
        raise ValueError("invalid projected margin distribution")
    # Only integer and half-point US football spreads are supported.
    if abs(2*line-round(2*line))>1e-7:
        raise ValueError("unsupported fractional point spread")
    target=-float(line)
    if abs(target-round(target))<1e-8:
        high=target+0.5
        low=target-0.5
        p_win=1-_normal_cdf((high-mu)/sigma)
        p_lose=_normal_cdf((low-mu)/sigma)
        p_push=max(0.0,1-p_win-p_lose)
    else:
        boundary=math.ceil(target)-0.5
        p_win=1-_normal_cdf((boundary-mu)/sigma)
        p_push=0.0
        p_lose=1-p_win
    return p_win,p_push,p_lose


def _v55_bet_key(row):
    """Stable across 30m repricing: same game, market and bet side."""
    key="|".join(("NCAAF","V55","SPREADS",
        str(row.get("Game_ID") or "").strip(),
        str(row.get("Game_Start_UTC") or "").strip(),
        _norm_team(row.get("Selected_Team"))))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:28]


def apply_v55_owner_betting_policy(frame, *, now=None, enabled=None):
    """V5.5 CORE + independently qualified systems, prior NCAAF decision ladder.

    CORE-only edges become CANDIDATE, one qualified independent supporter BET,
    two different supporting evidence families STRONG BET. Every verified
    opposition vetoes. Never describe uncalibrated probabilities as OOS.
    Set NCAAF_V55_BET_AUTHORITY_MODE=OFF as an immediate kill switch.
    """
    import pandas as pd
    out=frame.copy()
    # Explicit default follows the current owner's instruction; scheduler runs
    # the policy at the next market refresh, with no new dropdown/job.
    mode=str(os.getenv("NCAAF_V55_BET_AUTHORITY_MODE","USER_DIRECTED_PROVISIONAL")).upper().strip()
    active=(mode=="USER_DIRECTED_PROVISIONAL") if enabled is None else bool(enabled)
    now=_timestamp(now if now is not None else datetime.now(timezone.utc))
    if pd.isna(now): raise ValueError("INVALID_POLICY_TIME")
    new_fields=("V55_Authority_State","V55_Authority_Policy_Id","V55_Bet_Key",
        "V55_Policy_Win_Prob","V55_Policy_Push_Prob","V55_Policy_Lose_Prob",
        "V55_Policy_EV","V55_Policy_Edge","V55_Policy_Model_Edge_Points",
        "V55_Policy_Stake_Units","V55_Policy_Quote_Age_Min","V55_Authority_Hold_Reason")
    for k in new_fields: out[k]=None
    out["V55_Bet_Decision"]="PASS"
    out["V55_Bet_Authority"]=0
    status={"v55_authority_policy_id":V55_POLICY_ID,
        "v55_authority_policy_status":V55_POLICY_STATUS if active else "DISABLED_BY_OPERATOR",
        "v55_authority_probability_basis":"UNVALIDATED_CFBD_V55_NORMAL_MARGIN",
        "v55_authorized_bet_rows":0,"v55_strong_bet_rows":0,
        "v55_candidate_rows":0,"v55_authority_evaluable":0,"v55_bet_authority":0,
        "v55_max_bets_per_snapshot":V55_POLICY_MAX_BETS,
        "v55_policy_min_ev":V55_POLICY_MIN_EV,
        "v55_policy_min_margin_points":V55_POLICY_MIN_MARGIN_POINTS,
        "v55_automatic_wager_execution":False,"v55_v1_selector_used":False,
        "v55_policy_status":"NO_AUTHORIZED_BETS",
        "v55_decision_ladder":"CORE_CANDIDATE__ONE_SUPPORT_BET__TWO_SUPPORT_STRONG",
        "v55_research_historical_seasons":"2013-2025_PRE_2026_ONLY"}
    if not active:
        out["V55_Bet_Decision"]="AUTHORITY_DISABLED"
        return out,status
    candidates=[]
    for i,r in out.iterrows():
        if not str(r.get("Quote_State") or "").startswith("FRESH_TWO_SIDED_"):
            out.at[i,"V55_Authority_Hold_Reason"]="NO_VERIFIED_TWO_SIDED_QUOTE"
            out.at[i,"V55_Bet_Decision"]="RESEARCH_ONLY_NO_AUTHORITY"
            continue
        if str(r.get("CORE_Model"))!="NONLINEAR_HGB":
            out.at[i,"V55_Authority_Hold_Reason"]="UNVERIFIED_MODEL"
            continue
        if str(r.get("V55_Research_Selected_Side") or "") not in ("HOME","AWAY"):
            out.at[i,"V55_Authority_Hold_Reason"]="NO_PROVISIONAL_RESEARCH_SIDE"
            continue
        kick=_timestamp(r.get("Game_Start_UTC"))
        qt=_timestamp(r.get("Backend_Quote_Timestamp"))
        if pd.isna(kick) or pd.isna(qt) or not now<kick<=now+pd.Timedelta(days=7) or not qt<=now or qt>=kick:
            out.at[i,"V55_Authority_Hold_Reason"]="INVALID_OR_OUT_OF_HORIZON_KICKOFF_QUOTE"
            continue
        quote_age=(now-qt).total_seconds()/60.0
        if quote_age>V55_POLICY_MAX_QUOTE_AGE_MINUTES:
            out.at[i,"V55_Authority_Hold_Reason"]="QUOTE_OLDER_THAN_POLICY_LIMIT"
            continue
        if _price(r.get("CORE_Feature_Coverage")) is None or _price(r.get("CORE_Feature_Coverage"))<0.95:
            out.at[i,"V55_Authority_Hold_Reason"]="INSUFFICIENT_CORE_FEATURE_COVERAGE"
            continue
        if _price(r.get("Oppose_Count")) is not None and _price(r.get("Oppose_Count"))>0:
            out.at[i,"V55_Authority_Hold_Reason"]="VERIFIED_INDEPENDENT_CONFLICT_VETO"
            out.at[i,"V55_Bet_Decision"]="VETO"
            continue
        if (not str(r.get("Game_ID") or "").strip()
                or not str(r.get("Selected_Team") or "").strip()
                or not str(r.get("Backend_Quote_Book") or "").strip()):
            out.at[i,"V55_Authority_Hold_Reason"]="UNPRICED_OR_UNIDENTIFIED_SIDE"
            continue
        spread=_price(r.get("Current_Spread"))
        american=_price(r.get("Odds_American"))
        mu=_price(r.get("CORE_Projected_Home_Margin"))
        sigma=_price(r.get("CORE_Residual_SD"))
        if any(v is None for v in (spread,american,mu,sigma)) or not 8<=sigma<=30 or not _valid_american(american):
            out.at[i,"V55_Authority_Hold_Reason"]="INVALID_DISTRIBUTION_OR_EXECUTABLE_PRICE"
            continue
        if not -125<=american<=130:
            out.at[i,"V55_Authority_Hold_Reason"]="PRICE_OUTSIDE_POLICY_BAND"
            continue
        side=str(r.get("V55_Research_Selected_Side"))
        side_mu=mu if side=="HOME" else -mu
        model_points=side_mu+spread
        try: pw,pp,pl=_v55_integer_margin_win_push_loss(side_mu,sigma,spread)
        except (ValueError,OverflowError):
            out.at[i,"V55_Authority_Hold_Reason"]="UNSUPPORTED_SPREAD_TICK"
            continue
        net=_american_net(american)
        if net is None: continue
        ev=pw*net-pl
        conditional=pw/(pw+pl) if pw+pl>0 else 0
        edge=conditional-1/(1+net)
        out.at[i,"V55_Policy_Win_Prob"]=pw
        out.at[i,"V55_Policy_Push_Prob"]=pp
        out.at[i,"V55_Policy_Lose_Prob"]=pl
        out.at[i,"V55_Policy_EV"]=ev
        out.at[i,"V55_Policy_Edge"]=edge
        out.at[i,"V55_Policy_Model_Edge_Points"]=model_points
        out.at[i,"V55_Policy_Quote_Age_Min"]=round(quote_age,1)
        status["v55_authority_evaluable"]+=1
        if (ev<V55_POLICY_MIN_EV or model_points<V55_POLICY_MIN_MARGIN_POINTS
                or conditional<V55_POLICY_MIN_CONDITIONAL_PROB):
            out.at[i,"V55_Authority_Hold_Reason"]="BELOW_CONSERVATIVE_PROVISIONAL_BET_THRESHOLDS"
            out.at[i,"V55_Bet_Decision"]="CANDIDATE" if ev>0.08 and model_points>=3 else "PASS"
            if out.at[i,"V55_Bet_Decision"]=="CANDIDATE":status["v55_candidate_rows"]+=1
            continue
        # Original NCAAF ladder: CORE candidate is NOT automatically a BET.
        # One genuinely independent historically qualified supporter is needed
        # for BET, and two independent supporting evidence families for STRONG.
        verified=int(_price(r.get('Confirm_Count')) or 0)
        if verified<1:
            out.at[i,'V55_Bet_Decision']='CANDIDATE'
            out.at[i,'V55_Authority_Hold_Reason']='CORE_ONLY__NEEDS_QUALIFIED_INDEPENDENT_CONFIRMATION'
            status['v55_candidate_rows']+=1
            continue
        # Eligible rows are ranked before applying a deterministic cap. Never
        # turn a stale or non-verifiable quote into a formal recommendation.
        candidates.append((float(ev),float(model_points),str(r.get("Game_ID")),i))
    candidates.sort(key=lambda x:(-x[0],-x[1],x[2]))
    for rank,(_,_,_,i) in enumerate(candidates):
        if rank>=V55_POLICY_MAX_BETS:
            out.at[i,"V55_Bet_Decision"]="CANDIDATE"
            out.at[i,"V55_Authority_Hold_Reason"]="MAX_THREE_MODEL_BETS_PER_SNAPSHOT"
            status["v55_candidate_rows"]+=1
            continue
        r=out.loc[i]
        qualified=int(_price(r.get("Confirm_Count")) or 0)
        decision="STRONG BET" if qualified>=2 else "BET"
        out.at[i,"V55_Bet_Decision"]=decision
        out.at[i,"V55_Bet_Authority"]=1
        out.at[i,"V55_Authority_State"]=V55_POLICY_STATUS
        out.at[i,"V55_Authority_Policy_Id"]=V55_POLICY_ID
        out.at[i,"V55_Bet_Key"]=_v55_bet_key(r)
        out.at[i,"V55_Policy_Stake_Units"]=0.5 if qualified>=2 else 0.25
        out.at[i,"V55_Authority_Hold_Reason"]=None
        status["v55_authorized_bet_rows"]+=1
        if qualified>=2: status["v55_strong_bet_rows"]+=1
    status["v55_bet_authority"]=1 if status["v55_authorized_bet_rows"] else 0
    status["v55_policy_status"]="USER_DIRECTED_PROVISIONAL_BETS_PUBLISHED" if status["v55_authorized_bet_rows"] else "NO_BETS_PASSED_POLICY"
    return out,status
