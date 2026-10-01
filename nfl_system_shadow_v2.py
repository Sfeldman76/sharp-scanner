"""NFL Research V2.2 prospective tracker for newly consolidated mechanism families.

This module does not mine, tune, or promote systems. It reads the immutable
mechanism-family registry produced by nfl_system_lab_v3, creates one append-only
clock per supported NEW family, records future pregame triggers, and settles
those triggers after games finish. The pre-existing ROLE_FLIP family remains on
its original V1 clock and is tracked by nfl_system_shadow_v1.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd
from google.cloud import bigquery as b
from google.cloud import storage

import nfl_market_shadow_v1 as market
import nfl_system_shadow_v1 as roleflip_shadow
import nfl_prospective_ledger_v4 as ledger

SOURCE_TAG = "nfl-system-shadow-v2.2-mechanism-family-tracker-20261001"
PRODUCTION_AUTHORITY = 0
POINTER_KEY = "nfl-research/v2_0/system_lab/latest_family_registry_pointer_v3.json"
PROJECT_ID = ledger.PROJECT_ID
DATASET_ID = ledger.DATASET_ID
CLOCK_TABLE = f"{PROJECT_ID}.{DATASET_ID}.nfl_system_family_clock_v2"
TRIGGER_TABLE = f"{PROJECT_ID}.{DATASET_ID}.nfl_system_family_trigger_v2"
RESULT_TABLE = f"{PROJECT_ID}.{DATASET_ID}.nfl_system_family_trigger_results_v2"
MARKET_SOURCE = market.MARKET_SOURCE

HOME_FAV_FAMILY = "NFL_SPREAD_HOME_FAVORITE_OFF_SU_LOSS_FADE"
EARLY_DIV_UNDER_FAMILY = "NFL_TOTAL_EARLY_DIVISION_UNDER"
SUPPORTED_EVALUATORS = {
    "SPREAD_HOME_FAVORITE_OFF_SU_LOSS_FADE_V1",
    "TOTAL_DIVISION_EARLY_WEEK_UNDER_V1",
}

_DIVISION = {
    "BUF":"AFC_EAST","MIA":"AFC_EAST","NE":"AFC_EAST","NYJ":"AFC_EAST",
    "BAL":"AFC_NORTH","CIN":"AFC_NORTH","CLE":"AFC_NORTH","PIT":"AFC_NORTH",
    "HOU":"AFC_SOUTH","IND":"AFC_SOUTH","JAX":"AFC_SOUTH","TEN":"AFC_SOUTH",
    "DEN":"AFC_WEST","KC":"AFC_WEST","LV":"AFC_WEST","LAC":"AFC_WEST",
    "DAL":"NFC_EAST","NYG":"NFC_EAST","PHI":"NFC_EAST","WAS":"NFC_EAST",
    "CHI":"NFC_NORTH","DET":"NFC_NORTH","GB":"NFC_NORTH","MIN":"NFC_NORTH",
    "ATL":"NFC_SOUTH","CAR":"NFC_SOUTH","NO":"NFC_SOUTH","TB":"NFC_SOUTH",
    "ARI":"NFC_WEST","LAR":"NFC_WEST","SF":"NFC_WEST","SEA":"NFC_WEST",
}


def _num(x):
    try:
        z=float(x)
        return z if math.isfinite(z) else np.nan
    except Exception:
        return np.nan


def _txt(x):
    if x is None:return ""
    try:
        if bool(pd.isna(x)):return ""
    except Exception:pass
    return re.sub(r"\s+"," ",str(x).strip())


def _family_hash_payload(reg:dict)->str:
    d=dict(reg); d.pop("family_registry_sha256",None)
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def verify_family_registry(storage_client:storage.Client,bucket_name="sharp-models"):
    bucket=storage_client.bucket(bucket_name)
    pblob=bucket.blob(POINTER_KEY)
    if not pblob.exists():
        raise RuntimeError("NFL_SYSTEM_FAMILY_V2_POINTER_MISSING_RUN_SYSTEM_LAB_V3_FIRST")
    ptr=json.loads(pblob.download_as_bytes().decode("utf-8"))
    uri=str(ptr.get("family_registry_uri") or "")
    sha=str(ptr.get("family_registry_sha256") or "")
    prefix=f"gs://{bucket_name}/"
    if not uri.startswith(prefix) or not sha:
        raise RuntimeError("NFL_SYSTEM_FAMILY_V2_POINTER_INVALID")
    blob=bucket.blob(uri[len(prefix):])
    if not blob.exists():raise RuntimeError("NFL_SYSTEM_FAMILY_V2_REGISTRY_MISSING "+uri)
    reg=json.loads(blob.download_as_bytes().decode("utf-8"))
    if str(reg.get("family_registry_sha256") or "")!=sha or _family_hash_payload(reg)!=sha:
        raise RuntimeError("NFL_SYSTEM_FAMILY_V2_REGISTRY_SHA_MISMATCH")
    fams=[]
    for f in reg.get("families") or []:
        if f.get("prospective_action")!="START_NEW_CLOCK":continue
        if f.get("prospective_evaluator_id") not in SUPPORTED_EVALUATORS:continue
        fams.append(dict(f))
    return {"status":"PASS","pointer_uri":f"gs://{bucket_name}/{POINTER_KEY}","registry_uri":uri,"registry_sha256":sha,"families":fams,"production_authority":0}


def _schema_clock():
    S=b.SchemaField
    return [
        S("system_clock_id","STRING",mode="REQUIRED"),S("activated_at","TIMESTAMP"),S("source_tag","STRING"),
        S("system_family_id","STRING"),S("market","STRING"),S("direction","STRING"),S("representative_rule_id","STRING"),
        S("family_registry_sha256","STRING"),S("family_registry_uri","STRING"),S("evaluator_id","STRING"),S("note","STRING"),S("production_authority","INT64"),
    ]


def _schema_trigger():
    S=b.SchemaField
    return [
        S("trigger_event_id","STRING",mode="REQUIRED"),S("system_clock_id","STRING"),S("source_tag","STRING"),S("system_family_id","STRING"),
        S("family_registry_sha256","STRING"),S("captured_at","TIMESTAMP"),S("physical_game_id","STRING",mode="REQUIRED"),S("game_start","TIMESTAMP"),S("season","INT64"),
        S("market","STRING"),S("direction","STRING"),S("home_team","STRING"),S("away_team","STRING"),S("target_team","STRING"),S("play_team","STRING"),
        S("opening_spread","FLOAT64"),S("opening_total","FLOAT64"),S("week_number","INT64"),S("is_division_game","BOOL"),S("prev_actual_margin","FLOAT64"),
        S("representative_rule_id","STRING"),S("active_variant_ids_json","STRING"),S("context_json","STRING"),S("data_quality_status","STRING"),S("production_authority","INT64"),
    ]


def _schema_result():
    S=b.SchemaField
    return [
        S("trigger_result_event_id","STRING",mode="REQUIRED"),S("trigger_event_id","STRING",mode="REQUIRED"),S("system_clock_id","STRING"),S("system_family_id","STRING"),
        S("physical_game_id","STRING"),S("settled_at","TIMESTAMP"),S("market","STRING"),S("direction","STRING"),S("home_score","FLOAT64"),S("away_score","FLOAT64"),
        S("opening_spread","FLOAT64"),S("opening_total","FLOAT64"),S("graded_result","STRING"),S("hit","FLOAT64"),S("production_authority","INT64"),
    ]


def _ensure_table(client,table_id,schema,partition=None,cluster=None):
    from google.api_core.exceptions import NotFound
    try:
        t=client.get_table(table_id); existing={f.name for f in t.schema}; missing=[f for f in schema if f.name not in existing]
        if missing:
            bad=[f.name for f in missing if f.mode=="REQUIRED"]
            if bad:raise RuntimeError(f"NFL_SYSTEM_FAMILY_V2_REQUIRED_SCHEMA_MIGRATION table={table_id} fields={bad}")
            t.schema=list(t.schema)+missing; client.update_table(t,["schema"])
        return False,[f.name for f in missing]
    except NotFound:
        t=b.Table(table_id,schema=schema); t.description="Append-only NFL mechanism-family prospective evidence. Zero production authority."
        if partition:t.time_partitioning=b.TimePartitioning(type_=b.TimePartitioningType.DAY,field=partition)
        if cluster:t.clustering_fields=list(cluster)
        client.create_table(t); return True,[]


def ensure_tables(client):
    made=[]; added={}
    for table,schema,part,cluster in (
        (CLOCK_TABLE,_schema_clock(),"activated_at",["system_family_id"]),
        (TRIGGER_TABLE,_schema_trigger(),"captured_at",["physical_game_id","system_family_id"]),
        (RESULT_TABLE,_schema_result(),"settled_at",["physical_game_id","system_family_id"]),
    ):
        c,m=_ensure_table(client,table,schema,part,cluster)
        if c:made.append(table)
        if m:added[table]=m
    return {"status":"READY","created":made,"fields_added":added,"production_authority":0}


def _clock_id(family_id,registry_sha):
    slug=re.sub(r"[^A-Z0-9]+","_",family_id.upper()).strip("_")[:72]
    return f"NFL_SYS_FAM_{slug}_{registry_sha[:10]}"


def get_or_create_clock(client,family,registry,now=None):
    ensure_tables(client); now=pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now,utc=True)
    cid=_clock_id(family["system_family_id"],registry["registry_sha256"])
    cfg=b.QueryJobConfig(query_parameters=[b.ScalarQueryParameter("cid","STRING",cid)])
    q=f"SELECT * FROM `{CLOCK_TABLE}` WHERE system_clock_id=@cid ORDER BY activated_at LIMIT 1"
    d=client.query(q,job_config=cfg).to_dataframe(create_bqstorage_client=False)
    if not d.empty:
        r=d.iloc[0].to_dict(); r["activated_at"]=pd.to_datetime(r.get("activated_at"),utc=True,errors="coerce"); r["created_now"]=False; return r
    row={
        "system_clock_id":cid,"activated_at":now,"source_tag":SOURCE_TAG,"system_family_id":family["system_family_id"],"market":family.get("market"),"direction":family.get("direction"),
        "representative_rule_id":family.get("representative_rule_id"),"family_registry_sha256":registry["registry_sha256"],"family_registry_uri":registry["registry_uri"],
        "evaluator_id":family.get("prospective_evaluator_id"),"note":"Clock begins only after mechanism-family registry freeze; no earlier result is prospective evidence.","production_authority":0,
    }
    errs=client.insert_rows_json(CLOCK_TABLE,_serialize(pd.DataFrame([row]),("activated_at",)))
    if errs:raise RuntimeError("NFL_SYSTEM_FAMILY_V2_CLOCK_INSERT_FAILED "+str(errs[:3]))
    return {**row,"created_now":True}


def _resolve_total_open_schema(client,table=MARKET_SOURCE):
    cols={f.name for f in client.get_table(table).schema}
    def first(*names):return next((x for x in names if x in cols),None)
    out={
        "sport":first("Sport"),"game_start":first("Game_Start","Commence_Hour","feat_Game_Start"),"snapshot":first("Snapshot_Timestamp","snapshot_timestamp","Observed_At","Captured_At"),
        "market":first("Market"),"book":first("Bookmaker","Book","Sportsbook"),"home":first("Home_Team","Home_Team_Norm","Home"),"away":first("Away_Team","Away_Team_Norm","Away"),
        "opening_total":first("Opening_Total","Consensus_Open_Total","First_Line_Value","Open_Value","Opening_Line"),"week":first("Week_Number","Week","NFL_Week"),
    }
    need=[k for k in ("sport","game_start","snapshot","market","home","away","opening_total") if not out.get(k)]
    if need:return out,"HOLD_MISSING_TRUE_OPENING_TOTAL_FIELDS:"+",".join(need)
    if not out.get("week"):return out,"HOLD_WEEK_NUMBER_UNAVAILABLE"
    return out,"READY"


def fetch_total_openings(client,clock_start,now=None,lookahead_days=8,table=MARKET_SOURCE):
    now=pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now,utc=True); clock=pd.to_datetime(clock_start,utc=True)
    m,status=_resolve_total_open_schema(client,table)
    if status!="READY":return pd.DataFrame(),{"status":status,"resolved":m}
    book=f"CAST(`{m['book']}` AS STRING)" if m.get("book") else "''"
    q=f"""
      SELECT SAFE_CAST(`{m['game_start']}` AS TIMESTAMP) Game_Start, SAFE_CAST(`{m['snapshot']}` AS TIMESTAMP) Snapshot_Timestamp,
             CAST(`{m['market']}` AS STRING) Market, {book} Bookmaker, CAST(`{m['home']}` AS STRING) Home_Team, CAST(`{m['away']}` AS STRING) Away_Team,
             SAFE_CAST(`{m['opening_total']}` AS FLOAT64) Opening_Total, SAFE_CAST(`{m['week']}` AS INT64) Week_Number
      FROM `{table}`
      WHERE UPPER(TRIM(CAST(`{m['sport']}` AS STRING)))='NFL'
        AND LOWER(TRIM(CAST(`{m['market']}` AS STRING))) IN ('total','totals','ou','o/u')
        AND SAFE_CAST(`{m['snapshot']}` AS TIMESTAMP)>=@clock_start AND SAFE_CAST(`{m['snapshot']}` AS TIMESTAMP)<=@now
        AND SAFE_CAST(`{m['game_start']}` AS TIMESTAMP)>@now AND SAFE_CAST(`{m['game_start']}` AS TIMESTAMP)<=TIMESTAMP_ADD(@now,INTERVAL {int(lookahead_days)} DAY)
        AND SAFE_CAST(`{m['snapshot']}` AS TIMESTAMP)<SAFE_CAST(`{m['game_start']}` AS TIMESTAMP)
        AND SAFE_CAST(`{m['opening_total']}` AS FLOAT64) IS NOT NULL AND SAFE_CAST(`{m['week']}` AS INT64) IS NOT NULL
    """
    cfg=b.QueryJobConfig(query_parameters=[b.ScalarQueryParameter("clock_start","TIMESTAMP",clock.to_pydatetime()),b.ScalarQueryParameter("now","TIMESTAMP",now.to_pydatetime())])
    raw=client.query(q,job_config=cfg).to_dataframe(create_bqstorage_client=False)
    if raw.empty:return pd.DataFrame(),{"status":"NO_POSTCLOCK_TOTAL_OPENINGS","source_rows":0,"resolved":m}
    d=raw.copy(); d["game_start"]=pd.to_datetime(d.Game_Start,utc=True,errors="coerce"); d["snapshot_timestamp"]=pd.to_datetime(d.Snapshot_Timestamp,utc=True,errors="coerce")
    d["home_team"]=d.Home_Team.map(_txt); d["away_team"]=d.Away_Team.map(_txt); d["home_key"]=d.home_team.map(roleflip_shadow._norm_team); d["away_key"]=d.away_team.map(roleflip_shadow._norm_team)
    d["physical_game_id"]=[market.physical_game_id(gs,h,a) for gs,h,a in zip(d.game_start,d.home_team,d.away_team)]; d["opening_total"]=pd.to_numeric(d.Opening_Total,errors="coerce"); d["week_number"]=pd.to_numeric(d.Week_Number,errors="coerce")
    d=d.loc[d.physical_game_id.ne("")&d.opening_total.notna()&d.week_number.notna()].copy()
    rows=[]
    for gid,g in d.groupby("physical_game_id",sort=False):
        # latest row per book, then median true opening total across books
        g=g.sort_values(["Bookmaker","snapshot_timestamp"],kind="mergesort").groupby("Bookmaker",dropna=False,as_index=False).tail(1)
        vals=pd.to_numeric(g.opening_total,errors="coerce").dropna().to_numpy(float); r=g.sort_values("snapshot_timestamp").iloc[-1]
        if not len(vals):continue
        rows.append({"physical_game_id":gid,"game_start":r.game_start,"home_team":r.home_team,"away_team":r.away_team,"home_key":r.home_key,"away_key":r.away_key,"opening_total":float(np.median(vals)),"week_number":int(r.week_number),"opening_book_count":int(g.Bookmaker.replace("",np.nan).nunique())})
    out=pd.DataFrame(rows)
    return out,{"status":"READY" if len(out) else "NO_TOTAL_OPENINGS","source_rows":int(len(raw)),"games":int(len(out)),"opening_total_column":m["opening_total"],"week_column":m["week"]}


def _active_variants(family):
    # Only the representative definition is evaluated prospectively here.
    # Specialist variants remain diagnostic metadata in the immutable registry.
    rid=family.get("representative_rule_id")
    return [rid] if rid else []

def _season_from_ts(x):
    t=pd.to_datetime(x,utc=True,errors="coerce")
    if pd.isna(t):return 0
    return int(t.year-1 if int(t.month)<=3 else t.year)


def derive_home_favorite_fade(openings,history,family,clock,registry_sha,captured_at=None):
    cap=pd.Timestamp.now(tz="UTC") if captured_at is None else pd.to_datetime(captured_at,utc=True)
    if openings is None or openings.empty or history is None or history.empty:return pd.DataFrame(),{"status":"NO_INPUT","qualifying_games":0}
    rows=[]; skip={}
    for _,g in openings.iterrows():
        hh=history.loc[history.team_key.eq(str(g.home_key))].sort_values(["Game_Date","Source_Name","Source_Game_ID"],kind="mergesort")
        if hh.empty:skip["MISSING_HOME_HISTORY"]=skip.get("MISSING_HOME_HISTORY",0)+1;continue
        op=_num(g.home_opening_spread); prev=_num(hh.iloc[-1].actual_margin)
        if not np.isfinite(op) or not np.isfinite(prev):skip["MISSING_STATE"]=skip.get("MISSING_STATE",0)+1;continue
        if not (op<0 and prev<0):continue
        eid=hashlib.sha256(f"{clock['system_clock_id']}|{family['system_family_id']}|{g.physical_game_id}".encode()).hexdigest()
        rows.append({
            "trigger_event_id":eid,"system_clock_id":clock["system_clock_id"],"source_tag":SOURCE_TAG,"system_family_id":family["system_family_id"],"family_registry_sha256":registry_sha,"captured_at":cap,
            "physical_game_id":str(g.physical_game_id),"game_start":pd.to_datetime(g.game_start,utc=True),"season":_season_from_ts(g.game_start),"market":"SPREADS","direction":"FADE",
            "home_team":str(g.home_team),"away_team":str(g.away_team),"target_team":str(g.home_team),"play_team":str(g.away_team),"opening_spread":op,"opening_total":np.nan,"week_number":None,"is_division_game":bool(_DIVISION.get(str(g.home_key))==_DIVISION.get(str(g.away_key))),"prev_actual_margin":prev,
            "representative_rule_id":family.get("representative_rule_id"),"active_variant_ids_json":json.dumps(_active_variants(family),separators=(",",":")),"context_json":json.dumps({"definition":"home opening favorite off SU loss; fade home team ATS","family_status":family.get("family_status")},sort_keys=True,separators=(",",":")),"data_quality_status":"READY" if _num(g.opening_spread_range)<=1 else "OPENING_BOOK_DISPERSION_GT_1","production_authority":0,
        })
    return pd.DataFrame(rows),{"status":"READY" if rows else "NO_QUALIFIERS","scanned_games":int(len(openings)),"qualifying_games":len(rows),"skip_reasons":skip}


def derive_early_division_under(openings,family,clock,registry_sha,captured_at=None):
    cap=pd.Timestamp.now(tz="UTC") if captured_at is None else pd.to_datetime(captured_at,utc=True)
    if openings is None or openings.empty:return pd.DataFrame(),{"status":"NO_INPUT","qualifying_games":0}
    rows=[]
    for _,g in openings.iterrows():
        hk,ak=str(g.home_key),str(g.away_key); wk=int(g.week_number); div=bool(_DIVISION.get(hk) and _DIVISION.get(hk)==_DIVISION.get(ak)); ot=_num(g.opening_total)
        if not (div and 1<=wk<=4 and np.isfinite(ot)):continue
        eid=hashlib.sha256(f"{clock['system_clock_id']}|{family['system_family_id']}|{g.physical_game_id}".encode()).hexdigest()
        rows.append({
            "trigger_event_id":eid,"system_clock_id":clock["system_clock_id"],"source_tag":SOURCE_TAG,"system_family_id":family["system_family_id"],"family_registry_sha256":registry_sha,"captured_at":cap,
            "physical_game_id":str(g.physical_game_id),"game_start":pd.to_datetime(g.game_start,utc=True),"season":_season_from_ts(g.game_start),"market":"TOTALS","direction":"UNDER",
            "home_team":str(g.home_team),"away_team":str(g.away_team),"target_team":"","play_team":"UNDER","opening_spread":np.nan,"opening_total":ot,"week_number":wk,"is_division_game":True,"prev_actual_margin":np.nan,
            "representative_rule_id":family.get("representative_rule_id"),"active_variant_ids_json":json.dumps(_active_variants(family),separators=(",",":")),"context_json":json.dumps({"definition":"division game in NFL Weeks 1-4; play UNDER opening total","family_status":family.get("family_status")},sort_keys=True,separators=(",",":")),"data_quality_status":"READY","production_authority":0,
        })
    return pd.DataFrame(rows),{"status":"READY" if rows else "NO_QUALIFIERS","scanned_games":int(len(openings)),"qualifying_games":len(rows)}


def _serialize(df,time_cols:Iterable[str]):
    out=[]
    for r in df.where(pd.notna(df),None).to_dict("records"):
        for c in time_cols:
            if isinstance(r.get(c),pd.Timestamp):r[c]=r[c].isoformat()
        out.append(r)
    return out


def _append_idempotent(client,df,table,id_col,time_cols=()):
    if df is None or df.empty:return {"status":"NO_ROWS","inserted":0,"existing":0}
    ids=[str(x) for x in df[id_col].astype(str)]; cfg=b.QueryJobConfig(query_parameters=[b.ArrayQueryParameter("ids","STRING",ids)])
    got=client.query(f"SELECT `{id_col}` FROM `{table}` WHERE `{id_col}` IN UNNEST(@ids)",job_config=cfg).to_dataframe(create_bqstorage_client=False); existing=set(got[id_col].astype(str)) if not got.empty else set()
    add=df.loc[~df[id_col].astype(str).isin(existing)].copy()
    if add.empty:return {"status":"ALREADY_PRESENT","inserted":0,"existing":len(existing)}
    errs=client.insert_rows_json(table,_serialize(add,time_cols))
    if errs:raise RuntimeError(f"NFL_SYSTEM_FAMILY_V2_INSERT_FAILED table={table} errors={errs[:3]}")
    return {"status":"INSERTED","inserted":int(len(add)),"existing":len(existing)}


def _pending(client):
    return client.query(f"SELECT t.* FROM `{TRIGGER_TABLE}` t LEFT JOIN `{RESULT_TABLE}` r USING(trigger_event_id) WHERE r.trigger_event_id IS NULL AND t.game_start<CURRENT_TIMESTAMP()").to_dataframe(create_bqstorage_client=False)


def settle_triggers(pending,scores,settled_at=None):
    if pending is None or pending.empty or scores is None or scores.empty:return pd.DataFrame()
    when=pd.Timestamp.now(tz="UTC") if settled_at is None else pd.to_datetime(settled_at,utc=True); smap=scores.drop_duplicates("physical_game_id",keep="last").set_index("physical_game_id"); rows=[]
    for _,r in pending.iterrows():
        gid=str(r.physical_game_id)
        if gid not in smap.index:continue
        s=smap.loc[gid]; hs=_num(s.home_score); aws=_num(s.away_score)
        if not np.isfinite(hs) or not np.isfinite(aws):continue
        market_name=str(r.market).upper(); direction=str(r.direction).upper(); result="PUSH"; hit=np.nan
        if market_name=="SPREADS":
            op=_num(r.opening_spread)
            if not np.isfinite(op):continue
            target_margin=hs-aws; ats=target_margin+op
            if direction=="FADE":result="WIN" if ats<0 else "LOSS" if ats>0 else "PUSH"
            else:result="WIN" if ats>0 else "LOSS" if ats<0 else "PUSH"
        elif market_name=="TOTALS":
            ot=_num(r.opening_total)
            if not np.isfinite(ot):continue
            delta=(hs+aws)-ot
            if direction=="UNDER":result="WIN" if delta<0 else "LOSS" if delta>0 else "PUSH"
            else:result="WIN" if delta>0 else "LOSS" if delta<0 else "PUSH"
        if result=="WIN":hit=1.0
        elif result=="LOSS":hit=0.0
        rid=hashlib.sha256(f"{r.trigger_event_id}|SETTLED".encode()).hexdigest()
        rows.append({"trigger_result_event_id":rid,"trigger_event_id":str(r.trigger_event_id),"system_clock_id":str(r.system_clock_id),"system_family_id":str(r.system_family_id),"physical_game_id":gid,"settled_at":when,"market":market_name,"direction":direction,"home_score":hs,"away_score":aws,"opening_spread":_num(r.opening_spread),"opening_total":_num(r.opening_total),"graded_result":result,"hit":hit,"production_authority":0})
    return pd.DataFrame(rows)


def run_new_mechanism_family_shadow(*,bq_client,storage_client,bucket_name="sharp-models",now=None,lookahead_days=8,log_func=print):
    c=bq_client; now=pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now,utc=True); tables=ensure_tables(c); reg=verify_family_registry(storage_client,bucket_name)
    histories,season=roleflip_shadow.fetch_completed_history(c); family_summaries=[]; all_triggers=[]
    for fam in reg["families"]:
        clock=get_or_create_clock(c,fam,reg,now=now); start=pd.to_datetime(clock["activated_at"],utc=True); evaluator=fam.get("prospective_evaluator_id")
        if evaluator=="SPREAD_HOME_FAVORITE_OFF_SU_LOSS_FADE_V1":
            openings,meta=roleflip_shadow.fetch_current_openings(c,clock_start=start,now=now,lookahead_days=lookahead_days)
            tr,tmeta=derive_home_favorite_fade(openings,histories,fam,clock,reg["registry_sha256"],captured_at=now)
        elif evaluator=="TOTAL_DIVISION_EARLY_WEEK_UNDER_V1":
            openings,meta=fetch_total_openings(c,start,now=now,lookahead_days=lookahead_days)
            tr,tmeta=derive_early_division_under(openings,fam,clock,reg["registry_sha256"],captured_at=now) if meta.get("status")=="READY" else (pd.DataFrame(),{"status":meta.get("status"),"qualifying_games":0})
        else:
            continue
        if not tr.empty:all_triggers.append(tr)
        family_summaries.append({"system_family_id":fam["system_family_id"],"clock_id":clock["system_clock_id"],"clock_activated_at":start.isoformat(),"clock_created_now":bool(clock.get("created_now",False)),"evaluator_id":evaluator,"opening_meta":meta,"trigger_meta":tmeta})
    trigger_df=pd.concat(all_triggers,ignore_index=True) if all_triggers else pd.DataFrame(); twrite=_append_idempotent(c,trigger_df,TRIGGER_TABLE,"trigger_event_id",("captured_at","game_start"))
    pending=_pending(c)
    if pending.empty:settled=pd.DataFrame();rwrite={"status":"NO_ROWS","inserted":0,"existing":0}
    else:
        earliest=min([pd.to_datetime(x["clock_activated_at"],utc=True) for x in family_summaries],default=now)
        scores=market.fetch_final_scores(c,since=earliest-pd.Timedelta(days=1)); settled=settle_triggers(pending,scores,settled_at=now); rwrite=_append_idempotent(c,settled,RESULT_TABLE,"trigger_result_event_id",("settled_at",))
    summary={"status":"NFL_RESEARCH_V2_MECHANISM_FAMILY_PROSPECTIVE_TRACKER_ACTIVE","source_tag":SOURCE_TAG,"family_registry_sha256":reg["registry_sha256"],"family_registry_uri":reg["registry_uri"],"families":family_summaries,"triggers_found_this_run":int(len(trigger_df)),"triggers_inserted":int(twrite.get("inserted",0)),"results_inserted":int(rwrite.get("inserted",0)),"trigger_write":twrite,"result_write":rwrite,"tables":tables,"automatic_promotion":False,"production_authority":0}
    log_func("[NFL-RESEARCH-V2-SYSTEM-FAMILY-SHADOW-V2] "+json.dumps(summary,sort_keys=True,default=str)); return summary


def self_test():
    family={"system_family_id":HOME_FAV_FAMILY,"family_status":"LEGIT_FAMILY_REQUIRES_PROSPECTIVE","representative_rule_id":"OFF_SU_LOSS__HOME__OPEN_FAVORITE","specialist_rule_ids":["OFF_SU_LOSS__HOME__OPEN_FAVORITE"]}
    clock={"system_clock_id":"TEST_CLOCK"}; openings=pd.DataFrame([{"physical_game_id":"g1","game_start":pd.Timestamp("2026-10-04T17:00:00Z"),"home_team":"Baltimore Ravens","away_team":"Cleveland Browns","home_key":"BAL","away_key":"CLE","home_opening_spread":-3.0,"opening_spread_range":0.5}])
    hist=pd.DataFrame([{"team_key":"BAL","Game_Date":pd.Timestamp("2026-09-20"),"Source_Name":"T","Source_Game_ID":"1","actual_margin":-7.0}])
    tr,meta=derive_home_favorite_fade(openings,hist,family,clock,"abc",pd.Timestamp("2026-10-01T20:00:00Z"))
    if len(tr)!=1 or tr.iloc[0].direction!="FADE":raise AssertionError("HOME_FAV_FADE_TRIGGER_TEST_FAILED")
    tfam={"system_family_id":EARLY_DIV_UNDER_FAMILY,"family_status":"LEGIT_FAMILY_REQUIRES_PROSPECTIVE","representative_rule_id":"DIVISION_GAME__EARLY_WK1_4","specialist_rule_ids":["DIVISION_GAME__EARLY_WK1_4"]}
    tops=pd.DataFrame([{"physical_game_id":"g2","game_start":pd.Timestamp("2027-09-12T17:00:00Z"),"home_team":"Baltimore Ravens","away_team":"Cincinnati Bengals","home_key":"BAL","away_key":"CIN","opening_total":45.5,"week_number":2}])
    tt,_=derive_early_division_under(tops,tfam,clock,"abc",pd.Timestamp("2027-09-10T20:00:00Z"))
    if len(tt)!=1 or tt.iloc[0].direction!="UNDER":raise AssertionError("EARLY_DIV_UNDER_TRIGGER_TEST_FAILED")
    scores=pd.DataFrame([{"physical_game_id":"g2","home_score":20.0,"away_score":17.0}]); st=settle_triggers(tt,scores,pd.Timestamp("2027-09-13T00:00:00Z"))
    if len(st)!=1 or st.iloc[0].graded_result!="WIN":raise AssertionError("TOTAL_SETTLEMENT_TEST_FAILED")
    return {"status":"PASS","new_families":2,"production_authority":0}
