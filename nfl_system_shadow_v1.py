"""NFL Research V2 prospective system-family tracker.

Tracks the already-frozen NFL_ROLE_FLIP_FADE family prospectively. This module
never mines, tunes, re-ranks, or changes the frozen system definitions. It only:
  * verifies the immutable System Lab registry,
  * establishes a new post-deployment system clock,
  * detects future qualifying games using pregame opening-line context plus
    completed prior-game history,
  * writes one append-only family trigger per physical game, and
  * settles those triggers after final scores arrive.

The three related variants are recorded as one correlated SYSTEM family, never
as three independent votes. Production authority is always zero.
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
import nfl_prospective_ledger_v4 as ledger

SOURCE_TAG = "nfl-system-shadow-v1.9.5.1-role-flip-family-20261001"
PRODUCTION_AUTHORITY = 0

SYSTEM_CLOCK_ID = "NFL_RESEARCH_V2_ROLE_FLIP_FADE_PROSPECTIVE_CLOCK"
SYSTEM_FAMILY_ID = "NFL_ROLE_FLIP_FADE"
SYSTEM_REGISTRY_SHA256 = "b9fc3d469248f40f580211f5e5decc792976c90596a0b09abd38e279c46cbc81"
SYSTEM_REGISTRY_URI = "gs://sharp-models/nfl-research/v2_0/system_lab/b9fc3d469248f40f/system_registry.json"

# These are exact frozen System Lab rule IDs. Do not change boundaries or combine
# them into a new rule without creating a new research hypothesis and clock.
VARIANT_A = "ROLE_FLIP_DOG_TO_FAVORITE__OFF_SU_LOSS__HOME__NEGATIVE_LAST5_MARGIN"
VARIANT_B = "ROLE_FLIP_DOG_TO_FAVORITE__NEGATIVE_LAST5_MARGIN__HOME__OFF_ATS_LOSS"
VARIANT_C = "ROLE_FLIP_DOG_TO_FAVORITE__NEGATIVE_LAST5_MARGIN__HOME__OFF_ATS_LOSS__OPP_WINPCT_LE_500"
FROZEN_VARIANTS = (VARIANT_A, VARIANT_B, VARIANT_C)

PROJECT_ID = ledger.PROJECT_ID
DATASET_ID = ledger.DATASET_ID
SYSTEM_CLOCK_TABLE = f"{PROJECT_ID}.{DATASET_ID}.nfl_system_research_clock_v1"
SYSTEM_TRIGGER_TABLE = f"{PROJECT_ID}.{DATASET_ID}.nfl_system_trigger_v1"
SYSTEM_RESULT_TABLE = f"{PROJECT_ID}.{DATASET_ID}.nfl_system_trigger_results_v1"
HIST_VIEW = "sharplogger.sharp_data.nfl_historical_core_training_vw"
MARKET_SOURCE = market.MARKET_SOURCE

# Historical/current naming normalization is intentionally franchise-level so a
# display-name difference does not silently drop a system trigger.
_TEAM_ALIASES = {
    "ari":"ARI","arizona":"ARI","arizona cardinals":"ARI","cardinals":"ARI",
    "atl":"ATL","atlanta":"ATL","atlanta falcons":"ATL","falcons":"ATL",
    "bal":"BAL","baltimore":"BAL","baltimore ravens":"BAL","ravens":"BAL",
    "buf":"BUF","buffalo":"BUF","buffalo bills":"BUF","bills":"BUF",
    "car":"CAR","carolina":"CAR","carolina panthers":"CAR","panthers":"CAR",
    "chi":"CHI","chicago":"CHI","chicago bears":"CHI","bears":"CHI",
    "cin":"CIN","cincinnati":"CIN","cincinnati bengals":"CIN","bengals":"CIN",
    "cle":"CLE","cleveland":"CLE","cleveland browns":"CLE","browns":"CLE",
    "dal":"DAL","dallas":"DAL","dallas cowboys":"DAL","cowboys":"DAL",
    "den":"DEN","denver":"DEN","denver broncos":"DEN","broncos":"DEN",
    "det":"DET","detroit":"DET","detroit lions":"DET","lions":"DET",
    "gb":"GB","green bay":"GB","green bay packers":"GB","packers":"GB",
    "hou":"HOU","houston":"HOU","houston texans":"HOU","texans":"HOU",
    "ind":"IND","indianapolis":"IND","indianapolis colts":"IND","colts":"IND",
    "jax":"JAX","jac":"JAX","jacksonville":"JAX","jacksonville jaguars":"JAX","jaguars":"JAX",
    "kc":"KC","kansas city":"KC","kansas city chiefs":"KC","chiefs":"KC",
    "lv":"LV","las vegas":"LV","las vegas raiders":"LV","raiders":"LV","oakland raiders":"LV","oak":"LV",
    "lac":"LAC","la chargers":"LAC","los angeles chargers":"LAC","chargers":"LAC",
    "lar":"LAR","la rams":"LAR","los angeles rams":"LAR","rams":"LAR","st louis rams":"LAR",
    "mia":"MIA","miami":"MIA","miami dolphins":"MIA","dolphins":"MIA",
    "min":"MIN","minnesota":"MIN","minnesota vikings":"MIN","vikings":"MIN",
    "ne":"NE","new england":"NE","new england patriots":"NE","patriots":"NE",
    "no":"NO","new orleans":"NO","new orleans saints":"NO","saints":"NO",
    "nyg":"NYG","new york giants":"NYG","giants":"NYG",
    "nyj":"NYJ","new york jets":"NYJ","jets":"NYJ",
    "phi":"PHI","philadelphia":"PHI","philadelphia eagles":"PHI","eagles":"PHI",
    "pit":"PIT","pittsburgh":"PIT","pittsburgh steelers":"PIT","steelers":"PIT",
    "sea":"SEA","seattle":"SEA","seattle seahawks":"SEA","seahawks":"SEA",
    "sf":"SF","san francisco":"SF","san francisco 49ers":"SF","49ers":"SF","niners":"SF",
    "tb":"TB","tampa bay":"TB","tampa bay buccaneers":"TB","buccaneers":"TB","bucs":"TB",
    "ten":"TEN","tennessee":"TEN","tennessee titans":"TEN","titans":"TEN",
    "was":"WAS","wsh":"WAS","washington":"WAS","washington commanders":"WAS","commanders":"WAS","washington football team":"WAS","washington redskins":"WAS","redskins":"WAS",
}


def _txt(x):
    if x is None:
        return ""
    try:
        if bool(pd.isna(x)):
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(x).strip())


def _norm_team(x):
    s=_txt(x).lower()
    s=re.sub(r"[^a-z0-9]+"," ",s).strip()
    return _TEAM_ALIASES.get(s, s.upper() if s else "")


def _num(x):
    try:
        z=float(x)
        return z if math.isfinite(z) else np.nan
    except Exception:
        return np.nan


def _schema_clock():
    S=b.SchemaField
    return [
        S("system_clock_id","STRING",mode="REQUIRED"),S("activated_at","TIMESTAMP"),S("source_tag","STRING"),
        S("system_family_id","STRING"),S("system_registry_sha256","STRING"),S("system_registry_uri","STRING"),
        S("note","STRING"),S("production_authority","INT64"),
    ]


def _schema_trigger():
    S=b.SchemaField
    return [
        S("trigger_event_id","STRING",mode="REQUIRED"),S("system_clock_id","STRING"),S("source_tag","STRING"),
        S("system_family_id","STRING"),S("system_registry_sha256","STRING"),S("captured_at","TIMESTAMP"),
        S("physical_game_id","STRING",mode="REQUIRED"),S("game_start","TIMESTAMP"),S("season","INT64"),
        S("home_team","STRING"),S("away_team","STRING"),S("target_team","STRING"),S("play_team","STRING"),
        S("target_opening_spread","FLOAT64"),S("opening_spread_source","STRING"),S("opening_book_count","INT64"),
        S("opening_spread_range","FLOAT64"),S("prev_opening_spread","FLOAT64"),S("prev_actual_margin","FLOAT64"),
        S("prev_opening_ats_margin","FLOAT64"),S("last5_avg_su_margin","FLOAT64"),S("opponent_win_pct_prior","FLOAT64"),
        S("role_flip_dog_to_favorite","BOOL"),S("negative_last5_margin","BOOL"),S("off_su_loss","BOOL"),
        S("off_ats_loss","BOOL"),S("opponent_win_pct_le_500","BOOL"),S("variant_a_active","BOOL"),
        S("variant_b_active","BOOL"),S("variant_c_active","BOOL"),S("variant_count","INT64"),
        S("active_variant_ids_json","STRING"),S("context_json","STRING"),S("data_quality_status","STRING"),
        S("production_authority","INT64"),
    ]


def _schema_result():
    S=b.SchemaField
    return [
        S("trigger_result_event_id","STRING",mode="REQUIRED"),S("trigger_event_id","STRING",mode="REQUIRED"),
        S("system_clock_id","STRING"),S("system_family_id","STRING"),S("physical_game_id","STRING"),
        S("settled_at","TIMESTAMP"),S("home_score","FLOAT64"),S("away_score","FLOAT64"),
        S("target_actual_margin","FLOAT64"),S("target_opening_spread","FLOAT64"),S("target_opening_ats_margin","FLOAT64"),
        S("fade_result","STRING"),S("fade_hit","FLOAT64"),S("production_authority","INT64"),
    ]


def _ensure_table(client, table_id, schema, *, partition=None, cluster=None):
    from google.api_core.exceptions import NotFound
    try:
        t=client.get_table(table_id)
        existing={f.name for f in t.schema}
        missing=[f for f in schema if f.name not in existing]
        if missing:
            bad=[f.name for f in missing if f.mode=="REQUIRED"]
            if bad:
                raise RuntimeError(f"NFL_SYSTEM_SHADOW_REQUIRED_SCHEMA_MIGRATION table={table_id} fields={bad}")
            t.schema=list(t.schema)+missing
            client.update_table(t,["schema"])
        return False,[f.name for f in missing]
    except NotFound:
        t=b.Table(table_id,schema=schema)
        t.description="Append-only NFL Research V2 prospective system-family evidence. Zero production authority."
        if partition:
            t.time_partitioning=b.TimePartitioning(type_=b.TimePartitioningType.DAY,field=partition)
        if cluster:
            t.clustering_fields=list(cluster)
        client.create_table(t)
        return True,[]


def ensure_tables(client):
    created=[]; added={}
    for table,schema,part,cluster in (
        (SYSTEM_CLOCK_TABLE,_schema_clock(),"activated_at",["system_family_id"]),
        (SYSTEM_TRIGGER_TABLE,_schema_trigger(),"captured_at",["physical_game_id","system_family_id"]),
        (SYSTEM_RESULT_TABLE,_schema_result(),"settled_at",["physical_game_id","system_family_id"]),
    ):
        made,miss=_ensure_table(client,table,schema,partition=part,cluster=cluster)
        if made: created.append(table)
        if miss: added[table]=miss
    return {"status":"READY","created":created,"fields_added":added,"production_authority":0}


def verify_frozen_registry(storage_client: storage.Client, bucket_name="sharp-models"):
    prefix=f"gs://{bucket_name}/"
    if not SYSTEM_REGISTRY_URI.startswith(prefix):
        raise RuntimeError("NFL_SYSTEM_SHADOW_REGISTRY_BUCKET_MISMATCH")
    key=SYSTEM_REGISTRY_URI[len(prefix):]
    blob=storage_client.bucket(bucket_name).blob(key)
    if not blob.exists():
        raise RuntimeError(f"NFL_SYSTEM_SHADOW_FROZEN_REGISTRY_MISSING {SYSTEM_REGISTRY_URI}")
    reg=json.loads(blob.download_as_bytes().decode("utf-8"))
    if str(reg.get("registry_sha256","")) != SYSTEM_REGISTRY_SHA256:
        raise RuntimeError("NFL_SYSTEM_SHADOW_REGISTRY_SHA_MISMATCH")
    found=set(reg.get("spread_promising_rule_ids") or [])
    missing=[x for x in FROZEN_VARIANTS if x not in found]
    if missing:
        raise RuntimeError("NFL_SYSTEM_SHADOW_FROZEN_VARIANTS_MISSING "+str(missing))
    return {
        "status":"PASS","uri":SYSTEM_REGISTRY_URI,"registry_sha256":SYSTEM_REGISTRY_SHA256,
        "frozen_variant_ids":list(FROZEN_VARIANTS),"production_authority":0,
    }


def get_or_create_system_clock(client, *, now=None):
    ensure_tables(client)
    cfg=b.QueryJobConfig(query_parameters=[b.ScalarQueryParameter("cid","STRING",SYSTEM_CLOCK_ID)])
    q=f"SELECT * FROM `{SYSTEM_CLOCK_TABLE}` WHERE system_clock_id=@cid ORDER BY activated_at LIMIT 1"
    d=client.query(q,job_config=cfg).to_dataframe(create_bqstorage_client=False)
    if not d.empty:
        r=d.iloc[0].to_dict(); r["activated_at"]=pd.to_datetime(r.get("activated_at"),utc=True,errors="coerce"); r["created_now"]=False
        return r
    ts=pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now,utc=True)
    row={
        "system_clock_id":SYSTEM_CLOCK_ID,"activated_at":ts,"source_tag":SOURCE_TAG,
        "system_family_id":SYSTEM_FAMILY_ID,"system_registry_sha256":SYSTEM_REGISTRY_SHA256,
        "system_registry_uri":SYSTEM_REGISTRY_URI,
        "note":"First successful prospective system-tracker run freezes the NFL_ROLE_FLIP_FADE family. No result before this clock is prospective evidence.",
        "production_authority":0,
    }
    errs=client.insert_rows_json(SYSTEM_CLOCK_TABLE,[{**row,"activated_at":ts.isoformat()}])
    if errs:
        d=client.query(q,job_config=cfg).to_dataframe(create_bqstorage_client=False)
        if d.empty:
            raise RuntimeError("NFL_SYSTEM_SHADOW_CLOCK_CREATE_FAILED "+str(errs[:3]))
        r=d.iloc[0].to_dict(); r["activated_at"]=pd.to_datetime(r.get("activated_at"),utc=True,errors="coerce"); r["created_now"]=False
        return r
    return {**row,"created_now":True}


def _resolve_opening_schema(client, table=MARKET_SOURCE):
    cols={f.name for f in client.get_table(table).schema}
    def first(*names): return next((x for x in names if x in cols),None)
    out={
        "sport":first("Sport"),"game_start":first("Game_Start","Commence_Hour","feat_Game_Start"),
        "snapshot":first("Snapshot_Timestamp","snapshot_timestamp","Observed_At","Captured_At"),
        "market":first("Market"),"outcome":first("Outcome","Team","Side"),"book":first("Bookmaker","Book","Sportsbook"),
        "home":first("Home_Team","Home_Team_Norm","Home"),"away":first("Away_Team","Away_Team_Norm","Away"),
        "opening":first("Opening_Spread","First_Line_Value","Consensus_Open_Spread","Open_Value","Opening_Line"),
    }
    miss=[k for k in ("sport","game_start","snapshot","market","outcome","home","away","opening") if not out.get(k)]
    if miss:
        raise RuntimeError(f"NFL_SYSTEM_SHADOW_OPENING_SOURCE_MISSING {miss} resolved={out}")
    return out


def fetch_current_openings(client, *, clock_start, now=None, lookahead_days=8, table=MARKET_SOURCE):
    now=pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now,utc=True)
    clock=pd.to_datetime(clock_start,utc=True)
    m=_resolve_opening_schema(client,table)
    book_expr=f"CAST(`{m['book']}` AS STRING)" if m.get("book") else "''"
    q=f"""
      SELECT
        SAFE_CAST(`{m['game_start']}` AS TIMESTAMP) AS Game_Start,
        SAFE_CAST(`{m['snapshot']}` AS TIMESTAMP) AS Snapshot_Timestamp,
        CAST(`{m['market']}` AS STRING) AS Market,
        CAST(`{m['outcome']}` AS STRING) AS Outcome,
        {book_expr} AS Bookmaker,
        CAST(`{m['home']}` AS STRING) AS Home_Team,
        CAST(`{m['away']}` AS STRING) AS Away_Team,
        SAFE_CAST(`{m['opening']}` AS FLOAT64) AS Opening_Spread
      FROM `{table}`
      WHERE UPPER(TRIM(CAST(`{m['sport']}` AS STRING)))='NFL'
        AND SAFE_CAST(`{m['snapshot']}` AS TIMESTAMP) >= @clock_start
        AND SAFE_CAST(`{m['snapshot']}` AS TIMESTAMP) <= @now
        AND SAFE_CAST(`{m['game_start']}` AS TIMESTAMP) > @now
        AND SAFE_CAST(`{m['game_start']}` AS TIMESTAMP) <= TIMESTAMP_ADD(@now, INTERVAL {int(lookahead_days)} DAY)
        AND SAFE_CAST(`{m['snapshot']}` AS TIMESTAMP) < SAFE_CAST(`{m['game_start']}` AS TIMESTAMP)
        AND SAFE_CAST(`{m['opening']}` AS FLOAT64) IS NOT NULL
    """
    cfg=b.QueryJobConfig(query_parameters=[
        b.ScalarQueryParameter("clock_start","TIMESTAMP",clock.to_pydatetime()),
        b.ScalarQueryParameter("now","TIMESTAMP",now.to_pydatetime()),
    ])
    raw=client.query(q,job_config=cfg).to_dataframe(create_bqstorage_client=False)
    if raw.empty:
        return pd.DataFrame(),{"status":"NO_POSTCLOCK_OPENING_ROWS","source_rows":0,"opening_column":m["opening"]}
    d=raw.copy()
    d["game_start"]=pd.to_datetime(d.Game_Start,utc=True,errors="coerce")
    d["snapshot_timestamp"]=pd.to_datetime(d.Snapshot_Timestamp,utc=True,errors="coerce")
    d["home_team"]=d.Home_Team.map(_txt); d["away_team"]=d.Away_Team.map(_txt); d["outcome"]=d.Outcome.map(_txt)
    d["bookmaker"]=d.Bookmaker.map(_txt); d["opening_spread"]=pd.to_numeric(d.Opening_Spread,errors="coerce")
    d["physical_game_id"]=[market.physical_game_id(gs,h,a) for gs,h,a in zip(d.game_start,d.home_team,d.away_team)]
    d["home_key"]=d.home_team.map(_norm_team); d["outcome_key"]=d.outcome.map(_norm_team)
    d=d.loc[d.home_key.ne("") & d.home_key.eq(d.outcome_key) & d.physical_game_id.ne("") & d.opening_spread.notna()].copy()
    if d.empty:
        return pd.DataFrame(),{"status":"NO_HOME_OUTCOME_OPENING_ROWS","source_rows":int(len(raw)),"opening_column":m["opening"]}
    # Retain one latest source row per book, then use the median across books.
    d=d.sort_values(["physical_game_id","bookmaker","snapshot_timestamp"],kind="mergesort")
    d=d.groupby(["physical_game_id","bookmaker"],as_index=False,dropna=False).tail(1)
    rows=[]
    for gid,g in d.groupby("physical_game_id",sort=False):
        vals=pd.to_numeric(g.opening_spread,errors="coerce").dropna().to_numpy(float)
        if not len(vals): continue
        spread=float(np.median(vals)); rng=float(np.max(vals)-np.min(vals)) if len(vals)>1 else 0.0
        r=g.sort_values("snapshot_timestamp").iloc[-1]
        rows.append({
            "physical_game_id":gid,"game_start":r.game_start,"home_team":r.home_team,"away_team":r.away_team,
            "home_key":r.home_key,"away_key":_norm_team(r.away_team),"home_opening_spread":spread,
            "opening_book_count":int(g.bookmaker.replace("",np.nan).nunique()),"opening_spread_range":rng,
            "opening_spread_source":f"MEDIAN_BOOK_{m['opening']}",
        })
    out=pd.DataFrame(rows)
    return out,{"status":"READY" if len(out) else "NO_OPENINGS","source_rows":int(len(raw)),"games":int(len(out)),"opening_column":m["opening"]}


def fetch_completed_history(client, *, season_hint=None):
    # Current system features only require completed same-season history. The
    # latest available season is selected from the audited historical view.
    if season_hint is None:
        sy=client.query(f"SELECT MAX(Season) AS season FROM `{HIST_VIEW}` WHERE Historical_Core_Eligible=1").to_dataframe(create_bqstorage_client=False)
        season_hint=int(sy.iloc[0].season) if not sy.empty and pd.notna(sy.iloc[0].season) else None
    if season_hint is None:
        return pd.DataFrame(),None
    q=f"""
      SELECT Season, Game_Date, Source_Name, Source_Game_ID, Team_Norm, Opponent_Norm,
             Team_Score, Opponent_Score, Opening_Spread, Season_Stage
      FROM `{HIST_VIEW}`
      WHERE Season=@season AND Historical_Core_Eligible=1
        AND Season_Stage IN ('REGULAR','POSTSEASON')
      ORDER BY Game_Date, Source_Name, Source_Game_ID, Team_Norm
    """
    cfg=b.QueryJobConfig(query_parameters=[b.ScalarQueryParameter("season","INT64",int(season_hint))])
    d=client.query(q,job_config=cfg).to_dataframe(create_bqstorage_client=False)
    if d.empty: return d,int(season_hint)
    d["Game_Date"]=pd.to_datetime(d.Game_Date,errors="coerce")
    d["team_key"]=d.Team_Norm.map(_norm_team); d["opp_key"]=d.Opponent_Norm.map(_norm_team)
    d["actual_margin"]=pd.to_numeric(d.Team_Score,errors="coerce")-pd.to_numeric(d.Opponent_Score,errors="coerce")
    d["opening_spread"]=pd.to_numeric(d.Opening_Spread,errors="coerce")
    d["opening_ats_margin"]=d.actual_margin+d.opening_spread
    d=d.loc[d.team_key.ne("") & d.actual_margin.notna()].copy()
    return d,int(season_hint)


def derive_role_flip_triggers(openings: pd.DataFrame, history: pd.DataFrame, *, season: int, captured_at=None):
    cap=pd.Timestamp.now(tz="UTC") if captured_at is None else pd.to_datetime(captured_at,utc=True)
    if openings is None or openings.empty or history is None or history.empty:
        return pd.DataFrame(),{"status":"NO_INPUT","scanned_games":0,"qualifying_games":0}
    rows=[]; skip={}
    for _,g in openings.iterrows():
        hk,ak=str(g.home_key),str(g.away_key)
        hh=history.loc[history.team_key.eq(hk)].sort_values(["Game_Date","Source_Name","Source_Game_ID"],kind="mergesort")
        ah=history.loc[history.team_key.eq(ak)].sort_values(["Game_Date","Source_Name","Source_Game_ID"],kind="mergesort")
        if hh.empty or ah.empty:
            skip["MISSING_SAME_SEASON_HISTORY"]=skip.get("MISSING_SAME_SEASON_HISTORY",0)+1; continue
        prev=hh.iloc[-1]
        last5=hh.tail(5)
        last5_margin=float(pd.to_numeric(last5.actual_margin,errors="coerce").mean())
        opp_vals=np.where(pd.to_numeric(ah.actual_margin,errors="coerce")>0,1.0,np.where(pd.to_numeric(ah.actual_margin,errors="coerce")<0,0.0,0.5))
        opp_wp=float(np.mean(opp_vals)) if len(opp_vals) else np.nan
        cur_open=_num(g.home_opening_spread); prev_open=_num(prev.opening_spread); prev_margin=_num(prev.actual_margin); prev_ats=_num(prev.opening_ats_margin)
        if not all(np.isfinite(x) for x in (cur_open,prev_open,prev_margin,last5_margin,opp_wp)):
            skip["MISSING_REQUIRED_STATE"]=skip.get("MISSING_REQUIRED_STATE",0)+1; continue
        role_flip=bool(prev_open>0 and cur_open<0)
        neg5=bool(last5_margin<0); off_su=bool(prev_margin<0); off_ats=bool(prev_ats<0); opp_le=bool(opp_wp<=0.5)
        a=role_flip and neg5 and off_su
        bvar=role_flip and neg5 and off_ats
        cvar=bvar and opp_le
        if not (a or bvar or cvar):
            continue
        active=[]
        if a: active.append(VARIANT_A)
        if bvar: active.append(VARIANT_B)
        if cvar: active.append(VARIANT_C)
        quality="READY"
        if _num(g.opening_spread_range)>1.0: quality="OPENING_BOOK_DISPERSION_GT_1"
        context={
            "frozen_family":SYSTEM_FAMILY_ID,"direction":"FADE_HOME_FAVORITE_PLAY_AWAY_ATS",
            "variant_a":"prev opening dog + current home favorite + negative last5 margin + off SU loss",
            "variant_b":"prev opening dog + current home favorite + negative last5 margin + off ATS loss",
            "variant_c":"variant B + opponent pregame win pct <= .500",
            "opening_line_role":"qualification/reference; no executable historical ROI claim",
        }
        eid=hashlib.sha256(f"{SYSTEM_CLOCK_ID}|{SYSTEM_FAMILY_ID}|{g.physical_game_id}".encode()).hexdigest()
        rows.append({
            "trigger_event_id":eid,"system_clock_id":SYSTEM_CLOCK_ID,"source_tag":SOURCE_TAG,
            "system_family_id":SYSTEM_FAMILY_ID,"system_registry_sha256":SYSTEM_REGISTRY_SHA256,"captured_at":cap,
            "physical_game_id":str(g.physical_game_id),"game_start":pd.to_datetime(g.game_start,utc=True),"season":int(season),
            "home_team":str(g.home_team),"away_team":str(g.away_team),"target_team":str(g.home_team),"play_team":str(g.away_team),
            "target_opening_spread":cur_open,"opening_spread_source":str(g.opening_spread_source),
            "opening_book_count":int(g.opening_book_count or 0),"opening_spread_range":_num(g.opening_spread_range),
            "prev_opening_spread":prev_open,"prev_actual_margin":prev_margin,"prev_opening_ats_margin":prev_ats,
            "last5_avg_su_margin":last5_margin,"opponent_win_pct_prior":opp_wp,
            "role_flip_dog_to_favorite":role_flip,"negative_last5_margin":neg5,"off_su_loss":off_su,"off_ats_loss":off_ats,
            "opponent_win_pct_le_500":opp_le,"variant_a_active":a,"variant_b_active":bvar,"variant_c_active":cvar,
            "variant_count":len(active),"active_variant_ids_json":json.dumps(active,separators=(",",":")),
            "context_json":json.dumps(context,sort_keys=True,separators=(",",":")),"data_quality_status":quality,"production_authority":0,
        })
    out=pd.DataFrame(rows)
    return out,{"status":"READY" if len(out) else "NO_QUALIFIERS","scanned_games":int(len(openings)),"qualifying_games":int(len(out)),"skip_reasons":skip}


def _serialize(df:pd.DataFrame,time_cols:Iterable[str]):
    out=[]
    for r in df.where(pd.notna(df),None).to_dict("records"):
        for c in time_cols:
            if isinstance(r.get(c),pd.Timestamp): r[c]=r[c].isoformat()
        out.append(r)
    return out


def _append_idempotent(client, df, table, id_col, *, time_cols=()):
    if df is None or df.empty: return {"status":"NO_ROWS","inserted":0,"existing":0}
    ids=[str(x) for x in df[id_col].astype(str)]
    cfg=b.QueryJobConfig(query_parameters=[b.ArrayQueryParameter("ids","STRING",ids)])
    got=client.query(f"SELECT `{id_col}` FROM `{table}` WHERE `{id_col}` IN UNNEST(@ids)",job_config=cfg).to_dataframe(create_bqstorage_client=False)
    existing=set(got[id_col].astype(str)) if not got.empty else set()
    add=df.loc[~df[id_col].astype(str).isin(existing)].copy()
    if add.empty: return {"status":"ALREADY_PRESENT","inserted":0,"existing":len(existing)}
    errs=client.insert_rows_json(table,_serialize(add,time_cols))
    if errs: raise RuntimeError(f"NFL_SYSTEM_SHADOW_INSERT_FAILED table={table} errors={errs[:3]}")
    return {"status":"INSERTED","inserted":int(len(add)),"existing":len(existing)}


def _pending_triggers(client):
    q=f"""
      SELECT t.* FROM `{SYSTEM_TRIGGER_TABLE}` t
      LEFT JOIN `{SYSTEM_RESULT_TABLE}` r USING(trigger_event_id)
      WHERE r.trigger_event_id IS NULL AND t.game_start < CURRENT_TIMESTAMP()
    """
    return client.query(q).to_dataframe(create_bqstorage_client=False)


def settle_triggers(pending:pd.DataFrame,scores:pd.DataFrame,*,settled_at=None):
    if pending is None or pending.empty or scores is None or scores.empty: return pd.DataFrame()
    settled=pd.Timestamp.now(tz="UTC") if settled_at is None else pd.to_datetime(settled_at,utc=True)
    smap=scores.drop_duplicates("physical_game_id",keep="last").set_index("physical_game_id")
    rows=[]
    for _,r in pending.iterrows():
        gid=str(r.physical_game_id)
        if gid not in smap.index: continue
        s=smap.loc[gid]; hs=_num(s.home_score); aws=_num(s.away_score); op=_num(r.target_opening_spread)
        if not all(np.isfinite(x) for x in (hs,aws,op)): continue
        margin=hs-aws; ats=margin+op
        result="WIN" if ats< -1e-9 else "LOSS" if ats>1e-9 else "PUSH"
        hit=1.0 if result=="WIN" else 0.0 if result=="LOSS" else np.nan
        rid=hashlib.sha256(f"{r.trigger_event_id}|SETTLED".encode()).hexdigest()
        rows.append({
            "trigger_result_event_id":rid,"trigger_event_id":str(r.trigger_event_id),"system_clock_id":SYSTEM_CLOCK_ID,
            "system_family_id":SYSTEM_FAMILY_ID,"physical_game_id":gid,"settled_at":settled,
            "home_score":hs,"away_score":aws,"target_actual_margin":margin,"target_opening_spread":op,
            "target_opening_ats_margin":ats,"fade_result":result,"fade_hit":hit,"production_authority":0,
        })
    return pd.DataFrame(rows)


def run_system_family_shadow(*,bq_client,storage_client,bucket_name="sharp-models",now=None,lookahead_days=8,log_func=print):
    c=bq_client; now=pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now,utc=True)
    table_health=ensure_tables(c); registry=verify_frozen_registry(storage_client,bucket_name); clock=get_or_create_system_clock(c,now=now)
    clock_start=pd.to_datetime(clock["activated_at"],utc=True)
    openings,open_meta=fetch_current_openings(c,clock_start=clock_start,now=now,lookahead_days=lookahead_days)
    hist,season=fetch_completed_history(c)
    triggers,trigger_meta=derive_role_flip_triggers(openings,hist,season=int(season) if season is not None else 0,captured_at=now)
    twrite=_append_idempotent(c,triggers,SYSTEM_TRIGGER_TABLE,"trigger_event_id",time_cols=("captured_at","game_start"))
    pending=_pending_triggers(c)
    if pending.empty:
        settled=pd.DataFrame(); rwrite={"status":"NO_ROWS","inserted":0,"existing":0}
    else:
        scores=market.fetch_final_scores(c,since=clock_start-pd.Timedelta(days=1))
        settled=settle_triggers(pending,scores,settled_at=now)
        rwrite=_append_idempotent(c,settled,SYSTEM_RESULT_TABLE,"trigger_result_event_id",time_cols=("settled_at",))
    summary={
        "status":"NFL_RESEARCH_V2_ROLE_FLIP_FADE_PROSPECTIVE_TRACKER_ACTIVE",
        "source_tag":SOURCE_TAG,"system_clock_id":SYSTEM_CLOCK_ID,"clock_activated_at":clock_start.isoformat(),
        "clock_created_now":bool(clock.get("created_now",False)),"system_family_id":SYSTEM_FAMILY_ID,
        "system_registry_sha256":SYSTEM_REGISTRY_SHA256,"frozen_variant_ids":list(FROZEN_VARIANTS),
        "variant_independence_policy":"ONE_CORRELATED_FAMILY_NOT_THREE_VOTES",
        "opening_meta":open_meta,"history_season":season,"trigger_meta":trigger_meta,
        "triggers_found_this_run":int(len(triggers)),"triggers_inserted":int(twrite.get("inserted",0)),
        "results_inserted":int(rwrite.get("inserted",0)),"trigger_write":twrite,"result_write":rwrite,
        "tables":table_health,"registry_verification":registry,"automatic_promotion":False,"production_authority":0,
    }
    log_func("[NFL-RESEARCH-V2-SYSTEM-SHADOW] "+json.dumps(summary,sort_keys=True,default=str))
    return summary


def self_test():
    hist=[]
    # Home target: prior dog; last five negative; last game SU/ATS loss.
    for i,m in enumerate([-3,-7,2,-4,-6]):
        hist.append({"Season":2026,"Game_Date":pd.Timestamp(f"2026-09-{5+i*4:02d}"),"Source_Name":"T","Source_Game_ID":str(i),
                     "Team_Norm":"Baltimore Ravens","Opponent_Norm":"X","Team_Score":20+m,"Opponent_Score":20,
                     "Opening_Spread":3.0 if i==4 else -1.0,"team_key":"BAL","opp_key":"X","actual_margin":float(m),
                     "opening_spread":3.0 if i==4 else -1.0,"opening_ats_margin":float(m)+(3.0 if i==4 else -1.0)})
    # Opponent .500 prior.
    for i,m in enumerate([3,-2,7,-5]):
        hist.append({"Season":2026,"Game_Date":pd.Timestamp(f"2026-09-{6+i*4:02d}"),"Source_Name":"T","Source_Game_ID":"a"+str(i),
                     "Team_Norm":"Cleveland Browns","Opponent_Norm":"Y","Team_Score":20+m,"Opponent_Score":20,
                     "Opening_Spread":1.0,"team_key":"CLE","opp_key":"Y","actual_margin":float(m),"opening_spread":1.0,"opening_ats_margin":float(m)+1.0})
    openings=pd.DataFrame([{"physical_game_id":"g1","game_start":pd.Timestamp("2026-10-04T17:00:00Z"),"home_team":"Baltimore Ravens","away_team":"Cleveland Browns",
                            "home_key":"BAL","away_key":"CLE","home_opening_spread":-2.5,"opening_book_count":3,"opening_spread_range":0.5,"opening_spread_source":"TEST"}])
    tr,meta=derive_role_flip_triggers(openings,pd.DataFrame(hist),season=2026,captured_at=pd.Timestamp("2026-10-01T18:00:00Z"))
    if len(tr)!=1 or not bool(tr.iloc[0].variant_a_active) or not bool(tr.iloc[0].variant_b_active) or not bool(tr.iloc[0].variant_c_active):
        raise AssertionError("ROLE_FLIP_VARIANT_TEST_FAILED")
    if int(tr.iloc[0].variant_count)!=3: raise AssertionError("CORRELATED_VARIANT_COUNT_FAILED")
    scores=pd.DataFrame([{"physical_game_id":"g1","home_score":17.0,"away_score":20.0}])
    settled=settle_triggers(tr,scores,settled_at=pd.Timestamp("2026-10-05T00:00:00Z"))
    if len(settled)!=1 or settled.iloc[0].fade_result!="WIN": raise AssertionError("SYSTEM_SETTLEMENT_TEST_FAILED")
    return {"status":"PASS","family":SYSTEM_FAMILY_ID,"variants":3,"settlement":"PASS","production_authority":0}
