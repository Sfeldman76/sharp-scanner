"""NFL V1.9.5 live market microstructure capture.

This is a new-information research path: timestamped pregame book quotes are
canonicalized into an append-only stream and fixed as-of states.  No historical
closing line is treated as an executable predictor.  No pre-clock source quote is
admitted to the V1.9.5 prospective evidence set.
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

from nfl_prospective_ledger_v4 import (
    SOURCE_TAG as LEDGER_SOURCE_TAG, LEDGER_VERSION, RESEARCH_CLOCK_ID,
    FROZEN_V194_REGISTRY_SHA256, FROZEN_V194_EDGE_GATE_SHA256,
)

SOURCE_TAG = "nfl-market-shadow-v1.9.5-prospective-microstructure-20261001"
MARKET_SOURCE = "sharplogger.sharp_data.sharp_moves_master"
SCORE_SOURCE = "sharplogger.sharp_data.game_scores_final"
PRODUCTION_AUTHORITY = 0
MIN_CONSENSUS_BOOKS = 2

SHARP_BOOKS = {
    "pinnacle","betus","mybookieag","smarkets","betfair_ex_eu","betfair_ex_uk","betfair_ex_au",
    "lowvig","betonlineag","matchbook","sport888",
}
REC_BOOKS = {
    "betmgm","bet365","draftkings","fanduel","betrivers","fanatics","espnbet","hardrockbet",
    "williamhillus","ballybet","bet365_au","betopenly",
}
BOOK_ALIASES = {
    "pinny":"pinnacle","pinn":"pinnacle","betonline":"betonlineag","mybookie":"mybookieag",
    "dk":"draftkings","fd":"fanduel","mgm":"betmgm","hardrock":"hardrockbet",
    "sports888":"sport888","888sport":"sport888","888-sport":"sport888","888 sport":"sport888",
    "betfair_exchange":"betfair","betfair-exchange":"betfair","betfair exchange":"betfair","betfair_ex":"betfair",
}

ALIASES = {
    "sport":("Sport",),
    "game_start":("Game_Start","Commence_Hour","feat_Game_Start"),
    "snapshot_timestamp":("Snapshot_Timestamp","snapshot_timestamp","Observed_At","Captured_At"),
    "market":("Market",),
    "outcome":("Outcome","Team","Side"),
    "bookmaker":("Bookmaker","Book","Sportsbook"),
    "value":("Value","Line","Spread","Total"),
    "odds_price":("Odds_Price","Price","Odds"),
    "home_team":("Home_Team","Home_Team_Norm","Home"),
    "away_team":("Away_Team","Away_Team_Norm","Away"),
    "source_game_key":("Game_Key","Merge_Key_Short","Event_ID","Game_ID"),
}


def _norm_text(x):
    if x is None: return ""
    try:
        if bool(pd.isna(x)): return ""
    except Exception:
        pass
    return re.sub(r"\s+"," ",str(x).strip()).strip()


def _name(x):
    s=_norm_text(x).casefold()
    return re.sub(r"\s+"," ",re.sub(r"[^\w\s]","",s)).strip()


def _book(x):
    s=_norm_text(x).lower().replace(" ","")
    return BOOK_ALIASES.get(s,s)


def _market(x):
    s=_norm_text(x).lower()
    if s in {"spread","spreads","ats"}: return "spreads"
    if s in {"total","totals","ou","o/u"}: return "totals"
    if s in {"h2h","moneyline","money_line","ml","headtohead"}: return "h2h"
    return s


def _flt(x):
    try:
        z=float(x); return z if math.isfinite(z) else np.nan
    except Exception: return np.nan


def _implied_american(x):
    z=_flt(x)
    if not np.isfinite(z) or z==0: return np.nan
    return 100.0/(z+100.0) if z>0 else (-z)/((-z)+100.0)


def physical_game_id(game_start, home, away):
    dt=pd.to_datetime(game_start,utc=True,errors="coerce")
    h,a=_name(home),_name(away)
    if pd.isna(dt) or not h or not a or h==a: return ""
    return hashlib.sha256(f"NFL|{dt.isoformat()}|{h}|{a}".encode()).hexdigest()[:24]


def _resolve_schema(client, table=MARKET_SOURCE):
    cols={f.name for f in client.get_table(table).schema}
    out={}
    for key,cands in ALIASES.items():
        out[key]=next((c for c in cands if c in cols),None)
    required=("sport","game_start","snapshot_timestamp","market","outcome","bookmaker")
    missing=[k for k in required if not out.get(k)]
    if missing:
        raise RuntimeError(f"NFL_V1_9_5_MARKET_SOURCE_MISSING_FIELDS {missing} resolved={out}")
    if not out.get("value") and not out.get("odds_price"):
        raise RuntimeError("NFL_V1_9_5_MARKET_SOURCE_NO_VALUE_OR_PRICE")
    if not out.get("home_team") or not out.get("away_team"):
        raise RuntimeError(f"NFL_V1_9_5_MARKET_SOURCE_MISSING_HOME_AWAY resolved={out}")
    return out


def fetch_market_rows(client, *, clock_start, now=None, lookahead_days=8, table=MARKET_SOURCE):
    """Fetch only quotes that are both post-clock and pre-kickoff."""
    now=pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now,utc=True)
    clock_start=pd.to_datetime(clock_start,utc=True)
    m=_resolve_schema(client,table)
    def expr(k,alias,cast=None):
        c=m.get(k)
        if not c: return f"NULL AS `{alias}`"
        raw=f"`{c}`"
        if cast: raw=f"SAFE_CAST({raw} AS {cast})"
        return f"{raw} AS `{alias}`"
    selected=[
        expr("sport","Sport"),expr("game_start","Game_Start","TIMESTAMP"),expr("snapshot_timestamp","Snapshot_Timestamp","TIMESTAMP"),
        expr("market","Market"),expr("outcome","Outcome"),expr("bookmaker","Bookmaker"),expr("value","Value","FLOAT64"),
        expr("odds_price","Odds_Price","FLOAT64"),expr("home_team","Home_Team"),expr("away_team","Away_Team"),
        expr("source_game_key","Source_Game_Key"),
    ]
    q=f"""
      SELECT {', '.join(selected)}
      FROM `{table}`
      WHERE UPPER(TRIM(CAST(`{m['sport']}` AS STRING)))='NFL'
        AND SAFE_CAST(`{m['snapshot_timestamp']}` AS TIMESTAMP) >= @clock_start
        AND SAFE_CAST(`{m['snapshot_timestamp']}` AS TIMESTAMP) <= @now
        AND SAFE_CAST(`{m['game_start']}` AS TIMESTAMP) > @now
        AND SAFE_CAST(`{m['game_start']}` AS TIMESTAMP) <= TIMESTAMP_ADD(@now, INTERVAL {int(lookahead_days)} DAY)
        AND SAFE_CAST(`{m['snapshot_timestamp']}` AS TIMESTAMP) < SAFE_CAST(`{m['game_start']}` AS TIMESTAMP)
      ORDER BY Game_Start, Snapshot_Timestamp
    """
    cfg=b.QueryJobConfig(query_parameters=[
        b.ScalarQueryParameter("clock_start","TIMESTAMP",clock_start.to_pydatetime()),
        b.ScalarQueryParameter("now","TIMESTAMP",now.to_pydatetime()),
    ])
    d=client.query(q,job_config=cfg).to_dataframe(create_bqstorage_client=False)
    return d,m


def canonicalize_quotes(rows: pd.DataFrame, *, clock_start, captured_at=None):
    if rows is None or rows.empty:
        return pd.DataFrame(),{"status":"NO_SOURCE_ROWS","source_rows":0,"eligible_rows":0}
    clock=pd.to_datetime(clock_start,utc=True)
    cap=pd.Timestamp.now(tz="UTC") if captured_at is None else pd.to_datetime(captured_at,utc=True)
    d=rows.copy()
    d["game_start"]=pd.to_datetime(d.get("Game_Start"),utc=True,errors="coerce")
    d["snapshot_timestamp"]=pd.to_datetime(d.get("Snapshot_Timestamp"),utc=True,errors="coerce")
    d["home_team"]=d.get("Home_Team","").map(_norm_text)
    d["away_team"]=d.get("Away_Team","").map(_norm_text)
    d["market"]=d.get("Market","").map(_market)
    d["outcome"]=d.get("Outcome","").map(_norm_text)
    d["bookmaker"]=d.get("Bookmaker","").map(_book)
    d["value"]=pd.to_numeric(d.get("Value"),errors="coerce")
    d["odds_price"]=pd.to_numeric(d.get("Odds_Price"),errors="coerce")
    d["source_game_key"]=d.get("Source_Game_Key","").map(_norm_text)
    d["physical_game_id"]=[physical_game_id(gs,h,a) for gs,h,a in zip(d.game_start,d.home_team,d.away_team)]
    good=(d.snapshot_timestamp.notna() & d.game_start.notna() & d.snapshot_timestamp.ge(clock) & d.snapshot_timestamp.lt(d.game_start)
          & d.game_start.gt(cap) & d.market.isin(["spreads","totals","h2h"]) & d.physical_game_id.ne("")
          & d.outcome.ne("") & d.bookmaker.ne(""))
    d=d.loc[good].copy()
    if d.empty:
        return pd.DataFrame(),{"status":"NO_POSTCLOCK_PREGAME_QUOTES","source_rows":int(len(rows)),"eligible_rows":0}
    d["is_sharp_book"]=d.bookmaker.isin(SHARP_BOOKS)
    d["is_recreational_book"]=d.bookmaker.isin(REC_BOOKS)
    d["captured_at"]=cap
    d["research_clock_id"]=RESEARCH_CLOCK_ID
    d["ledger_version"]=LEDGER_VERSION
    d["production_authority"]=0
    payload_keys=["physical_game_id","market","outcome","bookmaker","value","odds_price","snapshot_timestamp","game_start"]
    def digest(r):
        payload={k:(r[k].isoformat() if isinstance(r[k],pd.Timestamp) else (None if pd.isna(r[k]) else r[k])) for k in payload_keys}
        return hashlib.sha256(json.dumps(payload,sort_keys=True,default=str,separators=(",",":")).encode()).hexdigest()
    d["source_payload_sha256"]=[digest(r) for _,r in d.iterrows()]
    d["quote_event_id"]=[hashlib.sha256(f"{RESEARCH_CLOCK_ID}|{h}".encode()).hexdigest() for h in d.source_payload_sha256]
    cols=["quote_event_id","research_clock_id","ledger_version","captured_at","physical_game_id","source_game_key","game_start","home_team","away_team","market","outcome","bookmaker","value","odds_price","snapshot_timestamp","is_sharp_book","is_recreational_book","source_payload_sha256","production_authority"]
    d=d[cols].drop_duplicates("quote_event_id",keep="last").sort_values(["game_start","market","outcome","snapshot_timestamp","bookmaker"])
    return d.reset_index(drop=True),{"status":"READY","source_rows":int(len(rows)),"eligible_rows":int(len(d)),"games":int(d.physical_game_id.nunique())}


def _metric(frame: pd.DataFrame):
    market=str(frame.market.iloc[0])
    if market=="h2h": return frame.odds_price.map(_implied_american)
    return pd.to_numeric(frame.value,errors="coerce")


def _latest_per_book(g: pd.DataFrame, cutoff):
    q=g.loc[g.snapshot_timestamp.le(cutoff)].sort_values(["bookmaker","snapshot_timestamp"],kind="mergesort")
    if q.empty: return q
    return q.groupby("bookmaker",as_index=False,sort=False).tail(1)


def _consensus(g: pd.DataFrame, cutoff, *, book_subset=None):
    q=_latest_per_book(g,cutoff)
    if book_subset is not None: q=q.loc[q.bookmaker.isin(book_subset)]
    if q.empty: return {"n":0,"value":np.nan,"odds":np.nan,"prob":np.nan,"std":np.nan,"iqr":np.nan,"odds_std":np.nan,"median_age":np.nan,"max_age":np.nan}
    x=_metric(q); x=pd.to_numeric(x,errors="coerce")
    op=pd.to_numeric(q.odds_price,errors="coerce")
    prob=op.map(_implied_american)
    finite=x[np.isfinite(x)]
    def iqr(v):
        v=np.asarray(v,float); v=v[np.isfinite(v)]
        return float(np.quantile(v,.75)-np.quantile(v,.25)) if len(v) else np.nan
    ages=(pd.Timestamp(cutoff)-q.snapshot_timestamp).dt.total_seconds()/60.0
    return {
        "n":int(q.bookmaker.nunique()),
        "value":float(np.nanmedian(finite)) if len(finite) else np.nan,
        "odds":float(np.nanmedian(op)) if np.isfinite(op).any() else np.nan,
        "prob":float(np.nanmedian(prob)) if np.isfinite(prob).any() else np.nan,
        "std":float(np.nanstd(finite,ddof=0)) if len(finite) else np.nan,
        "iqr":iqr(finite),
        "odds_std":float(np.nanstd(op[np.isfinite(op)],ddof=0)) if np.isfinite(op).any() else np.nan,
        "median_age":float(np.nanmedian(ages)) if np.isfinite(ages).any() else np.nan,
        "max_age":float(np.nanmax(ages)) if np.isfinite(ages).any() else np.nan,
    }


def _first_eligible_cutoff(g: pd.DataFrame):
    seen=set()
    for ts,p in g.sort_values("snapshot_timestamp").groupby("snapshot_timestamp",sort=True):
        seen.update(p.bookmaker.astype(str))
        if len(seen)>=MIN_CONSENSUS_BOOKS: return pd.Timestamp(ts)
    return pd.Timestamp(g.snapshot_timestamp.min())


def _move(g, cutoff, mins, *, books=None):
    a=_consensus(g,cutoff,book_subset=books)
    b=_consensus(g,pd.Timestamp(cutoff)-pd.Timedelta(minutes=mins),book_subset=books)
    if np.isfinite(a["value"]) and np.isfinite(b["value"]): return float(a["value"]-b["value"])
    return np.nan


def _price_move(g, cutoff, mins):
    a=_consensus(g,cutoff); b=_consensus(g,pd.Timestamp(cutoff)-pd.Timedelta(minutes=mins))
    if np.isfinite(a["prob"]) and np.isfinite(b["prob"]): return float(a["prob"]-b["prob"])
    return np.nan


def _cross(v0,v1,key):
    if not (np.isfinite(v0) and np.isfinite(v1)): return False
    for k in (-float(key),float(key)):
        if (v0-k)*(v1-k)<=0 and abs(v1-v0)>1e-12 and not (v0==v1==k): return True
    return False


def build_market_states(quotes: pd.DataFrame, *, clock_start, now=None):
    if quotes is None or quotes.empty: return pd.DataFrame(),{"status":"NO_QUOTES","states":0}
    now=pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now,utc=True)
    clock=pd.to_datetime(clock_start,utc=True)
    rows=[]
    for (gid,market,outcome),g in quotes.groupby(["physical_game_id","market","outcome"],sort=False):
        g=g.sort_values("snapshot_timestamp").copy()
        gs=pd.to_datetime(g.game_start.iloc[-1],utc=True)
        if not (now<gs): continue
        first_cut=_first_eligible_cutoff(g)
        targets=[("FIRST_ELIGIBLE",first_cut,False)]
        for mins,canonical in ((120,False),(60,True),(30,False)):
            t=gs-pd.Timedelta(minutes=mins)
            if t>=clock and now>=t:
                targets.append((f"T_MINUS_{mins}",t,canonical))
        latest_cut=min(now,gs-pd.Timedelta(microseconds=1))
        targets.append(("CURRENT_STREAM",latest_cut,False))
        first=_consensus(g,first_cut)
        for stype,target,canonical in targets:
            cur=_consensus(g,target)
            if cur["n"]<=0: continue
            latest=_latest_per_book(g,target)
            cutoff=pd.Timestamp(latest.snapshot_timestamp.max())
            sh30=_move(g,target,30,books=SHARP_BOOKS); sh60=_move(g,target,60,books=SHARP_BOOKS)
            sf30=_move(g,target,30,books=REC_BOOKS); sf60=_move(g,target,60,books=REC_BOOKS)
            m30=_move(g,target,30); m60=_move(g,target,60); m120=_move(g,target,120)
            past60=_consensus(g,target-pd.Timedelta(minutes=60))
            disp_change=(past60["std"]-cur["std"]) if np.isfinite(past60["std"]) and np.isfinite(cur["std"]) else np.nan
            quality="MULTI_BOOK" if cur["n"]>=MIN_CONSENSUS_BOOKS else "SINGLE_BOOK"
            if np.isfinite(cur["max_age"]) and cur["max_age"]>180: quality+="_STALE"
            payload={
                "source_tag":SOURCE_TAG,"books":sorted(latest.bookmaker.astype(str).unique()),
                "sharp_books":sorted(latest.loc[latest.is_sharp_book,"bookmaker"].astype(str).unique()),
                "rec_books":sorted(latest.loc[latest.is_recreational_book,"bookmaker"].astype(str).unique()),
                "clock_start":clock.isoformat(),"target":pd.Timestamp(target).isoformat(),
            }
            eid_basis=cutoff if stype=="CURRENT_STREAM" else pd.Timestamp(target)
            eid=hashlib.sha256(f"{RESEARCH_CLOCK_ID}|{gid}|{market}|{outcome}|{stype}|{pd.Timestamp(eid_basis).isoformat()}".encode()).hexdigest()
            direction=0 if not np.isfinite(m60) or abs(m60)<1e-12 else (1 if m60>0 else -1)
            v60=past60["value"]
            row={
                "state_event_id":eid,"research_clock_id":RESEARCH_CLOCK_ID,"ledger_version":LEDGER_VERSION,
                "frozen_registry_sha256":FROZEN_V194_REGISTRY_SHA256,"edge_gate_registry_sha256":FROZEN_V194_EDGE_GATE_SHA256,
                "captured_at":now,"physical_game_id":gid,"source_game_key":str(g.source_game_key.dropna().iloc[-1]) if g.source_game_key.notna().any() else "",
                "game_start":gs,"home_team":str(g.home_team.iloc[-1]),"away_team":str(g.away_team.iloc[-1]),"market":market,"outcome":outcome,
                "snapshot_type":stype,"target_asof":pd.Timestamp(target),"quote_cutoff_timestamp":cutoff,"canonical_evaluation":bool(canonical),
                "consensus_value":cur["value"],"consensus_odds":cur["odds"],"consensus_implied_probability":cur["prob"],
                "book_count":int(cur["n"]),"sharp_book_count":int(latest.is_sharp_book.sum()),"recreational_book_count":int(latest.is_recreational_book.sum()),
                "distinct_snapshot_times":int(g.loc[g.snapshot_timestamp.le(target),"snapshot_timestamp"].nunique()),
                "timed_span_minutes":float((cutoff-g.snapshot_timestamp.min()).total_seconds()/60.0),
                "median_quote_age_minutes":cur["median_age"],"max_quote_age_minutes":cur["max_age"],
                "book_value_std":cur["std"],"book_value_iqr":cur["iqr"],"odds_price_std":cur["odds_std"],
                "line_move_from_first":(cur["value"]-first["value"]) if np.isfinite(cur["value"]) and np.isfinite(first["value"]) else np.nan,
                "line_move_30m":m30,"line_move_60m":m60,"line_move_120m":m120,
                "line_velocity_30m":m30/30.0 if np.isfinite(m30) else np.nan,"line_velocity_60m":m60/60.0 if np.isfinite(m60) else np.nan,
                "price_move_30m":_price_move(g,target,30),"price_move_60m":_price_move(g,target,60),
                "sharp_move_30m":sh30,"sharp_move_60m":sh60,"soft_move_30m":sf30,"soft_move_60m":sf60,
                "sharp_soft_move_divergence_30m":(sh30-sf30) if np.isfinite(sh30) and np.isfinite(sf30) else np.nan,
                "sharp_soft_move_divergence_60m":(sh60-sf60) if np.isfinite(sh60) and np.isfinite(sf60) else np.nan,
                "book_dispersion_change_60m":disp_change,
                "crossed_key_3_last60m":bool(market=="spreads" and _cross(v60,cur["value"],3)),
                "crossed_key_7_last60m":bool(market=="spreads" and _cross(v60,cur["value"],7)),
                "crossed_key_10_last60m":bool(market=="spreads" and _cross(v60,cur["value"],10)),
                "crossed_key_14_last60m":bool(market=="spreads" and _cross(v60,cur["value"],14)),
                "market_direction_60m":int(direction),"data_quality_status":quality,
                "microstructure_json":json.dumps(payload,sort_keys=True,separators=(",",":")),"production_authority":0,
            }
            rows.append(row)
    out=pd.DataFrame(rows)
    if not out.empty: out=out.drop_duplicates("state_event_id",keep="last")
    return out,{"status":"READY" if len(out) else "NO_STATES","states":int(len(out)),"games":int(out.physical_game_id.nunique()) if len(out) else 0,
                "canonical_t60_states":int(((out.snapshot_type=="T_MINUS_60") & out.canonical_evaluation).sum()) if len(out) else 0}


def fetch_final_scores(client, *, since=None):
    cols={f.name for f in client.get_table(SCORE_SOURCE).schema}
    required=["Sport","Game_Start","Home_Team","Away_Team","Score_Home_Score","Score_Away_Score"]
    miss=[c for c in required if c not in cols]
    if miss: raise RuntimeError("NFL_V1_9_5_SCORE_SOURCE_MISSING "+str(miss))
    where=["UPPER(TRIM(CAST(Sport AS STRING)))='NFL'","Score_Home_Score IS NOT NULL","Score_Away_Score IS NOT NULL"]
    params=[]
    if since is not None:
        where.append("TIMESTAMP(Game_Start)>=@since")
        params.append(b.ScalarQueryParameter("since","TIMESTAMP",pd.to_datetime(since,utc=True).to_pydatetime()))
    q=f"SELECT Game_Start,Home_Team,Away_Team,SAFE_CAST(Score_Home_Score AS FLOAT64) home_score,SAFE_CAST(Score_Away_Score AS FLOAT64) away_score FROM `{SCORE_SOURCE}` WHERE {' AND '.join(where)}"
    d=client.query(q,job_config=b.QueryJobConfig(query_parameters=params)).to_dataframe(create_bqstorage_client=False)
    if d.empty: return d
    d["game_start"]=pd.to_datetime(d.Game_Start,utc=True,errors="coerce")
    d["physical_game_id"]=[physical_game_id(gs,h,a) for gs,h,a in zip(d.game_start,d.Home_Team,d.Away_Team)]
    return d.drop_duplicates("physical_game_id",keep="last")


def _same_team(outcome, team): return _name(outcome)==_name(team)


def settle_states(states: pd.DataFrame, scores: pd.DataFrame, *, settled_at=None):
    if states is None or states.empty or scores is None or scores.empty: return pd.DataFrame()
    d=states.merge(scores[["physical_game_id","home_score","away_score"]],on="physical_game_id",how="inner",validate="many_to_one")
    when=pd.Timestamp.now(tz="UTC") if settled_at is None else pd.to_datetime(settled_at,utc=True)
    out=[]
    for _,r in d.iterrows():
        hs=_flt(r.home_score); aws=_flt(r.away_score); val=_flt(r.consensus_value)
        if not np.isfinite(hs) or not np.isfinite(aws): continue
        margin=hs-aws; total=hs+aws; result="UNAVAILABLE"; err=np.nan
        m=str(r.market); oc=str(r.outcome)
        if m=="spreads" and np.isfinite(val):
            if _same_team(oc,r.home_team): z=margin+val; fair=-val; err=abs(margin-fair)
            elif _same_team(oc,r.away_team): z=-margin+val; fair=val; err=abs(margin-fair)
            else: z=np.nan
            if np.isfinite(z): result="WIN" if z>0 else "LOSS" if z<0 else "PUSH"
        elif m=="totals" and np.isfinite(val):
            s=oc.lower(); z=(total-val) if s.startswith("o") else (val-total) if s.startswith("u") else np.nan
            if np.isfinite(z): result="WIN" if z>0 else "LOSS" if z<0 else "PUSH"; err=abs(total-val)
        elif m=="h2h":
            if margin==0: result="PUSH"
            elif _same_team(oc,r.home_team): result="WIN" if margin>0 else "LOSS"
            elif _same_team(oc,r.away_team): result="WIN" if margin<0 else "LOSS"
        rid=hashlib.sha256(f"{r.state_event_id}|FINAL|{hs}|{aws}".encode()).hexdigest()
        out.append({
            "state_result_event_id":rid,"state_event_id":str(r.state_event_id),"research_clock_id":RESEARCH_CLOCK_ID,
            "physical_game_id":str(r.physical_game_id),"market":m,"outcome":oc,"snapshot_type":str(r.snapshot_type),"settled_at":when,
            "home_score":float(hs),"away_score":float(aws),"actual_margin":float(margin),"actual_total":float(total),
            "market_result":result,"market_absolute_error":float(err) if np.isfinite(err) else np.nan,"production_authority":0,
        })
    return pd.DataFrame(out)
