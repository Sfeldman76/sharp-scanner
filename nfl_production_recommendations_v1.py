"""NFL Production V1.4 recommendation + performance sidecar.

This module is deliberately downstream from the frozen model contract.
It does not train or promote models. Historical recommendation thresholds are
published by nfl_production_replay_v1 using a 2021-2023 discovery / 2024-2025
confirmation split. Live recommendations apply that frozen policy to the
frozen champion's fair values and a current consensus market snapshot.

Recommendation events and settlements are immutable GCS objects so this layer
does not require additional BigQuery write IAM beyond the existing paired model
ledger. System-family triggers are displayed as separate supporting evidence;
they do not increase model vote count or formal recommendation authority.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from google.api_core.exceptions import PreconditionFailed
from google.cloud import bigquery

import nfl_live_feature_parity_v1 as parity

SOURCE_TAG = "nfl-production-v1.4-recommendation-performance-ui-20261002"
PROJECT = "sharplogger"
DATASET = "sharp_data"
MARKET_SOURCE = f"{PROJECT}.{DATASET}.sharp_moves_master"
PAIRED_SETTLEMENT_TABLE = f"{PROJECT}.{DATASET}.nfl_production_v1_paired_settlements"
SYSTEM_TRIGGER_V1 = f"{PROJECT}.{DATASET}.nfl_system_trigger_v1"
SYSTEM_TRIGGER_V2 = f"{PROJECT}.{DATASET}.nfl_system_family_trigger_v2"

REPORT_OBJECT = "production/nfl/v1/historical_replay/current_report.json"
POLICY_OBJECT = "production/nfl/v1/historical_replay/current_policy.json"
CURRENT_OBJECT = "production/nfl/v1/recommendations/current.json"
EVENT_PREFIX = "production/nfl/v1/recommendations/events"
SETTLEMENT_PREFIX = "production/nfl/v1/recommendations/settlements"


def _sha(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _norm(x) -> str:
    return parity._norm_name(x)


def _num(x):
    try:
        z=float(x)
        return z if math.isfinite(z) else np.nan
    except Exception:
        return np.nan


def _american_implied(o):
    o=_num(o)
    if not math.isfinite(o) or o == 0:
        return np.nan
    return (-o)/((-o)+100.0) if o < 0 else 100.0/(o+100.0)


def _american_profit(o, won):
    o=_num(o)
    if not math.isfinite(o) or o == 0:
        return np.nan
    if not won:
        return -1.0
    return 100.0/(-o) if o < 0 else o/100.0


def _read_json(storage_client, bucket_name: str, name: str):
    blob=storage_client.bucket(bucket_name).blob(name)
    if not blob.exists():
        return None
    try:
        return json.loads(blob.download_as_text())
    except Exception:
        return None


def _write_json(storage_client, bucket_name: str, name: str, payload: dict):
    storage_client.bucket(bucket_name).blob(name).upload_from_string(
        json.dumps(payload, sort_keys=True, indent=2, default=str),
        content_type="application/json",
    )
    return f"gs://{bucket_name}/{name}"


def _write_immutable_json(storage_client, bucket_name: str, name: str, payload: dict):
    blob=storage_client.bucket(bucket_name).blob(name)
    data=json.dumps(payload, sort_keys=True, indent=2, default=str)
    try:
        blob.upload_from_string(data, content_type="application/json", if_generation_match=0)
        return True
    except PreconditionFailed:
        return False


def load_policy(storage_client, bucket_name="sharp-models") -> dict | None:
    p=_read_json(storage_client,bucket_name,POLICY_OBJECT)
    if not isinstance(p,dict):
        return None
    if p.get("status") not in ("NFL_PRODUCTION_V1_RECOMMENDATION_POLICY_READY","NFL_PRODUCTION_V1_RECOMMENDATION_POLICY_PARTIAL"):
        return None
    return p


def _resolve_market_schema(client):
    cols={f.name for f in client.get_table(MARKET_SOURCE).schema}
    def first(*xs): return next((x for x in xs if x in cols),None)
    m={
        "sport":first("Sport"),"market":first("Market"),"outcome":first("Outcome"),"value":first("Value"),
        "odds":first("Odds_Price","Odds","Price"),"book":first("Bookmaker","Book","Sportsbook"),
        "game_start":first("Game_Start","Commence_Hour","feat_Game_Start"),
        "snapshot":first("Snapshot_Timestamp","snapshot_timestamp","Observed_At","Captured_At"),
        "home":first("Home_Team_Norm","Home_Team","Home"),"away":first("Away_Team_Norm","Away_Team","Away"),
    }
    missing=[k for k,v in m.items() if not v]
    if missing:
        raise RuntimeError("NFL_PROD_V1_REC_MARKET_SCHEMA_MISSING "+str(missing))
    return m


def _fetch_market_rows(client, now, lookahead_days=8):
    m=_resolve_market_schema(client)
    sql=f"""
      SELECT CAST(`{m['market']}` AS STRING) market,
             CAST(`{m['outcome']}` AS STRING) outcome,
             SAFE_CAST(`{m['value']}` AS FLOAT64) value,
             SAFE_CAST(`{m['odds']}` AS FLOAT64) odds,
             CAST(`{m['book']}` AS STRING) bookmaker,
             SAFE_CAST(`{m['game_start']}` AS TIMESTAMP) game_start,
             SAFE_CAST(`{m['snapshot']}` AS TIMESTAMP) snapshot_ts,
             CAST(`{m['home']}` AS STRING) home_team,
             CAST(`{m['away']}` AS STRING) away_team
      FROM `{MARKET_SOURCE}`
      WHERE UPPER(TRIM(CAST(`{m['sport']}` AS STRING)))='NFL'
        AND SAFE_CAST(`{m['game_start']}` AS TIMESTAMP) > @now
        AND SAFE_CAST(`{m['game_start']}` AS TIMESTAMP) <= TIMESTAMP_ADD(@now, INTERVAL {int(lookahead_days)} DAY)
        AND SAFE_CAST(`{m['snapshot']}` AS TIMESTAMP) < SAFE_CAST(`{m['game_start']}` AS TIMESTAMP)
    """
    cfg=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("now","TIMESTAMP",pd.to_datetime(now,utc=True).to_pydatetime())])
    d=client.query(sql,job_config=cfg).to_dataframe(create_bqstorage_client=False)
    if d.empty:
        return d
    d["game_start"]=pd.to_datetime(d.game_start,utc=True,errors="coerce")
    d["snapshot_ts"]=pd.to_datetime(d.snapshot_ts,utc=True,errors="coerce")
    d["home_key"]=d.home_team.map(_norm); d["away_key"]=d.away_team.map(_norm); d["outcome_key"]=d.outcome.map(_norm)
    d["market_norm"]=d.market.astype(str).str.lower().str.strip()
    d["bookmaker"]=d.bookmaker.astype(str).str.strip()
    # latest quote per book/outcome/game
    d=d.sort_values("snapshot_ts").drop_duplicates(["game_start","home_key","away_key","market_norm","outcome_key","bookmaker"],keep="last")
    return d.reset_index(drop=True)


def _consensus_market(client, now, lookahead_days=8):
    d=_fetch_market_rows(client,now,lookahead_days)
    out={}
    if d.empty:
        return out
    for (gs,hk,ak),g in d.groupby(["game_start","home_key","away_key"],sort=False):
        rec={"game_start":gs,"home_key":hk,"away_key":ak}
        # spread oriented to home team
        sp=g[g.market_norm.eq("spreads")].copy()
        if not sp.empty:
            vals=[]
            for _,r in sp.iterrows():
                v=_num(r.value)
                if not math.isfinite(v): continue
                if r.outcome_key==hk: vals.append(v)
                elif r.outcome_key==ak: vals.append(-v)
            if vals: rec["home_spread"]=float(np.median(vals)); rec["spread_books"]=int(sp.bookmaker.nunique())
        # totals use Over line as canonical
        to=g[g.market_norm.eq("totals")].copy()
        if not to.empty:
            ov=to[to.outcome.astype(str).str.lower().str.contains("over",na=False)]
            vals=pd.to_numeric(ov.value,errors="coerce").dropna().to_numpy(float)
            if len(vals): rec["total"]=float(np.median(vals)); rec["total_books"]=int(ov.bookmaker.nunique())
        # H2H no-vig consensus by book plus best side prices
        h2=g[g.market_norm.eq("h2h")].copy()
        probs=[]; home_odds=[]; away_odds=[]
        if not h2.empty:
            for book,bg in h2.groupby("bookmaker",sort=False):
                ho=pd.to_numeric(bg.loc[bg.outcome_key.eq(hk),"odds"],errors="coerce").dropna()
                ao=pd.to_numeric(bg.loc[bg.outcome_key.eq(ak),"odds"],errors="coerce").dropna()
                if len(ho): home_odds.extend(ho.astype(float).tolist())
                if len(ao): away_odds.extend(ao.astype(float).tolist())
                if len(ho) and len(ao):
                    ih=_american_implied(ho.iloc[-1]); ia=_american_implied(ao.iloc[-1])
                    if math.isfinite(ih) and math.isfinite(ia) and ih+ia>0: probs.append(ih/(ih+ia))
            if probs: rec["home_novig_probability"]=float(np.median(probs)); rec["h2h_books"]=len(probs)
            if home_odds: rec["best_home_moneyline"]=float(max(home_odds))
            if away_odds: rec["best_away_moneyline"]=float(max(away_odds))
        key=(pd.Timestamp(gs).round("s").isoformat(),hk,ak)
        out[key]=rec
    return out


def _fetch_system_triggers(client, now, lookahead_days=8):
    systems={}
    end=pd.to_datetime(now,utc=True)+pd.Timedelta(days=lookahead_days)
    for table,kind in ((SYSTEM_TRIGGER_V1,"ROLE_FLIP"),(SYSTEM_TRIGGER_V2,"FAMILY")):
        try:
            q=f"SELECT * FROM `{table}` WHERE game_start>=@lo AND game_start<=@hi ORDER BY captured_at"
            cfg=bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("lo","TIMESTAMP",(pd.to_datetime(now,utc=True)-pd.Timedelta(days=1)).to_pydatetime()),
                bigquery.ScalarQueryParameter("hi","TIMESTAMP",end.to_pydatetime()),
            ])
            d=client.query(q,job_config=cfg).to_dataframe(create_bqstorage_client=False)
        except Exception:
            continue
        if d.empty: continue
        for _,r in d.iterrows():
            gs=pd.to_datetime(r.get("game_start"),utc=True,errors="coerce")
            if pd.isna(gs): continue
            hk=_norm(r.get("home_team")); ak=_norm(r.get("away_team"))
            k=(gs.round("s").isoformat(),hk,ak)
            fam=str(r.get("system_family_id") or kind)
            market=str(r.get("market") or "SPREADS").upper()
            direction=str(r.get("direction") or "").upper()
            play=str(r.get("play_team") or r.get("target_team") or "").strip()
            systems.setdefault(k,[]).append({"family":fam,"market":market,"direction":direction,"play":play,"production_authority":0})
    return systems


def _policy_perf(policy_market: dict):
    return {
        "threshold":policy_market.get("threshold"),
        "confirmation":policy_market.get("confirmation"),
        "all_period":policy_market.get("all_period"),
        "status":policy_market.get("status"),
    }


def derive_live_rows(*, client, prediction_rows: list[dict], policy: dict, now):
    quotes=_consensus_market(client,now)
    systems=_fetch_system_triggers(client,now)
    rows=[]
    markets=(policy.get("markets") or {}) if isinstance(policy,dict) else {}
    for p in prediction_rows:
        gs=pd.to_datetime(p.get("game_start"),utc=True,errors="coerce")
        hk=_norm(p.get("home_team")); ak=_norm(p.get("away_team"))
        key=(gs.round("s").isoformat() if pd.notna(gs) else "",hk,ak)
        q=quotes.get(key,{})
        sys=systems.get(key,[])
        base={
            "prediction_pair_id":p.get("prediction_pair_id"),"game_start":None if pd.isna(gs) else gs.isoformat(),
            "home_team":p.get("home_team"),"away_team":p.get("away_team"),"systems":sys,
        }
        # Spread
        pm=markets.get("SPREADS",{}); line=_num(q.get("home_spread")); fair=_num(p.get("champion_fair_margin"))
        edge=fair+line if math.isfinite(fair) and math.isfinite(line) else np.nan
        rows.append(_make_live_row(base,"SPREADS",pm,edge,line,fair,
            (p.get("home_team") if edge>0 else p.get("away_team")) if math.isfinite(edge) else None,
            "HOME_SPREAD_CONSENSUS"))
        # Totals
        pm=markets.get("TOTALS",{}); line=_num(q.get("total")); fair=_num(p.get("champion_fair_total"))
        edge=fair-line if math.isfinite(fair) and math.isfinite(line) else np.nan
        rows.append(_make_live_row(base,"TOTALS",pm,edge,line,fair,("OVER" if edge>0 else "UNDER") if math.isfinite(edge) else None,"TOTAL_CONSENSUS"))
        # H2H
        pm=markets.get("H2H",{}); ref=_num(q.get("home_novig_probability")); fair=_num(p.get("champion_home_win_probability"))
        edge=fair-ref if math.isfinite(fair) and math.isfinite(ref) else np.nan
        selected=(p.get("home_team") if edge>0 else p.get("away_team")) if math.isfinite(edge) else None
        row=_make_live_row(base,"H2H",pm,edge,ref,fair,selected,"NOVIG_HOME_PROB_CONSENSUS")
        row["selected_moneyline"]=_num(q.get("best_home_moneyline" if edge>0 else "best_away_moneyline")) if math.isfinite(edge) else None
        rows.append(row)
    return rows


def _make_live_row(base,market,policy_market,edge,market_value,model_value,selected,reference):
    threshold=_num(policy_market.get("threshold")) if isinstance(policy_market,dict) else np.nan
    confirmed=isinstance(policy_market,dict) and policy_market.get("status")=="CONFIRMED"
    action="PLAY" if confirmed and math.isfinite(edge) and math.isfinite(threshold) and abs(edge)>=threshold else "MODEL LEAN" if math.isfinite(edge) else "NO MARKET"
    return {
        **base,"market":market,"action":action,"selected":selected,"model_edge":None if not math.isfinite(edge) else float(edge),
        "market_value":None if not math.isfinite(market_value) else float(market_value),"model_value":None if not math.isfinite(model_value) else float(model_value),
        "threshold":None if not math.isfinite(threshold) else float(threshold),"market_reference":reference,
        "historical_policy":_policy_perf(policy_market or {}),
    }


def _list_json(storage_client,bucket_name,prefix):
    out=[]
    for blob in storage_client.bucket(bucket_name).list_blobs(prefix=prefix.rstrip("/")+"/"):
        if not blob.name.endswith(".json"): continue
        try: out.append(json.loads(blob.download_as_text()))
        except Exception: pass
    return out


def _capture_events(storage_client,bucket_name,policy,live_rows,now):
    psha=str(policy.get("policy_sha256") or _sha(policy))
    inserted=0; existing=0
    for r in live_rows:
        if r.get("action")!="PLAY": continue
        rid=_sha({"policy":psha,"prediction_pair_id":r.get("prediction_pair_id"),"market":r.get("market")})
        event={**r,"recommendation_id":rid,"policy_sha256":psha,"captured_at":pd.to_datetime(now,utc=True).isoformat(),"source_tag":SOURCE_TAG,
               "formal_betting_authority":False,"automatic_execution":False}
        name=f"{EVENT_PREFIX}/{rid}.json"
        if _write_immutable_json(storage_client,bucket_name,name,event): inserted+=1
        else: existing+=1
    return {"inserted":inserted,"existing":existing}


def _settlement_rows(client, pair_ids):
    if not pair_ids: return pd.DataFrame()
    q=f"SELECT * FROM `{PAIRED_SETTLEMENT_TABLE}` WHERE prediction_pair_id IN UNNEST(@ids)"
    cfg=bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("ids","STRING",list(pair_ids))])
    try: return client.query(q,job_config=cfg).to_dataframe(create_bqstorage_client=False)
    except Exception: return pd.DataFrame()


def _settle_events(client,storage_client,bucket_name,now):
    events=_list_json(storage_client,bucket_name,EVENT_PREFIX)
    settled_existing={str(x.get("recommendation_id")):x for x in _list_json(storage_client,bucket_name,SETTLEMENT_PREFIX)}
    pending=[e for e in events if str(e.get("recommendation_id")) not in settled_existing]
    pairs={str(e.get("prediction_pair_id")) for e in pending if e.get("prediction_pair_id")}
    sd=_settlement_rows(client,pairs)
    smap={str(r.prediction_pair_id):r for _,r in sd.iterrows()} if not sd.empty else {}
    inserted=0
    for e in pending:
        s=smap.get(str(e.get("prediction_pair_id")))
        if s is None: continue
        market=e.get("market"); selected=str(e.get("selected") or "")
        win=loss=push=False; profit=np.nan
        if market=="SPREADS":
            line=_num(e.get("market_value")); margin=_num(s.actual_margin); edge=margin+line
            home_sel=_norm(selected)==_norm(e.get("home_team"))
            val=edge if home_sel else -edge
            push=math.isfinite(val) and abs(val)<1e-9; win=math.isfinite(val) and val>0; loss=math.isfinite(val) and val<0
            profit=0.0 if push else (100/110 if win else -1.0 if loss else np.nan)
        elif market=="TOTALS":
            line=_num(e.get("market_value")); actual=_num(s.actual_total); val=actual-line
            if selected.upper()=="UNDER": val=-val
            push=math.isfinite(val) and abs(val)<1e-9; win=math.isfinite(val) and val>0; loss=math.isfinite(val) and val<0
            profit=0.0 if push else (100/110 if win else -1.0 if loss else np.nan)
        elif market=="H2H":
            y=_num(s.home_win_label); home_sel=_norm(selected)==_norm(e.get("home_team")); won=(y==1.0) if home_sel else (y==0.0)
            win=bool(won); loss=not win; profit=_american_profit(e.get("selected_moneyline"),win)
        result="PUSH" if push else "WIN" if win else "LOSS" if loss else "UNRESOLVED"
        payload={"recommendation_id":e.get("recommendation_id"),"prediction_pair_id":e.get("prediction_pair_id"),"market":market,
                 "selected":selected,"result":result,"profit_per_unit":None if not math.isfinite(_num(profit)) else float(profit),
                 "settled_at":pd.to_datetime(now,utc=True).isoformat(),"actual_margin":_num(s.actual_margin),"actual_total":_num(s.actual_total),
                 "home_win_label":_num(s.home_win_label),"source_tag":SOURCE_TAG}
        if _write_immutable_json(storage_client,bucket_name,f"{SETTLEMENT_PREFIX}/{e.get('recommendation_id')}.json",payload): inserted+=1
    return {"events":len(events),"pending_before":len(pending),"inserted":inserted}


def _performance(storage_client,bucket_name):
    s=_list_json(storage_client,bucket_name,SETTLEMENT_PREFIX)
    def agg(rows):
        if not rows:return {"n":0,"wins":0,"losses":0,"pushes":0,"hit_rate":None,"roi_per_unit":None}
        wins=sum(x.get("result")=="WIN" for x in rows); losses=sum(x.get("result")=="LOSS" for x in rows); pushes=sum(x.get("result")=="PUSH" for x in rows)
        dec=wins+losses; profits=[_num(x.get("profit_per_unit")) for x in rows if math.isfinite(_num(x.get("profit_per_unit")))]
        return {"n":len(rows),"wins":wins,"losses":losses,"pushes":pushes,"hit_rate":round(wins/dec,6) if dec else None,
                "roi_per_unit":round(float(np.mean(profits)),6) if profits else None}
    out={"ALL":agg(s)}
    for m in ("SPREADS","H2H","TOTALS"): out[m]=agg([x for x in s if x.get("market")==m])
    return out


def prospective_model_performance(client, champion_sha: str) -> dict:
    q=f"SELECT * FROM `{PAIRED_SETTLEMENT_TABLE}` WHERE champion_registry_sha256=@sha ORDER BY settled_at"
    cfg=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("sha","STRING",champion_sha)])
    try:d=client.query(q,job_config=cfg).to_dataframe(create_bqstorage_client=False)
    except Exception:return {"settled_games":0}
    if d.empty:return {"settled_games":0}
    def mean(c):
        z=pd.to_numeric(d.get(c),errors="coerce").dropna(); return round(float(z.mean()),6) if len(z) else None
    return {
        "settled_games":int(len(d)),
        "champion_spread_mae":mean("champion_spread_abs_error"),"challenger_spread_mae":mean("challenger_spread_abs_error"),
        "champion_total_mae":mean("champion_total_abs_error"),"challenger_total_mae":mean("challenger_total_abs_error"),
        "champion_h2h_log_loss":mean("champion_h2h_log_loss"),"challenger_h2h_log_loss":mean("challenger_h2h_log_loss"),
        "champion_h2h_brier":mean("champion_h2h_brier"),"challenger_h2h_brier":mean("challenger_h2h_brier"),
    }


def update_recommendation_state(*, bq_client, storage_client, bucket_name, prediction_rows, champion_sha, now, log_func=print):
    policy=load_policy(storage_client,bucket_name)
    if policy is None:
        out={"status":"HOLD_NO_HISTORICAL_POLICY","recommendations":[],"historical_policy":None,"performance":_performance(storage_client,bucket_name)}
        out["current_uri"]=_write_json(storage_client,bucket_name,CURRENT_OBJECT,out)
        log_func("[NFL-PROD-V1-RECOMMENDATIONS] "+json.dumps(out,sort_keys=True,default=str))
        return out
    live=derive_live_rows(client=bq_client,prediction_rows=prediction_rows,policy=policy,now=now)
    capture=_capture_events(storage_client,bucket_name,policy,live,now)
    settlement=_settle_events(bq_client,storage_client,bucket_name,now)
    perf=_performance(storage_client,bucket_name)
    model_perf=prospective_model_performance(bq_client,champion_sha)
    plays=[x for x in live if x.get("action")=="PLAY"]
    out={
        "status":"NFL_PRODUCTION_V1_RECOMMENDATIONS_ACTIVE","source_tag":SOURCE_TAG,"policy_sha256":policy.get("policy_sha256"),
        "generated_at_utc":pd.to_datetime(now,utc=True).isoformat(),"live_rows":live,"recommendations":plays,
        "recommendation_count":len(plays),"capture":capture,"settlement":settlement,"performance":perf,
        "prospective_model_performance":model_perf,"historical_policy":policy,
        "formal_betting_authority":False,"automatic_execution":False,
    }
    out["current_uri"]=_write_json(storage_client,bucket_name,CURRENT_OBJECT,out)
    log_func("[NFL-PROD-V1-RECOMMENDATIONS] "+json.dumps({
        "status":out["status"],"policy_sha256":out["policy_sha256"],"recommendation_count":out["recommendation_count"],
        "capture":capture,"settlement":settlement,"performance":perf,"prospective_model_performance":model_perf,
        "current_uri":out["current_uri"],"formal_betting_authority":False,
    },sort_keys=True,default=str))
    return out


def read_dashboard_state(*, bq_client=None, storage_client=None, bucket_name="sharp-models"):
    if storage_client is None:
        from google.cloud import storage
        storage_client=storage.Client()
    cur=_read_json(storage_client,bucket_name,CURRENT_OBJECT) or {}
    pol=_read_json(storage_client,bucket_name,POLICY_OBJECT) or {}
    report=_read_json(storage_client,bucket_name,REPORT_OBJECT) or {}
    return {"current":cur,"policy":pol,"report":report}


def _self_test():
    pm={"status":"CONFIRMED","threshold":3.0,"confirmation":{"CLOSE":{"n":60,"hit_rate":.56,"roi":.06}},"all_period":{}}
    b={"prediction_pair_id":"x","game_start":"2026-10-04T17:00:00+00:00","home_team":"a","away_team":"b","systems":[]}
    r=_make_live_row(b,"SPREADS",pm,4.0,-3.0,1.0,"a","HOME_SPREAD_CONSENSUS")
    assert r["action"]=="PLAY"
    assert _american_profit(-110,True)>0 and _american_profit(150,True)==1.5
    return {"status":"PASS","source_tag":SOURCE_TAG}

if __name__=="__main__":
    print(json.dumps(_self_test(),sort_keys=True))
