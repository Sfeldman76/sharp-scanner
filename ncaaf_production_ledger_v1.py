"""Immutable prospective NCAAF Production V1 record + settlement.

The scanner (utils.detect_sharp_moves) creates records independently of dashboard
visits. Never rescore a game after it starts. Predictions and results live in
separate append-only BigQuery tables, isolated from the old V13 shadow ledger.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
import os
import re
import uuid

import numpy as np
import pandas as pd

PRED_TABLE = "sharplogger.sharp_data.ncaaf_production_v1_predictions"
RESULT_TABLE = "sharplogger.sharp_data.ncaaf_production_v1_results"
# Segregates pre-fix (outcome-keyed) predictions without rewriting immutable history.
LEDGER_VERSION = "ncaaf-production-v2-core-candidate-bet-authority-20261005"
_ALLOWED_MARKETS = {"spreads", "h2h", "totals"}
_ALLOWED_ACTIONS = {"BET", "STRONG BET", "CANDIDATE", "PLAY", "STRONG PLAY", "MODEL ONLY", "PASS", "PASS — CONFLICT", "EDGE — NO EXEC QUOTE"}


def _slug(v):
    if v is None or pd.isna(v): return ""
    return re.sub(r"[^a-z0-9]+", "_", str(v).lower()).strip("_")


def _name(v):
    if v is None or pd.isna(v): return ""
    return " ".join(str(v).strip().lower().split())


def _str(v):
    if v is None: return ""
    try:
        if pd.isna(v): return ""
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return "" if s.lower() in ("", "nan", "nat", "none") else s


def _flt(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _ts(v):
    return pd.to_datetime(v, utc=True, errors="coerce")


def _utc_now(now=None):
    z = pd.Timestamp.now(tz="UTC") if now is None else _ts(now)
    if pd.isna(z): raise ValueError("invalid now timestamp")
    return z


def _hash(s):
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()


def _win_profit(american):
    o=_flt(american)
    if o is None or o==0: return None
    return (100.0/-o) if o<0 else (o/100.0)


def _be(american):
    p=_win_profit(american)
    return 1.0/(1.0+p) if p is not None else None


def _team_value(r,normalized,original):
    return _str(r.get(normalized)) or _str(r.get(original))


def _physical(r):
    for c in ("_prod_game_id", "Merge_Key_Short", "Physical_Game_ID", "physical_game_id"):
        x=_str(r.get(c))
        if x: return x.lower(), "PHYSICAL_ID"
    h=_slug(_team_value(r,"Home_Team_Norm","Home_Team"))
    a=_slug(_team_value(r,"Away_Team_Norm","Away_Team"))
    g=_ts(r.get("Game_Start"))
    if not h or not a or pd.isna(g): return "", "MISSING"
    return "derived_"+_hash(f"{h}|{a}|{g.isoformat()}")[:24], "TEAM_KICKOFF"


def _market(v):
    s=_name(v)
    return {"spread":"spreads", "ats":"spreads", "moneyline":"h2h", "ml":"h2h", "total":"totals"}.get(s,s)


def _action(r):
    decision=_str(r.get("_prod_decision")).upper()
    act=_str(r.get("_prod_action")).upper()
    if decision in ("BET","STRONG_BET"):
        if not bool(r.get("_exec", False)) or _flt(r.get("_odds",r.get("Odds_Price"))) is None:
            return "EDGE — NO EXEC QUOTE"
        return "STRONG BET" if decision=="STRONG_BET" else "BET"
    if decision=="CANDIDATE": return "CANDIDATE"
    # Backward compatibility for historical pre-V2 selectors.
    if decision in ("EDGE_SINGLE", "EDGE_MULTI"):
        if not bool(r.get("_exec", False)) or _flt(r.get("_odds",r.get("Odds_Price"))) is None:
            return "EDGE — NO EXEC QUOTE"
        return "STRONG PLAY" if decision=="EDGE_MULTI" else "PLAY"
    if decision=="PASS_CONFLICT": return "PASS — CONFLICT"
    if decision=="PASS": return "PASS"
    return "MODEL ONLY" if act=="MODEL ONLY" else "PASS"

def prepare_prediction_events(picks, contract, *, now=None, source="BACKGROUND_SCANNER", max_quote_age_minutes=180):
    """Convert the SAME model+edge-selected rows used by the UI into pregame locks.

    No timestamp is backdated. No artifact hash => no formally gradeable record.
    The first lock is keyed by artifact+game+market, not by side, so later line
    moves / directional flips cannot silently rewrite the initial prediction.
    """
    if picks is None or picks.empty:
        return pd.DataFrame(), {"status":"NO_PICKS", "attempted":0}
    if not isinstance(contract, dict) or int(contract.get("production_authority",0) or 0)!=1:
        return pd.DataFrame(), {"status":"CONTRACT_UNAVAILABLE", "attempted":0}
    sha=_str(contract.get("_artifact_sha256"))
    if len(sha)!=64:
        return pd.DataFrame(), {"status":"MISSING_ARTIFACT_HASH", "attempted":0}
    # A distinct selector instance prevents the old, incorrectly side-keyed
    # FIRST lock from blocking a corrected pregame pick for the same artifact.
    instance="NCAAF_PROD_BETAUTH_V2__"+sha[:16]
    n=_utc_now(now)
    seen=set(); records=[]; rejected={}
    def reject(k): rejected[k]=rejected.get(k,0)+1
    # Safety for mixed/partial deployments: if a caller supplies both outcomes
    # for one physical game and market, lock NEITHER. First-row-wins is unsafe.
    candidate_keys=[]
    for _,candidate in picks.iterrows():
        physical_id,_=_physical(candidate)
        candidate_keys.append((physical_id,_market(candidate.get("Market"))))
    from collections import Counter
    ambiguous={key for key,count in Counter(candidate_keys).items() if key[0] and count>1}
    for (_,r),physical_key in zip(picks.iterrows(),candidate_keys):
        if physical_key in ambiguous:
            reject("AMBIGUOUS_PHYSICAL_GAME_MARKET"); continue
        market=_market(r.get("Market"))
        if market not in _ALLOWED_MARKETS: reject("MARKET"); continue
        g=_ts(r.get("Game_Start")); q=_ts(r.get("Snapshot_Timestamp"))
        if pd.isna(g) or pd.isna(q) or g<=n+pd.Timedelta(minutes=5) or q>n+pd.Timedelta(minutes=1) or q>=g:
            reject("NOT_PROSPECTIVE"); continue
        if (n-q).total_seconds()>max_quote_age_minutes*60:
            reject("STALE_QUOTE"); continue
        side=_str(r.get("Outcome")); home=_team_value(r,"Home_Team_Norm","Home_Team"); away=_team_value(r,"Away_Team_Norm","Away_Team")
        if market=="totals":
            if _name(side) not in ("over","under"): reject("TOTAL_SIDE"); continue
        elif _slug(side) not in (_slug(home),_slug(away)) or not _slug(side):
            reject("TEAM_SIDE"); continue
        prob=_flt(r.get("_pred",r.get("_model_prob")))
        if prob is None or not 0<=prob<=1: reject("PROBABILITY"); continue
        odds=_flt(r.get("_odds",r.get("Odds_Price")))
        line=_flt(r.get("_line",r.get("Value")))
        if (market in ("spreads","totals") and line is None): reject("LINE"); continue
        phys, match_type=_physical(r)
        if not phys: reject("GAME_ID"); continue
        game_key=(instance,phys,market)
        if game_key in seen: reject("DUPLICATE_GAME_MARKET"); continue
        seen.add(game_key)
        exec_quote=bool(r.get("_exec",False)) and odds is not None and odds!=0
        act=_action(r)
        if act in ("BET","STRONG BET","PLAY","STRONG PLAY") and not exec_quote: act="EDGE — NO EXEC QUOTE"
        edge=prob-_be(odds) if _be(odds) is not None else None
        ev=prob*_win_profit(odds)-(1-prob) if _win_profit(odds) is not None else None
        common={
            "ledger_version":LEDGER_VERSION,"model_version":_str(contract.get("source_tag")),"model_instance_id":instance,
            "artifact_sha256":sha,"artifact_gcs_uri":_str(contract.get("_artifact_gcs_uri")),"artifact_generation":_str(contract.get("_artifact_generation")),
            "model_published_utc":_ts(contract.get("published_utc")),
            "physical_game_id":phys,"game_match_type":match_type,"game_key":_str(r.get("Game_Key")),
            "game_start":g,"home_team":home,"away_team":away,"market":market,"side":side,
            "book":_str(r.get("_book",r.get("Bookmaker"))),"line":line,"odds":odds,
            "quote_timestamp":q,"prediction_timestamp":n,"prediction_source":source,"is_prospective":True,
            "model_probability":prob,"break_even_probability":_be(odds),"model_edge":edge,"model_ev":ev,
            "model_id":_str(r.get("_model_id")),"decision":_str(r.get("_prod_decision")),"action":act,
            "production_authority":int(r.get("_prod_authority",0) or 0),"quote_executable":exec_quote,
            "edge_sources":_str(r.get("_prod_sources")),"edge_mechanisms":_str(r.get("_prod_mechanisms")),
            "edge_family_count":int(r.get("_prod_family_count",0) or 0),
            "independent_mechanisms":int(r.get("_prod_independent_mechanisms",0) or 0),
            "edge_reason":_str(r.get("_prod_reason")),
            "bet_authority_policy":_str(r.get("_bet_authority_policy")),"core_qualifies":bool(r.get("_core_qualifies",False)),
            "core_edge_gate":_flt(r.get("_core_edge_gate")),"core_ev_gate":_flt(r.get("_core_ev_gate")),
            "system_support_count":int(r.get("_system_support_count",0) or 0),"system_support_families":_str(r.get("_system_support_families")),
            "system_support_sources":_str(r.get("_system_support_sources")),"system_conflict_count":int(r.get("_system_conflict_count",0) or 0),
            "system_conflict_families":_str(r.get("_system_conflict_families")),"system_conflict_sources":_str(r.get("_system_conflict_sources")),
            "bet_authority_confidence":_str(r.get("_bet_authority_confidence")),
            "pathi_active":_str(r.get("Pathi_Active_Text")),"bigal_active":_str(r.get("BigAl_Active_Text")),
            "miner_system_summary":_str(r.get("NCAAF_RV2_System_Summary")) or _str(r.get("Miner_System_Summary")),
            "system_triggers":" | ".join(dict.fromkeys(x for x in (
                _str(r.get("_prod_sources")),_str(r.get("Pathi_Active_Text")),
                _str(r.get("BigAl_Active_Text")),_str(r.get("NCAAF_RV2_System_Summary")),_str(r.get("Miner_System_Summary")))
                if x and x not in ("—","-"))),
            "line_hash":_str(r.get("Line_Hash")),
        }
        hours=(g-n).total_seconds()/3600.0
        locks=["ROLLING","FIRST"]
        if 18<=hours<=30: locks.append("T24")
        if 4<=hours<=8: locks.append("T6")
        if .5<=hours<=1.5: locks.append("T1")
        for lock in locks:
            rec=dict(common); rec["lock_type"]=lock
            key=f"{instance}|{phys}|{market}|{lock}"
            if lock=="ROLLING":
                key+="|"+_hash(f"{side}|{rec['book']}|{line}|{odds}|{q.isoformat()}")
            rec["prediction_event_id"]=_hash(key)
            records.append(rec)
    status="PREPARED" if records else "NO_ELIGIBLE_ROWS"
    return pd.DataFrame(records), {"status":status,"attempted":len(records),"rejected":rejected,"model_instance_id":instance}


def _schema_predictions():
    from google.cloud import bigquery as b
    S=b.SchemaField
    fields={
      "prediction_event_id":"STRING", "ledger_version":"STRING", "model_version":"STRING","model_instance_id":"STRING","artifact_sha256":"STRING",
      "artifact_gcs_uri":"STRING","artifact_generation":"STRING","model_published_utc":"TIMESTAMP",
      "physical_game_id":"STRING","game_match_type":"STRING","game_key":"STRING","game_start":"TIMESTAMP","home_team":"STRING","away_team":"STRING",
      "market":"STRING","side":"STRING","book":"STRING","line":"FLOAT64","odds":"FLOAT64","quote_timestamp":"TIMESTAMP","prediction_timestamp":"TIMESTAMP",
      "prediction_source":"STRING","is_prospective":"BOOL","model_probability":"FLOAT64","break_even_probability":"FLOAT64","model_edge":"FLOAT64","model_ev":"FLOAT64",
      "model_id":"STRING","decision":"STRING","action":"STRING","production_authority":"INT64","quote_executable":"BOOL","edge_sources":"STRING",
      "edge_mechanisms":"STRING","edge_family_count":"INT64","independent_mechanisms":"INT64","edge_reason":"STRING",
      "bet_authority_policy":"STRING","core_qualifies":"BOOL","core_edge_gate":"FLOAT64","core_ev_gate":"FLOAT64",
      "system_support_count":"INT64","system_support_families":"STRING","system_support_sources":"STRING",
      "system_conflict_count":"INT64","system_conflict_families":"STRING","system_conflict_sources":"STRING","bet_authority_confidence":"STRING",
      "pathi_active":"STRING","bigal_active":"STRING","miner_system_summary":"STRING","system_triggers":"STRING","line_hash":"STRING","lock_type":"STRING",
    }
    return [S(k,t,"REQUIRED" if k=="prediction_event_id" else "NULLABLE") for k,t in fields.items()]


def _schema_results():
    from google.cloud import bigquery as b
    S=b.SchemaField
    f={
      "result_event_id":"STRING","prediction_event_id":"STRING","ledger_version":"STRING","settled_at":"TIMESTAMP", "model_instance_id":"STRING",
      "artifact_sha256":"STRING","physical_game_id":"STRING","game_start":"TIMESTAMP","market":"STRING","side":"STRING","lock_type":"STRING",
      "decision":"STRING","action":"STRING","quote_executable":"BOOL","locked_book":"STRING","locked_line":"FLOAT64","locked_odds":"FLOAT64",
      "locked_probability":"FLOAT64","home_score":"FLOAT64","away_score":"FLOAT64","result":"STRING","profit_per_unit":"FLOAT64","brier":"FLOAT64","log_loss":"FLOAT64",
      "closing_book":"STRING","closing_line":"FLOAT64","closing_odds":"FLOAT64","closing_quote_timestamp":"TIMESTAMP","closing_source":"STRING","clv_points":"FLOAT64","clv_implied_prob_delta":"FLOAT64",
    }
    return [S(k,t,"REQUIRED" if k in ("result_event_id","prediction_event_id") else "NULLABLE") for k,t in f.items()]


# BigQuery's Python API can return canonical field_type names (FLOAT, BOOLEAN,
# INTEGER) for schemas originally declared as FLOAT64, BOOL, INT64. Comparing
# those names literally causes a false schema mismatch on an EXISTING table.
# Normalize aliases only; a genuine type change must still fail closed.
_BQ_TYPE_ALIASES = {
    "FLOAT64": "FLOAT", "FLOAT": "FLOAT",
    "BOOL": "BOOLEAN", "BOOLEAN": "BOOLEAN",
    "INT64": "INTEGER", "INTEGER": "INTEGER",
}


def _bq_type(v):
    name = str(v).strip().upper()
    return _BQ_TYPE_ALIASES.get(name, name)


def _ensure_one(client,table_fq,schema,partition,clusters):
    from google.cloud import bigquery as b
    from google.api_core.exceptions import NotFound
    try:
        t=client.get_table(table_fq)
        d={x.name:x.field_type for x in t.schema}
        conflicts=[f.name for f in schema if f.name in d and _bq_type(d[f.name])!=_bq_type(f.field_type)]
        if conflicts:
            detail={name:{"existing":d[name],"expected":next(f.field_type for f in schema if f.name==name)} for name in conflicts}
            raise RuntimeError(f"schema type mismatch {table_fq}: {detail}")
        aliases=[f.name for f in schema if f.name in d and str(d[f.name]).strip().upper()!=str(f.field_type).strip().upper()]
        if aliases:
            logging.info("[NCAAF-PROD-V1-SCHEMA] status=ALIAS_COMPATIBLE table=%s fields=%s",table_fq,",".join(aliases))
        missing=[f for f in schema if f.name not in d]
        if missing:
            t.schema=list(t.schema)+missing
            client.update_table(t,["schema"])
    except NotFound:
        t=b.Table(table_fq,schema=schema)
        t.time_partitioning=b.TimePartitioning(type_=b.TimePartitioningType.DAY,field=partition)
        t.clustering_fields=clusters[:4]
        client.create_table(t)


def ensure_tables(client=None):
    from google.cloud import bigquery as b
    c=client or b.Client(project="sharplogger")
    _ensure_one(c,PRED_TABLE,_schema_predictions(),"prediction_timestamp",["model_instance_id","physical_game_id","market","lock_type"])
    _ensure_one(c,RESULT_TABLE,_schema_results(),"settled_at",["model_instance_id","physical_game_id","market","lock_type"])
    return {"status":"READY","predictions":PRED_TABLE,"results":RESULT_TABLE}


def _append_unique(client,table,frame,key):
    if frame.empty: return {"status":"NO_ROWS","attempted":0,"inserted":0}
    from google.cloud import bigquery as b
    frame=frame.drop_duplicates(key).copy()
    schema=client.get_table(table).schema
    cols=[f.name for f in schema]
    for c in cols:
        if c not in frame: frame[c]=None
    frame=frame[cols]
    for f in schema:
        if f.field_type=="TIMESTAMP": frame[f.name]=pd.to_datetime(frame[f.name],errors="coerce",utc=True)
    stage=table+"__stage_"+uuid.uuid4().hex[:12]
    t=b.Table(stage,schema=schema)
    t.expires=dt.datetime.now(dt.timezone.utc)+dt.timedelta(hours=2)
    client.create_table(t)
    try:
        client.load_table_from_dataframe(frame,stage,job_config=b.LoadJobConfig(schema=schema,write_disposition="WRITE_TRUNCATE")).result()
        csql=",".join(f"`{v}`" for v in cols)
        vsql=",".join(f"S.`{v}`" for v in cols)
        job=client.query(f"MERGE `{table}` T USING `{stage}` S ON T.`{key}`=S.`{key}` WHEN NOT MATCHED THEN INSERT ({csql}) VALUES ({vsql})")
        job.result()
        return {"status":"PASS","attempted":len(frame),"inserted":int(getattr(job,"num_dml_affected_rows",0) or 0)}
    finally:
        client.delete_table(stage,not_found_ok=True)


def record_predictions(picks,contract,*,client=None,now=None,source="BACKGROUND_SCANNER"):
    if os.getenv("NCAAF_PROD_V1_LEDGER_ENABLED","1").strip().lower() in ("0","false","off","no"):
        return {"status":"DISABLED","attempted":0,"inserted":0}
    frame,meta=prepare_prediction_events(picks,contract,now=now,source=source)
    if frame.empty: return dict(meta,inserted=0)
    try:
        from google.cloud import bigquery as b
        c=client or b.Client(project="sharplogger")
        ensure_tables(c)
        res=_append_unique(c,PRED_TABLE,frame,"prediction_event_id")
        return {**meta,**res}
    except Exception as exc:
        logging.exception("[NCAAF-PROD-V1-LEDGER] persistence error")
        return {**meta,"status":"ERROR","inserted":0,"error":f"{type(exc).__name__}:{exc}"}


def _result_for_prediction(p, f):
    """Grade only immutable quoted side/line/odds against verified final scores."""
    h=_slug(p.get("home_team")); a=_slug(p.get("away_team"))
    fh=_slug(f.get("Home_Team")); fa=_slug(f.get("Away_Team"))
    if not h or not a or h!=fh or a!=fa: return None
    gs=_ts(p.get("game_start")); fg=_ts(f.get("Game_Start"))
    if pd.isna(gs) or pd.isna(fg) or abs((gs-fg).total_seconds())>7200: return None
    hs=_flt(f.get("home_score")); aws=_flt(f.get("away_score"))
    if hs is None or aws is None or hs<0 or aws<0: return None
    m=_market(p.get("market")); side=_slug(p.get("side")); line=_flt(p.get("line"))
    if m=="spreads":
        if line is None: return None
        if side==h: outcome=(hs-aws)+line
        elif side==a: outcome=(aws-hs)+line
        else: return None
    elif m=="h2h":
        if side==h: outcome=hs-aws
        elif side==a: outcome=aws-hs
        else: return None
        if outcome==0: return None # No ties in the frozen binary home-win label.
    elif m=="totals":
        if line is None: return None
        if side=="over": outcome=hs+aws-line
        elif side=="under": outcome=line-(hs+aws)
        else: return None
    else: return None
    result="PUSH" if abs(outcome)<1e-9 else ("WIN" if outcome>0 else "LOSS")
    prob=_flt(p.get("model_probability"))
    if prob is None or not 0<=prob<=1: return None
    y=1.0 if result=="WIN" else (0.0 if result=="LOSS" else None)
    brier=(prob-y)**2 if y is not None else None
    clipped=min(max(prob,1e-12),1-1e-12)
    ll=-(y*math.log(clipped)+(1-y)*math.log(1-clipped)) if y is not None else None
    profit=0.0 if result=="PUSH" else (_win_profit(p.get("odds")) if result=="WIN" else -1.0)
    if not bool(p.get("quote_executable",False)) or _win_profit(p.get("odds")) is None: profit=None
    return {"result":result,"home_score":hs,"away_score":aws,"profit_per_unit":profit,"brier":brier,"log_loss":ll}


def grade_prediction_rows(predictions,finals,*,now=None):
    """Pure, testable grading. No prediction mutation or postgame rescoring."""
    n=_utc_now(now); out=[]
    if predictions is None or predictions.empty or finals is None or finals.empty:
        return pd.DataFrame()
    finals=finals.copy(); finals["physical_game_id"]=finals["physical_game_id"].astype(str).str.lower().str.strip()
    # Fail closed on ambiguous final-score records; never pick arbitrarily.
    ambiguity=finals.groupby("physical_game_id").agg(home_score_n=("home_score","nunique"),away_score_n=("away_score","nunique"))
    valid_ids=ambiguity[(ambiguity.home_score_n<=1)&(ambiguity.away_score_n<=1)].index
    fmap={p:g.iloc[0] for p,g in finals[finals.physical_game_id.isin(valid_ids)].groupby("physical_game_id")}
    for _,p in predictions.iterrows():
        pid=_str(p.get("prediction_event_id")); phys=_str(p.get("physical_game_id")).lower(); f=fmap.get(phys)
        if not pid or f is None: continue
        g=_ts(p.get("game_start")); pts=_ts(p.get("prediction_timestamp")); qts=_ts(p.get("quote_timestamp"))
        if pd.isna(g) or pd.isna(pts) or pd.isna(qts) or pts>=g or qts>=g or pts>n: continue
        if g>n: continue
        base=_result_for_prediction(p,f)
        if base is None: continue
        rec={
          "result_event_id":_hash(pid+"|SETTLEMENT"),"prediction_event_id":pid,"ledger_version":LEDGER_VERSION,"settled_at":n,
          "model_instance_id":_str(p.get("model_instance_id")),"artifact_sha256":_str(p.get("artifact_sha256")),
          "physical_game_id":phys,"game_start":g,"market":_market(p.get("market")),"side":_str(p.get("side")),"lock_type":_str(p.get("lock_type")),
          "decision":_str(p.get("decision")),"action":_str(p.get("action")),"quote_executable":bool(p.get("quote_executable",False)),
          "locked_book":_str(p.get("book")),"locked_line":_flt(p.get("line")),"locked_odds":_flt(p.get("odds")),"locked_probability":_flt(p.get("model_probability")),
          "closing_book":None,"closing_line":None,"closing_odds":None,"closing_quote_timestamp":None,"closing_source":"UNAVAILABLE","clv_points":None,"clv_implied_prob_delta":None,
          **base,
        }
        out.append(rec)
    return pd.DataFrame(out)


def settle_results(*,client=None,lookback_days=35):
    if os.getenv("NCAAF_PROD_V1_LEDGER_ENABLED","1").strip().lower() in ("0","false","off","no"):
        return {"status":"DISABLED","settled":0}
    try:
        from google.cloud import bigquery as b
        c=client or b.Client(project="sharplogger")
        ensure_tables(c)
        days=int(max(1,min(int(lookback_days),180)))
        pred=c.query(f"""
            SELECT p.* FROM `{PRED_TABLE}` p
            LEFT JOIN `{RESULT_TABLE}` r USING(prediction_event_id)
            WHERE r.prediction_event_id IS NULL
              AND p.is_prospective=TRUE
              AND p.prediction_timestamp < p.game_start AND p.quote_timestamp < p.game_start
              AND p.game_start < CURRENT_TIMESTAMP()
              AND p.game_start >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
        """).to_dataframe(create_bqstorage_client=False)
        if pred.empty: return {"status":"NO_UNSETTLED","settled":0}
        ids=sorted(set(pred.physical_game_id.dropna().astype(str).str.lower()))
        cfg=b.QueryJobConfig(query_parameters=[b.ArrayQueryParameter("ids","STRING",ids)])
        finals=c.query("""
            SELECT LOWER(TRIM(CAST(Merge_Key_Short AS STRING))) AS physical_game_id,
                   Home_Team,Away_Team,Game_Start,
                   SAFE_CAST(Score_Home_Score AS FLOAT64) AS home_score,
                   SAFE_CAST(Score_Away_Score AS FLOAT64) AS away_score
            FROM `sharplogger.sharp_data.game_scores_final`
            WHERE LOWER(TRIM(CAST(Merge_Key_Short AS STRING))) IN UNNEST(@ids)
              AND Score_Home_Score IS NOT NULL AND Score_Away_Score IS NOT NULL
        """,job_config=cfg).to_dataframe(create_bqstorage_client=False)
        # Derived IDs can only be matched by an exact home/away/kickoff composite;
        # never compare fuzzy names or scores against arbitrary games.
        derived={v for v in ids if v.startswith("derived_")}
        if derived:
            f2=c.query(f"""
                SELECT Home_Team,Away_Team,Game_Start,
                       SAFE_CAST(Score_Home_Score AS FLOAT64) AS home_score,
                       SAFE_CAST(Score_Away_Score AS FLOAT64) AS away_score
                FROM `sharplogger.sharp_data.game_scores_final`
                WHERE SAFE_CAST(Game_Start AS TIMESTAMP) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days+2} DAY)
                  AND Score_Home_Score IS NOT NULL AND Score_Away_Score IS NOT NULL
                  AND Game_Start IS NOT NULL
            """).to_dataframe(create_bqstorage_client=False)
            if not f2.empty:
                f2["physical_game_id"]=f2.apply(lambda x:"derived_"+_hash(f"{_slug(x.Home_Team)}|{_slug(x.Away_Team)}|{_ts(x.Game_Start).isoformat()}")[:24],axis=1)
                finals=pd.concat([finals,f2[f2.physical_game_id.isin(derived)]],ignore_index=True)
        if finals.empty: return {"status":"NO_MATCHED_FINALS","settled":0,"unsettled":len(pred)}
        events=grade_prediction_rows(pred,finals)
        if events.empty: return {"status":"NO_GRADEABLE_FINALS","settled":0,"unsettled":len(pred)}
        # Use the latest captured quote for the same model/game/market/side/book as
        # tracked close. Historical predictions are never changed.
        clos=c.query(f"""
            SELECT model_instance_id,physical_game_id,market,side,book,line,odds,quote_timestamp,prediction_timestamp
            FROM `{PRED_TABLE}` WHERE physical_game_id IN UNNEST(@ids) AND lock_type='ROLLING'
              AND quote_timestamp < game_start AND prediction_timestamp < game_start
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY model_instance_id,physical_game_id,market,LOWER(side),LOWER(book)
                ORDER BY quote_timestamp DESC,prediction_timestamp DESC
            )=1
        """,job_config=cfg).to_dataframe(create_bqstorage_client=False)
        if not clos.empty:
            cmap={(str(x.model_instance_id),str(x.physical_game_id).lower(),_market(x.market),_slug(x.side),_slug(x.book)):x for _,x in clos.iterrows()}
            pmap={str(x.prediction_event_id):x for _,x in pred.iterrows()}
            for ix,e in events.iterrows():
                p=pmap[str(e.prediction_event_id)]; key=(str(p.model_instance_id),str(p.physical_game_id).lower(),_market(p.market),_slug(p.side),_slug(p.book))
                cl=cmap.get(key)
                if cl is None: continue
                events.at[ix,"closing_book"]=_str(cl.book); events.at[ix,"closing_line"]=_flt(cl.line)
                events.at[ix,"closing_source"]="LATEST_SELECTED_ROLLING_QUOTE"
                events.at[ix,"closing_odds"]=_flt(cl.odds); events.at[ix,"closing_quote_timestamp"]=_ts(cl.quote_timestamp)
                old=_flt(p.line); new=_flt(cl.line); market=_market(p.market)
                if old is not None and new is not None:
                    if market=="spreads": events.at[ix,"clv_points"]=old-new
                    if market=="totals": events.at[ix,"clv_points"]=(new-old) if _slug(p.side)=="over" else (old-new)
                oldp=_be(p.odds); newp=_be(cl.odds)
                if oldp is not None and newp is not None and (market=="h2h" or (old is not None and new is not None and abs(old-new)<1e-9)):
                    events.at[ix,"clv_implied_prob_delta"]=newp-oldp
        res=_append_unique(c,RESULT_TABLE,events,"result_event_id")
        logging.info("[NCAAF-PROD-V1-SETTLEMENT] status=%s attempted=%s inserted=%s",res["status"],res["attempted"],res["inserted"])
        return {"status":res["status"],"settled":res["inserted"],"attempted":res["attempted"],"unsettled":len(pred)}
    except Exception as exc:
        logging.exception("[NCAAF-PROD-V1-SETTLEMENT] failed")
        return {"status":"ERROR","settled":0,"error":f"{type(exc).__name__}:{exc}"}


def read_summary(*,client=None,days=60):
    """Read-only reporting; performance never retrains or changes authority."""
    try:
        from google.cloud import bigquery as b
        c=client or b.Client(project="sharplogger")
        # Read-only UI: the background scanner is responsible for DDL.
        days=int(max(1,min(int(days),365)))
        return c.query(f"""
          SELECT p.model_instance_id,p.market,p.lock_type,p.action,
                 COUNT(*) AS captured,
                 COUNTIF(r.result IN ('WIN','LOSS')) AS decided,
                 COUNTIF(r.result='WIN') AS wins,COUNTIF(r.result='PUSH') AS pushes,
                 SAFE_DIVIDE(COUNTIF(r.result='WIN'),COUNTIF(r.result IN ('WIN','LOSS'))) AS hit_rate,
                 SAFE_DIVIDE(SUM(IF(p.action IN ('PLAY','STRONG PLAY','BET','STRONG BET') AND p.quote_executable AND r.result IN ('WIN','LOSS','PUSH'), r.profit_per_unit, NULL)),
                   COUNTIF(p.action IN ('PLAY','STRONG PLAY','BET','STRONG BET') AND p.quote_executable AND r.result IN ('WIN','LOSS','PUSH'))) AS play_roi,
                 AVG(r.brier) AS brier,AVG(r.log_loss) AS log_loss,AVG(r.clv_points) AS avg_clv_points,
                 MAX(p.prediction_timestamp) AS last_prediction
          FROM `{PRED_TABLE}` p LEFT JOIN `{RESULT_TABLE}` r USING(prediction_event_id)
          WHERE p.prediction_timestamp>=TIMESTAMP_SUB(CURRENT_TIMESTAMP(),INTERVAL {days} DAY)
            AND p.ledger_version='{LEDGER_VERSION}'
            AND p.lock_type IN ('FIRST','T24','T6','T1')
          GROUP BY 1,2,3,4 ORDER BY last_prediction DESC
        """).to_dataframe(create_bqstorage_client=False)
    except Exception as exc:
        logging.warning("[NCAAF-PROD-V1-SUMMARY] unavailable: %s:%s",type(exc).__name__,exc)
        return pd.DataFrame()



def read_details(*,client=None,days=60,lock_type="FIRST",limit=300):
    """Exact historical pick and result drill-down; never re-run a model."""
    lock=str(lock_type).upper()
    if lock not in {"FIRST","T24","T6","T1"}:
        raise ValueError("unsupported lock_type")
    try:
        from google.cloud import bigquery as b
        c=client or b.Client(project="sharplogger")
        d=int(max(1,min(int(days),365))); n=int(max(1,min(int(limit),1000)))
        return c.query(f"""
            SELECT p.game_start,p.home_team,p.away_team,p.model_instance_id,
                   p.market,p.lock_type,p.action,p.decision,p.side,
                   p.book,p.line AS locked_line,p.odds AS locked_odds,
                   p.model_probability,p.model_edge,p.model_ev,
                   p.edge_sources,p.edge_mechanisms,p.system_triggers,
                   p.quote_timestamp,p.prediction_timestamp,
                   p.quote_executable,
                   IFNULL(r.result,'PENDING') AS grade,r.home_score,r.away_score,
                   r.profit_per_unit,r.brier,r.log_loss,r.clv_points,r.clv_implied_prob_delta,r.closing_source,
                   r.settled_at
            FROM `{PRED_TABLE}` p LEFT JOIN `{RESULT_TABLE}` r USING(prediction_event_id)
            WHERE p.prediction_timestamp>=TIMESTAMP_SUB(CURRENT_TIMESTAMP(),INTERVAL {d} DAY)
              AND p.ledger_version='{LEDGER_VERSION}'
              AND p.lock_type='{lock}'
            ORDER BY p.game_start DESC LIMIT {n}
        """).to_dataframe(create_bqstorage_client=False)
    except Exception as exc:
        logging.warning("[NCAAF-PROD-V1-DETAILS] unavailable: %s:%s",type(exc).__name__,exc)
        return pd.DataFrame()


if __name__=="__main__":
    import sys
    if sys.argv[1:] == ["settle"]:
        print(json.dumps(settle_results(),default=str))
    else:
        raise SystemExit("Usage: python -m ncaaf_production_ledger_v1 settle")
