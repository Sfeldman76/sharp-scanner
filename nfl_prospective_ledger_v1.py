"""Append-only NFL research prediction ledger for V1.9+.

This module creates storage for future pregame predictions but never backfills
historical games and never recomputes a locked record. Integration with the live
NFL scanner should call append_prediction_rows only with genuinely pregame rows.
"""
from __future__ import annotations
import hashlib, json, math
import pandas as pd

SOURCE_TAG="nfl-prospective-ledger-v1.9-20260930"
PRED_TABLE="sharplogger.sharp_data.nfl_research_v1_predictions"
RESULT_TABLE="sharplogger.sharp_data.nfl_research_v1_results"
LEDGER_VERSION="nfl-research-v1.9-frozen-registry-20260930"


def _schema_predictions():
    from google.cloud import bigquery as b; S=b.SchemaField
    return [S("prediction_event_id","STRING",mode="REQUIRED"),S("ledger_version","STRING",mode="REQUIRED"),S("model_version","STRING"),S("registry_sha256","STRING"),S("physical_game_id","STRING",mode="REQUIRED"),S("game_start","TIMESTAMP"),S("home_team","STRING"),S("away_team","STRING"),S("market","STRING"),S("side","STRING"),S("line","FLOAT64"),S("odds","FLOAT64"),S("quote_timestamp","TIMESTAMP"),S("prediction_timestamp","TIMESTAMP"),S("core_margin_pred","FLOAT64"),S("core_total_pred","FLOAT64"),S("h2h_probability","FLOAT64"),S("spread_consensus_edge","FLOAT64"),S("total_consensus_edge","FLOAT64"),S("spread_model_gap","FLOAT64"),S("total_model_gap","FLOAT64"),S("uncertainty_q80","FLOAT64"),S("uncertainty_q90","FLOAT64"),S("bigal_active","STRING"),S("pathi_family_opinion","STRING"),S("miner_active","STRING"),S("resolved_family_opinions","STRING"),S("snapshot_payload_json","STRING"),S("is_prospective","BOOL")]


def _schema_results():
    from google.cloud import bigquery as b; S=b.SchemaField
    return [S("result_event_id","STRING",mode="REQUIRED"),S("prediction_event_id","STRING",mode="REQUIRED"),S("settled_at","TIMESTAMP"),S("home_score","FLOAT64"),S("away_score","FLOAT64"),S("result","STRING"),S("market_error","FLOAT64"),S("core_error","FLOAT64"),S("brier","FLOAT64"),S("log_loss","FLOAT64"),S("clv_points","FLOAT64"),S("closing_line","FLOAT64"),S("closing_odds","FLOAT64"),S("closing_quote_timestamp","TIMESTAMP")]


def ensure_tables(client=None):
    from google.cloud import bigquery as b
    c=client or b.Client(project="sharplogger"); made=[]
    for table,schema in ((PRED_TABLE,_schema_predictions()),(RESULT_TABLE,_schema_results())):
        try: c.get_table(table)
        except Exception:
            t=b.Table(table,schema=schema); t.description="Append-only NFL research ledger. No production betting authority."; c.create_table(t); made.append(table)
    return {"status":"READY","prediction_table":PRED_TABLE,"result_table":RESULT_TABLE,"created":made,"ledger_version":LEDGER_VERSION,"production_authority":0}


def prepare_prediction_rows(rows:pd.DataFrame,*,model_version:str,registry_sha256:str,now=None):
    """Prepare immutable events. Rejects rows that are not genuinely pregame."""
    if rows is None or rows.empty: return pd.DataFrame(),{"status":"NO_ROWS"}
    n=pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now,utc=True)
    out=[]; rejected={}
    def rej(k): rejected[k]=rejected.get(k,0)+1
    for _,r in rows.iterrows():
        gs=pd.to_datetime(r.get("game_start",r.get("Game_Start")),errors="coerce",utc=True); qt=pd.to_datetime(r.get("quote_timestamp",r.get("Snapshot_Timestamp")),errors="coerce",utc=True)
        if pd.isna(gs) or pd.isna(qt) or not (qt<n+pd.Timedelta(minutes=1) and n<gs and qt<gs): rej("NOT_PROSPECTIVE"); continue
        gid=str(r.get("physical_game_id",r.get("Physical_Game_ID",r.get("Merge_Key_Short","")))).strip().lower()
        market=str(r.get("market",r.get("Market",""))).strip().lower(); side=str(r.get("side",r.get("Outcome",""))).strip()
        if not gid or market not in {"spreads","totals","h2h"}: rej("IDENTITY_OR_MARKET"); continue
        key=f"{LEDGER_VERSION}|{registry_sha256}|{gid}|{market}|{side}|{qt.isoformat()}"; eid=hashlib.sha256(key.encode()).hexdigest()
        def flt(x):
            try:
                z=float(x); return z if math.isfinite(z) else None
            except Exception:return None
        payload={k:(None if pd.isna(v) else v) for k,v in r.to_dict().items()}
        out.append({"prediction_event_id":eid,"ledger_version":LEDGER_VERSION,"model_version":model_version,"registry_sha256":registry_sha256,"physical_game_id":gid,"game_start":gs,"home_team":str(r.get("home_team",r.get("Home_Team",""))),"away_team":str(r.get("away_team",r.get("Away_Team",""))),"market":market,"side":side,"line":flt(r.get("line",r.get("Value"))),"odds":flt(r.get("odds",r.get("Odds_Price"))),"quote_timestamp":qt,"prediction_timestamp":n,"core_margin_pred":flt(r.get("core_margin_pred")),"core_total_pred":flt(r.get("core_total_pred")),"h2h_probability":flt(r.get("h2h_probability",r.get("h2h_prob"))),"spread_consensus_edge":flt(r.get("spread_consensus_edge")),"total_consensus_edge":flt(r.get("total_consensus_edge")),"spread_model_gap":flt(r.get("spread_model_gap")),"total_model_gap":flt(r.get("total_model_gap")),"uncertainty_q80":flt(r.get("uncertainty_q80")),"uncertainty_q90":flt(r.get("uncertainty_q90")),"bigal_active":str(r.get("bigal_active","")),"pathi_family_opinion":str(r.get("pathi_family_opinion","")),"miner_active":str(r.get("miner_active","")),"resolved_family_opinions":str(r.get("resolved_family_opinions","")),"snapshot_payload_json":json.dumps(payload,default=str,sort_keys=True),"is_prospective":True})
    return pd.DataFrame(out),{"status":"PREPARED" if out else "NO_ELIGIBLE_ROWS","prepared":len(out),"rejected":rejected}


def append_prediction_rows(rows:pd.DataFrame,*,client=None):
    if rows is None or rows.empty:return {"status":"NO_ROWS","inserted":0}
    from google.cloud import bigquery as b
    c=client or b.Client(project="sharplogger"); ensure_tables(c)
    ids=rows.prediction_event_id.astype(str).tolist(); cfg=b.QueryJobConfig(query_parameters=[b.ArrayQueryParameter("ids","STRING",ids)])
    existing=set(c.query(f"SELECT prediction_event_id FROM `{PRED_TABLE}` WHERE prediction_event_id IN UNNEST(@ids)",job_config=cfg).to_dataframe(create_bqstorage_client=False).prediction_event_id.astype(str))
    new=rows.loc[~rows.prediction_event_id.astype(str).isin(existing)].copy()
    if new.empty:return {"status":"NO_NEW_ROWS","inserted":0,"existing":len(existing)}
    recs=[]
    for rec in new.where(pd.notna(new),None).to_dict("records"):
        for k in ("game_start","quote_timestamp","prediction_timestamp"):
            v=rec.get(k)
            if isinstance(v,pd.Timestamp): rec[k]=v.isoformat()
        recs.append(rec)
    errors=c.insert_rows_json(PRED_TABLE,recs)
    if errors: raise RuntimeError("NFL_LEDGER_INSERT_FAILED "+str(errors[:3]))
    return {"status":"INSERTED","inserted":int(len(new)),"existing":int(len(existing)),"production_authority":0}
