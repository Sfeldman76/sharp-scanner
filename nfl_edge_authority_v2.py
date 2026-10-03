"""NFL Edge Authority V2 — NCAAF-method transfer.

Primary decision layer for NFL after Betting Engine V1 failed its historical
promotion gate.  V1 remains a benchmark/shadow.  V2 ports the architecture that
worked in NCAAF Production V1:

  frozen fair-value model -> conditional edge mechanisms -> family collapse ->
  independent-mechanism resolution -> historical confirmation -> prospective.

2021-2023 is the shared discovery window for newly researched NFL selectors;
2024-2025 is untouched confirmation.  Previously frozen direct-system families
are re-evaluated with their natural 2017-2022 discovery / 2023-2025 confirmation
windows.  2026 is never used for training, threshold selection, or historical
promotion decisions.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
try:
    from google.api_core.exceptions import PreconditionFailed
    from google.cloud import bigquery
except ModuleNotFoundError:
    class PreconditionFailed(Exception): pass
    bigquery=None

import nfl_production_v1 as prod
import nfl_betting_engine_v1 as benchmark
import sports_edge_authority_v1 as shared
import nfl_stat_selector_v23 as statv23
import nfl_advanced_stat_research_v24 as statv24

SOURCE_TAG="nfl-edge-authority-v2.4-advanced-stat-research-shadow-20261003"
ENGINE_VERSION="NFL_EDGE_AUTHORITY_V2_4"
PROJECT="sharplogger"; DATASET="sharp_data"
PREFIX="production/nfl/v2/edge_authority"
ARTIFACT_OBJECT=f"{PREFIX}/current_edge_engine.joblib"
META_OBJECT=f"{PREFIX}/current_contract.json"
CURRENT_OBJECT=f"{PREFIX}/current_state.json"
EVENT_PREFIX=f"{PREFIX}/events"; SETTLEMENT_PREFIX=f"{PREFIX}/settlements"
PAIRED_SETTLEMENT_TABLE=f"{PROJECT}.{DATASET}.nfl_production_v1_paired_settlements"
SYSTEM_TRIGGER_V1=f"{PROJECT}.{DATASET}.nfl_system_trigger_v1"
SYSTEM_TRIGGER_V2=f"{PROJECT}.{DATASET}.nfl_system_family_trigger_v2"

DISCOVERY_SEASONS=(2021,2022,2023)
CONFIRM_SEASONS=(2024,2025)
SYSTEM_DISCOVERY_SEASONS=(2017,2018,2019,2020,2021,2022)
SYSTEM_CONFIRM_SEASONS=(2023,2024,2025)

ROLE_FLIP_FAMILY="NFL_SPREAD_ROLE_FLIP_DOG_TO_FAVORITE_FADE"
HOME_FAV_FAMILY="NFL_SPREAD_HOME_FAVORITE_OFF_SU_LOSS_FADE"
EARLY_DIV_UNDER_FAMILY="NFL_TOTAL_EARLY_DIVISION_UNDER"

FAIR_THRESHOLDS={"SPREADS":(2.,3.,4.,5.,6.),"TOTALS":(2.,3.,4.,5.,6.),"H2H":(.02,.05,.075,.10,.15)}
MARKET_THRESHOLDS={"SPREADS":(.5,1.,1.5,2.),"TOTALS":(.5,1.,1.5,2.),"H2H":(.01,.02,.03,.05)}

FAMILY_POLICY={
    "FAIR_VALUE":{"min_discovery_n":80,"min_confirmation_n":40},
    "STAT_SELECTOR":{"min_discovery_n":60,"min_confirmation_n":30},
    "MARKET_CONFIRMATION":{"min_discovery_n":50,"min_confirmation_n":25},
    "SYSTEM":{"min_discovery_n":30,"min_confirmation_n":15},
}


def _num(x):
    try:
        z=float(x); return z if math.isfinite(z) else np.nan
    except Exception:return np.nan

def _norm(x):
    return benchmark._norm(x)

def _sha(x): return shared.sha256_json(x)

def _write_json(sc,bucket,name,payload):
    sc.bucket(bucket).blob(name).upload_from_string(json.dumps(payload,sort_keys=True,indent=2,default=str),content_type="application/json")
    return f"gs://{bucket}/{name}"

def _read_json(sc,bucket,name):
    b=sc.bucket(bucket).blob(name)
    if not b.exists(): return None
    try:return json.loads(b.download_as_text())
    except Exception:return None

def _write_immutable(sc,bucket,name,payload):
    b=sc.bucket(bucket).blob(name)
    try:
        b.upload_from_string(json.dumps(payload,sort_keys=True,indent=2,default=str),content_type="application/json",if_generation_match=0); return True
    except PreconditionFailed:return False


def _profit_american(o): return benchmark._american_profit_if_win(o)


def _record_df(d:pd.DataFrame,mask=None)->dict:
    if d is None or d.empty:return {"n":0,"graded_n":0,"wins":0,"losses":0,"pushes":0,"hit_rate":None,"roi_per_unit":None,"wilson95":[None,None]}
    q=d if mask is None else d.loc[np.asarray(mask,bool)]
    return shared.record(pd.to_numeric(q.get("target"),errors="coerce").to_numpy(float),pd.to_numeric(q.get("profit_if_win"),errors="coerce").to_numpy(float))


def _by_season(d:pd.DataFrame,mask,seasons)->dict:
    m=np.asarray(mask,bool); out={}
    for sy in seasons:
        out[str(sy)]=_record_df(d,m & pd.to_numeric(d.season,errors="coerce").eq(sy).to_numpy(bool))
    return out


def _positive_roi_seasons(by:dict)->int:
    return sum(1 for x in by.values() if math.isfinite(_num(x.get("roi_per_unit"))) and _num(x.get("roi_per_unit"))>0)


def _discovery_gate(rec:dict,by:dict,market:str,kind:str)->dict:
    p=FAMILY_POLICY[kind]; reasons=[]
    if int(rec.get("n") or 0)<p["min_discovery_n"]:reasons.append("DISCOVERY_N")
    roi=_num(rec.get("roi_per_unit")); hit=_num(rec.get("hit_rate"))
    if not math.isfinite(roi) or roi<=0:reasons.append("DISCOVERY_ROI")
    if _positive_roi_seasons(by)<2:reasons.append("DISCOVERY_SEASON_STABILITY")
    if market in {"SPREADS","TOTALS"} and (not math.isfinite(hit) or hit<=shared.BREAK_EVEN_110):reasons.append("DISCOVERY_HIT_BELOW_BREAK_EVEN")
    return {"status":"PASS" if not reasons else "HOLD","reasons":reasons}


def _final_gate(discovery,confirm,disc_by,conf_by,market,kind):
    p=FAMILY_POLICY[kind]
    return shared.family_gate(discovery,confirm,market=market,
        min_discovery_n=p["min_discovery_n"],min_confirmation_n=p["min_confirmation_n"],
        discovery_positive_seasons=_positive_roi_seasons(disc_by),confirmation_positive_seasons=_positive_roi_seasons(conf_by))


def _historical_system_family_context(bq_client)->dict[str,dict]:
    from nfl_feature_audit_v1 import VIEW
    cols=("Season","Season_Stage","Source_Name","Source_Game_ID","Game_Date","Historical_Core_Eligible",
          "Team_Norm","Opponent_Norm","Is_Home","Is_Away","Team_Score","Opponent_Score","Opening_Spread","Is_Division_Game","Week_Number")
    schema={f.name for f in bq_client.get_table(VIEW).schema}; missing=sorted(set(cols)-schema)
    if missing: raise RuntimeError("[NFL-EDGE-V2-HOLD] SYSTEM_CONTEXT_COLUMNS_MISSING "+str(missing))
    q="SELECT "+", ".join(f"`{c}`" for c in cols)+f" FROM `{VIEW}` WHERE Season BETWEEN 2017 AND 2025 AND Historical_Core_Eligible=1 AND Team_Score IS NOT NULL AND Opponent_Score IS NOT NULL ORDER BY Season, Game_Date, Source_Name, Source_Game_ID, Team_Norm"
    d=bq_client.query(q).to_dataframe(create_bqstorage_client=False)
    if d.empty: raise RuntimeError("[NFL-EDGE-V2-HOLD] SYSTEM_CONTEXT_EMPTY")
    d["Season"]=pd.to_numeric(d.Season,errors="coerce").astype(int); d["Game_Date"]=pd.to_datetime(d.Game_Date,errors="coerce")
    d["physical_game_id"]=d[["Season","Source_Name","Source_Game_ID"]].astype(str).agg("|".join,axis=1)
    d["actual_margin"]=pd.to_numeric(d.Team_Score,errors="coerce")-pd.to_numeric(d.Opponent_Score,errors="coerce")
    d["opening_dog"]=pd.to_numeric(d.Opening_Spread,errors="coerce").gt(0).astype(float)
    d=d.sort_values(["Season","Team_Norm","Game_Date","Source_Name","Source_Game_ID"],kind="mergesort")
    grp=d.groupby(["Season","Team_Norm"],sort=False,dropna=False)
    d["prev_open_dog"]=grp["opening_dog"].shift(1); d["prev_margin"]=grp["actual_margin"].shift(1)
    out={}
    for gid,g in d.groupby("physical_game_id",sort=False):
        hrows=g.loc[pd.to_numeric(g.Is_Home,errors="coerce").eq(1)]
        if hrows.empty:continue
        h=hrows.iloc[0]; home=_norm(h.Team_Norm); away=_norm(h.Opponent_Norm); votes=[]
        for _,r in g.iterrows():
            op=_num(r.get("Opening_Spread")); prevdog=_num(r.get("prev_open_dog"))
            prevmargin=_num(r.get("prev_margin"))
            if prevdog==1.0 and math.isfinite(prevmargin) and prevmargin<0 and math.isfinite(op) and op<0:
                play=_norm(r.get("Opponent_Norm")); direction=1 if play==home else -1 if play==away else 0
                if direction:votes.append({"market":"SPREADS","family":ROLE_FLIP_FAMILY,"direction":direction,"label":"Role Flip dog→favorite off-SU-loss fade"})
        hop=_num(h.get("Opening_Spread")); hprev=_num(h.get("prev_margin"))
        if math.isfinite(hop) and hop<0 and math.isfinite(hprev) and hprev<0:
            votes.append({"market":"SPREADS","family":HOME_FAV_FAMILY,"direction":-1,"label":"Home favorite off SU loss fade"})
        wk=_num(h.get("Week_Number")); div=_num(h.get("Is_Division_Game"))
        if div==1.0 and math.isfinite(wk) and 1<=wk<=4:
            votes.append({"market":"TOTALS","family":EARLY_DIV_UNDER_FAMILY,"direction":-1,"label":"Early division UNDER"})
        out[str(gid)]={"home":home,"away":away,"season":int(h.Season),"actual_margin":_num(h.actual_margin),"opening_spread":_num(h.Opening_Spread),"votes":votes}
    return out


def _attach_actuals(rows:dict[str,pd.DataFrame],replay:pd.DataFrame)->dict[str,pd.DataFrame]:
    keep=["physical_game_id","actual_margin","actual_total","close_spread","close_total","home_win_label","home_close_ml","away_close_ml"]
    base=replay.loc[:,[c for c in keep if c in replay.columns]].drop_duplicates("physical_game_id")
    return {m:d.merge(base,on="physical_game_id",how="left",validate="one_to_one") for m,d in rows.items()}


def _variant(d,market,fid,kind,variant_id,threshold,mask,disc_seasons=DISCOVERY_SEASONS,conf_seasons=CONFIRM_SEASONS,complexity=1):
    seasons=pd.to_numeric(d.season,errors="coerce").to_numpy(float); mask=np.asarray(mask,bool)
    dm=mask & np.isin(seasons,np.asarray(disc_seasons,float)); cm=mask & np.isin(seasons,np.asarray(conf_seasons,float))
    dr=_record_df(d,dm); cr=_record_df(d,cm); dby=_by_season(d,dm,disc_seasons); cby=_by_season(d,cm,conf_seasons)
    dg=_discovery_gate(dr,dby,market,kind)
    return {"mechanism_family_id":fid,"mechanism_class":kind,"market":market,"variant_id":variant_id,"threshold":threshold,"complexity":complexity,
            "discovery":dr,"confirmation":cr,"discovery_by_season":dby,"confirmation_by_season":cby,"discovery_gate":dg,
            "evidence_windows":{"discovery":list(disc_seasons),"confirmation":list(conf_seasons)}}


def _system_candidate_frame(replay:pd.DataFrame,ctx:dict,family:str,market:str)->pd.DataFrame:
    rec=[]
    for _,r in replay.iterrows():
        votes=[v for v in (ctx.get(str(r.physical_game_id),{}).get("votes") or []) if v["family"]==family and v["market"]==market]
        if not votes:continue
        direction=int(votes[0]["direction"])
        if market=="SPREADS":
            # Direct systems were frozen/validated against the opening line; preserve that contract.
            line=_num(r.get("opening_spread")); actual=_num(r.get("actual_margin")); settle=direction*(actual+line)
            if not (math.isfinite(line) and math.isfinite(actual)) or abs(settle)<=1e-12:continue
            target=float(settle>0); profit=100/110
        elif market=="TOTALS":
            # Direct systems were frozen/validated against the opening total; preserve that contract.
            line=_num(r.get("opening_total")); actual=_num(r.get("actual_total")); settle=direction*(actual-line)
            if not (math.isfinite(line) and math.isfinite(actual)) or abs(settle)<=1e-12:continue
            target=float(settle>0); profit=100/110
        else:continue
        rec.append({"physical_game_id":r.physical_game_id,"season":int(r.season),"direction":direction,"target":target,"profit_if_win":profit})
    return pd.DataFrame(rec)



def _minus110_record_from_rate(rec:dict|None)->dict:
    rec=rec or {}; n=int(rec.get("n") or 0); rate=_num(rec.get("rate"))
    wins=int(rec.get("wins") or (round(rate*n) if math.isfinite(rate) else 0))
    losses=max(0,n-wins)
    roi=(wins*(100.0/110.0)-losses)/n if n else np.nan
    return {"n":n,"graded_n":n,"wins":wins,"losses":losses,"pushes":0,
            "hit_rate":round(wins/n,6) if n else None,
            "roi_per_unit":round(float(roi),6) if n else None,
            "wilson95":rec.get("wilson95") or shared.wilson95(wins,n)}


def _system_family_variants_from_report(system_report:dict|None,log_func=print)->list[dict]:
    if not isinstance(system_report,dict):
        log_func('[NFL-EDGE-V2-SYSTEM-AUTHORITY-SOURCE] status=HOLD reason=SYSTEM_LAB_REPORT_MISSING no_system_family_may_receive_authority=TRUE')
        return []
    fams=list(system_report.get("mechanism_families") or [])
    miners=(system_report.get("miner") or {})
    rule_lookup={}
    for market in ("spreads","totals"):
        for r in ((miners.get(market) or {}).get("retained_rules") or []):
            rid="__".join(r.get("conditions") or [])
            rule_lookup[(str(r.get("market") or market).upper(),rid)]=r
    out=[]
    for fam in fams:
        if not fam.get("prospective_eligible") or not str(fam.get("family_status") or '').startswith('LEGIT'):
            continue
        market=str(fam.get("market") or '').upper()
        if market not in {"SPREADS","TOTALS"}: continue
        rid=str(fam.get("representative_rule_id") or '')
        rep=rule_lookup.get((market,rid),{})
        recs=fam.get("representative_records") or rep.get("records") or {}
        disc=_minus110_record_from_rate(recs.get("discovery"))
        conf=_minus110_record_from_rate(recs.get("validation_2023_2025"))
        dby={str(y):_minus110_record_from_rate((rep.get("discovery_year_by_year") or {}).get(str(y))) for y in SYSTEM_DISCOVERY_SEASONS}
        cby={
            "2023":_minus110_record_from_rate(recs.get("shadow")),
            "2024":_minus110_record_from_rate(recs.get("confirmation")),
            "2025":_minus110_record_from_rate(recs.get("final_2025")),
        }
        dg=_discovery_gate(disc,dby,market,"SYSTEM")
        v={"mechanism_family_id":str(fam.get("system_family_id")),"mechanism_class":"SYSTEM","market":market,
           "variant_id":rid,"threshold":None,"complexity":max(1,len(fam.get("representative_conditions") or [])),
           "discovery":disc,"confirmation":conf,"discovery_by_season":dby,"confirmation_by_season":cby,
           "discovery_gate":dg,"evidence_windows":{"discovery":list(SYSTEM_DISCOVERY_SEASONS),"confirmation":list(SYSTEM_CONFIRM_SEASONS)},
           "system_lab_family_status":fam.get("family_status"),"system_lab_prospective_action":fam.get("prospective_action"),
           "representative_conditions":fam.get("representative_conditions") or [],"representative_records":recs,
           "system_lab_registry_sha256":((system_report.get("family_registry") or {}).get("family_registry_sha256") or (system_report.get("registry") or {}).get("family_registry_sha256"))}
        out.append(v)
        log_func("[NFL-EDGE-V2-SYSTEM-AUTHORITY-SOURCE] "+json.dumps({k:v.get(k) for k in ("mechanism_family_id","market","variant_id","representative_conditions","discovery","confirmation","system_lab_family_status","system_lab_prospective_action")},sort_keys=True,default=str))
    return out



def _system_trigger_map(sysctx:dict, family_id:str)->dict[str,int]:
    out={}
    for gid,ctx in (sysctx or {}).items():
        for v in (ctx.get("votes") or []):
            if str(v.get("family"))==str(family_id) and str(v.get("market"))=="SPREADS":
                out[str(gid)]=int(v.get("direction") or 0)
                break
    return out


def _system_record_for_ids(sysctx:dict, ids:set[str], dirs:dict[str,int])->dict:
    y=[]
    for gid in sorted(ids):
        ctx=(sysctx or {}).get(str(gid)) or {}
        direction=int(dirs.get(str(gid),0) or 0)
        actual=_num(ctx.get("actual_margin")); line=_num(ctx.get("opening_spread"))
        if direction==0 or not (math.isfinite(actual) and math.isfinite(line)):
            continue
        settle=direction*(actual+line)
        if abs(settle)<=1e-12:
            y.append(.5)
        else:
            y.append(1.0 if settle>0 else 0.0)
    return shared.record(y,np.full(len(y),100.0/110.0)) if y else shared.record([])


def _pair_system_dependency(a:dict,b:dict,sysctx:dict)->dict:
    aid=str(a.get("mechanism_family_id")); bid=str(b.get("mechanism_family_id"))
    amap=_system_trigger_map(sysctx,aid); bmap=_system_trigger_map(sysctx,bid)
    aset=set(amap); bset=set(bmap); union=aset|bset; inter=aset&bset
    same={g for g in inter if int(amap.get(g,0))==int(bmap.get(g,0)) and int(amap.get(g,0))!=0}
    conflict={g for g in inter if int(amap.get(g,0))!=int(bmap.get(g,0))}
    aonly=aset-bset; bonly=bset-aset
    shared_conditions=sorted(set(a.get("representative_conditions") or []) & set(b.get("representative_conditions") or []))
    def in_seasons(ids,seasons):
        return {g for g in ids if int(((sysctx.get(g) or {}).get("season") or -1)) in set(seasons)}
    out={
        "family_a":aid,"family_b":bid,
        "shared_conditions":shared_conditions,
        "jaccard":round(len(inter)/len(union),6) if union else 0.0,
        "overlap_games":len(inter),"same_direction_overlap_games":len(same),"conflict_overlap_games":len(conflict),
    }
    for label,seasons in (("discovery",SYSTEM_DISCOVERY_SEASONS),("confirmation",SYSTEM_CONFIRM_SEASONS)):
        sa=in_seasons(same,seasons); aa=in_seasons(aonly,seasons); bb=in_seasons(bonly,seasons)
        both_dirs={g:amap[g] for g in sa}; adirs={g:amap[g] for g in aa}; bdirs={g:bmap[g] for g in bb}
        out[label]={
            "both_same_direction":_system_record_for_ids(sysctx,sa,both_dirs),
            "a_only":_system_record_for_ids(sysctx,aa,adirs),
            "b_only":_system_record_for_ids(sysctx,bb,bdirs),
            "conflict_n":len(in_seasons(conflict,seasons)),
        }
    conf=out["confirmation"]; both=conf["both_same_direction"]; ao=conf["a_only"]; bo=conf["b_only"]
    bh=_num(both.get("hit_rate")); br=_num(both.get("roi_per_unit")); ah=_num(ao.get("hit_rate")); ch=_num(bo.get("hit_rate"))
    comparators=[x for x in (ah,ch) if math.isfinite(x)]
    best_single=max(comparators) if comparators else .50
    incremental_pass=(
        int(both.get("n") or 0)>=30 and math.isfinite(br) and br>0.03
        and math.isfinite(bh) and bh>shared.BREAK_EVEN_110
        and bh>=best_single+0.025
        and _num((out["discovery"]["both_same_direction"] or {}).get("roi_per_unit"))>0
    )
    structurally_correlated=bool(shared_conditions) or out["jaccard"]>=0.20
    out["incrementality_gate"]={
        "status":"PASS" if incremental_pass else "HOLD",
        "requires_confirmation_overlap_n":30,"requires_confirmation_overlap_roi_gt":0.03,
        "requires_overlap_hit_advantage_vs_best_single":0.025,"best_single_confirmation_hit":best_single,
    }
    out["independent_for_escalation"]=bool((not structurally_correlated) or incremental_pass)
    out["dependency_reason"]=(
        "INCREMENTAL_CONFIRMATION_OVERRIDES_SHARED_PARENT" if structurally_correlated and incremental_pass
        else "CORRELATED_SHARED_PARENT_OR_OVERLAP" if structurally_correlated
        else "DISTINCT_MECHANISMS"
    )
    return out


def _apply_system_dependency_audit(families:list[dict],sysctx:dict,log_func=print)->tuple[list[dict],dict]:
    fams=[dict(f) for f in families]
    for f in fams:
        f["independence_key"]=str(f.get("mechanism_family_id"))
        if f.get("mechanism_class")=="STAT_SELECTOR":
            f["independence_key"]=f"NFL_{f.get('market')}_STAT_SELECTOR"
        elif f.get("mechanism_class")=="FAIR_VALUE":
            f["independence_key"]=f"NFL_{f.get('market')}_FAIR_VALUE"
        elif f.get("mechanism_class")=="MARKET_CONFIRMATION":
            f["independence_key"]=f"NFL_{f.get('market')}_MARKET_CONFIRMATION"
    systems=[f for f in fams if f.get("mechanism_class")=="SYSTEM" and f.get("production_authority") and f.get("market")=="SPREADS"]
    parent={str(f["mechanism_family_id"]):str(f["mechanism_family_id"]) for f in systems}
    def find(x):
        while parent.get(x,x)!=x:
            parent[x]=parent.get(parent[x],parent[x]);x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[max(ra,rb)]=min(ra,rb)
    pairs=[]
    for i,a in enumerate(systems):
        for b in systems[i+1:]:
            d=_pair_system_dependency(a,b,sysctx);pairs.append(d)
            if not d.get("independent_for_escalation"):
                union(str(a["mechanism_family_id"]),str(b["mechanism_family_id"]))
            log_func("[NFL-EDGE-V23-SYSTEM-DEPENDENCY] "+json.dumps(d,sort_keys=True,default=str))
    clusters={}
    for fid in parent:
        clusters.setdefault(find(fid),[]).append(fid)
    cluster_key={}
    for root,members in clusters.items():
        members=sorted(members)
        key=members[0] if len(members)==1 else "NFL_SPREAD_SYSTEM_CLUSTER_"+hashlib.sha256("|".join(members).encode()).hexdigest()[:12]
        for fid in members:cluster_key[fid]=key
    for f in fams:
        fid=str(f.get("mechanism_family_id"))
        if fid in cluster_key:f["independence_key"]=cluster_key[fid]
    report={"pairs":pairs,"clusters":clusters,"independence_key_by_family":{str(f.get("mechanism_family_id")):f.get("independence_key") for f in fams}}
    log_func("[NFL-EDGE-V23-INDEPENDENCE-MAP] "+json.dumps(report,sort_keys=True,default=str))
    return fams,report

def _core_perf_line(replay:pd.DataFrame,market:str,season=None)->dict:
    q=replay if season is None else replay.loc[pd.to_numeric(replay.season,errors="coerce").eq(int(season))]
    if market=="SPREADS":
        a=pd.to_numeric(q.actual_margin,errors="coerce").to_numpy(float); p=pd.to_numeric(q.frozen_fair_margin,errors="coerce").to_numpy(float)
        m=np.isfinite(a)&np.isfinite(p)
        return {"n":int(m.sum()),"mae":round(float(np.mean(np.abs(a[m]-p[m]))),6) if m.any() else None,"rmse":round(float(np.sqrt(np.mean((a[m]-p[m])**2))),6) if m.any() else None,"corr":round(float(np.corrcoef(a[m],p[m])[0,1]),6) if m.sum()>2 else None}
    if market=="TOTALS":
        a=pd.to_numeric(q.actual_total,errors="coerce").to_numpy(float); p=pd.to_numeric(q.frozen_fair_total,errors="coerce").to_numpy(float)
        m=np.isfinite(a)&np.isfinite(p)
        return {"n":int(m.sum()),"mae":round(float(np.mean(np.abs(a[m]-p[m]))),6) if m.any() else None,"rmse":round(float(np.sqrt(np.mean((a[m]-p[m])**2))),6) if m.any() else None,"corr":round(float(np.corrcoef(a[m],p[m])[0,1]),6) if m.sum()>2 else None}
    y=pd.to_numeric(q.home_win_label,errors="coerce").to_numpy(float); p=pd.to_numeric(q.frozen_home_win_probability,errors="coerce").to_numpy(float); m=np.isfinite(y)&np.isfinite(p)
    if not m.any(): return {"n":0}
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    return {"n":int(m.sum()),"auc":round(float(roc_auc_score(y[m],p[m])),6) if len(np.unique(y[m]))>1 else None,"log_loss":round(float(log_loss(y[m],np.clip(p[m],1e-6,1-1e-6))),6),"brier":round(float(brier_score_loss(y[m],p[m])),6)}


def _build_research_summary(replay_rows:pd.DataFrame,research_report:dict|None,system_report:dict|None)->dict:
    _all_model_seasons=tuple(sorted(set(DISCOVERY_SEASONS+CONFIRM_SEASONS)))
    core={m:{"overall":_core_perf_line(replay_rows,m),"by_season":{str(y):_core_perf_line(replay_rows,m,y) for y in _all_model_seasons}} for m in ("SPREADS","H2H","TOTALS")}
    structured=((research_report or {}).get("structured") or {})
    stat={}
    for m in ("spreads","totals"):
        rr=((structured.get("residual") or {}).get(m) or {})
        stat[m.upper()]={"selected_families":rr.get("prospective_freeze_selected_families") or [],"families":rr.get("families") or {},"nested_stable_blend":rr.get("nested_stable_blend") or {},"selection_detail":rr.get("prospective_selection_detail") or {},"nested_selection":rr.get("nested_selection") or {}}
    _sm=(system_report or {}).get("miner") or {}
    systems={
        "registry":(system_report or {}).get("registry") or {},
        "mechanism_families":(system_report or {}).get("mechanism_families") or [],
        "search_summary":{
            m.upper():{
                "raw_tested":((_sm.get(m) or {}).get("raw_tested")),
                "unique_hypotheses":((_sm.get(m) or {}).get("unique_hypotheses")),
                "retained_rules":len(((_sm.get(m) or {}).get("retained_rules") or [])),
            } for m in ("spreads","totals")
        },
    }
    return {
        "core_contract":(prod.production_contract().get("backbones") or {}),
        "core":core,"core_challengers":structured.get("core_challengers") or {},"stat":stat,"systems":systems,
        "core_stat_disagreement":structured.get("disagreement") or {},
        "residual_miner":((research_report or {}).get("miner") or {}),"edge_gate":((research_report or {}).get("edge_gate") or {})
    }


def _emit_full_research_diagnostics(replay_rows:pd.DataFrame,research_report:dict|None,system_report:dict|None,summary:dict,log_func=print):
    pc=prod.production_contract(); backs=pc.get("backbones") or {}
    for market in ("SPREADS","H2H","TOTALS"):
        b=backs.get(market) or backs.get(market.lower()) or {}
        log_func("[NFL-EDGE-V2-CORE-CONTRACT] "+json.dumps({"market":market,"family":b.get("family"),"target":b.get("target"),"features":b.get("features") or [],"ridge_alpha":b.get("ridge_alpha"),"logistic_C":b.get("logistic_C"),"role":"FROZEN_FAIR_VALUE_BACKBONE_MARKET_INDEPENDENT_CURRENT_GAME"},sort_keys=True,default=str))
        _cp=((summary.get("core") or {}).get(market) or {})
        log_func("[NFL-EDGE-V2-CORE-PERFORMANCE] "+json.dumps({"market":market,"sample":"ALL_2021_2025",**(_cp.get("overall") or {})},sort_keys=True,default=str))
        for _sy,_met in (_cp.get("by_season") or {}).items():
            log_func("[NFL-EDGE-V2-CORE-PERFORMANCE] "+json.dumps({"market":market,"sample":str(_sy),**(_met or {})},sort_keys=True,default=str))
    for market,models in ((summary.get("core_challengers") or {}).items()):
        for name,met in (models or {}).items():
            log_func("[NFL-EDGE-V2-CORE-CHALLENGER] "+json.dumps({"market":market.upper(),"model":name,**(met or {})},sort_keys=True,default=str))
    for market,block in (summary.get("stat") or {}).items():
        selected=set(block.get("selected_families") or [])
        for fam,met in (block.get("families") or {}).items():
            log_func("[NFL-EDGE-V2-STAT-FAMILY] "+json.dumps({"market":market,"family":fam,"selected_for_prospective_shadow":fam in selected,**(met or {})},sort_keys=True,default=str))
        log_func("[NFL-EDGE-V2-STAT-SELECTION] "+json.dumps({"market":market,"selected_families":sorted(selected),"nested_stable_blend":block.get("nested_stable_blend") or {},"selection_detail":block.get("selection_detail") or {},"nested_selection":block.get("nested_selection") or {},"role":"MARKET_RESIDUAL_SELECTOR_NOT_CORE_REWRITE"},sort_keys=True,default=str))
    disagreement=summary.get("core_stat_disagreement") or {}
    for market,block in disagreement.items():
        if market=="production_authority" or not isinstance(block,dict): continue
        log_func("[NFL-EDGE-V2-CORE-STAT-DISAGREEMENT] "+json.dumps({"market":str(market).upper(),**block},sort_keys=True,default=str))
    edge_gate=summary.get("edge_gate") or {}
    for market,block in ((edge_gate.get("markets") or {}).items()):
        log_func("[NFL-EDGE-V2-EDGE-GATE] "+json.dumps({
            "market":str(market).upper(),
            "fair_line_scorecard":(block or {}).get("fair_line_scorecard") or {},
            "transparent_gates":(block or {}).get("transparent_gates") or {},
            "core_gate_probability":(block or {}).get("core_gate_probability") or {},
            "core_gate_by_season":(block or {}).get("core_gate_by_season") or {},
            "core_gate_fixed_probability_bands":(block or {}).get("core_gate_fixed_probability_bands") or {},
            "consensus_gate_probability":(block or {}).get("consensus_gate_probability") or {},
            "consensus_gate_by_season":(block or {}).get("consensus_gate_by_season") or {},
            "prospective_selected_residual_families":(block or {}).get("prospective_selected_residual_families") or [],
            "prospective_consensus_gate_applicable":bool((block or {}).get("prospective_consensus_gate_applicable")),
            "selection_policy":(block or {}).get("selection_policy"),
        },sort_keys=True,default=str))
    miner=summary.get("residual_miner") or {}
    for market,mr in ((miner.get("markets") or {}).items()):
        log_func("[NFL-EDGE-V2-RESIDUAL-MINER-SUMMARY] "+json.dumps({"market":market.upper(),"candidate_conditions":mr.get("candidate_conditions"),"single_discovery_fdr_pass":mr.get("single_discovery_fdr_pass"),"pair_discovery_fdr_pass":mr.get("pair_discovery_fdr_pass"),"validated_rule_count":mr.get("validated_rule_count"),"promising_count":mr.get("promising_count"),"selection_basis":mr.get("selection_basis")},sort_keys=True,default=str))
        for r in (mr.get("promising_rules") or [])[:20]:
            log_func("[NFL-EDGE-V2-RESIDUAL-MINER-TOP] "+json.dumps({"market":market.upper(),"rule_id":r.get("rule_id"),"type":r.get("type"),"direction":r.get("direction"),"conditions":r.get("conditions"),"discovery":r.get("discovery"),"shadow":r.get("shadow"),"confirm":r.get("confirm"),"same_direction_all_splits":r.get("same_direction_all_splits")},sort_keys=True,default=str))
    sr=summary.get("systems") or {}; reg=sr.get("registry") or {}
    _sm=(system_report or {}).get("miner") or {}
    log_func("[NFL-EDGE-V2-SYSTEM-MINER-SUMMARY] "+json.dumps({
        "status":(system_report or {}).get("status"),
        "spread_raw_tested":((_sm.get("spreads") or {}).get("raw_tested")),
        "spread_unique_hypotheses":((_sm.get("spreads") or {}).get("unique_hypotheses")),
        "spread_retained":len(((_sm.get("spreads") or {}).get("retained_rules") or [])),
        "spread_legit":len(reg.get("spread_legit_rule_ids") or []),"spread_promising":len(reg.get("spread_promising_rule_ids") or []),"spread_watch":len(reg.get("spread_watch_rule_ids") or []),
        "total_raw_tested":((_sm.get("totals") or {}).get("raw_tested")),
        "total_unique_hypotheses":((_sm.get("totals") or {}).get("unique_hypotheses")),
        "total_retained":len(((_sm.get("totals") or {}).get("retained_rules") or [])),
        "total_legit":len(reg.get("total_legit_rule_ids") or []),"total_promising":len(reg.get("total_promising_rule_ids") or []),"total_watch":len(reg.get("total_watch_rule_ids") or []),
        "mechanism_families":len(sr.get("mechanism_families") or []),"prospective_family_ids":reg.get("prospective_family_ids") or [],"family_registry_sha256":reg.get("family_registry_sha256")
    },sort_keys=True,default=str))
    for fam in sr.get("mechanism_families") or []:
        log_func("[NFL-EDGE-V2-SYSTEM-FAMILY-DETAIL] "+json.dumps({k:fam.get(k) for k in ("system_family_id","market","direction","family_status","member_count","representative_rule_id","representative_conditions","representative_records","strongest_validation_rule_id","strongest_validation_records","prospective_action","prospective_eligible")},sort_keys=True,default=str))
    for market,mr in (((system_report or {}).get("miner") or {}).items()):
        for r in (mr.get("retained_rules") or [])[:25]:
            log_func("[NFL-EDGE-V2-SYSTEM-TOP] "+json.dumps({"market":market.upper(),"rank":r.get("rank"),"status":r.get("status"),"rule_id":"__".join(r.get("conditions") or []),"conditions":r.get("conditions"),"families":r.get("families"),"direction":r.get("direction"),"records":r.get("records"),"min_fold_rate":r.get("min_fold_rate"),"min_loso_rate":r.get("min_loso_rate"),"remove_best":r.get("remove_best_season_record"),"bootstrap_ci95":r.get("bootstrap_ci95"),"maxstat_p":r.get("permutation_max_pvalue"),"hierarchical_fdr_q":r.get("hierarchical_family_qvalue")},sort_keys=True,default=str))

def _research_families(*,bq_client,replay_rows,games,log_func=print,research_report=None,system_report=None):
    base=benchmark.build_historical_engine_rows(bq_client=bq_client,replay_rows=replay_rows,games=games)
    base=_attach_actuals(base,replay_rows)
    sysctx=_historical_system_family_context(bq_client)
    variants=[]
    # FAIR_VALUE + MARKET_CONFIRMATION use frozen CORE direction. These remain
    # interpretable benchmarks and are not allowed to rewrite fair-value models.
    for market in ("SPREADS","H2H","TOTALS"):
        d=base[market]
        for t in FAIR_THRESHOLDS[market]:
            variants.append(_variant(d,market,f"NFL_{market}_FAIR_VALUE_EDGE","FAIR_VALUE",f"EDGE_GE_{t}",t,pd.to_numeric(d.raw_edge,errors="coerce").ge(t).to_numpy(bool)))
        move_col="market_prob_move_toward_selected" if market=="H2H" else "line_move_toward_selected"
        for t in MARKET_THRESHOLDS[market]:
            variants.append(_variant(d,market,f"NFL_{market}_MARKET_CONFIRMATION","MARKET_CONFIRMATION",f"MOVE_GE_{t}",t,pd.to_numeric(d.get(move_col),errors="coerce").ge(t).to_numpy(bool)))

    # SYSTEM evidence comes from the authoritative full-history System Miner V3
    # mechanism registry, never reconstructed from the shorter production replay.
    variants.extend(_system_family_variants_from_report(system_report,log_func=log_func))
    collapsed=shared.collapse_family_variants(variants)
    families=[]
    for f in collapsed:
        gate=_final_gate(f.get("discovery") or {},f.get("confirmation") or {},f.get("discovery_by_season") or {},f.get("confirmation_by_season") or {},f["market"],f["mechanism_class"])
        f={**f,"confirmation_gate":gate,"family_status":"LEGIT_FAMILY_REQUIRES_PROSPECTIVE" if gate["status"]=="PASS" else "HOLD_HISTORICAL","production_authority":bool(gate["status"]=="PASS")}
        families.append(f)
        log_func("[NFL-EDGE-V2-FAMILY] "+json.dumps({k:f.get(k) for k in ("mechanism_family_id","mechanism_class","market","variant_id","threshold","discovery","confirmation","discovery_gate","confirmation_gate","family_status")},sort_keys=True,default=str))

    # NCAAF-method STAT layer: rich prior-only statistical families predict market
    # residual reliability. Discovery fixes representatives; confirmation only validates.
    stat_report,stat_bundles,stat_lookup=statv23.run_historical_selector_research(
        bq_client=bq_client,games=games,replay_rows=replay_rows,log_func=log_func,
    )
    for market in ("SPREADS","TOTALS"):
        block=((stat_report.get("markets") or {}).get(market) or {})
        for c in (block.get("preregistered") or []):
            rank=int(c.get("preregistered_rank") or 0)
            group=str(c.get("dependency_group") or c.get("family") or "STAT")
            fid=f"NFL_{market}_STAT_V23_{group}_{rank}"
            passed=(c.get("confirmation_gate") or {}).get("status")=="PASS"
            f={
                "mechanism_family_id":fid,"mechanism_class":"STAT_SELECTOR","market":market,
                "variant_id":c.get("selector_id"),"selector_id":c.get("selector_id"),
                "threshold":c.get("selector_threshold"),"selector_threshold":c.get("selector_threshold"),
                "core_gap_min":c.get("core_gap_min"),"alpha":c.get("alpha"),"stat_family":c.get("family"),
                "stat_dependency_group":group,"complexity":1,
                "discovery":c.get("discovery") or {},"confirmation":c.get("confirmation") or {},
                "discovery_by_season":c.get("discovery_by_season") or {},"confirmation_by_season":c.get("confirmation_by_season") or {},
                "discovery_gate":c.get("discovery_gate") or {},"confirmation_gate":c.get("confirmation_gate") or {},
                "family_status":"LEGIT_FAMILY_REQUIRES_PROSPECTIVE" if passed else "HOLD_CONFIRMATION",
                "production_authority":bool(passed),
                "evidence_windows":{"discovery":list(DISCOVERY_SEASONS),"confirmation":list(CONFIRM_SEASONS)},
                "correlation_policy":"ALL_VALIDATED_STAT_SELECTORS_COLLAPSE_TO_ONE_STAT_MECHANISM",
            }
            families.append(f)
            log_func("[NFL-EDGE-V2-FAMILY] "+json.dumps({k:f.get(k) for k in ("mechanism_family_id","mechanism_class","market","variant_id","threshold","core_gap_min","stat_family","stat_dependency_group","discovery","confirmation","discovery_gate","confirmation_gate","family_status")},sort_keys=True,default=str))

    # Advanced V2.4 STAT research is deliberately research-only. It orthogonalizes
    # football-performance statistics against the frozen CORE feature set and tests
    # both unexplained MARKET residual and unexplained CORE outcome error, then
    # evaluates CREATE / CONFIRM / VETO / SYSTEM_FILTER roles. No V2.4 family can enter
    # betting authority from this retrospective run.
    advanced_stat_report=statv24.run_advanced_stat_research(
        bq_client=bq_client,games=games,replay_rows=replay_rows,sysctx=sysctx,log_func=log_func,
    )

    # Parent/child and overlap audit is applied *after* historical family gates.
    # Correlated system families receive one shared independence key unless their
    # overlap demonstrates predeclared incremental confirmation.
    families,dependency_report=_apply_system_dependency_audit(families,sysctx,log_func=log_func)
    return families,stat_bundles,stat_lookup,base,sysctx,stat_report,advanced_stat_report,dependency_report


def _family_lookup(families):return {f["mechanism_family_id"]:f for f in families if f.get("production_authority")}


def _historical_votes_for_game(market,row,families,stat_lookup,sysctx):
    legit=_family_lookup(families); votes=[]; gid=str(row.physical_game_id)
    core_dir=1 if ((market=="SPREADS" and _num(row.get("selected_is_home"))==1) or (market=="TOTALS" and _num(row.get("selected_over"))==1) or (market=="H2H" and _num(row.get("selected_is_home"))==1)) else -1
    fid=f"NFL_{market}_FAIR_VALUE_EDGE"
    if fid in legit and _num(row.get("raw_edge"))>=_num(legit[fid].get("threshold")):
        votes.append({"mechanism_family_id":fid,"independence_key":legit[fid].get("independence_key",fid),"direction":core_dir,"label":"Fair-value edge"})
    fid=f"NFL_{market}_MARKET_CONFIRMATION"; move_col="market_prob_move_toward_selected" if market=="H2H" else "line_move_toward_selected"
    if fid in legit and _num(row.get(move_col))>=_num(legit[fid].get("threshold")):
        votes.append({"mechanism_family_id":fid,"independence_key":legit[fid].get("independence_key",fid),"direction":core_dir,"label":"Market confirmation"})
    if market in {"SPREADS","TOTALS"}:
        for sfid,f in legit.items():
            if f.get("market")!=market or f.get("mechanism_class")!="STAT_SELECTOR":continue
            sid=str(f.get("selector_id") or f.get("variant_id") or "")
            oo=(((stat_lookup or {}).get(market) or {}).get(sid) or {}).get(gid)
            if not oo:continue
            pred=_num(oo.get("selector_prediction")); scaled=_num(oo.get("selector_scaled"))
            sdir=int(np.sign(pred)) if math.isfinite(pred) else 0
            if (sdir==core_dir and math.isfinite(scaled) and scaled>=_num(f.get("selector_threshold"))
                    and _num(row.get("raw_edge"))>=_num(f.get("core_gap_min"))):
                votes.append({"mechanism_family_id":sfid,"independence_key":f.get("independence_key",f"NFL_{market}_STAT_SELECTOR"),"direction":core_dir,
                              "label":f"STAT {f.get('stat_family')}","selector_id":sid,"selector_scaled":scaled})
    for v in (sysctx.get(gid,{}).get("votes") or []):
        if v["market"]==market and v["family"] in legit:
            f=legit[v["family"]]
            votes.append({"mechanism_family_id":v["family"],"independence_key":f.get("independence_key",v["family"]),"direction":int(v["direction"]),"label":v.get("label",v["family"])})
    return votes


def _resolved_performance(base,families,stat_lookup,sysctx,market,seasons):
    d=base[market]; rec=[]
    for _,r in d.iterrows():
        if int(r.season) not in seasons:continue
        votes=_historical_votes_for_game(market,r,families,stat_lookup,sysctx); res=shared.resolve_votes(votes,strong_play_allowed=True)
        if res["direction"]==0:continue
        core_dir=1 if ((market=="SPREADS" and _num(r.get("selected_is_home"))==1) or (market=="TOTALS" and _num(r.get("selected_over"))==1) or (market=="H2H" and _num(r.get("selected_is_home"))==1)) else -1
        # If resolved direction differs from CORE, invert target/profit only for spread/totals (-110).
        if int(res["direction"])==core_dir:
            target=_num(r.get("target")); profit=_num(r.get("profit_if_win"))
        elif market in {"SPREADS","TOTALS"}:
            target=1.0-_num(r.get("target")); profit=100/110
        else:
            # H2H direct non-core votes are not currently generated.
            continue
        rec.append({"season":int(r.season),"decision":res["decision"],"target":target,"profit_if_win":profit,"n_mech":res["independent_mechanisms"]})
    x=pd.DataFrame(rec)
    out={}
    for key,mask in (("ALL",np.ones(len(x),bool) if len(x) else np.array([],bool)),("EDGE_SINGLE",x.n_mech.eq(1).to_numpy(bool) if len(x) else np.array([],bool)),("EDGE_MULTI",x.n_mech.ge(2).to_numpy(bool) if len(x) else np.array([],bool))):
        out[key]=_record_df(x,mask) if len(x) else {"n":0}
    out["by_season"]={str(s):_record_df(x,x.season.eq(s).to_numpy(bool)) if len(x) else {"n":0} for s in seasons}
    return out


def train_publish_edge_authority(*,bq_client,storage_client,bucket_name:str,replay_rows:pd.DataFrame,games:pd.DataFrame,research_report=None,system_report=None,log_func=print)->dict:
    log_func("[NFL-EDGE-V2-PREFLIGHT] "+json.dumps({
        "status":"START","source_tag":SOURCE_TAG,"shared_framework":shared.framework_contract(),
        "discovery_seasons":list(DISCOVERY_SEASONS),"confirmation_seasons":list(CONFIRM_SEASONS),
        "year_2026_queried":False,"betting_engine_v1_role":"BENCHMARK_SHADOW",
        "stat_selector_source_tag":statv23.SOURCE_TAG,"advanced_stat_source_tag":statv24.SOURCE_TAG,
    },sort_keys=True))
    families,stat_bundles,stat_lookup,base,sysctx,stat_report,advanced_stat_report,dependency_report=_research_families(
        bq_client=bq_client,replay_rows=replay_rows,games=games,research_report=research_report,
        system_report=system_report,log_func=log_func,
    )
    research_summary=_build_research_summary(replay_rows,research_report,system_report)
    research_summary["stat_selector_v23"]=stat_report
    research_summary["advanced_stat_v24"]=advanced_stat_report
    research_summary["pbp_advanced_diagnostic_v24"]=(research_report or {}).get("pbp_advanced_diagnostic_v24") or {}
    research_summary["system_dependency_audit_v23"]=dependency_report
    _emit_full_research_diagnostics(replay_rows,research_report,system_report,research_summary,log_func=log_func)

    action_perf={}; market_contract={}
    for market in ("SPREADS","H2H","TOTALS"):
        disc=_resolved_performance(base,families,stat_lookup,sysctx,market,DISCOVERY_SEASONS)
        conf=_resolved_performance(base,families,stat_lookup,sysctx,market,CONFIRM_SEASONS)
        dmulti=disc.get("EDGE_MULTI") or {"n":0}; cmulti=conf.get("EDGE_MULTI") or {"n":0}
        strong_allowed=bool(
            int(dmulti.get("n") or 0)>=30 and _num(dmulti.get("roi_per_unit"))>0
            and int(cmulti.get("n") or 0)>=20 and _num(cmulti.get("roi_per_unit"))>0
            and (market=="H2H" or (_num(cmulti.get("hit_rate"))>shared.BREAK_EVEN_110 and _num(dmulti.get("hit_rate"))>shared.BREAK_EVEN_110))
        )
        legit=[f for f in families if f.get("market")==market and f.get("production_authority")]
        keys=sorted(set(str(f.get("independence_key") or f.get("mechanism_family_id")) for f in legit))
        market_contract[market]={
            "production_authority":bool(legit),"legit_family_ids":[f["mechanism_family_id"] for f in legit],
            "independence_keys":keys,"independent_mechanism_count":len(keys),
            "strong_play_allowed":strong_allowed,
            "policy":"FAMILY_COLLAPSED_EDGE_AUTHORITY" if legit else "MODEL_ONLY",
        }
        action_perf[market]={"discovery":disc,"confirmation":conf}
        log_func("[NFL-EDGE-V2-ACTION-STATE] "+json.dumps({
            "market":market,"legit_families":market_contract[market]["legit_family_ids"],
            "independence_keys":keys,"strong_play_allowed":strong_allowed,
            "strong_play_gate":{"discovery_multi_min_n":30,"confirmation_multi_min_n":20,"requires_positive_roi_both":True},
            "discovery":disc,"confirmation":conf,
        },sort_keys=True,default=str))
    stable_families=[]
    for f in families:
        stable_families.append({k:v for k,v in f.items() if k not in {"model","scaler","pipeline"}})
    contract={
        "status":"NFL_EDGE_AUTHORITY_V2_ACTIVE","source_tag":SOURCE_TAG,"engine_version":ENGINE_VERSION,
        "framework":shared.framework_contract(),"production_contract_sha256":prod.production_contract()["contract_sha256"],
        "discovery_seasons":list(DISCOVERY_SEASONS),"confirmation_seasons":list(CONFIRM_SEASONS),
        "system_evidence_windows":{"discovery":list(SYSTEM_DISCOVERY_SEASONS),"confirmation":list(SYSTEM_CONFIRM_SEASONS)},
        "year_2026_queried":False,"families":stable_families,"markets":market_contract,
        "historical_action_performance":action_perf,"research_summary":research_summary,
        "stat_selector_contract_sha256":stat_report.get("selector_contract_sha256"),
        "advanced_stat_research_contract_sha256":advanced_stat_report.get("research_contract_sha256"),
        "system_dependency_audit":dependency_report,
        "betting_engine_v1_role":"BENCHMARK_SHADOW","automatic_execution":False,"automatic_model_promotion":False,
    }
    fingerprint=_sha({
        "framework":contract["framework"]["contract_sha256"],
        "stat_selector":contract.get("stat_selector_contract_sha256"),
        "advanced_stat_research":contract.get("advanced_stat_research_contract_sha256"),
        "families":[{k:f.get(k) for k in ("mechanism_family_id","variant_id","threshold","core_gap_min","family_status","independence_key")} for f in stable_families],
        "markets":market_contract,
    })
    contract["contract_sha256"]=fingerprint
    obj={"metadata":contract,"stat_selector_bundles":stat_bundles}
    bio=io.BytesIO(); joblib.dump(obj,bio,compress=3); payload=bio.getvalue(); artifact_sha=hashlib.sha256(payload).hexdigest()
    storage_client.bucket(bucket_name).blob(ARTIFACT_OBJECT).upload_from_string(payload,content_type="application/octet-stream")
    contract["artifact_sha256"]=artifact_sha; contract["artifact_uri"]=f"gs://{bucket_name}/{ARTIFACT_OBJECT}"
    meta_uri=_write_json(storage_client,bucket_name,META_OBJECT,contract); contract["meta_uri"]=meta_uri
    log_func("[NFL-EDGE-V2-PUBLISH] "+json.dumps({
        "status":contract["status"],"contract_sha256":fingerprint,"artifact_sha256":artifact_sha,"artifact_uri":contract["artifact_uri"],
        "market_contract":market_contract,"stat_selector_contract_sha256":contract.get("stat_selector_contract_sha256"),
        "advanced_stat_research_contract_sha256":contract.get("advanced_stat_research_contract_sha256"),
        "betting_engine_v1_role":"BENCHMARK_SHADOW",
    },sort_keys=True,default=str))
    return contract


def _load_engine(sc,bucket):
    meta=_read_json(sc,bucket,META_OBJECT)
    if not isinstance(meta,dict) or meta.get("source_tag")!=SOURCE_TAG:raise RuntimeError("[NFL-EDGE-V2-HOLD] CONTRACT_NOT_PUBLISHED")
    b=sc.bucket(bucket).blob(ARTIFACT_OBJECT)
    if not b.exists():raise RuntimeError("[NFL-EDGE-V2-HOLD] ARTIFACT_MISSING")
    raw=b.download_as_bytes()
    if hashlib.sha256(raw).hexdigest()!=meta.get("artifact_sha256"):raise RuntimeError("[NFL-EDGE-V2-HOLD] ARTIFACT_SHA_MISMATCH")
    obj=joblib.load(io.BytesIO(raw))
    if (obj.get("metadata") or {}).get("contract_sha256")!=meta.get("contract_sha256"):raise RuntimeError("[NFL-EDGE-V2-HOLD] CONTRACT_SHA_MISMATCH")
    return obj


def _live_system_events(client,now,lookahead_days=8):
    end=pd.to_datetime(now,utc=True)+pd.Timedelta(days=lookahead_days); out={}
    for table in (SYSTEM_TRIGGER_V1,SYSTEM_TRIGGER_V2):
        try:
            q=f"SELECT * FROM `{table}` WHERE game_start>=@lo AND game_start<=@hi ORDER BY captured_at"
            cfg=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("lo","TIMESTAMP",(pd.to_datetime(now,utc=True)-pd.Timedelta(days=1)).to_pydatetime()),bigquery.ScalarQueryParameter("hi","TIMESTAMP",end.to_pydatetime())])
            d=client.query(q,job_config=cfg).to_dataframe(create_bqstorage_client=False)
        except Exception:continue
        for _,r in d.iterrows():
            gs=pd.to_datetime(r.get("game_start"),utc=True,errors="coerce")
            if pd.isna(gs):continue
            hk=_norm(r.get("home_team"));ak=_norm(r.get("away_team"));k=(gs.round("s").isoformat(),hk,ak)
            fam=str(r.get("system_family_id") or ""); market=str(r.get("market") or "SPREADS").upper(); direction=str(r.get("direction") or "").upper(); play=_norm(r.get("play_team")); target=_norm(r.get("target_team"))
            if not play and target: play=ak if direction=="FADE" and target==hk else hk if direction=="FADE" and target==ak else target
            vote=0
            if market=="SPREADS": vote=1 if play==hk else -1 if play==ak else 0
            elif market=="TOTALS": vote=-1 if direction=="UNDER" else 1 if direction=="OVER" else 0
            if fam and vote:out.setdefault(k,[]).append({"market":market,"family":fam,"direction":vote,"label":fam})
    return out


def _parse_features(p):return benchmark._parse_features(p)


def _market_base_row(p,market,q):
    f=_parse_features(p); home=p.get("home_team");away=p.get("away_team")
    if market=="SPREADS":
        line=_num(q.get("current_home_spread"));op=_num(q.get("open_home_spread"));fair=_num(p.get("champion_fair_margin"))
        if not (math.isfinite(line) and math.isfinite(fair)):return None
        edge=fair+line; core=1 if edge>0 else -1 if edge<0 else 0; odds=_num(q.get("current_home_spread_odds" if core>0 else "current_away_spread_odds"))
        return {"core_direction":core,"raw_edge":abs(edge),"move_toward_core":-core*(line-op) if math.isfinite(op) else np.nan,"selected":home if core>0 else away,"market_value":line if core>0 else -line,"model_value":fair*core,"selected_price":odds,"features":f}
    if market=="TOTALS":
        line=_num(q.get("current_total"));op=_num(q.get("open_total"));fair=_num(p.get("champion_fair_total"))
        if not (math.isfinite(line) and math.isfinite(fair)):return None
        edge=fair-line;core=1 if edge>0 else -1 if edge<0 else 0;odds=_num(q.get("current_over_odds" if core>0 else "current_under_odds"))
        return {"core_direction":core,"raw_edge":abs(edge),"move_toward_core":core*(line-op) if math.isfinite(op) else np.nan,"selected":"OVER" if core>0 else "UNDER","market_value":line,"model_value":fair,"selected_price":odds,"features":f}
    if market=="H2H":
        ref=_num(q.get("current_home_novig_probability"));op=_num(q.get("open_home_novig_probability"));fair=_num(p.get("champion_home_win_probability"))
        if not (math.isfinite(ref) and math.isfinite(fair)):return None
        edge=fair-ref;core=1 if edge>0 else -1 if edge<0 else 0;odds=_num(q.get("current_home_ml" if core>0 else "current_away_ml"))
        return {"core_direction":core,"raw_edge":abs(edge),"move_toward_core":core*(ref-op) if math.isfinite(op) else np.nan,"selected":home if core>0 else away,"market_value":ref if core>0 else 1-ref,"model_value":fair if core>0 else 1-fair,"selected_price":odds,"features":f}
    return None


def score_live(*,bq_client,storage_client,bucket_name,prediction_rows,now):
    eng=_load_engine(storage_client,bucket_name); meta=eng["metadata"]
    stat_bundles=eng.get("stat_selector_bundles") or {}
    legit={f["mechanism_family_id"]:f for f in meta.get("families",[]) if f.get("production_authority")}
    quotes=benchmark._consensus_market(bq_client,now); systems=_live_system_events(bq_client,now); out=[]
    stat_legit=[f for f in legit.values() if f.get("mechanism_class")=="STAT_SELECTOR"]
    stat_features={}; stat_live_meta={"status":"NOT_REQUIRED_NO_VALIDATED_STAT_SELECTOR"}
    if stat_legit:
        try:
            stat_features,stat_live_meta=statv23.build_live_feature_map(bq_client=bq_client,prediction_rows=prediction_rows)
        except Exception as exc:
            stat_features={};stat_live_meta={"status":"HOLD_LIVE_STAT_FEATURE_BUILD","error":f"{type(exc).__name__}:{exc}"}
    for p in prediction_rows:
        pid=str(p.get("prediction_pair_id") or "")
        gs=pd.to_datetime(p.get("game_start"),utc=True,errors="coerce")
        k=(gs.round("s").isoformat(),_norm(p.get("home_team")),_norm(p.get("away_team"))) if pd.notna(gs) else None
        q=quotes.get(k,{}) if k else {};sys=systems.get(k,[]) if k else []
        rich_features=stat_features.get(pid)
        for market in ("SPREADS","H2H","TOTALS"):
            b=_market_base_row(p,market,q)
            if not b or b["core_direction"]==0:
                out.append({"prediction_pair_id":p.get("prediction_pair_id"),"game_start":p.get("game_start"),"home_team":p.get("home_team"),"away_team":p.get("away_team"),"market":market,"action":"NO MARKET","edge_sources":[],"system_labels":[x["label"] for x in sys if x["market"]==market],"stat_selector_live_status":stat_live_meta.get("status")});continue
            votes=[];core=b["core_direction"];stat_support=[]
            fid=f"NFL_{market}_FAIR_VALUE_EDGE"
            if fid in legit and b["raw_edge"]>=_num(legit[fid].get("threshold")):
                votes.append({"mechanism_family_id":fid,"independence_key":legit[fid].get("independence_key",fid),"direction":core,"label":"FAIR_VALUE"})
            fid=f"NFL_{market}_MARKET_CONFIRMATION"
            if fid in legit and _num(b.get("move_toward_core"))>=_num(legit[fid].get("threshold")):
                votes.append({"mechanism_family_id":fid,"independence_key":legit[fid].get("independence_key",fid),"direction":core,"label":"MARKET_CONFIRMATION"})
            if market in {"SPREADS","TOTALS"} and rich_features:
                for sfid,f in legit.items():
                    if f.get("market")!=market or f.get("mechanism_class")!="STAT_SELECTOR":continue
                    sid=str(f.get("selector_id") or f.get("variant_id") or "")
                    bundle=((stat_bundles.get(market) or {}).get(sid))
                    if not bundle:continue
                    scored=statv23.score_live_bundle(bundle,rich_features)
                    qualifies=bool(
                        scored.get("status")=="SCORED"
                        and int(scored.get("selector_direction") or 0)==core
                        and _num(scored.get("selector_scaled"))>=_num(f.get("selector_threshold"))
                        and b["raw_edge"]>=_num(f.get("core_gap_min"))
                    )
                    stat_support.append({
                        "family":f.get("stat_family"),"dependency_group":f.get("stat_dependency_group"),
                        "selector_id":sid,"selector_threshold":f.get("selector_threshold"),
                        "core_gap_min":f.get("core_gap_min"),"qualifies":qualifies,**scored,
                    })
                    if not qualifies:continue
                    votes.append({"mechanism_family_id":sfid,"independence_key":f.get("independence_key",f"NFL_{market}_STAT_SELECTOR"),
                                  "direction":core,"label":f"STAT {f.get('stat_family')}","selector_id":sid,
                                  "selector_prediction":scored.get("selector_prediction"),"selector_scaled":scored.get("selector_scaled")})
            for sv in sys:
                if sv["market"]==market and sv["family"] in legit:
                    f=legit[sv["family"]]
                    votes.append({"mechanism_family_id":sv["family"],"independence_key":f.get("independence_key",sv["family"]),"direction":int(sv["direction"]),"label":sv["label"]})
            mc=(meta.get("markets") or {}).get(market,{})
            res=shared.resolve_votes(votes,strong_play_allowed=bool(mc.get("strong_play_allowed")))
            action=res["action"]
            if action in {"PLAY","STRONG PLAY"} and not math.isfinite(_num(b.get("selected_price"))):action="EDGE — NO EXEC QUOTE"
            selected=b["selected"]
            if res["direction"] and res["direction"]!=core:
                if market=="SPREADS":selected=p.get("away_team") if core>0 else p.get("home_team")
                elif market=="TOTALS":selected="UNDER" if core>0 else "OVER"
                elif market=="H2H":selected=p.get("away_team") if core>0 else p.get("home_team")
            out.append({"prediction_pair_id":p.get("prediction_pair_id"),"game_start":p.get("game_start"),"home_team":p.get("home_team"),"away_team":p.get("away_team"),"market":market,
                        "action":action,"decision":res["decision"],"selected":selected,"market_value":b["market_value"],"model_value":b["model_value"],"raw_model_edge":b["raw_edge"],"selected_price":b["selected_price"],
                        "independent_mechanisms":res["independent_mechanisms"],"independence_keys":res.get("independence_keys") or [],"edge_sources":res["families"],"edge_votes":res["votes"],
                        "system_labels":[x["label"] for x in sys if x["market"]==market],"stat_selector_support":stat_support,"stat_selector_live_status":stat_live_meta.get("status"),
                        "betting_decision_authority":bool(action in {"PLAY","STRONG PLAY"}),"automatic_execution":False,"edge_contract_sha256":meta.get("contract_sha256")})
    return out


def _list_json(sc,bucket,prefix):
    out=[]
    for b in sc.list_blobs(bucket,prefix=prefix):
        if str(b.name).endswith(".json"):
            try:out.append(json.loads(b.download_as_text()))
            except Exception:pass
    return out


def _capture(sc,bucket,rows,now,contract_sha):
    n=0;existing=0
    for r in rows:
        if r.get("action") not in {"PLAY","STRONG PLAY"}:continue
        rid=_sha({"contract":contract_sha,"prediction_pair_id":r.get("prediction_pair_id"),"market":r.get("market")}); event={**r,"edge_event_id":rid,"captured_at":pd.to_datetime(now,utc=True).isoformat(),"source_tag":SOURCE_TAG}
        if _write_immutable(sc,bucket,f"{EVENT_PREFIX}/{rid}.json",event):n+=1
        else:existing+=1
    return {"inserted":n,"existing":existing}


def _settlement_rows(client,pair_ids):
    if not pair_ids:return pd.DataFrame()
    q=f"SELECT * FROM `{PAIRED_SETTLEMENT_TABLE}` WHERE prediction_pair_id IN UNNEST(@ids)"
    cfg=bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("ids","STRING",list(pair_ids))])
    return client.query(q,job_config=cfg).to_dataframe(create_bqstorage_client=False)


def _settle(client,sc,bucket,now):
    events=_list_json(sc,bucket,EVENT_PREFIX); settled={str(x.get("edge_event_id")) for x in _list_json(sc,bucket,SETTLEMENT_PREFIX)}; pending=[e for e in events if str(e.get("edge_event_id")) not in settled]
    if not pending:return {"pending_before":0,"inserted":0}
    s=_settlement_rows(client,{str(e.get("prediction_pair_id")) for e in pending}); by={str(r.prediction_pair_id):r for _,r in s.iterrows()};n=0
    for e in pending:
        r=by.get(str(e.get("prediction_pair_id")))
        if r is None:continue
        market=e.get("market");sel=str(e.get("selected") or "");result="UNRESOLVED";profit=np.nan;price=_num(e.get("selected_price"));pwin=_profit_american(price)
        if market=="SPREADS":
            line=_num(e.get("market_value"));margin=_num(r.actual_margin);home_sel=_norm(sel)==_norm(e.get("home_team"));v=(margin+line) if home_sel else -(margin+line)
            result="WIN" if v>0 else "LOSS" if v<0 else "PUSH";profit=pwin if result=="WIN" else -1 if result=="LOSS" else 0
        elif market=="TOTALS":
            line=_num(e.get("market_value"));actual=_num(r.actual_total);v=actual-line;v=-v if sel.upper()=="UNDER" else v
            result="WIN" if v>0 else "LOSS" if v<0 else "PUSH";profit=pwin if result=="WIN" else -1 if result=="LOSS" else 0
        elif market=="H2H":
            home_win=_num(r.home_win_label)==1;won=home_win if _norm(sel)==_norm(e.get("home_team")) else not home_win
            result="WIN" if won else "LOSS";profit=pwin if won else -1
        payload={"edge_event_id":e.get("edge_event_id"),"prediction_pair_id":e.get("prediction_pair_id"),"market":market,"selected":sel,"result":result,"profit_per_unit":None if not math.isfinite(_num(profit)) else float(profit),"settled_at":pd.to_datetime(now,utc=True).isoformat(),"source_tag":SOURCE_TAG}
        if _write_immutable(sc,bucket,f"{SETTLEMENT_PREFIX}/{e['edge_event_id']}.json",payload):n+=1
    return {"pending_before":len(pending),"inserted":n}


def _performance(sc,bucket):
    s=_list_json(sc,bucket,SETTLEMENT_PREFIX)
    def agg(x):
        w=sum(1 for r in x if r.get("result")=="WIN");l=sum(1 for r in x if r.get("result")=="LOSS");p=sum(1 for r in x if r.get("result")=="PUSH");profits=[_num(r.get("profit_per_unit")) for r in x if math.isfinite(_num(r.get("profit_per_unit")))]
        return {"n":len(x),"wins":w,"losses":l,"pushes":p,"hit_rate":round(w/(w+l),6) if w+l else None,"roi_per_unit":round(float(np.mean(profits)),6) if profits else None}
    out={"ALL":agg(s)}
    for m in ("SPREADS","H2H","TOTALS"):out[m]=agg([r for r in s if r.get("market")==m])
    return out


def update_live_state(*,bq_client,storage_client,bucket_name,prediction_rows,now,log_func=print):
    eng=_load_engine(storage_client,bucket_name);meta=eng["metadata"];rows=score_live(bq_client=bq_client,storage_client=storage_client,bucket_name=bucket_name,prediction_rows=prediction_rows,now=now);capture=_capture(storage_client,bucket_name,rows,now,meta.get("contract_sha256"));settle=_settle(bq_client,storage_client,bucket_name,now);perf=_performance(storage_client,bucket_name)
    counts={a:sum(1 for r in rows if r.get("action")==a) for a in ("PLAY","STRONG PLAY","MODEL ONLY","PASS — CONFLICT","EDGE — NO EXEC QUOTE","NO MARKET")}
    state={"status":"NFL_EDGE_AUTHORITY_V2_4_LIVE_ACTIVE","source_tag":SOURCE_TAG,"generated_at_utc":pd.to_datetime(now,utc=True).isoformat(),"contract":meta,"live_rows":rows,"action_counts":counts,"capture":capture,"settlement":settle,"live_performance":perf,"automatic_execution":False}
    state["current_uri"]=_write_json(storage_client,bucket_name,CURRENT_OBJECT,state)
    _stat_status=sorted(set(str(r.get("stat_selector_live_status") or "") for r in rows if r.get("stat_selector_live_status")))
    log_func("[NFL-EDGE-V2-LIVE] "+json.dumps({"status":state["status"],"action_counts":counts,"capture":capture,"settlement":settle,"live_performance":perf,"stat_selector_live_status":_stat_status,"contract_sha256":meta.get("contract_sha256")},sort_keys=True,default=str))
    return state


def read_dashboard_state(*,storage_client,bucket_name="sharp-models"):
    return {"meta":_read_json(storage_client,bucket_name,META_OBJECT),"current":_read_json(storage_client,bucket_name,CURRENT_OBJECT),"status":"READY"}


def _self_test():
    assert shared.framework_contract()["framework_version"]=="SPORTS_EDGE_AUTHORITY_V1_1"
    assert set(DISCOVERY_SEASONS).isdisjoint(CONFIRM_SEASONS)
    assert 2026 not in DISCOVERY_SEASONS+CONFIRM_SEASONS
    v=shared.resolve_votes([{"mechanism_family_id":"A","direction":1},{"mechanism_family_id":"B","direction":1}],strong_play_allowed=False)
    assert v["action"]=="PLAY"
    return {"status":"PASS","source_tag":SOURCE_TAG,"shared_framework_sha":shared.framework_contract()["contract_sha256"],"discovery":DISCOVERY_SEASONS,"confirmation":CONFIRM_SEASONS}

if __name__=="__main__":print(json.dumps(_self_test(),sort_keys=True))
