"""NFL Production V1 — fixed compact backbones + weekly challenger refresh.

This module is the first production-model layer after live-feature parity.
It deliberately separates three concepts:

1. A frozen baseline champion trained only through 2025.
2. A weekly challenger using the exact same architecture and fixed features,
   refreshed with all completed games available at run time.
3. Promotion authority, which remains CLOSED here. Promotion requires a later
   prospective paired-prediction review; a weekly retrain never replaces the
   champion automatically.

The three markets remain separate:
- SPREADS: compact Ridge fair-margin model.
- H2H: compact logistic win-probability model.
- TOTALS: compact Ridge fair-total model.

Every predictor is a member of the live-parity-approved compact contract. Market
lines/prices, current-game outcomes, research-only Is_Neutral/Is_Night_Game,
PBP sidecars, systems and market-microstructure signals are not model inputs.
Those remain independent layers for the future edge engine.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd

from nfl_feature_audit_v1 import VIEW

SOURCE_TAG = "nfl-production-v1.1.1-publish-receipt-normalization-20261002"
BASELINE_MAX_SEASON = 2025
MARKETS = ("SPREADS", "H2H", "TOTALS")

SPREAD_FEATURES = (
    "Week_Number", "Is_Home", "Is_Division_Game", "Rest_Differential_Days",
    "WinPct_Prior_Diff", "ATS_WinPct_Prior_Diff", "Avg_SU_Margin_Last5_Diff",
    "Off_YPP_vs_Opp_Def_Last3_Diff", "Def_YPP_vs_Opp_Off_Last3_Diff",
    "Prior_Season_WinPct_Diff",
)
H2H_FEATURES = (
    "Week_Number", "Is_Home", "Is_Division_Game", "Rest_Differential_Days",
    "WinPct_Prior_Diff", "Avg_SU_Margin_Last5_Diff",
    "Off_YPP_vs_Opp_Def_Last3_Diff", "Def_YPP_vs_Opp_Off_Last3_Diff",
    "Prior_Season_WinPct_Diff",
)
TOTAL_FEATURES = (
    "Week_Number", "Is_Home", "Is_Division_Game", "Rest_Differential_Days",
    "Avg_Points_For_Last5_Prior", "Avg_Points_Against_Last5_Prior",
    "Opp_Avg_Points_For_Last5_Prior", "Opp_Avg_Points_Against_Last5_Prior",
    "Avg_Off_YPP_Last3_Prior", "Avg_Def_YPP_Last3_Prior",
    "Opp_Avg_Off_YPP_Last3_Prior", "Opp_Avg_Def_YPP_Last3_Prior",
)
PRODUCTION_FEATURES = tuple(OrderedDict.fromkeys(
    x for xs in (SPREAD_FEATURES, H2H_FEATURES, TOTAL_FEATURES) for x in xs
))

# The final production feature contract is intentionally smaller than the earlier
# research compact set because Is_Neutral and Is_Night_Game did not pass exact
# live-feature parity.  Therefore older research metrics are references only and
# are never used as an exact reproducibility gate.  Publication is earned by
# season-forward validation of THIS final feature contract against simple,
# transparent training-only baselines.
VALIDATION_POLICY = {
    "SPREADS": {
        "candidate_metric": "mae",
        "baseline": "TRAINING_MEAN_MARGIN",
        "required_seasons_better": 3,
        "required_corr_gt": 0.0,
    },
    "H2H": {
        "candidate_metric": "log_loss",
        "baseline": "LAPLACE_TRAINING_WIN_RATE",
        "required_seasons_better": 4,
        "required_auc_gt": 0.55,
    },
    "TOTALS": {
        "candidate_metric": "mae",
        "baseline": "TRAINING_MEAN_TOTAL",
        "required_seasons_better": 3,
        "required_corr_gt": 0.0,
    },
}

BASE_PREFIX = "production/nfl/v1"
BASELINE_POINTER = f"{BASE_PREFIX}/current_champion.json"
CHALLENGER_POINTER = f"{BASE_PREFIX}/latest_challenger.json"

IDENTITY = ("Season", "Source_Name", "Source_Game_ID")
BASE_COLUMNS = (
    "Season", "Season_Stage", "Source_Name", "Source_Game_ID", "Game_Date",
    "Historical_Core_Eligible", "Team_Norm", "Opponent_Norm",
    "Is_Home", "Is_Away", "Is_Neutral", "Team_Score", "Opponent_Score",
)
QUERY_COLUMNS = tuple(OrderedDict.fromkeys((*BASE_COLUMNS, *PRODUCTION_FEATURES)))


def _json_bytes(x: Any) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), default=str).encode()


def _sha(x: Any) -> str:
    return hashlib.sha256(_json_bytes(x)).hexdigest()


def _training_data_sha256(games: pd.DataFrame) -> str:
    """Deterministic fingerprint of every field that can affect Production V1 fitting."""
    cols = [
        "physical_game_id", "Season", "Game_Date",
        *PRODUCTION_FEATURES, "actual_margin", "actual_total", "H2H_label",
    ]
    missing = [c for c in cols if c not in games.columns]
    if missing:
        raise RuntimeError("[NFL-PROD-V1-HOLD] TRAINING_FINGERPRINT_COLUMNS_MISSING " + str(missing))
    d = games.loc[:, cols].copy().sort_values(["Season", "Game_Date", "physical_game_id"], kind="mergesort").reset_index(drop=True)
    recs = []
    for rec in d.to_dict("records"):
        out = {}
        for k, v in rec.items():
            if isinstance(v, (pd.Timestamp, datetime)):
                t = pd.to_datetime(v, utc=True, errors="coerce")
                out[k] = None if pd.isna(t) else t.isoformat()
            elif isinstance(v, (np.integer,)):
                out[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                out[k] = None if not math.isfinite(float(v)) else round(float(v), 12)
            elif pd.isna(v):
                out[k] = None
            else:
                out[k] = v
        recs.append(out)
    return _sha(recs)


def production_contract() -> dict:
    c = {
        "source_tag": SOURCE_TAG,
        "markets": list(MARKETS),
        "baseline_training_max_season": BASELINE_MAX_SEASON,
        "backbones": {
            "SPREADS": {"family": "COMPACT_RIDGE", "target": "actual_margin", "features": list(SPREAD_FEATURES), "ridge_alpha": 12.0},
            "H2H": {"family": "H2H_COMPACT_LOGISTIC", "target": "home_or_canonical_win", "features": list(H2H_FEATURES), "logistic_C": 0.35},
            "TOTALS": {"family": "COMPACT_RIDGE", "target": "actual_total", "features": list(TOTAL_FEATURES), "ridge_alpha": 12.0},
        },
        "production_feature_count": len(PRODUCTION_FEATURES),
        "production_features": list(PRODUCTION_FEATURES),
        "excluded_from_model_inputs": [
            "Is_Neutral", "Is_Night_Game", "Spread_Value", "Opening_Spread",
            "Current_Total", "Opening_Total", "ML_Odds", "market_microstructure",
            "systems", "PBP_CORE2",
        ],
        "model_prediction_authority": True,
        "betting_decision_authority": False,
        "automatic_promotion": False,
        "refresh_policy": "weekly challenger refit after completed NFL week",
        "promotion_policy": "separate post-freeze paired-prediction review; target cadence every ~4 weeks and >=60 post-freeze paired games",
        "promotion_evidence_start": "AFTER_FROZEN_CHAMPION_AND_PAIRED_LEDGER_ACTIVATION",
    }
    c["contract_sha256"] = _sha(c)
    return c


def _build_query(*, max_season: int | None = None) -> str:
    qcols = ", ".join(f"`{c}`" for c in QUERY_COLUMNS)
    where = ["Season >= 2017", "Season_Stage IN ('REGULAR','POSTSEASON')", "Historical_Core_Eligible = 1", "Team_Score IS NOT NULL", "Opponent_Score IS NOT NULL"]
    if max_season is not None:
        where.append(f"Season <= {int(max_season)}")
    return f"SELECT {qcols} FROM `{VIEW}` WHERE " + " AND ".join(where) + " ORDER BY Season, Game_Date, Source_Name, Source_Game_ID, Team_Norm"


def _validate_source_schema(bq_client) -> None:
    cols = {f.name for f in bq_client.get_table(VIEW).schema}
    missing = sorted(set(QUERY_COLUMNS) - cols)
    if missing:
        raise RuntimeError("[NFL-PROD-V1-HOLD] SOURCE_COLUMNS_MISSING " + str(missing))


def _num(v) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else math.nan
    except Exception:
        return math.nan


def _physical_games(side: pd.DataFrame) -> pd.DataFrame:
    """One physical game = one model row, home-oriented when possible.

    Neutral-site rows are allowed in training, but Is_Neutral is not a predictor.
    For a neutral game, lexical Team_Norm chooses a deterministic orientation.
    """
    if side is None or side.empty:
        raise RuntimeError("[NFL-PROD-V1-HOLD] NO_COMPLETED_HISTORY")
    missing = sorted(set(QUERY_COLUMNS) - set(side.columns))
    if missing:
        raise RuntimeError("[NFL-PROD-V1-HOLD] QUERY_RESULT_COLUMNS_MISSING " + str(missing))
    d = side.copy()
    d["Season"] = pd.to_numeric(d["Season"], errors="coerce")
    if d["Season"].isna().any() or d["Season"].mod(1).ne(0).any():
        raise RuntimeError("[NFL-PROD-V1-HOLD] INVALID_SEASON")
    d["Season"] = d["Season"].astype(int)
    rows=[]; anomalies=[]
    for ident,p in d.groupby(list(IDENTITY), sort=False, dropna=False):
        if len(p) != 2:
            anomalies.append({"game":"|".join(map(str,ident)),"reason":"BAD_SIDE_COUNT","rows":int(len(p))}); continue
        teams=[str(x or "").strip().lower() for x in p.Team_Norm]
        opps=[str(x or "").strip().lower() for x in p.Opponent_Norm]
        if len(set(teams)) != 2 or sorted(teams) != sorted(opps) or any(a==b for a,b in zip(teams,opps)):
            anomalies.append({"game":"|".join(map(str,ident)),"reason":"BAD_TEAM_PAIR"}); continue
        if p[["Team_Score","Opponent_Score"]].isna().any().any():
            continue
        a,b=p.iloc[0],p.iloc[1]
        if _num(a.Team_Score)!=_num(b.Opponent_Score) or _num(a.Opponent_Score)!=_num(b.Team_Score):
            anomalies.append({"game":"|".join(map(str,ident)),"reason":"NONRECIPROCAL_SCORE"}); continue
        h=pd.to_numeric(p.Is_Home,errors="coerce").fillna(-1)
        aw=pd.to_numeric(p.Is_Away,errors="coerce").fillna(-1)
        ne=pd.to_numeric(p.Is_Neutral,errors="coerce").fillna(-1)
        reg=sorted(zip(h.tolist(),aw.tolist(),ne.tolist()))==[(0,1,0),(1,0,0)]
        neutral=bool(h.eq(0).all() and aw.eq(0).all() and ne.eq(1).all())
        if reg:
            chosen=p.loc[h.eq(1)].iloc[0]
            orientation="HOME"
        elif neutral:
            chosen=p.sort_values("Team_Norm",kind="mergesort").iloc[0]
            orientation="NEUTRAL_CANONICAL"
        else:
            anomalies.append({"game":"|".join(map(str,ident)),"reason":"BAD_VENUE_PAIR"}); continue
        rec=chosen.to_dict()
        rec["physical_game_id"]="|".join(map(str,ident))
        rec["orientation"]=orientation
        ts=_num(chosen.Team_Score); os=_num(chosen.Opponent_Score)
        rec["actual_margin"]=ts-os
        rec["actual_total"]=ts+os
        rec["H2H_label"]=1.0 if ts>os else 0.0 if ts<os else np.nan
        rows.append(rec)
    if anomalies:
        raise RuntimeError("[NFL-PROD-V1-HOLD] GAME_GRAIN_ANOMALIES "+json.dumps(anomalies[:12],sort_keys=True))
    out=pd.DataFrame(rows).sort_values(["Season","Game_Date","physical_game_id"],kind="mergesort").reset_index(drop=True)
    if out.empty or out.physical_game_id.duplicated().any():
        raise RuntimeError("[NFL-PROD-V1-HOLD] INVALID_PHYSICAL_GAME_FRAME")
    return out


def _xy(train: pd.DataFrame, valid: pd.DataFrame, features: tuple[str,...]):
    a=(train.loc[:,features].apply(pd.to_numeric,errors="coerce").astype(float).replace([np.inf,-np.inf],np.nan))
    b=(valid.loc[:,features].apply(pd.to_numeric,errors="coerce").astype(float).replace([np.inf,-np.inf],np.nan))
    med=a.median(axis=0).fillna(0.0).astype(float)
    return a.fillna(med).to_numpy(float), b.fillna(med).to_numpy(float), med.to_dict()


def _fit_predict(market: str, train: pd.DataFrame, valid: pd.DataFrame):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, LogisticRegression
    if market=="SPREADS":
        features=SPREAD_FEATURES; target="actual_margin"
        A,B,_=_xy(train,valid,features); y=pd.to_numeric(train[target],errors="coerce").to_numpy(float)
        sc=StandardScaler().fit(A); m=Ridge(alpha=12.0,random_state=17).fit(sc.transform(A),y)
        return np.asarray(m.predict(sc.transform(B)),float)
    if market=="TOTALS":
        features=TOTAL_FEATURES; target="actual_total"
        A,B,_=_xy(train,valid,features); y=pd.to_numeric(train[target],errors="coerce").to_numpy(float)
        sc=StandardScaler().fit(A); m=Ridge(alpha=12.0,random_state=17).fit(sc.transform(A),y)
        return np.asarray(m.predict(sc.transform(B)),float)
    if market=="H2H":
        tr=train.loc[train.H2H_label.notna()].copy(); va=valid.copy(); features=H2H_FEATURES
        A,B,_=_xy(tr,va,features); y=tr.H2H_label.to_numpy(float)
        sc=StandardScaler().fit(A); m=LogisticRegression(C=.35,max_iter=900,solver="lbfgs",random_state=17).fit(sc.transform(A),y)
        idx=list(m.classes_).index(1.0)
        return np.clip(m.predict_proba(sc.transform(B))[:,idx],.001,.999)
    raise ValueError(market)


def _continuous(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); m=np.isfinite(y)&np.isfinite(p); y=y[m]; p=p[m]
    if not len(y): return {"n":0}
    e=p-y; corr=float(np.corrcoef(y,p)[0,1]) if len(y)>2 and np.std(y)>0 and np.std(p)>0 else math.nan
    return {"n":int(len(y)),"mae":round(float(np.mean(np.abs(e))),6),"rmse":round(float(np.sqrt(np.mean(e*e))),6),"bias":round(float(np.mean(e)),6),"corr":round(corr,6) if math.isfinite(corr) else None}


def _classification(y,p):
    from sklearn.metrics import roc_auc_score
    y=np.asarray(y,float); p=np.asarray(p,float); m=np.isfinite(y)&np.isfinite(p); y=y[m]; p=np.clip(p[m],.001,.999)
    if not len(y): return {"n":0}
    ll=float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p)))); br=float(np.mean((p-y)**2))
    auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else math.nan
    bins=np.minimum((p*5).astype(int),4)
    ece=sum(abs(float(np.mean(y[bins==i]))-float(np.mean(p[bins==i])))*int(np.sum(bins==i))/len(y) for i in range(5) if np.any(bins==i))
    return {"n":int(len(y)),"log_loss":round(ll,6),"brier":round(br,6),"auc":round(auc,6) if math.isfinite(auc) else None,"ece5":round(float(ece),6)}


def _baseline_predict(market: str, train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    """Training-only null comparator for the final production contract."""
    if market == "SPREADS":
        y = pd.to_numeric(train["actual_margin"], errors="coerce")
        mu = float(y.mean())
        return np.full(len(valid), mu, dtype=float)
    if market == "TOTALS":
        y = pd.to_numeric(train["actual_total"], errors="coerce")
        mu = float(y.mean())
        return np.full(len(valid), mu, dtype=float)
    if market == "H2H":
        tr = train.loc[train.H2H_label.notna()].copy()
        y = pd.to_numeric(tr.H2H_label, errors="coerce").dropna().to_numpy(float)
        # Laplace smoothing avoids a degenerate 0/1 probability in small samples
        # and is determined entirely from the training fold.
        rate = float((np.sum(y) + 1.0) / (len(y) + 2.0))
        return np.full(len(valid), rate, dtype=float)
    raise ValueError(market)


def season_forward_oof(games: pd.DataFrame) -> dict:
    """2021-2025 expanding-window OOF for the FINAL live-parity feature set.

    Every fold trains on seasons strictly before the validation season.  The
    candidate and its null baseline see the same validation games.
    """
    folds = {m: [] for m in MARKETS}
    candidate_store = {m: [] for m in MARKETS}
    baseline_store = {m: [] for m in MARKETS}
    for year in range(2021, 2026):
        tr = games.loc[games.Season.lt(year)].copy()
        va = games.loc[games.Season.eq(year)].copy()
        if len(tr) < 1000 or len(va) < 250:
            raise RuntimeError(f"[NFL-PROD-V1-HOLD] INCOMPLETE_OOF_YEAR_{year} train={len(tr)} valid={len(va)}")
        for market in MARKETS:
            if market == "H2H":
                vv = va.loc[va.H2H_label.notna()].copy()
                pred = _fit_predict(market, tr, vv)
                base = _baseline_predict(market, tr, vv)
                cand_met = _classification(vv.H2H_label.to_numpy(float), pred)
                base_met = _classification(vv.H2H_label.to_numpy(float), base)
            else:
                vv = va
                target = "actual_margin" if market == "SPREADS" else "actual_total"
                pred = _fit_predict(market, tr, vv)
                base = _baseline_predict(market, tr, vv)
                cand_met = _continuous(vv[target].to_numpy(float), pred)
                base_met = _continuous(vv[target].to_numpy(float), base)
            metric = "log_loss" if market == "H2H" else "mae"
            cval = cand_met.get(metric)
            bval = base_met.get(metric)
            better = bool(
                cval is not None and bval is not None
                and math.isfinite(float(cval)) and math.isfinite(float(bval))
                and float(cval) < float(bval)
            )
            folds[market].append({
                "validate_season": year,
                "train_games": int(len(tr)),
                "validate_games": int(len(vv)),
                "candidate": cand_met,
                "baseline": base_met,
                "candidate_better": better,
                "delta_candidate_minus_baseline": round(float(cval) - float(bval), 6)
                    if cval is not None and bval is not None else None,
            })
            candidate_store[market].append((vv, np.asarray(pred, float)))
            baseline_store[market].append((vv, np.asarray(base, float)))

    summary = {}
    for market in MARKETS:
        frames = [f for f, _ in candidate_store[market]]
        allv = pd.concat(frames, ignore_index=True)
        cand = np.concatenate([p for _, p in candidate_store[market]])
        base = np.concatenate([p for _, p in baseline_store[market]])
        if market == "H2H":
            y = allv.H2H_label.to_numpy(float)
            cand_met = _classification(y, cand)
            base_met = _classification(y, base)
            metric = "log_loss"
        else:
            target = "actual_margin" if market == "SPREADS" else "actual_total"
            y = allv[target].to_numpy(float)
            cand_met = _continuous(y, cand)
            base_met = _continuous(y, base)
            metric = "mae"
        better_count = int(sum(bool(x["candidate_better"]) for x in folds[market]))
        summary[market] = {
            "candidate": cand_met,
            "baseline": base_met,
            "primary_metric": metric,
            "aggregate_delta_candidate_minus_baseline": round(
                float(cand_met[metric]) - float(base_met[metric]), 6
            ),
            "seasons_better_than_baseline": better_count,
            "seasons_compared": len(folds[market]),
        }
    return {"folds": folds, "summary": summary}


def _validation_gate(oof: dict) -> dict:
    """Fail closed unless each fixed backbone clears its simple OOF baseline."""
    out = {}
    for market in MARKETS:
        policy = VALIDATION_POLICY[market]
        s = oof["summary"][market]
        cand = s["candidate"]
        base = s["baseline"]
        metric = policy["candidate_metric"]
        reasons = []
        cv = cand.get(metric)
        bv = base.get(metric)
        if cv is None or bv is None or not math.isfinite(float(cv)) or not math.isfinite(float(bv)):
            reasons.append("PRIMARY_METRIC_NOT_FINITE")
        elif not float(cv) < float(bv):
            reasons.append("AGGREGATE_NOT_BETTER_THAN_BASELINE")
        if int(s.get("seasons_better_than_baseline", 0)) < int(policy["required_seasons_better"]):
            reasons.append("INSUFFICIENT_SEASON_CONSISTENCY")
        if market == "H2H":
            auc = cand.get("auc")
            if auc is None or not math.isfinite(float(auc)) or not float(auc) > float(policy["required_auc_gt"]):
                reasons.append("AUC_GATE_FAILED")
        else:
            corr = cand.get("corr")
            if corr is None or not math.isfinite(float(corr)) or not float(corr) > float(policy["required_corr_gt"]):
                reasons.append("CORRELATION_GATE_FAILED")
        out[market] = {
            "status": "PASS" if not reasons else "HOLD",
            "policy": policy,
            "candidate": cand,
            "baseline": base,
            "aggregate_delta_candidate_minus_baseline": s.get("aggregate_delta_candidate_minus_baseline"),
            "seasons_better_than_baseline": s.get("seasons_better_than_baseline"),
            "reasons": reasons,
        }
    if any(x["status"] != "PASS" for x in out.values()):
        raise RuntimeError("[NFL-PROD-V1-HOLD] FINAL_18_FEATURE_OOF_GATE " + json.dumps(out, sort_keys=True, default=str))
    return out

def _fit_one(market: str, games: pd.DataFrame) -> dict:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, LogisticRegression
    if market=="SPREADS": features=SPREAD_FEATURES; target="actual_margin"; fam="COMPACT_RIDGE"
    elif market=="TOTALS": features=TOTAL_FEATURES; target="actual_total"; fam="COMPACT_RIDGE"
    else: features=H2H_FEATURES; target="H2H_label"; fam="H2H_COMPACT_LOGISTIC"
    d=games.loc[games[target].notna()].copy()
    A,_,med=_xy(d,d.iloc[:1].copy(),features)
    y=pd.to_numeric(d[target],errors="coerce").to_numpy(float)
    sc=StandardScaler().fit(A)
    if market=="H2H":
        model=LogisticRegression(C=.35,max_iter=900,solver="lbfgs",random_state=17).fit(sc.transform(A),y)
        hyper={"C":0.35}
    else:
        model=Ridge(alpha=12.0,random_state=17).fit(sc.transform(A),y); hyper={"alpha":12.0}
    return {"market":market,"family":fam,"features":list(features),"target":target,"median":{k:float(v) for k,v in med.items()},"scaler":sc,"model":model,"hyperparameters":hyper,"training_rows":int(len(d))}


def fit_bundle(games: pd.DataFrame, *, role: str, cutoff_label: str) -> dict:
    models={m:_fit_one(m,games) for m in MARKETS}
    meta={
        "source_tag":SOURCE_TAG,"role":role,"cutoff_label":cutoff_label,
        "training_games":int(len(games)),"season_counts":{str(int(k)):int(v) for k,v in games.groupby("Season").size().items()},
        "max_training_season":int(games.Season.max()),"max_training_game_date":str(pd.to_datetime(games.Game_Date,errors="coerce").max()),
        "contract_sha256":production_contract()["contract_sha256"],"automatic_promotion":False,
        "model_prediction_authority": role=="FROZEN_CHAMPION",
        "betting_decision_authority":False,
    }
    return {"metadata":meta,"models":models}


def _blob_json(storage_client,bucket_name,name):
    blob=storage_client.bucket(bucket_name).blob(name)
    if not blob.exists(): return None
    return json.loads(blob.download_as_text())


def _upload_immutable(storage_client,bucket_name,name,data,content_type):
    from google.api_core.exceptions import PreconditionFailed
    blob=storage_client.bucket(bucket_name).blob(name)
    try:
        blob.upload_from_string(data,content_type=content_type,if_generation_match=0)
        return {"uri":f"gs://{bucket_name}/{name}","created":True}
    except PreconditionFailed:
        return {"uri":f"gs://{bucket_name}/{name}","created":False,"reason":"ALREADY_EXISTS_IMMUTABLE"}


def _write_pointer(storage_client,bucket_name,name,payload):
    blob=storage_client.bucket(bucket_name).blob(name)
    blob.upload_from_string(json.dumps(payload,sort_keys=True,indent=2,default=str),content_type="application/json")
    return f"gs://{bucket_name}/{name}"


def _serialize(bundle: dict) -> bytes:
    bio=io.BytesIO(); joblib.dump(bundle,bio,compress=3); return bio.getvalue()


def _publish_bundle(storage_client,bucket_name,bundle:dict,registry:dict,prefix:str):
    """Publish immutable model + registry and return a stable URI contract.

    Do not depend on the exact receipt shape returned by the upload helper. The
    immutable object names are deterministic, so their gs:// URIs are known
    before upload. This also makes retries after a partial publish safe.
    """
    data=_serialize(bundle)
    art_sha=hashlib.sha256(data).hexdigest()
    registry=dict(registry)
    registry["artifact_sha256"]=art_sha

    artifact_name=f"{prefix}/backbones.joblib"
    registry_name=f"{prefix}/registry.json"
    artifact_uri=f"gs://{bucket_name}/{artifact_name}"
    registry_uri=f"gs://{bucket_name}/{registry_name}"

    a_receipt=_upload_immutable(
        storage_client,bucket_name,artifact_name,data,"application/octet-stream"
    )
    r_receipt=_upload_immutable(
        storage_client,bucket_name,registry_name,
        json.dumps(registry,sort_keys=True,indent=2,default=str).encode(),
        "application/json",
    )

    # Normalize the outward-facing contract even if a storage helper/version
    # returns only creation metadata. Preserve the raw receipts for diagnostics.
    artifact={"uri":artifact_uri,"object_name":artifact_name,"upload_receipt":a_receipt}
    registry_pub={"uri":registry_uri,"object_name":registry_name,"upload_receipt":r_receipt}
    return {
        "artifact":artifact,
        "registry":registry_pub,
        "artifact_sha256":art_sha,
        "registry_payload":registry,
    }


def _training_registry(*,role,games,oof,validation_gate,freeze_utc=None):
    reg={
        "source_tag":SOURCE_TAG,"role":role,"contract":production_contract(),
        "training_games":int(len(games)),"season_counts":{str(int(k)):int(v) for k,v in games.groupby("Season").size().items()},
        "max_training_season":int(games.Season.max()),"max_training_game_date":str(pd.to_datetime(games.Game_Date,errors="coerce").max()),
        "oof_2021_2025":oof,"validation_gate":validation_gate,"automatic_promotion":False,
        "model_prediction_authority":role=="FROZEN_CHAMPION","betting_decision_authority":False,
        "freeze_utc":freeze_utc,
    }
    reg["registry_sha256"]=_sha(reg)
    return reg


def run_nfl_production_refresh(*, bq_client, storage_client, bucket_name="sharp-models", log_func=print, now=None) -> dict:
    now=now or datetime.now(timezone.utc)
    contract=production_contract()
    log_func("[NFL-PROD-V1-REFRESH-PREFLIGHT] "+json.dumps({
        "status":"START","source_tag":SOURCE_TAG,"contract_sha256":contract["contract_sha256"],
        "production_feature_count":len(PRODUCTION_FEATURES),"markets":list(MARKETS),
        "automatic_promotion":False,"baseline_training_max_season":BASELINE_MAX_SEASON,
        "oof_validation_policy":VALIDATION_POLICY,
        "promotion_evidence":"POSTFREEZE_PAIRED_ONLY",
    },sort_keys=True))
    _validate_source_schema(bq_client)

    # Always rebuild the protected <=2025 evidence frame. The first production
    # champion is frozen from this exact window, keeping 2026 prospective.
    hist=bq_client.query(_build_query(max_season=BASELINE_MAX_SEASON)).to_dataframe(create_bqstorage_client=False)
    baseline_games=_physical_games(hist)
    if int(baseline_games.Season.max())!=BASELINE_MAX_SEASON:
        raise RuntimeError("[NFL-PROD-V1-HOLD] BASELINE_HISTORY_DOES_NOT_REACH_2025")
    oof=season_forward_oof(baseline_games)
    gate=_validation_gate(oof)
    log_func("[NFL-PROD-V1-OOF] "+json.dumps({"status":"PASS","summary":oof["summary"],"validation_gate":gate},sort_keys=True,default=str))

    existing=_blob_json(storage_client,bucket_name,BASELINE_POINTER)
    baseline_created=False
    if existing is None:
        baseline_bundle=fit_bundle(baseline_games,role="FROZEN_CHAMPION",cutoff_label="THROUGH_2025_ONLY")
        baseline_reg=_training_registry(role="FROZEN_CHAMPION",games=baseline_games,oof=oof,validation_gate=gate,freeze_utc=now.isoformat())
        prefix=f"{BASE_PREFIX}/champions/{baseline_reg['registry_sha256'][:16]}"
        pub=_publish_bundle(storage_client,bucket_name,baseline_bundle,baseline_reg,prefix)
        pointer={
            "status":"NFL_PRODUCTION_V1_FROZEN_CHAMPION","source_tag":SOURCE_TAG,
            "registry_sha256":baseline_reg["registry_sha256"],"artifact_sha256":pub["artifact_sha256"],
            "artifact_uri":pub["artifact"]["uri"],"registry_uri":pub["registry"]["uri"],
            "frozen_at_utc":now.isoformat(),"training_max_season":BASELINE_MAX_SEASON,
            "model_prediction_authority":True,"betting_decision_authority":False,"automatic_promotion":False,
        }
        _write_pointer(storage_client,bucket_name,BASELINE_POINTER,pointer)
        existing=pointer; baseline_created=True
    else:
        if existing.get("source_tag")!=SOURCE_TAG:
            raise RuntimeError("[NFL-PROD-V1-HOLD] EXISTING_CHAMPION_SOURCE_TAG_MISMATCH")
        if existing.get("training_max_season")!=BASELINE_MAX_SEASON:
            raise RuntimeError("[NFL-PROD-V1-HOLD] EXISTING_CHAMPION_TRAINING_WINDOW_MISMATCH")
    log_func("[NFL-PROD-V1-CHAMPION] "+json.dumps({**existing,"created_now":baseline_created},sort_keys=True,default=str))

    # Weekly challenger: same architecture, latest completed history. No model
    # selection is performed here and this object cannot mutate current champion.
    all_side=bq_client.query(_build_query(max_season=None)).to_dataframe(create_bqstorage_client=False)
    all_games=_physical_games(all_side)
    if len(all_games)<len(baseline_games):
        raise RuntimeError("[NFL-PROD-V1-HOLD] CURRENT_HISTORY_SMALLER_THAN_BASELINE")
    post_2025_training_games=all_games.loc[(all_games.Season>BASELINE_MAX_SEASON)].copy()
    cutoff=pd.to_datetime(all_games.Game_Date,errors="coerce").max()
    cutoff_s=cutoff.strftime("%Y%m%d") if pd.notna(cutoff) else "unknown"
    training_data_sha256=_training_data_sha256(all_games)

    # Reuse the exact same challenger when the completed-game training frame has
    # not changed. This makes weekly refresh idempotent and prevents a no-new-data
    # rerun from manufacturing a new prospective challenger identity.
    prior_ch=_blob_json(storage_client,bucket_name,CHALLENGER_POINTER)
    challenger_reused_existing=bool(
        isinstance(prior_ch,dict)
        and prior_ch.get("source_tag")==SOURCE_TAG
        and prior_ch.get("training_data_sha256")==training_data_sha256
        and prior_ch.get("champion_registry_sha256")==existing.get("registry_sha256")
        and prior_ch.get("artifact_uri")
        and prior_ch.get("registry_uri")
    )
    if challenger_reused_existing:
        ch_pointer=prior_ch
    else:
        challenger_bundle=fit_bundle(all_games,role="WEEKLY_CHALLENGER",cutoff_label="LATEST_COMPLETED_GAMES")
        ch_reg=_training_registry(role="WEEKLY_CHALLENGER",games=all_games,oof=oof,validation_gate=gate,freeze_utc=None)
        ch_reg.update({
            "champion_registry_sha256":existing.get("registry_sha256"),
            "training_data_sha256":training_data_sha256,
            "completed_training_games_after_2025":int(len(post_2025_training_games)),
            "promotion_review_target_min_postfreeze_paired_games":60,
            "promotion_review_target_cadence_weeks":4,
            "postfreeze_paired_games":0,
            "promotion_count_gate_met":False,
            "promotion_evidence_policy":"ONLY_PREDICTIONS_RECORDED_AFTER_CHAMPION_AND_LEDGER_ACTIVATION_COUNT",
            "promotion_status":"NOT_EVALUATED_PROSPECTIVE_PAIRED_LEDGER_REQUIRED",
        })
        ch_reg["registry_sha256"]=_sha(ch_reg)
        prefix=f"{BASE_PREFIX}/challengers/{cutoff_s}-{ch_reg['registry_sha256'][:12]}"
        ch_pub=_publish_bundle(storage_client,bucket_name,challenger_bundle,ch_reg,prefix)
        ch_pointer={
            "status":"NFL_PRODUCTION_V1_WEEKLY_CHALLENGER_READY","source_tag":SOURCE_TAG,
            "registry_sha256":ch_reg["registry_sha256"],"artifact_sha256":ch_pub["artifact_sha256"],
            "artifact_uri":ch_pub["artifact"]["uri"],"registry_uri":ch_pub["registry"]["uri"],
            "champion_registry_sha256":existing.get("registry_sha256"),
            "training_data_sha256":training_data_sha256,
            "data_cutoff":str(cutoff),"completed_training_games_after_2025":int(len(post_2025_training_games)),
            "postfreeze_paired_games":0,"promotion_count_gate_met":False,"automatic_promotion":False,
            "production_authority":0,
        }
        _write_pointer(storage_client,bucket_name,CHALLENGER_POINTER,ch_pointer)
    log_func("[NFL-PROD-V1-CHALLENGER] "+json.dumps({**ch_pointer,"reused_existing":challenger_reused_existing},sort_keys=True,default=str))

    report={
        "status":"NFL_PRODUCTION_V1_BACKBONES_FROZEN_AND_CHALLENGER_REFRESHED",
        "source_tag":SOURCE_TAG,"contract_sha256":contract["contract_sha256"],
        "champion":existing,"champion_created_now":baseline_created,"challenger":ch_pointer,
        "challenger_reused_existing":challenger_reused_existing,
        "oof_summary":oof["summary"],"validation_gate":gate,
        "backbones":contract["backbones"],"production_feature_count":len(PRODUCTION_FEATURES),
        "completed_training_games_after_2025":int(len(post_2025_training_games)),
        "postfreeze_paired_games":0,
        "next_step":"WIRE_LIVE_SCORER_AND_PROSPECTIVE_PAIRED_PREDICTION_LEDGER",
        "automatic_promotion":False,"betting_decision_authority":False,
    }
    log_func("[NFL-PROD-V1-REFRESH-CONTRACT] "+json.dumps(report,sort_keys=True,default=str))
    return report


def score_feature_rows(bundle: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Score already-parity-approved oriented game rows.

    This helper is included for the next live-scorer stage. It does not fetch
    market data or create betting actions.
    """
    out=rows.copy()
    for market,key in (("SPREADS","fair_margin"),("TOTALS","fair_total"),("H2H","win_probability")):
        spec=bundle["models"][market]; features=tuple(spec["features"])
        X=out.loc[:,features].apply(pd.to_numeric,errors="coerce").astype(float).replace([np.inf,-np.inf],np.nan)
        med=pd.Series(spec["median"],dtype=float); X=X.fillna(med)
        Z=spec["scaler"].transform(X.to_numpy(float))
        if market=="H2H":
            model=spec["model"]; idx=list(model.classes_).index(1.0); pred=model.predict_proba(Z)[:,idx]
            pred=np.clip(pred,.001,.999)
        else:
            pred=spec["model"].predict(Z)
        out[key]=np.asarray(pred,float)
    return out


def _self_test():
    # Contract relationship to the parity-approved union.
    assert len(PRODUCTION_FEATURES)==18
    assert set(SPREAD_FEATURES)|set(H2H_FEATURES)|set(TOTAL_FEATURES)==set(PRODUCTION_FEATURES)
    assert "Is_Neutral" not in PRODUCTION_FEATURES and "Is_Night_Game" not in PRODUCTION_FEATURES
    assert len(SPREAD_FEATURES)==10 and len(H2H_FEATURES)==9 and len(TOTAL_FEATURES)==12
    c=production_contract(); assert c["contract_sha256"]==_sha({k:v for k,v in c.items() if k!="contract_sha256"})
    return {"status":"PASS","features":len(PRODUCTION_FEATURES),"source_tag":SOURCE_TAG}


if __name__ == "__main__":
    print(json.dumps(_self_test(),sort_keys=True))
