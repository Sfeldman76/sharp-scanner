"""NFL V1.9.4 edge-gate and dual-scorecard research.

Research-only.  The fair-line model and the betting-edge decision are treated as
separate problems:

* FAIR LINE asks for the expected margin / total and is scored with MAE/RMSE.
* EDGE GATE asks when the incumbent CORE disagreement with the market is likely
  to be directionally correct and is scored with Brier/log loss/calibration plus
  transparent coverage/hit-rate gates.

The edge gate never sees 2026.  OOF gate predictions are season-forward and the
final fitted objects have zero production authority.  Fixed probability bands
are reported; none is auto-selected or promoted.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from typing import Iterable

import numpy as np
import pandas as pd

SOURCE_TAG = "nfl-edge-gate-v1.9.4-season-forward-dual-scorecard-20261001"
PRODUCTION_AUTHORITY = 0
MAX_RESEARCH_SEASON = 2025
GATE_VALIDATION_YEARS = (2023, 2024, 2025)
FIXED_PROBABILITY_BANDS = (0.53, 0.55, 0.57, 0.60)
LOGISTIC_C = 0.25
MIN_TRAIN_ROWS = 450

# Deliberately compact.  These are prior/pregame context fields already present
# in the audited research frame.  The gate is not allowed to become another
# 100-feature generic prediction model.
EDGE_CONTEXT_FEATURES = (
    "Is_Division_Game",
    "Is_Conference_Game",
    "same_season_rematch",
    "same_season_role_flip",
    "first_home_game",
    "first_road_game",
    "schedule_strength_diff",
    "rush_matchup_diff",
    "pass_matchup_diff",
    "sack_matchup_diff",
    "points_per_play_context_diff",
    "third_down_context_diff",
    "turnover_margin_context_diff",
    "giveaway_context_diff",
    "points_trend_context_diff",
    "off_ypp_trend_context_diff",
    "def_ypp_trend_context_diff",
    "close_game_win_pct_diff",
    "close_game_rate_diff",
    "blowout_rate_diff",
)


def _num(x, index=None) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    if index is not None:
        s.index = index
    return s


def _mae(a, b):
    x=np.asarray(a,float); y=np.asarray(b,float)
    m=np.isfinite(x)&np.isfinite(y)
    return float(np.mean(np.abs(x[m]-y[m]))) if m.any() else math.nan


def _rmse(a, b):
    x=np.asarray(a,float); y=np.asarray(b,float)
    m=np.isfinite(x)&np.isfinite(y)
    return float(np.sqrt(np.mean((x[m]-y[m])**2))) if m.any() else math.nan


def _brier(y, p):
    yy=np.asarray(y,float); pp=np.asarray(p,float)
    m=np.isfinite(yy)&np.isfinite(pp)
    return float(np.mean((pp[m]-yy[m])**2)) if m.any() else math.nan


def _logloss(y, p):
    yy=np.asarray(y,float); pp=np.asarray(p,float)
    m=np.isfinite(yy)&np.isfinite(pp)
    if not m.any(): return math.nan
    pp=np.clip(pp[m],1e-6,1-1e-6); yy=yy[m]
    return float(-np.mean(yy*np.log(pp)+(1-yy)*np.log(1-pp)))


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    yy=np.asarray(y,float); pp=np.asarray(p,float)
    m=np.isfinite(yy)&np.isfinite(pp)
    if m.sum()<10 or len(np.unique(yy[m]))<2: return None
    return float(roc_auc_score(yy[m],pp[m]))


def _wilson(wins: int, n: int, z: float = 1.959963984540054):
    if n <= 0: return (None,None)
    p=wins/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d
    h=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/d
    return (max(0.0,c-h), min(1.0,c+h))


def _market_parts(df: pd.DataFrame, market: str):
    if market == "spreads":
        actual=_num(df["actual_margin"]); base=-_num(df["Spread_Value"])
    elif market == "totals":
        actual=_num(df["actual_total"]); base=_num(df["Current_Total"])
    else:
        raise ValueError(market)
    return actual,base,actual-base


def _core_parts(df: pd.DataFrame, market: str):
    if market == "spreads":
        return _num(df["core_margin_pred"]), _num(df["direct_margin_pred"]), _num(df["score_margin_pred"]), _num(df["spread_model_gap"])
    return _num(df["core_total_pred"]), _num(df["direct_total_pred"]), _num(df["score_total_pred"]), _num(df["total_model_gap"])


def _family_columns(df: pd.DataFrame, market: str) -> list[str]:
    prefix=f"resid__{market}__"
    return sorted(c for c in df.columns if c.startswith(prefix) and not c.endswith("__ALL_STATS"))


def build_edge_frame(oof: pd.DataFrame, market: str) -> pd.DataFrame:
    actual,base,residual=_market_parts(oof,market)
    core,direct,score,internal_gap=_core_parts(oof,market)
    core_gap=core-base
    direct_gap=direct-base
    score_gap=score-base

    out=pd.DataFrame(index=oof.index)
    out["Season"]=_num(oof["Season"])
    if "physical_game_id" in oof: out["physical_game_id"]=oof["physical_game_id"].astype(str)
    out["market_base"]=base
    out["actual"]=actual
    out["market_residual"]=residual
    out["core_pred"]=core
    out["core_gap"]=core_gap
    out["abs_core_gap"]=core_gap.abs()
    out["direct_gap"]=direct_gap
    out["score_gap"]=score_gap
    out["direct_score_gap"]=(direct-score).abs()
    out["core_internal_gap"]=_num(internal_gap).abs()
    out["direct_score_same_direction"]=(
        direct_gap.notna() & score_gap.notna() & direct_gap.ne(0) & score_gap.ne(0)
        & np.sign(direct_gap).eq(np.sign(score_gap))
    ).astype(float)
    out["market_magnitude"]=base.abs()

    # Use nested STAT only as a historically leak-safe OOF agreement signal.
    # Missing early-fold signal is encoded as zero plus an availability flag.
    stat_col=f"nested_stable_correction__{market}"
    stat=_num(oof[stat_col]) if stat_col in oof else pd.Series(np.nan,index=oof.index)
    out["stat_signal_available"]=stat.notna().astype(float)
    stat=stat.fillna(0.0)
    out["stat_gap"]=stat
    out["abs_stat_gap"]=stat.abs()
    out["core_stat_same_direction"]=(
        core_gap.notna() & core_gap.ne(0) & stat.ne(0) & np.sign(core_gap).eq(np.sign(stat))
    ).astype(float)

    # Repurpose V1.9.3 CORE challengers as epistemic disagreement diagnostics,
    # not as extra votes. They failed to beat incumbent CORE on MAE.
    chal_cols=[c for c in oof.columns if c.startswith(f"core_chal_{'margin' if market=='spreads' else 'total'}__")]
    chal_gaps=[]
    for c in sorted(chal_cols):
        g=_num(oof[c])-base
        key=c.split("__",1)[1].lower()
        out[f"chal_gap__{key}"]=g
        out[f"chal_agrees_core__{key}"]=(g.notna() & g.ne(0) & core_gap.ne(0) & np.sign(g).eq(np.sign(core_gap))).astype(float)
        chal_gaps.append(g)
    if chal_gaps:
        cg=pd.concat(chal_gaps,axis=1)
        out["challenger_gap_dispersion"]=cg.std(axis=1,ddof=0)
        out["challenger_mean_gap"]=cg.mean(axis=1)
        out["challenger_core_agreement_count"]=sum(out[c] for c in out.columns if c.startswith("chal_agrees_core__"))
    else:
        out["challenger_gap_dispersion"]=np.nan
        out["challenger_mean_gap"]=np.nan
        out["challenger_core_agreement_count"]=0.0

    # Residual-family contribution diagnostics. These are OOF residual predictions,
    # not independent votes; summary features only.
    fcols=_family_columns(oof,market)
    if fcols:
        fam=oof[fcols].apply(pd.to_numeric,errors="coerce")
        signs=np.sign(fam)
        csign=np.sign(core_gap).to_numpy()[:,None]
        out["family_same_direction_count"]=((signs.to_numpy()==csign)&np.isfinite(signs.to_numpy())&(csign!=0)).sum(axis=1)
        out["family_opposite_direction_count"]=((signs.to_numpy()==-csign)&np.isfinite(signs.to_numpy())&(csign!=0)).sum(axis=1)
        out["family_mean_abs_correction"]=fam.abs().mean(axis=1)
        out["family_max_abs_correction"]=fam.abs().max(axis=1)
    else:
        out["family_same_direction_count"]=0.0
        out["family_opposite_direction_count"]=0.0
        out["family_mean_abs_correction"]=np.nan
        out["family_max_abs_correction"]=np.nan

    for c in EDGE_CONTEXT_FEATURES:
        if c in oof.columns:
            out[c]=_num(oof[c])

    # Target: whether following incumbent CORE away from the market was directionally
    # correct. Market pushes and zero CORE disagreement are excluded from fitting.
    valid=core_gap.notna() & residual.notna() & core_gap.ne(0) & residual.ne(0)
    out["edge_target"]=np.where(valid,(np.sign(core_gap)==np.sign(residual)).astype(float),np.nan)
    out["candidate_direction"]=np.sign(core_gap)
    return out


def _base_feature_names(edge: pd.DataFrame) -> list[str]:
    preferred=[
        "abs_core_gap","direct_score_gap","core_internal_gap","direct_score_same_direction",
        "market_magnitude","challenger_gap_dispersion","challenger_mean_gap","challenger_core_agreement_count",
        "family_same_direction_count","family_opposite_direction_count","family_mean_abs_correction","family_max_abs_correction",
    ] + list(EDGE_CONTEXT_FEATURES)
    out=[]
    for c in preferred:
        if c not in edge.columns: continue
        x=_num(edge[c])
        if x.notna().any() and x.nunique(dropna=True)>1:
            out.append(c)
    return out


def _consensus_feature_names(edge: pd.DataFrame) -> list[str]:
    names=_base_feature_names(edge)+["stat_signal_available","abs_stat_gap","core_stat_same_direction"]
    out=[]
    for c in names:
        if c not in edge.columns: continue
        x=_num(edge[c])
        if x.notna().any() and x.nunique(dropna=True)>1:
            out.append(c)
    return list(dict.fromkeys(out))


def _pipeline():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("logit",LogisticRegression(C=LOGISTIC_C,max_iter=3000,solver="lbfgs")),
    ])


def _fit_gate(train: pd.DataFrame, valid: pd.DataFrame, features: list[str]):
    y=_num(train.edge_target)
    mask=y.notna()
    usable=[]
    for c in features:
        if c not in train.columns: continue
        x=_num(train.loc[mask,c])
        if x.notna().any() and x.nunique(dropna=True)>1:
            usable.append(c)
    if int(mask.sum()) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"NFL_V1_9_4_EDGE_GATE_TOO_FEW_TRAIN_ROWS n={int(mask.sum())}")
    if not usable:
        raise RuntimeError("NFL_V1_9_4_EDGE_GATE_NO_USABLE_FEATURES")
    model=_pipeline(); model.fit(train.loc[mask,usable],y.loc[mask].astype(int))
    p=model.predict_proba(valid[usable])[:,1]
    return model,p,usable


def _probability_scorecard(df: pd.DataFrame, pcol: str) -> dict:
    y=_num(df.edge_target); p=_num(df[pcol]); m=y.notna()&p.notna()
    if not m.any(): return {"n":0}
    yy=y[m]; pp=p[m]
    out={
        "n":int(m.sum()),
        "brier":round(_brier(yy,pp),6),
        "log_loss":round(_logloss(yy,pp),6),
        "auc":None,
        "naive_50_brier":0.25,
        "naive_50_log_loss":round(math.log(2),6),
        "mean_probability":round(float(pp.mean()),6),
        "actual_rate":round(float(yy.mean()),6),
    }
    a=_auc(yy,pp); out["auc"]=round(a,6) if a is not None else None
    # Fixed calibration bins, never optimized.
    bins=[0.0,0.45,0.50,0.53,0.55,0.57,0.60,0.65,1.000001]
    labels=["<45","45-50","50-53","53-55","55-57","57-60","60-65","65+"]
    b=pd.cut(pp,bins=bins,labels=labels,right=False,include_lowest=True)
    cal={}
    for lab in labels:
        q=(b==lab)
        if not q.any(): continue
        cal[lab]={"n":int(q.sum()),"mean_p":round(float(pp[q].mean()),6),"actual_rate":round(float(yy[q].mean()),6)}
    out["calibration_bins"]=cal
    return out


def _season_probability_scorecard(df: pd.DataFrame, pcol: str) -> dict:
    out={}
    for year,q in df.groupby(_num(df.Season)):
        if pd.isna(year): continue
        out[str(int(year))]=_probability_scorecard(q,pcol)
    return out


def _gate_result(edge: pd.DataFrame, mask: pd.Series) -> dict:
    y=_num(edge.edge_target); m=mask.fillna(False)&y.notna()
    n=int(m.sum())
    if n==0: return {"n":0,"wins":0,"rate":None,"wilson95":[None,None],"by_season":{}}
    wins=int(y[m].sum()); lo,hi=_wilson(wins,n)
    by={}
    for year,q in edge.loc[m].groupby(_num(edge.loc[m,"Season"])):
        yy=_num(q.edge_target).dropna(); nn=len(yy); ww=int(yy.sum())
        by[str(int(year))]={"n":nn,"wins":ww,"rate":round(ww/nn,6) if nn else None}
    return {"n":n,"wins":wins,"rate":round(wins/n,6),"wilson95":[round(lo,6),round(hi,6)],"by_season":by}


def transparent_gates(edge: pd.DataFrame) -> OrderedDict:
    a=_num(edge.abs_core_gap)
    same=_num(edge.core_stat_same_direction).fillna(0).eq(1)
    ds=_num(edge.direct_score_same_direction).fillna(0).eq(1)
    avail=_num(edge.stat_signal_available).fillna(0).eq(1)
    gates=OrderedDict()
    for t in (2.0,4.0,6.0):
        gates[f"CORE_GAP_{int(t)}_PLUS"]=a.ge(t)
    gates["DIRECT_SCORE_AGREE_GAP4"]=ds&a.ge(4.0)
    gates["CORE_STAT_SAME_DIRECTION"]=avail&same
    gates["CORE_STAT_SAME_DIR_GAP4"]=avail&same&a.ge(4.0)
    gates["TRIPLE_AGREE_GAP4"]=avail&same&ds&a.ge(4.0)
    return OrderedDict((name,_gate_result(edge,mask)) for name,mask in gates.items())


def _probability_bands(edge: pd.DataFrame, pcol: str) -> dict:
    p=_num(edge[pcol]); out={}
    for t in FIXED_PROBABILITY_BANDS:
        out[f"P_GE_{int(round(t*100))}"]=_gate_result(edge,p.ge(t))
    return out


def _fair_line_scorecard(oof: pd.DataFrame, market: str) -> dict:
    actual,base,_=_market_parts(oof,market)
    core,_,_,_=_core_parts(oof,market)
    models=OrderedDict({"MARKET":base,"INCUMBENT_CORE":core})
    stat_col=f"nested_stable_correction__{market}"
    if stat_col in oof:
        c=_num(oof[stat_col]); models["NESTED_STAT_CORRECTED_MARKET"]=base+c
    for c in sorted(x for x in oof.columns if x.startswith(f"core_chal_{'margin' if market=='spreads' else 'total'}__")):
        models[c.split("__",1)[1]]=_num(oof[c])
    out={}
    for name,p in models.items():
        m=actual.notna()&p.notna()
        out[name]={"n":int(m.sum()),"mae":round(_mae(actual[m],p[m]),6) if m.any() else None,"rmse":round(_rmse(actual[m],p[m]),6) if m.any() else None,"by_season":{}}
        for year,qidx in oof.loc[m].groupby(_num(oof.loc[m,"Season"])).groups.items():
            ix=list(qidx); aa=actual.loc[ix]; pp=p.loc[ix]
            out[name]["by_season"][str(int(year))]={"n":len(ix),"mae":round(_mae(aa,pp),6),"rmse":round(_rmse(aa,pp),6)}
    return out


def _fit_final_model(edge: pd.DataFrame, features: list[str]):
    y=_num(edge.edge_target); mask=y.notna()
    usable=[]
    for c in features:
        if c not in edge: continue
        x=_num(edge.loc[mask,c])
        if x.notna().any() and x.nunique(dropna=True)>1: usable.append(c)
    if int(mask.sum())<MIN_TRAIN_ROWS or not usable:
        return None,[]
    model=_pipeline(); model.fit(edge.loc[mask,usable],y.loc[mask].astype(int))
    return model,usable


def run_edge_gate_research(oof: pd.DataFrame, structured_report: dict, *, log_func=print):
    if oof is None or oof.empty: raise RuntimeError("NFL_V1_9_4_EDGE_GATE_EMPTY_OOF")
    seasons=_num(oof.Season)
    if int(seasons.max())>MAX_RESEARCH_SEASON: raise RuntimeError("NFL_V1_9_4_2026_DATA_FORBIDDEN")

    report={"status":"NFL_V1_9_4_EDGE_GATE_RESEARCH_COMPLETE","source_tag":SOURCE_TAG,"year_2026_queried":False,"production_authority":0,"markets":{}}
    bundle={"metadata":{"source_tag":SOURCE_TAG,"max_training_season":MAX_RESEARCH_SEASON,"production_authority":0},"markets":{}}

    for market in ("spreads","totals"):
        edge=build_edge_frame(oof,market)
        base_features=_base_feature_names(edge)
        consensus_features=_consensus_feature_names(edge)
        edge["core_gate_oof_p"]=np.nan
        edge["consensus_gate_oof_p"]=np.nan
        folds=[]
        for year in GATE_VALIDATION_YEARS:
            tr=edge.loc[seasons.lt(year)].copy(); va=edge.loc[seasons.eq(year)].copy()
            fold={"validate_season":year,"train_seasons":sorted(int(x) for x in _num(tr.Season).dropna().unique())}
            m,p,used=_fit_gate(tr,va,base_features)
            edge.loc[va.index,"core_gate_oof_p"]=p
            fold["core_gate_features"]=used
            # Consensus model may collapse to the same feature set in early folds;
            # that is acceptable and recorded rather than forcing synthetic STAT.
            m2,p2,used2=_fit_gate(tr,va,consensus_features)
            edge.loc[va.index,"consensus_gate_oof_p"]=p2
            fold["consensus_gate_features"]=used2
            folds.append(fold)

        core_mask=edge.core_gate_oof_p.notna()&edge.edge_target.notna()
        cons_mask=edge.consensus_gate_oof_p.notna()&edge.edge_target.notna()
        selected_resid=(structured_report.get("prospective_selected_residual_families",{}) or {}).get(market,[])
        prospective_consensus_applicable=bool(selected_resid)

        market_report={
            "fair_line_scorecard":_fair_line_scorecard(oof,market),
            "transparent_gates":transparent_gates(edge),
            "core_gate_probability":_probability_scorecard(edge.loc[core_mask],"core_gate_oof_p"),
            "core_gate_by_season":_season_probability_scorecard(edge.loc[core_mask],"core_gate_oof_p"),
            "core_gate_fixed_probability_bands":_probability_bands(edge.loc[core_mask],"core_gate_oof_p"),
            "consensus_gate_probability":_probability_scorecard(edge.loc[cons_mask],"consensus_gate_oof_p"),
            "consensus_gate_by_season":_season_probability_scorecard(edge.loc[cons_mask],"consensus_gate_oof_p"),
            "consensus_gate_fixed_probability_bands":_probability_bands(edge.loc[cons_mask],"consensus_gate_oof_p"),
            "prospective_selected_residual_families":selected_resid,
            "prospective_consensus_gate_applicable":prospective_consensus_applicable,
            "folds":folds,
            "selection_policy":"NO_THRESHOLD_SELECTED; all fixed probability bands remain shadow diagnostics",
        }
        report["markets"][market]=market_report

        final_core,final_core_features=_fit_final_model(edge,base_features)
        final_cons,final_cons_features=_fit_final_model(edge,consensus_features)
        bundle["markets"][market]={
            "core_gate_model":final_core,
            "core_gate_features":final_core_features,
            "consensus_gate_model":final_cons if prospective_consensus_applicable else None,
            "consensus_gate_features":final_cons_features if prospective_consensus_applicable else [],
            "prospective_consensus_gate_applicable":prospective_consensus_applicable,
            "selected_residual_families":selected_resid,
            "fixed_probability_bands":FIXED_PROBABILITY_BANDS,
        }

    spec={
        "source_tag":SOURCE_TAG,
        "validation_years":GATE_VALIDATION_YEARS,
        "fixed_probability_bands":FIXED_PROBABILITY_BANDS,
        "logistic_c":LOGISTIC_C,
        "context_features":EDGE_CONTEXT_FEATURES,
        "market_feature_policy":"pregame market baseline/magnitude plus OOF model disagreement; no postgame predictors",
        "threshold_policy":"fixed bands reported; no automatic promotion",
        "year_2026_queried":False,
        "production_authority":0,
    }
    sha=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    report["edge_gate_registry_sha256"]=sha
    bundle["metadata"]["edge_gate_registry_sha256"]=sha
    bundle["metadata"]["spec"]=spec
    log_func("[NFL-V1.9.4-EDGE-GATE] "+json.dumps(report,sort_keys=True,default=str))
    return report,bundle
