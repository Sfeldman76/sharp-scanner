"""NFL Advanced STAT Research V2.4 — orthogonal hidden-signal lab.

Research-only extension.  The frozen NFL production models and Edge Authority V2.3
remain unchanged.  This module asks a deliberately different question from CORE:

    Do football-performance statistics contain incremental information after the
    production CORE feature set has already been explained away?

Method
------
* 2017-2025 historical source only; 2026 is forbidden.
* Uses only prior-game football-stat context from nfl_stats_context_v1.
* Excludes schedule/venue/system/market variables from the advanced STAT families.
* Two Frisch-Waugh-Lovell style targets are tested separately:
    MARKET_ORTHOGONAL:
      1) predict market residual from frozen CORE feature columns;
      2) residualize every STAT feature against those same CORE columns;
      3) fit residualized STAT -> market error unexplained by CORE.
    CORE_CORRECTION:
      1) predict actual margin/total from the frozen CORE feature contract;
      2) residualize every STAT feature against those same CORE columns;
      3) fit residualized STAT -> outcome error unexplained by CORE.
  This prevents STAT from receiving credit for simply relearning CORE and lets
  us distinguish market-mispricing signal from genuine CORE-correction signal.
* Discovery (2021-2023) chooses one threshold per family.  Confirmation
  (2024-2025) is reported unchanged.  Because these years have already been
  inspected during project development, ALL V2.4 results are retrospective
  research only and cannot earn production authority.
* Evaluates four useful roles: CREATE, CONFIRM, VETO, and SYSTEM_FILTER.
* Multiple thresholds/families are reported, not silently discarded.

The PBP/EPA lane is intentionally separate. Historical Validation runs the frozen
PBP attribution diagnostic alongside this module so EPA/success/CPOE/neutral-pass
information can be compared without contaminating this box-score orthogonal test.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

import nfl_production_v1 as prod
import nfl_stat_selector_v23 as stat23
import nfl_stats_context_v1 as stats
import sports_edge_authority_v1 as shared

SOURCE_TAG = "nfl-advanced-stat-v2.4-orthogonal-hidden-signal-research-20261003"
STATUS = "NFL_ADVANCED_STAT_V24_RETROSPECTIVE_RESEARCH_COMPLETE"
SEALED_SEASON = 2026
DISCOVERY_SEASONS = (2021, 2022, 2023)
CONFIRM_SEASONS = (2024, 2025)
ALPHAS = (12.0, 24.0, 48.0)
POINT_THRESHOLDS = (0.50, 0.75, 1.00, 1.50)
TAIL_QUANTILES = (0.70, 0.80, 0.90)
MARKET_NUISANCE_ALPHA = 24.0
CORE_OUTCOME_NUISANCE_ALPHA = 12.0  # matches frozen Spread/Totals production Ridge alpha
FEATURE_NUISANCE_ALPHA = 24.0
TARGET_MODES = ("MARKET_ORTHOGONAL", "CORE_CORRECTION")

# Deliberately omit DIVISION_REMATCH, SCHEDULE_SEQUENCE and VENUE_FORM. Those are
# situational/context concepts and belong to CORE/SYSTEMS, not this STAT lane.
BASE_ADVANCED_FAMILIES = OrderedDict({
    "OPPONENT_ADJUSTED_EFFICIENCY": tuple(stats.FEATURE_FAMILIES["OPPONENT_ADJUSTED"]),
    "RUN_PASS_TRENCH_MATCHUP": tuple(stats.FEATURE_FAMILIES["RUN_PASS_MATCHUP"]),
    "PACE_CONVERSION_EFFICIENCY": tuple(stats.FEATURE_FAMILIES["PACE_EFFICIENCY"]),
    "TURNOVER_REGRESSION": tuple(stats.FEATURE_FAMILIES["TURNOVER_REGRESSION"]),
    "NONOFFENSIVE_SCORING_REGRESSION": tuple(stats.FEATURE_FAMILIES["NONOFFENSIVE_SCORING"]),
    "HALF_GAME_ADJUSTMENT_PROFILE": tuple(stats.FEATURE_FAMILIES["QUARTER_HALF_PROFILE"]),
    "DISCIPLINE_FOURTH_DOWN": tuple(stats.FEATURE_FAMILIES["DISCIPLINE_FOURTH_DOWN"]),
    "VOLATILITY_TREND": tuple(stats.FEATURE_FAMILIES["VOLATILITY_TREND"]),
    "CLOSE_GAME_REGRESSION": tuple(stats.FEATURE_FAMILIES["CLOSE_GAME_STATE"]),
})
ADVANCED_FAMILIES = OrderedDict({
    **BASE_ADVANCED_FAMILIES,
    # Nonlinear interaction families deliberately test information a simple
    # additive CORE/Ridge model cannot express directly. Columns are created by
    # _add_hidden_interactions() strictly from prior-only STAT inputs.
    "MATCHUP_NONLINEAR": (
        "adv_pass_trench_interaction","adv_rush_pass_matchup_interaction","adv_adjusted_efficiency_interaction",
    ),
    "PACE_GAME_SHAPE_NONLINEAR": (
        "adv_pace_efficiency_interaction","adv_possession_efficiency_interaction","adv_conversion_interaction",
    ),
    "SUSTAINABILITY_DIVERGENCE": (
        "adv_turnover_nonoff_interaction","adv_turnover_scoring_tension","adv_close_blowout_tension",
    ),
    "HALF_ADJUSTMENT_ASYMMETRY": (
        "adv_half_swing","adv_half_interaction","adv_first_half_share_volatility",
    ),
    "VOLATILITY_TREND_NONLINEAR": (
        "adv_points_volatility_trend","adv_ypp_volatility_trend","adv_total_volatility_efficiency",
    ),
    "UNDERLYING_EFFICIENCY_COMPOSITE": tuple(dict.fromkeys(
        BASE_ADVANCED_FAMILIES["OPPONENT_ADJUSTED_EFFICIENCY"]
        + BASE_ADVANCED_FAMILIES["RUN_PASS_TRENCH_MATCHUP"]
        + BASE_ADVANCED_FAMILIES["PACE_CONVERSION_EFFICIENCY"]
    )),
    "SUSTAINABILITY_REGRESSION_COMPOSITE": tuple(dict.fromkeys(
        BASE_ADVANCED_FAMILIES["TURNOVER_REGRESSION"]
        + BASE_ADVANCED_FAMILIES["NONOFFENSIVE_SCORING_REGRESSION"]
        + BASE_ADVANCED_FAMILIES["VOLATILITY_TREND"]
        + BASE_ADVANCED_FAMILIES["CLOSE_GAME_REGRESSION"]
    )),
    "COACHING_GAMEFLOW_COMPOSITE": tuple(dict.fromkeys(
        BASE_ADVANCED_FAMILIES["HALF_GAME_ADJUSTMENT_PROFILE"]
        + BASE_ADVANCED_FAMILIES["DISCIPLINE_FOURTH_DOWN"]
    )),
})


def _num(x):
    try:
        z=float(x)
        return z if math.isfinite(z) else np.nan
    except Exception:
        return np.nan


def _sha(x: Any) -> str:
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def _core_cols(market: str) -> list[str]:
    return list(prod.SPREAD_FEATURES if market == "SPREADS" else prod.TOTAL_FEATURES)


def _target_market_residual(d: pd.DataFrame, market: str) -> pd.Series:
    if market == "SPREADS":
        return pd.to_numeric(d.actual_margin,errors="coerce") + pd.to_numeric(d.Spread_Value,errors="coerce")
    if market == "TOTALS":
        return pd.to_numeric(d.actual_total,errors="coerce") - pd.to_numeric(d.Current_Total,errors="coerce")
    raise ValueError(market)


def _target_core_outcome(d: pd.DataFrame, market: str) -> pd.Series:
    if market == "SPREADS":
        return pd.to_numeric(d.actual_margin,errors="coerce")
    if market == "TOTALS":
        return pd.to_numeric(d.actual_total,errors="coerce")
    raise ValueError(market)


def _training_target(d: pd.DataFrame, market: str, target_mode: str) -> pd.Series:
    if target_mode == "MARKET_ORTHOGONAL":
        return _target_market_residual(d,market)
    if target_mode == "CORE_CORRECTION":
        return _target_core_outcome(d,market)
    raise ValueError(target_mode)


def _add_hidden_interactions(d: pd.DataFrame) -> pd.DataFrame:
    """Add predeclared nonlinear prior-only STAT interactions.

    These are not searched formulas.  They represent football mechanisms chosen
    before seeing V2.4 results: trench x passing matchup, pace x efficiency,
    conversion interaction, turnover/non-offensive scoring sustainability,
    first-half/second-half asymmetry, and volatility x trend.
    """
    x=d.copy()
    def n(c):
        return pd.to_numeric(x.get(c),errors="coerce") if c in x.columns else pd.Series(np.nan,index=x.index,dtype=float)
    x["adv_pass_trench_interaction"]=n("pass_matchup_diff") * (-n("sack_matchup_diff"))
    x["adv_rush_pass_matchup_interaction"]=n("rush_matchup_diff") * n("pass_matchup_diff")
    x["adv_adjusted_efficiency_interaction"]=n("adj_off_ypp_context_diff") * (-n("adj_def_ypp_context_diff"))
    x["adv_pace_efficiency_interaction"]=n("plays_context_diff") * n("points_per_play_context_diff")
    x["adv_possession_efficiency_interaction"]=n("time_possession_context_diff") * n("first_down_rate_context_diff")
    x["adv_conversion_interaction"]=n("first_down_rate_context_diff") * n("third_down_context_diff")
    x["adv_turnover_nonoff_interaction"]=n("turnover_margin_context_diff") * n("nonoff_event_context_diff")
    x["adv_turnover_scoring_tension"]=(n("giveaway_context_diff")+n("pass_int_context_diff")+n("fumble_lost_context_diff")) * n("points_trend_context_diff")
    x["adv_close_blowout_tension"]=n("close_game_rate_diff") - n("blowout_rate_diff")
    x["adv_half_swing"]=n("second_half_margin_context_diff") - n("first_half_margin_context_diff")
    x["adv_half_interaction"]=n("first_half_margin_context_diff") * n("second_half_margin_context_diff")
    x["adv_first_half_share_volatility"]=n("first_half_share_context_diff") * n("points_sd_game_sum")
    x["adv_points_volatility_trend"]=n("points_sd_game_sum") * n("points_trend_context_diff")
    x["adv_ypp_volatility_trend"]=n("off_ypp_sd_game_sum") * n("off_ypp_trend_context_diff")
    x["adv_total_volatility_efficiency"]=n("total_sd_game_sum") * n("points_per_play_context_diff")
    return x


def _feature_cols(d: pd.DataFrame, family: str) -> list[str]:
    out=[]
    for c in ADVANCED_FAMILIES[family]:
        if c not in d.columns: continue
        s=pd.to_numeric(d[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
        if s.notna().any(): out.append(c)
    return out


def _matrix(train: pd.DataFrame, valid: pd.DataFrame, cols: list[str]):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    a=train.loc[:,cols].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan).astype(float)
    b=valid.loc[:,cols].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan).astype(float)
    imp=SimpleImputer(strategy="median")
    A=imp.fit_transform(a); B=imp.transform(b)
    sc=StandardScaler()
    return sc.fit_transform(A),sc.transform(B),imp,sc


def _orthogonal_fold(train: pd.DataFrame, valid: pd.DataFrame, *, family: str, market: str, target_mode: str, alpha: float) -> pd.DataFrame:
    from sklearn.linear_model import Ridge
    core_cols=[c for c in _core_cols(market) if c in train.columns and c in valid.columns]
    stat_cols=_feature_cols(train,family)
    if len(core_cols) < 6: raise RuntimeError(f"CORE_NUISANCE_FEATURES_MISSING market={market} n={len(core_cols)}")
    if not stat_cols: raise RuntimeError(f"NO_ADVANCED_FEATURES family={family}")
    ytr=_training_target(train,market,target_mode)
    ok=ytr.notna()
    if int(ok.sum()) < 800: raise RuntimeError(f"TOO_FEW_TRAIN_ROWS family={family} market={market} n={int(ok.sum())}")
    tr=train.loc[ok].copy()
    y=ytr.loc[ok].to_numpy(float)
    Ct,Cv,_,_=_matrix(tr,valid,core_cols)
    Xt,Xv,_,_=_matrix(tr,valid,stat_cols)

    # Nuisance model 1: what CORE-type information already explains in market error.
    nuisance_alpha = CORE_OUTCOME_NUISANCE_ALPHA if target_mode == "CORE_CORRECTION" else MARKET_NUISANCE_ALPHA
    core_model=Ridge(alpha=nuisance_alpha)
    core_model.fit(Ct,y)
    y_resid=y-np.asarray(core_model.predict(Ct),float)
    core_valid=np.asarray(core_model.predict(Cv),float)

    # Nuisance model 2: remove from every STAT feature the portion predictable by CORE.
    x_model=Ridge(alpha=FEATURE_NUISANCE_ALPHA)
    x_model.fit(Ct,Xt)
    Xtr_res=Xt-np.asarray(x_model.predict(Ct),float)
    Xva_res=Xv-np.asarray(x_model.predict(Cv),float)

    stat_model=Ridge(alpha=float(alpha))
    stat_model.fit(Xtr_res,y_resid)
    score=np.asarray(stat_model.predict(Xva_res),float)
    actual_market=_target_market_residual(valid,market).to_numpy(float)
    actual_target=_training_target(valid,market,target_mode).to_numpy(float)
    actual_unexplained=actual_target-core_valid
    return pd.DataFrame({
        "physical_game_id":valid.physical_game_id.astype(str).to_numpy(),
        "season":pd.to_numeric(valid.Season,errors="coerce").to_numpy(),
        "target_mode":target_mode,
        "stat_incremental_score":score,
        "core_nuisance_prediction":core_valid,
        "actual_market_residual":actual_market,
        "actual_unexplained_residual":actual_unexplained,
        "feature_count":len(stat_cols),
    })


def _oof_family(d: pd.DataFrame, *, family: str, market: str, target_mode: str, alpha: float) -> pd.DataFrame:
    parts=[]
    for year in (*DISCOVERY_SEASONS,*CONFIRM_SEASONS):
        tr=d.loc[pd.to_numeric(d.Season,errors="coerce").lt(year)].copy()
        va=d.loc[pd.to_numeric(d.Season,errors="coerce").eq(year)].copy()
        if len(va) < 240: continue
        q=_orthogonal_fold(tr,va,family=family,market=market,target_mode=target_mode,alpha=alpha)
        parts.append(q)
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()


def _reliability_beta(oof: pd.DataFrame) -> float:
    q=oof.loc[pd.to_numeric(oof.season,errors="coerce").isin(DISCOVERY_SEASONS)].copy()
    p=pd.to_numeric(q.stat_incremental_score,errors="coerce").to_numpy(float)
    y=pd.to_numeric(q.actual_unexplained_residual,errors="coerce").to_numpy(float)
    m=np.isfinite(p)&np.isfinite(y)
    if int(m.sum()) < 100 or np.var(p[m]) <= 1e-12:return 0.0
    beta=float(np.cov(p[m],y[m],ddof=0)[0,1]/np.var(p[m]))
    return float(np.clip(beta,0.0,1.0))


def _record_from_direction(d: pd.DataFrame, direction: pd.Series) -> dict:
    actual=pd.to_numeric(d.actual_market_residual,errors="coerce")
    dr=pd.to_numeric(direction,errors="coerce")
    valid=actual.notna()&dr.notna()&dr.ne(0)&~np.isclose(actual,0)
    n=int(valid.sum())
    if not n:return {"n":0,"graded_n":0,"wins":0,"losses":0,"pushes":0,"hit_rate":None,"roi_per_unit":None,"wilson95":[None,None]}
    target=((np.sign(actual[valid].to_numpy(float))*np.sign(dr[valid].to_numpy(float)))>0).astype(float)
    return shared.record(target,np.full(len(target),100.0/110.0))


def _by_season(d: pd.DataFrame, direction: pd.Series, seasons) -> dict:
    return {str(s):_record_from_direction(d.loc[pd.to_numeric(d.season,errors="coerce").eq(s)],direction.loc[pd.to_numeric(d.season,errors="coerce").eq(s)]) for s in seasons}


def _threshold_specs(oof: pd.DataFrame,beta: float) -> list[dict]:
    q=oof.loc[pd.to_numeric(oof.season,errors="coerce").isin(DISCOVERY_SEASONS)].copy()
    scaled=np.abs(pd.to_numeric(q.stat_incremental_score,errors="coerce")*beta)
    specs=[{"kind":"POINT","label":f"PTS_{x:.2f}","threshold":float(x)} for x in POINT_THRESHOLDS]
    for qq in TAIL_QUANTILES:
        vals=scaled[np.isfinite(scaled)]
        if len(vals):
            specs.append({"kind":"TAIL","label":f"TAIL_Q{int(qq*100)}","threshold":float(np.quantile(vals,qq)),"quantile":qq})
    # deterministic dedupe if a percentile lands on a point threshold
    seen=set();out=[]
    for s in specs:
        key=round(float(s["threshold"]),8)
        if key in seen:continue
        seen.add(key);out.append(s)
    return out


def _eval_threshold(oof: pd.DataFrame,beta:float,spec:dict)->dict:
    d=oof.copy()
    score=pd.to_numeric(d.stat_incremental_score,errors="coerce")*float(beta)
    strong=score.abs().ge(float(spec["threshold"]))&score.notna()
    direction=pd.Series(np.where(strong,np.sign(score),0.0),index=d.index,dtype=float)
    disc=d.loc[pd.to_numeric(d.season,errors="coerce").isin(DISCOVERY_SEASONS)].copy()
    conf=d.loc[pd.to_numeric(d.season,errors="coerce").isin(CONFIRM_SEASONS)].copy()
    dd=direction.loc[disc.index]; cd=direction.loc[conf.index]
    return {
        "threshold_spec":spec,
        "discovery":_record_from_direction(disc,dd),
        "confirmation":_record_from_direction(conf,cd),
        "discovery_by_season":_by_season(disc,dd,DISCOVERY_SEASONS),
        "confirmation_by_season":_by_season(conf,cd,CONFIRM_SEASONS),
        "fire_rate_discovery":round(float((dd!=0).mean()),6) if len(dd) else None,
        "fire_rate_confirmation":round(float((cd!=0).mean()),6) if len(cd) else None,
    }


def _positive_roi_seasons(by:dict)->int:
    return sum(1 for r in by.values() if math.isfinite(_num((r or {}).get("roi_per_unit"))) and _num(r.get("roi_per_unit"))>0)


def _discovery_rank(x:dict)->tuple:
    r=x.get("discovery") or {}; by=x.get("discovery_by_season") or {}; w=r.get("wilson95") or [None,None]
    lo=_num(w[0])
    return (
        _positive_roi_seasons(by),
        lo if math.isfinite(lo) else -1,
        _num(r.get("roi_per_unit")) if math.isfinite(_num(r.get("roi_per_unit"))) else -999,
        _num(r.get("hit_rate")) if math.isfinite(_num(r.get("hit_rate"))) else -999,
        math.log1p(int(r.get("n") or 0)),
        -float((x.get("threshold_spec") or {}).get("threshold") or 0),
    )


def _classification(best:dict)->str:
    d=best.get("discovery") or {}; c=best.get("confirmation") or {}
    dn=int(d.get("n") or 0); cn=int(c.get("n") or 0)
    dh=_num(d.get("hit_rate")); ch=_num(c.get("hit_rate")); dr=_num(d.get("roi_per_unit")); cr=_num(c.get("roi_per_unit"))
    if dn>=30 and cn>=15 and dh>shared.BREAK_EVEN_110 and ch>shared.BREAK_EVEN_110 and dr>0 and cr>0:
        return "RETROSPECTIVE_REPEATABLE_REQUIRES_NEW_PROSPECTIVE_TEST"
    if dn>=20 and cn>=10 and ((math.isfinite(dr) and dr>0) or (math.isfinite(cr) and cr>0)):
        return "RESEARCH_WATCH"
    return "NO_INCREMENTAL_EDGE_FOUND"


def _core_interaction(base:pd.DataFrame,oof:pd.DataFrame,beta:float,threshold:float,target_mode:str)->dict:
    d=base[["physical_game_id","season","core_edge","market_residual"]].copy()
    d["physical_game_id"]=d.physical_game_id.astype(str)
    q=oof[["physical_game_id","stat_incremental_score"]].copy();q["physical_game_id"]=q.physical_game_id.astype(str)
    d=d.merge(q,on="physical_game_id",how="inner",validate="one_to_one")
    score=pd.to_numeric(d.stat_incremental_score,errors="coerce")*beta
    core_edge=pd.to_numeric(d.core_edge,errors="coerce")
    core=np.sign(core_edge); stat=np.sign(score)
    strong=score.abs().ge(threshold)&score.notna()&core.ne(0)
    corrected=np.sign(core_edge + score)
    def rec(mask,which="core"):
        z=d.loc[mask].copy()
        if z.empty:return {"n":0,"graded_n":0,"wins":0,"losses":0,"pushes":0,"hit_rate":None,"roi_per_unit":None,"wilson95":[None,None]}
        direction=(core.loc[z.index] if which=="core" else stat.loc[z.index])
        z["actual_market_residual"]=pd.to_numeric(z.market_residual,errors="coerce")
        return _record_from_direction(z,direction)
    out={}
    for seasons,label in ((DISCOVERY_SEASONS,"DISCOVERY"),(CONFIRM_SEASONS,"CONFIRMATION")):
        sy=pd.to_numeric(d.season,errors="coerce").isin(seasons)
        out[label]={
            "CORE_ALL":rec(sy&core.ne(0),"core"),
            "CORE_WHEN_STAT_AGREES":rec(sy&strong&stat.eq(core),"core"),
            "CORE_WHEN_STAT_CONFLICTS":rec(sy&strong&stat.eq(-core),"core"),
            "STAT_STANDALONE_STRONG":rec(sy&strong,"stat"),
        }
        if target_mode == "CORE_CORRECTION":
            z=d.loc[sy&strong].copy()
            if z.empty:
                out[label]["CORE_PLUS_STAT_CORRECTION"]={"n":0,"graded_n":0,"wins":0,"losses":0,"pushes":0,"hit_rate":None,"roi_per_unit":None,"wilson95":[None,None]}
            else:
                z["actual_market_residual"]=pd.to_numeric(z.market_residual,errors="coerce")
                out[label]["CORE_PLUS_STAT_CORRECTION"]=_record_from_direction(z,corrected.loc[z.index])
    return out


def _system_interaction(base:pd.DataFrame,oof:pd.DataFrame,beta:float,threshold:float,sysctx:dict,market:str)->dict:
    d=base[["physical_game_id","season","market_residual"]].copy();d["physical_game_id"]=d.physical_game_id.astype(str)
    q=oof[["physical_game_id","stat_incremental_score"]].copy();q["physical_game_id"]=q.physical_game_id.astype(str)
    d=d.merge(q,on="physical_game_id",how="inner",validate="one_to_one")
    score=pd.to_numeric(d.stat_incremental_score,errors="coerce")*beta
    d["stat_direction"]=np.sign(score);d["stat_strong"]=score.abs().ge(threshold)&score.notna()
    recs=[]
    for ix,r in d.iterrows():
        votes=[v for v in (sysctx.get(str(r.physical_game_id),{}).get("votes") or []) if str(v.get("market"))==market]
        if not votes:continue
        dirs=[int(v.get("direction") or 0) for v in votes if int(v.get("direction") or 0)!=0]
        if not dirs or len(set(dirs))!=1:continue
        recs.append({
            "ix":ix,"season":int(r.season),"market_residual":_num(r.market_residual),
            "system_direction":dirs[0],"system_vote_count":len(dirs),
            "stat_direction":int(r.stat_direction) if math.isfinite(_num(r.stat_direction)) else 0,
            "stat_strong":bool(r.stat_strong),
        })
    x=pd.DataFrame(recs)
    def agg(q):
        if q.empty:return {"n":0,"graded_n":0,"wins":0,"losses":0,"pushes":0,"hit_rate":None,"roi_per_unit":None,"wilson95":[None,None]}
        q=q.copy();q["actual_market_residual"]=q.market_residual
        return _record_from_direction(q,pd.Series(q.system_direction.to_numpy(float),index=q.index))
    out={}
    if x.empty:return {"DISCOVERY":{},"CONFIRMATION":{}}
    for seasons,label in ((DISCOVERY_SEASONS,"DISCOVERY"),(CONFIRM_SEASONS,"CONFIRMATION")):
        z=x.loc[x.season.isin(seasons)]
        out[label]={
            "SYSTEM_ALL":agg(z),
            "SYSTEM_STAT_AGREE":agg(z.loc[z.stat_strong & z.stat_direction.eq(z.system_direction)]),
            "SYSTEM_STAT_CONFLICT":agg(z.loc[z.stat_strong & z.stat_direction.eq(-z.system_direction)]),
            "SYSTEM_STAT_NEUTRAL":agg(z.loc[~z.stat_strong]),
            "SYSTEM_MULTI_ALL":agg(z.loc[z.system_vote_count.ge(2)]),
            "SYSTEM_MULTI_STAT_AGREE":agg(z.loc[z.system_vote_count.ge(2)&z.stat_strong&z.stat_direction.eq(z.system_direction)]),
            "SYSTEM_MULTI_STAT_CONFLICT":agg(z.loc[z.system_vote_count.ge(2)&z.stat_strong&z.stat_direction.eq(-z.system_direction)]),
        }
    return out


def run_advanced_stat_research(*,bq_client,games:pd.DataFrame,replay_rows:pd.DataFrame,sysctx:dict|None=None,log_func=print)->dict:
    log_func("[NFL-STAT-V24-PREFLIGHT] "+json.dumps({
        "status":"START","source_tag":SOURCE_TAG,"year_2026_queried":False,
        "role":"ORTHOGONAL_INCREMENTAL_STAT_RESEARCH_ONLY",
        "core_separation":"FWL_RESIDUALIZE_TARGET_AND_STAT_FEATURES_AGAINST_FROZEN_CORE_FEATURE_SET",
        "families":list(ADVANCED_FAMILIES),"target_modes":list(TARGET_MODES),"alphas":list(ALPHAS),
        "point_thresholds":list(POINT_THRESHOLDS),"tail_quantiles":list(TAIL_QUANTILES),
        "production_authority":0,"confirmation_is_retrospective":True,
    },sort_keys=True))
    sg=_add_hidden_interactions(stat23.build_historical_stats_games(bq_client=bq_client,games=games))
    if int(pd.to_numeric(sg.Season,errors="coerce").max())>=SEALED_SEASON:raise RuntimeError("NFL_STAT_V24_2026_LEAK")
    report={"status":STATUS,"source_tag":SOURCE_TAG,"year_2026_queried":False,"production_authority":0,
            "confirmation_is_retrospective":True,"markets":{},"families":{k:list(v) for k,v in ADVANCED_FAMILIES.items()},
            "method":"FWL_ORTHOGONAL_STATS_VS_CORE_DUAL_TARGET"}
    sysctx=sysctx or {}
    for market in ("SPREADS","TOTALS"):
        base=stat23._core_eval_frame(sg,replay_rows,market)
        fam_out=[]
        rankings={}
        repeatable={}
        for target_mode in TARGET_MODES:
            mode_out=[]
            for family in ADVANCED_FAMILIES:
                alpha_runs=[]
                for alpha in ALPHAS:
                    try:oof=_oof_family(sg,family=family,market=market,target_mode=target_mode,alpha=alpha)
                    except Exception as exc:
                        log_func("[NFL-STAT-V24-FAMILY-HOLD] "+json.dumps({"market":market,"target_mode":target_mode,"family":family,"alpha":alpha,"reason":f"{type(exc).__name__}:{exc}"},sort_keys=True));continue
                    if oof.empty:continue
                    beta=_reliability_beta(oof)
                    variants=[]
                    for spec in _threshold_specs(oof,beta):variants.append(_eval_threshold(oof,beta,spec))
                    if not variants:continue
                    best=sorted(variants,key=_discovery_rank,reverse=True)[0]
                    alpha_runs.append({"alpha":alpha,"beta":beta,"oof":oof,"best":best,"all_thresholds":variants})
                if not alpha_runs:continue
                chosen=sorted(alpha_runs,key=lambda z:_discovery_rank(z["best"]),reverse=True)[0]
                b=chosen["best"];threshold=float((b.get("threshold_spec") or {}).get("threshold") or 0)
                public={
                    "market":market,"target_mode":target_mode,"family":family,"alpha":chosen["alpha"],"reliability_beta":round(float(chosen["beta"]),6),
                    "feature_count":int(pd.to_numeric(chosen["oof"].feature_count,errors="coerce").max()),
                    "selected_threshold":b.get("threshold_spec"),"discovery":b.get("discovery"),"confirmation":b.get("confirmation"),
                    "discovery_by_season":b.get("discovery_by_season"),"confirmation_by_season":b.get("confirmation_by_season"),
                    "fire_rate_discovery":b.get("fire_rate_discovery"),"fire_rate_confirmation":b.get("fire_rate_confirmation"),
                    "classification":_classification(b),"production_authority":0,
                    "core_interaction":_core_interaction(base,chosen["oof"],chosen["beta"],threshold,target_mode),
                    "system_interaction":_system_interaction(base,chosen["oof"],chosen["beta"],threshold,sysctx,market),
                    "threshold_scan":[{
                        "threshold_spec":v.get("threshold_spec"),"discovery":v.get("discovery"),"confirmation":v.get("confirmation"),
                        "fire_rate_discovery":v.get("fire_rate_discovery"),"fire_rate_confirmation":v.get("fire_rate_confirmation")
                    } for v in chosen["all_thresholds"]],
                }
                fam_out.append(public); mode_out.append(public)
                log_func("[NFL-STAT-V24-FAMILY] "+json.dumps({k:public.get(k) for k in ("market","target_mode","family","alpha","reliability_beta","feature_count","selected_threshold","discovery","confirmation","fire_rate_discovery","fire_rate_confirmation","classification")},sort_keys=True,default=str))
                log_func("[NFL-STAT-V24-CORE-INTERACTION] "+json.dumps({"market":market,"target_mode":target_mode,"family":family,"interaction":public["core_interaction"]},sort_keys=True,default=str))
                log_func("[NFL-STAT-V24-SYSTEM-INTERACTION] "+json.dumps({"market":market,"target_mode":target_mode,"family":family,"interaction":public["system_interaction"]},sort_keys=True,default=str))
            ranked=sorted(mode_out,key=lambda x:_discovery_rank({"discovery":x.get("discovery"),"discovery_by_season":x.get("discovery_by_season"),"threshold_spec":x.get("selected_threshold")}),reverse=True)
            rankings[target_mode]=[x["family"] for x in ranked]
            repeatable[target_mode]=[x["family"] for x in ranked if str(x.get("classification")).startswith("RETROSPECTIVE_REPEATABLE")]
            log_func("[NFL-STAT-V24-RANKING] "+json.dumps({"market":market,"target_mode":target_mode,"ranking":[{"rank":i+1,"family":x["family"],"classification":x["classification"],"beta":x["reliability_beta"],"threshold":x["selected_threshold"],"discovery":x["discovery"],"confirmation":x["confirmation"]} for i,x in enumerate(ranked)]},sort_keys=True,default=str))
        report["markets"][market]={"families":fam_out,"ranking_by_target":rankings,"repeatable_research_only_by_target":repeatable}
    stable={"source_tag":SOURCE_TAG,"method":report["method"],"families":list(ADVANCED_FAMILIES),"target_modes":list(TARGET_MODES),"markets":{m:{"ranking_by_target":report["markets"][m]["ranking_by_target"],"repeatable_by_target":report["markets"][m]["repeatable_research_only_by_target"]} for m in report["markets"]}}
    report["research_contract_sha256"]=_sha(stable)
    log_func("[NFL-STAT-V24-CONTRACT] "+json.dumps({
        "status":STATUS,"source_tag":SOURCE_TAG,"research_contract_sha256":report["research_contract_sha256"],
        "spread_repeatable_research_only_by_target":report["markets"].get("SPREADS",{}).get("repeatable_research_only_by_target",{}),
        "total_repeatable_research_only_by_target":report["markets"].get("TOTALS",{}).get("repeatable_research_only_by_target",{}),
        "production_authority":0,"year_2026_queried":False,
        "next_test":"FREEZE_ANY_SURVIVOR_BEFORE_2026_PROSPECTIVE_USE; NO RETROSPECTIVE PROMOTION",
    },sort_keys=True,default=str))
    return report


def _self_test():
    # Pure sanity checks: families exclude obvious CORE context, and ranking does
    # not inspect confirmation when choosing discovery representatives.
    banned={"Is_Home","Is_Division_Game","Rest_Differential_Days","WinPct_Prior_Diff","ATS_WinPct_Prior_Diff","Avg_SU_Margin_Last5_Diff","Week_Number"}
    assert not any(banned.intersection(v) for v in ADVANCED_FAMILIES.values())
    assert TARGET_MODES == ("MARKET_ORTHOGONAL","CORE_CORRECTION")
    a={"discovery":{"n":40,"hit_rate":.56,"roi_per_unit":.07,"wilson95":[.51,.61]},"discovery_by_season":{"2021":{"roi_per_unit":.01},"2022":{"roi_per_unit":.02},"2023":{"roi_per_unit":.03}},"threshold_spec":{"threshold":.75}}
    b=json.loads(json.dumps(a)); b["confirmation"]={"n":100,"hit_rate":0.1,"roi_per_unit":-0.9}
    assert _discovery_rank(a)==_discovery_rank(b)
    return True

if __name__=="__main__":
    assert _self_test()
    print("NFL_ADVANCED_STAT_V24_SELF_TEST_PASS")
