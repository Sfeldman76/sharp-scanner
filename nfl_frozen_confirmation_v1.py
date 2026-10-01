"""NFL V1.9 frozen 2026 confirmation + prospective ledger readiness.

Purpose
-------
Open the previously sealed 2026 season exactly once against hypotheses and model
specifications frozen at the end of V1.8. No 2026 row may influence fitting,
threshold selection, feature selection, system definition, or calibration.

This route is research-only and has zero production betting authority.
"""
from __future__ import annotations

import json, math, hashlib
from typing import Iterable
import numpy as np
import pandas as pd

from nfl_feature_audit_v1 import VIEW
from nfl_challenger_v1 import REQUIRED_COLUMNS, IDENTITY, physical_games
import nfl_challenger_v1 as _challenger_module
from nfl_specialized_v1 import _fit_regression, _h2h_predictions
from nfl_score_engine_v1 import _mask_unobserved_prior_season, _validate_side_grain, _side_predictions, _physical_game_predictions, _swap_within_game
from nfl_intelligence_v1 import (
    _float_array, _physical_id_frame, _status_from_margin, _bigal_systems,
    _generate_oof_core, _attach_side_calc_to_games,
)
from nfl_stats_context_v1 import (
    RAW, FEATURE_FAMILIES, RIDGE_ALPHA, RAW_STATS_COLUMNS, build_raw_stats_query,
    derive_stats_context, attach_stats_features, development_oof, holdout_correction, context_holdout_slices,
)

SOURCE_TAG = "nfl-frozen-confirmation-v1.9.1-expanded-stats-2026-holdout-20261001"
HOLDOUT_YEAR = 2026
TRAIN_MAX_YEAR = 2025
EXTRA_MARKET_COLUMNS = ("Opening_Spread", "Opening_Total")

# FROZEN BEFORE 2026 IS QUERIED BY THIS ROUTE. Do not alter after observing V1.9.
FROZEN_REGISTRY = {
    "CORE_SPREAD_EDGE_5_PLUS": {
        "market":"spreads", "definition":"both frozen CORE margin models agree on side and |consensus edge| >= 5.0 points",
        "direction":"FOLLOW_CORE", "status":"PRIMARY_FROZEN_TEST"
    },
    "REVENGE_CONSENSUS_UNDER_4": {
        "market":"totals", "definition":"revenge flag and both frozen CORE totals models agree UNDER with consensus edge <= -4.0",
        "direction":"UNDER", "status":"PRIMARY_FROZEN_TEST"
    },
    "REVENGE_CONSENSUS_OVER_4": {
        "market":"totals", "definition":"revenge flag and both frozen CORE totals models agree OVER with consensus edge >= +4.0",
        "direction":"OVER", "status":"NEGATIVE_CONTROL_FROZEN_TEST"
    },
    "PRIMETIME_TOTAL_MODELS_WITHIN_2": {
        "market":"totals", "definition":"primetime and direct/score total predictions within 2 points",
        "direction":"UNDER", "status":"WATCHLIST_FROZEN_TEST"
    },
    "BA-NFL2": {"market":"spreads", "definition":"documented late-season home dog off two SU losses vs opponent <= .500", "direction":"SYSTEM", "status":"NAMED_SYSTEM_FROZEN_TEST"},
    "BA-NFL2-ATS": {"market":"spreads", "definition":"BA-NFL2 plus opponent off ATS loss", "direction":"SYSTEM", "status":"NAMED_SYSTEM_FROZEN_TEST"},
    "H2H_CORE_VS_MARKET": {"market":"h2h", "definition":"frozen H2H_BLEND50 probability compared with retrospective closing no-vig market", "direction":"PROBABILITY_METRICS", "status":"GLOBAL_FROZEN_TEST"},
    "CORE_EDGE5_CONTEXT_SLICES": {
        "market":"spreads",
        "definition":"frozen CORE 5+ point edge evaluated without retuning in division, conference, same-season rematch, division-rematch, role-flip and venue slices",
        "direction":"FOLLOW_CORE", "status":"CONTEXT_CONFIRMATION_ONLY"
    },
    "STATS_CONTEXT_MARKET_CORRECTION": {
        "market":"spreads+totals",
        "definition":"fixed Ridge residual corrections to retrospective closing market using only existing box-score-derived prior context families",
        "direction":"CONTINUOUS_RESIDUAL", "status":"STATS_ONLY_FROZEN_CHALLENGER"
    },
}

FROZEN_STATS_REGISTRY = {
    "source":"existing BigDataBall NFL box-score dataset only",
    "external_data":False,
    "ridge_alpha":RIDGE_ALPHA,
    "feature_families":{k:list(v) for k,v in FEATURE_FAMILIES.items()},
    "development_folds":[2021,2022,2023,2024,2025],
    "holdout":2026,
    "development_results_are_descriptive_only":True,
    "no_2026_feature_selection":True,
    "no_automatic_promotion":True,
}


def registry_hash() -> str:
    payload=json.dumps({"hypotheses":FROZEN_REGISTRY,"stats_registry":FROZEN_STATS_REGISTRY},sort_keys=True,separators=(",",":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def build_confirmation_query(view_columns: Iterable[str] | None=None) -> str:
    cols=tuple(dict.fromkeys((*REQUIRED_COLUMNS,*EXTRA_MARKET_COLUMNS)))
    if view_columns is not None:
        miss=set(cols)-set(view_columns)
        if miss: raise RuntimeError("[NFL-V1.9-HOLD] SOURCE_COLUMNS_MISSING "+str(sorted(miss)))
    quoted=", ".join(f"`{c}`" for c in cols)
    return (f"SELECT {quoted} FROM `{VIEW}` WHERE Season BETWEEN 2017 AND 2026 "
            "AND Season_Stage IN ('REGULAR','POSTSEASON') AND Historical_Core_Eligible=1 "
            "ORDER BY Season, Game_Date, Source_Name, Source_Game_ID, Team_Norm")



def _physical_games_with_2026(df: pd.DataFrame) -> pd.DataFrame:
    """Use the already-audited physical-game builder while locally opening 2026.

    The underlying function's only 2026 block is the global experiment-season
    guard. The pairing/label logic is unchanged. Restore the module constant
    immediately so V1.8 remains sealed in every other route.
    """
    old=_challenger_module.EXPERIMENT_SEASONS
    try:
        _challenger_module.EXPERIMENT_SEASONS=tuple(range(2017,2027))
        return physical_games(df)
    finally:
        _challenger_module.EXPERIMENT_SEASONS=old

def _prepare_state_allow_2026(side_rows: pd.DataFrame) -> pd.DataFrame:
    d=side_rows.copy(); _validate_side_grain(d)
    d["Season"]=pd.to_numeric(d.Season,errors="coerce").astype(int)
    if not d.Season.between(2017,2026).all(): raise RuntimeError("[NFL-V1.9-HOLD] BAD_SEASON")
    d["physical_game_id"]=_physical_id_frame(d)
    d["actual_margin"]=pd.to_numeric(d.Team_Score,errors="coerce")-pd.to_numeric(d.Opponent_Score,errors="coerce")
    sp=pd.to_numeric(d.Spread_Value,errors="coerce")
    d["ATS_status_current"]=[_status_from_margin(float(m+s)) if np.isfinite(m) and np.isfinite(s) else "MISSING" for m,s in zip(pd.to_numeric(d.actual_margin,errors="coerce"),sp)]
    d["SU_win_current"]=pd.to_numeric(d.actual_margin,errors="coerce").gt(0).astype(float)
    d["SU_loss_current"]=pd.to_numeric(d.actual_margin,errors="coerce").lt(0).astype(float)
    d["SU_tie_current"]=pd.to_numeric(d.actual_margin,errors="coerce").eq(0).astype(float)
    d["dog_current"]=sp.gt(0).astype(float)
    d=d.sort_values(["Season","Team_Norm","Game_Date","Source_Name","Source_Game_ID"],kind="mergesort").copy()
    grp=d.groupby(["Season","Team_Norm"],sort=False,dropna=False)
    d["calc_prev1_su_loss"]=grp["SU_loss_current"].shift(1); d["calc_prev2_su_loss"]=grp["SU_loss_current"].shift(2)
    d["calc_prev1_su_win"]=grp["SU_win_current"].shift(1); d["calc_prev1_points"]=grp["Team_Score"].shift(1)
    d["calc_prev1_ats_loss"]=grp["ATS_status_current"].shift(1).eq("LOSS").astype(float)
    games_before=grp.cumcount().astype(float); wins_before=grp["SU_win_current"].cumsum()-d["SU_win_current"]; ties_before=grp["SU_tie_current"].cumsum()-d["SU_tie_current"]
    d["calc_win_pct_prior"]=np.where(games_before.gt(0),(wins_before+.5*ties_before)/games_before,np.nan)
    dogs_before=grp["dog_current"].cumsum()-d["dog_current"]; d["calc_dog_rate_prior"]=np.where(games_before.gt(0),dogs_before/games_before,np.nan)
    for c in ("calc_prev1_su_loss","calc_prev2_su_loss","calc_prev1_su_win","calc_prev1_points","calc_prev1_ats_loss","calc_win_pct_prior","calc_dog_rate_prior"):
        d["Opp_"+c]=_swap_within_game(_float_array(d[c]),d)
    postseason={int(sy):set(p.Team_Norm.astype(str).str.lower()) for sy,p in d.loc[d.Season_Stage.eq("POSTSEASON")].groupby("Season")}
    d["calc_prior_season_playoff"]=[float(str(t).lower() in postseason.get(int(s)-1,set())) if int(s)-1 in postseason else np.nan for s,t in zip(d.Season,d.Team_Norm)]
    d["Opp_calc_prior_season_playoff"]=_swap_within_game(_float_array(d.calc_prior_season_playoff),d)
    # Do NOT infer the current season's final week from partial observed data.
    # NFL regular seasons used 17 scheduled weeks through 2020 and 18 from 2021 onward.
    # This calendar rule is frozen independently of 2026 outcomes and prevents early
    # 2026 games from falsely triggering the Big Al "final two weeks" system.
    d["calc_regular_final_week"]=np.where(d.Season.le(2020),17.0,18.0)
    return d.sort_values(["Season","Game_Date","Source_Name","Source_Game_ID","Team_Norm"],kind="mergesort").reset_index(drop=True)


def _frozen_core_2026(all_side: pd.DataFrame) -> pd.DataFrame:
    train=all_side.loc[pd.to_numeric(all_side.Season,errors="coerce").le(TRAIN_MAX_YEAR)].copy()
    hold=all_side.loc[pd.to_numeric(all_side.Season,errors="coerce").eq(HOLDOUT_YEAR)].copy()
    if hold.empty: raise RuntimeError("[NFL-V1.9-HOLD] NO_2026_GAMES")
    if len(hold)//2 < 20: raise RuntimeError("[NFL-V1.9-HOLD] TOO_FEW_2026_GAMES")
    train=_mask_unobserved_prior_season(train); hold=_mask_unobserved_prior_season(hold)
    tr_g=_physical_games_with_2026(train); va_g=_physical_games_with_2026(hold)
    tr_g["actual_margin"]=pd.to_numeric(tr_g.Team_Score,errors="coerce")-pd.to_numeric(tr_g.Opponent_Score,errors="coerce")
    tr_g["actual_total"]=pd.to_numeric(tr_g.Team_Score,errors="coerce")+pd.to_numeric(tr_g.Opponent_Score,errors="coerce")
    va_g["actual_margin"]=pd.to_numeric(va_g.Team_Score,errors="coerce")-pd.to_numeric(va_g.Opponent_Score,errors="coerce")
    va_g["actual_total"]=pd.to_numeric(va_g.Team_Score,errors="coerce")+pd.to_numeric(va_g.Opponent_Score,errors="coerce")
    dm=_fit_regression("COMPACT_RIDGE","SPREADS",tr_g,va_g,"actual_margin")
    dt=_fit_regression("BLEND50","TOTALS",tr_g,va_g,"actual_total")
    score_side=_side_predictions(train,hold)["SCORE_BLEND"]
    sg=_physical_game_predictions(hold,score_side)[["physical_game_id","pred_margin","pred_total"]]
    htr=tr_g.loc[tr_g.H2H_label.notna()].copy(); hva=va_g.loc[va_g.H2H_label.notna()].copy(); hp,_=_h2h_predictions(htr,hva); hmap=dict(zip(hva.physical_game_id,hp["H2H_BLEND50"]))
    out=va_g.copy(); out["direct_margin_pred"]=np.asarray(dm,float); out["direct_total_pred"]=np.asarray(dt,float)
    out=out.merge(sg,on="physical_game_id",how="left",validate="one_to_one").rename(columns={"pred_margin":"score_margin_pred","pred_total":"score_total_pred"}); out["h2h_prob"]=out.physical_game_id.map(hmap)
    sp=pd.to_numeric(out.Spread_Value,errors="coerce"); tt=pd.to_numeric(out.Current_Total,errors="coerce")
    out["direct_spread_edge"]=pd.to_numeric(out.direct_margin_pred,errors="coerce")+sp; out["score_spread_edge"]=pd.to_numeric(out.score_margin_pred,errors="coerce")+sp
    out["spread_consensus_edge"]=(out.direct_spread_edge+out.score_spread_edge)/2; out["spread_model_gap"]=(pd.to_numeric(out.direct_margin_pred,errors="coerce")-pd.to_numeric(out.score_margin_pred,errors="coerce")).abs()
    out["spread_models_agree"]=(np.sign(_float_array(out.direct_spread_edge))==np.sign(_float_array(out.score_spread_edge))) & out.direct_spread_edge.notna() & out.score_spread_edge.notna() & out.direct_spread_edge.ne(0) & out.score_spread_edge.ne(0)
    out["direct_total_edge"]=pd.to_numeric(out.direct_total_pred,errors="coerce")-tt; out["score_total_edge"]=pd.to_numeric(out.score_total_pred,errors="coerce")-tt
    out["total_consensus_edge"]=(out.direct_total_edge+out.score_total_edge)/2; out["total_model_gap"]=(pd.to_numeric(out.direct_total_pred,errors="coerce")-pd.to_numeric(out.score_total_pred,errors="coerce")).abs()
    out["total_models_agree"]=(np.sign(_float_array(out.direct_total_edge))==np.sign(_float_array(out.score_total_edge))) & out.direct_total_edge.notna() & out.score_total_edge.notna() & out.direct_total_edge.ne(0) & out.score_total_edge.ne(0)
    out["core_margin_pred"]=(pd.to_numeric(out.direct_margin_pred,errors="coerce")+pd.to_numeric(out.score_margin_pred,errors="coerce"))/2
    out["core_total_pred"]=(pd.to_numeric(out.direct_total_pred,errors="coerce")+pd.to_numeric(out.score_total_pred,errors="coerce"))/2
    out["h2h_market_delta"]=pd.to_numeric(out.h2h_prob,errors="coerce")-pd.to_numeric(out.H2H_close_novig_reference,errors="coerce")
    return out


def _rate(correct: pd.Series) -> dict:
    x=pd.Series(correct).dropna().astype(bool); n=len(x); w=int(x.sum())
    return {"n":n,"wins":w,"losses":n-w,"rate":round(w/n,6) if n else None}


def _logloss(y,p):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),1e-6,1-1e-6); m=np.isfinite(y)&np.isfinite(p)
    return float(-np.mean(y[m]*np.log(p[m])+(1-y[m])*np.log(1-p[m]))) if m.any() else None


def _brier(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); m=np.isfinite(y)&np.isfinite(p)
    return float(np.mean((y[m]-p[m])**2)) if m.any() else None


def _evaluate_frozen(core: pd.DataFrame, state: pd.DataFrame) -> dict:
    c=core.copy(); state26=state.loc[state.Season.eq(HOLDOUT_YEAR)].copy()
    ats=pd.to_numeric(c.actual_margin,errors="coerce")+pd.to_numeric(c.Spread_Value,errors="coerce"); valid_ats=ats.notna()&~np.isclose(ats,0)
    ce=pd.to_numeric(c.spread_consensus_edge,errors="coerce"); agree=c.spread_models_agree.astype(bool)
    m=valid_ats&agree&ce.abs().ge(5); spread5=_rate(pd.Series(np.sign(ce[m].to_numpy(float))==np.sign(ats[m].to_numpy(float))))
    # Pre-frozen neighborhood is reported descriptively; primary test remains 5+.
    neigh=[]
    for t in (3.0,3.5,4.0,4.5,5.0):
        mm=valid_ats&agree&ce.abs().ge(t); r=_rate(pd.Series(np.sign(ce[mm].to_numpy(float))==np.sign(ats[mm].to_numpy(float)))); neigh.append({"threshold":t,**r})

    actual_total=pd.to_numeric(c.actual_total,errors="coerce")-pd.to_numeric(c.Current_Total,errors="coerce"); rev=pd.to_numeric(c.Revenge_Flag_Current,errors="coerce").eq(1); te=pd.to_numeric(c.total_consensus_edge,errors="coerce"); ta=c.total_models_agree.astype(bool)
    under=rev&ta&te.le(-4)&actual_total.notna()&~np.isclose(actual_total,0); over=rev&ta&te.ge(4)&actual_total.notna()&~np.isclose(actual_total,0)
    revenge_under=_rate(actual_total[under].lt(0)); revenge_over=_rate(actual_total[over].gt(0))
    prim=pd.to_numeric(c.Is_PrimeTime,errors="coerce").eq(1)&pd.to_numeric(c.total_model_gap,errors="coerce").le(2)&actual_total.notna()&~np.isclose(actual_total,0)
    prim_under=_rate(actual_total[prim].lt(0))

    # Named Big Al systems reconstructed from the entire chronology, then graded only on 2026.
    _,plays=_bigal_systems(state); ids=set(state26.physical_game_id.astype(str)); p=plays.loc[plays.physical_game_id.astype(str).isin(ids)] if not plays.empty else pd.DataFrame()
    ba={}
    for sid in ("BA-NFL2","BA-NFL2-ATS"):
        q=p.loc[p.system_id.eq(sid)] if not p.empty else pd.DataFrame(); s=q.status.astype(str).str.upper() if not q.empty else pd.Series(dtype=str); n=int(s.isin(["WIN","LOSS"]).sum()); w=int(s.eq("WIN").sum()); ba[sid]={"n":n,"wins":w,"losses":int(s.eq("LOSS").sum()),"pushes":int(s.eq("PUSH").sum()),"rate":round(w/n,6) if n else None}

    # H2H must beat market probability, not 50%.
    y=pd.to_numeric(c.H2H_label,errors="coerce"); pm=pd.to_numeric(c.H2H_close_novig_reference,errors="coerce"); pc=pd.to_numeric(c.h2h_prob,errors="coerce"); hv=y.notna()&pm.between(0,1)&pc.between(0,1)
    h2h={"n":int(hv.sum()),"market_log_loss":round(_logloss(y[hv],pm[hv]),6) if hv.any() else None,"core_log_loss":round(_logloss(y[hv],pc[hv]),6) if hv.any() else None,"market_brier":round(_brier(y[hv],pm[hv]),6) if hv.any() else None,"core_brier":round(_brier(y[hv],pc[hv]),6) if hv.any() else None}
    if hv.any():
        h2h["delta_log_loss_core_minus_market"]=round(h2h["core_log_loss"]-h2h["market_log_loss"],6); h2h["delta_brier_core_minus_market"]=round(h2h["core_brier"]-h2h["market_brier"],6)
        big=hv&(pc-pm).abs().ge(.05); h2h["abs_model_market_gap_5p"]={"n":int(big.sum()),"mean_actual_minus_market":round(float((y[big]-pm[big]).mean()),6) if big.any() else None,"mean_actual_minus_core":round(float((y[big]-pc[big]).mean()),6) if big.any() else None}

    return {"registry_hash":registry_hash(),"holdout_year":HOLDOUT_YEAR,"holdout_games":int(len(c)),"spread_core_5_plus":spread5,"spread_prefrozen_threshold_profile":neigh,
            "revenge_consensus_under_4":revenge_under,"revenge_consensus_over_4_control":revenge_over,"primetime_total_models_within_2_under":prim_under,
            "bigal":ba,"h2h_core_vs_market":h2h,"production_authority":0}


def _uncertainty_holdout(history_side: pd.DataFrame, hold_core: pd.DataFrame) -> dict:
    hist=_generate_oof_core(history_side)
    out={}
    for market,pred,actual in (("spreads","core_margin_pred","actual_margin"),("totals","core_total_pred","actual_total")):
        resid=(pd.to_numeric(hist[actual],errors="coerce")-pd.to_numeric(hist[pred],errors="coerce")).abs().dropna(); q80=float(np.quantile(resid,.8)); q90=float(np.quantile(resid,.9))
        val=(pd.to_numeric(hold_core[actual],errors="coerce")-pd.to_numeric(hold_core[pred],errors="coerce")).abs().dropna()
        out[market]={"calibration_n":int(len(resid)),"holdout_n":int(len(val)),"q80":round(q80,4),"q90":round(q90,4),"coverage80":round(float((val<=q80).mean()),6),"coverage90":round(float((val<=q90).mean()),6)}
    return out


def run_nfl_frozen_confirmation_v1(*,bq_client=None,audit_report=None,log_func=print,ensure_ledger=True):
    if not isinstance(audit_report,dict) or audit_report.get("status")!="READY_FOR_OFFLINE_CHALLENGER_SANDBOX": raise RuntimeError("[NFL-V1.9-HOLD] V1_3_AUDIT_NOT_GREEN")
    from google.cloud import bigquery
    bq=bq_client or bigquery.Client(project="sharplogger")
    cols={f.name for f in bq.get_table(VIEW).schema}; query=build_confirmation_query(cols)
    raw_cols={f.name for f in bq.get_table(RAW).schema}; raw_query=build_raw_stats_query(raw_cols)
    frozen_hash=registry_hash()
    # Critical pre-registration: emit the complete registry BEFORE any 2026 data query.
    log_func(f"[NFL-V1.9.1-PREFLIGHT] status=PASS tag={SOURCE_TAG} holdout=2026 frozen_registry_sha256={frozen_hash} train_through=2025 stats_only=TRUE production_authority=0")
    log_func("[NFL-V1.9.1-FROZEN-REGISTRY] "+json.dumps({"sha256":frozen_hash,"hypotheses":FROZEN_REGISTRY,"stats_registry":FROZEN_STATS_REGISTRY},sort_keys=True))
    df=bq.query(query,job_config=bigquery.QueryJobConfig(use_query_cache=True,maximum_bytes_billed=20*1024**3)).to_dataframe(); _validate_side_grain(df)
    raw=bq.query(raw_query,job_config=bigquery.QueryJobConfig(use_query_cache=True,maximum_bytes_billed=20*1024**3)).to_dataframe()
    counts={str(int(k)):int(v//2) for k,v in df.groupby("Season").size().items()}
    raw_counts={str(int(k)):int(v//2) for k,v in raw.groupby("Season").size().items()}
    if counts != raw_counts: raise RuntimeError("[NFL-V1.9.1-HOLD] RAW_CONTEXT_GRAIN_MISMATCH")
    log_func("[NFL-V1.9.1-GRAIN] "+json.dumps({"source_side_rows":int(len(df)),"physical_games":int(len(df)//2),"games_by_season":counts,"raw_stats_side_rows":int(len(raw))},sort_keys=True))
    # Explicit leakage gate: all fitting data are sliced before any model call.
    train=df.loc[pd.to_numeric(df.Season,errors="coerce").le(TRAIN_MAX_YEAR)].copy(); hold=df.loc[pd.to_numeric(df.Season,errors="coerce").eq(HOLDOUT_YEAR)].copy()
    if train.Season.max()!=TRAIN_MAX_YEAR or not hold.Season.eq(HOLDOUT_YEAR).all(): raise RuntimeError("[NFL-V1.9-HOLD] TRAIN_HOLDOUT_SPLIT_FAILED")
    core=_frozen_core_2026(df); state=_prepare_state_allow_2026(df); core=_attach_side_calc_to_games(core,state)
    evaluation=_evaluate_frozen(core,state); uncertainty=_uncertainty_holdout(train,core)

    # Expanded stats-only research: derive every new predictor strictly from prior rows.
    stats_side=derive_stats_context(raw)
    all_games=_physical_games_with_2026(df)
    all_games["actual_margin"]=pd.to_numeric(all_games.Team_Score,errors="coerce")-pd.to_numeric(all_games.Opponent_Score,errors="coerce")
    all_games["actual_total"]=pd.to_numeric(all_games.Team_Score,errors="coerce")+pd.to_numeric(all_games.Opponent_Score,errors="coerce")
    stats_games=attach_stats_features(all_games,stats_side)
    stats_train=stats_games.loc[pd.to_numeric(stats_games.Season,errors="coerce").le(TRAIN_MAX_YEAR)].copy()
    stats_hold=stats_games.loc[pd.to_numeric(stats_games.Season,errors="coerce").eq(HOLDOUT_YEAR)].copy()
    stats_dev=development_oof(stats_train)
    stats_holdout=holdout_correction(stats_train,stats_hold,core)
    context_slices=context_holdout_slices(stats_hold,core)

    log_func("[NFL-V1.9.1-HOLDOUT] "+json.dumps(evaluation,sort_keys=True,default=str))
    log_func("[NFL-V1.9.1-UNCERTAINTY] "+json.dumps(uncertainty,sort_keys=True,default=str))
    log_func("[NFL-V1.9.1-STATS-DEVELOPMENT] "+json.dumps(stats_dev,sort_keys=True,default=str))
    log_func("[NFL-V1.9.1-STATS-HOLDOUT] "+json.dumps(stats_holdout,sort_keys=True,default=str))
    log_func("[NFL-V1.9.1-CONTEXT-HOLDOUT] "+json.dumps(context_slices,sort_keys=True,default=str))
    ledger={"status":"NOT_REQUESTED"}
    if ensure_ledger:
        try:
            from nfl_prospective_ledger_v1 import ensure_tables
            ledger=ensure_tables(bq); log_func("[NFL-V1.9.1-LEDGER-READY] "+json.dumps(ledger,sort_keys=True,default=str))
        except Exception as exc:
            ledger={"status":"ERROR","error":f"{type(exc).__name__}:{exc}"}; log_func("[NFL-V1.9.1-LEDGER-READY] "+json.dumps(ledger,sort_keys=True))
    report={"status":"FROZEN_2026_EXPANDED_STATS_CONFIRMATION_COMPLETE","source_tag":SOURCE_TAG,"registry_hash":frozen_hash,"evaluation":evaluation,"uncertainty":uncertainty,"stats_development":stats_dev,"stats_holdout":stats_holdout,"context_holdout":context_slices,"ledger":ledger,"production_authority":0,"ncaaf":"UNCHANGED","legacy_nfl":"UNCHANGED"}
    log_func("[NFL-V1.9.1-CONTRACT] status=FROZEN_2026_EXPANDED_STATS_CONFIRMATION_COMPLETE train_through=2025 holdout=2026 registry_frozen=TRUE stats_only=TRUE no_2026_tuning=TRUE no_auto_promotion=TRUE production_authority=0 ncaaf=UNCHANGED legacy_nfl=UNCHANGED")
    return report
