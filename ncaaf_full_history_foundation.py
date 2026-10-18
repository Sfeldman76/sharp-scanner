"""NCAAF full-history, opening-line-independent research foundation.

Standalone score-based, strictly season-forward shadow models for margin, total,
and straight-up wins. Market/open/close/book/ATS/current-game postgame inputs
are forbidden predictors. Historical outcomes are labels only. 2026 is sealed.

This *adds* score-based research; it does not replace market-residual models,
market-only feature gates, published systems, or frozen Production V1.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np
import pandas as pd

CONTRACT = "NCAAF_SCORE_HISTORY_ALL_AVAILABLE_2013_2025_NO_OPEN_MARKET_REQUIRED_V1"
LAST_RESEARCH_YEAR = 2025
MIN_TRAIN_ROWS = 200
MIN_VALID_ROWS = 50
MAX_FEATURES = 56

# Be deliberately conservative. Any uncertain field remains excluded until a
# verified point-in-time source contract exists. Prior-season ratings safe.
_FORBIDDEN = (
    "market", "opening", "closing", "open_", "close_", "spread", "moneyline", "odds",
    "betting", "wager", "_ats_", "ats_", "_clv", "_line_", "line_move", "linevalue",
    "postgame", "post_game", "actual_", "target", "label", "final_", "result_",
    "team_score", "opponent_score", "q1_points", "q2_points", "q3_points",
    "q4_points", "_score_current", "_outcome", "predicted_", "fair_line",
    "prediction_tracker", "pt_meta", "consensus", "bookmaker", "_v1355_",
)
_ALLOWED_PREFIXES = (
    "a_", "b_", "diff_", "sum_", "mean_", "matchup_", "expert_",
    "coverage_", "context_", "state_", "rawseason_", "rawrecent",
    "h2h2_", "prev_", "avg_", "opp_prev_", "opp_avg_", "cfbd_",
)
_CFBD_PREGAME_TOKENS = (
    "_prior", "recent3_", "recent5_", "season_", "recruit", "talent",
    "roster_", "returning_", "portal_", "elo_prior", "home_elevation",
    "venue_elevation", "altitude_", "travel_", "priorseason_",
)

def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _safe_feature(name: str) -> bool:
    lower = str(name).lower()
    if not lower.startswith(_ALLOWED_PREFIXES):
        return False
    if any(t in lower for t in _FORBIDDEN):
        return False
    if lower.startswith("cfbd_"):
        # CFBD_Game_* and CFBD_Opp_Game_* are not proven prior-only.
        if lower.startswith(("cfbd_game_", "cfbd_opp_game_")):
            return False
        if not any(t in lower for t in _CFBD_PREGAME_TOKENS):
            return False
    if lower.startswith("expert_") and any(t in lower for t in ("favorite", "underdog", "role")):
        return False
    return True


def _choose_features(frame: pd.DataFrame, candidates: list[str] | tuple[str, ...]) -> list[str]:
    # Deterministic schema/availability screen, no outcome-driven selection and
    # no use of 2024-2025 confirmation labels or 2026 records.
    pre=frame.loc[_num(frame, "Season").le(2023)]
    ordered=list(dict.fromkeys([*list(candidates),
        *[c for c in frame.columns if str(c).startswith("CFBD_")]]))
    selected=[]
    for c in ordered:
        if c not in pre.columns or not _safe_feature(c):
            continue
        values=pd.to_numeric(pre[c],errors="coerce")
        if int(values.notna().sum())<80 or values.nunique(dropna=True)<2:
            continue
        selected.append(c)
        if len(selected)>=MAX_FEATURES:
            break
    # Explicit intercept gives every historical game a legitimate baseline,
    # even where some later provider-specific features don't yet exist.
    if "Context_Intercept" not in selected:
        selected.insert(0,"Context_Intercept")
    return selected


def _fit_columns(df: pd.DataFrame, training: np.ndarray, features: list[str]) -> list[str]:
    tr=df.loc[training]
    cols=[]
    for c in features:
        if c not in tr:
            continue
        s=pd.to_numeric(tr[c],errors="coerce")
        if s.notna().sum()<max(10,int(0.05*len(tr))):
            continue
        if c!="Context_Intercept" and s.nunique(dropna=True)<2:
            continue
        cols.append(c)
    return cols or ["Context_Intercept"]


def _frame_X(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: _num(frame,c).to_numpy(dtype=float) for c in cols},index=frame.index)


def _model(regression: bool):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    if regression:
        from sklearn.linear_model import Ridge
        final=Ridge(alpha=40.0,random_state=None)
    else:
        from sklearn.linear_model import LogisticRegression
        final=LogisticRegression(C=0.08,max_iter=350,solver="lbfgs")
    return Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",final)])


def _metric_reg(y: np.ndarray, p: np.ndarray) -> dict[str,Any]:
    keep=np.isfinite(y)&np.isfinite(p)
    if not keep.any():
        return {"n":0,"rmse":None,"mae":None}
    diff=y[keep]-p[keep]
    return {"n":int(keep.sum()),"rmse":float(np.sqrt(np.mean(diff*diff))),"mae":float(np.mean(np.abs(diff)))}


def _metric_h2h(margin: np.ndarray,p: np.ndarray) -> dict[str,Any]:
    keep=np.isfinite(margin)&np.isfinite(p)&~np.isclose(margin,0.)
    if not keep.any():
        return {"n":0,"logloss":None,"brier":None}
    y=(margin[keep]>0).astype(float); prob=np.clip(p[keep],1e-5,1-1e-5)
    return {"n":int(keep.sum()),"logloss":float(-np.mean(y*np.log(prob)+(1-y)*np.log(1-prob))),
            "brier":float(np.mean((y-prob)**2))}


def _closing_grade(frame: pd.DataFrame, margin: np.ndarray, total: np.ndarray,
                   prediction: dict[str,np.ndarray]) -> dict[str,Any]:
    """Evaluation-only close ATS/O-U, including years without *any* opener.

    Closing prices are NOT predictor columns, input to fit, or opening values.
    A source-described closing quote is only used to grade historical decisions;
    real-time availability, line provenance, and vig are not assumed here.
    """
    spread_col=next((c for c in ("Consensus_Close_Spread_Audit", "Consensus_Close_Spread") if c in frame),None)
    total_col=next((c for c in ("Consensus_Close_Total_Audit", "Consensus_Close_Total") if c in frame),None)
    def grade(actual_edge: np.ndarray,pred_edge: np.ndarray) -> dict[str,Any]:
        good=np.isfinite(actual_edge)&np.isfinite(pred_edge)&~np.isclose(actual_edge,0.)&~np.isclose(pred_edge,0.)
        return {"n":int(good.sum()),"direction_accuracy":float(np.mean(np.sign(actual_edge[good])==np.sign(pred_edge[good]))) if good.any() else None}
    spread = _num(frame,spread_col).to_numpy(float) if spread_col else np.full(len(frame),np.nan)
    totals = _num(frame,total_col).to_numpy(float) if total_col else np.full(len(frame),np.nan)
    return {"status":"EVALUATION_ONLY_NOT_PREDICTOR", "source_columns":{"spread":spread_col,"total":total_col},
            "spread_vs_close":grade(margin+spread,prediction["margin"]+spread),
            "totals_vs_close":grade(total-totals,prediction["total"]-totals),
            "rule":"NO_OPENING_REQUIRED__NO_PRICE_IN_FIT__NO_ROI_WITHOUT_VERIFIED_ODDS"}


def frame_fingerprint(frame: pd.DataFrame) -> str:
    keys=[x for x in ("Season","Source_Game_ID","Team_Norm","Opponent_Norm","Actual_Margin","Actual_Total") if x in frame]
    if not keys:
        raise ValueError("No stable identity for the research cache")
    arr=pd.util.hash_pandas_object(frame.loc[:,keys].reset_index(drop=True),index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:20]


def run_full_history_foundation(games: pd.DataFrame, candidates: list[str] | tuple[str, ...], *, log_func=print,
                                min_train_rows: int=MIN_TRAIN_ROWS,
                                min_valid_rows: int=MIN_VALID_ROWS) -> dict[str,Any]:
    if games is None or games.empty:
        raise ValueError("historical games are empty")
    frame=games.loc[_num(games,"Season").between(2013,LAST_RESEARCH_YEAR)].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("no 2013-2025 research games")
    season=_num(frame,"Season").to_numpy(float)
    margin=_num(frame,"Actual_Margin").to_numpy(float)
    total=_num(frame,"Actual_Total").to_numpy(float)
    if not np.isfinite(margin).any() or not np.isfinite(total).any():
        raise ValueError("missing score targets")
    # Never require market columns, even if they are entirely absent.
    frame["Context_Intercept"]=1.0
    features=_choose_features(frame,candidates)
    prediction={"margin":np.full(len(frame),np.nan),"total":np.full(len(frame),np.nan),
                "h2h":np.full(len(frame),np.nan)}
    folds=[]
    historical_years=sorted(int(s) for s in np.unique(season[np.isfinite(season)]))
    for val in historical_years[1:]:
        tr=(season<float(val)); va=(season==float(val))
        fit_counts={"margin":int(np.sum(tr&np.isfinite(margin))),
                    "total":int(np.sum(tr&np.isfinite(total))),
                    "h2h":int(np.sum(tr&np.isfinite(margin)&~np.isclose(margin,0.)))}
        result={"validation_year":val,"training_years":[y for y in historical_years if y<val],
                "prior_games":int(tr.sum()),"validation_games":int(va.sum()),
                "fit_target_rows":fit_counts,"score_based":True,"opening_required":False}
        if int(va.sum())<min_valid_rows or min(fit_counts.values())<min_train_rows:
            result["status"]="INSUFFICIENT_SCORE_HISTORY";folds.append(result);continue
        # Regression heads and H2H head are trained *only* on years preceding val.
        feat=_fit_columns(frame,tr,features)
        xtr=_frame_X(frame.loc[tr],feat)
        xval=_frame_X(frame.loc[va],feat)
        for target,y in (("margin",margin),("total",total)):
            good=np.isfinite(y[tr]);model=_model(True)
            model.fit(xtr.loc[good],y[tr][good]);prediction[target][va]=np.asarray(model.predict(xval),float)
        good=np.isfinite(margin[tr])&~np.isclose(margin[tr],0.)
        labels=(margin[tr][good]>0).astype(int)
        if len(np.unique(labels))>1:
            clf=_model(False);clf.fit(xtr.loc[good],labels)
            prediction["h2h"][va]=np.asarray(clf.predict_proba(xval)[:,1],float)
        result["status"]="PASS";result["feature_count"]=len(feat)
        result["margin"]=_metric_reg(margin[va],prediction["margin"][va])
        result["total"]=_metric_reg(total[va],prediction["total"][va])
        result["h2h"]=_metric_h2h(margin[va],prediction["h2h"][va])
        folds.append(result)
        log_func(f"[NCAAF-FULL-HISTORY-OOF] val_season={val} train_score_rows={fit_counts['margin']} "
                 f"val_rows={int(va.sum())} margin_oof={result['margin']['n']} "
                 f"total_oof={result['total']['n']} h2h_oof={result['h2h']['n']} "
                 f"opening_required=FALSE features={len(feat)} authority=0")
    # A separate hierarchical, dynamic challenger; the established score
    # foundation and all market-dependent lanes remain unchanged. It never
    # selects on the 2024/2025 confirmation outcomes or touches production.
    from ncaaf_hierarchical_research import run_hierarchical_dynamic_research
    hierarchical=run_hierarchical_dynamic_research(
        frame, features, prediction, log_func=log_func,
    )
    # Independent historical-family discovery uses only the *year-forward OOF*
    # score foundation as a benchmark. Outcomes from the validation year may
    # grade a candidate, but never enter its training fit.
    from ncaaf_historical_signal_discovery import run_historical_signal_discovery
    signal_discovery=run_historical_signal_discovery(
        frame, prediction, features, log_func=log_func,
    )
    report={
        "historical_signal_discovery":signal_discovery,
        "status":"PASS" if any(x.get("status")=="PASS" for x in folds) else "NO_SCORE_OOF",
        "contract":CONTRACT,"production_authority":0,"automatic_promotion":False,
        "source_years":historical_years,"game_rows":int(len(frame)),
        "score_label_counts":{"margin":int(np.isfinite(margin).sum()),"total":int(np.isfinite(total).sum()),
                              "h2h":int(np.sum(np.isfinite(margin)&~np.isclose(margin,0.)))},
        "feature_count":len(features),"features":features,
        "rejected_market_features":sorted({c for c in candidates if c in frame.columns and not _safe_feature(c)})[:60],
        "folds":folds,
        "aggregate":{"margin":_metric_reg(margin,prediction["margin"]),
                     "total":_metric_reg(total,prediction["total"]),
                     "h2h":_metric_h2h(margin,prediction["h2h"])},
        "closing_line_evaluation":_closing_grade(frame,margin,total,prediction),
        "hierarchical_dynamic_challenger":hierarchical,
        "confirmation_years":[2024,2025],
        "opening_dependent_lanes": "UNCHANGED__KEEP_ONLY_GENUINE_OPENING_MARKET_ELIGIBILITY",
        "leakage_contract":"PRE_GAME_INPUTS_ONLY__YEAR_FORWARD_FIT__CURRENT_RESULTS_TARGET_ONLY__2026_SEALED",
        "fingerprint":frame_fingerprint(frame),
    }
    if any(y in historical_years for y in range(2013,2022)) and not any(x.get("status")=="PASS" and x["validation_year"]<2022 for x in folds):
        report["status"]="FAIL_PRE2022_SCORE_FOLDS_MISSING"
    log_func(f"[NCAAF-FULL-HISTORY-CONTRACT] status={report['status']} years={historical_years} "
             f"score_games={len(frame)} oof_margin={report['aggregate']['margin']['n']} "
             f"oof_total={report['aggregate']['total']['n']} oof_h2h={report['aggregate']['h2h']['n']} "
             f"opening_required=FALSE production_authority=0")
    return report
