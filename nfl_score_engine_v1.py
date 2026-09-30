"""NFL Team Score Engine V1.6.

RESEARCH ONLY. No publication, no production authority, no BigQuery writes.
2026 is sealed. Development uses chronological outer folds 2021-2025.

Purpose
-------
Predict each team's points from two explicit perspectives:
  1) offense projection: what this offense should score against the opponent;
  2) defense allowance projection: what this defense should allow to the opponent.

The two side-level projections are reconciled into physical-game projected scores.
Those projected scores then imply game margin, total, winner, spread disagreement,
and totals disagreement. This keeps the score model as the common statistical core
instead of fitting unrelated betting classifiers.

Historical closing lines are retrospective diagnostics only. They are not model
features and do not establish executable ROI/CLV.
"""
from __future__ import annotations

import json
import math
from typing import Iterable

import numpy as np
import pandas as pd

from nfl_feature_audit_v1 import VIEW, _flatten_manifest
from nfl_challenger_v1 import (
    EXPERIMENT_SEASONS, VALIDATION_SEASONS, SEALED_YEAR,
    IDENTITY, build_readonly_query, _feature_matrix, _numeric,
)
from nfl_specialized_v1 import _continuous_metrics, _edge_diagnostics

SOURCE_TAG = "nfl-score-engine-v1.6-team-offense-defense-20260930"
MIN_GAMES_PER_COMPLETE_SEASON = 200

# Human-reviewable pregame context shared by the two component models.
_CONTEXT = (
    "Week_Number", "Is_Home", "Is_Away", "Is_Neutral", "Is_Regular_Season",
    "Is_Postseason", "Is_Division_Game", "Is_Conference_Game",
    "Is_Interconference_Game", "Is_Thursday_Game", "Is_Monday_Game",
    "Is_Sunday_Game", "Is_Saturday_Game", "Is_Night_Game", "Is_PrimeTime",
    "Days_Since_Last_Game_System", "Is_Short_Rest", "Is_Standard_Rest",
    "Is_Long_Rest", "Is_Bye_Like_Rest", "Is_Thursday_Short_Week",
    "Opp_Days_Since_Last_Game_System", "Opp_Is_Short_Rest", "Opp_Is_Long_Rest",
    "Rest_Differential_Days",
)

# Team offense + opponent defensive context. Every field is prior-only in the V1.3 manifest.
OFFENSE_FEATURES = tuple(dict.fromkeys(_CONTEXT + (
    "Team_Game_Number", "Prev_Points_For", "Prev_Off_Yards_Per_Play",
    "Prev_Turnover_Margin", "Prev_Third_Down_Pct", "Prev_Sack_Rate_Allowed",
    "Prev_Total_Yards", "Prev_Total_Plays", "Prev_Pass_Yards", "Prev_Rush_Yards",
    "Prev_Turnovers", "WinPct_Prior_System", "Avg_Points_For_Prior",
    "Avg_SU_Margin_Prior", "Avg_Off_YPP_Last3_Prior", "Avg_Rush_YPA_Last3_Prior",
    "Avg_Net_Pass_YPA_Last3_Prior", "Avg_Turnover_Margin_Last3_Prior",
    "Avg_Third_Down_Pct_Last3_Prior", "Avg_Sack_Rate_Allowed_Last3_Prior",
    "Avg_Time_Possession_Last3_Prior", "Avg_Points_For_Last5_Prior",
    "Avg_SU_Margin_Last5_Prior", "Avg_Off_YPP_Last5_Prior",
    "Prior_Season_WinPct", "Prior_Season_Avg_Margin", "Prior_Season_Avg_Points_For",
    "Opp_WinPct_Prior_System", "Opp_Avg_Points_Against_Last5_Prior",
    "Opp_Avg_Def_YPP_Last3_Prior", "Off_YPP_vs_Opp_Def_Last3_Diff",
    "Turnover_Margin_Last3_Diff", "Prior_Season_WinPct_Diff",
)))

# Defender state + incoming opponent offense. Target is Opponent_Score for the defending row.
DEFENSE_ALLOW_FEATURES = tuple(dict.fromkeys(_CONTEXT + (
    "Team_Game_Number", "Prev_Points_Against", "Prev_Def_Yards_Per_Play_Allowed",
    "Prev_Turnover_Margin", "Prev_Defensive_Sacks", "Prev_Total_Yards_Allowed",
    "WinPct_Prior_System", "Avg_Points_Against_Prior", "Avg_SU_Margin_Prior",
    "Avg_Def_YPP_Last3_Prior", "Avg_Turnover_Margin_Last3_Prior",
    "Avg_Defensive_Sacks_Last3_Prior", "Avg_Points_Against_Last5_Prior",
    "Avg_SU_Margin_Last5_Prior", "Avg_Def_YPP_Last5_Prior",
    "Prior_Season_WinPct", "Prior_Season_Avg_Margin", "Prior_Season_Avg_Points_Against",
    "Opp_WinPct_Prior_System", "Opp_Avg_Points_For_Last5_Prior",
    "Opp_Avg_Off_YPP_Last3_Prior", "Def_YPP_vs_Opp_Off_Last3_Diff",
    "Turnover_Margin_Last3_Diff", "Prior_Season_WinPct_Diff",
)))

MATCHUP_FEATURES = tuple(dict.fromkeys(OFFENSE_FEATURES + DEFENSE_ALLOW_FEATURES))
ALL_MANIFEST = tuple(_flatten_manifest())
assert set(OFFENSE_FEATURES).issubset(set(ALL_MANIFEST))
assert set(DEFENSE_ALLOW_FEATURES).issubset(set(ALL_MANIFEST))
assert set(MATCHUP_FEATURES).issubset(set(ALL_MANIFEST))

FAMILIES = (
    "LEAGUE_MEAN", "RECENT_AVG_BASELINE", "OFFENSE_RIDGE", "DEFENSE_RIDGE",
    "DUAL_COMPONENT", "MATCHUP_RIDGE", "MATCHUP_TREE", "SCORE_BLEND",
)


def _fit_ridge(train: pd.DataFrame, valid: pd.DataFrame, features: Iterable[str], target: str, alpha: float):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    cols = list(features)
    A, B = _feature_matrix(train, valid, cols)
    y = pd.to_numeric(train[target], errors="coerce").to_numpy(float)
    if not np.isfinite(y).all():
        raise RuntimeError(f"NONFINITE_TARGET {target}")
    sc = StandardScaler().fit(A)
    m = Ridge(alpha=float(alpha), random_state=17)
    m.fit(sc.transform(A), y)
    return np.asarray(m.predict(sc.transform(B)), dtype=float)


def _fit_tree(train: pd.DataFrame, valid: pd.DataFrame, target: str):
    from sklearn.ensemble import HistGradientBoostingRegressor
    A, B = _feature_matrix(train, valid, list(ALL_MANIFEST))
    y = pd.to_numeric(train[target], errors="coerce").to_numpy(float)
    if not np.isfinite(y).all():
        raise RuntimeError(f"NONFINITE_TARGET {target}")
    m = HistGradientBoostingRegressor(
        max_iter=140, max_leaf_nodes=7, min_samples_leaf=70,
        learning_rate=0.035, l2_regularization=24,
        early_stopping=False, random_state=17,
    )
    m.fit(A, y)
    return np.asarray(m.predict(B), dtype=float)


def _mask_unobserved_prior_season(df: pd.DataFrame) -> pd.DataFrame:
    """2017 is the first sourced NFL season; block prior-season fields there.

    This mirrors the V1.4 physical-game safeguard so the score engine cannot use
    an unobserved prior year merely because a source column happens to be populated.
    """
    d = df.copy()
    prior_cols = [c for c in ALL_MANIFEST if c.startswith("Prior_Season_") or c.startswith("Opp_Prior_Season_") or c == "Prior_Season_WinPct_Diff"]
    if prior_cols:
        d.loc[pd.to_numeric(d.Season, errors="coerce").eq(2017), prior_cols] = np.nan
    return d


def _validate_side_grain(df: pd.DataFrame):
    """Require exactly two reciprocal team-side rows per physical game."""
    if df.empty:
        raise RuntimeError("NO_SIDE_ROWS")
    problems = []
    for ident, p in df.groupby(list(IDENTITY), sort=False, dropna=False):
        if len(p) != 2:
            problems.append((ident, "BAD_SIDE_COUNT", len(p))); continue
        teams = [str(x or "").strip().lower() for x in p.Team_Norm]
        opps = [str(x or "").strip().lower() for x in p.Opponent_Norm]
        if len(set(teams)) != 2 or sorted(teams) != sorted(opps) or any(t == o for t, o in zip(teams, opps)):
            problems.append((ident, "BAD_TEAM_OPP_PAIR", None)); continue
        r0, r1 = p.iloc[0], p.iloc[1]
        if _numeric(r0.Team_Score) != _numeric(r1.Opponent_Score) or _numeric(r0.Opponent_Score) != _numeric(r1.Team_Score):
            problems.append((ident, "NONRECIPROCAL_SCORE", None))
    if problems:
        raise RuntimeError("[NFL-SCORE-V1-HOLD] SIDE_GRAIN " + json.dumps(problems[:12], default=str))


def _swap_within_game(values: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """For each team-side row, return the corresponding value from the opponent row."""
    values = np.asarray(values, dtype=float)
    if len(values) != len(frame):
        raise RuntimeError("SWAP_LENGTH_MISMATCH")
    out = np.full(len(frame), np.nan, dtype=float)
    pos = pd.Series(np.arange(len(frame)), index=frame.index)
    for _, p in frame.groupby(list(IDENTITY), sort=False, dropna=False):
        if len(p) != 2:
            raise RuntimeError("SWAP_BAD_SIDE_COUNT")
        i0, i1 = [int(pos.loc[ix]) for ix in p.index]
        out[i0] = values[i1]
        out[i1] = values[i0]
    return out


def _side_predictions(train: pd.DataFrame, valid: pd.DataFrame):
    """Return team-points projections for each side-row under each family."""
    league = float(pd.to_numeric(train.Team_Score, errors="coerce").mean())
    base = np.full(len(valid), league, dtype=float)
    recent_for = pd.to_numeric(valid.get("Avg_Points_For_Last5_Prior"), errors="coerce").to_numpy(float)
    recent_against = pd.to_numeric(valid.get("Opp_Avg_Points_Against_Last5_Prior"), errors="coerce").to_numpy(float)
    recent = np.nanmean(np.c_[recent_for, recent_against], axis=1)
    recent = np.where(np.isfinite(recent), recent, league)

    offense = _fit_ridge(train, valid, OFFENSE_FEATURES, "Team_Score", alpha=18.0)
    # Defender row predicts how many points its opponent will score.
    defense_allow_on_defender = _fit_ridge(train, valid, DEFENSE_ALLOW_FEATURES, "Opponent_Score", alpha=18.0)
    # To estimate this row's team points, use the opposing row's defensive allowance prediction.
    defense_for_team = _swap_within_game(defense_allow_on_defender, valid)
    dual = (offense + defense_for_team) / 2.0

    matchup_ridge = _fit_ridge(train, valid, MATCHUP_FEATURES, "Team_Score", alpha=28.0)
    matchup_tree = _fit_tree(train, valid, "Team_Score")
    score_blend = (dual + matchup_tree) / 2.0

    return {
        "LEAGUE_MEAN": base,
        "RECENT_AVG_BASELINE": recent,
        "OFFENSE_RIDGE": offense,
        "DEFENSE_RIDGE": defense_for_team,
        "DUAL_COMPONENT": dual,
        "MATCHUP_RIDGE": matchup_ridge,
        "MATCHUP_TREE": matchup_tree,
        "SCORE_BLEND": score_blend,
    }


def _physical_game_predictions(valid: pd.DataFrame, side_pred: np.ndarray) -> pd.DataFrame:
    """Reconcile side-level points into one oriented physical-game row."""
    d = valid.copy()
    d["_pred_points"] = np.asarray(side_pred, dtype=float)
    rows = []
    for ident, p in d.groupby(list(IDENTITY), sort=False, dropna=False):
        if len(p) != 2:
            raise RuntimeError("GAME_PRED_BAD_SIDE_COUNT")
        homes = pd.to_numeric(p.Is_Home, errors="coerce").fillna(-1)
        neutrals = pd.to_numeric(p.Is_Neutral, errors="coerce").fillna(-1)
        if int((homes == 1).sum()) == 1:
            chosen = p.loc[homes.eq(1)].iloc[0]
        elif neutrals.eq(1).all():
            chosen = p.sort_values("Team_Norm", kind="mergesort").iloc[0]
        else:
            raise RuntimeError("GAME_PRED_BAD_VENUE")
        other = p.loc[p.index != chosen.name].iloc[0]
        ts, os = _numeric(chosen.Team_Score), _numeric(chosen.Opponent_Score)
        ps, po = float(chosen._pred_points), float(other._pred_points)
        spread = _numeric(chosen.Spread_Value)
        total_line = _numeric(chosen.Current_Total)
        margin = ts - os
        actual_total = ts + os
        spread_status = "MISSING"
        if math.isfinite(spread):
            z = margin + spread
            spread_status = "WIN" if z > 1e-7 else "LOSS" if z < -1e-7 else "PUSH"
        total_status = "MISSING"
        if math.isfinite(total_line):
            z = actual_total - total_line
            total_status = "WIN" if z > 1e-7 else "LOSS" if z < -1e-7 else "PUSH"
        rows.append({
            "physical_game_id": "|".join(map(str, ident)),
            "Season": int(chosen.Season),
            "actual_team_points": ts,
            "actual_opp_points": os,
            "pred_team_points": ps,
            "pred_opp_points": po,
            "actual_margin": margin,
            "pred_margin": ps - po,
            "actual_total": actual_total,
            "pred_total": ps + po,
            "Spread_Value": spread,
            "Current_Total": total_line,
            "SPREADS_status": spread_status,
            "TOTALS_status": total_status,
        })
    out = pd.DataFrame(rows)
    if out.physical_game_id.duplicated().any():
        raise RuntimeError("DUPLICATE_GAME_PREDICTION")
    return out


def _winner_accuracy(actual_margin, pred_margin):
    a = np.asarray(actual_margin, float); p = np.asarray(pred_margin, float)
    mask = np.isfinite(a) & np.isfinite(p) & (a != 0) & (p != 0)
    if not mask.any(): return {"n": 0, "accuracy": None}
    return {"n": int(mask.sum()), "accuracy": round(float(np.mean(np.sign(a[mask]) == np.sign(p[mask]))), 6)}


def _family_metrics(valid: pd.DataFrame, pred: np.ndarray):
    actual_side = pd.to_numeric(valid.Team_Score, errors="coerce").to_numpy(float)
    gp = _physical_game_predictions(valid, pred)
    spread_edge = gp.pred_margin.to_numpy(float) + pd.to_numeric(gp.Spread_Value, errors="coerce").to_numpy(float)
    total_edge = gp.pred_total.to_numpy(float) - pd.to_numeric(gp.Current_Total, errors="coerce").to_numpy(float)
    return {
        "team_points": _continuous_metrics(actual_side, pred),
        "game_margin": _continuous_metrics(gp.actual_margin, gp.pred_margin),
        "game_total": _continuous_metrics(gp.actual_total, gp.pred_total),
        "winner": _winner_accuracy(gp.actual_margin, gp.pred_margin),
        "spread_disagreement": _edge_diagnostics(gp.SPREADS_status.to_numpy(object), spread_edge, thresholds=(0,1,2,3,4,5,6)),
        "total_disagreement": _edge_diagnostics(gp.TOTALS_status.to_numpy(object), total_edge, thresholds=(0,1,2,3,4,5,6)),
    }, gp


def _append_store(store, family, valid, pred, gp):
    s = store[family]
    s["actual_side"].extend(pd.to_numeric(valid.Team_Score, errors="coerce").tolist())
    s["pred_side"].extend(np.asarray(pred, float).tolist())
    for c in ("actual_margin","pred_margin","actual_total","pred_total","SPREADS_status","TOTALS_status","Spread_Value","Current_Total"):
        s[c].extend(gp[c].tolist())


def score_engine_tournament(side_rows: pd.DataFrame, log_func=print):
    side_rows = _mask_unobserved_prior_season(side_rows)
    if side_rows.Season.eq(SEALED_YEAR).any() or not side_rows.Season.isin(EXPERIMENT_SEASONS).all():
        raise RuntimeError("[NFL-SCORE-V1-HOLD] SEALED_YEAR_PRESENT")
    _validate_side_grain(side_rows)
    games_per_season = (side_rows.groupby("Season").size() // 2).to_dict()
    for y in EXPERIMENT_SEASONS:
        if int(games_per_season.get(y, 0)) < MIN_GAMES_PER_COMPLETE_SEASON:
            raise RuntimeError(f"[NFL-SCORE-V1-HOLD] INCOMPLETE_SEASON_{y} games={games_per_season.get(y,0)}")

    report = {
        "status": "RESEARCH_RESULTS_ONLY", "source_tag": SOURCE_TAG,
        "publication": False, "production_authority": 0, "year2026": "SEALED",
        "architecture": "TEAM_OFFENSE_PLUS_OPPONENT_DEFENSE_TO_PROJECTED_SCORE",
        "folds": [], "summary": {},
        "limitations": [
            "2021-2025 are development folds, not fresh confirmation",
            "historical closing lines are retrospective diagnostics only",
            "no historical ROI or CLV claim",
        ],
    }
    store = {f: {k: [] for k in (
        "actual_side","pred_side","actual_margin","pred_margin","actual_total","pred_total",
        "SPREADS_status","TOTALS_status","Spread_Value","Current_Total"
    )} for f in FAMILIES}

    for vy in VALIDATION_SEASONS:
        tr = side_rows.loc[side_rows.Season.lt(vy)].copy()
        va = side_rows.loc[side_rows.Season.eq(vy)].copy()
        _validate_side_grain(tr); _validate_side_grain(va)
        preds = _side_predictions(tr, va)
        row = {
            "validate_season": int(vy), "train_side_rows": int(len(tr)),
            "train_games": int(len(tr)//2), "validate_side_rows": int(len(va)),
            "validate_games": int(len(va)//2), "families": {},
        }
        for fam, pred in preds.items():
            metrics, gp = _family_metrics(va, pred)
            row["families"][fam] = metrics
            _append_store(store, fam, va, pred, gp)
        report["folds"].append(row)
        log_func("[NFL-SCORE-V1-FOLD] " + json.dumps(row, sort_keys=True, default=str))

    for fam, s in store.items():
        spread_edge = np.asarray(s["pred_margin"], float) + np.asarray(s["Spread_Value"], float)
        total_edge = np.asarray(s["pred_total"], float) - np.asarray(s["Current_Total"], float)
        summary = {
            "team_points": _continuous_metrics(s["actual_side"], s["pred_side"]),
            "game_margin": _continuous_metrics(s["actual_margin"], s["pred_margin"]),
            "game_total": _continuous_metrics(s["actual_total"], s["pred_total"]),
            "winner": _winner_accuracy(s["actual_margin"], s["pred_margin"]),
            "spread_disagreement": _edge_diagnostics(np.asarray(s["SPREADS_status"], object), spread_edge, thresholds=(0,1,2,3,4,5,6)),
            "total_disagreement": _edge_diagnostics(np.asarray(s["TOTALS_status"], object), total_edge, thresholds=(0,1,2,3,4,5,6)),
        }
        report["summary"][fam] = summary

    # Deterministic descriptive ranking only; does not grant production authority.
    # Primary score-engine criterion is game-level margin RMSE; total RMSE is secondary.
    report["descriptive_margin_rmse_order"] = sorted(
        FAMILIES, key=lambda f: report["summary"][f]["game_margin"].get("rmse", float("inf"))
    )
    report["descriptive_total_rmse_order"] = sorted(
        FAMILIES, key=lambda f: report["summary"][f]["game_total"].get("rmse", float("inf"))
    )
    log_func("[NFL-SCORE-V1-SUMMARY] " + json.dumps(report["summary"], sort_keys=True, default=str))
    log_func("[NFL-SCORE-V1-CONTRACT] status=RESEARCH_RESULTS_ONLY publication=FALSE production_authority=0 "
             "year2026=SEALED ncaaf=UNCHANGED legacy_nfl=UNCHANGED historical_roi_clv=NOT_VERIFIED")
    return report


def run_nfl_score_engine_v1(*, bq_client=None, audit_report=None, log_func=print):
    if not isinstance(audit_report, dict) or audit_report.get("status") != "READY_FOR_OFFLINE_CHALLENGER_SANDBOX":
        raise RuntimeError("[NFL-SCORE-V1-HOLD] V1_3_AUDIT_NOT_GREEN")
    from google.cloud import bigquery
    bq = bq_client or bigquery.Client(project="sharplogger")
    cols = {f.name for f in bq.get_table(VIEW).schema}
    query = build_readonly_query(cols)
    log_func(f"[NFL-SCORE-V1-PREFLIGHT] status=PASS tag={SOURCE_TAG} publication=FALSE year2026=SEALED "
             f"offense_features={len(OFFENSE_FEATURES)} defense_features={len(DEFENSE_ALLOW_FEATURES)}")
    df = bq.query(query, job_config=bigquery.QueryJobConfig(
        use_query_cache=True, maximum_bytes_billed=20*1024**3
    )).to_dataframe()
    _validate_side_grain(df)
    counts = {str(k): int(v//2) for k, v in df.groupby("Season").size().items()}
    log_func("[NFL-SCORE-V1-GRAIN] " + json.dumps({
        "source_side_rows": int(len(df)), "physical_games": int(len(df)//2),
        "games_by_season": counts,
    }, sort_keys=True))
    report = score_engine_tournament(df, log_func=log_func)

    # Apples-to-apples benchmark against the already-defined V1.5 direct margin/total
    # models on the same 2017-2025 rows. Suppress its verbose fold logs here.
    from nfl_challenger_v1 import physical_games
    from nfl_specialized_v1 import specialized_tournament
    v15 = specialized_tournament(physical_games(df), log_func=lambda _msg: None)
    bench = {
        "spreads_direct_score_models": {k: v["score_model"] for k, v in v15["spreads"]["summary"].items()},
        "totals_direct_score_models": {k: v["score_model"] for k, v in v15["totals"]["summary"].items()},
        "note": "V1.5 direct margin/total models rerun on identical development rows; descriptive comparison only.",
    }
    report["v15_direct_benchmark"] = bench
    log_func("[NFL-SCORE-V1-V15-BENCHMARK] " + json.dumps(bench, sort_keys=True, default=str))
    return report
