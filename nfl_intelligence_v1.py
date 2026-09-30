"""NFL Intelligence Research V1.8.

RESEARCH ONLY. Zero production authority. No BigQuery writes. 2026 is sealed.

This layer sits above the already-audited NFL statistical brains and asks a more
useful question than "can another generic classifier beat the market?":

    When independently-built statistical models agree or disagree with the
    market, in which football situations has that disagreement historically
    carried information?

Lanes
-----
1) CORE agreement: V1.5 direct margin/total + V1.6 score engine + H2H blend.
2) BIG AL: exact publicly documented NFL systems BA-NFL1..3 where this dataset
   can reproduce the trigger. BA-NFL4..5 are preseason and are explicitly not
   tested because the audited NFL history excludes preseason.
3) PATHI: football price/key-number engineering translations already used in
   the NCAAF research stack. These are NOT claimed to be verified Pathi NFL
   mechanical systems.
4) RESIDUAL / ARBITRATION: learn when CORE is more trustworthy than the market,
   quantify prediction uncertainty, and mine continuous market/model errors.
5) MINER: bounded, family-aware discovery over pregame context + OOF model
   states. Discovery=2021-2023, shadow=2024, confirmation=2025. Candidate
   direction is frozen from discovery before later seasons are scored.
6) MECHANISM: reconcile conflicts inside each source family first, then summarize
   agreement/conflict across independent families. No source can self-promote.

Historical closing/opening lines are retrospective research context only. They
are not asserted to be timestamped executable quotes and no ROI/CLV claim is
made from them.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from itertools import combinations
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from nfl_feature_audit_v1 import VIEW, _flatten_manifest
from nfl_challenger_v1 import (
    REQUIRED_COLUMNS, EXPERIMENT_SEASONS, SEALED_YEAR, IDENTITY,
    physical_games, _numeric,
)
from nfl_specialized_v1 import _fit_regression, _h2h_predictions
from nfl_score_engine_v1 import (
    _mask_unobserved_prior_season, _validate_side_grain, _side_predictions,
    _physical_game_predictions, _swap_within_game,
)

SOURCE_TAG = "nfl-intelligence-v1.8-residual-arbitration-reconciliation-20260930"
DISCOVERY_SEASONS = (2021, 2022, 2023)
SHADOW_SEASON = 2024
CONFIRM_SEASON = 2025
KEY_NUMBERS = (3.0, 6.0, 7.0, 10.0, 14.0)
EXTRA_MARKET_COLUMNS = ("Opening_Spread", "Opening_Total")

# Publicly documented exact Big Al NFL systems recovered in the project research.
# Published records are metadata only; they are never used as predictors/weights.
BIGAL_REGISTRY = {
    "BA-NFL1": {
        "name": "Week 1 Fade Prior-Year Playoff Team",
        "source_class": "HISTORICAL_DATABASE_SYSTEM",
        "claimed_record": "prior-playoff side 131-157-6 ATS; home play-on tightener 90-68-5 ATS",
        "implementation": "Week 1: play non-playoff team ATS vs prior-season playoff opponent",
    },
    "BA-NFL1-HOME": {
        "name": "Week 1 Fade Prior-Year Playoff Team - Home Tightener",
        "source_class": "HISTORICAL_DATABASE_SYSTEM_TIGHTENER",
        "claimed_record": "home play-on tightener 90-68-5 ATS",
        "implementation": "BA-NFL1 plus play-on side is home",
    },
    "BA-NFL2": {
        "name": "Late-Season Home Dog off Two Losses",
        "source_class": "HISTORICAL_DATABASE_SYSTEM",
        "claimed_record": "36-17-2 ATS",
        "implementation": "final two regular weeks; home dog; off two SU losses; opponent win pct <= .500",
    },
    "BA-NFL2-ATS": {
        "name": "Late-Season Home Dog off Two Losses - Opponent ATS-Loss Tightener",
        "source_class": "HISTORICAL_DATABASE_SYSTEM_TIGHTENER",
        "claimed_record": "15-2 ATS at publication",
        "implementation": "BA-NFL2 plus opponent off ATS loss",
    },
    "BA-NFL3": {
        "name": "Playoff High-Scoring Road/Neutral Fade",
        "source_class": "HISTORICAL_DATABASE_SYSTEM",
        "claimed_record": "qualifying high-scoring side 30-54 ATS",
        "implementation": "playoffs; fade non-home previous winner that scored 35+ when opponent prior score <35",
    },
    "BA-NFL4": {
        "name": "Preseason Bet Against Major Spread Move",
        "source_class": "HISTORICAL_DATABASE_SYSTEM",
        "status": "UNAVAILABLE_NO_PRESEASON_HISTORY",
    },
    "BA-NFL5": {
        "name": "Preseason Low-Offense OVER",
        "source_class": "HISTORICAL_DATABASE_SYSTEM",
        "status": "UNAVAILABLE_NO_PRESEASON_HISTORY",
    },
}

# These names/rules are copied from the existing NCAAF Pathi engineering layer.
# They are explicit engineering translations, not attributed as verified Pathi NFL formulas.
PATHI_ENGINEERING_REGISTRY = {
    "Pathi_FB_Dog_Hook_Above_3": "dog >3 and <4",
    "Pathi_FB_Dog_Hook_Above_7": "dog >7 and <8",
    "Pathi_FB_Dog_Hook_Above_10": "dog >10 and <11",
    "Pathi_FB_Favorite_Below_Key_3": "favorite abs(spread) >2 and <3",
    "Pathi_FB_Favorite_Below_Key_7": "favorite abs(spread) >6 and <7",
    "Pathi_FB_Favorite_Below_Key_10": "favorite abs(spread) >9 and <10",
    "Pathi_FB_Dog_Below_Key_3": "dog >2 and <3",
    "Pathi_FB_Dog_Below_Key_7": "dog >6 and <7",
    "Pathi_FB_Favorite_Laying_Hook_3": "favorite abs(spread) >3 and <4",
    "Pathi_FB_Favorite_Laying_Hook_7": "favorite abs(spread) >7 and <8",
    "Pathi_FB_Dog_TotalSpread_Gap_LE10": "current underdog and total - abs(spread) <=10",
}


def build_intelligence_query(view_columns: Iterable[str] | None = None) -> str:
    cols = tuple(dict.fromkeys((*REQUIRED_COLUMNS, *EXTRA_MARKET_COLUMNS)))
    if view_columns is not None:
        missing = set(cols) - set(view_columns)
        if missing:
            raise RuntimeError("[NFL-INTEL-V1-HOLD] SOURCE_COLUMNS_MISSING " + str(sorted(missing)))
    quoted = ", ".join(f"`{c}`" for c in cols)
    return (
        f"SELECT {quoted} FROM `{VIEW}` "
        "WHERE Season BETWEEN 2017 AND 2025 "
        "AND Season_Stage IN ('REGULAR','POSTSEASON') "
        "AND Historical_Core_Eligible = 1 "
        "ORDER BY Season, Game_Date, Source_Name, Source_Game_ID, Team_Norm"
    )




def _float_array(values) -> np.ndarray:
    """Convert pandas nullable numeric data to ordinary float64 with NaN.

    BigQuery integer columns can arrive as pandas nullable Int64/Float64 arrays.
    Their ``to_numpy(float)`` path raises when ``pd.NA`` is present unless an
    explicit NA representation is provided.  Normalize once here so every
    research path has identical, predictable missing-value semantics.
    """
    x = pd.to_numeric(values, errors="coerce")
    if isinstance(x, pd.Series):
        return x.astype("float64").to_numpy(dtype=np.float64, copy=False)
    return np.asarray(x, dtype=np.float64)

def _status_from_margin(v: float) -> str:
    if not math.isfinite(v):
        return "MISSING"
    if v > 1e-9:
        return "WIN"
    if v < -1e-9:
        return "LOSS"
    return "PUSH"


def _invert_status(v: str) -> str:
    x = str(v).upper()
    return "LOSS" if x == "WIN" else "WIN" if x == "LOSS" else x


def _physical_id_frame(d: pd.DataFrame) -> pd.Series:
    return d[list(IDENTITY)].astype(str).agg("|".join, axis=1)


def _wilson(w: int, n: int, z: float = 1.959963984540054) -> Tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = w / n
    den = 1.0 + z*z/n
    ctr = (p + z*z/(2*n))/den
    half = z*math.sqrt((p*(1-p) + z*z/(4*n))/n)/den
    return max(0.0, ctr-half), min(1.0, ctr+half)


def _system_metrics(rows: pd.DataFrame, mask: np.ndarray, status_col: str = "ATS_status_current") -> dict:
    m = np.asarray(mask, bool)
    if len(m) != len(rows):
        raise RuntimeError("SYSTEM_MASK_LENGTH_MISMATCH")
    s = rows.loc[m, status_col].astype(str).str.upper()
    wins = int((s == "WIN").sum()); losses = int((s == "LOSS").sum()); pushes = int((s == "PUSH").sum())
    n = wins + losses
    lo, hi = _wilson(wins, n)
    by = {}
    for sy, p in rows.loc[m].groupby("Season"):
        q = p[status_col].astype(str).str.upper()
        w = int((q == "WIN").sum()); l = int((q == "LOSS").sum()); pu = int((q == "PUSH").sum())
        by[str(int(sy))] = {"n": w+l, "wins": w, "losses": l, "pushes": pu,
                            "hit_rate": round(w/(w+l), 6) if (w+l) else None}
    return {
        "trigger_rows": int(m.sum()), "n": n, "wins": wins, "losses": losses, "pushes": pushes,
        "hit_rate": round(wins/n, 6) if n else None,
        "wilson95": [round(lo, 6), round(hi, 6)] if lo is not None else None,
        "by_season": by,
    }


def _prepare_side_state(side_rows: pd.DataFrame) -> pd.DataFrame:
    d = side_rows.copy()
    _validate_side_grain(d)
    d["Season"] = pd.to_numeric(d["Season"], errors="coerce").astype(int)
    if d.Season.eq(SEALED_YEAR).any() or not d.Season.isin(EXPERIMENT_SEASONS).all():
        raise RuntimeError("[NFL-INTEL-V1-HOLD] SEALED_YEAR_PRESENT")
    d["physical_game_id"] = _physical_id_frame(d)
    d["actual_margin"] = pd.to_numeric(d.Team_Score, errors="coerce") - pd.to_numeric(d.Opponent_Score, errors="coerce")
    sp = pd.to_numeric(d.Spread_Value, errors="coerce")
    d["ATS_status_current"] = [
        _status_from_margin(float(m+s)) if np.isfinite(m) and np.isfinite(s) else "MISSING"
        for m, s in zip(pd.to_numeric(d.actual_margin, errors="coerce"), sp)
    ]
    d["SU_win_current"] = pd.to_numeric(d.actual_margin, errors="coerce").gt(0).astype(float)
    d["SU_loss_current"] = pd.to_numeric(d.actual_margin, errors="coerce").lt(0).astype(float)
    d["SU_tie_current"] = pd.to_numeric(d.actual_margin, errors="coerce").eq(0).astype(float)
    d["dog_current"] = sp.gt(0).astype(float)

    # Independent sequential history used for systems that require two prior results
    # or prior role-rate. Sorting uses only event identity/time, never outcomes.
    d = d.sort_values(["Season", "Team_Norm", "Game_Date", "Source_Name", "Source_Game_ID"], kind="mergesort").copy()
    grp = d.groupby(["Season", "Team_Norm"], sort=False, dropna=False)
    d["calc_prev1_su_loss"] = grp["SU_loss_current"].shift(1)
    d["calc_prev2_su_loss"] = grp["SU_loss_current"].shift(2)
    d["calc_prev1_su_win"] = grp["SU_win_current"].shift(1)
    d["calc_prev1_points"] = grp["Team_Score"].shift(1)
    d["calc_prev1_ats_loss"] = grp["ATS_status_current"].shift(1).eq("LOSS").astype(float)
    games_before = grp.cumcount().astype(float)
    wins_before = grp["SU_win_current"].cumsum() - d["SU_win_current"]
    ties_before = grp["SU_tie_current"].cumsum() - d["SU_tie_current"]
    d["calc_win_pct_prior"] = np.where(games_before.gt(0), (wins_before + .5*ties_before)/games_before, np.nan)
    dogs_before = grp["dog_current"].cumsum() - d["dog_current"]
    d["calc_dog_rate_prior"] = np.where(games_before.gt(0), dogs_before/games_before, np.nan)

    # Pair opponent's independently-derived state at the same physical game.
    for c in ("calc_prev1_su_loss", "calc_prev2_su_loss", "calc_prev1_su_win", "calc_prev1_points",
              "calc_prev1_ats_loss", "calc_win_pct_prior", "calc_dog_rate_prior"):
        d["Opp_" + c] = _swap_within_game(_float_array(d[c]), d)

    # Prior-season playoff membership is reconstructed from observed postseason participation;
    # the source's excluded prior-playoff flags are not used.
    postseason = {
        int(sy): set(p.Team_Norm.astype(str).str.lower())
        for sy, p in d.loc[d.Season_Stage.eq("POSTSEASON")].groupby("Season")
    }
    d["calc_prior_season_playoff"] = [
        float(str(t).lower() in postseason.get(int(s)-1, set())) if int(s)-1 in postseason else np.nan
        for s, t in zip(d.Season, d.Team_Norm)
    ]
    d["Opp_calc_prior_season_playoff"] = _swap_within_game(
        _float_array(d.calc_prior_season_playoff), d
    )

    reg = d.loc[d.Season_Stage.eq("REGULAR")].copy()
    max_week = pd.to_numeric(reg.Week_Number, errors="coerce").groupby(reg.Season).max().to_dict()
    d["calc_regular_final_week"] = d.Season.map(max_week)
    return d.sort_values(["Season", "Game_Date", "Source_Name", "Source_Game_ID", "Team_Norm"], kind="mergesort").reset_index(drop=True)


def _generate_oof_core(side_rows: pd.DataFrame) -> pd.DataFrame:
    side_rows = _mask_unobserved_prior_season(side_rows.copy())
    _validate_side_grain(side_rows)
    games = physical_games(side_rows)
    games["actual_margin"] = pd.to_numeric(games.Team_Score, errors="coerce") - pd.to_numeric(games.Opponent_Score, errors="coerce")
    games["actual_total"] = pd.to_numeric(games.Team_Score, errors="coerce") + pd.to_numeric(games.Opponent_Score, errors="coerce")
    pieces = []
    for vy in range(2021, 2026):
        tr_s = side_rows.loc[side_rows.Season.lt(vy)].copy(); va_s = side_rows.loc[side_rows.Season.eq(vy)].copy()
        tr_g = games.loc[games.Season.lt(vy)].copy(); va_g = games.loc[games.Season.eq(vy)].copy()
        if len(tr_g) < 1000 or len(va_g) < 250:
            raise RuntimeError(f"[NFL-INTEL-V1-HOLD] INSUFFICIENT_FOLD_{vy}")

        # Frozen statistical brains from V1.5/V1.6.
        direct_margin = _fit_regression("COMPACT_RIDGE", "SPREADS", tr_g, va_g, "actual_margin")
        direct_total = _fit_regression("BLEND50", "TOTALS", tr_g, va_g, "actual_total")
        score_side = _side_predictions(tr_s, va_s)["SCORE_BLEND"]
        score_game = _physical_game_predictions(va_s, score_side)[["physical_game_id", "pred_margin", "pred_total"]]

        htr = tr_g.loc[tr_g.H2H_label.notna()].copy(); hva = va_g.loc[va_g.H2H_label.notna()].copy()
        hpred, _ = _h2h_predictions(htr, hva)
        hmap = dict(zip(hva.physical_game_id, hpred["H2H_BLEND50"]))

        part = va_g.copy()
        part["direct_margin_pred"] = np.asarray(direct_margin, float)
        part["direct_total_pred"] = np.asarray(direct_total, float)
        part = part.merge(score_game, on="physical_game_id", how="left", validate="one_to_one")
        part = part.rename(columns={"pred_margin": "score_margin_pred", "pred_total": "score_total_pred"})
        part["h2h_prob"] = part.physical_game_id.map(hmap)
        pieces.append(part)

    out = pd.concat(pieces, ignore_index=True)
    if out.physical_game_id.duplicated().any():
        raise RuntimeError("DUPLICATE_OOF_CORE_GAME")
    sp = pd.to_numeric(out.Spread_Value, errors="coerce")
    tt = pd.to_numeric(out.Current_Total, errors="coerce")
    out["direct_spread_edge"] = pd.to_numeric(out.direct_margin_pred, errors="coerce") + sp
    out["score_spread_edge"] = pd.to_numeric(out.score_margin_pred, errors="coerce") + sp
    out["spread_consensus_edge"] = (out.direct_spread_edge + out.score_spread_edge) / 2.0
    out["spread_model_gap"] = (pd.to_numeric(out.direct_margin_pred, errors="coerce") - pd.to_numeric(out.score_margin_pred, errors="coerce")).abs()
    out["spread_models_agree"] = (
        np.sign(_float_array(out.direct_spread_edge)) == np.sign(_float_array(out.score_spread_edge))
    ) & np.isfinite(out.direct_spread_edge) & np.isfinite(out.score_spread_edge) & out.direct_spread_edge.ne(0) & out.score_spread_edge.ne(0)
    out["direct_total_edge"] = pd.to_numeric(out.direct_total_pred, errors="coerce") - tt
    out["score_total_edge"] = pd.to_numeric(out.score_total_pred, errors="coerce") - tt
    out["total_consensus_edge"] = (out.direct_total_edge + out.score_total_edge) / 2.0
    out["total_model_gap"] = (pd.to_numeric(out.direct_total_pred, errors="coerce") - pd.to_numeric(out.score_total_pred, errors="coerce")).abs()
    out["total_models_agree"] = (
        np.sign(_float_array(out.direct_total_edge)) == np.sign(_float_array(out.score_total_edge))
    ) & np.isfinite(out.direct_total_edge) & np.isfinite(out.score_total_edge) & out.direct_total_edge.ne(0) & out.score_total_edge.ne(0)
    out["h2h_market_delta"] = pd.to_numeric(out.h2h_prob, errors="coerce") - pd.to_numeric(out.H2H_close_novig_reference, errors="coerce")
    out["core_margin_pred"] = (pd.to_numeric(out.direct_margin_pred, errors="coerce") + pd.to_numeric(out.score_margin_pred, errors="coerce")) / 2.0
    out["core_total_pred"] = (pd.to_numeric(out.direct_total_pred, errors="coerce") + pd.to_numeric(out.score_total_pred, errors="coerce")) / 2.0
    out["spread_market_error"] = pd.to_numeric(out.actual_margin, errors="coerce") + sp
    out["spread_core_error"] = pd.to_numeric(out.actual_margin, errors="coerce") - pd.to_numeric(out.core_margin_pred, errors="coerce")
    out["total_market_error"] = pd.to_numeric(out.actual_total, errors="coerce") - tt
    out["total_core_error"] = pd.to_numeric(out.actual_total, errors="coerce") - pd.to_numeric(out.core_total_pred, errors="coerce")
    out["h2h_market_residual"] = pd.to_numeric(out.H2H_label, errors="coerce") - pd.to_numeric(out.H2H_close_novig_reference, errors="coerce")
    return out


def _bigal_systems(state: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    d = state.copy()
    week = pd.to_numeric(d.Week_Number, errors="coerce")
    spread = pd.to_numeric(d.Spread_Value, errors="coerce")
    opp_wp = pd.to_numeric(d.Opp_calc_win_pct_prior, errors="coerce")
    final_week = pd.to_numeric(d.calc_regular_final_week, errors="coerce")
    masks = {}
    # BA-NFL1 is expressed from the play-on/non-playoff side.
    masks["BA-NFL1"] = (
        d.Season_Stage.eq("REGULAR") & week.eq(1) &
        pd.to_numeric(d.calc_prior_season_playoff, errors="coerce").eq(0) &
        pd.to_numeric(d.Opp_calc_prior_season_playoff, errors="coerce").eq(1)
    ).to_numpy(bool)
    masks["BA-NFL1-HOME"] = masks["BA-NFL1"] & pd.to_numeric(d.Is_Home, errors="coerce").eq(1).to_numpy(bool)
    late = d.Season_Stage.eq("REGULAR") & week.notna() & final_week.notna() & week.ge(final_week - 1)
    base2 = (
        late & pd.to_numeric(d.Is_Home, errors="coerce").eq(1) & spread.gt(0) &
        pd.to_numeric(d.calc_prev1_su_loss, errors="coerce").eq(1) &
        pd.to_numeric(d.calc_prev2_su_loss, errors="coerce").eq(1) & opp_wp.le(.5)
    )
    masks["BA-NFL2"] = base2.to_numpy(bool)
    masks["BA-NFL2-ATS"] = (base2 & pd.to_numeric(d.Opp_calc_prev1_ats_loss, errors="coerce").eq(1)).to_numpy(bool)
    # Express BA-NFL3 from the play-on side: its opponent is the qualifying high-scoring non-home side.
    masks["BA-NFL3"] = (
        d.Season_Stage.eq("POSTSEASON") &
        pd.to_numeric(d.Opp_calc_prev1_su_win, errors="coerce").eq(1) &
        pd.to_numeric(d.Opp_calc_prev1_points, errors="coerce").ge(35) &
        pd.to_numeric(d.calc_prev1_points, errors="coerce").lt(35) &
        pd.to_numeric(d.Is_Home, errors="coerce").eq(1)  # if neutral, handled below
    ).to_numpy(bool)
    # Neutral playoff games: both sides are neutral, so allow the opponent of the qualifier.
    neutral3 = (
        d.Season_Stage.eq("POSTSEASON") & pd.to_numeric(d.Is_Neutral, errors="coerce").eq(1) &
        pd.to_numeric(d.Opp_calc_prev1_su_win, errors="coerce").eq(1) &
        pd.to_numeric(d.Opp_calc_prev1_points, errors="coerce").ge(35) &
        pd.to_numeric(d.calc_prev1_points, errors="coerce").lt(35)
    ).to_numpy(bool)
    masks["BA-NFL3"] |= neutral3

    report = {}
    plays = []
    for sid in ("BA-NFL1", "BA-NFL1-HOME", "BA-NFL2", "BA-NFL2-ATS", "BA-NFL3"):
        met = _system_metrics(d, masks[sid])
        report[sid] = {**BIGAL_REGISTRY[sid], **met, "production_authority": 0, "role": "DESCRIPTIVE_REPLICATION"}
        for _, r in d.loc[masks[sid]].iterrows():
            plays.append({"source": "BIGAL", "system_id": sid, "physical_game_id": r.physical_game_id,
                          "bet_team": str(r.Team_Norm), "status": str(r.ATS_status_current).upper()})
    for sid in ("BA-NFL4", "BA-NFL5"):
        report[sid] = {**BIGAL_REGISTRY[sid], "production_authority": 0, "n": 0}
    return report, pd.DataFrame(plays)


def _pathi_engineering(state: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    d = state.copy(); sp = pd.to_numeric(d.Spread_Value, errors="coerce"); ab = sp.abs(); total = pd.to_numeric(d.Current_Total, errors="coerce")
    masks = {
        "Pathi_FB_Dog_Hook_Above_3": sp.gt(3)&sp.lt(4),
        "Pathi_FB_Dog_Hook_Above_7": sp.gt(7)&sp.lt(8),
        "Pathi_FB_Dog_Hook_Above_10": sp.gt(10)&sp.lt(11),
        "Pathi_FB_Favorite_Below_Key_3": sp.lt(0)&ab.gt(2)&ab.lt(3),
        "Pathi_FB_Favorite_Below_Key_7": sp.lt(0)&ab.gt(6)&ab.lt(7),
        "Pathi_FB_Favorite_Below_Key_10": sp.lt(0)&ab.gt(9)&ab.lt(10),
        "Pathi_FB_Dog_Below_Key_3": sp.gt(2)&sp.lt(3),
        "Pathi_FB_Dog_Below_Key_7": sp.gt(6)&sp.lt(7),
        "Pathi_FB_Favorite_Laying_Hook_3": sp.lt(0)&ab.gt(3)&ab.lt(4),
        "Pathi_FB_Favorite_Laying_Hook_7": sp.lt(0)&ab.gt(7)&ab.lt(8),
        "Pathi_FB_Dog_TotalSpread_Gap_LE10": sp.gt(0)&total.notna()&(total-ab).le(10),
    }
    report={}; plays=[]
    for sid, m in masks.items():
        mm=pd.Series(m,index=d.index).fillna(False).to_numpy(bool)
        met=_system_metrics(d,mm)
        report[sid]={"name":sid,"rule":PATHI_ENGINEERING_REGISTRY[sid],"source_class":"PATHI_ENGINEERING_TRANSLATION",
                     **met,"production_authority":0,"role":"RESEARCH_ONLY_NOT_VERIFIED_PATHI_NFL_SYSTEM"}
        for _,r in d.loc[mm].iterrows():
            plays.append({"source":"PATHI","system_id":sid,"physical_game_id":r.physical_game_id,
                          "bet_team":str(r.Team_Norm),"status":str(r.ATS_status_current).upper()})
    return report,pd.DataFrame(plays)


def _crossed_key(open_spread: pd.Series, close_spread: pd.Series) -> pd.Series:
    o=pd.to_numeric(open_spread,errors="coerce"); c=pd.to_numeric(close_spread,errors="coerce")
    out=pd.Series(False,index=o.index)
    for k in KEY_NUMBERS:
        for sk in (k,-k):
            out |= (o.sub(sk)*c.sub(sk)).lt(0)
    return out & o.notna() & c.notna()


def _miner_atoms(g: pd.DataFrame) -> List[dict]:
    n=lambda c: pd.to_numeric(g.get(c,pd.Series(np.nan,index=g.index)),errors="coerce")
    atoms=[]
    def add(name,family,mask,desc=None):
        mm=pd.Series(mask,index=g.index).fillna(False).astype(bool).to_numpy()
        disc=np.isin(_float_array(g.Season),DISCOVERY_SEASONS)
        cnt=int((mm&disc).sum())
        if 30<=cnt<int(disc.sum()):
            atoms.append({"name":name,"family":family,"mask":mm,"description":desc or name})
    sp=n("Spread_Value"); tot=n("Current_Total"); week=n("Week_Number")
    # market role/price
    add("HOME_DOG","MARKET_ROLE",sp.gt(0)); add("HOME_FAVORITE","MARKET_ROLE",sp.lt(0))
    for lo,hi in ((0,3),(3,7),(7,10),(10,14),(14,99)):
        add(f"DOG_{lo}_{hi}","MARKET_PRICE",sp.gt(lo)&sp.le(hi))
        add(f"FAV_{lo}_{hi}","MARKET_PRICE",(-sp).gt(lo)&(-sp).le(hi))
    for lo,hi in ((0,42),(42,46),(46,50),(50,99)):
        add(f"TOTAL_{lo}_{hi}","TOTAL_REGIME",tot.ge(lo)&tot.lt(hi))
    # schedule/timing
    add("DIVISION_GAME","SCHEDULE",n("Is_Division_Game").eq(1)); add("NON_DIVISION_GAME","SCHEDULE",n("Is_Division_Game").eq(0))
    add("EARLY_WK1_4","SEASON_TIMING",week.between(1,4)); add("MID_WK5_10","SEASON_TIMING",week.between(5,10)); add("LATE_WK11_PLUS","SEASON_TIMING",week.ge(11))
    add("PRIMETIME","SCHEDULE",n("Is_PrimeTime").eq(1)); add("POSTSEASON","SCHEDULE",g.Season_Stage.astype(str).eq("POSTSEASON"))
    rest=n("Rest_Differential_Days"); add("REST_ADV_2_PLUS","REST",rest.ge(2)); add("REST_DISADV_2_PLUS","REST",rest.le(-2))
    # prior form / history
    add("OFF_SU_WIN","PRIOR_RESULT",n("Prev_SU_Margin").gt(0)); add("OFF_SU_LOSS","PRIOR_RESULT",n("Prev_SU_Margin").lt(0))
    add("OFF_BLOWOUT_WIN_14","PRIOR_RESULT",n("Prev_SU_Margin").ge(14)); add("OFF_BLOWOUT_LOSS_14","PRIOR_RESULT",n("Prev_SU_Margin").le(-14))
    add("OFF_ATS_WIN","ATS_FORM",n("Prev_ATS_Win").eq(1)); add("OFF_ATS_LOSS","ATS_FORM",n("Prev_ATS_Loss").eq(1))
    add("REVENGE","MATCHUP_HISTORY",n("Revenge_Flag_Current").eq(1)); add("RECENT_H2H_730D","MATCHUP_HISTORY",n("Days_Since_Last_Matchup_System").le(730)&n("Days_Since_Last_Matchup_System").notna())
    # independent model state
    de=n("direct_spread_edge"); se=n("score_spread_edge"); ce=n("spread_consensus_edge"); gap=n("spread_model_gap")
    agree=g.get("spread_models_agree",pd.Series(False,index=g.index)).astype(bool)
    add("DIRECT_EDGE_HOME_2","MODEL_STATE",de.ge(2)); add("DIRECT_EDGE_AWAY_2","MODEL_STATE",de.le(-2))
    add("SCORE_EDGE_HOME_2","MODEL_STATE",se.ge(2)); add("SCORE_EDGE_AWAY_2","MODEL_STATE",se.le(-2))
    add("BOTH_EDGE_HOME_2","MODEL_STATE",agree&de.ge(2)&se.ge(2)); add("BOTH_EDGE_AWAY_2","MODEL_STATE",agree&de.le(-2)&se.le(-2))
    add("CONSENSUS_EDGE_ABS_4","MODEL_STATE",agree&ce.abs().ge(4)); add("CONSENSUS_EDGE_ABS_6","MODEL_STATE",agree&ce.abs().ge(6))
    add("CONSENSUS_EDGE_HOME_4","MODEL_STATE",agree&ce.ge(4)); add("CONSENSUS_EDGE_AWAY_4","MODEL_STATE",agree&ce.le(-4))
    add("MARGIN_MODELS_WITHIN_1","MODEL_AGREEMENT",gap.le(1)); add("MARGIN_MODELS_WITHIN_2","MODEL_AGREEMENT",gap.le(2))
    dt=n("direct_total_edge"); st=n("score_total_edge"); tc=n("total_consensus_edge"); tg=n("total_model_gap")
    tagree=g.get("total_models_agree",pd.Series(False,index=g.index)).astype(bool)
    add("BOTH_TOTAL_OVER_3","MODEL_STATE",tagree&dt.ge(3)&st.ge(3)); add("BOTH_TOTAL_UNDER_3","MODEL_STATE",tagree&dt.le(-3)&st.le(-3))
    add("TOTAL_CONSENSUS_ABS_4","MODEL_STATE",tagree&tc.abs().ge(4)); add("TOTAL_CONSENSUS_OVER_4","MODEL_STATE",tagree&tc.ge(4)); add("TOTAL_CONSENSUS_UNDER_4","MODEL_STATE",tagree&tc.le(-4))
    add("TOTAL_MODELS_WITHIN_2","MODEL_AGREEMENT",tg.le(2))
    h=n("h2h_market_delta"); add("H2H_MODEL_OVER_MARKET_5P","MODEL_STATE",h.ge(.05)); add("H2H_MODEL_UNDER_MARKET_5P","MODEL_STATE",h.le(-.05))
    # Pathi-style market structure contexts, not credited as verified Pathi NFL systems.
    op=n("Opening_Spread"); crossed=_crossed_key(op,sp); toward=sp.lt(op); away=sp.gt(op)
    add("KEY_CROSSED_TOWARD_SELECTED","KEY_NUMBER",crossed&toward); add("KEY_CROSSED_AWAY_SELECTED","KEY_NUMBER",crossed&away)
    add("DOG_TOTAL_SPREAD_GAP_LE10","CROSS_MARKET",sp.gt(0)&(tot-sp.abs()).le(10))
    role=n("calc_dog_rate_prior")
    add("USUALLY_DOG_NOW_FAVORITE","ROLE_REVERSAL",role.ge(.70)&sp.lt(0)); add("USUALLY_FAVORITE_NOW_DOG","ROLE_REVERSAL",role.le(.30)&sp.gt(0))
    return atoms


def _target(g: pd.DataFrame, market: str):
    """Return outcome, validity and market expectation for miner evaluation.

    For H2H the null is the no-vig closing market probability, NOT 50%.  That
    prevents ordinary favorites from masquerading as profitable systems merely
    because they win more than half their games.
    """
    am=_float_array(g.actual_margin); at=_float_array(g.actual_total)
    sp=_float_array(g.Spread_Value); tt=_float_array(g.Current_Total)
    if market=="spreads":
        raw=am+sp; valid=np.isfinite(raw)&~np.isclose(raw,0,atol=1e-9); y=(raw>0).astype(float); expected=np.full(len(g),.5)
    elif market=="totals":
        raw=at-tt; valid=np.isfinite(raw)&~np.isclose(raw,0,atol=1e-9); y=(raw>0).astype(float); expected=np.full(len(g),.5)
    elif market=="h2h":
        expected=_float_array(g.H2H_close_novig_reference)
        valid=np.isfinite(am)&~np.isclose(am,0,atol=1e-9)&np.isfinite(expected)&(expected>0)&(expected<1)
        y=(am>0).astype(float)
    else: raise ValueError(market)
    return y,valid,expected


def _bh(pvals: List[float]) -> np.ndarray:
    a=np.asarray(pvals,float); out=np.full(len(a),np.nan); ok=np.flatnonzero(np.isfinite(a))
    if not len(ok): return out
    order=ok[np.argsort(a[ok])]; m=len(order); prev=1.0
    for rank in range(m,0,-1):
        i=order[rank-1]; q=min(prev,float(a[i])*m/rank,1.0); out[i]=q; prev=q
    return out


def _normal_one_sided(mean: float, se: float) -> float:
    if not (math.isfinite(mean) and math.isfinite(se)) or se<=0: return 1.0
    z=mean/se
    return .5*math.erfc(z/math.sqrt(2.0))


def _candidate_eval(y, expected, valid, season, mask, direction, market):
    """Evaluate a frozen direction in discovery/shadow/confirmation blocks."""
    out={}
    for label,sm in (("discovery",np.isin(season,DISCOVERY_SEASONS)),("shadow",season==SHADOW_SEASON),("confirm",season==CONFIRM_SEASON)):
        ix=np.flatnonzero(mask&valid&sm)
        if market=="h2h":
            yy=y[ix] if direction=="PLAY_ON" else 1-y[ix]
            ee=expected[ix] if direction=="PLAY_ON" else 1-expected[ix]
            edge=float(np.mean(yy-ee)) if len(ix) else np.nan
            out[label]={"n":len(ix),"rate":float(np.mean(yy)) if len(ix) else np.nan,
                        "expected":float(np.mean(ee)) if len(ix) else np.nan,"edge":edge}
        else:
            obs=y[ix] if direction=="PLAY_ON" else 1-y[ix]
            out[label]={"n":len(ix),"rate":float(np.mean(obs)) if len(ix) else np.nan,"expected":.5,
                        "edge":float(np.mean(obs)-.5) if len(ix) else np.nan}
    return out


def _ablation_report(g, atoms_by_name, conditions, direction, market, y, expected, valid, season):
    if len(conditions)<2: return []
    full=np.ones(len(g),bool)
    for c in conditions: full &= atoms_by_name[c]
    full_eval=_candidate_eval(y,expected,valid,season,full,direction,market)
    out=[]
    for omit in conditions:
        parent=np.ones(len(g),bool)
        for c in conditions:
            if c!=omit: parent &= atoms_by_name[c]
        pe=_candidate_eval(y,expected,valid,season,parent,direction,market)
        metric="edge" if market=="h2h" else "rate"
        out.append({"omitted":omit,
                    "parent_discovery":pe["discovery"].get(metric),
                    "parent_shadow":pe["shadow"].get(metric),
                    "parent_confirm":pe["confirm"].get(metric),
                    "increment_discovery":full_eval["discovery"].get(metric)-pe["discovery"].get(metric) if math.isfinite(full_eval["discovery"].get(metric,np.nan)) and math.isfinite(pe["discovery"].get(metric,np.nan)) else np.nan,
                    "increment_shadow":full_eval["shadow"].get(metric)-pe["shadow"].get(metric) if math.isfinite(full_eval["shadow"].get(metric,np.nan)) and math.isfinite(pe["shadow"].get(metric,np.nan)) else np.nan,
                    "increment_confirm":full_eval["confirm"].get(metric)-pe["confirm"].get(metric) if math.isfinite(full_eval["confirm"].get(metric,np.nan)) and math.isfinite(pe["confirm"].get(metric,np.nan)) else np.nan})
    return out


def _continuous_residual(g: pd.DataFrame, market: str) -> np.ndarray:
    if market=="spreads": return _float_array(g.spread_market_error)
    if market=="totals": return _float_array(g.total_market_error)
    raise ValueError(market)


def _residual_context(g, mask, market, direction):
    r=_continuous_residual(g,market)
    if direction in ("FADE","OPPOSITE_SIDE","UNDER"): r=-r
    season=_float_array(g.Season); valid=np.isfinite(r)
    result={}
    for name,sm in (("discovery",np.isin(season,DISCOVERY_SEASONS)),("shadow",season==SHADOW_SEASON),("confirm",season==CONFIRM_SEASON)):
        x=r[mask&valid&sm]
        result[name]={"n":int(len(x)),"mean_signed_market_error":float(np.mean(x)) if len(x) else np.nan,
                      "sd":float(np.std(x,ddof=1)) if len(x)>1 else np.nan}
    return result


def _miner(g: pd.DataFrame, market: str, max_depth: int = 2) -> dict:
    y,valid,expected=_target(g,market); atoms=_miner_atoms(g); season=_float_array(g.Season)
    disc=valid&np.isin(season,DISCOVERY_SEASONS); shadow=valid&(season==SHADOW_SEASON); confirm=valid&(season==CONFIRM_SEASON)
    if disc.sum()<700 or shadow.sum()<250 or confirm.sum()<250:
        return {"status":"INSUFFICIENT_HISTORY","market":market,"atoms":len(atoms),"production_authority":0}
    candidates=[]; seen_masks=set(); combos=[]; atoms_by_name={a["name"]:a["mask"] for a in atoms}
    for a in atoms: combos.append((a,))
    if max_depth>=2:
        for a,b in combinations(atoms,2):
            if a["family"]==b["family"]: continue
            if not ({a["family"],b["family"]}&{"MODEL_STATE","MODEL_AGREEMENT"}): continue
            combos.append((a,b))
    for combo in combos:
        m=np.ones(len(g),bool); names=[]; fams=[]
        for a in combo: m &= a["mask"]; names.append(a["name"]); fams.append(a["family"])
        ix=np.flatnonzero(m&disc); sx=np.flatnonzero(m&shadow); cx=np.flatnonzero(m&confirm)
        if len(ix)<45 or len(sx)<10 or len(cx)<10: continue
        key=hashlib.sha1(np.packbits(m&disc).tobytes()).hexdigest()[:16]
        if key in seen_masks: continue
        seen_masks.add(key)
        if market=="h2h":
            raw_edge=float(np.mean(y[ix]-expected[ix])); direction="PLAY_ON" if raw_edge>=0 else "FADE"
        else:
            raw=float(np.mean(y[ix])); direction="PLAY_ON" if raw>=.5 else "FADE"
        ev=_candidate_eval(y,expected,valid,season,m,direction,market)
        rates=[]; blocks=[]
        for sy in DISCOVERY_SEASONS:
            jj=np.flatnonzero(m&valid&(season==sy))
            if len(jj)<10: continue
            if market=="h2h":
                yy=y[jj] if direction=="PLAY_ON" else 1-y[jj]; ee=expected[jj] if direction=="PLAY_ON" else 1-expected[jj]
                val=float(np.mean(yy-ee))
            else:
                obs=y[jj] if direction=="PLAY_ON" else 1-y[jj]; val=float(np.mean(obs))
            rates.append(val); blocks.append((sy,jj,val))
        if len(rates)<2: continue
        best=max(blocks,key=lambda x:x[2])[0]
        rem=np.concatenate([jj for sy,jj,rr in blocks if sy!=best])
        if market=="h2h":
            yy=y[rem] if direction=="PLAY_ON" else 1-y[rem]; ee=expected[rem] if direction=="PLAY_ON" else 1-expected[rem]
            remove_best=float(np.mean(yy-ee)) if len(rem) else np.nan
            yy_d=y[ix] if direction=="PLAY_ON" else 1-y[ix]; ee_d=expected[ix] if direction=="PLAY_ON" else 1-expected[ix]
            mean_edge=float(np.mean(yy_d-ee_d)); se=math.sqrt(float(np.sum(ee_d*(1-ee_d))))/len(ix)
            pval=_normal_one_sided(mean_edge,se)
        else:
            obs=y if direction=="PLAY_ON" else 1-y
            remove_best=float(np.mean(obs[rem])) if len(rem) else np.nan
            dr=ev["discovery"]["rate"]; z=(dr-.5)/max(math.sqrt(.25/len(ix)),1e-9); pval=.5*math.erfc(z/math.sqrt(2.0))
        c={"conditions":names,"families":sorted(set(fams)),"direction":direction,
           "discovery_n":len(ix),"shadow_n":len(sx),"confirm_n":len(cx),
           "discovery_season_values":rates,"remove_best_discovery_value":remove_best,
           "stable_discovery_fraction":float(np.mean(np.asarray(rates)>(0 if market=="h2h" else .5))),
           "pvalue":pval,"mask":m,
           "ablation":_ablation_report(g,atoms_by_name,names,direction,market,y,expected,valid,season)}
        for block in ("discovery","shadow","confirm"):
            if market=="h2h":
                c[f"{block}_rate"]=ev[block]["rate"]; c[f"{block}_market_expected"]=ev[block]["expected"]; c[f"{block}_market_edge"]=ev[block]["edge"]
            else:
                c[f"{block}_rate"]=ev[block]["rate"]
        if market in ("spreads","totals"):
            c["continuous_residual"]=_residual_context(g,m,market,direction)
        candidates.append(c)
    if not candidates:
        return {"status":"NO_SYSTEMS","market":market,"atoms":len(atoms),"tested_hypotheses":0,"systems":[],"production_authority":0}
    q=_bh([c["pvalue"] for c in candidates])
    for c,qv in zip(candidates,q):
        c["fdr_qvalue"]=float(qv)
        c["system_id"]="NFL-"+market.upper()+"-"+hashlib.sha1((c["direction"]+"|"+"|".join(c["conditions"])).encode()).hexdigest()[:10].upper()
        if market=="h2h":
            c["quality"]=2*c["discovery_market_edge"]+max(0,c["shadow_market_edge"])+max(0,c["confirm_market_edge"])+max(0,c["remove_best_discovery_value"])
            promising=bool(c["discovery_n"]>=60 and c["shadow_n"]>=18 and c["confirm_n"]>=18 and c["discovery_market_edge"]>=.04 and c["shadow_market_edge"]>=.02 and c["confirm_market_edge"]>=.02 and c["remove_best_discovery_value"]>=.025 and c["stable_discovery_fraction"]>=1.0 and c["fdr_qvalue"]<=.10)
            watch=bool(c["discovery_n"]>=50 and c["discovery_market_edge"]>=.02 and c["shadow_market_edge"]>0 and c["confirm_market_edge"]>0 and c["fdr_qvalue"]<=.20)
            raw=c["discovery_rate"]; c["shrunk_discovery_rate"]=(raw*c["discovery_n"]+.5*60)/(c["discovery_n"]+60)
        else:
            c["quality"]=(c["discovery_rate"]-.5)*2+max(0,c["shadow_rate"]-.5)+max(0,c["confirm_rate"]-.5)+max(0,c["remove_best_discovery_value"]-.5)
            promising=bool(c["discovery_n"]>=60 and c["shadow_n"]>=20 and c["confirm_n"]>=20 and c["discovery_rate"]>=.56 and c["shadow_rate"]>=.535 and c["confirm_rate"]>=.535 and c["remove_best_discovery_value"]>=.54 and c["stable_discovery_fraction"]>=1.0 and c["fdr_qvalue"]<=.10)
            watch=bool(c["discovery_n"]>=55 and c["discovery_rate"]>=.54 and c["shadow_rate"]>=.50 and c["confirm_rate"]>=.50 and c["remove_best_discovery_value"]>=.515 and c["stable_discovery_fraction"]>=.67 and c["fdr_qvalue"]<=.20)
            raw=c["discovery_rate"]; c["shrunk_discovery_rate"]=(raw*c["discovery_n"]+.5*60)/(c["discovery_n"]+60)
        c["research_state"]="PROMISING" if promising else "WATCHLIST" if watch else "SHADOW"
        c["production_authority"]=0
    rank={"PROMISING":2,"WATCHLIST":1,"SHADOW":0}
    candidates.sort(key=lambda x:(rank[x["research_state"]],x["quality"],x.get("confirm_market_edge",x.get("confirm_rate",0)),x["discovery_n"]),reverse=True)
    reps=[]
    for c in candidates:
        m=np.asarray(c["mask"],bool)&valid; keep=True
        for r in reps:
            rm=np.asarray(r["mask"],bool)&valid; union=(m|rm).sum(); jac=float((m&rm).sum()/union) if union else 0.0
            if jac>.80: keep=False; break
        if keep: reps.append(c)
        if len(reps)>=20: break
    def clean_obj(v):
        if isinstance(v,float): return round(v,6) if math.isfinite(v) else None
        if isinstance(v,dict): return {k:clean_obj(x) for k,x in v.items()}
        if isinstance(v,list): return [clean_obj(x) for x in v]
        return v
    def clean(c): return {k:clean_obj(v) for k,v in c.items() if k not in ("mask","pvalue","quality")}
    systems=[clean(c) for c in candidates[:50]]; independent=[clean(c) for c in reps]
    return {"status":"RESEARCH_COMPLETE","market":market,"atoms":len(atoms),"tested_hypotheses":len(candidates),
            "systems":systems,"independent_representatives":independent,
            "promising_count":sum(s["research_state"]=="PROMISING" for s in systems),
            "watchlist_count":sum(s["research_state"]=="WATCHLIST" for s in systems),
            "discovery_seasons":list(DISCOVERY_SEASONS),"shadow_season":SHADOW_SEASON,"confirm_season":CONFIRM_SEASON,
            "production_authority":0,"admission_contract":"DISCOVERY_DIRECTION_FROZEN__H2H_MARKET_NULL__2024_SHADOW__2025_CONFIRM__STRICT_PROMISING__ABLATION__FDR__MASK_DEDUP__ZERO_AUTHORITY"}


def _residual_miner(g: pd.DataFrame, market: str) -> dict:
    if market not in ("spreads","totals"): raise ValueError(market)
    r=_continuous_residual(g,market); valid=np.isfinite(r); season=_float_array(g.Season); atoms=_miner_atoms(g)
    disc=valid&np.isin(season,DISCOVERY_SEASONS); shadow=valid&(season==SHADOW_SEASON); confirm=valid&(season==CONFIRM_SEASON)
    combos=[(a,) for a in atoms]
    for a,b in combinations(atoms,2):
        if a["family"]!=b["family"] and ({a["family"],b["family"]}&{"MODEL_STATE","MODEL_AGREEMENT"}): combos.append((a,b))
    rows=[]; seen=set()
    for combo in combos:
        m=np.ones(len(g),bool)
        for a in combo: m&=a["mask"]
        ix=np.flatnonzero(m&disc); sx=np.flatnonzero(m&shadow); cx=np.flatnonzero(m&confirm)
        if len(ix)<50 or len(sx)<12 or len(cx)<12: continue
        key=hashlib.sha1(np.packbits(m&disc).tobytes()).hexdigest()[:16]
        if key in seen: continue
        seen.add(key)
        mu=float(np.mean(r[ix])); direction=("SELECTED_SIDE" if mu>=0 else "OPPOSITE_SIDE") if market=="spreads" else ("OVER" if mu>=0 else "UNDER")
        sign=1 if mu>=0 else -1
        d=sign*r[ix]; sh=sign*r[sx]; co=sign*r[cx]
        sd=float(np.std(d,ddof=1)) if len(d)>1 else np.nan; se=sd/math.sqrt(len(d)) if sd>0 else np.nan
        pval=_normal_one_sided(float(np.mean(d)),se)
        vals=[]; blocks=[]
        for sy in DISCOVERY_SEASONS:
            xx=sign*r[m&valid&(season==sy)]
            if len(xx)>=10: vals.append(float(np.mean(xx))); blocks.append((sy,xx,vals[-1]))
        if len(vals)<2: continue
        best=max(blocks,key=lambda z:z[2])[0]; rem=np.concatenate([xx for sy,xx,v in blocks if sy!=best]); rb=float(np.mean(rem))
        base_sd=float(np.std(sign*r[disc],ddof=1)) if disc.sum()>1 else np.nan
        rows.append({"conditions":[a["name"] for a in combo],"families":sorted({a["family"] for a in combo}),"direction":direction,
                     "discovery_n":len(d),"discovery_mean_error":float(np.mean(d)),"shadow_n":len(sh),"shadow_mean_error":float(np.mean(sh)),
                     "confirm_n":len(co),"confirm_mean_error":float(np.mean(co)),"discovery_season_mean_errors":vals,
                     "remove_best_discovery_mean_error":rb,"discovery_sd":sd,"variance_ratio_vs_market_all":(sd/base_sd if base_sd>0 else np.nan),
                     "pvalue":pval,"mask":m})
    if not rows: return {"status":"NO_RESIDUAL_SYSTEMS","market":market,"production_authority":0}
    q=_bh([x["pvalue"] for x in rows])
    for x,qv in zip(rows,q):
        x["fdr_qvalue"]=float(qv)
        x["research_state"]="PROMISING" if (x["discovery_mean_error"]>=1.0 and x["shadow_mean_error"]>=.5 and x["confirm_mean_error"]>=.5 and x["remove_best_discovery_mean_error"]>=.75 and x["fdr_qvalue"]<=.10) else "WATCHLIST" if (x["discovery_mean_error"]>=.75 and x["shadow_mean_error"]>0 and x["confirm_mean_error"]>0 and x["fdr_qvalue"]<=.20) else "SHADOW"
        x["system_id"]="NFL-RESIDUAL-"+market.upper()+"-"+hashlib.sha1((x["direction"]+"|"+"|".join(x["conditions"])).encode()).hexdigest()[:10].upper(); x["production_authority"]=0
    rank={"PROMISING":2,"WATCHLIST":1,"SHADOW":0}; rows.sort(key=lambda x:(rank[x["research_state"]],x["confirm_mean_error"],x["discovery_n"]),reverse=True)
    def clean(x): return {k:(round(v,6) if isinstance(v,float) and math.isfinite(v) else None if isinstance(v,float) else v) for k,v in x.items() if k not in ("mask","pvalue")}
    return {"status":"RESEARCH_COMPLETE","market":market,"systems":[clean(x) for x in rows[:25]],
            "promising_count":sum(x["research_state"]=="PROMISING" for x in rows),"watchlist_count":sum(x["research_state"]=="WATCHLIST" for x in rows),
            "tested_hypotheses":len(rows),"production_authority":0}


def _attach_side_calc_to_games(games: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    # physical_games chooses home except neutral canonical. Merge the matching side-derived state.
    cols=["physical_game_id","Team_Norm","calc_dog_rate_prior"]
    lookup=state[cols].copy()
    out=games.merge(lookup,on=["physical_game_id","Team_Norm"],how="left",validate="one_to_one")
    return out


def _family_shrinkage(report: dict, plays: pd.DataFrame, prior_strength: float = 40.0) -> dict:
    out={k:dict(v) for k,v in report.items()}
    if plays is None or plays.empty: return out
    uniq=plays.drop_duplicates(["physical_game_id","bet_team","status"])
    w=int(uniq.status.astype(str).str.upper().eq("WIN").sum()); l=int(uniq.status.astype(str).str.upper().eq("LOSS").sum()); n=w+l
    family_rate=w/n if n else .5
    for k,v in out.items():
        nn=int(v.get("wins",0))+int(v.get("losses",0))
        if nn:
            raw=float(v.get("wins",0))/nn
            v["family_prior_rate"]=round(family_rate,6); v["family_prior_strength"]=prior_strength
            v["shrunk_hit_rate"]=round((raw*nn+family_rate*prior_strength)/(nn+prior_strength),6)
    return out


def _arbitration_features(g: pd.DataFrame, market: str) -> Tuple[pd.DataFrame,np.ndarray,np.ndarray]:
    n=lambda c: pd.to_numeric(g.get(c,pd.Series(np.nan,index=g.index)),errors="coerce")
    if market=="spreads":
        core_pred=n("core_margin_pred"); market_pred=-n("Spread_Value"); actual=n("actual_margin")
        edge=n("spread_consensus_edge"); gap=n("spread_model_gap"); agree=g.spread_models_agree.astype(float)
    elif market=="totals":
        core_pred=n("core_total_pred"); market_pred=n("Current_Total"); actual=n("actual_total")
        edge=n("total_consensus_edge"); gap=n("total_model_gap"); agree=g.total_models_agree.astype(float)
    else: raise ValueError(market)
    core_err=(actual-core_pred).abs(); market_err=(actual-market_pred).abs(); valid=core_err.notna()&market_err.notna()&~np.isclose(core_err,market_err)
    y=(core_err<market_err).astype(float).to_numpy(float)
    X=pd.DataFrame({
        "edge_abs":edge.abs(),"edge_signed":edge,"model_gap":gap,"models_agree":agree,
        "market_spread_abs":n("Spread_Value").abs(),"market_total":n("Current_Total"),"week":n("Week_Number"),
        "division":n("Is_Division_Game"),"primetime":n("Is_PrimeTime"),"rest_diff":n("Rest_Differential_Days"),
        "prev_su_margin":n("Prev_SU_Margin"),"prev_ats_win":n("Prev_ATS_Win"),"revenge":n("Revenge_Flag_Current"),
        "h2h_model_market_gap":n("h2h_market_delta").abs(),
    })
    return X,y,valid.to_numpy(bool)


def _arbitration_model(g: pd.DataFrame, market: str) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    X,y,valid=_arbitration_features(g,market); seasons=_float_array(g.Season); pieces=[]
    for vy in (2023,2024,2025):
        tr=valid&(seasons<vy)&(seasons>=2021); va=valid&(seasons==vy)
        if tr.sum()<450 or va.sum()<200: continue
        A=X.loc[tr].apply(pd.to_numeric,errors="coerce").astype(float); B=X.loc[va].apply(pd.to_numeric,errors="coerce").astype(float)
        med=A.median(axis=0,skipna=True).fillna(0.0); A=A.fillna(med); B=B.fillna(med)
        sc=StandardScaler().fit(A); model=LogisticRegression(C=.25,max_iter=1000,solver="lbfgs",random_state=31)
        model.fit(sc.transform(A),y[tr]); p=model.predict_proba(sc.transform(B))[:,1]
        pieces.append(pd.DataFrame({"Season":seasons[va],"y":y[va],"p":p}))
    if not pieces: return {"status":"INSUFFICIENT_OOF","market":market,"production_authority":0}
    o=pd.concat(pieces,ignore_index=True); yy=o.y.to_numpy(float); pp=o.p.to_numpy(float); base=float(np.mean(yy))
    auc=float(roc_auc_score(yy,pp)) if len(np.unique(yy))>1 else np.nan
    qlo=np.quantile(pp,.25); qhi=np.quantile(pp,.75)
    return {"status":"OOF_RESEARCH_COMPLETE","market":market,"n":int(len(o)),"seasons":sorted(int(x) for x in o.Season.unique()),
            "core_beats_market_rate":round(base,6),"auc":round(auc,6) if math.isfinite(auc) else None,
            "brier":round(float(brier_score_loss(yy,pp)),6),"log_loss":round(float(log_loss(yy,np.clip(pp,.001,.999))),6),
            "top_quartile_n":int((pp>=qhi).sum()),"top_quartile_core_win_rate":round(float(np.mean(yy[pp>=qhi])),6),
            "bottom_quartile_n":int((pp<=qlo).sum()),"bottom_quartile_core_win_rate":round(float(np.mean(yy[pp<=qlo])),6),
            "note":"Predicts whether CORE absolute error beats the retrospective closing market; research only.","production_authority":0}


def _uncertainty_report(g: pd.DataFrame) -> dict:
    out={}
    for market,pred_col,actual_col,gap_col in (("spreads","core_margin_pred","actual_margin","spread_model_gap"),("totals","core_total_pred","actual_total","total_model_gap")):
        resid=(pd.to_numeric(g[actual_col],errors="coerce")-pd.to_numeric(g[pred_col],errors="coerce")).abs(); season=pd.to_numeric(g.Season,errors="coerce")
        folds=[]
        for vy in (2022,2023,2024,2025):
            cal=resid[(season<vy)&(season>=2021)].dropna(); val=resid[season.eq(vy)].dropna()
            if len(cal)<250 or len(val)<200: continue
            q80=float(np.quantile(cal,.80)); q90=float(np.quantile(cal,.90))
            folds.append({"season":vy,"calibration_n":int(len(cal)),"validation_n":int(len(val)),"q80":round(q80,4),"q90":round(q90,4),
                          "coverage80":round(float(np.mean(val<=q80)),6),"coverage90":round(float(np.mean(val<=q90)),6)})
        gap=pd.to_numeric(g[gap_col],errors="coerce"); low=resid[gap.le(1)].dropna(); high=resid[gap.gt(2)].dropna()
        out[market]={"folds":folds,"low_model_gap_le1_mae":round(float(low.mean()),6) if len(low) else None,
                     "high_model_gap_gt2_mae":round(float(high.mean()),6) if len(high) else None,
                     "low_gap_n":int(len(low)),"high_gap_n":int(len(high)),"production_authority":0}
    return out


def _neighborhood_diagnostics(g: pd.DataFrame) -> dict:
    sp=pd.to_numeric(g.Spread_Value,errors="coerce"); am=pd.to_numeric(g.actual_margin,errors="coerce")
    ats=am+sp; valid=ats.notna()&~np.isclose(ats,0); rows=[]
    for lo in (6.0,6.5,7.0,7.5,8.0):
        hi=lo+1.0; m=valid&sp.gt(lo)&sp.lt(hi); n=int(m.sum())
        rows.append({"window":f"dog>{lo:g}&<{hi:g}","n":n,"hit_rate":round(float((ats[m]>0).mean()),6) if n else None})
    threshold=[]
    for t in (3.0,3.5,4.0,4.5,5.0):
        agree=g.spread_models_agree.astype(bool); ce=pd.to_numeric(g.spread_consensus_edge,errors="coerce"); m=valid&agree&ce.abs().ge(t); n=int(m.sum())
        correct=np.sign(ce[m].to_numpy(float))==np.sign(ats[m].to_numpy(float))
        threshold.append({"threshold":t,"n":n,"follow_core_accuracy":round(float(np.mean(correct)),6) if n else None})
    return {"dog_key_7_neighborhood":rows,"spread_consensus_threshold_neighborhood":threshold,"production_authority":0}


def _market_learning(g: pd.DataFrame) -> dict:
    out={}
    for market,pred_col,open_col,close_col,actual_col,kind in (
        ("spreads","core_margin_pred","Opening_Spread","Spread_Value","actual_margin","spread"),
        ("totals","core_total_pred","Opening_Total","Current_Total","actual_total","total")):
        pred=pd.to_numeric(g[pred_col],errors="coerce"); op=pd.to_numeric(g[open_col],errors="coerce"); cl=pd.to_numeric(g[close_col],errors="coerce"); actual=pd.to_numeric(g[actual_col],errors="coerce")
        if kind=="spread": op=-op; cl=-cl
        valid=pred.notna()&op.notna()&cl.notna()&actual.notna(); changed=valid&~np.isclose(op,cl)
        before=(pred-op).abs(); after=(pred-cl).abs(); toward=changed&(after<before); away=changed&(after>before)
        def block(mask):
            n=int(mask.sum())
            return {"n":n,"core_beats_close_rate":round(float(((actual-pred).abs()[mask]<(actual-cl).abs()[mask]).mean()),6) if n else None,
                    "mean_close_distance_to_core":round(float(after[mask].mean()),6) if n else None}
        out[market]={"changed_line_games":int(changed.sum()),"moved_toward_core_rate":round(float(toward.sum()/changed.sum()),6) if changed.sum() else None,
                     "mean_distance_reduction":round(float((before[changed]-after[changed]).mean()),6) if changed.sum() else None,
                     "toward_core":block(toward),"away_from_core":block(away),
                     "note":"Retrospective opener-to-close diagnostic only; not executable CLV.","production_authority":0}
    return out


def _revenge_total_directional(g: pd.DataFrame) -> dict:
    rev=pd.to_numeric(g.Revenge_Flag_Current,errors="coerce").eq(1); edge=pd.to_numeric(g.total_consensus_edge,errors="coerce")
    actual=pd.to_numeric(g.actual_total,errors="coerce")-pd.to_numeric(g.Current_Total,errors="coerce"); season=pd.to_numeric(g.Season,errors="coerce")
    out={}
    for name,m,follow_over in (("REVENGE_CONSENSUS_OVER_4",rev&edge.ge(4),True),("REVENGE_CONSENSUS_UNDER_4",rev&edge.le(-4),False)):
        d={}
        for label,sm in (("discovery",season.isin(DISCOVERY_SEASONS)),("shadow",season.eq(SHADOW_SEASON)),("confirm",season.eq(CONFIRM_SEASON))):
            mm=m&sm&actual.notna()&~np.isclose(actual,0); n=int(mm.sum())
            follow=(actual[mm]>0) if follow_over else (actual[mm]<0)
            d[label]={"n":n,"follow_model_rate":round(float(follow.mean()),6) if n else None,"mean_market_error":round(float(actual[mm].mean()),6) if n else None}
        out[name]=d
    return out


def _source_topology(core: pd.DataFrame, state: pd.DataFrame, bigal_plays: pd.DataFrame,
                     pathi_plays: pd.DataFrame, miner: dict) -> dict:
    g=core.loc[core.Season.eq(CONFIRM_SEASON)].copy()
    if g.empty: return {"status":"NO_CONFIRM_GAMES"}
    selected=dict(zip(g.physical_game_id,g.Team_Norm.astype(str))); raw=[]
    for _,r in g.iterrows():
        de=float(r.direct_spread_edge); se=float(r.score_spread_edge)
        if np.isfinite(de) and np.isfinite(se) and np.sign(de)==np.sign(se) and min(abs(de),abs(se))>=4:
            raw.append((r.physical_game_id,"CORE",1 if de>0 else -1,"CORE_CONSENSUS"))
    def add_frame(p,source):
        if p is None or p.empty:return
        for _,r in p.loc[p.physical_game_id.isin(g.physical_game_id)].iterrows():
            sel=selected.get(r.physical_game_id); raw.append((r.physical_game_id,source,1 if str(r.bet_team)==str(sel) else -1,str(r.system_id)))
    add_frame(bigal_plays,"BIGAL"); add_frame(pathi_plays,"PATHI")
    reps=(miner.get("spreads") or {}).get("independent_representatives") or []; atoms={a["name"]:a["mask"] for a in _miner_atoms(core)}
    for sys in reps:
        if sys.get("research_state")!="PROMISING": continue
        mm=np.ones(len(core),bool)
        for c in sys.get("conditions") or []: mm&=atoms.get(c,np.zeros(len(core),bool))
        for i in np.flatnonzero(mm&core.Season.eq(CONFIRM_SEASON).to_numpy(bool)):
            raw.append((core.iloc[i].physical_game_id,"MINER",1 if sys.get("direction")=="PLAY_ON" else -1,sys.get("system_id")))
    if not raw:return {"status":"NO_SOURCE_EVENTS","confirm_season":CONFIRM_SEASON}
    ev=pd.DataFrame(raw,columns=["physical_game_id","source","direction","system_id"])
    resolved=[]; internal={}
    for (gid,src),p in ev.groupby(["physical_game_id","source"]):
        dirs=set(int(x) for x in p.direction)
        if len(dirs)==1: resolved.append((gid,src,next(iter(dirs))))
        else: internal[src]=internal.get(src,0)+1
    rv=pd.DataFrame(resolved,columns=["physical_game_id","source","direction"])
    rows=[]
    for gid,p in rv.groupby("physical_game_id"):
        dirs=set(int(x) for x in p.direction); rows.append({"physical_game_id":gid,"family_count":p.source.nunique(),"state":"AGREE" if len(dirs)==1 else "CONFLICT"})
    rr=pd.DataFrame(rows)
    multi=rr.family_count.ge(2) if not rr.empty else pd.Series([],dtype=bool)
    return {"status":"DESCRIPTIVE_ONLY","confirm_season":CONFIRM_SEASON,"raw_event_counts":ev.source.value_counts().to_dict(),
            "resolved_family_events":rv.source.value_counts().to_dict() if not rv.empty else {},"internal_conflict_games_by_source":internal,
            "games_with_resolved_family_opinion":int(len(rr)),"multi_family_games":int(multi.sum()) if len(rr) else 0,
            "cross_family_agreement_games":int((multi&rr.state.eq("AGREE")).sum()) if len(rr) else 0,
            "cross_family_conflict_games":int((multi&rr.state.eq("CONFLICT")).sum()) if len(rr) else 0,
            "production_authority":0}


def intelligence_research(side_rows: pd.DataFrame, log_func=print) -> dict:
    if side_rows.Season.eq(SEALED_YEAR).any() or not side_rows.Season.isin(EXPERIMENT_SEASONS).all():
        raise RuntimeError("[NFL-INTEL-V1.8-HOLD] SEALED_YEAR_PRESENT")
    _validate_side_grain(side_rows)
    state=_prepare_side_state(side_rows); core=_attach_side_calc_to_games(_generate_oof_core(side_rows),state)
    spread_agree=core.spread_models_agree.to_numpy(bool); total_agree=core.total_models_agree.to_numpy(bool)
    core_report={"oof_games":int(len(core)),"seasons":sorted(int(x) for x in core.Season.unique()),
        "spread_model_agreement_games":int(spread_agree.sum()),"spread_both_edge_4_games":int((spread_agree&core.direct_spread_edge.abs().ge(4)&core.score_spread_edge.abs().ge(4)).sum()),
        "total_model_agreement_games":int(total_agree.sum()),"total_both_edge_4_games":int((total_agree&core.direct_total_edge.abs().ge(4)&core.score_total_edge.abs().ge(4)).sum()),
        "margin_model_gap_mae":round(float(core.spread_model_gap.mean()),6),"total_model_gap_mae":round(float(core.total_model_gap.mean()),6),
        "interpretation":"DIRECT_AND_SCORE_ARE_ONE_CORE_FAMILY_NOT_TWO_INDEPENDENT_VOTES","production_authority":0}
    log_func("[NFL-INTEL-V1.8-CORE] "+json.dumps(core_report,sort_keys=True,default=str))

    bigal,bigal_plays=_bigal_systems(state); bigal=_family_shrinkage(bigal,bigal_plays); log_func("[NFL-INTEL-V1.8-BIGAL] "+json.dumps(bigal,sort_keys=True,default=str))
    pathi,pathi_plays=_pathi_engineering(state); pathi=_family_shrinkage(pathi,pathi_plays); log_func("[NFL-INTEL-V1.8-PATHI] "+json.dumps(pathi,sort_keys=True,default=str))

    arbitration={m:_arbitration_model(core,m) for m in ("spreads","totals")}; log_func("[NFL-INTEL-V1.8-ARBITRATION] "+json.dumps(arbitration,sort_keys=True,default=str))
    uncertainty=_uncertainty_report(core); log_func("[NFL-INTEL-V1.8-UNCERTAINTY] "+json.dumps(uncertainty,sort_keys=True,default=str))
    neighborhood=_neighborhood_diagnostics(core); log_func("[NFL-INTEL-V1.8-NEIGHBORHOOD] "+json.dumps(neighborhood,sort_keys=True,default=str))
    market_learning=_market_learning(core); log_func("[NFL-INTEL-V1.8-MARKET-LEARNING] "+json.dumps(market_learning,sort_keys=True,default=str))
    revenge_total=_revenge_total_directional(core); log_func("[NFL-INTEL-V1.8-REVENGE-TOTAL] "+json.dumps(revenge_total,sort_keys=True,default=str))

    miner={m:_miner(core,m,max_depth=2) for m in ("spreads","h2h","totals")}
    for m,v in miner.items():
        compact={k:v.get(k) for k in ("status","market","atoms","tested_hypotheses","promising_count","watchlist_count","discovery_seasons","shadow_season","confirm_season","production_authority","admission_contract") if k in v}
        compact["independent_representatives"]=(v.get("independent_representatives") or [])[:15]
        log_func(f"[NFL-INTEL-V1.8-MINER-{m.upper()}] "+json.dumps(compact,sort_keys=True,default=str))
    residual={m:_residual_miner(core,m) for m in ("spreads","totals")}
    for m,v in residual.items(): log_func(f"[NFL-INTEL-V1.8-RESIDUAL-{m.upper()}] "+json.dumps(v,sort_keys=True,default=str))
    topology=_source_topology(core,state,bigal_plays,pathi_plays,miner); log_func("[NFL-INTEL-V1.8-MECHANISMS] "+json.dumps(topology,sort_keys=True,default=str))
    report={"status":"RESEARCH_RESULTS_ONLY","source_tag":SOURCE_TAG,"publication":False,"production_authority":0,"year2026":"SEALED","ncaaf":"UNCHANGED","legacy_nfl":"UNCHANGED",
            "core":core_report,"bigal":bigal,"pathi":pathi,"arbitration":arbitration,"uncertainty":uncertainty,"neighborhood":neighborhood,
            "market_learning":market_learning,"revenge_total_directional":revenge_total,"miner":miner,"residual_miner":residual,"mechanisms":topology,
            "limitations":["2026 remains sealed","2025 is confirmation but not globally untouched prospective evidence","historical open/close lines are retrospective and not verified executable quotes","no ROI/CLV claim"]}
    log_func("[NFL-INTEL-V1.8-CONTRACT] status=RESEARCH_RESULTS_ONLY publication=FALSE production_authority=0 year2026=SEALED ncaaf=UNCHANGED legacy_nfl=UNCHANGED h2h_null=MARKET_NOVIG historical_roi_clv=NOT_VERIFIED")
    return report


def run_nfl_intelligence_v1(*, bq_client=None, audit_report=None, log_func=print):
    if not isinstance(audit_report,dict) or audit_report.get("status")!="READY_FOR_OFFLINE_CHALLENGER_SANDBOX":
        raise RuntimeError("[NFL-INTEL-V1.8-HOLD] V1_3_AUDIT_NOT_GREEN")
    from google.cloud import bigquery
    bq=bq_client or bigquery.Client(project="sharplogger")
    cols={f.name for f in bq.get_table(VIEW).schema}; query=build_intelligence_query(cols)
    log_func(f"[NFL-INTEL-V1.8-PREFLIGHT] status=PASS tag={SOURCE_TAG} publication=FALSE year2026=SEALED discovery=2021-2023 shadow=2024 confirm=2025 h2h_null=MARKET_NOVIG")
    df=bq.query(query,job_config=bigquery.QueryJobConfig(use_query_cache=True,maximum_bytes_billed=20*1024**3)).to_dataframe()
    _validate_side_grain(df)
    log_func("[NFL-INTEL-V1.8-GRAIN] "+json.dumps({"source_side_rows":int(len(df)),"physical_games":int(len(df)//2),"games_by_season":{str(int(k)):int(v//2) for k,v in df.groupby("Season").size().items()}},sort_keys=True))
    return intelligence_research(df,log_func=log_func)

