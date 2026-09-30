"""NFL Intelligence Research V1.7.

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
4) MINER: bounded, family-aware discovery over pregame context + OOF model
   states. Discovery=2021-2023, shadow=2024, confirmation=2025. Candidate
   direction is frozen from discovery before later seasons are scored.
5) MECHANISM: de-duplicate near-identical mined masks and summarize source
   agreement/conflict. No source can self-promote to production.

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

SOURCE_TAG = "nfl-intelligence-v1.7-core-bigal-pathi-miner-20260930"
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
    add("MARGIN_MODELS_WITHIN_1","MODEL_AGREEMENT",gap.le(1)); add("MARGIN_MODELS_WITHIN_2","MODEL_AGREEMENT",gap.le(2))
    dt=n("direct_total_edge"); st=n("score_total_edge"); tc=n("total_consensus_edge"); tg=n("total_model_gap")
    tagree=g.get("total_models_agree",pd.Series(False,index=g.index)).astype(bool)
    add("BOTH_TOTAL_OVER_3","MODEL_STATE",tagree&dt.ge(3)&st.ge(3)); add("BOTH_TOTAL_UNDER_3","MODEL_STATE",tagree&dt.le(-3)&st.le(-3))
    add("TOTAL_CONSENSUS_ABS_4","MODEL_STATE",tagree&tc.abs().ge(4)); add("TOTAL_MODELS_WITHIN_2","MODEL_AGREEMENT",tg.le(2))
    h=n("h2h_market_delta"); add("H2H_MODEL_OVER_MARKET_5P","MODEL_STATE",h.ge(.05)); add("H2H_MODEL_UNDER_MARKET_5P","MODEL_STATE",h.le(-.05))
    # Pathi-style market structure contexts, not credited as verified Pathi NFL systems.
    op=n("Opening_Spread"); crossed=_crossed_key(op,sp); toward=sp.lt(op); away=sp.gt(op)
    add("KEY_CROSSED_TOWARD_SELECTED","KEY_NUMBER",crossed&toward); add("KEY_CROSSED_AWAY_SELECTED","KEY_NUMBER",crossed&away)
    add("DOG_TOTAL_SPREAD_GAP_LE10","CROSS_MARKET",sp.gt(0)&(tot-sp.abs()).le(10))
    role=n("calc_dog_rate_prior")
    add("USUALLY_DOG_NOW_FAVORITE","ROLE_REVERSAL",role.ge(.70)&sp.lt(0)); add("USUALLY_FAVORITE_NOW_DOG","ROLE_REVERSAL",role.le(.30)&sp.gt(0))
    return atoms


def _target(g: pd.DataFrame, market: str):
    am=_float_array(g.actual_margin); at=_float_array(g.actual_total)
    sp=_float_array(g.Spread_Value); tt=_float_array(g.Current_Total)
    if market=="spreads":
        raw=am+sp; valid=np.isfinite(raw)&~np.isclose(raw,0,atol=1e-9); y=(raw>0).astype(float)
    elif market=="totals":
        raw=at-tt; valid=np.isfinite(raw)&~np.isclose(raw,0,atol=1e-9); y=(raw>0).astype(float)
    elif market=="h2h":
        valid=np.isfinite(am)&~np.isclose(am,0,atol=1e-9); y=(am>0).astype(float)
    else: raise ValueError(market)
    return y,valid


def _bh(pvals: List[float]) -> np.ndarray:
    a=np.asarray(pvals,float); out=np.full(len(a),np.nan); ok=np.flatnonzero(np.isfinite(a))
    if not len(ok): return out
    order=ok[np.argsort(a[ok])]; m=len(order); prev=1.0
    for rank in range(m,0,-1):
        i=order[rank-1]; q=min(prev,float(a[i])*m/rank,1.0); out[i]=q; prev=q
    return out


def _miner(g: pd.DataFrame, market: str, max_depth: int = 2) -> dict:
    from math import erf, sqrt
    y,valid=_target(g,market); atoms=_miner_atoms(g); season=_float_array(g.Season)
    disc=valid&np.isin(season,DISCOVERY_SEASONS); shadow=valid&(season==SHADOW_SEASON); confirm=valid&(season==CONFIRM_SEASON)
    if disc.sum()<700 or shadow.sum()<250 or confirm.sum()<250:
        return {"status":"INSUFFICIENT_HISTORY","market":market,"atoms":len(atoms),"production_authority":0}
    candidates=[]; seen_masks=set()
    combos=[]
    for a in atoms: combos.append((a,))
    if max_depth>=2:
        for a,b in combinations(atoms,2):
            if a["family"]==b["family"]: continue
            # Require a model-state/mechanism atom in deeper rules; avoids arbitrary database fishing.
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
        raw=float(np.mean(y[ix])); direction="PLAY_ON" if raw>=.5 else "FADE"; obs=y if direction=="PLAY_ON" else 1-y
        dr=float(np.mean(obs[ix])); sr=float(np.mean(obs[sx])); cr=float(np.mean(obs[cx]))
        # Discovery-season stability and remove-best-season.
        rates=[]; blocks=[]
        for sy in DISCOVERY_SEASONS:
            jj=np.flatnonzero(m&valid&(season==sy))
            if len(jj)>=10: rates.append(float(np.mean(obs[jj]))); blocks.append((sy,jj,rates[-1]))
        if len(rates)<2: continue
        best=max(blocks,key=lambda x:x[2])[0]
        rem=np.concatenate([jj for sy,jj,rr in blocks if sy!=best])
        remove_best=float(np.mean(obs[rem])) if len(rem) else np.nan
        z=(dr-.5)/max(math.sqrt(.25/len(ix)),1e-9); pval=.5*(1-erf(z/sqrt(2)))
        candidates.append({
            "conditions":names,"families":sorted(set(fams)),"direction":direction,
            "discovery_n":len(ix),"discovery_rate":dr,"shadow_n":len(sx),"shadow_rate":sr,
            "confirm_n":len(cx),"confirm_rate":cr,"discovery_season_rates":rates,
            "stable_discovery_fraction":float(np.mean(np.asarray(rates)>.5)),
            "remove_best_discovery_rate":remove_best,"pvalue":pval,"mask":m,
        })
    if not candidates:
        return {"status":"NO_SYSTEMS","market":market,"atoms":len(atoms),"tested_hypotheses":0,"systems":[],"production_authority":0}
    q=_bh([c["pvalue"] for c in candidates])
    for c,qv in zip(candidates,q):
        c["fdr_qvalue"]=float(qv)
        c["system_id"]="NFL-"+market.upper()+"-"+hashlib.sha1((c["direction"]+"|"+"|".join(c["conditions"])).encode()).hexdigest()[:10].upper()
        c["quality"]=(c["discovery_rate"]-.5)*2 + max(0,c["shadow_rate"]-.5) + max(0,c["confirm_rate"]-.5) + max(0,c["remove_best_discovery_rate"]-.5)
        promising=bool(c["discovery_n"]>=55 and c["discovery_rate"]>=.54 and c["shadow_rate"]>=.50 and c["confirm_rate"]>=.50 and c["remove_best_discovery_rate"]>=.515 and c["stable_discovery_fraction"]>=.67 and c["fdr_qvalue"]<=.20)
        c["research_state"]="PROMISING" if promising else "SHADOW"
        c["production_authority"]=0
    candidates.sort(key=lambda x:(x["research_state"]=="PROMISING",x["quality"],x["confirm_rate"],x["discovery_n"]),reverse=True)
    # Remove near-duplicate descriptions from the headline set. Jaccard > .80 is treated as same mechanism footprint.
    reps=[]
    for c in candidates:
        m=np.asarray(c["mask"],bool)&valid
        keep=True
        for r in reps:
            rm=np.asarray(r["mask"],bool)&valid; union=(m|rm).sum(); jac=float((m&rm).sum()/union) if union else 0.0
            if jac>.80: keep=False; break
        if keep: reps.append(c)
        if len(reps)>=20: break
    def clean(c):
        return {k:(round(v,6) if isinstance(v,float) and math.isfinite(v) else v) for k,v in c.items() if k not in ("mask","pvalue","quality")}
    systems=[clean(c) for c in candidates[:50]]; independent=[clean(c) for c in reps]
    return {"status":"RESEARCH_COMPLETE","market":market,"atoms":len(atoms),"tested_hypotheses":len(candidates),
            "systems":systems,"independent_representatives":independent,
            "promising_count":sum(s["research_state"]=="PROMISING" for s in systems),
            "discovery_seasons":list(DISCOVERY_SEASONS),"shadow_season":SHADOW_SEASON,"confirm_season":CONFIRM_SEASON,
            "production_authority":0,"admission_contract":"DISCOVERY_DIRECTION_FROZEN__2024_SHADOW__2025_CONFIRM__FDR__REMOVE_BEST_SEASON__MASK_DEDUP__ZERO_AUTHORITY"}


def _attach_side_calc_to_games(games: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    # physical_games chooses home except neutral canonical. Merge the matching side-derived state.
    cols=["physical_game_id","Team_Norm","calc_dog_rate_prior"]
    lookup=state[cols].copy()
    out=games.merge(lookup,on=["physical_game_id","Team_Norm"],how="left",validate="one_to_one")
    return out


def _source_topology(core: pd.DataFrame, state: pd.DataFrame, bigal_plays: pd.DataFrame,
                     pathi_plays: pd.DataFrame, miner: dict) -> dict:
    # 2025 is used only as the miner confirmation season / descriptive topology.
    g=core.loc[core.Season.eq(CONFIRM_SEASON)].copy()
    if g.empty: return {"status":"NO_CONFIRM_GAMES"}
    selected=dict(zip(g.physical_game_id,g.Team_Norm.astype(str)))
    events=[]
    # CORE = two independent margin constructions agree and each differs from market by >=4.
    for _,r in g.iterrows():
        de=float(r.direct_spread_edge); se=float(r.score_spread_edge)
        if np.isfinite(de) and np.isfinite(se) and np.sign(de)==np.sign(se) and min(abs(de),abs(se))>=4:
            events.append((r.physical_game_id,"CORE",1 if de>0 else -1))
    def add_play_frame(p,source):
        if p is None or p.empty: return
        for _,r in p.loc[p.physical_game_id.isin(g.physical_game_id)].iterrows():
            sel=selected.get(r.physical_game_id); direction=1 if str(r.bet_team)==str(sel) else -1
            events.append((r.physical_game_id,source,direction))
    add_play_frame(bigal_plays,"BIGAL"); add_play_frame(pathi_plays,"PATHI")
    # Miner headline systems are evaluated by rebuilding atom masks on the OOF game frame.
    # The stored conditions are deterministic and were frozen before 2025 confirmation.
    reps=(miner.get("spreads") or {}).get("independent_representatives") or []
    atoms={a["name"]:a["mask"] for a in _miner_atoms(core)}
    for sys in reps:
        if sys.get("research_state")!="PROMISING": continue
        mm=np.ones(len(core),bool)
        for c in sys.get("conditions") or []: mm &= atoms.get(c,np.zeros(len(core),bool))
        for i in np.flatnonzero(mm & core.Season.eq(CONFIRM_SEASON).to_numpy(bool)):
            direction=1 if sys.get("direction")=="PLAY_ON" else -1
            events.append((core.iloc[i].physical_game_id,"MINER",direction))
    if not events: return {"status":"NO_SOURCE_EVENTS","confirm_season":CONFIRM_SEASON}
    ev=pd.DataFrame(events,columns=["physical_game_id","source","direction"])
    rows=[]
    for gid,p in ev.groupby("physical_game_id"):
        src=sorted(set(p.source)); dirs=set(int(x) for x in p.direction)
        rows.append({"physical_game_id":gid,"sources":src,"source_count":len(src),"state":"AGREE" if len(dirs)==1 else "CONFLICT"})
    rr=pd.DataFrame(rows)
    return {"status":"DESCRIPTIVE_ONLY","confirm_season":CONFIRM_SEASON,"games_with_any_source":int(len(rr)),
            "multi_source_games":int(rr.source_count.ge(2).sum()),"agreement_games":int(rr.state.eq("AGREE").sum()),
            "conflict_games":int(rr.state.eq("CONFLICT").sum()),
            "source_counts":ev.source.value_counts().to_dict(),"production_authority":0}


def intelligence_research(side_rows: pd.DataFrame, log_func=print) -> dict:
    if side_rows.Season.eq(SEALED_YEAR).any() or not side_rows.Season.isin(EXPERIMENT_SEASONS).all():
        raise RuntimeError("[NFL-INTEL-V1-HOLD] SEALED_YEAR_PRESENT")
    _validate_side_grain(side_rows)
    state=_prepare_side_state(side_rows)
    core=_generate_oof_core(side_rows)
    core=_attach_side_calc_to_games(core,state)

    # Core descriptive diagnostics, without selecting a winner or granting authority.
    spread_agree=core.spread_models_agree.to_numpy(bool); total_agree=core.total_models_agree.to_numpy(bool)
    core_report={
        "oof_games":int(len(core)),"seasons":sorted(int(x) for x in core.Season.unique()),
        "spread_model_agreement_games":int(spread_agree.sum()),
        "spread_both_edge_4_games":int((spread_agree & core.direct_spread_edge.abs().ge(4) & core.score_spread_edge.abs().ge(4)).sum()),
        "total_model_agreement_games":int(total_agree.sum()),
        "total_both_edge_4_games":int((total_agree & core.direct_total_edge.abs().ge(4) & core.score_total_edge.abs().ge(4)).sum()),
        "margin_model_gap_mae":round(float(core.spread_model_gap.mean()),6),
        "total_model_gap_mae":round(float(core.total_model_gap.mean()),6),
        "production_authority":0,
    }
    log_func("[NFL-INTEL-V1-CORE] "+json.dumps(core_report,sort_keys=True,default=str))

    bigal,bigal_plays=_bigal_systems(state); log_func("[NFL-INTEL-V1-BIGAL] "+json.dumps(bigal,sort_keys=True,default=str))
    pathi,pathi_plays=_pathi_engineering(state); log_func("[NFL-INTEL-V1-PATHI] "+json.dumps(pathi,sort_keys=True,default=str))

    miner={m:_miner(core,m,max_depth=2) for m in ("spreads","h2h","totals")}
    for m,v in miner.items():
        compact={k:v.get(k) for k in ("status","market","atoms","tested_hypotheses","promising_count","discovery_seasons","shadow_season","confirm_season","production_authority","admission_contract") if k in v}
        compact["independent_representatives"]=(v.get("independent_representatives") or [])[:15]
        log_func(f"[NFL-INTEL-V1-MINER-{m.upper()}] "+json.dumps(compact,sort_keys=True,default=str))

    topology=_source_topology(core,state,bigal_plays,pathi_plays,miner)
    log_func("[NFL-INTEL-V1-MECHANISMS] "+json.dumps(topology,sort_keys=True,default=str))
    report={"status":"RESEARCH_RESULTS_ONLY","source_tag":SOURCE_TAG,"publication":False,"production_authority":0,
            "year2026":"SEALED","ncaaf":"UNCHANGED","legacy_nfl":"UNCHANGED","core":core_report,
            "bigal":bigal,"pathi":pathi,"miner":miner,"mechanisms":topology,
            "limitations":["2026 remains sealed","2025 is miner confirmation but not globally untouched prospective evidence",
                           "historical open/close lines are retrospective and not verified executable quotes","no ROI/CLV claim"]}
    log_func("[NFL-INTEL-V1-CONTRACT] status=RESEARCH_RESULTS_ONLY publication=FALSE production_authority=0 year2026=SEALED ncaaf=UNCHANGED legacy_nfl=UNCHANGED historical_roi_clv=NOT_VERIFIED")
    return report


def run_nfl_intelligence_v1(*, bq_client=None, audit_report=None, log_func=print):
    if not isinstance(audit_report,dict) or audit_report.get("status")!="READY_FOR_OFFLINE_CHALLENGER_SANDBOX":
        raise RuntimeError("[NFL-INTEL-V1-HOLD] V1_3_AUDIT_NOT_GREEN")
    from google.cloud import bigquery
    bq=bq_client or bigquery.Client(project="sharplogger")
    cols={f.name for f in bq.get_table(VIEW).schema}
    query=build_intelligence_query(cols)
    log_func(f"[NFL-INTEL-V1-PREFLIGHT] status=PASS tag={SOURCE_TAG} publication=FALSE year2026=SEALED discovery=2021-2023 shadow=2024 confirm=2025")
    df=bq.query(query,job_config=bigquery.QueryJobConfig(use_query_cache=True,maximum_bytes_billed=20*1024**3)).to_dataframe()
    _validate_side_grain(df)
    log_func("[NFL-INTEL-V1-GRAIN] "+json.dumps({"source_side_rows":int(len(df)),"physical_games":int(len(df)//2),
             "games_by_season":{str(int(k)):int(v//2) for k,v in df.groupby("Season").size().items()}},sort_keys=True))
    return intelligence_research(df,log_func=log_func)
