"""NFL Research V2.4 System Lab / Betting-System Miner V3.14.0 — Season Record State + Prediction Tracker.

This module is the dedicated situational/system brain.  It deliberately does
NOT use CORE predictions, residual-model outputs, edge-gate probabilities, or
live market-microstructure features during system discovery.  That separation
keeps SYSTEMS genuinely independent from FUNDAMENTAL and MARKET brains.

The lab follows the research discipline developed in the NCAAF System Miner V3 more literally:
logical/domain atoms, bounded beam search, one-side-per-game enforcement,
near-duplicate mask collapse, chronological/LOSO/bootstrap robustness,
remove-best-season and drop-one-condition tests, multiple-testing control, and
frozen later-period validation, and tiered retention of LEGIT / PROMISING / WATCH systems.

Historical design:
  DISCOVERY       2017-2022  (direction + rule selection only)
  SHADOW          2023       (no tuning)
  CONFIRMATION    2024       (no tuning)
  FINAL CHECK     2025       (untouched by selection)
  SEALED          2026       (never queried)

No discovered system has production authority.  V3.11.0 keeps the V3.10.3 horizon-symmetric
advanced-miner grammar and adds an exact expert-occurrence atom bridge. Pathi engineering and
Big Al replay occurrences are materialized from the same occurrence ledgers used for their W/L
records, then exposed as bounded Miner atoms without reconstructing the systems a second time.
Expert-bridge mechanisms are research/shadow only at introduction and cannot immediately become
production confirmation votes. Lookback 1 carries complete prior-game
result/ATS/role/location/magnitude/opponent-state context; lookback 2 adds every exact SU/ATS/
location/role sequence plus two-game magnitude/trend and opponent symmetry; lookback 3 retains
the exact sequence/magnitude/opponent-symmetry grammar from V3.10. Sequence depth is capped at
three completed games and mechanism-family/near-duplicate controls remain mandatory so added
expressiveness does not become independent-vote inflation.
Historical opening/closing lines do not establish executable historical ROI because timestamped
prices and juice are not available at identical canonical snapshots.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from nfl_feature_audit_v1 import VIEW
from nfl_intelligence_v1 import (
    build_intelligence_query, _prepare_side_state, _bigal_systems, _pathi_engineering,
    _swap_within_game,
)
from nfl_research_v2_contract import assert_contract, contract_hash

SOURCE_TAG = "nfl-system-lab-v3.14.0-season-record-state-miner-20261008"
PRODUCTION_AUTHORITY = 0
DISCOVERY_SEASONS = (2017,2018,2019,2020,2021,2022)
SHADOW_SEASON = 2023
CONFIRM_SEASON = 2024
FINAL_CHECK_SEASON = 2025
SEALED_SEASON = 2026
BREAK_EVEN_REFERENCE = 0.52381
MAX_DEPTH = 6
MAX_SEQUENCE_DEPTH = 3
BEAM_WIDTH = 32
FINALISTS = 40
JACCARD_CUTOFF = 0.98
MAXSTAT_REPS = 500
BOOTSTRAP_REPS = 800
STATUS = "NFL_RESEARCH_V2_SYSTEM_LAB_V3_14_0_SEASON_RECORD_STATE_MINER_COMPLETE"

ACADEMIC_REGISTRY = {
    "ACADEMIC_DIVISION_HOME_ATS": {
        "status":"REPLICATION_HYPOTHESIS",
        "source":"Shank (2019), NFL betting-market efficiency/divisional-rivals study",
        "hypothesis":"Divisional familiarity may reduce home-team ATS performance relative to non-division games.",
        "authority":0,
    },
    "ACADEMIC_DIVISION_TOTAL_UNDER": {
        "status":"REPLICATION_HYPOTHESIS",
        "source":"Shank (2019), NFL betting-market efficiency/divisional-rivals study",
        "hypothesis":"Divisional familiarity may lower OVER probability / favor UNDER relative to non-division games.",
        "authority":0,
    },
    "ACADEMIC_REST_ERA_CHANGE": {
        "status":"REPLICATION_HYPOTHESIS",
        "source":"Lopez/Bliss (2024) NFL rest-differential research",
        "hypothesis":"Traditional rest/bye advantage may be weaker in the modern era; treat rest as an era-sensitive hypothesis rather than folklore.",
        "authority":0,
    },
}


def _num(d,c):
    if c not in d: return pd.Series(np.nan,index=d.index,dtype=float)
    return pd.to_numeric(d[c],errors="coerce")


def _status(v):
    if not np.isfinite(v): return "MISSING"
    if v>1e-9: return "WIN"
    if v<-1e-9: return "LOSS"
    return "PUSH"


def prepare_system_state(side_rows: pd.DataFrame) -> pd.DataFrame:
    """Add opening-line settlement and additional prior-only system state."""
    d=_prepare_side_state(side_rows)
    if d.Season.max()>2025: raise RuntimeError("NFL_SYSTEM_LAB_2026_DATA_LEAK")
    op=_num(d,"Opening_Spread"); ot=_num(d,"Opening_Total")
    d["Opening_ATS_Status"]=[_status(m+s) for m,s in zip(_num(d,"actual_margin"),op)]
    d["Opening_ATS_Win"]=d.Opening_ATS_Status.eq("WIN").astype(float)
    d["Opening_ATS_Loss"]=d.Opening_ATS_Status.eq("LOSS").astype(float)
    # Settlement margin is used only as a completed-prior-game descriptor.  The
    # current game's closing outcome never enters discovery features.
    d["Opening_ATS_Cover_Margin"]=_num(d,"actual_margin")+op
    total_points=_num(d,"Team_Score")+_num(d,"Opponent_Score")
    d["Opening_Total_Status"]=["OVER" if np.isfinite(t) and np.isfinite(o) and t-o>1e-9 else "UNDER" if np.isfinite(t) and np.isfinite(o) and t-o<-1e-9 else "PUSH" if np.isfinite(t) and np.isfinite(o) else "MISSING" for t,o in zip(total_points,ot)]
    d["Opening_Dog"]=op.gt(0).astype(float); d["Opening_Favorite"]=op.lt(0).astype(float)
    d["Opening_Spread_Abs"]=op.abs()
    d=d.sort_values(["Season","Team_Norm","Game_Date","Source_Name","Source_Game_ID"],kind="mergesort").copy()
    grp=d.groupby(["Season","Team_Norm"],sort=False,dropna=False)
    d["calc_team_game_number"]=grp.cumcount()+1
    d["calc_prev1_opening_spread"]=grp["Opening_Spread"].shift(1)
    d["calc_prev2_opening_spread"]=grp["Opening_Spread"].shift(2)
    d["calc_prev3_opening_spread"]=grp["Opening_Spread"].shift(3)
    d["calc_prev1_open_dog"]=grp["Opening_Dog"].shift(1)
    d["calc_prev2_open_dog"]=grp["Opening_Dog"].shift(2)
    d["calc_prev3_open_dog"]=grp["Opening_Dog"].shift(3)
    d["calc_prev1_open_favorite"]=grp["Opening_Favorite"].shift(1)
    d["calc_prev2_open_favorite"]=grp["Opening_Favorite"].shift(2)
    d["calc_prev3_open_favorite"]=grp["Opening_Favorite"].shift(3)
    d["calc_prev1_home"]=grp["Is_Home"].shift(1)
    d["calc_prev1_away"]=grp["Is_Away"].shift(1)
    prior_games=grp.cumcount().astype(float)
    dog_before=grp["Opening_Dog"].cumsum()-d["Opening_Dog"]
    home_before=grp["Is_Home"].cumsum()-pd.to_numeric(d["Is_Home"],errors="coerce").fillna(0)
    away_before=grp["Is_Away"].cumsum()-pd.to_numeric(d["Is_Away"],errors="coerce").fillna(0)
    d["calc_open_dog_rate_prior"]=np.where(prior_games.gt(0),dog_before/prior_games,np.nan)
    d["calc_first_home_game"]=(pd.to_numeric(d["Is_Home"],errors="coerce").eq(1)&home_before.eq(0)).astype(float)
    d["calc_first_road_game"]=(pd.to_numeric(d["Is_Away"],errors="coerce").eq(1)&away_before.eq(0)).astype(float)
    d["calc_prev1_margin"]=grp["actual_margin"].shift(1)
    d["calc_prev2_margin"]=grp["actual_margin"].shift(2)
    d["calc_prev3_margin"]=grp["actual_margin"].shift(3)
    d["calc_prev1_open_ats_loss"]=grp["Opening_ATS_Loss"].shift(1)
    d["calc_prev2_open_ats_loss"]=grp["Opening_ATS_Loss"].shift(2)
    d["calc_prev3_open_ats_loss"]=grp["Opening_ATS_Loss"].shift(3)
    d["calc_prev1_open_ats_win"]=grp["Opening_ATS_Win"].shift(1)
    d["calc_prev2_open_ats_win"]=grp["Opening_ATS_Win"].shift(2)
    d["calc_prev3_open_ats_win"]=grp["Opening_ATS_Win"].shift(3)
    d["calc_prev1_open_ats_margin"]=grp["Opening_ATS_Cover_Margin"].shift(1)
    d["calc_prev2_open_ats_margin"]=grp["Opening_ATS_Cover_Margin"].shift(2)
    d["calc_prev3_open_ats_margin"]=grp["Opening_ATS_Cover_Margin"].shift(3)

    # V3.3 sequential-context grammar: retain the prior TWO games' location,
    # scoring, opponent class and schedule context.  This is intentionally
    # prior-only and lets the miner discover Big-Al-style multi-game structures
    # without hard-coding the published answer.
    d["calc_prev2_home"]=grp["Is_Home"].shift(2)
    d["calc_prev3_home"]=grp["Is_Home"].shift(3)
    d["calc_prev2_away"]=grp["Is_Away"].shift(2)
    d["calc_prev3_away"]=grp["Is_Away"].shift(3)
    d["calc_prev1_points_for"]=grp["Team_Score"].shift(1)
    d["calc_prev2_points_for"]=grp["Team_Score"].shift(2)
    d["calc_prev3_points_for"]=grp["Team_Score"].shift(3)
    d["calc_prev1_points_against"]=grp["Opponent_Score"].shift(1)
    d["calc_prev2_points_against"]=grp["Opponent_Score"].shift(2)
    d["calc_prev3_points_against"]=grp["Opponent_Score"].shift(3)
    d["calc_prev1_division"]=grp["Is_Division_Game"].shift(1)
    d["calc_prev2_division"]=grp["Is_Division_Game"].shift(2)
    d["calc_prev3_division"]=grp["Is_Division_Game"].shift(3)
    d["calc_prev1_conference"]=grp["Is_Conference_Game"].shift(1)
    d["calc_prev2_conference"]=grp["Is_Conference_Game"].shift(2)
    d["calc_prev3_conference"]=grp["Is_Conference_Game"].shift(3)
    d["calc_prev1_primetime"]=grp["Is_PrimeTime"].shift(1)
    d["calc_prev2_primetime"]=grp["Is_PrimeTime"].shift(2)
    d["calc_prev3_primetime"]=grp["Is_PrimeTime"].shift(3)
    d["calc_prev1_opp_win_pct"]=grp["Opp_calc_win_pct_prior"].shift(1)
    d["calc_prev2_opp_win_pct"]=grp["Opp_calc_win_pct_prior"].shift(2)
    d["calc_prev3_opp_win_pct"]=grp["Opp_calc_win_pct_prior"].shift(3)
    d["calc_prev1_opponent_norm"]=grp["Opponent_Norm"].shift(1)
    d["calc_immediate_rematch"]=(d["calc_prev1_opponent_norm"].astype(str)==d["Opponent_Norm"].astype(str)).astype(float)

    # "Island" is computed from the schedule rather than guessed from a TV label:
    # one physical NFL game in a date/hour kickoff slot.  It is a historical
    # schedule-context feature, not a market-movement feature.
    _slot=d[["physical_game_id","Game_Date","Game_Hour_ET"]].drop_duplicates("physical_game_id").copy()
    _slot["_date"]=pd.to_datetime(_slot["Game_Date"],errors="coerce").dt.strftime("%Y-%m-%d")
    _slot["_hour"]=pd.to_numeric(_slot["Game_Hour_ET"],errors="coerce")
    _slot["_slot_key"]=_slot["_date"].astype(str)+"|"+_slot["_hour"].astype(str)
    _slot["_slot_games"]=_slot.groupby("_slot_key",dropna=False)["physical_game_id"].transform("nunique")
    _island_ids=set(_slot.loc[_slot["_slot_games"].eq(1),"physical_game_id"].astype(str))
    d["calc_island_game"]=d["physical_game_id"].astype(str).isin(_island_ids).astype(float)
    d["calc_prev1_island"]=grp["calc_island_game"].shift(1)
    d["calc_prev2_island"]=grp["calc_island_game"].shift(2)
    d["calc_prev3_island"]=grp["calc_island_game"].shift(3)

    d["calc_b2b_su_wins"]=(d["calc_prev1_margin"].gt(0)&d["calc_prev2_margin"].gt(0)).astype(float)
    d["calc_b2b_su_losses"]=(d["calc_prev1_margin"].lt(0)&d["calc_prev2_margin"].lt(0)).astype(float)
    d["calc_b2b_home_games"]=(d["calc_prev1_home"].eq(1)&d["calc_prev2_home"].eq(1)).astype(float)
    d["calc_b2b_road_games"]=(d["calc_prev1_away"].eq(1)&d["calc_prev2_away"].eq(1)).astype(float)
    d["calc_b2b_home_su_wins"]=(d["calc_b2b_home_games"].eq(1)&d["calc_b2b_su_wins"].eq(1)).astype(float)
    d["calc_b2b_road_su_wins"]=(d["calc_b2b_road_games"].eq(1)&d["calc_b2b_su_wins"].eq(1)).astype(float)
    d["calc_b2b_scored_31_plus"]=(d["calc_prev1_points_for"].ge(31)&d["calc_prev2_points_for"].ge(31)).astype(float)
    d["calc_b2b_scored_28_plus"]=(d["calc_prev1_points_for"].ge(28)&d["calc_prev2_points_for"].ge(28)).astype(float)
    d["calc_b2b_allowed_17_or_less"]=(d["calc_prev1_points_against"].le(17)&d["calc_prev2_points_against"].le(17)).astype(float)
    d["calc_b2b_home_wins_31_plus"]=(d["calc_b2b_home_su_wins"].eq(1)&d["calc_b2b_scored_31_plus"].eq(1)).astype(float)
    d["calc_b2b_division_games"]=(d["calc_prev1_division"].eq(1)&d["calc_prev2_division"].eq(1)).astype(float)
    d["calc_b2b_nondivision_games"]=(d["calc_prev1_division"].eq(0)&d["calc_prev2_division"].eq(0)).astype(float)
    d["calc_b2b_open_ats_wins"]=(d["calc_prev1_open_ats_win"].eq(1)&d["calc_prev2_open_ats_win"].eq(1)).astype(float)
    d["calc_b2b_opp_ge_600"]=(d["calc_prev1_opp_win_pct"].ge(.6)&d["calc_prev2_opp_win_pct"].ge(.6)).astype(float)
    d["calc_b2b_opp_le_400"]=(d["calc_prev1_opp_win_pct"].le(.4)&d["calc_prev2_opp_win_pct"].le(.4)&d["calc_prev1_opp_win_pct"].notna()&d["calc_prev2_opp_win_pct"].notna()).astype(float)

    # V3.10.2 lookback-two state.  The thresholds mirror the three-game trend
    # definitions so the miner can compare one-, two-, and three-game horizons
    # without silently changing what "meaningful movement" means.
    p1=_num(d,"calc_prev1_margin"); p2=_num(d,"calc_prev2_margin")
    a1=_num(d,"calc_prev1_open_ats_margin"); a2=_num(d,"calc_prev2_open_ats_margin")
    pf1=_num(d,"calc_prev1_points_for"); pf2=_num(d,"calc_prev2_points_for")
    pa1=_num(d,"calc_prev1_points_against"); pa2=_num(d,"calc_prev2_points_against")
    oq1=_num(d,"calc_prev1_opp_win_pct"); oq2=_num(d,"calc_prev2_opp_win_pct")
    d["calc_margin_improving_2"]=(p1.gt(p2)&(p1-p2).ge(10)).astype(float)
    d["calc_margin_worsening_2"]=(p1.lt(p2)&(p2-p1).ge(10)).astype(float)
    d["calc_ats_margin_improving_2"]=(a1.gt(a2)&(a1-a2).ge(7)).astype(float)
    d["calc_ats_margin_worsening_2"]=(a1.lt(a2)&(a2-a1).ge(7)).astype(float)
    d["calc_scoring_improving_2"]=(pf1.gt(pf2)&(pf1-pf2).ge(7)).astype(float)
    d["calc_scoring_worsening_2"]=(pf1.lt(pf2)&(pf2-pf1).ge(7)).astype(float)
    d["calc_defense_improving_2"]=(pa1.lt(pa2)&(pa2-pa1).ge(7)).astype(float)
    d["calc_defense_worsening_2"]=(pa1.gt(pa2)&(pa1-pa2).ge(7)).astype(float)
    d["calc_opp_quality_rising_2"]=(oq1.gt(oq2)&(oq1-oq2).ge(.15)).astype(float)
    d["calc_opp_quality_falling_2"]=(oq1.lt(oq2)&(oq2-oq1).ge(.15)).astype(float)

    # V3.10 bounded prior-three sequence and magnitude state. Every value below
    # is constructed from completed games 1/2/3 immediately before the row.
    p1=_num(d,"calc_prev1_margin"); p2=_num(d,"calc_prev2_margin"); p3=_num(d,"calc_prev3_margin")
    a1=_num(d,"calc_prev1_open_ats_margin"); a2=_num(d,"calc_prev2_open_ats_margin"); a3=_num(d,"calc_prev3_open_ats_margin")
    pf1=_num(d,"calc_prev1_points_for"); pf2=_num(d,"calc_prev2_points_for"); pf3=_num(d,"calc_prev3_points_for")
    pa1=_num(d,"calc_prev1_points_against"); pa2=_num(d,"calc_prev2_points_against"); pa3=_num(d,"calc_prev3_points_against")
    oq1=_num(d,"calc_prev1_opp_win_pct"); oq2=_num(d,"calc_prev2_opp_win_pct"); oq3=_num(d,"calc_prev3_opp_win_pct")
    d["calc_3_su_wins"]=(p1.gt(0)&p2.gt(0)&p3.gt(0)).astype(float)
    d["calc_3_su_losses"]=(p1.lt(0)&p2.lt(0)&p3.lt(0)).astype(float)
    d["calc_3_ats_wins"]=(d["calc_prev1_open_ats_win"].eq(1)&d["calc_prev2_open_ats_win"].eq(1)&d["calc_prev3_open_ats_win"].eq(1)).astype(float)
    d["calc_3_ats_losses"]=(d["calc_prev1_open_ats_loss"].eq(1)&d["calc_prev2_open_ats_loss"].eq(1)&d["calc_prev3_open_ats_loss"].eq(1)).astype(float)
    d["calc_3_home_games"]=(d["calc_prev1_home"].eq(1)&d["calc_prev2_home"].eq(1)&d["calc_prev3_home"].eq(1)).astype(float)
    d["calc_3_road_games"]=(d["calc_prev1_away"].eq(1)&d["calc_prev2_away"].eq(1)&d["calc_prev3_away"].eq(1)).astype(float)
    d["calc_3_scored_28_plus"]=(pf1.ge(28)&pf2.ge(28)&pf3.ge(28)).astype(float)
    d["calc_3_scored_31_plus"]=(pf1.ge(31)&pf2.ge(31)&pf3.ge(31)).astype(float)
    d["calc_3_allowed_17_or_less"]=(pa1.le(17)&pa2.le(17)&pa3.le(17)).astype(float)
    d["calc_3_opp_ge_600"]=(oq1.ge(.6)&oq2.ge(.6)&oq3.ge(.6)).astype(float)
    d["calc_3_opp_le_400"]=(oq1.le(.4)&oq2.le(.4)&oq3.le(.4)&oq1.notna()&oq2.notna()&oq3.notna()).astype(float)
    d["calc_2of3_opp_ge_600"]=(oq1.ge(.6).astype(int)+oq2.ge(.6).astype(int)+oq3.ge(.6).astype(int)>=2).astype(float)
    d["calc_2of3_opp_le_400"]=((oq1.le(.4)&oq1.notna()).astype(int)+(oq2.le(.4)&oq2.notna()).astype(int)+(oq3.le(.4)&oq3.notna()).astype(int)>=2).astype(float)
    d["calc_margin_improving_3"]=(p1.gt(p2)&p2.gt(p3)&(p1-p3).ge(10)).astype(float)
    d["calc_margin_worsening_3"]=(p1.lt(p2)&p2.lt(p3)&(p3-p1).ge(10)).astype(float)
    d["calc_ats_margin_improving_3"]=(a1.gt(a2)&a2.gt(a3)&(a1-a3).ge(7)).astype(float)
    d["calc_ats_margin_worsening_3"]=(a1.lt(a2)&a2.lt(a3)&(a3-a1).ge(7)).astype(float)
    d["calc_scoring_improving_3"]=(pf1.gt(pf2)&pf2.gt(pf3)&(pf1-pf3).ge(7)).astype(float)
    d["calc_scoring_worsening_3"]=(pf1.lt(pf2)&pf2.lt(pf3)&(pf3-pf1).ge(7)).astype(float)
    d["calc_defense_improving_3"]=(pa1.lt(pa2)&pa2.lt(pa3)&(pa3-pa1).ge(7)).astype(float)
    d["calc_defense_worsening_3"]=(pa1.gt(pa2)&pa2.gt(pa3)&(pa1-pa3).ge(7)).astype(float)
    d["calc_opp_quality_rising_3"]=(oq1.gt(oq2)&oq2.gt(oq3)&(oq1-oq3).ge(.15)).astype(float)
    d["calc_opp_quality_falling_3"]=(oq1.lt(oq2)&oq2.lt(oq3)&(oq3-oq1).ge(.15)).astype(float)

    # Team-context memory across ALL earlier seasons/games.  At each row the
    # counters are written before the current result is consumed.  This permits
    # team-agnostic rules such as "team historically strong in primetime" while
    # keeping exact TEAM_IDENTITY patterns as a separate, more conservative lane.
    _ctx_masks={
        "primetime":_num(d,"Is_PrimeTime").eq(1),
        "island":_num(d,"calc_island_game").eq(1),
        "division":_num(d,"Is_Division_Game").eq(1),
        "nondivision":_num(d,"Is_Division_Game").eq(0),
        "conference":_num(d,"Is_Conference_Game").eq(1),
        "interconference":_num(d,"Is_Interconference_Game").eq(1),
        "home_favorite":(_num(d,"Is_Home").eq(1)&_num(d,"Opening_Spread").lt(0)),
        "road_dog":(_num(d,"Is_Away").eq(1)&_num(d,"Opening_Spread").gt(0)),
        # V3.10 sequence-conditioned team memory. These contexts remain prior-only:
        # historical counters are emitted before the current result is consumed.
        "after_3_su_wins":_num(d,"calc_3_su_wins").eq(1),
        "after_3_su_losses":_num(d,"calc_3_su_losses").eq(1),
        "after_3_ats_wins":_num(d,"calc_3_ats_wins").eq(1),
        "after_3_ats_losses":_num(d,"calc_3_ats_losses").eq(1),
        "after_blowout_win21":_num(d,"calc_prev1_margin").ge(21),
        "after_blowout_loss21":_num(d,"calc_prev1_margin").le(-21),
        "after_close_game3":_num(d,"calc_prev1_margin").abs().le(3),
        "after_3_home":_num(d,"calc_3_home_games").eq(1),
        "after_3_road":_num(d,"calc_3_road_games").eq(1),
    }
    # Allocate team-memory columns in one concat to avoid pandas fragmentation on
    # the wider V3.10 horizon state.  This is a performance-only change.
    _team_memory_cols={}
    for _ctx in _ctx_masks:
        _team_memory_cols[f"calc_team_{_ctx}_ats_n_prior"]=pd.Series(0.0,index=d.index,dtype=float)
        _team_memory_cols[f"calc_team_{_ctx}_ats_rate_prior"]=pd.Series(np.nan,index=d.index,dtype=float)
    if _team_memory_cols:
        d=pd.concat([d,pd.DataFrame(_team_memory_cols,index=d.index)],axis=1).copy()
    _ord=d.sort_values(["Team_Norm","Game_Date","Source_Name","Source_Game_ID"],kind="mergesort")
    for _tm,_idx0 in _ord.groupby("Team_Norm",sort=False,dropna=False).groups.items():
        _counts={k:0 for k in _ctx_masks}; _wins={k:0 for k in _ctx_masks}
        for _ri in list(_idx0):
            for _ctx in _ctx_masks:
                _n=_counts[_ctx]; d.at[_ri,f"calc_team_{_ctx}_ats_n_prior"]=float(_n)
                d.at[_ri,f"calc_team_{_ctx}_ats_rate_prior"]=(float(_wins[_ctx]/_n) if _n else np.nan)
            _st=str(d.at[_ri,"Opening_ATS_Status"])
            if _st in {"WIN","LOSS"}:
                for _ctx,_mask in _ctx_masks.items():
                    # Nullable boolean masks can contain pd.NA early in a team's
                    # history.  pd.NA has no truth value, so treat it as not matched
                    # rather than calling bool(pd.NA), which raises TypeError.
                    _flag=_mask.loc[_ri]
                    if pd.notna(_flag) and bool(_flag):
                        _counts[_ctx]+=1; _wins[_ctx]+=int(_st=="WIN")

    # Prior game total is computed explicitly below; avoid groupby.apply so the
    # module stays portable across the pandas versions used by Cloud Run images.
    d["calc_total_points_current"]=_num(d,"Team_Score")+_num(d,"Opponent_Score")
    d["calc_prev1_total_points"]=d.groupby(["Season","Team_Norm"],sort=False)["calc_total_points_current"].shift(1)
    d["calc_prev2_total_points"]=d.groupby(["Season","Team_Norm"],sort=False)["calc_total_points_current"].shift(2)
    d["calc_prev3_total_points"]=d.groupby(["Season","Team_Norm"],sort=False)["calc_total_points_current"].shift(3)
    d["calc_2_prior_totals_50_plus"]=(d["calc_prev1_total_points"].ge(50)&d["calc_prev2_total_points"].ge(50)).astype(float)
    d["calc_2_prior_totals_under_42"]=(d["calc_prev1_total_points"].lt(42)&d["calc_prev2_total_points"].lt(42)&d["calc_prev1_total_points"].notna()&d["calc_prev2_total_points"].notna()).astype(float)
    d["calc_total_improving_2"]=(d["calc_prev1_total_points"].gt(d["calc_prev2_total_points"])&(d["calc_prev1_total_points"]-d["calc_prev2_total_points"]).ge(7)).astype(float)
    d["calc_total_worsening_2"]=(d["calc_prev1_total_points"].lt(d["calc_prev2_total_points"])&(d["calc_prev2_total_points"]-d["calc_prev1_total_points"]).ge(7)).astype(float)
    d["calc_3_prior_totals_50_plus"]=(d["calc_prev1_total_points"].ge(50)&d["calc_prev2_total_points"].ge(50)&d["calc_prev3_total_points"].ge(50)).astype(float)
    d["calc_3_prior_totals_under_42"]=(d["calc_prev1_total_points"].lt(42)&d["calc_prev2_total_points"].lt(42)&d["calc_prev3_total_points"].lt(42)).astype(float)
    d["calc_b2b_open_ats_losses"]=(d["calc_prev1_open_ats_loss"].eq(1)&d["calc_prev2_open_ats_loss"].eq(1)).astype(float)

    # NCAAF-Miner-style prior streak state. Values are written BEFORE the current
    # result is consumed, so the current game's outcome never enters its atoms.
    d["calc_su_win_current"]=np.where(_num(d,"actual_margin").notna(),(_num(d,"actual_margin")>0).astype(float),np.nan)
    d["calc_open_ats_win_current"]=np.where(d["Opening_ATS_Status"].isin(["WIN","LOSS"]),d["Opening_ATS_Status"].eq("WIN").astype(float),np.nan)
    for c in ("calc_su_win_streak_prior","calc_su_loss_streak_prior","calc_ats_win_streak_prior","calc_ats_loss_streak_prior"):
        d[c]=0.0
    for _,ix0 in d.groupby(["Season","Team_Norm"],sort=False,dropna=False).groups.items():
        ix=np.asarray(list(ix0),dtype=int)
        su=pd.to_numeric(d.loc[ix,"calc_su_win_current"],errors="coerce").to_numpy(dtype=float)
        ats=pd.to_numeric(d.loc[ix,"calc_open_ats_win_current"],errors="coerce").to_numpy(dtype=float)
        sw=sl=aw=al=0
        for j,rowix in enumerate(ix):
            d.at[rowix,"calc_su_win_streak_prior"]=float(sw); d.at[rowix,"calc_su_loss_streak_prior"]=float(sl)
            d.at[rowix,"calc_ats_win_streak_prior"]=float(aw); d.at[rowix,"calc_ats_loss_streak_prior"]=float(al)
            if np.isfinite(su[j]):
                if su[j]>=.5: sw+=1; sl=0
                else: sl+=1; sw=0
            if np.isfinite(ats[j]):
                if ats[j]>=.5: aw+=1; al=0
                else: al+=1; aw=0

    # Prior H2H state from the current team's perspective. This is across seasons
    # and uses only completed earlier meetings for the same team/opponent pair.
    team=d["Team_Norm"].astype(str); opp=d["Opponent_Norm"].astype(str)
    d["calc_h2h_pair"]=np.where(team<=opp,team+"|"+opp,opp+"|"+team)
    d["calc_h2h_prior_meetings"]=0.0; d["calc_h2h_last_margin"]=np.nan
    d["calc_h2h_win_streak_prior"]=0.0; d["calc_h2h_loss_streak_prior"]=0.0
    for _,ix0 in d.groupby(["Team_Norm","calc_h2h_pair"],sort=False,dropna=False).groups.items():
        ix=np.asarray(sorted(list(ix0),key=lambda z:(pd.Timestamp(d.at[z,"Game_Date"]),str(d.at[z,"physical_game_id"]))),dtype=int)
        wins=losses=0; last_margin=np.nan
        for j,rowix in enumerate(ix):
            d.at[rowix,"calc_h2h_prior_meetings"]=float(j); d.at[rowix,"calc_h2h_last_margin"]=last_margin
            d.at[rowix,"calc_h2h_win_streak_prior"]=float(wins); d.at[rowix,"calc_h2h_loss_streak_prior"]=float(losses)
            m=float(pd.to_numeric(pd.Series([d.at[rowix,"actual_margin"]]),errors="coerce").iloc[0]) if pd.notna(d.at[rowix,"actual_margin"]) else np.nan
            if np.isfinite(m):
                last_margin=m
                if m>0: wins+=1; losses=0
                elif m<0: losses+=1; wins=0
                else: wins=losses=0

    _swap_cols=("calc_prev1_margin","calc_prev2_margin","calc_prev3_margin","calc_prev1_open_ats_loss","calc_prev2_open_ats_loss","calc_prev3_open_ats_loss","calc_prev1_open_ats_win","calc_prev2_open_ats_win","calc_prev3_open_ats_win","calc_prev1_open_ats_margin","calc_prev2_open_ats_margin","calc_prev3_open_ats_margin","calc_prev1_total_points","calc_prev2_total_points","calc_prev3_total_points","calc_3_prior_totals_50_plus","calc_3_prior_totals_under_42","calc_team_game_number","calc_prev1_opening_spread","calc_prev2_opening_spread","calc_prev3_opening_spread","calc_prev1_open_dog","calc_prev2_open_dog","calc_prev3_open_dog","calc_prev1_open_favorite","calc_prev2_open_favorite","calc_prev3_open_favorite","calc_open_dog_rate_prior","calc_first_home_game","calc_first_road_game","calc_prev1_home","calc_prev2_home","calc_prev3_home","calc_prev1_away","calc_prev2_away","calc_prev3_away","calc_prev1_points_for","calc_prev2_points_for","calc_prev3_points_for","calc_prev1_points_against","calc_prev2_points_against","calc_prev3_points_against","calc_prev1_division","calc_prev2_division","calc_prev3_division","calc_prev1_conference","calc_prev2_conference","calc_prev3_conference","calc_prev1_primetime","calc_prev2_primetime","calc_prev3_primetime","calc_prev1_opp_win_pct","calc_prev2_opp_win_pct","calc_prev3_opp_win_pct","calc_immediate_rematch","calc_island_game","calc_prev1_island","calc_prev2_island","calc_prev3_island","calc_b2b_su_wins","calc_b2b_su_losses","calc_b2b_home_games","calc_b2b_road_games","calc_b2b_home_su_wins","calc_b2b_road_su_wins","calc_b2b_scored_31_plus","calc_b2b_scored_28_plus","calc_b2b_allowed_17_or_less","calc_b2b_home_wins_31_plus","calc_b2b_division_games","calc_b2b_nondivision_games","calc_b2b_open_ats_losses","calc_b2b_open_ats_wins","calc_b2b_opp_ge_600","calc_b2b_opp_le_400","calc_margin_improving_2","calc_margin_worsening_2","calc_ats_margin_improving_2","calc_ats_margin_worsening_2","calc_scoring_improving_2","calc_scoring_worsening_2","calc_defense_improving_2","calc_defense_worsening_2","calc_opp_quality_rising_2","calc_opp_quality_falling_2","calc_2_prior_totals_50_plus","calc_2_prior_totals_under_42","calc_total_improving_2","calc_total_worsening_2","calc_3_su_wins","calc_3_su_losses","calc_3_ats_wins","calc_3_ats_losses","calc_3_home_games","calc_3_road_games","calc_3_scored_28_plus","calc_3_scored_31_plus","calc_3_allowed_17_or_less","calc_3_opp_ge_600","calc_3_opp_le_400","calc_2of3_opp_ge_600","calc_2of3_opp_le_400","calc_margin_improving_3","calc_margin_worsening_3","calc_ats_margin_improving_3","calc_ats_margin_worsening_3","calc_scoring_improving_3","calc_scoring_worsening_3","calc_defense_improving_3","calc_defense_worsening_3","calc_opp_quality_rising_3","calc_opp_quality_falling_3","calc_su_win_streak_prior","calc_su_loss_streak_prior","calc_ats_win_streak_prior","calc_ats_loss_streak_prior","calc_h2h_prior_meetings","calc_h2h_last_margin","calc_h2h_win_streak_prior","calc_h2h_loss_streak_prior","calc_team_primetime_ats_n_prior","calc_team_primetime_ats_rate_prior","calc_team_island_ats_n_prior","calc_team_island_ats_rate_prior","calc_team_division_ats_n_prior","calc_team_division_ats_rate_prior","calc_team_nondivision_ats_n_prior","calc_team_nondivision_ats_rate_prior","calc_team_conference_ats_n_prior","calc_team_conference_ats_rate_prior","calc_team_interconference_ats_n_prior","calc_team_interconference_ats_rate_prior","calc_team_home_favorite_ats_n_prior","calc_team_home_favorite_ats_rate_prior","calc_team_road_dog_ats_n_prior","calc_team_road_dog_ats_rate_prior","calc_team_after_3_su_wins_ats_n_prior","calc_team_after_3_su_wins_ats_rate_prior","calc_team_after_3_su_losses_ats_n_prior","calc_team_after_3_su_losses_ats_rate_prior","calc_team_after_3_ats_wins_ats_n_prior","calc_team_after_3_ats_wins_ats_rate_prior","calc_team_after_3_ats_losses_ats_n_prior","calc_team_after_3_ats_losses_ats_rate_prior","calc_team_after_blowout_win21_ats_n_prior","calc_team_after_blowout_win21_ats_rate_prior","calc_team_after_blowout_loss21_ats_n_prior","calc_team_after_blowout_loss21_ats_rate_prior","calc_team_after_close_game3_ats_n_prior","calc_team_after_close_game3_ats_rate_prior","calc_team_after_3_home_ats_n_prior","calc_team_after_3_home_ats_rate_prior","calc_team_after_3_road_ats_n_prior","calc_team_after_3_road_ats_rate_prior")
    _opp_cols={}
    for c in _swap_cols:
        _opp_cols["Opp_"+c]=_swap_within_game(pd.to_numeric(d[c],errors="coerce").astype("float64").to_numpy(),d)
    d=pd.concat([d,pd.DataFrame(_opp_cols,index=d.index)],axis=1)
    return d.sort_values(["Season","Game_Date","Source_Name","Source_Game_ID","Team_Norm"],kind="mergesort").reset_index(drop=True)


def _wilson(w,n,z=1.959963984540054):
    if n<=0:return [None,None]
    p=w/n; den=1+z*z/n; ctr=(p+z*z/(2*n))/den; half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return [max(0.0,ctr-half),min(1.0,ctr+half)]


def _rec(labels: np.ndarray, idx: np.ndarray, direction: str) -> dict:
    idx=np.asarray(idx,dtype=int)
    if len(idx)==0:return {"n":0,"wins":0,"rate":None,"wilson95":[None,None]}
    y=np.asarray(labels,float)[idx]; m=np.isfinite(y); y=y[m]
    if not len(y):return {"n":0,"wins":0,"rate":None,"wilson95":[None,None]}
    z=y if direction in ("PLAY_ON","OVER") else 1-y
    w=int(z.sum()); n=int(len(z)); ci=_wilson(w,n)
    return {"n":n,"wins":w,"rate":round(w/n,6),"wilson95":[round(ci[0],6),round(ci[1],6)]}


def _candidate_rows_side(state: pd.DataFrame,mask: np.ndarray) -> tuple[np.ndarray,int]:
    mm=np.asarray(mask,bool); tmp=pd.DataFrame({"i":np.arange(len(state)),"game":state.physical_game_id.astype(str),"m":mm.astype(int)})
    cnt=tmp.groupby("game",sort=False)["m"].transform("sum").to_numpy()
    return np.where(mm&(cnt==1))[0],int(np.sum(mm&(cnt>1)))


def _home_or_neutral_one_side(state: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for _,p in state.groupby("physical_game_id",sort=False):
        home=p.loc[_num(p,"Is_Home").eq(1)]
        if len(home)==1: rows.append(home.iloc[0])
        else: rows.append(p.sort_values("Team_Norm",kind="mergesort").iloc[0])
    out=pd.DataFrame(rows).reset_index(drop=True)
    if out.physical_game_id.duplicated().any():raise RuntimeError("NFL_SYSTEM_LAB_DUPLICATE_GAME_ORIENTATION")
    return out



def _expert_token(v):
    return re.sub(r"[^A-Za-z0-9]+","_",str(v or "")).strip("_").upper()


def _expert_occurrence_atom_bridge(state:pd.DataFrame,bigal_plays:pd.DataFrame,pathi_plays:pd.DataFrame,log_func=print):
    """Materialize exact Pathi/Big Al occurrence ledgers as Miner atoms.

    The occurrence frames are produced by the same builders that grade each expert
    system.  This function never re-derives an expert rule from columns.  A Miner
    atom is therefore a projection of an already-recorded (game, team, system)
    occurrence onto the exact side-state used by the Miner.

    Pathi mirror/nested systems share their normalized family category and Big Al
    nested variants share their existing independence family.  That prevents the
    beam grammar from combining correlated variants as if they were independent.
    """
    if not isinstance(state,pd.DataFrame) or state.empty:
        return [],{"status":"UNAVAILABLE_EMPTY_STATE","atom_count":0,"production_authority":0}
    d=state.copy()
    if "physical_game_id" not in d or "Team_Norm" not in d:
        raise RuntimeError("NFL_V311_EXPERT_BRIDGE_STATE_KEY_MISSING")
    side_keys=(d["physical_game_id"].astype(str)+"||"+d["Team_Norm"].astype(str))
    if side_keys.duplicated().any():
        raise RuntimeError("NFL_V311_EXPERT_BRIDGE_DUPLICATE_SIDE_KEY")
    key_to_i={k:i for i,k in enumerate(side_keys.tolist())}
    seasons=pd.to_numeric(d.get("Season"),errors="coerce").to_numpy()
    disc=np.isin(seasons,DISCOVERY_SEASONS)
    atoms=[]; manifest=[]; unmatched=[]
    row_pathi_families=[set() for _ in range(len(d))]
    row_bigal_families=[set() for _ in range(len(d))]
    source_counts={"PATHI":0,"BIGAL":0}; matched_counts={"PATHI":0,"BIGAL":0}

    def project(plays,source):
        if not isinstance(plays,pd.DataFrame) or plays.empty:return
        q=plays.copy()
        for c in ("system_id","physical_game_id","bet_team"):
            if c not in q: raise RuntimeError(f"NFL_V311_EXPERT_BRIDGE_{source}_{c.upper()}_MISSING")
        q["system_id"]=q["system_id"].astype(str)
        q["__key"]=q["physical_game_id"].astype(str)+"||"+q["bet_team"].astype(str)
        # One occurrence per system/side/game is the grading contract.
        before=len(q); q=q.drop_duplicates(["system_id","__key"],keep="first")
        dupes=before-len(q)
        for sid,g in q.groupby("system_id",sort=True):
            keys=set(g["__key"].astype(str))
            source_counts[source]+=len(keys)
            bad=sorted(k for k in keys if k not in key_to_i)
            if bad:
                unmatched.extend({"source":source,"system_id":sid,"side_key":k} for k in bad[:25])
            idx=sorted(key_to_i[k] for k in keys if k in key_to_i)
            matched_counts[source]+=len(idx)
            if not idx: continue
            mm=np.zeros(len(d),dtype=bool); mm[idx]=True
            if source=="PATHI":
                fam=str(PATHI_FAMILY_MAP.get(str(sid),str(sid)))
                name="EXPERT_PATHI_"+_expert_token(str(sid).replace("Pathi_FB_",""))
                category="EXPERT_PATHI_"+_expert_token(fam)
                for i in idx: row_pathi_families[i].add(fam)
            else:
                fam=str(_BIGAL_INDEPENDENCE_FAMILY.get(str(sid),str(sid)))
                name="EXPERT_BIGAL_"+_expert_token(sid)
                category="EXPERT_BIGAL_"+_expert_token(fam)
                for i in idx: row_bigal_families[i].add(fam)
            dn=int((mm&disc).sum())
            atoms.append({"name":name,"family":category,"mask":mm,"description":f"{source} occurrence {sid}","boundary":None,"team_specific":False,
                          "expert_source":source,"expert_system_id":str(sid),"expert_independence_family":fam,"expert_bridge":True,"production_authority":0})
            manifest.append({"source":source,"system_id":str(sid),"independence_family":fam,"atom":name,"category":category,
                             "occurrences":int(len(idx)),"discovery_occurrences":dn,"duplicate_occurrences_removed":int(dupes),"production_authority":0})

    project(pathi_plays,"PATHI"); project(bigal_plays,"BIGAL")
    if unmatched:
        raise RuntimeError("NFL_V311_EXPERT_BRIDGE_UNMATCHED_OCCURRENCES "+json.dumps(unmatched[:10],sort_keys=True))

    # Aggregate confluence atoms count distinct normalized expert families, not
    # raw nested/mirror systems. They are single atoms and therefore cannot create
    # independent-vote inflation by themselves.
    pc=np.array([len(x) for x in row_pathi_families],dtype=int)
    bc=np.array([len(x) for x in row_bigal_families],dtype=int)
    aggregate_specs=(
        ("EXPERT_PATHI_MULTI_2PLUS","EXPERT_PATHI_CONFLUENCE",pc>=2,"PATHI","2+ normalized Pathi families"),
        ("EXPERT_PATHI_MULTI_3PLUS","EXPERT_PATHI_CONFLUENCE3",pc>=3,"PATHI","3+ normalized Pathi families"),
        ("EXPERT_BIGAL_MULTI_2PLUS","EXPERT_BIGAL_CONFLUENCE",bc>=2,"BIGAL","2+ independent Big Al families"),
        ("EXPERT_PATHI_AND_BIGAL","EXPERT_CROSS_SOURCE_CONFLUENCE",(pc>=1)&(bc>=1),"CROSS_SOURCE","Pathi and Big Al agree on the same side"),
    )
    for name,fam,mm,source,desc in aggregate_specs:
        mm=np.asarray(mm,bool); n=int(mm.sum())
        if n:
            atoms.append({"name":name,"family":fam,"mask":mm,"description":desc,"boundary":None,"team_specific":False,
                          "expert_source":source,"expert_system_id":None,"expert_independence_family":fam,"expert_bridge":True,"production_authority":0})
            manifest.append({"source":source,"system_id":None,"independence_family":fam,"atom":name,"category":fam,
                             "occurrences":n,"discovery_occurrences":int((mm&disc).sum()),"aggregate":True,"production_authority":0})
    bridge={
        "status":"PASS","contract":"EXACT_GRADED_OCCURRENCE_LEDGER_TO_MINER_SIDE_STATE_NO_RULE_RECONSTRUCTION",
        "state_rows":int(len(d)),"atom_count":int(len(atoms)),"pathi_atom_count":sum(1 for a in atoms if a.get("expert_source")=="PATHI"),
        "bigal_atom_count":sum(1 for a in atoms if a.get("expert_source")=="BIGAL"),
        "cross_source_atom_count":sum(1 for a in atoms if a.get("expert_source")=="CROSS_SOURCE"),
        "pathi_occurrences":int(source_counts["PATHI"]),"pathi_matched":int(matched_counts["PATHI"]),
        "bigal_occurrences":int(source_counts["BIGAL"]),"bigal_matched":int(matched_counts["BIGAL"]),
        "pathi_sides_with_2plus_families":int((pc>=2).sum()),"pathi_sides_with_3plus_families":int((pc>=3).sum()),
        "bigal_sides_with_2plus_families":int((bc>=2).sum()),"cross_source_sides":int(((pc>=1)&(bc>=1)).sum()),
        "manifest":manifest,"production_authority":0,"automatic_promotion":False,"year_2026_queried":False,
    }
    log_func("[NFL-SYSTEM-V311-EXPERT-OCCURRENCE-BRIDGE] "+json.dumps({k:v for k,v in bridge.items() if k!="manifest"},sort_keys=True,default=str))
    return atoms,bridge



# -------------------------- Prediction Tracker external families --------------------------
# V3.12.0 applies the NCAAF lessons without importing NCAAF code or data:
#   * exact source-native headers only
#   * sparse/missing source is allowed
#   * legacy Miner lane is insulated from the external predictor universe
#   * one correlated external family per market
#   * source-cluster-balanced consensus
#   * predictor behavior learned on DISCOVERY only, validated later
#   * 2026 is never queried by Heavy Research
#   * no automatic promotion / zero production authority

PT_RAW_PREFIX = "research/nfl/external/prediction_tracker/raw"
PT_HISTORY_SEASONS = tuple(range(2017, 2026))
PT_SPREAD_EXTERNAL_FAMILY = "NFL_SPREAD_EXTERNAL_RATINGS_FAMILY"
PT_TOTAL_EXTERNAL_FAMILY = "NFL_TOTALS_EXTERNAL_RATINGS_FAMILY"
PT_MIN_CLUSTER_CONSENSUS = 8
PT_SPREAD_EDGE_THRESHOLD = 2.0
PT_TOTAL_EDGE_THRESHOLD = 2.0
PT_SPREAD_SEED_CAP = 48
PT_TOTAL_SEED_CAP = 24
PT_SUPPORT_CAP = 90
PT_EXTERNAL_BEAM_WIDTH = 48
PT_EXTERNAL_MAX_DEPTH = 3

PT_SPREAD_EXCLUDE = {
    "line","lineopen","lineavg","linemed","linemedian","linestd","linemidweek",
    "phcover","phwin","hscore","rscore","vscore","actual","total","week","date",
}
PT_TOTAL_EXCLUDE = {
    "totavg","totmed","totmedian","totstd",
}

# The only multi-header provider clusters we collapse are relationships evident
# from the source-native header names themselves. Every other predictor is its own
# cluster. This is intentionally conservative.
PT_SPREAD_SOURCE_CLUSTERS = {
    "SAGARIN": {"linesag","linesaggm","linesagr","linesagp"},
    "PI_RATINGS": {"linepi","linepim","linepib"},
    "REGRESSION_VARIANTS": {"linel1","linel2","linel2h","linel2to","linelog"},
}

_PT_TEAM_ALIASES = {
    "ARI": ("ari","arizona","arizona cardinals","cardinals"),
    "ATL": ("atl","atlanta","atlanta falcons","falcons"),
    "BAL": ("bal","baltimore","baltimore ravens","ravens"),
    "BUF": ("buf","buffalo","buffalo bills","bills"),
    "CAR": ("car","carolina","carolina panthers","panthers"),
    "CHI": ("chi","chicago","chicago bears","bears"),
    "CIN": ("cin","cincinnati","cincinnati bengals","bengals"),
    "CLE": ("cle","cleveland","cleveland browns","browns"),
    "DAL": ("dal","dallas","dallas cowboys","cowboys"),
    "DEN": ("den","denver","denver broncos","broncos"),
    "DET": ("det","detroit","detroit lions","lions"),
    "GB": ("gb","green bay","green bay packers","packers"),
    "HOU": ("hou","houston","houston texans","texans"),
    "IND": ("ind","indianapolis","indianapolis colts","colts"),
    "JAX": ("jax","jac","jacksonville","jacksonville jaguars","jaguars"),
    "KC": ("kc","kansas city","kansas city chiefs","chiefs"),
    "LV": ("lv","las vegas","las vegas raiders","oakland","oakland raiders","raiders"),
    "LAC": ("lac","la chargers","los angeles chargers","san diego","san diego chargers","chargers"),
    "LAR": ("lar","la rams","los angeles rams","st louis","st louis rams","rams"),
    "MIA": ("mia","miami","miami dolphins","dolphins"),
    "MIN": ("min","minnesota","minnesota vikings","vikings"),
    "NE": ("ne","new england","new england patriots","patriots"),
    "NO": ("no","new orleans","new orleans saints","saints"),
    "NYG": ("nyg","new york giants","giants"),
    "NYJ": ("nyj","new york jets","jets"),
    "PHI": ("phi","philadelphia","philadelphia eagles","eagles"),
    "PIT": ("pit","pittsburgh","pittsburgh steelers","steelers"),
    "SEA": ("sea","seattle","seattle seahawks","seahawks"),
    "SF": ("sf","sfo","san francisco","san francisco 49ers","49ers"),
    "TB": ("tb","tampa bay","tampa bay buccaneers","buccaneers","bucs"),
    "TEN": ("ten","tennessee","tennessee titans","titans"),
    "WAS": ("was","wsh","washington","washington commanders","washington football team","washington redskins","commanders"),
}


def _pt_norm_text(v):
    s=str(v or "").lower().strip()
    s=s.replace("&"," and ")
    s=re.sub(r"[^a-z0-9]+"," ",s)
    return re.sub(r"\s+"," ",s).strip()


_PT_ALIAS_TO_CODE={}
for _code,_aliases in _PT_TEAM_ALIASES.items():
    for _a in _aliases:
        _PT_ALIAS_TO_CODE[_pt_norm_text(_a)] = _code


def _pt_team_code(v):
    s=_pt_norm_text(v)
    if not s:
        return ""
    if s in _PT_ALIAS_TO_CODE:
        return _PT_ALIAS_TO_CODE[s]
    # Exact tokenized suffix/prefix match only; no fuzzy matching.
    hits=[]
    for alias,code in _PT_ALIAS_TO_CODE.items():
        if len(alias)>=4 and (s==alias or s.startswith(alias+" ") or s.endswith(" "+alias)):
            hits.append((len(alias),code))
    if not hits:
        return ""
    hits.sort(reverse=True)
    if len(hits)>1 and hits[0][0]==hits[1][0] and hits[0][1]!=hits[1][1]:
        return ""
    return hits[0][1]


def _pt_find_col(df, *names):
    lower={str(c).strip().lower():c for c in df.columns}
    for n in names:
        if str(n).lower() in lower:
            return lower[str(n).lower()]
    return None


def _pt_spread_predictor_cols(df):
    out=[]
    for c in df.columns:
        k=str(c).strip().lower()
        if k.startswith("line") and k not in PT_SPREAD_EXCLUDE:
            out.append((k,c))
    return out


def _pt_total_predictor_cols(df):
    out=[]
    for c in df.columns:
        k=str(c).strip().lower()
        if k.startswith("tot") and k not in PT_TOTAL_EXCLUDE:
            out.append((k,c))
    return out


def _pt_source_cluster(canon, market):
    c=str(canon).lower()
    if str(market).upper()=="SPREADS":
        for fam,members in PT_SPREAD_SOURCE_CLUSTERS.items():
            if c in members:
                return fam
    return c.upper()


def _pt_read_gcs_history(storage_client,bucket_name,market,log_func=print):
    market=str(market).upper()
    frames=[]; diagnostics=[]; predictor_union=set()
    challenge=(b"cloudflare",b"just a moment",b"cf-chl",b"checking your browser",b"enable javascript")
    for sy in PT_HISTORY_SEASONS:
        name=(f"nfl{str(sy)[-2:]}.csv" if market=="SPREADS" else f"nfltotals{str(sy)[-2:]}.csv")
        path=f"{PT_RAW_PREFIX}/{name}"
        rec={"season":sy,"file":name,"gcs_path":f"gs://{bucket_name}/{path}","status":"MISSING"}
        try:
            blob=storage_client.bucket(bucket_name).blob(path)
            if not blob.exists():
                diagnostics.append(rec); continue
            raw=blob.download_as_bytes()
            rec["bytes"]=len(raw)
            if len(raw)<500 or any(x in raw.lower() for x in challenge):
                rec["status"]="INVALID_PAYLOAD"; diagnostics.append(rec); continue
            q=pd.read_csv(io.BytesIO(raw))
            home=_pt_find_col(q,"home")
            road=_pt_find_col(q,"road","away","visitor")
            week=_pt_find_col(q,"week")
            line=_pt_find_col(q,"line")
            if not home or not road or not week or not line:
                rec["status"]="INVALID_SCHEMA"; rec["missing_required"]=[x for x,v in (("home",home),("road",road),("week",week),("line",line)) if not v]
                diagnostics.append(rec); continue
            preds=_pt_spread_predictor_cols(q) if market=="SPREADS" else _pt_total_predictor_cols(q)
            if len(preds)<2:
                rec["status"]="NO_PREDICTORS"; diagnostics.append(rec); continue
            z=pd.DataFrame(index=q.index)
            z["Season"]=int(sy)
            z["Week"]=pd.to_numeric(q[week],errors="coerce")
            z["Home_Raw"]=q[home].astype(str)
            z["Road_Raw"]=q[road].astype(str)
            z["Home_Code"]=z["Home_Raw"].map(_pt_team_code)
            z["Road_Code"]=z["Road_Raw"].map(_pt_team_code)
            z["PT_Line"]=pd.to_numeric(q[line],errors="coerce")
            open_col=_pt_find_col(q,"lineopen")
            z["PT_Open"]=pd.to_numeric(q[open_col],errors="coerce") if open_col else np.nan
            for canon,orig in preds:
                z[f"PRED__{canon}"]=pd.to_numeric(q[orig],errors="coerce")
                predictor_union.add(canon)
            # Source outcomes/probability labels are intentionally not copied.
            good=z["Home_Code"].ne("")&z["Road_Code"].ne("")&z["Week"].notna()
            z=z.loc[good].copy()
            rec.update(status="PASS",rows=int(len(z)),predictor_headers=len(preds),mapped_teams=int(good.sum()))
            frames.append(z); diagnostics.append(rec)
        except Exception as exc:
            rec["status"]="ERROR"; rec["error"]=f"{type(exc).__name__}:{exc}"; diagnostics.append(rec)
    hist=pd.concat(frames,ignore_index=True,sort=False) if frames else pd.DataFrame()
    status="PASS" if not hist.empty else "NO_HISTORY_AVAILABLE"
    log_func("[NFL-PT-LOAD] "+json.dumps({
        "status":status,"market":market,"rows":int(len(hist)),
        "seasons_loaded":[d["season"] for d in diagnostics if d.get("status")=="PASS"],
        "seasons_missing":[d["season"] for d in diagnostics if d.get("status")=="MISSING"],
        "predictor_headers":len(predictor_union),"gcs_prefix":f"gs://{bucket_name}/{PT_RAW_PREFIX}/",
        "outcome_columns_used_as_features":False,"production_authority":0,"year_2026_queried":False,
    },sort_keys=True,default=str))
    return hist,{"status":status,"market":market,"rows":int(len(hist)),"predictors":sorted(predictor_union),"files":diagnostics,"production_authority":0}


def _pt_key(season,week,home,road):
    try:
        sy=int(float(season)); wk=int(float(week))
    except Exception:
        return ""
    h=_pt_team_code(home); r=_pt_team_code(road)
    return f"{sy}|{wk}|{h}|{r}" if h and r else ""


def _pt_build_match_map(state,pt):
    if state.empty or pt.empty:
        return {},{"matched_games":0,"internal_games":0,"pt_rows":int(len(pt)),"duplicate_pt_keys":0}
    home=state.loc[_num(state,"Is_Home").eq(1)].copy()
    if home.empty:
        return {},{"matched_games":0,"internal_games":0,"pt_rows":int(len(pt)),"duplicate_pt_keys":0}
    home["__pt_key"]=[_pt_key(s,w,t,o) for s,w,t,o in zip(home["Season"],home["Week_Number"],home["Team_Norm"],home["Opponent_Norm"])]
    pp=pt.copy()
    pp["__pt_key"]=[_pt_key(s,w,h,r) for s,w,h,r in zip(pp["Season"],pp["Week"],pp["Home_Code"],pp["Road_Code"])]
    vc=pp["__pt_key"].value_counts()
    duplicate=set(vc[vc.gt(1)].index)
    unique=pp.loc[pp["__pt_key"].ne("")&~pp["__pt_key"].isin(duplicate)].drop_duplicates("__pt_key",keep=False).set_index("__pt_key",drop=False)
    out={}
    exact_matched=0
    for _,r in home.iterrows():
        k=str(r["__pt_key"])
        if k and k in unique.index:
            out[str(r["physical_game_id"])]=unique.loc[k]; exact_matched+=1

    # Safe fallback for source/week-number drift: use season+home+road only when
    # that pair is unique in BOTH the internal schedule and the PT season. This
    # never guesses through a regular-season/postseason rematch.
    home["__pt_pair"]=[
        f"{int(float(s))}|{_pt_team_code(t)}|{_pt_team_code(o)}"
        for s,t,o in zip(home["Season"],home["Team_Norm"],home["Opponent_Norm"])
    ]
    pp["__pt_pair"]=[
        f"{int(float(s))}|{_pt_team_code(h)}|{_pt_team_code(r)}"
        for s,h,r in zip(pp["Season"],pp["Home_Code"],pp["Road_Code"])
    ]
    int_pair_count=home["__pt_pair"].value_counts()
    pt_pair_count=pp["__pt_pair"].value_counts()
    pt_pair_unique=pp.loc[pp["__pt_pair"].map(pt_pair_count).eq(1)].set_index("__pt_pair",drop=False)
    fallback=0
    for _,r in home.iterrows():
        gid=str(r["physical_game_id"])
        if gid in out: continue
        pk=str(r["__pt_pair"])
        if pk and int(int_pair_count.get(pk,0))==1 and pk in pt_pair_unique.index:
            out[gid]=pt_pair_unique.loc[pk]; fallback+=1

    return out,{
        "matched_games":int(len(out)),"exact_week_matches":int(exact_matched),"unique_pair_fallback_matches":int(fallback),
        "internal_games":int(home["physical_game_id"].nunique()),
        "pt_rows":int(len(pt)),"duplicate_pt_keys":int(len(duplicate)),
        "coverage":float(len(out)/max(home["physical_game_id"].nunique(),1)),
    }


def _pt_cluster_matrix(pt_row,predictors,market):
    groups={}
    for p in predictors:
        v=pd.to_numeric(pd.Series([pt_row.get(f"PRED__{p}",np.nan)]),errors="coerce").iloc[0]
        if pd.isna(v):
            continue
        groups.setdefault(_pt_source_cluster(p,market),[]).append(float(v))
    vals={}
    for g,x in groups.items():
        if x:
            vals[g]=float(np.median(np.asarray(x,float)))
    return vals


def _pt_attach_spread(state,pt,log_func=print):
    d=state.copy()
    predictors=sorted({c[len("PRED__"):] for c in pt.columns if str(c).startswith("PRED__")})
    match,mdiag=_pt_build_match_map(d,pt)
    n=len(d)
    fair={p:np.full(n,np.nan) for p in predictors}
    edge={p:np.full(n,np.nan) for p in predictors}
    consensus_edge=np.full(n,np.nan); agree=np.full(n,np.nan); dispersion=np.full(n,np.nan); cluster_count=np.zeros(n,float)
    matched=np.zeros(n,bool)
    for i,r in d.reset_index(drop=True).iterrows():
        pr=match.get(str(r["physical_game_id"]))
        if pr is None:
            continue
        matched[i]=True
        home=bool(float(pd.to_numeric(pd.Series([r.get("Is_Home")]),errors="coerce").iloc[0] or 0)==1)
        orient=1.0 if home else -1.0
        team_market=pd.to_numeric(pd.Series([r.get("Opening_Spread")]),errors="coerce").iloc[0]
        if pd.isna(team_market):
            continue
        for p in predictors:
            hv=pd.to_numeric(pd.Series([pr.get(f"PRED__{p}",np.nan)]),errors="coerce").iloc[0]
            if pd.notna(hv):
                tv=float(hv) if home else -float(hv)
                fair[p][i]=tv
                edge[p][i]=float(team_market)-tv
        cv=_pt_cluster_matrix(pr,predictors,"SPREADS")
        ce=[]
        for _,home_line in cv.items():
            tv=float(home_line) if home else -float(home_line)
            ce.append(float(team_market)-tv)
        if ce:
            arr=np.asarray(ce,float)
            consensus_edge[i]=float(np.median(arr))
            agree[i]=float(np.mean(arr>0))
            cluster_count[i]=float(len(arr))
            if len(arr)>=2:
                dispersion[i]=float(np.std(arr))
    d=d.reset_index(drop=True)
    for p in predictors:
        d[f"PTSP_FAIR__{p}"]=fair[p]
        d[f"PTSP_EDGE__{p}"]=edge[p]
    d["PTSP_CONSENSUS_EDGE"]=consensus_edge
    d["PTSP_AGREE_FRAC"]=agree
    d["PTSP_CLUSTER_STD"]=dispersion
    d["PTSP_CLUSTER_COUNT"]=cluster_count
    d["PTSP_MATCHED"]=matched.astype(float)
    log_func("[NFL-PT-MATCH] "+json.dumps({
        "market":"SPREADS",**mdiag,"side_rows":int(len(d)),"predictors":len(predictors),
        "rows_with_consensus":int(np.isfinite(consensus_edge).sum()),
        "production_authority":0,"year_2026_queried":False,
    },sort_keys=True,default=str))
    return d,predictors,mdiag


def _pt_attach_totals(games,pt,log_func=print):
    g=games.copy().reset_index(drop=True)
    predictors=sorted({c[len("PRED__"):] for c in pt.columns if str(c).startswith("PRED__")})
    match,mdiag=_pt_build_match_map(g,pt)
    n=len(g)
    pred={p:np.full(n,np.nan) for p in predictors}
    edge={p:np.full(n,np.nan) for p in predictors}
    consensus_edge=np.full(n,np.nan); over_frac=np.full(n,np.nan); under_frac=np.full(n,np.nan); dispersion=np.full(n,np.nan); cluster_count=np.zeros(n,float)
    matched=np.zeros(n,bool)
    for i,r in g.iterrows():
        pr=match.get(str(r["physical_game_id"]))
        if pr is None:
            continue
        matched[i]=True
        market=pd.to_numeric(pd.Series([r.get("Opening_Total")]),errors="coerce").iloc[0]
        if pd.isna(market):
            continue
        for p in predictors:
            v=pd.to_numeric(pd.Series([pr.get(f"PRED__{p}",np.nan)]),errors="coerce").iloc[0]
            if pd.notna(v):
                pred[p][i]=float(v); edge[p][i]=float(v)-float(market)
        cv=_pt_cluster_matrix(pr,predictors,"TOTALS")
        ce=[float(v)-float(market) for v in cv.values()]
        if ce:
            arr=np.asarray(ce,float)
            consensus_edge[i]=float(np.median(arr))
            over_frac[i]=float(np.mean(arr>0)); under_frac[i]=float(np.mean(arr<0))
            cluster_count[i]=float(len(arr))
            if len(arr)>=2: dispersion[i]=float(np.std(arr))
    for p in predictors:
        g[f"PTTOT_PRED__{p}"]=pred[p]; g[f"PTTOT_EDGE__{p}"]=edge[p]
    g["PTTOT_CONSENSUS_EDGE"]=consensus_edge
    g["PTTOT_OVER_FRAC"]=over_frac; g["PTTOT_UNDER_FRAC"]=under_frac
    g["PTTOT_CLUSTER_STD"]=dispersion; g["PTTOT_CLUSTER_COUNT"]=cluster_count
    g["PTTOT_MATCHED"]=matched.astype(float)
    log_func("[NFL-PT-MATCH] "+json.dumps({
        "market":"TOTALS",**mdiag,"game_rows":int(len(g)),"predictors":len(predictors),
        "rows_with_consensus":int(np.isfinite(consensus_edge).sum()),
        "production_authority":0,"year_2026_queried":False,
    },sort_keys=True,default=str))
    return g,predictors,mdiag


def _pt_behavior_record(seasons,follow_result,eligible):
    seasons=np.asarray(seasons,int); fr=np.asarray(follow_result,float); eligible=np.asarray(eligible,bool)&np.isfinite(fr)
    disc=np.isin(seasons,DISCOVERY_SEASONS)&eligible
    n=int(disc.sum())
    follow=float(np.mean(fr[disc])) if n else None
    behavior=("FOLLOW" if follow is not None and follow>=.5 else "FADE") if n else None
    def rec(mask):
        z=fr[mask&eligible]
        z=z[np.isfinite(z)]
        if not len(z): return {"n":0,"rate":None}
        if behavior=="FADE": z=1.0-z
        return {"n":int(len(z)),"rate":float(np.mean(z))}
    shadow=rec(seasons==SHADOW_SEASON); confirm=rec(seasons==CONFIRM_SEASON); final=rec(seasons==FINAL_CHECK_SEASON)
    valid=rec(np.isin(seasons,(SHADOW_SEASON,CONFIRM_SEASON,FINAL_CHECK_SEASON)))
    dr=rec(disc)
    blocks=[shadow,confirm,final]
    positive=sum(1 for x in blocks if x["n"]>=12 and x["rate"] is not None and x["rate"]>.5)
    confirmed=bool(n>=100 and valid["n"]>=50 and valid["rate"] is not None and valid["rate"]>BREAK_EVEN_REFERENCE and positive>=2)
    strong=bool(confirmed and valid["rate"]>=.54 and positive==3)
    score=(abs((follow if follow is not None else .5)-.5)*math.sqrt(max(n,1))) if n else 0.0
    return {
        "behavior":behavior,"discovery":dr,"shadow_2023":shadow,"confirmation_2024":confirm,"final_2025":final,
        "validation_2023_2025":valid,"positive_validation_seasons":int(positive),
        "confirmed":confirmed,"strong_confirmed":strong,"discovery_selection_score":float(score),
    }


def _pt_predictor_behavior_spread(state,predictors):
    home=state.loc[_num(state,"Is_Home").eq(1)].copy().reset_index(drop=True)
    seasons=pd.to_numeric(home["Season"],errors="coerce").fillna(-1).astype(int).to_numpy()
    y=np.where(home["Opening_ATS_Status"].eq("WIN"),1.0,np.where(home["Opening_ATS_Status"].eq("LOSS"),0.0,np.nan))
    rows=[]
    for p in predictors:
        e=pd.to_numeric(home.get(f"PTSP_EDGE__{p}"),errors="coerce").to_numpy(float)
        eligible=np.isfinite(e)&(np.abs(e)>=PT_SPREAD_EDGE_THRESHOLD)&np.isfinite(y)
        follow=np.full(len(home),np.nan)
        follow[eligible&(e>0)]=y[eligible&(e>0)]
        follow[eligible&(e<0)]=1.0-y[eligible&(e<0)]
        rec=_pt_behavior_record(seasons,follow,eligible)
        band={}
        for name,lo,hi in (("2_TO_LT3",2,3),("3_TO_LT5",3,5),("5_PLUS",5,1e9)):
            mm=eligible&(np.abs(e)>=lo)&(np.abs(e)<hi)
            z=follow[mm]; z=z[np.isfinite(z)]
            if rec["behavior"]=="FADE": z=1-z
            band[name]={"n":int(len(z)),"rate":float(np.mean(z)) if len(z) else None}
        rows.append({"predictor":p,"source_cluster":_pt_source_cluster(p,"SPREADS"),**rec,"edge_bands":band})
    return rows


def _pt_predictor_behavior_totals(games,predictors):
    seasons=pd.to_numeric(games["Season"],errors="coerce").fillna(-1).astype(int).to_numpy()
    y=np.where(games["Opening_Total_Status"].eq("OVER"),1.0,np.where(games["Opening_Total_Status"].eq("UNDER"),0.0,np.nan))
    rows=[]
    for p in predictors:
        e=pd.to_numeric(games.get(f"PTTOT_EDGE__{p}"),errors="coerce").to_numpy(float)
        eligible=np.isfinite(e)&(np.abs(e)>=PT_TOTAL_EDGE_THRESHOLD)&np.isfinite(y)
        follow=np.full(len(games),np.nan)
        follow[eligible&(e>0)]=y[eligible&(e>0)]
        follow[eligible&(e<0)]=1.0-y[eligible&(e<0)]
        rec=_pt_behavior_record(seasons,follow,eligible)
        band={}
        for name,lo,hi in (("2_TO_LT3",2,3),("3_TO_LT5",3,5),("5_PLUS",5,1e9)):
            mm=eligible&(np.abs(e)>=lo)&(np.abs(e)<hi)
            z=follow[mm]; z=z[np.isfinite(z)]
            if rec["behavior"]=="FADE": z=1-z
            band[name]={"n":int(len(z)),"rate":float(np.mean(z)) if len(z) else None}
        rows.append({"predictor":p,"source_cluster":_pt_source_cluster(p,"TOTALS"),**rec,"edge_bands":band})
    return rows


def _pt_top_discovery_predictors(behavior,cap):
    # Selection uses DISCOVERY only. Later validation fields are reporting only.
    q=[x for x in behavior if int((x.get("discovery") or {}).get("n") or 0)>=100]
    q=sorted(q,key=lambda x:(float(x.get("discovery_selection_score") or 0),int((x.get("discovery") or {}).get("n") or 0)),reverse=True)
    return [x["predictor"] for x in q[:int(cap)]]


def _pt_external_spread_atoms(state,predictors,behavior):
    atoms=[]
    seeds=set(_pt_top_discovery_predictors(behavior,PT_SPREAD_SEED_CAP))
    for p in predictors:
        if p not in seeds: continue
        e=pd.to_numeric(state.get(f"PTSP_EDGE__{p}"),errors="coerce")
        mm=e.ge(PT_SPREAD_EDGE_THRESHOLD).fillna(False).to_numpy(bool)
        if int(mm.sum()):
            atoms.append({"name":"PTSP_"+_expert_token(p)+"_RECOMMENDS_SIDE_2PLUS","family":PT_SPREAD_EXTERNAL_FAMILY,"mask":mm,
                          "description":f"Prediction Tracker {p} recommends this side by >=2","boundary":("ptspread",p,2.0),
                          "team_specific":False,"pt_predictor":p,"pt_source_cluster":_pt_source_cluster(p,"SPREADS"),
                          "external_family":True,"production_authority":0})
    ce=pd.to_numeric(state.get("PTSP_CONSENSUS_EDGE"),errors="coerce")
    ag=pd.to_numeric(state.get("PTSP_AGREE_FRAC"),errors="coerce")
    sd=pd.to_numeric(state.get("PTSP_CLUSTER_STD"),errors="coerce")
    cn=pd.to_numeric(state.get("PTSP_CLUSTER_COUNT"),errors="coerce")
    specs=[
        ("PTSP_CLUSTER_CONSENSUS_EDGE_2PLUS",cn.ge(PT_MIN_CLUSTER_CONSENSUS)&ce.ge(2), "cluster-balanced consensus recommends side by >=2"),
        ("PTSP_CLUSTER_CONSENSUS_EDGE_3PLUS",cn.ge(PT_MIN_CLUSTER_CONSENSUS)&ce.ge(3), "cluster-balanced consensus recommends side by >=3"),
        ("PTSP_CLUSTER_CONSENSUS_70_EDGE2",cn.ge(PT_MIN_CLUSTER_CONSENSUS)&ce.ge(2)&ag.ge(.70), ">=70% source clusters agree and median edge >=2"),
        ("PTSP_CLUSTER_TIGHT_70_EDGE2",cn.ge(PT_MIN_CLUSTER_CONSENSUS)&ce.ge(2)&ag.ge(.70)&sd.le(4), ">=70% source clusters agree, edge >=2, dispersion <=4"),
    ]
    for name,mask,desc in specs:
        mm=pd.Series(mask,index=state.index).fillna(False).to_numpy(bool)
        if int(mm.sum()):
            atoms.append({"name":name,"family":PT_SPREAD_EXTERNAL_FAMILY,"mask":mm,"description":desc,"boundary":None,
                          "team_specific":False,"external_family":True,"pt_consensus":True,"production_authority":0})
    return atoms


def _pt_external_total_atoms(games,predictors,behavior):
    atoms=[]
    seeds=set(_pt_top_discovery_predictors(behavior,PT_TOTAL_SEED_CAP))
    for p in predictors:
        if p not in seeds: continue
        e=pd.to_numeric(games.get(f"PTTOT_EDGE__{p}"),errors="coerce")
        for side,mask in (("OVER",e.ge(PT_TOTAL_EDGE_THRESHOLD)),("UNDER",e.le(-PT_TOTAL_EDGE_THRESHOLD))):
            mm=pd.Series(mask,index=games.index).fillna(False).to_numpy(bool)
            if int(mm.sum()):
                atoms.append({"name":"PTTOT_"+_expert_token(p)+f"_SAYS_{side}_2PLUS","family":PT_TOTAL_EXTERNAL_FAMILY,"mask":mm,
                              "description":f"Prediction Tracker {p} projects {side} by >=2","boundary":("pttotal",p,side,2.0),
                              "team_specific":False,"pt_predictor":p,"pt_source_cluster":_pt_source_cluster(p,"TOTALS"),
                              "external_family":True,"production_authority":0})
    ce=pd.to_numeric(games.get("PTTOT_CONSENSUS_EDGE"),errors="coerce")
    of=pd.to_numeric(games.get("PTTOT_OVER_FRAC"),errors="coerce"); uf=pd.to_numeric(games.get("PTTOT_UNDER_FRAC"),errors="coerce")
    sd=pd.to_numeric(games.get("PTTOT_CLUSTER_STD"),errors="coerce"); cn=pd.to_numeric(games.get("PTTOT_CLUSTER_COUNT"),errors="coerce")
    specs=[
        ("PTTOT_CLUSTER_CONSENSUS_OVER_2PLUS",cn.ge(PT_MIN_CLUSTER_CONSENSUS)&ce.ge(2),"cluster-balanced consensus projects OVER by >=2"),
        ("PTTOT_CLUSTER_CONSENSUS_UNDER_2PLUS",cn.ge(PT_MIN_CLUSTER_CONSENSUS)&ce.le(-2),"cluster-balanced consensus projects UNDER by >=2"),
        ("PTTOT_CLUSTER_70_OVER_EDGE2",cn.ge(PT_MIN_CLUSTER_CONSENSUS)&ce.ge(2)&of.ge(.70),">=70% clusters project OVER and median edge >=2"),
        ("PTTOT_CLUSTER_70_UNDER_EDGE2",cn.ge(PT_MIN_CLUSTER_CONSENSUS)&ce.le(-2)&uf.ge(.70),">=70% clusters project UNDER and median edge >=2"),
        ("PTTOT_CLUSTER_TIGHT_OVER_EDGE2",cn.ge(PT_MIN_CLUSTER_CONSENSUS)&ce.ge(2)&of.ge(.70)&sd.le(4),"tight OVER cluster consensus"),
        ("PTTOT_CLUSTER_TIGHT_UNDER_EDGE2",cn.ge(PT_MIN_CLUSTER_CONSENSUS)&ce.le(-2)&uf.ge(.70)&sd.le(4),"tight UNDER cluster consensus"),
    ]
    for name,mask,desc in specs:
        mm=pd.Series(mask,index=games.index).fillna(False).to_numpy(bool)
        if int(mm.sum()):
            atoms.append({"name":name,"family":PT_TOTAL_EXTERNAL_FAMILY,"mask":mm,"description":desc,"boundary":None,
                          "team_specific":False,"external_family":True,"pt_consensus":True,"production_authority":0})
    return atoms


def _pt_select_support_atoms(atoms,rows,cap=PT_SUPPORT_CAP):
    seasons=pd.to_numeric(rows["Season"],errors="coerce").fillna(-1).astype(int).to_numpy()
    disc=np.isin(seasons,DISCOVERY_SEASONS)
    scored=[]
    for i,a in enumerate(atoms):
        if str(a.get("family")) in {PT_SPREAD_EXTERNAL_FAMILY,PT_TOTAL_EXTERNAL_FAMILY}: continue
        n=int((np.asarray(a.get("mask"),bool)&disc).sum())
        expert=1 if a.get("expert_bridge") else 0
        if n>0:
            scored.append((expert,n,-i,a))
    scored=sorted(scored,reverse=True,key=lambda x:(x[0],x[1],x[2]))
    # Preserve exact atom objects/masks; this only limits the external lane's
    # support grammar and never changes the legacy lane.
    return [x[3] for x in scored[:int(cap)]]


def _pt_external_authority_eligible(rule):
    """Source-neutral system gate for PT-derived Miner rules.

    PT does not receive model weight.  A *specific derived system* may, however,
    earn the same bounded confirmation/conflict privilege as an ordinary Miner
    system when its <=2025 validation evidence clears the existing system gate.
    2026 is never consulted here.
    """
    if not isinstance(rule,dict) or int(rule.get("ambiguous_both_sides") or 0)>0:
        return False
    vr=((rule.get("records") or {}).get("validation_2023_2025") or {})
    try:
        n=int(vr.get("n") or 0); rate=float(vr.get("rate"))
    except Exception:
        return False
    status=str(rule.get("status") or "")
    state=str(rule.get("current_evidence_state") or "")
    evidence_ok=(
        status in {"LEGIT_CANDIDATE_REQUIRES_PROSPECTIVE","PROMISING_HISTORICAL_SYSTEM"}
        or state in {"CURRENT_EDGE_VALIDATED","VALIDATED_BUT_WEAKENED","CURRENT_EDGE_POSITIVE","POSITIVE_BUT_BELOW_MINUS110_BREAK_EVEN"}
    )
    return bool(evidence_ok and n>=24 and rate>BREAK_EVEN_REFERENCE)


def _pt_external_mechanisms(miner,market):
    out=[]
    family=PT_SPREAD_EXTERNAL_FAMILY if str(market).upper()=="SPREADS" else PT_TOTAL_EXTERNAL_FAMILY
    for r in miner.get("retained_rules") or []:
        cond=list(r.get("conditions") or [])
        payload=f"{market}|{r.get('direction')}|{'|'.join(cond)}"
        mid="NFL_PT_"+hashlib.sha1(payload.encode()).hexdigest()[:10].upper()
        qualified=_pt_external_authority_eligible(r)
        out.append({
            "external_mechanism_id":mid,"system_id":mid,"source":"PT","market":market,"direction":r.get("direction"),
            "conditions":cond,"representative_conditions":cond,"records":r.get("records") or {},"status":r.get("status"),
            "current_evidence_state":r.get("current_evidence_state"),
            "evidence_family_key":family,"family_id":family,"independence_family":family,
            "qualified_current":qualified,"normalized_vote_eligible":qualified,
            "correlation_policy":"ONE_CORRELATED_PREDICTION_TRACKER_EXTERNAL_FAMILY_WITHIN_EACH_MARKET",
            "prospective_eligible":qualified,
            "introduction_policy":"SOURCE_NEUTRAL_SYSTEM_GATE__PT_MODEL_WEIGHT_REMAINS_ZERO",
            "overlay_authority":"BOUNDED_SYSTEM_CONFIRM_OR_CONFLICT" if qualified else "NONE",
            "production_authority":0,
        })
    return out


def _pt_run_external_research(*,state,games,spread_label,total_label,legacy_spread_atoms,legacy_total_atoms,storage_client,bucket_name,log_func=print):
    spread_raw,spread_load=_pt_read_gcs_history(storage_client,bucket_name,"SPREADS",log_func=log_func)
    total_raw,total_load=_pt_read_gcs_history(storage_client,bucket_name,"TOTALS",log_func=log_func)
    if spread_raw.empty and total_raw.empty:
        report={
            "status":"NO_HISTORY_AVAILABLE","spread_load":spread_load,"total_load":total_load,
            "contract":"OPTIONAL_SPARSE_EXTERNAL_SOURCE__LEGACY_MINER_UNCHANGED_IF_ABSENT",
            "production_authority":0,"automatic_promotion":False,"year_2026_queried":False,
        }
        log_func("[NFL-PT-CONTRACT] "+json.dumps(report,sort_keys=True,default=str))
        return report

    spread_state=state.copy(); spread_predictors=[]; spread_match={}
    spread_behavior=[]; spread_atoms=[]; spread_lane={"status":"NO_SPREAD_HISTORY","retained_rules":[]}
    if not spread_raw.empty:
        spread_state,spread_predictors,spread_match=_pt_attach_spread(state,spread_raw,log_func=log_func)
        spread_behavior=_pt_predictor_behavior_spread(spread_state,spread_predictors)
        spread_atoms=_pt_external_spread_atoms(spread_state,spread_predictors,spread_behavior)
        supports=_pt_select_support_atoms(legacy_spread_atoms,spread_state)
        combo=spread_atoms+supports
        spread_lane=_search_market(
            spread_state,spread_label,combo,"SPREADS",True,log_func=log_func,
            seed_indices=list(range(len(spread_atoms))),
            extension_indices=list(range(len(spread_atoms),len(combo))),
            max_depth_override=PT_EXTERNAL_MAX_DEPTH,
            beam_width_override=PT_EXTERNAL_BEAM_WIDTH,
            lane="PREDICTION_TRACKER_EXTERNAL",
        )
        spread_lane["external_seed_atoms"]=len(spread_atoms); spread_lane["support_atoms"]=len(supports)

    total_games=games.copy(); total_predictors=[]; total_match={}
    total_behavior=[]; total_atoms=[]; total_lane={"status":"NO_TOTAL_HISTORY","retained_rules":[]}
    if not total_raw.empty:
        total_games,total_predictors,total_match=_pt_attach_totals(games,total_raw,log_func=log_func)
        total_behavior=_pt_predictor_behavior_totals(total_games,total_predictors)
        total_atoms=_pt_external_total_atoms(total_games,total_predictors,total_behavior)
        supports=_pt_select_support_atoms(legacy_total_atoms,total_games)
        combo=total_atoms+supports
        total_lane=_search_market(
            total_games,total_label,combo,"TOTALS",False,log_func=log_func,
            seed_indices=list(range(len(total_atoms))),
            extension_indices=list(range(len(total_atoms),len(combo))),
            max_depth_override=PT_EXTERNAL_MAX_DEPTH,
            beam_width_override=PT_EXTERNAL_BEAM_WIDTH,
            lane="PREDICTION_TRACKER_EXTERNAL",
        )
        total_lane["external_seed_atoms"]=len(total_atoms); total_lane["support_atoms"]=len(supports)

    spread_confirmed=[x for x in spread_behavior if x.get("confirmed")]
    total_confirmed=[x for x in total_behavior if x.get("confirmed")]
    mechanisms=_pt_external_mechanisms(spread_lane,"SPREADS")+_pt_external_mechanisms(total_lane,"TOTALS")
    report={
        "status":"PASS",
        "contract":"TWO_SEPARATE_EXTERNAL_FAMILIES_BY_MARKET__LEGACY_LANE_INSULATED__ONE_EXTERNAL_SIGNAL_PER_RULE__DISCOVERY_2017_2022__VALIDATION_2023_2025__2026_SEALED",
        "spread_load":spread_load,"total_load":total_load,
        "spread_match":spread_match,"total_match":total_match,
        "spread_predictor_count":len(spread_predictors),"total_predictor_count":len(total_predictors),
        "spread_predictor_behavior":spread_behavior,"total_predictor_behavior":total_behavior,
        "spread_confirmed_behavior_count":len(spread_confirmed),"total_confirmed_behavior_count":len(total_confirmed),
        "spread_strong_confirmed_behavior_count":sum(1 for x in spread_behavior if x.get("strong_confirmed")),
        "total_strong_confirmed_behavior_count":sum(1 for x in total_behavior if x.get("strong_confirmed")),
        "spread_external_atoms":len(spread_atoms),"total_external_atoms":len(total_atoms),
        "miner":{"spreads":spread_lane,"totals":total_lane},
        "external_mechanisms":mechanisms,
        "external_mechanism_count":len(mechanisms),
        "correlation_policy":"ONE_EXTERNAL_RATINGS_FAMILY_PER_MARKET__NO_MULTI_PREDICTOR_VOTE_STACKING",
        "source_cluster_policy":"SAGARIN_PI_REGRESSION_VARIANTS_COLLAPSED_FOR_CONSENSUS__OTHER_HEADERS_ONE_CLUSTER_EACH",
        "source_outcome_fields_policy":"SOURCE_R/H_SCORES_PHCOVER_PHWIN_NEVER_FEATURES",
        "prospective_eligible":bool(any(bool(x.get("qualified_current")) for x in mechanisms)),"qualified_system_count":sum(1 for x in mechanisms if x.get("qualified_current")),"pt_family_vote_cap":1,"pt_model_weight":0.0,"production_authority":0,"automatic_promotion":False,"year_2026_queried":False,
    }
    log_func("[NFL-PT-PREDICTOR-BEHAVIOR] "+json.dumps({
        "spread_predictors":len(spread_behavior),"spread_confirmed":len(spread_confirmed),
        "spread_strong_confirmed":sum(1 for x in spread_behavior if x.get("strong_confirmed")),
        "total_predictors":len(total_behavior),"total_confirmed":len(total_confirmed),
        "total_strong_confirmed":sum(1 for x in total_behavior if x.get("strong_confirmed")),
        "selection_period":"2017_2022_ONLY","validation_period":"2023_2025","year_2026_selection":False,"production_authority":0,
    },sort_keys=True,default=str))
    log_func("[NFL-PT-MINER-LANE] "+json.dumps({
        "spread_external_seeds":len(spread_atoms),"spread_supports":spread_lane.get("support_atoms",0),
        "spread_tested":spread_lane.get("raw_tested",0),"spread_retained":len(spread_lane.get("retained_rules") or []),
        "total_external_seeds":len(total_atoms),"total_supports":total_lane.get("support_atoms",0),
        "total_tested":total_lane.get("raw_tested",0),"total_retained":len(total_lane.get("retained_rules") or []),
        "legacy_lane_mutated":False,"production_authority":0,
    },sort_keys=True,default=str))
    log_func("[NFL-PT-CONTRACT] "+json.dumps({
        "status":"PASS","spread_rows":spread_load.get("rows",0),"total_rows":total_load.get("rows",0),
        "spread_matched_games":spread_match.get("matched_games",0),"total_matched_games":total_match.get("matched_games",0),
        "external_mechanisms":len(mechanisms),"correlation_policy":report["correlation_policy"],
        "legacy_lane_mutated":False,"year_2026_queried":False,"automatic_promotion":False,"production_authority":0,
    },sort_keys=True,default=str))
    return report


def _spread_atoms(d:pd.DataFrame)->list[dict]:
    op=_num(d,"Opening_Spread"); oa=op.abs(); wk=_num(d,"Week_Number"); game_no=_num(d,"calc_team_game_number"); rest=_num(d,"Rest_Differential_Days")
    atoms=[]
    def add(name,family,mask,desc=None,boundary=None,team_specific=False):
        mm=pd.Series(mask,index=d.index).fillna(False).to_numpy(bool)
        disc=np.isin(_num(d,"Season").to_numpy(),DISCOVERY_SEASONS); n=int((mm&disc).sum())
        minimum=60 if team_specific else 28
        if n>=minimum and n<int(disc.sum()):atoms.append({"name":name,"family":family,"mask":mm,"description":desc or name,"boundary":boundary,"team_specific":bool(team_specific)})
    # Market role/price: required to define the wager, but no line movement or model state.
    add("HOME","LOCATION",_num(d,"Is_Home").eq(1)); add("ROAD","LOCATION",_num(d,"Is_Away").eq(1));
    add("OPEN_DOG","ROLE",op.gt(0)); add("OPEN_FAVORITE","ROLE",op.lt(0)); add("HOME_DOG","ROLE_COMPOSITE",_num(d,"Is_Home").eq(1)&op.gt(0)); add("ROAD_DOG","ROLE_COMPOSITE",_num(d,"Is_Away").eq(1)&op.gt(0));
    for lo,hi in ((0,3),(3,7),(7,10),(10,14),(14,99)):
        add(f"DOG_{lo}_{hi}","PRICE_BAND",op.gt(lo)&op.le(hi),boundary=("dog",lo,hi)); add(f"FAV_{lo}_{hi}","PRICE_BAND",(-op).gt(lo)&(-op).le(hi),boundary=("fav",lo,hi))
    add("DOG_10_PLUS","PRICE_EXTREME",op.ge(10),boundary=("dog_ge",10)); add("FAVORITE_10_PLUS","PRICE_EXTREME",op.le(-10),boundary=("fav_ge",10))
    # V3.14 buy-low/desperation thresholds. One PRICE_EXTREME atom per rule
    # is allowed by the grammar, preventing cut-point stacking.
    for cut,label in ((6.5,"6P5"),(7.0,"7"),(8.0,"8"),(9.0,"9")):
        add(f"DOG_{label}_PLUS","PRICE_EXTREME",op.ge(cut),boundary=("dog_ge",cut),desc=f"underdog +{cut:g} or larger")
    add("WEEK_1","SEASON_TIMING",wk.eq(1)); add("WEEK_2","SEASON_TIMING",wk.eq(2)); add("EARLY_WK1_4","SEASON_TIMING",wk.between(1,4)); add("MID_WK5_10","SEASON_TIMING",wk.between(5,10)); add("LATE_WK11_PLUS","SEASON_TIMING",wk.ge(11))
    add("GAME_1","SCHEDULE_POSITION",game_no.eq(1)); add("GAME_2","SCHEDULE_POSITION",game_no.eq(2)); add("GAME_7_PLUS","SCHEDULE_POSITION",game_no.ge(7)); add("GAME_9_PLUS","SCHEDULE_POSITION",game_no.ge(9))
    for _n in (3,4,5,6): add(f"TEAM_GAME_{_n}","SCHEDULE_POSITION",game_no.eq(_n),desc=f"team game {_n}")
    add("FIRST_HOME_GAME","SCHEDULE_POSITION",_num(d,"calc_first_home_game").eq(1)); add("FIRST_ROAD_GAME","SCHEDULE_POSITION",_num(d,"calc_first_road_game").eq(1))
    add("POSTSEASON","REGIME",d.Season_Stage.astype(str).eq("POSTSEASON")); add("DIVISION_GAME","MATCHUP",_num(d,"Is_Division_Game").eq(1)); add("NON_DIVISION_GAME","MATCHUP",_num(d,"Is_Division_Game").eq(0)); add("CONFERENCE_GAME","MATCHUP",_num(d,"Is_Conference_Game").eq(1)); add("INTERCONFERENCE_GAME","MATCHUP",_num(d,"Is_Interconference_Game").eq(1)); add("CONFERENCE_NON_DIVISION","MATCHUP",_num(d,"Is_Conference_Game").eq(1)&_num(d,"Is_Division_Game").eq(0)); add("REVENGE","MATCHUP",_num(d,"Revenge_Flag_Current").eq(1)); add("REMATCH_730D","MATCHUP",_num(d,"Days_Since_Last_Matchup_System").between(1,730)); add("IMMEDIATE_REMATCH","MATCHUP",_num(d,"calc_immediate_rematch").eq(1))
    add("PRIMETIME","SCHEDULE_CONTEXT",_num(d,"Is_PrimeTime").eq(1)); add("ISLAND_GAME","SCHEDULE_CONTEXT",_num(d,"calc_island_game").eq(1)); add("NIGHT_GAME","SCHEDULE_CONTEXT",_num(d,"Is_Night_Game").eq(1)); add("TNF","SCHEDULE_CONTEXT",_num(d,"Is_TNF").eq(1)); add("MNF","SCHEDULE_CONTEXT",_num(d,"Is_MNF").eq(1)); add("SNF","SCHEDULE_CONTEXT",_num(d,"Is_SNF").eq(1))
    add("REST_ADV_2_PLUS","REST",rest.ge(2),boundary=("rest_adv",2)); add("REST_DISADV_2_PLUS","REST",rest.le(-2),boundary=("rest_disadv",2)); add("THURSDAY_SHORT_WEEK","REST",_num(d,"Is_Thursday_Short_Week").eq(1)); add("BYE_LIKE_REST","REST",_num(d,"Is_Bye_Like_Rest").eq(1))
    pm=_num(d,"calc_prev1_margin"); add("OFF_SU_WIN","PRIOR_SU",pm.gt(0)); add("OFF_SU_LOSS","PRIOR_SU",pm.lt(0)); add("OFF_BLOWOUT_WIN_14_PLUS","PRIOR_SU",pm.ge(14),boundary=("prev_margin_ge",14)); add("OFF_BLOWOUT_LOSS_14_PLUS","PRIOR_SU",pm.le(-14),boundary=("prev_margin_le",-14)); add("OFF_BLOWOUT_LOSS_21_PLUS","PRIOR_SU",pm.le(-21),boundary=("prev_margin_le",-21))
    add("OFF_UPSET_WIN","PRIOR_ROLE_RESULT",pm.gt(0)&_num(d,"calc_prev1_open_dog").eq(1)); add("OFF_UPSET_LOSS_AS_FAVORITE","PRIOR_ROLE_RESULT",pm.lt(0)&_num(d,"calc_prev1_open_favorite").eq(1)); add("OPP_OFF_UPSET_WIN","OPP_ROLE_RESULT",_num(d,"Opp_calc_prev1_margin").gt(0)&_num(d,"Opp_calc_prev1_open_dog").eq(1)); add("OPP_OFF_UPSET_LOSS_AS_FAVORITE","OPP_ROLE_RESULT",_num(d,"Opp_calc_prev1_margin").lt(0)&_num(d,"Opp_calc_prev1_open_favorite").eq(1))
    # V3.10.2 lookback-one completion: the immediately prior game now exposes
    # result, ATS result, role, location, magnitude, and opponent state symmetrically.
    add("LAST_GAME_HOME","PRIOR_LOCATION1",_num(d,"calc_prev1_home").eq(1)); add("LAST_GAME_ROAD","PRIOR_LOCATION1",_num(d,"calc_prev1_away").eq(1))
    add("LAST_GAME_DOG","PRIOR_ROLE1",_num(d,"calc_prev1_open_dog").eq(1)); add("LAST_GAME_FAVORITE","PRIOR_ROLE1",_num(d,"calc_prev1_open_favorite").eq(1))
    add("OPP_OFF_SU_WIN","OPP_PRIOR_SU1",_num(d,"Opp_calc_prev1_margin").gt(0)); add("OPP_OFF_SU_LOSS","OPP_PRIOR_SU1",_num(d,"Opp_calc_prev1_margin").lt(0))
    add("OPP_OFF_ATS_WIN","OPP_PRIOR_ATS1",_num(d,"Opp_calc_prev1_open_ats_win").eq(1)); add("OPP_OFF_ATS_LOSS","OPP_PRIOR_ATS1",_num(d,"Opp_calc_prev1_open_ats_loss").eq(1))
    add("OPP_LAST_GAME_HOME","OPP_PRIOR_LOCATION1",_num(d,"Opp_calc_prev1_home").eq(1)); add("OPP_LAST_GAME_ROAD","OPP_PRIOR_LOCATION1",_num(d,"Opp_calc_prev1_away").eq(1))
    add("OPP_LAST_GAME_DOG","OPP_PRIOR_ROLE1",_num(d,"Opp_calc_prev1_open_dog").eq(1)); add("OPP_LAST_GAME_FAVORITE","OPP_PRIOR_ROLE1",_num(d,"Opp_calc_prev1_open_favorite").eq(1))
    add("OFF_ATS_WIN","PRIOR_ATS",_num(d,"calc_prev1_open_ats_win").eq(1)); add("OFF_ATS_LOSS","PRIOR_ATS",_num(d,"calc_prev1_open_ats_loss").eq(1)); add("BACK_TO_BACK_ATS_LOSSES","PRIOR_ATS",_num(d,"calc_b2b_open_ats_losses").eq(1))
    add("SU_WIN_STREAK_2_PLUS","SEQUENCE",_num(d,"calc_su_win_streak_prior").ge(2)); add("SU_LOSS_STREAK_2_PLUS","SEQUENCE",_num(d,"calc_su_loss_streak_prior").ge(2))
    add("ATS_WIN_STREAK_2_PLUS","ATS_SEQUENCE",_num(d,"calc_ats_win_streak_prior").ge(2)); add("ATS_LOSS_STREAK_2_PLUS","ATS_SEQUENCE",_num(d,"calc_ats_loss_streak_prior").ge(2))
    add("B2B_HOME_GAMES","LOCATION_SEQUENCE",_num(d,"calc_b2b_home_games").eq(1)); add("B2B_ROAD_GAMES","LOCATION_SEQUENCE",_num(d,"calc_b2b_road_games").eq(1)); add("B2B_HOME_SU_WINS","LOCATION_SEQUENCE",_num(d,"calc_b2b_home_su_wins").eq(1)); add("B2B_ROAD_SU_WINS","LOCATION_SEQUENCE",_num(d,"calc_b2b_road_su_wins").eq(1))
    add("B2B_SCORED_31_PLUS","SCORING_SEQUENCE",_num(d,"calc_b2b_scored_31_plus").eq(1)); add("B2B_SCORED_28_PLUS","SCORING_SEQUENCE",_num(d,"calc_b2b_scored_28_plus").eq(1)); add("B2B_ALLOWED_17_OR_LESS","DEFENSE_SEQUENCE",_num(d,"calc_b2b_allowed_17_or_less").eq(1)); add("B2B_HOME_WINS_31_PLUS","SEQUENCE_COMPOSITE",_num(d,"calc_b2b_home_wins_31_plus").eq(1))
    add("LAST_GAME_DIVISION","PRIOR_MATCHUP_CLASS",_num(d,"calc_prev1_division").eq(1)); add("LAST_GAME_NON_DIVISION","PRIOR_MATCHUP_CLASS",_num(d,"calc_prev1_division").eq(0)); add("B2B_DIVISION_GAMES","PRIOR_MATCHUP_SEQUENCE",_num(d,"calc_b2b_division_games").eq(1)); add("B2B_NON_DIVISION_GAMES","PRIOR_MATCHUP_SEQUENCE",_num(d,"calc_b2b_nondivision_games").eq(1))
    add("LAST_GAME_CONFERENCE","PRIOR_CONFERENCE_CLASS",_num(d,"calc_prev1_conference").eq(1)); add("LAST_GAME_INTERCONFERENCE","PRIOR_CONFERENCE_CLASS",_num(d,"calc_prev1_conference").eq(0)); add("LAST_GAME_PRIMETIME","PRIOR_SCHEDULE_CONTEXT",_num(d,"calc_prev1_primetime").eq(1)); add("LAST_GAME_ISLAND","PRIOR_SCHEDULE_CONTEXT",_num(d,"calc_prev1_island").eq(1))
    add("LAST_OPP_WINPCT_GE_600","PRIOR_OPP_QUALITY",_num(d,"calc_prev1_opp_win_pct").ge(.6)); add("LAST_OPP_WINPCT_LE_400","PRIOR_OPP_QUALITY",_num(d,"calc_prev1_opp_win_pct").le(.4)&_num(d,"calc_prev1_opp_win_pct").notna()); add("B2B_OPP_WINPCT_GE_600","PRIOR_OPP_QUALITY_SEQUENCE",_num(d,"calc_prev1_opp_win_pct").ge(.6)&_num(d,"calc_prev2_opp_win_pct").ge(.6))
    add("OPP_SU_WIN_STREAK_2_PLUS","OPP_SEQUENCE",_num(d,"Opp_calc_su_win_streak_prior").ge(2)); add("OPP_SU_LOSS_STREAK_2_PLUS","OPP_SEQUENCE",_num(d,"Opp_calc_su_loss_streak_prior").ge(2))
    add("OPP_ATS_WIN_STREAK_2_PLUS","OPP_ATS_SEQUENCE",_num(d,"Opp_calc_ats_win_streak_prior").ge(2)); add("OPP_ATS_LOSS_STREAK_2_PLUS","OPP_ATS_SEQUENCE",_num(d,"Opp_calc_ats_loss_streak_prior").ge(2))
    add("OFF_SU_AND_ATS_WIN","PRIOR_COMBO",pm.gt(0)&_num(d,"calc_prev1_open_ats_win").eq(1)); add("OFF_SU_AND_ATS_LOSS","PRIOR_COMBO",pm.lt(0)&_num(d,"calc_prev1_open_ats_loss").eq(1))
    add("ROLE_FLIP_DOG_TO_FAVORITE","ROLE_CHANGE",_num(d,"calc_prev1_open_dog").eq(1)&op.lt(0)); add("ROLE_FLIP_FAVORITE_TO_DOG","ROLE_CHANGE",_num(d,"calc_prev1_open_favorite").eq(1)&op.gt(0))
    add("H2H_2_PLUS_PRIOR_MEETINGS","H2H",_num(d,"calc_h2h_prior_meetings").ge(2)); add("H2H_3_PLUS_PRIOR_MEETINGS","H2H",_num(d,"calc_h2h_prior_meetings").ge(3))
    add("H2H_WIN_STREAK_2_PLUS","H2H_SEQUENCE",_num(d,"calc_h2h_win_streak_prior").ge(2)); add("H2H_LOSS_STREAK_2_PLUS","H2H_SEQUENCE",_num(d,"calc_h2h_loss_streak_prior").ge(2))
    add("LAST_H2H_LOSS","H2H_RESULT",_num(d,"calc_h2h_last_margin").lt(0)); add("LAST_H2H_BLOWOUT_LOSS_10_PLUS","H2H_RESULT",_num(d,"calc_h2h_last_margin").le(-10))
    wp=_num(d,"calc_win_pct_prior"); owp=_num(d,"Opp_calc_win_pct_prior"); add("TEAM_WINPCT_LE_400","TEAM_STATE",wp.le(.4)&wp.notna()); add("TEAM_WINPCT_GE_600","TEAM_STATE",wp.ge(.6)); add("OPP_WINPCT_LE_500","OPP_STATE",owp.le(.5)&owp.notna()); add("OPP_WINPCT_GE_600","OPP_STATE",owp.ge(.6))
    # V3.14 SEASON_RECORD_STATE. calc_team_game_number is 1-based current
    # game, so prior_games=current-1. Streaks and win pct are pregame features.
    prior_games=game_no-1
    su_winless=prior_games.ge(1)&wp.eq(0)
    ats_loss_streak=_num(d,"calc_ats_loss_streak_prior")
    ats_coverless=prior_games.ge(1)&ats_loss_streak.ge(prior_games)
    add("SU_WINLESS_PRIOR","SEASON_RECORD_STATE",su_winless,desc="zero SU wins entering game")
    add("ATS_COVERLESS_PRIOR","SEASON_RECORD_STATE",ats_coverless,desc="zero ATS covers; every prior graded ATS game was a loss")
    add("SU_AND_ATS_WINLESS_PRIOR","SEASON_RECORD_STATE",su_winless&ats_coverless,desc="zero SU wins and zero ATS covers entering game")
    for _n in (2,3,4):
        add(f"SU_WINLESS_AFTER_{_n}_PLUS","SEASON_RECORD_STATE",su_winless&prior_games.ge(_n),desc=f"SU winless after {_n}+ completed games")
        add(f"ATS_COVERLESS_AFTER_{_n}_PLUS","SEASON_RECORD_STATE",ats_coverless&prior_games.ge(_n),desc=f"ATS coverless after {_n}+ completed games")
        add(f"SU_AND_ATS_WINLESS_AFTER_{_n}_PLUS","SEASON_RECORD_STATE",su_winless&ats_coverless&prior_games.ge(_n),desc=f"SU and ATS winless after {_n}+ completed games")
    add("OPP_HAS_SU_WIN","OPP_STATE",owp.gt(0),desc="opponent has at least one prior SU win")
    add("NEGATIVE_LAST5_MARGIN","FORM",_num(d,"Avg_SU_Margin_Last5_Prior").lt(0)); add("POSITIVE_LAST5_MARGIN","FORM",_num(d,"Avg_SU_Margin_Last5_Prior").gt(0)); add("TURNOVER_NEG_LAST3","FORM",_num(d,"Avg_Turnover_Margin_Last3_Prior").lt(-.5)); add("TURNOVER_POS_LAST3","FORM",_num(d,"Avg_Turnover_Margin_Last3_Prior").gt(.5))

    # V3.10.2 exact prior-two sequences.  Every 2-state SU/ATS/location/role
    # permutation is represented as one atom, preserving bounded grammar while
    # letting the evidence choose whether one, two, or three prior games matter.
    su2=( _num(d,"calc_prev2_margin"), _num(d,"calc_prev1_margin") )
    for pat in ("WW","WL","LW","LL"):
        mm=pd.Series(True,index=d.index)
        for v,ch in zip(su2,pat): mm &= (v.gt(0) if ch=="W" else v.lt(0))
        add("SU_SEQ2_"+pat,"SU_SEQUENCE2",mm)
    aw2s=( _num(d,"calc_prev2_open_ats_win"), _num(d,"calc_prev1_open_ats_win") ); al2s=( _num(d,"calc_prev2_open_ats_loss"), _num(d,"calc_prev1_open_ats_loss") )
    for pat in ("WW","WL","LW","LL"):
        mm=pd.Series(True,index=d.index)
        for w,l,ch in zip(aw2s,al2s,pat): mm &= (w.eq(1) if ch=="W" else l.eq(1))
        add("ATS_SEQ2_"+pat,"ATS_SEQUENCE2",mm)
    h2s=( _num(d,"calc_prev2_home"), _num(d,"calc_prev1_home") )
    for pat in ("HH","HR","RH","RR"):
        mm=h2s[0].notna()&h2s[1].notna()
        for v,ch in zip(h2s,pat): mm &= v.eq(1 if ch=="H" else 0)
        add("LOCATION_SEQ2_"+pat,"LOCATION_SEQUENCE2",mm)
    r2s=( _num(d,"calc_prev2_opening_spread"), _num(d,"calc_prev1_opening_spread") )
    for pat in ("DD","DF","FD","FF"):
        mm=r2s[0].ne(0)&r2s[1].ne(0)&r2s[0].notna()&r2s[1].notna()
        for v,ch in zip(r2s,pat): mm &= (v.gt(0) if ch=="D" else v.lt(0))
        add("ROLE_SEQ2_"+pat,"ROLE_SEQUENCE2",mm)

    # Two-game magnitude/trend grammar mirrors the V3.10 three-game definitions.
    pm2=_num(d,"calc_prev2_margin"); am2=_num(d,"calc_prev2_open_ats_margin")
    add("B2B_CLOSE_GAME_3","PRIOR_MARGIN_MAGNITUDE2",pm.abs().le(3)&pm2.abs().le(3)&pm.notna()&pm2.notna())
    add("B2B_CLOSE_GAME_7","PRIOR_MARGIN_MAGNITUDE2",pm.abs().le(7)&pm2.abs().le(7)&pm.notna()&pm2.notna())
    for cut in (10,14,21,28):
        add(f"B2B_WIN_{cut}_PLUS","PRIOR_MARGIN_MAGNITUDE2",pm.ge(cut)&pm2.ge(cut)); add(f"B2B_LOSS_{cut}_PLUS","PRIOR_MARGIN_MAGNITUDE2",pm.le(-cut)&pm2.le(-cut))
    am1b=_num(d,"calc_prev1_open_ats_margin")
    for cut in (7,14,21):
        add(f"B2B_ATS_COVER_{cut}_PLUS","PRIOR_ATS_MAGNITUDE2",am1b.ge(cut)&am2.ge(cut)); add(f"B2B_ATS_MISS_{cut}_PLUS","PRIOR_ATS_MAGNITUDE2",am1b.le(-cut)&am2.le(-cut))
    add("MARGIN_IMPROVING_2","MARGIN_TREND2",_num(d,"calc_margin_improving_2").eq(1)); add("MARGIN_WORSENING_2","MARGIN_TREND2",_num(d,"calc_margin_worsening_2").eq(1))
    add("ATS_MARGIN_IMPROVING_2","ATS_MARGIN_TREND2",_num(d,"calc_ats_margin_improving_2").eq(1)); add("ATS_MARGIN_WORSENING_2","ATS_MARGIN_TREND2",_num(d,"calc_ats_margin_worsening_2").eq(1))
    add("SCORING_IMPROVING_2","SCORING_TREND2",_num(d,"calc_scoring_improving_2").eq(1)); add("SCORING_WORSENING_2","SCORING_TREND2",_num(d,"calc_scoring_worsening_2").eq(1))
    add("DEFENSE_IMPROVING_2","DEFENSE_TREND2",_num(d,"calc_defense_improving_2").eq(1)); add("DEFENSE_WORSENING_2","DEFENSE_TREND2",_num(d,"calc_defense_worsening_2").eq(1))
    add("OPP_QUALITY_RISING_2","PRIOR_OPP_QUALITY_TREND2",_num(d,"calc_opp_quality_rising_2").eq(1)); add("OPP_QUALITY_FALLING_2","PRIOR_OPP_QUALITY_TREND2",_num(d,"calc_opp_quality_falling_2").eq(1))

    # One- and two-game opponent symmetry. These are single comparison atoms so
    # mirrored descriptions cannot inflate Bet Authority as independent votes.
    add("TEAM_SUW1_VS_OPP_SUL1","OPPONENT_SEQUENCE_SYMMETRY1",pm.gt(0)&_num(d,"Opp_calc_prev1_margin").lt(0))
    add("TEAM_SUL1_VS_OPP_SUW1","OPPONENT_SEQUENCE_SYMMETRY1",pm.lt(0)&_num(d,"Opp_calc_prev1_margin").gt(0))
    add("TEAM_ATSW1_VS_OPP_ATSL1","OPPONENT_ATS_SYMMETRY1",_num(d,"calc_prev1_open_ats_win").eq(1)&_num(d,"Opp_calc_prev1_open_ats_loss").eq(1))
    add("TEAM_ATSL1_VS_OPP_ATSW1","OPPONENT_ATS_SYMMETRY1",_num(d,"calc_prev1_open_ats_loss").eq(1)&_num(d,"Opp_calc_prev1_open_ats_win").eq(1))
    add("TEAM_2SUW_VS_OPP_2SUL","OPPONENT_SEQUENCE_SYMMETRY2",_num(d,"calc_b2b_su_wins").eq(1)&_num(d,"Opp_calc_b2b_su_losses").eq(1))
    add("TEAM_2SUL_VS_OPP_2SUW","OPPONENT_SEQUENCE_SYMMETRY2",_num(d,"calc_b2b_su_losses").eq(1)&_num(d,"Opp_calc_b2b_su_wins").eq(1))
    add("TEAM_2ATSW_VS_OPP_2ATSL","OPPONENT_ATS_SYMMETRY2",_num(d,"calc_b2b_open_ats_wins").eq(1)&_num(d,"Opp_calc_b2b_open_ats_losses").eq(1))
    add("TEAM_2ATSL_VS_OPP_2ATSW","OPPONENT_ATS_SYMMETRY2",_num(d,"calc_b2b_open_ats_losses").eq(1)&_num(d,"Opp_calc_b2b_open_ats_wins").eq(1))
    add("TEAM_MARGIN_UP2_OPP_MARGIN_DOWN2","OPPONENT_TREND_SYMMETRY2",_num(d,"calc_margin_improving_2").eq(1)&_num(d,"Opp_calc_margin_worsening_2").eq(1))
    add("TEAM_MARGIN_DOWN2_OPP_MARGIN_UP2","OPPONENT_TREND_SYMMETRY2",_num(d,"calc_margin_worsening_2").eq(1)&_num(d,"Opp_calc_margin_improving_2").eq(1))
    add("TEAM_ATS_MARGIN_UP2_OPP_DOWN2","OPPONENT_ATS_TREND_SYMMETRY2",_num(d,"calc_ats_margin_improving_2").eq(1)&_num(d,"Opp_calc_ats_margin_worsening_2").eq(1))
    add("TEAM_ATS_MARGIN_DOWN2_OPP_UP2","OPPONENT_ATS_TREND_SYMMETRY2",_num(d,"calc_ats_margin_worsening_2").eq(1)&_num(d,"Opp_calc_ats_margin_improving_2").eq(1))
    add("TEAM_2STRONG_OPP_VS_OPP_2WEAK_OPP","OPPONENT_QUALITY_SYMMETRY2",_num(d,"calc_b2b_opp_ge_600").eq(1)&_num(d,"Opp_calc_b2b_opp_le_400").eq(1))
    add("TEAM_2WEAK_OPP_VS_OPP_2STRONG_OPP","OPPONENT_QUALITY_SYMMETRY2",_num(d,"calc_b2b_opp_le_400").eq(1)&_num(d,"Opp_calc_b2b_opp_ge_600").eq(1))

    # V3.10 exact prior-three sequences.  These are single bounded atoms, not
    # three separately combinable lag atoms, so sequence depth can never exceed 3.
    def _sign_pattern3(cols,pattern,zero_positive=False):
        vals=[_num(d,c) for c in cols]; mm=pd.Series(True,index=d.index)
        for v,ch in zip(vals,pattern):
            mm &= (v.ge(0) if (zero_positive and ch in {"W","H","D"}) else v.gt(0)) if ch in {"W","H","D"} else (v.lt(0) if not zero_positive else v.lt(.5))
        return mm
    sucols=("calc_prev3_margin","calc_prev2_margin","calc_prev1_margin")
    for pat in ("WWW","WWL","WLW","WLL","LWW","LWL","LLW","LLL"):
        add("SU_SEQ3_"+pat,"SU_SEQUENCE3",_sign_pattern3(sucols,pat))
    aw3=_num(d,"calc_prev3_open_ats_win"); aw2=_num(d,"calc_prev2_open_ats_win"); aw1=_num(d,"calc_prev1_open_ats_win")
    al3=_num(d,"calc_prev3_open_ats_loss"); al2=_num(d,"calc_prev2_open_ats_loss"); al1=_num(d,"calc_prev1_open_ats_loss")
    for pat in ("WWW","WWL","WLW","WLL","LWW","LWL","LLW","LLL"):
        mm=pd.Series(True,index=d.index)
        for w,l,ch in zip((aw3,aw2,aw1),(al3,al2,al1),pat): mm &= (w.eq(1) if ch=="W" else l.eq(1))
        add("ATS_SEQ3_"+pat,"ATS_SEQUENCE3",mm)
    h3=_num(d,"calc_prev3_home"); h2=_num(d,"calc_prev2_home"); h1=_num(d,"calc_prev1_home")
    for pat in ("HHH","HHR","HRH","HRR","RHH","RHR","RRH","RRR"):
        mm=h3.notna()&h2.notna()&h1.notna()
        for v,ch in zip((h3,h2,h1),pat): mm &= v.eq(1 if ch=="H" else 0)
        add("LOCATION_SEQ3_"+pat,"LOCATION_SEQUENCE3",mm)
    rsp3=_num(d,"calc_prev3_opening_spread"); rsp2=_num(d,"calc_prev2_opening_spread"); rsp1=_num(d,"calc_prev1_opening_spread")
    for pat in ("DDD","DDF","DFD","DFF","FDD","FDF","FFD","FFF"):
        mm=rsp3.ne(0)&rsp2.ne(0)&rsp1.ne(0)&rsp3.notna()&rsp2.notna()&rsp1.notna()
        for v,ch in zip((rsp3,rsp2,rsp1),pat): mm &= (v.gt(0) if ch=="D" else v.lt(0))
        add("ROLE_SEQ3_"+pat,"ROLE_SEQUENCE3",mm)

    # Magnitude and acceleration/deceleration atoms.
    pm1=_num(d,"calc_prev1_margin"); am1=_num(d,"calc_prev1_open_ats_margin")
    add("OFF_CLOSE_GAME_3","PRIOR_MARGIN_MAGNITUDE",pm1.abs().le(3)&pm1.notna(),boundary=("abs_prev_margin_le",3))
    add("OFF_CLOSE_GAME_7","PRIOR_MARGIN_MAGNITUDE",pm1.abs().le(7)&pm1.notna(),boundary=("abs_prev_margin_le",7))
    for cut in (10,14,21,28):
        add(f"OFF_WIN_{cut}_PLUS","PRIOR_MARGIN_MAGNITUDE",pm1.ge(cut),boundary=("prev_margin_ge",cut))
        add(f"OFF_LOSS_{cut}_PLUS","PRIOR_MARGIN_MAGNITUDE",pm1.le(-cut),boundary=("prev_margin_le",-cut))
    for cut in (7,14,21):
        add(f"OFF_ATS_COVER_{cut}_PLUS","PRIOR_ATS_MAGNITUDE",am1.ge(cut),boundary=("prev_ats_margin_ge",cut))
        add(f"OFF_ATS_MISS_{cut}_PLUS","PRIOR_ATS_MAGNITUDE",am1.le(-cut),boundary=("prev_ats_margin_le",-cut))
    add("MARGIN_IMPROVING_3","MARGIN_TREND3",_num(d,"calc_margin_improving_3").eq(1)); add("MARGIN_WORSENING_3","MARGIN_TREND3",_num(d,"calc_margin_worsening_3").eq(1))
    add("ATS_MARGIN_IMPROVING_3","ATS_MARGIN_TREND3",_num(d,"calc_ats_margin_improving_3").eq(1)); add("ATS_MARGIN_WORSENING_3","ATS_MARGIN_TREND3",_num(d,"calc_ats_margin_worsening_3").eq(1))
    add("SCORING_IMPROVING_3","SCORING_TREND3",_num(d,"calc_scoring_improving_3").eq(1)); add("SCORING_WORSENING_3","SCORING_TREND3",_num(d,"calc_scoring_worsening_3").eq(1))
    add("DEFENSE_IMPROVING_3","DEFENSE_TREND3",_num(d,"calc_defense_improving_3").eq(1)); add("DEFENSE_WORSENING_3","DEFENSE_TREND3",_num(d,"calc_defense_worsening_3").eq(1))
    add("THREE_STRAIGHT_SU_WINS","SU_SEQUENCE3_SUMMARY",_num(d,"calc_3_su_wins").eq(1)); add("THREE_STRAIGHT_SU_LOSSES","SU_SEQUENCE3_SUMMARY",_num(d,"calc_3_su_losses").eq(1))
    add("THREE_STRAIGHT_ATS_WINS","ATS_SEQUENCE3_SUMMARY",_num(d,"calc_3_ats_wins").eq(1)); add("THREE_STRAIGHT_ATS_LOSSES","ATS_SEQUENCE3_SUMMARY",_num(d,"calc_3_ats_losses").eq(1))
    add("THREE_HOME_GAMES","LOCATION_SEQUENCE3_SUMMARY",_num(d,"calc_3_home_games").eq(1)); add("THREE_ROAD_GAMES","LOCATION_SEQUENCE3_SUMMARY",_num(d,"calc_3_road_games").eq(1))
    add("THREE_OPP_GE_600","PRIOR_OPP_QUALITY_SEQUENCE3",_num(d,"calc_3_opp_ge_600").eq(1)); add("THREE_OPP_LE_400","PRIOR_OPP_QUALITY_SEQUENCE3",_num(d,"calc_3_opp_le_400").eq(1))
    add("TWO_OF_THREE_OPP_GE_600","PRIOR_OPP_QUALITY_SEQUENCE3",_num(d,"calc_2of3_opp_ge_600").eq(1)); add("TWO_OF_THREE_OPP_LE_400","PRIOR_OPP_QUALITY_SEQUENCE3",_num(d,"calc_2of3_opp_le_400").eq(1))
    add("OPP_QUALITY_RISING_3","PRIOR_OPP_QUALITY_TREND3",_num(d,"calc_opp_quality_rising_3").eq(1)); add("OPP_QUALITY_FALLING_3","PRIOR_OPP_QUALITY_TREND3",_num(d,"calc_opp_quality_falling_3").eq(1))

    # Opponent-vs-team sequence symmetry. Each is one mechanism atom and therefore
    # cannot count as multiple independent votes inside a mined family.
    add("TEAM_3SUW_VS_OPP_3SUL","OPPONENT_SEQUENCE_SYMMETRY",_num(d,"calc_3_su_wins").eq(1)&_num(d,"Opp_calc_3_su_losses").eq(1))
    add("TEAM_3SUL_VS_OPP_3SUW","OPPONENT_SEQUENCE_SYMMETRY",_num(d,"calc_3_su_losses").eq(1)&_num(d,"Opp_calc_3_su_wins").eq(1))
    add("TEAM_3ATSW_VS_OPP_3ATSL","OPPONENT_ATS_SYMMETRY",_num(d,"calc_3_ats_wins").eq(1)&_num(d,"Opp_calc_3_ats_losses").eq(1))
    add("TEAM_3ATSL_VS_OPP_3ATSW","OPPONENT_ATS_SYMMETRY",_num(d,"calc_3_ats_losses").eq(1)&_num(d,"Opp_calc_3_ats_wins").eq(1))
    add("TEAM_MARGIN_UP_OPP_MARGIN_DOWN","OPPONENT_TREND_SYMMETRY",_num(d,"calc_margin_improving_3").eq(1)&_num(d,"Opp_calc_margin_worsening_3").eq(1))
    add("TEAM_MARGIN_DOWN_OPP_MARGIN_UP","OPPONENT_TREND_SYMMETRY",_num(d,"calc_margin_worsening_3").eq(1)&_num(d,"Opp_calc_margin_improving_3").eq(1))
    add("TEAM_ATS_MARGIN_UP_OPP_DOWN","OPPONENT_ATS_TREND_SYMMETRY",_num(d,"calc_ats_margin_improving_3").eq(1)&_num(d,"Opp_calc_ats_margin_worsening_3").eq(1))
    add("TEAM_ATS_MARGIN_DOWN_OPP_UP","OPPONENT_ATS_TREND_SYMMETRY",_num(d,"calc_ats_margin_worsening_3").eq(1)&_num(d,"Opp_calc_ats_margin_improving_3").eq(1))
    add("TEAM_STRONG_OPP_RUN_VS_OPP_WEAK_OPP_RUN","OPPONENT_QUALITY_SYMMETRY",_num(d,"calc_2of3_opp_ge_600").eq(1)&_num(d,"Opp_calc_2of3_opp_le_400").eq(1))
    add("TEAM_WEAK_OPP_RUN_VS_OPP_STRONG_OPP_RUN","OPPONENT_QUALITY_SYMMETRY",_num(d,"calc_2of3_opp_le_400").eq(1)&_num(d,"Opp_calc_2of3_opp_ge_600").eq(1))

    # Team-history priors are themselves prior-only. These are general rules
    # ("a team with demonstrated strength in this context"), not a hard-coded
    # franchise identity, and therefore can generalize across teams.
    add("TEAM_PRIMETIME_ATS_GE_550_N6","TEAM_CONTEXT_HISTORY",_num(d,"Is_PrimeTime").eq(1)&_num(d,"calc_team_primetime_ats_n_prior").ge(6)&_num(d,"calc_team_primetime_ats_rate_prior").ge(.55)); add("TEAM_PRIMETIME_ATS_LE_450_N6","TEAM_CONTEXT_HISTORY",_num(d,"Is_PrimeTime").eq(1)&_num(d,"calc_team_primetime_ats_n_prior").ge(6)&_num(d,"calc_team_primetime_ats_rate_prior").le(.45))
    add("TEAM_ISLAND_ATS_GE_550_N6","TEAM_ISLAND_HISTORY",_num(d,"calc_island_game").eq(1)&_num(d,"calc_team_island_ats_n_prior").ge(6)&_num(d,"calc_team_island_ats_rate_prior").ge(.55)); add("TEAM_ISLAND_ATS_LE_450_N6","TEAM_ISLAND_HISTORY",_num(d,"calc_island_game").eq(1)&_num(d,"calc_team_island_ats_n_prior").ge(6)&_num(d,"calc_team_island_ats_rate_prior").le(.45))
    add("TEAM_DIVISION_ATS_GE_550_N8","TEAM_MATCHUP_HISTORY",_num(d,"Is_Division_Game").eq(1)&_num(d,"calc_team_division_ats_n_prior").ge(8)&_num(d,"calc_team_division_ats_rate_prior").ge(.55)); add("TEAM_DIVISION_ATS_LE_450_N8","TEAM_MATCHUP_HISTORY",_num(d,"Is_Division_Game").eq(1)&_num(d,"calc_team_division_ats_n_prior").ge(8)&_num(d,"calc_team_division_ats_rate_prior").le(.45))
    add("TEAM_NONDIV_ATS_GE_550_N8","TEAM_NONDIV_HISTORY",_num(d,"Is_Division_Game").eq(0)&_num(d,"calc_team_nondivision_ats_n_prior").ge(8)&_num(d,"calc_team_nondivision_ats_rate_prior").ge(.55)); add("TEAM_NONDIV_ATS_LE_450_N8","TEAM_NONDIV_HISTORY",_num(d,"Is_Division_Game").eq(0)&_num(d,"calc_team_nondivision_ats_n_prior").ge(8)&_num(d,"calc_team_nondivision_ats_rate_prior").le(.45))
    add("TEAM_CONFERENCE_ATS_GE_550_N8","TEAM_CONF_HISTORY",_num(d,"Is_Conference_Game").eq(1)&_num(d,"calc_team_conference_ats_n_prior").ge(8)&_num(d,"calc_team_conference_ats_rate_prior").ge(.55)); add("TEAM_INTERCONF_ATS_GE_550_N6","TEAM_INTERCONF_HISTORY",_num(d,"Is_Interconference_Game").eq(1)&_num(d,"calc_team_interconference_ats_n_prior").ge(6)&_num(d,"calc_team_interconference_ats_rate_prior").ge(.55))
    add("TEAM_HOME_FAV_ATS_GE_550_N8","TEAM_ROLE_HISTORY",_num(d,"Is_Home").eq(1)&op.lt(0)&_num(d,"calc_team_home_favorite_ats_n_prior").ge(8)&_num(d,"calc_team_home_favorite_ats_rate_prior").ge(.55)); add("TEAM_HOME_FAV_ATS_LE_450_N8","TEAM_ROLE_HISTORY",_num(d,"Is_Home").eq(1)&op.lt(0)&_num(d,"calc_team_home_favorite_ats_n_prior").ge(8)&_num(d,"calc_team_home_favorite_ats_rate_prior").le(.45))
    add("TEAM_ROAD_DOG_ATS_GE_550_N8","TEAM_ROAD_DOG_HISTORY",_num(d,"Is_Away").eq(1)&op.gt(0)&_num(d,"calc_team_road_dog_ats_n_prior").ge(8)&_num(d,"calc_team_road_dog_ats_rate_prior").ge(.55)); add("TEAM_ROAD_DOG_ATS_LE_450_N8","TEAM_ROAD_DOG_HISTORY",_num(d,"Is_Away").eq(1)&op.gt(0)&_num(d,"calc_team_road_dog_ats_n_prior").ge(8)&_num(d,"calc_team_road_dog_ats_rate_prior").le(.45))
    # V3.10 prior-only team sequence memory. Six or more prior occurrences are
    # required before a team's own context history can emit an atom; global
    # discovery/validation/multiple-testing gates still apply afterward.
    for ctx,flag in (("after_3_su_wins","calc_3_su_wins"),("after_3_su_losses","calc_3_su_losses"),("after_3_ats_wins","calc_3_ats_wins"),("after_3_ats_losses","calc_3_ats_losses"),("after_blowout_win21",None),("after_blowout_loss21",None),("after_close_game3",None),("after_3_home","calc_3_home_games"),("after_3_road","calc_3_road_games")):
        cur=( _num(d,flag).eq(1) if flag else (pm1.ge(21) if ctx=="after_blowout_win21" else pm1.le(-21) if ctx=="after_blowout_loss21" else pm1.abs().le(3)&pm1.notna()) )
        ncol=f"calc_team_{ctx}_ats_n_prior"; rcol=f"calc_team_{ctx}_ats_rate_prior"
        tag=ctx.upper().replace("AFTER_","")
        add(f"TEAM_MEMORY_{tag}_GE_600_N6","TEAM_SEQUENCE_MEMORY",cur&_num(d,ncol).ge(6)&_num(d,rcol).ge(.60))
        add(f"TEAM_MEMORY_{tag}_LE_400_N6","TEAM_SEQUENCE_MEMORY",cur&_num(d,ncol).ge(6)&_num(d,rcol).le(.40))
    # Team-specific hypotheses are retained as a distinct research lane. They
    # receive a stricter discovery sample requirement. They never bypass the
    # later shadow/confirmation/final and multiple-testing gates.
    teams=d.Team_Norm.astype(str).fillna("");
    for tm,cnt in teams[np.isin(_num(d,"Season"),DISCOVERY_SEASONS)].value_counts().items():
        if tm and cnt>=60:
            label=re.sub(r"[^A-Za-z0-9]+","_",tm).strip("_").upper()[:40]; add("TEAM_"+label,"TEAM_IDENTITY",teams.eq(tm),team_specific=True)
    return atoms


def _total_atoms(g:pd.DataFrame)->list[dict]:
    ot=_num(g,"Opening_Total"); wk=_num(g,"Week_Number"); rest=_num(g,"Rest_Differential_Days")
    atoms=[]
    def add(name,family,mask,desc=None,boundary=None):
        mm=pd.Series(mask,index=g.index).fillna(False).to_numpy(bool); disc=np.isin(_num(g,"Season").to_numpy(),DISCOVERY_SEASONS); n=int((mm&disc).sum())
        if n>=28 and n<int(disc.sum()):atoms.append({"name":name,"family":family,"mask":mm,"description":desc or name,"boundary":boundary,"team_specific":False})
    for lo,hi in ((0,41.5),(41.5,45.5),(45.5,49.5),(49.5,99)):
        add(f"OPEN_TOTAL_{str(lo).replace('.','P')}_{str(hi).replace('.','P')}","TOTAL_BAND",ot.ge(lo)&ot.lt(hi),boundary=("total",lo,hi))
    add("DIVISION_GAME","MATCHUP",_num(g,"Is_Division_Game").eq(1)); add("NON_DIVISION_GAME","MATCHUP",_num(g,"Is_Division_Game").eq(0)); add("CONFERENCE_GAME","MATCHUP",_num(g,"Is_Conference_Game").eq(1)); add("INTERCONFERENCE_GAME","MATCHUP",_num(g,"Is_Interconference_Game").eq(1)); add("CONFERENCE_NON_DIVISION","MATCHUP",_num(g,"Is_Conference_Game").eq(1)&_num(g,"Is_Division_Game").eq(0)); add("REVENGE","MATCHUP",_num(g,"Revenge_Flag_Current").eq(1)); add("REMATCH_730D","MATCHUP",_num(g,"Days_Since_Last_Matchup_System").between(1,730)); add("IMMEDIATE_REMATCH","MATCHUP",_num(g,"calc_immediate_rematch").eq(1))
    add("PRIMETIME","SCHEDULE",_num(g,"Is_PrimeTime").eq(1)); add("ISLAND_GAME","SCHEDULE",_num(g,"calc_island_game").eq(1)); add("THURSDAY","SCHEDULE",_num(g,"Is_Thursday_Game").eq(1)); add("POSTSEASON","REGIME",g.Season_Stage.astype(str).eq("POSTSEASON")); add("EARLY_WK1_4","SEASON_TIMING",wk.between(1,4)); add("LATE_WK11_PLUS","SEASON_TIMING",wk.ge(11))
    add("REST_IMBALANCE_2_PLUS","REST",rest.abs().ge(2)); add("HOME_SHORT_REST","REST",_num(g,"Is_Short_Rest").eq(1)); add("OPP_SHORT_REST","REST",_num(g,"Opp_Is_Short_Rest").eq(1))
    p1=_num(g,"calc_prev1_total_points"); op1=_num(g,"Opp_calc_prev1_total_points"); add("BOTH_PRIOR_TOTALS_50_PLUS","PRIOR_SCORING",p1.ge(50)&op1.ge(50)); add("BOTH_PRIOR_TOTALS_UNDER_42","PRIOR_SCORING",p1.lt(42)&op1.lt(42)); add("ONE_PRIOR_TOTAL_55_PLUS","PRIOR_SCORING",p1.ge(55)|op1.ge(55))
    add("TEAM_B2B_SCORED_31_PLUS","SCORING_SEQUENCE",_num(g,"calc_b2b_scored_31_plus").eq(1)); add("OPP_B2B_SCORED_31_PLUS","OPP_SCORING_SEQUENCE",_num(g,"Opp_calc_b2b_scored_31_plus").eq(1)); add("TEAM_B2B_ALLOWED_17_OR_LESS","DEFENSE_SEQUENCE",_num(g,"calc_b2b_allowed_17_or_less").eq(1)); add("OPP_B2B_ALLOWED_17_OR_LESS","OPP_DEFENSE_SEQUENCE",_num(g,"Opp_calc_b2b_allowed_17_or_less").eq(1))
    add("LAST_GAME_DIVISION","PRIOR_MATCHUP_CLASS",_num(g,"calc_prev1_division").eq(1)); add("LAST_GAME_PRIMETIME","PRIOR_SCHEDULE_CONTEXT",_num(g,"calc_prev1_primetime").eq(1)); add("LAST_GAME_ISLAND","PRIOR_SCHEDULE_CONTEXT",_num(g,"calc_prev1_island").eq(1))
    add("BOTH_OFFENSE_LAST5_24_PLUS","OFFENSE_FORM",_num(g,"Avg_Points_For_Last5_Prior").ge(24)&_num(g,"Opp_Avg_Points_For_Last5_Prior").ge(24)); add("BOTH_OFFENSE_LAST5_UNDER_21","OFFENSE_FORM",_num(g,"Avg_Points_For_Last5_Prior").lt(21)&_num(g,"Opp_Avg_Points_For_Last5_Prior").lt(21))
    add("TURNOVER_VOLATILE","FORM",_num(g,"Avg_Turnover_Margin_Last3_Prior").abs().ge(1)|_num(g,"Opp_Avg_Turnover_Margin_Last3_Prior").abs().ge(1))
    # V3.10.2 totals horizon symmetry: explicit one-game scoring/defense state and
    # two-game scoring/defense/total trends before the existing three-game layer.
    add("TEAM_LAST_SCORED_28_PLUS","SCORING1",_num(g,"calc_prev1_points_for").ge(28)); add("OPP_LAST_SCORED_28_PLUS","OPP_SCORING1",_num(g,"Opp_calc_prev1_points_for").ge(28))
    add("TEAM_LAST_SCORED_31_PLUS","SCORING1",_num(g,"calc_prev1_points_for").ge(31)); add("OPP_LAST_SCORED_31_PLUS","OPP_SCORING1",_num(g,"Opp_calc_prev1_points_for").ge(31))
    add("TEAM_LAST_ALLOWED_17_OR_LESS","DEFENSE1",_num(g,"calc_prev1_points_against").le(17)&_num(g,"calc_prev1_points_against").notna()); add("OPP_LAST_ALLOWED_17_OR_LESS","OPP_DEFENSE1",_num(g,"Opp_calc_prev1_points_against").le(17)&_num(g,"Opp_calc_prev1_points_against").notna())
    add("TEAM_B2B_SCORED_28_PLUS","SCORING_SEQUENCE2",_num(g,"calc_b2b_scored_28_plus").eq(1)); add("OPP_B2B_SCORED_28_PLUS","OPP_SCORING_SEQUENCE2",_num(g,"Opp_calc_b2b_scored_28_plus").eq(1))
    add("TEAM_TWO_PRIOR_TOTALS_50_PLUS","PRIOR_TOTAL_SEQUENCE2",_num(g,"calc_2_prior_totals_50_plus").eq(1)); add("OPP_TWO_PRIOR_TOTALS_50_PLUS","OPP_PRIOR_TOTAL_SEQUENCE2",_num(g,"Opp_calc_2_prior_totals_50_plus").eq(1))
    add("TEAM_TWO_PRIOR_TOTALS_UNDER_42","PRIOR_TOTAL_SEQUENCE2",_num(g,"calc_2_prior_totals_under_42").eq(1)); add("OPP_TWO_PRIOR_TOTALS_UNDER_42","OPP_PRIOR_TOTAL_SEQUENCE2",_num(g,"Opp_calc_2_prior_totals_under_42").eq(1))
    add("TEAM_SCORING_IMPROVING_2","SCORING_TREND2",_num(g,"calc_scoring_improving_2").eq(1)); add("OPP_SCORING_IMPROVING_2","OPP_SCORING_TREND2",_num(g,"Opp_calc_scoring_improving_2").eq(1))
    add("TEAM_SCORING_WORSENING_2","SCORING_TREND2",_num(g,"calc_scoring_worsening_2").eq(1)); add("OPP_SCORING_WORSENING_2","OPP_SCORING_TREND2",_num(g,"Opp_calc_scoring_worsening_2").eq(1))
    add("TEAM_DEFENSE_IMPROVING_2","DEFENSE_TREND2",_num(g,"calc_defense_improving_2").eq(1)); add("OPP_DEFENSE_IMPROVING_2","OPP_DEFENSE_TREND2",_num(g,"Opp_calc_defense_improving_2").eq(1))
    add("TEAM_DEFENSE_WORSENING_2","DEFENSE_TREND2",_num(g,"calc_defense_worsening_2").eq(1)); add("OPP_DEFENSE_WORSENING_2","OPP_DEFENSE_TREND2",_num(g,"Opp_calc_defense_worsening_2").eq(1))
    add("TEAM_TOTAL_IMPROVING_2","PRIOR_TOTAL_TREND2",_num(g,"calc_total_improving_2").eq(1)); add("OPP_TOTAL_IMPROVING_2","OPP_PRIOR_TOTAL_TREND2",_num(g,"Opp_calc_total_improving_2").eq(1))
    add("TEAM_TOTAL_WORSENING_2","PRIOR_TOTAL_TREND2",_num(g,"calc_total_worsening_2").eq(1)); add("OPP_TOTAL_WORSENING_2","OPP_PRIOR_TOTAL_TREND2",_num(g,"Opp_calc_total_worsening_2").eq(1))
    add("TEAM_SCORING_UP2_OPP_DEFENSE_DOWN2","OPPONENT_SCORING_SYMMETRY2",_num(g,"calc_scoring_improving_2").eq(1)&_num(g,"Opp_calc_defense_worsening_2").eq(1))
    add("TEAM_SCORING_DOWN2_OPP_DEFENSE_UP2","OPPONENT_SCORING_SYMMETRY2",_num(g,"calc_scoring_worsening_2").eq(1)&_num(g,"Opp_calc_defense_improving_2").eq(1))
    # V3.10 totals sequence/magnitude layer.
    add("TEAM_THREE_SCORED_28_PLUS","SCORING_SEQUENCE3",_num(g,"calc_3_scored_28_plus").eq(1)); add("OPP_THREE_SCORED_28_PLUS","OPP_SCORING_SEQUENCE3",_num(g,"Opp_calc_3_scored_28_plus").eq(1))
    add("TEAM_THREE_SCORED_31_PLUS","SCORING_SEQUENCE3",_num(g,"calc_3_scored_31_plus").eq(1)); add("OPP_THREE_SCORED_31_PLUS","OPP_SCORING_SEQUENCE3",_num(g,"Opp_calc_3_scored_31_plus").eq(1))
    add("TEAM_THREE_ALLOWED_17_OR_LESS","DEFENSE_SEQUENCE3",_num(g,"calc_3_allowed_17_or_less").eq(1)); add("OPP_THREE_ALLOWED_17_OR_LESS","OPP_DEFENSE_SEQUENCE3",_num(g,"Opp_calc_3_allowed_17_or_less").eq(1))
    add("TEAM_THREE_PRIOR_TOTALS_50_PLUS","PRIOR_TOTAL_SEQUENCE3",_num(g,"calc_3_prior_totals_50_plus").eq(1)); add("OPP_THREE_PRIOR_TOTALS_50_PLUS","OPP_PRIOR_TOTAL_SEQUENCE3",_num(g,"Opp_calc_3_prior_totals_50_plus").eq(1))
    add("TEAM_THREE_PRIOR_TOTALS_UNDER_42","PRIOR_TOTAL_SEQUENCE3",_num(g,"calc_3_prior_totals_under_42").eq(1)); add("OPP_THREE_PRIOR_TOTALS_UNDER_42","OPP_PRIOR_TOTAL_SEQUENCE3",_num(g,"Opp_calc_3_prior_totals_under_42").eq(1))
    add("TEAM_SCORING_IMPROVING_3","SCORING_TREND3",_num(g,"calc_scoring_improving_3").eq(1)); add("OPP_SCORING_IMPROVING_3","OPP_SCORING_TREND3",_num(g,"Opp_calc_scoring_improving_3").eq(1))
    add("TEAM_SCORING_WORSENING_3","SCORING_TREND3",_num(g,"calc_scoring_worsening_3").eq(1)); add("OPP_SCORING_WORSENING_3","OPP_SCORING_TREND3",_num(g,"Opp_calc_scoring_worsening_3").eq(1))
    add("TEAM_DEFENSE_IMPROVING_3","DEFENSE_TREND3",_num(g,"calc_defense_improving_3").eq(1)); add("OPP_DEFENSE_IMPROVING_3","OPP_DEFENSE_TREND3",_num(g,"Opp_calc_defense_improving_3").eq(1))
    add("BOTH_SCORING_IMPROVING_3","OPPONENT_SCORING_SYMMETRY",_num(g,"calc_scoring_improving_3").eq(1)&_num(g,"Opp_calc_scoring_improving_3").eq(1))
    add("TEAM_SCORING_UP_OPP_DEFENSE_DOWN","OPPONENT_SCORING_SYMMETRY",_num(g,"calc_scoring_improving_3").eq(1)&_num(g,"Opp_calc_defense_worsening_3").eq(1))
    add("TEAM_SCORING_DOWN_OPP_DEFENSE_UP","OPPONENT_SCORING_SYMMETRY",_num(g,"calc_scoring_worsening_3").eq(1)&_num(g,"Opp_calc_defense_improving_3").eq(1))
    return atoms


def _redundant(names:set[str])->bool:
    pairs=(("ROAD_DOG","ROAD"),("ROAD_DOG","OPEN_DOG"),("HOME_DOG","HOME"),("HOME_DOG","OPEN_DOG"),("OFF_BLOWOUT_LOSS_21_PLUS","OFF_BLOWOUT_LOSS_14_PLUS"),("BACK_TO_BACK_ATS_LOSSES","OFF_ATS_LOSS"),("OFF_SU_AND_ATS_WIN","OFF_SU_WIN"),("OFF_SU_AND_ATS_WIN","OFF_ATS_WIN"),("OFF_SU_AND_ATS_LOSS","OFF_SU_LOSS"),("OFF_SU_AND_ATS_LOSS","OFF_ATS_LOSS"),("FIRST_HOME_GAME","HOME"),("FIRST_ROAD_GAME","ROAD"))
    if any(a in names and b in names for a,b in pairs):return True
    if "OPEN_DOG" in names and any(n.startswith("DOG_") for n in names):return True
    if "OPEN_FAVORITE" in names and any(n.startswith("FAV_") for n in names):return True
    return False


def _bh_qvalues(vals):
    pv=np.asarray(vals,float); m=len(pv); q=np.ones(m,float)
    if m==0:return q
    order=np.argsort(pv); prev=1.0
    for pos in range(m-1,-1,-1):
        idx=int(order[pos]); rank=pos+1; cur=min(prev,float(pv[idx])*m/max(rank,1)); q[idx]=min(1.0,cur); prev=q[idx]
    return q


def _bootstrap_ci(labels,idx,direction,reps=BOOTSTRAP_REPS,seed=20261001):
    idx=np.asarray(idx,int); y=np.asarray(labels,float)[idx]; y=y[np.isfinite(y)]
    if len(y)<10:return [None,None]
    z=y if direction in ("PLAY_ON","OVER") else 1-y; rng=np.random.default_rng(seed); means=[]
    for _ in range(reps):means.append(float(np.mean(rng.choice(z,size=len(z),replace=True))))
    return [round(float(np.quantile(means,.025)),6),round(float(np.quantile(means,.975)),6)]


def _search_market(rows:pd.DataFrame,labels:np.ndarray,atoms:list[dict],market:str,side_mode:bool,log_func=print,seed_indices=None,extension_indices=None,max_depth_override=None,beam_width_override=None,lane="LEGACY")->dict:
    seasons=pd.to_numeric(rows.Season,errors="coerce").astype(int).to_numpy(); disc=np.isin(seasons,DISCOVERY_SEASONS); shadow=seasons==SHADOW_SEASON; confirm=seasons==CONFIRM_SEASON; final=seasons==FINAL_CHECK_SEASON; valid_label=np.isfinite(np.asarray(labels,float))
    candidate_rows=(lambda m:_candidate_rows_side(rows,m)) if side_mode else (lambda m:(np.where(np.asarray(m,bool))[0],0))
    _seed_indices=list(range(len(atoms))) if seed_indices is None else [int(i) for i in seed_indices if 0<=int(i)<len(atoms)]
    _extension_indices=list(range(len(atoms))) if extension_indices is None else [int(i) for i in extension_indices if 0<=int(i)<len(atoms)]
    _max_depth=int(MAX_DEPTH if max_depth_override is None else max_depth_override)
    _beam_width=int(BEAM_WIDTH if beam_width_override is None else beam_width_override)
    def rank(mask,depth,team_specific=False):
        ix,amb=candidate_rows(mask); si=ix[disc[ix]&valid_label[ix]]; y=np.asarray(labels,float)[si]
        min_n=(24+5*max(depth-1,0)) if team_specific else (50+8*max(depth-1,0))
        if len(y)<min_n:return None
        raw=float(np.mean(y)); direction=("PLAY_ON" if raw>=.5 else "FADE") if market=="SPREADS" else ("OVER" if raw>=.5 else "UNDER")
        hit=max(raw,1-raw); quality=(hit-.5)*math.sqrt(len(y))*5-.12*depth
        return {"selection_games":int(len(y)),"selection_rate":float(hit),"direction":direction,"quality":float(quality),"ambiguous":int(amb)}
    beam=[]; survivors=[]; seen=set(); tested=0
    for i in _seed_indices:
        a=atoms[i]
        r=rank(a["mask"],1,bool(a.get("team_specific",False))); tested+=1
        if r:beam.append({"idx":(i,),"atoms":(a["name"],),"cats":(a["family"],),"mask":a["mask"],"team_specific":bool(a.get("team_specific",False)),**r})
    beam=sorted(beam,key=lambda x:(x["quality"],x["selection_games"]),reverse=True)[:_beam_width]; survivors+=beam
    for depth in range(2,_max_depth+1):
        nxt=[]
        for st in beam:
            used=set(st["cats"])
            for i in _extension_indices:
                a=atoms[i]
                if i in st["idx"] or a["family"] in used:continue
                names=tuple(sorted(st["atoms"]+(a["name"],))); z=set(names)
                if names in seen or _redundant(z):continue
                seen.add(names); mm=np.asarray(st["mask"],bool)&np.asarray(a["mask"],bool); team_specific=bool(st.get("team_specific",False) or a.get("team_specific",False)); r=rank(mm,depth,team_specific); tested+=1
                if r:nxt.append({"idx":st["idx"]+(i,),"atoms":st["atoms"]+(a["name"],),"cats":st["cats"]+(a["family"],),"mask":mm,"team_specific":team_specific,**r})
        beam=sorted(nxt,key=lambda x:(x["quality"],x["selection_games"]),reverse=True)[:_beam_width]
        if not beam:break
        survivors+=beam
    # Near-duplicate hypothesis collapse.
    unique=[]; pruned=0
    for st in sorted(survivors,key=lambda x:(len(x["atoms"]),-x["selection_games"],-x["quality"])):
        ix,_=candidate_rows(st["mask"]); sig=frozenset(str(rows.iloc[i].physical_game_id) for i in ix[disc[ix]])
        if not sig:continue
        duplicate=False
        for k in unique:
            ks=k["_sig"]; u=len(sig|ks); jac=len(sig&ks)/u if u else 1
            if jac>=JACCARD_CUTOFF:duplicate=True;break
        if duplicate:pruned+=1;continue
        q=dict(st);q["_sig"]=sig;unique.append(q)
    finals=sorted(unique,key=lambda x:(x["quality"],x["selection_games"],-len(x["atoms"])),reverse=True)[:FINALISTS]
    atom_by={a["name"]:a for a in atoms}; results=[]
    for rankno,st in enumerate(finals,1):
        ix,amb=candidate_rows(st["mask"]); direction=st["direction"]
        validation=(shadow|confirm|final)
        recs={
            "discovery":_rec(labels,ix[disc[ix]],direction),"shadow":_rec(labels,ix[shadow[ix]],direction),"confirmation":_rec(labels,ix[confirm[ix]],direction),"final_2025":_rec(labels,ix[final[ix]],direction),
            "validation_2023_2025":_rec(labels,ix[validation[ix]],direction),"all_history":_rec(labels,ix,direction)
        }
        # Year-by-year and LOSO are discovery-only robustness tests.
        yby={};
        for sy in DISCOVERY_SEASONS:yby[str(sy)]=_rec(labels,ix[seasons[ix]==sy],direction)
        loso={}
        for sy in DISCOVERY_SEASONS:loso[str(sy)]=_rec(labels,ix[disc[ix]&(seasons[ix]!=sy)],direction)
        valid_loso=[v["rate"] for v in loso.values() if v["rate"] is not None]; min_loso=min(valid_loso) if valid_loso else None
        folds={"2017_2018":_rec(labels,ix[np.isin(seasons[ix],[2017,2018])],direction),"2019_2020":_rec(labels,ix[np.isin(seasons[ix],[2019,2020])],direction),"2021_2022":_rec(labels,ix[np.isin(seasons[ix],[2021,2022])],direction)}
        valid_fold=[v["rate"] for v in folds.values() if v["rate"] is not None]; min_fold=min(valid_fold) if valid_fold else None
        # Remove discovery's best season.
        best=max(DISCOVERY_SEASONS,key=lambda sy:(yby[str(sy)]["rate"] if yby[str(sy)]["rate"] is not None else -1))
        remove_best=_rec(labels,ix[disc[ix]&(seasons[ix]!=best)],direction)
        # Drop-one-condition robustness.
        drops=[]
        if len(st["atoms"])>1:
            for omit in st["atoms"]:
                mm=np.ones(len(rows),bool)
                for nm in st["atoms"]:
                    if nm!=omit:mm &= np.asarray(atom_by[nm]["mask"],bool)
                dix,_=candidate_rows(mm); drops.append({"dropped":omit,**_rec(labels,dix[disc[dix]],direction)})
        structural=min([x["rate"] for x in drops if x["rate"] is not None],default=None)
        si=ix[disc[ix]]; boot=_bootstrap_ci(labels,si,direction,seed=20261001+rankno)
        raw_y=np.asarray(labels,float)[si]; raw_y=raw_y[np.isfinite(raw_y)]; z=raw_y if direction in ("PLAY_ON","OVER") else 1-raw_y
        pnom=float(binomtest(int(z.sum()),len(z),.5,alternative="greater").pvalue) if len(z) else 1.0
        results.append({"rank":rankno,"market":market,"lane":lane,"conditions":list(st["atoms"]),"families":list(st["cats"]),"team_specific":bool(st.get("team_specific",False)),"direction":direction,"selection_games":st["selection_games"],"selection_rate":round(st["selection_rate"],6),"records":recs,"discovery_year_by_year":yby,"chronological_folds":folds,"min_fold_rate":min_fold,"leave_one_season_out":loso,"min_loso_rate":min_loso,"remove_best_season":best,"remove_best_season_record":remove_best,"drop_one_condition":drops,"structural_floor":structural,"bootstrap_ci95":boot,"nominal_pvalue":pnom,"ambiguous_both_sides":amb,"quality":st["quality"],"production_authority":0})
    # Multiple testing: max-stat over finalist masks with independent fair-coin null.
    sel_ix=[]; obs=[]
    for r,st in zip(results,finals):
        ix,_=candidate_rows(st["mask"]); si=ix[disc[ix]&valid_label[ix]]; sel_ix.append(si); yy=np.asarray(labels,float)[si]; obs.append(abs(float(np.mean(yy))-.5)*math.sqrt(len(yy)) if len(yy) else math.inf)
    rng=np.random.default_rng(20261002 if market=="SPREADS" else 20261003); nullmax=np.zeros(MAXSTAT_REPS)
    for b in range(MAXSTAT_REPS):
        yb=rng.integers(0,2,size=len(rows))
        mx=0.0
        for si in sel_ix:
            if len(si):mx=max(mx,abs(float(np.mean(yb[si]))-.5)*math.sqrt(len(si)))
        nullmax[b]=mx
    global_q=_bh_qvalues([r["nominal_pvalue"] for r in results]) if results else []
    families={}
    for i,r in enumerate(results):
        fk="|".join(r["families"][:2]) if r["families"] else "UNCLASSIFIED"; r["hypothesis_family"]=fk; families.setdefault(fk,[]).append(i)
    fam_names=list(families); fam_p=[]
    for fk in fam_names:
        pv=sorted(results[i]["nominal_pvalue"] for i in families[fk]); m=len(pv); fam_p.append(min([min(1,p*m/(j+1)) for j,p in enumerate(pv)],default=1))
    fam_q=_bh_qvalues(fam_p) if fam_p else []
    fam_map={fk:float(fam_q[j]) for j,fk in enumerate(fam_names)}
    for fk,ii in families.items():
        wq=_bh_qvalues([results[i]["nominal_pvalue"] for i in ii])
        for j,i in enumerate(ii):results[i]["within_family_qvalue"]=float(wq[j]);results[i]["hierarchical_family_qvalue"]=fam_map[fk]
    for i,r in enumerate(results):
        pmax=float((1+np.sum(nullmax>=obs[i]))/(MAXSTAT_REPS+1)) if np.isfinite(obs[i]) else 1.0
        r["permutation_max_pvalue"]=pmax;r["global_fdr_qvalue"]=float(global_q[i]);r["multiple_testing_pass"]=bool(pmax<=.10 and r.get("hierarchical_family_qvalue",1)<=.10 and r.get("within_family_qvalue",1)<=.10)
        sr=r["records"]["shadow"];cr=r["records"]["confirmation"];fr=r["records"]["final_2025"];vr=r["records"]["validation_2023_2025"];dr=r["records"]["discovery"]
        blocks=(sr,cr,fr)
        positive_50=sum(1 for x in blocks if x["n"]>=8 and x["rate"] is not None and x["rate"]>.50)
        positive_be=sum(1 for x in blocks if x["n"]>=8 and x["rate"] is not None and x["rate"]>BREAK_EVEN_REFERENCE)
        r["validation_positive_seasons_50"]=int(positive_50); r["validation_positive_seasons_break_even"]=int(positive_be)
        discovery_strict=dr["n"]>=60 and dr["rate"] is not None and dr["rate"]>=.55 and r["bootstrap_ci95"][0] is not None and r["bootstrap_ci95"][0]>.50
        validation_strict=vr["n"]>=36 and vr["rate"] is not None and vr["rate"]>BREAK_EVEN_REFERENCE and positive_50>=2
        floors_ok=(r["min_fold_rate"] is not None and r["min_fold_rate"]>=.50 and r["min_loso_rate"] is not None and r["min_loso_rate"]>=.51 and r["remove_best_season_record"]["rate"] is not None and r["remove_best_season_record"]["rate"]>=.51)
        team_sample_years=sum(1 for x in yby.values() if x.get("n",0)>=4)
        team_positive_years=sum(1 for x in yby.values() if x.get("n",0)>=4 and x.get("rate") is not None and x.get("rate")>=.50)
        r["team_identity_season_stability"]={"sample_years":int(team_sample_years),"positive_years":int(team_positive_years)} if r.get("team_specific",False) else None
        team_strict=(not r.get("team_specific",False)) or (dr["n"]>=45 and vr["n"]>=18 and positive_50>=2 and team_sample_years>=4 and team_positive_years>=3)
        legit=bool(discovery_strict and validation_strict and floors_ok and r["multiple_testing_pass"] and team_strict)

        # NCAAF-style research retention: do not confuse "not ready for promotion"
        # with "not interesting." Direction/rule selection is still frozen in
        # 2017-2022; later blocks only classify historical strength.
        promising_discovery=dr["n"]>=44 and dr["rate"] is not None and dr["rate"]>=.55
        promising_validation=vr["n"]>=24 and vr["rate"] is not None and vr["rate"]>.50 and positive_50>=2
        team_promising=(not r.get("team_specific",False)) or (dr["n"]>=30 and vr["n"]>=15 and team_sample_years>=3 and team_positive_years>=2)
        promising=bool((not legit) and promising_discovery and promising_validation and team_promising)

        watch_discovery=dr["n"]>=36 and dr["rate"] is not None and dr["rate"]>=.54
        watch_validation=vr["n"]>=15 and vr["rate"] is not None and vr["rate"]>=.50
        strong_small=dr["n"]>=50 and dr["rate"] is not None and dr["rate"]>=.57 and vr["n"]>=12
        watch=bool((not legit) and (not promising) and ((watch_discovery and watch_validation) or strong_small))

        # Evidence lifecycle contract (V3.3): discovery evidence is permanent provenance.
        # Later validation changes CURRENT authority/state, never erases the historical rule.
        team_history_discovery=bool(r.get("team_specific",False) and dr["n"]>=24 and dr["rate"] is not None and dr["rate"]>=.58)
        discovery_edge_established=bool(watch_discovery or strong_small or promising_discovery or discovery_strict or team_history_discovery)
        dr_rate=float(dr["rate"]) if dr.get("rate") is not None else None
        vr_rate=float(vr["rate"]) if vr.get("rate") is not None else None
        edge_decay_pp=(100.0*(dr_rate-vr_rate)) if dr_rate is not None and vr_rate is not None else None
        if legit:
            current_state="CURRENT_EDGE_VALIDATED"
        elif discovery_edge_established and vr.get("n",0)>=24 and vr_rate is not None and vr_rate>BREAK_EVEN_REFERENCE:
            current_state="VALIDATED_BUT_WEAKENED" if (dr_rate is not None and vr_rate+0.005<dr_rate) else "CURRENT_EDGE_POSITIVE"
        elif discovery_edge_established and vr.get("n",0)>=15 and vr_rate is not None and vr_rate>.50:
            current_state="POSITIVE_BUT_BELOW_MINUS110_BREAK_EVEN"
        elif discovery_edge_established and vr.get("n",0)>=15 and vr_rate is not None:
            current_state="DORMANT_RECENTLY_DEGRADED"
        elif discovery_edge_established:
            current_state="HISTORICAL_EDGE_AWAITING_MORE_VALIDATION"
        else:
            current_state="EXPLORATORY_ONLY"

        status=("LEGIT_CANDIDATE_REQUIRES_PROSPECTIVE" if legit else
                "PROMISING_HISTORICAL_SYSTEM" if promising else
                "WATCHLIST" if watch else
                "TEAM_SPECIFIC_HISTORICAL_PATTERN" if (r.get("team_specific",False) and discovery_edge_established) else
                "DORMANT_HISTORICAL_EDGE" if discovery_edge_established else "EXPLORATORY")
        r["status"]=status
        r["historical_discovery_status"]="DISCOVERY_EDGE_ESTABLISHED" if discovery_edge_established else "DISCOVERY_HYPOTHESIS_ONLY"
        r["historical_discovery_retained"]=bool(discovery_edge_established)
        r["current_evidence_state"]=current_state
        r["edge_decay_percentage_points"]=round(edge_decay_pp,3) if edge_decay_pp is not None else None
        r["research_retained"]=bool(status!="EXPLORATORY" or discovery_edge_established)
    legit_rules=[r for r in results if r["status"]=="LEGIT_CANDIDATE_REQUIRES_PROSPECTIVE"]
    promising_rules=[r for r in results if r["status"]=="PROMISING_HISTORICAL_SYSTEM"]
    watch_rules=[r for r in results if r["status"]=="WATCHLIST"]
    dormant_rules=[r for r in results if r["status"]=="DORMANT_HISTORICAL_EDGE"]
    retained_rules=[r for r in results if r.get("research_retained")]
    strong_rules=legit_rules+promising_rules
    log_func(f"[NFL-RESEARCH-V2-SYSTEM-MINER-V313] market={market} lane={lane} atoms={len(atoms)} seed_atoms={len(_seed_indices)} extension_atoms={len(_extension_indices)} raw_tested={tested} survivors={len(survivors)} unique_hypotheses={len(unique)} duplicate_masks_pruned={pruned} finalists={len(results)} legit={len(legit_rules)} promising={len(promising_rules)} watch={len(watch_rules)} dormant_historical={len(dormant_rules)} retained={len(retained_rules)} discovery=2017-2022 frozen_validation=2023-2025 year_2026_queried=FALSE")
    return {"market":market,"lane":lane,"atom_count":len(atoms),"seed_atom_count":len(_seed_indices),"extension_atom_count":len(_extension_indices),"raw_tested":tested,"survivor_rules":len(survivors),"unique_hypotheses":len(unique),"duplicate_masks_pruned":pruned,"jaccard_cutoff":JACCARD_CUTOFF,"max_depth":_max_depth,"max_sequence_depth":MAX_SEQUENCE_DEPTH,"beam_width":_beam_width,"maxstat_reps":MAXSTAT_REPS,"bootstrap_reps":BOOTSTRAP_REPS,"finalists":results,"legit_rules":legit_rules,"promising_rules":promising_rules,"strong_rules":strong_rules,"watch_rules":watch_rules,"dormant_historical_rules":dormant_rules,"retained_rules":retained_rules,"production_authority":0}



# -------------------------- mechanism-family consolidation --------------------------
_STATUS_PRIORITY={
    "LEGIT_CANDIDATE_REQUIRES_PROSPECTIVE":4,
    "PROMISING_HISTORICAL_SYSTEM":3,
    "WATCHLIST":2,
    "TEAM_SPECIFIC_HISTORICAL_PATTERN":1,
    "DORMANT_HISTORICAL_EDGE":1,
    "EXPLORATORY":0,
}

def _rule_id(r:dict)->str:
    return "__".join(r.get("conditions") or [])

def _ambiguity_share(r:dict)->float:
    # ambiguous_both_sides counts side rows; divide by two to estimate physical games.
    a=max(0.0,float(r.get("ambiguous_both_sides") or 0.0))/2.0
    n=float(((r.get("records") or {}).get("all_history") or {}).get("n") or 0.0)
    return float(a/(a+n)) if a+n>0 else 0.0

def _mechanism_id(r:dict)->str:
    c=set(r.get("conditions") or []); market=str(r.get("market") or "").upper(); direction=str(r.get("direction") or "").upper()
    if market=="SPREADS":
        if "SEASON_RECORD_STATE" in {str(x).upper() for x in (r.get("families") or [])}:
            return f"NFL_SPREAD_SEASON_RECORD_STATE_{direction}"
        if "ROLE_FLIP_DOG_TO_FAVORITE" in c:
            return "NFL_SPREAD_ROLE_FLIP_DOG_TO_FAVORITE_FADE"
        if {"OFF_SU_LOSS","HOME","OPEN_FAVORITE"}.issubset(c):
            return "NFL_SPREAD_HOME_FAVORITE_OFF_SU_LOSS_FADE"
        if {"TEAM_WINPCT_LE_400","OPP_WINPCT_LE_500"}.issubset(c):
            return "NFL_SPREAD_WEAK_TEAM_VS_WEAK_OPP"
        if {"ROAD_DOG","OPP_WINPCT_LE_500"}.issubset(c):
            return "NFL_SPREAD_ROAD_DOG_VS_WEAK_OPP"
    if market=="TOTALS":
        if {"DIVISION_GAME","EARLY_WK1_4"}.issubset(c):
            return "NFL_TOTAL_EARLY_DIVISION_UNDER"
        if "PRIMETIME" in c:
            return f"NFL_TOTAL_PRIMETIME_{direction}"
        if "OPEN_TOTAL_49P5_99" in c:
            return f"NFL_TOTAL_HIGH_OPENING_TOTAL_{direction}"
        if {"LATE_WK11_PLUS","REST_IMBALANCE_2_PLUS"}.issubset(c):
            return f"NFL_TOTAL_LATE_REST_IMBALANCE_{direction}"
        if {"REMATCH_730D","EARLY_WK1_4"}.issubset(c):
            return f"NFL_TOTAL_EARLY_REMATCH_{direction}"
    fam=[str(x).upper() for x in (r.get("families") or []) if str(x).strip()]
    core="__".join(fam[:2]) if fam else "UNCLASSIFIED"
    return f"NFL_{market}_{direction}_{core}"

def _family_evaluator_id(fid:str)->str|None:
    return {
        "NFL_SPREAD_ROLE_FLIP_DOG_TO_FAVORITE_FADE":"EXISTING_ROLE_FLIP_FAMILY_V1",
        "NFL_SPREAD_HOME_FAVORITE_OFF_SU_LOSS_FADE":"SPREAD_HOME_FAVORITE_OFF_SU_LOSS_FADE_V1",
        "NFL_TOTAL_EARLY_DIVISION_UNDER":"TOTAL_DIVISION_EARLY_WEEK_UNDER_V1",
    }.get(fid)

def _representative_sort_key(r:dict):
    vr=((r.get("records") or {}).get("validation_2023_2025") or {})
    ci=vr.get("wilson95") or [None,None]
    lo=float(ci[0]) if ci and ci[0] is not None else -1.0
    rate=float(vr.get("rate")) if vr.get("rate") is not None else -1.0
    n=int(vr.get("n") or 0)
    return (_STATUS_PRIORITY.get(r.get("status"),0), n, lo, rate, -len(r.get("conditions") or []))

def _strongest_sort_key(r:dict):
    vr=((r.get("records") or {}).get("validation_2023_2025") or {})
    ci=vr.get("wilson95") or [None,None]
    lo=float(ci[0]) if ci and ci[0] is not None else -1.0
    rate=float(vr.get("rate")) if vr.get("rate") is not None else -1.0
    n=int(vr.get("n") or 0)
    return (_STATUS_PRIORITY.get(r.get("status"),0), lo, rate, n, -len(r.get("conditions") or []))

def _collapse_mechanism_families(spread_miner:dict,total_miner:dict)->list[dict]:
    members=list(spread_miner.get("retained_rules") or [])+list(total_miner.get("retained_rules") or [])
    grouped={}
    for r in members:
        rr=dict(r); rr["rule_id"]=_rule_id(rr); rr["ambiguity_share"]=_ambiguity_share(rr)
        grouped.setdefault(_mechanism_id(rr),[]).append(rr)
    out=[]
    for fid,rs in grouped.items():
        rs=sorted(rs,key=_representative_sort_key,reverse=True)
        clean=[r for r in rs if int(r.get("ambiguous_both_sides") or 0)==0]
        clean_legit=[r for r in clean if r.get("status")=="LEGIT_CANDIDATE_REQUIRES_PROSPECTIVE"]
        clean_prom=[r for r in clean if r.get("status")=="PROMISING_HISTORICAL_SYSTEM"]
        clean_hist=[r for r in clean if r.get("historical_discovery_retained")]
        clean_weakened=[r for r in clean_hist if r.get("current_evidence_state") in {"VALIDATED_BUT_WEAKENED","CURRENT_EDGE_POSITIVE","POSITIVE_BUT_BELOW_MINUS110_BREAK_EVEN"}]
        if clean_legit: fam_status="LEGIT_FAMILY_REQUIRES_PROSPECTIVE"
        elif clean_prom: fam_status="PROMISING_FAMILY"
        elif clean_weakened: fam_status="HISTORICAL_EDGE_WEAKENED_FAMILY"
        elif clean_hist: fam_status="DORMANT_HISTORICAL_EDGE_FAMILY"
        elif clean: fam_status="WATCH_FAMILY"
        else: fam_status="HOLD_AMBIGUOUS_BOTH_SIDES"
        pool=clean_legit or clean_prom or clean_weakened or clean_hist or clean or rs
        representative=sorted(pool,key=_representative_sort_key,reverse=True)[0]
        strongest=sorted(pool,key=_strongest_sort_key,reverse=True)[0]
        cond_sets=[set(r.get("conditions") or []) for r in rs]
        common=sorted(set.intersection(*cond_sets)) if cond_sets else []
        evaluator=_family_evaluator_id(fid)
        if fid=="NFL_SPREAD_ROLE_FLIP_DOG_TO_FAVORITE_FADE" and fam_status.startswith("LEGIT"):
            action="CONTINUE_EXISTING_CLOCK"
        elif fam_status.startswith("LEGIT") and evaluator:
            action="START_NEW_CLOCK"
        elif fam_status.startswith("LEGIT"):
            action="REVIEW_BEFORE_CLOCK"
        elif fam_status in {"PROMISING_FAMILY","HISTORICAL_EDGE_WEAKENED_FAMILY"}:
            action="SHADOW_TRACK_NO_AUTHORITY"
        elif fam_status=="DORMANT_HISTORICAL_EDGE_FAMILY":
            action="SHADOW_TRACK_FOR_RECURRENCE"
        else:
            action="RESEARCH_ONLY"
        representative_conditions=list(representative.get("conditions") or [])
        expert_sources=[]
        if any(str(c).startswith("EXPERT_PATHI_") for c in representative_conditions): expert_sources.append("PATHI")
        if any(str(c).startswith("EXPERT_BIGAL_") for c in representative_conditions): expert_sources.append("BIGAL")
        if any(str(c)=="EXPERT_PATHI_AND_BIGAL" for c in representative_conditions): expert_sources=list(sorted(set(expert_sources+["PATHI","BIGAL"])))
        expert_bridge=bool(expert_sources)
        # New expert-derived Miner families enter as research/shadow mechanisms only.
        # They must accumulate prospective evidence before ever becoming a live
        # confirmation family; historical discovery/validation alone is insufficient.
        if expert_bridge:
            action="SHADOW_TRACK_NO_AUTHORITY"
        out.append({
            "system_family_id":fid,
            "market":representative.get("market"),
            "direction":representative.get("direction"),
            "family_status":fam_status,
            "member_count":len(rs),
            "member_rule_ids":[r["rule_id"] for r in rs],
            "member_status_counts":{k:sum(1 for r in rs if r.get("status")==k) for k in _STATUS_PRIORITY},
            "historical_discovery_member_count":sum(1 for r in rs if r.get("historical_discovery_retained")),
            "current_evidence_states":sorted(set(str(r.get("current_evidence_state")) for r in rs if r.get("current_evidence_state"))),
            "common_conditions":common,
            "representative_rule_id":representative["rule_id"],
            "representative_conditions":representative_conditions,
            "expert_bridge":expert_bridge,
            "expert_sources":expert_sources,
            "expert_bridge_authority_policy":"NEW_EXPERT_BRIDGE_REQUIRES_PROSPECTIVE_EVIDENCE_BEFORE_LIVE_CONFIRMATION" if expert_bridge else None,
            "representative_records":representative.get("records") or {},
            "representative_ambiguity_share":representative.get("ambiguity_share",0.0),
            "strongest_validation_rule_id":strongest["rule_id"],
            "strongest_validation_records":strongest.get("records") or {},
            "specialist_rule_ids":[r["rule_id"] for r in sorted(pool,key=_strongest_sort_key,reverse=True)[:5]],
            "ambiguous_member_count":sum(1 for r in rs if int(r.get("ambiguous_both_sides") or 0)>0),
            "all_clean_members":bool(clean),
            "prospective_evaluator_id":evaluator,
            "prospective_action":action,
            "prospective_eligible":bool(action in {"CONTINUE_EXISTING_CLOCK","START_NEW_CLOCK"}) and not expert_bridge,
            "shadow_tracking_target":bool(action in {"SHADOW_TRACK_NO_AUTHORITY","SHADOW_TRACK_FOR_RECURRENCE"}),
            "correlation_policy":"ONE_MECHANISM_FAMILY_ONE_VOTE__VARIANTS_ARE_DIAGNOSTIC_NOT_INDEPENDENT",
            "production_authority":0,
        })
    pri={"LEGIT_FAMILY_REQUIRES_PROSPECTIVE":0,"PROMISING_FAMILY":1,"HISTORICAL_EDGE_WEAKENED_FAMILY":2,"DORMANT_HISTORICAL_EDGE_FAMILY":3,"WATCH_FAMILY":4,"HOLD_AMBIGUOUS_BOTH_SIDES":5}
    return sorted(out,key=lambda x:(pri.get(x["family_status"],9),x["market"],x["system_family_id"]))

def _family_registry_payload(families:list[dict])->dict:
    frozen=[f for f in families if f.get("prospective_eligible")]
    return {
        "source_tag":SOURCE_TAG,
        "family_registry_version":"NFL_SYSTEM_MECHANISM_FAMILY_V1",
        "history_through":2025,
        "year_2026_queried":False,
        "family_count":len(families),
        "families":families,
        "prospective_family_ids":[f["system_family_id"] for f in frozen],
        "new_clock_family_ids":[f["system_family_id"] for f in frozen if f.get("prospective_action")=="START_NEW_CLOCK"],
        "continue_clock_family_ids":[f["system_family_id"] for f in frozen if f.get("prospective_action")=="CONTINUE_EXISTING_CLOCK"],
        "shadow_tracking_family_ids":[f["system_family_id"] for f in families if f.get("shadow_tracking_target")],
        "ambiguous_rule_policy":"AMBIGUOUS_BOTH_SIDES_EXCLUDED_FROM_PROSPECTIVE_FAMILY_REPRESENTATIVES",
        "automatic_promotion":False,
        "production_authority":0,
    }

CONDITION_TEXT = {
    "HOME":"team is home", "ROAD":"team is on the road", "OPEN_DOG":"team opened as an underdog", "OPEN_FAVORITE":"team opened as a favorite",
    "HOME_DOG":"home underdog", "ROAD_DOG":"road underdog", "DOG_0_3":"underdog from >0 to +3", "DOG_3_7":"underdog from >+3 to +7",
    "DOG_7_10":"underdog from >+7 to +10", "DOG_10_14":"underdog from >+10 to +14", "DOG_14_99":"underdog above +14",
    "FAV_0_3":"favorite from >0 to -3", "FAV_3_7":"favorite from >-3 to -7", "FAV_7_10":"favorite from >-7 to -10",
    "FAV_10_14":"favorite from >-10 to -14", "FAV_14_99":"favorite beyond -14", "DOG_10_PLUS":"underdog +10 or more", "FAVORITE_10_PLUS":"favorite -10 or more",
    "WEEK_1":"Week 1", "WEEK_2":"Week 2", "EARLY_WK1_4":"Weeks 1-4", "MID_WK5_10":"Weeks 5-10", "LATE_WK11_PLUS":"Week 11 or later",
    "GAME_1":"team's first game", "GAME_2":"team's second game", "GAME_7_PLUS":"team game 7 or later", "GAME_9_PLUS":"team game 9 or later",
    "FIRST_HOME_GAME":"team's first home game", "FIRST_ROAD_GAME":"team's first road game", "POSTSEASON":"postseason",
    "DIVISION_GAME":"division game", "CONFERENCE_GAME":"conference game", "REVENGE":"revenge/rematch context", "REMATCH_730D":"teams met within 730 days",
    "REST_ADV_2_PLUS":"rest advantage of 2+ days", "REST_DISADV_2_PLUS":"rest disadvantage of 2+ days", "THURSDAY_SHORT_WEEK":"Thursday short week",
    "BYE_LIKE_REST":"bye-like rest", "OFF_SU_WIN":"off a straight-up win", "OFF_SU_LOSS":"off a straight-up loss",
    "OFF_BLOWOUT_WIN_14_PLUS":"off a 14+ point win", "OFF_BLOWOUT_LOSS_14_PLUS":"off a 14+ point loss", "OFF_BLOWOUT_LOSS_21_PLUS":"off a 21+ point loss",
    "OFF_UPSET_WIN":"off an outright underdog win", "OFF_UPSET_LOSS_AS_FAVORITE":"off an upset loss as favorite", "OPP_OFF_UPSET_WIN":"opponent off an outright underdog win",
    "OPP_OFF_UPSET_LOSS_AS_FAVORITE":"opponent off an upset loss as favorite", "OFF_ATS_WIN":"off an ATS win", "OFF_ATS_LOSS":"off an ATS loss",
    "BACK_TO_BACK_ATS_LOSSES":"off back-to-back ATS losses", "SU_WIN_STREAK_2_PLUS":"2+ game SU win streak", "SU_LOSS_STREAK_2_PLUS":"2+ game SU loss streak",
    "ATS_WIN_STREAK_2_PLUS":"2+ game ATS win streak", "ATS_LOSS_STREAK_2_PLUS":"2+ game ATS loss streak", "OPP_SU_WIN_STREAK_2_PLUS":"opponent on 2+ game SU win streak",
    "OPP_SU_LOSS_STREAK_2_PLUS":"opponent on 2+ game SU loss streak", "OPP_ATS_WIN_STREAK_2_PLUS":"opponent on 2+ game ATS win streak", "OPP_ATS_LOSS_STREAK_2_PLUS":"opponent on 2+ game ATS loss streak",
    "OFF_SU_AND_ATS_WIN":"off both SU and ATS win", "OFF_SU_AND_ATS_LOSS":"off both SU and ATS loss", "ROLE_FLIP_DOG_TO_FAVORITE":"was dog last game, now favorite",
    "ROLE_FLIP_FAVORITE_TO_DOG":"was favorite last game, now dog", "H2H_2_PLUS_PRIOR_MEETINGS":"2+ prior H2H meetings", "H2H_3_PLUS_PRIOR_MEETINGS":"3+ prior H2H meetings",
    "H2H_WIN_STREAK_2_PLUS":"2+ H2H win streak", "H2H_LOSS_STREAK_2_PLUS":"2+ H2H loss streak", "LAST_H2H_LOSS":"lost last H2H meeting",
    "LAST_H2H_BLOWOUT_LOSS_10_PLUS":"lost last H2H meeting by 10+", "TEAM_WINPCT_LE_400":"team prior win pct <= .400", "TEAM_WINPCT_GE_600":"team prior win pct >= .600",
    "OPP_WINPCT_LE_500":"opponent prior win pct <= .500", "OPP_WINPCT_GE_600":"opponent prior win pct >= .600", "NEGATIVE_LAST5_MARGIN":"negative average SU margin last 5",
    "POSITIVE_LAST5_MARGIN":"positive average SU margin last 5", "TURNOVER_NEG_LAST3":"turnover margin below -0.5 last 3", "TURNOVER_POS_LAST3":"turnover margin above +0.5 last 3",
    "PRIMETIME":"primetime game", "THURSDAY":"Thursday game", "REST_IMBALANCE_2_PLUS":"rest imbalance of 2+ days", "HOME_SHORT_REST":"home side on short rest",
    "OPP_SHORT_REST":"opponent on short rest", "BOTH_PRIOR_TOTALS_50_PLUS":"both teams' prior games totaled 50+", "BOTH_PRIOR_TOTALS_UNDER_42":"both teams' prior games totaled under 42",
    "ONE_PRIOR_TOTAL_55_PLUS":"at least one team's prior game totaled 55+", "BOTH_OFFENSE_LAST5_24_PLUS":"both offenses averaged 24+ over last 5",
    "BOTH_OFFENSE_LAST5_UNDER_21":"both offenses averaged under 21 over last 5", "TURNOVER_VOLATILE":"high recent turnover volatility",
    "OPEN_TOTAL_0_41P5":"opening total below 41.5", "OPEN_TOTAL_41P5_45P5":"opening total 41.5-45.5", "OPEN_TOTAL_45P5_49P5":"opening total 45.5-49.5",
    "OPEN_TOTAL_49P5_99":"opening total 49.5+",
    "NON_DIVISION_GAME":"non-division game", "INTERCONFERENCE_GAME":"interconference game", "CONFERENCE_NON_DIVISION":"same-conference non-division game", "IMMEDIATE_REMATCH":"immediate rematch vs last opponent",
    "ISLAND_GAME":"standalone kickoff slot (one NFL game at that date/hour)", "NIGHT_GAME":"night game", "TNF":"Thursday Night Football", "MNF":"Monday Night Football", "SNF":"Sunday Night Football",
    "B2B_HOME_GAMES":"previous two games were at home", "B2B_ROAD_GAMES":"previous two games were on road", "B2B_HOME_SU_WINS":"won previous two games, both at home", "B2B_ROAD_SU_WINS":"won previous two games, both on road",
    "B2B_SCORED_31_PLUS":"scored 31+ in each of previous two games", "B2B_SCORED_28_PLUS":"scored 28+ in each of previous two games", "B2B_ALLOWED_17_OR_LESS":"allowed 17 or fewer in each of previous two games", "B2B_HOME_WINS_31_PLUS":"won previous two home games and scored 31+ in each",
    "LAST_GAME_DIVISION":"previous game was divisional", "LAST_GAME_NON_DIVISION":"previous game was non-divisional", "B2B_DIVISION_GAMES":"previous two games were divisional", "B2B_NON_DIVISION_GAMES":"previous two games were non-divisional",
    "LAST_GAME_CONFERENCE":"previous game was same-conference", "LAST_GAME_INTERCONFERENCE":"previous game was interconference", "LAST_GAME_PRIMETIME":"previous game was primetime", "LAST_GAME_ISLAND":"previous game was a standalone kickoff slot",
    "LAST_OPP_WINPCT_GE_600":"previous opponent entered at .600+", "LAST_OPP_WINPCT_LE_400":"previous opponent entered at .400 or worse", "B2B_OPP_WINPCT_GE_600":"previous two opponents entered at .600+",
    "TEAM_PRIMETIME_ATS_GE_550_N6":"team had 55%+ prior ATS rate in primetime with 6+ prior games", "TEAM_PRIMETIME_ATS_LE_450_N6":"team had 45%-or-worse prior ATS rate in primetime with 6+ prior games",
    "TEAM_ISLAND_ATS_GE_550_N6":"team had 55%+ prior ATS rate in standalone slots with 6+ prior games", "TEAM_ISLAND_ATS_LE_450_N6":"team had 45%-or-worse prior ATS rate in standalone slots with 6+ prior games",
    "TEAM_DIVISION_ATS_GE_550_N8":"team had 55%+ prior ATS rate in division games with 8+ prior games", "TEAM_DIVISION_ATS_LE_450_N8":"team had 45%-or-worse prior ATS rate in division games with 8+ prior games",
    "TEAM_NONDIV_ATS_GE_550_N8":"team had 55%+ prior ATS rate in non-division games with 8+ prior games", "TEAM_NONDIV_ATS_LE_450_N8":"team had 45%-or-worse prior ATS rate in non-division games with 8+ prior games",
    "TEAM_CONFERENCE_ATS_GE_550_N8":"team had 55%+ prior ATS rate in same-conference games with 8+ prior games", "TEAM_INTERCONF_ATS_GE_550_N6":"team had 55%+ prior ATS rate in interconference games with 6+ prior games",
    "TEAM_HOME_FAV_ATS_GE_550_N8":"team had 55%+ prior ATS rate as home favorite with 8+ prior games", "TEAM_HOME_FAV_ATS_LE_450_N8":"team had 45%-or-worse prior ATS rate as home favorite with 8+ prior games",
    "TEAM_ROAD_DOG_ATS_GE_550_N8":"team had 55%+ prior ATS rate as road dog with 8+ prior games", "TEAM_ROAD_DOG_ATS_LE_450_N8":"team had 45%-or-worse prior ATS rate as road dog with 8+ prior games",
    "TEAM_B2B_SCORED_31_PLUS":"team scored 31+ in each previous two games", "OPP_B2B_SCORED_31_PLUS":"opponent scored 31+ in each previous two games", "TEAM_B2B_ALLOWED_17_OR_LESS":"team allowed <=17 in each previous two games", "OPP_B2B_ALLOWED_17_OR_LESS":"opponent allowed <=17 in each previous two games",
}

FAMILY_DISPLAY_NAMES = {
    "NFL_SPREAD_ROLE_FLIP_DOG_TO_FAVORITE_FADE":"Role Flip: Dog to Favorite Fade",
    "NFL_SPREAD_HOME_FAVORITE_OFF_SU_LOSS_FADE":"Home Favorite off SU Loss Fade",
    "NFL_SPREAD_WEAK_TEAM_VS_WEAK_OPP":"Weak Team vs Weak Opponent",
    "NFL_SPREAD_ROAD_DOG_VS_WEAK_OPP":"Road Dog vs Weak Opponent",
    "NFL_TOTAL_EARLY_DIVISION_UNDER":"Early Division Under",
}

PATHI_COVERAGE = {
    "PF1_KEY_NUMBERS":{"status":"IMPLEMENTED_RESEARCH_AND_LIVE_CONTEXT","evidence":["key 3/7/10/14 distance/on-key structures","miner key-number contexts"]},
    "PF2_HOOK":{"status":"IMPLEMENTED_RESEARCH","evidence":["dog/favorite hook engineering flags"]},
    "PF3_ONTO_OFF_THROUGH_KEY":{"status":"PARTIAL_LIVE_MARKET_CONTEXT","evidence":["key-crossing contexts exist; full topology remains market-layer research"]},
    "PF4_MARKET_INFO_VS_PRICE_VALUE":{"status":"IMPLEMENTED_LIVE_EXECUTION_CONTEXT","evidence":["current price/book/quote age separated from market-state evidence"]},
    "PF5_OPENING_VS_CURRENT":{"status":"IMPLEMENTED_CONTEXT_NOT_PRODUCTION_PREDICTOR","evidence":["opening and current lines available; retrospective movement excluded from CORE"]},
    "PF6_DOG_BANDS":{"status":"IMPLEMENTED_RESEARCH","evidence":["0-3/3-7/7-10/10-14/14+ bands"]},
    "PF7_ROLE_SPECIFIC_TRENDS":{"status":"PARTIAL_RESEARCH","evidence":["favorite/dog/home/road roles and prior role rate; full ATS-role history not live authority"]},
    "PF8_ROLE_TRANSFORMATION":{"status":"IMPLEMENTED_RESEARCH","evidence":["dog-to-favorite and favorite-to-dog role flips"]},
    "PF9_POSTSEASON_REGIME":{"status":"IMPLEMENTED_RESEARCH","evidence":["postseason atom/regime"]},
    "PF10_INFORMATION_BEFORE_NARRATIVE":{"status":"NOT_FULLY_IMPLEMENTED","evidence":["no complete QB/injury/news latency feed in this System Lab contract"]},
    "PF11_NO_AUTOMATIC_BOUNCE_BACK":{"status":"PRINCIPLE_NOT_MECHANICAL_RULE","evidence":["off-loss states are tested conditionally, never automatic authority"]},
    "PF12_ENGINEERING_FEATURE_SET":{"status":"PARTIAL_TO_IMPLEMENTED","evidence":["key distance/bands/role flip/regime present; live move topology remains market research"]},
}

BIGAL_EXPECTED_IDS = ("BA-NFL1","BA-NFL1-HOME","BA-NFL2","BA-NFL2-ATS","BA-NFL3","BA-NFL4","BA-NFL5","BA-NFL6")


def _rule_text(conditions, direction):
    def _ct(c):
        z=str(c)
        if z.startswith("EXPERT_PATHI_"):
            return "Pathi expert atom: "+z.replace("EXPERT_PATHI_","").replace("_"," ").lower()
        if z.startswith("EXPERT_BIGAL_"):
            return "Big Al expert atom: "+z.replace("EXPERT_BIGAL_","").replace("_"," ").lower()
        if z=="EXPERT_PATHI_AND_BIGAL": return "Pathi and Big Al trigger on the same side"
        return CONDITION_TEXT.get(z,z.replace("_"," ").lower())
    parts=[_ct(c) for c in (conditions or [])]
    prefix={"PLAY_ON":"Play the qualifying team ATS", "FADE":"Fade the qualifying team ATS", "OVER":"Play OVER", "UNDER":"Play UNDER"}.get(str(direction).upper(), str(direction).title())
    return prefix + (" when " + "; ".join(parts) if parts else "")


def _record_text(rec):
    """Human record as W-L-P (ATS%). Supports both Miner `rate` and Pathi `hit_rate`."""
    if not isinstance(rec,dict): return "—"
    n=int(rec.get("n") or 0); w=int(rec.get("wins") or 0)
    l=int(rec.get("losses") if rec.get("losses") is not None else max(n-w,0))
    p=int(rec.get("pushes") or 0)
    if n<=0 and p<=0: return "—"
    rate=rec.get("rate") if rec.get("rate") is not None else rec.get("hit_rate")
    pct=f" ({100*float(rate):.1f}% ATS)" if rate is not None else ""
    return f"{w}-{l}-{p}{pct}"


def _period_record(meta, years):
    by=(meta or {}).get("by_season") or {}; w=l=p=0
    for sy in years:
        r=by.get(str(int(sy))) or {}
        w+=int(r.get("wins") or 0); l+=int(r.get("losses") or 0); p+=int(r.get("pushes") or 0)
    n=w+l
    return {"n":n,"wins":w,"losses":l,"pushes":p,"hit_rate":(w/n if n else None)}


PATHI_MINUS110_BREAK_EVEN = 110.0 / 210.0
PATHI_MIN_VALIDATION_N = 30
PATHI_MAJOR_REVERSAL_PCT = 0.08

# Mirror sides and directly nested variants share one Pathi mechanism family.
# Raw member records stay visible for audit, but the live overlay is allowed only
# one normalized vote per family and one Pathi lane vote per game.
PATHI_FAMILY_MAP = {
    "Pathi_FB_Dog_Below_Key_3": "PATHI_KEY_3_BELOW",
    "Pathi_FB_Favorite_Below_Key_3": "PATHI_KEY_3_BELOW",
    "Pathi_FB_Dog_Below_Key_7": "PATHI_KEY_7_BELOW",
    "Pathi_FB_Favorite_Below_Key_7": "PATHI_KEY_7_BELOW",
    "Pathi_FB_Dog_Hook_Above_3": "PATHI_KEY_3_HOOK",
    "Pathi_FB_Favorite_Laying_Hook_3": "PATHI_KEY_3_HOOK",
    "Pathi_FB_Dog_Hook_Above_7": "PATHI_KEY_7_HOOK",
    "Pathi_FB_Favorite_Laying_Hook_7": "PATHI_KEY_7_HOOK",
    "Pathi_FB_Dog_10_Plus": "PATHI_DOG_10_PLUS",
    "Pathi_FB_Dog_Hook_Above_10": "PATHI_DOG_10_PLUS",
    "Pathi_FB_Favorite_Below_Key_10": "PATHI_FAVORITE_BELOW_10",
    "Pathi_FB_Dog_TotalSpread_Gap_LE10": "PATHI_TOTAL_SPREAD_GAP_LE10",
    "Pathi_FB_Dog_Moved_Below_Key_3": "PATHI_MOVE_BELOW_3",
    "Pathi_FB_Dog_Moved_Above_Key_3": "PATHI_MOVE_ABOVE_3",
    "Pathi_FB_Dog_Moved_Below_Key_7": "PATHI_MOVE_BELOW_7",
    "Pathi_FB_Dog_Moved_Above_Key_7": "PATHI_MOVE_ABOVE_7",
    "Pathi_FB_Dog_Moved_Below_Key_10": "PATHI_MOVE_BELOW_10",
    "Pathi_FB_Dog_Moved_Above_Key_10": "PATHI_MOVE_ABOVE_10",
}
PATHI_FAMILY_LABELS = {
    "PATHI_KEY_3_BELOW": "Key 3 — below",
    "PATHI_KEY_7_BELOW": "Key 7 — below",
    "PATHI_KEY_3_HOOK": "Key 3 — hook",
    "PATHI_KEY_7_HOOK": "Key 7 — hook",
    "PATHI_DOG_10_PLUS": "Dog 10+",
    "PATHI_FAVORITE_BELOW_10": "Favorite below 10",
    "PATHI_TOTAL_SPREAD_GAP_LE10": "Dog total/spread gap <=10",
    "PATHI_MOVE_BELOW_3": "Move below 3",
    "PATHI_MOVE_ABOVE_3": "Move above 3",
    "PATHI_MOVE_BELOW_7": "Move below 7",
    "PATHI_MOVE_ABOVE_7": "Move above 7",
    "PATHI_MOVE_BELOW_10": "Move below 10",
    "PATHI_MOVE_ABOVE_10": "Move above 10",
}
PATHI_EVIDENCE_RANK = {
    "NO_SAMPLE": 0,
    "NO_SUPPORT": 1,
    "REGIME_REVERSAL": 2,
    "EMERGING_SUPPORT": 3,
    "WEAK_SUPPORT": 4,
    "STRONG_SUPPORT": 5,
}

def _pathi_member_role(system_id):
    sid=str(system_id or "")
    if "_Dog_" in sid: return "DOG"
    if "_Favorite_" in sid: return "FAVORITE"
    return "QUALIFIER"

def _pathi_evidence_detail(meta):
    disc=_period_record(meta,(2017,2018,2019,2020,2021,2022))
    val=_period_record(meta,(2023,2024,2025))
    overall={
        "n":int((meta or {}).get("n") or 0),
        "wins":int((meta or {}).get("wins") or 0),
        "losses":int((meta or {}).get("losses") or 0),
        "pushes":int((meta or {}).get("pushes") or 0),
        "hit_rate":(meta or {}).get("hit_rate"),
    }
    dn=int(disc.get("n") or 0); vn=int(val.get("n") or 0)
    dr=disc.get("hit_rate"); vr=val.get("hit_rate"); orate=overall.get("hit_rate")
    reversal=bool(
        dn >= PATHI_MIN_VALIDATION_N and vn >= PATHI_MIN_VALIDATION_N
        and dr is not None and vr is not None
        and float(dr) < 0.50
        and (float(vr)-float(dr)) >= PATHI_MAJOR_REVERSAL_PCT
    )
    if vn==0 or vr is None:
        level="NO_SAMPLE"
    elif float(vr) < PATHI_MINUS110_BREAK_EVEN:
        level="NO_SUPPORT"
    elif vn < PATHI_MIN_VALIDATION_N:
        level="EMERGING_SUPPORT"
    elif reversal:
        level="REGIME_REVERSAL"
    elif orate is not None and float(orate) >= PATHI_MINUS110_BREAK_EVEN:
        if float(vr) >= 0.55 and float(orate) >= 0.55 and (dr is None or float(dr) >= 0.50):
            level="STRONG_SUPPORT"
        else:
            level="WEAK_SUPPORT"
    else:
        level="REGIME_REVERSAL" if (dr is not None and float(dr) < 0.50) else "NO_SUPPORT"

    eligible=level in {"STRONG_SUPPORT","WEAK_SUPPORT"}
    status={
        "NO_SAMPLE":"PATHI_NO_HISTORICAL_MATCHES",
        "NO_SUPPORT":"PATHI_NO_SUPPORT",
        "EMERGING_SUPPORT":"PATHI_EMERGING_SUPPORT",
        "REGIME_REVERSAL":"PATHI_REGIME_REVERSAL",
        "WEAK_SUPPORT":"PATHI_WEAK_SUPPORT",
        "STRONG_SUPPORT":"PATHI_STRONG_SUPPORT",
    }[level]
    next_step={
        "NO_SAMPLE":"TRACK_ONLY_NO_REINFORCEMENT",
        "NO_SUPPORT":"TRACK_ONLY_NO_REINFORCEMENT",
        "EMERGING_SUPPORT":"TRACK_PENDING_SAMPLE",
        "REGIME_REVERSAL":"TRACK_REGIME_REVERSAL_NO_REINFORCEMENT",
        "WEAK_SUPPORT":"SHADOW_OVERLAY_SUPPORT",
        "STRONG_SUPPORT":"SHADOW_OVERLAY_SUPPORT",
    }[level]
    current_state={
        "NO_SAMPLE":"NO_VALIDATION_SAMPLE",
        "NO_SUPPORT":"BELOW_MINUS110_BREAK_EVEN",
        "EMERGING_SUPPORT":"POSITIVE_SMALL_SAMPLE",
        "REGIME_REVERSAL":"RECENT_POSITIVE_BUT_MAJOR_REGIME_REVERSAL",
        "WEAK_SUPPORT":"CURRENT_EDGE_VALIDATED_WEAK",
        "STRONG_SUPPORT":"CURRENT_EDGE_VALIDATED_STRONG",
    }[level]
    return {
        "evidence_level":level,
        "evidence_rank":PATHI_EVIDENCE_RANK[level],
        "normalized_vote_eligible":eligible,
        "status":status,
        "prospective_action":next_step,
        "current_evidence_state":current_state,
        "discovery_record":disc,
        "validation_record":val,
        "overall_record":overall,
        "major_regime_reversal":reversal,
    }

def _pathi_evidence(meta):
    d=_pathi_evidence_detail(meta)
    return d["status"],d["prospective_action"],d["current_evidence_state"]

def _pathi_family_summaries(rows):
    groups={}
    for r in rows or []:
        if str(r.get("source"))!="PATHI_SYSTEM": continue
        fid=str(r.get("pathi_family_id") or r.get("system_id") or "")
        groups.setdefault(fid,[]).append(r)
    out=[]
    for fid,members in sorted(groups.items()):
        ordered=sorted(
            members,
            key=lambda x:(int(x.get("evidence_rank") or 0), float(x.get("validation_hit_rate") or 0.0), int(x.get("validation_n") or 0)),
            reverse=True,
        )
        leader=ordered[0] if ordered else {}
        elig=[x for x in ordered if bool(x.get("normalized_vote_eligible"))]
        out.append({
            "family_id":fid,
            "family_label":PATHI_FAMILY_LABELS.get(fid,fid.replace("PATHI_","").replace("_"," ").title()),
            "member_system_ids":[str(x.get("system_id")) for x in members],
            "member_count":len(members),
            "leader_system_id":leader.get("system_id"),
            "leader_evidence_level":leader.get("evidence_level"),
            "normalized_vote_eligible":bool(elig),
            "eligible_member_ids":[str(x.get("system_id")) for x in elig],
            "correlation_policy":"ONE_PATHI_FAMILY_ONE_VOTE__MIRRORS_AND_NESTED_VARIANTS_NOT_INDEPENDENT",
            "production_authority":0,
        })
    return out


def _build_rules_index(mechanism_families,bigal_report,bigal_open,pathi_report,academic):
    rows=[]
    for fam in mechanism_families or []:
        recs=fam.get("representative_records") or {}
        rows.append({
            "system_id":fam.get("system_family_id"), "name":FAMILY_DISPLAY_NAMES.get(fam.get("system_family_id"), str(fam.get("system_family_id") or "").replace("NFL_","").replace("_"," ").title()),
            "source":"MINED", "source_class":"DATA_MINED_MECHANISM_FAMILY", "market":fam.get("market"), "action":fam.get("direction"),
            "rule_text":_rule_text(fam.get("representative_conditions") or [],fam.get("direction")), "machine_conditions":fam.get("representative_conditions") or [],
            "status":fam.get("family_status"), "prospective_action":fam.get("prospective_action"), "prospective_eligible":bool(fam.get("prospective_eligible")),
            "historical_discovery_status":"DISCOVERY_EDGE_ESTABLISHED" if int(fam.get("historical_discovery_member_count") or 0)>0 else "DISCOVERY_HYPOTHESIS_ONLY",
            "current_evidence_state":" | ".join(fam.get("current_evidence_states") or []),
            "historical_discovery_retained":bool(int(fam.get("historical_discovery_member_count") or 0)>0),
            "shadow_tracking_target":bool(fam.get("shadow_tracking_target")),
            "discovery":_record_text(recs.get("discovery")), "shadow_2023":_record_text(recs.get("shadow")), "confirmation_2024":_record_text(recs.get("confirmation")),
            "final_2025":_record_text(recs.get("final_2025")), "validation_2023_2025":_record_text(recs.get("validation_2023_2025")),
            "member_count":fam.get("member_count"), "representative_rule_id":fam.get("representative_rule_id"), "production_authority":0,
            "live_scoring":"PROSPECTIVE_FAMILY" if fam.get("prospective_eligible") else ("SHADOW_TRACK_TARGET" if fam.get("shadow_tracking_target") else "RESEARCH_INDEX_ONLY"),
        })
    for sid in BIGAL_EXPECTED_IDS:
        meta=dict((bigal_report or {}).get(sid) or {})
        op=dict((bigal_open or {}).get(sid) or {})
        rows.append({
            "system_id":sid, "name":meta.get("name") or sid, "source":"BIG_AL_PUBLISHED", "source_class":meta.get("source_class") or "HISTORICAL_DATABASE_SYSTEM",
            "market":"SPREADS" if sid!="BA-NFL5" else "TOTALS", "action":"EXTERNAL_RULE", "rule_text":meta.get("implementation") or meta.get("status") or sid,
            "machine_conditions":[], "status":meta.get("status") or meta.get("role") or "DESCRIPTIVE_REPLICATION", "prospective_action":"EXTERNAL_TRACK_ONLY", "prospective_eligible":False,
            "discovery":"—", "shadow_2023":"—", "confirmation_2024":"—", "final_2025":"—", "validation_2023_2025":_record_text({"n":op.get("n"),"wins":op.get("wins"),"rate":op.get("rate")}) if op else "—",
            "published_record":meta.get("claimed_record"), "source_url":meta.get("source_url"), "production_authority":0,
            "live_scoring":"UNAVAILABLE_NO_PRESEASON_HISTORY" if sid in {"BA-NFL4","BA-NFL5"} else "EXTERNAL_SHADOW",
        })
    for sid,meta0 in sorted((pathi_report or {}).items()):
        meta=dict(meta0 or {})
        discovery=_period_record(meta,(2017,2018,2019,2020,2021,2022))
        validation=_period_record(meta,(2023,2024,2025))
        overall={"n":int(meta.get("n") or 0),"wins":int(meta.get("wins") or 0),"losses":int(meta.get("losses") or 0),"pushes":int(meta.get("pushes") or 0),"hit_rate":meta.get("hit_rate")}
        ed=_pathi_evidence_detail(meta)
        status=ed["status"]; next_step=ed["prospective_action"]; current_state=ed["current_evidence_state"]
        family_id=PATHI_FAMILY_MAP.get(sid,sid)
        rows.append({
            "system_id":sid,
            "name":sid.replace("Pathi_FB_","").replace("_"," ").title(),
            "source":"PATHI_SYSTEM",
            "source_class":"PATHI_ENGINEERING_TRANSLATION_NOT_VERBATIM_PUBLISHED_FORMULA",
            "market":"SPREADS",
            "action":"PLAY_ON",
            "rule_text":meta.get("rule") or sid,
            "machine_conditions":[],
            "status":status,
            "prospective_action":next_step,
            "prospective_eligible":False,
            "historical_discovery_status":"HISTORICAL_RECORD_MEASURED",
            "current_evidence_state":current_state,
            "historical_discovery_retained":bool(discovery.get("n")),
            "shadow_tracking_target":True,
            "discovery":_record_text(discovery),
            "shadow_2023":_record_text(_period_record(meta,(2023,))),
            "confirmation_2024":_record_text(_period_record(meta,(2024,))),
            "final_2025":_record_text(_period_record(meta,(2025,))),
            "validation_2023_2025":_record_text(validation),
            "overall_2017_2025":_record_text(overall),
            "validation_n":int(validation.get("n") or 0),
            "validation_hit_rate":validation.get("hit_rate"),
            "pathi_family_id":family_id,
            "pathi_family_label":PATHI_FAMILY_LABELS.get(family_id,family_id.replace("PATHI_","").replace("_"," ").title()),
            "pathi_member_role":_pathi_member_role(sid),
            "evidence_level":ed["evidence_level"],
            "evidence_rank":ed["evidence_rank"],
            "major_regime_reversal":bool(ed["major_regime_reversal"]),
            "normalized_vote_eligible":bool(ed["normalized_vote_eligible"]),
            "correlation_policy":"ONE_PATHI_FAMILY_ONE_VOTE__MIRRORS_AND_NESTED_VARIANTS_NOT_INDEPENDENT",
            "production_authority":0,
            "overlay_authority":"SUPPORT_ONLY" if ed["normalized_vote_eligible"] else "NONE",
            "live_scoring":"SYSTEM_OVERLAY_SHADOW",
        })
    for sid,meta0 in sorted((academic or {}).items()):
        meta=dict(meta0 or {})
        rows.append({
            "system_id":sid,"name":sid.replace("ACADEMIC_","").replace("_"," ").title(),"source":"ACADEMIC","source_class":"REPLICATION_HYPOTHESIS","market":"RESEARCH",
            "action":"HYPOTHESIS","rule_text":meta.get("hypothesis") or sid,"machine_conditions":[],"status":meta.get("status") or "REPLICATION_HYPOTHESIS","prospective_action":"RESEARCH_ONLY",
            "prospective_eligible":False,"discovery":"—","shadow_2023":"—","confirmation_2024":"—","final_2025":"—","validation_2023_2025":"—","production_authority":0,"live_scoring":"NO",
        })
    return {
        "source_tag":SOURCE_TAG,
        "index_version":"NFL_SYSTEM_RULES_INDEX_V1_2_ADVANCED_MINER",
        "history_through":2025,
        "year_2026_queried":False,
        "rows":rows,
        "row_count":len(rows),
        "pathi_families":_pathi_family_summaries(rows),
        "pathi_vote_policy":"ONE_PATHI_FAMILY_ONE_VOTE__ONE_PATHI_LANE_VOTE_PER_GAME__NO_PRODUCTION_AUTHORITY",
        "production_authority":0,
    }


def _coverage_audit(rules_index,bigal_report,mechanism_families):
    bigal_present=sorted(set((bigal_report or {}).keys()))
    missing=[x for x in BIGAL_EXPECTED_IDS if x not in bigal_present]
    mined_ids=sorted(str(f.get("system_family_id")) for f in (mechanism_families or []))
    market_exec={
        "current_executable_price":{"status":"IMPLEMENTED_PRODUCTION","fields":["selected_price","selected_book"]},
        "quote_age":{"status":"IMPLEMENTED_PRODUCTION","fields":["quote_age_minutes"]},
        "market_direction_context":{"status":"IMPLEMENTED_SHADOW","fields":["market_state","market_move_toward_model"]},
        "opening_vs_current":{"status":"AVAILABLE_CONTEXT_NOT_CORE_PREDICTOR","fields":["Opening_Spread","current market"]},
        "key_crossing_topology":{"status":"PARTIAL_RESEARCH","fields":["Pathi key-crossing contexts"]},
        "book_lead_lag":{"status":"NOT_YET_VALIDATED_PROSPECTIVELY","fields":[]},
        "timing_policy":{"status":"NOT_YET_PROMOTED","fields":[]},
    }
    return {
        "status":"PASS" if not missing else "HOLD_MISSING_EXTERNAL_RULES", "source_tag":SOURCE_TAG,
        "bigal_expected_ids":list(BIGAL_EXPECTED_IDS),"bigal_present_ids":bigal_present,"bigal_missing_ids":missing,
        "pathi_framework_coverage":PATHI_COVERAGE,"mined_family_count":len(mined_ids),"mined_family_ids":mined_ids,
        "rules_index_rows":int((rules_index or {}).get("row_count") or 0),"market_execution_coverage":market_exec,
        "miner_anatomy_coverage":{
            "multi_game_sequences":"IMPLEMENTED_HORIZON_SYMMETRIC_LOOKBACK_1_2_3_DEPTH_CAP_3",
            "lookback_1":"IMPLEMENTED_RESULT_ATS_ROLE_LOCATION_MAGNITUDE_OPPONENT_STATE",
            "lookback_2":"IMPLEMENTED_EXACT_SU_ATS_LOCATION_ROLE_PLUS_MAGNITUDE_TREND_AND_OPPONENT_SYMMETRY",
            "lookback_3":"IMPLEMENTED_EXACT_SU_ATS_LOCATION_ROLE_PLUS_MAGNITUDE_TREND_AND_OPPONENT_SYMMETRY",
            "horizon_symmetry":"PASS_LOOKBACK_1_2_3",
            "location_sequences":"IMPLEMENTED_EXACT_PRIOR_TWO_AND_THREE_HOME_ROAD_SEQUENCE",
            "scoring_sequences":"IMPLEMENTED_PRIOR_ONE_TWO_THREE_SCORING_DEFENSE_AND_MAGNITUDE",
            "prior_opponent_quality":"IMPLEMENTED_PRIOR_ONE_TWO_THREE_QUALITY_STATE_SEQUENCE_AND_TREND",
            "ats_sequences":"IMPLEMENTED_EXACT_PRIOR_TWO_AND_THREE_ATS_SEQUENCE_PLUS_SINGLE_GAME_STATE",
            "role_sequences":"IMPLEMENTED_EXACT_PRIOR_TWO_AND_THREE_FAVORITE_DOG_SEQUENCE_PLUS_SINGLE_GAME_STATE",
            "magnitude_atoms":"IMPLEMENTED_LOOKBACK_1_2_3_SU_AND_ATS_COVER_MAGNITUDE",
            "opponent_sequence_symmetry":"IMPLEMENTED_LOOKBACK_1_2_3_TEAM_VS_OPPONENT_COMPARISON",
            "current_opponent_relationship":"IMPLEMENTED_DIVISION_CONFERENCE_INTERCONFERENCE_REMATCH",
            "primetime":"IMPLEMENTED",
            "island_games":"IMPLEMENTED_AS_UNIQUE_DATE_HOUR_KICKOFF_SLOT",
            "team_context_history":"IMPLEMENTED_PRIOR_ONLY_ACROSS_SEASONS_PLUS_SEQUENCE_MEMORY",
            "exact_team_identity":"IMPLEMENTED_SEPARATE_RESEARCH_LANE_WITH_CROSS_SEASON_STABILITY_GATE",
            "sequence_depth_cap":MAX_SEQUENCE_DEPTH,
            "production_authority":0,
        },
        "production_authority":0,"automatic_promotion":False,"year_2026_queried":False,
    }


def _write_rules_pointer(storage_client,bucket_name,payload:dict):
    key="nfl-research/v2_0/system_lab/latest_rules_index_pointer_v1.json"
    blob=storage_client.bucket(bucket_name).blob(key); blob.cache_control="no-store"
    blob.upload_from_string(json.dumps(payload,sort_keys=True,indent=2).encode(),content_type="application/json")
    return f"gs://{bucket_name}/{key}"


def read_rules_index(*,storage_client,bucket_name="sharp-models"):
    key="nfl-research/v2_0/system_lab/latest_rules_index_pointer_v1.json"
    b=storage_client.bucket(bucket_name).blob(key)
    if not b.exists(): return {"status":"RULES_INDEX_UNAVAILABLE","rows":[]}
    ptr=json.loads(b.download_as_text())
    uri=str(ptr.get("rules_index_uri") or "")
    prefix=f"gs://{bucket_name}/"
    if not uri.startswith(prefix): return {"status":"RULES_INDEX_POINTER_INVALID","rows":[]}
    obj=storage_client.bucket(bucket_name).blob(uri[len(prefix):])
    if not obj.exists(): return {"status":"RULES_INDEX_OBJECT_MISSING","rows":[]}
    out=json.loads(obj.download_as_text()); out["pointer"]=ptr; return out


def _write_family_pointer(storage_client,bucket_name,payload:dict):
    key="nfl-research/v2_0/system_lab/latest_family_registry_pointer_v3.json"
    blob=storage_client.bucket(bucket_name).blob(key)
    blob.cache_control="no-store"
    blob.upload_from_string(json.dumps(payload,sort_keys=True,indent=2).encode(),content_type="application/json")
    return f"gs://{bucket_name}/{key}"

def _metrics_status(series:pd.Series,positive_label:str)->dict:
    s=series.astype(str).str.upper(); valid=s.isin([positive_label,"WIN" if positive_label=="LOSS" else "LOSS","OVER" if positive_label=="UNDER" else "UNDER"])
    # specialized helpers below are clearer; keep this generic function unused.
    return {"n":int(valid.sum())}


def _academic_replications(state:pd.DataFrame)->dict:
    # One deterministic physical-game orientation for game-level comparisons.
    g=_home_or_neutral_one_side(state)
    out={k:dict(v) for k,v in ACADEMIC_REGISTRY.items()}
    home=g.loc[_num(g,"Is_Home").eq(1)].copy(); div=_num(home,"Is_Division_Game").eq(1)
    def ats_rate(p):
        q=p.Opening_ATS_Status.astype(str); n=int(q.isin(["WIN","LOSS"]).sum()); w=int((q=="WIN").sum()); return {"n":n,"rate":round(w/n,6) if n else None,"wins":w,"wilson95":[round(x,6) for x in _wilson(w,n)] if n else [None,None]}
    out["ACADEMIC_DIVISION_HOME_ATS"]["our_replication"]={"division_home":ats_rate(home.loc[div]),"nondivision_home":ats_rate(home.loc[~div]),"note":"Descriptive replication; no claim that original paper specification is exactly reproduced."}
    def under_rate(p):
        q=p.Opening_Total_Status.astype(str); n=int(q.isin(["OVER","UNDER"]).sum()); w=int((q=="UNDER").sum()); return {"n":n,"under_rate":round(w/n,6) if n else None,"unders":w,"wilson95":[round(x,6) for x in _wilson(w,n)] if n else [None,None]}
    out["ACADEMIC_DIVISION_TOTAL_UNDER"]["our_replication"]={"division":under_rate(g.loc[_num(g,"Is_Division_Game").eq(1)]),"nondivision":under_rate(g.loc[_num(g,"Is_Division_Game").eq(0)]),"note":"Descriptive replication using opening total; no ROI claim."}
    rest=_num(state,"Rest_Differential_Days"); adv=state.loc[rest.ge(2)].copy()
    out["ACADEMIC_REST_ERA_CHANGE"]["our_replication"]={"rest_advantage_2_plus":ats_rate(adv),"by_season":{str(int(sy)):ats_rate(p) for sy,p in adv.groupby("Season")},"note":"Current dataset begins 2017, so this tests modern-period behavior only; it cannot reproduce pre-2011 comparison."}
    return out


def _opening_retest_from_bigal(state:pd.DataFrame,bigal_plays:pd.DataFrame)->dict:
    out={}
    if bigal_plays.empty:return out
    key=state[["physical_game_id","Team_Norm","Opening_ATS_Status","Season"]].copy(); key["Team_Norm"]=key.Team_Norm.astype(str)
    for sid,p in bigal_plays.groupby("system_id"):
        x=p.merge(key,left_on=["physical_game_id","bet_team"],right_on=["physical_game_id","Team_Norm"],how="left",validate="many_to_one")
        q=x.Opening_ATS_Status.astype(str); n=int(q.isin(["WIN","LOSS"]).sum());w=int((q=="WIN").sum()); out[sid]={"n":n,"wins":w,"rate":round(w/n,6) if n else None,"wilson95":[round(z,6) for z in _wilson(w,n)] if n else [None,None],"by_season":{str(int(sy)):{"n":int(z.Opening_ATS_Status.isin(["WIN","LOSS"]).sum()),"rate":round(float(z.Opening_ATS_Status.eq("WIN").sum()/z.Opening_ATS_Status.isin(["WIN","LOSS"]).sum()),6) if z.Opening_ATS_Status.isin(["WIN","LOSS"]).sum() else None} for sy,z in x.groupby("Season")},"role":"OPENING_LINE_RETROSPECTIVE_RETEST_NO_ROI_CLAIM"}
    return out


def _cross_sport_preregistered_retests(state:pd.DataFrame)->dict:
    """Test externally specified structures that were discovered outside NFL.

    The primary rule below was frozen in the NCAAF System Miner V3 work as
    PLAY_ON: DOG_10_PLUS + OFF_BLOWOUT_LOSS_14_PLUS + BACK_TO_BACK_ATS_LOSSES.
    Because its structure/direction came from a different sport, NFL history is
    treated as a cross-sport transfer/replication test, not as NFL discovery.
    It still has zero production authority and no ROI claim.
    """
    op=_num(state,"Opening_Spread");
    masks={
      "NCAAF_V1344_TRANSFER__DOG10__BLOWOUTLOSS14__B2B_ATS_LOSSES": op.ge(10)&_num(state,"calc_prev1_margin").le(-14)&_num(state,"calc_b2b_open_ats_losses").eq(1),
      "NCAAF_V1344_TRANSFER__DOG10__BLOWOUTLOSS14__OFF_ATS_LOSS": op.ge(10)&_num(state,"calc_prev1_margin").le(-14)&_num(state,"calc_prev1_open_ats_loss").eq(1),
      "NCAAF_V1344_TRANSFER__DOG10__BLOWOUTLOSS14": op.ge(10)&_num(state,"calc_prev1_margin").le(-14),
    }
    out={}
    labels=np.where(state.Opening_ATS_Status.eq("WIN"),1.0,np.where(state.Opening_ATS_Status.eq("LOSS"),0.0,np.nan))
    for name,mask in masks.items():
        ix,amb=_candidate_rows_side(state,pd.Series(mask,index=state.index).fillna(False).to_numpy(bool))
        rec=_rec(labels,ix,"PLAY_ON")
        by={str(int(sy)):_rec(labels,ix[pd.to_numeric(state.Season,errors="coerce").to_numpy()[ix]==sy],"PLAY_ON") for sy in sorted(state.Season.unique())}
        out[name]={"direction":"PLAY_ON","origin":"NCAAF_V13_4_4_SYSTEM_MINER_V3_EXTERNAL_TO_NFL","status":"CROSS_SPORT_REPLICATION_ONLY","all_nfl_history":rec,"by_season":by,"ambiguous_both_sides":amb,"opening_line_target":True,"production_authority":0}
    return out



# -------------------------- V3.10.3 research attribution publisher --------------------------
_BIGAL_INDEPENDENCE_FAMILY={
    "BA-NFL1":"BIGAL_BA_NFL1", "BA-NFL1-HOME":"BIGAL_BA_NFL1",
    "BA-NFL2":"BIGAL_BA_NFL2", "BA-NFL2-ATS":"BIGAL_BA_NFL2",
    "BA-NFL3":"BIGAL_BA_NFL3", "BA-NFL6":"BIGAL_BA_NFL6",
}
_BIGAL_CANONICAL={"BA-NFL1","BA-NFL2","BA-NFL3","BA-NFL6"}


def _flat110_status_record(statuses):
    q=[str(x or "").upper() for x in list(statuses or [])]
    w=sum(x=="WIN" for x in q); l=sum(x=="LOSS" for x in q); p=sum(x=="PUSH" for x in q)
    graded=w+l; total=graded+p
    hit=(w/graded) if graded else None
    roi=((w*(100.0/110.0)-l)/graded) if graded else None
    ci=_wilson(w,graded) if graded else (None,None)
    return {
        "n":int(total),"graded_n":int(graded),"wins":int(w),"losses":int(l),"pushes":int(p),
        "hit_rate":round(float(hit),6) if hit is not None else None,
        "flat_minus110_roi_reference":round(float(roi),6) if roi is not None else None,
        "wilson95":[round(float(ci[0]),6),round(float(ci[1]),6)] if graded else [None,None],
    }


def _invert_ats_status(x):
    z=str(x or "").upper()
    return "LOSS" if z=="WIN" else "WIN" if z=="LOSS" else "PUSH" if z=="PUSH" else ""


def _research_system_attribution(state,spread_atoms,mechanism_families,bigal_plays,pathi_plays,rules_index,pt_external_mechanisms=None):
    """Publish normalized 2023-25 source/confluence attribution for research display only.

    One Miner mechanism family is one vote. Pathi mirror/nested members collapse to the
    normalized Pathi family leader. Big Al nested variants remain visible diagnostically,
    while confluence uses canonical independent systems only. This is historical research
    attribution, not prospective evidence and not production authority.
    """
    if not isinstance(state,pd.DataFrame) or state.empty:
        return {"status":"UNAVAILABLE_EMPTY_STATE","production_authority":0,"year_2026_queried":False}
    d=state.copy()
    d["Season"]=pd.to_numeric(d.get("Season"),errors="coerce")
    d=d.loc[d.Season.isin([2023,2024,2025])].copy()
    if d.empty:
        return {"status":"UNAVAILABLE_NO_2023_2025","production_authority":0,"year_2026_queried":False}
    d["physical_game_id"]=d.physical_game_id.astype(str)
    d["Team_Norm"]=d.Team_Norm.astype(str)
    d["Opponent_Norm"]=d.Opponent_Norm.astype(str)
    season_by_game=d.groupby("physical_game_id",sort=False)["Season"].first().astype(int).to_dict()
    status_by_game_team={(str(r.physical_game_id),str(r.Team_Norm)):str(r.Opening_ATS_Status).upper() for _,r in d.iterrows()}

    rule_rows=list((rules_index or {}).get("rows") or [])
    row_by_id={str(r.get("system_id")):r for r in rule_rows if r.get("system_id")}
    pathi_summaries=_pathi_family_summaries(rule_rows)
    pathi_family_by_sid={}
    pathi_leader_by_family={}
    pathi_eligible_family={}
    for f in pathi_summaries:
        fid=str(f.get("family_id") or "")
        if not fid: continue
        pathi_leader_by_family[fid]=str(f.get("leader_system_id") or "")
        pathi_eligible_family[fid]=bool(f.get("normalized_vote_eligible"))
        for sid in f.get("member_system_ids") or []: pathi_family_by_sid[str(sid)]=fid

    triggers=[]
    def add_trigger(source,system_id,family_id,gid,bet_team,status,qualified_current,evidence=None,display_name=None,normalized=True):
        gid=str(gid); team=str(bet_team or ""); st=str(status or "").upper()
        sy=season_by_game.get(gid)
        if sy not in {2023,2024,2025} or st not in {"WIN","LOSS","PUSH"} or not team:return
        triggers.append({
            "source":str(source),"system_id":str(system_id),"independence_family":str(family_id),
            "physical_game_id":gid,"season":int(sy),"bet_team":team,"status":st,
            "qualified_current":bool(qualified_current),"normalized":bool(normalized),
            "evidence":str(evidence or ""),"name":str(display_name or system_id),
        })

    # Big Al: all playable systems are kept for ranking; only canonical base systems
    # are normalized confluence votes so HOME/ATS nested variants cannot double count.
    if isinstance(bigal_plays,pd.DataFrame) and not bigal_plays.empty:
        for _,r in bigal_plays.iterrows():
            sid=str(r.get("system_id") or ""); fam=_BIGAL_INDEPENDENCE_FAMILY.get(sid,sid)
            rr=row_by_id.get(sid) or {}; name=rr.get("name") or sid
            add_trigger("BIG AL",sid,fam,r.get("physical_game_id"),r.get("bet_team"),r.get("status"),False,
                        evidence=rr.get("status"),display_name=name,normalized=(sid in _BIGAL_CANONICAL))

    # Pathi: preserve every member for diagnostics but normalize each mirror/nested
    # family to its evidence-ranked leader. Only WEAK/STRONG family leaders qualify
    # for the current bounded overlay lane.
    if isinstance(pathi_plays,pd.DataFrame) and not pathi_plays.empty:
        for _,r in pathi_plays.iterrows():
            sid=str(r.get("system_id") or ""); fid=pathi_family_by_sid.get(sid,sid); rr=row_by_id.get(sid) or {}
            leader=pathi_leader_by_family.get(fid,sid); norm=(sid==leader)
            add_trigger("PATHI",sid,fid,r.get("physical_game_id"),r.get("bet_team"),r.get("status"),
                        bool(norm and pathi_eligible_family.get(fid,False)),evidence=rr.get("evidence_level") or rr.get("status"),
                        display_name=rr.get("name") or sid,normalized=norm)

    # Miner: reconstruct each mechanism-family representative on the same frozen state.
    atom_by={str(a.get("name")):a for a in (spread_atoms or [])}
    for fam in mechanism_families or []:
        if str(fam.get("market") or "").upper()!="SPREADS":continue
        fid=str(fam.get("system_family_id") or ""); conds=list(fam.get("representative_conditions") or [])
        if not fid or not conds or any(c not in atom_by for c in conds):continue
        mm=np.ones(len(state),dtype=bool)
        for c in conds:mm &= np.asarray(atom_by[c]["mask"],bool)
        ix,amb=_candidate_rows_side(state,mm)
        # Representative families chosen from clean members should not be ambiguous;
        # if a future family is ambiguous, omit those games rather than inventing a side.
        direction=str(fam.get("direction") or "PLAY_ON").upper()
        vr=((fam.get("representative_records") or {}).get("validation_2023_2025") or {})
        qualified=bool(
            not bool(fam.get("expert_bridge"))
            and str(fam.get("family_status") or "") in {"LEGIT_FAMILY_REQUIRES_PROSPECTIVE","PROMISING_FAMILY","HISTORICAL_EDGE_WEAKENED_FAMILY"}
            and int(vr.get("n") or 0)>=24 and vr.get("rate") is not None and float(vr.get("rate"))>BREAK_EVEN_REFERENCE
        )
        rr=row_by_id.get(fid) or {}; name=rr.get("name") or fid
        for i in ix:
            r=state.iloc[int(i)]; sy=int(pd.to_numeric(pd.Series([r.get("Season")]),errors="coerce").iloc[0]) if pd.notna(r.get("Season")) else -1
            if sy not in {2023,2024,2025}:continue
            play_on=(direction=="PLAY_ON")
            bet_team=str(r.get("Team_Norm") if play_on else r.get("Opponent_Norm"))
            st=str(r.get("Opening_ATS_Status") or "").upper(); st=st if play_on else _invert_ats_status(st)
            add_trigger("MINER",fid,fid,r.get("physical_game_id"),bet_team,st,qualified,evidence=fam.get("family_status"),display_name=name,normalized=True)

    # PT-derived Miner systems use the same historical qualification standard but
    # stay in one correlated external-ratings family per market.  They are added
    # directly to the normalized registry; raw PT ratings themselves retain zero
    # production/model authority.  Live trigger evaluation occurs in Model Authority.
    pt_rank_rows=[]
    for mech in pt_external_mechanisms or []:
        market=str(mech.get("market") or "").upper()
        vr=((mech.get("records") or {}).get("validation_2023_2025") or {})
        try:
            n=int(vr.get("n") or 0); w=int(vr.get("wins") or 0)
        except Exception:
            n=0; w=0
        l=int(vr.get("losses") if vr.get("losses") is not None else max(n-w,0)) if isinstance(vr,dict) else max(n-w,0)
        psh=int(vr.get("pushes") or 0) if isinstance(vr,dict) else 0
        graded=max(w+l,0)
        hit=(w/graded) if graded else (float(vr.get("rate")) if isinstance(vr,dict) and vr.get("rate") is not None else None)
        ci=_wilson(w,graded) if graded else (None,None)
        pt_rank_rows.append({
            "source":"PT","system_id":str(mech.get("system_id") or mech.get("external_mechanism_id") or ""),
            "family_id":str(mech.get("evidence_family_key") or mech.get("family_id") or ""),
            "name":"PT-derived Miner: "+" + ".join(str(x) for x in (mech.get("conditions") or [])),
            "evidence":str(mech.get("status") or ""),"qualified_current":bool(mech.get("qualified_current")),
            "market":market,"direction":str(mech.get("direction") or "").upper(),
            "representative_conditions":list(mech.get("conditions") or []),
            "n":int(n+psh),"graded_n":int(graded),"wins":int(w),"losses":int(l),"pushes":int(psh),
            "hit_rate":round(float(hit),6) if hit is not None else None,
            "flat_minus110_roi_reference":round(float(((w*(100.0/110.0)-l)/graded)),6) if graded else None,
            "wilson95":[round(float(ci[0]),6),round(float(ci[1]),6)] if graded else [None,None],
            "origin_brain":"PREDICTION_TRACKER","pt_model_weight":0.0,"pt_family_vote_cap":1,
        })

    # Calculate independent-system records first, then historical qualification for Big Al.
    records=[]
    family_rows={}
    for t in triggers:
        if not t.get("normalized"):continue
        key=(t["source"],t["independence_family"],t["system_id"])
        family_rows.setdefault(key,[]).append(t)
    qualified_family={}
    for (src,fid,sid),rows in family_rows.items():
        rec=_flat110_status_record([x["status"] for x in rows])
        q=bool(any(x.get("qualified_current") for x in rows))
        if src=="BIG AL":
            q=bool(rec.get("graded_n",0)>=20 and rec.get("hit_rate") is not None and float(rec["hit_rate"])>BREAK_EVEN_REFERENCE)
        qualified_family[(src,fid)]=q
        first=rows[0]
        records.append({
            "source":src,"system_id":sid,"family_id":fid,"name":first.get("name") or sid,"evidence":first.get("evidence") or "",
            "qualified_current":q,**rec,
        })

    def _resolve(scope_qualified):
        use=[t for t in triggers if t.get("normalized") and (not scope_qualified or qualified_family.get((t["source"],t["independence_family"]),False))]
        by_game={}
        for t in use:
            by_game.setdefault(t["physical_game_id"],[]).append(t)
        settled=[]; internal_conflicts=0; cross_source_conflicts=0
        for gid,rows in by_game.items():
            fam_pick={}; fam_src={}
            for t in rows:
                fam_pick.setdefault(t["independence_family"],set()).add(t["bet_team"]); fam_src[t["independence_family"]]=t["source"]
            if any(len(v)!=1 for v in fam_pick.values()): internal_conflicts+=1; continue
            fam_team={k:next(iter(v)) for k,v in fam_pick.items()}
            source_teams={}
            for fam,team in fam_team.items():source_teams.setdefault(fam_src[fam],set()).add(team)
            if any(len(v)!=1 for v in source_teams.values()): internal_conflicts+=1; continue
            source_pick={s:next(iter(v)) for s,v in source_teams.items()}
            teams=set(source_pick.values())
            if len(teams)!=1: cross_source_conflicts+=1; continue
            team=next(iter(teams)); st=status_by_game_team.get((gid,team),"")
            if st not in {"WIN","LOSS","PUSH"}:continue
            sources=sorted(source_pick)
            settled.append({
                "physical_game_id":gid,"season":season_by_game.get(gid),"bet_team":team,"status":st,
                "sources":sources,"source_key":" + ".join(sources),"source_count":len(sources),
                "independent_family_count":len(fam_team),"family_ids":sorted(fam_team),
            })
        cats={}
        for z in settled:cats.setdefault(z["source_key"],[]).append(z["status"])
        source_combinations=[]
        desired=["BIG AL","MINER","PATHI","BIG AL + MINER","BIG AL + PATHI","MINER + PATHI","BIG AL + MINER + PATHI"]
        for k in desired:
            rec=_flat110_status_record(cats.get(k,[])); source_combinations.append({"combination":k,**rec})
        return {
            "resolved_games":int(len(settled)),"internal_conflict_games":int(internal_conflicts),"cross_source_conflict_games":int(cross_source_conflicts),
            "source_combinations":source_combinations,
            "two_plus_independent_families":_flat110_status_record([z["status"] for z in settled if int(z["independent_family_count"])>=2]),
            "three_plus_independent_families":_flat110_status_record([z["status"] for z in settled if int(z["independent_family_count"])>=3]),
            "two_plus_independent_sources":_flat110_status_record([z["status"] for z in settled if int(z["source_count"])>=2]),
            "three_independent_sources":_flat110_status_record([z["status"] for z in settled if int(z["source_count"])>=3]),
            "_settled":settled,
        }

    all_scope=_resolve(False); qualified_scope=_resolve(True)

    # Rank systems conservatively by validation Wilson lower bound, not raw hit rate.
    # Also show how each family performs when it is part of a resolved 2+ family confluence game.
    multi_games={z["physical_game_id"]:z for z in qualified_scope.get("_settled",[]) if int(z.get("independent_family_count") or 0)>=2}
    for rec in records:
        fam=rec["family_id"]; src=rec["source"]
        rows=[t for t in triggers if t.get("normalized") and t["source"]==src and t["independence_family"]==fam and t["physical_game_id"] in multi_games and multi_games[t["physical_game_id"]]["bet_team"]==t["bet_team"]]
        mr=_flat110_status_record([t["status"] for t in rows])
        rec["multi_confluence"]=mr
    records.extend(pt_rank_rows)
    eligible_rank=[r for r in records if int(r.get("graded_n") or 0)>=20]
    eligible_rank=sorted(eligible_rank,key=lambda r:((r.get("wilson95") or [None])[0] if (r.get("wilson95") or [None])[0] is not None else -1,float(r.get("hit_rate") or 0),int(r.get("graded_n") or 0)),reverse=True)
    for i,r in enumerate(eligible_rank,1):r["validation_rank"]=i
    rank_map={(r["source"],r["family_id"]):r.get("validation_rank") for r in eligible_rank}
    for r in records:r["validation_rank"]=rank_map.get((r["source"],r["family_id"]))
    records=sorted(records,key=lambda r:(r.get("validation_rank") is None,r.get("validation_rank") or 9999,r["source"],r["family_id"]))
    for scope in (all_scope,qualified_scope):scope.pop("_settled",None)

    return {
        "status":"READY","window":"2023-2025","market":"SPREADS",
        "methodology":"SOURCE_NEUTRAL_SYSTEM_QUALIFICATION__ONE_NORMALIZED_INDEPENDENT_FAMILY_ONE_VOTE__PT_EXTERNAL_FAMILY_CAP_1__FLAT_MINUS110_REFERENCE_ONLY",
        "selection_caveat":"MINER/PATHI current qualification uses historical validation evidence; descriptive research attribution only, not untouched prospective proof.",
        "qualified_current":qualified_scope,"all_normalized_research":all_scope,"system_rankings":records,
        "ranking_method":"MIN_VALIDATION_GRADED_N_20__WILSON95_LOWER_BOUND_THEN_HIT_RATE_THEN_SAMPLE",
        "production_authority":0,"automatic_promotion":False,"year_2026_queried":False,
    }


def _upload_immutable(storage_client,bucket_name,name,data:bytes,content_type:str):
    from google.api_core.exceptions import PreconditionFailed
    blob=storage_client.bucket(bucket_name).blob(name)
    try:
        blob.upload_from_string(data,content_type=content_type,if_generation_match=0);return {"uri":f"gs://{bucket_name}/{name}","created":True}
    except PreconditionFailed:return {"uri":f"gs://{bucket_name}/{name}","created":False,"reason":"ALREADY_EXISTS_IMMUTABLE"}


def run_nfl_system_lab_v3(*,bq_client,storage_client,bucket_name="sharp-models",audit_report=None,log_func=print):
    if not isinstance(audit_report,dict) or audit_report.get("status")!="READY_FOR_OFFLINE_CHALLENGER_SANDBOX":raise RuntimeError("NFL_RESEARCH_V2_SYSTEM_AUDIT_NOT_GREEN")
    contract=assert_contract(); view_cols={f.name for f in bq_client.get_table(VIEW).schema}; side=bq_client.query(build_intelligence_query(view_cols)).to_dataframe(create_bqstorage_client=False)
    if int(pd.to_numeric(side.Season,errors="coerce").max())>2025:raise RuntimeError("NFL_RESEARCH_V2_SYSTEM_2026_QUERY_LEAK")
    state=prepare_system_state(side)
    log_func(f"[NFL-RESEARCH-V2-SYSTEM-PREFLIGHT] status=PASS source_tag={SOURCE_TAG} data_through=2025 year_2026_queried=FALSE model_state_in_discovery=FALSE production_authority=0 contract_sha256={contract_hash()}")
    bigal_report,bigal_plays=_bigal_systems(state); bigal_open=_opening_retest_from_bigal(state,bigal_plays); pathi_report,pathi_plays=_pathi_engineering(state); academic=_academic_replications(state); cross_sport=_cross_sport_preregistered_retests(state)
    sp_label=np.where(state.Opening_ATS_Status.eq("WIN"),1.0,np.where(state.Opening_ATS_Status.eq("LOSS"),0.0,np.nan))
    spread_atoms=_spread_atoms(state)
    expert_atoms,expert_atom_bridge=_expert_occurrence_atom_bridge(state,bigal_plays,pathi_plays,log_func=log_func)
    spread_atoms.extend(expert_atoms)

    # LEGACY lane is intentionally identical to V3.11 except for the lane label.
    # Prediction Tracker atoms never enter this beam or its multiple-testing budget.
    spread_miner=_search_market(state,sp_label,spread_atoms,"SPREADS",True,log_func=log_func,lane="LEGACY")

    games=_home_or_neutral_one_side(state)
    total_label=np.where(games.Opening_Total_Status.eq("OVER"),1.0,np.where(games.Opening_Total_Status.eq("UNDER"),0.0,np.nan))
    total_atoms=_total_atoms(games)
    total_miner=_search_market(games,total_label,total_atoms,"TOTALS",False,log_func=log_func,lane="LEGACY")

    # Separate PT external-family lane. It can learn individual predictor behavior
    # and interact with bounded legacy/expert context, but cannot crowd out the
    # legacy Miner or create production authority.
    prediction_tracker_external=_pt_run_external_research(
        state=state,games=games,spread_label=sp_label,total_label=total_label,
        legacy_spread_atoms=spread_atoms,legacy_total_atoms=total_atoms,
        storage_client=storage_client,bucket_name=bucket_name,log_func=log_func,
    )

    mechanism_families=_collapse_mechanism_families(spread_miner,total_miner)
    family_payload=_family_registry_payload(mechanism_families)
    family_sha=hashlib.sha256(json.dumps(family_payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    family_payload["family_registry_sha256"]=family_sha
    family_prefix=f"nfl-research/v2_0/system_lab/mechanism_families/{family_sha[:16]}"
    family_upload=_upload_immutable(storage_client,bucket_name,f"{family_prefix}/family_registry.json",json.dumps(family_payload,sort_keys=True,indent=2).encode(),"application/json")
    family_uri=str(family_upload.get("uri") or "")
    if not family_uri.startswith(f"gs://{bucket_name}/"):
        raise RuntimeError("NFL_SYSTEM_FAMILY_V3_REGISTRY_UPLOAD_URI_INVALID")
    pointer={"family_registry_uri":family_uri,"family_registry_sha256":family_sha,"source_tag":SOURCE_TAG,"production_authority":0}
    pointer_uri=_write_family_pointer(storage_client,bucket_name,pointer)

    registry={
        "source_tag":SOURCE_TAG,"contract_sha256":contract_hash(),"discovery_seasons":list(DISCOVERY_SEASONS),"shadow_season":SHADOW_SEASON,"confirmation_season":CONFIRM_SEASON,"final_historical_check":FINAL_CHECK_SEASON,"year_2026_queried":False,
        "methodology":"NCAAF_SYSTEM_MINER_V3_DIRECT_BETTING_SYSTEM_DISCOVERY__NFL_V3_MECHANISM_FAMILY_CONSOLIDATION","model_state_in_discovery":False,"opening_line_primary_research_target":True,"historical_roi_claim":False,
        "bigal_system_ids":sorted(bigal_report),"pathi_status":"ENGINEERING_TRANSLATION_SHADOW_ONLY","academic_status":"REPLICATION_HYPOTHESES_ONLY","cross_sport_preregistered_ids":sorted(cross_sport),
        "spread_legit_rule_ids":[_rule_id(r) for r in spread_miner["legit_rules"]],"total_legit_rule_ids":[_rule_id(r) for r in total_miner["legit_rules"]],
        "spread_promising_rule_ids":[_rule_id(r) for r in spread_miner["promising_rules"]],"total_promising_rule_ids":[_rule_id(r) for r in total_miner["promising_rules"]],
        "spread_watch_rule_ids":[_rule_id(r) for r in spread_miner["watch_rules"]],"total_watch_rule_ids":[_rule_id(r) for r in total_miner["watch_rules"]],
        "spread_dormant_historical_rule_ids":[_rule_id(r) for r in spread_miner.get("dormant_historical_rules",[])],"total_dormant_historical_rule_ids":[_rule_id(r) for r in total_miner.get("dormant_historical_rules",[])],
        "spread_historical_discovery_retained_rule_ids":[_rule_id(r) for r in spread_miner.get("retained_rules",[]) if r.get("historical_discovery_retained")],"total_historical_discovery_retained_rule_ids":[_rule_id(r) for r in total_miner.get("retained_rules",[]) if r.get("historical_discovery_retained")],
        "mechanism_family_ids":[f["system_family_id"] for f in mechanism_families],
        "expert_atom_bridge_status":expert_atom_bridge.get("status"),
        "expert_atom_count":expert_atom_bridge.get("atom_count"),
        "expert_bridge_mechanism_family_ids":[f["system_family_id"] for f in mechanism_families if f.get("expert_bridge")],
        "expert_bridge_mechanism_count":sum(1 for f in mechanism_families if f.get("expert_bridge")),
        "expert_bridge_live_authority":0,
        "prediction_tracker_status":prediction_tracker_external.get("status"),
        "prediction_tracker_external_mechanism_count":prediction_tracker_external.get("external_mechanism_count",0),
        "prediction_tracker_spread_predictor_count":prediction_tracker_external.get("spread_predictor_count",0),
        "prediction_tracker_total_predictor_count":prediction_tracker_external.get("total_predictor_count",0),
        "prediction_tracker_correlation_policy":prediction_tracker_external.get("correlation_policy"),
        "prediction_tracker_prospective_eligible":bool(prediction_tracker_external.get("prospective_eligible")),
        "prediction_tracker_qualified_system_count":int(prediction_tracker_external.get("qualified_system_count") or 0),
        "prediction_tracker_system_authority_policy":"SOURCE_NEUTRAL_SAME_AS_MINER__ONE_PT_FAMILY_VOTE_MAX",
        "prediction_tracker_family_vote_cap":1,
        "prediction_tracker_model_weight":0.0,
        "prospective_family_ids":family_payload["prospective_family_ids"],
        "family_registry_uri":family_uri,"family_registry_sha256":family_sha,"family_registry_pointer_uri":pointer_uri,
        "status_count_contract":"EVIDENCE_LIFECYCLE__DISCOVERY_IS_PERMANENT_PROVENANCE__CURRENT_STATE_CONTROLS_AUTHORITY",
        "production_authority":0,"automatic_promotion":False,"prospective_clock":"PER_FAMILY_APPEND_ONLY_CLOCK",
    }
    sha=hashlib.sha256(json.dumps(registry,sort_keys=True,separators=(",",":")).encode()).hexdigest();registry["registry_sha256"]=sha
    rules_index=_build_rules_index(mechanism_families,bigal_report,bigal_open,pathi_report,academic)
    research_system_attribution=_research_system_attribution(state,spread_atoms,mechanism_families,bigal_plays,pathi_plays,rules_index,prediction_tracker_external.get("external_mechanisms") or [])
    coverage_audit=_coverage_audit(rules_index,bigal_report,mechanism_families)
    if coverage_audit.get("status")!="PASS": raise RuntimeError("NFL_SYSTEM_RULES_COVERAGE_HOLD "+str(coverage_audit.get("bigal_missing_ids")))
    report={"status":STATUS,"registry":registry,"mechanism_families":mechanism_families,"family_registry":family_payload,"expert_atom_bridge":expert_atom_bridge,"prediction_tracker_external":prediction_tracker_external,"rules_index":rules_index,"coverage_audit":coverage_audit,"research_system_attribution":research_system_attribution,"bigal":{"documented_close_reference":bigal_report,"opening_line_retest":bigal_open},"pathi_engineering":pathi_report,"academic_replications":academic,"cross_sport_preregistered_retests":cross_sport,"miner":{"spreads":spread_miner,"totals":total_miner,"prediction_tracker_spreads":((prediction_tracker_external.get("miner") or {}).get("spreads") or {}),"prediction_tracker_totals":((prediction_tracker_external.get("miner") or {}).get("totals") or {})},"production_authority":0,"year_2026_queried":False,"ncaaf":"UNCHANGED","legacy_nfl":"V3_11_LEGACY_LANES_PRESERVED"}
    pt_payload=json.dumps(prediction_tracker_external,sort_keys=True,default=str).encode()
    pt_sha=hashlib.sha256(pt_payload).hexdigest()
    pt_prefix=f"nfl-research/v2_0/prediction_tracker/{pt_sha[:16]}"
    pt_upload=_upload_immutable(storage_client,bucket_name,f"{pt_prefix}/prediction_tracker_report.json",pt_payload,"application/json")
    try:
        storage_client.bucket(bucket_name).blob("nfl-research/v2_0/prediction_tracker/current_report.json").upload_from_string(
            json.dumps({"report_uri":pt_upload.get("uri"),"sha256":pt_sha,"source_tag":SOURCE_TAG,"production_authority":0},sort_keys=True).encode(),
            content_type="application/json",
        )
    except Exception:
        pass
    prefix=f"nfl-research/v2_0/system_lab/{sha[:16]}"
    rules_bytes=json.dumps(rules_index,sort_keys=True,indent=2,default=str).encode()
    coverage_bytes=json.dumps(coverage_audit,sort_keys=True,indent=2,default=str).encode()
    rules_upload=_upload_immutable(storage_client,bucket_name,f"{prefix}/rules_index.json",rules_bytes,"application/json")
    coverage_upload=_upload_immutable(storage_client,bucket_name,f"{prefix}/coverage_audit.json",coverage_bytes,"application/json")
    rules_pointer_uri=_write_rules_pointer(storage_client,bucket_name,{"source_tag":SOURCE_TAG,"rules_index_uri":rules_upload.get("uri"),"coverage_audit_uri":coverage_upload.get("uri"),"family_registry_uri":family_uri,"family_registry_sha256":family_sha,"production_authority":0})
    arts={"registry":_upload_immutable(storage_client,bucket_name,f"{prefix}/system_registry.json",json.dumps(registry,sort_keys=True,indent=2).encode(),"application/json"),"report":_upload_immutable(storage_client,bucket_name,f"{prefix}/system_lab_report.json",json.dumps(report,sort_keys=True,default=str).encode(),"application/json"),"prediction_tracker":pt_upload,"family_registry":family_upload,"family_pointer":{"created":True,"uri":pointer_uri},"rules_index":rules_upload,"coverage_audit":coverage_upload,"rules_pointer":{"created":True,"uri":rules_pointer_uri}}
    for _market_name, _miner in (("SPREADS", spread_miner), ("TOTALS", total_miner)):
        for _r in _miner.get("retained_rules", []):
            _detail = {
                "market": _market_name,"rule_id": _rule_id(_r),"rank": _r.get("rank"),"conditions": _r.get("conditions", []),"families": _r.get("families", []),"direction": _r.get("direction"),
                "records": _r.get("records", {}),"discovery_year_by_year": _r.get("discovery_year_by_year", {}),"chronological_folds": _r.get("chronological_folds", {}),"min_fold_rate": _r.get("min_fold_rate"),
                "min_loso_rate": _r.get("min_loso_rate"),"remove_best_season": _r.get("remove_best_season"),"remove_best_season_record": _r.get("remove_best_season_record", {}),"structural_floor": _r.get("structural_floor"),
                "bootstrap_ci95": _r.get("bootstrap_ci95"),"nominal_pvalue": _r.get("nominal_pvalue"),"permutation_max_pvalue": _r.get("permutation_max_pvalue"),"global_fdr_qvalue": _r.get("global_fdr_qvalue"),
                "hierarchical_family_qvalue": _r.get("hierarchical_family_qvalue"),"within_family_qvalue": _r.get("within_family_qvalue"),"multiple_testing_pass": _r.get("multiple_testing_pass"),"team_specific": _r.get("team_specific"),"team_identity_season_stability":_r.get("team_identity_season_stability"),
                "ambiguous_both_sides": _r.get("ambiguous_both_sides"),"ambiguity_share":_ambiguity_share(_r),"expert_bridge":any(str(c).startswith("EXPERT_") for c in (_r.get("conditions") or [])),"expert_sources":sorted(set("PATHI" if str(c).startswith("EXPERT_PATHI_") else "BIGAL" if str(c).startswith("EXPERT_BIGAL_") else "" for c in (_r.get("conditions") or []))-{""}),"status": _r.get("status"),"historical_discovery_status":_r.get("historical_discovery_status"),"historical_discovery_retained":_r.get("historical_discovery_retained"),"current_evidence_state":_r.get("current_evidence_state"),"edge_decay_percentage_points":_r.get("edge_decay_percentage_points"),"mechanism_family_id":_mechanism_id(_r),"production_authority": 0,
            }
            log_func("[NFL-RESEARCH-V2-SYSTEM-RETAINED] "+json.dumps(_detail,sort_keys=True,default=str))
    for fam in mechanism_families:
        log_func("[NFL-RESEARCH-V2-SYSTEM-MECHANISM-FAMILY] "+json.dumps(fam,sort_keys=True,default=str))
    log_func("[NFL-RESEARCH-V2-SYSTEM-FAMILY-REGISTRY] "+json.dumps({"family_registry_sha256":family_sha,"family_registry_uri":family_uri,"family_registry_pointer_uri":pointer_uri,"prospective_family_ids":family_payload["prospective_family_ids"],"new_clock_family_ids":family_payload["new_clock_family_ids"],"continue_clock_family_ids":family_payload["continue_clock_family_ids"],"shadow_tracking_family_ids":family_payload.get("shadow_tracking_family_ids",[]),"production_authority":0},sort_keys=True))
    log_func("[NFL-RESEARCH-V2-SYSTEM-EXPERT-ATOM-BRIDGE] "+json.dumps({"status":expert_atom_bridge.get("status"),"atom_count":expert_atom_bridge.get("atom_count"),"pathi_atom_count":expert_atom_bridge.get("pathi_atom_count"),"bigal_atom_count":expert_atom_bridge.get("bigal_atom_count"),"cross_source_atom_count":expert_atom_bridge.get("cross_source_atom_count"),"expert_bridge_mechanisms":sum(1 for f in mechanism_families if f.get("expert_bridge")),"expert_bridge_family_ids":[f.get("system_family_id") for f in mechanism_families if f.get("expert_bridge")],"live_authority":0,"production_authority":0},sort_keys=True,default=str))
    log_func("[NFL-RESEARCH-V2-SYSTEM-BIGAL] "+json.dumps({"documented_ids":sorted(bigal_report),"opening_retest":bigal_open},sort_keys=True,default=str))
    log_func("[NFL-RESEARCH-V2-SYSTEM-ACADEMIC] "+json.dumps(academic,sort_keys=True,default=str))
    log_func("[NFL-RESEARCH-V2-SYSTEM-CROSS-SPORT] "+json.dumps(cross_sport,sort_keys=True,default=str))
    log_func("[NFL-SYSTEM-RULES-INDEX] "+json.dumps({"status":"PASS","rows":rules_index.get("row_count"),"rules_index_uri":rules_upload.get("uri"),"pointer_uri":rules_pointer_uri,"sources":sorted(set(str(x.get("source")) for x in rules_index.get("rows",[]))),"production_authority":0},sort_keys=True,default=str))
    log_func("[NFL-SYSTEM-SOURCE-ATTRIBUTION] "+json.dumps(research_system_attribution,sort_keys=True,default=str))
    log_func("[NFL-SYSTEM-COVERAGE-AUDIT] "+json.dumps(coverage_audit,sort_keys=True,default=str))
    log_func("[NFL-PT-SYSTEM-INTEGRATION] "+json.dumps({
        "status":prediction_tracker_external.get("status"),
        "spread_predictors":prediction_tracker_external.get("spread_predictor_count",0),
        "total_predictors":prediction_tracker_external.get("total_predictor_count",0),
        "spread_external_retained":len((((prediction_tracker_external.get("miner") or {}).get("spreads") or {}).get("retained_rules") or [])),
        "total_external_retained":len((((prediction_tracker_external.get("miner") or {}).get("totals") or {}).get("retained_rules") or [])),
        "external_mechanisms":prediction_tracker_external.get("external_mechanism_count",0),
        "qualified_pt_systems":prediction_tracker_external.get("qualified_system_count",0),
        "pt_family_vote_cap":1,"pt_model_weight":0.0,
        "legacy_spread_retained":len(spread_miner.get("retained_rules") or []),
        "legacy_total_retained":len(total_miner.get("retained_rules") or []),
        "legacy_lane_mutated":False,"production_authority":0,"year_2026_queried":False,
    },sort_keys=True,default=str))
    log_func("[NFL-RESEARCH-V2-SYSTEM-CONTRACT] "+json.dumps({"status":STATUS,"registry_sha256":sha,"family_registry_sha256":family_sha,"spread_legit":len(spread_miner["legit_rules"]),"spread_promising":len(spread_miner["promising_rules"]),"spread_watch":len(spread_miner["watch_rules"]),"spread_dormant_historical":len(spread_miner.get("dormant_historical_rules",[])),"total_legit":len(total_miner["legit_rules"]),"total_promising":len(total_miner["promising_rules"]),"total_watch":len(total_miner["watch_rules"]),"total_dormant_historical":len(total_miner.get("dormant_historical_rules",[])),"mechanism_families":len(mechanism_families),"expert_atom_count":expert_atom_bridge.get("atom_count"),"expert_bridge_mechanisms":sum(1 for f in mechanism_families if f.get("expert_bridge")),"expert_bridge_live_authority":0,"prediction_tracker_status":prediction_tracker_external.get("status"),"prediction_tracker_external_mechanisms":prediction_tracker_external.get("external_mechanism_count",0),"prediction_tracker_spread_retained":len((((prediction_tracker_external.get("miner") or {}).get("spreads") or {}).get("retained_rules") or [])),"prediction_tracker_total_retained":len((((prediction_tracker_external.get("miner") or {}).get("totals") or {}).get("retained_rules") or [])),"prospective_families":len(family_payload["prospective_family_ids"]),"year_2026_queried":False,"model_state_in_discovery":False,"production_authority":0,"ncaaf":"UNCHANGED","legacy_nfl":"UNCHANGED","artifacts":arts},sort_keys=True,default=str))
    return {**report,"artifacts":arts}


# Explicit compatibility aliases for offline callers; source tag identifies V3.
run_nfl_system_lab_v2 = run_nfl_system_lab_v3
run_nfl_system_lab_v1 = run_nfl_system_lab_v3


# ------------------------------- synthetic tests -------------------------------
def self_test() -> dict:
    # Generic miner smoke test with deliberately small atom universe and enough
    # rows to exercise frozen-period logic. No assertion that a system must pass.
    rows=[]
    for sy in range(2017,2026):
        for i in range(80):
            rows.append({"Season":sy,"physical_game_id":f"{sy}|{i}","A":i%2==0,"B":i%3==0})
    d=pd.DataFrame(rows); y=np.array([1.0 if (r.A and (r.Season<=2022)) else float((idx*17+3)%2) for idx,r in d.iterrows()])
    atoms=[{"name":"A","family":"F1","mask":d.A.to_numpy(bool),"description":"A","boundary":None},{"name":"B","family":"F2","mask":d.B.to_numpy(bool),"description":"B","boundary":None}]
    rep=_search_market(d,y,atoms,"TOTALS",False,log_func=lambda *_:None)
    if rep["raw_tested"]<=0:raise AssertionError("MINER_DID_NOT_TEST")
    # Selection periods are hard-coded and 2025 cannot choose direction.
    if FINAL_CHECK_SEASON in DISCOVERY_SEASONS:raise AssertionError("FINAL_CHECK_LEAK")
    mock_f=[{"system_family_id":"NFL_SPREAD_TEST","market":"SPREADS","direction":"PLAY_ON","family_status":"WATCH_FAMILY","representative_conditions":["ROAD_DOG","OFF_ATS_WIN"],"representative_records":{"discovery":{"n":60,"wins":34,"rate":34/60},"validation_2023_2025":{"n":30,"wins":16,"rate":16/30}},"member_count":1,"representative_rule_id":"ROAD_DOG__OFF_ATS_WIN","prospective_action":"RESEARCH_ONLY","prospective_eligible":False}]
    mock_b={sid:{"name":sid,"source_class":"TEST"} for sid in BIGAL_EXPECTED_IDS}
    idx=_build_rules_index(mock_f,mock_b,{}, {}, {})
    cov=_coverage_audit(idx,mock_b,mock_f)
    if cov.get("status")!="PASS" or not any(x.get("system_id")=="BA-NFL6" for x in idx.get("rows",[])):raise AssertionError("RULES_INDEX_COVERAGE_FAIL")
    # Discovery evidence is a permanent provenance field in V3.3; current evidence may weaken authority but cannot erase it.
    assert all("historical_discovery_status" in x for x in idx.get("rows",[]) if x.get("source")=="MINED")
    _dormant_rule={"market":"SPREADS","direction":"PLAY_ON","status":"DORMANT_HISTORICAL_EDGE","historical_discovery_retained":True,"current_evidence_state":"DORMANT_RECENTLY_DEGRADED","conditions":["ROAD_DOG"],"families":["ROLE_COMPOSITE"],"ambiguous_both_sides":0,"records":{"discovery":{"n":100,"wins":60,"rate":.60},"validation_2023_2025":{"n":60,"wins":28,"rate":28/60},"all_history":{"n":160,"wins":88,"rate":.55}}}
    _dfam=_collapse_mechanism_families({"retained_rules":[_dormant_rule]},{"retained_rules":[]})
    assert _dfam and _dfam[0].get("family_status")=="DORMANT_HISTORICAL_EDGE_FAMILY"
    assert _dfam[0].get("shadow_tracking_target") is True
    if MAX_SEQUENCE_DEPTH!=3: raise AssertionError("SEQUENCE_DEPTH_CAP_CHANGED")
    # Exact expert occurrence ledger -> side-state bridge regression.
    es=pd.DataFrame({"Season":[2020,2020,2020,2020],"physical_game_id":["g1","g1","g2","g2"],"Team_Norm":["A","B","C","D"]})
    pp=pd.DataFrame([{"source":"PATHI","system_id":"Pathi_FB_Dog_Hook_Above_3","physical_game_id":"g1","bet_team":"A","status":"WIN"},{"source":"PATHI","system_id":"Pathi_FB_Dog_Hook_Above_7","physical_game_id":"g2","bet_team":"D","status":"LOSS"}])
    bp=pd.DataFrame([{"source":"BIGAL","system_id":"BA-NFL1","physical_game_id":"g1","bet_team":"A","status":"WIN"}])
    ea,eb=_expert_occurrence_atom_bridge(es,bp,pp,log_func=lambda *_:None)
    eby={a["name"]:a for a in ea}
    assert eby["EXPERT_PATHI_DOG_HOOK_ABOVE_3"]["mask"].tolist()==[True,False,False,False]
    assert eby["EXPERT_BIGAL_BA_NFL1"]["mask"].tolist()==[True,False,False,False]
    assert eb.get("pathi_occurrences")==2 and eb.get("pathi_matched")==2 and eb.get("bigal_occurrences")==1 and eb.get("bigal_matched")==1
    _erule={"market":"SPREADS","direction":"PLAY_ON","status":"LEGIT_CANDIDATE_REQUIRES_PROSPECTIVE","historical_discovery_retained":True,"current_evidence_state":"CURRENT_EDGE_POSITIVE","conditions":["EXPERT_PATHI_DOG_HOOK_ABOVE_3","ROAD_DOG"],"families":["EXPERT_PATHI_PATHI_KEY_3_HOOK","ROLE_COMPOSITE"],"ambiguous_both_sides":0,"records":{"discovery":{"n":100,"wins":60,"rate":.60},"validation_2023_2025":{"n":60,"wins":35,"rate":35/60},"all_history":{"n":160,"wins":95,"rate":95/160}}}
    _pt_good={"status":"LEGIT_CANDIDATE_REQUIRES_PROSPECTIVE","current_evidence_state":"CURRENT_EDGE_VALIDATED","ambiguous_both_sides":0,"records":{"validation_2023_2025":{"n":60,"wins":35,"losses":25,"rate":35/60}}}
    _pt_bad={"status":"WATCHLIST","current_evidence_state":"POSITIVE_BUT_BELOW_MINUS110_BREAK_EVEN","ambiguous_both_sides":0,"records":{"validation_2023_2025":{"n":60,"wins":31,"losses":29,"rate":31/60}}}
    assert _pt_external_authority_eligible(_pt_good) is True
    assert _pt_external_authority_eligible(_pt_bad) is False
    _efam=_collapse_mechanism_families({"retained_rules":[_erule]},{"retained_rules":[]})
    assert _efam and _efam[0].get("expert_bridge") is True and _efam[0].get("prospective_eligible") is False and _efam[0].get("prospective_action")=="SHADOW_TRACK_NO_AUTHORITY"
    # Prediction Tracker contracts: source outcomes/market summaries are not predictors,
    # obvious same-provider variants collapse to one consensus source cluster, and the
    # external Miner can be seeded only by external atoms.
    _pt_mock=pd.DataFrame(columns=["line","lineopen","linemidweek","lineavg","linestd","linesag","linesaggm","linepi","linepim","phcover","hscore","rscore"])
    _pt_pred=[x[0] for x in _pt_spread_predictor_cols(_pt_mock)]
    assert "linesag" in _pt_pred and "linesaggm" in _pt_pred and "linepi" in _pt_pred and "linepim" in _pt_pred
    assert all(x not in _pt_pred for x in ("line","lineopen","linemidweek","lineavg","linestd","phcover","hscore","rscore"))
    assert _pt_source_cluster("linesag","SPREADS")=="SAGARIN" and _pt_source_cluster("linesaggm","SPREADS")=="SAGARIN"
    assert _pt_source_cluster("linepi","SPREADS")=="PI_RATINGS" and _pt_source_cluster("linepim","SPREADS")=="PI_RATINGS"
    assert _pt_team_code("Oakland Raiders")=="LV" and _pt_team_code("Washington Football Team")=="WAS"
    _ss_rows=[]
    for _i in range(64):
        _on=_i<32
        _ss_rows.append({"Season":2020,"Season_Stage":"REGULAR","Team_Norm":f"T{_i%8}","Opening_Spread":9.0 if _on else -3.0,"Week_Number":5,
                         "calc_team_game_number":5 if _on else 3,"calc_win_pct_prior":0.0 if _on else .5,"Opp_calc_win_pct_prior":.5 if _on else 0.0,
                         "calc_ats_loss_streak_prior":4.0 if _on else 0.0})
    _ss_atoms={a["name"] for a in _spread_atoms(pd.DataFrame(_ss_rows))}
    for _need in ("SU_WINLESS_PRIOR","ATS_COVERLESS_PRIOR","SU_AND_ATS_WINLESS_AFTER_4_PLUS","TEAM_GAME_5","DOG_9_PLUS","OPP_HAS_SU_WIN"):
        assert _need in _ss_atoms, f"SEASON_RECORD_ATOM_MISSING:{_need}"
    _ss_rule={"market":"SPREADS","direction":"PLAY_ON","families":["SEASON_RECORD_STATE","PRICE_EXTREME"],"conditions":["SU_AND_ATS_WINLESS_AFTER_4_PLUS","DOG_9_PLUS"]}
    assert _mechanism_id(_ss_rule)=="NFL_SPREAD_SEASON_RECORD_STATE_PLAY_ON"

    return {"status":"PASS","version":"NFL_SYSTEM_LAB_V3_14_0","raw_tested":rep["raw_tested"],"unique_hypotheses":rep["unique_hypotheses"],"rules_index_rows":idx.get("row_count"),"bigal6_indexed":True,"coverage_status":cov.get("status"),"discovery_evidence_permanent":True,"dormant_family_preserved":True,"sequential_context_grammar":True,"horizon_symmetry_1_2_3":True,"exact_sequence_2_game":True,"prior_three_sequence_cap":MAX_SEQUENCE_DEPTH,"magnitude_intelligence":True,"opponent_sequence_symmetry":True,"team_sequence_memory":True,"team_context_history":True,"island_slot_context":True,"expert_occurrence_atom_bridge":True,"expert_bridge_shadow_only":True,"prediction_tracker_external_family":True,"prediction_tracker_separate_lane":True,"prediction_tracker_source_cluster_balanced":True,"prediction_tracker_2026_sealed":True,"season_record_state_layer":True,"season_record_state_family_vote_cap":1,"season_record_state_test_atoms":sorted(_ss_atoms),"year_2026_queried":False,"model_state_in_discovery":False}
