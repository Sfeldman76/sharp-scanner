"""NFL V1.9.1 stats-only context research layer.

Builds leakage-safe pregame context strictly from the existing BigDataBall NFL
box-score dataset. No external injury/QB/weather/stadium data are used.

The module is research-only. Historical closing lines are retrospective
benchmarks, never claimed executable historical prices.
"""
from __future__ import annotations

import math
from typing import Iterable
import numpy as np
import pandas as pd

RAW = "sharplogger.sharp_data.nfl_historical_game_side_raw"
IDENTITY = ("Season", "Source_Name", "Source_Game_ID")
SOURCE_TAG = "nfl-stats-context-v1.9.1-existing-dataset-only-20261001"
RIDGE_ALPHA = 25.0  # frozen before 2026 holdout is queried

RAW_STATS_COLUMNS = (
    "Season","Season_Stage","Source_Name","Source_Game_ID","Game_Date","Week","Start_Time_ET",
    "Team_Norm","Opponent_Norm","Is_Home","Is_Away","Is_Neutral","Team_Score","Opponent_Score",
    "Postgame_First_Downs","Postgame_Rush_Att","Postgame_Rush_Yards","Postgame_Rush_TD",
    "Postgame_Pass_Comp","Postgame_Pass_Att","Postgame_Pass_Yards","Postgame_Pass_TD",
    "Postgame_Pass_INT","Postgame_Sacked","Postgame_Net_Pass_Yards","Postgame_Total_Yards",
    "Postgame_Fumbles","Postgame_Fumbles_Lost","Postgame_Turnovers","Postgame_Penalties",
    "Postgame_Penalty_Yards","Postgame_Third_Down_Made","Postgame_Third_Down_Att",
    "Postgame_Fourth_Down_Made","Postgame_Fourth_Down_Att","Postgame_Total_Plays",
    "Postgame_Time_Possession_Minutes","Postgame_Def_Sacks","Postgame_Opp_Fumbles_Recovered",
    "Postgame_Def_Fumble_Recovery_TD","Postgame_Def_INT_Return_TD","Postgame_Blocked_Kick_Return_TD",
    "Postgame_Return_TD","Postgame_Safeties","Postgame_Def_INT","Postgame_FGM",
    "Q1_Points","Q2_Points","Q3_Points","Q4_Points","OT_Points",
)

# Fixed, concept-driven families. They are intentionally small enough to audit.
FEATURE_FAMILIES = {
    "DIVISION_REMATCH": (
        "Is_Division_Game","Is_Conference_Game","Is_Interconference_Game",
        "same_season_rematch","division_rematch","same_season_role_flip",
        "days_since_same_season_matchup","prev_same_season_margin",
    ),
    "SCHEDULE_SEQUENCE": (
        "first_home_game","first_road_game","home_streak_prior","road_streak_prior",
        "opponent_first_home_game","opponent_first_road_game","opponent_home_streak_prior","opponent_road_streak_prior",
    ),
    "VENUE_FORM": (
        "Is_Neutral","prior_home_margin_mean","prior_away_margin_mean","home_away_margin_diff",
        "prior_home_off_ypp_mean","prior_away_off_ypp_mean","home_away_ypp_diff",
        "venue_matchup_margin_diff","venue_matchup_ypp_diff",
    ),
    "OPPONENT_ADJUSTED": (
        "adj_off_ypp_last3_prior","adj_def_ypp_last3_prior","schedule_strength_winpct_prior",
        "adj_off_ypp_context_diff","adj_def_ypp_context_diff","schedule_strength_diff",
    ),
    "RUN_PASS_MATCHUP": (
        "rush_ypa_last3_prior","pass_ypdb_last3_prior","def_rush_ypa_allowed_last3_prior",
        "def_pass_ypdb_allowed_last3_prior","rush_matchup_diff","pass_matchup_diff",
        "sack_rate_allowed_last3_prior","def_sack_rate_last3_prior","sack_matchup_diff",
    ),
    "PACE_EFFICIENCY": (
        "total_plays_last3_prior","time_possession_last3_prior","points_per_play_last3_prior",
        "first_down_rate_last3_prior","third_down_pct_last3_prior","plays_context_diff",
        "time_possession_context_diff","points_per_play_context_diff","first_down_rate_context_diff","third_down_context_diff",
    ),
    "TURNOVER_REGRESSION": (
        "giveaways_last3_prior","takeaways_last3_prior","turnover_margin_last3_prior",
        "pass_int_last3_prior","fumbles_lost_last3_prior","fumble_takeaways_last3_prior",
        "turnover_margin_context_diff","giveaway_context_diff","pass_int_context_diff","fumble_lost_context_diff",
    ),
    "NONOFFENSIVE_SCORING": (
        "nonoff_score_events_last5_prior","def_int_td_last5_prior","def_fumble_td_last5_prior","return_td_last5_prior",
        "nonoff_event_context_diff",
    ),
    "QUARTER_HALF_PROFILE": (
        "first_half_points_last3_prior","second_half_points_last3_prior",
        "first_half_margin_last3_prior","second_half_margin_last3_prior","first_half_share_last3_prior",
        "first_half_margin_context_diff","second_half_margin_context_diff","first_half_share_context_diff",
    ),
    "DISCIPLINE_FOURTH_DOWN": (
        "penalty_yards_per_play_last3_prior","fourth_down_pct_last3_prior","penalties_last3_prior",
        "penalty_rate_context_diff","fourth_down_context_diff",
    ),
    "VOLATILITY_TREND": (
        "margin_sd_last5_prior","points_sd_last5_prior","off_ypp_sd_last5_prior","total_sd_last5_prior",
        "points_trend_last3_vs_prior","off_ypp_trend_last3_vs_prior","def_ypp_trend_last3_vs_prior",
        "margin_sd_game_sum","points_sd_game_sum","off_ypp_sd_game_sum","total_sd_game_sum",
        "points_trend_context_diff","off_ypp_trend_context_diff","def_ypp_trend_context_diff",
    ),
    "CLOSE_GAME_STATE": (
        "close_game_win_pct_prior","close_game_rate_last5_prior","blowout_rate_last5_prior",
        "close_game_win_pct_diff","close_game_rate_diff","blowout_rate_diff",
    ),
}
ALL_STATS_FEATURES = tuple(dict.fromkeys(x for vals in FEATURE_FAMILIES.values() for x in vals))


def build_raw_stats_query(raw_columns: Iterable[str] | None = None) -> str:
    if raw_columns is not None:
        miss = set(RAW_STATS_COLUMNS) - set(raw_columns)
        if miss:
            raise RuntimeError("[NFL-V1.9.1-HOLD] RAW_STATS_COLUMNS_MISSING " + str(sorted(miss)))
    q = ", ".join(f"`{c}`" for c in RAW_STATS_COLUMNS)
    return (
        f"SELECT {q} FROM `{RAW}` WHERE Season BETWEEN 2017 AND 2026 "
        "AND Season_Stage IN ('REGULAR','POSTSEASON') ORDER BY Season, Game_Date, Source_Name, Source_Game_ID, Team_Norm"
    )


def _n(s):
    return pd.to_numeric(s, errors="coerce").astype(float)


def _safe_div(a, b):
    aa, bb = _n(a), _n(b)
    return aa.div(bb.where(bb.abs() > 1e-12))


def _roll_prior(d: pd.DataFrame, col: str, window: int, func: str = "mean", min_periods: int = 1) -> pd.Series:
    keys = [d["Season"], d["Team_Norm"]]
    s = _n(d[col])
    def f(x):
        z = x.shift(1).rolling(window, min_periods=min_periods)
        return getattr(z, func)()
    return s.groupby(keys, sort=False).transform(f)


def _exp_prior(d: pd.DataFrame, col: str) -> pd.Series:
    keys = [d["Season"], d["Team_Norm"]]
    s = _n(d[col])
    return s.groupby(keys, sort=False).transform(lambda x: x.shift(1).expanding(min_periods=1).mean())


def _conditional_exp_prior(d: pd.DataFrame, col: str, mask: pd.Series) -> pd.Series:
    keys = [d["Season"], d["Team_Norm"]]
    s = _n(d[col]).where(mask.astype(bool))
    return s.groupby(keys, sort=False).transform(lambda x: x.shift(1).expanding(min_periods=1).mean())


def _pair_current(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keys = ["Season","Source_Name","Source_Game_ID"]
    opp = d[keys + ["Team_Norm"] + cols].copy()
    opp = opp.rename(columns={"Team_Norm":"Opponent_Norm", **{c:"opp_current__"+c for c in cols}})
    return d.merge(opp, on=keys+["Opponent_Norm"], how="left", validate="one_to_one")


def derive_stats_context(raw: pd.DataFrame) -> pd.DataFrame:
    """Create prior-only stats context for every side, including sequential 2026 rows.

    Same-game box score fields are used only to create outcomes which are shifted or
    rolled before becoming predictors for a later game.
    """
    missing = set(RAW_STATS_COLUMNS) - set(raw.columns)
    if missing:
        raise RuntimeError("RAW_STATS_INPUT_MISSING " + str(sorted(missing)))
    d = raw.copy()
    d["Season"] = pd.to_numeric(d["Season"], errors="raise").astype(int)
    d["Game_Date"] = pd.to_datetime(d["Game_Date"], errors="coerce")
    d["Team_Norm"] = d["Team_Norm"].astype(str).str.strip().str.lower()
    d["Opponent_Norm"] = d["Opponent_Norm"].astype(str).str.strip().str.lower()
    d = d.sort_values(["Season","Team_Norm","Game_Date","Source_Name","Source_Game_ID"], kind="mergesort").reset_index(drop=True)

    # Current-game outcomes (never admitted directly as predictors for this same row).
    d["actual_margin"] = _n(d.Team_Score) - _n(d.Opponent_Score)
    d["actual_total"] = _n(d.Team_Score) + _n(d.Opponent_Score)
    d["off_ypp"] = _safe_div(d.Postgame_Total_Yards, d.Postgame_Total_Plays)
    d["rush_ypa"] = _safe_div(d.Postgame_Rush_Yards, d.Postgame_Rush_Att)
    drops = _n(d.Postgame_Pass_Att).fillna(0) + _n(d.Postgame_Sacked).fillna(0)
    d["pass_ypdb"] = _safe_div(d.Postgame_Net_Pass_Yards, drops)
    d["sack_rate_allowed"] = _safe_div(d.Postgame_Sacked, drops)
    d["first_down_rate"] = _safe_div(d.Postgame_First_Downs, d.Postgame_Total_Plays)
    d["third_down_pct"] = _safe_div(d.Postgame_Third_Down_Made, d.Postgame_Third_Down_Att)
    d["fourth_down_pct"] = _safe_div(d.Postgame_Fourth_Down_Made, d.Postgame_Fourth_Down_Att)
    d["points_per_play"] = _safe_div(d.Team_Score, d.Postgame_Total_Plays)
    d["penalty_yards_per_play"] = _safe_div(d.Postgame_Penalty_Yards, d.Postgame_Total_Plays)
    d["first_half_points"] = _n(d.Q1_Points).fillna(0) + _n(d.Q2_Points).fillna(0)
    d["second_half_points"] = _n(d.Q3_Points).fillna(0) + _n(d.Q4_Points).fillna(0) + _n(d.OT_Points).fillna(0)
    d["nonoff_score_events"] = (
        _n(d.Postgame_Def_Fumble_Recovery_TD).fillna(0) + _n(d.Postgame_Def_INT_Return_TD).fillna(0)
        + _n(d.Postgame_Blocked_Kick_Return_TD).fillna(0) + _n(d.Postgame_Return_TD).fillna(0)
        + _n(d.Postgame_Safeties).fillna(0)
    )

    pair_cols = [
        "Postgame_Rush_Att","Postgame_Rush_Yards","Postgame_Pass_Att","Postgame_Sacked","Postgame_Net_Pass_Yards",
        "Postgame_Turnovers","first_half_points","second_half_points","off_ypp",
    ]
    d = _pair_current(d, pair_cols)
    d["def_rush_ypa_allowed"] = _safe_div(d["opp_current__Postgame_Rush_Yards"], d["opp_current__Postgame_Rush_Att"])
    opp_drops = _n(d["opp_current__Postgame_Pass_Att"]).fillna(0) + _n(d["opp_current__Postgame_Sacked"]).fillna(0)
    d["def_pass_ypdb_allowed"] = _safe_div(d["opp_current__Postgame_Net_Pass_Yards"], opp_drops)
    d["def_ypp_allowed"] = _n(d["opp_current__off_ypp"])
    d["def_sack_rate"] = _safe_div(d.Postgame_Def_Sacks, opp_drops)
    d["takeaways"] = _n(d["opp_current__Postgame_Turnovers"])
    d["giveaways"] = _n(d.Postgame_Turnovers)
    d["turnover_margin"] = d.takeaways - d.giveaways
    d["first_half_margin"] = d.first_half_points - _n(d["opp_current__first_half_points"])
    d["second_half_margin"] = d.second_half_points - _n(d["opp_current__second_half_points"])
    d["first_half_share"] = _safe_div(d.first_half_points, _n(d.Team_Score))

    # Pregame season-to-date ratings.
    d["prior_win_pct"] = _n(d.actual_margin.gt(0).astype(float)).groupby([d.Season,d.Team_Norm],sort=False).transform(lambda x: x.shift(1).expanding().mean())
    d["prior_off_ypp_mean"] = _exp_prior(d,"off_ypp")
    d["prior_def_ypp_mean"] = _exp_prior(d,"def_ypp_allowed")

    # Pair opponent PRE-game ratings on the same game; safe by construction.
    d = _pair_current(d, ["prior_win_pct","prior_off_ypp_mean","prior_def_ypp_mean"])
    d["off_ypp_adjusted_outcome"] = d.off_ypp - _n(d["opp_current__prior_def_ypp_mean"])
    d["def_ypp_adjusted_outcome"] = d.def_ypp_allowed - _n(d["opp_current__prior_off_ypp_mean"])
    d["opponent_strength_at_game"] = _n(d["opp_current__prior_win_pct"])

    # Same-season matchup/rematch state.
    mk = [d.Season,d.Team_Norm,d.Opponent_Norm]
    d["same_season_matchup_number"] = d.groupby(["Season","Team_Norm","Opponent_Norm"],sort=False).cumcount() + 1
    d["same_season_rematch"] = d.same_season_matchup_number.ge(2).astype(float)
    prev_date = d["Game_Date"].groupby(mk,sort=False).shift(1)
    d["days_since_same_season_matchup"] = (d.Game_Date - prev_date).dt.days.astype(float)
    d["prev_same_season_margin"] = d["actual_margin"].groupby(mk,sort=False).shift(1)
    prev_home = _n(d.Is_Home).groupby(mk,sort=False).shift(1)
    d["same_season_role_flip"] = np.where(d.same_season_rematch.eq(1), (_n(d.Is_Home) != prev_home).astype(float), 0.0)

    # Schedule-position / venue sequence state, derived only from previous games.
    gkeys=["Season","Team_Norm"]
    home_before=_n(d.Is_Home).groupby([d.Season,d.Team_Norm],sort=False).cumsum()-_n(d.Is_Home)
    road_before=_n(d.Is_Away).groupby([d.Season,d.Team_Norm],sort=False).cumsum()-_n(d.Is_Away)
    d["first_home_game"]=( _n(d.Is_Home).eq(1) & home_before.eq(0) ).astype(float)
    d["first_road_game"]=( _n(d.Is_Away).eq(1) & road_before.eq(0) ).astype(float)
    d["home_streak_prior"]=0.0; d["road_streak_prior"]=0.0
    for _,idxs in d.groupby(gkeys,sort=False).groups.items():
        hs=rs=0
        for ix in idxs:
            d.at[ix,"home_streak_prior"]=float(hs); d.at[ix,"road_streak_prior"]=float(rs)
            if float(d.at[ix,"Is_Home"] or 0)==1.0: hs+=1; rs=0
            elif float(d.at[ix,"Is_Away"] or 0)==1.0: rs+=1; hs=0
            else: hs=rs=0

    # Close-game / blowout state as a direct regression-to-mean diagnostic.
    d["close_game"] = d.actual_margin.abs().le(8).astype(float)
    d["blowout_game"] = d.actual_margin.abs().ge(14).astype(float)
    d["su_value"] = np.where(d.actual_margin.gt(0),1.0,np.where(d.actual_margin.lt(0),0.0,0.5))
    d["close_win_value"] = pd.Series(d.su_value,index=d.index).where(d.close_game.eq(1))
    d["close_game_win_pct_prior"] = _exp_prior(d,"close_win_value")
    d["close_game_rate_last5_prior"] = _roll_prior(d,"close_game",5,"mean",1)
    d["blowout_rate_last5_prior"] = _roll_prior(d,"blowout_game",5,"mean",1)

    # Venue-specific priors.
    d["prior_home_margin_mean"] = _conditional_exp_prior(d,"actual_margin",_n(d.Is_Home).eq(1))
    d["prior_away_margin_mean"] = _conditional_exp_prior(d,"actual_margin",_n(d.Is_Away).eq(1))
    d["home_away_margin_diff"] = d.prior_home_margin_mean - d.prior_away_margin_mean
    d["prior_home_off_ypp_mean"] = _conditional_exp_prior(d,"off_ypp",_n(d.Is_Home).eq(1))
    d["prior_away_off_ypp_mean"] = _conditional_exp_prior(d,"off_ypp",_n(d.Is_Away).eq(1))
    d["home_away_ypp_diff"] = d.prior_home_off_ypp_mean - d.prior_away_off_ypp_mean

    # Rolling football state.
    means3 = {
        "rush_ypa":"rush_ypa_last3_prior","pass_ypdb":"pass_ypdb_last3_prior",
        "def_rush_ypa_allowed":"def_rush_ypa_allowed_last3_prior","def_pass_ypdb_allowed":"def_pass_ypdb_allowed_last3_prior",
        "sack_rate_allowed":"sack_rate_allowed_last3_prior","def_sack_rate":"def_sack_rate_last3_prior",
        "Postgame_Total_Plays":"total_plays_last3_prior","Postgame_Time_Possession_Minutes":"time_possession_last3_prior",
        "points_per_play":"points_per_play_last3_prior","first_down_rate":"first_down_rate_last3_prior",
        "third_down_pct":"third_down_pct_last3_prior","fourth_down_pct":"fourth_down_pct_last3_prior",
        "penalty_yards_per_play":"penalty_yards_per_play_last3_prior","Postgame_Penalties":"penalties_last3_prior",
        "giveaways":"giveaways_last3_prior","takeaways":"takeaways_last3_prior","turnover_margin":"turnover_margin_last3_prior",
        "Postgame_Pass_INT":"pass_int_last3_prior","Postgame_Fumbles_Lost":"fumbles_lost_last3_prior",
        "Postgame_Opp_Fumbles_Recovered":"fumble_takeaways_last3_prior",
        "first_half_points":"first_half_points_last3_prior","second_half_points":"second_half_points_last3_prior",
        "first_half_margin":"first_half_margin_last3_prior","second_half_margin":"second_half_margin_last3_prior",
        "first_half_share":"first_half_share_last3_prior",
        "off_ypp_adjusted_outcome":"adj_off_ypp_last3_prior","def_ypp_adjusted_outcome":"adj_def_ypp_last3_prior",
    }
    for src,dst in means3.items():
        d[dst] = _roll_prior(d,src,3,"mean",1)
    d["schedule_strength_winpct_prior"] = _roll_prior(d,"opponent_strength_at_game",5,"mean",1)
    d["nonoff_score_events_last5_prior"] = _roll_prior(d,"nonoff_score_events",5,"mean",1)
    d["def_int_td_last5_prior"] = _roll_prior(d,"Postgame_Def_INT_Return_TD",5,"mean",1)
    d["def_fumble_td_last5_prior"] = _roll_prior(d,"Postgame_Def_Fumble_Recovery_TD",5,"mean",1)
    d["return_td_last5_prior"] = _roll_prior(d,"Postgame_Return_TD",5,"mean",1)

    # Matchup deltas: offense minus opponent defense or protection vs rush.
    # Opponent rolling priors are paired only after those priors exist.
    d = _pair_current(d, [
        "rush_ypa_last3_prior","pass_ypdb_last3_prior","def_rush_ypa_allowed_last3_prior",
        "def_pass_ypdb_allowed_last3_prior","sack_rate_allowed_last3_prior","def_sack_rate_last3_prior",
    ])
    d["rush_matchup_diff"] = d.rush_ypa_last3_prior - _n(d["opp_current__def_rush_ypa_allowed_last3_prior"])
    d["pass_matchup_diff"] = d.pass_ypdb_last3_prior - _n(d["opp_current__def_pass_ypdb_allowed_last3_prior"])
    d["sack_matchup_diff"] = d.sack_rate_allowed_last3_prior - _n(d["opp_current__def_sack_rate_last3_prior"])

    # Volatility and trend.
    d["margin_sd_last5_prior"] = _roll_prior(d,"actual_margin",5,"std",2)
    d["points_sd_last5_prior"] = _roll_prior(d,"Team_Score",5,"std",2)
    d["off_ypp_sd_last5_prior"] = _roll_prior(d,"off_ypp",5,"std",2)
    d["total_sd_last5_prior"] = _roll_prior(d,"actual_total",5,"std",2)
    d["points_last3_prior"] = _roll_prior(d,"Team_Score",3,"mean",1)
    d["points_prior_mean"] = _exp_prior(d,"Team_Score")
    d["off_ypp_last3_prior"] = _roll_prior(d,"off_ypp",3,"mean",1)
    d["def_ypp_last3_prior"] = _roll_prior(d,"def_ypp_allowed",3,"mean",1)
    d["points_trend_last3_vs_prior"] = d.points_last3_prior - d.points_prior_mean
    d["off_ypp_trend_last3_vs_prior"] = d.off_ypp_last3_prior - d.prior_off_ypp_mean
    d["def_ypp_trend_last3_vs_prior"] = d.def_ypp_last3_prior - d.prior_def_ypp_mean

    # Pair the opponent's prior-only state so physical-game models see both teams,
    # not just the oriented home/selected side.
    pair_prior=[
        "first_home_game","first_road_game","home_streak_prior","road_streak_prior",
        "prior_home_margin_mean","prior_away_margin_mean","prior_home_off_ypp_mean","prior_away_off_ypp_mean",
        "adj_off_ypp_last3_prior","adj_def_ypp_last3_prior","schedule_strength_winpct_prior",
        "total_plays_last3_prior","time_possession_last3_prior","points_per_play_last3_prior","first_down_rate_last3_prior","third_down_pct_last3_prior",
        "turnover_margin_last3_prior","giveaways_last3_prior","pass_int_last3_prior","fumbles_lost_last3_prior",
        "nonoff_score_events_last5_prior","first_half_margin_last3_prior","second_half_margin_last3_prior","first_half_share_last3_prior",
        "penalty_yards_per_play_last3_prior","fourth_down_pct_last3_prior",
        "margin_sd_last5_prior","points_sd_last5_prior","off_ypp_sd_last5_prior","total_sd_last5_prior",
        "points_trend_last3_vs_prior","off_ypp_trend_last3_vs_prior","def_ypp_trend_last3_vs_prior",
        "close_game_win_pct_prior","close_game_rate_last5_prior","blowout_rate_last5_prior",
    ]
    d=_pair_current(d,pair_prior)
    d["opponent_first_home_game"]=_n(d["opp_current__first_home_game"]); d["opponent_first_road_game"]=_n(d["opp_current__first_road_game"])
    d["opponent_home_streak_prior"]=_n(d["opp_current__home_streak_prior"]); d["opponent_road_streak_prior"]=_n(d["opp_current__road_streak_prior"])
    d["venue_matchup_margin_diff"] = np.where(_n(d.Is_Home).eq(1), d.prior_home_margin_mean-_n(d["opp_current__prior_away_margin_mean"]),
        np.where(_n(d.Is_Away).eq(1), d.prior_away_margin_mean-_n(d["opp_current__prior_home_margin_mean"]), np.nan))
    d["venue_matchup_ypp_diff"] = np.where(_n(d.Is_Home).eq(1), d.prior_home_off_ypp_mean-_n(d["opp_current__prior_away_off_ypp_mean"]),
        np.where(_n(d.Is_Away).eq(1), d.prior_away_off_ypp_mean-_n(d["opp_current__prior_home_off_ypp_mean"]), np.nan))
    d["adj_off_ypp_context_diff"]=d.adj_off_ypp_last3_prior-_n(d["opp_current__adj_off_ypp_last3_prior"])
    d["adj_def_ypp_context_diff"]=d.adj_def_ypp_last3_prior-_n(d["opp_current__adj_def_ypp_last3_prior"])
    d["schedule_strength_diff"]=d.schedule_strength_winpct_prior-_n(d["opp_current__schedule_strength_winpct_prior"])
    d["plays_context_diff"]=d.total_plays_last3_prior-_n(d["opp_current__total_plays_last3_prior"])
    d["time_possession_context_diff"]=d.time_possession_last3_prior-_n(d["opp_current__time_possession_last3_prior"])
    d["points_per_play_context_diff"]=d.points_per_play_last3_prior-_n(d["opp_current__points_per_play_last3_prior"])
    d["first_down_rate_context_diff"]=d.first_down_rate_last3_prior-_n(d["opp_current__first_down_rate_last3_prior"])
    d["third_down_context_diff"]=d.third_down_pct_last3_prior-_n(d["opp_current__third_down_pct_last3_prior"])
    d["turnover_margin_context_diff"]=d.turnover_margin_last3_prior-_n(d["opp_current__turnover_margin_last3_prior"])
    d["giveaway_context_diff"]=d.giveaways_last3_prior-_n(d["opp_current__giveaways_last3_prior"])
    d["pass_int_context_diff"]=d.pass_int_last3_prior-_n(d["opp_current__pass_int_last3_prior"])
    d["fumble_lost_context_diff"]=d.fumbles_lost_last3_prior-_n(d["opp_current__fumbles_lost_last3_prior"])
    d["nonoff_event_context_diff"]=d.nonoff_score_events_last5_prior-_n(d["opp_current__nonoff_score_events_last5_prior"])
    d["first_half_margin_context_diff"]=d.first_half_margin_last3_prior-_n(d["opp_current__first_half_margin_last3_prior"])
    d["second_half_margin_context_diff"]=d.second_half_margin_last3_prior-_n(d["opp_current__second_half_margin_last3_prior"])
    d["first_half_share_context_diff"]=d.first_half_share_last3_prior-_n(d["opp_current__first_half_share_last3_prior"])
    d["penalty_rate_context_diff"]=d.penalty_yards_per_play_last3_prior-_n(d["opp_current__penalty_yards_per_play_last3_prior"])
    d["fourth_down_context_diff"]=d.fourth_down_pct_last3_prior-_n(d["opp_current__fourth_down_pct_last3_prior"])
    d["margin_sd_game_sum"]=d.margin_sd_last5_prior+_n(d["opp_current__margin_sd_last5_prior"])
    d["points_sd_game_sum"]=d.points_sd_last5_prior+_n(d["opp_current__points_sd_last5_prior"])
    d["off_ypp_sd_game_sum"]=d.off_ypp_sd_last5_prior+_n(d["opp_current__off_ypp_sd_last5_prior"])
    d["total_sd_game_sum"]=d.total_sd_last5_prior+_n(d["opp_current__total_sd_last5_prior"])
    d["points_trend_context_diff"]=d.points_trend_last3_vs_prior-_n(d["opp_current__points_trend_last3_vs_prior"])
    d["off_ypp_trend_context_diff"]=d.off_ypp_trend_last3_vs_prior-_n(d["opp_current__off_ypp_trend_last3_vs_prior"])
    d["def_ypp_trend_context_diff"]=d.def_ypp_trend_last3_vs_prior-_n(d["opp_current__def_ypp_trend_last3_vs_prior"])
    d["close_game_win_pct_diff"]=d.close_game_win_pct_prior-_n(d["opp_current__close_game_win_pct_prior"])
    d["close_game_rate_diff"]=d.close_game_rate_last5_prior-_n(d["opp_current__close_game_rate_last5_prior"])
    d["blowout_rate_diff"]=d.blowout_rate_last5_prior-_n(d["opp_current__blowout_rate_last5_prior"])

    keep = list(dict.fromkeys(list(IDENTITY) + ["Game_Date","Team_Norm","Opponent_Norm","Is_Home","Is_Away","Is_Neutral"] + list(ALL_STATS_FEATURES)))
    # raw-only rematch fields; division_rematch is added after context merge.
    keep = [c for c in keep if c in d.columns and c != "division_rematch"]
    return d[keep].copy()


def attach_stats_features(games: pd.DataFrame, stats_side: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    s = stats_side.copy()
    for z in (g,s):
        z["Team_Norm"] = z["Team_Norm"].astype(str).str.strip().str.lower()
    keys = ["Season","Source_Name","Source_Game_ID","Team_Norm"]
    cols = keys + [c for c in ALL_STATS_FEATURES if c in s.columns and c != "division_rematch"]
    m = g.merge(s[cols], on=keys, how="left", validate="one_to_one", suffixes=("","_stats"))
    m["division_rematch"] = (
        pd.to_numeric(m.get("Is_Division_Game"),errors="coerce").eq(1)
        & pd.to_numeric(m.get("same_season_rematch"),errors="coerce").eq(1)
    ).astype(float)
    return m


def _pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=False)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])


def _mae(a,b):
    x,y=np.asarray(a,float),np.asarray(b,float); m=np.isfinite(x)&np.isfinite(y)
    return float(np.mean(np.abs(x[m]-y[m]))) if m.any() else math.nan


def _feature_cols(df: pd.DataFrame, family: str) -> list[str]:
    cols = list(ALL_STATS_FEATURES if family == "ALL_STATS" else FEATURE_FAMILIES[family])
    return [c for c in cols if c in df.columns]


def _fit_family(train: pd.DataFrame, valid: pd.DataFrame, family: str, market: str):
    cols = _feature_cols(train, family)
    if not cols: raise RuntimeError("NO_FEATURES_"+family)
    if market == "spreads":
        y = pd.to_numeric(train.actual_margin,errors="coerce") + pd.to_numeric(train.Spread_Value,errors="coerce")
        market_pred = -pd.to_numeric(valid.Spread_Value,errors="coerce")
        actual = pd.to_numeric(valid.actual_margin,errors="coerce")
    else:
        y = pd.to_numeric(train.actual_total,errors="coerce") - pd.to_numeric(train.Current_Total,errors="coerce")
        market_pred = pd.to_numeric(valid.Current_Total,errors="coerce")
        actual = pd.to_numeric(valid.actual_total,errors="coerce")
    trmask = y.notna()
    if int(trmask.sum()) < 500:
        raise RuntimeError(f"TOO_FEW_TRAIN_ROWS_{family}_{market}_{int(trmask.sum())}")
    model=_pipeline(); model.fit(train.loc[trmask,cols],y.loc[trmask])
    correction=np.asarray(model.predict(valid[cols]),float)
    corrected=np.asarray(market_pred,float)+correction
    return corrected, correction, cols


def development_oof(stats_games: pd.DataFrame) -> dict:
    """Fixed season-forward development diagnostic; never selects a family."""
    rows=[]
    for year in (2021,2022,2023,2024,2025):
        tr=stats_games.loc[pd.to_numeric(stats_games.Season,errors="coerce").lt(year)].copy()
        va=stats_games.loc[pd.to_numeric(stats_games.Season,errors="coerce").eq(year)].copy()
        if len(va)<200: continue
        for fam in (*FEATURE_FAMILIES.keys(),"ALL_STATS"):
            for market in ("spreads","totals"):
                try: pred,_,cols=_fit_family(tr,va,fam,market)
                except Exception: continue
                if market=="spreads":
                    actual=pd.to_numeric(va.actual_margin,errors="coerce"); base=-pd.to_numeric(va.Spread_Value,errors="coerce")
                else:
                    actual=pd.to_numeric(va.actual_total,errors="coerce"); base=pd.to_numeric(va.Current_Total,errors="coerce")
                rows.append({"season":year,"family":fam,"market":market,"n":int(actual.notna().sum()),
                             "market_mae":_mae(actual,base),"corrected_mae":_mae(actual,pred),"feature_count":len(cols)})
    out={}
    rdf=pd.DataFrame(rows)
    for fam in (*FEATURE_FAMILIES.keys(),"ALL_STATS"):
        out[fam]={}
        for market in ("spreads","totals"):
            q=rdf.loc[(rdf.family.eq(fam))&(rdf.market.eq(market))]
            if q.empty: continue
            # Weighted by games via concatenation-equivalent approximation.
            n=int(q.n.sum()); b=float(np.average(q.market_mae,weights=q.n)); c=float(np.average(q.corrected_mae,weights=q.n))
            out[fam][market]={"n":n,"market_mae":round(b,6),"corrected_mae":round(c,6),"mae_improvement":round(b-c,6),
                              "season_improvements":[round(float(a-bb),6) for a,bb in zip(q.market_mae,q.corrected_mae)]}
    return {"status":"DEVELOPMENT_DESCRIPTIVE_ONLY","ridge_alpha":RIDGE_ALPHA,"families":out,"production_authority":0}


def holdout_correction(train_games: pd.DataFrame, hold_games: pd.DataFrame, core_hold: pd.DataFrame) -> dict:
    out={}
    core_index=core_hold.set_index(["Season","Source_Name","Source_Game_ID"])
    for fam in (*FEATURE_FAMILIES.keys(),"ALL_STATS"):
        out[fam]={}
        for market in ("spreads","totals"):
            pred,corr,cols=_fit_family(train_games,hold_games,fam,market)
            if market=="spreads":
                actual=pd.to_numeric(hold_games.actual_margin,errors="coerce"); base=-pd.to_numeric(hold_games.Spread_Value,errors="coerce")
                cp=pd.to_numeric(core_index.reindex(hold_games.set_index(["Season","Source_Name","Source_Game_ID"]).index).core_margin_pred,errors="coerce")
            else:
                actual=pd.to_numeric(hold_games.actual_total,errors="coerce"); base=pd.to_numeric(hold_games.Current_Total,errors="coerce")
                cp=pd.to_numeric(core_index.reindex(hold_games.set_index(["Season","Source_Name","Source_Game_ID"]).index).core_total_pred,errors="coerce")
            bm=_mae(actual,base); cm=_mae(actual,pred); corem=_mae(actual,cp)
            out[fam][market]={"n":int(actual.notna().sum()),"feature_count":len(cols),"market_mae":round(bm,6),"stats_corrected_market_mae":round(cm,6),
                              "core_mae":round(corem,6),"improvement_vs_market":round(bm-cm,6),"improvement_vs_core":round(corem-cm,6),
                              "mean_abs_correction":round(float(np.nanmean(np.abs(corr))),6)}
    return {"status":"FROZEN_2026_STATS_CONTEXT_HOLDOUT","ridge_alpha":RIDGE_ALPHA,"families":out,"production_authority":0}


def context_holdout_slices(hold_games: pd.DataFrame, core_hold: pd.DataFrame) -> dict:
    keys=["Season","Source_Name","Source_Game_ID"]
    h=hold_games.merge(core_hold[keys+["spread_consensus_edge","spread_models_agree"]],on=keys,how="left",validate="one_to_one")
    ats=pd.to_numeric(h.actual_margin,errors="coerce")+pd.to_numeric(h.Spread_Value,errors="coerce")
    edge=pd.to_numeric(h.spread_consensus_edge,errors="coerce"); valid=ats.notna()&edge.notna()&h.spread_models_agree.astype(bool)&edge.abs().ge(5)&~np.isclose(ats,0)
    def rate(mask):
        m=valid&pd.Series(mask,index=h.index).fillna(False).astype(bool); n=int(m.sum())
        if not n:return {"n":0,"wins":0,"rate":None}
        w=int((np.sign(edge[m].to_numpy(float))==np.sign(ats[m].to_numpy(float))).sum())
        return {"n":n,"wins":w,"rate":round(w/n,6)}
    div=pd.to_numeric(h.get("Is_Division_Game"),errors="coerce").eq(1)
    conf=pd.to_numeric(h.get("Is_Conference_Game"),errors="coerce").eq(1)
    rem=pd.to_numeric(h.get("same_season_rematch"),errors="coerce").eq(1)
    neutral=pd.to_numeric(h.get("Is_Neutral"),errors="coerce").eq(1)
    return {"core_edge_5":{
        "division":rate(div),"nondivision":rate(~div),"division_rematch":rate(div&rem),"same_season_rematch":rate(rem),
        "conference_nondivision":rate(conf&~div),"interconference":rate(pd.to_numeric(h.get("Is_Interconference_Game"),errors="coerce").eq(1)),
        "neutral_site":rate(neutral),"regular_home_away":rate(~neutral),
        "home_team_first_home_game":rate(pd.to_numeric(h.get("first_home_game"),errors="coerce").eq(1)),
        "away_team_first_road_game":rate(pd.to_numeric(h.get("opponent_first_road_game"),errors="coerce").eq(1)),
        "away_team_road_streak_2plus":rate(pd.to_numeric(h.get("opponent_road_streak_prior"),errors="coerce").ge(2)),
        "home_team_returning_from_road":rate(pd.to_numeric(h.get("road_streak_prior"),errors="coerce").ge(1)),
        "role_flip_rematch":rate(rem & pd.to_numeric(h.get("same_season_role_flip"),errors="coerce").eq(1)),
    },"production_authority":0}
