"""NFL Betting Engine V1 — unified second-stage bet decision model.

This is the decision layer that was intentionally missing from the earlier
Production V1 work.  The frozen Spread/H2H/Totals backbones remain fair-value
models.  This module learns when those fair values are actionable against the
market.

Historical training discipline
------------------------------
* Base predictions are the leak-safe Production V1 historical replay rows.
* Betting-engine validation is itself walk-forward by season:
    2023 <- train 2021-2022
    2024 <- train 2021-2023
    2025 <- train 2021-2024
* Final live engine is fit only after that validation, using 2021-2025 replay
  rows.  2026 is never used to train or tune Betting Engine V1.
* Fixed model family: regularized logistic regression.  No AutoFS/search.
* Fixed economic action rule: candidate BET when model-estimated expected
  profit is >= +2% per unit at the observed price.  The +2% rule is not selected
  from historical ROI.
* Each market receives formal BET authority only if the walk-forward engine
  clears a predeclared historical gate.  Failed markets still publish fair-value
  and model-lean information, but cannot emit an official BET.

Systems/market integration
--------------------------
The engine is one decision object, but it preserves information independence:
* CORE/STAT enter through the frozen fair-value predictions + parity features.
* MARKET enters through current price/line, opening/current movement, key-number
  crossings, and break-even probability.
* Validated system-family triggers enter as explicit support/conflict features.
They are not counted as extra votes after the model decision.

Automatic wagering and automatic model promotion are always disabled.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
try:
    from google.api_core.exceptions import PreconditionFailed
    from google.cloud import bigquery
except ModuleNotFoundError:  # local/offline validation; Cloud Run supplies these
    class PreconditionFailed(Exception):
        pass
    bigquery = None

import nfl_production_v1 as prod

SOURCE_TAG = "nfl-betting-engine-v1.0-unified-decision-20261002"
ENGINE_VERSION = "NFL_BETTING_ENGINE_V1"
PROJECT = "sharplogger"
DATASET = "sharp_data"
MARKET_SOURCE = f"{PROJECT}.{DATASET}.sharp_moves_master"
PAIRED_SETTLEMENT_TABLE = f"{PROJECT}.{DATASET}.nfl_production_v1_paired_settlements"
SYSTEM_TRIGGER_V1 = f"{PROJECT}.{DATASET}.nfl_system_trigger_v1"
SYSTEM_TRIGGER_V2 = f"{PROJECT}.{DATASET}.nfl_system_family_trigger_v2"

PREFIX = "production/nfl/v1/betting_engine"
ARTIFACT_OBJECT = f"{PREFIX}/current_engine.joblib"
META_OBJECT = f"{PREFIX}/current_engine.json"
CURRENT_OBJECT = f"{PREFIX}/current_state.json"
EVENT_PREFIX = f"{PREFIX}/events"
SETTLEMENT_PREFIX = f"{PREFIX}/settlements"

TRAIN_SEASONS = (2021, 2022, 2023, 2024, 2025)
VALIDATION_SEASONS = (2023, 2024, 2025)
EV_BET_MIN = 0.02
LOGISTIC_C = 0.35

DIFF_FEATURES = (
    "Rest_Differential_Days", "WinPct_Prior_Diff", "ATS_WinPct_Prior_Diff",
    "Avg_SU_Margin_Last5_Diff", "Off_YPP_vs_Opp_Def_Last3_Diff",
    "Def_YPP_vs_Opp_Off_Last3_Diff", "Prior_Season_WinPct_Diff",
)

SPREAD_FEATURES = (
    "abs_edge", "model_disagreement_abs", "models_direction_agree",
    "selected_is_home", "selected_is_favorite", "market_spread_abs",
    "opening_spread_abs", "line_move_toward_selected", "crossed_key_3",
    "crossed_key_7", "week_number", "is_division_game",
    "rest_diff_selected", "winpct_diff_selected", "ats_winpct_diff_selected",
    "margin5_diff_selected", "off_ypp_matchup_selected",
    "def_ypp_matchup_selected", "prior_season_winpct_selected",
    "system_net_support_selected", "system_trigger_count",
)
H2H_FEATURES = (
    "abs_edge", "model_disagreement_abs", "models_direction_agree",
    "selected_is_home", "selected_market_probability", "selected_price_implied",
    "market_prob_move_toward_selected", "week_number", "is_division_game",
    "rest_diff_selected", "winpct_diff_selected", "margin5_diff_selected",
    "off_ypp_matchup_selected", "def_ypp_matchup_selected",
    "prior_season_winpct_selected", "system_net_support_selected",
    "system_trigger_count",
)
TOTAL_FEATURES = (
    "abs_edge", "model_disagreement_abs", "models_direction_agree",
    "selected_over", "market_total", "opening_total",
    "line_move_toward_selected", "week_number", "is_division_game",
    "rest_diff_abs", "avg_points_for_last5", "avg_points_against_last5",
    "opp_avg_points_for_last5", "opp_avg_points_against_last5",
    "avg_off_ypp_last3", "avg_def_ypp_last3", "opp_avg_off_ypp_last3",
    "opp_avg_def_ypp_last3", "system_net_support_selected",
    "system_trigger_count",
)
FEATURES = {"SPREADS": SPREAD_FEATURES, "H2H": H2H_FEATURES, "TOTALS": TOTAL_FEATURES}
MIN_BETS = {"SPREADS": 35, "H2H": 25, "TOTALS": 35}
MIN_FINAL_YEAR_BETS = {"SPREADS": 10, "H2H": 8, "TOTALS": 10}


def _num(x):
    try:
        z = float(x)
        return z if math.isfinite(z) else np.nan
    except Exception:
        return np.nan


def _sha(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _norm(x) -> str:
    try:
        import nfl_live_feature_parity_v1 as _parity
        return _parity._norm_name(x)
    except Exception:
        import re
        s=str(x or "").lower().strip()
        return re.sub(r"[^a-z0-9]+"," ",s).strip()


def _american_implied(o):
    o = _num(o)
    if not math.isfinite(o) or o == 0:
        return np.nan
    return (-o) / ((-o) + 100.0) if o < 0 else 100.0 / (o + 100.0)


def _american_profit_if_win(o):
    o = _num(o)
    if not math.isfinite(o) or o == 0:
        return np.nan
    return 100.0 / (-o) if o < 0 else o / 100.0


def _key_cross(a, b, key):
    a, b = _num(a), _num(b)
    if not (math.isfinite(a) and math.isfinite(b)):
        return 0.0
    for k in (float(key), -float(key)):
        if (a - k) * (b - k) < 0:
            return 1.0
    return 0.0


def _write_json(storage_client, bucket_name, name, payload):
    storage_client.bucket(bucket_name).blob(name).upload_from_string(
        json.dumps(payload, sort_keys=True, indent=2, default=str), content_type="application/json"
    )
    return f"gs://{bucket_name}/{name}"


def _read_json(storage_client, bucket_name, name):
    b = storage_client.bucket(bucket_name).blob(name)
    if not b.exists():
        return None
    try:
        return json.loads(b.download_as_text())
    except Exception:
        return None


def _write_immutable_json(storage_client, bucket_name, name, payload):
    b = storage_client.bucket(bucket_name).blob(name)
    try:
        b.upload_from_string(
            json.dumps(payload, sort_keys=True, indent=2, default=str),
            content_type="application/json", if_generation_match=0,
        )
        return True
    except PreconditionFailed:
        return False


def _historical_system_context(bq_client) -> dict[str, dict]:
    """Rebuild only the three system families used by the live engine.

    This implementation is deliberately self-contained so Production does not
    depend on the old research miner/intelligence modules. Every state variable
    is shifted before the current game result is consumed.
    """
    from nfl_feature_audit_v1 import VIEW
    cols=("Season","Season_Stage","Source_Name","Source_Game_ID","Game_Date","Historical_Core_Eligible",
          "Team_Norm","Opponent_Norm","Is_Home","Is_Away","Team_Score","Opponent_Score",
          "Opening_Spread","Is_Division_Game","Week_Number")
    schema={f.name for f in bq_client.get_table(VIEW).schema}
    missing=sorted(set(cols)-schema)
    if missing:
        raise RuntimeError("[NFL-BET-ENGINE-V1-HOLD] SYSTEM_CONTEXT_COLUMNS_MISSING "+str(missing))
    q="SELECT "+", ".join(f"`{c}`" for c in cols)+f" FROM `{VIEW}` WHERE Season BETWEEN 2017 AND 2025 AND Historical_Core_Eligible=1 AND Team_Score IS NOT NULL AND Opponent_Score IS NOT NULL ORDER BY Season, Game_Date, Source_Name, Source_Game_ID, Team_Norm"
    d=bq_client.query(q).to_dataframe(create_bqstorage_client=False)
    if d.empty:
        raise RuntimeError("[NFL-BET-ENGINE-V1-HOLD] SYSTEM_CONTEXT_EMPTY")
    d["Season"]=pd.to_numeric(d.Season,errors="coerce").astype(int)
    d["Game_Date"]=pd.to_datetime(d.Game_Date,errors="coerce")
    d["physical_game_id"]=d[["Season","Source_Name","Source_Game_ID"]].astype(str).agg("|".join,axis=1)
    d["actual_margin"]=pd.to_numeric(d.Team_Score,errors="coerce")-pd.to_numeric(d.Opponent_Score,errors="coerce")
    d["opening_dog"]=pd.to_numeric(d.Opening_Spread,errors="coerce").gt(0).astype(float)
    d=d.sort_values(["Season","Team_Norm","Game_Date","Source_Name","Source_Game_ID"],kind="mergesort").copy()
    grp=d.groupby(["Season","Team_Norm"],sort=False,dropna=False)
    d["prev_open_dog"]=grp["opening_dog"].shift(1)
    d["prev_margin"]=grp["actual_margin"].shift(1)
    out={}
    for gid,g in d.groupby("physical_game_id",sort=False):
        home_rows=g.loc[pd.to_numeric(g.Is_Home,errors="coerce").eq(1)]
        if home_rows.empty:
            continue
        h=home_rows.iloc[0]; home=_norm(h.Team_Norm); away=_norm(h.Opponent_Norm)
        rec={"spread_home_support":0.0,"spread_trigger_count":0.0,"total_under_support":0.0,"total_trigger_count":0.0}
        # Role Flip dog -> favorite: fade the side that changed role.
        for _,r in g.iterrows():
            op=_num(r.get("Opening_Spread")); prevdog=_num(r.get("prev_open_dog"))
            if prevdog==1.0 and math.isfinite(op) and op<0:
                play=_norm(r.get("Opponent_Norm")); rec["spread_trigger_count"]+=1.0
                if play==home: rec["spread_home_support"]+=1.0
                elif play==away: rec["spread_home_support"]-=1.0
        # Home favorite off SU loss: fade the home favorite.
        hop=_num(h.get("Opening_Spread")); hprev=_num(h.get("prev_margin"))
        if math.isfinite(hop) and hop<0 and math.isfinite(hprev) and hprev<0:
            rec["spread_trigger_count"]+=1.0; rec["spread_home_support"]-=1.0
        # Early division UNDER.
        wk=_num(h.get("Week_Number")); div=_num(h.get("Is_Division_Game"))
        if div==1.0 and math.isfinite(wk) and 1<=wk<=4:
            rec["total_under_support"]=1.0; rec["total_trigger_count"]=1.0
        out[str(gid)]=rec
    return out

def _merge_context(replay_rows: pd.DataFrame, games: pd.DataFrame, system_context: dict[str, dict]) -> pd.DataFrame:
    cols = ["physical_game_id", *prod.PRODUCTION_FEATURES]
    missing = [c for c in cols if c not in games.columns]
    if missing:
        raise RuntimeError("[NFL-BET-ENGINE-V1-HOLD] CONTEXT_COLUMNS_MISSING " + str(missing))
    c = games.loc[:, cols].drop_duplicates("physical_game_id").copy()
    d = replay_rows.merge(c, on="physical_game_id", how="left", validate="one_to_one")
    if d[list(prod.PRODUCTION_FEATURES)].isna().all(axis=1).any():
        raise RuntimeError("[NFL-BET-ENGINE-V1-HOLD] REPLAY_CONTEXT_JOIN_FAILED")
    sys = pd.DataFrame.from_dict(system_context, orient="index") if system_context else pd.DataFrame()
    if not sys.empty:
        sys.index.name = "physical_game_id"; sys = sys.reset_index()
        d = d.merge(sys, on="physical_game_id", how="left")
    for c0 in ("spread_home_support", "spread_trigger_count", "total_under_support", "total_trigger_count"):
        if c0 not in d: d[c0] = 0.0
        d[c0] = pd.to_numeric(d[c0], errors="coerce").fillna(0.0)
    return d


def _orient(v, sign):
    z = _num(v)
    return z * sign if math.isfinite(z) else np.nan


def _historical_market_rows(d: pd.DataFrame, market: str) -> pd.DataFrame:
    rows = []
    for _, r in d.iterrows():
        base = {
            "physical_game_id": r.get("physical_game_id"),
            "season": int(r.get("season")),
            "week_number": _num(r.get("week_number")),
            "home_team": r.get("home_team"), "away_team": r.get("away_team"),
        }
        if market == "SPREADS":
            line = _num(r.get("close_spread")); op = _num(r.get("opening_spread"))
            fair = _num(r.get("frozen_fair_margin")); alt = _num(r.get("adaptive_fair_margin")); actual = _num(r.get("actual_margin"))
            if not all(math.isfinite(x) for x in (line, fair, actual)): continue
            edge_home = fair + line
            if abs(edge_home) <= 1e-12: continue
            sign = 1.0 if edge_home > 0 else -1.0
            settle = sign * (actual + line)
            if abs(settle) <= 1e-12: continue
            selected_line = line if sign > 0 else -line
            adaptive_edge = alt + line if math.isfinite(alt) else np.nan
            feat = {
                "abs_edge": abs(edge_home),
                "model_disagreement_abs": abs(alt - fair) if math.isfinite(alt) else np.nan,
                "models_direction_agree": float(math.isfinite(adaptive_edge) and np.sign(adaptive_edge) == np.sign(edge_home)),
                "selected_is_home": float(sign > 0), "selected_is_favorite": float(selected_line < 0),
                "market_spread_abs": abs(line), "opening_spread_abs": abs(op) if math.isfinite(op) else np.nan,
                "line_move_toward_selected": -sign * (line - op) if math.isfinite(op) else np.nan,
                "crossed_key_3": _key_cross(op, line, 3), "crossed_key_7": _key_cross(op, line, 7),
                "week_number": _num(r.get("week_number")), "is_division_game": _num(r.get("Is_Division_Game")),
                "rest_diff_selected": _orient(r.get("Rest_Differential_Days"), sign),
                "winpct_diff_selected": _orient(r.get("WinPct_Prior_Diff"), sign),
                "ats_winpct_diff_selected": _orient(r.get("ATS_WinPct_Prior_Diff"), sign),
                "margin5_diff_selected": _orient(r.get("Avg_SU_Margin_Last5_Diff"), sign),
                "off_ypp_matchup_selected": _orient(r.get("Off_YPP_vs_Opp_Def_Last3_Diff"), sign),
                "def_ypp_matchup_selected": _orient(r.get("Def_YPP_vs_Opp_Off_Last3_Diff"), sign),
                "prior_season_winpct_selected": _orient(r.get("Prior_Season_WinPct_Diff"), sign),
                "system_net_support_selected": sign * _num(r.get("spread_home_support")),
                "system_trigger_count": _num(r.get("spread_trigger_count")),
            }
            rows.append({**base, **feat, "target": float(settle > 0), "profit_if_win": 100.0/110.0,
                         "selected": r.get("home_team") if sign > 0 else r.get("away_team"),
                         "market_value": selected_line, "model_value": fair * sign,
                         "raw_edge": abs(edge_home), "selected_price": -110.0})
        elif market == "TOTALS":
            line = _num(r.get("close_total")); op = _num(r.get("opening_total")); fair = _num(r.get("frozen_fair_total")); alt = _num(r.get("adaptive_fair_total")); actual = _num(r.get("actual_total"))
            if not all(math.isfinite(x) for x in (line, fair, actual)): continue
            edge = fair - line
            if abs(edge) <= 1e-12: continue
            sign = 1.0 if edge > 0 else -1.0
            settle = sign * (actual - line)
            if abs(settle) <= 1e-12: continue
            adaptive_edge = alt - line if math.isfinite(alt) else np.nan
            sys_under = _num(r.get("total_under_support")); sys_net = -sign * sys_under if math.isfinite(sys_under) else 0.0
            feat = {
                "abs_edge": abs(edge), "model_disagreement_abs": abs(alt-fair) if math.isfinite(alt) else np.nan,
                "models_direction_agree": float(math.isfinite(adaptive_edge) and np.sign(adaptive_edge) == np.sign(edge)),
                "selected_over": float(sign > 0), "market_total": line, "opening_total": op,
                "line_move_toward_selected": sign * (line-op) if math.isfinite(op) else np.nan,
                "week_number": _num(r.get("week_number")), "is_division_game": _num(r.get("Is_Division_Game")),
                "rest_diff_abs": abs(_num(r.get("Rest_Differential_Days"))) if math.isfinite(_num(r.get("Rest_Differential_Days"))) else np.nan,
                "avg_points_for_last5": _num(r.get("Avg_Points_For_Last5_Prior")),
                "avg_points_against_last5": _num(r.get("Avg_Points_Against_Last5_Prior")),
                "opp_avg_points_for_last5": _num(r.get("Opp_Avg_Points_For_Last5_Prior")),
                "opp_avg_points_against_last5": _num(r.get("Opp_Avg_Points_Against_Last5_Prior")),
                "avg_off_ypp_last3": _num(r.get("Avg_Off_YPP_Last3_Prior")), "avg_def_ypp_last3": _num(r.get("Avg_Def_YPP_Last3_Prior")),
                "opp_avg_off_ypp_last3": _num(r.get("Opp_Avg_Off_YPP_Last3_Prior")), "opp_avg_def_ypp_last3": _num(r.get("Opp_Avg_Def_YPP_Last3_Prior")),
                "system_net_support_selected": sys_net, "system_trigger_count": _num(r.get("total_trigger_count")),
            }
            rows.append({**base, **feat, "target": float(settle > 0), "profit_if_win": 100.0/110.0,
                         "selected": "OVER" if sign > 0 else "UNDER", "market_value": line, "model_value": fair,
                         "raw_edge": abs(edge), "selected_price": -110.0})
        elif market == "H2H":
            ref = _num(r.get("close_novig_home_probability")); opref = _num(r.get("opening_novig_home_probability")); fair = _num(r.get("frozen_home_win_probability")); alt = _num(r.get("adaptive_home_win_probability")); y = _num(r.get("home_win_label"))
            if not all(math.isfinite(x) for x in (ref, fair, y)): continue
            edge_home = fair - ref
            if abs(edge_home) <= 1e-12: continue
            sign = 1.0 if edge_home > 0 else -1.0
            target = y if sign > 0 else 1.0-y
            odds = _num(r.get("home_close_ml" if sign > 0 else "away_close_ml")); pwin = _american_profit_if_win(odds)
            if not math.isfinite(pwin): continue
            sel_ref = ref if sign > 0 else 1.0-ref
            sel_imp = _american_implied(odds)
            adaptive_edge = alt-ref if math.isfinite(alt) else np.nan
            feat = {
                "abs_edge": abs(edge_home), "model_disagreement_abs": abs(alt-fair) if math.isfinite(alt) else np.nan,
                "models_direction_agree": float(math.isfinite(adaptive_edge) and np.sign(adaptive_edge) == np.sign(edge_home)),
                "selected_is_home": float(sign > 0), "selected_market_probability": sel_ref,
                "selected_price_implied": sel_imp, "market_prob_move_toward_selected": sign*(ref-opref) if math.isfinite(opref) else np.nan,
                "week_number": _num(r.get("week_number")), "is_division_game": _num(r.get("Is_Division_Game")),
                "rest_diff_selected": _orient(r.get("Rest_Differential_Days"), sign), "winpct_diff_selected": _orient(r.get("WinPct_Prior_Diff"), sign),
                "margin5_diff_selected": _orient(r.get("Avg_SU_Margin_Last5_Diff"), sign),
                "off_ypp_matchup_selected": _orient(r.get("Off_YPP_vs_Opp_Def_Last3_Diff"), sign),
                "def_ypp_matchup_selected": _orient(r.get("Def_YPP_vs_Opp_Off_Last3_Diff"), sign),
                "prior_season_winpct_selected": _orient(r.get("Prior_Season_WinPct_Diff"), sign),
                "system_net_support_selected": sign * _num(r.get("spread_home_support")),
                "system_trigger_count": _num(r.get("spread_trigger_count")),
            }
            rows.append({**base, **feat, "target": float(target), "profit_if_win": pwin,
                         "selected": r.get("home_team") if sign > 0 else r.get("away_team"),
                         "market_value": sel_ref, "model_value": fair if sign > 0 else 1.0-fair,
                         "raw_edge": abs(edge_home), "selected_price": odds})
        else:
            raise ValueError(market)
    return pd.DataFrame(rows)


def build_historical_engine_rows(*, bq_client, replay_rows: pd.DataFrame, games: pd.DataFrame) -> dict[str, pd.DataFrame]:
    systems = _historical_system_context(bq_client)
    d = _merge_context(replay_rows, games, systems)
    return {m: _historical_market_rows(d, m) for m in ("SPREADS", "H2H", "TOTALS")}


def _fit(df: pd.DataFrame, market: str) -> dict:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    f = FEATURES[market]
    x = df.loc[:, f].apply(pd.to_numeric, errors="coerce").astype(float).replace([np.inf,-np.inf], np.nan)
    med = x.median(axis=0).fillna(0.0).astype(float)
    x = x.fillna(med)
    y = pd.to_numeric(df.target, errors="coerce").to_numpy(float)
    if len(df) < 100 or len(np.unique(y[np.isfinite(y)])) < 2:
        raise RuntimeError(f"[NFL-BET-ENGINE-V1-HOLD] INSUFFICIENT_TRAINING market={market} n={len(df)}")
    sc = StandardScaler().fit(x.to_numpy(float))
    model = LogisticRegression(C=LOGISTIC_C, max_iter=1200, solver="lbfgs", random_state=41).fit(sc.transform(x.to_numpy(float)), y)
    return {"market": market, "features": list(f), "medians": med.to_dict(), "scaler": sc, "model": model}


def _predict(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    f = bundle["features"]
    x = df.loc[:, f].apply(pd.to_numeric, errors="coerce").astype(float).replace([np.inf,-np.inf], np.nan)
    med = pd.Series(bundle["medians"], dtype=float)
    x = x.fillna(med).fillna(0.0)
    p = bundle["model"].predict_proba(bundle["scaler"].transform(x.to_numpy(float)))
    idx = list(bundle["model"].classes_).index(1.0)
    return np.clip(np.asarray(p[:,idx], float), .001, .999)


def _classification(y, p):
    from sklearn.metrics import roc_auc_score
    y=np.asarray(y,float); p=np.asarray(p,float); m=np.isfinite(y)&np.isfinite(p); y=y[m]; p=np.clip(p[m],.001,.999)
    if not len(y): return {"n":0}
    ll=float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p)))); br=float(np.mean((p-y)**2))
    auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan
    return {"n":int(len(y)),"log_loss":round(ll,6),"brier":round(br,6),"auc":round(auc,6) if math.isfinite(auc) else None}


def _add_economics(d: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    z=d.copy(); z["bet_win_probability"]=p
    profit=pd.to_numeric(z.profit_if_win,errors="coerce").to_numpy(float)
    z["expected_value"]=p*profit-(1.0-p)
    z["break_even_probability"]=1.0/(1.0+profit)
    z["bet_candidate"]=z.expected_value.ge(EV_BET_MIN)
    target=pd.to_numeric(z.target,errors="coerce").to_numpy(float)
    z["realized_profit"]=np.where(target>=.5,profit,-1.0)
    return z


def _bet_perf(z: pd.DataFrame) -> dict:
    b=z.loc[z.bet_candidate].copy()
    if b.empty:
        return {"n":0,"wins":0,"losses":0,"hit_rate":None,"roi_per_unit":None}
    y=pd.to_numeric(b.target,errors="coerce")
    profits=pd.to_numeric(b.realized_profit,errors="coerce")
    wins=int(y.ge(.5).sum()); losses=int(y.lt(.5).sum()); n=wins+losses
    return {"n":n,"wins":wins,"losses":losses,"hit_rate":round(wins/n,6) if n else None,
            "roi_per_unit":round(float(profits.mean()),6) if len(profits) else None,
            "avg_expected_value":round(float(pd.to_numeric(b.expected_value,errors="coerce").mean()),6)}


def _walk_forward(market_rows: dict[str,pd.DataFrame]) -> tuple[dict, dict[str,pd.DataFrame]]:
    summary={}; scored={}
    for market,d0 in market_rows.items():
        pieces=[]; seasons={}
        for sy in VALIDATION_SEASONS:
            tr=d0.loc[pd.to_numeric(d0.season,errors="coerce").lt(sy)].copy()
            va=d0.loc[pd.to_numeric(d0.season,errors="coerce").eq(sy)].copy()
            if tr.empty or va.empty: continue
            b=_fit(tr,market); p=_predict(b,va); zz=_add_economics(va,p); pieces.append(zz)
            seasons[str(sy)]={"predictive":_classification(zz.target,zz.bet_win_probability),"bets":_bet_perf(zz)}
        oos=pd.concat(pieces,ignore_index=True,sort=False) if pieces else pd.DataFrame()
        pred=_classification(oos.target,oos.bet_win_probability) if not oos.empty else {"n":0}
        bets=_bet_perf(oos) if not oos.empty else {"n":0}
        positive=sum(1 for x in seasons.values() if (x.get("bets") or {}).get("roi_per_unit") is not None and x["bets"]["roi_per_unit"]>=0)
        final=(seasons.get("2025") or {}).get("bets") or {"n":0,"roi_per_unit":None}
        gate=(
            int(bets.get("n") or 0)>=MIN_BETS[market]
            and bets.get("roi_per_unit") is not None and float(bets["roi_per_unit"])>0
            and pred.get("log_loss") is not None and float(pred["log_loss"])<0.693147
            and positive>=2
            and int(final.get("n") or 0)>=MIN_FINAL_YEAR_BETS[market]
            and final.get("roi_per_unit") is not None and float(final["roi_per_unit"])>=0
        )
        summary[market]={
            "predictive":pred,"bets":bets,"by_season":seasons,"positive_roi_seasons":positive,
            "authority_status":"BET_AUTHORITY_OPEN" if gate else "SHADOW_ONLY",
            "betting_decision_authority":bool(gate),"fixed_ev_threshold":EV_BET_MIN,
            "gate":{
                "min_bets":MIN_BETS[market],"min_final_year_bets":MIN_FINAL_YEAR_BETS[market],
                "requires_overall_roi_gt_0":True,"requires_log_loss_lt_random":True,
                "requires_positive_roi_seasons":2,"requires_2025_roi_ge_0":True,
            },
        }
        scored[market]=oos
    return summary,scored


def _training_fingerprint(rows: dict[str,pd.DataFrame]) -> str:
    payload={}
    for market,d0 in sorted(rows.items()):
        cols=["physical_game_id","season",*FEATURES[market],"target","profit_if_win"]
        d=d0.loc[:,cols].copy().sort_values(["season","physical_game_id"],kind="mergesort").reset_index(drop=True)
        recs=[]
        for rec in d.to_dict("records"):
            z={}
            for k,v in rec.items():
                if isinstance(v,(float,np.floating,int,np.integer)):
                    fv=float(v); z[k]=None if not math.isfinite(fv) else round(fv,10)
                elif pd.isna(v): z[k]=None
                else: z[k]=str(v)
            recs.append(z)
        payload[market]=recs
    return _sha(payload)


def train_publish_engine(*, bq_client, storage_client, bucket_name: str, replay_rows: pd.DataFrame, games: pd.DataFrame, log_func=print) -> dict:
    log_func("[NFL-BET-ENGINE-V1-PREFLIGHT] "+json.dumps({
        "status":"START","source_tag":SOURCE_TAG,"base_prediction_source":"LEAK_SAFE_PRODUCTION_REPLAY",
        "train_seasons":list(TRAIN_SEASONS),"validation_seasons":list(VALIDATION_SEASONS),
        "model_family":"REGULARIZED_LOGISTIC","fixed_ev_bet_min":EV_BET_MIN,
        "automatic_execution":False,
    },sort_keys=True))
    rows=build_historical_engine_rows(bq_client=bq_client,replay_rows=replay_rows,games=games)
    oos,scored=_walk_forward(rows)
    log_func("[NFL-BET-ENGINE-V1-OOS] "+json.dumps(oos,sort_keys=True,default=str))
    bundles={m:_fit(d.loc[pd.to_numeric(d.season,errors="coerce").isin(TRAIN_SEASONS)].copy(),m) for m,d in rows.items()}
    gates={m:{"authority_status":oos[m]["authority_status"],"betting_decision_authority":oos[m]["betting_decision_authority"]} for m in oos}
    training_sha=_training_fingerprint(rows)
    stable_contract={
        "source_tag":SOURCE_TAG,"engine_version":ENGINE_VERSION,"production_contract_sha256":prod.production_contract()["contract_sha256"],
        "training_data_sha256":training_sha,"training_seasons":list(TRAIN_SEASONS),"validation_seasons":list(VALIDATION_SEASONS),
        "market_gates":gates,"historical_oos":oos,"fixed_ev_bet_min":EV_BET_MIN,"model_family":"REGULARIZED_LOGISTIC",
        "system_features":"ROLE_FLIP_HOME_FAVORITE_FADE_EARLY_DIVISION_UNDER_ONLY",
    }
    registry_sha=_sha(stable_contract)
    artifact_meta={**stable_contract,"engine_registry_sha256":registry_sha}
    payload={"metadata":artifact_meta,"models":bundles}
    bio=io.BytesIO(); joblib.dump(payload,bio,compress=3); raw=bio.getvalue(); sha=hashlib.sha256(raw).hexdigest()
    storage_client.bucket(bucket_name).blob(ARTIFACT_OBJECT).upload_from_string(raw,content_type="application/octet-stream")
    meta={**artifact_meta,"status":"NFL_BETTING_ENGINE_V1_ACTIVE","published_at_utc":datetime.now(timezone.utc).isoformat(),
          "artifact_sha256":sha,"artifact_uri":f"gs://{bucket_name}/{ARTIFACT_OBJECT}","automatic_execution":False,
          "betting_decision_authority":bool(any(x["betting_decision_authority"] for x in gates.values()))}
    meta["metadata_uri"]=f"gs://{bucket_name}/{META_OBJECT}"
    _write_json(storage_client,bucket_name,META_OBJECT,meta)
    log_func("[NFL-BET-ENGINE-V1-PUBLISH] "+json.dumps({
        "status":meta["status"],"artifact_uri":meta["artifact_uri"],"artifact_sha256":sha,
        "market_gates":gates,"betting_decision_authority":meta["betting_decision_authority"],"automatic_execution":False,
    },sort_keys=True,default=str))
    return meta


def _load_engine(storage_client,bucket_name):
    meta=_read_json(storage_client,bucket_name,META_OBJECT)
    if not isinstance(meta,dict) or meta.get("source_tag")!=SOURCE_TAG:
        raise RuntimeError("[NFL-BET-ENGINE-V1-HOLD] ENGINE_METADATA_MISSING_OR_STALE")
    blob=storage_client.bucket(bucket_name).blob(ARTIFACT_OBJECT)
    if not blob.exists(): raise RuntimeError("[NFL-BET-ENGINE-V1-HOLD] ENGINE_ARTIFACT_MISSING")
    raw=blob.download_as_bytes()
    if hashlib.sha256(raw).hexdigest()!=meta.get("artifact_sha256"):
        raise RuntimeError("[NFL-BET-ENGINE-V1-HOLD] ENGINE_ARTIFACT_SHA_MISMATCH")
    obj=joblib.load(io.BytesIO(raw))
    if obj.get("metadata",{}).get("source_tag")!=SOURCE_TAG:
        raise RuntimeError("[NFL-BET-ENGINE-V1-HOLD] ENGINE_ARTIFACT_TAG_MISMATCH")
    obj["metadata"]={**(obj.get("metadata") or {}),**meta}
    return obj


def _resolve_market_schema(client):
    cols={f.name for f in client.get_table(MARKET_SOURCE).schema}
    def first(*xs): return next((x for x in xs if x in cols),None)
    m={
        "sport":first("Sport"),"market":first("Market"),"outcome":first("Outcome"),"value":first("Value"),
        "odds":first("Odds_Price","Odds","Price"),"book":first("Bookmaker","Book","Sportsbook"),
        "game_start":first("Game_Start","Commence_Hour","feat_Game_Start"),
        "snapshot":first("Snapshot_Timestamp","snapshot_timestamp","Observed_At","Captured_At"),
        "home":first("Home_Team_Norm","Home_Team","Home"),"away":first("Away_Team_Norm","Away_Team","Away"),
    }
    missing=[k for k,v in m.items() if not v]
    if missing: raise RuntimeError("[NFL-BET-ENGINE-V1-HOLD] MARKET_SCHEMA_MISSING "+str(missing))
    return m


def _fetch_market_rows(client,now,lookahead_days=8):
    m=_resolve_market_schema(client)
    sql=f"""
      SELECT CAST(`{m['market']}` AS STRING) market, CAST(`{m['outcome']}` AS STRING) outcome,
             SAFE_CAST(`{m['value']}` AS FLOAT64) value, SAFE_CAST(`{m['odds']}` AS FLOAT64) odds,
             CAST(`{m['book']}` AS STRING) bookmaker, SAFE_CAST(`{m['game_start']}` AS TIMESTAMP) game_start,
             SAFE_CAST(`{m['snapshot']}` AS TIMESTAMP) snapshot_ts, CAST(`{m['home']}` AS STRING) home_team,
             CAST(`{m['away']}` AS STRING) away_team
      FROM `{MARKET_SOURCE}`
      WHERE UPPER(TRIM(CAST(`{m['sport']}` AS STRING)))='NFL'
        AND SAFE_CAST(`{m['game_start']}` AS TIMESTAMP)>@now
        AND SAFE_CAST(`{m['game_start']}` AS TIMESTAMP)<=TIMESTAMP_ADD(@now, INTERVAL {int(lookahead_days)} DAY)
        AND SAFE_CAST(`{m['snapshot']}` AS TIMESTAMP)<SAFE_CAST(`{m['game_start']}` AS TIMESTAMP)
    """
    cfg=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("now","TIMESTAMP",pd.to_datetime(now,utc=True).to_pydatetime())])
    d=client.query(sql,job_config=cfg).to_dataframe(create_bqstorage_client=False)
    if d.empty:return d
    d["game_start"]=pd.to_datetime(d.game_start,utc=True,errors="coerce"); d["snapshot_ts"]=pd.to_datetime(d.snapshot_ts,utc=True,errors="coerce")
    d["home_key"]=d.home_team.map(_norm); d["away_key"]=d.away_team.map(_norm); d["outcome_key"]=d.outcome.map(_norm); d["market_norm"]=d.market.astype(str).str.lower().str.strip()
    return d.dropna(subset=["game_start","snapshot_ts"]).reset_index(drop=True)


def _consensus_market(client,now):
    d=_fetch_market_rows(client,now)
    out={}
    if d.empty:return out
    # earliest and latest quote per book/outcome preserve opening vs current state.
    d=d.sort_values("snapshot_ts")
    first=d.drop_duplicates(["game_start","home_key","away_key","market_norm","outcome_key","bookmaker"],keep="first")
    last=d.drop_duplicates(["game_start","home_key","away_key","market_norm","outcome_key","bookmaker"],keep="last")
    for key,gall in d.groupby(["game_start","home_key","away_key"],sort=False):
        gs,hk,ak=key; rec={"game_start":gs,"home_key":hk,"away_key":ak}
        for label,frame in (("open",first),("current",last)):
            g=frame[(frame.game_start.eq(gs))&(frame.home_key.eq(hk))&(frame.away_key.eq(ak))]
            sp=g[g.market_norm.eq("spreads")].copy(); vals=[]
            for _,r in sp.iterrows():
                v=_num(r.value)
                if not math.isfinite(v):continue
                if r.outcome_key==hk: vals.append(v)
                elif r.outcome_key==ak: vals.append(-v)
            if vals:
                line=float(np.median(vals)); rec[label+"_home_spread"]=line
                hodd=pd.to_numeric(sp.loc[sp.outcome_key.eq(hk)&pd.to_numeric(sp.value,errors="coerce").sub(line).abs().lt(.011),"odds"],errors="coerce").dropna()
                aodd=pd.to_numeric(sp.loc[sp.outcome_key.eq(ak)&pd.to_numeric(sp.value,errors="coerce").add(line).abs().lt(.011),"odds"],errors="coerce").dropna()
                rec[label+"_home_spread_odds"]=float(hodd.max()) if len(hodd) else -110.0
                rec[label+"_away_spread_odds"]=float(aodd.max()) if len(aodd) else -110.0
            to=g[g.market_norm.eq("totals")].copy(); ov=to[to.outcome.astype(str).str.lower().str.contains("over",na=False)]; uv=to[to.outcome.astype(str).str.lower().str.contains("under",na=False)]
            tv=pd.to_numeric(ov.value,errors="coerce").dropna()
            if len(tv):
                line=float(np.median(tv)); rec[label+"_total"]=line
                oo=pd.to_numeric(ov.loc[pd.to_numeric(ov.value,errors="coerce").sub(line).abs().lt(.011),"odds"],errors="coerce").dropna()
                uo=pd.to_numeric(uv.loc[pd.to_numeric(uv.value,errors="coerce").sub(line).abs().lt(.011),"odds"],errors="coerce").dropna()
                rec[label+"_over_odds"]=float(oo.max()) if len(oo) else -110.0; rec[label+"_under_odds"]=float(uo.max()) if len(uo) else -110.0
            h2=g[g.market_norm.eq("h2h")].copy(); probs=[]; hos=[]; aos=[]
            for book,bg in h2.groupby("bookmaker",sort=False):
                ho=pd.to_numeric(bg.loc[bg.outcome_key.eq(hk),"odds"],errors="coerce").dropna(); ao=pd.to_numeric(bg.loc[bg.outcome_key.eq(ak),"odds"],errors="coerce").dropna()
                if len(ho): hos.extend(ho.tolist())
                if len(ao): aos.extend(ao.tolist())
                if len(ho) and len(ao):
                    ih=_american_implied(ho.iloc[-1]); ia=_american_implied(ao.iloc[-1])
                    if math.isfinite(ih) and math.isfinite(ia) and ih+ia>0: probs.append(ih/(ih+ia))
            if probs: rec[label+"_home_novig_probability"]=float(np.median(probs))
            if hos: rec[label+"_home_ml"]=float(max(hos))
            if aos: rec[label+"_away_ml"]=float(max(aos))
        out[(pd.Timestamp(gs).round("s").isoformat(),hk,ak)]=rec
    return out


def _live_system_context(client,now,lookahead_days=8):
    end=pd.to_datetime(now,utc=True)+pd.Timedelta(days=lookahead_days); out={}
    for table in (SYSTEM_TRIGGER_V1,SYSTEM_TRIGGER_V2):
        try:
            q=f"SELECT * FROM `{table}` WHERE game_start>=@lo AND game_start<=@hi ORDER BY captured_at"
            cfg=bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("lo","TIMESTAMP",(pd.to_datetime(now,utc=True)-pd.Timedelta(days=1)).to_pydatetime()),
                bigquery.ScalarQueryParameter("hi","TIMESTAMP",end.to_pydatetime()),])
            d=client.query(q,job_config=cfg).to_dataframe(create_bqstorage_client=False)
        except Exception:
            continue
        for _,r in d.iterrows():
            gs=pd.to_datetime(r.get("game_start"),utc=True,errors="coerce")
            if pd.isna(gs):continue
            hk=_norm(r.get("home_team")); ak=_norm(r.get("away_team")); k=(gs.round("s").isoformat(),hk,ak)
            rec=out.setdefault(k,{"spread_home_support":0.0,"spread_trigger_count":0.0,"total_under_support":0.0,"total_trigger_count":0.0,"labels":[]})
            market=str(r.get("market") or "SPREADS").upper(); direction=str(r.get("direction") or "").upper(); play=_norm(r.get("play_team")); target=_norm(r.get("target_team"))
            if not play and target:
                if direction=="FADE": play=ak if target==hk else hk if target==ak else ""
                else: play=target
            fam=str(r.get("system_family_id") or "SYSTEM")
            rec["labels"].append(fam)
            if market=="SPREADS":
                rec["spread_trigger_count"]+=1.0
                if play==hk:rec["spread_home_support"]+=1.0
                elif play==ak:rec["spread_home_support"]-=1.0
            elif market=="TOTALS":
                rec["total_trigger_count"]+=1.0
                if direction=="UNDER":rec["total_under_support"]+=1.0
                elif direction=="OVER":rec["total_under_support"]-=1.0
    return out


def _parse_features(p):
    x=p.get("features_json")
    if isinstance(x,dict): return x
    try:return json.loads(x or "{}")
    except Exception:return {}


def _live_feature_row(p,market,q,sys):
    f=_parse_features(p); home=p.get("home_team"); away=p.get("away_team")
    if market=="SPREADS":
        line=_num(q.get("current_home_spread")); op=_num(q.get("open_home_spread")); fair=_num(p.get("champion_fair_margin")); alt=_num(p.get("challenger_fair_margin"))
        if not (math.isfinite(line) and math.isfinite(fair)):return None
        edge=fair+line
        if abs(edge)<=1e-12:return None
        sign=1.0 if edge>0 else -1.0; adaptive_edge=alt+line if math.isfinite(alt) else np.nan; selected_line=line if sign>0 else -line
        odds=_num(q.get("current_home_spread_odds" if sign>0 else "current_away_spread_odds")); odds=odds if math.isfinite(odds) else -110.0
        feat={
            "abs_edge":abs(edge),"model_disagreement_abs":abs(alt-fair) if math.isfinite(alt) else np.nan,"models_direction_agree":float(math.isfinite(adaptive_edge) and np.sign(adaptive_edge)==np.sign(edge)),
            "selected_is_home":float(sign>0),"selected_is_favorite":float(selected_line<0),"market_spread_abs":abs(line),"opening_spread_abs":abs(op) if math.isfinite(op) else np.nan,
            "line_move_toward_selected":-sign*(line-op) if math.isfinite(op) else np.nan,"crossed_key_3":_key_cross(op,line,3),"crossed_key_7":_key_cross(op,line,7),
            "week_number":_num(f.get("Week_Number")),"is_division_game":_num(f.get("Is_Division_Game")),"rest_diff_selected":_orient(f.get("Rest_Differential_Days"),sign),
            "winpct_diff_selected":_orient(f.get("WinPct_Prior_Diff"),sign),"ats_winpct_diff_selected":_orient(f.get("ATS_WinPct_Prior_Diff"),sign),"margin5_diff_selected":_orient(f.get("Avg_SU_Margin_Last5_Diff"),sign),
            "off_ypp_matchup_selected":_orient(f.get("Off_YPP_vs_Opp_Def_Last3_Diff"),sign),"def_ypp_matchup_selected":_orient(f.get("Def_YPP_vs_Opp_Off_Last3_Diff"),sign),"prior_season_winpct_selected":_orient(f.get("Prior_Season_WinPct_Diff"),sign),
            "system_net_support_selected":sign*_num(sys.get("spread_home_support",0)),"system_trigger_count":_num(sys.get("spread_trigger_count",0)),
        }
        return {**feat,"selected":home if sign>0 else away,"market_value":selected_line,"model_value":fair*sign,"raw_edge":abs(edge),"selected_price":odds,"profit_if_win":_american_profit_if_win(odds) or 100/110}
    if market=="TOTALS":
        line=_num(q.get("current_total")); op=_num(q.get("open_total")); fair=_num(p.get("champion_fair_total")); alt=_num(p.get("challenger_fair_total"))
        if not (math.isfinite(line) and math.isfinite(fair)):return None
        edge=fair-line
        if abs(edge)<=1e-12:return None
        sign=1.0 if edge>0 else -1.0; adaptive_edge=alt-line if math.isfinite(alt) else np.nan; odds=_num(q.get("current_over_odds" if sign>0 else "current_under_odds")); odds=odds if math.isfinite(odds) else -110.0
        feat={
            "abs_edge":abs(edge),"model_disagreement_abs":abs(alt-fair) if math.isfinite(alt) else np.nan,"models_direction_agree":float(math.isfinite(adaptive_edge) and np.sign(adaptive_edge)==np.sign(edge)),
            "selected_over":float(sign>0),"market_total":line,"opening_total":op,"line_move_toward_selected":sign*(line-op) if math.isfinite(op) else np.nan,
            "week_number":_num(f.get("Week_Number")),"is_division_game":_num(f.get("Is_Division_Game")),"rest_diff_abs":abs(_num(f.get("Rest_Differential_Days"))) if math.isfinite(_num(f.get("Rest_Differential_Days"))) else np.nan,
            "avg_points_for_last5":_num(f.get("Avg_Points_For_Last5_Prior")),"avg_points_against_last5":_num(f.get("Avg_Points_Against_Last5_Prior")),"opp_avg_points_for_last5":_num(f.get("Opp_Avg_Points_For_Last5_Prior")),"opp_avg_points_against_last5":_num(f.get("Opp_Avg_Points_Against_Last5_Prior")),
            "avg_off_ypp_last3":_num(f.get("Avg_Off_YPP_Last3_Prior")),"avg_def_ypp_last3":_num(f.get("Avg_Def_YPP_Last3_Prior")),"opp_avg_off_ypp_last3":_num(f.get("Opp_Avg_Off_YPP_Last3_Prior")),"opp_avg_def_ypp_last3":_num(f.get("Opp_Avg_Def_YPP_Last3_Prior")),
            "system_net_support_selected":-sign*_num(sys.get("total_under_support",0)),"system_trigger_count":_num(sys.get("total_trigger_count",0)),
        }
        return {**feat,"selected":"OVER" if sign>0 else "UNDER","market_value":line,"model_value":fair,"raw_edge":abs(edge),"selected_price":odds,"profit_if_win":_american_profit_if_win(odds) or 100/110}
    if market=="H2H":
        ref=_num(q.get("current_home_novig_probability")); opref=_num(q.get("open_home_novig_probability")); fair=_num(p.get("champion_home_win_probability")); alt=_num(p.get("challenger_home_win_probability"))
        if not (math.isfinite(ref) and math.isfinite(fair)):return None
        edge=fair-ref
        if abs(edge)<=1e-12:return None
        sign=1.0 if edge>0 else -1.0; adaptive_edge=alt-ref if math.isfinite(alt) else np.nan; odds=_num(q.get("current_home_ml" if sign>0 else "current_away_ml")); pwin=_american_profit_if_win(odds)
        if not math.isfinite(pwin):return None
        sel_ref=ref if sign>0 else 1-ref
        feat={
            "abs_edge":abs(edge),"model_disagreement_abs":abs(alt-fair) if math.isfinite(alt) else np.nan,"models_direction_agree":float(math.isfinite(adaptive_edge) and np.sign(adaptive_edge)==np.sign(edge)),
            "selected_is_home":float(sign>0),"selected_market_probability":sel_ref,"selected_price_implied":_american_implied(odds),"market_prob_move_toward_selected":sign*(ref-opref) if math.isfinite(opref) else np.nan,
            "week_number":_num(f.get("Week_Number")),"is_division_game":_num(f.get("Is_Division_Game")),"rest_diff_selected":_orient(f.get("Rest_Differential_Days"),sign),"winpct_diff_selected":_orient(f.get("WinPct_Prior_Diff"),sign),
            "margin5_diff_selected":_orient(f.get("Avg_SU_Margin_Last5_Diff"),sign),"off_ypp_matchup_selected":_orient(f.get("Off_YPP_vs_Opp_Def_Last3_Diff"),sign),"def_ypp_matchup_selected":_orient(f.get("Def_YPP_vs_Opp_Off_Last3_Diff"),sign),"prior_season_winpct_selected":_orient(f.get("Prior_Season_WinPct_Diff"),sign),
            "system_net_support_selected":sign*_num(sys.get("spread_home_support",0)),"system_trigger_count":_num(sys.get("spread_trigger_count",0)),
        }
        return {**feat,"selected":home if sign>0 else away,"market_value":sel_ref,"model_value":fair if sign>0 else 1-fair,"raw_edge":abs(edge),"selected_price":odds,"profit_if_win":pwin}
    return None


def score_live(*, bq_client, storage_client, bucket_name, prediction_rows, now) -> list[dict]:
    engine=_load_engine(storage_client,bucket_name); meta=engine["metadata"]; quotes=_consensus_market(bq_client,now); systems=_live_system_context(bq_client,now)
    out=[]
    for p in prediction_rows:
        gs=pd.to_datetime(p.get("game_start"),utc=True,errors="coerce"); hk=_norm(p.get("home_team")); ak=_norm(p.get("away_team")); key=(gs.round("s").isoformat() if pd.notna(gs) else "",hk,ak)
        q=quotes.get(key,{}); sys=systems.get(key,{})
        for market in ("SPREADS","H2H","TOTALS"):
            row=_live_feature_row(p,market,q,sys)
            if row is None:
                out.append({"prediction_pair_id":p.get("prediction_pair_id"),"game_start":p.get("game_start"),"home_team":p.get("home_team"),"away_team":p.get("away_team"),"market":market,"action":"NO MARKET","system_labels":sys.get("labels",[])})
                continue
            df=pd.DataFrame([row]); prob=float(_predict(engine["models"][market],df)[0]); profit=_num(row.get("profit_if_win")); ev=prob*profit-(1-prob); be=1/(1+profit) if math.isfinite(profit) and profit>0 else np.nan
            authority=bool((meta.get("market_gates") or {}).get(market,{}).get("betting_decision_authority"))
            if ev>=EV_BET_MIN and authority: action="BET"
            elif ev>0: action="LEAN"
            else: action="PASS"
            out.append({
                "prediction_pair_id":p.get("prediction_pair_id"),"game_start":pd.to_datetime(p.get("game_start"),utc=True,errors="coerce").isoformat(),
                "home_team":p.get("home_team"),"away_team":p.get("away_team"),"market":market,"selected":row.get("selected"),"action":action,
                "bet_win_probability":prob,"break_even_probability":be,"expected_value":ev,"raw_model_edge":row.get("raw_edge"),
                "market_value":row.get("market_value"),"model_value":row.get("model_value"),"selected_price":row.get("selected_price"),
                "system_net_support":row.get("system_net_support_selected"),"system_trigger_count":row.get("system_trigger_count"),"system_labels":sys.get("labels",[]),
                "market_authority":(meta.get("market_gates") or {}).get(market,{}),"engine_artifact_sha256":meta.get("artifact_sha256"),
            })
    return out


def _list_json(storage_client,bucket_name,prefix):
    out=[]
    for b in storage_client.bucket(bucket_name).list_blobs(prefix=prefix.rstrip("/")+"/"):
        if not b.name.endswith(".json"):continue
        try:out.append(json.loads(b.download_as_text()))
        except Exception:pass
    return out


def _capture_bets(storage_client,bucket_name,rows,now,engine_sha):
    inserted=existing=0
    for r in rows:
        if r.get("action")!="BET":continue
        rid=_sha({"engine":engine_sha,"prediction_pair_id":r.get("prediction_pair_id"),"market":r.get("market")})
        event={**r,"bet_id":rid,"captured_at":pd.to_datetime(now,utc=True).isoformat(),"source_tag":SOURCE_TAG,"automatic_execution":False}
        if _write_immutable_json(storage_client,bucket_name,f"{EVENT_PREFIX}/{rid}.json",event):inserted+=1
        else:existing+=1
    return {"inserted":inserted,"existing":existing}


def _settlement_rows(client,pair_ids):
    if not pair_ids:return pd.DataFrame()
    q=f"SELECT * FROM `{PAIRED_SETTLEMENT_TABLE}` WHERE prediction_pair_id IN UNNEST(@ids)"
    cfg=bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("ids","STRING",list(pair_ids))])
    try:return client.query(q,job_config=cfg).to_dataframe(create_bqstorage_client=False)
    except Exception:return pd.DataFrame()


def _settle_bets(client,storage_client,bucket_name,now):
    events=_list_json(storage_client,bucket_name,EVENT_PREFIX); settled={str(x.get("bet_id")) for x in _list_json(storage_client,bucket_name,SETTLEMENT_PREFIX)}; pending=[e for e in events if str(e.get("bet_id")) not in settled]
    sd=_settlement_rows(client,{str(e.get("prediction_pair_id")) for e in pending if e.get("prediction_pair_id")}); smap={str(r.prediction_pair_id):r for _,r in sd.iterrows()} if not sd.empty else {}; inserted=0
    for e in pending:
        s=smap.get(str(e.get("prediction_pair_id")))
        if s is None:continue
        market=e.get("market"); sel=str(e.get("selected") or ""); result="UNRESOLVED"; profit=np.nan
        if market=="SPREADS":
            line=_num(e.get("market_value")); margin=_num(s.actual_margin); home_sel=_norm(sel)==_norm(e.get("home_team")); v=(margin+line) if home_sel else -(margin+line)
            result="PUSH" if math.isfinite(v) and abs(v)<1e-9 else "WIN" if math.isfinite(v) and v>0 else "LOSS" if math.isfinite(v) else "UNRESOLVED"
        elif market=="TOTALS":
            line=_num(e.get("market_value")); actual=_num(s.actual_total); v=actual-line; v=-v if sel.upper()=="UNDER" else v
            result="PUSH" if math.isfinite(v) and abs(v)<1e-9 else "WIN" if math.isfinite(v) and v>0 else "LOSS" if math.isfinite(v) else "UNRESOLVED"
        elif market=="H2H":
            y=_num(s.home_win_label); home_sel=_norm(sel)==_norm(e.get("home_team")); won=(y==1.0) if home_sel else (y==0.0); result="WIN" if won else "LOSS"
        if result=="WIN":profit=_american_profit_if_win(e.get("selected_price"))
        elif result=="LOSS":profit=-1.0
        elif result=="PUSH":profit=0.0
        payload={"bet_id":e.get("bet_id"),"prediction_pair_id":e.get("prediction_pair_id"),"market":market,"selected":sel,"result":result,"profit_per_unit":None if not math.isfinite(_num(profit)) else float(profit),"settled_at":pd.to_datetime(now,utc=True).isoformat(),"source_tag":SOURCE_TAG}
        if _write_immutable_json(storage_client,bucket_name,f"{SETTLEMENT_PREFIX}/{e.get('bet_id')}.json",payload):inserted+=1
    return {"events":len(events),"pending_before":len(pending),"inserted":inserted}


def _live_performance(storage_client,bucket_name):
    s=_list_json(storage_client,bucket_name,SETTLEMENT_PREFIX)
    def agg(rows):
        wins=sum(x.get("result")=="WIN" for x in rows); losses=sum(x.get("result")=="LOSS" for x in rows); pushes=sum(x.get("result")=="PUSH" for x in rows); n=wins+losses+pushes; dec=wins+losses; profits=[_num(x.get("profit_per_unit")) for x in rows if math.isfinite(_num(x.get("profit_per_unit")))]
        return {"n":n,"wins":wins,"losses":losses,"pushes":pushes,"hit_rate":round(wins/dec,6) if dec else None,"roi_per_unit":round(float(np.mean(profits)),6) if profits else None}
    out={"ALL":agg(s)}
    for m in ("SPREADS","H2H","TOTALS"):out[m]=agg([x for x in s if x.get("market")==m])
    return out


def prospective_model_performance(client,champion_sha):
    q=f"SELECT * FROM `{PAIRED_SETTLEMENT_TABLE}` WHERE champion_registry_sha256=@sha ORDER BY settled_at"; cfg=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("sha","STRING",champion_sha)])
    try:d=client.query(q,job_config=cfg).to_dataframe(create_bqstorage_client=False)
    except Exception:return {"settled_games":0}
    if d.empty:return {"settled_games":0}
    def mean(c):
        z=pd.to_numeric(d.get(c),errors="coerce").dropna(); return round(float(z.mean()),6) if len(z) else None
    return {"settled_games":int(len(d)),"champion_spread_mae":mean("champion_spread_abs_error"),"challenger_spread_mae":mean("challenger_spread_abs_error"),"champion_total_mae":mean("champion_total_abs_error"),"challenger_total_mae":mean("challenger_total_abs_error"),"champion_h2h_log_loss":mean("champion_h2h_log_loss"),"challenger_h2h_log_loss":mean("challenger_h2h_log_loss")}


def update_live_state(*,bq_client,storage_client,bucket_name,prediction_rows,champion_sha,now,log_func=print):
    engine=_load_engine(storage_client,bucket_name); meta=engine["metadata"]
    rows=score_live(bq_client=bq_client,storage_client=storage_client,bucket_name=bucket_name,prediction_rows=prediction_rows,now=now)
    capture=_capture_bets(storage_client,bucket_name,rows,now,meta.get("engine_registry_sha256")); settle=_settle_bets(bq_client,storage_client,bucket_name,now); live_perf=_live_performance(storage_client,bucket_name); model_perf=prospective_model_performance(bq_client,champion_sha)
    counts={k:sum(1 for x in rows if x.get("action")==k) for k in ("BET","LEAN","PASS","NO MARKET")}
    out={"status":"NFL_BETTING_ENGINE_V1_LIVE_ACTIVE","source_tag":SOURCE_TAG,"generated_at_utc":pd.to_datetime(now,utc=True).isoformat(),"engine_meta":meta,"live_rows":rows,"action_counts":counts,"capture":capture,"settlement":settle,"live_bet_performance":live_perf,"prospective_model_performance":model_perf,"automatic_execution":False}
    out["current_uri"]=_write_json(storage_client,bucket_name,CURRENT_OBJECT,out)
    log_func("[NFL-BET-ENGINE-V1-LIVE] "+json.dumps({"status":out["status"],"action_counts":counts,"capture":capture,"settlement":settle,"live_bet_performance":live_perf,"current_uri":out["current_uri"]},sort_keys=True,default=str))
    return out


def read_dashboard_state(*,storage_client,bucket_name="sharp-models"):
    return {"meta":_read_json(storage_client,bucket_name,META_OBJECT) or {},"current":_read_json(storage_client,bucket_name,CURRENT_OBJECT) or {}}


def _self_test():
    d=pd.DataFrame([{c:0.0 for c in SPREAD_FEATURES} for _ in range(120)]); d["target"]=[0,1]*60; d["season"]=[2021]*60+[2022]*60; d["profit_if_win"]=100/110
    b=_fit(d,"SPREADS"); p=_predict(b,d.iloc[:3]); assert len(p)==3 and np.all((p>0)&(p<1))
    assert abs((.55*(100/110)-.45)-.05)<.001
    return {"status":"PASS","source_tag":SOURCE_TAG,"feature_counts":{m:len(f) for m,f in FEATURES.items()},"fixed_ev_bet_min":EV_BET_MIN}


if __name__=="__main__":
    print(json.dumps(_self_test(),sort_keys=True))
