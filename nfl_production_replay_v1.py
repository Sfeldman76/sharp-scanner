"""NFL Production V2.6 — historical replay + frozen model betting-authority policy.

Purpose
-------
Recreate how the current NFL Production V1 operating architecture would have
behaved on older seasons WITHOUT scoring old games with a model trained on
future seasons.

For each validation season 2021-2025:
* FROZEN control: fit once on all completed games from seasons < validation year.
* ADAPTIVE challenger: refit before each NFL week using only games completed
  before the first game of that week.
* Score the same games with the exact Production V1 final 18-feature contract.
* Compare fair margin / H2H probability / fair total to outcomes and to the
  historical closing market references already present in the historical view.

The historical closing lines are retrospective references. Spread/Totals ROI is
therefore a standard -110 proxy, and H2H ROI uses the stored historical close
moneyline when available. Neither is represented as verified executable pricing.

This module is read-only with respect to BigQuery and never changes champion,
challenger, or model-promotion authority. After generating leak-safe replay rows,
it preserves Betting Engine V1 and Edge Authority V2.5 as research/attribution benchmarks, then freezes a separate V2.6 betting policy whose only authority source is the frozen production model. CORE/STAT/SYSTEM/MARKET diagnostics cannot create or reverse a wager. It may also publish replay artifacts to GCS.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

import nfl_production_v1 as prod
import nfl_betting_engine_v1 as betting
import nfl_edge_authority_v2 as edge_v2
import nfl_model_authority_v26 as model_auth
from nfl_feature_audit_v1 import VIEW

SOURCE_TAG = "nfl-production-v2.6-historical-model-authority-freeze-20261003"
VALIDATION_SEASONS = (2021, 2022, 2023, 2024, 2025)
VALIDATION_STAGE = "REGULAR"
MARKET_COLUMNS = (
    "Spread_Value", "Current_Total", "ML_Odds",
    "Opening_Spread", "Opening_Total", "Opening_ML_Odds",
)
BASE_COLUMNS = tuple(OrderedDict.fromkeys((*prod.QUERY_COLUMNS, *MARKET_COLUMNS)))
EDGE_THRESHOLDS_POINTS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
H2H_EDGE_THRESHOLDS = (0.0, 0.02, 0.05, 0.075, 0.10, 0.15)
ARTIFACT_PREFIX = "production/nfl/v1/historical_replay"
CURRENT_REPORT_OBJECT = f"{ARTIFACT_PREFIX}/current_report.json"


def _f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else math.nan
    except Exception:
        return math.nan


def _sha_obj(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _query() -> str:
    cols = ", ".join(f"`{c}`" for c in BASE_COLUMNS)
    return (
        f"SELECT {cols} FROM `{VIEW}` "
        "WHERE Season BETWEEN 2017 AND 2025 "
        "AND Season_Stage IN ('REGULAR','POSTSEASON') "
        "AND Historical_Core_Eligible = 1 "
        "AND Team_Score IS NOT NULL AND Opponent_Score IS NOT NULL "
        "ORDER BY Season, Game_Date, Source_Name, Source_Game_ID, Team_Norm"
    )


def _validate_schema(client) -> None:
    cols = {f.name for f in client.get_table(VIEW).schema}
    missing = sorted(set(BASE_COLUMNS) - cols)
    if missing:
        raise RuntimeError("[NFL-PROD-V1-REPLAY-HOLD] SOURCE_COLUMNS_MISSING " + str(missing))


def _augment_opposing_market(side: pd.DataFrame) -> pd.DataFrame:
    d = side.copy()
    d["Opp_ML_Odds"] = np.nan
    d["Opp_Opening_ML_Odds"] = np.nan
    anomalies = []
    for ident, idx in d.groupby(list(prod.IDENTITY), sort=False, dropna=False).groups.items():
        ii = list(idx)
        if len(ii) != 2:
            continue  # prod._physical_games will fail the game grain itself.
        a, b = ii
        d.at[a, "Opp_ML_Odds"] = d.at[b, "ML_Odds"]
        d.at[b, "Opp_ML_Odds"] = d.at[a, "ML_Odds"]
        d.at[a, "Opp_Opening_ML_Odds"] = d.at[b, "Opening_ML_Odds"]
        d.at[b, "Opp_Opening_ML_Odds"] = d.at[a, "Opening_ML_Odds"]
        sa, sb = _f(d.at[a, "Spread_Value"]), _f(d.at[b, "Spread_Value"])
        if math.isfinite(sa) and math.isfinite(sb) and abs(sa + sb) > 1e-5:
            anomalies.append({"game": "|".join(map(str, ident)), "reason": "NONRECIPROCAL_CLOSE_SPREAD", "a": sa, "b": sb})
        ta, tb = _f(d.at[a, "Current_Total"]), _f(d.at[b, "Current_Total"])
        if math.isfinite(ta) and math.isfinite(tb) and abs(ta - tb) > 1e-5:
            anomalies.append({"game": "|".join(map(str, ident)), "reason": "CONFLICTING_CLOSE_TOTAL", "a": ta, "b": tb})
    if anomalies:
        raise RuntimeError("[NFL-PROD-V1-REPLAY-HOLD] MARKET_GRAIN_ANOMALIES " + json.dumps(anomalies[:12], sort_keys=True))
    return d


def _load_games(client) -> pd.DataFrame:
    _validate_schema(client)
    side = client.query(_query()).to_dataframe(create_bqstorage_client=False)
    side = _augment_opposing_market(side)
    games = prod._physical_games(side)
    games["Game_Date"] = pd.to_datetime(games["Game_Date"], errors="coerce")
    if games["Game_Date"].isna().any():
        raise RuntimeError("[NFL-PROD-V1-REPLAY-HOLD] INVALID_GAME_DATE")
    games["Week_Number"] = pd.to_numeric(games["Week_Number"], errors="coerce")
    if games.loc[games.Season_Stage.eq("REGULAR"), "Week_Number"].isna().any():
        raise RuntimeError("[NFL-PROD-V1-REPLAY-HOLD] REGULAR_WEEK_NUMBER_MISSING")
    games["Week_Number"] = games["Week_Number"].astype("Int64")
    if int(games.Season.min()) != 2017 or int(games.Season.max()) != 2025:
        raise RuntimeError("[NFL-PROD-V1-REPLAY-HOLD] HISTORY_WINDOW_NOT_2017_2025")
    return games


def _amer_implied(odds) -> float:
    o = _f(odds)
    if not math.isfinite(o) or o == 0:
        return math.nan
    return 100.0 / (100.0 + o) if o > 0 else (-o) / ((-o) + 100.0)


def _novig_probability(home_odds, away_odds) -> float:
    a, b = _amer_implied(home_odds), _amer_implied(away_odds)
    den = a + b
    return a / den if math.isfinite(a) and math.isfinite(b) and den > 0 else math.nan


def _american_profit(odds, won: bool) -> float:
    if not won:
        return -1.0
    o = _f(odds)
    if not math.isfinite(o) or o == 0:
        return math.nan
    return o / 100.0 if o > 0 else 100.0 / abs(o)


def _make_replay_rows(games: pd.DataFrame, log_func=print) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    week_meta = []
    regular = games.loc[games.Season_Stage.eq(VALIDATION_STAGE)].copy()

    for season in VALIDATION_SEASONS:
        season_games = regular.loc[regular.Season.eq(season)].copy()
        weeks = sorted(int(x) for x in season_games.Week_Number.dropna().unique())
        frozen_train = games.loc[games.Season.lt(season)].copy()
        if len(frozen_train) < 1000:
            raise RuntimeError(f"[NFL-PROD-V1-REPLAY-HOLD] FROZEN_TRAIN_TOO_SMALL season={season} n={len(frozen_train)}")
        frozen_bundle = prod.fit_bundle(
            frozen_train, role="HISTORICAL_FROZEN_CONTROL", cutoff_label=f"BEFORE_{season}_SEASON"
        )
        season_rows = 0
        for week in weeks:
            val = season_games.loc[season_games.Week_Number.eq(week)].copy()
            if val.empty:
                continue
            cutoff = val.Game_Date.min().normalize()
            adaptive_train = games.loc[games.Game_Date.lt(cutoff)].copy()
            # Hard leakage contract: current validation games and all later dates are absent.
            if adaptive_train.empty or adaptive_train.Game_Date.max() >= cutoff:
                raise RuntimeError(f"[NFL-PROD-V1-REPLAY-HOLD] WEEKLY_ASOF_LEAK season={season} week={week}")
            challenger_bundle = prod.fit_bundle(
                adaptive_train, role="HISTORICAL_WEEKLY_CHALLENGER", cutoff_label=f"{season}_W{week}_ASOF_{cutoff.date()}"
            )
            fs = prod.score_feature_rows(frozen_bundle, val)
            ad = prod.score_feature_rows(challenger_bundle, val)
            if len(fs) != len(val) or len(ad) != len(val):
                raise RuntimeError("[NFL-PROD-V1-REPLAY-HOLD] SCORE_ROW_COUNT_MISMATCH")
            for i in range(len(val)):
                r = val.iloc[i]
                rec = {
                    "physical_game_id": r.physical_game_id,
                    "season": int(season),
                    "week_number": int(week),
                    "week_cutoff": cutoff,
                    "game_date": r.Game_Date,
                    "home_team": str(r.Team_Norm),
                    "away_team": str(r.Opponent_Norm),
                    "actual_margin": _f(r.actual_margin),
                    "actual_total": _f(r.actual_total),
                    "home_win_label": _f(r.H2H_label),
                    "close_spread": _f(r.Spread_Value),
                    "close_total": _f(r.Current_Total),
                    "home_close_ml": _f(r.ML_Odds),
                    "away_close_ml": _f(r.Opp_ML_Odds),
                    "close_novig_home_probability": _novig_probability(r.ML_Odds, r.Opp_ML_Odds),
                    "opening_spread": _f(r.Opening_Spread),
                    "opening_total": _f(r.Opening_Total),
                    "home_open_ml": _f(r.Opening_ML_Odds),
                    "away_open_ml": _f(r.Opp_Opening_ML_Odds),
                    "opening_novig_home_probability": _novig_probability(r.Opening_ML_Odds, r.Opp_Opening_ML_Odds),
                    "frozen_fair_margin": _f(fs.iloc[i].fair_margin),
                    "adaptive_fair_margin": _f(ad.iloc[i].fair_margin),
                    "frozen_home_win_probability": _f(fs.iloc[i].win_probability),
                    "adaptive_home_win_probability": _f(ad.iloc[i].win_probability),
                    "frozen_fair_total": _f(fs.iloc[i].fair_total),
                    "adaptive_fair_total": _f(ad.iloc[i].fair_total),
                    "frozen_training_games": int(len(frozen_train)),
                    "adaptive_training_games": int(len(adaptive_train)),
                    "adaptive_training_max_date": adaptive_train.Game_Date.max(),
                }
                rows.append(rec)
            season_rows += len(val)
            week_meta.append({
                "season": int(season), "week_number": int(week), "cutoff": str(cutoff.date()),
                "validation_games": int(len(val)), "frozen_training_games": int(len(frozen_train)),
                "adaptive_training_games": int(len(adaptive_train)),
                "adaptive_training_max_date": str(adaptive_train.Game_Date.max().date()),
            })
        log_func("[NFL-PROD-V1-REPLAY-SEASON] " + json.dumps({
            "season": season, "weeks": len(weeks), "prediction_games": season_rows,
            "frozen_training_games": int(len(frozen_train)),
        }, sort_keys=True))
    out = pd.DataFrame(rows)
    if out.empty or out.physical_game_id.duplicated().any():
        raise RuntimeError("[NFL-PROD-V1-REPLAY-HOLD] INVALID_REPLAY_GRAIN")
    return out, week_meta


def _cont(y, p) -> dict:
    return prod._continuous(np.asarray(y, float), np.asarray(p, float))


def _cls(y, p) -> dict:
    return prod._classification(np.asarray(y, float), np.asarray(p, float))


def _by_season(rows: pd.DataFrame, role: str) -> dict:
    out = {}
    for s, g in rows.groupby("season", sort=True):
        out[str(int(s))] = {
            "SPREADS": _cont(g.actual_margin, g[f"{role}_fair_margin"]),
            "H2H": _cls(g.home_win_label, g[f"{role}_home_win_probability"]),
            "TOTALS": _cont(g.actual_total, g[f"{role}_fair_total"]),
        }
    return out


def _predictive_summary(rows: pd.DataFrame, role: str) -> dict:
    return {
        "SPREADS": _cont(rows.actual_margin, rows[f"{role}_fair_margin"]),
        "H2H": _cls(rows.home_win_label, rows[f"{role}_home_win_probability"]),
        "TOTALS": _cont(rows.actual_total, rows[f"{role}_fair_total"]),
        "by_season": _by_season(rows, role),
    }


def _minus110_table(edge, settle, thresholds=EDGE_THRESHOLDS_POINTS) -> dict:
    e = np.asarray(edge, float); s = np.asarray(settle, float)
    out = {}
    for th in thresholds:
        m = np.isfinite(e) & np.isfinite(s) & (np.abs(e) >= float(th)) & (np.abs(e) > 1e-12)
        push = m & np.isclose(s, 0.0, atol=1e-9)
        dec = m & ~push
        win = dec & (((e > 0) & (s > 0)) | ((e < 0) & (s < 0)))
        loss = dec & ~win
        n = int(dec.sum()); w = int(win.sum()); l = int(loss.sum()); p = int(push.sum())
        roi = ((w * (100.0 / 110.0)) - l) / n if n else math.nan
        out[str(th)] = {
            "n": n, "wins": w, "losses": l, "pushes": p,
            "hit_rate": round(w / n, 6) if n else None,
            "roi_minus110_proxy": round(float(roi), 6) if math.isfinite(roi) else None,
        }
    return out


def _h2h_edge_table(rows: pd.DataFrame, role: str, reference: str) -> dict:
    if reference == "close":
        ref_col, hm_col, am_col = "close_novig_home_probability", "home_close_ml", "away_close_ml"
    elif reference == "open":
        ref_col, hm_col, am_col = "opening_novig_home_probability", "home_open_ml", "away_open_ml"
    else:
        raise ValueError(reference)
    p = pd.to_numeric(rows[f"{role}_home_win_probability"], errors="coerce").to_numpy(float)
    ref = pd.to_numeric(rows[ref_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(rows.home_win_label, errors="coerce").to_numpy(float)
    hm = pd.to_numeric(rows[hm_col], errors="coerce").to_numpy(float)
    am = pd.to_numeric(rows[am_col], errors="coerce").to_numpy(float)
    edge = p - ref
    out = {}
    for th in H2H_EDGE_THRESHOLDS:
        base = np.isfinite(edge) & np.isfinite(y) & (np.abs(edge) >= float(th)) & (np.abs(edge) > 1e-12)
        wins = 0; losses = 0; profits = []
        for i in np.where(base)[0]:
            home_sel = edge[i] > 0
            won = bool((y[i] == 1.0) if home_sel else (y[i] == 0.0))
            odds = hm[i] if home_sel else am[i]
            prof = _american_profit(odds, won)
            if won: wins += 1
            else: losses += 1
            if math.isfinite(prof): profits.append(prof)
        n = wins + losses
        out[str(th)] = {
            "n": n, "wins": wins, "losses": losses,
            "hit_rate": round(wins / n, 6) if n else None,
            "priced_n": len(profits),
            "roi_historical_moneyline": round(float(np.mean(profits)), 6) if profits else None,
        }
    return out


def _bucket_labels(values, bounds, labels):
    v = pd.to_numeric(pd.Series(values), errors="coerce")
    return pd.cut(v.abs(), bins=bounds, labels=labels, right=False, include_lowest=True)


def _edge_bucket_summary(rows: pd.DataFrame, role: str, reference: str) -> dict:
    res = {}
    if reference == "close":
        spread_line_col, total_line_col = "close_spread", "close_total"
        ref_prob_col, hm_col, am_col = "close_novig_home_probability", "home_close_ml", "away_close_ml"
    elif reference == "open":
        spread_line_col, total_line_col = "opening_spread", "opening_total"
        ref_prob_col, hm_col, am_col = "opening_novig_home_probability", "home_open_ml", "away_open_ml"
    else:
        raise ValueError(reference)
    # Spread / Totals bucket summaries use the same direction-settlement logic.
    for market, pred_col, line_col, actual_col in (
        ("SPREADS", f"{role}_fair_margin", spread_line_col, "actual_margin"),
        ("TOTALS", f"{role}_fair_total", total_line_col, "actual_total"),
    ):
        pred = pd.to_numeric(rows[pred_col], errors="coerce")
        line = pd.to_numeric(rows[line_col], errors="coerce")
        actual = pd.to_numeric(rows[actual_col], errors="coerce")
        edge = pred + line if market == "SPREADS" else pred - line
        settle = actual + line if market == "SPREADS" else actual - line
        labels = ["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6+"]
        b = _bucket_labels(edge, [0,1,2,3,4,5,6,np.inf], labels)
        z = pd.DataFrame({"edge": edge, "settle": settle, "bucket": b})
        bout = {}
        for lab, g in z.groupby("bucket", observed=True):
            e = g.edge.to_numpy(float); s = g.settle.to_numpy(float)
            push = np.isclose(s,0.0,atol=1e-9); dec = ~push & np.isfinite(e) & np.isfinite(s) & (np.abs(e)>1e-12)
            win = dec & (((e>0)&(s>0))|((e<0)&(s<0)))
            n=int(dec.sum()); w=int(win.sum()); l=n-w
            bout[str(lab)]={"n":n,"wins":w,"losses":l,"pushes":int(push.sum()),"hit_rate":round(w/n,6) if n else None,
                            "roi_minus110_proxy":round(((w*(100/110))-l)/n,6) if n else None}
        res[market]=bout
    # H2H probability-edge buckets.
    p = pd.to_numeric(rows[f"{role}_home_win_probability"], errors="coerce")
    ref = pd.to_numeric(rows[ref_prob_col], errors="coerce")
    edge = p-ref
    labels=["0-2%","2-5%","5-7.5%","7.5-10%","10-15%","15%+"]
    b=_bucket_labels(edge,[0,.02,.05,.075,.10,.15,np.inf],labels)
    z=rows.copy(); z["__edge"]=edge; z["__bucket"]=b
    bout={}
    for lab,g in z.groupby("__bucket",observed=True):
        e=pd.to_numeric(g.__edge,errors="coerce").to_numpy(float); y=pd.to_numeric(g.home_win_label,errors="coerce").to_numpy(float)
        hm=pd.to_numeric(g[hm_col],errors="coerce").to_numpy(float); am=pd.to_numeric(g[am_col],errors="coerce").to_numpy(float)
        wins=0; losses=0; profits=[]
        for i in range(len(g)):
            if not math.isfinite(e[i]) or not math.isfinite(y[i]) or abs(e[i])<=1e-12: continue
            hs=e[i]>0; won=bool(y[i]==1 if hs else y[i]==0); odds=hm[i] if hs else am[i]
            wins += int(won); losses += int(not won)
            pr=_american_profit(odds,won)
            if math.isfinite(pr): profits.append(pr)
        n=wins+losses
        bout[str(lab)]={"n":n,"wins":wins,"losses":losses,"hit_rate":round(wins/n,6) if n else None,
                        "priced_n":len(profits),"roi_historical_moneyline":round(float(np.mean(profits)),6) if profits else None}
    res["H2H"]=bout
    return res


def _market_edge_summary(rows: pd.DataFrame, role: str, reference: str) -> dict:
    if reference == "close":
        sp_col, to_col = "close_spread", "close_total"
    elif reference == "open":
        sp_col, to_col = "opening_spread", "opening_total"
    else:
        raise ValueError(reference)
    sp_line = pd.to_numeric(rows[sp_col], errors="coerce")
    to_line = pd.to_numeric(rows[to_col], errors="coerce")
    sp_edge = pd.to_numeric(rows[f"{role}_fair_margin"], errors="coerce") + sp_line
    sp_settle = pd.to_numeric(rows.actual_margin, errors="coerce") + sp_line
    to_edge = pd.to_numeric(rows[f"{role}_fair_total"], errors="coerce") - to_line
    to_settle = pd.to_numeric(rows.actual_total, errors="coerce") - to_line
    return {
        "SPREADS": _minus110_table(sp_edge, sp_settle),
        "H2H": _h2h_edge_table(rows, role, reference),
        "TOTALS": _minus110_table(to_edge, to_settle),
        "edge_buckets": _edge_bucket_summary(rows, role, reference),
        "pricing_note": (
            "Spread/Totals ROI is a standard -110 proxy on the historical %s reference; "
            "H2H ROI uses the stored historical %s moneyline. Historical source pricing is retrospective "
            "and not independently verified executable pricing." % (reference, reference)
        ),
    }



def _compact_edge_log(edge: dict, market: str) -> dict:
    out={}
    for role in ("FROZEN","ADAPTIVE"):
        out[role]={}
        for ref in ("OPEN","CLOSE"):
            block=edge[role][ref]
            out[role][ref]={
                "thresholds":block.get(market,{}),
                "buckets":block.get("edge_buckets",{}).get(market,{}),
            }
    return out


def _write_current(storage_client,bucket_name,name,payload):
    storage_client.bucket(bucket_name).blob(name).upload_from_string(
        json.dumps(payload,sort_keys=True,indent=2,default=str),content_type="application/json"
    )
    return f"gs://{bucket_name}/{name}"

def _adaptive_vs_frozen(rows: pd.DataFrame) -> dict:
    frozen = _predictive_summary(rows, "frozen")
    adaptive = _predictive_summary(rows, "adaptive")
    out = {
        "SPREADS": {
            "frozen_mae": frozen["SPREADS"]["mae"], "adaptive_mae": adaptive["SPREADS"]["mae"],
            "delta_adaptive_minus_frozen": round(adaptive["SPREADS"]["mae"]-frozen["SPREADS"]["mae"],6),
        },
        "H2H": {
            "frozen_log_loss": frozen["H2H"]["log_loss"], "adaptive_log_loss": adaptive["H2H"]["log_loss"],
            "delta_adaptive_minus_frozen": round(adaptive["H2H"]["log_loss"]-frozen["H2H"]["log_loss"],6),
            "frozen_auc": frozen["H2H"]["auc"], "adaptive_auc": adaptive["H2H"]["auc"],
        },
        "TOTALS": {
            "frozen_mae": frozen["TOTALS"]["mae"], "adaptive_mae": adaptive["TOTALS"]["mae"],
            "delta_adaptive_minus_frozen": round(adaptive["TOTALS"]["mae"]-frozen["TOTALS"]["mae"],6),
        },
    }
    for market in ("SPREADS","H2H","TOTALS"):
        wins=0
        for s in map(str,VALIDATION_SEASONS):
            f=frozen["by_season"][s][market]; a=adaptive["by_season"][s][market]
            metric="log_loss" if market=="H2H" else "mae"
            if a.get(metric) is not None and f.get(metric) is not None and float(a[metric])<float(f[metric]): wins+=1
        out[market]["adaptive_seasons_better"] = wins
        out[market]["seasons_compared"] = len(VALIDATION_SEASONS)
    return out


def _publish(storage_client, bucket_name: str, report: dict, rows: pd.DataFrame) -> dict:
    stable_report = {k:v for k,v in report.items() if k not in ("generated_at_utc","artifacts","betting_engine","betting_engine_v1_benchmark","edge_authority_v2","edge_authority_v2_shadow","model_authority_v26")}
    digest = _sha_obj({"source_tag":SOURCE_TAG,"contract":prod.production_contract()["contract_sha256"],"report":stable_report})
    prefix = f"{ARTIFACT_PREFIX}/{digest[:16]}"
    report_name = f"{prefix}/report.json"
    rows_name = f"{prefix}/replay_rows.csv.gz"
    rb = json.dumps(report, sort_keys=True, indent=2, default=str).encode()
    csv = rows.to_csv(index=False).encode()
    gz = gzip.compress(csv, compresslevel=6)
    rr = prod._upload_immutable(storage_client,bucket_name,report_name,rb,"application/json")
    dr = prod._upload_immutable(storage_client,bucket_name,rows_name,gz,"application/gzip")
    return {"replay_sha256":digest,"report_uri":rr["uri"],"rows_uri":dr["uri"],"report_created":rr.get("created"),"rows_created":dr.get("created")}


def run_nfl_production_historical_replay(*, bq_client, storage_client=None, bucket_name="sharp-models", log_func=print, now=None, research_report=None, system_report=None) -> dict:
    now = now or datetime.now(timezone.utc)
    contract = prod.production_contract()
    log_func("[NFL-PROD-V1-REPLAY-PREFLIGHT] " + json.dumps({
        "status":"START", "source_tag":SOURCE_TAG, "production_source_tag":prod.SOURCE_TAG,
        "production_contract_sha256":contract["contract_sha256"], "production_feature_count":len(prod.PRODUCTION_FEATURES),
        "validation_seasons":list(VALIDATION_SEASONS), "validation_stage":VALIDATION_STAGE,
        "cadence":"FROZEN_SEASON_CONTROL_VS_WEEKLY_ADAPTIVE_CHALLENGER",
        "production_authority":"FROZEN_MODEL_POLICY_PENDING", "betting_decision_authority":False, "automatic_promotion":False,
    }, sort_keys=True))
    games = _load_games(bq_client)
    replay, weeks = _make_replay_rows(games, log_func=log_func)
    log_func("[NFL-PROD-V1-REPLAY-GRAIN] " + json.dumps({
        "historical_physical_games":int(len(games)), "replay_games":int(len(replay)), "weeks":int(len(weeks)),
        "games_by_season":{str(int(k)):int(v) for k,v in replay.groupby("season").size().items()},
        "duplicate_replay_games":int(replay.physical_game_id.duplicated().sum()),
    }, sort_keys=True))

    frozen_pred = _predictive_summary(replay, "frozen")
    adaptive_pred = _predictive_summary(replay, "adaptive")
    compare = _adaptive_vs_frozen(replay)
    frozen_edge = {"OPEN": _market_edge_summary(replay, "frozen", "open"), "CLOSE": _market_edge_summary(replay, "frozen", "close")}
    adaptive_edge = {"OPEN": _market_edge_summary(replay, "adaptive", "open"), "CLOSE": _market_edge_summary(replay, "adaptive", "close")}

    edge_all={"FROZEN":frozen_edge,"ADAPTIVE":adaptive_edge}
    log_func("[NFL-PROD-V1-REPLAY-PREDICTIVE] " + json.dumps({"FROZEN":frozen_pred,"ADAPTIVE":adaptive_pred,"adaptive_vs_frozen":compare}, sort_keys=True, default=str))
    for _m in ("SPREADS","H2H","TOTALS"):
        log_func(f"[NFL-PROD-V1-REPLAY-EDGE-{_m}] " + json.dumps(_compact_edge_log(edge_all,_m), sort_keys=True, default=str))

    betting_meta = {}
    edge_meta = {}
    model_auth_meta = {}
    if storage_client is not None:
        # V1 Betting Engine and V2.5 Edge Authority remain benchmark/evidence layers.
        # V2.6 freezes betting authority from the production model itself.
        betting_meta = betting.train_publish_engine(
            bq_client=bq_client, storage_client=storage_client, bucket_name=bucket_name,
            replay_rows=replay, games=games, log_func=log_func,
        )
        edge_meta = edge_v2.train_publish_edge_authority(
            bq_client=bq_client, storage_client=storage_client, bucket_name=bucket_name,
            replay_rows=replay, games=games, research_report=research_report,
            system_report=system_report, log_func=log_func,
        )
        model_auth_meta = model_auth.train_publish_model_authority(
            replay_rows=replay, storage_client=storage_client, bucket_name=bucket_name, log_func=log_func,
        )

    report = {
        "status":"NFL_PRODUCTION_V1_HISTORICAL_WEEKLY_REPLAY_PASS",
        "source_tag":SOURCE_TAG, "generated_at_utc":now.isoformat(),
        "production_contract_sha256":contract["contract_sha256"], "production_feature_count":len(prod.PRODUCTION_FEATURES),
        "validation_seasons":list(VALIDATION_SEASONS), "validation_stage":VALIDATION_STAGE,
        "weeks":int(len(weeks)), "prediction_games":int(len(replay)),
        "frozen": {"predictive":frozen_pred,"market_edge":frozen_edge},
        "adaptive": {"predictive":adaptive_pred,"market_edge":adaptive_edge},
        "adaptive_vs_frozen":compare,
        "week_cutoffs":weeks,
        "leakage_contract":"Each validation week is scored before its first game; adaptive training max Game_Date is strictly earlier than that cutoff. Frozen control is trained only on seasons before the validation season.",
        "market_reference_policy":"Historical opening and closing lines/prices are source-provided retrospective references only; no independently verified executable-price or historical CLV claim.",
        "production_authority":"FROZEN_PRODUCTION_MODEL_ONLY",
        "betting_decision_authority":bool(any(bool((x or {}).get("production_authority")) for x in (model_auth_meta.get("markets") or {}).values())),
        "automatic_promotion":False,
        "live_paired_ledger":"SEPARATE_AND_CONTINUES_UNCHANGED",
        "systems":"SHADOW_EVIDENCE_ONLY; SYSTEMS_CANNOT_CREATE_REVERSE_OR_ESCALATE_A_BET",
        "betting_engine_v1_benchmark":betting_meta,
        "edge_authority_v2_shadow":edge_meta,
        "model_authority_v26":model_auth_meta,
        "research_diagnostics": {
            "research_status": (research_report or {}).get("status") if isinstance(research_report,dict) else None,
            "system_status": (system_report or {}).get("status") if isinstance(system_report,dict) else None,
            "system_family_registry_sha256": (((system_report or {}).get("registry") or {}).get("family_registry_sha256") if isinstance(system_report,dict) else None),
        },
    }
    pub = _publish(storage_client,bucket_name,report,replay) if storage_client is not None else {}
    report["artifacts"] = pub
    if storage_client is not None:
        report["current_report_uri"]=_write_current(storage_client,bucket_name,CURRENT_REPORT_OBJECT,report)
    log_func("[NFL-PROD-V1-REPLAY-CONTRACT] " + json.dumps({
        "status":report["status"], "source_tag":SOURCE_TAG, "weeks":report["weeks"], "prediction_games":report["prediction_games"],
        "adaptive_vs_frozen":compare, "artifacts":pub,
        "betting_engine_v1_benchmark_status":betting_meta.get("status"),
        "edge_authority_v2_shadow_status":edge_meta.get("status"),
        "multidimensional_research_contract_sha256":edge_meta.get("multidimensional_research_contract_sha256"),
        "model_authority_v26_status":model_auth_meta.get("status"),
        "model_authority_v26_contract_sha256":model_auth_meta.get("contract_sha256"),
        "model_authority_v26_markets":model_auth_meta.get("markets",{}),
        "research_diagnostics":report.get("research_diagnostics",{}),
        "current_report_uri":report.get("current_report_uri"),
        "production_authority":"FROZEN_PRODUCTION_MODEL_ONLY", "automatic_promotion":False,
        "next_step":"RUN NFL PRODUCTION — WEEKLY UPDATE; FINAL UI USES MODEL -> ACTION AND SHADOW LANES FOR ATTRIBUTION ONLY",
    }, sort_keys=True, default=str))
    return report


def _self_test():
    assert len(prod.PRODUCTION_FEATURES)==18
    assert set(VALIDATION_SEASONS)==set(range(2021,2026))
        # Edge direction sanity: favorite/home fair margin better than market and Over fair total.
    t=_minus110_table(np.array([3.,-3.]),np.array([1.,-1.]),thresholds=(0,))
    assert t["0"]["wins"]==2 and t["0"]["losses"]==0
    assert _american_profit(-200,True)==0.5 and _american_profit(150,True)==1.5 and _american_profit(150,False)==-1.0
    return {"status":"PASS","source_tag":SOURCE_TAG,"production_features":len(prod.PRODUCTION_FEATURES)}


if __name__ == "__main__":
    print(json.dumps(_self_test(), sort_keys=True))
