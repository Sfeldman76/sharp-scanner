"""REFIT_CADENCE_TEST_V1

Research-only controlled cadence experiment for the NCAAF statistical backbones.

Only refit timing changes. Feature definitions, model heads, hyperparameters, and
market targets are fixed across all arms.

Cadence arms:
- FROZEN: fit on prior seasons once; no current-season refit.
- WEEKLY_7D: controlled weekly benchmark (not the retired adaptive weekly model).
- QUARTER_28D: refit every 28 days.
- SIX_WEEK_42D: refit every 42 days.
- MIDSEASON_1X: one refit at the chronological midpoint of that season.

Every within-season refit may use only completed games strictly before the block
cutoff. 2023-2025 are comparison/discovery seasons. 2026 is confirmation only.
No production authority is granted by this module.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

REFIT_CADENCE_TEST_V1_SOURCE_TAG = "refit-cadence-test-v1-fixed-backbones-asof"
DISCOVERY_SEASONS = (2023, 2024, 2025)
CONFIRM_SEASON = 2026
BREAK_EVEN_110 = 110.0 / 210.0

# Exact currently admitted STAT feature snapshot. Context_Intercept is the
# synthetic intercept already present in the production game frame.
SPREAD_FEATURES = (
    "Context_Intercept",
    "Diff_RawRecent3_Off_YPP",
    "B_RawSeason_GameAdj_Def_Rush_YPA",
)
TOTAL_FEATURES = (
    "Context_Intercept",
    "A_RawRecent3_Def_Rush_YPA_Allowed",
)
H2H_FEATURES = (
    "H2H2_Prior_Meetings",
    "H2H2_Prior_Margin_Current_Orientation",
    "H2H2_Prior_Total",
    "H2H2_Days_Since",
    "H2H2_Same_Home_Role",
    "H2H2_Recency_Margin",
    "H2H2_Recency_Total",
)

CADENCES = {
    "FROZEN": None,
    "WEEKLY_7D": 7,
    "QUARTER_28D": 28,
    "SIX_WEEK_42D": 42,
    "MIDSEASON_1X": "MID",
}


def _num(v, index=None):
    if isinstance(v, pd.Series):
        return pd.to_numeric(v, errors="coerce")
    if v is None:
        return pd.Series(np.nan, index=index)
    return pd.to_numeric(pd.Series(v, index=index), errors="coerce")


def _roi_110(hit):
    if not np.isfinite(hit):
        return np.nan
    return float(hit * (100.0 / 110.0) - (1.0 - hit))


def _auc(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    if int(ok.sum()) < 20 or np.unique(y[ok]).size < 2:
        return np.nan
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y[ok].astype(int), p[ok]))
    except Exception:
        return np.nan


def _logloss(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any():
        return np.nan
    yy = y[ok]; pp = np.clip(p[ok], 1e-6, 1 - 1e-6)
    return float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)))


def _brier(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.mean((p[ok] - y[ok]) ** 2)) if ok.any() else np.nan


def _reg_metrics(actual, pred, market, market_type):
    a = np.asarray(actual, float); p = np.asarray(pred, float); m = np.asarray(market, float)
    ok = np.isfinite(a) & np.isfinite(p)
    err = a[ok] - p[ok]
    out = {
        "n": int(ok.sum()),
        "rmse": float(np.sqrt(np.mean(err ** 2))) if ok.any() else np.nan,
        "mae": float(np.mean(np.abs(err))) if ok.any() else np.nan,
    }
    mk = ok & np.isfinite(m)
    if mk.any():
        me = a[mk] - m[mk]
        out["market_rmse"] = float(np.sqrt(np.mean(me ** 2)))
        out["market_mae"] = float(np.mean(np.abs(me)))
    else:
        out["market_rmse"] = np.nan; out["market_mae"] = np.nan

    edge = p - m
    realized = a - m
    side_ok = np.isfinite(edge) & np.isfinite(realized) & ~np.isclose(edge, 0.0, atol=1e-9) & ~np.isclose(realized, 0.0, atol=1e-9)
    correct = np.sign(edge[side_ok]) == np.sign(realized[side_ok])
    hit = float(np.mean(correct)) if correct.size else np.nan
    out.update({"direction_n": int(correct.size), "hit": hit, "roi": _roi_110(hit)})
    for th in (1.0, 2.0, 3.0, 4.0):
        z = side_ok & (np.abs(edge) >= th)
        c = np.sign(edge[z]) == np.sign(realized[z])
        h = float(np.mean(c)) if c.size else np.nan
        out[f"edge{int(th)}_n"] = int(c.size)
        out[f"edge{int(th)}_hit"] = h
        out[f"edge{int(th)}_roi"] = _roi_110(h)
    return out


def _h2h_metrics(actual_margin, pred, market_prob):
    a = np.asarray(actual_margin, float); p = np.asarray(pred, float); mp = np.asarray(market_prob, float)
    y = np.where(np.isfinite(a), (a > 0).astype(float), np.nan)
    ok = np.isfinite(y) & np.isfinite(p)
    mok = np.isfinite(y) & np.isfinite(mp)
    return {
        "n": int(ok.sum()),
        "auc": _auc(y, p),
        "logloss": _logloss(y, p),
        "brier": _brier(y, p),
        "market_n": int(mok.sum()),
        "market_auc": _auc(y, mp),
        "market_logloss": _logloss(y, mp),
        "market_brier": _brier(y, mp),
    }


def _season_blocks(dates, cadence):
    """Return [(start, end)] prediction blocks. start is the refit cutoff."""
    dd = pd.to_datetime(pd.Series(dates), errors="coerce", utc=True).dropna().sort_values()
    if dd.empty:
        return []
    lo = dd.iloc[0].normalize()
    hi = dd.iloc[-1].normalize() + pd.Timedelta(days=1)
    if cadence == "FROZEN":
        return [(lo, hi)]
    spec = CADENCES[cadence]
    if spec == "MID":
        span_days = max(1, int((hi - lo).total_seconds() // 86400))
        mid = lo + pd.Timedelta(days=max(1, span_days // 2))
        return [(lo, mid), (mid, hi)] if mid < hi else [(lo, hi)]
    step = int(spec)
    out = []
    cur = lo
    while cur < hi:
        nxt = min(cur + pd.Timedelta(days=step), hi)
        out.append((cur, nxt)); cur = nxt
    return out


def _fit_stat_block(dashboard_module, g, train_mask, pred_mask, margin_cols, total_cols):
    mm, tm = dashboard_module._ncaaf_stat_fit_models_for_rows(
        g, margin_cols, total_cols, train_mask, target_mode="MARKET_ERROR_RESIDUAL"
    )
    edge_m = dashboard_module._ncaaf_stat_blend_predict(mm, g.loc[pred_mask, margin_cols], 0.75)
    edge_t = dashboard_module._ncaaf_stat_blend_predict(tm, g.loc[pred_mask, total_cols], 0.75)
    market_m = _num(g.loc[pred_mask, "Market_Open_Margin"], g.loc[pred_mask].index).to_numpy(float)
    market_t = _num(g.loc[pred_mask, "Market_Open_Total"], g.loc[pred_mask].index).to_numpy(float)
    return market_m + edge_m, market_t + edge_t


def _stat_predictions_for_season(dashboard_module, g, season_arr, target_season, cadence, margin_cols, total_cols):
    dates = pd.to_datetime(g.get("Game_Date"), errors="coerce", utc=True)
    target = np.isfinite(season_arr) & (season_arr == float(target_season)) & dates.notna().to_numpy()
    pm = np.full(len(g), np.nan, float); pt = np.full(len(g), np.nan, float)
    refits = 0; current_train_rows = []
    for start, end in _season_blocks(dates[target], cadence):
        pred_mask = target & dates.ge(start).to_numpy() & dates.lt(end).to_numpy()
        if not pred_mask.any():
            continue
        prior = np.isfinite(season_arr) & (season_arr < float(target_season))
        # At a block cutoff, only results strictly before that cutoff are available.
        current_prior = target & dates.lt(start).to_numpy()
        train_mask = prior | current_prior
        # Targets must exist for every fitted row.
        ym = _num(g.get("Market_Error_Margin"), g.index).to_numpy(float)
        yt = _num(g.get("Market_Error_Total"), g.index).to_numpy(float)
        train_mask = train_mask & np.isfinite(ym) & np.isfinite(yt)
        if int(train_mask.sum()) < 500:
            continue
        mhat, that = _fit_stat_block(dashboard_module, g, train_mask, pred_mask, margin_cols, total_cols)
        ix = np.flatnonzero(pred_mask)
        pm[ix] = mhat; pt[ix] = that
        refits += 1
        current_train_rows.append(int(current_prior.sum()))
    return pm, pt, refits, current_train_rows


def _new_h2h_pipe():
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=0.20, max_iter=1500, class_weight=None)),
    ])


def _h2h_predictions_for_season(g, season_arr, frozen_oof_margin, target_season, cadence, h2h_cols):
    dates = pd.to_datetime(g.get("Game_Date"), errors="coerce", utc=True)
    actual = _num(g.get("Actual_Margin"), g.index).to_numpy(float)
    target = np.isfinite(season_arr) & (season_arr == float(target_season)) & dates.notna().to_numpy()
    p = np.full(len(g), np.nan, float); refits = 0; current_train_rows = []
    X = g[list(h2h_cols)].copy()
    X["STAT_Fair_Margin"] = np.asarray(frozen_oof_margin, float)
    y = (actual > 0).astype(int)
    valid_x_anchor = np.isfinite(np.asarray(frozen_oof_margin, float)) & np.isfinite(actual)
    for start, end in _season_blocks(dates[target], cadence):
        pred_mask = target & dates.ge(start).to_numpy() & dates.lt(end).to_numpy() & valid_x_anchor
        if not pred_mask.any():
            continue
        prior = np.isfinite(season_arr) & (season_arr < float(target_season)) & valid_x_anchor
        current_prior = target & dates.lt(start).to_numpy() & valid_x_anchor
        train_mask = prior | current_prior
        if int(train_mask.sum()) < 500:
            continue
        pipe = _new_h2h_pipe(); pipe.fit(X.loc[train_mask], y[train_mask])
        p[np.flatnonzero(pred_mask)] = np.asarray(pipe.predict_proba(X.loc[pred_mask])[:, 1], float)
        refits += 1; current_train_rows.append(int(current_prior.sum()))
    return p, refits, current_train_rows


def _fmt(x, digits=4):
    try:
        x = float(x)
        return f"{x:.{digits}f}" if np.isfinite(x) else "nan"
    except Exception:
        return "nan"


def _aggregate_reg(season_records, market):
    # Aggregate via pooled row-level predictions, supplied by caller separately.
    raise RuntimeError("internal aggregate helper should not be called")


def run_refit_cadence_test_v1(*, dashboard_module, log_func=print, hard_fail=True):
    try:
        cache = getattr(dashboard_module, "_V1357_SPREAD_RESEARCH_CACHE", {})
        g = cache.get("games") if isinstance(cache, dict) else None
        frozen_oof_margin = cache.get("oof_margin") if isinstance(cache, dict) else None
        frozen_oof_total = cache.get("oof_total") if isinstance(cache, dict) else None
        if not isinstance(g, pd.DataFrame) or g.empty or frozen_oof_margin is None or frozen_oof_total is None:
            raise RuntimeError("V13 research cache unavailable")
        g = g.copy()
        g["Game_Date"] = pd.to_datetime(g.get("Game_Date"), errors="coerce", utc=True)
        season_arr = _num(g.get("Season"), g.index).to_numpy(float)
        available_seasons = sorted(int(x) for x in pd.Series(season_arr).dropna().unique())
        eval_seasons = [s for s in (*DISCOVERY_SEASONS, CONFIRM_SEASON) if s in available_seasons]
        if CONFIRM_SEASON not in eval_seasons or len([s for s in DISCOVERY_SEASONS if s in eval_seasons]) < 2:
            raise RuntimeError(f"insufficient cadence seasons={available_seasons}")

        missing_sp = [c for c in SPREAD_FEATURES if c not in g.columns]
        missing_tot = [c for c in TOTAL_FEATURES if c not in g.columns]
        h2h_cols = [c for c in H2H_FEATURES if c in g.columns]
        if missing_sp or missing_tot or len(h2h_cols) < 4:
            raise RuntimeError(f"fixed backbone feature contract failed missing_spread={missing_sp} missing_total={missing_tot} h2h_cols={h2h_cols}")

        log_func(
            f"[CADENCE-V1-PREFLIGHT] status=PASS source_tag={REFIT_CADENCE_TEST_V1_SOURCE_TAG} games={len(g)} "
            f"seasons={eval_seasons} discovery_seasons={list(DISCOVERY_SEASONS)} confirm_season={CONFIRM_SEASON} "
            f"cadences={','.join(CADENCES)} spread_features={list(SPREAD_FEATURES)} total_features={list(TOTAL_FEATURES)} "
            f"h2h_features={h2h_cols} feature_selection_refit=FALSE model_head_selection=FALSE same_day_leakage_blocked=TRUE "
            f"weekly_arm_is_controlled_fixed_backbone_benchmark=TRUE retired_weekly_adaptive_model_reactivated=FALSE production_authority=0"
        )

        actual_m = _num(g.get("Actual_Margin"), g.index).to_numpy(float)
        actual_t = _num(g.get("Actual_Total"), g.index).to_numpy(float)
        market_m = _num(g.get("Market_Open_Margin"), g.index).to_numpy(float)
        market_t = _num(g.get("Market_Open_Total"), g.index).to_numpy(float)
        market_h = _num(g.get("Market_Open_H2H_Fair"), g.index).to_numpy(float)

        preds = {}
        rows = []
        for cadence in CADENCES:
            sm = np.full(len(g), np.nan, float); st = np.full(len(g), np.nan, float); hp = np.full(len(g), np.nan, float)
            for sy in eval_seasons:
                mhat, that, stat_refits, stat_cur = _stat_predictions_for_season(
                    dashboard_module, g, season_arr, sy, cadence, list(SPREAD_FEATURES), list(TOTAL_FEATURES)
                )
                ph, h_refits, h_cur = _h2h_predictions_for_season(
                    g, season_arr, frozen_oof_margin, sy, cadence, h2h_cols
                )
                sy_mask = np.isfinite(season_arr) & (season_arr == float(sy))
                sm[sy_mask] = mhat[sy_mask]; st[sy_mask] = that[sy_mask]; hp[sy_mask] = ph[sy_mask]
                smet = _reg_metrics(actual_m[sy_mask], sm[sy_mask], market_m[sy_mask], "SPREAD")
                tmet = _reg_metrics(actual_t[sy_mask], st[sy_mask], market_t[sy_mask], "TOTAL")
                hmet = _h2h_metrics(actual_m[sy_mask], hp[sy_mask], market_h[sy_mask])
                rows.append({"cadence": cadence, "season": sy, "spread": smet, "totals": tmet, "h2h": hmet})
                log_func(
                    f"[CADENCE-V1-SEASON] cadence={cadence} season={sy} stat_refits={stat_refits} h2h_refits={h_refits} "
                    f"max_current_rows_before_stat_refit={max(stat_cur) if stat_cur else 0} max_current_rows_before_h2h_refit={max(h_cur) if h_cur else 0} "
                    f"spread_n={smet['n']} spread_rmse={_fmt(smet['rmse'])} spread_mae={_fmt(smet['mae'])} spread_hit={_fmt(smet['hit'])} spread_roi={_fmt(smet['roi'])} spread_edge2_n={smet['edge2_n']} spread_edge2_hit={_fmt(smet['edge2_hit'])} "
                    f"totals_n={tmet['n']} totals_rmse={_fmt(tmet['rmse'])} totals_mae={_fmt(tmet['mae'])} totals_hit={_fmt(tmet['hit'])} totals_roi={_fmt(tmet['roi'])} totals_edge2_n={tmet['edge2_n']} totals_edge2_hit={_fmt(tmet['edge2_hit'])} "
                    f"h2h_n={hmet['n']} h2h_auc={_fmt(hmet['auc'])} h2h_ll={_fmt(hmet['logloss'],6)} h2h_brier={_fmt(hmet['brier'],6)} market_h2h_ll={_fmt(hmet['market_logloss'],6)} production_authority=0"
                )
            preds[cadence] = {"spread": sm, "totals": st, "h2h": hp}

        disc_mask = np.isin(season_arr, np.asarray(DISCOVERY_SEASONS, float))
        conf_mask = season_arr == float(CONFIRM_SEASON)
        summaries = {}
        for cadence in CADENCES:
            pp = preds[cadence]
            sd = _reg_metrics(actual_m[disc_mask], pp["spread"][disc_mask], market_m[disc_mask], "SPREAD")
            td = _reg_metrics(actual_t[disc_mask], pp["totals"][disc_mask], market_t[disc_mask], "TOTAL")
            hd = _h2h_metrics(actual_m[disc_mask], pp["h2h"][disc_mask], market_h[disc_mask])
            sc = _reg_metrics(actual_m[conf_mask], pp["spread"][conf_mask], market_m[conf_mask], "SPREAD")
            tc = _reg_metrics(actual_t[conf_mask], pp["totals"][conf_mask], market_t[conf_mask], "TOTAL")
            hc = _h2h_metrics(actual_m[conf_mask], pp["h2h"][conf_mask], market_h[conf_mask])
            summaries[cadence] = {"discovery": {"spread": sd, "totals": td, "h2h": hd}, "confirm": {"spread": sc, "totals": tc, "h2h": hc}}
            log_func(
                f"[CADENCE-V1-SUMMARY] sample=DISCOVERY cadence={cadence} "
                f"spread_n={sd['n']} spread_rmse={_fmt(sd['rmse'])} spread_hit={_fmt(sd['hit'])} spread_edge2_n={sd['edge2_n']} spread_edge2_hit={_fmt(sd['edge2_hit'])} "
                f"totals_n={td['n']} totals_rmse={_fmt(td['rmse'])} totals_hit={_fmt(td['hit'])} totals_edge2_n={td['edge2_n']} totals_edge2_hit={_fmt(td['edge2_hit'])} "
                f"h2h_n={hd['n']} h2h_auc={_fmt(hd['auc'])} h2h_ll={_fmt(hd['logloss'],6)} h2h_brier={_fmt(hd['brier'],6)} production_authority=0"
            )
            log_func(
                f"[CADENCE-V1-SUMMARY] sample=CONFIRM_2026 cadence={cadence} "
                f"spread_n={sc['n']} spread_rmse={_fmt(sc['rmse'])} spread_hit={_fmt(sc['hit'])} spread_edge2_n={sc['edge2_n']} spread_edge2_hit={_fmt(sc['edge2_hit'])} "
                f"totals_n={tc['n']} totals_rmse={_fmt(tc['rmse'])} totals_hit={_fmt(tc['hit'])} totals_edge2_n={tc['edge2_n']} totals_edge2_hit={_fmt(tc['edge2_hit'])} "
                f"h2h_n={hc['n']} h2h_auc={_fmt(hc['auc'])} h2h_ll={_fmt(hc['logloss'],6)} h2h_brier={_fmt(hc['brier'],6)} production_authority=0"
            )

        # Select a discovery leader independently by market. Lower proper regression
        # error / logloss wins. Confirmation is never part of selection.
        spread_leader = min(CADENCES, key=lambda c: summaries[c]["discovery"]["spread"]["rmse"] if np.isfinite(summaries[c]["discovery"]["spread"]["rmse"]) else np.inf)
        total_leader = min(CADENCES, key=lambda c: summaries[c]["discovery"]["totals"]["rmse"] if np.isfinite(summaries[c]["discovery"]["totals"]["rmse"]) else np.inf)
        h2h_leader = min(CADENCES, key=lambda c: summaries[c]["discovery"]["h2h"]["logloss"] if np.isfinite(summaries[c]["discovery"]["h2h"]["logloss"]) else np.inf)

        def _decision(market, leader, metric):
            d0 = summaries["FROZEN"]["discovery"][market][metric]
            dl = summaries[leader]["discovery"][market][metric]
            c0 = summaries["FROZEN"]["confirm"][market][metric]
            cl = summaries[leader]["confirm"][market][metric]
            disc_gain = float(d0 - dl) if np.isfinite(d0) and np.isfinite(dl) else np.nan
            conf_gain = float(c0 - cl) if np.isfinite(c0) and np.isfinite(cl) else np.nan
            if leader == "FROZEN":
                state = "FROZEN_REMAINS_BEST_DISCOVERY"
            elif np.isfinite(disc_gain) and disc_gain > 0 and np.isfinite(conf_gain) and conf_gain > 0:
                state = "CADENCE_IMPROVEMENT_REPEATS_2026"
            elif np.isfinite(disc_gain) and disc_gain > 0:
                state = "DISCOVERY_IMPROVEMENT_NOT_CONFIRMED"
            else:
                state = "NO_DISCOVERY_IMPROVEMENT"
            log_func(
                f"[CADENCE-V1-DECISION] market={market.upper()} discovery_leader={leader} metric={metric} "
                f"frozen_discovery={_fmt(d0,6)} leader_discovery={_fmt(dl,6)} discovery_gain={_fmt(disc_gain,6)} "
                f"frozen_confirm={_fmt(c0,6)} leader_confirm={_fmt(cl,6)} confirm_gain={_fmt(conf_gain,6)} "
                f"state={state} selection_uses_2026=FALSE auto_production_change=FALSE production_authority=0"
            )
            return {"leader": leader, "state": state, "discovery_gain": disc_gain, "confirm_gain": conf_gain}

        decisions = {
            "spread": _decision("spread", spread_leader, "rmse"),
            "totals": _decision("totals", total_leader, "rmse"),
            "h2h": _decision("h2h", h2h_leader, "logloss"),
        }

        # Stability: compare cadence-side signs with the frozen arm on the same rows.
        for cadence in CADENCES:
            if cadence == "FROZEN":
                continue
            for sample, mask in (("DISCOVERY", disc_mask), ("CONFIRM_2026", conf_mask)):
                sf = preds["FROZEN"]["spread"] - market_m
                sc = preds[cadence]["spread"] - market_m
                tf = preds["FROZEN"]["totals"] - market_t
                tc = preds[cadence]["totals"] - market_t
                so = mask & np.isfinite(sf) & np.isfinite(sc) & ~np.isclose(sf,0) & ~np.isclose(sc,0)
                to = mask & np.isfinite(tf) & np.isfinite(tc) & ~np.isclose(tf,0) & ~np.isclose(tc,0)
                sagree = float(np.mean(np.sign(sf[so]) == np.sign(sc[so]))) if so.any() else np.nan
                tagree = float(np.mean(np.sign(tf[to]) == np.sign(tc[to]))) if to.any() else np.nan
                log_func(f"[CADENCE-V1-STABILITY] sample={sample} cadence={cadence} spread_same_side_n={int(so.sum())} spread_side_agree={_fmt(sagree)} totals_same_side_n={int(to.sum())} totals_side_agree={_fmt(tagree)} production_authority=0")

        log_func(
            f"[CADENCE-V1-CONTRACT] status=PASS cadences={','.join(CADENCES)} discovery_seasons={list(DISCOVERY_SEASONS)} confirm_season={CONFIRM_SEASON} "
            f"feature_set_fixed=TRUE model_heads_fixed=TRUE hyperparameters_fixed=TRUE only_refit_timing_changes=TRUE current_season_rows_prior_only=TRUE same_day_results_excluded=TRUE "
            f"weekly_adaptive_legacy_remains_retired=TRUE spread_edge_policy_unchanged=TRUE totals_edge_families_unchanged=TRUE h2h_edge_authority_unchanged=TRUE "
            f"selection_uses_2026=FALSE auto_promotion=FALSE production_authority=0"
        )
        return {"status":"PASS","source_tag":REFIT_CADENCE_TEST_V1_SOURCE_TAG,"summaries":summaries,"decisions":decisions,"production_authority":0}
    except Exception as e:
        log_func(f"[CADENCE-V1-CONTRACT] status=FAILED error={type(e).__name__}:{e} production_authority=0")
        if hard_fail:
            raise
        return {"status":"FAILED","error":f"{type(e).__name__}:{e}","production_authority":0}
