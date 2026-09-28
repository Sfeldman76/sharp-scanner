"""NCAAF V14 Clean Room V1 research challenger.

Purpose
-------
Test whether a simpler, directly market-relative ATS probability model improves on
V13 STAT without changing any production artifact, threshold, authority, or serving
path.  V14 deliberately reuses only the leakage-safe one-game feature frame built by
V13; it does NOT reuse V13 feature admission, champion selection, resolver weights,
systems, or production calibration.

Research contract
-----------------
* Spread only in V1.
* Fixed model classes / hyperparameters; no broad tuning search.
* No supervised one-feature-at-a-time admission.
* Feature availability is frozen using pre-evaluation history only.
* Season-forward OOF begins only once two prior seasons exist.
* Platt calibration is fit only from earlier OOF predictions.
* Weekly walk-forward is expanding-window and frozen within each week.
* Systems/Miner are excluded in V1 so the base probability question is isolated.
* All authority remains zero; this module never writes production artifacts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

V14_CLEAN_ROOM_SOURCE_TAG = "v14-clean-room-v1-direct-ats-market-relative"
V14_MODEL_NAMES = ("L2_LOGIT", "ELASTIC_NET", "SHALLOW_HGB")
V14_THRESHOLDS = (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05)
V14_EDGE_BINS = (0.0, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, np.inf)
V14_EDGE_LABELS = ("0-.01", ".01-.02", ".02-.025", ".025-.03", ".03-.04", ".04-.05", ".05+")


def _num(s, index=None):
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce")
    if index is None:
        return pd.Series(pd.to_numeric(s, errors="coerce"))
    return pd.Series(s, index=index, dtype="float64").pipe(pd.to_numeric, errors="coerce")


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def _ece(y, p, bins=10):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]; p = np.clip(p[ok], 1e-6, 1 - 1e-6)
    if len(y) == 0:
        return np.nan
    edges = np.linspace(0.0, 1.0, bins + 1)
    ans = 0.0
    for i in range(bins):
        hi = edges[i + 1] + (1e-12 if i == bins - 1 else 0.0)
        m = (p >= edges[i]) & (p < hi)
        if m.any():
            ans += float(m.mean()) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(ans)


def _metrics(y, p):
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    yy = y[ok].astype(int); pp = np.clip(p[ok], 1e-6, 1 - 1e-6)
    out = {"n": int(len(yy)), "auc": np.nan, "logloss": np.nan, "brier": np.nan,
           "ece": np.nan, "cal_intercept": np.nan, "cal_slope": np.nan}
    if len(yy) < 20 or np.unique(yy).size < 2:
        return out
    out["auc"] = float(roc_auc_score(yy, pp))
    out["logloss"] = float(log_loss(yy, pp, labels=[0, 1]))
    out["brier"] = float(np.mean((pp - yy) ** 2))
    out["ece"] = _ece(yy, pp)
    if len(yy) >= 100:
        try:
            lr = LogisticRegression(C=10.0, solver="lbfgs", max_iter=1500)
            lr.fit(_logit(pp).reshape(-1, 1), yy)
            out["cal_intercept"] = float(lr.intercept_[0])
            out["cal_slope"] = float(lr.coef_[0, 0])
        except Exception:
            pass
    return out


def _fit_platt(y, raw_p):
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y, dtype=float); raw_p = np.asarray(raw_p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(raw_p)
    yy = y[ok].astype(int); pp = np.clip(raw_p[ok], 1e-6, 1 - 1e-6)
    if len(yy) < 250 or np.unique(yy).size < 2:
        return None
    try:
        lr = LogisticRegression(C=10.0, solver="lbfgs", max_iter=1500)
        lr.fit(_logit(pp).reshape(-1, 1), yy)
        return (float(lr.intercept_[0]), float(lr.coef_[0, 0]), int(len(yy)))
    except Exception:
        return None


def _apply_platt(raw_p, cal):
    raw_p = np.asarray(raw_p, dtype=float)
    if cal is None:
        return np.clip(raw_p, 1e-5, 1 - 1e-5)
    a, b, _ = cal
    return np.clip(_sigmoid(a + b * _logit(raw_p)), 1e-5, 1 - 1e-5)


def _new_model(name: str):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier

    name = str(name).upper()
    if name == "L2_LOGIT":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.10, solver="lbfgs", max_iter=2000, tol=1e-4)),
        ])
    if name == "ELASTIC_NET":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.10, l1_ratio=0.15, solver="saga", max_iter=1800, tol=1e-3, random_state=42)),
        ])
    if name == "SHALLOW_HGB":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=False, keep_empty_features=True)),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.035, max_iter=120, max_leaf_nodes=7, max_depth=2,
                min_samples_leaf=35, l2_regularization=5.0, random_state=42,
            )),
        ])
    raise ValueError(f"Unknown V14 model {name}")


def _feature_family(name: str, helper=None):
    c = str(name)
    if c.startswith("CR_Market_"):
        return "market_context"
    if c.startswith("Context_Prev_") or c.startswith("Context_Opp_Prev_"):
        return "prior_form"
    if c.startswith("Context_"):
        return "context"
    if c.startswith("Matchup_"):
        return "matchup"
    if c.startswith("Sum_") or c.startswith("Mean_"):
        return "combination"
    if helper is not None:
        try:
            return str(helper(c))
        except Exception:
            pass
    if "RawRecent3" in c or "Recent3" in c:
        return "recency"
    if "RawSeason" in c or "State" in c:
        return "team_state"
    return "other"


def _prepare_dataset(games: pd.DataFrame, candidate_feature_cols: Iterable[str], feature_family_helper=None):
    if games is None or games.empty:
        raise RuntimeError("V14 clean room received empty game frame")
    g = games.copy()
    g["Game_Date"] = pd.to_datetime(g.get("Game_Date"), errors="coerce", utc=True)
    g["Season"] = _num(g.get("Season"), g.index)
    margin = _num(g.get("Actual_Margin"), g.index)
    spread = _num(g.get("Consensus_Open_Spread"), g.index)
    total = _num(g.get("Consensus_Open_Total"), g.index)
    fair_ml = _num(g.get("Market_Open_H2H_Fair"), g.index)
    ats_margin = margin + spread
    g["V14_ATS_TARGET"] = np.where(ats_margin.notna() & ~np.isclose(ats_margin, 0.0, atol=1e-9), (ats_margin > 0).astype(float), np.nan)

    # Explicit market context.  For ATS, 0.50 is the probability prior; the line
    # itself is the market's information-rich difficulty adjustment and therefore
    # belongs in the feature frame rather than being reconstructed downstream.
    g["CR_Market_Spread"] = spread
    g["CR_Market_AbsSpread"] = spread.abs()
    g["CR_Market_ImpliedMargin"] = -spread
    g["CR_Market_Total"] = total
    g["CR_Market_H2H_Fair"] = fair_ml
    g["CR_Market_IsFavorite"] = np.where(spread.notna(), (spread < 0).astype(float), np.nan)
    g["CR_Market_IsDog"] = np.where(spread.notna(), (spread > 0).astype(float), np.nan)
    g["CR_Market_SpreadToTotal"] = spread.abs() / total.where(total.abs() > 1e-9)

    cols = []
    for c in list(dict.fromkeys(candidate_feature_cols or [])):
        if c in g.columns and c != "Context_Intercept":
            cols.append(c)
    market_cols = [c for c in g.columns if c.startswith("CR_Market_")]
    cols = list(dict.fromkeys(cols + market_cols))

    seasons = sorted(int(x) for x in g["Season"].dropna().unique())
    if len(seasons) < 3:
        raise RuntimeError(f"V14 needs >=3 seasons, found {seasons}")
    # Evaluation begins only after two prior seasons exist. Feature availability is
    # frozen from those two seasons and never selected with target performance.
    first_eval = seasons[2]
    pre = g["Season"].lt(float(first_eval)) & g["V14_ATS_TARGET"].notna()
    kept = []
    availability = {}
    for c in cols:
        s = _num(g.loc[pre, c])
        finite = s[np.isfinite(s.to_numpy(dtype=float, na_value=np.nan))]
        n = int(len(finite)); availability[c] = n
        if n < 50:
            continue
        if float(finite.std(ddof=0)) <= 1e-12:
            continue
        kept.append(c)
    if len(kept) < 10:
        raise RuntimeError(f"V14 feature universe too small after pre-eval availability freeze: {len(kept)}")
    families = {c: _feature_family(c, feature_family_helper) for c in kept}
    return g, kept, families, seasons, first_eval, availability


def _v13_control_probs(g: pd.DataFrame, oof_margin, eval_seasons, empirical_prob_fn):
    """Existing V13 season-forward reference, kept separate from clean-room fits."""
    om = np.asarray(oof_margin, dtype=float)
    if len(om) != len(g):
        return np.full(len(g), np.nan)
    season = _num(g["Season"]).to_numpy(dtype=float)
    actual = _num(g["Actual_Margin"]).to_numpy(dtype=float)
    spread = _num(g["Consensus_Open_Spread"]).to_numpy(dtype=float)
    out = np.full(len(g), np.nan)
    for sy in eval_seasons:
        hist = np.isfinite(season) & (season < float(sy)) & np.isfinite(actual) & np.isfinite(om)
        resid = (actual - om)[hist]
        test = np.isfinite(season) & (season == float(sy)) & np.isfinite(om) & np.isfinite(spread)
        if int(hist.sum()) < 100 or int(test.sum()) == 0:
            continue
        out[test] = empirical_prob_fn(-(om[test] + spread[test]), resid)
    return out


def _season_forward_predictions(g, features, seasons, first_eval, model_name, log_func=lambda *_: None):
    y = _num(g["V14_ATS_TARGET"]).to_numpy(dtype=float)
    season = _num(g["Season"]).to_numpy(dtype=float)
    raw_oof = np.full(len(g), np.nan); cal_oof = np.full(len(g), np.nan)
    prior_oof_y: List[float] = []; prior_oof_p: List[float] = []
    fold_rows = []
    # Seed calibration history with the first season that has one prior season.
    seed_and_eval = seasons[1:]
    for sy in seed_and_eval:
        tr = np.isfinite(season) & (season < float(sy)) & np.isfinite(y)
        va = np.isfinite(season) & (season == float(sy)) & np.isfinite(y)
        if int(tr.sum()) < 500 or int(va.sum()) < 75:
            continue
        model = _new_model(model_name)
        try:
            model.fit(g.loc[tr, features], y[tr].astype(int))
            rp = np.asarray(model.predict_proba(g.loc[va, features])[:, 1], dtype=float)
        except Exception as e:
            log_func(f"[V14-MODEL-FOLD] model={model_name} season={sy} status=FAILED error={type(e).__name__}:{e}")
            continue
        idx = np.where(va)[0]
        raw_oof[idx] = rp
        cal = _fit_platt(np.asarray(prior_oof_y), np.asarray(prior_oof_p)) if prior_oof_y else None
        cp = _apply_platt(rp, cal)
        cal_oof[idx] = cp
        if sy >= first_eval:
            mr = _metrics(y[idx], rp); mc = _metrics(y[idx], cp)
            fold_rows.append({"season": int(sy), "n": int(len(idx)), "raw": mr, "cal": mc,
                              "calibration_n": int(cal[2]) if cal is not None else 0})
        # Only after scoring this season may it become calibration history.
        ok = np.isfinite(rp) & np.isfinite(y[idx])
        prior_oof_p.extend(rp[ok].tolist()); prior_oof_y.extend(y[idx][ok].tolist())
    return raw_oof, cal_oof, fold_rows


def _bet_summary(y, p, threshold=0.025):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok].astype(int); p = p[ok]
    side_team = p >= 0.5
    prob = np.maximum(p, 1.0 - p); edge = prob - 0.5
    won = np.where(side_team, y == 1, y == 0)
    bet = edge >= float(threshold)
    ret = np.where(won, 100.0 / 110.0, -1.0)
    return {
        "n": int(len(y)), "bets": int(bet.sum()), "wins": int(won[bet].sum()) if bet.any() else 0,
        "hit_rate": float(won[bet].mean()) if bet.any() else np.nan,
        "roi": float(ret[bet].mean()) if bet.any() else np.nan,
        "avg_edge": float(edge[bet].mean()) if bet.any() else np.nan,
    }


def _threshold_rows(model_name, y, p):
    rows = []
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p); yy = y[ok].astype(int); pp = p[ok]
    side_team = pp >= 0.5; won = np.where(side_team, yy == 1, yy == 0)
    edge = np.maximum(pp, 1.0 - pp) - 0.5; ret = np.where(won, 100.0 / 110.0, -1.0)
    for th in V14_THRESHOLDS:
        b = edge >= th
        rows.append({"model": model_name, "threshold": th, "bets": int(b.sum()),
                     "wins": int(won[b].sum()) if b.any() else 0,
                     "hit_rate": float(won[b].mean()) if b.any() else np.nan,
                     "roi": float(ret[b].mean()) if b.any() else np.nan})
    return rows


def _edge_bucket_rows(model_name, y, p):
    rows = []
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p); yy = y[ok].astype(int); pp = p[ok]
    side_team = pp >= 0.5; won = np.where(side_team, yy == 1, yy == 0)
    edge = np.maximum(pp, 1.0 - pp) - 0.5; ret = np.where(won, 100.0 / 110.0, -1.0)
    for lo, hi, lab in zip(V14_EDGE_BINS[:-1], V14_EDGE_BINS[1:], V14_EDGE_LABELS):
        m = (edge >= lo) & (edge < hi)
        rows.append({"model": model_name, "bucket": lab, "n": int(m.sum()),
                     "hit_rate": float(won[m].mean()) if m.any() else np.nan,
                     "roi": float(ret[m].mean()) if m.any() else np.nan,
                     "avg_edge": float(edge[m].mean()) if m.any() else np.nan})
    return rows


def _weekly_walk_forward(g, features, model_names, start_season=2023, log_func=lambda *_: None):
    dates = pd.to_datetime(g["Game_Date"], errors="coerce", utc=True)
    season = _num(g["Season"]).to_numpy(dtype=float)
    y = _num(g["V14_ATS_TARGET"]).to_numpy(dtype=float)
    week = dates.dt.normalize() - pd.to_timedelta(dates.dt.weekday, unit="D")
    eligible = dates.notna().to_numpy() & np.isfinite(season) & (season >= float(start_season)) & np.isfinite(y)
    weeks = sorted(pd.Timestamp(x) for x in week[eligible].dropna().unique())
    records = []
    for model_name in model_names:
        cal_y: List[float] = []; cal_p: List[float] = []
        for ws in weeks:
            tr = dates.notna().to_numpy() & (dates < ws).to_numpy() & np.isfinite(y)
            va = dates.notna().to_numpy() & (week == ws).to_numpy() & eligible
            if int(tr.sum()) < 500 or int(va.sum()) == 0:
                continue
            model = _new_model(model_name)
            try:
                model.fit(g.loc[tr, features], y[tr].astype(int))
                rp = np.asarray(model.predict_proba(g.loc[va, features])[:, 1], dtype=float)
            except Exception as e:
                log_func(f"[V14-WF-FOLD] model={model_name} week={ws.date()} status=FAILED error={type(e).__name__}:{e}")
                continue
            cal = _fit_platt(np.asarray(cal_y), np.asarray(cal_p)) if cal_y else None
            cp = _apply_platt(rp, cal)
            idx = np.where(va)[0]
            for j, pos in enumerate(idx):
                records.append({"model": model_name, "week": ws, "season": int(season[pos]),
                                "row_index": int(pos), "y": int(y[pos]), "raw_p": float(rp[j]),
                                "p": float(cp[j]), "train_rows": int(tr.sum()),
                                "calibration_n": int(cal[2]) if cal is not None else 0})
            ok = np.isfinite(rp) & np.isfinite(y[idx])
            cal_p.extend(rp[ok].tolist()); cal_y.extend(y[idx][ok].tolist())
    return pd.DataFrame(records)


def _season_metrics_rows(model, y, p, season, eval_seasons):
    rows = []
    for sy in eval_seasons:
        m = np.isfinite(season) & (season == float(sy)) & np.isfinite(y) & np.isfinite(p)
        if int(m.sum()) < 20:
            continue
        met = _metrics(y[m], p[m]); bet = _bet_summary(y[m], p[m], 0.025)
        rows.append({"model": model, "season": int(sy), **met, **{f"bet_{k}": v for k, v in bet.items() if k != "n"}})
    return rows


def _run_ablation(g, features, families, seasons, first_eval, model_name, baseline_ll, log_func=lambda *_: None):
    y = _num(g["V14_ATS_TARGET"]).to_numpy(dtype=float)
    season = _num(g["Season"]).to_numpy(dtype=float)
    out = []
    fam_to_cols: Dict[str, List[str]] = {}
    for c in features:
        fam_to_cols.setdefault(families.get(c, "other"), []).append(c)
    for fam, cols in sorted(fam_to_cols.items()):
        if len(cols) < 2 or len(features) - len(cols) < 10:
            continue
        use = [c for c in features if c not in set(cols)]
        _, poof, _ = _season_forward_predictions(g, use, seasons, first_eval, model_name, log_func=lambda *_: None)
        m = np.isfinite(season) & (season >= float(first_eval)) & np.isfinite(y) & np.isfinite(poof)
        met = _metrics(y[m], poof[m])
        delta = float(met["logloss"] - baseline_ll) if np.isfinite(met.get("logloss", np.nan)) and np.isfinite(baseline_ll) else np.nan
        out.append({"family": fam, "removed_features": int(len(cols)), "remaining_features": int(len(use)),
                    "n": int(m.sum()), "logloss": met.get("logloss", np.nan), "brier": met.get("brier", np.nan),
                    "auc": met.get("auc", np.nan), "delta_logloss_vs_full": delta})
    return out


def run_v14_clean_room(*, dashboard_module, log_func=print, hard_fail=True):
    """Run isolated NCAAF Spread clean-room research. Never changes production state."""
    try:
        cache = getattr(dashboard_module, "_V1357_SPREAD_RESEARCH_CACHE", {})
        games = cache.get("games") if isinstance(cache, dict) else None
        oof_margin = cache.get("oof_margin") if isinstance(cache, dict) else None
        stat_cache = getattr(dashboard_module, "_NCAAF_STAT_TRAIN_CACHE", {})
        bundle = stat_cache.get("bundle") if isinstance(stat_cache, dict) else None
        if games is None or not isinstance(games, pd.DataFrame) or games.empty:
            raise RuntimeError("V13 research game cache unavailable; run V14 only after NCAAF Spread training")
        candidate_cols = list((bundle or {}).get("candidate_feature_cols") or [])
        if not candidate_cols:
            # Fail-safe reconstruction from known leakage-safe feature namespaces.
            candidate_cols = [c for c in games.columns if str(c).startswith(("A_", "B_", "Diff_", "Matchup_", "Sum_", "Mean_", "Context_"))]
        helper = getattr(dashboard_module, "_ncaaf_stat_feature_family", None)
        empirical = getattr(dashboard_module, "_ncaaf_stat_empirical_prob_gt", None)
        if empirical is None:
            raise RuntimeError("V13 empirical probability helper unavailable")

        g, features, families, seasons, first_eval, availability = _prepare_dataset(games, candidate_cols, helper)
        y = _num(g["V14_ATS_TARGET"]).to_numpy(dtype=float)
        season = _num(g["Season"]).to_numpy(dtype=float)
        eval_seasons = [s for s in seasons if s >= first_eval]
        log_func(f"[V14-CLEANROOM-PREFLIGHT] status=PASS source_tag={V14_CLEAN_ROOM_SOURCE_TAG} rows={len(g)} seasons={seasons} first_eval={first_eval} candidate_features={len(candidate_cols)} frozen_features={len(features)} production_authority=0 systems_excluded=TRUE")
        fam_counts = pd.Series([families[c] for c in features]).value_counts().to_dict()
        log_func(f"[V14-FEATURE-UNIVERSE] rule=PRE_EVAL_AVAILABILITY_ONLY__NO_TARGET_SCREEN__NO_GREEDY_ADMISSION features={len(features)} families={fam_counts} market_context={[c for c in features if c.startswith('CR_Market_')]}")

        preds: Dict[str, np.ndarray] = {}
        raw_preds: Dict[str, np.ndarray] = {}
        folds: Dict[str, list] = {}
        for model_name in V14_MODEL_NAMES:
            raw, cal, fr = _season_forward_predictions(g, features, seasons, first_eval, model_name, log_func=log_func)
            raw_preds[model_name] = raw; preds[model_name] = cal; folds[model_name] = fr
            for row in fr:
                r = row["raw"]; c = row["cal"]
                log_func(f"[V14-SEASON] model={model_name} season={row['season']} n={row['n']} raw_auc={r['auc']:.4f} raw_ll={r['logloss']:.6f} raw_brier={r['brier']:.6f} cal_auc={c['auc']:.4f} cal_ll={c['logloss']:.6f} cal_brier={c['brier']:.6f} cal_ece={c['ece']:.4f} cal_slope={c['cal_slope']:.3f} calibration_history_n={row['calibration_n']}")

        # Existing V13 control is evaluated on the same clean-room seasons.  It is
        # intentionally a reference only; its historical feature-selection path is
        # not copied into V14.
        v13p = _v13_control_probs(g, oof_margin, eval_seasons, empirical) if oof_margin is not None else np.full(len(g), np.nan)
        preds["V13_STAT_CONTROL"] = v13p

        overall = {}; season_rows = []
        eval_mask = np.isfinite(season) & (season >= float(first_eval)) & np.isfinite(y)
        market_p = np.full(len(g), 0.5, dtype=float)
        market_met = _metrics(y[eval_mask], market_p[eval_mask])
        log_func(f"[V14-MARKET-BASELINE] n={market_met['n']} auc={market_met['auc']:.4f} ll={market_met['logloss']:.6f} brier={market_met['brier']:.6f} prior=0.500000")
        for model_name, p in preds.items():
            m = eval_mask & np.isfinite(p)
            met = _metrics(y[m], p[m]); bet = _bet_summary(y[m], p[m], 0.025)
            overall[model_name] = {**met, **bet,
                                   "ll_skill_vs_market": float(market_met["logloss"] - met["logloss"]) if np.isfinite(met["logloss"]) else np.nan,
                                   "brier_skill_vs_market": float(market_met["brier"] - met["brier"]) if np.isfinite(met["brier"]) else np.nan}
            log_func(f"[V14-OVERALL] model={model_name} n={met['n']} auc={met['auc']:.4f} ll={met['logloss']:.6f} brier={met['brier']:.6f} ece={met['ece']:.4f} cal_intercept={met['cal_intercept']:.3f} cal_slope={met['cal_slope']:.3f} ll_skill_vs_market={overall[model_name]['ll_skill_vs_market']:+.6f} brier_skill_vs_market={overall[model_name]['brier_skill_vs_market']:+.6f} bets_025={bet['bets']} hit_025={bet['hit_rate']:.4f} roi_025={bet['roi']:.4f} production_authority=0")
            sr = _season_metrics_rows(model_name, y, p, season, eval_seasons); season_rows.extend(sr)
            for row in sr:
                log_func(f"[V14-SEASON-SUMMARY] model={model_name} season={row['season']} n={row['n']} auc={row['auc']:.4f} ll={row['logloss']:.6f} brier={row['brier']:.6f} ece={row['ece']:.4f} bets_025={row['bet_bets']} hit_025={row['bet_hit_rate']:.4f} roi_025={row['bet_roi']:.4f}")
            for row in _threshold_rows(model_name, y[m], p[m]):
                log_func(f"[V14-THRESHOLD] model={model_name} threshold={row['threshold']:.3f} bets={row['bets']} wins={row['wins']} hit_rate={row['hit_rate']:.4f} roi={row['roi']:.4f} diagnostic_only=TRUE")
            for row in _edge_bucket_rows(model_name, y[m], p[m]):
                log_func(f"[V14-EDGE-BUCKET] model={model_name} bucket={row['bucket']} n={row['n']} hit_rate={row['hit_rate']:.4f} roi={row['roi']:.4f} avg_edge={row['avg_edge']:.4f}")

        # Research leader is descriptive only and cannot promote. Proper score is
        # deliberately primary; ROI is not used to choose the model.
        clean_candidates = [m for m in V14_MODEL_NAMES if m in overall and np.isfinite(overall[m].get("logloss", np.nan))]
        if not clean_candidates:
            raise RuntimeError("No V14 challenger produced season-forward OOF probabilities")
        leader = min(clean_candidates, key=lambda m: (overall[m]["logloss"], overall[m]["brier"]))
        leader_met = overall[leader]
        beats_market = bool(leader_met["logloss"] < market_met["logloss"] and leader_met["brier"] < market_met["brier"])
        v13_ll = overall.get("V13_STAT_CONTROL", {}).get("logloss", np.nan)
        v13_br = overall.get("V13_STAT_CONTROL", {}).get("brier", np.nan)
        beats_v13 = bool(np.isfinite(v13_ll) and np.isfinite(v13_br) and leader_met["logloss"] < v13_ll and leader_met["brier"] < v13_br)
        log_func(f"[V14-LEADER] model={leader} selection=LOWEST_SEASON_FORWARD_LOGLOSS_THEN_BRIER ll={leader_met['logloss']:.6f} brier={leader_met['brier']:.6f} beats_market={beats_market} beats_v13={beats_v13} production_authority=0")

        # Feature-family ablation is performed only on the descriptive leader.
        ablation = _run_ablation(g, features, families, seasons, first_eval, leader, leader_met["logloss"], log_func=log_func)
        for row in ablation:
            direction = "HURTS_WHEN_REMOVED" if np.isfinite(row["delta_logloss_vs_full"]) and row["delta_logloss_vs_full"] > 0 else "HELPS_WHEN_REMOVED"
            log_func(f"[V14-ABLATION] leader={leader} family={row['family']} removed_features={row['removed_features']} n={row['n']} ll_without={row['logloss']:.6f} brier_without={row['brier']:.6f} auc_without={row['auc']:.4f} delta_ll_vs_full={row['delta_logloss_vs_full']:+.6f} interpretation={direction}")

        # Full production-like weekly replay with sequential OOS calibration.
        wf = _weekly_walk_forward(g, features, V14_MODEL_NAMES, start_season=seasons[1], log_func=log_func)
        wf_summary = {}
        if wf.empty:
            raise RuntimeError("V14 weekly walk-forward produced no predictions")
        for model_name in V14_MODEL_NAMES:
            d = wf.loc[wf.model.eq(model_name)].copy()
            met = _metrics(d.y.to_numpy(dtype=float), d.p.to_numpy(dtype=float)); bet = _bet_summary(d.y, d.p, 0.025)
            wf_summary[model_name] = {**met, **bet}
            log_func(f"[V14-WALK-FORWARD] model={model_name} weeks={d.week.nunique()} predictions={len(d)} auc={met['auc']:.4f} ll={met['logloss']:.6f} brier={met['brier']:.6f} ece={met['ece']:.4f} bets_025={bet['bets']} hit_025={bet['hit_rate']:.4f} roi_025={bet['roi']:.4f}")
            for sy, dd in d.groupby("season"):
                sm = _metrics(dd.y, dd.p); sb = _bet_summary(dd.y, dd.p, 0.025)
                log_func(f"[V14-WF-SEASON] model={model_name} season={int(sy)} predictions={len(dd)} auc={sm['auc']:.4f} ll={sm['logloss']:.6f} brier={sm['brier']:.6f} ece={sm['ece']:.4f} bets_025={sb['bets']} hit_025={sb['hit_rate']:.4f} roi_025={sb['roi']:.4f}")

        prior = (bundle or {}).get("weekly_walk_forward_v1") or {}
        old_sp = ((prior.get("summaries") or {}).get("spreads") or {}) if isinstance(prior, dict) else {}
        if old_sp:
            log_func(f"[V14-V13-WF-BENCHMARK] source=V13.5.7.2 predictions={int(old_sp.get('predictions',0) or 0)} auc={float(old_sp.get('auc',np.nan)):.4f} ll={float(old_sp.get('logloss',np.nan)):.6f} brier={float(old_sp.get('brier',np.nan)):.6f} bets_025={int(old_sp.get('bets',0) or 0)} hit_025={float(old_sp.get('hit_rate',np.nan)):.4f} roi_025={float(old_sp.get('roi',np.nan)):.4f}")

        # Fail-loudly execution contract. This is a diagnostics contract, not a
        # statistical promotion rule.
        enough_sf = all(int(overall[m].get("n", 0)) >= 500 for m in V14_MODEL_NAMES)
        enough_wf = all(int(wf_summary[m].get("n", 0)) >= 500 for m in V14_MODEL_NAMES)
        if not enough_sf or not enough_wf or len(ablation) < 3:
            raise RuntimeError(f"V14 diagnostics incomplete enough_sf={enough_sf} enough_wf={enough_wf} ablations={len(ablation)}")
        result = {"status": "PASS", "version": V14_CLEAN_ROOM_SOURCE_TAG, "leader": leader,
                  "features": features, "families": families, "overall": overall,
                  "weekly": wf_summary, "ablation": ablation, "production_authority": 0,
                  "systems_excluded": True, "first_eval_season": first_eval}
        log_func(f"[V14-CLEANROOM-CONTRACT] status=PASS leader={leader} features={len(features)} season_forward_models={len(V14_MODEL_NAMES)} weekly_models={len(V14_MODEL_NAMES)} ablations={len(ablation)} beats_market={beats_market} beats_v13={beats_v13} systems_excluded=TRUE production_authority=0")
        return result
    except Exception as e:
        log_func(f"[V14-CLEANROOM-CONTRACT] status=FAILED error={type(e).__name__}:{e} production_authority=0")
        if hard_fail:
            raise
        return {"status": "FAILED", "error": f"{type(e).__name__}:{e}", "production_authority": 0}
