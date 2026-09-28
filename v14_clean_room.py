"""NCAAF V14.1 STAT Residual Corrector research challenger.

Evidence-driven follow-up to V14 Clean Room V1.

The V14 V1 experiment showed that a 721-feature direct ATS classifier was worse than
V13 STAT, while family ablation identified a smaller set of useful information
families.  V14.1 therefore keeps the existing V13 season-forward STAT probability as
its anchor and asks a much narrower question:

    Can a compact, leakage-safe context model make a small OOS correction to STAT?

Research contract
-----------------
* Spread only.
* V13 STAT is the immutable probability anchor / control.
* Only feature families supported by V14 V1 ablation are eligible.
* Within-family reduction is UNSUPERVISED: pre-evaluation availability plus
  correlation pruning only. No target-based one-feature screen or greedy admission.
* Three fixed, deliberately constrained corrector model classes.
* Candidate probabilities are shrunk toward the V13 anchor in logit space.
  The correction weight is chosen only on a chronological inner holdout from the
  training period. If it does not improve BOTH log loss and Brier, alpha=0 and the
  candidate falls back to STAT.
* Calibration is monotonic-only. A constrained positive-slope Platt map is accepted
  only if it improves BOTH log loss and Brier on a chronological calibration gate;
  otherwise identity calibration is used. Calibration can compress probabilities but
  can never reverse their ordering.
* Primary evidence is season-forward / season-frozen. Weekly refitting is retained as
  a secondary challenger only.
* No ROI-based model selection. No systems, Miner, production write, threshold change,
  or serving authority.
"""
from __future__ import annotations

from typing import Dict, Iterable, List
import math

import numpy as np
import pandas as pd

V14_CLEAN_ROOM_SOURCE_TAG = "v14.1-stat-residual-corrector"
V141_MODEL_NAMES = ("L2_CORRECTOR", "ELASTIC_NET_CORRECTOR", "SHALLOW_HGB_CORRECTOR")
V141_SUPPORTED_FAMILIES = (
    "matchup",
    "opponent_adjusted",
    "passing",
    "down_conversion_proxy",
    "efficiency",
    "market_context",
)
V141_FAMILY_CAPS = {
    "matchup": 20,
    "opponent_adjusted": 14,
    "passing": 10,
    "down_conversion_proxy": 8,
    "efficiency": 8,
    "market_context": 8,
}
V141_CORR_LIMIT = 0.965
V141_ALPHA_GRID = (0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0)
V141_THRESHOLDS = (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05)
V141_EDGE_BINS = (0.0, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, np.inf)
V141_EDGE_LABELS = ("0-.01", ".01-.02", ".02-.025", ".025-.03", ".03-.04", ".04-.05", ".05+")


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


def _new_model(name: str):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier

    n = str(name).upper()
    if n == "L2_CORRECTOR":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.05, solver="lbfgs", max_iter=2500, tol=1e-4)),
        ])
    if n == "ELASTIC_NET_CORRECTOR":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.05, l1_ratio=0.20, solver="saga", max_iter=2500,
                                         tol=8e-4, random_state=42)),
        ])
    if n == "SHALLOW_HGB_CORRECTOR":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=False, keep_empty_features=True)),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.025, max_iter=100, max_leaf_nodes=5, max_depth=2,
                min_samples_leaf=55, l2_regularization=10.0, random_state=42,
            )),
        ])
    raise ValueError(f"Unknown V14.1 model {name}")


def _family(name: str, helper=None):
    c = str(name)
    if c.startswith("CR_Market_"):
        return "market_context"
    if c.startswith("Matchup_"):
        return "matchup"
    if helper is not None:
        try:
            return str(helper(c))
        except Exception:
            pass
    return "other"


def _add_market_context(g: pd.DataFrame):
    spread = _num(g.get("Consensus_Open_Spread"), g.index)
    total = _num(g.get("Consensus_Open_Total"), g.index)
    fair_ml = _num(g.get("Market_Open_H2H_Fair"), g.index)
    g["CR_Market_Spread"] = spread
    g["CR_Market_AbsSpread"] = spread.abs()
    g["CR_Market_ImpliedMargin"] = -spread
    g["CR_Market_Total"] = total
    g["CR_Market_H2H_Fair"] = fair_ml
    g["CR_Market_IsFavorite"] = np.where(spread.notna(), (spread < 0).astype(float), np.nan)
    g["CR_Market_IsDog"] = np.where(spread.notna(), (spread > 0).astype(float), np.nan)
    g["CR_Market_SpreadToTotal"] = spread.abs() / total.where(total.abs() > 1e-9)
    return g


def _v13_control_probs(g: pd.DataFrame, oof_margin, eval_seasons, empirical_prob_fn):
    """Reconstruct V13 season-forward cover probabilities from its OOF fair margin."""
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


def _select_compact_features(g: pd.DataFrame, candidates: Iterable[str], helper, pre_mask):
    """Unsupervised family reduction using only pre-evaluation availability/correlation."""
    raw = []
    for c in list(dict.fromkeys(candidates or [])):
        if c not in g.columns or c == "Context_Intercept":
            continue
        fam = _family(c, helper)
        if fam not in V141_SUPPORTED_FAMILIES:
            continue
        s = _num(g.loc[pre_mask, c])
        a = s.to_numpy(dtype=float, na_value=np.nan)
        finite = np.isfinite(a)
        n = int(finite.sum())
        if n < 100:
            continue
        sd = float(np.nanstd(a))
        if not np.isfinite(sd) or sd <= 1e-12:
            continue
        raw.append((c, fam, n, sd))

    selected = []
    selection_rows = []
    for fam in V141_SUPPORTED_FAMILIES:
        pool = [x for x in raw if x[1] == fam]
        # Availability first; lexical tie-break makes the selection deterministic.
        pool.sort(key=lambda x: (-x[2], x[0]))
        cap = int(V141_FAMILY_CAPS.get(fam, 8))
        chosen = []
        for c, _, n, sd in pool:
            if len(chosen) >= cap:
                break
            keep = True
            sc = _num(g.loc[pre_mask, c])
            for prior in chosen:
                sp = _num(g.loc[pre_mask, prior])
                pair = sc.notna() & sp.notna()
                if int(pair.sum()) < 100:
                    continue
                corr = sc[pair].corr(sp[pair])
                if pd.notna(corr) and abs(float(corr)) >= V141_CORR_LIMIT:
                    keep = False
                    break
            if keep:
                chosen.append(c)
                selected.append(c)
                selection_rows.append({"feature": c, "family": fam, "pre_n": n, "pre_sd": sd})
    return selected, selection_rows


def _prepare_dataset(games, candidate_cols, helper, oof_margin, empirical_prob_fn):
    if games is None or not isinstance(games, pd.DataFrame) or games.empty:
        raise RuntimeError("V14.1 received empty V13 research frame")
    g = games.copy()
    g["Game_Date"] = pd.to_datetime(g.get("Game_Date"), errors="coerce", utc=True)
    g["Season"] = _num(g.get("Season"), g.index)
    margin = _num(g.get("Actual_Margin"), g.index)
    spread = _num(g.get("Consensus_Open_Spread"), g.index)
    ats_margin = margin + spread
    g["V141_ATS_TARGET"] = np.where(ats_margin.notna() & ~np.isclose(ats_margin, 0.0, atol=1e-9),
                                       (ats_margin > 0).astype(float), np.nan)
    g = _add_market_context(g)

    seasons = sorted(int(x) for x in g["Season"].dropna().unique())
    if len(seasons) < 4:
        raise RuntimeError(f"V14.1 needs >=4 seasons, found {seasons}")
    first_eval = seasons[2]
    # Need the prior season's V13 OOF probability so the first corrector can train.
    base_p = _v13_control_probs(g, oof_margin, seasons[1:], empirical_prob_fn)
    g["V141_STAT_PROB"] = base_p
    g["V141_STAT_LOGIT"] = np.where(np.isfinite(base_p), _logit(np.where(np.isfinite(base_p), base_p, 0.5)), np.nan)
    g["V141_STAT_EDGE"] = np.where(np.isfinite(base_p), base_p - 0.5, np.nan)
    g["V141_STAT_ABS_EDGE"] = np.where(np.isfinite(base_p), np.abs(base_p - 0.5), np.nan)

    pre = g["Season"].lt(float(first_eval)) & g["V141_ATS_TARGET"].notna()
    all_candidates = list(dict.fromkeys(list(candidate_cols or []) + [c for c in g.columns if c.startswith("CR_Market_")]))
    compact, selection_rows = _select_compact_features(g, all_candidates, helper, pre)
    if len(compact) < 12:
        raise RuntimeError(f"V14.1 compact context too small: {len(compact)}")
    model_features = ["V141_STAT_LOGIT"] + compact
    families = {c: ("stat_anchor" if c == "V141_STAT_LOGIT" else _family(c, helper)) for c in model_features}
    return g, model_features, compact, families, selection_rows, seasons, first_eval


def _blend_to_anchor(base_p, candidate_p, alpha):
    base_p = np.asarray(base_p, dtype=float); candidate_p = np.asarray(candidate_p, dtype=float)
    out = np.full(len(base_p), np.nan)
    ok = np.isfinite(base_p) & np.isfinite(candidate_p)
    if ok.any():
        out[ok] = _sigmoid(_logit(base_p[ok]) + float(alpha) * (_logit(candidate_p[ok]) - _logit(base_p[ok])))
    return np.clip(out, 1e-5, 1 - 1e-5)


def _choose_alpha_chronological(model_name, g, train_idx, features, y, base_p):
    """Choose correction strength on the last 30% of training history only."""
    idx = np.asarray(train_idx, dtype=int)
    if len(idx) < 700:
        return 0.0, {"status": "INSUFFICIENT_INNER_HISTORY", "n": int(len(idx))}
    d = pd.to_datetime(g.loc[idx, "Game_Date"], errors="coerce", utc=True)
    order = idx[np.argsort(d.fillna(pd.Timestamp("1900-01-01", tz="UTC")).to_numpy())]
    cut = int(max(450, min(len(order) - 150, math.floor(len(order) * 0.70))))
    fit_idx = order[:cut]; val_idx = order[cut:]
    if len(fit_idx) < 400 or len(val_idx) < 125:
        return 0.0, {"status": "INSUFFICIENT_INNER_SPLIT", "n": int(len(idx))}
    model = _new_model(model_name)
    try:
        model.fit(g.loc[fit_idx, features], y[fit_idx].astype(int))
        cand = np.asarray(model.predict_proba(g.loc[val_idx, features])[:, 1], dtype=float)
    except Exception as e:
        return 0.0, {"status": f"INNER_MODEL_FAILED:{type(e).__name__}", "n": int(len(idx))}
    bm = _metrics(y[val_idx], base_p[val_idx])
    best_alpha = 0.0; best = bm
    for a in V141_ALPHA_GRID[1:]:
        p = _blend_to_anchor(base_p[val_idx], cand, a)
        m = _metrics(y[val_idx], p)
        if not np.isfinite(m["logloss"]) or not np.isfinite(m["brier"]):
            continue
        # Both proper scores must improve. The tolerance blocks microscopic noise.
        if (m["logloss"] <= bm["logloss"] - 1e-4 and m["brier"] <= bm["brier"] - 5e-5 and
                (best_alpha == 0.0 or (m["logloss"], m["brier"]) < (best["logloss"], best["brier"]))):
            best_alpha = float(a); best = m
    return best_alpha, {
        "status": "PASS" if best_alpha > 0 else "FALLBACK_TO_STAT",
        "n": int(len(val_idx)), "base_ll": bm["logloss"], "base_brier": bm["brier"],
        "chosen_ll": best["logloss"], "chosen_brier": best["brier"], "alpha": best_alpha,
    }


def _fit_positive_platt(y, p):
    """Constrained Platt: slope is always positive, so ordering cannot reverse."""
    from scipy.optimize import minimize
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    yy = y[ok].astype(float); xx = _logit(p[ok])
    if len(yy) < 250 or np.unique(yy).size < 2:
        return None
    def obj(theta):
        a, b = float(theta[0]), float(theta[1])
        q = np.clip(_sigmoid(a + b * xx), 1e-6, 1 - 1e-6)
        return float(-np.mean(yy * np.log(q) + (1.0 - yy) * np.log(1.0 - q)))
    try:
        res = minimize(obj, x0=np.array([0.0, 1.0]), method="L-BFGS-B",
                       bounds=[(-2.5, 2.5), (0.05, 3.0)])
        if not res.success or not np.all(np.isfinite(res.x)):
            return None
        return (float(res.x[0]), float(res.x[1]), int(len(yy)))
    except Exception:
        return None


def _apply_positive_platt(p, cal):
    p = np.asarray(p, dtype=float)
    if cal is None:
        return np.clip(p, 1e-5, 1 - 1e-5)
    a, b, _ = cal
    # b is constrained positive by construction.
    return np.clip(_sigmoid(a + b * _logit(p)), 1e-5, 1 - 1e-5)


def _gated_monotonic_calibrator(hist_y, hist_p, hist_dates):
    """Use monotonic calibration only when an earlier->later gate proves useful."""
    yy = np.asarray(hist_y, dtype=float); pp = np.asarray(hist_p, dtype=float)
    dd = pd.to_datetime(pd.Series(hist_dates), errors="coerce", utc=True)
    ok = np.isfinite(yy) & np.isfinite(pp) & dd.notna().to_numpy()
    yy = yy[ok]; pp = pp[ok]; dd = dd[ok].reset_index(drop=True)
    if len(yy) < 600:
        return None, {"status": "IDENTITY_INSUFFICIENT_HISTORY", "n": int(len(yy))}
    order = np.argsort(dd.to_numpy())
    yy = yy[order]; pp = pp[order]
    cut = int(math.floor(len(yy) * 0.70))
    if cut < 350 or len(yy) - cut < 150:
        return None, {"status": "IDENTITY_INSUFFICIENT_GATE", "n": int(len(yy))}
    gate_cal = _fit_positive_platt(yy[:cut], pp[:cut])
    if gate_cal is None:
        return None, {"status": "IDENTITY_FIT_FAILED", "n": int(len(yy))}
    base = _metrics(yy[cut:], pp[cut:])
    cand = _metrics(yy[cut:], _apply_positive_platt(pp[cut:], gate_cal))
    if (np.isfinite(cand["logloss"]) and np.isfinite(cand["brier"]) and
            cand["logloss"] <= base["logloss"] - 5e-5 and
            cand["brier"] <= base["brier"] - 2e-5):
        full = _fit_positive_platt(yy, pp)
        if full is not None:
            return full, {"status": "MONOTONIC_APPLIED", "n": int(len(yy)), "slope": full[1],
                          "gate_ll_gain": float(base["logloss"] - cand["logloss"]),
                          "gate_brier_gain": float(base["brier"] - cand["brier"])}
    return None, {"status": "IDENTITY_NO_GATE_GAIN", "n": int(len(yy)),
                  "gate_ll_gain": float(base["logloss"] - cand["logloss"]) if np.isfinite(cand["logloss"]) else np.nan,
                  "gate_brier_gain": float(base["brier"] - cand["brier"]) if np.isfinite(cand["brier"]) else np.nan}


def _season_forward_corrector(g, features, seasons, first_eval, model_name, log_func=lambda *_: None):
    y = _num(g["V141_ATS_TARGET"]).to_numpy(dtype=float)
    season = _num(g["Season"]).to_numpy(dtype=float)
    base_p = _num(g["V141_STAT_PROB"]).to_numpy(dtype=float)
    final_oof = np.full(len(g), np.nan); anchored_oof = np.full(len(g), np.nan); candidate_oof = np.full(len(g), np.nan)
    hist_y: List[float] = []; hist_p: List[float] = []; hist_dates: List[pd.Timestamp] = []
    rows = []
    for sy in [s for s in seasons if s >= first_eval]:
        tr = np.isfinite(season) & (season < float(sy)) & np.isfinite(y) & np.isfinite(base_p)
        va = np.isfinite(season) & (season == float(sy)) & np.isfinite(y) & np.isfinite(base_p)
        if int(tr.sum()) < 600 or int(va.sum()) < 75:
            continue
        tr_idx = np.where(tr)[0]; va_idx = np.where(va)[0]
        alpha, alpha_diag = _choose_alpha_chronological(model_name, g, tr_idx, features, y, base_p)
        model = _new_model(model_name)
        try:
            model.fit(g.loc[tr, features], y[tr].astype(int))
            cand = np.asarray(model.predict_proba(g.loc[va, features])[:, 1], dtype=float)
        except Exception as e:
            log_func(f"[V14.1-FOLD] model={model_name} season={sy} status=FAILED error={type(e).__name__}:{e}")
            continue
        anchored = _blend_to_anchor(base_p[va_idx], cand, alpha)
        cal, cal_diag = _gated_monotonic_calibrator(hist_y, hist_p, hist_dates)
        final = _apply_positive_platt(anchored, cal)
        candidate_oof[va_idx] = cand; anchored_oof[va_idx] = anchored; final_oof[va_idx] = final

        mb = _metrics(y[va_idx], base_p[va_idx]); mc = _metrics(y[va_idx], cand)
        ma = _metrics(y[va_idx], anchored); mf = _metrics(y[va_idx], final)
        rows.append({"season": int(sy), "n": int(len(va_idx)), "alpha": float(alpha),
                     "base": mb, "candidate": mc, "anchored": ma, "final": mf,
                     "alpha_diag": alpha_diag, "cal_diag": cal_diag})
        log_func(
            f"[V14.1-SEASON] model={model_name} season={sy} n={len(va_idx)} alpha={alpha:.2f} "
            f"alpha_status={alpha_diag.get('status')} cal_status={cal_diag.get('status')} "
            f"base_auc={mb['auc']:.4f} base_ll={mb['logloss']:.6f} base_brier={mb['brier']:.6f} "
            f"candidate_auc={mc['auc']:.4f} candidate_ll={mc['logloss']:.6f} "
            f"anchored_auc={ma['auc']:.4f} anchored_ll={ma['logloss']:.6f} anchored_brier={ma['brier']:.6f} "
            f"final_auc={mf['auc']:.4f} final_ll={mf['logloss']:.6f} final_brier={mf['brier']:.6f} "
            f"final_ece={mf['ece']:.4f} final_cal_slope={mf['cal_slope']:.3f}"
        )
        # Only completed OOS predictions can inform future calibration.
        ok = np.isfinite(anchored) & np.isfinite(y[va_idx])
        hist_y.extend(y[va_idx][ok].tolist())
        hist_p.extend(anchored[ok].tolist())
        hist_dates.extend(pd.to_datetime(g.loc[va_idx[ok], "Game_Date"], utc=True, errors="coerce").tolist())
    return final_oof, anchored_oof, candidate_oof, rows


def _bet_summary(y, p, threshold=0.025):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok].astype(int); p = p[ok]
    side = p >= 0.5; prob = np.maximum(p, 1.0 - p); edge = prob - 0.5
    won = np.where(side, y == 1, y == 0); bet = edge >= float(threshold)
    ret = np.where(won, 100.0 / 110.0, -1.0)
    return {"n": int(len(y)), "bets": int(bet.sum()), "wins": int(won[bet].sum()) if bet.any() else 0,
            "hit_rate": float(won[bet].mean()) if bet.any() else np.nan,
            "roi": float(ret[bet].mean()) if bet.any() else np.nan,
            "avg_edge": float(edge[bet].mean()) if bet.any() else np.nan}


def _emit_diagnostics(prefix, name, y, p, log_func):
    m = _metrics(y, p); b = _bet_summary(y, p, 0.025)
    log_func(f"[{prefix}] model={name} n={m['n']} auc={m['auc']:.4f} ll={m['logloss']:.6f} brier={m['brier']:.6f} ece={m['ece']:.4f} cal_intercept={m['cal_intercept']:.3f} cal_slope={m['cal_slope']:.3f} bets_025={b['bets']} hit_025={b['hit_rate']:.4f} roi_025={b['roi']:.4f}")
    return {**m, **b}


def _threshold_rows(y, p):
    rows = []
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p); yy = y[ok].astype(int); pp = p[ok]
    side = pp >= 0.5; won = np.where(side, yy == 1, yy == 0)
    edge = np.maximum(pp, 1.0 - pp) - 0.5; ret = np.where(won, 100.0 / 110.0, -1.0)
    for th in V141_THRESHOLDS:
        b = edge >= th
        rows.append({"threshold": th, "bets": int(b.sum()), "wins": int(won[b].sum()) if b.any() else 0,
                     "hit_rate": float(won[b].mean()) if b.any() else np.nan,
                     "roi": float(ret[b].mean()) if b.any() else np.nan})
    return rows


def _edge_rows(y, p):
    rows = []
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p); yy = y[ok].astype(int); pp = p[ok]
    side = pp >= 0.5; won = np.where(side, yy == 1, yy == 0)
    edge = np.maximum(pp, 1.0 - pp) - 0.5; ret = np.where(won, 100.0 / 110.0, -1.0)
    for lo, hi, lab in zip(V141_EDGE_BINS[:-1], V141_EDGE_BINS[1:], V141_EDGE_LABELS):
        m = (edge >= lo) & (edge < hi)
        rows.append({"bucket": lab, "n": int(m.sum()), "hit_rate": float(won[m].mean()) if m.any() else np.nan,
                     "roi": float(ret[m].mean()) if m.any() else np.nan,
                     "avg_edge": float(edge[m].mean()) if m.any() else np.nan})
    return rows


def _season_rows(name, y, p, base_p, season, eval_seasons):
    out = []
    for sy in eval_seasons:
        m = np.isfinite(season) & (season == float(sy)) & np.isfinite(y) & np.isfinite(p) & np.isfinite(base_p)
        if int(m.sum()) < 30:
            continue
        a = _metrics(y[m], p[m]); b = _metrics(y[m], base_p[m]); bet = _bet_summary(y[m], p[m], 0.025)
        out.append({"season": int(sy), "n": int(m.sum()), "model": a, "base": b, "bet": bet})
    return out


def _ablation(g, features, families, seasons, first_eval, model_name, full_ll, log_func):
    rows = []
    for fam in V141_SUPPORTED_FAMILIES:
        removed = [c for c in features if families.get(c) == fam]
        if not removed:
            continue
        use = [c for c in features if c not in set(removed)]
        p, _, _, _ = _season_forward_corrector(g, use, seasons, first_eval, model_name, log_func=lambda *_: None)
        y = _num(g["V141_ATS_TARGET"]).to_numpy(dtype=float); s = _num(g["Season"]).to_numpy(dtype=float)
        m = np.isfinite(y) & np.isfinite(p) & (s >= float(first_eval))
        met = _metrics(y[m], p[m])
        rows.append({"family": fam, "removed": len(removed), "n": met["n"], "auc": met["auc"],
                     "logloss": met["logloss"], "brier": met["brier"],
                     "delta_ll": float(met["logloss"] - full_ll) if np.isfinite(met["logloss"]) else np.nan})
    return rows


def _weekly_corrector(g, features, model_names, start_season, log_func):
    """Secondary challenger: corrector coefficients refit weekly; V13 anchor remains season-frozen."""
    dates = pd.to_datetime(g["Game_Date"], errors="coerce", utc=True)
    season = _num(g["Season"]).to_numpy(dtype=float)
    y = _num(g["V141_ATS_TARGET"]).to_numpy(dtype=float)
    base_p = _num(g["V141_STAT_PROB"]).to_numpy(dtype=float)
    rows = []
    histories = {m: {"y": [], "p": [], "date": []} for m in model_names}
    valid_dates = dates.notna() & np.isfinite(y) & np.isfinite(base_p) & (season >= float(start_season))
    if not valid_dates.any():
        return pd.DataFrame()
    # Monday weekly cutoffs; all games in the same weekly block are frozen together.
    week = dates.dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")
    for ws in sorted(pd.Series(week[valid_dates].dropna().unique()).tolist()):
        va = valid_dates.to_numpy() & (week.to_numpy() == ws)
        tr = dates.lt(ws).to_numpy() & np.isfinite(y) & np.isfinite(base_p)
        if int(tr.sum()) < 650 or int(va.sum()) < 5:
            continue
        tr_idx = np.where(tr)[0]; va_idx = np.where(va)[0]
        for name in model_names:
            alpha, adiag = _choose_alpha_chronological(name, g, tr_idx, features, y, base_p)
            model = _new_model(name)
            try:
                model.fit(g.loc[tr, features], y[tr].astype(int))
                cand = np.asarray(model.predict_proba(g.loc[va, features])[:, 1], dtype=float)
            except Exception:
                continue
            anchored = _blend_to_anchor(base_p[va_idx], cand, alpha)
            h = histories[name]
            cal, _ = _gated_monotonic_calibrator(h["y"], h["p"], h["date"])
            final = _apply_positive_platt(anchored, cal)
            for j, ix in enumerate(va_idx):
                rows.append({"model": name, "week": ws, "season": int(season[ix]), "idx": int(ix),
                             "y": float(y[ix]), "base_p": float(base_p[ix]), "p": float(final[j]),
                             "anchored_p": float(anchored[j]), "alpha": float(alpha)})
            ok = np.isfinite(anchored) & np.isfinite(y[va_idx])
            h["y"].extend(y[va_idx][ok].tolist()); h["p"].extend(anchored[ok].tolist())
            h["date"].extend(pd.to_datetime(g.loc[va_idx[ok], "Game_Date"], utc=True, errors="coerce").tolist())
    return pd.DataFrame(rows)


def run_v14_clean_room(*, dashboard_module, log_func=print, hard_fail=True):
    """Run V14.1 isolated STAT-residual research. Never mutates production state."""
    try:
        cache = getattr(dashboard_module, "_V1357_SPREAD_RESEARCH_CACHE", {})
        games = cache.get("games") if isinstance(cache, dict) else None
        oof_margin = cache.get("oof_margin") if isinstance(cache, dict) else None
        stat_cache = getattr(dashboard_module, "_NCAAF_STAT_TRAIN_CACHE", {})
        bundle = stat_cache.get("bundle") if isinstance(stat_cache, dict) else None
        if games is None or not isinstance(games, pd.DataFrame) or games.empty or oof_margin is None:
            raise RuntimeError("V13 STAT research cache unavailable; V14.1 must run after NCAAF Spread training")
        candidate_cols = list((bundle or {}).get("candidate_feature_cols") or [])
        if not candidate_cols:
            candidate_cols = [c for c in games.columns if str(c).startswith(("A_", "B_", "Diff_", "Matchup_", "Context_"))]
        helper = getattr(dashboard_module, "_ncaaf_stat_feature_family", None)
        empirical = getattr(dashboard_module, "_ncaaf_stat_empirical_prob_gt", None)
        if empirical is None:
            raise RuntimeError("V13 empirical probability helper unavailable")

        g, features, compact, families, selected_rows, seasons, first_eval = _prepare_dataset(
            games, candidate_cols, helper, oof_margin, empirical
        )
        y = _num(g["V141_ATS_TARGET"]).to_numpy(dtype=float)
        season = _num(g["Season"]).to_numpy(dtype=float)
        base_p = _num(g["V141_STAT_PROB"]).to_numpy(dtype=float)
        eval_seasons = [s for s in seasons if s >= first_eval]
        fam_counts = pd.Series([families[c] for c in features]).value_counts().to_dict()
        log_func(f"[V14.1-PREFLIGHT] status=PASS source_tag={V14_CLEAN_ROOM_SOURCE_TAG} rows={len(g)} seasons={seasons} first_eval={first_eval} candidate_features={len(candidate_cols)} compact_features={len(features)} family_counts={fam_counts} primary=SEASON_FROZEN production_authority=0")
        for fam in V141_SUPPORTED_FAMILIES:
            names = [r["feature"] for r in selected_rows if r["family"] == fam]
            log_func(f"[V14.1-FEATURE-FAMILY] family={fam} selected={len(names)} cap={V141_FAMILY_CAPS.get(fam)} features={names}")

        overall: Dict[str, dict] = {}
        preds: Dict[str, np.ndarray] = {}
        fold_rows: Dict[str, list] = {}
        eval_mask = np.isfinite(y) & np.isfinite(base_p) & (season >= float(first_eval))
        base_overall = _emit_diagnostics("V14.1-BASE-STAT", "V13_STAT_CONTROL", y[eval_mask], base_p[eval_mask], log_func)

        for name in V141_MODEL_NAMES:
            p, anchored, candidate, fr = _season_forward_corrector(g, features, seasons, first_eval, name, log_func=log_func)
            preds[name] = p; fold_rows[name] = fr
            m = eval_mask & np.isfinite(p)
            met = _emit_diagnostics("V14.1-OVERALL", name, y[m], p[m], log_func)
            met["ll_gain_vs_stat"] = float(base_overall["logloss"] - met["logloss"]) if np.isfinite(met["logloss"]) else np.nan
            met["brier_gain_vs_stat"] = float(base_overall["brier"] - met["brier"]) if np.isfinite(met["brier"]) else np.nan
            overall[name] = met
            log_func(f"[V14.1-INCREMENTAL] model={name} ll_gain_vs_stat={met['ll_gain_vs_stat']:+.6f} brier_gain_vs_stat={met['brier_gain_vs_stat']:+.6f} beats_stat_both={bool(met['ll_gain_vs_stat']>0 and met['brier_gain_vs_stat']>0)}")
            for row in _season_rows(name, y, p, base_p, season, eval_seasons):
                a=row['model']; b=row['base']; bt=row['bet']
                log_func(f"[V14.1-SEASON-SUMMARY] model={name} season={row['season']} n={row['n']} auc={a['auc']:.4f} ll={a['logloss']:.6f} brier={a['brier']:.6f} base_auc={b['auc']:.4f} base_ll={b['logloss']:.6f} base_brier={b['brier']:.6f} bets_025={bt['bets']} hit_025={bt['hit_rate']:.4f} roi_025={bt['roi']:.4f}")
            for r in _threshold_rows(y[m], p[m]):
                log_func(f"[V14.1-THRESHOLD] model={name} threshold={r['threshold']:.3f} bets={r['bets']} wins={r['wins']} hit_rate={r['hit_rate']:.4f} roi={r['roi']:.4f} diagnostic_only=TRUE")
            for r in _edge_rows(y[m], p[m]):
                log_func(f"[V14.1-EDGE-BUCKET] model={name} bucket={r['bucket']} n={r['n']} hit_rate={r['hit_rate']:.4f} roi={r['roi']:.4f} avg_edge={r['avg_edge']:.4f}")

        candidates = [m for m in V141_MODEL_NAMES if np.isfinite(overall[m].get("logloss", np.nan))]
        if not candidates:
            raise RuntimeError("No V14.1 corrector produced season-forward probabilities")
        leader = min(candidates, key=lambda m: (overall[m]["logloss"], overall[m]["brier"]))
        lm = overall[leader]
        season_better = 0; season_tested = 0
        for r in _season_rows(leader, y, preds[leader], base_p, season, eval_seasons):
            season_tested += 1
            if r["model"]["logloss"] < r["base"]["logloss"] and r["model"]["brier"] < r["base"]["brier"]:
                season_better += 1
        beats_stat = bool(lm["logloss"] < base_overall["logloss"] and lm["brier"] < base_overall["brier"])
        stable = bool(season_tested >= 2 and season_better >= math.ceil(season_tested / 2))
        log_func(f"[V14.1-LEADER] model={leader} selection=LOWEST_SEASON_FORWARD_LOGLOSS_THEN_BRIER ll={lm['logloss']:.6f} brier={lm['brier']:.6f} base_ll={base_overall['logloss']:.6f} base_brier={base_overall['brier']:.6f} beats_stat={beats_stat} seasons_better_both={season_better}/{season_tested} stability_pass={stable} production_authority=0")

        ablation = _ablation(g, features, families, seasons, first_eval, leader, lm["logloss"], log_func)
        for r in ablation:
            interp = "HURTS_WHEN_REMOVED" if np.isfinite(r["delta_ll"]) and r["delta_ll"] > 0 else "HELPS_WHEN_REMOVED"
            log_func(f"[V14.1-ABLATION] leader={leader} family={r['family']} removed_features={r['removed']} n={r['n']} auc_without={r['auc']:.4f} ll_without={r['logloss']:.6f} brier_without={r['brier']:.6f} delta_ll_vs_full={r['delta_ll']:+.6f} interpretation={interp}")

        # Secondary weekly-refit challenger. It is never the primary selection path.
        wf = _weekly_corrector(g, features, V141_MODEL_NAMES, start_season=seasons[1], log_func=log_func)
        wf_summary = {}
        if wf.empty:
            raise RuntimeError("V14.1 weekly challenger produced no predictions")
        for name in V141_MODEL_NAMES:
            d = wf.loc[wf.model.eq(name)].copy()
            met = _metrics(d.y, d.p); base = _metrics(d.y, d.base_p); bet = _bet_summary(d.y, d.p, 0.025)
            wf_summary[name] = {**met, **bet, "base_ll": base["logloss"], "base_brier": base["brier"]}
            log_func(f"[V14.1-WEEKLY-CHALLENGER] model={name} weeks={d.week.nunique()} predictions={len(d)} auc={met['auc']:.4f} ll={met['logloss']:.6f} brier={met['brier']:.6f} base_auc={base['auc']:.4f} base_ll={base['logloss']:.6f} base_brier={base['brier']:.6f} bets_025={bet['bets']} hit_025={bet['hit_rate']:.4f} roi_025={bet['roi']:.4f} primary=FALSE")

        enough_sf = all(int(overall[m].get("n", 0)) >= 500 for m in V141_MODEL_NAMES)
        enough_wf = all(int(wf_summary[m].get("n", 0)) >= 500 for m in V141_MODEL_NAMES)
        monotonic_ok = True
        # Any applied calibrator must have a positive slope; identity gates are allowed.
        for name, rows in fold_rows.items():
            for r in rows:
                st = str(r.get("cal_diag", {}).get("status", ""))
                sl = r.get("cal_diag", {}).get("slope", np.nan)
                if st == "MONOTONIC_APPLIED" and (not np.isfinite(sl) or float(sl) <= 0):
                    monotonic_ok = False
        if not enough_sf or not enough_wf or len(ablation) < 4 or not monotonic_ok:
            raise RuntimeError(f"V14.1 diagnostics incomplete enough_sf={enough_sf} enough_wf={enough_wf} ablations={len(ablation)} monotonic_ok={monotonic_ok}")

        result = {"status": "PASS", "version": V14_CLEAN_ROOM_SOURCE_TAG, "leader": leader,
                  "features": features, "families": families, "overall": overall,
                  "base": base_overall, "weekly": wf_summary, "ablation": ablation,
                  "beats_stat": beats_stat, "stability_pass": stable, "production_authority": 0}
        log_func(f"[V14.1-CONTRACT] status=PASS leader={leader} compact_features={len(features)} season_forward_models={len(V141_MODEL_NAMES)} weekly_challengers={len(V141_MODEL_NAMES)} ablations={len(ablation)} beats_stat={beats_stat} stability_pass={stable} calibration=MONOTONIC_ONLY primary=SEASON_FROZEN systems_excluded=TRUE production_authority=0")
        return result
    except Exception as e:
        log_func(f"[V14.1-CONTRACT] status=FAILED error={type(e).__name__}:{e} production_authority=0")
        if hard_fail:
            raise
        return {"status": "FAILED", "error": f"{type(e).__name__}:{e}", "production_authority": 0}
