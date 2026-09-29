"""STAT_COMBINATION_V1 — controlled NCAAF market-error ensemble research.

Purpose
-------
Build an interpretable statistical challenger that asks where the opening spread
is wrong.  It does NOT replace production V13 STAT and has zero production
authority.

Design principles
-----------------
* Predict point-spread market error: Actual_Margin - Market_Open_Margin.
* Use season-forward OOF only; no final-refit history is used for evaluation.
* Build separate football-family experts (efficiency, passing, rushing, etc.).
* Family models use target-free availability/correlation pruning; no raw feature
  must win a standalone target screen before it can contribute to its family.
* Combine at most three family experts. Combination selection uses only prior
  season-forward OOF predictions and equal-weight averaging, limiting meta-fit
  overfitting and preserving interpretability.
* Compare every challenger on the exact same physical games against BOTH the
  market residual baseline (0) and the existing V13 residual baseline.
* 2026 is confirmation only.  No result from 2026 is used to select the 2026
  combination.
* Big Al / Pathi are evaluated as independent confirmation/conflict overlays;
  they are never fed into the fair-value statistical model.
* Report a Market Error Atlas and closing-line movement audit when data exists.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple
import math
import re

import numpy as np
import pandas as pd

SCV1_SOURCE_TAG = "stat-combination-v1-market-error-small-ensemble"
SCV1_DISCOVERY_START = 2023
SCV1_CONFIRMATION_SEASON = 2026
SCV1_MAX_FAMILY_FEATURES = 36
SCV1_MIN_FEATURE_COVERAGE = 0.45
SCV1_CORR_CUTOFF = 0.965
SCV1_MIN_TRAIN_ROWS = 500
SCV1_MIN_VALID_ROWS = 100
SCV1_MAX_COMBO_SIZE = 3
SCV1_MIN_COMBO_PRIOR_ROWS = 250
SCV1_ALLOWED_FAMILIES = (
    "efficiency",
    "passing",
    "rushing",
    "opponent_adjusted",
    "down_conversion_proxy",
    "turnovers",
    "tempo_play_mix",
    "matchup",
    "context",
)


def _num(x, index=None):
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce")
    if index is None:
        return pd.Series(pd.to_numeric(x, errors="coerce"))
    return pd.to_numeric(pd.Series(x, index=index), errors="coerce")


def _roi(hit: float) -> float:
    if not np.isfinite(hit):
        return np.nan
    return float(hit * (100.0 / 110.0) - (1.0 - hit))


def _physical_key(g: pd.DataFrame) -> pd.Series:
    if "Source_Game_ID" in g.columns:
        k = g["Source_Game_ID"].astype(str).str.strip().str.lower()
        if k.ne("").sum() >= int(0.9 * len(g)):
            return k
    season = _num(g.get("Season"), g.index).astype("Int64").astype(str)
    date = pd.to_datetime(g.get("Game_Date"), errors="coerce", utc=True).dt.strftime("%Y-%m-%d").fillna("")
    team = g.get("Team_Norm", pd.Series("", index=g.index)).astype(str).str.lower().str.strip()
    opp = g.get("Opponent_Norm", pd.Series("", index=g.index)).astype(str).str.lower().str.strip()
    return season + "|" + date + "|" + team + "|" + opp


def _new_ridge(alpha: float = 24.0):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=float(alpha))),
    ])


def _target_free_prune(g: pd.DataFrame, cols: List[str], train_mask: np.ndarray) -> List[str]:
    """Availability/variance/correlation pruning only. No target is consulted."""
    tr = np.asarray(train_mask, dtype=bool)
    usable = []
    for c in cols:
        if c not in g.columns:
            continue
        s = pd.to_numeric(g.loc[tr, c], errors="coerce")
        cov = float(s.notna().mean()) if len(s) else 0.0
        if cov < SCV1_MIN_FEATURE_COVERAGE:
            continue
        if int(s.nunique(dropna=True)) < 5:
            continue
        sd = float(s.std(skipna=True))
        if not np.isfinite(sd) or sd <= 1e-10:
            continue
        usable.append((cov, c))
    usable.sort(key=lambda z: (-z[0], z[1]))
    # Keep correlation calculation bounded before final cap.
    pre = [c for _, c in usable[: max(SCV1_MAX_FAMILY_FEATURES * 3, SCV1_MAX_FAMILY_FEATURES)]]
    kept: List[str] = []
    if not pre:
        return kept
    x = g.loc[tr, pre].apply(pd.to_numeric, errors="coerce")
    for c in pre:
        if len(kept) >= SCV1_MAX_FAMILY_FEATURES:
            break
        if not kept:
            kept.append(c)
            continue
        sc = x[c]
        reject = False
        for k in kept:
            pair = pd.concat([sc, x[k]], axis=1).dropna()
            if len(pair) < 100:
                continue
            corr = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
            if np.isfinite(corr) and abs(corr) >= SCV1_CORR_CUTOFF:
                reject = True
                break
        if not reject:
            kept.append(c)
    return kept


def _metrics(target: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    t = np.asarray(target, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.asarray(mask, dtype=bool) & np.isfinite(t) & np.isfinite(p)
    if int(m.sum()) == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "ats_hit": np.nan, "roi": np.nan,
                "signed_market_error": np.nan, "corr": np.nan}
    tt, pp = t[m], p[m]
    err = tt - pp
    nz = ~np.isclose(tt, 0.0, atol=1e-9) & ~np.isclose(pp, 0.0, atol=1e-12)
    if nz.any():
        hit = float(np.mean(np.sign(pp[nz]) == np.sign(tt[nz])))
        signed = float(np.mean(np.sign(pp[nz]) * tt[nz]))
    else:
        hit = signed = np.nan
    corr = float(np.corrcoef(tt, pp)[0, 1]) if len(tt) >= 3 and np.nanstd(tt) > 0 and np.nanstd(pp) > 0 else np.nan
    return {
        "n": int(m.sum()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "ats_hit": hit,
        "roi": _roi(hit),
        "signed_market_error": signed,
        "corr": corr,
    }


def _same_row_compare(target, candidate, v13_pred, mask) -> dict:
    t = np.asarray(target, float); c = np.asarray(candidate, float); v = np.asarray(v13_pred, float)
    m = np.asarray(mask, bool) & np.isfinite(t) & np.isfinite(c) & np.isfinite(v)
    cm = _metrics(t, c, m)
    vm = _metrics(t, v, m)
    mm = _metrics(t, np.zeros(len(t), dtype=float), m)
    return {
        "n": cm["n"], "candidate": cm, "v13": vm, "market": mm,
        "rmse_gain_vs_market": float(mm["rmse"] - cm["rmse"]) if cm["n"] else np.nan,
        "rmse_gain_vs_v13": float(vm["rmse"] - cm["rmse"]) if cm["n"] else np.nan,
        "mae_gain_vs_market": float(mm["mae"] - cm["mae"]) if cm["n"] else np.nan,
        "mae_gain_vs_v13": float(vm["mae"] - cm["mae"]) if cm["n"] else np.nan,
    }


def _family_columns(g: pd.DataFrame, candidate_cols: List[str], family_fn) -> Dict[str, List[str]]:
    out = {f: [] for f in SCV1_ALLOWED_FAMILIES}
    for c in candidate_cols:
        if c not in g.columns:
            continue
        fam = str(family_fn(c))
        if fam in out:
            out[fam].append(c)
    return out


def _family_oof(g: pd.DataFrame, family_cols: Dict[str, List[str]], target: np.ndarray,
                season: np.ndarray, log_func=print) -> Tuple[Dict[str, np.ndarray], dict]:
    preds = {fam: np.full(len(g), np.nan, dtype=float) for fam in family_cols}
    fold_features: Dict[str, dict] = {fam: {} for fam in family_cols}
    years = sorted(int(x) for x in pd.Series(season).dropna().unique())
    for sy in years:
        if sy < SCV1_DISCOVERY_START:
            continue
        tr = np.isfinite(season) & (season < float(sy)) & np.isfinite(target)
        va = np.isfinite(season) & (season == float(sy)) & np.isfinite(target)
        if int(tr.sum()) < SCV1_MIN_TRAIN_ROWS or int(va.sum()) < SCV1_MIN_VALID_ROWS:
            continue
        for fam, raw_cols in family_cols.items():
            cols = _target_free_prune(g, raw_cols, tr)
            fold_features[fam][sy] = list(cols)
            if not cols:
                continue
            model = _new_ridge(24.0)
            try:
                model.fit(g.loc[tr, cols], target[tr])
                pr = np.asarray(model.predict(g.loc[va, cols]), dtype=float)
            except Exception as e:
                log_func(f"[STAT-COMBO-FAMILY-FOLD] family={fam} season={sy} status=FAILED error={type(e).__name__}:{e}")
                continue
            preds[fam][np.where(va)[0]] = pr
            log_func(
                f"[STAT-COMBO-FAMILY-FOLD] family={fam} season={sy} train_n={int(tr.sum())} valid_n={int(va.sum())} "
                f"raw_features={len(raw_cols)} retained_features={len(cols)} target_free_pruning=TRUE "
                f"sample_features={cols[:5]}"
            )
    return preds, fold_features


def _dynamic_strength_oof(dashboard_module, g: pd.DataFrame) -> np.ndarray:
    out = np.full(len(g), np.nan, dtype=float)
    cache = getattr(dashboard_module, "_V133112_NATIVE_BRAIN_DIAG_CACHE", {})
    core = cache.get("CORE") if isinstance(cache, dict) else None
    if core is None or not isinstance(core, pd.DataFrame) or core.empty:
        return out
    d = core.copy()
    if "__role" in d.columns:
        d = d.loc[d["__role"].astype(str).str.upper().eq("HOME")].copy()
    if "CoreV4_OOF_Fair_Margin" not in d.columns or "Source_Game_ID" not in d.columns:
        return out
    d["__k"] = d["Source_Game_ID"].astype(str).str.strip().str.lower()
    d = d.drop_duplicates("__k", keep="last")
    mp = pd.Series(pd.to_numeric(d["CoreV4_OOF_Fair_Margin"], errors="coerce").to_numpy(), index=d["__k"]).to_dict()
    k = _physical_key(g)
    fair = k.map(mp).to_numpy(dtype=float)
    market = _num(g.get("Market_Open_Margin"), g.index).to_numpy(dtype=float)
    ok = np.isfinite(fair) & np.isfinite(market)
    out[ok] = fair[ok] - market[ok]
    return out


def _combo_pred(expert_preds: Dict[str, np.ndarray], names: Tuple[str, ...]) -> np.ndarray:
    a = np.column_stack([np.asarray(expert_preds[n], dtype=float) for n in names])
    ok = np.all(np.isfinite(a), axis=1)
    out = np.full(len(a), np.nan, dtype=float)
    out[ok] = np.mean(a[ok], axis=1)
    return out


def _combo_candidates(expert_preds: Dict[str, np.ndarray], target: np.ndarray, v13_pred: np.ndarray,
                      season: np.ndarray, target_season: int) -> List[dict]:
    prior = np.isfinite(season) & (season >= SCV1_DISCOVERY_START) & (season < float(target_season))
    prior_years = sorted(int(x) for x in pd.Series(season[prior]).dropna().unique())
    names = sorted(expert_preds)
    rows = []
    for size in range(1, min(SCV1_MAX_COMBO_SIZE, len(names)) + 1):
        for combo in combinations(names, size):
            pred = _combo_pred(expert_preds, combo)
            comp = _same_row_compare(target, pred, v13_pred, prior)
            if int(comp["n"]) < SCV1_MIN_COMBO_PRIOR_ROWS:
                continue
            season_gains = []
            for sy in prior_years:
                z = _same_row_compare(target, pred, v13_pred, prior & (season == float(sy)))
                if z["n"] >= 50:
                    season_gains.append((sy, z["rmse_gain_vs_market"], z["rmse_gain_vs_v13"], z["n"]))
            pos_market = int(sum(np.isfinite(x[1]) and x[1] > 0 for x in season_gains))
            pos_v13 = int(sum(np.isfinite(x[2]) and x[2] > 0 for x in season_gains))
            stable_market = bool(season_gains and pos_market >= math.ceil(len(season_gains) / 2))
            beats_market = bool(np.isfinite(comp["rmse_gain_vs_market"]) and comp["rmse_gain_vs_market"] > 0 and comp["mae_gain_vs_market"] >= -0.01)
            beats_v13 = bool(np.isfinite(comp["rmse_gain_vs_v13"]) and comp["rmse_gain_vs_v13"] > 0 and comp["mae_gain_vs_v13"] >= -0.01)
            rows.append({
                "combo": combo, "pred": pred, "prior": comp, "season_gains": season_gains,
                "positive_market_seasons": pos_market, "positive_v13_seasons": pos_v13,
                "stable_market": stable_market, "beats_market": beats_market, "beats_v13": beats_v13,
                "beats_both": bool(beats_market and beats_v13 and stable_market),
            })
    rows.sort(key=lambda r: (
        0 if r["beats_both"] else 1,
        -float(r["prior"]["rmse_gain_vs_market"] if np.isfinite(r["prior"]["rmse_gain_vs_market"]) else -999),
        -float(r["prior"]["rmse_gain_vs_v13"] if np.isfinite(r["prior"]["rmse_gain_vs_v13"]) else -999),
        len(r["combo"]),
        r["combo"],
    ))
    return rows


def _side_for_prediction(g: pd.DataFrame, pred: np.ndarray) -> dict:
    p = np.asarray(pred, dtype=float)
    home_side = p >= 0.0
    home = g.get("Team_Norm", pd.Series("", index=g.index)).astype(str).to_numpy(dtype=object)
    away = g.get("Opponent_Norm", pd.Series("", index=g.index)).astype(str).to_numpy(dtype=object)
    spread = _num(g.get("Consensus_Open_Spread"), g.index).to_numpy(dtype=float)
    return {
        "stat_home": home_side,
        "selected_team": np.where(home_side, home, away),
        "selected_opp": np.where(home_side, away, home),
        "selected_spread": np.where(home_side, spread, -spread),
        "edge_points": p,
    }


def _atlas_row(target, pred, mask) -> dict:
    met = _metrics(target, pred, mask)
    return met


def _emit_atlas(g: pd.DataFrame, target: np.ndarray, pred: np.ndarray, season: np.ndarray,
                dashboard_module, log_func=print):
    discovery = np.isfinite(season) & (season >= SCV1_DISCOVERY_START) & (season <= 2025)
    confirm = season == float(SCV1_CONFIRMATION_SEASON)
    abs_pred = np.abs(np.asarray(pred, dtype=float))
    spread = np.abs(_num(g.get("Consensus_Open_Spread"), g.index).to_numpy(dtype=float))
    regimes = [
        ("PRED_LT0P5", abs_pred < 0.5),
        ("PRED_0P5_TO1", (abs_pred >= 0.5) & (abs_pred < 1.0)),
        ("PRED_1_TO2", (abs_pred >= 1.0) & (abs_pred < 2.0)),
        ("PRED_2_PLUS", abs_pred >= 2.0),
        ("SPREAD_0_TO3", spread <= 3.0),
        ("SPREAD_GT3_TO7", (spread > 3.0) & (spread <= 7.0)),
        ("SPREAD_GT7_TO14", (spread > 7.0) & (spread <= 14.0)),
        ("SPREAD_GT14", spread > 14.0),
    ]
    # Add independent system agreement relative to the combination side.
    try:
        import v14_stat_reliability as rel
        sys_hist = getattr(dashboard_module, "_V143_SYSTEM_HISTORY_CACHE", {})
        sys_masks = rel._system_occurrence_masks(g, _side_for_prediction(g, pred), sys_hist, log_func=lambda *_: None)
        for n in ("BIGAL_AGREE", "BIGAL_CONFLICT", "SYSTEM_ANY_AGREE", "SYSTEM_ANY_CONFLICT"):
            if n in sys_masks:
                regimes.append((n, np.asarray(sys_masks[n], dtype=bool)))
    except Exception as e:
        log_func(f"[STAT-COMBO-SYSTEM-OVERLAY] status=UNAVAILABLE error={type(e).__name__}:{e}")

    for name, rm in regimes:
        for label, base in (("DISCOVERY_2023_2025", discovery), ("CONFIRM_2026", confirm)):
            m = np.asarray(rm, bool) & base
            met = _atlas_row(target, pred, m)
            if met["n"] < 5:
                continue
            log_func(
                f"[STAT-COMBO-MARKET-ERROR-ATLAS] regime={name} sample={label} n={met['n']} "
                f"ats_hit={met['ats_hit']:.4f} roi={met['roi']:+.4f} avg_signed_market_error_pts={met['signed_market_error']:+.3f} "
                f"rmse={met['rmse']:.4f}"
            )


def _emit_clv(g: pd.DataFrame, target: np.ndarray, pred: np.ndarray, season: np.ndarray, log_func=print):
    close_col = None
    for c in ("Consensus_Close_Spread_Audit", "Closing_Spread_For_Team", "Closing_Spread", "Close_Spread"):
        if c in g.columns and int(pd.to_numeric(g[c], errors="coerce").notna().sum()) >= 200:
            close_col = c
            break
    if close_col is None:
        log_func("[STAT-COMBO-CLV] status=UNAVAILABLE reason=NO_CLOSE_SPREAD_WITH_200_ROWS")
        return
    open_sp = _num(g.get("Consensus_Open_Spread"), g.index).to_numpy(dtype=float)
    close_sp = _num(g.get(close_col), g.index).to_numpy(dtype=float)
    open_margin = -open_sp; close_margin = -close_sp
    p = np.asarray(pred, dtype=float)
    move_toward = np.sign(p) * (close_margin - open_margin)
    for sy_label, base in (("DISCOVERY_2023_2025", (season >= 2023) & (season <= 2025)), ("CONFIRM_2026", season == 2026)):
        valid = np.asarray(base, bool) & np.isfinite(move_toward) & np.isfinite(p) & ~np.isclose(p, 0.0, atol=1e-12)
        if int(valid.sum()) < 20:
            continue
        mv = move_toward[valid]
        log_func(
            f"[STAT-COMBO-CLV] sample={sy_label} source={close_col} n={int(valid.sum())} "
            f"mean_move_toward_model={float(np.mean(mv)):+.3f} median_move_toward_model={float(np.median(mv)):+.3f} "
            f"pct_market_moved_toward_model={float(np.mean(mv>0)):.4f} descriptive_only=TRUE"
        )
        for nm, cond in (("MARKET_AGREES", mv > 0.25), ("MARKET_STATIC", np.abs(mv) <= 0.25), ("MARKET_REJECTS", mv < -0.25)):
            idx = np.where(valid)[0][cond]
            m = np.zeros(len(g), dtype=bool); m[idx] = True
            met = _metrics(target, p, m)
            if met["n"] >= 10:
                log_func(
                    f"[STAT-COMBO-CLV-BUCKET] sample={sy_label} state={nm} n={met['n']} "
                    f"ats_hit={met['ats_hit']:.4f} roi={met['roi']:+.4f} avg_signed_market_error_pts={met['signed_market_error']:+.3f}"
                )


def run_stat_combination_v1(*, dashboard_module, log_func=print, hard_fail=True):
    try:
        cache = getattr(dashboard_module, "_V1357_SPREAD_RESEARCH_CACHE", {})
        g = cache.get("games") if isinstance(cache, dict) else None
        oof_margin = cache.get("oof_margin") if isinstance(cache, dict) else None
        candidate_cols = cache.get("candidate_feature_cols") if isinstance(cache, dict) else None
        if g is None or not isinstance(g, pd.DataFrame) or g.empty or oof_margin is None:
            raise RuntimeError("STAT_COMBINATION_V1 requires V13 season-forward research cache")
        g = g.copy()
        g["Season"] = _num(g.get("Season"), g.index)
        season = g["Season"].to_numpy(dtype=float)
        target = _num(g.get("Market_Error_Margin"), g.index).to_numpy(dtype=float)
        market_margin = _num(g.get("Market_Open_Margin"), g.index).to_numpy(dtype=float)
        if candidate_cols is None:
            prefixes = ("A_Raw", "B_Raw", "Diff_Raw", "A_State_", "B_State_", "Diff_State_", "A_Recent3_", "B_Recent3_", "Diff_Recent3_", "Matchup_", "Context_")
            candidate_cols = [c for c in g.columns if str(c).startswith(prefixes)]
        candidate_cols = [c for c in list(dict.fromkeys(candidate_cols)) if c in g.columns]
        family_fn = getattr(dashboard_module, "_ncaaf_stat_feature_family", None)
        if family_fn is None:
            raise RuntimeError("missing _ncaaf_stat_feature_family")
        fam_cols = _family_columns(g, candidate_cols, family_fn)
        fam_cols = {k: v for k, v in fam_cols.items() if len(v) >= 1}
        if len(fam_cols) < 5:
            raise RuntimeError(f"too few statistical families={list(fam_cols)}")
        key = _physical_key(g)
        valid_key = key.ne("") & key.ne("nan")
        dup = int(key[valid_key].duplicated().sum())
        if dup:
            raise RuntimeError(f"physical game duplication rows={dup}")

        v13_pred = np.asarray(oof_margin, dtype=float) - market_margin
        log_func(
            f"[STAT-COMBO-PREFLIGHT] status=PASS source_tag={SCV1_SOURCE_TAG} rows={len(g)} "
            f"candidate_features={len(candidate_cols)} families={{{', '.join(f'{k}:{len(v)}' for k,v in fam_cols.items())}}} "
            f"physical_game_duplicates=0 target=MARKET_ERROR_MARGIN market_baseline=ZERO V13_role=BENCHMARK_NOT_PROTECTED production_authority=0"
        )

        family_preds, fold_features = _family_oof(g, fam_cols, target, season, log_func=log_func)
        dyn = _dynamic_strength_oof(dashboard_module, g)
        if int(np.isfinite(dyn).sum()) >= 500:
            family_preds["dynamic_strength"] = dyn
            log_func(f"[STAT-COMBO-DYNAMIC-STRENGTH] status=READY oof_rows={int(np.isfinite(dyn).sum())} source=CORE_V4_MARKET_BLIND_SEASON_FORWARD")
        else:
            log_func(f"[STAT-COMBO-DYNAMIC-STRENGTH] status=UNAVAILABLE oof_rows={int(np.isfinite(dyn).sum())}")

        # Family-level diagnostics: discovery and current-season confirmation.
        discovery = np.isfinite(season) & (season >= 2023) & (season <= 2025)
        confirm = season == float(SCV1_CONFIRMATION_SEASON)
        for fam, pred in sorted(family_preds.items()):
            d = _same_row_compare(target, pred, v13_pred, discovery)
            c = _same_row_compare(target, pred, v13_pred, confirm)
            log_func(
                f"[STAT-COMBO-FAMILY] family={fam} discovery_n={d['n']} rmse={d['candidate']['rmse']:.4f} "
                f"rmse_gain_vs_market={d['rmse_gain_vs_market']:+.4f} rmse_gain_vs_v13={d['rmse_gain_vs_v13']:+.4f} "
                f"mae_gain_vs_market={d['mae_gain_vs_market']:+.4f} ats_hit={d['candidate']['ats_hit']:.4f} "
                f"confirm_2026_n={c['n']} confirm_rmse_gain_vs_market={c['rmse_gain_vs_market']:+.4f} "
                f"confirm_rmse_gain_vs_v13={c['rmse_gain_vs_v13']:+.4f} confirm_ats_hit={c['candidate']['ats_hit']:.4f}"
            )

        # Chronological combination replay. The 2026 row is the primary challenger test.
        primary = None
        for target_season in (2025, 2026):
            cand = _combo_candidates(family_preds, target, v13_pred, season, target_season)
            if not cand:
                log_func(f"[STAT-COMBO-SELECTION] target_season={target_season} status=NO_CANDIDATES")
                continue
            leader = cand[0]
            test = _same_row_compare(target, leader["pred"], v13_pred, season == float(target_season))
            status = "CHALLENGER_PASS" if leader["beats_both"] else ("MARKET_ONLY_PASS" if leader["beats_market"] else "NO_PRIOR_SKILL")
            log_func(
                f"[STAT-COMBO-SELECTION] target_season={target_season} prior_seasons={[y for y in sorted(set(int(x) for x in pd.Series(season).dropna().unique())) if 2023 <= y < target_season]} "
                f"combo={'+'.join(leader['combo'])} size={len(leader['combo'])} prior_n={leader['prior']['n']} "
                f"prior_rmse_gain_vs_market={leader['prior']['rmse_gain_vs_market']:+.4f} prior_rmse_gain_vs_v13={leader['prior']['rmse_gain_vs_v13']:+.4f} "
                f"positive_market_seasons={leader['positive_market_seasons']} positive_v13_seasons={leader['positive_v13_seasons']} "
                f"status={status} target_n={test['n']} target_rmse_gain_vs_market={test['rmse_gain_vs_market']:+.4f} "
                f"target_rmse_gain_vs_v13={test['rmse_gain_vs_v13']:+.4f} target_mae_gain_vs_market={test['mae_gain_vs_market']:+.4f} "
                f"target_ats_hit={test['candidate']['ats_hit']:.4f} target_roi={test['candidate']['roi']:+.4f} chronological=TRUE production_authority=0"
            )
            # Top five are useful for understanding complementarity, without allowing an unbounded search dump.
            for rank, rec in enumerate(cand[:5], start=1):
                log_func(
                    f"[STAT-COMBO-TOP] target_season={target_season} rank={rank} combo={'+'.join(rec['combo'])} "
                    f"prior_n={rec['prior']['n']} rmse_gain_market={rec['prior']['rmse_gain_vs_market']:+.4f} "
                    f"rmse_gain_v13={rec['prior']['rmse_gain_vs_v13']:+.4f} beats_both={rec['beats_both']} stable_market={rec['stable_market']}"
                )
            if target_season == 2026:
                primary = {"leader": leader, "test": test, "status": status}

        if primary is None:
            raise RuntimeError("no 2026 chronological combination challenger was produced")

        leader = primary["leader"]
        _emit_atlas(g, target, leader["pred"], season, dashboard_module, log_func=log_func)
        _emit_clv(g, target, leader["pred"], season, log_func=log_func)

        # Hard comparison contract on the exact 2026 physical games.
        tmask = (season == 2026) & np.isfinite(target) & np.isfinite(leader["pred"]) & np.isfinite(v13_pred)
        n = int(tmask.sum()); u = int(_physical_key(g).loc[tmask].nunique())
        if n != u:
            raise RuntimeError(f"2026 comparison not one physical game per row n={n} unique={u}")
        log_func(
            f"[STAT-COMBO-COMPARISON-CONTRACT] status=PASS target_season=2026 same_exact_rows=TRUE rows={n} unique_physical_games={u} "
            f"market_baseline=ZERO_RESIDUAL v13_baseline=CURRENT_V13_RESIDUAL no_cross_row_comparison=TRUE"
        )
        log_func(
            f"[STAT-COMBO-CONTRACT] status=PASS source_tag={SCV1_SOURCE_TAG} primary_combo={'+'.join(leader['combo'])} "
            f"primary_status={primary['status']} target_2026_n={primary['test']['n']} "
            f"target_rmse_gain_vs_market={primary['test']['rmse_gain_vs_market']:+.4f} "
            f"target_rmse_gain_vs_v13={primary['test']['rmse_gain_vs_v13']:+.4f} production_authority=0"
        )
        return {"status": "PASS", "source_tag": SCV1_SOURCE_TAG, "primary": primary, "production_authority": 0}
    except Exception as e:
        log_func(f"[STAT-COMBO-CONTRACT] status=FAILED error={type(e).__name__}:{e} production_authority=0")
        if hard_fail:
            raise
        return {"status": "FAILED", "error": f"{type(e).__name__}:{e}", "production_authority": 0}
