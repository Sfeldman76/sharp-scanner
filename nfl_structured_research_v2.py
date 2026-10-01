"""NFL V1.9.4 null-safe structured challenger research.

Research-only.  This module does four things without touching 2026:

1. Keeps the incumbent independent CORE OOF predictions as the benchmark.
2. Builds new market-independent CORE challengers from prior-only football context.
3. Builds structured market-residual challengers by feature family rather than one
   opaque ALL_STATS model.
4. Decomposes CORE/market/STAT disagreement so later prospective results can tell
   us *why* a disagreement worked or failed.

Every fit is season-forward.  No model in this module has production authority.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from typing import Iterable

import numpy as np
import pandas as pd

from nfl_stats_context_v1 import FEATURE_FAMILIES, ALL_STATS_FEATURES
from nfl_feature_audit_v1 import EXPLICIT_FEATURE_MANIFEST

# Preserve the older audited prior-only context as research inputs while adding
# the newer box-score-derived families. These families may overlap in information;
# source reconciliation later prevents them from being counted as independent votes.
RESEARCH_FEATURE_FAMILIES = OrderedDict({
    **{k: tuple(v) for k, v in FEATURE_FAMILIES.items()},
    "LEGACY_PREGAME_SCHEDULE": tuple(EXPLICIT_FEATURE_MANIFEST["pregame_schedule"]),
    "LEGACY_TEAM_STATE": tuple(EXPLICIT_FEATURE_MANIFEST["prior_team_state"]),
    "LEGACY_OPP_MATCHUP": tuple(EXPLICIT_FEATURE_MANIFEST["prior_opponent_and_matchups"]),
    "LEGACY_REST_SEASON": tuple(EXPLICIT_FEATURE_MANIFEST["prior_season_and_rest"]),
})
RESEARCH_ALL_FEATURES = tuple(dict.fromkeys(x for vals in RESEARCH_FEATURE_FAMILIES.values() for x in vals))

SOURCE_TAG = "nfl-structured-research-v1.9.4-null-safe-season-forward-20261001"
PRODUCTION_AUTHORITY = 0
MAX_RESEARCH_SEASON = 2025
DEVELOPMENT_YEARS = (2021, 2022, 2023, 2024, 2025)
RIDGE_ALPHA_RESIDUAL = 25.0
RIDGE_ALPHA_CORE = 35.0
CORRECTION_CLIP = {"spreads": 6.0, "totals": 8.0}

# These are concept groups, not feature-selected groups.  The lean challenger is
# deliberately football/efficiency heavy; the broad challenger adds situational
# and stability context while remaining completely market-independent.
CORE_FEATURE_SETS = OrderedDict({
    "CORE_EFFICIENCY_RIDGE": tuple(dict.fromkeys(
        FEATURE_FAMILIES["OPPONENT_ADJUSTED"]
        + FEATURE_FAMILIES["RUN_PASS_MATCHUP"]
        + FEATURE_FAMILIES["PACE_EFFICIENCY"]
        + FEATURE_FAMILIES["VENUE_FORM"]
    )),
    "CORE_STRUCTURED_RIDGE": tuple(dict.fromkeys(
        FEATURE_FAMILIES["OPPONENT_ADJUSTED"]
        + FEATURE_FAMILIES["RUN_PASS_MATCHUP"]
        + FEATURE_FAMILIES["PACE_EFFICIENCY"]
        + FEATURE_FAMILIES["VENUE_FORM"]
        + FEATURE_FAMILIES["SCHEDULE_SEQUENCE"]
        + FEATURE_FAMILIES["DIVISION_REMATCH"]
        + FEATURE_FAMILIES["TURNOVER_REGRESSION"]
        + FEATURE_FAMILIES["VOLATILITY_TREND"]
    )),
})


def _finite_series(x, index=None) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").astype(float)
    if index is not None:
        s.index = index
    return s.replace([np.inf, -np.inf], np.nan)


def _pipeline(alpha: float):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=float(alpha))),
    ])


def _mae(a, b) -> float:
    x = np.asarray(a, float); y = np.asarray(b, float)
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.mean(np.abs(x[m] - y[m]))) if m.any() else math.nan


def _rmse(a, b) -> float:
    x = np.asarray(a, float); y = np.asarray(b, float)
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.sqrt(np.mean((x[m] - y[m]) ** 2))) if m.any() else math.nan


def _safe_corr(a, b):
    x = np.asarray(a, float); y = np.asarray(b, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.nanstd(x[m]) <= 1e-12 or np.nanstd(y[m]) <= 1e-12:
        return None
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _feature_cols(df: pd.DataFrame, names: Iterable[str]) -> list[str]:
    # V1.9.4: require at least one finite observed value in the training frame.
    # This removes dead columns such as Game_Hour_ET from sklearn pipelines
    # instead of asking SimpleImputer to warn and silently skip them each fold.
    out = []
    for c in dict.fromkeys(names):
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if x.notna().any():
            out.append(c)
    return out


def _market_parts(df: pd.DataFrame, market: str):
    if market == "spreads":
        actual = _finite_series(df["actual_margin"])
        base = -_finite_series(df["Spread_Value"])
        residual = actual - base
    elif market == "totals":
        actual = _finite_series(df["actual_total"])
        base = _finite_series(df["Current_Total"])
        residual = actual - base
    else:
        raise ValueError(market)
    return actual, base, residual


def _fit_predict(train: pd.DataFrame, valid: pd.DataFrame, cols: list[str], y: pd.Series, alpha: float):
    if not cols:
        raise RuntimeError("NO_FEATURES")
    mask = _finite_series(y).notna()
    if int(mask.sum()) < 500:
        raise RuntimeError(f"TOO_FEW_TRAIN_ROWS n={int(mask.sum())}")
    model = _pipeline(alpha)
    model.fit(train.loc[mask, cols], _finite_series(y.loc[mask]))
    pred = np.asarray(model.predict(valid[cols]), float)
    return model, pred


def _metrics(actual, pred, *, baseline=None) -> dict:
    out = {
        "n": int(np.isfinite(np.asarray(actual, float)).sum()),
        "mae": round(_mae(actual, pred), 6),
        "rmse": round(_rmse(actual, pred), 6),
    }
    c = _safe_corr(actual, pred)
    out["corr"] = round(c, 6) if c is not None else None
    if baseline is not None:
        bm = _mae(actual, baseline)
        out["baseline_mae"] = round(bm, 6)
        out["mae_improvement_vs_baseline"] = round(bm - _mae(actual, pred), 6)
    return out


def _strict_majority(n: int) -> int:
    return int(n // 2 + 1)


def _family_year_summary(oof: pd.DataFrame, market: str, family: str, years: Iterable[int]) -> dict:
    rows = []
    for year in years:
        q = oof.loc[pd.to_numeric(oof.Season, errors="coerce").eq(int(year))].copy()
        if q.empty:
            continue
        actual, base, _ = _market_parts(q, market)
        col = f"resid__{market}__{family}"
        if col not in q:
            continue
        correction = _finite_series(q[col])
        pred = base + correction
        bm = _mae(actual, base); cm = _mae(actual, pred)
        rows.append({
            "season": int(year), "n": int(actual.notna().sum()),
            "market_mae": bm, "corrected_mae": cm, "improvement": bm - cm,
            "mean_abs_correction": float(np.nanmean(np.abs(correction))) if correction.notna().any() else math.nan,
        })
    if not rows:
        return {"family": family, "n": 0, "positive_seasons": 0, "season_improvements": []}
    n = sum(r["n"] for r in rows)
    imp = float(np.average([r["improvement"] for r in rows], weights=[r["n"] for r in rows]))
    return {
        "family": family,
        "n": int(n),
        "weighted_improvement": round(imp, 6),
        "positive_seasons": int(sum(r["improvement"] > 0 for r in rows)),
        "season_count": int(len(rows)),
        "season_improvements": [round(float(r["improvement"]), 6) for r in rows],
        "mean_abs_correction": round(float(np.average([r["mean_abs_correction"] for r in rows], weights=[r["n"] for r in rows])), 6),
        "last_season_improvement": round(float(rows[-1]["improvement"]), 6),
    }


def select_stable_families(oof: pd.DataFrame, market: str, years: Iterable[int]) -> tuple[list[str], dict]:
    yrs = tuple(int(y) for y in years)
    need = _strict_majority(len(yrs)) if yrs else 999
    details = {}
    selected = []
    for fam in RESEARCH_FEATURE_FAMILIES:
        s = _family_year_summary(oof, market, fam, yrs)
        stable = (
            s.get("season_count", 0) == len(yrs)
            and s.get("weighted_improvement", -999) > 0
            and s.get("positive_seasons", 0) >= need
            and s.get("last_season_improvement", -999) > 0
        )
        s["stable_by_prefrozen_rule"] = bool(stable)
        s["required_positive_seasons"] = int(need)
        s["requires_most_recent_selection_season_positive"] = True
        details[fam] = s
        if stable:
            selected.append(fam)
    return selected, details


def _nested_stable_blend(oof: pd.DataFrame, market: str) -> tuple[pd.Series, dict]:
    out = pd.Series(np.nan, index=oof.index, dtype=float)
    folds = []
    for year in (2023, 2024, 2025):
        prior_years = [y for y in DEVELOPMENT_YEARS if y < year]
        selected, selection = select_stable_families(oof, market, prior_years)
        m = pd.to_numeric(oof.Season, errors="coerce").eq(year)
        if selected:
            cols = [f"resid__{market}__{fam}" for fam in selected]
            correction = oof.loc[m, cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
            correction = correction.clip(-CORRECTION_CLIP[market], CORRECTION_CLIP[market])
        else:
            correction = pd.Series(0.0, index=oof.index[m])
        out.loc[m] = correction
        folds.append({
            "validate_season": int(year),
            "selection_years": prior_years,
            "selected_families": selected,
            "selected_count": len(selected),
            "selection": selection,
        })
    return out, {"folds": folds, "selection_rule": "weighted_improvement>0 AND strict-majority positive seasons AND most-recent selection season positive; selection uses prior development folds only"}


def _bucket_label(x: float) -> str:
    if not math.isfinite(x): return "MISSING"
    a = abs(x)
    if a < 2: return "0_2"
    if a < 4: return "2_4"
    if a < 6: return "4_6"
    return "6_PLUS"


def _direction_accuracy(edge, result_margin):
    e = np.asarray(edge, float); r = np.asarray(result_margin, float)
    m = np.isfinite(e) & np.isfinite(r) & (np.abs(e) > 1e-12) & (np.abs(r) > 1e-12)
    if not m.any():
        return {"n": 0, "wins": 0, "rate": None}
    wins = np.sign(e[m]) == np.sign(r[m])
    return {"n": int(m.sum()), "wins": int(wins.sum()), "rate": round(float(wins.mean()), 6)}


def disagreement_report(oof: pd.DataFrame) -> dict:
    report = {"spreads": {}, "totals": {}, "production_authority": 0}
    for market in ("spreads", "totals"):
        actual, base, residual = _market_parts(oof, market)
        core = _finite_series(oof["core_margin_pred"] if market == "spreads" else oof["core_total_pred"])
        nested = _finite_series(oof[f"nested_stable_correction__{market}"])
        core_gap = core - base
        stat_gap = nested
        same = core_gap.notna() & stat_gap.notna() & core_gap.ne(0) & stat_gap.ne(0) & np.sign(core_gap).eq(np.sign(stat_gap))
        market_result = residual
        buckets = {}
        for lab in ("0_2", "2_4", "4_6", "6_PLUS"):
            m = core_gap.map(_bucket_label).eq(lab)
            buckets[lab] = {
                "games": int(m.sum()),
                "follow_core_direction": _direction_accuracy(core_gap[m], market_result[m]),
                "market_mae": round(_mae(actual[m], base[m]), 6) if m.any() else None,
                "core_mae": round(_mae(actual[m], core[m]), 6) if m.any() else None,
            }
        report[market] = {
            "core_market_gap_buckets": buckets,
            "core_stat_same_direction": {
                "same": _direction_accuracy(core_gap[same], market_result[same]),
                "different_or_zero": _direction_accuracy(core_gap[~same], market_result[~same]),
            },
            "same_direction_games": int(same.sum()),
        }
    return report


def run_structured_research(stats_games: pd.DataFrame, incumbent_core_oof: pd.DataFrame, *, log_func=print) -> tuple[dict, pd.DataFrame, dict]:
    """Return (report, OOF frame, final fitted model bundle).

    The bundle is fit through 2025 only and is intended for future shadow scoring,
    never production selection.
    """
    if stats_games is None or stats_games.empty:
        raise RuntimeError("NFL_V1_9_3_NO_STATS_GAMES")
    seasons = pd.to_numeric(stats_games.Season, errors="coerce")
    if seasons.isna().any() or int(seasons.max()) > MAX_RESEARCH_SEASON:
        raise RuntimeError("NFL_V1_9_3_2026_OR_FUTURE_DATA_FORBIDDEN")
    if set(DEVELOPMENT_YEARS) - set(int(x) for x in seasons.unique()):
        raise RuntimeError("NFL_V1_9_3_MISSING_DEVELOPMENT_SEASON")

    core_cols = ["physical_game_id", "direct_margin_pred", "score_margin_pred", "core_margin_pred",
                 "direct_total_pred", "score_total_pred", "core_total_pred", "spread_model_gap", "total_model_gap"]
    missing_core = [c for c in core_cols if c not in incumbent_core_oof.columns]
    if missing_core:
        raise RuntimeError("NFL_V1_9_3_INCUMBENT_CORE_COLUMNS_MISSING "+str(missing_core))

    pieces = []
    for year in DEVELOPMENT_YEARS:
        tr = stats_games.loc[seasons.lt(year)].copy()
        va = stats_games.loc[seasons.eq(year)].copy()
        if len(tr) < 900 or len(va) < 200:
            raise RuntimeError(f"NFL_V1_9_3_INSUFFICIENT_FOLD_{year} train={len(tr)} valid={len(va)}")
        part = va.copy()

        # Structured market-residual family predictions.
        for market in ("spreads", "totals"):
            _, _, yresid = _market_parts(tr, market)
            for fam, names in (*RESEARCH_FEATURE_FAMILIES.items(), ("ALL_STATS", RESEARCH_ALL_FEATURES)):
                cols = _feature_cols(tr, names)
                if not cols:
                    continue
                _, corr = _fit_predict(tr, va, cols, yresid, RIDGE_ALPHA_RESIDUAL)
                corr = np.clip(corr, -CORRECTION_CLIP[market], CORRECTION_CLIP[market])
                part[f"resid__{market}__{fam}"] = corr

        # New independent CORE challengers. Market variables are never features or targets here.
        for name, names in CORE_FEATURE_SETS.items():
            cols = _feature_cols(tr, names)
            _, pm = _fit_predict(tr, va, cols, _finite_series(tr.actual_margin), RIDGE_ALPHA_CORE)
            _, pt = _fit_predict(tr, va, cols, _finite_series(tr.actual_total), RIDGE_ALPHA_CORE)
            part[f"core_chal_margin__{name}"] = pm
            part[f"core_chal_total__{name}"] = pt
        pieces.append(part)

    oof = pd.concat(pieces, ignore_index=True)
    core_small = incumbent_core_oof[core_cols].copy()
    oof = oof.merge(core_small, on="physical_game_id", how="left", validate="one_to_one")
    if oof["core_margin_pred"].isna().all() or oof["core_total_pred"].isna().all():
        raise RuntimeError("NFL_V1_9_3_CORE_OOF_JOIN_FAILED")

    # Nested stable-family blend: selection for each outer season uses only earlier OOF seasons.
    nested_info = {}
    for market in ("spreads", "totals"):
        corr, info = _nested_stable_blend(oof, market)
        oof[f"nested_stable_correction__{market}"] = corr
        nested_info[market] = info

    residual_report = {}
    final_selected = {}
    for market in ("spreads", "totals"):
        actual, base, _ = _market_parts(oof, market)
        fam_report = {}
        for fam in (*RESEARCH_FEATURE_FAMILIES.keys(), "ALL_STATS"):
            col = f"resid__{market}__{fam}"
            if col not in oof.columns:
                fam_report[fam] = {"n":0,"status":"NO_AVAILABLE_FEATURES","by_season":_family_year_summary(oof, market, fam, DEVELOPMENT_YEARS)}
                continue
            corr = _finite_series(oof[col]).clip(-CORRECTION_CLIP[market], CORRECTION_CLIP[market])
            fam_report[fam] = _metrics(actual, base + corr, baseline=base)
            fam_report[fam]["by_season"] = _family_year_summary(oof, market, fam, DEVELOPMENT_YEARS)
        nested_corr = _finite_series(oof[f"nested_stable_correction__{market}"])
        nested_mask = nested_corr.notna()
        nested_metrics = _metrics(actual[nested_mask], (base + nested_corr)[nested_mask], baseline=base[nested_mask]) if nested_mask.any() else {"n":0}
        selected, final_detail = select_stable_families(oof, market, DEVELOPMENT_YEARS)
        final_selected[market] = selected
        residual_report[market] = {
            "families": fam_report,
            "nested_stable_blend": nested_metrics,
            "nested_selection": nested_info[market],
            "prospective_freeze_selected_families": selected,
            "prospective_selection_detail": final_detail,
            "correction_clip_points": CORRECTION_CLIP[market],
        }

    core_report = {"spreads": {}, "totals": {}}
    for market in ("spreads", "totals"):
        actual = _finite_series(oof.actual_margin if market == "spreads" else oof.actual_total)
        incumbent = _finite_series(oof.core_margin_pred if market == "spreads" else oof.core_total_pred)
        core_report[market]["INCUMBENT_CORE"] = _metrics(actual, incumbent)
        for name in CORE_FEATURE_SETS:
            col = f"core_chal_{'margin' if market == 'spreads' else 'total'}__{name}"
            challenger = _finite_series(oof[col])
            core_report[market][name] = _metrics(actual, challenger, baseline=incumbent)
            blend = (challenger + incumbent) / 2.0
            core_report[market][f"INCUMBENT50_{name}50"] = _metrics(actual, blend, baseline=incumbent)

    disagree = disagreement_report(oof)

    # Fit frozen prospective-shadow model objects through 2025 only.
    final_models = {"residual": {}, "core": {}, "metadata": {}}
    for market in ("spreads", "totals"):
        final_models["residual"][market] = {}
        _, _, yresid = _market_parts(stats_games, market)
        for fam in final_selected[market]:
            cols = _feature_cols(stats_games, RESEARCH_FEATURE_FAMILIES[fam])
            model, _ = _fit_predict(stats_games, stats_games.iloc[:1], cols, yresid, RIDGE_ALPHA_RESIDUAL)
            final_models["residual"][market][fam] = {"model": model, "features": cols}
    for name, names in CORE_FEATURE_SETS.items():
        cols = _feature_cols(stats_games, names)
        mm, _ = _fit_predict(stats_games, stats_games.iloc[:1], cols, _finite_series(stats_games.actual_margin), RIDGE_ALPHA_CORE)
        mt, _ = _fit_predict(stats_games, stats_games.iloc[:1], cols, _finite_series(stats_games.actual_total), RIDGE_ALPHA_CORE)
        final_models["core"][name] = {"margin_model": mm, "total_model": mt, "features": cols}

    final_models["metadata"] = {
        "source_tag": SOURCE_TAG,
        "max_training_season": MAX_RESEARCH_SEASON,
        "development_years": DEVELOPMENT_YEARS,
        "selected_residual_families": final_selected,
        "ridge_alpha_residual": RIDGE_ALPHA_RESIDUAL,
        "ridge_alpha_core": RIDGE_ALPHA_CORE,
        "correction_clip": CORRECTION_CLIP,
        "core_feature_sets": {k:list(v) for k,v in CORE_FEATURE_SETS.items()},
        "production_authority": 0,
    }

    spec = {
        "source_tag": SOURCE_TAG,
        "max_training_season": MAX_RESEARCH_SEASON,
        "development_years": DEVELOPMENT_YEARS,
        "selected_residual_families": final_selected,
        "core_feature_sets": {k:list(v) for k,v in CORE_FEATURE_SETS.items()},
        "residual_feature_families": {k:list(v) for k,v in RESEARCH_FEATURE_FAMILIES.items()},
        "selection_rule": "weighted_improvement>0, strict-majority positive development seasons, and most-recent development season positive",
        "nested_outer_years": [2023, 2024, 2025],
        "production_authority": 0,
    }
    registry_sha = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    final_models["metadata"]["structured_registry_sha256"] = registry_sha

    report = {
        "status": "NFL_V1_9_4_STRUCTURED_RESEARCH_COMPLETE",
        "source_tag": SOURCE_TAG,
        "structured_registry_sha256": registry_sha,
        "data_through": 2025,
        "year_2026_queried": False,
        "residual": residual_report,
        "core_challengers": core_report,
        "disagreement": disagree,
        "prospective_selected_residual_families": final_selected,
        "production_authority": 0,
    }
    log_func("[NFL-V1.9.4-STRUCTURED] "+json.dumps(report, sort_keys=True, default=str))
    return report, oof, final_models
