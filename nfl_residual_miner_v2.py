"""NFL residual Miner V2 for V1.9.3.

The miner searches *market residual error*, not raw favorite/dog hit rates.
Discovery is 2021-2023, shadow is 2024 and confirmation is 2025. Numerical
thresholds are learned from discovery only. Multiple testing is controlled with
Benjamini-Hochberg FDR. 2026 is forbidden.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_stats_context_v1 import FEATURE_FAMILIES, ALL_STATS_FEATURES
from nfl_feature_audit_v1 import EXPLICIT_FEATURE_MANIFEST

MINER_CONTEXT_FAMILIES = {
    **{k: tuple(v) for k, v in FEATURE_FAMILIES.items()},
    "LEGACY_PREGAME_SCHEDULE": tuple(EXPLICIT_FEATURE_MANIFEST["pregame_schedule"]),
    "LEGACY_TEAM_STATE": tuple(EXPLICIT_FEATURE_MANIFEST["prior_team_state"]),
    "LEGACY_OPP_MATCHUP": tuple(EXPLICIT_FEATURE_MANIFEST["prior_opponent_and_matchups"]),
    "LEGACY_REST_SEASON": tuple(EXPLICIT_FEATURE_MANIFEST["prior_season_and_rest"]),
}
MINER_CONTEXT_FEATURES = tuple(dict.fromkeys(x for vals in MINER_CONTEXT_FAMILIES.values() for x in vals))

SOURCE_TAG = "nfl-residual-miner-v2.0-market-error-fdr-20261001"
PRODUCTION_AUTHORITY = 0
DISCOVERY_YEARS = (2021, 2022, 2023)
SHADOW_YEAR = 2024
CONFIRM_YEAR = 2025
MAX_RESEARCH_SEASON = 2025

FEATURE_TO_FAMILY = {c:fam for fam, cols in MINER_CONTEXT_FAMILIES.items() for c in cols}


def _num(s):
    return pd.to_numeric(s, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)


def _bh_qvalues(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return np.asarray([], float)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n, float); out[order] = q
    return out


def _one_sample_p(x):
    from scipy.stats import ttest_1samp
    a = np.asarray(x, float); a = a[np.isfinite(a)]
    if len(a) < 8 or np.nanstd(a) <= 1e-12:
        return 1.0
    return float(ttest_1samp(a, 0.0, nan_policy="omit").pvalue)


def _mask_from_rule(df: pd.DataFrame, rule: dict) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for cond in rule["conditions"]:
        s = _num(df[cond["feature"]])
        if cond["op"] == ">=": m &= s.ge(float(cond["value"]))
        elif cond["op"] == "<=": m &= s.le(float(cond["value"]))
        elif cond["op"] == "==": m &= s.eq(float(cond["value"]))
        else: raise ValueError(cond["op"])
    return m.fillna(False)


def _stats(df: pd.DataFrame, residual: pd.Series, mask: pd.Series) -> dict:
    x = _num(residual[mask]).dropna()
    n = len(x)
    if not n:
        return {"n": 0, "mean_residual": None, "median_residual": None, "direction_accuracy": None, "p": None}
    mu = float(x.mean())
    direction = 1.0 if mu > 0 else -1.0 if mu < 0 else 0.0
    acc = float((np.sign(x.to_numpy(float)) == direction).mean()) if direction else None
    return {
        "n": int(n),
        "mean_residual": round(mu, 6),
        "median_residual": round(float(x.median()), 6),
        "direction_accuracy": round(acc, 6) if acc is not None else None,
        "p": round(_one_sample_p(x), 8),
    }


def _candidate_conditions(discovery: pd.DataFrame) -> list[dict]:
    out = []
    for feature in MINER_CONTEXT_FEATURES:
        if feature not in discovery.columns:
            continue
        s = _num(discovery[feature]).dropna()
        if len(s) < 120:
            continue
        vals = np.unique(np.round(s.to_numpy(float), 10))
        family = FEATURE_TO_FAMILY.get(feature, "OTHER")
        if len(vals) <= 3 and set(vals).issubset({0.0, 1.0}):
            prevalence = float((s >= .5).mean())
            if .05 <= prevalence <= .95:
                out.append({"feature":feature,"family":family,"op":"==","value":1.0,"label":f"{feature}=1"})
            continue
        q25, q75 = float(s.quantile(.25)), float(s.quantile(.75))
        if math.isfinite(q25) and math.isfinite(q75) and q75 > q25:
            out.append({"feature":feature,"family":family,"op":"<=","value":q25,"label":f"{feature}<=Q25"})
            out.append({"feature":feature,"family":family,"op":">=","value":q75,"label":f"{feature}>=Q75"})
    return out


def _rule_id(market: str, conditions: list[dict]) -> str:
    raw = json.dumps({"market":market,"conditions":conditions}, sort_keys=True, separators=(",", ":"))
    return "RM2-" + hashlib.sha256(raw.encode()).hexdigest()[:12].upper()


def _market_residual(df: pd.DataFrame, market: str) -> pd.Series:
    if market == "spreads":
        return _num(df.actual_margin) + _num(df.Spread_Value)
    if market == "totals":
        return _num(df.actual_total) - _num(df.Current_Total)
    raise ValueError(market)


def _validate_rule(rule: dict, df: pd.DataFrame, market: str) -> dict:
    resid = _market_residual(df, market)
    disc = df.Season.isin(DISCOVERY_YEARS)
    shadow = _num(df.Season).eq(SHADOW_YEAR)
    confirm = _num(df.Season).eq(CONFIRM_YEAR)
    m = _mask_from_rule(df, rule)
    return {
        "discovery": _stats(df, resid, m & disc),
        "shadow": _stats(df, resid, m & shadow),
        "confirm": _stats(df, resid, m & confirm),
    }


def _same_nonzero_sign(a, b, c) -> bool:
    vals = [a,b,c]
    if any(v is None or not math.isfinite(float(v)) or abs(float(v)) < 1e-12 for v in vals):
        return False
    s = [1 if float(v)>0 else -1 for v in vals]
    return s[0] == s[1] == s[2]


def run_residual_miner_v2(stats_oof: pd.DataFrame, *, log_func=print) -> dict:
    if stats_oof is None or stats_oof.empty:
        raise RuntimeError("NFL_MINER_V2_EMPTY_INPUT")
    years = _num(stats_oof.Season)
    if years.isna().any() or int(years.max()) > MAX_RESEARCH_SEASON:
        raise RuntimeError("NFL_MINER_V2_2026_OR_FUTURE_FORBIDDEN")
    needed = set((*DISCOVERY_YEARS, SHADOW_YEAR, CONFIRM_YEAR))
    if not needed.issubset(set(int(x) for x in years.unique())):
        raise RuntimeError("NFL_MINER_V2_MISSING_SPLITS")

    # Model-disagreement variables are legitimate pregame research context because
    # they are OOF predictions, not same-game outcomes. They let the Miner ask when
    # independent CORE disagreement with market is informative.
    stats_oof = stats_oof.copy()
    stats_oof["miner_core_market_margin_gap"] = _num(stats_oof.get("core_margin_pred")) + _num(stats_oof.get("Spread_Value"))
    stats_oof["miner_core_market_total_gap"] = _num(stats_oof.get("core_total_pred")) - _num(stats_oof.get("Current_Total"))
    if "spread_model_gap" in stats_oof: stats_oof["miner_core_internal_margin_gap"] = _num(stats_oof["spread_model_gap"])
    if "total_model_gap" in stats_oof: stats_oof["miner_core_internal_total_gap"] = _num(stats_oof["total_model_gap"])
    if "nested_stable_correction__spreads" in stats_oof: stats_oof["miner_structured_spread_correction"] = _num(stats_oof["nested_stable_correction__spreads"])
    if "nested_stable_correction__totals" in stats_oof: stats_oof["miner_structured_total_correction"] = _num(stats_oof["nested_stable_correction__totals"])
    for c in [x for x in stats_oof.columns if x.startswith("miner_")]:
        FEATURE_TO_FAMILY[c] = "MODEL_DISAGREEMENT"

    report = {"status":"NFL_RESIDUAL_MINER_V2_COMPLETE","source_tag":SOURCE_TAG,"markets":{},"production_authority":0}
    discovery = stats_oof.loc[years.isin(DISCOVERY_YEARS)].copy()
    conditions = _candidate_conditions(discovery)
    # Append OOF model-disagreement conditions using discovery-only quartiles.
    for feature in [c for c in discovery.columns if c.startswith("miner_")]:
        ss=_num(discovery[feature]).dropna()
        if len(ss)>=120:
            q25,q75=float(ss.quantile(.25)),float(ss.quantile(.75))
            if math.isfinite(q25) and math.isfinite(q75) and q75>q25:
                conditions.append({"feature":feature,"family":"MODEL_DISAGREEMENT","op":"<=","value":q25,"label":f"{feature}<=Q25"})
                conditions.append({"feature":feature,"family":"MODEL_DISAGREEMENT","op":">=","value":q75,"label":f"{feature}>=Q75"})

    for market in ("spreads","totals"):
        resid = _market_residual(stats_oof, market)
        disc_mask = years.isin(DISCOVERY_YEARS)
        single = []
        for cond in conditions:
            rule = {"conditions":[cond]}
            m = _mask_from_rule(stats_oof, rule) & disc_mask
            met = _stats(stats_oof, resid, m)
            if met["n"] < 50 or met["mean_residual"] is None:
                continue
            single.append({"rule":rule,"discovery":met})
        q = _bh_qvalues([x["discovery"]["p"] for x in single])
        for x, qv in zip(single, q): x["discovery"]["q"] = round(float(qv), 8)
        effect_floor = 1.0 if market == "spreads" else 1.25
        disc_pass = [x for x in single if abs(x["discovery"]["mean_residual"]) >= effect_floor and x["discovery"]["q"] <= .10]
        disc_pass.sort(key=lambda x:(x["discovery"]["q"], -abs(x["discovery"]["mean_residual"])))

        # Pair search is intentionally bounded to discovery-surviving single rules.
        top = disc_pass[:12]
        pairs = []
        for i in range(len(top)):
            for j in range(i+1, len(top)):
                c1 = top[i]["rule"]["conditions"][0]; c2 = top[j]["rule"]["conditions"][0]
                if c1["feature"] == c2["feature"] or c1["family"] == c2["family"]:
                    continue
                rule = {"conditions":[c1,c2]}
                m = _mask_from_rule(stats_oof, rule) & disc_mask
                met = _stats(stats_oof, resid, m)
                if met["n"] < 35 or met["mean_residual"] is None:
                    continue
                pairs.append({"rule":rule,"discovery":met})
        pq = _bh_qvalues([x["discovery"]["p"] for x in pairs])
        for x, qv in zip(pairs, pq): x["discovery"]["q"] = round(float(qv), 8)
        pair_pass = [x for x in pairs if abs(x["discovery"]["mean_residual"]) >= 1.5*effect_floor and x["discovery"]["q"] <= .10]

        validated = []
        for typ, items in (("SINGLE",disc_pass),("PAIR",pair_pass)):
            for x in items:
                rule = x["rule"]
                splits = _validate_rule(rule, stats_oof, market)
                d,s,c = splits["discovery"], splits["shadow"], splits["confirm"]
                same = _same_nonzero_sign(d["mean_residual"], s["mean_residual"], c["mean_residual"])
                enough = s["n"] >= (15 if typ=="SINGLE" else 10) and c["n"] >= (15 if typ=="SINGLE" else 10)
                post_effect = min(abs(float(s["mean_residual"] or 0)), abs(float(c["mean_residual"] or 0)))
                promising = bool(same and enough and post_effect >= .25)
                direction = None
                if d["mean_residual"] is not None:
                    if market == "spreads": direction = "HOME_ORIENTED" if d["mean_residual"] > 0 else "AWAY_ORIENTED"
                    else: direction = "OVER" if d["mean_residual"] > 0 else "UNDER"
                rid = _rule_id(market, rule["conditions"])
                validated.append({
                    "rule_id": rid, "type":typ, "market":market, "direction":direction,
                    "conditions":rule["conditions"], "discovery":d, "shadow":s, "confirm":c,
                    "same_direction_all_splits":same, "minimum_forward_n_pass":enough,
                    "promising":promising, "production_authority":0,
                })
        promising = [x for x in validated if x["promising"]]
        promising.sort(key=lambda x:(-abs(float(x["confirm"]["mean_residual"] or 0)), -x["confirm"]["n"]))
        report["markets"][market] = {
            "candidate_conditions": len(conditions),
            "single_discovery_fdr_pass": len(disc_pass),
            "pair_discovery_fdr_pass": len(pair_pass),
            "validated_rule_count": len(validated),
            "promising_count": len(promising),
            "promising_rules": promising[:25],
            "all_validated_rules": validated[:80],
            "split_contract": {"discovery":[2021,2022,2023],"shadow":2024,"confirm":2025,"year_2026_queried":False},
            "selection_basis":"market residual magnitude + discovery FDR + same sign in shadow and confirmation; never raw ATS hit rate alone",
        }
    raw = json.dumps(report["markets"], sort_keys=True, default=str, separators=(",", ":"))
    report["miner_registry_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    log_func("[NFL-V1.9.3-MINER-V2] "+json.dumps(report, sort_keys=True, default=str))
    return report
