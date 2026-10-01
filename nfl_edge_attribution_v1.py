"""NFL V1.9.2 edge-attribution helpers.

Pure diagnostics: preserve independent CORE predictions and market-residual
predictions side-by-side and explain disagreement.  This module never selects a
bet, never promotes a model, and never mutates NCAAF/legacy NFL paths.
"""
from __future__ import annotations

import json
import math
from typing import Iterable

import numpy as np
import pandas as pd

SOURCE_TAG = "nfl-edge-attribution-v1.9.2-independent-margin-20261001"
PRODUCTION_AUTHORITY = 0


def _numeric(df: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _mean_available(a: pd.Series, b: pd.Series, fallback: pd.Series) -> pd.Series:
    two = pd.concat([a, b], axis=1)
    out = two.mean(axis=1, skipna=True)
    return out.where(two.notna().any(axis=1), fallback)


def enrich_edge_attribution(rows: pd.DataFrame) -> pd.DataFrame:
    """Add transparent margin/total disagreement diagnostics.

    Conventions:
      * margin values are HOME minus AWAY expected points.
      * market_margin_reference must use the same HOME-minus-AWAY convention.
      * total values are expected combined points.
      * positive stats_spread_correction moves the fair HOME margin upward.

    If the caller does not supply market_margin_reference/market_total_reference,
    those diagnostics remain null rather than guessing from a side-specific line.
    """
    if rows is None:
        return pd.DataFrame()
    d = rows.copy()

    direct_margin = _numeric(d, ["core_direct_margin_pred", "direct_margin_pred"])
    score_margin = _numeric(d, ["core_score_margin_pred", "score_margin_pred"])
    fallback_margin = _numeric(d, ["core_margin_pred"])
    direct_total = _numeric(d, ["core_direct_total_pred", "direct_total_pred"])
    score_total = _numeric(d, ["core_score_total_pred", "score_total_pred"])
    fallback_total = _numeric(d, ["core_total_pred"])

    d["core_margin_family_pred"] = _mean_available(direct_margin, score_margin, fallback_margin)
    d["core_total_family_pred"] = _mean_available(direct_total, score_total, fallback_total)
    d["core_margin_internal_gap"] = (direct_margin - score_margin).abs()
    d["core_total_internal_gap"] = (direct_total - score_total).abs()

    market_margin = _numeric(d, ["market_margin_reference", "market_fair_margin", "market_margin"])
    market_total = _numeric(d, ["market_total_reference", "market_fair_total", "market_total"])
    spread_corr = _numeric(d, ["stats_spread_correction", "spread_market_correction"])
    total_corr = _numeric(d, ["stats_total_correction", "total_market_correction"])
    corrected_margin_given = _numeric(d, ["stats_corrected_margin_pred"])
    corrected_total_given = _numeric(d, ["stats_corrected_total_pred"])

    calc_margin = market_margin + spread_corr
    calc_total = market_total + total_corr
    d["stats_corrected_margin_pred"] = corrected_margin_given.where(corrected_margin_given.notna(), calc_margin)
    d["stats_corrected_total_pred"] = corrected_total_given.where(corrected_total_given.notna(), calc_total)
    d["core_market_margin_gap"] = d["core_margin_family_pred"] - market_margin
    d["core_market_total_gap"] = d["core_total_family_pred"] - market_total
    d["stats_market_margin_gap"] = d["stats_corrected_margin_pred"] - market_margin
    d["stats_market_total_gap"] = d["stats_corrected_total_pred"] - market_total

    # Agreement is descriptive only.  No threshold is used to create betting authority.
    cm = pd.to_numeric(d["core_market_margin_gap"], errors="coerce")
    sm = pd.to_numeric(d["stats_market_margin_gap"], errors="coerce")
    ct = pd.to_numeric(d["core_market_total_gap"], errors="coerce")
    st = pd.to_numeric(d["stats_market_total_gap"], errors="coerce")
    d["spread_core_stat_same_direction"] = (
        cm.notna() & sm.notna() & cm.ne(0) & sm.ne(0) & np.sign(cm).eq(np.sign(sm))
    )
    d["total_core_stat_same_direction"] = (
        ct.notna() & st.notna() & ct.ne(0) & st.ne(0) & np.sign(ct).eq(np.sign(st))
    )

    def _finite(v):
        try:
            x = float(v)
            return x if math.isfinite(x) else None
        except Exception:
            return None

    attrs = []
    for _, r in d.iterrows():
        attrs.append(json.dumps({
            "core": {
                "direct_margin": _finite(r.get("core_direct_margin_pred", r.get("direct_margin_pred"))),
                "score_margin": _finite(r.get("core_score_margin_pred", r.get("score_margin_pred"))),
                "family_margin": _finite(r.get("core_margin_family_pred")),
                "direct_total": _finite(r.get("core_direct_total_pred", r.get("direct_total_pred"))),
                "score_total": _finite(r.get("core_score_total_pred", r.get("score_total_pred"))),
                "family_total": _finite(r.get("core_total_family_pred")),
                "margin_internal_gap": _finite(r.get("core_margin_internal_gap")),
                "total_internal_gap": _finite(r.get("core_total_internal_gap")),
            },
            "market": {
                "margin_reference": _finite(r.get("market_margin_reference", r.get("market_fair_margin"))),
                "total_reference": _finite(r.get("market_total_reference", r.get("market_fair_total"))),
            },
            "stats_residual": {
                "spread_correction": _finite(r.get("stats_spread_correction", r.get("spread_market_correction"))),
                "corrected_margin": _finite(r.get("stats_corrected_margin_pred")),
                "total_correction": _finite(r.get("stats_total_correction", r.get("total_market_correction"))),
                "corrected_total": _finite(r.get("stats_corrected_total_pred")),
            },
            "disagreement": {
                "core_minus_market_margin": _finite(r.get("core_market_margin_gap")),
                "stats_minus_market_margin": _finite(r.get("stats_market_margin_gap")),
                "core_minus_market_total": _finite(r.get("core_market_total_gap")),
                "stats_minus_market_total": _finite(r.get("stats_market_total_gap")),
                "spread_core_stat_same_direction": bool(r.get("spread_core_stat_same_direction", False)),
                "total_core_stat_same_direction": bool(r.get("total_core_stat_same_direction", False)),
            },
            "production_authority": 0,
        }, sort_keys=True))
    d["edge_attribution_json"] = attrs
    d["edge_attribution_source_tag"] = SOURCE_TAG
    d["production_authority"] = 0
    return d
