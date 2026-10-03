"""NFL STAT Selector V2.3 — NCAAF-style conditional edge research.

Purpose
-------
This module ports the *method* that worked in NCAAF rather than copying any
college-football rule literally.  Frozen NFL fair-value models remain untouched.
The statistical layer asks a narrower question:

    When is the frozen model's disagreement with the market trustworthy?

The selector is built from prior-only football-stat families.  Discovery
(2021-2023) fixes family/alpha/strength/core-gap representatives; confirmation
(2024-2025) validates those exact representatives without re-optimizing them.
2026 is excluded from historical selection and is only used later for prospective
live scoring.

All validated STAT families collapse to one STAT_SELECTOR mechanism for betting
authority.  Multiple correlated statistical views can never create multiple
independent votes.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

import nfl_stats_context_v1 as stats
import sports_edge_authority_v1 as shared

SOURCE_TAG = "nfl-stat-selector-v2.3-ncaaf-style-market-reliability-20261002"
MAX_HISTORICAL_SEASON = 2025
DISCOVERY_SEASONS = (2021, 2022, 2023)
CONFIRM_SEASONS = (2024, 2025)
ALPHAS = (12.0, 24.0, 48.0)
SCALED_THRESHOLDS = (0.25, 0.50, 0.75, 1.00, 1.25)
CORE_GAP_THRESHOLDS = (0.0, 2.0, 4.0)
MAX_PREREGISTERED_GROUPS = 3
MIN_DISCOVERY_N = {"SPREADS": 60, "TOTALS": 50}
MIN_CONFIRM_N = {"SPREADS": 30, "TOTALS": 25}

# Concept families remain interpretable.  Composite families are intentionally
# small and football-specific; ALL_STATS is logged as a diagnostic but cannot be
# chosen as an authority selector.
BASE_FAMILIES = OrderedDict((k, tuple(v)) for k, v in stats.FEATURE_FAMILIES.items())
SELECTOR_FAMILIES = OrderedDict({
    **BASE_FAMILIES,
    "EFFICIENCY_COMPOSITE": tuple(dict.fromkeys(
        BASE_FAMILIES["OPPONENT_ADJUSTED"]
        + BASE_FAMILIES["RUN_PASS_MATCHUP"]
        + BASE_FAMILIES["PACE_EFFICIENCY"]
    )),
    "REGRESSION_COMPOSITE": tuple(dict.fromkeys(
        BASE_FAMILIES["TURNOVER_REGRESSION"]
        + BASE_FAMILIES["NONOFFENSIVE_SCORING"]
        + BASE_FAMILIES["VOLATILITY_TREND"]
        + BASE_FAMILIES["CLOSE_GAME_STATE"]
    )),
    "SITUATIONAL_COMPOSITE": tuple(dict.fromkeys(
        BASE_FAMILIES["DIVISION_REMATCH"]
        + BASE_FAMILIES["SCHEDULE_SEQUENCE"]
        + BASE_FAMILIES["VENUE_FORM"]
    )),
    "GAME_FLOW_COMPOSITE": tuple(dict.fromkeys(
        BASE_FAMILIES["QUARTER_HALF_PROFILE"]
        + BASE_FAMILIES["DISCIPLINE_FOURTH_DOWN"]
    )),
})

DEPENDENCY_GROUP = {
    "OPPONENT_ADJUSTED": "EFFICIENCY",
    "RUN_PASS_MATCHUP": "EFFICIENCY",
    "PACE_EFFICIENCY": "EFFICIENCY",
    "EFFICIENCY_COMPOSITE": "EFFICIENCY",
    "TURNOVER_REGRESSION": "REGRESSION",
    "NONOFFENSIVE_SCORING": "REGRESSION",
    "VOLATILITY_TREND": "REGRESSION",
    "CLOSE_GAME_STATE": "REGRESSION",
    "REGRESSION_COMPOSITE": "REGRESSION",
    "DIVISION_REMATCH": "SITUATIONAL",
    "SCHEDULE_SEQUENCE": "SITUATIONAL",
    "VENUE_FORM": "SITUATIONAL",
    "SITUATIONAL_COMPOSITE": "SITUATIONAL",
    "QUARTER_HALF_PROFILE": "GAME_FLOW",
    "DISCIPLINE_FOURTH_DOWN": "GAME_FLOW",
    "GAME_FLOW_COMPOSITE": "GAME_FLOW",
}


def _num(x):
    try:
        z = float(x)
        return z if math.isfinite(z) else np.nan
    except Exception:
        return np.nan


def _sha(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _feature_cols(df: pd.DataFrame, family: str) -> list[str]:
    cols = []
    for c in SELECTOR_FAMILIES[family]:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if s.notna().any():
            cols.append(c)
    return cols


def _conference_fields(team, opp) -> tuple[float, float, float]:
    try:
        import nfl_live_feature_parity_v1 as parity
        a = parity._team_code(team)
        b = parity._team_code(opp)
        da = parity._DIVISION.get(a)
        db = parity._DIVISION.get(b)
    except Exception:
        # Local-test fallback. Production uses the audited parity module above.
        aliases = {
            "buffalo bills":"BUF","miami dolphins":"MIA","new england patriots":"NE","new york jets":"NYJ",
            "baltimore ravens":"BAL","cincinnati bengals":"CIN","cleveland browns":"CLE","pittsburgh steelers":"PIT",
            "houston texans":"HOU","indianapolis colts":"IND","jacksonville jaguars":"JAX","tennessee titans":"TEN",
            "denver broncos":"DEN","kansas city chiefs":"KC","las vegas raiders":"LV","los angeles chargers":"LAC",
            "dallas cowboys":"DAL","new york giants":"NYG","philadelphia eagles":"PHI","washington commanders":"WAS",
            "chicago bears":"CHI","detroit lions":"DET","green bay packers":"GB","minnesota vikings":"MIN",
            "atlanta falcons":"ATL","carolina panthers":"CAR","new orleans saints":"NO","tampa bay buccaneers":"TB",
            "arizona cardinals":"ARI","los angeles rams":"LAR","san francisco 49ers":"SF","seattle seahawks":"SEA",
        }
        div = {
            "BUF":"AFC_EAST","MIA":"AFC_EAST","NE":"AFC_EAST","NYJ":"AFC_EAST",
            "BAL":"AFC_NORTH","CIN":"AFC_NORTH","CLE":"AFC_NORTH","PIT":"AFC_NORTH",
            "HOU":"AFC_SOUTH","IND":"AFC_SOUTH","JAX":"AFC_SOUTH","TEN":"AFC_SOUTH",
            "DEN":"AFC_WEST","KC":"AFC_WEST","LV":"AFC_WEST","LAC":"AFC_WEST",
            "DAL":"NFC_EAST","NYG":"NFC_EAST","PHI":"NFC_EAST","WAS":"NFC_EAST",
            "CHI":"NFC_NORTH","DET":"NFC_NORTH","GB":"NFC_NORTH","MIN":"NFC_NORTH",
            "ATL":"NFC_SOUTH","CAR":"NFC_SOUTH","NO":"NFC_SOUTH","TB":"NFC_SOUTH",
            "ARI":"NFC_WEST","LAR":"NFC_WEST","SF":"NFC_WEST","SEA":"NFC_WEST",
        }
        def code(x):
            z=str(x or "").strip().lower(); u=z.upper()
            return u if u in div else aliases.get(z, u)
        a,b=code(team),code(opp); da,db=div.get(a),div.get(b)
    if not da or not db:
        return np.nan, np.nan, np.nan
    ca = da.split("_", 1)[0]
    cb = db.split("_", 1)[0]
    return float(da == db), float(ca == cb), float(ca != cb)


def _add_alignment_fields(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    vals = [_conference_fields(t, o) for t, o in zip(out.Team_Norm, out.Opponent_Norm)]
    if "Is_Division_Game" not in out.columns:
        out["Is_Division_Game"] = [x[0] for x in vals]
    else:
        current = pd.to_numeric(out["Is_Division_Game"], errors="coerce")
        fallback = pd.Series([x[0] for x in vals], index=out.index, dtype=float)
        out["Is_Division_Game"] = current.where(current.notna(), fallback)
    out["Is_Conference_Game"] = [x[1] for x in vals]
    out["Is_Interconference_Game"] = [x[2] for x in vals]
    if "same_season_rematch" in out.columns:
        out["division_rematch"] = (
            pd.to_numeric(out["Is_Division_Game"], errors="coerce").eq(1)
            & pd.to_numeric(out["same_season_rematch"], errors="coerce").eq(1)
        ).astype(float)
    return out


def _raw_history_query(max_season: int, completed_only: bool = False) -> str:
    cols = ", ".join(f"`{c}`" for c in stats.RAW_STATS_COLUMNS)
    where = [f"Season BETWEEN 2017 AND {int(max_season)}", "Season_Stage IN ('REGULAR','POSTSEASON')"]
    if completed_only:
        where += ["Team_Score IS NOT NULL", "Opponent_Score IS NOT NULL"]
    return f"SELECT {cols} FROM `{stats.RAW}` WHERE " + " AND ".join(where) + " ORDER BY Season, Game_Date, Source_Name, Source_Game_ID, Team_Norm"


def build_historical_stats_games(*, bq_client, games: pd.DataFrame) -> pd.DataFrame:
    raw_cols = {f.name for f in bq_client.get_table(stats.RAW).schema}
    missing = sorted(set(stats.RAW_STATS_COLUMNS) - raw_cols)
    if missing:
        raise RuntimeError("[NFL-STAT-V23-HOLD] RAW_COLUMNS_MISSING " + str(missing))
    raw = bq_client.query(_raw_history_query(MAX_HISTORICAL_SEASON, completed_only=False)).to_dataframe(create_bqstorage_client=False)
    seasons = pd.to_numeric(raw.Season, errors="coerce")
    if raw.empty or seasons.isna().any() or int(seasons.max()) > MAX_HISTORICAL_SEASON:
        raise RuntimeError("[NFL-STAT-V23-HOLD] HISTORICAL_SOURCE_INVALID_OR_2026_LEAK")
    side = stats.derive_stats_context(raw)
    out = stats.attach_stats_features(games, side)
    out = _add_alignment_fields(out)
    if pd.to_numeric(out.Season, errors="coerce").max() > MAX_HISTORICAL_SEASON:
        raise RuntimeError("[NFL-STAT-V23-HOLD] HISTORICAL_STATS_GAMES_2026_LEAK")
    return out


def _target_market_residual(d: pd.DataFrame, market: str) -> pd.Series:
    if market == "SPREADS":
        return pd.to_numeric(d.actual_margin, errors="coerce") + pd.to_numeric(d.Spread_Value, errors="coerce")
    if market == "TOTALS":
        return pd.to_numeric(d.actual_total, errors="coerce") - pd.to_numeric(d.Current_Total, errors="coerce")
    raise ValueError(market)


def _fit_family(train: pd.DataFrame, valid: pd.DataFrame, *, family: str, market: str, alpha: float):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    cols = _feature_cols(train, family)
    if not cols:
        raise RuntimeError("NO_FEATURES_" + family)
    y = _target_market_residual(train, market)
    ok = y.notna()
    if int(ok.sum()) < 500:
        raise RuntimeError(f"TOO_FEW_TRAIN_ROWS family={family} market={market} n={int(ok.sum())}")
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=float(alpha))),
    ])
    pipe.fit(train.loc[ok, cols], y.loc[ok].to_numpy(float))
    pred = np.asarray(pipe.predict(valid[cols]), float)
    train_pred = np.asarray(pipe.predict(train.loc[ok, cols]), float)
    err = y.loc[ok].to_numpy(float) - train_pred
    scale = float(np.nanstd(err, ddof=0))
    if not math.isfinite(scale) or scale <= 1e-9:
        scale = float(np.nanstd(y.loc[ok].to_numpy(float), ddof=0))
    if not math.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return pipe, pred, scale, cols


def _oof_family(stats_games: pd.DataFrame, *, family: str, market: str, alpha: float) -> pd.DataFrame:
    pieces = []
    for sy in (*DISCOVERY_SEASONS, *CONFIRM_SEASONS):
        tr = stats_games.loc[pd.to_numeric(stats_games.Season, errors="coerce").lt(sy)].copy()
        va = stats_games.loc[pd.to_numeric(stats_games.Season, errors="coerce").eq(sy)].copy()
        if tr.empty or va.empty:
            continue
        _, pred, scale, cols = _fit_family(tr, va, family=family, market=market, alpha=alpha)
        pieces.append(pd.DataFrame({
            "physical_game_id": va.physical_game_id.astype(str).to_numpy(),
            "season": int(sy),
            "selector_prediction": pred,
            "selector_scaled": np.abs(pred) / max(scale, 1e-9),
            "selector_scale": scale,
            "feature_count": len(cols),
        }))
    return pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()


def _core_eval_frame(stats_games: pd.DataFrame, replay_rows: pd.DataFrame, market: str) -> pd.DataFrame:
    keep = [
        "physical_game_id", "season", "frozen_fair_margin", "frozen_fair_total",
        "close_spread", "close_total", "actual_margin", "actual_total",
    ]
    r = replay_rows[[c for c in keep if c in replay_rows.columns]].drop_duplicates("physical_game_id").copy()
    s = stats_games.copy()
    s["physical_game_id"] = s.physical_game_id.astype(str)
    r["physical_game_id"] = r.physical_game_id.astype(str)
    d = s.merge(r, on="physical_game_id", how="inner", suffixes=("", "__replay"), validate="one_to_one")
    d["season"] = pd.to_numeric(d.get("season"), errors="coerce").fillna(pd.to_numeric(d.Season, errors="coerce"))
    if market == "SPREADS":
        close = pd.to_numeric(d.close_spread, errors="coerce")
        fair = pd.to_numeric(d.frozen_fair_margin, errors="coerce")
        actual = pd.to_numeric(d["actual_margin__replay"] if "actual_margin__replay" in d else d.actual_margin, errors="coerce")
        d["core_edge"] = fair + close
        d["market_residual"] = actual + close
    else:
        close = pd.to_numeric(d.close_total, errors="coerce")
        fair = pd.to_numeric(d.frozen_fair_total, errors="coerce")
        actual = pd.to_numeric(d["actual_total__replay"] if "actual_total__replay" in d else d.actual_total, errors="coerce")
        d["core_edge"] = fair - close
        d["market_residual"] = actual - close
    d["core_direction"] = np.sign(pd.to_numeric(d.core_edge, errors="coerce"))
    oriented = d.core_direction * pd.to_numeric(d.market_residual, errors="coerce")
    d["core_bet_target"] = np.where(oriented > 0, 1.0, np.where(oriented < 0, 0.0, np.nan))
    return d


def _record(q: pd.DataFrame) -> dict:
    if q is None or q.empty:
        return {"n": 0, "graded_n": 0, "wins": 0, "losses": 0, "pushes": 0, "hit_rate": None, "roi_per_unit": None, "wilson95": [None, None]}
    return shared.record(pd.to_numeric(q.core_bet_target, errors="coerce"), np.full(len(q), 100.0 / 110.0))


def _by_season(q: pd.DataFrame, seasons) -> dict:
    return {str(s): _record(q.loc[pd.to_numeric(q.season, errors="coerce").eq(int(s))]) for s in seasons}


def _positive_roi_seasons(by: dict) -> int:
    n = 0
    for x in by.values():
        r = _num((x or {}).get("roi_per_unit"))
        if math.isfinite(r) and r > 0:
            n += 1
    return n


def _discovery_gate(candidate: dict, market: str) -> dict:
    rec = candidate["discovery"]
    by = candidate["discovery_by_season"]
    reasons = []
    if int(rec.get("n") or 0) < MIN_DISCOVERY_N[market]: reasons.append("DISCOVERY_N")
    if not math.isfinite(_num(rec.get("roi_per_unit"))) or _num(rec.get("roi_per_unit")) <= 0: reasons.append("DISCOVERY_ROI")
    if not math.isfinite(_num(rec.get("hit_rate"))) or _num(rec.get("hit_rate")) <= shared.BREAK_EVEN_110: reasons.append("DISCOVERY_HIT_BELOW_BREAK_EVEN")
    if _positive_roi_seasons(by) < 2: reasons.append("DISCOVERY_SEASON_STABILITY")
    last = by.get("2023") or {}
    if int(last.get("n") or 0) >= 10 and (_num(last.get("hit_rate")) < 0.50 if math.isfinite(_num(last.get("hit_rate"))) else True):
        reasons.append("DISCOVERY_LATEST_SEASON_WEAK")
    return {"status": "PASS" if not reasons else "HOLD", "reasons": reasons}


def _confirmation_gate(candidate: dict, market: str) -> dict:
    rec = candidate["confirmation"]
    by = candidate["confirmation_by_season"]
    reasons = []
    if int(rec.get("n") or 0) < MIN_CONFIRM_N[market]: reasons.append("CONFIRM_N")
    if not math.isfinite(_num(rec.get("roi_per_unit"))) or _num(rec.get("roi_per_unit")) <= 0: reasons.append("CONFIRM_ROI")
    if not math.isfinite(_num(rec.get("hit_rate"))) or _num(rec.get("hit_rate")) <= shared.BREAK_EVEN_110: reasons.append("CONFIRM_HIT_BELOW_BREAK_EVEN")
    if _positive_roi_seasons(by) < 1: reasons.append("CONFIRM_SEASON_STABILITY")
    last = by.get("2025") or {}
    if int(last.get("n") or 0) >= 8 and (_num(last.get("hit_rate")) < 0.50 if math.isfinite(_num(last.get("hit_rate"))) else True):
        reasons.append("CONFIRM_2025_WEAK")
    return {"status": "PASS" if not reasons else "HOLD", "reasons": reasons}


def _candidate_score(c: dict) -> tuple:
    """Discovery-only deterministic ranking. Confirmation is deliberately absent."""
    r = c.get("discovery") or {}
    by = c.get("discovery_by_season") or {}
    wilson = r.get("wilson95") or [None, None]
    lo = _num(wilson[0]) if wilson else np.nan
    return (
        1 if (c.get("discovery_gate") or {}).get("status") == "PASS" else 0,
        _positive_roi_seasons(by),
        lo if math.isfinite(lo) else -1,
        _num(r.get("roi_per_unit")) if math.isfinite(_num(r.get("roi_per_unit"))) else -999,
        _num(r.get("hit_rate")) if math.isfinite(_num(r.get("hit_rate"))) else -999,
        math.log1p(int(r.get("n") or 0)),
        -abs(float(c.get("alpha", 24.0)) - 24.0),
        -abs(float(c.get("selector_threshold", 0.75)) - 0.75),
        -float(c.get("core_gap_min", 0.0)),
    )


def _evaluate_variant(base: pd.DataFrame, oof: pd.DataFrame, *, family: str, market: str, alpha: float, selector_threshold: float, core_gap_min: float) -> dict:
    d = base.merge(oof[["physical_game_id", "selector_prediction", "selector_scaled", "feature_count"]], on="physical_game_id", how="left", validate="one_to_one")
    pred = pd.to_numeric(d.selector_prediction, errors="coerce")
    scaled = pd.to_numeric(d.selector_scaled, errors="coerce")
    core = pd.to_numeric(d.core_edge, errors="coerce")
    mask = pred.notna() & scaled.ge(selector_threshold) & core.abs().ge(core_gap_min) & np.sign(pred).eq(np.sign(core)) & pd.to_numeric(d.core_bet_target, errors="coerce").notna()
    selected = d.loc[mask].copy()
    disc = selected.loc[pd.to_numeric(selected.season, errors="coerce").isin(DISCOVERY_SEASONS)]
    conf = selected.loc[pd.to_numeric(selected.season, errors="coerce").isin(CONFIRM_SEASONS)]
    all_oof = d.loc[pred.notna() & pd.to_numeric(d.market_residual, errors="coerce").notna()].copy()
    dir_acc = float((np.sign(pd.to_numeric(all_oof.selector_prediction, errors="coerce")) == np.sign(pd.to_numeric(all_oof.market_residual, errors="coerce"))).mean()) if len(all_oof) else np.nan
    out = {
        "selector_id": f"{market}__{family}__A{int(alpha)}__S{selector_threshold:.2f}__G{core_gap_min:.1f}",
        "market": market,
        "family": family,
        "dependency_group": DEPENDENCY_GROUP.get(family, family),
        "alpha": float(alpha),
        "selector_threshold": float(selector_threshold),
        "core_gap_min": float(core_gap_min),
        "feature_count": int(pd.to_numeric(oof.get("feature_count"), errors="coerce").dropna().max()) if not oof.empty else 0,
        "discovery": _record(disc),
        "confirmation": _record(conf),
        "discovery_by_season": _by_season(disc, DISCOVERY_SEASONS),
        "confirmation_by_season": _by_season(conf, CONFIRM_SEASONS),
        "fire_rate_discovery": round(float(len(disc) / max(1, int(pd.to_numeric(base.season, errors="coerce").isin(DISCOVERY_SEASONS).sum()))), 6),
        "fire_rate_confirmation": round(float(len(conf) / max(1, int(pd.to_numeric(base.season, errors="coerce").isin(CONFIRM_SEASONS).sum()))), 6),
        "selector_direction_accuracy_all_oof": round(dir_acc, 6) if math.isfinite(dir_acc) else None,
    }
    out["discovery_gate"] = _discovery_gate(out, market)
    out["confirmation_gate"] = _confirmation_gate(out, market)
    return out


def _fit_final_bundle(stats_games: pd.DataFrame, candidate: dict) -> dict:
    family = candidate["family"]
    market = candidate["market"]
    alpha = float(candidate["alpha"])
    # Fit on all completed history through 2025 only. The validation decision was
    # already made by fixed 2021-23 -> 2024-25 logic; this refit cannot change it.
    train = stats_games.loc[pd.to_numeric(stats_games.Season, errors="coerce").le(2025)].copy()
    pipe, _, scale, cols = _fit_family(train, train.iloc[:1].copy(), family=family, market=market, alpha=alpha)
    return {
        "selector_id": candidate["selector_id"],
        "market": market,
        "family": family,
        "dependency_group": candidate["dependency_group"],
        "alpha": alpha,
        "selector_threshold": float(candidate["selector_threshold"]),
        "core_gap_min": float(candidate["core_gap_min"]),
        "feature_cols": cols,
        "scale": float(scale),
        "pipeline": pipe,
        "trained_through": 2025,
        "source_tag": SOURCE_TAG,
    }


def _candidate_public(c: dict) -> dict:
    return {k: c.get(k) for k in (
        "selector_id", "market", "family", "dependency_group", "alpha", "selector_threshold", "core_gap_min", "feature_count",
        "discovery", "confirmation", "discovery_by_season", "confirmation_by_season", "fire_rate_discovery", "fire_rate_confirmation",
        "selector_direction_accuracy_all_oof", "discovery_gate", "confirmation_gate", "status", "preregistered_rank",
    )}


def run_historical_selector_research(*, bq_client, games: pd.DataFrame, replay_rows: pd.DataFrame, log_func=print) -> tuple[dict, dict, dict]:
    log_func("[NFL-STAT-V23-PREFLIGHT] " + json.dumps({
        "status": "START", "source_tag": SOURCE_TAG, "role": "SELECTOR_OF_SHARED_EDGE_NOT_CORE_REWRITE",
        "discovery_seasons": list(DISCOVERY_SEASONS), "confirmation_seasons": list(CONFIRM_SEASONS),
        "alphas": list(ALPHAS), "scaled_thresholds": list(SCALED_THRESHOLDS), "core_gap_thresholds": list(CORE_GAP_THRESHOLDS),
        "family_count": len(SELECTOR_FAMILIES), "year_2026_queried": False,
    }, sort_keys=True))
    stats_games = build_historical_stats_games(bq_client=bq_client, games=games)
    report = {"status": "NFL_STAT_SELECTOR_V23_RESEARCH_COMPLETE", "source_tag": SOURCE_TAG, "year_2026_queried": False,
              "role": "SELECTOR_OF_SHARED_EDGE_NOT_CORE_REWRITE", "markets": {}, "feature_families": {k: list(v) for k, v in SELECTOR_FAMILIES.items()},
              "dependency_groups": DEPENDENCY_GROUP.copy(), "all_stats_diagnostic_only": True}
    bundles = {}
    lookup = {"SPREADS": {}, "TOTALS": {}}

    for market in ("SPREADS", "TOTALS"):
        base = _core_eval_frame(stats_games, replay_rows, market)
        oof_cache = {}
        family_reps = []
        all_candidates = []
        for family in SELECTOR_FAMILIES:
            family_candidates = []
            for alpha in ALPHAS:
                try:
                    oof = _oof_family(stats_games, family=family, market=market, alpha=alpha)
                except Exception as exc:
                    log_func("[NFL-STAT-V23-FAMILY-HOLD] " + json.dumps({"market": market, "family": family, "alpha": alpha, "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
                    continue
                oof_cache[(family, alpha)] = oof
                for st in SCALED_THRESHOLDS:
                    for gap in CORE_GAP_THRESHOLDS:
                        family_candidates.append(_evaluate_variant(base, oof, family=family, market=market, alpha=alpha, selector_threshold=st, core_gap_min=gap))
            if not family_candidates:
                continue
            rep = sorted(family_candidates, key=_candidate_score, reverse=True)[0]
            rep = {**rep, "status": "DISCOVERY_REPRESENTATIVE"}
            family_reps.append(rep)
            all_candidates.extend(family_candidates)
            log_func("[NFL-STAT-V23-FAMILY] " + json.dumps(_candidate_public(rep), sort_keys=True, default=str))

        # Pre-register one discovery representative per dependency group, then at
        # most three groups overall. Confirmation cannot substitute a different
        # family after these representatives are fixed.
        group_best = []
        for group in sorted(set(DEPENDENCY_GROUP.values())):
            q = [x for x in family_reps if x.get("dependency_group") == group and (x.get("discovery_gate") or {}).get("status") == "PASS"]
            if q:
                group_best.append(sorted(q, key=_candidate_score, reverse=True)[0])
        group_best = sorted(group_best, key=_candidate_score, reverse=True)[:MAX_PREREGISTERED_GROUPS]
        prereg = []
        for rank, c0 in enumerate(group_best, start=1):
            c = dict(c0)
            c["preregistered_rank"] = rank
            c["status"] = "LEGIT_FAMILY_REQUIRES_PROSPECTIVE" if (c.get("confirmation_gate") or {}).get("status") == "PASS" else "HOLD_CONFIRMATION"
            prereg.append(c)
            log_func("[NFL-STAT-V23-PRESELECT] " + json.dumps({
                "market": market, "rank": rank, "selector_id": c["selector_id"], "family": c["family"], "dependency_group": c["dependency_group"],
                "discovery": c["discovery"], "confirmation": c["confirmation"], "discovery_gate": c["discovery_gate"],
                "confirmation_gate": c["confirmation_gate"], "status": c["status"], "selection_uses_confirmation": False,
            }, sort_keys=True, default=str))
        validated = [c for c in prereg if c["status"].startswith("LEGIT")]

        # Rich diagnostic: compare CORE results when a validated/preregistered
        # selector fires versus when none fires. This does not select a new rule.
        diag_rows = []
        for c in prereg:
            oof = oof_cache.get((c["family"], float(c["alpha"])))
            if oof is None or oof.empty: continue
            dd = base.merge(oof[["physical_game_id", "selector_prediction", "selector_scaled"]], on="physical_game_id", how="left", validate="one_to_one")
            m = (
                pd.to_numeric(dd.selector_prediction, errors="coerce").notna()
                & pd.to_numeric(dd.selector_scaled, errors="coerce").ge(float(c["selector_threshold"]))
                & pd.to_numeric(dd.core_edge, errors="coerce").abs().ge(float(c["core_gap_min"]))
                & np.sign(pd.to_numeric(dd.selector_prediction, errors="coerce")).eq(np.sign(pd.to_numeric(dd.core_edge, errors="coerce")))
                & pd.to_numeric(dd.core_bet_target, errors="coerce").notna()
            )
            for syset, label in ((DISCOVERY_SEASONS, "DISCOVERY"), (CONFIRM_SEASONS, "CONFIRMATION")):
                q = dd.loc[m & pd.to_numeric(dd.season, errors="coerce").isin(syset)]
                diag_rows.append({"selector_id": c["selector_id"], "sample": label, **_record(q)})
        log_func("[NFL-STAT-V23-CORE-CONDITIONAL] " + json.dumps({"market": market, "selectors": diag_rows}, sort_keys=True, default=str))

        # Final bundles exist for every preregistered selector so failed-confirmation
        # candidates can still be shadow-tracked, but authority is opened only for
        # confirmation PASS candidates.
        market_bundles = {}
        for c in prereg:
            b = _fit_final_bundle(stats_games, c)
            market_bundles[c["selector_id"]] = b
            oof = oof_cache.get((c["family"], float(c["alpha"])), pd.DataFrame())
            lookup[market][c["selector_id"]] = {
                str(r.physical_game_id): {"selector_prediction": _num(r.selector_prediction), "selector_scaled": _num(r.selector_scaled)}
                for _, r in oof.iterrows()
            }
        bundles[market] = market_bundles

        top = sorted(family_reps, key=_candidate_score, reverse=True)[:20]
        report["markets"][market] = {
            "family_representatives": [_candidate_public(x) for x in family_reps],
            "top_discovery_representatives": [_candidate_public(x) for x in top],
            "preregistered": [_candidate_public(x) for x in prereg],
            "validated": [_candidate_public(x) for x in validated],
            "authority_selector_ids": [x["selector_id"] for x in validated],
            "authority_mechanism_independence_key": f"NFL_{market}_STAT_SELECTOR",
            "production_authority": bool(validated),
        }
        log_func("[NFL-STAT-V23-RANKING] " + json.dumps({
            "market": market,
            "top": [{"rank": i + 1, "selector_id": x["selector_id"], "family": x["family"], "group": x["dependency_group"], "discovery": x["discovery"], "discovery_gate": x["discovery_gate"]} for i, x in enumerate(top[:12])],
        }, sort_keys=True, default=str))
        log_func("[NFL-STAT-V23-SELECTOR] " + json.dumps({
            "market": market, "preregistered_selector_ids": [x["selector_id"] for x in prereg],
            "validated_selector_ids": [x["selector_id"] for x in validated],
            "authority_open": bool(validated), "independence_policy": "ALL_VALIDATED_STAT_SELECTORS_COLLAPSE_TO_ONE_STAT_MECHANISM",
        }, sort_keys=True, default=str))

    stable = {
        "source_tag": SOURCE_TAG,
        "role": report["role"],
        "discovery_seasons": list(DISCOVERY_SEASONS),
        "confirmation_seasons": list(CONFIRM_SEASONS),
        "markets": {m: {
            "preregistered": [x["selector_id"] for x in report["markets"][m]["preregistered"]],
            "validated": report["markets"][m]["authority_selector_ids"],
        } for m in ("SPREADS", "TOTALS")},
    }
    report["selector_contract_sha256"] = _sha(stable)
    log_func("[NFL-STAT-V23-CONTRACT] " + json.dumps({
        "status": report["status"], "source_tag": SOURCE_TAG, "selector_contract_sha256": report["selector_contract_sha256"],
        "spread_validated": report["markets"]["SPREADS"]["authority_selector_ids"],
        "total_validated": report["markets"]["TOTALS"]["authority_selector_ids"],
        "year_2026_queried": False, "production_probability_authority": 0,
    }, sort_keys=True, default=str))
    return report, bundles, lookup


def _live_dummy_rows(prediction_rows: list[dict]) -> pd.DataFrame:
    records = []
    for p in prediction_rows:
        gs = pd.to_datetime(p.get("game_start"), utc=True, errors="coerce")
        if pd.isna(gs): continue
        local_date = gs.tz_convert("America/New_York").tz_localize(None).normalize()
        season = int(p.get("season") or (local_date.year - 1 if local_date.month <= 3 else local_date.year))
        week = p.get("week_number")
        gid = str(p.get("prediction_pair_id") or f"LIVE|{season}|{local_date.date()}|{p.get('home_team')}|{p.get('away_team')}")
        home = str(p.get("home_team") or "").strip().lower(); away = str(p.get("away_team") or "").strip().lower()
        if not home or not away: continue
        for team, opp, ih, ia in ((home, away, 1, 0), (away, home, 0, 1)):
            rec = {c: np.nan for c in stats.RAW_STATS_COLUMNS}
            rec.update({
                "Season": season, "Season_Stage": "REGULAR", "Source_Name": "LIVE_STAT_SELECTOR_V23", "Source_Game_ID": gid,
                "Game_Date": local_date, "Week": week, "Start_Time_ET": gs.tz_convert("America/New_York").strftime("%H:%M"),
                "Team_Norm": team, "Opponent_Norm": opp, "Is_Home": ih, "Is_Away": ia, "Is_Neutral": 0.0,
            })
            records.append(rec)
    return pd.DataFrame(records, columns=list(stats.RAW_STATS_COLUMNS)) if records else pd.DataFrame(columns=list(stats.RAW_STATS_COLUMNS))


def build_live_feature_map(*, bq_client, prediction_rows: list[dict]) -> tuple[dict[str, dict], dict]:
    if not prediction_rows:
        return {}, {"status": "NO_PREDICTION_ROWS"}
    seasons = [int(p.get("season")) for p in prediction_rows if p.get("season") is not None]
    max_season = max(seasons) if seasons else 2026
    raw_cols = {f.name for f in bq_client.get_table(stats.RAW).schema}
    missing = sorted(set(stats.RAW_STATS_COLUMNS) - raw_cols)
    if missing:
        return {}, {"status": "HOLD_RAW_COLUMNS_MISSING", "missing": missing}
    raw = bq_client.query(_raw_history_query(max_season, completed_only=True)).to_dataframe(create_bqstorage_client=False)
    dummy = _live_dummy_rows(prediction_rows)
    if dummy.empty:
        return {}, {"status": "HOLD_NO_DUMMY_ROWS"}
    combo = pd.concat([raw, dummy], ignore_index=True, sort=False)
    side = stats.derive_stats_context(combo)
    live = side.loc[side.Source_Name.astype(str).eq("LIVE_STAT_SELECTOR_V23")].copy()
    live = _add_alignment_fields(live)
    live = live.loc[pd.to_numeric(live.Is_Home, errors="coerce").eq(1)].copy()
    fmap = {}
    for _, r in live.iterrows():
        pid = str(r.Source_Game_ID)
        fmap[pid] = r.to_dict()
    expected = {str(p.get("prediction_pair_id")) for p in prediction_rows}
    missing_ids = sorted(expected - set(fmap))
    status = "READY" if not missing_ids else "PARTIAL_MISSING_LIVE_STATS_ROWS"
    return fmap, {"status": status, "rows": len(fmap), "expected": len(expected), "missing_prediction_pair_ids": missing_ids[:20],
                  "feature_builder": "SAME_DERIVE_STATS_CONTEXT_CODE_AS_HISTORICAL", "postgame_fields_for_live_rows": "NULL_BY_CONSTRUCTION"}


def score_live_bundle(bundle: dict, features: dict) -> dict:
    cols = list(bundle.get("feature_cols") or [])
    if not cols:
        return {"status": "HOLD_NO_FEATURES"}
    x = pd.DataFrame([{c: features.get(c) for c in cols}])
    try:
        pred = float(bundle["pipeline"].predict(x)[0])
    except Exception as exc:
        return {"status": "HOLD_SCORE_ERROR", "error": f"{type(exc).__name__}:{exc}"}
    scale = max(_num(bundle.get("scale")), 1e-9)
    return {"status": "SCORED", "selector_prediction": pred, "selector_scaled": abs(pred) / scale,
            "selector_direction": int(np.sign(pred)) if math.isfinite(pred) else 0}


def _self_test():
    assert 2026 not in DISCOVERY_SEASONS + CONFIRM_SEASONS
    assert set(DISCOVERY_SEASONS).isdisjoint(CONFIRM_SEASONS)
    assert DEPENDENCY_GROUP["RUN_PASS_MATCHUP"] == "EFFICIENCY"
    # Ranking must not inspect confirmation: mutate confirmation and score stays equal.
    c = {"discovery": {"n": 100, "roi_per_unit": .05, "hit_rate": .55, "wilson95": [.51, .59]},
         "discovery_by_season": {"2021": {"roi_per_unit": .01}, "2022": {"roi_per_unit": .02}, "2023": {"roi_per_unit": .03}},
         "discovery_gate": {"status": "PASS"}, "alpha": 24, "selector_threshold": .75, "core_gap_min": 2,
         "confirmation": {"roi_per_unit": -1}}
    a = _candidate_score(c)
    c["confirmation"] = {"roi_per_unit": 99}
    assert a == _candidate_score(c)
    return {"status": "PASS", "source_tag": SOURCE_TAG, "families": len(SELECTOR_FAMILIES), "discovery": DISCOVERY_SEASONS, "confirmation": CONFIRM_SEASONS}


if __name__ == "__main__":
    print(json.dumps(_self_test(), sort_keys=True))
