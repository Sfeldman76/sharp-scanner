"""NCAAF V14.2 STAT Reliability + Signed Fade research layer.

Purpose
-------
V14/V14.1 showed that replacing or generically correcting V13 STAT did not add
reliable season-forward value.  V14.2 therefore freezes V13 STAT as the probability
engine and studies *where its selections are reliable*.

A negative regime is first-class information.  For every predefined regime V14.2
reports both directions:

    FOLLOW_STAT hit rate / ROI
    FADE_STAT   hit rate / ROI

A regime can become FOLLOW_QUALIFIED or FADE_QUALIFIED only after it survives
sample-size, multi-season, remove-best-season, and leave-one-season-out stability
checks.  Qualification is research/shadow only and never changes production.

No arbitrary combination mining is performed here.  Regimes are intentionally
predefined and low-dimensional to limit multiple-testing risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

import numpy as np
import pandas as pd

from v14_clean_room import _num, _v13_control_probs

V142_SOURCE_TAG = "v14.2-stat-reliability-signed-fade"
V142_BREAK_EVEN = 110.0 / 210.0
V142_MIN_EVAL_N = 60
V142_CANDIDATE_N = 100
V142_QUALIFIED_N = 150
V142_MIN_SEASON_N = 20
V142_QUALIFIED_HIT = 0.535
V142_MAX_REGIMES = 40


def _safe_float(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _roi_from_hit(hit_rate: float) -> float:
    if not np.isfinite(hit_rate):
        return np.nan
    return float(hit_rate * (100.0 / 110.0) - (1.0 - hit_rate))


def _derive_week(g: pd.DataFrame) -> pd.Series:
    # Prefer an explicit pregame week if available.
    for c in ("Week", "Week_Number", "WeekNum", "BigAl_Context_Week_Number"):
        if c in g.columns:
            s = pd.to_numeric(g[c], errors="coerce")
            if int(s.notna().sum()) >= max(100, int(0.25 * len(g))):
                return s
    # Leakage-safe date-relative fallback within each season.
    d = pd.to_datetime(g.get("Game_Date"), errors="coerce", utc=True)
    s = pd.to_numeric(g.get("Season"), errors="coerce")
    out = pd.Series(np.nan, index=g.index, dtype="float64")
    for sy in sorted(s.dropna().unique()):
        m = s.eq(sy) & d.notna()
        if not m.any():
            continue
        start = d[m].min().normalize()
        out.loc[m] = ((d.loc[m] - start).dt.days // 7 + 1).astype(float)
    return out


def _one_physical_game_audit(g: pd.DataFrame, eval_mask: np.ndarray) -> Tuple[str, int, int]:
    d = g.loc[eval_mask].copy()
    if "Source_Game_ID" in d.columns:
        k = d["Source_Game_ID"].astype(str)
        valid = k.notna() & k.ne("") & k.ne("nan")
        if valid.any():
            n = int(valid.sum()); u = int(k[valid].nunique())
            if n != u:
                raise RuntimeError(f"V14.2 duplicate physical games by Source_Game_ID rows={n} unique={u}")
            return "Source_Game_ID", n, u
    cols = [c for c in ("Season", "Game_Date", "Team_Norm", "Opponent_Norm") if c in d.columns]
    if len(cols) >= 3:
        k = d[cols].astype(str).agg("|".join, axis=1)
        n = len(k); u = int(k.nunique())
        if n != u:
            raise RuntimeError(f"V14.2 duplicate physical games by composite key rows={n} unique={u}")
        return "+".join(cols), n, u
    return "ROW_INDEX", int(len(d)), int(len(d))


def _system_direction_regimes(g: pd.DataFrame, stat_dir: np.ndarray) -> List[Tuple[str, np.ndarray, str]]:
    out: List[Tuple[str, np.ndarray, str]] = []
    specs = [
        ("BIGAL", "Brain_Expert_BigAl_Active", "Brain_Expert_BigAl_Direction"),
        ("PATHI", "Brain_Expert_Pathi_Active", "Brain_Expert_Pathi_Direction"),
    ]
    for label, ac, dc in specs:
        if ac not in g.columns or dc not in g.columns:
            continue
        active = pd.to_numeric(g[ac], errors="coerce").fillna(0).to_numpy(dtype=float) > 0.5
        direction = pd.to_numeric(g[dc], errors="coerce").fillna(0).to_numpy(dtype=float)
        agree = active & np.isfinite(direction) & (np.sign(direction) == stat_dir) & (np.abs(direction) > 0)
        conflict = active & np.isfinite(direction) & (np.sign(direction) == -stat_dir) & (np.abs(direction) > 0)
        out.append((f"{label}_AGREES_WITH_STAT", agree, "system"))
        out.append((f"{label}_CONFLICTS_WITH_STAT", conflict, "system"))
    if "System_Net_Signal" in g.columns:
        sig = pd.to_numeric(g["System_Net_Signal"], errors="coerce").fillna(0).to_numpy(dtype=float)
        out.append(("SYSTEM_NET_AGREES_WITH_STAT", (np.sign(sig) == stat_dir) & (np.abs(sig) > 0), "system"))
        out.append(("SYSTEM_NET_CONFLICTS_WITH_STAT", (np.sign(sig) == -stat_dir) & (np.abs(sig) > 0), "system"))
    return out


def _predefined_regimes(g: pd.DataFrame, base_p: np.ndarray, eval_mask: np.ndarray) -> List[Tuple[str, np.ndarray, str]]:
    n = len(g)
    p = np.asarray(base_p, dtype=float)
    stat_dir = np.where(p >= 0.5, 1.0, -1.0)
    edge = np.abs(p - 0.5)
    spread = pd.to_numeric(g.get("Consensus_Open_Spread"), errors="coerce").to_numpy(dtype=float)
    total = pd.to_numeric(g.get("Consensus_Open_Total"), errors="coerce").to_numpy(dtype=float)
    selected_spread = np.where(stat_dir > 0, spread, -spread)
    abs_spread = np.abs(selected_spread)
    week = _derive_week(g).to_numpy(dtype=float)

    regs: List[Tuple[str, np.ndarray, str]] = [
        ("ALL_STAT_SELECTIONS", np.ones(n, dtype=bool), "global"),
        ("EDGE_0_TO_1", (edge >= 0.00) & (edge < 0.01), "edge"),
        ("EDGE_1_TO_2", (edge >= 0.01) & (edge < 0.02), "edge"),
        ("EDGE_2_TO_2P5", (edge >= 0.02) & (edge < 0.025), "edge"),
        ("EDGE_2P5_TO_3", (edge >= 0.025) & (edge < 0.03), "edge"),
        ("EDGE_3_TO_4", (edge >= 0.03) & (edge < 0.04), "edge"),
        ("EDGE_4_TO_5", (edge >= 0.04) & (edge < 0.05), "edge"),
        ("EDGE_5_PLUS", edge >= 0.05, "edge"),
        ("STAT_ON_FAVORITE", np.isfinite(selected_spread) & (selected_spread < -0.25), "role"),
        ("STAT_ON_DOG", np.isfinite(selected_spread) & (selected_spread > 0.25), "role"),
        ("STAT_ON_PICKEM", np.isfinite(selected_spread) & (np.abs(selected_spread) <= 0.25), "role"),
        ("SPREAD_0_TO_3", np.isfinite(abs_spread) & (abs_spread <= 3.0), "spread_band"),
        ("SPREAD_GT3_TO_7", np.isfinite(abs_spread) & (abs_spread > 3.0) & (abs_spread <= 7.0), "spread_band"),
        ("SPREAD_GT7_TO_14", np.isfinite(abs_spread) & (abs_spread > 7.0) & (abs_spread <= 14.0), "spread_band"),
        ("SPREAD_GT14", np.isfinite(abs_spread) & (abs_spread > 14.0), "spread_band"),
        ("TOTAL_LT45", np.isfinite(total) & (total < 45.0), "total_band"),
        ("TOTAL_45_TO_55", np.isfinite(total) & (total >= 45.0) & (total <= 55.0), "total_band"),
        ("TOTAL_GT55", np.isfinite(total) & (total > 55.0), "total_band"),
        ("EARLY_SEASON_WK1_4", np.isfinite(week) & (week <= 4), "season_stage"),
        ("MID_SEASON_WK5_9", np.isfinite(week) & (week >= 5) & (week <= 9), "season_stage"),
        ("LATE_SEASON_WK10_PLUS", np.isfinite(week) & (week >= 10), "season_stage"),
    ]

    conf_col = None
    for c in ("Brain_Regime_ConferenceGame", "BigAl_Context_Is_Conference_Game", "Is_Conference_Game", "Conference_Game"):
        if c in g.columns:
            conf_col = c; break
    if conf_col:
        c = pd.to_numeric(g[conf_col], errors="coerce").to_numpy(dtype=float)
        regs.extend([
            ("CONFERENCE_GAME", np.isfinite(c) & (c > 0.5), "conference"),
            ("NONCONFERENCE_GAME", np.isfinite(c) & (c <= 0.5), "conference"),
        ])

    # These are pregame system-state fields when present.  They are attribution only.
    regs.extend(_system_direction_regimes(g, stat_dir))

    # Enforce predefined/controlled scope.
    out = []
    for name, mask, family in regs[:V142_MAX_REGIMES]:
        m = np.asarray(mask, dtype=bool) & np.asarray(eval_mask, dtype=bool)
        out.append((name, m, family))
    return out


def _directional_stats(stat_hit: np.ndarray, season: np.ndarray, mask: np.ndarray) -> Dict[str, object]:
    m = np.asarray(mask, dtype=bool) & np.isfinite(season)
    h = np.asarray(stat_hit, dtype=float)[m]
    sy = np.asarray(season, dtype=float)[m]
    n = int(len(h))
    if n == 0:
        return {"n": 0}
    follow = float(np.mean(h))
    fade = float(1.0 - follow)
    direction = "FOLLOW" if follow >= fade else "FADE"
    chosen_hits = h if direction == "FOLLOW" else (1.0 - h)
    chosen_rate = float(np.mean(chosen_hits))
    chosen_roi = _roi_from_hit(chosen_rate)

    seasons = []
    for s in sorted(int(x) for x in np.unique(sy[np.isfinite(sy)])):
        sm = sy == float(s)
        if int(sm.sum()) < V142_MIN_SEASON_N:
            continue
        fh = float(np.mean(h[sm]))
        ch = fh if direction == "FOLLOW" else 1.0 - fh
        seasons.append({"season": s, "n": int(sm.sum()), "follow_hit": fh,
                        "chosen_hit": ch, "chosen_roi": _roi_from_hit(ch)})

    # Remove the best season in the chosen direction.
    remove_best_hit = np.nan
    min_loso = np.nan
    if len(seasons) >= 2:
        best_season = max(seasons, key=lambda r: r["chosen_hit"])["season"]
        rb = sy != float(best_season)
        if int(rb.sum()) > 0:
            rr = h[rb] if direction == "FOLLOW" else (1.0 - h[rb])
            remove_best_hit = float(np.mean(rr))
        loso = []
        for r in seasons:
            lm = sy != float(r["season"])
            if int(lm.sum()) == 0:
                continue
            lh = h[lm] if direction == "FOLLOW" else (1.0 - h[lm])
            loso.append(float(np.mean(lh)))
        if loso:
            min_loso = float(np.min(loso))

    positive_seasons = int(sum(r["chosen_hit"] > 0.5 for r in seasons))
    profitable_seasons = int(sum(r["chosen_hit"] > V142_BREAK_EVEN for r in seasons))
    season_count = len(seasons)

    candidate = bool(n >= V142_CANDIDATE_N and chosen_rate > V142_BREAK_EVEN)
    stability = bool(
        n >= V142_QUALIFIED_N
        and season_count >= 3
        and chosen_rate >= V142_QUALIFIED_HIT
        and np.isfinite(remove_best_hit) and remove_best_hit > V142_BREAK_EVEN
        and np.isfinite(min_loso) and min_loso > V142_BREAK_EVEN
        and positive_seasons >= math.ceil(0.67 * season_count)
    )
    if stability:
        status = f"{direction}_QUALIFIED"
    elif candidate:
        status = f"{direction}_SHADOW"
    else:
        status = "NEUTRAL"

    return {
        "n": n,
        "follow_hit": follow,
        "follow_roi": _roi_from_hit(follow),
        "fade_hit": fade,
        "fade_roi": _roi_from_hit(fade),
        "direction": direction,
        "chosen_hit": chosen_rate,
        "chosen_roi": chosen_roi,
        "status": status,
        "season_count": season_count,
        "positive_seasons": positive_seasons,
        "profitable_seasons": profitable_seasons,
        "remove_best_hit": remove_best_hit,
        "min_loso": min_loso,
        "season_rows": seasons,
    }


def run_v14_stat_reliability(*, dashboard_module, log_func=print, hard_fail=True):
    """Run the signed V13 STAT reliability study. Research only."""
    try:
        cache = getattr(dashboard_module, "_V1357_SPREAD_RESEARCH_CACHE", {})
        games = cache.get("games") if isinstance(cache, dict) else None
        oof_margin = cache.get("oof_margin") if isinstance(cache, dict) else None
        empirical = getattr(dashboard_module, "_ncaaf_stat_empirical_prob_gt", None)
        if games is None or not isinstance(games, pd.DataFrame) or games.empty or oof_margin is None or empirical is None:
            raise RuntimeError("V13 STAT research cache unavailable; V14.2 must run after NCAAF Spread training")

        g = games.copy()
        g["Season"] = pd.to_numeric(g.get("Season"), errors="coerce")
        g["Game_Date"] = pd.to_datetime(g.get("Game_Date"), errors="coerce", utc=True)
        margin = pd.to_numeric(g.get("Actual_Margin"), errors="coerce")
        spread = pd.to_numeric(g.get("Consensus_Open_Spread"), errors="coerce")
        ats_margin = margin + spread
        y = np.where(ats_margin.notna() & ~np.isclose(ats_margin, 0.0, atol=1e-9),
                     (ats_margin > 0).astype(float), np.nan)
        season = g["Season"].to_numpy(dtype=float)
        seasons = sorted(int(x) for x in g["Season"].dropna().unique())
        if len(seasons) < 3:
            raise RuntimeError(f"V14.2 needs >=3 seasons, found {seasons}")

        base_p = _v13_control_probs(g, oof_margin, seasons[1:], empirical)
        eval_mask = np.isfinite(y) & np.isfinite(base_p) & np.isfinite(season)
        if int(eval_mask.sum()) < 1000:
            raise RuntimeError(f"V14.2 insufficient OOF STAT rows={int(eval_mask.sum())}")
        key_name, key_rows, key_unique = _one_physical_game_audit(g, eval_mask)

        stat_side = base_p >= 0.5
        stat_hit = np.where(stat_side, y == 1.0, y == 0.0).astype(float)
        overall_follow = float(np.mean(stat_hit[eval_mask]))
        overall_fade = 1.0 - overall_follow
        log_func(
            f"[V14.2-PREFLIGHT] status=PASS source_tag={V142_SOURCE_TAG} rows={len(g)} "
            f"oof_eval_rows={int(eval_mask.sum())} seasons={seasons} key={key_name} key_rows={key_rows} "
            f"key_unique={key_unique} one_physical_game=TRUE production_authority=0"
        )
        log_func(
            f"[V14.2-STAT-BASELINE] n={int(eval_mask.sum())} follow_hit={overall_follow:.4f} "
            f"follow_roi={_roi_from_hit(overall_follow):+.4f} fade_hit={overall_fade:.4f} "
            f"fade_roi={_roi_from_hit(overall_fade):+.4f} break_even={V142_BREAK_EVEN:.6f}"
        )

        regimes = _predefined_regimes(g, base_p, eval_mask)
        results = []
        for name, mask, family in regimes:
            n = int(mask.sum())
            if n < V142_MIN_EVAL_N:
                log_func(f"[V14.2-RELIABILITY-SKIP] regime={name} family={family} n={n} reason=N_LT_{V142_MIN_EVAL_N}")
                continue
            r = _directional_stats(stat_hit, season, mask)
            r.update({"regime": name, "family": family})
            results.append(r)
            log_func(
                f"[V14.2-RELIABILITY] regime={name} family={family} n={r['n']} "
                f"follow_hit={r['follow_hit']:.4f} follow_roi={r['follow_roi']:+.4f} "
                f"fade_hit={r['fade_hit']:.4f} fade_roi={r['fade_roi']:+.4f} "
                f"chosen={r['direction']} chosen_hit={r['chosen_hit']:.4f} chosen_roi={r['chosen_roi']:+.4f} "
                f"status={r['status']} seasons={r['season_count']} positive_seasons={r['positive_seasons']} "
                f"profitable_seasons={r['profitable_seasons']} remove_best_hit={r['remove_best_hit']:.4f} "
                f"min_loso={r['min_loso']:.4f} production_authority=0"
            )
            for sr in r["season_rows"]:
                log_func(
                    f"[V14.2-RELIABILITY-SEASON] regime={name} chosen={r['direction']} season={sr['season']} "
                    f"n={sr['n']} follow_hit={sr['follow_hit']:.4f} chosen_hit={sr['chosen_hit']:.4f} "
                    f"chosen_roi={sr['chosen_roi']:+.4f}"
                )

        if len(results) < 12:
            raise RuntimeError(f"V14.2 too few evaluated predefined regimes={len(results)}")

        qualified = [r for r in results if r["status"].endswith("_QUALIFIED")]
        fade_qualified = [r for r in qualified if r["direction"] == "FADE"]
        follow_qualified = [r for r in qualified if r["direction"] == "FOLLOW"]
        shadow = [r for r in results if r["status"].endswith("_SHADOW")]
        for r in sorted(qualified, key=lambda x: (-x["chosen_hit"], -x["n"])):
            log_func(
                f"[V14.2-QUALIFIED] regime={r['regime']} family={r['family']} action={r['direction']} "
                f"n={r['n']} hit={r['chosen_hit']:.4f} roi={r['chosen_roi']:+.4f} "
                f"remove_best_hit={r['remove_best_hit']:.4f} min_loso={r['min_loso']:.4f} "
                f"shadow_only=TRUE production_authority=0"
            )

        # Explicitly audit the requested bad-signal -> opposite-direction transformation.
        bad_to_good = [r for r in results if r["direction"] == "FADE" and r["fade_hit"] > V142_BREAK_EVEN]
        for r in sorted(bad_to_good, key=lambda x: (-x["fade_hit"], -x["n"])):
            log_func(
                f"[V14.2-BAD-TO-GOOD] regime={r['regime']} original_stat_ats={r['follow_hit']:.4f} "
                f"opposite_ats={r['fade_hit']:.4f} opposite_roi={r['fade_roi']:+.4f} status={r['status']} "
                f"rule=FADE_ONLY_IF_STABILITY_SURVIVES production_authority=0"
            )

        log_func(
            f"[V14.2-COMPARISON-CONTRACT] status=PASS comparison_basis=SAME_EXACT_OOF_STAT_ROWS "
            f"row_count={int(eval_mask.sum())} no_cross_row_beats_stat=TRUE one_physical_game=TRUE"
        )
        log_func(
            f"[V14.2-CONTRACT] status=PASS predefined_regimes={len(regimes)} evaluated_regimes={len(results)} "
            f"qualified={len(qualified)} follow_qualified={len(follow_qualified)} fade_qualified={len(fade_qualified)} "
            f"shadow={len(shadow)} bad_to_good={len(bad_to_good)} break_even={V142_BREAK_EVEN:.6f} "
            f"predictor=V13_STAT_FROZEN research_mode=SIGNED_RELIABILITY systems_weighted=FALSE "
            f"production_authority=0"
        )
        return {
            "status": "PASS", "version": V142_SOURCE_TAG, "results": results,
            "qualified": qualified, "fade_qualified": fade_qualified,
            "follow_qualified": follow_qualified, "bad_to_good": bad_to_good,
            "production_authority": 0,
        }
    except Exception as e:
        log_func(f"[V14.2-CONTRACT] status=FAILED error={type(e).__name__}:{e} production_authority=0")
        if hard_fail:
            raise
        return {"status": "FAILED", "error": f"{type(e).__name__}:{e}", "production_authority": 0}
