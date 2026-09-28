"""NCAAF V14.3 STAT Reliability Hardening + Signed Decision research layer.

V14.3 does not train a replacement prediction model. V13 STAT remains frozen.
The purpose is to determine, on season-forward OOF predictions, when STAT should
be followed, when a persistently bad STAT regime should be faded, and when an
unstable/negative regime should be suppressed rather than bet.

Research discipline:
* Direction reliability uses every available season-forward OOF point edge (2023+).
* Probability-edge buckets use only seasons with a valid historical probability
  calibration (normally 2024+); they are explicitly a smaller universe.
* Rule discovery is restricted to completed 2023-2025 history.
* 2026 is CURRENT_SEASON_CONFIRMATION only; it is excluded from discovery because
  we have already observed 2026 results in prior research runs.
* Only games after 2026-09-28 may be labeled genuinely prospective.
* Big Al / Pathi agreement is reconstructed from the historical system occurrence
  ledger and joined to the exact STAT-selected game side.
* No reliability finding has production authority.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import math
import re

import numpy as np
import pandas as pd

V143_SOURCE_TAG = "v14.3-stat-reliability-hardening"
V143_BREAK_EVEN = 110.0 / 210.0
V143_DISCOVERY_END_SEASON = 2025
V143_CONFIRMATION_SEASON = 2026
V143_RESEARCH_FREEZE_UTC = pd.Timestamp("2026-09-28T23:59:59Z")
V143_MIN_EVAL_N = 50
V143_CANDIDATE_N = 100
V143_STRONG_N = 150
V143_MIN_SEASON_N = 20
V143_STRONG_HIT = 0.535
V143_MAX_REGIMES = 60


def _num(v, index=None):
    if isinstance(v, pd.Series):
        return pd.to_numeric(v, errors="coerce")
    if index is None:
        return pd.Series(pd.to_numeric(v, errors="coerce"))
    return pd.to_numeric(pd.Series(v, index=index), errors="coerce")


def _roi(hit: float) -> float:
    if not np.isfinite(hit):
        return np.nan
    return float(hit * (100.0 / 110.0) - (1.0 - hit))


def _tok(x) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def _derive_week(g: pd.DataFrame) -> pd.Series:
    for c in ("Week", "Week_Number", "WeekNum", "BigAl_Context_Week_Number"):
        if c in g.columns:
            s = pd.to_numeric(g[c], errors="coerce")
            if int(s.notna().sum()) >= max(100, int(0.25 * len(g))):
                return s
    d = pd.to_datetime(g.get("Game_Date"), errors="coerce", utc=True)
    season = pd.to_numeric(g.get("Season"), errors="coerce")
    out = pd.Series(np.nan, index=g.index, dtype="float64")
    for sy in sorted(season.dropna().unique()):
        m = season.eq(sy) & d.notna()
        if not m.any():
            continue
        start = d[m].min().normalize()
        out.loc[m] = ((d.loc[m] - start).dt.days // 7 + 1).astype(float)
    return out


def _physical_game_audit(g: pd.DataFrame, mask: np.ndarray) -> Tuple[str, int, int]:
    d = g.loc[np.asarray(mask, dtype=bool)].copy()
    if "Source_Game_ID" in d.columns:
        k = d["Source_Game_ID"].astype(str).str.strip()
        valid = k.ne("") & k.ne("nan") & k.ne("None")
        if valid.any():
            n = int(valid.sum()); u = int(k[valid].nunique())
            if n != u:
                raise RuntimeError(f"V14.3 duplicate physical games rows={n} unique={u}")
            return "Source_Game_ID", n, u
    k = (pd.to_numeric(d.get("Season"), errors="coerce").astype("Int64").astype(str) + "|" +
         pd.to_datetime(d.get("Game_Date"), errors="coerce", utc=True).dt.strftime("%Y-%m-%d").fillna("") + "|" +
         d.get("Team_Norm", pd.Series("", index=d.index)).astype(str) + "|" +
         d.get("Opponent_Norm", pd.Series("", index=d.index)).astype(str))
    n = len(k); u = int(k.nunique())
    if n != u:
        raise RuntimeError(f"V14.3 duplicate physical games composite rows={n} unique={u}")
    return "SEASON_DATE_TEAMS", n, u


def _calibrated_stat_probs(g: pd.DataFrame, oof_margin, empirical_prob_fn) -> np.ndarray:
    """Reconstruct historical V13 cover probabilities without inventing a first-fold calibration."""
    om = np.asarray(oof_margin, dtype=float)
    out = np.full(len(g), np.nan, dtype=float)
    season = _num(g.get("Season"), g.index).to_numpy(dtype=float)
    actual = _num(g.get("Actual_Margin"), g.index).to_numpy(dtype=float)
    spread = _num(g.get("Consensus_Open_Spread"), g.index).to_numpy(dtype=float)
    seasons = sorted(int(x) for x in pd.Series(season).dropna().unique())
    for sy in seasons:
        # OOF residuals from earlier seasons only. This intentionally leaves the
        # first OOF season uncalibrated if no earlier OOF residual distribution exists.
        hist = np.isfinite(season) & (season < sy) & np.isfinite(actual) & np.isfinite(om)
        test = np.isfinite(season) & (season == sy) & np.isfinite(om) & np.isfinite(spread)
        if int(hist.sum()) < 100 or int(test.sum()) == 0:
            continue
        resid = (actual - om)[hist]
        out[test] = empirical_prob_fn(-(om[test] + spread[test]), resid)
    return out


def _stat_side_arrays(g: pd.DataFrame, oof_margin) -> Dict[str, np.ndarray]:
    om = np.asarray(oof_margin, dtype=float)
    spread = _num(g.get("Consensus_Open_Spread"), g.index).to_numpy(dtype=float)
    edge_points = om + spread
    stat_home = edge_points >= 0.0
    home = g.get("Team_Norm", pd.Series("", index=g.index)).astype(str).to_numpy(dtype=object)
    away = g.get("Opponent_Norm", pd.Series("", index=g.index)).astype(str).to_numpy(dtype=object)
    selected_team = np.where(stat_home, home, away)
    selected_opp = np.where(stat_home, away, home)
    selected_spread = np.where(stat_home, spread, -spread)
    return {
        "edge_points": edge_points,
        "stat_home": stat_home,
        "selected_team": selected_team,
        "selected_opp": selected_opp,
        "selected_spread": selected_spread,
    }


def _system_occurrence_masks(g: pd.DataFrame, side: Dict[str, np.ndarray], system_history: dict, log_func=print) -> Dict[str, np.ndarray]:
    """Attach directional Big Al / Pathi occurrences to exact STAT-selected sides."""
    n = len(g)
    dates = pd.to_datetime(g.get("Game_Date"), errors="coerce", utc=True)
    seasons = pd.to_numeric(g.get("Season"), errors="coerce")
    sel_keys = []
    opp_keys = []
    for i in range(n):
        sy = seasons.iloc[i] if i < len(seasons) else np.nan
        dt = dates.iloc[i] if i < len(dates) else pd.NaT
        ds = dt.strftime("%Y-%m-%d") if pd.notna(dt) else ""
        yr = int(sy) if np.isfinite(sy) else None
        if yr is None:
            sel_keys.append(""); opp_keys.append(""); continue
        st = _tok(side["selected_team"][i]); op = _tok(side["selected_opp"][i])
        sel_keys.append(f"{yr}|{ds}|{st}|{op}")
        opp_keys.append(f"{yr}|{ds}|{op}|{st}")
    sel_keys = np.asarray(sel_keys, dtype=object); opp_keys = np.asarray(opp_keys, dtype=object)

    family_occ = {"BigAl": set(), "Pathi": set()}
    named_occ: Dict[str, set] = {}
    for name, rec in (system_history or {}).items():
        if not isinstance(rec, dict) or rec.get("role") != "directional" or not rec.get("source_validation_pass", True):
            continue
        fam = str(rec.get("family", ""))
        if fam not in family_occ:
            continue
        occ = {str(o.get("key")) for o in (rec.get("occurrences") or []) if isinstance(o, dict) and o.get("key")}
        if not occ:
            continue
        family_occ[fam].update(occ)
        named_occ[str(name)] = occ

    out: Dict[str, np.ndarray] = {}
    for fam in ("BigAl", "Pathi"):
        occ = family_occ[fam]
        agree = np.fromiter((k in occ for k in sel_keys), dtype=bool, count=n) if occ else np.zeros(n, bool)
        conflict = np.fromiter((k in occ for k in opp_keys), dtype=bool, count=n) if occ else np.zeros(n, bool)
        mixed = agree & conflict
        out[f"{fam.upper()}_AGREE"] = agree & ~mixed
        out[f"{fam.upper()}_CONFLICT"] = conflict & ~mixed
        out[f"{fam.upper()}_MIXED"] = mixed
        log_func(
            f"[V14.3-SYSTEM-JOIN] family={fam.upper()} occurrence_keys={len(occ)} "
            f"agree_games={int((agree & ~mixed).sum())} conflict_games={int((conflict & ~mixed).sum())} "
            f"mixed_games={int(mixed.sum())} source=HISTORICAL_DIRECTIONAL_OCCURRENCE_LEDGER"
        )

    any_agree = out.get("BIGAL_AGREE", np.zeros(n,bool)) | out.get("PATHI_AGREE", np.zeros(n,bool))
    any_conf = out.get("BIGAL_CONFLICT", np.zeros(n,bool)) | out.get("PATHI_CONFLICT", np.zeros(n,bool))
    out["SYSTEM_ANY_AGREE"] = any_agree & ~any_conf
    out["SYSTEM_ANY_CONFLICT"] = any_conf & ~any_agree
    out["SYSTEM_MIXED"] = (any_agree & any_conf) | out.get("BIGAL_MIXED", np.zeros(n,bool)) | out.get("PATHI_MIXED", np.zeros(n,bool))
    return out


def _regimes(g: pd.DataFrame, side: Dict[str,np.ndarray], probs: np.ndarray, direction_mask: np.ndarray,
             prob_mask: np.ndarray, system_masks: Dict[str,np.ndarray]) -> List[dict]:
    n = len(g)
    spread_sel = np.asarray(side["selected_spread"], dtype=float)
    abs_spread = np.abs(spread_sel)
    total = _num(g.get("Consensus_Open_Total"), g.index).to_numpy(dtype=float)
    week = _derive_week(g).to_numpy(dtype=float)
    edge = np.abs(np.asarray(probs, dtype=float) - 0.5)
    large = np.isfinite(abs_spread) & (abs_spread > 7.0)
    fav = np.isfinite(spread_sel) & (spread_sel < -0.25)
    dog = np.isfinite(spread_sel) & (spread_sel > 0.25)
    early = np.isfinite(week) & (week <= 4)
    mid = np.isfinite(week) & (week >= 5) & (week <= 9)
    late = np.isfinite(week) & (week >= 10)

    regs = [
        ("ALL_STAT_SELECTIONS", np.ones(n,bool), "global", False),
        ("STAT_ON_FAVORITE", fav, "role", False),
        ("STAT_ON_DOG", dog, "role", False),
        ("SPREAD_0_TO_3", np.isfinite(abs_spread)&(abs_spread<=3), "spread_band", False),
        ("SPREAD_GT3_TO_7", np.isfinite(abs_spread)&(abs_spread>3)&(abs_spread<=7), "spread_band", False),
        ("SPREAD_GT7_TO_14", np.isfinite(abs_spread)&(abs_spread>7)&(abs_spread<=14), "spread_band", False),
        ("SPREAD_GT14", np.isfinite(abs_spread)&(abs_spread>14), "spread_band", False),
        ("SPREAD_GT7", large, "spread_parent", False),
        ("TOTAL_LT45", np.isfinite(total)&(total<45), "total_band", False),
        ("TOTAL_45_TO_55", np.isfinite(total)&(total>=45)&(total<=55), "total_band", False),
        ("TOTAL_GT55", np.isfinite(total)&(total>55), "total_band", False),
        ("EARLY_SEASON_WK1_4", early, "season_stage", False),
        ("MID_SEASON_WK5_9", mid, "season_stage", False),
        ("LATE_SEASON_WK10_PLUS", late, "season_stage", False),
        # Evidence-led interactions from V14.2 large-spread finding.
        ("SPREAD_GT7__STAT_FAVORITE", large&fav, "controlled_interaction", False),
        ("SPREAD_GT7__STAT_DOG", large&dog, "controlled_interaction", False),
        ("SPREAD_GT7__EARLY", large&early, "controlled_interaction", False),
        ("SPREAD_GT7__MID", large&mid, "controlled_interaction", False),
        ("SPREAD_GT7__LATE", large&late, "controlled_interaction", False),
        # Probability-edge buckets: smaller calibrated universe, explicitly labeled.
        ("EDGE_0_TO_1", (edge>=0)&(edge<.01), "probability_edge", True),
        ("EDGE_1_TO_2", (edge>=.01)&(edge<.02), "probability_edge", True),
        ("EDGE_2_TO_2P5", (edge>=.02)&(edge<.025), "probability_edge", True),
        ("EDGE_2P5_TO_3", (edge>=.025)&(edge<.03), "probability_edge", True),
        ("EDGE_3_TO_4", (edge>=.03)&(edge<.04), "probability_edge", True),
        ("EDGE_4_TO_5", (edge>=.04)&(edge<.05), "probability_edge", True),
        ("EDGE_5_PLUS", edge>=.05, "probability_edge", True),
    ]
    for nm, mask in system_masks.items():
        regs.append((nm, np.asarray(mask,bool), "system_context", False))
        if nm in ("SYSTEM_ANY_AGREE","SYSTEM_ANY_CONFLICT"):
            regs.append((f"SPREAD_GT7__{nm}", large & np.asarray(mask,bool), "controlled_interaction", False))

    out=[]
    for name, mask, family, prob_only in regs[:V143_MAX_REGIMES]:
        base = prob_mask if prob_only else direction_mask
        out.append({"name":name,"family":family,"prob_only":prob_only,"mask":np.asarray(mask,bool)&np.asarray(base,bool)})
    return out


def _stats(stat_hit: np.ndarray, season: np.ndarray, mask: np.ndarray) -> Dict[str,object]:
    m = np.asarray(mask,bool) & np.isfinite(season)
    h = np.asarray(stat_hit,float)[m]; sy = np.asarray(season,float)[m]
    n=int(len(h))
    if n==0: return {"n":0}
    follow=float(np.mean(h)); fade=1-follow
    direction="FOLLOW" if follow>=fade else "FADE"
    ch=h if direction=="FOLLOW" else 1-h
    chosen=float(np.mean(ch))
    rows=[]
    for s in sorted(int(x) for x in np.unique(sy[np.isfinite(sy)])):
        sm=sy==float(s)
        if int(sm.sum())<V143_MIN_SEASON_N: continue
        fh=float(np.mean(h[sm])); hh=fh if direction=="FOLLOW" else 1-fh
        rows.append({"season":s,"n":int(sm.sum()),"follow_hit":fh,"chosen_hit":hh,"chosen_roi":_roi(hh)})
    rb=np.nan; min_loso=np.nan
    if len(rows)>=2:
        best=max(rows,key=lambda r:r["chosen_hit"])["season"]
        mm=sy!=float(best)
        if mm.any(): rb=float(np.mean((h if direction=="FOLLOW" else 1-h)[mm]))
        loso=[]
        for r in rows:
            lm=sy!=float(r["season"])
            if lm.any(): loso.append(float(np.mean((h if direction=="FOLLOW" else 1-h)[lm])))
        if loso: min_loso=float(np.min(loso))
    pos=int(sum(r["chosen_hit"]>.5 for r in rows)); prof=int(sum(r["chosen_hit"]>V143_BREAK_EVEN for r in rows))
    candidate=bool(n>=V143_CANDIDATE_N and chosen>V143_BREAK_EVEN)
    strong=bool(n>=V143_STRONG_N and len(rows)>=3 and chosen>=V143_STRONG_HIT and np.isfinite(rb) and rb>V143_BREAK_EVEN and np.isfinite(min_loso) and min_loso>V143_BREAK_EVEN and pos>=math.ceil(.67*len(rows)))
    if strong:
        decision=f"{direction}_STRONG_SHADOW"
    elif candidate and direction=="FOLLOW":
        decision="FOLLOW_SHADOW"
    elif direction=="FADE" and follow<V143_BREAK_EVEN:
        # A negative STAT regime without robust opposite-side evidence is useful
        # as a bet-avoidance signal, not as an automatic opposite-side wager.
        decision="SUPPRESS"
    else:
        decision="NEUTRAL"
    return {"n":n,"follow_hit":follow,"follow_roi":_roi(follow),"fade_hit":fade,"fade_roi":_roi(fade),
            "direction":direction,"chosen_hit":chosen,"chosen_roi":_roi(chosen),"season_count":len(rows),
            "positive_seasons":pos,"profitable_seasons":prof,"remove_best_hit":rb,"min_loso":min_loso,
            "strong":strong,"candidate":candidate,"decision":decision,"season_rows":rows}


def _evaluate_fixed_direction(stat_hit, mask, direction):
    h=np.asarray(stat_hit,float)[np.asarray(mask,bool)]
    if len(h)==0: return {"n":0,"hit":np.nan,"roi":np.nan}
    z=h if direction=="FOLLOW" else 1-h
    return {"n":int(len(z)),"hit":float(np.mean(z)),"roi":_roi(float(np.mean(z)))}


def _policy_action_from_prior(stat_hit, season, mask, prior_seasons) -> Dict[str,object]:
    pm=np.asarray(mask,bool)&np.isin(np.asarray(season,float),np.asarray(prior_seasons,float))
    r=_stats(stat_hit,season,pm)
    if r.get("n",0)<V143_CANDIDATE_N:
        return {**r,"policy_action":"PASS","reason":"INSUFFICIENT_PRIOR_N"}
    # Chronological replay is deliberately stricter than descriptive shadow labels.
    # With >=3 prior seasons, require the full strong-stability gate. With only two
    # prior seasons (the 2025 replay), require both seasons above break-even AND a
    # 54% aggregate directional hit rate so the replay does not manufacture bets
    # from marginal two-season noise.
    rows=r.get("season_rows") or []
    replay_ready = bool(
        (len(rows)>=3 and r.get("strong",False)) or
        (len(rows)==2 and int(r.get("n",0))>=200 and r.get("chosen_hit",0)>=0.54 and all(x["chosen_hit"]>V143_BREAK_EVEN for x in rows))
    )
    if replay_ready:
        if r.get("direction")=="FADE":
            return {**r,"policy_action":"FADE","reason":"PRIOR_STABILITY_GATE_PASS"}
        return {**r,"policy_action":"FOLLOW","reason":"PRIOR_STABILITY_GATE_PASS"}
    if r.get("direction")=="FADE" and r.get("follow_hit",1)>=0 and r.get("follow_hit",1)<V143_BREAK_EVEN:
        return {**r,"policy_action":"SUPPRESS","reason":"NEGATIVE_STAT_NOT_STABLE_ENOUGH_TO_FADE"}
    return {**r,"policy_action":"PASS","reason":"PRIOR_STABILITY_NOT_PROVEN"}


def run_v14_stat_reliability(*, dashboard_module, log_func=print, hard_fail=True):
    try:
        cache=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{})
        games=cache.get("games") if isinstance(cache,dict) else None
        oof_margin=cache.get("oof_margin") if isinstance(cache,dict) else None
        empirical=getattr(dashboard_module,"_ncaaf_stat_empirical_prob_gt",None)
        system_history=getattr(dashboard_module,"_V143_SYSTEM_HISTORY_CACHE",{})
        if games is None or not isinstance(games,pd.DataFrame) or games.empty or oof_margin is None or empirical is None:
            raise RuntimeError("V13 STAT OOF cache unavailable")
        g=games.copy(); g["Season"]=_num(g.get("Season"),g.index); g["Game_Date"]=pd.to_datetime(g.get("Game_Date"),errors="coerce",utc=True)
        season=g["Season"].to_numpy(dtype=float)
        margin=_num(g.get("Actual_Margin"),g.index).to_numpy(dtype=float)
        spread=_num(g.get("Consensus_Open_Spread"),g.index).to_numpy(dtype=float)
        ats_margin=margin+spread
        y=np.where(np.isfinite(ats_margin)&~np.isclose(ats_margin,0,atol=1e-9),(ats_margin>0).astype(float),np.nan)
        side=_stat_side_arrays(g,oof_margin)
        direction_mask=np.isfinite(y)&np.isfinite(side["edge_points"])&np.isfinite(season)
        key_name,key_rows,key_unique=_physical_game_audit(g,direction_mask)
        stat_hit=np.where(side["stat_home"],y==1.0,y==0.0).astype(float)
        probs=_calibrated_stat_probs(g,oof_margin,empirical)
        prob_mask=direction_mask&np.isfinite(probs)

        years=sorted(int(x) for x in pd.Series(season[direction_mask]).dropna().unique())
        if 2023 not in years:
            raise RuntimeError(f"V14.3 expected 2023 direction OOF coverage; got seasons={years}")
        if int(direction_mask.sum())<2800:
            raise RuntimeError(f"V14.3 expected near-full STAT OOF direction coverage; rows={int(direction_mask.sum())}")

        log_func(f"[V14.3-PREFLIGHT] status=PASS source_tag={V143_SOURCE_TAG} direction_oof_rows={int(direction_mask.sum())} probability_oof_rows={int(prob_mask.sum())} seasons={years} key={key_name} key_rows={key_rows} key_unique={key_unique} one_physical_game=TRUE production_authority=0")
        log_func(f"[V14.3-UNIVERSE-CONTRACT] direction_universe=SEASON_FORWARD_POINT_EDGE_2023_PLUS probability_edge_universe=HISTORICALLY_CALIBRATED_PROBABILITY_SUBSET no_2023_probability_fabrication=TRUE")

        # System-state repair: occurrence ledger comes from leakage-safe historical
        # Big Al/Pathi rule construction. This is context only; it never changes STAT.
        system_masks=_system_occurrence_masks(g,side,system_history,log_func=log_func)
        if int(system_masks.get("BIGAL_AGREE",np.zeros(len(g),bool)).sum()+system_masks.get("BIGAL_CONFLICT",np.zeros(len(g),bool)).sum())==0:
            raise RuntimeError("V14.3 Big Al historical system join produced zero mapped games")
        if int(system_masks.get("PATHI_AGREE",np.zeros(len(g),bool)).sum()+system_masks.get("PATHI_CONFLICT",np.zeros(len(g),bool)).sum())==0:
            raise RuntimeError("V14.3 Pathi historical system join produced zero mapped games")

        regs=_regimes(g,side,probs,direction_mask,prob_mask,system_masks)
        discovery_years=[y for y in years if y<=V143_DISCOVERY_END_SEASON]
        disc_base=direction_mask&np.isin(season,np.asarray(discovery_years,float))
        disc_follow=float(np.mean(stat_hit[disc_base]))
        log_func(f"[V14.3-DISCOVERY-BASELINE] seasons={discovery_years} n={int(disc_base.sum())} follow_hit={disc_follow:.4f} follow_roi={_roi(disc_follow):+.4f} rule_discovery_uses_2026=FALSE")

        results=[]
        for z in regs:
            name,family=z["name"],z["family"]
            disc_mask=z["mask"]&np.isin(season,np.asarray(discovery_years,float))
            if int(disc_mask.sum())<V143_MIN_EVAL_N:
                log_func(f"[V14.3-RELIABILITY-SKIP] regime={name} family={family} discovery_n={int(disc_mask.sum())} reason=N_LT_{V143_MIN_EVAL_N}")
                continue
            r=_stats(stat_hit,season,disc_mask); r.update({"regime":name,"family":family,"prob_only":z["prob_only"]})
            results.append(r)
            log_func(f"[V14.3-RELIABILITY] regime={name} family={family} universe={'PROBABILITY' if z['prob_only'] else 'DIRECTION'} discovery_n={r['n']} follow_hit={r['follow_hit']:.4f} follow_roi={r['follow_roi']:+.4f} fade_hit={r['fade_hit']:.4f} fade_roi={r['fade_roi']:+.4f} chosen={r['direction']} chosen_hit={r['chosen_hit']:.4f} chosen_roi={r['chosen_roi']:+.4f} decision={r['decision']} seasons={r['season_count']} positive_seasons={r['positive_seasons']} profitable_seasons={r['profitable_seasons']} remove_best_hit={r['remove_best_hit']:.4f} min_loso={r['min_loso']:.4f} production_authority=0")
            for sr in r["season_rows"]:
                log_func(f"[V14.3-RELIABILITY-SEASON] regime={name} chosen={r['direction']} season={sr['season']} n={sr['n']} follow_hit={sr['follow_hit']:.4f} chosen_hit={sr['chosen_hit']:.4f} chosen_roi={sr['chosen_roi']:+.4f} discovery_component=TRUE")

            # 2026 is excluded from selection and reported separately.
            cm=z["mask"]&(season==float(V143_CONFIRMATION_SEASON))
            conf=_evaluate_fixed_direction(stat_hit,cm,r["direction"])
            if conf["n"]>0:
                log_func(f"[V14.3-CURRENT-SEASON-CONFIRMATION] regime={name} frozen_action={r['direction']} season=2026 n={conf['n']} hit={conf['hit']:.4f} roi={conf['roi']:+.4f} used_for_discovery=FALSE independent_prospective=FALSE")
                r["confirmation_2026"]=conf

            # Truly prospective means after the declared research freeze date.
            post=z["mask"]&(g["Game_Date"].gt(V143_RESEARCH_FREEZE_UTC).to_numpy())
            prospect=_evaluate_fixed_direction(stat_hit,post,r["direction"])
            if prospect["n"]>0:
                log_func(f"[V14.3-PROSPECTIVE] regime={name} frozen_action={r['direction']} after=2026-09-28 n={prospect['n']} hit={prospect['hit']:.4f} roi={prospect['roi']:+.4f} genuinely_post_freeze=TRUE")

        if len(results)<15:
            raise RuntimeError(f"V14.3 too few evaluated regimes={len(results)}")

        # Historical decision replay: use only information that existed before each target season.
        by_name={r["regime"]:r for r in results}
        for z in regs:
            if z["name"] not in by_name: continue
            for target in (2025,2026):
                prior=[y for y in years if y<target]
                prior=[y for y in prior if y>=2023]
                pol=_policy_action_from_prior(stat_hit,season,z["mask"],prior)
                tm=z["mask"]&(season==float(target))
                act=pol.get("policy_action","PASS")
                if act in ("FOLLOW","FADE"):
                    test=_evaluate_fixed_direction(stat_hit,tm,act)
                else:
                    test={"n":int(tm.sum()),"hit":np.nan,"roi":np.nan}
                log_func(f"[V14.3-POLICY-REPLAY] regime={z['name']} decision_season={target} prior_seasons={prior} train_n={int(pol.get('n',0) or 0)} action={act} reason={pol.get('reason')} test_n={test['n']} test_hit={test['hit']:.4f} test_roi={test['roi']:+.4f} chronological=TRUE")

        strong=[r for r in results if r["decision"].endswith("STRONG_SHADOW")]
        suppress=[r for r in results if r["decision"]=="SUPPRESS"]
        bad_to_good=[r for r in results if r["direction"]=="FADE" and r["fade_hit"]>V143_BREAK_EVEN]
        for r in sorted(strong,key=lambda x:(-x["chosen_hit"],-x["n"])):
            c=r.get("confirmation_2026",{})
            log_func(f"[V14.3-STRONG-SHADOW] regime={r['regime']} family={r['family']} action={r['direction']} discovery_n={r['n']} discovery_hit={r['chosen_hit']:.4f} discovery_roi={r['chosen_roi']:+.4f} remove_best={r['remove_best_hit']:.4f} min_loso={r['min_loso']:.4f} confirm_2026_n={int(c.get('n',0) or 0)} confirm_2026_hit={float(c.get('hit',np.nan)):.4f} production_authority=0")
        for r in bad_to_good:
            log_func(f"[V14.3-BAD-TO-GOOD] regime={r['regime']} original_stat_ats={r['follow_hit']:.4f} opposite_ats={r['fade_hit']:.4f} opposite_roi={r['fade_roi']:+.4f} decision={r['decision']} auto_fade=FALSE production_authority=0")
        for r in suppress:
            log_func(f"[V14.3-SUPPRESS] regime={r['regime']} stat_ats={r['follow_hit']:.4f} opposite_ats={r['fade_hit']:.4f} reason=STAT_NEGATIVE_BUT_FADE_NOT_ROBUST_ENOUGH no_bet_action=TRUE")

        # 2026 leakage/shift audit. OOF source is structurally season-forward; compare
        # magnitude/coverage to prior seasons so a broad current-season jump is visible.
        for sy in [y for y in years if y>=2023]:
            m=direction_mask&(season==float(sy)); ae=np.abs(side["edge_points"][m])
            log_func(f"[V14.3-SEASON-FORENSIC] season={sy} oof_rows={int(m.sum())} mean_abs_point_edge={float(np.nanmean(ae)):.4f} median_abs_point_edge={float(np.nanmedian(ae)):.4f} stat_hit={float(np.mean(stat_hit[m])):.4f} source=SEASON_FORWARD_OOF_POINT_EDGE final_refit_used=FALSE")

        log_func(f"[V14.3-COMPARISON-CONTRACT] status=PASS direction_rows={int(direction_mask.sum())} probability_rows={int(prob_mask.sum())} exact_physical_game_dedup=TRUE discovery_seasons={discovery_years} current_season_confirmation=2026 current_season_used_for_rule_selection=FALSE")
        log_func(f"[V14.3-PROSPECTIVE-CONTRACT] freeze_utc={V143_RESEARCH_FREEZE_UTC.isoformat()} games_after_freeze={int((direction_mask & g['Game_Date'].gt(V143_RESEARCH_FREEZE_UTC).to_numpy()).sum())} rule=ONLY_POST_FREEZE_GAMES_MAY_BE_CALLED_PROSPECTIVE")
        log_func(f"[V14.3-CONTRACT] status=PASS predictor=V13_STAT_FROZEN evaluated_regimes={len(results)} strong_shadow={len(strong)} suppress={len(suppress)} bad_to_good={len(bad_to_good)} system_context_attached=TRUE systems_weighted=FALSE replacement_models=NONE production_authority=0")
        return {"status":"PASS","version":V143_SOURCE_TAG,"results":results,"strong_shadow":strong,"suppress":suppress,"bad_to_good":bad_to_good,"production_authority":0}
    except Exception as e:
        log_func(f"[V14.3-CONTRACT] status=FAILED error={type(e).__name__}:{e} production_authority=0")
        if hard_fail: raise
        return {"status":"FAILED","error":f"{type(e).__name__}:{e}","production_authority":0}
