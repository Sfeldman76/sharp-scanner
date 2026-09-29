"""SIBLING_MARKET_EDGE_RESEARCH_V1

Independent NCAAF H2H and Totals edge research lanes.

This module intentionally does NOT alter the frozen Spread policy. It consumes the
same leakage-safe season-forward STAT cache and the already-built System Miner
registry, but evaluates H2H and Totals under market-specific economics.

H2H north star: beat the de-vigged opening moneyline probability / opening price.
Totals north star: beat the opening total with stable O/U accuracy and signed
market error. High favorite win rate is never treated as an H2H edge by itself.

All outputs are research/shadow only; production authority remains 0.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple
import math
import numpy as np
import pandas as pd

SIBLING_MARKET_EDGE_RESEARCH_V1_SOURCE_TAG = "sibling-market-edge-research-v1-h2h-totals-independent"
DISCOVERY_SEASONS = (2023, 2024, 2025)
CONFIRM_SEASON = 2026
BREAK_EVEN_110 = 110.0 / 210.0

# These thresholds are predeclared diagnostics. Only the established primary
# thresholds (5 percentage points H2H and 4 total points) can become a STAT edge
# candidate in V1; the others are sensitivity checks, not an optimization grid.
H2H_STAT_GAP_THRESHOLDS = (0.025, 0.05, 0.075, 0.10)
H2H_PRIMARY_GAP = 0.05
TOTAL_STAT_POINT_THRESHOLDS = (1.0, 2.0, 3.0, 4.0)
TOTAL_PRIMARY_EDGE = 4.0


def _n(s, idx):
    return pd.to_numeric(s, errors="coerce").reindex(idx).to_numpy(float)


def _amer_profit(odds: np.ndarray) -> np.ndarray:
    o = np.asarray(odds, float); p = np.full(len(o), np.nan)
    neg = np.isfinite(o) & (o < 0); pos = np.isfinite(o) & (o > 0)
    p[neg] = 100.0 / (-o[neg]); p[pos] = o[pos] / 100.0
    return p


def _period(g):
    s = pd.to_numeric(g.get("Season"), errors="coerce").to_numpy(float)
    return np.isin(s, np.asarray(DISCOVERY_SEASONS, float)), s == float(CONFIRM_SEASON), s


def _h2h_metrics(g, sign, mask):
    am = pd.to_numeric(g.get("Actual_Margin"), errors="coerce").to_numpy(float)
    fair = pd.to_numeric(g.get("Market_Open_H2H_Fair"), errors="coerce").to_numpy(float)
    aod = pd.to_numeric(g.get("Consensus_Open_Moneyline"), errors="coerce").to_numpy(float)
    bod = pd.to_numeric(g.get("Opp_Consensus_Open_Moneyline"), errors="coerce").to_numpy(float)
    sg = np.asarray(sign, float); mk = np.asarray(mask, bool)
    valid = mk & np.isfinite(am) & ~np.isclose(am, 0.0) & np.isfinite(sg) & ~np.isclose(sg, 0.0) & np.isfinite(fair)
    ix = np.flatnonzero(valid)
    if not len(ix):
        return {"n":0,"hit":np.nan,"roi":np.nan,"market_residual":np.nan,"avg_market_prob":np.nan,"avg_odds":np.nan}
    win = (sg[ix] * am[ix] > 0).astype(float)
    mp = np.where(sg[ix] > 0, fair[ix], 1.0 - fair[ix])
    odds = np.where(sg[ix] > 0, aod[ix], bod[ix])
    prof = _amer_profit(odds)
    ret = np.where(win > 0.5, prof, -1.0)
    roi = float(np.nanmean(ret)) if np.isfinite(ret).any() else np.nan
    return {
        "n":int(len(ix)), "hit":float(win.mean()), "roi":roi,
        "market_residual":float(np.mean(win - mp)), "avg_market_prob":float(np.mean(mp)),
        "avg_odds":float(np.nanmean(odds)) if np.isfinite(odds).any() else np.nan,
    }


def _totals_metrics(g, sign, mask):
    at = pd.to_numeric(g.get("Actual_Total"), errors="coerce").to_numpy(float)
    ot = pd.to_numeric(g.get("Consensus_Open_Total"), errors="coerce").to_numpy(float)
    close_col = next((c for c in ("Consensus_Close_Total_Audit","Closing_Total","Consensus_Close_Total","Current_Total") if c in g.columns), None)
    ct = pd.to_numeric(g.get(close_col), errors="coerce").to_numpy(float) if close_col else np.full(len(g), np.nan)
    sg = np.asarray(sign, float); mk = np.asarray(mask, bool)
    diff = at - ot
    valid = mk & np.isfinite(diff) & ~np.isclose(diff, 0.0) & np.isfinite(sg) & ~np.isclose(sg, 0.0)
    ix = np.flatnonzero(valid)
    if not len(ix):
        return {"n":0,"hit":np.nan,"roi":np.nan,"signed":np.nan,"clv":np.nan}
    win = (sg[ix] * diff[ix] > 0).astype(float)
    ret = np.where(win > 0.5, 100.0/110.0, -1.0)
    clv_arr = sg[ix] * (ct[ix] - ot[ix])
    return {
        "n":int(len(ix)), "hit":float(win.mean()), "roi":float(ret.mean()),
        "signed":float(np.mean(sg[ix] * diff[ix])),
        "clv":float(np.nanmean(clv_arr)) if np.isfinite(clv_arr).any() else np.nan,
    }


def _robustness(g, sign, mask, market):
    _, _, season = _period(g)
    vals = []
    for sy in sorted({int(x) for x in season[np.isfinite(season)]}):
        mm = np.asarray(mask, bool) & (season == float(sy))
        met = _h2h_metrics(g, sign, mm) if market == "h2h" else _totals_metrics(g, sign, mm)
        if met["n"] >= 10:
            vals.append((sy, met))
    if not vals:
        return {"season_metrics":[],"remove_best_hit":np.nan,"min_loso_hit":np.nan,"remove_best_roi":np.nan,"min_loso_roi":np.nan}
    # remove best season by market-appropriate ROI; fallback to hit if ROI missing
    def score(x):
        m=x[1]; return m.get("roi") if np.isfinite(m.get("roi",np.nan)) else m.get("hit",np.nan)
    best = max(vals, key=score)[0]
    rem_mask = np.asarray(mask, bool) & np.isfinite(season) & (season != float(best)) & np.isin(season, np.asarray(DISCOVERY_SEASONS,float))
    rem = _h2h_metrics(g, sign, rem_mask) if market == "h2h" else _totals_metrics(g, sign, rem_mask)
    loso=[]
    disc=np.isin(season,np.asarray(DISCOVERY_SEASONS,float))
    for sy,_ in vals:
        lm=np.asarray(mask,bool)&disc&(season!=float(sy))
        met=_h2h_metrics(g,sign,lm) if market=="h2h" else _totals_metrics(g,sign,lm)
        if met["n"]>=30: loso.append(met)
    return {
        "season_metrics":vals, "remove_best_hit":rem.get("hit",np.nan), "remove_best_roi":rem.get("roi",np.nan),
        "min_loso_hit":min([m.get("hit",np.nan) for m in loso],default=np.nan),
        "min_loso_roi":min([m.get("roi",np.nan) for m in loso],default=np.nan),
    }


def _season_forward_h2h_stat_prob(g, oof_margin, dashboard_module):
    """Convert season-forward fair margins into season-forward outright probabilities.

    Residual distributions are built only from earlier seasons with an already-OOF
    fair margin. This avoids fitting a probability map on the validation season.
    """
    season = pd.to_numeric(g.get("Season"), errors="coerce").to_numpy(float)
    actual = pd.to_numeric(g.get("Actual_Margin"), errors="coerce").to_numpy(float)
    fm = np.asarray(oof_margin, float)
    p = np.full(len(g), np.nan)
    for sy in sorted({int(x) for x in season[np.isfinite(season)]}):
        va = np.isfinite(season) & (season == float(sy)) & np.isfinite(fm)
        tr = np.isfinite(season) & (season < float(sy)) & np.isfinite(fm) & np.isfinite(actual)
        resid = actual[tr] - fm[tr]
        if va.sum() < 20 or len(resid) < 250:
            continue
        p[va] = dashboard_module._ncaaf_stat_empirical_prob_gt(-fm[va], resid)
    return p


def _miner_candidates(g, dashboard_module, registry, market):
    systems = list(((registry.get(market) or {}).get("systems") or [])) if isinstance(registry,dict) else []
    atoms = {a["name"]:np.asarray(a["mask"],bool) for a in dashboard_module._v1355_system_atoms(g)}
    out=[]
    for s in systems:
        cond=list(s.get("conditions") or [])
        if not cond: continue
        mm=np.ones(len(g),bool)
        for c in cond:
            mm &= atoms.get(c,np.zeros(len(g),bool))
        direction=str(s.get("direction") or "PLAY_ON")
        sign=np.where(mm, 1.0 if direction=="PLAY_ON" else -1.0, 0.0)
        # MODEL_STATE conditions are circular with the sibling STAT source; keep
        # them observable but mark them ineligible as an independent mechanism.
        families=list(s.get("families") or [])
        independent=not any(str(f).upper()=="MODEL_STATE" for f in families)
        out.append({"id":str(s.get("system_id")),"kind":"MINER","mask":mm,"sign":sign,"conditions":cond,"families":families,
                    "upstream_state":str(s.get("authority_state")),"independent":independent})
    return out


def _mechanisms(candidate):
    if candidate.get("kind")=="STAT": return ("STATISTICAL_MODEL",)
    mp={
        "MARKET_ROLE":"MARKET_STRUCTURE","MARKET_PRICE":"MARKET_STRUCTURE","TOTAL_REGIME":"MARKET_STRUCTURE",
        "VENUE":"SITUATIONAL","SEASON_TIMING":"SITUATIONAL","SCHEDULE":"SITUATIONAL",
        "PRIOR_RESULT":"TEAM_STATE","ATS_FORM":"TEAM_STATE","SU_FORM":"TEAM_STATE",
        "CONFERENCE":"IDENTITY_CONTEXT","CONFERENCE_PAIR":"IDENTITY_CONTEXT","TEAM_SPECIFIC":"IDENTITY_CONTEXT","COACH_ERA":"IDENTITY_CONTEXT",
        "H2H":"MATCHUP_HISTORY","RIVALRY":"MATCHUP_HISTORY","MODEL_STATE":"STATISTICAL_MODEL",
    }
    return tuple(sorted({mp.get(str(x).upper(), str(x).upper()) for x in candidate.get("families") or []}))


def _candidate_state(market, d, r, c, primary=True):
    if not primary: return "DIAGNOSTIC_ONLY"
    if market=="h2h":
        good_d = d["n"]>=80 and np.isfinite(d["roi"]) and d["roi"]>0 and d["market_residual"]>=0.015
        good_r = (not np.isfinite(r["remove_best_roi"]) or r["remove_best_roi"]>=0) and (not np.isfinite(r["min_loso_roi"]) or r["min_loso_roi"]>=-0.01)
        if not (good_d and good_r): return "NO_REPEATABLE_EDGE"
        if c["n"]<15: return "DISCOVERY_EDGE_AWAIT_CONFIRMATION"
        if c["roi"]>0 and c["market_residual"]>=0: return "REPEATABLE_H2H_EDGE"
        return "DISCOVERY_EDGE_NOT_CONFIRMED"
    good_d = d["n"]>=80 and d["hit"]>BREAK_EVEN_110 and d["roi"]>0 and d["signed"]>0
    good_r = (not np.isfinite(r["remove_best_hit"]) or r["remove_best_hit"]>=0.52) and (not np.isfinite(r["min_loso_hit"]) or r["min_loso_hit"]>=0.515)
    if not (good_d and good_r): return "NO_REPEATABLE_EDGE"
    if c["n"]<15: return "DISCOVERY_EDGE_AWAIT_CONFIRMATION"
    if c["hit"]>BREAK_EVEN_110 and c["signed"]>0: return "REPEATABLE_TOTAL_EDGE"
    return "DISCOVERY_EDGE_NOT_CONFIRMED"


def _evaluate_candidates(g, market, candidates, dashboard_module, log_func):
    disc, conf, _ = _period(g); rows=[]
    for c in candidates:
        d=_h2h_metrics(g,c["sign"],disc & c["mask"]) if market=="h2h" else _totals_metrics(g,c["sign"],disc & c["mask"])
        cf=_h2h_metrics(g,c["sign"],conf & c["mask"]) if market=="h2h" else _totals_metrics(g,c["sign"],conf & c["mask"])
        rb=_robustness(g,c["sign"],disc & c["mask"],market)
        state=_candidate_state(market,d,rb,cf,primary=bool(c.get("primary",True)))
        c={**c,"discovery":d,"confirm":cf,"robustness":rb,"state":state,"mechanisms":_mechanisms(c)}; rows.append(c)
        if market=="h2h":
            log_func(f"[SIBLING-EDGE-V1-RULE] market=H2H kind={c['kind']} id={c['id']} state={state} independent={c.get('independent',True)} "
                     f"n={d['n']} hit={d['hit']:.4f} roi={d['roi']:+.4f} market_residual={d['market_residual']:+.4f} remove_best_roi={rb['remove_best_roi']:+.4f} "
                     f"confirm_n={cf['n']} confirm_hit={cf['hit']:.4f} confirm_roi={cf['roi']:+.4f} confirm_market_residual={cf['market_residual']:+.4f} "
                     f"mechanisms={'+'.join(c['mechanisms']) or 'NONE'} production_authority=0")
        else:
            log_func(f"[SIBLING-EDGE-V1-RULE] market=TOTALS kind={c['kind']} id={c['id']} state={state} independent={c.get('independent',True)} "
                     f"n={d['n']} hit={d['hit']:.4f} roi={d['roi']:+.4f} signed={d['signed']:+.3f} clv={d['clv']:+.3f} remove_best_hit={rb['remove_best_hit']:.4f} min_loso_hit={rb['min_loso_hit']:.4f} "
                     f"confirm_n={cf['n']} confirm_hit={cf['hit']:.4f} confirm_roi={cf['roi']:+.4f} confirm_signed={cf['signed']:+.3f} confirm_clv={cf['clv']:+.3f} "
                     f"mechanisms={'+'.join(c['mechanisms']) or 'NONE'} production_authority=0")
    return rows


def _stack_eval(g, market, rows, log_func):
    repeat_prefix="REPEATABLE_H2H_EDGE" if market=="h2h" else "REPEATABLE_TOTAL_EDGE"
    eligible=[r for r in rows if r["state"]==repeat_prefix and r.get("independent",True)]
    # Collapse nested STAT thresholds into one source and use mechanism identity,
    # not rule count, as confirmation strength.
    disc, conf, _=_period(g)
    for label,pm in (("DISCOVERY",disc),("CONFIRM_2026",conf)):
        signs=np.zeros(len(g),float); indep=np.zeros(len(g),int); conflicts=np.zeros(len(g),bool)
        for i in np.flatnonzero(pm):
            active=[]; s=[]
            for r in eligible:
                if r["mask"][i] and np.isfinite(r["sign"][i]) and not np.isclose(r["sign"][i],0):
                    active.append(r); s.append(float(np.sign(r["sign"][i])))
            if not active: continue
            if len(set(s))>1: conflicts[i]=True; continue
            signs[i]=s[0]
            groups=set()
            stat_seen=False
            for r in active:
                for m in r["mechanisms"]:
                    if m=="STATISTICAL_MODEL": stat_seen=True
                    else: groups.add(m)
            indep[i]=len(groups)+(1 if stat_seen else 0)
        for k in (1,2,3):
            mm=pm & ~conflicts & (indep>=k) & ~np.isclose(signs,0)
            if mm.sum() < (15 if label=="DISCOVERY" else 5): continue
            met=_h2h_metrics(g,signs,mm) if market=="h2h" else _totals_metrics(g,signs,mm)
            if market=="h2h":
                log_func(f"[SIBLING-EDGE-V1-STACK] market=H2H sample={label} min_independent_mechanisms={k} n={met['n']} hit={met['hit']:.4f} roi={met['roi']:+.4f} market_residual={met['market_residual']:+.4f} production_authority=0")
            else:
                log_func(f"[SIBLING-EDGE-V1-STACK] market=TOTALS sample={label} min_independent_mechanisms={k} n={met['n']} hit={met['hit']:.4f} roi={met['roi']:+.4f} signed={met['signed']:+.3f} clv={met['clv']:+.3f} production_authority=0")
    return eligible


def run_sibling_market_edge_research_v1(*, dashboard_module, log_func=print, hard_fail=True):
    try:
        cache=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{})
        g=cache.get("miner_games") if isinstance(cache,dict) else None
        if not isinstance(g,pd.DataFrame) or g.empty: raise RuntimeError("miner games unavailable")
        oof_margin=np.asarray(cache.get("oof_margin"),float); oof_total=np.asarray(cache.get("oof_total"),float)
        registry=cache.get("system_miner_v2") or {}; latest=int(cache.get("latest") or CONFIRM_SEASON)
        if len(oof_margin)!=len(g) or len(oof_total)!=len(g): raise RuntimeError("OOF cache length mismatch")
        log_func(f"[SIBLING-EDGE-V1-PREFLIGHT] status=PASS source_tag={SIBLING_MARKET_EDGE_RESEARCH_V1_SOURCE_TAG} games={len(g)} latest={latest} "
                 f"markets=H2H,TOTALS spread_policy_unchanged=TRUE h2h_market_relative=TRUE totals_target_separate=TRUE production_authority=0")

        # H2H STAT probability: season-forward fair-margin residual mapping.
        hp=_season_forward_h2h_stat_prob(g,oof_margin,dashboard_module)
        hm=pd.to_numeric(g.get("Market_Open_H2H_Fair"),errors="coerce").to_numpy(float); hgap=hp-hm
        h2h_candidates=[]
        for th in H2H_STAT_GAP_THRESHOLDS:
            mm=np.isfinite(hgap)&(np.abs(hgap)>=th); sg=np.where(mm,np.sign(hgap),0.0)
            h2h_candidates.append({"id":f"H2H_STAT_GAP_{int(th*1000):03d}BP","kind":"STAT","mask":mm,"sign":sg,"families":["MODEL_STATE"],"independent":True,"primary":math.isclose(th,H2H_PRIMARY_GAP)})
        h2h_candidates += _miner_candidates(g,dashboard_module,registry,"h2h")
        hrows=_evaluate_candidates(g,"h2h",h2h_candidates,dashboard_module,log_func)
        helig=_stack_eval(g,"h2h",hrows,log_func)

        # Existing H2H model siblings are reported as model-level challengers. They
        # remain separate from rule authority until row-level prospective ledgers exist.
        season=np.asarray(cache.get("season_arr"),float)
        hmatch=dashboard_module._v1354_h2h_matchup_v2(g,season,oof_margin,latest)
        hresid=dashboard_module._v13542_h2h_market_residual_v1(g,season,oof_margin,latest)
        log_func(f"[SIBLING-EDGE-V1-H2H-MODEL] id=H2H_MATCHUP_V2 status={hmatch.get('status')} n={int(hmatch.get('n',0) or 0)} auc={float(hmatch.get('auc',np.nan)):.4f} ll={float(hmatch.get('logloss',np.nan)):.6f} brier={float(hmatch.get('brier',np.nan)):.6f} prospective_n={int(hmatch.get('prospective_n',0) or 0)} production_authority=0")
        log_func(f"[SIBLING-EDGE-V1-H2H-MODEL] id=H2H_MARKET_RESIDUAL_V1 status={hresid.get('status')} n={int(hresid.get('n',0) or 0)} auc={float(hresid.get('auc',np.nan)):.4f} market_auc={float(hresid.get('market_auc',np.nan)):.4f} ll={float(hresid.get('logloss',np.nan)):.6f} market_ll={float(hresid.get('market_logloss',np.nan)):.6f} production_authority=0")

        # TOTAL STAT fair-total vs opening-total disagreement.
        ot=pd.to_numeric(g.get("Consensus_Open_Total"),errors="coerce").to_numpy(float); tedge=oof_total-ot
        total_candidates=[]
        for th in TOTAL_STAT_POINT_THRESHOLDS:
            mm=np.isfinite(tedge)&(np.abs(tedge)>=th); sg=np.where(mm,np.sign(tedge),0.0)
            total_candidates.append({"id":f"TOTAL_STAT_EDGE_{int(th)}P","kind":"STAT","mask":mm,"sign":sg,"families":["MODEL_STATE"],"independent":True,"primary":math.isclose(th,TOTAL_PRIMARY_EDGE)})
        total_candidates += _miner_candidates(g,dashboard_module,registry,"totals")
        trows=_evaluate_candidates(g,"totals",total_candidates,dashboard_module,log_func)
        telig=_stack_eval(g,"totals",trows,log_func)

        hrep=sum(r["state"]=="REPEATABLE_H2H_EDGE" for r in hrows); trep=sum(r["state"]=="REPEATABLE_TOTAL_EDGE" for r in trows)
        log_func(f"[SIBLING-EDGE-V1-CONTRACT] status=PASS h2h_candidates={len(hrows)} h2h_repeatable={hrep} h2h_stack_eligible={len(helig)} "
                 f"totals_candidates={len(trows)} totals_repeatable={trep} totals_stack_eligible={len(telig)} "
                 f"h2h_hit_rate_not_edge_without_price=TRUE totals_old_TOTAL_SCORE_V2_remains_retired=TRUE markets_independent=TRUE spread_unchanged=TRUE production_authority=0")
        return {"status":"PASS","source_tag":SIBLING_MARKET_EDGE_RESEARCH_V1_SOURCE_TAG,"h2h_rules":hrows,"totals_rules":trows,"h2h_repeatable":hrep,"totals_repeatable":trep,"production_authority":0}
    except Exception as e:
        log_func(f"[SIBLING-EDGE-V1-CONTRACT] status=FAILED error={type(e).__name__}:{e} production_authority=0")
        if hard_fail: raise
        return {"status":"FAILED","error":f"{type(e).__name__}:{e}","production_authority":0}
