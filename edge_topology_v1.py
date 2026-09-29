"""EDGE_TOPOLOGY_V1 — map how independent NCAAF spread edges coexist.

Research-only.  This module does not blend probabilities and has zero production
bet authority.  It answers a narrower question: which independent edge sources
are active on the same game, do they agree on direction, and what happens to
ATS, realized market error, and (when available) closing-line value?

Discovery topologies are descriptive/hypothesis-generating because the currently
published Miner inventory and evidence grades were selected using that history.
2026 is confirmation only, not independent post-freeze prospective evidence.
"""
from __future__ import annotations

from typing import Dict, List, Any, Tuple
import hashlib
import math
import os
import numpy as np
import pandas as pd

EDGE_TOPOLOGY_V1_SOURCE_TAG = "edge-topology-v1-peer-source-map-clv-ledger"
BREAK_EVEN = 110.0 / 210.0
DISCOVERY_SEASONS = (2023, 2024, 2025)
CONFIRM_SEASON = 2026
SOURCE_ORDER = ("STAT", "BIGAL", "PATHI", "MINER")
STRENGTH_RANK = {"STRONG_SHADOW": 3, "MODERATE_SHADOW": 2, "WEAK_SHADOW": 1, "RESEARCH_ONLY": 0}

# Separate from the legacy V13 forward ledger.  These tables are intentionally
# not written by the training job.  They are the destination contract for a
# subsequent live edge scorer so pregame states can be frozen before kickoff.
EDGE_LEDGER_PREDICTIONS_TABLE = "sharplogger.sharp_data.ncaaf_edge_forward_shadow_predictions"
EDGE_LEDGER_RESULTS_TABLE = "sharplogger.sharp_data.ncaaf_edge_forward_shadow_results"
EDGE_LEDGER_VERSION = "2026-09-29-edge-topology-v1"


def _num(x, index=None):
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce")
    return pd.Series(pd.to_numeric(x, errors="coerce"), index=index)


def _roi(hit: float) -> float:
    if not np.isfinite(hit):
        return np.nan
    return float(hit * (100.0 / 110.0) - (1.0 - hit))


def _norm_text(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip().lower()


def _physical_key(g: pd.DataFrame) -> pd.Series:
    if "Source_Game_ID" in g.columns:
        k = g["Source_Game_ID"].astype(str).str.strip().str.lower()
        if k.ne("").sum() >= int(0.9 * len(g)):
            return k
    season = pd.to_numeric(g.get("Season"), errors="coerce").astype("Int64").astype(str)
    date = pd.to_datetime(g.get("Game_Date"), errors="coerce", utc=True).dt.strftime("%Y-%m-%d").fillna("")
    a = g.get("Team_Norm", pd.Series("", index=g.index)).astype(str).str.lower().str.strip()
    b = g.get("Opponent_Norm", pd.Series("", index=g.index)).astype(str).str.lower().str.strip()
    return season + "|" + date + "|" + a + "|" + b


def _side_key(g: pd.DataFrame) -> pd.Series:
    season = pd.to_numeric(g.get("Season"), errors="coerce").astype("Int64").astype(str)
    date = pd.to_datetime(g.get("Game_Date"), errors="coerce", utc=True).dt.strftime("%Y-%m-%d").fillna("")
    a = g.get("Team_Norm", pd.Series("", index=g.index)).astype(str).str.lower().str.strip()
    b = g.get("Opponent_Norm", pd.Series("", index=g.index)).astype(str).str.lower().str.strip()
    return season + "|" + date + "|" + a + "|" + b


def _registry_strength(registry_out: dict) -> Dict[Tuple[str, str], str]:
    out = {}
    for e in (registry_out or {}).get("entries") or []:
        if not isinstance(e, dict):
            continue
        out[(str(e.get("source", "")).upper(), str(e.get("edge_id", "")))] = str(e.get("strength", "RESEARCH_ONLY"))
    return out


def _source_vote_from_occurrences(g: pd.DataFrame, history: dict, family: str,
                                  strength_lookup: dict, min_strength: int = 1):
    """Return a {-1,0,+1} family vote plus an internal-mixed flag.

    Occurrence records are side-oriented. +1 means the rule selected the anchor
    Team_Norm side in g, -1 means it selected the anchor opponent.
    """
    n = len(g)
    pos = np.zeros(n, dtype=bool); neg = np.zeros(n, dtype=bool)
    key_to_idx = {str(k): i for i, k in enumerate(_side_key(g).tolist())}
    used = []
    for name, rec in (history or {}).items():
        if not isinstance(rec, dict) or str(rec.get("role", "directional")).lower() != "directional":
            continue
        if str(rec.get("family", "")).upper() != family.upper():
            continue
        st = strength_lookup.get((family.upper(), str(name)), "RESEARCH_ONLY")
        if STRENGTH_RANK.get(st, 0) < min_strength:
            continue
        used.append(str(name))
        for o in rec.get("occurrences") or []:
            if not isinstance(o, dict):
                continue
            try:
                sy = int(o.get("season"))
            except Exception:
                continue
            ds = _norm_text(o.get("date"))
            tm = _norm_text(o.get("team")); op = _norm_text(o.get("opponent"))
            k = f"{sy}|{ds}|{tm}|{op}"
            i = key_to_idx.get(k)
            if i is not None:
                pos[i] = True
                continue
            rk = f"{sy}|{ds}|{op}|{tm}"
            i = key_to_idx.get(rk)
            if i is not None:
                neg[i] = True
    mixed = pos & neg
    vote = np.zeros(n, dtype=float)
    vote[pos & ~mixed] = 1.0
    vote[neg & ~mixed] = -1.0
    vote[mixed] = np.nan
    return vote, mixed, used


def _miner_vote(g: pd.DataFrame, dashboard_module, strength_lookup: dict, *, min_strength: int = 1):
    n = len(g); pos = np.zeros(n, bool); neg = np.zeros(n, bool); used=[]; excluded=[]
    cache = getattr(dashboard_module, "_V1357_SPREAD_RESEARCH_CACHE", {})
    reg = (cache.get("system_miner_v2") or {}).get("spreads") or {}
    mg = cache.get("miner_games")
    if not isinstance(mg, pd.DataFrame) or len(mg) != n:
        return np.zeros(n, float), np.zeros(n, bool), used, ["MINER_FRAME_UNAVAILABLE"]
    atoms = {a["name"]: np.asarray(a["mask"], bool) for a in dashboard_module._v1355_system_atoms(mg)}
    for sys in reg.get("systems") or []:
        sid = str(sys.get("system_id", ""))
        cond = list(sys.get("conditions") or [])
        # Independent topology support cannot use model-state clauses; that would
        # count STAT as confirming itself.
        if any(str(c).startswith(("STAT_EDGE_", "H2H_STAT_", "TOTAL_MODEL_")) for c in cond):
            excluded.append(sid); continue
        st = strength_lookup.get(("MINER", sid), "RESEARCH_ONLY")
        if STRENGTH_RANK.get(st, 0) < min_strength:
            continue
        mm = np.ones(n, bool)
        for c in cond:
            mm &= atoms.get(c, np.zeros(n, bool))
        if not mm.any():
            continue
        used.append(sid)
        direction = str(sys.get("direction", "PLAY_ON")).upper()
        if direction == "FADE": neg |= mm
        else: pos |= mm
    mixed = pos & neg
    vote = np.zeros(n, dtype=float); vote[pos & ~mixed] = 1.0; vote[neg & ~mixed] = -1.0; vote[mixed] = np.nan
    return vote, mixed, used, excluded


def _stat_vote(g: pd.DataFrame, stat_out: dict, sample: str):
    """Use the frozen 2026 primary tail for descriptive/confirmation topology.

    DISCOVERY is hypothesis-generating only. CONFIRM_2026 is the untouched target
    season for that 2026 selection. The separate walk-forward summary remains the
    authority for chronological 2025+2026 policy performance.
    """
    p = ((stat_out or {}).get("primary") or {})
    leader = p.get("leader") or {}
    pred = leader.get("pred")
    if pred is None:
        return np.zeros(len(g), float), np.zeros(len(g), bool)
    pred = np.asarray(pred, float)
    if sample == "DISCOVERY":
        mask = np.asarray(((p.get("prior") or {}).get("mask")), bool)
    else:
        mask = np.asarray(p.get("test_mask"), bool)
    vote = np.zeros(len(g), float)
    vote[mask & np.isfinite(pred) & (pred > 0)] = 1.0
    vote[mask & np.isfinite(pred) & (pred < 0)] = -1.0
    return vote, mask


def _clv_vector(g: pd.DataFrame, dashboard_module, selected_sign: np.ndarray):
    """Return selected-side line CLV, positive when the locked open was better than close.

    For the anchor side: CLV = open_spread - close_spread.  For the opponent the
    sign is reversed.  Closing values are evaluation-only and never enter source
    activation or direction.
    """
    n = len(g); out = np.full(n, np.nan, float)
    cc = getattr(dashboard_module, "_EDGE_RESEARCH_CLV_CACHE", None)
    if not isinstance(cc, pd.DataFrame) or cc.empty:
        return out, {"status":"UNAVAILABLE","rows":0,"coverage":0.0}
    c = cc.copy()
    c["__key"] = c.get("side_key", pd.Series("", index=c.index)).astype(str)
    mp = c.drop_duplicates("__key", keep="last").set_index("__key")
    k = _side_key(g)
    close = pd.to_numeric(k.map(mp.get("close_spread", pd.Series(dtype=float))), errors="coerce").to_numpy(float)
    op = pd.to_numeric(g.get("Consensus_Open_Spread"), errors="coerce").to_numpy(float)
    s = np.asarray(selected_sign, float)
    ok = np.isfinite(op) & np.isfinite(close) & np.isfinite(s) & ~np.isclose(s, 0.0)
    out[ok] = (op[ok] - close[ok]) * s[ok]
    return out, {"status":"READY" if ok.any() else "NO_MATCHES", "rows":int(ok.sum()), "coverage":float(ok.mean()) if n else 0.0}


def _metrics(target: np.ndarray, sign: np.ndarray, mask: np.ndarray, clv: np.ndarray | None = None) -> dict:
    t=np.asarray(target,float); s=np.asarray(sign,float); m=np.asarray(mask,bool)&np.isfinite(t)&np.isfinite(s)&~np.isclose(s,0.0)&~np.isclose(t,0.0,atol=1e-9)
    if not m.any():
        return {"n":0,"hit":np.nan,"roi":np.nan,"signed":np.nan,"clv_n":0,"clv":np.nan,"clv_pos":np.nan}
    realized=t[m]*s[m]; hit=float(np.mean(realized>0))
    out={"n":int(m.sum()),"hit":hit,"roi":_roi(hit),"signed":float(np.mean(realized)),"clv_n":0,"clv":np.nan,"clv_pos":np.nan}
    if clv is not None:
        c=np.asarray(clv,float)[m]; c=c[np.isfinite(c)]
        if len(c): out.update({"clv_n":int(len(c)),"clv":float(np.mean(c)),"clv_pos":float(np.mean(c>0))})
    return out


def _aggregate_sources(votes: Dict[str, np.ndarray], mixed: Dict[str, np.ndarray]):
    n=len(next(iter(votes.values()))) if votes else 0
    labels=np.empty(n,dtype=object); sign=np.zeros(n,float); counts=np.zeros(n,int); conflicts=np.zeros(n,bool)
    for i in range(n):
        active=[]; vals=[]; internal=False
        for src in SOURCE_ORDER:
            v=float(votes[src][i]) if np.isfinite(votes[src][i]) else np.nan
            if bool(mixed[src][i]):
                active.append(src); internal=True; continue
            if np.isfinite(v) and not np.isclose(v,0.0):
                active.append(src); vals.append(v)
        counts[i]=len(active)
        if not active:
            labels[i]="NONE"; continue
        if internal or (vals and len(set(np.sign(vals)))>1):
            labels[i]="+".join(active)+"_CONFLICT"; conflicts[i]=True; sign[i]=0.0
        elif vals:
            labels[i]="+".join(active)+"_AGREE"; sign[i]=float(np.sign(vals[0]))
        else:
            labels[i]="+".join(active)+"_MIXED"; conflicts[i]=True
    return labels,sign,counts,conflicts


def _topology_table(g: pd.DataFrame, target: np.ndarray, votes: dict, mixed: dict, sample_mask: np.ndarray,
                    dashboard_module, sample_label: str, log_func=print):
    labels,sign,count,conflict=_aggregate_sources(votes,mixed)
    clv,clv_info=_clv_vector(g,dashboard_module,sign)
    rows=[]
    for lab in sorted(set(labels[np.asarray(sample_mask,bool)])):
        if lab=="NONE": continue
        m=np.asarray(sample_mask,bool)&(labels==lab)
        # Conflicts have no consensus side; report frequency but do not manufacture a bet.
        if lab.endswith("_CONFLICT") or lab.endswith("_MIXED"):
            met={"n":int(m.sum()),"hit":np.nan,"roi":np.nan,"signed":np.nan,"clv_n":0,"clv":np.nan,"clv_pos":np.nan}
        else:
            met=_metrics(target,sign,m,clv)
        rows.append({"topology":lab,"source_count":int(np.nanmax(count[m])) if m.any() else 0,**met})
    rows.sort(key=lambda z:(-int(z.get("n",0)),str(z.get("topology"))))
    for r in rows:
        log_func(
            f"[EDGE-TOPOLOGY-V1] sample={sample_label} topology={r['topology']} sources={r['source_count']} n={r['n']} "
            f"hit={float(r['hit']):.4f} roi={float(r['roi']):+.4f} signed_market_error={float(r['signed']):+.3f} "
            f"clv_n={r['clv_n']} avg_clv={float(r['clv']):+.3f} clv_positive={float(r['clv_pos']):.4f} production_authority=0"
        )
    return rows, {"labels":labels,"sign":sign,"source_count":count,"conflict":conflict,"clv":clv,"clv_info":clv_info}


def _source_count_ladder(target, agg, sample_mask, sample_label, log_func=print):
    out=[]
    for k in (1,2,3,4):
        m=np.asarray(sample_mask,bool)&~agg["conflict"]&(agg["source_count"]>=k)&np.isfinite(agg["sign"])&~np.isclose(agg["sign"],0.0)
        met=_metrics(target,agg["sign"],m,agg.get("clv"))
        if met["n"]:
            out.append({"min_sources":k,**met})
            log_func(f"[EDGE-TOPOLOGY-V1-STACK] sample={sample_label} min_agreeing_sources={k} n={met['n']} hit={met['hit']:.4f} roi={met['roi']:+.4f} signed_market_error={met['signed']:+.3f} clv_n={met['clv_n']} avg_clv={met['clv']:+.3f} production_authority=0")
    return out


def edge_prospective_ledger_contract() -> dict:
    """Schema contract only. Training never writes prospective rows.

    A later live scorer must call the writer BEFORE kickoff.  STAT Combo fields are
    nullable until the frozen live family scorer is wired; this prevents fake backfill.
    """
    return {
        "ledger_version":EDGE_LEDGER_VERSION,
        "predictions_table":EDGE_LEDGER_PREDICTIONS_TABLE,
        "results_table":EDGE_LEDGER_RESULTS_TABLE,
        "immutable_prediction_fields":[
            "prediction_event_id","recorded_at","physical_game_id","game_start","side","book","spread","odds",
            "stat_combo_active","stat_combo_points","stat_combo_family","stat_combo_threshold","v13_direction",
            "bigal_active","pathi_active","miner_active_ids","edge_topology","source_count","consensus_state","decision_status"
        ],
        "settlement_fields":["closing_line","closing_odds","clv_points","ats_margin","ats_result","profit_per_unit"],
        "write_policy":"PREKICKOFF_APPEND_MERGE_BY_EVENT_ID__NO_RESULT_BACKFILL_IN_PREDICTION_ROW",
        "training_job_writes":False,
        "live_stat_combo_required_before_full_topology_write":True,
    }


def run_edge_topology_v1(*, dashboard_module, stat_out: dict, registry_out: dict, log_func=print, hard_fail=True):
    try:
        cache=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{})
        g=cache.get("games") if isinstance(cache,dict) else None
        if g is None or not isinstance(g,pd.DataFrame) or g.empty:
            raise RuntimeError("EDGE_TOPOLOGY_V1 requires spread research games")
        g=g.copy(); season=pd.to_numeric(g.get("Season"),errors="coerce").to_numpy(float)
        target=pd.to_numeric(g.get("Market_Error_Margin"),errors="coerce").to_numpy(float)
        dup=int(_physical_key(g).duplicated().sum())
        if dup: raise RuntimeError(f"physical game duplication rows={dup}")
        strength=_registry_strength(registry_out)
        hist=getattr(dashboard_module,"_V143_SYSTEM_HISTORY_CACHE",{})
        bvote,bmix,bused=_source_vote_from_occurrences(g,hist,"BIGAL",strength,min_strength=1)
        pvote,pmix,pused=_source_vote_from_occurrences(g,hist,"PATHI",strength,min_strength=1)
        mvote,mmix,mused,mexcluded=_miner_vote(g,dashboard_module,strength,min_strength=1)
        stat_disc,stat_disc_mask=_stat_vote(g,stat_out,"DISCOVERY")
        stat_conf,stat_conf_mask=_stat_vote(g,stat_out,"CONFIRM_2026")
        disc=np.isin(season,np.asarray(DISCOVERY_SEASONS,float))
        conf=season==float(CONFIRM_SEASON)
        log_func(
            f"[EDGE-TOPOLOGY-V1-PREFLIGHT] status=PASS source_tag={EDGE_TOPOLOGY_V1_SOURCE_TAG} games={len(g)} physical_game_duplicates=0 "
            f"bigal_edges={len(bused)} pathi_edges={len(pused)} miner_independent_edges={len(mused)} miner_model_state_excluded={len(mexcluded)} "
            f"discovery_role=DESCRIPTIVE_HYPOTHESIS_GENERATION confirm_2026_role=OUT_OF_DISCOVERY_CONFIRMATION_NOT_POSTFREEZE_PROSPECTIVE production_authority=0"
        )
        base_mixed={"BIGAL":bmix,"PATHI":pmix,"MINER":mmix}
        # Discovery uses the frozen 2026-primary STAT tail only descriptively.
        votes_d={"STAT":stat_disc,"BIGAL":bvote,"PATHI":pvote,"MINER":mvote}
        mixed_d={"STAT":np.zeros(len(g),bool),**base_mixed}
        drows,dagg=_topology_table(g,target,votes_d,mixed_d,disc,dashboard_module,"DISCOVERY_2023_2025_DESCRIPTIVE",log_func)
        dstacks=_source_count_ladder(target,dagg,disc,"DISCOVERY_2023_2025_DESCRIPTIVE",log_func)
        votes_c={"STAT":stat_conf,"BIGAL":bvote,"PATHI":pvote,"MINER":mvote}
        mixed_c={"STAT":np.zeros(len(g),bool),**base_mixed}
        crows,cagg=_topology_table(g,target,votes_c,mixed_c,conf,dashboard_module,"CONFIRM_2026",log_func)
        cstacks=_source_count_ladder(target,cagg,conf,"CONFIRM_2026",log_func)
        # Explicit conflict audit: disagreement should not be silently turned into a side.
        for label,mask,agg in (("DISCOVERY_2023_2025_DESCRIPTIVE",disc,dagg),("CONFIRM_2026",conf,cagg)):
            any_edge=np.asarray(mask,bool)&(agg["source_count"]>0)
            conflicts=any_edge&agg["conflict"]
            log_func(f"[EDGE-TOPOLOGY-V1-CONFLICT] sample={label} edge_games={int(any_edge.sum())} conflict_games={int(conflicts.sum())} conflict_rate={(float(conflicts.sum()/max(1,int(any_edge.sum())))):.4f} policy=PASS_NO_FORCED_SIDE production_authority=0")
        # Chronological STAT policy remains the clean reference rather than the descriptive topology table.
        wf=(stat_out or {}).get("walkforward") or {}; wfm=wf.get("metrics") or {}; wfd=wf.get("diag") or {}
        log_func(f"[EDGE-TOPOLOGY-V1-WALKFORWARD-REFERENCE] stat_seasons={wf.get('seasons')} n={int(wfm.get('n',0) or 0)} hit={float(wfm.get('hit',np.nan)):.4f} roi={float(wfm.get('roi',np.nan)):+.4f} signed_market_error={float(wfm.get('signed',np.nan)):+.3f} role={wfd.get('role','UNPROVEN')} topology_discovery_not_substitute_for_walkforward=TRUE")
        clv_ready=max(int(dagg["clv_info"].get("rows",0)),int(cagg["clv_info"].get("rows",0)))
        ledger=edge_prospective_ledger_contract()
        log_func(f"[EDGE-TOPOLOGY-V1-CLV] status={'READY' if clv_ready else 'UNAVAILABLE'} matched_rows={clv_ready} source=HISTORICAL_CORE_FINAL_PREGAME_CLOSE evaluation_only=TRUE selection_input=FALSE")
        log_func(f"[EDGE-PROSPECTIVE-LEDGER] status=SCHEMA_READY_NOT_WRITING predictions_table={ledger['predictions_table']} results_table={ledger['results_table']} immutable_pregame=TRUE training_job_writes=FALSE live_stat_combo_scorer=REQUIRED_BEFORE_FULL_TOPOLOGY_WRITE no_retroactive_backfill=TRUE")
        log_func(f"[EDGE-TOPOLOGY-V1-CONTRACT] status=PASS source_tag={EDGE_TOPOLOGY_V1_SOURCE_TAG} discovery_topologies={len(drows)} confirmation_topologies={len(crows)} peer_sources=STAT,BIGAL,PATHI,MINER miner_model_state_circularity_blocked=TRUE conflicts_passed=TRUE clv_evaluation_only=TRUE prospective_ledger_schema_ready=TRUE zero_production_authority=TRUE")
        return {"status":"PASS","source_tag":EDGE_TOPOLOGY_V1_SOURCE_TAG,"discovery":drows,"confirmation":crows,"discovery_stacks":dstacks,"confirmation_stacks":cstacks,"ledger_contract":ledger,"production_authority":0}
    except Exception as e:
        log_func(f"[EDGE-TOPOLOGY-V1-CONTRACT] status=FAILED error={type(e).__name__}:{e} production_authority=0")
        if hard_fail: raise
        return {"status":"FAILED","error":f"{type(e).__name__}:{e}","production_authority":0}
