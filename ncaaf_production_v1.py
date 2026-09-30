"""NCAAF Production V1 — promoted edge-authority layer.

This module deliberately separates two concepts:

1. Probability models stay frozen and target-specific (Spread / H2H / Totals).
2. The validated edge research is promoted as a *decision selector* on top of
   those probabilities.  It does not blend or rewrite the model probability.

The contract promoted here is frozen to the evidence reviewed on 2026-09-29:
- SPREAD: Frozen Spread Edge Policy V1 authority families.
- TOTALS: three family-collapsed repeatable evidence families; no extra authority
  is granted for a two-mechanism stack because that stack did not repeat in 2026.
- H2H: model display only; no edge authority was repeatable enough to promote.

The spread STAT selector is reconstructed exactly from the 2026 confirmation
fold: rushing family, Ridge(alpha=24), target-free feature pruning, trained only
on seasons < 2026, reliability beta frozen from prior-only OOF evidence, and
absolute scaled market-error threshold 0.75.
"""
from __future__ import annotations

import gzip
import json
import pickle
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from google.cloud import storage

NCAAF_PRODUCTION_V1_SOURCE_TAG = "ncaaf-production-v1-fixed-backbones-edge-authority-20260929"
NCAAF_PRODUCTION_V1_ARTIFACT = "production/ncaaf/ncaaf_production_v1.pkl.gz"
NCAAF_PRODUCTION_V1_META = "production/ncaaf/ncaaf_production_v1.json"
NCAAF_PRODUCTION_V1_FREEZE_UTC = "2026-09-29T22:47:00+00:00"

EXPECTED_STAT_COMBO = ("rushing",)
EXPECTED_STAT_THRESHOLD = 0.75
EXPECTED_TOTAL_REPRESENTATIVES = {
    "SYS-TOTALS-CEC45A15",  # FAV_14_99 & TOTAL_52_60
    "SYS-TOTALS-144776B2",  # CONF_MAC
    "SYS-TOTALS-86C3A59D",  # CONF_ACC & FAV_14_99
}

# Exact fixed backbones validated by REFIT_CADENCE_TEST_V1.  These are the
# production probability contracts; no AutoFS or multi-head feature search runs
# in the dashboard.
PROD_SPREAD_FEATURES = (
    "Context_Intercept",
    "Diff_RawRecent3_Off_YPP",
    "B_RawSeason_GameAdj_Def_Rush_YPA",
)
PROD_TOTAL_FEATURES = (
    "Context_Intercept",
    "A_RawRecent3_Def_Rush_YPA_Allowed",
)
PROD_H2H_FEATURES = (
    "H2H2_Prior_Meetings",
    "H2H2_Prior_Margin_Current_Orientation",
    "H2H2_Prior_Total",
    "H2H2_Days_Since",
    "H2H2_Same_Home_Role",
    "H2H2_Recency_Margin",
    "H2H2_Recency_Total",
)
PROD_LINEAR_WEIGHT = 0.75


def _f(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _clean_metrics(m: dict | None) -> dict:
    m = m or {}
    keys = ("n", "hit", "roi", "signed", "clv", "rmse", "mae", "auc", "logloss", "brier")
    out = {}
    for k in keys:
        if k not in m:
            continue
        if k == "n":
            try: out[k] = int(m[k])
            except Exception: out[k] = 0
        else:
            v = _f(m[k])
            out[k] = v if np.isfinite(v) else None
    return out


def _norm_team(x: Any) -> str:
    s = str(x or "").strip().lower()
    return " ".join(s.split())


def _norm_conf(x: Any) -> str:
    s = str(x or "").upper().strip()
    s = s.replace("CONFERENCE", "").replace("-", " ").replace("_", " ")
    return " ".join(s.split())


def _first_num(row: pd.Series, names: Iterable[str]) -> float:
    for c in names:
        if c in row.index:
            v = pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0]
            if pd.notna(v):
                return float(v)
    return np.nan


def _first_text(row: pd.Series, names: Iterable[str]) -> str:
    for c in names:
        if c in row.index:
            s = str(row.get(c) or "").strip()
            if s and s.lower() not in {"nan", "none"}:
                return s
    return ""


def _outcome_norm(row: pd.Series) -> str:
    if "Outcome_Norm" in row.index:
        s = _norm_team(row.get("Outcome_Norm"))
        if s: return s
    return _norm_team(row.get("Outcome"))


def _home_away(group: pd.DataFrame) -> Tuple[str, str]:
    if group.empty:
        return "", ""
    r = group.iloc[0]
    home = _norm_team(r.get("Home_Team_Norm", r.get("Home_Team", "")))
    away = _norm_team(r.get("Away_Team_Norm", r.get("Away_Team", "")))
    return home, away


def _opposite_team(target: str, home: str, away: str) -> str:
    t = _norm_team(target)
    if t == home: return away
    if t == away: return home
    return ""


def _home_context_row(group: pd.DataFrame) -> pd.Series:
    """Use the home-side spread row as the canonical historical orientation."""
    if group.empty:
        return pd.Series(dtype=object)
    home, _ = _home_away(group)
    outs = group.apply(_outcome_norm, axis=1)
    hit = group.loc[outs.eq(home)] if home else pd.DataFrame()
    return (hit.iloc[0] if not hit.empty else group.iloc[0]).copy()


def _condition_true(row: pd.Series, condition: str) -> bool:
    """Evaluate only the frozen production atoms; unknown atoms fail closed."""
    c = str(condition or "").upper().strip()
    sp = _first_num(row, ("Consensus_Open_Spread", "Opening_Spread", "First_Line_Value", "Open_Value", "Opening_Line"))
    tt = _first_num(row, ("Consensus_Open_Total", "Opening_Total", "TOT_Open", "Open_Total"))
    if c == "TOTAL_60_99": return bool(np.isfinite(tt) and tt >= 60.0 and tt < 99.0)
    if c == "TOTAL_52_60": return bool(np.isfinite(tt) and tt >= 52.0 and tt < 60.0)
    if c == "DOG_3_7": return bool(np.isfinite(sp) and sp > 3.0 and sp <= 7.0)
    if c == "FAV_14_99": return bool(np.isfinite(sp) and (-sp) > 14.0 and (-sp) <= 99.0)
    if c == "CURRENT_FAVORITE": return bool(np.isfinite(sp) and sp < 0.0)
    if c == "HOME": return True  # canonical row is explicitly home-oriented
    conf = _norm_conf(_first_text(row, ("Conference", "Team_Conference", "Conference_Norm", "Context_Conference")))
    if c == "CONF_BIG_12": return conf in {"BIG 12", "BIG12"}
    if c == "CONF_MAC": return conf in {"MAC", "MID AMERICAN", "MIDAMERICAN"}
    if c == "CONF_ACC": return conf == "ACC"
    return False


def _rule_true(row: pd.Series, conditions: Iterable[str]) -> bool:
    cond = list(conditions or [])
    return bool(cond) and all(_condition_true(row, x) for x in cond)


def _maximum_matching(active_families: List[dict]) -> int:
    match: Dict[str, int] = {}
    def dfs(i: int, seen: set[str]) -> bool:
        for mech in sorted(active_families[i].get("mechanisms") or ()):  # one family -> one independent mechanism
            if mech in seen: continue
            seen.add(mech)
            if mech not in match or dfs(match[mech], seen):
                match[mech] = i
                return True
        return False
    score = 0
    for i in range(len(active_families)):
        if dfs(i, set()): score += 1
    return score


def _find_system(spread_systems: list, system_id: str) -> dict:
    for s in spread_systems or []:
        if str(s.get("system_id")) == str(system_id):
            return s
    return {}


def _new_h2h_pipe():
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=0.20, max_iter=1500, class_weight=None)),
    ])


def _blend_predict(models, X: pd.DataFrame, weight: float = PROD_LINEAR_WEIGHT) -> np.ndarray:
    if not isinstance(models, (tuple, list)) or len(models) != 2:
        return np.full(len(X), np.nan, dtype=float)
    try:
        p1=np.asarray(models[0].predict(X), dtype=float)
        p2=np.asarray(models[1].predict(X), dtype=float)
        w=float(np.clip(weight,0.0,1.0))
        return w*p1+(1.0-w)*p2
    except Exception:
        return np.full(len(X), np.nan, dtype=float)


def _empirical_prob_gt(threshold, residuals) -> np.ndarray:
    thr=np.asarray(threshold,dtype=float)
    rr=np.asarray(residuals if residuals is not None else [],dtype=float)
    rr=np.sort(rr[np.isfinite(rr)])
    out=np.full(thr.shape,np.nan,dtype=float)
    if rr.size < 30:
        return out
    ok=np.isfinite(thr)
    pos=np.searchsorted(rr,thr[ok],side="right")
    out[ok]=1.0-(pos.astype(float)/float(rr.size))
    eps=0.5/float(rr.size+1)
    return np.clip(out,eps,1.0-eps)


def _direct_feature_profile_col(feature: str) -> str | None:
    c=str(feature or "")
    if c == "Context_Intercept":
        return None
    prefixes=(
        ("A_RawSeason_","RawSeason"),("B_RawSeason_","RawSeason"),("Diff_RawSeason_","RawSeason"),
        ("A_RawRecent3_","RawRecent3"),("B_RawRecent3_","RawRecent3"),("Diff_RawRecent3_","RawRecent3"),
        ("A_State_","State"),("B_State_","State"),("Diff_State_","State"),
        ("A_Recent3_","Recent3"),("B_Recent3_","Recent3"),("Diff_Recent3_","Recent3"),
    )
    for prefix,kind in prefixes:
        if c.startswith(prefix):
            metric=c[len(prefix):]
            if kind=="RawSeason": return f"Profile_RawSeason_{metric}"
            if kind=="RawRecent3": return f"Profile_RawRecent3_{metric}"
            if kind=="Recent3": return f"Profile_Recent3_{metric}"
            return f"Profile_{metric}"
    return ""


def _compact_profiles(full_profiles: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    if not isinstance(full_profiles,pd.DataFrame) or full_profiles.empty or "Team_Norm" not in full_profiles.columns:
        raise RuntimeError("latest NCAAF profiles unavailable")
    req=[]
    for f in feature_cols:
        pc=_direct_feature_profile_col(f)
        if pc == "":
            raise RuntimeError(f"production fixed feature is not a direct profile feature: {f}")
        if pc:
            req.append(pc)
    req=list(dict.fromkeys(req))
    missing=[c for c in req if c not in full_profiles.columns]
    if missing:
        raise RuntimeError(f"production profile columns missing: {missing[:10]}")
    return full_profiles[["Team_Norm"]+req].copy(deep=True)


def _build_fixed_features(rows: pd.DataFrame, profiles: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    cols=list(feature_cols or [])
    out=pd.DataFrame(index=rows.index)
    if not cols:
        return out
    p=profiles.copy()
    p["Team_Norm"]=p["Team_Norm"].astype(str).str.lower().str.strip()
    p=p.drop_duplicates("Team_Norm",keep="last").set_index("Team_Norm")
    home=rows.get("Home_Team_Norm",rows.get("Home_Team",pd.Series("",index=rows.index))).astype(str).str.lower().str.strip()
    away=rows.get("Away_Team_Norm",rows.get("Away_Team",pd.Series("",index=rows.index))).astype(str).str.lower().str.strip()
    cache={}
    direct_prefixes=(
        "A_RawSeason_","B_RawSeason_","Diff_RawSeason_",
        "A_RawRecent3_","B_RawRecent3_","Diff_RawRecent3_",
        "A_State_","B_State_","Diff_State_",
        "A_Recent3_","B_Recent3_","Diff_Recent3_",
    )
    def parsed(c):
        c=str(c)
        if c=="Context_Intercept": return ("I","","")
        for prefix in direct_prefixes:
            if c.startswith(prefix):
                side="Diff" if prefix.startswith("Diff_") else ("A" if prefix.startswith("A_") else "B")
                if "RawSeason" in prefix: kind="RawSeason"
                elif "RawRecent3" in prefix: kind="RawRecent3"
                elif "Recent3" in prefix: kind="Recent3"
                else: kind="State"
                return side,kind,c[len(prefix):]
        raise ValueError(f"unsupported NCAAF production feature: {c}")
    def mapped(kind,metric,which):
        key=(kind,metric,which)
        if key in cache: return cache[key]
        if kind=="RawSeason": pc=f"Profile_RawSeason_{metric}"
        elif kind=="RawRecent3": pc=f"Profile_RawRecent3_{metric}"
        elif kind=="Recent3": pc=f"Profile_Recent3_{metric}"
        else: pc=f"Profile_{metric}"
        teams=home if which=="A" else away
        v=pd.to_numeric(teams.map(p[pc]),errors="coerce") if pc in p.columns else pd.Series(np.nan,index=rows.index)
        cache[key]=pd.Series(v,index=rows.index,dtype="float64")
        return cache[key]
    for c in cols:
        side,kind,metric=parsed(c)
        if side=="I":
            out[c]=0.0
            continue
        av=mapped(kind,metric,"A"); bv=mapped(kind,metric,"B")
        out[c]=av if side=="A" else (bv if side=="B" else av-bv)
    return out[cols].replace([np.inf,-np.inf],np.nan)


def _build_h2h_pair_state(games: pd.DataFrame) -> dict:
    if games is None or games.empty:
        return {}
    g=games.copy()
    g["__date"]=pd.to_datetime(g.get("Game_Date"),errors="coerce",utc=True)
    g["__team"]=g.get("Team_Norm",pd.Series("",index=g.index)).astype(str).str.lower().str.strip()
    g["__opp"]=g.get("Opponent_Norm",pd.Series("",index=g.index)).astype(str).str.lower().str.strip()
    g["__margin"]=pd.to_numeric(g.get("Actual_Margin"),errors="coerce")
    g["__total"]=pd.to_numeric(g.get("Actual_Total"),errors="coerce")
    g=g[g["__date"].notna()&g["__team"].ne("")&g["__opp"].ne("")&g["__margin"].notna()].sort_values("__date")
    if g.empty: return {}
    lo=np.where(g["__team"].to_numpy()<=g["__opp"].to_numpy(),g["__team"].to_numpy(),g["__opp"].to_numpy())
    hi=np.where(g["__team"].to_numpy()<=g["__opp"].to_numpy(),g["__opp"].to_numpy(),g["__team"].to_numpy())
    g["__pair"]=pd.Series(lo,index=g.index).astype(str)+"|"+pd.Series(hi,index=g.index).astype(str)
    state={}
    for pair,grp in g.groupby("__pair",sort=False):
        last=grp.iloc[-1]
        state[str(pair)]={
            "prior_meetings":int(len(grp)),
            "last_team":str(last["__team"]),
            "last_margin":_f(last["__margin"]),
            "last_total":_f(last["__total"]),
            "last_date":pd.Timestamp(last["__date"]).isoformat(),
        }
    return state


def _live_h2h_features(rows: pd.DataFrame, pair_state: dict, fair_home_margin: np.ndarray) -> pd.DataFrame:
    home=rows.get("Home_Team_Norm",rows.get("Home_Team",pd.Series("",index=rows.index))).astype(str).str.lower().str.strip()
    away=rows.get("Away_Team_Norm",rows.get("Away_Team",pd.Series("",index=rows.index))).astype(str).str.lower().str.strip()
    date=pd.to_datetime(rows.get("Game_Start"),errors="coerce",utc=True)
    recs=[]
    for i,(h,a,dt) in enumerate(zip(home,away,date)):
        lo,hi=sorted((str(h),str(a))); st=(pair_state or {}).get(f"{lo}|{hi}") or {}
        n=float(st.get("prior_meetings",0) or 0)
        lm=_f(st.get("last_margin")); lt=_f(st.get("last_total")); last_team=_norm_team(st.get("last_team"))
        prev_date=pd.to_datetime(st.get("last_date"),errors="coerce",utc=True)
        days=float((dt-prev_date).total_seconds()/86400.0) if pd.notna(dt) and pd.notna(prev_date) else np.nan
        oriented=lm if last_team==_norm_team(h) else (-lm if np.isfinite(lm) else np.nan)
        same=1.0 if last_team and last_team==_norm_team(h) else (0.0 if last_team else np.nan)
        decay=float(np.exp(-max(days,0.0)/730.0)) if np.isfinite(days) else np.nan
        recs.append({
            "H2H2_Prior_Meetings":n,
            "H2H2_Prior_Margin_Current_Orientation":oriented,
            "H2H2_Prior_Total":lt,
            "H2H2_Days_Since":days,
            "H2H2_Same_Home_Role":same,
            "H2H2_Recency_Margin":oriented*decay if np.isfinite(oriented) and np.isfinite(decay) else np.nan,
            "H2H2_Recency_Total":lt*decay if np.isfinite(lt) and np.isfinite(decay) else np.nan,
            "STAT_Fair_Margin":float(fair_home_margin[i]) if i < len(fair_home_margin) and np.isfinite(fair_home_margin[i]) else np.nan,
        })
    return pd.DataFrame(recs,index=rows.index)


def build_production_contract(*, dashboard_module, stat_module, stat_out: dict,
                              frozen_spread_out: dict, totals_out: dict,
                              cadence_module, cadence_out: dict,
                              bucket_name: str, storage_client=None, log_func=print) -> dict:
    """Build and publish the explicitly promoted NCAAF Production V1 contract.

    This function is intentionally strict.  A changed research winner does not
    silently rewrite production; the promotion job fails and requires a new
    explicit production version.
    """
    if (stat_out or {}).get("status") != "PASS":
        raise RuntimeError("STAT Combo V2.1 did not pass")
    if (frozen_spread_out or {}).get("status") != "PASS":
        raise RuntimeError("Frozen Spread Edge Policy V1 did not pass")
    if (totals_out or {}).get("status") != "PASS":
        raise RuntimeError("Totals Atomic Refinement V1 did not pass")
    if (cadence_out or {}).get("status") != "PASS":
        raise RuntimeError("Refit Cadence V1 did not pass")
    if tuple(getattr(cadence_module,"SPREAD_FEATURES",())) != PROD_SPREAD_FEATURES:
        raise RuntimeError("spread production backbone drifted from cadence contract")
    if tuple(getattr(cadence_module,"TOTAL_FEATURES",())) != PROD_TOTAL_FEATURES:
        raise RuntimeError("totals production backbone drifted from cadence contract")
    if tuple(getattr(cadence_module,"H2H_FEATURES",())) != PROD_H2H_FEATURES:
        raise RuntimeError("H2H production backbone drifted from cadence contract")

    primary = (stat_out or {}).get("primary") or {}
    leader = primary.get("leader") or {}
    combo = tuple(leader.get("combo") or ())
    threshold = _f(leader.get("threshold"))
    role = str((primary.get("same_row") or {}).get("role") or "")
    if combo != EXPECTED_STAT_COMBO or not np.isclose(threshold, EXPECTED_STAT_THRESHOLD):
        raise RuntimeError(f"production freeze mismatch stat_combo={combo} threshold={threshold}")
    if str(primary.get("status")) != "TAIL_PASS" or role != "SELECTOR_OF_SHARED_EDGE":
        raise RuntimeError(f"STAT selector not promotion-ready status={primary.get('status')} role={role}")

    cache = getattr(dashboard_module, "_V1357_SPREAD_RESEARCH_CACHE", {}) or {}
    g = cache.get("games")
    candidate_cols = cache.get("candidate_feature_cols")
    if not isinstance(g, pd.DataFrame) or g.empty:
        raise RuntimeError("NCAAF spread research cache unavailable")
    g = g.copy()
    season = pd.to_numeric(g.get("Season"), errors="coerce").to_numpy(float)
    target = pd.to_numeric(g.get("Market_Error_Margin"), errors="coerce").to_numpy(float)
    if candidate_cols is None:
        prefixes=("A_Raw","B_Raw","Diff_Raw","A_State_","B_State_","Diff_State_","A_Recent3_","B_Recent3_","Diff_Recent3_","Matchup_","Context_")
        candidate_cols=[c for c in g.columns if str(c).startswith(prefixes)]
    candidate_cols=[c for c in list(dict.fromkeys(candidate_cols)) if c in g.columns]
    fam_fn=getattr(dashboard_module,"_ncaaf_stat_feature_family",None)
    if fam_fn is None:
        raise RuntimeError("missing _ncaaf_stat_feature_family")
    fam_cols=stat_module._family_columns(g,candidate_cols,fam_fn)
    raw_cols=list(fam_cols.get("rushing") or [])
    tr=np.isfinite(season)&(season<float(2026))&np.isfinite(target)
    cols=stat_module._target_free_prune(g,raw_cols,tr)
    if not cols:
        raise RuntimeError("rushing runtime selector has no production features")
    model=stat_module._new_ridge(24.0)
    model.fit(g.loc[tr,cols],target[tr])
    beta=_f(((leader.get("scales") or {}).get("rushing") or {}).get("beta"),0.0)
    if beta <= 0:
        raise RuntimeError(f"invalid frozen rushing reliability beta={beta}")

    # Freeze the small, cadence-validated probability backbones.  These are the
    # only model heads used by the live NCAAF dashboard.
    missing_sp=[c for c in PROD_SPREAD_FEATURES if c not in g.columns]
    missing_tot=[c for c in PROD_TOTAL_FEATURES if c not in g.columns]
    missing_h2h=[c for c in PROD_H2H_FEATURES if c not in g.columns]
    if missing_sp or missing_tot or missing_h2h:
        raise RuntimeError(f"production backbone columns missing spread={missing_sp} totals={missing_tot} h2h={missing_h2h}")
    oof_margin=np.asarray(cache.get("oof_margin"),dtype=float)
    oof_total=np.asarray(cache.get("oof_total"),dtype=float)
    if len(oof_margin)!=len(g) or len(oof_total)!=len(g):
        raise RuntimeError("production OOF backbone arrays unavailable")
    train_prior=np.isfinite(season)&(season<float(2026))
    spread_models,total_models=dashboard_module._ncaaf_stat_fit_models_for_rows(
        g,list(PROD_SPREAD_FEATURES),list(PROD_TOTAL_FEATURES),train_prior,target_mode="MARKET_ERROR_RESIDUAL"
    )
    actual_m=pd.to_numeric(g.get("Actual_Margin"),errors="coerce").to_numpy(dtype=float)
    actual_t=pd.to_numeric(g.get("Actual_Total"),errors="coerce").to_numpy(dtype=float)
    prior_resid_m=train_prior&np.isfinite(actual_m)&np.isfinite(oof_margin)
    prior_resid_t=train_prior&np.isfinite(actual_t)&np.isfinite(oof_total)
    resid_margin=(actual_m[prior_resid_m]-oof_margin[prior_resid_m]).astype(np.float32)
    resid_total=(actual_t[prior_resid_t]-oof_total[prior_resid_t]).astype(np.float32)
    if len(resid_margin)<500 or len(resid_total)<500:
        raise RuntimeError(f"production residual pools too small margin={len(resid_margin)} total={len(resid_total)}")

    h2h_X=g[list(PROD_H2H_FEATURES)].copy()
    h2h_X["STAT_Fair_Margin"]=oof_margin
    h2h_train=train_prior&np.isfinite(actual_m)&np.isfinite(oof_margin)
    if int(h2h_train.sum())<500:
        raise RuntimeError(f"H2H production training rows too small n={int(h2h_train.sum())}")
    h2h_model=_new_h2h_pipe()
    h2h_model.fit(h2h_X.loc[h2h_train,list(PROD_H2H_FEATURES)+["STAT_Fair_Margin"]],(actual_m[h2h_train]>0).astype(int))
    h2h_pair_state=_build_h2h_pair_state(g)

    stat_cache=getattr(dashboard_module,"_NCAAF_STAT_TRAIN_CACHE",{}) or {}
    stat_brain=stat_cache.get("bundle") if isinstance(stat_cache,dict) else None
    full_profiles=(stat_brain or {}).get("latest_profiles") if isinstance(stat_brain,dict) else None
    live_feature_union=list(PROD_SPREAD_FEATURES)+list(PROD_TOTAL_FEATURES)+list(cols)
    profiles=_compact_profiles(full_profiles,live_feature_union)

    spread_systems = (((cache.get("system_miner_v2") or {}).get("spreads") or {}).get("systems") or [])
    spread_authority = {}
    for fid,spec in (frozen_spread_out.get("authority_families") or {}).items():
        rec={
            "family_id":fid,
            "root":spec.get("root"),
            "orientation":spec.get("orientation"),
            "mechanisms":list(spec.get("mechanisms") or []),
            "aliases":list(spec.get("aliases") or []),
            "evidence":spec.get("evidence"),
        }
        root=str(spec.get("root") or "")
        if root.startswith("MINER:"):
            sid=root.split(":",1)[1]
            srec=_find_system(spread_systems,sid)
            if not srec:
                raise RuntimeError(f"frozen spread miner rule missing from current registry: {sid}")
            rec["conditions"]=list(srec.get("conditions") or [])
            rec["system_direction"]=str(srec.get("direction") or "")
        spread_authority[fid]=rec

    repeat_fams=[]
    reps=set()
    for fam in totals_out.get("families") or []:
        if not fam.get("repeatable"): continue
        rep=fam.get("representative") or {}
        rid=str(rep.get("id") or "")
        if rid not in EXPECTED_TOTAL_REPRESENTATIVES:
            continue
        if not str(rep.get("refine_state") or "").startswith("REPEATABLE_"):
            raise RuntimeError(f"totals representative lost repeatability: {rid}")
        reps.add(rid)
        repeat_fams.append({
            "family_id":str(fam.get("family_id") or ""),
            "representative":rid,
            "conditions":list(rep.get("conditions") or []),
            "orientation":str(rep.get("chosen") or "PLAY"),
            "mechanisms":list(fam.get("mechanisms") or []),
            "evidence":str(rep.get("refine_state") or ""),
        })
    if reps != EXPECTED_TOTAL_REPRESENTATIVES or len(repeat_fams) != 3:
        raise RuntimeError(f"totals production freeze mismatch representatives={sorted(reps)}")

    sm = frozen_spread_out.get("metrics") or {}
    tsm = totals_out.get("stack_metrics") or {}
    performance={
        "spread":{
            "discovery_edge_single":_clean_metrics(sm.get(("DISCOVERY","EDGE_SINGLE"))),
            "confirm_2026_edge_single":_clean_metrics(sm.get(("CONFIRM_2026","EDGE_SINGLE"))),
            "discovery_edge_multi":_clean_metrics(sm.get(("DISCOVERY","EDGE_MULTI"))),
            "confirm_2026_edge_multi":_clean_metrics(sm.get(("CONFIRM_2026","EDGE_MULTI"))),
        },
        "totals":{
            "discovery_one_family":_clean_metrics(tsm.get(("DISCOVERY",1))),
            "confirm_2026_one_family":_clean_metrics(tsm.get(("CONFIRM_2026",1))),
            "discovery_two_mechanism":_clean_metrics(tsm.get(("DISCOVERY",2))),
            "confirm_2026_two_mechanism":_clean_metrics(tsm.get(("CONFIRM_2026",2))),
        },
        "h2h":{"production_authority":0,"reason":"NO_REPEATABLE_EDGE_CANDIDATE"},
    }

    contract={
        "status":"PRODUCTION",
        "source_tag":NCAAF_PRODUCTION_V1_SOURCE_TAG,
        "published_utc":datetime.now(timezone.utc).isoformat(),
        "research_freeze_utc":NCAAF_PRODUCTION_V1_FREEZE_UTC,
        "production_authority":1,
        "probability_models":"FROZEN_AND_SEPARATE",
        "probability_runtime":"FIXED_FEATURE_CONTRACT_NO_AUTOFS_NO_MULTIHEAD",
        "probability_rewrite":False,
        "profiles":profiles,
        "backbones":{
            "spread":{
                "feature_cols":list(PROD_SPREAD_FEATURES),"models":spread_models,
                "linear_weight":PROD_LINEAR_WEIGHT,"residuals":resid_margin,
                "target":"ACTUAL_MARGIN_MINUS_OPEN_MARKET_MARGIN","cadence":"FROZEN",
            },
            "totals":{
                "feature_cols":list(PROD_TOTAL_FEATURES),"models":total_models,
                "linear_weight":PROD_LINEAR_WEIGHT,"residuals":resid_total,
                "target":"ACTUAL_TOTAL_MINUS_OPEN_MARKET_TOTAL","cadence":"FROZEN",
            },
            "h2h":{
                "feature_cols":list(PROD_H2H_FEATURES)+["STAT_Fair_Margin"],
                "model":h2h_model,"pair_state":h2h_pair_state,
                "target":"HOME_WIN","cadence":"FROZEN",
            },
        },
        "spread":{
            "production_authority":1,
            "policy":"FROZEN_EDGE_FAMILY_SELECTOR",
            "authority_families":spread_authority,
            "stat_selector":{
                "family":"rushing","combo":["rushing"],"threshold":float(threshold),
                "beta":float(beta),"alpha":24.0,"feature_cols":list(cols),"model":model,
                "train_seasons":"<2026","role":"SELECTOR_OF_SHARED_EDGE",
                "direction_rule":"SCALED_MARKET_ERROR_SIGN",
            },
            "conflict_rule":"PASS",
            "multi_rule":"AT_LEAST_2_INDEPENDENT_MECHANISMS",
        },
        "h2h":{
            "production_authority":0,
            "policy":"MODEL_ONLY",
            "reason":"54 candidates tested; 0 repeatable edge candidates",
        },
        "totals":{
            "production_authority":1,
            "policy":"REPEATABLE_FAMILY_COLLAPSED_SINGLE_AUTHORITY",
            "families":repeat_fams,
            "conflict_rule":"PASS",
            "multi_strength_bonus":False,
            "multi_strength_reason":"2026 two-mechanism confirmation did not repeat; family overlap does not increase authority",
        },
        "performance":performance,
        "cadence":{
            "production":"FROZEN",
            "reason":"cadence RMSE gains were too small to justify production churn; cadence remains research-only",
            "decisions":cadence_out.get("decisions") or {},
            "frozen_summary":((cadence_out.get("summaries") or {}).get("FROZEN") or {}),
        },
    }

    client=storage_client or storage.Client()
    raw=pickle.dumps(contract,protocol=pickle.HIGHEST_PROTOCOL)
    payload=gzip.compress(raw,compresslevel=6)
    blob=client.bucket(bucket_name).blob(NCAAF_PRODUCTION_V1_ARTIFACT)
    blob.upload_from_string(payload,content_type="application/gzip")
    meta={
        "status":contract["status"],"source_tag":contract["source_tag"],
        "published_utc":contract["published_utc"],"research_freeze_utc":contract["research_freeze_utc"],
        "production_authority":1,"spread_authority_families":len(spread_authority),
        "spread_stat_family":"rushing","spread_stat_threshold":float(threshold),
        "spread_stat_features":len(cols),"totals_authority_families":len(repeat_fams),
        "spread_probability_features":len(PROD_SPREAD_FEATURES),"totals_probability_features":len(PROD_TOTAL_FEATURES),
        "h2h_probability_features":len(PROD_H2H_FEATURES)+1,"probability_runtime":"FIXED_FEATURE_CONTRACT_NO_AUTOFS_NO_MULTIHEAD",
        "h2h_production_authority":0,"artifact":NCAAF_PRODUCTION_V1_ARTIFACT,
    }
    client.bucket(bucket_name).blob(NCAAF_PRODUCTION_V1_META).upload_from_string(
        json.dumps(meta,indent=2,sort_keys=True),content_type="application/json"
    )
    log_func(
        f"[NCAAF-PROD-V1-PUBLISH] status=PASS source_tag={NCAAF_PRODUCTION_V1_SOURCE_TAG} "
        f"artifact=gs://{bucket_name}/{NCAAF_PRODUCTION_V1_ARTIFACT} spread_families={len(spread_authority)} "
        f"stat_family=rushing stat_features={len(cols)} beta={beta:.4f} threshold={threshold:.2f} "
        f"probability_features=spread:{len(PROD_SPREAD_FEATURES)},h2h:{len(PROD_H2H_FEATURES)+1},totals:{len(PROD_TOTAL_FEATURES)} "
        f"runtime_autofs=FALSE multi_head=FALSE totals_families={len(repeat_fams)} h2h_edge_authority=0 production_authority=1"
    )
    return contract


def load_production_contract(bucket_name: str = "sharp-models", storage_client=None) -> dict | None:
    try:
        client=storage_client or storage.Client()
        blob=client.bucket(bucket_name).blob(NCAAF_PRODUCTION_V1_ARTIFACT)
        payload=blob.download_as_bytes()
        obj=pickle.loads(gzip.decompress(payload))
        if not isinstance(obj,dict) or obj.get("source_tag") != NCAAF_PRODUCTION_V1_SOURCE_TAG or int(obj.get("production_authority",0) or 0)!=1:
            return None
        return obj
    except Exception:
        return None


def score_live_rows(rows: pd.DataFrame, contract: dict | None) -> pd.DataFrame:
    """Score live NCAAF rows with the compact frozen production backbones.

    No research model, AutoFS selector, multi-head resolver, rich-market model, or
    challenger search runs here.  Only the exact fixed feature contracts stored in
    the promoted production artifact are materialized.
    """
    out=rows.copy()
    out["_model_prob"]=np.nan
    out["_model_id"]=""
    out["_prob_source"]=""
    out["NCAAF_Stat_Expected_Margin"]=np.nan
    out["NCAAF_Stat_Expected_Total"]=np.nan
    out["NCAAF_Prod_Stat_Selector_Margin"]=np.nan
    out["NCAAF_Prod_Stat_Selector_Active"]=0
    if out.empty or not isinstance(contract,dict) or int(contract.get("production_authority",0) or 0)!=1:
        return out
    profiles=contract.get("profiles")
    backs=contract.get("backbones") or {}
    if not isinstance(profiles,pd.DataFrame) or profiles.empty:
        return out

    market=out.get("Market",pd.Series("",index=out.index)).astype(str).str.lower().str.strip()
    home=out.get("Home_Team_Norm",out.get("Home_Team",pd.Series("",index=out.index))).astype(str).str.lower().str.strip()
    away=out.get("Away_Team_Norm",out.get("Away_Team",pd.Series("",index=out.index))).astype(str).str.lower().str.strip()
    outcome=out.get("Outcome_Norm",out.get("Outcome",pd.Series("",index=out.index))).astype(str).str.lower().str.strip()
    is_home=outcome.eq(home)|outcome.eq("home")
    is_away=outcome.eq(away)|outcome.eq("away")

    def canonical_open_spread(ix):
        z=out.loc[ix]
        for c in ("Consensus_Open_Spread","Opening_Spread"):
            if c in z.columns:
                v=pd.to_numeric(z[c],errors="coerce")
                if v.notna().any(): return v.to_numpy(dtype=float)
        return np.full(len(z),np.nan,dtype=float)

    # SPREAD probability backbone + frozen STAT edge selector.
    six=out.index[market.eq("spreads")]
    if len(six):
        b=backs.get("spread") or {}; cols=list(b.get("feature_cols") or [])
        X=_build_fixed_features(out.loc[six],profiles,cols)
        residual_edge=_blend_predict(b.get("models"),X,float(b.get("linear_weight",PROD_LINEAR_WEIGHT)))
        open_sp=canonical_open_spread(six); fair_home=(-open_sp)+residual_edge
        cur=pd.to_numeric(out.loc[six].get("Value"),errors="coerce").to_numpy(dtype=float)
        ih=is_home.loc[six].to_numpy(dtype=bool); ia=is_away.loc[six].to_numpy(dtype=bool)
        side_fair=np.where(ih,fair_home,np.where(ia,-fair_home,np.nan))
        prob=np.full(len(six),np.nan,dtype=float); resid=np.asarray(b.get("residuals") if b.get("residuals") is not None else [],dtype=float)
        if ih.any(): prob[ih]=_empirical_prob_gt(-(side_fair[ih]+cur[ih]),resid)
        if ia.any(): prob[ia]=_empirical_prob_gt(-(side_fair[ia]+cur[ia]),-resid)
        out.loc[six,"_model_prob"]=prob
        out.loc[six,"_model_id"]="NCAAF_SPREAD_FIXED_V1"
        out.loc[six,"_prob_source"]="NCAAF Spread Fixed V1"
        out.loc[six,"NCAAF_Stat_Prob"]=prob
        out.loc[six,"NCAAF_Stat_Expected_Margin"]=side_fair

        spec=((contract.get("spread") or {}).get("stat_selector") or {})
        scols=list(spec.get("feature_cols") or []); smodel=spec.get("model")
        if scols and smodel is not None:
            try:
                SX=_build_fixed_features(out.loc[six],profiles,scols)
                sval=np.asarray(smodel.predict(SX[scols]),dtype=float)*float(spec.get("beta",0.0) or 0.0)
                th=float(spec.get("threshold",EXPECTED_STAT_THRESHOLD) or EXPECTED_STAT_THRESHOLD)
                out.loc[six,"NCAAF_Prod_Stat_Selector_Margin"]=sval
                out.loc[six,"NCAAF_Prod_Stat_Selector_Active"]=(np.isfinite(sval)&(np.abs(sval)>=th)).astype("int8")
            except Exception:
                pass

    # TOTALS probability backbone.
    tix=out.index[market.eq("totals")]
    if len(tix):
        b=backs.get("totals") or {}; cols=list(b.get("feature_cols") or [])
        X=_build_fixed_features(out.loc[tix],profiles,cols)
        residual_edge=_blend_predict(b.get("models"),X,float(b.get("linear_weight",PROD_LINEAR_WEIGHT)))
        z=out.loc[tix]
        open_total=None
        for c in ("Consensus_Open_Total","Opening_Total"):
            if c in z.columns:
                v=pd.to_numeric(z[c],errors="coerce")
                if v.notna().any(): open_total=v.to_numpy(dtype=float); break
        if open_total is None: open_total=np.full(len(z),np.nan,dtype=float)
        fair_total=open_total+residual_edge
        cur=pd.to_numeric(z.get("Value"),errors="coerce").to_numpy(dtype=float)
        resid=np.asarray(b.get("residuals") if b.get("residuals") is not None else [],dtype=float)
        p_over=_empirical_prob_gt(cur-fair_total,resid)
        outc=outcome.loc[tix]
        prob=np.where(outc.eq("under").to_numpy(),1.0-p_over,p_over)
        out.loc[tix,"_model_prob"]=prob
        out.loc[tix,"_model_id"]="NCAAF_TOTALS_FIXED_V1"
        out.loc[tix,"_prob_source"]="NCAAF Totals Fixed V1"
        out.loc[tix,"NCAAF_Stat_Prob"]=prob
        out.loc[tix,"NCAAF_Stat_Expected_Total"]=fair_total

    # H2H probability backbone.  This is a separate logistic model with its own
    # historical matchup features plus the frozen Spread fair-margin anchor.
    hix=out.index[market.eq("h2h")]
    if len(hix):
        b=backs.get("h2h") or {}; model=b.get("model")
        sb=backs.get("spread") or {}; scols=list(sb.get("feature_cols") or [])
        Xs=_build_fixed_features(out.loc[hix],profiles,scols)
        sedge=_blend_predict(sb.get("models"),Xs,float(sb.get("linear_weight",PROD_LINEAR_WEIGHT)))
        open_sp=canonical_open_spread(hix); fair_home=(-open_sp)+sedge
        HX=_live_h2h_features(out.loc[hix],b.get("pair_state") or {},fair_home)
        hcols=list(b.get("feature_cols") or [])
        ph=np.full(len(hix),np.nan,dtype=float)
        if model is not None and hcols:
            try: ph=np.asarray(model.predict_proba(HX[hcols])[:,1],dtype=float)
            except Exception: pass
        ih=is_home.loc[hix].to_numpy(dtype=bool); ia=is_away.loc[hix].to_numpy(dtype=bool)
        prob=np.where(ih,ph,np.where(ia,1.0-ph,np.nan))
        side_fair=np.where(ih,fair_home,np.where(ia,-fair_home,np.nan))
        out.loc[hix,"_model_prob"]=prob
        out.loc[hix,"_model_id"]="NCAAF_H2H_FIXED_V1"
        out.loc[hix,"_prob_source"]="NCAAF H2H Fixed V1"
        out.loc[hix,"NCAAF_Stat_Prob"]=prob
        out.loc[hix,"NCAAF_Stat_Expected_Margin"]=side_fair
    return out


def attach_spread_stat_selector(rows: pd.DataFrame, sb: dict, contract: dict,
                                runtime_feature_builder) -> pd.DataFrame:
    out=rows.copy()
    out["NCAAF_Prod_Stat_Selector_Margin"]=np.nan
    out["NCAAF_Prod_Stat_Selector_Active"]=0
    spec=((contract or {}).get("spread") or {}).get("stat_selector") or {}
    model=spec.get("model"); cols=list(spec.get("feature_cols") or [])
    beta=_f(spec.get("beta"),0.0); threshold=_f(spec.get("threshold"),np.nan)
    if model is None or not cols or beta<=0 or not np.isfinite(threshold):
        return out
    try:
        X=runtime_feature_builder(out,sb,cols)
        raw=np.asarray(model.predict(X[cols]),dtype=float)
        pred=raw*beta
        out["NCAAF_Prod_Stat_Selector_Margin"]=pred
        out["NCAAF_Prod_Stat_Selector_Active"]=(np.isfinite(pred)&(np.abs(pred)>=threshold)).astype("int8")
    except Exception:
        pass
    return out


def _spread_votes(group: pd.DataFrame, contract: dict) -> Tuple[List[dict], List[str]]:
    home,away=_home_away(group)
    if not home or not away:
        return [],["MISSING_HOME_AWAY"]
    votes=[]; diag=[]
    specs=(((contract or {}).get("spread") or {}).get("authority_families") or {})

    # 1) Frozen STAT selector: positive market-error residual = home side; negative = away side.
    stat_spec=specs.get("SPREAD_STAT_COMBO") or {}
    sel=pd.to_numeric(group.get("NCAAF_Prod_Stat_Selector_Margin"),errors="coerce")
    vals=sel[np.isfinite(sel)] if isinstance(sel,pd.Series) else pd.Series(dtype=float)
    threshold=_f((((contract or {}).get("spread") or {}).get("stat_selector") or {}).get("threshold"),0.75)
    if not vals.empty:
        v=float(vals.iloc[0])
        if abs(v)>=threshold:
            votes.append({"family_id":"SPREAD_STAT_COMBO","target":home if v>0 else away,"mechanisms":list(stat_spec.get("mechanisms") or ["STATISTICAL_MATCHUP"]),"source":f"STAT rushing {v:+.2f}"})

    # 2) Named Big Al / Pathi rules are side-aware live flags.
    def side_flag_votes(col,fid,fade=False):
        if col not in group.columns: return
        targets=[]
        for _,r in group.iterrows():
            val=pd.to_numeric(pd.Series([r.get(col)]),errors="coerce").fillna(0).iloc[0]
            if float(val)!=1.0: continue
            base=_outcome_norm(r)
            if base not in {home,away}: continue
            targets.append(_opposite_team(base,home,away) if fade else base)
        targets=sorted(set(x for x in targets if x))
        if len(targets)==1:
            sp=specs.get(fid) or {}
            votes.append({"family_id":fid,"target":targets[0],"mechanisms":list(sp.get("mechanisms") or []),"source":fid})
        elif len(targets)>1:
            diag.append(f"{fid}:INTERNAL_CONFLICT")
    side_flag_votes("BigAl_CF1_Week2Home42Win","BIGAL_CF1_WEEK2_HOME42",False)
    side_flag_votes("Pathi_FB_Crossed_Key_Away_From_Team","PATHI_CROSSED_KEY_AWAY_FADE",True)

    # 3) Frozen Miner roots are evaluated in the same canonical home orientation
    # used by the historical one-game-per-row research frame.
    ctx=_home_context_row(group)
    for fid in ("MINER_HIGH_TOTAL_60_PLUS","MINER_DOG_3_7","MINER_CONF_BIG12_HOME_FAVORITE"):
        sp=specs.get(fid) or {}
        cond=sp.get("conditions") or []
        if not _rule_true(ctx,cond): continue
        direction=str(sp.get("system_direction") or "PLAY_ON").upper()
        base=home
        target=base if direction=="PLAY_ON" else away
        votes.append({"family_id":fid,"target":target,"mechanisms":list(sp.get("mechanisms") or []),"source":fid})
    return votes,diag


def _totals_votes(group: pd.DataFrame, contract: dict, context_group: pd.DataFrame | None = None) -> Tuple[List[dict], List[str]]:
    fams=(((contract or {}).get("totals") or {}).get("families") or [])
    # Totals families were discovered against canonical OVER/UNDER target.  Use
    # the spread-side game context when available because conference/home/favorite
    # fields are guaranteed there even when the live totals quote rows are sparse.
    # The totals decision still targets only OVER/UNDER rows.
    ctx_source=context_group if isinstance(context_group,pd.DataFrame) and not context_group.empty else group
    ctx=_home_context_row(ctx_source)
    votes=[]; diag=[]
    for fam in fams:
        if not _rule_true(ctx,fam.get("conditions") or []): continue
        ori=str(fam.get("orientation") or "PLAY").upper()
        target="over" if ori=="PLAY" else "under"
        votes.append({"family_id":fam.get("family_id"),"target":target,"mechanisms":list(fam.get("mechanisms") or []),"source":str(fam.get("representative") or fam.get("family_id"))})
    return votes,diag


def apply_live_authority(sides: pd.DataFrame, contract: dict | None) -> pd.DataFrame:
    """Attach promoted edge action to one-best-quote-per-outcome rows.

    Probability columns are untouched.  The edge engine only chooses whether a
    market has production edge authority and which outcome that authority points to.
    """
    out=sides.copy()
    defaults={
        "_prod_decision":"MODEL_ONLY","_prod_target":"","_prod_action":"MODEL ONLY",
        "_prod_sources":"","_prod_mechanisms":"","_prod_family_count":0,
        "_prod_independent_mechanisms":0,"_prod_authority":0,"_prod_reason":"PRODUCTION_CONTRACT_UNAVAILABLE",
    }
    for c,v in defaults.items(): out[c]=v
    if not isinstance(contract,dict) or int(contract.get("production_authority",0) or 0)!=1:
        return out
    gcols=[c for c in ("Game_Key","Market") if c in out.columns]
    if len(gcols)<2:
        return out
    for _,ix in out.groupby(gcols,dropna=False,sort=False).groups.items():
        idx=list(ix); g=out.loc[idx].copy(); market=str(g["Market"].iloc[0]).lower()
        if market=="h2h":
            out.loc[idx,"_prod_decision"]="MODEL_ONLY"
            out.loc[idx,"_prod_action"]="MODEL ONLY"
            out.loc[idx,"_prod_reason"]="H2H_EDGE_AUTHORITY_CLOSED"
            continue
        if market=="spreads":
            votes,diag=_spread_votes(g,contract)
        else:
            game_key=g["Game_Key"].iloc[0] if "Game_Key" in g.columns else None
            spread_ctx=out[(out.get("Game_Key")==game_key)&(out.get("Market").astype(str).str.lower()=="spreads")].copy() if game_key is not None else pd.DataFrame()
            votes,diag=_totals_votes(g,contract,spread_ctx)
        if not votes:
            out.loc[idx,"_prod_decision"]="PASS"
            out.loc[idx,"_prod_action"]="PASS"
            out.loc[idx,"_prod_reason"]="NO_PROMOTED_EDGE_FAMILY"
            continue
        targets=sorted(set(_norm_team(v.get("target")) for v in votes if _norm_team(v.get("target"))))
        sources=" | ".join(str(v.get("source")) for v in votes)
        mechs=sorted({m for v in votes for m in (v.get("mechanisms") or [])})
        if len(targets)!=1 or diag:
            out.loc[idx,"_prod_decision"]="PASS_CONFLICT"
            out.loc[idx,"_prod_action"]="PASS — CONFLICT"
            out.loc[idx,"_prod_sources"]=sources
            out.loc[idx,"_prod_mechanisms"]=" + ".join(mechs)
            out.loc[idx,"_prod_family_count"]=len(votes)
            out.loc[idx,"_prod_reason"]="VALIDATED_EDGE_CONFLICT"
            continue
        target=targets[0]
        if market=="spreads":
            independent=_maximum_matching(votes)
            decision="EDGE_MULTI" if independent>=2 else "EDGE_SINGLE"
            action="STRONG PLAY" if decision=="EDGE_MULTI" else "PLAY"
        else:
            # Two-mechanism totals overlap were explicitly NOT promoted as a
            # stronger state after 2026 failed to confirm the discovery boost.
            independent=1
            decision="EDGE_SINGLE"
            action="PLAY"
        out.loc[idx,"_prod_decision"]=decision
        out.loc[idx,"_prod_target"]=target
        out.loc[idx,"_prod_action"]=action
        out.loc[idx,"_prod_sources"]=sources
        out.loc[idx,"_prod_mechanisms"]=" + ".join(mechs)
        out.loc[idx,"_prod_family_count"]=len(votes)
        out.loc[idx,"_prod_independent_mechanisms"]=int(independent)
        out.loc[idx,"_prod_authority"]=1
        out.loc[idx,"_prod_reason"]="PROMOTED_EDGE_AUTHORITY"
    return out


def select_market_rows(sides: pd.DataFrame) -> pd.DataFrame:
    """Choose the promoted edge side when one exists; otherwise show model leader."""
    if sides is None or sides.empty:
        return pd.DataFrame()
    picked=[]
    gcols=[c for c in ("Game_Key","Market") if c in sides.columns]
    if len(gcols)<2:
        return sides.copy()
    for _,g in sides.groupby(gcols,dropna=False,sort=False):
        dec=str(g.get("_prod_decision",pd.Series("",index=g.index)).iloc[0])
        target=_norm_team(g.get("_prod_target",pd.Series("",index=g.index)).iloc[0])
        chosen=None
        if dec in {"EDGE_SINGLE","EDGE_MULTI"} and target:
            on=g.apply(_outcome_norm,axis=1)
            hit=g.loc[on.eq(target)]
            if not hit.empty:
                chosen=hit.iloc[0]
        if chosen is None:
            gg=g.copy()
            gg["__edge"]=pd.to_numeric(gg.get("_edge"),errors="coerce").fillna(-999.0)
            gg["__ev"]=pd.to_numeric(gg.get("_ev"),errors="coerce").fillna(-999.0)
            gg["__pred"]=pd.to_numeric(gg.get("_pred"),errors="coerce").fillna(-999.0)
            gg=gg.sort_values(["__edge","__ev","__pred"],ascending=[False,False,False])
            chosen=gg.iloc[0].drop(labels=["__edge","__ev","__pred"],errors="ignore")
        picked.append(chosen)
    return pd.DataFrame(picked).reset_index(drop=True)
