"""STAT_COMBINATION_V2 — consensus-tail NCAAF market-error research.

V1 taught us that broad family models and equal-weight averages do not improve the
market globally.  V2 therefore asks a narrower question: when multiple independent
football families agree that the opening spread is materially wrong, is that tail
signal stable out of sample?

Research only. Zero production authority.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import math
import numpy as np
import pandas as pd

SCV2_SOURCE_TAG = "stat-combination-v2-consensus-tail-market-error"
SCV2_DISCOVERY_START = 2023
SCV2_CONFIRMATION_SEASON = 2026
SCV2_MAX_FAMILY_FEATURES = 36
SCV2_MIN_FEATURE_COVERAGE = 0.45
SCV2_CORR_CUTOFF = 0.965
SCV2_MIN_TRAIN_ROWS = 500
SCV2_MIN_VALID_ROWS = 100
SCV2_MIN_SCALE_ROWS = 100
SCV2_THRESHOLDS = (0.75, 1.0, 1.5, 2.0, 2.5, 3.0)
SCV2_MIN_TAIL_PRIOR_N = 80
SCV2_MIN_TAIL_SEASON_N = 20
SCV2_BREAK_EVEN = 110.0 / 210.0

# The current CORE-derived dynamic-strength residual was decisively harmful in V1
# (about -1.31 RMSE points vs market in discovery and -3.58 in 2026).  The concept
# can be rebuilt later, but this implementation is retired rather than allowed to
# create accidental cancellation inside averages.
SCV2_RETIRED_COMPONENTS = ("dynamic_strength_v1",)

SCV2_ALLOWED_FAMILIES = (
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

# Predeclared football-rationale combinations.  This deliberately avoids testing
# every possible subset.  Systems (Big Al/Pathi) remain independent overlays.
SCV2_COMBOS: Tuple[Tuple[str, ...], ...] = (
    ("efficiency",), ("passing",), ("rushing",), ("opponent_adjusted",),
    ("matchup",), ("context",), ("down_conversion_proxy",), ("turnovers",),
    ("tempo_play_mix",),
    ("efficiency", "passing"), ("efficiency", "rushing"),
    ("efficiency", "opponent_adjusted"), ("efficiency", "matchup"),
    ("passing", "rushing"), ("passing", "opponent_adjusted"),
    ("passing", "matchup"), ("rushing", "opponent_adjusted"),
    ("rushing", "matchup"), ("opponent_adjusted", "matchup"),
    ("efficiency", "context"), ("passing", "context"),
    ("rushing", "context"), ("opponent_adjusted", "context"),
    ("efficiency", "down_conversion_proxy"),
    ("passing", "down_conversion_proxy"), ("rushing", "turnovers"),
    ("efficiency", "tempo_play_mix"),
    ("efficiency", "passing", "opponent_adjusted"),
    ("efficiency", "rushing", "opponent_adjusted"),
    ("passing", "rushing", "opponent_adjusted"),
    ("efficiency", "passing", "matchup"),
    ("efficiency", "rushing", "matchup"),
    ("passing", "rushing", "matchup"),
    ("efficiency", "opponent_adjusted", "context"),
    ("passing", "opponent_adjusted", "context"),
    ("rushing", "opponent_adjusted", "context"),
    ("context", "passing", "rushing"),
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
    tr = np.asarray(train_mask, dtype=bool)
    usable = []
    for c in cols:
        if c not in g.columns:
            continue
        s = pd.to_numeric(g.loc[tr, c], errors="coerce")
        cov = float(s.notna().mean()) if len(s) else 0.0
        if cov < SCV2_MIN_FEATURE_COVERAGE or int(s.nunique(dropna=True)) < 5:
            continue
        sd = float(s.std(skipna=True))
        if not np.isfinite(sd) or sd <= 1e-10:
            continue
        usable.append((cov, c))
    usable.sort(key=lambda z: (-z[0], z[1]))
    pre = [c for _, c in usable[: max(SCV2_MAX_FAMILY_FEATURES * 3, SCV2_MAX_FAMILY_FEATURES)]]
    if not pre:
        return []
    x = g.loc[tr, pre].apply(pd.to_numeric, errors="coerce")
    kept: List[str] = []
    for c in pre:
        if len(kept) >= SCV2_MAX_FAMILY_FEATURES:
            break
        reject = False
        for k in kept:
            pair = pd.concat([x[c], x[k]], axis=1).dropna()
            if len(pair) < 100:
                continue
            corr = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
            if np.isfinite(corr) and abs(corr) >= SCV2_CORR_CUTOFF:
                reject = True
                break
        if not reject:
            kept.append(c)
    return kept


def _family_columns(g: pd.DataFrame, candidate_cols: List[str], family_fn) -> Dict[str, List[str]]:
    out = {f: [] for f in SCV2_ALLOWED_FAMILIES}
    for c in candidate_cols:
        if c not in g.columns:
            continue
        fam = str(family_fn(c))
        if fam in out:
            out[fam].append(c)
    return {k: v for k, v in out.items() if v}


def _family_oof(g: pd.DataFrame, family_cols: Dict[str, List[str]], target: np.ndarray,
                season: np.ndarray, log_func=print) -> Dict[str, np.ndarray]:
    preds = {fam: np.full(len(g), np.nan, dtype=float) for fam in family_cols}
    years = sorted(int(x) for x in pd.Series(season).dropna().unique())
    for sy in years:
        if sy < SCV2_DISCOVERY_START:
            continue
        tr = np.isfinite(season) & (season < float(sy)) & np.isfinite(target)
        va = np.isfinite(season) & (season == float(sy)) & np.isfinite(target)
        if int(tr.sum()) < SCV2_MIN_TRAIN_ROWS or int(va.sum()) < SCV2_MIN_VALID_ROWS:
            continue
        for fam, raw_cols in family_cols.items():
            cols = _target_free_prune(g, raw_cols, tr)
            if not cols:
                continue
            try:
                model = _new_ridge(24.0)
                model.fit(g.loc[tr, cols], target[tr])
                preds[fam][np.where(va)[0]] = np.asarray(model.predict(g.loc[va, cols]), dtype=float)
                log_func(
                    f"[STAT-COMBO-V2-FAMILY-FOLD] family={fam} season={sy} train_n={int(tr.sum())} valid_n={int(va.sum())} "
                    f"raw_features={len(raw_cols)} retained_features={len(cols)} target_free_pruning=TRUE"
                )
            except Exception as e:
                log_func(f"[STAT-COMBO-V2-FAMILY-FOLD] family={fam} season={sy} status=FAILED error={type(e).__name__}:{e}")
    return preds


def _metrics(target: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    t = np.asarray(target, float); p = np.asarray(pred, float)
    m = np.asarray(mask, bool) & np.isfinite(t) & np.isfinite(p)
    if int(m.sum()) == 0:
        return {"n":0,"rmse":np.nan,"mae":np.nan,"hit":np.nan,"roi":np.nan,"signed":np.nan,"corr":np.nan}
    tt, pp = t[m], p[m]
    nz = ~np.isclose(tt, 0.0, atol=1e-9) & ~np.isclose(pp, 0.0, atol=1e-12)
    hit = float(np.mean(np.sign(pp[nz]) == np.sign(tt[nz]))) if nz.any() else np.nan
    signed = float(np.mean(np.sign(pp[nz]) * tt[nz])) if nz.any() else np.nan
    corr = float(np.corrcoef(tt, pp)[0,1]) if len(tt)>=3 and np.nanstd(tt)>0 and np.nanstd(pp)>0 else np.nan
    return {
        "n": int(m.sum()),
        "rmse": float(np.sqrt(np.mean((tt-pp)**2))),
        "mae": float(np.mean(np.abs(tt-pp))),
        "hit": hit, "roi": _roi(hit), "signed": signed, "corr": corr,
    }


def _market_metrics(target: np.ndarray, mask: np.ndarray) -> dict:
    t = np.asarray(target, float); m = np.asarray(mask, bool) & np.isfinite(t)
    if not m.any(): return {"n":0,"rmse":np.nan,"mae":np.nan}
    tt=t[m]
    return {"n":int(m.sum()),"rmse":float(np.sqrt(np.mean(tt**2))),"mae":float(np.mean(np.abs(tt)))}


def _reliability_beta(target: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    t=np.asarray(target,float); p=np.asarray(pred,float)
    m=np.asarray(mask,bool)&np.isfinite(t)&np.isfinite(p)
    if int(m.sum()) < SCV2_MIN_SCALE_ROWS:
        return {"n":int(m.sum()),"beta":0.0,"corr":np.nan}
    tt=t[m]; pp=p[m]
    var=float(np.var(pp))
    if not np.isfinite(var) or var <= 1e-12:
        return {"n":int(m.sum()),"beta":0.0,"corr":np.nan}
    cov=float(np.mean((pp-np.mean(pp))*(tt-np.mean(tt))))
    beta=float(np.clip(cov/var,0.0,1.0))
    corr=float(np.corrcoef(tt,pp)[0,1]) if np.std(tt)>0 and np.std(pp)>0 else np.nan
    return {"n":int(m.sum()),"beta":beta,"corr":corr}


def _scaled_family_preds(family_preds: Dict[str,np.ndarray], target: np.ndarray, prior: np.ndarray, log_func, target_season:int):
    out={}; scales={}
    for fam,pred in sorted(family_preds.items()):
        z=_reliability_beta(target,pred,prior)
        scales[fam]=z
        out[fam]=np.asarray(pred,float)*float(z["beta"])
        log_func(
            f"[STAT-COMBO-V2-FAMILY-SCALE] target_season={target_season} family={fam} prior_n={z['n']} "
            f"beta={z['beta']:.4f} corr={z['corr'] if np.isfinite(z['corr']) else np.nan:.4f} orientation=NO_FLIP"
        )
    return out,scales


def _combo_prediction(scaled_preds:Dict[str,np.ndarray], combo:Tuple[str,...]):
    if any(f not in scaled_preds for f in combo):
        return None,None
    a=np.column_stack([np.asarray(scaled_preds[f],float) for f in combo])
    finite=np.all(np.isfinite(a),axis=1)
    pred=np.full(len(a),np.nan,float)
    pred[finite]=np.mean(a[finite],axis=1)
    if len(combo)==1:
        agree=finite.copy()
    else:
        signs=np.sign(a)
        nonzero=np.all(np.abs(a)>1e-12,axis=1)
        agree=finite & nonzero & (np.all(signs>0,axis=1)|np.all(signs<0,axis=1))
    return pred,agree


def _tail_eval(target,pred,v13_pred,season,prior,agree,threshold):
    abs_p=np.abs(np.asarray(pred,float))
    m=np.asarray(prior,bool)&np.asarray(agree,bool)&np.isfinite(abs_p)&(abs_p>=float(threshold))
    met=_metrics(target,pred,m); v13=_metrics(target,v13_pred,m); market=_market_metrics(target,m)
    years=sorted(int(x) for x in pd.Series(np.asarray(season)[m]).dropna().unique())
    per=[]
    for sy in years:
        sm=m&(np.asarray(season,float)==float(sy))
        mm=_metrics(target,pred,sm)
        if mm["n"]>=SCV2_MIN_TAIL_SEASON_N:
            per.append((sy,mm))
    pos_signed=sum(np.isfinite(x[1]["signed"]) and x[1]["signed"]>0 for x in per)
    profitable=sum(np.isfinite(x[1]["hit"]) and x[1]["hit"]>SCV2_BREAK_EVEN for x in per)
    min_hit=min((x[1]["hit"] for x in per if np.isfinite(x[1]["hit"])),default=np.nan)
    min_signed=min((x[1]["signed"] for x in per if np.isfinite(x[1]["signed"])),default=np.nan)
    # Remove the season with the best hit rate, then recompute.
    rem={"n":0,"hit":np.nan,"signed":np.nan}
    if per:
        best=max(per,key=lambda z:z[1]["hit"] if np.isfinite(z[1]["hit"]) else -999)[0]
        rm=m&(np.asarray(season,float)!=float(best))
        rem=_metrics(target,pred,rm)
    nse=len(per)
    required_pos = nse if nse<=2 else max(2,nse-1)
    stability=(nse>=2 and pos_signed>=required_pos and np.isfinite(rem.get("signed",np.nan)) and rem["signed"]>0.50 and
               np.isfinite(min_hit) and min_hit>=0.49)
    econ=(met["n"]>=SCV2_MIN_TAIL_PRIOR_N and np.isfinite(met["hit"]) and met["hit"]>=0.535 and
          np.isfinite(met["signed"]) and met["signed"]>=1.0 and np.isfinite(met["roi"]) and met["roi"]>0)
    # We do not require global RMSE improvement for a selective betting tail, but
    # block catastrophically mis-scaled tails.
    scale_ok=(market["n"]==met["n"] and (not np.isfinite(met["rmse"]) or not np.isfinite(market["rmse"]) or met["rmse"]<=market["rmse"]+0.75))
    passed=bool(stability and econ and scale_ok)
    return {"mask":m,"met":met,"v13":v13,"market":market,"per":per,"pos_signed":pos_signed,"profitable":profitable,
            "min_hit":min_hit,"min_signed":min_signed,"remove_best":rem,"passed":passed,"stability":stability,"econ":econ,"scale_ok":scale_ok}


def _candidate_rows(family_preds,target,v13_pred,season,target_season,log_func=print):
    season=np.asarray(season,float)
    prior=np.isfinite(season)&(season>=SCV2_DISCOVERY_START)&(season<float(target_season))&np.isfinite(target)
    scaled,scales=_scaled_family_preds(family_preds,target,prior,log_func,target_season)
    rows=[]
    for combo in SCV2_COMBOS:
        if any(f not in scaled or scales[f]["beta"]<=0.0 for f in combo):
            continue
        pred,agree=_combo_prediction(scaled,combo)
        if pred is None: continue
        for th in SCV2_THRESHOLDS:
            z=_tail_eval(target,pred,v13_pred,season,prior,agree,th)
            if z["met"]["n"]<SCV2_MIN_TAIL_PRIOR_N: continue
            rows.append({"combo":combo,"threshold":float(th),"pred":pred,"agree":agree,"prior":z,"scales":scales})
    rows.sort(key=lambda r:(
        0 if r["prior"]["passed"] else 1,
        -float(r["prior"]["remove_best"].get("hit",np.nan) if np.isfinite(r["prior"]["remove_best"].get("hit",np.nan)) else -999),
        -float(r["prior"]["met"]["signed"] if np.isfinite(r["prior"]["met"]["signed"]) else -999),
        -float(r["prior"]["met"]["hit"] if np.isfinite(r["prior"]["met"]["hit"]) else -999),
        len(r["combo"]), r["threshold"], r["combo"],
    ))
    return rows


def _side_for_prediction(g:pd.DataFrame,pred:np.ndarray)->dict:
    p=np.asarray(pred,float); home_side=p>=0
    home=g.get("Team_Norm",pd.Series("",index=g.index)).astype(str).to_numpy(object)
    away=g.get("Opponent_Norm",pd.Series("",index=g.index)).astype(str).to_numpy(object)
    spread=_num(g.get("Consensus_Open_Spread"),g.index).to_numpy(float)
    return {"stat_home":home_side,"selected_team":np.where(home_side,home,away),"selected_opp":np.where(home_side,away,home),
            "selected_spread":np.where(home_side,spread,-spread),"edge_points":p}


def _system_overlay(g,target,pred,base_mask,dashboard_module,label,log_func=print):
    try:
        import v14_stat_reliability as rel
        sys_hist=getattr(dashboard_module,"_V143_SYSTEM_HISTORY_CACHE",{})
        masks=rel._system_occurrence_masks(g,_side_for_prediction(g,pred),sys_hist,log_func=lambda *_:None)
        for n in ("BIGAL_AGREE","BIGAL_CONFLICT","SYSTEM_ANY_AGREE","SYSTEM_ANY_CONFLICT"):
            if n not in masks: continue
            m=np.asarray(base_mask,bool)&np.asarray(masks[n],bool)
            met=_metrics(target,pred,m)
            if met["n"]>=10:
                log_func(f"[STAT-COMBO-V2-SYSTEM] sample={label} state={n} n={met['n']} hit={met['hit']:.4f} roi={met['roi']:+.4f} signed_market_error={met['signed']:+.3f}")
    except Exception as e:
        log_func(f"[STAT-COMBO-V2-SYSTEM] sample={label} status=UNAVAILABLE error={type(e).__name__}:{e}")


def run_stat_combination_v2(*,dashboard_module,log_func=print,hard_fail=True):
    try:
        cache=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{})
        g=cache.get("games") if isinstance(cache,dict) else None
        oof_margin=cache.get("oof_margin") if isinstance(cache,dict) else None
        candidate_cols=cache.get("candidate_feature_cols") if isinstance(cache,dict) else None
        if g is None or not isinstance(g,pd.DataFrame) or g.empty or oof_margin is None:
            raise RuntimeError("STAT_COMBINATION_V2 requires V13 season-forward research cache")
        g=g.copy(); g["Season"]=_num(g.get("Season"),g.index)
        season=g["Season"].to_numpy(float)
        target=_num(g.get("Market_Error_Margin"),g.index).to_numpy(float)
        market_margin=_num(g.get("Market_Open_Margin"),g.index).to_numpy(float)
        if candidate_cols is None:
            prefixes=("A_Raw","B_Raw","Diff_Raw","A_State_","B_State_","Diff_State_","A_Recent3_","B_Recent3_","Diff_Recent3_","Matchup_","Context_")
            candidate_cols=[c for c in g.columns if str(c).startswith(prefixes)]
        candidate_cols=[c for c in list(dict.fromkeys(candidate_cols)) if c in g.columns]
        family_fn=getattr(dashboard_module,"_ncaaf_stat_feature_family",None)
        if family_fn is None: raise RuntimeError("missing _ncaaf_stat_feature_family")
        fam_cols=_family_columns(g,candidate_cols,family_fn)
        if len(fam_cols)<5: raise RuntimeError(f"too few statistical families={list(fam_cols)}")
        key=_physical_key(g); valid=key.ne("")&key.ne("nan"); dup=int(key[valid].duplicated().sum())
        if dup: raise RuntimeError(f"physical game duplication rows={dup}")
        v13_pred=np.asarray(oof_margin,float)-market_margin
        log_func(f"[STAT-COMBO-V2-PREFLIGHT] status=PASS source_tag={SCV2_SOURCE_TAG} rows={len(g)} candidate_features={len(candidate_cols)} families={len(fam_cols)} predeclared_combos={len(SCV2_COMBOS)} thresholds={list(SCV2_THRESHOLDS)} dynamic_strength_v1=RETIRED physical_game_duplicates=0 target=MARKET_ERROR_MARGIN production_authority=0")
        family_preds=_family_oof(g,fam_cols,target,season,log_func)

        primary=None
        for target_season in (2025,2026):
            rows=_candidate_rows(family_preds,target,v13_pred,season,target_season,log_func)
            if not rows:
                log_func(f"[STAT-COMBO-V2-SELECTION] target_season={target_season} status=NO_CANDIDATES production_authority=0")
                continue
            leader=rows[0]; p=leader["prior"]
            status="TAIL_PASS" if p["passed"] else "NO_TAIL_PASS"
            test_base=(season==float(target_season))&leader["agree"]&np.isfinite(leader["pred"])&(np.abs(leader["pred"])>=leader["threshold"])
            test=_metrics(target,leader["pred"],test_base); v13t=_metrics(target,v13_pred,test_base)
            markett=_market_metrics(target,test_base)
            log_func(
                f"[STAT-COMBO-V2-SELECTION] target_season={target_season} combo={'+'.join(leader['combo'])} threshold={leader['threshold']:.2f} status={status} "
                f"prior_n={p['met']['n']} prior_hit={p['met']['hit']:.4f} prior_roi={p['met']['roi']:+.4f} prior_signed_market_error={p['met']['signed']:+.3f} "
                f"prior_remove_best_hit={p['remove_best']['hit']:.4f} prior_remove_best_signed={p['remove_best']['signed']:+.3f} positive_signed_seasons={p['pos_signed']} "
                f"target_n={test['n']} target_hit={test['hit']:.4f} target_roi={test['roi']:+.4f} target_signed_market_error={test['signed']:+.3f} "
                f"target_v13_hit_same_rows={v13t['hit']:.4f} target_market_rmse={markett['rmse']:.4f} target_combo_rmse={test['rmse']:.4f} chronological=TRUE production_authority=0"
            )
            for rank,r in enumerate(rows[:8],1):
                z=r["prior"]
                log_func(f"[STAT-COMBO-V2-TOP] target_season={target_season} rank={rank} combo={'+'.join(r['combo'])} threshold={r['threshold']:.2f} pass={z['passed']} n={z['met']['n']} hit={z['met']['hit']:.4f} roi={z['met']['roi']:+.4f} signed={z['met']['signed']:+.3f} remove_best_hit={z['remove_best']['hit']:.4f} min_hit={z['min_hit']:.4f}")
            # Explicit per-season evidence for the chosen tail.
            for sy,mm in p["per"]:
                log_func(f"[STAT-COMBO-V2-SEASON] target_season={target_season} combo={'+'.join(leader['combo'])} threshold={leader['threshold']:.2f} evidence_season={sy} n={mm['n']} hit={mm['hit']:.4f} roi={mm['roi']:+.4f} signed={mm['signed']:+.3f}")
            _system_overlay(g,target,leader["pred"],p["mask"],dashboard_module,f"PRIOR_TO_{target_season}",log_func)
            _system_overlay(g,target,leader["pred"],test_base,dashboard_module,f"TARGET_{target_season}",log_func)
            if target_season==2026:
                primary={"leader":leader,"prior":p,"test":test,"v13":v13t,"market":markett,"status":status,"test_mask":test_base}

        if primary is None: raise RuntimeError("no 2026 V2 tail challenger produced")
        n=int(primary["test_mask"].sum()); u=int(_physical_key(g).loc[primary["test_mask"]].nunique())
        if n!=u: raise RuntimeError(f"2026 V2 tail comparison not one physical game per row n={n} unique={u}")
        log_func(f"[STAT-COMBO-V2-COMPARISON-CONTRACT] status=PASS target_season=2026 tail_rows={n} unique_physical_games={u} prior_only_selection=TRUE same_rows_for_v13=TRUE production_authority=0")
        L=primary["leader"]
        log_func(f"[STAT-COMBO-V2-CONTRACT] status=PASS source_tag={SCV2_SOURCE_TAG} primary_combo={'+'.join(L['combo'])} threshold={L['threshold']:.2f} primary_status={primary['status']} prior_n={primary['prior']['met']['n']} prior_hit={primary['prior']['met']['hit']:.4f} target_2026_n={primary['test']['n']} target_2026_hit={primary['test']['hit']:.4f} target_2026_signed_market_error={primary['test']['signed']:+.3f} dynamic_strength_v1=RETIRED stat_combo_v1=RETIRED production_authority=0")
        return {"status":"PASS","source_tag":SCV2_SOURCE_TAG,"primary":primary,"production_authority":0}
    except Exception as e:
        log_func(f"[STAT-COMBO-V2-CONTRACT] status=FAILED error={type(e).__name__}:{e} production_authority=0")
        if hard_fail: raise
        return {"status":"FAILED","error":f"{type(e).__name__}:{e}","production_authority":0}
