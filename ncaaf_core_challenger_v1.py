"""NCAAF CORE Challenger V1 — compact, protected fair-line research.

Purpose
-------
Test whether the frozen NCAAF Spread CORE recipe can be improved without
mutating Production V1.  This lane is deliberately small and leakage-safe:

* 2022 = initial fit season.
* 2023 = discovery/recipe selection only.
* 2024 and 2025 = untouched confirmation seasons.
* 2026+ = completely sealed; never queried for selection, tuning, confirmation,
  ranking, promotion, or threshold choice.
* Production V1 remains the champion unless a later explicit promotion job is
  created and approved.
* No generic AutoFS, multi-head runtime, system signals, Pathi, Big Al, Miner,
  current-market movement, or 2026 outcomes may enter this research lane.

The search compares the incumbent compact market-residual recipe with compact
market-residual and market-blind fair-margin challengers selected using 2023
only.  A hybrid fair-line challenger is also tested.  Confirmation reports
point-error quality and ATS-direction diagnostics separately; no betting policy
is promoted here.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import pickle
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

NCAAF_CORE_CHALLENGER_V1_SOURCE_TAG = "ncaaf-core-challenger-v1.0-compact-fairline-20261005"
NCAAF_CORE_CHALLENGER_V1_VERSION = "1.0.0"
REPORT_CURRENT_BLOB = "research/ncaaf/core_challenger/v1/current_report.json"
REPORT_HISTORY_PREFIX = "research/ncaaf/core_challenger/v1/history"
BUNDLE_CURRENT_BLOB = "research/ncaaf/core_challenger/v1/current_bundle.pkl"

DISCOVERY_TRAIN_SEASONS = (2022,)
DISCOVERY_VALIDATION_SEASON = 2023
CONFIRMATION_SEASONS = (2024, 2025)
PROSPECTIVE_MIN_SEASON = 2026

INCUMBENT_FEATURES = (
    "Context_Intercept",
    "Diff_RawRecent3_Off_YPP",
    "B_RawSeason_GameAdj_Def_Rush_YPA",
)

# Small preregistered model grid.  Discovery chooses; confirmation only validates.
RIDGE_ALPHAS = (12.0, 24.0, 48.0)
BLEND_WEIGHTS = (0.50, 0.75, 1.00)  # 1.0 = Ridge only
HYBRID_FAIR_WEIGHTS = (0.25, 0.50, 0.75)  # weight on market-blind fair margin
MAX_FEATURES = 6
SCREEN_TOP_K = 30
FAMILY_CAP = 2
MIN_INCREMENTAL_RMSE_GAIN = 0.02
MAX_INCREMENTAL_MAE_GIVEBACK = 0.03
MIN_DISCOVERY_TRAIN_ROWS = 500
MIN_DISCOVERY_VALID_ROWS = 400
MIN_CONFIRM_ROWS = 400
BOOTSTRAP_REPS = 500
EDGE_BANDS = (1.0, 2.0, 3.0, 4.0, 5.0)

_BLOCKED_FEATURE_TOKENS = (
    "actual_", "team_score", "opponent_score", "postgame", "final_", "result",
    "market_error", "zero_market", "cover_result", "ats_result", "target",
    "q1_points", "q2_points", "q3_points",  # current-game scoring is never pregame
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _rmse(y, p) -> float:
    a=np.asarray(y,dtype=float); b=np.asarray(p,dtype=float); ok=np.isfinite(a)&np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok]-b[ok])**2))) if ok.any() else float("nan")


def _mae(y, p) -> float:
    a=np.asarray(y,dtype=float); b=np.asarray(p,dtype=float); ok=np.isfinite(a)&np.isfinite(b)
    return float(np.mean(np.abs(a[ok]-b[ok]))) if ok.any() else float("nan")


def _safe(v, default=float("nan")):
    try:
        z=float(v)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _json_safe(x):
    if isinstance(x, dict): return {str(k):_json_safe(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray): return [_json_safe(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, pd.Timestamp): return x.isoformat()
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def _new_models(alpha: float):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    lin=Pipeline([
        ("imputer",SimpleImputer(strategy="median",add_indicator=True)),
        ("scale",StandardScaler()),
        ("ridge",Ridge(alpha=float(alpha))),
    ])
    hgb=Pipeline([
        ("imputer",SimpleImputer(strategy="median",add_indicator=False)),
        ("hgb",HistGradientBoostingRegressor(
            loss="squared_error",learning_rate=0.035,max_iter=120,
            max_leaf_nodes=12,max_depth=3,min_samples_leaf=30,
            l2_regularization=4.0,random_state=2601,
        )),
    ])
    return lin,hgb


def _fit_predict(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], target: str,
                 *, alpha: float, blend_weight: float) -> np.ndarray:
    feats=list(features)
    y=_num(train,target)
    good=y.notna().to_numpy()
    if int(good.sum())<100:
        return np.full(len(valid),np.nan)
    lin,hgb=_new_models(alpha)
    X=train.loc[:,feats]
    lin.fit(X.loc[good],y.loc[good].to_numpy(dtype=float))
    p1=np.asarray(lin.predict(valid.loc[:,feats]),dtype=float)
    w=float(blend_weight)
    if w>=0.999:
        return p1
    hgb.fit(X.loc[good],y.loc[good].to_numpy(dtype=float))
    p2=np.asarray(hgb.predict(valid.loc[:,feats]),dtype=float)
    return w*p1+(1.0-w)*p2


def _family(feature: str, dashboard_module=None) -> str:
    fn=getattr(dashboard_module,"_ncaaf_stat_feature_family",None)
    if callable(fn):
        try: return str(fn(feature))
        except Exception: pass
    s=str(feature)
    for token,label in (
        ("Rush","rushing"),("YPP","efficiency"),("Pass","passing"),("Turnover","turnover"),
        ("FirstDown","conversion"),("Play","pace"),("GameAdj","opponent_adjusted"),("Context_","context"),
    ):
        if token.lower() in s.lower(): return label
    return "other"


def _feature_pool(g: pd.DataFrame, candidate_cols: list[str]) -> list[str]:
    out=[]
    disc=g[_num(g,"Season").isin([2022,2023])]
    for c in list(dict.fromkeys(candidate_cols)):
        if c=="Context_Intercept" or c not in g.columns: continue
        lc=str(c).lower()
        if any(t in lc for t in _BLOCKED_FEATURE_TOKENS): continue
        s=pd.to_numeric(disc[c],errors="coerce")
        if int(s.notna().sum()) < 500: continue
        if s.nunique(dropna=True) < 2: continue
        out.append(c)
    return out


def _target_series(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode=="RESIDUAL":
        return _num(df,"Actual_Margin")-_num(df,"Market_Open_Margin")
    return _num(df,"Actual_Margin")


def _select_features(g: pd.DataFrame, pool: list[str], mode: str, dashboard_module=None, log_func=print) -> dict[str,Any]:
    tr=g[_num(g,"Season").eq(2022)].copy(); va=g[_num(g,"Season").eq(2023)].copy()
    tr["__TARGET"]=_target_series(tr,mode); va["__TARGET"]=_target_series(va,mode)
    if len(tr)<MIN_DISCOVERY_TRAIN_ROWS or len(va)<MIN_DISCOVERY_VALID_ROWS:
        raise RuntimeError(f"discovery rows insufficient mode={mode} train={len(tr)} valid={len(va)}")
    baseline=["Context_Intercept"]
    base=_fit_predict(tr,va,baseline,"__TARGET",alpha=24.0,blend_weight=1.0)
    by=_num(va,"__TARGET").to_numpy(dtype=float)
    base_rmse=_rmse(by,base); base_mae=_mae(by,base)
    screened=[]
    for c in pool:
        try:
            p=_fit_predict(tr,va,baseline+[c],"__TARGET",alpha=24.0,blend_weight=1.0)
            rr=_rmse(by,p); mm=_mae(by,p)
            if not np.isfinite(rr): continue
            screened.append({"feature":c,"rmse":rr,"mae":mm,"rmse_gain":base_rmse-rr,"mae_gain":base_mae-mm,"family":_family(c,dashboard_module)})
        except Exception:
            continue
    screened.sort(key=lambda x:(x["rmse_gain"],x["mae_gain"]),reverse=True)
    shortlist=[x for x in screened if x["rmse_gain"]>0][:SCREEN_TOP_K]
    accepted=list(baseline); family_counts={}; current_rmse=base_rmse; current_mae=base_mae; steps=[]
    while len(accepted)-1<MAX_FEATURES:
        best=None
        for rec in shortlist:
            c=rec["feature"]
            if c in accepted: continue
            fam=rec["family"]
            if family_counts.get(fam,0)>=FAMILY_CAP: continue
            try:
                p=_fit_predict(tr,va,accepted+[c],"__TARGET",alpha=24.0,blend_weight=1.0)
                rr=_rmse(by,p); mm=_mae(by,p)
            except Exception:
                continue
            gain=current_rmse-rr; mae_gain=current_mae-mm
            if gain<MIN_INCREMENTAL_RMSE_GAIN or mae_gain < -MAX_INCREMENTAL_MAE_GIVEBACK: continue
            cand={"feature":c,"family":fam,"rmse":rr,"mae":mm,"rmse_gain":gain,"mae_gain":mae_gain}
            if best is None or (cand["rmse"],cand["mae"]) < (best["rmse"],best["mae"]): best=cand
        if best is None: break
        accepted.append(best["feature"]); family_counts[best["family"]]=family_counts.get(best["family"],0)+1
        current_rmse=best["rmse"]; current_mae=best["mae"]; steps.append(best)
    features=accepted
    log_func(f"[NCAAF-CORE-CHALLENGER-FEATURES] mode={mode} pool={len(pool)} screened={len(screened)} selected={len(features)-1} features={features} discovery_rmse={current_rmse:.4f} discovery_mae={current_mae:.4f}")
    return {"mode":mode,"features":features,"screened_top":shortlist[:15],"steps":steps,"baseline_rmse":base_rmse,"baseline_mae":base_mae,"final_rmse":current_rmse,"final_mae":current_mae}


def _recipe_predict(train: pd.DataFrame, valid: pd.DataFrame, recipe: dict[str,Any]) -> np.ndarray:
    kind=recipe["kind"]
    if kind in {"RESIDUAL","INCUMBENT"}:
        tt=train.copy(); vv=valid.copy(); tt["__TARGET"]=_target_series(tt,"RESIDUAL")
        edge=_fit_predict(tt,vv,recipe["features"],"__TARGET",alpha=recipe["alpha"],blend_weight=recipe["blend"])
        return _num(vv,"Market_Open_Margin").to_numpy(dtype=float)+edge
    if kind=="FAIR_MARGIN":
        tt=train.copy(); vv=valid.copy(); tt["__TARGET"]=_target_series(tt,"FAIR_MARGIN")
        return _fit_predict(tt,vv,recipe["features"],"__TARGET",alpha=recipe["alpha"],blend_weight=recipe["blend"])
    if kind=="HYBRID":
        p_fair=_recipe_predict(train,valid,recipe["fair_recipe"])
        p_res=_recipe_predict(train,valid,recipe["resid_recipe"])
        w=float(recipe["fair_weight"])
        return w*p_fair+(1.0-w)*p_res
    raise ValueError(f"unknown recipe kind={kind}")


def _point_metrics(df: pd.DataFrame, pred: np.ndarray) -> dict[str,Any]:
    actual=_num(df,"Actual_Margin").to_numpy(dtype=float); market=_num(df,"Market_Open_Margin").to_numpy(dtype=float)
    ok=np.isfinite(actual)&np.isfinite(pred)&np.isfinite(market)
    if not ok.any(): return {"n":0}
    a=actual[ok]; p=np.asarray(pred,dtype=float)[ok]; m=market[ok]
    imp=np.abs(a-m)-np.abs(a-p)
    return {
        "n":int(ok.sum()),"rmse":_rmse(a,p),"mae":_mae(a,p),
        "market_rmse":_rmse(a,m),"market_mae":_mae(a,m),
        "rmse_improvement_vs_market":_rmse(a,m)-_rmse(a,p),
        "mae_improvement_vs_market":_mae(a,m)-_mae(a,p),
        "closer_to_actual_than_market_rate":float(np.mean(imp>0)),
        "mean_abs_error_improvement_vs_market":float(np.mean(imp)),
    }


def _ats_metrics(df: pd.DataFrame, pred: np.ndarray) -> dict[str,Any]:
    actual=_num(df,"Actual_Margin").to_numpy(dtype=float); market=_num(df,"Market_Open_Margin").to_numpy(dtype=float); p=np.asarray(pred,dtype=float)
    edge=p-market; ats=actual-market
    out={}
    for band in EDGE_BANDS:
        sel=np.isfinite(edge)&np.isfinite(ats)&(np.abs(edge)>=float(band))&(~np.isclose(ats,0.0,atol=1e-9))
        n=int(sel.sum())
        if not n:
            out[str(band)]={"n":0,"hit_rate":None,"roi_at_minus110":None}
            continue
        win=np.where(edge[sel]>=0,ats[sel]>0,ats[sel]<0)
        hit=float(np.mean(win)); roi=hit*(100.0/110.0)-(1.0-hit)
        out[str(band)]={"n":n,"hit_rate":hit,"roi_at_minus110":float(roi)}
    return out


def _bootstrap_vs_incumbent(df: pd.DataFrame, incumbent: np.ndarray, challenger: np.ndarray, reps=BOOTSTRAP_REPS) -> dict[str,Any]:
    a=_num(df,"Actual_Margin").to_numpy(dtype=float); i=np.asarray(incumbent,dtype=float); c=np.asarray(challenger,dtype=float)
    ok=np.isfinite(a)&np.isfinite(i)&np.isfinite(c)
    a=a[ok]; i=i[ok]; c=c[ok]
    if len(a)<100: return {"n":int(len(a)),"rmse_gain":None,"mae_gain":None,"rmse_gain_ci95":[None,None],"mae_gain_ci95":[None,None]}
    rg=_rmse(a,i)-_rmse(a,c); mg=_mae(a,i)-_mae(a,c)
    rng=np.random.default_rng(2605); rs=[]; ms=[]; n=len(a)
    for _ in range(int(reps)):
        idx=rng.integers(0,n,n)
        aa=a[idx]; ii=i[idx]; cc=c[idx]
        rs.append(_rmse(aa,ii)-_rmse(aa,cc)); ms.append(_mae(aa,ii)-_mae(aa,cc))
    return {"n":n,"rmse_gain":rg,"mae_gain":mg,"rmse_gain_ci95":[float(np.quantile(rs,.025)),float(np.quantile(rs,.975))],"mae_gain_ci95":[float(np.quantile(ms,.025)),float(np.quantile(ms,.975))]}


def _tune_recipe(g: pd.DataFrame, features: list[str], kind: str, log_func=print) -> dict[str,Any]:
    tr=g[_num(g,"Season").eq(2022)].copy(); va=g[_num(g,"Season").eq(2023)].copy()
    actual=_num(va,"Actual_Margin").to_numpy(dtype=float)
    rows=[]
    for alpha in RIDGE_ALPHAS:
        for blend in BLEND_WEIGHTS:
            rec={"kind":kind,"features":list(features),"alpha":float(alpha),"blend":float(blend)}
            p=_recipe_predict(tr,va,rec); met=_point_metrics(va,p)
            rows.append({**rec,"metrics":met})
    rows.sort(key=lambda r:(_safe((r["metrics"] or {}).get("rmse"),1e9),_safe((r["metrics"] or {}).get("mae"),1e9)))
    best=rows[0]
    log_func(f"[NCAAF-CORE-CHALLENGER-RECIPE] kind={kind} alpha={best['alpha']} blend={best['blend']} features={len(features)} discovery_rmse={best['metrics'].get('rmse'):.4f} discovery_mae={best['metrics'].get('mae'):.4f}")
    return {"recipe":{k:v for k,v in best.items() if k!="metrics"},"discovery_metrics":best["metrics"],"grid":rows}


def _evaluate_recipe(g: pd.DataFrame, recipe: dict[str,Any], name: str, log_func=print) -> dict[str,Any]:
    seasons={}; preds={}
    for yr in CONFIRMATION_SEASONS:
        tr=g[_num(g,"Season").lt(yr)].copy(); va=g[_num(g,"Season").eq(yr)].copy()
        if len(va)<MIN_CONFIRM_ROWS: raise RuntimeError(f"confirmation rows insufficient year={yr} n={len(va)}")
        p=_recipe_predict(tr,va,recipe)
        seasons[str(yr)]={"point":_point_metrics(va,p),"ats":_ats_metrics(va,p)}
        preds[yr]=(va,p)
        pm=seasons[str(yr)]["point"]
        log_func(f"[NCAAF-CORE-CHALLENGER-CONFIRM] candidate={name} season={yr} n={pm.get('n')} rmse={pm.get('rmse'):.4f} mae={pm.get('mae'):.4f} vs_market_rmse={pm.get('rmse_improvement_vs_market'):+.4f} closer={pm.get('closer_to_actual_than_market_rate'):.3f}")
    pooled_df=pd.concat([preds[y][0] for y in CONFIRMATION_SEASONS],ignore_index=True)
    pooled_pred=np.concatenate([preds[y][1] for y in CONFIRMATION_SEASONS])
    return {"name":name,"recipe":recipe,"confirmation":seasons,"pooled":{"point":_point_metrics(pooled_df,pooled_pred),"ats":_ats_metrics(pooled_df,pooled_pred)},"_pooled_df":pooled_df,"_pooled_pred":pooled_pred,"_season_preds":preds}


def run_ncaaf_core_challenger_v1(*, dashboard_module, bucket_name="sharp-models", storage_client=None, log_func=print, hard_fail=True) -> dict[str,Any]:
    try:
        cache=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{}) or {}
        games=cache.get("games"); candidate_cols=list(cache.get("candidate_feature_cols") or [])
        if games is None or getattr(games,"empty",True): raise RuntimeError("historical research games cache missing")
        g=games.copy(); season=_num(g,"Season")
        # Hard seal 2026+: never pass those rows into any selection/evaluation function.
        g=g.loc[season<=max(CONFIRMATION_SEASONS)].reset_index(drop=True)
        seasons=sorted(set(int(x) for x in _num(g,"Season").dropna().unique()))
        if not set((2022,2023,2024,2025)).issubset(seasons): raise RuntimeError(f"required seasons missing seasons={seasons}")
        missing=[c for c in INCUMBENT_FEATURES if c not in g.columns]
        if missing: raise RuntimeError(f"incumbent features missing {missing}")
        if int((_num(g,"Season")>=2026).sum())!=0: raise RuntimeError("2026 seal failed")
        log_func(f"[NCAAF-CORE-CHALLENGER-PREFLIGHT] status=PASS source={NCAAF_CORE_CHALLENGER_V1_SOURCE_TAG} rows={len(g)} seasons={seasons} discovery_train=2022 discovery_validate=2023 confirmation=2024,2025 year_2026_queried=FALSE production_authority=0")

        pool=_feature_pool(g,candidate_cols)
        fair_sel=_select_features(g,pool,"FAIR_MARGIN",dashboard_module=dashboard_module,log_func=log_func)
        resid_sel=_select_features(g,pool,"RESIDUAL",dashboard_module=dashboard_module,log_func=log_func)

        inc_recipe={"kind":"INCUMBENT","features":list(INCUMBENT_FEATURES),"alpha":24.0,"blend":0.75}
        inc=_evaluate_recipe(g,inc_recipe,"INCUMBENT_RECIPE",log_func=log_func)

        fair_tuned=_tune_recipe(g,fair_sel["features"],"FAIR_MARGIN",log_func=log_func)
        resid_tuned=_tune_recipe(g,resid_sel["features"],"RESIDUAL",log_func=log_func)
        fair=_evaluate_recipe(g,fair_tuned["recipe"],"FAIR_MARGIN_COMPACT",log_func=log_func)
        resid=_evaluate_recipe(g,resid_tuned["recipe"],"RESIDUAL_COMPACT",log_func=log_func)

        # Hybrid weight is selected on 2023 only, then frozen for both confirmation years.
        tr22=g[_num(g,"Season").eq(2022)].copy(); va23=g[_num(g,"Season").eq(2023)].copy()
        pf=_recipe_predict(tr22,va23,fair_tuned["recipe"]); pr=_recipe_predict(tr22,va23,resid_tuned["recipe"])
        hybrids=[]
        for w in HYBRID_FAIR_WEIGHTS:
            p=float(w)*pf+(1.0-float(w))*pr
            hybrids.append({"fair_weight":float(w),"metrics":_point_metrics(va23,p)})
        hybrids.sort(key=lambda x:(_safe(x["metrics"].get("rmse"),1e9),_safe(x["metrics"].get("mae"),1e9)))
        hw=hybrids[0]["fair_weight"]
        hybrid_recipe={"kind":"HYBRID","fair_weight":hw,"fair_recipe":fair_tuned["recipe"],"resid_recipe":resid_tuned["recipe"]}
        log_func(f"[NCAAF-CORE-CHALLENGER-RECIPE] kind=HYBRID fair_weight={hw:.2f} discovery_rmse={hybrids[0]['metrics'].get('rmse'):.4f} discovery_mae={hybrids[0]['metrics'].get('mae'):.4f}")
        hybrid=_evaluate_recipe(g,hybrid_recipe,"HYBRID_COMPACT",log_func=log_func)

        candidates=[fair,resid,hybrid]
        # Incumbent comparison uses the exact same 2024/2025 rows.
        inc_pool_df=inc["_pooled_df"]; inc_pool_pred=inc["_pooled_pred"]
        for c in candidates:
            c["vs_incumbent_pooled"]=_bootstrap_vs_incumbent(c["_pooled_df"],inc_pool_pred,c["_pooled_pred"])
            year_gains={}
            for yr in CONFIRMATION_SEASONS:
                incm=inc["confirmation"][str(yr)]["point"]; cm=c["confirmation"][str(yr)]["point"]
                year_gains[str(yr)]={"rmse_gain":_safe(incm.get("rmse"))-_safe(cm.get("rmse")),"mae_gain":_safe(incm.get("mae"))-_safe(cm.get("mae"))}
            c["vs_incumbent_by_season"]=year_gains
            b=c["vs_incumbent_pooled"]
            both_rmse=all(_safe(year_gains[str(y)].get("rmse_gain"),-9)>0 for y in CONFIRMATION_SEASONS)
            pooled_rmse=_safe(b.get("rmse_gain"),-9)>0; pooled_mae=_safe(b.get("mae_gain"),-9)>=0
            ci=(b.get("rmse_gain_ci95") or [None,None]); ci_low=_safe(ci[0],-9)
            c["state"]=("STRONG_CHALLENGER" if both_rmse and pooled_rmse and pooled_mae and ci_low>0 else
                        "PROMOTION_ELIGIBLE_RESEARCH" if both_rmse and pooled_rmse and pooled_mae else
                        "MIXED" if pooled_rmse else "NO_IMPROVEMENT")
            log_func(f"[NCAAF-CORE-CHALLENGER-SCORECARD] candidate={c['name']} state={c['state']} pooled_rmse_gain={_safe(b.get('rmse_gain')):+.4f} pooled_mae_gain={_safe(b.get('mae_gain')):+.4f} rmse_ci95={b.get('rmse_gain_ci95')} year_gains={year_gains} production_authority=0")

        ranked=sorted(candidates,key=lambda c:(_safe((c.get("vs_incumbent_pooled") or {}).get("rmse_gain"),-9),_safe((c.get("vs_incumbent_pooled") or {}).get("mae_gain"),-9)),reverse=True)
        best=ranked[0] if ranked else None
        recommendation="KEEP_INCUMBENT"
        if best and best.get("state") in {"STRONG_CHALLENGER","PROMOTION_ELIGIBLE_RESEARCH"}: recommendation="CHALLENGER_DESERVES_PROSPECTIVE_SHADOW"

        def clean_eval(e):
            return {k:v for k,v in e.items() if not str(k).startswith("_")}
        report={
            "source_tag":NCAAF_CORE_CHALLENGER_V1_SOURCE_TAG,"version":NCAAF_CORE_CHALLENGER_V1_VERSION,"created_utc":_now(),
            "status":"NCAAF_CORE_CHALLENGER_V1_COMPLETE","production_authority":0,"production_mutated":False,"automatic_promotion":False,
            "year_2026_queried":False,"discovery":{"train":[2022],"validation":2023},"confirmation_seasons":[2024,2025],"prospective_min_season":2026,
            "incumbent":{"features":list(INCUMBENT_FEATURES),"recipe":inc_recipe,"evaluation":clean_eval(inc)},
            "feature_search":{"pool_size":len(pool),"fair_margin":fair_sel,"market_residual":resid_sel},
            "discovery_recipe_search":{"fair_margin":fair_tuned,"market_residual":resid_tuned,"hybrid_weights":hybrids},
            "challengers":[clean_eval(c) for c in ranked],
            "best_challenger":({"name":best.get("name"),"state":best.get("state"),"recipe":best.get("recipe"),"vs_incumbent_pooled":best.get("vs_incumbent_pooled"),"vs_incumbent_by_season":best.get("vs_incumbent_by_season")} if best else None),
            "recommendation":recommendation,
            "promotion_contract":"NO_AUTOMATIC_PROMOTION__2024_AND_2025_BOTH_MUST_BEAT_INCUMBENT_RMSE__POOLED_MAE_NONINFERIOR__PROSPECTIVE_SHADOW_REQUIRED_BEFORE_ANY_PRODUCTION_CHANGE",
            "next_step":"If a challenger is promotion-eligible, freeze that exact recipe and run prospective 2026 shadow beside Production V1; otherwise keep Production V1 unchanged.",
        }
        if storage_client is None:
            from google.cloud import storage
            storage_client=storage.Client()
        body=json.dumps(_json_safe(report),sort_keys=True,separators=(",",":"),default=str).encode()
        sha=hashlib.sha256(body).hexdigest(); hist=f"{REPORT_HISTORY_PREFIX}/{sha[:16]}/report.json"; b=storage_client.bucket(bucket_name)
        b.blob(hist).upload_from_string(body,content_type="application/json"); b.blob(REPORT_CURRENT_BLOB).upload_from_string(body,content_type="application/json")
        bundle={"report":report,"source_tag":NCAAF_CORE_CHALLENGER_V1_SOURCE_TAG}
        bio=io.BytesIO(); pickle.dump(bundle,bio,protocol=pickle.HIGHEST_PROTOCOL); b.blob(BUNDLE_CURRENT_BLOB).upload_from_string(bio.getvalue(),content_type="application/octet-stream")
        report["artifact"]={"current_report":f"gs://{bucket_name}/{REPORT_CURRENT_BLOB}","current_bundle":f"gs://{bucket_name}/{BUNDLE_CURRENT_BLOB}","history_report":f"gs://{bucket_name}/{hist}","sha256":sha}
        log_func(f"[NCAAF-CORE-CHALLENGER-CONTRACT] status=PASS best={None if best is None else best.get('name')} best_state={None if best is None else best.get('state')} recommendation={recommendation} report=gs://{bucket_name}/{REPORT_CURRENT_BLOB} sha={sha[:16]} year_2026_queried=FALSE production_authority=0 production_mutated=FALSE")
        return report
    except Exception as exc:
        log_func(f"[NCAAF-CORE-CHALLENGER-FAIL] {type(exc).__name__}: {exc}")
        if hard_fail: raise
        return {"source_tag":NCAAF_CORE_CHALLENGER_V1_SOURCE_TAG,"status":"FAILED","error":f"{type(exc).__name__}:{exc}","production_authority":0}


def load_current_report(bucket_name="sharp-models", storage_client=None):
    try:
        if storage_client is None:
            from google.cloud import storage
            storage_client=storage.Client()
        blob=storage_client.bucket(bucket_name).blob(REPORT_CURRENT_BLOB)
        if not blob.exists(): return None
        obj=json.loads(blob.download_as_text())
        return obj if obj.get("source_tag")==NCAAF_CORE_CHALLENGER_V1_SOURCE_TAG else None
    except Exception:
        return None


def self_test() -> dict[str,Any]:
    # Structural contract test only; the real challenge requires the historical cache.
    ok=(DISCOVERY_TRAIN_SEASONS==(2022,) and DISCOVERY_VALIDATION_SEASON==2023 and CONFIRMATION_SEASONS==(2024,2025) and PROSPECTIVE_MIN_SEASON==2026 and len(INCUMBENT_FEATURES)==3)
    return {"status":"PASS" if ok else "FAIL","source_tag":NCAAF_CORE_CHALLENGER_V1_SOURCE_TAG,"production_authority":0,"automatic_promotion":False,"year_2026_queried":False,"incumbent_features":list(INCUMBENT_FEATURES)}


if __name__=="__main__":
    print(json.dumps(self_test(),indent=2,sort_keys=True))
