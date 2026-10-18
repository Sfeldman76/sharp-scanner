"""Frozen-production-safe NCAAF research challenger.

Learns strictly-as-of-game opponent-adjusted team strength from past scores;
compares recency policies on pre-2024 years, runs 2014-2025 year-forward OOF
Margin/H2H/Totals, and tests a separate, optional *prior-only* modern CFBD
residual specialist selected on 2023 and evaluated on 2024-2025.

NO production mutation / GCS writes / automatic promotion / 2026 labels / open
market input. Numeric performance is not asserted before live replay.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime
import hashlib
import math
import re
from typing import Any
import numpy as np
import pandas as pd

VERSION="NCAAF_HIERARCHICAL_DYNAMIC_SCORE_CHALLENGER_20261017_V1"
FIRST=2013
LAST=2025
RECENCY_YEARS={"full":None,"half_life_4y":4.0,"half_life_2y":2.0}
HEADS=("margin","total","h2h")
MAX_BASE_FEATURES=36
MAX_MODERN_FIELDS=12


def _num(frame, col):
    if col not in frame: return pd.Series(np.nan,index=frame.index,dtype=float)
    return pd.to_numeric(frame[col],errors="coerce").replace([np.inf,-np.inf],np.nan)


def _key(x):
    if x is None or pd.isna(x):return ""
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def _get_identity(frame, names):
    result=pd.Series("",index=frame.index,dtype="string")
    for c in names:
        if c in frame:
            candidate=frame[c].astype("string").fillna("").map(_key)
            result=result.mask(result.eq(""),candidate)
    return result.fillna("").astype(str)


def _game_dates(frame):
    out=pd.Series(pd.NaT,index=frame.index,dtype="datetime64[ns, UTC]")
    for c in ("Game_Date","Game_Start","Game_Start_Date","Start_Date","Start_Time","Date"):
        if c not in frame:continue
        val=pd.to_datetime(frame[c],errors="coerce",utc=True)
        out=out.fillna(val)
    return out.dt.normalize()


def _metric(y, p, head):
    y=np.asarray(y,float);p=np.asarray(p,float)
    mask=np.isfinite(y)&np.isfinite(p)
    if head=="h2h":
        mask &= ~np.isclose(y,0.)
        q=np.clip(p[mask],1e-6,1-1e-6);truth=(y[mask]>0).astype(float)
        return {"n":int(mask.sum()),"logloss":float(-np.mean(truth*np.log(q)+(1-truth)*np.log1p(-q))) if len(q) else None,
                "brier":float(np.mean((truth-q)**2)) if len(q) else None}
    dif=y[mask]-p[mask]
    return {"n":int(mask.sum()),"rmse":float(np.sqrt(np.mean(dif*dif))) if len(dif) else None,
            "mae":float(np.mean(np.abs(dif))) if len(dif) else None}


def _objective(score,head):
    return score.get("logloss") if head=="h2h" else score.get("rmse")


def reconstruct_asof_team_strength(frame):
    """No game-day result can influence a feature for another game that day.

    State is regressed at season boundaries. Games with invalid dates/identities
    receive missing dynamic features (not synthetic evidence); are not used to
    update team strength. This leaves them available to static score models.
    """
    n=len(frame)
    yr=_num(frame,"Season").to_numpy(float)
    margin=_num(frame,"Actual_Margin").to_numpy(float)
    total=_num(frame,"Actual_Total").to_numpy(float)
    date=_game_dates(frame)
    team=_get_identity(frame,("Team_Norm","Team","Source_Team_Name","Home_Team_Norm"))
    opp=_get_identity(frame,("Opponent_Norm","Opponent","Opp_Norm","Away_Team_Norm"))
    home=_num(frame,"Is_Home").to_numpy(float)
    if not np.isfinite(home).any():
        home=_num(frame,"Home").to_numpy(float)
    columns=["Dyn_Power_Diff","Dyn_Offense_Diff","Dyn_Defense_Diff", "Dyn_Expected_Margin",
             "Dyn_Expected_Total","Dyn_Opponent_Quality_Diff","Dyn_Team_Games_Prior",
             "Dyn_Opp_Games_Prior","Dyn_Prior_Season_Power_Diff","Dyn_Rest_Days_Diff",
             "Dyn_League_Points_AsOf"]
    arr={c:np.full(n,np.nan,dtype=float) for c in columns}
    # Persist program effects across years but regress aggressively at new season.
    states=defaultdict(lambda: {"power":0.,"off":0.,"def":0.,"n":0,"last":None,
                                "sos":0.,"prev_power":0.,"prior_n":0})
    global_pts=27.0
    ready=[]
    for i in range(n):
        if np.isfinite(yr[i]) and FIRST<=yr[i]<=LAST and pd.notna(date.iloc[i]) and team.iloc[i] and opp.iloc[i] and team.iloc[i]!=opp.iloc[i]:
            ready.append(i)
    if not ready:
        return pd.DataFrame(arr,index=frame.index),{"status":"NO_VALID_PREGAME_IDENTITIES","valid_games":0,"invalid_games":n}
    # A source frame is expected to hold exactly one row per physical game.
    # Fail closed instead of learning twice from mirrored home/away game sides.
    if "Source_Game_ID" in frame:
        ids=frame.loc[ready,"Source_Game_ID"].astype("string").fillna("").str.strip()
        valid=ids.ne("")&~ids.str.lower().isin(["nan","none","<na>"])
        game_keys=frame.loc[ready,"Season"].astype(str)+":"+ids
        if bool(game_keys.loc[valid].duplicated().any()):
            raise ValueError("DUPLICATE_PHYSICAL_GAME_ID_IN_SCORE_RESEARCH")
    ready.sort(key=lambda i:(date.iloc[i],int(yr[i]),i))
    season_last=None
    for ds in sorted(set(date.iloc[i] for i in ready)):
        day_indices=[i for i in ready if date.iloc[i]==ds]
        # Some same-day slate contains a season-change edge; process season separately.
        per_year=defaultdict(list)
        for i in day_indices:per_year[int(yr[i])].append(i)
        for s,idxs in sorted(per_year.items()):
            if season_last!=s:
                if season_last is not None:
                    for st in states.values():
                        st["prev_power"]=st["power"]
                        st["power"]*=.58; st["off"]*=.58; st["def"]*=.58
                        st["n"]=0;st["sos"]=0.;st["prior_n"]=0;st["last"]=None
                season_last=s
            to_update=[]
            for i in idxs:
                a=states[team.iloc[i]];b=states[opp.iloc[i]]
                ha=(float(home[i]) if np.isfinite(home[i]) else 0.5)
                site_adv=2.4 if ha>=0.9 else (-2.4 if ha<=0.1 else 0.)
                strength=a["power"]-b["power"]
                calc_margin=float(strength+site_adv)
                points_est=float(2*global_pts + a["off"]+b["off"]-a["def"]-b["def"])
                fields={
                    "Dyn_Power_Diff":strength,
                    "Dyn_Offense_Diff":a["off"]-b["off"],
                    "Dyn_Defense_Diff":b["def"]-a["def"],
                    "Dyn_Expected_Margin":calc_margin,
                    "Dyn_Expected_Total":points_est,
                    "Dyn_Opponent_Quality_Diff":a["sos"]-b["sos"],
                    "Dyn_Team_Games_Prior":float(a["n"]),
                    "Dyn_Opp_Games_Prior":float(b["n"]),
                    "Dyn_Prior_Season_Power_Diff":a["prev_power"]-b["prev_power"],
                    "Dyn_Rest_Days_Diff": float((ds-a["last"]).days-(ds-b["last"]).days) if a["last"] is not None and b["last"] is not None else np.nan,
                    "Dyn_League_Points_AsOf":float(global_pts),
                }
                for col,v in fields.items():arr[col][i]=v
                if np.isfinite(margin[i]) and np.isfinite(total[i]):
                    to_update.append((i,calc_margin,points_est,global_pts,float(a["power"]),float(b["power"])))
            # Commit yesterday's information only AFTER all same-day features exist.
            points=[]
            for i,ex_m,ex_t,reference,pa,pb in to_update:
                a=states[team.iloc[i]];b=states[opp.iloc[i]]
                resid=float(np.clip(margin[i]-ex_m,-42.,42.))
                # Hierarchical smoothing: new teams update more, experienced
                # teams' ratings fluctuate less. These are fixed, not tuned on holdout.
                ka=0.19/(1+.045*a["n"]);kb=0.19/(1+.045*b["n"])
                a["power"]+=ka*resid;b["power"]-=kb*resid
                pfor=(total[i]+margin[i])/2.;pagainst=(total[i]-margin[i])/2.
                # Simple shrinkage of scoring components toward established
                # league baseline and each side's *prior* defensive quality.
                erra=float(np.clip(pfor-(reference+a["off"]-b["def"]),-35,35))
                errb=float(np.clip(pagainst-(reference+b["off"]-a["def"]),-35,35))
                a["off"]+=.10*erra;b["def"]-=.10*erra
                b["off"]+=.10*errb;a["def"]-=.10*errb
                a["sos"]=(a["sos"]*a["prior_n"]+pb)/(a["prior_n"]+1)
                b["sos"]=(b["sos"]*b["prior_n"]+pa)/(b["prior_n"]+1)
                a["prior_n"]+=1;b["prior_n"]+=1;a["n"]+=1;b["n"]+=1
                a["last"]=ds;b["last"]=ds
                points.append(float(total[i]/2))
            if points:
                # Slowly track scoring-era changes from *past* results.
                global_pts=float(.965*global_pts+.035*float(np.mean(points)))
    season_counts={int(v):{"all":int(np.sum(yr==v)),"valid":0} for v in np.unique(yr[np.isfinite(yr)])}
    for i in ready:season_counts[int(yr[i])]["valid"]+=1
    for v in season_counts.values():v["coverage"]=round(v["valid"]/v["all"],4) if v["all"] else None
    population=len(ready)/n if n else 0.
    audit={"status":"PASS" if population>=.90 else "PARTIAL_IDENTITIES_OR_DATES",
           "valid_games":len(ready),"invalid_games":n-len(ready),
           "asof_coverage":round(population,4),"by_season":season_counts,
           "seasons":sorted({int(yr[i]) for i in ready}),"dynamic_columns":columns,
           "same_day_update_policy":"ONLY_AFTER_ALL_SAME_DAY_PREGAME_FEATURES",
           "offseason_regression":.58,"production_authority":0}
    return pd.DataFrame(arr,index=frame.index),audit


def _select_base_columns(frame, fixed_candidates, *, train_mask):
    from ncaaf_full_history_foundation import _safe_feature
    curated=list(dict.fromkeys(["Context_Intercept",*fixed_candidates,
       *[x for x in frame.columns if x.startswith("Dyn_")]]))
    sel=[]
    for col in curated:
        if col not in frame or (not col.startswith("Dyn_") and not _safe_feature(col)):
            continue
        v=_num(frame.loc[train_mask],col)
        if col!="Context_Intercept" and (v.notna().sum()<max(30,.07*train_mask.sum()) or v.nunique(dropna=True)<2):
            continue
        sel.append(col)
        if len(sel)>=MAX_BASE_FEATURES:break
    return sel or ["Context_Intercept"]


def _fit_predict(Xtr,Xval,y,head,weights):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge,LogisticRegression
    keep=np.isfinite(y)
    if head=="h2h":keep &= ~np.isclose(y,0.)
    if int(keep.sum())<200:return np.full(len(Xval),np.nan)
    if head=="h2h" and len(np.unique(y[keep]>0))<2:return np.full(len(Xval),np.nan)
    imp=SimpleImputer(strategy="median",keep_empty_features=True)
    x=imp.fit_transform(Xtr[keep]);v=imp.transform(Xval)
    scaler=StandardScaler();x=scaler.fit_transform(x);v=scaler.transform(v)
    if head=="h2h":
        fit=LogisticRegression(C=.07,solver="lbfgs",max_iter=220)
        fit.fit(x,(y[keep]>0).astype(int),sample_weight=weights[keep])
        return fit.predict_proba(v)[:,1]
    fit=Ridge(alpha=45.)
    fit.fit(x,y[keep],sample_weight=weights[keep]);return np.asarray(fit.predict(v),float)


def _loss_array(y,p,head):
    truth=(np.asarray(y,float)>0).astype(float)
    if head=="h2h":
        prob=np.clip(np.asarray(p,float),1e-6,1-1e-6)
        return -(truth*np.log(prob)+(1-truth)*np.log1p(-prob))
    return (np.asarray(y,float)-np.asarray(p,float))**2


def _paired_holdout(frame, y, baseline, challenger, head, *, years=(2024,2025), boot=250):
    """Calendar-day/block bootstrap, paired on the exact same held-out games.

    The 2024/25 holdout is evaluation-only: never used for selecting a feature,
    weight or history window. A positive improvement means lower error.
    """
    sy=_num(frame,"Season").to_numpy(float)
    keep=np.isin(sy,years)&np.isfinite(y)&np.isfinite(baseline)&np.isfinite(challenger)
    if head=="h2h":keep &= ~np.isclose(y,0.)
    sel=np.flatnonzero(keep)
    if len(sel)<100:return {"status":"INSUFFICIENT_PAIRED_CONFIRMATION","n":int(len(sel))}
    old=_loss_array(y[sel],np.asarray(baseline)[sel],head)
    new=_loss_array(y[sel],np.asarray(challenger)[sel],head)
    if head=="h2h":
        gain=float(np.mean(old)-np.mean(new));metric="logloss"
    else:
        gain=float(np.sqrt(np.mean(old))-np.sqrt(np.mean(new)));metric="rmse"
    d=_game_dates(frame)
    clusters=(frame["Season"].astype(str)+"_"+d.dt.strftime("%Y-%m-%d").fillna("unknown")).iloc[sel].to_numpy()
    unique=np.unique(clusters)
    groups=[np.flatnonzero(clusters==u) for u in unique]
    rng=np.random.default_rng(172910)
    boots=[]
    for _ in range(boot):
        ids=np.concatenate([groups[j] for j in rng.integers(0,len(groups),size=len(groups))])
        if head=="h2h":v=float(np.mean(old[ids])-np.mean(new[ids]))
        else:v=float(np.sqrt(np.mean(old[ids]))-np.sqrt(np.mean(new[ids])))
        boots.append(v)
    low,high=np.quantile(boots,[.025,.975])
    return {"status":"PAIRED_CONFIRMATION_SCORED","n":int(len(sel)),"metric":metric,
            "improvement":gain,"block_bootstrap_ci95":[float(low),float(high)],
            "cluster_count":len(groups),"beats_baseline_with_positive_ci":bool(low>0),
            "note":"Exploratory retrospective evidence; not a fresh prospective holdout"}


def _modern_fields(frame, season):
    """The only allowed specialist source is independently shifted pregame CFBD."""
    candidates=[c for c in frame if re.match(r"^CFBD_(?:Opp_)?(?:Season|Recent3|Recent5|PrevSeason)_CFBD_Game_",str(c)) and str(c).endswith("_Prior")]
    look=frame.loc[np.isin(season,[2022,2023])]
    groups=[]
    for c in candidates:
        v=_num(look,c)
        coverage=float(v.notna().mean()) if len(v) else 0.
        if coverage<.40 or v.nunique(dropna=True)<2:continue
        # Keep one or two representatives per conceptual family rather than
        # 20 highly correlated PPA variants.
        field=c.split("_CFBD_Game_",1)[-1]
        family=re.sub(r"(?:_Per_|_Adjusted|_Average|_Rate|_Allowed).*", "",field)
        groups.append((coverage,c,family))
    groups.sort(key=lambda row:(-row[0],row[1]))
    chosen=[];seen=defaultdict(int)
    for _,col,family in groups:
        if seen[family]>=2:continue
        chosen.append(col);seen[family]+=1
        if len(chosen)>=MAX_MODERN_FIELDS:break
    return chosen


def _modern_incremental(frame,season,margin,total,base_pred,modern_cols):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    report={"status":"NO_VERIFIED_MODERN_PREGAME_DATA","fields":modern_cols,"per_head":{},
            "selection_year":2023,"confirmation_years":[2024,2025],"authority":0}
    n=len(frame)
    if not modern_cols:return report
    # Build one normalised feature matrix. Never use 2024/25 to choose
    # columns, blend weights or residual-model hyperparameters.
    X=frame[modern_cols].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    if int(np.sum((season==2022)&np.isfinite(base_pred["margin"])))<300:return report
    chosen={}
    out={h:np.full(n,np.nan) for h in HEADS}
    for head in HEADS:
        label=margin if head!="total" else total
        train=(season==2022)&np.isfinite(base_pred[head])&np.isfinite(label)
        valid=(season==2023)&np.isfinite(base_pred[head])&np.isfinite(label)
        if head=="h2h":train &= ~np.isclose(label,0.);valid &= ~np.isclose(label,0.)
        if train.sum()<300 or valid.sum()<300:
            report["per_head"][head]={"status":"INSUFFICIENT_MODERN_OOF"};continue
        def calc(train_ix, predict_ix):
            Xtrain=X[train_ix];Xvalid=X[predict_ix]
            observed=np.isfinite(Xtrain).sum(axis=0)>max(30,.15*len(Xtrain))
            observed &= np.array([len(np.unique(Xtrain[np.isfinite(Xtrain[:,k]),k]))>1 for k in range(Xtrain.shape[1])])
            if not observed.any():return np.zeros(predict_ix.sum()),0
            imp=SimpleImputer(strategy="median");a=imp.fit_transform(Xtrain[:,observed]);b=imp.transform(Xvalid[:,observed])
            scal=StandardScaler();a=scal.fit_transform(a);b=scal.transform(b)
            target=(label[train_ix]>0).astype(float) if head=="h2h" else label[train_ix]
            resid=target-base_pred[head][train_ix]
            model=Ridge(alpha=120.)
            model.fit(a,resid)
            correction=np.asarray(model.predict(b),float)
            return np.clip(correction,-.22,.22) if head=="h2h" else np.clip(correction,-8.,8.),int(observed.sum())
        delta,feature_count=calc(train,valid)
        raw=base_pred[head][valid]
        actual=label[valid]
        weights=[0,.25,.5]
        scores=[]
        for w in weights:
            p=raw+w*delta
            if head=="h2h":p=np.clip(p,.00001,.99999)
            scores.append((_objective(_metric(actual,p,head),head),w))
        scores=[(score,w) for score,w in scores if score is not None and np.isfinite(score)]
        if not scores:
            report["per_head"][head]={"status":"NO_TESTABLE_VALIDATION"};continue
        chosen_w=min(scores,key=lambda z:(z[0],z[1]))[1]
        chosen[head]=chosen_w
        per={"status":"SELECTED_ON_2023","selected_blend_weight":chosen_w,
             "selection_rmse_or_logloss":{"baseline":next((v for v,w in scores if w==0),None),
                                         "chosen":next((v for v,w in scores if w==chosen_w),None)},
             "selected_fields_used":feature_count,"holdout":{}}
        for sy in (2024,2025):
            tr=(season>=2022)&(season<sy)&np.isfinite(base_pred[head])&np.isfinite(label)
            va=(season==sy)&np.isfinite(base_pred[head])&np.isfinite(label)
            if head=="h2h":tr &= ~np.isclose(label,0);va &= ~np.isclose(label,0)
            if tr.sum()<300 or va.sum()<200:continue
            diff,k=calc(tr,va)
            pred=base_pred[head][va]+chosen_w*diff
            if head=="h2h":pred=np.clip(pred,1e-5,1-1e-5)
            out[head][va]=pred
            base_m=_metric(label[va],base_pred[head][va],head)
            new_m=_metric(label[va],pred,head)
            v0=_objective(base_m,head);v1=_objective(new_m,head)
            per["holdout"][str(sy)]={"baseline":base_m,"specialist":new_m,
               "improvement":float(v0-v1) if v0 is not None and v1 is not None else None,
               "features_fit":k}
        report["per_head"][head]=per
    report["status"]="RESEARCH_ONLY_OOF_EVALUATED" if chosen else "INSUFFICIENT_MODERN_OOF"
    report["decision"]="NEVER_PROMOTE_AUTOMATICALLY"
    return report


def run_hierarchical_dynamic_research(frame, base_features, baseline_predictions,
                                     *, frozen_source=None, log_func=print):
    """Research-only; never imports or modifies production code."""
    from ncaaf_core_feature_parity import audit_core_parity, required_research_features
    df=frame.copy().reset_index(drop=True)
    season=_num(df,"Season").to_numpy(float)
    if np.isfinite(season).any() and np.nanmax(season)>LAST:
        raise ValueError("2026+ is not permitted in hierarchical research")
    margin=_num(df,"Actual_Margin").to_numpy(float)
    total=_num(df,"Actual_Total").to_numpy(float)
    dyn,audit=reconstruct_asof_team_strength(df)
    for c in dyn:df[c]=dyn[c].to_numpy()
    df["Context_Intercept"]=1.
    candidate=list(dict.fromkeys([*required_research_features(),
                                   *[c for c in df if c.startswith("Dyn_")],
                                   *base_features]))
    predictions={k:{h:np.full(len(df),np.nan) for h in HEADS} for k in RECENCY_YEARS}
    folds=[]; features_by_fold={}
    for y in sorted({int(a) for a in season[np.isfinite(season)] if FIRST<a<=LAST}):
        tr=season<y;va=season==y
        if min(np.isfinite(margin[tr]).sum(),np.isfinite(total[tr]).sum())<200 or va.sum()<50:continue
        cols=_select_base_columns(df,candidate,train_mask=tr)
        features_by_fold[y]=cols
        xtr=df.loc[tr,cols].apply(pd.to_numeric,errors="coerce").to_numpy(float)
        xval=df.loc[va,cols].apply(pd.to_numeric,errors="coerce").to_numpy(float)
        all_y={"margin":margin[tr],"total":total[tr],"h2h":margin[tr]}
        report={"validation_year":y,"train_games":int(tr.sum()),"validation_games":int(va.sum()),
                "train_from_year":int(np.min(season[tr])),"feature_count":len(cols),"heads":{}}
        for k,half in RECENCY_YEARS.items():
            age=y-season[tr]
            weights=np.ones(len(age)) if half is None else np.maximum(.15,2**(-age/half))
            for head in HEADS:
                pred=_fit_predict(xtr,xval,all_y[head],head,weights)
                predictions[k][head][va]=pred
                result=_metric(margin[va] if head!="total" else total[va],pred,head)
                report["heads"].setdefault(head,{})[k]=result
        folds.append(report)
        log_func(f"[NCAAF-HIERARCHICAL-OOF] val_season={y} train_games={int(tr.sum())} "
                 f"valid_games={int(va.sum())} features={len(cols)} policy=ALL_HISTORY_2013_FORWARD "
                 f"opening_required=FALSE authority=0")
    # Hyperparameters chosen using pre-2024 validation alone. Confirmation
    # years are excluded from policy and field selection.
    chosen={};pre_scores={};confirm={}
    selection=np.isin(season,np.arange(2018,2024))
    for head in HEADS:
        y=margin if head!="total" else total
        vals=[]
        for key in RECENCY_YEARS:
            m=_metric(y[selection],predictions[key][head][selection],head)
            measure=_objective(m,head)
            per=[f["heads"][head][key] for f in folds if 2018<=f["validation_year"]<=2023]
            yearly=[_objective(x,head) for x in per]
            yearly=[v for v in yearly if v is not None]
            # Prefer stability and a larger historical effective sample.
            penal=float(np.std(yearly))*0.03 if len(yearly)>1 else 0.
            if measure is not None:
                vals.append((measure+penal,list(RECENCY_YEARS).index(key),key))
            pre_scores[key]={**pre_scores.get(key,{}),head:{"selection_metrics":m,"year_to_year_std":float(np.std(yearly)) if yearly else None}}
        if not vals:raise RuntimeError(f"NO_CHRONOLOGICAL_POLICY_SELECTION_{head}")
        best=min(vals)
        chosen[head]=best[2]
        before={}
        for sy in (2024,2025):
            ix=(season==sy)
            before[str(sy)]={"selected":_metric(y[ix],predictions[best[2]][head][ix],head),
                             "full_history":_metric(y[ix],predictions["full"][head][ix],head)}
        confirm[head]=before
    # Evaluate modern residual specialists strictly on pregame source fields.
    modern=_modern_fields(df,season)
    chosen_preds={h:predictions[chosen[h]][h] for h in HEADS}
    modern_report=_modern_incremental(df,season,margin,total,chosen_preds,modern)
    parity=audit_core_parity(df,list(dict.fromkeys(sum(features_by_fold.values(),[]))),features_by_fold,
                             source_file=frozen_source)
    selected_report={h:{"policy":chosen[h],"aggregate":_metric(margin if h!="total" else total,chosen_preds[h],h)} for h in HEADS}
    paired={h:_paired_holdout(df,margin if h!="total" else total,
                             baseline_predictions[h],chosen_preds[h],h) for h in HEADS}
    result={"status":"PASS" if audit["status"]=="PASS" else "LIMITED_DYNAMIC_COVERAGE",
            "version":VERSION,"authority":0,"automatic_promotion":False,
            "years":sorted({int(s) for s in season[np.isfinite(season)]}),
            "dynamic_source":audit,"core_feature_parity":parity,
            "folds":folds,"history_policy_selection":{"search_years":[2018,2019,2020,2021,2022,2023],
                                                   "policies":pre_scores,"selected":chosen},
            "selected_heads":selected_report,"holdout_2024_2025":confirm,
            "paired_vs_existing_score_foundation_2024_2025":paired,
            "modern_specialist":modern_report,"source_seal":"2026+ EXCLUDED; NO MARKET OPEN/CLOSE AS PREDICTORS",
            "rule":"NO_AUTO_PROMOTION__PROSPECTIVE_SHADOW_REQUIRED__FROZEN_PRODUCTION_V1"}
    log_func("[NCAAF-HIERARCHICAL-SELECT] "+str({"status":result["status"],
            "by_head":chosen,"frozen_feature_parity":parity["status"],
            "modern_specialist":modern_report["status"],"production_authority":0}))
    return result
