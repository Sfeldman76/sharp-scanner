"""NFL specialized modeling V1.5: H2H blend/calibration + score-domain Spread/Totals models.

RESEARCH ONLY. No artifact publication, no production authority, no BigQuery writes.
2026 remains sealed. The model-development seasons are 2017-2025 and the outer
validation seasons remain 2021-2025, so V1.5 results are development evidence,
not fresh confirmation after the V1.4 architecture decision.

Design:
- H2H: preserve V1.4 compact logistic + shallow tree and test fixed/OOF-stacked blends.
- SPREADS: predict actual oriented scoring margin, then map fair-margin vs the
  retrospective closing spread to cover probability using ONLY prior-season OOF
  residuals. The closing spread is never a fitted predictor.
- TOTALS: predict actual game total, then map fair-total vs retrospective closing
  total to Over probability using ONLY prior-season OOF residuals. The closing
  total is never a fitted predictor.
- Historical closing prices/lines remain retrospective diagnostics and do not
  establish executable ROI or CLV.
"""
from __future__ import annotations

import json
import math
from typing import Iterable

import numpy as np
import pandas as pd

from nfl_feature_audit_v1 import _flatten_manifest
from nfl_challenger_v1 import (
    EXPERIMENT_SEASONS, VALIDATION_SEASONS, SEALED_YEAR,
    COMPACT_FEATURES, physical_games, build_readonly_query,
    _feature_matrix, _scores, _numeric,
)

SOURCE_TAG = 'nfl-specialized-v1.5-score-domain-h2h-stack-20260930'
MIN_INTERNAL_VALIDATION_GAMES = 180
H2H_FAMILIES = ('H2H_HIST_RATE','H2H_COMPACT','H2H_TREE','H2H_BLEND50','H2H_STACKED')
REG_FAMILIES = ('BASE_MEAN','COMPACT_RIDGE','FULL_RIDGE','TREE_REG','BLEND50')


def _safe_logit(p):
    p=np.clip(np.asarray(p,dtype=float),0.005,0.995)
    return np.log(p/(1-p))


def _fit_h2h_base(family,train,valid):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    y=train['H2H_label'].to_numpy(float)
    if family=='H2H_COMPACT':
        cols=list(COMPACT_FEATURES['H2H']); A,B=_feature_matrix(train,valid,cols)
        sc=StandardScaler().fit(A)
        model=LogisticRegression(C=.35,max_iter=900,solver='lbfgs',random_state=17)
        model.fit(sc.transform(A),y)
        p=model.predict_proba(sc.transform(B))[:,list(model.classes_).index(1.)]
    elif family=='H2H_TREE':
        cols=list(_flatten_manifest()); A,B=_feature_matrix(train,valid,cols)
        model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,
            min_samples_leaf=65,learning_rate=.04,l2_regularization=15,
            early_stopping=False,random_state=17)
        model.fit(A,y)
        p=model.predict_proba(B)[:,list(model.classes_).index(1.)]
    else:
        raise ValueError(f'UNKNOWN_H2H_BASE {family}')
    return np.clip(np.asarray(p,float),.001,.999)


def _h2h_internal_oof(train):
    """Season-forward OOF predictions entirely inside the outer training window."""
    rows=[]
    seasons=sorted(int(x) for x in train.Season.unique())
    # Need at least 3 full prior seasons before an internal validation season.
    for s in seasons:
        prior=[y for y in seasons if y<s]
        if len(prior)<3: continue
        fit=train.loc[train.Season.lt(s)&train.H2H_label.notna()].copy()
        val=train.loc[train.Season.eq(s)&train.H2H_label.notna()].copy()
        if len(fit)<650 or len(val)<MIN_INTERNAL_VALIDATION_GAMES: continue
        pc=_fit_h2h_base('H2H_COMPACT',fit,val)
        pt=_fit_h2h_base('H2H_TREE',fit,val)
        for i,(_,r) in enumerate(val.iterrows()):
            rows.append((s,float(r.H2H_label),float(pc[i]),float(pt[i])))
    return pd.DataFrame(rows,columns=['Season','y','compact','tree'])


def _h2h_predictions(train,val):
    from sklearn.linear_model import LogisticRegression
    pc=_fit_h2h_base('H2H_COMPACT',train,val)
    pt=_fit_h2h_base('H2H_TREE',train,val)
    blend=np.clip((pc+pt)/2,.001,.999)
    oof=_h2h_internal_oof(train)
    if len(oof)>=MIN_INTERNAL_VALIDATION_GAMES and oof.y.nunique()==2:
        X=np.c_[_safe_logit(oof.compact),_safe_logit(oof.tree)]
        meta=LogisticRegression(C=.35,max_iter=600,solver='lbfgs',random_state=17)
        meta.fit(X,oof.y.to_numpy(float))
        Xv=np.c_[_safe_logit(pc),_safe_logit(pt)]
        stacked=meta.predict_proba(Xv)[:,list(meta.classes_).index(1.)]
        stacked=np.clip(stacked,.001,.999)
        stack_info={'status':'OOF_STACKED','internal_oof_games':int(len(oof)),
                    'internal_oof_seasons':sorted(int(x) for x in oof.Season.unique()),
                    'coef':[round(float(x),6) for x in meta.coef_.ravel()],
                    'intercept':round(float(meta.intercept_[0]),6)}
    else:
        stacked=blend.copy()
        stack_info={'status':'FALLBACK_BLEND50','internal_oof_games':int(len(oof))}
    hist=np.full(len(val),(float(train.H2H_label.sum())+1)/(len(train)+2),dtype=float)
    return {'H2H_HIST_RATE':hist,'H2H_COMPACT':pc,'H2H_TREE':pt,'H2H_BLEND50':blend,'H2H_STACKED':stacked},stack_info


def _regression_features(market,family):
    if family=='COMPACT_RIDGE': return list(COMPACT_FEATURES[market])
    return list(_flatten_manifest())


def _fit_regression(family,market,train,valid,target):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingRegressor
    if family=='BASE_MEAN':
        mu=float(pd.to_numeric(train[target],errors='coerce').mean())
        return np.full(len(valid),mu,dtype=float)
    if family=='BLEND50':
        a=_fit_regression('COMPACT_RIDGE',market,train,valid,target)
        b=_fit_regression('TREE_REG',market,train,valid,target)
        return (a+b)/2
    cols=_regression_features(market,family)
    A,B=_feature_matrix(train,valid,cols)
    y=pd.to_numeric(train[target],errors='coerce').to_numpy(float)
    if not np.isfinite(y).all(): raise RuntimeError(f'NONFINITE_TARGET {market} {target}')
    if family in ('COMPACT_RIDGE','FULL_RIDGE'):
        sc=StandardScaler().fit(A)
        alpha=12.0 if family=='COMPACT_RIDGE' else 55.0
        model=Ridge(alpha=alpha,random_state=17)
        model.fit(sc.transform(A),y)
        pred=model.predict(sc.transform(B))
    elif family=='TREE_REG':
        model=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=7,
            min_samples_leaf=65,learning_rate=.04,l2_regularization=20,
            early_stopping=False,random_state=17)
        model.fit(A,y)
        pred=model.predict(B)
    else: raise ValueError(f'UNKNOWN_REG_FAMILY {family}')
    return np.asarray(pred,dtype=float)


def _regression_internal_residuals(train,market,family,target):
    residuals=[]; used=[]
    seasons=sorted(int(x) for x in train.Season.unique())
    for s in seasons:
        prior=[y for y in seasons if y<s]
        if len(prior)<3: continue
        fit=train.loc[train.Season.lt(s)].copy(); val=train.loc[train.Season.eq(s)].copy()
        if len(fit)<650 or len(val)<MIN_INTERNAL_VALIDATION_GAMES: continue
        pred=_fit_regression(family,market,fit,val,target)
        actual=pd.to_numeric(val[target],errors='coerce').to_numpy(float)
        mask=np.isfinite(actual)&np.isfinite(pred)
        if mask.any():
            residuals.extend((actual[mask]-pred[mask]).tolist()); used.append(s)
    r=np.asarray(residuals,float)
    r=r[np.isfinite(r)]
    if len(r)<MIN_INTERNAL_VALIDATION_GAMES:
        raise RuntimeError(f'INSUFFICIENT_OOF_RESIDUALS {market} {family} n={len(r)}')
    return r,sorted(set(used))


def _empirical_prob_above(residuals,threshold):
    """Laplace-smoothed empirical P(residual > threshold)."""
    r=np.sort(np.asarray(residuals,float)); t=np.asarray(threshold,float)
    idx=np.searchsorted(r,t,side='right')
    n=len(r); greater=n-idx
    return np.clip((greater+0.5)/(n+1.0),.001,.999)


def _continuous_metrics(y,pred):
    y=np.asarray(y,float); pred=np.asarray(pred,float)
    mask=np.isfinite(y)&np.isfinite(pred); y=y[mask]; pred=pred[mask]
    if not len(y): return {'n':0}
    err=pred-y
    corr=float(np.corrcoef(y,pred)[0,1]) if len(y)>2 and np.std(y)>0 and np.std(pred)>0 else None
    return {'n':int(len(y)), 'mae':round(float(np.mean(np.abs(err))),6),
            'rmse':round(float(np.sqrt(np.mean(err**2))),6),
            'bias':round(float(np.mean(err)),6),
            'corr':round(corr,6) if corr is not None and math.isfinite(corr) else None}


def _edge_diagnostics(status,edge,thresholds=(0,1,2,3,4)):
    """Retrospective direction accuracy by absolute model-vs-close disagreement.

    status is WIN/LOSS/PUSH for the oriented Home/neutral-canonical side.
    Positive edge selects oriented side; negative edge selects opposite side.
    Pushes excluded. No odds/ROI claim.
    """
    s=np.asarray(status,dtype=object); e=np.asarray(edge,float)
    out={}
    for th in thresholds:
        mask=np.isfinite(e)&(np.abs(e)>=float(th))&(s!='PUSH')&(s!='MISSING')
        if not mask.any():
            out[str(th)]={'n':0,'direction_accuracy':None}; continue
        win=((e[mask]>0)&(s[mask]=='WIN'))|((e[mask]<0)&(s[mask]=='LOSS'))
        # exact zero edge has no directional play
        nonzero=e[mask]!=0; win=win[nonzero]
        out[str(th)]={'n':int(len(win)), 'direction_accuracy':round(float(np.mean(win)),6) if len(win) else None}
    return out


def _paired_logloss_delta(y,baseline,challenge):
    y=np.asarray(y,float); b=np.clip(np.asarray(baseline,float),.001,.999); p=np.clip(np.asarray(challenge,float),.001,.999)
    lb=-(y*np.log(b)+(1-y)*np.log(1-b)); lp=-(y*np.log(p)+(1-y)*np.log(1-p))
    d=lp-lb
    return round(float(np.mean(d)),6)


def specialized_tournament(games:pd.DataFrame,log_func=print):
    if games.Season.eq(SEALED_YEAR).any() or not games.Season.isin(EXPERIMENT_SEASONS).all():
        raise RuntimeError('[NFL-SPECIALIZED-V1-HOLD] SEALED_YEAR_PRESENT')
    games=games.copy()
    games['actual_margin']=pd.to_numeric(games.Team_Score,errors='coerce')-pd.to_numeric(games.Opponent_Score,errors='coerce')
    games['actual_total']=pd.to_numeric(games.Team_Score,errors='coerce')+pd.to_numeric(games.Opponent_Score,errors='coerce')
    if games[['actual_margin','actual_total']].isna().any().any(): raise RuntimeError('MISSING_SCORE_TARGET')
    report={'status':'RESEARCH_RESULTS_ONLY','source_tag':SOURCE_TAG,'publication':False,'production_authority':0,
            'year2026':'SEALED','development_seasons':'2017-2025','outer_validation_seasons':list(VALIDATION_SEASONS),
            'h2h':{},'spreads':{},'totals':{},
            'limitations':['2021-2025 are development folds already inspected in V1.4; not fresh confirmation',
                           'historical close lines are retrospective and not verified executable quotes',
                           'no historical ROI or CLV claim']}

    # ---- H2H specialized blend ----
    h_store={f:{'y':[],'p':[],'close':[],'seasons':[]} for f in H2H_FAMILIES}
    h_folds=[]
    for vy in VALIDATION_SEASONS:
        tr=games.loc[games.Season.lt(vy)&games.H2H_label.notna()].copy()
        va=games.loc[games.Season.eq(vy)&games.H2H_label.notna()].copy()
        preds,stack_info=_h2h_predictions(tr,va); y=va.H2H_label.to_numpy(float)
        ref=pd.to_numeric(va.H2H_close_novig_reference,errors='coerce').to_numpy(float)
        row={'validate_season':vy,'train_games':len(tr),'validate_games':len(va),'stack':stack_info,'families':{}}
        for fam,p in preds.items():
            m=_scores(y,p); mask=np.isfinite(ref)
            if mask.sum()>=150:
                m['delta_logloss_vs_close_novig']=_paired_logloss_delta(y[mask],ref[mask],p[mask])
            row['families'][fam]=m
            h_store[fam]['y'].extend(y.tolist());h_store[fam]['p'].extend(p.tolist());h_store[fam]['close'].extend(ref.tolist());h_store[fam]['seasons'].extend([vy]*len(y))
        if np.isfinite(ref).sum()>=150: row['close_novig_reference']=_scores(y[np.isfinite(ref)],ref[np.isfinite(ref)])
        h_folds.append(row); log_func('[NFL-SPECIALIZED-V1-H2H-FOLD] '+json.dumps(row,sort_keys=True,default=str))
    h_summary={}
    for fam,o in h_store.items():
        h_summary[fam]=_scores(o['y'],o['p'])
        ref=np.asarray(o['close'],float); y=np.asarray(o['y'],float); p=np.asarray(o['p'],float); mask=np.isfinite(ref)
        if mask.sum(): h_summary[fam]['delta_logloss_vs_close_novig']=_paired_logloss_delta(y[mask],ref[mask],p[mask])
    report['h2h']={'folds':h_folds,'summary':h_summary}
    log_func('[NFL-SPECIALIZED-V1-H2H] '+json.dumps(h_summary,sort_keys=True))

    # ---- Spread margin / totals points models ----
    for market,target,line_col,status_col,out_key in [
        ('SPREADS','actual_margin','Spread_Value','SPREADS_status','spreads'),
        ('TOTALS','actual_total','Current_Total','TOTALS_status','totals')]:
        store={f:{'y':[],'p':[],'actual':[],'pred':[],'edge':[],'status':[],'seasons':[]} for f in REG_FAMILIES}
        folds=[]
        for vy in VALIDATION_SEASONS:
            tr=games.loc[games.Season.lt(vy)].copy(); va=games.loc[games.Season.eq(vy)].copy()
            line=pd.to_numeric(va[line_col],errors='coerce').to_numpy(float)
            actual=pd.to_numeric(va[target],errors='coerce').to_numpy(float)
            labels=va[market+'_label'].to_numpy(float)
            evalmask=np.isfinite(line)&np.isfinite(labels)
            row={'validate_season':vy,'train_games':len(tr),'validate_games':len(va),'settled_games':int(evalmask.sum()),'families':{}}
            for fam in REG_FAMILIES:
                pred=_fit_regression(fam,market,tr,va,target)
                resid,cal_seasons=_regression_internal_residuals(tr,market,fam,target)
                if market=='SPREADS':
                    edge=pred+line
                    threshold=-(pred+line)
                else:
                    edge=pred-line
                    threshold=line-pred
                p=_empirical_prob_above(resid,threshold)
                cont=_continuous_metrics(actual,pred)
                prob=_scores(labels[evalmask],p[evalmask])
                prob['delta_logloss_vs_50']=round(prob['log_loss']-(-math.log(.5)),6)
                row['families'][fam]={'score_model':cont,'settlement_probability':prob,
                    'oof_residual_games':int(len(resid)),'oof_residual_seasons':cal_seasons,
                    'residual_sd':round(float(np.std(resid,ddof=1)),6),
                    'edge_diagnostics':_edge_diagnostics(va[status_col].to_numpy(object),edge)}
                st=store[fam]; st['y'].extend(labels[evalmask].tolist()); st['p'].extend(p[evalmask].tolist()); st['actual'].extend(actual.tolist()); st['pred'].extend(pred.tolist()); st['edge'].extend(edge.tolist());st['status'].extend(va[status_col].tolist());st['seasons'].extend([vy]*len(va))
            folds.append(row);log_func(f'[NFL-SPECIALIZED-V1-{market}-FOLD] '+json.dumps(row,sort_keys=True,default=str))
        summary={}
        for fam,st in store.items():
            summary[fam]={'score_model':_continuous_metrics(st['actual'],st['pred']),
                          'settlement_probability':_scores(st['y'],st['p']),
                          'edge_diagnostics':_edge_diagnostics(st['status'],st['edge'])}
            summary[fam]['settlement_probability']['delta_logloss_vs_50']=round(summary[fam]['settlement_probability']['log_loss']-(-math.log(.5)),6)
        report[out_key]={'folds':folds,'summary':summary}
        log_func(f'[NFL-SPECIALIZED-V1-{market}] '+json.dumps(summary,sort_keys=True,default=str))

    log_func('[NFL-SPECIALIZED-V1-CONTRACT] status=RESEARCH_RESULTS_ONLY publication=FALSE production_authority=0 '
             'year2026=SEALED ncaaf=UNCHANGED legacy_nfl=UNCHANGED historical_roi_clv=NOT_VERIFIED')
    return report


def run_nfl_specialized_v1(*,bq_client=None,audit_report=None,log_func=print):
    if not isinstance(audit_report,dict) or audit_report.get('status')!='READY_FOR_OFFLINE_CHALLENGER_SANDBOX':
        raise RuntimeError('[NFL-SPECIALIZED-V1-HOLD] V1_3_AUDIT_NOT_GREEN')
    from google.cloud import bigquery
    from nfl_feature_audit_v1 import VIEW
    bq=bq_client or bigquery.Client(project='sharplogger')
    cols={f.name for f in bq.get_table(VIEW).schema}
    query=build_readonly_query(cols)
    log_func(f'[NFL-SPECIALIZED-V1-PREFLIGHT] status=PASS tag={SOURCE_TAG} publication=FALSE year2026=SEALED')
    df=bq.query(query,job_config=bigquery.QueryJobConfig(use_query_cache=True,maximum_bytes_billed=20*1024**3)).to_dataframe()
    games=physical_games(df)
    log_func('[NFL-SPECIALIZED-V1-GRAIN] '+json.dumps({'source_rows':len(df),'physical_games':len(games),
             'seasons':{str(k):int(v) for k,v in games.groupby('Season').size().items()}},sort_keys=True))
    return specialized_tournament(games,log_func=log_func)
