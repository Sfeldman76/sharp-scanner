"""NFL Challenger Sandbox V1.4: read-only chronological three-market baseline tournament.

No publication, no production authority, no BigQuery DDL/DML or model writes.
Runs ONLY if independent NFL V1.3 audit returned READY_FOR_OFFLINE_CHALLENGER_SANDBOX.

One game = one statistical unit (home row or stable neutral-side choice). Models
see ONLY the named V1.3 pregame/prior-only features. Historical close lines are
retrospective settlement labels, NOT as-of predictors or proof of ROI/CLV.
2026 is sealed and cannot enter model fit, calibration, or model selection.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from nfl_feature_audit_v1 import EXPLICIT_FEATURE_MANIFEST, EXCLUDED_SOURCE_FIELDS, VIEW, _flatten_manifest, fold_blueprint

SOURCE_TAG = 'nfl-challenger-v1.4-season-forward-three-market-no-publish-20260930'
EXPERIMENT_SEASONS = tuple(range(2017, 2026))
VALIDATION_SEASONS = (2021, 2022, 2023, 2024, 2025)
SEALED_YEAR = 2026
MARKETS = ('SPREADS', 'H2H', 'TOTALS')
FAMILIES = ('HIST_RATE', 'MATCHUP_COMPACT', 'REGULARIZED_FULL', 'TREE_SHALLOW')
IDENTITY = ('Season', 'Source_Name', 'Source_Game_ID')
LABEL_COLUMNS = ('Team_Score', 'Opponent_Score', 'Spread_Value', 'Current_Total',
                 'ML_Odds', 'ATS_Result_Close', 'Total_Result_Close')
REQUIRED_COLUMNS = tuple(dict.fromkeys((
    *IDENTITY, 'Season_Stage', 'Historical_Core_Eligible', 'Game_Date',
    'Team_Norm','Opponent_Norm','Is_Home','Is_Away','Is_Neutral', *LABEL_COLUMNS,
    *_flatten_manifest()
)))
# Explicitly small human-reviewable feature family; no market/current-game values.
COMPACT_FEATURES = {
 'SPREADS': ('Week_Number','Is_Home','Is_Neutral','Is_Division_Game',
  'Rest_Differential_Days','WinPct_Prior_Diff','ATS_WinPct_Prior_Diff',
  'Avg_SU_Margin_Last5_Diff','Off_YPP_vs_Opp_Def_Last3_Diff',
  'Def_YPP_vs_Opp_Off_Last3_Diff','Prior_Season_WinPct_Diff'),
 'H2H': ('Week_Number','Is_Home','Is_Neutral','Is_Division_Game',
  'Rest_Differential_Days','WinPct_Prior_Diff','Avg_SU_Margin_Last5_Diff',
  'Off_YPP_vs_Opp_Def_Last3_Diff','Def_YPP_vs_Opp_Off_Last3_Diff',
  'Prior_Season_WinPct_Diff'),
 'TOTALS': ('Week_Number','Is_Home','Is_Neutral','Is_Night_Game',
  'Is_Division_Game','Rest_Differential_Days','Avg_Points_For_Last5_Prior',
  'Avg_Points_Against_Last5_Prior','Opp_Avg_Points_For_Last5_Prior',
  'Opp_Avg_Points_Against_Last5_Prior','Avg_Off_YPP_Last3_Prior',
  'Avg_Def_YPP_Last3_Prior','Opp_Avg_Off_YPP_Last3_Prior',
  'Opp_Avg_Def_YPP_Last3_Prior'),
}
assert all(set(f).issubset(set(_flatten_manifest())) for f in COMPACT_FEATURES.values())
assert not (set(_flatten_manifest()) & set(EXCLUDED_SOURCE_FIELDS))
assert not ({'Season','Source_Game_ID','Team_Score','Opponent_Score','Spread_Value','Current_Total','ML_Odds'} & set(_flatten_manifest()))


def build_readonly_query(view_columns=None):
    """Select only explicitly named columns; no SELECT *, and no 2026 rows."""
    if view_columns is not None:
        missing=set(REQUIRED_COLUMNS)-set(view_columns)
        if missing: raise RuntimeError('[NFL-CHALLENGER-V1-HOLD] SOURCE_COLUMNS_MISSING '+str(sorted(missing)))
    quoted=', '.join(f'`{c}`' for c in REQUIRED_COLUMNS)
    return (f'SELECT {quoted} FROM `{VIEW}` '
            "WHERE Season BETWEEN 2017 AND 2025 AND Season_Stage IN ('REGULAR','POSTSEASON') "
            'AND Historical_Core_Eligible = 1 '
            'ORDER BY Season, Game_Date, Source_Name, Source_Game_ID, Team_Norm')


def _numeric(v):
    try:
        a=float(v)
        return a if math.isfinite(a) else math.nan
    except (ValueError, TypeError):return math.nan


def _label_status(margin):
    if not math.isfinite(margin):return 'MISSING'
    return 'WIN' if margin > 1e-7 else 'LOSS' if margin < -1e-7 else 'PUSH'


def _implied_american(odds):
    v=_numeric(odds)
    if not math.isfinite(v) or v==0:return math.nan
    return 100/(100+v) if v>0 else -v/(100-v)


def _paired_game(p:pd.DataFrame)->dict:
    """Reject malformed game grain, choose exactly one deterministic oriented side."""
    if len(p)!=2:raise ValueError('BAD_SIDE_COUNT')
    teams=[str(v or '').strip().lower() for v in p['Team_Norm']]
    if len(set(teams))!=2 or not all(teams):raise ValueError('BAD_TEAM_PAIR')
    opps=[str(v or '').strip().lower() for v in p['Opponent_Norm']]
    if sorted(teams)!=sorted(opps) or any(t==o for t,o in zip(teams,opps)):raise ValueError('BAD_OPPONENT_PAIR')
    for field in ('Team_Score','Opponent_Score'):
        if p[field].isna().any():raise ValueError('MISSING_SCORE')
    r0,r1=p.iloc[0],p.iloc[1]
    if (_numeric(r0.Team_Score)!=_numeric(r1.Opponent_Score) or
        _numeric(r0.Opponent_Score)!=_numeric(r1.Team_Score)):
        raise ValueError('NONRECIPROCAL_SCORE')
    for c in ('Season_Stage','Game_Date'):
        if len(p[c].astype(str).unique())!=1:raise ValueError('CONFLICTING_'+c)
    homes=pd.to_numeric(p['Is_Home'],errors='coerce').fillna(-1).to_list()
    aways=pd.to_numeric(p['Is_Away'],errors='coerce').fillna(-1).to_list()
    neutral=pd.to_numeric(p['Is_Neutral'],errors='coerce').fillna(-1).to_list()
    is_reg_venue=sorted(zip(homes,aways,neutral))==[(0,1,0),(1,0,0)]
    is_neutral_venue=homes==[0,0] and aways==[0,0] and neutral==[1,1]
    if not (is_reg_venue or is_neutral_venue):raise ValueError('BAD_VENUE_PAIR')
    if is_reg_venue:
        picked=p[p['Is_Home'].astype(int)==1].iloc[0]
    else:
        picked=p.sort_values('Team_Norm',kind='mergesort').iloc[0]
    other=p.loc[p.index!=picked.name].iloc[0]
    # Opposite closing spreads must be reciprocal if both available.
    s1,s2=_numeric(picked.Spread_Value),_numeric(other.Spread_Value)
    if math.isfinite(s1) and math.isfinite(s2) and abs(s1+s2)>1e-5:
        raise ValueError('NONRECIPROCAL_CLOSE_SPREAD')
    t1,t2=_numeric(picked.Current_Total),_numeric(other.Current_Total)
    if math.isfinite(t1) and math.isfinite(t2) and abs(t1-t2)>1e-5:
        raise ValueError('CONFLICTING_CLOSE_TOTAL')
    return {'selected':picked,'other':other,'neutral':is_neutral_venue}


def physical_games(df:pd.DataFrame)->pd.DataFrame:
    """Build one physical game, precompute retrospective targets and diagnostic odds.

    The row selected is Home if available, or lexically first team for both-neutral.
    For each game the result labels are fixed per-oriented side, so a two-sided
    duplicated matchup CANNOT inflate validation metrics.
    """
    missing=set(REQUIRED_COLUMNS)-set(df.columns)
    if missing:raise RuntimeError('MISSING_SOURCE_COLUMNS '+str(sorted(missing)))
    d=df.copy()
    d['Season']=pd.to_numeric(d.Season,errors='coerce')
    if d.Season.isna().any() or d.Season.mod(1).ne(0).any():raise RuntimeError('INVALID_SEASON')
    if not d.Season.isin(EXPERIMENT_SEASONS).all():raise RuntimeError('SEALED_YEAR_OR_UNEXPECTED_SEASON_IN_QUERY')
    if not d.Season_Stage.isin(['REGULAR','POSTSEASON']).all():raise RuntimeError('INELIGIBLE_STAGE')
    if not pd.to_numeric(d.Historical_Core_Eligible,errors='coerce').eq(1).all():raise RuntimeError('INELIGIBLE_ROW')
    if d[list(IDENTITY)].isna().any().any():raise RuntimeError('NULL_GAME_KEY')
    records=[]; anomalies=[]
    for identity,p in d.groupby(list(IDENTITY),sort=False,dropna=False):
        try: pair=_paired_game(p)
        except ValueError as e:
            anomalies.append({'game_id':'|'.join(map(str,identity)),'reason':str(e)})
            continue
        chosen=pair['selected']; other=pair['other']
        rec=chosen.to_dict()
        rec['physical_game_id']='|'.join(map(str,identity))
        rec['orientation']='NEUTRAL_CANONICAL' if pair['neutral'] else 'HOME'
        rec['opposing_moneyline']=other.ML_Odds
        ts,os=_numeric(chosen.Team_Score),_numeric(chosen.Opponent_Score)
        margin=ts-os
        rec['H2H_label']=float(margin>0) if margin!=0 else np.nan
        spread=_numeric(chosen.Spread_Value)
        total=_numeric(chosen.Current_Total)
        spread_result=_label_status(margin+spread) if math.isfinite(spread) else 'MISSING'
        total_result=_label_status(ts+os-total) if math.isfinite(total) else 'MISSING'
        rec['SPREADS_label']=1.0 if spread_result=='WIN' else 0.0 if spread_result=='LOSS' else np.nan
        rec['TOTALS_label']=1.0 if total_result=='WIN' else 0.0 if total_result=='LOSS' else np.nan
        rec['SPREADS_status']=spread_result;rec['TOTALS_status']=total_result
        source_spread=str(chosen.ATS_Result_Close or '').strip().upper()
        source_total=str(chosen.Total_Result_Close or '').strip().upper()
        if (source_spread and spread_result!='MISSING' and source_spread!=spread_result):
            anomalies.append({'game_id':rec['physical_game_id'],'reason':'SOURCE_SPREAD_LABEL_CONFLICT'})
        if (source_total and total_result!='MISSING' and source_total!=('OVER' if total_result=='WIN' else 'UNDER' if total_result=='LOSS' else 'PUSH')):
            anomalies.append({'game_id':rec['physical_game_id'],'reason':'SOURCE_TOTAL_LABEL_CONFLICT'})
        p1,p2=_implied_american(chosen.ML_Odds),_implied_american(other.ML_Odds)
        rec['H2H_close_novig_reference']=(p1/(p1+p2)) if math.isfinite(p1) and math.isfinite(p2) and (p1+p2)>0 else np.nan
        records.append(rec)
    if anomalies:raise RuntimeError('[NFL-CHALLENGER-V1-HOLD] GAME_GRAIN_OR_LABEL_ANOMALIES '+json.dumps(anomalies[:12]))
    if not records:raise RuntimeError('NO_VALID_PHYSICAL_GAMES')
    out=pd.DataFrame.from_records(records).sort_values(['Season','Game_Date','physical_game_id']).reset_index(drop=True)
    if out['physical_game_id'].duplicated().any():raise RuntimeError('DUPLICATE_PHYSICAL_GAME_ID')
    # First sourced season lacks a fully observed prior NFL season in this pipeline.
    prior_cols=[x for x in _flatten_manifest() if x.startswith('Prior_Season_') or x.startswith('Opp_Prior_Season_') or x=='Prior_Season_WinPct_Diff']
    out.loc[out.Season.eq(2017),prior_cols]=np.nan
    return out


def _feature_matrix(train,valid,cols):
    # Train-only medians. Never fit preprocessing on validation or on 2026.
    # BigQuery/pandas may materialize integer features as nullable Int64.  A
    # train-fold median can legitimately be fractional (for example 107.5),
    # and pandas Int64 refuses that value during fillna.  Normalize predictors
    # to float64 before computing/applying medians so missing-value imputation
    # is dtype-safe while preserving the same train-only preprocessing contract.
    a=(train.loc[:,cols]
       .apply(pd.to_numeric,errors='coerce')
       .astype(np.float64)
       .replace([np.inf,-np.inf],np.nan))
    b=(valid.loc[:,cols]
       .apply(pd.to_numeric,errors='coerce')
       .astype(np.float64)
       .replace([np.inf,-np.inf],np.nan))
    med=a.median(axis=0).fillna(0.0).astype(np.float64)
    A=a.fillna(med).to_numpy(dtype=np.float64)
    B=b.fillna(med).to_numpy(dtype=np.float64)
    return A,B


def _model_predict(family,market,train,valid,y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    if family=='HIST_RATE':return np.full(len(valid),(float(y.sum())+1)/(len(y)+2),dtype=float)
    cols=COMPACT_FEATURES[market] if family=='MATCHUP_COMPACT' else _flatten_manifest()
    A,B=_feature_matrix(train,valid,list(cols))
    if family in ('MATCHUP_COMPACT','REGULARIZED_FULL'):
        sc=StandardScaler().fit(A)
        C=0.35 if family=='MATCHUP_COMPACT' else 0.06
        model=LogisticRegression(C=C,max_iter=900,solver='lbfgs',random_state=17)
        model.fit(sc.transform(A),y)
        p=model.predict_proba(sc.transform(B))[:,list(model.classes_).index(1.)]
    elif family=='TREE_SHALLOW':
        model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,
                min_samples_leaf=65,learning_rate=0.04,l2_regularization=15,
                early_stopping=False,random_state=17)
        model.fit(A,y)
        p=model.predict_proba(B)[:,list(model.classes_).index(1.)]
    else:raise ValueError('UNKNOWN_MODEL_FAMILY')
    return np.clip(np.asarray(p,dtype=float),0.001,0.999)


def _scores(y,p):
    from sklearn.metrics import roc_auc_score
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    if len(y)==0 or len(y)!=len(p) or not np.isfinite(p).all():raise RuntimeError('BAD_METRIC_INPUT')
    p=np.clip(p,0.001,0.999)
    loss=-(y*np.log(p)+(1-y)*np.log(1-p))
    brier=(p-y)**2
    bins=np.minimum((p*5).astype(int),4)
    ece=sum(abs(float(np.mean(y[bins==i]))-float(np.mean(p[bins==i])))*int(np.sum(bins==i))/len(y)
            for i in range(5) if np.any(bins==i))
    return {'n':len(y),'positives':int(sum(y)),'log_loss':round(float(loss.mean()),6),
            'brier':round(float(brier.mean()),6),'ece5':round(float(ece),6),
            'auc':round(float(roc_auc_score(y,p)),6) if len(np.unique(y))==2 else None}


def _paired_ci(y,base,challenge,seasons,seed=17):
    """Descriptive paired resampling within validation seasons, per physical game."""
    y=np.asarray(y,dtype=float);base=np.clip(np.asarray(base,dtype=float),.001,.999)
    p=np.clip(np.asarray(challenge,dtype=float),.001,.999)
    d=(-(y*np.log(p)+(1-y)*np.log(1-p)))-(-(y*np.log(base)+(1-y)*np.log(1-base)))
    if not len(d):return None
    rng=np.random.default_rng(seed)
    yrs=np.asarray(seasons)
    if len(yrs)!=len(d):raise RuntimeError('INVALID_BOOTSTRAP_GAME_SEASONS')
    groups=[np.flatnonzero(yrs==yr) for yr in sorted(set(yrs))]
    bs=[]
    for _ in range(400):
        samples=np.concatenate([ids[rng.integers(0,len(ids),len(ids))] for ids in groups])
        bs.append(float(np.mean(d[samples])))
    return {'delta_log_loss':round(float(np.mean(d)),6),
            'paired_game_bootstrap95_fixed_season_mix':[round(float(v),6) for v in np.quantile(bs,[.025,.975])]}


def tournament(games:pd.DataFrame,log_func=print)->dict:
    """Run all 2021-25 chronological folds; 2026 is rejected even if caller includes it."""
    if games.Season.eq(SEALED_YEAR).any() or not games.Season.isin(EXPERIMENT_SEASONS).all():
        raise RuntimeError('[NFL-CHALLENGER-V1-HOLD] SEALED_YEAR_PRESENT')
    if games.physical_game_id.duplicated().any():raise RuntimeError('DUPLICATE_GAME')
    counts=games.groupby('Season').size().to_dict()
    for y in EXPERIMENT_SEASONS:
        if counts.get(y,0)<200:raise RuntimeError(f'[NFL-CHALLENGER-V1-HOLD] INCOMPLETE_SEASON_{y} count={counts.get(y,0)}')
    report={'status':'EXPERIMENT_ONLY','source_tag':SOURCE_TAG,'model_publication':False,
       'production_authority':0,'incumbent_comparison_verified':False,
       'sealed_2026_touched':False,'full_history_games':len(games),
       'season_game_counts':{str(k):int(v) for k,v in counts.items()},
       'feature_count':len(_flatten_manifest()),'folds':{},'market_summary':{},
       'scope':'2017-2025; rolling fits; retrospective close-based spread/totals labels only',
       'not_claimed':['historical executable ROI','CLV','incumbent outperformance','live as-of feature proof']}
    for market in MARKETS:
        label=market+'_label'
        score_records={f:[] for f in FAMILIES}
        out_of_fold={f:{'y':[],'p':[],'seasons':[]} for f in FAMILIES}
        fold_rows=[]
        for val_year in VALIDATION_SEASONS:
            train=games.loc[games.Season.lt(val_year)&games[label].notna()].copy()
            val=games.loc[games.Season.eq(val_year)&games[label].notna()].copy()
            if len(train)<650 or len(val)<180:
                raise RuntimeError(f'[NFL-CHALLENGER-V1-HOLD] LABEL_COVERAGE {market} {val_year} train={len(train)} val={len(val)}')
            y=train[label].to_numpy(float);v=val[label].to_numpy(float)
            if len(np.unique(y))!=2:raise RuntimeError('TRAIN_HAS_ONE_CLASS')
            row={'validate_season':val_year,'train_games':len(train),'validate_games':len(val),
                 'excluded_validation_pushes_or_missing':int(sum(games.Season.eq(val_year)))-len(val),
                 'families':{}}
            for family in FAMILIES:
                pred=_model_predict(family,market,train,val,y)
                metrics=_scores(v,pred)
                row['families'][family]=metrics
                score_records[family].append({'season':val_year,**metrics})
                out_of_fold[family]['y'].extend(v.tolist())
                out_of_fold[family]['p'].extend(pred.tolist())
                out_of_fold[family]['seasons'].extend([val_year]*len(v))
            if market=='H2H':
                ref=pd.to_numeric(val.H2H_close_novig_reference,errors='coerce').to_numpy(float)
                mask=np.isfinite(ref)
                row['retrospective_close_novig_reference']=_scores(v[mask],ref[mask]) if sum(mask)>=150 else {'status':'LOW_REFERENCE_COVERAGE','n':int(sum(mask))}
            fold_rows.append(row)
            log_func('[NFL-CHALLENGER-V1-FOLD] '+json.dumps({'market':market,**row},sort_keys=True,default=str))
        report['folds'][market]=fold_rows
        summary={}
        b=out_of_fold['HIST_RATE']
        for family in FAMILIES:
            o=out_of_fold[family];score=_scores(o['y'],o['p'])
            ds=[r['log_loss']-b_rec['log_loss'] for r,b_rec in zip(score_records[family],score_records['HIST_RATE'])]
            ci=_paired_ci(o['y'],b['p'],o['p'],o['seasons'])
            summary[family]={**score,'seasons_better_logloss_than_hist_rate':sum(x<-1e-8 for x in ds),
                   'seasons_compared':len(ds),'season_logloss_deltas_vs_hist_rate':[round(float(x),6) for x in ds],
                   'worst_season_delta':round(max(ds),6),'paired_difference':ci}
        report['market_summary'][market]=summary
        log_func('[NFL-CHALLENGER-V1-MARKET] '+json.dumps({'market':market,'families':summary},sort_keys=True,default=str))
    report['status']='RESEARCH_RESULTS_ONLY'
    log_func('[NFL-CHALLENGER-V1-CONTRACT] status=RESEARCH_RESULTS_ONLY publication=FALSE production_authority=0 '
             'incumbent_provenance=UNVERIFIED year2026=SEALED ncaaf=UNCHANGED legacy_nfl=UNCHANGED '
             'market_roi_clv=NOT_VERIFIED')
    return report


def run_nfl_challenger_v1(*,bq_client=None,audit_report=None,log_func=print):
    if not isinstance(audit_report,dict) or audit_report.get('status')!='READY_FOR_OFFLINE_CHALLENGER_SANDBOX':
        raise RuntimeError('[NFL-CHALLENGER-V1-HOLD] V1_3_AUDIT_NOT_GREEN')
    feature_audit=audit_report.get('feature_leakage_review') or {}
    if (feature_audit.get('status')!='READY_FOR_OFFLINE_CHALLENGER_SANDBOX' or
        feature_audit.get('manifest',{}).get('status')!='EXPLICIT_ALLOWLIST_PASS' or
        feature_audit.get('recompute',{}).get('status')!='PRIOR_RECOMPUTE_PASS' or
        feature_audit.get('view_contract',{}).get('status')!='VIEW_CONTRACT_PASS' or
        feature_audit.get('fold_blueprint',{}).get('status')!='SEASON_FORWARD_BLUEPRINT_PASS'):
        raise RuntimeError('[NFL-CHALLENGER-V1-HOLD] FEATURE_AUDIT_CONTRACT_NOT_GREEN')
    from google.cloud import bigquery
    bq=bq_client or bigquery.Client(project='sharplogger')
    live_columns={f.name for f in bq.get_table(VIEW).schema}
    query=build_readonly_query(live_columns)
    log_func(f'[NFL-CHALLENGER-V1-PREFLIGHT] status=PASS tag={SOURCE_TAG} publication=FALSE '
             f'features={len(_flatten_manifest())} season_end=2025 year2026=SEALED')
    df=bq.query(query,job_config=bigquery.QueryJobConfig(use_query_cache=True,
                   maximum_bytes_billed=20*1024**3)).to_dataframe()
    games=physical_games(df)
    log_func('[NFL-CHALLENGER-V1-GRAIN] '+json.dumps({'source_rows':len(df),
      'physical_games':len(games),'seasons':{str(k):int(v) for k,v in games.groupby('Season').size().items()},
      'spread_pushes':int(sum(games.SPREADS_status.eq('PUSH'))),
      'totals_pushes':int(sum(games.TOTALS_status.eq('PUSH'))),
      'h2h_ties':int(sum(games.H2H_label.isna()))},sort_keys=True))
    return tournament(games,log_func=log_func)
