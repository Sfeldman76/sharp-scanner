"""Independent NCAAF full-history margin challengers (research only).

Source: *only* the published, pregame-gated CFBD 2013-2025 historical view
via the archive opt-in reader in ncaaf_core_challenger_v2. No 2026 rows,
market predictors, downstream CORE bundle/Bet Authority mutation, or research
feature discovery using 2024/25 outcomes.

2017-2023: expanding-origin discovery / selection using prior years only.
2024/25: retrospective, repeatedly inspected diagnostics (not pristine holdout).
2026: excluded from all queries and model fitting.

No claim of full pregame certification merely from publisher's column names.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import re
from typing import Any
import numpy as np
import pandas as pd

VERSION = 'NCAAF_FULL_HISTORY_CHALLENGER_V3_20261019'
SOURCE = 'sharplogger.sharp_data.ncaaf_cfbd_archive_research_v1'
SELECTION_YEARS = tuple(range(2017, 2024))
RETROSPECTIVE_YEARS = (2024, 2025)
MIN_TRAIN = 1400
MIN_EVAL = 250
MIN_JOIN_RATE = 0.40
MIN_EARLY_ROWS = 4000
SEED = 20261019

# Deliberate name-based exclusion in addition to the read-only view schema gate.
UNSAFE = re.compile(r'(?:market|spread|odds|line_move|opening|closing|clv|moneyline|'
                    r'actual|final|result|team_score|opponent_score|final_score|label|bet_|pt_|current_game|'
                    r'weather|injury|portal|roster|returning|recruit|talent)', re.I)


def _num(df: pd.DataFrame, col: str):
    return pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float) if col in df else np.full(len(df),np.nan)


def _metrics(y, pred):
    y=np.asarray(y,dtype=float); pred=np.asarray(pred,dtype=float)
    m=np.isfinite(y)&np.isfinite(pred)
    if not m.any():return {'n':0,'rmse':None,'mae':None}
    err=y[m]-pred[m]
    return {'n':int(m.sum()),'rmse':float(np.sqrt(np.mean(err**2))), 'mae':float(np.mean(np.abs(err)))}


def _paired(y, incumbent, challenger):
    y=np.asarray(y,float); incumbent=np.asarray(incumbent,float); challenger=np.asarray(challenger,float)
    ok=np.isfinite(y)&np.isfinite(incumbent)&np.isfinite(challenger)
    if not ok.any():return {'n':0,'rmse_gain':None,'mae_gain':None}
    i=_metrics(y[ok],incumbent[ok]);c=_metrics(y[ok],challenger[ok])
    return {'n':int(ok.sum()),'rmse_gain':float(i['rmse']-c['rmse']), 'mae_gain':float(i['mae']-c['mae'])}


def _timestamp():return datetime.now(timezone.utc).isoformat()


def _feature_groups(train: pd.DataFrame):
    """Unsupervised, fit-fold-only coverage gate; no target-based screening."""
    found=[c for c in train if c.startswith('CFBD_') and not UNSAFE.search(c)]
    coverage={}
    for col in found:
        arr=_num(train,col)
        good=arr[np.isfinite(arr)]
        if len(good)<max(100,int(len(train)*.12)) or len(np.unique(good))<3:
            continue
        coverage[col]=len(good)/len(train)
    # Deterministic highest-coverage, not ranked by validation outcomes.
    cols=sorted(coverage,key=lambda c:(-coverage[c],c))
    matchup=[c for c in cols if c.startswith(('CFBD_Matchup_','CFBD_Opp_')) or '_Off_' in c or '_Def_' in c]
    return {'full':cols[:260], 'matchup':matchup[:110], 'nonlinear':cols[:72],
            'observed':len(found), 'qualified':len(cols), 'coverage':coverage}


def _train_numeric(train, valid, cols, kind, *, recency=False, min_train=None):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor

    min_train=MIN_TRAIN if min_train is None else min_train
    n=len(valid)
    if not cols or len(train)<min_train:return np.full(n,np.nan)
    # Re-evaluate support separately on each training fold. No 2024/25 feature ranking.
    cols=[c for c in cols if c in train and int(np.isfinite(_num(train,c)).sum())>=max(100,int(len(train)*.12))]
    if len(cols)<3:return np.full(n,np.nan)
    xtr=train[cols].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    xva=valid[cols].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    good=np.isfinite(_num(train,'Actual_Margin'))
    if good.sum()<min_train:return np.full(n,np.nan)
    y=_num(train,'Actual_Margin')[good]
    if kind=='hgb':
        # HistGradientBoosting handles NaN directly; winsor bounds learned on fit fold.
        lo=xtr.quantile(.001);hi=xtr.quantile(.999)
        xtr=xtr.clip(lower=lo,upper=hi,axis=1)
        xva=xva.clip(lower=lo,upper=hi,axis=1)
        model=HistGradientBoostingRegressor(max_iter=140, max_leaf_nodes=12,
                learning_rate=.045,min_samples_leaf=45,l2_regularization=20.,
                early_stopping=False,random_state=SEED)
    else:
        model=Pipeline([('median',SimpleImputer(strategy='median')),
             ('robust',RobustScaler()),('ridge',Ridge(alpha=400.0))])
    kw={}
    if recency:
        ages=np.maximum(0.,float(np.nanmax(_num(train,'Season')))-_num(train,'Season')[good])
        kw['sample_weight']=np.power(.5,ages/3.)
    if kind=='hgb':
        model.fit(xtr.loc[good],y,**kw)
    else:
        model.fit(xtr.loc[good],y,**({'ridge__sample_weight':kw['sample_weight']} if kw else {}))
    return np.asarray(model.predict(xva),float)


def _program_effects(train,valid,*,min_train=None):
    """Regularized program-strength contrasts + football matchup context.

    Team identity enters only as a shrinkable +1/-1 effect. This is a separate
    structural competitor to the numeric-only and nonlinear CFBD models.
    """
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import Ridge
    from scipy.sparse import hstack,csr_matrix
    min_train=MIN_TRAIN if min_train is None else min_train
    names=('Team_Norm','Opponent_Norm')
    if len(train)<min_train or any(c not in train or c not in valid for c in names):
        return np.full(len(valid),np.nan)
    trn=train['Team_Norm'].astype('string').fillna('').str.lower().str.strip()
    opn=train['Opponent_Norm'].astype('string').fillna('').str.lower().str.strip()
    vn=valid['Team_Norm'].astype('string').fillna('').str.lower().str.strip()
    vo=valid['Opponent_Norm'].astype('string').fillna('').str.lower().str.strip()
    if not (trn.ne('').mean()>.85 and opn.ne('').mean()>.85):
        return np.full(len(valid),np.nan)
    def rows(a,b):
        return [({('program:'+str(x)):1.0,('program:'+str(y)):-1.0}
                 if x!=y and x and y else {}) for x,y in zip(a,b)]
    enc=DictVectorizer(sparse=True)
    xtr=enc.fit_transform(rows(trn,opn));xva=enc.transform(rows(vn,vo))
    cols=['CFBD_Season_Net_PPA_Prior','CFBD_Recent3_Net_PPA_Prior',
          'CFBD_Matchup_Net_PPA_Adv','CFBD_Opp_Season_Net_PPA_Prior']
    usable=[c for c in cols if c in train and c in valid and np.isfinite(_num(train,c)).sum()>=max(100,int(len(train)*.12))]
    if usable:
        a=train[usable].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
        b=valid[usable].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
        med=a.median().fillna(0)
        a=a.fillna(med);b=b.fillna(med)
        scale=(a.quantile(.75)-a.quantile(.25)).replace(0,1).fillna(1)
        xtr=hstack([xtr,csr_matrix(((a-med)/scale).to_numpy(dtype=float))],format='csr')
        xva=hstack([xva,csr_matrix(((b-med)/scale).to_numpy(dtype=float))],format='csr')
    yy=_num(train,'Actual_Margin');good=np.isfinite(yy)
    if good.sum()<min_train:return np.full(len(valid),np.nan)
    model=Ridge(alpha=90.,solver='lsqr',fit_intercept=True)
    model.fit(xtr[good],yy[good])
    return np.asarray(model.predict(xva),float)


def _variants(train,valid,*,min_train=None,feature_pool=None):
    min_train=MIN_TRAIN if min_train is None else min_train
    pool=_feature_groups(train) if feature_pool is None else feature_pool
    result={
        'CFBD_ALL_RIDGE':_train_numeric(train,valid,pool['full'],'ridge',min_train=min_train),
        'CFBD_RECENCY_RIDGE':_train_numeric(train,valid,pool['full'],'ridge',recency=True,min_train=min_train),
        'CFBD_MATCHUP_RIDGE':_train_numeric(train,valid,pool['matchup'],'ridge',min_train=min_train),
        'CFBD_NONLINEAR_HGB':_train_numeric(train,valid,pool['nonlinear'],'hgb',min_train=min_train),
        'PROGRAM_STRENGTH_RIDGE':_program_effects(train,valid,min_train=min_train),
    }
    return result,pool


def _audit_frame(g, archive_diag):
    if g.empty:raise RuntimeError('V3_NO_GAME_ROWS')
    required=('Season','Actual_Margin','Game_Date','CFBD_Season_Net_PPA_Prior')
    missing=[c for c in required if c not in g.columns]
    if missing:raise RuntimeError('V3_REQUIRED_FIELDS_MISSING:'+','.join(missing))
    sy=_num(g,'Season')
    if np.nanmax(sy)>=2026:raise RuntimeError('V3_2026_OUTCOME_SEAL_FAILED')
    early=(sy>=2013)&(sy<=2021)
    if int(early.sum())<MIN_EARLY_ROWS:raise RuntimeError(f'V3_EARLY_HISTORY_MISSING rows={int(early.sum())}')
    matched=np.isfinite(_num(g,'CFBD_Season_Net_PPA_Prior'))
    early_rate=float(matched[early].mean()) if early.any() else 0.
    if early_rate<MIN_JOIN_RATE:raise RuntimeError(f'V3_ARCHIVE_JOIN_INADEQUATE early_rate={early_rate:.3f}')
    counts={str(year):{'rows':int((sy==year).sum()),
                       'advanced_rows':int((matched&(sy==year)).sum())}
            for year in range(2013,2026)}
    source=(archive_diag or {}).get('table')
    if source!=SOURCE:raise RuntimeError(f'V3_ARCHIVE_SOURCE_MISMATCH source={source!r}')
    # Only one game-target per physical game (no double weighting via team sides).
    if 'Source_Game_ID' in g:
        ids=g['Source_Game_ID'].astype('string').fillna('').str.strip()
        meaningful=~ids.str.lower().isin(('','nan','none','<na>'))
        keys=g.loc[meaningful,'Season'].astype(str)+'|'+ids[meaningful]
        if keys.duplicated().any():raise RuntimeError('V3_DUPLICATED_PHYSICAL_GAME_ORIENTATION')
    return {'source':source,'season_counts':counts,'early_advanced_join_rate':early_rate,
            'archive_pre_kickoff_certification':'INCOMPLETE_REQUIRES_TIMING_AUDIT',
            'join_diagnostics':{k:v for k,v in (archive_diag or {}).items()
                                if isinstance(v,(str,int,float,bool,type(None)))}}


def _select_discovery(folds):
    names=('CFBD_ALL_RIDGE','CFBD_RECENCY_RIDGE','CFBD_MATCHUP_RIDGE','CFBD_NONLINEAR_HGB','PROGRAM_STRENGTH_RIDGE')
    ranked=[]
    for name in names:
        ok=[]; scores=[]
        for year in SELECTION_YEARS:
            row=folds.get(str(year),{}); s=row.get('models',{}).get(name,{})
            if s.get('n',0)>=MIN_EVAL and s.get('rmse') is not None:
                ok.append(year);scores.append((s['n'],s['rmse'],s['mae']))
        if len(ok)<5:
            ranked.append({'model':name,'status':'INSUFFICIENT_DISCOVERY_FOLDS','folds':ok});continue
        pooled_rmse=float(np.sqrt(sum(n*r*r for n,r,_ in scores)/sum(n for n,_,_ in scores)))
        pooled_mae=float(sum(n*a for n,_,a in scores)/sum(n for n,_,_ in scores))
        ranked.append({'model':name,'status':'ELIGIBLE_DISCOVERY','folds':ok,
                       'discovery_rmse':pooled_rmse,'discovery_mae':pooled_mae})
    eligible=sorted((r for r in ranked if r['status']=='ELIGIBLE_DISCOVERY'),
                    key=lambda v:(v['discovery_rmse'],v['discovery_mae'],v['model']))
    return (eligible[0]['model'] if eligible else None),ranked


def _year_fold(frame, year, log_func):
    sy=_num(frame,'Season')
    tr=frame.loc[(sy<year)&np.isfinite(_num(frame,'Actual_Margin'))].copy()
    va=frame.loc[(sy==year)&np.isfinite(_num(frame,'Actual_Margin'))].copy()
    if len(tr)<MIN_TRAIN or len(va)<MIN_EVAL:
        return {'status':'INSUFFICIENT_ROWS','train':len(tr),'evaluation':len(va)},{}
    y=_num(va,'Actual_Margin')
    model_predictions,feature_pool=_variants(tr,va)
    # Market-blind, minimal pregame CFBD benchmark. This is NOT frozen Production V1.
    football_control=_train_numeric(tr,va,
        ['CFBD_Season_Net_PPA_Prior','CFBD_Recent3_Net_PPA_Prior',
         'CFBD_Matchup_Net_PPA_Adv','CFBD_Opp_Season_Net_PPA_Prior'],
        'ridge',min_train=MIN_TRAIN)
    baseline={'minimal_cfbd_margin_control':_metrics(y,football_control)}
    records={'status':'PASS','train':len(tr),'evaluation':len(va),
             'train_start':int(np.nanmin(_num(tr,'Season'))),'train_end':int(np.nanmax(_num(tr,'Season'))),
             'baseline':baseline,'feature_pool':{k:feature_pool[k] for k in ('observed','qualified')},
             'models':{},'paired_to_football_control':{}}
    for name,p in model_predictions.items():
        records['models'][name]=_metrics(y,p)
        records['paired_to_football_control'][name]=_paired(y,football_control,p)
    # Market comparisons require actual authentic source; no closing-line proxy.
    op=_num(va,'Market_Open_Margin'); raw=_num(va,'Consensus_Open_Spread')
    opened=(np.isfinite(op)&np.isfinite(raw)&np.isclose(op,-raw,atol=1e-5,rtol=0)
            &(np.abs(op)<=80)&(_num(va,'Season')>=2021))
    if opened.any():
        records['authentic_opening_market']={'status':'DIAGNOSTIC_ONLY','n':int(opened.sum()),
            'opening':_metrics(y[opened],op[opened]),
            'models':{k:_metrics(y[opened],v[opened]) for k,v in model_predictions.items()}}
    log_func('[NCAAF-V3-FOLD] '+f'year={year} train={len(tr)} valid={len(va)} '
             f'model_rmse='+str({k:round(v['rmse'],3) if v['rmse'] is not None else None
                                 for k,v in records['models'].items()})+' authority=0')
    return records, {'y':y,'controls':football_control,'predictions':model_predictions}


def run_on_enriched_frame(frame:pd.DataFrame, archive_diag:dict,*,log_func=print):
    """Injected-frame entrypoint for deterministic tests; no GCP write needed."""
    g=frame.loc[_num(frame,'Season')<=2025].copy().reset_index(drop=True)
    audit=_audit_frame(g,archive_diag)
    folds={}; for_report={}
    for year in (*SELECTION_YEARS,*RETROSPECTIVE_YEARS):
        fold,arrays=_year_fold(g,year,log_func)
        folds[str(year)]=fold
        for_report[str(year)]=arrays
    selected,ranked=_select_discovery(folds)
    retrospective={}
    if selected:
        for year in RETROSPECTIVE_YEARS:
            data=for_report.get(str(year),{});p=(data.get('predictions') or {}).get(selected)
            if p is not None:
                retrospective[str(year)]=_paired(data['y'],data['controls'],p)
    # Historical ablation on 2023 only. Same 2023 evaluation games.
    ablation={'status':'UNAVAILABLE'}
    if 2023 in SELECTION_YEARS:
        train_full=g.loc[(_num(g,'Season')<2023)&np.isfinite(_num(g,'Actual_Margin'))]
        train_short=train_full.loc[_num(train_full,'Season')>=2022]
        valid=g.loc[(_num(g,'Season')==2023)&np.isfinite(_num(g,'Actual_Margin'))]
        # Short-history model needs enough for this controlled comparison; do not
        # lower the global minimum needed for the primary challenger.
        if len(train_full)>=MIN_TRAIN and len(train_short)>=400 and len(valid)>=MIN_EVAL:
            pool=_feature_groups(train_short)  # fixed from 2022-only training; same exact columns
            full,_=_variants(train_full,valid,min_train=400,feature_pool=pool)
            short,_=_variants(train_short,valid,min_train=400,feature_pool=pool)
            if selected:
                ablation={'status':'PASS','validation_year':2023,'full_train':len(train_full),
                          'short_train':len(train_short),'same_games':len(valid),
                          'candidate':selected,'feature_universe':'FROZEN_FROM_2022_TRAIN_FOR_BOTH_ARMS','2013_start':_metrics(_num(valid,'Actual_Margin'),full[selected]),
                          '2022_start':_metrics(_num(valid,'Actual_Margin'),short[selected]),
                          'paired_full_vs_short':_paired(_num(valid,'Actual_Margin'),short[selected],full[selected])}
    status='RESEARCH_COMPLETE_NO_PROMOTION' if selected else 'DISCOVERY_INCOMPLETE'
    return {'status':status,'version':VERSION,'generated_utc':_timestamp(),
            'source':SOURCE,'source_audit':audit,'chronology':{
                'discovery_folds':list(SELECTION_YEARS),'retrospective_not_pristine':list(RETROSPECTIVE_YEARS),
                '2026_label_access':False,'selection_uses_2024_2025_outcomes':False},
            'models':'CFBD_ALL_RIDGE,CFBD_RECENCY_RIDGE,CFBD_MATCHUP_RIDGE,CFBD_NONLINEAR_HGB,PROGRAM_STRENGTH_RIDGE',
            'folds':folds,'discovery_rank':ranked,'selected_pre2024':selected,
            'retrospective_selected':retrospective,'history_ablation_2023':ablation,
            'baseline_note':'minimal_cfbd_margin_control is a market-blind research control, NOT frozen Production V1; requires exact-game frozen production ledger comparison',
            'confirmation_note':'2024/2025 outcomes were previously examined repeatedly; retrospective checks, NOT sealed holdouts',
            'mandatory_next_gate':'Compare exact-game frozen Production V1 OOF ledger and prospective 2026 shadow before any authority discussion',
            'production_authority':0,'automatic_promotion':False,'production_mutated':False,
            'bet_authority_vote':False}


def run_full_history_challenger_v3(*,dashboard_module,storage_client=None,
                                   bucket_name='sharp-models',log_func=print):
    import ncaaf_core_challenger_v2 as core
    if str(os.getenv('NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED','0')).lower() not in ('1','true','yes','on'):
        raise RuntimeError('V3_REQUIRES_EXPLICIT_ARCHIVE_RESEARCH_OPT_IN')
    cache=getattr(dashboard_module,'_V1357_SPREAD_RESEARCH_CACHE',{}) or {}
    games=cache.get('games')
    if games is None or games.empty:raise RuntimeError('V3_HISTORICAL_GAME_CACHE_MISSING')
    history=games.loc[_num(games,'Season')<=2025].copy().reset_index(drop=True)
    # The CORE archive reader is the already-audited source adapter.
    joined,diag=core._attach_cfbd_core_context(dashboard_module,history,log_func=log_func)
    if str(diag.get('table'))!=SOURCE:raise RuntimeError('V3_WRONG_ARCHIVE_READ_PATH')
    result=run_on_enriched_frame(joined,diag,log_func=log_func)
    payload=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)
    if storage_client is not None:
        # Exclusive research namespace: no production training/prediction artifacts.
        blob=storage_client.bucket(bucket_name).blob('research/ncaaf/full_history_challenger_v3/current_report.json')
        blob.upload_from_string(payload,content_type='application/json')
        result['research_report_uri']=f'gs://{bucket_name}/research/ncaaf/full_history_challenger_v3/current_report.json'
    log_func(f'[NCAAF-FULL-HISTORY-V3] status={result["status"]} source={SOURCE} '
             f'selected_discovery={result["selected_pre2024"]} early_coverage={result["source_audit"]["early_advanced_join_rate"]:.3f} '
             f'year_2026_queried=FALSE production_mutation=FALSE authority=0')
    return result
