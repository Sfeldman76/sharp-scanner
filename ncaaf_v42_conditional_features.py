"""NCAAF V4.2: controlled, research-only football feature-family experiments.

NO market predictors. NO 2026 outcomes. NO frozen Production V1 changes.
All features use the verified archive *column names* ending _Prior; original
publication-time provenance remains UNVERIFIED and results cannot be promoted.

The baseline reproduces V4 NONLINEAR_STATE_HGB exactly (fixed 12 features,
fixed HGB parameters). Five small, preregistered football hypotheses are added
separately and in six football-motivated combinations. Supervised fits are
expanding-year; candidate ranking uses pre-2024 folds only, but is EXPLORATORY.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
import hashlib
from collections import OrderedDict
import numpy as np
import pandas as pd
import ncaaf_margin_tournament_v4 as v4

VERSION = 'NCAAF_V42_CONDITIONAL_FAMILIES_20261020_V1'
SOURCE = v4.SOURCE
DISCOVERY = tuple(range(2017, 2024))
SCREEN = (2017, 2018, 2019, 2020)
INTERNAL_CONFIRM = (2021, 2022, 2023)
RETROSPECTIVE = (2024, 2025)
BASE = 'NONLINEAR_STATE_HGB'
MIN_TRAIN_COVERAGE = .45
MIN_VALID_COVERAGE = .45
MIN_FEATURES = 2
STATE = ['V4_'+x for x in (
    'power_diff','off_diff','def_diff','expected_margin','expected_team_points',
    'expected_opp_points','power_uncertainty','prior_games_min','rest_diff',
    'prior_margin_recent_diff','prior_margin_season_diff','site_adv')]

# Fixed linear combinations of two-sided PRIOR measurements. A +/- sign in an
# expression merely defines orientation; the learned nonlinear coefficient is
# NOT forced to agree with that sign. Def_PPA measures PPA allowed in this data.
# This is a *matchup contrast*, not a claim of fully opponent-adjusted historical
# schedules. Actual schedule-adjusted rates require an additional audited pass.
def term(*pairs):
    return tuple((float(s), 'CFBD_'+name) for s,name in pairs)

def matchup(metric):
    return term((1,'Season_Off_'+metric+'_Prior'),
                (1,'Opp_Season_Def_'+metric+'_Prior'),
                (-1,'Opp_Season_Off_'+metric+'_Prior'),
                (-1,'Season_Def_'+metric+'_Prior'))

def direct(ours,theirs):
    return term((1,ours),(-1,'Opp_'+theirs))

FAMILIES = OrderedDict({
 'A_EFFICIENCY': OrderedDict({
     'a_ppa_matchup': matchup('PPA'),
     'a_success_matchup': matchup('Success'),
     'a_net_ppa_diff': direct('Season_Net_PPA_Prior','Season_Net_PPA_Prior'),
     'a_recent_ppa_trend_diff': term((1,'Recent3_Net_PPA_Prior'),(-1,'Season_Net_PPA_Prior'),
                                      (-1,'Opp_Recent3_Net_PPA_Prior'),(1,'Opp_Season_Net_PPA_Prior')),
 }),
 'B_PASS_RUSH_MATCHUPS': OrderedDict({
     'b_pass_ppa': matchup('Pass_PPA'),
     'b_rush_ppa': matchup('Rush_PPA'),
     'b_pass_success': matchup('Pass_Success'),
     'b_rush_success': matchup('Rush_Success'),
 }),
 'C_DRIVES_FIELD': OrderedDict({
     'c_drive_ppa': matchup('CFBD_Game_Off_PPA_Per_Drive'), # replaced below with explicit Off/Def game prior
     'c_field_position': term((1,'Season_CFBD_Game_Off_Field_Position_Prior'),
                              (-1,'Opp_Season_CFBD_Game_Off_Field_Position_Prior')),
     'c_scoring_opps': term((1,'Season_CFBD_Game_Off_Scoring_Opportunity_Rate_Prior'),
                             (1,'Opp_Season_CFBD_Game_Def_Scoring_Opportunity_Rate_Prior'),
                             (-1,'Opp_Season_CFBD_Game_Off_Scoring_Opportunity_Rate_Prior'),
                             (-1,'Season_CFBD_Game_Def_Scoring_Opportunity_Rate_Prior')),
     'c_plays_drive': term((1,'Season_CFBD_Game_Off_Plays_Per_Drive_Prior'),
                            (-1,'Opp_Season_CFBD_Game_Off_Plays_Per_Drive_Prior')),
 }),
 'D_EXPLOSIVENESS_FINISH': OrderedDict({
     'd_explosive_matchup': matchup('Explosiveness'),
     'd_pass_explosive_matchup': matchup('Pass_Explosiveness'),
     'd_explosive20': term((1,'Season_CFBD_Game_Off_Explosive20_Rate_Prior'),
                            (1,'Opp_Season_CFBD_Game_Def_Explosive20_Rate_Prior'),
                            (-1,'Opp_Season_CFBD_Game_Off_Explosive20_Rate_Prior'),
                            (-1,'Season_CFBD_Game_Def_Explosive20_Rate_Prior')),
     'd_scoring_opps_recent_trend': term((1,'Recent3_CFBD_Game_Off_Scoring_Opportunity_Rate_Prior'),
                                         (-1,'Season_CFBD_Game_Off_Scoring_Opportunity_Rate_Prior'),
                                         (-1,'Opp_Recent3_CFBD_Game_Off_Scoring_Opportunity_Rate_Prior'),
                                         (1,'Opp_Season_CFBD_Game_Off_Scoring_Opportunity_Rate_Prior')),
 }),
 'E_TRENCH_DISRUPTION': OrderedDict({
     'e_stuff_rate': matchup('StuffRate'),
     'e_power_success': matchup('PowerSuccess'),
     'e_sack_rate': term((1,'Season_CFBD_Game_Off_Sack_Rate_Prior'),
                          (1,'Opp_Season_CFBD_Game_Def_Sack_Rate_Prior'),
                          (-1,'Opp_Season_CFBD_Game_Off_Sack_Rate_Prior'),
                          (-1,'Season_CFBD_Game_Def_Sack_Rate_Prior')),
     'e_earlydown_success': term((1,'Season_CFBD_Game_Off_EarlyDown_Success_Prior'),
                                  (1,'Opp_Season_CFBD_Game_Def_EarlyDown_Success_Prior'),
                                  (-1,'Opp_Season_CFBD_Game_Off_EarlyDown_Success_Prior'),
                                  (-1,'Season_CFBD_Game_Def_EarlyDown_Success_Prior')),
 }),
})
# Actual drive columns in the archive are named CFBD_Season_CFBD_Game_*_Prior;
# matchup() is not appropriate for this family.
FAMILIES['C_DRIVES_FIELD']['c_drive_ppa'] = term(
    (1,'Season_CFBD_Game_Off_PPA_Per_Drive_Prior'),
    (1,'Opp_Season_CFBD_Game_Def_PPA_Per_Drive_Prior'),
    (-1,'Opp_Season_CFBD_Game_Off_PPA_Per_Drive_Prior'),
    (-1,'Season_CFBD_Game_Def_PPA_Per_Drive_Prior'))

# Only football-motivated combinations. ALL FIVE is a stress-test benchmark,
# never an automatically selected winner.
COMBOS = OrderedDict({
  'AB_EFF_PASS': ('A_EFFICIENCY','B_PASS_RUSH_MATCHUPS'),
  'AC_EFF_DRIVE': ('A_EFFICIENCY','C_DRIVES_FIELD'),
  'BE_PASS_TRENCH': ('B_PASS_RUSH_MATCHUPS','E_TRENCH_DISRUPTION'),
  'CD_DRIVE_EXPLOSIVE': ('C_DRIVES_FIELD','D_EXPLOSIVENESS_FINISH'),
  'ABC_EFF_PASS_DRIVE': ('A_EFFICIENCY','B_PASS_RUSH_MATCHUPS','C_DRIVES_FIELD'),
  'ABCDE_ALL_DIAGNOSTIC': tuple(FAMILIES),
})
EXPERIMENTS = OrderedDict((name,(name,)) for name in FAMILIES)
EXPERIMENTS.update(COMBOS)

# One pre-specified complementary pair of *conditional interactions* for B,
# avoiding arbitrary cross-products of many raw archive columns.
INTERACTIONS = OrderedDict({
    'b_pass_minus_rush': ('B_PASS_RUSH_MATCHUPS','b_pass_ppa','b_rush_ppa'),
})
FAMILY_OUTPUTS = {family:list(definitions) for family,definitions in FAMILIES.items()}
for output,(family,*_) in INTERACTIONS.items():
    FAMILY_OUTPUTS[family].append(output)
FAMILY_OUTPUTS['A_EFFICIENCY'].append('a_ppa_state_residual')

def _safe_numeric(frame,col):
    return pd.to_numeric(frame[col],errors='coerce').replace([np.inf,-np.inf],np.nan) if col in frame else None

def feature_frame(raw):
    """Transform ONLY pregame archive columns. Missing inputs stay NaN (no proxy)."""
    n=len(raw); feat=pd.DataFrame(index=raw.index)
    provenance={}
    for family, spec in FAMILIES.items():
        for name,terms in spec.items():
            missing=[col for _,col in terms if col not in raw]
            if missing:
                feat[name]=np.nan
                provenance[name]={'family':family,'status':'SOURCE_COLUMNS_ABSENT','missing_columns':missing}
                continue
            if any((not col.endswith('_Prior') or v4.BAD.search(col)) for _,col in terms):
                raise RuntimeError('V42_UNSAFE_FEATURE_SOURCE:'+name)
            pieces=[_safe_numeric(raw,col) for _,col in terms]
            complete=np.logical_and.reduce([x.notna().to_numpy() for x in pieces])
            values=sum(sign*x.fillna(0) for (sign,_),x in zip(terms,pieces))
            feat[name]=values.where(complete,np.nan)
            provenance[name]={'family':family,'status':'COLUMN_NAMES_PRIOR_ONLY__SOURCE_VINTAGE_UNCERTIFIED',
                              'source_columns':[col for _,col in terms],
                              'global_coverage':float(np.mean(complete)) if n else 0.0}
    # Derived interaction uses already generated prior-only signals.
    feat['b_pass_minus_rush']=feat['b_pass_ppa']-feat['b_rush_ppa']
    for name,(parent_family,l,r) in INTERACTIONS.items():
        provenance[name]={'family':parent_family,'status':'DERIVED_FROM_TWO_PRIOR_ONLY_CONTRASTS',
                          'parents':[l,r],'global_coverage':float(feat[name].notna().mean())}
    return feat,provenance

def _add_fold_train_only_efficiency_residual(train,valid):
    """Unsupervised X-on-X orthogonalization: no final-margin target used.

    Removes the component of the prior net-PPA matchup correlated with the
    existing dynamic state, to test the incremental efficiency information.
    Ridge fit is exclusively on prior training seasons; valid season never
    influences preprocessing. Does not claim historical opponent-adjustment.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    col='a_net_ppa_diff'
    output='a_ppa_state_residual'
    train[output]=np.nan;valid[output]=np.nan
    ok=train[col].notna().to_numpy()
    if ok.sum()<100:return
    imp=SimpleImputer(strategy='median',add_indicator=False)
    sc=StandardScaler()
    xtr=sc.fit_transform(imp.fit_transform(train[STATE]))
    xva=sc.transform(imp.transform(valid[STATE]))
    model=Ridge(alpha=25.0)
    model.fit(xtr[ok],train.loc[ok,col].to_numpy(dtype=float))
    train.loc[ok,output]=train.loc[ok,col].to_numpy(dtype=float)-model.predict(xtr[ok])
    vaok=valid[col].notna().to_numpy()
    valid.loc[vaok,output]=valid.loc[vaok,col].to_numpy(dtype=float)-model.predict(xva[vaok])

def _family_cols(frame, family):
    return [c for c in frame if c in FAMILY_OUTPUTS[family]]

def _coverage(xtr,xva, family):
    cols=_family_cols(xtr,family)
    passable=[]; details={}
    for c in cols:
        z=xtr[c];w=xva[c]
        a=float(z.notna().mean());b=float(w.notna().mean())
        variable=z.nunique(dropna=True)>3
        ok=(a>=MIN_TRAIN_COVERAGE and b>=MIN_VALID_COVERAGE and variable)
        details[c]={'train':round(a,4),'valid':round(b,4),'variable':bool(variable),'admitted':bool(ok)}
        if ok:passable.append(c)
    return passable,details

def _fold(enriched, state_frame, feature_data, year, *, log_func=print):
    y=v4.nums(enriched,'Actual_Margin');tot=v4.nums(enriched,'Actual_Total');years=v4.nums(enriched,'Season')
    tr=(years<year)&np.isfinite(y)&np.isfinite(tot)
    va=(years==year)&np.isfinite(y)&np.isfinite(tot)
    if tr.sum()<v4.MIN_TRAIN or va.sum()<v4.MIN_TEST:
        return {'status':'INSUFFICIENT_ROWS','train':int(tr.sum()),'valid':int(va.sum())},pd.DataFrame()
    work=pd.concat([enriched.reset_index(drop=True),state_frame.reset_index(drop=True),
                    feature_data.reset_index(drop=True)],axis=1)
    train=work.loc[tr].copy();valid=work.loc[va].copy()
    _add_fold_train_only_efficiency_residual(train,valid)
    actual=v4.nums(valid,'Actual_Margin')
    benchmark=v4._matrix_fit_predict(train,valid,STATE,v4.nums(train,'Actual_Margin'),kind='hgb')
    if v4.metric(actual,benchmark)['n']!=int(va.sum()):
        raise RuntimeError('V42_BASELINE_MISSING_PREDICTIONS')
    pred={BASE:benchmark};status={};coverage={};chosen={}
    for family in FAMILIES:
        chosen[family],coverage[family]=_coverage(train,valid,family)
    for name,families in EXPERIMENTS.items():
        eligible=all(len(chosen[f])>=MIN_FEATURES for f in families)
        if not eligible:
            pred[name]=np.full(len(valid),np.nan)
            status[name]='HOLD_COVERAGE_OR_SOURCE'
            continue
        cols=STATE+list(dict.fromkeys(c for f in families for c in chosen[f]))
        pred[name]=v4._matrix_fit_predict(train,valid,cols,v4.nums(train,'Actual_Margin'),kind='hgb')
        status[name]='PASS' if np.isfinite(pred[name]).all() else 'MODEL_INCOMPLETE'
    models={name:v4.metric(actual,p) for name,p in pred.items()}
    delta={name:v4.paired(actual,benchmark,p) for name,p in pred.items() if name!=BASE}
    records={'status':'PASS','train':int(tr.sum()),'valid':int(va.sum()),'train_end_year':year-1,
             'models':models,'paired_to_frozen_v4':delta,'experiments':status,
             'coverage':coverage,'source_feature_counts':{k:len(chosen[k]) for k in FAMILIES},
             'frozen_base_cols':STATE,'market_inputs_used':False}
    ledger=pd.DataFrame({'Season':int(year),'Game_Date':valid['Game_Date'].astype(str).to_numpy(),
        'Game_ID':valid['Source_Game_ID'].astype(str).to_numpy() if 'Source_Game_ID' in valid else np.arange(len(valid)).astype(str),
        'Team':valid['Team_Norm'].astype(str).to_numpy(),'Opponent':valid['Opponent_Norm'].astype(str).to_numpy(),
        'Actual_Margin':actual})
    for name,p in pred.items():ledger[name]=p
    log_func('[NCAAF-V42-FOLD] year={} n={} baseline_rmse={:.4f} pass={} holds={}'.format(year,len(ledger),
             models[BASE]['rmse'],sum(x=='PASS' for x in status.values()),
             [n for n,s in status.items() if s!='PASS']))
    return records,ledger

def _pool(rows, names, years):
    cut=rows.loc[rows['Season'].isin(years)]
    return {name:v4.metric(v4.nums(cut,'Actual_Margin'),v4.nums(cut,name)) for name in names}

def _rank(folds,names,years):
    scores=[]
    for name in names:
        metrics=[folds[str(year)]['models'].get(name,{'n':0,'rmse':None}) for year in years if folds[str(year)]['status']=='PASS']
        good=[m for m in metrics if m['n']>=v4.MIN_TEST and m['rmse'] is not None]
        n=sum(m['n'] for m in good)
        scores.append({'model':name,'folds':len(good),'n':n,
                       'pooled_rmse':float(np.sqrt(sum(m['n']*m['rmse']**2 for m in good)/n)) if len(good)==len(years) and n else None,
                       'status':'COMPLETE' if years and len(good)==len(years) else 'INCOMPLETE_COVERAGE'})
    return sorted(scores,key=lambda x:(x['pooled_rmse'] is None,x['pooled_rmse'] if x['pooled_rmse'] is not None else 1e5,x['model']))

def run_on_frame(frame,archive_diag,*,v4_reference=None,log_func=print,years=None):
    """Pure offline research API. Real runs require V4 reference reconciliation."""
    raw=frame.reset_index(drop=True).copy()
    input_audit=v4.input_audit(raw,archive_diag) # enforce 2013–2025 and 2026 seal
    # Real source-vintage still pending; no success status implies a certified source.
    state=v4.dynamic_features(raw,gain=.13,retain=.65)
    engineered,provenance=feature_frame(raw)
    years=tuple(years) if years is not None else DISCOVERY+RETROSPECTIVE
    folds={};ledgers=[]
    for year in years:
        fold,ledger=_fold(raw,state,engineered,year,log_func=log_func)
        folds[str(year)]=fold
        if len(ledger):ledgers.append(ledger)
    if not ledgers:raise RuntimeError('V42_NO_VALID_YEAR_FOLDS')
    all_ledger=pd.concat(ledgers,ignore_index=True)
    # Validate numerical identity of the V4 champion on all published folds.
    parity={}
    if v4_reference is not None:
        if v4_reference.get('selected_pre2024')!=BASE:
            raise RuntimeError('V42_EXPECTED_V4_CHAMPION_MISMATCH')
        for year in years:
            old=v4_reference.get('folds',{}).get(str(year),{})
            if old.get('status')!='PASS':continue
            lhs=folds[str(year)]['models'][BASE];rhs=old['models'][BASE]
            if lhs['n']!=rhs['n'] or abs(lhs['rmse']-rhs['rmse'])>1e-5 or abs(lhs['mae']-rhs['mae'])>1e-5:
                raise RuntimeError(f'V42_V4_BASELINE_PARITY_FAILED_{year}:new={lhs},old={rhs}')
            parity[str(year)]='EXACT_REPRODUCTION'
    ranks={
      'screen_2017_2020':_rank(folds,[BASE]+list(EXPERIMENTS),[x for x in SCREEN if x in years]),
      'internal_confirmation_2021_2023':_rank(folds,[BASE]+list(EXPERIMENTS),[x for x in INTERNAL_CONFIRM if x in years]),
      'all_discovery_2017_2023':_rank(folds,[BASE]+list(EXPERIMENTS),[x for x in DISCOVERY if x in years]),
    }
    retrospect=_pool(all_ledger,[BASE]+list(EXPERIMENTS),RETROSPECTIVE)
    paired_boot={}
    if set(RETROSPECTIVE).issubset(set(years)):
        for name in EXPERIMENTS:
            paired_boot[name]=v4._paired_cluster_boot(all_ledger,name,BASE,boot=350)
    # A derived feature cannot be included in an audit unless its inputs existed.
    spec = {'state':STATE,'families':FAMILIES,'derived':INTERACTIONS,'combos':COMBOS,
            'hgb_model_version':v4.VERSION,'coverage_train':MIN_TRAIN_COVERAGE,
            'coverage_valid':MIN_VALID_COVERAGE,'min_features':MIN_FEATURES}
    spec_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    report={'version':VERSION,'status':'EXPLORATORY_RESEARCH__SOURCE_VINTAGE_HOLD',
      'experiment_spec_sha256':spec_sha256,'parent_v4_version':v4.VERSION,
      'generated_utc':datetime.now(timezone.utc).isoformat(),
      'source':SOURCE,'audit':input_audit,'frozen_champion':BASE,
      'folds':folds,'ranking':ranks,'retrospective_pooled':retrospect,
      'retrospective_paired_bootstrap':paired_boot,
      'source_columns':{**provenance,'a_ppa_state_residual':{'family':'A_EFFICIENCY','status':'TRAIN_ONLY_X_ON_X_RESIDUALIZATION','parents':['a_net_ppa_diff']+STATE}},'feature_families':FAMILY_OUTPUTS,
      'combinations':{k:list(v) for k,v in COMBOS.items()},
      'parity_to_published_v4':parity,'baseline_feature_count':len(STATE),
      'supplemental_features_requested':len(provenance),
      'market_data_used_for_training':False,'year_2026_archive_queried':False,
      'year_2026_outcomes_used_for_training':False,
      'historical_dashboard_cache_may_include_2026_before_filter':True,
      'source_vintage_certified':False,'point_in_time_certification':'HOLD_PENDING_RAW_PUBLICATION_TIMESTAMPS',
      'candidate_status':'NO_AUTOMATIC_ADMISSION','selected_new_champion':None,
      'not_a_fresh_holdout':True,'production_mutated':False,'bet_authority':0,
      'limitations':['2017-2025 outcomes have been previously inspected; findings exploratory',
        'Pregame-named columns not independently certified from raw publication timestamps',
        'Matchup contrasts are not fully opponent-adjusted season statistics',
        'Simultaneous family comparisons can create multiple-testing selection bias',
        'No promotion or inference model included; frozen production untouched']}
    log_func('[NCAAF-V42-DONE] folds={} experimental_families={} combos={} '
             'status={} authority=0'.format(len(folds),len(FAMILIES),len(COMBOS),report['status']))
    return report,all_ledger

def run_cloud_research(*,dashboard_module,storage_client,bucket_name='sharp-models',log_func=print):
    if str(os.getenv('NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED','0')).lower() not in ('1','true','yes','on'):
        raise RuntimeError('V42_REQUIRES_EXPLICIT_ARCHIVE_RESEARCH_OPT_IN')
    import ncaaf_core_challenger_v2 as core
    cache=getattr(dashboard_module,'_V1357_SPREAD_RESEARCH_CACHE',{}) or {}
    games=cache.get('games')
    if games is None or games.empty:raise RuntimeError('V42_HISTORICAL_GAME_CACHE_MISSING')
    history=games.loc[v4.nums(games,'Season')<=2025].copy().reset_index(drop=True)
    enriched,diag=core._attach_cfbd_core_context(dashboard_module,history,log_func=log_func)
    bucket=storage_client.bucket(bucket_name)
    ref_blob=bucket.blob('research/ncaaf/margin_tournament_v4/current_report.json')
    try:reference=json.loads(ref_blob.download_as_text())
    except Exception as exc:raise RuntimeError('V42_MISSING_V4_REFERENCE_REPORT') from exc
    report,ledger=run_on_frame(enriched,diag,v4_reference=reference,log_func=log_func)
    for y in DISCOVERY+RETROSPECTIVE:
        if report['folds'][str(y)]['status']!='PASS' or str(y) not in report['parity_to_published_v4']:
            raise RuntimeError(f'V42_INCOMPLETE_OR_UNMATCHED_FOLD_{y}')
    runid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    base=f'research/ncaaf/conditional_features_v42/runs/{runid}'
    payload=json.dumps(report,sort_keys=True,indent=2,allow_nan=False,default=str)
    # Research-only namespace. Versioned run, then convenience latest pointer.
    bucket.blob(f'{base}/report.json').upload_from_string(payload,content_type='application/json')
    bucket.blob(f'{base}/game_predictions.csv').upload_from_string(ledger.to_csv(index=False),content_type='text/csv')
    bucket.blob('research/ncaaf/conditional_features_v42/latest_report.json').upload_from_string(payload,content_type='application/json')
    report['report_uri']=f'gs://{bucket_name}/{base}/report.json'
    report['predictions_uri']=f'gs://{bucket_name}/{base}/game_predictions.csv'
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description='V4.2 research on an enriched pregame historical CSV')
    p.add_argument('--enriched-csv',required=True)
    p.add_argument('--reference-v4',required=True)
    p.add_argument('--out-dir',required=True)
    a=p.parse_args()
    os.makedirs(a.out_dir,exist_ok=True)
    frame=pd.read_csv(a.enriched_csv,low_memory=False)
    with open(a.reference_v4) as f: ref=json.load(f)
    report,ledger=run_on_frame(frame,{'table':SOURCE},v4_reference=ref)
    with open(os.path.join(a.out_dir,'report.json'),'w') as f:json.dump(report,f,indent=2,allow_nan=False)
    ledger.to_csv(os.path.join(a.out_dir,'predictions.csv'),index=False)
    print('V42_LOCAL_COMPLETE',report['status'],len(ledger))
