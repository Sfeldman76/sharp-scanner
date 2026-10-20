"""NCAAF V5.1 research: V4/A point edges, situational attribution, optional historical brains.

READ ONLY: no fitted-model changes, no execution prices/ROI, and no automatic BET authority.
The V5 input has one row per (game, model, *cumulative* threshold). Deduplicate
before every inference; never count thresholds as independent bets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import StringIO
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

VERSION = 'NCAAF_V51_EDGE_ATTRIBUTION_20261020_V1'
MODELS = ('NONLINEAR_STATE_HGB', 'A_EFFICIENCY')
FAMILIES = ('BIG_AL', 'PATHI', 'PREDICTION_TRACKER', 'SYSTEM_MINER', 'STATE_BOUNCE_BACK')
EDGE_CUTS = (0., 1., 2., 3., 5., 7., 10., 15., float('inf'))
EDGE_LABELS = ('0–1', '1–2', '2–3', '3–5', '5–7', '7–10', '10–15', '15+')
REQUIRED = ('Season','Game_ID','Game_Date','Team','Opponent','Model','Model_Margin',
            'Open_Market_Margin','Gap_Points','Threshold','Side','Result')
TRUE_RESULTS = ('WIN','LOSS','PUSH')
KEY = ('Season','Game_ID')


def _metric(df):
    wins=int((df.Result=='WIN').sum())
    losses=int((df.Result=='LOSS').sum())
    pushes=int((df.Result=='PUSH').sum())
    decided=wins+losses
    ci=None
    if decided:
        z=1.959963984540054
        p=wins/decided
        den=1+z*z/decided
        center=(p+z*z/(2*decided))/den
        d=z*math.sqrt(p*(1-p)/decided+z*z/(4*decided*decided))/den
        ci=[float(center-d),float(center+d)]
    return {'observations':int(len(df)), 'wins':wins, 'losses':losses, 'pushes':pushes,
            'decisions':decided, 'hit_rate':float(wins/decided) if decided else None,
            'wilson95':ci}


def _tier(x):
    return pd.cut(x, bins=EDGE_CUTS, labels=EDGE_LABELS, right=False, include_lowest=True)


def validate_and_dedupe(raw, *, v4=None):
    missing=set(REQUIRED)-set(raw.columns)
    if missing: raise ValueError('V51_MISSING_SIGNAL_COLUMNS:'+','.join(sorted(missing)))
    data=raw.copy()
    data['Season']=pd.to_numeric(data['Season'], errors='coerce')
    if data.Season.isna().any() or ((data.Season<2021)|(data.Season>2025)).any():
        raise ValueError('V51_INVALID_HISTORICAL_YEAR')
    data['Season']=data.Season.astype(int)
    data['Game_ID']=data.Game_ID.astype(str).str.strip()
    if data.Game_ID.isin(('', 'nan', 'None')).any():raise ValueError('V51_BAD_GAME_ID')
    data['Game_Date']=pd.to_datetime(data.Game_Date,utc=True,errors='coerce').dt.normalize()
    if data.Game_Date.isna().any():raise ValueError('V51_BAD_DATE')
    for col in ('Model_Margin','Open_Market_Margin','Gap_Points','Threshold'):
        data[col]=pd.to_numeric(data[col],errors='coerce')
        if not np.isfinite(data[col].to_numpy(dtype=float)).all():
            raise ValueError('V51_NONFINITE:'+col)
    data['Model']=data.Model.astype(str)
    if not set(data.Model.unique()).issubset({'NONLINEAR_STATE_HGB','A_EFFICIENCY','AB_EFF_PASS','AC_EFF_DRIVE'}):
        raise ValueError('V51_UNKNOWN_MODEL')
    data=data[data.Model.isin(MODELS)].copy()
    if not len(data):raise ValueError('V51_NO_PRIMARY_MODELS')
    for col in ('Side','Result'):
        data[col]=data[col].astype(str).str.upper().str.strip()
    if not data.Side.isin(('TEAM','OPPONENT')).all() or not data.Result.isin(TRUE_RESULTS).all():
        raise ValueError('V51_UNKNOWN_SIDE_OR_RESULT')
    gap=data.Model_Margin-data.Open_Market_Margin
    if not np.allclose(gap,data.Gap_Points,atol=1e-6,rtol=1e-9):
        raise ValueError('V51_EDGE_ORIENTATION_MISMATCH')
    expected=np.where(gap>0,'TEAM','OPPONENT')
    if (data.Side.to_numpy()!=expected).any() or (gap.abs()<1e-9).any():
        raise ValueError('V51_SELECTION_ORIENTATION_MISMATCH')
    if (data.Threshold.to_numpy()>np.abs(gap.to_numpy())+1e-8).any():
        raise ValueError('V51_THRESHOLD_EXCEEDS_EDGE')
    for col in ('Team','Opponent','Open_Market_Margin','Model_Margin','Gap_Points','Side','Result','Game_Date'):
        nuniq=data.groupby([*KEY,'Model'],dropna=False)[col].nunique(dropna=False)
        if (nuniq>1).any():raise ValueError('V51_CONTRADICTORY_THRESHOLD_DUPLICATES:'+col)
    # No duplicate same threshold or distinct game metadata within key.
    if data.duplicated([*KEY,'Model','Threshold']).any():
        raise ValueError('V51_DUPLICATE_SAME_THRESHOLD')
    for col in ('Team','Opponent','Open_Market_Margin','Game_Date'):
        if (data.groupby(list(KEY))[col].nunique(dropna=False)>1).any():
            raise ValueError('V51_GAME_ORIENTATION_CONFLICT:'+col)
    # Keep exactly one row per physical game and model, regardless of how many
    # cumulative threshold duplicates the original historical V5 CSV contains.
    dedup=(data.sort_values('Threshold').drop_duplicates([*KEY,'Model'],keep='first')
           .sort_values(['Season','Game_Date','Game_ID','Model']).reset_index(drop=True))
    dedup['Abs_Edge']=dedup.Gap_Points.abs()
    dedup['Edge_Bin']=_tier(dedup.Abs_Edge).astype(str)
    dedup['Market_Abs_Spread']=dedup.Open_Market_Margin.abs()
    dedup['Spread_Bin']=pd.cut(dedup.Market_Abs_Spread,
       bins=[-0.001,3,7,14,21,30,float('inf')],
       labels=['0–3','3–7','7–14','14–21','21–30','30+'],right=False).astype(str)
    dedup['Selection_Type']=np.where(dedup.Open_Market_Margin==0,'PICKEM',
       np.where(((dedup.Side=='TEAM')&(dedup.Open_Market_Margin>0))|
                ((dedup.Side=='OPPONENT')&(dedup.Open_Market_Margin<0)),'FAVORITE','UNDERDOG'))
    dedup['Selection_Home_Away']='UNKNOWN_ORIENTATION'  # source has no trustworthy home team identifier
    dedup['Week_Window']=np.where(dedup.Game_Date.dt.month.isin([8,9]),'AUG_SEP','OCT_OR_LATER')
    dedup['Period']=np.where(dedup.Season<=2023,'development_2021_2023','examined_2024_2025')
    dedup['Is_Extreme_Spread']=dedup.Market_Abs_Spread>=30
    if v4 is not None:
        dedup=_verify_independent_v4(dedup,v4)
    return dedup


def _verify_independent_v4(dedup, source):
    required=['Season','Game_ID','Team','Opponent','Actual_Margin','Verified_Open_Margin_Diagnostic','NONLINEAR_STATE_HGB']
    if missing:=set(required)-set(source.columns):
        raise ValueError('V51_V4_MISSING_COLUMNS:'+','.join(sorted(missing)))
    s=source[required].copy()
    s.Season=pd.to_numeric(s.Season,errors='raise').astype(int)
    s.Game_ID=s.Game_ID.astype(str).str.strip()
    if s.duplicated(list(KEY)).any():raise ValueError('V51_V4_DUPLICATE_GAMES')
    s=s.rename(columns={'Team':'V4_Team','Opponent':'V4_Opp','Actual_Margin':'Actual_Margin',
                        'Verified_Open_Margin_Diagnostic':'Verified_Open','NONLINEAR_STATE_HGB':'V4_Base'})
    m=dedup.merge(s,on=list(KEY),how='left',validate='many_to_one',indicator=True)
    if (m._merge!='both').any():raise ValueError('V51_V4_SOURCE_MATCH_MISSING')
    if (m.Team.astype(str)!=m.V4_Team.astype(str)).any() or (m.Opponent.astype(str)!=m.V4_Opp.astype(str)).any():
        raise ValueError('V51_V4_TEAM_ORIENTATION_MISMATCH')
    if not np.allclose(m.Open_Market_Margin,m.Verified_Open,rtol=0,atol=1e-6):
        raise ValueError('V51_V4_OPEN_MARGIN_MISMATCH')
    base=m[m.Model=='NONLINEAR_STATE_HGB']
    if not np.allclose(base.Model_Margin,base.V4_Base,rtol=0,atol=1e-5):
        raise ValueError('V51_V4_FROZEN_PREDICTION_MISMATCH')
    actual=pd.to_numeric(m.Actual_Margin,errors='coerce')
    if not np.isfinite(actual.to_numpy(float)).all():raise ValueError('V51_V4_MISSING_ACTUAL')
    directional=(actual-m.Open_Market_Margin)*np.where(m.Side=='TEAM',1,-1)
    observed=np.where(abs(directional)<1e-8,'PUSH',np.where(directional>0,'WIN','LOSS'))
    if (observed!=m.Result).any():raise ValueError('V51_V4_RESULT_ORIENTATION_MISMATCH')
    return m.drop(columns=['_merge','V4_Team','V4_Opp','Verified_Open','V4_Base'])


def _brain_join(dedup, raw_brains):
    # Eligibility is deliberately conservative. These are research diagnostics,
    # NOT independently certified archived signals or authorities.
    blank={'status':'NO_VERIFIABLE_HISTORICAL_BRAIN_SNAPSHOTS', 'received':0,'eligible_records':0,
           'independent_archive_proof':'NOT_ESTABLISHED','authoritative_brain_votes':0,
           'families':{f:0 for f in FAMILIES}}
    for f in FAMILIES:dedup['Brain_'+f]='MISSING'
    if raw_brains is None or not len(raw_brains):return dedup,blank,pd.DataFrame()
    from ncaaf_v5_betting_research import validate_brains
    matched=dedup[list(KEY)].drop_duplicates()
    b,audit=validate_brains(raw_brains,matched)
    b=b[b.Rule_Predeclared_Before_Season].copy()
    if not len(b):
        blank.update({'status':'NO_PRESEASON_PREDECLARED_BRAIN_RULES', 'received':len(raw_brains)})
        return dedup,blank,pd.DataFrame()
    # Avoid correlated overlapping rules becoming counted as votes.
    for family in FAMILIES:
        cur=b[b.Family==family]
        if cur.empty:continue
        pair=cur.groupby(list(KEY))['Side'].agg(lambda v: frozenset(v))
        for pos,row in dedup.iterrows():
            sides=pair.get((row.Season,row.Game_ID),frozenset())
            if not sides:continue
            dedup.at[pos,'Brain_'+family]=('CONFLICT' if len(sides)>1 else
                ('CONFIRM' if row.Side in sides else 'OPPOSE'))
    blank.update({'status':'HISTORICAL_BRAINS_DIAGNOSTIC_ONLY__ORIGINAL_ARCHIVE_UNCERTIFIED',
        'received':int(len(raw_brains)),'eligible_records':int(len(b)),
        'families':{f:int((b.Family==f).sum()) for f in FAMILIES}})
    return dedup,blank,b


def _group_rows(df, keys, min_decisions=0):
    if isinstance(keys,str):keys=[keys]
    group=[]
    for k,part in df.groupby(keys,dropna=False,observed=True,sort=True):
        stats=_metric(part)
        if stats['decisions'] < min_decisions:continue
        kvals=list(k) if isinstance(k,tuple) else [k]
        group.append({'group':dict(zip(keys,[str(v) for v in kvals])),**stats,
            'eligible_for_inference':False,
            'note':'Exploratory subgroup; previously examined data and multiple comparisons'})
    return group


def _one_model(df):
    out={}
    for period,part in df.groupby('Period'):
        out[period]={
            'overall':_metric(part),
            'by_year':_group_rows(part,['Season']),
            'exclusive_edge_bins':_group_rows(part,['Edge_Bin']),
            'edge_bins_by_year':_group_rows(part,['Season','Edge_Bin']),
            'by_selection_type':_group_rows(part,['Selection_Type']),
            'by_spread_bin':_group_rows(part,['Spread_Bin']),
            'by_edge_and_selection':_group_rows(part,['Edge_Bin','Selection_Type']),
            'by_spread_and_edge':_group_rows(part,['Spread_Bin','Edge_Bin']),
            'by_season_window':_group_rows(part,['Week_Window']),
            'by_market_extremity':_group_rows(part,['Is_Extreme_Spread']),
            'by_direction':_group_rows(part,['Side']),
            'conditional_gte_edges':[
                {'min_edge_points':t,**_metric(part[part.Abs_Edge>=t])}
                for t in (1.,2.,3.,5.,7.,10.,15.)
            ],
        }
    return out


def _comparison(df):
    # Exactly one row per model/game. Pair evaluation years and same market.
    base=df[df.Model==MODELS[0]]
    a=df[df.Model==MODELS[1]]
    j=base.merge(a,on=list(KEY),validate='one_to_one',suffixes=('_V4','_A'))
    if not len(j):raise ValueError('V51_NO_PAIRED_V4_A_GAMES')
    if (j.Result_V4!=j.Result_A).any():
        side_disagree=j.Side_V4!=j.Side_A
        # If sides are same, outcomes MUST match.
        if (j.loc[~side_disagree,'Result_V4']!=j.loc[~side_disagree,'Result_A']).any():
            raise ValueError('V51_PAIRED_ATS_OUTCOME_CONFLICT')
    results={}
    for period,sub in j.groupby('Period_V4'):
        diff=sub.Side_V4!=sub.Side_A
        opposed=sub[diff]
        agreed=sub[~diff]
        results[period]={'paired_games':int(len(sub)),
                         'same_side_games':int(len(agreed)),
                         'opposing_sides_games':int(len(opposed)),
                         'v4_on_disagreements':_metric(opposed.rename(columns={'Result_V4':'Result'})),
                         'a_on_disagreements':_metric(opposed.rename(columns={'Result_A':'Result'})),
                         'large_edge_both_same_side':_metric(agreed[(agreed.Abs_Edge_V4>=3)&(agreed.Abs_Edge_A>=3)]
                            .rename(columns={'Result_V4':'Result'})),
                         'notes':'Paired forecasts highly correlated; this comparison is NOT independent betting evidence'}
    return results


def _brain_scorecard(df):
    score={}
    for model, m in df.groupby('Model'):
        score[model]={}
        for family in FAMILIES:
            sub={}
            for status in ('CONFIRM','OPPOSE','CONFLICT','MISSING'):
                for period in ('development_2021_2023','examined_2024_2025'):
                    part=m[(m['Brain_'+family]==status)&(m.Period==period)]
                    sub[f'{period}/{status}']=_metric(part)
            score[model][family]=sub
    return score


def _per_system_research(data, eligible):
    if eligible is None or eligible.empty:return {}
    results={}
    # Brain rules were already checked for pregame timestamp claims and
    # preseason declaration. Neither is a substitute for true archive proof.
    for (family,system_id),rule in eligible.groupby(['Family','System_ID'],sort=True):
        matched=data.merge(rule[[*KEY,'Side']].rename(columns={'Side':'Rule_Side'}),
                           on=list(KEY),how='inner',validate='many_to_one')
        matched['Relation']=np.where(matched.Side==matched.Rule_Side,'CONFIRM','OPPOSE')
        by_model={}
        for model,group in matched.groupby('Model'):
            by_model[model]={}
            for period,part in group.groupby('Period'):
                by_model[model][period]={status:_metric(part[part.Relation==status])
                  for status in ('CONFIRM','OPPOSE')}
        results[f'{family}/{system_id}']={'snapshots':int(len(rule)),
            'eligible_for_bet_authority':False,
            'original_archive_verified':False,
            'threshold_policy':'No promotion; source authenticity and independent prospective results required',
            'by_model':by_model}
    return results


def research(raw, *, v4=None, brains=None, source=None):
    data=validate_and_dedupe(raw,v4=v4)
    data,brain_audit,eligible_brains=_brain_join(data,brains)
    report={'version':VERSION,'status':'HISTORICAL_EDGE_ATTRIBUTION__SOURCE_VINTAGE_HOLD',
       'generated_utc':datetime.now(timezone.utc).isoformat(),
       'v4_prediction_verified_with_source':bool(v4 is not None),
       'source_vintage_certified':False,
       'source':source or {},
       'raw_threshold_rows':int(len(raw)),'dedup_model_game_rows':int(len(data)),
       'physical_games':int(data[list(KEY)].drop_duplicates().shape[0]),
       'verified_market_years':sorted(data.Season.unique().astype(int).tolist()),
       'model_analyses':{model:_one_model(data[data.Model==model]) for model in MODELS},
       'paired_v4_vs_efficiency':_comparison(data),
       'brain_coverage':brain_audit,
       'brain_conditional_ats':_brain_scorecard(data),
       'individual_system_research':_per_system_research(data,eligible_brains),
       'warnings':[
          'No executable historical bookmaker odds: NO ROI, no betting profitability claim',
          'Brain confirmations with no verified source snapshots are MISSING, not false votes',
          'Snapshot timestamp claims do not independently certify archived authenticity',
          'Opening spreads are retrospective diagnostics; cannot assume available at a future quote',
          '2024–2025 already inspected, not fresh holdouts',
          'Each exclusive edge bin is one nonduplicated model/game, not cumulative threshold rows',
          'Stratifications are exploratory multiple comparisons: NO automatic bet systems',
          'No verified home/away, FBS/FCS, current spreads, line movement or contemporaneous odds in V5 CSV',
          'No automatic BET or STRONG BET authority; legacy production files unchanged'],
       'market_quote_basis':'VERIFIED_HISTORICAL_OPEN_MARGIN_ONLY',
       'decision_categories':{'CANDIDATE':'Hypothesis only; points alone not betting authority',
            'BET':'NOT_AUTHORIZED','STRONG_BET':'NOT_AUTHORIZED','NO_BET':'Default live action pending evidence'},
       'roi':None, 'production_authority':0,'production_modified':False}
    return report,data


def cloud_run(*, storage_client, bucket_name='sharp-models', log_func=print):
    bucket=storage_client.bucket(bucket_name)
    parent=json.loads(bucket.blob('research/ncaaf/v5_market_value/latest_report.json').download_as_text())
    if parent.get('status')!='HISTORICAL_DIRECTIONAL_RESEARCH__SOURCE_VINTAGE_HOLD':
        raise RuntimeError('V51_UNEXPECTED_V5_PARENT_REPORT_STATUS')
    uri=parent.get('signals_uri')
    prefix=f'gs://{bucket_name}/research/ncaaf/v5_market_value/runs/'
    if not isinstance(uri,str) or not uri.startswith(prefix) or not uri.endswith('/research_signals_not_bets.csv'):
        raise RuntimeError('V51_UNSAFE_OR_MISSING_IMMUTABLE_V5_SIGNAL_POINTER')
    key=uri[len(f'gs://{bucket_name}/'):]
    raw=pd.read_csv(StringIO(bucket.blob(key).download_as_text()),low_memory=False)
    v4=pd.read_csv(StringIO(bucket.blob('research/ncaaf/margin_tournament_v4/game_level_predictions.csv').download_as_text()),low_memory=False)
    brains=None
    bpath=os.getenv('NCAAF_V5_HISTORICAL_BRAINS_GCS_OBJECT','').strip()
    if bpath:
        if not bpath.startswith('research/ncaaf/v5_inputs/') or not bpath.endswith('.csv'):
            raise RuntimeError('V51_UNSAFE_BRAIN_INPUT_PATH')
        brains=pd.read_csv(StringIO(bucket.blob(bpath).download_as_text()),low_memory=False)
    report,data=research(raw,v4=v4,brains=brains,source={'v5_immutable_signals':uri,
        'v4_pred_source':f'gs://{bucket_name}/research/ncaaf/margin_tournament_v4/game_level_predictions.csv',
        'historical_brains_path':bpath or None})
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=f'research/ncaaf/v51_edge_attribution/runs/{run_id}'
    report['report_uri']=f'gs://{bucket_name}/{path}/report.json'
    report['dedup_ledger_uri']=f'gs://{bucket_name}/{path}/unique_model_game_edges.csv'
    # Publish only versioned outputs first, then latest pointer.
    bucket.blob(f'{path}/report.json').upload_from_string(json.dumps(report,indent=2,allow_nan=False),content_type='application/json',if_generation_match=0)
    bucket.blob(f'{path}/unique_model_game_edges.csv').upload_from_string(data.to_csv(index=False),content_type='text/csv',if_generation_match=0)
    bucket.blob('research/ncaaf/v51_edge_attribution/latest_report.json').upload_from_string(json.dumps(report,indent=2,allow_nan=False),content_type='application/json')
    log_func(f"[NCAAF-V51] status={report['status']} raw_signals={report['raw_threshold_rows']} unique_model_game={report['dedup_model_game_rows']} games={report['physical_games']} brains={report['brain_coverage']['eligible_records']} betting_authority=0")
    return report


def main():
    parser=argparse.ArgumentParser(description='Read-only NCAAF V5.1 edge analysis')
    parser.add_argument('--signals',required=True)
    parser.add_argument('--v4',help='Optional original V4 game-level CSV for independent outcome/orientation parity')
    parser.add_argument('--brains',help='Optional historical timestamped brain-snapshot CSV')
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(args.signals,low_memory=False)
    v4=pd.read_csv(args.v4,low_memory=False) if args.v4 else None
    brains=pd.read_csv(args.brains,low_memory=False) if args.brains else None
    rep,ledger=research(raw,v4=v4,brains=brains)
    (out/'latest_report.json').write_text(json.dumps(rep,indent=2,allow_nan=False))
    ledger.to_csv(out/'unique_model_game_edges.csv',index=False)
    print(f"[NCAAF-V51] raw={rep['raw_threshold_rows']} unique={rep['dedup_model_game_rows']} games={rep['physical_games']} brain_qualified={rep['brain_coverage']['eligible_records']} ROI=UNAVAILABLE")

if __name__=='__main__':main()
