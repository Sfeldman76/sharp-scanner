"""V5 NCAAF independent-margin -> ATS/market-value research, NO production authority.

Consumes versioned, season-forward V4 + V4.2 prediction artifacts. Authentic V4
opening margins are BENCHMARKS only. Forecasts never use market inputs.

Big Al, Pathi, Prediction Tracker, Miner, and state/bounce-back are *optional*
independently timestamped evidence sources, not automatically approved votes.
No historical executable quote means no historical ROI calculation.
"""
from __future__ import annotations
from datetime import datetime, timezone
from collections import defaultdict
from hashlib import sha256
from io import StringIO
import json, math, os, re
import numpy as np
import pandas as pd

VERSION = 'NCAAF_V5_MARGIN_TO_VALUE_RESEARCH_20261020_V1'
BASE = 'NONLINEAR_STATE_HGB'
CHALLENGERS = ('A_EFFICIENCY', 'AB_EFF_PASS', 'AC_EFF_DRIVE')
MODELS = (BASE,) + CHALLENGERS
THRESHOLDS = (1., 2., 3., 5., 7.)
BRAIN_FAMILIES = ('BIG_AL','PATHI','PREDICTION_TRACKER','SYSTEM_MINER','STATE_BOUNCE_BACK')
KEY = ('Season','Game_ID')
PRE_DISCOVERY = (2021, 2022, 2023)
RETROSPECTIVE = (2024, 2025)


def _num(s):
    return pd.to_numeric(s, errors='coerce')


def _missing(df, cols):
    return sorted(set(cols) - set(df.columns))


def _finite(x):
    return np.isfinite(np.asarray(x, float))


def _rate_stats(df):
    wins = int((df['Result'] == 'WIN').sum())
    losses = int((df['Result'] == 'LOSS').sum())
    pushes = int((df['Result'] == 'PUSH').sum())
    n = wins+losses
    if n:
        z=1.959963984540054; p=wins/n; d=1+z*z/n
        center=(p+z*z/(2*n))/d
        half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
        ci=[float(center-half),float(center+half)]
    else:ci=None
    return {'signals_not_bets':int(len(df)), 'wins':wins, 'losses':losses,
            'pushes':pushes, 'hit_rate_ex_push':float(wins/n) if n else None,
            'wilson95':ci}


def _parse_key(df, *, label):
    out=df.copy()
    miss=_missing(out, ('Season','Game_ID','Game_Date','Actual_Margin'))
    if miss:raise ValueError(f'{label}:MISSING:{miss}')
    out['Season']=_num(out['Season'])
    if out['Season'].isna().any() or (out['Season']>=2026).any() or (out['Season']<2013).any():
        raise ValueError(f'{label}:INVALID_HISTORICAL_SEASONS')
    out['Season']=out['Season'].astype(int)
    out['Game_ID']=out['Game_ID'].astype('string').str.strip()
    if out['Game_ID'].isna().any() or out['Game_ID'].isin(['','nan','None']).any():
        raise ValueError(f'{label}:MISSING_GAME_ID')
    if out.duplicated(list(KEY)).any():raise ValueError(f'{label}:DUPLICATE_GAME_KEYS')
    out['Game_Date']=pd.to_datetime(out['Game_Date'],errors='coerce',utc=True).dt.normalize()
    if out['Game_Date'].isna().any():raise ValueError(f'{label}:INVALID_GAME_DATE')
    out['Actual_Margin']=_num(out['Actual_Margin'])
    return out


def join_frozen_predictions(v4_raw, v42_raw):
    """Fail on mismatch; V4-A train artifacts must be exact game-for-game partners."""
    v4=_parse_key(v4_raw,label='V4')
    v42=_parse_key(v42_raw,label='V42')
    need=('Verified_Open_Margin_Diagnostic',BASE,'Team','Opponent')
    if miss:=_missing(v4,need):raise ValueError(f'V4:MISSING:{miss}')
    if miss:=_missing(v42,MODELS):raise ValueError(f'V42:MISSING:{miss}')
    a=v4[list(KEY)+['Game_Date','Actual_Margin','Team','Opponent','Verified_Open_Margin_Diagnostic',BASE]].copy()
    a=a.rename(columns={'Game_Date':'V4_Date','Actual_Margin':'V4_Actual',BASE:'V4_Reference'})
    b=v42[list(KEY)+['Game_Date','Actual_Margin',*MODELS]].copy()
    b=b.rename(columns={'Game_Date':'V42_Date','Actual_Margin':'V42_Actual'})
    joined=a.merge(b,on=list(KEY),how='inner',validate='one_to_one')
    if len(joined)!=len(a) or len(joined)!=len(b):
        raise ValueError('SOURCE_GAME_SET_MISMATCH')
    if (joined.V4_Date!=joined.V42_Date).any():raise ValueError('GAME_DATE_MISMATCH')
    for col in ['V4_Actual','V42_Actual','V4_Reference',*MODELS]:
        joined[col]=_num(joined[col])
    if not np.allclose(joined.V4_Actual,joined.V42_Actual,atol=1e-6,equal_nan=True):
        raise ValueError('ACTUAL_OUTCOMES_DIFFER')
    if not np.allclose(joined.V4_Reference,joined[BASE],atol=1e-5,equal_nan=True):
        raise ValueError('FROZEN_V4_BASELINE_PARITY_BROKEN')
    if not all(_finite(joined[col]).all() for col in MODELS):
        raise ValueError('MISSING_OR_INF_FOOTBALL_MODEL_PREDICTIONS')
    joined=joined.rename(columns={'V42_Date':'Game_Date','V42_Actual':'Actual_Margin',
        'Verified_Open_Margin_Diagnostic':'Market_Open_Margin'})
    joined['Market_Open_Margin']=_num(joined['Market_Open_Margin'])
    if joined.loc[joined.Season<2021,'Market_Open_Margin'].notna().any():
        raise ValueError('UNEXPECTED_PRE2021_OPEN_MARKET_VALUES')
    return joined.drop(columns=['V4_Date','V4_Actual','V4_Reference'])


def validate_brains(brain_raw, matched):
    """Accept only explicit pregame records; historical discovery is NOT certified.

    Source timestamp and pre-season rule declaration are enforced. This cannot
    independently certify that a historical snapshot was *really* archived then;
    read-only analyses remain exploratory until external source evidence exists.
    """
    if brain_raw is None or len(brain_raw)==0:
        return pd.DataFrame(columns=[*KEY,'Family','System_ID','Side','Snapshot_UTC','Kickoff_UTC']), {
            'status':'NO_VERIFIABLE_SNAPSHOTS_PROVIDED', 'records_received':0,
            'pregame_records_kept':0, 'family_records':{f:0 for f in BRAIN_FAMILIES}}
    required=[*KEY,'Family','System_ID','Side','Snapshot_UTC','Kickoff_UTC',
              'Rule_Declared_UTC','Source_Artifact_SHA256']
    if miss:=_missing(brain_raw,required):raise ValueError('BRAIN_SCHEMA_MISSING:'+repr(miss))
    b=brain_raw.copy()
    b['Season']=_num(b['Season'])
    if b['Season'].isna().any() or (b['Season']>=2026).any():
        raise ValueError('BRAIN_INVALID_HISTORICAL_SEASON')
    b['Season']=b['Season'].astype(int)
    b['Game_ID']=b['Game_ID'].astype('string').str.strip()
    for col in ['Snapshot_UTC','Kickoff_UTC','Rule_Declared_UTC']:
        b[col]=pd.to_datetime(b[col],utc=True,errors='coerce')
    b['Family']=b['Family'].astype(str).str.upper().str.strip()
    b['Side']=b['Side'].astype(str).str.upper().str.strip()
    b['System_ID']=b['System_ID'].astype(str).str.strip()
    b['Source_Artifact_SHA256']=b['Source_Artifact_SHA256'].astype(str).str.strip()
    if (~b['Family'].isin(BRAIN_FAMILIES)).any() or (~b.Side.isin(['TEAM','OPPONENT'])).any():
        raise ValueError('UNKNOWN_BRAIN_FAMILY_OR_SIDE')
    if b[["Snapshot_UTC","Kickoff_UTC","Rule_Declared_UTC"]].isna().any().any():
        raise ValueError('BRAIN_MISSING_VINTAGE_TIMESTAMPS')
    if b['System_ID'].isin(['','nan','None']).any() or (~b.Source_Artifact_SHA256.str.fullmatch('[a-fA-F0-9]{64}')).any():
        raise ValueError('BRAIN_INVALID_SYSTEM_OR_SOURCE_HASH')
    if (b.Snapshot_UTC>=b.Kickoff_UTC).any():raise ValueError('BRAIN_POST_KICKOFF_SIGNAL')
    if (b.Rule_Declared_UTC>=b.Kickoff_UTC).any():raise ValueError('BRAIN_RULE_DECLARED_AFTER_GAME')
    keys=matched[list(KEY)].drop_duplicates()
    b=b.merge(keys,on=list(KEY),how='inner',validate='many_to_one')
    if b.duplicated(list(KEY)+['Family','System_ID']).any():
        raise ValueError('BRAIN_DUPLICATE_SYSTEM_GAME')
    # Context-only if rule discovered in/after the evaluation season; cannot
    # be treated as historical qualified evidence for that season.
    season_start=pd.to_datetime(b.Season.astype(str)+'-01-01',utc=True)
    b['Rule_Predeclared_Before_Season']=b.Rule_Declared_UTC.lt(season_start)
    counts=b.groupby('Family').size().to_dict()
    return b, {'status':'PRE_KICKOFF_CLAIM_FILTERED__SOURCE_ARCHIVE_AUTHENTICITY_UNCERTIFIED',
               'records_received':int(len(brain_raw)),'pregame_records_kept':int(len(b)),
               'preseason_rule_records':int(b.Rule_Predeclared_Before_Season.sum()),
               'family_records':{f:int(counts.get(f,0)) for f in BRAIN_FAMILIES}}


def _per_market_orientation(p, mkt):
    return 'TEAM' if p>mkt else 'OPPONENT'


def _calibrate(training, model, market, prediction, *, min_rows=400):
    """Empirical residual CDF from earlier seasons' true OOF predictions."""
    residual=_num(training['Actual_Margin'])-_num(training[model])
    residual=np.sort(residual.to_numpy(dtype=float)[_finite(residual)])
    if len(residual)<min_rows:return None,len(residual)
    needed=float(market)-float(prediction)
    count_right=len(residual)-int(np.searchsorted(residual,needed,side='right'))
    p=(count_right+0.5)/(len(residual)+1.0)
    return float(np.clip(p,0.01,0.99)),int(len(residual))


def _iso_date(x):return pd.Timestamp(x).strftime('%Y-%m-%d')


def run_historical(v4_raw, v42_raw, *, brain_raw=None, source_info=None):
    games=join_frozen_predictions(v4_raw,v42_raw)
    if len(games)<1000:raise ValueError('HISTORICAL_SAMPLE_TOO_SMALL')
    brains,brain_audit=validate_brains(brain_raw,games)
    by_game=defaultdict(list)
    for record in brains.itertuples(index=False):
        by_game[(int(record.Season),str(record.Game_ID))].append(record)
    rows=[];calibration_count={}
    for year in sorted(games.Season.unique()):
        if year<2021:continue
        tr=games.loc[(games.Season<year)&(games.Season>=2017)]
        va=games.loc[games.Season==year]
        calibration_count[str(year)]={}
        for model in MODELS:
            res=_num(tr.Actual_Margin)-_num(tr[model]);res=res.to_numpy(float)
            res=np.sort(res[np.isfinite(res)])
            calibration_count[str(year)][model]=int(len(res))
            for game in va.itertuples(index=False):
                market=getattr(game,'Market_Open_Margin')
                pred=float(getattr(game,model))
                if not math.isfinite(market) or not math.isfinite(pred):continue
                if len(res)<400:continue
                nright=len(res)-int(np.searchsorted(res,float(market)-pred,side='right'))
                pteam=float(np.clip((nright+0.5)/(len(res)+1),0.01,0.99))
                gap=pred-float(market)
                if abs(gap)<1e-12:continue
                side=_per_market_orientation(pred,float(market))
                pside=pteam if side=='TEAM' else 1-pteam
                performance=(float(game.Actual_Margin)-float(market))*(1 if side=='TEAM' else -1)
                outcome='PUSH' if abs(performance)<1e-9 else ('WIN' if performance>0 else 'LOSS')
                evidence=by_game.get((int(year),str(game.Game_ID)),[])
                # Only pre-season-declared rules contribute to historical
                # "confluence"; other records are kept for audit only.
                eligible=[x for x in evidence if x.Rule_Predeclared_Before_Season]
                agree=[x for x in eligible if x.Side==side]
                oppose=[x for x in eligible if x.Side!=side]
                family_status={f:'MISSING' for f in BRAIN_FAMILIES}
                for family in BRAIN_FAMILIES:
                    frows=[x for x in eligible if x.Family==family]
                    if frows:
                        fsides={x.Side for x in frows}
                        family_status[family]= ('CONFLICT' if len(fsides)>1 else
                            ('CONFIRM' if side in fsides else 'OPPOSE'))
                for threshold in THRESHOLDS:
                    if abs(gap)+1e-12<threshold:continue
                    rows.append({'Season':int(year),'Game_ID':str(game.Game_ID),
                        'Game_Date':_iso_date(game.Game_Date),'Team':str(game.Team),
                        'Opponent':str(game.Opponent),'Model':model,
                        'Model_Margin':pred,'Open_Market_Margin':float(market),
                        'Gap_Points':float(gap),'Threshold':float(threshold),
                        'Side':side,'P_Cover_Empirical':pside,'Calibration_N':int(len(res)),
                        'Result':outcome,'Brain_Confirm_N':len(agree),
                        'Brain_Oppose_N':len(oppose),
                        'Brain_Predeclared_Records_N':len(eligible),
                        'Brain_Unqualified_Records_N':len(evidence)-len(eligible),
                        **{f'Brain_{f}':family_status[f] for f in BRAIN_FAMILIES}})
    if not rows:
        # A successful empty job must not be represented as validated.
        status='NO_VERIFIED_HISTORICAL_OPEN_MARKET_GAMES__RESEARCH_BLOCKED'
        ledger=pd.DataFrame()
    else:
        status='HISTORICAL_DIRECTIONAL_RESEARCH__SOURCE_VINTAGE_HOLD'
        ledger=pd.DataFrame(rows).sort_values(['Season','Game_Date','Game_ID','Model','Threshold']).reset_index(drop=True)
    strat={};season_report={};brain_report={};system_report={};calibration_quality={}
    # Measure probability quality on non-push games. The 1-point slice is a
    # diagnostic SUBSET, not a comprehensive unbiased reliability curve.
    if len(ledger):
        for period,lo,hi in [('development_2021_2023',2021,2023),('retrospective_2024_2025',2024,2025)]:
            calibration_quality[period]={}
            for model in MODELS:
                sub=ledger.loc[(ledger.Season.between(lo,hi))&(ledger.Model==model)&
                               (ledger.Threshold==1.)&(ledger.Result!='PUSH')]
                if len(sub):
                    observed=(sub.Result=='WIN').to_numpy(float)
                    prob=sub.P_Cover_Empirical.to_numpy(float)
                    calibration_quality[period][model]={
                        'n':len(sub),'brier':float(np.mean((observed-prob)**2)),
                        'log_loss':float(np.mean(-(observed*np.log(prob)+(1-observed)*np.log1p(-prob)))),
                        'scope':'ONLY_ABSOLUTE_GAP_GE_1_NOT_ALL_GAMES'}
                else:calibration_quality[period][model]={'n':0,'status':'NO_EVALUABLE_CALIBRATION'}

    # Every named system is evaluated only on records with an independently
    # pre-season-declared rule, and always separately from model signals.
    # Multiple-testing and source-archive provenance remain explicit holds.
    if len(brains):
        outcome_map=games[list(KEY)+['Actual_Margin','Market_Open_Margin']].copy()
        qualified=brains.loc[brains.Rule_Predeclared_Before_Season].merge(
            outcome_map,on=list(KEY),how='inner',validate='many_to_one')
        qualified=qualified.loc[qualified.Market_Open_Margin.notna()].copy()
        qualified['Edge_Outcome']=(qualified.Actual_Margin-qualified.Market_Open_Margin)*np.where(qualified.Side=='TEAM',1,-1)
        qualified['Result']=np.where(np.isclose(qualified.Edge_Outcome,0), 'PUSH',
                             np.where(qualified.Edge_Outcome>0,'WIN','LOSS'))
        for (family,system),part in qualified.groupby(['Family','System_ID']):
            key=f'{family}/{system}'
            system_report[key]={**_rate_stats(part),'seasons':sorted(int(x) for x in part.Season.unique()),
                'status':'EXPLORATORY_PREDECLARED_RULE__ORIGINAL_SNAPSHOT_AUTHENTICITY_HOLD',
                'qualified_for_betting':False}

    for period,lo,hi in [('development_2021_2023',2021,2023),('retrospective_2024_2025',2024,2025)]:
        strat[period]={}
        for model in MODELS:
            strat[period][model]={}
            for t in THRESHOLDS:
                sub=(ledger.loc[(ledger.Season.between(lo,hi))&(ledger.Model==model)&(ledger.Threshold==t)]
                     if len(ledger) else pd.DataFrame(columns=['Result']))
                strat[period][model][str(t)]=_rate_stats(sub)
        if len(ledger):
            subset=ledger.loc[ledger.Season.between(lo,hi)]
            # Avoid duplicate thresholds/model exposures in brain frequency.
            ref=subset.loc[(subset.Model=='A_EFFICIENCY')&(subset.Threshold==2.)]
            for family in BRAIN_FAMILIES:
                brain_report[f'{period}/{family}']={k:_rate_stats(ref.loc[ref['Brain_'+family]==k])
                    for k in ('CONFIRM','OPPOSE','CONFLICT','MISSING')}
    if len(ledger):
        for year,subset in ledger.groupby('Season'):
            season_report[str(year)]={m:_rate_stats(subset.loc[(subset.Model==m)&(subset.Threshold==3.)]) for m in MODELS}
    calibration_2026={}
    for model in MODELS:
        past=games.loc[games.Season.between(2017,2025)]
        errors=(_num(past.Actual_Margin)-_num(past[model])).to_numpy(float)
        errors=np.sort(errors[np.isfinite(errors)])
        calibration_2026[model]={'type':'EMPIRICAL_OOF_MARGIN_ERROR_CDF',
            'n':len(errors),'residuals':errors.tolist(),
            'historical_years':'2017-2025','source_vintage_certified':False}
    source_lines={str(y):int(games.loc[games.Season==y,'Market_Open_Margin'].notna().sum())
                  for y in sorted(games.Season.unique())}
    report={'version':VERSION,'status':status,
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'historical_games':int(len(games)),'verified_opening_coverage_by_year':source_lines,
        'baseline_parity':'EXACT_V4_TO_V42_PER_GAME',
        'calibration_training_rows_by_year':calibration_count,
        'opening_not_model_input':True, 'research_model_set':list(MODELS),
        'precommitted_betting_threshold':None, 'selection_status':'NO_THRESHOLD_OPTIMIZED_OR_PROMOTED',
        'thresholds_predeclared_points':list(THRESHOLDS),
        'period_summaries':strat,'year_summaries':season_report,
        'cover_probability_diagnostics':calibration_quality,
        'brain_family_status':brain_audit,'brain_confluence_diagnostics':brain_report,
        'individual_brain_system_scorecards':system_report,
        'historical_roi':None,'executable_odds':False,
        'point_in_time_certification':'HOLD_PENDING_RAW_CFBD_PUBLICATION_TIMESTAMPS',
        'historical_brain_certification':'HOLD_UNTIL_INDEPENDENT_SNAPSHOT_ARCHIVE_PROOF',
        'historical_2024_2025_holdout':'PREVIOUSLY_EXAMINED_NOT_PRISTINE',
        'promotion_authority':0,'production_mutated':False,
        'limits':['Direction and cover probability only; no verified historical executable quotes',
            'Thresholds are research strata, NOT bet recommendations',
            'Brain missing means UNKNOWN, not a negative vote',
            'Brains with rules developed during a season may not confirm that season',
            'Historical brain input claims require independent timestamp provenance',
            'Market opening coverage may limit historical ROI research',
            '2026 future pregame model inference is not supplied by this engine'],
        'source_artifacts':source_info or {}}
    calibration={'version':VERSION,'status':'RESEARCH_CALIBRATION_ONLY__SOURCE_VINTAGE_HOLD',
        'generated_utc':report['generated_utc'],'model_calibration':calibration_2026,
        'no_betting_authority':True,'requires_certified_pregame_predictions':True}
    return report,ledger,calibration


def cloud_run(*, storage_client, bucket_name='sharp-models', log_func=print):
    from ncaaf_v43_validation import _read_source_artifacts
    bucket=storage_client.bucket(bucket_name)
    src_report,v42,source_report_uri,source_ledger_uri=_read_source_artifacts(bucket)
    if src_report.get('status')!='EXPLORATORY_RESEARCH__SOURCE_VINTAGE_HOLD':
        raise RuntimeError('V5_UNEXPECTED_V42_SOURCE_STATUS')
    if any(src_report.get('parity_to_published_v4',{}).get(str(y))!='EXACT_REPRODUCTION'
           for y in range(2017,2026)):
        raise RuntimeError('V5_V42_PUBLISHED_V4_PARITY_NOT_PROVEN')
    if any(src_report.get('parity_to_selected_v4_champion',{}).get(str(y))!='EXACT_REPRODUCTION'
           for y in range(2017,2026)):
        raise RuntimeError('V5_V42_SELECTED_V4_CHAMPION_PARITY_NOT_PROVEN')
    if src_report.get('frozen_champion') not in ('NONLINEAR_STATE_HGB','NONLINEAR_HGB'):
        raise RuntimeError('V5_V4_CHAMPION_UNRECOGNIZED')
    if src_report.get('v4_ledger_exact_game_parity',{}).get('status')!='EXACT_PER_GAME':
        raise RuntimeError('V5_V42_EXACT_V4_GAME_LEVEL_PARITY_NOT_PROVEN')
    v4=pd.read_csv(StringIO(bucket.blob('research/ncaaf/margin_tournament_v4/game_level_predictions.csv').download_as_text()),low_memory=False)
    brains=None
    brain_path=os.getenv('NCAAF_V5_HISTORICAL_BRAINS_GCS_OBJECT','').strip()
    if brain_path:
        if not brain_path.startswith('research/ncaaf/v5_inputs/') or not brain_path.endswith('.csv'):
            raise RuntimeError('V5_UNAPPROVED_BRAIN_INPUT_PATH')
        brains=pd.read_csv(StringIO(bucket.blob(brain_path).download_as_text()),low_memory=False)
    source={'v42_report_gcs':source_report_uri,'v42_predictions_gcs':source_ledger_uri,
        'v4_predictions_gcs':'research/ncaaf/margin_tournament_v4/game_level_predictions.csv',
        'optional_brain_gcs':brain_path or None}
    report,ledger,calibration=run_historical(v4,v42,brain_raw=brains,source_info=source)
    # These V5 candidates are tested against the fixed-state V4 reference.
    # The winner comparison is preserved in the V4.2 audit, but this engine
    # must not falsely describe STATE vs A as a champion-vs-challenger test.
    winner=src_report['frozen_champion']
    report['actual_v4_research_winner']=winner
    report['v42_actual_v4_winner_parity']='EXACT_ALL_NINE_FOLDS'
    report['v5_candidates_include_actual_v4_winner']=bool(winner in MODELS)
    report['v5_candidate_comparison_benchmark']=BASE
    report['v5_research_selection_scope']='FIXED_STATE_VS_ENGINEERED_CANDIDATES__ACTUAL_V4_WINNER_AUDITED_SEPARATELY'
    report['limits'].append('This V5 candidate ledger does not automatically establish improvement over the actual V4 research winner.')
    runid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    base=f'research/ncaaf/v5_market_value/runs/{runid}'
    artifacts={f'{base}/report.json':(json.dumps(report,indent=2,allow_nan=False), 'application/json'),
        f'{base}/research_signals_not_bets.csv':(ledger.to_csv(index=False),'text/csv'),
        f'{base}/calibration_2026_research_only.json':(json.dumps(calibration,allow_nan=False),'application/json')}
    # versioned writes first; latest pointer only after all artifacts succeeded
    for name,(value,ctype) in artifacts.items():
        bucket.blob(name).upload_from_string(value,content_type=ctype,if_generation_match=0)
    report['report_uri']=f'gs://{bucket_name}/{base}/report.json'
    report['signals_uri']=f'gs://{bucket_name}/{base}/research_signals_not_bets.csv'
    report['calibration_uri']=f'gs://{bucket_name}/{base}/calibration_2026_research_only.json'
    bucket.blob('research/ncaaf/v5_market_value/latest_report.json').upload_from_string(
        json.dumps(report,indent=2,allow_nan=False),content_type='application/json')
    log_func(f"[NCAAF-V5] status={report['status']} historical_games={report['historical_games']} "
             f"signals={len(ledger)} brains={report['brain_family_status']['pregame_records_kept']} "
             'production_authority=0 ROI=UNAVAILABLE')
    return report


def main():
    import argparse
    from pathlib import Path
    a=argparse.ArgumentParser()
    a.add_argument('--v4',required=True);a.add_argument('--v42',required=True)
    a.add_argument('--brains');a.add_argument('--output-dir',required=True)
    args=a.parse_args();o=Path(args.output_dir);o.mkdir(parents=True,exist_ok=True)
    brains=pd.read_csv(args.brains) if args.brains else None
    report,signals,calibration=run_historical(pd.read_csv(args.v4),pd.read_csv(args.v42),brain_raw=brains)
    (o/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (o/'calibration_2026_research_only.json').write_text(json.dumps(calibration,allow_nan=False))
    signals.to_csv(o/'research_signals_not_bets.csv',index=False)
    print('V5',report['status'],'SIGNALS',len(signals))

if __name__=='__main__':main()
