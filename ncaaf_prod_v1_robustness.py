"""Read-only NCAAF Production V1 robustness diagnostics.

Analyzes ACTUAL locked pregame predictions, not reconstructed or manufactured
historical selections. Claims about 2013-2025 performance are impossible unless
identical point-in-time replay evidence is independently built and validated.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import math
import numpy as np
import pandas as pd

VERSION='NCAAF_PRODUCTION_V1_PROSPECTIVE_ROBUSTNESS_READ_ONLY_20261020'
PRED='sharplogger.sharp_data.ncaaf_production_v1_predictions'
RESULT='sharplogger.sharp_data.ncaaf_production_v1_results'
ACTIONS=('BET','STRONG BET','PLAY','STRONG PLAY')
REQUIRED=('prediction_event_id','game_start','prediction_timestamp','quote_timestamp',
          'physical_game_id','market','action','odds','model_probability',
          'quote_executable','is_prospective','lock_type','result','profit_per_unit',
          'model_version','artifact_sha256')


def _wilson(wins,n,z=1.9599639845):
    if n<=0:return None
    p=wins/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    spread=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(center-spread),float(center+spread)]


def _moneyline_implied(o):
    a=pd.to_numeric(o,errors='coerce').to_numpy(dtype=float)
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.where(a<0,(-a)/(-a+100),np.where(a>0,100/(a+100),np.nan))


def audit(raw):
    missing=[c for c in REQUIRED if c not in raw]
    if missing: return {'status':'NO_SUPPORTED_PRODUCTION_SCHEMA','missing_columns':missing,'production_modified':False}
    df=raw.copy()
    if df.prediction_event_id.isna().any() or df.prediction_event_id.astype(str).duplicated().any():
        raise RuntimeError('PRODUCTION_AUDIT_DUPLICATED_LOCK_IDS')
    for c in ('game_start','prediction_timestamp','quote_timestamp'):
        df[c]=pd.to_datetime(df[c],utc=True,errors='coerce')
    eligible = (df.lock_type.eq('FIRST') & df.is_prospective.eq(True) &
                (df.prediction_timestamp<df.game_start) & (df.quote_timestamp<df.game_start))
    invalid = int((~eligible).sum())
    df=df.loc[eligible].copy()
    grades = df.result.astype(str).str.upper().str.strip()
    decided = grades.isin(['WIN','LOSS','PUSH'])
    df=df.loc[decided].copy()
    df['result']=df.result.astype(str).str.upper().str.strip()
    selected = df.action.astype(str).str.upper().isin(ACTIONS) & df.quote_executable.eq(True)
    bets=df.loc[selected].copy()
    bets['profit_per_unit']=pd.to_numeric(bets.profit_per_unit,errors='coerce')
    bets['model_probability']=pd.to_numeric(bets.model_probability,errors='coerce')
    bets['break_even']=_moneyline_implied(bets.odds)
    bets=bets.loc[bets.profit_per_unit.notna()].copy()
    bets['date']=bets.game_start.dt.strftime('%Y-%m-%d')
    bets['season']=bets.game_start.dt.year
    bets['week']=bets.game_start.dt.tz_convert(None).dt.to_period('W-SUN').astype(str)
    def summarize(a):
        n=len(a);nonpush=a.result.ne('PUSH');wins=int(a.result.eq('WIN').sum());n2=int(nonpush.sum())
        profit=float(a.profit_per_unit.sum()) if n else 0.
        return {'bets':n,'wins':wins,'losses':int(a.result.eq('LOSS').sum()),
                'pushes':int(a.result.eq('PUSH').sum()),
                'decisions':n2,'hit_rate':float(wins/n2) if n2 else None,
                'hit_rate_wilson95':_wilson(wins,n2),
                'units':profit,'roi_per_unit':float(profit/n) if n else None,
                'mean_book_implied_break_even':float(a.break_even.mean()) if n and a.break_even.notna().any() else None,
                'market_implied_vs_realized_hit_rate_note':'Book implied probabilities include vig; descriptive only.'}
    by_market={str(k):summarize(g) for k,g in bets.groupby('market')}
    by_year={str(k):summarize(g) for k,g in bets.groupby('season')}
    by_week={str(k):summarize(g) for k,g in bets.groupby('week')}
    by_artifact={str(k):summarize(g) for k,g in bets.groupby('artifact_sha256')}
    # Whole-day bootstrap of portfolio ROI, not IID per-bet draws.
    portfolio=summarize(bets)
    if len(bets)>=20 and bets.date.nunique()>=10:
        day=bets.groupby('date').profit_per_unit.agg(['sum','count'])
        totals=day['sum'].to_numpy(dtype=float);counts=day['count'].to_numpy(dtype=float)
        rng=np.random.default_rng(20261020);k=len(day);arr=np.empty(2000)
        for i in range(len(arr)):
            ix=rng.integers(k,size=k)
            arr[i]=totals[ix].sum()/counts[ix].sum()
        portfolio['roi_game_date_bootstrap_ci95']=[float(q) for q in np.quantile(arr,[.025,.975])]
        portfolio['roi_bootstrap_probability_nonpositive']=float((arr<=0).mean())
        portfolio['date_clusters']=k
    else:
        portfolio['roi_game_date_bootstrap_ci95']=None
        portfolio['roi_uncertainty_status']='TOO_FEW_BETS_OR_DATES'
    dates = [x.isoformat() for x in bets.game_start if pd.notna(x)]
    return {'version':VERSION,'status':'PROSPECTIVE_LEDGER_ANALYSIS_ONLY',
            'generated_utc':datetime.now(timezone.utc).isoformat(),
            'source_tables':[PRED,RESULT],'observed_locked_rows':len(df),
            'excluded_non_prospective_or_non_first':invalid,
            'graded_executable_bets':len(bets),'portfolio':portfolio,
            'by_market':by_market,'by_season':by_year,'by_week':by_week,
            'by_artifact_hash':by_artifact,
            'first_bet_kickoff':min(dates) if dates else None,
            'last_bet_kickoff':max(dates) if dates else None,
            'historical_replay_validated':False,
            'historical_2013_2025_performance':'NOT_MEASURABLE_FROM_2026_PROSPECTIVE_LEDGER',
            'caveats':['Prospective tracked bets only, selected actions, not all possible games.',
                       'One immutable FIRST lock per game-market; subsequent line updates excluded.',
                       'Possible model versions / artifact hashes must be evaluated separately.',
                       'Short samples, correlated selections, multiple testing and vig can make historical profitability appear lucky.',
                       'No claim of performance for pre-deployment seasons.',
                       'No resimulation of old historical games with current features or quotes.'],
            'production_modified':False,'bet_authority':0}


def read_bigquery(*,project='sharplogger'):
    from google.cloud import bigquery
    cli=bigquery.Client(project=project)
    sql=f'''WITH results AS (
      SELECT prediction_event_id, result, profit_per_unit, settled_at,
             ROW_NUMBER() OVER (PARTITION BY prediction_event_id ORDER BY settled_at DESC) AS rn
      FROM `{RESULT}`
    )
    SELECT p.prediction_event_id,p.game_start,p.prediction_timestamp,p.quote_timestamp,
      p.physical_game_id,p.market,p.action,p.odds,p.model_probability,p.quote_executable,
      p.is_prospective,p.lock_type,p.model_version,p.artifact_sha256,
      r.result,r.profit_per_unit
    FROM `{PRED}` p
    LEFT JOIN results r ON p.prediction_event_id=r.prediction_event_id AND r.rn=1
    WHERE p.lock_type='FIRST' AND p.is_prospective=TRUE
    '''
    return cli.query(sql).to_dataframe(create_bqstorage_client=False)


def run_cloud_audit(*,storage_client,bucket_name='sharp-models',log_func=print):
    bucket=storage_client.bucket(bucket_name)
    try:
        df=read_bigquery()
        out=audit(df)
    except Exception as exc:
        out={'version':VERSION,'status':'PRODUCTION_LEDGER_UNAVAILABLE',
             'reason':f'{type(exc).__name__}:{exc}',
             'production_modified':False,'bet_authority':0}
    # Separate namespace and optional status: V4.3 core audit NEVER passes/fails
    # based on production grading connection availability.
    t=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    payload=json.dumps(out,indent=2,sort_keys=True,allow_nan=False,default=str)
    bucket.blob(f'research/ncaaf/production_v1_robustness/runs/{t}/report.json').upload_from_string(payload,content_type='application/json')
    bucket.blob('research/ncaaf/production_v1_robustness/latest_report.json').upload_from_string(payload,content_type='application/json')
    log_func('[NCAAF-PROD-ROBUSTNESS] status={} authority=0'.format(out['status']))
    return out


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    df=pd.read_csv(a.csv,low_memory=False)
    with open(a.output,'w') as f:json.dump(audit(df),f,indent=2,allow_nan=False)
