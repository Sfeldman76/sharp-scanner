"""NCAAF V5 prospective research-only shadow snapshots.

This is a *consumer* of independently fitted and recorded V4/A predictions;
it DOES NOT reconstruct live state ratings or invent predictions. Historical
calibration is empirical and research-only pending source-vintage certification.
Outputs are written exactly once (GCS if_generation_match=0) by the caller.
"""
from __future__ import annotations
from datetime import datetime, timezone
from hashlib import sha256
import json, math
import numpy as np
import pandas as pd

ALLOWED=('NONLINEAR_STATE_HGB','A_EFFICIENCY')
BRAIN_FAMILIES=('BIG_AL','PATHI','PREDICTION_TRACKER','SYSTEM_MINER','STATE_BOUNCE_BACK')
VERSION='NCAAF_V5_SHADOW_LOCK_V1_20261020'


def _ts(s):return pd.to_datetime(s,utc=True,errors='coerce')

def _american_profit(odds):
    if odds>0:return odds/100.
    if odds<0:return 100./(-odds)
    raise ValueError('INVALID_ZERO_AMERICAN_ODDS')


def _require(df,names,label):
    missing=set(names)-set(df.columns)
    if missing:raise ValueError(f'{label}_MISSING:{sorted(missing)}')


def _sha(s):return isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdefABCDEF' for c in s)


def build_shadow(predictions, quotes, calibration, brains=None):
    """Build research observations, never authorized BET/STRONG BET votes."""
    _require(predictions,['Game_ID','Season','Team','Opponent','Model','Model_Margin',
        'Forecast_AsOf_UTC','Feature_Cutoff_UTC','Kickoff_UTC','Model_Artifact_SHA256',
        'Feature_Source_SHA256','Feature_Vintage_Certified'], 'PREDICTIONS')
    _require(quotes,['Game_ID','Season','Team','Opponent','Quote_AsOf_UTC','Kickoff_UTC',
        'Book','Market_Team_Margin','Team_American_Odds','Opponent_American_Odds','Quote_Source_SHA256'],'QUOTES')
    pred=predictions.copy();q=quotes.copy()
    for df,name in ((pred,'PREDICTIONS'),(q,'QUOTES')):
        df['Season']=pd.to_numeric(df['Season'],errors='coerce')
        if df.Season.isna().any() or (df.Season<2026).any():raise ValueError(name+'_NOT_PROSPECTIVE')
        df['Game_ID']=df.Game_ID.astype('string').str.strip()
        if df.Game_ID.isna().any() or (df.Game_ID=='').any():raise ValueError(name+'_NULL_GAME_ID')
        for col in ('Team','Opponent'):df[col]=df[col].astype(str).str.strip().str.lower()
        if (df.Team==df.Opponent).any():raise ValueError(name+'_MIRROR_SAME_TEAM')
        for col in ('Kickoff_UTC',):
            df[col]=_ts(df[col]);
            if df[col].isna().any():raise ValueError(name+'_INVALID_KICKOFF')
    for col in ('Forecast_AsOf_UTC','Feature_Cutoff_UTC'):
        pred[col]=_ts(pred[col])
        if pred[col].isna().any():raise ValueError('PREDICTION_BAD_TIMESTAMP')
    q['Quote_AsOf_UTC']=_ts(q['Quote_AsOf_UTC'])
    if q.Quote_AsOf_UTC.isna().any():raise ValueError('QUOTE_BAD_TIMESTAMP')
    if (pred.Forecast_AsOf_UTC>=pred.Kickoff_UTC).any() or (pred.Feature_Cutoff_UTC>pred.Forecast_AsOf_UTC).any():
        raise ValueError('PREDICTION_FUTURE_LEAKAGE')
    if (q.Quote_AsOf_UTC>=q.Kickoff_UTC).any():raise ValueError('QUOTE_POST_KICKOFF')
    if (~pred.Model.isin(ALLOWED)).any():raise ValueError('UNKNOWN_LIVE_MODEL')
    if pred.duplicated(['Game_ID','Season','Model']).any():raise ValueError('DUPLICATE_MODEL_GAME')
    if q.duplicated(['Game_ID','Season']).any():raise ValueError('DUPLICATE_QUOTE_GAME')
    for col,df in [('Model_Artifact_SHA256',pred),('Feature_Source_SHA256',pred),('Quote_Source_SHA256',q)]:
        if not df[col].astype(str).map(_sha).all():raise ValueError('SOURCE_SHA256_REQUIRED:'+col)
    for c in ('Model_Margin',):
        pred[c]=pd.to_numeric(pred[c],errors='coerce')
        if not np.isfinite(pred[c]).all():raise ValueError('INVALID_MARGIN_PREDICTION')
    for c in ('Market_Team_Margin','Team_American_Odds','Opponent_American_Odds'):
        q[c]=pd.to_numeric(q[c],errors='coerce')
        if not np.isfinite(q[c]).all():raise ValueError('INVALID_QUOTE:'+c)
    if (q[['Team_American_Odds','Opponent_American_Odds']].abs()<100).any().any():
        raise ValueError('INVALID_AMERICAN_ODDS_RANGE')
    if any(s not in (True,False,'true','false','True','False','TRUE','FALSE','1','0',0,1)
           for s in pred.Feature_Vintage_Certified):
        raise ValueError('INVALID_CERTIFICATION_FLAG')
    merged=pred.merge(q,on=['Game_ID','Season'],how='left',suffixes=('','_QUOTE'),validate='many_to_one',indicator=True)
    if (merged._merge!='both').any():raise ValueError('MISSING_PREGAME_QUOTES')
    if (merged.Team!=merged.Team_QUOTE).any() or (merged.Opponent!=merged.Opponent_QUOTE).any():
        raise ValueError('QUOTE_TEAM_ORIENTATION_MISMATCH')
    if (merged.Kickoff_UTC!=merged.Kickoff_UTC_QUOTE).any():raise ValueError('QUOTE_KICKOFF_MISMATCH')
    if (merged.Quote_AsOf_UTC<merged.Forecast_AsOf_UTC).any():
        raise ValueError('QUOTE_PREDATES_FORECAST_FOR_DECISION')
    if 'model_calibration' not in calibration:raise ValueError('INVALID_V5_CALIBRATION_ARTIFACT')
    rows=[]
    for r in merged.itertuples(index=False):
        spec=calibration['model_calibration'].get(r.Model)
        if not spec or spec.get('type')!='EMPIRICAL_OOF_MARGIN_ERROR_CDF':
            raise ValueError('CALIBRATION_MISSING_MODEL:'+r.Model)
        errors=np.asarray(spec['residuals'],float)
        if len(errors)<400 or not np.isfinite(errors).all():raise ValueError('CALIBRATION_INSUFFICIENT')
        margin=float(r.Market_Team_Margin)
        pteam=float(np.clip((len(errors)-np.searchsorted(np.sort(errors),margin-float(r.Model_Margin),side='right')+0.5)/(len(errors)+1),.01,.99))
        side='TEAM' if float(r.Model_Margin)>margin else 'OPPONENT'
        gap=abs(float(r.Model_Margin)-margin)
        p=pteam if side=='TEAM' else 1-pteam
        price=float(r.Team_American_Odds if side=='TEAM' else r.Opponent_American_Odds)
        profit=_american_profit(price)
        ev=p*profit-(1-p)
        certified=str(r.Feature_Vintage_Certified).lower() in ('true','1')
        # Cannot use source uncertified research probabilities as live bet authority.
        label=('RESEARCH_EDGE_CANDIDATE__CALIBRATION_VINTAGE_HOLD' if certified and gap>=2 and ev>=.03
               else 'SHADOW_HOLD_UNCERTIFIED' if not certified else 'SHADOW_NO_EDGE')
        rows.append({'Season':int(r.Season),'Game_ID':str(r.Game_ID),'Team':r.Team,
            'Opponent':r.Opponent,'Model':r.Model,'Forecast_AsOf_UTC':r.Forecast_AsOf_UTC.isoformat(),
            'Quote_AsOf_UTC':r.Quote_AsOf_UTC.isoformat(),'Kickoff_UTC':r.Kickoff_UTC.isoformat(),
            'Model_Margin':float(r.Model_Margin),'Market_Team_Margin':margin,
            'Gap_Points':gap,'Side':side,'Book':r.Book,'Odds_American':price,
            'P_Cover_Empirical':p,'Break_Even_P':1/(1+profit),'EV_Per_Unit_Estimated':ev,
            'Forecast_Model_SHA256':r.Model_Artifact_SHA256,
            'Feature_Source_SHA256':r.Feature_Source_SHA256,
            'Quote_Source_SHA256':r.Quote_Source_SHA256,
            'Source_Vintage_Certified':certified,'Research_Status':label,
            'Bet_Authority':0,'Historical_Calibration_Vintage':'SOURCE_HOLD'})
    out=pd.DataFrame(rows)
    # Historical rule/system strength requires separately validated prospective
    # evidence. The optional brain snapshot is recorded, NOT voted automatically.
    summary={'version':VERSION,'status':'PREGAME_SHADOW_OBSERVATIONS_ONLY',
        'games':int(out[['Season','Game_ID']].drop_duplicates().shape[0]),
        'observations':len(out),'brain_snapshot_count':0,
        'no_auto_bet_authority':True,'no_prod_mutation':True,
        'research_model_calibration_source_vintage_hold':True,
        'quotes_are_observed_not_proven_executable':True}
    if brains is not None and len(brains):
        _require(brains,['Game_ID','Season','Family','System_ID','Side','Snapshot_UTC','Kickoff_UTC','Source_Artifact_SHA256'],'BRAINS')
        bs=brains.copy()
        bs['Snapshot_UTC']=_ts(bs.Snapshot_UTC);bs['Kickoff_UTC']=_ts(bs.Kickoff_UTC)
        if bs[['Snapshot_UTC','Kickoff_UTC']].isna().any().any() or (bs.Snapshot_UTC>=bs.Kickoff_UTC).any():
            raise ValueError('BRAIN_NOT_PREGAME')
        if (~bs.Family.isin(BRAIN_FAMILIES)).any() or (~bs.Side.isin(['TEAM','OPPONENT'])).any():
            raise ValueError('UNKNOWN_BRAIN')
        if not bs.Source_Artifact_SHA256.astype(str).map(_sha).all():raise ValueError('BRAIN_UNHASHED')
        valid_keys=set(zip(out.Season.astype(int),out.Game_ID.astype(str)))
        if any((int(a),str(b)) not in valid_keys for a,b in zip(bs.Season,bs.Game_ID)):
            raise ValueError('BRAIN_UNKNOWN_GAME')
        summary['brain_snapshot_count']=int(len(bs))
        # Attach counts for research diagnostics; scorecard voting remains gated.
        mapped={}
        for r in bs.itertuples(index=False):mapped.setdefault((int(r.Season),str(r.Game_ID)),[]).append(r)
        out['Brain_Confirm_N']=[sum(x.Side==r.Side for x in mapped.get((int(r.Season),str(r.Game_ID)),[])) for r in out.itertuples(index=False)]
        out['Brain_Oppose_N']=[sum(x.Side!=r.Side for x in mapped.get((int(r.Season),str(r.Game_ID)),[])) for r in out.itertuples(index=False)]
    else:
        out['Brain_Confirm_N']=pd.NA;out['Brain_Oppose_N']=pd.NA
    return summary,out


def immutable_gcs_lock(*,bucket,report,ledger,run_id):
    """No overwrite or upsert: same run-id rejects second call at GCS level."""
    if not run_id or any(c not in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-' for c in run_id):
        raise ValueError('UNSAFE_RUN_ID')
    path=f'research/ncaaf/v5_market_value/prospective_shadow/locks/{run_id}'
    report=dict(report);report['generated_utc']=datetime.now(timezone.utc).isoformat()
    # Publish report with generation 0; no backfill of previous locked forecasts.
    obs_csv=ledger.to_csv(index=False)
    report['observations_sha256']=sha256(obs_csv.encode('utf-8')).hexdigest()
    # Manifest is the commit marker: write the immutable CSV first.
    bucket.blob(f'{path}/observations.csv').upload_from_string(obs_csv,
        content_type='text/csv',if_generation_match=0)
    bucket.blob(f'{path}/manifest.json').upload_from_string(json.dumps(report,allow_nan=False),
        content_type='application/json',if_generation_match=0)
    return f'gs://{bucket.name}/{path}'


def main():
    from pathlib import Path
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',required=True);p.add_argument('--quotes',required=True)
    p.add_argument('--calibration',required=True);p.add_argument('--brains');p.add_argument('--output-dir',required=True)
    p.add_argument('--gcs-bucket',help='Optional: upload immutable research lock to this bucket')
    p.add_argument('--run-id',help='Required with --gcs-bucket. Never reuse.')
    a=p.parse_args();o=Path(a.output_dir);o.mkdir(parents=True,exist_ok=True)
    r,l=build_shadow(pd.read_csv(a.predictions),pd.read_csv(a.quotes),
        json.loads(Path(a.calibration).read_text()),pd.read_csv(a.brains) if a.brains else None)
    (o/'shadow_manifest.json').write_text(json.dumps(r,indent=2))
    l.to_csv(o/'shadow_observations_not_bets.csv',index=False)
    if a.gcs_bucket:
        if not a.run_id:raise ValueError('RUN_ID_REQUIRED_FOR_IMMUTABLE_GCS_LOCK')
        from google.cloud import storage
        destination=immutable_gcs_lock(bucket=storage.Client().bucket(a.gcs_bucket),
            report=r,ledger=l,run_id=a.run_id)
        print('IMMUTABLE_GCS_LOCK:',destination)
    print(r)

if __name__=='__main__':main()
