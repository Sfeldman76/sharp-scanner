"""NCAAF Heavy Research: primary chronological V5.5 edge -> ATS -> brain value.

No production mutation and no automatic policy promotion. Real NONLINEAR_HGB
historical folds come only from ncaaf_v552_live's own replay adapter or a
checksummed, independently audited input manifest. Legacy NCAAF CORE OOF is
never accepted as a fallback.
"""
from __future__ import annotations

import hashlib
import importlib
import io
import json
from pathlib import Path
from itertools import combinations
from datetime import datetime, timezone
from typing import Any

import pandas as pd



# Consolidated ATS evaluator definitions; previously v55_ats_evaluator.py.
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

MODEL = 'NONLINEAR_HGB'
FAMILIES = ('BIG_AL', 'PATHI', 'MINER', 'STAT', 'PT')
BUCKETS = ((0,2,'0-<2'),(2,4,'2-<4'),(4,6,'4-<6'),(6,8,'6-<8'),(8,float('inf'),'8+'))
THRESHOLDS = (0,1,2,3,4,5,6,7,8,10)
OOS_YEARS = (2024, 2025)

class ProvenanceError(ValueError):
    pass

def _require(df, fields, source):
    missing = sorted(set(fields)-set(df.columns))
    if missing:
        raise ProvenanceError(f'{source}: missing required fields: {missing}')

def _dt(df, field, source):
    r=pd.to_datetime(df[field],utc=True,errors='coerce')
    if r.isna().any():
        raise ProvenanceError(f'{source}: {int(r.isna().sum())} rows missing/invalid {field}')
    return r

def _bool_true(series):
    return series.astype(str).str.lower().isin(['true','1','yes','pass'])

def _wilson(w,n,z=1.96):
    if n <= 0: return (None,None)
    p=w/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(c-h,4),round(c+h,4))

def _edge_bucket(v):
    for lo,hi,name in BUCKETS:
        if lo <= v < hi: return name
    return 'INVALID'

def _roi_one(odds, won):
    if won==0:return 0.0
    if won<0:return -1.0
    if pd.isna(odds) or odds==0:return np.nan
    return float(odds)/100 if odds>0 else 100/(-float(odds))

def prepare_predictions(pred, *, allow_retrospective_recipe=False):
    required=['game_id','season','model_id','model_home_margin','prediction_at_utc',
              'training_end_utc','model_recipe_sha256','recipe_selected_at_utc','fold_id','true_oos','pregame_features_verified']
    _require(pred,required,'predictions')
    a=pred.copy();a['game_id']=a['game_id'].astype(str)
    if a['game_id'].duplicated().any(): raise ProvenanceError('predictions: duplicate game_id')
    if not a['model_id'].astype(str).eq(MODEL).all():
        raise ProvenanceError('predictions: contains model other than V5.5 NONLINEAR_HGB')
    if not _bool_true(a['true_oos']).all():
        raise ProvenanceError('predictions: unverified true_oos fold(s)')
    if not _bool_true(a['pregame_features_verified']).all():
        raise ProvenanceError('predictions: missing as-of feature verification')
    if a['model_recipe_sha256'].isna().any() or a['model_recipe_sha256'].astype(str).str.len().lt(12).any():
        raise ProvenanceError('predictions: model recipe identity not recorded')
    if a['model_recipe_sha256'].astype(str).nunique()!=1:
        raise ProvenanceError('predictions: V5.5 recipe differs across folds; analyze separate versioned studies')
    if a['fold_id'].isna().any(): raise ProvenanceError('predictions: fold identifier missing')
    a['prediction_time']=_dt(a,'prediction_at_utc','predictions')
    a['recipe_selection_time']=_dt(a,'recipe_selected_at_utc','predictions')
    if not allow_retrospective_recipe and (a['recipe_selection_time']>a['prediction_time']).any():
        raise ProvenanceError('predictions: V5.5 recipe chosen using results after prediction date')
    a['train_end_time']=_dt(a,'training_end_utc','predictions')
    if not (a['train_end_time']<a['prediction_time']).all():
        raise ProvenanceError('predictions: training includes prediction-date or future information')
    a['model_home_margin']=pd.to_numeric(a['model_home_margin'],errors='coerce')
    if not np.isfinite(a['model_home_margin']).all():
        raise ProvenanceError('predictions: nonfinite V5.5 predicted margin')
    a['season']=pd.to_numeric(a['season'],errors='raise').astype(int)
    if ((a['season']<2013)|(a['season']>2025)).any():
        raise ProvenanceError('predictions: outside sealed research 2013-2025 seasons')
    if not allow_retrospective_recipe and (a.loc[a.season.isin(OOS_YEARS),'recipe_selection_time']>=pd.Timestamp('2024-01-01',tz='UTC')).any():
        raise ProvenanceError('predictions: 2024-25 confirmation recipe was not frozen pre-2024')
    return a

def prepare_markets(markets,preferred_book=None):
    required=['game_id','season','kickoff_utc','home_team','away_team',
              'home_score','away_score','home_spread','home_odds','away_odds',
              'book','quote_at_utc','market_pregame_verified']
    _require(markets,required,'games/markets')
    m=markets.copy();m['game_id']=m['game_id'].astype(str)
    if preferred_book:
        m=m[m['book'].astype(str).str.lower().eq(preferred_book.lower())].copy()
    if m['game_id'].duplicated().any():
        raise ProvenanceError('markets: >1 market per game. Precommit a book and a fixed observation cutoff; do not select hindsight-best lines.')
    if not _bool_true(m['market_pregame_verified']).all():
        raise ProvenanceError('markets: unknown original quote vintage')
    m['kickoff_time']=_dt(m,'kickoff_utc','markets')
    m['quote_time']=_dt(m,'quote_at_utc','markets')
    if not (m['quote_time']<m['kickoff_time']).all():
        raise ProvenanceError('markets: at or after kickoff quote')
    for c in ('home_score','away_score','home_spread','home_odds','away_odds'):
        m[c]=pd.to_numeric(m[c],errors='coerce')
        if not np.isfinite(m[c]).all():raise ProvenanceError(f'markets: nonfinite {c}')
    if ((m.home_odds.abs()<100)|(m.away_odds.abs()<100)).any():
        raise ProvenanceError('markets: invalid American odds')
    m['season']=pd.to_numeric(m['season'],errors='raise').astype(int)
    return m

def merge_scored(pred, markets, *, preferred_book=None, max_quote_age_hours=168, allow_retrospective_recipe=False):
    p=prepare_predictions(pred,allow_retrospective_recipe=allow_retrospective_recipe); m=prepare_markets(markets,preferred_book)
    d=p.merge(m,on=['game_id','season'],how='inner',validate='one_to_one')
    if d.empty: raise ProvenanceError('no matched V5.5 predictions + historical games')
    # Pre-game original model prediction; no lookahead to result or closing after a decision.
    if not (d['prediction_time']<d['kickoff_time']).all():
        raise ProvenanceError('prediction at or after kickoff')
    # Actual decision time is max(prediction, quoted market), provided both pregame.
    d['decision_time']=d[['prediction_time','quote_time']].max(axis=1)
    if ((d['decision_time']>=d['kickoff_time'])|
        ((d['decision_time']-d['quote_time'])>pd.Timedelta(hours=max_quote_age_hours))).any():
        raise ProvenanceError('market not available as of prediction decision or too old')
    # Home spread from HOME perspective: -6.5 means home gives 6.5 points.
    d['home_point_edge']=d['model_home_margin']+d['home_spread']
    d['pick_side']=np.where(d.home_point_edge>0,'HOME',np.where(d.home_point_edge<0,'AWAY','NONE'))
    d=d[d.pick_side.ne('NONE')].copy()
    if d.empty: raise ProvenanceError('all V5.5 predicted edges are exactly zero')
    d['edge_points']=d.home_point_edge.abs()
    d['edge_bucket']=d.edge_points.apply(_edge_bucket)
    d['ats_home_margin']=d.home_score-d.away_score+d.home_spread
    d['ats_result']=np.where(d.ats_home_margin==0,0,
                   np.where(((d.ats_home_margin>0)&d.pick_side.eq('HOME'))|
                            ((d.ats_home_margin<0)&d.pick_side.eq('AWAY')),1,-1))
    d['pick_odds']=np.where(d.pick_side.eq('HOME'),d.home_odds,d.away_odds)
    d['profit_per_unit']=[_roi_one(o,int(r)) for o,r in zip(d.pick_odds,d.ats_result)]
    if allow_retrospective_recipe:
        d['study_period']=np.where(d.season<=2023,'RETRO_RECIPE_DISCOVERY',
                np.where(d.season==2024,'RETRO_RECIPE_2024','RETRO_RECIPE_2025'))
    else:
        d['study_period']=np.where(d.season<=2023,'DISCOVERY_OOS',np.where(d.season==2024,'HOLDOUT_2024','HOLDOUT_2025'))
    return d

def _summarize(g):
    n=int(len(g)); wins=int((g.ats_result==1).sum());losses=int((g.ats_result==-1).sum()); pushes=int((g.ats_result==0).sum())
    decided=wins+losses;lo,hi=_wilson(wins,decided)
    profit=float(g.profit_per_unit.sum()) if n else 0.0
    return {'games':n,'wins':wins,'losses':losses,'pushes':pushes,
            'ats_win_pct':round(100*wins/decided,2) if decided else None,
            'wilson95_low_pct':round(100*lo,2) if lo is not None else None,
            'wilson95_high_pct':round(100*hi,2) if hi is not None else None,
            'roi_pct':round(100*profit/n,2) if n else None,
            'units_net':round(profit,3)}

def summarize_grouped(df, groupcols):
    records=[]
    if df.empty:return pd.DataFrame(columns=groupcols+list(_summarize(df).keys()))
    for group,g in df.groupby(groupcols,dropna=False,sort=True):
        if not isinstance(group,tuple):group=(group,)
        records.append({**dict(zip(groupcols,group)),**_summarize(g)})
    return pd.DataFrame.from_records(records)

def prepare_brains(brains,scored):
    _require(brains,['game_id','family','system_id','signal_side','fired_at_utc',
                     'rule_frozen_at_utc','pregame_signal_verified','historical_rule_freeze_verified'],
             'brains')
    b=brains.copy();b['game_id']=b.game_id.astype(str)
    b['family']=b.family.astype(str).str.upper()
    bad=b[~b.family.isin(FAMILIES)]
    if not bad.empty:raise ProvenanceError(f'brains: unknown family: {bad.family.unique().tolist()}')
    b['signal_side']=b.signal_side.astype(str).str.upper()
    if (~b.signal_side.isin(['HOME','AWAY'])).any():raise ProvenanceError('brains: unknown side')
    b['signal_time']=_dt(b,'fired_at_utc','brains')
    b['rule_frozen_time']=_dt(b,'rule_frozen_at_utc','brains')
    if not _bool_true(b.pregame_signal_verified).all():
        raise ProvenanceError('brains: missing as-of feature/source validation')
    if not _bool_true(b.historical_rule_freeze_verified).all():
        raise ProvenanceError('brains: rule was not frozen before evaluation period')
    b=b.merge(scored[['game_id','decision_time','kickoff_time','pick_side','season']],on='game_id',how='inner',validate='many_to_one')
    if ((b.signal_time>b.decision_time)|(b.signal_time>=b.kickoff_time)|
        (b.rule_frozen_time>b.decision_time)).any():
        raise ProvenanceError('brains: signal or rule discovered after actual game decision time')
    # Holdout seasons require discovery hypotheses frozen prior to Jan 1 2024.
    # 2024 and 2025 are *never* used to mine new confirmation rules.
    if (b.loc[b.season.isin(OOS_YEARS),'rule_frozen_time']>=pd.Timestamp('2024-01-01',tz='UTC')).any():
        raise ProvenanceError('brains: 2024-25 confirmation uses post-2023 rule discovery')
    b['direction']=np.where(b.signal_side.eq(b.pick_side),'SUPPORT','OPPOSE')
    return b

def attach_brains(scored,brain_events):
    if brain_events is None or brain_events.empty:
        d=scored.copy();d['support_families']='';d['oppose_families']='';d['support_count']=0;d['oppose_count']=0
        return d
    b=prepare_brains(brain_events,scored)
    stats={}
    for key,g in b.groupby('game_id',sort=False):
        # Multiple correlated rules within a family count once. Opposing rules
        # within a single family are ambiguous, not independent support.
        support=set(g.loc[g.direction=='SUPPORT','family']);oppose=set(g.loc[g.direction=='OPPOSE','family'])
        ambiguous=support&oppose
        support-=ambiguous;oppose-=ambiguous
        stats[key]={'support_families':';'.join(sorted(support)),
                    'oppose_families':';'.join(sorted(oppose)),
                    'support_count':len(support),'oppose_count':len(oppose),
                    'ambiguous_families':';'.join(sorted(ambiguous))}
    d=scored.copy();d['support_families']=d.game_id.map(lambda x:stats.get(x,{}).get('support_families',''))
    d['oppose_families']=d.game_id.map(lambda x:stats.get(x,{}).get('oppose_families',''))
    d['support_count']=d.game_id.map(lambda x:stats.get(x,{}).get('support_count',0))
    d['oppose_count']=d.game_id.map(lambda x:stats.get(x,{}).get('oppose_count',0))
    d['ambiguous_families']=d.game_id.map(lambda x:stats.get(x,{}).get('ambiguous_families',''))
    return d

def _hasfam(s,f):return f in str(s).split(';')

def _fisher_pvalue(w1,l1,w0,l0):
    # Only distinct games and wins/losses are included; pushes are excluded.
    if min(w1+l1,w0+l0)<10:return None
    try:
        from scipy.stats import fisher_exact
        return float(fisher_exact([[w1,l1],[w0,l0]],alternative='greater').pvalue)
    except ImportError:
        return None

def _bh_qvalues(values):
    finite=[(i,float(v)) for i,v in enumerate(values) if v is not None and not pd.isna(v)]
    out=[None]*len(values);m=len(finite);prev=1.0
    for rank,(idx,val) in enumerate(sorted(finite,key=lambda x:x[1],reverse=True)):
        q=min(prev,val*m/(m-rank));out[idx]=round(q,6);prev=q
    return out

def research_tables(df):
    edged=summarize_grouped(df,['study_period','edge_bucket'])
    eyr=summarize_grouped(df,['season','edge_bucket'])
    threshold=[]
    for (period,season),g in df.groupby(['study_period','season'],sort=True):
        for t in THRESHOLDS:
            x=g[g.edge_points>=t]
            threshold.append({'study_period':period,'season':season,'min_edge_points':t,**_summarize(x)})
    combos=[]
    for (period,bucket),g in df.groupby(['study_period','edge_bucket'],sort=True):
        for family in FAMILIES:
            f=g[g.support_families.map(lambda x:_hasfam(x,family))]
            non=g[~g.support_families.map(lambda x:_hasfam(x,family))]
            fs=_summarize(f);bs=_summarize(non)
            pv=_fisher_pvalue(fs['wins'],fs['losses'],bs['wins'],bs['losses'])
            combos.append({'study_period':period,'edge_bucket':bucket,'family':family,
                           'confirmed_games':fs['games'],'confirmed_wins':fs['wins'],
                           'confirmed_losses':fs['losses'],'confirmed_pushes':fs['pushes'],
                           'confirmed_ats_pct':fs['ats_win_pct'],'confirmed_roi_pct':fs['roi_pct'],
                           'not_confirmed_games':bs['games'],'not_confirmed_ats_pct':bs['ats_win_pct'],
                           'not_confirmed_roi_pct':bs['roi_pct'],
                           'raw_ats_lift_pp':round(fs['ats_win_pct']-bs['ats_win_pct'],2)
                                   if fs['ats_win_pct'] is not None and bs['ats_win_pct'] is not None else None,
                           'fisher_one_sided_p_exploratory':round(pv,6) if pv is not None else None,
                           'interpretation':'DESCRIPTIVE_ONLY_NOT_CAUSAL'})
    combos_frame=pd.DataFrame(combos)
    if not combos_frame.empty:
        combos_frame['bh_q_exploratory']=None
        for period,sub in combos_frame.groupby('study_period'):
            combos_frame.loc[sub.index,'bh_q_exploratory']=_bh_qvalues(sub.fisher_one_sided_p_exploratory.tolist())
        combos_frame['interpretation']=np.where(combos_frame.confirmed_games.lt(30)|combos_frame.not_confirmed_games.lt(30),
            'UNDERPOWERED_DESCRIPTIVE_ONLY','DESCRIPTIVE_ASSOCIATION_NOT_CAUSAL')
    ladder=summarize_grouped(df.assign(brain_ladder=np.where(df.oppose_count>0,'OPPOSED',
                                 np.where(df.support_count>=2,'2+ SUPPORT',
                                 np.where(df.support_count==1,'1 SUPPORT','0 SUPPORT')))),
                                 ['study_period','edge_bucket','brain_ladder'])
    return {'core_edge_buckets.csv':edged,'core_by_season.csv':eyr,
            'edge_threshold_grid.csv':pd.DataFrame(threshold),
            'incremental_brain_comparisons.csv':combos_frame,
            'brain_ladder.csv':ladder}

def execute(predictions,markets,brains=None,*,book=None,allow_retrospective_recipe=False):
    scored=merge_scored(predictions,markets,preferred_book=book,
                         allow_retrospective_recipe=allow_retrospective_recipe)
    scored=attach_brains(scored,brains)
    tables=research_tables(scored)
    audit={'model':'V5.5 NONLINEAR_HGB ONLY',
           'state':('RETROSPECTIVE_RECIPE_TRAIN_ONLY_OOS_NOT_INDEPENDENT_HOLDOUT'
                    if allow_retrospective_recipe else 'OOS_INPUTS_VALIDATED_NOT_MODEL_RETRAINED'),
           'source_note':'Flags attest fold/feature correctness; verify signed training manifests separately.',
           'predictions':int(len(predictions)),'historical_market_rows':int(len(markets)),
           'matched_scored_games':int(len(scored)),
           'discovery_matched':int((scored.study_period=='DISCOVERY_OOS').sum()),
           'holdout_2024':int((scored.study_period=='HOLDOUT_2024').sum()),
           'holdout_2025':int((scored.study_period=='HOLDOUT_2025').sum()),
           'brain_event_rows':int(len(brains)) if brains is not None else 0,
           'oos_folds':sorted(set(scored.fold_id.astype(str))),
           'recipe_sha256s':sorted(set(scored.model_recipe_sha256.astype(str))),
           'trading_conclusion':('NO AUTHORITY: model-recipe selection hindsight; 2024/25 NOT holdouts'
                                 if allow_retrospective_recipe else
                                 'NOT CERTIFIED; review independent provenance, sample sizes and frozen holdout per-year before authorization')}
    return scored,tables,audit

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',required=True,help='Real NONLINEAR_HGB chronological OOS, not legacy CORE')
    p.add_argument('--markets',required=True,help='Pregame bookmaker/score fact set; one precommitted market per game')
    p.add_argument('--brains',help='Optional signed as-of fired brains; omit for CORE-only study')
    p.add_argument('--book',help='Limit analysis to one precommitted sportsbook')
    p.add_argument('--output',required=True)
    args=p.parse_args(argv)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    try:
        pred=pd.read_csv(args.predictions);market=pd.read_csv(args.markets)
        b=pd.read_csv(args.brains) if args.brains else None
        scored,tables,audit=execute(pred,market,b,book=args.book)
        scored.to_csv(out/'v55_oos_game_edges.csv',index=False)
        for name,df in tables.items():df.to_csv(out/name,index=False)
        (out/'audit.json').write_text(json.dumps(audit,indent=2))
        print(json.dumps(audit,indent=2))
    except (ValueError,KeyError) as e:
        (out/'audit.json').write_text(json.dumps({'status':'HOLD','reason':str(e),'model':MODEL},indent=2))
        raise SystemExit(f'V5.5 ATS study rejected inputs: {e}')


# End consolidated ATS evaluator.

INPUT_PREFIX = 'research/ncaaf/v55/ats_primary/inputs'
REPORT_PREFIX = 'research/ncaaf/v55/ats_primary'
SUCCESS = 'COMPLETE_V55_ATS_PRIMARY_RESEARCH'
HOLD = 'HOLD_V55_HISTORICAL_REPLAY_OR_PROVENANCE'
REQUIRED_MANIFEST = (
    'schema_version', 'model_id', 'model_implementation', 'recipe_sha256',
    'training_code_sha256', 'historical_oos_verified',
    'pregame_feature_audit_verified', 'market_vintage_audit_verified',
    'chronological_model_selection_audited', 'predictions_sha256',
    'markets_sha256', 'source_provenance',
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _as_true(value: Any) -> bool:
    return value is True  # Never accept arbitrary truthy strings in audit manifest.


def validate_manifest(manifest: dict, inputs: dict[str, bytes]) -> None:
    if not isinstance(manifest, dict):
        raise ProvenanceError('V5.5: historical input manifest must be an object')
    missing = sorted(set(REQUIRED_MANIFEST) - set(manifest))
    if missing:
        raise ProvenanceError(f'V5.5: missing historical provenance fields: {missing}')
    if manifest['schema_version'] != 1 or manifest['model_id'] != MODEL:
        raise ProvenanceError('V5.5: incorrect historical replay schema/model')
    if manifest['model_implementation'] != 'ncaaf_v552_live':
        raise ProvenanceError('V5.5: historical forecast provider is not the deployed V5.5 trainer')
    for x in ('historical_oos_verified', 'pregame_feature_audit_verified',
              'market_vintage_audit_verified', 'chronological_model_selection_audited'):
        if not _as_true(manifest[x]):
            raise ProvenanceError(f'V5.5: {x} must be verified before ATS inference')
    for x in ('recipe_sha256', 'training_code_sha256', 'source_provenance'):
        if not isinstance(manifest[x], str) or len(manifest[x]) < 12:
            raise ProvenanceError(f'V5.5: missing meaningful {x} in historical replay manifest')
    if len(manifest['recipe_sha256']) != 64 or len(manifest['training_code_sha256']) != 64:
        raise ProvenanceError('V5.5: recipe and training code hashes must be SHA256 digests')
    for name in ('predictions', 'markets'):
        if _sha(inputs[name]) != manifest[f'{name}_sha256']:
            raise ProvenanceError(f'V5.5: {name} content differs from immutable replay manifest')
    if 'brains' in inputs:
        if not manifest.get('brains_sha256') or _sha(inputs['brains']) != manifest['brains_sha256']:
            raise ProvenanceError('V5.5: historical brain events missing or different from manifest SHA256')
    if manifest.get('recipe_selected_using_2024_2025') is True:
        raise ProvenanceError('V5.5: selected recipe used confirmation-period outcomes')


def verify_deployed_training_code(manifest: dict) -> None:
    """Compare the replay attestation with the EXACT locally deployed V5.5 source."""
    try:
        model = importlib.import_module('ncaaf_v552_live')
        path = Path(model.__file__).resolve()
        deployed_digest = _sha(path.read_bytes())
    except (ImportError, AttributeError, OSError, TypeError) as exc:
        raise ProvenanceError(
            f'V5.5: deployed training implementation unavailable for independent digest check: {type(exc).__name__}:{exc}'
        ) from exc
    if manifest['training_code_sha256'] != deployed_digest:
        raise ProvenanceError('V5.5: replay produced by a different model implementation than the deployed trainer')


def _load_inputs(bucket) -> tuple[dict, dict[str, bytes]]:
    def get(path, optional=False):
        blob = bucket.blob(f'{INPUT_PREFIX}/{path}')
        if not blob.exists():
            if optional:
                return None
            raise ProvenanceError(f'V5.5: missing gs://<bucket>/{INPUT_PREFIX}/{path}')
        return blob.download_as_bytes()
    manifest_bytes = get('manifest.json')
    manifest = json.loads(manifest_bytes.decode('utf-8'))
    inputs = {'predictions': get('predictions.csv'), 'markets': get('markets.csv')}
    brain_bytes = get('brains.csv', optional=True)
    if brain_bytes is not None:
        inputs['brains'] = brain_bytes
    validate_manifest(manifest, inputs)
    verify_deployed_training_code(manifest)
    return manifest, inputs


def _attempt_replay(*, storage_client, bucket_name, log_func) -> str:
    """Invoke the actual deployed model's explicit chronological OOS replay API.

    No reinterpretation of weekly 2026 forecasts or a legacy CORE replay is allowed.
    An implementation without the hook returns an honest hold, not fabricated data.
    """
    try:
        model = importlib.import_module('ncaaf_v552_live')
    except (ImportError, ModuleNotFoundError) as exc:
        return f'MODEL_IMPLEMENTATION_NOT_DEPLOYED:{type(exc).__name__}:{exc}'
    producer = getattr(model, 'run_historical_oos_replay', None)
    if not callable(producer):
        return 'V55_HISTORICAL_OOS_REPLAY_HOLD:DEPLOYED_MODEL_REPLAY_HOOK_MISSING' 
    try:
        result = producer(storage_client=storage_client, bucket_name=bucket_name,
                          output_prefix=INPUT_PREFIX, end_season=2025,
                          prospective_season=2026, log_func=log_func)
    except Exception as exc:
        return f'V55_HISTORICAL_OOS_REPLAY_FAILED:{type(exc).__name__}:{exc}'
    if isinstance(result, dict) and result.get('status') == 'COMPLETE_V55_EXACT_RECIPE_RETROSPECTIVE_MARGINS':
        return 'REPLAY_RETROSPECTIVE_MARGINS_PRODUCED'
    if isinstance(result, dict) and result.get('status') == 'EXPLORATORY_V55_RETROSPECTIVE_RECIPE_REPLAY':
        return 'REPLAY_RETROSPECTIVE_RECIPE_PRODUCED'
    if not isinstance(result, dict) or result.get('status') != 'VERIFIED_V55_CHRONOLOGICAL_OOS_REPLAY':
        return f'V55_HISTORICAL_OOS_REPLAY_HOLD:{(result or {}).get("status") if isinstance(result, dict) else "INVALID_RESULT"}'
    return 'REPLAY_PRODUCED'


def discovery_threshold_screen(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Discovery-year diagnostics ONLY; no threshold authorization/promotion.

    Holdout seasons are deliberately never used to select the candidate threshold.
    This is a screen, not a publication of an approved betting policy.
    """
    t = tables['edge_threshold_grid.csv']
    if t.empty:
        return pd.DataFrame(columns=['min_edge_points','discovery_games','discovery_wins',
                                     'discovery_losses','discovery_roi_pct','screen_status'])
    t = t.loc[t.study_period.eq('DISCOVERY_OOS')].copy()
    rows = []
    for edge, g in t.groupby('min_edge_points', sort=True):
        # The evaluator has one row per year and threshold. Aggregate by
        # actual game counts/units rather than averaging win percentages.
        n = int(g.games.sum());w=int(g.wins.sum());l=int(g.losses.sum());push=int(g.pushes.sum())
        units=float(g.units_net.sum());roi=100*units/n if n else None
        # Research screen only; minimum counts are transparent diagnostics.
        enough = n >= 100 and w + l >= 100 and g.loc[g.games.gt(0),'season'].nunique() >= 3
        rows.append(dict(min_edge_points=float(edge),discovery_games=n,
                         discovery_wins=w,discovery_losses=l,discovery_pushes=push,
                         discovery_roi_pct=round(roi,3) if roi is not None else None,
                         screen_status='DISCOVERY_SAMPLE_FOR_REVIEW' if enough else 'INSUFFICIENT_DISCOVERY_SAMPLE'))
    return pd.DataFrame(rows)


def brain_pair_and_veto_tables(scored: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Explicit directional pair/veto comparisons within identical CORE edge bands."""
    pairs, vetos = [], []
    for (period, band), group in scored.groupby(['study_period', 'edge_bucket'], sort=True):
        for first, second in combinations(FAMILIES, 2):
            first_hit = group.support_families.fillna('').map(lambda x: first in str(x).split(';'))
            second_hit = group.support_families.fillna('').map(lambda x: second in str(x).split(';'))
            both = group.loc[first_hit & second_hit]
            one = group.loc[first_hit ^ second_hit]
            neither = group.loc[~first_hit & ~second_hit]
            a, b, c = [_summarize(x) for x in (both, one, neither)]
            pairs.append({
                'study_period':period, 'edge_bucket':band,
                'first_family':first, 'second_family':second,
                'both_games':a['games'], 'both_ats_win_pct':a['ats_win_pct'], 'both_roi_pct':a['roi_pct'],
                'one_games':b['games'], 'one_ats_win_pct':b['ats_win_pct'], 'one_roi_pct':b['roi_pct'],
                'neither_games':c['games'], 'neither_ats_win_pct':c['ats_win_pct'], 'neither_roi_pct':c['roi_pct'],
                'both_vs_one_ats_lift_pp':round(a['ats_win_pct'] - b['ats_win_pct'],2)
                    if a['ats_win_pct'] is not None and b['ats_win_pct'] is not None else None,
                'claim':'DESCRIPTIVE_HYPOTHESIS_NO_AUTOMATIC_CONFIRMATION',
            })
        opposed = group.loc[group.oppose_count.gt(0)]
        no_opposition = group.loc[group.oppose_count.eq(0)]
        a, b = _summarize(opposed), _summarize(no_opposition)
        vetos.append({
            'study_period':period, 'edge_bucket':band,
            'opposed_games':a['games'], 'opposed_ats_win_pct':a['ats_win_pct'],
            'opposed_roi_pct':a['roi_pct'],
            'unopposed_games':b['games'], 'unopposed_ats_win_pct':b['ats_win_pct'],
            'unopposed_roi_pct':b['roi_pct'],
            'oppose_vs_unopposed_ats_pp':round(a['ats_win_pct']-b['ats_win_pct'],2)
                if a['ats_win_pct'] is not None and b['ats_win_pct'] is not None else None,
            'claim':'CONFLICT_ASSOCIATION_NOT_PROVEN_VETO',
        })
    return {'brain_pair_interactions.csv':pd.DataFrame(pairs),
            'opposing_brain_veto_tests.csv':pd.DataFrame(vetos)}


def discovery_to_holdout_threshold_study(scored: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fix a point threshold using only <=2023 discovery OOS, test later untouched.

    Threshold selection is a predeclared illustrative screen, not a trading
    recommendation. The two holdouts are NEVER used to choose the threshold.
    """
    discover = scored.loc[scored.study_period.eq('DISCOVERY_OOS')]
    results=[];candidate=[]
    for t in THRESHOLDS:
        train = discover.loc[discover.edge_points.ge(t)]
        x = _summarize(train)
        eligible = x['games'] >= 100 and (x['wins']+x['losses']) >= 100 and train.season.nunique() >= 3
        if eligible and x['roi_pct'] is not None:
            candidate.append((x['roi_pct'], x['games'], -t, t))
        row={'min_edge_points':t,'discovery_sample_eligible':bool(eligible),
             'discovery_games':x['games'],'discovery_ats_win_pct':x['ats_win_pct'],
             'discovery_roi_pct':x['roi_pct']}
        for y in (2024,2025):
            test=scored.loc[(scored.season.eq(y)) & (scored.edge_points.ge(t))]
            stats=_summarize(test)
            row[f'holdout_{y}_games']=stats['games']
            row[f'holdout_{y}_wins']=stats['wins']
            row[f'holdout_{y}_losses']=stats['losses']
            row[f'holdout_{y}_ats_pct']=stats['ats_win_pct']
            row[f'holdout_{y}_roi_pct']=stats['roi_pct']
            row[f'holdout_{y}_wilson_low_pct']=stats['wilson95_low_pct']
        results.append(row)
    chosen = max(candidate)[3] if candidate else None
    for r in results:
        r['discovery_selected'] = (r['min_edge_points']==chosen)
        r['status'] = ('REQUIRES_INDEPENDENT_REVIEW_NOT_AN_AUTHORIZED_THRESHOLD'
                       if r['discovery_selected'] else 'DIAGNOSTIC_ONLY')
    metadata={'discovery_selected_min_edge_points':chosen,
              'selection_rule':'maximum discovery ROI, min 100 ATS decisions, at least 3 discovery seasons; compare 2024 and 2025 independently',
              'holdout_used_for_selection':False,'automatic_threshold_promotion':False,
              'holdout_caveat':'Holdouts only independent if recipe and hypotheses were frozen before 2024'}
    return pd.DataFrame(results),metadata


def _publish_json(bucket, key: str, obj: dict) -> None:
    bucket.blob(key).upload_from_string(json.dumps(obj, sort_keys=True, indent=2, default=str),
                                        content_type='application/json')


def _publish_retrospective_replay(*, bucket, bucket_name: str, log_func) -> dict:
    """Publish a research preview WITHOUT independent-holdout/selection claims."""
    EXPLORATORY_PREFIX = f'{REPORT_PREFIX}/retrospective_inputs'

    def get(name):
        b=bucket.blob(f'{EXPLORATORY_PREFIX}/{name}')
        if not b.exists():
            raise ProvenanceError('missing retrospective research source '+name)
        return b.download_as_bytes()

    manifest=json.loads(get('manifest.json').decode('utf8'))
    if (manifest.get('schema_version')!=1 or manifest.get('model_id')!=MODEL
            or manifest.get('model_implementation')!='ncaaf_v552_live'
            or manifest.get('recipe_selected_using_2024_2025') is not True
            or manifest.get('chronological_model_selection_audited') is not False):
        raise ProvenanceError('retrospective recipe must be explicitly labeled, without historical holdout authority')
    if (manifest.get('historical_oos_verified') is not True
            or manifest.get('pregame_feature_audit_verified') is not True
            or manifest.get('market_vintage_audit_verified') is not True):
        raise ProvenanceError('missing original as-of training / feature / sportsbook audit')
    verify_deployed_training_code(manifest)
    pred=get('predictions.csv');markets=get('markets.csv')
    if (_sha(pred)!=manifest.get('predictions_sha256') or
            _sha(markets)!=manifest.get('markets_sha256')):
        raise ProvenanceError('retrospective replay file digest mismatch')
    brains=None
    brainblob=bucket.blob(f'{EXPLORATORY_PREFIX}/brains.csv')
    if brainblob.exists():
        raw=brainblob.download_as_bytes()
        if _sha(raw)!=manifest.get('brains_sha256'):
            raise ProvenanceError('retrospective brain event digest mismatch')
        brains=pd.read_csv(io.BytesIO(raw))
    scored,tables,audit=execute(pd.read_csv(io.BytesIO(pred)),
                               pd.read_csv(io.BytesIO(markets)),brains,
                               allow_retrospective_recipe=True)
    if not scored.model_recipe_sha256.eq(manifest['recipe_sha256']).all():
        raise ProvenanceError('retrospective model recipe hash mismatch')
    tables.update(brain_pair_and_veto_tables(scored))
    output_id=_sha(pred+markets+json.dumps(manifest,sort_keys=True).encode())[:20]
    prefix=f'{REPORT_PREFIX}/exploratory_runs/{output_id}'
    for name,frame in {'v55_oos_game_edges.csv':scored,**tables}.items():
        bucket.blob(f'{prefix}/{name}').upload_from_string(
            frame.to_csv(index=False),content_type='text/csv')
    audit.update({'status':'EXPLORATORY_RETROSPECTIVE_RECIPE_NO_HOLDOUT_CLAIM',
                 '2024_2025_holdout_authority':False,
                 'threshold_selection_authority':False,
                 'production_modified':False,'bet_authority':0,
                 'model_source_code_sha256':manifest['training_code_sha256'],
                 'manifest':manifest})
    _publish_json(bucket,f'{prefix}/audit.json',audit)
    log_func(f'[NCAAF-V55-ATS-EXPLORATORY] status=COMPLETE rows={len(scored)} '
             f'output=gs://{bucket_name}/{prefix} holdout=FALSE promotion=FALSE')
    return {'status':'EXPLORATORY_RETROSPECTIVE_RECIPE_NO_HOLDOUT_CLAIM',
            'games':int(len(scored)),
            'output_prefix':f'gs://{bucket_name}/{prefix}'}


def run_cloud(*, storage_client, bucket_name='sharp-models', log_func=print,
              generate_replay=True) -> dict:
    """Permanent primary stage of NCAAF Heavy Research.

    Archive full game-level evidence and all tables if valid. A HOLD report is
    published if the genuine V5.5 replay producer or source data is missing.
    """
    started = datetime.now(timezone.utc).isoformat()
    bucket = storage_client.bucket(bucket_name)
    report = dict(status=HOLD, started_at_utc=started, model=MODEL,
                  research_job='NCAAF_RESEARCH_HEAVY_V55_ATS_PRIMARY',
                  production_modified=False, bet_authority=0,
                  policy_promotion=False, season_2026_selection=False)
    log_func('[NCAAF-V55-ATS-PRIMARY] start=TRUE source=NONLINEAR_HGB ' 
             'legacy_core_oof=PROHIBITED 2026_outcomes=SEALED production_mutation=FALSE')
    try:
        if generate_replay:
            replay_status = _attempt_replay(storage_client=storage_client,
                                             bucket_name=bucket_name, log_func=log_func)
        else:
            replay_status = 'EXISTING_VERIFIED_INPUTS_ONLY'
        report['replay_producer'] = replay_status
        if replay_status=='REPLAY_RETROSPECTIVE_MARGINS_PRODUCED':
            statusblob=bucket.blob(f'{REPORT_PREFIX}/latest_replay_status.json')
            pointer=json.loads(statusblob.download_as_text())
            if pointer.get('status')!='COMPLETE_V55_EXACT_RECIPE_RETROSPECTIVE_MARGINS':
                raise ProvenanceError('V5.5 rolling margins not published')
            report.update(status='COMPLETE_V55_RETROSPECTIVE_MARGINS__ATS_SOURCE_VINTAGE_HOLD',
                          model_evaluation_completed=True,
                          matched_games=0,
                          margin_prediction_games=pointer.get('total_games'),
                          model_replay_artifacts=f'gs://{bucket_name}/'+pointer['run_prefix'],
                          reason='V5.5 exact NONLINEAR_HGB expanding-history margin replay complete. Verified pregame bookmaker and brain historical snapshots are not yet available; ATS and bet gates withheld.',
                          independent_holdout_claim=False,
                          historical_feature_vintage_certified=False,
                          historical_market_vintage_certified=False)
            log_func('[NCAAF-V55-ATS-PRIMARY] status=RETROSPECTIVE_MARGINS_READY ATS=HOLD ' +
                     'predicted_games='+str(report['margin_prediction_games'])+
                     ' output='+report['model_replay_artifacts'])
            _publish_json(bucket, f'{REPORT_PREFIX}/latest_status.json', report)
            return report
        if replay_status=='REPLAY_RETROSPECTIVE_RECIPE_PRODUCED':
            report['exploratory_research']=_publish_retrospective_replay(
                bucket=bucket,bucket_name=bucket_name,log_func=log_func)
            raise ProvenanceError('RETROSPECTIVE_RECIPE_ONLY: actual V5.5 fold fits are available, but 2024/25 not untouched model-selection holdouts')
        manifest, inputs = _load_inputs(bucket)
        pred = pd.read_csv(io.BytesIO(inputs['predictions']))
        markets = pd.read_csv(io.BytesIO(inputs['markets']))
        brains = pd.read_csv(io.BytesIO(inputs['brains'])) if 'brains' in inputs else None
        scores, tables, audit = execute(pred, markets, brains)
        # Validate additional binding: model recipe identity in the manifest must
        # equal the recipe of every chronological fold in the actual predictions.
        if not scores.model_recipe_sha256.eq(manifest['recipe_sha256']).all():
            raise ProvenanceError('V5.5: model recipe does not match replay producer manifest')
        if not (scores.study_period.eq('HOLDOUT_2024').any() and
                scores.study_period.eq('HOLDOUT_2025').any() and
                scores.study_period.eq('DISCOVERY_OOS').any()):
            raise ProvenanceError('V5.5: discovery and 2024/2025 independent periods required')
        coverage = {
            'discovery_games': int((scores.study_period=='DISCOVERY_OOS').sum()),
            'holdout_2024_games': int((scores.season==2024).sum()),
            'holdout_2025_games': int((scores.season==2025).sum()),
            'discovery_years': sorted(set(scores.loc[scores.study_period=='DISCOVERY_OOS','season'].astype(int).tolist())),
            'brain_families_observed': sorted(set(brains.family.astype(str).str.upper())) if brains is not None else [],
        }
        coverage['evidence_depth'] = (
            'SUBSTANTIAL_SAMPLE_NEEDS_INDEPENDENT_REVIEW'
            if (coverage['discovery_games'] >= 300 and
                coverage['holdout_2024_games'] >= 100 and
                coverage['holdout_2025_games'] >= 100 and
                len(coverage['discovery_years']) >= 3)
            else 'SPARSE_SAMPLE_EXPLORATORY_ONLY'
        )
        screen = discovery_threshold_screen(tables)
        tables['discovery_only_threshold_screen.csv'] = screen
        tables.update(brain_pair_and_veto_tables(scores))
        threshold_review, threshold_meta = discovery_to_holdout_threshold_study(scores)
        tables['discovery_selected_holdout_thresholds.csv'] = threshold_review
        # Store a distinct immutable run directory. No production policy path.
        id_key = _sha(inputs['predictions']+inputs['markets']+json.dumps(manifest, sort_keys=True).encode())[:20]
        prefix = f'{REPORT_PREFIX}/runs/{id_key}'
        for name, frame in {'v55_oos_game_edges.csv': scores, **tables}.items():
            bucket.blob(f'{prefix}/{name}').upload_from_string(
                frame.to_csv(index=False), content_type='text/csv')
        audit.update({'status':'INPUT_CONTRACT_VALIDATED_REVIEW_ATS_EVIDENCE',
                      'producer_status':replay_status,
                      'manifest_sha256':_sha(json.dumps(manifest,sort_keys=True).encode()),
                      'source_provenance':manifest['source_provenance'],
                      'discovery_only_policy_screen':True,
                      'threshold_research':threshold_meta,
                      'coverage_and_power':coverage,
                      'independent_attestation_notice':'Producer-provided provenance must be audited against training and CFBD/market source logs; checksum alone does not prove OOS.',
                      'bet_authority':0,'policy_promotion':False,
                      'V55_input_manifest':manifest})
        _publish_json(bucket, f'{prefix}/audit.json', audit)
        report.update(status=SUCCESS, reason='PRIMARY_RESEARCH_TABLES_PUBLISHED_NO_POLICY_PROMOTION',
                      matched_games=int(len(scores)),
                      discovery_games=int((scores.study_period=='DISCOVERY_OOS').sum()),
                      holdout_2024_games=int((scores.study_period=='HOLDOUT_2024').sum()),
                      holdout_2025_games=int((scores.study_period=='HOLDOUT_2025').sum()),
                      system_event_rows=int(len(brains)) if brains is not None else 0,
                      output_prefix=f'gs://{bucket_name}/{prefix}',
                      research_policy='POINT_EDGE_ATS_INCREMENTAL_BRAINS',
                      evidence_depth=coverage['evidence_depth'],
                      observed_brain_families=coverage['brain_families_observed'])
        log_func('[NCAAF-V55-ATS-PRIMARY] status=COMPLETE '
                 f'priced={len(scores)} discovery={report["discovery_games"]} '
                 f'holdout_2024={report["holdout_2024_games"]} holdout_2025={report["holdout_2025_games"]} '
                 f'brain_events={report["system_event_rows"]} threshold_promotion=FALSE '
                 f'output={report["output_prefix"]}')
    except (Exception,) as exc:
        report.update(status=HOLD, reason=f'{type(exc).__name__}:{exc}')
        log_func(f'[NCAAF-V55-ATS-PRIMARY-HOLD] status=HOLD reason={report["reason"]} '
                 f'producer={report.get("replay_producer")} legacy_substitution=FALSE')
    # Always publish status. Only successful evidence is written under immutable runs.
    try:
        _publish_json(bucket, f'{REPORT_PREFIX}/latest_status.json', report)
    except Exception as exc:
        log_func(f'[NCAAF-V55-ATS-PRIMARY-PUBLISH-HOLD] {type(exc).__name__}:{exc}')
        report['status']=HOLD
        report['reason']='STATUS_PUBLICATION_FAILED'
    return report
