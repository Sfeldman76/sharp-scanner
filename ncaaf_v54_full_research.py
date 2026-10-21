"""NCAAF V5.4 full CFBD-authoritative research orchestration and audit journal.

One full run, no stage skipping, no automatic promotion. V4 CORE remains football-only;
older NCAAF system engines supply STAT hypotheses. Historic replay != archived bet.
"""
from __future__ import annotations
import hashlib
import importlib
import json
import re
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from io import StringIO

import pandas as pd
import numpy as np

VERSION = 'NCAAF_V54_CFBD_AUTHORITATIVE_FULL_RESEARCH_20261020_V1'
ROOT = 'research/ncaaf/v54_full_research'
BUCKET = 'sharp-models'
REQUIRED_LEGACY = {
    'ncaaf_research_v2': ['run_ncaaf_research_v2','_extended_atoms','_attach_exact_expert_flags_to_miner','refresh_prediction_tracker_external','_v214_occurrence_token','load_current_report_diagnostic'],
    'utils': ['fetch_pathi_bigal_history_backend','build_pathi_bigal_team_game_state','add_pathi_bigal_rule_flags','read_recent_sharp_moves','score_and_record_ncaaf_production_v1'],
    'sharp_line_dashboard':['fit_historical_ncaaf_core_expert','fit_ncaaf_statistical_brain'],
    'ncaaf_core_challenger_v2':['run_ncaaf_core_challenger_v2'],
    'ncaaf_margin_tournament_v4':['run_cloud_research'],
    'ncaaf_v42_conditional_features':['run_cloud_research'],
    'ncaaf_v5_betting_research':['cloud_run'],
    'ncaaf_v51_edge_research':['cloud_run'],
    'ncaaf_v52_brain_research':['cloud_run'],
}
SOURCES = {
  'legacy_report':'research/ncaaf/v2/current_report.json',
  'legacy_bundle':'research/ncaaf/v2/current_bundle.pkl',
  'big_al_observations':'research/expert_observations/big_al/observations.jsonl',
  'pt_feeder_manifest':'research/ncaaf/external/prediction_tracker/feeder_manifest.json',
  'pt_header_manifest':'research/ncaaf/external/prediction_tracker/header_manifest.json',
  'pt_current':'research/ncaaf/external/prediction_tracker/current.csv',
  'pt_snapshots':'research/ncaaf/external/prediction_tracker/snapshots/',
  'v4_forecasts':'research/ncaaf/margin_tournament_v4/game_level_predictions.csv',
  'v42_forecasts':'research/ncaaf/conditional_features_v42/latest_report.json',
  'v5_signals':'research/ncaaf/v5_market_value/latest_report.json',
  'v51_edges':'research/ncaaf/v51_edge_attribution/latest_report.json',
}
BQ_SOURCES = [
 'sharplogger.sharp_data.scores_with_features',
 'sharplogger.sharp_data.sharp_scores_full',
 'sharplogger.sharp_data.ncaaf_historical_game_side_context',
 'sharplogger.sharp_data.ncaaf_cfbd_miner_features_v1',
 'sharplogger.sharp_data.ncaaf_source_team_aliases',
]


def _utc(): return datetime.now(timezone.utc).isoformat()
def _token(s): return re.sub('[^a-z0-9]','',str(s or '').lower())
def _canonical_token(s,aliases=None):
    t=_token(s)
    return (aliases or {}).get(t,t)
def _date(x): return pd.to_datetime(x,utc=True,errors='coerce').strftime('%Y-%m-%d') if pd.notna(pd.to_datetime(x,utc=True,errors='coerce')) else ''
def _finite_num(x):
    try:
        z=float(x);return z if np.isfinite(z) else None
    except (TypeError,ValueError):return None

class Audit:
    def __init__(self,log_func=print):
        self.log=log_func;self.events=[];self.run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    def emit(self,stage,status,**values):
        event={'at_utc':_utc(),'stage':stage,'status':status,**values}
        self.events.append(event)
        self.log('[NCAAF-V54-AUDIT] '+json.dumps(event,sort_keys=True,default=str,allow_nan=False))
        return event
    def step(self,name,func,*,fatal=True):
        started=time.perf_counter();self.emit(name,'START')
        try:
            x=func()
            self.emit(name,'PASS',seconds=round(time.perf_counter()-started,2))
            return x
        except Exception as ex:
            self.emit(name,'FAIL' if fatal else 'DEGRADED',seconds=round(time.perf_counter()-started,2),
                      error_type=type(ex).__name__,error=str(ex)[:1200],trace=traceback.format_exc(limit=8)[-5000:])
            if fatal:raise
            return None


def module_contract(modules):
    out={}
    for k,fn in REQUIRED_LEGACY.items():
        obj=modules.get(k)
        missing=[x for x in fn if not callable(getattr(obj,x,None))]
        f=getattr(obj,'__file__',None)
        sha=hashlib.sha256(open(f,'rb').read()).hexdigest() if f and __import__('os').path.isfile(f) else None
        out[k]={'path':f,'sha256':sha,'expected_functions':fn,'missing':missing,'status':'PASS' if obj and not missing else 'FAIL'}
    return out


def runtime_alignment(modules):
    dash=modules['sharp_line_dashboard'];utils=modules['utils']
    a=getattr(dash,'V133_DEPLOY_BUILD_ID',None);b=getattr(utils,'V133_DEPLOY_BUILD_ID',None)
    return {'status':'PASS' if a and b and a==b else 'FAIL',
            'dashboard_build':a,'utils_build':b,
            'dashboard_tag':getattr(dash,'V1337_SOURCE_TAG',None),
            'utils_tag':getattr(utils,'V1337_SOURCE_TAG',None),
            'rv2_tag':getattr(modules['ncaaf_research_v2'],'NCAAF_RESEARCH_V2_SOURCE_TAG',None),
            'same_dashboard_utils_build':bool(a and b and a==b)}


def legacy_source_constants(rv2,utils):
    """Audit exact legacy-defined paths, so new code doesn't invent locations."""
    names={
      'REPORT_CURRENT_BLOB':SOURCES['legacy_report'],
      'BUNDLE_CURRENT_BLOB':SOURCES['legacy_bundle'],
      'BIG_AL_OBSERVATION_LEDGER_BLOB':SOURCES['big_al_observations'],
      'PT_FEEDER_MANIFEST_BLOB':SOURCES['pt_feeder_manifest'],
      'PT_HEADER_MANIFEST_BLOB':SOURCES['pt_header_manifest'],
      'PT_CURRENT_BLOB':SOURCES['pt_current'],
      'PT_FEEDER_SNAPSHOT_PREFIX':SOURCES['pt_snapshots'].rstrip('/'),
    }
    observed={k:getattr(rv2,k,None) for k in names}
    mismatched={k:{'expected':x,'actual':observed[k]} for k,x in names.items() if observed[k]!=x}
    # Validate that the fallback history loader actually retains the expected
    # functions; choose its own table order, do not reimplement the SQL here.
    import inspect
    try:loader_source=inspect.getsource(utils.fetch_pathi_bigal_history_backend)
    except (OSError,TypeError):loader_source=''
    fallback=['sharplogger.sharp_data.scores_with_features',
              'sharplogger.sharp_data.sharp_scores_full']
    missing_fallback=[x for x in fallback if x not in loader_source]
    return {'status':'PASS' if not mismatched and not missing_fallback else 'FAIL',
            'constants':observed,'mismatched':mismatched,'history_table_contract':fallback,
            'missing_history_fallbacks':missing_fallback,
            'legacy_source_tag':getattr(rv2,'NCAAF_RESEARCH_V2_SOURCE_TAG',None)}


def source_manifest(*,storage_client,bucket_name=BUCKET,bq_client=None):
    bucket=storage_client.bucket(bucket_name)
    objects={}
    for name,path in SOURCES.items():
        if path.endswith('/'):
            items=list(storage_client.list_blobs(bucket_name,prefix=path,max_results=8))
            objects[name]={'prefix':path,'sample_objects':len(items),'first_objects':[z.name for z in items]}
            continue
        blob=bucket.blob(path)
        try:
            blob.reload()
            objects[name]={'path':path,'exists':True,'generation':str(blob.generation),
                           'created_utc':str(blob.time_created),'bytes':blob.size}
        except Exception as ex:
            if type(ex).__name__=='NotFound':objects[name]={'path':path,'exists':False}
            else:objects[name]={'path':path,'status':'ERROR','error':str(ex)[:300]}
    tables={}
    if bq_client is not None:
        for name in BQ_SOURCES:
            try:
                tbl=bq_client.get_table(name); cols=[f.name for f in tbl.schema]
                flags=[x for x in cols if x.startswith(('BigAl_CF','Pathi_FB_')) and not x.endswith('_DataReady')]
                tables[name]={'status':'FOUND','columns':len(cols),'bigal_cf':sum(x.startswith('BigAl_CF') for x in flags),
                              'pathi_fb':sum(x.startswith('Pathi_FB_') for x in flags),'matched_flag_examples':flags[:15],
                              'candidate_time_cols':[x for x in ('Game_Start','Game_Date','Snapshot_Timestamp','feat_Game_Start') if x in cols],
                              'dataset_mutation_time':str(getattr(tbl,'modified',None))}
            except Exception as ex:tables[name]={'status':'ERROR','error':str(ex)[:300]}
    return {'objects':objects,'tables':tables,'source_contract':'LEGACY_SOURCE_FUNCTIONS_PLUS_BQ_GCS_INVENTORY__NO_PREGAME_PROOF'}


def export_legacy_occurrences(edges, history_cache, aliases=None):
    """Match directional graded occurrences to immutable V4 game IDs.

    Strict season/date/oriented teams matching. Source is retrospective, never
    automatically a contemporaneous bet and cannot contribute a verified vote.
    """
    base=edges.drop_duplicates(['Season','Game_ID']).copy()
    req={'Season','Game_ID','Game_Date','Team','Opponent'}
    if not req.issubset(base):raise ValueError('V53_EDGE_KEY_COLUMNS_MISSING')
    base['Season']=pd.to_numeric(base.Season,errors='raise').astype(int)
    base['__date']=pd.to_datetime(base.Game_Date,utc=True,errors='coerce').dt.strftime('%Y-%m-%d')
    base['__tm']=base.Team.map(lambda x:_canonical_token(x,aliases));base['__op']=base.Opponent.map(lambda x:_canonical_token(x,aliases))
    left=base[['Season','Game_ID','__date','__tm','__op']].rename(columns={'__tm':'team_token','__op':'opp_token'})
    right=base[['Season','Game_ID','__date','__tm','__op']].rename(columns={'__tm':'opp_token','__op':'team_token'})
    keys=pd.concat([left.assign(Side='TEAM'),right.assign(Side='OPPONENT')],ignore_index=True)
    # A source with the same season/date/teams resolving to multiple games is unsafe.
    amb=keys.duplicated(['Season','__date','team_token','opp_token'],keep=False)
    keys=keys[~amb].copy()
    source=[]
    stats={'cache_systems':len(history_cache),'directional_systems':0,'occurrences':0,'empty_or_invalid':0,'non_football_omitted':0}
    for system_id,desc in (history_cache or {}).items():
        if not isinstance(desc,dict) or desc.get('role')!='directional':continue
        family=str(desc.get('family',''))
        if family not in ('Pathi','BigAl'):continue
        fam='PATHI' if family=='Pathi' else 'BIG_AL'
        stats['directional_systems']+=1
        for entry in desc.get('occurrences') or []:
            stats['occurrences']+=1
            if not isinstance(entry,dict):stats['empty_or_invalid']+=1;continue
            try:season=int(float(entry.get('season')))
            except (ValueError,TypeError):stats['empty_or_invalid']+=1;continue
            d=str(entry.get('date',''))[:10];tm=_canonical_token(entry.get('team'),aliases);op=_canonical_token(entry.get('opponent'),aliases)
            if not (d and tm and op):stats['empty_or_invalid']+=1;continue
            source.append({'Season':season,'__date':d,'team_token':tm,'opp_token':op,'Family':fam,'System_ID':str(system_id)})
    columns=['Season','Game_ID','Family','System_ID','Side','Evidence_Class']
    if not source:return pd.DataFrame(columns=columns),{**stats,'joined':0,'unmatched':0,'ambiguous_keys':int(amb.sum())}
    records=pd.DataFrame(source)
    match=records.merge(keys,on=['Season','__date','team_token','opp_token'],how='left',indicator=True,validate='many_to_one')
    matched=match.loc[match._merge=='both',['Season','Game_ID','Family','System_ID','Side']].copy()
    conflicting=matched.groupby(['Season','Game_ID','Family','System_ID']).Side.nunique()
    conflict_keys=conflicting[conflicting>1].index
    if len(conflict_keys):
        idx=pd.MultiIndex.from_frame(matched[['Season','Game_ID','Family','System_ID']]).isin(conflict_keys)
        matched=matched.loc[~idx].copy()
    matched=matched.drop_duplicates(['Season','Game_ID','Family','System_ID'])
    matched['Evidence_Class']='RETROSPECTIVE_GRADED_OCCURRENCE__NOT_PREGAME_ARCHIVE'
    stats.update(joined=int(len(matched)),unmatched=int((match._merge!='both').sum()),
                 ambiguous_keys=int(amb.sum()),conflicting_system_game=int(len(conflict_keys)),
                 family_counts=matched.Family.value_counts().to_dict())
    return matched[columns],stats


def export_legacy_miner_mechanisms(edges, miner_games, miner_report, rv2, dashboard_module, aliases=None):
    """Evaluate legacy confirmed SPREAD mechanism conditions against the frozen V4 game keys.

    Pull real condition definitions from the existing registry, and real feature
    masks from the exact legacy _extended_atoms evaluator. This is retrospective
    exploratory research, NOT a claim that the rule was published in 2021.
    """
    cols=['Season','Game_ID','Family','System_ID','Side','Evidence_Class']
    diag={'source':'ncaaf_research_v2.run_system_miner_v3','mechanisms_considered':0,
          'condition_unavailable':0,'fired_rows':0,'matched_games':0,'unmatched_games':0,
          'confirmation_families':{},'status':'NO_MECHANISMS'}
    if miner_games is None or getattr(miner_games,'empty',True):
        return pd.DataFrame(columns=cols),{**diag,'status':'MISSING_MINER_GAME_FRAME'}
    report=miner_report.get('spreads') or {}
    mechanisms=[m for m in report.get('mechanism_families',[]) if m.get('confirmation_pass')]
    if not mechanisms:return pd.DataFrame(columns=cols),diag
    mg=miner_games.reset_index(drop=True).copy()
    req={'Season','Team_Norm','Opponent_Norm'}
    if not req.issubset(mg):raise RuntimeError('V53_MINER_EXPECTED_GAME_ORIENTATION_MISSING')
    mg['Season']=pd.to_numeric(mg.Season,errors='coerce')
    mg['__date']=pd.to_datetime(mg.get('Game_Date',mg.get('Game_Start')),
                                errors='coerce',utc=True).dt.strftime('%Y-%m-%d')
    mg['__home']=mg.Team_Norm.map(lambda x:_canonical_token(x,aliases));mg['__away']=mg.Opponent_Norm.map(lambda x:_canonical_token(x,aliases))
    mg['__pair']=mg.apply(lambda r: ':'.join(sorted((r['__home'],r['__away']))),axis=1)
    e=edges.drop_duplicates(['Season','Game_ID']).copy()
    e['Season']=pd.to_numeric(e.Season,errors='raise').astype(int)
    e['__date']=pd.to_datetime(e.Game_Date,utc=True,errors='coerce').dt.strftime('%Y-%m-%d')
    e['__home']=e.Team.map(lambda x:_canonical_token(x,aliases));e['__away']=e.Opponent.map(lambda x:_canonical_token(x,aliases))
    e['__pair']=e.apply(lambda r: ':'.join(sorted((r['__home'],r['__away']))),axis=1)
    e=e[['Season','Game_ID','__date','__home','__away','__pair']]
    collision=e.duplicated(['Season','__date','__pair'],keep=False)
    if collision.any():e=e[~collision].copy()
    j=mg[['Season','__date','__pair','__home','__away']].merge(e,
        on=['Season','__date','__pair'],how='left',suffixes=('_miner','_v4'),validate='many_to_one')
    # Strict pair equality + same date; orientation can be swapped.
    normal=(j.__home_miner==j.__home_v4)&(j.__away_miner==j.__away_v4)
    flipped=(j.__home_miner==j.__away_v4)&(j.__away_miner==j.__home_v4)
    valid=(normal|flipped)&j.Game_ID.notna()
    diag['matchable_miner_games']=int(valid.sum());diag['total_miner_games']=len(mg)
    diag['pair_collisions_v4']=int(collision.sum())
    atoms=rv2._extended_atoms(mg,dashboard_module,for_live=False,market='spreads')
    atom_map={str(a['name']):np.asarray(a['mask'],dtype=bool) for a in atoms}
    records=[]
    for mech in mechanisms:
        diag['mechanisms_considered']+=1
        conditions=list(mech.get('representative_conditions') or [])
        if not conditions or any(c not in atom_map for c in conditions):
            diag['condition_unavailable']+=1;continue
        firing=np.ones(len(mg),dtype=bool)
        for c in conditions:
            if len(atom_map[c])!=len(mg):raise RuntimeError('V53_MINER_ATOM_MASK_ALIGNMENT_BROKEN')
            firing &= atom_map[c]
        direction=str(mech.get('direction','PLAY_ON')).upper()
        if direction not in ('PLAY_ON','FADE'):raise RuntimeError('V53_UNEXPECTED_MINER_MECHANISM_DIRECTION')
        family=('STATE_BOUNCE_BACK' if (mech.get('uses_role_transition_bounceback_family') or mech.get('uses_season_record_state_family'))
                else 'PREDICTION_TRACKER' if mech.get('uses_external_ratings_family') else 'SYSTEM_MINER')
        sid=str(mech.get('mechanism_id') or '')
        if not sid:raise RuntimeError('V53_EMPTY_MECHANISM_ID')
        indices=np.flatnonzero(firing&valid.to_numpy(dtype=bool))
        diag['fired_rows']+=int(firing.sum())
        for i in indices:
            # PLAY_ON refers to the miner TEAM; FADE to its opponent. V4 may
            # have swapped orientation, so translate back into V4 canonical side.
            pick_miner_team=(direction=='PLAY_ON')
            v4_team= (pick_miner_team if bool(normal.iloc[i]) else not pick_miner_team)
            records.append({'Season':int(j.Season.iloc[i]),'Game_ID':str(j.Game_ID.iloc[i]),
                'Family':family,'System_ID':sid,'Side':'TEAM' if v4_team else 'OPPONENT',
                'Evidence_Class':'CHRONOLOGICAL_MINER_RETROSPECTIVE__INSPECTED_CONFIRMATION'})
    if not records:return pd.DataFrame(columns=cols),{**diag,'status':'ZERO_MATCHED_MECHANISM_FIRES'}
    result=pd.DataFrame(records).drop_duplicates(cols[:5])
    collisions=result.groupby(['Season','Game_ID','Family','System_ID']).Side.nunique()
    bad=collisions[collisions>1].index
    if len(bad):
        idx=pd.MultiIndex.from_frame(result[['Season','Game_ID','Family','System_ID']]).isin(bad)
        result=result.loc[~idx].copy()
    diag['matched_games']=int(result.Game_ID.nunique())
    diag['matched_system_game_rows']=len(result)
    diag['contradiction_quarantines']=len(bad)
    diag['confirmation_families']=result.Family.value_counts().to_dict()
    diag['status']='RETROSPECTIVE_MATCHED__NO_ARCHIVED_VOTE'
    return result[cols],diag


def _canonical_cfbd_research_join_keys(frame, *, source):
    """Copy and canonicalize CFBD's season/game keys for cross-pipeline joins.

    Cloud Run's real CFBD rule replay yields string game identifiers; the
    V5.1 CSV reader can infer int64 for the very same verified IDs. Never
    coerce IDs via float or alter identity to force a join. Any malformed
    key fails closed, leaving the immutable source and production untouched.
    """
    if not isinstance(frame, pd.DataFrame):
        raise RuntimeError('V548_EXPECTED_DATAFRAME source='+str(source))
    missing={'Season','Game_ID'}-set(frame.columns)
    if missing:
        raise RuntimeError('V548_REQUIRED_CFBD_JOIN_KEYS_MISSING source='+str(source)+' keys='+str(sorted(missing)))
    out=frame.copy()
    if out.empty:
        out['Season']=pd.Series(index=out.index,dtype='int64')
        out['Game_ID']=pd.Series(index=out.index,dtype='str')
        return out
    # Numeric season values must be an integer year, not an approximate
    # conversion. A nullable float is acceptable only when finite/integral.
    seasons=[]
    for v in out['Season'].tolist():
        if pd.isna(v) or isinstance(v,(bool,np.bool_)):
            raise RuntimeError('V548_INVALID_CFBD_SEASON source='+str(source))
        if isinstance(v,(int,np.integer)):
            yr=int(v)
        elif isinstance(v,(float,np.floating)):
            if not np.isfinite(v) or not float(v).is_integer():
                raise RuntimeError('V548_INVALID_CFBD_SEASON source='+str(source))
            yr=int(v)
        elif isinstance(v,str) and re.fullmatch(r'[0-9]{4}',v):
            yr=int(v)
        else:
            raise RuntimeError('V548_INVALID_CFBD_SEASON source='+str(source))
        if not (1900<=yr<=2100):
            raise RuntimeError('V548_INVALID_CFBD_SEASON source='+str(source))
        seasons.append(yr)
    ids=[]
    for v in out['Game_ID'].tolist():
        if pd.isna(v) or isinstance(v,(bool,np.bool_)):
            raise RuntimeError('V548_INVALID_CFBD_GAME_ID source='+str(source))
        if isinstance(v,(int,np.integer)):
            if int(v)<0:raise RuntimeError('V548_INVALID_CFBD_GAME_ID source='+str(source))
            game_id=str(int(v))
        elif isinstance(v,str) and re.fullmatch(r'[0-9]+',v):
            game_id=v
        else:
            # In particular, floats cannot certify identifier precision.
            raise RuntimeError('V548_INVALID_CFBD_GAME_ID source='+str(source))
        ids.append(game_id)
    out['Season']=pd.Series(seasons,index=out.index,dtype='int64')
    out['Game_ID']=pd.Series(ids,index=out.index,dtype='str')
    if out[['Season','Game_ID']].isna().any().any():
        raise RuntimeError('V548_CANONICAL_CFBD_JOIN_HAS_NULL source='+str(source))
    return out


def exploratory_attribution(edges,legacy):
    """Matched conditional comparisons, never historically confirmed bets."""
    base=_canonical_cfbd_research_join_keys(edges,source='V51_CORE_EDGE_LEDGER')
    assert not base.duplicated(['Season','Game_ID','Model']).any(), 'V53_DUPLICATE_MODEL_GAME'
    work=_canonical_cfbd_research_join_keys(legacy,source='CFBD_RECONSTRUCTED_SYSTEM_OCCURRENCES')
    if len(work):
        work=work.merge(base[['Season','Game_ID','Model','Side','Result','Abs_Edge','Selection_Type','Period']],
                        on=['Season','Game_ID'],validate='many_to_many',suffixes=('_System','_Core'))
        work['Relation']=np.where(work.Side_System==work.Side_Core,'CONFIRM','OPPOSE')
    else:
        work=pd.DataFrame(columns=['Season','Game_ID','Model','Family','System_ID','Relation','Result','Period'])
    system_rows=[]
    group_cols=['Model','Period','Family','System_ID','Relation']
    for key,g in work.groupby(group_cols,dropna=False):
        decis=g[g.Result.isin(['WIN','LOSS'])]; w=int((decis.Result=='WIN').sum()); n=len(decis)
        system_rows.append(dict(zip(group_cols,key),games=int(g.Game_ID.nunique()),decisions=n,wins=w,losses=n-w,
                                hit_rate=w/n if n else None,evidence='RETROSPECTIVE_CONTEXT_ONLY',authority=0))
    systems=pd.DataFrame(system_rows)
    # Count family at most once per game/model; no multiplication from correlated flags.
    if len(work):
        flags=work.groupby(['Season','Game_ID','Model','Family']).Relation.agg(lambda s:'CONFLICT' if s.nunique()>1 else s.iloc[0]).reset_index()
        joined=base.merge(flags,on=['Season','Game_ID','Model'],how='left',validate='one_to_many')
        rels=[]
        for (model,period,fam,relation),g in joined[joined.Relation.notna()].groupby(['Model','Period','Family','Relation']):
            q=g[g.Result.isin(['WIN','LOSS'])];w=int((q.Result=='WIN').sum());n=len(q)
            rels.append({'Model':model,'Period':period,'Family':fam,'Relation':relation,'observations':len(g),'decisions':n,
                         'wins':w,'losses':n-w,'hit_rate':w/n if n else None,'evidence':'RETROSPECTIVE_CONTEXT_ONLY'})
        families=pd.DataFrame(rels)
    else:families=pd.DataFrame(columns=['Model','Period','Family','Relation','observations','decisions','wins','losses','hit_rate','evidence'])
    return systems,families


def policy_shadow(edges,legacy,verified_count=0,source_vintage_certified=False):
    """V4 point-edge candidates plus descriptive confluence; never auto BET/STRONG."""
    base=_canonical_cfbd_research_join_keys(edges,source='V51_CORE_EDGE_LEDGER')
    expert=_canonical_cfbd_research_join_keys(legacy,source='CFBD_RECONSTRUCTED_SYSTEM_OCCURRENCES')
    out=base[['Season','Game_ID','Game_Date','Team','Opponent','Model','Side','Result','Abs_Edge','Edge_Bin','Selection_Type','Period']].copy()
    if not expert.empty:
        z=expert.groupby(['Season','Game_ID','Family']).Side.agg(
            lambda a: a.iloc[0] if a.nunique()==1 else 'CONFLICT').reset_index()
        o=out[['Season','Game_ID','Model','Side']].merge(z,on=['Season','Game_ID'],suffixes=('_Core','_Signal'),how='left')
        o['Relation']=np.where(o.Side_Signal.isna(),'MISSING',np.where(o.Side_Signal=='CONFLICT','CONFLICT',
                              np.where(o.Side_Signal==o.Side_Core,'CONFIRM','OPPOSE')))
        rel=o.groupby(['Season','Game_ID','Model']).agg(
           Legacy_Confirm_Families=('Relation',lambda x:int((x=='CONFIRM').sum())),
           Legacy_Oppose_Families=('Relation',lambda x:int((x=='OPPOSE').sum())),
           Legacy_Conflict_Families=('Relation',lambda x:int((x=='CONFLICT').sum())),
           Legacy_Families_Observed=('Relation',lambda x:int((x!='MISSING').sum()))).reset_index()
        out=out.merge(rel,on=['Season','Game_ID','Model'],validate='one_to_one')
    else:
        for k in ('Legacy_Confirm_Families','Legacy_Oppose_Families','Legacy_Conflict_Families','Legacy_Families_Observed'):
            out[k]=0
    # Count families once, not individual correlated triggers. These remain
    # retrospective descriptive indicators, NOT confirmed independent votes.
    out['Shadow_Evidence_Tier']=np.select([
       out.Legacy_Conflict_Families.gt(0)|out.Legacy_Oppose_Families.gt(0),
       out.Legacy_Confirm_Families.ge(2),out.Legacy_Confirm_Families.eq(1)],
       ['CONFLICT_REVIEW','MULTIFAMILY_RETROSPECTIVE','ONE_FAMILY_RETROSPECTIVE'],
       default='CORE_EDGE_ONLY')
    out['Candidate_Tier']='CANDIDATE_RESEARCH_ONLY'
    out['V5_Bet_Authority']=0
    out['V5_Bet_Decision']='NO_BET'
    out['Authority_Hold_Reason']='REQUIRES_VERIFIED_PROSPECTIVE_BRAINS_AND_LIVE_V4_PRICE_LOCK'
    return out,{'status':'SHADOW_ONLY__NO_UNPROVEN_AUTO_PROMOTION','bet_authority':0,
                'verified_archived_system_game_records':int(verified_count),
                'v4_source_vintage_certified':bool(source_vintage_certified),'research_candidate_rows':len(out),
                'descriptive_tier_rows':out.Shadow_Evidence_Tier.value_counts().to_dict(),
                'reasons':['V4 full feature publication-time proof pending','All historical subgroup results inspected',
                           'Genuine live V4/A inference and price lock not certified','Legacy historical graded occurrences are retrospective, not bet tickets'],
                'decision_contract':'MODEL_POINT_EDGE__STAT_ARCHIVE_VINTAGE__MARKET_QUOTE__EDGE_AUTHORITY'}


def save_run(bucket, audit,report,artifacts,*,latest=True):
    key=f'{ROOT}/runs/{audit.run_id}'
    for name,(payload,ctype) in artifacts.items():
        bucket.blob(f'{key}/{name}').upload_from_string(payload,content_type=ctype,if_generation_match=0)
    report['report_uri']=f'gs://{bucket.name}/{key}/report.json'
    report['audit_uri']=f'gs://{bucket.name}/{key}/audit_events.jsonl'
    report['source_manifest_uri']=f'gs://{bucket.name}/{key}/source_manifest.json'
    bucket.blob(f'{key}/report.json').upload_from_string(json.dumps(report,indent=2,default=str,allow_nan=False),content_type='application/json',if_generation_match=0)
    if latest:
        bucket.blob(f'{ROOT}/latest_report.json').upload_from_string(json.dumps(report,indent=2,default=str,allow_nan=False),content_type='application/json')
    return report


def run_legacy_with_promotion_held(rv2,*,audit,**kwargs):
    """Full legacy research with its optional automatic PT promotion disabled.

    The original engine still grades all PT systems and can publish research,
    but does not silently change a shared live-promotion controller during V5.
    """
    old=getattr(rv2,'pt_promotion_settle_and_review',None)
    if old is None:
        audit.emit('LEGACY_PT_AUTO_PROMOTION_HOLD','UNAVAILABLE_LEGACY_CONTROLLER')
        return rv2.run_ncaaf_research_v2(**kwargs)
    def shadow_only(*args,**kwargs):
        audit.emit('LEGACY_PT_AUTO_PROMOTION_HOLD','BLOCKED',
                   requested_allow_promotion=bool(kwargs.get('allow_promotion',False)))
        return {'status':'V53_RESEARCH_ONLY','review':{'status':'V53_RESEARCH_ONLY_NO_PROMOTION',
                'reason':'V5 full research cannot promote legacy PT automatically'}}
    setattr(rv2,'pt_promotion_settle_and_review',shadow_only)
    try:return rv2.run_ncaaf_research_v2(**kwargs)
    finally:setattr(rv2,'pt_promotion_settle_and_review',old)


def run_full(*,storage_client,bucket_name=BUCKET,log_func=print,modules=None,bq_client=None):
    """Single full run: historical football only from CFBD; other sources prices/rules only.

    No legacy game-cache fits and no mixed-source fallback. Failures produce a GCS
    stage-by-stage audit and then propagate as a nonzero Cloud Run task exit.
    """
    audit=Audit(log_func);bucket=storage_client.bucket(bucket_name)
    manifest={};results={};outputs={};cfbd_audit={}
    join_audit={'status':'NOT_REACHED'};miner_join={'status':'NOT_REACHED'}
    shadow_policy={'status':'NOT_REACHED','bet_authority':0}
    try:
        def _archive_opt_in():
            import os
            if str(os.getenv('NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED','0')).strip().lower() not in ('1','true','yes','on'):
                raise RuntimeError('V54_CFBD_ARCHIVE_OPT_IN_MISSING__SET_NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED=1')
            return {'CFBD_FULL_ARCHIVE_RESEARCH':True}
        audit.step('CFBD_ARCHIVE_RESEARCH_OPT_IN_PREFLIGHT',_archive_opt_in)
        if modules is None:modules={n:importlib.import_module(n) for n in REQUIRED_LEGACY}
        contract=audit.step('LEGACY_MODULE_CONTRACT',lambda:module_contract(modules))
        if any(v['status']!='PASS' for v in contract.values()):
            raise RuntimeError('V54_LEGACY_MODULE_FUNCTION_MISSING '+str({k:v['missing'] for k,v in contract.items() if v['missing']}))
        alignment=audit.step('MATCHED_DASHBOARD_UTILS_BUILD',lambda:runtime_alignment(modules))
        if alignment['status']!='PASS':raise RuntimeError('V54_LEGACY_RUNTIME_BUILD_MISMATCH')
        constants=audit.step('LEGACY_SOURCE_CONSTANTS_RULE_LOOKUP_ONLY',lambda:legacy_source_constants(modules['ncaaf_research_v2'],modules['utils']))
        if constants['status']!='PASS':raise RuntimeError('V54_LEGACY_SOURCE_CONSTANT_MISMATCH')
        if bq_client is None:
            from google.cloud import bigquery
            bq_client=bigquery.Client(project='sharplogger')
        manifest=audit.step('EXACT_LEGACY_SOURCE_MANIFEST_METADATA_ONLY',lambda:source_manifest(storage_client=storage_client,bucket_name=bucket_name,bq_client=bq_client))
        audit.emit('LEGACY_NON_CFBD_HISTORY_LOADERS','BYPASSED',
            loaders=['fit_historical_ncaaf_core_expert','fit_ncaaf_statistical_brain',
               'run_ncaaf_core_challenger_v2','run_ncaaf_research_v2'],
            purpose='no historical football facts or retrofitted scores from legacy market/dashboard tables')
        import ncaaf_v54_cfbd_truth as truth
        cfbd_home,cfbd_sides,cfbd_audit=audit.step('CFBD_CANONICAL_HISTORY_AND_SCORE_PROVENANCE',lambda:truth.load_from_bigquery(bq_client,log_func=log_func))
        audit.step('CFBD_SOURCE_FIREWALL_CORE',lambda:truth.verify_cfbd_frame(cfbd_home,require_unique_home=True))
        audit.emit('CFBD_HISTORICAL_COVERAGE','PASS',unique_home_games=len(cfbd_home),team_sides=len(cfbd_sides),
                   cfbd_seasons=cfbd_audit.get('source_match',{}).get('score_coverage_by_year'),
                   market_open_sides=cfbd_audit.get('market',{}).get('market_open_sides'),
                   football_source=truth.CFBD_GAME_ARCHIVE,
                   cfbd_feature_source=truth.CFBD_FEATURE_VIEW,
                   market_price_only=truth.MARKET_SIDE_TABLE)
        audit.emit('CFBD_MINER_PREGAME_FIELD_COVERAGE','PASS',
                   prior_cfbd_features=len(truth.safe_historical_cfbd_columns(cfbd_sides.columns)),
                   available_history_state_fields=[c for c in ('Prev_Points_For','Prev_Points_Against',
                     'ATS_WinPct_Prior','SU_Loss_Streak_Prior','ATS_Loss_Streak_Prior','Game_Number_Prior',
                     'Opp_ATS_WinPct_Prior','Opp_Prev_Points_For','Opp_Prev_Points_Against') if c in cfbd_sides],
                   unavailable_optional_context=[c for c in ('Rivalry_Flag','Revenge_Flag',
                     'Away_Travel_Miles','Away_Time_Zones_Crossed','PT_Meta_Home_Margin') if c not in cfbd_sides],
                   postgame_feature_columns_admitted=0,market_fields_are_price_only=True)
        outputs['cfbd_source_audit.json']=(json.dumps(cfbd_audit,indent=2,default=str,allow_nan=False),'application/json')
        dash=modules['sharp_line_dashboard']; core=modules['ncaaf_core_challenger_v2']
        rv2=modules['ncaaf_research_v2'];ut=modules['utils']
        old_cache=getattr(dash,'_V1357_SPREAD_RESEARCH_CACHE',None)
        old_cfbd_guard=getattr(dash,'_V54_CFBD_ONLY_ACTIVE',None)
        old_cfbd_source=getattr(core,'NCAAF_CFBD_MINER_FEATURE_TABLE',None)
        try:
            # Explicitly give V4/V4.2 the canonical CFBD-only dataframe.
            dash._V54_CFBD_ONLY_ACTIVE=True
            dash._V1357_SPREAD_RESEARCH_CACHE={
               'games':cfbd_home.copy(),'miner_games':cfbd_home.copy(),
               'historical_source':'CFBD_AUTHORITATIVE',
               'season_arr':pd.to_numeric(cfbd_home.Season,errors='coerce').to_numpy(float),
               'candidate_feature_cols':truth.safe_historical_cfbd_columns(cfbd_home.columns)}
            core.NCAAF_CFBD_MINER_FEATURE_TABLE=truth.CFBD_FEATURE_VIEW
            if isinstance(getattr(core,'_CFBD_CORE_CONTEXT_CACHE',None),dict):core._CFBD_CORE_CONTEXT_CACHE.clear()
            result_v4=audit.step('FULL_V4_FOOTBALL_ONLY_CFBD',lambda:modules['ncaaf_margin_tournament_v4'].run_cloud_research(
                dashboard_module=dash,storage_client=storage_client,bucket_name=bucket_name,log_func=log_func))
            if result_v4.get('status')!='RESEARCH_COMPLETE_NO_PROMOTION':raise RuntimeError('V54_V4_RESEARCH_NOT_COMPLETE')
            results['v4']={'status':result_v4['status'],'report_uri':result_v4.get('report_uri'),'football_source':'CFBD'}
            result_v42=audit.step('FULL_V42_ALL_FIVE_CFBD_FEATURE_FAMILIES',lambda:modules['ncaaf_v42_conditional_features'].run_cloud_research(
                dashboard_module=dash,storage_client=storage_client,bucket_name=bucket_name,log_func=log_func,
                v4_reference=result_v4))
            if not result_v42.get('predictions_uri'):raise RuntimeError('V54_V42_LEDGER_NOT_PUBLISHED')
            if result_v42.get('v4_ledger_exact_game_parity',{}).get('status')!='EXACT_PER_GAME':
                raise RuntimeError('V545_V42_SELECTED_CHAMPION_GAME_PARITY_NOT_PROVEN')
            if result_v42.get('frozen_champion')!=result_v4.get('selected_pre2024'):
                raise RuntimeError('V545_V42_REFERENCE_CHAMPION_CHANGED')
            results['v42']={'status':result_v42.get('status'),'report_uri':result_v42.get('report_uri'),
                            'football_source':'CFBD','selected_v4_research_winner':result_v42.get('frozen_champion'),
                            'v4_ledger_exact_game_parity':result_v42.get('v4_ledger_exact_game_parity')}
            # No old expert historical results or future labels; only CFBD reconstructed inputs.
            d=audit.step('CFBD_ONLY_EXPERT_GAME_KEY_ADAPTER',
                         lambda:truth.prepare_cfbd_expert_rule_frame(cfbd_sides))
            audit.emit('CFBD_EXPERT_ADAPTER_COVERAGE','PASS',
                       cfbd_game_keys=d.Game_Key.nunique(),team_sides=len(d),
                       game_key_source='CFBD_SEASON_PLUS_CFBD_GAME_ID',
                       game_start_source='CFBD_GAME_DATE',
                       source_vintage_certified=False,bookmaker_team_identity_used=False)
            flagged=audit.step('CFBD_ONLY_PATHI_BIGAL_RULE_REPLAY',
                               lambda:truth.replay_cfbd_expert_rules_without_leaking_current_results(d,ut))
            audit.emit('CFBD_EXPERT_SOURCE_IDENTITY_PARITY','PASS',
                       team_sides=len(flagged),cfbd_game_keys=flagged.Game_Key.nunique(),
                       season_dtype=str(flagged.Season.dtype),
                       check='ROW_ORDER_AND_SEASON_NUMERIC_VALUE_AND_ALL_EXACT_TEXT_IDS',
                       original_cfbd_identity_restored=True,
                       current_game_outcomes_not_used_for_rule_scoring=True)
            audit.emit('CFBD_EXPERT_RULE_VINTAGE_GUARD','PASS',
                       current_game_labels_suppressed_during_rule_replay=True,
                       prior_system_reliability_recomputed=False,
                       historical_cfbd_outcome_labels_restored_for_retrospective_grade=True)
            flagged,pathi_quote_audit=audit.step('CFBD_ONLY_PATHI_MARKET_KEY_CONTEXT',
                         lambda:truth.add_cfbd_market_only_pathi_context(flagged,ut))
            audit.emit('CFBD_PATHI_MARKET_KEY_COVERAGE','PASS',**pathi_quote_audit)
            audit.emit('CFBD_PATHI_SOURCE_IDENTITY_PARITY','PASS',
                       team_sides=len(flagged),cfbd_game_keys=flagged.Game_Key.nunique(),
                       season_dtype=str(flagged.Season.dtype),
                       original_cfbd_identity_restored=True,
                       market_quote_fields_only=True)
            expert_occ,expert_audit=audit.step('CFBD_ONLY_EXPERT_DIRECTIONAL_OCCURRENCES',lambda:truth.generate_cfbd_expert_occurrences(cfbd_sides,flagged,ut))
            audit.emit('CFBD_EXPERT_SYSTEM_COUNTS','PASS',**expert_audit)
            # Miner must consume the SAME CFBD-only historical frame, not
            # old ncaaf_historical_game_side_raw or market-staged contexts.
            miner_sides=flagged.loc[flagged.Is_Home.eq(1)].copy().reset_index(drop=True)
            truth.verify_cfbd_frame(miner_sides,require_unique_home=True)
            seasons=pd.to_numeric(miner_sides.Season,errors="raise").to_numpy(dtype=int)
            if len(seasons)!=len(miner_sides):raise RuntimeError("V54_MINER_SEASON_ROW_ALIGNMENT_BROKEN")
            miners={}
            def _run_all_miners():
                for market in ('spreads','h2h','totals'):
                    audit.emit('CFBD_ONLY_MINER_MARKET','START',market=market,games=len(miner_sides))
                    result=rv2.run_system_miner_v3(miner_sides,seasons,market,
                            dashboard_module=None,log_func=log_func,max_depth=5)
                    if not isinstance(result,dict):raise RuntimeError('V54_MINER_RETURNED_NON_DICT '+market)
                    status=str(result.get('status','UNKNOWN'))
                    audit.emit('CFBD_ONLY_MINER_MARKET','PASS',market=market,status_code=status,
                        mechanisms=len(result.get('mechanism_families') or []),
                        confirmed=sum(bool(m.get('confirmation_pass')) for m in result.get('mechanism_families',[])))
                    if market=='spreads' and status=='INSUFFICIENT_HISTORY':
                        raise RuntimeError('V54_CFBD_SPREAD_MINER_INSUFFICIENT_PRICED_HISTORY__NO_OLD_SOURCE_FALLBACK')
                    miners[market]=result
                return miners
            audit.step('FULL_SYSTEM_MINER_ALL_MARKETS_CFBD_ONLY',_run_all_miners)
            results['cfbd_only_miner']={k:{'status':v.get('status'),
                'mechanisms':len(v.get('mechanism_families') or []),
                'confirmed':sum(bool(m.get('confirmation_pass')) for m in v.get('mechanism_families',[])),
                'atoms':v.get('atoms'),'atom_family_counts':v.get('atom_family_counts',{}),
                'football_source':'CFBD'} for k,v in miners.items()}
        finally:
            if old_cache is None:
                if hasattr(dash,'_V1357_SPREAD_RESEARCH_CACHE'):delattr(dash,'_V1357_SPREAD_RESEARCH_CACHE')
            else:dash._V1357_SPREAD_RESEARCH_CACHE=old_cache
            if old_cfbd_guard is None:
                if hasattr(dash,'_V54_CFBD_ONLY_ACTIVE'):delattr(dash,'_V54_CFBD_ONLY_ACTIVE')
            else:dash._V54_CFBD_ONLY_ACTIVE=old_cfbd_guard
            if old_cfbd_source is not None:core.NCAAF_CFBD_MINER_FEATURE_TABLE=old_cfbd_source
            elif hasattr(core,'NCAAF_CFBD_MINER_FEATURE_TABLE'):delattr(core,'NCAAF_CFBD_MINER_FEATURE_TABLE')
            if isinstance(getattr(core,'_CFBD_CORE_CONTEXT_CACHE',None),dict):core._CFBD_CORE_CONTEXT_CACHE.clear()
        # Historical PT is external intelligence; its original file archives
        # remain separate from canonical CFBD football training features.
        pt=audit.step('ORIGINAL_PREDICTION_TRACKER_EXTERNAL_IMPORT',lambda:rv2.refresh_prediction_tracker_external(
            dashboard_module=dash,storage_client=storage_client,bucket_name=bucket_name,
            include_history=True,include_current=True,force=False,log_func=log_func),fatal=False)
        results['prediction_tracker_external']={'status':pt.get('status','UNKNOWN') if isinstance(pt,dict) else 'UNAVAILABLE',
            'football_training_source':'NOT_USED'}
        v5=audit.step('FULL_V5_MARGIN_TO_MARKET_RESEARCH',lambda:modules['ncaaf_v5_betting_research'].cloud_run(
            storage_client=storage_client,bucket_name=bucket_name,log_func=log_func))
        results['v5']={'status':v5.get('status'),'signals':v5.get('signals'),'report_uri':v5.get('report_uri')}
        v51=audit.step('FULL_V51_POINT_EDGE_ORIENTATION_REPLAY',lambda:modules['ncaaf_v51_edge_research'].cloud_run(
            storage_client=storage_client,bucket_name=bucket_name,log_func=log_func))
        if not v51.get('v4_prediction_verified_with_source'):raise RuntimeError('V54_EDGE_PARITY_NOT_VERIFIED')
        results['v51']={'status':v51.get('status'),'games':v51.get('physical_games'),
                        'model_game_rows':v51.get('dedup_model_game_rows')}
        edge_uri=v51.get('dedup_ledger_uri') or ''
        if not edge_uri.startswith(f'gs://{bucket_name}/research/ncaaf/v51_edge_attribution/runs/'):
            raise RuntimeError('V54_UNVERIFIED_EDGE_URI')
        edges=pd.read_csv(StringIO(bucket.blob(edge_uri[len(f'gs://{bucket_name}/'):]).download_as_text()),low_memory=False)
        # Immutable authoritative CFBD game ledger provides an additional firewall
        # on the edge join: only valid CFBD game IDs can be attributed.
        game_keys=set(zip(cfbd_home.Season.astype(int),cfbd_home.Game_ID.astype(str)))
        edge_keys=set(zip(pd.to_numeric(edges.Season).astype(int),edges.Game_ID.astype(str)))
        if not edge_keys.issubset(game_keys):
            raise RuntimeError('V54_V4_EDGES_HAVE_NON_CFBD_GAME_IDS '+str(len(edge_keys-game_keys)))
        audit.emit('V4_EDGE_JOIN','PASS',rows=len(edges),game_ids=edges.Game_ID.nunique(),
                   model_counts=edges.Model.value_counts().to_dict(),cfbd_authoritative_games=len(game_keys))
        def _aliases():
            names=list(pd.concat([edges.Team,edges.Opponent]).dropna().astype(str).unique())
            raw=rv2._pt_load_canonical_team_aliases(dash,names,log_func=log_func)
            aliases={_token(k):_token(v) for k,v in raw.items() if k and v}
            for n in names:aliases[_token(n)]=_token(n)
            return aliases
        aliases=audit.step('LEGACY_TEAM_ALIAS_IDENTITY_ONLY',_aliases,fatal=False) or {}
        # All expertise reconstructed only on CFBD derived records.
        miner_occ,miner_join=audit.step('CFBD_MINER_MECHANISM_MATCH_TO_V4',lambda:export_legacy_miner_mechanisms(
            edges,miner_sides,miners,rv2,None,aliases=aliases))
        occ=pd.concat([expert_occ,miner_occ],ignore_index=True).drop_duplicates(
                   ['Season','Game_ID','Family','System_ID','Side'])
        join_audit={**expert_audit,'cfbd_miner_occurrences':len(miner_occ),
                    'total_cfbd_reconstructed_occurrences':len(occ),'status':'CFBD_ONLY_RETROSPECTIVE'}
        audit.emit('CFBD_LEGACY_BRAIN_MATCH','PASS',**{k:v for k,v in join_audit.items() if k!='status'})
        canon_edges=audit.step('V548_CANONICAL_CFBD_V51_GAME_IDS',lambda:_canonical_cfbd_research_join_keys(
            edges,source='V51_CORE_EDGE_LEDGER'))
        canon_occ=audit.step('V548_CANONICAL_CFBD_BRAIN_GAME_IDS',lambda:_canonical_cfbd_research_join_keys(
            occ,source='CFBD_RECONSTRUCTED_SYSTEM_OCCURRENCES'))
        audit.emit('V548_CFBD_GAME_ID_JOIN_PARITY','PASS',
                   core_game_id_dtype=str(canon_edges.Game_ID.dtype),
                   expert_game_id_dtype=str(canon_occ.Game_ID.dtype),
                   core_season_dtype=str(canon_edges.Season.dtype),
                   expert_season_dtype=str(canon_occ.Season.dtype),
                   core_model_game_rows=len(canon_edges),
                   brain_system_occurrences=len(canon_occ),
                   common_physical_games=len(set(zip(canon_edges.Season,canon_edges.Game_ID)) &
                                             set(zip(canon_occ.Season,canon_occ.Game_ID))),
                   source_keys_unchanged=True,bet_authority=0)
        sys_score,fam_score=audit.step('FULL_CFBD_SYSTEM_INCREMENTAL_ATTRIBUTION',
                                      lambda:exploratory_attribution(canon_edges,canon_occ))
        shadow,shadow_policy=audit.step('V5_CFBD_EDGE_SHADOW_AUTHORITY',lambda:policy_shadow(
            canon_edges,canon_occ,verified_count=0,source_vintage_certified=False))
        v52=audit.step('V52_GENUINE_ARCHIVED_SIGNALS_SEPARATE_PROVENANCE',lambda:modules['ncaaf_v52_brain_research'].cloud_run(
            storage_client=storage_client,bucket_name=bucket_name,log_func=log_func),fatal=False)
        results['v52_archive_bridge']={'status':v52.get('status') if v52 else 'UNAVAILABLE',
             'verified_records':v52.get('verified_system_game_records',0) if v52 else 0}
        audit.emit('V54_FINAL_AUTHORITY','SHADOW_ONLY',brain_occurrences=len(occ),
                   systems_attributed=len(sys_score),families_attributed=len(fam_score),
                   bet_authority=0,source_vintage_certified=False,roi='UNAVAILABLE')
        for name,df in [('cfbd_reconstructed_system_occurrences.csv',occ),
                        ('system_attribution.csv',sys_score),('family_attribution.csv',fam_score),
                        ('v5_shadow_edge_candidates.csv',shadow)]:
            outputs[name]=(df.to_csv(index=False),'text/csv')
        status='COMPLETE_FULL_RESEARCH__SHADOW_AUTHORITY__VINTAGE_HOLD'
    except Exception as ex:
        status='FAILED_FULL_RESEARCH__AUDIT_PUBLISHED'
        results['failure']={'error_type':type(ex).__name__,'error':str(ex)[:1500]}
        audit.emit('FULL_RUN','FAIL',error_type=type(ex).__name__,error=str(ex)[:1200])
    report={'version':VERSION,'status':status,'generated_utc':_utc(),'run_id':audit.run_id,
            'stages':audit.events,'research_components':results,'source_manifest':manifest,
            'cfbd_source_truth':cfbd_audit,'runtime_alignment':locals().get('alignment',{}),
            'legacy_exact_source_constants':locals().get('constants',{}),
            'legacy_occurrence_join':join_audit,'miner_occurrence_join':miner_join,
            'shadow_policy':shadow_policy,'historical_roi':None,'production_modified':False,
            'bet_authority':0,'source_vintage_certified':False,'not_a_pristine_holdout':True,
            'historical_football_source':'CFBD_ONLY',
            'source_contract':'CFBD_GAMES_SCORES_FEATURES__MARKET_PRICE_ONLY__EXTERNAL_BRAINS_SEPARATE',
            'source_vintage_warning':'Archived CFBD pregame feature publication timestamps still unverified',
            'recommended_live_model':'V4_POINT_EDGE__CFBD_RESEARCHED_STAT__MARKET__SHADOW_UNTIL_PROSPECTIVE_VERIFIED'}
    outputs['audit_events.jsonl']=(''.join(json.dumps(z,default=str,allow_nan=False)+'\n' for z in audit.events),'application/x-ndjson')
    outputs['source_manifest.json']=(json.dumps(manifest,indent=2,default=str,allow_nan=False),'application/json')
    report=save_run(bucket,audit,report,outputs,latest=True)
    audit.log('[NCAAF-V54] status='+status+' report='+report['report_uri']+' stages='+str(len(audit.events))+' authority=0')
    if status.startswith('FAILED'):
        raise RuntimeError('V54_FULL_RESEARCH_FAILED__SEE_AUDIT_REPORT '+report['report_uri']+' cause='+str(results.get('failure')))
    return report
