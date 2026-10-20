"""NCAAF V5.3 complete research orchestration, legacy-brain bridge and audit journal.

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

VERSION = 'NCAAF_V53_FULL_RESEARCH_LEGACY_INTEGRATION_20261020_V1'
ROOT = 'research/ncaaf/v53_full_research'
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
        self.log('[NCAAF-V53-AUDIT] '+json.dumps(event,sort_keys=True,default=str,allow_nan=False))
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


def exploratory_attribution(edges,legacy):
    """Matched conditional comparisons, never historically confirmed bets."""
    base=edges.copy()
    assert not base.duplicated(['Season','Game_ID','Model']).any(), 'V53_DUPLICATE_MODEL_GAME'
    work=legacy.copy()
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
    out=edges[['Season','Game_ID','Game_Date','Team','Opponent','Model','Side','Result','Abs_Edge','Edge_Bin','Selection_Type','Period']].copy()
    if not legacy.empty:
        z=legacy.groupby(['Season','Game_ID','Family']).Side.agg(
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
    """Full research in one job. Reports partial failed audits before re-raising."""
    audit=Audit(log_func);bucket=storage_client.bucket(bucket_name)
    manifest={};results={};outputs={}
    try:
        # Full V4/V4.2 archive research is an explicitly selected V5.3 workflow.
        # Check the original V4 safeguard BEFORE expensive legacy history/cache fits.
        def _require_v4_archive_opt_in():
            import os
            value = str(os.getenv('NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED', '0')).strip().lower()
            if value not in ('1', 'true', 'yes', 'on'):
                raise RuntimeError(
                    'V53_ARCHIVE_OPT_IN_MISSING__SET_NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED=1__'
                    'DASHBOARD_V53_ENV_MAPPING_MAY_BE_OUTDATED')
            return {'archive_research_enabled': True, 'scope': 'V53_FULL_RESEARCH_ONLY'}
        audit.step('V4_ARCHIVE_RESEARCH_OPT_IN_PREFLIGHT', _require_v4_archive_opt_in)
        if modules is None:
            modules={n:importlib.import_module(n) for n in REQUIRED_LEGACY}
        contract=audit.step('LEGACY_MODULE_CONTRACT',lambda:module_contract(modules))
        if any(v['status']!='PASS' for v in contract.values()):
            raise RuntimeError('V53_LEGACY_MISSING_FUNCTIONS:'+str({k:v['missing'] for k,v in contract.items() if v['missing']}))
        alignment=audit.step('MATCHED_DASHBOARD_UTILS_BUILD',lambda:runtime_alignment(modules))
        if alignment['status']!='PASS':raise RuntimeError('V53_MIXED_LEGACY_RUNTIME_BUILD '+str(alignment))
        constants=audit.step('LEGACY_SOURCE_CONSTANTS',lambda:legacy_source_constants(
            modules['ncaaf_research_v2'],modules['utils']))
        if constants['status']!='PASS':raise RuntimeError('V53_LEGACY_SOURCE_CONSTANT_MISMATCH '+str(constants))
        if bq_client is None:
            from google.cloud import bigquery
            bq_client=bigquery.Client(project='sharplogger')
        manifest=audit.step('EXACT_LEGACY_SOURCE_MANIFEST',lambda:source_manifest(storage_client=storage_client,bucket_name=bucket_name,bq_client=bq_client))
        audit.emit('SOURCE_PATHS','PASS',modules={k:v['sha256'] for k,v in contract.items()},
                   gcs_sources={k:v.get('exists',v.get('sample_objects')) for k,v in manifest['objects'].items()},
                   bq_sources={k:v.get('status') for k,v in manifest['tables'].items()})
        dash=modules['sharp_line_dashboard']
        audit.step('BUILD_HISTORICAL_EXPERT_CACHE',lambda:dash.fit_historical_ncaaf_core_expert('spreads',log_func=log_func))
        audit.step('BUILD_STAT_CACHE',lambda:dash.fit_ncaaf_statistical_brain(log_func=log_func))
        cache=getattr(dash,'_V1357_SPREAD_RESEARCH_CACHE',{}) or {}
        if cache.get('games') is None or getattr(cache['games'],'empty',True) or cache.get('miner_games') is None:
            raise RuntimeError('V53_HISTORICAL_CACHE_MISSING_AFTER_LEGACY_BUILD')
        audit.emit('CACHE_KEYS','PASS',games=len(cache['games']),miner_games=len(cache['miner_games']),
                   expert_history_systems=len(getattr(dash,'_V143_SYSTEM_HISTORY_CACHE',{}) or {}),
                   seasons=sorted(int(x) for x in pd.to_numeric(cache['games'].get('Season'),errors='coerce').dropna().unique()))
        # Entire proven football-only research, not a quick report-only incremental pass.
        result_v4=audit.step('FULL_V4_FOOTBALL_MARGIN_RESEARCH',lambda:modules['ncaaf_margin_tournament_v4'].run_cloud_research(
            dashboard_module=dash,storage_client=storage_client,bucket_name=bucket_name,log_func=log_func))
        if result_v4.get('status')!='RESEARCH_COMPLETE_NO_PROMOTION':raise RuntimeError('V53_V4_RESEARCH_NOT_COMPLETE')
        results['v4']={'status':result_v4['status'],'report_uri':result_v4.get('report_uri')}
        result_v42=audit.step('FULL_V42_ALL_FIVE_FEATURE_FAMILIES',lambda:modules['ncaaf_v42_conditional_features'].run_cloud_research(
            dashboard_module=dash,storage_client=storage_client,bucket_name=bucket_name,log_func=log_func))
        if not result_v42.get('predictions_uri'):raise RuntimeError('V53_V42_LEDGER_NOT_PUBLISHED')
        results['v42']={'status':result_v42.get('status'),'report_uri':result_v42.get('report_uri')}
        core=audit.step('LEGACY_FULL_CORE_SPECIALIST_RESEARCH',lambda:modules['ncaaf_core_challenger_v2'].run_ncaaf_core_challenger_v2(
            dashboard_module=dash,bucket_name=bucket_name,storage_client=storage_client,log_func=log_func,hard_fail=True))
        if core.get('status')!='NCAAF_CORE_CHALLENGER_V2_COMPLETE':raise RuntimeError('V53_LEGACY_CORE_INCOMPLETE')
        results['legacy_core']={'status':core.get('status'),'best':(core.get('best_challenger') or {}).get('name')}
        legacy=audit.step('FULL_LEGACY_PATHI_BIGAL_PT_MINER_STATE_RESEARCH',lambda:run_legacy_with_promotion_held(
            modules['ncaaf_research_v2'],audit=audit,
            dashboard_module=dash,utils_module=modules['utils'],production_module=None,
            bucket_name=bucket_name,storage_client=storage_client,log_func=log_func,hard_fail=True))
        if legacy.get('status')!='NCAAF_RESEARCH_V2_COMPLETE':raise RuntimeError('V53_LEGACY_RESEARCH_INCOMPLETE')
        miners=legacy.get('system_miner_v3') or {}
        results['legacy_brains']={'status':legacy.get('status'),'market_keys':list(miners),
          'confirmed_mechanisms_by_market':{k:sum(bool(m.get('confirmation_pass')) for m in v.get('mechanism_families',[])) for k,v in miners.items()},
          'pathi_bigal':legacy.get('expert_side_bridge'),
          'pt':{'status':(legacy.get('external_rating_metamodel') or {}).get('status'),
                'source':(legacy.get('external_rating_metamodel') or {}).get('source')},
          'state':legacy.get('historical_state_context_bridge'),
          'cfbd':legacy.get('cfbd_context_bridge'),
          'travel_rivalry':legacy.get('situational_context_bridge')}
        audit.emit('LEGACY_BRAINS_COUNTS','PASS',miner_counts=results['legacy_brains']['confirmed_mechanisms_by_market'],
                   bigal_obs=(legacy.get('big_al_observation_ledger') or {}).get('ncaaf_observations'),
                   history_cache=len(getattr(dash,'_V143_SYSTEM_HISTORY_CACHE',{}) or {}))
        v5=audit.step('FULL_V5_MARGIN_TO_MARKET_RESEARCH',lambda:modules['ncaaf_v5_betting_research'].cloud_run(
            storage_client=storage_client,bucket_name=bucket_name,log_func=log_func))
        results['v5']={'status':v5.get('status'),'signals':v5.get('signals'),'report_uri':v5.get('report_uri')}
        v51=audit.step('FULL_V51_POINT_EDGE_ORIENTATION_REPLAY',lambda:modules['ncaaf_v51_edge_research'].cloud_run(
            storage_client=storage_client,bucket_name=bucket_name,log_func=log_func))
        if not v51.get('v4_prediction_verified_with_source'):raise RuntimeError('V53_EDGE_PARITY_NOT_VERIFIED')
        results['v51']={'status':v51.get('status'),'games':v51.get('physical_games'),'model_game_rows':v51.get('dedup_model_game_rows')}
        edge_uri=v51.get('dedup_ledger_uri') or ''
        if not edge_uri.startswith(f'gs://{bucket_name}/research/ncaaf/v51_edge_attribution/runs/'):
            raise RuntimeError('V53_BAD_UNIQUE_EDGE_URI')
        edges=pd.read_csv(StringIO(bucket.blob(edge_uri[len(f'gs://{bucket_name}/'):]).download_as_text()),low_memory=False)
        audit.emit('V4_EDGE_JOIN','PASS',rows=len(edges),game_ids=edges.Game_ID.nunique(),
                   model_counts=edges.Model.value_counts().to_dict(),missing_scores=int((~edges.Result.isin(['WIN','LOSS','PUSH'])).sum()))
        def _source_aliases():
            rv=modules['ncaaf_research_v2']
            source_names=list(pd.concat([edges.Team,edges.Opponent]).dropna().astype(str).unique())
            raw=rv._pt_load_canonical_team_aliases(dash,source_names,log_func=log_func)
            aliases={_token(k):_token(v) for k,v in raw.items() if k and v}
            # In V4, identity is itself canonical, even if a historic alias
            # table maps to a conflicting older label. Exact canonical identity wins.
            for name in source_names:aliases[_token(name)]=_token(name)
            return aliases
        aliases=audit.step('EXISTING_NCAAF_CANONICAL_TEAM_ALIAS_RESOLVER',_source_aliases,fatal=False) or {}
        audit.emit('TEAM_ALIAS_COVERAGE','PASS' if aliases else 'HOLD_NO_ALIAS_MAP',alias_keys=len(aliases))
        occurrences,join_audit=audit.step('LEGACY_PATHI_BIGAL_OCCURRENCE_JOIN',lambda:export_legacy_occurrences(
            edges,getattr(dash,'_V143_SYSTEM_HISTORY_CACHE',{}) or {},aliases=aliases))
        audit.emit('LEGACY_JOIN_COVERAGE','PASS' if join_audit['joined'] else 'HOLD_NO_MATCHED_OCCURRENCES',**join_audit)
        miner_occ,miner_join=audit.step('MINER_CONFIRMED_MECHANISM_FIRING_TO_V4',lambda:export_legacy_miner_mechanisms(
            edges,(getattr(dash,'_V1357_SPREAD_RESEARCH_CACHE',{}) or {}).get('miner_games'),
            miners,modules['ncaaf_research_v2'],dash,aliases=aliases))
        audit.emit('MINER_JOIN_COVERAGE',miner_join['status'],**{k:v for k,v in miner_join.items() if k!='status'})
        occurrences=pd.concat([occurrences,miner_occ],ignore_index=True).drop_duplicates(
            ['Season','Game_ID','Family','System_ID','Side'])
        sys_score,fam_score=audit.step('MATCHED_LEGACY_CONDITIONAL_ATTRIBUTION',lambda:exploratory_attribution(edges,occurrences))
        shadow,shadow_policy=audit.step('V5_SHADOW_AUTHORITY_POLICY',lambda:policy_shadow(
            edges,occurrences,verified_count=0,source_vintage_certified=False))
        v52=audit.step('V52_VERIFIED_ARCHIVE_SOURCE_BRIDGE',lambda:modules['ncaaf_v52_brain_research'].cloud_run(
            storage_client=storage_client,bucket_name=bucket_name,log_func=log_func),fatal=False)
        results['v52_archive_bridge']={'status':v52.get('status') if v52 else 'UNAVAILABLE',
             'verified_records':v52.get('verified_system_game_records',0) if v52 else 0}
        if v52 is not None and v52.get('verified_system_game_records',0):
            # Do not implicitly grant votes until they are audited against the V5.3 shadow selection.
            shadow_policy['verified_archived_system_game_records']=v52['verified_system_game_records']
            audit.emit('GCS_ARCHIVE_SYSTEMS','AVAILABLE_REQUIRES_AUDITED_V53_JOIN',count=v52['verified_system_game_records'])
        audit.emit('V53_FINAL_AUTHORITY','SHADOW_ONLY',brain_occurrences=len(occurrences),
                   systems_attributed=len(sys_score),families_attributed=len(fam_score),
                   bet_authority=0,source_vintage_certified=False,roi='UNAVAILABLE')
        for name,df in [('legacy_occurrences.csv',occurrences),('system_attribution.csv',sys_score),
                        ('family_attribution.csv',fam_score),('v5_shadow_edge_candidates.csv',shadow)]:
            outputs[name]=(df.to_csv(index=False),'text/csv')
        status='COMPLETE_FULL_RESEARCH__SHADOW_AUTHORITY__VINTAGE_HOLD'
    except Exception as ex:
        status='FAILED_FULL_RESEARCH__AUDIT_PUBLISHED'
        results['failure']={'error_type':type(ex).__name__,'error':str(ex)[:1500]}
        audit.emit('FULL_RUN','FAIL',error_type=type(ex).__name__,error=str(ex)[:1200])
    report={'version':VERSION,'status':status,'generated_utc':_utc(),'run_id':audit.run_id,
            'stages':audit.events,'research_components':results,'source_manifest':manifest,'runtime_alignment':locals().get('alignment',{}),
            'legacy_exact_source_constants':locals().get('constants',{}),
            'legacy_occurrence_join':locals().get('join_audit',{'status':'NOT_REACHED'}),
            'miner_occurrence_join':locals().get('miner_join',{'status':'NOT_REACHED'}),
            'shadow_policy':locals().get('shadow_policy',{'status':'NOT_REACHED','bet_authority':0}),
            'recommended_live_model':'V4_FOOTBALL_CORE__POINT_EDGE__LEGACY_STAT__MARKET__SHADOW_UNTIL_PROSPECTIVE_VERIFIED',
            'historical_roi':None,'production_modified':False,'bet_authority':0,
            'not_a_pristine_holdout':True,'source_vintage_certified':False}
    outputs['audit_events.jsonl']=(''.join(json.dumps(z,default=str,allow_nan=False)+'\n' for z in audit.events),'application/x-ndjson')
    outputs['source_manifest.json']=(json.dumps(manifest,indent=2,default=str,allow_nan=False),'application/json')
    report=save_run(bucket,audit,report,outputs,latest=True)
    audit.log('[NCAAF-V53] status='+status+' report='+report['report_uri']+' stages='+str(len(audit.events))+' authority=0')
    if status.startswith('FAILED'):
        raise RuntimeError('V53_FULL_RESEARCH_FAILED__SEE_AUDIT_REPORT '+report['report_uri']+' cause='+str(results.get('failure')))
    return report
