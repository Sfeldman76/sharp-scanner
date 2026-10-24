"""Read-only real-data identity preflight. No refit, GCS writes or betting changes.

Execute in authenticated Cloud Shell *before* redeploying costly Heavy Research.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import pandas as pd
from ncaaf_historical_brain_replay_v1 import (_stage_cfbd,_stage_context,_attach_context,
                                               load_bigquery_source_frames)
from ncaaf_brain_source_recovery import retrospective_core_from_immutable_replay


def _core_from_gcs(bucket):
    from google.cloud import storage
    client=storage.Client()
    # Only immutable replay audit objects; do not assume latest pointer schema.
    prefix='research/ncaaf/v55/ats_primary/replay_runs/'
    candidates=sorted(blob.name[:-len('/audit.json')] for blob in client.list_blobs(bucket,prefix=prefix)
                      if blob.name.endswith('/audit.json'))
    for key in reversed(candidates):
        value,err=retrospective_core_from_immutable_replay(client,bucket,
                  {'model_replay_artifacts':f'gs://{bucket}/{key}'})
        if value is not None:return value,{'status':'HASH_VERIFIED_RETROSPECTIVE','replay_prefix':key,'core_rows':len(value)}
    return None,{'status':'NOT_FOUND_OR_NOT_CHECKSUM_VERIFIED','audit_paths_seen':len(candidates)}


def inspect(project,dataset,bucket,include_core=True):
    cfbd,context,aliases=load_bigquery_source_frames(project=project,dataset=dataset)
    u,_,meta=_stage_cfbd(cfbd)
    ctx,_=_stage_context(context)
    g,join=_attach_context(u,ctx,aliases=aliases)
    years=g.groupby('season').agg(cfbd_games=('game_id','size'),context_matches=('has_context','sum')).reset_index()
    out={'status':'IDENTITY_PREFLIGHT_READ_ONLY','generated_utc':datetime.now(timezone.utc).isoformat(),
         'project':project,'dataset':dataset,'bet_authority':0,'production_mutation':False,
         'cfbd_games':len(g),'context_home_rows':len(ctx),'context_match_ratio':round(float(g.has_context.mean()),4),
         'context_join':join,'per_year':years.to_dict('records'),
         'core':{'status':'DISABLED'}}
    if include_core:
        core,cmeta=_core_from_gcs(bucket)
        if core is None:out['core']=cmeta
        else:
            lookup=g[['season','game_id']].copy()
            lookup['source_game_id']=g.source_game_id.astype('string').fillna('')
            ck=pd.MultiIndex.from_frame(core[['season','game_id']].astype({'game_id':'string'}))
            dk=pd.MultiIndex.from_frame(lookup[['season','game_id']].astype({'game_id':'string'}))
            sk=pd.MultiIndex.from_arrays([lookup.season,lookup.source_game_id])
            direct=int(ck.isin(dk).sum()); via_context=int(ck.isin(sk).sum())
            out['core']={**cmeta,'same_cfbd_id':direct,'same_legacy_source_id':via_context,
                         'either_known_id':int((ck.isin(dk)|ck.isin(sk)).sum())}
    out['next_action']='NO_HEAVY_RUN_UNTIL_MATCHES_VERIFIED' if out['context_join']['matched_context_games']==0 or out['core'].get('either_known_id',1)==0 else 'REVIEW_MATCH_COVERAGE_BEFORE_HEAVY'
    return out


if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--project',default='sharplogger');a.add_argument('--dataset',default='sharp_data')
    a.add_argument('--bucket',default='sharp-models');a.add_argument('--out',default='replay_identity_preflight.json')
    a.add_argument('--skip-core',action='store_true')
    args=a.parse_args()
    result=inspect(args.project,args.dataset,args.bucket,include_core=not args.skip_core)
    Path(args.out).write_text(json.dumps(result,indent=2,default=str))
    print('Real-data identity preflight:', result['next_action'])
    print('CFBD games:',result['cfbd_games'],'matched historical context:',result['context_join']['matched_context_games'])
    print('CORE:',result['core'])
    print('Report:',str(Path(args.out).resolve()))
