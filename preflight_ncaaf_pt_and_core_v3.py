"""Read-only PT/Core replay coverage preflight; requires authenticated Cloud Shell."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from google.cloud import storage
from ncaaf_historical_brain_replay_v1 import (load_bigquery_source_frames, _stage_cfbd,
    _stage_context, _attach_context, pt_archive_inputs, attach_pt, compare_to_core)
from preflight_ncaaf_replay_identity import _core_from_gcs
import pandas as pd


def inspect(project,dataset,bucket):
    cfbd,context,aliases=load_bigquery_source_frames(project=project,dataset=dataset)
    u,_,_=_stage_cfbd(cfbd)
    ctx,_=_stage_context(context)
    games,join=_attach_context(u,ctx,aliases=aliases)
    sc=storage.Client(project=project)
    pt,pt_gcs=pt_archive_inputs(sc,bucket)
    pt_signals,pt_meta=attach_pt(games,pt,aliases=aliases)
    core,receipt=_core_from_gcs(bucket)
    core_rows,_,core_meta=compare_to_core(games,pt_signals,core)
    # Source-driven context missing: a CFBD game that is not covered cannot
    # be fixed by adding an alias. Count unlinked context separately.
    linked=set(zip(games.loc[games.has_context,'season'].astype(int),
                   games.loc[games.has_context,'source_game_id'].astype(str)))
    unmapped=ctx.loc[[
        (int(r.season),str(r.source_game_id)) not in linked
        for r in ctx.itertuples(index=False)
    ],['season','source_game_id','Team','Opponent','Game_Date']]
    return {'status':'NCAAF_READ_ONLY_PT_CORE_PREFLIGHT_V3',
        'generated_utc':datetime.now(timezone.utc).isoformat(),'bet_authority':0,
        'production_mutation':False,'cfbd_games':len(games),
        'context_matched':int(games.has_context.sum()),
        'context_total_source':len(ctx),
        'unmapped_context_rows':len(unmapped),
        'unmapped_context_examples':unmapped.head(40).to_dict('records'),
        'cfbd_games_not_in_context':len(games)-int(games.has_context.sum()),
        'pt_source':pt_gcs,'pt_matching':pt_meta,
        'core_source':receipt,'core_attrition':core_meta,
        'core_compared_rows':len(core_rows),
        'historical_publication_vintage_certified':False}

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--project',default='sharplogger');a.add_argument('--dataset',default='sharp_data')
    a.add_argument('--bucket',default='sharp-models');a.add_argument('--out',default='pt_core_identity_v3.json')
    args=a.parse_args()
    report=inspect(args.project,args.dataset,args.bucket)
    Path(args.out).write_text(json.dumps(report,indent=2,default=str))
    print('NCAAF historical preflight:',report['status'])
    print('CFBD:',report['cfbd_games'],'context:',report['context_matched'])
    print('PT match:',report['pt_matching'].get('matched_games'),'/ ',report['pt_matching'].get('pt_source_rows'))
    print('CORE attrition:',json.dumps(report['core_attrition'],sort_keys=True,default=str))
    print('Report:',str(Path(args.out).resolve()))
