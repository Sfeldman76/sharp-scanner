"""Read-only discovery of NCAAF brain data sources in the existing BigQuery/GCS project.

IMPORTANT: Schema discovery does not prove pregame provenance or mean a row fired.
No BigQuery table contents are queried unless an explicit source_map.json is provided.
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re

import pandas as pd
from ncaaf_v52_brain_bridge import FAMILIES, FQ_TABLE, IDENTIFIER

CORE_TABLES = (
 'scores_with_features', 'sharp_scores_full', 'bigal_game_context_enriched',
 'ncaaf_historical_game_side_raw', 'ncaaf_historical_game_side_context',
 'ncaaf_cfbd_miner_features_v1', 'ncaaf_v13_forward_shadow_predictions',
 'ncaaf_production_v1_predictions', 'ncaaf_production_v1_results',
)
SEMANTIC_ALIASES={
 'season': ['Season','season'],
 'game_id': ['Game_ID','game_id','CFBD_Game_ID','cfbd_game_id','physical_game_id'],
 'game_date': ['Game_Date','game_date','Date','Game_Start','feat_Game_Start'],
 'team': ['Team','team','Home_Team_Norm','Home_Team'],
 'opponent': ['Opponent','opponent','Away_Team_Norm','Away_Team'],
 'side': ['Side','side','Signal_Side','selection_side'],
 'side_team': ['Side_Team','Selected_Team','Outcome','Team'],
 'family':['Family','family','System_Family','system_family'],
 'system_id':['System_ID','system_id','Rule_ID','rule_id'],
 'snapshot_utc':['Snapshot_UTC','snapshot_utc','Snapshot_Timestamp','prediction_timestamp','captured_at'],
 'kickoff_utc':['Kickoff_UTC','kickoff_utc','Game_Start','feat_Game_Start','Commence_Hour'],
 'rule_declared_utc':['Rule_Declared_UTC','rule_declared_utc','Rule_Registered_At'],
 'source_gcs_uri':['Source_GCS_URI','source_gcs_uri','original_snapshot_gcs_uri'],
 'source_artifact_sha256':['Source_Artifact_SHA256','source_artifact_sha256','artifact_sha256'],
}


def _suggest_mapping(columns):
    cols=set(columns)
    mp={}
    for key,aliases in SEMANTIC_ALIASES.items():
        found=next((c for c in aliases if c in cols),None)
        if found:mp[key]=found
    return mp


def _candidate_table(table_name, columns):
    cl=list(columns); colsl={x.lower() for x in cl}
    prefixes={
      'BIG_AL':tuple(x for x in cl if x.startswith(('BigAl_','Big_Al_'))),
      'PATHI':tuple(x for x in cl if x.startswith(('Pathi_','PATHI_'))),
      'PREDICTION_TRACKER':tuple(x for x in cl if x.startswith(('PT_','Prediction_Tracker_','PredictionTracker_'))),
      'SYSTEM_MINER':tuple(x for x in cl if x.startswith(('Miner_','System_Miner_','NCAAF_Miner_'))),
      'STATE_BOUNCE_BACK':tuple(x for x in cl if x.startswith(('State_','Bounce_Back_','StateMiner_'))),
    }
    mp=_suggest_mapping(cl)
    has_long=all(k in mp for k in ('season','game_id','family','system_id','side'))
    score=(70 if has_long else 0)+sum(min(len(v),25) for v in prefixes.values())
    score+= 8 if table_name in CORE_TABLES else 0
    score+= 7 if any(k in table_name.lower() for k in ['system','miner','pathi','bigal','prediction','ncaaf']) else 0
    proof=all(k in mp for k in ('snapshot_utc','kickoff_utc','rule_declared_utc','source_gcs_uri','source_artifact_sha256'))
    formats=['long'] if has_long else []
    if any(prefixes.values()):formats.append('wide_context_candidate')
    return {'table':table_name,'score':score,'columns':len(cl),
            'family_column_counts':{k:len(v) for k,v in prefixes.items()},
            'family_column_examples':{k:list(v[:12]) for k,v in prefixes.items() if v},
            'potential_formats':formats,'semantic_mapping_suggestion':mp,
            'has_all_archive_proof_fields':proof,
            'warning':'COLUMN NAMES ONLY; no historical records or provenance have been verified'}


def discover(client, dataset='sharplogger.sharp_data', max_tables=1500):
    if not re.fullmatch(r'[A-Za-z0-9_-]+\.[A-Za-z_][A-Za-z0-9_]*',dataset):
        raise ValueError('V52_INVALID_DATASET')
    items=[];errors=[];total=0
    for t in client.list_tables(dataset):
        total+=1
        if total>max_tables:
            errors.append('TABLE_LIMIT_REACHED');break
        try:
            meta=client.get_table(f'{dataset}.{t.table_id}')
            columns=[c.name for c in meta.schema]
            rank=_candidate_table(t.table_id,columns)
            if rank['score']>0:
                rank.update({'table':f'{dataset}.{t.table_id}',
                    'table_type':str(getattr(t,'table_type','UNKNOWN')),
                    'created_utc':getattr(meta,'created',None).isoformat() if getattr(meta,'created',None) else None,
                    'modified_utc':getattr(meta,'modified',None).isoformat() if getattr(meta,'modified',None) else None})
                items.append(rank)
        except Exception as exc:
            errors.append(f'{t.table_id}: {type(exc).__name__}: {str(exc)[:160]}')
    items.sort(key=lambda x:(-x['score'],x['table']))
    return {'status':'SCHEMA_DISCOVERY_NOT_PROOF_OF_PREGAME_DATA',
            'generated_utc':datetime.now(timezone.utc).isoformat(),
            'dataset':dataset,'scanned_tables':min(total,max_tables),
            'candidate_tables':items,'errors':errors,
            'next_action':'Review source inventory and configure exact mappings for authentic signals. Rebuilt feature views are research context only.'}


def extract_configured_source(client, spec, *, dataset='sharplogger.sharp_data', max_rows=150000):
    """Read explicit named data columns only; no SELECT * or auto-invented side."""
    table=str(spec.get('table',''))
    if not FQ_TABLE.fullmatch(table) or not table.startswith(dataset+'.'):
        raise ValueError('V52_SOURCE_TABLE_NOT_ALLOWLISTED:'+table)
    fmt=spec.get('format','long')
    if fmt not in ('long','wide'):raise ValueError('V52_BAD_FORMAT')
    mp=spec.get('mapping') or {}
    if 'season' not in mp:raise ValueError('V52_EXTRACT_SEASON_COLUMN_REQUIRED')
    fields=list(dict.fromkeys([*mp.values(),*(spec.get('system_columns') or {}).keys()]))
    if not fields or not all(IDENTIFIER.fullmatch(x) for x in fields):
        raise ValueError('V52_BAD_COLUMN_NAME')
    if not IDENTIFIER.fullmatch(mp['season']):raise ValueError('V52_BAD_SEASON_COLUMN')
    metadata=client.get_table(table)
    present={f.name for f in metadata.schema}
    missing=sorted(set(fields)-present)
    if missing:raise ValueError('V52_MISSING_MAPPED_TABLE_FIELDS:'+repr(missing))
    if max_rows<1 or max_rows>1_000_000:raise ValueError('V52_MAX_ROWS_OUT_OF_RANGE')
    quoted=', '.join(f'`{x}`' for x in fields)
    # LIMIT is only an operational safety bound. Any truncated extraction is
    # clearly marked INCOMPLETE and cannot be used for full attribution claims.
    from google.cloud import bigquery
    args=[bigquery.ScalarQueryParameter('maxrows','INT64',int(max_rows)+1)]
    params=[f'SAFE_CAST(`{mp["season"]}` AS INT64) BETWEEN 2021 AND 2025']
    if 'Sport' in present:
        params.append('UPPER(TRIM(CAST(`Sport` AS STRING))) IN UNNEST(@sports)')
        args.append(bigquery.ArrayQueryParameter('sports','STRING',spec.get('sport_values',['NCAAF','NCAA FOOTBALL','COLLEGE FOOTBALL'])))
    sql=f'SELECT {quoted} FROM `{table}` WHERE '+ ' AND '.join(params) +' LIMIT @maxrows'
    frame=client.query(sql,job_config=bigquery.QueryJobConfig(query_parameters=args)).to_dataframe(create_bqstorage_client=False)
    truncated=len(frame)>max_rows
    if truncated:frame=frame.iloc[:max_rows].copy()
    return frame,{'table':table,'extracted_rows':len(frame),'truncated':truncated,
                  'status':'INCOMPLETE_CAPPED_SOURCE' if truncated else 'QUERY_COMPLETE',
                  'note':'Table rows may have been retrospectively materialized; extraction alone does not establish pregame availability'}


def discover_gcs(storage_client, bucket='sharp-models', prefixes=None, max_objects_per_prefix=200):
    prefixes=prefixes or ['research/ncaaf/','ncaaf/']
    results=[]
    for prefix in prefixes:
        seen=0
        for blob in storage_client.list_blobs(bucket,prefix=prefix,max_results=max_objects_per_prefix):
            if any(k in blob.name.lower() for k in ('system','miner','pathi','bigal','snapshot','pt_','ledger','brain')):
                results.append({'uri':f'gs://{bucket}/{blob.name}',
                    'created_utc':blob.time_created.isoformat() if blob.time_created else None,
                    'bytes':blob.size,'archive_proof':'UNASSESSED'})
            seen+=1
    return {'candidate_gcs_objects':results,
       'status':'PREFIX_SAMPLE_ONLY__NO_ASSUMPTION_OF_COMPLETENESS',
       'max_objects_per_prefix':max_objects_per_prefix}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='sharplogger.sharp_data')
    p.add_argument('--output',default='ncaaf_v52_source_inventory.json')
    p.add_argument('--gcs',action='store_true')
    args=p.parse_args()
    from google.cloud import bigquery,storage
    inventory=discover(bigquery.Client(project=args.dataset.split('.')[0]),dataset=args.dataset)
    if args.gcs:
        inventory['gcs']=discover_gcs(storage.Client(project=args.dataset.split('.')[0]))
    Path(args.output).write_text(json.dumps(inventory,indent=2))
    print('Candidate tables:',len(inventory['candidate_tables']))
    for r in inventory['candidate_tables'][:20]:
        print(f" {r['table']} score={r['score']} family_fields={r['family_column_counts']} proof_cols={r['has_all_archive_proof_fields']}")
    print('Saved inventory:',args.output)

if __name__=='__main__':main()
