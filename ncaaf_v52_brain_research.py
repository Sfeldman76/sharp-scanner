"""NCAAF V5.2 Cloud Run source inventory + fail-closed brain attribution.

Reuse immutable V5.1 edge ledger. No V4 retrain and no Production V1 writes.
The BQ catalog step is automatic. Reading historical brain records requires an
explicit, audited source map; no guesses about team orientation or rule vintage.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from io import StringIO
import json
import os
from pathlib import Path

import pandas as pd
from ncaaf_v52_brain_bridge import (normalize_records,verify_gcs_originals,attribute,
   fingerprint,FAMILIES)
from ncaaf_v52_source_inventory import discover,discover_gcs,extract_configured_source

VERSION='NCAAF_V52_SOURCE_DISCOVERY_AND_BRAIN_ATTRIBUTION_20261020_V1'
BUCKET='sharp-models'
INPUT_PREFIX='research/ncaaf/v52_inputs/'
OUTPUT_PREFIX='research/ncaaf/v52_brain_attribution/'
PARENT='research/ncaaf/v51_edge_attribution/latest_report.json'


def load_source_map(contents):
    raw=json.loads(contents) if isinstance(contents,str) else contents
    if not isinstance(raw,dict) or not isinstance(raw.get('sources',[]),list):
        raise ValueError('V52_INVALID_SOURCE_MAP_STRUCTURE')
    for s in raw.get('sources',[]):
        if not isinstance(s,dict) or not s.get('table') or not s.get('mapping'):
            raise ValueError('V52_MALFORMED_SOURCE_SPEC')
        if s.get('format','long')=='long' and 'side' in s.get('mapping',{}) and not ({'team','opponent'}.issubset(s['mapping']) or s.get('side_orientation')=='V4_CANONICAL'):
            raise ValueError('V52_UNVERIFIED_SIDE_REFERENCE')
    return raw


def analyze(edges, *, config=None, source_frames=None, storage_client=None, inventory=None, artifact_list=None):
    """Inject source_frames in tests or perform configured BigQuery reads in Cloud Run."""
    config=config or {'sources':[]}
    audit=[];frames=[];proof=[]
    source_frames=source_frames or []
    if source_frames and len(source_frames)!=len(config['sources']):
        raise ValueError('V52_SOURCE_FRAME_COUNT_MISMATCH')
    for i,spec in enumerate(config['sources']):
        if not source_frames:
            raise ValueError('V52_CLOUD_READS_REQUIRED_FOR_CONFIGURED_SOURCES')
        raw=source_frames[i]
        candidates,a=normalize_records(raw,spec,edges)
        audit.append(a)
        if not candidates.empty:frames.append(candidates)
    if frames:
        brain=pd.concat(frames,ignore_index=True)
        # Duplicate source rows from different tables are not counted as
        # independently confirmed systems. Conflicting sides -> quarantine.
        bad=brain.groupby(['Season','Game_ID','Family','System_ID']).Side.nunique()
        bad_keys=bad[bad>1].index
        if len(bad_keys):
            idx=pd.MultiIndex.from_frame(brain[['Season','Game_ID','Family','System_ID']]).isin(bad_keys)
            brain=brain[~idx].copy()
        brain=brain.sort_values(['Season','Game_ID','Family','System_ID','snapshot_utc'],na_position='last')
        brain=brain.drop_duplicates(['Season','Game_ID','Family','System_ID'])
        if storage_client is not None:
            brain,proof=verify_gcs_originals(brain,storage_client)
    else:
        brain=pd.DataFrame(columns=['Season','Game_ID','Family','System_ID','Side','archive_verified','preseason_declared','time_claim_valid'])
    report,ledger=attribute(edges,brain)
    report.update({'version':VERSION,'source_inventory':inventory or {},
        'gcs_inventory':artifact_list or {},'configured_sources':len(config['sources']),
        'source_extraction_audits':audit,'archive_verification':proof,
        'status':('SOURCE_DISCOVERY_REQUIRED__NO_CONFIGURED_BRAIN_SOURCES' if not config['sources'] else report['status']),
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'source_vintage_certified':False,
        'caution':'Only individually verified original archived pregame artifacts can support historical brain attribution; zero verified does not mean zero actual systems.',
        'warnings':[
           'No historical executable bookmaker prices: no ROI claims',
           'No bet decisions, no grading of retrospective discovered rules as predeclared',
           'Vintage certification for V4 CFBD features remains unresolved',
           'Retrospective system features are research context, not historical bet-confirmation votes',
           'All 2024–2025 results previously inspected; no pristine retrospective holdout'
        ]})
    return report,ledger,brain


def cloud_run(*,storage_client,bucket_name=BUCKET,log_func=print):
    from google.cloud import bigquery
    bucket=storage_client.bucket(bucket_name)
    parent=json.loads(bucket.blob(PARENT).download_as_text())
    uri=parent.get('dedup_ledger_uri','')
    if not (parent.get('version','').startswith('NCAAF_V51_') and
        isinstance(uri,str) and uri.startswith(f'gs://{bucket_name}/research/ncaaf/v51_edge_attribution/runs/') and
        uri.endswith('/unique_model_game_edges.csv') and parent.get('v4_prediction_verified_with_source') is True):
        raise RuntimeError('V52_PARENT_NOT_INDEPENDENTLY_VERIFIED_V51_LEDGER')
    edges=pd.read_csv(StringIO(bucket.blob(uri.replace(f'gs://{bucket_name}/','')).download_as_text()),low_memory=False)
    edges['Season']=pd.to_numeric(edges.Season,errors='raise').astype(int)
    edges['Game_ID']=edges.Game_ID.astype(str).str.strip()
    bq=bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT','sharplogger'))
    dataset=os.getenv('NCAAF_V52_BQ_DATASET','sharplogger.sharp_data')
    try:
        inventory=discover(bq,dataset=dataset)
    except Exception as ex:
        inventory={'status':'BQ_SOURCE_DISCOVERY_FAILED','error':f'{type(ex).__name__}: {str(ex)[:250]}',
                   'candidate_tables':[]}
    try:
        gcs_inventory=discover_gcs(storage_client,bucket=bucket_name)
    except Exception as ex:
        gcs_inventory={'status':'GCS_DISCOVERY_FAILED','error':str(ex)[:250]}
    key=os.getenv('NCAAF_V52_SOURCE_MAP_OBJECT','').strip()
    config={'sources':[]};extraction=[];sources=[]
    if key:
        if not key.startswith(INPUT_PREFIX) or not key.endswith('.json'):
            raise RuntimeError('V52_INVALID_SOURCE_MAP_GCS_PATH')
        config=load_source_map(bucket.blob(key).download_as_text())
        for s in config['sources']:
            frame,status=extract_configured_source(bq,s,dataset=dataset,
                max_rows=int(s.get('max_rows',150000)))
            extraction.append(status);sources.append(frame)
    report,ledger,brain=analyze(edges,config=config,source_frames=sources,
        storage_client=storage_client,inventory=inventory,artifact_list=gcs_inventory)
    report['source_table_query_audit']=extraction
    if any(x.get('truncated') for x in extraction):
        report['status']='BLOCKED_INCOMPLETE_CAPPED_SOURCE'
    report['v51_parent_uri']=uri
    report['v51_parent_sha256']=fingerprint(edges)
    report['source_map_object']=key or None
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root=f'{OUTPUT_PREFIX}runs/{run}'
    report['report_uri']=f'gs://{bucket_name}/{root}/report.json'
    report['source_inventory_uri']=f'gs://{bucket_name}/{root}/source_inventory.json'
    report['candidate_ledger_uri']=f'gs://{bucket_name}/{root}/brain_candidate_ledger.csv'
    report['attributed_ledger_uri']=f'gs://{bucket_name}/{root}/research_model_edges.csv'
    artifacts={
       f'{root}/report.json':(json.dumps(report,indent=2,allow_nan=False),'application/json'),
       f'{root}/source_inventory.json':(json.dumps({'bq':inventory,'gcs':gcs_inventory},indent=2),'application/json'),
       f'{root}/brain_candidate_ledger.csv':(brain.to_csv(index=False),'text/csv'),
       f'{root}/research_model_edges.csv':(ledger.to_csv(index=False),'text/csv'),
    }
    for path,(payload,ctype) in artifacts.items():
        bucket.blob(path).upload_from_string(payload,content_type=ctype,if_generation_match=0)
    bucket.blob(f'{OUTPUT_PREFIX}latest_report.json').upload_from_string(json.dumps(report,indent=2,allow_nan=False),content_type='application/json')
    top=inventory.get('candidate_tables',[])[:8]
    for candidate in top:
        log_func(f"[NCAAF-V52-SOURCE] table={candidate['table']} rank={candidate['score']} family_fields={candidate['family_column_counts']} claimed_archive_proof_fields={candidate['has_all_archive_proof_fields']}")
    log_func(f"[NCAAF-V52] status={report['status']} games={report['games']} sources={report['configured_sources']} candidates={report['brain_candidates']} verified={report['verified_system_game_records']} authority=0 inventory={report['source_inventory_uri']} report={report['report_uri']}")
    return report


def cli():
    p=argparse.ArgumentParser(description='V5.2 offline research using true V5.1 edges')
    p.add_argument('--edges',required=True,help='V5.1 unique_model_game_edges.csv')
    p.add_argument('--source-map',help='JSON with exact source mappings and CSV file paths')
    p.add_argument('--output-dir',required=True)
    a=p.parse_args()
    edges=pd.read_csv(a.edges)
    edges['Game_ID']=edges.Game_ID.astype(str).str.strip()
    edges['Season']=pd.to_numeric(edges.Season,errors='raise').astype(int)
    config={'sources':[]};frames=[]
    if a.source_map:
        config=load_source_map(Path(a.source_map).read_text())
        for source in config['sources']:
            if not source.get('local_csv'):
                raise ValueError('V52_CLI_SOURCE_MISSING_LOCAL_CSV')
            frames.append(pd.read_csv(source['local_csv'],low_memory=False))
    report,ledger,brain=analyze(edges,config=config,source_frames=frames)
    target=Path(a.output_dir);target.mkdir(parents=True,exist_ok=True)
    (target/'latest_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    ledger.to_csv(target/'research_model_edges.csv',index=False)
    brain.to_csv(target/'brain_candidate_ledger.csv',index=False)
    print(f"[NCAAF-V52] {report['status']} candidates={report['brain_candidates']} archive_verified={report['verified_system_game_records']} bets=0")

if __name__=='__main__':cli()
