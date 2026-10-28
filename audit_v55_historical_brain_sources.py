"""Read-only historical source-vintage inventory for Big Al / Pathi / PT / STAT.

Run in Cloud Shell: python3 audit_v55_historical_brain_sources.py
This does not create a registry, change BigQuery/GCS, or grant betting votes.
"""
import json
import re
from collections import defaultdict

PROJECT='sharplogger'
DATASET='sharp_data'
BUCKET='sharp-models'
HOLD_KEY='research/ncaaf/v55/qualified_independent_brains_2013_2025.json'
AUDIT_KEY='research/ncaaf/v55/independent_brain_registry_audit_latest.json'
TABLES=('bigal_game_context','bigal_game_context_enriched',
        'ncaaf_historical_game_side_context','ncaaf_historical_game_side_raw',
        'ncaaf_cfbd_archive_research_v1', 'moves_with_features_merged',
        'ncaaf_production_v1_results')
PATTERN=re.compile(r'(^|_)(observed_at|snapshot_timestamp|source_record_id|original_pregame_verified|captured_at|updated_at|created_at|as_of|published_at|ingested_at|bookmaker|opening_spread|open_value|game_start|game_date|season|game_id)($|_)',re.I)


def main():
    try:
        from google.cloud import bigquery, storage
    except ImportError as exc:
        raise SystemExit('Install google-cloud-bigquery and google-cloud-storage in Cloud Shell: '+str(exc))
    result={'status':'READ_ONLY','note':'Schema columns are not historical vintage certification; confirm actual source-level timestamps and identity.',
            'tables':{},'gcs':{},'certification_obstacles':[],'production_authority':0}
    client=bigquery.Client(project=PROJECT)
    sql=f'''SELECT table_name,column_name,data_type
       FROM `{PROJECT}.{DATASET}.INFORMATION_SCHEMA.COLUMNS`
       WHERE table_name IN UNNEST(@names) ORDER BY table_name,ordinal_position'''
    from google.cloud.bigquery import QueryJobConfig, ArrayQueryParameter
    cfg=QueryJobConfig(query_parameters=[ArrayQueryParameter('names','STRING',list(TABLES))])
    for row in client.query(sql,job_config=cfg).result():
        d=result['tables'].setdefault(row.table_name,{'source_columns':[], 'vintage_columns':[], 'flags':0})
        col=row.column_name
        if col.startswith(('BigAl_','Pathi_','NCAAF_PT_','NCAAF_Stat_')): d['flags']+=1
        if PATTERN.search(col):d['source_columns'].append({'field':col,'type':row.data_type})
        if col.endswith(('_Observed_At','_Source_Record_ID','_Original_Pregame_Verified')):
            d['vintage_columns'].append(col)
    store=storage.Client(project=PROJECT)
    bucket=store.bucket(BUCKET)
    for path in (HOLD_KEY,AUDIT_KEY):
        try:
            blob=bucket.blob(path)
            if not blob.exists():
                result['gcs'][path]={'status':'NOT_FOUND'}
                continue
            data=json.loads(blob.download_as_bytes())
            # Do not dump raw evidence with identities in logs. Only approved metadata.
            info={'status':str(data.get('status') or 'UNKNOWN'),
                  'reason':str(data.get('reason') or ''),
                  'systems':len(data.get('systems') or []),
                  'created_utc':str(data.get('created_utc') or '')}
            for k in ('missing_source_proofs','invalid_source_proofs','source_audit','missing_fields','candidate_count','qualified'):
                if k in data:
                    v=data[k]
                    info[k]=list(v)[:20] if isinstance(v,dict) else v if isinstance(v,(str,int,float,bool,list)) else str(type(v))
            result['gcs'][path]=info
        except Exception as exc:
            result['gcs'][path]={'status':'READ_ERROR','type':type(exc).__name__}
    for name in ('bigal_game_context','bigal_game_context_enriched','ncaaf_historical_game_side_context'):
        info=result['tables'].get(name,{})
        vintage=info.get('vintage_columns') or []
        if not vintage:
            result['certification_obstacles'].append(name+': no per-rule original-source-vintage attestation columns in schema')
    print(json.dumps(result,indent=2,sort_keys=True,default=str))
    print('INVENTORY_COMPLETE_READ_ONLY')

if __name__=='__main__':main()
