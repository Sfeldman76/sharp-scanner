#!/usr/bin/env python3
"""Read-only: compare CFBD warehouse field population with research Miner frame.

Run after a Heavy Research execution has written feature_lineage_latest.json.
The BigQuery table may contain team-sides, while the research frame contains
physical games, so compare COVERAGE FRACTIONS, never absolute row counts.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path

DEFAULT_TABLE = 'sharplogger.sharp_data.ncaaf_cfbd_miner_features_v1'
DEFAULT_ARTIFACT = 'gs://sharp-models/research/ncaaf/v2/feature_lineage_latest.json'


def parse_table(s):
    parts=s.replace('`','').split('.')
    if len(parts)!=3 or not all(re.fullmatch(r'[A-Za-z0-9_\-]+',p) for p in parts):
        raise ValueError('Expected safe project.dataset.table name')
    return parts


def read_lineage(uri):
    if uri.startswith('gs://'):
        from google.cloud import storage
        bucket,path=uri[5:].split('/',1)
        text=storage.Client().bucket(bucket).blob(path).download_as_text()
    else:
        text=Path(uri).read_text(encoding='utf-8')
    return json.loads(text)


def audit_query(table, season_col, fields):
    """Single read-only aggregation for a small field batch."""
    if len(fields)>40:
        raise ValueError('batch cannot exceed 40 fields')
    safe_fields=[]
    for col in [season_col]+list(fields):
        if not re.fullmatch(r'[A-Za-z0-9_]+',col):
            raise ValueError('Unsafe BigQuery column identifier')
    for c in fields:
        safe_fields.append(f'COUNTIF(SAFE_CAST(`{c}` AS FLOAT64) IS NOT NULL) AS `{c}`')
    return (f'SELECT CAST(SAFE_CAST(`{season_col}` AS INT64) AS INT64) AS season, COUNT(*) AS source_rows, '
            +', '.join(safe_fields)+f' FROM `{table}` '
            +f'WHERE SAFE_CAST(`{season_col}` AS INT64) BETWEEN 2013 AND 2025 '
            +f'GROUP BY season ORDER BY season')


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--table',default=DEFAULT_TABLE)
    pa.add_argument('--source-only',action='store_true',help='Inspect BigQuery alone now; no new Heavy run needed')
    pa.add_argument('--lineage',default=DEFAULT_ARTIFACT,
                    help='GCS URI or local JSON file from the feature audit')
    pa.add_argument('--output',default='ncaaf_research_field_join_gaps.csv')
    pa.add_argument('--source-population-threshold',type=float,default=.5)
    pa.add_argument('--miner-population-threshold',type=float,default=.10)
    args=pa.parse_args()
    project,dataset,table=parse_table(args.table)
    from google.cloud import bigquery
    cli=bigquery.Client(project=project)
    schema=list(cli.query(f'SELECT column_name,data_type FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS` '
                       'WHERE table_name=@table ORDER BY ordinal_position',
          job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter('table','STRING',table)])).result())
    kinds={r['column_name']:r['data_type'] for r in schema}
    scol=next((c for c in ('Season','season','Season_Year','season_year') if c in kinds),None)
    if not scol: raise SystemExit('No season column found in BigQuery source')
    fields=[c for c,t in kinds.items() if c.startswith('CFBD_') and t in ('INT64','FLOAT64','NUMERIC','BIGNUMERIC','BOOL')]
    if not fields:
        raise SystemExit('No numeric CFBD_ fields found; inspect schema and prefix first')
    print(f'SOURCE={args.table} fields={len(fields)} season_column={scol}')
    source={}
    for i in range(0,len(fields),35):
        batch=fields[i:i+35]
        sql=audit_query(args.table,scol,batch)
        for r in cli.query(sql).result():
            d=dict(r.items())
            yr=int(d['season']); src=source.setdefault(yr,{'source_rows':int(d['source_rows']),'fields':{}})
            src['fields'].update({k:int(d[k] or 0) for k in batch})
    if args.source_only:
        records={}
    else:
        try:
            lineage=read_lineage(args.lineage)
        except Exception as exc:
            raise SystemExit(f'Historical audit artifact unavailable; deploy audit and run Heavy once: {exc}')
        records={x['field']:x for x in lineage.get('fields',[])}
    output=[]
    for yr in sorted(source):
        src=source[yr]; src_n=src['source_rows']
        for field,source_n in sorted(src['fields'].items()):
            rec=records.get(field)
            details=(rec or {}).get('per_year_miner',{}).get(str(yr),{})
            miner_rows=int(details.get('rows',0)); miner_n=int(details.get('non_null',0))
            source_pct=source_n/max(src_n,1)
            miner_pct=miner_n/max(miner_rows,1)
            flagged=bool(rec and source_pct>=args.source_population_threshold and miner_rows and miner_pct<args.miner_population_threshold)
            output.append({'season':yr,'field':field,'warehouse_rows':src_n,'warehouse_populated':source_n,
                'warehouse_pct':round(source_pct,4),'miner_rows':miner_rows,'miner_populated':miner_n,
                'miner_pct':round(miner_pct,4),'status':('WAREHOUSE_SOURCE_ONLY' if args.source_only else
                    'SUSPECT_JOIN_OR_LINEAGE_GAP' if flagged else
                    'NO_LINEAGE_FIELD' if not rec else 'CHECK_COVERAGE'),
                'in_stat_candidate_registry':bool((rec or {}).get('in_stat_registry',False))})
    dst=Path(args.output)
    with dst.open('w',encoding='utf-8',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(output[0]))
        w.writeheader();w.writerows(output)
    flagged=[r for r in output if r['status']=='SUSPECT_JOIN_OR_LINEAGE_GAP']
    missing=[r for r in output if r['status']=='NO_LINEAGE_FIELD']
    print(json.dumps({'status':'READ_ONLY_SOURCE_AUDIT' if args.source_only else 'READ_ONLY_JOIN_COMPARISON','field_seasons_checked':len(output),
                      'possible_join_failures':len(flagged),'source_features_not_in_research_audit':len(missing),
                      'output_file':str(dst),'note':'Source team-sides and research game rows have different denominators; inspect flagged joins manually.'}))
    for r in flagged[:30]: print('FLAG',r['season'],r['field'], 'warehouse_pct',r['warehouse_pct'],'miner_pct',r['miner_pct'])

if __name__=='__main__': main()
