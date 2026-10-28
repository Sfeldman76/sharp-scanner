#!/usr/bin/env python3
"""READ-ONLY Cloud Shell audit: exact remaining frozen input atoms.
Run: python3 audit_remaining_miner_atoms.py
Requires google-cloud-bigquery already used by scanner. Never writes BQ/GCS.
"""
import json
from google.cloud import bigquery
from ncaaf_miner_advanced_asof_v1 import RAW_SPECS

TABLE='sharplogger.sharp_data.ncaaf_cfbd_miner_features_v1'
client=bigquery.Client(project='sharplogger')
table=client.get_table(TABLE)
fields={f.name for f in table.schema}
print('[NCAAF-MINER-REMAINING-INPUT-SCHEMA]',json.dumps({
 'table':TABLE,
 'raw_metric_columns':[x for x in sorted(fields) if x.startswith('CFBD_Game_') or x.startswith('CFBD_QB_Game_')],
 'possible_replay_pairs':[
   {'required_pregame_prior':prior,'raw_options':raw,
    'pregame_prior_available':prior in fields,
    'raw_available':[r for r in raw if r in fields],
    'min_prior_games':minimum}
   for prior,raw,minimum in RAW_SPECS],
 'matchup_net_success_available':'CFBD_Matchup_Net_Success_Adv' in fields,
 'side_identity':[n for n in ('Season','Game_Date','Game_ID','Team_Key','Team','Opponent_Key','Opponent') if n in fields],
 'pt_linemidweek_requirement':'NAMED_EXTERNAL_PROVIDER_PREDICTIONS_UNAVAILABLE_NO_REPLACEMENT',
 'stat_edge_requirement':'FROZEN_INDEPENDENT_MODEL_STATE_NOT_VERIFIED',
},indent=2,sort_keys=True))
if {'Season','Game_Date'}.issubset(fields):
    interesting=sorted({c for prior,raw,_ in RAW_SPECS for c in (prior,*raw) if c in fields})
    sums=','.join(f'COUNTIF(SAFE_CAST(`{col}` AS FLOAT64) IS NOT NULL) AS `{col}`' for col in interesting)
    metric_sql=(', '+sums) if sums else ''
    query=f'''SELECT COUNT(*) AS rows_2026,
    COUNTIF(SAFE_CAST(Game_Date AS TIMESTAMP)<CURRENT_TIMESTAMP()) AS pre_now_rows
    {metric_sql} FROM `{TABLE}` WHERE SAFE_CAST(Season AS INT64)=2026'''
    try:
        data=next(iter(client.query(query).result()))
        print('[NCAAF-MINER-REMAINING-INPUT-COVERAGE]',json.dumps(dict(data.items()),sort_keys=True,default=str))
    except Exception as ex:
        print('[NCAAF-MINER-REMAINING-INPUT-COVERAGE] ERROR',type(ex).__name__,str(ex))
