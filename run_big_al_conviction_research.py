#!/usr/bin/env python3
"""Read-only one-shot Big Al pregame feature / conviction study; NO Heavy refit.

Default BigQuery path reads CFBD prior fields for just 2026 observed card dates.
Outputs locally only, with no 2026 outcomes, score lookup, or GCS mutation.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

from big_al_conviction_differential_v1 import PRIOR_FIELDS,study

SELECT_COLS=('Season','Game_ID','Game_Date','Team','Opponent','Is_Home')+PRIOR_FIELDS


def load_ledger(path):
    data=[json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not data: raise ValueError('EMPTY_BIG_AL_LEDGER')
    return data


def load_side_rows_bq(project,dataset,table,game_dates):
    from google.cloud import bigquery
    if not all(str(x).startswith('2026-') for x in game_dates):
        raise ValueError('THIS_ONE_SHOT_IS_RESTRICTED_TO_2026_BIG_AL_CARD_DATES')
    safe=lambda x:x.replace('_','').isalnum() and x[0].isalpha()
    if not all(safe(x) for x in (project,dataset,table)):
        # GCP project IDs may contain dashes; allow project dashes explicitly.
        if not (project.replace('-','').isalnum() and safe(dataset) and safe(table)):
            raise ValueError('INVALID_BIGQUERY_IDENTIFIER')
    client=bigquery.Client(project=project)
    columns=', '.join('`'+x+'`' for x in SELECT_COLS)
    # Two-sided rows from the warehouse; no final scores, no today stats.
    query=f'''SELECT {columns}
       FROM `{project}.{dataset}.{table}`
       WHERE Season=2026 AND DATE(Game_Date) BETWEEN DATE_SUB(@first_card, INTERVAL 1 DAY)
           AND DATE_ADD(@last_card, INTERVAL 1 DAY)'''
    config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter('first_card','DATE',min(game_dates)),
        bigquery.ScalarQueryParameter('last_card','DATE',max(game_dates))],
        maximum_bytes_billed=10*1024**3,
        use_query_cache=True)
    job=client.query(query,job_config=config)
    return [dict(row.items()) for row in job.result()]


def run(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ledger',default='big_al_observations.jsonl')
    p.add_argument('--project',default='sharplogger')
    p.add_argument('--dataset',default='sharp_data')
    p.add_argument('--table',default='ncaaf_cfbd_miner_features_v1')
    p.add_argument('--side-rows-json',help='Local two-sided CFBD fixture/snapshot instead of BigQuery')
    p.add_argument('--shadow-evidence-json',help='OPTIONAL immutable, pregame, independently validated support by observation_id')
    p.add_argument('--out-dir',default='bigal_conviction_research')
    args=p.parse_args(argv)
    obs=load_ledger(args.ledger)
    dates=sorted({str(x.get('game_date')) for x in obs if x.get('sport')=='NCAAF' and x.get('game_date')})
    if args.side_rows_json:
        side_rows=json.loads(Path(args.side_rows_json).read_text())
    else:
        side_rows=load_side_rows_bq(args.project,args.dataset,args.table,dates)
    evidence=json.loads(Path(args.shadow_evidence_json).read_text()) if args.shadow_evidence_json else None
    result=study(obs,side_rows,shadow_evidence=evidence)
    result['source_meta']={'origin':'USER_PROVIDED_CFBD_SNAPSHOT' if args.side_rows_json else 'BQ_2026_PRIOR_FEATURE_ONLY',
                           'project':args.project,'dataset':args.dataset,'table':args.table,
                           'observed_card_dates':dates,'side_rows_read':len(side_rows)}
    root=Path(args.out_dir);root.mkdir(parents=True,exist_ok=True)
    (root/'bigal_conviction_fingerprints.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    cols=['observation_id','game_date','rating','game_id','home','away','selected_side','identity_method',
          'spread_role','observed_line','selected_vs_opp_season_net_ppa','selected_vs_opp_elo',
          'selected_vs_opp_off_success','selected_off_ppa_vs_opp_def_ppa','selected_recent3_minus_season_net_ppa',
          'under_total_59_plus','feature_vintage_certified']
    with (root/'bigal_matched_fingerprints.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for row in result['fingerprints']:
            fs=row['features'];w.writerow({**{k:row.get(k) for k in cols},
                **{k:fs.get(k) for k in fs.keys() & set(cols)},
                'feature_vintage_certified':False})
    with (root/'bigal_conviction_contrasts.csv').open('w',newline='') as f:
        cols2=['higher_observation_id','higher_rating','lower_observation_id','lower_rating',
               'market','price_distance','same_card_date','prior_feature_deltas_high_minus_low','interpretation']
        w=csv.DictWriter(f,fieldnames=cols2);w.writeheader()
        for c in result['paired_conviction_contrasts']:
            w.writerow({**c,'prior_feature_deltas_high_minus_low':json.dumps(c['prior_feature_deltas_high_minus_low'])})
    (root/'shadow_rule_audit.json').write_text(json.dumps({
        'rule':'BIG_AL_HIGH_CONVICTION_CONFIRMATION','counts':result['shadow_status_counts'],
        'rows':result['shadow_rule_evaluations'],'bet_authority':0,
        'original_release_vintage_certified':False},indent=2)+'\n')
    print('[BIGAL-CONVICTION-RESEARCH] '+json.dumps({
        'status':result['status'],'observations':result['ncaaf_observations'],
        'games':result['cfbd_games_in_snapshot'], 'matched':result['matched_fingerprints'],
        'unmatched':len(result['unmatched_observations']),
        'comparisons':result['contrasts_computed'], 'shadow':result['shadow_status_counts'],
        'bet_authority':0,'out_dir':str(root)},sort_keys=True))
    return result

if __name__=='__main__': run()
