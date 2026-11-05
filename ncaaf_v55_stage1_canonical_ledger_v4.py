#!/usr/bin/env python3
"""Stage 1 V4: Adopt verified CFBD identity into historical result/market ledgers.

Does NOT replay brains, validate original pregame capture, train Miner/CORE,
change official GCS pointers or tables, grant betting authority, or infer side
identity from final scores/odds. Consumes local V3 crosswalk and Stage1 CSVs.
Read-only BigQuery call uses the historical source's ORIGINAL naming columns.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path

from ncaaf_v55_stage1_crosswalk_v3 import read_bq_identity, school, season, readcsv, writecsv

CFBD = 'cfbd_canonical_game_sides.csv'
SIDE = 'historical_game_side_results.csv'
PROVIDER = 'historical_provider_quotes.csv'
SNAPSHOT = 'historical_snapshot_quotes.csv'
SOURCE_AUDIT = 'stage1_source_separation_audit.json'
CROSSWALK = 'historical_source_cfbd_crosswalk_v3.csv'
CROSSWALK_AUDIT = 'stage1_crosswalk_v3_audit.json'
NEEDED = (CFBD, SIDE, PROVIDER, SNAPSHOT, SOURCE_AUDIT)


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_verified_inputs(srcdir, xdir):
    srcdir, xdir = Path(srcdir), Path(xdir)
    for n in NEEDED:
        if not (srcdir / n).is_file():
            raise RuntimeError('MISSING_STAGE1_SOURCE:' + n)
    for n in (CROSSWALK, CROSSWALK_AUDIT):
        if not (xdir / n).is_file():
            raise RuntimeError('MISSING_V3_CROSSWALK:' + n)
    xa = json.loads((xdir / CROSSWALK_AUDIT).read_text())
    sa = json.loads((srcdir / SOURCE_AUDIT).read_text())
    if xa.get('status') != 'STAGE1_CROSSWALK_V3_RESEARCH_ONLY':
        raise RuntimeError('UNEXPECTED_CROSSWALK_AUDIT_STATUS')
    if xa.get('output_sha256') != file_hash(xdir / CROSSWALK):
        raise RuntimeError('CROSSWALK_CHECKSUM_MISMATCH')
    if xa.get('source_manifest_sha256') != file_hash(srcdir / SOURCE_AUDIT):
        raise RuntimeError('MISMATCHED_STAGE1_SOURCE_EXPORT')
    if xa.get('bet_authority') != 0 or xa.get('original_capture_certified') is not False:
        raise RuntimeError('INPUT_CROSSWALK_UNEXPECTED_AUTHORITY')
    for k in ('bet_authority', 'source_capture_certified'):
        if sa.get(k) not in (0, False):
            raise RuntimeError('SOURCE_AUDIT_UNEXPECTED_AUTHORITY:' + k)
    rows = readcsv(xdir / CROSSWALK)
    accepted = {}
    by_cfbd = {}
    holds = Counter()
    seen_source_keys = set()
    for r in rows:
        y = season(r.get('season'))
        sid = str(r.get('source_game_id', '')).strip()
        gid = str(r.get('cfbd_game_id', '')).strip()
        status = r.get('status', '')
        if not y or not sid:
            raise RuntimeError('INVALID_CROSSWALK_SOURCE_ID')
        key = (y, sid)
        if key in seen_source_keys:
            raise RuntimeError('DUPLICATE_SOURCE_ID:'+str(key))
        seen_source_keys.add(key)
        if status == 'MATCH_UNIQUE_WAREHOUSE_NAME_VARIANT' and gid:
            if (y, gid) in by_cfbd:
                raise RuntimeError('DUPLICATE_CANONICAL_CLAIM:'+str((y, gid)))
            accepted[key] = r
            by_cfbd[(y, gid)] = key
        else:
            if gid:
                raise RuntimeError('UNEXPECTED_HELD_CFBD_ID:'+str(key))
            holds[status or 'HOLD_UNKNOWN'] += 1
    if len(accepted) != int(xa['matched_games']):
        raise RuntimeError('CROSSWALK_COUNT_AUDIT_MISMATCH')
    if len(rows) != int(xa['source_games_evaluated']):
        raise RuntimeError('SOURCE_COUNT_AUDIT_MISMATCH')
    return accepted, by_cfbd, xa, sa, holds


def canonical_two_sided(rows):
    groups = defaultdict(list)
    for r in rows:
        y = season(r.get('season'))
        gid = str(r.get('cfbd_game_id', '')).strip()
        if y and gid:
            groups[(y, gid)].append(r)
    valid, bad = {}, Counter()
    for k, sides in groups.items():
        if len(sides) != 2:
            bad['CFBD_NOT_EXACTLY_TWO_SIDES'] += 1
            continue
        a,b=sides
        if (a.get('team_key') != b.get('opponent_key') or
            b.get('team_key') != a.get('opponent_key') or
            a.get('game_date') != b.get('game_date') or
            not a.get('team_key') or not b.get('team_key')):
            bad['CFBD_NONRECIPROCAL_SIDES'] += 1
            continue
        valid[k] = {school(a['team_key']):a, school(b['team_key']):b}
        if len(valid[k]) != 2:
            del valid[k]; bad['CFBD_AMBIGUOUS_SCHOOLS'] += 1
    return valid,bad


def source_sides_from_warehouse(rows):
    groups=defaultdict(list)
    for r in rows:
        y=season(r.get('Season'))
        sid=str(r.get('Source_Game_ID') or '').strip()
        if y and sid:
            groups[(y,sid)].append(r)
    return groups


def source_result_name_matches(result, row):
    expected = school(result.get('team_key'))
    variants = ('Team','Team_Norm','Canonical_Team_Norm','Canonical_Team_Name')
    return bool(expected and expected in {school(row.get(c)) for c in variants})


def align_game_sides(src_rows, cfbd_sides, source_id, source_results):
    """Strict 2-sided alignment via actual native source-name fields, not scores.

    Additional historical source aliases are not guessed. If the original name
    or side mapping is missing/ambiguous, the WHOLE game is held.
    """
    if len(src_rows) != 2 or len(source_results) != 2 or len(cfbd_sides) != 2:
        return None, 'HOLD_SOURCE_SIDE_COUNT'
    if len({x.get('team_key') for x in source_results}) != 2:
        return None, 'HOLD_DUPLICATE_RESULT_SIDE'
    used_warehouse = set(); used_cfbd = set(); aligned = []
    for r in source_results:
        matched = [i for i,s in enumerate(src_rows) if source_result_name_matches(r,s)]
        if len(matched) != 1 or matched[0] in used_warehouse:
            return None, 'HOLD_RESULT_TO_WAREHOUSE_SIDE'
        n = matched[0];used_warehouse.add(n); original=src_rows[n]
        a,b=school(original.get('Source_Team_Name')),school(original.get('Source_Opponent_Name'))
        if not a or not b or a==b:
            return None,'HOLD_ORIGINAL_SOURCE_NAMES_MISSING'
        candidates=[c for c in cfbd_sides.values() if school(c.get('team_key'))==a and school(c.get('opponent_key'))==b]
        if len(candidates)!=1:
            return None, 'HOLD_SOURCE_TO_CFBD_DIRECTION'
        c=candidates[0];ck=c['team_key']
        if ck in used_cfbd:
            return None,'HOLD_DUPLICATE_CFBD_SIDE'
        used_cfbd.add(ck)
        if r.get('cfbd_game_id') and str(r['cfbd_game_id']).strip()!=str(c['cfbd_game_id']).strip():
            # Initial source export has an all-zero crosswalk; nonzero must agree.
            return None,'HOLD_ORIGINAL_GAME_ID_CONFLICT'
        if r.get('is_home') in ('0','1') and c.get('is_home') in ('0','1') and r['is_home']!=c['is_home']:
            return None,'HOLD_HOME_AWAY_FLAG_CONFLICT'
        aligned.append((r,c))
    x,y=aligned
    if x[1]['team_key'] != y[1]['opponent_key'] or y[1]['team_key'] != x[1]['opponent_key']:
        return None,'HOLD_NONRECIPROCAL_ALIGNED_SIDES'
    # Validate result symmetry where both sides have recorded scores.
    for col in ('team_score','opponent_score'):
        if (x[0].get(col) in ('',None)) != (y[0].get(col) in ('',None)):
            return None,'HOLD_PARTIAL_GAME_SCORE'
    if x[0].get('team_score') not in ('',None):
        if (x[0]['team_score']!=y[0]['opponent_score'] or
            x[0]['opponent_score']!=y[0]['team_score']):
            return None,'HOLD_CONFLICTING_GAME_SCORES'
    return aligned,None


def walkforward_split(y):
    if y<=2023:return 'DISCOVERY_2013_2023'
    if y==2024:return 'CONFIRMATION_2024'
    if y==2025:return 'FINAL_HOLDOUT_2025'
    raise ValueError('OUTSIDE_HISTORY')


def build_ledger(accepted, canonical, side_results, warehouse_sides, providers, snapshots):
    valid,bad=canonical_two_sided(canonical)
    results_by=defaultdict(list)
    for r in side_results:
        y=season(r.get('season'));sid=str(r.get('source_game_id','')).strip()
        if y and sid:results_by[(y,sid)].append(r)
    source_by=source_sides_from_warehouse(warehouse_sides)
    quote_stats=defaultdict(Counter)
    for row in providers:
        y=season(row.get('season'));gid=str(row.get('cfbd_game_id','')).strip()
        if y and gid and row.get('game_join')=='EXACT_CFBD_ID_AND_TEAMS':
            quote_stats[(y,gid)]['provider_rows_untimed']+=1
    for row in snapshots:
        y=season(row.get('season'));gid=str(row.get('cfbd_game_id','')).strip()
        if y and gid and row.get('game_join')=='EXACT_DATE_HOME_AWAY':
            quote_stats[(y,gid)]['snapshot_rows_time_observed']+=1
            if row.get('quote_chronology')=='PREKICK_TIME_SOURCE_CAPTURE_UNATTESTED':
                quote_stats[(y,gid)]['snapshot_rows_prekick_unattested']+=1
    output=[];manifest=[];reasons=Counter();by_season=defaultdict(Counter)
    for (year,sid),xref in sorted(accepted.items()):
        gid=str(xref['cfbd_game_id']).strip(); k=(year,gid)
        if k not in valid:
            reasons['HOLD_NO_VALID_CFBD_TWO_SIDED']+=1;by_season[year]['held_alignment']+=1;continue
        aligned,reason=align_game_sides(source_by.get((year,sid),[]),valid[k],(year,sid),results_by.get((year,sid),[]))
        if reason:
            reasons[reason]+=1;by_season[year]['held_alignment']+=1;continue
        for r,c in aligned:
            output.append({
                'season':year,'cfbd_game_id':gid,'source_game_id':sid,
                'game_date':c['game_date'],'source_game_date':r.get('game_date',''),
                'cfbd_team_key':c['team_key'],'cfbd_opponent_key':c['opponent_key'],
                'source_team_label':r['team_key'],'source_opponent_label':r['opponent_key'],
                'is_home':c.get('is_home',''),'team_score':r.get('team_score',''),
                'opponent_score':r.get('opponent_score',''),
                'retrospective_open_spread':r.get('historical_context_open_spread',''),
                'retrospective_open_spread_result':r.get('retrospective_open_spread_result',''),
                'retrospective_price_vintage':'UNTIMED_HISTORICAL_OPEN_RESEARCH_ONLY',
                'split':walkforward_split(year),'original_signal_capture_certified':'false',
                'original_price_capture_certified':'false','bet_authority':0})
        q=quote_stats.get(k,Counter())
        manifest.append({
            'season':year,'cfbd_game_id':gid,'source_game_id':sid,
            'game_date':valid[k][next(iter(valid[k]))]['game_date'],
            'split':walkforward_split(year),'source_game_match':'VERIFIED_TWO_SIDED_V3',
            'canonical_side_alignment':'VERIFIED_TWO_SIDED_V4',
            'number_of_sides':len(aligned),
            'provider_quote_rows_untimed':q['provider_rows_untimed'],
            'snapshot_rows_timestamped_unattested':q['snapshot_rows_time_observed'],
            'snapshot_rows_prekick_unattested':q['snapshot_rows_prekick_unattested'],
            'source_signals_original_asof_verified':'false',
            'historical_prices_original_capture_verified':'false',
            'bet_authority':0})
        by_season[year]['aligned_games']+=1
        by_season[year]['provider_quote_rows_untimed']+=q['provider_rows_untimed']
        by_season[year]['snapshot_rows_prekick_unattested']+=q['snapshot_rows_prekick_unattested']
    return manifest,output,{'held_alignment_reasons':dict(reasons),
        'cfbd_invalid_games':dict(bad),'by_season':{str(k):dict(v) for k,v in sorted(by_season.items())}}


def run(source_dir,crosswalk_dir,out_dir,client):
    source_dir,crosswalk_dir,out_dir=map(Path,(source_dir,crosswalk_dir,out_dir))
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError('REFUSE_TO_OVERWRITE_NONEMPTY_OUTPUT_DIRECTORY')
    accepted,by_cfbd,xa,sa,holds=load_verified_inputs(source_dir,crosswalk_dir)
    warehouse,manifest=read_bq_identity(client,'ncaaf_historical_game_side_context')
    if len(warehouse)==0:raise RuntimeError('WAREHOUSE_SOURCE_IDENTITIES_EMPTY')
    can=readcsv(source_dir/CFBD); sides=readcsv(source_dir/SIDE)
    providers=readcsv(source_dir/PROVIDER); snapshots=readcsv(source_dir/SNAPSHOT)
    game_ledger,side_ledger,detail=build_ledger(accepted,can,sides,warehouse,providers,snapshots)
    # No silent approval of weak/empty side joins.
    if not game_ledger:raise RuntimeError('ZERO_ALIGNED_GAMES_NO_PUBLICATION')
    report={
        'status':'STAGE1_CANONICAL_RESULTS_AND_MARKET_LEDGER_RESEARCH_ONLY',
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'historical_range':[2013,2025],
        'discovery_seasons':[2013,2023], 'confirmation_season':2024,'final_holdout_season':2025,
        'source_games':int(xa['source_games_evaluated']),
        'v3_matched_games':len(accepted),
        'v3_held_games':sum(holds.values()),
        'v3_hold_reasons':dict(holds),
        'canonical_aligned_games':len(game_ledger),
        'canonical_aligned_sides':len(side_ledger),
        'held_after_v3_alignment':len(accepted)-len(game_ledger),
        'quote_rows_from_original_exports':{'provider_rows':len(providers),'snapshot_rows':len(snapshots)},
        'warehouse_identity_manifest':manifest,
        'alignment':detail,
        'source_sha256':{n:file_hash(source_dir/n) for n in NEEDED},
        'crosswalk_sha256':file_hash(crosswalk_dir/CROSSWALK),
        'crosswalk_audit_sha256':file_hash(crosswalk_dir/CROSSWALK_AUDIT),
        'original_brain_signal_capture_certified':False,
        'original_book_quote_capture_certified':False,
        'combination_out_of_sample_certified':False,
        'frozen_miner_345_rule_library_modified':False,
        'all_existing_brains_modified':False,
        'production_tables_modified':False,
        'bet_authority':0,
        'next':'JOIN_EXISTING_FROZEN_BRAIN_EVENT_RECORDS_WITH_PROVENANCE_AND_WALKFORWARD; VERIFY_SOURCE_QUOTE_CAPTURE'
    }
    out_dir.mkdir(parents=True,exist_ok=True)
    mcols=['season','cfbd_game_id','source_game_id','game_date','split','source_game_match',
      'canonical_side_alignment','number_of_sides','provider_quote_rows_untimed',
      'snapshot_rows_timestamped_unattested','snapshot_rows_prekick_unattested',
      'source_signals_original_asof_verified','historical_prices_original_capture_verified','bet_authority']
    scols=['season','cfbd_game_id','source_game_id','game_date','source_game_date',
      'cfbd_team_key','cfbd_opponent_key','source_team_label','source_opponent_label',
      'is_home','team_score','opponent_score','retrospective_open_spread',
      'retrospective_open_spread_result','retrospective_price_vintage','split',
      'original_signal_capture_certified','original_price_capture_certified','bet_authority']
    writecsv(out_dir/'historical_cfbd_game_manifest_v4.csv',game_ledger,mcols)
    writecsv(out_dir/'historical_cfbd_game_sides_v4.csv',side_ledger,scols)
    report['outputs_sha256']={p.name:file_hash(p) for p in sorted(out_dir.glob('*.csv'))}
    (out_dir/'stage1_canonical_ledger_v4_audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('STAGE1_V4_AUDIT='+str(out_dir/'stage1_canonical_ledger_v4_audit.json'))
    print('V3_MATCHED_GAMES='+str(len(accepted)))
    print('CANONICAL_ALIGNED_GAMES='+str(len(game_ledger)))
    print('CANONICAL_ALIGNED_SIDES='+str(len(side_ledger)))
    print('HELD_ALIGNMENT='+str(len(accepted)-len(game_ledger)))
    print('BY_YEAR='+json.dumps(detail['by_season'],sort_keys=True))
    print('RESEARCH_ONLY; CERTIFIED_COMBINATIONS=0; BET_AUTHORITY=0; FROZEN_BRAINS_UNMODIFIED')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--crosswalk-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args(argv)
    from google.cloud import bigquery
    return run(args.source_dir,args.crosswalk_dir,args.out_dir,bigquery.Client(project='sharplogger'))

if __name__=='__main__':main()
