#!/usr/bin/env python3
"""Read-only Stage 1 V5.2 neutral-site correction to the verified V5 ledger.

Preserves historical CFBD game IDs, game sides, source labels, prices, dates,
frozen brain research, and all original-vintage/bet-authority holds.
Queries BOTH historical source-side Is_Neutral flags for ALL linked games.
Neutral games remain blocked for home/away-sensitive mechanisms and directional
home/away market joins. Outputs the V5-compatible file names for V6.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

MANIFEST = 'historical_cfbd_game_manifest_v5.csv'
SIDES = 'historical_cfbd_game_sides_v5.csv'
AUDIT = 'stage1_canonical_ledger_v5_audit.json'
SOURCE_TABLE = 'sharplogger.sharp_data.ncaaf_historical_game_side_context'
SQL = f"""
SELECT Season AS season, CAST(Source_Game_ID AS STRING) AS source_game_id,
       COUNT(*) AS source_sides,
       COUNTIF(Is_Neutral = 1) AS neutral_sides,
       COUNTIF(Is_Neutral = 0) AS non_neutral_sides,
       COUNTIF(Is_Neutral IS NULL) AS unknown_neutral_sides
FROM `{SOURCE_TABLE}`
WHERE Season BETWEEN 2013 AND 2025
GROUP BY Season, source_game_id
"""
NEUTRAL = 'NEUTRAL_SITE_SOURCE_BOTH_SIDES_CONFIRMED'
NOT_NEUTRAL = 'NON_NEUTRAL_SOURCE_BOTH_SIDES_CONFIRMED'
UNCERTAIN = 'NEUTRAL_STATUS_SOURCE_UNVERIFIED'
OLD_HOLD = 'HOME_AWAY_CONFLICT_UNRESOLVED'


def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def readcsv(path):
    with open(path,encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f)
        if not r.fieldnames:raise RuntimeError('EMPTY_CSV_HEADER:'+str(path))
        return list(r),list(r.fieldnames)


def writecsv(path, rows, fields):
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(rows)


def source_lookup(source_rows):
    """Rows: season, native source ID, two side counts from live warehouse."""
    look={}
    for r in source_rows:
        year=int(r['season']);sid=str(r['source_game_id']).strip()
        if not sid or not 2013<=year<=2025:continue
        k=(year,sid)
        if k in look:raise RuntimeError('SOURCE_NEUTRAL_QUERY_DUPLICATE_KEY:'+str(k))
        look[k]={n:int(r[n]) for n in ('source_sides','neutral_sides','non_neutral_sides','unknown_neutral_sides')}
    return look


def classify(c):
    if not c or c['source_sides']!=2 or (c['neutral_sides']+c['non_neutral_sides']+c['unknown_neutral_sides'])!=2:
        return UNCERTAIN
    if c['neutral_sides']==2:return NEUTRAL
    if c['non_neutral_sides']==2:return NOT_NEUTRAL
    return UNCERTAIN


def reconcile(games, sides, lookup, previous):
    """Never invent an orientation. Use two original Is_Neutral=1 rows to confirm neutral."""
    gmap={};smap=defaultdict(list)
    for g in games:
        k=(int(g['season']),str(g['source_game_id']).strip())
        if k in gmap:raise RuntimeError('DUPLICATE_LEDGER_GAME:'+str(k))
        gmap[k]=g
    for s in sides:
        k=(int(s['season']),str(s['source_game_id']).strip())
        if k not in gmap:raise RuntimeError('ORPHAN_LEDGER_SIDE:'+str(k))
        smap[k].append(s)
    if len(gmap)!=previous['cfbd_games_linked_identity_only'] or len(sides)!=previous['cfbd_sides_linked_identity_only']:
        raise RuntimeError('INPUT_V5_COUNTS_DIFFER_FROM_AUDIT')
    old_holds=sum(g.get('role_verification_status')==OLD_HOLD for g in games)
    if old_holds!=previous['cfbd_games_held_home_away_role']:
        raise RuntimeError('INPUT_V5_HOLD_COUNTS_DIFFER_FROM_AUDIT')
    if any(len(v)!=2 or len({s['cfbd_team_key'] for s in v})!=2 for v in smap.values()):
        raise RuntimeError('INPUT_LEDGER_NOT_TWO_DISTINCT_SIDES')
    revised_games=copy.deepcopy(games);revised_sides=copy.deepcopy(sides)
    rev_map={(int(g['season']),str(g['source_game_id']).strip()):g for g in revised_games}
    rev_sides=defaultdict(list)
    for s in revised_sides:rev_sides[(int(s['season']),str(s['source_game_id']).strip())].append(s)
    category=Counter();byseason=defaultdict(Counter);previous_conflict=Counter()
    for k,g in rev_map.items():
        status=classify(lookup.get(k))
        old=g['role_verification_status']
        if old==OLD_HOLD:previous_conflict[status]+=1
        # Source-neutral confirmation is a site classification, NOT a home/away designation.
        is_neutral=status==NEUTRAL
        is_non_neutral=status==NOT_NEUTRAL
        newrole=(NEUTRAL if is_neutral else old if is_non_neutral else UNCERTAIN)
        verified_home_away=(newrole=='HOME_AWAY_VERIFIED')
        fields={
            'original_v5_role_status':old,
            'role_verification_status':newrole,
            'source_neutral_site_status':status,
            'source_neutral_both_sides_verified':'true' if is_neutral else 'false' if is_non_neutral else '',
            'neutral_site_research_eligible':'true' if is_neutral else 'false',
            'role_sensitive_research_eligible':'true' if verified_home_away else 'false',
            'home_away_market_join_eligible':'true' if verified_home_away else 'false',
        }
        g.update(fields)
        for s in rev_sides[k]:
            s.update(fields)
            s['is_home_verified']=s.get('cfbd_is_home_reported','') if verified_home_away else ''
        category[newrole]+=1;byseason[str(k[0])][newrole]+=1
    # The actual 768 disputed games must all be confirmed neutral; otherwise
    # source data changed or the earlier observation did not match the ledger.
    if previous_conflict[NEUTRAL]!=old_holds:
        raise RuntimeError('SOURCE_NEUTRAL_DOES_NOT_RECONCILE_OLD_ROLE_HOLDS:'+str(dict(previous_conflict)))
    if len(revised_games)!=len(games) or len(revised_sides)!=len(sides):
        raise RuntimeError('LEDGER_RECORD_COUNT_CHANGED')
    if any(g['bet_authority']!='0' or g['historical_prices_original_capture_verified']!='false' or g['source_signals_original_asof_verified']!='false' for g in revised_games):
        raise RuntimeError('UNEXPECTED_SOURCE_BET_AUTHORITY')
    return revised_games,revised_sides,{
        'original_home_away_conflicts':old_holds,
        'original_conflicts_confirmed_neutral':previous_conflict[NEUTRAL],
        'original_conflicts_by_new_site_status':dict(previous_conflict),
        'neutral_site_games_all':category[NEUTRAL],
        'non_neutral_games_home_away_verified':category['HOME_AWAY_VERIFIED'],
        'neutral_site_unknown_or_mixed':category[UNCERTAIN],
        'remaining_original_home_away_conflicts_on_non_neutral_games':category[OLD_HOLD],
        'role_status_counts':dict(category),
        'by_season':{k:dict(v) for k,v in sorted(byseason.items())},
        'source_venue_confirmation':'BOTH_HISTORICAL_TEAM_SIDES_IS_NEUTRAL_1_NOT_INDEPENDENT_VENUE_PROVENANCE',
    }


def run(input_dir,out_dir,client):
    input_dir=Path(input_dir);out_dir=Path(out_dir)
    if out_dir.resolve()==input_dir.resolve():raise RuntimeError('REFUSE_TO_OVERWRITE_V5_SOURCE')
    if out_dir.exists() and any(out_dir.iterdir()):raise RuntimeError('REFUSE_TO_OVERWRITE_NONEMPTY_OUTPUT_DIRECTORY')
    old=json.loads((input_dir/AUDIT).read_text())
    if old.get('status')!='STAGE1_CANONICAL_SOURCE_LEDGER_V5_RESEARCH_ONLY' or old.get('bet_authority')!=0 or old.get('v4_reconciliation',{}).get('status')!='PASS_EXACT_V4_GAME_MEMBERSHIP_REPLAY' or old.get('historic_brain_original_asof_certified') is not False or old.get('historic_book_quote_capture_certified') is not False:
        raise RuntimeError('V5_SOURCE_AUDIT_NOT_VALID')
    for f in (MANIFEST,SIDES):
        if old['outputs_sha256'].get(f)!=digest(input_dir/f):raise RuntimeError('V5_SOURCE_CHECKSUM_MISMATCH:'+f)
    games,gcols=readcsv(input_dir/MANIFEST);sides,scols=readcsv(input_dir/SIDES)
    recs=[]
    for r in client.query(SQL).result():
        recs.append({k:getattr(r,k) for k in ('season','source_game_id','source_sides','neutral_sides','non_neutral_sides','unknown_neutral_sides')})
    revised_games,revised_sides,summary=reconcile(games,sides,source_lookup(recs),old)
    out_dir.mkdir(parents=True,exist_ok=True)
    added=('original_v5_role_status','source_neutral_site_status','source_neutral_both_sides_verified','neutral_site_research_eligible')
    writecsv(out_dir/MANIFEST,revised_games,gcols+[x for x in added if x not in gcols])
    writecsv(out_dir/SIDES,revised_sides,scols+[x for x in added if x not in scols])
    audit=copy.deepcopy(old)
    audit['stage1_neutral_v5_2']={'status':'PASS_SOURCE_NEUTRAL_TWO_SIDED_RECONCILIATION',
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'input_v5_audit_sha256':digest(input_dir/AUDIT),
        'input_v5_csv_sha256':{f:digest(input_dir/f) for f in (MANIFEST,SIDES)},
        'source_table':SOURCE_TABLE,'read_only':True,'v5_compatibility':'PRESERVED_FOR_STAGE1_V6',
        **summary}
    audit['cfbd_games_with_verified_home_away']=summary['non_neutral_games_home_away_verified']
    audit['cfbd_games_held_home_away_role']=len(revised_games)-summary['non_neutral_games_home_away_verified']
    audit['cfbd_games_source_confirmed_neutral']=summary['neutral_site_games_all']
    audit['details']['role_status_counts']=summary['role_status_counts']
    audit['outputs_sha256']={f:digest(out_dir/f) for f in (MANIFEST,SIDES)}
    (out_dir/AUDIT).write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    print('STAGE1_NEUTRAL_V5_2='+str(out_dir/AUDIT))
    for k in ('original_home_away_conflicts','original_conflicts_confirmed_neutral','neutral_site_games_all','non_neutral_games_home_away_verified','neutral_site_unknown_or_mixed','remaining_original_home_away_conflicts_on_non_neutral_games'):
        print(k.upper()+'='+str(summary[k]))
    print('BY_SEASON='+json.dumps(summary['by_season'],sort_keys=True))
    print('V5_COMPATIBLE_FOR_V6=true; FROZEN_345_RULES_UNCHANGED; BET_AUTHORITY=0')
    return audit


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',required=True,type=Path)
    p.add_argument('--out-dir',required=True,type=Path)
    ns=p.parse_args(argv)
    from google.cloud import bigquery
    run(ns.input_dir,ns.out_dir,bigquery.Client(project='sharplogger'))

if __name__=='__main__':main()
