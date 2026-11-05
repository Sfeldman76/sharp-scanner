#!/usr/bin/env python3
"""Stage 1 V5: repair identity-vs-home-role conflation in V4.

V3 crosswalk is authoritative for accepted (season, native ID)->CFBD Game_ID.
Requires reciprocal source team names, matched result sides, and reciprocal
scores; never infers a match from score, ATS, bookmaker, or outcome. Retains
identity-matched games even where source Is_Home conflicts with CFBD Is_Home,
while explicitly withholding home/away role, orientation-sensitive mechanisms,
and directional market joins for those games.

Research only. Does not modify existing brains, Cloud Run, BigQuery, GCS,
existing outputs or bet authority. All historical prices remain uncertified.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone

import ncaaf_v55_stage1_canonical_ledger_v4 as v4


def _paired_flags(values):
    """Both opposing sides must be explicitly and oppositely marked 1/0."""
    return len(values) == 2 and set(values) == {'0','1'}


def match_game_sides(src_rows, cfbd_sides, source_results):
    """Return (side-pairs, reason, role status), never a guessed orientation.

    Source-original game name can be used to resolve true side/team identity;
    Is_Home is an *independent* piece of evidence and cannot veto game-ID
    identity that has already been uniquely verified from original names.
    """
    if len(src_rows) != 2 or len(source_results) != 2 or len(cfbd_sides) != 2:
        return None, 'HOLD_SOURCE_SIDE_COUNT', None
    if len({r.get('team_key') for r in source_results}) != 2:
        return None, 'HOLD_DUPLICATE_RESULT_SIDE', None
    aligned=[]; seen_warehouse=set();seen_cfbd=set()
    for r in source_results:
        matches=[i for i,s in enumerate(src_rows) if v4.source_result_name_matches(r,s)]
        if len(matches)!=1 or matches[0] in seen_warehouse:
            return None,'HOLD_RESULT_TO_WAREHOUSE_SIDE',None
        i=matches[0];seen_warehouse.add(i)
        original=src_rows[i]
        a=v4.school(original.get('Source_Team_Name'))
        b=v4.school(original.get('Source_Opponent_Name'))
        if not a or not b or a==b:
            return None,'HOLD_ORIGINAL_SOURCE_NAMES_MISSING',None
        matches=[c for c in cfbd_sides.values()
            if v4.school(c.get('team_key'))==a and v4.school(c.get('opponent_key'))==b]
        if len(matches)!=1:return None,'HOLD_SOURCE_TO_CFBD_DIRECTION',None
        c=matches[0]
        if c['team_key'] in seen_cfbd:return None,'HOLD_DUPLICATE_CFBD_SIDE',None
        seen_cfbd.add(c['team_key'])
        if r.get('cfbd_game_id') and str(r['cfbd_game_id']).strip()!=str(c['cfbd_game_id']).strip():
            return None,'HOLD_ORIGINAL_GAME_ID_CONFLICT',None
        aligned.append((r,c))
    (ra,ca),(rb,cb)=aligned
    if ca['team_key']!=cb['opponent_key'] or cb['team_key']!=ca['opponent_key']:
        return None,'HOLD_NONRECIPROCAL_ALIGNED_SIDES',None
    for col in ('team_score','opponent_score'):
        if (ra.get(col) in ('',None)) != (rb.get(col) in ('',None)):
            return None,'HOLD_PARTIAL_GAME_SCORE',None
    if ra.get('team_score') not in ('',None):
        if ra['team_score']!=rb['opponent_score'] or ra['opponent_score']!=rb['team_score']:
            return None,'HOLD_CONFLICTING_GAME_SCORES',None

    sf=[str(r.get('is_home','')).strip() for r,c in aligned]
    cf=[str(c.get('is_home','')).strip() for r,c in aligned]
    # Reproduce V4's *per-side* explicit mismatch criterion exactly. A source
    # pair such as [1,1] against CFBD [1,0] has a contradictory side even
    # though source flags are not reciprocal. Previous V5 misclassified that
    # as merely unknown, causing false V4/V5 reconciliation failures.
    explicit_contradiction=any(
        a in ('0','1') and b in ('0','1') and a != b
        for a,b in zip(sf,cf)
    )
    if explicit_contradiction:
        role='HOME_AWAY_CONFLICT_UNRESOLVED'
    elif _paired_flags(sf) and _paired_flags(cf) and sf == cf:
        role='HOME_AWAY_VERIFIED'
    else:
        role='HOME_AWAY_UNVERIFIED_MISSING_OR_NONRECIPROCAL'
    return aligned,None,role


def build_ledger(accepted, canonical, side_results, warehouse_sides, providers, snapshots):
    valid,bad=v4.canonical_two_sided(canonical)
    result_index=defaultdict(list)
    for r in side_results:
        y=v4.season(r.get('season')); sid=str(r.get('source_game_id','')).strip()
        if y and sid: result_index[(y,sid)].append(r)
    warehouse_index=v4.source_sides_from_warehouse(warehouse_sides)
    quotes=defaultdict(Counter)
    for q in providers:
        y=v4.season(q.get('season'));gid=str(q.get('cfbd_game_id','')).strip()
        if y and gid and q.get('game_join')=='EXACT_CFBD_ID_AND_TEAMS':
            quotes[(y,gid)]['provider_rows_untimed']+=1
    for q in snapshots:
        y=v4.season(q.get('season'));gid=str(q.get('cfbd_game_id','')).strip()
        if y and gid and q.get('game_join')=='EXACT_DATE_HOME_AWAY':
            quotes[(y,gid)]['snapshot_rows_timestamped_unattested']+=1
            if q.get('quote_chronology')=='PREKICK_TIME_SOURCE_CAPTURE_UNATTESTED':
                quotes[(y,gid)]['snapshot_rows_prekick_unattested']+=1
    side_rows=[];games=[];hold=Counter();role_counts=Counter();yrs=defaultdict(Counter)
    for (year,sid),xref in sorted(accepted.items()):
        gid=str(xref['cfbd_game_id']).strip();k=(year,gid)
        if k not in valid:
            hold['HOLD_NO_VALID_CFBD_TWO_SIDED']+=1;yrs[year]['held_game_identity']+=1
            continue
        paired, reason, role=match_game_sides(warehouse_index.get((year,sid),[]),
                   valid[k],result_index.get((year,sid),[]))
        if reason:
            hold[reason]+=1;yrs[year]['held_game_identity']+=1
            continue
        verified=(role=='HOME_AWAY_VERIFIED')
        role_counts[role]+=1;yrs[year]['linked_games']+=1
        yrs[year][role]+=1
        for r,c in paired:
            sf=str(r.get('is_home','')).strip();cf=str(c.get('is_home','')).strip()
            side_rows.append({
                'season':year,'cfbd_game_id':gid,'source_game_id':sid,
                'game_date':c['game_date'],'source_game_date':r.get('game_date',''),
                'cfbd_team_key':c['team_key'],'cfbd_opponent_key':c['opponent_key'],
                'source_team_label':r['team_key'],'source_opponent_label':r['opponent_key'],
                'source_is_home_reported':sf,'cfbd_is_home_reported':cf,
                'is_home_verified':cf if verified else '',
                'role_verification_status':role,
                'role_sensitive_research_eligible':'true' if verified else 'false',
                'home_away_market_join_eligible':'true' if verified else 'false',
                'team_score':r.get('team_score',''),'opponent_score':r.get('opponent_score',''),
                'retrospective_open_spread':r.get('historical_context_open_spread',''),
                'retrospective_open_spread_result':r.get('retrospective_open_spread_result',''),
                'retrospective_price_vintage':'UNTIMED_HISTORICAL_OPEN_RESEARCH_ONLY',
                'split':v4.walkforward_split(year),
                'original_signal_capture_certified':'false',
                'original_price_capture_certified':'false','bet_authority':0})
        q=quotes.get(k,Counter())
        games.append({'season':year,'cfbd_game_id':gid,'source_game_id':sid,
            'game_date':paired[0][1]['game_date'],'split':v4.walkforward_split(year),
            'source_game_match':'VERIFIED_TWO_SIDED_V3',
            'canonical_side_identity':'VERIFIED_TWO_SIDED_V5',
            'role_verification_status':role,
            'role_sensitive_research_eligible':'true' if verified else 'false',
            'home_away_market_join_eligible':'true' if verified else 'false',
            'number_of_sides':2,
            'provider_quote_rows_untimed':q['provider_rows_untimed'],
            'snapshot_rows_timestamped_unattested':q['snapshot_rows_timestamped_unattested'],
            'snapshot_rows_prekick_unattested':q['snapshot_rows_prekick_unattested'],
            'source_signals_original_asof_verified':'false',
            'historical_prices_original_capture_verified':'false',
            'bet_authority':0})
        yrs[year]['provider_quote_rows_untimed']+=q['provider_rows_untimed']
        yrs[year]['snapshot_rows_prekick_unattested']+=q['snapshot_rows_prekick_unattested']
    details={'held_identity_reasons':dict(hold), 'role_status_counts':dict(role_counts),
       'by_season':{str(k):dict(v) for k,v in sorted(yrs.items())},
       'cfbd_invalid_games':dict(bad)}
    return games,side_rows,details


def run(source_dir, crosswalk_dir, out_dir, client, previous_audit_dir=None):
    source_dir,crosswalk_dir,out_dir=map(Path,(source_dir,crosswalk_dir,out_dir))
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError('REFUSE_TO_OVERWRITE_NONEMPTY_OUTPUT_DIRECTORY')
    accepted,_,xa,sa,held=v4.load_verified_inputs(source_dir,crosswalk_dir)
    source_identities,identity_manifest=v4.read_bq_identity(client,'ncaaf_historical_game_side_context')
    if not source_identities:raise RuntimeError('WAREHOUSE_SOURCE_IDENTITIES_EMPTY')
    can=v4.readcsv(source_dir/v4.CFBD)
    results=v4.readcsv(source_dir/v4.SIDE)
    providers=v4.readcsv(source_dir/v4.PROVIDER)
    snapshots=v4.readcsv(source_dir/v4.SNAPSHOT)
    games,sides,details=build_ledger(accepted,can,results,source_identities,providers,snapshots)
    if not games or len(sides)!=2*len(games):
        raise RuntimeError('INVALID_OR_EMPTY_TWO_SIDED_RESULT_LEDGER')
    if previous_audit_dir:
        audit_path=Path(previous_audit_dir)/'stage1_canonical_ledger_v4_audit.json'
        prev=json.loads(audit_path.read_text())
        if (prev.get('v3_matched_games')!=len(accepted) or
            prev.get('crosswalk_sha256')!=v4.file_hash(crosswalk_dir/v4.CROSSWALK) or
            prev.get('source_sha256')!={n:v4.file_hash(source_dir/n) for n in v4.NEEDED}):
            raise RuntimeError('PREVIOUS_V4_INPUTS_DIFFER_FROM_V5')
        # Replay the actual V4 alignment contract against the same immutable
        # inputs, rather than comparing different concepts (identity count vs.
        # fully aligned count). This also verifies *membership*, not just totals.
        v4_games, v4_sides, v4_detail=v4.build_ledger(
            accepted,can,results,source_identities,providers,snapshots)
        expected_v4=int(prev['canonical_aligned_games'])
        if len(v4_games) != expected_v4 or len(v4_sides) != 2*len(v4_games):
            raise RuntimeError('V4_INPUT_REPLAY_MISMATCH:'+
                f'{len(v4_games)}!={expected_v4}')
        if prev.get('alignment',{}).get('held_alignment_reasons') is not None:
            if v4_detail['held_alignment_reasons'] != prev['alignment']['held_alignment_reasons']:
                raise RuntimeError('V4_INPUT_REPLAY_REASON_MISMATCH:'+
                    repr(v4_detail['held_alignment_reasons']))
        v4_members={
            (int(g['season']),str(g['source_game_id']),str(g['cfbd_game_id']))
            for g in v4_games}
        v5_strict_members={
            (int(g['season']),str(g['source_game_id']),str(g['cfbd_game_id']))
            for g in games
            if g['role_verification_status']!='HOME_AWAY_CONFLICT_UNRESOLVED'}
        if v4_members != v5_strict_members:
            only_v4=sorted(v4_members-v5_strict_members)[:5]
            only_v5=sorted(v5_strict_members-v4_members)[:5]
            raise RuntimeError('V4_V5_RECONCILIATION_MISMATCH:'+
                f'v4={len(v4_members)} v5_strict={len(v5_strict_members)} '+
                f'only_v4={only_v4!r} only_v5={only_v5!r}')
        reconciliation={
            'status':'PASS_EXACT_V4_GAME_MEMBERSHIP_REPLAY',
            'v4_games_from_previous_audit':expected_v4,
            'v4_games_replayed_from_source':len(v4_members),
            'v5_games_recovered_identity_only':len(games),
            'v5_identity_games_with_explicit_home_away_contradiction':sum(
                g['role_verification_status']=='HOME_AWAY_CONFLICT_UNRESOLVED'
                for g in games),
            'v5_strict_game_count_matching_v4':len(v5_strict_members)
        }
    else:
        reconciliation={'status':'NOT_CHECKED_NO_PREVIOUS_V4_AUDIT'}
    out_dir.mkdir(parents=True,exist_ok=True)
    mcols=['season','cfbd_game_id','source_game_id','game_date','split',
      'source_game_match','canonical_side_identity','role_verification_status',
      'role_sensitive_research_eligible','home_away_market_join_eligible',
      'number_of_sides','provider_quote_rows_untimed',
      'snapshot_rows_timestamped_unattested','snapshot_rows_prekick_unattested',
      'source_signals_original_asof_verified','historical_prices_original_capture_verified',
      'bet_authority']
    scols=['season','cfbd_game_id','source_game_id','game_date','source_game_date',
      'cfbd_team_key','cfbd_opponent_key','source_team_label','source_opponent_label',
      'source_is_home_reported','cfbd_is_home_reported','is_home_verified',
      'role_verification_status','role_sensitive_research_eligible',
      'home_away_market_join_eligible','team_score','opponent_score',
      'retrospective_open_spread','retrospective_open_spread_result',
      'retrospective_price_vintage','split','original_signal_capture_certified',
      'original_price_capture_certified','bet_authority']
    v4.writecsv(out_dir/'historical_cfbd_game_manifest_v5.csv',games,mcols)
    v4.writecsv(out_dir/'historical_cfbd_game_sides_v5.csv',sides,scols)
    r={'status':'STAGE1_CANONICAL_SOURCE_LEDGER_V5_RESEARCH_ONLY',
       'generated_at_utc':datetime.now(timezone.utc).isoformat(),
       'historical_window':[2013,2025],'discovery':[2013,2023],
       'confirmation_season':2024,'final_holdout_season':2025,
       'source_games':int(xa['source_games_evaluated']),
       'v3_matched_games':len(accepted),'v3_held_games':sum(held.values()),
       'cfbd_games_linked_identity_only':len(games),
       'cfbd_sides_linked_identity_only':len(sides),
       'cfbd_games_with_verified_home_away':sum(g['role_sensitive_research_eligible']=='true' for g in games),
       'cfbd_games_held_home_away_role':sum(g['role_sensitive_research_eligible']!='true' for g in games),
       'held_identity_after_v3':len(accepted)-len(games),
       'v4_reconciliation':reconciliation,
       'details':details,
       'source_sha256':{n:v4.file_hash(source_dir/n) for n in v4.NEEDED},
       'crosswalk_sha256':v4.file_hash(crosswalk_dir/v4.CROSSWALK),
       'warehouse_identity_manifest':identity_manifest,
       'historic_book_quote_capture_certified':False,
       'historic_brain_original_asof_certified':False,
       'historical_combination_certified':False,
       'miner_345_rule_library_modified':False,
       'production_tables_modified':False,
       'model_retrained':False,'bet_authority':0,
       'next':'ATTACH_FROZEN_BRAIN_EVIDENCE_WITH_INDEPENDENT_ORIGINAL_VINTAGE_AND_QUOTE_AUDIT'}
    r['outputs_sha256']={p.name:v4.file_hash(p) for p in sorted(out_dir.glob('*.csv'))}
    (out_dir/'stage1_canonical_ledger_v5_audit.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print('STAGE1_V5_AUDIT='+str(out_dir/'stage1_canonical_ledger_v5_audit.json'))
    print('V3_MATCHED_GAMES='+str(len(accepted)))
    print('IDENTITY_LINKED_GAMES='+str(len(games)))
    print('HOME_AWAY_VERIFIED_GAMES='+str(r['cfbd_games_with_verified_home_away']))
    print('HOME_AWAY_ROLE_HELD_GAMES='+str(r['cfbd_games_held_home_away_role']))
    print('HELD_GAME_IDENTITY='+str(r['held_identity_after_v3']))
    print('V4_RECONCILIATION='+json.dumps(reconciliation,sort_keys=True))
    print('ROLE_STATUS='+json.dumps(details['role_status_counts'],sort_keys=True))
    print('BY_YEAR='+json.dumps(details['by_season'],sort_keys=True))
    print('RESEARCH_ONLY; BET_AUTHORITY=0; ALL_EXISTING_BRAINS_FROZEN')
    return r


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--crosswalk-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--previous-audit-dir',type=Path,default=None)
    ns=p.parse_args(argv)
    from google.cloud import bigquery
    return run(ns.source_dir,ns.crosswalk_dir,ns.out_dir,
      bigquery.Client(project='sharplogger'),ns.previous_audit_dir)

if __name__=='__main__':main()
