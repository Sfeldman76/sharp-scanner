#!/usr/bin/env python3
"""NCAAF V5.5 Stage 2 -- existing seven-brain historical evidence combination.

Read-only, provenance-explicit, fail-closed. No mining, refitting, BQ writes,
GCS publishes, production deployment, or betting authority. Every accepted
observation must come from a frozen existing event record; a rule definition
or postgame feature value alone is NOT a fired pregame pick.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

BRAINS = ('CORE','STAT','SYSTEM_MINER','PATHI','BIG_AL','PT','MARKET_RICH')
V6_NAMES = {'PREDICTION_TRACKER':'PT','MINER':'SYSTEM_MINER', 'BIGAL':'BIG_AL', 'MARKETRICH':'MARKET_RICH'}
LEDGER_AUDIT = 'stage1_canonical_ledger_v5_audit.json'
LEDGER_GAMES = 'historical_cfbd_game_manifest_v5.csv'
LEDGER_SIDES = 'historical_cfbd_game_sides_v5.csv'
V6_AUDIT = 'stage1_frozen_brain_attach_v6_audit.json'
V6_OBS = 'historical_frozen_brain_observations_v6.csv'
PROVIDER = 'historical_provider_quotes.csv'
SNAPSHOTS = 'historical_snapshot_quotes.csv'
SOURCE_AUDIT = 'stage1_source_separation_audit.json'


def text(x):return str(x or '').strip()
def yn(x):return text(x).lower() in ('true','1','yes','y')
def norm(x):return re.sub(r'[^a-z0-9]+','',text(x).lower())
def hashfile(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def readcsv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames:raise RuntimeError('EMPTY_CSV_HEADER:'+str(path))
        return list(reader),list(reader.fieldnames)
def writecsv(path,data,fields):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(data)
def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def year(x):
    try:r=int(x)
    except (TypeError,ValueError):return None
    return r if 2013<=r<=2025 else None

def utc(x):
    x=text(x)
    if not x:return None
    try:
        d=datetime.fromisoformat(x.replace('Z','+00:00'))
        return d if d.tzinfo is not None else None
    except ValueError:return None

def validated_ledger(root):
    root=Path(root)
    audit=load_json(root/LEDGER_AUDIT)
    if (audit.get('status')!='STAGE1_CANONICAL_SOURCE_LEDGER_V5_RESEARCH_ONLY'
            or audit.get('bet_authority')!=0
            or audit.get('stage1_neutral_v5_2',{}).get('status')!='PASS_SOURCE_NEUTRAL_TWO_SIDED_RECONCILIATION'
            or audit.get('v4_reconciliation',{}).get('status')!='PASS_EXACT_V4_GAME_MEMBERSHIP_REPLAY'):
        raise RuntimeError('V5_2_LEDGER_NOT_VALIDATED')
    games,_=readcsv(root/LEDGER_GAMES);sides,_=readcsv(root/LEDGER_SIDES)
    for f in (LEDGER_GAMES,LEDGER_SIDES):
        if audit.get('outputs_sha256',{}).get(f)!=hashfile(root/f):
            raise RuntimeError('V5_2_LEDGER_HASH_MISMATCH:'+f)
    if len(games)!=int(audit['cfbd_games_linked_identity_only']) or len(sides)!=2*len(games):
        raise RuntimeError('V5_2_LEDGER_COUNT_MISMATCH')
    lookup={}; native={}; byside=defaultdict(dict)
    for g in games:
        y=year(g.get('season'));gid=text(g.get('cfbd_game_id'));sid=text(g.get('source_game_id'))
        if not y or not gid or not sid or (y,gid) in lookup or (y,sid) in native:
            raise RuntimeError('BAD_OR_DUPLICATED_LEDGER_IDENTITY')
        if text(g['role_verification_status']) not in ('HOME_AWAY_VERIFIED','NEUTRAL_SITE_SOURCE_BOTH_SIDES_CONFIRMED'):
            raise RuntimeError('UNKNOWN_LEDGER_SITE_ROLE')
        lookup[(y,gid)]=g;native[(y,sid)]=gid
    for s in sides:
        key=(year(s.get('season')),text(s.get('cfbd_game_id')))
        if key not in lookup:raise RuntimeError('ORPHAN_LEDGER_SIDE')
        side=text(s.get('cfbd_team_key'))
        if not side or side in byside[key]:raise RuntimeError('DUPLICATE_LEDGER_SIDE')
        byside[key][side]=s
    if len(byside)!=len(lookup) or any(len(v)!=2 for v in byside.values()):
        raise RuntimeError('LEDGER_NOT_TWO_SIDED')
    return audit,lookup,native,byside


def validated_v6(root,ledger_root,lookup):
    root=Path(root)
    audit=load_json(root/V6_AUDIT)
    if (audit.get('status')!='STAGE1_FROZEN_BRAIN_OCCURRENCES_ATTACHED_RETROSPECTIVE_ONLY'
            or audit.get('bet_authority')!=0 or audit.get('original_signal_vintage_certified') is not False
            or audit.get('miner_345_rules_unchanged') is not True):
        raise RuntimeError('V6_NOT_VALIDATED')
    if audit.get('identity_linked_games')!=len(lookup):raise RuntimeError('V6_V5_GAME_COUNT_MISMATCH')
    for fname in (LEDGER_AUDIT,LEDGER_GAMES,LEDGER_SIDES):
        if audit['input_v5_sha256'].get(fname)!=hashfile(Path(ledger_root)/fname):
            raise RuntimeError('V6_WAS_BUILT_FROM_A_DIFFERENT_LEDGER:'+fname)
    if audit.get('output_sha256',{}).get(V6_OBS)!=hashfile(root/V6_OBS):
        raise RuntimeError('V6_OBSERVATIONS_HASH_MISMATCH')
    obs,_=readcsv(root/V6_OBS)
    if len(obs)!=audit['observations_attached']:raise RuntimeError('V6_OBS_COUNT_MISMATCH')
    return audit,obs


def normalize_events(v6_rows,lookup,byside):
    """V6 records are evidence only, never proof of an original pregame pick."""
    events=[];holds=Counter();unique={}
    for i,r in enumerate(v6_rows):
        y=year(r.get('season'));gid=text(r.get('cfbd_game_id')); b=V6_NAMES.get(text(r.get('brain')).upper(),text(r.get('brain')).upper())
        if not y or (y,gid) not in lookup or b not in BRAINS:
            holds['BAD_GAME_OR_BRAIN']+=1;continue
        market=text(r.get('market')).lower()
        if market not in ('spreads','h2h','totals'):
            holds['BAD_MARKET']+=1;continue
        direction=text(r.get('direction')).upper()
        team=text(r.get('side_cfbd_team_key'))
        if market!='totals':
            if direction not in ('PLAY_ON_TEAM','PLAY_ON_NAMED_TEAM') or team not in byside[(y,gid)]:
                holds['BAD_DIRECTIONAL_SIDE']+=1;continue
            target=team
        else:
            if direction not in ('OVER','UNDER'):
                holds['BAD_TOTALS_DIRECTION']+=1;continue
            target=direction
        sys=text(r.get('system_id'))
        if not sys:
            holds['MISSING_SYSTEM_ID']+=1;continue
        if text(r.get('source_game_id'))!=text(lookup[(y,gid)].get('source_game_id')):
            holds['GAME_ID_NAMESPACES_DIFFER']+=1;continue
        if yn(r.get('authority_eligible')) or yn(r.get('original_pregame_signal_certified')) or text(r.get('bet_authority')) not in ('','0'):
            raise RuntimeError('UNEXPECTED_AUTHORIZED_V6_ROW:'+str(i))
        key=(y,gid,b,sys,market,target)
        evidence={
            'season':y,'cfbd_game_id':gid,'game_date':lookup[(y,gid)]['game_date'],
            'split':lookup[(y,gid)]['split'],'brain':b,'system_id':sys,'market':market,'target':target,
            'site_status':lookup[(y,gid)]['role_verification_status'],
            'source_name':text(r.get('source_name')),
            'source_grade':text(r.get('source_grade')).upper(),
            'original_observed_at':text(r.get('original_observed_at')),
            'signal_capture_certified':False,'rule_frozen_at_game_certified':False,'authority_eligible':False
        }
        if key in unique:
            if unique[key]!=evidence:raise RuntimeError('CONFLICTING_FROZEN_EVENTS:'+repr(key))
            holds['IDENTICAL_DUPLICATE_SUPPRESSED']+=1;continue
        unique[key]=evidence;events.append(evidence)
    return events,holds


def external_adapter(raw,lookup,native,byside,label,id_namespace):
    """Attach existing, explicit FIRE records. Never infer FIRE from a rule definition.

    Required: season, brain, system_id, market, evaluation_status='FIRE',
    direction, game_id/source_game_id/cfbd_game_id; explicit named team for
    spread/H2H.  NO asof certification without separate original proof.
    """
    out=[];holds=Counter()
    for r in raw:
        if text(r.get('evaluation_status')).upper() not in ('FIRE','FIRED'):
            holds['NOT_AN_EXPLICIT_FIRE']+=1;continue
        y=year(r.get('season'));b=V6_NAMES.get(text(r.get('brain')).upper(),text(r.get('brain')).upper())
        if not y or b not in BRAINS:
            holds['BAD_SEASON_OR_BRAIN']+=1;continue
        external_id=text(r.get('game_id') or (r.get('source_game_id') if id_namespace=='source' else r.get('cfbd_game_id')))
        gid=native.get((y,external_id),'') if id_namespace=='source' else external_id
        if (y,gid) not in lookup:
            holds['NO_VERIFIED_CANONICAL_GAME_ID']+=1;continue
        market=text(r.get('market')).lower()
        if market not in ('spreads','h2h','totals'):
            holds['UNSUPPORTED_MARKET']+=1;continue
        direction=text(r.get('direction')).upper()
        if market=='totals':
            if direction not in ('OVER','UNDER'):holds['INVALID_TOTALS_DIRECTION']+=1;continue
            target=direction
        else:
            requested=text(r.get('team_key') or r.get('team') or r.get('side_cfbd_team_key'))
            teams=byside[(y,gid)];matches=[t for t,s in teams.items() if norm(requested) in (norm(t),norm(s['source_team_label']))]
            if direction in ('HOME','AWAY','ROAD','PLAY_HOME','PLAY_AWAY'):
                if lookup[(y,gid)]['role_verification_status']!='HOME_AWAY_VERIFIED':
                    holds['VENUE_DIRECTION_NOT_VERIFIED']+=1;continue
                desired='1' if 'HOME' in direction else '0'
                matches=[t for t,s in teams.items() if text(s.get('is_home_verified'))==desired]
            elif direction not in ('PLAY_ON_TEAM','PLAY_ON_NAMED_TEAM','TEAM'):
                holds['NO_EXPLICIT_PLAY_ON_SIDE']+=1;continue
            if len(matches)!=1:holds['SIDE_AMBIGUOUS_OR_MISSING']+=1;continue
            target=matches[0]
        sid=text(r.get('system_id') or r.get('mechanism_id'))
        if not sid:holds['MISSING_SYSTEM_ID']+=1;continue
        out.append({'season':y,'cfbd_game_id':gid,'game_date':lookup[(y,gid)]['game_date'],
            'split':lookup[(y,gid)]['split'],'brain':b,'system_id':sid,'market':market,
            'target':target,'site_status':lookup[(y,gid)]['role_verification_status'],
            'source_name':label,'source_grade':'UNKNOWN','original_observed_at':text(r.get('original_observed_at')),
            'signal_capture_certified':False,'rule_frozen_at_game_certified':False,'authority_eligible':False})
    return out,holds


def validate_market_files(source_dir,lookup):
    root=Path(source_dir);a=load_json(root/SOURCE_AUDIT)
    if a.get('status')!='STAGE1_SOURCE_SEPARATION_BUILT_RESEARCH_ONLY' or a.get('bet_authority')!=0:
        raise RuntimeError('INVALID_STAGE1_MARKET_EXPORT')
    for n in (PROVIDER,SNAPSHOTS):
        if a.get('files_sha256',{}).get(n)!=hashfile(root/n):
            raise RuntimeError('MARKET_SOURCE_SHA256_MISMATCH:'+n)
    pr,_=readcsv(root/PROVIDER);sn,_=readcsv(root/SNAPSHOTS)
    counts=Counter();pergame=defaultdict(Counter);prices=[];seen=set()
    for r in pr:
        counts['provider_rows_read']+=1
        y=year(r.get('season'));gid=text(r.get('cfbd_game_id'))
        if (y,gid) not in lookup or r.get('game_join')!='EXACT_CFBD_ID_AND_TEAMS':
            counts['provider_unlinked_or_not_verified']+=1;continue
        counts['provider_games_linked_rows']+=1
        pergame[(y,gid)]['untimed_provider_quotes']+=1
        # No original capture timestamp. Prices may be used descriptively only.
        prices.append({'season':y,'cfbd_game_id':gid,'market_source':'PROVIDER_ARCHIVE',
            'provider':text(r.get('provider')),'market':'MIXED_UNTIMED',
            'observed_utc':'','kickoff_utc':'','raw_market':text(r.get('spread_quoted')),
            'price_status':'HISTORICAL_UNTIMED_RESEARCH_ONLY','quote_certified':False})
    for r in sn:
        counts['snapshot_rows_read']+=1
        y=year(r.get('season'));gid=text(r.get('cfbd_game_id'))
        if (y,gid) not in lookup or r.get('game_join')!='EXACT_DATE_HOME_AWAY':
            counts['snapshot_no_verified_cfbd_link']+=1;continue
        d=utc(r.get('snapshot_utc'));kick=utc(r.get('kickoff_utc'))
        if d is None or kick is None or d>=kick:
            counts['snapshot_linked_not_prekick']+=1;continue
        # Link must already be valid in Stage1. Chronology alone is not capture proof.
        k=(y,gid,text(r.get('book')),text(r.get('market')),text(r.get('side')),d.isoformat(),text(r.get('line')),text(r.get('odds')))
        if k in seen:counts['snapshot_duplicate_suppressed']+=1;continue
        seen.add(k)
        counts['snapshot_linked_timestamped_prekick_unattested']+=1
        pergame[(y,gid)]['timestamped_prekick_unattested']+=1
        prices.append({'season':y,'cfbd_game_id':gid,'market_source':'SNAPSHOT_STAGE1',
            'provider':text(r.get('book')),'market':text(r.get('market')),
            'observed_utc':d.isoformat(),'kickoff_utc':kick.isoformat(),
            'raw_market':text(r.get('line')),'price_status':'PREKICK_TIMESTAMP_UNATTESTED',
            'quote_certified':False})
    return prices,counts,pergame


def retrospective_grade(e, side_map):
    """Descriptive result only; reference is untimed historical context open."""
    if e['market']=='totals':return 'NO_SIDE_SPECIFIC_TOTAL_QUOTE'
    s=side_map[(e['season'],e['cfbd_game_id'])].get(e['target'])
    if not s:return 'NO_RESULT_SIDE'
    if e['market']=='spreads':
        x=text(s.get('retrospective_open_spread_result')).upper()
        return x if x in ('WIN','LOSS','PUSH') else 'UNPRICED_OR_UNGRADED'
    try:a=float(s['team_score']);b=float(s['opponent_score'])
    except (ValueError,TypeError):return 'UNSCORED'
    return 'WIN' if a>b else ('LOSS' if a<b else 'PUSH')


def combine(events,lookup,byside,pergame):
    """Preserve brain family independence, side direction and conflict vetoes."""
    dedup={};suppressed=0
    for e in events:
        k=(e['season'],e['cfbd_game_id'],e['brain'],e['system_id'],e['market'],e['target'])
        if k in dedup:
            if dedup[k]!=e:raise RuntimeError('CONTRADICTORY_DUPLICATE_BRAIN_FIRE:'+repr(k))
            suppressed+=1;continue
        dedup[k]=e
    clean=sorted(dedup.values(),key=lambda e:(e['season'],e['cfbd_game_id'],e['market'],e['target'],e['brain'],e['system_id']))
    games=defaultdict(lambda:defaultdict(set))
    grouped_events=defaultdict(list)
    for e in clean:
        group=(e['season'],e['cfbd_game_id'],e['market'])
        games[group][e['brain']].add(e['target'])
        grouped_events[group].append(e)
    candidates=[]; summaries=Counter()
    for (y,gid,market),bybrain in sorted(games.items()):
        # Cohort is a property of the game, NOT of an individual candidate.
        # A game with only intra-family contradictions has no stable target,
        # so the candidate loop below may run zero times.
        cohort=lookup[(y,gid)]['split']
        stable={b:next(iter(ts)) for b,ts in bybrain.items() if len(ts)==1}
        conflicting_within={b for b,ts in bybrain.items() if len(ts)>1}
        targets=set(stable.values())
        for target in sorted(targets):
            supporters=sorted(b for b,t in stable.items() if t==target)
            contradictions=sorted(b for b,t in stable.items() if t!=target)
            signals=[e for e in grouped_events[(y,gid,market)] if e['target']==target and e['brain'] in supporters]
            # A brain can fire multiple systems in one family, but counts once.
            grade=retrospective_grade({'season':y,'cfbd_game_id':gid,'market':market,'target':target},byside)
            price_count=pergame[(y,gid)]
            candidates.append({'season':y,'cfbd_game_id':gid,'market':market,'candidate_target':target,
                'split':cohort,'site_status':lookup[(y,gid)]['role_verification_status'],
                'supporting_brains':';'.join(supporters),'supporting_family_count':len(supporters),
                'opposing_brains':';'.join(contradictions),'intrafamily_conflicted_brains':';'.join(sorted(conflicting_within)),
                'conflict_veto_research':str(bool(contradictions or conflicting_within)).lower(),
                'distinct_supporting_systems':len({(x['brain'],x['system_id']) for x in signals}),
                'retrospective_result_at_untimed_reference':grade,
                'untimed_provider_quote_rows':price_count['untimed_provider_quotes'],
                'timestamped_prekick_unattested_quote_rows':price_count['timestamped_prekick_unattested'],
                'original_signal_capture_certified':'false','original_price_capture_certified':'false',
                'certified_ev':'','authorized_bet':'false','bet_authority':0})
            summaries[(cohort,market,'candidates')]+=1
        if conflicting_within:
            summaries[(cohort,market,'games_intrafamily_conflicted')]+=1
            if not targets:summaries[(cohort,market,'games_intrafamily_conflicted_no_stable_side')]+=1
    return clean,candidates,summaries,suppressed


def summarize_coalitions(candidates):
    """Historical agreement counts, NOT out-of-sample-selected bet systems.

    Enumerates exact same-side supporting-family subsets without naming any
    particular source as a new independently validated betting system.
    """
    totals=defaultdict(Counter)
    for r in candidates:
        brains=tuple(filter(None,r['supporting_brains'].split(';')))
        if r['conflict_veto_research']=='true':continue
        for n in range(1,len(brains)+1):
            for subset in itertools.combinations(brains,n):
                key=(r['split'],r['market'],'+'.join(subset))
                totals[key]['candidate_count']+=1
                grade=r['retrospective_result_at_untimed_reference']
                if grade in ('WIN','LOSS','PUSH'):totals[key][grade]+=1
                else:totals[key]['UNGRADABLE']+=1
    out=[]
    for (split,market,fams),counts in sorted(totals.items()):
        n=counts['WIN']+counts['LOSS']
        win_rate=(counts['WIN']/n if n else None)
        out.append({'split':split,'market':market,'supporting_brain_set':fams,
            'total_candidates':counts['candidate_count'],'graded_wins':counts['WIN'],
            'graded_losses':counts['LOSS'],'pushes':counts['PUSH'],
            'ungradable':counts['UNGRADABLE'],
            'descriptive_win_rate_ex_pushes':round(win_rate,6) if win_rate is not None else '',
            'historical_price_capture_certified':'false','historical_signal_capture_certified':'false',
            'out_of_sample_certified':'false','historical_roi':'',
            'authority_eligible':'false','bet_authority':0})
    return out


def coverage_by_season(lookup,events):
    """Count distinct games per brain, not raw system firings or team-side records."""
    universe=defaultdict(set);observed=defaultdict(set);firing_rows=Counter()
    for (y,gid),g in lookup.items():universe[(y,g['split'])].add(gid)
    for r in events:
        k=(r['season'],r['split'],r['brain'])
        observed[k].add(r['cfbd_game_id']);firing_rows[k]+=1
    coverage=[]
    for (season,split),games in sorted(universe.items()):
        for brain in BRAINS:
            k=(season,split,brain)
            coverage.append({'season':season,'split':split,'brain':brain,
                'canonical_games':len(games),'games_with_frozen_signal':len(observed[k]),
                'frozen_event_rows':firing_rows[k],
                'signal_presence_status':'ATTACHED' if observed[k] else 'NO_ATTACHED_EVENTS',
                'original_capture_certified':'false','authority_eligible':'false'})
    return coverage


def run(ledger_dir,v6_dir,source_dir,out_dir,extra_sources=()):
    out_dir=Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):raise RuntimeError('REFUSE_TO_OVERWRITE_NONEMPTY_OUTPUT_DIR')
    audit,lookup,native,byside=validated_ledger(ledger_dir)
    v6audit,raw=validated_v6(v6_dir,ledger_dir,lookup)
    events,holds=normalize_events(raw,lookup,byside)
    external_sources=[]
    for path,namespace in extra_sources:
        rr,_=readcsv(path);more,errors=external_adapter(rr,lookup,native,byside,str(path),namespace)
        events.extend(more);holds.update(errors)
        external_sources.append({'file':str(path),'sha256':hashfile(path),'rows_read':len(rr),
                                 'attached_event_rows':len(more),'source_id_namespace':namespace,'holds':dict(errors)})
    prices,quote_counts,pergame=validate_market_files(source_dir,lookup)
    events,candidates,group_count,dupes=combine(events,lookup,byside,pergame)
    coals=summarize_coalitions(candidates)
    coverage=coverage_by_season(lookup,events)
    observed=Counter(e['brain'] for e in events)
    groups=Counter(r['supporting_family_count'] for r in candidates)
    output={'status':'STAGE2_SEVEN_BRAIN_COMBINATION_RESEARCH_UNCERTIFIED',
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'historical_game_identity_count':len(lookup),
        'neutral_site_games':sum(r['role_verification_status']=='NEUTRAL_SITE_SOURCE_BOTH_SIDES_CONFIRMED' for r in lookup.values()),
        'by_split_games':dict(Counter(g['split'] for g in lookup.values())),
        'v6_observations_consumed':len(raw),'brain_event_rows':dict(observed),
        'brains_not_attached':sorted(set(BRAINS)-set(observed)),
        'brain_family_count_expected':len(BRAINS),'duplicate_observations_deduplicated':dupes,
        'external_source_inputs':external_sources,'observation_holds':dict(holds),
        'market_quote_counts':dict(quote_counts),'market_rows_attached_to_games':len(prices),
        'candidate_count_by_family_count':{str(k):v for k,v in groups.items()},
        'candidate_rows':len(candidates),'descriptive_coalition_rows':len(coals),
        'game_market_diagnostics':{'|'.join(map(str,k)):v for k,v in sorted(group_count.items())},
        'brain_season_coverage_rows':len(coverage),
        'conflict_veto_candidate_rows':sum(yn(r['conflict_veto_research']) for r in candidates),
        'all_brains_frozen_unchanged':True,'miner_345_rules_rebuilt':False,
        'original_signal_capture_certified':False,'original_quote_capture_certified':False,
        'out_of_sample_bet_authority_certified':False,'authorized_bets':0,'bet_authority':0,
        'production_modified':False,'gcs_modified':False,'bigquery_modified':False,
        'selection_uses_2025_results':False,
        'source_sha256':{'ledger_audit':hashfile(Path(ledger_dir)/LEDGER_AUDIT),
            'v6_audit':hashfile(Path(v6_dir)/V6_AUDIT),
            'market_audit':hashfile(Path(source_dir)/SOURCE_AUDIT)},
        'remaining_gates':['ATTACH_MISSING_FROZEN_EVENT_FAMILIES',
           'ORIGINAL_PREGAME_BRAIN_VINTAGE_EVIDENCE',
           'ORIGINAL_EXECUTABLE_QUOTE_CAPTURE_AND_ODDS',
           'CHRONOLOGICAL_INDEPENDENT_FAMILY_COMBINATION_CERTIFICATION']}
    out_dir.mkdir(parents=True,exist_ok=True)
    ef=['season','cfbd_game_id','game_date','split','brain','system_id','market','target','site_status','source_name','source_grade','original_observed_at','signal_capture_certified','rule_frozen_at_game_certified','authority_eligible']
    cf=['season','cfbd_game_id','market','candidate_target','split','site_status','supporting_brains','supporting_family_count','opposing_brains','intrafamily_conflicted_brains','conflict_veto_research','distinct_supporting_systems','retrospective_result_at_untimed_reference','untimed_provider_quote_rows','timestamped_prekick_unattested_quote_rows','original_signal_capture_certified','original_price_capture_certified','certified_ev','authorized_bet','bet_authority']
    wf=['split','market','supporting_brain_set','total_candidates','graded_wins','graded_losses','pushes','ungradable','descriptive_win_rate_ex_pushes','historical_price_capture_certified','historical_signal_capture_certified','out_of_sample_certified','historical_roi','authority_eligible','bet_authority']
    pf=['season','cfbd_game_id','market_source','provider','market','observed_utc','kickoff_utc','raw_market','price_status','quote_certified']
    covercols=['season','split','brain','canonical_games','games_with_frozen_signal','frozen_event_rows','signal_presence_status','original_capture_certified','authority_eligible']
    files={'stage2_frozen_brain_events.csv':(events,ef),'stage2_same_side_candidate_matrix.csv':(candidates,cf),
           'stage2_descriptive_coalition_history.csv':(coals,wf),'stage2_historical_price_inventory.csv':(prices,pf),
           'stage2_brain_season_coverage.csv':(coverage,covercols)}
    for filename,(data,fields) in files.items():writecsv(out_dir/filename,data,fields)
    output['output_sha256']={f:hashfile(out_dir/f) for f in files}
    (out_dir/'stage2_combination_audit.json').write_text(json.dumps(output,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print('STAGE2_AUDIT='+str(out_dir/'stage2_combination_audit.json'))
    print('IDENTITY_GAMES='+str(len(lookup))+' NEUTRAL_GAMES='+str(output['neutral_site_games']))
    print('BRAIN_ROWS='+json.dumps(output['brain_event_rows'],sort_keys=True))
    print('BRAINS_PENDING='+json.dumps(output['brains_not_attached']))
    print('HISTORICAL_CANDIDATES='+str(len(candidates)))
    print('HISTORICAL_COALITIONS='+str(len(coals)))
    print('MARKET_QUOTE_COUNTS='+json.dumps(dict(quote_counts),sort_keys=True))
    print('RESEARCH_ONLY; NO BET_AUTHORITY; NO_MINER_REBUILD; NO_PRODUCTION_CHANGES')
    return output


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ledger-dir',type=Path,required=True)
    ap.add_argument('--v6-dir',type=Path,required=True)
    ap.add_argument('--source-dir',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--frozen-event-csv',type=Path,action='append',default=[],help='Strict explicit historical FIRE records with canonical CFBD game IDs')
    ap.add_argument('--native-event-csv',type=Path,action='append',default=[],help='Strict explicit FIRE records with Source_Game_ID; never inferred from team strings')
    args=ap.parse_args(argv)
    pairs=[(p,'cfbd') for p in args.frozen_event_csv]+[(p,'source') for p in args.native_event_csv]
    run(args.ledger_dir,args.v6_dir,args.source_dir,args.out_dir,pairs)
if __name__=='__main__':main()
