#!/usr/bin/env python3
"""Attach already-published V5.5 CORE, Miner, STAT, PT evidence to Stage 2.

Reads existing immutable GCS objects; never retrains, writes to BigQuery/GCS,
changes production, or certifies an original pregame signal or market price.
Exact game identities + explicit directional events only.  Unknown records go
into row-level holds. Existing 345-rule Miner library is not modified.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

CORE_ROOT='research/ncaaf/v55/ats_primary/replay_runs/'
MINER_ROOT='research/ncaaf/v2/miner_exact_path_shadow/runs/'
COMBINED_ROOT='research/ncaaf/v55/combined_brains_shadow/runs/'
CORE_FILE='retrospective_core_ats_game_edges.csv'
MINER_FILE='retrospective_game_decision_ledger_v5.csv'
FROZEN_FILE='season_forward_frozen_rules_v5.csv'
STAT_FILE='retrospective_replay/rule_evaluations_retrospective.csv'
PT_FILE='retrospective_replay/pt_evidence_retrospective.csv'
BRAINS=('CORE','SYSTEM_MINER','STAT','PT')
OUT_FIELDS=('season','game_id','brain','system_id','market','evaluation_status','direction','team','original_observed_at')

def t(v): return str(v if v is not None else '').strip()
def yes(v): return t(v).lower() in ('true','yes','y','1')
def normalized(v):return re.sub(r'[^a-z0-9]','',t(v).lower())
def readcsv_bytes(b,label):
    try:r=csv.DictReader(io.StringIO(b.decode('utf-8-sig')))
    except UnicodeError as exc:raise RuntimeError('BAD_CSV_ENCODING:'+label) from exc
    if not r.fieldnames:raise RuntimeError('MISSING_CSV_HEADER:'+label)
    return list(r),r.fieldnames

def read_file(path):return readcsv_bytes(Path(path).read_bytes(),str(path))
def writecsv(path,rows,fields):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def sha256(b):return hashlib.sha256(b).hexdigest()
def year(r):
    try:x=int(t(r))
    except ValueError:return None
    return x if 2013<=x<=2025 else None

def game_index(ledger_dir):
    """Only exact canonical ID lookup, or exact Stage 1 verified source ID."""
    root=Path(ledger_dir)
    audit=json.loads((root/'stage1_canonical_ledger_v5_audit.json').read_text())
    if audit.get('stage1_neutral_v5_2',{}).get('status')!='PASS_SOURCE_NEUTRAL_TWO_SIDED_RECONCILIATION' or audit.get('bet_authority')!=0:
        raise RuntimeError('REQUIRE_VALID_V5_2_NEUTRAL_LEDGER')
    manifest,mb=read_file(root/'historical_cfbd_game_manifest_v5.csv')
    sides,sb=read_file(root/'historical_cfbd_game_sides_v5.csv')
    for n in ('historical_cfbd_game_manifest_v5.csv','historical_cfbd_game_sides_v5.csv'):
        if audit.get('outputs_sha256',{}).get(n)!=sha256((root/n).read_bytes()):raise RuntimeError('LEDGER_TAMPER:'+n)
    if len(manifest)!=audit.get('cfbd_games_linked_identity_only') or len(sides)!=2*len(manifest):raise RuntimeError('LEDGER_COUNT_MISMATCH')
    cfbd={};native={};side_map=defaultdict(dict)
    for r in manifest:
        y=year(r.get('season'));gid=t(r.get('cfbd_game_id'));sid=t(r.get('source_game_id'))
        if not y or not gid or not sid or (y,gid) in cfbd or (y,sid) in native:raise RuntimeError('AMBIGUOUS_LEDGER_IDS')
        cfbd[(y,gid)]=r;native[(y,sid)]=gid
    for r in sides:
        key=(year(r.get('season')),t(r.get('cfbd_game_id')))
        s=t(r.get('cfbd_team_key'))
        if key not in cfbd or not s or s in side_map[key]:raise RuntimeError('BAD_LEDGER_SIDE')
        side_map[key][s]=r
    if len(side_map)!=len(cfbd) or any(len(v)!=2 for v in side_map.values()):raise RuntimeError('LEDGER_SIDES_NOT_UNIQUE')
    return cfbd,native,side_map

def side_from_home_away(y,gid,direction,cfbd,sides,home='',away=''):
    """Where possible verify both named schools; otherwise require verified admin role.

    A neutral game may have administrative home/away slots, but these cannot be
    promoted into real venue home/road roles. Explicit TWO name matches are safe.
    """
    key=(y,gid)
    if key not in cfbd:return '', 'NO_EXACT_CFBD_GAME'
    members=sides[key];d=t(direction).upper()
    if d not in ('HOME','AWAY','ROAD'):return '', 'INVALID_DIRECTION'
    home_matches=[];away_matches=[]
    if home and away:
        for team,s in members.items():
            choices={normalized(team),normalized(s.get('source_team_label'))}
            if normalized(home) in choices:home_matches.append(team)
            if normalized(away) in choices:away_matches.append(team)
        if len(home_matches)==1 and len(away_matches)==1 and home_matches[0]!=away_matches[0]:
            return (home_matches[0] if d=='HOME' else away_matches[0]),''
        return '', 'NAMED_HOME_AWAY_NOT_VERIFIED'
    # No named teams: never infer admin-home on source-confirmed neutral games.
    if cfbd[key].get('role_verification_status')!='HOME_AWAY_VERIFIED':return '','NEUTRAL_SITE_UNNAMED_DIRECTION_HELD'
    want='1' if d=='HOME' else '0'
    matches=[team for team,s in members.items() if t(s.get('is_home_verified'))==want]
    return (matches[0],'') if len(matches)==1 else ('','HOME_AWAY_UNVERIFIED')

def named_side(y,gid,r,side_map):
    key=(y,gid)
    if key not in side_map:return '', 'NO_EXACT_CFBD_GAME'
    request=t(r.get('team_key') or r.get('team') or r.get('pick_team') or r.get('side_cfbd_team_key'))
    if not request:return '', 'MISSING_NAMED_SIDE'
    matches=[tm for tm,s in side_map[key].items() if normalized(request) in {normalized(tm),normalized(s.get('source_team_label'))}]
    return (matches[0],'') if len(matches)==1 else ('','UNVERIFIED_NAMED_SIDE')

def fire_record(y,gid,brain,sysid,market,side):
    return dict(season=y,game_id=gid,brain=brain,system_id=sysid,market=market,
                evaluation_status='FIRE',direction='PLAY_ON_TEAM' if market!='totals' else side,
                team='' if market=='totals' else side,original_observed_at='')

def convert_core(rows,cfbd,sides):
    """CORE retrospective picks, never originally-captured picks or bet authority."""
    out=[];holds=Counter();samples=[]
    for r in rows:
        y=year(r.get('season'));gid=t(r.get('game_id'));side=t(r.get('pick_side')).upper()
        if not y or (y,gid) not in cfbd:reason='NO_EXACT_CFBD_GAME'
        elif side not in ('HOME','AWAY'):reason='CORE_HAS_NO_EXPLICIT_PICK'
        else:
            chosen,reason=side_from_home_away(y,gid,side,cfbd,sides,home=t(r.get('home_team')),away=t(r.get('away_team')))
            if not reason:
                eid=t(r.get('model_version') or r.get('model') or 'V55_CORE_RETROSPECTIVE')
                out.append(fire_record(y,gid,'CORE',eid,'spreads',chosen))
                continue
        holds[reason]+=1
        if len(samples)<15:samples.append({'reason':reason,'season':r.get('season'),'game_id':gid,'pick_side':side,'home_team':r.get('home_team'),'away_team':r.get('away_team')})
    return out,holds,samples

def convert_miner(rows,frozen,cfbd,native,sides):
    """Only already-fired, prior-year-selected frozen decisions. No mining."""
    frozen_ids={(year(r.get('test_season')),t(r.get('market')).lower(),t(r.get('rule_id'))) for r in frozen}
    out=[];holds=Counter();samples=[]
    for r in rows:
        y=year(r.get('test_season'));source=t(r.get('source_game_id'));market=t(r.get('market')).lower();rid=t(r.get('rule_id'));direction=t(r.get('rule_decision_side')).upper()
        gid=native.get((y,source),'')
        if not y or not gid:reason='NO_VERIFIED_SOURCE_ID_CROSSWALK'
        elif not yes(r.get('rule_selected_before_test_year')):reason='NO_PRIOR_YEAR_FREEZE'
        elif (y,market,rid) not in frozen_ids:reason='RULE_NOT_IN_PRIOR_YEAR_FROZEN_FILE'
        elif market not in ('spreads','h2h','totals'):reason='UNSUPPORTED_MARKET'
        elif direction in ('OVER','UNDER') and market=='totals':
            event=fire_record(y,source,'SYSTEM_MINER',rid,market,direction)
            out.append(event);continue
        elif direction in ('HOME','AWAY','ROAD') and market!='totals':
            selected,reason=side_from_home_away(y,gid,direction,cfbd,sides)
            if not reason:
                event=fire_record(y,source,'SYSTEM_MINER',rid,market,selected)
                out.append(event);continue
        else:reason='INCOMPATIBLE_MINER_DECISION_SIDE'
        holds[reason]+=1
        if len(samples)<15:samples.append({'reason':reason,'season':r.get('test_season'),'source_game_id':source,'market':market,'rule_id':rid,'decision_side':direction})
    return out,holds,samples

def convert_combined(rows,brain,cfbd,sides,market_if_missing=''):
    """Require exact FIRE and game ID; uncertain market must be explicit opt-in."""
    out=[];holds=Counter();samples=[]
    for r in rows:
        if t(r.get('brain')).upper()!=brain:continue
        y=year(r.get('season'));gid=t(r.get('game_id'));fire=t(r.get('evaluation_status')).upper()
        market=t(r.get('market')).lower() or market_if_missing
        direction=t(r.get('direction')).upper();sysid=t(r.get('system_id'))
        if fire not in ('FIRE','FIRED'):reason='NOT_EXPLICIT_FIRE'
        elif not y or (y,gid) not in cfbd:reason='NO_EXACT_CFBD_GAME'
        elif not sysid:reason='MISSING_SYSTEM_ID'
        elif market not in ('spreads','h2h','totals'):reason='MISSING_OR_UNSUPPORTED_MARKET'
        elif market=='totals' and direction in ('OVER','UNDER'):
            out.append(fire_record(y,gid,brain,sysid,market,direction));continue
        elif market in ('spreads','h2h') and direction in ('HOME','AWAY','ROAD'):
            team,reason=side_from_home_away(y,gid,direction,cfbd,sides,home=t(r.get('home_team_key') or r.get('home_team')),away=t(r.get('away_team_key') or r.get('away_team')))
            if not reason:out.append(fire_record(y,gid,brain,sysid,market,team));continue
        elif market in ('spreads','h2h') and direction in ('PLAY_ON_TEAM','PLAY_ON_NAMED_TEAM','TEAM'):
            team,reason=named_side(y,gid,r,sides)
            if not reason:out.append(fire_record(y,gid,brain,sysid,market,team));continue
        else:reason='UNSUPPORTED_MARKET_DIRECTION'
        holds[reason]+=1
        if len(samples)<15:samples.append({'reason':reason,'season':r.get('season'),'game_id':gid,'direction':direction,'market':market,'system_id':sysid})
    return out,holds,samples

def choose_runs(client,bucket_name,overrides=None):
    """Pin latest COMPLETE independent families; do not mix runs within a family."""
    overrides=overrides or {}
    targets={'CORE':(CORE_ROOT,(CORE_FILE,)),
             'SYSTEM_MINER':(MINER_ROOT,(MINER_FILE,FROZEN_FILE)),
             'STAT_PT':(COMBINED_ROOT,(STAT_FILE,PT_FILE))}
    out={}
    for family,(prefix,files) in targets.items():
        found=defaultdict(set)
        if family in overrides:
            specified=t(overrides[family]).removeprefix('gs://'+bucket_name+'/').strip('/')+'/'
            if not specified.startswith(prefix):raise RuntimeError('OVERRIDE_PREFIX_OUTSIDE_TRUSTED_SOURCE:'+family)
            root=specified
            for fn in files:
                if not client.bucket(bucket_name).blob(root+fn).exists():raise RuntimeError('MISSING_OVERRIDE_ARTIFACT:'+root+fn)
        else:
            for obj in client.list_blobs(bucket_name,prefix=prefix):
                tail=obj.name[len(prefix):]
                if '/' not in tail:continue
                stamp,rest=tail.split('/',1)
                if re.fullmatch(r'\d{8}T\d{12}Z',stamp) and rest in files:found[stamp].add(rest)
            complete=sorted(stamp for stamp,existing in found.items() if all(fn in existing for fn in files))
            if not complete:raise RuntimeError('NO_COMPLETE_RESEARCH_RUN_FOUND:'+family)
            root=prefix+complete[-1]+'/'
        out[family]={fn:'gs://'+bucket_name+'/'+root+fn for fn in files}
    return out

def read_artifacts(bucket,selected):
    entries={};receipts={}
    for family,files in selected.items():
        for fn,uri in files.items():
            key=(family,fn);prefix='gs://'+bucket.name+'/'
            if not uri.startswith(prefix):raise RuntimeError('BAD_GCS_URI')
            obj=bucket.blob(uri[len(prefix):]);b=obj.download_as_bytes();rows,headers=readcsv_bytes(b,uri)
            entries[key]=rows
            receipts[family+':'+fn]={'uri':uri,'rows':len(rows),'header':headers,'sha256':sha256(b),'bytes':len(b),'generation':obj.generation}
    return entries,receipts

def produce(ledger_dir,entries,receipts,out_dir,combined_market=''):
    out=Path(out_dir)
    if out.exists() and any(out.iterdir()):raise RuntimeError('REFUSE_TO_OVERWRITE_NONEMPTY_OUTPUT_DIRECTORY')
    cfbd,native,sides=game_index(ledger_dir)
    all_output={};diagnostics={};samples={}
    for name,(rows,holds,examples) in {
        'CORE':convert_core(entries[('CORE',CORE_FILE)],cfbd,sides),
        'SYSTEM_MINER':convert_miner(entries[('SYSTEM_MINER',MINER_FILE)],entries[('SYSTEM_MINER',FROZEN_FILE)],cfbd,native,sides),
        'STAT':convert_combined(entries[('STAT_PT',STAT_FILE)],'STAT',cfbd,sides,combined_market),
        'PT':convert_combined(entries[('STAT_PT',PT_FILE)],'PT',cfbd,sides,combined_market),
    }.items():
        diagnostics[name]={'attached':len(rows),'held':sum(holds.values()),'hold_reasons':dict(holds)}
        all_output[name]=rows;samples[name]=examples
    canonical=all_output['CORE']+all_output['STAT']+all_output['PT'];native_rows=all_output['SYSTEM_MINER']
    # Do not allow a conflicting duplicate declaration to silently cancel another.
    uniq=set()
    for r in canonical+native_rows:
        k=(r['season'],r['game_id'],r['brain'],r['system_id'],r['market'],r['direction'],r['team'])
        uniq.add(k)
    for k,v in diagnostics.items():v['unique_possible_firings']=len({(r['season'],r['game_id'],r['system_id'],r['market'],r['direction'],r['team']) for r in all_output[k]})
    out.mkdir(parents=True,exist_ok=True)
    writecsv(out/'stage2_existing_cfbd_fires_v3.csv',canonical,OUT_FIELDS)
    writecsv(out/'stage2_existing_source_fires_v3.csv',native_rows,OUT_FIELDS)
    report={'status':'STAGE2_EXISTING_FROZEN_EVIDENCE_ATTACHED_RESEARCH_ONLY',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'canonical_ledger_games':len(cfbd),'families':diagnostics,
        'attachments_considered':['CORE','STAT','SYSTEM_MINER','PT'],
        'already_attached_by_v6':['PATHI','BIG_AL'],
        'market_rich':'NO_INDEPENDENT_HISTORICAL_FIRE_SOURCE_VERIFIED; RAW_PROVIDER_QUOTES_ARE_PRICE_CONTEXT_ONLY',
        'combined_market_fallback':combined_market or 'NONE: REQUIRE_SOURCE_MARKET_COLUMN',
        'source_receipts':receipts,'hold_examples':samples,
        'staged_cfbd_rows':len(canonical),'staged_native_rows':len(native_rows),
        'csv_sha256':{n:sha256((out/n).read_bytes()) for n in ('stage2_existing_cfbd_fires_v3.csv','stage2_existing_source_fires_v3.csv')},
        'frozen_miner_345_library_modified':False,'raw_market_quotes_converted_to_fire':False,
        'original_signal_capture_certified':False,'historical_market_prices_certified':False,
        'bet_authority':0,'production_mutated':False,'bigquery_mutated':False,'gcs_mutated':False}
    (out/'stage2_existing_brain_attach_v3_audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('EXISTING_BRAINS_AUDIT='+str(out/'stage2_existing_brain_attach_v3_audit.json'))
    print('ATTACH_COUNTS='+json.dumps({k:{'attached':v['attached'],'held':v['held']} for k,v in diagnostics.items()},sort_keys=True))
    print('HOLDS_BY_BRAIN='+json.dumps({k:v['hold_reasons'] for k,v in diagnostics.items()},sort_keys=True))
    print('MARKET_RICH_PENDING=TRUE; BET_AUTHORITY=0; MINER_UNCHANGED=TRUE')
    return report

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ledger-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--bucket',default='sharp-models')
    p.add_argument('--project',default='sharplogger')
    p.add_argument('--core-prefix',default='')
    p.add_argument('--miner-prefix',default='')
    p.add_argument('--combined-prefix',default='')
    p.add_argument('--combined-market',choices=('','spreads','h2h','totals'),default='',help='If source does not contain market, explicit research assumption; do not use for certified bets')
    p.add_argument('--run-stage2',action='store_true',help='Run existing Stage2 engine using attached files')
    p.add_argument('--v6-dir',type=Path)
    p.add_argument('--source-dir',type=Path)
    p.add_argument('--stage2-out-dir',type=Path)
    p.add_argument('--stage2-script',type=Path,default=Path('ncaaf_v55_stage2_combined_research.py'))
    a=p.parse_args(argv)
    if a.run_stage2 and not all((a.v6_dir,a.source_dir,a.stage2_out_dir)):
        p.error('--run-stage2 requires --v6-dir --source-dir --stage2-out-dir')
    from google.cloud import storage
    client=storage.Client(project=a.project)
    paths=choose_runs(client,a.bucket,{k:v for k,v in {'CORE':a.core_prefix,'SYSTEM_MINER':a.miner_prefix,'STAT_PT':a.combined_prefix}.items() if v})
    sources,receipts=read_artifacts(client.bucket(a.bucket),paths)
    results=produce(a.ledger_dir,sources,receipts,a.out_dir,a.combined_market)
    if a.run_stage2:
        cmd=[sys.executable,str(a.stage2_script),'--ledger-dir',str(a.ledger_dir),'--v6-dir',str(a.v6_dir),
             '--source-dir',str(a.source_dir),'--out-dir',str(a.stage2_out_dir),
             '--frozen-event-csv',str(a.out_dir/'stage2_existing_cfbd_fires_v3.csv'),
             '--native-event-csv',str(a.out_dir/'stage2_existing_source_fires_v3.csv')]
        print('EXECUTE_STAGE2_WITH_EXISTING_BRAINS=TRUE',flush=True)
        subprocess.run(cmd,check=True)

if __name__=='__main__':main()
