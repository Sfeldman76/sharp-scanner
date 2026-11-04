#!/usr/bin/env python3
"""One deterministic, executable 2013–2025 NCAAF Miner SPREAD/H2H/TOTALS library.

Input is the completed V5 unified next-season freeze, NOT an inherited incumbent
registry. Uses EXACT native atom names from ncaaf_research_v2._extended_atoms.
No live authority is inferred from retrospective fold scores; all rules remain
research-only until a separate matching OOS/point-in-time certificate exists.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import io
import json
import math
import numbers
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = 'research/ncaaf/v55/unified_executable_miner_library_v2'
CURRENT = ROOT + '/current_library.json'
V5_PATH = 'research/ncaaf/v2/miner_exact_path_shadow/runs/'
ALTERNATE_PATH = 'research/ncaaf/v2/unified_full_history_miner/runs/'
PROTOCOL='V55-EXECUTABLE-UNIFIED-FULL-HISTORY-2013-2025-R2'
MARKETS=('spreads','h2h','totals')
REQUIRED_FUTURE={'rule_id','market','direction','condition_atoms','selection_end_season'}
FORBIDDEN_RE=re.compile(r'(?:^|_)(?:POSTGAME|POST_GAME|FINAL_MARGIN|ACTUAL_MARGIN|ATS_RESULT|ATS_OUTCOME|GAME_RESULT|2026)(?:_|$)',re.I)
ATOM_RE=re.compile(r'^[A-Za-z0-9_.:+-]{2,150}$')


def rule_fingerprint(market,direction,atoms):
    """Byte-for-byte compatible with ncaaf_unified_history_miner_v5.fingerprint."""
    return 'UNIFIED-'+hashlib.sha256(json.dumps((str(market).lower(),str(direction).upper(),tuple(sorted(atoms)))).encode()).hexdigest()[:24].upper()


def executable_id(rule_id):
    """New namespace: deterministic alias for the SAME rule, not a legacy ID mapping."""
    assert re.fullmatch(r'UNIFIED-[A-F0-9]{24}',rule_id)
    return 'NCAAF-MECH-U'+rule_id[8:]


def _read_csv(blob):
    if isinstance(blob,bytes):blob=blob.decode('utf-8-sig')
    return list(csv.DictReader(io.StringIO(blob)))


def _int(v):
    try:
        f=float(v)
        if not math.isfinite(f) or f!=int(f):return None
        return int(f)
    except (ValueError,TypeError,OverflowError):return None


def _num(v):
    try:
        f=float(v)
        return f if math.isfinite(f) else None
    except (ValueError,TypeError,OverflowError):return None


def parse_atoms(expression):
    if not isinstance(expression,str) or not expression.strip():raise ValueError('EMPTY_EXPRESSION')
    atoms=[s.strip() for s in expression.split(' AND ')]
    if len(atoms)>4 or not atoms or len(set(atoms))!=len(atoms) or not all(ATOM_RE.fullmatch(a) for a in atoms):
        raise ValueError('INVALID_ATOM_EXPRESSION')
    if any(FORBIDDEN_RE.search(a) for a in atoms):raise ValueError('POSTGAME_OR_FUTURE_ATOM')
    return tuple(sorted(atoms))


def atom_provider_sha256(path):
    """Freeze the real named-atom producer implementation across publication/live."""
    src=Path(path).expanduser().read_text(encoding='utf-8')
    tree=ast.parse(src)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_extended_atoms']
    if len(functions)!=1:raise ValueError('ATOM_PROVIDER_FUNCTION_NOT_UNIQUE')
    code=ast.get_source_segment(src,functions[0])
    return hashlib.sha256(code.encode('utf-8')).hexdigest()


def build_library(future_rows,fold_rows,*, source_uri,source_hashes,audit=None,provider_hash=None):
    """All next-year frozen spread/H2H/totals definitions with deterministic IDs.

    2024/2025 fold results are descriptive, not a certified fixed-rule holdout:
    in V5, the 2025 fold was reselected using 2024 training outcomes.
    """
    if not future_rows: raise ValueError('NO_NEXT_YEAR_RULE_DEFINITIONS')
    if not str(source_uri).startswith('gs://sharp-models/research/ncaaf/v2/') and not str(source_uri).startswith('local://'):
        raise ValueError('UNTRUSTED_RESEARCH_SOURCE')
    audits=audit or {}
    if audits and (audits.get('2026_outcomes_used_for_selection') is not False or
                   audits.get('test_year_outcomes_used_for_selection') is not False):
        raise ValueError('SOURCE_SELECTION_NOT_LEAKAGE_SAFE')
    if audits and sorted(audits.get('historical_seasons',[]))!=list(range(2013,2026)):
        raise ValueError('SOURCE_HISTORY_2013_2025_INCOMPLETE')
    by_id={}; rejected={}; duplicate_count=0
    def reject(reason):rejected[reason]=rejected.get(reason,0)+1
    for row in future_rows:
        try:
            if not REQUIRED_FUTURE.issubset(row):raise ValueError('MISSING_RULE_FIELDS')
            market=str(row['market']).lower().strip()
            if market not in MARKETS:raise ValueError('UNSUPPORTED_MARKET:'+market)
            if str(row['direction']).upper().strip() not in ('PLAY_ON','FADE'):raise ValueError('DIRECTION')
            if _int(row['selection_end_season'])!=2025:raise ValueError('NOT_2025_NEXT_YEAR_FREEZE')
            if str(row.get('catalog_exact_duplicate_id') or '').strip():raise ValueError('LEGACY_CATALOG_DUPLICATE')
            atoms=parse_atoms(row['condition_atoms'])
            rid=rule_fingerprint(market,row['direction'],atoms)
            if row['rule_id']!=rid:raise ValueError('RULE_ID_DEFINITION_MISMATCH')
            if rid in by_id:
                if (by_id[rid]['atoms']!=list(atoms) or by_id[rid]['direction']!=row['direction'].upper()):
                    raise ValueError('RULE_ID_COLLISION')
                duplicate_count+=1;continue
            by_id[rid]={
                'rule_id':rid,'mechanism_id':executable_id(rid), 'market':market,
                'direction':row['direction'].upper(), 'atoms':list(atoms),
                'source_categories':sorted(set((row.get('sources') or '').split('|'))-set([''])),
                'selection_end_season':2025,'first_live_season':2026,
                'train_games':_int(row.get('train_games')),
                'train_shrunken_lift':_num(row.get('train_shrunken_lift')),
                'origin':'UNIFIED_V5_2026_FROZEN_NEXT_YEAR_RULE',
                'legacy_incumbent':False,'rule_definition_executable':True,
                'historical_oos_certified':False,'live_authority_eligible':False,
                'certificate_status':'NOT_CERTIFIED_NEEDS_FIXED_FREEZE_OOS_AND_FEATURE_VINTAGE',
                'bet_authority':0}
        except ValueError as exc:reject(str(exc))
    if not by_id:raise ValueError('NO_VALID_2026_FROZEN_MARKET_RULES:'+json.dumps(rejected))
    folds={}
    for rec in fold_rows:
        rid=str(rec.get('rule_id') or '')
        year=_int(rec.get('test_season'))
        if rid in by_id and year in (2024,2025) and str(rec.get('market')).lower().strip()==by_id[rid]['market']:
            # Never silently treat duplicates within a season as independent.
            key=(rid,year)
            if key in folds:raise ValueError('DUPLICATE_FOLD_RULE_YEAR:'+rid)
            if parse_atoms(rec.get('condition_atoms',''))!=tuple(by_id[rid]['atoms']):
                raise ValueError('FOLD_DEFINITION_CHANGED:'+rid)
            folds[key]={
                'year':year,'train_end_season':_int(rec.get('train_end_season')),
                'wins':_int(rec.get('test_wins')), 'losses':_int(rec.get('test_losses')),
                'pushes_or_ungraded':_int(rec.get('test_pushes_or_ungraded')),
                'unique_games':_int(rec.get('test_unique_games')),
                'selection_uses_prior_seasons_only':_int(rec.get('train_end_season'))==year-1,
                'historical_vintage_certified':str(rec.get('historical_vintage_certified','')).lower()=='true',
            }
    for rid,r in by_id.items():
        data=[folds[(rid,y)] for y in (2024,2025) if (rid,y) in folds]
        r['retrospective_fold_results']=data
        r['appears_in_both_2024_2025_fold_selections']=len(data)==2
        r['independent_2024_2025_fixed_freeze_confirmed']=False
    result={
        'schema':PROTOCOL,'status':'AUTHORITATIVE_EXECUTABLE_RULE_DEFINITIONS_RESEARCH_ONLY',
        'historical_range':'2013-2025','train_end_season':2025,'first_live_season':2026,
        'source_uri':source_uri,'source_sha256':source_hashes,
        'atom_provider_sha256':provider_hash,
        'source_audit':{'historical_seasons':audits.get('historical_seasons'),
                        'audit_present':bool(audits),
                        'test_year_outcomes_used_for_selection':audits.get('test_year_outcomes_used_for_selection'),
                        'present_day_feature_definition_vintage_certified':audits.get('present_day_feature_definition_vintage_certified')},
        'rejected_rows':rejected,'deduplicated_identical_rules':duplicate_count,
        'rule_count':len(by_id),'market_counts':{market:sum(r['market']==market for r in by_id.values()) for market in MARKETS},
        'rules':[by_id[k] for k in sorted(by_id)],
        'authority_status':'RESEARCH_ONLY_NO_LIVE_BET_CERTIFICATION',
        'bet_authority':0,
    }
    return result


def validate_library(library):
    if not isinstance(library,dict) or library.get('schema')!=PROTOCOL:
        raise ValueError('INVALID_LIBRARY_SCHEMA')
    if library.get('bet_authority')!=0 or library.get('authority_status')!='RESEARCH_ONLY_NO_LIVE_BET_CERTIFICATION':
        raise ValueError('UNEXPECTED_AUTHORITY_PROMOTION')
    rules=library.get('rules')
    if not isinstance(rules,list) or len(rules)!=library.get('rule_count') or not rules:
        raise ValueError('BAD_RULE_COUNT')
    if set(library.get('market_counts') or {})!=set(MARKETS):raise ValueError('MARKET_COUNTS_MISSING')
    if sum(library['market_counts'].values())!=len(rules):raise ValueError('MARKET_TOTAL_COUNT_MISMATCH')
    for market in MARKETS:
        if library['market_counts'][market]!=sum(r.get('market')==market for r in rules):
            raise ValueError('MARKET_COUNT_MISMATCH:'+market)
    seen=set()
    for r in rules:
        if r.get('market') not in MARKETS:raise ValueError('UNSUPPORTED_MARKET_IN_LIBRARY')
        a=r.get('atoms')
        if not isinstance(a,list) or not a:raise ValueError('MISSING_ATOMS')
        if rule_fingerprint(r.get('market'),r.get('direction'),a)!=r.get('rule_id'):
            raise ValueError('FINGERPRINT_TAMPER')
        if executable_id(r['rule_id'])!=r.get('mechanism_id') or r['rule_id'] in seen:
            raise ValueError('DUPLICATE_OR_INVALID_ID')
        if r.get('live_authority_eligible') is not False or r.get('legacy_incumbent') is not False:
            raise ValueError('ILLEGAL_AUTHORITY_OR_LEGACY')
        seen.add(r['rule_id'])
    return True


def _strict_mask(v,n):
    if len(v)!=n:raise ValueError('ATOM_ROW_ALIGNMENT')
    out=[]
    for x in v:
        if x is None or type(x).__name__=='NAType':
            out.append(None)
        elif isinstance(x,numbers.Number) or isinstance(x,bool):
            f=float(x)
            if not math.isfinite(f):out.append(None)
            elif f==1:out.append(True)
            elif f==0:out.append(False)
            else:raise ValueError('ATOM_NON_BINARY_VALUE')
        else:raise ValueError('ATOM_NON_BINARY_VALUE')
    return out


def evaluate(library,atoms_by_name,*, rows_count,metadata=None):
    """Shared runtime: evaluate *same exact* named atom masks on any frame.

    Three-valued logic: missing atom or null input => UNKNOWN, never FALSE.
    No historical scores or bet outputs read; no authority granted.
    metadata optionally provides actual pregame source-clock checks per row.
    """
    validate_library(library)
    masks={k:_strict_mask(v,rows_count) for k,v in atoms_by_name.items()}
    if metadata is not None and len(metadata)!=rows_count:raise ValueError('METADATA_ROW_ALIGNMENT')
    results=[];diag={'rule_count':len(library['rules']),'market_counts':library['market_counts'],
                     'rows':rows_count,'fired':0,'unknown':0,
                     'nonfired':0,'pregame_clock_unverified':0,'authority_events':0}
    for rule in library['rules']:
        for i in range(rows_count):
            fields=[masks[a][i] if a in masks else None for a in rule['atoms']]
            state='FALSE' if False in fields else 'UNKNOWN' if None in fields else 'TRUE'
            row_meta=metadata[i] if metadata is not None else None
            if row_meta is not None:
                kickoff=_parse_time(row_meta.get('kickoff_utc'))
                observed=_parse_time(row_meta.get('observed_at_utc'))
                if observed is None or kickoff is None or observed>=kickoff:
                    if state=='TRUE':diag['pregame_clock_unverified']+=1
                    state='UNKNOWN'
            diag['fired' if state=='TRUE' else 'nonfired' if state=='FALSE' else 'unknown']+=1
            if state=='TRUE':
                results.append({'row_index':i,'rule_id':rule['rule_id'],
                                'mechanism_id':rule['mechanism_id'], 'market':rule['market'],
                                'direction':rule['direction'],
                                'target_orientation':('OVER' if rule['direction']=='PLAY_ON' else 'UNDER')
                                    if rule['market']=='totals' else ('SOURCE_TEAM' if rule['direction']=='PLAY_ON' else 'OPPONENT'),
                                'evaluation_state':'FIRED_RESEARCH_ONLY', 'bet_authority':0})
    return results,diag


def _parse_time(v):
    try:
        if not isinstance(v,str):return None
        x=datetime.fromisoformat(v.replace('Z','+00:00'))
        return x.astimezone(timezone.utc) if x.tzinfo else None
    except (ValueError,TypeError,OverflowError):return None


def evaluate_research_frame(library,frame,*,dashboard_module=None,market='spreads'):
    """Evaluate exactly one market on its own source-quote frame.

    This must NOT transpose spread price/side features into totals or H2H.
    The input is already one oriented source row per game, with real kickoff
    and real snapshot times. The same named-atom code is used in history/live.
    """
    import ncaaf_research_v2 as miner
    import numpy as np
    market=str(market).lower().strip()
    if market not in MARKETS:raise ValueError('UNSUPPORTED_MARKET')
    frozen_hash=library.get('atom_provider_sha256')
    if frozen_hash and atom_provider_sha256(miner.__file__)!=frozen_hash:
        raise RuntimeError('LIVE_ATOM_PROVIDER_DEFINITION_DRIFT')
    raw=miner._extended_atoms(frame,dashboard_module,for_live=True,market=market,enforce_support_floor=False)
    atoms={x['name']:np.asarray(x['mask']).tolist() for x in raw}
    metadata=[]
    for _,r in frame.iterrows():
        observed=r.get('Snapshot_Timestamp')
        kickoff=r.get('Game_Start')
        metadata.append({'observed_at_utc':str(observed) if observed is not None else None,
                         'kickoff_utc':str(kickoff) if kickoff is not None else None})
    only={**library,'rules':[r for r in library['rules'] if r['market']==market]}
    only['rule_count']=len(only['rules'])
    only['market_counts']={m:sum(r['market']==m for r in only['rules']) for m in MARKETS}
    if not only['rules']:
        return [],{'status':'NO_FROZEN_RULES_FOR_MARKET','market':market,'fired':0,
                   'unknown':0,'nonfired':0,'authority_events':0,'rule_count':0}
    fired,diag=evaluate(only,atoms,rows_count=len(frame),metadata=metadata)
    diag['market']=market
    return fired,diag


def _extract_v5_audit(doc):
    """Find V5 audit whether embedded in exact-path or standalone publication."""
    if not isinstance(doc,dict):return {}
    if ('historical_seasons' in doc and 'test_year_outcomes_used_for_selection' in doc
            and 'status' in doc):return doc
    for key in ('unified_full_history_miner_v5','audit_v5','audit','miner_exact_path_shadow_audit'):
        if key in doc:
            found=_extract_v5_audit(doc[key])
            if found:return found
    return {}


def _collect_sources(client,bucket,prefix=None):
    b=client.bucket(bucket)
    if prefix:
        if prefix.startswith('gs://'):
            head='gs://'+bucket+'/'
            if not prefix.startswith(head):raise ValueError('WRONG_SOURCE_BUCKET')
            prefix=prefix[len(head):]
        choices=[prefix.rstrip('/')+'/']
    else:
        # The V5 exporter is sometimes included in the exact-path publisher,
        # sometimes published by its independent V5 output writer.
        found=[]
        for root in (V5_PATH,ALTERNATE_PATH):
            blobs=client.list_blobs(bucket,prefix=root)
            found.extend(x.name.rsplit('/',1)[0]+'/' for x in blobs if x.name.endswith('/new_full_history_paths_v5.csv'))
        choices=sorted(set(found),reverse=True)
    for base in choices:
        required=('new_full_history_paths_v5.csv','season_forward_frozen_rules_v5.csv')
        names=[base+x for x in required]
        objs=[b.blob(n) for n in names]
        if not all(v.exists() for v in objs):continue
        raw=[v.download_as_bytes() for v in objs]
        audit={}
        for name in ('audit_v5.json','audit.json'):
            o=b.blob(base+name)
            if o.exists():
                candidate=json.loads(o.download_as_bytes())
                audit=_extract_v5_audit(candidate)
                if 'historical_seasons' in audit:break
        source_hashes={name:hashlib.sha256(content).hexdigest() for name,content in zip(required,raw)}
        return _read_csv(raw[0]),_read_csv(raw[1]), 'gs://'+bucket+'/'+base,source_hashes,audit
    raise FileNotFoundError('NO_COMPLETE_V5_FROZEN_RULE_SOURCE_FOUND')


def _save_json(path,obj):
    p=Path(path).expanduser();p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gcs-prefix',help='Immutable completed full-history V5 run prefix; defaults to latest complete')
    p.add_argument('--bucket',default='sharp-models')
    p.add_argument('--future-csv',help='Local previous-run next-season frozen paths (for offline replay)')
    p.add_argument('--folds-csv',help='Local previous-run frozen fold results (for offline replay)')
    p.add_argument('--output',default='~/ncaaf_v55_unified_executable_miner_library.json')
    p.add_argument('--publish',action='store_true',help='Write immutable GCS research-only library and current pointer')
    p.add_argument('--research-source',default='~/sharp-scanner/ncaaf_research_v2.py',help='Current canonical atom producer source to freeze')
    a=p.parse_args()
    if bool(a.future_csv)!=bool(a.folds_csv):p.error('Local input requires BOTH --future-csv and --folds-csv')
    client=None
    if a.future_csv:
        f=Path(a.future_csv).expanduser().read_bytes();h=Path(a.folds_csv).expanduser().read_bytes()
        future,folds=_read_csv(f),_read_csv(h)
        uri='local://previous-full-history-v5-export'
        hashes={'new_full_history_paths_v5.csv':hashlib.sha256(f).hexdigest(),
                'season_forward_frozen_rules_v5.csv':hashlib.sha256(h).hexdigest()}
        audit={}
    else:
        from google.cloud import storage
        client=storage.Client(project='sharplogger')
        future,folds,uri,hashes,audit=_collect_sources(client,a.bucket,a.gcs_prefix)
    provider_hash=atom_provider_sha256(a.research_source)
    if a.publish and a.future_csv:p.error('Publication requires immutable GCS source, not user-supplied local CSVs')
    if a.publish and (not audit or audit.get('status')!='COMPLETE_RESEARCH_ONLY_NOT_CERTIFIED'):
        raise ValueError('V5_SOURCE_AUDIT_NOT_VERIFIED__CANNOT_PUBLISH_CURRENT_LIBRARY')
    obj=build_library(future,folds,source_uri=uri,source_hashes=hashes,audit=audit,
                      provider_hash=provider_hash)
    validate_library(obj)
    _save_json(a.output,obj)
    print('STATUS='+obj['status'])
    print('SOURCE='+uri)
    print('TOTAL_NEW_EXECUTABLE_MARKET_RULES='+str(obj['rule_count']))
    print('MARKET_COUNTS='+json.dumps(obj['market_counts'],sort_keys=True))
    print('REJECTED='+json.dumps(obj['rejected_rows'],sort_keys=True))
    print('LIBRARY='+str(Path(a.output).expanduser()))
    print('BET_AUTHORITY=0')
    if a.publish:
        if client is None:
            from google.cloud import storage
            client=storage.Client(project='sharplogger')
        b=client.bucket(a.bucket)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        path=ROOT+'/runs/'+stamp+'/library.json'
        raw=json.dumps(obj,sort_keys=True,separators=(',',':')).encode()
        b.blob(path).upload_from_string(raw,content_type='application/json',if_generation_match=0)
        pointer={'schema':PROTOCOL,'status':obj['status'],'latest_uri':'gs://'+a.bucket+'/'+path,
                 'sha256':hashlib.sha256(raw).hexdigest(),'rule_count':obj['rule_count'],
                 'market_counts':obj['market_counts'],'bet_authority':0}
        # Stable canonical CURRENT object is an index to the immutable payload,
        # not a second copy with divergent rule definitions. CAS prevents race.
        current=b.blob(CURRENT)
        generation=None
        try:
            current.reload();generation=current.generation
        except Exception as exc:
            if type(exc).__name__!='NotFound':raise
        current.upload_from_string(json.dumps(pointer,sort_keys=True),content_type='application/json',
                                   if_generation_match=(int(generation) if generation else 0))
        print('PUBLISHED=gs://'+a.bucket+'/'+path)
        print('CURRENT_POINTER=gs://'+a.bucket+'/'+CURRENT)

if __name__=='__main__':main()
