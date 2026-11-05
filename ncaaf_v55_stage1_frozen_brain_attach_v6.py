#!/usr/bin/env python3
"""NCAAF V5.5 Stage1 V6 -- attach immutable historic directional expert occurrences.

No discovery, model refit, write to GCP, live votes, certified EV or bets.
Refuses to turn retrospectively reconstructed occurrences into live authority.
Frozen existing 345 Miner rules and all other brains remain untouched.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,re
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

FAMILIES=('CORE','STAT','SYSTEM_MINER','PATHI','BIG_AL','PREDICTION_TRACKER','MARKET_RICH')
DEFAULT_EXPERT='gs://sharp-models/research/ncaaf/v55/expert_occurrence_snapshots_v1/runs/20261105T012018833244Z_9c0eb9ea4c6d/v143_directional_cache.json'
GAMES='historical_cfbd_game_manifest_v5.csv'; SIDES='historical_cfbd_game_sides_v5.csv'; AUDIT='stage1_canonical_ledger_v5_audit.json'
COLS=('season','cfbd_game_id','source_game_id','game_date','side_cfbd_team_key','side_cfbd_opponent_key','brain','system_id','market','direction','source_grade','historical_split','role_status','source_name','original_observed_at','original_pregame_signal_certified','frozen_before_game_verified','historical_price_asof_certified','authority_eligible','signal_status','bet_authority')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def norm(x):return re.sub(r'[^a-z0-9]+','',str(x or '').lower())
def rows(path):
    with open(path,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def dump(path,rr,cols):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rr)
def yrs(x):
    try:y=int(float(x))
    except (ValueError,TypeError):return None
    return y if 2013<=y<=2025 else None

def verified_v5(root):
    root=Path(root)
    a=json.loads((root/AUDIT).read_text())
    if (a.get('status')!='STAGE1_CANONICAL_SOURCE_LEDGER_V5_RESEARCH_ONLY' or a.get('bet_authority')!=0 or
        a.get('v4_reconciliation',{}).get('status')!='PASS_EXACT_V4_GAME_MEMBERSHIP_REPLAY' or
        a.get('historic_book_quote_capture_certified') is not False or
        a.get('historic_brain_original_asof_certified') is not False):
        raise RuntimeError('INVALID_OR_UNVERIFIED_V5_AUDIT')
    for n in (GAMES,SIDES):
        if a['outputs_sha256'].get(n)!=sha(root/n):raise RuntimeError('V5_OUTPUT_HASH_MISMATCH:'+n)
    games=rows(root/GAMES); sides=rows(root/SIDES)
    if len(games)!=a['cfbd_games_linked_identity_only'] or len(sides)!=a['cfbd_sides_linked_identity_only']:
        raise RuntimeError('V5_GAME_SIDE_COUNTS_MISMATCH')
    gmap={}; smap=defaultdict(list); name_index=defaultdict(set); bynative={}
    for g in games:
        y=yrs(g['season']); gid=g['cfbd_game_id'].strip(); sid=g['source_game_id'].strip()
        if y is None or not gid or not sid or (y,gid) in gmap or (y,sid) in bynative:
            raise RuntimeError('V5_GAME_NOT_UNIQUE')
        gmap[(y,gid)]=g;bynative[(y,sid)]=gid
    for s in sides:
        y=yrs(s['season']);k=(y,s['cfbd_game_id'].strip())
        if y is None or k not in gmap:raise RuntimeError('V5_SIDE_GAME_NOT_FOUND')
        smap[k].append(s)
        name_variants={
            (norm(s['source_team_label']),norm(s['source_opponent_label'])),
            (norm(s['cfbd_team_key']),norm(s['cfbd_opponent_key'])),
        }
        for tm,opp in name_variants:
            if tm and opp and tm!=opp:
                name_index[(y, s['game_date'][:10],tm,opp)].add((y,s['cfbd_game_id'],s['cfbd_team_key']))
    if any(len(v)!=2 for v in smap.values()) or len(smap)!=len(gmap):
        raise RuntimeError('V5_NOT_TWO_SIDED')
    # Every side token used to attach votes must identify at most one result side.
    ambiguous={k for k,v in name_index.items() if len(v)!=1}
    return a,gmap,smap,name_index,ambiguous,bynative

def read_blob(uri):
    if not uri.startswith('gs://'):return Path(uri).read_bytes()
    from google.cloud import storage
    part=uri[5:].split('/',1)
    if len(part)!=2 or not part[1]:raise ValueError('INVALID_EXPERT_GCS_URI')
    return storage.Client().bucket(part[0]).blob(part[1]).download_as_bytes()

def extract_cache(doc):
    """Supports validated directional cache maps only; no speculative recursive parsing."""
    if not isinstance(doc,dict):raise RuntimeError('EXPERT_CACHE_NOT_JSON_OBJECT')
    for field in ('systems','cache','directional_cache','historical_system_cache'):
        if isinstance(doc.get(field),dict) and any(isinstance(v,dict) and 'occurrences' in v for v in doc[field].values()):
            return doc[field],field
    if any(isinstance(v,dict) and 'occurrences' in v for v in doc.values()):return doc,'ROOT'
    raise RuntimeError('EXPERT_CACHE_SCHEMA_UNSUPPORTED top_keys='+repr(list(doc)[:12]))

def attach_expert(cache,gmap,smap,name_index,ambiguous):
    feeds=[]; holds=Counter(); byfamily=Counter(); matched_by_year=Counter(); seen=set(); total=Counter()
    for name,st in sorted(cache.items()):
        if not isinstance(st,dict) or str(st.get('role','')).lower()!='directional':continue
        family={'Pathi':'PATHI','BigAl':'BIG_AL'}.get(str(st.get('family','')))
        if not family:continue
        occ=st.get('occurrences')
        if not isinstance(occ,list):raise RuntimeError('EXPERT_OCCURRENCES_NOT_LIST:'+name)
        total[family]+=len(occ)
        for r in occ:
            if not isinstance(r,dict):holds['HOLD_INVALID_OCCURRENCE']+=1;continue
            y=yrs(r.get('season')); dt=str(r.get('date') or '')[:10]
            tm=norm(r.get('team')); opp=norm(r.get('opponent'))
            if not y or len(dt)!=10 or not tm or not opp:
                holds['HOLD_INCOMPLETE_EXPERT_IDENTITY']+=1;continue
            k=(y,dt,tm,opp)
            if k in ambiguous:holds['HOLD_AMBIGUOUS_EXPERT_IDENTITY']+=1;continue
            target=name_index.get(k,set())
            if len(target)!=1:
                holds['HOLD_EXPERT_NO_EXACT_GAME_SIDE']+=1;continue
            _,gid,team=next(iter(target));g=gmap[(y,gid)]; pair=smap[(y,gid)]
            sides=[s for s in pair if s['cfbd_team_key']==team]
            if len(sides)!=1:raise RuntimeError('EXPERT_TEAM_SIDE_NOT_UNIQUE')
            side=sides[0];unique=(y,gid,family,name,team)
            if unique in seen:holds['EXACT_EXPERT_DUPLICATE_SUPPRESSED']+=1;continue
            seen.add(unique)
            srcgrade=r.get('ats_win')
            grade=('WIN' if srcgrade in (1,1.0,'1','1.0',True) else
                   'LOSS' if srcgrade in (0,0.0,'0','0.0',False) else 'UNKNOWN')
            feeds.append(dict(season=y,cfbd_game_id=gid,source_game_id=g['source_game_id'],game_date=g['game_date'],
               side_cfbd_team_key=team,side_cfbd_opponent_key=side['cfbd_opponent_key'],
               brain=family,system_id=name,market='spreads',direction='PLAY_ON_NAMED_TEAM',source_grade=grade,
               historical_split=g['split'],role_status=g['role_verification_status'],source_name='V143_FROZEN_EXPERT_OCCURRENCE_CACHE',
               original_observed_at='',original_pregame_signal_certified='false',frozen_before_game_verified='false',
               historical_price_asof_certified='false',authority_eligible='false',signal_status='RETROSPECTIVE_GRADED_ONLY',bet_authority=0))
            byfamily[family]+=1;matched_by_year[str(y)]+=1
    return feeds,holds,byfamily,matched_by_year,total

def normalize_external(rows_in,label,gmap,smap,bynative):
    """Ingest source-native or canonical frozen brain EVENT records via explicit strict columns.

    This is a reconstruction staging layer, not a vintage certificate. Do not
    convert arbitrary scored/prediction output into independently certified votes.
    """
    out=[];h=Counter()
    for i,r in enumerate(rows_in):
        y=yrs(r.get('season')); b=str(r.get('brain','')).upper().strip()
        if y is None or b not in FAMILIES:
            h['INVALID_YEAR_OR_BRAIN']+=1;continue
        gid=str(r.get('cfbd_game_id') or '').strip(); sid=str(r.get('source_game_id') or '').strip()
        if gid and sid and bynative.get((y,sid))!=gid:
            h['CONFLICTING_ID_NAMESPACES']+=1;continue
        if not gid:gid=bynative.get((y,sid),'')
        k=(y,gid)
        if k not in gmap:
            h['NO_VERIFIED_CFBD_GAME_MATCH']+=1;continue
        teams=smap[k]
        t=norm(r.get('team_key') or r.get('team') or '')
        side=[s for s in teams if t in (norm(s['source_team_label']),norm(s['cfbd_team_key']))]
        if len(side)!=1:
            h['NO_UNIQUE_SIDE_IDENTITY']+=1;continue
        s=side[0]; g=gmap[k]; sys=str(r.get('system_id') or '').strip()
        market=str(r.get('market') or '').strip().lower()
        if market not in ('spreads','h2h','totals') or not sys:
            h['INVALID_MARKET_OR_SYSTEM_ID']+=1;continue
        direction=str(r.get('direction') or '').strip().upper()
        if market=='totals' and direction not in ('OVER','UNDER'):
            h['INVALID_TOTALS_DIRECTION']+=1;continue
        if market in ('spreads','h2h') and direction not in ('PLAY_ON_NAMED_TEAM','PLAY_ON_TEAM'):
            h['INVALID_SPREAD_H2H_DIRECTION']+=1;continue
        out.append(dict(season=y,cfbd_game_id=gid,source_game_id=g['source_game_id'],game_date=g['game_date'],
            side_cfbd_team_key=s['cfbd_team_key'],side_cfbd_opponent_key=s['cfbd_opponent_key'],
            brain=b,system_id=sys,market=market,direction=direction,source_grade=str(r.get('source_grade','') or 'UNKNOWN'),
            historical_split=g['split'],role_status=g['role_verification_status'],source_name=label,
            original_observed_at=str(r.get('original_observed_at','')),original_pregame_signal_certified='false',
            frozen_before_game_verified='false',historical_price_asof_certified='false',
            authority_eligible='false',signal_status='EXTERNAL_FROZEN_FEED_UNCERTIFIED_RESEARCH_ONLY',bet_authority=0))
    return out,h

def assemble(ledger,cache,optional=None):
    a,gmap,smap,name_index,ambig,bynative=verified_v5(ledger)
    expert,holds,byfam,byyear,total=attach_expert(cache,gmap,smap,name_index,ambig)
    feeds=list(expert); extra=Counter();feed_labels=[]
    for label,raw in optional or []:
        add,bad=normalize_external(raw,label,gmap,smap,bynative)
        feeds+=add;extra.update(bad);feed_labels.append({'source':label,'rows':len(raw),'attached':len(add)})
    feeds.sort(key=lambda r:(r['season'],r['cfbd_game_id'],r['brain'],r['system_id'],r['side_cfbd_team_key']))
    # Each system per side at most one occurrence; repeated source families are
    # independent only with independent provenance, never by counting two systems
    # in one family as two brains.
    dedup={};duplicated=0
    for r in feeds:
        k=(r['season'],r['cfbd_game_id'],r['brain'],r['system_id'],r['market'],r['side_cfbd_team_key'],r['direction'])
        if k in dedup:
            if r!=dedup[k]:raise RuntimeError('CROSS_SOURCE_DUPLICATE_CONFLICT:'+str(k))
            duplicated+=1;continue
        dedup[k]=r
    feeds=list(dedup.values())
    pairs=defaultdict(set); families=defaultdict(set);bymarket=defaultdict(set)
    for r in feeds:
        key=(r['season'],r['cfbd_game_id'],r['market'])
        pairs[key].add(r['side_cfbd_team_key'] if r['market']!='totals' else r['direction'])
        families[key].add(r['brain']);bymarket[key].add((r['brain'],r['side_cfbd_team_key']))
    combos=[]
    for (y,gid,m),fam in sorted(families.items()):
        combos.append({'season':y,'cfbd_game_id':gid,'market':m,'split':gmap[(y,gid)]['split'],
         'brain_families_firing':';'.join(sorted(fam)),'independent_family_count_research_only':len(fam),
         'distinct_signal_directions':len(pairs[(y,gid,m)]),
         'opposite_sides_observed':str(len(pairs[(y,gid,m)])>1).lower(),
         'original_signal_capture_certified':'false','original_quote_capture_certified':'false',
         'combination_certified':'false','bet_authority':0})
    audit={'status':'STAGE1_FROZEN_BRAIN_OCCURRENCES_ATTACHED_RETROSPECTIVE_ONLY',
       'input_v5_sha256':{n:sha(Path(ledger)/n) for n in (AUDIT,GAMES,SIDES)},
       'identity_linked_games':len(gmap),'source_expert_occurrences':dict(total),
       'expert_attached_by_brain':dict(byfam),'expert_attached_by_year':dict(byyear),
       'expert_holds':dict(holds),'external_feeds':feed_labels,'external_holds':dict(extra),
       'other_sources_duplicated':duplicated,'observations_attached':len(feeds),
       'distinct_game_market_firing_groups':len(combos),'game_market_groups_with_opposite_directions':sum(x['opposite_sides_observed']=='true' for x in combos),
       'brain_families_with_attached_rows':sorted({r['brain'] for r in feeds}),
       'brain_families_not_in_input':sorted(set(FAMILIES)-{r['brain'] for r in feeds}),
       'original_signal_vintage_certified':False,'original_book_capture_certified':False,
       'out_of_sample_combination_certified':False,'bet_authority':0,
       '2026_outcomes_used':False,'production_modified':False,'miner_345_rules_unchanged':True}
    return feeds,combos,audit

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ledger-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--expert-uri',default=DEFAULT_EXPERT)
    p.add_argument('--feed-csv',type=Path,action='append',default=[],help='Additional explicitly frozen signals; no automatic vintage certification')
    x=p.parse_args(argv)
    if x.out_dir.exists() and any(x.out_dir.iterdir()):raise RuntimeError('OUTPUT_DIRECTORY_NONEMPTY')
    blob=read_blob(x.expert_uri); doc=json.loads(blob);cache,cache_shape=extract_cache(doc)
    extra=[(str(n),rows(n)) for n in x.feed_csv]
    obs,combos,audit=assemble(x.ledger_dir,cache,extra)
    audit['expert_cache_uri']=x.expert_uri
    audit['expert_cache_sha256']=hashlib.sha256(blob).hexdigest()
    audit['expert_cache_shape']=cache_shape
    audit['generated_at_utc']=datetime.now(timezone.utc).isoformat()
    x.out_dir.mkdir(parents=True,exist_ok=True)
    dump(x.out_dir/'historical_frozen_brain_observations_v6.csv',obs,COLS)
    dump(x.out_dir/'historical_brain_cooccurrence_v6.csv',combos,
         ('season','cfbd_game_id','market','split','brain_families_firing','independent_family_count_research_only',
          'distinct_signal_directions','opposite_sides_observed','original_signal_capture_certified','original_quote_capture_certified','combination_certified','bet_authority'))
    audit['output_sha256']={f.name:sha(f) for f in x.out_dir.glob('*.csv')}
    out=x.out_dir/'stage1_frozen_brain_attach_v6_audit.json';out.write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    print('STAGE1_V6_AUDIT='+str(out))
    print('IDENTITY_LINKED_GAMES='+str(audit['identity_linked_games']))
    print('EXPERT_ATTACHED_BY_BRAIN='+json.dumps(audit['expert_attached_by_brain'],sort_keys=True))
    print('EXPERT_HOLDS='+json.dumps(audit['expert_holds'],sort_keys=True))
    print('COOCCURRENCE_GROUPS='+str(len(combos)))
    print('OTHER_FAMILIES_PENDING='+json.dumps(audit['brain_families_not_in_input']))
    print('RESEARCH_ONLY; BET_AUTHORITY=0; FROZEN_MINER_345_RULES_UNCHANGED')
if __name__=='__main__':main()
