#!/usr/bin/env python3
"""V5.5 Stage 1 research-only identity repair using canonical warehouse name fields.

Reads saved Stage 1 CSVs plus narrow, read-only BigQuery identity projections.
No historical model, miner, betting authority, or BigQuery table is modified.
"""
from __future__ import annotations
import argparse,csv,json,hashlib,re,unicodedata
from collections import defaultdict,Counter
from datetime import date,datetime,timezone
from pathlib import Path

PROJECT='sharplogger';DATASET='sharp_data'
FILES=('cfbd_canonical_game_sides.csv','historical_source_cfbd_crosswalk.csv',
       'historical_game_side_results.csv','historical_snapshot_quotes.csv',
       'historical_provider_quotes.csv','stage1_source_separation_audit.json')
TABLES=('ncaaf_historical_game_side_context','ncaaf_cfbd_archive_research_v1',
        'ncaaf_cfbd_miner_features_v1')
SOURCE_FIELDS=('Season','Source_Game_ID','Game_Date','Team','Opponent',
  'Source_Team_Name','Source_Opponent_Name','Team_Norm','Opponent_Norm',
  'Canonical_Team_Norm','Opponent_Canonical_Team_Norm','Canonical_Team_Name',
  'Opponent_Canonical_Team_Name')
CFBD_FIELDS=('Season','Game_ID','Game_Date','Team','Opponent','Team_Key',
  'Opponent_Key','Home_Team','Away_Team')
PAIRS_SOURCE=(('Team','Opponent'),('Source_Team_Name','Source_Opponent_Name'),
  ('Team_Norm','Opponent_Norm'),('Canonical_Team_Norm','Opponent_Canonical_Team_Norm'),
  ('Canonical_Team_Name','Opponent_Canonical_Team_Name'))
PAIRS_CFBD=(('Team_Key','Opponent_Key'),('Team','Opponent'))
PAIRS_BOTH=(('Home_Team','Away_Team'),('home_team','away_team'))
ALIASES={
'alabamast':'alabamastate','arizonast':'arizonastate','arkansasst':'arkansasstate',
'boisest':'boisestate','coloradost':'coloradostate','floridast':'floridastate',
'fresnost':'fresnostate','georgiast':'georgiastate','iowast':'iowastate',
'kansasst':'kansasstate','kentst':'kentstate','michiganst':'michiganstate',
'mississippist':'mississippistate','newmexicost':'newmexicostate',
'oklahomast':'oklahomastate','oregonst':'oregonstate','pennst':'pennstate',
'sandiegost':'sandiegostate','sanjosest':'sanjosestate','utahst':'utahstate',
'washingtonst':'washingtonstate','westernkentuckyst':'westernkentuckystate',
'olemiss':'mississippi','lsu':'louisianastate','appstate':'appalachianstate',
'ncstate':'northcarolinastate','unc':'northcarolina','ucf':'centralflorida',
'usf':'southflorida','utsa':'texassanantonio','utep':'texaselpaso',
'southernmiss':'southernmississippi','miamioh':'miamiohio'}

def clean(x):
    return re.sub('[^a-z0-9]+','',unicodedata.normalize('NFKC',str(x or '')).lower().replace('&',' and '))

def school(x):
    k=clean(x)
    return ALIASES.get(k,k)

def season(x):
    try:
        y=int(x);return y if 2013<=y<=2025 else None
    except (TypeError,ValueError):return None

def dateof(x):
    try:return date.fromisoformat(str(x)[:10])
    except (ValueError,TypeError):return None

def readcsv(p):
    with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def writecsv(p,rs,fs):
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fs,extrasaction='ignore');w.writeheader();w.writerows(rs)

def base_games(rows,idkey,*,with_home=False):
    """Restrict candidate universe to verified two-sided games from saved exports."""
    g=defaultdict(list)
    for r in rows:
        y=season(r.get('season'));gid=str(r.get(idkey) or '').strip()
        if y and gid:g[(y,gid)].append(r)
    out={};bad=Counter()
    for key,items in g.items():
        if len(items)!=2:bad['NOT_TWO_SIDES']+=1;continue
        a,b=items
        a1,a2=school(a.get('team_key')),school(a.get('opponent_key'))
        b1,b2=school(b.get('team_key')),school(b.get('opponent_key'))
        d1,d2=dateof(a.get('game_date')),dateof(b.get('game_date'))
        if not (d1 and d1==d2 and a1 and a2 and a1==b2 and a2==b1 and a1!=a2):
            bad['NONRECIPROCAL_OR_BAD_DATE']+=1;continue
        if with_home:
            h1,h2=str(a.get('is_home','')).strip(),str(b.get('is_home','')).strip()
            if h1 in ('0','1') and h2 in ('0','1') and h1==h2:
                bad['HOME_FLAGS_CONFLICT']+=1;continue
        out[key]={'date':d1,'base_names':(a1,a2),'pairs':{'CSV_BASE':frozenset((a1,a2))}}
    return out,dict(bad)

def variants(rows, *, idkey, pairs, allow_home_away=False):
    """Only take reciprocal per-side pairs. Home/away require two source records.

    Each variant comes from an actual named source field. No fuzzy matching.
    """
    sightings=defaultdict(set)
    both=defaultdict(set)
    for r in rows:
        y=season(r.get('Season')); gid=str(r.get(idkey) or '').strip()
        if not y or not gid:continue
        key=(y,gid)
        for a,b in pairs:
            av,bv=school(r.get(a)),school(r.get(b))
            if av and bv and av!=bv:sightings[(key,f'{a}/{b}',frozenset((av,bv)))].add((av,bv))
        if allow_home_away:
            for a,b in PAIRS_BOTH:
                av,bv=school(r.get(a)),school(r.get(b))
                if av and bv and av!=bv:both[(key,f'{a}/{b}',frozenset((av,bv)))].add((av,bv))
    out=defaultdict(dict)
    for (key,label,pair),directions in sightings.items():
        if len(directions)==2:
            out[key][label]=pair
    # Canonical CFBD archive may be one row/game with explicit home & away,
    # but the game must independently pass the two-sided canonical CSV check.
    for (key,label,pair),directions in both.items():
        if len(pair)==2:out[key][label]=pair
    return out

def read_bq_identity(client,table):
    """Read narrow warehouse source-identity fields with case-safe aliases."""
    schema={f.name.lower():f.name for f in client.get_table(f'{PROJECT}.{DATASET}.{table}').schema}
    names=SOURCE_FIELDS if table=='ncaaf_historical_game_side_context' else CFBD_FIELDS
    aliases={'Game_Date':('Game_Date','Start_Date','start_date')}
    selected={}
    for n in names:
        for candidate in aliases.get(n,(n,)):
            if candidate.lower() in schema:
                selected[n]=schema[candidate.lower()];break
    if 'Season' not in selected:
        raise RuntimeError('ID_FIELDS_MISSING:'+table+':Season')
    if table=='ncaaf_historical_game_side_context' and 'Source_Game_ID' not in selected:
        raise RuntimeError('ID_FIELDS_MISSING:'+table+':Source_Game_ID')
    if table!='ncaaf_historical_game_side_context' and 'Game_ID' not in selected:
        raise RuntimeError('ID_FIELDS_MISSING:'+table+':Game_ID')
    cols=', '.join(f'`{v}` AS `{k}`' for k,v in selected.items())
    sql=(f'SELECT {cols} FROM `{PROJECT}.{DATASET}.{table}` '
         f'WHERE SAFE_CAST(`{selected["Season"]}` AS INT64) BETWEEN 2013 AND 2025')
    if table=='ncaaf_historical_game_side_context' and 'sport' in schema:
        sql+=(f" AND UPPER(TRIM(`{schema['sport']}`)) IN "
              "('NCAAF','COLLEGE FOOTBALL','COLLEGE_FOOTBALL')")
    from google.cloud import bigquery
    rows=[dict(r.items()) for r in client.query(sql,job_config=bigquery.QueryJobConfig(
        maximum_bytes_billed=10*1024**3)).result()]
    return rows,{'table':table,'fields':list(selected),'row_count':len(rows),'read_only':True}

def attach_variants(base,additions):
    for k,v in additions.items():
        if k in base:base[k]['pairs'].update(v)

def reconcile(cfbd,src):
    """Unique two-sided season/school match, exact date or UTC/local 1-day shift."""
    index=defaultdict(set); names_by_year=defaultdict(set)
    for key,g in cfbd.items():
        for pair in set(g['pairs'].values()):
            index[(key[0],pair)].add(key)
            names_by_year[key[0]].update(pair)
    results=[];stats=Counter();diag=[];match_basis=Counter();yearstats=defaultdict(Counter)
    for key,g in sorted(src.items()):
        y,sid=key;candidates=defaultdict(set);match_basis_by_candidate=defaultdict(set)
        any_pair=False;any_date_shift=False
        for label,pair in g['pairs'].items():
            candidates_for_pair=index.get((y,pair),())
            if candidates_for_pair:any_pair=True
            for ckey in candidates_for_pair:
                delta=(cfbd[ckey]['date']-g['date']).days
                if abs(delta)<=1:
                    candidates[ckey].add(delta)
                    match_basis_by_candidate[ckey].add(label)
                else:any_date_shift=True
        if not candidates:
            status='HOLD_TEAM_PAIR_DATE_MISMATCH' if any_pair else 'HOLD_NO_TEAM_PAIR_MATCH'
            accepted=None
        elif len(candidates)!=1:
            status='HOLD_AMBIGUOUS_CFBD_MATCH';accepted=None
        else:
            accepted=next(iter(candidates))
            status='MATCH_UNIQUE_WAREHOUSE_NAME_VARIANT'
        stats[status]+=1;yearstats[str(y)][status]+=1
        yearstats[str(y)]['source_games']+=1
        yearstats[str(y)]['any_school_overlap']+=int(bool(names_by_year[y]&set().union(*g['pairs'].values())))
        basis='|'.join(sorted(match_basis_by_candidate.get(accepted,()))) if accepted else ''
        if accepted:match_basis[basis]+=1
        results.append({'season':y,'source_game_id':sid,'cfbd_game_id':accepted[1] if accepted else '',
            'game_date':g['date'].isoformat(),'status':status,'match_evidence':basis,
            'date_shift_days':min(candidates[accepted],key=abs) if accepted else '',
            'original_pregame_vintage_certified':False,'bet_authority':0})
        if len(diag)<30 and (not accepted) and len(diag)<15 or (accepted and len(diag)<30):
            if len(diag)<30:
                diag.append({'season':y,'source_id':sid,'source_date':str(g['date']),
                  'source_name_variants':{label:sorted(pair) for label,pair in list(g['pairs'].items())[:8]},
                  'candidate_count':len(candidates),'reason':status,
                  'some_school_overlap':bool(names_by_year[y]&set().union(*g['pairs'].values()))})
    # If two native IDs map same CFBD ID, fail both, not just the later one.
    ownership=defaultdict(list)
    for i,row in enumerate(results):
        if row['cfbd_game_id']:ownership[(row['season'],row['cfbd_game_id'])].append(i)
    for ix in ownership.values():
        if len(ix)>1:
            for i in ix:
                stats[results[i]['status']]-=1
                results[i].update(cfbd_game_id='',status='HOLD_CFBD_GAME_REUSED',match_evidence='',date_shift_days='')
                stats['HOLD_CFBD_GAME_REUSED']+=1
    match_basis=Counter(r['match_evidence'] for r in results if r['cfbd_game_id'])
    audit={'status_counts':{k:v for k,v in stats.items() if v},
      'matched_games':sum(bool(r['cfbd_game_id']) for r in results),
      'source_games_evaluated':len(results),'cfbd_games_available':len(cfbd),
      'match_evidence_fields':dict(match_basis),'coverage_by_season':dict(yearstats),
      'diagnostic_samples':diag,'bet_authority':0,'original_capture_certified':False,
      'source_game_namespaces_not_assumed_equal':True,'scores_not_used_for_matching':True}
    return results,audit

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    ns=p.parse_args(argv)
    if ns.out_dir.exists() and any(ns.out_dir.iterdir()):
        raise SystemExit('REFUSE_TO_OVERWRITE_NONEMPTY_OUTPUT_DIRECTORY')
    for f in FILES:
        if not (ns.source_dir/f).is_file():raise SystemExit('MISSING_BRIDGE_FILE:'+f)
    from google.cloud import bigquery
    client=bigquery.Client(project=PROJECT)
    cfbd,cbad=base_games(readcsv(ns.source_dir/FILES[0]),'cfbd_game_id',with_home=True)
    src,sbad=base_games(readcsv(ns.source_dir/FILES[2]),'source_game_id')
    manifests=[];cfbd_extra=defaultdict(dict);src_extra=defaultdict(dict)
    for table in TABLES:
        try:
            rows,manifest=read_bq_identity(client,table)
            manifests.append(manifest)
        except Exception as exc:
            manifests.append({'table':table,'status':'QUERY_ERROR','error':str(exc)[:400]})
            continue
        if table=='ncaaf_historical_game_side_context':
            src_extra.update(variants(rows,idkey='Source_Game_ID',pairs=PAIRS_SOURCE))
        else:
            additional=variants(rows,idkey='Game_ID',pairs=PAIRS_CFBD,allow_home_away=True)
            for key,v in additional.items():cfbd_extra[key].update(v)
    attach_variants(cfbd,cfbd_extra);attach_variants(src,src_extra)
    result,audit=reconcile(cfbd,src)
    audit['name_variant_coverage_by_source']={
        'cfbd_extended_games':sum(bool(k in cfbd_extra) for k in cfbd),
        'source_extended_games':sum(bool(k in src_extra) for k in src),
        'cfbd_variant_type_counts':dict(Counter(label for g in cfbd.values() for label in g['pairs'])),
        'source_variant_type_counts':dict(Counter(label for g in src.values() for label in g['pairs']))}
    audit['canonical_name_samples']={str(year):[
        {'game_id':key[1],'date':str(g['date']),
         'name_variants':{label:sorted(pair) for label,pair in list(g['pairs'].items())[:6]}}
        for key,g in sorted(cfbd.items()) if key[0]==year][:4]
        for year in (2013,2018,2022,2025)}
    audit.update(source_tables=manifests,cfbd_invalid_sides=cbad,
      historical_invalid_sides=sbad,generated_at_utc=datetime.now(timezone.utc).isoformat(),
      status='STAGE1_CROSSWALK_V3_RESEARCH_ONLY',
      missing_new_identity_fields=all(x.get('status')=='QUERY_ERROR' for x in manifests),
      quote_capture_certified=False,brain_signal_capture_certified=False,
      certified_bet_rows=0,db_mutated=False)
    ns.out_dir.mkdir(parents=True,exist_ok=True)
    out=ns.out_dir/'historical_source_cfbd_crosswalk_v3.csv'
    writecsv(out,result,['season','source_game_id','cfbd_game_id','game_date','status',
      'match_evidence','date_shift_days','original_pregame_vintage_certified','bet_authority'])
    # Leave the historical snapshot/provider outputs untouched; this is only identity mapping.
    report=ns.out_dir/'stage1_crosswalk_v3_audit.json'
    audit['source_manifest_sha256']=hashlib.sha256((ns.source_dir/FILES[-1]).read_bytes()).hexdigest()
    audit['output_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
    report.write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    print('STAGE1_CROSSWALK_V3='+str(report))
    print('CFBD_GAMES='+str(len(cfbd))+' SOURCE_GAMES='+str(len(src)))
    print('MATCHED_GAMES='+str(audit['matched_games']))
    print('BY_REASON='+json.dumps(audit['status_counts'],sort_keys=True))
    print('SOURCE_IDENTITY_FIELDS='+json.dumps(manifests,sort_keys=True))
    if audit['matched_games']==0:
        print('ZERO_MATCHES_EXAMPLES='+json.dumps(audit['diagnostic_samples'][:3],sort_keys=True))
    print('RESEARCH_ONLY; BET_AUTHORITY=0; FROZEN_MINER_UNCHANGED')
    return audit

if __name__=='__main__':main()
