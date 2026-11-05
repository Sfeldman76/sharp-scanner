#!/usr/bin/env python3
"""V5.5 Stage 1: historical CFBD/football + market source separation.

Produces a source-audited, research-only, point-in-time-status-preserving
historical game/quote bridge. Does NOT rediscover rules, refit models,
change production tables, infer original signals from retrospective flags,
or write a betting certificate. Source-native ID != canonical CFBD ID.

Input: existing BigQuery tables. Output: local CSVs + audit JSON, optionally an
immutable versioned GCS run (--publish). No BigQuery writes or model mutations.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import re
import unicodedata
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT = 'sharplogger'
DATASET = 'sharp_data'
BUCKET = 'sharp-models'
HISTORY_START, HISTORY_END = 2013, 2025
OUTPUT_PREFIX = 'research/ncaaf/v55/source_separated_historical_bridge_v1/runs'
CANONICAL_SOURCES = ('ncaaf_cfbd_archive_research_v1','ncaaf_cfbd_miner_features_v1')
GAME_SOURCE = 'ncaaf_historical_game_side_context'
MARKET_SOURCES = ('scores_with_features','ncaaf_archive_market_provider_quotes_v1')
FROZEN_SOURCE_FILES = ('ncaaf_unified_history_miner_v5.py','ncaaf_research_v2.py',
    'ncaaf_v55_combined_brain.py','ncaaf_v55_stage_combined_brains_v2.py',
    'ncaaf_v55_rebuild_walkforward_evidence.py','ncaaf_v55_unified_authority.py')


def norm(value):
    """Exact-ish spelling normalization only; no speculative team aliases."""
    if value is None:return ''
    raw=unicodedata.normalize('NFKC', str(value)).lower().replace('&', ' and ')
    return re.sub(r'[^a-z0-9]+','',raw)


def string(value):
    return '' if value is None else str(value).strip()


def year_num(value):
    try:
        y=int(value)
    except (ValueError,TypeError):return None
    return y if HISTORY_START<=y<=HISTORY_END else None


def date_key(value):
    if value is None:return ''
    s=string(value)
    return s[:10] if re.fullmatch(r'\d{4}-\d{2}-\d{2}',s[:10]) else ''


def stamp(value):
    if value is None:return None
    if isinstance(value,datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    try:
        dt=datetime.fromisoformat(string(value).replace('Z','+00:00'))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    except ValueError:return None


def finite(value):
    try:
        f=float(value)
        return f if math.isfinite(f) else None
    except (TypeError,ValueError):return None


def choice(cols,*names):
    lookup={k.lower():k for k in cols}
    for name in names:
        if name.lower() in lookup:return lookup[name.lower()]
    return None


def field(col, type_name='STRING'):
    """Identifiers must come from trusted schema; no interpolation of external names."""
    if not col or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',col):
        raise ValueError('INVALID_SCHEMA_COLUMN:'+str(col))
    return f'SAFE_CAST(`{col}` AS {type_name})'


def table_name(table):
    if table not in (GAME_SOURCE,*CANONICAL_SOURCES,*MARKET_SOURCES):
        raise ValueError('UNAPPROVED_TABLE:'+table)
    return f'`{PROJECT}.{DATASET}.{table}`'


def read_table(client, table, roles, *, required=(), where=None, limit=None):
    """Return projected dictionary rows and manifest; absent columns -> NULL.

    Unknown source schemas are held, never guessed. Query is read-only. All
    source/column names come from approved fixed choices or BigQuery schema.
    """
    result={'table':table,'status':'NOT_INSPECTED','mapped':{},'row_count':0}
    try:
        meta=client.get_table(f'{PROJECT}.{DATASET}.{table}')
        columns={f.name:f.field_type for f in meta.schema}
    except Exception as exc:
        return [],dict(result,status='TABLE_NOT_AVAILABLE',exception=type(exc).__name__)
    mapped={role:choice(columns,*opts) for role,opts in roles.items()}
    result['mapped']=mapped
    missing=[role for role in required if mapped.get(role) is None]
    if missing:return [],dict(result,status='REQUIRED_FIELDS_MISSING',missing=missing)
    sel=[]
    for role,col in mapped.items():
        typ='INT64' if role in ('season','is_home') else 'STRING'
        sel.append(f'{field(col,typ)} AS `{role}`' if col else f'CAST(NULL AS {typ}) AS `{role}`')
    sql=f'SELECT {", ".join(sel)} FROM {table_name(table)}'
    filt=[]
    if mapped.get('season'):
        filt.append(f'{field(mapped["season"],"INT64")} BETWEEN {HISTORY_START} AND {HISTORY_END}')
    elif mapped.get('game_date'):
        filt.append(f'SAFE_CAST(`{mapped["game_date"]}` AS DATE) BETWEEN DATE({HISTORY_START},1,1) AND DATE({HISTORY_END},12,31)')
    if mapped.get('sport'):
        filt.append(f"UPPER(TRIM({field(mapped['sport'])})) IN ('NCAAF','COLLEGE FOOTBALL','COLLEGE_FOOTBALL')")
    if filt:sql+=' WHERE '+' AND '.join(filt)
    if limit is not None:
        # Only for local tests; never silently cap production evidence.
        if limit < 1:raise ValueError('INVALID_LIMIT')
        sql+=f' LIMIT {int(limit)}'
    try:
        from google.cloud import bigquery
        config=bigquery.QueryJobConfig(maximum_bytes_billed=10 * 1024**3)
        query=client.query(sql,job_config=config)
        rows=[dict(row.items()) for row in query.result()]
    except Exception as exc:
        return [],dict(result,status='QUERY_FAILED',exception=type(exc).__name__,message=str(exc)[:240])
    return rows,dict(result,status='READ',row_count=len(rows),schema_fingerprint=hashlib.sha256(
        json.dumps(columns,sort_keys=True).encode()).hexdigest())


HISTORICAL_ROLES={
 'season':('Season',),'source_game_id':('Source_Game_ID',),
 'game_date':('Game_Date',),'team':('Team_Norm','Canonical_Team_Norm','Team'),
 'opponent':('Opponent_Norm','Opponent_Canonical_Team_Norm','Opponent'),
 'is_home':('Is_Home',),'sport':('Sport',),
 'score_for':('Team_Score',),'score_against':('Opponent_Score',),
 'open_spread':('Opening_Spread',),'open_total':('Opening_Total',),
 'team_win':('SU_Win',)
}
CFBD_ROLES={
 'season':('Season','season',),'cfbd_game_id':('Game_ID','game_id','Id'),
 'game_date':('Game_Date','game_date','Start_Date','start_date'),
 'team':('Team','team'),'opponent':('Opponent','opponent'),
 'is_home':('Is_Home','is_home'),
 'home':('Home_Team','home_team','home'),'away':('Away_Team','away_team','away')
}
SNAPSHOT_ROLES={
 'sport':('Sport',),'game_date':('feat_Game_Start','Game_Start'),
 'home':('Home_Team_Norm',),'away':('Away_Team_Norm',),
 'market':('Market',),'side':('Outcome',),
 'book':('Bookmaker',),'book_fallback':('Book',),
 'snapshot':('Snapshot_Timestamp',),'line':('Value',),
 'odds':('Odds_Price',),'source_game_key':('Game_Key',)
}
PROVIDER_ROLES={
 'season':('season',),'cfbd_game_id':('game_id',),
 'home':('home_team',),'away':('away_team',),
 'provider':('provider',),'spread':('spread_quoted',),
 'open_spread':('spread_open',),'total':('total_quoted',),
 'open_total':('total_open',),'home_ml':('home_ml_quoted',),
 'away_ml':('away_ml_quoted',),'timing_label':('price_timing',),
 'quote_index':('quote_index',)
}


def canonical_rows(sources):
    """CFBD canonical game sides. Archive and miner-feature duplicates collapse."""
    keyed={}; collision=set(); by_id=defaultdict(set)
    for origin, rows in sources.items():
        for r in rows:
            y=year_num(r.get('season')); gid=string(r.get('cfbd_game_id'))
            date=date_key(r.get('game_date'))
            if not (y and gid and date):continue
            team,opp=norm(r.get('team')),norm(r.get('opponent'))
            if team and opp and team!=opp:
                sides=[(team,opp,r.get('is_home'))]
            else:
                home,away=norm(r.get('home')),norm(r.get('away'))
                sides=[(home,away,'1'),(away,home,'0')] if home and away and home!=away else []
            for a,b,is_home in sides:
                flag=string(is_home)
                home_tag=flag if flag in ('0','1') else ''
                key=(y,gid,a)
                val={'season':y,'cfbd_game_id':gid,'game_date':date,
                     'team_key':a,'opponent_key':b,'is_home':home_tag,
                     'cfbd_origin':origin,'feature_vintage':'RETROSPECTIVE_DERIVATION_NOT_ORIGINAL_CAPTURE'}
                if key in keyed:
                    old=keyed[key]
                    if (old['game_date'],old['opponent_key'])!=(date,b):collision.add(key)
                    # Preserve home/away orientation from the CFBD miner feed
                    # when the broader CFBD archive has the same game-side
                    # without an Is_Home attribute.
                    if not old['is_home'] and home_tag: old['is_home']=home_tag
                    # keep previous source; one original game-side, never count both twice
                else:keyed[key]=val
                by_id[(y,gid)].add((date,a,b))
    for key in collision:keyed.pop(key,None)
    index=defaultdict(set)
    for r in keyed.values():
        idx=(r['season'],r['game_date'],r['team_key'],r['opponent_key'])
        index[idx].add(r['cfbd_game_id'])
    return list(keyed.values()),index,by_id,collision


def historical_crosswalk(historical, cfbd_index):
    """Unique season/date/opponent link, requiring both opposite game sides.

    Never map Source_Game_ID by text similarity, scores, spread or implied odds.
    """
    groups=defaultdict(list)
    for r in historical:
        y=year_num(r.get('season')); sid=string(r.get('source_game_id'))
        date=date_key(r.get('game_date'))
        team,opp=norm(r.get('team')),norm(r.get('opponent'))
        if y and sid and date and team and opp and team!=opp:
            groups[(y,sid)].append((date,team,opp,r))
    out=[]
    for (year,sid),items in sorted(groups.items()):
        pairs={(dt,a,b) for dt,a,b,_ in items}
        date_values={x[0] for x in pairs}
        side_values={x[1] for x in pairs}
        complete=len(side_values)==2 and len(pairs)==2 and len(date_values)==1 and all(
            (dt,b,a) in pairs for dt,a,b in pairs)
        ids=[cfbd_index.get((year,dt,a,b),set()) for dt,a,b in pairs]
        intersection=set.intersection(*ids) if ids else set()
        if not complete:status='HOLD_INCOMPLETE_OR_DUPLICATE_SOURCE_SIDES'
        elif not all(ids):status='HOLD_NO_EXACT_CFBD_MATCH'
        elif len(intersection)!=1:status='HOLD_AMBIGUOUS_CFBD_MATCH'
        else:status='MATCH_UNIQUE_TWO_SIDED'
        out.append({'season':year,'source_game_id':sid,
                    'cfbd_game_id':next(iter(intersection)) if status=='MATCH_UNIQUE_TWO_SIDED' else '',
                    'game_date':next(iter(date_values)) if len(date_values)==1 else '',
                    'team_side_count':len(side_values),'status':status,
                    'evidence':'EXACT_SEASON_DATE_TEAM_OPPONENT_BOTH_SIDES',
                    'original_pregame_vintage_certified':False,'bet_authority':0})
    return out


def historical_game_side_results(historical, crosswalk):
    """Build 2013-25 result labels from existing historical source side-rows.

    No brain flags are reconstructed and an opening spread without original
    quote-time evidence is NOT a certified historic executable price.
    """
    linked={(x['season'],x['source_game_id']):x for x in crosswalk}
    seen={}; clashes=set()
    for r in historical:
        y=year_num(r.get('season'));sid=string(r.get('source_game_id'))
        team,opp=norm(r.get('team')),norm(r.get('opponent'))
        date=date_key(r.get('game_date'))
        if not (y and sid and date and team and opp and team!=opp):continue
        mapping=linked.get((y,sid),{})
        spread=finite(r.get('open_spread'))
        points_for=finite(r.get('score_for'));points_against=finite(r.get('score_against'))
        grade='UNPRICED_OR_UNGRADED'
        if None not in (spread,points_for,points_against):
            margin=points_for+spread-points_against
            grade='WIN' if margin>1e-7 else 'LOSS' if margin< -1e-7 else 'PUSH'
        row={'season':y,'source_game_id':sid,'cfbd_game_id':mapping.get('cfbd_game_id',''),
             'game_date':date,'team_key':team,'opponent_key':opp,
             'is_home':string(r.get('is_home')),'team_score':points_for,
             'opponent_score':points_against,'historical_context_open_spread':spread,
             'historical_context_open_total':finite(r.get('open_total')),
             'retrospective_open_spread_result':grade,
             'context_price_provenance':'UNTIMED_HISTORICAL_OPEN_RESEARCH_ONLY',
             'crosswalk_status':mapping.get('status','HOLD_UNLINKED'),
             'original_brain_signal_provenance_certified':False,
             'original_quote_capture_certified':False,'bet_authority':0}
        key=(y,sid,team)
        if key in seen:
            before=seen[key]
            if any(before[k]!=row[k] for k in ('game_date','opponent_key','team_score','opponent_score','historical_context_open_spread')):
                clashes.add(key)
        else:seen[key]=row
    for key in clashes:
        # Retain no uncertain score/rule result when duplicate source sides disagree.
        x=seen[key]; x['team_score']=None;x['opponent_score']=None
        x['retrospective_open_spread_result']='HOLD_CONFLICTING_SOURCE_ROWS'
        x['crosswalk_status']='HOLD_CONFLICTING_SOURCE_ROWS'
    return list(seen.values()),len(clashes)


def canonical_matchup_keys(cfbd_sides):
    by_game=defaultdict(list)
    for s in cfbd_sides:
        by_game[(s['season'],s['cfbd_game_id'])].append(s)
    idx=defaultdict(set)
    for (season,gid),group in by_game.items():
        if len(group)!=2:continue
        a,b=group
        if not (a['team_key']==b['opponent_key'] and b['team_key']==a['opponent_key']
                and a['game_date']==b['game_date']):continue
        if a['is_home'] in ('0','1') and b['is_home'] in ('0','1'):
            if a['is_home']==b['is_home']:continue
            home=a['team_key'] if a['is_home']=='1' else b['team_key']
            away=b['team_key'] if a['is_home']=='1' else a['team_key']
            idx[(season,a['game_date'],home,away)].add(gid)
    return idx


def price_snapshot_rows(rows, matchup_index):
    """Keep original price row independent of postgame-enriched features.

    Chronology is necessary, but never sufficient, for source-capture proof.
    """
    out=[]; keys=set(); counts=Counter()
    for r in rows:
        observed=stamp(r.get('snapshot'))
        game_start=stamp(r.get('game_date'))
        if game_start is None:counts['MISSING_GAME_START']+=1;continue
        y=year_num(game_start.year)
        if not y:continue
        book=string(r.get('book')) or string(r.get('book_fallback'))
        home,away=norm(r.get('home')),norm(r.get('away'))
        market=string(r.get('market')).lower()
        side=string(r.get('side'))
        line=finite(r.get('line'));odds=finite(r.get('odds'))
        if not (home and away and home!=away and book and side and market and line is not None):
            counts['INCOMPLETE_MARKET_IDENTITY']+=1;continue
        if not observed:status='HOLD_NO_OBSERVATION_TIME'
        elif observed>=game_start:status='HOLD_ON_OR_AFTER_KICKOFF'
        else:status='PREKICK_TIME_SOURCE_CAPTURE_UNATTESTED'
        candidates=matchup_index.get((y,date_key(game_start.isoformat()),home,away),set())
        gid=next(iter(candidates)) if len(candidates)==1 else ''
        if not gid:counts['HOLD_NO_UNIQUE_CFBD_MATCH']+=1
        if status.startswith('HOLD'):counts[status]+=1
        else:counts['PREKICK_TIMESTAMP_ROWS']+=1
        key=(gid,y,market,side,book,observed.isoformat() if observed else '',line,odds)
        if key in keys:counts['DUPLICATE_QUOTES_SUPPRESSED']+=1;continue
        keys.add(key)
        out.append({'season':y,'cfbd_game_id':gid,'game_date':date_key(game_start.isoformat()),
                    'snapshot_utc':observed.isoformat() if observed else '',
                    'kickoff_utc':game_start.isoformat(),'source':'scores_with_features',
                    'market':market,'side':side,'book':book,'line':line,
                    'odds':odds if odds is not None else '',
                    'quote_chronology':status,
                    'game_join':'EXACT_DATE_HOME_AWAY' if gid else 'HOLD_UNMATCHED_OR_AMBIGUOUS',
                    'original_quote_capture_certified':False,
                    'retrospective_brain_flags_ignored':True,'bet_authority':0})
    return out,dict(counts)


def provider_price_rows(rows, canonical_by_id):
    out=[];keys=set();counts=Counter()
    for r in rows:
        y=year_num(r.get('season'));gid=string(r.get('cfbd_game_id'))
        if not y or not gid:continue
        sides=canonical_by_id.get((y,gid),set())
        home,away=norm(r.get('home')),norm(r.get('away'))
        match=(len(sides)==2 and any((a==home and b==away) for dt,a,b in sides)
               and any((a==away and b==home) for dt,a,b in sides))
        if not match:counts['HOLD_ID_OR_TEAMS_NOT_VALIDATED']+=1
        book=string(r.get('provider'))
        if not book:counts['NO_PROVIDER']+=1;continue
        i=string(r.get('quote_index'))
        key=(y,gid,book,i,tuple(string(r.get(x)) for x in ('spread','open_spread','total','open_total','home_ml','away_ml')))
        if key in keys:counts['DUPLICATE_PROVIDER_QUOTES_SUPPRESSED']+=1;continue
        keys.add(key)
        out.append({'season':y,'cfbd_game_id':gid if match else '',
            'provider_game_id':gid,'home':home,'away':away,'provider':book,
            'quote_index':i,'source':'ncaaf_archive_market_provider_quotes_v1',
            'spread_quoted':r.get('spread'),'spread_open':r.get('open_spread'),
            'total_quoted':r.get('total'),'total_open':r.get('open_total'),
            'home_ml_quoted':r.get('home_ml'),'away_ml_quoted':r.get('away_ml'),
            'price_timing_label':string(r.get('timing_label')),
            'game_join':'EXACT_CFBD_ID_AND_TEAMS' if match else 'HOLD_NOT_MATCHED',
            'original_quote_timestamp':'','original_quote_capture_certified':False,
            'historical_price_status':'UNTIMED_DESCRIPTIVE_ONLY',
            'bet_authority':0})
    return out,dict(counts)


def summary_report(manifest, canonical, crosswalk, game_sides, snapshot_quotes, provider_quotes, counts):
    season_coverage={}
    for y in range(HISTORY_START,HISTORY_END+1):
        rows=[r for r in crosswalk if r['season']==y]
        season_coverage[str(y)]={'source_games':len(rows),
            'linked_cfbd_games':sum(r['status']=='MATCH_UNIQUE_TWO_SIDED' for r in rows),
            'snapshot_quote_rows':sum(r['season']==y for r in snapshot_quotes),
            'provider_quote_rows':sum(r['season']==y for r in provider_quotes)}
    out={
        'status':'STAGE1_SOURCE_SEPARATION_BUILT_RESEARCH_ONLY',
        'history':[HISTORY_START,HISTORY_END],
        'architecture':{
           'football':'CFBD_CANONICAL_GAME_IDS_AND_DERIVED_FEATURES',
           'historical_game_context':'NCAAF_HISTORICAL_GAME_SIDE_CONTEXT_SOURCE_NATIVE',
           'market_live':'MOVES_WITH_FEATURES_MERGED_NOT_HISTORICAL_INPUT',
           'market_history':'SCORES_WITH_FEATURES_SNAPSHOTS_PLUS_PROVIDER_ARCHIVE',
           'pt':'EXTERNAL_PREDICTION_TRACKER_EVIDENCE_SEPARATE',
           'market_rich':'EXTERNAL_MARKET_SNAPSHOTS_NOT_CFBD',
        },
        'source_manifest':manifest,
        'coverage_by_season':season_coverage,
        'canonical_cfbd_side_rows':len(canonical),
        'historical_source_games':len(crosswalk),
        'historical_game_side_result_rows':len(game_sides),
        'retrospectively_graded_open_spread_sides':sum(x['retrospective_open_spread_result'] in ('WIN','LOSS','PUSH') for x in game_sides),
        'unique_cfbd_crosswalk_games':sum(x['status']=='MATCH_UNIQUE_TWO_SIDED' for x in crosswalk),
        'crosswalk_holds':dict(Counter(x['status'] for x in crosswalk if x['status']!='MATCH_UNIQUE_TWO_SIDED')),
        'historical_market_snapshot_rows':len(snapshot_quotes),
        'historical_market_provider_rows':len(provider_quotes),
        'market_snapshot_metrics':counts['snapshot'],
        'provider_quote_metrics':counts['provider'],
        'pricing_certified_rows':0,
        'historical_combination_certified':False,
        'certified_combinations_written':0,
        'source_capture_certified':False,
        'bet_authority':0,
        'no_2026_selection':True,
        'new_miner_discovery_executed':False,
        'existing_frozen_miner_library_modified':False,
        'core_refit_executed':False,
        'production_tables_mutated':False,
        'remaining_holds':['ORIGINAL_QUOTE_CAPTURE_PROVENANCE_REQUIRED',
          'SOURCE_NATIVE_TO_CFBD_CROSSWALK_MUST_COVER_OOS_GAMES',
          'FROZEN_INDEPENDENT_BRAIN_FIRINGS_REQUIRE_ORIGINAL_ASOF_PROOF',
          'COMBINED_WALKFORWARD_CERTIFICATION_NOT_YET_EXECUTED']
    }
    return out


def write_csv(path,rows,columns):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def write_outputs(outdir, canonical, crosswalk, game_sides, snapshots, providers, report):
    outdir=Path(outdir)
    if outdir.exists() and any(outdir.iterdir()):raise ValueError('OUTPUT_DIRECTORY_NOT_EMPTY:'+str(outdir))
    outdir.mkdir(parents=True,exist_ok=True)
    items={
       'cfbd_canonical_game_sides.csv':(canonical,['season','cfbd_game_id','game_date','team_key','opponent_key','is_home','cfbd_origin','feature_vintage']),
       'historical_source_cfbd_crosswalk.csv':(crosswalk,['season','source_game_id','cfbd_game_id','game_date','team_side_count','status','evidence','original_pregame_vintage_certified','bet_authority']),
       'historical_game_side_results.csv':(game_sides,['season','source_game_id','cfbd_game_id','game_date','team_key','opponent_key','is_home','team_score','opponent_score','historical_context_open_spread','historical_context_open_total','retrospective_open_spread_result','context_price_provenance','crosswalk_status','original_brain_signal_provenance_certified','original_quote_capture_certified','bet_authority']),
       'historical_snapshot_quotes.csv':(snapshots,['season','cfbd_game_id','game_date','snapshot_utc','kickoff_utc','source','market','side','book','line','odds','quote_chronology','game_join','original_quote_capture_certified','retrospective_brain_flags_ignored','bet_authority']),
       'historical_provider_quotes.csv':(providers,['season','cfbd_game_id','provider_game_id','home','away','provider','quote_index','source','spread_quoted','spread_open','total_quoted','total_open','home_ml_quoted','away_ml_quoted','price_timing_label','game_join','original_quote_timestamp','original_quote_capture_certified','historical_price_status','bet_authority'])
    }
    for name,(rows,cols) in items.items():write_csv(outdir/name,rows,cols)
    report['files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(outdir.glob('*.csv'))}
    (outdir/'stage1_source_separation_audit.json').write_text(json.dumps(report,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')
    return [outdir/name for name in (*items,'stage1_source_separation_audit.json')]


def source_fingerprints(repo):
    repo=Path(repo)
    out={}
    for name in FROZEN_SOURCE_FILES:
        file=repo/name
        out[name]={'present':file.is_file(),
                   'sha256':hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None}
    return out


def publish_immutable(paths,*,bucket=BUCKET,storage_client=None):
    from google.cloud import storage
    client=storage_client or storage.Client()
    prefix=OUTPUT_PREFIX+'/'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:12]
    b=client.bucket(bucket)
    for file in paths:
        blob=b.blob(prefix+'/'+file.name)
        # Never overwrite an earlier evidence publication or current registry.
        blob.upload_from_filename(str(file),if_generation_match=0)
    return 'gs://'+bucket+'/'+prefix


def run(client, outdir, *, limit=None, publish=False, bucket=BUCKET, repo=Path(".")):
    if limit is not None and publish:raise ValueError('CANNOT_PUBLISH_TRUNCATED_EVIDENCE')
    manifest={}; data={}
    specs=[(GAME_SOURCE,HISTORICAL_ROLES,('season','source_game_id','game_date','team','opponent')),
           *((t,CFBD_ROLES,('season','cfbd_game_id','game_date')) for t in CANONICAL_SOURCES),
           ('scores_with_features',SNAPSHOT_ROLES,('game_date','home','away','snapshot','market','side','line')),
           ('ncaaf_archive_market_provider_quotes_v1',PROVIDER_ROLES,('season','cfbd_game_id','provider','home','away'))]
    for t,roles,needed in specs:
        rows,inf=read_table(client,t,roles,required=needed,limit=limit)
        manifest[t]=inf;data[t]=rows
    if manifest[GAME_SOURCE]['status']!='READ':
        raise RuntimeError('HISTORICAL_GAME_CONTEXT_NOT_READ:'+str(manifest[GAME_SOURCE]['status']))
    if not any(manifest[t]['status']=='READ' and data[t] for t in CANONICAL_SOURCES):
        raise RuntimeError('CFBD_CANONICAL_SOURCES_NOT_READ')
    cfbd,index,by_id,collisions=canonical_rows({t:data[t] for t in CANONICAL_SOURCES})
    crosswalk=historical_crosswalk(data[GAME_SOURCE],index)
    game_sides,game_side_conflicts=historical_game_side_results(data[GAME_SOURCE],crosswalk)
    matchups=canonical_matchup_keys(cfbd)
    snapshots,ms=price_snapshot_rows(data['scores_with_features'],matchups)
    providers,mp=provider_price_rows(data['ncaaf_archive_market_provider_quotes_v1'],by_id)
    report=summary_report(manifest,cfbd,crosswalk,game_sides,snapshots,providers,{'snapshot':ms,'provider':mp})
    report['historical_conflicting_game_side_rows']=game_side_conflicts
    report['conflicting_cfbd_game_side_keys']=len(collisions)
    report['frozen_source_fingerprints_read_only']=source_fingerprints(repo)
    report['publication_type']='IMMUTABLE_RESEARCH_ONLY' if publish else 'LOCAL_RESEARCH_ONLY'
    paths=write_outputs(outdir,cfbd,crosswalk,game_sides,snapshots,providers,report)
    if publish:
        uri=publish_immutable(paths,bucket=bucket)
        print('IMMUTABLE_RESEARCH_URI='+uri)
    print('STAGE1_SOURCE_SEPARATION='+str(Path(outdir).resolve()))
    print('CFBD_SIDES='+str(len(cfbd))+' HISTORICAL_SOURCE_GAMES='+str(len(crosswalk)))
    print('CFBD_CROSSWALK_PASS='+str(report['unique_cfbd_crosswalk_games']))
    print('HISTORICAL_SNAPSHOTS='+str(len(snapshots))+' UNTIMED_PROVIDER_QUOTES='+str(len(providers)))
    print('CERTIFIED_BETS=0; NO_PRODUCTION_CHANGES; MINER_NOT_REBUILT')
    return report


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--bucket',default=BUCKET)
    ap.add_argument('--repo',type=Path,default=Path('.'),help='Current project source directory, inspected read-only.')
    ap.add_argument('--limit',type=int,default=None,help='Local diagnostics only; never publish truncated outputs.')
    ap.add_argument('--publish',action='store_true',help='Write new immutable GCS research run, not betting authority.')
    args=ap.parse_args(argv)
    from google.cloud import bigquery
    client=bigquery.Client(project=PROJECT)
    run(client,args.out_dir,limit=args.limit,publish=args.publish,bucket=args.bucket,repo=args.repo)


if __name__=='__main__':main()
