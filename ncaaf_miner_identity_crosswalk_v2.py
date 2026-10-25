"""Strict, reproducible NCAAF Miner nickname/legacy-ID -> CFBD crosswalk.

Retrospective RESEARCH ONLY. Identity is established from a complete two-school
matchup and either the physical CFBD ID or a unique matchup on the same (±1 day)
calendar date. No scores, market prices, fuzzy distances, or outcomes used.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import timedelta
import re
import pandas as pd

VERSION = 'NCAAF_MINER_IDENTITY_CROSSWALK_V2_20261024'
# Only explicit school branding alternates; resolved names must be in the *game*.
ALIASES = {
    'mississippirebels':'olemiss',
    'connecticuthuskies':'uconn',
    'centralfloridaknights':'ucf',
    'texassanantonioroadrunners':'utsa',
    'texaselpasominers':'utep',
    'louisianamonroewarhawks':'ulmonroe',
    'louisianalafayetteragincajuns':'louisiana',
    'southernmississippigoldeneagles':'southernmiss',
    'appalachianstatemountaineers':'appstate',
    'samhoustonstatebearkats':'samhouston',
    'miamihurricanes':'miami',
    'miamiohioredhawks':'miamioh',
    'floridainternationalpanthers':'fiu',
    'northcarolinacharlotte49ers':'charlotte',
    'massachusettsminutemen':'umass',
}

def key(value):
    if pd.isna(value):
        return ''
    return re.sub(r'[^a-z0-9]', '', str(value).lower())

def _school_score(source, school, *, allow_prefix=False, verified_aliases=None):
    a, b = key(source), key(school)
    if not a or not b:
        return 0
    if a == b:
        return 10000 + len(b)
    if ALIASES.get(a) == b or (verified_aliases or {}).get(a) == b:
        return 9000 + len(b)
    # Mascot suffixes are permitted ONLY with independently verified physical
    # game ID or date+pair. A school prefix alone never identifies the game.
    if allow_prefix and len(b) >= 4 and a.startswith(b) and len(a) > len(b):
        return 1000 + len(b)
    return 0

def sides_for_matchup(src, opp, home, away, *, allow_prefix=False, verified_aliases=None):
    """Return unique source/opponent orientation given full two-school identity."""
    home_first = _school_score(src, home, allow_prefix=allow_prefix, verified_aliases=verified_aliases) * _school_score(opp, away, allow_prefix=allow_prefix, verified_aliases=verified_aliases)
    away_first = _school_score(src, away, allow_prefix=allow_prefix, verified_aliases=verified_aliases) * _school_score(opp, home, allow_prefix=allow_prefix, verified_aliases=verified_aliases)
    if home_first == 0 and away_first == 0:
        return None
    if home_first == away_first:
        return None
    return 'HOME' if home_first > away_first else 'AWAY'

def _day(v):
    d = pd.to_datetime(v, utc=True, errors='coerce')
    return d.date() if pd.notna(d) else None

def build_crosswalk(firings: pd.DataFrame, universe: pd.DataFrame):
    """Match unique source-game identities without outcomes or betting lines.

    Returns (crosswalk table, counts). One ambiguous source-game match is always
    rejected, never resolved by which record has a more convenient outcome.
    """
    required = {'season','source_game_id','source_team','source_opponent','game_date_raw'}
    missing = required - set(firings.columns)
    if missing: raise ValueError('FIRINGS_MISSING:'+','.join(sorted(missing)))
    required_u = {'season','game_id','home_team_key','away_team_key'}
    if required_u - set(universe.columns):
        raise ValueError('CFBD_UNIVERSE_MISSING:'+','.join(sorted(required_u-set(universe.columns))))
    u = universe.copy()
    u['season'] = pd.to_numeric(u.season, errors='raise').astype(int)
    u['game_id'] = u.game_id.astype('string').str.strip()
    if u.duplicated(['season','game_id']).any():
        raise ValueError('DUPLICATE_CFBD_ID')
    by_id = {(int(r.season),str(r.game_id)):r for r in u.itertuples(index=False)}
    by_day=defaultdict(list)
    if 'game_date_key' in u.columns:
        for r in u.itertuples(index=False):
            day = _day(r.game_date_key)
            if day:
                for n in (-1,0,1):
                    by_day[(int(r.season),day+timedelta(days=n))].append(r)
    source = firings[list(required)].drop_duplicates().copy()
    source['season'] = pd.to_numeric(source.season,errors='raise').astype(int)
    source['source_game_id'] = source.source_game_id.astype('string').str.strip()
    # Learn exact mascot-bearing aliases using matching numeric CFBD game IDs.
    # Only accept aliases that point to ONE school across all ID-anchored games.
    # For synthetic legacy IDs, never use school-prefix guessing: both schools
    # must be explicitly known or independently anchored in another physical game.
    learned=defaultdict(set)
    for row in source.itertuples(index=False):
        game=by_id.get((int(row.season), str(row.source_game_id)))
        if game is None: continue
        side=sides_for_matchup(row.source_team,row.source_opponent,game.home_team_key,game.away_team_key,allow_prefix=True)
        if side=='HOME':
            learned[key(row.source_team)].add(key(game.home_team_key))
            learned[key(row.source_opponent)].add(key(game.away_team_key))
        elif side=='AWAY':
            learned[key(row.source_team)].add(key(game.away_team_key))
            learned[key(row.source_opponent)].add(key(game.home_team_key))
    verified_aliases={s:next(iter(targets)) for s,targets in learned.items() if len(targets)==1}
    candidates=[]
    for row in source.itertuples(index=False):
        yr = int(row.season)
        sid = str(row.source_game_id)
        raw_day = _day(row.game_date_raw)
        physical = by_id.get((yr,sid))
        route='VERIFIED_PHYSICAL_GAME_ID' if physical is not None else 'UNIQUE_DATE_TWO_SCHOOLS'
        if physical is not None:
            options=[physical]
        elif raw_day is None:
            options=[]
            route='SOURCE_DATE_MISSING'
        elif 'game_date_key' not in u:
            options=[]
            route='UNIVERSE_DATE_MISSING'
        else:
            # date bucket includes CFBD games from +/-1 day
            options=by_day.get((yr,raw_day),[])
        hits=[]
        for g in options:
            side=sides_for_matchup(row.source_team,row.source_opponent,g.home_team_key,g.away_team_key,
                    allow_prefix=physical is not None, verified_aliases=verified_aliases)
            if side:
                hits.append((str(g.game_id),side))
        hits=list(dict.fromkeys(hits))
        if len(hits)==1:
            gid, side=hits[0]
            candidates.append((yr,sid,row.source_team,row.source_opponent,row.game_date_raw,gid,side,route,'MATCHED'))
        else:
            status = ('SOURCE_TEAMS_DO_NOT_MATCH_CFBD_GAME' if physical is not None else
                      'AMBIGUOUS_DATE_PAIR' if len(hits)>1 else route if not options else 'DATE_PAIR_NOT_RESOLVED')
            candidates.append((yr,sid,row.source_team,row.source_opponent,row.game_date_raw,'','',route,status))
    cols=['season','source_game_id','source_team','source_opponent','game_date_raw','game_id','source_side','identity_route','identity_status']
    cross=pd.DataFrame(candidates,columns=cols)
    # If a source ID maps to >1 physical CFBD game, suppress *all* source rows.
    unique=(cross[cross.identity_status.eq('MATCHED')].groupby(['season','source_game_id']).game_id.nunique())
    conflicted=set(unique[unique>1].index)
    if conflicted:
        mask=cross.apply(lambda r:(r.season,r.source_game_id) in conflicted,axis=1)
        cross.loc[mask,['game_id','source_side']]=''
        cross.loc[mask,'identity_status']='SOURCE_ID_MAPS_MULTIPLE_GAMES'
    # A source-game ID may appear on both school-side rows, with independently
    # reversed orientations. Source-team-dependent orientation is retained.
    n=Counter(cross.identity_status)
    routes=Counter(cross[cross.identity_status.eq('MATCHED')].identity_route)
    return cross,{'version':VERSION,'source_identity_rows':len(cross),
                  'matched_source_identity_rows':int(cross.identity_status.eq('MATCHED').sum()),
                  'status_counts':dict(n),'matched_route_counts':dict(routes),
                  'numeric_id_anchored_name_aliases':len(verified_aliases),
                  'ambiguous_alias_names_excluded':sum(1 for targets in learned.values() if len(targets)>1),
                  'fuzzy_matches':0,'score_or_market_used_for_match':False,
                  'promotion_authority':0}

def attach_verified_identity(firings, universe):
    """Return all firing rows with match status, CFBD id and target side."""
    cross,audit=build_crosswalk(firings,universe)
    f=firings.copy()
    on=['season','source_game_id','source_team','source_opponent','game_date_raw']
    f['season']=pd.to_numeric(f.season,errors='raise').astype(int)
    f['source_game_id']=f.source_game_id.astype('string').str.strip()
    f=f.merge(cross,on=on,how='left',validate='many_to_one')
    # If the requested target is the source-team then it is the source side;
    # if opponent, invert. Never infer from FADE/PLAY_ON without confirming it.
    target=f.target_team.map(key)
    src=f.source_team.map(key)
    opp=f.source_opponent.map(key)
    direction=[]
    for matched,src_side,t,s,o in zip(f.identity_status.eq('MATCHED'),f.source_side,target,src,opp):
        if not matched or not src_side or t not in (s,o) or s==o:
            direction.append('UNKNOWN_TARGET' if matched else 'UNKNOWN_IDENTITY')
        elif t==s:
            direction.append(src_side)
        else:
            direction.append('AWAY' if src_side=='HOME' else 'HOME')
    f['direction_verified']=direction
    audit['total_firing_rows']=len(f)
    audit['verified_firing_rows']=int(f.direction_verified.isin(['HOME','AWAY']).sum())
    audit['unmatched_firing_rows']=len(f)-audit['verified_firing_rows']
    return f,cross,audit
