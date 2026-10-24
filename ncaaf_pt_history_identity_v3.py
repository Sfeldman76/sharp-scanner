"""Read-only, deterministic historical Prediction Tracker -> CFBD identity matching.

PT annual CSVs often lack a game-date column. Match undated games ONLY when
both teams and season uniquely identify one physical CFBD game. Historical
rematches remain UNKNOWN; do not use game scores, market lines, or fuzzy scores.
This is retroactive identity matching, never contemporaneous prediction proof.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import pandas as pd

from ncaaf_replay_identity_v2 import make_aliases, norm

# Existing legacy PT identity overrides, expressed in short CFBD school names.
# Destination must resolve to an actually observed school in that season.
# No generic 'Miami', 'Georgia', 'Texas' or similar ambiguous abbreviation.
KNOWN_PT_SCHOOL_ALIASES = {
    'miamiflorida':'miami', 'miamifl':'miami',
    'miamiohio':'miamioh', 'miamioh':'miamioh',
    'mississippi':'olemiss', 'centralflorida':'ucf',
    'texassanantonio':'utsa', 'floridaintl':'floridainternational',
    'troystate':'troy', 'louisianalafayette':'louisiana',
    'appalachianstate':'appstate', 'southernmiss':'southernmississippi',
    # Actual 2022-25 PT feeds use the left spellings for CFBD schools on
    # the right. The target must be independently present in CFBD that year.
    'connecticut':'uconn', 'kent':'kentstate',
    # 2023 PT calls the school Sam Houston State; the CFBD FBS identifier is Sam Houston.
    # Compare only against schools actually present in that season.
    'samhoustonstate':'samhouston',
    'louisianamonroe':'ulmonroe', 'louisianamonroewarhawks':'ulmonroe',
    'northernorthernillinois':'northernillinois',
    'southernmississippiissippi':'southernmississippi',
}


def _value_set(values, alias_map):
    keys=set()
    for value in values:
        v=norm(value)
        if not v:
            continue
        keys.add(v)
        if v in alias_map: keys.add(alias_map[v])
        if v in KNOWN_PT_SCHOOL_ALIASES: keys.add(KNOWN_PT_SCHOOL_ALIASES[v])
        # Allow aliases to compare through their (unique) canonical destination.
        for item in tuple(keys):
            if item in alias_map: keys.add(alias_map[item])
    return keys


def match_pt_to_cfbd(games, pt, aliases=None):
    """Return one-to-one PT row -> CFBD game mapping and exhaustive audit.

    Identifies CFBD games using physical matchup and season; source dates are
    OPTIONAL only for globally unique season + ordered-home-away matchups.
    Any source/game duplicate or ambiguous pair is excluded, never resolved by
    scores, predicted lines, or fuzzy nearest-name guessing.
    """
    if pt.empty:
        return pd.DataFrame(columns=['_pt_row_id','season','game_id','identity_method','identity_orientation']), {
            'pt_source_rows':0,'matched_games':0,'reasons':{},'source_date_present':0,
            'ambiguous_pairs_forced':0,'unmatched_examples':[]}
    required_games={'season','game_id','home_team_key','away_team_key','game_date_key'}
    required_pt={'season','home_key','away_key','game_date'}
    if not required_games.issubset(games) or not required_pt.issubset(pt):
        raise ValueError('PT_CFBD_IDENTITY_SCHEMA_MISSING')
    alias_map=make_aliases(aliases)
    g=games.reset_index(drop=True)
    p=pt.reset_index(drop=True)
    p_seasons=pd.to_numeric(p.season,errors='coerce')
    g_seasons=pd.to_numeric(g.season,errors='coerce')
    home_index=defaultdict(set)
    away_index=defaultdict(set)
    g_dates=pd.to_datetime(g.game_date_key,errors='coerce').dt.date
    for i,r in g.iterrows():
        if pd.isna(g_seasons.iloc[i]): continue
        s=int(g_seasons.iloc[i])
        h=_value_set((r.get('home_team_key'),r.get('cfbd_home_team_raw'),r.get('cfbd_home_team_key_raw')),alias_map)
        a=_value_set((r.get('away_team_key'),r.get('cfbd_away_team_raw'),r.get('cfbd_away_team_key_raw')),alias_map)
        for k in h: home_index[(s,k)].add(i)
        for k in a: away_index[(s,k)].add(i)
    found=[]
    reasons=Counter()
    examples=[]
    has_date=0
    reverse_matched=0
    reverse_candidates=0
    unresolved_teams=Counter()
    pair_no_game=Counter()
    for row_id,r in p.iterrows():
        if pd.isna(p_seasons.iloc[row_id]):
            reason='INVALID_SEASON'; options=[]
        else:
            season=int(p_seasons.iloc[row_id])
            hv=_value_set((r.home_key,r.get('home_raw')),alias_map)
            av=_value_set((r.away_key,r.get('away_raw')),alias_map)
            if not hv or not av:
                reason='BLANK_TEAM'; options=[]; reversed_options=[]
            else:
                hc=set().union(*(home_index.get((season,k),set()) for k in hv))
                ac=set().union(*(away_index.get((season,k),set()) for k in av))
                options=sorted(hc & ac)
                # Independent unordered-pair check; neutral-site feeds can
                # designate the *opposite* home team. This is an identity
                # route, never a guess based on scores, lines or predictions.
                rh=set().union(*(away_index.get((season,k),set()) for k in hv))
                ra=set().union(*(home_index.get((season,k),set()) for k in av))
                reversed_options=sorted((rh & ra)-set(options))
                if not hc and not rh or not ac and not ra:
                    reason='TEAM_NOT_RESOLVED'
                elif not options and not reversed_options:
                    reason='PAIR_NOT_IN_CFBD'
                else:reason='CANDIDATE'
                if reason=='TEAM_NOT_RESOLVED':
                    if not hc and not rh: unresolved_teams['HOME:'+str(r.get('home_key',''))]+=1
                    if not ac and not ra: unresolved_teams['AWAY:'+str(r.get('away_key',''))]+=1
                if reason=='PAIR_NOT_IN_CFBD':
                    pair_no_game[(str(r.get('home_key','')),str(r.get('away_key','')))] += 1
            raw_date='' if pd.isna(r.game_date) else str(r.game_date).strip()
            d=pd.to_datetime(raw_date,errors='coerce')
            if pd.notna(d):
                has_date+=1
                day=d.date()
                # Same game can cross UTC/local calendar boundaries.
                match_date=lambda i:pd.notna(g_dates.iloc[i]) and abs((g_dates.iloc[i]-day).days)<=1
                options=[i for i in options if match_date(i)]
                reversed_options=[i for i in reversed_options if match_date(i)]
                if reason=='CANDIDATE' and not options and not reversed_options:
                    reason='SOURCE_DATE_MISMATCH'
            else:day=None
            if reason=='CANDIDATE':
                # An undated record cannot distinguish an in-season rematch,
                # even if the designated home/away ordering happens to differ.
                all_options=set(options)|set(reversed_options)
                if len(all_options)==1:
                    i=next(iter(all_options))
                    reversed_direction=i in reversed_options
                    if reversed_direction: reverse_matched+=1
                    reason='MATCHED'
                    found.append({'_pt_row_id':row_id,'season':season,
                                  'game_id':str(g.iloc[i].game_id),
                                  'identity_method':('SEASON_UNIQUE_REVERSED_PAIR_DATE' if day else 'SEASON_UNIQUE_REVERSED_PAIR_NO_SOURCE_DATE') if reversed_direction else ('SEASON_ORDERED_PAIR_DATE' if day else 'SEASON_UNIQUE_ORDERED_PAIR_NO_SOURCE_DATE'),
                                  'identity_orientation':'REVERSED' if reversed_direction else 'SAME'})
                elif len(all_options)>1:
                    reason='AMBIGUOUS_REMATCH_NO_DATE' if day is None else 'AMBIGUOUS_DATE_AND_PAIR'
                    reverse_candidates+=bool(reversed_options)
        reasons[reason]+=1
        if reason!='MATCHED' and len(examples)<30:
            examples.append({'season':None if pd.isna(p_seasons.iloc[row_id]) else int(p_seasons.iloc[row_id]),
                'home':str(r.get('home_key','')),'away':str(r.get('away_key','')),'reason':reason})
    mapping=pd.DataFrame(found,columns=['_pt_row_id','season','game_id','identity_method','identity_orientation'])
    # A duplicated game from multiple PT records is NOT an unambiguous ledger row.
    if not mapping.empty:
        collisions=mapping.duplicated(['season','game_id'],keep=False)
        if collisions.any():
            n=int(collisions.sum())
            reasons['DUPLICATE_PT_ROWS_SAME_GAME']+=n
            reasons['MATCHED']-=n
            mapping=mapping.loc[~collisions].copy()
    audit={'pt_source_rows':len(p),'matched_games':len(mapping),
           'source_date_present':has_date,'source_date_missing':len(p)-has_date,
           'identity_routes':mapping.identity_method.value_counts().to_dict(),
           'reasons':{k:int(v) for k,v in sorted(reasons.items())},
           'ambiguous_pairs_forced':0,'fuzzy_matches':0,'unmatched_examples':examples[:20],
           'missing_date_fallback':'UNIQUE_SEASON_ORDERED_PAIR_ONLY',
           'historical_publication_timestamps_certified':False,
           'reversed_orientation_matches':int(mapping.identity_orientation.eq('REVERSED').sum()) if not mapping.empty else 0,
           'reversed_orientation_ambiguous_excluded':int(reverse_candidates),
           'unresolved_pt_team_names_ranked':[
               {'team_key':key,'count':int(count)} for key,count in unresolved_teams.most_common(40)],
           'pair_not_in_cfbd_ranked':[
               {'home_key':home,'away_key':away,'count':int(count)}
               for (home,away),count in pair_no_game.most_common(30)],
           'matched_per_season':{str(k):int(v) for k,v in
               mapping.groupby('season').size().to_dict().items()} if not mapping.empty else {}}
    return mapping,audit
