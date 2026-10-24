"""Unambiguous, read-only, game identity reconciliation for retrospective NCAAF.

Only exact normalized source names or explicit alias-table equivalences count.
Same-game UTC/local date drift of <=1 calendar day is allowed if the mapping
is uniquely one-to-one on BOTH sides. No fuzzy or score-based matching.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import timedelta
import re
import pandas as pd


def norm(value):
    if pd.isna(value): return ""
    return re.sub(r'[^a-z0-9]+', '', str(value).lower())


def make_aliases(frame):
    """Never admit aliases that map to two distinct normalized schools."""
    if not isinstance(frame,pd.DataFrame) or frame.empty:return {}
    source='Source_Team_Name' if 'Source_Team_Name' in frame else 'source'
    target='Team_Norm' if 'Team_Norm' in frame else 'target'
    if source not in frame or target not in frame:return {}
    pairs=defaultdict(set)
    for s,t in zip(frame[source],frame[target]):
        a,b=norm(s),norm(t)
        if a and b:pairs[a].add(b)
    return {k:next(iter(v)) for k,v in pairs.items() if len(v)==1}


def _variants(row,columns,alias_map):
    keys=set()
    for col in columns:
        if col in row and pd.notna(row[col]):
            key=norm(row[col])
            if key:
                keys.add(key)
                if key in alias_map:keys.add(alias_map[key])
    return keys


def reconcile(universe,context,aliases=None):
    """Return mapping CFBD game (season,game_id) -> unique context row index.

    Context must have been reduced to one home row per original source game.
    Log genuine 0-match coverage rather than silently treating as no signal.
    """
    amap=make_aliases(aliases)
    c=context.reset_index(drop=True)
    u=universe.reset_index(drop=True)
    ctx_days=pd.to_datetime(c.ctx_kickoff,utc=True,errors='coerce').dt.date
    cfbd_days=pd.to_datetime(u.kickoff_utc,utc=True,errors='coerce').dt.date
    right=defaultdict(list)
    for i,(season,day) in enumerate(zip(c.season,ctx_days)):
        if day is not pd.NaT and pd.notna(day):right[(int(season),day)].append(i)
    uv=[(_variants(r,('home_team_key','cfbd_home_team_raw','cfbd_home_team_key_raw'),amap),
         _variants(r,('away_team_key','cfbd_away_team_raw','cfbd_away_team_key_raw'),amap)) for _,r in u.iterrows()]
    cv=[(_variants(r,('ctx_home_key','Team','Source_Team_Name','Team_Norm','Canonical_Team_Norm'),amap),
         _variants(r,('ctx_away_key','Opponent','Source_Opponent_Name','Opponent_Norm','Opponent_Canonical_Team_Norm'),amap)) for _,r in c.iterrows()]
    candidates=defaultdict(list)
    id_candidates=0
    for i,(season,day,gid) in enumerate(zip(u.season,cfbd_days,u.game_id)):
        if pd.isna(day):continue
        for delta in (-1,0,1):
            for j in right.get((int(season),day+timedelta(days=delta)),()):
                if uv[i][0].isdisjoint(cv[j][0]) or uv[i][1].isdisjoint(cv[j][1]):continue
                same_id=norm(gid)==norm(c.iloc[j].source_game_id)
                if same_id:id_candidates+=1
                candidates[i].append((j,0 if same_id else 1,abs(delta)))
    # ID-qualified pairs first, then unique name/date pairs. If either side has
    # >1 plausible candidate in its priority class, never pick arbitrarily.
    assigned={};taken=set()
    for priority in (0,1):
        proposals=defaultdict(list)
        for i,options in candidates.items():
            if i in assigned:continue
            best=[(j,d) for j,p,d in options if p==priority and j not in taken]
            if not best:continue
            # Different calendar dates may represent distinct rematches.
            # Even if one is closer, two viable context rows are ambiguous.
            if len(best)==1:proposals[best[0][0]].append(i)
        for j,idxs in proposals.items():
            if len(idxs)==1:
                assigned[idxs[0]]=(j,'ID_AND_TEAMS' if priority==0 else 'DATE_AND_TEAMS')
                taken.add(j)
    rows=[{'season':int(u.iloc[i].season),'game_id':str(u.iloc[i].game_id),
           'ctx_row_id':int(j),'identity_method':method} for i,(j,method) in assigned.items()]
    out=pd.DataFrame(rows,columns=['season','game_id','ctx_row_id','identity_method'])
    # Reject duplicated context rows or CFBD games; tests intentionally exercise.
    if out.duplicated(['season','game_id']).any() or out.ctx_row_id.duplicated().any():
        raise ValueError('REPLAY_IDENTITY_NOT_ONE_TO_ONE')
    audit={'cfbd_games':len(u),'context_home_rows':len(c),'context_matched':len(out),
           'matched_same_source_id':int(out.identity_method.eq('ID_AND_TEAMS').sum()),
           'matched_by_source_team_and_date':int(out.identity_method.eq('DATE_AND_TEAMS').sum()),
           'unmatched_cfbd_games':len(u)-len(out), 'candidate_game_count':len(candidates),
           'explicit_alias_mappings':len(amap),'date_tolerance_calendar_days':1,
           'fuzzy_matches':0,'ambiguous_pairs_forced':0,
           'examples_unmatched_cfbd':u.loc[[i for i in range(len(u)) if i not in assigned],
               ['season','game_id','home_team_key','away_team_key','game_date_key']].head(8).to_dict('records')}
    return out,audit
