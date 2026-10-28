"""Conservative, as-of conference context for frozen NCAAF Miner mechanisms.

This module does NOT infer advanced game ratings, missing PT providers, STAT edges,
or outcome-based features. A conference is accepted only from an unambiguous
same-season, pre-kickoff source observation. All counts are lower bounds from
completed, de-duplicated, strictly earlier games; unknowns stay NaN.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
import pandas as pd

TEAM_CONF_FIELDS = ('Team_Conference','Team_Conference_Norm','Conference','Conference_Norm',
                    'Context_Conference','CFBD_Team_Conference','CFBD_Conference')
OPP_CONF_FIELDS = ('Opponent_Conference','Opponent_Conference_Norm','Opp_Conference',
                   'Context_Opp_Conference','CFBD_Opponent_Conference')


def _normal_conf(v):
    if pd.isna(v): return ''
    z=' '.join(str(v).lower().replace('_',' ').split())
    if z in ('','nan','none','null','unknown','n/a','<na>','fbs','fcs','division i'):
        return ''
    # Never equate two unrelated independents as conference opponents.
    if z in ('independent','independents','fbs independents'): return 'independent'
    aliases={
        'southeastern conference':'sec',
        'atlantic coast conference':'acc',
        'big ten conference':'big ten', 'big 10':'big ten', 'b1g':'big ten',
        'big 12 conference':'big 12', 'big twelve':'big 12',
        'pacific 12':'pac 12', 'pac 12 conference':'pac 12', 'pac12':'pac 12',
        'american athletic':'aac', 'american athletic conference':'aac', 'american':'aac',
        'mountain west':'mwc','mountain west conference':'mwc',
        'mid american':'mac','mid american conference':'mac',
        'sun belt conference':'sun belt',
        'conference usa':'c usa','conference u s a':'c usa','cusa':'c usa',
    }
    return aliases.get(z,z)


def _first(row, cols):
    for c in cols:
        if c in row.index:
            z=_normal_conf(row[c])
            if z:return z
    return ''


def _dt(s): return pd.to_datetime(s,errors='coerce',utc=True)


def restore_verified_conference_context(live, prior, *, team_key, asof=None):
    """Attach only provable pregame conference facts and lower-bound counts.

    `prior` must be prior completed game records (e.g. CFBD miner features), not
    snapshots whose rows are created or revised using the upcoming result.
    Teams with conflicting season-level affiliations remain unknown.  Never
    copy a prior game's own SAME_CONFERENCE flag to a different matchup.
    """
    out=live.copy()
    diag=dict(status='NO_SOURCE',rows=len(out),pairs_verified=0,same_conference_filled=0,
              conference_4plus_filled=0,at_least_three_prior_games=0,
              source_team_conference_records=0,ambiguous_team_conferences=0,
              source_completed_sides=0,source_conference_games=0,
              reason='NO_PRIOR_WITH_CONFERENCE_METADATA',authority_policy='FROZEN_RULES_ONLY')
    if out.empty or not isinstance(prior,pd.DataFrame) or prior.empty or 'Season' not in prior or 'Game_Date' not in prior:
        return out,diag
    if not (set(prior.columns)&set(TEAM_CONF_FIELDS+OPP_CONF_FIELDS)):
        return out,diag
    now=_dt(asof if asof is not None else datetime.now(timezone.utc))
    # All current sides must be future to this run, even if an archived future
    # record has accidentally entered the CFBD source table.
    dcol='Game_Start' if 'Game_Start' in out else 'Game_Date'
    if dcol not in out:return out,diag
    live_dates=_dt(out[dcol]);live_sy=pd.to_numeric(out.get('Season'),errors='coerce')
    # The authoritative live frame is home-oriented by attach_live_miner_votes.
    home_col=next((x for x in ('Home_Team_Norm','Home_Team','Team_Norm','Team') if x in out),None)
    away_col=next((x for x in ('Away_Team_Norm','Away_Team','Opponent_Norm','Opponent') if x in out),None)
    if not home_col or not away_col:return out,diag
    key=lambda x:team_key(x) if pd.notna(x) else ''
    hist=prior.copy()
    hist['_sy']=pd.to_numeric(hist['Season'],errors='coerce')
    hist['_date']=_dt(hist['Game_Date'])
    tc=next((x for x in ('Team_Key','Team_Norm','Team') if x in hist),None)
    oc=next((x for x in ('Opponent_Key','Opponent_Norm','Opponent') if x in hist),None)
    if not tc or not oc:return out,diag
    hist['_t']=hist[tc].map(key);hist['_o']=hist[oc].map(key)
    hist=hist.loc[hist['_sy'].notna() & hist['_date'].notna() & hist['_date'].le(now)
                  & hist['_t'].ne('') & hist['_o'].ne('')].copy()
    # A source row timestamped after the future fixture is never eligible.
    if hist.empty:return out,diag
    # Build season-specific metadata from each team's completed past games.
    # Do not infer this matchup's conference identity from prior opponents.
    counts=defaultdict(int); memberships=defaultdict(set)
    game_seen=set(); same_games=set();src_conf=0
    for _,r in hist.iterrows():
        sy=int(r['_sy']);t=r['_t'];o=r['_o'];date=r['_date']
        c1=_first(r,TEAM_CONF_FIELDS); c2=_first(r,OPP_CONF_FIELDS)
        if c1:memberships[(sy,t)].add(c1);src_conf+=1
        if c2:memberships[(sy,o)].add(c2);src_conf+=1
        # Game ID dedup handles both mirrored team sides and multiple source rows.
        gid=str(r.get('Game_ID','')).strip()
        if not gid or gid.lower() in ('nan','none','<na>'):
            gid='|'.join(sorted((t,o)))+'|'+date.strftime('%Y-%m-%d')
        token=(sy,t,gid)
        if token in game_seen: continue
        game_seen.add(token)
        counts[(sy,t)] += 1
        # For explicit conference-game counts, require both conf memberships
        # known ON THIS HISTORICAL GAME, not retrospectively inferred.
        if c1 and c2 and c1==c2 and c1!='independent':same_games.add(token)
        # Mirrored side may be absent. Count games for opponents only from their
        # own row: no assuming full source coverage.
    if not src_conf:return out,diag
    invalid={k for k,v in memberships.items() if len(v)>1}
    # A one-sided conference field can still identify previous conference games
    # if BOTH teams have their own unique season membership in earlier records.
    # This is conference metadata only; it never examines previous results.
    for _,r in hist.iterrows():
        sy=int(r['_sy']);t=r['_t'];o=r['_o']
        m1=memberships.get((sy,t),set());m2=memberships.get((sy,o),set())
        if len(m1)!=1 or len(m2)!=1 or 'independent' in m1 or m1!=m2:continue
        gid=str(r.get('Game_ID','')).strip()
        if not gid or gid.lower() in ('nan','none','<na>'):
            gid='|'.join(sorted((t,o)))+'|'+r['_date'].strftime('%Y-%m-%d')
        same_games.add((sy,t,gid))
    ccol='CFBD_Same_Conference';gcol='CFBD_Conference_Games_Prior';pcol='CFBD_Games_Prior'
    same=pd.to_numeric(out[ccol],errors='coerce').copy() if ccol in out else pd.Series(np.nan,index=out.index,dtype=float)
    confn=pd.to_numeric(out[gcol],errors='coerce').copy() if gcol in out else pd.Series(np.nan,index=out.index,dtype=float)
    games=pd.to_numeric(out[pcol],errors='coerce').copy() if pcol in out else pd.Series(np.nan,index=out.index,dtype=float)
    pairs=0;flsame=0;flconf=0;flgames=0
    # Counts from full side history strictly before each individual kickoff.
    byteam=defaultdict(list)
    for _,r in hist.iterrows():
        sy=int(r['_sy']);t=r['_t'];o=r['_o']; date=r['_date']
        gid=str(r.get('Game_ID','')).strip()
        if not gid or gid.lower() in ('nan','none','<na>'):gid='|'.join(sorted((t,o)))+'|'+date.strftime('%Y-%m-%d')
        byteam[(sy,t)].append((date,gid,(sy,t,gid) in same_games))
    for i in out.index:
        start=live_dates.loc[i];sy=live_sy.loc[i]
        if pd.isna(start) or start<=now or pd.isna(sy):continue
        s=int(sy);h=key(out.loc[i,home_col]);a=key(out.loc[i,away_col])
        if not h or not a or h==a:continue
        m1=memberships.get((s,h),set());m2=memberships.get((s,a),set())
        if (s,h) in invalid or (s,a) in invalid:continue
        if len(m1)==1 and len(m2)==1:
            pairs+=1
            isc=float(next(iter(m1))==next(iter(m2)) and 'independent' not in m1 and 'independent' not in m2)
            if pd.isna(same.loc[i]):same.loc[i]=isc;flsame+=1
        # Count lower bounds only: 3+ complete prior conferences is provable
        # even if other completed games are missing from the source.
        prev={gid for date,gid,_ in byteam[(s,h)] if date<start}
        if len(prev)>=3 and pd.isna(games.loc[i]):games.loc[i]=len(prev);flgames+=1
        prev_conf={gid for date,gid,is_same in byteam[(s,h)] if date<start and is_same}
        if same.loc[i]==1 and len(prev_conf)>=3 and pd.isna(confn.loc[i]):
            confn.loc[i]=len(prev_conf);flconf+=1
    if flsame:out[ccol]=same
    if flconf:out[gcol]=confn
    if flgames:out[pcol]=games
    diag.update(status='PASS',reason='STRICTLY_PRIOR_CONFERENCE_METADATA',pairs_verified=pairs,
                same_conference_filled=flsame,conference_4plus_filled=flconf,
                at_least_three_prior_games=flgames,source_team_conference_records=src_conf,
                ambiguous_team_conferences=len(invalid),source_completed_sides=len(game_seen),
                source_conference_games=len(same_games))
    return out,diag

# Secondary source is used ONLY for factual conference identity and countable
# completed fixtures. Never import its postgame features into the live frame.
_CONFERENCE_META_TABLES=(
    'sharplogger.sharp_data.ncaaf_historical_game_side_context',
    'sharplogger.sharp_data.ncaaf_historical_game_side_raw',
    'sharplogger.sharp_data.ncaaf_cfbd_archive_research_v1',
)

def load_verified_conference_history(bq_client, *, season=2026, asof=None):
    """Schema-aware, read-only load of earlier game identities and conferences.

    Skips a table if missing a trustworthy conference column or side identity;
    never attempts positional lookup or constructs conference membership from
    current bookmaker odds. All queries restrict dates to the current UTC time.
    """
    diag=dict(status='NO_VERIFIED_CONFERENCE_TABLE',source='',rows=0,errors=[])
    if bq_client is None:return pd.DataFrame(),{**diag,'status':'NO_BQ_CLIENT'}
    now=_dt(asof if asof is not None else datetime.now(timezone.utc))
    def pick(cols,names):return next((s for s in names if s in cols),None)
    for table in _CONFERENCE_META_TABLES:
        try:
            fields={f.name for f in bq_client.get_table(table).schema}
            sc=pick(fields,('Season','season'));dc=pick(fields,('Game_Date','Game_Start','Start_Date','game_date'))
            tc=pick(fields,('Team_Norm','Team_Key','Team','Home_Team_Norm','home_team'))
            oc=pick(fields,('Opponent_Norm','Opponent_Key','Opponent','Away_Team_Norm','away_team'))
            tcc=pick(fields,TEAM_CONF_FIELDS)
            occ=pick(fields,OPP_CONF_FIELDS)
            home_conf=pick(fields,('Home_Conference','Home_Conference_Norm','home_conference'))
            away_conf=pick(fields,('Away_Conference','Away_Conference_Norm','away_conference'))
            home_col=tc in ('Home_Team_Norm','Home_Team','home_team')
            away_col=oc in ('Away_Team_Norm','Away_Team','away_team')
            is_home_col=pick(fields,('Is_Home','is_home'))
            # Home/Away conference metadata is only usable if its orientation
            # relative to Team/Opponent is proven; otherwise do not guess.
            if home_conf and away_conf and not (tcc or occ):
                if home_col and away_col:
                    tcc=home_conf;occ=away_conf
                elif is_home_col:
                    home_flag=f'LOWER(CAST({is_home_col} AS STRING)) IN ("1","true")'
                    tcc=f'IF({home_flag},{home_conf},{away_conf})'
                    occ=f'IF({home_flag},{away_conf},{home_conf})'
            if not (sc and dc and tc and oc and (tcc or occ)):
                continue
            # Safe column names are selected from explicit allowlists above.
            projections=[f'{sc} AS Season',f'{dc} AS Game_Date',
                         f'{tc} AS Team_Key',f'{oc} AS Opponent_Key']
            if tcc:projections.append(f'{tcc} AS Conference')
            if occ:projections.append(f'{occ} AS Opponent_Conference')
            gc=pick(fields,('Game_ID','Source_Game_ID','game_id'))
            if gc:projections.append(f'{gc} AS Game_ID')
            sql=(f'SELECT {", ".join(projections)} FROM `{table}` '
                 f'WHERE SAFE_CAST({sc} AS INT64) = {int(season)} '
                 f'AND SAFE_CAST({dc} AS TIMESTAMP) <= TIMESTAMP("{now.isoformat()}")')
            try:r=bq_client.query(sql).to_dataframe(create_bqstorage_client=False)
            except TypeError:r=bq_client.query(sql).to_dataframe()
            if r is not None and not r.empty:
                return r,dict(status='PASS',source=table,rows=len(r),errors=diag['errors'],
                              conference_columns=[x for x in ('Conference','Opponent_Conference') if x in r])
        except Exception as exc:
            diag['errors'].append({'source':table,'error':type(exc).__name__,'message':str(exc)[:100]})
    return pd.DataFrame(),diag
