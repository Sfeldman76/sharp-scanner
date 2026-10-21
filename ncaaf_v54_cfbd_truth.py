"""V5.4 historical football source firewall.

Historical football GAME IDs, teams, dates, scores and feature statistics are
sourced only from CFBD archives. Rich market quotes are joined separately and
cannot overwrite any football fact. Missing history is reported as missing.

Never claim historical source-publication vintage merely from archival presence.
"""
from __future__ import annotations
import hashlib
import json
import re
from collections import defaultdict
import numpy as np
import pandas as pd

VERSION = 'NCAAF_V54_CFBD_AUTHORITATIVE_HISTORICAL_SOURCE_20261020'
CFBD_FEATURE_VIEW = 'sharplogger.sharp_data.ncaaf_cfbd_archive_research_v1'
CFBD_GAME_ARCHIVE = 'sharplogger.sharp_data.ncaaf_cfbd_api_archive'
MARKET_SIDE_TABLE = 'sharplogger.sharp_data.ncaaf_historical_game_side_raw'
FIRST_YEAR, LAST_HISTORICAL = 2013, 2025


def ident(v):
    return re.sub(r'[^a-z0-9]', '', str(v or '').lower())


def gid(v):
    if v is None or pd.isna(v): return ''
    s = str(v).strip()
    return re.sub(r'\.0$', '', s) if s.lower() not in ('nan','none','<na>','') else ''


def get_df(client, statement):
    result=client.query(statement)
    try:return result.to_dataframe(create_bqstorage_client=False)
    except TypeError:return result.to_dataframe()


def parse_cfbd_archive_rows(records):
    """Verify immutable row hashes and take the earliest consistently completed result.

    Multiple snapshots may exist. Differing final scores are quarantined as
    ambiguous rather than picking the latest post-hoc revision.
    """
    wanted={'id','homeTeam','awayTeam','homePoints','awayPoints'}
    full=[]; audit={'api_rows':len(records),'bad_hash':0,'invalid_payload':0,'completed':0,'score_conflicts':0}
    for r in records.to_dict('records'):
        path=str(r.get('Endpoint') or '')
        if not path.rstrip('/').endswith('/games'):continue
        pjson=str(r.get('Params_JSON') or '{}');payload=str(r.get('Payload_JSON') or '')
        try:
            source_hash=hashlib.sha256(f'{path}|{pjson}|{payload}'.encode('utf-8')).hexdigest()
            if source_hash!=str(r.get('Archive_Row_ID')):
                audit['bad_hash']+=1;continue
            obj=json.loads(payload)
            if not isinstance(obj,dict) or not wanted.issubset(obj):raise ValueError('bad game body')
            if obj.get('completed') is False:continue
            hs=float(obj['homePoints']);aw=float(obj['awayPoints'])
            if not np.isfinite(hs) or not np.isfinite(aw):continue
            yr=r.get('Season')
            if yr is None or pd.isna(yr):yr=json.loads(pjson).get('year')
            if yr is None:raise ValueError('missing season')
            date=pd.to_datetime(obj.get('startDate') or obj.get('startTime'),utc=True,errors='coerce')
            if pd.isna(date):raise ValueError('missing kickoff')
            full.append({'Season':int(yr),'Game_ID':gid(obj['id']),
               'Game_Date_CFBD':date,'CFBD_Home':str(obj['homeTeam']),
               'CFBD_Away':str(obj['awayTeam']),'CFBD_Home_Points':hs,
               'CFBD_Away_Points':aw,'Retrieved_At':r.get('Retrieved_At'),
               'Source_Row_SHA256':source_hash})
            audit['completed']+=1
        except (ValueError,TypeError,KeyError,json.JSONDecodeError):audit['invalid_payload']+=1
    if audit['bad_hash']:
        raise RuntimeError(f'CFBD_API_ARCHIVE_INTEGRITY_FAILURE hash_mismatches={audit["bad_hash"]}')
    if not full:raise RuntimeError('CFBD_GAME_ARCHIVE_NO_VERIFIABLE_COMPLETED_GAMES')
    raw=pd.DataFrame(full)
    disagreements=raw.groupby(['Season','Game_ID']).agg(
        home_score=('CFBD_Home_Points','nunique'),away_score=('CFBD_Away_Points','nunique'),
        home=('CFBD_Home','nunique'),away=('CFBD_Away','nunique'))
    bad=disagreements[(disagreements>1).any(axis=1)]
    audit['score_conflicts']=len(bad)
    if len(bad):raise RuntimeError('CFBD_ARCHIVE_CONFLICTING_FINAL_SCORE_OR_TEAMS n='+str(len(bad)))
    data=(raw.sort_values('Retrieved_At',kind='stable')
            .drop_duplicates(['Season','Game_ID'],keep='first').reset_index(drop=True))
    audit['unique_games']=len(data)
    audit['seasons']={str(int(y)):int(n) for y,n in data.groupby('Season').size().items()}
    return data,audit


def safe_historical_cfbd_columns(columns):
    """Use prior-game football statistics only; static publication vintage pending."""
    exclude=('portal','roster','recruit','talent','weather','windspeed',
             'temperature','returning','injur','current','postgame','result',
             'score','moneyline','market','odds','spread','clv','pt_')
    safe=[]
    for c in columns:
        name=str(c);lower=name.lower()
        if not name.startswith('CFBD_') or any(q in lower for q in exclude):continue
        if name.startswith(('CFBD_Game_','CFBD_Opp_Game_','CFBD_QB_Game_')):continue
        if 'Prior' not in name and 'Matchup_' not in name:continue
        safe.append(name)
    return safe


def _prepare_archive_features(raw):
    essential={'Season','Game_ID','Game_Date','Team','Opponent'}
    if not essential.issubset(raw):raise RuntimeError('CFBD_RESEARCH_ARCHIVE_MISSING_FIELDS '+str(sorted(essential-set(raw))))
    a=raw.copy();a['Season']=pd.to_numeric(a.Season,errors='coerce')
    a=a[a.Season.between(FIRST_YEAR,LAST_HISTORICAL)].copy()
    a['Game_ID']=a.Game_ID.map(gid)
    a['__t']=a.get('Team_Key',a.Team).map(ident);a['__o']=a.get('Opponent_Key',a.Opponent).map(ident)
    a['__date']=pd.to_datetime(a.Game_Date,errors='coerce',utc=True)
    if a['__date'].isna().any():raise RuntimeError('CFBD_RESEARCH_ARCHIVE_INVALID_GAME_DATES')
    if a[['Season','Game_ID','__t']].duplicated().any():raise RuntimeError('CFBD_RESEARCH_ARCHIVE_DUPLICATE_TEAM_GAME')
    if a.Game_ID.eq('').any():raise RuntimeError('CFBD_RESEARCH_ARCHIVE_EMPTY_GAME_IDS')
    for _,g in a.groupby(['Season','Game_ID']):
        if len(g)!=2 or set(g.__t)!=set(g.__o) or g.__t.nunique()!=2:
            raise RuntimeError('CFBD_RESEARCH_ARCHIVE_TWO_SIDE_RECIPROCITY_FAILURE')
    return a


def build_cfbd_canonical_features(features, scores, *, enforce_full_history=True):
    """Scores/identity always from CFBD; never borrow them from market records."""
    a=_prepare_archive_features(features)
    s=scores.copy();s['Season']=pd.to_numeric(s.Season,errors='coerce')
    s['Game_ID']=s.Game_ID.map(gid)
    if s.duplicated(['Season','Game_ID']).any():raise RuntimeError('CFBD_DUPLICATE_FINAL_GAMES')
    m=a.merge(s,on=['Season','Game_ID'],how='left',validate='many_to_one',indicator=True)
    m['__home']=m.CFBD_Home.map(ident);m['__away']=m.CFBD_Away.map(ident)
    orientation=np.select([m.__t.eq(m.__home),m.__t.eq(m.__away)],[1,0],default=-1)
    wrong=(m._merge.eq('both')) & (orientation<0)
    if wrong.any():raise RuntimeError('CFBD_ARCHIVE_SCORES_TEAM_ORIENTATION_CONFLICT n='+str(int(wrong.sum())))
    valid=m._merge.eq('both') & (orientation>=0)
    m['Is_Home']=np.where(valid,orientation,np.nan)
    m['Team_Score']=np.where(orientation==1,m.CFBD_Home_Points,m.CFBD_Away_Points)
    m['Opponent_Score']=np.where(orientation==1,m.CFBD_Away_Points,m.CFBD_Home_Points)
    m.loc[~valid,['Team_Score','Opponent_Score']]=np.nan
    # Time zones can place the calendar date on either adjacent day; >1 day fails.
    date_gap=(m.loc[valid,'__date'].dt.normalize()-m.loc[valid,'Game_Date_CFBD'].dt.normalize()).abs().dt.days
    if date_gap.gt(1).any():raise RuntimeError('CFBD_ARCHIVE_GAME_DATE_DISAGREEMENT')
    m['Source_Game_ID']=m.Game_ID
    m['Team_Norm']=m.Team.astype(str).str.lower().str.strip()
    m['Opponent_Norm']=m.Opponent.astype(str).str.lower().str.strip()
    m['Actual_Margin']=m.Team_Score-m.Opponent_Score
    m['Actual_Total']=m.Team_Score+m.Opponent_Score
    keep_base=['Season','Game_ID','Source_Game_ID','Game_Date','Team','Opponent','Team_Norm',
         'Opponent_Norm','Team_Score','Opponent_Score','Is_Home','Actual_Margin','Actual_Total',
         'Week','Season_Type','CFBD_Conference','CFBD_Opp_Conference']
    keep_base=[c for c in keep_base if c in m]
    safe=safe_historical_cfbd_columns(features.columns)
    canonical=m[keep_base+safe].copy()
    canonical['CFBD_History_Eligible']=valid.astype(int).to_numpy()
    canonical['Historical_Football_Source']='CFBD_API_ARCHIVE_GAMES_PLUS_CFBD_RESEARCH_FEATURES'
    info={'cfbd_feature_rows':len(a),'unique_feature_games':int(a.Game_ID.nunique()),
       'cfbd_scored_side_rows':int(valid.sum()),'unscored_sides':int((~valid).sum()),
       'score_match_rate':float(valid.mean()),'safe_pregame_feature_fields':len(safe),
       'excluded_cfbd_fields':len([c for c in features if str(c).startswith('CFBD_')])-len(safe),
       'scores_cfbd_only':True,'market_joined_to_football_fields':False,
       'score_coverage_by_year':{str(int(y)):{'side_rows':len(g),'scored':int(g.CFBD_History_Eligible.sum())}
            for y,g in canonical.groupby('Season')}}
    if enforce_full_history and not 0.8<=info['score_match_rate']<=1.0:
        raise RuntimeError('CFBD_HISTORICAL_SCORE_COVERAGE_INSUFFICIENT '+str(info['score_match_rate']))
    # No silently restricting to 2022+: full-history core requires 2013 onward.
    if enforce_full_history:
        for y in range(FIRST_YEAR,LAST_HISTORICAL+1):
            d=info['score_coverage_by_year'].get(str(y),{})
            if (not d or d.get('side_rows',0)<500 or
                    d.get('scored',0)<int(.8*d.get('side_rows',0))):
                raise RuntimeError(f'CFBD_HISTORICAL_SEASON_INCOMPLETE year={y} '+str(d))
    return canonical,info


def _market_id_frame(markets):
    req={'Season','Source_Game_ID','Consensus_Open_Spread'}
    if not req.issubset(markets):raise RuntimeError('MARKET_RECORD_MISSING_JOIN_KEYS')
    d=markets.copy();d['Season']=pd.to_numeric(d.Season,errors='coerce')
    if 'Source_Name' not in d:
        raise RuntimeError('MARKET_SOURCE_NAME_MISSING__CANNOT_PROVE_CFBD_GAME_ID')
    d=d.loc[d.Source_Name.astype(str).str.casefold().eq('collegefootballdata')].copy()
    if d.empty:
        raise RuntimeError('MARKET_CFBD_GAME_ID_SOURCE_EMPTY')
    d['Game_ID']=d.Source_Game_ID.map(gid)
    if 'Source_Team_Name' not in d:
        raise RuntimeError('MARKET_SOURCE_TEAM_NAME_MISSING')
    d['__team']=d.Source_Team_Name.map(ident)
    if d.__team.eq('').any():raise RuntimeError('MARKET_EMPTY_SOURCE_TEAM')
    d['__home']=pd.to_numeric(d.get('Is_Home',pd.Series(np.nan,index=d.index)),errors='coerce')
    cols=['Consensus_Open_Spread','Consensus_Open_Total','Consensus_Open_Moneyline',
          'Consensus_Close_Spread','Consensus_Close_Total','Consensus_Close_Moneyline']
    for c in cols:
        if c not in d:d[c]=np.nan
        d[c]=pd.to_numeric(d[c],errors='coerce')
    # Conflicting historical quote snapshots are NOT averaged or cherry-picked:
    # they remain missing until original quote timestamps can separate them.
    groupcols=['Season','Game_ID','__team','__home']
    for c in cols:
        count=d.groupby(groupcols,dropna=False)[c].transform(lambda s:s.dropna().round(5).nunique())
        d.loc[count>1,c]=np.nan
    d=d.sort_values(groupcols,kind='stable').drop_duplicates(groupcols)
    for (yr,gid_key),group in d.groupby(['Season','Game_ID']):
        if len(group)>2:
            raise RuntimeError(f'MARKET_MORE_THAN_TWO_SIDES season={yr} game={gid_key}')
        if len(group)==2:
            for c in ('Consensus_Open_Spread','Consensus_Close_Spread'):
                prices=group[c].dropna()
                if len(prices)==2 and not np.isclose(float(prices.sum()),0.,atol=0.001):
                    raise RuntimeError(f'MARKET_SPREAD_SIDES_NOT_OPPOSITES season={yr} game={gid_key} col={c}')
    return d[groupcols+cols]


def attach_market_only(cfbd, market):
    """Allowlist sportsbook fields only, with validated original side identity."""
    d=cfbd.copy();d['__team']=d.Team.map(ident)
    mk=_market_id_frame(market)
    cols=[c for c in mk if c.startswith('Consensus_')]
    # Exact CFBD game/team source name is first choice.
    j=d.merge(mk.drop(columns='__home'),on=['Season','Game_ID','__team'],
              how='left',validate='many_to_one')
    exact=int(j.Consensus_Open_Spread.notna().sum())
    # Some old markets carry mascot-inclusive aliases. Only if there is a
    # unique one-to-one game/Is_Home alignment may the market quote be rescued.
    candidates=mk[mk.__home.isin([0,1])].copy()
    distinct=candidates.groupby(['Season','Game_ID','__home']).size()
    candidates=candidates.loc[~pd.MultiIndex.from_frame(candidates[['Season','Game_ID','__home']]).isin(distinct[distinct>1].index)].copy()
    if not candidates.empty:
        right=candidates.drop(columns='__team').rename(columns={c:'__market_'+c for c in cols})
        more=j[['Season','Game_ID','Is_Home']].merge(right,left_on=['Season','Game_ID','Is_Home'],
                           right_on=['Season','Game_ID','__home'],how='left',validate='many_to_one')
        for c in cols:
            src=more['__market_'+c].to_numpy()
            j[c]=j[c].where(j[c].notna(),pd.Series(src,index=j.index))
    j=j.drop(columns='__team')
    if len(j)!=len(cfbd):raise RuntimeError('MARKET_JOIN_CHANGED_CFBD_ROW_COUNT')
    j['Market_Open_Margin']=-pd.to_numeric(j.Consensus_Open_Spread,errors='coerce')
    j['Opening_Spread']=j['Consensus_Open_Spread']
    j['ATS_Win']=np.where(j.Actual_Margin.notna() & j.Consensus_Open_Spread.notna(),
                           np.where(j.Actual_Margin+j.Consensus_Open_Spread>0,1.,
                                    np.where(j.Actual_Margin+j.Consensus_Open_Spread<0,0.,np.nan)),np.nan)
    aud={'football_source':'CFBD_ONLY','market_source':MARKET_SIDE_TABLE,
         'market_fields_imported':cols,
         'no_market_football_fact_overwrite':True,
         'exact_market_open_sides':exact,
         'market_open_sides':int(j.Consensus_Open_Spread.notna().sum()),
         'side_orientation_rescued':int(j.Consensus_Open_Spread.notna().sum())-exact}
    return j,aud


def home_games(cfbd_sides):
    scored=cfbd_sides.loc[cfbd_sides.CFBD_History_Eligible.eq(1)].copy()
    chosen=scored[scored.Is_Home.eq(1)].copy()
    if chosen.duplicated(['Season','Game_ID']).any():raise RuntimeError('CFBD_DUPLICATE_HOME_ORIENTATION')
    all_games=scored[['Season','Game_ID']].drop_duplicates()
    if len(chosen)!=len(all_games):
        raise RuntimeError(f'CFBD_HOME_ORIENTATION_INCOMPLETE home={len(chosen)} games={len(all_games)}')
    return chosen.reset_index(drop=True)



def verify_cfbd_frame(frame, *, require_unique_home=False):
    """Fail if another table substituted historical outcomes/teams/features."""
    if frame is None or frame.empty:raise RuntimeError('CFBD_SOURCE_FRAME_EMPTY')
    req={'Historical_Football_Source','CFBD_History_Eligible','Game_ID',
         'Season','Team','Opponent','Actual_Margin','Actual_Total','Game_Date'}
    if not req.issubset(frame):raise RuntimeError('CFBD_FRAME_MISSING_REQUIRED '+str(sorted(req-set(frame))))
    if not frame.Historical_Football_Source.astype(str).eq(
            'CFBD_API_ARCHIVE_GAMES_PLUS_CFBD_RESEARCH_FEATURES').all():
        raise RuntimeError('CFBD_SOURCE_FRAME_UNVERIFIED_OR_MIXED')
    if not pd.to_numeric(frame.CFBD_History_Eligible,errors='coerce').eq(1).all():
        raise RuntimeError('CFBD_SOURCE_FRAME_UNSCORED_ROWS')
    if pd.to_numeric(frame.Season,errors='coerce').ge(2026).any():
        raise RuntimeError('CFBD_SOURCE_FRAME_2026_LABEL_ACCESS')
    if frame[['Season','Game_ID','Team']].duplicated().any():
        raise RuntimeError('CFBD_SOURCE_FRAME_DUPLICATE_SIDES')
    if require_unique_home and frame[['Season','Game_ID']].duplicated().any():
        raise RuntimeError('CFBD_SOURCE_FRAME_NOT_ONE_PER_GAME')
    if not np.isfinite(pd.to_numeric(frame.Actual_Margin,errors='coerce')).all():
        raise RuntimeError('CFBD_SOURCE_FRAME_INVALID_RESULTS')
    return {'status':'PASS','rows':len(frame),'games':int(frame.Game_ID.nunique()),
            'source':'CFBD_API_ARCHIVE_GAMES_PLUS_CFBD_RESEARCH_FEATURES',
            '2026_outcomes_used':False,'legacy_football_fields_used':False}


def derived_pregame_states(sides):
    """Prior-only team W/L/ATS record and role state from CFBD scores + market odds.

    Sort by date and apply observations only after all same-day games are evaluated.
    """
    d=sides.copy();d['__dt']=pd.to_datetime(d.Game_Date,utc=True).dt.normalize()
    d['__tm']=d.Team.map(ident);d['__g']=d.Game_ID.map(gid)
    d=d.sort_values(['Season','__dt','__g','__tm'],kind='stable').reset_index(drop=True)
    state={};output=defaultdict(list)
    for i,row in d.iterrows():
        key=(int(row['Season']),row['__tm'])
        hist=state.setdefault(key,[])
        earlier=[e for e in hist if e['date']<row['__dt']]
        prior=len(earlier);wins=sum(x['win']==1 for x in earlier)
        ats=[x['ats'] for x in earlier if np.isfinite(x['ats'])]
        a_losses=0
        for x in reversed(ats):
            if x==0:a_losses+=1
            else:break
        for k,v in [('Game_Number_Prior',prior),('WinPct_Prior',wins/prior if prior else np.nan),
                    ('ATS_Loss_Streak_Prior',a_losses if ats else np.nan),
                    ('Prev_Points_For',earlier[-1]['for'] if earlier else np.nan),
                    ('Prev_SU_Win',earlier[-1]['win'] if earlier else np.nan),
                    ('Prev_Is_Home',earlier[-1]['home'] if earlier else np.nan),
                    ('Prev_Is_ML_Dog',earlier[-1]['dog'] if earlier else np.nan),
                    ('Prev_Is_ML_Favorite',earlier[-1]['favorite'] if earlier else np.nan)]:
            output[k].append(v)
        margin=float(row['Actual_Margin']) if pd.notna(row['Actual_Margin']) else np.nan
        line=float(row['Consensus_Open_Spread']) if pd.notna(row.get('Consensus_Open_Spread')) else np.nan
        obs={'date':row['__dt'],'win':float(margin>0) if np.isfinite(margin) else np.nan,
             'ats':float(margin+line>0) if np.isfinite(margin) and np.isfinite(line) and margin+line!=0 else np.nan,
             'for':row.get('Team_Score'), 'home':row.get('Is_Home'),
             'dog':float(line>0) if np.isfinite(line) else np.nan,
             'favorite':float(line<0) if np.isfinite(line) else np.nan}
        hist.append(obs)
    for c,vals in output.items():d[c]=vals
    # Mirror opponent states on the same CFBD Game_ID, never using the current result.
    prior_cols=list(output)
    opp=d[['Season','Game_ID','Team']+prior_cols].copy()
    opp=opp.rename(columns={'Team':'Opponent',**{c:'Opp_'+c for c in prior_cols}})
    d=d.merge(opp,on=['Season','Game_ID','Opponent'],validate='one_to_one')
    return d.drop(columns=['__dt','__tm','__g'])


def build_cfbd_research_inputs(archive_features, raw_api_games, rich_market):
    scores,api_audit=parse_cfbd_archive_rows(raw_api_games)
    sides,score_audit=build_cfbd_canonical_features(archive_features,scores)
    joint,market_audit=attach_market_only(sides,rich_market)
    # Never let partial CFBD archive gaps be silently repaired by market scores.
    # The archive coverage audit above still documents the missing games.
    joint=joint.loc[joint.CFBD_History_Eligible.eq(1)].copy()
    joint=derived_pregame_states(joint)
    home=home_games(joint)
    verify_cfbd_frame(home,require_unique_home=True)
    audit={'status':'CFBD_AUTHORITATIVE__MARKET_SEPARATE','api_game_archive':api_audit,
           'source_match':score_audit,'market':market_audit,
           'unique_home_games':len(home),'team_side_rows':len(joint),
           '2026_in_training':False,'vintage_certified':False,
           'policy':'NO_HISTORICAL_FOOTBALL_FACTS_FROM_LEGACY_SCORES_OR_MARKET_TABLES'}
    return home,joint,audit


def load_from_bigquery(bq_client):
    """No BigQuery mutations; retrieve the three strictly separated inputs."""
    features=get_df(bq_client,f'SELECT * FROM `{CFBD_FEATURE_VIEW}` WHERE Season BETWEEN 2013 AND 2025')
    api=get_df(bq_client,f'''SELECT Archive_Row_ID, Endpoint, Params_JSON, Season, Retrieved_At, Payload_JSON
        FROM `{CFBD_GAME_ARCHIVE}` WHERE Season BETWEEN 2013 AND 2025
        AND REGEXP_CONTAINS(LOWER(Endpoint),r'/games/?$')''')
    # Market-only SQL projection: it cannot leak football outcomes, scores or
    # postgame statistics because those fields are never queried.
    mk=get_df(bq_client,f'''SELECT Source_Name, Season, Source_Game_ID, Source_Team_Name, Is_Home,
        Consensus_Open_Spread, Consensus_Open_Total, Consensus_Open_Moneyline,
        Consensus_Close_Spread, Consensus_Close_Total, Consensus_Close_Moneyline
        FROM `{MARKET_SIDE_TABLE}` WHERE Source_Name = 'CollegeFootballData'
        AND Season BETWEEN 2013 AND 2025''')
    home,sides,audit=build_cfbd_research_inputs(features,api,mk)
    return home,sides,audit


def generate_cfbd_expert_occurrences(all_sides, flagged, utils_module):
    """Map only declared football expert systems to the CFBD physical game.

    Expert components are retrospective rule applications, not pregame tickets.
    Context-only key screens are not treated as directional expert selections.
    """
    cfbd_ids={(int(y),str(g)) for y,g in zip(all_sides.Season,all_sides.Game_ID)}
    if len(flagged)!=len(all_sides):raise RuntimeError('CFBD_EXPERT_FLAG_ROW_ALIGNMENT_BROKEN')
    exact=[x for x in getattr(utils_module,'BIGAL_EXACT_SIGNAL_COLS',()) if x.startswith('BigAl_CF')]
    # Pathi: only predeclared directional key-movement events, not all key contexts.
    pathi=list(getattr(utils_module,'PATHI_KEY_EVENT_COLS',()))
    records=[]; ready_count={}; fires={}
    for fam,names in [('BIG_AL',exact),('PATHI',pathi)]:
        for name in names:
            if name not in flagged:continue
            ready_col=name+'_DataReady'
            ready=pd.to_numeric(flagged[ready_col],errors='coerce').fillna(0).eq(1) if ready_col in flagged else pd.Series(False,index=flagged.index)
            fire=pd.to_numeric(flagged[name],errors='coerce').fillna(0).eq(1) & ready
            ready_count[name]=int(ready.sum());fires[name]=int(fire.sum())
            for _,row in flagged.loc[fire].iterrows():
                key=(int(row['Season']),str(row['Game_ID']))
                if key not in cfbd_ids:raise RuntimeError('CFBD_EXPERT_FLAG_UNKNOWN_GAME_ID')
                records.append({'Season':key[0],'Game_ID':key[1],
                    'Family':fam,'System_ID':name,
                    'Side':'TEAM' if int(row['Is_Home'])==1 else 'OPPONENT',
                    'Evidence_Class':'CFBD_RECONSTRUCTED_RETROSPECTIVE__NO_ARCHIVE_VOTE'})
    d=pd.DataFrame(records,columns=['Season','Game_ID','Family','System_ID','Side','Evidence_Class'])
    conflicts=0
    if not d.empty:
        groups=d.groupby(['Season','Game_ID','Family','System_ID']).Side.nunique()
        bad=groups[groups>1].index
        conflicts=len(bad)
        if len(bad):d=d.loc[~pd.MultiIndex.from_frame(d[['Season','Game_ID','Family','System_ID']]).isin(bad)]
        d=d.drop_duplicates(['Season','Game_ID','Family','System_ID','Side'])
    return d,{'systems_considered':len(exact)+len(pathi),'systems_ready':ready_count,
              'systems_fired':fires,'game_system_records':len(d),'contradictions_quarantined':conflicts,
              'source':'CFBD_GAME_ARCHIVE_PLUS_PREGAME_CFBD_STATE_PLUS_MARKET_PRICE_ONLY',
              'predeclared_archived_votes':0}
