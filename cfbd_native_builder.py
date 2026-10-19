"""CFBD-native bronze->silver->strictly-pregame gold research feature construction.

All source data is read from immutable CFBD endpoint archives, not existing Core rows.
Past games can update football state only once their results are demonstrably prior to
prediction cutoff. Without verified completion timestamp an eight? NO: enforce a
conservative 24h embargo from the kickoff, even across same-day game slots.

Raw measurements remain distinct from previously-derived CFBD or Big Al features.
"""
from __future__ import annotations
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import heapq
import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from cfbd_native_archive import load_family


def num(v: Any) -> float:
    """Numeric parsing: excludes '-1 for 5' box lines, percentages need separate handling."""
    if isinstance(v,bool) or v is None: return float('nan')
    if isinstance(v,(int,float)):
        return float(v) if math.isfinite(v) else float('nan')
    text=str(v).strip().replace(',','')
    if not text: return float('nan')
    if text.endswith('%'):
        try: return float(text[:-1])/100.0
        except ValueError: return float('nan')
    try: return float(text)
    except ValueError: return float('nan')


def team_name(value:Any) -> str:
    return re.sub(r'[^a-z0-9]','',str(value or '').casefold())


def parse_time(v:Any) -> pd.Timestamp:
    t=pd.to_datetime(v,errors='coerce',utc=True)
    return t if not pd.isna(t) else pd.NaT


def get_nested(d: dict, *path: str) -> Any:
    cur=d
    for p in path:
        if not isinstance(cur,dict): return None
        cur=cur.get(p)
    return cur


def build_games(data:list[dict]) -> pd.DataFrame:
    records=[]; ids=set()
    for g in data:
        gid=g.get('id')
        if gid is None: continue
        gid=str(gid)
        if gid in ids: raise ValueError(f'Duplicate physical game ID {gid}')
        ids.add(gid)
        season=g.get('season')
        kickoff=parse_time(g.get('startDate') or g.get('startTime'))
        home=g.get('homeId') if g.get('homeId') is not None else g.get('homeTeam')
        away=g.get('awayId') if g.get('awayId') is not None else g.get('awayTeam')
        if home is None or away is None or str(home)==str(away):
            raise ValueError(f'Invalid team IDs {gid}')
        hp=num(g.get('homePoints')); ap=num(g.get('awayPoints'))
        if pd.notna(hp) != pd.notna(ap): raise ValueError(f'Incomplete score pair {gid}')
        if pd.notna(hp) and (hp<0 or ap<0): raise ValueError(f'Negative score {gid}')
        records.append({'game_id':gid,'season':int(season), 'week':int(g.get('week') or 0),
                        'kickoff':kickoff,'home_id':str(home), 'away_id':str(away),
                        'home_name':str(g.get('homeTeam') or home),
                        'away_name':str(g.get('awayTeam') or away),
                        'home_points':hp,'away_points':ap,
                        'neutral_site':int(bool(g.get('neutralSite',False))),
                        'season_type':str(g.get('seasonType') or ''),
                        'home_classification':g.get('homeClassification'),
                        'away_classification':g.get('awayClassification')})
    df=pd.DataFrame.from_records(records)
    if df.empty: return df
    if df['kickoff'].isna().any():
        raise ValueError('Kickoff missing: game is unsafe for as-of reconstruction')
    return df.sort_values(['kickoff','game_id']).reset_index(drop=True)


def _game_teams(games:pd.DataFrame) -> dict:
    x={}
    for g in games.itertuples(index=False):
        x[g.game_id]={team_name(g.home_name):g.home_id,team_name(g.away_name):g.away_id}
    return x


def parse_team_box(data:list[dict], games:pd.DataFrame) -> tuple[pd.DataFrame, list]:
    by_game=_game_teams(games); records=[]; issues=[]
    for bundle in data:
        gid=str(bundle.get('id') or bundle.get('gameId') or '')
        if gid not in by_game: continue
        for team in bundle.get('teams') or []:
            raw=str(team.get('teamId') or team.get('team') or '')
            tid=raw if raw in by_game[gid].values() else by_game[gid].get(team_name(team.get('team')))
            if not tid:
                issues.append(f'TEAM_BOX_UNRESOLVED:{gid}:{team.get("team")}'); continue
            row={'game_id':gid,'team_id':tid,'box_present':1}
            # Keep only raw observed metrics as string-category num, no inferred meaning.
            for s in team.get('stats') or []:
                cat=str(s.get('category') or '')
                category=team_name(cat)
                if category in ('totalyards','rushingyards','passingyards','turnovers',
                                'firstdowns','totalpenalties','penalties','sacks',
                                'timesofpossession','fumbleslost','interceptions'):
                    val=num(s.get('stat'))
                    if pd.notna(val): row['box_'+category]=val
            records.append(row)
    return pd.DataFrame(records),issues


def parse_advanced(data:list[dict],games:pd.DataFrame) -> tuple[pd.DataFrame,list]:
    by_game=_game_teams(games); records=[]; issues=[]
    attrs={
        'ppa':('ppa',), 'success':('successRate',), 'explosive':('explosiveness',),
        'pass_ppa':('passingPlays','ppa'), 'rush_ppa':('rushingPlays','ppa'),
        'pass_success':('passingPlays','successRate'),
        'rush_success':('rushingPlays','successRate'),
        'passing_down_ppa':('passingDowns','ppa'),
        'standard_down_ppa':('standardDowns','ppa'),
        'line_yards':('lineYards',), 'stuff_rate':('stuffRate',),
        'plays':('plays',), 'drives':('drives',),
    }
    for s in data:
        gid=str(s.get('gameId') or '')
        if gid not in by_game: continue
        tid=by_game[gid].get(team_name(s.get('team')))
        if tid is None:
            issues.append(f'ADVANCED_UNRESOLVED:{gid}:{s.get("team")}');continue
        row={'game_id':gid,'team_id':tid,'advanced_present':1}
        for unit in ('offense','defense'):
            stats=s.get(unit) or {}
            for key,path in attrs.items():
                val=num(get_nested(stats,*path))
                if pd.notna(val): row[f'{unit}_{key}']=val
        records.append(row)
    return pd.DataFrame(records),issues


def parse_drives(data:list[dict],games:pd.DataFrame) -> tuple[pd.DataFrame,list]:
    by_game=_game_teams(games); rows=[]; issues=[]
    for d in data:
        gid=str(d.get('gameId') or '')
        if gid not in by_game: continue
        tid=by_game[gid].get(team_name(d.get('offense')))
        if tid is None:
            issues.append(f'DRIVE_UNRESOLVED:{gid}:{d.get("offense")}');continue
        pts=num(d.get('endOffenseScore'))-num(d.get('startOffenseScore'))
        plays=num(d.get('plays')); yards=num(d.get('yards'))
        # -ve score deltas and extraordinary play counts are invalid, NOT filled.
        if not np.isfinite(pts) or pts<0 or not np.isfinite(plays) or plays<0:
            issues.append(f'DRIVE_INVALID:{gid}:{d.get("id")}');continue
        rows.append({'game_id':gid,'team_id':tid,'drive_id':str(d.get('id')),
                     'drive_points':pts,'drive_plays':plays,'drive_yards':yards,
                     'drive_start_ytg':num(d.get('startYardsToGoal')),
                     'drive_scoring':int(pts>0),'drive_result':str(d.get('driveResult') or '')})
    return pd.DataFrame(rows),issues


def parse_plays(data:list[dict],games:pd.DataFrame) -> tuple[pd.DataFrame,list]:
    by_game=_game_teams(games); rows=[]; issues=[]
    for p in data:
        gid=str(p.get('gameId') or '')
        if gid not in by_game: continue
        tid=by_game[gid].get(team_name(p.get('offense')))
        if tid is None:
            issues.append(f'PLAY_UNRESOLVED:{gid}:{p.get("offense")}');continue
        # Play PPA is descriptive of this completed game; NEVER fed into same-game predictor.
        rows.append({'game_id':gid,'team_id':tid,'play_id':str(p.get('id')),
                     'play_ppa':num(p.get('ppa')),
                     'play_yards':num(p.get('yardsGained')),
                     'play_down':num(p.get('down')),
                     'play_distance':num(p.get('distance'))})
    return pd.DataFrame(rows),issues


def reduce_game_team(games:pd.DataFrame, raw:dict[str,list[dict]]) -> tuple[pd.DataFrame,dict]:
    sides=[]; lookup={}
    for g in games.itertuples(index=False):
        for kind,tid in [('home',g.home_id),('away',g.away_id)]:
            key=(g.game_id,tid)
            lookup[key]={'game_id':g.game_id,'team_id':tid,'is_home':int(kind=='home')}
    report={'source_rows':{k:len(v) for k,v in raw.items()},'issues':[]}
    for fam,fn in [('team_box',parse_team_box),('advanced',parse_advanced),
                   ('drives',parse_drives),('plays',parse_plays)]:
        parsed,problems=fn(raw.get(fam,[]),games)
        report['issues']+=problems
        if parsed.empty: continue
        if fam in ('team_box','advanced'):
            if parsed.duplicated(['game_id','team_id']).any():
                raise ValueError(f'Multiple {fam} records for same game-team; source reconciliation needed')
            derived=parsed
        else:
            agg={}
            if fam=='drives':
                agg={'drive_points':['mean','sum'],'drive_plays':['mean'],
                     'drive_yards':['mean'],'drive_start_ytg':['mean'],
                     'drive_scoring':['mean']}
            else:
                agg={'play_ppa':['mean'],'play_yards':['mean'],'play_down':['mean']}
            # Do not treat missing measurement as zero.
            derived=parsed.groupby(['game_id','team_id'],as_index=False).agg(agg)
            derived.columns=['game_id','team_id']+[
                ('drives_' if fam=='drives' else 'plays_')+ a.replace('drive_','').replace('play_','') + '_' + b
                for a,b in derived.columns.tolist()[2:]]
            counts=parsed.groupby(['game_id','team_id']).size().reset_index(name=fam+'_n')
            derived=derived.merge(counts,on=['game_id','team_id'],validate='one_to_one')
        for _,r in derived.iterrows():
            key=(str(r['game_id']),str(r['team_id']))
            if key not in lookup: continue
            for col in derived.columns:
                if col not in ('game_id','team_id'):
                    v=r[col]
                    if col in lookup[key] and pd.notna(v) and lookup[key][col]!=v:
                        raise ValueError(f'Contradictory source value {key} {col}')
                    lookup[key][col]=v
    for v in lookup.values(): sides.append(v)
    frame=pd.DataFrame(sides)
    report['coverage_by_family']={}
    for fam,signal in [('team_box','box_present'),('advanced','advanced_present'),
                       ('drives','drives_n'),('plays','plays_n')]:
        report['coverage_by_family'][fam]=int(frame[signal].notna().sum()) if signal in frame else 0
    return frame,report


def build_native_frame(raw:dict[str,list[dict]]) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    games=build_games(raw.get('games',[]))
    if games.empty: raise ValueError('No source games: fetch /games first')
    sides,report=reduce_game_team(games,raw)
    gold,labels=build_pregame_state(games,sides)
    report['games']=len(games)
    report['feature_games']=len(gold)
    report['scored_games']=int(np.isfinite(labels['home_margin']).sum())
    report['seasons']=sorted(games['season'].unique().tolist())
    report['gold_feature_names']=[c for c in gold if c not in ('game_id','season','week','kickoff')]
    return gold,labels,report


# Family grouping is deliberately independent of frozen Production V1 / Big Al.
FAMILY_FEATURES={
    'score_history':['margin_for','points_for','points_against','opp_prior_strength','adjusted_margin'],
    'team_box':['box_totalyards','box_rushingyards','box_passingyards','box_turnovers','box_firstdowns'],
    'drives':['drives_points_mean','drives_plays_mean','drives_start_ytg_mean',
              'drives_scoring_mean','drives_yards_mean'],
    'plays':['plays_ppa_mean','plays_yards_mean'],
    'advanced':['offense_ppa','defense_ppa','offense_success','defense_success',
                'offense_explosive','defense_explosive','offense_pass_ppa',
                'defense_pass_ppa','offense_rush_ppa','defense_rush_ppa',
                'offense_passing_down_ppa','defense_passing_down_ppa'],
}


def _weighted_prior(history: deque, feature:str, kickoff:pd.Timestamp) -> tuple[float,int]:
    obs=[]
    for date,season,value in history:
        if not np.isfinite(value.get(feature,float('nan'))): continue
        # Higher value to same-season, recent games, but all past eras usable.
        delta=max(0,(kickoff-date).total_seconds()/86400)
        wt=2**(-delta/360.0)
        obs.append((value[feature],wt))
    if not obs: return float('nan'),0
    denom=sum(w for _,w in obs)
    return float(sum(x*w for x,w in obs)/denom),len(obs)


def build_pregame_state(games:pd.DataFrame,sides:pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Pregame state NEVER uses a game's own box/drives/plays/score before its cutoff.

    In the absence of a verified final-publish timestamp, completed-game features only
    become eligible 24h after kickoff. Same-day started games never count.
    No after-the-fact season-stat snapshot or price is consulted.
    """
    sindex={(str(x.game_id),str(x.team_id)):x._asdict() for x in sides.itertuples(index=False)}
    histories=defaultdict(lambda:deque(maxlen=100))
    pending=[]; features=[]; targets=[]
    features_by_family=sorted({f for group in FAMILY_FEATURES.values() for f in group})
    for g in games.itertuples(index=False):
        cutoff=g.kickoff
        while pending and pending[0][0]<=cutoff:
            when,_,tid,year,measure=heapq.heappop(pending)
            histories[tid].append((when,year,measure))
        row={'game_id':g.game_id,'season':g.season,'week':g.week,
             'kickoff':cutoff.isoformat(),'neutral_site':g.neutral_site}
        for prefix,tid in [('home',g.home_id),('away',g.away_id)]:
            h=histories[tid]
            row[f'{prefix}_prior_games']=len(h)
            row[f'{prefix}_season_games']=sum(x[1]==g.season for x in h)
            for feature in features_by_family:
                value,available=_weighted_prior(h,feature,cutoff)
                row[f'{prefix}_{feature}']=value
                row[f'{prefix}_{feature}_n']=available
        # Disagreement features are native football matchups, not betting residuals.
        for feature in features_by_family:
            a,b=row[f'home_{feature}'],row[f'away_{feature}']
            row[f'diff_{feature}']=a-b if np.isfinite(a) and np.isfinite(b) else float('nan')
        for attack, defend, name in [('home','away','home'),('away','home','away')]:
            off=row.get(f'{attack}_offense_ppa',float('nan'))
            oppdef=row.get(f'{defend}_defense_ppa',float('nan'))
            row[f'{name}_off_vs_opp_def_ppa']=off-oppdef if np.isfinite(off) and np.isfinite(oppdef) else float('nan')
        features.append(row)
        hp,ap=g.home_points,g.away_points
        targets.append({'game_id':g.game_id,'season':g.season,
                        'home_margin': hp-ap if np.isfinite(hp) and np.isfinite(ap) else float('nan'),
                        'game_total': hp+ap if np.isfinite(hp) and np.isfinite(ap) else float('nan'),
                        'home_win':int(hp>ap) if np.isfinite(hp) and np.isfinite(ap) and hp!=ap else float('nan')})
        if not (np.isfinite(hp) and np.isfinite(ap)): continue
        publish_at=cutoff+pd.Timedelta(hours=24)
        for tid,opp,pts,opp_pts in [(g.home_id,g.away_id,hp,ap),(g.away_id,g.home_id,ap,hp)]:
            past=sindex.get((g.game_id,tid),{})
            measure={f:num(past.get(f)) for f in features_by_family}
            measure['margin_for']=pts-opp_pts
            measure['points_for']=pts
            measure['points_against']=opp_pts
            # Structural bonus: context of actual opponent at prior kickoff, preserving
            # opponent strength as it was before the game; no forward recomputation.
            pastopp=histories[opp]
            opponent_prior,n=_weighted_prior(pastopp,'margin_for',cutoff)
            if np.isfinite(opponent_prior):
                measure['opp_prior_strength']=opponent_prior
                measure['adjusted_margin']=(pts-opp_pts)+0.60*opponent_prior
            heapq.heappush(pending,(publish_at,len(pending)+len(features)*5,tid,g.season,measure))
    return pd.DataFrame(features),pd.DataFrame(targets)


def coverage_by_year(raw:dict[str,list[dict]], games:pd.DataFrame,
                     feature_frame:pd.DataFrame,labels:pd.DataFrame) -> pd.DataFrame:
    out=[]
    for yr,grp in games.groupby('season'):
        gg=feature_frame.loc[feature_frame.season==yr]
        l=labels.loc[labels.season==yr]
        rec={'season':int(yr),'games':len(grp),'scored_games':int(l.home_margin.notna().sum())}
        for fam in ('team_box','drives','plays','advanced'):
            src=raw.get(fam,[])
            if fam=='team_box':
                ids={str(x.get('id')) for x in src}
            else:
                ids={str(x.get('gameId')) for x in src}
            rec[f'{fam}_games_raw']=sum(grp.game_id.isin(ids))
            cols=[f'home_{z}' for z in FAMILY_FEATURES[fam] if z in ['points_for'] or f'home_{z}' in gg]
            rec[f'{fam}_pregame_fraction']=round(float(gg[cols].notna().any(axis=1).mean()),4) if cols and len(gg) else 0
        out.append(rec)
    return pd.DataFrame(out)


def write_research_exports(raw_root:str|Path, destination:str|Path,years:list[int]|None=None) -> dict:
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    raw={key:load_family(raw_root,key,years) for key in ('games','team_box','advanced','drives','plays')}
    feats,labels,report=build_native_frame(raw)
    games=build_games(raw['games'])
    coverage=coverage_by_year(raw,games,feats,labels)
    feats.to_csv(destination/'cfbd_native_pregame_features.csv',index=False)
    labels.to_csv(destination/'cfbd_native_outcomes_RESEARCH_ONLY.csv',index=False)
    coverage.to_csv(destination/'cfbd_native_season_coverage.csv',index=False)
    (destination/'cfbd_native_quality_report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    return {'output':str(destination),'games':len(games),'reports':list(report.keys())}
