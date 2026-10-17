"""NCAAF Core research: reconstruct per-team CFBD game metrics as strictly pregame context.

Game observations from CFBD_Game_* are *postgame source labels*. They are NEVER
returned to Core as predictors. Uses only earlier dated historical games to
build season-to-date, last 3, last 5 and previous-season averages, then joins
opponent prior metrics by the same physical game. Neutral sites are supported.

2026+ is sealed; no production authority, promotions, or BigQuery modifications.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any
import re
import numpy as np
import pandas as pd

PRE_GAME_CFBD_VERSION = "NCAAF_CFBD_STRICT_PRIOR_RECONSTRUCTION_20261017_V1"
MAX_HISTORICAL_YEAR = 2025
RESEARCH_ONLY = True


def _key(v: Any) -> str:
    if v is None or pd.isna(v): return ""
    return re.sub(r"[^a-z0-9]", "", str(v).strip().lower())


def forbidden_raw_cfbd(c: str) -> bool:
    """Identify same-game measurements, including misleading opponent mirrors."""
    s = str(c).lower()
    if not s.startswith('cfbd_'): return False
    if s.startswith('cfbd_opp_'): s = 'cfbd_' + s[len('cfbd_opp_'):]
    # CFBD_Game_* always refers to the contemporaneous box score unless it has
    # been *independently* reconstructed into a Season/RecentN/PrevSeason prior.
    if s.startswith(('cfbd_game_', 'cfbd_qb_game_')): return True
    if s.startswith(('cfbd_temperature','cfbd_dewpoint','cfbd_humidity',
                     'cfbd_precipitation','cfbd_snowfall','cfbd_windspeed',
                     'cfbd_weather_')): return True
    return False


def select_safe_cfbd_source_columns(columns: list[str]) -> list[str]:
    """Never join raw game observations into the research model frame."""
    return [c for c in columns if str(c).startswith('CFBD_') and not forbidden_raw_cfbd(c)]


def _mean_finite(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float('nan')


def _field_label(c: str, horizon: str, opponent: bool=False) -> str:
    return 'CFBD_' + ('Opp_' if opponent else '') + horizon + '_' + c + '_Prior'


def build_strict_pregame_cfbd(raw: pd.DataFrame, *, max_season: int=2025) -> tuple[pd.DataFrame,dict[str,Any]]:
    """Return exclusively prior-game numeric context and its source keys.

    Time rule: a completed observation is included only if observation date <
    game date. All matches played on a date are scored before any same-day
    history is committed. Missing games occupy recent-N slots as NaNs rather
    than reaching past the last N completed games to cherry-pick older values.

    Historical game completeness: explicit `Completed==1` when available;
    otherwise a historical (<=2025) game with at least one finite CFBD_Game_*
    measurement is counted as observed. This conservative proxy avoids treating
    empty/unscheduled rows as history. No games from 2026+ enter.
    """
    keys = ['Season','Game_ID','Game_Date','Team_Key','Opponent_Key']
    if raw is None or raw.empty:
        return pd.DataFrame(columns=keys), {'status':'NO_SOURCE','team_games':0,'features':0}
    missing=[c for c in keys if c not in raw.columns]
    if missing: raise ValueError(f'CFBD_PRE_GAME_MISSING_KEYS {missing}')
    z=raw.copy()
    z['Season']=pd.to_numeric(z['Season'],errors='coerce')
    z=z.loc[z['Season'].between(2013,min(int(max_season),MAX_HISTORICAL_YEAR))].copy()
    z['Game_Date']=pd.to_datetime(z['Game_Date'],errors='coerce',utc=True)
    if z['Game_Date'].isna().any():
        raise ValueError(f'CFBD_PRE_GAME_UNDATED_GAMES={int(z["Game_Date"].isna().sum())}')
    z['__date']=z['Game_Date'].dt.normalize()
    z['__team']=z['Team_Key'].map(_key); z['__opp']=z['Opponent_Key'].map(_key)
    z['__game']=z['Game_ID'].astype('string').fillna('').str.replace(r'\.0$','',regex=True).astype(str).str.strip()
    if z['__team'].eq('').any() or z['__opp'].eq('').any() or z['__game'].eq('').any():
        raise ValueError('CFBD_PRE_GAME_EMPTY_TEAM_OR_GAME_ID')
    cols=sorted(c for c in z.columns if str(c).startswith('CFBD_Game_'))
    cols=[c for c in cols if pd.api.types.is_numeric_dtype(z[c]) or pd.to_numeric(z[c],errors='coerce').notna().any()]
    if not cols: raise ValueError('CFBD_PRE_GAME_NO_GAME_METRIC_COLUMNS')
    # Unique game/team observations, and cannot reconstruct ambiguous same-date
    # double-headers without individually verified kickoff timestamps.
    if z.duplicated(['Season','__game','__team']).any():
        raise ValueError('CFBD_PRE_GAME_DUPLICATE_GAME_TEAM')
    if z.duplicated(['Season','__date','__team']).any():
        raise ValueError('CFBD_PRE_GAME_AMBIGUOUS_SAME_DATE_TEAM_GAMES')
    z=z.sort_values(['__date','Season','__game','__team'],kind='stable').reset_index(drop=True)
    mat=z[cols].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
    observed_metric=np.isfinite(mat).any(axis=1)
    if 'Completed' in z.columns:
        # Explicit incomplete games cannot update history. If completion was
        # not recorded at all, accept observed historical game metrics only.
        status=pd.to_numeric(z['Completed'],errors='coerce')
        completed=(status.eq(1) | (status.isna() & observed_metric)).to_numpy(dtype=bool)
    else:
        completed=observed_metric
    n=len(z); k=len(cols)
    data={}
    for horizon in ('Season','Recent3','Recent5','PrevSeason'):
        data[horizon]=np.full((n,k),np.nan,dtype=float)
    gp=np.zeros(n,dtype=float)
    # Group by team across seasons to allow previous-year context without
    # borrowing present-season observations. Strict chronological date batches.
    team_indices=z.groupby('__team',sort=False).indices
    for _, positions in team_indices.items():
        positions=sorted(map(int,positions),key=lambda i:(z['__date'].iat[i],z['__game'].iat[i]))
        prior_by_season: dict[int,list[np.ndarray]]=defaultdict(list)
        for d in sorted({z['__date'].iat[i] for i in positions}):
            group=[i for i in positions if z['__date'].iat[i]==d]
            for i in group:
                yr=int(z['Season'].iat[i]); hist=prior_by_season[yr]
                gp[i]=len(hist)
                if hist:
                    a=np.vstack(hist)
                    data['Season'][i]=np.array([_mean_finite(a[:,j]) for j in range(k)])
                    data['Recent3'][i]=np.array([_mean_finite(a[-3:,j]) for j in range(k)])
                    data['Recent5'][i]=np.array([_mean_finite(a[-5:,j]) for j in range(k)])
                ph=prior_by_season.get(yr-1,[])
                if ph:
                    a=np.vstack(ph)
                    data['PrevSeason'][i]=np.array([_mean_finite(a[:,j]) for j in range(k)])
            # Same-day scores cannot affect each other even if timestamps differ.
            for i in group:
                if completed[i]: prior_by_season[int(z['Season'].iat[i])].append(mat[i].copy())
    numeric={'CFBD_Rebuilt_Games_Prior':gp}
    for horizon in ('Season','Recent3','Recent5','PrevSeason'):
        for j,c in enumerate(cols):numeric[_field_label(c,horizon)]=data[horizon][:,j]

    # Current game opponent features must be the opponent's own *prior state*,
    # not the team's contemporaneous CFBD_Opp_Game values.
    opp_index={(int(z['Season'].iat[i]),z['__game'].iat[i],z['__team'].iat[i]):i for i in range(n)}
    match_idx=np.full(n,-1,dtype=int)
    for i in range(n):
        match_idx[i]=opp_index.get((int(z['Season'].iat[i]),z['__game'].iat[i],z['__opp'].iat[i]),-1)
        if match_idx[i]>=0:
            j=match_idx[i]
            if z['__opp'].iat[j]!=z['__team'].iat[i] or z['__date'].iat[j]!=z['__date'].iat[i]:
                raise ValueError('CFBD_PRE_GAME_INCONSISTENT_OPPONENT_PAIR')
    for horizon in ('Season','Recent3','Recent5','PrevSeason'):
        arr=np.full((n,k),np.nan,dtype=float)
        have=match_idx>=0
        arr[have]=data[horizon][match_idx[have]]
        for j,c in enumerate(cols):numeric[_field_label(c,horizon,opponent=True)]=arr[:,j]

    # Interpretable offense vs opposing defensive efficiency and vice-versa.
    # Do not claim schedule-strength opponent adjustment here; those belong to
    # a separately proven opponent-adjusted upstream model.
    for horizon in ('Season','Recent3','Recent5','PrevSeason'):
        for c in cols:
            if not c.startswith('CFBD_Game_Off_'):continue
            tail=c[len('CFBD_Game_Off_'):]
            dc='CFBD_Game_Def_'+tail
            if dc not in cols:continue
            team_c=_field_label(c,horizon); opp_c=_field_label(dc,horizon,True)
            name='CFBD_Pregame_Matchup_Off_vs_OppDef_'+tail+'_'+horizon
            numeric[name]=numeric[team_c]-numeric[opp_c]
    result=pd.concat([z[keys].reset_index(drop=True),pd.DataFrame(numeric,index=z.index)],axis=1)
    # Recompute independent expected priors and overwrite source precalculated
    # fields only inside the returned research-only table; not in BigQuery.
    diag={'status':'PASS','version':PRE_GAME_CFBD_VERSION,'team_games':n,
          'game_metrics':k,'pregame_numeric_fields':len(result.columns)-len(keys),
          'opponent_pairs':int((match_idx>=0).sum()),'completed_source_rows':int(completed.sum()),
          'seasons':sorted(set(z['Season'].astype(int))),'future_years_queried':False,
          'same_day_excluded':True,'production_mutation':False,
          'source_label_fields_blocked':len(cols)}
    return result,diag


def replace_or_attach_pregame(df: pd.DataFrame, rebuilt: pd.DataFrame) -> pd.DataFrame:
    """Augment Core source rows by a verified (season,game,team) join.

    Removes *all* raw game CFBD labels and all reconstructed-column names from
    the frame before merging so no stale or same-game sourced prior survives.
    """
    if rebuilt is None or rebuilt.empty:return df.copy()
    keys=['Season','Game_ID','Team_Key']
    if not set(keys).issubset(df.columns):
        raise ValueError('CFBD_PRE_GAME_SOURCE_JOIN_KEYS_MISSING')
    if rebuilt.duplicated(keys).any() or df.duplicated(keys).any():
        raise ValueError('CFBD_PRE_GAME_JOIN_AMBIGUOUS')
    feature_cols=[c for c in rebuilt.columns if c not in ('Season','Game_ID','Game_Date','Team_Key','Opponent_Key')]
    out=df.drop(columns=[c for c in df.columns if forbidden_raw_cfbd(c) or c in feature_cols],errors='ignore')
    result=out.merge(rebuilt[keys+feature_cols],on=keys,how='left',validate='one_to_one',sort=False)
    if len(result)!=len(df):raise RuntimeError('CFBD_PRE_GAME_ROW_COUNT_CHANGED')
    return result
