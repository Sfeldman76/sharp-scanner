"""Restore advanced NCAAF Miner atoms ONLY when replayed pregame definitions agree.

Source: already-loaded CFBD game-side cache. No API calls, new market prices,
postgame features for current fixtures, or changes to frozen research contracts.
Absent raw game metrics and unverifiable formulas remain MISSING_LIVE_ATOMS.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# (historical pregame column, permitted actual-game metric source, minimum earlier games)
RAW_SPECS = (
    ('CFBD_QB_Game_PPA_SeasonPrior', ('CFBD_QB_Game_PPA',), 2),
    ('CFBD_Season_CFBD_Game_Def_Havoc_Prior', ('CFBD_Game_Def_Havoc',), 3),
    ('CFBD_Season_CFBD_Game_Off_Havoc_Allowed_Prior', ('CFBD_Game_Off_Havoc_Allowed',), 3),
    ('CFBD_Season_CFBD_Game_Off_Sack_Rate_Prior', ('CFBD_Game_Off_Sack_Rate',), 3),
    ('CFBD_Season_Net_Success_Prior', ('CFBD_Game_Net_Success',), 3),
)


def _numeric(frame, name):
    return pd.to_numeric(frame[name], errors='coerce') if name in frame else pd.Series(np.nan, index=frame.index, dtype=float)


def _team(frame, names, key):
    out=pd.Series('',index=frame.index,dtype='object')
    for name in names:
        if name in frame:
            v=frame[name].map(lambda x: key(x) if pd.notna(x) else '')
            out=out.where(out.ne(''),v)
    return out.fillna('')


def _replay_metric(hist, raw, prior, *, min_reconciliation=8, tolerance=0.001):
    """Cross-check point-in-time definition with observed pregame archive values.

    `prior` must be saved shift(1) pregame, `raw` must be actual same-game.
    A mismatched aggregation (weighted averages, roster change, etc.) is rejected.
    """
    records=[]
    latest={}
    for (season, team),group in hist.groupby(['_sy','_team'],sort=False):
        if not team or pd.isna(season): continue
        group=group.sort_values(['_dt','_gid'],kind='stable')
        values=[]
        for _,row in group.iterrows():
            actual=row.get(raw)
            declared=row.get(prior)
            if len(values)>=1 and pd.notna(declared):
                records.append((float(np.mean(values)),float(declared),len(values)))
            if pd.notna(actual) and np.isfinite(float(actual)):
                values.append(float(actual))
        if values:
            latest[(int(season),team)]={'value':float(np.mean(values)), 'n':len(values)}
    if len(records)<min_reconciliation:
        return None,{'status':'HOLD_NOT_ENOUGH_HISTORICAL_RECONCILIATION','pairs':len(records)}
    # The actual definition must reproduce the historical pregame values.
    error=np.abs(np.array([r[0]-r[1] for r in records],dtype=float))
    match=np.isfinite(error) & (error<=tolerance)
    if not match.all():
        return None,{'status':'HOLD_HISTORICAL_DEFINITION_MISMATCH','pairs':len(records),
                     'matched':int(match.sum()),'max_error':float(np.max(error))}
    return latest,{'status':'REPLAY_VERIFIED','pairs':len(records),'max_error':float(np.max(error))}


def _verify_matchup_formula(hist, target='CFBD_Matchup_Net_Success_Adv',
                             base='CFBD_Season_Net_Success_Prior',min_pairs=8,tolerance=.001):
    """Prove that saved matchup advantage equals team minus opponent priors.

    Exactly one row per (season,game,team) is required. Never infer a definition
    from names alone, nor derive a fresh rule/threshold.
    """
    if target not in hist or base not in hist:return False,{'status':'HOLD_NO_MATCHUP_FORMULA_FIELDS'}
    sample=[]
    keyed=hist.set_index(['_sy','_gid','_team'],drop=False)
    for _,row in hist.iterrows():
        if not row['_opp'] or not row['_gid']:continue
        try: opp=keyed.loc[(row['_sy'],row['_gid'],row['_opp'])]
        except KeyError:continue
        if isinstance(opp,pd.DataFrame):continue
        t=row[base]; o=opp[base]; observed=row[target]
        if pd.notna(t) and pd.notna(o) and pd.notna(observed):
            sample.append((float(t)-float(o),float(observed)))
    if len(sample)<min_pairs:return False,{'status':'HOLD_NO_FORMULA_RECONCILIATION','pairs':len(sample)}
    delta=np.abs(np.array([a-b for a,b in sample]))
    if not np.isfinite(delta).all() or not (delta<=tolerance).all():
        return False,{'status':'HOLD_MATCHUP_FORMULA_MISMATCH','pairs':len(sample), 'max_error':float(np.nanmax(delta))}
    return True,{'status':'REPLAY_VERIFIED','pairs':len(sample),'max_error':float(delta.max())}


def restore_advanced_asof(live, cfbd_history, *, team_key, asof=None, min_reconciliation=8):
    """Repair only verified missing atoms with prior complete-game evidence.

    Returns (frame, audit); leaves existing fields unchanged. Original absence is
    NOT interpreted as false. Avoids same-game or 2026 outcome selection leakage.
    """
    out=live.copy()
    audit={'status':'NO_VERIFIED_SOURCE','rows':len(out),'filled_cells':0,'filled_by_field':{},
           'replay':{},'held_fields':{},'source':'EXISTING_CFBD_CACHE_COMPLETE_GAMES',
           'pt_linemidweek':'BLOCKED_MISSING_NAMED_SOURCE',
           'stat_edge':'BLOCKED_UNCERTIFIED_INDEPENDENT_STAT'}
    if not isinstance(cfbd_history,pd.DataFrame) or cfbd_history.empty or out.empty:
        return out,audit
    now=pd.to_datetime(asof if asof is not None else pd.Timestamp.now(tz='UTC'),utc=True)
    hist=cfbd_history.copy()
    required=['Season','Game_Date','Game_ID']
    if any(c not in hist for c in required):
        audit['status']='HOLD_MISSING_SOURCE_IDENTITY'; return out,audit
    hist['_sy']=_numeric(hist,'Season').astype('Int64')
    hist['_dt']=pd.to_datetime(hist['Game_Date'],errors='coerce',utc=True)
    hist['_team']=_team(hist,('Team_Key','Team'),team_key)
    hist['_opp']=_team(hist,('Opponent_Key','Opponent'),team_key)
    hist['_gid']=hist['Game_ID'].astype('string').fillna('').str.replace(r'\.0$','',regex=True)
    # We require completed past fixtures, not same-day games with uncertain final.
    # Game_Date often stores midnight rather than actual kickoff.
    hist=hist.loc[hist['_sy'].notna() & hist['_team'].ne('') & hist['_opp'].ne('') &
                  hist['_gid'].ne('') & hist['_dt'].notna() &
                  hist['_dt'].lt(now.normalize())].copy()
    if hist.empty:
        audit['status']='HOLD_NO_PRIOR_COMPLETED_GAME_SIDES';return out,audit
    # Distinct physical games; ambiguous repeated records fail rather than arbitrarily choose.
    unique_cols=['_sy','_gid','_team']
    duplicates=hist.duplicated(unique_cols,keep=False)
    audit['ambiguous_source_sides']=int(duplicates.sum())
    hist=hist.loc[~duplicates].copy().sort_values(['_sy','_team','_dt','_gid'],kind='stable')
    if hist.empty:
        audit['status']='HOLD_AMBIGUOUS_SOURCE';return out,audit
    live_sy=_numeric(out,'Season')
    live_date=pd.to_datetime(out.get('Game_Start',out.get('Game_Date',pd.Series(pd.NaT,index=out.index))),errors='coerce',utc=True)
    home=_team(out,('Team_Norm','Home_Team_Norm','Home_Team'),team_key)
    away=_team(out,('Opponent_Norm','Away_Team_Norm','Away_Team'),team_key)
    audit['historical_complete_sides']=len(hist)
    # Prevent future-date or wrong-season injection for current games.
    eligible=(live_date.gt(now) & live_sy.notna() & home.ne('') & away.ne(''))
    patched={}
    candidates={}
    for prior,raw_names,minimum in RAW_SPECS:
        raw=next((c for c in raw_names if c in hist.columns and _numeric(hist,c).notna().any()),None)
        if raw is None or prior not in hist.columns:
            audit['held_fields'][prior]='RAW_AND_PRIOR_PAIR_NOT_AVAILABLE';continue
        hist[raw]=_numeric(hist,raw)
        hist[prior]=_numeric(hist,prior)
        verified,proof=_replay_metric(hist,raw,prior,min_reconciliation=min_reconciliation)
        audit['replay'][prior]=proof
        if verified is None:
            audit['held_fields'][prior]=proof['status'];continue
        candidates[prior]=verified
        for prefix,names in (('',home),('CFBD_Opp_',away)):
            dest=prior if not prefix else 'CFBD_Opp_'+prior[len('CFBD_'):]
            existing=_numeric(out,dest)
            new=existing.copy()
            filled=0
            for j in range(len(out)):
                if not eligible.iloc[j] or pd.notna(existing.iloc[j]):continue
                state=verified.get((int(live_sy.iloc[j]),names.iloc[j]))
                if state and state['n']>=minimum:
                    new.iloc[j]=state['value'];filled+=1
            if filled:patched[dest]=new
            audit['filled_by_field'][dest]=filled
            audit['filled_cells']+=filled
    # Derived matchup only if its exact definition is proven historically.
    matchup_ok,form=_verify_matchup_formula(hist,min_pairs=min_reconciliation)
    audit['replay']['CFBD_Matchup_Net_Success_Adv']=form
    dest='CFBD_Matchup_Net_Success_Adv'
    if matchup_ok and 'CFBD_Season_Net_Success_Prior' in candidates:
        existing=_numeric(out,dest);new=existing.copy();filled=0
        source=candidates['CFBD_Season_Net_Success_Prior']
        for j in range(len(out)):
            if not eligible.iloc[j] or pd.notna(existing.iloc[j]):continue
            team=source.get((int(live_sy.iloc[j]),home.iloc[j]))
            opp=source.get((int(live_sy.iloc[j]),away.iloc[j]))
            if team and opp and min(team['n'],opp['n'])>=3:
                new.iloc[j]=team['value']-opp['value'];filled+=1
        if filled:patched[dest]=new
        audit['filled_by_field'][dest]=filled;audit['filled_cells']+=filled
    else:audit['held_fields'][dest]=form['status']
    if patched:
        out=pd.concat([out.drop(columns=list(patched),errors='ignore'),
                       pd.DataFrame(patched,index=out.index)],axis=1)
    audit['status']='PASS_VERIFIED_RECONSTRUCTION' if audit['filled_cells'] else 'HOLD_NO_REPLAY_VERIFIED_FIELDS'
    return out,audit
