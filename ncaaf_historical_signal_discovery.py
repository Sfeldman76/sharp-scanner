"""Independent NCAAF historical signal discovery, shadow research only.

Uses strictly prior-game derived statistics plus explicitly certified pregame
columns. Features are discovered from 2013-2023 data, tested in chronological
OOF folds against the incumbent full-history score-based OOF forecasts, with
2024-2025 retrospective confirmation. No market-opening inputs or postgame
same-game fields. No 2026 labels, bets, promotion or production modification.
"""
from __future__ import annotations
from collections import defaultdict, deque
import re
from typing import Any
import numpy as np
import pandas as pd

VERSION = 'NCAAF_HISTORICAL_SIGNAL_DISCOVERY_20261017_V1'
HEADS = ('margin', 'total', 'h2h')
SELECT_YEARS = tuple(range(2018, 2024))
CONFIRM_YEARS = (2024, 2025)
MIN_TRAIN = 250
MAX_FIELDS_PER_FAMILY = 3
MAX_FAMILY_COUNT = 12

# Source fields are admitted only with an explicit as-of/pregame naming
# contract. Broad 'A_*', 'B_*' or 'CFBD_*' checks are NOT sufficient.
TRUSTED_SOURCE = re.compile(
    r'^(?:(?:A|B|Diff|Mean|Sum)_(?:RawSeason_|RawRecent3_|RawRecent5_|State_|Prior_|Prev_)|'
    r'H2H2_(?:Prior_|Recency_|Days_Since|Same_Home_Role)|'
    r'CFBD_(?:Opp_)?(?:Season|Recent3|Recent5|PrevSeason)_CFBD_Game_.*_Prior$)',
    re.IGNORECASE,
)
BLOCKED = re.compile(
    r'(opening|closing|open_|close_|market|moneyline|odds|spread|ats|clv|'
    r'betting|book|line_move|result|actual|final|postgame|target|label|'
    r'team_score|opponent_score|current_game|game_outcome|prediction|pt_meta)',
    re.IGNORECASE,
)
FAMILY_PATTERNS = (
    ('run_efficiency', r'(rush|running|rushing|ypa|ground)'),
    ('pass_efficiency', r'(pass|complet|sack|qb|air_yard)'),
    ('explosiveness', r'(explosive|big_play|bigplay|20plus|yards_per_play|_ypp)'),
    ('down_conversion', r'(third_down|3rd|fourth_down|4th|conversion)'),
    ('ball_security', r'(turnover|fumble|interception|giveaway|takeaway)'),
    ('scoring_efficiency', r'(redzone|red_zone|points_per|touchdown|scoring|off_pp|def_pp)'),
    ('drive_success', r'(success|ppa|epa|drive|havoc)'),
    ('sos_matchup', r'(strength|sos|opponent_adj|oppadj|gameadj|schedule|power)'),
    ('discipline_special_teams', r'(penalt|special_team|punt|field_goal|kickoff|return)'),
    ('game_context', r'(rest|travel|time_zone|elevation|home|rival|revenge|h2h)'),
)
DERIVED_FAMILY = 'historical_score_form'
MODERN_FAMILY = 'modern_cfbd_pregame'


def _num(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df: return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors='coerce').replace([np.inf, -np.inf], np.nan)


def _name(x: Any) -> str:
    if x is None or pd.isna(x): return ''
    return re.sub('[^a-z0-9]', '', str(x).lower())


def _identities(frame: pd.DataFrame, names: tuple[str, ...]) -> list[str]:
    s = pd.Series('', index=frame.index, dtype='string')
    for c in names:
        if c in frame:
            value=frame[c].astype('string').fillna('').map(_name)
            s=s.mask(s.eq(''),value)
    return s.fillna('').astype(str).tolist()


def _dates(frame: pd.DataFrame) -> pd.Series:
    s=pd.Series(pd.NaT,index=frame.index,dtype='datetime64[ns, UTC]')
    for c in ('Game_Date','Game_Start','Game_Start_Date','Start_Date','Date'):
        if c in frame:
            s=s.fillna(pd.to_datetime(frame[c],errors='coerce',utc=True))
    return s.dt.normalize()


def reconstruct_historical_score_form(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute game-side histories using *strictly previous calendar days*.

    Updating yesterday's opposing teams only after every game that day is
    scored blocks same-day leakage. Prior-season values are retained but are
    not mislabelled as current-season means. Unknown dates/teams stay null.
    """
    n=len(frame); sy=_num(frame,'Season').to_numpy(float)
    if np.isfinite(sy).any() and float(np.nanmax(sy))>=2026:
        raise ValueError('2026+ research labels are sealed')
    m=_num(frame,'Actual_Margin').to_numpy(float)
    t=_num(frame,'Actual_Total').to_numpy(float)
    dates=_dates(frame)
    team=_identities(frame,('Team_Norm','Team','Source_Team_Name','Home_Team_Norm'))
    opp=_identities(frame,('Opponent_Norm','Opponent','Opp_Norm','Away_Team_Norm'))
    columns=(
        'Hist_Points_For_Recent3_Diff','Hist_Points_Allowed_Recent3_Diff',
        'Hist_Points_For_Recent5_Diff','Hist_Points_Allowed_Recent5_Diff',
        'Hist_Margin_Recent3_Diff','Hist_Margin_Recent5_Diff',
        'Hist_Win_Rate_Recent5_Diff','Hist_Margin_Volatility_Recent5_Diff',
        'Hist_Recent3_vs_Season_Margin_Diff','Hist_Pregame_Scoring_Matchup',
        'Hist_Recent3_Combined_Points_Avg','Hist_PrevSeason_Margin_Diff',
        'Hist_Season_Margin_Diff','Hist_Recent3_vs_Season_Total',
    )
    out={c:np.full(n,np.nan,float) for c in columns}
    state=defaultdict(lambda:{'year':None,'games':[],'prev':[]})
    ready=[i for i in range(n) if np.isfinite(sy[i]) and 2013<=sy[i]<=2025
           and pd.notna(dates.iloc[i]) and team[i] and opp[i] and team[i]!=opp[i]]
    if 'Source_Game_ID' in frame and ready:
        ids=frame.loc[ready,'Source_Game_ID'].astype('string').fillna('').str.strip()
        pairs=frame.loc[ready,'Season'].astype(str)+'/'+ids
        meaningful=ids.ne('')&~ids.str.lower().isin(('nan','none','<na>'))
        if bool(pairs[meaningful].duplicated().any()):
            raise ValueError('DUPLICATE_PHYSICAL_GAME_ID_IN_FEATURE_DISCOVERY')
    # Earlier seasons and same-day rows must be chronologically stable.
    grouped=defaultdict(list)
    for i in ready: grouped[(int(sy[i]),dates.iloc[i])].append(i)
    prior_year=None
    for (year,day),indices in sorted(grouped.items()):
        if prior_year!=year:
            for v in state.values():
                if v['year']!=year:
                    v['prev']=v['games'] if v['year']==year-1 else []
                    v['games']=[];v['year']=year
            prior_year=year
        def history(k):
            x=state[k]
            if x['year']!=year:
                x['prev']=x['games'] if x['year']==year-1 else []
                x['games']=[];x['year']=year
            return x
        def avg(v,which,num):
            a=[r[which] for r in v[-num:]]
            return float(np.mean(a)) if a else np.nan
        def std(v,num):
            a=[r['margin'] for r in v[-num:]]
            return float(np.std(a,ddof=1)) if len(a)>=3 else np.nan
        for i in indices:
            a=history(team[i]);b=history(opp[i]);ta=a['games'];tb=b['games']
            values={
                'Hist_Points_For_Recent3_Diff':avg(ta,'pf',3)-avg(tb,'pf',3),
                'Hist_Points_Allowed_Recent3_Diff':avg(tb,'pa',3)-avg(ta,'pa',3),
                'Hist_Points_For_Recent5_Diff':avg(ta,'pf',5)-avg(tb,'pf',5),
                'Hist_Points_Allowed_Recent5_Diff':avg(tb,'pa',5)-avg(ta,'pa',5),
                'Hist_Margin_Recent3_Diff':avg(ta,'margin',3)-avg(tb,'margin',3),
                'Hist_Margin_Recent5_Diff':avg(ta,'margin',5)-avg(tb,'margin',5),
                'Hist_Win_Rate_Recent5_Diff':avg(ta,'win',5)-avg(tb,'win',5),
                'Hist_Margin_Volatility_Recent5_Diff':std(ta,5)-std(tb,5),
                'Hist_Recent3_vs_Season_Margin_Diff':(avg(ta,'margin',3)-avg(ta,'margin',len(ta))) - (avg(tb,'margin',3)-avg(tb,'margin',len(tb))) if ta and tb else np.nan,
                'Hist_Pregame_Scoring_Matchup':avg(ta,'pf',3)-avg(tb,'pa',3),
                'Hist_Recent3_Combined_Points_Avg':0.5*(avg(ta,'total',3)+avg(tb,'total',3)),
                'Hist_PrevSeason_Margin_Diff':avg(a['prev'],'margin',len(a['prev']))-avg(b['prev'],'margin',len(b['prev'])) if a['prev'] and b['prev'] else np.nan,
                'Hist_Season_Margin_Diff':avg(ta,'margin',len(ta))-avg(tb,'margin',len(tb)) if ta and tb else np.nan,
                'Hist_Recent3_vs_Season_Total':(avg(ta,'total',3)-avg(ta,'total',len(ta)))+(avg(tb,'total',3)-avg(tb,'total',len(tb))) if ta and tb else np.nan,
            }
            for k,v in values.items():out[k][i]=v
        # Only now commit results; never contaminate this calendar day's rows.
        for i in indices:
            if not (np.isfinite(m[i]) and np.isfinite(t[i])):continue
            a=history(team[i]);b=history(opp[i]);mar=float(m[i]);ttl=float(t[i]);pf=(ttl+mar)/2;pa=(ttl-mar)/2
            a['games'].append({'pf':pf,'pa':pa,'margin':mar,'total':ttl,'win':float(mar>0)})
            b['games'].append({'pf':pa,'pa':pf,'margin':-mar,'total':ttl,'win':float(mar<0)})
    rows=len(ready)
    byyear={int(y):{'rows':int(np.sum(sy==y)),
                   'valid_identity_date':int(sum(sy[i]==y for i in ready))}
            for y in sorted(set(sy[np.isfinite(sy)].astype(int)))}
    return pd.DataFrame(out,index=frame.index),{'source':'STRICTLY_PREVIOUS_COMPLETED_GAMES',
        'asof_calendar_day':True,'reconstructed_rows':rows,'total_rows':n,
        'by_year':byyear,'no_current_game_input':True,'features':len(columns)}


def _source_group(col: str) -> str | None:
    if not TRUSTED_SOURCE.match(str(col)) or BLOCKED.search(str(col)):
        return None
    if col.startswith('CFBD_'):
        # Modern CFBD source MUST have the strict pregamed shifted contract.
        return MODERN_FAMILY
    lower=col.lower()
    for label,pattern in FAMILY_PATTERNS:
        if re.search(pattern,lower):return label
    return None


def _coverage(v: pd.Series, years: np.ndarray, lo: int, hi: int) -> float:
    a=v[np.isfinite(years)&(years>=lo)&(years<=hi)]
    return float(a.notna().mean()) if len(a) else 0.0


def _family_candidates(frame: pd.DataFrame, already_used: list[str],
                       *, max_groups:int=MAX_FAMILY_COUNT) -> tuple[dict[str,list[str]],dict[str,dict]]:
    years=_num(frame,'Season').to_numpy(float)
    baseline=set(already_used)
    selected=defaultdict(list)
    meta={}
    prep=np.isfinite(years)&(years<=2023)
    for c in frame.columns:
        if c in baseline:continue
        if str(c).startswith('Hist_'):
            family=DERIVED_FAMILY
        else:
            family=_source_group(str(c))
        if family is None:continue
        v=_num(frame,c)
        valid=int(v[prep].notna().sum())
        if valid<100 or v[prep].nunique(dropna=True)<3:continue
        count_seasons=int(sum(v[years==y].notna().sum()>=20 for y in range(2013,2024)))
        early=_coverage(v,years,2013,2021)
        late=_coverage(v,years,2022,2023)
        if family==MODERN_FAMILY:
            if late<0.4:continue
        elif family==DERIVED_FAMILY:
            if count_seasons<4:continue
        elif count_seasons<4 or early<.25:
            # Recent-only family isn't full history. Do not silently delete
            # old games; evidence stays modern-specific unless separately proven.
            continue
        meta[c]={'family':family,'historical_coverage':round(early,4),
                 'modern_2022_2023_coverage':round(late,4),'observed_pre2024':valid,
                 'covered_seasons_pre2024':count_seasons,
                 'provenance':'INDEPENDENT_ASOF_HISTORY' if family==DERIVED_FAMILY else
                              'STRICT_MODERN_CFBD_PREGAME' if family==MODERN_FAMILY else
                              'EXISTING_PRE_GAME_FEATURE_CONTRACT'}
        selected[family].append(c)
    # One cluster of highly correlated variables cannot overwhelm other families.
    chosen={}
    for f,cols in selected.items():
        cols=sorted(cols,key=lambda c:(-meta[c]['observed_pre2024'],c))
        chosen[f]=cols[:MAX_FIELDS_PER_FAMILY]
    chosen=dict(sorted(chosen.items(),key=lambda item:(item[0]!=DERIVED_FAMILY,item[0]!=MODERN_FAMILY,item[0]))[:max_groups])
    return chosen,{c:meta[c] for cols in chosen.values() for c in cols}


def _head_loss(actual: np.ndarray, pred: np.ndarray, head: str) -> np.ndarray:
    if head=='h2h':
        q=np.clip(pred,1e-5,1-1e-5)
        y=(actual>0).astype(float)
        return -(y*np.log(q)+(1-y)*np.log1p(-q))
    return (actual-pred)**2


def _score(actual: np.ndarray, base: np.ndarray, candidate: np.ndarray, head: str, mask: np.ndarray) -> dict:
    mask=mask&np.isfinite(actual)&np.isfinite(base)&np.isfinite(candidate)
    if head=='h2h':mask &=~np.isclose(actual,0.)
    n=int(mask.sum())
    if n==0:return {'n':0,'metric':'logloss' if head=='h2h' else 'rmse','gain':None}
    a=_head_loss(actual[mask],base[mask],head);b=_head_loss(actual[mask],candidate[mask],head)
    gain=float(np.mean(a)-np.mean(b)) if head=='h2h' else float(np.sqrt(np.mean(a))-np.sqrt(np.mean(b)))
    return {'n':n,'metric':'logloss' if head=='h2h' else 'rmse','gain':gain,
            'baseline':float(np.mean(a)) if head=='h2h' else float(np.sqrt(np.mean(a))),
            'challenger':float(np.mean(b)) if head=='h2h' else float(np.sqrt(np.mean(b)))}


def _corrected_pvalues(rows: list[dict]) -> None:
    # Holm family-wise error control for each independently scored prediction head.
    order=sorted((r for r in rows if r['p_one_sided'] is not None), key=lambda x:x['p_one_sided'])
    m=len(order)
    running=0.
    for j,r in enumerate(order):
        running=max(running, min(1.,(m-j)*r['p_one_sided']))
        r['holm_adjusted_p']=round(running,5)


def _sign_flip_pvalue(actual,base,pred,head,mask,dates,*,repeats=399,seed=101)->float|None:
    mask=mask&np.isfinite(actual)&np.isfinite(base)&np.isfinite(pred)
    if head=='h2h':mask &=~np.isclose(actual,0.)
    indices=np.flatnonzero(mask)
    if len(indices)<250:return None
    x=_head_loss(actual[indices],base[indices],head)-_head_loss(actual[indices],pred[indices],head)
    # Date blocks preserve correlated matchups within each Saturday's slate.
    labels=dates.iloc[indices].dt.strftime('%Y-%m-%d').fillna('unknown').to_numpy()
    groups=defaultdict(float)
    for k,v in zip(labels,x):groups[k]+=float(v)
    vals=np.asarray(list(groups.values()),float)
    if len(vals)<15:return None
    observed=float(vals.sum());rng=np.random.default_rng(seed)
    signed=rng.choice(np.array([-1.,1.]),size=(repeats,len(vals)))
    null=(signed*vals).sum(axis=1)
    return float((1+np.count_nonzero(null>=observed))/(repeats+1))


def _fit_residual(X: np.ndarray, y: np.ndarray, b: np.ndarray, train: np.ndarray,
                  valid: np.ndarray, head: str)->np.ndarray|None:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    tr=train&np.isfinite(y)&np.isfinite(b)
    va=valid&np.isfinite(b)
    if head=='h2h':tr &=~np.isclose(y,0.)
    if int(tr.sum())<MIN_TRAIN or int(va.sum())==0:return None
    obs=np.isfinite(X[tr]).sum(axis=0)>=max(30,int(.2*tr.sum()))
    obs &= np.array([np.unique(X[tr,j][np.isfinite(X[tr,j])]).size>=3 for j in range(X.shape[1])])
    if not np.any(obs):return None
    a=X[tr][:,obs];v=X[va][:,obs]
    imp=SimpleImputer(strategy='median');a=imp.fit_transform(a);v=imp.transform(v)
    scale=StandardScaler();a=scale.fit_transform(a);v=scale.transform(v)
    target=(y[tr]>0).astype(float) if head=='h2h' else y[tr]
    residual=target-b[tr]
    model=Ridge(alpha=150.)
    model.fit(a,residual)
    delta=model.predict(v)
    return np.clip(delta,-.2,.2) if head=='h2h' else np.clip(delta,-8.,8.)


def run_historical_signal_discovery(frame:pd.DataFrame, foundation_predictions:dict[str,np.ndarray],
                                    base_features:list[str],*, log_func=print)->dict:
    """Discover families on pre-2024 seasons. Never auto-change a live model."""
    if frame.empty:raise ValueError('empty historical frame')
    if not frame.index.equals(pd.RangeIndex(len(frame))):raise ValueError('row index must be contiguous')
    season=_num(frame,'Season').to_numpy(float)
    if np.isfinite(season).any() and np.nanmax(season)>2025:
        raise ValueError('2026+ research data prohibited')
    for h in HEADS:
        if h not in foundation_predictions or len(foundation_predictions[h])!=len(frame):
            raise ValueError('baseline OOF arrays must align with game rows')
    margin=_num(frame,'Actual_Margin').to_numpy(float)
    total=_num(frame,'Actual_Total').to_numpy(float)
    dates=_dates(frame)
    hist,audit=reconstruct_historical_score_form(frame)
    working=pd.concat([frame,hist],axis=1)
    groups,fields=_family_candidates(working,base_features)
    if not groups:return {'status':'NO_ELIGIBLE_HISTORICAL_FAMILIES','source_audit':audit,'discovered':0,
                          'authority':0,'automatic_promotion':False}
    # Year-forward, out-of-fold correction to year-forward baseline predictions.
    results={}; rows_for_adjustment=defaultdict(list)
    for family,cols in groups.items():
        X=working[cols].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        lane=MODERN_FAMILY if family==MODERN_FAMILY else 'historical'
        byhead={}
        for h in HEADS:
            y=total if h=='total' else margin
            base=np.asarray(foundation_predictions[h],float)
            out=np.full(len(frame),np.nan,float)
            fold_records=[]
            for yr in (*SELECT_YEARS,*CONFIRM_YEARS):
                # A 2022-only source must qualify using its actually available
                # historical era. Earlier missing CFBD values are NOT zeros.
                # All-year foundation predictions remain unchanged.
                start_year=2022 if lane==MODERN_FAMILY else 2014
                if lane==MODERN_FAMILY and yr<2023:
                    continue
                tr=(season<yr)&(season>=start_year)&np.isfinite(base)
                va=season==yr
                # Do not add 2024/25 outcomes to the 2023 selection fold.
                correction=_fit_residual(X,y,base,tr,va,h)
                if correction is None:
                    fold_records.append({'year':yr,'status':'INSUFFICIENT_PRIOR_OOF_OR_FEATURES'});continue
                pred=base[va]+.35*correction
                if h=='h2h':pred=np.clip(pred,1e-5,1-1e-5)
                out[va]=pred
                result=_score(y,base,out,h,va)
                fold_records.append({'year':yr,'status':'SCORED',**result})
            selection=np.isin(season,SELECT_YEARS if lane=='historical' else [2023])
            confirmation=np.isin(season,CONFIRM_YEARS)
            select_score=_score(y,base,out,h,selection)
            conf_score=_score(y,base,out,h,confirmation)
            valid_folds=[x for x in fold_records if x['year'] in SELECT_YEARS and x.get('status')=='SCORED']
            modern_counts=(
                {'selection_train_years':[2022],'selection_validation_years':[2023],
                 '2022_observed_rows':int(np.isfinite(X[season==2022]).any(axis=1).sum()),
                 '2023_observed_rows':int(np.isfinite(X[season==2023]).any(axis=1).sum())}
                if lane==MODERN_FAMILY else {})
            positive_years=sum(x.get('gain',0)>0 for x in valid_folds)
            p=_sign_flip_pvalue(y,base,out,h,selection,dates,seed=101+len(cols))
            rec={'family':family,'lane':lane,'head':h,'features':cols,
                 'selection_years':list(SELECT_YEARS if lane=='historical' else [2023]),
                 'era_training_audit':modern_counts,
                 'selection':select_score,'positive_years':positive_years,
                 'eligible_selection_years':len(valid_folds),
                 'p_one_sided':round(p,5) if p is not None else None,
                 'holm_adjusted_p':None,
                 'retrospective_2024_2025':conf_score,
                 'per_year':fold_records,
                 'decision':'EXPLORATORY_NOT_PROMOTED'}
            byhead[h]=rec
            if lane=='historical' and p is not None:
                rows_for_adjustment[h].append(rec)
        results[family]={'lane':lane,'features':cols,'heads':byhead}
    for h,rows in rows_for_adjustment.items():
        _corrected_pvalues(rows)
    candidate_summary={}
    for h in HEADS:
        eligible=[]
        for family,item in results.items():
            r=item['heads'][h];s=r['selection']
            if (item['lane']=='historical' and s.get('gain') is not None and
                s['gain']>({'margin':.06,'total':.06,'h2h':.002}[h]) and
                r['eligible_selection_years']>=4 and r['positive_years']>=3 and
                r.get('holm_adjusted_p') is not None and r['holm_adjusted_p']<=.1):
                r['decision']='HISTORICAL_RESEARCH_CANDIDATE_FOR_PROSPECTIVE_SHADOW'
                eligible.append(r)
        eligible.sort(key=lambda z:z['selection']['gain'],reverse=True)
        candidate_summary[h]={'selection_passed':[x['family'] for x in eligible],
                              'top_family':eligible[0]['family'] if eligible else None,
                              'prospective_authority':0}
    for family,item in results.items():
        for h,rec in item['heads'].items():
            sc=rec['selection'];conf=rec['retrospective_2024_2025']
            log_func('[NCAAF-HISTORICAL-SIGNAL-FAMILY] '+str({
                'family':family,'head':h,'source':item['lane'],
                'selection_n':sc['n'],'selection_gain':sc.get('gain'),
                'holm_p':rec['holm_adjusted_p'],
                'confirm_n':conf['n'],'confirm_gain':conf.get('gain'),
                'modern_selection_train_years':rec.get('era_training_audit',{}).get('selection_train_years'),
                'decision':rec['decision']}))
    report={'status':'RESEARCH_COMPLETE_NO_PROMOTION','version':VERSION,'years':sorted({int(y) for y in season[np.isfinite(season)]}),
            'opening_lines_required':False,'production_mutation':False,
            'source_audit':audit,'source_fields':fields,'families':results,
            'family_count':len(groups),'tested_hypotheses':sum(len(x['heads']) for x in results.values()),
            'selected_pre2024_only':True,'confirmation_2024_2025_is_retrospective':True,
            'multiple_testing':'PER_HEAD_HOLM_ADJUSTED_BLOCK_SIGN_FLIP_P_VALUES',
            'selection':candidate_summary,
            'promoted':False,'authority':0,
            'market_models_unchanged':True,
            'note':'Any selection is hypothesis-generating; requires frozen prospective testing before live consideration.'}
    log_func('[NCAAF-HISTORICAL-SIGNAL-CONTRACT] '+str({
        'status':report['status'],'families':len(groups),
        'features':len(fields),'hypotheses':report['tested_hypotheses'],
        'head_selection':candidate_summary,'authority':0}))
    return report
