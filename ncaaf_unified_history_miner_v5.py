"""Permanent NCAAF full-history cross-brain Miner research engine (NO BET AUTHORITY).

Builds new pregame *candidate* conjunctions directly from the full native atom
universe (Miner/Pathi/Big Al STYLE/CFBD/rich-market/PT) rather than from
previously published winning mechanism IDs. Fits/freeze candidates on seasons
strictly before each evaluation season. No current-year outcomes in selection.

Historical feature/quote/provider vintage is unverified; even forward-year folds
are retrospective simulations, NOT independently certified backtests. Missing
original Big Al pick records and market snapshots are never fabricated.
"""
from __future__ import annotations
from collections import defaultdict, Counter
from hashlib import sha256
from itertools import combinations
from datetime import datetime, timezone
import io, json, traceback
import numpy as np
import pandas as pd

VERSION = 'NCAAF_UNIFIED_FULL_HISTORY_MINER_V5_1_20261025'
RESEARCH_ROOT = 'research/ncaaf/v2/unified_full_history_miner/runs'
CATEGORY_KEYS = ('MINER_CONTEXT','PATHI_STYLE','BIG_AL_STYLE','CFBD','RICH_MARKET','PT_EXTERNAL','CORE_MODEL')


def category(family, name):
    s = (str(family)+' '+str(name)).upper()
    if 'BIGAL' in s or 'BIG_AL' in s: return 'BIG_AL_STYLE'
    if 'PATHI' in s or 'KEY_VALUE' in s or 'KEY_JOURNEY' in s: return 'PATHI_STYLE'
    if any(x in s for x in ('SHARP_SOFT','MARKET_PATH','MARKET_PRICE','MARKET_ROLE','OBS_SPREAD_MARKET','OBS_TOTAL_MARKET', 'KEY_NUMBER','CROSS_MARKET','MARKET_KEY','MARKET_REVERSAL','LINE_MOVE', 'TOTAL_MOVED', 'SHARP_MOVE')): return 'RICH_MARKET'
    if 'PT_' in s or 'EXTERNAL_' in s or 'META_PT' in s: return 'PT_EXTERNAL'
    if 'CFBD' in s: return 'CFBD'
    if 'MODEL_STATE' in s or 'CORE_' in s: return 'CORE_MODEL'
    return 'MINER_CONTEXT'


def fingerprint(market, direction, names):
    canonical = (str(market).lower(),str(direction).upper(),tuple(sorted(names)))
    return 'UNIFIED-'+sha256(json.dumps(canonical).encode()).hexdigest()[:24].upper()


def _string_field(frame, choices):
    for name in choices:
        if name in frame:
            return frame[name].astype('string').fillna('').str.strip().to_numpy(dtype=str)
    return np.full(len(frame), '', dtype=str)


def _num_field(frame, choices):
    for name in choices:
        if name in frame:
            return pd.to_numeric(frame[name],errors='coerce').to_numpy(dtype=float)
    return np.full(len(frame),np.nan)


def _source_rows(games,seasons):
    n=len(games)
    sy=np.asarray(seasons,dtype=float)
    if len(sy)!=n: raise ValueError('UNIFIED_SEASON_ROW_ALIGNMENT')
    ids=_string_field(games,('Source_Game_ID','Game_ID','game_id','CFBD_Game_ID'))
    teams=_string_field(games,('Team','Team_Name','Home_Team'))
    opps=_string_field(games,('Opponent','Opponent_Name','Away_Team'))
    dates=_string_field(games,('Game_Date','Game_Start','Kickoff_UTC','Date'))
    home=_num_field(games,('Is_Home',))
    if not np.isin(home[np.isfinite(home)],[0.,1.]).all() or not np.isfinite(home).any():
        raise ValueError('UNIFIED_SIDE_ORIENTATION_UNAVAILABLE')
    valid=(np.isfinite(sy)&(sy>=2013)&(sy<=2025)&np.isfinite(home)&
           (np.char.str_len(ids.astype(str))>0)&(np.char.str_len(teams.astype(str))>0)&
           (np.char.str_len(opps.astype(str))>0))
    keys=np.asarray([f'{int(y) if np.isfinite(y) else 0}|{i}' for y,i in zip(sy,ids)],object)
    # Match by the source's exact season+game id, never by score, selected side or market.
    unique, group_index=np.unique(keys[valid],return_inverse=True)
    local_indices=np.flatnonzero(valid)
    all_group=np.full(n,-1,dtype=int)
    all_group[local_indices]=group_index
    num_groups=len(unique)
    home_rows=np.bincount(group_index,weights=(home[valid]==1).astype(float),minlength=num_groups)
    away_rows=np.bincount(group_index,weights=(home[valid]==0).astype(float),minlength=num_groups)
    # Fail closed for incomplete or more than one physical row per side.
    sound=(home_rows==1)&(away_rows==1)
    return {'season':sy,'ids':ids,'teams':teams,'opponents':opps,'dates':dates,'home':home,
            'valid':valid,'group':all_group,'keys':unique,'sound':sound,'group_count':num_groups,
            'side_rows':int(valid.sum()),'sound_games':int(sound.sum())}


def _make_atom_bank(resolver,market,n,year_mask,*, cap_per_family=12):
    seen=set(); bank=[]; invalid=[]
    for raw in resolver(market):
        name=str(raw.get('name') or '').strip()
        family=str(raw.get('family') or '').strip()
        if not name or name in seen: continue
        seen.add(name)
        # Fail closed on unavailable snapshot values. Casting float NaN directly
        # to bool evaluates True in NumPy, silently fabricating rule firings.
        try:
            values=np.asarray(raw['mask'])
            if values.ndim!=1 or len(values)!=n:
                raise ValueError('UNIFIED_ATOM_ROW_ALIGNMENT:'+name)
            if values.dtype.kind=='b':
                mask=values.astype(bool,copy=False)
            elif values.dtype.kind in ('i','u','f'):
                finite=np.isfinite(values)
                if np.any(finite & ~np.isin(values,[0,1])):
                    raise TypeError('NON_BINARY_NUMERIC_MASK')
                mask=finite & (values==1)
                if not finite.all():
                    invalid.append((name,'MISSING_MASK_VALUES:'+str(int((~finite).sum()))))
            else:
                missing=np.asarray(pd.isna(values),dtype=bool)
                present=values[~missing]
                def _bit(x):
                    if isinstance(x,(bool,np.bool_)):
                        return bool(x)
                    if isinstance(x,(int,float,np.integer,np.floating)) and np.isfinite(x) and x in (0,1):
                        return bool(x)
                    raise TypeError('NON_BINARY_MASK_VALUE')
                mapped=np.asarray([_bit(x) for x in present],dtype=bool)
                mask=np.zeros(n,dtype=bool)
                mask[~missing]=mapped
                if missing.any():
                    invalid.append((name,'MISSING_MASK_VALUES:'+str(int(missing.sum()))))
        except ValueError as exc:
            if str(exc).startswith('UNIFIED_ATOM_ROW_ALIGNMENT:'):
                raise
            invalid.append((name,'NON_BOOL_MASK'));continue
        except (TypeError,OverflowError):
            invalid.append((name,'NON_BOOL_MASK'));continue
        source=category(family,name)
        # Old CORE_MODEL atoms are not V5.5 independent evidence and must not
        # leak old OOF predictions into this new discovery lane.
        if source=='CORE_MODEL':
            invalid.append((name,'LEGACY_CORE_MODEL_ATOM_EXCLUDED'))
            continue
        count=int((mask&year_mask).sum())
        # A missing snapshot is UNKNOWN, not an observed zero.
        if count==0: invalid.append((name,'NO_POSITIVE_COVERAGE'));continue
        bank.append({'name':name,'family':family,'source':source,'mask':mask,'count':count,
                     'description':str(raw.get('description') or '')})
    # Avoid feature-threshold clones dominating the candidate cap. Selection here
    # uses source availability and historical frequency ONLY, never outcomes.
    families=defaultdict(list)
    for a in bank: families[(a['source'],a['family'])].append(a)
    trimmed=[]
    for family, members in sorted(families.items()):
        members=sorted(members,key=lambda a:(-a['count'],a['name']))
        admitted=[]
        for a in members:
            # Exactly identical masks are the same economic situation.
            if any(np.array_equal(a['mask'][year_mask], b['mask'][year_mask]) for b in admitted):
                invalid.append((a['name'],'DUPLICATE_EXACT_MASK'))
                continue
            admitted.append(a)
            if len(admitted)>=cap_per_family:break
        trimmed.extend(admitted)
    # Exact cross-source duplicates are one scenario, not independent brains.
    # Global, deterministic source ordering avoids counting the same binary
    # firing pattern as Miner + Pathi/Big Al/market confirmations.
    independent=[]
    for a in trimmed:
        if any(np.array_equal(a['mask'][year_mask],b['mask'][year_mask]) for b in independent):
            invalid.append((a['name'],'DUPLICATE_MASK_CROSS_FAMILY'))
            continue
        independent.append(a)
    return independent,invalid,bank


def _safe_mask(mask, n, label):
    """Normalize pandas/nullable/list-backed historical predicates before indexing.

    This is a strict data contract, not a truthiness conversion: NaN is never a
    system firing and malformed shapes fail explicitly rather than receiving an
    uninformative NumPy list/array indexing TypeError.
    """
    vals=np.asarray(mask)
    if vals.ndim!=1 or len(vals)!=n:
        raise ValueError(f'UNIFIED_ROW_MASK_ALIGNMENT:{label}:shape={vals.shape} expected={n}')
    if vals.dtype.kind=='b': return vals.astype(bool,copy=False)
    if vals.dtype.kind in 'iuf':
        good=np.isfinite(vals)
        if np.any(good & ~np.isin(vals,[0,1])):
            raise ValueError(f'UNIFIED_NON_BINARY_MASK:{label}')
        return good & (vals==1)
    missing=np.asarray(pd.isna(vals),dtype=bool)
    good=np.zeros(n,dtype=bool)
    for idx in np.flatnonzero(~missing):
        v=vals[idx]
        if v is True or isinstance(v,np.bool_) and v:
            good[idx]=True
        elif v is False or isinstance(v,np.bool_) and not v:
            continue
        elif isinstance(v,(int,float,np.integer,np.floating)) and np.isfinite(v) and v in (0,1):
            good[idx]=bool(v)
        else:
            raise ValueError(f'UNIFIED_NON_BINARY_MASK:{label}:row={idx}')
    return good


def _event_stats(mask, direction, rowmeta, result, *, index_limit=None):
    """Exactly one physical-game outcome; conflicting source-side fires abstain."""
    group=np.asarray(rowmeta['group'],dtype=np.int64)
    side=np.asarray(rowmeta['home'],dtype=float)
    good=(group>=0)&_safe_mask(mask,len(group),'event_rule')
    if index_limit is not None: good &=_safe_mask(index_limit,len(group),'event_window')
    ix=np.flatnonzero(good)
    if not len(ix):return 0,0,0,0, np.array([],int)
    gi=group[ix]; target_side=np.where(side[ix]==1,1,-1)
    if direction=='FADE':target_side=-target_side
    home_votes=np.bincount(gi,weights=(target_side==1).astype(float),minlength=rowmeta['group_count'])
    away_votes=np.bincount(gi,weights=(target_side==-1).astype(float),minlength=rowmeta['group_count'])
    selected=np.flatnonzero(((home_votes>0)^(away_votes>0))&rowmeta['sound'])
    targets=np.where(home_votes[selected]>0,1,-1)
    # Outcome orientation is home. result is +1 home WIN, -1 home LOSS, 0 push.
    actual=np.asarray(result,dtype=float)[selected]
    evaluable=np.isfinite(actual)&(actual!=0)
    decisions=actual[evaluable]*targets[evaluable]
    return int((decisions>0).sum()),int((decisions<0).sum()),int((~evaluable).sum()),int(len(selected)),selected


def _group_outcomes(games,rowmeta,market):
    actual_margin=_num_field(games,('Actual_Margin',))
    actual_total=_num_field(games,('Actual_Total',))
    sp=_num_field(games,('Consensus_Open_Spread','Opening_Spread'))
    total=_num_field(games,('Consensus_Open_Total','Opening_Total'))
    raw=(actual_margin+sp) if market=='spreads' else (actual_total-total if market=='totals' else actual_margin)
    # Convert each per-team perspective to canonical HOME outcome.
    oriented=np.where(rowmeta['home']==1,raw,-raw) if market!='totals' else raw
    keys=rowmeta['group']; result=np.full(rowmeta['group_count'],np.nan)
    hvals=defaultdict(list)
    for i in np.flatnonzero((keys>=0)&np.isfinite(oriented)):
        hvals[int(keys[i])].append(float(oriented[i]))
    contradictions=0
    for key,values in hvals.items():
        if len(values)!=2 or not rowmeta['sound'][key] or abs(values[0]-values[1])>0.01:
            contradictions+=1;continue
        val=values[0]
        result[key]=(1 if val>0 else -1 if val<0 else 0)
    return result, contradictions


def _evaluate(mask, direction, meta, outcome, year_mask, min_n=35):
    w,l,p,all_games,_=_event_stats(mask,direction,meta,outcome,index_limit=year_mask)
    n=w+l
    if n<min_n: return None
    rate=(w+10)/(n+20)  # beta shrinkage; ranked within training window only
    lift=rate-.5
    return {'wins':w,'losses':l,'pushes':p,'games':all_games,'n':n,
            'rate':w/n if n else None, 'shrunken_lift':lift,
            'rank_score':lift*np.sqrt(min(n,350)/150.)}


def _overlap_duplicate(a,b,train_mask):
    x=np.asarray(a,dtype=bool)&np.asarray(train_mask,dtype=bool)
    y=np.asarray(b,dtype=bool)&np.asarray(train_mask,dtype=bool)
    both=int((x&y).sum()); union=int((x|y).sum())
    return union>0 and both/union>=.97


def _core_group_arrays(meta,coremap):
    total=meta['group_count']
    side=np.zeros(total,dtype=np.int8)
    ats=np.full(total,np.nan,dtype=float)
    edge=np.full(total,np.nan,dtype=float)
    for idx,text in enumerate(meta['keys']):
        yr,sid=str(text).split('|',1)
        ref=coremap.get((int(yr),sid))
        if not ref:continue
        s,res,p=ref
        if s not in ('HOME','AWAY') or not np.isfinite(p) or p<2:continue
        side[idx]=1 if s=='HOME' else -1
        ats[idx]={'WIN':1.,'LOSS':-1.,'PUSH':0.}.get(res,np.nan)
        edge[idx]=p
    return side,ats,edge


def _conditional_core_train(mask,direction,meta,train,core_arrays,min_n=25):
    """Use ONLY prior-season matched CORE candidates for support/veto discovery.

    A contradictory rule can add value by filtering CORE's *losing* cohort. This
    is a shadow research ranking; selecting it does not grant live veto power.
    """
    side,ats,edge=core_arrays
    if not np.any(side):return None
    group=np.asarray(meta['group'],dtype=np.int64)
    train_rows=(group>=0)&_safe_mask(train,len(group),'core_train_window')
    good=train_rows&_safe_mask(mask,len(group),'core_conditional')
    ix=np.flatnonzero(good)
    if not len(ix):return None
    gi=group[ix]
    target=np.where(meta['home'][ix]==1,1,-1)
    if direction=='FADE':target=-target
    home_votes=np.bincount(gi,weights=(target==1).astype(float),minlength=meta['group_count'])
    away_votes=np.bincount(gi,weights=(target==-1).astype(float),minlength=meta['group_count'])
    selected=np.flatnonzero(((home_votes>0)^(away_votes>0))&meta['sound'])
    voted=np.where(home_votes[selected]>0,1,-1)
    valid=(side[selected]!=0)&np.isfinite(ats[selected])&(ats[selected]!=0)
    supported=selected[valid&(voted==side[selected])]
    vetoed=selected[valid&(voted==-side[selected])]
    # Baseline comes from the same strictly prior-season CORE candidate pool.
    eligible_groups=np.unique(group[train_rows])
    pool=eligible_groups[(side[eligible_groups]!=0)&np.isfinite(ats[eligible_groups])&
                         (ats[eligible_groups]!=0)]
    baseline=(int((ats[pool]>0).sum())+10)/(len(pool)+20) if len(pool) else .5
    candidates=[]
    for purpose,eligible in (('SUPPORT',supported),('VETO',vetoed)):
        core_w=int((ats[eligible]>0).sum())
        core_l=int((ats[eligible]<0).sum())
        n=core_w+core_l
        if n<min_n:continue
        # Incremental improvement versus the correct prior-period baseline:
        # support wants a higher win rate, veto wants a higher *loss* rate.
        desirable=core_w if purpose=='SUPPORT' else core_l
        rate=(desirable+10)/(n+20)
        benchmark=baseline if purpose=='SUPPORT' else 1-baseline
        lift=rate-benchmark
        candidates.append({'core_train_wins':core_w,'core_train_losses':core_l,
            'core_train_n':n,'core_train_shrunken_lift':lift,
            'core_train_baseline_winrate':baseline,
            'core_train_purpose':purpose,
            'rank_score':lift*np.sqrt(min(n,350)/150.)})
    return max(candidates,key=lambda z:z['rank_score']) if candidates else None


def _rank_with_core(ev,mask,direction,meta,train,core_arrays):
    if ev is None:return None
    if core_arrays is not None:
        conditional=_conditional_core_train(mask,direction,meta,train,core_arrays)
        if conditional:
            # When a genuine prior-season CORE cohort exists, optimizing
            # conditional CORE win rate supersedes standalone ATS selection.
            ev={**ev,**conditional,'rank_score':conditional['rank_score'],
                'selection_basis':'PRIOR_SEASON_V55_CORE_'+conditional['core_train_purpose']}
            return ev
    return {**ev,'selection_basis':'PRIOR_SEASON_STANDALONE_ATS'}


def _select_for_fold(atom_bank,meta,outcome,market,eval_year,*,
                     core_arrays=None,seed_limit=80,beam_width=32,max_depth=4,min_train_n=40):
    yrs=meta['season']; train=(yrs<eval_year)&(yrs>=2013)&meta['valid']
    if len(set(yrs[train].astype(int)))<3:return [],0
    # One-sided outcome *direction* is part of every recipe. Search both.
    singles=[]
    for a in atom_bank:
        for direction in ('PLAY_ON','FADE'):
            score=_rank_with_core(_evaluate(a['mask'],direction,meta,outcome,train,min_n=min_train_n),
                                   a['mask'],direction,meta,train,core_arrays)
            if score:
                singles.append({'atoms':(a['name'],),'sources':frozenset((a['source'],)),
                                'families':frozenset((a['family'],)),'mask':a['mask'],
                                'direction':direction,'score':score})
    # Atoms with source siblings are allowed in *separate* candidate rules but
    # cannot manufacture independent support by stacking inside a conjunction.
    eligible=sorted(singles,key=lambda r:(-r['score']['rank_score'],r['atoms'],r['direction']))
    best=[]; by_source=Counter()
    for r in eligible:
        src=next(iter(r['sources']))
        if by_source[src]>=max(8,seed_limit//4): continue
        if any(r['direction']==b['direction'] and _overlap_duplicate(r['mask'],b['mask'],train)
               and r['sources']==b['sources'] for b in best): continue
        best.append(r);by_source[src]+=1
        if len(best)>=seed_limit:break
    source_lookup={a['name']:a for a in atom_bank}
    # Search only the best prior-year atoms per family, preserving source breadth.
    supports=best[:];beam=best[:beam_width]
    candidates=list(best);seen={(r['atoms'],r['direction']) for r in best};evaluated=len(singles)
    for depth in range(2,max_depth+1):
        extension=[]
        for base in beam:
            for item in supports:
                if item['direction']!=base['direction']:continue
                if base['sources']&item['sources']:continue
                if base['families']&item['families']:continue
                name=item['atoms'][0]
                if name in base['atoms']:continue
                names=tuple(sorted(base['atoms']+(name,)))
                key=(names,base['direction'])
                if key in seen:continue
                seen.add(key)
                mask=base['mask']&source_lookup[name]['mask']
                # require material narrowing beyond parent rather than duplicate.
                if _overlap_duplicate(mask,base['mask'],train):continue
                ev=_rank_with_core(_evaluate(mask,base['direction'],meta,outcome,train,min_n=min_train_n),
                                   mask,base['direction'],meta,train,core_arrays)
                evaluated+=1
                if not ev: continue
                # Real interaction must improve the parent training posterior;
                # still a research hypothesis only, never evidence of OOS edge.
                if ev['rank_score'] < base['score']['rank_score']+.005: continue
                extension.append({'atoms':names,'sources':base['sources']|item['sources'],
                                  'families':base['families']|item['families'],
                                  'mask':mask,'direction':base['direction'],'score':ev})
        extension=sorted(extension,key=lambda r:(-r['score']['rank_score'],r['atoms']))
        chosen=[]
        for r in extension:
            if any(_overlap_duplicate(r['mask'],s['mask'],train) and r['direction']==s['direction'] for s in chosen):continue
            chosen.append(r)
            if len(chosen)>=beam_width:break
        if not chosen:break
        candidates.extend(chosen)
        beam=chosen
    return sorted(candidates,key=lambda r:(-r['score']['rank_score'],r['atoms'],r['direction']))[:120], evaluated


def _normalize_key(s):
    from ncaaf_miner_identity_crosswalk_v2 import key
    return key(s)


def _build_core_map(games,seasons,core):
    """Strict school+game+season matching. No market/outcome-based joining."""
    if core is None or core.empty:return {}, {'status':'CORE_NOT_AVAILABLE','matched':0}
    required={'game_id','season','home_team','away_team','pick_side','ats_result','edge_points','kickoff_utc'}
    if required-set(core):raise ValueError('UNIFIED_CORE_MISSING_FIELDS:'+str(sorted(required-set(core))))
    from ncaaf_miner_identity_crosswalk_v2 import build_crosswalk
    universe=core.rename(columns={'home_team':'home_team_key','away_team':'away_team_key','kickoff_utc':'game_date_key'}).copy()
    universe=universe[['season','game_id','home_team_key','away_team_key','game_date_key']].drop_duplicates()
    ids=_string_field(games,('Source_Game_ID','Game_ID','game_id','CFBD_Game_ID'))
    team=_string_field(games,('Team','Team_Name','Home_Team'))
    opp=_string_field(games,('Opponent','Opponent_Name','Away_Team'))
    date=_string_field(games,('Game_Date','Game_Start','Kickoff_UTC','Date'))
    raw=pd.DataFrame({'season':np.asarray(seasons).astype(int),'source_game_id':ids,
                      'source_team':team,'source_opponent':opp,'game_date_raw':date}).drop_duplicates()
    cross,audit=build_crosswalk(raw,universe)
    cross=cross[cross.identity_status.eq('MATCHED')].copy()
    coreidx={str(r.game_id):(r.pick_side,r.ats_result,float(r.edge_points)) for r in core.itertuples()}
    mapped=defaultdict(set)
    for row in cross.itertuples(index=False):
        mapped[(int(row.season),str(row.source_game_id))].add(str(row.game_id))
    out={k:coreidx[next(iter(ids))] for k,ids in mapped.items() if len(ids)==1 and next(iter(ids)) in coreidx}
    audit.update({'status':'MATCHED_RESEARCH_ONLY','mapped_games':len(out),'ambiguous_core_ids':sum(len(v)>1 for v in mapped.values())})
    return out,audit


def _fold_rule_core_stats(mask,direction,meta,coremap,year):
    year_mask=(meta['season']==year)&meta['valid']
    # Group source rows by physical game; leave self-contradicting rules out.
    dummy=np.full(meta['group_count'],np.nan)
    _,_,_,_,which=_event_stats(mask,direction,meta,dummy,index_limit=year_mask)
    positive=np.flatnonzero(_safe_mask(mask,len(year_mask),'fold_core') & year_mask & (meta['group']>=0))
    groups=defaultdict(set)
    for i in positive:
        target=('HOME' if meta['home'][i]==1 else 'AWAY')
        if direction=='FADE':target='AWAY' if target=='HOME' else 'HOME'
        groups[int(meta['group'][i])].add(target)
    wins=losses=pushes=unknown=opposed=agree=0
    for idx in which:
        yr,sid=str(meta['keys'][idx]).split('|',1)
        ref=coremap.get((int(yr),sid))
        if not ref: unknown+=1;continue
        core_side,core_outcome,edge=ref
        if not np.isfinite(edge) or edge<2 or core_side not in ('HOME','AWAY'):continue
        rule_side=next(iter(groups[idx]))
        if rule_side!=core_side:opposed+=1;continue
        agree+=1
        if core_outcome=='WIN':wins+=1
        elif core_outcome=='LOSS':losses+=1
        elif core_outcome=='PUSH':pushes+=1
    return {'core_support_games':agree,'core_support_wins':wins,'core_support_losses':losses,
            'core_support_pushes':pushes,'core_opposition_games':opposed,'core_unmapped_games':unknown}


def run_full_history(games, seasons, atom_resolver, *, core_games=None, existing_catalog=None,
                     markets=('spreads','totals','h2h'), first_test_year=2017,
                     last_test_year=2025, max_depth=4, progress=None):
    """All available 2013–2025 historical seasons. Bounded season-forward search.

    Features may not be point-in-time certified: this is an honest shadow,
    with source coverage and missing inputs exposed separately from nonfires.
    """
    def emit(phase, market='', **metrics):
        if progress is not None:
            progress({'phase':phase,'market':market,'metrics':metrics,'research_only':True,'bet_authority':0})
    emit('SOURCE_IDENTITY_START')
    meta=_source_rows(games,seasons)
    emit('SOURCE_IDENTITY_PASS',sound_games=meta['sound_games'],side_rows=meta['side_rows'])
    emit('CORE_CROSSWALK_START')
    coremap,coreaudit=_build_core_map(games,seasons,core_games)
    core_arrays=_core_group_arrays(meta,coremap)
    emit('CORE_CROSSWALK_PASS',core_matches=len(coremap))
    catalog_defs = {}
    for market, registry in (existing_catalog or {}).items():
        for m in (registry or {}).get('mechanism_families', []):
            names=tuple(str(a).strip().upper() for a in m.get('representative_conditions',[]) if str(a).strip())
            if names:
                fid=fingerprint(market,str(m.get('direction','')),names)
                catalog_defs[fid]=str(m.get('mechanism_id') or '')
    fold_records=[];ranked=[];coverage=[];windows=[];future=[];retested=[];ledger=[]
    game_home_row=np.full(meta['group_count'],-1,dtype=np.int64)
    for _j in np.flatnonzero(meta['valid'] & (meta['home']==1)):
        _group=int(meta['group'][_j])
        if _group>=0 and meta['sound'][_group]:game_home_row[_group]=int(_j)
    for market in markets:
        emit('MARKET_START',market)
        try: outcome,contradictions=_group_outcomes(games,meta,market)
        except Exception as e:
            coverage.append({'market':market,'status':'TARGET_HOLD','reason':type(e).__name__+':'+str(e)})
            continue
        # Preserve the archived price as evidence, NOT as an authenticated
        # as-of betting quote. Never silently substitute a close/open.
        market_price=(_num_field(games,('Consensus_Open_Spread','Opening_Spread'))
                      if market=='spreads' else
                      _num_field(games,('Consensus_Open_Total','Opening_Total'))
                      if market=='totals' else np.full(len(games),np.nan))
        historical=meta['valid']&(meta['season']<=last_test_year)
        # Resolve each market atom universe once; fold-level dedup still uses
        # strictly pre-evaluation-year masks/frequencies.
        emit('ATOM_RESOLUTION_START',market)
        cached_market_atoms=atom_resolver(market)
        emit('ATOM_RESOLUTION_PASS',market,raw_atoms=len(cached_market_atoms))
        bank,invalid,raw_bank=_make_atom_bank(lambda _m:cached_market_atoms,market,len(games),historical)
        emit('ATOM_BANK_PASS',market,atoms=len(bank),invalid=len(invalid))
        by_src=Counter(a['source'] for a in bank)
        for c in CATEGORY_KEYS:
            coverage.append({'market':market,'category':c,'available_atoms':by_src[c],
                             'status':'AVAILABLE' if by_src[c] else 'MISSING_OR_UNEVALUABLE',
                             'original_historical_source_timestamp_verified':False,
                             'actual_big_al_pick_ledger_verified':False,
                             'rich_market_quote_timestamp_verified':False})
        coverage.append({'market':market,'category':'TARGET','available_atoms':int(np.isfinite(outcome).sum()),
                         'status':'RETROSPECTIVE_ONLY','contradictory_or_incomplete_games':contradictions})
        for year in range(max(2013,first_test_year),min(2025,last_test_year)+1):
            baseline=[]
            if market=='spreads':
                for (cy,sourceid),(side,core_ats_result,edge) in coremap.items():
                    if cy==year and np.isfinite(edge) and edge>=2 and side in ('HOME','AWAY'):
                        if core_ats_result in ('WIN','LOSS','PUSH'):
                            baseline.append(core_ats_result)
            baseline_w=baseline.count('WIN');baseline_l=baseline.count('LOSS')
            baseline_rate=baseline_w/(baseline_w+baseline_l) if baseline_w+baseline_l else None
            # Rebuild availability, duplication and threshold clones using ONLY
            # the prior training seasons. Future feature frequencies cannot
            # silently choose which atoms survive the candidate cap.
            year_bank,_,_=_make_atom_bank(lambda _m:cached_market_atoms,market,len(games),
                            meta['valid']&(meta['season']<year))
            emit('FOLD_SEARCH_START',market,year=year,atoms=len(year_bank))
            rules,examined=_select_for_fold(year_bank,meta,outcome,market,year,
                core_arrays=core_arrays if market=='spreads' else None,max_depth=max_depth)
            windows.append({'market':market,'test_season':year,'train_end_season':year-1,
                            'train_start_season':2013,'evaluated_candidate_conditions':examined,
                            'selected_frozen_rules':len(rules),'selection_uses_test_season_outcomes':False,
                            'independently_certified':False,'bet_authority':0})
            emit('FOLD_SEARCH_PASS',market,year=year,frozen_rules=len(rules),tested_candidates=examined)
            foldmask=(meta['season']==year)&meta['valid']
            for ix,r in enumerate(rules):
                w,l,p,n,selected=_event_stats(r['mask'],r['direction'],meta,outcome,index_limit=foldmask)
                rid=fingerprint(market,r['direction'],r['atoms'])
                row={'rule_id':rid,'market':market,'test_season':year,'train_end_season':year-1,
                     'direction':r['direction'],'condition_atoms':' AND '.join(r['atoms']),
                     'catalog_exact_duplicate_id':catalog_defs.get(rid,''),
                     'underlying_sources':'|'.join(sorted(r['sources'])),
                     'depth':len(r['atoms']),'frozen_rank':ix+1,
                     'train_wins':r['score']['wins'],'train_losses':r['score']['losses'],
                     'train_shrunken_lift':round(r['score']['shrunken_lift'],6),
                     'selection_basis':r['score']['selection_basis'],
                     'prior_core_support_n':r['score'].get('core_train_n',0),
                     'prior_core_support_shrunken_lift':r['score'].get('core_train_shrunken_lift'),
                     'prior_core_discovery_purpose':r['score'].get('core_train_purpose'),
                     'prior_core_baseline_winrate':r['score'].get('core_train_baseline_winrate'),
                     'test_wins':w,'test_losses':l,'test_pushes_or_ungraded':p,
                     'test_unique_games':n,'test_nonpush_winrate':w/(w+l) if w+l else None,
                     'historical_vintage_certified':False,'holdout_independent_of_present_day_feature_selection':False,
                     'bet_authority':0}
                if market=='spreads':
                    row.update(_fold_rule_core_stats(r['mask'],r['direction'],meta,coremap,year))
                    row['core_year_baseline_wins']=baseline_w
                    row['core_year_baseline_losses']=baseline_l
                    row['core_year_baseline_winrate']=baseline_rate
                    support_n=row['core_support_wins']+row['core_support_losses']
                    row['core_support_vs_year_baseline_pp']=(
                        100*(row['core_support_wins']/support_n-baseline_rate)
                        if support_n and baseline_rate is not None else None)
                    row['incremental_provenance']='RETROSPECTIVE_UNVERIFIED_QUOTES_AND_NONINDEPENDENT_RECIPE'
                fold_records.append(row)
                # One physical game per rule: a reconstruction ledger, never an
                # archived actual betting recommendation. Unknown quote vintage
                # is explicit; conflicting side fires were already abstained.
                if selected.size:
                    gi=np.asarray(meta['group'],dtype=np.int64)
                    eligible_rows=np.flatnonzero((gi>=0)&r['mask']&foldmask)
                    by_game=defaultdict(set)
                    for j in eligible_rows:
                        side='HOME' if meta['home'][j]==1 else 'AWAY'
                        if r['direction']=='FADE':side='AWAY' if side=='HOME' else 'HOME'
                        by_game[int(gi[j])].add(side)
                    for game_idx in selected:
                        year_id,source_id=str(meta['keys'][game_idx]).split('|',1)
                        voted=by_game.get(int(game_idx),set())
                        if len(voted)!=1:continue
                        decision_side=next(iter(voted))
                        c=coremap.get((int(year_id),source_id)) if market=='spreads' else None
                        core_side,core_result,core_edge=(c if c else ('','',''))
                        if c and np.isfinite(c[2]) and c[2]>=2 and core_side in ('HOME','AWAY'):
                            relationship='SUPPORT' if decision_side==core_side else 'VETO_OR_OPPOSITION'
                        else:
                            relationship='CORE_UNAVAILABLE_OR_NOT_CANDIDATE'
                        home_row=int(game_home_row[game_idx])
                        archived_price=float(market_price[home_row]) if home_row>=0 else float('nan')
                        kickoff=str(meta['dates'][home_row]) if home_row>=0 else ''
                        actual=float(outcome[game_idx])
                        decided=(1 if decision_side=='HOME' else -1)*actual if np.isfinite(actual) else np.nan
                        ledger.append({'market':market,'test_season':year,'source_game_id':source_id,
                            'source_identity':str(meta['keys'][game_idx]),'rule_id':rid,
                            'frozen_rule_rank':ix+1,'direction':r['direction'],
                            'rule_sources':'|'.join(sorted(r['sources'])),
                            'rule_decision_side':decision_side,
                            'kickoff_date_unverified':kickoff,
                            'archived_market_open_price_unverified':archived_price,
                            'retrospective_rule_result':('WIN' if decided>0 else 'LOSS' if decided<0 else 'PUSH' if decided==0 else 'UNKNOWN'),
                            'core_relationship':relationship,'core_side':core_side,
                            'core_historical_result':core_result,'core_edge_points':core_edge,
                            'rule_selected_before_test_year':True,
                            'market_quote_asof_verified':False,
                            'actual_original_signal_archive_verified':False,
                            'point_in_time_feature_certified':False,
                            'bet_authority':0})
        # 2026+ outcomes explicitly not processed. Freeze next-season recipe using
        # all historic data available at completion, not 2026 results.
        emit('NEXT_YEAR_FREEZE_START',market,atoms=len(bank))
        final,examined=_select_for_fold(bank,meta,outcome,market,2026,
            core_arrays=core_arrays if market=='spreads' else None,max_depth=max_depth)
        for rank,r in enumerate(final):
            rid=fingerprint(market,r['direction'],r['atoms'])
            if rid in catalog_defs:
                retested.append({'rule_id':rid,'market':market,
                                 'existing_mechanism_id':catalog_defs[rid],
                                 'status':'EXISTING_CATALOG_DEFINITION_RETESTED_NOT_NEW',
                                 'bet_authority':0})
                continue
            future.append({'rule_id':rid,
                           'market':market,'direction':r['direction'],'condition_atoms':' AND '.join(r['atoms']),
                           'sources':'|'.join(sorted(r['sources'])),'depth':len(r['atoms']),
                           'selection_end_season':2025,'train_games':r['score']['games'],
                           'train_shrunken_lift':r['score']['shrunken_lift'],
                           'selection_basis':r['score']['selection_basis'],
                           'prior_core_discovery_purpose':r['score'].get('core_train_purpose'),
                           'requires_original_quote_asof_for_live_use':True,
                           'current_2026_fires_not_certified':True,'bet_authority':0})
        emit('MARKET_PASS',market,fold_rows=sum(x['market']==market for x in fold_records),future_rules=len(final))
    folds=pd.DataFrame(fold_records)
    if not folds.empty:
        # Each year selected from past only; do not pooled-rank winners using
        # target year outcomes and then call results an independent holdout.
        for _,group in folds.groupby(['market','rule_id'],sort=False):
            r=group.sort_values('test_season')
            ranked.append({'market':r.iloc[0]['market'],'rule_id':r.iloc[0]['rule_id'],
                           'first_frozen_test_season':int(r.test_season.min()),
                           'tested_seasons':len(r),'test_wins':int(r.test_wins.sum()),
                           'test_losses':int(r.test_losses.sum()),
                           'test_unique_game_firings_sum':int(r.test_unique_games.sum()),
                           'source_categories':r.iloc[0]['underlying_sources'],
                           'depth':r.iloc[0]['depth'],'test_result_used_to_choose_rule':False,
                           'retrospective_unverified_feature_vintage':True,'bet_authority':0})
    audit={'version':VERSION,'status':'COMPLETE_RESEARCH_ONLY_NOT_CERTIFIED',
           'historical_seasons':[int(x) for x in sorted(set(meta['season'][meta['valid']].astype(int)))],
           'side_rows_with_identity':meta['side_rows'],'two_sided_source_games':meta['sound_games'],
           'cross_brain_market_atom_lanes':sorted(set((r['category'] for r in coverage if r.get('available_atoms',0)>0))),
           'markets':list(markets),'max_interaction_depth':max_depth,
           'season_forward_recipe_rows':len(fold_records),'distinct_frozen_rule_ids':len(set(x['rule_id'] for x in fold_records)),
           'core_link':coreaudit,'core_conditional_training_used_when_prior_sample_sufficient':True,
           'price_timestamp_verified':False,
           'historical_big_al_actual_pick_source':False,
           '2026_outcomes_used_for_selection':False,'test_year_outcomes_used_for_selection':False,
           'future_year_atom_frequency_used_for_fold_selection':False,
           'legacy_core_oof_atoms_excluded':True,
           'present_day_feature_definition_vintage_certified':False,
           'full_holdout_independence_certified':False,
           'rule_discovery_multiple_testing_promotion_cleared':False,
           'known_existing_catalog_exact_rules_omitted_from_new_paths':len(retested),
           'source_coverage_not_a_no_fire':True,
           'historical_decision_ledger_is_retrospective_only':True,
           'historical_decision_ledger_rows':len(ledger),
           'production_mutation':False,'automatic_promotion':False,'bet_authority':0}
    return {'audit_v5':audit, 'feature_source_coverage_v5':pd.DataFrame(coverage),
            'season_forward_windows_v5':pd.DataFrame(windows),
            'season_forward_frozen_rules_v5':folds,
            'season_forward_summary_v5':pd.DataFrame(ranked),
            'new_full_history_paths_v5':pd.DataFrame(future),
            'existing_catalog_retested_v5':pd.DataFrame(retested),
            'retrospective_game_decision_ledger_v5':pd.DataFrame(ledger)}


def load_current_v55_core(storage_client,bucket_name):
    """Use the proper V5.5 CFBD roll-forward, never legacy CORE OOF."""
    bucket=storage_client.bucket(bucket_name)
    pointer=bucket.blob('research/ncaaf/v55/ats_primary/latest_replay_status.json')
    if not pointer.exists():return None,{'status':'NO_V55_REPLAY_POINTER'}
    status=json.loads(pointer.download_as_text())
    prefix=str(status.get('run_prefix') or '')
    if not prefix.startswith('research/ncaaf/v55/ats_primary/replay_runs/'):
        raise ValueError('V55_CORE_REPLAY_PREFIX_NOT_TRUSTED')
    blob=bucket.blob(prefix+'/retrospective_core_ats_game_edges.csv')
    if not blob.exists():return None,{'status':'V55_ATS_GAME_EDGES_NOT_FOUND','prefix':prefix}
    frame=pd.read_csv(io.BytesIO(blob.download_as_bytes()),dtype={'game_id':str})
    return frame,{'status':'V55_REPLAY_RETROSPECTIVE_ONLY','prefix':prefix,'rows':len(frame)}


def publish(storage_client,bucket_name,outputs,*, stamp=None):
    """Immutable timestamped research results, no production/current report mutation."""
    timestamp=stamp or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    prefix=RESEARCH_ROOT+'/'+timestamp
    bucket=storage_client.bucket(bucket_name)
    for name,obj in outputs.items():
        if isinstance(obj,pd.DataFrame):
            raw=obj.to_csv(index=False).encode('utf-8');filename=name+'.csv';ctype='text/csv'
        else:
            raw=json.dumps(obj,default=str,indent=2,sort_keys=True).encode('utf-8');filename=name+'.json';ctype='application/json'
        bucket.blob(prefix+'/'+filename).upload_from_string(raw,content_type=ctype,if_generation_match=0)
    return {'status':outputs['audit_v5']['status'],'artifact_prefix':f'gs://{bucket_name}/{prefix}',
            'audit_v5':outputs['audit_v5'],'bet_authority':0,'production_mutation':False}
