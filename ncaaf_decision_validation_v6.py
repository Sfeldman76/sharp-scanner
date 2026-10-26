"""NCAAF V6 offline game-level cross-brain validation (SHADOW ONLY).

Consumes immutable V5.2 outputs; does not alter deployed services, selection,
current_report.json or production betting authority. No retrospective signal is
represented as an archived real-time pick. Unknown market/CORE quote timing is HOLD.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from collections import Counter
import numpy as np
import pandas as pd

VERSION = 'NCAAF_DECISION_VALIDATION_V6_4_20261026'
REQUIRED = ('market','test_season','source_identity','source_game_id','rule_id',
            'frozen_rule_rank','rule_decision_side','rule_sources',
            'retrospective_rule_result','core_side','core_historical_result',
            'core_edge_points','market_quote_asof_verified',
            'point_in_time_feature_certified','bet_authority')
MARKET_SIDES = {'spreads': {'HOME','AWAY'},'h2h':{'HOME','AWAY'},'totals':{'OVER','UNDER'}}
RESULTS = {'WIN':1,'LOSS':-1,'PUSH':0}
POLICIES=('CORE_ONLY','NO_OPPOSITION','TWO_CATEGORY_VETO','TWO_CATEGORY_SUPPORT_NO_OPPOSITION')
CSV_FILES=('retrospective_game_decision_ledger_v5.csv','season_forward_frozen_rules_v5.csv',
           'new_full_history_paths_v5.csv','feature_source_coverage_v5.csv')


def read_inputs(folder):
    folder=Path(folder)
    d=pd.read_csv(folder / CSV_FILES[0],low_memory=False,dtype={'source_game_id':str,'source_identity':str,'rule_id':str})
    f=pd.read_csv(folder / CSV_FILES[1],low_memory=False,dtype={'rule_id':str})
    n=pd.read_csv(folder / CSV_FILES[2],low_memory=False,dtype={'rule_id':str})
    s=pd.read_csv(folder / CSV_FILES[3],low_memory=False)
    return d,f,n,s


def _truth(v):
    # Explicit boolean parser: bool('False') is True in Python.
    return str(v).strip().lower() in ('true','1','yes','y')


def _sign(s):
    return {'WIN':1,'LOSS':-1,'PUSH':0}.get(str(s).upper(),np.nan)


def _fired_home_sign(row):
    side=str(row['rule_decision_side'])
    result=_sign(row['retrospective_rule_result'])
    if not np.isfinite(result):return np.nan
    return result if side in ('HOME','OVER') else -result


def canonical_from_research_frame(games, seasons):
    """Recompute game settlement from original research frame, separately from V5.

    Both market and history are RETROSPECTIVE, not contemporaneous certified.
    Ambiguous mirrored source rows are excluded rather than silently chosen.
    """
    if not isinstance(games,pd.DataFrame):raise TypeError('CANONICAL_SOURCE_MUST_BE_DATAFRAME')
    def first_field(names):
        for c in names:
            if c in games:return games[c]
        return pd.Series([np.nan]*len(games),index=games.index)
    sy=pd.to_numeric(pd.Series(seasons),errors='coerce').reset_index(drop=True)
    if len(sy)!=len(games):raise ValueError('CANONICAL_SOURCE_ROW_ALIGNMENT')
    ids=first_field(('Source_Game_ID','Game_ID','game_id','CFBD_Game_ID')).astype('string').fillna('').str.strip().reset_index(drop=True)
    home=pd.to_numeric(first_field(('Is_Home',)),errors='coerce').reset_index(drop=True)
    margin=pd.to_numeric(first_field(('Actual_Margin',)),errors='coerce').reset_index(drop=True)
    total=pd.to_numeric(first_field(('Actual_Total',)),errors='coerce').reset_index(drop=True)
    spread=pd.to_numeric(first_field(('Consensus_Open_Spread','Opening_Spread')),errors='coerce').reset_index(drop=True)
    total_line=pd.to_numeric(first_field(('Consensus_Open_Total','Opening_Total')),errors='coerce').reset_index(drop=True)
    valid=sy.between(2013,2025)&ids.ne('')&home.isin((0,1))
    frame=pd.DataFrame({'test_season':sy[valid].astype(int).to_numpy(),
            'source_identity':(sy[valid].astype(int).astype(str)+'|'+ids[valid]).to_numpy(),
            'actual_home_margin':np.where(home[valid].eq(1),margin[valid],-margin[valid]),
            'actual_total':total[valid].to_numpy(),
            'spread_home':np.where(home[valid].eq(1),spread[valid],-spread[valid]),
            'total_line':total_line[valid].to_numpy(),
            'canonical_source':'RESEARCH_FRAME_NOT_INDEPENDENTLY_ATTESTED',
            'quote_timestamp_verified':False,'feature_vintage_verified':False})
    rows=[]
    for key,group in frame.groupby(['test_season','source_identity'],sort=False):
        clean={}
        for field in ('actual_home_margin','actual_total','spread_home','total_line'):
            vals=group[field].dropna().astype(float).to_numpy()
            if vals.size and float(vals.max()-vals.min())>1e-6:
                raise ValueError('CANONICAL_MIRRORED_SOURCE_CONTRADICTION:'+key[1]+':'+field)
            clean[field]=float(vals[0]) if vals.size else np.nan
        rows.append({'test_season':key[0],'source_identity':key[1],**clean,
                     'canonical_source':'RESEARCH_FRAME_NOT_INDEPENDENTLY_ATTESTED',
                     'quote_timestamp_verified':False,'feature_vintage_verified':False})
    return pd.DataFrame(rows)


def _grade_canonical(row, canonical):
    """Only when canonical standardized physical-game outcomes are supplied.

    Canonical dataframe schema: test_season,source_identity,actual_home_margin,
    actual_total,spread_home,total_line,core_spread_home (optional),
    quote_timestamp_verified,feature_vintage_verified.
    The supplied source must document identity and orientation externally.
    """
    market=row['market']; home=canonical['actual_home_margin']; actual_total=canonical['actual_total']
    if market=='spreads':
        spread=canonical['spread_home']
        if pd.isna(home) or pd.isna(spread):return np.nan
        score=float(home)+float(spread)
    elif market=='totals':
        total=canonical['total_line']
        if pd.isna(actual_total) or pd.isna(total):return np.nan
        score=float(actual_total)-float(total)
    else:
        if pd.isna(home):return np.nan
        score=float(home)
    result=1 if score>1e-8 else -1 if score<-1e-8 else 0
    return result*(1 if row['rule_decision_side'] in ('HOME','OVER') else -1)


def _normalise(d,f):
    missing=set(REQUIRED)-set(d.columns)
    if missing:raise ValueError('MISSING_LEDGER_COLUMNS:'+','.join(sorted(missing)))
    if d.empty:raise ValueError('EMPTY_LEDGER')
    if bool(d['bet_authority'].fillna(0).astype(int).ne(0).any()):raise ValueError('INPUT_HAS_LIVE_AUTHORITY')
    d=d.copy()
    d['market']=d.market.astype(str).str.lower()
    if not d.market.isin(MARKET_SIDES).all():raise ValueError('UNKNOWN_MARKET')
    for market,sides in MARKET_SIDES.items():
        sel=d.market.eq(market)
        if not d.loc[sel,'rule_decision_side'].isin(sides).all():raise ValueError('INVALID_RULE_SIDE:'+market)
    if (pd.to_numeric(d.test_season,errors='coerce')>=2026).any():raise ValueError('CURRENT_OR_FUTURE_OUTCOME_INPUT')
    if not (d.source_identity.astype(str).str.split('|',regex=False).str[0]==d.test_season.astype(str)).all():
        raise ValueError('SOURCE_YEAR_GAME_IDENTITY_MISMATCH')
    if 'train_end_season' in f and (f.train_end_season>=f.test_season).any():
        raise ValueError('NON_PRIOR_YEAR_FROZEN_RECIPE')
    d['frozen_rule_rank']=pd.to_numeric(d.frozen_rule_rank,errors='raise').astype(int)
    if (d.frozen_rule_rank<=0).any():raise ValueError('INVALID_FROZEN_RANK')
    if 'rule_selected_before_test_year' in d and not d.rule_selected_before_test_year.map(_truth).all():
        raise ValueError('TEST_YEAR_SELECTION_LEAK')
    f=f.copy()
    valid=f[['market','test_season','rule_id']].drop_duplicates()
    valid['market']=valid.market.astype(str).str.lower()
    chk=d.merge(valid.assign(_rule_preselected=1),on=['market','test_season','rule_id'],how='left',validate='many_to_one')
    if chk._rule_preselected.isna().any():raise ValueError('LEDGER_RULE_NOT_IN_PRIOR_FROZEN_CATALOG')
    # Unique source+rule+year: repeated original team side firings are not independent events.
    key=['market','test_season','source_identity','rule_id']
    if chk.duplicated(key).any():
        dup=chk.loc[chk.duplicated(key,keep=False),key+['rule_decision_side','retrospective_rule_result']]
        if dup.groupby(key,dropna=False).agg({'rule_decision_side':'nunique','retrospective_rule_result':'nunique'}).gt(1).any().any():
            raise ValueError('CONFLICTING_DUPLICATE_RULE_GAME')
        chk=chk.drop_duplicates(key)
    return chk.drop(columns='_rule_preselected')


# V6.2: situational-edge discovery on the EXISTING validated V5.2 ledger.
# No new model, no majority voting, no live decisions. Historical season-forward
# grades are only retrospective until source and book-price vintage is certified.
def _wilson_lower(w, l, z=1.96):
    n = w + l
    if n <= 0:
        return np.nan
    p = w / n
    return (p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1+z*z/n)


def _edge_summary(rows, group, *, kind):
    """One result per distinct (condition, physical game), not rule firings/votes."""
    out = []
    keys = list(group) if isinstance(group, (tuple, list)) else [group]
    for key, g in rows.groupby(keys, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        ident = dict(zip(keys, key))
        w = int((g.result == 'WIN').sum())
        l = int((g.result == 'LOSS').sum())
        push = int((g.result == 'PUSH').sum())
        seasons = g.groupby('test_season').result.agg(
            n=lambda x: int(x.isin(('WIN', 'LOSS')).sum()),
            wins=lambda x: int(x.eq('WIN').sum()))
        meaningful = seasons.loc[seasons.n >= 8]
        positives = int((meaningful.wins / meaningful.n > 0.5238095238).sum())
        out.append({**ident, 'condition_type':kind,
            'graded_games': w+l, 'wins': w, 'losses': l, 'pushes':push,
            'winrate_descriptive':w/(w+l) if w+l else np.nan,
            'rate_shrunk_to_half_n40':(w+20)/(w+l+40),
            'wilson95_lower':_wilson_lower(w,l),
            'tested_seasons':int(len(seasons)),
            'seasons_with_8plus_grades':int(len(meaningful)),
            'seasons_above_110_breakeven':positives,
            'min_meaningful_season_winrate':float((meaningful.wins/meaningful.n).min()) if len(meaningful) else np.nan,
            'price_vintage_certified':False,
            'qualified_roi':False,
            'selection_status':'RETROSPECTIVE_HYPOTHESIS_NOT_CERTIFIED',
            'bet_authority':0})
    return pd.DataFrame(out)


def _prior_year_rollforward(years, groupcols, *, pair=False):
    """Prior HELD-OUT season grades only; NEVER use this year's outcome to score its signal."""
    rows = []
    for key, frame in years.groupby(groupcols, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        identity = dict(zip(groupcols,key))
        history = []
        for _, obs in frame.sort_values('test_season').iterrows():
            yw, yl = int(obs.wins), int(obs.losses)
            pw = sum(x[1] for x in history)
            pl = sum(x[2] for x in history)
            prior_seasons = sum(x[1]+x[2]>=8 for x in history)
            prev_rates = [x[1]/(x[1]+x[2]) for x in history if x[1]+x[2]>=8]
            # Pooled evidence matters more than a binary season-by-season vote.
            # A single down year reduces total results and increases uncertainty,
            # but NEVER disqualifies a condition merely because that year lost.
            p_shrunk = (pw+20)/(pw+pl+40)
            strength = p_shrunk-0.5
            meaningful = [(y,w,l) for y,w,l in history if w+l>=8]
            bad = [(y,w,l) for y,w,l in meaningful if w/(w+l)<(110/210)]
            n_good = len(meaningful)-len(bad)
            # Separate ordinary binomial season noise from observed between-year
            # variation; only EXCESS heterogeneity reduces research ranking.
            # This is a descriptive risk adjustment, NOT validated predictive EV.
            meaningful_n = sum(w+l for _,w,l in meaningful)
            if len(meaningful)>=2 and meaningful_n:
                mean_p = sum(w for _,w,_ in meaningful)/meaningful_n
                season_mse = sum((w+l)*((w/(w+l))-mean_p)**2 for _,w,l in meaningful)/meaningful_n
                sampling_mse = len(meaningful)*mean_p*(1-mean_p)/meaningful_n
                excess_season_sd = math.sqrt(max(0.,season_mse-sampling_mse))
            else:
                excess_season_sd = 0.
            variance_penalty = min(0.05, 0.5*excess_season_sd)
            # Whether a single season dominates the historical edge is visible,
            # not an implicit veto. Evaluate excluding EACH meaningful season.
            leave_one_season_out_min = min(
                ((pw-w+20)/(pw+pl-w-l+40)-0.5)
                for _,w,l in meaningful) if meaningful else np.nan
            last_bad_streak = 0
            for y,w,l in reversed(meaningful):
                if w/(w+l) >= 110/210:
                    break
                last_bad_streak+=1
            repeated_bad_penalty = min(0.03,0.01*max(0,last_bad_streak-1))
            robust_score = strength-variance_penalty-repeated_bad_penalty
            # This probabilistic proxy is deliberately independent of observed
            # TEST-year results and is NOT certified ROI or executable pricing.
            post_sd = math.sqrt(p_shrunk*(1-p_shrunk)/(pw+pl+41))
            prob_above_110 = 0.5*math.erfc(((110/210)-p_shrunk)/(math.sqrt(2)*post_sd)) if post_sd else np.nan
            # Keep the historical prior-only selection gate unchanged. No
            # per-season positive-result requirement was ever intended.
            evidence_ok = (pw+pl >= 40 and prior_seasons>=2 and
                           n_good>=1 and strength>=0.035 and identity['market'] != 'h2h')
            if not meaningful:
                regime='INSUFFICIENT_YEARLY_OBSERVATIONS'
            elif len(bad)==0:
                regime='NO_MEANINGFUL_DOWN_YEAR'
            elif last_bad_streak>=2:
                regime='REPEATED_RECENT_DOWN_YEARS_REVIEW'
            elif last_bad_streak==1:
                regime='ONE_RECENT_DOWN_YEAR_NOT_A_VETO'
            else:
                regime='RECOVERED_AFTER_DOWN_YEAR'
            rows.append({**identity,'test_season':int(obs.test_season),
                'prior_tested_seasons':len(history),
                'prior_8plus_grade_seasons':prior_seasons,
                'prior_wins':pw,'prior_losses':pl,
                'prior_shrunk_excess_over_half':strength,
                'prior_robust_support_score':robust_score,
                'prior_excess_between_season_sd':excess_season_sd,
                'prior_variance_risk_penalty':variance_penalty,
                'prior_repeated_down_year_penalty':repeated_bad_penalty,
                'prior_meaningful_down_seasons':len(bad),
                'prior_meaningful_up_seasons':n_good,
                'prior_last_down_year_streak':last_bad_streak,
                'prior_leave_one_season_out_min_excess':leave_one_season_out_min,
                'prior_probability_above_minus110_proxy':prob_above_110,
                'prior_evidence_regime':regime,
                'prior_wilson95_lower':_wilson_lower(pw,pl),
                'prior_min_meaningful_season_winrate':min(prev_rates) if prev_rates else np.nan,
                'prior_only_stability_screen':bool(evidence_ok),
                'test_wins':yw,'test_losses':yl,'test_pushes':int(obs.pushes),
                'test_nonpush_winrate':yw/(yw+yl) if yw+yl else np.nan,
                'test_outcome_used_in_prior_score':False,
                'market_price_certified':False,'bet_authority':0})
            history.append((int(obs.test_season),yw,yl))
    return pd.DataFrame(rows)


def _complete_exact_pair_research(simple, rule_years):
    """Full rule-pair intersections with game identity, not per-game rank trim.

    The rule definition and pair key are frozen and source categories sorted.
    A category is not proof of independent signal provenance. Rule pair selection
    uses only prior evaluated seasons in _prior_year_rollforward.
    """
    from itertools import combinations
    pair_keys=['market','test_season','source_identity','rule_decision_side']
    gkeys=['market','test_season','source_identity','rule_pair']
    simple=simple.copy()
    simple['category_rank']=(simple.sort_values(['frozen_rule_rank','rule_id'])
       .groupby(pair_keys+['rule_sources'],sort=False).cumcount().add(1)
       .reindex(simple.index))
    # Category-specific joins bound the peak memory. Never form all 2.3m
    # same-game permutations or drop true pair members by ranking.
    parts=[];raw_events=0;invalid_grades=0
    categories=sorted(simple.rule_sources.unique())
    for source_a,source_b in combinations(categories,2):
        a=simple.loc[simple.rule_sources.eq(source_a),
          pair_keys+['rule_id','result','category_rank']]
        b=simple.loc[simple.rule_sources.eq(source_b),
          pair_keys+['rule_id','result','category_rank']]
        if a.empty or b.empty:continue
        x=a.merge(b,on=pair_keys,how='inner',suffixes=('_a','_b'))
        if x.empty:continue
        raw_events+=len(x)
        bad=x.result_a.ne(x.result_b)
        invalid_grades+=int(bad.sum())
        x=x.loc[~bad].copy()
        if x.empty:continue
        x['rule_pair']=x.rule_id_a+'|'+x.rule_id_b
        x['rank_restricted_v62']=(x.category_rank_a.le(2)&x.category_rank_b.le(2))
        x['category_a']=source_a;x['category_b']=source_b
        x=x.rename(columns={'rule_decision_side':'side','result_a':'result'})
        parts.append(x[['market','test_season','source_identity','side','rule_pair',
            'rule_id_a','rule_id_b','category_a','category_b','result','rank_restricted_v62']])
    cols=['market','test_season','source_identity','side','rule_pair',
          'rule_id_a','rule_id_b','category_a','category_b','result','rank_restricted_v62']
    events=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=cols)
    if events.empty:
        empty_scores=pd.DataFrame(columns=['market','rule_pair','graded_games','wins','losses','pushes','bet_authority'])
        empty_fold=pd.DataFrame(columns=['market','rule_pair','test_season','prior_only_stability_screen','bet_authority'])
        empty_components=pd.DataFrame(columns=['market','rule_pair','test_season','wins','losses','pushes','bet_authority'])
        return (events,empty_fold,empty_scores,empty_scores.copy(),empty_components,
                {'raw_join_rows':raw_events,'invalid_pair_grade_rows':invalid_grades,
                 'pair_game_conflict_rows_removed':0,'full_pair_events':0,
                 'ranked_subset_events':0,'all_pair_definitions':0,'retained_pairs':0})
    # One pair/physical game; exclude a pair when it simultaneously signals
    # both sides, rather than cherry-pick the more successful side.
    conflicts=events.groupby(gkeys,sort=False).side.transform('nunique').gt(1)
    removed=int(conflicts.sum())
    events=events.loc[~conflicts].drop_duplicates(gkeys).copy()
    all_event_count=len(events)
    w=events.result.eq('WIN').astype('int8');l=events.result.eq('LOSS').astype('int8');p=events.result.eq('PUSH').astype('int8')
    events['n_w']=w;events['n_l']=l;events['n_p']=p
    group=['market','rule_pair'];group_year=group+['test_season']
    years=events.groupby(group_year,as_index=False,sort=False).agg(
        wins=('n_w','sum'),losses=('n_l','sum'),pushes=('n_p','sum'),
        rank_subset_firings=('rank_restricted_v62','sum'))
    total=years.assign(nonpush=years.wins+years.losses).groupby(group,as_index=False).agg(
        total_graded=('nonpush','sum'),eval_seasons=('test_season','nunique'))
    # A 40-grade/2-season prior screen implies at least 25 grades/2 years
    # across the observed history; prune weak definitions before expensive
    # scorecards without hiding possible prior-screen selections.
    keep=total.loc[total.total_graded.ge(25)&total.eval_seasons.ge(2),group]
    if keep.empty:
        empty_scores=pd.DataFrame(columns=group+['graded_games','wins','losses','pushes','bet_authority'])
        empty_folds=pd.DataFrame(columns=group+['test_season','prior_only_stability_screen','bet_authority'])
        audit={'raw_join_rows':raw_events,'invalid_pair_grade_rows':invalid_grades,
               'pair_game_conflict_rows_removed':removed,'full_pair_events':all_event_count,
               'ranked_subset_events':int(events.rank_restricted_v62.sum()),
               'all_pair_definitions':int(len(total)),'retained_pairs':0,
               'retained_event_rows':0,'rank_restricted_population_changed_pairs':0}
        empty_components=pd.DataFrame(columns=['market','rule_pair','test_season','wins','losses','pushes','bet_authority'])
        return (events.iloc[0:0].copy(),empty_folds,empty_scores,
                empty_scores.copy(),empty_components,audit)
    yrs=years.merge(keep,on=group,how='inner',validate='many_to_one')
    selected_events=events.merge(keep,on=group,how='inner',validate='many_to_one')
    folds=_prior_year_rollforward(yrs[['market','rule_pair','test_season','wins','losses','pushes']],group)
    scores=_edge_summary(selected_events,group,kind='EXACT_FULL_INTERSECTION_DISJOINT_CATEGORY_PAIR')
    info=(selected_events.groupby(group,as_index=False,sort=False)
          .agg(rule_id_a=('rule_id_a','first'),rule_id_b=('rule_id_b','first'),
               category_a=('category_a','first'),category_b=('category_b','first')))
    scores=scores.merge(info,on=group,validate='one_to_one')
    scores['intersection_policy']='ALL_SAME_SIDE_FIRINGS__NO_PRE_RANK_CUTOFF'
    scores['source_independence_certified']=False
    # Sensitivity to *previous* algorithm's top-two restriction on identical
    # fixed rule definitions. This never determines rule eligibility.
    restricted=selected_events.loc[selected_events.rank_restricted_v62]
    rank_counts=(restricted.groupby(group,as_index=False).agg(
        rank2_wins=('n_w','sum'),rank2_losses=('n_l','sum'),rank2_pushes=('n_p','sum')))
    sens=scores[group+['graded_games','wins','losses','pushes']].merge(rank_counts,on=group,how='left')
    for col in ('rank2_wins','rank2_losses','rank2_pushes'):
        sens[col]=sens[col].fillna(0).astype(int)
    sens['full_rate']=sens.wins.div(sens.wins+sens.losses).where(sens.wins+sens.losses>0)
    sens['rank2_rate']=sens.rank2_wins.div(sens.rank2_wins+sens.rank2_losses).where(
        sens.rank2_wins+sens.rank2_losses>0)
    sens['rank_cutoff_changes_population']=(sens.graded_games.ne(sens.rank2_wins+sens.rank2_losses))
    sens['rank2_is_diagnostic_only']=True;sens['bet_authority']=0
    # Compare pair performance to both constituent rules across exactly the
    # same evaluation YEARS. On actual pair games, all three sides and grades
    # necessarily coincide. Incremental *selection* effect compares the pair
    # intersection against each rule's non-intersection games.
    component_years=rule_years[['market','test_season','rule_id','wins','losses','pushes']]
    components=yrs.merge(info,on=group,validate='many_to_one')
    for side in ('a','b'):
        src=component_years.rename(columns={
           'rule_id':'rule_id_'+side,'wins':'component_'+side+'_all_wins',
           'losses':'component_'+side+'_all_losses','pushes':'component_'+side+'_all_pushes'})
        components=components.merge(src,on=['market','test_season','rule_id_'+side],how='left',validate='many_to_one')
        for kind in ('wins','losses','pushes'):
            orig='component_'+side+'_all_'+kind
            if components[orig].isna().any():raise ValueError('EXACT_PAIR_COMPONENT_MISSING')
            components['component_'+side+'_only_'+kind]=(components[orig]-components[kind]).astype(int)
            if (components['component_'+side+'_only_'+kind]<0).any():
                raise ValueError('EXACT_PAIR_COMPONENT_SUBTRACTION_NEGATIVE')
    components['pair_nonpush']=components.wins+components.losses
    components['pair_rate']=components.wins.div(components.pair_nonpush).where(components.pair_nonpush>0)
    for side in ('a','b'):
        denom=components['component_'+side+'_only_wins']+components['component_'+side+'_only_losses']
        components['component_'+side+'_only_rate']=components['component_'+side+'_only_wins'].div(denom).where(denom>0)
        components['pair_minus_'+side+'_only_rate_pp']=100*(components.pair_rate-components['component_'+side+'_only_rate'])
    components['same_season_not_matched_control']=True
    components['source_vintage_certified']=False;components['bet_authority']=0
    audit={'raw_join_rows':raw_events,'invalid_pair_grade_rows':invalid_grades,
           'pair_game_conflict_rows_removed':removed,'full_pair_events':all_event_count,
           'ranked_subset_events':int(events.rank_restricted_v62.sum()),
           'all_pair_definitions':int(len(total)),'retained_pairs':int(len(scores)),
           'retained_event_rows':int(len(selected_events)),
           'rank_restricted_population_changed_pairs':int(sens.rank_cutoff_changes_population.sum())}
    return selected_events,folds,scores,sens,components,audit


def _exact_pair_core_milestone(specific, folds, core_pool):
    """Paired conditional evidence at original CORE opportunities, with hold gates.

    A same-side confirmation returns the SAME win/loss as CORE on that game;
    sample enrichment is a *selection* claim, not same-game prediction lift.
    Opposition is a potential fade, not a signed profitable bet without prices.
    """
    cols=['market','test_season','source_identity','rule_pair','side','result']
    empty_ev=pd.DataFrame(columns=cols+['core_side','role','prior_only_stability_screen','bet_authority'])
    empty_cmp=pd.DataFrame(columns=['market','rule_pair','test_season','role','core_bucket','core_wins','core_losses','pair_wins','pair_losses','matched_games','baseline_winrate','selection_lift_vs_bucket_pp','bet_authority'])
    empty_pol=pd.DataFrame(columns=['test_season','policy','selected_games','core_wins','core_losses','pair_wins','pair_losses','status','bet_authority'])
    audit={'prior_screened_pair_game_events':0,'pair_matched_core_events':0,'distinct_core_games':0,
           'core_pairs_grade_mismatch_excluded':0,'unique_game_pair_policy_rows':0,
           'status':'NO_PRESEASON_SCREENED_MATCHED_CORE_PAIRS',
           'certified_incremental_roi':False,'bet_authority':0}
    if specific.empty or folds.empty or core_pool.empty:
        return empty_ev,empty_cmp,empty_pol,audit
    eligible=folds.loc[folds.prior_only_stability_screen.fillna(False),
        ['market','rule_pair','test_season','prior_wins','prior_losses','prior_shrunk_excess_over_half','prior_robust_support_score','prior_evidence_regime','prior_8plus_grade_seasons']]
    if eligible.empty:return empty_ev,empty_cmp,empty_pol,audit
    x=(specific.loc[specific.market.eq('spreads')].merge(
        eligible,on=['market','rule_pair','test_season'],how='inner',validate='many_to_one'))
    audit['prior_screened_pair_game_events']=int(len(x))
    if x.empty:return empty_ev,empty_cmp,empty_pol,audit
    core=core_pool.loc[core_pool.market.eq('spreads')].copy()
    core['core_edge_points_unverified']=pd.to_numeric(core.core_edge_points_unverified,errors='coerce')
    x=x.merge(core[['market','test_season','source_identity','core_side','core_result_unverified','core_bucket',
           'core_edge_points_unverified']],on=['market','test_season','source_identity'],how='inner',validate='many_to_one')
    audit['pair_matched_core_events']=int(len(x))
    if x.empty:return empty_ev,empty_cmp,empty_pol,audit
    x['role']=np.where(x.side.eq(x.core_side),'CONFIRM_CORE','OPPOSE_CORE')
    x['core_grade_sign']=x.core_result_unverified.map(RESULTS)
    x['pair_grade_sign']=x.result.map(RESULTS)
    consistent=x.pair_grade_sign.eq(np.where(x.role.eq('CONFIRM_CORE'),x.core_grade_sign,-x.core_grade_sign))
    audit['core_pairs_grade_mismatch_excluded']=int((~consistent).sum())
    x=x.loc[consistent].copy()
    if x.empty:return empty_ev,empty_cmp,empty_pol,audit
    x['core_win']=x.core_result_unverified.eq('WIN').astype('int8')
    x['core_loss']=x.core_result_unverified.eq('LOSS').astype('int8')
    x['pair_win']=x.result.eq('WIN').astype('int8')
    x['pair_loss']=x.result.eq('LOSS').astype('int8')
    x['historical_bookmaker_quote_time_certified']=False
    x['source_signal_vintage_certified']=False;x['bet_authority']=0
    # Baseline exactly matched on market/season/CORE edge bucket AND CORE side.
    baseline=core.groupby(['market','test_season','core_bucket','core_side'],as_index=False).agg(
        baseline_wins=('core_result_unverified',lambda z:int(z.eq('WIN').sum())),
        baseline_losses=('core_result_unverified',lambda z:int(z.eq('LOSS').sum())))
    baseline['baseline_winrate']=(baseline.baseline_wins/
        (baseline.baseline_wins+baseline.baseline_losses))
    comparison=(x.groupby(['market','rule_pair','test_season','role','core_bucket','core_side'],as_index=False)
        .agg(core_wins=('core_win','sum'),core_losses=('core_loss','sum'),
             pair_wins=('pair_win','sum'),pair_losses=('pair_loss','sum'),
             matched_games=('source_identity','nunique'),
             prior_wins=('prior_wins','first'),prior_losses=('prior_losses','first'),
             prior_excess=('prior_shrunk_excess_over_half','first')))
    comparison=comparison.merge(baseline,on=['market','test_season','core_bucket','core_side'],
        how='left',validate='many_to_one')
    comparison['core_matched_winrate']=comparison.core_wins/(comparison.core_wins+comparison.core_losses)
    comparison['pair_matched_winrate']=comparison.pair_wins/(comparison.pair_wins+comparison.pair_losses)
    comparison['selection_lift_vs_bucket_pp']=100*(comparison.core_matched_winrate-comparison.baseline_winrate)
    comparison['pair_minus_core_same_game_pp']=100*(comparison.pair_matched_winrate-comparison.core_matched_winrate)
    comparison['same_game_profit_comparison_certified']=False
    comparison['independent_holdout_certified']=False
    comparison['support_is_selection_enrichment_not_new_prediction']=True
    comparison['bet_authority']=0
    # A bounded one-CORE-opportunity-per-game shadow policy. Competing exact
    # pairs are ranked by PRE-TEST-YEAR evidence alone. If strongest eligible
    # supports CORE we flag confirmation; if it opposes CORE flag research fade;
    # neither is authorized at historical unverified prices.
    x=x.sort_values(['market','test_season','source_identity','prior_robust_support_score',
                      'prior_shrunk_excess_over_half','prior_wins','rule_pair'],ascending=[True,True,True,False,False,False,True])
    distinct=x.drop_duplicates(['market','test_season','source_identity']).copy()
    distinct['policy']=np.where(distinct.role.eq('CONFIRM_CORE'),'PRIOR_QUALIFIED_CONFIRM_CORE',
                               'PRIOR_QUALIFIED_POTENTIAL_FADE')
    policy=distinct.groupby(['test_season','policy'],as_index=False).agg(
        selected_games=('source_identity','nunique'),core_wins=('core_win','sum'),
        core_losses=('core_loss','sum'),pair_wins=('pair_win','sum'),pair_losses=('pair_loss','sum'))
    policy['status']='RESEARCH_ONLY_EXACT_PAIR_PRIOR_SCREEN__PRICE_VINTAGE_HOLD'
    policy['bet_authority']=0
    audit['distinct_core_games']=int(len(distinct))
    audit['unique_game_pair_policy_rows']=int(len(policy))
    audit['status']='RESEARCH_ONLY_PRIOR_SCREENED_PAIRS_WITH_CORE__PRICE_VINTAGE_HOLD'
    return x,comparison,policy,audit


def _pair_forward_one_opportunity(specific, folds):
    """Retrospective season-forward screening, ONE opportunity per physical game.

    Do not sum overlapping frozen-rule portfolios, and do not resolve opposing
    directions by observed results. H2H returns remain unqualified without odds.
    """
    empty=pd.DataFrame(columns=['market','test_season','source_identity','side','result',
         'rule_pair','prior_wins','prior_losses','decision_state','bet_authority'])
    empty_summary=pd.DataFrame(columns=['market','test_season','graded_games','wins','losses','pushes',
        'conflict_games_passed','winrate_descriptive','status','bet_authority'])
    if specific.empty or folds.empty:return empty,empty_summary,{'qualified_pair_years':0,'eligible_firings':0,'conflict_games':0,'decisions':0}
    ok=folds.loc[folds.prior_only_stability_screen.fillna(False),
        ['market','rule_pair','test_season','prior_wins','prior_losses','prior_shrunk_excess_over_half','prior_robust_support_score','prior_evidence_regime']]
    ok=ok.loc[ok.market.isin(('spreads','totals'))]
    if ok.empty:return empty,empty_summary,{'qualified_pair_years':0,'eligible_firings':0,'conflict_games':0,'decisions':0}
    events=specific.merge(ok,on=['market','rule_pair','test_season'],how='inner',validate='many_to_one')
    if events.empty:return empty,empty_summary,{'qualified_pair_years':int(len(ok)),'eligible_firings':0,'conflict_games':0,'decisions':0}
    groups=['market','test_season','source_identity']
    conflicts=events.groupby(groups).side.nunique().gt(1)
    conflicts=conflicts.rename('opposing_prequalified_pairs').reset_index()
    events=events.merge(conflicts,on=groups,how='left',validate='many_to_one')
    events=events.sort_values(groups+['prior_robust_support_score','prior_shrunk_excess_over_half','prior_wins','rule_pair'],
                ascending=[True,True,True,False,False,False,True],kind='mergesort')
    representative=events.drop_duplicates(groups).copy()
    representative['qualified_pairs_on_game']=events.groupby(groups).rule_pair.transform('size').loc[representative.index].astype(int)
    representative['decision_state']=np.where(representative.opposing_prequalified_pairs,
                  'PASS_OPPOSING_PREQUALIFIED_PAIRS','EXPLORATORY_PRIOR_SCREEN_PAIR')
    representative['eligible_result']=representative.result.where(~representative.opposing_prequalified_pairs)
    representative['historical_price_certified']=False
    representative['independent_source_and_holdout_certified']=False
    representative['bet_authority']=0
    agg=[]
    for (market,year),g in representative.groupby(['market','test_season'],sort=True):
        take=g.loc[g.decision_state.eq('EXPLORATORY_PRIOR_SCREEN_PAIR')]
        w=int(take.eligible_result.eq('WIN').sum());l=int(take.eligible_result.eq('LOSS').sum())
        agg.append({'market':market,'test_season':int(year),'unique_opportunities':int(len(g)),
                    'graded_games':w+l,'wins':w,'losses':l,'pushes':int(take.eligible_result.eq('PUSH').sum()),
                    'conflict_games_passed':int(g.opposing_prequalified_pairs.sum()),
                    'winrate_descriptive':w/(w+l) if w+l else float('nan'),
                    'status':'EXPLORATORY_UNVERIFIED_HISTORICAL_PRICE_AND_MULTIPLE_TESTING',
                    'bet_authority':0})
    return representative,pd.DataFrame(agg),{'qualified_pair_years':int(len(ok)),
        'eligible_firings':int(len(events)),'conflict_games':int(representative.opposing_prequalified_pairs.sum()),
        'decisions':int(representative.decision_state.eq('EXPLORATORY_PRIOR_SCREEN_PAIR').sum())}


def build_situational_edge_research(d, games):
    """Identify stable conditions and combinations, with CORE only as base.

    This is a diagnostic *selection* engine; no probability adjustments or
    bet authorities are invented. Condition strength is estimated only from
    earlier season-forward grades. Cross-family independence remains unverified.
    """
    slim = d[['market','test_season','source_identity','rule_id','rule_decision_side',
              'rule_sources','retrospective_rule_result','frozen_rule_rank',
              'core_side','core_historical_result','core_edge_points']].copy()
    slim.rename(columns={'retrospective_rule_result':'result'},inplace=True)
    slim = slim.loc[slim.result.isin(('WIN','LOSS','PUSH'))].copy()
    slim['rule_sources'] = slim.rule_sources.fillna('')
    slim['n_w'] = slim.result.eq('WIN').astype(int)
    slim['n_l'] = slim.result.eq('LOSS').astype(int)
    slim['n_p'] = slim.result.eq('PUSH').astype(int)
    # V5.2 rule_id identifies a frozen definition; deduplicate any accidental
    # repeat of the same condition in a single physical game.
    slim = slim.sort_values(['market','test_season','source_identity','rule_id','frozen_rule_rank']).drop_duplicates(
        ['market','test_season','source_identity','rule_id'])
    rule_groups = ['market','rule_id']
    rule_years = slim.groupby(rule_groups+['test_season'],as_index=False).agg(
        wins=('n_w','sum'),losses=('n_l','sum'),pushes=('n_p','sum'),
        rule_sources=('rule_sources','first'), rule_firings=('result','size'))
    rule_summary=_edge_summary(slim,rule_groups,kind='FROZEN_SINGLE_OR_EXISTING_CROSS_SOURCE_RULE')
    desc = slim.groupby(rule_groups,as_index=False).agg(
        rule_sources=('rule_sources','first'),
        example_side=('rule_decision_side','first'))
    rule_summary=rule_summary.merge(desc,on=rule_groups,how='left',validate='one_to_one')
    rule_summary['cross_source_rule']=rule_summary.rule_sources.map(lambda x:len([c for c in str(x).split('|') if c])>1)
    rule_summary['h2h_price_required']=rule_summary.market.eq('h2h')
    rule_folds=_prior_year_rollforward(rule_years,rule_groups)
    rule_folds=rule_folds.merge(rule_years[rule_groups+['test_season','rule_sources']],
                               on=rule_groups+['test_season'],how='left',validate='one_to_one')

    # Pair two SINGLE-SOURCE categories firing ON THE SAME SIDE. The pair is
    # an exploratory new condition; never count several rules in one category as
    # several independent votes. Do not combine rules that internally already
    # depend on multiple source categories.
    simple=slim.loc[slim.rule_sources.str.count('\\|').eq(0)&slim.rule_sources.ne('')]
    per_category=(simple.sort_values(['frozen_rule_rank','rule_id'])
                  .drop_duplicates(['market','test_season','source_identity',
                                    'rule_decision_side','rule_sources']))
    # Vectorized category pairing. All 235k rule firings must not be expanded
    # into nested Python rule-pair loops; only six distinct source categories
    # can participate per physical game/side.
    pair_keys=['market','test_season','source_identity','rule_decision_side']
    left=per_category[pair_keys+['rule_sources','result']].copy()
    merged=left.merge(left,on=pair_keys,how='inner',suffixes=('_a','_b'))
    merged=merged.loc[(merged.rule_sources_a < merged.rule_sources_b) &
                      (merged.result_a == merged.result_b)].copy()
    combo_data=merged[pair_keys+['rule_sources_a','rule_sources_b','result_a']].copy()
    combo_data['source_pair']=combo_data.rule_sources_a+'|'+combo_data.rule_sources_b
    combo_data.rename(columns={'rule_decision_side':'side','result_a':'result'},inplace=True)
    combo_data=combo_data[['market','test_season','source_identity','side','source_pair','result']]
    if not combo_data.empty:
        # Do not quietly count BOTH opposite positions as two profitable bets.
        contradiction=(combo_data.groupby(['market','test_season','source_identity','source_pair'])
                       .side.transform('nunique').gt(1))
        combo_conflicts=int(contradiction.sum())
        combo_data=combo_data.loc[~contradiction].drop_duplicates(
            ['market','test_season','source_identity','source_pair'])
        combo_data['n_w']=combo_data.result.eq('WIN').astype(int)
        combo_data['n_l']=combo_data.result.eq('LOSS').astype(int)
        combo_data['n_p']=combo_data.result.eq('PUSH').astype(int)
        combo_groups=['market','source_pair']
        combo_years=combo_data.groupby(combo_groups+['test_season'],as_index=False).agg(
            wins=('n_w','sum'),losses=('n_l','sum'),pushes=('n_p','sum'))
        combo_score=_edge_summary(combo_data,combo_groups,kind='CROSS_SOURCE_PAIR_SAME_SIDE')
        combo_folds=_prior_year_rollforward(combo_years,combo_groups,pair=True)
        combo_score['independent_source_provenance_certified']=False
    else:
        combo_conflicts=0
        combo_score=pd.DataFrame(columns=['market','source_pair','wins','losses','graded_games','bet_authority'])
        combo_folds=pd.DataFrame(columns=['market','source_pair','test_season','prior_only_stability_screen','bet_authority'])

    # V6.3: use the COMPLETE intersection of each fixed pair of single-source
    # definitions. V6.2 mistakenly restricted the population to the two
    # highest-ranked rules per category/game BEFORE grading. Keep that subset
    # only as an explicit sensitivity analysis, never as the full backtest.
    specific, specific_folds, specific_scores, specific_rank_sensitivity, specific_components, specific_audit = (
        _complete_exact_pair_research(simple, rule_years))

    pair_forward_games, pair_forward_summary, pair_forward_audit = (
        _pair_forward_one_opportunity(specific, specific_folds))

    # CORE is the baseline. Attach ONLY before-test-year scores to each fired
    # condition. Never treat every explorer as a veto; evaluate strong prior
    # support or opposition separately and disclose the comparator cohort.
    relevant=games.loc[games.core_candidate_in_ledger,
        ['market','test_season','source_identity','core_side','core_result_unverified',
         'core_edge_points_unverified','core_rule_grading_disagreement',
         'rule_internal_grade_consistent']].copy()
    relevant=relevant.loc[(~relevant.core_rule_grading_disagreement)&
                          relevant.rule_internal_grade_consistent].copy()
    relevant['core_bucket']=pd.cut(pd.to_numeric(relevant.core_edge_points_unverified,errors='coerce'),
        bins=[1.999,4,7,np.inf],labels=['2_TO_4','4_TO_7','7_PLUS'],include_lowest=True)
    relevant['core_bucket']=relevant['core_bucket'].astype(str)
    core_pool=relevant.loc[relevant.core_result_unverified.isin(('WIN','LOSS'))].copy()
    baseline=(core_pool.groupby(['market','test_season','core_bucket'],as_index=False)
              .core_result_unverified.agg(
                   baseline_games='size',baseline_wins=lambda x:int(x.eq('WIN').sum())))
    baseline['baseline_winrate']=baseline.baseline_wins/baseline.baseline_games
    # One physical CORE bet is an observation for each *distinct tested rule*,
    # but counts MUST NOT be summed across conditions.
    with_core=slim.merge(core_pool[['market','test_season','source_identity','core_bucket',
                       'core_side','core_result_unverified']].rename(
                            columns={'core_side':'verified_core_side',
                                     'core_result_unverified':'verified_core_grade'}),
                       on=['market','test_season','source_identity'],how='inner',validate='many_to_one')
    with_core=with_core.loc[with_core.market.eq('spreads')]
    with_core['role']=np.where(with_core.rule_decision_side.eq(with_core.verified_core_side),
                               'CONFIRM_CORE','OPPOSE_CORE')
    with_core['core_win']=with_core.verified_core_grade.eq('WIN').astype(int)
    conditional=(with_core.groupby(['market','rule_id','test_season','core_bucket','role'],as_index=False)
                 .agg(core_wins=('core_win','sum'),core_games=('core_win','size')))
    conditional=conditional.merge(baseline,on=['market','test_season','core_bucket'],how='left',validate='many_to_one')
    conditional['core_winrate']=conditional.core_wins/conditional.core_games
    conditional['core_vs_bucket_baseline_pp']=100*(conditional.core_winrate-conditional.baseline_winrate)
    conditional['descriptive_not_causal_or_certified']=True
    conditional['bet_authority']=0

    # One CORE candidate per game. A rule only earns consideration once it has
    # previous season-forward observations. Its score is an evidence descriptor,
    # NOT a calibrated probability and NOT an additive model adjustment.
    triggered=with_core.merge(rule_folds[['market','rule_id','test_season',
                'prior_only_stability_screen','prior_shrunk_excess_over_half',
                'prior_robust_support_score','prior_evidence_regime',
                'prior_wins','prior_losses','prior_8plus_grade_seasons']],
                on=['market','rule_id','test_season'],how='left',validate='many_to_one')
    triggered=triggered.loc[triggered.prior_only_stability_screen.fillna(False)].copy()
    # Strongest previous-season evidence for each condition direction; sources
    # exposed so the same underlying family isn't mistaken for many voters.
    candidate_rows=[]
    idx=['market','test_season','source_identity']
    for key,g in triggered.groupby(idx,sort=True):
        sides={}
        for role,z in g.groupby('role'):
            top=z.sort_values(['prior_robust_support_score','prior_shrunk_excess_over_half','prior_wins','rule_id'],
                              ascending=[False,False,False,True]).iloc[0]
            sides[role]=(str(top.rule_id),str(top.rule_sources),
                         float(top.prior_shrunk_excess_over_half),int(top.prior_wins+top.prior_losses))
        same=sides.get('CONFIRM_CORE');opp=sides.get('OPPOSE_CORE')
        candidate_rows.append({'market':key[0],'test_season':key[1],'source_identity':key[2],
          'support_rule_id':same[0] if same else '',
          'support_source_categories':same[1] if same else '',
          'support_prior_shrunk_excess':same[2] if same else np.nan,
          'support_prior_graded':same[3] if same else 0,
          'opposition_rule_id':opp[0] if opp else '',
          'opposition_source_categories':opp[1] if opp else '',
          'opposition_prior_shrunk_excess':opp[2] if opp else np.nan,
          'opposition_prior_graded':opp[3] if opp else 0,
          'evidence_state':('BOTH_STABLE_CONDITIONS_CONFLICT' if same and opp else
                            'PRIOR_STABLE_CONFIRMATION' if same else 'PRIOR_STABLE_OPPOSITION'),
          'selection_is_preseason_evidence_only':True,'price_asof_certified':False,
          'bet_authority':0})
    candidate=pd.DataFrame(candidate_rows)
    if candidate.empty:
        candidate=pd.DataFrame(columns=idx+['evidence_state','bet_authority'])
    candidate=relevant.merge(candidate,on=idx,how='left',validate='one_to_one')
    candidate['evidence_state']=candidate.evidence_state.fillna('NO_STABLE_PRIOR_CONDITION')
    candidate['bet_authority']=0
    # Prior-season-selected *exact* pair events evaluated against precisely the
    # same CORE physical games. Pair profitability remains uncertified until
    # the original executable price, grade and signal vintages are validated.
    pair_core_events, pair_core_comparison, pair_core_policy, pair_core_audit = (
        _exact_pair_core_milestone(specific, specific_folds, core_pool))
    audit={'version':'NCAAF_EXISTING_V6_SITUATIONAL_EDGE_V6_4_20261026',
           'one_bad_season_is_automatic_disqualification':False,
           'prior_only_robust_season_variance_ranking':True,
           'specific_pair_preseason_qualified_one_recent_down_year':int((specific_folds.prior_only_stability_screen.fillna(False) & specific_folds.prior_evidence_regime.eq('ONE_RECENT_DOWN_YEAR_NOT_A_VETO')).sum()) if not specific_folds.empty else 0,
           'status':'SEASON_FORWARD_DESCRIPTIVE_ONLY_NOT_CERTIFIED',
           'new_python_modules':0,'rule_scorecards':int(len(rule_summary)),
           'rule_prior_only_folds':int(len(rule_folds)),
           'category_pair_scorecards':int(len(combo_score)),
           'specific_cross_source_rule_pairs_retained':int(len(specific_scores)),
           'specific_pair_prior_only_folds_retained':int(len(specific_folds)),
           'specific_pair_opposite_side_conflicts_excluded':specific_audit['pair_game_conflict_rows_removed'],
           'full_pair_events':specific_audit['full_pair_events'],
           'ranked_subset_events':specific_audit['ranked_subset_events'],
           'exact_pair_core_milestone':pair_core_audit,
           'exact_pair_one_opportunity_forward':pair_forward_audit,
           'category_pair_prior_only_folds':int(len(combo_folds)),
           'pair_opposite_side_conflicts_excluded':combo_conflicts,
           'core_conditional_comparisons':int(len(conditional)),
           'matched_core_games':int(len(candidate)),
           'core_games_with_prior_stable_evidence':int(candidate.evidence_state.ne('NO_STABLE_PRIOR_CONDITION').sum()),
           'calibrated_betting_probabilities':False,
           'certified_market_roi':False,
           'independent_vintage_or_holdout_certified':False,
           'historical_core_quote_grade_discrepancies_excluded':True,
           'production_mutation':False,'bet_authority':0}
    return {'situational_edge_rules_v6':rule_summary,
            'situational_edge_rollforward_v6':rule_folds,
            'situational_combo_rules_v6':combo_score,
            'situational_combo_rollforward_v6':combo_folds,
            'situational_specific_combos_v6':specific_scores,
            'situational_specific_combo_rollforward_v6':specific_folds,
            'situational_specific_combo_rank_sensitivity_v6':specific_rank_sensitivity,
            'situational_specific_combo_components_v6':specific_components,
            'situational_specific_combo_forward_games_v6':pair_forward_games,
            'situational_specific_combo_forward_summary_v6':pair_forward_summary,
            'situational_specific_combo_core_events_v6':pair_core_events,
            'situational_specific_combo_core_comparison_v6':pair_core_comparison,
            'situational_specific_combo_core_policy_v6':pair_core_policy,
            'situational_core_conditionals_v6':conditional,
            'situational_core_candidate_ledger_v6':candidate,
            'situational_edge_audit_v6':audit}


def _policy_decision(policy,core,positive,negative):
    if core not in ('HOME','AWAY'):return 'NO_CORE'
    p=len(positive);n=len(negative)
    if policy=='CORE_ONLY':return 'KEEP_CORE'
    if policy=='NO_OPPOSITION':return 'KEEP_CORE' if n==0 else 'ABSTAIN_OPPOSITION'
    if policy=='TWO_CATEGORY_VETO':return 'ABSTAIN_VETO' if n>=2 else 'KEEP_CORE'
    if policy=='TWO_CATEGORY_SUPPORT_NO_OPPOSITION':
        return 'KEEP_CORE' if p>=2 and n==0 else 'ABSTAIN_NOT_CONFIRMED'
    raise ValueError('UNKNOWN_POLICY')


def build(d,f,n,s,*,canonical=None):
    d=_normalise(d,f)
    n_current_recipes=len(n)
    groupcols=['market','test_season','source_identity']
    fields=['market','test_season','source_identity','source_game_id','core_side','core_historical_result',
            'core_edge_points','core_relationship','rule_id','rule_sources','rule_decision_side',
            'retrospective_rule_result','frozen_rule_rank','archived_market_open_price_unverified',
            'kickoff_date_unverified','market_quote_asof_verified','point_in_time_feature_certified']
    d=d[[col for col in fields if col in d.columns]].copy()
    d['rule_signed_home_grade']=np.where(
        d.retrospective_rule_result.isin(RESULTS),
        d.retrospective_rule_result.map(RESULTS).astype(float)*np.where(d.rule_decision_side.isin(('HOME','OVER')),1,-1),np.nan)
    d['category_tokens']=d.rule_sources.fillna('').map(lambda x:tuple(sorted(set(z for z in str(x).split('|') if z))))
    d['core_edge_points']=pd.to_numeric(d.core_edge_points,errors='coerce')
    d['historical_price']=pd.to_numeric(d.get('archived_market_open_price_unverified',np.nan),errors='coerce')
    d['price_recorded']=d.historical_price.notna()
    # Sort once globally instead of sorting inside each of ~14,448 game groups.
    d=d.sort_values(['market','test_season','source_identity',
                      'frozen_rule_rank','rule_id'],kind='mergesort')
    # Canonical source is optional. Without scores and both price definitions,
    # NEVER quietly mark grade reconciliation as passed.
    lookup={}
    if canonical is not None:
        cc=canonical.copy()
        required={'test_season','source_identity','actual_home_margin','actual_total','spread_home','total_line'}
        if missing:=required-set(cc.columns):raise ValueError('CANONICAL_MISSING:'+','.join(sorted(missing)))
        if cc.duplicated(['test_season','source_identity']).any():raise ValueError('CANONICAL_DUPLICATE_GAME')
        lookup={(int(r.test_season),str(r.source_identity)):r._asdict() for r in cc.itertuples(index=False)}
    game_rows=[]
    rec_rows=[]
    for (market,year,identity),g in d.groupby(groupcols,sort=True,observed=True):
        def unique_nonblank(field):
            vals=set(str(x) for x in g[field].dropna() if str(x) and str(x).lower() not in ('nan','none'))
            return vals
        core_sides=unique_nonblank('core_side')
        core_results=unique_nonblank('core_historical_result')
        core_edges=g.core_edge_points.dropna().unique()
        core_consistent=(len(core_sides)<=1 and len(core_results)<=1 and len(core_edges)<=1)
        core_side=next(iter(core_sides)) if len(core_sides)==1 else ''
        core_result=next(iter(core_results)) if len(core_results)==1 else ''
        core_edge=float(core_edges[0]) if len(core_edges)==1 else np.nan
        core_candidate=(market=='spreads' and core_consistent and core_side in ('HOME','AWAY')
                        and np.isfinite(core_edge) and core_edge>=2 and core_result in RESULTS)
        sides={a:int((g.rule_decision_side==a).sum()) for a in sorted(MARKET_SIDES[market])}
        source_cat={side:set(z for tokens in g.loc[g.rule_decision_side==side,'category_tokens'] for z in tokens)
                    for side in MARKET_SIDES[market]}
        supporting=source_cat.get(core_side,set()) if core_candidate else set()
        opposing=source_cat.get('AWAY' if core_side=='HOME' else 'HOME',set()) if core_candidate else set()
        graded_home=set(int(x) for x in g.rule_signed_home_grade.dropna().unique())
        rule_internal_grade_consistent=len(graded_home)<=1
        core_sign=_sign(core_result)
        theoretical_core_home=(core_sign*(1 if core_side=='HOME' else -1)
                                if core_candidate else np.nan)
        rule_home=next(iter(graded_home)) if len(graded_home)==1 else np.nan
        core_rule_grade_disagreement=(bool(core_candidate and np.isfinite(rule_home)
                                               and theoretical_core_home!=rule_home))
        unverified_quote=not g.market_quote_asof_verified.map(_truth).all()
        unverified_features=not g.point_in_time_feature_certified.map(_truth).all()
        prices=g.historical_price.dropna().unique()
        price_consistent=len(prices)<=1
        price=float(prices[0]) if len(prices)==1 else np.nan
        if market=='spreads' and not np.isfinite(price):
            market_quote_reason='MISSING_SPREAD'
        elif market=='totals' and not np.isfinite(price):
            market_quote_reason='MISSING_TOTAL'
        else:market_quote_reason='HISTORICAL_QUOTE_TIMESTAMP_UNVERIFIED' if unverified_quote else 'SOURCE_ATTESTED'
        canonical_row=lookup.get((int(year),str(identity)))
        canonical_grade='NOT_PROVIDED'
        if canonical_row:
            mismatches=0
            comparable=0
            for _,r in g.iterrows():
                grade=_grade_canonical(r,canonical_row)
                sr=_sign(r.retrospective_rule_result)
                if np.isfinite(grade) and np.isfinite(sr):
                    comparable+=1
                    if grade!=sr:mismatches+=1
            canonical_grade=('NOT_EVALUABLE' if comparable==0 else
                             'GRADED_MATCH' if mismatches==0 else 'CANONICAL_MISMATCH:'+str(mismatches))
        # CORE's original replay grade must use CORE's recorded line, which may
        # differ from Miner's open spread. Never substitute the Miner line.
        canonical_core_status='CORE_EXACT_PRICE_NOT_SUPPLIED'
        core_price_difference_explained=False
        if core_candidate and canonical_row and pd.notna(canonical_row.get('core_spread_home',np.nan)):
            hm=canonical_row.get('actual_home_margin',np.nan)
            if pd.notna(hm):
                core_home_grade=np.sign(float(hm)+float(canonical_row['core_spread_home']))
                expected=core_home_grade*(1 if core_side=='HOME' else -1)
                if expected==_sign(core_result):
                    canonical_core_status='CORE_REPLAY_GRADE_MATCH'
                    core_price_difference_explained=bool(core_rule_grade_disagreement and
                        canonical_grade=='GRADED_MATCH' and
                        not np.isclose(float(canonical_row['core_spread_home']),float(canonical_row['spread_home'])))
                else:
                    canonical_core_status='CORE_REPLAY_GRADE_MISMATCH'
        if not rule_internal_grade_consistent or not core_consistent or not price_consistent:
            integrity='HOLD_INTRA_GAME_CONTRADICTION'
        elif canonical_row and canonical_grade!='GRADED_MATCH':
            integrity='HOLD_CANONICAL_GRADE_MISMATCH'
        elif canonical_core_status=='CORE_REPLAY_GRADE_MISMATCH':
            integrity='HOLD_CORE_REPLAY_GRADE_MISMATCH'
        elif core_price_difference_explained:
            integrity='PARTIAL_DIFFERENT_QUOTES_EXPLAINED_VINTAGE_UNVERIFIED'
        elif core_rule_grade_disagreement:
            integrity='HOLD_CORE_VS_RULE_QUOTE_OR_SETTLEMENT_BRIDGE'
        elif canonical_row:
            integrity='PARTIAL_CANONICAL_GRADED_UNCERTIFIED_VINTAGE'
        else:integrity='HOLD_CANONICAL_SCORE_AND_QUOTE_BRIDGE_MISSING'
        # Internal grading disagreement by source_game occurs even if we lack
        # an externally attested canonical CFBD margin; never call it certified.
        conflict=len([v for v in sides.values() if v>0])>1
        best=g.iloc[0]  # rank pre-sorted once above
        decisions={p:_policy_decision(p,core_side if core_candidate else '',supporting,opposing) for p in POLICIES}
        game_rows.append({'market':market,'test_season':int(year),'source_identity':str(identity),
            'source_game_id':str(g.source_game_id.iloc[0]),
            'market_price_in_ledger_unverified':price,'market_price_reason':market_quote_reason,
            'rule_firings':len(g),'unique_rule_ids':g.rule_id.nunique(),
            'distinct_source_categories':len(set().union(*g.category_tokens.to_list())),
            'both_sides_fire':conflict,'side_firings':json.dumps(sides,sort_keys=True),
            'side_categories':json.dumps({k:sorted(v) for k,v in source_cat.items()},sort_keys=True),
            'category_diversity_not_independence_certification':True,
            'rule_internal_grade_consistent':rule_internal_grade_consistent,
            'price_consistent_within_ledger':price_consistent,
            'core_consistent_within_ledger':core_consistent,
            'core_candidate_in_ledger':core_candidate,
            'core_side':core_side,'core_result_unverified':core_result,'core_edge_points_unverified':core_edge,
            'core_support_categories':len(supporting),'core_opposition_categories':len(opposing),
            'core_support_category_names':'|'.join(sorted(supporting)),
            'core_opposition_category_names':'|'.join(sorted(opposing)),
            'core_rule_grading_disagreement':core_rule_grade_disagreement,
            'canonical_reconciliation':canonical_grade,
            'canonical_core_reconciliation':canonical_core_status,
            'core_price_difference_explained':core_price_difference_explained,
            'integrity_status':integrity,
            'best_ranked_rule_id':best.rule_id,'best_ranked_rule_side':best.rule_decision_side,
            'best_ranked_rule_result_unverified':best.retrospective_rule_result,
            **{'policy_'+p:decision for p,decision in decisions.items()},
            'market_quote_asof_verified':not unverified_quote,
            'point_in_time_features_verified':not unverified_features,
            'bet_authority':0})
        rec_rows.append({'market':market,'test_season':int(year),'source_identity':str(identity),
                        'rule_firings':len(g),'rule_internal_grade_consistent':rule_internal_grade_consistent,
                        'historical_quote_consistent':price_consistent,
                        'core_present':core_candidate,
                        'core_rule_same_quote_grade_disagreement':core_rule_grade_disagreement,
                        'canonical_reconciliation':canonical_grade,
                        'canonical_core_reconciliation':canonical_core_status,
                        'integrity_status':integrity,'production_authority':0})
    games=pd.DataFrame(game_rows)
    reconciliation=pd.DataFrame(rec_rows)
    # Fixed shadow policies, evaluated on the exact same CORE games; this is not
    # an historical executable bet ROI or untouched experiment.
    matched=games.loc[games.core_candidate_in_ledger].copy()
    summaries=[]
    for year,frame in [('ALL',matched), *[(str(y),g) for y,g in matched.groupby('test_season')]]:
        for policy in POLICIES:
            action=frame['policy_'+policy]
            take=action.eq('KEEP_CORE')
            core=frame.core_result_unverified
            baseline=core.isin(RESULTS)
            complete=take&baseline
            w=int((complete&core.eq('WIN')).sum());l=int((complete&core.eq('LOSS')).sum())
            pw=int((baseline&core.eq('WIN')).sum());pl=int((baseline&core.eq('LOSS')).sum())
            n=w+l;pn=pw+pl
            summaries.append({'test_season':year,'policy':policy,
                'matched_core_candidate_games':len(frame),'baseline_wins':pw,'baseline_losses':pl,
                'baseline_winrate':pw/pn if pn else np.nan,
                'selected_games':int(take.sum()),'selected_wins':w,'selected_losses':l,
                'selected_pushes':int((complete&core.eq('PUSH')).sum()),
                'selected_winrate':w/n if n else np.nan,
                'descriptive_selected_minus_baseline_winrate_pp':100*(w/n-pw/pn) if n and pn else np.nan,
                'opposed_conflict_games':int(frame.both_sides_fire.sum()),
                'core_rule_grade_disagreement_games':int(frame.core_rule_grading_disagreement.sum()),
                'executable_price_verified_games':int(frame.market_quote_asof_verified.sum()),
                'certified_roi':'HOLD_NO_VERIFIED_ASOF_QUOTES',
                'certified_incremental_lift':'HOLD_NONINDEPENDENT_RETROSPECTIVE_VINTAGE',
                'production_authority':0})
    paired=pd.DataFrame(summaries)
    # V6.2 situational-edge research: same existing module, no new Python scripts.
    situational=build_situational_edge_research(d,games)
    # Run-level quality gates fail closed, not implicit PASS through empty inputs.
    incomplete={
        'canonical_scores_missing':canonical is None,
        'core_original_spread_bridge_missing':canonical is None or 'core_spread_home' not in canonical.columns or
            bool(games.loc[games.core_candidate_in_ledger,'canonical_core_reconciliation'].eq('CORE_EXACT_PRICE_NOT_SUPPLIED').any()),
        'original_pregame_quote_timestamps_unverified':not games.market_quote_asof_verified.all(),
        'historical_features_vintage_unverified':not games.point_in_time_features_verified.all(),
        'original_historical_bigal_picks_unverified':True,
        'catalog_incumbent_comparison_missing':True,
        'historical_brain_event_manifest_missing':True,
        'paired_core_grading_disagreement':bool(games.core_rule_grading_disagreement.any()),
        'rule_game_settlement_contradiction':bool((~games.rule_internal_grade_consistent).any()),
        'one_rule_independent_family_qualification_unverified':True,
        'independent_untouched_oos_not_certified':True,
    }
    counts={str(m):{'unique_games':int(len(g)),'rule_conflict_games':int(g.both_sides_fire.sum()),
                    'internal_grade_contradictions':int((~g.rule_internal_grade_consistent).sum()),
                    'core_candidates_in_ledger':int(g.core_candidate_in_ledger.sum()),
                    'core_grade_disagreement_games':int(g.core_rule_grading_disagreement.sum())}
            for m,g in games.groupby('market')}
    return {**situational,'game_level_evidence_v6':games,'paired_core_shadow_policies_v6':paired,
        'settlement_reconciliation_v6':reconciliation,
        'validation_audit_v6':{'version':VERSION,'status':'RESEARCH_DIAGNOSTIC_ONLY_CERTIFICATION_HOLD',
            'input_ledger_rows':int(len(d)), 'unique_market_games':int(len(games)),
            'unique_rule_id_years':int(len(f)), 'current_research_recipes':int(n_current_recipes),
            'counts_by_market':counts,'hold_gates':incomplete,
            'canonical_regraded_rule_games':int(games.canonical_reconciliation.eq('GRADED_MATCH').sum()),
        'core_exact_price_reconciled_games':int(games.canonical_core_reconciliation.eq('CORE_REPLAY_GRADE_MATCH').sum()),
        'moneyline_roi_qualified':False,'features_asof_certified':False,
            'sources_independent_certified':False,'situational_edge_v6_2':situational['situational_edge_audit_v6'],
            'production_mutation':False,
            'automatic_promotion':False,'bet_authority':0}}


def save(outputs,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for k,v in outputs.items():
        p=destination/(k+('.csv' if isinstance(v,pd.DataFrame) else '.json'))
        if isinstance(v,pd.DataFrame):v.to_csv(p,index=False)
        else:p.write_text(json.dumps(v,indent=2,sort_keys=True),encoding='utf-8')
    return destination


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,help='Directory containing immutable V5.2 CSVs')
    p.add_argument('--gcs-prefix',help='gs://sharp-models/research/ncaaf/v2/miner_exact_path_shadow/runs/<timestamp>')
    p.add_argument('--canonical-results',type=Path,help='Optional independent standardized source game results CSV')
    p.add_argument('--output-dir',type=Path,default=Path('ncaaf_validation_v6_results'))
    p.add_argument('--publish-research-gcs',help='Research-only immutable gs:// prefix; no current_report mutation')
    args=p.parse_args(argv)
    if not args.gcs_prefix and not args.input_dir:p.error('--input-dir or --gcs-prefix required')
    if args.gcs_prefix:
        prefix=args.gcs_prefix.rstrip('/')
        if not prefix.startswith('gs://sharp-models/research/ncaaf/v2/miner_exact_path_shadow/runs/'):
            p.error('GCS_SOURCE_NOT_ON_EXPECTED_RESEARCH_PATH')
        tmp=tempfile.TemporaryDirectory()
        folder=Path(tmp.name)
        for name in CSV_FILES:
            subprocess.run(['gsutil','cp',prefix+'/'+name,str(folder/name)],check=True)
    else:folder=args.input_dir
    canon=pd.read_csv(args.canonical_results,low_memory=False) if args.canonical_results else None
    outputs=build(*read_inputs(folder),canonical=canon)
    dest=save(outputs,args.output_dir)
    print('VALIDATION',outputs['validation_audit_v6']['status'])
    print('UNIQUE_MARKET_GAMES',outputs['validation_audit_v6']['unique_market_games'])
    print('CORE_CANDIDATES',int(outputs['game_level_evidence_v6'].core_candidate_in_ledger.sum()))
    print('OUTPUT',dest.resolve())
    if args.publish_research_gcs:
        out=args.publish_research_gcs.rstrip('/')
        if not out.startswith('gs://sharp-models/research/ncaaf/v2/') or '/runs/' not in out:
            p.error('INVALID_RESEARCH_ONLY_PUBLISH_PREFIX')
        for f in dest.iterdir():subprocess.run(['gsutil','-h','x-goog-if-generation-match:0','cp',str(f),out+'/'+f.name],check=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
