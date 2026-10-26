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

VERSION = 'NCAAF_DECISION_VALIDATION_V6_20261025'
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
        best=g.sort_values(['frozen_rule_rank','rule_id']).iloc[0]
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
    return {'game_level_evidence_v6':games,'paired_core_shadow_policies_v6':paired,
        'settlement_reconciliation_v6':reconciliation,
        'validation_audit_v6':{'version':VERSION,'status':'RESEARCH_DIAGNOSTIC_ONLY_CERTIFICATION_HOLD',
            'input_ledger_rows':int(len(d)), 'unique_market_games':int(len(games)),
            'unique_rule_id_years':int(len(f)), 'current_research_recipes':int(n_current_recipes),
            'counts_by_market':counts,'hold_gates':incomplete,
            'canonical_regraded_rule_games':int(games.canonical_reconciliation.eq('GRADED_MATCH').sum()),
        'core_exact_price_reconciled_games':int(games.canonical_core_reconciliation.eq('CORE_REPLAY_GRADE_MATCH').sum()),
        'moneyline_roi_qualified':False,'features_asof_certified':False,
            'sources_independent_certified':False,'production_mutation':False,
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
