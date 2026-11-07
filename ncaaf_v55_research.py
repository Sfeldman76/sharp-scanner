#!/usr/bin/env python3
"""NCAAF V5.5 authoritative ROI-edge RESEARCH engine (no NCAAF module imports).

Input is the ORIGINAL, PRE-GAME captured seven-brain bet-event ledger. This file
never infers original signals/quotes from retrospective context. It discovers
hypotheses ONLY from discovery-period events, verifies rule freeze provenance,
tests 2024-25 or 2026+ OOS, calculates realized per-unit betting ROI, and
publishes a hash-checked registry only when the capture audit is independently
attested. No predictive win-probability model and no live betting.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, itertools, json, math, os, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCHEMA='NCAAF-V55-TWO-ENGINE-ROI-REGISTRY-V1'
BRAINS=frozenset(('CORE','STAT','SYSTEM_MINER','PATHI','BIG_AL','PREDICTION_TRACKER','MARKET_RICH'))
MARKETS=frozenset(('spreads','h2h','totals'))
CERT_STATUS='VALIDATED_OOS_BETTING_ROI'
REQUIRED=('game_id','season','market','side','book','odds','kickoff_utc','quote_observed_at_utc',
          'signal_observed_at_utc','feature_asof_utc','source_family','system_id',
          'evidence_family_key','home_score','away_score','signal_capture_ref','quote_capture_ref',
          'definition_frozen_at_utc','selection_end_season','frozen_definition_sha256')
AUDIT_BOOL=('original_signal_capture_verified','original_price_capture_verified',
            'feature_vintage_verified','final_score_identity_verified',
            'all_signal_occurrences_accounted_for','no_retroactive_signal_reconstruction',
            'selection_freeze_verified','test_universe_complete')

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False,default=str).encode()).hexdigest()

def clock(v):
    try:
        d=datetime.fromisoformat(str(v).replace('Z','+00:00'))
        return d.astimezone(timezone.utc) if d.tzinfo else None
    except (TypeError, ValueError, OverflowError):return None

def num(v):
    try:
        n=float(v);return n if math.isfinite(n) else None
    except (TypeError,ValueError):return None

def yes(v):return v is True or str(v).strip().lower() in ('true','1','yes')

def normal_market(v):
    s=str(v or '').lower().strip()
    return {'spread':'spreads','ats':'spreads','moneyline':'h2h','ml':'h2h','total':'totals'}.get(s,s)

def normal_family(v):
    s=str(v or '').upper().strip()
    return {'PT':'PREDICTION_TRACKER','MINER':'SYSTEM_MINER','BIGAL':'BIG_AL'}.get(s,s)

def odds_profit(odds):
    v=num(odds)
    if v is None or (-100<v<100):return None
    return v/100 if v>=100 else 100/(-v)

def grade(market,side,line,home,away):
    h,a=num(home),num(away)
    if h is None or a is None or h<0 or a<0 or h!=int(h) or a!=int(a):return None
    m=normal_market(market);s=str(side or '').upper();l=num(line)
    if m=='h2h':score=h-a if s=='HOME' else a-h if s=='AWAY' else None
    elif m=='spreads':score=(h-a if s=='HOME' else a-h if s=='AWAY' else 0)+l if l is not None and s in ('HOME','AWAY') else None
    elif m=='totals':score=(h+a-l)*(1 if s=='OVER' else -1) if l is not None and s in ('OVER','UNDER') else None
    else:score=None
    if score is None:return None
    return 'WIN' if score>1e-7 else 'LOSS' if score< -1e-7 else 'PUSH'

def row_normal(raw):
    miss=[x for x in REQUIRED if x not in raw or raw[x] is None or str(raw[x]).strip()=='']
    if miss:raise ValueError('MISSING_FIELDS:'+','.join(miss[:8]))
    fam=normal_family(raw['source_family']);m=normal_market(raw['market']);side=str(raw['side']).upper().strip()
    if fam not in BRAINS or m not in MARKETS or side not in (('OVER','UNDER') if m=='totals' else ('HOME','AWAY')):
        raise ValueError('INVALID_BRAIN_MARKET_SIDE')
    kick,quote,obs,feat,freeze=[clock(raw[k]) for k in ('kickoff_utc','quote_observed_at_utc','signal_observed_at_utc','feature_asof_utc','definition_frozen_at_utc')]
    if None in (kick,quote,obs,feat,freeze) or not feat<=obs<=quote<kick or (quote-obs).total_seconds()>5400:
        raise ValueError('POSTGAME_OR_UNVERIFIED_TIMESTAMP')
    season=int(raw['season']);select_end=int(raw['selection_end_season'])
    if not (2013<=season<=2026 and 2013<=select_end<=2025 and kick.year in (season,season+1)):
        raise ValueError('INVALID_SEASON')
    if freeze>obs or select_end>=season:
        raise ValueError('RULE_NOT_FROZEN_BEFORE_ORIGINAL_GAME_SIGNAL')
    if len(str(raw['frozen_definition_sha256']).strip())!=64 or any(c not in '0123456789abcdef' for c in str(raw['frozen_definition_sha256']).lower()):
        raise ValueError('MISSING_FROZEN_DEFINITION_HASH')
    for k in ('original_pregame_signal_verified','original_pregame_quote_verified','feature_vintage_verified'):
        if not yes(raw.get(k)):raise ValueError('SOURCE_CAPTURE_NOT_VERIFIED:'+k)
    line=num(raw.get('line'))
    if m!='h2h' and (line is None or line*2!=round(line*2)):
        raise ValueError('INVALID_HALF_POINT_LINE')
    if m=='h2h':line=None
    odds=num(raw['odds']);gain=odds_profit(odds)
    if gain is None:raise ValueError('INVALID_PRICE')
    res=grade(m,side,line,raw['home_score'],raw['away_score'])
    if res is None:raise ValueError('INVALID_SCORES')
    game=str(raw['game_id']).strip();book=str(raw['book']).lower().strip()
    if not game or not book:raise ValueError('MISSING_GAME_OR_BOOK')
    return dict(game_id=game,season=season,market=m,side=side,book=book,line=line,odds=odds,
        kickoff=kick,quote=quote,observed=obs,feature=feat,freeze=freeze,selection_end=select_end,
        family=fam,system_id=str(raw['system_id']).strip(),dependency=str(raw['evidence_family_key']).upper().strip(),
        component=fam+'|'+str(raw['system_id']).strip(),hash=str(raw['frozen_definition_sha256']).lower(),
        signal_ref=str(raw['signal_capture_ref']).strip(),price_ref=str(raw['quote_capture_ref']).strip(),
        home=int(float(raw['home_score'])),away=int(float(raw['away_score'])),result=res,
        profit=(gain if res=='WIN' else -1.0 if res=='LOSS' else 0.0))

def line_movement_research(groups, *, movement_points=(0.5, 1.0, 2.0, 3.0, 4.0, 6.0)):
    """Measure how existing betting results change at *hypothetical* tougher lines.

    Each recorded final score allows exact ATS/push settlement at another line,
    independent of whether a book ever quoted it. These are line-sensitivity
    outcomes, NEVER a claim that the alternative was a priced/executable bet.

    Independently, inspect *actual captured quote sequences* to describe
    opening-to-later line movement. A movement is observed only if two original
    pregame quote snapshots exist for the same game/market/side/book; a final
    outcome by itself is never labeled an 'overmove'.
    """
    unique={}
    for g in groups:
        r=g['record']
        key=(r['game_id'],r['market'],r['side'],r['book'],r['line'],r['quote'],r['odds'])
        unique[key]=r
    graded=list(unique.values())
    simulations={}
    for points in movement_points:
        if points<=0 or int(points*2)!=points*2:raise ValueError('INVALID_MOVEMENT_STEP')
        counters=Counter()
        games=set()
        for r in graded:
            if r['market'] not in ('spreads','totals'):continue
            # Moving the offer against the selected position always makes
            # the bet harder to win. For spreads, fewer awarded points is
            # harder for HOME or AWAY; for totals, OVER needs a higher total
            # and UNDER needs a lower total.
            alternate=(r['line']-points if r['market']=='spreads' or r['side']=='UNDER'
                       else r['line']+points)
            result=grade(r['market'],r['side'],alternate,r['home'],r['away'])
            counters[r['result']+'->'+result]+=1
            games.add(r['game_id'])
        if counters:
            simulations[str(points)]={
                'original_quote_rows':sum(counters.values()),
                'unique_games':len(games),
                'wins_turning_into_push_or_loss':sum(v for k,v in counters.items()
                                                    if k.startswith('WIN->') and k!='WIN->WIN'),
                'graded_outcome_transitions':dict(sorted(counters.items())),
                'alternative_line_executable_quote_claimed':False,
                'alternative_line_roi_certified':False}
    chains=defaultdict(list)
    for r in graded:
        if r['market'] in ('spreads','totals'):
            chains[(r['game_id'],r['market'],r['side'],r['book'])].append(r)
    observed=Counter()
    movement_examples=[]
    for key,chain in chains.items():
        chain.sort(key=lambda x:(x['quote'],x['price_ref']))
        first,last=chain[0],chain[-1]
        if first['quote']==last['quote'] or first['line']==last['line']:continue
        # Opening/last in this chain mean earliest/latest *captured*, not
        # necessarily the market's official opening/closing line.
        adverse=(first['line']-last['line'] if first['market']=='spreads' or first['side']=='UNDER'
                 else last['line']-first['line'])
        direction='ADVERSE' if adverse>0 else 'FAVORABLE'
        band='3_PLUS' if abs(adverse)>=3 else '1_TO_UNDER_3' if abs(adverse)>=1 else 'UNDER_1'
        observed[direction+'_'+band]+=1
        before=grade(first['market'],first['side'],first['line'],first['home'],first['away'])
        after=grade(last['market'],last['side'],last['line'],last['home'],last['away'])
        observed['ACTUAL_OBSERVED_QUOTES_'+before+'_TO_'+after]+=1
        if len(movement_examples)<20:
            movement_examples.append({'game_id':key[0], 'market':key[1], 'side':key[2],
                'book':key[3], 'earliest_captured_line':first['line'],
                'latest_captured_line':last['line'], 'adverse_points':adverse,
                'earliest_capture_utc':first['quote'].isoformat(),
                'latest_capture_utc':last['quote'].isoformat(),
                'earliest_grade':before,'latest_grade':after})
    return {'method':'REGRADING_AT_ALTERNATIVE_LINES_AND_CAPTURED_MARKET_MOVES',
            'alternative_spread_totals_scenarios':simulations,
            'observed_original_quote_movement_counts':dict(sorted(observed.items())),
            'observed_move_examples':movement_examples,
            'status':'DESCRIPTIVE_ONLY_REQUIRES_FROZEN_DISCOVERY_AND_OOS_VALIDATION',
            'not_a_certified_betting_edge':True}

def _stratum(r):
    # Executed-wager ROI uses its original sportsbook line and odds. A -3
    # historical result ALSO contributes to hypothetical -6 line sensitivity
    # in line_movement_research(), but not as a fabricated -6 priced wager.
    return (r['market'],r['side'],r['book'],r['line'])

def _group(rows):
    """One original quote per game/side/book/line, containing fired systems.

    Conflicting grade, timestamp, or prices for a purported shared quote HOLD
    the entire ledger. One historical event never counts twice in a system.
    """
    out={}
    for r in rows:
        key=(r['game_id'],)+_stratum(r)
        g=out.setdefault(key,{'components':{},'record':r})
        p=g['record']
        if (p['home'],p['away'],p['kickoff'])!=(r['home'],r['away'],r['kickoff']):
            raise ValueError('CONFLICTING_FINAL_OR_KICKOFF:'+r['game_id'])
        if abs(p['odds']-r['odds'])>1e-6 or p['quote']!=r['quote']:
            # Do not combine systems fired at different quotes/prices.
            # Preserve disjoint quote observations by keying on actual quote.
            key=key+(r['quote'].isoformat(),r['odds'])
            g=out.setdefault(key,{'components':{},'record':r})
        existing=g['components'].get(r['component'])
        if existing and existing['dependency']!=r['dependency']:
            raise ValueError('INCONSISTENT_EVIDENCE_DEPENDENCY')
        if existing is None or r['observed']<existing['observed']:
            g['components'][r['component']]=r
    return list(out.values())

def _return_lower(profits,alpha):
    # Hoeffding's one-sided conservative bound on actual per-dollar returns.
    # Unlike a sample-variance t interval, this does not report zero uncertainty
    # when every observed wager wins. Bonferroni adjusts all tested rules.
    n=len(profits)
    if n<2 or not 0<alpha<1:return None
    lo=-1.0;hi=max(0.0,max(profits))
    return sum(profits)/n-(hi-lo)*math.sqrt(math.log(1/alpha)/(2*n))

def _audit_ok(audit,source_sha):
    return (isinstance(audit,dict) and audit.get('status')=='PASS'
        and audit.get('observations_sha256')==source_sha
        and all(yes(audit.get(k)) for k in AUDIT_BOOL)
        and bool(audit.get('independent_capture_audit_ref')))

def candidate_combinations(eligible,max_combo=7):
    """Enumerate all 1..7 distinct-brain-family combinations, no CORE gate.

    Different systems in one brain are not automatically independent support.
    They remain alternative choices within that family; a rule may already
    encode multi-system Miner confluence in its frozen definition.
    """
    families=defaultdict(list)
    for system,rec in eligible.items():
        families[rec['family']].append(system)
    names=sorted(families)
    for n in range(1,min(max_combo,len(names))+1):
        for fams in itertools.combinations(names,n):
            for parts in itertools.product(*(sorted(families[f]) for f in fams)):
                if len({eligible[k]['dependency'] for k in parts})==len(parts):
                    yield parts

def research(records,*, capture_audit=None, source_sha256=None, holdout_start=2024,
             min_train_games=20,min_oos_games=100,min_year_games=25,min_roi=.02,max_combo=7):
    """Discover all observed single/pair hypotheses from TRAIN only, then OOS.

    A rule discovered/frozen in 2025 cannot be independently certified on 2024
    or 2025. We require pre-holdout freeze proof for each component; new 2025
    rules must await independently locked 2026+ evidence.
    """
    if hasattr(records,'to_dict'):records=records.to_dict('records')
    if not isinstance(records,list):records=list(records)
    source_sha256=source_sha256 or digest(records)
    diag=Counter();clean=[]
    for raw in records:
        try:clean.append(row_normal(raw))
        except (ValueError,TypeError) as e:diag[str(e)]+=1
    root={'schema':SCHEMA,'policy':'HISTORICAL_BETTING_ROI_NOT_PREDICTIVE_PROBABILITY',
          'version':'two-authoritative-engines-r1','created_utc':datetime.now(timezone.utc).isoformat(),
          'source_sha256':source_sha256,'capture_audit_sha256':digest(capture_audit) if capture_audit else None,
          'holdout_start':holdout_start,'raw_rows':len(records),'clean_rows':len(clean),
          'invalid_rows':dict(diag),'certificates':[],'rejected':{},'research_only':True,
          'historical_price_scope':'EXACT_SPORTSBOOK_AND_LINE'}
    if diag:
        root['status']='HOLD_INVALID_OR_INCOMPLETE_ORIGINAL_CAPTURE';return _seal(root)
    if not clean:
        root['status']='HOLD_NO_ORIGINAL_PREGAME_BET_EVENTS';return _seal(root)
    try:
        identities={}
        for r in clean:
            identity=(r['home'],r['away'],r['kickoff'])
            old=identities.setdefault(r['game_id'],identity)
            if identity!=old:raise ValueError('CONFLICTING_FINALS_OR_KICKOFF:'+r['game_id'])
        groups=_group(clean)
    except ValueError as e:
        root['status']='HOLD_CONFLICTING_SOURCE';root['error']=str(e);return _seal(root)
    # Actual unit-stake ROI from independently provided original game/price
    # observations. This descriptive result is NOT an OOS certificate. It lets
    # research expose prices, graded outcomes, and available betting ROI even
    # when an external source-capture attestation is still missing.
    descriptive={}
    sample_quotes={}
    for g in groups:
        r=g['record']
        key=(r['game_id'],r['market'],r['side'],r['book'],r['line'],r['quote'],r['odds'])
        sample_quotes[key]=r
    for m in sorted(MARKETS):
        ms=[r for r in sample_quotes.values() if r['market']==m]
        if ms:
            descriptive[m]={'unique_graded_bet_quotes':len(ms),
                            'wins':sum(r['result']=='WIN' for r in ms),
                            'losses':sum(r['result']=='LOSS' for r in ms),
                            'pushes':sum(r['result']=='PUSH' for r in ms),
                            'descriptive_realized_roi':sum(r['profit'] for r in ms)/len(ms),
                            'independent_out_of_sample':False}
    root['historical_roi_diagnostics']=descriptive
    root['line_movement_research']=line_movement_research(groups)
    if not _audit_ok(capture_audit,source_sha256):
        root['status']='HOLD_INDEPENDENT_ORIGINAL_CAPTURE_AUDIT_REQUIRED';return _seal(root)
    # Train-only selection from components that were demonstrably frozen before
    # 2024. No OOS outcome can create a new candidate or price stratum.
    train=[g for g in groups if g['record']['season']<holdout_start]
    counts=Counter();meta={};train_seen=set()
    for g in train:
        r=g['record'];stratum=_stratum(r)
        eligible={k:v for k,v in g['components'].items()
                  if v['selection_end']<holdout_start and v['freeze']<datetime(holdout_start,1,1,tzinfo=timezone.utc)}
        for pair in candidate_combinations(eligible,max_combo):
            hypo=(stratum,pair)
            if (hypo,r['game_id']) not in train_seen:
                counts[hypo]+=1
                train_seen.add((hypo,r['game_id']))
            proof=tuple((k,eligible[k]['hash'],eligible[k]['selection_end'],eligible[k]['freeze'].isoformat(),eligible[k]['dependency']) for k in pair)
            if hypo in meta and meta[hypo]!=proof:raise ValueError('SYSTEM_DEFINITION_CHANGED_INSIDE_DISCOVERY')
            meta[hypo]=proof
            if len(meta)>250000:
                root['status']='HOLD_EXHAUSTIVE_HYPOTHESIS_BUDGET_EXCEEDED'
                root['explored_hypotheses']=len(meta)
                return _seal(root)
    hypotheses=sorted((h for h,n in counts.items() if n>=min_train_games),key=str)
    root['train_games']=len({g['record']['game_id'] for g in train})
    root['tested_hypotheses']=len(hypotheses)
    root['historical_source_seasons']=sorted(set(x['season'] for x in clean))
    if not hypotheses:
        root['status']='HOLD_NO_FROZEN_PRE_HOLDOUT_HYPOTHESES';return _seal(root)
    # Frozen selection universe includes ALL train-selected combinations, not
    # merely combinations that happened to be profitable later.
    half={} ;hset=set(hypotheses)
    oos_groups=[g for g in groups if g['record']['season']>=holdout_start]
    for g in oos_groups:
        r=g['record'];stratum=_stratum(r);eligible=g['components']
        for pair in candidate_combinations(eligible,max_combo):
            hypo=(stratum,pair)
            if hypo not in hset:continue
            proof=tuple((k,eligible[k]['hash'],eligible[k]['selection_end'],eligible[k]['freeze'].isoformat(),eligible[k]['dependency']) for k in pair)
            if proof!=meta[hypo] or any(eligible[k]['selection_end']>=r['season'] or eligible[k]['freeze']>=r['observed'] for k in pair):
                continue
            key=(hypo,r['game_id'])
            prev=half.get(key)
            # Earliest original quote wins; no outcome-based price shopping.
            if prev is None or (r['quote'],r['book'])<(prev['quote'],prev['book']):half[key]=r
    alpha=.05/len(hypotheses)
    for hypo in hypotheses:
        (market,side,book,line),parts=hypo
        matches=sorted((r for (h,_),r in half.items() if h==hypo),key=lambda r:(r['kickoff'],r['game_id']))
        n=len(matches);profits=[r['profit'] for r in matches];yrs=defaultdict(list)
        for r in matches:yrs[r['season']].append(r['profit'])
        roi=sum(profits)/n if n else None
        lower=_return_lower(profits,alpha) if n>1 else None
        problems=[]
        if n<min_oos_games:problems.append('OOS_GAME_SAMPLE_INSUFFICIENT')
        if len(yrs)<2:problems.append('OOS_SEASONS_INSUFFICIENT')
        if any(len(v)<min_year_games or sum(v)<=0 for v in yrs.values()):problems.append('OOS_SEASON_STABILITY_FAILED')
        if roi is None or roi<min_roi:problems.append('OOS_REALIZED_ROI_BELOW_MIN')
        if lower is None or lower<=0:problems.append('MULTIPLICITY_ADJUSTED_OOS_ROI_BOUND_NOT_POSITIVE')
        identifier=digest((market,side,book,line,parts))[:24]
        if problems:
            root['rejected'][identifier]={'holds':problems,'market':market,'side':side,'book':book,'line':line,
                                          'components':list(parts),'oos_games':n,'roi':roi,'roi_lower':lower}
            continue
        proof=meta[hypo]
        cert={'id':identifier,'market':market,'side':side,'book':book,'line':line,
              'components':list(parts),'dependencies':{c[0]:c[4] for c in proof},
              'definition_hashes':{c[0]:c[1] for c in proof},
              'freeze_utc':{c[0]:c[3] for c in proof},'selection_end':{c[0]:c[2] for c in proof},
              'oos_unique_games':n,'holdout_seasons':sorted(yrs),
              'by_season':{str(y):{'n':len(v),'roi':sum(v)/len(v)} for y,v in yrs.items()},
              'wins':sum(x['result']=='WIN' for x in matches),
              'losses':sum(x['result']=='LOSS' for x in matches),
              'pushes':sum(x['result']=='PUSH' for x in matches),
              'realized_roi':roi,'oos_roi_lower_adjusted':lower,
              'min_historical_odds':min(x['odds'] for x in matches),
              'observed_odds':sorted({x['odds'] for x in matches}),
              'tested_hypotheses':len(hypotheses),'source_sha256':source_sha256,
              'confidence_method':'ONE_SIDED_BONFERRONI_HOEFFDING_RETURN_BOUND',
              'status':CERT_STATUS}
        root['certificates'].append(cert)
    root['certificates']=sorted(root['certificates'],key=lambda c:c['id'])
    root['status']='PASS_CERTIFIED_ROI_REGISTRY' if root['certificates'] else 'HOLD_NO_OOS_POSITIVE_ROI_EDGES'
    root['research_only']=not bool(root['certificates'])
    root['accepted_edges']=len(root['certificates'])
    return _seal(root)

def _seal(doc):
    doc['certificates_sha256']=digest(doc['certificates'])
    return doc

def _bq_dict(record):
    return dict(record.items()) if hasattr(record,'items') else dict(record)


def _entity_key(name):
    import re
    return re.sub(r'[^a-z0-9]', '', str(name or '').lower())


def _source_research_rows(sql, bq, *, limit=None):
    result=bq.query(sql).result(page_size=10000)
    for idx,row in enumerate(result):
        if limit is not None and idx>=limit:
            break
        yield _bq_dict(row)


def warehouse_replay_existing_sources(*, client=None, project='sharplogger',
                                      dataset='sharp_data', max_quote_rows=200000):
    """Research directly from the EXISTING NCAAF warehouse, no new manifest.

    Three independently labeled lanes:
      A. 2013-25 historically graded ATS at source OPEN/CLOSE spreads, with
         hypothetical movement re-grades. No bookmaker ROI is claimed.
      B. Originally logged sportsbook snapshots with timestamps, odds, and
         verified matching home/away scores; one earliest quote per wager.
         Returns are OBSERVED-PRICE DESCRIPTIVE, NOT certified signal ROI.
      C. CFBD provider archive historical coverage. The provider table has
         no recorded original quote timestamp or spread/totals wager odds,
         so it does not become certified executable ROI.

    A historical snapshot and a score match do NOT establish an originally
    fired betting brain. That requires source-specific signals and freeze proof.
    No BET certificate can be produced from this function alone.
    """
    import re
    from google.cloud import bigquery
    for name,kind in ((project,'project'),(dataset,'dataset')):
        pattern=r'[A-Za-z][A-Za-z0-9_-]*' if kind=='project' else r'[A-Za-z_][A-Za-z0-9_]*'
        if not re.fullmatch(pattern,name):raise ValueError('INVALID_'+kind.upper())
    if max_quote_rows<=0 or max_quote_rows>1000000:raise ValueError('INVALID_REPLAY_ROW_LIMIT')
    bq=client or bigquery.Client(project=project)
    fq=lambda t: '`'+project+'.'+dataset+'.'+t+'`'
    meta=f'`{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`'
    schema=defaultdict(set)
    for row in _source_research_rows('SELECT table_name,column_name FROM '+meta,bq):
        schema[str(row['table_name'])].add(str(row['column_name']))
    report={'status':'RESEARCH_WAREHOUSE_REPLAY_DIAGNOSTIC_ONLY',
        'source_tables':{},'graded_games':0,'priced_game_wagers':0,
        'unmatched_snapshot_game_keys':0,'rejected_snapshot_rows':dict(),
        'original_brain_signal_provenance_certified':False,
        'historical_executable_roi_certified':False,'bet_authority':0,
        'source_role':'CFBD games/results; historical quotes from original provider/source tables',
        'roi_policy':'RECORDED_PREGAME_PRICE_ROI_DESCRIPTIVE_ONLY; NO_SYNTHETIC_BOOK_PRICES',
        'historical_range':'2013-2025', 'brain_families_requested':sorted(BRAINS),
        'cfbd_role_crosswalk_v5_applied':False,
        'verification_note':'Only source Is_Home team-role rows are graded; independent CFBD stage1 role-crosswalk is NOT claimed by this replay.'}
    hist='ncaaf_historical_core_training_vw'
    hist_need={'Season','Game_Key','Is_Home','Team_Score','Opponent_Score','Opening_Spread',
               'Canonical_Team_Norm','Opponent_Canonical_Team_Norm'}
    hist_missing=sorted(hist_need-schema.get(hist,set()))
    report['source_tables'][hist]={'present':hist in schema,'missing_required_columns':hist_missing}
    games={};summary=defaultdict(Counter);movement=defaultdict(Counter)
    if not hist_missing:
        # One GAME per canonical Game_Key. The HOME side is not inferred from
        # score margin or quote outcomes. Conflicting same-key rows are held.
        close='SAFE_CAST(Consensus_Close_Spread_Audit AS FLOAT64)' if 'Consensus_Close_Spread_Audit' in schema[hist] else 'CAST(NULL AS FLOAT64)'
        sql=(f'SELECT Season,Game_Key,Team_Score,Opponent_Score,Opening_Spread,'
             f'Canonical_Team_Norm,Opponent_Canonical_Team_Norm,{close} AS closing_spread '
             f"FROM {fq(hist)} WHERE CAST(Is_Home AS STRING) IN ('1','true','TRUE') AND Season BETWEEN 2013 AND 2025 "
             f'AND Game_Key IS NOT NULL ORDER BY Season,Game_Key')
        conflicts=set()
        for x in _source_research_rows(sql,bq):
            key=str(x['Game_Key']).strip();year=num(x['Season'])
            h,a=num(x['Team_Score']),num(x['Opponent_Score'])
            if not key or year is None or h is None or a is None:continue
            info={'season':int(year),'home':h,'away':a,
                  'home_team':_entity_key(x['Canonical_Team_Norm']),
                  'away_team':_entity_key(x['Opponent_Canonical_Team_Norm']),
                  'open_spread':num(x['Opening_Spread']),
                  'close_spread':num(x['closing_spread'])}
            if key in games and games[key]!=info:conflicts.add(key)
            else:games[key]=info
        for k in conflicts:games.pop(k,None)
        report['conflicting_result_keys_excluded']=len(conflicts)
        report['graded_games']=len(games)
        for x in games.values():
            yr=x['season']; opening=x['open_spread']; closing=x['close_spread']
            if opening is None:continue
            res=grade('spreads','HOME',opening,x['home'],x['away'])
            summary[yr]['games_with_open_spread']+=1
            summary[yr][f'OPEN_{res}']+=1
            for points in (0.5,1.0,2.0,3.0,4.0,6.0):
                tougher=grade('spreads','HOME',opening-points,x['home'],x['away'])
                movement[str(points)][f'{res}_TO_{tougher}']+=1
            if closing is not None:
                res2=grade('spreads','HOME',closing,x['home'],x['away'])
                summary[yr]['games_with_reference_close']+=1
                summary[yr][f'CLOSE_{res2}']+=1
                summary[yr]['open_to_close_home_spread_changed']+=int(closing!=opening)
        report['historical_spread_ats_by_season']={str(y):dict(v) for y,v in sorted(summary.items())}
        report['counterfactual_tougher_spreads']={p:dict(v) for p,v in sorted(movement.items())}
        report['historical_spread_ats_caveat']='Reference consensus open/close grades are NOT priced book wagers or certified system-edge ROI.'
    # This provider archive was already recovered; do not lose 2013-2025
    # historical line movement coverage just because per-book prices were not
    # captured with independently attested timestamps.
    arc='ncaaf_archive_market_provider_quotes_v1'
    arc_need={'season','game_id','provider','spread_quoted','total_quoted',
              'home_ml_quoted','away_ml_quoted'}
    arc_missing=sorted(arc_need-schema.get(arc,set()))
    report['source_tables'][arc]={'present':arc in schema,'missing_required_columns':arc_missing,
        'original_quote_timestamp_column_present':False,
        'spread_total_odds_columns_present':False}
    if not arc_missing:
        sql=(f'SELECT season,provider,COUNT(*) AS n,COUNT(DISTINCT game_id) AS games, '
             f'COUNTIF(spread_quoted IS NOT NULL) AS spread_quotes, '
             f'COUNTIF(total_quoted IS NOT NULL) AS total_quotes, '
             f'COUNTIF(home_ml_quoted IS NOT NULL OR away_ml_quoted IS NOT NULL) AS ml_quotes '
             f'FROM {fq(arc)} WHERE season BETWEEN 2013 AND 2025 '
             f'GROUP BY season,provider ORDER BY season,provider')
        rows=list(_source_research_rows(sql,bq))
        report['archived_provider_coverage']=[{k:(int(v) if k in ('season','n','games','spread_quotes','total_quotes','ml_quotes') and v is not None else v)
                                              for k,v in r.items()} for r in rows]
        report['archived_provider_quote_rows']=sum(r['n'] for r in report['archived_provider_coverage'])
        report['archived_provider_quote_limitation']='Archive has provider lines and ML quotes but no original observed-at timestamp or spread/totals bet odds: not executable ROI.'
    # Only an actually timestamped pre-kick market quote is allowed into the
    # OBSERVED-PRICE research return lane. Require both normalized team names,
    # exact historical Game_Key, and home/away score from CFBD-derived history.
    snaps='sharp_scores_full'
    snap_need={'Game_Key','Sport','Market','Outcome','Book','Value','Odds_Price',
               'Snapshot_Timestamp','Commence_Hour','Home_Team_Norm','Away_Team_Norm'}
    snap_missing=sorted(snap_need-schema.get(snaps,set()))
    report['source_tables'][snaps]={'present':snaps in schema,'missing_required_columns':snap_missing}
    # Only explicitly named historical expert flags. These are retrospective
    # audit inputs, NEVER fresh Miner mechanism certifications. Exclude
    # DataReady, trusted-rate, context and computed-score fields.
    retro_flags=sorted(c for c in schema.get(snaps,set()) if
         (c.startswith('BigAl_CF') or c.startswith('Pathi_M'))
         and not c.endswith(('_DataReady','_Match_Ratio'))
         and all(token not in c for token in ('Context','Historical','Probability','Posterior','Count')))
    report['retrospective_expert_flag_columns']=retro_flags
    report['seven_brain_status']={
        'CORE':'HISTORICAL_FEATURES_NOT_VERIFIED_ORIGINAL_CORE_PICK',
        'STAT':'CFBD_FEATURES_INPUT_NOT_DIRECTIONAL_EVENT',
        'SYSTEM_MINER':'FROZEN_DEFINITIONS_REQUIRE_POINT_IN_TIME_REPLAY',
        'PATHI':'RETROSPECTIVE_FLAGS_PRESENT_NOT_VINTAGE_CERTIFIED' if any(c.startswith('Pathi_') for c in retro_flags) else 'NO_FLAG_SOURCE',
        'BIG_AL':'RETROSPECTIVE_FLAGS_PRESENT_NOT_VINTAGE_CERTIFIED' if any(c.startswith('BigAl_') for c in retro_flags) else 'NO_FLAG_SOURCE',
        'PREDICTION_TRACKER':'EXTERNAL_ORIGINAL_TRACKER_SIGNAL_JOIN_REQUIRED',
        'MARKET_RICH':'PROVIDER_QUOTES_CONTEXT_NOT_AUTOMATIC_DIRECTIONAL_VOTE'}
    if not snap_missing and games:
        extra_flags=','+','.join('`'+c+'`' for c in retro_flags) if retro_flags else ''
        sql=(f'SELECT Game_Key,Market,Outcome,Book,Value,Odds_Price,Snapshot_Timestamp,'
             f'Commence_Hour,Home_Team_Norm,Away_Team_Norm{extra_flags} '
             f"FROM {fq(snaps)} WHERE UPPER(Sport)='NCAAF' "
             f'AND Snapshot_Timestamp IS NOT NULL AND Commence_Hour IS NOT NULL '
             f'AND Snapshot_Timestamp < Commence_Hour '
             f"AND Snapshot_Timestamp >= TIMESTAMP('2013-01-01') "
             f"AND Snapshot_Timestamp < TIMESTAMP('2026-01-01') "
             f'ORDER BY Snapshot_Timestamp,Game_Key')
        # Enforce a strict practical export budget; report truncation instead
        # of certifying partial tests. Research may still use partial diagnostics.
        rejected=Counter();wagers={};all_quotes=[];expert_events={};count=0
        for q in _source_research_rows(sql,bq,limit=max_quote_rows+1):
            count+=1
            if count>max_quote_rows:break
            key=str(q.get('Game_Key') or '').strip()
            g=games.get(key)
            if not g:
                rejected['NO_EXACT_CANONICAL_GAME_KEY']+=1;continue
            if (_entity_key(q.get('Home_Team_Norm'))!=g['home_team'] or
                _entity_key(q.get('Away_Team_Norm'))!=g['away_team']):
                rejected['HOME_AWAY_NAMES_OR_ROLE_MISMATCH']+=1;continue
            book=str(q.get('Book') or '').lower().strip()
            market=normal_market(q.get('Market'))
            outcome=_entity_key(q.get('Outcome'))
            if market not in MARKETS or not book:
                rejected['UNKNOWN_MARKET_OR_BOOK']+=1;continue
            if market=='totals':side='OVER' if outcome=='over' else 'UNDER' if outcome=='under' else None
            else:side='HOME' if outcome in ('home',g['home_team']) else 'AWAY' if outcome in ('away',g['away_team']) else None
            if not side:
                rejected['UNMAPPED_QUOTE_SIDE']+=1;continue
            observed=clock(q.get('Snapshot_Timestamp'));kickoff=clock(q.get('Commence_Hour'))
            if not observed or not kickoff or observed>=kickoff:
                rejected['INVALID_PREGAME_CAPTURE_TIME']+=1;continue
            odds=num(q.get('Odds_Price'));line=num(q.get('Value'))
            if odds_profit(odds) is None:
                rejected['MISSING_EXECUTABLE_PRICE']+=1;continue
            if market!='h2h' and line is None:
                rejected['MISSING_BET_LINE']+=1;continue
            res=grade(market,side,line,g['home'],g['away'])
            if res is None:
                rejected['UNGRADABLE_QUOTE']+=1;continue
            profit=odds_profit(odds) if res=='WIN' else -1.0 if res=='LOSS' else 0.0
            entry={'game_id':key,'market':market,'side':side,'line':line if market!='h2h' else None,
                   'book':book,'odds':odds,'quote':observed,'result':res,'profit':profit,
                   'home':g['home'],'away':g['away'],'season':g['season'],
                   'price_ref':key+'|'+book+'|'+observed.isoformat()}
            all_quotes.append({'record':entry})
            for flag in retro_flags:
                if num(q.get(flag)) != 1.0:continue
                syskey=(flag,key,market,side,book)
                prior=expert_events.get(syskey)
                if prior is None or entry['quote']<prior['quote']:
                    expert_events[syskey]=entry
            wk=(key,market,side,book)
            prev=wagers.get(wk)
            if prev is None or (entry['quote'],entry['line'] or 0)<(prev['quote'],prev['line'] or 0):
                wagers[wk]=entry
        report['source_tables'][snaps]['query_rows_scanned']=min(count,max_quote_rows)
        report['source_tables'][snaps]['hit_row_limit']=count>max_quote_rows
        report['rejected_snapshot_rows']=dict(rejected)
        report['unmatched_snapshot_game_keys']=rejected['NO_EXACT_CANONICAL_GAME_KEY']
        report['priced_game_wagers']=len(wagers)
        by_season_market=defaultdict(list)
        for v in wagers.values():by_season_market[(v['season'],v['market'])].append(v)
        report['observed_price_roi_by_year_market']=[
            {'season':year,'market':market,'unique_wagers':len(ws),
             'wins':sum(x['result']=='WIN' for x in ws),
             'losses':sum(x['result']=='LOSS' for x in ws),
             'pushes':sum(x['result']=='PUSH' for x in ws),
             'unit_stake_roi':sum(x['profit'] for x in ws)/len(ws),
             'hypothesis_selection':'NONE: market-wide descriptive, not a betting strategy',
             'original_brain_signal_matched':False,
             'bet_authority':0} for (year,market),ws in sorted(by_season_market.items())]
        expert_returns=defaultdict(list)
        for (system,game,market,side,book),rec in expert_events.items():
            expert_returns[(system,market)].append(rec)
        report['retrospective_expert_flag_roi']=[
            {'system':system,'family':'BIG_AL' if system.startswith('BigAl_') else 'PATHI',
             'market':market,'unique_game_wagers':len({r['game_id'] for r in ws}),
             'wins':sum(r['result']=='WIN' for r in ws),
             'losses':sum(r['result']=='LOSS' for r in ws),
             'pushes':sum(r['result']=='PUSH' for r in ws),
             'descriptive_unit_stake_roi':sum(r['profit'] for r in ws)/len(ws),
             'original_signal_capture_verified':False,'bet_authority':0}
            for (system,market),ws in sorted(expert_returns.items())]
        if all_quotes:
            report['observed_price_line_movement']=line_movement_research(all_quotes)
            report['observed_price_line_movement']['description']='Within-book pregame quote-to-quote movement and counterfactual cover regrading; NOT an overmove strategy certificate.'
    report['status']='RESEARCH_WAREHOUSE_REPLAY_READY_FOR_BRAIN_RETROSPECTIVE_JOIN' if report['graded_games'] else 'HOLD_NO_VERIFIED_HOME_AWAY_RESULTS'
    return report



# ----------------- Existing Stage 1 / Stage 2 seven-brain replay -----------------
# Consume immutable V5.2 canonical game IDs and existing frozen observations.
# The matching algorithm was executed and audited on November 5; do NOT
# repeat fuzzy/heuristic matching or promote retrospective events to authority.

def _canonical_name(value):
    import re
    return re.sub(r'[^a-z0-9]', '', str(value or '').lower())


def _csv_bytes(content):
    if isinstance(content, str):content=content.encode()
    return list(csv.DictReader(io.StringIO(content.decode('utf-8-sig'))))


def _obj_bytes(uri, storage_client=None):
    if uri.startswith('gs://'):
        from google.cloud import storage
        bucket, key=uri[5:].split('/',1)
        return (storage_client or storage.Client()).bucket(bucket).blob(key).download_as_bytes()
    return Path(uri).read_bytes()


def _source_inventory(bucket, storage_client, *, canonical_prefix=None):
    """No directory-wide alias rediscovery. Find existing V5.2 artifact bundle.

    Default GCS location is a *staging location*, not a claim it has been
    published. A local path from prior Cloud Shell Stage 1 work also works.
    """
    prefix=canonical_prefix or os.environ.get('NCAAF_V55_CANONICAL_LEDGER_URI','').strip() or (
        'gs://'+bucket+'/research/ncaaf/v55/canonical_stage1/v5_2/')
    prefix=prefix.rstrip('/')+'/'
    names=['historical_cfbd_game_manifest_v5.csv',
           'historical_cfbd_game_sides_v5.csv','stage1_canonical_ledger_v5_audit.json']
    return {n:prefix+n for n in names}


def _checked_canonical(manifest_bytes,sides_bytes,audit_bytes):
    audit=json.loads(audit_bytes)
    if not isinstance(audit,dict) or audit.get('bet_authority')!=0:
        raise ValueError('CANONICAL_LEDGER_AUTHORITY_OR_SCHEMA_INVALID')
    if audit.get('v4_reconciliation',{}).get('status')!='PASS_EXACT_V4_GAME_MEMBERSHIP_REPLAY':
        raise ValueError('CANONICAL_V4_MEMBERSHIP_NOT_REPLAYED')
    # Explicit neutral V5.2 correction, not obsolete V5 where 768 were held.
    if audit.get('stage1_neutral_v5_2',{}).get('status')!='PASS_SOURCE_NEUTRAL_TWO_SIDED_RECONCILIATION':
        raise ValueError('CANONICAL_NEUTRAL_V5_2_NOT_VERIFIED')
    for nm,content in [('historical_cfbd_game_manifest_v5.csv',manifest_bytes),
                       ('historical_cfbd_game_sides_v5.csv',sides_bytes)]:
        if hashlib.sha256(content).hexdigest()!=audit.get('outputs_sha256',{}).get(nm):
            raise ValueError('CANONICAL_SOURCE_HASH_MISMATCH:'+nm)
    games=_csv_bytes(manifest_bytes);sides=_csv_bytes(sides_bytes)
    if (len(games)!=audit.get('cfbd_games_linked_identity_only') or
            len(sides)!=2*len(games)):
        raise ValueError('CANONICAL_SOURCE_COUNT_MISMATCH')
    bygame={};bynative={};teams=defaultdict(dict);names=defaultdict(set)
    for g in games:
        yr=int(g['season']);gid=str(g['cfbd_game_id']).strip();sid=str(g['source_game_id']).strip()
        if not gid or not sid or (yr,gid) in bygame or (yr,sid) in bynative:
            raise ValueError('CANONICAL_GAME_COLLISION')
        bygame[(yr,gid)]=g;bynative[(yr,sid)]=gid
    for r in sides:
        yr=int(r['season']);gid=str(r['cfbd_game_id']).strip();team=str(r['cfbd_team_key']).strip()
        if (yr,gid) not in bygame or not team or team in teams[(yr,gid)]:
            raise ValueError('CANONICAL_SIDE_MISSING_OR_COLLISION')
        teams[(yr,gid)][team]=r
        for n in (team,r.get('source_team_label')):
            norm=_canonical_name(n)
            if norm:names[(yr,gid,norm)].add(team)
    if len(teams)!=len(bygame) or any(len(ss)!=2 for ss in teams.values()):
        raise ValueError('CANONICAL_NOT_TWO_SIDED')
    return audit,bygame,bynative,teams,names


def replay_existing_frozen_brains(manifest_bytes,sides_bytes,audit_bytes, *,
                                  stage2_events=None, raw_sources=None,
                                  expert_cache=None, source_receipts=None):
    """Attach ALL available frozen brains to previously matched game identities.

    Frozen outputs and original raw source artifacts share the exact CFBD IDs.
    No new Miner discovery, score-based identity guessing, or fake original
    signal timestamps. Returns research evidence + auditable missing families.
    """
    audit,games,native,team_map,name_map=_checked_canonical(
        manifest_bytes,sides_bytes,audit_bytes)
    rows=[];holds=Counter();source_counts=Counter();uniq={};examples=[]
    def attach(yr,gid,brain,system,market,direction,*,team='',source=''):
        brain=normal_family(brain);market=normal_market(market)
        if brain not in BRAINS or market not in MARKETS or not system:
            holds['INVALID_BRAIN_MARKET_OR_SYSTEM']+=1;return
        if (yr,gid) not in games:
            holds['NO_VERIFIED_CFBD_GAME']+=1;return
        cand=games[(yr,gid)]; two=team_map[(yr,gid)]
        if market=='totals':
            if direction not in ('OVER','UNDER'):
                holds['BAD_TOTALS_DIRECTION']+=1;return
            selected=direction
        else:
            selection=str(team or '').strip()
            if selection:
                options=name_map.get((yr,gid,_canonical_name(selection)),set())
                if len(options)!=1:
                    holds['NAMED_SIDE_NOT_UNIQUELY_MATCHED']+=1;return
                selected=next(iter(options))
            elif direction in ('HOME','AWAY','ROAD'):
                if cand.get('role_verification_status')!='HOME_AWAY_VERIFIED':
                    holds['NEUTRAL_OR_UNVERIFIED_VENUE_ROLE']+=1;return
                desired='1' if direction=='HOME' else '0'
                options=[k for k,r in two.items() if str(r.get('is_home_verified','')).strip()==desired]
                if len(options)!=1:
                    holds['UNIQUE_HOME_AWAY_ROLE_NOT_PRESENT']+=1;return
                selected=options[0]
            else:
                holds['MISSING_NAMED_DIRECTION']+=1;return
        key=(yr,gid,brain,str(system).strip(),market,selected)
        existing=uniq.get(key)
        if existing:
            source_counts['DUPLICATE_FIRES_SUPPRESSED']+=1
            return
        rec={'season':yr,'cfbd_game_id':gid,'source_game_id':cand['source_game_id'],
             'game_date':cand.get('game_date'),'brain':brain,'system_id':str(system).strip(),
             'market':market,'target':selected,'role_status':cand.get('role_verification_status'),
             'source':source,'original_signal_capture_certified':False,
             'original_quote_capture_certified':False,'bet_authority':0}
        uniq[key]=rec;rows.append(rec);source_counts[brain]+=1
    def verify_named_pair(y,gid,home,away):
        if not home and not away:return True
        if not home or not away or (y,gid) not in team_map:
            holds['MISSING_VERIFIED_TWO_TEAM_NAMES']+=1;return False
        hh=name_map.get((y,gid,_canonical_name(home)),set())
        aa=name_map.get((y,gid,_canonical_name(away)),set())
        if len(hh)!=1 or len(aa)!=1 or hh==aa:
            holds['SOURCE_NAMED_PAIR_NOT_MATCHED_TO_CANONICAL']+=1;return False
        return True
    def yyear(r,key='season'):
        try:return int(float(r.get(key)))
        except (ValueError,TypeError):return 0
    def resolve(y,r,key='game_id',native_only=False):
        x=str(r.get(key) or '').strip()
        if native_only:return native.get((y,x),'')
        if (y,x) in games:return x
        return native.get((y,x),'')
    # Already combined, source-attached Stage 2 observations (preferred).
    for r in stage2_events or []:
        y=yyear(r);gid=resolve(y,r,'cfbd_game_id') or resolve(y,r,'game_id')
        sid=str(r.get('source_game_id') or '').strip()
        if sid and native.get((y,sid))!=gid:
            holds['SOURCE_CFBD_ID_CONTRADICTION']+=1;continue
        if not verify_named_pair(y,gid,r.get('home_team') or r.get('home_team_key'),
                                 r.get('away_team') or r.get('away_team_key')):
            continue
        market=normal_market(r.get('market'));direction=str(r.get('direction') or '').upper()
        selected=str(r.get('target') or r.get('side_cfbd_team_key') or r.get('team') or '').strip()
        if market=='totals':
            direction=str(r.get('direction') or r.get('target') or '').upper()
            selected=''
        elif direction in ('HOME','AWAY','ROAD') and not selected:pass
        elif direction not in ('PLAY_ON_TEAM','PLAY_ON_NAMED_TEAM','TEAM','') and not selected:
            holds['STAGE2_DIRECTION_NOT_EXPLICIT']+=1;continue
        # Stage2 output cannot grant original pregame certification on its own.
        if yes(r.get('authority_eligible')) or yes(r.get('signal_capture_certified')) or yes(r.get('bet_authority')):
            raise ValueError('UNEXPECTED_RETROSPECTIVE_BET_AUTHORITY')
        attach(y,gid,r.get('brain'),r.get('system_id'),market,direction,
               team=selected,source='STAGE2_FROZEN_EVENT')
    # The already published 2013-2025 raw brain artifacts from Stage 2 V3:
    # never re-mine systems, only attach frozen exact FIRE observations.
    raw=raw_sources or {}
    if raw.get('CORE'):
        for r in raw['CORE']:
            y=yyear(r);gid=resolve(y,r);direction=str(r.get('pick_side') or '').upper()
            if direction not in ('HOME','AWAY'):
                holds['CORE_NO_EXPLICIT_SIDE']+=1;continue
            h,a=r.get('home_team'),r.get('away_team')
            if not verify_named_pair(y,gid,h,a):continue
            # Exact named-team role match even on neutral games; otherwise
            # the V5.2 neutral hold applies.
            team=h if direction=='HOME' else a
            attach(y,gid,'CORE',r.get('model_version') or r.get('model') or 'CORE_V55_RETROSPECTIVE',
                   'spreads',direction,team=team,source='CORE_FROZEN_REPLAY')
    if raw.get('SYSTEM_MINER'):
        rules={(yyear(r,'test_season'),normal_market(r.get('market')),str(r.get('rule_id') or ''))
               for r in raw.get('FROZEN_MINER',[])}
        for r in raw['SYSTEM_MINER']:
            y=yyear(r,'test_season');gid=resolve(y,r,'source_game_id',native_only=True)
            m=normal_market(r.get('market'));rid=str(r.get('rule_id') or '')
            if not yes(r.get('rule_selected_before_test_year')) or (y,m,rid) not in rules:
                holds['MINER_NOT_FROZEN_BEFORE_TEST']+=1;continue
            d=str(r.get('rule_decision_side') or '').upper()
            attach(y,gid,'SYSTEM_MINER',rid,m,d,source='FULL_HISTORY_MINER_FROZEN_YEAR')
    for fam in ('STAT','PREDICTION_TRACKER','MARKET_RICH'):
        for r in raw.get(fam,[]):
            if normal_family(r.get('brain') or fam)!=fam:continue
            if str(r.get('evaluation_status') or '').upper() not in ('FIRE','FIRED'):
                holds['NOT_EXPLICIT_SOURCE_FIRE']+=1;continue
            y=yyear(r);gid=resolve(y,r);d=str(r.get('direction') or '').upper()
            m=normal_market(r.get('market'))
            if not verify_named_pair(y,gid,r.get('home_team') or r.get('home_team_key'),
                                     r.get('away_team') or r.get('away_team_key')):
                continue
            # Never invent the market from an unqualified combined replay.
            if m not in MARKETS:
                holds['UNKNOWN_SOURCE_MARKET']+=1;continue
            name=r.get('team') or r.get('team_key') or r.get('pick_team') or ''
            if d in ('HOME','AWAY','ROAD') and r.get('home_team') and r.get('away_team'):
                name=r['home_team'] if d=='HOME' else r['away_team']
            attach(y,gid,fam,r.get('system_id'),m,d,team=name,source='FROZEN_COMBINED_REPLAY')
    # Pathi/Big Al original frozen occurrence cache. Exact season, date,
    # team AND opponent identity, never heuristic score/name joins.
    if isinstance(expert_cache,dict):
        cache=next((expert_cache[k] for k in ('systems','cache','directional_cache','historical_system_cache')
                    if isinstance(expert_cache.get(k),dict)),expert_cache)
        named=defaultdict(set)
        for (yr,gid),sides in team_map.items():
            for tm,r in sides.items():
                for a in (r.get('cfbd_team_key'),r.get('source_team_label')):
                    for b in (r.get('cfbd_opponent_key'),r.get('source_opponent_label')):
                        if a and b:
                            date=str(r.get('game_date') or games[(yr,gid)].get('game_date') or '')[:10]
                            named[(yr,date,_canonical_name(a),_canonical_name(b))].add((gid,tm))
        for system,item in cache.items():
            if not isinstance(item,dict) or str(item.get('role','')).lower()!='directional':continue
            fam={'Pathi':'PATHI','BigAl':'BIG_AL'}.get(str(item.get('family') or ''))
            if fam is None:continue
            for occ in item.get('occurrences') or []:
                if not isinstance(occ,dict):holds['EXPERT_RECORD_INVALID']+=1;continue
                y=yyear(occ);date=str(occ.get('date') or '')[:10]
                k=(y,date,_canonical_name(occ.get('team')),_canonical_name(occ.get('opponent')))
                candidate=named.get(k,set())
                if len(candidate)!=1:
                    holds['EXPERT_GAME_AND_SIDE_NOT_UNIQUE']+=1;continue
                gid,tm=next(iter(candidate))
                attach(y,gid,fam,system,'spreads','PLAY_ON_TEAM',team=tm,source='V143_EXPERT_CACHE')
    family_set=sorted(set(x['brain'] for x in rows))
    overlaps=defaultdict(lambda:defaultdict(set));dupesides=Counter()
    for x in rows:overlaps[(x['season'],x['cfbd_game_id'],x['market'])][x['target']].add(x['brain'])
    groups=[]
    # Verify outcomes against the CANONICAL Stage 1 game-side scores. These are
    # ATS/straight-up research grades at the historical reference spread; they
    # are not captured executable sportsbook returns.
    research_ats=defaultdict(Counter)
    for r in rows:
        k=(r['season'],r['cfbd_game_id']);m=r['market'];t=r['target']
        if m=='totals':research_ats[(r['brain'],m)]['TOTAL_REQUIRES_ORIGINAL_QUOTE']+=1;continue
        selected=team_map[k].get(t)
        if selected is None:raise ValueError('CANONICAL_SELECTED_TEAM_MISSING')
        home=num(selected.get('team_score'));away=num(selected.get('opponent_score'))
        spread=num(selected.get('retrospective_open_spread'))
        if home is None or away is None or (m=='spreads' and spread is None):
            research_ats[(r['brain'],m)]['UNGRADABLE']+=1;continue
        edge=home-away+(spread if m=='spreads' else 0)
        result='WIN' if edge>1e-7 else 'LOSS' if edge< -1e-7 else 'PUSH'
        research_ats[(r['brain'],m)][result]+=1
        research_ats[(r['brain'],m)][str(r['season'])+'_'+result]+=1
    for (yr,gid,m),tgts in sorted(overlaps.items()):
        for side,fams in sorted(tgts.items()):
            oppose=sorted({f for t,fs in tgts.items() if t!=side for f in fs})
            groups.append({'season':yr,'cfbd_game_id':gid,'market':m,'target':side,
                           'families':sorted(fams),'independent_family_count_descriptive':len(fams),
                           'opposing_families':oppose,'conflict':bool(oppose),'bet_authority':0})
    summary={'status':'CANONICAL_SEVEN_BRAIN_RETROSPECTIVE_REPLAY_UNCERTIFIED',
             'canonical_games':len(games),'canonical_sides':sum(len(x) for x in team_map.values()),
             'neutral_site_games':sum('NEUTRAL_SITE' in str(x.get('role_verification_status')) for x in games.values()),
             'source_identity_mismatches_held':audit.get('held_identity_after_v3'),
             'by_brain':{fam:source_counts[fam] for fam in sorted(BRAINS)},
             'brain_families_present':family_set,
             'brain_families_unavailable':sorted(BRAINS-set(family_set)),
             'distinct_games_with_brain':len({(r['season'],r['cfbd_game_id']) for r in rows}),
             'attached_observations':len(rows),'candidate_sides':len(groups),
             'cross_brain_groups':sum(len(g['families'])>=2 for g in groups),
             'conflicted_groups':sum(g['conflict'] for g in groups),
             'holds':dict(holds),'duplicate_suppressed':source_counts['DUPLICATE_FIRES_SUPPRESSED'],
             'canonical_output_sha256':audit.get('outputs_sha256'),
             'source_receipts':source_receipts or {},
             'historical_reference_ats_by_brain_market':{
                 family+'|'+market:dict(counts)
                 for (family,market),counts in sorted(research_ats.items())},
             'historical_reference_ats_roi_certified':False,
             'original_signal_capture_certified':False,
             'original_quote_capture_certified':False,'bet_authority':0,
             'replay_frozen_miner_rebuilt':False}
    return summary,rows,groups


def replay_prior_brain_sources(*, storage_client=None,bucket='sharp-models',canonical_prefix=None,
                               publish_events=True):
    """Use previously frozen artifact locations directly. Never request a NEW
    identity matcher or fabricated historical event manifest.

    Supports local Stage 1 files in Cloud Shell and GCS copies, pinned raw
    historical family feeds, and the original verified V143 expert cache.
    """
    manifest=_source_inventory(bucket,storage_client,canonical_prefix=canonical_prefix)
    receipts={};data={}
    for key,path in manifest.items():
        try:blob=_obj_bytes(path,storage_client)
        except Exception as exc:
            # Surface the missing canonical handoff, not a misleading all-brain zero.
            return {'status':'HOLD_PREVIOUS_V5_2_CANONICAL_LEDGER_NOT_ACCESSIBLE',
                    'missing_object':path,'error':type(exc).__name__+':'+str(exc),
                    'existing_source_data_reused':False,'bet_authority':0}
        data[key]=blob;receipts[key]={'uri':path,'sha256':hashlib.sha256(blob).hexdigest()}
    # Exact, independently published full-history runs from the prior audit.
    prefixes={
        'CORE':'research/ncaaf/v55/ats_primary/replay_runs/20261103T170209583052Z/retrospective_core_ats_game_edges.csv',
        'SYSTEM_MINER':'research/ncaaf/v2/miner_exact_path_shadow/runs/20261105T011910606679Z/retrospective_game_decision_ledger_v5.csv',
        'FROZEN_MINER':'research/ncaaf/v2/miner_exact_path_shadow/runs/20261105T011910606679Z/season_forward_frozen_rules_v5.csv',
        'STAT':'research/ncaaf/v55/combined_brains_shadow/runs/20261103T174003954208Z/retrospective_replay/rule_evaluations_retrospective.csv',
        'PREDICTION_TRACKER':'research/ncaaf/v55/combined_brains_shadow/runs/20261103T174003954208Z/retrospective_replay/pt_evidence_retrospective.csv'}
    raw={};holds={}
    for fam,key in prefixes.items():
        uri='gs://'+bucket+'/'+key
        try:
            content=_obj_bytes(uri,storage_client);raw[fam]=_csv_bytes(content)
            receipts[fam]={'uri':uri,'sha256':hashlib.sha256(content).hexdigest(),'rows':len(raw[fam])}
        except Exception as exc:holds[fam]=type(exc).__name__+':'+str(exc)
    expert_uri='gs://'+bucket+'/research/ncaaf/v55/expert_occurrence_snapshots_v1/runs/20261105T012018833244Z_9c0eb9ea4c6d/v143_directional_cache.json'
    try:
        content=_obj_bytes(expert_uri,storage_client);expert=json.loads(content)
        receipts['PATHI_BIG_AL']={'uri':expert_uri,'sha256':hashlib.sha256(content).hexdigest()}
    except Exception as exc:
        expert=None;holds['PATHI_BIG_AL']=type(exc).__name__+':'+str(exc)
    # Optional already attached Stage2 CSV can fill in family slots that raw
    # streams could not attach; no separate stage2 authority is executed.
    stage2_uri=os.getenv('NCAAF_V55_FROZEN_EVENTS_URI','').strip()
    stage2=[]
    if stage2_uri:
        try:
            content=_obj_bytes(stage2_uri,storage_client);stage2=_csv_bytes(content)
            receipts['STAGE2_FROZEN']={'uri':stage2_uri,'sha256':hashlib.sha256(content).hexdigest(),'rows':len(stage2)}
        except Exception as exc:holds['STAGE2_FROZEN']=type(exc).__name__+':'+str(exc)
    summary,rows,groups=replay_existing_frozen_brains(
        data['historical_cfbd_game_manifest_v5.csv'],data['historical_cfbd_game_sides_v5.csv'],
        data['stage1_canonical_ledger_v5_audit.json'],stage2_events=stage2,
        raw_sources=raw,expert_cache=expert,source_receipts=receipts)
    # Persist one canonical historical occurrence ledger for repeatable ROI
    # research and clickable system-history lookup. Never pretend these events
    # have original pregame timing/price certification.
    if publish_events and rows:
        body=('\n'.join(json.dumps(x,sort_keys=True,allow_nan=False) for x in
                         sorted(rows,key=lambda r:(r['season'],r['cfbd_game_id'],r['market'],r['brain'],r['system_id'],r['target'])))+'\n').encode()
        key='research/ncaaf/v55/two_engine_roi/frozen_brain_canonical_events_latest.jsonl'
        from google.cloud import storage
        client=storage_client or storage.Client()
        try:
            client.bucket(bucket).blob(key).upload_from_string(body,content_type='application/x-ndjson')
        except Exception as exc:
            summary['canonical_event_publication']={'status':'HOLD_PUBLICATION_FAILED',
                 'error':type(exc).__name__+':'+str(exc),'bet_authority':0}
        else:
            summary['canonical_event_publication']={
                'status':'PUBLISHED_RESEARCH_ONLY','uri':'gs://'+bucket+'/'+key,
                'rows':len(rows),'sha256':hashlib.sha256(body).hexdigest(),
                'original_capture_certified':False,'bet_authority':0}
    summary['source_feed_holds']=holds
    summary['source_policy']='PREVIOUS_VERIFIED_CANONICAL_IDENTITIES_AND_FROZEN_FIRES_ONLY'
    summary['missing_original_vintage_not_promoted_to_bets']=True
    # Only compact diagnostic summaries are put in the published registry; per-
    # event replay stays in research memory unless explicitly saved as evidence.
    return summary

def publish_existing_canonical_ledger(local_dir, *, storage_client=None,
                                      target_prefix=None,bucket='sharp-models'):
    """One-time, idempotent transfer of the ALREADY verified Stage1 V5.2 CSVs.

    Never rerun game matching or overwrite a published identity manifest. GCS
    only receives exact bytes whose SHA and neutral audit already passed.
    """
    root=Path(local_dir).expanduser()
    destination=_source_inventory(bucket,storage_client,canonical_prefix=target_prefix)
    raw={name:(root/name).read_bytes() for name in destination}
    audit,games,native,sides,names=_checked_canonical(
        raw['historical_cfbd_game_manifest_v5.csv'],
        raw['historical_cfbd_game_sides_v5.csv'],
        raw['stage1_canonical_ledger_v5_audit.json'])
    if any(not p.startswith('gs://') for p in destination.values()):
        raise ValueError('CANONICAL_PUBLISH_DESTINATION_MUST_BE_GCS')
    from google.cloud import storage
    client=storage_client or storage.Client()
    # Preflight every existing destination before creating any missing object.
    for name,uri in destination.items():
        b,k=uri[5:].split('/',1);obj=client.bucket(b).blob(k)
        if obj.exists() and obj.download_as_bytes()!=raw[name]:
            raise ValueError('REFUSE_TO_OVERWRITE_DIFFERENT_CANONICAL_ARTIFACT:'+name)
    for name,uri in destination.items():
        b,k=uri[5:].split('/',1);obj=client.bucket(b).blob(k)
        if not obj.exists():
            obj.upload_from_string(raw[name],content_type='application/json' if name.endswith('.json') else 'text/csv',
                                   if_generation_match=0)
    return {'status':'PASS_REUSED_EXISTING_VERIFIED_STAGE1_V5_2',
            'canonical_games':len(games),'canonical_sides':sum(len(x) for x in sides.values()),
            'source_path':str(root),'destination':destination,
            'sha256':{k:hashlib.sha256(v).hexdigest() for k,v in raw.items()},
            'rematched_or_rediscovered_games':0,'bet_authority':0}


def warehouse_historical_inputs(*,client=None,project='sharplogger',dataset='sharp_data',
                                table_override=None):
    """Discover a real original-pregame signal+quote ledger IN the existing BQ warehouse.

    This is a schema-inspected ingest path, NOT a synthetic construction of
    historical signal timestamps from modern feature tables. An eligible table
    must already contain all the original captured evidence fields. Report
    precise per-table gaps so research can expose what ingestion is still needed.
    Historical ROI is computed by research() after the source passes this audit.
    """
    import re
    from google.cloud import bigquery
    if not (re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*',project) and
            re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',dataset)):
        raise ValueError('INVALID_WAREHOUSE_PROJECT_OR_DATASET')
    bq=client or bigquery.Client(project=project)
    schema_sql='SELECT table_name,column_name FROM `'+project+'.'+dataset+'.INFORMATION_SCHEMA.COLUMNS` ORDER BY table_name,column_name'
    schema=defaultdict(set)
    for row in bq.query(schema_sql).result():
        fields=dict(row.items()) if hasattr(row,'items') else {'table_name':row.table_name,'column_name':row.column_name}
        schema[str(fields['table_name'])].add(str(fields['column_name']))
    essential=set(REQUIRED)|{'original_pregame_signal_verified','original_pregame_quote_verified','feature_vintage_verified'}
    # `line` is optional only for h2h rows, but spread/total events must carry it.
    expected=('ncaaf_v55_original_pregame_bet_events_v1',
              'ncaaf_v55_original_signal_quote_ledger_v1',
              'ncaaf_v55_bet_event_ledger_v1',
              'ncaaf_v55_canonical_bet_events_v1')
    candidates=sorted((t for t in schema if ('ncaaf' in t.lower() or t in ('moves_with_features_merged','bigal_game_context'))),
                      key=lambda t:(0 if t in expected else 1,t))
    gaps={t:sorted(essential-schema[t]) for t in candidates}
    diagnostics={'status':'HOLD_WAREHOUSE_ORIGINAL_PREGAME_EVENT_LEDGER_NOT_FOUND',
                 'warehouse':project+'.'+dataset,
                 'sources_inspected':len(candidates),
                 'eligible_source_tables':[t for t in candidates if not gaps[t]],
                 'missing_columns_by_table':gaps,
                 'no_retroactive_signal_reconstruction':True,
                 'pricing_from_actual_historical_quotes_only':True}
    override=(table_override or '').strip()
    if override:
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',override):
            raise ValueError('INVALID_ORIGINAL_EVENTS_TABLE_OVERRIDE')
        if override not in schema or essential-schema[override]:
            diagnostics['status']='HOLD_EXPLICIT_EVENT_SOURCE_COLUMNS_INCOMPLETE'
            diagnostics['explicit_source_table']=override
            return [],None,diagnostics
        chosen=override
    else:
        eligible=[t for t in candidates if not gaps[t]]
        if not eligible:return [],None,diagnostics
        if len(eligible)!=1:
            diagnostics['status']='HOLD_MULTIPLE_ORIGINAL_EVENT_SOURCE_TABLES__SELECT_EXPLICIT'
            return [],None,diagnostics
        chosen=eligible[0]
    sql=('SELECT * FROM `'+project+'.'+dataset+'.'+chosen+'` '
         'WHERE SAFE_CAST(season AS INT64) BETWEEN 2013 AND 2026 '
         'ORDER BY game_id,market,side,book,quote_observed_at_utc,source_family,system_id')
    rows=[dict(row.items()) for row in bq.query(sql).result(page_size=10000)]
    diagnostics.update({'status':'WAREHOUSE_ORIGINAL_SOURCE_SCHEMA_FOUND__ROW_VINTAGE_AUDIT_PENDING',
                        'selected_event_table':project+'.'+dataset+'.'+chosen,
                        'source_rows':len(rows)})
    return rows,digest(rows),diagnostics

def load_rows(path):
    if path.startswith('bq://'):
        # Read a canonical BigQuery observations table directly, if one exists.
        # Retrospectively joined feature tables without original signal/quote
        # clocks fail the same strict source-vintage gate as CSV inputs.
        import re
        table=path[5:]
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*',table):
            raise ValueError('INVALID_BIGQUERY_TABLE_ID')
        from google.cloud import bigquery
        sql='SELECT * FROM `'+table+'` ORDER BY game_id,market,side,book,quote_observed_at_utc,source_family,system_id'
        rows=[dict(row.items()) for row in bigquery.Client().query(sql).result(page_size=10000)]
        return rows,digest(rows)
    if path.startswith('gs://'):
        from google.cloud import storage
        bucket,blob=path[5:].split('/',1)
        content=storage.Client().bucket(bucket).blob(blob).download_as_bytes()
    else:content=Path(path).read_bytes()
    h=hashlib.sha256(content).hexdigest()
    if path.lower().endswith('.csv'):
        rows=list(csv.DictReader(io.StringIO(content.decode('utf-8-sig'))))
    else:
        text=content.decode('utf-8-sig').strip()
        rows=json.loads(text) if text.startswith('[') else [json.loads(s) for s in text.splitlines() if s.strip()]
    if not isinstance(rows,list):raise ValueError('RECORD_INPUT_NOT_A_LIST')
    return rows,h

def read_json(path):
    if not path:return None
    if path.startswith('gs://'):
        from google.cloud import storage
        bucket,blob=path[5:].split('/',1)
        return json.loads(storage.Client().bucket(bucket).blob(blob).download_as_text())
    return json.loads(Path(path).read_text())

def put_json(path,doc):
    content=json.dumps(doc,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if path.startswith('gs://'):
        from google.cloud import storage
        bucket,blob=path[5:].split('/',1)
        storage.Client().bucket(bucket).blob(blob).upload_from_string(content,content_type='application/json')
    else:Path(path).write_text(content)

def main(argv=None):
    p=argparse.ArgumentParser(description='Single NCAAF 2013-2025 seven-brain ROI research engine')
    p.add_argument('--publish-existing-stage1',help='One-time upload of ALREADY verified V5.2 canonical files from Cloud Shell; do not rerun matching')
    p.add_argument('--observations',help='Original pregame captured historical betting events CSV or JSONL (local/gs://)')
    p.add_argument('--capture-audit',help='Independent original capture integrity attestation (local/gs://); missing audit yields HOLD')
    p.add_argument('--out',help='Local or gs:// research registry output')
    p.add_argument('--holdout-start',type=int,default=2024)
    args=p.parse_args(argv)
    try:
        if args.publish_existing_stage1:
            print(json.dumps(publish_existing_canonical_ledger(args.publish_existing_stage1),sort_keys=True))
            return 0
        if not args.observations or not args.out:
            p.error('--observations and --out are required unless --publish-existing-stage1 is specified')
        rows,sha=load_rows(args.observations)
        doc=research(rows,capture_audit=read_json(args.capture_audit),source_sha256=sha,holdout_start=args.holdout_start)
        put_json(args.out,doc)
        print(json.dumps({'status':doc['status'],'input_rows':doc['raw_rows'],'tested_hypotheses':doc.get('tested_hypotheses',0),
                          'certified_edges':len(doc['certificates']),'invalid_rows':doc.get('invalid_rows')},sort_keys=True))
        return 0 if doc['status']=='PASS_CERTIFIED_ROI_REGISTRY' else 2
    except Exception as exc:
        print('NCAAF_RESEARCH_ERROR:'+type(exc).__name__+':'+str(exc),file=sys.stderr)
        return 3
if __name__=='__main__':raise SystemExit(main())
