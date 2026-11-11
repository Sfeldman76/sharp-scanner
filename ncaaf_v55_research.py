#!/usr/bin/env python3
"""NCAAF V5.5 canonical research: evidence/ROI and expanding-origin ML training.

Input is the ORIGINAL, PRE-GAME captured seven-brain bet-event ledger. This file
never infers original signals/quotes from retrospective context. It discovers
hypotheses ONLY from discovery-period events, verifies rule freeze provenance,
tests 2024-25 or 2026+ OOS, calculates realized per-unit betting ROI, and
publishes a hash-checked registry only when the capture audit is independently
attested. ML training is research-only, produces separately versioned artifacts,
and cannot bypass the original ROI certification or live betting safeguards.
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
        # A source's Game_Key need not equal its canonical CFBD identifier.
        # Only accept an explicit canonical ID or an exact prior verified
        # native source ID; do not rematch by team/score/date heuristics.
        aliases=defaultdict(set)
        for canonical_id,canonical_game in games.items():
            season=canonical_game.get('season')
            source_id=str(canonical_game.get('source_game_id') or '').strip()
            if source_id and season is not None:
                aliases[(str(season),source_id)].add(canonical_id)
        # Reuse Stage 1 V5.2 identities. Do not create a second fuzzy matcher.
        identity_index={}
        try:
            existing=_source_inventory('sharp-models',None)
            original=[_obj_bytes(existing[n]) for n in (
                'historical_cfbd_game_manifest_v5.csv',
                'historical_cfbd_game_sides_v5.csv',
                'stage1_canonical_ledger_v5_audit.json')]
            identity_index=_verified_snapshot_identity_index(*original)
            report['stage1_identity_reuse']='VERIFIED_HASHED_V5_2'
        except Exception as exc:
            report['stage1_identity_reuse']='HOLD_STAGE1_IDENTITY_UNAVAILABLE:'+type(exc).__name__
        # Canonical-side lookup must corroborate warehouse historical results.
        # Match only unique season, same teams and roles: no score matching.
        game_pair_index=defaultdict(set)
        for hist_key, hist_game in games.items():
            game_pair_index[(hist_game['season'],hist_game['home_team'],hist_game['away_team'])].add(hist_key)
        report['snapshot_key_mapping_policy']='EXACT_NATIVE_OR_VERIFIED_V5_2_DATE_TEAM_ROLE_CROSSWALK'
        report['snapshot_unmatched_key_examples']=[]
        for q in _source_research_rows(sql,bq,limit=max_quote_rows+1):
            count+=1
            if count>max_quote_rows:break
            key=str(q.get('Game_Key') or '').strip()
            g=games.get(key)
            if not g:
                kick=clock(q.get('Commence_Hour'))
                possible=aliases.get((str(kick.year),key),set()) if kick else set()
                if len(possible)==1:
                    g=games[next(iter(possible))]
                    rejected['VERIFIED_NATIVE_GAME_KEY_RECOVERED']+=1
                else:
                    stage1_id,reason=_resolve_exact_snapshot_identity(q,identity_index)
                    # A valid canonical key by itself is not a scored warehouse
                    # game. Confirm exactly one historical result with same year,
                    # teams and roles before allowing grading.
                    hk=game_pair_index.get((kick.year,_entity_key(q.get('Home_Team_Norm')),
                                               _entity_key(q.get('Away_Team_Norm'))),set()) if kick else set()
                    if stage1_id and len(hk)==1:
                        g=games[next(iter(hk))]
                        rejected['VERIFIED_V5_2_ROLE_DATE_CROSSWALK_RECOVERED']+=1
                    else:
                        if stage1_id and len(hk)>1:reason='AMBIGUOUS_HISTORICAL_RESULT_FOR_CANONICAL_GAME'
                        if stage1_id and not hk:reason='NO_MATCHED_HISTORICAL_RESULT_FOR_CANONICAL_GAME'
                        rejected[reason if identity_index else 'NO_EXACT_CANONICAL_GAME_KEY']+=1
                        if len(report['snapshot_unmatched_key_examples'])<12:
                            report['snapshot_unmatched_key_examples'].append({
                                'key':key,'home':str(q.get('Home_Team_Norm') or ''),
                                'away':str(q.get('Away_Team_Norm') or ''),
                                'kickoff':str(q.get('Commence_Hour') or '')})
                        continue
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
            entry={'game_id':str(g.get('cfbd_game_id') or key),'market':market,'side':side,'line':line if market!='h2h' else None,
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


def _discover_verified_stage2_uri(bucket, storage_client=None):
    """Locate the one already-built Stage 2 event ledger; never regenerate it.

    An explicit URI is authoritative. Automatic lookup accepts only the exact
    published artifact filename, never a similarly named raw research file.
    """
    explicit=(os.getenv('NCAAF_V55_STAGE2_V3_FROZEN_EVENTS_URI','').strip()
              or os.getenv('NCAAF_V55_FROZEN_EVENTS_URI','').strip())
    if explicit:
        return explicit, 'EXPLICIT_PREVIOUS_STAGE2_URI'
    expected='research/ncaaf/v55/previously_verified_stage2_v3/stage2_frozen_brain_events.csv'
    if storage_client is None:
        from google.cloud import storage
        storage_client=storage.Client()
    if storage_client.bucket(bucket).blob(expected).exists():
        return 'gs://'+bucket+'/'+expected, 'AUTO_DISCOVERED_VERIFIED_STAGE2_URI'
    return '', 'HOLD_STAGE2_EVENT_LEDGER_NOT_PUBLISHED'


def _verified_snapshot_identity_index(manifest_bytes, sides_bytes, audit_bytes):
    """Exact date and two named team roles from already hash-verified V5.2.

    No score fitting, fuzzy aliases, or inference from sportsbook lines.
    Returns a unique canonical game only; ambiguous identities remain held.
    """
    _, bygame, _, sides, _ = _checked_canonical(manifest_bytes,sides_bytes,audit_bytes)
    candidates=defaultdict(set)
    for (season,gid), game in bygame.items():
        date=str(game.get('game_date') or '')[:10]
        if not date or len(date)!=10:continue
        members=sides.get((season,gid),{})
        home=[x for x in members.values() if str(x.get('is_home_verified') or '').lower() in ('true','1','yes')]
        away=[x for x in members.values() if str(x.get('is_home_verified') or '').lower() in ('false','0','no')]
        if len(home)!=1 or len(away)!=1:continue
        for hn in (home[0].get('cfbd_team_key'),home[0].get('source_team_label')):
            for an in (away[0].get('cfbd_team_key'),away[0].get('source_team_label')):
                if _entity_key(hn) and _entity_key(an):
                    candidates[(season,date,_entity_key(hn),_entity_key(an))].add(gid)
    return candidates


def _resolve_exact_snapshot_identity(row, index):
    kickoff=clock(row.get('Commence_Hour'))
    if not kickoff:return None, 'NO_KICKOFF_FOR_CANONICAL_IDENTITY'
    h=_entity_key(row.get('Home_Team_Norm'))
    a=_entity_key(row.get('Away_Team_Norm'))
    # No calendar-day shifting: only exact previously verified game date.
    keys=(kickoff.year,kickoff.date().isoformat(),h,a)
    options=index.get(keys,set())
    if len(options)==1:return next(iter(options)), 'VERIFIED_STAGE1_DATE_AND_TEAM_ROLES'
    return None, 'AMBIGUOUS_STAGE1_DATE_AND_ROLES' if len(options)>1 else 'NO_VERIFIED_STAGE1_DATE_AND_ROLES'


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


def verified_frozen_market(family, record, *, research_assumption=''):
    """Recover only a market explicitly evidenced by a frozen source row.

    A forecast and a HOME/AWAY direction do not establish spreads vs H2H.
    """
    existing=normal_market(record.get('market'))
    if existing in MARKETS:
        return existing
    # STAT's recorded spread and ATS result are explicit spread evidence.
    if family=='STAT' and num(record.get('open_home_spread')) is not None and \
            str(record.get('descriptive_ats_result') or '').strip().upper() in \
            ('WIN','LOSS','PUSH','W','L'):
        return 'spreads'
    # Stage 2 V3 supported an EXPLICIT research-only combined-market assumption.
    # Never promote this assumption to source truth or original betting authority.
    if family in ('STAT','PREDICTION_TRACKER') and research_assumption in MARKETS:
        return research_assumption
    # External PT evidence can only declare its own market, not inherit STAT.
    explicit=str(record.get('evidence_type') or '').strip().upper()
    declared={'ATS':'spreads','SPREAD':'spreads','SPREADS':'spreads',
              'H2H':'h2h','MONEYLINE':'h2h','ML':'h2h',
              'TOTAL':'totals','TOTALS':'totals','OVER_UNDER':'totals'}
    return declared.get(explicit)


def replay_existing_frozen_brains(manifest_bytes,sides_bytes,audit_bytes, *,
                                  stage2_events=None, raw_sources=None,
                                  expert_cache=None, source_receipts=None,
                                  combined_research_assumption=''):
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
            m=verified_frozen_market(fam,r,research_assumption=combined_research_assumption)
            if not verify_named_pair(y,gid,r.get('home_team') or r.get('home_team_key'),
                                     r.get('away_team') or r.get('away_team_key')):
                continue
            # Never invent the market from an unqualified combined replay.
            if m not in MARKETS:
                holds['UNVERIFIABLE_MARKET_'+fam]+=1;continue
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



def canonical_replay_edge_diagnostics(events, games, team_map, groups, *, top_n=100):
    """Analyze ALL attached canonical firings without pretending original prices.

    Discovery (2013-23) and retrospective confirmation (2024-25) are reported
    separately. Neither is called OOS/certified absent original pregame freeze,
    quote capture and an untouched post-selection test window. No win-prob model.
    """
    if top_n<1 or top_n>1000:raise ValueError('INVALID_REPORT_LIMIT')
    singles=defaultdict(Counter); families=defaultdict(Counter)
    combos=defaultdict(Counter); line_sensitivity=defaultdict(Counter)
    invalid=Counter(); graded=0; observed=Counter(); excluded_conflict_groups=0
    for r in events:
        season=r['season']; market=r['market']; team=r['target']
        observed[r['brain']]+=1
        if market=='totals':
            invalid['TOTAL_NO_VERIFIED_REFERENCE_TOTAL']+=1
            continue
        rec=team_map.get((season,r['cfbd_game_id']),{}).get(team)
        if rec is None:invalid['UNKNOWN_CANONICAL_SIDE']+=1;continue
        team_score=num(rec.get('team_score')); opponent=num(rec.get('opponent_score'))
        line=num(rec.get('retrospective_open_spread')) if market=='spreads' else None
        if team_score is None or opponent is None or (market=='spreads' and line is None):
            invalid['NO_GRADABLE_REFERENCE_LINE_OR_FINAL']+=1;continue
        # Re-grade named team as a virtual HOME orientation; the actual venue
        # role is deliberately not inferred (neutral-site safe).
        result=grade(market,'HOME',line,team_score,opponent)
        if result is None:invalid['NO_GRADABLE_OUTCOME']+=1;continue
        partition='DISCOVERY_2013_2023' if season<=2023 else 'RETROSPECTIVE_2024_2025'
        singles[(r['brain'],r['system_id'],market,partition)][result]+=1
        singles[(r['brain'],r['system_id'],market,partition)]['games']+=1
        families[(r['brain'],market,partition)][result]+=1
        families[(r['brain'],market,partition)]['occurrences']+=1
        graded+=1
        if market=='spreads':
            for shift in (0.5,1,3,6):
                alt=grade(market,'HOME',line-shift,team_score,opponent)
                line_sensitivity[(r['brain'],market,str(shift))][result+'->'+alt]+=1
    for g in groups:
        yr=g['season'];m=g['market'];ts=g['target']
        if m=='totals':continue
        if g['conflict']:
            excluded_conflict_groups+=1
            continue
        sides=team_map.get((yr,g['cfbd_game_id']),{})
        rec=sides.get(ts)
        if rec is None:continue
        tsco=num(rec.get('team_score'));osco=num(rec.get('opponent_score'))
        ln=num(rec.get('retrospective_open_spread')) if m=='spreads' else None
        if tsco is None or osco is None or (m=='spreads' and ln is None):continue
        outcome=grade(m,'HOME',ln,tsco,osco)
        if outcome is None:continue
        partition='DISCOVERY_2013_2023' if yr<=2023 else 'RETROSPECTIVE_2024_2025'
        fs=g['families']
        for a,b in itertools.combinations(fs,2):
            k=(a,b,m,partition)
            combos[k][outcome]+=1
            combos[k]['unique_game_sides']+=1
            if g['conflict']:combos[k]['has_opposing_source_families']+=1
    def summarize(k,c):
        n=c['WIN']+c['LOSS']+c['PUSH']
        return {'key':list(k),'n':n,'wins':c['WIN'],'losses':c['LOSS'],
                'pushes':c['PUSH'],'ats_cover_rate_excluding_pushes':
                    round(c['WIN']/(c['WIN']+c['LOSS']),6) if c['WIN']+c['LOSS'] else None}
    systems=[summarize(k,c) for k,c in singles.items()]
    systems.sort(key=lambda r:(-r['n'],r['key']))
    pairs=[dict(summarize(k,c),opposing_family_present=c['has_opposing_source_families']>0)
           for k,c in combos.items()]
    pairs.sort(key=lambda r:(-r['n'],r['key']))
    return {'status':'DESCRIPTIVE_CANONICAL_ATS__NOT_CERTIFIED_BETTING_ROI',
            'canonical_observations_input':len(events),
            'canonical_observations_by_family':dict(observed),
            'reference_line_graded_occurrences':graded,
            'system_market_partition_count':len(systems),
            'combination_market_partition_count':len(pairs),
            'excluded_opposing_signal_groups_from_combination_research':excluded_conflict_groups,
            'individual_systems_top':systems[:top_n],
            'family_market_partitions':[summarize(k,v) for k,v in sorted(families.items())],
            'same_side_family_pairs_top':pairs[:top_n],
            'reference_line_adverse_movement':[
                {'brain':k[0],'market':k[1],'points':k[2],'grade_transitions':dict(c)}
                for k,c in sorted(line_sensitivity.items())],
            'grading_holds':dict(invalid),
            'historical_book_odds_available_in_this_ledger':False,
            'original_pregame_source_capture_verified':False,
            'independent_out_of_sample':False,'certified_edges':0,'bet_authority':0}


def v55_historical_family_combinations(groups, team_map, *, max_families=7):
    """Grade every observed independent-family subset, 2013-25 only.

    One game/market/target counts once for a combination. Opposing family
    signals veto that specific market-side; no retrospective quote is called
    an executable bet, and unverified same-source families are not duplicated.
    Discovery and holdout grades are strictly separated and never promote a
    post-hoc combination to a pre-registered OOS ROI certificate.
    """
    counters=defaultdict(lambda:defaultdict(dict));opposed=Counter();invalid=Counter()
    for g in groups:
        try:
            y=int(g['season']);market=normal_market(g['market'])
            if not 2013<=y<=2025 or market!='spreads':continue
            fams=sorted(set(normal_family(f) for f in g.get('families',[])) & BRAINS)
            if not fams:continue
            if g.get('conflict') or g.get('opposing_families'):
                opposed[market]+=1;continue
            gid=str(g['cfbd_game_id']);target=str(g['target'])
            rec=team_map.get((y,gid),{}).get(target)
            if not rec:invalid['MISSING_SIDE']+=1;continue
            h,a,line=(num(rec.get(k)) for k in
                      ('team_score','opponent_score','retrospective_open_spread'))
            if None in (h,a,line):invalid['MISSING_REFERENCE_RESULT']+=1;continue
            result=grade('spreads','HOME',line,h,a)
            if result is None:invalid['UNGRADABLE']+=1;continue
            season_bin='DISCOVERY_2013_2023' if y<=2023 else 'RETROSPECTIVE_2024_2025'
            for size in range(1,min(max_families,len(fams))+1):
                for subset in itertools.combinations(fams,size):
                    counters[subset][season_bin][(y,gid,target)]=result
        except (KeyError,TypeError,ValueError):invalid['INVALID_GROUP']+=1
    summary=[]
    for subset,splits in sorted(counters.items()):
        bysplit={}
        for name,observations in sorted(splits.items()):
            values=list(observations.values());w=values.count('WIN');l=values.count('LOSS')
            bysplit[name]={'unique_game_sides':len(values),'wins':w,'losses':l,
                'pushes':values.count('PUSH'),
                'cover_rate_excluding_pushes':round(w/(w+l),6) if w+l else None}
        summary.append({'families':list(subset),'family_count':len(subset),
                        'partitions':bysplit,'bet_authority':0})
    return {'status':'ALL_1_TO_7_FAMILY_COMBINATIONS_REFERENCE_ATS_RESEARCH_ONLY',
        'independent_family_subsets_observed':len(summary),
        'by_combination':summary,'opposed_game_market_sides_vetoed':dict(opposed),
        'invalid_rows':dict(invalid),'uses_2026_outcomes':False,
        'independent_original_bet_price_certified':False,'bet_authority':0}

V55_QUALIFICATION_BLOB = 'research/ncaaf/v55/qualified_independent_brains_v55_current.json'
V55_QUALIFICATION_POLICY = 'V55-FROZEN-INDEPENDENT-BRAINS-2013-2025-R1'

# Shared V5.5 small-sample OOS screening contract. This is a research
# eligibility test, NOT an independently multiplicity-adjusted ROI certificate.
V55_OOS_MIN_DECISIONS = 12
V55_OOS_MIN_DECISIONS_PER_SEASON = 4
V55_OOS_POSTERIOR_MIN_PROB = 0.90
V55_OOS_REFERENCE_BASELINE = 0.50
V55_OOS_SCREEN_POLICY = 'V55-2024-2025-BETA11-POSTERIOR-90PCT-VS-50PCT-R1'


def v55_oos_statistical_screen(wins, losses, year_2024, year_2025):
    """Return auditable Beta(1,1) posterior one-sided P(p > 0.5).

    A small data sufficiency floor prevents five-game systems from qualifying.
    Pushes do not count as Bernoulli trials. Confirming outcomes must arise
    from 2024 and 2025; neither retrospective selection nor data provenance is
    certified by this screen. Independence/multiple-testing still gate authority.
    For integer beta shapes, the probability is an exact binomial CDF.
    """
    w, l = int(wins), int(losses)
    a, b = int(year_2024), int(year_2025)
    n = w + l
    if min(w,l,a,b) < 0 or a+b != n:
        return {'pass': False, 'reason': 'INCONSISTENT_CONFIRMATION_COUNTS',
                'decisive_n': n, 'posterior_prob': None}
    if n < V55_OOS_MIN_DECISIONS or min(a,b) < V55_OOS_MIN_DECISIONS_PER_SEASON:
        return {'pass': False, 'reason': 'INSUFFICIENT_STATISTICAL_EVIDENCE',
                'decisive_n': n, 'posterior_prob': None}
    # For P(Beta(w+1,l+1) > .5), the exact tail is
    # P(Binomial(n+1,.5) <= w); no scipy dependency required.
    probability = sum(math.comb(n+1,k) for k in range(w+1)) / (1 << (n+1))
    passed = probability >= V55_OOS_POSTERIOR_MIN_PROB
    return {'pass': passed,
            'reason': 'PASS_STATISTICAL_OOS_SCREEN' if passed else 'POSTERIOR_EDGE_EVIDENCE_BELOW_THRESHOLD',
            'decisive_n': n, 'posterior_prob': round(probability,8),
            'baseline_rate': V55_OOS_REFERENCE_BASELINE,
            'posterior_threshold': V55_OOS_POSTERIOR_MIN_PROB,
            'policy': V55_OOS_SCREEN_POLICY}





def _v55_qualification_evidence(family, system_id, source_receipts, attestations):
    """Attestations are independently captured proof, not retrospective labels.

    A receipt is verified only when the evidence names and hashes a pinned source
    and includes independently reviewable references. Absence is a HOLD.
    """
    key=family+'|'+system_id
    record=(attestations or {}).get(key) if isinstance(attestations,dict) else None
    if not isinstance(record,dict):return {}, {'MISSING_INDEPENDENT_SYSTEM_ATTESTATION'}
    holds=set(); receipts=source_receipts or {}
    source_name=str(record.get('pinned_source_name') or '')
    source=receipts.get(source_name)
    if not (isinstance(source,dict) and len(str(source.get('sha256') or ''))==64
            and source.get('sha256')==record.get('pinned_source_sha256')):
        holds.add('PINNED_SOURCE_HASH_NOT_VERIFIED')
    freeze=str(record.get('definition_frozen_at_utc') or '')
    frozen=clock(freeze)
    try: selection_end=int(record.get('selection_end_season') or 9999)
    except (TypeError,ValueError):selection_end=9999
    if (not frozen or frozen>=datetime(2024,1,1,tzinfo=timezone.utc)
            or selection_end>2023
            or not record.get('independent_preholdout_freeze_ref')):
        holds.add('INDEPENDENT_PRE_HOLDOUT_RULE_FREEZE_NOT_ATTESTED')
    fv=record.get('pregame_feature_audit') or {}
    if not (isinstance(fv,dict) and fv.get('status')=='PASS'
            and fv.get('source_ref') and fv.get('leakage_audit_ref')
            and fv.get('all_features_available_before_game') is True):
        holds.add('PREGAME_FEATURE_VINTAGE_NOT_ATTESTED')
    live=record.get('live_recipe') or {}
    if not (isinstance(live,dict) and live.get('role')=='directional'
            and live.get('direction') in ('PLAY_ON','FADE')
            and live.get('market')=='spreads'
            and live.get('feature_parity_ref')):
        holds.add('LIVE_RULE_FIELD_RECIPE_NOT_ATTESTED')
    else:
        if family=='SYSTEM_MINER':
            conditions=live.get('conditions')
            if not isinstance(conditions,list) or not conditions or any(
                not isinstance(c,dict) or c.get('op') not in ('==','!=','<=','<','>=','>','in')
                or not isinstance(c.get('field'),str)
                or not __import__('re').fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,127}',c['field'])
                or 'value' not in c for c in conditions):
                holds.add('NO_EXACT_EXECUTABLE_V5_MINER_CONDITIONS')
        else:
            f=live.get('signal_field')
            if (not isinstance(f,str) or not __import__('re').fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,127}',f)
                or live.get('signal_type') not in ('FLAG','SIDE')):
                holds.add('LIVE_RULE_FIELD_RECIPE_NOT_ATTESTED')
    multiple=record.get('multiple_testing_audit') or {}
    independence=record.get('source_independence_audit') or {}
    if not isinstance(multiple,dict):multiple={}
    if not isinstance(independence,dict):independence={}
    if not (multiple.get('status')=='PASS' and multiple.get('audit_ref')
            and multiple.get('selection_without_holdout_outcomes') is True
            and independence.get('status')=='PASS' and independence.get('audit_ref')
            and record.get('locked_holdout_ref')):
        holds.add('MULTIPLE_TESTING_AND_SOURCE_INDEPENDENCE_NOT_ATTESTED')
    if record.get('uses_2026_outcomes') is not False:
        holds.add('UNVERIFIED_2026_OUTCOME_EXCLUSION')
    return record,holds


def v55_frozen_qualification_report(events, team_map, source_receipts, *,
                                    generated_at=None, attestations=None):
    """Evidence-driven full-history rule qualification; no implicit approvals.

    Independently attested frozen rules are evaluated against sealed 2024-25
    reference-spread outcomes. Missing attestations are held per system, not
    universally asserted without examining the actual source receipts.
    This is rule eligibility only, never a certificate of sportsbook ROI.
    """
    created=generated_at or datetime.now(timezone.utc).isoformat()
    observations=defaultdict(lambda:defaultdict(dict))
    seasons_seen=set();invalid=Counter()
    for e in events:
        try:
            y=int(e['season']);fam=normal_family(e['brain']);sid=str(e['system_id']).strip()
            m=normal_market(e['market']);gid=str(e['cfbd_game_id']);target=str(e['target'])
            if not 2013<=y<=2025:
                invalid['OUTSIDE_FROZEN_HISTORICAL_SEASONS']+=1;continue
            seasons_seen.add(y)
            if not sid or fam not in BRAINS or m!='spreads':
                invalid['NON_SPREAD_OR_INVALID_SYSTEM']+=1;continue
            rec=team_map.get((y,gid),{}).get(target)
            if not rec:
                invalid['NO_CANONICAL_GAME_SIDE']+=1;continue
            ts,op,ln=(num(rec.get(k)) for k in
                        ('team_score','opponent_score','retrospective_open_spread'))
            if None in (ts,op,ln):
                invalid['REFERENCE_RESULT_OR_LINE_MISSING']+=1;continue
            result=grade('spreads','HOME',ln,ts,op)
            if result not in ('WIN','LOSS','PUSH'):
                invalid['UNGRADEABLE_REFERENCE']+=1;continue
            observations[(fam,sid)][y][(gid,target)]=result
        except (ValueError,TypeError,KeyError):invalid['INVALID_FROZEN_EVENT']+=1
    diagnostics=[];qualified=[];rejections=Counter()
    for (fam,sid),by_season in sorted(observations.items()):
        byyear={}
        for year,games in sorted(by_season.items()):
            vals=list(games.values());w=vals.count('WIN');l=vals.count('LOSS')
            byyear[str(year)]={'n':len(vals),'wins':w,'losses':l,'pushes':vals.count('PUSH'),
                 'rate':round(w/(w+l),6) if w+l else None}
        test=[v for yy,rr in by_season.items() if yy in (2024,2025) for v in rr.values()]
        w=test.count('WIN');l=test.count('LOSS');n=len(test)
        holds=[]
        if fam not in ('STAT','PATHI','BIG_AL','PREDICTION_TRACKER','SYSTEM_MINER'):
            holds.append('NO_EXECUTABLE_DIRECTIONAL_RULE_CONTRACT')
        year24=byyear.get('2024',{})
        year25=byyear.get('2025',{})
        screen=v55_oos_statistical_screen(w,l,
            int(year24.get('wins',0))+int(year24.get('losses',0)),
            int(year25.get('wins',0))+int(year25.get('losses',0)))
        if screen['reason']=='INSUFFICIENT_STATISTICAL_EVIDENCE':
            holds.append('RETROSPECTIVE_CONFIRMATION_INSUFFICIENT_STATISTICAL_EVIDENCE')
        elif not screen['pass']:
            holds.append('RETROSPECTIVE_CONFIRMATION_POSTERIOR_EDGE_BELOW_THRESHOLD')
        proof,proof_holds=_v55_qualification_evidence(fam,sid,source_receipts,attestations)
        # Report precise attestation gaps even when the proof registry is absent.
        if 'MISSING_INDEPENDENT_SYSTEM_ATTESTATION' in proof_holds:
            proof_holds.discard('MISSING_INDEPENDENT_SYSTEM_ATTESTATION')
            proof_holds.update(('INDEPENDENT_PRE_HOLDOUT_RULE_FREEZE_NOT_ATTESTED',
                 'PREGAME_FEATURE_VINTAGE_NOT_ATTESTED','LIVE_RULE_FIELD_RECIPE_NOT_ATTESTED',
                 'MULTIPLE_TESTING_AND_SOURCE_INDEPENDENCE_NOT_ATTESTED'))
        holds.extend(sorted(proof_holds))
        for h in holds:rejections[h]+=1
        item={'family':fam,'system_id':sid,'market':'spreads',
            'discovery_reference_games':sum(len(v) for y,v in by_season.items() if y<=2023),
            'confirmation_n':n,'confirmation_decisive_n':w+l,'confirmation_wins':w,'confirmation_losses':l,
            'confirmation_rate':round(w/(w+l),6) if w+l else None,
            'by_season':byyear,'oos_statistical_screen':screen,'holds':holds,'descriptive_reference_only':True,
            'oos_pass':not bool(holds),'evidence_key':fam+'|'+sid}
        diagnostics.append(item)
        if not holds:
            live=proof['live_recipe']
            rule=dict(item,role='directional',direction=live['direction'],
                evidence_family_key=str(proof.get('evidence_family_key') or fam),
                definition_frozen_at_utc=proof['definition_frozen_at_utc'],
                independent_preholdout_freeze_ref=proof['independent_preholdout_freeze_ref'],
                source_pinned_sha256=proof['pinned_source_sha256'],
                pregame_feature_vintage_pass=True,split_disjoint=True,
                source_neutral=True,overlap_audit='PASS',multiple_testing_audit='PASS',
                uses_2026_outcomes=False,live_recipe=live,
                live_signal_field=live.get('signal_field'),
                live_signal_type=live.get('signal_type'),
                data_ready_field=live.get('ready_field'),
                historical_reference_not_executable_roi=True)
            qualified.append(rule)
    approved=bool((attestations or {}).get('_operator_approval',{}).get('approved') is True
        and (attestations or {}).get('_operator_approval',{}).get('audit_ref'))
    status=('FROZEN_OOS_QUALIFIED' if qualified and approved
        else 'HOLD_OPERATOR_APPROVAL' if qualified else 'HOLD_INDEPENDENT_FREEZE_AND_VINTAGE_ATTESTATION')
    return {'schema':'NCAAF_V55_FROZEN_RULE_QUALIFICATION_AUDIT_V2',
        'policy':V55_QUALIFICATION_POLICY,'created_utc':created,'status':status,
        'operator_approved':approved,'uses_2026_outcomes':False,
        'history_first_season':min(seasons_seen) if seasons_seen else None,
        'discovery_last_season':2023,'confirmation_seasons':[2024,2025],
        'source_audit':{'observed_seasons':sorted(seasons_seen),
           'source_receipts':source_receipts or {},'oos_screen_policy':V55_OOS_SCREEN_POLICY,
           'invalid_firings':dict(invalid),
           'attestation_systems_supplied':len([k for k in (attestations or {}) if k!='_operator_approval'])},
        'systems':qualified,'candidate_system_count':len(diagnostics),
        'candidate_diagnostics':diagnostics,'hold_counts':dict(rejections),
        'retrospective_reference_not_executable_roi':True,
        'independent_original_pregame_capture_certified':False,'bet_authority':0}

def publish_v55_frozen_qualification(report, *, storage_client=None,
                                     bucket='sharp-models'):
    """Overwrite only the stable V5.5 qualification *audit*, not legacy V2.

    Stores hashed deterministic contents for reproducible source inspection.
    Does not publish a qualifying registry while freeze attestations are absent.
    """
    from google.cloud import storage
    client = storage_client or storage.Client()
    uri = 'gs://'+bucket+'/'+V55_QUALIFICATION_BLOB
    body = (json.dumps(report,sort_keys=True,default=str,allow_nan=False)+'\n').encode()
    client.bucket(bucket).blob(V55_QUALIFICATION_BLOB).upload_from_string(
        body,content_type='application/json')
    return {'status':report['status'],'uri':uri,'sha256':hashlib.sha256(body).hexdigest(),
            'candidate_systems':report['candidate_system_count'],
            'qualified_systems':len(report['systems']),'bet_authority':0}


def source_capture_field_diagnostics(raw):
    """Show exactly why a frozen raw family feed failed to attach signals."""
    data={}
    for fam,rows in raw.items():
        if fam=='FROZEN_MINER':continue
        allkeys=set();fires=Counter();markets=Counter();directions=Counter();ids=Counter()
        for r in rows:
            allkeys.update(r)
            fires[str(r.get('evaluation_status') or '<MISSING>').strip().upper()]+=1
            markets[str(r.get('market') or '<MISSING>').strip().lower()]+=1
            directions[str(r.get('direction') or r.get('rule_decision_side') or r.get('pick_side') or '<MISSING>').strip().upper()]+=1
            if r.get('cfbd_game_id'):ids['cfbd_game_id']+=1
            if r.get('source_game_id'):ids['source_game_id']+=1
            if r.get('game_id'):ids['game_id']+=1
        data[fam]={'source_rows':len(rows),'fields':sorted(allkeys),
                   'evaluation_status_counts':dict(fires.most_common(12)),
                   'market_value_counts':dict(markets.most_common(12)),
                   'direction_value_counts':dict(directions.most_common(12)),
                   'identity_field_counts':dict(ids)}
    return data

def replay_prior_brain_sources(*, storage_client=None,bucket='sharp-models',canonical_prefix=None,
                               publish_events=True,archive_client=None,
                               archive_project='sharplogger',archive_dataset='sharp_data'):
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
    stage2_uri,stage2_discovery=_discover_verified_stage2_uri(bucket,storage_client)
    stage2=[]
    if not stage2_uri:holds['STAGE2_FROZEN']=stage2_discovery
    if stage2_uri:
        try:
            content=_obj_bytes(stage2_uri,storage_client);stage2=_csv_bytes(content)
            receipts['STAGE2_FROZEN']={'uri':stage2_uri,'sha256':hashlib.sha256(content).hexdigest(),'rows':len(stage2)}
        except Exception as exc:holds['STAGE2_FROZEN']=type(exc).__name__+':'+str(exc)
    assumption=os.getenv('NCAAF_V55_COMBINED_MARKET_RESEARCH_ASSUMPTION','').strip().lower()
    if assumption and assumption not in MARKETS:
        raise ValueError('INVALID_RESEARCH_ONLY_COMBINED_MARKET_ASSUMPTION')
    summary,rows,groups=replay_existing_frozen_brains(
        data['historical_cfbd_game_manifest_v5.csv'],data['historical_cfbd_game_sides_v5.csv'],
        data['stage1_canonical_ledger_v5_audit.json'],stage2_events=stage2,
        raw_sources=raw,expert_cache=expert,source_receipts=receipts,
        combined_research_assumption=assumption)
    # Required integration: canonical firings MUST be evaluated as historical
    # research candidates, not simply printed and dropped after ROI() ran on an
    # empty original-capture table. ATS research is available even if original
    # executable quote ROI/certification is not.
    audit,bygame,_,team_map,_=_checked_canonical(
        data['historical_cfbd_game_manifest_v5.csv'],
        data['historical_cfbd_game_sides_v5.csv'],
        data['stage1_canonical_ledger_v5_audit.json'])
    summary['canonical_edge_research']=canonical_replay_edge_diagnostics(rows,bygame,team_map,groups)
    summary['raw_source_field_diagnostics']=source_capture_field_diagnostics(raw)
    summary['seven_brain_combination_research']=v55_historical_family_combinations(groups,team_map)
    # Reuse the SAME pinned canonical frozen fires for source-specific
    # qualification diagnostics. No parallel Miner discovery or legacy V2 JSON.
    # Only independently supplied system-by-system evidence can lift proof holds.
    # This is optional and read-only; the pinned canonical retrospective replay
    # is never allowed to self-attest an original rule freeze.
    qualification_evidence={}
    evidence_key=os.getenv('NCAAF_V55_QUALIFICATION_EVIDENCE_KEY',
        'research/ncaaf/v55/qualification_evidence/current.json')
    try:
        evidence_blob=storage_client.bucket(bucket).blob(evidence_key)
        qualification_evidence=json.loads(evidence_blob.download_as_text())
        if not isinstance(qualification_evidence,dict):
            raise ValueError('QUALIFICATION_EVIDENCE_NOT_OBJECT')
        summary['independent_evidence_source']={'uri':'gs://'+bucket+'/'+evidence_key,
            'status':'SOURCE_PRESENT_NOT_AUTOMATICALLY_CERTIFIED',
            'systems_supplied':len(qualification_evidence)}
    except Exception as exc:
        qualification_evidence={}
        summary['independent_evidence_source']={'status':'NOT_AVAILABLE_FAIL_CLOSED',
            'reason':type(exc).__name__,'uri':'gs://'+bucket+'/'+evidence_key}
    qualification=v55_frozen_qualification_report(rows,team_map,receipts,
        attestations=qualification_evidence)
    try:
        summary['frozen_rule_qualification']=publish_v55_frozen_qualification(
            qualification,storage_client=storage_client,bucket=bucket)
    except Exception as exc:
        summary['frozen_rule_qualification']={'status':'HOLD_QUALIFICATION_PUBLICATION_FAILED',
            'error':type(exc).__name__+':'+str(exc)[:240], 'bet_authority':0}

    summary['stage2_v3_reuse']={'frozen_events_uri':stage2_uri or None,
        'discovery_status':stage2_discovery,
        'events_loaded':len(stage2),'research_only_combined_market_assumption':assumption or None,
        'original_market_verified_by_assumption':False,'bet_authority':0}
    if archive_client is not None:
        try:
            summary['canonical_provider_market_research']=canonical_provider_brain_research(
                rows,bygame, _checked_canonical(
                    data['historical_cfbd_game_manifest_v5.csv'],
                    data['historical_cfbd_game_sides_v5.csv'],
                    data['stage1_canonical_ledger_v5_audit.json'])[2],
                team_map,client=archive_client,project=archive_project,
                dataset=archive_dataset)
        except Exception as exc:
            summary['canonical_provider_market_research']={
                'status':'HOLD_ARCHIVED_PROVIDER_JOIN_EXCEPTION',
                'exception':type(exc).__name__+':'+str(exc),
                'bet_authority':0}
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



# -------------------- Canonical archive × frozen-brain integration --------------------
def canonical_provider_brain_research(events, games, native_ids, team_map, *, client,
                                      project='sharplogger', dataset='sharp_data',
                                      max_quote_rows=200000, top_n=200):
    """Join frozen historical firings to EXISTING archived bookmaker lines.

    This is a retrospective MARKET-RESEARCH lane. Archive records do not carry
    original quote timestamps or spread/totals vig. Consequently its ATS and
    ML quoted-return measurements are NOT original-capture bet certificates.

    Exact (season, CFBD-id or preverified source-id) is mandatory; unmatched
    neutral games cannot silently acquire bookmaker HOME/AWAY conventions.
    Any line or odds convention requiring unverified inference is labeled and
    excluded from live betting authority. No ML bet is inferred from a spread
    fire or vice versa.
    """
    import re
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*',project) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',dataset):
        raise ValueError('INVALID_WAREHOUSE_NAME')
    if not 0 < max_quote_rows <= 1000000:raise ValueError('INVALID_ARCHIVE_MAX_ROWS')
    if not 0 < top_n <= 10000:raise ValueError('INVALID_ARCHIVE_REPORT_LIMIT')
    fq=f'`{project}.{dataset}.ncaaf_archive_market_provider_quotes_v1`'
    sql=(f'SELECT season,game_id,provider,spread_quoted,total_quoted,home_ml_quoted,away_ml_quoted '
         f'FROM {fq} WHERE SAFE_CAST(season AS INT64) BETWEEN 2013 AND 2025 '
         f'ORDER BY season,game_id,provider LIMIT {max_quote_rows+1}')
    by_game=defaultdict(list)
    for e in events:
        by_game[(int(e['season']),str(e['cfbd_game_id']))].append(e)
    counters=Counter(); by_family=defaultdict(Counter); by_system=defaultdict(Counter)
    by_pair=defaultdict(Counter); cover_by_year=defaultdict(Counter); provider_counts=Counter()
    # A game contributes once per provider/market/system, regardless of how
    # many raw firings or repeat quotes exist in the feed.
    seen=set(); seen_family=set(); matched_games=set(); evaluated_sides=set()
    # Untimed quote rows sometimes contain several values for the SAME
    # provider/game. Without a quote timestamp there is no defensible way to
    # choose the winning one. Reject that market's conflicting offers rather
    # than treating whichever row appears first as an executable price.
    quote_rows=list(_source_research_rows(sql,client))
    disagreement=defaultdict(lambda:defaultdict(set))
    for q in quote_rows[:max_quote_rows]:
        grp=(str(q.get('season')),str(q.get('game_id')),str(q.get('provider') or '').lower())
        for field in ('spread_quoted','total_quoted','home_ml_quoted','away_ml_quoted'):
            val=num(q.get(field))
            if val is not None:disagreement[grp][field].add(val)
    conflicts={k:{name for name,values in group.items() if len(values)>1}
               for k,group in disagreement.items()}
    for _,badfields in conflicts.items():
        for field in badfields:counters['UNTIMED_CONFLICT_'+field.upper()]+=1
    rows_read=0
    for r in quote_rows:
        rows_read+=1
        if rows_read>max_quote_rows:break
        counters['archived_rows_read']+=1
        year=int(r['season']); ident=str(r.get('game_id') or '').strip()
        g1=ident if (year,ident) in games else ''
        g2=native_ids.get((year,ident),'')
        if g1 and g2 and g1!=g2:
            counters['IDENTITY_COLLISION']+=1;continue
        gid=g1 or g2
        if not gid:
            counters['NO_VERIFIED_CANONICAL_GAME_ID']+=1;continue
        key=(year,gid);game=games[key];members=team_map[key]
        if game.get('role_verification_status')!='HOME_AWAY_VERIFIED':
            counters['NO_VERIFIED_ARCHIVE_HOME_AWAY_CONVENTION']+=1;continue
        home=[(t,v) for t,v in members.items() if str(v.get('is_home_verified')).strip()=='1']
        away=[(t,v) for t,v in members.items() if str(v.get('is_home_verified')).strip()=='0']
        if len(home)!=1 or len(away)!=1 or home[0][0]==away[0][0]:
            counters['HOME_AWAY_SIDES_NOT_UNIQUE']+=1;continue
        ht,hrow=home[0];at,arow=away[0]
        hs,as_=num(hrow.get('team_score')),num(arow.get('team_score'))
        if hs is None or as_ is None or (num(hrow.get('opponent_score'))!=as_) or (num(arow.get('opponent_score'))!=hs):
            counters['MISSING_OR_CONFLICTING_FINAL']+=1;continue
        fam_events=by_game.get(key,[])
        if not fam_events:
            counters['CANONICAL_GAME_WITH_NO_BRAIN_FIRE']+=1;continue
        provider=str(r.get('provider') or '').strip().lower()
        if not provider:
            counters['MISSING_BOOK']+=1;continue
        provider_counts[provider]+=1
        counters['PROVIDER_IDENTITY_MATCHED_ROWS']+=1
        matched_games.add(key)
        # The raw archive spread convention was not established by the source
        # audit. Treat 'spread_quoted' as a home-oriented REFERENCE hypothesis,
        # never as a verified executable wager. All this lane's cover grades
        # remain explicitly provisional until convention is verified.
        badfields=conflicts[(str(r.get('season')),str(r.get('game_id')),provider)]
        spread=None if 'spread_quoted' in badfields else num(r.get('spread_quoted'))
        total=None if 'total_quoted' in badfields else num(r.get('total_quoted'))
        ml={'HOME':None if 'home_ml_quoted' in badfields else num(r.get('home_ml_quoted')),
            'AWAY':None if 'away_ml_quoted' in badfields else num(r.get('away_ml_quoted'))}
        if spread is None:counters['SPREAD_REFERENCE_NOT_NUMERIC']+=1
        if total is None:counters['TOTAL_REFERENCE_NOT_NUMERIC']+=1
        fs=defaultdict(set)
        for ev in fam_events:
            market=normal_market(ev.get('market'))
            target=str(ev.get('target') or '')
            if market=='spreads':
                if target not in (ht,at):
                    counters['SPREAD_FIRE_NOT_NAMED_ON_THIS_GAME']+=1;continue
                if spread is None:continue
                side='HOME' if target==ht else 'AWAY'
                # Game must be graded from the SELECTED TEAM perspective;
                # HOME quote sign reverses for the away-team bet.
                line=spread if side=='HOME' else -spread
                score_for=hs if side=='HOME' else as_
                score_against=as_ if side=='HOME' else hs
                outcome=grade('spreads','HOME',line,score_for,score_against)
                unit_profit=None   # archive contains no verified spread vig
            elif market=='h2h':
                if target not in (ht,at):
                    counters['H2H_FIRE_NOT_NAMED_ON_THIS_GAME']+=1;continue
                side='HOME' if target==ht else 'AWAY'
                quote=ml[side]
                if odds_profit(quote) is None:
                    counters['MISSING_OR_INVALID_ML_ODDS']+=1;continue
                outcome=grade('h2h',side,None,hs,as_)
                unit_profit=odds_profit(quote) if outcome=='WIN' else -1.0 if outcome=='LOSS' else 0.0
            elif market=='totals':
                if total is None or target not in ('OVER','UNDER'):continue
                outcome=grade('totals',target,total,hs,as_)
                side=target;unit_profit=None
            else:continue
            if outcome not in ('WIN','LOSS','PUSH'):continue
            sk=(year,gid,provider,market,side,str(ev.get('system_id') or ''))
            if sk in seen:
                counters['DUPLICATE_PROVIDER_SYSTEM_GRADE_SUPPRESSED']+=1;continue
            seen.add(sk)
            fam=str(ev['brain']);system=str(ev['system_id'])
            family_key=(year,gid,provider,market,side,fam)
            if family_key not in seen_family:
                seen_family.add(family_key)
                for ctr in (by_family[(fam,market)],cover_by_year[(year,fam,market)]):
                    ctr[outcome]+=1
                    if unit_profit is not None:
                        ctr['ML_RETURN_UNITS']+=unit_profit;ctr['ML_RETURN_N']+=1
            else:
                counters['DUPLICATE_SAME_BRAIN_FAMILY_SUPPORT_NOT_INDEPENDENT']+=1
            ctr=by_system[(fam,system,market)]
            ctr[outcome]+=1
            if unit_profit is not None:
                ctr['ML_RETURN_UNITS']+=unit_profit;ctr['ML_RETURN_N']+=1
            fs[(market,side)].add(fam)
            evaluated_sides.add((year,gid,market,side))
            counters['BRAIN_PROVIDER_MARKET_GRADES']+=1
        # A pair of independent brain families supporting the SAME market and
        # side, on the SAME provider quote, is one combined observation.
        for (market,side),families in fs.items():
            opposing=('AWAY' if side=='HOME' else 'HOME' if side=='AWAY'
                      else 'UNDER' if side=='OVER' else 'OVER')
            if fs.get((market,opposing)):
                counters['CONFLICTED_PROVIDER_SIDE_EXCLUDED']+=1;continue
            for a,b in itertools.combinations(sorted(families),2):
                if market=='h2h':
                    quote=ml[side];profit=(odds_profit(quote) if side=='HOME' and hs>as_ or side=='AWAY' and as_>hs
                          else -1.0 if hs!=as_ else 0.0)
                else:profit=None
                k=(a,b,market)
                grade_outcome=grade('h2h',side,None,hs,as_) if market=='h2h' else (
                    grade('totals',side,total,hs,as_) if market=='totals' else
                    grade('spreads','HOME',spread if side=='HOME' else -spread,
                          hs if side=='HOME' else as_,as_ if side=='HOME' else hs))
                by_pair[k][grade_outcome]+=1
                if profit is not None:
                    by_pair[k]['ML_RETURN_UNITS']+=profit;by_pair[k]['ML_RETURN_N']+=1
                counters['INDEPENDENT_FAMILY_PAIR_GRADES']+=1
    if rows_read>max_quote_rows:counters['ARCHIVE_TRUNCATED__NOT_COMPLETE_HISTORY']=1
    def record(label,k,v):
        n=v['WIN']+v['LOSS']+v['PUSH'];m=v['ML_RETURN_N']
        return {'key':list(k),'n':n,'wins':v['WIN'],'losses':v['LOSS'],'pushes':v['PUSH'],
                'ats_or_ml_win_rate':round(v['WIN']/(v['WIN']+v['LOSS']),6) if v['WIN']+v['LOSS'] else None,
                'retrospective_untimed_ml_quote_roi':round(v['ML_RETURN_UNITS']/m,6) if m else None,
                'ml_price_observations':m,'source':'UNTIMED_ARCHIVE__NOT_CERTIFIED',
                'price_executable_verified':False}
    def limited(table):
        arr=[record('x',key,value) for key,value in table.items()]
        return sorted(arr,key=lambda x:(-x['n'],x['key']))[:top_n],len(arr)
    families,n_families=limited(by_family)
    systems,n_systems=limited(by_system)
    pairs,n_pairs=limited(by_pair)
    status=('ARCHIVE_MARKET_AND_BRAINS_JOINED_RESEARCH_ONLY' if counters['BRAIN_PROVIDER_MARKET_GRADES']
            else 'HOLD_ARCHIVE_HAS_NO_VERIFIABLE_BRAIN_MARKET_JOIN')
    return {'status':status,'source_table':f'{project}.{dataset}.ncaaf_archive_market_provider_quotes_v1',
            'source_rows_inspected':min(rows_read,max_quote_rows),'archive_truncated':rows_read>max_quote_rows,
            'distinct_games_joined':len(matched_games),'distinct_market_sides_graded':len(evaluated_sides),
            'provider_counts':dict(provider_counts),'join_and_grading_counts':dict(counters),
            'brain_market_research':families,'individual_systems_top':systems,
            'individual_systems_total':n_systems,'independent_family_pairs_top':pairs,
            'independent_family_pairs_total':n_pairs,
            'by_season_brain_market':[record('year',k,v) for k,v in sorted(cover_by_year.items())],
            'spread_quote_sign_convention_verified':False,
            'spread_roi_calculated':False,'moneyline_quote_captured_at_pregame_verified':False,
            'totals_roi_calculated':False,'original_quote_timestamps_available':False,
            'original_brain_signal_vintage_verified':False,
            'conservative_roi_certificate_count':0,'bet_authority':0}


def market_rich_feature_coverage(*, client=None, project='sharplogger', dataset='sharp_data',
                                 publish_uri=None, storage_client=None):
    """Ingest ALL recorded pregame Market Rich rows, including sparse/zero features.

    Market Rich is an observational feature family, NOT a directional BET family.
    Store the full source rows as a GCS gzip JSONL feed; null score, zero moves,
    missing directional legacy flag, and repeated movements are NOT exclusions.
    Historical 2013-2025 absence is reported as coverage, never backfilled.
    """
    import re, gzip, tempfile
    if not (re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*',project) and
            re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',dataset)):
        raise ValueError('INVALID_MARKET_RICH_DATASET')
    if client is None:
        from google.cloud import bigquery
        client=bigquery.Client(project=project)
    table=f'{project}.{dataset}.moves_with_features_merged'
    base=(f"FROM `{table}` WHERE UPPER(Sport) IN "
          "('NCAAF','NCAA FOOTBALL','COLLEGE FOOTBALL') "
          'AND Game_Start IS NOT NULL AND Time IS NOT NULL AND Time < Game_Start')
    sql=f"""SELECT EXTRACT(YEAR FROM Game_Start) AS season,
      COUNT(*) AS pregame_rows,
      COUNT(DISTINCT Game_Key) AS pregame_games,
      COUNTIF(SharpBetScore IS NOT NULL) AS pregame_scored_rows,
      COUNTIF(Sharp_Move_Signal = 1) AS pregame_sharp_move_rows,
      COUNTIF(Delta_vs_Sharp IS NOT NULL OR Sharp_Limit_Jump IS NOT NULL OR
          Sharp_Time_Score IS NOT NULL) AS pregame_feature_rows
    {base} GROUP BY season ORDER BY season"""
    try:
        rows=[dict(row.items()) for row in client.query(sql).result()]
    except Exception as exc:
        return {'status':'HOLD_MARKET_RICH_SOURCE_QUERY_FAILED',
                'source':table,'exception':type(exc).__name__+':'+str(exc),
                'legacy_SHARP_SIDE_TO_BET_required':False,'bet_authority':0}
    totals={k:sum(int(r.get(k) or 0) for r in rows) for k in
            ('pregame_rows','pregame_games','pregame_scored_rows',
             'pregame_sharp_move_rows','pregame_feature_rows')}
    # Counts are descriptive; NO feature completeness requirement for an event.
    result={'source':table,'by_season':rows,'totals':totals,
            # This *separate* published feed is for forward production only.
            # It must never be represented as historical Market Rich training.
            'data_role':'FORWARD_LIVE_MARKET_RICH_FEED_NOT_HISTORICAL_RESEARCH',
            'historical_training_source':f'{project}.{dataset}.scores_with_features',
            'historical_training_seasons':'2013-2025_ONLY',
            'forward_feed_excluded_from_historical_fitting':True,
            'historical_2013_2025_rows':sum(int(r.get('pregame_rows') or 0)
                for r in rows if 2013<=int(r.get('season') or 0)<=2025),
            'legacy_SHARP_SIDE_TO_BET_required':False,
            'directional_signal_required':False,'original_signal_capture_certified':False,
            'ingestion_policy':'ALL_PREKICK_ROWS_NO_SCORE_OR_SIGNAL_FILTER',
            'bet_authority':0}
    if not totals['pregame_rows']:
        result['status']='HOLD_NO_MARKET_RICH_PREGAME_ROWS'
        return result
    if publish_uri is None:
        result['status']='MARKET_RICH_PREGAME_ROWS_AVAILABLE_NOT_PUBLISHED'
        return result
    if not (publish_uri.startswith('gs://') and publish_uri.endswith('.jsonl.gz')):
        raise ValueError('INVALID_MARKET_RICH_PUBLISH_URI')
    from urllib.parse import urlsplit
    location=urlsplit(publish_uri)
    from collections import Counter as _Counter
    observed=_Counter(); count=0; sha=hashlib.sha256()
    try:
        with tempfile.TemporaryDirectory(prefix='v55_market_rich_') as temp:
            path=Path(temp)/'market_rich_pregame.jsonl.gz'
            with gzip.open(path,'wt',encoding='utf8') as out:
                # No LIMIT, no SharpBetScore requirement, no Sharp_Move_Signal filter.
                for raw in client.query(f'SELECT * {base} ORDER BY Game_Start, Time, Game_Key, Bookmaker').result():
                    obj=dict(raw.items()); out.write(json.dumps(obj,sort_keys=True,default=str,allow_nan=False)+'\n')
                    count+=1
                    stamp=obj.get('Game_Start')
                    if stamp is not None:observed[int(str(stamp)[:4])]+=1
            expected={int(r['season']):int(r['pregame_rows']) for r in rows}
            if count!=totals['pregame_rows'] or dict(observed)!=expected:
                result.update(status='HOLD_MARKET_RICH_ROW_RECONCILIATION_FAILED',
                              exported_rows=count,observed_by_season=dict(observed))
                return result
            if storage_client is None:
                from google.cloud import storage
                storage_client=storage.Client(project=project)
            bucket=storage_client.bucket(location.netloc)
            blob=bucket.blob(location.path.lstrip('/'))
            blob.upload_from_filename(str(path),content_type='application/gzip')
            with path.open('rb') as inp:
                for block in iter(lambda:inp.read(1024*1024),b''):sha.update(block)
        result.update(status='MARKET_RICH_PREGAME_FEATURE_FEED_AVAILABLE',
                      published_rows=count,publication_uri=publish_uri,
                      publication_sha256=sha.hexdigest(),by_season_published=dict(observed))
    except Exception as exc:
        result.update(status='HOLD_MARKET_RICH_PUBLISH_FAILED',exported_rows=count,
                      exception=type(exc).__name__+':'+str(exc))
    return result


# ---------------- V5.5 integrated ML research (not betting authority) ----------------
# This implementation deliberately shares the verified canonical Stage1/Stage2
# observation ledgers. No second candidate selector or independent job.
ML_CORE_FIELDS = ('Opening_Spread','Opening_Total','Away_Travel_Miles','Away_Time_Zones_Crossed',
                  'Neutral_Site','Venue_Elevation','Altitude_Gain',
                  'Team_Prior_ATS_Rate','Opponent_Prior_ATS_Rate',
                  'Team_Prior_Win_Rate','Opponent_Prior_Win_Rate',
                  'Team_Prior_Scoring_Avg','Opponent_Prior_Scoring_Avg')
ML_STAT_FIELDS = ('Opening_Spread','Opening_Total','Team_Prior_ATS_Rate','Opponent_Prior_ATS_Rate',
                  'Team_Prior_Win_Rate','Opponent_Prior_Win_Rate',
                  'Team_Prior_Scoring_Avg','Opponent_Prior_Scoring_Avg')
ML_MARKET_FIELDS = ('SharpBetScore','Sharp_Move_Signal','Delta_vs_Sharp',
                    'Sharp_Limit_Jump','Sharp_Time_Score','Limit_Jump',
                    'Market_Leader','Is_Pinnacle','Sharp_Limit_Total')
ML_SCHEMA = 'NCAAF_V55_ML_RESEARCH_V3_SPREAD_TOTAL_PREGAME_REFERENCES'
# These features are derived from historical scores_with_features snapshots.
# The current/live moves table is NEVER used for model fitting.
ML_HIST_MARKET_FIELDS = ('MARKET_SPREAD_OPEN','MARKET_SPREAD_LATEST',
                         'MARKET_SPREAD_MOVE','MARKET_TOTAL_OPEN','MARKET_TOTAL_LATEST',
                         'MARKET_TOTAL_MOVE','MARKET_H2H_HOME_PRICE',
                         'MARKET_SPREAD_BOOKS','MARKET_TOTAL_BOOKS',
                         'MARKET_SNAPSHOT_COUNT')

def ml_historical_market_rows(*,client,canonical_manifest,canonical_sides,
                              canonical_audit,project='sharplogger',dataset='sharp_data'):
    """Verified historical 2013-25 snapshots from scores_with_features ONLY.

    Match through the EXISTING V5.2 canonical crosswalk, accept only an
    observed pre-kick snapshot, and preserve bookmaker direction. No simulated
    historical sharp score or synthesized quote timestamp is permissible.
    """
    _, games, native, teams, _ = _checked_canonical(
        canonical_manifest,canonical_sides,canonical_audit)
    fq=f'{project}.{dataset}.scores_with_features'
    try:
        schema={f.name for f in client.get_table(fq).schema}
    except Exception as exc:
        return [],{'status':'HOLD_HISTORICAL_MARKET_TABLE_UNAVAILABLE',
                   'source':fq,'error':type(exc).__name__+':'+str(exc)}
    def column(*names):return next((n for n in names if n in schema),None)
    kickoff=column('feat_Game_Start','Game_Start','Commence_Hour')
    snapshot=column('Snapshot_Timestamp','Time','Observed_At','Captured_At','Quote_Timestamp')
    market=column('Market','market')
    outcome=column('Outcome','outcome')
    value=column('Value','Line_Value','Quoted_Value')
    key=column('Game_Key','game_key','source_game_id')
    home=column('Home_Team_Norm','Home_Team','home_team')
    away=column('Away_Team_Norm','Away_Team','away_team')
    opener=column('First_Line_Value','Open_Value','Opening_Line')
    book=column('Bookmaker','Book','bookmaker')
    missing=[name for name,v in [('kickoff',kickoff),('snapshot_timestamp',snapshot),
       ('market',market),('outcome',outcome),('value',value)] if v is None]
    if not key and (not home or not away):missing.append('game_identity')
    if missing:
        return [],{'status':'HOLD_HISTORICAL_MARKET_SOURCE_SCHEMA',
                   'source':fq,'missing':missing,'present_columns':sorted(schema)}
    selected=list(dict.fromkeys(x for x in (kickoff,snapshot,market,outcome,value,key,
                                            home,away,opener,book) if x))
    sql=('SELECT '+','.join('`'+c+'`' for c in selected)+f' FROM `{fq}` '
         f'WHERE EXTRACT(YEAR FROM SAFE_CAST(`{kickoff}` AS TIMESTAMP)) BETWEEN 2013 AND 2025 '
         f'AND SAFE_CAST(`{snapshot}` AS TIMESTAMP) < SAFE_CAST(`{kickoff}` AS TIMESTAMP) '
         f'AND `{value}` IS NOT NULL' +
         (" AND UPPER(TRIM(CAST(`Sport` AS STRING))) IN ('NCAAF','NCAA FOOTBALL','COLLEGE FOOTBALL')" if 'Sport' in schema else ''))
    try:
        rows=[dict(r.items()) for r in client.query(sql).result(page_size=10000)]
    except Exception as exc:
        return [],{'status':'HOLD_HISTORICAL_MARKET_SOURCE_QUERY',
                   'source':fq,'error':type(exc).__name__+':'+str(exc)}
    identity=_verified_snapshot_identity_index(canonical_manifest,canonical_sides,canonical_audit)
    holds=Counter(); grouped=defaultdict(list)
    for r in rows:
        kt=clock(r.get(kickoff));st=clock(r.get(snapshot))
        if kt is None or st is None or st>=kt:
            holds['NOT_VERIFIED_PREKICK']+=1;continue
        season=kt.year;gid=None
        src=str(r.get(key) or '').strip() if key else ''
        if src:gid=native.get((season,src)) or (src if (season,src) in games else None)
        # Existing identity recovery: both named teams, exact date, unique game.
        if not gid and home and away:
            test={'Commence_Hour':r.get(kickoff),
                  'Home_Team_Norm':r.get(home),'Away_Team_Norm':r.get(away)}
            gid,_=_resolve_exact_snapshot_identity(test,identity)
        if not gid:
            holds['NO_VERIFIED_CANONICAL_IDENTITY']+=1;continue
        g=games.get((season,gid))
        if not g or g.get('role_verification_status')!='HOME_AWAY_VERIFIED':
            holds['NO_VERIFIED_CANONICAL_ROLES']+=1;continue
        sides=teams[(season,gid)]
        homekeys=[k for k,v in sides.items() if str(v.get('is_home_verified'))=='1']
        if len(homekeys)!=1:
            holds['NON_UNIQUE_HOME_SIDE']+=1;continue
        if home and away:
            hn=_canonical_name(r.get(home));an=_canonical_name(r.get(away))
            sh=sides[homekeys[0]]
            ah=next(v for k,v in sides.items() if k!=homekeys[0])
            if hn not in {_canonical_name(sh.get('cfbd_team_key')),_canonical_name(sh.get('source_team_label'))} or \
               an not in {_canonical_name(ah.get('cfbd_team_key')),_canonical_name(ah.get('source_team_label'))}:
                holds['NAMED_SIDES_DISAGREE']+=1;continue
        side=str(r.get(outcome) or '').strip().lower()
        mk=normal_market(r.get(market))
        if mk not in MARKETS:
            holds['UNKNOWN_MARKET']+=1;continue
        if mk=='totals':
            if side not in ('over','under'):
                holds['TOTAL_DIRECTION_UNKNOWN']+=1;continue
        else:
            sh=sides[homekeys[0]]
            ah=next(v for k,v in sides.items() if k!=homekeys[0])
            hs={_canonical_name(sh.get('cfbd_team_key')),_canonical_name(sh.get('source_team_label'))}
            aws={_canonical_name(ah.get('cfbd_team_key')),_canonical_name(ah.get('source_team_label'))}
            sn=_canonical_name(side)
            if sn in hs:side='home'
            elif sn in aws:side='away'
            else:
                holds['UNVERIFIED_OUTCOME_SIDE']+=1;continue
        v=_ml_number(r.get(value));o=_ml_number(r.get(opener)) if opener else float('nan')
        if not math.isfinite(v):holds['INVALID_PRICE_VALUE']+=1;continue
        grouped[(season,gid)].append((st,mk,side,v,o,str(r.get(book) or '') if book else ''))
    result=[]
    for (season,gid),events in grouped.items():
        features={f:float('nan') for f in ML_HIST_MARKET_FIELDS}
        features['MARKET_SNAPSHOT_COUNT']=float(len(events))
        for mk,side,fopen,flast,fmove,fbooks in (
            ('spreads','home','MARKET_SPREAD_OPEN','MARKET_SPREAD_LATEST','MARKET_SPREAD_MOVE','MARKET_SPREAD_BOOKS'),
            ('totals','over','MARKET_TOTAL_OPEN','MARKET_TOTAL_LATEST','MARKET_TOTAL_MOVE','MARKET_TOTAL_BOOKS')):
            observed=sorted((e for e in events if e[1]==mk and e[2]==side),key=lambda e:(e[0],e[5]))
            if not observed:continue
            # Never manufacture movement by comparing DIFFERENT bookmakers.
            # Aggregate each book's own opening-to-latest change; do not
            # pick a bookmaker on the basis of the eventual result.
            per_book=defaultdict(list)
            for event in observed:per_book[event[5]].append(event)
            opens=[];lates=[];moves=[]
            for group in per_book.values():
                first,last=group[0],group[-1]
                opened=first[4] if math.isfinite(first[4]) else first[3]
                opens.append(opened);lates.append(last[3]);moves.append(last[3]-opened)
            features[fopen]=sum(opens)/len(opens)
            features[flast]=sum(lates)/len(lates)
            features[fmove]=sum(moves)/len(moves)
            features[fbooks]=float(len([b for b in per_book if b]))
        ml=sorted((e for e in events if e[1]=='h2h' and e[2]=='home'),key=lambda e:(e[0],e[5]))
        if ml:features['MARKET_H2H_HOME_PRICE']=ml[-1][3]
        if not any(math.isfinite(features[f]) for f in ML_HIST_MARKET_FIELDS if f!='MARKET_SNAPSHOT_COUNT'):
            holds['NO_DIRECTIONALLY_VERIFIED_MARKET_FEATURE']+=1;continue
        # The label reference is a verified prekick market quote, not a
        # retrospective closing number. Preserve direction (HOME/OVER).
        # No synthetic market lines are created from game scores.
        result.append({'season':season,'game_id':gid,'core_features':features,
                       'spread_home_line':features['MARKET_SPREAD_LATEST'],
                       'total_over_line':features['MARKET_TOTAL_LATEST'],
                       'spread_label_source':'SCORES_WITH_FEATURES_PREKICK_HOME',
                       'total_label_source':'SCORES_WITH_FEATURES_PREKICK_OVER',
                       'home_ml_quote':ml[-1][3] if ml else None,
                       'home_ml_quote_book':ml[-1][5] if ml else None,
                       'original_quote_capture_certified':False})
    return sorted(result,key=lambda x:(x['season'],x['game_id'])),{
        'status':'HISTORICAL_MARKET_SNAPSHOTS_JOINED' if result else 'HOLD_NO_HISTORICAL_MARKET_SNAPSHOTS_MATCHED',
        'source':fq,'snapshot_time_field':snapshot,'kickoff_field':kickoff,
        'scanned_prekick_rows':len(rows),'matched_games':len(result),'holds':dict(holds),
        'feature_vintage_certified':False,
        'source_policy':'PREKICK_TIMESTAMP_REQUIRED__STAGE1_V52_IDENTITIES__NO_FORWARD_MOVES'}


def _ml_number(value):
    if value in (None,'','NULL'):return float('nan')
    if isinstance(value,bool):return float(value)
    try:
        x=float(value)
        return x if math.isfinite(x) else float('nan')
    except (ValueError,TypeError):return float('nan')

def _ml_table_rows(client,table,required,optional,where):
    """Use ONLY explicit pregame allowlists; no outcomes can enter feature X."""
    if not all(x.replace('_','').isalnum() for x in required+optional):
        raise ValueError('INVALID_ML_FEATURE_IDENTIFIER')
    schema={f.name for f in client.get_table(table).schema}
    missing=sorted(set(required)-schema)
    if missing:return [],{'status':'HOLD_REQUIRED_SOURCE_COLUMNS_MISSING','missing':missing}
    included=[f for f in optional if f in schema]
    columns=list(required)+included
    sql='SELECT '+','.join('`'+c+'`' for c in columns)+' FROM `'+table+'` '+where
    rows=[dict(r.items()) for r in client.query(sql).result(page_size=10000)]
    return rows,{'status':'LOADED','source_rows':len(rows),'feature_columns':included,
                 'missing_optional':sorted(set(optional)-set(included))}

def ml_cfbd_prior_training_rows(*,client,canonical_manifest,canonical_sides,
                                canonical_audit,project='sharplogger',dataset='sharp_data'):
    """Build a game-level ML matrix from the existing CFBD pre-game *_Prior fields.

    CFBD Game_ID + Team_Key must match BOTH original V5.2 sides. Never use
    final score, contemporaneous game stats or retrospective system outcomes
    as input features. Historical spread and total labels are intentionally
    absent until the exact original line/total is independently established.
    """
    _,games,_,teams,_=_checked_canonical(canonical_manifest,canonical_sides,canonical_audit)
    table=f'{project}.{dataset}.ncaaf_cfbd_archive_research_v1'
    try:
        fields={f.name:getattr(getattr(f,'field_type',None),'upper',lambda: str(getattr(f,'field_type','')).upper())()
                for f in client.get_table(table).schema}
    except Exception as exc:
        return [],{'status':'HOLD_CFBD_SOURCE_UNAVAILABLE','source':table,
                   'error':type(exc).__name__+':'+str(exc)}
    required={'Game_ID','Game_Date','Team_Key'}
    if not required.issubset(fields):
        return [],{'status':'HOLD_CFBD_REQUIRED_FIELDS','source':table,
                   'missing':sorted(required-set(fields))}
    feature_cols=sorted(c for c,t in fields.items() if
        (c=='CFBD_Games_Prior' or (c.startswith('CFBD_') and c.endswith('_Prior')))
        and t in ('FLOAT','FLOAT64','INTEGER','INT64','NUMERIC','BIGNUMERIC'))
    if not feature_cols:
        return [],{'status':'HOLD_NO_CFBD_PREGAME_PRIORS','source':table}
    has_season='Season' in fields
    cols=['Game_ID','Game_Date','Team_Key']+(['Season'] if has_season else [])+feature_cols
    predicate=('SAFE_CAST(`Season` AS INT64) BETWEEN 2013 AND 2025' if has_season else
               'EXTRACT(YEAR FROM Game_Date) BETWEEN 2013 AND 2025')
    sql=('SELECT '+','.join('`'+c+'`' for c in cols)+f' FROM `{table}` WHERE '+predicate)
    try:
        source_rows=client.query(sql).result(page_size=10000)
    except Exception as exc:
        return [],{'status':'HOLD_CFBD_PRIOR_QUERY_FAILED','source':table,
                   'error':type(exc).__name__+':'+str(exc)}
    # Exactly two separately identified sides per canonical CFBD game.
    observed=defaultdict(dict);holds=Counter();raw_count=0
    # The immutable V5.2 two-sided ledger already supplies two independently
    # recorded names for each side. Only exact normalized, UNIQUE matches
    # against those ledger fields are admissible; never guess a team from a
    # score, opponent, fuzzy distance, or historical model prediction.
    side_aliases={}
    for game_key,side_map in teams.items():
        aliases=defaultdict(set)
        for side_key,side in side_map.items():
            for candidate in (side_key,side.get('cfbd_team_key'),side.get('source_team_label')):
                normalized=_canonical_name(candidate)
                if normalized:aliases[normalized].add(side_key)
        side_aliases[game_key]=aliases
    for rr in source_rows:
        raw_count+=1;r=dict(rr.items())
        gd=r.get('Game_Date')
        try:year=int(r['Season']) if has_season else int(str(gd)[:4]);gid=str(int(r['Game_ID']))
        except (ValueError,TypeError,OverflowError):
            holds['INVALID_GAME_ID_OR_DATE']+=1;continue
        if (year,gid) not in games:
            holds['NOT_IN_VERIFIED_STAGE1']+=1;continue
        if str(gd)[:10] != str(games[(year,gid)].get('game_date') or '')[:10]:
            holds['CFBD_GAME_DATE_DISAGREES_WITH_STAGE1']+=1;continue
        side_map=teams[(year,gid)]
        normalized_key=_canonical_name(r['Team_Key'])
        matches=side_aliases[(year,gid)].get(normalized_key,set())
        if len(matches)!=1:
            holds['CFBD_TEAM_KEY_NOT_EXACT_STAGE1_SIDE' if not matches else
                  'CFBD_TEAM_ALIAS_AMBIGUOUS_STAGE1_SIDES']+=1;continue
        k=next(iter(matches))
        if _canonical_name(k)!=normalized_key:
            holds['CFBD_VERIFIED_STAGE1_SOURCE_LABEL_ALIAS_RECOVERED']+=1
        current=observed[(year,gid)].get(k)
        if k in observed[(year,gid)]:
            # Exact repeat rows are a warehouse artifact, not two conflicting
            # observations. Compare all selected feature values and game keys;
            # a prior conflict (None) must NEVER be restored by later repeats.
            if current is not None and current==r:
                holds['IDENTICAL_CFBD_SIDE_REPEAT_DEDUPED']+=1
            else:
                observed[(year,gid)][k]=None
                holds['CONFLICTING_CFBD_SIDE_DUPLICATE']+=1
        else:observed[(year,gid)][k]=r
    matrix=[];matched_by_season=Counter();excluded_games_by_season=defaultdict(Counter)
    canonical_by_year=Counter(yr for yr,gid in games)
    observed_by_year=Counter(yr for yr,gid in observed)
    for year,gid in games:
        if (year,gid) not in observed:
            excluded_games_by_season[year]['NO_MATCHED_CFBD_SOURCE_SIDE_ROWS']+=1
    for (year,gid),side_rows in sorted(observed.items()):
        g=games[(year,gid)];side_map=teams[(year,gid)]
        if g.get('role_verification_status')!='HOME_AWAY_VERIFIED':
            holds['UNVERIFIED_GAME_ROLES']+=1
            excluded_games_by_season[year]['UNVERIFIED_GAME_ROLES']+=1
            continue
        home=[k for k,v in side_map.items() if str(v.get('is_home_verified')).lower() in ('1','true','yes')]
        if len(home)!=1 or len(side_rows)!=2 or set(side_rows)!=set(side_map) or any(v is None for v in side_rows.values()):
            holds['INCOMPLETE_OR_DUPLICATE_CFBD_TWO_SIDES']+=1
            excluded_games_by_season[year]['INCOMPLETE_OR_DUPLICATE_CFBD_TWO_SIDES']+=1
            continue
        hkey=home[0];akey=next(k for k in side_map if k!=hkey)
        hs=_ml_number(side_map[hkey].get('team_score'))
        av=_ml_number(side_map[akey].get('team_score'))
        h_opp=_ml_number(side_map[hkey].get('opponent_score'))
        a_opp=_ml_number(side_map[akey].get('opponent_score'))
        if not (all(math.isfinite(v) and v>=0 for v in (hs,av,h_opp,a_opp)) and hs==a_opp and av==h_opp):
            holds['UNVERIFIED_FINAL_SCORE_ORIENTATION']+=1;continue
        if hs==av:
            holds['TIED_FINAL_SCORE']+=1;continue
        hx=side_rows[hkey];ax=side_rows[akey]
        # Numeric prior-only feature differences preserve home/away orientation.
        features={}
        for c in feature_cols:
            hv=_ml_number(hx.get(c));vv=_ml_number(ax.get(c))
            features['CFBD_DIFF_'+c]=(hv-vv if math.isfinite(hv) and math.isfinite(vv) else float('nan'))
        if not any(math.isfinite(v) for v in features.values()):
            holds['ALL_PRIOR_FEATURES_NULL']+=1;continue
        matched_by_season[year]+=1
        matrix.append({'season':year,'game_id':gid,'date':g.get('game_date'),
                       'core_features':features,'home_score':hs,'away_score':av,
                       'label_h2h':int(hs>av),
                       'label_spread':None,'label_totals':None})
    audit={'status':'CFBD_PRIOR_MATRIX_BUILT' if matrix else 'HOLD_NO_MATCHED_CFBD_TRAINING_GAMES',
           'source':table,'source_rows':raw_count,'matched_games':len(matrix),
           'available_prior_columns':len(feature_cols),
           'canonical_games_by_season':dict(sorted(canonical_by_year.items())),
           'cfbd_games_with_at_least_one_matched_side_by_season':dict(sorted(observed_by_year.items())),
           'matched_games_by_season':dict(sorted(matched_by_season.items())),
           'excluded_canonical_games_by_season':{str(y):dict(c) for y,c in sorted(excluded_games_by_season.items())},
           'identity_recovery_policy':'ONLY_UNIQUE_VERIFIED_STAGE1_SIDE_LABEL_ALIASES; EXACT_REPEAT_DEDUP',
           'feature_policy':'ONLY_CFBD_PRIOR_FIELDS__HOME_MINUS_AWAY__CANONICAL_LABELS',
           'labels':{'h2h':len(matrix),'spreads':0,'totals':0},
           'holds':dict(holds),'feature_vintage_certified':False,
           'spread_total_policy':'TIMED_REFERENCES_FOR_PREGAME_LANE; UNTIMED_ARCHIVE_ONLY_FOR_SEPARATE_RETROSPECTIVE_LANE',
           'bet_authority':0}
    return matrix,audit


def ml_historical_training_rows(*,client=None,project='sharplogger',dataset='sharp_data',
                               canonical_manifest=None,canonical_sides=None,canonical_audit=None):
    """Canonical V5.2 + CFBD prior-feature dataset is the ML source of truth.

    Prior versions queried the legacy historical_core_training_vw first, then
    fell back to CFBD only when *zero* legacy games joined. This permitted a
    partially matching legacy view to silently substitute a small, differently
    populated sample for the 2013-2025 canonical CFBD population. Never do
    that. Reuse the already verified V5.2 ledger and historical CFBD prior
    features directly; Stage2/V6 frozen brains attach separately by that ID.

    Market snapshots and untimed reference quotes attach later, independently,
    preserving their separate executable/research-only evidence rules.
    """
    if canonical_manifest is None or canonical_sides is None or canonical_audit is None:
        return [],{'status':'HOLD_CANONICAL_V52_LEDGER_REQUIRED','bet_authority':0}
    if client is None:
        from google.cloud import bigquery
        client=bigquery.Client(project=project)
    rows,audit=ml_cfbd_prior_training_rows(client=client,
        canonical_manifest=canonical_manifest,canonical_sides=canonical_sides,
        canonical_audit=canonical_audit,project=project,dataset=dataset)
    audit['source_selection']='VERIFIED_V5_2_LEDGER_PLUS_CFBD_PRIORS_PRIMARY'
    audit['legacy_view_selection']='NEVER_USED_TO_REPLACE_VERIFIED_CFBD_POPULATION'
    audit['frozen_brain_inputs']='EXISTING_CANONICAL_V6_EVENTS_JOINED_BY_CFBD_GAME_ID'
    audit['historic_market_labels']='TIMED_SNAPSHOT_PRIMARY__UNTIMED_ARCHIVE_RETROSPECTIVE_ONLY'
    audit['bet_authority']=0
    return rows,audit

def _ml_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=90,learning_rate=.05,max_leaf_nodes=9,
             min_samples_leaf=55,l2_regularization=8.0,random_state=20261109)

def _ml_fitting_series(data,features,label,min_train=240,min_validation=30,period_field="season"):
    """Expanding-season OOF: only earlier years train each season."""
    import numpy as np
    X=np.array([[_ml_number(r.get('core_features',{}).get(k)) for k in features] for r in data],dtype=float)
    yy=np.array([r.get(label) if r.get(label) in (0,1) else -1 for r in data],dtype=int)
    years=np.array([int(r[period_field]) for r in data],dtype=int)
    # fixed fills based on each training window only; hist HGB handles NaNs directly.
    valid=yy>=0;pred=np.full(len(data),np.nan);folds=[]
    for yr in sorted(set(years)):
        train=np.where(valid & (years<yr))[0]
        test=np.where(valid & (years==yr))[0]
        if len(train)<min_train or len(test)<min_validation or len(set(yy[train]))<2:continue
        m=_ml_model();m.fit(X[train],yy[train]);pred[test]=m.predict_proba(X[test])[:,1]
        folds.append({'season':int(yr),'train_games':len(train),'test_games':len(test)})
    oof=np.isfinite(pred)
    report={'folds':folds,'oof_games':int(oof.sum()),'training_games':int(valid.sum()),
            'features':list(features),'period_type':period_field,'status':'EXPANDING_SEASON_OOF' if folds else 'HOLD_INSUFFICIENT_CHRONOLOGICAL_FOLDS'}
    if oof.any():
        from sklearn.metrics import log_loss, brier_score_loss
        report.update(oof_logloss=round(float(log_loss(yy[oof],pred[oof],labels=[0,1])),6),
                      oof_brier=round(float(brier_score_loss(yy[oof],pred[oof])),6))
    fitted=None
    if oof.any() and len(set(yy[valid]))==2:
        fitted=_ml_model();fitted.fit(X[valid],yy[valid])
    return pred,report,fitted


def _ml_fitting_within_season(data, features, label, *, min_train=240, min_validation=10):
    """Research-only weekly walk-forward when historical line labels cover one season.

    Same-week games never enter the earlier-week training sample. The model
    remains uncertified until availability of all features is independently
    established, even though market snapshots were observed before kickoff.
    """
    import numpy as np
    from sklearn.metrics import log_loss,brier_score_loss
    X=np.array([[_ml_number(r.get('core_features',{}).get(k)) for k in features] for r in data],dtype=float)
    yy=np.array([r.get(label) if r.get(label) in (0,1) else -1 for r in data],dtype=int)
    datekeys=[]
    for r in data:
        try:
            dt=datetime.fromisoformat(str(r.get('date') or '')[:10])
            yw=dt.isocalendar()
            datekeys.append(yw.year*100+yw.week)
        except (ValueError,AttributeError):datekeys.append(-1)
    dates=np.array(datekeys,dtype=int)
    valid=(yy>=0)&(dates>=0)
    p=np.full(len(data),np.nan);folds=[]
    for week in sorted(set(dates[valid])):
        train=np.where(valid & (dates<week))[0]
        test=np.where(valid & (dates==week))[0]
        if len(train)<min_train or len(test)<min_validation or len(set(yy[train]))<2:continue
        m=_ml_model();m.fit(X[train],yy[train]);p[test]=m.predict_proba(X[test])[:,1]
        folds.append({'iso_week':int(week),'train_games':len(train),'test_games':len(test)})
    tested=np.isfinite(p)
    audit={'status':'EXPLORATORY_WITHIN_SEASON_WEEKLY_OOF' if tested.any() else 'HOLD_INSUFFICIENT_WEEKLY_FOLDS',
           'period_type':'iso_week','training_games':int(valid.sum()),'oof_games':int(tested.sum()),
           'folds':folds,'features':list(features),'pre_game_feature_vintage_certified':False}
    if tested.any():
        audit.update(oof_brier=round(float(brier_score_loss(yy[tested],p[tested])),6),
                     oof_logloss=round(float(log_loss(yy[tested],p[tested],labels=[0,1])),6))
    fitted=None
    if tested.any() and len(set(yy[valid]))==2:
        fitted=_ml_model();fitted.fit(X[valid],yy[valid])
    return p,audit,fitted


def _ml_train_with_chronological_fallback(data,features,label,*,min_train=240):
    # Prefer season-isolated validation. Weekly fallback is explicitly
    # exploratory, and is only attempted when labeled data spans one season.
    pred,audit,model=_ml_fitting_series(data,features,label,min_train=min_train)
    if model is not None:return pred,audit,model
    seasons={int(r['season']) for r in data if r.get(label) in (0,1)}
    if len(seasons)!=1:return pred,audit,model
    wp,wa,wm=_ml_fitting_within_season(data,features,label,min_train=min_train)
    if wm is not None:
        wa['original_season_hold']=audit['status']
        return wp,wa,wm
    audit['weekly_fallback_hold']=wa['status']
    return pred,audit,model


def _ml_chronological_calibration(rows, oof_pred, label, *, min_train=240, min_validation=30):
    """Fit sigmoid calibrators only on earlier-season OOF base predictions.

    No current-season labels are available to the calibrator until a later
    season. A shadow forward calibrator is fit only after the audit succeeds.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss,brier_score_loss
    p=np.asarray(oof_pred,dtype=float)
    y=np.asarray([r.get(label) if r.get(label) in (0,1) else -1 for r in rows],dtype=int)
    years=np.asarray([int(r['season']) for r in rows],dtype=int)
    valid=np.isfinite(p)&(p>0)&(p<1)&(y>=0)
    logit=np.log(np.clip(p,1e-5,1-1e-5)/(1-np.clip(p,1e-5,1-1e-5)))
    cal=np.full(len(rows),np.nan); folds=[]
    for yr in sorted(set(years)):
        tr=np.where(valid & (years<yr))[0]
        te=np.where(valid & (years==yr))[0]
        if len(tr)<min_train or len(te)<min_validation or len(set(y[tr]))!=2:continue
        model=LogisticRegression(C=0.5,max_iter=300)
        model.fit(logit[tr].reshape(-1,1),y[tr])
        cal[te]=model.predict_proba(logit[te].reshape(-1,1))[:,1]
        folds.append({'season':int(yr),'earlier_oof_games':len(tr),'test_games':len(te)})
    tested=np.isfinite(cal)
    report={'status':'CALIBRATION_CHRONOLOGICAL_OOF' if tested.any() else 'HOLD_CALIBRATION_NO_VALID_FOLDS',
            'oof_games':int(tested.sum()),'folds':folds,'source_capture_certified':False}
    fitted=None
    if tested.any():
        report['same_game_raw_brier']=round(float(brier_score_loss(y[tested],p[tested])),6)
        report['same_game_raw_logloss']=round(float(log_loss(y[tested],p[tested],labels=[0,1])),6)
        report['calibrated_brier']=round(float(brier_score_loss(y[tested],cal[tested])),6)
        report['calibrated_logloss']=round(float(log_loss(y[tested],cal[tested],labels=[0,1])),6)
        report['calibration_improves_brier']=report['calibrated_brier']<report['same_game_raw_brier']
        fitted=LogisticRegression(C=0.5,max_iter=300)
        fitted.fit(logit[valid].reshape(-1,1),y[valid])
    return report,cal,fitted

def _ml_brain_counts(events):
    """No outcome-derived priors and no retroactive performance encoding."""
    counts=defaultdict(Counter);seen=set()
    for e in events:
        if e.get('market') not in MARKETS:continue
        family=str(e.get('brain') or '').upper()
        if family not in ('SYSTEM_MINER','BIG_AL','PATHI','PREDICTION_TRACKER'):continue
        try:
            key=(int(e['season']),str(e['cfbd_game_id']),str(e['target']),str(e['market']))
        except (KeyError,ValueError,TypeError):continue
        # One system cannot become multiple independent votes because the
        # same source event was ingested more than once. Different systems
        # remain distinct but the existing per-family cap still applies.
        system=str(e.get('system_id') or '').strip()
        # A missing system_id is not proof that two separate source events
        # came from one system. Deduplicate only byte-equivalent event
        # content when no explicit system identifier was recorded.
        identity=('SYSTEM',system) if system else ('EVENT',digest(e))
        unique=(key,family,identity)
        if unique in seen:continue
        seen.add(unique)
        counts[key][family]+=1
    return counts

def _ml_family_coverage(meta_rows, families=('SYSTEM_MINER','BIG_AL','PATHI','PREDICTION_TRACKER')):
    """Research feature-presence audit, not historical capture certification."""
    from collections import Counter
    by_family={}
    for family in families:
        field=family+'_NET'
        vals=[_ml_number(r.get('core_features',{}).get(field)) for r in meta_rows]
        active=[r for r,v in zip(meta_rows,vals) if math.isfinite(v) and v!=0]
        by_year=Counter(str(r['season']) for r in active)
        by_family[family]={'nonzero_net_games':len(active),'nonzero_net_games_by_season':dict(sorted(by_year.items())),
                           'feature_present_games':sum(math.isfinite(v) for v in vals),
                           'absent_or_balanced_net_games':len(meta_rows)-len(active),
                           'definition':'CAPPED_HOME_MINUS_AWAY_EVENT_COUNT__RETROSPECTIVE_UNCERTIFIED'}
    return {'meta_candidate_games':len(meta_rows),'families':by_family,
            'warning':'NONZERO_NET_IS_NOT_PROOF_OF_ORIGINALLY_AVAILABLE_PREGAME_EVIDENCE'}


def _ml_market_coverage(mrows):
    """Assess Market Rich fold viability without altering train/validation policy."""
    from collections import Counter
    by_year=Counter(str(r['season']) for r in mrows)
    by_label={k:dict(sorted(Counter(str(r['season']) for r in mrows if r.get(k) in (0,1)).items()))
              for k in ('label_h2h','label_spread','label_totals')}
    fields={f:sum(math.isfinite(_ml_number(r.get('core_features',{}).get(f))) for r in mrows)
            for f in ML_HIST_MARKET_FIELDS}
    return {'labeled_joined_games':len(mrows),'games_by_season':dict(sorted(by_year.items())),
            'label_games_by_season':by_label,'non_null_features':fields,
            'note':'Season folds require 240 earlier-year labeled games and 30 same-year labeled games by default.'}


def _ml_matched_probability_metrics(rows, predictions, label, min_games=1):
    """Strict paired-game scoring on exactly the same games for all named models."""
    from sklearn.metrics import brier_score_loss,log_loss
    from collections import Counter
    import numpy as np
    models=tuple(predictions)
    common=sorted(set.intersection(*(set(m) for m in predictions.values()))) if models else []
    by_key={(int(r['season']),str(r['game_id'])):r for r in rows}
    common=[key for key in common if key in by_key and by_key[key].get(label) in (0,1)
            and all(math.isfinite(float(predictions[name][key])) for name in models)]
    result={'status':'MATCHED_GAME_COMPARISON' if len(common)>=min_games else 'HOLD_NO_COMMON_OOF_GAMES',
            'same_games':len(common),'models':{},'by_season':dict(sorted(Counter(str(k[0]) for k in common).items())),
            'interpretation':'Paired metrics only, not a confidence interval or evidence of betting ROI.'}
    if not common:return result
    y=np.array([int(by_key[k][label]) for k in common])
    for name in models:
        p=np.array([float(predictions[name][k]) for k in common])
        result['models'][name]={'brier':round(float(brier_score_loss(y,p)),6),
                                'logloss':round(float(log_loss(y,p,labels=[0,1])),6)}
    return result


def _ml_attach_verified_market_labels(historical_rows, market_rows):
    """Attach market-conditional ATS/total targets to canonical *score* rows.

    Requires a prekick HOME spread or OVER total observation on a verified
    canonical game. Opening/movement values alone are not sufficient if the
    reference line is missing. Scores are labels ONLY, never model features.
    """
    index={(int(r['season']),str(r['game_id'])):r for r in historical_rows}
    audit=Counter(); examples=Counter()
    for market_row in market_rows or []:
        key=(int(market_row['season']),str(market_row['game_id']))
        r=index.get(key)
        if r is None:
            audit['NO_CANONICAL_TRAINING_ROW']+=1;continue
        hs=_ml_number(r.get('home_score'));aw=_ml_number(r.get('away_score'))
        if not all(math.isfinite(x) and x>=0 for x in (hs,aw)):
            audit['NO_CANONICAL_VERIFIED_SCORES']+=1;continue
        for kind,field,feature in (
            ('spread','spread_home_line','MARKET_SPREAD_LINE'),
            ('totals','total_over_line','MARKET_TOTAL_LINE')):
            line=_ml_number(market_row.get(field))
            if not math.isfinite(line):
                audit[kind+'_NO_VERIFIED_PREKICK_LINE']+=1;continue
            # Sanity bounds flag potentially mislabeled odds instead of
            # silently interpreting prices as points.
            if (kind=='spread' and not -70<=line<=70) or (kind=='totals' and not 1<=line<=150):
                audit[kind+'_OUT_OF_RANGE_LINE']+=1;continue
            if kind=='spread':
                margin=hs+line-aw;label='label_spread'
            else:
                margin=hs+aw-line;label='label_totals'
            if abs(margin)<1e-8:
                audit[kind+'_PUSH_EXCLUDED']+=1;continue
            target=int(margin>0)
            old=r.get(label)
            if old in (0,1) and old!=target:
                audit[kind+'_CONFLICTING_REFERENCE']+=1;continue
            r[label]=target
            r['core_features'][feature]=line
            r[kind+'_source']='VERIFIED_PREKICK_SCORES_WITH_FEATURES'
            audit[kind+'_LABELED']+=1
            examples[str(r['season'])+'_'+kind]+=1
    return {'status':'VERIFIED_PREGAME_MARKET_REFERENCE_LABEL_AUDIT',
            'counts':dict(audit),'labeled_by_season':dict(sorted(examples.items())),
            'market_reference':'LAST_PREKICK_OBSERVED_PER_BOOK__MEAN_ACROSS_BOOKS',
            'no_untimed_archive_fallback':True,
            'limitations':'Snapshot capture is prekick; independent market feature vintage certification remains required.'}


def _ml_ablation_report(meta_rows,fields,label,baseline_pred,*,min_train=240):
    """Same-game chronological leave-one-family-out comparison, research-only."""
    import numpy as np
    from sklearn.metrics import brier_score_loss,log_loss
    families=('SYSTEM_MINER','BIG_AL','PATHI','PREDICTION_TRACKER','MARKET_RICH')
    base=np.asarray(baseline_pred,dtype=float)
    y=np.asarray([r.get(label) if r.get(label) in (0,1) else -1 for r in meta_rows],dtype=int)
    results={}
    for fam in families:
        excluded={'MARKET_RICH_OOF'} if fam=='MARKET_RICH' else {fam+'_NET',fam+'_ORIGIN_OOF'}
        if not excluded.intersection(fields):continue
        reduced=[f for f in fields if f not in excluded]
        candidate,diagnostics,_=_ml_fitting_series(meta_rows,reduced,label,min_train=min_train)
        m=np.isfinite(base)&np.isfinite(candidate)&(y>=0)
        n=int(m.sum())
        if not n:
            results[fam]={'status':'HOLD_NO_IDENTICAL_OOF_GAMES','oof_games':0};continue
        bf=float(brier_score_loss(y[m],base[m]));af=float(brier_score_loss(y[m],candidate[m]))
        bl=float(log_loss(y[m],base[m],labels=[0,1]));al=float(log_loss(y[m],candidate[m],labels=[0,1]))
        results[fam]={'status':'EXPLORATORY_PAIRED_ABLATION','same_games':n,
                      'full_brier':round(bf,6),'without_brier':round(af,6),
                      'full_logloss':round(bl,6),'without_logloss':round(al,6),
                      'brier_gain_from_family':round(af-bf,6),
                      'logloss_gain_from_family':round(al-bl,6),
                      'warning':'RETROSPECTIVE_SOURCE_VINTAGE_UNCERTIFIED_NOT_PROOF_OF_BETTING_EDGE'}
    return {'status':'EXPLORATORY_SAME_GAME_CHRONOLOGICAL_ABLATION',
            'families':results,'capture_certified':False}



def ml_independent_candidate_audit(events, historical_rows, canonical_team_map, *, min_prior=20):
    """Audit EVERY family as a potential originator, without a CORE dependency.

    Each candidate is keyed to its own market/side/system. Prior historical W/L
    estimates use strictly earlier seasons, never the season being evaluated.
    Retrospective events lack original publication time and cannot authorize a bet.
    """
    families=('CORE','STAT','MARKET_RICH','SYSTEM_MINER','BIG_AL','PATHI','PREDICTION_TRACKER')
    by_game={(int(r['season']),str(r['game_id'])):r for r in historical_rows}
    seen=set(); accepted=[];holds=Counter(); coverage=defaultdict(Counter)
    for e in events:
        fam=str(e.get('brain') or '').upper(); mk=str(e.get('market') or '').lower()
        if fam not in families or mk not in ('h2h','spreads','totals'):
            holds['UNSUPPORTED_FAMILY_OR_MARKET']+=1;continue
        try:key=(int(e['season']),str(e['cfbd_game_id']))
        except (KeyError,ValueError,TypeError):
            holds['INVALID_GAME_KEY']+=1;continue
        target=str(e.get('target') or '').strip()
        system=str(e.get('system_id') or fam)
        unique=(key,mk,target,fam,system)
        if unique in seen:
            holds['DUPLICATE_SAME_SYSTEM']+=1;continue
        seen.add(unique)
        game=by_game.get(key)
        sides=canonical_team_map.get(key,{})
        if mk=='totals':
            if target not in ('OVER','UNDER'):
                holds['TOTAL_TARGET_NOT_EXPLICIT']+=1;continue
        else:
            if target not in sides:
                holds['TARGET_NOT_EXACT_CANONICAL_SIDE']+=1;continue
        coverage[(fam,mk)]['originations']+=1
        label={'h2h':'label_h2h','spreads':'label_spread','totals':'label_totals'}[mk]
        if not game or game.get(label) not in (0,1):
            holds['NO_MARKET_SPECIFIC_VERIFIED_LABEL']+=1;continue
        if mk=='totals':result=int(game[label] == int(target=='OVER'))
        else:
            side=sides[target]
            is_home=str(side.get('is_home_verified')).lower() in ('1','true','yes')
            if str(side.get('is_home_verified')).lower() not in ('0','1','true','false','yes','no'):
                holds['UNVERIFIED_HOME_AWAY_ROLE']+=1;continue
            result=int(game[label]==int(is_home))
        coverage[(fam,mk)]['graded']+=1
        coverage[(fam,mk)]['wins']+=result
        accepted.append((key[0],key[1],mk,target,fam,system,result))
    # Historical grade aggregation is based on unique GAME+FAMILY+SYSTEM+MARKET
    # origins; distinct source/market opportunities are not treated as independent
    # family support just because many rule observations fire for one game.
    years=sorted({row[0] for row in accepted})
    out_by_family=defaultdict(Counter)
    for yr in years:
        prior=defaultdict(lambda:[0,0])
        for y,g,m,t,f,system,result in accepted:
            if y>=yr:continue
            prior[(f,m,system)][0]+=1
            prior[(f,m,system)][1]+=result
        for y,g,m,t,f,system,result in accepted:
            if y!=yr:continue
            n,w=prior[(f,m,system)]
            if n>=min_prior:
                out_by_family[(f,m)]['prior_qualified']+=1
                # Beta(1,1) historical-only shrinkage, not a bet threshold.
    return {
        'status':'RETROSPECTIVE_INDEPENDENT_ORIGINATION_AUDIT_UNCERTIFIED',
        'core_required':False,'bet_authority':0,'source_capture_certified':False,
        'families':{f:{m:dict(coverage.get((f,m),{}),
                prior_qualified=out_by_family[(f,m)]['prior_qualified'])
                for m in ('h2h','spreads','totals')} for f in families},
        'unique_originations':sum(c['originations'] for c in coverage.values()),
        'graded_system_occurrences':len(accepted),
        'season_prior_qualified_candidates':sum(c['prior_qualified'] for c in out_by_family.values()),
        'holds':dict(holds),
        'warning':'FROZEN_RETROSPECTIVE_EVIDENCE_CAN_HAVE_SELECTION_BIAS; NO_AUTO_BET_AUTHORITY',
    }


def ml_archived_quote_vintage_audit(*,client,project='sharplogger',dataset='sharp_data'):
    """Discover earlier historical line coverage WITHOUT declaring untimed bets pregame.

    This query intentionally does not return archived lines to training. The
    archive lacks a certified capture timestamp and original source vintage.
    """
    fq=f'{project}.{dataset}.ncaaf_archive_market_provider_quotes_v1'
    try:
        table=client.get_table(fq)
        cols={c.name for c in table.schema}
        required={'season','game_id','provider','spread_quoted','total_quoted'}
        if not required.issubset(cols):
            return {'status':'HOLD_ARCHIVED_QUOTE_SCHEMA','missing_columns':sorted(required-cols)}
        # Reconstructing capture time from a file date or kickoff is forbidden.
        capture=next((c for c in ('quote_captured_at','captured_at','snapshot_timestamp','observed_at') if c in cols),None)
        sql=f"""SELECT SAFE_CAST(season AS INT64) AS season,
          COUNT(*) AS quote_rows, COUNT(DISTINCT CAST(game_id AS STRING)) AS distinct_source_games,
          COUNTIF(SAFE_CAST(spread_quoted AS FLOAT64) IS NOT NULL) AS spread_rows,
          COUNTIF(SAFE_CAST(total_quoted AS FLOAT64) IS NOT NULL) AS totals_rows
          FROM `{fq}` WHERE SAFE_CAST(season AS INT64) BETWEEN 2013 AND 2025
          GROUP BY season ORDER BY season"""
        years=[dict(r.items()) for r in client.query(sql).result()]
        return {'status':'ARCHIVED_MULTISEASON_QUOTES_UNTIMED_RESEARCH_ONLY',
                'source':fq,'by_season':years,'capture_column_present':bool(capture),
                'eligible_for_certified_training':False,
                'eligible_for_retrospective_reference_research':True,
                'hold':'ORIGINAL_CAPTURE_TIME_AND_LINE_CONVENTIONS_NOT_CERTIFIED',
                'policy':'ALLOW_SEPARATE_UNTIMED_REFERENCE_RESEARCH_ONLY; NEVER_LIVE_AUTHORITY_OR_CERTIFIED_OOF'}
    except Exception as exc:
        return {'status':'HOLD_ARCHIVED_QUOTE_AUDIT_FAILED','source':fq,
                'error':type(exc).__name__+':'+str(exc)}



# ------------------ Context-dependent timestamp policy ------------------
# A timestamp is essential to establish original/executable quote availability,
# intra-book movement, or the existence of a historical PT publication.
# It is NOT needed to grade a game against an *explicitly retrospective*
# reference spread/total. Such grades remain a separate, non-authoritative lane.

def ml_archived_reference_line_rows(*, client, canonical_manifest,
                                    canonical_sides, canonical_audit,
                                    project='sharplogger', dataset='sharp_data',
                                    max_rows=200000):
    """Load untimed historical references for exploratory, NOT executable, ML.

    Exact Stage1 V5.2 game identity and verified home/away orientation are
    required. A provider/game with inconsistent untimed values is excluded
    per market; valid providers aggregate by deterministic median. The archive
    spread's sign convention is UNVERIFIED: HOME is an explicit hypothesis and
    reverse-sign sensitivity is audited. No time is forged or inferred.
    """
    import re, statistics
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*',project) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',dataset):
        raise ValueError('INVALID_WAREHOUSE_NAME')
    if not 0 < max_rows <= 1000000:raise ValueError('INVALID_ARCHIVE_MAX_ROWS')
    _, games, native, teams, _=_checked_canonical(canonical_manifest,canonical_sides,canonical_audit)
    fq=f'{project}.{dataset}.ncaaf_archive_market_provider_quotes_v1'
    try:
        fields={f.name for f in client.get_table(fq).schema}
        required={'season','game_id','provider','spread_quoted','total_quoted'}
        if not required.issubset(fields):
            return [],{'status':'HOLD_ARCHIVE_REFERENCE_SCHEMA','missing':sorted(required-fields),
                       'source':fq,'bet_authority':0}
        sql=(f'SELECT season,game_id,provider,spread_quoted,total_quoted '
             f'FROM `{fq}` WHERE SAFE_CAST(season AS INT64) BETWEEN 2013 AND 2025 '
             f'ORDER BY season,game_id,provider LIMIT {max_rows+1}')
        rows=client.query(sql).result(page_size=10000)
    except Exception as exc:
        return [],{'status':'HOLD_ARCHIVE_REFERENCE_QUERY','source':fq,
                   'error':type(exc).__name__+':'+str(exc),'bet_authority':0}
    holds=Counter();providers=defaultdict(lambda:defaultdict(set));scanned=0
    for rr in rows:
        scanned+=1
        if scanned>max_rows:
            holds['ROW_LIMIT_REACHED']+=1;break
        r=dict(rr.items())
        try:yr=int(r.get('season'))
        except (ValueError,TypeError):holds['INVALID_SEASON']+=1;continue
        original=str(r.get('game_id') or '').strip()
        direct=original if (yr,original) in games else None
        cross=native.get((yr,original))
        if direct and cross and direct!=cross:
            holds['CANONICAL_ID_COLLISION']+=1;continue
        gid=direct or cross
        if not gid:
            holds['NO_VERIFIED_CANONICAL_GAME_ID']+=1;continue
        key=(yr,gid)
        if games[key].get('role_verification_status')!='HOME_AWAY_VERIFIED':
            holds['NO_VERIFIED_HOME_AWAY_ROLE']+=1;continue
        sides=teams[key]
        home=[a for a,v in sides.items() if str(v.get('is_home_verified')).strip()=='1']
        away=[a for a,v in sides.items() if str(v.get('is_home_verified')).strip()=='0']
        if len(home)!=1 or len(away)!=1 or home[0]==away[0]:
            holds['HOME_AWAY_NOT_UNIQUE']+=1;continue
        ht,at=sides[home[0]],sides[away[0]]
        h, a=num(ht.get('team_score')),num(at.get('team_score'))
        if (h is None or a is None or h<0 or a<0 or
            num(ht.get('opponent_score'))!=a or num(at.get('opponent_score'))!=h):
            holds['CANONICAL_SCORE_ORIENTATION_UNVERIFIED']+=1;continue
        book=str(r.get('provider') or '').strip().lower()
        if not book:holds['MISSING_PROVIDER']+=1;continue
        for market,field,low,high in (('spreads','spread_quoted',-70,70),
                                      ('totals','total_quoted',1,150)):
            v=num(r.get(field))
            if v is None or not low<=v<=high:
                holds[market+'_MISSING_OR_OUT_OF_RANGE']+=1;continue
            providers[(yr,gid,market)][book].add(float(v))
    if scanned>max_rows:
        return [],{'status':'HOLD_ARCHIVE_TRUNCATED','source':fq,
                   'rows_scanned':scanned,'holds':dict(holds),'bet_authority':0}
    game_refs=defaultdict(dict);by_year=Counter()
    for (yr,gid,market),by_book in sorted(providers.items()):
        good=[]
        for book,values in by_book.items():
            if len(values)!=1:
                holds['CONFLICTING_UNTIMED_PROVIDER_'+market.upper()]+=1;continue
            good.append(next(iter(values)))
        if not good:
            holds['ALL_PROVIDER_QUOTES_CONFLICT_'+market.upper()]+=1;continue
        # A wide spread across untimed providers may represent different
        # measurement periods or a bad line mapping. Do not synthesize a
        # representative reference from irreconcilable quotes.
        if len(good)>1 and max(good)-min(good)>(7.0 if market=='spreads' else 10.0):
            holds['CROSS_PROVIDER_DISPERSION_HOLD_'+market.upper()]+=1;continue
        line=float(statistics.median_low(good))
        game_refs[(yr,gid)][market]=(line,len(good))
        by_year[(yr,market)]+=1
    references=[]
    for (yr,gid),pairs in sorted(game_refs.items()):
        references.append({'season':yr,'game_id':gid,
            'spread_home_line_research':pairs.get('spreads',(None,0))[0],
            'total_over_line_research':pairs.get('totals',(None,0))[0],
            'spread_reference_books':pairs.get('spreads',(None,0))[1],
            'total_reference_books':pairs.get('totals',(None,0))[1],
            'reference_type':'UNTIMED_HISTORICAL_PROVIDER_MEDIAN',
            'spread_orientation':'ASSUMED_HOME_QUOTE_UNVERIFIED',
            'pregame_capture_verified':False,
            'executable_price_verified':False})
    return references,{
        'status':'ARCHIVED_REFERENCE_LINES_READY_RESEARCH_ONLY' if references else 'HOLD_NO_ARCHIVED_REFERENCE_LINES',
        'source':fq,'scanned_rows':scanned,'distinct_canonical_games':len(references),
        'reference_games_by_season_market':{f'{yr}_{market}':n for (yr,market),n in sorted(by_year.items())},
        'holds':dict(holds),'requires_original_capture_time':False,
        'does_not_verify_original_pregame_availability':True,
        'spread_sign_assumption':'HOME_QUOTED_UNVERIFIED_BY_HISTORICAL_SOURCE',
        'aggregation':'ACTUAL_QUOTED_MEDIAN_LOW_NONCONFLICTING_PROVIDERS_NOT_EXECUTABLE',
        'source_scope':'HISTORICAL_RESEARCH_ONLY_NO_LINE_MOVEMENT_ORIGINAL_BET_ROI',
        'bet_authority':0}


def ml_attach_research_reference_labels(historical_rows, archive_rows):
    """Preserve timed labels; attach separate retrospective ML targets only.

    Timed labels remain `label_spread`/`label_totals`; untimed backfills use
    distinct `*_reference` labels and features. Never overwrite certified
    inputs or give archived prices a fabricated opening/current timestamp.
    """
    indexed={(int(r['season']),str(r['game_id'])):r for r in historical_rows}
    audit=Counter();by_season=Counter();source_summary=Counter()
    # A valid observed-pregame reference is also valid for the research lane.
    for r in historical_rows:
        for mk,old,ref,field,newfield in (
            ('spreads','label_spread','label_spread_reference','MARKET_SPREAD_LINE','MARKET_SPREAD_LINE_REFERENCE'),
            ('totals','label_totals','label_totals_reference','MARKET_TOTAL_LINE','MARKET_TOTAL_LINE_REFERENCE')):
            if r.get(old) in (0,1):
                feature_values=r.get('core_features',{})
                timestamp_verified=r.get(mk+'_source')=='VERIFIED_PREKICK_SCORES_WITH_FEATURES'
                line=_ml_number(feature_values.get(field))
                if not timestamp_verified and not math.isfinite(line):
                    # The optional legacy training view names its columns
                    # Opening_Spread/Opening_Total, but cannot establish when
                    # they were observed. Preserve them only as hypotheses.
                    oldname='Opening_Spread' if mk=='spreads' else 'Opening_Total'
                    line=_ml_number(feature_values.get(oldname))
                if not math.isfinite(line):
                    audit[mk+'_EXISTING_LABEL_MISSING_REFERENCE_LINE']+=1
                    continue
                r[ref]=int(r[old]);r['core_features'][newfield]=line
                if timestamp_verified:
                    r[mk+'_reference_source']='VERIFIED_PREKICK_SNAPSHOT'
                    source_summary[mk+'_TIMED']+=1
                else:
                    r[mk+'_reference_source']='LEGACY_NAMED_REFERENCE_UNTIMED'
                    source_summary[mk+'_EXISTING_UNTIMED']+=1
    for entry in archive_rows:
        key=(int(entry['season']),str(entry['game_id']))
        r=indexed.get(key)
        if r is None:
            audit['NO_CANONICAL_TRAINING_ROW']+=1;continue
        hs=_ml_number(r.get('home_score'));away=_ml_number(r.get('away_score'))
        if not all(math.isfinite(v) and v>=0 for v in (hs,away)):
            audit['NO_CANONICAL_SCORES']+=1;continue
        for mk,linefield,label,feature in (
            ('spreads','spread_home_line_research','label_spread_reference','MARKET_SPREAD_LINE_REFERENCE'),
            ('totals','total_over_line_research','label_totals_reference','MARKET_TOTAL_LINE_REFERENCE')):
            line=_ml_number(entry.get(linefield))
            if not math.isfinite(line):continue
            if (mk=='spreads' and not -70<=line<=70) or (mk=='totals' and not 1<=line<=150):
                audit[mk+'_INVALID_REFERENCE_RANGE']+=1;continue
            margin=hs+line-away if mk=='spreads' else hs+away-line
            if abs(margin)<1e-8:
                audit[mk+'_PUSH_EXCLUDED']+=1;continue
            label_value=int(margin>0)
            old=r.get(label)
            if old in (0,1):
                audit[mk+'_EXISTING_REFERENCE_PRESERVED']+=1
                if old!=label_value:audit[mk+'_ARCHIVE_DIFFERS_FROM_EXISTING_LABEL']+=1
                # Compare sign hypotheses against a timed quote when one
                # exists, but never infer proof of original archive timing.
                if mk=='spreads' and r.get('spreads_source')=='VERIFIED_PREKICK_SCORES_WITH_FEATURES':
                    observed=_ml_number(r.get('core_features',{}).get('MARKET_SPREAD_LINE'))
                    if math.isfinite(observed) and abs(line)>.01:
                        same=abs(observed-line);flip=abs(observed+line)
                        if abs(same-flip)<1e-9:
                            audit['SPREAD_SIGN_SNAPSHOT_COMPARISON_TIED']+=1
                        elif same<flip:
                            audit['SPREAD_SIGN_SNAPSHOT_FAVORS_HOME_CONVENTION']+=1
                        else:
                            audit['SPREAD_SIGN_SNAPSHOT_FAVORS_REVERSED_CONVENTION']+=1
                continue
            r[label]=label_value
            r['core_features'][feature]=line
            r[mk+'_reference_source']='UNTIMED_ARCHIVE_HYPOTHESIS'
            source_summary[mk+'_UNTIMED_ARCHIVE']+=1
            by_season[(int(r['season']),mk)]+=1
            if mk=='spreads':
                # We deliberately do NOT use this opposite assumption to pick
                # the better-looking model after seeing results.
                opposite=hs-line-away
                if opposite*margin<0:
                    audit['SPREAD_SIGN_REVERSAL_CHANGES_LABEL']+=1
    counts={key:sum(r.get(key) in (0,1) for r in historical_rows)
            for key in ('label_spread','label_totals','label_spread_reference','label_totals_reference')}
    return {'status':'DUAL_LANE_RETROSPECTIVE_LABELS_BUILT',
        'labels':counts,'backfilled_by_season_market':{f'{yr}_{mk}':n for (yr,mk),n in sorted(by_season.items())},
        'by_source':dict(source_summary),'holds':dict(audit),
        'timestamp_requirement':'ORIGINAL_TIME_REQUIRED_ONLY_FOR_TIME_DEPENDENT_CLAIMS',
        'spread_orientation':'HOME_QUOTE_HYPOTHESIS_UNVERIFIED; SIGN_SENSITIVITY_REPORTED',
        'timed_training_labels_preserved':True,'untimed_reference_is_not_executable':True,
        'bet_authority':0}


def ml_american_odds_to_decimal(price, *, verified_format=None):
    """Fail closed on ambiguous odds encoding. No guessing based on magnitude."""
    x=_ml_number(price)
    if not math.isfinite(x):return None
    if verified_format=='american':
        if x>=100:return 1+x/100
        if x<=-100:return 1+100/abs(x)
    if verified_format=='decimal' and 1.0<x<=100:return x
    return None


def ml_priced_ev_audit(predictions, market_rows, *, price_format=None):
    """Research-only EV audit at an originally timestamped bookmaker quote.

    Even a computed EV is not an actionable bet without verified quote/capture,
    bookmaker, canonical identity, side and market-time parity.
    """
    matched=0;verified=0;rows=[];holds=Counter()
    by_game={(int(r['season']),str(r['game_id'])):r for r in market_rows}
    for key,p in predictions.items():
        quote=by_game.get(key)
        if not quote:continue
        matched+=1
        if not quote.get('original_quote_capture_certified'):
            holds['ORIGINAL_QUOTE_CAPTURE_UNCERTIFIED']+=1;continue
        decimal=ml_american_odds_to_decimal(quote.get('home_ml_quote'),verified_format=price_format)
        if decimal is None:
            holds['MISSING_VERIFIED_ODDS_FORMAT_OR_PRICE']+=1;continue
        if not math.isfinite(float(p)) or not 0<=float(p)<=1:continue
        verified+=1
        rows.append({'season':key[0],'game_id':key[1],'prob_home':float(p),
                     'decimal_odds':decimal,'expected_value_per_unit':round(float(p)*decimal-1,6)})
    return {'status':'RESEARCH_PRICED_EV_VERIFIED' if verified else 'HOLD_NO_CERTIFIED_ORIGINALLY_CAPTURED_PRICE',
            'prediction_market_matches':matched,'certified_price_matches':verified,
            'positive_ev_quotes':sum(r['expected_value_per_unit']>0 for r in rows),
            'sample':rows[:20], 'holds':dict(holds),'bet_authority':0}


def fit_v55_independent_brain_models(events, historical_rows, canonical_team_map, *, min_train=240, return_predictions=False):
    """Separate per-family/market research learner; CORE never gates admission.

    Signal and system credibility are season-prior only. Historical grades are
    computed from original-candidate side and the verified market-specific label,
    never used as features for that same/earlier season. Frozen retrospective
    source vintage is NOT certified, so all resulting models are research-only.
    """
    families=('SYSTEM_MINER','BIG_AL','PATHI','PREDICTION_TRACKER')
    labels={'h2h':'label_h2h','spreads':'label_spread','totals':'label_totals'}
    history={(int(r['season']),str(r['game_id'])):r for r in historical_rows}
    grouped=defaultdict(list); seen=set();holds=Counter()
    for e in events:
        fam=str(e.get('brain') or '').upper()
        mk=str(e.get('market') or '').lower()
        if fam not in families or mk not in labels:continue
        try:key=(int(e['season']),str(e['cfbd_game_id']))
        except (TypeError,ValueError,KeyError):continue
        target=str(e.get('target') or '').strip()
        side_map=canonical_team_map.get(key,{})
        if mk=='totals':
            if target not in ('OVER','UNDER'):
                holds['TOTAL_TARGET_NOT_VERIFIED']+=1;continue
            sign=1 if target=='OVER' else -1
        elif target in side_map:
            v=str(side_map[target].get('is_home_verified')).lower()
            if v not in ('1','0','true','false','yes','no'):
                holds['SIDE_ROLE_UNVERIFIED']+=1;continue
            sign=1 if v in ('1','true','yes') else -1
        else:
            holds['CANONICAL_SIDE_NOT_MATCHED']+=1;continue
        system=str(e.get('system_id') or fam)
        uniq=(key,fam,mk,target,system)
        if uniq in seen:continue
        seen.add(uniq)
        grouped[(fam,mk,key)].append((sign,system))
    audits={};models={};origin_oof={}
    for fam in families:
      for mk,label in labels.items():
        candidates=sorted((key,signals) for (f,m,key),signals in grouped.items() if f==fam and m==mk)
        prior=defaultdict(lambda:[0,0]);training=[];prior_year=None
        graded=0;missing=0
        for (yr,gid),signals in candidates:
            r=history.get((yr,gid)); actual=None if r is None else r.get(label)
            if actual not in (0,1):
                missing+=1;continue
            # Build predictive features BEFORE any results from this season are
            # added. Year-wide pending results become priors only next season.
            if yr!=prior_year:
                if prior_year is not None:
                    for k,(n,w) in season_pending.items():
                        prior[k][0]+=n;prior[k][1]+=w
                season_pending=defaultdict(lambda:[0,0]);prior_year=yr
            signed=sum(sign for sign,system in signals)
            evidence=[]
            for sign,system in signals:
                n,w=prior[system]
                evidence.append(sign*((w+1)/(n+2)-.5))
                # The evaluation label is used only to update a *future* year.
                won=int(actual==(1 if sign==1 else 0))
                season_pending[system][0]+=1
                season_pending[system][1]+=won
            # Capped counts prevent multiple correlated rules from generating
            # arbitrarily large pseudo-confidence.
            features={'ORIGIN_NET':float(max(-5,min(5,signed))),
                      'ORIGIN_SYSTEM_COUNT':float(min(12,len({sys for _,sys in signals}))),
                      'ORIGIN_SEASON_PRIOR':sum(evidence)/len(evidence) if evidence else 0.,
                      'ORIGIN_FAMILY_PRESENT':1.}
            training.append({'season':yr,'game_id':gid,'core_features':features,label:int(actual)})
            graded+=1
        key=f'ORIGIN_{fam}_{mk}'
        cols=['ORIGIN_NET','ORIGIN_SYSTEM_COUNT','ORIGIN_SEASON_PRIOR','ORIGIN_FAMILY_PRESENT']
        if training:
            preds,audit,model=_ml_train_with_chronological_fallback(training,cols,label,min_train=min_train)
            audit={k:v for k,v in audit.items() if k!='features'}
            if model is not None:
                origin_oof[(fam,mk)]={(int(r['season']),str(r['game_id'])):float(preds[i])
                    for i,r in enumerate(training) if math.isfinite(float(preds[i]))}
            if model is not None:
                models[key]={'model':model,'features':cols,
                             'source':'RETROSPECTIVE_FROZEN_FAMILY_SEASON_PRIOR_RESEARCH_ONLY',
                             'prediction_target':{'h2h':'HOME_WIN','spreads':'HOME_COVER','totals':'OVER'}[mk]}
        else:
            audit={'status':'HOLD_NO_MARKET_SPECIFIC_VERIFIED_LABELS',
                   'training_games':0,'oof_games':0}
        audits[key]={**audit,'candidate_games':len(candidates),
                     'labeled_candidate_games':graded,'unlabeled_candidate_games':missing,
                     'core_required':False,'original_signal_capture_certified':False,
                     'bet_authority':0}
    result={'status':'INDEPENDENT_FAMILY_ML_EVALUATED_RESEARCH_ONLY',
            'models_trained':sorted(models),'families':audits,'holds':dict(holds),
            'bet_authority':0}
    return (result,models,origin_oof) if return_predictions else (result,models)


def fit_v55_retrospective_reference_lane(historical_rows, events, team_map, *, min_train=240):
    """Chronologically train multiseason spread/total models on untimed REFERENCES.

    A game outcome and untimed market reference can define an ATS/OVER label
    without original quote capture. The distinction from an originally
    available, executable betting opportunity is permanent and machine-readable.
    Models here are NEVER forward-authority models.
    """
    import numpy as np
    report={'status':'HOLD_NO_RETROSPECTIVE_REFERENCE_LABELS',
            'reference_timing_policy':'TIMESTAMPS_NOT_REQUIRED_FOR_RETROSPECTIVE_REFERENCE_GRADES',
            'requires_verified_capture_for_forward_price_ev':True,
            'home_spread_sign_certified':False,
            'model_scope':'EXPLORATORY_RECONSTRUCTED_OUTCOMES_NOT_ORIGINAL_BETS',
            'bet_authority':0,'trained_models':[],'model_audits':{},
            'combined_brain_input_coverage':{},'combined_brain_ablation':{},
            'matched_game_model_comparison':{}}
    models={};oo={};features={};metrics={}
    core_fields=sorted({k for r in historical_rows for k in r.get('core_features',{})
                        if k.startswith('CFBD_DIFF_CFBD_') and k.endswith('_Prior')})
    if not core_fields:
        # Compatible with the original prior-only legacy training view.
        core_fields=sorted({k for r in historical_rows for k in r.get('core_features',{})
                            if k in ML_CORE_FIELDS})
    stat_fields=[k for k in core_fields if ('_Season_' in k or '_Recent3_' in k or '_Recent5_' in k)
                 and ('_Off_' in k or '_Def_' in k)]
    if len(stat_fields)<2:stat_fields=[f for f in ML_STAT_FIELDS if any(f in r.get('core_features',{}) for r in historical_rows)]
    if not core_fields or len(stat_fields)<2:
        report['status']='HOLD_NO_INDEPENDENT_CORE_STAT_PRIOR_FEATURES'
        return report,models
    # Shallow row copies are sufficient: labels change only in this research
    # view; original (verified-pregame) labels are never altered.
    research_rows=[]
    for r in historical_rows:
        research_rows.append({**r,
            'label_spread':r.get('label_spread_reference'),
            'label_totals':r.get('label_totals_reference')})
    # Original capture vintage is intentionally NOT used to gate retrospective
    # system reconstructions. Their prior-season reliability remains leakage-safe
    # for inference *conditional on the frozen system selection*.
    report['independent_candidate_reference_audit']=ml_independent_candidate_audit(
        events,research_rows,team_map)
    family_report,family_models,family_oof=fit_v55_independent_brain_models(
        events,research_rows,team_map,min_train=min_train,return_predictions=True)
    report['independent_brain_reference_ml']=family_report
    for key,spec in family_models.items():
        new=key+'_reference'
        models[new]={**spec,'source':'RETROSPECTIVE_REFERENCE_ONLY',
                     'bet_authority':0,'forward_eligible':False}
        report['trained_models'].append(new)
    counts=_ml_brain_counts(events)
    for market,label,linefeature in (
        ('spreads','label_spread','MARKET_SPREAD_LINE_REFERENCE'),
        ('totals','label_totals','MARKET_TOTAL_LINE_REFERENCE')):
        ref_count=sum(r.get(label) in (0,1) for r in research_rows)
        report['model_audits'][market]={'labeled_games':ref_count,
            'line_source':'TIMED_SNAPSHOT_OR_UNTIMED_ARCHIVE_RESEARCH',
            'line_sign_certified':market!='spreads',
            'original_price_capture_certified':False}
        if ref_count<min_train:
            report['model_audits'][market]['status']='HOLD_INSUFFICIENT_REFERENCE_GAMES'
            continue
        for fam,base in (('CORE',core_fields),('STAT',stat_fields)):
            cols=list(base)+[linefeature]
            pred,audit,fit=_ml_train_with_chronological_fallback(research_rows,cols,label,min_train=min_train)
            name=f'{fam}_{market}_reference'
            report['model_audits'][name]={k:v for k,v in audit.items() if k!='features'}
            if fit is None:continue
            oo[name]=pred
            models[name]={'model':fit,'features':cols,
                'source':'UNTIMED_REFERENCE_OR_CAPTURED_REFERENCE_RESEARCH_ONLY',
                'prediction_target':'HOME_COVER' if market=='spreads' else 'OVER',
                'bet_authority':0,'forward_eligible':False,
                'spread_sign_assumed':market=='spreads'}
            report['trained_models'].append(name)
        ck=f'CORE_{market}_reference';sk=f'STAT_{market}_reference'
        if ck not in oo or sk not in oo:
            report['model_audits'][f'COMBINED_{market}_reference']={'status':'HOLD_BASE_OOF_MISSING'}
            continue
        meta=[]
        for i,r in enumerate(research_rows):
            if r.get(label) not in (0,1):continue
            a=oo[ck][i];b=oo[sk][i]
            if not math.isfinite(float(a)) or not math.isfinite(float(b)):continue
            y=int(r['season']);gid=str(r['game_id']);sides=team_map.get((y,gid),{})
            if market=='totals':
                hc=counts.get((y,gid,'OVER',market),Counter())
                ac=counts.get((y,gid,'UNDER',market),Counter())
            else:
                home=[t for t,v in sides.items() if str(v.get('is_home_verified'))=='1']
                away=[t for t,v in sides.items() if str(v.get('is_home_verified'))=='0']
                if len(home)!=1 or len(away)!=1:continue
                hc=counts.get((y,gid,home[0],market),Counter())
                ac=counts.get((y,gid,away[0],market),Counter())
            x={'CORE_OOF':float(a),'STAT_OOF':float(b),
               # A STATIC line is context, not a synthetic Market Rich OOF or
               # sharp movement signal. Preserve its provenance in the report.
               'MARKET_REFERENCE_LINE':_ml_number(r.get('core_features',{}).get(linefeature))}
            for family in ('SYSTEM_MINER','BIG_AL','PATHI','PREDICTION_TRACKER'):
                x[family+'_NET']=float(min(3,hc.get(family,0))-min(3,ac.get(family,0)))
                x[family+'_ORIGIN_OOF']=family_oof.get((family,market),{}).get((y,gid),float('nan'))
            meta.append({'season':y,'game_id':gid,'core_features':x,label:int(r[label])})
        if not meta:continue
        fields=tuple(meta[0]['core_features'])
        report['combined_brain_input_coverage'][market]=_ml_family_coverage(meta)
        pred,audit,fit=_ml_fitting_series(meta,fields,label,min_train=min_train)
        name=f'COMBINED_{market}_reference'
        report['model_audits'][name]={**{k:v for k,v in audit.items() if k!='features'},
            'status':'EXPLORATORY_RETROSPECTIVE_META_OOF' if fit is not None else audit['status'],
            'original_expert_capture_certified':False,
            'reference_line_original_capture_certified':False}
        if fit is not None:
            models[name]={'model':fit,'features':list(fields),
                'source':'RETROSPECTIVE_UNTIMED_OR_TIMED_REFERENCE_STACK',
                'prediction_target':'HOME_COVER' if market=='spreads' else 'OVER',
                'bet_authority':0,'forward_eligible':False}
            report['trained_models'].append(name)
            report['combined_brain_ablation'][market]=_ml_ablation_report(
                meta,fields,label,pred,min_train=min_train)
            # An archived reference line is not Market Rich ML. Test its
            # incremental context explicitly without relabeling as a full
            # price-movement brain. Same OOF game sample only.
            from sklearn.metrics import brier_score_loss, log_loss
            reduced=[f for f in fields if f!='MARKET_REFERENCE_LINE']
            without,_,_=_ml_fitting_series(meta,reduced,label,min_train=min_train)
            common=np.isfinite(pred)&np.isfinite(without)
            if common.any():
                yt=np.asarray([r[label] for r in meta],dtype=int)
                report['reference_line_context_ablation_'+market]={
                    'status':'EXPLORATORY_SAME_GAME_STATIC_LINE_CONTEXT',
                    'same_games':int(common.sum()),
                    'with_line_brier':round(float(brier_score_loss(yt[common],pred[common])),6),
                    'without_line_brier':round(float(brier_score_loss(yt[common],without[common])),6),
                    'with_line_logloss':round(float(log_loss(yt[common],pred[common],labels=[0,1])),6),
                    'without_line_logloss':round(float(log_loss(yt[common],without[common],labels=[0,1])),6),
                    'not_a_market_rich_ml_model':True,'bet_authority':0}
            mapped={(int(r['season']),str(r['game_id'])):i for i,r in enumerate(research_rows)}
            pred_map={name0:{k:float(p[mapped[k]]) for k in mapped if math.isfinite(float(p[mapped[k]]))}
                for name0,p in (('CORE',oo[ck]),('STAT',oo[sk]))}
            pred_map['COMBINED']={(int(r['season']),str(r['game_id'])):float(pred[i])
                for i,r in enumerate(meta) if math.isfinite(float(pred[i]))}
            report['matched_game_model_comparison'][market]=_ml_matched_probability_metrics(
                research_rows,pred_map,label)
    report['status']='EXPLORATORY_RETROSPECTIVE_MODELS_TRAINED' if report['trained_models'] else 'HOLD_REFERENCE_MODEL_FOLDS'
    return report,models


def fit_v55_ml_research(historical_rows,canonical_events,canonical_team_map,*,
                        market_rows=None,min_train=240):
    """Three separate research ML learners and one combined stack, when feasible.

    Never authorizes bets. Stage2 frozen observations are retrospective; combined
    model's OOF numbers are exploratory until per-event pregame capture is proved.
    """
    import numpy as np
    report={'schema':ML_SCHEMA,'status':'ML_RESEARCH_HOLD','bet_authority':0,
            'pre_game_evidence_certified':False,'trained_models':[],
            'coverage':{'historical_game_rows':len(historical_rows),'frozen_events':len(canonical_events),
                        'market_feature_rows':len(market_rows or [])},
            'warnings':['FROZEN_STAGE2_EVENTS_NOT_CERTIFIED_ORIGINAL_PREGAME_CAPTURE',
                        'FEATURE_VINTAGE_REQUIRES_SOURCE_VALIDATION',
                        'NO_AUTOMATIC_BET_AUTHORITY']}
    models={};oof={};valid_fields=set()
    report['label_coverage']={k:sum(r.get(k) in (0,1) for r in historical_rows)
        for k in ('label_spread','label_h2h','label_totals')}
    for r in historical_rows:
        valid_fields.update(k for k,v in r.get('core_features',{}).items() if math.isfinite(_ml_number(v)))
    # CORE and STAT are distinct views of historical pregame inputs. On
    # the verified CFBD fallback CORE uses every available prior difference;
    # STAT uses season-form and recent-form offensive/defensive statistical
    # subsets, never outcomes or the same game's measured performance.
    cfbd=sorted(f for f in valid_fields if f.startswith('CFBD_DIFF_CFBD_') and f.endswith('_Prior'))
    stat_cfbd=[f for f in cfbd if any(x in f for x in
        ('_Season_', '_Recent3_', '_Recent5_')) and any(x in f for x in
        ('_Off_', '_Def_'))]
    brain_fields=(('CORE',tuple(cfbd) if cfbd else ML_CORE_FIELDS),
                  ('STAT',tuple(stat_cfbd) if stat_cfbd else ML_STAT_FIELDS))
    for brain,fields in brain_fields:
        base_cols=[f for f in fields if f in valid_fields]
        if brain=='STAT' and len(base_cols)<2:
            report[brain]={'status':'HOLD_STAT_INDEPENDENT_FEATURES_UNAVAILABLE','features':base_cols};continue
        if not base_cols:
            report[brain]={'status':'HOLD_NO_PREGAME_TRAINING_FEATURES'};continue
        report[brain+'_feature_selection']={'features':base_cols,'feature_count':len(base_cols),
            'source':'CFBD_PRIOR_DIFFERENCES' if cfbd else 'LEGACY_CORE_VIEW'}
        for market,label in (('spreads','label_spread'),('h2h','label_h2h'),('totals','label_totals')):
            reference={'spreads':'MARKET_SPREAD_LINE','totals':'MARKET_TOTAL_LINE'}.get(market)
            cols=base_cols+([reference] if reference and reference in valid_fields else [])
            prediction,audit,fitted=_ml_train_with_chronological_fallback(historical_rows,cols,label,min_train=min_train)
            report[brain+'_'+market]=audit
            if fitted is not None:
                key=brain+'_'+market
                models[key]={'model':fitted,'features':cols,'source':'STAGE1_V5_2_HISTORICAL',
                             'prediction_target':{'spreads':'HOME_COVER','h2h':'HOME_WIN','totals':'OVER'}[market]}
                oof[key]=prediction
                report['trained_models'].append(key)
    report['chronological_probability_calibration']={}
    for name,pred in oof.items():
        market=name.split('_',1)[-1]
        label={'h2h':'label_h2h','spreads':'label_spread','totals':'label_totals'}.get(market)
        if not label:continue
        ca,cp,cm=_ml_chronological_calibration(historical_rows,pred,label)
        report['chronological_probability_calibration'][name]=ca
        if cm is not None:
            models['CAL_'+name]={'model':cm,'features':['UNCALIBRATED_MODEL_PROBABILITY'],
                'input_transform':'LOGIT_CLIPPED_1E-5','source':'SHADOW_EARLIER_SEASON_OOF_CALIBRATION',
                'prediction_target':models[name]['prediction_target'],
                'bet_authority':0}
    # Historical Market Rich fits on canonical, verified pre-kick warehouse
    # snapshots. Forward/live moves are not a research-training source.
    market_predictions={};market_source='scores_with_features'
    report['market_rich_chronology_audit']={'status':'HOLD_NO_HISTORICAL_MARKET_ROWS'}
    if market_rows:
        hist_index={(int(r['season']),str(r['game_id'])):r for r in historical_rows}
        mrows=[]
        for source in market_rows:
            base=hist_index.get((int(source['season']),str(source['game_id'])))
            if not base:continue
            mrows.append({**source,'date':base.get('date'),**{k:base.get(k) for k in ('label_spread','label_h2h','label_totals')}})
        # Preserve missingness when a canonical historical game had no market
        # record. Fold predictions are keyed to game, never row index.
        report['market_historical_labeled_games']=len(mrows)
        report['market_rich_chronology_audit']=_ml_market_coverage(mrows)
        cols=[f for f in ML_HIST_MARKET_FIELDS if any(math.isfinite(_ml_number(r.get('core_features',{}).get(f))) for r in mrows)]
        if cols and mrows:
            for market,label in (('spreads','label_spread'),('h2h','label_h2h'),('totals','label_totals')):
                pred,audit,fitted=_ml_train_with_chronological_fallback(mrows,cols,label,min_train=min_train)
                report['MARKET_RICH_'+market]=audit
                if fitted is not None:
                    tag='MARKET_RICH_'+market
                    models[tag]={'model':fitted,'features':cols,
                        'source':'HISTORICAL_SCORES_WITH_FEATURES_STAGE1_V52',
                        'prediction_target':{'spreads':'HOME_COVER','h2h':'HOME_WIN','totals':'OVER'}[market]}
                    report['trained_models'].append(tag)
                    market_predictions[market]={(int(r['season']),str(r['game_id'])):float(pred[i])
                         for i,r in enumerate(mrows) if math.isfinite(float(pred[i]))}
            report['MARKET_RICH']={'status':'HISTORICAL_MARKET_ML_EVALUATED',
                                   'labeled_games':len(mrows),'features':cols}
        else:
            report['MARKET_RICH']={'status':'HOLD_HISTORICAL_MARKET_FEATURES_OR_LABELS_MISSING',
                                   'matched_games':len(mrows)}
    else:
        report['MARKET_RICH']={'status':'HOLD_HISTORICAL_MARKET_SNAPSHOT_SOURCE_MISSING'}
    report['independent_brain_origination']=ml_independent_candidate_audit(
        canonical_events,historical_rows,canonical_team_map)
    origin_audit,origin_models,origin_predictions=fit_v55_independent_brain_models(
        canonical_events,historical_rows,canonical_team_map,min_train=min_train,
        return_predictions=True)
    report['independent_brain_ml']=origin_audit
    models.update(origin_models)
    report['trained_models'].extend(sorted(origin_models))
    # Meta learners: market-specific, using only same-year OOF base predictions.
    # Historical frozen signals are research-only until original prekick capture.
    counts=_ml_brain_counts(canonical_events)
    report['combined_brain_input_coverage']={}
    report['matched_game_model_comparison']={}
    report['combined_brain_ablation']={}
    for market,label in (('spreads','label_spread'),('h2h','label_h2h'),('totals','label_totals')):
        core_key='CORE_'+market;stat_key='STAT_'+market;outkey='COMBINED_'+market
        if core_key not in oof or stat_key not in oof:
            report[outkey]={'status':'HOLD_MISSING_CORE_OR_STAT_OOF'};continue
        meta_rows=[]
        for i,r in enumerate(historical_rows):
            if r.get(label) not in (0,1):continue
            cp=oof[core_key][i];sp=oof[stat_key][i]
            if not math.isfinite(cp) or not math.isfinite(sp):continue
            y=int(r['season']);gid=str(r['game_id'])
            sides=canonical_team_map.get((y,gid),{})
            if market=='totals':
                # Over and Under are not independent family votes. Count only
                # recorded OVER minus UNDER and retain zero when no signal.
                hc=counts.get((y,gid,'OVER',market),Counter())
                ac=counts.get((y,gid,'UNDER',market),Counter())
            else:
                home=[name for name,side in sides.items() if str(side.get('is_home_verified'))=='1']
                away=[name for name in sides if name not in home]
                if len(home)!=1 or len(away)!=1:continue
                hc=counts.get((y,gid,home[0],market),Counter())
                ac=counts.get((y,gid,away[0],market),Counter())
            extras={f'{fam}_NET':float(min(3,hc.get(fam,0))-min(3,ac.get(fam,0)))
                   for fam in ('SYSTEM_MINER','BIG_AL','PATHI','PREDICTION_TRACKER')}
            extras['CORE_OOF']=float(cp);extras['STAT_OOF']=float(sp)
            extras['MARKET_RICH_OOF']=market_predictions.get(market,{}).get((y,gid),float('nan'))
            for fam in ('SYSTEM_MINER','BIG_AL','PATHI','PREDICTION_TRACKER'):
                extras[fam+'_ORIGIN_OOF']=origin_predictions.get((fam,market),{}).get((y,gid),float('nan'))
            meta_rows.append({'season':y,'game_id':gid,'core_features':extras,label:r[label]})
        fields=('CORE_OOF','STAT_OOF','MARKET_RICH_OOF',
                'SYSTEM_MINER_NET','BIG_AL_NET','PATHI_NET','PREDICTION_TRACKER_NET',
                'SYSTEM_MINER_ORIGIN_OOF','BIG_AL_ORIGIN_OOF','PATHI_ORIGIN_OOF',
                'PREDICTION_TRACKER_ORIGIN_OOF')
        report['combined_brain_input_coverage'][market]=_ml_family_coverage(meta_rows)
        meta_pred,meta_audit,model=_ml_fitting_series(meta_rows,fields,label,min_train=min_train)
        meta_audit.update({'status':('EXPLORATORY_RETROSPECTIVE_META_OOF' if model is not None else
                                  'HOLD_META_INSUFFICIENT_CHRONOLOGY'),
                          'market_rich_historical_oof_games':len(market_predictions.get(market,{})),
                          'source_event_capture_verified':False})
        report[outkey]=meta_audit
        if model is not None:
            report['combined_brain_ablation'][market]=_ml_ablation_report(
                meta_rows,fields,label,meta_pred,min_train=min_train)
        else:
            report['combined_brain_ablation'][market]={'status':'HOLD_META_MODEL_UNAVAILABLE'}
        # Retain only the same out-of-fold game keys for a fair paired metric.
        base_rows={(int(r['season']),str(r['game_id'])):i for i,r in enumerate(historical_rows)}
        predictions={}
        for brain in ('CORE','STAT'):
            x=oof.get(brain+'_'+market)
            if x is not None:
                predictions[brain]={(y,g):float(x[i]) for (y,g),i in base_rows.items()
                                    if math.isfinite(float(x[i]))}
        predictions['COMBINED']={(int(r['season']),str(r['game_id'])):float(meta_pred[i])
                                 for i,r in enumerate(meta_rows) if math.isfinite(float(meta_pred[i]))}
        if market=='h2h':
            report['priced_ev_research_audit']=ml_priced_ev_audit(
                predictions.get('COMBINED',{}),market_rows or [])
        report['matched_game_model_comparison'][market]=_ml_matched_probability_metrics(
            historical_rows,predictions,label) if len(predictions)==3 else {'status':'HOLD_BASE_OOF_MISSING'}
        if model is not None:
            models[outkey]={'model':model,'features':list(fields),
                'source':'RETROSPECTIVE_STAGE2_RESEARCH_ONLY',
                'prediction_target':{'spreads':'HOME_COVER','h2h':'HOME_WIN','totals':'OVER'}[market]}
            report['trained_models'].append(outkey)
    report['status']='ML_RESEARCH_TRAINED_UNCERTIFIED' if models else 'HOLD_ML_TRAINING_DATA_OR_FEATURES'
    report['model_features']={k:v['features'] for k,v in models.items()}
    return report,models

def publish_v55_ml_artifacts(report,models,*,publish_uri,storage_client=None):
    """Version ML package atomically by hash; publish pointer last.

    This manifest is NOT compatible with old ROI-certificate authority.
    """
    import joblib,tempfile
    from urllib.parse import urlsplit
    if not publish_uri.startswith('gs://') or not publish_uri.endswith('.json'):
        raise ValueError('INVALID_ML_MANIFEST_URI')
    if storage_client is None:
        from google.cloud import storage
        storage_client=storage.Client()
    loc=urlsplit(publish_uri);bucket=storage_client.bucket(loc.netloc)
    root=loc.path.lstrip('/').rsplit('/',1)[0]
    names={}
    with tempfile.TemporaryDirectory(prefix='v55_ml_') as d:
        for name,model_spec in sorted(models.items()):
            path=Path(d)/(name+'.joblib')
            joblib.dump(model_spec,path)
            h=hashlib.sha256(path.read_bytes()).hexdigest()
            key=f'{root}/models/{name}-{h}.joblib'
            bucket.blob(key).upload_from_filename(str(path),content_type='application/octet-stream')
            names[name]={'uri':'gs://'+loc.netloc+'/'+key,'sha256':h,
                         'features':model_spec['features'],
                         'forward_eligible':bool(model_spec.get('forward_eligible',False)),
                         'research_reference_only':name.endswith('_reference')}
    manifest=dict(report,artifacts=names,
                  trained_utc=datetime.now(timezone.utc).isoformat(),
                  registry_policy='ML_SHADOW_ONLY_NO_BET_AUTHORITY')
    bucket.blob(loc.path.lstrip('/')).upload_from_string(json.dumps(manifest,sort_keys=True,
         default=str,allow_nan=False),content_type='application/json')
    return {'status':report['status'],'manifest_uri':publish_uri,'models':sorted(names),
            'bet_authority':0}

def run_v55_ml_training(*,client=None,storage_client=None,project='sharplogger',
                        dataset='sharp_data',bucket='sharp-models',
                        manifest_uri=None,canonical_uri=None,events_uri=None,market_feed_uri=None):
    """Heavy Research entry point; reuse ONLY existing Stage1/Stage2 artifacts."""
    # A single initialized BigQuery client must be passed to BOTH historical
    # loaders. Previously the Market Rich loader received None on Cloud Run,
    # deterministically causing a HOLD regardless of warehouse contents.
    if client is None:
        from google.cloud import bigquery
        client=bigquery.Client(project=project)
    prefix=(canonical_uri or ('gs://'+bucket+'/research/ncaaf/v55/canonical_stage1/v5_2/')).rstrip('/')+'/'
    try:
        raw={name:_obj_bytes(prefix+name,storage_client) for name in
             ('historical_cfbd_game_manifest_v5.csv','historical_cfbd_game_sides_v5.csv',
              'stage1_canonical_ledger_v5_audit.json')}
        audit,games,native,team_map,names=_checked_canonical(
             raw['historical_cfbd_game_manifest_v5.csv'],
             raw['historical_cfbd_game_sides_v5.csv'],raw['stage1_canonical_ledger_v5_audit.json'])
    except Exception as exc:
        return {'status':'HOLD_ML_CANONICAL_SOURCE','error':type(exc).__name__+':'+str(exc),
                'bet_authority':0}
    rows,source=ml_historical_training_rows(client=client,project=project,dataset=dataset,
             canonical_manifest=raw['historical_cfbd_game_manifest_v5.csv'],
             canonical_sides=raw['historical_cfbd_game_sides_v5.csv'],
             canonical_audit=raw['stage1_canonical_ledger_v5_audit.json'])
    event_uri=events_uri or ('gs://'+bucket+'/research/ncaaf/v55/two_engine_roi/frozen_brain_canonical_events_latest.jsonl')
    try:
        event_bytes=_obj_bytes(event_uri,storage_client)
        events=[json.loads(line) for line in event_bytes.decode().splitlines() if line.strip()]
        # Strictly inspect membership; no new matching algorithm.
        if any((int(x['season']),str(x['cfbd_game_id'])) not in games for x in events):
            raise ValueError('FROZEN_EVENT_HAS_UNKNOWN_STAGE1_GAME')
        event_status={'status':'EVENTS_LOADED','rows':len(events),'uri':event_uri}
    except Exception as exc:
        events=[];event_status={'status':'HOLD_EVENTS_UNAVAILABLE',
                               'error':type(exc).__name__+':'+str(exc)}
    # Historical ML MUST read scores_with_features. The 2026 moves feed is
    # retained by Market Rich coverage for forward scoring, never ML fitting.
    market_data,market_audit=ml_historical_market_rows(client=client,
          canonical_manifest=raw['historical_cfbd_game_manifest_v5.csv'],
          canonical_sides=raw['historical_cfbd_game_sides_v5.csv'],
          canonical_audit=raw['stage1_canonical_ledger_v5_audit.json'],
          project=project,dataset=dataset)
    reference_audit=_ml_attach_verified_market_labels(rows,market_data)
    archived_references, archived_reference_source_audit=ml_archived_reference_line_rows(
        client=client,canonical_manifest=raw['historical_cfbd_game_manifest_v5.csv'],
        canonical_sides=raw['historical_cfbd_game_sides_v5.csv'],
        canonical_audit=raw['stage1_canonical_ledger_v5_audit.json'],
        project=project,dataset=dataset)
    reference_label_audit=ml_attach_research_reference_labels(rows,archived_references)
    report,models=fit_v55_ml_research(rows,events,team_map,market_rows=market_data)
    retrospective_report,retrospective_models=fit_v55_retrospective_reference_lane(
        rows,events,team_map)
    # Full CFBD 2013-2025 margin/total targets are score-derived, not
    # restricted to the 2025 market snapshot / historical odds population.
    # V5.2 game IDs and prior-feature construction are the sole training input.
    full_report, full_models = fit_full_history(rows, events, team_map)
    models.update(full_models)
    report['full_history_score_research'] = full_report
    report['full_history_coverage_funnel'] = {
        'verified_v5_2_canonical_games':len(games),
        'cfbd_prior_training_games':len(rows),
        'score_target_eligible_games':full_report.get('coverage',{}).get('canonical_cfbd_training_games'),
        'primary_spread_label_games':sum(r.get('label_spread') in (0,1) for r in rows),
        'primary_totals_label_games':sum(r.get('label_totals') in (0,1) for r in rows),
        'historical_spread_reference_games':sum(r.get('label_spread_reference') in (0,1) for r in rows),
        'historical_totals_reference_games':sum(r.get('label_totals_reference') in (0,1) for r in rows),
        'market_snapshot_games':len(market_data),
        'frozen_evidence_rows':len(events),
        'cfbd_source_holds':source.get('holds',{}) if isinstance(source,dict) else {},
        'reference_only_is_not_bet_price_certification':True,
        'bet_authority':0}
    report['full_history_source_revision'] = {
        'research_py_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inlined_full_history_engine':'ncaaf_v55_research.py',
        'source_module_count':2}

    report['full_history_live_handoff'] = {
        'status':'SHADOW_UNTIL_ORIGINAL_LIVE_FEATURE_PARITY',
        'feature_contract':'EXACT_CFBD_PRIOR_DIFFERENCE_COLUMNS_AND_CANONICAL_ID',
        'seven_brains':'CORE_STAT_SYSTEM_MINER_PATHI_BIG_AL_PT_PLUS_LIVE_MARKET_RICH',
        'historical_market_quote_required_for_score_training':False,
        'market_price_required_for_live_value':True,
        'authority':0}
    report['trained_models'].extend(sorted(full_models))
    report.setdefault('model_features',{}).update({name:spec['features'] for name,spec in full_models.items()})
    # Reference model artifacts have a distinct name and forward_eligible=False.
    # No older untimed quote can change the certified-input label columns.
    models.update(retrospective_models)
    report['reference_research_schema']='RETROSPECTIVE_REFERENCE_V1_CONTEXTUAL_TIME_POLICY'
    report['retrospective_reference_research']=retrospective_report
    report['trained_models'].extend(retrospective_report.get('trained_models',[]))
    report.setdefault('model_features',{}).update({name:spec['features']
        for name,spec in retrospective_models.items()})
    report['contextual_timestamp_policy']={
        'retrospective_scores_reference':'ORIGINAL_CAPTURE_TIME_NOT_REQUIRED',
        'observed_market_movement':'PREKICK_ORDERED_TIMESTAMPS_REQUIRED',
        'historical_executable_betting_roi':'ORIGINAL_PRICE_AND_SIGNAL_AVAILABILITY_REQUIRED',
        'current_forward_bets':'CURRENT_VERIFIED_PREKICK_PRICES_REQUIRED',
        'missing_historical_time':'NOT_A_GENERAL_RESEARCH_VETO',
        'untimed_reference_models_forward_eligible':False}
    if report['status']=='HOLD_ML_TRAINING_DATA_OR_FEATURES' and retrospective_models:
        report['status']='EXPLORATORY_REFERENCE_MODELS_TRAINED_UNCERTIFIED' 
    report['archived_reference_line_source_audit']=archived_reference_source_audit
    report['retrospective_reference_label_audit']=reference_label_audit
    report['market_line_label_audit']=reference_audit
    report['archived_multiseason_quote_audit']=ml_archived_quote_vintage_audit(
        client=client,project=project,dataset=dataset)
    report['market_rich_historical_source_audit']=market_audit
    report['market_rich_data_roles']={
        'historical_training':{'table':f'{project}.{dataset}.scores_with_features',
                               'season_max':2025,
                               'coverage_audit':market_audit,
                               'allows_2026':False},
        'forward_live':{'table':f'{project}.{dataset}.moves_with_features_merged',
                        'excluded_from_historical_training':True,
                        'feed_uri':market_feed_uri or ('gs://'+bucket+'/research/ncaaf/v55/market_rich/pregame_events_latest.jsonl.gz')}}
    report['historical_source_audit']=source
    # Separately explain CFBD matrix availability, verified pre-kick labels,
    # and untimed retrospective references. Reporting a small primary lane
    # as missing CFBD would obscure the actual line-source limitation.
    by_season=defaultdict(Counter)
    for r in rows:
        season=str(r['season'])
        by_season[season]['cfbd_prior_games']+=1
        for market in ('spread','totals'):
            if r.get('label_'+market) in (0,1):
                by_season[season][market+'_timed_label_games']+=1
            if r.get('label_'+market+'_reference') in (0,1):
                by_season[season][market+'_reference_label_games']+=1
    report['training_coverage_waterfall_v2']={
        'status':'AUDITED_SEPARATE_PRIMARY_AND_RETROSPECTIVE_LABEL_LANES',
        'by_season':{y:dict(c) for y,c in sorted(by_season.items())},
        'cfbd_prior_games':len(rows),
        'verified_prekick_spread_labels':sum(r.get('label_spread') in (0,1) for r in rows),
        'verified_prekick_totals_labels':sum(r.get('label_totals') in (0,1) for r in rows),
        'retrospective_spread_reference_labels':sum(r.get('label_spread_reference') in (0,1) for r in rows),
        'retrospective_totals_reference_labels':sum(r.get('label_totals_reference') in (0,1) for r in rows),
        'spread_reference_sign_verified':False,
        'retrospective_reference_forward_eligible':False,
        'unavailable_optional_brains_do_not_veto_canonical_games':True,
        'source_identity_recovery':source.get('identity_recovery_policy'),
        'research_models_available':sorted(retrospective_models),
        'bet_authority':0}
    report['canonical_stage1_games']=len(games)
    report['canonical_event_audit']=event_status
    report['market_rich_training_source']=f'{project}.{dataset}.scores_with_features'
    report['market_rich_forward_source']=f'{project}.{dataset}.moves_with_features_merged'
    report['forward_market_feed_uri']=market_feed_uri or ('gs://'+bucket+'/research/ncaaf/v55/market_rich/pregame_events_latest.jsonl.gz')
    report['market_rich_trained']=any(k.startswith('MARKET_RICH_') for k in models)
    try:
        published=publish_v55_ml_artifacts(report,models,publish_uri=(manifest_uri or
             'gs://'+bucket+'/research/ncaaf/v55/ml/current_manifest.json'),
             storage_client=storage_client)
        return dict(report,publication=published)
    except Exception as exc:
        report['publication']={'status':'HOLD_ML_ARTIFACT_PUBLICATION_FAILED',
                               'error':type(exc).__name__+':'+str(exc)}
        return report


def integration_acceptance_report(canonical, warehouse, provider, market_rich=None):
    """Validate directional brains and Market Rich feature coverage separately."""
    bybrain=canonical.get('by_brain') or {}
    directional=sorted(BRAINS-{'MARKET_RICH'})
    missing=[f for f in directional if not bybrain.get(f)]
    price_count=int((warehouse or {}).get('priced_game_wagers') or 0)
    joined=int((provider or {}).get('join_and_grading_counts',{}).get('BRAIN_PROVIDER_MARKET_GRADES') or 0)
    unresolved={}
    if not canonical.get('canonical_games'):unresolved['canonical']='NO_CANONICAL_GAME_LEDGER'
    if not canonical.get('canonical_edge_research',{}).get('reference_line_graded_occurrences'):
        unresolved['brain_grading']='CANONICAL_BRAINS_NOT_GRADED'
    if missing:unresolved['missing_directional_families']=missing
    if joined==0:unresolved['archive_brain_join']='ZERO_BRAIN_PROVIDER_MARKET_GRADES'
    if price_count==0:unresolved['timestamped_price_join']='ZERO_MATCHED_ORIGINALLY_CAPTURED_SNAPSHOT_PRICES'
    if not canonical.get('source_receipts'):unresolved['lineage']='SOURCE_RECEIPTS_MISSING'
    market=market_rich or {}
    if market.get('status')!='MARKET_RICH_PREGAME_FEATURE_FEED_AVAILABLE':
        unresolved['market_rich_features']=market.get('status') or 'MARKET_RICH_FEATURE_AUDIT_NOT_RUN'
    return {'status':'PASS_DATA_INTEGRATION' if not unresolved else 'FAIL_DATA_INTEGRATION',
            'missing_brain_families':missing,'canonical_games':canonical.get('canonical_games',0),
            'brain_occurrences':sum(int(bybrain.get(f) or 0) for f in directional),
            'archive_brain_market_grades':joined,'originally_priced_market_wagers':price_count,
            'market_rich_features':market,
            'blocking_reasons':unresolved,'does_not_grant_bet_authority':True}


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

# BEGIN INLINED NCAAF V5.5 FULL-HISTORY ENGINE (no third-module dependency)
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import NormalDist
import hashlib, json, math
_V55_FH_FAMILIES=('SYSTEM_MINER','BIG_AL','PATHI','PREDICTION_TRACKER')
_V55_FH_TARGETS={'margin':'HOME_SCORE_MINUS_AWAY_SCORE','total':'HOME_SCORE_PLUS_AWAY_SCORE'}
_V55_FH_VERSION='NCAAF_V55_FULL_HISTORY_V1_RESEARCH_SHADOW'
def _v55_fh_finite(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else float('nan')
    except (TypeError, ValueError):return float('nan')

def _v55_fh_evidence_features(events, team_map):
    """Canonical, market-specific unique family evidence; no results in X."""
    counts=defaultdict(Counter);seen=set();unknown=Counter()
    for e in events:
        try:
            fam=str(e['brain']);mkt=str(e['market']);yr=int(e['season']);gid=str(e['cfbd_game_id']);side=str(e['target'])
        except (ValueError,TypeError,KeyError):continue
        if fam not in _V55_FH_FAMILIES or mkt not in ('spreads','h2h','totals'):continue
        sysid=str(e.get('system_id') or e.get('mechanism_id') or '').strip()
        token=(yr,gid,mkt,side,fam,sysid if sysid else hashlib.sha256(json.dumps(e,sort_keys=True,default=str).encode()).hexdigest())
        if token in seen:continue
        seen.add(token);counts[(yr,gid,mkt,side)][fam]+=1
    out={}
    for key,sides in team_map.items():
        if not isinstance(key,tuple) or len(key)!=2:continue
        h=[name for name,side in sides.items() if str(side.get('is_home_verified')).lower() in ('1','true','yes')]
        a=[name for name,side in sides.items() if str(side.get('is_home_verified')).lower() in ('0','false','no')]
        if len(h)!=1 or len(a)!=1:continue
        yr,gid=key; features={}
        for fam in _V55_FH_FAMILIES:
            for mkt in ('spreads','h2h','totals'):
                hside,aside=('OVER','UNDER') if mkt=='totals' else (h[0],a[0])
                hv=counts.get((yr,gid,mkt,hside),Counter()).get(fam,0)
                av=counts.get((yr,gid,mkt,aside),Counter()).get(fam,0)
                features[f'EVIDENCE_{mkt}_{fam}_NET']=float(min(3,hv)-min(3,av))
        out[key]=features
    return out

def _v55_fh_prepare_rows(historical_rows,events,team_map):
    ev=_v55_fh_evidence_features(events,team_map)
    rows=[];holds=Counter();years=Counter()
    for r in historical_rows:
        try:
            year=int(r['season']);key=(year,str(r['game_id']))
            hs=_v55_fh_finite(r['home_score']);aw=_v55_fh_finite(r['away_score'])
        except (ValueError,TypeError,KeyError):holds['INVALID_SCORE_OR_ID']+=1;continue
        if not (math.isfinite(hs) and math.isfinite(aw) and hs>=0 and aw>=0):
            holds['INVALID_SCORE_OR_ID']+=1;continue
        prior={k:v for k,v in (r.get('core_features') or {}).items()
               if k.startswith('CFBD_DIFF_') and k.endswith('_Prior')}
        if not prior or not any(math.isfinite(_v55_fh_finite(v)) for v in prior.values()):
            holds['NO_USABLE_PRIOR_FEATURES']+=1;continue
        item={'season':year,'game_id':key[1], 'features':{**prior,**ev.get(key,{})},
              'margin':hs-aw,'total':hs+aw, 'reference_spread':_v55_fh_finite((r.get('core_features') or {}).get('MARKET_SPREAD_LINE_REFERENCE')),
              'reference_total':_v55_fh_finite((r.get('core_features') or {}).get('MARKET_TOTAL_LINE_REFERENCE'))}
        rows.append(item);years[year]+=1
    rows.sort(key=lambda r:(r['season'],r['game_id']))
    audit={'canonical_cfbd_training_games':len(rows),'eligible_games_by_season':dict(sorted(years.items())),
           'holds':dict(holds),'historical_frozen_evidence_games':sum((int(r['season']),str(r['game_id'])) in ev for r in rows),
           'source':'V5_2_CANONICAL_CFBD_PRIORS_AND_FINAL_SCORES','brain_history_original_pregame_capture_certified':False,
           '2026_training_games':sum(r['season']>=2026 for r in rows)}
    return rows,audit

def fit_full_history(historical_rows,events,team_map,*,min_train=240,min_test=30):
    """Expanding-season score regression plus earlier-OOF residual calibration.

    OOF residuals for season Y never use outcomes from season Y or later.
    The published forward residual scale is learned only from historical OOF.
    """
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error,mean_squared_error
    rows,coverage=_v55_fh_prepare_rows(historical_rows,events,team_map)
    from collections import Counter
    coverage['expert_nonzero_net_by_market_family']={}
    for mkt in ('spreads','h2h','totals'):
        for fam in _V55_FH_FAMILIES:
            col='EVIDENCE_'+mkt+'_'+fam+'_NET'
            coverage['expert_nonzero_net_by_market_family'][mkt+'_'+fam]=sum(
                _v55_fh_finite(r['features'].get(col))!=0 for r in rows if col in r['features'])
    report={'schema':_V55_FH_VERSION,'status':'HOLD_FULL_HISTORY_INSUFFICIENT','coverage':coverage,
            'models':{},'same_game_comparison':{},'seven_brain_policy':'CORE_STAT_MINER_BIG_AL_PATHI_PT_PLUS_SEPARATE_LIVE_MARKET_RICH',
            'historical_brain_capture_certified':False,'historical_feature_vintage_certified':False,
            'bet_authority':0,'live_scoring_authority':0}
    fitted={}
    if len(rows)<min_train+min_test:return report,fitted
    prior=sorted({k for r in rows for k in r['features'] if k.startswith('CFBD_DIFF_')})
    stat=[k for k in prior if ('_Season_' in k or '_Recent3_' in k or '_Recent5_' in k) and ('_Off_' in k or '_Def_' in k)]
    if len(stat)<2:stat=prior[:]
    expert=sorted({k for r in rows for k in r['features'] if k.startswith('EVIDENCE_')})
    definitions={'CORE':prior,'STAT':stat,'COMBINED':prior+expert}
    years=np.asarray([r['season'] for r in rows],int);allpred={}
    for target in _V55_FH_TARGETS:
        y=np.asarray([r[target] for r in rows],float)
        for brain,cols in definitions.items():
            if not cols:continue
            X=np.asarray([[_v55_fh_finite(r['features'].get(k)) for k in cols] for r in rows],float)
            pred=np.full(len(rows),np.nan);cal=[];folds=[];earlier_errors=[]
            for season in sorted(set(years)):
                train=np.where(years<season)[0];test=np.where(years==season)[0]
                if len(train)<min_train or len(test)<min_test:continue
                model=HistGradientBoostingRegressor(max_iter=90,learning_rate=.05,max_leaf_nodes=9,
                     min_samples_leaf=55,l2_regularization=8.0,random_state=20261109)
                model.fit(X[train],y[train]);p=model.predict(X[test]);pred[test]=p
                sigma=float(np.std(earlier_errors,ddof=1)) if len(earlier_errors)>=min_train else None
                # A calibration fold exists only once prior OOF residuals exist.
                if sigma is not None and sigma>0:
                    z=(y[test]-p)/sigma
                    cal.append({'season':int(season),'test_games':len(test),'earlier_oof_residuals':len(earlier_errors),
                                'sigma_before_season':round(sigma,6),'empirical_68pct_interval_coverage':round(float(np.mean(abs(z)<=1)),5)})
                folds.append({'season':int(season),'train_games':len(train),'test_games':len(test)})
                earlier_errors.extend((y[test]-p).tolist())
            valid=np.isfinite(pred);name=f'FULL_{brain}_{target}'
            info={'target':_V55_FH_TARGETS[target],'feature_count':len(cols),'training_games':len(rows),
                  'oof_games':int(valid.sum()),'folds':folds,'chronological_calibration_folds':cal,
                  'status':'EXPANDING_SEASON_SCORE_OOF' if valid.any() else 'HOLD_NO_OOF',
                  'oof_mae':round(float(mean_absolute_error(y[valid],pred[valid])),6) if valid.any() else None,
                  'oof_rmse':round(float(mean_squared_error(y[valid],pred[valid])**.5),6) if valid.any() else None,
                  'forward_residual_sigma':round(float(np.std(earlier_errors,ddof=1)),6) if len(earlier_errors)>1 else None,
                  'calibration_method':'EARLIER_SEASON_OOF_RESIDUAL_SIGMA_NORMAL_APPROX_SHADOW',
                  'original_feature_vintage_certified':False,'bet_authority':0}
            report['models'][name]=info;allpred[name]=pred
            if valid.any() and len(earlier_errors)>1:
                final=HistGradientBoostingRegressor(max_iter=90,learning_rate=.05,max_leaf_nodes=9,
                     min_samples_leaf=55,l2_regularization=8.0,random_state=20261109)
                final.fit(X,y)
                fitted[name]={'model':final,'features':cols,'source':_V55_FH_VERSION,
                    'prediction_target':_V55_FH_TARGETS[target], 'residual_sigma':info['forward_residual_sigma'],
                    'forward_eligible':False,'bet_authority':0,'feature_vintage_certified':False,
                    'family_evidence_capture_certified':False}
        names=[f'FULL_{b}_{target}' for b in definitions]
        if all(n in allpred for n in names):
            mask=np.logical_and.reduce([np.isfinite(allpred[n]) for n in names]);paired=int(mask.sum())
            if paired:
                report['same_game_comparison'][target]={'games':paired,**{b+'_mae':round(float(mean_absolute_error(y[mask],allpred[f'FULL_{b}_{target}'][mask])),6) for b in definitions},
                    'fully_paired_out_of_season':True}
    report['status']='FULL_HISTORY_SCORE_MODELS_TRAINED_SHADOW' if fitted else 'HOLD_FULL_HISTORY_NO_FITTED_MODELS'
    return report,fitted

# END INLINED FULL-HISTORY TRAINING

if __name__=='__main__':raise SystemExit(main())
