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

def _stratum(r):
    # Exact sportsbook and exact spread/total. No -3 record is used for -6.
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
    p.add_argument('--observations',required=True,help='Original pregame captured historical betting events CSV or JSONL (local/gs://)')
    p.add_argument('--capture-audit',help='Independent original capture integrity attestation (local/gs://); missing audit yields HOLD')
    p.add_argument('--out',required=True,help='Local or gs:// research registry output')
    p.add_argument('--holdout-start',type=int,default=2024)
    args=p.parse_args(argv)
    try:
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
