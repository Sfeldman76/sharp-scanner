#!/usr/bin/env python3
"""NCAAF V5.5 authoritative LIVE BET engine (no other NCAAF Python imports).

Consumes one original live signal stream, one same-book two-sided price stream,
and the sealed ROI research registry. One authoritative output ledger. All 7
brains may originate. Research ML manifests are audited in shadow only; no
model receives betting authority without independently validated evidence.
This module does NOT fetch sports odds, produce CFBD features or place wagers.
Those upstream data services must feed canonical source records.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, os, sys, uuid
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

SCHEMA='NCAAF-V55-TWO-ENGINE-ROI-REGISTRY-V1'
POLICY='V55-UNIFIED-SEVEN-BRAIN-VALIDATED-BETTING-EDGE-20261106-R1'
POLICY_STATUS='OOS_VALIDATED_BETTING_EDGE_AUTHORITY'
BRAINS=frozenset(('CORE','STAT','SYSTEM_MINER','PATHI','BIG_AL','PREDICTION_TRACKER','MARKET_RICH'))

def sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False,default=str).encode()).hexdigest()
def num(x):
    try:
        x=float(x);return x if math.isfinite(x) else None
    except (TypeError,ValueError):return None

def clock(x):
    try:
        x=datetime.fromisoformat(str(x).replace('Z','+00:00'))
        return x.astimezone(timezone.utc) if x.tzinfo else None
    except (TypeError,ValueError,OverflowError):return None

def s(x):return str(x).strip() if x is not None else ''
def slug(x):return ''.join(c for c in s(x).lower() if c.isalnum())
def truth(x):return x is True or s(x).lower() in ('true','yes','1')
def family(x):
    f=s(x).upper();return {'PT':'PREDICTION_TRACKER','MINER':'SYSTEM_MINER','BIGAL':'BIG_AL'}.get(f,f)
def market(x):
    m=s(x).lower();return {'ats':'spreads','spread':'spreads','ml':'h2h','moneyline':'h2h','total':'totals'}.get(m,m)
def profit(american):
    o=num(american)
    return (o/100 if o>=100 else 100/-o) if o is not None and not (-100<o<100) else None

def t_bound(profits,alpha):
    # Same conservative one-sided betting-RETURN bound as research.
    # All-win samples do not get artificial zero uncertainty.
    n=len(profits)
    if n<2 or not 0<alpha<1:return None
    low=-1.0;high=max(0.0,max(profits))
    return sum(profits)/n-(high-low)*math.sqrt(math.log(1/alpha)/(2*n))

def checked_registry(doc,at):
    """Validate registry envelope, OOS proofs, complete component keys.

    Does not blindly trust a PASS label or convert shadow Miner observations
    into certified matches. Fail-closed on altered certificate signatures.
    """
    if (not isinstance(doc,dict) or doc.get('schema')!=SCHEMA or
        doc.get('status')!='PASS_CERTIFIED_ROI_REGISTRY' or
        doc.get('policy')!='HISTORICAL_BETTING_ROI_NOT_PREDICTIVE_PROBABILITY' or
        doc.get('certificates_sha256')!=sha(doc.get('certificates',[])) or
        not doc.get('capture_audit_sha256') or not doc.get('source_sha256') or
        not clock(doc.get('created_utc')) or clock(doc['created_utc'])>at):return []
    universe=int(doc.get('tested_hypotheses') or 0)
    if universe<1 or doc.get('invalid_rows'):return []
    found=[];seen=set()
    for c in doc.get('certificates',[]):
        try:
            keys=c['components'];deps=c['dependencies'];hashes=c['definition_hashes']
            if (c['status']!='VALIDATED_OOS_BETTING_ROI' or c['source_sha256']!=doc['source_sha256']
                or c['tested_hypotheses']!=universe or not keys or keys!=sorted(set(keys))
                or set(keys)!=set(deps) or set(keys)!=set(hashes)
                or len(set(deps.values()))!=len(deps) or c['id'] in seen):continue
            if c['market'] not in ('spreads','h2h','totals') or c['side'] not in ('HOME','AWAY','OVER','UNDER'):continue
            if c['market']=='totals' and c['side'] not in ('OVER','UNDER'):continue
            if c['market']!='totals' and c['side'] not in ('HOME','AWAY'):continue
            if any(k.split('|')[0] not in BRAINS or len(s(hashes[k]))!=64 for k in keys):continue
            if c['market']=='h2h' and c.get('line') is not None:continue
            if c['market']!='h2h' and num(c.get('line')) is None:continue
            w,l,p=(int(c[k]) for k in ('wins','losses','pushes'))
            n=int(c['oos_unique_games'])
            if n<100 or w+l+p!=n or num(c['realized_roi']) is None or c['realized_roi']<.02:continue
            if num(c['oos_roi_lower_adjusted']) is None or c['oos_roi_lower_adjusted']<=0:continue
            years=c['holdout_seasons']
            if not isinstance(years,list) or len(years)<2:continue
            if any(int(c['selection_end'][k])>=min(years) for k in keys):continue
            if any(not clock(c['freeze_utc'][k]) or clock(c['freeze_utc'][k])>=datetime(min(years),1,1,tzinfo=timezone.utc) for k in keys):continue
            if not s(c.get('book')) or profit(c.get('min_historical_odds')) is None:continue
            if c.get('confidence_method')!='ONE_SIDED_BONFERRONI_HOEFFDING_RETURN_BOUND':continue
            seen.add(c['id']);found.append(c)
        except (KeyError,ValueError,TypeError,OverflowError):continue
    return found

def audit_ml_research_manifest(doc):
    """Shadow-only handoff audit; never reinterpret an ML model as bet authority.

    Production predictions need the exact training feature schema, original
    pregame feature capture and independently evaluated executable prices.
    Merely loading a model manifest does not grant any of those properties.
    """
    if not isinstance(doc,dict) or doc.get('schema') not in ('NCAAF_V55_ML_RESEARCH_V1','NCAAF_V55_ML_RESEARCH_V2_HISTORICAL_MARKET','NCAAF_V55_ML_RESEARCH_V3_SPREAD_TOTAL_PREGAME_REFERENCES'):
        return {'status':'HOLD_ML_MANIFEST_SCHEMA','bet_authority':0}
    models=doc.get('artifacts') or {}
    invalid=[k for k,v in models.items() if not isinstance(v,dict) or
             not s(v.get('uri')).startswith('gs://') or len(s(v.get('sha256')))!=64 or
             not isinstance(v.get('features'),list)]
    return {'status':('ML_SHADOW_MANIFEST_PRESENT' if models and not invalid else
                      'HOLD_ML_ARTIFACTS_MISSING_OR_INVALID'),
            'models':sorted(models),'invalid_models':invalid,
            'manifest_policy':doc.get('registry_policy'),
            'live_feature_parity_verified':False,'bet_authority':0}

def row_quotes(board):
    """Translate existing V5.5 paired HOME/AWAY backend prices, not inferred odds.

    Other markets require their own actual same-book quote rows in --quotes.
    """
    out=[]
    for r in board:
        gid=s(r.get('Game_ID') or r.get('game_id'))
        kick=r.get('Game_Start_UTC') or r.get('kickoff_utc')
        q=r.get('Backend_Quote_Timestamp') or r.get('quote_observed_at_utc')
        book=s(r.get('Backend_Quote_Book') or r.get('book')).lower()
        state=s(r.get('Quote_State'))
        if not gid or not book or not state.startswith('FRESH_'):continue
        for side,prefix in (('HOME','Home'),('AWAY','Away')):
            line=num(r.get('Backend_'+prefix+'_Spread'))
            odds=num(r.get('Backend_'+prefix+'_Odds_American'))
            if line is None or odds is None:continue
            out.append(dict(game_id=gid,kickoff_utc=kick,market='spreads',side=side,
                            book=book,line=line,odds=odds,quote_observed_at_utc=q,
                            home_team=r.get('Home_Team') or r.get('Home_Team_Norm'),
                            away_team=r.get('Away_Team') or r.get('Away_Team_Norm'),
                            paired_source_verified=True,backend_published_quote_verified=True))
    return out

def quote_normal(raw,at,age_limit=15):
    gid=s(raw.get('game_id') or raw.get('Game_ID'))
    m=market(raw.get('market') or raw.get('Market'))
    side=s(raw.get('side') or raw.get('Side')).upper()
    kick=clock(raw.get('kickoff_utc') or raw.get('Game_Start_UTC'))
    quoted=clock(raw.get('quote_observed_at_utc') or raw.get('Snapshot_Timestamp'))
    book=s(raw.get('book') or raw.get('Bookmaker')).lower()
    line=num(raw.get('line') if 'line' in raw else raw.get('Value'))
    odds=num(raw.get('odds') if 'odds' in raw else raw.get('Odds_Price'))
    if (not gid or m not in ('spreads','h2h','totals') or not book or
        side not in (('OVER','UNDER') if m=='totals' else ('HOME','AWAY')) or
        not kick or not quoted or not quoted<=at<kick or quoted>=kick or
        (at-quoted).total_seconds()>age_limit*60 or profit(odds) is None or
        (m!='h2h' and (line is None or round(line*2)!=line*2)) or
        not truth(raw.get('paired_source_verified')) or
        not (truth(raw.get('backend_published_quote_verified')) or s(raw.get('quote_capture_ref')))):
        return None
    return dict(game_id=gid,market=m,side=side,book=book,line=(line if m!='h2h' else None),
                odds=odds,quote=quoted,kickoff=kick)

def _v55_signal_quote_sequence(event, observed, quoted, book_quote, at):
    """Two legitimate prospective sequences; NEVER backdate model decisions.

    Original signals may precede the recorded sportsbook snapshot. A frozen
    rule evaluated *on* an already recorded current quote naturally fires
    after that snapshot. The latter must have its own immutable first-seen
    capture and an exact quote-source reference, all within 15 minutes.
    """
    qclock=book_quote['quote']
    kickoff=book_quote['kickoff']
    if not (observed and quoted and observed<=at<kickoff and quoted<=qclock<kickoff):
        return False
    if observed<=quoted:
        return True
    return (truth(event.get('prospective_evaluation_after_quote'))
        and bool(s(event.get('signal_capture_ref')))
        and bool(s(event.get('quote_capture_ref')))
        and quoted<=observed and (observed-quoted).total_seconds()<=900)


def signals_by_side(events,boards,quotes,at):
    """Only original pregame, direction-verified events from seven brains.

    The upstream event producer may be CORE, STAT, PT, expert, Market Rich or
    the fresh 345-rule Miner executor. No signal is inferred from display text.
    """
    origins=defaultdict(dict);rejected=Counter()
    bmap={s(r.get('Game_ID') or r.get('game_id')):r for r in boards}
    for e in events:
        gid=s(e.get('game_id') or e.get('Game_ID'))
        f=family(e.get('source_family') or e.get('Source_Family'))
        sid=s(e.get('system_id') or e.get('Signal_Family') or e.get('Rule_ID'))
        dep=s(e.get('evidence_family_key') or e.get('Evidence_Family_Key')).upper()
        hash_id=s(e.get('frozen_definition_sha256') or e.get('Rule_Definition_SHA256')).lower()
        book=s(e.get('book') or e.get('Observed_Book')).lower()
        m=market(e.get('market') or e.get('Market') or 'spreads')
        board=bmap.get(gid,{})
        rawside=s(e.get('side') or e.get('Signal_Side')).upper()
        home=s(board.get('Home_Team') or board.get('Home_Team_Norm'))
        away=s(board.get('Away_Team') or board.get('Away_Team_Norm'))
        if not home or not away:
            matchup=s(board.get('Matchup'))
            if ' @ ' in matchup:away,home=matchup.split(' @ ',1)
        side=rawside if rawside in ('HOME','AWAY','OVER','UNDER') else ('HOME' if slug(rawside)==slug(home) else 'AWAY' if slug(rawside)==slug(away) else '')
        observed=clock(e.get('signal_observed_at_utc') or e.get('Observed_At_UTC'))
        quoted=clock(e.get('quote_observed_at_utc') or e.get('Observed_Quote_UTC'))
        line=num(e.get('line') if 'line' in e else e.get('Observed_Spread'))
        if (not gid or f not in BRAINS or not sid or not dep or not observed or not book or not quoted or
            not s(e.get('signal_capture_ref') or e.get('source_capture_ref') or e.get('original_capture_id')) or
            observed>at or not truth(e.get('direction_verified',e.get('Direction_Verified'))) or
            side not in (('OVER','UNDER') if m=='totals' else ('HOME','AWAY'))):
            rejected['INVALID_LIVE_EVENT']+=1;continue
        # A blank definition hash can originate research but NEVER match a
        # certified edge (certificate requires exact frozen component hash).
        component=f+'|'+sid
        for q in quotes:
            if (q['game_id']!=gid or q['market']!=m or q['side']!=side or
                book!=q['book'] or not _v55_signal_quote_sequence(e,observed,quoted,q,at) or
                (q['quote']-quoted).total_seconds()>900 or
                (m!='h2h' and (line is None or abs(line-q['line'])>1e-7))):
                continue
            key=(gid,m,side,q['book'],q['line'])
            prev=origins[key].get(component)
            item={'component':component,'dependency':dep,'definition_hash':hash_id,'origin':f}
            if prev is not None and (prev['dependency']!=dep or prev['definition_hash']!=hash_id):
                rejected['DUPLICATE_COMPONENT_DEFINITION_CONFLICT']+=1
                origins[key].pop(component,None)
                continue
            if prev is None:origins[key][component]=item
    return origins,dict(rejected)

def payoff(c,odds):
    net=profit(odds)
    if net is None or num(odds)<num(c.get('min_historical_odds')):return None
    n=c['oos_unique_games'];w,l,p=c['wins'],c['losses'],c['pushes']
    returns=([net]*w)+([-1.0]*l)+([0.0]*p)
    roi=sum(returns)/n
    lower=t_bound(returns,.05/max(1,c['tested_hypotheses']))
    return {'repriced_roi':roi,'repriced_lower_roi':lower,'n':n} if lower is not None else None

def evaluate(board,signals,registry,*,now=None,extra_quotes=None,max_bets=3):
    """Return one canonical backend board and audit for all markets/seven brains.

    A missing/incomplete registry is a research HOLD. A certified opposing side
    vetoes a bet. An exact combo certificate is required for STRONG BET.
    """
    if hasattr(board,'to_dict'):board=board.to_dict('records')
    if hasattr(signals,'to_dict'):signals=signals.to_dict('records')
    if hasattr(extra_quotes,'to_dict'):extra_quotes=extra_quotes.to_dict('records')
    board=list(board);signals=list(signals)
    at=clock(now) if now else datetime.now(timezone.utc)
    if not at:raise ValueError('INVALID_TIME')
    certs=checked_registry(registry,at)
    rawquotes=row_quotes(board)+(extra_quotes or [])
    quotes=[q for r in rawquotes if (q:=quote_normal(r,at))]
    # A spread's HOME (-6) and AWAY (+6) MUST be paired by absolute
    # magnitude; grouping by the signed line would silently lose the pair.
    grouped=defaultdict(dict)
    for q in quotes:
        grouping_line=(abs(q['line']) if q['market']=='spreads' else q['line'])
        key=(q['game_id'],q['market'],q['book'],grouping_line)
        old=grouped[key].get(q['side'])
        if old is None or q['quote']>old['quote']:
            grouped[key][q['side']]=q
    pairs=[];rejected_quotes=0
    for _,g in grouped.items():
        m=next(iter(g.values()))['market']
        a,b=('OVER','UNDER') if m=='totals' else ('HOME','AWAY')
        if a not in g or b not in g:
            rejected_quotes+=1;continue
        x,y=g[a],g[b]
        if x['kickoff']!=y['kickoff'] or abs((x['quote']-y['quote']).total_seconds())>15*60:
            rejected_quotes+=1;continue
        if m=='spreads' and abs(x['line']+y['line'])>1e-8:
            rejected_quotes+=1;continue
        if m=='totals' and abs(x['line']-y['line'])>1e-8:
            rejected_quotes+=1;continue
        pairs.extend([x,y])
    unique={(q['game_id'],q['market'],q['side'],q['book'],q['line']):q for q in pairs}
    pairs=list(unique.values())
    origins,rej_signals=signals_by_side(signals,board,pairs,at)
    by_game=defaultdict(list);candidates=defaultdict(list);match_errors=Counter()
    for q in pairs:
        k=(q['game_id'],q['market'],q['side'],q['book'],q['line'])
        obs=origins.get(k,{})
        if not obs:continue
        comps=set(obs)
        verified=[]
        for c in certs:
            if (c['market'],c['side'],c['book'])!=(q['market'],q['side'],q['book']):continue
            if c['market']!='h2h' and abs(c['line']-q['line'])>1e-7:continue
            keys=set(c['components'])
            if not keys.issubset(comps):continue
            if any(obs[s]['dependency']!=c['dependencies'][s] or obs[s]['definition_hash']!=c['definition_hashes'][s] for s in keys):
                match_errors['DEFINITION_OR_DEPENDENCY_DRIFT']+=1;continue
            p=payoff(c,q['odds'])
            if p is None or p['repriced_lower_roi']<=0 or p['repriced_roi']<.02:continue
            verified.append((c,p))
        best=max(verified,key=lambda x:(len(x[0]['components']),x[1]['repriced_lower_roi'])) if verified else None
        record={'game_id':q['game_id'],'market':q['market'],'side':q['side'],
                'origins':sorted({r['origin'] for r in obs.values()}),
                'system_ids':sorted(comps),'raw_signal_count':len(obs),
                'price_book':q['book'],'price_spread':q['line'],'price_odds':q['odds'],
                'quote_utc':q['quote'].isoformat(), 'qualified_independent_families':len(best[0]['components']) if best else 0,
                'qualified_family_keys':list(best[0]['components']) if best else [],
                'certificate_id':best[0]['id'] if best else None,
                'validated_roi':best[1]['repriced_roi'] if best else None,
                'validated_roi_lower':best[1]['repriced_lower_roi'] if best else None,
                'status':'VALIDATED_EDGE' if best else ('NO_VALIDATED_EDGE_REGISTRY' if not certs else 'NO_EXACT_OOS_EDGE_AT_THIS_LINE_BOOK_PRICE'),
                'bet_authority':0}
        by_game[(q['game_id'],q['market'])].append(record)
        candidates[q['game_id']].append(record)
    selected=[];blocked=0
    for _,group in by_game.items():
        valid=[r for r in group if r['status']=='VALIDATED_EDGE']
        sides={r['side'] for r in valid}
        # Either side of the spread/ML, or Over vs Under, vetoes both.
        if len(sides)>1:
            blocked+=len(valid)
            for r in valid:r['status']='CERTIFIED_OPPOSING_SIDE_VETO'
            continue
        if not valid:continue
        best=max(valid,key=lambda r:(r['validated_roi_lower'],r['validated_roi']))
        selected.append(best)
    selected.sort(key=lambda r:(r['validated_roi_lower'],r['validated_roi']),reverse=True)
    authorized=[];authorized_game_ids=set()
    authority_on=s(os.getenv('NCAAF_V55_BET_AUTHORITY_MODE','OFF')).upper()=='ON'
    for rec in selected:
        if not authority_on:
            rec['status']='BETTING_AUTHORITY_EMERGENCY_OFF';continue
        if rec['game_id'] in authorized_game_ids:
            rec['status']='ONE_AUTHORIZED_MARKET_PER_GAME';continue
        if len(authorized)>=max_bets:
            rec['status']='WAGER_CAP_EXCEEDED';continue
        authorized_game_ids.add(rec['game_id'])
        rec['bet_authority']=1;rec['status']='STRONG BET' if rec['qualified_independent_families']>=2 else 'BET'
        authorized.append(rec)
    out=[]
    for raw in board:
        r=dict(raw);gid=s(r.get('Game_ID') or r.get('game_id'))
        cands=candidates.get(gid,[])
        h=s(r.get('Home_Team') or r.get('Home_Team_Norm'))
        a=s(r.get('Away_Team') or r.get('Away_Team_Norm'))
        if not h or not a:
            if ' @ ' in s(r.get('Matchup')):a,h=s(r.get('Matchup')).split(' @ ',1)
        # Selected_Team is overwritten during two-sided scanning and cannot
        # establish CORE origin. Only the original pre-scan research field can.
        core=s(r.get('V55_Research_Selected_Team') or r.get('CORE_Selected_Team'))
        for c in cands:
            c['team']=(h if c['side']=='HOME' else a if c['side']=='AWAY' else c['side'])
            c['core_reference_team']=core
            c['opposes_core']=c['market'] in ('spreads','h2h') and slug(core) in (slug(h),slug(a)) and slug(c['team'])!=slug(core)
            c['hold']=None if c['bet_authority'] else c['status']
        chosen=next((x for x in authorized if x['game_id']==gid),None)
        r.update({'V55_Selected_Market':chosen['market'] if chosen else None,
                  'V55_Unified_Research_State':'LIVE_CANDIDATES_PRESENT' if cands else 'NO_LIVE_CANDIDATE',
                  'V55_Bet_Authority':int(bool(chosen)),'V55_Bet_Decision':chosen['status'] if chosen else ('CANDIDATE' if cands else 'PASS'),
                  'V55_Authority_State':POLICY_STATUS if chosen else None,
                  'V55_Authority_Policy_Id':POLICY if chosen else None,
                  'V55_Authority_Hold_Reason':None if chosen else ('NO_VALIDATED_EDGE_REGISTRY' if not certs else 'NO_MATCHED_VALIDATED_ROI_EDGE'),
                  'V55_Unified_Candidates_JSON':json.dumps(cands,sort_keys=True,allow_nan=False),
                  'V55_Unified_Brains_Seen':'|'.join(sorted({f for c in cands for f in c['origins']})),
                  'V55_Validated_Edge_Lower_ROI':chosen['validated_roi_lower'] if chosen else None,
                  'V55_Validated_Edge_ROI':chosen['validated_roi'] if chosen else None,
                  'V55_Validated_Edge_Certificate_ID':chosen['certificate_id'] if chosen else None,
                  'V55_Policy_EV':chosen['validated_roi'] if chosen else None,
                  'V55_Policy_Win_Prob':None,'V55_Policy_Push_Prob':None,'V55_Policy_Lose_Prob':None,
                  'Confirm_Count':chosen['qualified_independent_families'] if chosen else 0,
                  'Confirm_Families':'|'.join(chosen['qualified_family_keys']) if chosen else None,
                  'Oppose_Families':None, 'Oppose_Count':0})
        if chosen:
            r['Selected_Team']=chosen['team'];r['V55_Research_Selected_Side']=chosen['side']
            r['Backend_Quote_Book']=chosen['price_book'];r['Backend_Quote_Timestamp']=chosen['quote_utc']
            r['Current_Spread']=chosen['price_spread'];r['Odds_American']=chosen['price_odds']
            r['V55_Bet_Key']=sha((gid,chosen['market'],chosen['side']))[:28]
            r['V55_Qualified_Evidence_Detail_JSON']=json.dumps(chosen['qualified_family_keys'])
            r['V55_Independent_Support_Keys']='|'.join(chosen['qualified_family_keys'])
            r['V55_Active_System_Names']='|'.join(chosen['system_ids'])
        out.append(r)
    audit={'status':'COMPLETE_V55_INDEPENDENT_WEEKLY_SHADOW','production_v1_probabilities_used':False,
           'v55_authority_policy_id':POLICY,'v55_authority_policy_status':POLICY_STATUS,
           'v55_authority_probability_basis':'NOT_USED__VALIDATED_OUT_OF_SAMPLE_BETTING_RETURNS',
           'v55_bet_authority':int(bool(authorized)), 'v55_authorized_bet_rows':len(authorized),
           'v55_strong_bet_rows':sum(c['status']=='STRONG BET' for c in authorized),
           'v55_candidate_rows':sum(len(x) for x in candidates.values()),
           'v55_validated_edge_certificates':len(certs),'v55_unified_candidate_sides':sum(len(x) for x in candidates.values()),
           'v55_unified_live_research_by_brain':dict(Counter(f for x in candidates.values() for c in x for f in c['origins'])),
           'v55_decision_ladder':'ANY_BRAIN__EXACT_MARKET_BOOK_LINE_OOS_ROI__CONFLICT_VETO__PRICE',
           'v55_authority_evaluable':len(selected),'v55_conflict_veto_rows':blocked,
           'v55_rejected_quote_pairs':rejected_quotes,'v55_live_signal_rejections':rej_signals,
           'v55_definition_mismatch_holds':dict(match_errors),'v55_policy_status':'VALIDATED_EDGE_BETS_PUBLISHED' if authorized else 'NO_AUTHORIZED_BETS',
           'v55_automatic_wager_execution':False,'publication_status':'CURRENT_MARKET',
           'engine':'ncaaf_v55_backend.py','published_at_utc':at.isoformat()}
    return out,audit

def read_rows(path):
    if not path:raise ValueError('EMPTY_SOURCE_RECORDS_URI')
    if path.startswith('bq://'):
        import re
        name=path[5:]
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*',name):
            raise ValueError('INVALID_BQ_SOURCE_TABLE_ID')
        from google.cloud import bigquery
        return [dict(row.items()) for row in bigquery.Client(project='sharplogger').query(
            'SELECT * FROM `'+name+'` LIMIT 100000').result()]
    if path.startswith('gs://'):
        from google.cloud import storage
        b,p=path[5:].split('/',1);content=storage.Client().bucket(b).blob(p).download_as_bytes()
    else:content=Path(path).read_bytes()
    text=content.decode('utf-8-sig').strip()
    if path.lower().endswith('.csv'):return list(csv.DictReader(io.StringIO(text)))
    return json.loads(text) if text.startswith('[') else [json.loads(line) for line in text.splitlines() if line.strip()]

def read_registry(path):
    if not path:return None
    if path.startswith('gs://'):
        from google.cloud import storage
        b,p=path[5:].split('/',1);return json.loads(storage.Client().bucket(b).blob(p).download_as_text())
    return json.loads(Path(path).read_text())

def write_json(path,doc):
    content=json.dumps(doc,sort_keys=True,indent=2,allow_nan=False,default=str)+'\n'
    if path.startswith('gs://'):
        from google.cloud import storage
        b,p=path[5:].split('/',1);storage.Client().bucket(b).blob(p).upload_from_string(content,content_type='application/json')
    else:Path(path).write_text(content)

def publish_snapshot(rows,audit,*,client=None,table='sharplogger.sharp_data.ncaaf_v55_live_board'):
    """Exactly one DATA/COMPLETE snapshot. No UI-side recalculation.

    Existing reader's _verify_audit accepts this format for research-only rows;
    the authorized policy ID must match the installed reader before any BET.
    """
    body=json.dumps(rows,separators=(',',':'),allow_nan=False,default=str)
    digest=hashlib.sha256(body.encode()).hexdigest();snapshot=uuid.uuid4().hex
    created=datetime.now(timezone.utc).isoformat()
    records=[{'snapshot_id':snapshot,'published_at':created,'record_kind':'DATA',
              'payload_json':body,'audit_json':None,'row_count':len(rows),'payload_sha256':digest,'publication_type':'MARKET_REFRESH'},
             {'snapshot_id':snapshot,'published_at':created,'record_kind':'COMPLETE',
              'payload_json':None,'audit_json':json.dumps(audit,allow_nan=False,sort_keys=True),
              'row_count':len(rows),'payload_sha256':digest,'publication_type':'MARKET_REFRESH'}]
    if audit.get('v55_bet_authority') and audit.get('v55_authority_policy_id')!=POLICY:
        raise RuntimeError('INVALID_LIVE_AUTHORITY_POLICY')
    # Operational bet kill switch: fail-closed even in LIVE publication mode.
    # A true certification is not equivalent to authorization to place bets.
    authorized_switch=os.environ.get('NCAAF_V55_BET_AUTHORITY_MODE','OFF').upper()=='ON'
    if not authorized_switch and any(r.get('V55_Bet_Authority') for r in rows):
        raise RuntimeError('BET_AUTHORITY_KILL_SWITCH_OFF__WOULD_PUBLISH_BET')
    # Prevent an H2H/totals action from reaching an old spread-only UI.
    # Enable only AFTER ncaaf_v55_dashboard_route_v3.py (market-aware ROI
    # renderer) has been deployed and tested in the actual Streamlit app.
    non_spread=[r for r in rows if r.get('V55_Bet_Authority') and r.get('V55_Selected_Market')!='spreads']
    if non_spread and os.getenv('NCAAF_V55_MARKET_AWARE_UI_READY','OFF').upper()!='ON':
        raise RuntimeError('MARKET_AWARE_DASHBOARD_RENDERER_NOT_CONFIRMED')
    # Verify every authorized game has an exact matching published side event.
    for row in rows:
        if not row.get('V55_Bet_Authority'):continue
        chosen_market=market(row.get('V55_Selected_Market'))
        selected_side=s(row.get('V55_Research_Selected_Side')).upper()
        if (chosen_market not in ('spreads','h2h','totals') or
            selected_side not in (('OVER','UNDER') if chosen_market=='totals' else ('HOME','AWAY')) or
            s(row.get('V55_Authority_Policy_Id'))!=POLICY or
            s(row.get('V55_Bet_Decision')) not in ('BET','STRONG BET') or
            not s(row.get('V55_Validated_Edge_Certificate_ID'))):
            raise RuntimeError('AUTHORIZED_MARKET_SIDE_POLICY_MISMATCH')
        candidate_records=json.loads(row.get('V55_Unified_Candidates_JSON') or '[]')
        matches=[c for c in candidate_records if c.get('bet_authority') and
                 market(c.get('market'))==chosen_market and
                 s(c.get('side')).upper()==selected_side and
                 s(c.get('certificate_id'))==s(row.get('V55_Validated_Edge_Certificate_ID')) and
                 s(c.get('price_book')).lower()==s(row.get('Backend_Quote_Book')).lower()]
        if len(matches)!=1:raise RuntimeError('AUTHORIZED_BET_MISSING_CANONICAL_CANDIDATE')
        c=matches[0]
        if num(c.get('price_odds'))!=num(row.get('Odds_American')):
            raise RuntimeError('AUTHORIZED_BET_ODDS_MISMATCH')
        if chosen_market!='h2h' and num(c.get('price_spread'))!=num(row.get('Current_Spread')):
            raise RuntimeError('AUTHORIZED_BET_LINE_MISMATCH')
    from google.cloud import bigquery
    bq=client or bigquery.Client(project='sharplogger')
    errors=bq.insert_rows_json(table,[records[0]],row_ids=[snapshot+':DATA'])
    if errors:raise RuntimeError('BQ_DATA_WRITE_ERROR:'+str(errors)[:300])
    check=list(bq.query('SELECT payload_sha256 FROM `'+table+'` WHERE snapshot_id=@sid AND record_kind="DATA"',
        job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter('sid','STRING',snapshot)])).result())
    if len(check)!=1 or check[0].payload_sha256!=digest:raise RuntimeError('BQ_DATA_NOT_READABLE')
    errors=bq.insert_rows_json(table,[records[1]],row_ids=[snapshot+':COMPLETE'])
    if errors:raise RuntimeError('BQ_COMPLETE_WRITE_ERROR:'+str(errors)[:300])
    return {'snapshot_id':snapshot,'status':'PASS','rows':len(rows),'authorized':int(audit.get('v55_authorized_bet_rows',0))}

def canonicalize_live_signal_game_ids(events,board):
    """Use existing exact stable game IDs, never fuzzy names or final scores.

    The previous Stage1 crosswalk is historical and cannot make an unverified
    current producer game ID a CFBD ID. For live observations accept only an
    exact canonical Game_ID or an explicit matching named ID in both records.
    """
    aliases=('CFBD_Game_ID','cfbd_game_id','ESPN_Game_ID','espn_game_id',
             'Canonical_Game_ID','canonical_game_id','Source_Game_ID','source_game_id')
    b=list(board.to_dict('records') if hasattr(board,'to_dict') else board)
    index={};ambiguous=set();canonical=set();by_canonical={}
    for row in b:
        gid=s(row.get('Game_ID') or row.get('game_id'))
        if not gid:continue
        canonical.add(gid);by_canonical[gid]=row
        for field in aliases:
            v=s(row.get(field))
            if not v:continue
            key=(field,v)
            if key in index and index[key]!=gid:ambiguous.add(key)
            index[key]=gid
    result=[];holds=Counter();per_family=Counter();matched=Counter();games=defaultdict(set)
    for item in events:
        if not isinstance(item,dict):holds['BAD_SIGNAL_ROW']+=1;continue
        fam=family(item.get('source_family') or item.get('Source_Family'))
        per_family[fam]+=1
        gid=s(item.get('game_id') or item.get('Game_ID'))
        if gid not in canonical:
            available=set()
            for field in aliases:
                v=s(item.get(field))
                if not v:continue
                k=(field,v)
                if k in ambiguous:
                    holds['AMBIGUOUS_EXACT_ID_ALIAS']+=1;continue
                if k in index:available.add(index[k])
            if len(available)!=1:
                holds['NO_UNIQUE_EXACT_LIVE_GAME_ID']+=1
                continue
            gid=next(iter(available))
        src_kick=clock(item.get('kickoff_utc') or item.get('Game_Start_UTC'))
        board_kick=clock(by_canonical[gid].get('Game_Start_UTC') or by_canonical[gid].get('kickoff_utc'))
        if src_kick and board_kick and abs((src_kick-board_kick).total_seconds())>15*60:
            holds['CANONICAL_ID_KICKOFF_CONFLICT']+=1;continue
        copy=dict(item);copy['game_id']=gid;copy['Game_ID']=gid
        result.append(copy);matched[fam]+=1;games[fam].add(gid)
    audit={'all_seven_expected':sorted(BRAINS),
           'by_family_input':dict(per_family),
           'by_family_matched':{fam:matched[fam] for fam in sorted(BRAINS)},
           'by_family_distinct_games':{fam:len(games[fam]) for fam in sorted(BRAINS)},
           'unmatched_reasons':dict(holds),
           'missing_live_families':sorted(BRAINS-set(matched)),
           'canonical_board_games':len(canonical),
           'matched_signal_rows':len(result)}
    return result,audit


def _v55_live_id(value):
    """Normalize BQ INT64 IDs and blank pandas values without fuzzy aliases."""
    if value is None:return ''
    if isinstance(value,float) and not math.isfinite(value):return ''
    raw=s(value)
    if raw.lower() in ('nan','none','null','nat','<na>'):return ''
    if raw.endswith('.0') and raw[:-2].isdigit():return raw[:-2]
    return raw



# Scope-limited aliases for 2026 CFBD game-side prior-feature identity.
# These are verified SAME-SCHOOL representations; NO fuzzy/substring joins.
# Do not apply this mapping to sportsbook quote provenance or signal authority.
def _v55_cfbd_school_key(value):
    import unicodedata
    raw=unicodedata.normalize('NFKD',s(value).lower())
    key=''.join(ch for ch in raw if ch.isalnum() and not unicodedata.combining(ch))
    explicit={
        'olemississippi':'olemiss',
        'southernmississippi':'southernmiss',
        'northcarolinastate':'ncstate',
        'texasaandm':'texasam',
    }
    return explicit.get(key,key)

def _v55_board_fixture_identity(row):
    """Read verified board ID/ordered teams/kickoff, or exact 'AWAY @ HOME'.

    The existing published weekly board can contain Matchup without separate
    Home_Team/Away_Team. Never infer sides from 'vs', a selected team, or odds.
    """
    board_id=_v55_live_id(row.get('Game_ID') or row.get('game_id'))
    home=_v55_cfbd_school_key(row.get('Home_Team') or row.get('Home_Team_Norm'))
    away=_v55_cfbd_school_key(row.get('Away_Team') or row.get('Away_Team_Norm'))
    method='EXPLICIT_HOME_AWAY'
    if not home or not away:
        matchup=s(row.get('Matchup') or row.get('matchup'))
        if matchup.count('@')==1:
            raw_away,raw_home=(v.strip() for v in matchup.split('@',1))
            if raw_away and raw_home:
                parsed_home,parsed_away=_v55_cfbd_school_key(raw_home),_v55_cfbd_school_key(raw_away)
                if parsed_home and parsed_away and parsed_home!=parsed_away:
                    if (not home or home==parsed_home) and (not away or away==parsed_away):
                        home,away=parsed_home,parsed_away
                        method='EXACT_MATCHUP_AWAY_AT_HOME'
    kickoff_field=next((k for k in ('Game_Start_UTC','kickoff_utc','Kickoff_UTC','Commence_Time_UTC','Game_Date')
                        if row.get(k) is not None and s(row.get(k)) not in ('','nan','NaT','None')),None)
    kick=clock(row.get(kickoff_field)) if kickoff_field else None
    return board_id,home,away,kick,method,kickoff_field


def build_live_cfbd_feature_snapshots(board_rows, cfbd_rows, required_features, *, now=None):
    """Build live HOME-AWAY model inputs from 2026 CFBD prior-team rows.

    The existing CFBD miner table has Game_ID, Team_Key, Is_Home,
    Game_Date, Season, Week and *_Prior features, but NO capture timestamp.
    An observed prekick read produces a NEW shadow snapshot; this does NOT
    certify a vintage of a historical observation or permit bet authority.

    Identity: first exact CFBD game ID (if available), otherwise UNIQUE
    ordered home/away team keys + kickoff within 15 minutes. Never use
    approximate names or ambiguous fixture matches.
    """
    now=clock(now or datetime.now(timezone.utc))
    if now is None:raise ValueError('NOW_MUST_BE_TIMEZONE_AWARE')
    holds=Counter();out=[];seen_board={};groups=defaultdict(list)
    verified_placeholder_hints={}
    verified_placeholder_links=0;past_source_placeholder_links=0
    conflict_types=Counter();conflict_examples=[]
    board_methods=Counter();board_kickoff_fields=Counter()
    board_input_fields=set()
    priors=sorted({name[len('CFBD_DIFF_'):] for name in required_features
                   if name.startswith('CFBD_DIFF_') and name.endswith('_Prior')})
    for r in board_rows:
        if not isinstance(r,dict):holds['INVALID_BOARD_ROW']+=1;continue
        board_input_fields.update(r.keys())
        board_id,home,away,kick,method,kick_field=_v55_board_fixture_identity(r)
        if not board_id:holds['BOARD_MISSING_GAME_ID']+=1;continue
        if not home or not away or home==away:holds['BOARD_MISSING_HOME_AWAY_TEAM_IDENTITY']+=1;continue
        if not kick:holds['BOARD_MISSING_OR_UNZONED_KICKOFF']+=1;continue
        if kick<=now:holds['BOARD_KICKOFF_NOT_FUTURE']+=1;continue
        board_methods[method]+=1
        board_kickoff_fields[kick_field]+=1
        cfbd_id=_v55_live_id(r.get('CFBD_Game_ID') or r.get('cfbd_game_id'))
        # Board pricing may have replaced CFBD's unknown midnight kickoff
        # with an independently verified, two-sided sportsbook event time.
        # Preserve its ORIGINAL timestamp solely to link the same CFBD
        # game-side prior-feature rows; NEVER use the placeholder as kickoff.
        original=clock(r.get('V55_Original_CFBD_Kickoff_UTC'))
        market_verified=(s(r.get('V55_Kickoff_Verification')) ==
            'VERIFIED_TWO_SIDED_SPORTSBOOK_REPLACED_CFBD_MIDNIGHT_PLACEHOLDER')
        if (market_verified and original and original!=kick
                and abs((original-kick).total_seconds())<=26*3600
                and original.astimezone(ZoneInfo('America/New_York')).hour==0
                and original.astimezone(ZoneInfo('America/New_York')).minute==0):
            verified_placeholder_hints[board_id]=original
        identity=(home,away,kick,cfbd_id)
        if board_id in seen_board and seen_board[board_id]!=identity:
            holds['CONFLICTING_BOARD_GAME_ID']+=1;seen_board[board_id]=None
        elif board_id not in seen_board:seen_board[board_id]=identity
    for r in cfbd_rows:
        if not isinstance(r,dict):holds['INVALID_CFBD_SOURCE_ROW']+=1;continue
        gid=_v55_live_id(r.get('Game_ID') or r.get('game_id'))
        kick=clock(r.get('Game_Date') or r.get('game_date') or r.get('Game_Start_UTC'))
        season=num(r.get('Season') if 'Season' in r else r.get('season'))
        side=num(r.get('Is_Home') if 'Is_Home' in r else r.get('is_home'))
        name=_v55_cfbd_school_key(r.get('Team_Key') or r.get('team_key'))
        # A CFBD MIDNIGHT SENTINEL may precede the verified sportsbook
        # kickoff on the same date. Admit a bounded source lookback for
        # identity evaluation only; ordinary past games cannot match a
        # future verified board event without the explicit saved hint.
        if (not gid or not kick or kick<=now-timedelta(hours=27)
                or season!=now.year or side not in (0.0,1.0) or not name):
            holds['SOURCE_NOT_CURRENT_FUTURE_TWO_SIDE_RECORD']+=1;continue
        groups[gid].append(r)
    # Only complete, non-duplicated, correctly oriented CFBD fixtures can be
    # candidates. This also prevents one-side rows creating phantom opponents.
    fixtures={};by_teams=defaultdict(list)
    for gid,records in groups.items():
        if len(records)!=2:
            holds['CFBD_GAME_NOT_EXACTLY_TWO_ROWS']+=1;continue
        home_rows=[r for r in records if num(r.get('Is_Home') if 'Is_Home' in r else r.get('is_home'))==1.0]
        away_rows=[r for r in records if num(r.get('Is_Home') if 'Is_Home' in r else r.get('is_home'))==0.0]
        if len(home_rows)!=1 or len(away_rows)!=1:
            holds['CFBD_GAME_HAS_DUPLICATE_OR_MISSING_ORIENTATION']+=1;continue
        hr,ar=home_rows[0],away_rows[0]
        h=_v55_cfbd_school_key(hr.get('Team_Key') or hr.get('team_key'))
        a=_v55_cfbd_school_key(ar.get('Team_Key') or ar.get('team_key'))
        ht=clock(hr.get('Game_Date') or hr.get('game_date') or hr.get('Game_Start_UTC'))
        at=clock(ar.get('Game_Date') or ar.get('game_date') or ar.get('Game_Start_UTC'))
        if not ht or not at or abs((ht-at).total_seconds())>900 or h==a:
            holds['CFBD_GAME_ROLE_OR_DATE_CONFLICT']+=1;continue
        fixtures[gid]=(h,a,ht,hr,ar)
        by_teams[(h,a)].append(gid)
    for board_id,identity in seen_board.items():
        if identity is None:continue
        home,away,kick,cfbd_id=identity
        # The sportsbook board Game_ID and CFBD's numeric Game_ID need not
        # share a namespace. Prefer a true explicitly mapped CFBD_Game_ID;
        # otherwise fall back to exact two-team and kickoff identity.
        ids=[cfbd_id] if cfbd_id else ([board_id] if board_id in fixtures else [])
        if ids:
            candidates=ids;method='EXACT_CFBD_GAME_ID'
        else:
            candidates=[gid for gid in by_teams.get((home,away),[])
                        if abs((fixtures[gid][2]-kick).total_seconds())<=900]
            method='EXACT_TEAM_PAIR_AND_KICKOFF'
            # When IDs have different namespaces, recover only a UNIQUE
            # exact team pair with the original verified CFBD placeholder.
            if not candidates and board_id in verified_placeholder_hints:
                orig=verified_placeholder_hints[board_id]
                candidates=[gid for gid in by_teams.get((home,away),[])
                            if abs((fixtures[gid][2]-orig).total_seconds())<=900]
                method='EXACT_TEAMS_AND_ORIGINAL_CFBD_MIDNIGHT_PLACEHOLDER'
        if len(candidates)!=1:
            holds['NO_UNIQUE_FUTURE_CFBD_FIXTURE' if not candidates else 'AMBIGUOUS_FUTURE_CFBD_FIXTURE']+=1
            continue
        source_id=candidates[0]
        fixture=fixtures.get(source_id)
        if not fixture:
            holds['MAPPED_CFBD_ID_NOT_IN_FORWARD_FEATURE_TABLE']+=1;continue
        h,a,cfbd_kick,hr,ar=fixture
        source_kickoff_gap=abs((cfbd_kick-kick).total_seconds())
        if h!=home or a!=away:
            holds['CFBD_FIXTURE_TEAM_OR_KICKOFF_CONFLICT']+=1
            conflict_types['EXACT_CFBD_GAME_ID_HOME_AWAY_NAME_MISMATCH']+=1
            if len(conflict_examples)<12:conflict_examples.append({
                'board_game_id':board_id,'cfbd_game_id':source_id,
                'board_teams':away+' @ '+home,'source_teams':a+' @ '+h,
                'market_kickoff_utc':kick.isoformat(),
                'source_game_date_utc':cfbd_kick.isoformat(),
                'reason':'HOME_AWAY_NAME_MISMATCH'})
            continue
        if source_kickoff_gap>900:
            orig=verified_placeholder_hints.get(board_id)
            if orig and abs((cfbd_kick-orig).total_seconds())<=900:
                # Both teams and original CFBD timestamp match exactly;
                # actual event kickoff remains the VERIFIED market time.
                verified_placeholder_links+=1
                if cfbd_kick<=now:past_source_placeholder_links+=1
                method+='__VERIFIED_MARKET_KICKOFF_CFBD_PLACEHOLDER_LINK'
            else:
                holds['CFBD_FIXTURE_TEAM_OR_KICKOFF_CONFLICT']+=1
                conflict_types['KICKOFF_CONFLICT_NO_VERIFIED_ORIGINAL_MATCH']+=1
                if len(conflict_examples)<12:conflict_examples.append({
                    'board_game_id':board_id,'cfbd_game_id':source_id,
                    'board_teams':away+' @ '+home,
                    'market_kickoff_utc':kick.isoformat(),
                    'source_game_date_utc':cfbd_kick.isoformat(),
                    'original_cfbd_hint_utc':orig.isoformat() if orig else None,
                    'reason':'KICKOFF_CONFLICT_NO_VERIFIED_ORIGINAL_MATCH'})
                continue
        missing=[c for c in priors if c not in hr or c not in ar]
        if missing:
            holds['MISSING_REQUIRED_CFBD_PRIOR_COLUMNS']+=1;continue
        # A row with a true feature timestamp must still be pregame. A table
        # without capture timestamps is usable for NEW forward shadow reads
        # only; never backdate it or claim historic/vintage certification.
        supplied_times=[clock(r.get('feature_asof_utc') or r.get('Feature_AsOf_UTC')) for r in (hr,ar)]
        if any((r.get('feature_asof_utc') or r.get('Feature_AsOf_UTC')) for r in (hr,ar)):
            if not all(t is not None and t<=now and t<kick for t in supplied_times):
                holds['INVALID_EXPLICIT_FEATURE_CAPTURE_TIMESTAMP']+=1;continue
            snapshot_time=max(supplied_times)
            vintage_verified=True
        else:
            snapshot_time=now
            vintage_verified=False
        features={}
        for c in priors:
            hv=num(hr[c]);av=num(ar[c]);features['CFBD_DIFF_'+c]=(hv-av if hv is not None and av is not None else float('nan'))
        if not any(math.isfinite(v) for v in features.values()):
            holds['NO_NUMERIC_PRIOR_DIFFERENCES']+=1;continue
        out.append({'game_id':board_id,'cfbd_game_id':source_id,
                    'feature_asof_utc':snapshot_time.isoformat(),
                    'feature_read_at_utc':now.isoformat(),
                    'cfbd_game_date_utc':cfbd_kick.isoformat(),
                    'verified_sportsbook_kickoff_utc':kick.isoformat(),
                    'core_features':features,
                    # Preserve exact independently oriented team priors for the
                    # full-history Miner V5 (ML sees only aggregate differences).
                    'team_side_features':{
                        'HOME':{c:hr.get(c) for c in priors},
                        'AWAY':{c:ar.get(c) for c in priors}},
                    'source':'EXISTING_CFBD_MINER_PREKICK_PRIORS_READ_TIME_SHADOW',
                    'identity_match_method':method,
                    'feature_vintage_verified':vintage_verified,
                    'bet_authority':0})
    return out,{'status':'PREGAME_CFBD_FEATURES_BUILT_SHADOW' if out else 'HOLD_NO_VERIFIED_FORWARD_FEATURES',
                'board_games':len(seen_board),'board_rows_examined':len(board_rows),
                'board_identity_methods':dict(board_methods),
                'board_kickoff_fields':dict(board_kickoff_fields),
                'board_identity_fields_present':sorted(board_input_fields.intersection({
                    'Game_ID','game_id','Matchup','matchup','Home_Team','Away_Team',
                    'Home_Team_Norm','Away_Team_Norm','Game_Start_UTC','kickoff_utc',
                    'Kickoff_UTC','Commence_Time_UTC','Game_Date','CFBD_Game_ID'})),
                'source_rows':len(cfbd_rows),
                'future_source_fixture_groups':len(fixtures),
                'feature_games':len(out),'prior_columns_required':len(priors),
                'verified_market_placeholder_links':verified_placeholder_links,
                'past_cfbd_placeholder_date_links':past_source_placeholder_links,
                'cfbd_school_alias_version':'V55_CFBD_EXPLICIT_SCHOOL_ALIAS_20261110',
                'cfbd_conflict_types':dict(conflict_types),
                'cfbd_conflict_examples':conflict_examples,
                'capture_timestamp_available':any(r.get('feature_vintage_verified') for r in out),
                'read_time_only_snapshots':sum(not r['feature_vintage_verified'] for r in out),
                'identity_methods':dict(Counter(r['identity_match_method'] for r in out)),
                'holds':dict(holds),'bet_authority':0}


def read_live_cfbd_rows_from_bq(bq_client, table_id, required_features, *,now=None):
    """Read existing miner prior features only for future 2026 fixtures.

    Game_Date is a KICKOFF, not a feature capture. An un-timestamped source
    may generate a current shadow snapshot only while kickoff is in the
    future. Never use archive/postgame/research tables as a live source.
    """
    import re
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*',table_id or ''):
        raise ValueError('INVALID_LIVE_CFBD_TABLE_ID')
    if table_id.lower()!='sharplogger.sharp_data.ncaaf_cfbd_miner_features_v1':
        raise ValueError('LIVE_CFBD_TABLE_NOT_APPROVED_AS_FORWARD_PRIOR_SOURCE')
    table=bq_client.get_table(table_id)
    fields={field.name for field in table.schema}
    required={'Game_ID','Team_Key','Is_Home','Game_Date','Season','Week'}
    if not required.issubset(fields):
        raise ValueError('EXISTING_CFBD_SOURCE_MISSING_GAME_ROLE_OR_KICKOFF_FIELDS:'+str(sorted(required-fields)))
    priors={name[len('CFBD_DIFF_'):] for name in required_features
            if name.startswith('CFBD_DIFF_') and name.endswith('_Prior')}
    if not priors.issubset(fields):
        raise ValueError('LIVE_CFBD_SOURCE_MISSING_MODEL_PRIOR_FIELDS:'+str(sorted(priors-fields)[:12]))
    now=clock(now or datetime.now(timezone.utc))
    if not now:raise ValueError('NOW_MUST_BE_TIMEZONE_AWARE')
    optional=[c for c in ('feature_asof_utc','Feature_AsOf_UTC') if c in fields]
    wanted=['Game_ID','Team_Key','Is_Home','Game_Date','Season','Week']+optional+sorted(priors)
    sql=('SELECT '+','.join('`'+c+'`' for c in wanted)+f' FROM `{table_id}` '
         # Include a bounded earlier CFBD midnight sentinel so an actual
         # future market kickoff later the same day can be verified.
         'WHERE `Season` = @season AND `Game_Date` > TIMESTAMP_SUB(@now, INTERVAL 27 HOUR) '
         'AND `Game_Date` <= TIMESTAMP_ADD(@now, INTERVAL 45 DAY) LIMIT 10000')
    from google.cloud import bigquery
    config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter('season','INT64',now.year),
        bigquery.ScalarQueryParameter('now','TIMESTAMP',now)])
    return [dict(row.items()) for row in bq_client.query(sql,job_config=config).result()]


def read_original_live_signals_from_bq(bq_client,table_id,*,now=None):
    """Read recorded prekick directional events, not reconstructed picks."""
    import re
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*',table_id or ''):
        raise ValueError('INVALID_LIVE_SIGNAL_TABLE_ID')
    if any(word in table_id.lower() for word in ('postgame','archive','research')):
        raise ValueError('HISTORICAL_SIGNAL_TABLE_FORBIDDEN_AS_LIVE_INPUT')
    fields={field.name for field in bq_client.get_table(table_id).schema}
    required={'game_id','market','side','source_family','signal_observed_at_utc','kickoff_utc'}
    if not required.issubset(fields):
        raise ValueError('LIVE_SIGNAL_TABLE_MISSING_CAPTURE_SCHEMA:'+str(sorted(required-fields)))
    now=now or datetime.now(timezone.utc)
    sql=(f'SELECT * FROM `{table_id}` '
         'WHERE signal_observed_at_utc >= TIMESTAMP_SUB(@now, INTERVAL 14 DAY) '
         'AND signal_observed_at_utc <= @now '
         'AND kickoff_utc > @now LIMIT 10000')
    from google.cloud import bigquery
    config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter('now','TIMESTAMP',now)])
    return [dict(row.items()) for row in bq_client.query(sql,job_config=config).result()]



# Unified live signal ingestion: original observations ONLY. This is a source
# connector, not a seventh competing decision/prediction engine.
SEVEN_BRAIN_MANIFEST_SCHEMA='NCAAF_V55_SEVEN_BRAIN_LIVE_SOURCES_V1'

def read_seven_brain_signal_sources(manifest_uri, bq_client, *, now=None):
    """Read every configured brain's original directional observations.

    Individual brains may originate with no CORE selection. Missing families
    are disclosed but do not veto other legitimate origins. A source failure
    is a HOLD; historical replay/ML shadow documents are never interpreted as
    live observations. The source manifest must itself be explicitly supplied.
    """
    at=now or datetime.now(timezone.utc)
    manifest=read_registry(manifest_uri)
    if not isinstance(manifest,dict) or manifest.get('schema')!=SEVEN_BRAIN_MANIFEST_SCHEMA:
        raise ValueError('INVALID_SEVEN_BRAIN_SOURCE_MANIFEST_SCHEMA')
    sources=manifest.get('sources')
    if not isinstance(sources,list) or not sources or len(sources)>25:
        raise ValueError('INVALID_SEVEN_BRAIN_SOURCE_LIST')
    records=[];source_results=[];failures=[];dedup=set()
    by_family=Counter();seen_source=set()
    for j,entry in enumerate(sources):
        if not isinstance(entry,dict):raise ValueError('INVALID_SOURCE_ENTRY_'+str(j))
        fam=family(entry.get('family'))
        uri=s(entry.get('uri'))
        if fam not in BRAINS and fam!='ALL' or not uri.startswith(('gs://','bq://')) or uri in seen_source:
            raise ValueError('UNSAFE_OR_DUPLICATE_BRAIN_SOURCE_'+str(j))
        # Never allow a GCS research-only/replay product to masquerade as a
        # newly observed live prediction; only explicitly named live sources.
        uri_lower=uri.lower()
        if (any(t in uri_lower for t in ('historical','retrospective','postgame','archive',
                'frozen_brain_canonical_events','replay','ml/current_manifest'))
            or ('research/' in uri_lower and '/live/' not in uri_lower)):
            raise ValueError('RESEARCH_ONLY_SOURCE_FORBIDDEN_IN_LIVE_MANIFEST:'+fam)
        seen_source.add(uri)
        try:
            if uri.startswith('bq://'):
                rows=read_original_live_signals_from_bq(bq_client,uri[5:],now=at)
            else:
                rows=read_rows(uri)
            if not isinstance(rows,list):raise ValueError('SIGNALS_NOT_A_LIST')
            accepted=0;rejected=Counter()
            for e in rows:
                if not isinstance(e,dict):rejected['INVALID_ROW']+=1;continue
                reported=family(e.get('source_family') or e.get('Source_Family'))
                if reported not in BRAINS or (fam!='ALL' and reported!=fam):
                    rejected['UNEXPECTED_SOURCE_FAMILY']+=1;continue
                gid=s(e.get('game_id') or e.get('Game_ID'))
                sid=s(e.get('system_id') or e.get('Signal_Family') or e.get('Rule_ID'))
                evtime=clock(e.get('signal_observed_at_utc') or e.get('Observed_At_UTC'))
                kickoff=clock(e.get('kickoff_utc') or e.get('Game_Start_UTC'))
                quote_time=clock(e.get('quote_observed_at_utc') or e.get('Observed_Quote_UTC'))
                capture=s(e.get('signal_capture_ref') or e.get('source_capture_ref') or e.get('original_capture_id'))
                if not gid or not sid or not capture or not evtime or not kickoff or not quote_time:
                    rejected['MISSING_ORIGINAL_EVENT_PROVENANCE']+=1;continue
                producer_after_quote=(truth(e.get('prospective_evaluation_after_quote')) and
                    quote_time<=evtime<kickoff and (evtime-quote_time).total_seconds()<=900 and
                    bool(s(e.get('quote_capture_ref'))) and bool(s(e.get('signal_capture_ref'))))
                if not ((evtime<=quote_time<kickoff) or producer_after_quote) or quote_time>at or evtime>at:
                    rejected['NOT_VERIFIED_PREGAME_CHRONOLOGY']+=1;continue
                if not truth(e.get('direction_verified',e.get('Direction_Verified'))):
                    rejected['DIRECTION_UNVERIFIED']+=1;continue
                dep=s(e.get('evidence_family_key') or e.get('Evidence_Family_Key'))
                def_hash=s(e.get('frozen_definition_sha256') or e.get('Rule_Definition_SHA256')).lower()
                if not dep or len(def_hash)!=64 or any(c not in '0123456789abcdef' for c in def_hash):
                    rejected['MISSING_FROZEN_COMPONENT_IDENTITY']+=1;continue
                emarket=market(e.get('market') or e.get('Market') or 'spreads')
                eside=s(e.get('side') or e.get('Signal_Side')).upper()
                ebook=s(e.get('book') or e.get('Observed_Book'))
                eline=num(e.get('line') if 'line' in e else e.get('Observed_Spread'))
                if emarket not in ('spreads','h2h','totals') or not eside or not ebook or (emarket!='h2h' and eline is None):
                    rejected['MISSING_LIVE_MARKET_SIDE_BOOK_LINE']+=1;continue
                identity=(reported,gid,sid,s(e.get('market') or e.get('Market')),
                          s(e.get('side') or e.get('Signal_Side')),
                          s(e.get('book') or e.get('Observed_Book')).lower(),
                          s(e.get('line') if 'line' in e else e.get('Observed_Spread')),
                          evtime.isoformat(),quote_time.isoformat(),capture)
                if identity in dedup:rejected['DUPLICATE_OBSERVATION']+=1;continue
                dedup.add(identity);records.append(dict(e));accepted+=1;by_family[reported]+=1
            source_results.append({'family':fam,'uri':uri,'source_rows':len(rows),
                                   'accepted_original_events':accepted,'rejected':dict(rejected)})
        except Exception as exc:
            failures.append({'family':fam,'uri':uri,'error':type(exc).__name__+':'+str(exc)[:240]})
    return records,{'status':'PASS' if not failures else 'HOLD_SOURCE_ERRORS',
                    'schema':SEVEN_BRAIN_MANIFEST_SCHEMA,'source_results':source_results,
                    'source_failures':failures,'accepted_events':len(records),
                    'by_family':dict(by_family),
                    'families_without_current_live_events':sorted(BRAINS-set(by_family)),
                    'all_seven_brains_supported':True,
                    'original_capture_required_for_execution':True,
                    'retrospective_observations_not_reclassified_as_live':True}

# NCAAF V5.5: prospective producer -> unified authority handoff.
# Only the EXISTING board's frozen/qualified expert evaluator is reused.
# The retired NCAAF Research V2 Miner evaluator and the old V55 publisher are
# NEVER invoked from this bridge. Current V5 full-history Miner needs its
# separately published prospective rule firings (not the old RV22 votes).
V55_PROSPECTIVE_PRODUCER_TAG='V55_FROZEN_EXPERT_ORIGIN_CAPTURE_20261110_V1'


def _v55_producer_definition_hash(rule, family_name, system_id, registry_sha):
    """Exact deterministic frozen rule identity for prospective captures."""
    recipe={'registry_sha256':registry_sha,'family':family_name,'system_id':system_id,
        'source_signal_field':rule['signal'],'signal_type':rule['signal_type'],
        'ready_field':rule['ready_field'],'direction':rule['direction'],
        'dependency':rule['key'],'qualified_at':rule['qualified_at'].isoformat()}
    return sha(recipe)


def normalize_v55_producer_events(raw_events, board_rows, qualified_registry, *, now=None):
    """Adapt current frozen-rule firings to the existing 7-brain contract.

    Only actual pregame evaluation rows produced NOW are accepted. Match an
    exact paired sportsbook, canonical game ID, HOME/AWAY direction, current
    line and price. Do not transform archived Big Al/Pathi context into votes.
    """
    at=clock(now) if now is not None else datetime.now(timezone.utc)
    if at is None:raise ValueError('PRODUCER_NOW_NOT_UTC_AWARE')
    if hasattr(raw_events,'to_dict'):raw_events=raw_events.to_dict('records')
    if hasattr(board_rows,'to_dict'):board_rows=board_rows.to_dict('records')
    rules=(qualified_registry or {}).get('rules') or {}
    registry_sha=s((qualified_registry or {}).get('report_sha256'))
    out=[];holds=Counter();seen=set()
    for e in raw_events:
        fam=family(e.get('Source_Family'));sid=s(e.get('Signal_Family'))
        rule=rules.get((fam,sid))
        if fam not in ('STAT','PATHI','BIG_AL','PREDICTION_TRACKER') or not rule:
            holds['FROZEN_RULE_ID_NOT_VERIFIED']+=1;continue
        if s(e.get('Qualification_Status'))!='QUALIFIED_OOS' or s(e.get('OOS_Status'))!='PASS' or not truth(e.get('Direction_Verified')):
            holds['LIVE_QUALIFICATION_NOT_VERIFIED']+=1;continue
        if (s(e.get('Registry_SHA256'))!=registry_sha or
            s(e.get('Evidence_Family_Key'))!=s(rule['key']) or
            not registry_sha):
            holds['REGISTRY_OR_DEPENDENCY_MISMATCH']+=1;continue
        if s(e.get('Provenance_Status'))!='PRE_GAME_VERIFIED_LIVE_RECALCULATED':
            holds['NOT_DIRECT_FROZEN_RULE_EVALUATION']+=1;continue
        observed=clock(e.get('Observed_At_UTC'))
        source_quote=clock(e.get('Observed_Quote_UTC'))
        if not (observed and source_quote and
                rule['qualified_at']<=source_quote<=observed<=at and
                (observed-source_quote).total_seconds()<=900):
            holds['NO_CURRENT_ORIGINAL_QUOTE_PROVENANCE']+=1;continue
        gid=s(e.get('Game_ID'));direction=s(e.get('Signal_Side'))
        source_book=s(e.get('Observed_Book')).lower()
        options=[]
        for r in board_rows:
            if gid!=s(r.get('Game_ID') or r.get('game_id')):continue
            kick=clock(r.get('Game_Start_UTC'))
            quote=clock(r.get('Backend_Quote_Timestamp'))
            if not (kick and quote and observed<kick and source_quote<=quote<=at and
                    (at-quote).total_seconds()<=900 and
                    source_book==s(r.get('Backend_Quote_Book')).lower() and
                    s(r.get('Quote_State')).startswith('FRESH_')):continue
            matchup=s(r.get('Matchup'))
            if ' @ ' not in matchup:continue
            away,home=matchup.split(' @ ',1)
            # The old helper returns a named school, not HOME/AWAY. Require
            # exact normalized name or canonical quoted team identity.
            try:
                import ncaaf_v55_board as board_tools
                norm=board_tools._norm_team
            except ImportError:
                norm=slug
            t=norm(direction)
            side='HOME' if t==norm(home) else 'AWAY' if t==norm(away) else ''
            if not side:continue
            prefix='Home' if side=='HOME' else 'Away'
            line=num(r.get('Backend_'+prefix+'_Spread'))
            odds=num(r.get('Backend_'+prefix+'_Odds_American'))
            if line is None or profit(odds) is None:continue
            options.append((r,kick,quote,side,line,odds))
        if len(options)!=1:
            holds['NO_UNIQUE_FRESH_SAME_BOOK_SIDE_QUOTE']+=1;continue
        r,kick,quote,side,line,odds=options[0]
        if s(e.get('Game_Start_UTC')) and clock(e.get('Game_Start_UTC'))!=kick:
            holds['KICKOFF_CONFLICT']+=1;continue
        if s(e.get('Observed_Spread')):
            # The original observed spread belongs to the row that triggered
            # the rule. For FADE rules the actual *selected* side has the
            # opposite spread, so it must come from the paired board.
            orig_line=num(e.get('Observed_Spread'))
            if orig_line is None or min(abs(line-orig_line),abs(line+orig_line))>1e-6:
                holds['QUOTE_LINE_NOT_PAIRED']+=1;continue
        component_hash=_v55_producer_definition_hash(rule,fam,sid,registry_sha)
        identity=(gid,side,source_book,line,fam,sid,component_hash,source_quote.isoformat())
        if identity in seen:
            holds['DUPLICATE_RULE_FIRE']+=1;continue
        seen.add(identity)
        # A stable capture ID identifies this run's prospective observation;
        # the actual GCS object URI is added BEFORE persistence below.
        capture_id=sha((V55_PROSPECTIVE_PRODUCER_TAG,identity,observed.isoformat()))
        quote_ref='bq://sharplogger.sharp_data.sharp_moves_master#'+sha(
            (gid,source_book,side,line,source_quote.isoformat()))
        out.append({'game_id':gid,'market':'spreads','side':side,
            'source_family':fam,'system_id':sid,
            'evidence_family_key':s(rule['key']),
            'frozen_definition_sha256':component_hash,
            'book':source_book,'line':line,'odds':odds,
            'kickoff_utc':kick.isoformat(),
            'signal_observed_at_utc':observed.isoformat(),
            'quote_observed_at_utc':source_quote.isoformat(),
            'direction_verified':True,
            'prospective_evaluation_after_quote':True,
            'quote_capture_ref':quote_ref,
            'signal_capture_ref':None,
            'capture_id':capture_id,
            'rule_registry_sha256':registry_sha,
            'producer':V55_PROSPECTIVE_PRODUCER_TAG,
            'original_event_created_prospectively':True,
            'historical_betting_roi_certified':False})
    return out,{'status':'PROSPECTIVE_RULE_FIRINGS_PREPARED' if out else 'NO_VERIFIED_PROSPECTIVE_RULE_FIRINGS',
                'prepared':len(out),'by_brain':dict(Counter(e['source_family'] for e in out)),
                'holds':dict(holds),'legacy_miner_used':False,
                'research_replays_promoted':False}


def persist_v55_prospective_events(events, storage_client, bucket, *, now=None):
    """Write one immutable append-only capture before authority may read it."""
    if not events:return [],{'status':'NO_EVENTS_TO_CAPTURE','captured':0}
    at=clock(now) if now is not None else datetime.now(timezone.utc)
    if at is None:raise ValueError('CAPTURE_NOW_NOT_AWARE')
    name='research/ncaaf/v55/live/prospective_events/'+at.strftime('%Y%m%d')+'/'+at.strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex+'.jsonl'
    uri='gs://'+bucket+'/'+name
    captured=[]
    for row in events:
        item=dict(row)
        item['signal_capture_ref']=uri+'#'+s(item['capture_id'])
        captured.append(item)
    payload=('\n'.join(json.dumps(e,sort_keys=True,allow_nan=False) for e in captured)+'\n').encode()
    blob=storage_client.bucket(bucket).blob(name)
    blob.upload_from_string(payload,content_type='application/x-ndjson',if_generation_match=0)
    return captured,{'status':'IMMUTABLE_PROSPECTIVE_CAPTURED','captured':len(captured),
                     'uri':uri,'sha256':hashlib.sha256(payload).hexdigest(),
                     'legacy_miner_used':False,'historical_certification_claimed':False}


# V5.5 shared qualification contract (research publisher -> live backend).
V55_QUALIFICATION_SCHEMA='NCAAF_V55_FROZEN_RULE_QUALIFICATION_AUDIT_V2'
V55_QUALIFICATION_POLICY='V55-FROZEN-INDEPENDENT-BRAINS-2013-2025-R1'

# Shared V5.5 small-sample OOS screening contract. This is a research
# eligibility test, NOT an independently multiplicity-adjusted ROI certificate.
V55_OOS_MIN_DECISIONS = 12
V55_OOS_MIN_DECISIONS_PER_SEASON = 4
V55_OOS_POSTERIOR_MIN_PROB = 0.80
V55_OOS_REFERENCE_BASELINE = 0.50
V55_OOS_SCREEN_POLICY = 'V55-2024-2025-BETA11-POSTERIOR-80PCT-VS-50PCT-R2'


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





def validate_v55_current_qualification(doc, raw_sha='', *, now=None):
    """Validate exact V5.5 research schema WITHOUT consulting legacy V2 registry.

    Original evidence proof fields and OOS results must be present in each
    accepted system. The issuer's status alone NEVER grants an individual rule.
    """
    import re
    at=clock(now) if now is not None else datetime.now(timezone.utc)
    rejects=Counter();rules={}
    if not isinstance(doc,dict) or doc.get('schema')!=V55_QUALIFICATION_SCHEMA:
        return None,'QUALIFICATION_SCHEMA_MISMATCH'
    if doc.get('policy')!=V55_QUALIFICATION_POLICY or doc.get('uses_2026_outcomes') is not False:
        return None,'QUALIFICATION_POLICY_OR_LEAKAGE_MISMATCH'
    if doc.get('status')!='FROZEN_OOS_QUALIFIED' or doc.get('operator_approved') is not True:
        return None,'NO_INDEPENDENTLY_APPROVED_FROZEN_OOS_RULES:'+s(doc.get('status'))
    created=clock(doc.get('created_utc'))
    if not created or not at or created>at:
        return None,'INVALID_QUALIFICATION_PUBLICATION_TIME'
    if doc.get('discovery_last_season')!=2023 or doc.get('confirmation_seasons')!=[2024,2025]:
        return None,'UNSEALED_HOLDOUT_CONTRACT'
    for entry in doc.get('systems') or []:
        if not isinstance(entry,dict):rejects['BAD_RULE']+=1;continue
        fam=family(entry.get('family'));sid=s(entry.get('system_id'))
        key=(fam,sid);recipe=entry.get('live_recipe') or {}
        if not sid or fam not in BRAINS or key in rules:
            rejects['DUPLICATE_OR_INVALID_RULE_ID']+=1;continue
        if not (entry.get('oos_pass') is True and entry.get('pregame_feature_vintage_pass') is True
            and entry.get('split_disjoint') is True and entry.get('source_neutral') is True
            and entry.get('overlap_audit')=='PASS' and entry.get('multiple_testing_audit')=='PASS'
            and entry.get('uses_2026_outcomes') is False
            and entry.get('independent_preholdout_freeze_ref')
            and entry.get('source_pinned_sha256')
            and not entry.get('holds')):
            rejects['EVIDENCE_ATTESTATION_INCOMPLETE']+=1;continue
        frozen=clock(entry.get('definition_frozen_at_utc'))
        if not frozen or frozen>=datetime(2024,1,1,tzinfo=timezone.utc):
            rejects['RULE_NOT_FROZEN_BEFORE_HOLDOUT']+=1;continue
        try:
            n=int(entry.get('confirmation_n')); rate=float(entry.get('confirmation_rate'))
            season=entry.get('by_season') or {}
            a,b=season['2024'],season['2025']
            wins=int(entry.get('confirmation_wins'))
            losses=int(entry.get('confirmation_losses'))
            decisive=wins+losses
            season24=int(a['wins'])+int(a['losses'])
            season25=int(b['wins'])+int(b['losses'])
            if (decisive<=0 or n!=int(a['n'])+int(b['n']) or decisive!=season24+season25
                or abs(rate-wins/decisive)>0.000002 or
                entry.get('oos_statistical_screen',{}).get('policy')!=V55_OOS_SCREEN_POLICY):
                raise ValueError('OOS_COUNTS_OR_POLICY')
            checked=v55_oos_statistical_screen(wins,losses,season24,season25)
            recorded=entry['oos_statistical_screen']
            if (not checked['pass'] or recorded.get('pass') is not True or
                recorded.get('posterior_prob')!=checked.get('posterior_prob')):
                raise ValueError('OOS_SCREEN_FAILED')
        except (ValueError,KeyError,TypeError):
            rejects['OOS_RESULTS_INVALID']+=1;continue
        if not (recipe.get('role')=='directional' and recipe.get('market')=='spreads'
            and recipe.get('direction') in ('PLAY_ON','FADE') and recipe.get('feature_parity_ref')):
            rejects['LIVE_RECIPE_UNVERIFIED']+=1;continue
        if fam=='SYSTEM_MINER':
            conditions=recipe.get('conditions')
            if not isinstance(conditions,list) or not conditions or any(
               not isinstance(c,dict) or c.get('op') not in ('==','!=','<=','<','>=','>','in')
               or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,127}',s(c.get('field')))
               or 'value' not in c for c in conditions):
                rejects['MINER_CONDITION_GRAMMAR_UNSUPPORTED']+=1;continue
        else:
            f=s(recipe.get('signal_field'));t=s(recipe.get('signal_type'))
            expected={'STAT':'NCAAF_Stat_','PATHI':'Pathi_','BIG_AL':'BigAl_'}
            if (not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,127}',f) or t not in ('FLAG','SIDE')
                or (fam in expected and not f.startswith(expected[fam]))
                or (fam=='PREDICTION_TRACKER' and t!='SIDE')):
                rejects['EXPERT_LIVE_FIELD_UNSUPPORTED']+=1;continue
        rules[key]={'family':fam,'id':sid,'signal':recipe.get('signal_field'),
            'signal_type':recipe.get('signal_type'),'direction':recipe['direction'],
            'ready_field':s(recipe.get('ready_field')),'conditions':recipe.get('conditions'),
            'key':s(entry.get('evidence_family_key') or fam),'qualified_at':created,
            'n':n,'rate':rate,'frozen_hash':s(entry.get('source_pinned_sha256')),
            'feature_parity_ref':recipe['feature_parity_ref']}
    return {'rules':rules,'created_utc':created,'report_sha256':raw_sha,
        'source_seasons':'2013-2025','rejected_by_reason':dict(rejects),
        'miner_rules':sum(k[0]=='SYSTEM_MINER' for k in rules)},None


def _v55_exact_condition(features, condition):
    """Exact, declared predicates only. Missing inputs never match."""
    field=condition['field'];op=condition['op'];wanted=condition['value']
    if field not in features or features[field] is None:return False
    val=features[field]
    if op=='in':
        return isinstance(wanted,(list,tuple)) and val in wanted
    if op in ('==','!='):
        return (val==wanted) if op=='==' else (val!=wanted)
    try:
        x=float(val);y=float(wanted)
        if not math.isfinite(x) or not math.isfinite(y):return False
        return {'<':lambda:x<y,'<=':lambda:x<=y,'>':lambda:x>y,'>=':lambda:x>=y}[op]()
    except (ValueError,TypeError,KeyError):return False


def produce_v55_full_history_miner(board, feature_rows, qualified_registry, *,
                                    storage_client, bucket, now=None):
    """Execute independently attested V5 frozen Miner conditions on live priors.

    The actual feature record MUST identify the team side and pass feature
    vintage verification. Game-level ML diff features alone are insufficient.
    Never invoke retired RV22 votes, infer missing conditions, or flip sides.
    """
    at=clock(now) if now is not None else datetime.now(timezone.utc)
    rules={k:r for k,r in (qualified_registry or {}).get('rules',{}).items()
           if k[0]=='SYSTEM_MINER'}
    holds=Counter();prepared=[];seen=set()
    audit={'status':'NO_QUALIFIED_V5_MINER_RULES','frozen_rules':len(rules),
        'evaluated_side_features':0,'fired':0,'holds':{},'legacy_v2_used':False,
        'bet_authority':0}
    if not rules:return [],audit
    if hasattr(board,'to_dict'):board=board.to_dict('records')
    byid={s(r.get('Game_ID') or r.get('game_id')):r for r in board}
    oriented=[]
    for item in feature_rows:
        if not isinstance(item,dict):holds['INVALID_FEATURE_RECORD']+=1;continue
        sides=item.get('team_side_features')
        if isinstance(sides,dict):
            for side in ('HOME','AWAY'):
                if isinstance(sides.get(side),dict):
                    oriented.append(dict(item,team_side=side,features=sides[side],
                                         core_features=sides[side]))
        else:
            oriented.append(item)
    for item in oriented:
        gid=s(item.get('game_id') or item.get('Game_ID'))
        side=s(item.get('team_side') or item.get('Team_Side')).upper()
        if side not in ('HOME','AWAY'):
            holds['NO_TEAM_SIDE_FEATURE_VIEW']+=1;continue
        if item.get('feature_vintage_verified') is not True:
            holds['PREGAME_FEATURE_VINTAGE_NOT_VERIFIED']+=1;continue
        feats=item.get('core_features') or item.get('features') or {}
        if not isinstance(feats,dict):holds['FEATURES_NOT_A_DICTIONARY']+=1;continue
        row=byid.get(gid)
        if not row:holds['NO_MATCHED_BOARD_GAME']+=1;continue
        quote=clock(row.get('Backend_Quote_Timestamp'))
        kickoff=clock(row.get('Game_Start_UTC'))
        book=s(row.get('Backend_Quote_Book')).lower()
        if not (quote and kickoff and quote<=at<kickoff and
                (at-quote).total_seconds()<=900 and book and
                s(row.get('Quote_State')).startswith('FRESH_')):
            holds['NO_CURRENT_BOOK_PRICE']+=1;continue
        prefix='Home' if side=='HOME' else 'Away'
        line=num(row.get('Backend_'+prefix+'_Spread'))
        odds=num(row.get('Backend_'+prefix+'_Odds_American'))
        if line is None or profit(odds) is None:
            holds['SIDE_HAS_NO_EXECUTABLE_SPREAD_AND_ODDS']+=1;continue
        audit['evaluated_side_features']+=1
        for (fam,sid),rule in rules.items():
            conditions=rule.get('conditions') or []
            if not conditions or any(c['field'] not in feats for c in conditions):
                holds['FROZEN_RULE_FEATURES_NOT_AVAILABLE']+=1;continue
            if not all(_v55_exact_condition(feats,c) for c in conditions):continue
            target=side if rule['direction']=='PLAY_ON' else ('AWAY' if side=='HOME' else 'HOME')
            selected='Home' if target=='HOME' else 'Away'
            selected_line=num(row.get('Backend_'+selected+'_Spread'))
            selected_odds=num(row.get('Backend_'+selected+'_Odds_American'))
            if selected_line is None or profit(selected_odds) is None:
                holds['FADE_SELECTED_SIDE_HAS_NO_PRICE']+=1;continue
            identity=(gid,target,book,selected_line,fam,sid,quote.isoformat())
            if identity in seen:holds['DUPLICATE_FIRING']+=1;continue
            seen.add(identity)
            definition=sha((rule['frozen_hash'],sid,conditions,rule['direction']))
            capture_id=sha(('V55_MINER_FROZEN_V5',identity,at.isoformat()))
            prepared.append({'game_id':gid,'market':'spreads','side':target,
                'source_family':'SYSTEM_MINER','system_id':sid,
                'evidence_family_key':rule['key'],'frozen_definition_sha256':definition,
                'book':book,'line':selected_line,'odds':selected_odds,
                'kickoff_utc':kickoff.isoformat(),
                'signal_observed_at_utc':at.isoformat(),
                'quote_observed_at_utc':quote.isoformat(),
                'direction_verified':True,'prospective_evaluation_after_quote':True,
                'quote_capture_ref':'bq://sharplogger.sharp_data.sharp_moves_master#'+sha(identity),
                'signal_capture_ref':None,'capture_id':capture_id,
                'original_event_created_prospectively':True,
                'historical_betting_roi_certified':False,
                'source_frozen_miner_v5':True})
    audit['fired']=len(prepared);audit['holds']=dict(holds)
    if not prepared:
        audit['status']='NO_VERIFIABLE_V5_MINER_FIRINGS';return [],audit
    try:
        captured,cap=persist_v55_prospective_events(prepared,storage_client,bucket,now=at)
        audit.update(status='PASS_V5_MINER_CAPTURED',capture=cap)
        return captured,audit
    except Exception as exc:
        audit.update(status='HOLD_MINER_CAPTURE_WRITE_FAILED',error=type(exc).__name__+':'+str(exc)[:160])
        return [],audit

def produce_v55_verified_expert_signals(board, book_quotes, *, storage_client,
                                         bucket, now=None, log_func=None):
    """Current genuine frozen-rule producer. Never executes old Miner/V1 authority."""
    from ncaaf_v55_board import (_v55_direct_current_quote_rows,
        derive_v55_other_qualified_events)
    import pandas as pd
    at=clock(now) if now is not None else datetime.now(timezone.utc)
    log=log_func or (lambda *_:None)
    audit={'status':'NOT_STARTED','producer':V55_PROSPECTIVE_PRODUCER_TAG,
        'families_eligible_for_production':['STAT','PATHI','BIG_AL','PREDICTION_TRACKER'],
        'full_history_miner_status':'REQUIRES_V5_FROZEN_RULE_LIVE_FIRING_PRODUCER_NOT_LEGACY_V2',
        'core_status':'ML_RESEARCH_SHADOW_NOT_AUTOMATIC_ORIGINAL_SELECTION',
        'market_rich_status':'DESCRIPTIVE_MARKET_FEATURES_NOT_AUTOMATIC_DIRECTIONAL_RULES',
        'source_capture_before_authority':True,'bet_authority':0}
    # Read the CURRENT V5.5 qualification output, not the October 27
    # ncaaf-research-v2.33.0 placeholder. A HOLD is a legitimate research
    # outcome and may not be relabeled as FROZEN_OOS_QUALIFIED.
    qualification_key=os.getenv('NCAAF_V55_QUALIFIED_RULE_REGISTRY_KEY',
        'research/ncaaf/v55/qualified_independent_brains_v55_current.json')
    frozen=None;problem=None
    try:
        raw=storage_client.bucket(bucket).blob(qualification_key).download_as_bytes()
        document=json.loads(raw)
        audit['qualification_registry_uri']='gs://'+bucket+'/'+qualification_key
        audit['qualification_registry_status']=s(document.get('status'))
        audit['qualification_candidates']=int(document.get('candidate_system_count') or 0)
        audit['qualification_systems']=len(document.get('systems') or [])
        audit['qualification_hold_counts']=document.get('hold_counts') or {}
        frozen,problem=validate_v55_current_qualification(
            document,raw_sha=hashlib.sha256(raw).hexdigest())
        if not frozen:
            problem=s(document.get('status'))+':'+s(problem)
    except Exception as exc:
        problem='V55_QUALIFICATION_REGISTRY_UNAVAILABLE:'+type(exc).__name__

    if problem or not frozen or not frozen.get('rules'):
        audit.update(status='HOLD_NO_FROZEN_EXPERT_LIVE_RULE_REGISTRY',reason=str(problem or 'NO_FROZEN_RULES'))
        return [],audit
    expert_rules={k:v for k,v in frozen['rules'].items() if k[0]!='SYSTEM_MINER'}
    if not expert_rules:
        audit.update(status='NO_QUALIFIED_EXPERT_RULES_MINER_MAY_STILL_RUN',
                     accepted_events=0,full_history_miner_registry_rules=frozen.get('miner_rules',0))
        return [],audit
    frozen=dict(frozen,rules=expert_rules)
    # Every live evaluation uses current executable paired quotes only; long
    # research-reference quotes cannot be used to declare original predictions.
    fresh=board.loc[board['Backend_Quote_Timestamp'].map(lambda x:
        bool((t:=clock(x)) and 0<=(at-t).total_seconds()<=900))].copy()
    if fresh.empty:
        audit['status']='NO_EXECUTION_FRESH_PAIRED_QUOTES';return [],audit
    live_rows,qa=_v55_direct_current_quote_rows(fresh,book_quotes,now=at)
    audit['paired_quote_audit']=qa
    if live_rows.empty:
        audit['status']='NO_ORIGINAL_PAIRED_QUOTE_ROWS';return [],audit
    try:
        import utils
        enriched=utils.attach_pathi_bigal_backend_features(live_rows,'NCAAF',days_back=None)
        if not isinstance(enriched,pd.DataFrame) or len(enriched)!=len(live_rows):
            raise RuntimeError('FROZEN_FEATURE_PRODUCER_CHANGED_QUOTE_ROWS')
        # Evaluation occurs NOW, AFTER the original sportsbook quote. Do not
        # backdate the newly computed Pathi/Big Al/STAT rule firing.
        enriched=enriched.copy()
        evaluated_at=datetime.now(timezone.utc)
        enriched['V55_Original_Quote_Timestamp']=enriched['Snapshot_Timestamp']
        # The time is recorded AFTER feature evaluation finished. Recording
        # the start of a long enrichment job would falsely backdate signals.
        enriched['Snapshot_Timestamp']=pd.Timestamp(evaluated_at)
        enriched['V55_Live_Computed_On_Current_Quote']=True
    except Exception as exc:
        audit.update(status='HOLD_EXPERT_FEATURE_PRODUCER_FAILED',error=type(exc).__name__+':'+str(exc)[:240])
        return [],audit
    both=[]
    for row in fresh.to_dict('records'):
        match=s(row.get('Matchup'))
        if ' @ ' not in match:continue
        away,home=match.split(' @ ',1)
        for side,prefix,name in [('HOME','Home',home),('AWAY','Away',away)]:
            ln=num(row.get('Backend_'+prefix+'_Spread'))
            if ln is None:continue
            item=dict(row);item['Selected_Team']=name;item['Current_Spread']=ln
            both.append(item)
    if not both:
        audit['status']='NO_UNAMBIGUOUS_TWO_SIDED_GAME_MATCHUP';return [],audit
    observed,qa=derive_v55_other_qualified_events(pd.DataFrame(both),enriched,frozen,now=evaluated_at)
    audit['rule_firing_audit']=qa
    prepared,validation=normalize_v55_producer_events(observed,board,frozen,now=evaluated_at)
    audit['normalization_audit']=validation
    if not prepared:
        audit['status']='NO_VERIFIED_CURRENT_FROZEN_RULE_FIRINGS';return [],audit
    try:
        captured,capture=persist_v55_prospective_events(prepared,storage_client,bucket,now=evaluated_at)
        audit['capture_audit']=capture
        audit.update(status='PASS_PROSPECTIVE_CAPTURED',accepted_events=len(captured),
                     by_brain=dict(Counter(e['source_family'] for e in captured)))
        return captured,audit
    except Exception as exc:
        audit.update(status='HOLD_PROSPECTIVE_IMMUTABLE_CAPTURE_WRITE_FAILED',
                     error=type(exc).__name__+':'+str(exc)[:240])
        return [],audit




def seven_brain_live_combination_audit(signals, *, max_examples=150):
    """Enumerate cross-family live evidence without approving it by count.

    All seven brain families may originate. A single family contributes at most
    one independent unit even with many correlated systems. Opposing directions
    and missing dependency evidence remain explicit holds. This function NEVER
    changes `evaluate` or certified ROI authority.
    """
    import itertools
    by_market=defaultdict(lambda:defaultdict(lambda:defaultdict(set)))
    by_system=defaultdict(lambda:defaultdict(lambda:defaultdict(set)))
    invalid=Counter()
    for record in signals:
        if not isinstance(record,dict):continue
        gid=s(record.get('game_id') or record.get('Game_ID'))
        fam=family(record.get('source_family') or record.get('Source_Family'))
        mk=market(record.get('market') or record.get('Market'))
        side=s(record.get('side') or record.get('Side')).upper()
        system=s(record.get('system_id') or record.get('System_ID'))
        dependency=s(record.get('evidence_family_key') or record.get('Evidence_Family_Key'))
        if not gid or fam not in BRAINS or mk not in ('spreads','h2h','totals') or not system or side not in (('OVER','UNDER') if mk=='totals' else ('HOME','AWAY')):
            invalid['INVALID_SIGNAL_IDENTITY']+=1;continue
        key=(gid,mk)
        by_market[key][side][fam].add(dependency if dependency else fam)
        by_system[key][side][fam].add(system)
    examples=[];counts=Counter()
    for (gid,mk),sides in sorted(by_market.items()):
        sides_present=set(sides)
        opposite=bool(('HOME' in sides_present and 'AWAY' in sides_present) or ('OVER' in sides_present and 'UNDER' in sides_present))
        for side,families in sorted(sides.items()):
            names=sorted(families)
            counts['candidate_sides']+=1
            if opposite:counts['opposed_candidate_sides']+=1
            # All subsets of distinct families (up to 127 for 7 families).
            combos=[]
            for n in range(1,len(names)+1):
                for parts in itertools.combinations(names,n):
                    dependencies=[next(iter(families[f])) if len(families[f])==1 else '' for f in parts]
                    if not all(dependencies) or len(set(dependencies))!=len(parts):
                        counts['dependent_or_ambiguous_combinations']+=1;continue
                    combos.append(parts)
            counts['independent_family_combinations']+=len(combos)
            if len(examples)<max_examples:
                examples.append({'game_id':gid,'market':mk,'side':side,
                    'families':names,'system_counts':{f:len(by_system[(gid,mk)][side][f]) for f in names},
                    'opposing_family_evidence':opposite,
                    'distinct_family_combo_count':len(combos),
                    'largest_independent_combo_size':max(map(len,combos),default=0),
                    'status':'COMBINATION_RESEARCH_ONLY__NEEDS_OOS_CERTIFIED_EV',
                    'bet_authority':0})
    return {'status':'SEVEN_BRAIN_COMBINATION_AUDIT','families':sorted(BRAINS),
            'counts':dict(counts),'invalid':dict(invalid),'examples':examples,
            'truncated_examples':counts['candidate_sides']>len(examples),
            'automatic_authority':False}

def run_backend_refresh(*,storage_client=None,bq_client=None,bucket_name=None,log_func=None):
    """Production scanner dispatch. ONE NCAAF authority/publisher.

    Existing frozen weekly forecasts and BigQuery paired quotes are data
    readers only. NO legacy V55 board decisions, candidate selector, or writer
    executes. Requires separately captured live directional signals; missing
    feed is clearly held and cannot become certified betting evidence.
    """
    import logging
    from google.cloud import storage,bigquery
    from google.api_core.exceptions import NotFound
    import ncaaf_v552_live as frozen_source
    # Source-only import: does NOT execute the historical authority, board
    # publisher, V7 shadow selector, or any other decision-making routine.
    from ncaaf_v55_board import read_scanner_quotes,attach_verified_backend_quotes
    log=log_func or logging.info
    bucket=bucket_name or os.environ.get('MODEL_BUCKET','sharp-models')
    gcs=storage_client or storage.Client()
    bq=bq_client or bigquery.Client(project='sharplogger')
    mode=s(os.environ.get('NCAAF_V55_CUTOVER_MODE','SHADOW')).upper()
    if mode not in ('SHADOW','LIVE'):
        raise RuntimeError('INVALID_NCAAF_TWO_ENGINE_CUTOVER_MODE')

    # SHA-verifies the GCS frozen weekly snapshot; never call refresh_market_only
    # or the old market_refresh_and_publish because they invoke old decisions.
    weekly,source_audit=frozen_source.read_weekly_latest(gcs,bucket_name=bucket)
    if source_audit.get('production_v1_probabilities_used') is not False:
        raise RuntimeError('INVALID_FROZEN_FORECAST_PROVENANCE')
    # Clear older authority and candidate fields from input before the new
    # engine writes a fresh canonical record. Preserve CORE's original pick.
    base=weekly.copy()
    # Weekly Selected_Team is chosen using a former CORE probability/EV
    # market scan and is NOT a certified ROI system's original selection.
    # Never attribute it to CORE. Allow the verified quote matcher to return
    # both sides with no selected-side shortcut.
    base['Selected_Team']=None
    legacy=[c for c in base.columns if c.startswith(('V55_','V7_','Confirm_','Oppose_'))]
    for c in legacy:base.drop(columns=[c],inplace=True)
    # Market source and current prices are always the scanner's BigQuery feed.
    book_quotes=read_scanner_quotes(bq,minutes=180,horizon_days=14)
    board,pricing=attach_verified_backend_quotes(base,book_quotes,max_age_minutes=180)
    pricing["reference_quote_max_age_minutes"]=180
    pricing["execution_quote_max_age_minutes"]=15
    pricing["reference_quote_never_grants_bet_authority"]=True
    # Source-specific signal timestamps and frozen hashes are mandatory.
    # Never infer a "directional signal" from CORE probabilities, names, or
    # a descriptive Pathi/Big Al context-only display overlay.
    signals_uri=s(os.environ.get('NCAAF_V55_LIVE_SIGNALS_URI'))
    signals_table=s(os.environ.get('NCAAF_V55_LIVE_SIGNALS_BQ_TABLE'))
    sources_manifest=s(os.environ.get('NCAAF_V55_SEVEN_BRAIN_MANIFEST_URI'))
    source_hold=None
    seven_brain_input_audit={'status':'NOT_CONFIGURED','accepted_events':0}
    try:
        if sources_manifest:
            signals,seven_brain_input_audit=read_seven_brain_signal_sources(sources_manifest,bq)
            if seven_brain_input_audit['status']!='PASS':
                source_hold='LIVE_SEVEN_BRAIN_SOURCE_VALIDATION_HOLD'
        elif signals_uri:
            signals=read_rows(signals_uri)
            seven_brain_input_audit={'status':'SINGLE_SOURCE_LEGACY_URI','source':signals_uri,
                                     'accepted_events':len(signals) if isinstance(signals,list) else 0}
        elif signals_table:
            signals=read_original_live_signals_from_bq(bq,signals_table)
            seven_brain_input_audit={'status':'SINGLE_SOURCE_LEGACY_BQ','source':signals_table,
                                     'accepted_events':len(signals)}
        else:
            signals=[];source_hold='LIVE_ORIGINAL_DIRECTIONAL_SIGNAL_FEED_NOT_CONFIGURED'
    except (NotFound,FileNotFoundError) as exc:
        signals=[];source_hold='LIVE_DIRECTIONAL_SIGNAL_FEED_MISSING:'+type(exc).__name__
    except Exception as exc:
        signals=[];source_hold='LIVE_DIRECTIONAL_SIGNAL_FEED_INVALID:'+type(exc).__name__+':'+str(exc)
    if not isinstance(signals,list):
        raise RuntimeError('INVALID_LIVE_SIGNAL_FEED_SCHEMA')
    # The existing frozen expert feature evaluators are now a producer of
    # prospectively captured original decisions. This runs in the SAME backend
    # scan and enters the SAME seven-brain evaluator; no additional service or
    # permanent Python file. It may yield zero when a family has no qualified
    # current rule, and NEVER imports the retired Miner V2 executor.
    producer_audit={'status':'DISABLED_BY_OPERATOR','accepted_events':0}
    if s(os.getenv('NCAAF_V55_PRODUCER_BRIDGE_ENABLED','1')).lower() not in ('0','off','false','no'):
        try:
            newly_captured,producer_audit=produce_v55_verified_expert_signals(
                board,book_quotes,storage_client=gcs,bucket=bucket,log_func=log)
            if newly_captured:
                signals.extend(newly_captured)
                if source_hold=='LIVE_ORIGINAL_DIRECTIONAL_SIGNAL_FEED_NOT_CONFIGURED':
                    source_hold=None
        except Exception as exc:
            producer_audit={'status':'HOLD_PRODUCER_BRIDGE_EXCEPTION',
                 'error':type(exc).__name__+':'+str(exc)[:240], 'accepted_events':0}
    log('[NCAAF-V55-SEVEN-BRAIN-PRODUCER-AUDIT] '+json.dumps(producer_audit,sort_keys=True,default=str))
    seven_brain_input_audit['prospective_producer']=producer_audit
    seven_brain_input_audit['total_original_events_including_new_capture']=len(signals)
    seven_brain_input_audit['prospectively_captured_new_events']=int(producer_audit.get('accepted_events',0) or 0)
    if (producer_audit.get('status')=='PASS_PROSPECTIVE_CAPTURED' and
            seven_brain_input_audit.get('status')=='NOT_CONFIGURED'):
        seven_brain_input_audit['status']='PASS_PROSPECTIVE_PRODUCER_CAPTURE'
    log('[NCAAF-V55-SEVEN-BRAIN-INPUT-AUDIT] '+json.dumps(seven_brain_input_audit,sort_keys=True))
    registry_uri=s(os.environ.get('NCAAF_V55_ROI_REGISTRY_URI')) or (
        'gs://'+bucket+'/research/ncaaf/v55/two_engine_roi/current_registry.json')
    try:
        registry=read_registry(registry_uri)
    except (NotFound,FileNotFoundError) as exc:
        registry=None
        source_hold=(source_hold+';' if source_hold else '')+'ROI_REGISTRY_MISSING:'+type(exc).__name__
    # Certified ROI is a separate, existing research contract. A large
    # retrospective sample with zero certificates is never a betting edge.
    registry_audit={'registry_uri':registry_uri,
                    'status':s((registry or {}).get('status')) if isinstance(registry,dict) else 'UNAVAILABLE',
                    'certification_status':s((registry or {}).get('edge_certification_status')) if isinstance(registry,dict) else '',
                    'certified_edge_count':len(checked_registry(registry,datetime.now(timezone.utc))),
                    'research_only':bool((registry or {}).get('research_only',False)) if isinstance(registry,dict) else None}
    log('[NCAAF-V55-SEVEN-BRAIN-ROI-READINESS] '+json.dumps(registry_audit,sort_keys=True))

    ml_uri=s(os.environ.get('NCAAF_V55_ML_MANIFEST_URI')) or (
        'gs://'+bucket+'/research/ncaaf/v55/ml/current_manifest.json')
    try:
        ml_audit=audit_ml_research_manifest(read_registry(ml_uri))
    except Exception as exc:
        ml_audit={'status':'HOLD_ML_MANIFEST_UNAVAILABLE',
                  'error':type(exc).__name__+':'+str(exc),'bet_authority':0}
    log('[NCAAF-V55-ML-SHADOW-MANIFEST] '+json.dumps(ml_audit,sort_keys=True))
    # Full-history score-regression research artifacts are loaded only when
    # current pregame CFBD feature records are supplied with exact provenance.
    # No hidden feature inference from the weekly board or postgame warehouse.
    feature_uri=s(os.environ.get('NCAAF_V55_ML_LIVE_FEATURES_URI'))
    feature_table=s(os.environ.get('NCAAF_V55_ML_LIVE_FEATURES_BQ_TABLE')) or 'sharplogger.sharp_data.ncaaf_cfbd_miner_features_v1'
    ml_feature_rows=[]
    feature_audit={'status':'HOLD_NO_FORWARD_FEATURE_SOURCE','bet_authority':0}
    try:
        if feature_uri:
            ml_feature_rows=read_rows(feature_uri)
            if not isinstance(ml_feature_rows,list):raise ValueError('INVALID_ML_FEATURE_ROWS')
            feature_audit={'status':'LIVE_FEATURE_ROWS_LOADED','source':feature_uri,
                           'feature_rows':len(ml_feature_rows),'bet_authority':0}
        elif feature_table:
            manifest=read_registry(ml_uri)
            wanted={c for key,spec in (manifest.get('artifacts') or {}).items()
                    if key.startswith('FULL_') for c in spec.get('features',[])}
            # Only explicit qualified Miner prior-field recipes extend the
            # existing forward CFBD read. No guessing from atom names.
            try:
                qkey=os.getenv('NCAAF_V55_QUALIFIED_RULE_REGISTRY_KEY',
                    'research/ncaaf/v55/qualified_independent_brains_v55_current.json')
                qraw=gcs.bucket(bucket).blob(qkey).download_as_bytes()
                qreg,qproblem=validate_v55_current_qualification(
                    json.loads(qraw),raw_sha=hashlib.sha256(qraw).hexdigest())
                if qreg:
                    for (fam,_),rule in qreg['rules'].items():
                        if fam!='SYSTEM_MINER':continue
                        for condition in rule.get('conditions') or []:
                            field=s(condition['field'])
                            if field.endswith('_Prior'):
                                wanted.add('CFBD_DIFF_'+field)
            except Exception:
                # Still score frozen ML features; Miner audit will show hold.
                pass
            source=read_live_cfbd_rows_from_bq(bq,feature_table,wanted)
            ml_feature_rows,feature_audit=build_live_cfbd_feature_snapshots(
                board.to_dict('records'),source,wanted)
            feature_audit['source']=feature_table
            feature_audit['feature_contract']='READ_TIME_SHADOW_ONLY_NO_ORIGINAL_CAPTURE_CLAIM'
            feature_audit['source_adapter_version']='20261110_CFBD_MINER_FORWARD_V2_BOARD_MATCHUP'
        else:
            feature_audit['status']='LIVE_FEATURES_URI_OR_BQ_TABLE_NOT_CONFIGURED'
    except Exception as exc:
        ml_feature_rows=[]
        feature_audit={'status':'HOLD_LIVE_CFBD_FEATURE_SOURCE',
                       'error':type(exc).__name__+':'+str(exc),'bet_authority':0}
    log('[NCAAF-V55-FULL-HISTORY-FEATURE-AUDIT] '+json.dumps(feature_audit,sort_keys=True))
    # Full-history Miner V5 evaluates side-specific, audited pregame feature
    # records; no V2 votes and no retrospective replay promoted to live.
    miner_registry=None;miner_registry_hold='NOT_CONFIGURED'
    try:
        qualification_key=os.getenv('NCAAF_V55_QUALIFIED_RULE_REGISTRY_KEY',
            'research/ncaaf/v55/qualified_independent_brains_v55_current.json')
        qraw=gcs.bucket(bucket).blob(qualification_key).download_as_bytes()
        miner_registry,miner_registry_hold=validate_v55_current_qualification(
            json.loads(qraw),raw_sha=hashlib.sha256(qraw).hexdigest())
    except Exception as exc:
        miner_registry_hold='V55_MINER_REGISTRY_READ_FAILED:'+type(exc).__name__
    miner_audit={'status':'HOLD_QUALIFICATION_REGISTRY', 'reason':miner_registry_hold,
                 'captured':0}
    if miner_registry is not None:
        new_miner,miner_audit=produce_v55_full_history_miner(
            board,ml_feature_rows,miner_registry,storage_client=gcs,bucket=bucket)
        signals.extend(new_miner)
        if new_miner and source_hold=='LIVE_ORIGINAL_DIRECTIONAL_SIGNAL_FEED_NOT_CONFIGURED':
            source_hold=None
    log('[NCAAF-V55-FULL-HISTORY-MINER-PRODUCER-AUDIT] '+json.dumps(
        miner_audit,sort_keys=True,default=str))
    seven_brain_input_audit['full_history_miner_producer']=miner_audit
    seven_brain_input_audit['total_original_events_including_new_capture']=len(signals)
    extras_uri=s(os.environ.get('NCAAF_V55_OTHER_MARKET_QUOTES_URI'))
    extra_quotes=read_rows(extras_uri) if extras_uri else None
    signals,live_identity_audit=canonicalize_live_signal_game_ids(signals,board)
    combination_audit=seven_brain_live_combination_audit(signals)
    log('[NCAAF-V55-SEVEN-BRAIN-LIVE-COMBINATIONS] '+json.dumps(
        {k:v for k,v in combination_audit.items() if k!='examples'},sort_keys=True))
    output,audit=evaluate(board.to_dict('records'),signals,registry,extra_quotes=extra_quotes)
    audit['two_engine_live_identity_matching']=live_identity_audit
    audit['seven_brain_live_combinations']={k:v for k,v in combination_audit.items() if k!='examples'}
    audit['seven_brain_source_input_audit']=seven_brain_input_audit
    audit['seven_brain_roi_registry_audit']=registry_audit
    audit['ml_shadow_manifest']=ml_audit
    # Explicit research-to-live scoring interface. Keep authority unchanged:
    # the original exact-OOS ROI registry remains the sole certifier.
    shadow={'status':'HOLD_ML_FEATURE_SOURCE_OR_MANIFEST','bet_authority':0}
    if ml_feature_rows and ml_audit.get('status')=='ML_SHADOW_MANIFEST_PRESENT':
        try:
            import joblib
            import io
            md=read_registry(ml_uri)
            specs={}
            for name,ref in sorted((md.get('artifacts') or {}).items()):
                if not name.startswith('FULL_'):continue
                uri=s(ref.get('uri'))
                if not uri.startswith('gs://'):continue
                b,k=uri[5:].split('/',1)
                blob=gcs.bucket(b).blob(k).download_as_bytes()
                if hashlib.sha256(blob).hexdigest()!=s(ref.get('sha256')):
                    raise ValueError('ML_ARTIFACT_SHA256_MISMATCH:'+name)
                spec=joblib.load(io.BytesIO(blob))
                if list(spec.get('features',[]))!=list(ref.get('features',[])):
                    raise ValueError('ML_ARTIFACT_FEATURE_SCHEMA_MISMATCH:'+name)
                if spec.get('bet_authority')!=0 or spec.get('forward_eligible') is not False:
                    raise ValueError('ML_SHADOW_ONLY_ARTIFACT_REQUIRED:'+name)
                specs[name]=spec
            current_quotes=row_quotes(board.to_dict('records'))+(extra_quotes or [])
            shadow_quote_clock=datetime.now(timezone.utc)
            valid_quotes=[q for raw in current_quotes if (q:=quote_normal(raw,shadow_quote_clock,age_limit=180))]
            # quote_normal dates are objects; score_snapshot receives text.
            qrows=[{**q,'kickoff_utc':q['kickoff'].isoformat(),
                     'quote_observed_at_utc':q['quote'].isoformat()} for q in valid_quotes]
            shadow=score_snapshot(specs,md,ml_feature_rows,qrows,signals)
            # Store a compact per-game audit without modifying ROI authority.
            per_game=defaultdict(list)
            for record in shadow.get('records',[]):per_game[record['game_id']].append(record)
            for row in output:
                gid=s(row.get('Game_ID') or row.get('game_id'))
                row['V55_ML_Full_History_Shadow_JSON']=json.dumps(per_game.get(gid,[]),sort_keys=True,allow_nan=False)
                row['V55_ML_Full_History_Shadow_Status']=shadow['status'] if gid in per_game else 'NO_MATCHED_ML_SHADOW'
            log('[NCAAF-V55-FULL-HISTORY-LIVE-SCORING] '+json.dumps(
                {k:v for k,v in shadow.items() if k!='records'},sort_keys=True))
        except Exception as exc:
            shadow={'status':'HOLD_FULL_HISTORY_SCORING_EXCEPTION',
                    'error':type(exc).__name__+':'+str(exc),'bet_authority':0}
            log('[NCAAF-V55-FULL-HISTORY-SCORING-HOLD] '+json.dumps(shadow,sort_keys=True))
    audit['ml_full_history_live_handoff']={k:v for k,v in shadow.items() if k!='records'}
    audit['ml_full_history_forward_feature_source']=feature_audit

    # Deployment kill switch also controls PUBLICATION of authority. Keep
    # research candidates but never place BET / STRONG BET on the board in OFF.
    auth_on=os.getenv('NCAAF_V55_BET_AUTHORITY_MODE','OFF').upper()=='ON'
    if not auth_on:
        would_bet=int(audit.get('v55_authorized_bet_rows',0))
        for row in output:
            cand=json.loads(row.get('V55_Unified_Candidates_JSON') or '[]')
            for c in cand:
                if c.get('bet_authority'):
                    c['bet_authority']=0
                    c['hold']='BET_AUTHORITY_OPERATOR_OFF'
                    c['status']='RESEARCH_ONLY__BET_AUTHORITY_OFF'
            row['V55_Unified_Candidates_JSON']=json.dumps(cand,sort_keys=True)
            row['V55_Bet_Authority']=0
            row['V55_Bet_Decision']='CANDIDATE' if cand else 'PASS'
            row['V55_Authority_State']=None
            row['V55_Authority_Hold_Reason']='BET_AUTHORITY_OPERATOR_OFF' if cand else row.get('V55_Authority_Hold_Reason')
            for field in ('V55_Validated_Edge_Certificate_ID','V55_Validated_Edge_ROI','V55_Validated_Edge_Lower_ROI',
                          'V55_Bet_Key','V55_Qualified_Evidence_Detail_JSON','V55_Independent_Support_Keys'):
                row[field]=None
        audit['two_engine_operator_disabled_authority_candidates']=would_bet
        audit['v55_bet_authority']=0
        audit['v55_authorized_bet_rows']=0
        audit['v55_strong_bet_rows']=0
    # Sanitization is SERIALIZATION only: never turn a missing numeric
    # feature into an authorized decision or a synthetic price.
    def clean_for_json(value):
        if isinstance(value,dict):return {str(k):clean_for_json(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [clean_for_json(v) for v in value]
        if value is None:return None
        if hasattr(value,'item'):
            try:return clean_for_json(value.item())
            except (ValueError,TypeError):pass
        if isinstance(value,float) and not math.isfinite(value):return None
        if not isinstance(value,(str,int,float,bool)):
            try:
                import pandas as pd
                if pd.isna(value):return None
            except (ValueError,TypeError):pass
            if isinstance(value,(datetime,)) and value.tzinfo:
                return value.isoformat()
            return str(value)
        return value
    output=clean_for_json(output)
    audit.update({'two_engine_source_status':'PASS' if not source_hold else 'HOLD',
                  'two_engine_source_hold':source_hold,
                  'two_engine_original_signal_rows':len(signals),
                  'two_engine_live_signal_families':dict(Counter(family(x.get('source_family') or x.get('Source_Family')) for x in signals)),
                  'two_engine_registry_uri':registry_uri,
                  'two_engine_source_forecast_run':source_audit.get('run_prefix'),
                  'two_engine_quote_ingestion':pricing,
                  'two_engine_cutover_mode':mode})
    # No public BET authority in SHADOW regardless of certificates.
    if mode=='SHADOW':
        audit['two_engine_shadow_authority_would_qualify']=audit.get('v55_authorized_bet_rows',0)
        audit['two_engine_shadow_status']='PUBLISH_BLOCKED_UNTIL_EXPLICIT_LIVE_CUTOVER'
        log('[NCAAF-V55-TWO-ENGINE-SHADOW] '+json.dumps({
            'games':len(output),'prices':pricing.get('priced_games'),
            'live_signal_rows':len(signals),'certificates':audit.get('v55_validated_edge_certificates'),
            'would_authorize':audit.get('v55_authorized_bet_rows'),
            'hold':source_hold,'old_authority_executed':False},sort_keys=True))
        return {'status':'SHADOW_NO_PUBLICATION','rows':len(output),
                'signals':len(signals),'authority':0,'hold':source_hold}

    # LIVE is an explicit cutover, never a fallback to the old NCAAF publisher.
    # Missing original signal capture cannot be published as a deceptively
    # complete candidate board. It is safe to show a source HOLD via logs.
    if source_hold:
        log('[NCAAF-V55-TWO-ENGINE-LIVE-HOLD] '+str(source_hold))
        return {'status':'HOLD_NO_PUBLICATION','hold':source_hold,'authority':0}
    result=publish_snapshot(output,audit,client=bq)
    log('[NCAAF-V55-TWO-ENGINE-BACKEND-PUBLISH] '+json.dumps({
        **result,'signals':len(signals),'certificates':audit.get('v55_validated_edge_certificates'),
        'source_holds':source_hold,'old_authority_executed':False},sort_keys=True))
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description='Single NCAAF seven-brain ROI-only live betting backend')
    p.add_argument('--board',required=True,help='Existing V55 same-book, two-sided board JSON or CSV')
    p.add_argument('--signals',required=True,help='Original timestamped seven-brain live signal observations')
    p.add_argument('--registry',required=True,help='Research ROI registry (local or gs://)')
    p.add_argument('--quotes',help='Additional actual two-sided h2h / totals quote rows')
    p.add_argument('--out',required=True,help='Output game board JSON local or gs://')
    p.add_argument('--audit',required=True,help='Output authority audit JSON local or gs://')
    p.add_argument('--publish',action='store_true',help='Publish only after existing dashboard reader contract is migrated')
    args=p.parse_args(argv)
    try:
        out,audit=evaluate(read_rows(args.board),read_rows(args.signals),read_registry(args.registry),
                           extra_quotes=read_rows(args.quotes) if args.quotes else None)
        write_json(args.out,out);write_json(args.audit,audit)
        if args.publish:print(json.dumps(publish_snapshot(out,audit),sort_keys=True))
        print(json.dumps({'status':'PASS','candidates':audit['v55_candidate_rows'],
                          'certified_edges':audit['v55_validated_edge_certificates'],
                          'authorized':audit['v55_authorized_bet_rows']},sort_keys=True))
        return 0
    except Exception as exc:
        print('NCAAF_BACKEND_ERROR:'+type(exc).__name__+':'+str(exc),file=sys.stderr)
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

def score_snapshot(models, manifest, inputs, quotes, signals=(), *,now=None):
    """Score audited forward feature snapshots and actual fresh paired quotes.

    No probability or odds is synthesized if a required feature or current quote
    is absent. Outputs never change production authority fields.
    """
    import numpy as np
    now=now or datetime.now(timezone.utc)
    if now.tzinfo is None:raise ValueError('NAIVE_NOW')
    by_id={str(r.get('game_id') or r.get('Game_ID')):r for r in inputs if isinstance(r,dict)}
    signal_counts=defaultdict(set)
    directional_counts=Counter()
    for s in signals:
        key=str(s.get('game_id') or s.get('Game_ID') or '')
        fam=str(s.get('source_family') or s.get('Source_Family') or '').upper()
        if key and fam in ('CORE','STAT','SYSTEM_MINER','BIG_AL','PATHI','PREDICTION_TRACKER','MARKET_RICH'):
            signal_counts[key].add(fam)
        market=str(s.get('market') or s.get('Market') or '').lower()
        side=str(s.get('side') or s.get('Side') or '').upper()
        if key and fam in _V55_FH_FAMILIES and market in ('spreads','h2h','totals') and side in ('HOME','AWAY','OVER','UNDER'):
            # This is only a shadow evidence count, never a certification.
            uniq=str(s.get('system_id') or s.get('System_ID') or '')
            directional_counts[(key,market,side,fam,uniq)]+=1
    result=[];holds=Counter();records=[]
    for q in quotes:
        gid=str(q.get('game_id') or '')
        r=by_id.get(gid)
        if not r:holds['NO_EXACT_LIVE_FEATURE_GAME_ID']+=1;continue
        try:
            kick=datetime.fromisoformat(str(q['kickoff_utc']).replace('Z','+00:00'))
            asof=datetime.fromisoformat(str(r['feature_asof_utc']).replace('Z','+00:00'))
            quoted=datetime.fromisoformat(str(q['quote_observed_at_utc']).replace('Z','+00:00'))
            if any(t.tzinfo is None for t in (kick,asof,quoted)) or not asof<kick or not quoted<=now<kick:
                holds['FEATURE_OR_QUOTE_CHRONOLOGY_INVALID']+=1;continue
        except (ValueError,TypeError,KeyError):holds['MISSING_VERIFIABLE_FEATURE_TIME']+=1;continue
        raw=r.get('core_features')
        if not isinstance(raw,dict):holds['NO_EXPLICIT_FEATURE_SCHEMA']+=1;continue
        raw=dict(raw)
        # Stage2/V6 expert counts are explicit in training. For live records,
        # construct the same *capped directional difference* from the captured
        # seven-brain signal stream; never use historical signals for new games.
        for mkt in ('spreads','h2h','totals'):
            left,right=('OVER','UNDER') if mkt=='totals' else ('HOME','AWAY')
            for fam in _V55_FH_FAMILIES:
                nleft=len({ident for (game,market,side,family,ident) in directional_counts
                           if (game,market,side,family)==(gid,mkt,left,fam)})
                nright=len({ident for (game,market,side,family,ident) in directional_counts
                            if (game,market,side,family)==(gid,mkt,right,fam)})
                raw['EVIDENCE_'+mkt+'_'+fam+'_NET']=float(min(3,nleft)-min(3,nright))
            nets=[raw['EVIDENCE_'+mkt+'_'+fam+'_NET'] for fam in _V55_FH_FAMILIES]
            raw['CONFLUENCE_'+mkt+'_SUPPORT_HOME']=float(sum(v>0 for v in nets))
            raw['CONFLUENCE_'+mkt+'_SUPPORT_AWAY']=float(sum(v<0 for v in nets))
            raw['CONFLUENCE_'+mkt+'_CONFLICT']=float(any(v>0 for v in nets) and any(v<0 for v in nets))
            raw['CONFLUENCE_'+mkt+'_INDEPENDENT_FAMILIES']=float(sum(v!=0 for v in nets))
        raw['MARKET_RICH_HIST_PRESENT']=float(any(
            k in raw and math.isfinite(_v55_fh_finite(raw[k]))
            for k in ('MARKET_SPREAD_LATEST','MARKET_TOTAL_LATEST','MARKET_H2H_HOME_PRICE')))
        for col in ('MARKET_SPREAD_OPEN','MARKET_SPREAD_LATEST','MARKET_SPREAD_MOVE',
                    'MARKET_TOTAL_OPEN','MARKET_TOTAL_LATEST','MARKET_TOTAL_MOVE',
                    'MARKET_H2H_HOME_PRICE','MARKET_SPREAD_BOOKS',
                    'MARKET_TOTAL_BOOKS','MARKET_SNAPSHOT_COUNT'):
            raw.setdefault(col,float('nan'))
        target=str(q.get('market') or '')
        if target not in ('spreads','h2h','totals'):continue
        model_target='total' if target=='totals' else 'margin'
        per={}
        for brain in ('CORE','STAT','COMBINED'):
            name=f'FULL_{brain}_{model_target}';spec=models.get(name)
            if spec is None:continue
            cols=spec['features']
            # Exact parity: missing columns != zero, and all numeric nonfinite
            # source values are preserved as explicitly present NaN only.
            if any(k not in raw for k in cols):holds['FEATURE_SCHEMA_PARITY_MISSING_COLUMNS']+=1;continue
            x=np.asarray([[_v55_fh_finite(raw[k]) for k in cols]],float)
            if not np.isfinite(x).any():holds['NO_NUMERIC_LIVE_FEATURES']+=1;continue
            p=float(spec['model'].predict(x)[0]);sigma=_v55_fh_finite(spec.get('residual_sigma'))
            if not math.isfinite(sigma) or sigma<=0:holds['MISSING_OOF_RESIDUAL_CALIBRATION']+=1;continue
            line=_v55_fh_finite(q.get('line'))
            if target=='spreads':threshold=-line if str(q.get('side')).upper()=='HOME' else line
            elif target=='h2h':threshold=0
            else:threshold=line
            if not math.isfinite(threshold):holds['NO_CURRENT_MARKET_LINE']+=1;continue
            p_over=NormalDist(mu=p,sigma=sigma).cdf(threshold)
            higher=1-p_over
            side=str(q.get('side') or '').upper()
            probability=(higher if side in ('HOME','OVER') else p_over)
            if target=='spreads' and side=='AWAY':probability=1-higher
            if target=='h2h' and side=='AWAY':probability=1-higher
            # ML price-screening only. Never turn this expected value into
            # authority without independent OOS proof, source capture and a
            # qualified ROI/EV contract. Prices are the exact current quote.
            net=profit(q.get('odds'))
            estimated_ev=(probability*net-(1-probability)) if net is not None else None
            per[brain]={'predicted_home_margin_or_total':round(p,5),
                        'shadow_probability':round(probability,6),
                        'shadow_ev_per_unit':round(estimated_ev,6) if estimated_ev is not None else None,
                        'positive_ev_shadow':estimated_ev is not None and estimated_ev>0,
                        'sigma':sigma}
        if per:
            records.append({'game_id':gid,'market':target,'side':q.get('side'), 'book':q.get('book'),
                            'quoted_line':q.get('line'),'quoted_odds':q.get('odds'),
                            'quote_age_minutes':round((now-quoted).total_seconds()/60.0,2),
                            'shadow_price_status':('REFERENCE_QUOTE_NOT_EXECUTABLE__OVER_15_MIN'
                                                   if (now-quoted).total_seconds()>900
                                                   else 'REFERENCE_QUOTE_WITHIN_15_MIN__NO_AUTHORITY'),
                            'models':per,'brain_signal_families':sorted(signal_counts.get(gid,set())),
                            'status':'SHADOW_NOT_AUTHORIZED','bet_authority':0})
    return {'status':'SHADOW_SCORED' if records else 'HOLD_LIVE_SCORING_INPUTS',
            'scored_market_sides':len(records),'scored_games':len(set(r['game_id'] for r in records)),
            'holds':dict(holds),'records':records,'bet_authority':0}

# END INLINED FULL-HISTORY SCORER


# === V5.5 VERIFIED SOURCE DISCOVERY AND GUARDED ACTIVATION ===
# Kept in the existing backend: no additional application Python module.
# Run: python3 ncaaf_v55_backend.py activate --auto-source [--apply-shadow]
# Default backend CLI and production scanner refresh remain unchanged.
import re, subprocess, tempfile

_ACTIVATION_PROJECT='sharplogger'
_ACTIVATION_DEFAULT_REGISTRY='gs://sharp-models/research/ncaaf/v55/two_engine_roi/current_registry.json'
_ACTIVATION_REQUIRED_FIELDS={'game_id','market','side','source_family','signal_observed_at_utc','kickoff_utc'}
_ACTIVATION_EVIDENCE_FIELDS={'quote_observed_at_utc','direction_verified','system_id','evidence_family_key','frozen_definition_sha256'}
_ACTIVATION_CAPTURE_FIELDS={'signal_capture_ref','source_capture_ref','original_capture_id'}


def activation_discover(client):
    sql='SELECT table_name, column_name FROM `sharplogger.sharp_data.INFORMATION_SCHEMA.COLUMNS` ORDER BY table_name,column_name'
    schemas=defaultdict(set)
    for row in client.query(sql).result():
        x=dict(row.items())
        schemas[x['table_name']].add(x['column_name'])
    potential=[]
    for name,fields in schemas.items():
        if 'ncaaf' not in name.lower() and 'signal' not in name.lower() and 'brain' not in name.lower():
            continue
        match=_ACTIVATION_REQUIRED_FIELDS&fields
        if len(match)<3:continue
        missing=sorted(_ACTIVATION_EVIDENCE_FIELDS-fields)
        if not _ACTIVATION_CAPTURE_FIELDS&fields:missing.append('ORIGINAL_CAPTURE_REF_REQUIRED')
        potential.append({'table':f'{_ACTIVATION_PROJECT}.sharp_data.{name}',
                          'required_fields_present':len(match),
                          'required_fields_missing':sorted(_ACTIVATION_REQUIRED_FIELDS-fields),
                          'provenance_fields_missing':missing,
                          'original_capture_certified':False})
    potential.sort(key=lambda x:(len(x['required_fields_missing'])+len(x['provenance_fields_missing']),x['table']))
    return potential[:50]


def activation_inspect(manifest_uri,registry_uri,client):
    result={'at_utc':datetime.now(timezone.utc).isoformat(),
            'project':_ACTIVATION_PROJECT,'seven_brains_supported':sorted(BRAINS),
            'configuration_only':True,'bets_not_placed_by_this_tool':True,
            'manifest_uri':manifest_uri,'roi_registry_uri':registry_uri}
    try:
        registry=read_registry(registry_uri)
        certs=checked_registry(registry,datetime.now(timezone.utc))
        result['registry']={'status':registry.get('status'),
                            'edge_certification_status':registry.get('edge_certification_status'),
                            'valid_oos_roi_certificates':len(certs),
                            'historical_brain_observations':(registry.get('historical_seven_brain_replay') or {}).get('attached_observations'),
                            'no_retroactive_certification':True}
    except Exception as exc:
        result['registry']={'status':'UNAVAILABLE','error':type(exc).__name__+':'+str(exc)[:240],
                            'valid_oos_roi_certificates':0}
    if manifest_uri:
        try:
            events,audit=read_seven_brain_signal_sources(manifest_uri,client)
            result['signal_sources']=audit
            result['original_live_signal_rows']=len(events)
        except Exception as exc:
            result['signal_sources']={'status':'HOLD','error':type(exc).__name__+':'+str(exc)[:240]}
            result['original_live_signal_rows']=0
    else:
        result['signal_sources']={'status':'MANIFEST_NOT_CONFIGURED'}
        result['original_live_signal_rows']=0
        result['discovered_bq_candidates']=activation_discover(client)
    ready=(result['original_live_signal_rows']>0 and
           result['signal_sources']['status']=='PASS' and
           result['registry']['valid_oos_roi_certificates']>0)
    result['authority_preflight']='READY_TO_ENABLE_CONFIRMED_OOS_ONLY' if ready else 'HOLD_EVIDENCE_OR_SOURCE_NOT_READY'
    return result


def activation_main(argv=None):
    parser=argparse.ArgumentParser(description='NCAAF V5.5 seven-brain guarded enablement')
    source_select=parser.add_mutually_exclusive_group()
    source_select.add_argument('--manifest',help='gs:// source manifest with ORIGINAL live directional signals for available brains')
    source_select.add_argument('--auto-source',action='store_true',help='Find one existing independently captured BQ live event ledger (no historical replay)')
    parser.add_argument('--registry',default=_ACTIVATION_DEFAULT_REGISTRY)
    parser.add_argument('--report',default='ncaaf_v55_unified_readiness.json')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--apply-shadow',action='store_true',help='Connect configured live sources but leave bet authority OFF')
    group.add_argument('--apply-live',action='store_true',help='Enable unified LIVE publisher only with valid sources + OOS ROI certificates')
    args=parser.parse_args(argv)
    from google.cloud import bigquery
    client=bigquery.Client(project=_ACTIVATION_PROJECT)
    selected_manifest=None
    if args.auto_source:
        candidates=[c for c in activation_discover(client) if not c['required_fields_missing']
                    and not c['provenance_fields_missing']
                    and not any(word in c['table'].lower() for word in
                                ('historical','retrospective','postgame','archive','research','replay'))]
        qualified=[]
        for cand in candidates:
            candidate_manifest={'schema':SEVEN_BRAIN_MANIFEST_SCHEMA,
                  'sources':[{'family':'ALL','uri':'bq://'+cand['table']}],
                  'declaration':'Existing original, captured pregame evidence only'}
            with tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=True) as tmp:
                json.dump(candidate_manifest,tmp);tmp.flush()
                try:
                    events,audit=read_seven_brain_signal_sources(tmp.name,client)
                    if audit['status']=='PASS' and events:
                        qualified.append((cand['table'],candidate_manifest,len(events),set(audit['by_family'])))
                except Exception:continue
        # One combined seven-brain original feed OR distinct independent
        # per-family feeds are accepted. Ambiguous overlapping sources HOLD.
        selected=qualified
        if len(qualified)>1:
            all_brains=set().union(*(r[3] for r in qualified))
            supersets=[r for r in qualified if r[3]==all_brains]
            if len(supersets)==1:
                selected=supersets
            else:
                seen_families=set();disjoint=True
                for r in qualified:
                    if seen_families & r[3]:disjoint=False;break
                    seen_families|=r[3]
                if not disjoint:selected=[]
        if selected:
            selected_manifest={'schema':SEVEN_BRAIN_MANIFEST_SCHEMA,
                'sources':[e for r in selected for e in r[1]['sources']],
                'declaration':'Verified original live directional signal sources, no historical replay'}
            with tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=True) as tmp:
                json.dump(selected_manifest,tmp);tmp.flush()
                status=activation_inspect(tmp.name,args.registry,client)
            status['auto_source_tables']=[r[0] for r in selected]
            status['auto_source_rows']=sum(r[2] for r in selected)
            status['manifest_uri_for_deployment']='gs://sharp-models/research/ncaaf/v55/live/seven_brain_sources_verified_auto_v1.json'
        else:
            status=activation_inspect(None,args.registry,client)
            status['auto_source_status']='HOLD_NONE_OR_AMBIGUOUS_VALID_ORIGINAL_LIVE_LEDGER'
            status['auto_source_candidates_qualified']=[x[0] for x in qualified]
    else:
        status=activation_inspect(args.manifest,args.registry,client)
    Path(args.report).write_text(json.dumps(status,indent=2,sort_keys=True,default=str)+'\n')
    print(json.dumps({'authority_preflight':status['authority_preflight'],
                      'original_live_signal_rows':status['original_live_signal_rows'],
                      'valid_roi_certificates':status['registry']['valid_oos_roi_certificates'],
                      'discovered_bq_candidates':status.get('discovered_bq_candidates',[])[:15],
                      'report':args.report},indent=2))
    if not (args.apply_shadow or args.apply_live):return 0
    if not (args.manifest or selected_manifest) or status['signal_sources']['status']!='PASS' or not status['original_live_signal_rows']:
        print('HOLD: no valid original live signal source manifest; Cloud Run unchanged',file=sys.stderr)
        return 2
    if args.apply_live and status['authority_preflight']!='READY_TO_ENABLE_CONFIRMED_OOS_ONLY':
        print('HOLD: no certified OOS ROI edge registry; Cloud Run unchanged',file=sys.stderr)
        return 2
    manifest_to_deploy=args.manifest or status['manifest_uri_for_deployment']
    if not re.fullmatch(r'gs://[a-zA-Z0-9_.-]+/[a-zA-Z0-9_./-]+',manifest_to_deploy):
        print('HOLD: manifest GCS URI invalid; Cloud Run unchanged',file=sys.stderr)
        return 2
    if selected_manifest:
        # Only upload a source manifest after the BQ schema and real event
        # rows have been read and verified, and the user chose --apply-*.
        from google.cloud import storage
        bucket,key=manifest_to_deploy[5:].split('/',1)
        storage.Client().bucket(bucket).blob(key).upload_from_string(
            json.dumps(selected_manifest,sort_keys=True),content_type='application/json')
    mode='LIVE' if args.apply_live else 'SHADOW'
    auth='ON' if args.apply_live else 'OFF'
    # Existing CFBD/price readers are untouched. One publisher only.
    cmd=['gcloud','run','services','update','sharp-detection-trigger',
         '--project='+_ACTIVATION_PROJECT,'--region=us-east4',
         '--update-env-vars='+','.join((
             'NCAAF_V55_SEVEN_BRAIN_MANIFEST_URI='+manifest_to_deploy,
             'NCAAF_V55_CUTOVER_MODE='+mode,
             'NCAAF_V55_BET_AUTHORITY_MODE='+auth))]
    print('APPLYING_UNIFIED_ENGINE_MODE='+mode+' BET_AUTHORITY_MODE='+auth,flush=True)
    subprocess.run(cmd,check=True)
    return 0



if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='activate':
        raise SystemExit(activation_main(sys.argv[2:]))
    raise SystemExit(main())

