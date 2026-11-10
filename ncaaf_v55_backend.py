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
from datetime import datetime,timezone
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
            observed>at or not truth(e.get('direction_verified',e.get('Direction_Verified'))) or
            side not in (('OVER','UNDER') if m=='totals' else ('HOME','AWAY'))):
            rejected['INVALID_LIVE_EVENT']+=1;continue
        # A blank definition hash can originate research but NEVER match a
        # certified edge (certificate requires exact frozen component hash).
        component=f+'|'+sid
        for q in quotes:
            if (q['game_id']!=gid or q['market']!=m or q['side']!=side or
                book!=q['book'] or not (observed<=quoted<=q['quote']<q['kickoff']) or
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
    authority_on=s(os.getenv('NCAAF_V55_BET_AUTHORITY_MODE','OFF')).upper() in ('ON','USER_DIRECTED_PROVISIONAL')
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


def _v55_board_fixture_identity(row):
    """Read verified board ID/ordered teams/kickoff, or exact 'AWAY @ HOME'.

    The existing published weekly board can contain Matchup without separate
    Home_Team/Away_Team. Never infer sides from 'vs', a selected team, or odds.
    """
    board_id=_v55_live_id(row.get('Game_ID') or row.get('game_id'))
    home=slug(row.get('Home_Team') or row.get('Home_Team_Norm'))
    away=slug(row.get('Away_Team') or row.get('Away_Team_Norm'))
    method='EXPLICIT_HOME_AWAY'
    if not home or not away:
        matchup=s(row.get('Matchup') or row.get('matchup'))
        if matchup.count('@')==1:
            raw_away,raw_home=(v.strip() for v in matchup.split('@',1))
            if raw_away and raw_home:
                parsed_home,parsed_away=slug(raw_home),slug(raw_away)
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
        name=slug(r.get('Team_Key') or r.get('team_key'))
        if not gid or not kick or kick<=now or season!=now.year or side not in (0.0,1.0) or not name:
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
        h=slug(hr.get('Team_Key') or hr.get('team_key'))
        a=slug(ar.get('Team_Key') or ar.get('team_key'))
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
        if len(candidates)!=1:
            holds['NO_UNIQUE_FUTURE_CFBD_FIXTURE' if not candidates else 'AMBIGUOUS_FUTURE_CFBD_FIXTURE']+=1
            continue
        source_id=candidates[0]
        fixture=fixtures.get(source_id)
        if not fixture:
            holds['MAPPED_CFBD_ID_NOT_IN_FORWARD_FEATURE_TABLE']+=1;continue
        h,a,cfbd_kick,hr,ar=fixture
        if h!=home or a!=away or abs((cfbd_kick-kick).total_seconds())>900:
            holds['CFBD_FIXTURE_TEAM_OR_KICKOFF_CONFLICT']+=1;continue
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
                    'core_features':features,
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
         'WHERE `Season` = @season AND `Game_Date` > @now '
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
    book_quotes=read_scanner_quotes(bq,minutes=15,horizon_days=14)
    board,pricing=attach_verified_backend_quotes(base,book_quotes,max_age_minutes=15)
    # Source-specific signal timestamps and frozen hashes are mandatory.
    # Never infer a "directional signal" from CORE probabilities, names, or
    # a descriptive Pathi/Big Al context-only display overlay.
    signals_uri=s(os.environ.get('NCAAF_V55_LIVE_SIGNALS_URI'))
    signals_table=s(os.environ.get('NCAAF_V55_LIVE_SIGNALS_BQ_TABLE'))
    source_hold=None
    try:
        if signals_uri:
            signals=read_rows(signals_uri)
        elif signals_table:
            signals=read_original_live_signals_from_bq(bq,signals_table)
        else:
            signals=[];source_hold='LIVE_ORIGINAL_DIRECTIONAL_SIGNAL_FEED_NOT_CONFIGURED'
    except (NotFound,FileNotFoundError) as exc:
        signals=[];source_hold='LIVE_DIRECTIONAL_SIGNAL_FEED_MISSING:'+type(exc).__name__
    except Exception as exc:
        signals=[];source_hold='LIVE_DIRECTIONAL_SIGNAL_FEED_INVALID:'+type(exc).__name__+':'+str(exc)
    if not isinstance(signals,list):
        raise RuntimeError('INVALID_LIVE_SIGNAL_FEED_SCHEMA')
    registry_uri=s(os.environ.get('NCAAF_V55_ROI_REGISTRY_URI')) or (
        'gs://'+bucket+'/research/ncaaf/v55/two_engine_roi/current_registry.json')
    try:
        registry=read_registry(registry_uri)
    except (NotFound,FileNotFoundError) as exc:
        registry=None
        source_hold=(source_hold+';' if source_hold else '')+'ROI_REGISTRY_MISSING:'+type(exc).__name__

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
    extras_uri=s(os.environ.get('NCAAF_V55_OTHER_MARKET_QUOTES_URI'))
    extra_quotes=read_rows(extras_uri) if extras_uri else None
    signals,live_identity_audit=canonicalize_live_signal_game_ids(signals,board)
    output,audit=evaluate(board.to_dict('records'),signals,registry,extra_quotes=extra_quotes)
    audit['two_engine_live_identity_matching']=live_identity_audit
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
            valid_quotes=[q for raw in current_quotes if (q:=quote_normal(raw,datetime.now(timezone.utc)))]
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
            per[brain]={'predicted_home_margin_or_total':round(p,5), 'shadow_probability':round(probability,6),
                        'sigma':sigma}
        if per:
            records.append({'game_id':gid,'market':target,'side':q.get('side'), 'book':q.get('book'),
                            'quoted_line':q.get('line'),'quoted_odds':q.get('odds'),
                            'models':per,'brain_signal_families':sorted(signal_counts.get(gid,set())),
                            'status':'SHADOW_NOT_AUTHORIZED','bet_authority':0})
    return {'status':'SHADOW_SCORED' if records else 'HOLD_LIVE_SCORING_INPUTS',
            'scored_market_sides':len(records),'scored_games':len(set(r['game_id'] for r in records)),
            'holds':dict(holds),'records':records,'bet_authority':0}

# END INLINED FULL-HISTORY SCORER

if __name__=='__main__':raise SystemExit(main())
