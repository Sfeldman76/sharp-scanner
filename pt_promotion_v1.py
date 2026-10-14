"""Shared, fail-closed Prediction Tracker champion/challenger promotion contract.

GCS is the source of truth. Neither historical MAE alone nor current-week
forecasts grant a betting vote. Each candidate must pass a prespecified paired
historical gate AND a forward-locked, CORE-candidate-specific prospective gate.
The old PT family remains authoritative until promotion is recorded. Never
changes CORE, markets, execution or the one-independent-family vote limit.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from collections import defaultdict

TAG = 'pt-promotion-v1.0-candidate-forward-evidence-20261013'
PREFIX = 'research/pt_promotion_v1'
MIN_HOLDOUT_GAMES = 300
MIN_HIST_GAIN_POINTS = .10
MIN_PROSPECTIVE_PAIRED = 60
MIN_PROSPECTIVE_WEEKS = 4
MIN_DISAGREEMENTS = 15
MIN_PROSPECTIVE_DIRECTION_GAIN = .03
MIN_DIFFERENCE_DIRECTION_GAIN = .10
MIN_SIGNAL_EDGE_POINTS = 2.0


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _json(sc, bucket, path):
    blob=sc.bucket(bucket).blob(path)
    try:
        return json.loads(blob.download_as_bytes()) if blob.exists() else None
    except Exception:
        return None


def _atomic(sc, bucket, path, obj):
    blob=sc.bucket(bucket).blob(path)
    body=json.dumps(obj,sort_keys=True,separators=(',',':'),default=str)
    try:
        blob.upload_from_string(body,content_type='application/json',if_generation_match=0)
        return True
    except Exception as e:
        if e.__class__.__name__ in {'PreconditionFailed','Conflict'} or getattr(e,'code',None)==412:
            return False
        raise


def _sha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()


def _root(sport,market):
    s=str(sport).upper(); m=str(market).upper()
    if s not in {'NFL','NCAAF'} or m not in {'SPREADS','TOTALS'}:
        raise ValueError('unsupported sport or market')
    return f'{PREFIX}/{s.lower()}/{m.lower()}'


def read_selection(sc,bucket,sport,market):
    """Read the one immutable source lineup for this promotion campaign."""
    obj=_json(sc,bucket,f'{_root(sport,market)}/locked_five_sources.json')
    return obj if isinstance(obj,dict) and len(obj.get('sources') or [])==5 and obj.get('selection_sha256') else None


def lock_selection(sc,bucket,sport,market,selection):
    """Freeze a five-source lineup before collecting any prospective evidence.

    A different subsequent selector result CANNOT silently change the lineup.
    Such a change requires a new campaign rather than contaminating evidence.
    """
    root=_root(sport,market)
    selected=selection if isinstance(selection,dict) else {}
    sources=selected.get('sources') or []
    sha=selected.get('selection_sha256')
    existing=read_selection(sc,bucket,sport,market)
    if existing:
        return {**existing,'lock_status':'EXISTING_LOCKED' if sha==existing.get('selection_sha256') else 'CHALLENGER_CHANGED_LOCK_RETAINED'}
    if selected.get('status')!='FIVE_SELECTED_DEVELOPMENT_ONLY' or len(sources)!=5 or len(set(sources))!=5 or not sha:
        return {'status':'NO_VALID_FIVE_TO_LOCK','sources':[],'selection_sha256':None}
    frozen={'status':'FIVE_LOCKED','sources':list(sources),'source_providers':selected.get('source_providers') or {},
        'selection_sha256':sha,'development_years':selected.get('development_years'),
        'source_tag':TAG,'role':'PREDECLARED_FIVE_SOURCE_CAMPAIGN'}
    created=_atomic(sc,bucket,f'{root}/locked_five_sources.json',frozen)
    if not created:
        existing=read_selection(sc,bucket,sport,market)
        if existing:return {**existing,'lock_status':'EXISTING_LOCKED'}
        return {'status':'LOCK_WRITE_FAILED','sources':[],'selection_sha256':None}
    return {**frozen,'lock_status':'CREATED'}


def read_state(sc,bucket,sport,market):
    root=_root(sport,market)
    state=_json(sc,bucket,f'{root}/current_champion.json')
    if not isinstance(state,dict) or state.get('status')!='PROMOTED' or state.get('champion')!='FIVE_SOURCE':
        return {'status':'INCUMBENT','champion':'LEGACY','betting_authority':0}
    return state


def _weekly_period(s):
    try:
        d=dt.datetime.fromisoformat(str(s).replace('Z','+00:00'))
        return d.date().isocalendar()[:2]
    except Exception:
        return None


def historical_gate(evidence):
    """Evidence must be paired, chronologically split, and include per-year metrics.

    The pre-2026 bootstrap lower confidence limit is required: a small raw MAE
    win without demonstrated robustness cannot be promoted.
    """
    p=evidence if isinstance(evidence,dict) else {}
    hold=p.get('holdout') or {}
    n=int(hold.get('n') or 0)
    improvement=_num(hold.get('incumbent_mae'))
    challenger=_num(hold.get('challenger_mae'))
    gain=improvement-challenger if improvement is not None and challenger is not None else None
    ci=p.get('paired_bootstrap_ci95') or []
    lower=_num(ci[0]) if isinstance(ci,(list,tuple)) and len(ci)>1 else None
    yearly=p.get('holdout_by_year') or {}
    yrs=tuple(p.get('holdout_years') or (2024,2025))
    yrs_good=all(int((yearly.get(str(y)) or {}).get('n') or 0)>=100 and
                 _num((yearly.get(str(y)) or {}).get('gain_points')) is not None and
                 float(yearly[str(y)]['gain_points'])>0 for y in yrs)
    ok=bool(p.get('periods_disjoint') is True and p.get('asof_max_year',9999)<=2025 and
            len(yrs)>=2 and yrs_good and n>=MIN_HOLDOUT_GAMES and gain is not None and
            gain>=MIN_HIST_GAIN_POINTS and lower is not None and lower>0)
    return {'pass':ok,'n':n,'mae_gain_points':gain,'ci95_lower':lower,
            'positive_holdout_years':yrs_good,'requirements':'PAIRED_2024_25_POSITIVE_EACH_YEAR_AND_BOOTSTRAP_GT0'}


def record_historical(sc,bucket,sport,market,evidence):
    root=_root(sport,market)
    locked=read_selection(sc,bucket,sport,market)
    if not locked or not evidence.get('selection_sha256') or evidence['selection_sha256']!=locked['selection_sha256']:
        return {'pass':False,'status':'HOLD_SELECTION_NOT_LOCKED_OR_CHANGED'}
    payload={'source_tag':TAG,'sport':sport,'market':market,'historical':evidence,
             'gate':historical_gate(evidence),'evidence_sha256':_sha(evidence)}
    # Immutable, content addressed; current pointer is optional and can be stale.
    _atomic(sc,bucket,f'{root}/historical/{payload["evidence_sha256"]}.json',payload)
    sc.bucket(bucket).blob(f'{root}/current_historical.json').upload_from_string(
        json.dumps(payload,sort_keys=True,default=str),content_type='application/json')
    return payload['gate']


def capture(sc,bucket,sport,market,games,now=None):
    """Capture FIRST pregame snapshot only; no retrospective or in-place edits.

    Each game requires CORE candidate, independent PT predictions, a valid
    pregame quote and stable prediction/game identity. Current-week outcomes
    are explicitly never read by this method.
    """
    root=_root(sport,market)
    locked=read_selection(sc,bucket,sport,market)
    now=now or dt.datetime.now(dt.timezone.utc)
    if isinstance(now,str): now=dt.datetime.fromisoformat(now.replace('Z','+00:00'))
    if now.tzinfo is None: raise ValueError('timezone-aware capture timestamp required')
    result={'inserted':0,'existing':0,'skipped':0}
    for r in games:
        sid=str(r.get('game_id') or '').strip()
        try: kickoff=dt.datetime.fromisoformat(str(r.get('game_start')).replace('Z','+00:00'))
        except Exception: result['skipped']+=1; continue
        old=_num(r.get('incumbent_margin_home')); new=_num(r.get('challenger_margin_home'))
        line=_num(r.get('home_spread')); side=_num(r.get('core_home_direction'))
        count=int(_num(r.get('source_count')) or 0)
        if not (locked and str(r.get('selection_sha256') or '')==locked.get('selection_sha256') and
                sid and kickoff.tzinfo is not None and now<kickoff and
                bool(r.get('core_candidate')) and old is not None and new is not None and
                line is not None and side in {-1.,1.} and count>=3):
            result['skipped']+=1; continue
        event={
            'source_tag':TAG,'sport':str(sport).upper(),'market':str(market).upper(),
            'game_id':sid,'game_start':kickoff.isoformat(),
            'captured_at':now.isoformat(), 'core_candidate':True,
            'core_home_direction':int(side),'home_spread':line,
            'incumbent_margin_home':old,'challenger_margin_home':new,
            'source_count':count,'selection_sha256':locked['selection_sha256'],
            'prediction_contract':str(r.get('prediction_contract') or ''),
        }
        # Hash avoids arbitrary separators or path traversal in external game IDs.
        path=f'{root}/pregame/{hashlib.sha256(sid.encode()).hexdigest()}.json'
        result['inserted' if _atomic(sc,bucket,path,event) else 'existing']+=1
    return result


def pending_game_ids(sc,bucket,sport,market):
    """Get ALL pending snapshots, including games no longer on today's slate."""
    root=_root(sport,market)
    settled={b.name.rsplit('/',1)[-1] for b in sc.bucket(bucket).list_blobs(prefix=f'{root}/settled/')}
    pending=[]
    for blob in sc.bucket(bucket).list_blobs(prefix=f'{root}/pregame/'):
        if blob.name.rsplit('/',1)[-1] in settled:continue
        try:
            row=json.loads(blob.download_as_bytes())
            if row.get('game_id'):pending.append(str(row['game_id']))
        except Exception:continue
    return sorted(set(pending))


def settle(sc,bucket,sport,market,finals,now=None):
    """Grade existing immutable pregame PT snapshots from verified final scores."""
    root=_root(sport,market)
    now=now or dt.datetime.now(dt.timezone.utc)
    if isinstance(now,str): now=dt.datetime.fromisoformat(now.replace('Z','+00:00'))
    if now.tzinfo is None: raise ValueError('settlement timestamp must include timezone')
    by_id={str(r.get('game_id')):_num(r.get('actual_home_margin')) for r in finals}
    n=0; skipped=0
    for blob in sc.bucket(bucket).list_blobs(prefix=f'{root}/pregame/'):
        try: p=json.loads(blob.download_as_bytes())
        except Exception: skipped+=1; continue
        gid=str(p.get('game_id')); actual=by_id.get(gid)
        if actual is None: continue
        try:
            kickoff=dt.datetime.fromisoformat(str(p['game_start']).replace('Z','+00:00'))
            captured=dt.datetime.fromisoformat(str(p['captured_at']).replace('Z','+00:00'))
        except Exception: skipped+=1; continue
        if not (captured<kickoff<=now and p.get('core_candidate') is True):
            skipped+=1; continue
        settle_path=f'{root}/settled/{hashlib.sha256(gid.encode()).hexdigest()}.json'
        obj={'source_tag':TAG,'game_id':gid,'game_start':p['game_start'],'captured_at':p['captured_at'],
             'core_home_direction':p['core_home_direction'],'home_spread':p['home_spread'],
             'incumbent_margin_home':p['incumbent_margin_home'],
             'challenger_margin_home':p['challenger_margin_home'],
             'source_count':p['source_count'],'selection_sha256':p.get('selection_sha256'),
             'actual_home_margin':actual,
             'settled_at':now.isoformat(),'pregame_sha256':_sha(p)}
        if _atomic(sc,bucket,settle_path,obj): n+=1
    return {'inserted':n,'invalid_skipped':skipped}


def prospective_gate(settled_rows, selection_sha256=None):
    paired=[]; weeks=set(); differing=[]; correct_old=[]; correct_new=[]
    for r in settled_rows:
        if not selection_sha256 or r.get('selection_sha256')!=selection_sha256:continue
        # Require first-before-kickoff evidence, not retrospectively fabricated predictions.
        try:
            start=dt.datetime.fromisoformat(str(r['game_start']).replace('Z','+00:00'))
            captured=dt.datetime.fromisoformat(str(r['captured_at']).replace('Z','+00:00'))
        except Exception: continue
        if not captured<start: continue
        a=_num(r.get('actual_home_margin')); old=_num(r.get('incumbent_margin_home'))
        new=_num(r.get('challenger_margin_home')); line=_num(r.get('home_spread'))
        side=_num(r.get('core_home_direction'))
        if None in (a,old,new,line) or side not in {-1.,1.} or int(_num(r.get('source_count')) or 0)<3:
            continue
        # Evaluate whether each signal correctly supports or vetoes the SAME
        # frozen CORE candidate. Ambiguous/push games do not qualify.
        settle=(a+line)*side
        old_edge=(old+line)*side; new_edge=(new+line)*side
        if abs(settle)<1e-9 or abs(old_edge)<MIN_SIGNAL_EDGE_POINTS or abs(new_edge)<MIN_SIGNAL_EDGE_POINTS: continue
        truth=1 if settle>0 else -1
        od=1 if old_edge>0 else -1; nd=1 if new_edge>0 else -1
        o=int(od==truth); n=int(nd==truth)
        paired.append(r); correct_old.append(o);correct_new.append(n)
        wk=_weekly_period(r['game_start'])
        if wk: weeks.add(wk)
        if od!=nd: differing.append((o,n))
    n=len(paired); diff=len(differing)
    gain=(sum(correct_new)-sum(correct_old))/n if n else None
    disagreement_gain=(sum(y-x for x,y in differing)/diff) if diff else None
    ok=(n>=MIN_PROSPECTIVE_PAIRED and len(weeks)>=MIN_PROSPECTIVE_WEEKS and
        diff>=MIN_DISAGREEMENTS and gain is not None and gain>=MIN_PROSPECTIVE_DIRECTION_GAIN and
        disagreement_gain is not None and disagreement_gain>=MIN_DIFFERENCE_DIRECTION_GAIN)
    return {'pass':bool(ok),'paired_n':n,'distinct_weeks':len(weeks),'disagreement_n':diff,
        'new_minus_old_directional_accuracy':gain,'disagreement_accuracy_gain':disagreement_gain,
        'role':'LOCKED_CORE_CANDIDATE_CONFIRM_OR_CONFLICT_ONLY'}


def review(sc,bucket,sport,market,now=None):
    """Automated, idempotent promotion. No automatic demotion or CORE changes."""
    root=_root(sport,market)
    active=read_state(sc,bucket,sport,market)
    if active.get('status')=='PROMOTED':return {'status':'ALREADY_PROMOTED','champion':'FIVE_SOURCE'}
    historic=_json(sc,bucket,f'{root}/current_historical.json') or {}
    locked=read_selection(sc,bucket,sport,market)
    if not locked or (historic.get('historical') or {}).get('selection_sha256')!=locked.get('selection_sha256'):
        return {'status':'KEEP_INCUMBENT','reason':'UNLOCKED_OR_CHANGED_CANDIDATE'}
    h=historical_gate(historic.get('historical'))
    if not h['pass']:
        return {'status':'KEEP_INCUMBENT','historical':h,'prospective':None}
    rows=[]
    for blob in sc.bucket(bucket).list_blobs(prefix=f'{root}/settled/'):
        try: rows.append(json.loads(blob.download_as_bytes()))
        except Exception: continue
    p=prospective_gate(rows,locked["selection_sha256"])
    if not p['pass']:
        return {'status':'KEEP_INCUMBENT','historical':h,'prospective':p}
    when=now or dt.datetime.now(dt.timezone.utc)
    if isinstance(when,dt.datetime):when=when.isoformat()
    state={'source_tag':TAG,'status':'PROMOTED','champion':'FIVE_SOURCE',
           'sport':str(sport).upper(),'market':str(market).upper(),
           'promoted_at':when,'historical':h,'prospective':p,
           'historical_evidence_sha256':historic.get('evidence_sha256'),
           'selection_sha256':locked['selection_sha256'],'selected_sources':locked['sources'],
           'core_model_unchanged':True,'pt_family_vote_cap':1,'automatic_execution':False,
           'vote_rule':'PREGAME_FAIR_MINUS_MARKET_SIGNED_EDGE_MIN2_AND_SOURCE_COUNT_MIN3'}
    created=_atomic(sc,bucket,f'{root}/current_champion.json',state)
    if created:
        _atomic(sc,bucket,f'{root}/promotions/{_sha(state)}.json',state)
        return {'status':'PROMOTED','champion':'FIVE_SOURCE','historical':h,'prospective':p}
    return {'status':'CONCURRENT_REVIEW_OR_ALREADY_PROMOTED','champion':read_state(sc,bucket,sport,market).get('champion')}


def promoted_vote(state, fair_margin_home, home_spread, source_count, selection_sha256=None):
    """Return HOME=+1, AWAY=-1, or 0 for insufficient/neutral PT evidence."""
    if ((state or {}).get('status')!='PROMOTED' or not selection_sha256 or
        selection_sha256!=(state or {}).get('selection_sha256')):return 0
    fair=_num(fair_margin_home); line=_num(home_spread)
    if fair is None or line is None or int(_num(source_count) or 0)<3:return 0
    edge=fair+line
    return 1 if edge>=MIN_SIGNAL_EDGE_POINTS else -1 if edge<=-MIN_SIGNAL_EDGE_POINTS else 0


def _self_test():
    assert not historical_gate({})['pass']
    assert not prospective_gate([])['pass']
    assert promoted_vote({'status':'PROMOTED','selection_sha256':'a'},5.0,-2,5,'a')==1
    assert promoted_vote({'status':'PROMOTED','selection_sha256':'a'},-5.0,-2,5,'a')==-1
    assert promoted_vote({'status':'INCUMBENT'},5.0,-2,5,'a')==0
    # One-sided excellent backtest cannot promote without prospective proof.
    assert historical_gate({'periods_disjoint':True,'asof_max_year':2025,'holdout':{'n':1000,
        'incumbent_mae':10,'challenger_mae':9},'holdout_by_year':{'2024':{'n':500,'gain_points':1},
        '2025':{'n':500,'gain_points':1}},'paired_bootstrap_ci95':[.5,1.5]})['pass'] is True
    return True

if __name__=='__main__':
    print('PASS' if _self_test() else 'FAIL')
