"""Execute frozen 2013-25 named Miner atoms on original-price-attested live rows.

This adapter uses the *existing* ncaaf_research_v2._extended_atoms(...,
for_live=True) through ncaaf_v55_executable_miner_library.evaluate_research_frame.
The frozen provider hash must match exactly. Emitted signals are exclusively
RESEARCH_ONLY, never independent OOS certificates or betting authority.
"""
from __future__ import annotations
from datetime import datetime, timezone


def _clock(v):
    import pandas as pd
    try:
        t=pd.to_datetime(v,utc=True,errors='coerce')
        return None if pd.isna(t) else t.to_pydatetime()
    except (ValueError,TypeError,OverflowError):
        return None


def _num(v):
    import math
    try:
        f=float(v)
        return f if math.isfinite(f) else None
    except (ValueError,TypeError,OverflowError):
        return None


def evaluate_verified_live_features(sides, source_rows, library, *, now=None,
                                    max_quote_age_minutes=15):
    import pandas as pd
    from ncaaf_v55_board import _matchup_teams, _norm_team
    at=_clock(now) if now is not None else datetime.now(timezone.utc)
    if at is None:raise ValueError('LIVE_FEATURE_BRIDGE_INVALID_EVALUATION_CLOCK')
    audit={"status":"NO_LIVE_FEATURE_ROWS","input_rows":0,"eligible_rows":0,
           "evaluated_rows":0,"fired_rows":0,"fired_rule_ids":0,
           "research_event_rows":0,"certified_event_rows":0,
           "rejected":{},"provider_identity":"FROZEN_ATOM_SHA256_REQUIRED",
           "live_bet_authority":0}
    cols=['Game_ID','Game_Start_UTC','Source_Family','Signal_Family',
          'Evidence_Family_Key','Signal_Side','Observed_At_UTC',
          'Observed_Book','Observed_Spread','Observed_Quote_UTC',
          'Qualified_At_UTC','Qualification_Status','OOS_Status',
          'Direction_Verified','Provenance_Status','Registry_Verified',
          'Market','Rule_Atoms']
    empty=pd.DataFrame(columns=cols)
    if not isinstance(library,dict) or not isinstance(sides,pd.DataFrame) or sides.empty:
        audit['status']='MISSING_FROZEN_LIBRARY_OR_PAIRED_SIDES';return empty,audit
    if not isinstance(source_rows,pd.DataFrame) or source_rows.empty:
        return empty,audit
    audit['input_rows']=len(source_rows)
    # Required columns make it impossible to pass unobserved aggregates or
    # reconstruct an original quote timestamp after the fact.
    required={'Home_Team_Norm','Away_Team_Norm','Outcome','Game_Start',
              'Value','Snapshot_Timestamp','V55_Original_Quote_Timestamp',
              'V55_Live_Computed_On_Current_Quote'}
    if not required.issubset(source_rows.columns):
        audit['status']='MISSING_ORIGINAL_LIVE_SOURCE_FIELDS'
        audit['missing_fields']=sorted(required-set(source_rows.columns))
        return empty,audit
    lookup={}
    for _,side in sides.iterrows():
        gid=str(side.get('Game_ID') or '').strip()
        (h,a),ok=_matchup_teams(side)
        team=_norm_team(side.get('Selected_Team'))
        kick=_clock(side.get('Game_Start_UTC'))
        if gid and ok and team in (h,a) and kick is not None:
            lookup.setdefault((h,a),[]).append((side,gid,team,kick))
    rejections={};eligible=[];targets=[]
    def reject(code):rejections[code]=rejections.get(code,0)+1
    for _,rec in source_rows.iterrows():
        if rec.get('V55_Live_Computed_On_Current_Quote') is not True:
            reject('NOT_COMPUTED_ON_CURRENT_QUOTE');continue
        h=_norm_team(rec.get('Home_Team_Norm'))
        a=_norm_team(rec.get('Away_Team_Norm'))
        side_team=_norm_team(rec.get('Outcome'))
        if not h or not a or h==a or side_team not in (h,a):
            reject('INVALID_ORIENTED_SIDE');continue
        source_kick=_clock(rec.get('Game_Start'))
        evaluated=_clock(rec.get('Snapshot_Timestamp'))
        original=_clock(rec.get('V55_Original_Quote_Timestamp'))
        line=_num(rec.get('Value'))
        book=str(rec.get('Bookmaker') or rec.get('Book') or '').lower().strip()
        if not book or line is None or any(x is None for x in (source_kick,evaluated,original)):
            reject('MISSING_SOURCE_PRICE_OR_CLOCK');continue
        if not (original <= evaluated <= at < source_kick and
                (at-evaluated).total_seconds()<=300 and
                (evaluated-original).total_seconds()<=40*60):
            reject('INVALID_PREGAME_SOURCE_VINTAGE');continue
        same=[]
        for s,gid,team,kick in lookup.get((h,a),[]):
            q=_clock(s.get('Backend_Quote_Timestamp'))
            candidate_line=_num(s.get('Current_Spread'))
            if q is None or candidate_line is None:continue
            if not (abs((kick-source_kick).total_seconds())<=15*60 and
                    original<=q<=evaluated and 0<=(at-q).total_seconds()/60<=max_quote_age_minutes):
                continue
            if str(s.get('Backend_Quote_Book') or '').lower().strip()!=book:continue
            if abs(line-(candidate_line if side_team==team else -candidate_line))>.001:continue
            same.append((s,gid,team,kick))
        # Every source row must match ONE physical game and one source side.
        source_hits=[x for x in same if x[2]==side_team]
        if len(source_hits)!=1 or len({x[1] for x in same})!=1:
            reject('NO_UNIQUE_SAME_BOOK_SOURCE_SIDE');continue
        chosen,gid,_,kick=source_hits[0]
        eligible.append(rec)
        targets.append((gid,h,a,side_team,book,line,original,evaluated,kick))
    audit['eligible_rows']=len(eligible);audit['rejected']=rejections
    if not eligible:
        audit['status']='NO_VERIFIED_SAME_BOOK_LIVE_FEATURE_ROWS';return empty,audit
    # The frozen provider SHA guard is enforced inside the reusable evaluator.
    # All missing atoms remain UNKNOWN; a missing field is never an observed 0.
    try:
        from ncaaf_v55_executable_miner_library import evaluate_research_frame
        import utils
        source=pd.DataFrame.from_records(eligible).reset_index(drop=True)
        results,diag=evaluate_research_frame(
            library,source,dashboard_module=utils,market='spreads')
        audit['evaluated_rows']=len(source)
        audit['atom_evaluator']=diag
    except Exception as exc:
        audit['status']='FROZEN_FULL_ATOM_PROVIDER_HELD'
        audit['provider_error']=type(exc).__name__+':'+str(exc)[:200]
        return empty,audit
    by_side={}
    for _,s in sides.iterrows():
        gid=str(s.get('Game_ID') or '').strip()
        team=_norm_team(s.get('Selected_Team'))
        if gid and team:by_side[(gid,team)]=s
    rule_map={r['rule_id']:r for r in library.get('rules',[]) if r.get('market')=='spreads'}
    emitted={}
    for firing in results:
        try:i=int(firing['row_index'])
        except (KeyError,ValueError,TypeError):continue
        if not 0<=i<len(targets):continue
        gid,h,a,source_team,book,line,orig,obs,kick=targets[i]
        rid=str(firing.get('rule_id') or '')
        rule=rule_map.get(rid)
        if rule is None:continue
        direction=rule.get('direction')
        target=source_team if direction=='PLAY_ON' else (a if source_team==h else h) if direction=='FADE' else None
        if target not in (h,a):continue
        target_side=by_side.get((gid,target))
        if target_side is None:continue
        target_line=_num(target_side.get('Current_Spread'))
        if (str(target_side.get('Backend_Quote_Book') or '').lower().strip()!=book
                or target_line is None or abs(target_line-(line if target==source_team else -line))>.001):continue
        key=(gid,rid,target)
        emitted[key]={
            'Game_ID':gid,'Game_Start_UTC':kick.isoformat(),
            'Source_Family':'SYSTEM_MINER','Signal_Family':rid,
            'Evidence_Family_Key':'FROZEN_ATOMS::'+'|'.join(sorted(rule['atoms'])),
            'Signal_Side':target,'Observed_At_UTC':obs.isoformat(),
            'Observed_Book':book,'Observed_Spread':target_line,
            'Observed_Quote_UTC':orig.isoformat(),
            'Qualified_At_UTC':None,'Qualification_Status':'RESEARCH_ONLY',
            'OOS_Status':'NOT_CERTIFIED','Direction_Verified':True,
            'Provenance_Status':'PRE_GAME_VERIFIED_LIVE_RECALCULATED',
            'Registry_Verified':False,'Market':'spreads',
            'Rule_Atoms':'|'.join(rule['atoms']),
        }
    audit['fired_rows']=len(results)
    audit['fired_rule_ids']=len({v['Signal_Family'] for v in emitted.values()})
    audit['research_event_rows']=len(emitted)
    audit['status']='PASS_FROZEN_FULL_ATOM_LIVE_RESEARCH' if emitted else 'PASS_NO_FROZEN_FULL_ATOM_FIRINGS'
    return pd.DataFrame.from_records(list(emitted.values()),columns=cols),audit
