"""Attest independent non-Miner research signals against original live quote rows.

Expected column on an existing live system DataFrame:
 V55_Independent_Shadow_Signals_JSON: list of records with
 source_family (BIG_AL/PATHI/PREDICTION_TRACKER/STAT/MARKET_RICH),
 signal_id, target, observed_at_utc, observed_quote_utc.

The producer must have evaluated the original current sportsbook quote. No
signals are inferred from descriptive columns, ratings, or historical results.
"""
from __future__ import annotations
import json
from collections import Counter
from datetime import datetime, timezone

FAMILIES = frozenset({'BIG_AL','PATHI','PREDICTION_TRACKER','STAT','MARKET_RICH'})
FIELD = 'V55_Independent_Shadow_Signals_JSON'

def build_other_shadow_events(live_rows, two_sided, *, now=None):
    import pandas as pd
    from ncaaf_v55_board import _matchup_teams, _norm_team, _price, _timestamp
    clock = _timestamp(now if now is not None else datetime.now(timezone.utc))
    audit = {'status':'SHADOW_ONLY', 'source_column_present':False,
             'structured_signals':0, 'accepted':0, 'rejected':{},
             'by_family':{}, 'historical_qualification':False,'authority':0}
    empty = pd.DataFrame()
    if not isinstance(live_rows,pd.DataFrame) or not isinstance(two_sided,pd.DataFrame) or live_rows.empty or two_sided.empty:
        audit['status']='NO_SOURCE_OR_TWO_SIDED_ROWS';return empty,audit
    if FIELD not in live_rows.columns:
        audit['status']='STRUCTURED_OTHER_BRAIN_FEED_NOT_CONNECTED';return empty,audit
    audit['source_column_present']=True
    rejected=Counter(); accepted={}
    sides={}
    for _,s in two_sided.iterrows():
        gid=str(s.get('Game_ID') or '').strip()
        team=_norm_team(s.get('Selected_Team'))
        (home,away),oriented=_matchup_teams(s)
        if not gid or not oriented or team not in (home,away):continue
        sides.setdefault((home,away),[]).append((s,gid,team))
    for _,r in live_rows.iterrows():
        raw=r.get(FIELD)
        if raw is None or (isinstance(raw,float) and pd.isna(raw)) or raw=='':continue
        try:signals=json.loads(raw) if isinstance(raw,str) else raw
        except (TypeError,ValueError):rejected['INVALID_JSON']+=1;continue
        if not isinstance(signals,list):rejected['NOT_LIST']+=1;continue
        for e in signals:
            audit['structured_signals']+=1
            if not isinstance(e,dict):rejected['NOT_OBJECT']+=1;continue
            family=str(e.get('source_family') or '').upper().strip()
            mid=str(e.get('signal_id') or '').strip()
            if family not in FAMILIES or not mid:rejected['FAMILY_OR_SYSTEM_ID']+=1;continue
            if r.get('V55_Live_Computed_On_Current_Quote') is not True:
                rejected['NOT_LIVE_COMPUTED_ON_QUOTE']+=1;continue
            home=_norm_team(r.get('Home_Team_Norm'));away=_norm_team(r.get('Away_Team_Norm'))
            target=_norm_team(e.get('target'))
            source_side=_norm_team(r.get('Outcome'))
            if target not in (home,away) or source_side not in (home,away):rejected['DIRECTION_NOT_VERIFIED']+=1;continue
            seen=_timestamp(e.get('observed_at_utc')); row_seen=_timestamp(r.get('Snapshot_Timestamp'))
            original=_timestamp(e.get('observed_quote_utc')); source_quote=_timestamp(r.get('V55_Original_Quote_Timestamp'))
            kick=_timestamp(r.get('Game_Start'))
            if any(pd.isna(x) for x in (seen,row_seen,original,source_quote,kick,clock)):
                rejected['MISSING_TIMESTAMP']+=1;continue
            if not (original==source_quote and original<=row_seen<=seen<=clock<kick
                    and (clock-seen).total_seconds()<=300 and (seen-original).total_seconds()<=900):
                rejected['VINTAGE_OR_AGE']+=1;continue
            book=str(r.get('Bookmaker') or r.get('Book') or '').lower().strip()
            spread=_price(r.get('Value'))
            if not book or spread is None:rejected['SOURCE_BOOK_OR_SPREAD']+=1;continue
            found=False
            for s,gid,team in sides.get((home,away),[]):
                if target!=team:continue
                sk=_timestamp(s.get('Game_Start_UTC'))
                qt=_timestamp(s.get('Backend_Quote_Timestamp'))
                ss=_price(s.get('Current_Spread'))
                sb=str(s.get('Backend_Quote_Book') or '').lower().strip()
                if (pd.isna(sk) or pd.isna(qt) or abs((sk-kick).total_seconds())>900
                    or not source_quote<=qt<=clock<sk
                    or (clock-qt).total_seconds()>900
                    or (qt-source_quote).total_seconds()>900
                    or sb!=book or ss is None
                    or abs(spread-(ss if source_side==team else -ss))>.001):continue
                key=(gid,team,family,mid)
                val={'Game_ID':gid,'Signal_Side':team,'Source_Family':family,
                     'Signal_Family':mid,'Observed_At_UTC':seen.isoformat(),
                     'Observed_Quote_UTC':source_quote.isoformat(),
                     'Observed_Book':book,'Observed_Spread':ss,
                     'Game_Start_UTC':sk.isoformat(),
                     'Research_Evidence_State':'UNQUALIFIED_PROSPECTIVE_SHADOW',
                     'V55_Evidence_Qualified':False}
                if key not in accepted or val['Observed_At_UTC']>accepted[key]['Observed_At_UTC']:
                    accepted[key]=val
                found=True
            if not found:rejected['NO_EXACT_PAIRED_EXECUTABLE_QUOTE']+=1
    audit['accepted']=len(accepted)
    audit['rejected']=dict(rejected)
    audit['by_family']=dict(Counter(v['Source_Family'] for v in accepted.values()))
    if audit['accepted']==0:audit['status']='NO_VERIFIED_DIRECTIONAL_OTHER_BRAIN_SIGNALS'
    return pd.DataFrame(list(accepted.values())),audit
