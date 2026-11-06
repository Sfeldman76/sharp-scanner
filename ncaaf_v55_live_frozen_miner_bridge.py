"""Evaluate *observable* frozen unified Miner R2 predicates on live V5.5 spread sides.

This module does NOT certify historical returns or authorize betting.  It emits
research-only, timestamped events from original paired sportsbook quotes, and
explicitly reports atoms it cannot evaluate.  No guessing missing CFBD, PT,
Pathi, personnel, or market-history inputs.  All historical rule IDs unchanged.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import math

SUPPORTED_SPREAD_ATOMS = frozenset({
    'CURRENT_DOG', 'CURRENT_FAVORITE', 'DOG_6P5_PLUS', 'DOG_10_PLUS',
    'DOG_7_PLUS', 'FAV_14_99',
})


def _num(v):
    try:
        n=float(v)
        return n if math.isfinite(n) else None
    except (ValueError,TypeError):
        return None


def _dt(v):
    import pandas as pd
    p=pd.to_datetime(v,utc=True,errors='coerce')
    return None if pd.isna(p) else p.to_pydatetime()


def _atom(name, spread):
    if name == 'CURRENT_DOG':return spread > 0
    if name == 'CURRENT_FAVORITE':return spread < 0
    if name == 'DOG_6P5_PLUS':return spread >= 6.5
    if name == 'DOG_7_PLUS':return spread >= 7.0
    if name == 'DOG_10_PLUS':return spread >= 10.0
    if name == 'FAV_14_99':return -99.0 <= spread <= -14.0
    return None


def evaluate_quote_atoms(sides, miner_rules, *, now=None, max_quote_age_minutes=15):
    """Return (research_events_dataframe, audit) from existing two-sided scan.

    Only basic *quote-derived* predicates are understood, and then only at the
    exact same bookmaker, line, game and side passed by the V5.5 backend.
    All unknown atom conditions fail closed. Rules trained on CFBD / historical
    features must be evaluated by an appropriately timestamped feature provider.
    """
    import pandas as pd
    import ncaaf_v55_board as board
    at=_dt(now) if now is not None else datetime.now(timezone.utc)
    if at is None:raise ValueError('UNIFIED_MINER_INVALID_CLOCK')
    diag={
        'status':'LIVE_MONITORING_STARTED_NO_CERTIFIED_FIRE',
        'execution_mode':'LIVE_MONITORING',
        'live_tracking_enabled_rules':sum(r.get('live_tracking_enabled') is True for r in miner_rules.values()),
        'bet_authority_eligible_rules':sum(r.get('historical_oos_certified') is True and r.get('live_authority_eligible') is True for r in miner_rules.values()),
        'frozen_rule_count':len(miner_rules),
        'spread_rules':sum(r.get('market')=='spreads' for r in miner_rules.values()),
        'quote_atom_rules_evaluable':0,
        'rules_with_missing_atom_providers':0,
        'unavailable_atoms':{},
        'side_rows':len(sides),
        'fresh_side_rows':0,
        'fired_rule_ids':0,
        'fired_game_sides':0,
        'research_event_rows':0,
        'certified_event_rows':0,
    }
    cols=('Game_ID','Game_Start_UTC','Source_Family','Signal_Family',
          'Evidence_Family_Key','Signal_Side','Observed_At_UTC',
          'Observed_Book','Observed_Spread','Observed_Quote_UTC',
          'Qualified_At_UTC','Qualification_Status','OOS_Status',
          'Direction_Verified','Provenance_Status','Registry_Verified',
          'Market','Rule_Atoms')
    if not isinstance(sides,pd.DataFrame) or sides.empty or not miner_rules:
        return pd.DataFrame(columns=cols),diag
    rules=[]
    missing=Counter()
    for rid, r in sorted(miner_rules.items()):
        if r.get('market')!='spreads':continue
        atoms=r.get('atoms')
        direction=str(r.get('direction') or '').upper()
        if not isinstance(atoms,list) or not atoms or direction not in ('PLAY_ON','FADE'):
            diag['rules_with_missing_atom_providers']+=1
            missing['RULE_SHAPE_UNSUPPORTED']+=1
            continue
        unmet=[a for a in atoms if a not in SUPPORTED_SPREAD_ATOMS]
        if unmet:
            diag['rules_with_missing_atom_providers']+=1
            missing.update(set(str(x) for x in unmet))
            continue
        diag['quote_atom_rules_evaluable']+=1
        rules.append((rid, r, atoms, direction))
    diag['unavailable_atoms']=dict(missing.most_common(30))
    rows=[]; fired=set(); game_sides=set()
    # Require the exact TARGET-side price and its own original timestamp, even
    # when a FADE mechanism fires on the opposite team's source-side features.
    paired = {}
    for _, sr in sides.iterrows():
        gid = str(sr.get('Game_ID') or '').strip()
        selected = board._norm_team(sr.get('Selected_Team'))
        if gid and selected:
            paired[(gid, selected)] = sr
    for _,side in sides.iterrows():
        gid=str(side.get('Game_ID') or '').strip()
        book=str(side.get('Backend_Quote_Book') or '').strip()
        pick=str(side.get('Selected_Team') or '').strip()
        spread=_num(side.get('Current_Spread'))
        price=_num(side.get('Odds_American'))
        q=_dt(side.get('Backend_Quote_Timestamp'))
        kick=_dt(side.get('Game_Start_UTC'))
        if (not gid or not book or not pick or spread is None or price is None
            or q is None or kick is None or not q<=at<kick
            or not 0<=(at-q).total_seconds()/60<=max_quote_age_minutes
            or not str(side.get('Quote_State') or '').startswith('FRESH_')):
            continue
        (home,away),oriented=board._matchup_teams(side)
        selected=board._norm_team(pick)
        if not oriented or selected not in (home,away):continue
        diag['fresh_side_rows']+=1
        for rid,r,atoms,direction in rules:
            if not all(_atom(a,spread) is True for a in atoms):continue
            target=selected if direction=='PLAY_ON' else (away if selected==home else home)
            if target not in (home,away):continue
            target_row=paired.get((gid,target))
            if target_row is None:continue
            target_book=str(target_row.get('Backend_Quote_Book') or '').strip()
            target_spread=_num(target_row.get('Current_Spread'))
            target_q=_dt(target_row.get('Backend_Quote_Timestamp'))
            if (target_book.lower()!=book.lower() or target_spread is None or
                target_q is None or abs(abs(target_spread)-abs(spread))>0.001 or
                not 0<=(at-target_q).total_seconds()/60<=max_quote_age_minutes):
                continue
            key=(gid,rid,target)
            if key in fired:continue
            fired.add(key)
            game_sides.add((gid,target))
            # Explicitly RESEARCH_ONLY. Even a historically profitable rule may
            # not become a vote before its independent OOS/vintage certification.
            rows.append({
                'Game_ID':gid,
                'Game_Start_UTC':kick.isoformat(),
                'Source_Family':'SYSTEM_MINER',
                'Signal_Family':rid,
                'Evidence_Family_Key':'MINER_QUOTE_ATOMS::'+'|'.join(sorted(atoms)),
                'Signal_Side':target,
                'Observed_At_UTC':at.isoformat(),
                'Observed_Book':target_book,
                'Observed_Spread':target_spread,
                'Observed_Quote_UTC':target_q.isoformat(),
                'Qualified_At_UTC':None,
                'Qualification_Status':'RESEARCH_ONLY',
                'OOS_Status':'NOT_CERTIFIED',
                'Direction_Verified':True,
                'Provenance_Status':'PRE_GAME_VERIFIED_LIVE_RECALCULATED',
                'Registry_Verified':False,
                'Market':'spreads',
                'Rule_Atoms':'|'.join(atoms),
                'Live_Evaluation_Status':'LIVE_OBSERVED_RULE_TRIGGER',
                'Bet_Certification_Status':'UNCERTIFIED_FROZEN_RULE',
            })
    diag['fired_rule_ids']=len({x['Signal_Family'] for x in rows})
    diag['fired_game_sides']=len(game_sides)
    diag['research_event_rows']=len(rows)
    diag['status']='LIVE_FIRINGS_OBSERVED_OOS_UNCERTIFIED' if rows else 'LIVE_SCANNED_NO_SUPPORTED_RULE_FIRINGS'
    return pd.DataFrame.from_records(rows,columns=cols),diag
