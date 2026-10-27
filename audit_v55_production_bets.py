"""Read-only production evidence audit for the latest NCAAF V5.5 board.

Usage: python3 audit_v55_production_bets.py
No writes, no model refits, no authority changes.
"""
from __future__ import annotations
import json
from collections import Counter

def main():
    import ncaaf_v55_board as v55
    import pandas as pd
    board,meta=v55.read_published_latest(max_market_age_minutes=180)
    print('[NCAAF-V55-PUBLISHED-BOARD]',json.dumps({
        'snapshot_id':meta.get('snapshot_id'),
        'published_at_utc':str(meta.get('published_at_utc')),
        'published_age_minutes':meta.get('published_age_minutes'),
        'publication_status':meta.get('publication_status'),
        'priced_games':meta.get('priced_games'),
        'authorized_bets_current':meta.get('v55_authorized_bet_rows'),
        'strong_bets_current':meta.get('v55_strong_bet_rows'),
        'research_registry_sha256':(meta.get('miner_live_bridge') or {}).get('research_registry_sha256'),
        'miner_qualified_events':(meta.get('miner_live_bridge') or {}).get('qualified_events'),
        'other_brain_registry_status':meta.get('other_brain_registry_status'),
        'other_brain_qualified_events':(meta.get('long_history_other_brain_bridge') or {}).get('qualified_events'),
        'brain_evidence_state':meta.get('evidence_state'),
        'policy_probability_basis':meta.get('v55_authority_probability_basis'),
    },default=str,sort_keys=True))
    if board.empty:
        print('NO_BOARD_ROWS'); return
    summary=Counter(board.get('V55_Bet_Decision',pd.Series(dtype='object')).fillna('UNKNOWN').astype(str))
    print('[NCAAF-V55-DECISION-COUNTS]',json.dumps(dict(summary),sort_keys=True))
    actions=board.loc[board.get('V55_Bet_Authority',pd.Series(0,index=board.index)).eq(1)]
    if actions.empty:
        print('[NCAAF-V55-ACTIVE-BETS] NONE — quote may have expired or no verified supporters')
    for _,r in actions.iterrows():
        try:ev=json.loads(r.get('V55_Qualified_Evidence_Detail_JSON') or '[]')
        except (TypeError,ValueError):ev=[]
        if not isinstance(ev,list):ev=[]
        frozen=[e for e in ev if isinstance(e,dict) and e.get('historical_source')=='FROZEN_V55_REGISTRY_MATCH']
        details={'game_id':r.get('Game_ID'), 'matchup':r.get('Matchup'),
            'selected':r.get('Selected_Team'),'decision':r.get('V55_Bet_Decision'),
            'book':r.get('Backend_Quote_Book'),'spread':r.get('Current_Spread'),
            'american_odds':r.get('Odds_American'),
            'core_model':r.get('CORE_Model'),'core_margin':r.get('CORE_Projected_Home_Margin'),
            'provisional_policy_ev':r.get('V55_Policy_EV'),
            'confirmations':r.get('Confirm_Count'),'oppositions':r.get('Oppose_Count'),
            'independent_support_keys':r.get('V55_Independent_Support_Keys'),
            'support_families':r.get('Confirm_Families'),
            'system_ids':[e.get('system_id') for e in frozen],
            'systems':[{'source':e.get('source'),'system_id':e.get('system_id'),
                        'direction':e.get('direction'), 'evidence_family_key':e.get('evidence_family_key'),
                        'historical_n':e.get('historical_n'),'historical_ats_rate':e.get('historical_ats_rate')}
                       for e in frozen],
            'quote_timestamp':str(r.get('Backend_Quote_Timestamp')),
            'calibrated':meta.get('calibration_state'),
        }
        print('[NCAAF-V55-ACTIVE-BET-EVIDENCE]',json.dumps(details,default=str,sort_keys=True))
        count=int(pd.to_numeric(r.get('Confirm_Count'),errors='coerce') or 0)
        if count < (2 if r.get('V55_Bet_Decision')=='STRONG BET' else 1) or not frozen:
            raise RuntimeError('ACTIVE_BET_LACKS_VISIBLE_FROZEN_INDEPENDENT_EVIDENCE')
    # Also show which mechanisms are actually opposing or supporting any
    # selected live game, even if the bet policy has not promoted it.
    relevant=board.loc[pd.to_numeric(board.get('Confirm_Count',pd.Series(0,index=board.index)),errors='coerce').fillna(0).gt(0) |
                       pd.to_numeric(board.get('Oppose_Count',pd.Series(0,index=board.index)),errors='coerce').fillna(0).gt(0)]
    print('[NCAAF-V55-EVIDENCE-ACTIVE-GAMES]',json.dumps({'games':len(relevant),
        'support_games':int((pd.to_numeric(relevant.get('Confirm_Count',pd.Series(0,index=relevant.index)),errors='coerce').fillna(0)>0).sum()),
        'veto_games':int((pd.to_numeric(relevant.get('Oppose_Count',pd.Series(0,index=relevant.index)),errors='coerce').fillna(0)>0).sum())}))

if __name__=='__main__': main()
