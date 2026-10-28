"""NCAAF V5.5 production evidence observability (read-only, zero voting authority).

Pure audit: compares what the LIVE engines computed with what the independent
2013-25 frozen OOS registry actually permitted.  Never promotes raw overlays,
PT forecasts, descriptive Pathi key rules, or 2026 prospective results.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any


def _text(v: Any) -> str:
    if v is None: return ''
    try:
        import pandas as pd
        if not isinstance(v, (dict, list, tuple)) and bool(pd.isna(v)): return ''
    except (ValueError, TypeError, ImportError): pass
    s = str(v).strip()
    return '' if s.lower() in ('none', 'nan', '<na>', 'null', '—', 'not_published') else s


def _integer(v: Any) -> int:
    try:
        f = float(v)
        return int(f) if math.isfinite(f) else 0
    except (ValueError, TypeError, OverflowError): return 0


def _details(v: Any) -> list[dict]:
    if isinstance(v, str):
        try: v = json.loads(v)
        except (ValueError, TypeError): return []
    return [a for a in v if isinstance(a, dict)] if isinstance(v, list) else []


def audit_v55_production_evidence(board, live_rows, *,
                                   miner_registry=None, other_registry=None,
                                   other_registry_problem=None,
                                   miner_bridge=None, other_bridge=None):
    """Return compact health report and per-game evidence trace; no I/O/writes.

    A published BET is checked against an actual FROZEN system ID, not a
    preexisting count.  Readiness is not a new authorization rule; the
    production board still controls bet decisions.
    """
    import pandas as pd
    if not isinstance(board, pd.DataFrame):
        raise TypeError('V55_EVIDENCE_AUDIT_REQUIRES_BOARD_DATAFRAME')
    live = live_rows if isinstance(live_rows, pd.DataFrame) else pd.DataFrame()
    raw = dict(bigal_exact_quote_rows=0, pathi_exact_quote_rows=0,
               pathi_context_quote_rows=0, pt_forecast_quote_rows=0,
               stat_directional_vote_quote_rows=0, live_quote_rows=len(live))
    if not live.empty:
        signals = live.get('System_Signals_Text', pd.Series('',index=live.index)).fillna('').astype(str)
        raw['bigal_exact_quote_rows'] = int(signals.str.contains('BIG AL ', regex=False).sum())
        raw['pathi_exact_quote_rows'] = int(signals.str.contains('PATHI ', regex=False).sum())
        col = live.get('Pathi_Active_Text', pd.Series('', index=live.index)).fillna('').astype(str)
        raw['pathi_context_quote_rows'] = int(col.str.contains('FOOTBALL RESEARCH CONTEXT ONLY', regex=False).sum())
        pred = pd.to_numeric(live.get('NCAAF_PT_Five_Source_Count',pd.Series(index=live.index,dtype=float)), errors='coerce')
        raw['pt_forecast_quote_rows'] = int(pred.ge(1).sum())
        for src in ('NCAAF_STAT_Votes','STAT_Votes'):
            if src in live:
                raw['stat_directional_vote_quote_rows'] += int(live[src].map(lambda v: len(_details(v))>0).sum())

    registered = (other_registry or {}).get('rules', {}) if isinstance(other_registry,dict) else {}
    miner_rules = (miner_registry or {}).get('mechanisms', {}) if isinstance(miner_registry,dict) else {}
    details_by_family = Counter()
    decisions = Counter()
    problems = []
    trace = []
    for _,row in board.iterrows():
        decision = _text(row.get('V55_Bet_Decision'))
        decisions[decision or 'NO_DECISION'] += 1
        if decision not in ('BET', 'STRONG BET', 'CANDIDATE', 'VETO'):
            continue
        supports = _integer(row.get('Confirm_Count'))
        opposes = _integer(row.get('Oppose_Count'))
        evidence = _details(row.get('V55_Qualified_Evidence_Detail_JSON'))
        qualified = [v for v in evidence if v.get('historical_source')=='FROZEN_V55_REGISTRY_MATCH']
        for v in qualified: details_by_family[_text(v.get('source'))] += 1
        source_keys = sorted({_text(v.get('source')) for v in qualified if _text(v.get('source'))})
        named = sorted({_text(v.get('system_id')) for v in qualified if _text(v.get('system_id'))})
        names = _text(row.get('V55_Independent_Support_Keys'))
        raw_ba = _text(row.get('V55_Raw_BigAl'))
        raw_pa = _text(row.get('V55_Raw_Pathi'))
        raw_pt = _text(row.get('V55_Raw_PT'))
        raw_stat = _text(row.get('V55_Raw_STAT'))
        reasons = []
        if supports==0:reasons.append('NO_QUALIFIED_INDEPENDENT_SUPPORT')
        if opposes>0:reasons.append('QUALIFIED_OPPOSING_EVIDENCE_VETO')
        if supports==1 and opposes==0:reasons.append('ONLY_ONE_INDEPENDENT_SOURCE_FAMILY')
        if raw_ba and not registered:reasons.append('BIG_AL_RAW_ONLY__FROZEN_OOS_REGISTRY_MISSING')
        if raw_pa and not registered:reasons.append('PATHI_RAW_OR_CONTEXT_ONLY__FROZEN_OOS_REGISTRY_MISSING')
        if raw_pt and not registered:reasons.append('PT_RAW_FORECAST_ONLY__SEPARATE_QUALIFICATION_MISSING')
        if raw_stat and raw_stat.upper().startswith('ACTIVE') and not registered:
            reasons.append('STAT_RAW_ONLY__SEPARATE_QUALIFICATION_MISSING')
        if decision in ('BET','STRONG BET'):
            if supports < (2 if decision=='STRONG BET' else 1):
                problems.append({'game_id':_text(row.get('Game_ID')), 'failure':'BET_WITHOUT_REQUIRED_FAMILY_COUNT'})
            if opposes>0:
                problems.append({'game_id':_text(row.get('Game_ID')), 'failure':'BET_IGNORED_VETO'})
            if not names or not qualified:
                problems.append({'game_id':_text(row.get('Game_ID')), 'failure':'BET_MISSING_FROZEN_EVIDENCE_DETAILS'})
            if _integer(row.get('V55_Bet_Authority'))!=1:
                problems.append({'game_id':_text(row.get('Game_ID')), 'failure':'BET_NOT_AUTHORIZED'})
        trace.append({'game_id':_text(row.get('Game_ID')),
                      'matchup':_text(row.get('Matchup')),
                      'selected':_text(row.get('Selected_Team')),
                      'decision':decision,
                      'quote_book':_text(row.get('Backend_Quote_Book')),
                      'confirmed_independent_families':supports,
                      'opposing_independent_families':opposes,
                      'qualified_brains':source_keys,
                      'qualified_system_ids':named[:12],
                      'raw_big_al_present':bool(raw_ba),
                      'raw_pathi_present':bool(raw_pa),
                      'raw_pt_present':bool(raw_pt),
                      'raw_stat_active':raw_stat.upper().startswith('ACTIVE'),
                      'blockers':list(dict.fromkeys(reasons))[:8]})
    trace.sort(key=lambda r: (0 if r['decision'] in ('STRONG BET','BET') else 1,
                              0 if r['decision']=='VETO' else 1,
                              r['game_id']))
    blockers = Counter(b for item in trace for b in item['blockers'])
    report = {
        'status':'FAIL_CLOSED_EVIDENCE_INTEGRITY_ALERT' if problems else 'PASS_AUDIT_ONLY',
        'policy':'RESEARCH_OOS_2013_2025__NO_2026_SELECTION_INFLUENCE',
        'changes_betting_authority':False,
        'board_rows':len(board),
        'decision_rows':dict(decisions),
        'audit_rows':len(trace),
        'live_sources':raw,
        'frozen_miner_rules':len(miner_rules),
        'frozen_nonminer_rules':len(registered),
        'nonminer_registry_status':_text(other_registry_problem) or 'PASS',
        'miner_verified_event_count':_integer((miner_bridge or {}).get('qualified_events')),
        'other_verified_event_count':_integer((other_bridge or {}).get('qualified_events')),
        'verified_system_firings_by_brain':dict(details_by_family),
        'blocking_reason_game_counts':dict(blockers),
        'violations':problems[:10],
        'trace':trace[:60],
    }
    return report
