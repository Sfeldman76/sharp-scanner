"""NCAAF V5.5: NFL-identical 19-column betting BOARD, NCAAF backend evidence.

Drop-in replacement for the module already imported by the installed V3 route.
The UI only READS ncaaf_v55_board.read_published_latest(). No V1 inference,
no book lookups, no rule scoring, no user-side promotion of bets.
"""
from __future__ import annotations

import json
import math
from collections import Counter

NFL_COLUMNS = [
    'Game Time', 'Matchup',
    'Spr Action', 'Spr Pick', 'Spr Prob', 'Spr Model Edge',
    'H2H Action', 'H2H Pick', 'H2H Prob', 'H2H Model Edge',
    'Tot Action', 'Tot Pick', 'Tot Prob', 'Tot Model Edge',
    'Production Plays', 'Pathi State', 'Pathi Overlay', 'Edge Sources', 'System Trigger',
]
BET_ACTIONS = {'BET', 'STRONG BET'}
MISSING = {'', '—', 'NONE', 'NAN', 'NULL', 'NAT', '<NA>', 'NOT_PUBLISHED', 'NOT_EVALUABLE', 'NO CORE PICK'}


def text(value):
    if value is None:
        return ''
    try:
        import pandas as pd
        missing = pd.isna(value)
        if isinstance(missing, (bool, type(pd.NA))) and missing is True:
            return ''
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return '' if s.upper() in MISSING else s


def num(value):
    try:
        v = float(text(value).replace('%', '').replace(',', '').replace('pts', '').strip())
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def is_auth(value):
    return text(value).lower() in {'1', '1.0', 'true'}


def pct(value):
    n = num(value)
    if n is None or n < 0:
        return '—'
    if n <= 1:
        n *= 100
    return f'{n:.1f}%' if n <= 100 else '—'


def families(value):
    import re
    if isinstance(value, list):
        seq = value
    else:
        seq = re.split(r'[|,;]+', text(value))
    return sorted(set(x.strip().upper() for x in map(str, seq) if text(x)))


def candidate_info(value):
    if isinstance(value, list):
        candidates = value
    elif isinstance(value, str) and value.strip():
        try:
            candidates = json.loads(value)
        except (ValueError, TypeError):
            return [], 'CANDIDATE_JSON_INVALID'
    else:
        return [], None
    if not isinstance(candidates, list):
        return [], 'CANDIDATE_JSON_NOT_LIST'
    return [x for x in candidates if isinstance(x, dict)], None


def _quote_current(row):
    return text(row.get('Quote_State')).startswith('FRESH_')


def _validated_bet(row):
    """Authority must be granted on this very published game and selected side."""
    if not (is_auth(row.get('V55_Bet_Authority')) and
            text(row.get('V55_Bet_Decision')).upper() in BET_ACTIONS and
            _quote_current(row) and text(row.get('Selected_Team')) and
            text(row.get('Backend_Quote_Book'))):
        return False
    spread, odds = num(row.get('Current_Spread')), num(row.get('Odds_American'))
    return spread is not None and odds is not None and (odds <= -100 or odds >= 100)


def _side_label(row):
    selected = text(row.get('Selected_Team'))
    if not selected or not _quote_current(row):
        return '—'
    spread, price = num(row.get('Current_Spread')), num(row.get('Odds_American'))
    # The chosen SIDE must have its own published price, not the opponent's quote.
    if spread is None or price is None:
        return selected + ' (price unavailable)'
    return f'{selected} {spread:+g} ({price:+.0f})'


def nfl_shaped_game_board(weekly, *, now=None):
    """Return exactly NFL_COLUMNS, one game per row, plus diagnostic detail.

    Only the backend may decide BET, STRONG BET, candidate, or conflict.
    Research candidates without published selections remain visible in details,
    never converted by this renderer into actual bets.
    """
    import pandas as pd
    if not isinstance(weekly, pd.DataFrame):
        raise TypeError('PUBLISHED_WEEKLY_NOT_DATAFRAME')
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    now = now.tz_localize('UTC') if now.tzinfo is None else now.tz_convert('UTC')
    board, detail, candidates_detail, identity = [], [], [], set()
    diagnostics = Counter()
    for _, row in weekly.iterrows():
        kick = pd.to_datetime(row.get('Game_Start_UTC'), utc=True, errors='coerce')
        if pd.notna(kick) and kick <= now:
            continue
        match = text(row.get('Matchup')) or '—'
        gid = text(row.get('Game_ID'))
        key = gid or (match, str(kick))
        if key in identity:
            diagnostics['duplicate_game_rows_suppressed'] += 1
            # Duplicated physical-game publications are not another bet.
            # The caller must inspect backend identity collisions.
            continue
        identity.add(key)
        fresh = _quote_current(row)
        bet = _validated_bet(row)
        published_action = text(row.get('V55_Bet_Decision')).upper()
        if published_action in BET_ACTIONS and not bet:
            action = 'BLOCKED — VERIFY BET'
        else:
            action = published_action or 'PASS'
        sel = text(row.get('Selected_Team'))
        if not fresh and not bet and action not in {'BLOCKED — VERIFY BET'}:
            # Retain the publisher's PASS/CANDIDATE decision; price status is
            # shown in detail and the main spread selection is unavailable.
            diagnostics['unpriced_or_stale_games'] += 1
        if bet:
            diagnostics['displayed_authorized_bets'] += 1
        if action == 'CANDIDATE':
            diagnostics['published_candidate_games'] += 1
        if fresh:
            diagnostics['fresh_priced_games'] += 1
        raw_miner = text(row.get('V55_Raw_Miner')) or text(row.get('SYSTEM_MINER'))
        if not raw_miner:
            diagnostics['miner_unavailable_games'] += 1
        support = families(row.get('Confirm_Families'))
        opposition = families(row.get('Oppose_Families'))
        pathi = text(row.get('V55_Raw_Pathi'))
        pathi_state = ('MIXED' if 'PATHI' in support and 'PATHI' in opposition
                       else 'SUPPORT' if 'PATHI' in support
                       else 'CONFLICT' if 'PATHI' in opposition
                       else 'TRACK' if pathi else '—')
        origin = text(row.get('V55_Unified_Brains_Seen'))
        edge_sources = []
        if origin:
            edge_sources.append('ORIGIN: ' + origin)
        if support:
            edge_sources.append('SUPPORT: ' + ', '.join(support))
        if opposition:
            edge_sources.append('OPPOSE: ' + ', '.join(opposition))
        system = (text(row.get('V55_Active_System_Rules')) or
                  text(row.get('V55_Active_System_Names')) or '—')
        prob = (pct(row.get('V55_Policy_Win_Prob')) if fresh and sel
                else '—')
        # Policy prob may be missing or intentionally withheld; show a published
        # CORE probability only if it belongs to the selected published side.
        if prob == '—' and fresh and sel:
            prob = pct(row.get('CORE_Cover_Probability'))
        points = (num(row.get('CORE_Project_Cover_Points')) if fresh and sel
                  else None)
        point_label = f'{points:+.2f} pts' if points is not None else '—'
        pick = _side_label(row)
        game_time = (kick.tz_convert('America/New_York').strftime('%a %I:%M %p')
                     if pd.notna(kick) else '—')
        record = {
            'Game Time': game_time, 'Matchup': match,
            'Spr Action': action, 'Spr Pick': pick, 'Spr Prob': prob,
            'Spr Model Edge': point_label,
            'H2H Action': '—', 'H2H Pick': '—', 'H2H Prob': '—', 'H2H Model Edge': '—',
            'Tot Action': '—', 'Tot Pick': '—', 'Tot Prob': '—', 'Tot Model Edge': '—',
            'Production Plays': ('Spr: ' + pick) if bet else '—',
            'Pathi State': pathi_state, 'Pathi Overlay': pathi or '—',
            'Edge Sources': ' | '.join(edge_sources) or '—',
            'System Trigger': system,
            '_kick': kick, '_gid': gid,
        }
        board.append(record)
        cands, candidate_error = candidate_info(row.get('V55_Unified_Candidates_JSON'))
        if candidate_error:
            diagnostics[candidate_error] += 1
        for c in cands:
            origins = c.get('origins') or []
            if not isinstance(origins, list): origins = [origins]
            candidate_team = text(c.get('team'))
            candidate_origins = ', '.join(str(x) for x in origins if text(x)) or '—'
            candidate_spread = num(c.get('price_spread'))
            candidate_odds = num(c.get('price_odds'))
            candidates_detail.append({
                'Game Time': game_time, 'Matchup': match,
                'Candidate Team': candidate_team or '—',
                'Side': text(c.get('side')) or '—',
                'Originating Brains': candidate_origins,
                'Published Spread': candidate_spread if fresh else None,
                'Published Odds': candidate_odds if fresh else None,
                'Book': text(c.get('price_book')) if fresh else '—',
                'Qualified Support': ' | '.join(map(str, c.get('qualified_family_keys') or [])) or '—',
                'Conflicts': ' | '.join(map(str, c.get('opposing_family_keys') or [])) or '—',
                'Research EV': num(c.get('estimated_ev')),
                'Research Hold': text(c.get('hold')) or '—',
                'Published Bet Action': action if bet and candidate_team.casefold() == sel.casefold() else 'RESEARCH ONLY',
                'Game ID': gid,
            })
            if not origins:
                diagnostics['candidates_without_named_origin'] += 1
            if any(str(x).upper().find('MINER') >= 0 for x in origins):
                diagnostics['miner_originating_candidate_sides'] += 1
        detail.append({
            'Game Time': game_time, 'Matchup': match, 'Game ID': gid,
            'Backend Decision': published_action or '—',
            'Bet Authority': int(bet), 'Published Authority Flag': text(row.get('V55_Bet_Authority')) or '0',
            'Published Selected Team': sel or '—',
            'Selected Spread': num(row.get('Current_Spread')) if fresh else None,
            'Selected Odds': num(row.get('Odds_American')) if fresh else None,
            'Home Spread': num(row.get('Backend_Home_Spread')) if fresh else None,
            'Home Odds': num(row.get('Backend_Home_Odds_American')) if fresh else None,
            'Away Spread': num(row.get('Backend_Away_Spread')) if fresh else None,
            'Away Odds': num(row.get('Backend_Away_Odds_American')) if fresh else None,
            'Sportsbook': text(row.get('Backend_Quote_Book')) or '—',
            'Quote State': text(row.get('Quote_State')) or '—',
            'Quote Time': text(row.get('Backend_Quote_Timestamp')) or '—',
            'Policy EV': num(row.get('V55_Policy_EV')),
            'Probability State': text(row.get('V55_Probability_State')) or '—',
            'Originating Brain Families': origin or '—',
            'Miner State': raw_miner or '—',
            'Support Families': ' | '.join(support) or '—',
            'Opposition Families': ' | '.join(opposition) or '—',
            'Active Systems': system,
            'Authority Hold Reason': text(row.get('V55_Authority_Hold_Reason')) or '—',
            'Unified Research State': text(row.get('V55_Unified_Research_State')) or '—',
            'Unified Candidates': len(cands),
        })
    main = pd.DataFrame(board)
    if len(main):
        main = main.sort_values(['_kick', 'Matchup'], na_position='last', kind='stable')
        main = main[NFL_COLUMNS].reset_index(drop=True)
    else:
        main = pd.DataFrame(columns=NFL_COLUMNS)
    details = pd.DataFrame(detail)
    all_candidates = pd.DataFrame(candidates_detail)
    diagnostics['unique_upcoming_games'] = len(main)
    diagnostics['candidate_side_records'] = len(all_candidates)
    diagnostics['qualified_support_candidate_sides'] = int(sum(
        r.get('Qualified Support') not in (None, '', '—') for r in candidates_detail))
    diagnostics['published_miners_with_named_origin'] = diagnostics['miner_originating_candidate_sides']
    return main, details, all_candidates, dict(diagnostics)


def render_published_v55_board(st):
    """Replace previous double research table with exact NFL board format."""
    import pandas as pd
    try:
        from ncaaf_v55_board import read_published_latest
        weekly, audit = read_published_latest()
    except Exception as exc:
        st.error('Verified NCAAF V5.5 publication unavailable; no V1 fallback.')
        st.code(f'{type(exc).__name__}: {str(exc)[:450]}')
        return
    if (audit.get('status') != 'COMPLETE_V55_INDEPENDENT_WEEKLY_SHADOW' or
            audit.get('production_v1_probabilities_used') is not False):
        st.error('Backend publication provenance failed; board withheld.')
        st.json({'status':audit.get('status'),'v1_probabilities_used':audit.get('production_v1_probabilities_used')})
        return
    if not isinstance(weekly, pd.DataFrame):
        st.error('Invalid backend published table.')
        return
    try:
        main, details, candidate_ledger, diag = nfl_shaped_game_board(weekly)
    except Exception as exc:
        st.error('Cannot construct verified NCAAF board from backend publication.')
        st.code(f'{type(exc).__name__}: {str(exc)[:500]}')
        return
    # NFL UI: date filter, subtitle, five metrics, one 19-col betting board,
    # then a single on-demand additional-detail selector.
    # Use backend start times in the detail table to reconstruct the ET date.
    d = details.copy()
    date_options = ['All']
    if len(d):
        dates = sorted({x[:11] for x in d['Game Time'].astype(str) if x != '—'})
        # The source uses short weekday/time. Group instead using actual kickoff.
        if 'Game_Start_UTC' in weekly:
            wk = weekly.copy()
            wk['_et_date'] = pd.to_datetime(wk['Game_Start_UTC'], utc=True, errors='coerce').dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
            mapping = {text(r.get('Game_ID')) or text(r.get('Matchup')): r['_et_date'] for _, r in wk.iterrows()}
            d['_date'] = d.apply(lambda x: mapping.get(text(x['Game ID']) or text(x['Matchup']), ''), axis=1)
            date_options = ['All'] + sorted(x for x in d['_date'].dropna().unique().tolist() if x)
    date_choice = st.selectbox('Game date', date_options, index=0, key='ncaaf-v55-nfl-parity-date')
    if date_choice != 'All' and '_date' in d:
        game_ids = set(d.loc[d['_date'].eq(date_choice),'Game ID'])
        matchups = set(d.loc[d['_date'].eq(date_choice),'Matchup'])
        d = d.loc[d['_date'].eq(date_choice)].copy()
        main = main.loc[main['Matchup'].isin(matchups)].copy()
        if len(candidate_ledger):
            candidate_ledger = candidate_ledger.loc[candidate_ledger['Matchup'].isin(matchups)].copy()
    st.subheader('NCAAF V5.5 Production — Spread, H2H & Totals')
    st.caption('Same 19-column layout as NFL. V5.5 publishes SPREADS only at present; H2H/Totals are not substituted from V1. All actions, prices, and supporting evidence come from the verified NCAAF backend.')
    st.caption(f"Background publication: {audit.get('published_at_utc','—')} · snapshot {audit.get('snapshot_id','—')} · status {audit.get('publication_status','—')}")
    if audit.get('publication_status') == 'STALE_MARKET':
        st.error('Backend quotes are stale. No current bets can be shown.')
    spr_bets = int(main['Spr Action'].isin(BET_ACTIONS).sum())
    strong_bets = int(main['Spr Action'].eq('STRONG BET').sum())
    candidates = int(main['Spr Action'].eq('CANDIDATE').sum())
    fresh = int(d.get('Quote State',pd.Series(dtype=str)).fillna('').astype(str).str.startswith('FRESH_').sum())
    a,b,c,e,f = st.columns(5)
    a.metric('Upcoming games',len(main))
    b.metric('Spread candidates',candidates)
    c.metric('Confirmed spread bets',spr_bets)
    e.metric('Strong bets',strong_bets)
    f.metric('Fresh priced',fresh)
    if main.empty:
        st.info('No upcoming published games.')
    else:
        st.dataframe(main, use_container_width=True, hide_index=True)
    extra = st.selectbox('Additional NCAAF tables — load only when selected',[
        'Betting Board only', 'System Details', 'Model / Bet Authority Details',
        'Research Candidate Sides', 'Publication / Price Diagnostics',
    ], index=0, key='ncaaf-v55-nfl-parity-detail')
    if extra == 'System Details':
        cols = ['Game Time','Matchup','Originating Brain Families','Miner State','Support Families',
                'Opposition Families','Active Systems','Unified Research State']
        st.dataframe(d[[x for x in cols if x in d]], use_container_width=True, hide_index=True)
        # The live failure is visible here. Do not silently swap in history.
        miner_available = int(d['Miner State'].astype(str).isin(['—','NOT_EVALUABLE','NOT_PUBLISHED']).sum()) if len(d) else 0
        if miner_available:
            st.warning(f'{miner_available} game rows have no evaluable/published live Miner signal. Historical Miner performance does not automatically become a live vote; fix the backend evidence publication.')
    elif extra == 'Model / Bet Authority Details':
        st.dataframe(d.drop(columns=['_date'], errors='ignore'),use_container_width=True,hide_index=True)
    elif extra == 'Research Candidate Sides':
        if len(candidate_ledger):
            st.dataframe(candidate_ledger,hide_index=True,use_container_width=True)
            st.caption('Retrospective or live research candidates are NOT bets unless published authority and prices match the selected game side. These rows do not grant bet authority.')
        else:
            st.info('No independent side-level candidates published for the selected date.')
    elif extra == 'Publication / Price Diagnostics':
        st.json({**diag, 'snapshot_id':audit.get('snapshot_id'),
                 'backend_authorized_bets':audit.get('v55_authorized_bet_rows'),
                 'miner_live_bridge':audit.get('miner_live_bridge'),
                 'v55_live_system_evaluation':audit.get('v55_live_system_evaluation'),
                 'v55_unified_artifact_hold':audit.get('v55_unified_artifact_hold'),
                 'v55_authority_policy_status':audit.get('v55_authority_policy_status')})
        if diag.get('duplicate_game_rows_suppressed'):
            st.error('Backend duplicated game identities. Duplicated records were withheld; publication needs repair.')
    if spr_bets == 0:
        st.caption('No NCAAF wager has current backend authorization in this publication. Research candidates remain available in the detail selector; the UI does not authorize wagers.')
