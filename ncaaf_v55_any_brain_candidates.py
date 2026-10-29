"""Two-sided NCAAF V5.5 candidate discovery, NOT betting authority.

Reuses verified same-book home/away quotes and the existing frozen-OOS
qualified event contract. Allows all existing independently validated families
(and separately recorded, explicitly directional unqualified evidence) to
originate a shadow candidate regardless of the CORE-selected side.

The existing owner-authority model, lock, price selection, and V7 shadow
selector are not changed by this module.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

BRAINS = frozenset((
    'CORE', 'SYSTEM_MINER', 'PATHI', 'BIG_AL', 'STAT',
    'PREDICTION_TRACKER', 'MARKET_RICH',
))
MAX_CANDIDATE_QUOTE_AGE_MIN = 90


def _helpers():
    # Imported lazily to avoid a circular dependency in the existing board.
    from ncaaf_v55_board import _matchup_teams, _norm_team, _price, _timestamp, _valid_american
    return _matchup_teams, _norm_team, _price, _timestamp, _valid_american


def build_two_sided_scan(board, *, now=None):
    """Build side-specific research rows only from VERIFIED paired prices.

    These rows are for running the EXISTING Miner/Pathi/BigAl/PT/STAT event
    validators on both sides. No model forecast or bet authority is synthesized.
    """
    import pandas as pd
    match, norm, price, stamp, valid_odds = _helpers()
    at = stamp(now if now is not None else datetime.now(timezone.utc))
    out = []
    for _, row in board.iterrows():
        if not str(row.get('Quote_State') or '').startswith('FRESH_TWO_SIDED_'):
            continue
        (home, away), oriented = match(row)
        gid = str(row.get('Game_ID') or '').strip()
        kick = stamp(row.get('Game_Start_UTC'))
        if not (gid and oriented and home and away and pd.notna(kick) and at < kick):
            continue
        records = []
        for side, team, label in [('HOME', home, 'Home'), ('AWAY', away, 'Away')]:
            book = str(row.get('Backend_'+label+'_Book') or '').strip().lower()
            line = price(row.get('Backend_'+label+'_Spread'))
            odds = price(row.get('Backend_'+label+'_Odds_American'))
            observed = stamp(row.get('Backend_'+label+'_Quote_UTC'))
            if (not book or line is None or odds is None or not valid_odds(odds)
                or pd.isna(observed) or not observed <= at or not observed < kick
                or (at-observed).total_seconds()/60 > MAX_CANDIDATE_QUOTE_AGE_MIN):
                records = []
                break
            records.append((side, team, book, line, odds, observed))
        if len(records) != 2:
            continue
        if (records[0][2] != records[1][2]
            or abs(records[0][3]+records[1][3]) > .01
            or abs((records[0][5]-records[1][5]).total_seconds()) > 900):
            continue
        for side, team, book, line, odds, observed in records:
            rec=row.to_dict()
            rec['Selected_Team']=team
            rec['V55_Research_Selected_Side']=side
            rec['Current_Spread']=line
            rec['Odds_American']=odds
            rec['Backend_Quote_Book']=book
            rec['Backend_Quote_Timestamp']=observed.isoformat()
            rec['Quote_State']='FRESH_TWO_SIDED_BACKEND_PRICE_BRAIN_SCAN'
            rec['V55_Brain_Scan_Side']=side
            rec['V55_Brain_Scan_Research_Only']=True
            rec['V55_Bet_Authority']=0
            rec['V55_Bet_Decision']='RESEARCH_ONLY_NO_AUTHORITY'
            out.append(rec)
    return pd.DataFrame(out, columns=list(board.columns)+[
        'V55_Brain_Scan_Side','V55_Brain_Scan_Research_Only'])


def attach_any_brain_candidates(board, vetted_sides, *, raw_events=None, live_research_events=None, now=None):
    """Summarize two-sided discovery into the existing game-level board.

    Qualified events come ONLY from the current, already-vetted V55
    attach_v55_qualified_evidence() contract. Optional unqualified events
    (e.g. named directional MARKET_RICH signals) may originate a research
    candidate, but NEVER receive qualified status or betting authority.

    A candidate on the OTHER side is preserved even when CORE selected HOME.
    """
    import pandas as pd
    match, norm, price, stamp, _ = _helpers()
    at=stamp(now if now is not None else datetime.now(timezone.utc))
    out=board.copy()
    for col,value in (
        ('V55_Any_Brain_Candidate_Count', 0),
        ('V55_Any_Brain_Qualified_Candidate_Count', 0),
        ('V55_Any_Brain_Origin_Families', None),
        ('V55_Any_Brain_Candidates_JSON','[]'),
        ('V55_Any_Brain_Research_Status','NO_CANDIDATES'),
    ):
        out[col]=value
    audit={'status':'SHADOW_ONLY','games':len(out),'two_sided_rows':len(vetted_sides),
           'candidate_games':0,'candidate_game_sides':0,
           'qualified_candidate_game_sides':0,'opposing_core_side_candidates':0,
           'by_origin':{},'bet_authority_changed':False,
           'any_brain_can_originate':sorted(BRAINS)}
    if not isinstance(vetted_sides,pd.DataFrame):
        vetted_sides=pd.DataFrame()
    if not isinstance(raw_events,pd.DataFrame):
        raw_events=pd.DataFrame()
    # Live research signals are independent discoveries, never qualified votes.
    # Use the exact same original-quote and pregame matching gate as raw events.
    if not isinstance(live_research_events,pd.DataFrame):
        live_research_events=pd.DataFrame()
    audit['live_research_input_rows']=int(len(live_research_events))
    audit['raw_event_input_rows']=int(len(raw_events))
    candidate_event_frames = []
    for frame, is_live in ((raw_events, False), (live_research_events, True)):
        if not frame.empty:
            tagged = frame.copy()
            tagged['_V55_Live_Research_Event'] = is_live
            candidate_event_frames.append(tagged)
    discovery_events = (pd.concat(candidate_event_frames, ignore_index=True, sort=False)
                        if candidate_event_frames else pd.DataFrame())
    audit['discovery_event_rows']=int(len(discovery_events))
    audit['research_only_events'] = True
    # A live signal is a research observation, not a qualified production vote.
    # Count INPUT and REJECTED events separately; otherwise valid live signals
    # can silently disappear at this final origination gate.
    rejected_events = Counter()
    accepted_events = Counter()
    if not discovery_events.empty:
        known_game_ids = set(out['Game_ID'].astype(str)) if 'Game_ID' in out else set()
        for _, event in discovery_events.iterrows():
            if str(event.get('Game_ID') or '').strip() not in known_game_ids:
                rejected_events['GAME_ID_NOT_ON_BOARD'] += 1
    for idx,row in out.iterrows():
        gid=str(row.get('Game_ID') or '').strip()
        kick=stamp(row.get('Game_Start_UTC'))
        (home,away),oriented=match(row)
        if not gid or not oriented or pd.isna(kick):
            continue
        sides=vetted_sides.loc[vetted_sides['Game_ID'].astype(str).eq(gid)] if ('Game_ID' in vetted_sides) else pd.DataFrame()
        opportunities={}
        for _,v in sides.iterrows():
            team=norm(v.get('Selected_Team'))
            side=str(v.get('V55_Brain_Scan_Side') or '')
            if team not in (home,away) or side not in ('HOME','AWAY'):
                continue
            n=int(price(v.get('Confirm_Count')) or 0)
            opposing=int(price(v.get('Oppose_Count')) or 0)
            if n<1:
                continue
            try: details=json.loads(v.get('V55_Qualified_Evidence_Detail_JSON') or '[]')
            except (TypeError,ValueError):details=[]
            if not isinstance(details,list):details=[]
            named=[x for x in details if isinstance(x,dict)
                   and norm(x.get('direction'))==team
                   and str(x.get('source') or '').upper().strip() in BRAINS]
            sources=sorted({str(x['source']).upper().strip() for x in named})
            # An explicit, direction-verified independently qualified vote is
            # mandatory. Do not infer a candidate from raw/unnamed counts.
            if not sources:
                continue
            opportunities[team]={
                'game_id':gid,'team':team,'side':side,
                'origins':sources,'qualified_independent_families':n,
                'opposing_independent_families':opposing,
                'status':('QUALIFIED_CONFLICTED_SHADOW' if opposing else 'QUALIFIED_SHADOW'),
                'price_book':str(v.get('Backend_Quote_Book') or ''),
                'price_spread':price(v.get('Current_Spread')),
                'price_odds':price(v.get('Odds_American')),
                'price_observed_utc':str(v.get('Backend_Quote_Timestamp') or ''),
                'system_ids':sorted({str(x.get('system_id') or '') for x in named
                                     if str(x.get('system_id') or '').strip()})[:15],
                'bet_authority':0,
            }
        # The explicit external event feed may include unqualified named,
        # directional pregame signals. Allow discovery only when its ORIGINAL
        # book, side, line, and clock match the frozen two-sided quote.
        if not discovery_events.empty:
            # We only inspect source events tied to THIS game. Every rejected
            # observation receives a specific reason in the discovery audit.
            for _, e in discovery_events.iterrows():
                if str(e.get('Game_ID') or '').strip() != gid:
                    continue
                family = str(e.get('Source_Family') or '').upper().strip()
                if family not in BRAINS or family == 'CORE':
                    rejected_events['INVALID_INDEPENDENT_FAMILY'] += 1
                    continue
                team = norm(e.get('Signal_Side'))
                if team not in (home, away):
                    rejected_events['DIRECTION_NOT_IN_GAME'] += 1
                    continue
                if sides.empty or 'Selected_Team' not in sides:
                    rejected_events['NO_VETTED_PAIRED_SIDES'] += 1
                    continue
                side_row = sides.loc[sides['Selected_Team'].map(norm).eq(team)]
                if side_row.empty:
                    rejected_events['SIDE_MISSING_FROM_VETTED_PAIR'] += 1
                    continue
                v = side_row.iloc[0]
                observed_at = stamp(e.get('Observed_At_UTC'))
                original_quote = stamp(e.get('Observed_Quote_UTC'))
                event_kick = stamp(e.get('Game_Start_UTC'))
                current_quote = stamp(v.get('Backend_Quote_Timestamp'))
                if any(pd.isna(t) for t in (observed_at, event_kick, current_quote, at)):
                    rejected_events['MISSING_EVENT_OR_QUOTE_TIMESTAMP'] += 1
                    continue
                if event_kick != kick:
                    rejected_events['KICKOFF_MISMATCH'] += 1
                    continue
                if bool(e.get('_V55_Live_Research_Event')):
                    # A LIVE signal is computed after its original quote.
                    # The old gate required observed_at <= current_quote and
                    # inadvertently blocked genuine contemporaneous evidence.
                    # Do not infer a missing original-quote timestamp.
                    if pd.isna(original_quote):
                        rejected_events['MISSING_EVENT_OR_QUOTE_TIMESTAMP'] += 1
                        continue
                    if not original_quote <= observed_at <= at < kick:
                        rejected_events['NON_PREGAME_OR_BAD_EVENT_CLOCK'] += 1
                        continue
                    if not original_quote <= current_quote <= at:
                        rejected_events['INVALID_ORIGINAL_QUOTE_VINTAGE'] += 1
                        continue
                    if (observed_at - original_quote).total_seconds() > 15 * 60:
                        rejected_events['EVALUATION_TOO_FAR_FROM_ORIGINAL_QUOTE'] += 1
                        continue
                    if (current_quote - original_quote).total_seconds() > 15 * 60:
                        rejected_events['BOARD_QUOTE_NOT_SAME_VINTAGE'] += 1
                        continue
                    if (at - observed_at).total_seconds() > 15 * 60:
                        rejected_events['STALE_RESEARCH_EVENT'] += 1
                        continue
                else:
                    # Backward-compatible LEGACY raw-event contract. It is a
                    # quote-timestamped observation that must PRECEDE the
                    # vetted quote, not a newly computed live signal.
                    if not observed_at <= current_quote <= at < kick:
                        rejected_events['LEGACY_QUOTE_CLOCK_MISMATCH'] += 1
                        continue
                    if (current_quote - observed_at).total_seconds() > 15 * 60:
                        rejected_events['LEGACY_QUOTE_TOO_OLD'] += 1
                        continue
                book = str(e.get('Observed_Book') or '').lower().strip()
                vetted_book = str(v.get('Backend_Quote_Book') or '').lower().strip()
                if not book or book != vetted_book:
                    rejected_events['BOOK_MISMATCH'] += 1
                    continue
                spread = price(e.get('Observed_Spread'))
                chosen_spread = price(v.get('Current_Spread'))
                if spread is None or chosen_spread is None or abs(spread - chosen_spread) > .001:
                    rejected_events['SPREAD_MISMATCH'] += 1
                    continue
                # The two-sided scan already certified same-book opposing
                # spreads. These events remain shadow-only, even if accepted.
                accepted_events[family] += 1
                if team not in opportunities:
                    opportunities[team] = {
                        'game_id': gid, 'team': team,
                        'side': ('HOME' if team == home else 'AWAY'),
                        'origins': [], 'qualified_independent_families': 0,
                        'opposing_independent_families': 0,
                        'status': 'UNQUALIFIED_SHADOW',
                        'price_book': vetted_book,
                        'price_spread': chosen_spread,
                        'price_odds': price(v.get('Odds_American')),
                        'price_observed_utc': str(v.get('Backend_Quote_Timestamp') or ''),
                        'system_ids': [], 'bet_authority': 0,
                    }
                target = opportunities[team]
                target['origins'] = sorted(set(target['origins']) | {family})
                if str(e.get('Signal_Family') or '').strip():
                    target['system_ids'] = sorted(set(target['system_ids']) | {str(e['Signal_Family'])})[:15]
        # Preserve an already-established CORE candidate as a discovery origin;
        # do not independently certify a new probability from raw CORE margin.
        core_dec=str(row.get('V55_Bet_Decision') or '')
        core_team=norm(row.get('Selected_Team'))
        if core_team in (home,away) and core_dec in ('CANDIDATE','BET','STRONG BET'):
            if core_team not in opportunities:
                opportunities[core_team]={
                    'game_id':gid,'team':core_team,
                    'side':('HOME' if core_team==home else 'AWAY'),
                    'origins':['CORE'],'qualified_independent_families':0,
                    'opposing_independent_families':0,'status':'CORE_CANDIDATE',
                    'price_book':str(row.get('Backend_Quote_Book') or ''),
                    'price_spread':price(row.get('Current_Spread')),
                    'price_odds':price(row.get('Odds_American')),
                    'price_observed_utc':str(row.get('Backend_Quote_Timestamp') or ''),
                    'system_ids':[],'bet_authority':0}
            else:
                opportunities[core_team]['origins']=sorted(set(opportunities[core_team]['origins'])|{'CORE'})
        items=sorted(opportunities.values(),key=lambda r:r['side'])
        if not items:continue
        out.at[idx,'V55_Any_Brain_Candidate_Count']=len(items)
        out.at[idx,'V55_Any_Brain_Qualified_Candidate_Count']=sum(x['qualified_independent_families']>0 for x in items)
        origin_set=sorted(set(f for x in items for f in x['origins']))
        out.at[idx,'V55_Any_Brain_Origin_Families']='|'.join(origin_set)
        out.at[idx,'V55_Any_Brain_Candidates_JSON']=json.dumps(items,sort_keys=True)
        out.at[idx,'V55_Any_Brain_Research_Status']='CANDIDATES_FOUND_SHADOW_ONLY'
        audit['candidate_games']+=1
        audit['candidate_game_sides']+=len(items)
        audit['qualified_candidate_game_sides']+=sum(x['qualified_independent_families']>0 for x in items)
        if core_team in (home,away):
            audit['opposing_core_side_candidates']+=sum(x['team']!=core_team for x in items)
        for f in origin_set:audit['by_origin'][f]=audit['by_origin'].get(f,0)+1
    audit['live_event_accepted_rows'] = int(sum(accepted_events.values()))
    audit['live_event_accepted_by_family'] = dict(accepted_events)
    audit['live_event_rejections'] = dict(rejected_events)
    audit['live_event_rejected_rows'] = int(sum(rejected_events.values()))
    audit['live_event_accounted_for'] = (
        audit['live_event_accepted_rows'] + audit['live_event_rejected_rows']
        == audit['discovery_event_rows']
    )
    # Strict boundary: this module cannot manufacture or update bet authority.
    for col in ('V55_Bet_Authority','V55_Bet_Decision','V55_Policy_EV',
                'V55_Policy_Stake_Units','V55_Authority_Policy_Id'):
        if col in board.columns and not out[col].equals(board[col]):
            raise RuntimeError('BRAIN_CANDIDATE_ILLEGAL_AUTHORITY_MUTATION:'+col)
    return out,audit
