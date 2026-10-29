"""Prospective, shadow-only independent signal discovery on original verified quote rows.

No historical qualification, no bet authority. A structured directional Miner vote
is a research observation only. Other families require equally structured sources;
context flags and named text are deliberately not promoted to picks.
"""
from __future__ import annotations
import json
from collections import Counter


def build_shadow_events(live_rows, two_sided, *, now=None):
    import pandas as pd
    from datetime import datetime, timezone
    from ncaaf_v55_board import (_matchup_teams, _norm_team, _price, _timestamp,
                                _v55_parse_structured_votes)

    clock = _timestamp(now if now is not None else datetime.now(timezone.utc))
    audit = {'status': 'SHADOW_ONLY', 'live_rows': 0, 'candidate_side_rows': 0,
             'miner_vote_inputs': 0, 'accepted': 0, 'rejected': {},
             'by_family': {}, 'authority': 0}
    if not isinstance(live_rows, pd.DataFrame) or not isinstance(two_sided, pd.DataFrame):
        return pd.DataFrame(), audit
    audit['live_rows'] = len(live_rows)
    audit['candidate_side_rows'] = len(two_sided)
    if live_rows.empty or two_sided.empty:
        return pd.DataFrame(), audit
    required = {'Game_Start', 'Home_Team_Norm', 'Away_Team_Norm', 'Outcome',
                'Snapshot_Timestamp', 'Value', 'NCAAF_RV22_Miner_Votes'}
    if not required.issubset(live_rows.columns):
        audit['status'] = 'MISSING_STRUCTURED_SOURCE_FIELDS'
        audit['missing_fields'] = sorted(required - set(live_rows.columns))
        return pd.DataFrame(), audit
    rejected = Counter()
    accepted = {}
    for _, s in two_sided.iterrows():
        gid = str(s.get('Game_ID') or '').strip()
        (home, away), oriented = _matchup_teams(s)
        team = _norm_team(s.get('Selected_Team'))
        kick = _timestamp(s.get('Game_Start_UTC'))
        qt = _timestamp(s.get('Backend_Quote_Timestamp'))
        book = str(s.get('Backend_Quote_Book') or '').strip().lower()
        spread = _price(s.get('Current_Spread'))
        if (not oriented or not gid or team not in (home, away) or not book or
            spread is None or pd.isna(kick) or pd.isna(qt) or not qt <= clock < kick or
            (clock - qt).total_seconds() > 15*60):
            rejected['SIDE_PRICE_NOT_FRESH'] += 1
            continue
        for _, r in live_rows.iterrows():
            if _norm_team(r.get('Home_Team_Norm')) != home or _norm_team(r.get('Away_Team_Norm')) != away:
                continue
            start = _timestamp(r.get('Game_Start'))
            seen = _timestamp(r.get('Snapshot_Timestamp'))
            source_team = _norm_team(r.get('Outcome'))
            source_book = str(r.get('Bookmaker') or r.get('Book') or '').strip().lower()
            source_spread = _price(r.get('Value'))
            if (pd.isna(start) or pd.isna(seen) or abs((start-kick).total_seconds())>15*60 or
                not seen <= clock < kick or (clock-seen).total_seconds() > 5*60 or
                source_book != book or source_team not in (home, away) or source_spread is None or
                abs(source_spread - (spread if source_team == team else -spread)) > .001):
                rejected['SOURCE_IDENTITY_PRICE_OR_TIME'] += 1
                continue
            # Live calculation is permitted prospectively ONLY on a verified current quote.
            # Never substitute calculation timestamp for original quote observation time.
            if r.get('V55_Live_Computed_On_Current_Quote') is not True:
                rejected['NOT_LIVE_COMPUTED_ON_ORIGINAL_QUOTE'] += 1
                continue
            original_qt = _timestamp(r.get('V55_Original_Quote_Timestamp'))
            if (pd.isna(original_qt) or original_qt > seen or original_qt > qt or
                (qt-original_qt).total_seconds() > 15*60):
                rejected['ORIGINAL_QUOTE_PROVENANCE_MISSING'] += 1
                continue
            votes = _v55_parse_structured_votes(r.get('NCAAF_RV22_Miner_Votes'))
            audit['miner_vote_inputs'] += len(votes)
            for vote in votes:
                if str(vote.get('market') or '').lower() != 'spreads':
                    rejected['NOT_SPREAD_VOTE'] += 1
                    continue
                if str(vote.get('source_type') or '').upper() != 'MINER':
                    rejected['NOT_MINER_VOTE'] += 1
                    continue
                target = _norm_team(vote.get('target'))
                if target != team:
                    rejected['NONMATCHING_DIRECTION'] += 1
                    continue
                mid = str(vote.get('family_id') or '').strip()
                if not mid:
                    rejected['MISSING_MECHANISM_ID'] += 1
                    continue
                # Neither live_authority_eligible nor a claimed sample rate certifies OOS.
                key = (gid, team, 'SYSTEM_MINER', mid)
                accepted[key] = {
                    'Game_ID': gid, 'Signal_Side': team, 'Source_Family': 'SYSTEM_MINER',
                    'Signal_Family': mid, 'Observed_At_UTC': seen.isoformat(),
                    'Observed_Quote_UTC': original_qt.isoformat(),
                    'Observed_Book': book, 'Observed_Spread': spread,
                    'Game_Start_UTC': kick.isoformat(),
                    'Research_Evidence_State': 'UNQUALIFIED_PROSPECTIVE_SHADOW',
                    'V55_Evidence_Qualified': False,
                }
    audit['accepted'] = len(accepted)
    audit['rejected'] = dict(rejected)
    audit['by_family'] = {'SYSTEM_MINER': len(accepted)} if accepted else {}
    return pd.DataFrame(accepted.values()), audit
