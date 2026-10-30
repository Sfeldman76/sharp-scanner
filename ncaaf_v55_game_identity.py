"""Shared research and settlement identity for NCAAF V5.5.

A stable identity uses NCAAF + game calendar date + oriented normalized
home and away teams. UTC kickoff remains a separate pregame/vintage control;
original Game_ID and official source identifiers must NOT be rewritten.

Never use this key on its own to grant a verified settlement, qualify an
independent brain, change V5.5 owner betting authority or fabricate a quote.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

SPORT = 'NCAAF'
# Use a consistent NCAA market calendar basis unless an original, explicit
# source game-date is supplied. Bare UTC DATE() would shift evening games.
DEFAULT_DATE_ZONE = 'America/New_York'


def canonical_team(value, normalizer=None):
    """Conservative identity; pipeline callers pass the frozen board normalizer."""
    if normalizer is not None:
        value = normalizer(value)
    if value is None:
        return ''
    result = re.sub(r'[^a-z0-9]+', ' ', str(value).casefold()).strip()
    result = ' '.join(result.split())
    return '' if result in {'', 'none', 'null', 'nan', 'nat'} else result


def aware_utc(value):
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            s = str(value or '').strip()
            if not s or s in ('NaT', 'nan', 'None'):
                return None
            dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
        except (ValueError, TypeError, OverflowError):
            return None
    if dt.tzinfo is None or dt.utcoffset() is None:
        return None
    return dt.astimezone(timezone.utc)


def event_calendar_date(*, kickoff_utc=None, game_date=None, date_zone=DEFAULT_DATE_ZONE):
    """Date is an original stated date, otherwise a zone-aware kickoff date.

    Do not silently treat a naive local datetime as UTC. An explicit game date
    must be exactly ISO YYYY-MM-DD; all sources must use the same date basis.
    """
    if game_date is not None and str(game_date).strip():
        value = str(game_date).strip()
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            return None
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    stamp = aware_utc(kickoff_utc)
    if stamp is None:
        return None
    try:
        return stamp.astimezone(ZoneInfo(date_zone)).date()
    except (KeyError, ValueError):
        return None


def stable_game_key(home, away, kickoff_utc=None, *, game_date=None,
                    date_zone=DEFAULT_DATE_ZONE, normalizer=None):
    h = canonical_team(home, normalizer)
    a = canonical_team(away, normalizer)
    d = event_calendar_date(kickoff_utc=kickoff_utc, game_date=game_date, date_zone=date_zone)
    if not h or not a or h == a or d is None:
        return None
    return f'{SPORT}|{d.isoformat()}|{h}|{a}'


def crosswalk_matches(source, target, *, source_original_id=None, target_original_id=None,
                      require_unique=True, count_of_matching_distinct_events=1):
    """Identity comparison only; caller still validates side, quote and clocks.

    Two exact original IDs may match without a fallback. If IDs differ, a
    stable-key match is permitted only after separate uniqueness testing.
    """
    if not source or not target or source != target:
        return 'NOT_SAME_GAME'
    if require_unique and count_of_matching_distinct_events != 1:
        return 'AMBIGUOUS_DATE_TEAMS_HOLD'
    if source_original_id and target_original_id and source_original_id == target_original_id:
        return 'EXACT_ORIGINAL_ID'
    return 'DATE_TEAMS_RESEARCH_CROSSWALK'
