"""Full-history research Miner library transition (2013-2025).

Runs AFTER fresh run_system_miner_v3 discovery and BEFORE report publication.
Never imports incumbents into the new qualification catalog. Historical overlap
is diagnostic only. The existing miner search and OOS gates are not loosened.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

POLICY = 'NCAAF_V55_FRESH_FULL_HISTORY_2013_2025_V1'
ORIGIN = 'FRESH_2013_2025_DISCOVERY'
SEASONS = tuple(range(2013, 2026))
MIN_ROWS_PER_SEASON = 100  # preflight only; NOT a new mechanism sample gate
MIN_WAREHOUSE_COVERAGE = 0.95  # require near-complete eligible home-game source


def _warehouse_reference_counts(dashboard_module):
    """Ground the research frame in the expanded NCAAF history warehouse.

    Counts are a conservative coverage gate, not proof of exact game/feature
    lineage. That still requires source-ID and feature-as-of audits.
    """
    client = getattr(dashboard_module, 'bq_client', None) if dashboard_module is not None else None
    if client is None:
        return {}, 'BQ_CLIENT_UNAVAILABLE'
    query = """
      SELECT Season AS season, COUNT(1) AS home_games
      FROM `sharplogger.sharp_data.ncaaf_historical_core_training_vw`
      WHERE Historical_Core_Eligible = 1
        AND Is_Home = 1 AND Season BETWEEN 2013 AND 2025
      GROUP BY Season
    """
    try:
        rows = list(client.query(query).result())
        ref = {str(int(row['season'])): int(row['home_games']) for row in rows}
        if set(ref) != set(map(str, SEASONS)) or any(v < 1 for v in ref.values()):
            return ref, 'WAREHOUSE_SEASON_COVERAGE_INCOMPLETE'
        return ref, 'PASS'
    except Exception as exc:
        return {}, 'WAREHOUSE_AUDIT_QUERY_FAILED:'+type(exc).__name__+':'+str(exc)[:200]


def _signature(mech, market):
    return (str(mech.get('market') or market).lower(),
            str(mech.get('direction') or '').upper(),
            tuple(sorted(map(str, mech.get('representative_conditions') or []))))


def _inputs_audit(games, seasons, warehouse_season_counts, warehouse_status):
    import pandas as pd
    sy = pd.to_numeric(pd.Series(seasons), errors='coerce').reset_index(drop=True)
    if games is None or len(games) != len(sy):
        return {'status': 'HOLD_FRAME_ALIGNMENT', 'rows': 0}
    frame = games.reset_index(drop=True)
    by = Counter(int(v) for v in sy.dropna() if float(v).is_integer())
    observed = sorted(by)
    issues = []
    if observed != list(SEASONS):
        issues.append('MISSING_OR_UNEXPECTED_SEASONS')
    if any(by.get(year, 0) < MIN_ROWS_PER_SEASON for year in SEASONS):
        issues.append('SPARSE_YEAR_UNDER_100_ROWS')
    if len(frame) != len(sy) or any(y > 2025 or y < 2013 for y in by):
        issues.append('SEASON_FRAME_RANGE_INVALID')
    warehouse_coverage = {}
    if warehouse_status != 'PASS':
        issues.append('EXPANDED_WAREHOUSE_REFERENCE_UNAVAILABLE')
    else:
        for year in SEASONS:
            reference = int(warehouse_season_counts.get(str(year), 0))
            rate = by.get(year, 0) / reference if reference else 0.0
            warehouse_coverage[str(year)] = round(rate, 6)
            if rate < MIN_WAREHOUSE_COVERAGE:
                issues.append('EXPANDED_WAREHOUSE_COVERAGE_BELOW_95_PERCENT')
        issues = list(dict.fromkeys(issues))

    # A real game-level ID or unambiguous (home,away,kickoff) identity is
    # required. Do not silently count sportsbook snapshots as distinct games.
    id_col = next((c for c in ('Game_ID','Source_Game_ID','Game_Key') if c in frame.columns), None)
    if id_col is not None:
        game_key = frame[id_col].astype('string').str.strip()
        missing = int(game_key.isna().sum() + game_key.fillna('').eq('').sum())
        unique_key = sy.astype('string').str.cat(game_key.fillna(''), sep='|')
        identity_source = id_col
    else:
        a = next((c for c in ('Home_Team_Norm','Home_Team','Team_Norm','Team') if c in frame.columns), None)
        b = next((c for c in ('Away_Team_Norm','Away_Team','Opponent_Norm','Opponent') if c in frame.columns), None)
        d = next((c for c in ('Game_Start','Game_Date') if c in frame.columns), None)
        if a is None or b is None or d is None:
            unique_key = pd.Series([''] * len(frame))
            missing = len(frame)
            identity_source = 'UNAVAILABLE'
        else:
            date = pd.to_datetime(frame[d], errors='coerce', utc=True).dt.strftime('%Y-%m-%dT%H:%M')
            unique_key = (date.fillna('') + '|' + frame[a].astype('string').fillna('').str.lower()
                          + '|' + frame[b].astype('string').fillna('').str.lower())
            missing = int((date.isna() | frame[a].isna() | frame[b].isna()).sum())
            identity_source = f'{d}+{a}+{b}'
    duplicate_rows = int(unique_key[unique_key.ne('')].duplicated().sum())
    if missing: issues.append('MISSING_GAME_IDENTITY')
    if duplicate_rows: issues.append('DUPLICATED_GAME_IDENTITY')

    # Do not label an unverified historical feature frame as certified. The
    # quote timestamps below only check the basic pregame observation envelope;
    # historical feature-level vintage still needs upstream lineage evidence.
    start_col = next((c for c in ('Game_Start','Game_Date') if c in frame.columns), None)
    obs_col = next((c for c in ('Snapshot_Timestamp','Observed_At_UTC','Snapshot_UTC') if c in frame.columns), None)
    time_bad = None
    if start_col is None or obs_col is None:
        issues.append('ORIGINAL_PREGAME_TIMESTAMPS_UNAVAILABLE')
    else:
        start = pd.to_datetime(frame[start_col], errors='coerce', utc=True)
        obs = pd.to_datetime(frame[obs_col], errors='coerce', utc=True)
        time_bad = int((start.isna() | obs.isna() | ~(obs < start)).sum())
        if time_bad: issues.append('PREGAME_SNAPSHOT_VINTAGE_INVALID')

    # Sufficient season counts are necessary, not proof of the completeness of
    # every CFBD atom. This remains a separate source/feature lineage audit.
    digest = hashlib.sha256(json.dumps({
        'by_season': dict(sorted(by.items())),
        'identity': identity_source,
        'total': len(frame),
        'duplicates': duplicate_rows,
        'vintage_bad': time_bad,
    }, sort_keys=True).encode()).hexdigest()
    return {
        'status': 'PASS' if not issues else 'HOLD',
        'issues': issues,
        'rows': int(len(frame)),
        'season_rows': {str(y): int(by.get(y, 0)) for y in SEASONS},
        'observed_seasons': observed,
        'identity_source': identity_source,
        'missing_identity_rows': missing,
        'duplicate_game_identity_rows': duplicate_rows,
        'warehouse_reference_status': warehouse_status,
        'warehouse_reference_by_season': warehouse_season_counts,
        'warehouse_coverage_by_season': warehouse_coverage,
        'min_warehouse_coverage_required': MIN_WAREHOUSE_COVERAGE,
        'pregame_timestamp_bad_rows': time_bad,
        'source_frame_sha256': digest,
        'note': 'Row and timestamp checks do not certify individual feature lineage',
    }


def finalize_clean_miner_library(games, seasons, miners, *, previous_report=None, dashboard_module=None, reference_season_counts=None, log_func=print):
    """Return fresh library + explicit migration audit, without restoring incumbents.

    This function NEVER grants more authority than run_system_miner_v3 itself
    granted. It can only preserve or revoke existing live eligibility.
    """
    from copy import deepcopy
    fresh = deepcopy(miners)
    if reference_season_counts is None:
        ref, ref_status = _warehouse_reference_counts(dashboard_module)
    else:
        ref, ref_status = reference_season_counts, 'PASS'
    src = _inputs_audit(games, seasons, ref, ref_status)
    old_sigs = set()
    if isinstance(previous_report, dict):
        for market, mr in (previous_report.get('system_miner_v3') or {}).items():
            for m in (mr or {}).get('mechanism_families') or []:
                old_sigs.add(_signature(m, market))
    counts = Counter()
    signatures = set()
    for market, mr in (fresh or {}).items():
        if not isinstance(mr, dict):
            continue
        kept = []
        for m in mr.get('mechanism_families') or []:
            counts['fresh_candidates'] += 1
            signature = _signature(m, market)
            if signature in signatures:
                counts['duplicate_mechanism_signatures_removed'] += 1
                continue
            signatures.add(signature)
            if signature in old_sigs:
                counts['also_seen_in_legacy_archive'] += 1
            old_origin = str(m.get('library_origin') or '')
            if old_origin.startswith('INCUMBENT_') or str(m.get('incumbent_revalidation_status') or '') in (
                    'LEGACY_CONTRACT_HOLD','HOLD_NOT_EVALUABLE'):
                # A migrated/inherited mechanism is not a fresh discovery.
                counts['inherited_candidates_excluded'] += 1
                continue
            m['library_origin'] = 'FULL_HISTORY_FRESH_DISCOVERY'
            m['full_history_miner_origin'] = ORIGIN
            m['full_history_miner_policy'] = POLICY
            # Keep all existing holdout/overlap/market rules; no threshold relax.
            split = (m.get('split_audit') or {}).get('disjoint') is True
            originally_live = m.get('live_authority_eligible') is True
            eligible = (src['status'] == 'PASS' and
                        m.get('confirmation_pass') is True and split and originally_live)
            m['full_history_miner_certified'] = eligible
            m['full_history_miner_frame_sha256'] = src.get('source_frame_sha256')
            if not eligible:
                m['live_authority_eligible'] = False
                m['current_qualified'] = False
                counts['fresh_shadows_only'] += 1
            else:
                counts['fresh_certified_live'] += 1
            kept.append(m)
        mr['mechanism_families'] = kept
        mr['mechanism_family_count'] = len(kept)
        mr['confirmed_mechanism_count'] = sum(v.get('confirmation_pass') is True for v in kept)
        mr['incumbent_revalidation_applied'] = False
        mr['full_history_discovery_origin'] = ORIGIN
    audit = {
        'status': 'PASS' if src['status'] == 'PASS' else 'HOLD_DATA_LINEAGE',
        'policy': POLICY,
        'origin': ORIGIN,
        'original_legacy_reconciliation_disabled': True,
        'retired_old_systems_can_vote': False,
        'historical_input': src,
        'fresh_candidates': counts['fresh_candidates'],
        'fresh_certified_live': counts['fresh_certified_live'],
        'fresh_shadows_only': counts['fresh_shadows_only'],
        'inherited_candidates_excluded': counts['inherited_candidates_excluded'],
        'duplicate_mechanism_signatures_removed': counts['duplicate_mechanism_signatures_removed'],
        'legacy_overlap_researched_only': counts['also_seen_in_legacy_archive'],
        'prior_confirmed': len(old_sigs),
        'prior_current_confirmed': len(old_sigs),
        'new_challenger_confirmed': counts['fresh_certified_live'],
        'revalidated_retained': 0,
        'carried_hold': 0,
        'carried_contract_hold': 0,
        'demoted': 0,
        'rediscovered': 0,
        'restored_missing': 0,
        'selection_uses_2026': False,
        'production_authority': 0,
        'details': [],
        'deployment_note': 'New live authority requires deployed board AND live vote producer plus newly published research report',
    }
    log_func('[NCAAF-MINER-CLEAN-HISTORY] '+json.dumps(audit,sort_keys=True,default=str))
    return fresh,audit
