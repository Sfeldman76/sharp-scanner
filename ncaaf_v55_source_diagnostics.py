"""Research-only, read-only producer input diagnostics. Does not emit events.

Identify why a current-quote evaluated source row was not an explicit event.
Diagnostics are never used to bypass original quote or pregame gates.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone


def audit_live_source_inputs(live_rows, *, now=None):
    import pandas as pd
    from ncaaf_v55_live_source_connectors import (
        _clock, _num, _team, _text, _true,
        MAX_OBSERVATION_AGE_S, MAX_QUOTE_TO_EVALUATION_S,
        PT_SELECTION, PT_COUNT, PT_SOURCE_CLOCKS, PT_RECOMPUTED,
        PT_TEAM_FIELDS, PT_SELECTION_SHA, _is_sha256, _pt_selection,
        BIG_AL_NCAAF_SPREAD_RULES,
    )
    if not isinstance(live_rows, pd.DataFrame):
        raise TypeError('live_rows must be a DataFrame')
    clock = _clock(now if now is not None else datetime.now(timezone.utc))
    if pd.isna(clock):
        raise ValueError('invalid audit clock')
    time_reasons = Counter()
    all_time_reasons = Counter()
    ages = Counter()
    pt_counts = Counter()
    pt_team_matches = Counter()
    pt_clock = Counter()
    pt_identity = Counter()
    quote_age = Counter()
    quote_lag_s = []
    stale_quote_examples = []
    big_al_ready_dropped_by_age = 0
    samples = {'pt_non_team_selections': [], 'pt_invalid_source_counts': []}
    n_current, n_timing_pass, pt_present = 0, 0, 0
    for _, row in live_rows.iterrows():
        if not _true(row.get('V55_Live_Computed_On_Current_Quote')):
            continue
        n_current += 1
        orig = _clock(row.get('V55_Original_Quote_Timestamp'))
        seen = _clock(row.get('Snapshot_Timestamp'))
        kick = _clock(row.get('Game_Start'))
        if any(pd.isna(x) for x in (orig, seen, kick)):
            time_reasons['MISSING_TIME_FIELD'] += 1
            all_time_reasons['MISSING_TIME_FIELD'] += 1
            continue
        age = (clock - seen).total_seconds()
        quote_to_eval = (seen - orig).total_seconds()
        quote_lag_s.append(quote_to_eval)
        if quote_to_eval < 0: quote_age['QUOTE_FROM_FUTURE']+=1
        elif quote_to_eval <= 300: quote_age['QUOTE_0_5M']+=1
        elif quote_to_eval <= 900: quote_age['QUOTE_5_15M']+=1
        elif quote_to_eval <= 1800: quote_age['QUOTE_15_30M']+=1
        elif quote_to_eval <= 3600: quote_age['QUOTE_30_60M']+=1
        else: quote_age['QUOTE_OVER_60M']+=1
        if age < 0: ages['EVALUATED_IN_FUTURE'] += 1
        elif age <= 300: ages['EVALUATION_0_5M'] += 1
        elif age <= 900: ages['EVALUATION_5_15M'] += 1
        elif age <= 1800: ages['EVALUATION_15_30M'] += 1
        else: ages['EVALUATION_OVER_30M'] += 1
        errors = []
        if orig > seen:
            errors.append('QUOTE_AFTER_EVALUATION')
        if seen > clock:
            errors.append('EVALUATION_IN_FUTURE')
        if clock >= kick:
            errors.append('GAME_STARTED')
        if quote_to_eval > MAX_QUOTE_TO_EVALUATION_S:
            errors.append('QUOTE_TO_EVALUATION_OVER_15M')
        if age > MAX_OBSERVATION_AGE_S:
            errors.append('EVALUATION_OLDER_THAN_5M')
        ready = any(_num(row.get(flag)) == 1 and _num(row.get(flag+'_DataReady')) == 1
                    for flag in BIG_AL_NCAAF_SPREAD_RULES)
        if errors and ready:
            big_al_ready_dropped_by_age += 1
        if quote_to_eval > MAX_QUOTE_TO_EVALUATION_S and len(stale_quote_examples) < 5:
            stale_quote_examples.append({
                'book': _text(row.get('Bookmaker') or row.get('Book')),
                'game': _text(row.get('Game'))[:100],
                'quote_utc': orig.isoformat(),
                'evaluation_utc': seen.isoformat(),
                'quote_to_evaluation_seconds': int(quote_to_eval),
                'big_al_exact_ready': ready,
            })
        if errors:
            time_reasons[errors[0]] += 1
            all_time_reasons.update(errors)
        else:
            n_timing_pass += 1
        # PT formatting diagnostics deliberately run independently of freshness
        # to distinguish missing producer data from valid sources on older rows.
        selection, source_field, pt_problem = _pt_selection(row)
        sha = _text(row.get(PT_SELECTION_SHA))
        if sha:
            pt_identity['SHA_FINGERPRINT_PRESENT'] += 1
            if _is_sha256(sha): pt_identity['VALID_SHA256_FINGERPRINT_FORMAT'] += 1
            else: pt_identity['NON_SHA_VALUE_IN_SHA_FIELD'] += 1
        if pt_problem:
            pt_identity[pt_problem] += 1
            # Source counts are analyzed below even when selection is absent.
            valid_team = False
        else:
            pt_present += 1
            home, away, chosen = (_team(row.get(k)) for k in (
                'Home_Team_Norm', 'Away_Team_Norm', source_field))
            chosen = _team(selection)
            target = _team(row.get('Outcome'))
            valid_team = chosen in (home, away) and home != away and bool(chosen)
            if valid_team:
                pt_team_matches['EXACT_TEAM_MATCH'] += 1
                if chosen != target: pt_team_matches['OPPOSITE_PRICED_SIDE'] += 1
                else: pt_team_matches['MATCHED_PRICED_SIDE'] += 1
            else:
                pt_team_matches['NOT_AN_EXPLICIT_GAME_TEAM'] += 1
                if len(samples['pt_non_team_selections']) < 5:
                    samples['pt_non_team_selections'].append({
                        'selection': selection[:90],
                        'home': _text(row.get('Home_Team_Norm'))[:90],
                        'away': _text(row.get('Away_Team_Norm'))[:90],
                    })
        count = _num(row.get(PT_COUNT))
        if count is None:
            pt_counts['MISSING_OR_NON_NUMERIC'] += 1
            src_valid = False
        elif count == 0:
            pt_counts['ZERO_SOURCES'] += 1
            src_valid = False
        elif count < 0:
            pt_counts['NEGATIVE_SOURCES'] += 1
            src_valid = False
        elif count > 5:
            pt_counts['ABOVE_FIVE_SOURCES'] += 1
            src_valid = False
        elif count != int(count):
            pt_counts['NON_INTEGER_SOURCE_COUNT'] += 1
            src_valid = False
        else:
            pt_counts[f'VALID_{int(count)}_OF_5'] += 1
            src_valid = True
        if not src_valid and len(samples['pt_invalid_source_counts']) < 5:
            samples['pt_invalid_source_counts'].append({
                'selection': (selection or '')[:90], 'source_count': _text(row.get(PT_COUNT))[:30]})
        if valid_team and chosen == target and src_valid:
            clocks = [(col, _clock(row.get(col))) for col in PT_SOURCE_CLOCKS if col in live_rows.columns]
            real = [(col,dt) for col,dt in clocks if pd.notna(dt)]
            if real:
                pt_clock['PRODUCER_SOURCE_CLOCK_PRESENT'] += 1
                t = real[0][1]
                if t > seen: pt_clock['CLOCK_AFTER_EVALUATION'] += 1
                if t >= kick: pt_clock['CLOCK_AFTER_GAME_START'] += 1
                if (seen - t).total_seconds() > 86400: pt_clock['CLOCK_MORE_THAN_24H_BEFORE_EVALUATION'] += 1
            elif _true(row.get(PT_RECOMPUTED)):
                pt_clock['EXPLICIT_CURRENT_QUOTE_RECOMPUTATION'] += 1
            else:
                pt_clock['NATIVE_SOURCE_CLOCK_MISSING'] += 1
    return {
        'status': 'READ_ONLY_DIAGNOSTICS_NO_GATE_CHANGES',
        'rows': int(len(live_rows)),
        'rows_live_computed_on_quote': n_current,
        'rows_passing_connector_timing': n_timing_pass,
        'timing_first_failure': dict(time_reasons),
        'timing_all_conditions': dict(all_time_reasons),
        'evaluation_age_buckets': dict(ages),
        'quote_to_evaluation_age_buckets': dict(quote_age),
        'quote_to_evaluation_seconds_min': min(quote_lag_s) if quote_lag_s else None,
        'quote_to_evaluation_seconds_max': max(quote_lag_s) if quote_lag_s else None,
        'stale_quote_examples': stale_quote_examples,
        'big_al_data_ready_rows_blocked_by_time': big_al_ready_dropped_by_age,
        'pt_identity': dict(pt_identity),
        'pt_team_fields_present': [c for c in PT_TEAM_FIELDS if c in live_rows.columns],
        'pt_sha_field_present': PT_SELECTION_SHA in live_rows.columns,
        'pt_explicit_nonblank_rows': pt_present,
        'pt_team_match': dict(pt_team_matches),
        'pt_source_count_distribution': dict(pt_counts),
        'pt_source_clock_evidence': dict(pt_clock),
        'pt_producer_clock_columns': [c for c in PT_SOURCE_CLOCKS if c in live_rows.columns],
        'pt_recomputed_proof_column_present': PT_RECOMPUTED in live_rows.columns,
        'pt_examples': samples,
        'big_al_exact_columns_present': [c for c in BIG_AL_NCAAF_SPREAD_RULES if c in live_rows.columns],
        'bet_authority_changed': False,
        'historical_qualification': False,
    }
