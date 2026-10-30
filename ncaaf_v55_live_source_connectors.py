"""Explicit source-level NCAAF spread-origination bridge, research only.

Only deterministic, play-on-side CF Big Al flags and an explicitly identified,
source-timestamped Prediction Tracker selection are generated here.

The existing ncaaf_v55_producer_handoff and ncaaf_v55_other_shadow_signals
modules provide additional normalization and mandatory independent quote
attestation. This module NEVER grants historical qualification or bet authority.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone

# These are the three canonical *play-on-side* college football exact rules
# defined by the existing V13 system builder, not generic context/enhancers.
BIG_AL_NCAAF_SPREAD_RULES = (
    'BigAl_CF1_Week2Home42Win',
    'BigAl_CF2_LateSeasonRevengeDog',
    'BigAl_CF3_Fade19PlusFavoriteUpsetLoss',
)

# SHA is only a fingerprint of the selection. It is NEVER a team selection.
# These fields are an interface contract for an upstream PT producer; the
# adapter does not manufacture the fields when they are absent.
PT_SELECTION_SHA = 'NCAAF_PT_Five_Source_Selection_SHA'
PT_TEAM_FIELDS = (
    'NCAAF_PT_Five_Source_Selected_Team',
    'NCAAF_PT_Five_Source_Selection_Team',
    'V55_PT_Five_Source_Selected_Team',
    'V55_PT_Explicit_Team_Selection',
)
PT_SELECTION = PT_TEAM_FIELDS[0]  # kept for modules importing PT_SELECTION
PT_COUNT = 'NCAAF_PT_Five_Source_Count'
# No PT event is reconstructed if the original producer clock isn't available.
PT_SOURCE_CLOCKS = (
    'NCAAF_PT_Five_Source_Observed_At_UTC',
    'NCAAF_PT_Five_Source_Snapshot_UTC',
    'NCAAF_PT_Five_Source_Source_Timestamp',
    'V55_PT_Source_Observed_At_UTC',
)
# An explicit boolean from the producer can also attest recomputation on the
# live quote. The generic live evaluator boolean does NOT do that for PT.
PT_RECOMPUTED = 'V55_PT_Five_Source_Recomputed_On_Current_Quote'
FIELD = 'V55_Independent_Shadow_Signals_JSON'
MAX_QUOTE_TO_EVALUATION_S = 900
MAX_OBSERVATION_AGE_S = 300
# No age cutoff for the latest verified weekly PT source.


def _text(v):
    if v is None: return ''
    try:
        import pandas as pd
        if pd.isna(v): return ''
    except (TypeError, ValueError):
        pass
    return str(v).strip()


def _team(v):
    import re
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]+', ' ', _text(v).lower())).strip()


def _clock(v):
    import pandas as pd
    try:
        out = pd.to_datetime(v, utc=True, errors='coerce')
        if isinstance(out, pd.Timestamp) and pd.notna(out): return out
    except (ValueError, TypeError, OverflowError):
        pass
    return pd.NaT


def _true(v):
    return v is True or type(v).__name__ == 'bool_' and str(v) == 'True' or _text(v) in ('1', 'True', 'true')


def _num(v):
    try:
        f=float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError): return None


def _is_sha256(v):
    return bool(re.fullmatch(r'[a-fA-F0-9]{64}', _text(v)))


PT_FAIR_HOME_MARGIN = 'NCAAF_PT_Five_Source_Margin_Home'
PT_DERIVED_FIELD = 'PT_DERIVED_FAIR_AT_VERIFIED_SPREAD'
MIN_PT_CONTRIBUTORS = 3
MIN_PT_FAIR_EDGE_POINTS = 2.0


def _pt_model_side(row):
    """Compute a research-only ATS side from PT's numeric home-margin model.

    This is NOT a Prediction Tracker published team pick, nor proof that the
    contributing source forecasts all agree. It requires an original market
    side/spread; candidate publication still requires a separate source clock.
    """
    # Source rows must already be attested against the actual scanner quote.
    if not _true(row.get('V55_Original_Quote_Provenance_Verified')):
        return None, 'PT_MODEL_ORIGINAL_QUOTE_NOT_VERIFIED'
    count = _num(row.get(PT_COUNT))
    if count is None or count != int(count) or not MIN_PT_CONTRIBUTORS <= count <= 5:
        return None, 'PT_MODEL_INSUFFICIENT_SOURCES'
    if not _is_sha256(row.get(PT_SELECTION_SHA)):
        return None, 'PT_MODEL_FINGERPRINT_NOT_VERIFIED'
    fair_home = _num(row.get(PT_FAIR_HOME_MARGIN))
    side_spread = _num(row.get('Value'))
    home = _team(row.get('Home_Team_Norm'))
    away = _team(row.get('Away_Team_Norm'))
    offered_side = _team(row.get('Outcome'))
    if fair_home is None or side_spread is None or not home or not away or home == away or offered_side not in (home,away):
        return None, 'PT_MODEL_MARGIN_OR_VERIFIED_SIDE_MISSING'
    home_spread = side_spread if offered_side == home else -side_spread
    home_cover_edge_points = fair_home + home_spread
    if abs(home_cover_edge_points) < MIN_PT_FAIR_EDGE_POINTS:
        return None, 'PT_MODEL_EDGE_BELOW_TWO_POINTS'
    return {'team': home if home_cover_edge_points > 0 else away,
            'predicted_home_margin':fair_home,
            'verified_home_spread':home_spread,
            'home_cover_edge_points':home_cover_edge_points,
            'source_count':int(count)}, None


def _pt_selection(row):
    # A genuine PT producer must expose a named team separate from the SHA.
    raw = [(field, _text(row.get(field))) for field in PT_TEAM_FIELDS]
    values = [(field, v) for field, v in raw if v]
    if not values:
        # PT supplies a numeric fair margin, not a published team selection.
        # Derive a separate research challenger against the ACTUAL verified
        # sportsbook side. This never invents the required source clock.
        model, failure = _pt_model_side(row)
        if model is not None:
            return model['team'], PT_DERIVED_FIELD, None
        return None, None, failure if _num(row.get(PT_FAIR_HOME_MARGIN)) is not None else (
            'PT_SHA_ONLY_NO_TEAM_SELECTION' if _is_sha256(row.get(PT_SELECTION_SHA)) else 'PT_NO_TEAM_SELECTION')
    if any(_is_sha256(v) for _, v in values):
        return None, None, 'PT_HASH_IN_TEAM_FIELD'
    canonical = {_team(v) for _, v in values}
    if len(canonical) != 1:
        return None, None, 'PT_CONFLICTING_EXPLICIT_TEAM_FIELDS'
    return values[0][1], values[0][0], None


def _decode(raw):
    if raw is None or _text(raw) in ('', 'nan', 'None'): return []
    if isinstance(raw, str): raw=json.loads(raw)
    if not isinstance(raw, list): raise ValueError('not a JSON array')
    return raw


def attach_live_source_signals(live_rows, *, now=None):
    """Emit explicit directional observation payloads in the existing column.

    Input rows must already be the verified original-book, original-spread,
    live-evaluated SIDE rows from _v55_evaluate_existing_live_systems.
    The downstream bridge independently validates the full two-sided quote.
    """
    import pandas as pd
    if not isinstance(live_rows, pd.DataFrame):
        raise TypeError('live_rows must be pandas DataFrame')
    clock=_clock(now if now is not None else datetime.now(timezone.utc))
    if pd.isna(clock): raise ValueError('invalid current time')
    out=live_rows.copy()
    counts=Counter(); reject=Counter(); seen=Counter(); hints=Counter()
    selected_clocks=[c for c in PT_SOURCE_CLOCKS if c in out.columns]
    explicit_team_columns=[c for c in PT_TEAM_FIELDS if c in out.columns]
    if FIELD in out:
        pass
    else:
        out[FIELD]='[]'
    if out.empty:
        return out, {'status':'NO_LIVE_ROWS', 'by_family':{}, 'rejections':{},
                     'producer_clock_columns':selected_clocks,
                     'historical_qualification':False, 'bet_authority_changed':False}
    rewritten=[]
    for _,row in out.iterrows():
        try: existing=_decode(row.get(FIELD))
        except (TypeError, ValueError):
            reject['EXISTING_INVALID_JSON_PRESERVED_FAIL_CLOSED']+=1
            rewritten.append(row.get(FIELD))
            continue
        # Never silently discard existing explicit per-family evidence.
        emitted=[]
        existing_ids={(str(e.get('source_family') or ''),str(e.get('signal_id') or ''))
                      for e in existing if isinstance(e,dict)}
        if not _true(row.get('V55_Live_Computed_On_Current_Quote')):
            hints['NOT_LIVE_ON_VERIFIED_QUOTE']+=1
            rewritten.append(json.dumps(existing,default=str));continue
        home=_team(row.get('Home_Team_Norm'))
        away=_team(row.get('Away_Team_Norm'))
        target=_team(row.get('Outcome'))
        book=(_text(row.get('Bookmaker')) or _text(row.get('Book'))).lower()
        line=_num(row.get('Value'))
        original=_clock(row.get('V55_Original_Quote_Timestamp'))
        evaluated=_clock(row.get('Snapshot_Timestamp'))
        kick=_clock(row.get('Game_Start'))
        market=_text(row.get('Market')).lower()
        if market and market not in ('spread','spreads'):
            reject['NON_SPREAD_SOURCE_MARKET']+=1
            rewritten.append(json.dumps(existing,default=str));continue
        if not home or not away or home==away or target not in (home,away):
            reject['MISSING_OR_INVALID_GAME_SIDE']+=1
            rewritten.append(json.dumps(existing,default=str));continue
        if not book or line is None:
            reject['ORIGINAL_BOOK_OR_SPREAD_MISSING']+=1
            rewritten.append(json.dumps(existing,default=str));continue
        if any(pd.isna(x) for x in (original,evaluated,kick)):
            reject['MISSING_ORIGINAL_PREGAME_TIMESTAMPS']+=1
            rewritten.append(json.dumps(existing,default=str));continue
        if not (original<=evaluated<=clock<kick and
                (evaluated-original).total_seconds()<=MAX_QUOTE_TO_EVALUATION_S and
                (clock-evaluated).total_seconds()<=MAX_OBSERVATION_AGE_S):
            reject['ORIGINAL_QUOTE_OR_EVALUATION_STALE']+=1
            rewritten.append(json.dumps(existing,default=str));continue
        base={'target':_text(row.get('Outcome')),
              'observed_at_utc':evaluated.isoformat(),
              'observed_quote_utc':original.isoformat()}
        # Exact deterministic system state, on the existing builder's play-on side.
        for flag in BIG_AL_NCAAF_SPREAD_RULES:
            if flag not in out.columns: continue
            if _num(row.get(flag))!=1: continue
            seen['BIG_AL_FLAG_FIRED']+=1
            ready=flag+'_DataReady'
            if ready not in out.columns or _num(row.get(ready))!=1:
                reject['BIG_AL_DATA_READY_NOT_VERIFIED']+=1
                continue
            if ('BIG_AL',flag) in existing_ids:continue
            emitted.append(dict(base,source_family='BIG_AL',signal_id=flag))
            counts['BIG_AL']+=1
        # Prediction Tracker requires an explicit named selection and a
        # producer-native source clock (or explicit on-quote recompute proof).
        # A historical index or blank selection is NOT a directional prediction.
        pt_raw, pt_field, pt_problem = _pt_selection(row)
        if pt_problem:
            # No PT candidate originates from a hash, blank, or conflict.
            if _is_sha256(row.get(PT_SELECTION_SHA)):
                hints['PT_SHA_FINGERPRINT_PRESENT'] += 1
            if pt_problem != 'PT_NO_TEAM_SELECTION':
                hints[pt_problem] += 1
        else:
            if pt_field == PT_DERIVED_FIELD:
                seen['PT_DERIVED_AT_VERIFIED_LINE'] += 1
            else:
                seen['PT_EXPLICIT_TEAM_PRESENT'] += 1
            count=_num(row.get(PT_COUNT))
            if count is None or count < 1 or count>5 or int(count)!=count:
                reject['PT_INVALID_SOURCE_COUNT']+=1
            else:
                pt_target=_team(pt_raw)
                if pt_target not in (home,away):
                    reject['PT_SELECTION_NOT_AN_EXPLICIT_GAME_TEAM']+=1
                elif pt_target!=target:
                    hints['PT_SELECTION_OPPOSITE_ROW']+=1
                else:
                    original_pt=None
                    for col in selected_clocks:
                        test=_clock(row.get(col))
                        if pd.notna(test): original_pt=test;break
                    if original_pt is None and _true(row.get(PT_RECOMPUTED)):
                        original_pt=evaluated
                    if original_pt is None:
                        reject['PT_PRODUCER_SOURCE_CLOCK_MISSING']+=1
                    elif not original_pt<=evaluated or not original_pt<kick:
                        reject['PT_PRODUCER_CLOCK_AFTER_EVALUATION_OR_KICK']+=1
                    else:
                        # PT is uploaded weekly.
                        # Keep the latest verified upload until superseded.
                        # Source-clock, kickoff, quote and evaluation
                        # validation remain mandatory.
                        # Research-only; no betting authority.
                        model = _pt_model_side(row)[0] if pt_field == PT_DERIVED_FIELD else None
                        signal_id = ('PT_FIVE_SOURCE_FAIR_EDGE_2PLUS_SHADOW' if model is not None
                                     else 'PT_FIVE_SOURCE_SELECTION')
                        if ('PREDICTION_TRACKER',signal_id) not in existing_ids:
                            additional = ({'pt_predicted_home_margin':model['predicted_home_margin'],
                                           'pt_verified_home_spread':model['verified_home_spread'],
                                           'pt_fair_edge_points_home':model['home_cover_edge_points'],
                                           'pt_agreement_verified':False}
                                          if model is not None else {})
                            emitted.append(dict(base,source_family='PREDICTION_TRACKER',
                                                signal_id=signal_id,
                                                pt_source_observed_at_utc=original_pt.isoformat(),
                                                pt_selection_field=pt_field,
                                                source_count=int(count),**additional))
                            counts['PREDICTION_TRACKER']+=1
        rewritten.append(json.dumps(existing+emitted,sort_keys=True,default=str))
    out[FIELD]=rewritten
    audit={'status':'EXPLICIT_LIVE_SOURCE_SIGNALS_RESEARCH_ONLY',
           'rows':int(len(out)), 'by_family':dict(counts),
           'source_flags':dict(seen), 'rejections':dict(reject),
           'diagnostic_hints':dict(hints),
           'big_al_exact_columns_present':[c for c in BIG_AL_NCAAF_SPREAD_RULES if c in out.columns],
           'pt_selection_column_present':bool(explicit_team_columns),
           'pt_team_columns_present':explicit_team_columns,
           'pt_sha_field_present':PT_SELECTION_SHA in out.columns,
           'pt_derived_field_supported':PT_DERIVED_FIELD,
           'pt_model_fair_margin_available':PT_FAIR_HOME_MARGIN in out.columns,
           'pt_fair_edge_min_points':MIN_PT_FAIR_EDGE_POINTS,
           'pt_min_forecast_sources':MIN_PT_CONTRIBUTORS,
           'pt_direction_basis':'MODEL_HOME_MARGIN_PLUS_VERIFIED_HOME_SPREAD_NOT_FOUR_OF_FIVE_VOTES',
           'pt_native_source_timestamp_still_required':True,
           'pt_producer_clock_columns':selected_clocks,
           'pt_explicit_recomputed_column_present':PT_RECOMPUTED in out.columns,
           'pathi_status':'EXPLICIT_DIRECTIONAL_PRODUCER_REQUIRED_NO_CONTEXT_PROMOTION',
           'stat_status':'EXPLICIT_DIRECTIONAL_PRODUCER_REQUIRED',
           'market_rich_status':'EXPLICIT_DIRECTIONAL_PRODUCER_REQUIRED',
           'historical_qualification':False, 'bet_authority_changed':False}
    # Diagnostic-only: audit input distribution without changing any signal,
    # original-quote verification, or candidate origination decision.
    from ncaaf_v55_source_diagnostics import audit_live_source_inputs
    audit['input_diagnostics'] = audit_live_source_inputs(live_rows, now=clock)
    return out,audit
