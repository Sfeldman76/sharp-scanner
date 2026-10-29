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
from collections import Counter
from datetime import datetime, timezone

# These are the three canonical *play-on-side* college football exact rules
# defined by the existing V13 system builder, not generic context/enhancers.
BIG_AL_NCAAF_SPREAD_RULES = (
    'BigAl_CF1_Week2Home42Win',
    'BigAl_CF2_LateSeasonRevengeDog',
    'BigAl_CF3_Fade19PlusFavoriteUpsetLoss',
)

# Explicit, sourced selection. Not the historical Miner/PT 'index' overlay.
PT_SELECTION = 'NCAAF_PT_Five_Source_Selection_SHA'
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
MAX_PT_SOURCE_AGE_S = 24 * 60 * 60


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
        if PT_SELECTION in out.columns and _text(row.get(PT_SELECTION)):
            seen['PT_EXPLICIT_SELECTION_PRESENT']+=1
            count=_num(row.get(PT_COUNT))
            if count is None or count < 1 or count>5 or int(count)!=count:
                reject['PT_INVALID_SOURCE_COUNT']+=1
            else:
                pt_target=_team(row.get(PT_SELECTION))
                if pt_target not in (home,away):
                    reject['PT_SELECTION_NOT_AN_EXPLICIT_GAME_TEAM']+=1
                elif pt_target!=target:
                    # The same selection will be found on its own side row;
                    # NEVER take the opposing priced side.
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
                    elif (evaluated-original_pt).total_seconds()>MAX_PT_SOURCE_AGE_S:
                        reject['PT_PRODUCER_SOURCE_OVER_24H_OLD']+=1
                    elif ('PREDICTION_TRACKER','PT_FIVE_SOURCE_SELECTION') not in existing_ids:
                        emitted.append(dict(base,source_family='PREDICTION_TRACKER',
                                            signal_id='PT_FIVE_SOURCE_SELECTION',
                                            pt_source_observed_at_utc=original_pt.isoformat(),
                                            source_count=int(count)))
                        counts['PREDICTION_TRACKER']+=1
        rewritten.append(json.dumps(existing+emitted,sort_keys=True,default=str))
    out[FIELD]=rewritten
    audit={'status':'EXPLICIT_LIVE_SOURCE_SIGNALS_RESEARCH_ONLY',
           'rows':int(len(out)), 'by_family':dict(counts),
           'source_flags':dict(seen), 'rejections':dict(reject),
           'diagnostic_hints':dict(hints),
           'big_al_exact_columns_present':[c for c in BIG_AL_NCAAF_SPREAD_RULES if c in out.columns],
           'pt_selection_column_present':PT_SELECTION in out.columns,
           'pt_producer_clock_columns':selected_clocks,
           'pt_explicit_recomputed_column_present':PT_RECOMPUTED in out.columns,
           'pathi_status':'EXPLICIT_DIRECTIONAL_PRODUCER_REQUIRED_NO_CONTEXT_PROMOTION',
           'stat_status':'EXPLICIT_DIRECTIONAL_PRODUCER_REQUIRED',
           'market_rich_status':'EXPLICIT_DIRECTIONAL_PRODUCER_REQUIRED',
           'historical_qualification':False, 'bet_authority_changed':False}
    return out,audit
