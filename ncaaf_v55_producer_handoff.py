"""NCAAF V5.5: normalize explicit pregame directional producer evidence.

Research only. Never synthesize directions from descriptive flags or model scores.
Per-brain JSON producers may populate the generic column or family-specific
columns. Preserve original evidence; the downstream quote matcher is authoritative.
"""
from __future__ import annotations
import json
from collections import Counter

FAMILIES = ('BIG_AL', 'PATHI', 'PREDICTION_TRACKER', 'STAT', 'MARKET_RICH')
GENERIC = 'V55_Independent_Shadow_Signals_JSON'
PRODUCER_COLUMNS = {
    'BIG_AL': ('V55_BigAl_Directional_Signals_JSON', 'V55_BigAl_Exact_Firings_JSON'),
    'PATHI': ('V55_Pathi_Directional_Signals_JSON', 'V55_Pathi_Exact_Firings_JSON'),
    'PREDICTION_TRACKER': ('V55_PT_Directional_Signals_JSON', 'V55_PredictionTracker_Directional_Signals_JSON'),
    'STAT': ('V55_Stat_Directional_Signals_JSON', 'V55_STAT_Directional_Signals_JSON'),
    'MARKET_RICH': ('V55_MarketRich_Directional_Signals_JSON', 'V55_Market_Rich_Directional_Signals_JSON'),
}
REQUIRED = ('signal_id', 'target', 'observed_at_utc', 'observed_quote_utc')


def _decode(value):
    if value is None: return []
    if isinstance(value, str):
        if not value.strip(): return []
        return json.loads(value)
    return value


def normalize_producer_signals(live_rows):
    """Copy DataFrame with canonical explicit signal column; return audit.

    A configured feed is *not* an accepted signal. The existing source-side,
    quote-vintage and paired-book checks must still accept it downstream.
    """
    import pandas as pd
    if not isinstance(live_rows, pd.DataFrame):
        raise TypeError('live_rows must be a DataFrame')
    out = live_rows.copy()
    sources = {fam: [c for c in cols if c in out.columns]
               for fam, cols in PRODUCER_COLUMNS.items()}
    counts, rejected = Counter(), Counter()
    encoded = []
    for _, row in out.iterrows():
        combined = []
        candidates = [(GENERIC, None)] if GENERIC in out.columns else []
        candidates += [(col, family) for family, cols in sources.items() for col in cols]
        for col, constrained in candidates:
            try: val = _decode(row.get(col))
            except (ValueError, TypeError):
                rejected[col+':INVALID_JSON'] += 1
                continue
            if not isinstance(val, list):
                if val is not None: rejected[col+':NOT_LIST'] += 1
                continue
            for signal in val:
                if not isinstance(signal, dict):
                    rejected[col+':NOT_OBJECT'] += 1
                    continue
                family = str(signal.get('source_family') or constrained or '').upper().strip()
                if family not in FAMILIES or (constrained and family != constrained):
                    rejected[col+':FAMILY_MISMATCH'] += 1
                    continue
                if any(not str(signal.get(field) or '').strip() for field in REQUIRED):
                    rejected[col+':MISSING_DIRECTIONAL_PROVENANCE'] += 1
                    continue
                event = dict(signal)
                event['source_family'] = family
                combined.append(event)
                counts[family] += 1
        encoded.append(json.dumps(combined, sort_keys=True, default=str))
    out[GENERIC] = encoded
    audit = {
        'status': 'NORMALIZED_EXPLICIT_PRODUCERS_RESEARCH_ONLY',
        'rows': len(out), 'producer_columns_present': sources,
        'generic_column_present': GENERIC in live_rows.columns,
        'signals_structurally_complete': dict(counts),
        'rejections': dict(rejected),
        'producer_contract': 'EXPLICIT_SIDE_AND_ORIGINAL_QUOTE_REQUIRED',
        'historical_qualification': False,
        'bet_authority_changed': False,
    }
    return out, audit
