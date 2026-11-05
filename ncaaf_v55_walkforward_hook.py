"""Permanent research-only hook: materialize verified chronological selections in Heavy Research.

This hook reuses the already-computed miner run; no second discovery or CORE refit.
"""
from __future__ import annotations
import csv
import io
import json
import os
from ncaaf_v55_rebuild_walkforward_evidence import validate_inputs

APPROVED = 'research/ncaaf/v55/combined_brains_shadow/input_staging/runs/'

def materialize_walkforward_research(unified_results, storage_client, bucket_name='sharp-models'):
    import pandas as pd
    prefix = os.environ.get('NCAAF_BRAIN_HISTORY_INPUT_PREFIX','').removeprefix('gs://'+bucket_name+'/').rstrip('/')
    if not prefix.startswith(APPROVED):
        raise ValueError('WALKFORWARD_CANONICAL_INPUT_NOT_PINNED')
    raw = storage_client.bucket(bucket_name).blob(prefix+'/games.csv').download_as_bytes()
    canonical_games=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    def as_dict(name):
        frame=unified_results.get(name)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError('WALKFORWARD_OUTPUT_MISSING:'+name)
        # Pandas values with missingness remain text/NaN; identities/years are checked inside validator.
        return frame.astype(object).where(pd.notna(frame),'').to_dict('records')
    decisions, summaries, audit = validate_inputs(
        canonical_games,
        as_dict('season_forward_windows_v5'),
        as_dict('season_forward_frozen_rules_v5'),
        as_dict('retrospective_game_decision_ledger_v5'),
        {'audit_v5':unified_results.get('audit_v5') or {}},
    )
    audit['canonical_input_prefix']='gs://'+bucket_name+'/'+prefix
    audit['materialized_inside_same_research_execution']=True
    # Existing shadow publisher writes DataFrames with immutable timestamps.
    return {'reconstructed_walkforward_games_v1':pd.DataFrame(decisions),
            'reconstructed_walkforward_market_year_v1':pd.DataFrame(summaries),
            'reconstructed_walkforward_audit_v1':audit}
