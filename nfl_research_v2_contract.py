"""NFL Research V2.0 architecture contract.

This contract is intentionally stricter than a version label.  It prevents the
project from rerunning old experiments under new names and preserves the useful
components already established in V1.9.2-V1.9.5 while opening genuinely new
information lanes.

Research V2 has four independent information programs:
  * FUNDAMENTAL/PBP: market-blind play-by-play football intelligence.
  * SYSTEMS: documented/domain/academic/mined situational hypotheses.
  * MARKET: timestamped prospective market microstructure (V1.9.5).
  * EXTERNAL CONTEXT: future QB/injury/weather/participation additions.

Reconciliation is deliberately deferred until independent lanes demonstrate
value.  No V2 component has production authority.
"""
from __future__ import annotations

import hashlib
import json

SOURCE_TAG = "nfl-research-v2.0-foundation-expansion-20261001"
CONTRACT_VERSION = "nfl-research-v2.0-three-brain-new-information-20261001"
PRODUCTION_AUTHORITY = 0
DATA_MAX_SEASON = 2025
SEALED_SEASON = 2026

CONTRACT = {
    "contract_version": CONTRACT_VERSION,
    "source_tag": SOURCE_TAG,
    "status": "READY",
    "production_authority": 0,
    "ncaaf": "UNCHANGED",
    "legacy_nfl": "UNCHANGED",
    "protected_components": {
        "INCUMBENT_CORE": "PROTECTED_BASELINE",
        "V1_9_5_MARKET_CLOCK": "PROTECTED_RESEARCH_INFRASTRUCTURE",
        "BIG_AL_DOCUMENTED_SYSTEMS": "PROTECTED_RESEARCH",
        "PATHI_FOOTBALL_MARKET_TRANSLATIONS": "SHADOW_ONLY_ENGINEERING_TRANSLATION",
        "NCAAF_SYSTEM_MINER_V3_METHODOLOGY": "REUSED_RESEARCH_METHOD",
        "PROSPECTIVE_APPEND_ONLY_LEDGER": "PROTECTED_RESEARCH_INFRASTRUCTURE",
    },
    "new_lanes": {
        "PBP_CORE_V2": "CHALLENGER_MARKET_BLIND",
        "NFL_SYSTEM_LAB_V2": "CHALLENGER_RESEARCH",
        "MARKET_MICROSTRUCTURE": "PROSPECTIVE_RESEARCH",
        "EXTERNAL_CONTEXT": "PLANNED_NEW_INFORMATION",
    },
    "rules": {
        "no_2026_development": True,
        "no_automatic_promotion": True,
        "no_incumbent_mutation": True,
        "new_research_may_add_not_silently_replace": True,
        "do_not_rerun_old_experiment_without_new_information_target_or_validation_design": True,
        "fundamental_system_market_lanes_remain_independent": True,
        "reconciliation_deferred_until_independent_lanes_have_evidence": True,
        "pbp_outcomes_not_authoritative": True,
        "authoritative_scores_and_historical_lines_come_from_existing_nfl_history": True,
        "same_game_pbp_cannot_enter_same_game_predictors": True,
        "current_game_primary_qb_from_pbp_cannot_be_predictor": True,
        "system_discovery_cannot_use_core_or_market_model_predictions": True,
        "documented_systems_must_remain_distinct_from_engineering_translations": True,
        "academic_hypotheses_are_replication_hypotheses_not_production_rules": True,
        "system_boundaries_freeze_before_final_historical_check": True,
        "prospective_retest_cannot_feed_back_into_discovery": True,
    },
    "research_periods": {
        "pbp_oof_validation": [2021, 2022, 2023, 2024, 2025],
        "system_discovery": [2017, 2018, 2019, 2020, 2021, 2022],
        "system_shadow": [2023],
        "system_confirmation": [2024],
        "system_final_historical_check": [2025],
        "sealed": [2026],
    },
    "future_external_context_priority": [
        "pregame_starting_qb_identity",
        "injury_and_availability",
        "weather_wind_temperature",
        "depth_chart_and_participation",
        "travel_body_clock",
    ],
}


def contract_hash() -> str:
    payload = json.dumps(CONTRACT, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def assert_contract() -> dict:
    c = json.loads(json.dumps(CONTRACT))
    if c["production_authority"] != 0:
        raise RuntimeError("NFL_RESEARCH_V2_PRODUCTION_AUTHORITY_MUST_BE_ZERO")
    if not c["rules"]["no_2026_development"]:
        raise RuntimeError("NFL_RESEARCH_V2_2026_MUST_REMAIN_SEALED")
    if c["protected_components"].get("INCUMBENT_CORE") != "PROTECTED_BASELINE":
        raise RuntimeError("NFL_RESEARCH_V2_INCUMBENT_CORE_NOT_PROTECTED")
    if c["protected_components"].get("V1_9_5_MARKET_CLOCK") != "PROTECTED_RESEARCH_INFRASTRUCTURE":
        raise RuntimeError("NFL_RESEARCH_V2_MARKET_CLOCK_NOT_PROTECTED")
    if not c["rules"]["fundamental_system_market_lanes_remain_independent"]:
        raise RuntimeError("NFL_RESEARCH_V2_BRAIN_INDEPENDENCE_BROKEN")
    c["contract_sha256"] = contract_hash()
    return c
