"""NFL V1.9.2 research architecture contract.

This file exists to prevent new NFL research from silently deleting or replacing
useful components already established in V1.4-V1.9.1.  It is metadata/control
plane only: it does not select bets and has zero production authority.
"""
from __future__ import annotations

import hashlib
import json

SOURCE_TAG = "nfl-research-contract-v1.9.2-protected-architecture-20261001"
CONTRACT_VERSION = "nfl-research-v1.9.2-protected-architecture-20261001"
PRODUCTION_AUTHORITY = 0

# Status vocabulary is intentionally explicit.  PROTECTED does not mean
# production-approved; it means the component may not be silently removed by a
# later research revision.  Replacement requires an explicit, evidence-backed
# migration entry.
COMPONENTS = {
    "MARKET_BASELINE": {
        "family": "MARKET",
        "status": "PROTECTED_BASELINE",
        "independent_vote_family": "MARKET",
        "purpose": "Consensus market reference for spread, total and H2H probability.",
    },
    "CORE_DIRECT_MARGIN": {
        "family": "CORE",
        "status": "PROTECTED_RESEARCH",
        "independent_vote_family": "CORE",
        "purpose": "Independent direct prediction of game margin; does not use market as target.",
    },
    "CORE_SCORE_DERIVED_MARGIN": {
        "family": "CORE",
        "status": "PROTECTED_RESEARCH",
        "independent_vote_family": "CORE",
        "purpose": "Independent team-score decomposition; diagnostic within CORE, not a second vote.",
    },
    "CORE_DIRECT_TOTAL": {
        "family": "CORE",
        "status": "PROTECTED_RESEARCH",
        "independent_vote_family": "CORE",
        "purpose": "Independent direct prediction of total points.",
    },
    "CORE_SCORE_DERIVED_TOTAL": {
        "family": "CORE",
        "status": "PROTECTED_RESEARCH",
        "independent_vote_family": "CORE",
        "purpose": "Score-derived total; diagnostic within CORE, not a second vote.",
    },
    "CORE_H2H_PROBABILITY": {
        "family": "CORE",
        "status": "PROTECTED_RESEARCH",
        "independent_vote_family": "CORE",
        "purpose": "Independent H2H probability challenger evaluated against no-vig market probability.",
    },
    "STATS_MARKET_RESIDUAL": {
        "family": "STAT",
        "status": "PROTECTED_RESEARCH",
        "independent_vote_family": "STAT",
        "purpose": "Predicts bounded statistical correction to the market rather than replacing CORE.",
    },
    "BIG_AL_DOCUMENTED_SYSTEMS": {
        "family": "BIG_AL",
        "status": "PROTECTED_RESEARCH",
        "independent_vote_family": "BIG_AL",
        "purpose": "Only documented NFL system definitions; no invented Big Al rules.",
    },
    "PATHI_NFL_ENGINEERING_TRANSLATIONS": {
        "family": "PATHI",
        "status": "SHADOW_ONLY",
        "independent_vote_family": "PATHI",
        "purpose": "Engineering translations only; not represented as verified Pathi NFL teachings.",
    },
    "RESIDUAL_MINER": {
        "family": "MINER",
        "status": "CHALLENGER",
        "independent_vote_family": "MINER",
        "purpose": "Searches for repeatable market residual error, not raw favorite/dog hit-rate patterns.",
    },
    "CONTEXT_FAMILIES": {
        "family": "CONTEXT",
        "status": "PROTECTED_RESEARCH_INPUT",
        "independent_vote_family": None,
        "purpose": "Division/rematch, venue role, rest, opponent-adjusted, matchup, pace, turnover, scoring composition, discipline, trend and volatility context.",
    },
    "UNCERTAINTY": {
        "family": "UNCERTAINTY",
        "status": "PROTECTED_RESEARCH",
        "independent_vote_family": None,
        "purpose": "Empirical error intervals and internal model disagreement diagnostics.",
    },
    "SOURCE_RECONCILIATION": {
        "family": "RECONCILIATION",
        "status": "PROTECTED_INFRASTRUCTURE",
        "independent_vote_family": None,
        "purpose": "Prevents correlated signals in one family from being counted as independent evidence.",
    },
    "PROSPECTIVE_LEDGER": {
        "family": "LEDGER",
        "status": "PROTECTED_INFRASTRUCTURE",
        "independent_vote_family": None,
        "purpose": "Immutable pregame snapshots and append-only result grading.",
    },
    "V1_8_ARBITRATION_CLASSIFIER": {
        "family": "ARBITRATION",
        "status": "RETIRED_RESEARCH",
        "independent_vote_family": None,
        "purpose": "Preserved in history; V1.8 evidence did not support predictive authority.",
    },
}

RULES = {
    "new_research_may_add_not_silently_replace": True,
    "core_and_score_are_one_vote_family": True,
    "market_and_core_remain_separate": True,
    "stats_residual_does_not_replace_core": True,
    "systems_require_documented_or_explicit_research_definition": True,
    "miner_must_search_residual_error": True,
    "no_automatic_promotion": True,
    "no_2026_retuning_of_v1_9_1_frozen_holdout": True,
    "future_models_informed_by_observed_2026_start_new_prospective_clock": True,
    "production_authority": 0,
    "ncaaf": "UNCHANGED",
    "legacy_nfl": "UNCHANGED",
}


def contract_hash() -> str:
    payload = json.dumps(
        {"version": CONTRACT_VERSION, "components": COMPONENTS, "rules": RULES},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def contract_report() -> dict:
    independent = sorted(
        {v["independent_vote_family"] for v in COMPONENTS.values() if v.get("independent_vote_family")}
    )
    protected = sorted(
        k for k, v in COMPONENTS.items() if str(v.get("status", "")).startswith("PROTECTED")
    )
    return {
        "status": "READY",
        "source_tag": SOURCE_TAG,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": contract_hash(),
        "protected_components": protected,
        "independent_vote_families": independent,
        "component_count": len(COMPONENTS),
        "rules": RULES,
        "production_authority": PRODUCTION_AUTHORITY,
    }


def assert_contract() -> dict:
    required = {
        "MARKET_BASELINE",
        "CORE_DIRECT_MARGIN",
        "CORE_SCORE_DERIVED_MARGIN",
        "STATS_MARKET_RESIDUAL",
        "BIG_AL_DOCUMENTED_SYSTEMS",
        "RESIDUAL_MINER",
        "UNCERTAINTY",
        "PROSPECTIVE_LEDGER",
    }
    missing = sorted(required - set(COMPONENTS))
    if missing:
        raise RuntimeError("NFL_V1_9_2_PROTECTED_COMPONENT_MISSING " + str(missing))
    if COMPONENTS["CORE_DIRECT_MARGIN"]["independent_vote_family"] != "CORE":
        raise RuntimeError("NFL_V1_9_2_CORE_FAMILY_CONTRACT_BROKEN")
    if COMPONENTS["CORE_SCORE_DERIVED_MARGIN"]["independent_vote_family"] != "CORE":
        raise RuntimeError("NFL_V1_9_2_SCORE_DOUBLE_COUNT_CONTRACT_BROKEN")
    if COMPONENTS["STATS_MARKET_RESIDUAL"]["independent_vote_family"] == "CORE":
        raise RuntimeError("NFL_V1_9_2_STAT_CORE_COLLAPSE_CONTRACT_BROKEN")
    if int(RULES.get("production_authority", 1)) != 0:
        raise RuntimeError("NFL_V1_9_2_PRODUCTION_AUTHORITY_MUST_BE_ZERO")
    return contract_report()
