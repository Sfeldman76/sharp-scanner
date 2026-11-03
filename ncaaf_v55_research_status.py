#!/usr/bin/env python3
"""Read-only, source-preserving research-execution status for NCAAF V5.5.

This module NEVER overwrites the bridge's own ``status``, never changes source
certification, and never changes or grants betting authority. Its independent
research status permits downstream readers to distinguish an executed
retrospective study from live-certification readiness.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SEVEN = frozenset({
    "games", "core", "signals", "coverage", "eligibility",
    "rule_evaluations", "universe_manifest",
})
RESEARCH_PREFIX = "research/ncaaf/v55/combined_brains_shadow/input_staging/runs/"
ACCEPTED_PROVENANCE_HOLD = "PROVENANCE_OR_HISTORY_COVERAGE_HOLD"


class ResearchIntegrityError(ValueError):
    """The research-only PASS is prohibited because an invariant failed."""


def classify_bridge_result(result: dict, *, input_prefix: str, expected_inputs=SEVEN) -> dict:
    """Provide *separate* execution/certification statuses without mutating result.

    A valid bridge result with seven immutable GCS inputs and only provenance
    restrictions is eligible for research-only PASS. This is not a certification
    that historical rules were frozen, price quotes were contemporaneous, or
    feature vintages were known. All preexisting source warnings are retained.
    """
    if not isinstance(result, dict):
        raise ResearchIntegrityError("BRIDGE_RESULT_NOT_OBJECT")

    expected = frozenset(expected_inputs)
    if expected != SEVEN:
        raise ResearchIntegrityError("EXPECTED_INPUTS_NOT_SEVEN")

    if not isinstance(input_prefix, str) or not input_prefix.startswith(RESEARCH_PREFIX):
        raise ResearchIntegrityError("UNAPPROVED_RESEARCH_INPUT_PREFIX")

    if result.get("bet_authority") != 0:
        raise ResearchIntegrityError("BET_AUTHORITY_NOT_ZERO")
    if result.get("production_mutation") is not False:
        raise ResearchIntegrityError("PRODUCTION_MUTATION_NOT_FALSE")

    receipts = result.get("input_receipts")
    if not isinstance(receipts, list):
        raise ResearchIntegrityError("MISSING_INPUT_RECEIPTS")
    names = [rec.get("input") if isinstance(rec, dict) else None for rec in receipts]
    counts = Counter(names)
    if len(names) != len(SEVEN) or any(counts[name] != 1 for name in SEVEN):
        raise ResearchIntegrityError("MISSING_OR_DUPLICATED_INPUT_RECEIPTS")
    if set(names) != SEVEN:
        raise ResearchIntegrityError("UNEXPECTED_INPUT_RECEIPTS")

    rows = {}
    for receipt in receipts:
        key = receipt["input"]
        row_count = receipt.get("rows")
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
            raise ResearchIntegrityError(f"MISSING_OR_EMPTY_INPUT:{key}")
        rows[key] = row_count

    bridge_status = result.get("status")
    bridge_reason = result.get("reason")
    if bridge_status != "HOLD" or bridge_reason != ACCEPTED_PROVENANCE_HOLD:
        # This classifier is intentionally narrow: unknown statuses require review.
        raise ResearchIntegrityError(
            f"UNRECOGNIZED_BRIDGE_RESULT:{bridge_status}:{bridge_reason}"
        )

    warnings = result.get("warnings", [])
    if not isinstance(warnings, list):
        raise ResearchIntegrityError("WARNINGS_NOT_LIST")

    return {
        "schema_version": 1,
        "research_execution_status": "PASS_RESEARCH_ONLY",
        "research_mode": "RETROSPECTIVE_EXPLORATORY",
        "historical_certification_status": "HOLD_UNCERTIFIED",
        "betting_authority_status": "BLOCKED",
        "bridge_original_status": bridge_status,
        "bridge_original_reason": bridge_reason,
        "bridge_original_warnings": list(warnings),
        "verified_input_receipts": sorted(rows),
        "input_rows": dict(sorted(rows.items())),
        "input_prefix": input_prefix,
        "bet_authority": 0,
        "production_mutation": False,
        "may_qualify_live_systems": False,
        "certifies_point_in_time_features": False,
        "certifies_frozen_rule_replay": False,
        "certifies_historical_sportsbook_returns": False,
        "core_changed": False,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_audit(path: Path, audit: dict) -> Path:
    """Write a local research report. Refuses to overwrite existing audit files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path
