"""Early, read-only seven-input bridge validation for *manual* NCAAF Heavy Research.

Must execute before CORE training or Miner research.  A provenance HOLD can
be a successful exploratory replay, but never authorizes betting or certifies
point-in-time inputs. No production/live artifact is changed by this module.
"""
from __future__ import annotations

import os
from typing import Mapping, Any

REQUIRED_PREFIX = 'research/ncaaf/v55/combined_brains_shadow/input_staging/runs/'


class PreflightBlocked(RuntimeError):
    """Fail fast rather than run expensive research with a disconnected bridge."""


def check_manual_heavy_bridge(*, storage_client: Any, bucket_name: str = "sharp-models",
                              log_func=print, env: Mapping[str, str] | None = None,
                              suppress_audit_publication: bool = False) -> dict:
    """Actually load all seven inputs and execute the *real* historical bridge.

    Expected exploratory result is HOLD with reason PROVENANCE_OR_HISTORY_COVERAGE_HOLD,
    subsequently classified independently as PASS_RESEARCH_ONLY.  Other HOLD
    conditions fail early.  This remains a shadow data integrity check; the
    historical certification and source vintage limitations are preserved.
    """
    env = dict(os.environ if env is None else env)
    prefix = str(env.get('NCAAF_BRAIN_HISTORY_INPUT_PREFIX') or '').strip()
    if not prefix.startswith(REQUIRED_PREFIX):
        raise PreflightBlocked('ARCHIVE_PREFIX_NOT_CONFIGURED_OR_NOT_IMMUTABLE')
    if not prefix[len(REQUIRED_PREFIX):]:
        raise PreflightBlocked('ARCHIVE_RUN_ID_MISSING')

    try:
        from ncaaf_heavy_brain_bridge import run_heavy_brain_shadow
        from ncaaf_v55_research_status import classify_bridge_result
    except (ImportError, SyntaxError) as exc:
        raise PreflightBlocked('HISTORICAL_BRIDGE_IMPORT_FAILED:' + str(exc)) from exc

    # Runtime bridge wrapper emits research-only status if explicitly enabled;
    # local read-only smoke can force the wrapper off without changing job config.
    if suppress_audit_publication:
        from unittest.mock import patch
        with patch.dict(os.environ, {'NCAAF_V55_RESEARCH_STATUS_ENABLED': '0'}, clear=False):
            outcome = run_heavy_brain_shadow(
                storage_client=storage_client, bucket_name=bucket_name,
                log_func=log_func, input_prefix=prefix, env=env, publish=False)
    else:
        outcome = run_heavy_brain_shadow(
            storage_client=storage_client, bucket_name=bucket_name,
            log_func=log_func, input_prefix=prefix, env=env, publish=False)

    try:
        audit = classify_bridge_result(outcome, input_prefix=prefix)
    except (ValueError, KeyError, TypeError) as exc:
        reason = outcome.get('reason') if isinstance(outcome, dict) else 'INVALID_BRIDGE_OUTPUT'
        raise PreflightBlocked('SEVEN_INPUT_BRIDGE_INVALID:' + str(exc) + '; bridge_reason=' + str(reason)) from exc

    if audit.get('research_execution_status') != 'PASS_RESEARCH_ONLY':
        raise PreflightBlocked('RESEARCH_STATUS_NOT_PASS')
    if audit.get('historical_certification_status') != 'HOLD_UNCERTIFIED' or audit.get('bet_authority') != 0:
        raise PreflightBlocked('CERTIFICATION_OR_AUTHORITY_INVARIANT_FAILED')
    log_func('[NCAAF-V55-EARLY-GATE] status=PASS_RESEARCH_ONLY inputs=7 '
             'certification=HOLD_UNCERTIFIED bet_authority=0 before_model_training=TRUE')
    return audit
