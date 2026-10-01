"""NFL V1.9.3 extension of the V1.9.2 prospective research ledger.

Adds nullable challenger/attribution fields to the same isolated sharp_research
ledger.  V1.9.2 tables and immutable prediction rows remain valid and untouched.
"""
from __future__ import annotations

from google.cloud import bigquery as b

import nfl_prospective_ledger_v1 as base

SOURCE_TAG = "nfl-prospective-ledger-v2-v1.9.3-enhanced-shadow-20261001"
LEDGER_VERSION = "nfl-research-v1.9.3-prospective-shadow-20261001"
PRODUCTION_AUTHORITY = 0


def _prediction_additions():
    S=b.SchemaField
    return [
        S("research_clock_id","STRING"),
        S("research_engine_version","STRING"),
        S("challenger_registry_sha256","STRING"),
        S("core_challenger_name","STRING"),
        S("core_challenger_margin_pred","FLOAT64"),
        S("core_challenger_total_pred","FLOAT64"),
        S("structured_spread_correction","FLOAT64"),
        S("structured_total_correction","FLOAT64"),
        S("structured_corrected_margin_pred","FLOAT64"),
        S("structured_corrected_total_pred","FLOAT64"),
        S("residual_family_contributions_json","STRING"),
        S("disagreement_driver_json","STRING"),
        S("miner_v2_active","STRING"),
        S("miner_v2_rules_json","STRING"),
        S("prospective_eligible","BOOL"),
    ]


def _result_additions():
    S=b.SchemaField
    return [
        S("core_challenger_error","FLOAT64"),
        S("structured_corrected_error","FLOAT64"),
        S("research_clock_id","STRING"),
        S("challenger_registry_sha256","STRING"),
    ]


def _add_nullable(client, table_id, fields):
    table=client.get_table(table_id)
    existing={f.name for f in table.schema}
    missing=[f for f in fields if f.name not in existing]
    if missing:
        table.schema=list(table.schema)+missing
        client.update_table(table,["schema"])
    return [f.name for f in missing]


def ensure_tables(client=None):
    c=client or b.Client(project=base.PROJECT_ID)
    ready=base.ensure_tables(c)
    p=_add_nullable(c,base.PRED_TABLE,_prediction_additions())
    r=_add_nullable(c,base.RESULT_TABLE,_result_additions())
    return {
        **ready,
        "status":"READY_V1_9_3",
        "ledger_version_v2":LEDGER_VERSION,
        "prediction_fields_added_v1_9_3":p,
        "result_fields_added_v1_9_3":r,
        "production_authority":0,
    }


def ledger_health_check(client=None, *, service_identity_hint=None):
    c=client or b.Client(project=base.PROJECT_ID)
    ready=ensure_tables(c)
    base_health=base.ledger_health_check(c,service_identity_hint=service_identity_hint)
    pnames={f.name for f in c.get_table(base.PRED_TABLE).schema}
    rnames={f.name for f in c.get_table(base.RESULT_TABLE).schema}
    missing_p=[f.name for f in _prediction_additions() if f.name not in pnames]
    missing_r=[f.name for f in _result_additions() if f.name not in rnames]
    if missing_p or missing_r:
        raise RuntimeError(f"NFL_V1_9_3_LEDGER_SCHEMA_MISSING predictions={missing_p} results={missing_r}")
    return {
        **base_health, **ready,
        "V1_9_3_CHALLENGER_SCHEMA_PASS":True,
        "status":"HEALTHY_V1_9_3",
        "production_authority":0,
    }
