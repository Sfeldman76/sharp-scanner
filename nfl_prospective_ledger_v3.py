"""NFL V1.9.4 prospective ledger extension for edge-gate shadow research.

Adds nullable edge-probability and dual-scorecard fields to the isolated
sharp_research ledger. Existing V1.9.2/V1.9.3 prediction rows remain untouched.
"""
from __future__ import annotations

from google.cloud import bigquery as b

import nfl_prospective_ledger_v2 as base2
import nfl_prospective_ledger_v1 as base1

SOURCE_TAG = "nfl-prospective-ledger-v3-v1.9.4-edge-gate-shadow-20261001"
LEDGER_VERSION = "nfl-research-v1.9.4-edge-gate-shadow-20261001"
PRODUCTION_AUTHORITY = 0


def _prediction_additions():
    S=b.SchemaField
    return [
        S("edge_gate_version","STRING"),
        S("edge_gate_registry_sha256","STRING"),
        S("core_edge_probability","FLOAT64"),
        S("consensus_edge_probability","FLOAT64"),
        S("edge_candidate_direction","STRING"),
        S("edge_probability_band","STRING"),
        S("edge_gate_features_json","STRING"),
        S("edge_simple_gates_json","STRING"),
        S("fair_line_scorecard_version","STRING"),
        S("edge_scorecard_version","STRING"),
    ]


def _result_additions():
    S=b.SchemaField
    return [
        S("edge_candidate_correct","BOOL"),
        S("core_edge_brier","FLOAT64"),
        S("core_edge_log_loss","FLOAT64"),
        S("consensus_edge_brier","FLOAT64"),
        S("consensus_edge_log_loss","FLOAT64"),
        S("edge_gate_registry_sha256","STRING"),
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
    c=client or b.Client(project=base1.PROJECT_ID)
    ready=base2.ensure_tables(c)
    p=_add_nullable(c,base1.PRED_TABLE,_prediction_additions())
    r=_add_nullable(c,base1.RESULT_TABLE,_result_additions())
    return {
        **ready,
        "status":"READY_V1_9_4",
        "ledger_version_v3":LEDGER_VERSION,
        "prediction_fields_added_v1_9_4":p,
        "result_fields_added_v1_9_4":r,
        "production_authority":0,
    }


def ledger_health_check(client=None, *, service_identity_hint=None):
    c=client or b.Client(project=base1.PROJECT_ID)
    ready=ensure_tables(c)
    health=base2.ledger_health_check(c,service_identity_hint=service_identity_hint)
    pnames={f.name for f in c.get_table(base1.PRED_TABLE).schema}
    rnames={f.name for f in c.get_table(base1.RESULT_TABLE).schema}
    missing_p=[f.name for f in _prediction_additions() if f.name not in pnames]
    missing_r=[f.name for f in _result_additions() if f.name not in rnames]
    if missing_p or missing_r:
        raise RuntimeError(f"NFL_V1_9_4_LEDGER_SCHEMA_MISSING predictions={missing_p} results={missing_r}")
    return {
        **health, **ready,
        "V1_9_4_EDGE_GATE_SCHEMA_PASS":True,
        "status":"HEALTHY_V1_9_4",
        "production_authority":0,
    }
