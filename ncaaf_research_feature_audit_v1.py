"""NCAAF research feature-lineage and candidate-suppression audit (read-only).

Runs on the already-built historical CORE and enriched Miner research frames.
No feature or prediction is modified. 2026 rows must be removed upstream;
this module also refuses to inspect them for candidate readiness.
"""
from __future__ import annotations
import json
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

DISCOVERY_END = 2023
CONFIRMATION_YEARS = (2024, 2025)
FEATURE_PREFIXES = ("CFBD_", "Deep_", "RQ_", "CTX_", "PTIDX_", "PT_", "META_PT_", "EXPERT_", "Pathi_", "BigAl_", "A_", "B_", "Diff_")
BLOCKED_NAME_TOKENS = ("actual_", "final_", "result_", "outcome", "postgame", "post_game", "__actual", "_score_final", "__oppactual", "winner", "target", "label")
DEFAULT_MIN_DISCOVERY = 100


def _family(name: str) -> str:
    s = str(name)
    if s.startswith("CFBD_"): return "CFBD"
    if s.startswith(("PTIDX_", "PT_", "META_PT_")): return "PREDICTION_TRACKER"
    if s.startswith("Pathi_"): return "PATHI"
    if s.startswith("BigAl_"): return "BIG_AL"
    if s.startswith(("EXPERT_",)): return "EXPERT"
    if s.startswith(("Deep_", "CTX_", "RQ_")): return "DEEP_CONTEXT"
    if s.startswith(("A_", "B_", "Diff_")): return "CORE_STYLE"
    return "OTHER"


def _safe_numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    s = frame[name]
    if isinstance(s, pd.DataFrame):  # duplicated source column name
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    vals = pd.to_numeric(s, errors="coerce")
    return vals.where(np.isfinite(vals), np.nan)


def _rates(frame: pd.DataFrame, sy: np.ndarray, name: str) -> tuple[dict, dict]:
    s = _safe_numeric(frame, name)
    v = s.to_numpy(dtype=float)
    actual = np.isfinite(v)
    zero = actual & (v == 0)
    d = np.isfinite(sy) & (sy <= DISCOVERY_END)
    yrs = {str(y): {"rows": int((sy == y).sum()),
                    "non_null": int((actual & (sy == y)).sum()),
                    "numeric_pct": round(float((actual & (sy == y)).sum()/max((sy == y).sum(), 1)), 4)}
           for y in sorted({int(y) for y in sy if np.isfinite(y) and y <= 2025})}
    con = {str(y): int((actual & (sy == y)).sum()) for y in CONFIRMATION_YEARS}
    disc_values = v[d & actual]
    support = {
        "non_null_total": int(actual.sum()), "non_null_discovery": int((actual & d).sum()),
        "non_null_confirmation": con,
        "numeric_pct_total": round(float(actual.mean()),4) if len(actual) else 0.0,
        "zero_pct_of_numeric": round(float(zero.sum()/max(actual.sum(),1)),4),
        "distinct_discovery": int(len(np.unique(disc_values))) if disc_values.size else 0,
        "ever_nonzero": bool(np.any(actual & (v != 0))),
        "discovery_training_floor_100_and_unique4": bool((actual & d).sum() >= DEFAULT_MIN_DISCOVERY and len(np.unique(disc_values)) >= 4),
        "confirmation_both_years": bool(all(con[str(y)] > 0 for y in CONFIRMATION_YEARS)),
    }
    return support, yrs


def audit_research_feature_lineage(core_games: pd.DataFrame, miner_games: pd.DataFrame,
                                   seasons, candidate_feature_cols, *, log_func=print,
                                   details_limit: int=3000) -> dict[str, Any]:
    """Compare same-row historic CORE/STAT input, Miner-enriched input and candidate registry.

    Does not declare a feature safe based on its name; point-in-time validation
    remains separate. This audit only discovers plausible exclusions or coverage gaps.
    """
    sy = np.asarray(seasons, dtype=float)
    if len(core_games) != len(miner_games) or len(core_games) != len(sy):
        raise ValueError("research frame alignment mismatch")
    if not np.all(np.isfinite(sy) & (sy <= 2025)):
        raise ValueError("feature research audit requires <=2025 historical-only aligned rows")
    declared = set(map(str, candidate_feature_cols))
    core_cols = set(map(str, core_games.columns))
    miner_cols = set(map(str, miner_games.columns))
    duplicates = {"core": sorted(set(core_games.columns[core_games.columns.duplicated()].tolist())),
                  "miner": sorted(set(miner_games.columns[miner_games.columns.duplicated()].tolist()))}
    for c in duplicates.values():
        if c: log_func("[NCAAF-RESEARCH-FEATURE-ALERT] DUPLICATE_COLUMNS:" + ','.join(map(str,c[:20])))
    wanted = sorted((core_cols | miner_cols | declared))
    interesting = [c for c in wanted if c in declared or c.startswith(FEATURE_PREFIXES)]
    rows = []; type_counts = Counter(); grouped = Counter()
    for name in interesting:
        # Excluded from hypothetical modeling; still expose the name but do not suggest it as a candidate.
        leak_risk = any(t in name.lower() for t in BLOCKED_NAME_TOKENS)
        cin = name in core_cols
        min_ = name in miner_cols
        base, base_years = _rates(core_games, sy, name) if cin else ({}, {})
        enriched, enriched_years = _rates(miner_games, sy, name) if min_ else ({}, {})
        in_registry = name in declared
        core_nonnull = int(base.get("non_null_total",0))
        miner_nonnull = int(enriched.get("non_null_total",0))
        if min_ and miner_nonnull>0 and not cin:
            state="MINER_ONLY_NOT_IN_STAT_FRAME"
        elif min_ and miner_nonnull>0 and not in_registry:
            state="AVAILABLE_NOT_IN_STAT_CANDIDATE_REGISTRY"
        elif min_ and miner_nonnull > core_nonnull + max(20, int(0.05*len(sy))):
            state="ENRICHED_COVERAGE_GAIN_NOT_REFLECTED_IN_STAT"
        elif in_registry and core_nonnull==0:
            state="DECLARED_STAT_CANDIDATE_UNPOPULATED"
        elif in_registry and not base.get("discovery_training_floor_100_and_unique4",False):
            state="STAT_CANDIDATE_FAILS_TRAINING_COVERAGE_GATE"
        elif in_registry and not base.get("confirmation_both_years",False):
            state="STAT_CANDIDATE_MISSING_CONFIRMATION_SEASON"
        elif min_ and miner_nonnull==0:
            state="MINER_COLUMN_PRESENT_BUT_UNPOPULATED"
        else:
            state="OBSERVED_NO_OBVIOUS_COVERAGE_GAP"
        if leak_risk:
            candidate_assessment="EXCLUDE_POTENTIAL_POSTGAME_FIELD"
        elif state in ("MINER_ONLY_NOT_IN_STAT_FRAME", "AVAILABLE_NOT_IN_STAT_CANDIDATE_REGISTRY", "ENRICHED_COVERAGE_GAIN_NOT_REFLECTED_IN_STAT"):
            candidate_assessment="REVIEW_LANE_EXCLUSION_AND_POINT_IN_TIME_PROVENANCE"
        else:
            candidate_assessment="NOT_AUTOMATICALLY_ELIGIBLE"
        grouped[(state,_family(name))] += 1; type_counts[state]+=1
        rows.append({"field":name,"family":_family(name),"status":state,
                     "candidate_assessment":candidate_assessment,
                     "in_stat_registry":in_registry,"core_present":cin,"miner_present":min_,
                     "core":base,"miner":enriched,
                     "per_year_core":base_years,"per_year_miner":enriched_years})
    priority = {"DECLARED_STAT_CANDIDATE_UNPOPULATED":0,
                "STAT_CANDIDATE_FAILS_TRAINING_COVERAGE_GATE":1,
                "STAT_CANDIDATE_MISSING_CONFIRMATION_SEASON":2,
                "MINER_ONLY_NOT_IN_STAT_FRAME":3,
                "ENRICHED_COVERAGE_GAIN_NOT_REFLECTED_IN_STAT":4,
                "AVAILABLE_NOT_IN_STAT_CANDIDATE_REGISTRY":5,
                "MINER_COLUMN_PRESENT_BUT_UNPOPULATED":6}
    rows.sort(key=lambda x:(priority.get(x['status'],20), x['candidate_assessment'] == 'EXCLUDE_POTENTIAL_POSTGAME_FIELD', x['field']))
    serious = [x for x in rows if x["status"] in priority and x["candidate_assessment"] != "EXCLUDE_POTENTIAL_POSTGAME_FIELD"]
    summary = {"version":"NCAAF-RESEARCH-FEATURE-LINEAGE-V1",
        "status":"WARNING_FEATURE_LANE_GAPS" if serious or any(duplicates.values()) else "PASS_NO_OBVIOUS_COVERAGE_GAP",
        "policy":"DIAGNOSTIC_ONLY_NO_FEATURE_PROMOTION_NO_MODEL_OR_BET_AUTHORITY",
        "historical_rows":int(len(sy)),"discovery_rows":int((sy <= DISCOVERY_END).sum()),
        "confirmation_rows":{str(y):int((sy==y).sum()) for y in CONFIRMATION_YEARS},
        "core_columns":len(core_cols),"miner_columns":len(miner_cols),"stat_candidate_registry_size":len(declared),
        "declared_missing_from_core":sorted(declared - core_cols),
        "new_fields_after_enrichment":len(miner_cols-core_cols),
        "duplicate_field_names":duplicates,"status_counts":dict(type_counts),
        "family_status_counts":{s+"::"+f:n for (s,f),n in sorted(grouped.items())},
        "flagged_fields":len(serious),"candidate_research_changes":0,"2026_outcomes_used":False}
    log_func("[NCAAF-RESEARCH-FEATURE-LINEAGE] "+json.dumps(summary,sort_keys=True,default=str))
    for r in serious[:40]:
        log_func("[NCAAF-RESEARCH-FEATURE-MISSED-CANDIDATE] "+json.dumps({
            "field":r['field'],"status":r['status'],"family":r['family'],
            "stat_registry":r['in_stat_registry'],
            "core_non_null":r['core'].get('non_null_total',0),
            "miner_non_null":r['miner'].get('non_null_total',0),
            "miner_discovery":r['miner'].get('non_null_discovery',0),
            "miner_confirm":r['miner'].get('non_null_confirmation',{}),
            "point_in_time_certified":False,"action":"REVIEW_NOT_PROMOTE"},sort_keys=True))
    return {"summary":summary,"fields":rows[:details_limit],"truncated_fields":max(0,len(rows)-details_limit)}


def audit_stat_family_caps(candidate_cols, family_tokens, *, cap=32, sparse_candidates=None, core_frame=None, log_func=print):
    """Reveal silently excluded STAT features, including name-sort family caps.

    Mirrors current STAT family token and exclusions but never adjusts selection.
    """
    members={}
    for fam, toks in family_tokens.items():
        m=sorted(set(c for c in candidate_cols
            if not any(t in str(c).lower() for t in BLOCKED_NAME_TOKENS)
            and any(t in str(c).lower() for t in toks)))
        members[str(fam)]={"matched":len(m),"tested_under_current_32_cap":len(m[:cap]),
                            "excluded_due_to_name_sorted_cap":m[cap:]}
    sparse={}
    if sparse_candidates is not None and core_frame is not None:
        for market,features in sparse_candidates.items():
            sparse[str(market)]={"predeclared":list(features),
                "missing_from_core_frame":sorted(set(features)-set(core_frame.columns))}
    out={"status":"RESEARCH_DIAGNOSTIC_ONLY","family_caps":members,
         "total_feature_family_exclusions":sum(len(x['excluded_due_to_name_sorted_cap']) for x in members.values()),
         "sparse_gate":sparse,"selection_changed":False,"production_authority":0}
    log_func('[NCAAF-RESEARCH-STAT-FAMILY-CAP-AUDIT] '+json.dumps(out,sort_keys=True,default=str))
    return out
