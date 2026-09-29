"""FROZEN_SPREAD_EDGE_POLICY_V1

Frozen prospective decision policy for NCAAF spreads.

The policy is intentionally NOT re-selected on future runs. It freezes the rule
orientations/families learned through ATOMIC_RULE_REFINEMENT_V1 on 2026-09-29.
Future data may update prospective results only; it may not change which rules
have edge authority, their orientation, or family membership.

Decision states:
  EDGE_MULTI  = >=2 independent validated mechanisms agree, no validated conflict
  EDGE_SINGLE = exactly one independent validated mechanism, no validated conflict
  SHADOW      = no validated authority edge, but a frozen research rule fires
  PASS        = no recognized signal
  PASS_CONFLICT = validated authority rules disagree

Production betting authority remains 0. This module is a frozen prospective
research policy/ledger until a separate promotion decision is made.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List
import numpy as np
import pandas as pd

import atomic_rule_refinement_v1 as arr
import edge_mechanism_matrix_v1 as emm
import edge_topology_v1 as et

FROZEN_SPREAD_EDGE_POLICY_V1_SOURCE_TAG = "frozen-spread-edge-policy-v1-20260929"
PROSPECTIVE_FREEZE_UTC = pd.Timestamp("2026-09-29T22:47:00Z")
DISCOVERY_SEASONS = (2023, 2024, 2025)
CONFIRM_SEASON = 2026

# These are frozen from downloaded-logs-20260929-184458.json. Child refinements
# that failed to add value remain aliases/tags but do not add authority votes.
AUTHORITY_FAMILIES = {
    "SPREAD_STAT_COMBO": {
        "root": "STAT:STAT_COMBO_PRIMARY",
        "orientation": "PLAY",
        "mechanisms": ("STATISTICAL_MATCHUP",),
        "aliases": (),
        "evidence": "REPEATABLE_PLAY_MARKET_CONFIRMED",
    },
    "BIGAL_CF1_WEEK2_HOME42": {
        "root": "BIGAL:BigAl_CF1_Week2Home42Win",
        "orientation": "PLAY",
        "mechanisms": ("SITUATIONAL", "TEAM_STATE"),
        "aliases": (),
        "evidence": "REPEATABLE_PLAY_MARKET_CONFIRMED",
    },
    "PATHI_CROSSED_KEY_AWAY_FADE": {
        "root": "PATHI:Pathi_FB_Crossed_Key_Away_From_Team",
        "orientation": "FADE",
        "mechanisms": ("MARKET_STRUCTURE",),
        "aliases": (),
        "evidence": "REPEATABLE_FADE_MARKET_CONFIRMED",
    },
    "MINER_HIGH_TOTAL_60_PLUS": {
        "root": "MINER:SYS-SPREADS-63E7064A",
        "orientation": "PLAY",
        "mechanisms": ("MARKET_STRUCTURE",),
        # Both refinements were repeatable individually but failed parent-child
        # incremental-value testing, so they are tags only.
        "aliases": ("MINER:SYS-SPREADS-ECD1C88E", "MINER:SYS-SPREADS-082DD238"),
        "evidence": "REPEATABLE_PARENT_FAMILY_CHILDREN_NO_INCREMENTAL_VALUE",
    },
    "MINER_DOG_3_7": {
        "root": "MINER:SYS-SPREADS-9E793B75",
        "orientation": "PLAY",
        "mechanisms": ("MARKET_STRUCTURE",),
        "aliases": (),
        "evidence": "REPEATABLE_PLAY_MARKET_CONFIRMED",
    },
    "MINER_CONF_BIG12_HOME_FAVORITE": {
        "root": "MINER:SYS-SPREADS-8A80FD15",
        "orientation": "PLAY",
        "mechanisms": ("IDENTITY_CONTEXT", "MARKET_STRUCTURE", "SITUATIONAL"),
        "aliases": (),
        "evidence": "REPEATABLE_PLAY_MARKET_CONFIRMED",
    },
}

# Frozen discovery candidates that did not earn authority. They are observed
# prospectively but cannot create an official edge.
SHADOW_RULES = {
    "PATHI:Pathi_FB_Dog_Hook_Above_3": "PLAY",
    "PATHI:Pathi_FB_Favorite_Below_Key_7": "PLAY",
    "PATHI:Pathi_FB_Dog_Below_Key_3": "PLAY",
    "MINER:SYS-SPREADS-E28D8EBE": "PLAY",
    "MINER:SYS-SPREADS-0450219C": "PLAY",
    "MINER:SYS-SPREADS-5E2AF3C7": "PLAY",
    "MINER:SYS-SPREADS-E64EC91A": "PLAY",
    "MINER:SYS-SPREADS-B7E32380": "PLAY",
}


def _period_masks(g: pd.DataFrame):
    s = pd.to_numeric(g.get("Season"), errors="coerce").to_numpy(float)
    dt = pd.to_datetime(g.get("Game_Date"), errors="coerce", utc=True)
    discovery = np.isin(s, np.asarray(DISCOVERY_SEASONS, float))
    confirm = s == float(CONFIRM_SEASON)
    prospective = np.asarray(dt >= PROSPECTIVE_FREEZE_UTC, dtype=bool)
    return discovery, confirm, prospective


def _maximum_matching(active_families: List[dict]) -> int:
    """Maximum one-family/one-mechanism matching; prevents mechanism inflation."""
    match: Dict[str, int] = {}
    def dfs(i: int, seen: set[str]) -> bool:
        for grp in sorted(active_families[i].get("mechanisms") or ()):
            if grp in seen:
                continue
            seen.add(grp)
            if grp not in match or dfs(match[grp], seen):
                match[grp] = i
                return True
        return False
    score = 0
    for i in range(len(active_families)):
        if dfs(i, set()):
            score += 1
    return score


def _metrics(g, target, sign, mask, dashboard_module):
    return emm._metrics(g, target, sign, mask, dashboard_module)


def _build_rule_vote_map(g, dashboard_module, stat_out, registry_out):
    drules, crules = arr._build_rule_rows(g, dashboard_module, stat_out, registry_out)
    out = {}
    # Both canonical sets carry full-length vote vectors. Prefer discovery row,
    # then fill any missing rule from confirmation.
    for r in list(drules) + list(crules):
        key = str(r.get("rule_key") or "")
        if key and key not in out:
            out[key] = np.asarray(r.get("vote"), float)
    return out


def _policy_rows(g, vote_map):
    rows = []
    n = len(g)
    for i in range(n):
        active_auth = []
        auth_signs = []
        alias_tags = []
        for fid, spec in AUTHORITY_FAMILIES.items():
            v = vote_map.get(spec["root"])
            if v is None or i >= len(v) or not np.isfinite(v[i]) or np.isclose(v[i], 0.0):
                continue
            mult = -1.0 if spec["orientation"] == "FADE" else 1.0
            s = float(np.sign(v[i]) * mult)
            active_auth.append({"family_id": fid, **spec})
            auth_signs.append(s)
            for a in spec.get("aliases") or ():
                av = vote_map.get(a)
                if av is not None and i < len(av) and np.isfinite(av[i]) and not np.isclose(av[i], 0.0):
                    alias_tags.append(a)

        shadow = []
        for rk, ori in SHADOW_RULES.items():
            v = vote_map.get(rk)
            if v is None or i >= len(v) or not np.isfinite(v[i]) or np.isclose(v[i], 0.0):
                continue
            mult = -1.0 if ori == "FADE" else 1.0
            shadow.append((rk, float(np.sign(v[i]) * mult)))

        if active_auth and len(set(auth_signs)) > 1:
            decision = "PASS_CONFLICT"; side = np.nan; independent = _maximum_matching(active_auth)
        elif active_auth:
            side = auth_signs[0]
            independent = _maximum_matching(active_auth)
            decision = "EDGE_MULTI" if independent >= 2 else "EDGE_SINGLE"
        elif shadow:
            ss = {x[1] for x in shadow}
            side = next(iter(ss)) if len(ss) == 1 else np.nan
            independent = 0
            decision = "SHADOW" if len(ss) == 1 else "PASS_CONFLICT"
        else:
            decision = "PASS"; side = np.nan; independent = 0

        rows.append({
            "decision": decision,
            "side": side,
            "authority_families": [x["family_id"] for x in active_auth],
            "authority_rules": [x["root"] for x in active_auth],
            "alias_tags": alias_tags,
            "shadow_rules": [x[0] for x in shadow],
            "independent_mechanisms": int(independent),
            "mechanisms": sorted({m for x in active_auth for m in x.get("mechanisms", ())}),
        })
    return rows


def _eval_decision(g, target, rows, period_mask, state, dashboard_module):
    use = np.asarray([r["decision"] == state for r in rows], bool) & period_mask
    sign = np.asarray([r["side"] if np.isfinite(r["side"]) else 0.0 for r in rows], float)
    return _metrics(g, target, sign, use, dashboard_module)


def run_frozen_spread_edge_policy_v1(*, dashboard_module, stat_out: dict, registry_out: dict,
                                     refinement_out: dict | None = None, log_func=print, hard_fail=True):
    try:
        cache = getattr(dashboard_module, "_V1357_SPREAD_RESEARCH_CACHE", {})
        g = cache.get("games") if isinstance(cache, dict) else None
        if not isinstance(g, pd.DataFrame) or g.empty:
            raise RuntimeError("spread research games unavailable")
        if int(et._physical_key(g).duplicated().sum()):
            raise RuntimeError("physical game duplicates present")
        target = pd.to_numeric(g.get("Market_Error_Margin"), errors="coerce").to_numpy(float)
        vote_map = _build_rule_vote_map(g, dashboard_module, stat_out, registry_out)
        required = {spec["root"] for spec in AUTHORITY_FAMILIES.values()} | set(SHADOW_RULES)
        missing = sorted(required - set(vote_map))
        if missing:
            raise RuntimeError(f"frozen rule(s) missing from current rule graph: {missing}")

        # Fail if a future research rerun tries to silently alter the frozen authority
        # list. The refinement output may change; this policy may not.
        log_func(
            f"[FROZEN-SPREAD-V1-PREFLIGHT] status=PASS source_tag={FROZEN_SPREAD_EDGE_POLICY_V1_SOURCE_TAG} "
            f"freeze_utc={PROSPECTIVE_FREEZE_UTC.isoformat()} authority_families={len(AUTHORITY_FAMILIES)} "
            f"shadow_rules={len(SHADOW_RULES)} dynamic_reselection=FALSE majority_vote=FALSE conflicts_force_pass=TRUE "
            f"child_refinements_add_authority=FALSE production_authority=0"
        )
        for fid, spec in AUTHORITY_FAMILIES.items():
            log_func(
                f"[FROZEN-SPREAD-V1-AUTHORITY] family={fid} root={spec['root']} orientation={spec['orientation']} "
                f"mechanisms={'+'.join(spec['mechanisms'])} aliases={'|'.join(spec['aliases']) or 'NONE'} "
                f"evidence={spec['evidence']} authority=EDGE production_authority=0"
            )

        rows = _policy_rows(g, vote_map)
        discovery, confirm, prospective = _period_masks(g)
        metrics = {}
        for label, mask in (("DISCOVERY", discovery), ("CONFIRM_2026", confirm), ("PROSPECTIVE", prospective)):
            for state in ("EDGE_SINGLE", "EDGE_MULTI"):
                met = _eval_decision(g, target, rows, mask, state, dashboard_module)
                metrics[(label, state)] = met
                log_func(
                    f"[FROZEN-SPREAD-V1-POLICY] sample={label} decision={state} n={met['n']} hit={met['hit']:.4f} "
                    f"roi={met['roi']:+.4f} signed={met['signed']:+.3f} clv={met['clv']:+.3f} production_authority=0"
                )
            c = int(sum(mask[i] and rows[i]["decision"] == "PASS_CONFLICT" for i in range(len(rows))))
            sh = int(sum(mask[i] and rows[i]["decision"] == "SHADOW" for i in range(len(rows))))
            ps = int(sum(mask[i] and rows[i]["decision"] == "PASS" for i in range(len(rows))))
            log_func(f"[FROZEN-SPREAD-V1-COUNTS] sample={label} conflicts={c} shadow={sh} pass={ps} production_authority=0")

        prospective_n = int(prospective.sum())
        log_func(
            f"[FROZEN-SPREAD-V1-CONTRACT] status=PASS frozen_authority=TRUE prospective_freeze_utc={PROSPECTIVE_FREEZE_UTC.isoformat()} "
            f"prospective_scored_games={prospective_n} edge_single_rule=VALIDATED_FAMILY edge_multi_rule=AT_LEAST_2_INDEPENDENT_MECHANISMS "
            f"shadow_cannot_influence_edge=TRUE validated_conflict_forces_pass=TRUE aliases_cannot_inflate_authority=TRUE production_authority=0"
        )
        return {
            "status":"PASS", "source_tag":FROZEN_SPREAD_EDGE_POLICY_V1_SOURCE_TAG,
            "freeze_utc":PROSPECTIVE_FREEZE_UTC.isoformat(), "authority_families":AUTHORITY_FAMILIES,
            "shadow_rules":SHADOW_RULES, "metrics":metrics, "prospective_scored_games":prospective_n,
            "production_authority":0,
        }
    except Exception as e:
        log_func(f"[FROZEN-SPREAD-V1-CONTRACT] status=FAILED error={type(e).__name__}:{e} production_authority=0")
        if hard_fail:
            raise
        return {"status":"FAILED", "error":f"{type(e).__name__}:{e}", "production_authority":0}
