"""ATOMIC_RULE_REFINEMENT_V1 — play/fade + lineage + independence research.

Research-only; zero production authority.

This stage deliberately does NOT create another prediction model. It refines the
existing atomic edge library by answering four questions:

1) Is each named rule useful in its written PLAY direction, useful as a FADE, or
   not repeatable in either direction?
2) When a child rule adds conditions to a parent rule, do those extra conditions
   add out-of-sample value or just fragment the same underlying edge?
3) Which rules are related enough that they should count as one evidence family
   rather than multiple confirmations?
4) After related rules are collapsed, does agreement among genuinely independent
   mechanisms improve edge quality?

Discovery (2023-2025) chooses orientation and candidate status. 2026 is used only
as confirmation. CLV is tracked as supporting evidence and never as a veto.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Any, Tuple, Iterable
import hashlib
import math
import re

import numpy as np
import pandas as pd

import atomic_edge_graph_v1 as aeg
import edge_mechanism_matrix_v1 as emm
import edge_topology_v1 as et

ATOMIC_RULE_REFINEMENT_V1_SOURCE_TAG = "atomic-rule-refinement-v1-play-fade-lineage-independence"
DISCOVERY_SEASONS = (2023, 2024, 2025)
CONFIRM_SEASON = 2026
BREAK_EVEN = 110.0 / 210.0

# Selection thresholds are intentionally conservative. Orientation is always
# frozen from discovery before 2026 is examined.
MIN_ORIENTATION_DISCOVERY_N = 20
MIN_ORIENTATION_CONFIRM_N = 5
MIN_LINEAGE_DISCOVERY_N = 8
MIN_LINEAGE_CONFIRM_N = 3
MIN_STACK_DISCOVERY_N = 15
MIN_STACK_CONFIRM_N = 5
MIN_LOG_N = 3

# Similarity is only used to prevent double counting, never to promote an edge.
RELATED_JACCARD_DISCOVERY = 0.80
RELATED_JACCARD_CONFIRM = 0.65
MIN_CORR_UNION_DISCOVERY = 20
MIN_CORR_UNION_CONFIRM = 5


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _metrics(g, target, sign, mask, dashboard_module):
    return emm._metrics(g, target, sign, mask, dashboard_module)


def _robustness(g, target, sign, mask, dashboard_module):
    return emm._robustness(g, target, sign, mask, dashboard_module, discovery_only=True)


def _clv_state(m):
    return emm._clv_class(m)


def _active(v):
    a = np.asarray(v, float)
    return np.isfinite(a) & ~np.isclose(a, 0.0)


def _normalize_atom(x: Any) -> str:
    s = str(x or "").strip().upper()
    s = re.sub(r"\s+", "_", s)
    return s


def _condition_atoms(rule: dict, dashboard_module) -> frozenset[str]:
    """Return only explicit, auditable condition atoms.

    Miner supplies exact condition atoms directly. Big Al / Pathi use the
    canonical expected-system registry only for explicit parent metadata and a
    human-readable description; we do not pretend prose parsing is equivalent to
    formal Boolean atoms.
    """
    cond = rule.get("conditions") or []
    if isinstance(cond, str):
        cond = [x for x in re.split(r"\s+AND\s+|\s*&\s*", cond, flags=re.I) if x]
    atoms = frozenset(_normalize_atom(x) for x in cond if str(x).strip())
    return atoms


def _registry_meta(rule: dict, dashboard_module) -> dict:
    reg = getattr(dashboard_module, "V132321_EXPECTED_NCAAF_SYSTEM_REGISTRY", {}) or {}
    rid = str(rule.get("id") or "")
    rec = reg.get(rid) if isinstance(reg, dict) else None
    return dict(rec) if isinstance(rec, dict) else {}


def _build_rule_rows(g, dashboard_module, stat_out, registry_out):
    rd, _ = emm._mechanism_rows(g, dashboard_module, stat_out, registry_out, "DISCOVERY")
    rc, _ = emm._mechanism_rows(g, dashboard_module, stat_out, registry_out, "CONFIRM")
    dr = aeg._canonical_rule_rows({"DISCOVERY": rd}, "DISCOVERY")
    cr = aeg._canonical_rule_rows({"CONFIRM": rc}, "CONFIRM")
    cby = {r["rule_key"]: r for r in cr}
    for r in dr:
        meta = _registry_meta(r, dashboard_module)
        r["condition_atoms"] = _condition_atoms(r, dashboard_module)
        r["explicit_parent_id"] = str(meta.get("parent") or "")
        r["rule_text"] = str(meta.get("rule") or "")
        r["builder"] = str(meta.get("builder") or "")
        rcurr = cby.get(r["rule_key"])
        if rcurr is not None:
            rcurr["condition_atoms"] = r["condition_atoms"]
            rcurr["explicit_parent_id"] = r["explicit_parent_id"]
            rcurr["rule_text"] = r["rule_text"]
            rcurr["builder"] = r["builder"]
    return dr, cr


def _period_masks(g):
    season = pd.to_numeric(g.get("Season"), errors="coerce").to_numpy(float)
    return (
        np.isin(season, np.asarray(DISCOVERY_SEASONS, float)),
        season == float(CONFIRM_SEASON),
    )


def _orientation_candidate(dm, dr) -> bool:
    n = int(dm.get("n", 0) or 0)
    if n < MIN_ORIENTATION_DISCOVERY_N:
        return False
    hit = _safe_float(dm.get("hit")); signed = _safe_float(dm.get("signed"))
    if not (np.isfinite(hit) and np.isfinite(signed) and hit > BREAK_EVEN and signed > 0):
        return False
    rb = _safe_float(dr.get("remove_best_hit")); lo = _safe_float(dr.get("min_loso_hit"))
    if np.isfinite(rb) and rb <= 0.50:
        return False
    if np.isfinite(lo) and lo <= 0.50:
        return False
    return True


def _orientation_state(chosen: str, dm, dr, cm) -> str:
    if chosen == "NONE":
        return "NO_DISCOVERY_DIRECTION"
    dn = int(dm.get("n", 0) or 0); cn = int(cm.get("n", 0) or 0)
    if dn < MIN_ORIENTATION_DISCOVERY_N:
        return "SMALL_DISCOVERY"
    dgood = _orientation_candidate(dm, dr)
    if not dgood:
        return "DISCOVERY_DIRECTION_NOT_ROBUST"
    if cn < MIN_ORIENTATION_CONFIRM_N:
        return "AWAIT_CONFIRMATION"
    cgood = _safe_float(cm.get("hit")) > BREAK_EVEN and _safe_float(cm.get("signed")) > 0
    if not cgood:
        return f"{chosen}_DISCOVERY_NOT_CONFIRMED"
    cc = _clv_state(cm)
    if cc == "MARKET_CONFIRMED":
        return f"REPEATABLE_{chosen}_MARKET_CONFIRMED"
    if cc == "CONTRARIAN_MARKET_MOVED_AGAINST":
        return f"REPEATABLE_{chosen}_CONTRARIAN"
    return f"REPEATABLE_{chosen}"


def _select_rule_orientations(g, target, drules, crules, dashboard_module, log_func):
    dmask, cmask = _period_masks(g)
    cby = {r["rule_key"]: r for r in crules}
    rows = []
    for r in drules:
        rr = cby.get(r["rule_key"], r)
        dv = np.asarray(r["vote"], float); cv = np.asarray(rr["vote"], float)
        da = dmask & _active(dv); ca = cmask & _active(cv)

        dplay = _metrics(g, target, np.sign(dv), da, dashboard_module)
        dfade = _metrics(g, target, -np.sign(dv), da, dashboard_module)
        rplay = _robustness(g, target, np.sign(dv), da, dashboard_module)
        rfade = _robustness(g, target, -np.sign(dv), da, dashboard_module)

        play_ok = _orientation_candidate(dplay, rplay)
        fade_ok = _orientation_candidate(dfade, rfade)
        if play_ok and not fade_ok:
            chosen, mult, dm, dr = "PLAY", 1.0, dplay, rplay
        elif fade_ok and not play_ok:
            chosen, mult, dm, dr = "FADE", -1.0, dfade, rfade
        elif play_ok and fade_ok:
            # Should be exceedingly rare. Discovery signed error is the tie-breaker;
            # 2026 never participates in the choice.
            if _safe_float(dplay.get("signed")) >= _safe_float(dfade.get("signed")):
                chosen, mult, dm, dr = "PLAY", 1.0, dplay, rplay
            else:
                chosen, mult, dm, dr = "FADE", -1.0, dfade, rfade
        else:
            chosen, mult, dm, dr = "NONE", 0.0, dplay, rplay

        if chosen == "NONE":
            cm = _metrics(g, target, np.sign(cv), ca, dashboard_module)
        else:
            cm = _metrics(g, target, mult * np.sign(cv), ca, dashboard_module)
        state = _orientation_state(chosen, dm, dr, cm)
        row = {
            "rule_key": r["rule_key"], "source": r["source"], "rule_id": r.get("id"),
            "chosen": chosen, "multiplier": mult, "state": state,
            "discovery_play": dplay, "discovery_fade": dfade,
            "discovery": dm, "confirm": cm, "robustness": dr,
            "mechanisms": sorted(r.get("mechanisms") or []),
            "mechanism_groups": sorted(r.get("mechanism_groups") or []),
            "condition_atoms": sorted(r.get("condition_atoms") or []),
            "explicit_parent_id": r.get("explicit_parent_id") or "",
            "rule_text": r.get("rule_text") or "",
            "builder": r.get("builder") or "",
            "vote_discovery": dv, "vote_confirm": cv,
        }
        rows.append(row)
        log_func(
            f"[RULE-REFINE-V1-ORIENTATION] source={r['source']} rule={r.get('id')} chosen={chosen} state={state} "
            f"play_n={dplay['n']} play_hit={dplay['hit']:.4f} play_roi={dplay['roi']:+.4f} play_signed={dplay['signed']:+.3f} play_clv={dplay['clv']:+.3f} "
            f"fade_n={dfade['n']} fade_hit={dfade['hit']:.4f} fade_roi={dfade['roi']:+.4f} fade_signed={dfade['signed']:+.3f} fade_clv={dfade['clv']:+.3f} "
            f"remove_best_hit={dr['remove_best_hit']:.4f} min_loso_hit={dr['min_loso_hit']:.4f} fire_rate_ratio={dr['rate_ratio']:.4f} "
            f"confirm_n={cm['n']} confirm_hit={cm['hit']:.4f} confirm_roi={cm['roi']:+.4f} confirm_signed={cm['signed']:+.3f} confirm_clv={cm['clv']:+.3f} "
            f"confirm_clv_state={_clv_state(cm)} orientation_selected_from_discovery_only=TRUE production_authority=0"
        )
    return rows


def _lineage_edges(orows: List[dict]):
    """Return auditable parent->child edges.

    * Curated Big Al/Pathi children use explicit registry parent metadata.
    * Miner parents are inferred only from strict condition-set containment and
      only among rules with the same source. Immediate parents (largest strict
      subsets) are preferred to avoid redundant transitive edges.
    """
    by_key = {r["rule_key"]: r for r in orows}
    by_source_id = {(r["source"], str(r["rule_id"])): r for r in orows}
    edges = []
    seen = set()

    for child in orows:
        pid = str(child.get("explicit_parent_id") or "")
        if pid:
            parent = by_source_id.get((child["source"], pid))
            if parent:
                k = (parent["rule_key"], child["rule_key"])
                if k not in seen:
                    edges.append({"parent": parent["rule_key"], "child": child["rule_key"], "basis": "EXPLICIT_REGISTRY_PARENT"})
                    seen.add(k)

    miners = [r for r in orows if r["source"] == "MINER" and r.get("condition_atoms")]
    for child in miners:
        ca = set(child["condition_atoms"])
        candidates = []
        for parent in miners:
            if parent["rule_key"] == child["rule_key"]:
                continue
            pa = set(parent["condition_atoms"])
            if pa and pa < ca:
                candidates.append(parent)
        if not candidates:
            continue
        max_atoms = max(len(set(p["condition_atoms"])) for p in candidates)
        for parent in candidates:
            if len(set(parent["condition_atoms"])) != max_atoms:
                continue
            k = (parent["rule_key"], child["rule_key"])
            if k not in seen:
                edges.append({"parent": parent["rule_key"], "child": child["rule_key"], "basis": "MINER_STRICT_CONDITION_SUPERSET"})
                seen.add(k)
    return edges


def _oriented_vote(row, sample: str):
    v = row["vote_discovery"] if sample == "DISCOVERY" else row["vote_confirm"]
    if row.get("chosen") not in {"PLAY", "FADE"}:
        return np.zeros(len(v), float)
    return float(row["multiplier"]) * np.asarray(v, float)


def _lineage_incrementality(g, target, orows, edges, dashboard_module, log_func):
    dmask, cmask = _period_masks(g)
    by = {r["rule_key"]: r for r in orows}
    out = []
    for e in edges:
        p = by[e["parent"]]; c = by[e["child"]]
        if p["chosen"] == "NONE" or c["chosen"] == "NONE":
            state = "NO_DISCOVERY_ORIENTATION"
            log_func(
                f"[RULE-REFINE-V1-LINEAGE] parent={p['rule_key']} child={c['rule_key']} basis={e['basis']} state={state} "
                f"parent_orientation={p['chosen']} child_orientation={c['chosen']} production_authority=0"
            )
            out.append({**e, "state": state})
            continue

        pdv = _oriented_vote(p, "DISCOVERY"); cdv = _oriented_vote(c, "DISCOVERY")
        pcv = _oriented_vote(p, "CONFIRM"); ccv = _oriented_vote(c, "CONFIRM")

        def calc(pm, cm, smask):
            pa = smask & _active(pm); ca = smask & _active(cm)
            both = pa & ca
            same = both & (np.sign(pm) == np.sign(cm))
            conflict = both & (np.sign(pm) == -np.sign(cm))
            parent_only = pa & ~ca
            child_met = _metrics(g, target, np.sign(cm), ca, dashboard_module)
            parent_only_met = _metrics(g, target, np.sign(pm), parent_only, dashboard_module)
            parent_all_met = _metrics(g, target, np.sign(pm), pa, dashboard_module)
            return {
                "child": child_met, "parent_only": parent_only_met, "parent_all": parent_all_met,
                "same_n": int(same.sum()), "conflict_n": int(conflict.sum()),
                "subset_rate": float(ca[pa].sum() / max(1, ca.sum())) if ca.sum() else np.nan,
            }

        d = calc(pdv, cdv, dmask); cfm = calc(pcv, ccv, cmask)
        dh = _safe_float(d["child"].get("hit")) - _safe_float(d["parent_only"].get("hit"))
        ds = _safe_float(d["child"].get("signed")) - _safe_float(d["parent_only"].get("signed"))
        ch = _safe_float(cfm["child"].get("hit")) - _safe_float(cfm["parent_only"].get("hit"))
        cs = _safe_float(cfm["child"].get("signed")) - _safe_float(cfm["parent_only"].get("signed"))

        if d["conflict_n"] > 0 or cfm["conflict_n"] > 0:
            state = "REVERSAL_BRANCH"
        elif d["child"]["n"] < MIN_LINEAGE_DISCOVERY_N or d["parent_only"]["n"] < MIN_LINEAGE_DISCOVERY_N:
            state = "SMALL_LINEAGE_SAMPLE"
        elif np.isfinite(dh) and np.isfinite(ds) and dh > 0 and ds > 0:
            if cfm["child"]["n"] >= MIN_LINEAGE_CONFIRM_N and cfm["parent_only"]["n"] >= MIN_LINEAGE_CONFIRM_N and np.isfinite(ch) and np.isfinite(cs):
                state = "REFINEMENT_ADDS_VALUE_BOTH" if ch > 0 and cs > 0 else "DISCOVERY_REFINEMENT_NOT_CONFIRMED"
            else:
                state = "DISCOVERY_REFINEMENT_AWAIT_CONFIRMATION"
        else:
            state = "NO_INCREMENTAL_VALUE"

        log_func(
            f"[RULE-REFINE-V1-LINEAGE] parent={p['rule_key']} child={c['rule_key']} basis={e['basis']} state={state} "
            f"parent_orientation={p['chosen']} child_orientation={c['chosen']} "
            f"discovery_parent_only_n={d['parent_only']['n']} discovery_parent_only_hit={d['parent_only']['hit']:.4f} discovery_parent_only_signed={d['parent_only']['signed']:+.3f} "
            f"discovery_child_n={d['child']['n']} discovery_child_hit={d['child']['hit']:.4f} discovery_child_signed={d['child']['signed']:+.3f} delta_hit={dh:+.4f} delta_signed={ds:+.3f} "
            f"confirm_parent_only_n={cfm['parent_only']['n']} confirm_parent_only_hit={cfm['parent_only']['hit']:.4f} confirm_parent_only_signed={cfm['parent_only']['signed']:+.3f} "
            f"confirm_child_n={cfm['child']['n']} confirm_child_hit={cfm['child']['hit']:.4f} confirm_child_signed={cfm['child']['signed']:+.3f} confirm_delta_hit={ch:+.4f} confirm_delta_signed={cs:+.3f} "
            f"discovery_conflict_n={d['conflict_n']} confirm_conflict_n={cfm['conflict_n']} production_authority=0"
        )
        out.append({**e, "state": state, "discovery": d, "confirm": cfm, "delta_hit": dh, "delta_signed": ds, "confirm_delta_hit": ch, "confirm_delta_signed": cs})
    return out


def _signed_jaccard(a, b, mask):
    aa = mask & _active(a); bb = mask & _active(b)
    union = aa | bb
    un = int(union.sum())
    if un == 0:
        return np.nan, 0, 0, 0
    same = aa & bb & (np.sign(a) == np.sign(b))
    conflict = aa & bb & (np.sign(a) == -np.sign(b))
    return float(same.sum() / un), un, int(same.sum()), int(conflict.sum())


class _UF:
    def __init__(self, keys): self.p = {k:k for k in keys}
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra


def _evidence_families(g, orows, lineage_rows, log_func):
    dmask, cmask = _period_masks(g)
    candidates = [r for r in orows if r["chosen"] in {"PLAY", "FADE"}]
    uf = _UF([r["rule_key"] for r in candidates])
    reasons = defaultdict(list)

    # Parent/child rules always share a base evidence lineage for independence
    # counting even when the child adds useful incremental value.
    keys = {r["rule_key"] for r in candidates}
    for lr in lineage_rows:
        if lr["parent"] in keys and lr["child"] in keys:
            uf.union(lr["parent"], lr["child"])
            reasons[tuple(sorted((lr["parent"], lr["child"])))].append("LINEAGE")

    for a, b in combinations(candidates, 2):
        if not (set(a["mechanism_groups"]) & set(b["mechanism_groups"])):
            continue
        ad = _oriented_vote(a, "DISCOVERY"); bd = _oriented_vote(b, "DISCOVERY")
        ac = _oriented_vote(a, "CONFIRM"); bc = _oriented_vote(b, "CONFIRM")
        jd, ud, _, cd = _signed_jaccard(ad, bd, dmask)
        jc, uc, _, cc = _signed_jaccard(ac, bc, cmask)
        corr = (
            ud >= MIN_CORR_UNION_DISCOVERY and np.isfinite(jd) and jd >= RELATED_JACCARD_DISCOVERY
            and (uc < MIN_CORR_UNION_CONFIRM or (np.isfinite(jc) and jc >= RELATED_JACCARD_CONFIRM))
            and cd == 0 and cc == 0
        )
        if corr:
            uf.union(a["rule_key"], b["rule_key"])
            reasons[tuple(sorted((a["rule_key"], b["rule_key"])))].append(f"SIGNED_JACCARD_D={jd:.3f}_C={jc:.3f}")

    grouped = defaultdict(list)
    for r in candidates:
        grouped[uf.find(r["rule_key"])].append(r)
    families = []
    rule_to_family = {}
    for idx, (_, members) in enumerate(sorted(grouped.items(), key=lambda kv: sorted(x["rule_key"] for x in kv[1]))):
        member_keys = sorted(x["rule_key"] for x in members)
        fid = f"EF{idx+1:03d}-{hashlib.sha256('|'.join(member_keys).encode()).hexdigest()[:8]}"
        groups = sorted({m for r in members for m in r["mechanism_groups"]})
        for k in member_keys: rule_to_family[k] = fid
        families.append({"family_id": fid, "members": member_keys, "mechanism_groups": groups})
        log_func(
            f"[RULE-REFINE-V1-EVIDENCE-FAMILY] family={fid} members={len(member_keys)} rules={'|'.join(member_keys)} "
            f"mechanism_groups={'+'.join(groups) or 'NONE'} policy=COUNT_AS_ONE_RELATED_EVIDENCE_FAMILY production_authority=0"
        )
    return families, rule_to_family


def _discovery_selected_candidates(orows):
    """Freeze candidate rules from discovery only for stack confirmation."""
    out = []
    for r in orows:
        if r["chosen"] not in {"PLAY", "FADE"}:
            continue
        if not _orientation_candidate(r["discovery"], r["robustness"]):
            continue
        out.append(r)
    return out


def _family_vote(family, by_rule, sample: str):
    votes = []
    for rk in family["members"]:
        r = by_rule.get(rk)
        if not r or r["chosen"] not in {"PLAY", "FADE"}:
            continue
        votes.append(_oriented_vote(r, sample))
    if not votes:
        return None
    a = np.vstack(votes)
    pos = np.any(np.isfinite(a) & (a > 0), axis=0)
    neg = np.any(np.isfinite(a) & (a < 0), axis=0)
    v = np.zeros(a.shape[1], float)
    v[pos & ~neg] = 1.0; v[neg & ~pos] = -1.0; v[pos & neg] = np.nan
    return v


def _maximum_mechanism_matching(active_families):
    match = {}
    def dfs(i, seen):
        for grp in sorted(active_families[i].get("mechanism_groups") or []):
            if grp in seen: continue
            seen.add(grp)
            if grp not in match or dfs(match[grp], seen):
                match[grp] = i; return True
        return False
    score = 0
    for i in range(len(active_families)):
        if dfs(i, set()): score += 1
    return score


def _independent_stack_eval(g, target, orows, families, dashboard_module, log_func):
    dmask, cmask = _period_masks(g)
    selected = _discovery_selected_candidates(orows)
    selected_keys = {r["rule_key"] for r in selected}
    by_rule = {r["rule_key"]: r for r in orows if r["rule_key"] in selected_keys}
    sfamilies = []
    for f in families:
        members = [k for k in f["members"] if k in selected_keys]
        if not members: continue
        sfamilies.append({**f, "members": members})

    results = {}
    for label, smask, sample in (("DISCOVERY", dmask, "DISCOVERY"), ("CONFIRM_2026", cmask, "CONFIRM")):
        fvotes = []
        for f in sfamilies:
            v = _family_vote(f, by_rule, sample)
            if v is not None: fvotes.append((f, v))
        rows = []
        for i in range(len(g)):
            if not smask[i]: continue
            active = []
            signs = []
            for f, v in fvotes:
                x = v[i]
                if np.isfinite(x) and not np.isclose(x, 0.0):
                    active.append(f); signs.append(float(np.sign(x)))
            if not active: continue
            if len(set(signs)) != 1:
                rows.append((i, np.nan, len(active), _maximum_mechanism_matching(active), True, tuple(sorted({m for f in active for m in f["mechanism_groups"]}))))
            else:
                rows.append((i, signs[0], len(active), _maximum_mechanism_matching(active), False, tuple(sorted({m for f in active for m in f["mechanism_groups"]}))))

        # Monotonic stack test: does requiring more independent evidence improve?
        max_ind = max([r[3] for r in rows], default=0)
        for k in range(1, min(max_ind, 5) + 1):
            use = [r for r in rows if (not r[4]) and r[3] >= k]
            threshold = MIN_STACK_DISCOVERY_N if label == "DISCOVERY" else MIN_STACK_CONFIRM_N
            if len(use) < threshold: continue
            mask = np.zeros(len(g), bool); sign = np.zeros(len(g), float)
            for i, s, *_ in use: mask[i] = True; sign[i] = s
            met = _metrics(g, target, sign, mask, dashboard_module)
            log_func(
                f"[RULE-REFINE-V1-INDEPENDENT-STACK] sample={label} min_independent_mechanisms={k} n={met['n']} hit={met['hit']:.4f} "
                f"roi={met['roi']:+.4f} signed={met['signed']:+.3f} clv={met['clv']:+.3f} clv_state={_clv_state(met)} "
                f"candidate_orientations_frozen_from_discovery=TRUE production_authority=0"
            )
            results[(label, k)] = met

        # Exact generalized mechanism signatures, independent of source/rule names.
        by_sig = defaultdict(list)
        for r in rows:
            i, s, nf, ni, conflict, sig = r
            if conflict: continue
            by_sig[(sig, ni)].append((i, s))
        for (sig, ni), vals in sorted(by_sig.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            threshold = MIN_STACK_DISCOVERY_N if label == "DISCOVERY" else MIN_STACK_CONFIRM_N
            if len(vals) < threshold: continue
            mask = np.zeros(len(g), bool); sign = np.zeros(len(g), float)
            for i, s in vals: mask[i] = True; sign[i] = s
            met = _metrics(g, target, sign, mask, dashboard_module)
            log_func(
                f"[RULE-REFINE-V1-MECHANISM-STACK] sample={label} mechanisms={'+'.join(sig) or 'NONE'} independent_mechanisms={ni} "
                f"n={met['n']} hit={met['hit']:.4f} roi={met['roi']:+.4f} signed={met['signed']:+.3f} clv={met['clv']:+.3f} "
                f"clv_state={_clv_state(met)} production_authority=0"
            )
    return results, selected_keys


def run_atomic_rule_refinement_v1(*, dashboard_module, stat_out: dict, registry_out: dict, atomic_graph_out: dict | None = None, log_func=print, hard_fail=True):
    try:
        cache = getattr(dashboard_module, "_V1357_SPREAD_RESEARCH_CACHE", {})
        g = cache.get("games") if isinstance(cache, dict) else None
        if not isinstance(g, pd.DataFrame) or g.empty:
            raise RuntimeError("spread research games unavailable")
        g = g.copy()
        if int(et._physical_key(g).duplicated().sum()):
            raise RuntimeError("physical game duplicates present")
        target = pd.to_numeric(g.get("Market_Error_Margin"), errors="coerce").to_numpy(float)

        drules, crules = _build_rule_rows(g, dashboard_module, stat_out, registry_out)
        if not drules:
            raise RuntimeError("no atomic rules available")
        counts = {s: sum(r["source"] == s for r in drules) for s in emm.SOURCE_ORDER}
        if any(counts.get(s, 0) == 0 for s in ("STAT", "BIGAL", "PATHI", "MINER")):
            raise RuntimeError(f"missing peer source nodes counts={counts}")

        log_func(
            f"[RULE-REFINE-V1-PREFLIGHT] status=PASS source_tag={ATOMIC_RULE_REFINEMENT_V1_SOURCE_TAG} games={len(g)} atomic_rules={len(drules)} "
            f"rule_counts={counts} discovery_seasons={list(DISCOVERY_SEASONS)} confirm_season={CONFIRM_SEASON} "
            f"orientation_selected_from_discovery_only=TRUE play_and_fade_both_evaluated=TRUE lineage_condition_sets_auditable=TRUE "
            f"clv_support_not_veto=TRUE zero_production_authority=TRUE"
        )

        orientation_rows = _select_rule_orientations(g, target, drules, crules, dashboard_module, log_func)
        edges = _lineage_edges(orientation_rows)
        lineage_rows = _lineage_incrementality(g, target, orientation_rows, edges, dashboard_module, log_func)
        families, rule_to_family = _evidence_families(g, orientation_rows, lineage_rows, log_func)
        stack_rows, selected_keys = _independent_stack_eval(g, target, orientation_rows, families, dashboard_module, log_func)

        repeat_play = sum(str(r["state"]).startswith("REPEATABLE_PLAY") for r in orientation_rows)
        repeat_fade = sum(str(r["state"]).startswith("REPEATABLE_FADE") for r in orientation_rows)
        add_value = sum(r.get("state") == "REFINEMENT_ADDS_VALUE_BOTH" for r in lineage_rows)
        no_value = sum(r.get("state") == "NO_INCREMENTAL_VALUE" for r in lineage_rows)
        log_func(
            f"[RULE-REFINE-V1-CONTRACT] status=PASS source_tag={ATOMIC_RULE_REFINEMENT_V1_SOURCE_TAG} atomic_rules={len(orientation_rows)} "
            f"repeatable_play_rules={repeat_play} repeatable_fade_rules={repeat_fade} lineage_edges={len(lineage_rows)} "
            f"lineage_refinements_add_value={add_value} lineage_refinements_no_value={no_value} related_evidence_families={len(families)} "
            f"discovery_selected_stack_rules={len(selected_keys)} play_fade_orientation_frozen_before_2026=TRUE parent_child_incrementality_tested=TRUE "
            f"related_rules_count_once_for_independence=TRUE independent_mechanism_stacks_retested=TRUE clv_tracked=TRUE clv_is_support_not_veto=TRUE zero_production_authority=TRUE"
        )
        return {
            "status": "PASS", "source_tag": ATOMIC_RULE_REFINEMENT_V1_SOURCE_TAG,
            "orientations": [{k:v for k,v in r.items() if not str(k).startswith("vote_")} for r in orientation_rows],
            "lineage": lineage_rows, "families": families, "rule_to_family": rule_to_family,
            "independent_stacks": stack_rows, "selected_rule_keys": sorted(selected_keys),
            "production_authority": 0,
        }
    except Exception as e:
        log_func(f"[RULE-REFINE-V1-CONTRACT] status=FAILED error={type(e).__name__}:{e} production_authority=0")
        if hard_fail:
            raise
        return {"status":"FAILED", "error":f"{type(e).__name__}:{e}", "production_authority":0}
