"""Shadow-only extraction of *actual* NCAAF Miner mechanisms on the enriched research frame.

This is NOT a certified historical backtest. Exact mechanism conditions are evaluated
using the present-day research library; mechanism discovery and data/quote vintages
are not proven point-in-time. No rule, quote, or absent trigger is fabricated.

Outputs are an evidence bridge for subsequent matched CORE and cross-brain tests.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

VERSION = "NCAAF_MINER_EXACT_PATH_SHADOW_V1_20261024"
ROOT = "research/ncaaf/v2/miner_exact_path_shadow/runs"


def _safe_num(v, default=-1.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def _field(frame, choices):
    for name in choices:
        if name in frame.columns:
            return frame[name].astype("string").fillna("").str.strip().to_numpy(dtype=str)
    return np.full(len(frame), "", dtype=object)


def _discovery_rank(system, market):
    """Never uses confirmation_pass/rate/ROI or any 2024-25 field."""
    disc = system.get("h2h_discovery_price") or {}
    roi = _safe_num(disc.get("roi")) if market == "h2h" else -1.0
    return (
        bool(system.get("discovery_pass")),
        roi,
        _safe_num(system.get("discovery_rate")),
        int(_safe_num(system.get("discovery_n"), 0)),
        str(system.get("system_id") or ""),
    )


def discovery_only_representatives(miners):
    records = []
    for market, registry in sorted((miners or {}).items()):
        candidates = {str(s.get("system_id")): s for s in (registry or {}).get("systems", [])}
        for mech in (registry or {}).get("mechanism_families", []):
            ids = [str(x) for x in mech.get("member_system_ids") or []]
            eligible = [candidates[k] for k in ids if k in candidates]
            chosen = max(eligible, key=lambda x: _discovery_rank(x, market)) if eligible else None
            incumbent = str(mech.get("representative_system_id") or "")
            records.append({
                "market": market, "mechanism_id": mech.get("mechanism_id"),
                "reported_representative_id": incumbent,
                "discovery_only_representative_id": str(chosen.get("system_id")) if chosen else "",
                "discovery_only_conditions": " AND ".join(chosen.get("conditions") or []) if chosen else "",
                "discovery_only_n": chosen.get("discovery_n") if chosen else None,
                "discovery_only_rate": chosen.get("discovery_rate") if chosen else None,
                "changed": bool(chosen is not None and str(chosen.get("system_id")) != incumbent),
                "candidate_member_count": len(eligible),
                "member_count_reported": len(ids),
                "selection_status": "SHADOW_ONLY" if chosen else "MEMBER_HISTORY_NOT_AVAILABLE",
                "selection_uses_confirmation": False,
                "report_representative_selection_can_use_confirmation": True,
                "production_authority": 0,
            })
    return pd.DataFrame.from_records(records)


def extract_exact_path_shadow(games: pd.DataFrame, seasons, miners: dict,
                              atom_resolver, *, max_year=2025):
    """Evaluate exact recorded mechanism conditions, never reconstructed proxies.

    atom_resolver(market) must return the native Miner atoms on *the same* frame.
    Only positive, complete conjunctions are called FIRE. Missing atoms -> UNKNOWN.
    A false boolean mask is NOT certified NO_FIRE when historical source coverage
    or point-in-time vintage is not verified.
    """
    if not isinstance(games, pd.DataFrame):
        raise TypeError("games must be a DataFrame")
    sy = pd.to_numeric(pd.Series(seasons), errors="coerce").to_numpy(dtype=float)
    if len(sy) != len(games):
        raise ValueError("ROW_ALIGNMENT_MISMATCH")
    if np.any(np.isfinite(sy) & (sy > max_year)):
        # A masked dataset is allowed, but its future rows must NEVER be emitted.
        pass
    n = len(games)
    ids = _field(games, ("Source_Game_ID", "Game_ID", "game_id", "CFBD_Game_ID"))
    team = _field(games, ("Team", "Team_Name", "Home_Team"))
    opp = _field(games, ("Opponent", "Opponent_Name", "Away_Team"))
    game_date = _field(games, ("Game_Date", "Game_Start", "Kickoff_UTC", "Date"))
    fire_records, mech_records, alternative_records = [], [], []
    evaluable_alternatives = set()
    discovery_reps = discovery_only_representatives(miners)
    alternative_map = {}
    if not discovery_reps.empty:
        for _rep in discovery_reps.to_dict("records"):
            if _rep["changed"] and _rep["discovery_only_conditions"]:
                alternative_map[(str(_rep["market"]),str(_rep["mechanism_id"]))] = [
                    c.strip() for c in str(_rep["discovery_only_conditions"]).split(" AND ") if c.strip()]
    statuses = defaultdict(int)
    for market, registry in sorted((miners or {}).items()):
        original_atoms = atom_resolver(market)
        atoms = {}
        for atom in original_atoms:
            name = str(atom.get("name") or "")
            if not name:
                continue
            mask = pd.Series(atom["mask"]).fillna(False).astype(bool).to_numpy()
            if len(mask) != n:
                raise ValueError("ATOM_ROW_ALIGNMENT_MISMATCH:" + name)
            if name in atoms:
                raise ValueError("DUPLICATE_ATOM:" + name)
            atoms[name] = mask
        for mech in (registry or {}).get("mechanism_families", []):
            mid = str(mech.get("mechanism_id") or "")
            cond = list(mech.get("representative_conditions") or [])
            if not mid or not cond:
                statuses["MISSING_EXACT_DEFINITION"] += 1
                continue
            missing = sorted(set(cond) - atoms.keys())
            evidence = str(mech.get("evidence_family_key") or mid)
            direction = str(mech.get("direction") or "").upper()
            if direction not in ("PLAY_ON", "FADE"):
                statuses["INVALID_DIRECTION"] += 1
                continue
            has_bigal = bool(mech.get("uses_big_al_observation_hypothesis") or any(c.startswith(("EXPERT_BIGAL", "OBS_")) for c in cond))
            has_pathi = any(c.startswith("EXPERT_PATHI") for c in cond)
            origin = ("BIG_AL_PATHI_INTERACTION" if has_bigal and has_pathi else
                      "BIG_AL_STYLE_VOCABULARY" if has_bigal else
                      "PATHI_CONTEXT" if has_pathi else "GENERIC_MINER")
            reval = str(mech.get("incumbent_revalidation_status") or "UNSPECIFIED")
            meta = {
                "market": market, "mechanism_id": mid, "evidence_family_key": evidence,
                "direction": direction, "conditions": " AND ".join(cond),
                "condition_count": len(cond), "path_origin": origin,
                "revalidation_status": reval,
                "legacy_held": reval in ("LEGACY_CONTRACT_HOLD", "HOLD_NOT_EVALUABLE"),
                "confirmed_in_legacy_report": bool(mech.get("confirmation_pass")),
                "live_eligible_in_legacy_report": bool(mech.get("live_authority_eligible")),
                "source_vintage_certified": False,
                "rule_selection_vintage_certified": False,
                "historical_holdout_independent": False,
                "bet_authority": 0,
            }
            alt_cond = alternative_map.get((str(market),mid))
            if alt_cond:
                alt_missing = sorted(set(alt_cond) - atoms.keys())
                if alt_missing:
                    statuses["DISCOVERY_SHADOW_MISSING_ATOMS"] += 1
                else:
                    evaluable_alternatives.add(mid)
                    alt_mask = np.ones(n, dtype=bool)
                    for c in alt_cond:
                        alt_mask &= atoms[c]
                    alt_mask &= np.isfinite(sy) & (sy <= max_year)
                    for ix in np.flatnonzero(alt_mask):
                        tgt = team[ix] if direction == "PLAY_ON" else opp[ix]
                        alternative_records.append({
                            "season": int(sy[ix]), "source_game_id": str(ids[ix]),
                            "game_date_raw": str(game_date[ix]), "source_row_index": int(ix),
                            "source_team": str(team[ix]), "source_opponent": str(opp[ix]),
                            "target_team": str(tgt), "market": market,
                            "mechanism_id": mid, "evidence_family_key": evidence,
                            "direction": direction, "conditions": " AND ".join(alt_cond),
                            "path_origin": origin, "revalidation_status": reval,
                            "legacy_held": meta["legacy_held"],
                            "evaluation": "DISCOVERY_REP_FIRES_RETROSPECTIVE_UNCERTIFIED",
                            "source_vintage_certified": False,
                            "original_rule_release_verified": False,
                            "bet_authority": 0,
                        })
            if missing:
                mech_records.append({**meta, "status": "UNKNOWN_MISSING_ATOMS", "missing_atoms": ",".join(missing), "firing_rows": 0})
                statuses["UNKNOWN_MISSING_ATOMS"] += 1
                continue
            mask = np.ones(n, dtype=bool)
            for c in cond:
                mask &= atoms[c]
            mask &= np.isfinite(sy) & (sy <= max_year)
            matched = np.flatnonzero(mask)
            mech_records.append({**meta, "status": "POSITIVE_FIRINGS_EVALUATED", "missing_atoms": "", "firing_rows": len(matched)})
            statuses["POSITIVE_FIRINGS_EVALUATED"] += 1
            for i in matched:
                t = team[i] if direction == "PLAY_ON" else opp[i]
                if not t or not ids[i]:
                    statuses["UNMATCHABLE_FIRING_MISSING_GAME_ID_OR_TARGET"] += 1
                fire_records.append({
                    "season": int(sy[i]), "source_game_id": str(ids[i]), "game_date_raw": str(game_date[i]),
                    "source_row_index": int(i), "source_team": str(team[i]), "source_opponent": str(opp[i]),
                    "target_team": str(t), "market": market,
                    "mechanism_id": mid, "evidence_family_key": evidence,
                    "direction": direction, "conditions": " AND ".join(cond),
                    "path_origin": origin, "revalidation_status": reval,
                    "legacy_held": meta["legacy_held"],
                    "evaluation": "FIRE_RETROSPECTIVE_UNCERTIFIED",
                    "source_vintage_certified": False,
                    "original_rule_release_verified": False,
                    "bet_authority": 0,
                })
    firings = pd.DataFrame.from_records(fire_records)
    catalog = pd.DataFrame.from_records(mech_records)
    rep = discovery_reps
    alternates = pd.DataFrame.from_records(alternative_records)
    confluence, pairs = summarize_confluence(firings)
    # Counterfactual *selection* experiment. Change only mechanism representatives
    # for which an alternative was actually evaluable; retain the original
    # representative for all other mechanisms (with explicit incompleteness).
    if evaluable_alternatives:
        # An alternative that produces *zero* fires must still replace its
        # incumbent. Otherwise comparison silently biases toward incumbents.
        holdouts = firings[~firings.mechanism_id.isin(evaluable_alternatives)].copy()
        discovery_cohort = pd.concat([holdouts, alternates], ignore_index=True)
        shadow_confluence, _ = summarize_confluence(discovery_cohort)
    else:
        shadow_confluence = pd.DataFrame()
    audit = {
        "version": VERSION, "status": "SHADOW_RETROSPECTIVE_EXACT_DEFINITION_EXTRACTION",
        "production_mutation": False, "bet_authority": 0,
        "source_rows": n,
        "historical_rows_in_scope": int((np.isfinite(sy) & (sy <= max_year)).sum()),
        "historical_min_year": int(np.nanmin(sy[np.isfinite(sy) & (sy <= max_year)])) if np.any(np.isfinite(sy) & (sy <= max_year)) else None,
        "historical_max_year": max_year,
        "mechanisms_cataloged": len(catalog), "retrospective_positive_firing_rows": len(firings),
        "firing_rows_without_strict_identity": int(((firings.source_game_id.eq("")) | (firings.target_team.eq(""))).sum()) if len(firings) else 0,
        "discovery_only_representatives_compared": len(rep),
        "discovery_only_rep_differs_count": int(rep.changed.sum()) if len(rep) else 0,
        "discovery_only_alternative_mechanisms_evaluable": len(evaluable_alternatives),
        "discovery_only_alternative_firing_rows": len(alternates),
        "status_counts": dict(statuses),
        "family_confluence_games": len(confluence), "family_cofire_pairs": len(pairs),
        "certified_NO_FIRE_rows": 0, "missing_rule_evaluation_coverage_certified": False,
        "no_original_big_al_pick_history_inferred": True,
        "holdout_warning": "Rules/representatives and source fields may postdate historical games and 2024-2025. This is NOT a prior-only holdout or contemporaneous market test.",
        "next_dependency": "STRICT_CFBD_GAME_ID_AND_ORIENTATION_MATCH_WITH_V55_ROLLFORWARD_CORE_AND_VERIFIED_PRICE_VINTAGE",
    }
    return {"audit": audit, "exact_firings": firings, "exact_mechanisms": catalog,
            "discovery_only_representatives": rep, "discovery_only_alternative_firings": alternates,
            "discovery_only_confluence": shadow_confluence,
            "independent_family_confluence": confluence, "family_cofire_pairs": pairs}


def summarize_confluence(firings):
    """Count independent *evidence families*, not correlated system IDs.

    Does not infer NO_FIRE when no event appears and never assumes CORE agreement.
    All source IDs must be nonempty; unknown games are excluded from grouping.
    """
    if firings.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame = firings[(firings.source_game_id != "") & (firings.target_team != "")].copy()
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    family_votes = (frame.groupby(["season", "source_game_id", "market", "evidence_family_key"])
                    .agg(targets=("target_team", lambda x: tuple(sorted({str(t).strip().lower() for t in x if str(t).strip()}))),
                         mechanism_ids=("mechanism_id", lambda x: tuple(sorted(set(x)))),
                         paths=("path_origin", lambda x: tuple(sorted(set(x)))))
                    .reset_index())
    groups, pairs = [], defaultdict(int)
    for (season, gameid, market), cohort in family_votes.groupby(["season", "source_game_id", "market"]):
        eligible = cohort[cohort.targets.map(len).eq(1)]
        conflicts = cohort[cohort.targets.map(len).gt(1)]
        tally = defaultdict(set)
        for row in eligible.itertuples():
            tally[row.targets[0]].add(row.evidence_family_key)
        winning = max(tally, key=lambda x: len(tally[x])) if tally else ""
        other_target_count = max(0, len(tally) - 1)
        state = "INTERNAL_FAMILY_CONFLICT" if len(conflicts) else "CROSS_FAMILY_CONFLICT" if other_target_count else "UNANIMOUS_OR_SINGLE_FAMILY" if tally else "UNKNOWN"
        groups.append({"season": season, "source_game_id": gameid, "market": market,
                       "independent_evidence_families": len(cohort), "uncontested_families": len(eligible),
                       "internally_conflicted_families": len(conflicts),
                       "distinct_targets_after_family_collapse": len(tally), "max_same_target_families": len(tally[winning]) if winning else 0,
                       "family_state": state, "all_brains_present": False,
                       "core_match_verified": False, "price_vintage_verified": False,
                       "production_authority": 0})
        for target, families in tally.items():
            for a, b in combinations(sorted(families), 2):
                pairs[(market, a, b)] += 1
    combinations_frame = pd.DataFrame([{"market": m, "family_a": a, "family_b": b,
                                        "games_same_target_fired": count,
                                        "historical_incremental_value_proven": False,
                                        "production_authority": 0}
                                       for (m, a, b), count in sorted(pairs.items())])
    return pd.DataFrame(groups), combinations_frame


def publish_shadow(storage_client, bucket_name, shadow, *, run_stamp=None):
    """Write only new, immutable GCS research objects; never mutate latest bets."""
    stamp = run_stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = f"{ROOT}/{stamp}"
    bucket = storage_client.bucket(bucket_name)
    receipts = []
    for key, val in shadow.items():
        if key == "audit":
            continue
        if not isinstance(val, pd.DataFrame):
            continue
        raw = val.to_csv(index=False).encode("utf-8")
        blob = bucket.blob(f"{root}/{key}.csv")
        blob.upload_from_string(raw, content_type="text/csv", if_generation_match=0)
        receipts.append({"key": key, "rows": len(val), "sha256": hashlib.sha256(raw).hexdigest(),
                         "gcs": f"gs://{bucket_name}/{root}/{key}.csv"})
    report = {**shadow["audit"], "artifact_prefix": f"gs://{bucket_name}/{root}", "files": receipts}
    body = json.dumps(report, sort_keys=True, indent=2, default=str).encode("utf-8")
    bucket.blob(f"{root}/audit.json").upload_from_string(body, content_type="application/json", if_generation_match=0)
    return report
