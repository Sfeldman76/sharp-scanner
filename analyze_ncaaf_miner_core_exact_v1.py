"""Research-only exact Miner vs V5.5 CORE pairing, with explicit unmatched coverage.

Do not use this result as independent historical ATS certification: rule definitions,
feature vintages, and bookmaker timestamps are not attested. Join by physical CFBD
Game_ID only and require target team to equal the independently provided home/away
name. Never equate absence of a reconstructed firing to a known NO_FIRE.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

VERSION = "NCAAF_MINER_CORE_EXACT_RETROSPECTIVE_PAIR_V1_20261024"


def _team(x):
    return re.sub(r"[^a-z0-9]", "", str(x or "").strip().lower())


def _keys(df, label):
    required = {"season", "game_id"}
    if not required <= set(df.columns):
        raise ValueError(label + ":REQUIRED_COLUMNS_MISSING:" + ",".join(sorted(required - set(df.columns))))
    df = df.copy()
    df["season"] = pd.to_numeric(df.season, errors="raise").astype(int)
    df["game_id"] = df.game_id.astype("string").str.strip()
    if df.duplicated(["season", "game_id"]).any():
        raise ValueError(label + ":DUPLICATE_PHYSICAL_GAME_ID")
    return df


def run_exact_core_pairing(core, universe, exact_firings):
    """Pair only IDs and teams verified on *both* sides; no fuzzy fallback."""
    required = {"core_side", "core_ats_sign", "season", "game_id"}
    if not required <= set(core.columns):
        raise ValueError("CORE_MISSING:" + ",".join(sorted(required - set(core.columns))))
    universe = _keys(universe, "CFBD_UNIVERSE")
    for t in ("home_team_key", "away_team_key"):
        if t not in universe.columns:
            raise ValueError("CFBD_UNIVERSE_MISSING:" + t)
    core = _keys(core, "CORE")
    core = core.merge(universe[["season", "game_id", "home_team_key", "away_team_key"]],
                      on=["season", "game_id"], how="left", validate="one_to_one")
    core["split"] = core.season.map(lambda y: "DISCOVERY" if y <= 2021 else "SELECTION" if y <= 2023 else "RETROSPECTIVE_CONFIRMATION")
    missing_cols = {"season", "source_game_id", "target_team", "market", "mechanism_id", "evidence_family_key"} - set(exact_firings.columns)
    if missing_cols:
        raise ValueError("EXACT_FIRINGS_MISSING:" + ",".join(sorted(missing_cols)))
    f = exact_firings.copy()
    f = f[f.market.eq("spreads")].copy()  # CORE spread sign only
    f["season"] = pd.to_numeric(f.season, errors="coerce")
    f = f[f.season.notna() & f.season.between(2013, 2025)].copy()
    f["season"] = f.season.astype(int)
    f["game_id"] = f.source_game_id.astype("string").str.strip()
    f = f.merge(universe[["season", "game_id", "home_team_key", "away_team_key"]],
                on=["season", "game_id"], how="left", validate="many_to_one", indicator=True)
    f["resolved_target"] = f.target_team.map(_team)
    f["home_key"] = f.home_team_key.map(_team)
    f["away_key"] = f.away_team_key.map(_team)
    f["direction_verified"] = np.where(f._merge.ne("both"), "UNKNOWN_GAME_ID",
        np.where(f.resolved_target.eq(f.home_key) & f.home_key.ne(""), "HOME",
        np.where(f.resolved_target.eq(f.away_key) & f.away_key.ne(""), "AWAY", "UNKNOWN_TEAM_ORIENTATION")))
    bad = f[f.direction_verified.str.startswith("UNKNOWN")].copy()
    verified = f[f.direction_verified.isin(("HOME", "AWAY"))].copy()
    match_keys = core[["season", "game_id", "core_side", "core_ats_sign"]]
    verified = verified.merge(match_keys, on=["season", "game_id"], how="inner", validate="many_to_one")
    # One evidence family may have many highly correlated mechanism IDs. If
    # any point in opposite directions, family cannot cast a directional vote.
    votes = (verified.groupby(["season", "game_id", "evidence_family_key"], dropna=False)
             .agg(targets=("direction_verified", lambda x: tuple(sorted(set(x)))),
                  mechanisms=("mechanism_id", lambda x: tuple(sorted(set(x)))))
             .reset_index())
    bygame = defaultdict(list)
    for row in votes.itertuples(index=False):
        bygame[(int(row.season), str(row.game_id))].append(row)
    paired = []
    pair_summary = defaultdict(lambda: {"games": 0, "wins": 0, "losses": 0, "pushes": 0})
    for row in core.itertuples(index=False):
        year, gid = int(row.season), str(row.game_id)
        b = bygame.get((year, gid), [])
        core_side = str(row.core_side)
        if core_side not in ("HOME", "AWAY"):
            raise ValueError("INVALID_CORE_DIRECTION")
        supporting, opposing, conflicted = set(), set(), set()
        for vote in b:
            if len(vote.targets) > 1:
                conflicted.add(vote.evidence_family_key)
            elif vote.targets[0] == core_side:
                supporting.add(vote.evidence_family_key)
            else:
                opposing.add(vote.evidence_family_key)
        if conflicted:
            state = "INTERNAL_FAMILY_CONFLICT"
        elif supporting and opposing:
            state = "SUPPORT_AND_OPPOSITION"
        elif len(supporting) >= 2:
            state = "TWO_PLUS_INDEPENDENT_SUPPORT"
        elif supporting:
            state = "ONE_INDEPENDENT_SUPPORT"
        elif opposing:
            state = "OPPOSITION_ONLY"
        else:
            state = "NO_MATCHED_FIRING__NOT_CERTIFIED_NONFIRE"
        ats = pd.to_numeric(pd.Series([row.core_ats_sign]), errors="coerce").iloc[0]
        outcome = "WIN" if ats == 1 else "LOSS" if ats == -1 else "PUSH" if ats == 0 else "NO_OUTCOME"
        paired.append({"season": year, "game_id": gid, "split": row.split,
                       "core_side": core_side, "core_ats_sign": ats, "outcome": outcome,
                       "independent_support_family_count": len(supporting),
                       "independent_opposition_family_count": len(opposing),
                       "internally_conflicted_family_count": len(conflicted),
                       "support_families": "|".join(sorted(supporting)),
                       "opposition_families": "|".join(sorted(opposing)),
                       "conflicted_families": "|".join(sorted(conflicted)),
                       "research_cohort": state, "no_fire_certified": False,
                       "original_price_vintage_certified": False,
                       "independent_holdout_certified": False,
                       "bet_authority": 0})
        for a, b in combinations(sorted(supporting), 2):
            key = (year, a, b)
            p = pair_summary[key]
            p["games"] += 1
            p["wins"] += int(outcome == "WIN")
            p["losses"] += int(outcome == "LOSS")
            p["pushes"] += int(outcome == "PUSH")
    result = pd.DataFrame(paired)
    summary = (result.groupby(["split", "research_cohort"], dropna=False)
               .agg(games=("game_id", "size"),
                    wins=("core_ats_sign", lambda x: int(x.eq(1).sum())),
                    losses=("core_ats_sign", lambda x: int(x.eq(-1).sum())),
                    pushes=("core_ats_sign", lambda x: int(x.eq(0).sum())))
               .reset_index()) if len(result) else pd.DataFrame()
    by_year = (result.groupby(["season", "research_cohort"], dropna=False)
               .agg(games=("game_id", "size"),
                    wins=("core_ats_sign", lambda x: int(x.eq(1).sum())),
                    losses=("core_ats_sign", lambda x: int(x.eq(-1).sum())),
                    pushes=("core_ats_sign", lambda x: int(x.eq(0).sum())))
               .reset_index()) if len(result) else pd.DataFrame()
    for tbl in (summary, by_year):
        if not tbl.empty:
            tbl["nonpush_rate"] = tbl.wins / (tbl.wins + tbl.losses).replace(0, np.nan)
            tbl["certified_incremental_value"] = False
    pairwise = pd.DataFrame([{"season": yr, "family_a": a, "family_b": b,
                              **perf, "nonpush_rate": perf["wins"] / max(1, perf["wins"] + perf["losses"]),
                              "incremental_value_certified": False}
                             for (yr, a, b), perf in sorted(pair_summary.items())])
    audit = {"status": "STRICT_ID_AND_TEAM_MATCH_RETROSPECTIVE", "version": VERSION,
             "core_games": len(core), "total_exact_spread_firings": len(f),
             "firings_matching_universe_id_and_target": int((~f.direction_verified.str.startswith("UNKNOWN")).sum()),
             "firings_not_matching_universe_id_or_target": len(bad),
             "firings_paired_to_core": len(verified),
             "matched_core_games_with_positive_fires": len(bygame),
             "core_games_without_positive_fires_not_certified_nonfire": int(result.research_cohort.eq("NO_MATCHED_FIRING__NOT_CERTIFIED_NONFIRE").sum()) if len(result) else 0,
             "source_rule_definition_vintage_certified": False,
             "2024_2025_independent_holdout_certified": False,
             "bookmaker_original_quote_timestamp_certified": False,
             "production_mutation": False, "bet_authority": 0,
             "limitations": "Only direct CFBD IDs + direct normalized school names. Unmatched events are never inferred absent. No automatic betting promotion."}
    return {"audit": audit, "core_paired": result, "cohorts": summary, "by_season": by_year,
            "supporting_family_pairs": pairwise,
            "unmatched_firings": bad[["season", "source_game_id", "mechanism_id", "target_team", "direction_verified"]]}


def main():
    ap = argparse.ArgumentParser(description="Read-only exact Miner-to-CORE retrospective comparison")
    ap.add_argument("--core", required=True, help="replay/core_matched_game_edges.csv")
    ap.add_argument("--universe", required=True, help="replay/game_join_coverage.csv")
    ap.add_argument("--firings", required=True, help="new Miner shadow/exact_firings.csv")
    ap.add_argument("--out-dir", required=True)
    opts = ap.parse_args()
    data = run_exact_core_pairing(pd.read_csv(opts.core), pd.read_csv(opts.universe), pd.read_csv(opts.firings))
    out = Path(opts.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, value in data.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(out / (name + ".csv"), index=False)
        else:
            (out / (name + ".json")).write_text(json.dumps(value, indent=2, default=str))
    print(json.dumps(data["audit"], indent=2))


if __name__ == "__main__":
    main()
