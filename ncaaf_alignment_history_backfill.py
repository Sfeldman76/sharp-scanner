#!/usr/bin/env python3
"""Backfill leakage-safe historical NCAAF team alignment from CFBD game metadata.

Purpose
-------
Populate `team_sport_alignment_history` for historical seasons before the
current alignment table begins. The table remains append-only: this script never
updates or deletes existing alignment rows. It only inserts missing effective
change-points for canonical teams that can already be resolved from the existing
alignment/alias tables.

Design
------
* CFBD /games is the historical source for season-specific conference and
  classification metadata.
* Existing alignment rows are used only for stable canonical identity.
* Historical conference/subdivision is taken from the historical CFBD season,
  never borrowed backward from a future alignment row.
* A new row is emitted for the first observed season for a program, whenever its
  conference/subdivision changes, or after a gap in observed seasons.
* Ambiguous identity or conflicting season metadata fails closed and is skipped.
* Existing (Sport, Team_Norm, Effective_Season_From) rows are never overwritten.

Typical usage
-------------
  export CFBD_API_KEY="$(gcloud secrets versions access 1 --secret=cfbd-api-key --project=sharplogger)"

  # inspect only
  python3 ncaaf_alignment_history_backfill.py --project sharplogger --dataset sharp_data \
      --start-season 2013 --end-season 2021 --dry-run

  # append validated rows
  python3 ncaaf_alignment_history_backfill.py --project sharplogger --dataset sharp_data \
      --start-season 2013 --end-season 2021 --apply
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://api.collegefootballdata.com"
SOURCE_TAG = "ncaaf-alignment-history-cfbd-v1-20261015"


def _team_key(x: Any) -> str:
    s = "" if x is None else str(x)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _clean_str(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "<na>"}:
        return None
    return s


def _classification_to_subdivision(x: Any) -> str | None:
    s = _clean_str(x)
    if not s:
        return None
    u = s.upper().replace("-", " ").replace("_", " ")
    if "FBS" in u:
        return "FBS"
    if "FCS" in u:
        return "FCS"
    return s


class CFBD:
    def __init__(self) -> None:
        key = os.getenv("CFBD_API_KEY", "").strip()
        if not key:
            raise RuntimeError("CFBD_API_KEY is not set")
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {key}", "Accept": "application/json"})
        self.calls = 0

    def games(self, year: int) -> pd.DataFrame:
        url = BASE_URL + "/games"
        params = {"year": int(year), "seasonType": "both", "classification": "fbs"}
        r = self.s.get(url, params=params, timeout=90)
        self.calls += 1
        if r.status_code >= 400:
            raise RuntimeError(f"CFBD /games {year} failed HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        if not isinstance(data, list):
            raise RuntimeError(f"CFBD /games {year} returned unexpected payload type {type(data).__name__}")
        return pd.json_normalize(data)


@dataclass(frozen=True)
class Identity:
    team: str
    team_norm: str
    canonical_team_name: str
    canonical_team_norm: str
    entity_type: str | None
    league_group: str | None


def _load_identity_maps(client, project: str, dataset: str) -> tuple[dict[str, Identity], set[str]]:
    """Resolve stable canonical identity from all existing NCAAF alignment rows.

    Returns map and a set of keys that were conflicting and therefore removed.
    """
    table = f"{project}.{dataset}.team_sport_alignment_history"
    d = client.query(f"""
      SELECT Team, Team_Norm, Canonical_Team_Name, Canonical_Team_Norm,
             Entity_Type, League_Group, Effective_Season_From
      FROM `{table}`
      WHERE UPPER(TRIM(Sport))='NCAAF' AND COALESCE(Use_For_Context, TRUE)
      ORDER BY Effective_Season_From DESC
    """).to_dataframe(create_bqstorage_client=False)

    out: dict[str, Identity] = {}
    conflicts: set[str] = set()

    for _, r in d.iterrows():
        team = _clean_str(r.get("Team")) or _clean_str(r.get("Canonical_Team_Name")) or ""
        team_norm = _clean_str(r.get("Team_Norm")) or _team_key(team)
        canonical_name = _clean_str(r.get("Canonical_Team_Name")) or team
        canonical_norm = _clean_str(r.get("Canonical_Team_Norm")) or team_norm
        ident = Identity(
            team=team,
            team_norm=team_norm.lower().strip(),
            canonical_team_name=canonical_name,
            canonical_team_norm=canonical_norm.lower().strip(),
            entity_type=_clean_str(r.get("Entity_Type")) or "Team",
            league_group=_clean_str(r.get("League_Group")),
        )
        for v in (team, team_norm, canonical_name, canonical_norm):
            k = _team_key(v)
            if not k:
                continue
            prior = out.get(k)
            if prior is not None and prior.canonical_team_norm != ident.canonical_team_norm:
                conflicts.add(k)
                out.pop(k, None)
                continue
            if k not in conflicts:
                out[k] = ident

    # Add source aliases only when their target resolves uniquely.
    alias_table = f"{project}.{dataset}.ncaaf_source_team_aliases"
    try:
        a = client.query(f"""
          SELECT Source_Team_Name, Team_Norm
          FROM `{alias_table}`
          WHERE Source_Team_Name IS NOT NULL AND Team_Norm IS NOT NULL
          QUALIFY ROW_NUMBER() OVER (
            PARTITION BY LOWER(TRIM(Source_Team_Name))
            ORDER BY Updated_At DESC
          )=1
        """).to_dataframe(create_bqstorage_client=False)
    except Exception:
        a = pd.DataFrame()
    if not a.empty:
        for _, r in a.iterrows():
            src = _team_key(r.get("Source_Team_Name"))
            tgt = _team_key(r.get("Team_Norm"))
            ident = out.get(tgt)
            if not src or ident is None:
                continue
            prior = out.get(src)
            if prior is not None and prior.canonical_team_norm != ident.canonical_team_norm:
                conflicts.add(src)
                out.pop(src, None)
            elif src not in conflicts:
                out[src] = ident

    return out, conflicts


def _mode_or_conflict(values: list[str | None]) -> tuple[str | None, bool]:
    vals = [v for v in (_clean_str(x) for x in values) if v]
    if not vals:
        return None, False
    counts = collections.Counter(vals)
    if len(counts) == 1:
        return vals[0], False
    top = counts.most_common()
    # Conference should be stable within a season. If multiple values occur,
    # fail closed unless one value overwhelmingly dominates stray bad records.
    winner, n = top[0]
    total = sum(counts.values())
    if n / total >= 0.90:
        return winner, True
    return None, True


def _extract_team_seasons(games: pd.DataFrame, season: int, identities: dict[str, Identity]) -> tuple[pd.DataFrame, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    unresolved: collections.Counter[str] = collections.Counter()

    def add(team: Any, conf: Any, cls: Any) -> None:
        name = _clean_str(team)
        if not name:
            return
        ident = identities.get(_team_key(name))
        if ident is None:
            unresolved[name] += 1
            return
        k = ident.canonical_team_norm
        b = buckets.setdefault(k, {"identity": ident, "conference": [], "subdivision": [], "source_names": set()})
        b["conference"].append(_clean_str(conf))
        b["subdivision"].append(_classification_to_subdivision(cls))
        b["source_names"].add(name)

    for _, g in games.iterrows():
        add(g.get("homeTeam"), g.get("homeConference"), g.get("homeClassification"))
        add(g.get("awayTeam"), g.get("awayConference"), g.get("awayClassification"))

    rows = []
    conflicts = []
    for canon, b in buckets.items():
        conf, conf_conflict = _mode_or_conflict(b["conference"])
        subdiv, subdiv_conflict = _mode_or_conflict(b["subdivision"])
        if conf_conflict and conf is None:
            conflicts.append({"team": canon, "field": "Conference", "values": sorted(set(x for x in b["conference"] if x))})
            continue
        if subdiv_conflict and subdiv is None:
            conflicts.append({"team": canon, "field": "Subdivision", "values": sorted(set(x for x in b["subdivision"] if x))})
            continue
        ident: Identity = b["identity"]
        rows.append({
            "Season": int(season),
            "Team": ident.team,
            "Team_Norm": ident.team_norm,
            "Canonical_Team_Name": ident.canonical_team_name,
            "Canonical_Team_Norm": ident.canonical_team_norm,
            "Conference": conf,
            "Subdivision": subdiv,
            "Entity_Type": ident.entity_type or "Team",
            "League_Group": ident.league_group,
            "Source_Names": " | ".join(sorted(b["source_names"])),
            "Conference_Had_Minor_Conflict": bool(conf_conflict and conf is not None),
            "Subdivision_Had_Minor_Conflict": bool(subdiv_conflict and subdiv is not None),
        })

    diag = {
        "season": int(season),
        "resolved_teams": len(rows),
        "unresolved_team_names": len(unresolved),
        "unresolved_game_mentions": int(sum(unresolved.values())),
        "conflicting_team_seasons": len(conflicts),
        "unresolved_examples": unresolved.most_common(12),
        "conflict_examples": conflicts[:12],
    }
    return pd.DataFrame(rows), diag


def _compress_change_points(season_rows: pd.DataFrame) -> pd.DataFrame:
    if season_rows.empty:
        return season_rows
    out = []
    for team_norm, g in season_rows.sort_values(["Team_Norm", "Season"]).groupby("Team_Norm", sort=False):
        prev_state = None
        prev_season = None
        for _, r in g.iterrows():
            season = int(r["Season"])
            state = (_clean_str(r.get("Conference")), _clean_str(r.get("Subdivision")))
            gap = prev_season is not None and season > prev_season + 1
            if prev_state is None or state != prev_state or gap:
                q = dict(r)
                q["Effective_Season_From"] = season
                q["Change_Reason"] = "FIRST_HISTORICAL_OBSERVATION" if prev_state is None else ("RETURN_AFTER_GAP" if gap and state == prev_state else "ALIGNMENT_CHANGE")
                out.append(q)
            prev_state = state
            prev_season = season
    return pd.DataFrame(out)


def _existing_keys(client, project: str, dataset: str) -> set[tuple[str, int]]:
    tid = f"{project}.{dataset}.team_sport_alignment_history"
    d = client.query(f"""
      SELECT LOWER(TRIM(Team_Norm)) Team_Norm, Effective_Season_From
      FROM `{tid}`
      WHERE UPPER(TRIM(Sport))='NCAAF'
    """).to_dataframe(create_bqstorage_client=False)
    return {(str(r.Team_Norm), int(r.Effective_Season_From)) for r in d.itertuples(index=False)}


def _prepare_insert_rows(changes: pd.DataFrame, existing: set[tuple[str, int]]) -> pd.DataFrame:
    if changes.empty:
        return changes
    rows = []
    for _, r in changes.iterrows():
        key = (str(r["Team_Norm"]).strip().lower(), int(r["Effective_Season_From"]))
        if key in existing:
            continue
        rows.append({
            "Sport": "NCAAF",
            "Team": _clean_str(r.get("Team")),
            "Team_Norm": str(r.get("Team_Norm") or "").strip().lower(),
            "Canonical_Team_Name": _clean_str(r.get("Canonical_Team_Name")),
            "Canonical_Team_Norm": str(r.get("Canonical_Team_Norm") or "").strip().lower(),
            "Effective_Season_From": int(r["Effective_Season_From"]),
            "Conference": _clean_str(r.get("Conference")),
            "Division": None,
            "Subdivision": _clean_str(r.get("Subdivision")),
            # Do not project a modern League_Group backward; leave historical
            # league grouping null unless separately sourced historically.
            "League_Group": None,
            "Entity_Type": _clean_str(r.get("Entity_Type")) or "Team",
            "Active": True,
            "Use_For_Context": True,
            "Source_URL": "CollegeFootballData /games historical season metadata",
            "Notes": f"{SOURCE_TAG}; {r.get('Change_Reason')}; source_names={r.get('Source_Names')}",
        })
    return pd.DataFrame(rows)


def _append_rows(client, project: str, dataset: str, rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"status": "NO_NEW_ROWS", "inserted": 0}
    from google.cloud import bigquery
    tid = f"{project}.{dataset}.team_sport_alignment_history"
    tmp = f"{project}.{dataset}._ncaaf_alignment_history_cfbd_stage_{int(time.time())}"
    client.load_table_from_dataframe(rows, tmp, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")).result()
    cols = [
        "Sport","Team","Team_Norm","Canonical_Team_Name","Canonical_Team_Norm",
        "Effective_Season_From","Conference","Division","Subdivision","League_Group",
        "Entity_Type","Active","Use_For_Context","Source_URL","Notes"
    ]
    ic = ", ".join(f"`{c}`" for c in cols)
    iv = ", ".join(f"S.`{c}`" for c in cols)
    client.query(f"""
      MERGE `{tid}` T
      USING `{tmp}` S
      ON UPPER(TRIM(T.Sport))=UPPER(TRIM(S.Sport))
       AND LOWER(TRIM(T.Team_Norm))=LOWER(TRIM(S.Team_Norm))
       AND T.Effective_Season_From=S.Effective_Season_From
      WHEN NOT MATCHED THEN INSERT ({ic}) VALUES ({iv})
    """).result()
    client.delete_table(tmp, not_found_ok=True)
    return {"status": "PASS", "candidate_rows": len(rows)}


def _validate(client, project: str, dataset: str, start: int, end: int) -> dict[str, Any]:
    tid = f"{project}.{dataset}.team_sport_alignment_history"
    dup = client.query(f"""
      SELECT COUNT(*) n FROM (
        SELECT Team_Norm, Effective_Season_From, COUNT(*) c
        FROM `{tid}`
        WHERE UPPER(TRIM(Sport))='NCAAF'
        GROUP BY 1,2 HAVING c>1
      )
    """).to_dataframe(create_bqstorage_client=False)
    hist = client.query(f"""
      SELECT MIN(Effective_Season_From) min_season,
             MAX(Effective_Season_From) max_season,
             COUNT(*) rows,
             COUNT(DISTINCT Team_Norm) teams
      FROM `{tid}`
      WHERE UPPER(TRIM(Sport))='NCAAF'
        AND Effective_Season_From BETWEEN {int(start)} AND {int(end)}
    """).to_dataframe(create_bqstorage_client=False)
    return {
        "duplicate_effective_keys": int(dup.iloc[0]["n"]) if len(dup) else None,
        "historical_window": hist.iloc[0].to_dict() if len(hist) else {},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=os.getenv("BQ_PROJECT", "sharplogger"))
    p.add_argument("--dataset", default=os.getenv("BQ_DATASET", "sharp_data"))
    p.add_argument("--start-season", type=int, default=2013)
    p.add_argument("--end-season", type=int, default=2021)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Append missing historical effective rows")
    mode.add_argument("--dry-run", action="store_true", help="Inspect only (default)")
    args = p.parse_args()
    if args.end_season < args.start_season:
        raise SystemExit("end-season must be >= start-season")

    from google.cloud import bigquery
    bq = bigquery.Client(project=args.project)
    cfbd = CFBD()
    identities, conflicts = _load_identity_maps(bq, args.project, args.dataset)
    print(f"[ALIGNMENT-BACKFILL] identity_keys={len(identities)} identity_conflicts={len(conflicts)} source={SOURCE_TAG}")

    season_frames = []
    diagnostics = []
    for year in range(args.start_season, args.end_season + 1):
        print(f"[ALIGNMENT-SEASON] year={year} status=START calls={cfbd.calls}")
        games = cfbd.games(year)
        frame, diag = _extract_team_seasons(games, year, identities)
        diagnostics.append(diag)
        if not frame.empty:
            season_frames.append(frame)
        print(
            f"[ALIGNMENT-SEASON] year={year} status=PASS games={len(games)} "
            f"resolved_teams={diag['resolved_teams']} unresolved_names={diag['unresolved_team_names']} "
            f"conflicts={diag['conflicting_team_seasons']} calls={cfbd.calls}"
        )
        if diag["unresolved_examples"]:
            print(f"[ALIGNMENT-UNRESOLVED] year={year} examples={json.dumps(diag['unresolved_examples'])}")
        if diag["conflict_examples"]:
            print(f"[ALIGNMENT-CONFLICT] year={year} examples={json.dumps(diag['conflict_examples'])}")

    all_rows = pd.concat(season_frames, ignore_index=True) if season_frames else pd.DataFrame()
    changes = _compress_change_points(all_rows)
    existing = _existing_keys(bq, args.project, args.dataset)
    inserts = _prepare_insert_rows(changes, existing)

    print(
        f"[ALIGNMENT-CHANGEPOINTS] team_seasons={len(all_rows)} change_points={len(changes)} "
        f"new_rows={len(inserts)} existing_keys={len(existing)}"
    )
    if not inserts.empty:
        preview = inserts[["Team_Norm","Effective_Season_From","Conference","Subdivision","Notes"]].sort_values(["Team_Norm","Effective_Season_From"])
        print("[ALIGNMENT-PREVIEW]\n" + preview.head(80).to_string(index=False))

    if args.apply:
        res = _append_rows(bq, args.project, args.dataset, inserts)
        print(f"[ALIGNMENT-APPLY] {res}")
        print(f"[ALIGNMENT-VALIDATION] {_validate(bq,args.project,args.dataset,args.start_season,args.end_season)}")
    else:
        print("[ALIGNMENT-DRY-RUN] status=PASS no_bigquery_mutation=TRUE rerun_with=--apply")


if __name__ == "__main__":
    main()
