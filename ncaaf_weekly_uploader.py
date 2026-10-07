from __future__ import annotations

import csv
import datetime as dt
import difflib
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import pandas as pd
import streamlit as st
from google.cloud import bigquery, storage

from ncaaf_weekly_core import (
    SOURCE_NAME,
    TARGET_COLUMNS,
    build_upload_dataframe,
    normalize_match_text,
    parse_bigdataball_workbook,
    validate_upload_dataframe,
)

PROJECT_ID = os.getenv("BQ_PROJECT", "sharplogger")
DATASET_ID = os.getenv("BQ_DATASET", "sharp_data")
RAW_TABLE = f"{PROJECT_ID}.{DATASET_ID}.ncaaf_historical_game_side_raw"
CONTEXT_TABLE = f"{PROJECT_ID}.{DATASET_ID}.ncaaf_historical_game_side_context"
TRAIN_VIEW = f"{PROJECT_ID}.{DATASET_ID}.ncaaf_historical_core_training_vw"
ALIGNMENT_TABLE = f"{PROJECT_ID}.{DATASET_ID}.team_sport_alignment_history"
ALIAS_TABLE = f"{PROJECT_ID}.{DATASET_ID}.ncaaf_source_team_aliases"
STAGING_TABLE = f"{PROJECT_ID}.{DATASET_ID}.ncaaf_historical_game_side_weekly_staging"


# -----------------------------------------------------------------------------
# Verified NCAAF team bootstrap reference.
#
# BigDataBall often emits the school name only (for example ``Hampton``), while
# the alignment table uses the canonical team identity (``hampton pirates``).
# These records are only used when the source school is actually present in an
# uploaded workbook.  The MERGE is append-only by Sport + Team_Norm + effective
# season, so an existing alignment record is never overwritten.
# -----------------------------------------------------------------------------
NCAAF_TEAM_BOOTSTRAP_REFERENCE = {
    "hampton": {
        "team_norm": "hampton pirates",
        "rows": [
            {
                "team": "hampton pirates",
                "team_norm": "hampton pirates",
                "canonical_team_name": "hampton pirates",
                "canonical_team_norm": "hampton pirates",
                "effective_season_from": 2024,
                "conference": "CAA",
                "division": None,
                "subdivision": "FCS",
                "league_group": None,
                "source_url": "https://hamptonpirates.com/news/2025/12/16/hampton-university-athletics-unveils-2026-football-schedule",
                "notes": "Verified football alignment; BigDataBall school-name bootstrap.",
            }
        ],
    },
    "mercyhurst": {
        "team_norm": "mercyhurst lakers",
        "rows": [
            {
                "team": "mercyhurst lakers",
                "team_norm": "mercyhurst lakers",
                "canonical_team_name": "mercyhurst lakers",
                "canonical_team_norm": "mercyhurst lakers",
                "effective_season_from": 2024,
                "conference": "NEC",
                "division": None,
                "subdivision": "FCS",
                "league_group": None,
                "source_url": "https://hurstathletics.com/news/2026/3/2/nec-announces-2026-football-schedule.aspx",
                "notes": "Mercyhurst entered Division I/NEC football in 2024; BigDataBall school-name bootstrap.",
            }
        ],
    },
    "mississippi valley state": {
        "team_norm": "mississippi valley state delta devils",
        "rows": [
            {
                "team": "mississippi valley state delta devils",
                "team_norm": "mississippi valley state delta devils",
                "canonical_team_name": "mississippi valley state delta devils",
                "canonical_team_norm": "mississippi valley state delta devils",
                "effective_season_from": 2024,
                "conference": "SWAC",
                "division": None,
                "subdivision": "FCS",
                "league_group": None,
                "source_url": "https://mvsusports.com/news/2026/7/14/general-swac-football-media-day-set-for-july-15.aspx",
                "notes": "Verified SWAC football alignment; BigDataBall school-name bootstrap.",
            }
        ],
    },
    "tennessee state": {
        "team_norm": "tennessee state tigers",
        "rows": [
            {
                "team": "tennessee state tigers",
                "team_norm": "tennessee state tigers",
                "canonical_team_name": "tennessee state tigers",
                "canonical_team_norm": "tennessee state tigers",
                "effective_season_from": 2024,
                "conference": "Big South-OVC",
                "division": None,
                "subdivision": "FCS",
                "league_group": None,
                "source_url": "https://ovcsports.com/news/2026/6/16/ovc-announces-return-to-traditional-football-branding-for-2026-season.aspx",
                "notes": "2024 OVC/Big South football-association branding baseline.",
            },
            {
                "team": "tennessee state tigers",
                "team_norm": "tennessee state tigers",
                "canonical_team_name": "tennessee state tigers",
                "canonical_team_norm": "tennessee state tigers",
                "effective_season_from": 2025,
                "conference": "OVC-Big South",
                "division": None,
                "subdivision": "FCS",
                "league_group": None,
                "source_url": "https://ovcsports.com/news/2026/6/16/ovc-announces-return-to-traditional-football-branding-for-2026-season.aspx",
                "notes": "2025 OVC/Big South football-association branding.",
            },
            {
                "team": "tennessee state tigers",
                "team_norm": "tennessee state tigers",
                "canonical_team_name": "tennessee state tigers",
                "canonical_team_norm": "tennessee state tigers",
                "effective_season_from": 2026,
                "conference": "OVC",
                "division": None,
                "subdivision": "FCS",
                "league_group": None,
                "source_url": "https://ovcsports.com/news/2026/6/30/ovc-football-media-day-presented-by-shaw-sports-turf-coverage-guide.aspx",
                "notes": "OVC football branding restored for the 2026 season.",
            },
        ],
    },
    "utrgv": {
        "team_norm": "utrgv vaqueros",
        "rows": [
            {
                "team": "utrgv vaqueros",
                "team_norm": "utrgv vaqueros",
                "canonical_team_name": "utrgv vaqueros",
                "canonical_team_norm": "utrgv vaqueros",
                "effective_season_from": 2025,
                "conference": "Southland",
                "division": None,
                "subdivision": "FCS",
                "league_group": None,
                "source_url": "https://goutrgv.com/news/2026/1/6/football-unveils-2026-schedule.aspx",
                "notes": "UTRGV football began play in 2025 as a Southland FCS member; BigDataBall school-name bootstrap.",
            }
        ],
    },
    "west georgia": {
        "team_norm": "west georgia wolves",
        "rows": [
            {
                "team": "west georgia wolves",
                "team_norm": "west georgia wolves",
                "canonical_team_name": "west georgia wolves",
                "canonical_team_norm": "west georgia wolves",
                "effective_season_from": 2024,
                "conference": "UAC",
                "division": None,
                "subdivision": "FCS",
                "league_group": None,
                "source_url": "https://uwgathletics.com/news/2025/12/10/2026-conference-schedule-released-for-uwg-football.aspx",
                "notes": "Verified UAC football alignment; BigDataBall school-name bootstrap.",
            }
        ],
    },
}

HERE = Path(__file__).resolve().parent
REBUILD_SQL_PATH = HERE / "ncaaf_context_rebuild.sql"
UPLOADER_BUILD = "2026-10-07-V5-prediction-tracker-manual-gcs"

PT_BUCKET = os.getenv("MODEL_BUCKET", "sharp-models")
PT_RAW_PREFIX = "research/ncaaf/external/prediction_tracker/raw"
PT_SNAPSHOT_PREFIX = "research/ncaaf/external/prediction_tracker/snapshots"
PT_MANIFEST_BLOB = "research/ncaaf/external/prediction_tracker/feeder_manifest.json"
PT_CURRENT_SEASON = 2026
PT_COMPLETED_SEASONS = {2022, 2023, 2024, 2025}
PT_EXPECTED_FILES = [
    "ncaa2022.csv", "ncaa2023.csv", "ncaa2024.csv", "ncaa2025.csv",
    "ncaa2026.csv", "ncaapredictions.csv", "predncaa_live_page.txt",
]
PT_CHALLENGE_MARKERS = (
    b"just a moment", b"cloudflare", b"cf-chl", b"challenge-platform",
    b"attention required", b"checking your browser", b"enable javascript and cookies",
)

st.set_page_config(page_title="NCAAF Weekly Stats Uploader", page_icon="⬆️", layout="wide")


def bq_client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT_ID)


def rows_as_dicts(query_job) -> List[dict]:
    return [dict(r.items()) for r in query_job.result()]


def ensure_alias_table(client: bigquery.Client) -> None:
    sql = f"""
    CREATE TABLE IF NOT EXISTS `{ALIAS_TABLE}` (
      Source_Name STRING,
      Source_Team_Name STRING,
      Source_Team_Name_Norm STRING,
      Team_Norm STRING,
      Notes STRING,
      Updated_At TIMESTAMP
    )
    CLUSTER BY Source_Name, Source_Team_Name_Norm
    """
    client.query(sql).result()


def load_alignment(client: bigquery.Client, season: int) -> Dict[str, dict]:
    sql = f"""
    WITH ranked AS (
      SELECT
        Team,
        LOWER(TRIM(Team_Norm)) AS Team_Norm,
        Canonical_Team_Name,
        LOWER(TRIM(Canonical_Team_Norm)) AS Canonical_Team_Norm,
        Conference,
        Division,
        Subdivision,
        League_Group,
        Effective_Season_From,
        ROW_NUMBER() OVER (
          PARTITION BY LOWER(TRIM(Team_Norm))
          ORDER BY Effective_Season_From DESC
        ) AS rn
      FROM `{ALIGNMENT_TABLE}`
      WHERE UPPER(TRIM(Sport)) = 'NCAAF'
        AND Effective_Season_From <= @season
        AND COALESCE(Use_For_Context, TRUE)
    )
    SELECT * EXCEPT(rn)
    FROM ranked
    WHERE rn = 1
    ORDER BY Team_Norm
    """
    cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("season", "INT64", int(season))]
    )
    rows = rows_as_dicts(client.query(sql, job_config=cfg))
    return {str(r["Team_Norm"]).strip().lower(): r for r in rows}


def load_known_aliases(client: bigquery.Client, season: int, valid_team_norms: set[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    try:
        ensure_alias_table(client)
        rows = rows_as_dicts(client.query(f"""
          SELECT Source_Team_Name, Team_Norm
          FROM `{ALIAS_TABLE}`
          WHERE Source_Name = '{SOURCE_NAME}'
          QUALIFY ROW_NUMBER() OVER (
            PARTITION BY Source_Team_Name_Norm
            ORDER BY Updated_At DESC
          ) = 1
        """))
        for r in rows:
            src = str(r.get("Source_Team_Name") or "").strip()
            norm = str(r.get("Team_Norm") or "").strip().lower()
            if src and norm in valid_team_norms:
                mapping[src] = norm
    except Exception:
        pass

    # Existing historical raw table is the strongest automatic alias dictionary.
    rows = rows_as_dicts(client.query(f"""
      SELECT Source_Team_Name, LOWER(TRIM(Team_Norm)) AS Team_Norm
      FROM `{RAW_TABLE}`
      WHERE Source_Name = '{SOURCE_NAME}'
        AND Source_Team_Name IS NOT NULL
        AND Team_Norm IS NOT NULL
      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY LOWER(TRIM(Source_Team_Name))
        ORDER BY Season DESC, Game_Date DESC
      ) = 1
    """))
    for r in rows:
        src = str(r.get("Source_Team_Name") or "").strip()
        norm = str(r.get("Team_Norm") or "").strip().lower()
        if src and norm in valid_team_norms and src not in mapping:
            mapping[src] = norm
    return mapping


def resolve_source_teams(source_teams: List[str], alignment: Mapping[str, dict], known: Mapping[str, str]):
    resolved: Dict[str, str] = {}
    unresolved: Dict[str, List[Tuple[str, float]]] = {}

    # Exact lookup strings from season-aware alignment.
    exact: Dict[str, str] = {}
    label_by_norm: Dict[str, str] = {}
    identity_strings_by_norm: Dict[str, set[str]] = {}
    for norm, row in alignment.items():
        label_by_norm[norm] = str(row.get("Team") or norm)
        identity_strings_by_norm[norm] = set()
        for v in (
            row.get("Team"), row.get("Team_Norm"),
            row.get("Canonical_Team_Name"), row.get("Canonical_Team_Norm"),
        ):
            k = normalize_match_text(v)
            if k:
                exact[k] = norm
                identity_strings_by_norm[norm].add(k)

    for source_team in source_teams:
        if source_team in known and known[source_team] in alignment:
            resolved[source_team] = known[source_team]
            continue

        k = normalize_match_text(source_team)
        if k in exact:
            resolved[source_team] = exact[k]
            continue

        # BigDataBall frequently sends the school name without the mascot.
        # Resolve only when the school-name relationship identifies ONE database
        # team.  Ambiguous names (for example Alabama vs Alabama A&M) are never
        # guessed.
        school_matches = set()
        if k:
            for norm, identities in identity_strings_by_norm.items():
                if any(
                    ident.startswith(k + " ") or k.startswith(ident + " ")
                    for ident in identities
                    if ident
                ):
                    school_matches.add(norm)
        if len(school_matches) == 1:
            resolved[source_team] = next(iter(school_matches))
            continue

        # High-confidence fuzzy match only. Anything else requires human choice.
        candidates = []
        for norm, label in label_by_norm.items():
            score = difflib.SequenceMatcher(None, k, normalize_match_text(label)).ratio()
            candidates.append((norm, score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        if candidates and candidates[0][1] >= 0.96 and (len(candidates) == 1 or candidates[0][1] - candidates[1][1] >= 0.05):
            resolved[source_team] = candidates[0][0]
        else:
            unresolved[source_team] = candidates[:12]
    return resolved, unresolved, label_by_norm


def save_aliases(client: bigquery.Client, selections: Mapping[str, str], notes: str) -> None:
    ensure_alias_table(client)
    for src, team_norm in selections.items():
        if not team_norm:
            continue
        sql = f"""
        MERGE `{ALIAS_TABLE}` T
        USING (
          SELECT
            @source_name AS Source_Name,
            @source_team AS Source_Team_Name,
            @source_team_norm AS Source_Team_Name_Norm,
            @team_norm AS Team_Norm,
            @notes AS Notes,
            CURRENT_TIMESTAMP() AS Updated_At
        ) S
        ON T.Source_Name = S.Source_Name
           AND T.Source_Team_Name_Norm = S.Source_Team_Name_Norm
        WHEN MATCHED THEN UPDATE SET
          Source_Team_Name = S.Source_Team_Name,
          Team_Norm = S.Team_Norm,
          Notes = S.Notes,
          Updated_At = S.Updated_At
        WHEN NOT MATCHED THEN INSERT
          (Source_Name, Source_Team_Name, Source_Team_Name_Norm, Team_Norm, Notes, Updated_At)
        VALUES
          (S.Source_Name, S.Source_Team_Name, S.Source_Team_Name_Norm, S.Team_Norm, S.Notes, S.Updated_At)
        """
        cfg = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("source_name", "STRING", SOURCE_NAME),
            bigquery.ScalarQueryParameter("source_team", "STRING", src),
            bigquery.ScalarQueryParameter("source_team_norm", "STRING", normalize_match_text(src)),
            bigquery.ScalarQueryParameter("team_norm", "STRING", team_norm),
            bigquery.ScalarQueryParameter("notes", "STRING", notes),
        ])
        client.query(sql, job_config=cfg).result()


def save_manual_aliases(client: bigquery.Client, selections: Mapping[str, str]) -> None:
    save_aliases(client, selections, "MANUAL_UI")


def merge_alignment_reference_row(client: bigquery.Client, row: Mapping[str, object]) -> int:
    """Append one verified alignment row if that effective row does not exist."""
    sql = f"""
    MERGE `{ALIGNMENT_TABLE}` T
    USING (
      SELECT
        'NCAAF' AS Sport,
        @team AS Team,
        @team_norm AS Team_Norm,
        @canonical_team_name AS Canonical_Team_Name,
        @canonical_team_norm AS Canonical_Team_Norm,
        @effective_season_from AS Effective_Season_From,
        @conference AS Conference,
        @division AS Division,
        @subdivision AS Subdivision,
        @league_group AS League_Group,
        'Team' AS Entity_Type,
        TRUE AS Active,
        TRUE AS Use_For_Context,
        @source_url AS Source_URL,
        @notes AS Notes
    ) S
      ON UPPER(TRIM(T.Sport)) = S.Sport
     AND LOWER(TRIM(T.Team_Norm)) = LOWER(TRIM(S.Team_Norm))
     AND T.Effective_Season_From = S.Effective_Season_From
    WHEN NOT MATCHED THEN INSERT
      (Sport, Team, Team_Norm, Canonical_Team_Name, Canonical_Team_Norm,
       Effective_Season_From, Conference, Division, Subdivision, League_Group,
       Entity_Type, Active, Use_For_Context, Source_URL, Notes)
    VALUES
      (S.Sport, S.Team, S.Team_Norm, S.Canonical_Team_Name, S.Canonical_Team_Norm,
       S.Effective_Season_From, S.Conference, S.Division, S.Subdivision, S.League_Group,
       S.Entity_Type, S.Active, S.Use_For_Context, S.Source_URL, S.Notes)
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("team", "STRING", row.get("team")),
        bigquery.ScalarQueryParameter("team_norm", "STRING", row.get("team_norm")),
        bigquery.ScalarQueryParameter("canonical_team_name", "STRING", row.get("canonical_team_name")),
        bigquery.ScalarQueryParameter("canonical_team_norm", "STRING", row.get("canonical_team_norm")),
        bigquery.ScalarQueryParameter("effective_season_from", "INT64", int(row.get("effective_season_from"))),
        bigquery.ScalarQueryParameter("conference", "STRING", row.get("conference")),
        bigquery.ScalarQueryParameter("division", "STRING", row.get("division")),
        bigquery.ScalarQueryParameter("subdivision", "STRING", row.get("subdivision")),
        bigquery.ScalarQueryParameter("league_group", "STRING", row.get("league_group")),
        bigquery.ScalarQueryParameter("source_url", "STRING", row.get("source_url")),
        bigquery.ScalarQueryParameter("notes", "STRING", row.get("notes")),
    ])
    job = client.query(sql, job_config=cfg)
    job.result()
    return int(job.num_dml_affected_rows or 0)


def bootstrap_verified_source_teams(client: bigquery.Client, source_teams: List[str], season: int) -> List[str]:
    """Auto-add verified team/alignment rows and BigDataBall aliases on first sight.

    This deliberately does NOT invent conference membership for unknown schools.
    Teams absent from this verified reference and absent from the alignment table
    still fall through to the manual review UI.
    """
    inserted_sources: List[str] = []
    aliases: Dict[str, str] = {}

    for src in source_teams:
        ref = NCAAF_TEAM_BOOTSTRAP_REFERENCE.get(normalize_match_text(src))
        if not ref:
            continue

        changed = 0
        for row in ref["rows"]:
            if int(row["effective_season_from"]) <= int(season):
                changed += merge_alignment_reference_row(client, row)

        aliases[src] = str(ref["team_norm"]).strip().lower()
        if changed:
            inserted_sources.append(src)

    if aliases:
        save_aliases(client, aliases, "AUTO_VERIFIED_NCAAF_REFERENCE")

    return inserted_sources


def dataframe_to_bq_csv_bytes(df: pd.DataFrame, schema=None) -> bytes:
    """Serialize a DataFrame for a BigQuery CSV load using the target schema.

    Pandas promotes integer columns containing NULLs to float, which would otherwise
    write values such as 4500.0 into INT64 fields. BigQuery's CSV parser does not
    accept that representation for INTEGER columns. When a schema is supplied we
    render typed values explicitly before writing the CSV.
    """
    out = df.copy()

    if schema is not None:
        for field in schema:
            name = field.name
            if name not in out.columns:
                continue

            field_type = str(field.field_type or "").upper()

            if field_type in {"INTEGER", "INT64"}:
                original = out[name]
                missing = original.isna() | original.astype(str).str.strip().isin({"", "nan", "NaN", "None", "<NA>"})
                numeric = pd.to_numeric(original.where(~missing), errors="coerce")

                bad_parse = (~missing) & numeric.isna()
                if bad_parse.any():
                    sample = original[bad_parse].astype(str).head(5).tolist()
                    raise ValueError(f"Column {name} contains non-integer values that cannot be loaded to INT64: {sample}")

                non_null = numeric.dropna()
                fractional = (non_null - non_null.round()).abs() > 1e-9
                if fractional.any():
                    sample = non_null[fractional].head(5).tolist()
                    raise ValueError(f"Column {name} contains fractional values but BigQuery expects INT64: {sample}")

                out[name] = numeric.map(lambda x: "" if pd.isna(x) else str(int(round(float(x)))))

            elif field_type in {"FLOAT", "FLOAT64"}:
                original = out[name]
                missing = original.isna() | original.astype(str).str.strip().isin({"", "nan", "NaN", "None", "<NA>"})
                numeric = pd.to_numeric(original.where(~missing), errors="coerce")
                bad_parse = (~missing) & numeric.isna()
                if bad_parse.any():
                    sample = original[bad_parse].astype(str).head(5).tolist()
                    raise ValueError(f"Column {name} contains non-numeric values that cannot be loaded to FLOAT64: {sample}")
                out[name] = numeric.map(lambda x: "" if pd.isna(x) else format(float(x), ".15g"))

            elif field_type in {"BOOLEAN", "BOOL"}:
                def _bool_csv(v):
                    if pd.isna(v) or str(v).strip() in {"", "nan", "NaN", "None", "<NA>"}:
                        return ""
                    if isinstance(v, bool):
                        return "true" if v else "false"
                    text = str(v).strip().lower()
                    if text in {"true", "t", "1", "yes", "y"}:
                        return "true"
                    if text in {"false", "f", "0", "no", "n"}:
                        return "false"
                    raise ValueError(f"Column {name} contains invalid BOOL value: {v!r}")
                out[name] = out[name].map(_bool_csv)

    # Empty strings are retained for strings; NaN/NA are emitted blank for typed BigQuery NULLs.
    buf = io.StringIO()
    out.to_csv(buf, index=False, na_rep="")
    return buf.getvalue().encode("utf-8")


def load_staging(client: bigquery.Client, df: pd.DataFrame) -> None:
    raw = client.get_table(RAW_TABLE)
    raw_cols = [f.name for f in raw.schema]
    if raw_cols != TARGET_COLUMNS:
        missing = [c for c in raw_cols if c not in TARGET_COLUMNS]
        extra = [c for c in TARGET_COLUMNS if c not in raw_cols]
        raise RuntimeError(
            "Raw-table schema no longer matches the uploader. "
            f"Missing in uploader={missing}; extra in uploader={extra}"
        )

    client.query(f"CREATE OR REPLACE TABLE `{STAGING_TABLE}` AS SELECT * FROM `{RAW_TABLE}` WHERE FALSE").result()
    csv_bytes = dataframe_to_bq_csv_bytes(df[TARGET_COLUMNS], raw.schema)
    fh = io.BytesIO(csv_bytes)
    cfg = bigquery.LoadJobConfig(
        schema=raw.schema,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        allow_quoted_newlines=True,
        field_delimiter=",",
    )
    client.load_table_from_file(fh, STAGING_TABLE, job_config=cfg).result()


def _row_change_predicate() -> str:
    """BigQuery predicate that is TRUE only when a stored row actually changed."""
    keys = {"Season", "Source_Name", "Source_Game_ID", "Team_Norm"}
    compare_cols = [c for c in TARGET_COLUMNS if c not in keys]
    return " OR\n           ".join(f"T.{c} IS DISTINCT FROM S.{c}" for c in compare_cols)


def validate_existing_game_identity(client: bigquery.Client) -> None:
    """Prevent a previously loaded Source_Game_ID from acquiring a second pair of team identities."""
    rows = rows_as_dicts(client.query(f"""
      WITH staging_games AS (
        SELECT
          Season, Source_Name, Source_Game_ID,
          ARRAY_AGG(DISTINCT Team_Norm ORDER BY Team_Norm) AS staging_teams,
          COUNT(*) AS staging_sides
        FROM `{STAGING_TABLE}`
        GROUP BY Season, Source_Name, Source_Game_ID
      ),
      raw_games AS (
        SELECT
          R.Season, R.Source_Name, R.Source_Game_ID,
          ARRAY_AGG(DISTINCT R.Team_Norm ORDER BY R.Team_Norm) AS raw_teams,
          COUNT(*) AS raw_sides
        FROM `{RAW_TABLE}` R
        JOIN (
          SELECT DISTINCT Season, Source_Name, Source_Game_ID
          FROM staging_games
        ) S
          USING (Season, Source_Name, Source_Game_ID)
        GROUP BY R.Season, R.Source_Name, R.Source_Game_ID
      )
      SELECT
        S.Season, S.Source_Name, S.Source_Game_ID,
        S.staging_teams, R.raw_teams, S.staging_sides, R.raw_sides
      FROM staging_games S
      JOIN raw_games R
        USING (Season, Source_Name, Source_Game_ID)
      WHERE S.staging_sides != 2
         OR R.raw_sides != 2
         OR TO_JSON_STRING(S.staging_teams) != TO_JSON_STRING(R.raw_teams)
      LIMIT 20
    """))
    if rows:
        sample = "; ".join(
            f"{r.get('Source_Game_ID')}: incoming={r.get('staging_teams')} stored={r.get('raw_teams')}"
            for r in rows[:10]
        )
        raise RuntimeError(
            "Existing game identity conflict detected. The uploader will not create a second "
            "representation of an already-loaded Source_Game_ID. " + sample
        )


def estimate_merge(client: bigquery.Client) -> dict:
    changed = _row_change_predicate()
    rows = rows_as_dicts(client.query(f"""
      WITH classified AS (
        SELECT
          S.Season, S.Source_Name, S.Source_Game_ID,
          CASE
            WHEN T.Source_Game_ID IS NULL THEN 'INSERT'
            WHEN ({changed}) THEN 'CHANGE'
            ELSE 'UNCHANGED'
          END AS action
        FROM `{STAGING_TABLE}` S
        LEFT JOIN `{RAW_TABLE}` T
          ON T.Season = S.Season
         AND T.Source_Name = S.Source_Name
         AND T.Source_Game_ID = S.Source_Game_ID
         AND T.Team_Norm = S.Team_Norm
      ), game_classified AS (
        SELECT
          Season, Source_Name, Source_Game_ID,
          CASE
            WHEN COUNTIF(action='INSERT') > 0 THEN 'INSERT'
            WHEN COUNTIF(action='CHANGE') > 0 THEN 'CHANGE'
            ELSE 'UNCHANGED'
          END AS game_action
        FROM classified
        GROUP BY Season, Source_Name, Source_Game_ID
      )
      SELECT
        COUNT(*) AS staging_rows,
        COUNTIF(action='INSERT') AS insert_rows,
        COUNTIF(action='CHANGE') AS changed_rows,
        COUNTIF(action='UNCHANGED') AS unchanged_rows,
        (SELECT COUNTIF(game_action='INSERT') FROM game_classified) AS insert_games,
        (SELECT COUNTIF(game_action='CHANGE') FROM game_classified) AS changed_games,
        (SELECT COUNTIF(game_action='UNCHANGED') FROM game_classified) AS unchanged_games
      FROM classified
    """))
    return rows[0] if rows else {}


def merge_staging_into_raw(client: bigquery.Client) -> None:
    keys = {"Season", "Source_Name", "Source_Game_ID", "Team_Norm"}
    update_cols = [c for c in TARGET_COLUMNS if c not in keys]
    update_sql = ",\n        ".join(f"{c} = S.{c}" for c in update_cols)
    insert_cols = ", ".join(TARGET_COLUMNS)
    insert_vals = ", ".join(f"S.{c}" for c in TARGET_COLUMNS)
    changed = _row_change_predicate()
    sql = f"""
    MERGE `{RAW_TABLE}` T
    USING `{STAGING_TABLE}` S
      ON T.Season = S.Season
     AND T.Source_Name = S.Source_Name
     AND T.Source_Game_ID = S.Source_Game_ID
     AND T.Team_Norm = S.Team_Norm
    WHEN MATCHED AND ({changed}) THEN UPDATE SET
        {update_sql}
    WHEN NOT MATCHED THEN INSERT ({insert_cols})
    VALUES ({insert_vals})
    """
    client.query(sql).result()


def rebuild_context(client: bigquery.Client) -> None:
    sql = REBUILD_SQL_PATH.read_text(encoding="utf-8")
    client.query(sql).result()


def post_upload_health(client: bigquery.Client, season: int) -> dict:
    """Validate raw/context/training parity, leakage, and duplicate-game integrity."""
    params = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("season", "INT64", int(season))
    ])
    result_rows = rows_as_dicts(client.query(f"""
      WITH raw_stats AS (
        SELECT
          COUNT(*) AS raw_rows,
          COUNT(DISTINCT Source_Game_ID) AS raw_games
        FROM `{RAW_TABLE}`
        WHERE Season = @season
      ),
      context_stats AS (
        SELECT
          COUNT(*) AS context_rows,
          COUNT(DISTINCT Source_Game_ID) AS context_games
        FROM `{CONTEXT_TABLE}`
        WHERE Season = @season
      ),
      training_stats AS (
        SELECT
          COUNT(*) AS training_rows,
          COUNT(DISTINCT Source_Game_ID) AS training_games,
          COUNTIF(Historical_Core_Eligible = 1) AS training_eligible
        FROM `{TRAIN_VIEW}`
        WHERE Season = @season
      ),
      leakage_stats AS (
        SELECT COUNT(*) AS first_game_leakage_rows
        FROM `{CONTEXT_TABLE}`
        WHERE Season = @season
          AND Team_Game_Number = 1
          AND (
            Prev_Points_For IS NOT NULL OR Prev_Points_Against IS NOT NULL OR
            Prev_Total_Yards IS NOT NULL OR Prev_Total_Plays IS NOT NULL OR
            WinPct_Prior_System IS NOT NULL
          )
      ),
      duplicate_key_stats AS (
        SELECT COUNT(*) AS duplicate_keys
        FROM (
          SELECT
            Source_Name, Source_Game_ID, Team_Norm, COUNT(*) AS duplicate_count
          FROM `{RAW_TABLE}`
          WHERE Season = @season
          GROUP BY Source_Name, Source_Game_ID, Team_Norm
          HAVING COUNT(*) > 1
        )
      ),
      bad_game_side_stats AS (
        SELECT COUNT(*) AS bad_game_side_counts
        FROM (
          SELECT
            Source_Name, Source_Game_ID, COUNT(*) AS side_count
          FROM `{RAW_TABLE}`
          WHERE Season = @season
          GROUP BY Source_Name, Source_Game_ID
          HAVING COUNT(*) != 2
        )
      )
      SELECT
        raw_stats.raw_rows,
        raw_stats.raw_games,
        context_stats.context_rows,
        context_stats.context_games,
        training_stats.training_rows,
        training_stats.training_games,
        training_stats.training_eligible,
        leakage_stats.first_game_leakage_rows,
        duplicate_key_stats.duplicate_keys,
        bad_game_side_stats.bad_game_side_counts
      FROM raw_stats
      CROSS JOIN context_stats
      CROSS JOIN training_stats
      CROSS JOIN leakage_stats
      CROSS JOIN duplicate_key_stats
      CROSS JOIN bad_game_side_stats
    """, job_config=params))
    return result_rows[0] if result_rows else {}


def weekly_results_tab():
    st.subheader("Weekly Results / Stats")
    st.caption(f"Build: {UPLOADER_BUILD}")
    st.caption("Drop in the BigDataBall team-stat Excel file. The tool formats it, prevents duplicate loads, updates BigQuery, rebuilds the leakage-safe context table, and validates the training view.")

    with st.expander("What this tool does", expanded=False):
        st.markdown(
            """
            1. Reads the BigDataBall NCAAF team-stat workbook.  
            2. Detects the season and game pairs.  
            3. Maps source team names to your season-aware BigQuery alignment.  
            4. Creates the exact `ncaaf_historical_game_side_raw` schema.  
            5. MERGEs rows, so uploading the same week twice is safe.  
            6. Rebuilds `ncaaf_historical_game_side_context`.  
            7. Checks the Historical Core training view and first-game leakage contract.
            """
        )

    file = st.file_uploader("Upload weekly NCAAF stats (.xlsx)", type=["xlsx"])
    if not file:
        st.info("Upload the weekly BigDataBall Excel file to begin.")
        return

    try:
        parsed = parse_bigdataball_workbook(file.getvalue())
    except Exception as e:
        st.error(f"Workbook validation failed: {e}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(parsed.rows):,}")
    c2.metric("Games", f"{parsed.game_count:,}")
    c3.metric("Detected season", parsed.season if parsed.season else "Unknown")
    c4.metric("Date range", f"{parsed.min_date or '?'} → {parsed.max_date or '?'}")

    default_season = parsed.season or pd.Timestamp.utcnow().year
    season = int(st.number_input("Season", min_value=2020, max_value=2100, value=int(default_season), step=1))

    try:
        client = bq_client()
        bootstrapped = bootstrap_verified_source_teams(client, parsed.source_teams, season)
        alignment = load_alignment(client, season)
    except Exception as e:
        st.error(f"Could not prepare/read BigQuery season alignment: {e}")
        return

    if bootstrapped:
        st.success(
            "Auto-added verified NCAAF alignment for: " + ", ".join(sorted(bootstrapped))
        )

    if not alignment:
        st.error(f"No NCAAF alignment rows resolve for season {season}. Load/update team_sport_alignment_history first.")
        return

    known = load_known_aliases(client, season, set(alignment))
    resolved, unresolved, labels = resolve_source_teams(parsed.source_teams, alignment, known)

    # Persist new deterministic mappings (exact/prefix/high-confidence) so the
    # next weekly upload does not need to rediscover the same source spelling.
    auto_aliases = {
        src: norm
        for src, norm in resolved.items()
        if src not in known and normalize_match_text(src) != normalize_match_text(norm)
    }
    if auto_aliases:
        save_aliases(client, auto_aliases, "AUTO_DETERMINISTIC_MATCH")

    if unresolved:
        st.warning(
            f"{len(unresolved)} team name(s) remain genuinely unknown after automatic discovery. "
            "Review these before loading; the uploader will not guess a new school's football conference."
        )
        selections = {}
        for src, cand in unresolved.items():
            options = [""] + [norm for norm, _ in cand]
            pretty = {"": "— choose team —"}
            for norm, score in cand:
                pretty[norm] = f"{labels.get(norm, norm)}  [{norm}]  ({score:.0%} match)"
            choice = st.selectbox(
                src,
                options=options,
                format_func=lambda x, p=pretty: p.get(x, x),
                key=f"map_{normalize_match_text(src)}",
            )
            selections[src] = choice
        if st.button("Save team mappings", type="primary"):
            missing = [src for src, val in selections.items() if not val]
            if missing:
                st.error("Choose a database team for every unresolved source name first.")
            else:
                save_manual_aliases(client, selections)
                st.success("Mappings saved.")
                st.rerun()
        return

    try:
        upload_df = build_upload_dataframe(parsed, season, resolved, alignment)
        report = validate_upload_dataframe(upload_df)
    except Exception as e:
        st.error(f"Conversion validation failed: {e}")
        return

    st.success("File passed conversion and validation.")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Upload rows", f"{report['rows']:,}")
    v2.metric("Games", f"{report['games']:,}")
    v3.metric("Duplicate keys", report["duplicate_key_rows"])
    v4.metric("Missing required rows", report["missing_required_rows"])

    with st.expander("Preview transformed rows"):
        st.dataframe(upload_df.head(30), use_container_width=True)

    csv_bytes = dataframe_to_bq_csv_bytes(upload_df)
    st.download_button(
        "Download transformed CSV (optional audit copy)",
        data=csv_bytes,
        file_name=f"ncaaf_{season}_weekly_historical_game_side_upload.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Upload to BigQuery")
    st.caption("Idempotent upload: new game/team rows are inserted, changed rows are corrected, and identical already-loaded rows are skipped. Existing Source_Game_ID team identities are protected from duplicate re-creation.")

    if st.button("Upload + rebuild context", type="primary"):
        prog = st.progress(0)
        status = st.empty()
        try:
            status.write("1/4 Loading validated rows to staging…")
            load_staging(client, upload_df)
            prog.progress(25)

            validate_existing_game_identity(client)
            merge_est = estimate_merge(client)
            insert_rows = int(merge_est.get("insert_rows", 0))
            changed_rows = int(merge_est.get("changed_rows", 0))
            unchanged_rows = int(merge_est.get("unchanged_rows", 0))
            insert_games = int(merge_est.get("insert_games", 0))
            changed_games = int(merge_est.get("changed_games", 0))
            unchanged_games = int(merge_est.get("unchanged_games", 0))
            status.write(
                f"2/4 Raw-table merge — {insert_games:,} new games / {insert_rows:,} rows; "
                f"{changed_games:,} corrected games / {changed_rows:,} rows; "
                f"{unchanged_games:,} already-loaded games / {unchanged_rows:,} rows skipped."
            )
            if insert_rows or changed_rows:
                merge_staging_into_raw(client)
            prog.progress(50)

            if insert_rows or changed_rows:
                status.write("3/4 Rebuilding leakage-safe historical context…")
                rebuild_context(client)
            else:
                status.write("3/4 No new or changed rows — context rebuild not needed.")
            prog.progress(80)

            status.write("4/4 Running post-upload health checks…")
            health = post_upload_health(client, season)
            prog.progress(100)

            ok = (
                int(health.get("raw_rows", -1)) == int(health.get("context_rows", -2)) == int(health.get("training_rows", -3))
                and int(health.get("training_rows", -1)) == int(health.get("training_eligible", -2))
                and int(health.get("first_game_leakage_rows", 1)) == 0
                and int(health.get("duplicate_keys", 1)) == 0
                and int(health.get("bad_game_side_counts", 1)) == 0
            )
            if ok:
                st.success("Upload complete. Raw history, context table, and training view all passed validation.")
            else:
                st.error("Upload finished, but a validation check failed. Do not retrain until this is reviewed.")
            st.json(health)
            st.info("Next: run the normal NCAAF model training. V12 reads the raw historical stats directly and the Historical Brain reads the rebuilt training view.")
        except Exception as e:
            st.error(f"Upload/rebuild failed: {e}")
            st.exception(e)



def storage_client() -> storage.Client:
    return storage.Client(project=PROJECT_ID)


def _pt_decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def validate_prediction_tracker_file(filename: str, raw: bytes) -> dict:
    """Validate exact manual Prediction Tracker source bytes before any GCS overwrite."""
    name = Path(filename).name
    lower = raw.lower()
    out = {
        "name": name,
        "status": "INVALID",
        "ok": False,
        "bytes": len(raw),
        "rows": None,
        "columns": None,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "season": None,
        "kind": None,
        "message": "",
    }

    if name not in PT_EXPECTED_FILES:
        out["message"] = "Unexpected filename. Use one of the exact Prediction Tracker filenames shown below."
        return out
    if any(marker in lower for marker in PT_CHALLENGE_MARKERS):
        out["message"] = "Rejected: file contains a Cloudflare/browser challenge payload."
        return out

    if name == "predncaa_live_page.txt":
        out["kind"] = "LIVE_HTML"
        if len(raw) < 1000:
            out["message"] = "Live page is too small to be a real Prediction Tracker page."
            return out
        text = _pt_decode(raw).lower()
        rating_hits = sum(token in text for token in ("espn", "fpi", "dokter", "keeper", "pigskin", "pi-ratings", "pi ratings"))
        if "home" not in text or rating_hits < 2:
            out["message"] = "Live page does not contain enough named Prediction Tracker rating content."
            return out
        out.update(ok=True, status="VALID", message="Named live page passed validation.")
        return out

    # CSV validation. Do not require cryptic rating column names; identity is verified downstream.
    out["kind"] = "LIVE_CSV" if name == "ncaapredictions.csv" else "ARCHIVE"
    m = re.fullmatch(r"ncaa(\d{4})\.csv", name)
    if m:
        out["season"] = int(m.group(1))
    if len(raw) < 1000:
        out["message"] = "CSV is too small to be a real Prediction Tracker file."
        return out
    try:
        rows = list(csv.reader(io.StringIO(_pt_decode(raw))))
    except Exception as exc:
        out["message"] = f"CSV parse failed: {type(exc).__name__}: {exc}"
        return out
    if not rows:
        out["message"] = "CSV is empty."
        return out
    header = [str(v).strip() for v in rows[0]]
    data_rows = [r for r in rows[1:] if any(str(v).strip() for v in r)]
    out["rows"] = len(data_rows)
    out["columns"] = len(header)
    h = " ".join(header).lower()
    if len(header) < 5 or "<html" in h or "<!doctype" in h:
        out["message"] = "CSV header is not plausible Prediction Tracker data."
        return out
    if "home" not in h or not any(x in h for x in ("road", "away", "visitor")):
        out["message"] = "CSV header is missing recognizable Home/Road team fields."
        return out
    minimum_rows = 20 if out["kind"] == "ARCHIVE" else 5
    if len(data_rows) < minimum_rows:
        out["message"] = f"Only {len(data_rows)} data rows found; expected at least {minimum_rows}."
        return out
    out.update(ok=True, status="VALID", message="CSV passed manual-ingestion validation.")
    return out


def _pt_existing_status(client: storage.Client) -> list[dict]:
    bucket = client.bucket(PT_BUCKET)
    now = dt.datetime.now(dt.timezone.utc)
    rows = []
    for name in PT_EXPECTED_FILES:
        path = f"{PT_RAW_PREFIX}/{name}"
        blob = bucket.blob(path)
        rec = {"File": name, "GCS status": "MISSING", "Bytes": None, "Updated UTC": None, "Age hours": None}
        try:
            if not blob.exists():
                rows.append(rec)
                continue
            blob.reload()
            raw = blob.download_as_bytes()
            diag = validate_prediction_tracker_file(name, raw)
            updated = blob.updated
            age = None
            if updated is not None:
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=dt.timezone.utc)
                age = max(0.0, (now - updated).total_seconds() / 3600.0)
            status = "READY" if diag["ok"] else "INVALID"
            if diag["ok"] and name in {"ncaa2026.csv", "ncaapredictions.csv", "predncaa_live_page.txt"} and age is not None and age > 24:
                status = "STALE (>24h)"
            rec.update({
                "GCS status": status,
                "Bytes": len(raw),
                "Updated UTC": updated.strftime("%Y-%m-%d %H:%M:%S") if updated else None,
                "Age hours": round(age, 1) if age is not None else None,
                "sha256": diag["sha256"],
                "valid": diag["ok"],
            })
        except Exception as exc:
            rec["GCS status"] = f"ERROR: {type(exc).__name__}"
            rec["error"] = str(exc)
        rows.append(rec)
    return rows


def _pt_load_manifest(bucket) -> dict:
    try:
        blob = bucket.blob(PT_MANIFEST_BLOB)
        if not blob.exists():
            return {}
        return json.loads(blob.download_as_text())
    except Exception:
        return {}


def _pt_write_manifest(bucket, source_records: dict, stamp: str) -> None:
    old = _pt_load_manifest(bucket)
    merged_sources = dict(old.get("sources") or {})
    merged_sources.update(source_records)
    failures = {k: v.get("error") for k, v in merged_sources.items() if v.get("status") == "FAILED" and v.get("error")}
    manifest = {
        "status": "PASS" if not failures else "PARTIAL",
        "source_tag": "prediction-tracker-manual-ui-v1-20261007",
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "network_role": "MANUAL_BROWSER_UPLOAD",
        "provider": "MANUAL_UI",
        "bucket": PT_BUCKET,
        "source_count": len(merged_sources),
        "refreshed_count": len(source_records),
        "pass_count": sum(1 for v in merged_sources.values() if v.get("status") in {"PASS", "SKIPPED_IDENTICAL", "SKIPPED_PROTECTED"}),
        "snapshot_prefix": f"gs://{PT_BUCKET}/{PT_SNAPSHOT_PREFIX}/{stamp}/",
        "sources": merged_sources,
        "failures": failures,
        "policy": {
            "manual_browser_upload": True,
            "preserve_exact_source_bytes": True,
            "reject_challenge_pages": True,
            "protect_completed_history": True,
            "model_side_contract": "GCS_FIRST",
            "production_authority": 0,
        },
    }
    bucket.blob(PT_MANIFEST_BLOB).upload_from_string(
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        content_type="application/json",
    )


def _pt_upload_validated(client: storage.Client, uploads: list[tuple[str, bytes, dict]], allow_history_replace: bool) -> list[dict]:
    bucket = client.bucket(PT_BUCKET)
    now = dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    results = []
    manifest_sources = {}

    for name, raw, diag in uploads:
        raw_path = f"{PT_RAW_PREFIX}/{name}"
        blob = bucket.blob(raw_path)
        existing_raw = None
        existing_valid = False
        existing_sha = None
        generation = 0
        if blob.exists():
            blob.reload()
            generation = int(blob.generation or 0)
            existing_raw = blob.download_as_bytes()
            existing_diag = validate_prediction_tracker_file(name, existing_raw)
            existing_valid = bool(existing_diag["ok"])
            existing_sha = existing_diag["sha256"]

        if existing_sha == diag["sha256"]:
            status = "SKIPPED_IDENTICAL"
            msg = "Already stored; bytes are identical."
        elif diag.get("season") in PT_COMPLETED_SEASONS and existing_valid and not allow_history_replace:
            status = "SKIPPED_PROTECTED"
            msg = "Completed historical season is already valid and protected from replacement."
        else:
            content_type = "text/html" if name == "predncaa_live_page.txt" else "text/csv"
            snapshot_path = f"{PT_SNAPSHOT_PREFIX}/{stamp}/{name}"
            bucket.blob(snapshot_path).upload_from_string(raw, content_type=content_type)
            # Generation match avoids silently overwriting a file changed by another uploader session.
            blob.upload_from_string(raw, content_type=content_type, if_generation_match=generation)
            blob.metadata = {
                "source": "manual-ui",
                "uploader_build": UPLOADER_BUILD,
                "sha256": diag["sha256"],
                "validated": "true",
            }
            blob.patch()
            status = "PASS"
            msg = "Validated file uploaded and immutable snapshot saved."

        source_key = name.rsplit(".", 1)[0]
        rec = {
            "name": source_key,
            "status": status,
            "provider": "MANUAL_UI",
            "bytes": len(raw),
            "sha256": diag["sha256"],
            "fetched_utc": now.isoformat(),
            "raw_blob": f"gs://{PT_BUCKET}/{raw_path}",
            "validation": {"rows": diag.get("rows"), "columns": diag.get("columns"), "kind": diag.get("kind")},
        }
        manifest_sources[source_key] = rec
        results.append({"File": name, "Result": status, "Message": msg, "Rows": diag.get("rows"), "Bytes": len(raw)})

    _pt_write_manifest(bucket, manifest_sources, stamp)
    return results


def prediction_tracker_tab():
    st.subheader("Prediction Tracker")
    st.caption(
        "Manual browser upload to the model's GCS-first Prediction Tracker cache. "
        "Files are validated before overwrite; Cloudflare challenge pages are rejected."
    )
    st.code(f"gs://{PT_BUCKET}/{PT_RAW_PREFIX}/", language=None)

    with st.expander("Expected filenames", expanded=False):
        st.write("Completed history: `ncaa2022.csv`, `ncaa2023.csv`, `ncaa2024.csv`, `ncaa2025.csv`")
        st.write("Current YTD: `ncaa2026.csv`")
        st.write("Current games: `ncaapredictions.csv`")
        st.write("Named current page: `predncaa_live_page.txt`")
        st.caption("The NCAAF research loader treats 2026/current files as fresh for 24 hours by default.")

    try:
        client = storage_client()
        existing = _pt_existing_status(client)
        st.markdown("#### Current GCS status")
        st.dataframe(pd.DataFrame(existing)[["File", "GCS status", "Bytes", "Updated UTC", "Age hours"]], use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Could not read Prediction Tracker GCS status: {type(exc).__name__}: {exc}")
        st.info("The uploader Cloud Run service account needs GCS object access to the sharp-models bucket.")
        return

    files = st.file_uploader(
        "Upload Prediction Tracker files",
        type=["csv", "txt"],
        accept_multiple_files=True,
        key="prediction_tracker_files",
    )
    if not files:
        st.info("Drop in one or more Prediction Tracker files. You can update only the files that changed.")
        return

    seen = set()
    prepared = []
    validation_rows = []
    for f in files:
        name = Path(f.name).name
        raw = f.getvalue()
        if name in seen:
            validation_rows.append({"File": name, "Status": "INVALID", "Rows": None, "Bytes": len(raw), "Message": "Duplicate filename in this upload batch."})
            continue
        seen.add(name)
        diag = validate_prediction_tracker_file(name, raw)
        validation_rows.append({"File": name, "Status": diag["status"], "Rows": diag.get("rows"), "Bytes": diag["bytes"], "Message": diag["message"]})
        if diag["ok"]:
            prepared.append((name, raw, diag))

    st.markdown("#### Validation")
    st.dataframe(pd.DataFrame(validation_rows), use_container_width=True, hide_index=True)
    invalid = [r for r in validation_rows if r["Status"] != "VALID"]
    if invalid:
        st.error("One or more files failed validation. Nothing will be uploaded until every selected file is valid.")
        return

    completed_names = {f"ncaa{y}.csv" for y in PT_COMPLETED_SEASONS}
    uploaded_completed = any(name in completed_names for name, _, _ in prepared)
    allow_history_replace = False
    if uploaded_completed:
        allow_history_replace = st.checkbox(
            "Allow replacement of an already-valid completed historical season if the uploaded bytes differ",
            value=False,
            help="Normally leave this off. Valid 2022-2025 files are treated as frozen history.",
        )

    if st.button("Upload validated Prediction Tracker files", type="primary", use_container_width=True):
        try:
            results = _pt_upload_validated(client, prepared, allow_history_replace)
            st.success("Prediction Tracker upload finished.")
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            st.info("Next: run NCAAF Research — Heavy Challenger Search. The model reads these GCS files directly.")
        except Exception as exc:
            st.error(f"Prediction Tracker upload failed: {type(exc).__name__}: {exc}")
            st.exception(exc)


def main():
    st.title("NCAAF Data Uploader")
    st.caption(f"Build: {UPLOADER_BUILD}")
    results_tab, prediction_tab = st.tabs(["Weekly Results / Stats", "Prediction Tracker"])
    with results_tab:
        weekly_results_tab()
    with prediction_tab:
        prediction_tracker_tab()


if __name__ == "__main__":
    main()
