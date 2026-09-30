"""NFL historical and incumbent-source audit V1. Research only; no model fitting/publication.

Sources are inspected rather than assumed. All BigQuery access is SELECT or metadata;
GCS incumbent blobs are inspected by metadata, never unpickled, copied or rewritten.
The expected next stage (after reviewing real coverage) is independent, prior-only,
season-forward Spread/H2H/Totals challenger validation.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

SOURCE_TAG = "nfl-audit-v1-read-only-dataset-incumbent-contract-20260930"
PROJECT = "sharplogger"
DATASET = "sharp_data"
SCORES = f"{PROJECT}.{DATASET}.game_scores_final"
FEATURES = f"{PROJECT}.{DATASET}.scores_with_features"
MARKET = f"{PROJECT}.{DATASET}.sharp_moves_master"
REQUIRED_SCORES = ("Sport", "Game_Start", "Home_Team", "Away_Team", "Score_Home_Score", "Score_Away_Score")


def _name(x):
    if x is None or pd.isna(x):
        return ""
    s = str(x).casefold().strip()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s))


def physical_game_key(row):
    """Outcome-, bookmaker- and market-independent physical game identity."""
    dt = pd.to_datetime(row.get("Game_Start"), utc=True, errors="coerce")
    home, away = _name(row.get("Home_Team")), _name(row.get("Away_Team"))
    if pd.isna(dt) or not home or not away or home == away:
        return None
    # Include exact kickoff timestamp; retain raw values for review if changes occur.
    key = f"NFL|{dt.isoformat()}|{home}|{away}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def audit_scores(rows: pd.DataFrame) -> dict:
    """Pure score-source check. No deduplicated, conflicting game can silently train."""
    missing = sorted(set(REQUIRED_SCORES) - set(rows.columns))
    if missing:
        return {"status": "HOLD", "reason": "MISSING_REQUIRED_SCORE_COLUMNS", "columns": missing}
    d = rows.copy()
    d = d[d["Sport"].astype(str).str.upper().eq("NFL")].copy()
    d["Game_Start"] = pd.to_datetime(d["Game_Start"], utc=True, errors="coerce")
    for c in ("Score_Home_Score", "Score_Away_Score"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["physical_game_id"] = d.apply(physical_game_key, axis=1)
    valid_identity = d["physical_game_id"].notna()
    invalid = int((~valid_identity).sum())
    d = d.loc[valid_identity].copy()
    finished = d["Score_Home_Score"].notna() & d["Score_Away_Score"].notna()
    d = d.loc[finished].copy()
    if d.empty:
        return {"status": "HOLD", "reason": "NO_FINAL_NFL_SCORES", "invalid_identity_rows": invalid, "source_rows": int(len(rows))}
    # Conflicting raw score rows for the same physical game invalidate readiness.
    paircount = d.groupby("physical_game_id", dropna=False)[["Score_Home_Score", "Score_Away_Score"]].nunique()
    bad_ids = paircount.index[paircount.gt(1).any(axis=1)].tolist()
    duplicates = int(len(d) - d["physical_game_id"].nunique())
    if bad_ids:
        return {"status": "HOLD", "reason": "CONFLICTING_FINAL_SCORES", "source_rows": int(len(rows)),
                "conflict_games": len(bad_ids), "conflict_sample": bad_ids[:5], "duplicate_source_rows": duplicates,
                "invalid_identity_rows": invalid}
    g = d.drop_duplicates("physical_game_id").copy()
    # Historical preseason matches must never be silently combined with the
    # regular/postseason population used for NFL model validation.
    stage_columns = [c for c in ("Is_Regular_Season", "Is_Postseason", "Is_Preseason") if c in g.columns]
    def true_count(col):
        if col not in g:
            return None
        v = g[col].astype(str).str.strip().str.lower()
        return int(v.isin({"1", "true", "t", "yes"}).sum())
    stage_counts = {c: true_count(c) for c in stage_columns}
    # Retain the whole score coverage inventory, but block readiness if stage
    # cannot be confirmed in source metadata.
    stage_verified = ("Is_Regular_Season" in stage_columns and "Is_Postseason" in stage_columns)
    authoritative_season = "Season" in g and pd.to_numeric(g["Season"], errors="coerce").notna().all()
    inferred = g["Game_Start"].dt.year - g["Game_Start"].dt.month.le(3).astype(int)
    g["audit_season"] = pd.to_numeric(g["Season"], errors="coerce").astype(int) if authoritative_season else inferred
    season_counts = {str(int(k)): int(v) for k,v in sorted(g.groupby("audit_season").size().items())}
    train_years = [y for y,n in season_counts.items() if int(y) <= 2025 and n >= 200]
    confirmation = season_counts.get("2026", 0)
    # A score-source audit does not certify as-of feature availability or champion skill.
    ready = bool(len(train_years) >= 3 and confirmation >= 20 and invalid == 0 and stage_verified)
    g["margin"] = g["Score_Home_Score"] - g["Score_Away_Score"]
    return {
        "status": "READY_FOR_FEATURE_AND_ASOF_AUDIT" if ready else "HOLD",
        "reason": "COVERAGE_ONLY_NOT_MODEL_APPROVAL" if ready else "INSUFFICIENT_SEASON_OR_STAGE_COVERAGE",
        "source_rows": int(len(rows)), "physical_games": int(len(g)), "duplicate_source_rows": duplicates,
        "invalid_identity_rows": invalid, "conflict_games": 0,
        "season_source": "AUTHORITATIVE_SEASON" if authoritative_season else "INFERRED_JAN_MAR_PREVIOUS_YEAR",
        "games_per_season": season_counts, "stage_columns_present":stage_columns,
        "stage_counts":stage_counts, "stage_filter_verified":stage_verified,
        "note":"Season counts above describe all scored games, not an approved regular/postseason training universe.",
        "full_prior_seasons_200plus": train_years, "confirm_2026_games": confirmation,
        "ties": int((g["margin"] == 0).sum()),
        "first_kickoff_utc": g["Game_Start"].min().isoformat(),
        "last_kickoff_utc": g["Game_Start"].max().isoformat(),
        "scored_games_no_systems_or_odds_implied": True,
    }


def _schema(client, table):
    try:
        obj = client.get_table(table)
    except Exception as exc:
        return {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}", "columns": []}
    cols = [f.name for f in obj.schema]
    return {"status": "AVAILABLE", "columns": cols, "rows_metadata": getattr(obj,"num_rows",None),
            "type": str(getattr(obj,"table_type", "UNKNOWN"))}


def _query(client, sql, *parameters):
    # Import only in runtime, so pure score tests need no GCP installation/credentials.
    from google.cloud import bigquery
    return client.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=list(parameters), use_query_cache=True,
        maximum_bytes_billed=20 * 1024**3,
    )).to_dataframe()


def _param_nfl():
    from google.cloud import bigquery
    return bigquery.ScalarQueryParameter("sport", "STRING", "NFL")


def _summarize_feature_source(client, schema):
    cols = set(schema.get("columns") or [])
    if schema.get("status") != "AVAILABLE" or "Sport" not in cols:
        return {"status": "HOLD", "reason": "SOURCE_OR_SPORT_COLUMN_UNAVAILABLE"}
    # Inspect only schema and small aggregated counts; no label/feature assumptions.
    ts = next((c for c in ("feat_Game_Start", "Game_Start", "Commence_Hour", "Snapshot_Timestamp") if c in cols), None)
    count_sql = f"SELECT COUNT(*) AS rows FROM `{FEATURES}` WHERE UPPER(CAST(`Sport` AS STRING)) = @sport"
    n = int(_query(client, count_sql, _param_nfl()).iloc[0]["rows"])
    if not ts:
        return {"status": "HOLD", "reason": "TIMESTAMP_COLUMN_MISSING", "rows": n}
    year_sql = f"""SELECT EXTRACT(YEAR FROM DATE(SAFE_CAST(`{ts}` AS TIMESTAMP))) AS calendar_year,
          COUNT(*) AS source_rows
          FROM `{FEATURES}` WHERE UPPER(CAST(`Sport` AS STRING)) = @sport
          GROUP BY 1 ORDER BY 1"""
    yearly = _query(client, year_sql, _param_nfl())
    excluded_names = ("hit", "score", "actual", "outcome", "result", "grade", "target", "cover", "final", "payout", "profit", "roi")
    candidate_like = [c for c in cols if c.startswith(("feat_", "HC_", "Core_", "Market_", "Team_", "Opp_")) and not any(w in c.lower() for w in excluded_names)]
    return {"status": "INVENTORY_ONLY", "rows": n, "event_timestamp": ts,
            "calendar_year_rows": {str(int(row.calendar_year)) if pd.notna(row.calendar_year) else "UNKNOWN": int(row.source_rows)
                                  for row in yearly.itertuples(index=False)},
            "candidate_named_columns": len(candidate_like), "candidate_sample": sorted(candidate_like)[:25],
            "note": "Column names do not establish feature leakage-safety; inspect contracts before training."}


def _summarize_market_source(client, schema):
    cols = set(schema.get("columns") or [])
    if schema.get("status") != "AVAILABLE" or "Sport" not in cols:
        return {"status":"HOLD", "reason":"SOURCE_OR_SPORT_COLUMN_UNAVAILABLE"}
    required = ("Game_Start", "Snapshot_Timestamp", "Market")
    missing = [c for c in required if c not in cols]
    if missing:
        return {"status":"HOLD", "reason":"MISSING_ASOF_COLUMNS", "missing":missing}
    sql = f"""SELECT UPPER(CAST(`Market` AS STRING)) AS market,
       COUNT(*) AS source_rows,
       COUNTIF(SAFE_CAST(`Snapshot_Timestamp` AS TIMESTAMP) < SAFE_CAST(`Game_Start` AS TIMESTAMP)) AS pregame_rows,
       COUNTIF(SAFE_CAST(`Snapshot_Timestamp` AS TIMESTAMP) >= SAFE_CAST(`Game_Start` AS TIMESTAMP)) AS at_or_after_kickoff_rows,
       MIN(SAFE_CAST(`Snapshot_Timestamp` AS TIMESTAMP)) AS earliest_snapshot,
       MAX(SAFE_CAST(`Snapshot_Timestamp` AS TIMESTAMP)) AS latest_snapshot
       FROM `{MARKET}` WHERE UPPER(CAST(`Sport` AS STRING))=@sport
       GROUP BY 1 ORDER BY 1"""
    d = _query(client, sql, _param_nfl())
    out=[]
    for r in d.to_dict("records"):
        out.append({k:(v.isoformat() if hasattr(v,"isoformat") else (int(v) if isinstance(v,(int,)) else v)) for k,v in r.items()})
    return {"status":"INVENTORY_ONLY", "markets":out,
            "has_odds_columns": all(c in cols for c in ("Odds_Price", "Bookmaker", "Outcome", "Value")),
            "warning":"Snapshot rows are not physically deduplicated matched betting quotes; no ROI/CLV inferred."}


def _discover_nfl_tables(client):
    """List NFL-specific uploader/view candidates, without assuming their names."""
    sql=(f"SELECT table_name, table_type FROM `{PROJECT}.{DATASET}.INFORMATION_SCHEMA.TABLES` "
         "WHERE REGEXP_CONTAINS(LOWER(table_name), r'nfl|football') "
         "ORDER BY table_name LIMIT 100")
    d=_query(client,sql)
    return [{"name":str(r.table_name),"type":str(r.table_type)} for r in d.itertuples(index=False)]


def _artifact_inventory(storage_client, bucket_name):
    bucket=storage_client.bucket(bucket_name)
    output=[]
    for mkt, aliases in (("spreads",("spreads",)),("h2h",("h2h","moneyline","ml","headtohead")),("totals",("totals",))):
        found=[]
        for prefix in ("", "models/"):
            for alias in aliases:
                path=f"{prefix}sharp_win_model_nfl_{alias}.pkl"
                blob=bucket.blob(path)
                if blob.exists():
                    blob.reload()
                    found.append({"key":path,"size_bytes":blob.size,"updated_utc":blob.updated.isoformat() if blob.updated else None,
                                  "generation":str(blob.generation),"md5_hash":blob.md5_hash})
        output.append({"market":mkt,"found":found,"status":"PRESENT_METADATA_ONLY" if found else "MISSING",
                       "not_verified":"Predictor performance, calibration, training provenance, OOF and feature contract require further validation."})
    return output


def run_nfl_audit_v1(*, bq_client=None, storage_client=None, bucket_name="sharp-models", log_func=print, hard_fail=True):
    """Run read-only NFL coverage preflight; it NEVER fits or publishes an NFL model."""
    from google.cloud import bigquery, storage
    bq=bq_client or bigquery.Client(project=PROJECT)
    gcs=storage_client or storage.Client()
    out={"source_tag": SOURCE_TAG, "run_utc":datetime.now(timezone.utc).isoformat(),"sport":"NFL",
         "scope":"READ_ONLY_PREFLIGHT", "model_publication":False,"production_authority":0}
    log_func(f"[NFL-AUDIT-V1-PREFLIGHT] status=START tag={SOURCE_TAG} sport=NFL publication=FALSE authority=0")
    schemas={key:_schema(bq,key) for key in (SCORES,FEATURES,MARKET)}
    out["sources"]={name:{"status":v["status"],"type":v.get("type"),"row_metadata":v.get("rows_metadata"),
                            "column_count":len(v.get("columns") or []),"error":v.get("error")}
                    for name,v in schemas.items()}
    score_cols=set(schemas[SCORES].get("columns") or [])
    if not set(REQUIRED_SCORES).issubset(score_cols):
        score_report={"status":"HOLD", "reason":"GAME_SCORES_FINAL_SCHEMA_MISSING", "missing":sorted(set(REQUIRED_SCORES)-score_cols)}
    else:
        extras=[c for c in ("Season","Is_Regular_Season","Is_Preseason","Is_Postseason", "Merge_Key_Short") if c in score_cols]
        projection=", ".join(f"`{c}`" for c in REQUIRED_SCORES+tuple(extras))
        sql=f"SELECT {projection} FROM `{SCORES}` WHERE UPPER(CAST(`Sport` AS STRING)) = @sport"
        score_report=audit_scores(_query(bq,sql,_param_nfl()))
    out["scores"]=score_report
    log_func(f"[NFL-AUDIT-V1-SCORES] {json.dumps(score_report,default=str,sort_keys=True)}")
    try:
        out["training_source"]=_summarize_feature_source(bq, schemas[FEATURES])
    except Exception as e:
        out["training_source"]={"status":"HOLD", "reason":"TRAINING_SOURCE_QUERY_FAILED", "error":f"{type(e).__name__}: {e}"}
    log_func(f"[NFL-AUDIT-V1-FEATURES] {json.dumps(out['training_source'],default=str,sort_keys=True)}")
    try:
        out["market_source"]=_summarize_market_source(bq,schemas[MARKET])
    except Exception as e:
        out["market_source"]={"status":"HOLD", "reason":"MARKET_SOURCE_QUERY_FAILED", "error":f"{type(e).__name__}: {e}"}
    log_func(f"[NFL-AUDIT-V1-MARKET] {json.dumps(out['market_source'],default=str,sort_keys=True)}")
    try:
        out["nfl_table_catalog"]=_discover_nfl_tables(bq)
    except Exception as e:
        out["nfl_table_catalog"]={"status":"UNAVAILABLE", "error":f"{type(e).__name__}: {e}"}
    log_func(f"[NFL-AUDIT-V1-CATALOG] {json.dumps(out['nfl_table_catalog'],default=str,sort_keys=True)}")
    try:
        out["incumbents"]=_artifact_inventory(gcs,bucket_name)
    except Exception as e:
        out["incumbents"]={"status":"HOLD", "reason":"ARTIFACT_METADATA_FAILED", "error":f"{type(e).__name__}: {e}"}
    log_func(f"[NFL-AUDIT-V1-INCUMBENTS] {json.dumps(out['incumbents'],default=str,sort_keys=True)}")
    ready=(score_report.get("status")=="READY_FOR_FEATURE_AND_ASOF_AUDIT"
           and out["training_source"].get("status")=="INVENTORY_ONLY"
           and out["market_source"].get("status")=="INVENTORY_ONLY"
           and isinstance(out["incumbents"],list)
           and all(x["status"]=="PRESENT_METADATA_ONLY" for x in out["incumbents"]))
    out["status"]="READY_FOR_LEAKAGE_AND_CHALLENGER_VALIDATION" if ready else "HOLD_REVIEW_COVERAGE"
    out["next_step"]=("Verify point-in-time feature derivation, retrieve true incumbent OOF/prospective predictions, "
                      "then run separate season-forward SPREAD/H2H/TOTALS challengers. No automatic promotion.")
    log_func(f"[NFL-AUDIT-V1-CONTRACT] status={out['status']} season_source={score_report.get('season_source','UNKNOWN')} "
             f"publication=FALSE production_authority=0 legacy_nfl=UNCHANGED ncaaf=UNCHANGED "
             f"data_coverage_only=TRUE champion_win_claim=FALSE")
    if hard_fail and score_report.get("status")=="HOLD" and score_report.get("reason") in {
            "MISSING_REQUIRED_SCORE_COLUMNS","GAME_SCORES_FINAL_SCHEMA_MISSING","CONFLICTING_FINAL_SCORES","NO_FINAL_NFL_SCORES"}:
        raise RuntimeError(f"[NFL-AUDIT-V1-CONTRACT] HARD_HOLD reason={score_report.get('reason')}; no model mutated")
    return out


if __name__ == "__main__":
    print(json.dumps(run_nfl_audit_v1(), indent=2,default=str))
