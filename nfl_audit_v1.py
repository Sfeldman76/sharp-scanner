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

SOURCE_TAG = "nfl-audit-v1.3-prior-feature-provenance-20260930"
PROJECT = "sharplogger"
DATASET = "sharp_data"
SCORES = f"{PROJECT}.{DATASET}.game_scores_final"
FEATURES = f"{PROJECT}.{DATASET}.scores_with_features"
MARKET = f"{PROJECT}.{DATASET}.sharp_moves_master"
NFL_RAW = f"{PROJECT}.{DATASET}.nfl_historical_game_side_raw"
NFL_CONTEXT = f"{PROJECT}.{DATASET}.nfl_historical_game_side_context"
NFL_VIEW = f"{PROJECT}.{DATASET}.nfl_historical_core_training_vw"
HIST_REQUIRED = ("Sport", "Season", "Season_Stage", "Source_Name", "Source_Game_ID", "Game_Date", "Week", "Team_Norm", "Team_Score", "Opponent_Score", "Is_Home", "Is_Away", "Is_Neutral")
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


def audit_uploaded_games(game_rows: pd.DataFrame) -> dict:
    """Read-only two-side game validation using the uploader's calendar and venue contract.

    Regular-season games may extend into January after their named season.
    Neutral-site games have TWO neutral sides rather than one home and one away.
    All other date, venue, grain and final-score anomalies remain hard holds.
    """
    if game_rows.empty:
        return {"status": "HOLD", "reason": "HISTORICAL_RAW_EMPTY", "physical_games": 0}
    required={"Season", "Source_Name", "Source_Game_ID", "side_rows", "distinct_teams",
              "home_rows", "away_rows", "neutral_rows", "bad_venue_flag_rows", "stage_variants",
              "stage", "date_variants", "game_date", "missing_scores", "min_team_score",
              "max_team_score", "min_opponent_score", "max_opponent_score"}
    missing=sorted(required - set(game_rows.columns))
    if missing:
        return {"status":"HOLD", "reason":"AUDIT_AGGREGATE_COLUMNS_MISSING", "missing":missing,
                "physical_games":int(len(game_rows))}
    d=game_rows.copy()
    for col in ("side_rows", "distinct_teams", "home_rows", "away_rows", "neutral_rows",
                "bad_venue_flag_rows", "stage_variants", "date_variants", "missing_scores",
                "min_team_score", "max_team_score", "min_opponent_score", "max_opponent_score"):
        d[col]=pd.to_numeric(d[col],errors="coerce")
    side_ok=d["side_rows"].eq(2) & d["distinct_teams"].eq(2)
    home_away=(d["home_rows"].eq(1) & d["away_rows"].eq(1) & d["neutral_rows"].eq(0))
    neutral=(d["home_rows"].eq(0) & d["away_rows"].eq(0) & d["neutral_rows"].eq(2))
    venue_ok=(home_away | neutral) & d["bad_venue_flag_rows"].eq(0)
    # Compare reciprocal score multisets, not home/away scores: neutral games
    # must be checked too. With exactly two side rows, this validates both
    # sides' final points even if neither has a designated Is_Home flag.
    reciprocal=(d["min_team_score"].eq(d["min_opponent_score"]) &
                d["max_team_score"].eq(d["max_opponent_score"]))
    valid_stage=d["stage"].isin({"REGULAR", "POSTSEASON", "PRESEASON"})
    season=pd.to_numeric(d["Season"],errors="coerce")
    dt=pd.to_datetime(d["game_date"],errors="coerce")
    year,month=dt.dt.year,dt.dt.month
    regular_valid=(d["stage"].eq("REGULAR") &
                   ((year.eq(season) & month.ge(8)) |
                    (year.eq(season+1) & month.eq(1))))
    post_valid=(d["stage"].eq("POSTSEASON") & year.eq(season+1) & month.isin((1,2)))
    pre_valid=(d["stage"].eq("PRESEASON") & year.eq(season) & month.between(7,9))
    date_ok=regular_valid | post_valid | pre_valid
    bad_date=(~date_ok) | dt.isna() | season.isna()
    errors={
        "bad_side_counts":int((~d["side_rows"].eq(2)).sum()),
        "bad_distinct_teams":int((~d["distinct_teams"].eq(2)).sum()),
        "bad_home_away_counts":int((~(home_away | neutral)).sum()),
        "bad_row_venue_flags":int(d["bad_venue_flag_rows"].fillna(1).gt(0).sum()),
        "stage_conflicts":int((~d["stage_variants"].eq(1)).sum()),
        "date_conflicts":int((~d["date_variants"].eq(1)).sum()),
        "missing_scores":int(d["missing_scores"].fillna(1).gt(0).sum()),
        "nonreciprocal_scores":int((~reciprocal).sum()),
        "invalid_stage":int((~valid_stage).sum()),
        "invalid_season":int(season.isna().sum()),
        "season_date_mismatch":int(bad_date.sum()),
    }
    bad=(~side_ok) | (~venue_ok) | (~d["stage_variants"].eq(1)) | (~d["date_variants"].eq(1)) | \
        d["missing_scores"].fillna(1).gt(0) | (~reciprocal) | (~valid_stage) | bad_date
    anomaly_samples=[]
    for i in d.index[bad][:10]:
        row=d.loc[i]
        anomaly_samples.append({"Season":None if pd.isna(season.loc[i]) else int(season.loc[i]),
          "Source_Game_ID":str(row["Source_Game_ID"]), "stage":str(row["stage"]),
          "game_date":None if pd.isna(dt.loc[i]) else dt.loc[i].date().isoformat(),
          "side_rows":int(row["side_rows"]) if pd.notna(row["side_rows"]) else None,
          "home_rows":int(row["home_rows"]) if pd.notna(row["home_rows"]) else None,
          "away_rows":int(row["away_rows"]) if pd.notna(row["away_rows"]) else None,
          "neutral_rows":int(row["neutral_rows"]) if pd.notna(row["neutral_rows"]) else None,
          "date_valid":bool(date_ok.loc[i]), "venue_valid":bool(venue_ok.loc[i]),
          "scores_reciprocal":bool(reciprocal.loc[i])})
    counts=d.groupby(["Season","stage"],dropna=False).size()
    by_season={}
    for (y,stage),num in counts.items():
        yr=str(int(y)) if pd.notna(y) else "UNKNOWN"
        by_season.setdefault(yr,{"REGULAR":0,"POSTSEASON":0,"PRESEASON":0})[str(stage)]=int(num)
    eligible={y:v["REGULAR"]+v["POSTSEASON"] for y,v in by_season.items()}
    full_prior=sorted(int(y) for y,n in eligible.items() if y.isdigit() and int(y)<=2025 and n>=200)
    hard_bad=any(errors.values())
    ready=not hard_bad and len(full_prior)>=3
    status="HISTORY_COVERAGE_PASS" if ready else "HOLD"
    reason=("AUTHORITATIVE_UPLOADER_HISTORY" if ready else
            "BAD_RAW_GAME_GRAIN_OR_SEASON_STAGE" if hard_bad else "FEWER_THAN_THREE_FULL_PRIOR_SEASONS")
    return {"status":status,"reason":reason,"physical_games":len(d),
            "seasons_from_source":"AUTHORITATIVE_UPLOADER_SEASON", "games_by_season_stage":by_season,
            "training_eligible_games_by_season":eligible,"full_prior_seasons_200plus":full_prior,
            "errors":errors,"anomaly_samples":anomaly_samples,
            "venue_inventory":{"neutral_site_games":int(neutral.sum()),
                               "home_away_games":int(home_away.sum())},
            "calendar_inventory":{"regular_games_next_january":int((d["stage"].eq("REGULAR") &
                            year.eq(season+1) & month.eq(1)).sum())},
            "first_game_date":dt.min().date().isoformat() if dt.notna().any() else None,
            "last_game_date":dt.max().date().isoformat() if dt.notna().any() else None,
            "note":"Uploader Season is authoritative. Neutral-site pairs and regular-season games in the next January are valid. Historical closing odds remain labels, never assumed known earlier."}


def _audit_nfl_uploader(client, raw_schema, context_schema, view_schema):
    raw_cols=set(raw_schema.get('columns') or [])
    missing=sorted(set(HIST_REQUIRED)-raw_cols)
    if raw_schema.get('status')!='AVAILABLE' or missing:
        return {"status":"HOLD", "reason":"HISTORICAL_RAW_SCHEMA_MISSING", "missing":missing,
                "raw_schema_status":raw_schema.get('status')}
    query=f"""SELECT Season, Source_Name, Source_Game_ID,
      COUNT(*) AS side_rows, COUNT(DISTINCT Team_Norm) AS distinct_teams,
      COUNTIF(Is_Home=1) AS home_rows, COUNTIF(Is_Away=1) AS away_rows,
      COUNTIF(Is_Neutral=1) AS neutral_rows,
      COUNTIF(Is_Home IS NULL OR Is_Away IS NULL OR Is_Neutral IS NULL OR
              COALESCE(Is_Home,0)+COALESCE(Is_Away,0)+COALESCE(Is_Neutral,0) != 1) AS bad_venue_flag_rows,
      COUNT(DISTINCT Season_Stage) AS stage_variants, ANY_VALUE(Season_Stage) AS stage,
      COUNT(DISTINCT Game_Date) AS date_variants, MIN(Game_Date) AS game_date,
      COUNTIF(Team_Score IS NULL OR Opponent_Score IS NULL) AS missing_scores,
      MIN(Team_Score) AS min_team_score, MAX(Team_Score) AS max_team_score,
      MIN(Opponent_Score) AS min_opponent_score, MAX(Opponent_Score) AS max_opponent_score
      FROM `{NFL_RAW}` WHERE UPPER(CAST(Sport AS STRING))=@sport
      GROUP BY Season,Source_Name,Source_Game_ID ORDER BY Season,Source_Name,Source_Game_ID"""
    result=audit_uploaded_games(_query(client,query,_param_nfl()))
    result['source_table']=NFL_RAW
    # The view is the separately built prior-game feature source, not the legacy
    # live score/feature tables. Check its grain and eligibility without
    # claiming that its feature expressions are leakage safe.
    view_cols=set(view_schema.get('columns') or [])
    req={'Season','Source_Game_ID','Historical_Core_Eligible'}
    view_report={"status":"HOLD", "reason":"HISTORICAL_TRAIN_VIEW_SCHEMA_MISSING",
                 "missing":sorted(req-view_cols)}
    if view_schema.get('status')=='AVAILABLE' and not(req-view_cols):
        view_sql=f"""SELECT Season, COUNT(*) AS source_rows,
          COUNT(DISTINCT CAST(Source_Game_ID AS STRING)) AS physical_games,
          COUNTIF(Historical_Core_Eligible=1) AS training_eligible_rows
          FROM `{NFL_VIEW}` GROUP BY Season ORDER BY Season"""
        v=_query(client,view_sql)
        per={str(int(r.Season)): {"source_rows":int(r.source_rows),"physical_games":int(r.physical_games),
             "training_eligible_rows":int(r.training_eligible_rows)} for r in v.itertuples(index=False)}
        mismatches={}
        for year,games in result.get('training_eligible_games_by_season',{}).items():
            p=per.get(year)
            if p is None or p['physical_games']!=sum(result['games_by_season_stage'][year].values()) or p['source_rows']!=2*p['physical_games'] or p['training_eligible_rows']!=2*games:
                mismatches[year]={"raw_eligible_games":games,"view":p}
        view_report={"status":"GRAIN_AND_ELIGIBILITY_PASS" if per and not mismatches else "HOLD",
                     "reason":"COUNTS_MATCH_UPLOADER" if per and not mismatches else "RAW_VIEW_GRAIN_MISMATCH",
                     "source_table":NFL_VIEW,"by_season":per,"mismatches":mismatches,
                     "note":"Feature column availability and prior-only expression safety still need inspection."}
    result['training_view']=view_report
    ccols=set(context_schema.get('columns') or [])
    prior=("Prev_Points_For", "Prev_Points_Against", "Prev_Total_Yards", "Prev_Total_Plays",
           "Prev_Off_Yards_Per_Play","Prev_Turnover_Margin","WinPct_Prior_System","ATS_WinPct_Prior_System")
    creq={'Team_Game_Number','Season','Source_Game_ID',*prior}
    c_report={"status":"HOLD","reason":"CONTEXT_SCHEMA_MISSING","missing":sorted(creq-ccols)}
    if context_schema.get('status')=='AVAILABLE' and not(creq-ccols):
        violation=' OR '.join(f'`{col}` IS NOT NULL' for col in prior)
        cq=f"""SELECT COUNT(*) AS context_rows,
          COUNTIF(Team_Game_Number=1) AS first_game_rows,
          COUNTIF(Team_Game_Number=1 AND ({violation})) AS first_game_leakage_rows
          FROM `{NFL_CONTEXT}`"""
        r=_query(client,cq).iloc[0]
        count=int(r['first_game_leakage_rows'])
        c_report={"status":"FIRST_GAME_CHECK_PASS" if count==0 else "HOLD",
                  "reason":"FIRST_GAME_PRIOR_ONLY" if count==0 else "FIRST_GAME_LEAKAGE",
                  "context_rows":int(r['context_rows']),"first_game_rows":int(r['first_game_rows']),
                  "first_game_leakage_rows":count,
                  "note":"First-game null guard is necessary but not full prior-only or as-of proof."}
    result['context']=c_report
    return result


def _summarize_feature_source(client, schema):
    cols = set(schema.get("columns") or [])
    if schema.get("status") != "AVAILABLE" or "Sport" not in cols:
        return {"status": "HOLD", "reason": "SOURCE_OR_SPORT_COLUMN_UNAVAILABLE"}
    # Inspect only schema and small aggregated counts; no label/feature assumptions.
    ts = next((c for c in ("feat_Game_Start", "Game_Start", "Commence_Hour", "Snapshot_Timestamp") if c in cols), None)
    count_sql = f"SELECT COUNT(*) AS row_count FROM `{FEATURES}` WHERE UPPER(CAST(`Sport` AS STRING)) = @sport"
    n = int(_query(client, count_sql, _param_nfl()).iloc[0]["row_count"])
    if not ts or n==0:
        return {"status": "HOLD", "reason": "TIMESTAMP_COLUMN_MISSING" if not ts else "NO_NFL_ROWS_IN_LEGACY_FEATURES",
                "rows": n, "note":"This legacy generic table is NOT the NFL historical core training view."}
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
            "note": "Legacy generic feature inventory only. NFL uploader view is audited separately; column names cannot prove leakage safety."}


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
    schemas={key:_schema(bq,key) for key in (SCORES,FEATURES,MARKET,NFL_RAW,NFL_CONTEXT,NFL_VIEW)}
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
    log_func(f"[NFL-AUDIT-V1-LIVE-SCORES] {json.dumps(score_report,default=str,sort_keys=True)}")
    # LIVE-SCORES are a secondary inventory: they are not the historical NFL
    # uploader and cannot override its authoritative Season and Season_Stage.
    try:
        out["historical_uploader"]=_audit_nfl_uploader(bq,schemas[NFL_RAW],schemas[NFL_CONTEXT],schemas[NFL_VIEW])
    except Exception as exc:
        out["historical_uploader"]={"status":"HOLD","reason":"HISTORICAL_SOURCE_QUERY_FAILED",
                                    "error":f"{type(exc).__name__}: {exc}"}
    log_func(f"[NFL-AUDIT-V1-HISTORICAL] {json.dumps(out['historical_uploader'],default=str,sort_keys=True)}")
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
    hist=out['historical_uploader']
    hist_ok=(hist.get('status')=='HISTORY_COVERAGE_PASS' and
             hist.get('training_view',{}).get('status')=='GRAIN_AND_ELIGIBILITY_PASS' and
             hist.get('context',{}).get('status')=='FIRST_GAME_CHECK_PASS')
    # Live score table and the generic score-with-features table are *secondary*
    # sources, not the historical uploader. A coverage PASS is NOT a model PASS.
    out["status"]=("READY_FOR_OFFLINE_FEATURE_LEAKAGE_REVIEW" if hist_ok else "HOLD_REVIEW_HISTORICAL_SOURCE")
    out["next_step"]=("Inspect the core-view SQL for strictly prior-only derivation and season/team alignment; "
                      "audit historical closing lines separately from time-stamped odds; find archived incumbent "
                      "OOF/prospective predictions, then test independent Spread/H2H/Totals season-forward challengers. "
                      "Do not use 2026 to select challengers or auto-promote models.")
    log_func(f"[NFL-AUDIT-V1-CONTRACT] status={out['status']} historical_source=AUTHORITATIVE_NFL_UPLOADER "
             f"hist_status={hist.get('status')} view_status={hist.get('training_view',{}).get('status')} "
             f"context_status={hist.get('context',{}).get('status')} "
             f"publication=FALSE production_authority=0 legacy_nfl=UNCHANGED ncaaf=UNCHANGED "
             f"data_coverage_only=TRUE champion_win_claim=FALSE")
    # V1.3 additional read-only prior-feature audit. The authoritative history
    # gate remains independent; do not run this on an unverified raw dataset.
    if out['status']=='READY_FOR_OFFLINE_FEATURE_LEAKAGE_REVIEW':
        try:
            import nfl_feature_audit_v1 as _feature_v13
            out['feature_leakage_review']=_feature_v13.run_feature_audit(bq,out,log_func=log_func)
        except Exception as e:
            out['feature_leakage_review']={'status':'HOLD','reason':'FEATURE_AUDIT_IMPORT_OR_EXECUTION_FAILED',
                                           'error':f'{type(e).__name__}: {e}'}
            log_func('[NFL-FEATURE-V1-CONTRACT] status=HOLD_FEATURE_REVIEW reason=FEATURE_AUDIT_EXCEPTION '
                     f'error={type(e).__name__}: {e} publication=FALSE authority=0')
        out['status']=out['feature_leakage_review']['status']
        log_func('[NFL-AUDIT-V1.3-FINAL] status='+out['status']+
                 ' publication=FALSE production_authority=0 ncaaf=UNCHANGED legacy_nfl=UNCHANGED')
    if hard_fail and hist.get('reason')=='BAD_RAW_GAME_GRAIN_OR_SEASON_STAGE':
        raise RuntimeError('[NFL-AUDIT-V1-CONTRACT] HARD_HOLD invalid historical NFL game side grain; no model mutated')
    return out


if __name__ == "__main__":
    print(json.dumps(run_nfl_audit_v1(), indent=2,default=str))
