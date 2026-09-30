"""NFL feature audit V1.3: read-only provenance and independent prior-only parity.

NO model fitting, no publication, no DDL/DML. The audited training view deliberately
contains labels and historical closing odds: ONLY EXPLICIT_FEATURE_MANIFEST can be
considered for a subsequent experimental model, subject to this audit passing.

The recorded historical closing price is not a time-stamped, executable pregame
quote. It may define a historical closing-line settlement target but is never
promoted into an as-of predictor or verified past ROI/CLV by this audit.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pandas as pd

PROJECT = 'sharplogger'
DATASET = 'sharp_data'
RAW = f'{PROJECT}.{DATASET}.nfl_historical_game_side_raw'
CTX = f'{PROJECT}.{DATASET}.nfl_historical_game_side_context'
VIEW = f'{PROJECT}.{DATASET}.nfl_historical_core_training_vw'
SOURCE_TAG = 'nfl-feature-audit-v1.3-prior-only-20260930'

# Reviewed against nfl_context_rebuild.sql (source SHA256
# eda30bf12d99ed35b76f9c467ff70566d14a5906b871195964e3a688cfc4ea25).
# This manifest is an EXPERIMENTAL feature contract, not a claimed full lineage
# proof of the currently materialized BigQuery context. Parity checks follow.
EXPLICIT_FEATURE_MANIFEST = {
 'pregame_schedule': (
  'Week_Number','Is_Home','Is_Away','Is_Neutral','Is_Regular_Season','Is_Postseason',
  'Is_Wild_Card','Is_Divisional_Round','Is_Conference_Championship','Is_Super_Bowl',
  'Is_Conference_Game','Is_Interconference_Game','Is_Division_Game',
  'Game_DOW','Game_Hour_ET','Is_Thursday_Game','Is_Monday_Game','Is_Sunday_Game',
  'Is_Saturday_Game','Is_Night_Game','Is_PrimeTime','Is_TNF','Is_MNF','Is_SNF',
 ),
 'prior_team_state': (
  'Team_Game_Number','Prev_SU_Win','Prev_SU_Loss','Prev_SU_Tie','Prev_Points_For',
  'Prev_Points_Against','Prev_SU_Margin','Prev_ATS_Win','Prev_ATS_Loss',
  'Prev_ATS_Cover_Margin','Prev_Off_Yards_Per_Play','Prev_Def_Yards_Per_Play_Allowed',
  'Prev_Turnover_Margin','Prev_Third_Down_Pct','Prev_Sack_Rate_Allowed',
  'Prev_Defensive_Sacks','Prev_Total_Yards_Allowed','Prev_Total_Yards',
  'Prev_Total_Plays','Prev_Pass_Yards','Prev_Rush_Yards','Prev_Turnovers',
  'WinPct_Prior_System','ATS_WinPct_Prior_System','Avg_Points_For_Prior',
  'Avg_Points_Against_Prior','Avg_SU_Margin_Prior','Avg_Off_YPP_Last3_Prior',
  'Avg_Def_YPP_Last3_Prior','Avg_Rush_YPA_Last3_Prior','Avg_Net_Pass_YPA_Last3_Prior',
  'Avg_Turnover_Margin_Last3_Prior','Avg_Third_Down_Pct_Last3_Prior',
  'Avg_Sack_Rate_Allowed_Last3_Prior','Avg_Defensive_Sacks_Last3_Prior',
  'Avg_Time_Possession_Last3_Prior','Avg_Points_For_Last5_Prior',
  'Avg_Points_Against_Last5_Prior','Avg_SU_Margin_Last5_Prior',
  'Avg_Off_YPP_Last5_Prior','Avg_Def_YPP_Last5_Prior',
 ),
 'prior_opponent_and_matchups': (
  'Opp_WinPct_Prior_System','Opp_ATS_WinPct_Prior_System',
  'Opp_Avg_Points_For_Last5_Prior','Opp_Avg_Points_Against_Last5_Prior',
  'Opp_Avg_SU_Margin_Last5_Prior','Opp_Avg_Off_YPP_Last3_Prior',
  'Opp_Avg_Def_YPP_Last3_Prior','Opp_Avg_Turnover_Margin_Last3_Prior',
  'Opp_Avg_Third_Down_Pct_Last3_Prior','Opp_Avg_Sack_Rate_Allowed_Last3_Prior',
  'Opp_Prior_Season_WinPct','Opp_Revenge_Flag_Current',
  'Last_Matchup_SU_Win_Value_System','Last_Matchup_SU_Margin_System',
  'Days_Since_Last_Matchup_System','Revenge_Flag_Current',
  'WinPct_Prior_Diff','ATS_WinPct_Prior_Diff','Avg_SU_Margin_Last5_Diff',
  'Off_YPP_vs_Opp_Def_Last3_Diff','Def_YPP_vs_Opp_Off_Last3_Diff',
  'Turnover_Margin_Last3_Diff','Prior_Season_WinPct_Diff',
 ),
 'prior_season_and_rest': (
  'Prior_Season_Regular_Games','Prior_Season_Wins','Prior_Season_Losses',
  'Prior_Season_Ties','Prior_Season_WinPct','Prior_Season_Avg_Margin',
  'Prior_Season_Avg_Points_For','Prior_Season_Avg_Points_Against',
  'Days_Since_Last_Game_System','Is_Short_Rest','Is_Standard_Rest','Is_Long_Rest',
  'Is_Bye_Like_Rest','Is_Thursday_Short_Week','Opp_Days_Since_Last_Game_System',
  'Opp_Is_Short_Rest','Opp_Is_Long_Rest','Rest_Differential_Days',
 ),
}

# Explicitly barred as predictor inputs, including derived CURRENT-game close
# flags and any same-game outcome. Prior-game ATS results are distinct and
# remain permitted only because they are lagged.
EXCLUDED_SOURCE_FIELDS = (
 'Team_Score','Opponent_Score','SU_Result','SU_Win','SU_Loss','SU_Tie',
 'SU_Win_Value','SU_Margin','ATS_Result_Close','ATS_Win','ATS_Loss',
 'ATS_Push','ATS_Win_Value','ATS_Cover_Margin','Total_Result_Close',
 'Spread_Value','Opening_Spread','Current_Total','Opening_Total',
 'ML_Odds','Opening_ML_Odds','Spread_Move_Open_to_Close','Total_Move_Open_to_Close',
 'Is_Spread_Favorite','Is_Spread_Dog','Is_ML_Favorite','Is_ML_Dog',
 'Source_Opening_Odds_Raw','Source_Line_Movement_1_Raw',
 'Source_Line_Movement_2_Raw','Source_Line_Movement_3_Raw',
 'Source_Closing_Odds_Raw',
 'Pathi_FB_On_Key_3','Pathi_FB_On_Key_6','Pathi_FB_On_Key_7',
 'Pathi_FB_On_Key_10','Pathi_FB_On_Key_14','Dist_to_3','Dist_to_6',
 'Dist_to_7','Dist_to_10','Dist_to_14','Pathi_FB_Moved_Through_Key',
 'Pathi_FB_Moved_Onto_Key','Pathi_FB_Moved_Off_Key',
 'Pathi_FB_Crossed_Key_Toward_Team','Pathi_FB_Crossed_Key_Away_From_Team',
 'Prior_Season_Playoff','Is_Defending_Champion','Opp_Prior_Season_Playoff',
 'Opp_Is_Defending_Champion',
)

# Predicted values and performance comparisons are NOT generated here.
FOLD_START_TRAIN_END = 2020
FOLD_LAST_VALIDATE = 2025
SEALED_CONFIRMATION = 2026


def _flatten_manifest():
    vals=[x for group in EXPLICIT_FEATURE_MANIFEST.values() for x in group]
    if len(vals)!=len(set(vals)):
        raise RuntimeError('NFL feature manifest has duplicate field names')
    overlap=set(vals)&set(EXCLUDED_SOURCE_FIELDS)
    if overlap:
        raise RuntimeError(f'Unsafe candidate/label overlap: {sorted(overlap)}')
    return vals


def classify_source_fields(view_columns):
    """Fail closed if an explicitly eligible field is missing from live view."""
    names=_flatten_manifest()
    cols=set(view_columns)
    missing=sorted(set(names)-cols)
    return {
      'status':'EXPLICIT_ALLOWLIST_PASS' if not missing else 'HOLD',
      'manifest_feature_count':len(names),
      'manifest_groups':{k:len(v) for k,v in EXPLICIT_FEATURE_MANIFEST.items()},
      'missing_manifest_fields':missing,
      'excluded_source_fields_present':sorted(set(EXCLUDED_SOURCE_FIELDS)&cols),
      'excluded_count':len(set(EXCLUDED_SOURCE_FIELDS)&cols),
      'unclassified_fields_not_automatically_admitted':len(cols-set(names)-set(EXCLUDED_SOURCE_FIELDS)),
      'prior_season_flags_blocked_when_prior_year_unknown':True,
      'note':'Only this named subset can enter later offline challengers. It is not sufficient alone to prove live as-of provenance.',
    }


def view_contract(view_query):
    """Inspect live VIEW SQL for required eligibility and no hidden market admission."""
    q=re.sub(r'\s+', ' ',str(view_query or '').strip()).casefold()
    checks={
      'context_source': 'nfl_historical_game_side_context' in q,
      'explicit_regular_post_eligibility': bool(re.search(r"season_stage\s+in\s*\(\s*'regular'\s*,\s*'postseason'\s*\)",q)),
      'modern_microstructure_disabled':bool(re.search(r'0\s+as\s+modern_book_microstructure_eligible\b',q)),
      'historical_close_features_disabled':bool(re.search(r'0\s+as\s+historical_close_features_eligible\b',q)),
      'timing_model_disabled':bool(re.search(r'0\s+as\s+timing_model_eligible\b',q)),
    }
    return {'status':'VIEW_CONTRACT_PASS' if all(checks.values()) else 'HOLD',
            'checks':checks,
            'note':'Live view text is inspectable, but the materialized context table does not expose the historical creation query. Independent raw/context parity runs separately.'}


def fold_blueprint(history_by_season):
    """Chronological DESIGN only; never fit or use sealed 2026 for selection."""
    counts={int(y):int(stages.get('REGULAR',0))+int(stages.get('POSTSEASON',0))
            for y,stages in history_by_season.items()}
    rows=[]
    for holdout in range(FOLD_START_TRAIN_END+1,FOLD_LAST_VALIDATE+1):
        train=[y for y in sorted(counts) if y<holdout and y<=2025 and counts[y]>=200]
        val=counts.get(holdout,0)
        rows.append({'train_through':holdout-1,'validation_season':holdout,
                     'training_seasons':train,'training_games':sum(counts[y] for y in train),
                     'validation_games':val,'eligible':len(train)>=4 and val>=200})
    all_ok=len(rows)==5 and all(r['eligible'] for r in rows)
    return {'status':'SEASON_FORWARD_BLUEPRINT_PASS' if all_ok else 'HOLD',
            'folds':rows,'sealed_confirmation_season':SEALED_CONFIRMATION,
            'sealed_confirmation_games':counts.get(2026,0),
            'rule':'2026 excluded from all tuning, calibration selection and promotion decisions. One physical game is one validation unit.',
            'metrics_planned':['log_loss','brier','calibration_by_season','discrimination','ATS_or_total_settlement_where_valid','coverage','decision_stability'],
            'models_fit':0}


def _q(client,query):
    from google.cloud import bigquery
    cfg=bigquery.QueryJobConfig(use_query_cache=True,maximum_bytes_billed=20*1024**3)
    return client.query(query,job_config=cfg).to_dataframe()


# Independently recompute temporal values from the authoritative RAW table,
# rather than checking that prior columns are non-NULL only on game one.
# This SELECT is deliberately row-complete, not a sample. The materialized
# context must exactly agree on raw identity and known prior-only derivations.
RAW_PRIOR_PARITY_SQL = f"""
WITH ordered_raw AS (
 SELECT r.Season,r.Source_Name,r.Source_Game_ID,
   LOWER(TRIM(r.Team_Norm)) AS Team_Norm,r.Game_Date,r.Season_Stage,
   CASE WHEN r.Season_Stage='PRESEASON' THEN 'PRESEASON' ELSE 'REGULAR_POST' END AS State_Group,
   COALESCE(DATETIME(r.Game_Date, SAFE.PARSE_TIME('%I:%M %p', UPPER(TRIM(r.Start_Time_ET)))), DATETIME(r.Game_Date)) AS Game_Start,
   CONCAT(CAST(r.Season AS STRING),'|',r.Source_Name,'|',r.Source_Game_ID) AS Game_Key,
   r.Team_Score,r.Opponent_Score,r.Postgame_Total_Yards,r.Postgame_Total_Plays,
   r.ATS_Result_Close,
   SAFE_DIVIDE(CAST(r.Postgame_Total_Yards AS FLOAT64),NULLIF(CAST(r.Postgame_Total_Plays AS FLOAT64),0.0)) AS Prior_Outcome_YPP,
   CASE WHEN r.Team_Score>r.Opponent_Score THEN 1.0 WHEN r.Team_Score<r.Opponent_Score THEN 0.0 ELSE 0.5 END AS Win_Val,
   CASE WHEN UPPER(r.ATS_Result_Close)='WIN' THEN 1.0 WHEN UPPER(r.ATS_Result_Close)='LOSS' THEN 0.0
       WHEN UPPER(r.ATS_Result_Close)='PUSH' THEN 0.5 ELSE NULL END AS ATS_Val
 FROM `{RAW}` r WHERE UPPER(TRIM(r.Sport))='NFL'
), prior AS (
 SELECT r.*,
   ROW_NUMBER() OVER w AS expected_game_number,
   LAG(Game_Date) OVER w AS expected_prev_game_date,
   LAG(Team_Score) OVER w AS expected_prev_points_for,
   LAG(Opponent_Score) OVER w AS expected_prev_points_against,
   LAG(Postgame_Total_Yards) OVER w AS expected_prev_total_yards,
   LAG(Postgame_Total_Plays) OVER w AS expected_prev_total_plays,
   LAG(Prior_Outcome_YPP) OVER w AS expected_prev_ypp,
   AVG(Win_Val) OVER w_prior AS expected_win_pct,
   AVG(ATS_Val) OVER w_prior AS expected_ats_pct,
   AVG(CAST(Team_Score AS FLOAT64)) OVER w_last5 AS expected_points_last5
 FROM ordered_raw r
 WINDOW
   w AS (PARTITION BY Season,State_Group,Team_Norm ORDER BY Game_Start,Game_Key),
   w_prior AS (PARTITION BY Season,State_Group,Team_Norm ORDER BY Game_Start,Game_Key ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
   w_last5 AS (PARTITION BY Season,State_Group,Team_Norm ORDER BY Game_Start,Game_Key ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING)
), checks AS (
 SELECT p.Season,p.Source_Name,p.Source_Game_ID,p.Team_Norm,p.Season_Stage,p.expected_game_number,
  c.Team_Norm IS NULL AS missing_context,
  p.Team_Game_Number IS NULL AS bad_impossible_dummy,
  c.Team_Game_Number IS DISTINCT FROM p.expected_game_number AS game_number_mismatch,
  c.Prev_Points_For IS DISTINCT FROM p.expected_prev_points_for AS prev_points_for_mismatch,
  c.Prev_Points_Against IS DISTINCT FROM p.expected_prev_points_against AS prev_points_against_mismatch,
  c.Prev_Total_Yards IS DISTINCT FROM p.expected_prev_total_yards AS prev_total_yards_mismatch,
  c.Prev_Total_Plays IS DISTINCT FROM p.expected_prev_total_plays AS prev_total_plays_mismatch,
  c.Prev_Off_Yards_Per_Play IS DISTINCT FROM p.expected_prev_ypp AS prev_ypp_mismatch,
  c.Days_Since_Last_Game_System IS DISTINCT FROM DATE_DIFF(p.Game_Date,p.expected_prev_game_date,DAY) AS rest_mismatch,
  c.WinPct_Prior_System IS DISTINCT FROM p.expected_win_pct AS win_pct_mismatch,
  c.ATS_WinPct_Prior_System IS DISTINCT FROM p.expected_ats_pct AS ats_pct_mismatch,
  c.Avg_Points_For_Last5_Prior IS DISTINCT FROM p.expected_points_last5 AS last5_points_mismatch,
  (p.expected_game_number=1 AND (c.Prev_Points_For IS NOT NULL OR c.Prev_Points_Against IS NOT NULL OR
     c.WinPct_Prior_System IS NOT NULL OR c.Prev_Total_Yards IS NOT NULL OR c.Avg_Points_For_Last5_Prior IS NOT NULL)) AS first_game_leakage
 FROM prior p LEFT JOIN `{CTX}` c
 ON c.Season=p.Season AND c.Source_Name=p.Source_Name AND c.Source_Game_ID=p.Source_Game_ID AND LOWER(TRIM(c.Team_Norm))=p.Team_Norm
)
SELECT COUNT(*) AS raw_rows,
 COUNTIF(missing_context) AS missing_context_rows,
 COUNTIF(game_number_mismatch) AS game_number_mismatch,
 COUNTIF(prev_points_for_mismatch) AS prev_points_for_mismatch,
 COUNTIF(prev_points_against_mismatch) AS prev_points_against_mismatch,
 COUNTIF(prev_total_yards_mismatch) AS prev_total_yards_mismatch,
 COUNTIF(prev_total_plays_mismatch) AS prev_total_plays_mismatch,
 COUNTIF(prev_ypp_mismatch) AS prev_ypp_mismatch,
 COUNTIF(rest_mismatch) AS rest_mismatch,
 COUNTIF(win_pct_mismatch) AS win_pct_mismatch,
 COUNTIF(ats_pct_mismatch) AS ats_pct_mismatch,
 COUNTIF(last5_points_mismatch) AS last5_points_mismatch,
 COUNTIF(first_game_leakage) AS first_game_leakage,
 COUNTIF(expected_game_number=1) AS first_games
FROM checks
""".replace('  p.Team_Game_Number IS NULL AS bad_impossible_dummy,\n','')

CONTEXT_EXTRA_ROWS_SQL = f"""SELECT COUNT(*) AS extra_context_rows FROM `{CTX}` c
 LEFT JOIN `{RAW}` r ON r.Season=c.Season AND r.Source_Name=c.Source_Name
   AND r.Source_Game_ID=c.Source_Game_ID AND LOWER(TRIM(r.Team_Norm))=LOWER(TRIM(c.Team_Norm))
 WHERE r.Source_Game_ID IS NULL"""

# The same-game outcome of the opponent may be present in RAW but its *prior*
# state in the context must match the paired side's own lagged state.
OPPONENT_PARITY_SQL = f"""SELECT COUNT(*) AS paired_rows,
 COUNTIF(o.Team_Norm IS NULL) AS missing_opponent_context,
 COUNTIF(t.Opp_WinPct_Prior_System IS DISTINCT FROM o.WinPct_Prior_System) AS opponent_win_pct_mismatch,
 COUNTIF(t.Opp_ATS_WinPct_Prior_System IS DISTINCT FROM o.ATS_WinPct_Prior_System) AS opponent_ats_pct_mismatch,
 COUNTIF(t.Opp_Days_Since_Last_Game_System IS DISTINCT FROM o.Days_Since_Last_Game_System) AS opponent_rest_mismatch,
 COUNTIF(t.Opp_Avg_Points_For_Last5_Prior IS DISTINCT FROM o.Avg_Points_For_Last5_Prior) AS opponent_last5_mismatch
 FROM `{CTX}` t LEFT JOIN `{CTX}` o
 ON o.Season=t.Season AND o.Source_Name=t.Source_Name AND o.Source_Game_ID=t.Source_Game_ID
 AND o.Team_Norm=t.Opponent_Norm AND o.Opponent_Norm=t.Team_Norm
 WHERE UPPER(TRIM(t.Sport))='NFL'"""

# Market labels are only an inventory. No attempt to treat Source_Close_* as an
# as-of predictor or infer historical executability from an uploaded box score.
LABEL_COVERAGE_SQL = f"""SELECT Season,
 COUNT(DISTINCT CONCAT(Source_Name,'|',Source_Game_ID)) AS physical_games,
 COUNTIF(Source_Close_Spread IS NOT NULL) AS close_spread_side_rows,
 COUNTIF(Source_Close_Total IS NOT NULL) AS close_total_side_rows,
 COUNTIF(Source_Close_Moneyline IS NOT NULL) AS close_moneyline_side_rows,
 COUNTIF(Source_Open_Spread IS NOT NULL) AS open_spread_side_rows,
 COUNTIF(Source_Open_Total IS NOT NULL) AS open_total_side_rows
 FROM `{RAW}` WHERE UPPER(TRIM(Sport))='NFL' AND Season_Stage IN ('REGULAR','POSTSEASON')
 GROUP BY Season ORDER BY Season"""


def assess_recompute(result):
    """Pure logic shared by tests and the live report."""
    mandatory=('raw_rows','missing_context_rows','game_number_mismatch','prev_points_for_mismatch',
       'prev_points_against_mismatch','prev_total_yards_mismatch','prev_total_plays_mismatch',
       'prev_ypp_mismatch','rest_mismatch','win_pct_mismatch','ats_pct_mismatch',
       'last5_points_mismatch','first_game_leakage','first_games','extra_context_rows',
       'paired_rows','missing_opponent_context','opponent_win_pct_mismatch',
       'opponent_ats_pct_mismatch','opponent_rest_mismatch','opponent_last5_mismatch')
    missing=[k for k in mandatory if k not in result or pd.isna(result.get(k))]
    if missing:return {'status':'HOLD','reason':'QUERY_FIELDS_MISSING','missing':missing}
    values={k:int(result[k]) for k in mandatory}
    bad={k:v for k,v in values.items() if k not in ('raw_rows','first_games','paired_rows') and v!=0}
    counts_ok=values['raw_rows']>0 and values['first_games']>0 and values['raw_rows']==values['paired_rows']
    return {'status':'PRIOR_RECOMPUTE_PASS' if not bad and counts_ok else 'HOLD',
            'reason':'INDEPENDENT_FULL_HISTORY_PRIOR_PARITY' if not bad and counts_ok else 'PRIOR_RECOMPUTE_DISCREPANCY',
            'counts':values,'nonzero_errors':bad,'count_consistency':counts_ok,
            'scope':'Entire NFL raw/context history, not a sample; checks selected representative prior features and opponent pairing.'}


def run_feature_audit(bq_client, base_report, log_func=print):
    """Only run after the V1.2 authoritative history gate passed. READ ONLY."""
    hist=base_report.get('historical_uploader') or {}
    if base_report.get('status') != 'READY_FOR_OFFLINE_FEATURE_LEAKAGE_REVIEW' or hist.get('status')!='HISTORY_COVERAGE_PASS':
        return {'status':'HOLD','reason':'BASE_NFL_AUDIT_NOT_READY','publication':False}
    schemas=base_report.get('sources') or {}
    table=bq_client.get_table(VIEW)
    manifest=classify_source_fields([f.name for f in table.schema])
    contract=view_contract(getattr(table,'view_query',None))
    log_func('[NFL-FEATURE-V1-ALLOWLIST] '+json.dumps(manifest,sort_keys=True))
    log_func('[NFL-FEATURE-V1-VIEW-CONTRACT] '+json.dumps(contract,sort_keys=True))
    try:
        p=_q(bq_client,RAW_PRIOR_PARITY_SQL).iloc[0].to_dict()
        p.update(_q(bq_client,CONTEXT_EXTRA_ROWS_SQL).iloc[0].to_dict())
        p.update(_q(bq_client,OPPONENT_PARITY_SQL).iloc[0].to_dict())
        recompute=assess_recompute(p)
    except Exception as exc:
        recompute={'status':'HOLD','reason':'PRIOR_RECOMPUTE_QUERY_FAILED','error':f'{type(exc).__name__}: {exc}'}
    log_func('[NFL-FEATURE-V1-RECOMPUTE] '+json.dumps(recompute,sort_keys=True,default=str))
    try:
        raw_labels=_q(bq_client,LABEL_COVERAGE_SQL)
        labels={str(int(r['Season'])):{k:int(r[k]) for k in raw_labels.columns if k!='Season'}
                for r in raw_labels.to_dict('records')}
        label_inventory={'status':'INVENTORY_ONLY','by_season':labels,
          'historical_close_feature_predictor_authority':False,
          'historical_odds_roi_clv_verified':False,
          'note':'Raw Source_Close_* fields are retrospective line/settlement context, not timestamped and executable historical snapshots.'}
    except Exception as exc:
        label_inventory={'status':'HOLD','reason':'LABEL_COVERAGE_QUERY_FAILED','error':f'{type(exc).__name__}: {exc}'}
    log_func('[NFL-FEATURE-V1-HISTORICAL-LINES] '+json.dumps(label_inventory,sort_keys=True,default=str))
    folds=fold_blueprint(hist.get('games_by_season_stage') or {})
    log_func('[NFL-FEATURE-V1-FOLDS] '+json.dumps(folds,sort_keys=True))
    good=(manifest['status']=='EXPLICIT_ALLOWLIST_PASS' and contract['status']=='VIEW_CONTRACT_PASS'
          and recompute['status']=='PRIOR_RECOMPUTE_PASS' and folds['status']=='SEASON_FORWARD_BLUEPRINT_PASS'
          and label_inventory['status']=='INVENTORY_ONLY')
    status='READY_FOR_OFFLINE_CHALLENGER_SANDBOX' if good else 'HOLD_FEATURE_REVIEW'
    out={'status':status,'manifest':manifest,'view_contract':contract,'recompute':recompute,
         'historical_lines':label_inventory,'fold_blueprint':folds,
         'publication':False,'production_authority':0,
         'incumbent_performance_verified':False,
         'market_asof_historical_backtest_verified':False,
         'next_step':'Score-only NFL challenger experiments may start if green, but require independent as-of odds evidence and archived OOF incumbent predictions before any market-edge promotion.'}
    log_func('[NFL-FEATURE-V1-CONTRACT] status='+status+' publication=FALSE authority=0 no_incumbent_mutation=TRUE '
             'ncaaf=UNCHANGED historical_close_predictor_authority=FALSE market_roi_verified=FALSE')
    return out
