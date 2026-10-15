-- =====================================================================
-- NCAAF HISTORICAL CONTEXT REBUILD
-- Rebuilds leakage-safe pregame state from every row currently loaded in
-- ncaaf_historical_game_side_raw. Designed to be run after each weekly MERGE.
-- =====================================================================

CREATE OR REPLACE TABLE
  `sharplogger.sharp_data.ncaaf_historical_game_side_context`
PARTITION BY Game_Date
CLUSTER BY Season, Team, Opponent
AS

WITH
raw_seasons AS (
  SELECT DISTINCT UPPER(TRIM(Sport)) AS Sport, Season
  FROM `sharplogger.sharp_data.ncaaf_historical_game_side_raw`
),

alignment_ranked AS (
  SELECT
    s.Sport,
    s.Season,
    LOWER(TRIM(a.Team_Norm)) AS Team_Norm,
    a.Conference,
    a.Division,
    a.Subdivision,
    a.League_Group,
    a.Effective_Season_From,
    ROW_NUMBER() OVER (
      PARTITION BY s.Sport, s.Season, LOWER(TRIM(a.Team_Norm))
      ORDER BY a.Effective_Season_From DESC
    ) AS rn
  FROM raw_seasons s
  JOIN `sharplogger.sharp_data.team_sport_alignment_history` a
    ON UPPER(TRIM(a.Sport)) = s.Sport
   AND a.Effective_Season_From <= s.Season
   AND COALESCE(a.Use_For_Context, TRUE)
),

alignment AS (
  SELECT * EXCEPT(rn)
  FROM alignment_ranked
  WHERE rn = 1
),

base AS (
  SELECT
    UPPER(TRIM(r.Sport)) AS Sport,
    r.Season,
    r.Source_Name,
    r.Source_Grain,
    r.Source_Game_ID,
    CONCAT(CAST(r.Season AS STRING), '|', r.Source_Game_ID) AS Game_Key,
    r.Game_Date,
    COALESCE(
      DATETIME(r.Game_Date, SAFE.PARSE_TIME('%I:%M %p', UPPER(TRIM(r.Start_Time_ET)))),
      DATETIME(r.Game_Date)
    ) AS Game_Start,
    r.Week,
    r.Week AS Week_Number,
    r.Start_Time_ET,
    r.Source_Team_Name,
    r.Source_Opponent_Name,
    r.Team,
    LOWER(TRIM(r.Team_Norm)) AS Team_Norm,
    r.Canonical_Team_Name,
    r.Canonical_Team_Norm,
    r.Opponent,
    LOWER(TRIM(r.Opponent_Norm)) AS Opponent_Norm,
    r.Opponent_Canonical_Team_Name,
    r.Opponent_Canonical_Team_Norm,
    COALESCE(NULLIF(ta.Conference, ''), r.Conference) AS Conference,
    COALESCE(NULLIF(oa.Conference, ''), r.Opponent_Conference) AS Opponent_Conference,
    ta.Division AS Division,
    oa.Division AS Opponent_Division,
    COALESCE(NULLIF(ta.Subdivision, ''), r.Subdivision) AS Subdivision,
    COALESCE(NULLIF(oa.Subdivision, ''), r.Opponent_Subdivision) AS Opponent_Subdivision,
    ta.League_Group AS League_Group,
    oa.League_Group AS Opponent_League_Group,
    ta.Effective_Season_From AS Team_Alignment_Effective_Season,
    oa.Effective_Season_From AS Opponent_Alignment_Effective_Season,
    r.Venue,
    r.Is_Home,
    r.Is_Away,
    r.Is_Neutral,
    r.Consensus_Close_Spread AS Spread_Value,
    r.Consensus_Open_Spread AS Opening_Spread,
    r.Consensus_Close_Total AS Current_Total,
    r.Consensus_Open_Total AS Opening_Total,
    r.Consensus_Close_Moneyline AS ML_Odds,
    r.Consensus_Open_Moneyline AS Opening_ML_Odds,
    r.Spread_Move_Open_to_Close,
    r.Total_Move_Open_to_Close,
    CASE
      WHEN r.Consensus_Close_Spread < 0 THEN 1
      WHEN r.Consensus_Close_Spread IS NULL THEN NULL
      ELSE 0
    END AS Is_Spread_Favorite,
    CASE
      WHEN r.Consensus_Close_Spread > 0 THEN 1
      WHEN r.Consensus_Close_Spread IS NULL THEN NULL
      ELSE 0
    END AS Is_Spread_Dog,
    CASE
      WHEN r.Consensus_Close_Moneyline < 0 THEN 1
      WHEN r.Consensus_Close_Moneyline IS NULL THEN NULL
      ELSE 0
    END AS Is_ML_Favorite,
    CASE
      WHEN r.Consensus_Close_Moneyline > 0 THEN 1
      WHEN r.Consensus_Close_Moneyline IS NULL THEN NULL
      ELSE 0
    END AS Is_ML_Dog,
    CASE
      WHEN REGEXP_CONTAINS(UPPER(COALESCE(r.Dataset, '')), r'REGULAR SEASON') THEN 1
      ELSE 0
    END AS Is_Regular_Season,
    CASE
      WHEN COALESCE(NULLIF(ta.Conference, ''), r.Conference) IS NOT NULL
       AND COALESCE(NULLIF(oa.Conference, ''), r.Opponent_Conference) IS NOT NULL
       AND COALESCE(NULLIF(ta.Conference, ''), r.Conference)
           = COALESCE(NULLIF(oa.Conference, ''), r.Opponent_Conference)
      THEN 1 ELSE 0
    END AS Is_Conference_Game,
    r.Team_Score,
    r.Opponent_Score,
    r.SU_Result,
    CASE
      WHEN r.Team_Score > r.Opponent_Score THEN 1
      WHEN r.Team_Score < r.Opponent_Score THEN 0
      ELSE NULL
    END AS SU_Win,
    CASE
      WHEN r.Team_Score < r.Opponent_Score THEN 1
      WHEN r.Team_Score > r.Opponent_Score THEN 0
      ELSE NULL
    END AS SU_Loss,
    r.Team_Score - r.Opponent_Score AS SU_Margin,
    r.ATS_Result_Close,
    CASE
      WHEN UPPER(r.ATS_Result_Close) = 'WIN' THEN 1
      WHEN UPPER(r.ATS_Result_Close) = 'LOSS' THEN 0
      ELSE NULL
    END AS ATS_Win,
    CASE
      WHEN UPPER(r.ATS_Result_Close) = 'LOSS' THEN 1
      WHEN UPPER(r.ATS_Result_Close) = 'WIN' THEN 0
      ELSE NULL
    END AS ATS_Loss,
    CASE
      WHEN r.Consensus_Close_Spread IS NOT NULL
      THEN r.Team_Score - r.Opponent_Score + r.Consensus_Close_Spread
      ELSE NULL
    END AS ATS_Cover_Margin,
    r.Total_Result_Close,
    r.Postgame_Total_Yards,
    r.Postgame_Total_Plays,
    r.Postgame_Pass_Yards,
    r.Postgame_Rush_Yards,
    r.Postgame_Turnovers,
    r.Dataset,
    r.Source_Opening_Odds_Raw,
    r.Source_Closing_Odds_Raw
  FROM `sharplogger.sharp_data.ncaaf_historical_game_side_raw` r
  LEFT JOIN alignment ta
    ON ta.Sport = UPPER(TRIM(r.Sport))
   AND ta.Season = r.Season
   AND ta.Team_Norm = LOWER(TRIM(r.Team_Norm))
  LEFT JOIN alignment oa
    ON oa.Sport = UPPER(TRIM(r.Sport))
   AND oa.Season = r.Season
   AND oa.Team_Norm = LOWER(TRIM(r.Opponent_Norm))
),

team_state AS (
  SELECT
    b.*,
    ROW_NUMBER() OVER (
      PARTITION BY Sport, Season, Team_Norm
      ORDER BY Game_Start, Game_Key
    ) AS Team_Game_Number,
    LAG(Opponent) OVER (
      PARTITION BY Sport, Season, Team_Norm ORDER BY Game_Start, Game_Key
    ) AS Prev_Opponent,
    LAG(SU_Win) OVER (
      PARTITION BY Sport, Season, Team_Norm ORDER BY Game_Start, Game_Key
    ) AS Prev_SU_Win,
    LAG(SU_Loss) OVER (
      PARTITION BY Sport, Season, Team_Norm ORDER BY Game_Start, Game_Key
    ) AS Prev_SU_Loss,
    LAG(Team_Score) OVER (
      PARTITION BY Sport, Season, Team_Norm ORDER BY Game_Start, Game_Key
    ) AS Prev_Points_For,
    LAG(Opponent_Score) OVER (
      PARTITION BY Sport, Season, Team_Norm ORDER BY Game_Start, Game_Key
    ) AS Prev_Points_Against,
    LAG(SU_Margin) OVER (
      PARTITION BY Sport, Season, Team_Norm ORDER BY Game_Start, Game_Key
    ) AS Prev_SU_Margin,
    LAG(ATS_Cover_Margin) OVER (
      PARTITION BY Sport, Season, Team_Norm ORDER BY Game_Start, Game_Key
    ) AS Prev_ATS_Cover_Margin,
    LAG(ATS_Win) OVER (
      PARTITION BY Sport, Season, Team_Norm ORDER BY Game_Start, Game_Key
    ) AS Prev_ATS_Win,
    LAG(ATS_Loss) OVER (
      PARTITION BY Sport, Season, Team_Norm ORDER BY Game_Start, Game_Key
    ) AS Prev_ATS_Loss,
    DATE_DIFF(
      Game_Date,
      LAG(Game_Date) OVER (
        PARTITION BY Sport, Season, Team_Norm ORDER BY Game_Start, Game_Key
      ),
      DAY
    ) AS Days_Since_Last_Game_System,
    AVG(CAST(Team_Score AS FLOAT64)) OVER (
      PARTITION BY Sport, Season, Team_Norm
      ORDER BY Game_Start, Game_Key
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS Avg_Points_For_Prior,
    AVG(CAST(Opponent_Score AS FLOAT64)) OVER (
      PARTITION BY Sport, Season, Team_Norm
      ORDER BY Game_Start, Game_Key
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS Avg_Points_Against_Prior,
    AVG(CAST(SU_Win AS FLOAT64)) OVER (
      PARTITION BY Sport, Season, Team_Norm
      ORDER BY Game_Start, Game_Key
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS WinPct_Prior_System,
    AVG(CAST(ATS_Win AS FLOAT64)) OVER (
      PARTITION BY Sport, Season, Team_Norm
      ORDER BY Game_Start, Game_Key
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS ATS_WinPct_Prior_System,
    LAG(Postgame_Total_Yards) OVER (
      PARTITION BY Sport, Season, Team_Norm ORDER BY Game_Start, Game_Key
    ) AS Prev_Total_Yards,
    LAG(Postgame_Total_Plays) OVER (
      PARTITION BY Sport, Season, Team_Norm ORDER BY Game_Start, Game_Key
    ) AS Prev_Total_Plays,
    LAG(Postgame_Pass_Yards) OVER (
      PARTITION BY Sport, Season, Team_Norm ORDER BY Game_Start, Game_Key
    ) AS Prev_Pass_Yards,
    LAG(Postgame_Rush_Yards) OVER (
      PARTITION BY Sport, Season, Team_Norm ORDER BY Game_Start, Game_Key
    ) AS Prev_Rush_Yards,
    LAG(Postgame_Turnovers) OVER (
      PARTITION BY Sport, Season, Team_Norm ORDER BY Game_Start, Game_Key
    ) AS Prev_Turnovers,
    AVG(CAST(Postgame_Total_Yards AS FLOAT64)) OVER (
      PARTITION BY Sport, Season, Team_Norm
      ORDER BY Game_Start, Game_Key
      ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) AS Avg_Total_Yards_Last3_Prior,
    AVG(CAST(Postgame_Pass_Yards AS FLOAT64)) OVER (
      PARTITION BY Sport, Season, Team_Norm
      ORDER BY Game_Start, Game_Key
      ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) AS Avg_Pass_Yards_Last3_Prior,
    AVG(CAST(Postgame_Rush_Yards AS FLOAT64)) OVER (
      PARTITION BY Sport, Season, Team_Norm
      ORDER BY Game_Start, Game_Key
      ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) AS Avg_Rush_Yards_Last3_Prior,
    AVG(CAST(Postgame_Turnovers AS FLOAT64)) OVER (
      PARTITION BY Sport, Season, Team_Norm
      ORDER BY Game_Start, Game_Key
      ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) AS Avg_Turnovers_Last3_Prior,
    AVG(CAST(Team_Score AS FLOAT64)) OVER (
      PARTITION BY Sport, Season, Team_Norm
      ORDER BY Game_Start, Game_Key
      ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ) AS Avg_Points_For_Last5_Prior
  FROM base b
),

h2h_state AS (
  SELECT
    t.*,
    LAG(SU_Win) OVER (
      PARTITION BY Sport, Team_Norm, Opponent_Norm ORDER BY Game_Start, Game_Key
    ) AS Last_Matchup_SU_Win_System,
    LAG(SU_Margin) OVER (
      PARTITION BY Sport, Team_Norm, Opponent_Norm ORDER BY Game_Start, Game_Key
    ) AS Last_Matchup_SU_Margin_System,
    DATE_DIFF(
      Game_Date,
      LAG(Game_Date) OVER (
        PARTITION BY Sport, Team_Norm, Opponent_Norm ORDER BY Game_Start, Game_Key
      ),
      DAY
    ) AS Days_Since_Last_Matchup_System
  FROM team_state t
),

with_revenge AS (
  SELECT
    h.*,
    CASE
      WHEN Last_Matchup_SU_Win_System IS NULL THEN NULL
      WHEN Last_Matchup_SU_Win_System = 0 THEN 1
      ELSE 0
    END AS Revenge_Flag_Current
  FROM h2h_state h
),

paired AS (
  SELECT
    t.*,
    o.Revenge_Flag_Current AS Opp_Revenge_Flag_Current,
    o.WinPct_Prior_System AS Opp_WinPct_Prior_System,
    o.Prev_SU_Win AS Opp_Prev_SU_Win,
    o.Prev_SU_Loss AS Opp_Prev_SU_Loss,
    o.Prev_Points_For AS Opp_Prev_Points_For,
    o.Prev_ATS_Loss AS Opp_Prev_ATS_Loss,
    o.Days_Since_Last_Game_System AS Opp_Days_Since_Last_Game_System
  FROM with_revenge t
  LEFT JOIN with_revenge o
    ON o.Season = t.Season
   AND o.Source_Game_ID = t.Source_Game_ID
   AND o.Team_Norm = t.Opponent_Norm
   AND o.Opponent_Norm = t.Team_Norm
)

SELECT
  Sport,
  Season,
  Source_Name,
  Source_Grain,
  Source_Game_ID,
  Game_Key,
  Game_Date,
  Game_Start,
  Week,
  Week_Number,
  Start_Time_ET,
  Source_Team_Name,
  Source_Opponent_Name,
  Team,
  Team_Norm,
  Canonical_Team_Name,
  Canonical_Team_Norm,
  Opponent,
  Opponent_Norm,
  Opponent_Canonical_Team_Name,
  Opponent_Canonical_Team_Norm,
  Conference,
  Opponent_Conference,
  Division,
  Opponent_Division,
  Subdivision,
  Opponent_Subdivision,
  League_Group,
  Opponent_League_Group,
  Team_Alignment_Effective_Season,
  Opponent_Alignment_Effective_Season,
  Venue,
  Is_Home,
  Is_Away,
  Is_Neutral,
  Spread_Value,
  Opening_Spread,
  Current_Total,
  Opening_Total,
  ML_Odds,
  Opening_ML_Odds,
  Spread_Move_Open_to_Close,
  Total_Move_Open_to_Close,
  Is_Spread_Favorite,
  Is_Spread_Dog,
  Is_ML_Favorite,
  Is_ML_Dog,
  Is_Regular_Season,
  Is_Conference_Game,
  Team_Game_Number,
  Prev_Opponent,
  Prev_SU_Win,
  Prev_SU_Loss,
  Prev_Points_For,
  Prev_Points_Against,
  Prev_SU_Margin,
  Prev_ATS_Cover_Margin,
  Prev_ATS_Win,
  Prev_ATS_Loss,
  Days_Since_Last_Game_System,
  Avg_Points_For_Prior,
  Avg_Points_Against_Prior,
  WinPct_Prior_System,
  ATS_WinPct_Prior_System,
  Prev_Total_Yards,
  Prev_Total_Plays,
  Prev_Pass_Yards,
  Prev_Rush_Yards,
  Prev_Turnovers,
  Avg_Total_Yards_Last3_Prior,
  Avg_Pass_Yards_Last3_Prior,
  Avg_Rush_Yards_Last3_Prior,
  Avg_Turnovers_Last3_Prior,
  Avg_Points_For_Last5_Prior,
  Last_Matchup_SU_Win_System,
  Last_Matchup_SU_Margin_System,
  Days_Since_Last_Matchup_System,
  Revenge_Flag_Current,
  Opp_Revenge_Flag_Current,
  Opp_WinPct_Prior_System,
  Opp_Prev_SU_Win,
  Opp_Prev_SU_Loss,
  Opp_Prev_Points_For,
  Opp_Prev_ATS_Loss,
  Opp_Days_Since_Last_Game_System,

  CASE
    WHEN Opening_Spread IS NULL OR Spread_Value IS NULL THEN 0
    WHEN (ABS(Opening_Spread) < 3 AND ABS(Spread_Value) > 3)
      OR (ABS(Opening_Spread) > 3 AND ABS(Spread_Value) < 3)
    THEN 1 ELSE 0
  END AS Crossed_Key_3,
  CASE
    WHEN Opening_Spread IS NULL OR Spread_Value IS NULL THEN 0
    WHEN (ABS(Opening_Spread) < 7 AND ABS(Spread_Value) > 7)
      OR (ABS(Opening_Spread) > 7 AND ABS(Spread_Value) < 7)
    THEN 1 ELSE 0
  END AS Crossed_Key_7,
  CASE
    WHEN Opening_Spread IS NULL OR Spread_Value IS NULL THEN 0
    WHEN (ABS(Opening_Spread) < 10 AND ABS(Spread_Value) > 10)
      OR (ABS(Opening_Spread) > 10 AND ABS(Spread_Value) < 10)
    THEN 1 ELSE 0
  END AS Crossed_Key_10,
  CASE
    WHEN Opening_Spread IS NULL OR Spread_Value IS NULL THEN 0
    WHEN (ABS(Opening_Spread) < 14 AND ABS(Spread_Value) > 14)
      OR (ABS(Opening_Spread) > 14 AND ABS(Spread_Value) < 14)
    THEN 1 ELSE 0
  END AS Crossed_Key_14,

  ABS(ABS(Spread_Value) - 3) AS Dist_to_3,
  ABS(ABS(Spread_Value) - 7) AS Dist_to_7,
  ABS(ABS(Spread_Value) - 10) AS Dist_to_10,
  ABS(ABS(Spread_Value) - 14) AS Dist_to_14,

  CAST(ABS(Spread_Value) = 3 AS INT64) AS Pathi_FB_On_Key_3,
  CAST(ABS(Spread_Value) = 7 AS INT64) AS Pathi_FB_On_Key_7,
  CAST(ABS(Spread_Value) = 10 AS INT64) AS Pathi_FB_On_Key_10,
  CAST(ABS(Spread_Value) = 14 AS INT64) AS Pathi_FB_On_Key_14,

  CASE
    WHEN (
      (ABS(Opening_Spread) < 3 AND ABS(Spread_Value) > 3)
      OR (ABS(Opening_Spread) > 3 AND ABS(Spread_Value) < 3)
      OR (ABS(Opening_Spread) < 7 AND ABS(Spread_Value) > 7)
      OR (ABS(Opening_Spread) > 7 AND ABS(Spread_Value) < 7)
      OR (ABS(Opening_Spread) < 10 AND ABS(Spread_Value) > 10)
      OR (ABS(Opening_Spread) > 10 AND ABS(Spread_Value) < 10)
      OR (ABS(Opening_Spread) < 14 AND ABS(Spread_Value) > 14)
      OR (ABS(Opening_Spread) > 14 AND ABS(Spread_Value) < 14)
    ) AND Spread_Value < Opening_Spread
    THEN 1 ELSE 0
  END AS Pathi_FB_Crossed_Key_Toward_Team,

  CASE
    WHEN (
      (ABS(Opening_Spread) < 3 AND ABS(Spread_Value) > 3)
      OR (ABS(Opening_Spread) > 3 AND ABS(Spread_Value) < 3)
      OR (ABS(Opening_Spread) < 7 AND ABS(Spread_Value) > 7)
      OR (ABS(Opening_Spread) > 7 AND ABS(Spread_Value) < 7)
      OR (ABS(Opening_Spread) < 10 AND ABS(Spread_Value) > 10)
      OR (ABS(Opening_Spread) > 10 AND ABS(Spread_Value) < 10)
      OR (ABS(Opening_Spread) < 14 AND ABS(Spread_Value) > 14)
      OR (ABS(Opening_Spread) > 14 AND ABS(Spread_Value) < 14)
    ) AND Spread_Value > Opening_Spread
    THEN 1 ELSE 0
  END AS Pathi_FB_Crossed_Key_Away_From_Team,

  CASE
    WHEN Opening_Spread IS NOT NULL
     AND Spread_Value IS NOT NULL
     AND ABS(Opening_Spread) NOT IN (3,7,10,14)
     AND ABS(Spread_Value) IN (3,7,10,14)
    THEN 1 ELSE 0
  END AS Pathi_FB_Moved_Onto_Key,

  CASE
    WHEN Opening_Spread IS NOT NULL
     AND Spread_Value IS NOT NULL
     AND ABS(Opening_Spread) IN (3,7,10,14)
     AND ABS(Spread_Value) NOT IN (3,7,10,14)
    THEN 1 ELSE 0
  END AS Pathi_FB_Moved_Off_Key,

  CASE
    WHEN (ABS(Opening_Spread) < 3 AND ABS(Spread_Value) > 3)
      OR (ABS(Opening_Spread) > 3 AND ABS(Spread_Value) < 3)
      OR (ABS(Opening_Spread) < 7 AND ABS(Spread_Value) > 7)
      OR (ABS(Opening_Spread) > 7 AND ABS(Spread_Value) < 7)
      OR (ABS(Opening_Spread) < 10 AND ABS(Spread_Value) > 10)
      OR (ABS(Opening_Spread) > 10 AND ABS(Spread_Value) < 10)
      OR (ABS(Opening_Spread) < 14 AND ABS(Spread_Value) > 14)
      OR (ABS(Opening_Spread) > 14 AND ABS(Spread_Value) < 14)
    THEN 1 ELSE 0
  END AS Pathi_FB_Moved_Through_Key,

  Team_Score,
  Opponent_Score,
  SU_Result,
  SU_Win,
  SU_Loss,
  SU_Margin,
  ATS_Result_Close,
  ATS_Win,
  ATS_Loss,
  ATS_Cover_Margin,
  Total_Result_Close,
  Dataset,
  Source_Opening_Odds_Raw,
  Source_Closing_Odds_Raw
FROM paired;
