-- Create once in the sharplogger project, sharp_data dataset (BigQuery Standard SQL).
-- V5.5 is independent of frozen Production V1 predictions/results tables.
CREATE TABLE IF NOT EXISTS `sharplogger.sharp_data.ncaaf_v55_live_board` (
  snapshot_id STRING NOT NULL,
  published_at TIMESTAMP NOT NULL,
  record_kind STRING NOT NULL,  -- DATA or COMPLETE
  payload_json STRING,         -- exact V5.5 2026 DataFrame as JSON
  audit_json STRING,           -- original V5.5 source-status and provenance
  row_count INT64,
  payload_sha256 STRING,
  publication_type STRING      -- WEEKLY or MARKET_REFRESH
)
PARTITION BY DATE(published_at)
CLUSTER BY record_kind, publication_type;

-- Most recent committed V5.5 snapshot and whether its backing DATA is present.
WITH commits AS (
  SELECT * FROM `sharplogger.sharp_data.ncaaf_v55_live_board`
  WHERE record_kind = 'COMPLETE'
  QUALIFY ROW_NUMBER() OVER (ORDER BY published_at DESC) = 1
)
SELECT c.snapshot_id,c.published_at,c.publication_type,c.row_count,
       TIMESTAMP_DIFF(CURRENT_TIMESTAMP(),c.published_at,MINUTE) AS age_minutes,
       JSON_VALUE(c.audit_json,'$.status') AS source_status,
       SAFE_CAST(JSON_VALUE(c.audit_json,'$.priced_games') AS INT64) AS priced_games,
       COUNTIF(d.record_kind='DATA' AND d.payload_sha256=c.payload_sha256) AS matching_data_rows
FROM commits c LEFT JOIN `sharplogger.sharp_data.ncaaf_v55_live_board` d
  ON d.snapshot_id=c.snapshot_id AND d.record_kind='DATA'
GROUP BY c.snapshot_id,c.published_at,c.publication_type,c.row_count,c.audit_json;

-- Fresh NCAAF bookmaker prices on the backend, for debugging V5.5 priced_games=0.
-- This diagnoses odds existence; it does NOT substitute prices into model output.
SELECT COUNT(*) AS market_rows,
       COUNT(DISTINCT CONCAT(CAST(Game_Start AS STRING),'|',
                     LOWER(CAST(Home_Team_Norm AS STRING)),'|',
                     LOWER(CAST(Away_Team_Norm AS STRING)))) AS priced_matchups,
       COUNTIF(LOWER(CAST(Market AS STRING))='spreads'
         AND SAFE_CAST(Value AS FLOAT64) IS NOT NULL
         AND SAFE_CAST(Odds_Price AS FLOAT64) IS NOT NULL) AS spread_price_rows,
       MAX(Snapshot_Timestamp) AS newest_quote
FROM `sharplogger.sharp_data.sharp_moves_master`
WHERE UPPER(TRIM(CAST(Sport AS STRING)))='NCAAF'
  AND Game_Start > CURRENT_TIMESTAMP()
  AND Snapshot_Timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(),INTERVAL 24 HOUR);
