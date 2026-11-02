-- READ ONLY: latest committed V5.5 publication -- no model or odds mutations.
WITH latest AS (
  SELECT snapshot_id, published_at, audit_json
  FROM `sharplogger.sharp_data.ncaaf_v55_live_board`
  WHERE record_kind='COMPLETE'
  ORDER BY published_at DESC
  LIMIT 1
), board_records AS (
  SELECT l.published_at,l.audit_json,r
  FROM latest l
  JOIN `sharplogger.sharp_data.ncaaf_v55_live_board` d
    USING(snapshot_id)
  CROSS JOIN UNNEST(JSON_QUERY_ARRAY(d.payload_json)) r
  WHERE d.record_kind='DATA'
)
SELECT
  MAX(published_at) AS latest_utc,
  COUNT(*) AS board_rows,
  MAX(JSON_VALUE(audit_json,'$.miner_registry_status')) AS miner_registry,
  MAX(JSON_VALUE(audit_json,'$.other_brain_registry_status')) AS other_brain_registry,
  MAX(JSON_VALUE(audit_json,'$.v55_live_system_evaluation.miner')) AS miner_runtime,
  MAX(JSON_VALUE(audit_json,'$.v55_live_system_evaluation.miner_report_status')) AS miner_report,
  MAX(JSON_VALUE(audit_json,'$.v55_live_system_evaluation.miner_vote_rows')) AS live_miner_vote_rows,
  MAX(JSON_VALUE(audit_json,'$.qualified_events_used')) AS qualified_events,
  MAX(JSON_VALUE(audit_json,'$.v55_raw_overlay_audit.matched_same_book_side')) AS same_book_sides,
  MAX(JSON_VALUE(audit_json,'$.v55_raw_overlay_audit.rich_rows')) AS rich_audit_rows,
  COUNTIF(JSON_VALUE(r,'$.V55_Raw_Market_Rich') IS NOT NULL AND
    JSON_VALUE(r,'$.V55_Raw_Market_Rich') NOT IN ('','NOT_PUBLISHED','NOT_VERIFIED_FOR_EXECUTION_BOOK')) AS real_rich_display_rows,
  COUNTIF(SAFE_CAST(JSON_VALUE(r,'$.V55_Any_Brain_Candidate_Count') AS INT64)>0) AS candidate_games,
  COUNTIF(SAFE_CAST(JSON_VALUE(r,'$.V55_Any_Brain_Qualified_Candidate_Count') AS INT64)>0) AS qualified_candidate_games,
  COUNTIF(SAFE_CAST(JSON_VALUE(r,'$.V55_Bet_Authority') AS INT64)>0) AS authorized_rows
FROM board_records;
