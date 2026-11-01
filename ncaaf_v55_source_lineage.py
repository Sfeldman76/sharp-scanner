"""V5.5 read-only CFBD / market source-to-research lineage reconciliation.

Runs in Heavy Research against the ACTUAL original research matrices.  Does not
fill missing values, merge sources, manufacture timestamps, choose features,
train models, adjust grades or change betting authority.  Source-vintage
certification is intentionally not inferred from column existence.
"""
from __future__ import annotations

from datetime import datetime, timezone
from collections import Counter
import json
import re
from typing import Any

import pandas as pd

VERSION = 'NCAAF-V55-SOURCE-LINEAGE-20261101-R1'
BLOB = 'research/ncaaf/v55/data_integrity/source_lineage_latest.json'
IDENT = re.compile(r'^[A-Za-z_][A-Za-z_0-9]*$')
TABLE = re.compile(r'^[A-Za-z_][\w-]*\.[A-Za-z_][\w]*\.[A-Za-z_][\w]*$')
MARKET_TABLES = ('sharplogger.sharp_data.moves_with_features_merged',
                 'sharplogger.sharp_data.sharp_moves_master')
DEFAULT_CFBD = 'sharplogger.sharp_data.ncaaf_cfbd_miner_features_postgame_v1'


def _empty_stats():
    return {'column_present': False, 'raw_non_null': 0, 'numeric_non_null': 0,
            'dtype': None, 'by_season_raw_non_null': {}}


def _stats(frame, column, seasons):
    """Observe actual values, not numeric-only coverage (the earlier audit bug)."""
    if frame is None or column not in frame.columns:
        return _empty_stats()
    series = frame[column]
    raw = int(series.notna().sum())
    try:
        numeric = int(pd.to_numeric(series, errors='coerce').notna().sum())
    except (TypeError, ValueError):
        numeric = 0
    by_year = {}
    if seasons is not None and len(seasons) == len(series):
        ys = pd.to_numeric(pd.Series(seasons).reset_index(drop=True), errors='coerce')
        valid = series.reset_index(drop=True).notna()
        for year in sorted(ys.dropna().astype(int).unique()):
            mask = ys.eq(year)
            by_year[str(year)] = int((mask & valid).sum())
    return {'column_present': True, 'raw_non_null': raw, 'numeric_non_null': numeric,
            'dtype': str(series.dtype), 'by_season_raw_non_null': by_year}


def _read_schema(bq_client, table):
    """Metadata-only source check: do not fall back to guessing column names."""
    if bq_client is None:
        return None, 'BQ_CLIENT_UNAVAILABLE'
    if not TABLE.fullmatch(table):
        return None, 'INVALID_TABLE_IDENTIFIER'
    try:
        obj = bq_client.get_table(table)
        return {str(x.name).lower(): {'name': str(x.name), 'type': str(x.field_type)}
                for x in obj.schema}, None
    except Exception as exc:
        return None, 'SCHEMA_QUERY_' + type(exc).__name__


def _query_counts(bq_client, table, names, *, min_season=2013, max_season=2025):
    """One read-only grouped query across existing source fields, historical rows only."""
    if not names or bq_client is None:
        return {}, None if not names else 'BQ_CLIENT_UNAVAILABLE'
    if not TABLE.fullmatch(table) or any(not IDENT.fullmatch(x) for x in names):
        return {}, 'INVALID_COLUMN_IDENTIFIER'
    expressions = [f'COUNTIF(`{name}` IS NOT NULL) AS `f{i}`' for i, name in enumerate(names)]
    sql = (f'SELECT CAST(Season AS INT64) AS season, COUNT(*) AS source_rows, '
           + ', '.join(expressions) + f' FROM `{table}` '
           + f'WHERE SAFE_CAST(Season AS INT64) BETWEEN {min_season} AND {max_season} '
           + 'GROUP BY season ORDER BY season')
    try:
        result = bq_client.query(sql)
        if hasattr(result, 'result'):
            rows = [dict(x.items()) for x in result.result()]
        else:
            rows = result.to_dataframe().to_dict('records')
        output = {name: {} for name in names}
        for row in rows:
            year = str(int(row['season']))
            for i, name in enumerate(names):
                output[name][year] = {'source_rows': int(row['source_rows']),
                                      'non_null': int(row[f'f{i}'])}
        return output, None
    except Exception as exc:
        return {}, 'COUNT_QUERY_' + type(exc).__name__


def _query_market_historical_counts(bq_client, table, schema, columns):
    """Read *historical* presence from the bounded enriched view only.

    This does NOT certify pregame timestamp provenance and does not scan the
    potentially much larger raw sharp_moves_master event log automatically.
    """
    if not columns or schema is None or bq_client is None:
        return {}, None if not columns else 'MARKET_SOURCE_UNVERIFIED'
    date_meta = schema.get('game_date')
    if date_meta is None:
        return {}, 'MARKET_SOURCE_HAS_NO_GAME_DATE_COLUMN'
    date_name = date_meta['name']
    if not IDENT.fullmatch(date_name) or any(not IDENT.fullmatch(x) for x in columns):
        return {}, 'INVALID_COLUMN_IDENTIFIER'
    expressions = [f'COUNTIF(`{name}` IS NOT NULL) AS `f{i}`'
                   for i,name in enumerate(columns)]
    sql = (f'SELECT '+', '.join(expressions)+f' FROM `{table}` WHERE '
           f'SAFE_CAST(`{date_name}` AS DATE) BETWEEN DATE \'2013-01-01\' '
           f'AND DATE \'2025-12-31\'')
    try:
        res = bq_client.query(sql)
        rows = [dict(x.items()) for x in res.result()] if hasattr(res, 'result') else res.to_dataframe().to_dict('records')
        record = rows[0] if rows else {}
        return {column: int(record.get(f'f{i}') or 0) for i,column in enumerate(columns)}, None
    except Exception as exc:
        return {}, 'MARKET_COUNT_QUERY_'+type(exc).__name__


def _market_frame_status(core, miner, column, seasons):
    # Market Rich may be a separately computed live signal, not a CORE feature.
    return {'core': _stats(core, column, seasons),
            'miner': _stats(miner, column, seasons)}


def reconcile_source_lineage(*, games, miner_games, lineage_report,
                             market_report=None, bq_client=None,
                             cfbd_table=DEFAULT_CFBD, market_tables=MARKET_TABLES,
                             storage_client=None, bucket_name='sharp-models',
                             persist=False, log_func=print):
    """Classify reported gaps without touching the input matrices or selection.

    FAIL: source has non-null historical values but the corresponding actual
    Miner matrix column is completely null.  This includes join losses and
    accidental downstream feature loss; it does not automatically repair them.
    UNKNOWN: a warehouse source could not be inspected, or vintage is unproven.
    """
    if games is None or miner_games is None or len(games) != len(miner_games):
        raise ValueError('CORE_MINER_ROW_ALIGNMENT_REQUIRED')
    if not isinstance(lineage_report, dict):
        raise ValueError('LINEAGE_REPORT_REQUIRED')
    years = (games['Season'] if 'Season' in games.columns else None)
    missing = sorted({str(x['field']) for x in (lineage_report.get('fields') or [])
                      if x.get('status') == 'MINER_COLUMN_PRESENT_BUT_UNPOPULATED'
                      and str(x.get('field', '')).startswith('CFBD_')})
    source_schema, schema_error = _read_schema(bq_client, cfbd_table)
    actual_names = sorted({source_schema[n.lower()]['name'] for n in missing
                           if source_schema is not None and n.lower() in source_schema})
    source_counts, count_error = _query_counts(bq_client, cfbd_table, actual_names)
    fields = []
    for name in missing:
        ms = _stats(miner_games, name, years)
        cs = _stats(games, name, years)
        meta = source_schema.get(name.lower()) if source_schema is not None else None
        source_name = meta['name'] if meta else None
        by_year = source_counts.get(source_name, {}) if source_name else {}
        source_observed = sum(x['non_null'] for x in by_year.values())
        # Never equate a nonnumeric string with an empty field.
        if ms['raw_non_null'] > 0:
            status = ('AUDIT_NUMERIC_COERCION_FALSE_ALARM'
                      if ms['numeric_non_null'] == 0 else 'HAS_DATA_IN_RESEARCH_MATRIX')
            severity = 'PASS'
        elif source_schema is None or (source_name and count_error):
            status = 'SOURCE_UNVERIFIED'; severity = 'UNKNOWN'
        elif source_name is None:
            status = 'FIELD_ABSENT_FROM_SELECTED_CFBD_SOURCE'; severity = 'WARNING'
        elif source_observed > 0:
            status = 'SOURCE_HAS_HISTORICAL_VALUES_BUT_MINER_EMPTY'; severity = 'FAIL'
        else:
            status = 'SOURCE_HAS_NO_NON_NULL_HISTORICAL_VALUES'; severity = 'WARNING'
        fields.append({'field': name, 'status': status, 'severity': severity,
                       'core_matrix': cs, 'miner_matrix': ms,
                       'warehouse': {'table': cfbd_table, 'field': source_name,
                                     'data_type': meta['type'] if meta else None,
                                     'by_season': by_year,
                                     'historical_non_null': source_observed,
                                     'verification_error': count_error or schema_error}})
    wanted = list((market_report or {}).get('historical_fields_requested') or [])
    market_schemas = {}
    for table in market_tables:
        schema, error = _read_schema(bq_client, table)
        market_schemas[table] = {'columns': schema, 'error': error}
    # Bound the market count probe to the small enriched view, never raw ticks.
    enriched = market_tables[0] if market_tables else None
    enriched_schema = market_schemas.get(enriched, {}).get('columns') if enriched else None
    enriched_names = sorted({enriched_schema[n.lower()]['name'] for n in wanted
                             if enriched_schema is not None and n.lower() in enriched_schema})
    market_historical_counts, market_historical_error = _query_market_historical_counts(
        bq_client, enriched, enriched_schema, enriched_names) if enriched else ({}, 'MARKET_VIEW_NOT_CONFIGURED')
    market_fields = []
    for name in wanted:
        frames = _market_frame_status(games, miner_games, name, years)
        holders = [table for table, sc in market_schemas.items()
                   if sc['columns'] is not None and name.lower() in sc['columns']]
        verified = all(sc['columns'] is not None for sc in market_schemas.values())
        if frames['core']['raw_non_null'] or frames['miner']['raw_non_null']:
            state, severity = 'RESEARCH_MATRIX_HAS_VALUES', 'PASS'
        elif holders:
            enriched_name = (enriched_schema.get(name.lower()) or {}).get('name') if enriched_schema else None
            if enriched_name is not None and market_historical_error is None:
                historical_observed = market_historical_counts.get(enriched_name, 0)
                if historical_observed:
                    state, severity = 'HISTORICAL_WAREHOUSE_VALUES_EXIST__PREGAME_VINTAGE_UNVERIFIED', 'WARNING'
                else:
                    state, severity = 'ENRICHED_VIEW_HAS_NO_HISTORICAL_VALUES', 'WARNING'
            else:
                state, severity = 'SOURCE_SCHEMA_HAS_COLUMN__HISTORICAL_VALUES_AND_VINTAGE_UNVERIFIED', 'UNKNOWN'
        elif verified:
            state, severity = 'FIELD_NOT_STORED_IN_CHECKED_MARKET_TABLES', 'WARNING'
        else:
            state, severity = 'MARKET_WAREHOUSE_SCHEMA_UNVERIFIED', 'UNKNOWN'
        market_fields.append({'field': name, 'status': state, 'severity': severity,
                              'matrix': frames, 'warehouse_tables_with_field': holders,
                              'enriched_historical_non_null': market_historical_counts.get(
                                  (enriched_schema.get(name.lower()) or {}).get('name')) if enriched_schema else None,
                              'historical_source_error': market_historical_error})
    counts = Counter(x['severity'] for x in fields + market_fields)
    statuses = Counter(x['status'] for x in fields + market_fields)
    status = ('FAIL' if counts['FAIL'] else 'WARNING' if counts['WARNING'] else
              'UNKNOWN' if counts['UNKNOWN'] else 'PASS')
    report = {'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'status': status, 'production_authority': 0, 'production_mutation': False,
              'feature_mutation': False, 'outcome_2026_selection': False,
              'source_certification': 'NOT_CERTIFIED_WITHOUT_ORIGINAL_PREGAME_VINTAGE',
              'source_cfbd_table': cfbd_table,
              'source_cfbd_schema_error': schema_error,
              'source_cfbd_count_error': count_error,
              'market_schema_errors': {k:v['error'] for k,v in market_schemas.items()},
              'market_historical_count_error': market_historical_error,
              'summary': {'total_cfbd_flagged': len(fields),
                          'market_fields_requested': len(market_fields),
                          'severity_counts': {s: counts.get(s, 0) for s in ('PASS','WARNING','FAIL','UNKNOWN')},
                          'status_counts': dict(sorted(statuses.items()))},
              'cfbd_fields': fields, 'market_rich_fields': market_fields}
    if persist and storage_client is not None:
        try:
            storage_client.bucket(bucket_name).blob(BLOB).upload_from_string(
                json.dumps(report, sort_keys=True, default=str), content_type='application/json')
            report['artifact_uri'] = f'gs://{bucket_name}/{BLOB}'
        except Exception as exc:
            report['artifact_publication_error'] = type(exc).__name__ + ':' + str(exc)
    log_func('[NCAAF-V55-SOURCE-LINEAGE] '+json.dumps({
        'status': status, 'summary': report['summary'],
        'cfbd_source_schema_error': schema_error,
        'cfbd_count_error': count_error, 'artifact': report.get('artifact_uri')},sort_keys=True))
    return report
