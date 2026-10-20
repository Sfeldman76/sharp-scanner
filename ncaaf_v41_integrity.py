"""NCAAF V4.1: retrospective integrity, residual diagnostics, prospective-readiness gate.

RESEARCH ONLY. Reads V4's already-published 2017-2025 out-of-fold ledger
and its immutable-by-content report; no BigQuery, no new training, no betting
or production writes. A 'PASS' here NEVER certifies vintage of CFBD inputs.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
import re
import numpy as np
import pandas as pd

VERSION = 'NCAAF_V41_HISTORICAL_INTEGRITY_20261020_V1'
SOURCE_PREFIX = 'research/ncaaf/margin_tournament_v4'
OUTPUT_PREFIX = 'research/ncaaf/v41_integrity'
DISCOVERY = tuple(range(2017, 2024))
RETRO = (2024, 2025)
SELECTED = 'NONLINEAR_STATE_HGB'
INCUMBENT = 'V3_PROGRAM_STRENGTH_RIDGE'
SEED = 20261020
EXPECTED_MIN_GAMES = 1500


def _n(frame, column):
    return pd.to_numeric(frame[column], errors='coerce').to_numpy(float)


def score(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    m = np.isfinite(y) & np.isfinite(pred)
    if not np.any(m):
        return {'n': 0, 'mae': None, 'rmse': None, 'bias_actual_minus_pred': None}
    e = y[m] - pred[m]
    return {'n': int(m.sum()), 'mae': float(np.mean(np.abs(e))),
            'rmse': float(np.sqrt(np.mean(e**2))),
            'bias_actual_minus_pred': float(np.mean(e))}


def paired_bootstrap(data, challenger, baseline, boot=500):
    """Date-cluster bootstrap; never computes separate confidence bands then subtracts."""
    y = _n(data, 'Actual_Margin')
    p = _n(data, challenger)
    b = _n(data, baseline)
    good = np.isfinite(y) & np.isfinite(p) & np.isfinite(b)
    if good.sum() < 100:
        return {'n': int(good.sum()), 'status': 'INSUFFICIENT_MATCHED'}
    sub = data.loc[good]
    y, p, b = y[good], p[good], b[good]
    cluster_labels = (sub['Season'].astype(str) + '|' + sub['Game_Date'].str[:10]).to_numpy()
    ids_by_cluster = defaultdict(list)
    for i, k in enumerate(cluster_labels):
        ids_by_cluster[k].append(i)
    clusters = [np.asarray(indices, dtype=int) for indices in ids_by_cluster.values()]
    rng = np.random.default_rng(SEED)
    a_sq, b_sq = (y-b)**2, (y-p)**2
    a_abs, b_abs = np.abs(y-b), np.abs(y-p)
    rmse_samples, mae_samples = [], []
    for _ in range(boot):
        selected = rng.integers(len(clusters), size=len(clusters))
        ix = np.concatenate([clusters[j] for j in selected])
        rmse_samples.append(float(np.sqrt(np.mean(a_sq[ix])) - np.sqrt(np.mean(b_sq[ix]))))
        mae_samples.append(float(np.mean(a_abs[ix] - b_abs[ix])))
    return {
        'status': 'EXPLORATORY_PAIRED_DATE_CLUSTER_BOOTSTRAP',
        'n': int(good.sum()), 'clusters': len(clusters),
        'rmse_gain': float(np.sqrt(np.mean(a_sq)) - np.sqrt(np.mean(b_sq))),
        'rmse_gain_ci95': [float(x) for x in np.quantile(rmse_samples, [.025, .975])],
        'mae_gain': float(np.mean(a_abs-b_abs)),
        'mae_gain_ci95': [float(x) for x in np.quantile(mae_samples, [.025, .975])],
        'not_a_fresh_holdout': True,
    }


def _require(cond, error):
    if not cond:
        raise ValueError('V41_INTEGRITY_FAIL:' + error)


def _verify(report, ledger):
    audit = report.get('audit', {})
    _require(report.get('status') == 'RESEARCH_COMPLETE_NO_PROMOTION', 'V4_INCOMPLETE')
    _require(report.get('selected_pre2024') == SELECTED, 'UNEXPECTED_SELECTED_MODEL')
    _require(report.get('source') == 'sharplogger.sharp_data.ncaaf_cfbd_archive_research_v1', 'SOURCE_MISMATCH')
    _require(audit.get('2026_labels_accessed') is False, 'ARCHIVE_AUDIT_2026_UNSEALED')
    _require(report.get('current_2026_outcomes_queried') is False, 'V4_2026_UNSEALED')
    _require(report.get('automatic_promotion') is False and report.get('production_mutated') is False, 'PRODUCTION_MUTATION')
    _require(report.get('training_dependency_on_opening_line') is False, 'MARKET_TRAINING_DEPENDENCY')
    _require(report.get('selection_uses_2024_2025') is False, 'RETRO_USED_IN_SELECTION')
    required = {'Season', 'Game_Date', 'Game_ID', 'Team', 'Opponent', 'Actual_Margin',
                SELECTED, INCUMBENT, 'DYNAMIC_MODERATE', 'NONLINEAR_HGB',
                'Verified_Open_Margin_Diagnostic'}
    _require(required.issubset(ledger), 'MISSING_LEDGER_COLUMNS:' + str(sorted(required-set(ledger))))
    _require(len(ledger) >= EXPECTED_MIN_GAMES, 'LEDGER_TOO_SMALL')
    seasons = _n(ledger, 'Season')
    _require(np.isfinite(seasons).all(), 'INVALID_SEASON')
    _require(all(y in (*DISCOVERY, *RETRO) for y in seasons), 'UNEXPECTED_SEASON_OR_2026')
    _require(all((seasons == y).sum() > 0 for y in (*DISCOVERY, *RETRO)), 'MISSING_FOLD')
    _require(np.isfinite(_n(ledger, 'Actual_Margin')).all(), 'MISSING_ACTUAL')
    date = pd.to_datetime(ledger['Game_Date'], utc=True, errors='coerce')
    _require(date.notna().all(), 'INVALID_DATES')
    _require((date.dt.year.to_numpy() == seasons).all() or
             (np.abs(date.dt.year.to_numpy() - seasons) <= 1).all(), 'DATE_SEASON_MISMATCH')
    unique = ledger['Season'].astype(str) + '|' + ledger['Game_ID'].astype(str)
    _require(unique.is_unique, 'DUPLICATE_PHYSICAL_GAMES')
    _require((ledger.Team.astype(str) != ledger.Opponent.astype(str)).all(), 'INVALID_TEAM_OPPONENT')
    _require(~ledger.Game_ID.astype(str).isin(['', 'nan', 'None', '<NA>']).any(), 'INVALID_GAME_ID')
    _require(np.isfinite(_n(ledger, SELECTED)).all(), 'MISSING_SELECTED_PREDICTIONS')
    valid = {}
    for yr in (*DISCOVERY, *RETRO):
        fold = report.get('folds', {}).get(str(yr), {})
        _require(fold.get('status') == 'PASS', f'FOLD_NOT_PASS_{yr}')
        _require(fold.get('train_end') == yr-1, f'INVALID_TRAIN_END_{yr}')
        sub = ledger.loc[seasons == yr]
        _require(len(sub) == fold.get('valid'), f'LEDGER_FOLD_ROWS_{yr}')
        for m in (SELECTED, INCUMBENT, 'DYNAMIC_MODERATE', 'NONLINEAR_HGB'):
            published = fold.get('models', {}).get(m, {})
            measured = score(_n(sub, 'Actual_Margin'), _n(sub, m))
            _require(measured['n'] == published.get('n'), f'SAMPLE_COUNT_{yr}_{m}')
            _require(abs(measured['rmse']-published.get('rmse', float('nan'))) < 1e-5,
                     f'RMSE_MISMATCH_{yr}_{m}')
            _require(abs(measured['mae']-published.get('mae', float('nan'))) < 1e-5,
                     f'MAE_MISMATCH_{yr}_{m}')
        valid[str(yr)] = len(sub)
    discovery_rank = report.get('discovery_rank', [])
    ranking = sorted([x for x in discovery_rank if x.get('status') == 'ELIGIBLE'],
                     key=lambda x: (x.get('pooled_rmse', float('inf')), x.get('model', '')))
    _require(bool(ranking) and ranking[0]['model'] == SELECTED, 'MODEL_NOT_SELECTED_FROM_DISCOVERY')
    return {'status': 'PASS_REPRODUCED_LEDGER_METRICS', 'n': len(ledger), 'fold_rows': valid,
            'source_audit_2026_sealed': True, 'opening_lines_only_diagnostics': True,
            'archive_point_in_time_certified': False}


def _summarize_groups(df, keys, minimum=25):
    out = []
    for grouping, sub in df.groupby(keys, observed=True, dropna=False):
        if len(sub) < minimum:
            continue
        rec = dict(zip(keys, grouping if isinstance(grouping, tuple) else (grouping,)))
        rec.update(score(_n(sub, 'Actual_Margin'), _n(sub, SELECTED)))
        rec['rmse_v3'] = score(_n(sub, 'Actual_Margin'), _n(sub, INCUMBENT))['rmse']
        rec['mean_prediction'] = float(np.mean(_n(sub, SELECTED)))
        rec['mean_actual'] = float(np.mean(_n(sub, 'Actual_Margin')))
        out.append(rec)
    return out


def _bin_abs(p):
    return pd.cut(np.abs(p), [-.01, 3, 7, 14, 21, 35, float('inf')],
                  labels=['0–3', '3–7', '7–14', '14–21', '21–35', '35+'], include_lowest=True)


def _calibration_table(df):
    """Prediction magnitude is a legitimate *pregame* stratum; not outcome-selected."""
    rows = []
    for s, part in df.groupby('forecast_absolute_margin_bucket', observed=True):
        y = _n(part, 'Actual_Margin'); p = _n(part, SELECTED)
        rows.append({'bucket': str(s), 'n': len(part),
                     'predicted_abs_mean': float(np.mean(np.abs(p))),
                     'actual_abs_mean': float(np.mean(np.abs(y))),
                     'mean_signed_residual': float(np.mean(y-p)),
                     'rmse': score(y, p)['rmse']})
    return rows


def _source_vintage_gate(report):
    """Never infer source-level point-in-time provenance from a *_Prior suffix."""
    a = report.get('audit', {})
    return {'status': 'HOLD_SOURCE_VINTAGE_NOT_CERTIFIED',
            'reported_v4_value': a.get('pregame_point_in_time_certification'),
            'individual_cfbd_feature_lineage_attested': False,
            'required_evidence': [
                'Original raw event/season data ingestion and publication timestamps',
                'Per-game as-of join reproducing every CFBD Prior/Matchup feature from strictly earlier observations',
                'Same-day game ordering, postponed-game handling, and cross-season reset checks',
                'Independent source-vintage review of roster/drive/advanced fields, even when named Prior',
                'Recomputed rolling folds from independently versioned point-in-time snapshot',
            ],
            'does_not_approve_production_or_betting': True}


def _leakage_canary():
    """Executable algorithmic invariance checks; not source-lineage certification."""
    import ncaaf_margin_tournament_v4 as v4
    games = pd.DataFrame([
        (2024, '2024-09-01', 'A', 'B', 10, 48),
        (2024, '2024-09-01', 'C', 'D', -4, 42),
        (2024, '2024-09-08', 'A', 'C', 3, 45),
        (2024, '2024-09-08', 'B', 'D', 7, 53),
        (2024, '2024-09-15', 'A', 'D', 9, 57),
        (2025, '2025-09-01', 'A', 'B', 6, 40),
    ], columns=['Season', 'Game_Date', 'Team_Norm', 'Opponent_Norm', 'Actual_Margin', 'Actual_Total'])
    games['Is_Home'] = 1
    original = v4.dynamic_features(games)
    changed = games.copy()
    changed.loc[0, ['Actual_Margin', 'Actual_Total']] = [70, 140]
    alt = v4.dynamic_features(changed)
    early = [0, 1]
    def same(a, b):
        return np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True, atol=0, rtol=0)
    _require(same(original.iloc[early], alt.iloc[early]), 'SAME_DAY_OUTCOME_LEAK')
    _require(not same(original.iloc[[2]], alt.iloc[[2]]), 'FUTURE_UPDATE_NOT_APPLIED')
    future = games.copy()
    future.loc[5, ['Actual_Margin', 'Actual_Total']] = [110, 190]
    _require(same(original.iloc[:5], v4.dynamic_features(future).iloc[:5]), 'FUTURE_SEASON_OUTCOME_LEAK')
    early_future = games.copy()
    early_future.loc[4, ['Actual_Margin', 'Actual_Total']] = [110, 190]
    _require(same(original.iloc[:4], v4.dynamic_features(early_future).iloc[:4]), 'FUTURE_DATE_OUTCOME_LEAK')
    return {'status': 'PASS_SYNTHETIC_DYNAMIC_STATE_CANARIES',
            'same_day_snapshot': 'PASS', 'future_date_independence': 'PASS',
            'future_season_independence': 'PASS',
            'source_data_vintage_certified': False}


def analyze(report, ledger, *, bootstrap=500, log_func=lambda msg: None):
    checks = _verify(report, ledger)
    canary = _leakage_canary()
    df = ledger.copy()
    df['Season'] = pd.to_numeric(df.Season).astype(int)
    df['Date_UTC'] = pd.to_datetime(df.Game_Date, utc=True)
    df['Error_Selected'] = _n(df, 'Actual_Margin') - _n(df, SELECTED)
    df['Error_V3'] = _n(df, 'Actual_Margin') - _n(df, INCUMBENT)
    df['Abs_Error_Selected'] = df.Error_Selected.abs()
    # Week-band proxy: first observed date per season, not an official football week.
    first_date = df.groupby('Season')['Date_UTC'].transform('min')
    day_in_season = (df.Date_UTC - first_date).dt.days
    df['Season_Time_Proxy'] = pd.cut(day_in_season, [-1, 27, 62, 999],
          labels=['First 28 calendar days', 'Days 29–63', 'Day 64 onward'])
    df['forecast_absolute_margin_bucket'] = _bin_abs(_n(df, SELECTED))
    df['forecast_side'] = np.where(_n(df, SELECTED) >= 0, 'Forecast Team Positive', 'Forecast Opponent Positive')
    retro = df[df.Season.isin(RETRO)].copy()
    open_ok = np.isfinite(_n(retro, 'Verified_Open_Margin_Diagnostic'))
    match = retro.loc[open_ok].copy()
    model_vs_market = paired_bootstrap(match, SELECTED, 'Verified_Open_Margin_Diagnostic', boot=bootstrap)
    if len(match):
        match['model_minus_open'] = _n(match, SELECTED) - _n(match, 'Verified_Open_Margin_Diagnostic')
        match['disagreement_bucket'] = _bin_abs(_n(match, 'model_minus_open'))
        market_disagreement = _summarize_groups(match, ['disagreement_bucket'], minimum=30)
    else:
        market_disagreement = []
    pooled = paired_bootstrap(retro, SELECTED, INCUMBENT, boot=bootstrap)
    year_pairs = {str(y): paired_bootstrap(retro.loc[retro.Season == y], SELECTED, INCUMBENT,
                                         boot=bootstrap) for y in RETRO}
    # No 2026 history or grader. 2024–25 have previously influenced decisions.
    gate = _source_vintage_gate(report)
    diagnostics = {
        'version': VERSION, 'generated_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'RETROSPECTIVE_AUDITED__SOURCE_VINTAGE_HOLD',
        'interpretation': 'Retrospective only; NOT a new holdout or prospective result',
        'selected_model': SELECTED, 'incumbent_model': INCUMBENT,
        'integrity': checks, 'state_filter_canaries': canary,
        'source_point_in_time_gate': gate,
        'retrospective_game_count': len(retro),
        'retrospective_paired_v3': pooled, 'retrospective_year_pairs': year_pairs,
        'opening_line_matched_n': len(match),
        'opening_line_matched_metrics': {
            'model': score(_n(match, 'Actual_Margin'), _n(match, SELECTED)),
            'market': score(_n(match, 'Actual_Margin'), _n(match, 'Verified_Open_Margin_Diagnostic')),
            'paired_model_gain_over_market': model_vs_market,
        },
        'by_season': _summarize_groups(df, ['Season']),
        'by_season_time_proxy': _summarize_groups(retro, ['Season', 'Season_Time_Proxy']),
        'by_forecast_strength': _summarize_groups(retro, ['forecast_absolute_margin_bucket']),
        'by_forecast_side': _summarize_groups(retro, ['forecast_side']),
        'forecast_magnitude_calibration': _calibration_table(retro),
        'market_disagreement_diagnostic': market_disagreement,
        'team_diagnostics_warning': 'Descriptive only; overlapping repeated team appearances, no selection authority',
        'worst_teams_min_12_games': sorted(_summarize_groups(retro, ['Team'], minimum=12),
                                         key=lambda x: -x['rmse'])[:15],
        'production_mutated': False, 'production_authority': 0,
        'bet_authority': 0, 'prospective_2026_prediction_count': 0,
        'prospective_status': 'NOT_STARTED_REQUIRE_PREGAME_DATA_AND_LOCKED_MODEL_ARTIFACT',
    }
    residual_cols = ['Season', 'Game_Date', 'Game_ID', 'Team', 'Opponent', 'Actual_Margin',
                     SELECTED, INCUMBENT, 'Verified_Open_Margin_Diagnostic', 'Error_Selected',
                     'Error_V3', 'Abs_Error_Selected', 'Season_Time_Proxy',
                     'forecast_absolute_margin_bucket', 'forecast_side']
    log_func(f'[NCAAF-V41-INTEGRITY] verified={checks["n"]} years={len(checks["fold_rows"])}')
    log_func(f'[NCAAF-V41-RETRO] n={pooled["n"]} rmse_gain={pooled["rmse_gain"]:.4f} ci={pooled["rmse_gain_ci95"]}')
    log_func(f'[NCAAF-V41-OPEN] matched={len(match)} model_minus_market_gain={model_vs_market.get("rmse_gain")}')
    log_func('[NCAAF-V41-GATE] source_vintage=HOLD prospective_2026=NOT_STARTED production_mutation=FALSE')
    return diagnostics, df.loc[:, residual_cols]


def run_cloud_audit(*, storage_client, bucket_name='sharp-models', log_func=print):
    """Isolated job path: read two V4 artifacts, audit, write research-only report."""
    bucket = storage_client.bucket(bucket_name)
    raw = bucket.blob(f'{SOURCE_PREFIX}/current_report.json').download_as_bytes()
    csv = bucket.blob(f'{SOURCE_PREFIX}/game_level_predictions.csv').download_as_bytes()
    report = json.loads(raw)
    ledger = pd.read_csv(BytesIO(csv), low_memory=False)
    result, residuals = analyze(report, ledger, log_func=log_func)
    result['input_sha256'] = {'v4_report': sha256(raw).hexdigest(),
                             'v4_predictions': sha256(csv).hexdigest()}
    blob_path = f'{OUTPUT_PREFIX}/runs/{result["input_sha256"]["v4_predictions"][:16]}/audit.json'
    result['immutable_research_report_uri'] = f'gs://{bucket_name}/{blob_path}'
    data = json.dumps(result, indent=2, allow_nan=False, sort_keys=True, default=str)
    # Repeated analysis on the same inputs is allowed, and does not alter old runs.
    run_prefix = blob_path.rsplit('/', 1)[0]
    try:
        bucket.blob(blob_path).upload_from_string(data, content_type='application/json', if_generation_match=0)
    except Exception as exc:
        if type(exc).__name__ != 'PreconditionFailed':
            raise
        log_func(f'[NCAAF-V41] immutable report already exists: {blob_path}')
    res_blob = bucket.blob(f'{run_prefix}/retrospective_game_errors.csv')
    try:
        res_blob.upload_from_string(residuals.to_csv(index=False), content_type='text/csv', if_generation_match=0)
    except Exception as exc:
        if type(exc).__name__ != 'PreconditionFailed':
            raise
    # Mutable convenience pointer exclusively under V4.1 research namespace.
    bucket.blob(f'{OUTPUT_PREFIX}/latest_audit.json').upload_from_string(data, content_type='application/json')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Audit V4 report and predictions without retraining')
    parser.add_argument('--report', required=True, help='Existing V4 report JSON')
    parser.add_argument('--predictions', required=True, help='Existing V4 per-game CSV')
    parser.add_argument('--output', default='v41_audit.json', help='Research-only audit output JSON')
    args = parser.parse_args()
    with open(args.report, encoding='utf-8') as stream: report = json.load(stream)
    frame = pd.read_csv(args.predictions)
    result, errors = analyze(report, frame, log_func=print)
    with open(args.output, 'w', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False, sort_keys=True, default=str)
    errors.to_csv(os.path.splitext(args.output)[0] + '_errors.csv', index=False)
    print('V41_AUDIT_WRITTEN', args.output)
