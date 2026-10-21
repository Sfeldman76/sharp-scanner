"""NCAAF V4.3: read-only validation of three preregistered V4.2 challengers.

Does NOT refit models, use market as predictor, read 2026 actual outcomes,
change production, or grant promotion. Consumes the immutable V4.2 game ledger.
Prior 2024-2025 inspection and uncertified CFBD publication times remain HOLDS.
"""
from __future__ import annotations
import json
import math
import hashlib
from datetime import datetime, timezone
from itertools import combinations
import numpy as np
import pandas as pd

VERSION = 'NCAAF_V43_FOCUSED_ARTIFACT_VALIDATION_20261020_V1'
BASE = 'NONLINEAR_STATE_HGB'
CANDIDATES = ('A_EFFICIENCY', 'AB_EFF_PASS', 'AC_EFF_DRIVE')
REQUIRED = ('Season', 'Game_Date', 'Game_ID', 'Actual_Margin', BASE) + CANDIDATES
DISCOVERY = tuple(range(2017, 2024))
RETROSPECTIVE = (2024, 2025)
BOOT = 1500
SEED = 20261020


def _metric(y, p):
    a = np.asarray(y, dtype=float); b = np.asarray(p, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    if not good.any(): return {'n': 0, 'rmse': None, 'mae': None, 'bias': None}
    d = b[good] - a[good]
    return {'n': int(good.sum()), 'rmse': float(np.sqrt(np.mean(d*d))),
            'mae': float(np.mean(np.abs(d))), 'bias': float(np.mean(d))}


def _cluster_boot(df, a, b, *, n_boot=BOOT, seed=SEED):
    """RMSE gain a-minus-b; resample complete GAME DATES, not individual bets."""
    y = df['Actual_Margin'].to_numpy(dtype=float)
    pa = df[a].to_numpy(dtype=float); pb = df[b].to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb)
    d = df.loc[mask, 'Game_Date'].astype(str).to_numpy()
    yy, aa, bb = y[mask], pa[mask], pb[mask]
    if len(yy) < 30:
        return {'status': 'INSUFFICIENT_PAIRED_GAMES', 'n': int(len(yy))}
    # Date-grouped sum of squared errors, then resample clusters.
    labels, inv = np.unique(d, return_inverse=True)
    k = len(labels)
    if k < 10: return {'status': 'INSUFFICIENT_DATE_CLUSTERS', 'clusters': k}
    counts = np.bincount(inv, minlength=k).astype(float)
    ss_a = np.bincount(inv, weights=(yy-aa)**2, minlength=k)
    ss_b = np.bincount(inv, weights=(yy-bb)**2, minlength=k)
    rng = np.random.default_rng(seed)
    gains = np.empty(n_boot, float)
    for i in range(n_boot):
        ix = rng.integers(k, size=k)
        denom = counts[ix].sum()
        gains[i] = math.sqrt(ss_a[ix].sum()/denom) - math.sqrt(ss_b[ix].sum()/denom)
    observed = float(math.sqrt(np.mean((yy-aa)**2)) - math.sqrt(np.mean((yy-bb)**2)))
    return {'status': 'EXPLORATORY_BOOTSTRAP_NOT_FRESH_HOLDOUT', 'n': int(len(yy)),
            'clusters': k, 'rmse_gain_a_minus_b': observed,
            'ci95': [float(x) for x in np.quantile(gains, [.025, .975])],
            'bootstrap_probability_gain_nonpositive': float(np.mean(gains <= 0)),
            'cluster_unit': 'Game_Date', 'bootstrap_repetitions': n_boot}


def _holm(pvalues):
    """Holm-Bonferroni adjustment for 3 *exploratory* vs-base comparisons."""
    pairs = sorted(pvalues.items(), key=lambda kv: kv[1])
    n = len(pairs); ret = {}; prev = 0.0
    for rank, (key, p) in enumerate(pairs):
        prev = max(prev, min(1.0, p*(n-rank)))
        ret[key] = prev
    return ret


def validate(report, raw_ledger, *, boot=BOOT):
    if report.get('status') != 'EXPLORATORY_RESEARCH__SOURCE_VINTAGE_HOLD':
        raise RuntimeError('V43_UNEXPECTED_V42_REPORT_STATUS')
    if report.get('source_vintage_certified') is not False:
        raise RuntimeError('V43_UNEXPECTED_SOURCE_CERTIFICATION_STATE')
    if any(report.get('parity_to_published_v4',{}).get(str(y)) != 'EXACT_REPRODUCTION'
           for y in DISCOVERY + RETROSPECTIVE):
        raise RuntimeError('V43_NOT_ALL_NINE_V4_FOLDS_HAVE_PARITY')
    if any(report.get('parity_to_selected_v4_champion',{}).get(str(y)) != 'EXACT_REPRODUCTION'
           for y in DISCOVERY + RETROSPECTIVE):
        raise RuntimeError('V43_SELECTED_V4_CHAMPION_NOT_REPRODUCED')
    if report.get('frozen_champion') not in ('NONLINEAR_STATE_HGB','NONLINEAR_HGB'):
        raise RuntimeError('V43_V4_SELECTED_CHAMPION_UNSUPPORTED')
    ledger = raw_ledger.copy()
    missing = [c for c in REQUIRED if c not in ledger]
    if missing: raise RuntimeError('V43_MISSING_LEDGER_COLS:'+','.join(missing))
    ledger['Season'] = pd.to_numeric(ledger['Season'],errors='coerce')
    if ledger['Season'].isna().any() or (ledger['Season'] >= 2026).any():
        raise RuntimeError('V43_PROSPECTIVE_SEAL_OR_BAD_SEASON')
    if ledger['Game_Date'].isna().any(): raise RuntimeError('V43_MISSING_GAME_DATES')
    ledger['Game_ID'] = ledger['Game_ID'].astype(str)
    if ledger.duplicated(['Season','Game_ID']).any():
        raise RuntimeError('V43_DUPLICATED_GAME_ID')
    for c in ('Actual_Margin', BASE) + CANDIDATES:
        ledger[c] = pd.to_numeric(ledger[c],errors='coerce')
    if any(y not in set(ledger['Season'].astype(int)) for y in DISCOVERY + RETROSPECTIVE):
        raise RuntimeError('V43_MISSING_VALIDATION_SEASONS')
    # Reconcile same-year n/RMSE to the published source report; fail closed.
    parity = {}
    for y in DISCOVERY + RETROSPECTIVE:
        subset = ledger.loc[ledger.Season.eq(y)]
        for name in (BASE,)+CANDIDATES:
            got = _metric(subset.Actual_Margin, subset[name]); prior = report['folds'][str(y)]['models'][name]
            if got['n'] != prior['n'] or got['rmse'] is None or abs(got['rmse']-prior['rmse'])>1e-6:
                raise RuntimeError(f'V43_SOURCE_LEDGER_PARITY_{y}_{name}')
        parity[str(y)] = 'EXACT_V42_PER_GAME_RECONCILIATION'

    metrics = {}
    for name, years in [('screen_2017_2020',(2017,2018,2019,2020)),
                        ('confirmation_2021_2023',(2021,2022,2023)),
                        ('discovery_2017_2023',DISCOVERY),
                        ('retrospective_2024_2025',RETROSPECTIVE)]:
        cut = ledger.loc[ledger.Season.isin(years)]
        metrics[name] = {m:_metric(cut.Actual_Margin,cut[m]) for m in (BASE,)+CANDIDATES}

    retrospective = ledger.loc[ledger.Season.isin(RETROSPECTIVE)]
    paired = {}
    for i, (a,b) in enumerate(combinations((BASE,)+CANDIDATES,2)):
        result = _cluster_boot(retrospective,a,b,n_boot=boot,seed=SEED+i)
        paired[f'{a}__minus__{b}'] = result
    raw_p = {m: min(1.0,2*min(paired[f'{BASE}__minus__{m}']['bootstrap_probability_gain_nonpositive'],
                                  1-paired[f'{BASE}__minus__{m}']['bootstrap_probability_gain_nonpositive']))
             for m in CANDIDATES}
    adjusted = _holm(raw_p)

    # Direct pairwise tests matter more than tiny differences against base.
    per_year = {}
    for y in DISCOVERY+RETROSPECTIVE:
        part=ledger[ledger.Season.eq(y)]
        per_year[str(y)] = {m:_metric(part.Actual_Margin,part[m]) for m in (BASE,)+CANDIDATES}
    # Error strata use *only baseline prediction* to avoid hindsight-defined categories.
    mag = np.abs(retrospective[BASE].to_numpy(float))
    bins = ((0,7,'near_even_0_7'),(7,17,'moderate_7_17'),(17,np.inf,'mismatch_17_plus'))
    strata = {}
    for lo,hi,label in bins:
        cut = retrospective.loc[(mag>=lo)&(mag<hi)]
        strata[label] = {m:_metric(cut.Actual_Margin,cut[m]) for m in (BASE,)+CANDIDATES}
    spec = report.get('experiment_spec_sha256')
    coverage = {str(y):{k:report['folds'][str(y)].get('coverage',{}).get(k,{})
                        for k in ('A_EFFICIENCY','B_PASS_RUSH_MATCHUPS','C_DRIVES_FIELD')}
                for y in DISCOVERY+RETROSPECTIVE}
    return {
        'version':VERSION,'generated_utc':datetime.now(timezone.utc).isoformat(),
        'status':'EXPLORATORY_FOCUSED_VALIDATION__SOURCE_VINTAGE_HOLD',
        'reference_v42_spec_sha256':spec,'source_v42_report_status':report['status'],
        'parity_to_published_v42':parity,'candidate_labels':list(CANDIDATES),
        'benchmark':BASE,'pooled_accuracy':metrics,
        'per_year':per_year,'forecast_strength_strata_2024_2025':strata,
        'paired_2024_2025':paired,
        'exploratory_bootstrap_two_sided_p_unadjusted':raw_p,
        'exploratory_holm_three_comparisons':adjusted,
        'source_family_coverage_by_fold':coverage,
        'source_vintage_certification':'HOLD_PENDING_RAW_PUBLICATION_TIMESTAMPS',
        'historical_holdout':'2024_2025_PREVIOUSLY_INSPECTED_NOT_FRESH',
        'interpretation':'All statistical inference is exploratory; 2024-2025 has been inspected and 3 candidates preselected.',
        'leader_not_promoted':'A_EFFICIENCY__PARSIMONIOUS_RESEARCH_PRIORITY_NOT_AN_AUTHORIZED_WINNER',
        'prod_v1_mutation':False,'promotion_authority':0,
        'future_requirement':'Real, timestamped pregame 2026 forecasts from fitted frozen inference models, not retrospective rescores.'}


def _read_source_artifacts(bucket):
    """Locate IMMUTABLE V4.2 versioned report+CSV pair corresponding to latest pointer."""
    latest = json.loads(bucket.blob('research/ncaaf/conditional_features_v42/latest_report.json').download_as_text())
    root = 'research/ncaaf/conditional_features_v42/runs/'
    compatible = []
    for blob in bucket.list_blobs(prefix=root):
        if not blob.name.endswith('/report.json'): continue
        candidate = json.loads(blob.download_as_text())
        if candidate.get('generated_utc') == latest.get('generated_utc') and candidate.get('experiment_spec_sha256') == latest.get('experiment_spec_sha256'):
            compatible.append((blob.name,candidate))
    if len(compatible)!=1:raise RuntimeError('V43_CANNOT_UNIQUELY_PAIR_VERSIONED_V42_ARTIFACTS')
    name, report = compatible[0]
    predictions = name[:-len('report.json')]+'game_predictions.csv'
    from io import StringIO
    csv = bucket.blob(predictions).download_as_text()
    return report,pd.read_csv(StringIO(csv),low_memory=False),name,predictions


def run_cloud_audit(*,storage_client,bucket_name='sharp-models',log_func=print):
    bucket = storage_client.bucket(bucket_name)
    source,ledger,rpath,lpath = _read_source_artifacts(bucket)
    out = validate(source,ledger)
    out['v42_immutable_report']=f'gs://{bucket_name}/{rpath}'
    out['v42_immutable_predictions']=f'gs://{bucket_name}/{lpath}'
    runid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    prefix=f'research/ncaaf/focused_validation_v43/runs/{runid}'
    payload=json.dumps(out,indent=2,sort_keys=True,allow_nan=False)
    bucket.blob(prefix+'/report.json').upload_from_string(payload,content_type='application/json')
    bucket.blob('research/ncaaf/focused_validation_v43/latest_report.json').upload_from_string(payload,content_type='application/json')
    log_func('[NCAAF-V43] status={} baseline={} candidates={} parity_folds={} production_authority=0'.format(
        out['status'],BASE,','.join(CANDIDATES),len(out['parity_to_published_v42'])))
    return out


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--v42-report',required=True);p.add_argument('--v42-ledger',required=True)
    p.add_argument('--output',required=True);opt=p.parse_args()
    with open(opt.v42_report) as f:report=json.load(f)
    r=validate(report,pd.read_csv(opt.v42_ledger,low_memory=False))
    with open(opt.output,'w') as f:json.dump(r,f,indent=2,allow_nan=False)
    print('V43_FOCUSED_VALIDATION',r['status'])
