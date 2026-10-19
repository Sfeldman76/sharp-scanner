"""Research-only, season-forward evaluation of independent football vs market forecasts.

The 2013--2025 first stage comes exclusively from the existing strictly
season-forward, market-BLIND score foundation. Original opening market quotes
may be used only for the second-stage, prior-years-trained meta model.

Outputs on matched games:
  Margin: football, authentic open, market-aware ridge.
  Totals: football, authentic opening total, market-aware ridge.
  H2H:   football probability, opening-spread-calibrated win probability,
         market + football logits.
  Baseline head score metrics over ALL seasons where OOF available, including
  years without original opening prices.

No 2024/25 outcomes used for recipe selection. Those years have been repeatedly
reviewed and are retrospective stress tests, not pristine heldouts.
2026 outcomes prohibited. No bet/ROI assertions; no model promotion.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

VERSION = 'NCAAF_MULTIHEAD_HISTORY_TO_MARKET_V1_20261018'
META_MIN_TRAIN = 100
META_MIN_TEST = 40
SELECTION_YEARS = (2022, 2023)
RETROSPECTIVE_YEARS = (2024, 2025)
MIN_AUTHENTIC_YEAR = 2021


def _num(g: pd.DataFrame, col: str) -> np.ndarray:
    if col not in g:
        return np.full(len(g), np.nan, dtype=float)
    return pd.to_numeric(g[col], errors='coerce').to_numpy(float)


def _regression(y, pred):
    m = np.isfinite(y) & np.isfinite(pred)
    if not m.any():
        return {'n': 0, 'rmse': None, 'mae': None}
    d = y[m] - pred[m]
    return {'n': int(m.sum()), 'rmse': float(np.sqrt(np.mean(d*d))), 'mae': float(np.mean(np.abs(d)))}


def _binary(y_margin, pred):
    m = np.isfinite(y_margin) & np.isfinite(pred) & (y_margin != 0)
    if not m.any():
        return {'n': 0, 'brier': None, 'logloss': None}
    labels = (y_margin[m] > 0).astype(float)
    p = np.clip(pred[m], .00001, .99999)
    return {'n': int(m.sum()), 'brier': float(np.mean((labels-p)**2)),
            'logloss': float(np.mean(-labels*np.log(p) - (1-labels)*np.log(1-p)))}


def _paired(y, market, candidate, *, binary=False):
    m = np.isfinite(y) & np.isfinite(market) & np.isfinite(candidate)
    if binary:
        m &= y != 0
    metric = _binary if binary else _regression
    a = metric(y[m], market[m]); b = metric(y[m], candidate[m])
    name = 'brier' if binary else 'rmse'
    return {'n': int(m.sum()), 'market': a, 'candidate': b,
            f'{name}_gain_vs_market': None if not m.any() else a[name]-b[name]}


def _genuine_total(g):
    """Only original opening TOTAL, never an observed close as a fake open.

    If a normalized Market_Open_Total exists, cross-check original source.
    A missing original source is unavailable, not a zero or an inferred line.
    """
    year = _num(g, 'Season')
    raw = _num(g, 'Consensus_Open_Total')
    if 'Market_Open_Total' in g.columns:
        norm = _num(g, 'Market_Open_Total')
        raw = np.where(np.isclose(raw, norm, atol=1e-5, rtol=0), raw, np.nan)
    valid = np.isfinite(raw) & (raw > 15) & (raw < 110) & np.isfinite(year) & (year >= MIN_AUTHENTIC_YEAR) & (year <= 2025)
    return np.where(valid, raw, np.nan)


def _genuine_margin(g):
    year = _num(g, 'Season')
    normalized = _num(g, 'Market_Open_Margin')
    original = _num(g, 'Consensus_Open_Spread')
    valid = (np.isfinite(year) & (year >= MIN_AUTHENTIC_YEAR) & (year <= 2025) &
             np.isfinite(normalized) & np.isfinite(original) &
             np.isclose(normalized, -original, atol=1e-5, rtol=0) & (np.abs(normalized) <= 80))
    return np.where(valid, normalized, np.nan)


def _logit(p):
    p = np.clip(p, 1e-5, 1-1e-5)
    return np.log(p/(1-p))


def _ridge(tr_x, tr_y, va_x, *, alpha=50.0):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    good_rows = np.isfinite(tr_y) & np.isfinite(tr_x).any(axis=1)
    if int(good_rows.sum()) < META_MIN_TRAIN:
        return np.full(len(va_x), np.nan)
    tr_x, tr_y = tr_x[good_rows], tr_y[good_rows]
    means = np.nanmean(tr_x, axis=0)
    good = np.isfinite(means)
    if not good.any():
        return np.full(len(va_x), np.nan)
    a, b = tr_x[:,good].copy(), va_x[:,good].copy()
    means = means[good]
    a = np.where(np.isfinite(a), a, means)
    b = np.where(np.isfinite(b), b, means)
    std = np.nanstd(a, axis=0)
    useful = std > 1e-8
    if not useful.any():
        return np.full(len(va_x), np.nan)
    sc = StandardScaler()
    a = sc.fit_transform(a[:, useful]); b = sc.transform(b[:, useful])
    model = Ridge(alpha=alpha)
    model.fit(a, tr_y)
    return model.predict(b)


def _h2h_fit(tr_x, tr_y, va_x):
    """Regularized binary forecast, not a bookmaker moneyline quote."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if len(tr_y) < META_MIN_TRAIN or len(set(tr_y)) < 2:
        return np.full(len(va_x), np.nan)
    mean = np.nanmean(tr_x, axis=0)
    keep = np.isfinite(mean)
    if not keep.any():
        return np.full(len(va_x), np.nan)
    a, b = tr_x[:,keep].copy(), va_x[:,keep].copy()
    mean = mean[keep]
    a = np.where(np.isfinite(a), a, mean)
    b = np.where(np.isfinite(b), b, mean)
    good = np.std(a,axis=0) > 1e-8
    if not good.any():
        return np.full(len(va_x), np.nan)
    scaler = StandardScaler()
    a = scaler.fit_transform(a[:,good]); b = scaler.transform(b[:,good])
    clf = LogisticRegression(C=.2, max_iter=300, solver='lbfgs')
    clf.fit(a, tr_y)
    return clf.predict_proba(b)[:,1]


def _head_report(g, first_oof, head, market, *, log_func):
    seasons = _num(g,'Season')
    if head=='margin':
        actual = _num(g,'Actual_Margin')
        baseline = np.asarray(first_oof['margin'],float)
    elif head=='total':
        actual = _num(g,'Actual_Total')
        baseline = np.asarray(first_oof['total'],float)
    else:
        actual = _num(g,'Actual_Margin')
        baseline = np.asarray(first_oof['h2h'],float)
    binary = head=='h2h'
    metric = _binary if binary else _regression
    all_oof = metric(actual,baseline)
    mask = np.isfinite(actual)&np.isfinite(baseline)&np.isfinite(market)
    if binary:mask &= actual != 0
    by_year = {}
    selection_candidates = {'MARKET_PLUS_FOOTBALL': [], 'MARKET_PLUS_FOOTBALL_WITH_DIFFERENCE': []}
    for year in (*SELECTION_YEARS,*RETROSPECTIVE_YEARS):
        earlier = mask & (seasons < year)
        tested = mask & (seasons == year)
        y = actual[tested]
        if int(earlier.sum()) < META_MIN_TRAIN or int(tested.sum()) < META_MIN_TEST:
            by_year[str(year)]={'status':'SOURCE_LIMITED','train':int(earlier.sum()),'eval':int(tested.sum())}
            log_func(f'[NCAAF-MULTIHEAD-FOLD] head={head} year={year} status=SOURCE_LIMITED prior={int(earlier.sum())} eval={int(tested.sum())}')
            continue
        f_tr, f_va = baseline[earlier], baseline[tested]
        m_tr, m_va = market[earlier], market[tested]
        market_open_forecast = None
        if not binary:
            # Predict the *opening market*, without observing that game's quote.
            # This estimates market expectation; it does not itself predict
            # the final score or justify a wager.
            market_open_forecast = _ridge(f_tr.reshape(-1,1),m_tr,f_va.reshape(-1,1))
        if binary:
            # Bookmaker-derived probability must be CALIBRATED from earlier
            # observed straight-up outcomes, never assumed from odds we lack.
            labels = (actual[earlier]>0).astype(int)
            market_only = _h2h_fit(m_tr.reshape(-1,1),labels,m_va.reshape(-1,1))
            joint_simple = _h2h_fit(np.column_stack([m_tr,_logit(f_tr)]),labels,
                                     np.column_stack([m_va,_logit(f_va)]))
            joint_diff = _h2h_fit(np.column_stack([m_tr,_logit(f_tr),m_tr*_logit(f_tr)/10]),labels,
                                   np.column_stack([m_va,_logit(f_va),m_va*_logit(f_va)/10]))
            # H2H market is a SPREAD-DERIVED probability, not a moneyline.
            market_label='SPREAD_DERIVED_SU_PROBABILITY__NOT_MONEYLINE_PRICE'
        else:
            market_only = m_va
            residual = actual[earlier] - m_tr
            joint_simple = m_va + _ridge((f_tr-m_tr).reshape(-1,1),residual,
                                         (f_va-m_va).reshape(-1,1))
            joint_diff = m_va + _ridge(np.column_stack([f_tr-m_tr,f_tr]),residual,
                                       np.column_stack([f_va-m_va,f_va]))
            market_label='AUTHENTIC_OPENING_PRICE'
        variants={'MARKET_PLUS_FOOTBALL':joint_simple,
                  'MARKET_PLUS_FOOTBALL_WITH_DIFFERENCE':joint_diff}
        o = {'status':'PASS', 'train':int(earlier.sum()),'eval':int(tested.sum()),
             'football_only':metric(y,f_va), 'market_only':metric(y,market_only),
             'market_definition':market_label,
             'market_open_forecast_diagnostic':(
                 {'target':'GENUINE_ORIGINAL_OPENING_PRICE',
                  'prediction_vs_actual_open':_regression(m_va,market_open_forecast),
                  'current_game_quote_used_to_predict':False,
                  'status':'DIAGNOSTIC_ONLY'} if not binary else
                 {'status':'NOT_APPLICABLE_H2H_USES_SPREAD_CALIBRATED_PROBABILITY'}),
             'variants':{k: {'metrics':metric(y,p),
                            'paired_vs_market':_paired(y,market_only,p,binary=binary),
                            'paired_vs_football':_paired(y,f_va,p,binary=binary)}
                         for k,p in variants.items()}}
        by_year[str(year)]=o
        for name,record in o['variants'].items():
            val = record['paired_vs_market'].get('brier_gain_vs_market' if binary else 'rmse_gain_vs_market')
            if year in SELECTION_YEARS and val is not None:
                selection_candidates[name].append({'year':year,'n':int(tested.sum()),'gain':val})
        log_func(f'[NCAAF-MULTIHEAD-FOLD] head={head} year={year} n={int(tested.sum())} '
                 f'football={o["football_only"]} market={o["market_only"]} '
                 f'market_plus_football={o["variants"]["MARKET_PLUS_FOOTBALL"]["metrics"]} authority=0')
    preselected=[]
    for name,records in selection_candidates.items():
        # No post-2023 experiment can influence selection.
        if len(records)==len(SELECTION_YEARS) and all(x['gain']>0 for x in records):
            weighted = sum(x['n']*x['gain'] for x in records)/sum(x['n'] for x in records)
            preselected.append((weighted,name))
    preselected.sort(reverse=True)
    selected=preselected[0][1] if preselected else None
    return {'head':head,'status':'PASS' if any(y.get('status')=='PASS' for y in by_year.values()) else 'SOURCE_LIMITED',
            'score_only_oof_all_available_history':all_oof,
            'market_eligible':int(mask.sum()),
            'market_eligible_by_year':{str(yr):int(np.sum(mask&(seasons==yr)))
                                       for yr in sorted(set(seasons[np.isfinite(seasons)].astype(int)))},
            'selection_years':list(SELECTION_YEARS), 'retrospective_years':list(RETROSPECTIVE_YEARS),
            'selected_from_pre2024_only':selected, 'by_year':by_year,
            'comparisons':'MATCHED_PHYSICAL_GAMES_BY_HEAD',
            'betting_authority':0, 'probability_promotion':False}


def run_multihead_historical_market_research(g:pd.DataFrame, oof:dict, *, log_func=print)->dict[str,Any]:
    """Pass the same canonical, game-level Core frame and full-history OOF sink."""
    if g is None or g.empty:raise ValueError('EMPTY_HISTORICAL_GAMES')
    year=_num(g,'Season')
    if (np.isfinite(year)&(year>=2026)).any():raise ValueError('YEAR_2026_OUTCOMES_SEALED')
    n=len(g)
    for head in ('margin','total','h2h'):
        if head not in oof or len(oof[head])!=n:
            raise ValueError('FULL_HISTORY_OOF_MISALIGNED_'+head)
    if oof.get('market_predictor_used') is not False:
        raise ValueError('FULL_HISTORY_OOF_MARKET_BLIND_NOT_PROVEN')
    # Source-canonical row order must agree with the previous stage. Avoid
    # comparing arrays to a differently filtered/reordered history frame.
    if oof.get('fingerprint'):
        from ncaaf_full_history_foundation import frame_fingerprint
        if oof['fingerprint'] != frame_fingerprint(g):
            raise ValueError('FULL_HISTORY_OOF_FINGERPRINT_MISMATCH')
    markets={'margin':_genuine_margin(g),'total':_genuine_total(g),'h2h':_genuine_margin(g)}
    heads={h:_head_report(g,oof,h,markets[h],log_func=log_func) for h in ('margin','total','h2h')}
    report={'version':VERSION,'status':'PASS' if heads['margin']['status']=='PASS' else 'SOURCE_LIMITED',
            'source_seasons':[int(y) for y in sorted(set(year[np.isfinite(year)].astype(int)))],
            'games':n,'heads':heads,'score_foundation_market_blind':True,
            'market_lineage':'GENUINE_OPENING_QUOTES_ONLY__NO_CLOSING_PROXY',
            'market_h2h':'SPREAD_DERIVED_SU_BASELINE_NOT_A_MONEYLINE',
            'source_limited_head_not_imputed':True,
            'pre2024_selection_only':True,
            'confirmation_status':'2024_2025_RETROSPECTIVE_NOT_PRISTINE',
            'prospective_from_season':2026, 'automatic_promotion':False,
            'production_authority':0,'production_mutation':False,
            'bet_authority_modified':False,
            'next':'Run on 2013-2025 BigQuery historical frame; compare both pre2024 and 2024/25 head metrics; prospective shadow required.'}
    log_func('[NCAAF-MULTIHEAD-HISTORICAL-MARKET] '
             f'status={report["status"]} scored={n} '
             f'margin_open={heads["margin"]["market_eligible"]} '
             f'totals_open={heads["total"]["market_eligible"]} '
             f'h2h_market={heads["h2h"]["market_eligible"]} '
             f'selection={ {h:r["selected_from_pre2024_only"] for h,r in heads.items()} } '
             'production_authority=0')
    return report
