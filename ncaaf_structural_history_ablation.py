"""NCAAF research-only structural audit and matched history-window ablation.

Two controlled tracks:
  FIT_WINDOW_ONLY: 2/5/all-year model fits over identical full-history as-of ratings.
  RESET_STATE_WITH_WINDOW: also rebuild all as-of ratings using only the window.
Both use the SAME feature names, estimator and validation games inside a fold.
The second track measures the complete impact of withholding old history,
including opponent-adjusted latent team state; the first isolates fit sample size.
No market data are predictors. Selection never uses 2024+ outcomes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

VERSION = "NCAAF_STRUCTURE_HISTORY_WINDOW_AUDIT_20261018_V1"
FIRST = 2013
LAST = 2025
WINDOWS = (2, 5, None)
HEADS = ("margin", "total", "h2h")
MIN_TRAIN = 200
MIN_VALID = 50
MAX_FIELDS = 36


def _metrics(y, p, head):
    from ncaaf_hierarchical_research import _metric
    return _metric(y, p, head)


def _objective(m, head):
    return m.get("logloss" if head == "h2h" else "rmse")


def _audit_source(frame):
    from ncaaf_hierarchical_research import _num, _game_dates, _get_identity
    from ncaaf_full_history_foundation import _safe_feature
    year = _num(frame, "Season")
    margin = _num(frame, "Actual_Margin").to_numpy(float)
    total = _num(frame, "Actual_Total").to_numpy(float)
    legal = np.isfinite(margin) & np.isfinite(total)
    impossible = legal & ((total < np.abs(margin) - 1e-6) | (total < 0))
    date = _game_dates(frame)
    home = _get_identity(frame, ("Team_Norm", "Team", "Source_Team_Name", "Home_Team_Norm"))
    away = _get_identity(frame, ("Opponent_Norm", "Opponent", "Opp_Norm", "Away_Team_Norm"))
    goodids = home.ne("") & away.ne("") & home.ne(away)
    records = []
    for y in sorted(set(year.dropna().astype(int))):
        idx = year.eq(y).to_numpy()
        records.append({"year": y, "games": int(idx.sum()), "scores_finite": int(legal[idx].sum()),
                        "date_available": int(date.notna().to_numpy()[idx].sum()),
                        "valid_team_pairs": int(goodids.to_numpy()[idx].sum()),
                        "impossible_score_geometry": int(impossible[idx].sum())})
    feature_leak_candidates = [x for x in frame.columns if any(s in str(x).lower() for s in
                              ("postgame", "current_game", "final_score"))]
    return {"status": "PASS" if not impossible.any() else "SCORE_CONSISTENCY_WARNING",
            "rows": int(len(frame)), "by_year": records,
            "impossible_total_lt_abs_margin": int(impossible.sum()),
            "feature_names_with_postgame_tokens_present_but_not_used": feature_leak_candidates[:40],
            "current_game_outcome_as_predictor": False,
            "note": "Identity/date coverage and score geometry are diagnostic, not proof of alias or margin orientation correctness."}


def _cols_for_fold(g, seasons, year, candidates):
    """Common features selected ONLY from the shortest window's prior data.

    No outcome-driven screening and no retrospective-period feature selection.
    A narrow common schema can under-use deep data, deliberately making the
    history-length ablation interpretable instead of mixing feature sets.
    """
    from ncaaf_full_history_foundation import _safe_feature
    from ncaaf_hierarchical_research import _num
    train = (seasons >= year - 2) & (seasons < year)
    priority = ["Context_Intercept", *[c for c in g if c.startswith("Dyn_")], *candidates]
    cols=[]
    for c in dict.fromkeys(priority):
        if c not in g or (not c.startswith("Dyn_") and not _safe_feature(c)):
            continue
        values = _num(g.loc[train], c)
        if c != "Context_Intercept" and (values.notna().sum() < max(15, .07 * train.sum()) or
                                            values.nunique(dropna=True) < 2):
            continue
        cols.append(c)
        if len(cols) >= MAX_FIELDS:
            break
    return cols


def _predictions(g, cols, train, valid, margin, total):
    from ncaaf_hierarchical_research import _fit_predict
    Xtr = g.loc[train, cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    Xva = g.loc[valid, cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    weights = np.ones(train.sum(), dtype=float)
    return {h: _fit_predict(Xtr, Xva, margin[train] if h != "total" else total[train], h, weights)
            for h in HEADS}


def run_structural_history_ablation(frame, candidate_features, *, log_func=print,
                                    min_train=MIN_TRAIN, min_valid=MIN_VALID):
    from ncaaf_hierarchical_research import (_num, reconstruct_asof_team_strength)
    from ncaaf_full_history_foundation import _safe_feature
    if frame is None or frame.empty:
        raise ValueError("STRUCTURAL_ABLATION_EMPTY_FRAME")
    g = frame.copy().reset_index(drop=True)
    year = _num(g, "Season").to_numpy(float)
    if (np.isfinite(year) & (year > LAST)).any():
        raise ValueError("STRUCTURAL_ABLATION_2026_SEAL")
    if not (np.isfinite(year) & (year <= 2013)).any():
        raise ValueError("STRUCTURAL_ABLATION_HISTORICAL_BASE_MISSING")
    margin = _num(g, "Actual_Margin").to_numpy(float)
    total = _num(g, "Actual_Total").to_numpy(float)
    g["Context_Intercept"] = 1.0
    audit = _audit_source(g)
    # As-of ratings from all seasons, computed BEFORE considering outcomes in
    # the current game; valid as the shared-feature counterfactual.
    dyn_all, dyn_audit = reconstruct_asof_team_strength(g)
    full = g.copy()
    for c in dyn_all:
        full[c] = dyn_all[c].to_numpy(float)
    safe_candidates = list(dict.fromkeys([c for c in candidate_features if c in full and _safe_feature(c)]))
    result_rows = []
    scores = {mode: {w: {h: np.full(len(g), np.nan) for h in HEADS} for w in WINDOWS}
              for mode in ("FIT_WINDOW_ONLY", "RESET_STATE_WITH_WINDOW")}
    for y in sorted(set(int(z) for z in year[np.isfinite(year)])):
        if y <= 2015: continue
        val = (year == y)
        if val.sum() < min_valid: continue
        cols = _cols_for_fold(full, year, y, safe_candidates)
        if not cols or not any(c.startswith("Dyn_") for c in cols):
            log_func(f"[NCAAF-STRUCTURAL-HISTORY-FOLD] year={y} status=NO_DYNAMIC_FEATURES")
            continue
        for window in WINDOWS:
            tr = (year < y) & ((year >= y - window) if window is not None else (year >= FIRST))
            if np.isfinite(margin[tr]).sum() < min_train or np.isfinite(total[tr]).sum() < min_train:
                continue
            label = "all" if window is None else str(window)
            for mode in scores:
                if mode == "FIT_WINDOW_ONLY":
                    features_frame = full
                else:
                    # The older results cannot enter even latent team state.
                    limited = g.loc[tr | val].reset_index(drop=False)
                    dyn, state_audit = reconstruct_asof_team_strength(limited)
                    features_frame = g.copy()
                    for c in dyn:
                        values = np.full(len(g), np.nan)
                        values[limited["index"].to_numpy(dtype=int)] = dyn[c].to_numpy(float)
                        features_frame[c] = values
                pred = _predictions(features_frame, cols, tr, val, margin, total)
                for h in HEADS:
                    scores[mode][window][h][val] = pred[h]
                record = {"validation_year": y, "mode": mode, "train_years": label,
                          "train_games": int(tr.sum()), "validation_games": int(val.sum()),
                          "features": cols, "feature_count": len(cols),
                          "heads": {h: _metrics(margin[val] if h != "total" else total[val], pred[h], h)
                                    for h in HEADS}}
                result_rows.append(record)
                log_func(f"[NCAAF-STRUCTURAL-HISTORY-FOLD] year={y} mode={mode} years={label} "
                         f"train={tr.sum()} val={val.sum()} features={len(cols)} "
                         f"margin_rmse={record['heads']['margin']['rmse']} "
                         f"total_rmse={record['heads']['total']['rmse']} "
                         f"h2h_logloss={record['heads']['h2h']['logloss']} authority=0")
    # Pre-2024 scores alone determine any research preference. No 2024/25
    # data can alter the selected window or the predictors for earlier folds.
    by_head={}
    for mode in scores:
        by_head[mode]={}
        for h in HEADS:
            actual=margin if h!="total" else total
            selection = (year>=2018)&(year<=2023)
            holdout = (year>=2024)&(year<=2025)
            chosen=[]
            for window in WINDOWS:
                m=_metrics(actual[selection],scores[mode][window][h][selection],h)
                v=_objective(m,h)
                if v is not None:
                    chosen.append((v, WINDOWS.index(window), window))
            selected=min(chosen)[2] if chosen else None
            out={"selected_pre2024": ("all" if selected is None else str(selected)) if chosen else None,
                 "selected_supported": bool(chosen),
                 "selection": {"all" if w is None else str(w): _metrics(actual[selection], scores[mode][w][h][selection],h)
                               for w in WINDOWS},
                 "retrospective": {"all" if w is None else str(w): _metrics(actual[holdout], scores[mode][w][h][holdout],h)
                                   for w in WINDOWS},
                 "paired_by_year": {}}
            for yr in (2024,2025):
                ix=year==yr
                # Compare ALL windows on the exact same physical games.
                present=np.logical_and.reduce([np.isfinite(scores[mode][w][h]) for w in WINDOWS]) & ix & np.isfinite(actual)
                if h=="h2h": present &= ~np.isclose(actual,0.)
                out["paired_by_year"][str(yr)]={"games":int(present.sum()),
                    "scores": {"all" if w is None else str(w): _metrics(actual[present],scores[mode][w][h][present],h)
                               for w in WINDOWS}}
            by_head[mode][h]=out
    # This is a structural experiment ONLY. Any years/hyperparameters evaluated
    # previously are retrospective diagnostics and cannot promote production.
    report={"version":VERSION,"status":"PASS" if result_rows else "SOURCE_LIMITED",
            "source_audit":audit,"full_history_dynamic_audit":dyn_audit,
            "modes":by_head,"folds":result_rows,"production_authority":0,
            "automatic_promotion":False,"selection_years":list(range(2018,2024)),
            "retrospective_years":[2024,2025],"market_predictors_used":False,
            "model_contract":"SAME_FEATURE_SCHEMA_AND_ESTIMATOR_ACROSS_WINDOWS_WITHIN_YEAR",
            "explanation":{"FIT_WINDOW_ONLY":"Old results retained only in pregame team-rating state; training examples vary.",
                           "RESET_STATE_WITH_WINDOW":"Old results absent from training and team-rating state."}}
    log_func(f"[NCAAF-STRUCTURAL-HISTORY-ABLATION] status={report['status']} "
             f"folds={len(result_rows)} selection={ {m:{h:val['selected_pre2024'] for h,val in heads.items()} for m,heads in by_head.items()} } "
             f"score_geometry_errors={audit['impossible_total_lt_abs_margin']} production_authority=0")
    return report
