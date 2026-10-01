"""NFL Research V2.0.2 — frozen PBP CORE2 attribution diagnostic.

This module answers one question only: does the already-frozen PBP CORE2 artifact
contain independent betting information that the protected incumbent CORE/market
stack does not already contain?

Hard contracts
--------------
* The frozen PBP registry SHA is fixed and verified before any analysis.
* PBP CORE2 is NEVER refit, tuned, calibrated, promoted, or overwritten here.
* Stored PBP OOF predictions and stored prior-context features are reused.
* 2026 is sealed: every BigQuery query is constrained to 2017-2025 and every
  downloaded frozen artifact declares year_2026_queried=false.
* Incumbent CORE remains the protected baseline and may be deterministically
  regenerated only to recreate its historical season-forward OOF predictions.
* No diagnostic result has production authority and no historical close is
  represented as an executable price or CLV observation.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from nfl_feature_audit_v1 import VIEW
from nfl_challenger_v1 import build_readonly_query, physical_games
from nfl_intelligence_v1 import _generate_oof_core
from nfl_pbp_foundation_v1 import match_pbp_to_authoritative, build_model_matrix
from nfl_research_v2_contract import assert_contract, contract_hash

SOURCE_TAG = "nfl-pbp-attribution-v1-research-v2.0.2-frozen-core2-20261001"
STATUS = "NFL_RESEARCH_V2_PBP_ATTRIBUTION_DIAGNOSTIC_COMPLETE"
PRODUCTION_AUTHORITY = 0
SEALED_SEASON = 2026

FROZEN_PBP_REGISTRY_SHA256 = "741474dc5ca410d32572275c8df08b7c1fd11745f8db81406626953d5f96b710"
FROZEN_PBP_PREFIX = f"nfl-research/v2_0/pbp/{FROZEN_PBP_REGISTRY_SHA256[:16]}"
FROZEN_SYSTEM_REGISTRY_SHA256 = "b9fc3d469248f40f580211f5e5decc792976c90596a0b09abd38e279c46cbc81"
FROZEN_SYSTEM_PREFIX = f"nfl-research/v2_0/system_lab/{FROZEN_SYSTEM_REGISTRY_SHA256[:16]}"

EDGE_BINS = (-np.inf, 1.0, 2.0, 3.0, 4.0, 6.0, np.inf)
EDGE_LABELS = ("0-1", "1-2", "2-3", "3-4", "4-6", "6+")


# ------------------------------- small helpers --------------------------------
def _num(s: Any, index=None) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce")
    if index is None:
        return pd.to_numeric(pd.Series(s), errors="coerce")
    return pd.to_numeric(pd.Series(s, index=index), errors="coerce")


def _finite_corr(a: Any, b: Any) -> float | None:
    x = np.asarray(pd.to_numeric(pd.Series(a), errors="coerce"), float)
    y = np.asarray(pd.to_numeric(pd.Series(b), errors="coerce"), float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) <= 1e-12 or np.std(y[m]) <= 1e-12:
        return None
    return round(float(np.corrcoef(x[m], y[m])[0, 1]), 6)


def _round(v: Any, nd: int = 6):
    try:
        f = float(v)
        return round(f, nd) if math.isfinite(f) else None
    except Exception:
        return v


def _wilson(w: int, n: int, z: float = 1.959963984540054) -> list[float | None]:
    if n <= 0:
        return [None, None]
    p = w / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / den
    return [round(max(0.0, center - half), 6), round(min(1.0, center + half), 6)]


def _side_record(selection_margin: Any) -> dict:
    m = np.asarray(pd.to_numeric(pd.Series(selection_margin), errors="coerce"), float)
    valid = np.isfinite(m) & (np.abs(m) > 1e-9)
    vv = m[valid]
    w = int((vv > 0).sum())
    l = int((vv < 0).sum())
    n = w + l
    roi = ((w * 100.0 - l * 110.0) / (n * 110.0)) if n else math.nan
    return {
        "n": n,
        "wins": w,
        "losses": l,
        "hit_rate": round(w / n, 6) if n else None,
        "wilson95": _wilson(w, n),
        "flat_minus110_roi_reference": round(roi, 6) if n else None,
    }


def _mae(y: Any, p: Any) -> float | None:
    a = np.asarray(pd.to_numeric(pd.Series(y), errors="coerce"), float)
    b = np.asarray(pd.to_numeric(pd.Series(p), errors="coerce"), float)
    m = np.isfinite(a) & np.isfinite(b)
    return round(float(np.mean(np.abs(a[m] - b[m]))), 6) if m.any() else None


def _safe_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, default=str, indent=2, allow_nan=False).encode()


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return round(f, 10) if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _download_bytes(storage_client, bucket_name: str, key: str) -> bytes:
    blob = storage_client.bucket(bucket_name).blob(key)
    if not blob.exists():
        raise RuntimeError(f"NFL_PBP_DIAGNOSTIC_REQUIRED_ARTIFACT_MISSING gs://{bucket_name}/{key}")
    return blob.download_as_bytes()


def _upload_immutable(storage_client, bucket_name: str, key: str, data: bytes, content_type: str) -> dict:
    from google.api_core.exceptions import PreconditionFailed
    blob = storage_client.bucket(bucket_name).blob(key)
    try:
        blob.upload_from_string(data, content_type=content_type, if_generation_match=0)
        return {"uri": f"gs://{bucket_name}/{key}", "created": True}
    except PreconditionFailed:
        return {"uri": f"gs://{bucket_name}/{key}", "created": False, "reason": "ALREADY_EXISTS_IMMUTABLE"}


def _registry_payload_hash(registry: Mapping[str, Any]) -> str:
    payload = dict(registry)
    payload.pop("registry_sha256", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_and_verify_frozen_pbp(storage_client, bucket_name: str) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    reg = json.loads(_download_bytes(storage_client, bucket_name, f"{FROZEN_PBP_PREFIX}/registry.json"))
    stated = str(reg.get("registry_sha256") or "")
    computed = _registry_payload_hash(reg)
    if stated != FROZEN_PBP_REGISTRY_SHA256 or computed != FROZEN_PBP_REGISTRY_SHA256:
        raise RuntimeError(
            "NFL_PBP_DIAGNOSTIC_FROZEN_REGISTRY_MISMATCH "
            f"expected={FROZEN_PBP_REGISTRY_SHA256} stated={stated} computed={computed}"
        )
    if bool(reg.get("year_2026_queried")) or int(reg.get("production_authority") or 0) != 0:
        raise RuntimeError("NFL_PBP_DIAGNOSTIC_FROZEN_REGISTRY_CONTRACT_VIOLATION")
    if reg.get("model") != "PBP_CORE2_FIXED_RIDGE":
        raise RuntimeError("NFL_PBP_DIAGNOSTIC_UNEXPECTED_FROZEN_MODEL")

    bundle = joblib.load(io.BytesIO(_download_bytes(storage_client, bucket_name, f"{FROZEN_PBP_PREFIX}/pbp_core2_bundle.joblib")))
    meta = dict(bundle.get("metadata") or {})
    if str(meta.get("registry_sha256") or "") != FROZEN_PBP_REGISTRY_SHA256:
        raise RuntimeError("NFL_PBP_DIAGNOSTIC_BUNDLE_METADATA_SHA_MISMATCH")

    oof = pd.read_csv(io.BytesIO(gzip.decompress(_download_bytes(storage_client, bucket_name, f"{FROZEN_PBP_PREFIX}/pbp_oof_predictions.csv.gz"))))
    prior = pd.read_csv(io.BytesIO(gzip.decompress(_download_bytes(storage_client, bucket_name, f"{FROZEN_PBP_PREFIX}/pbp_prior_context.csv.gz"))), low_memory=False)
    if pd.to_numeric(oof.get("Season"), errors="coerce").max() > 2025 or pd.to_numeric(prior.get("Season"), errors="coerce").max() > 2025:
        raise RuntimeError("NFL_PBP_DIAGNOSTIC_2026_ARTIFACT_LEAK")
    if oof.get("physical_game_id", pd.Series(dtype=str)).duplicated().any():
        raise RuntimeError("NFL_PBP_DIAGNOSTIC_DUPLICATE_FROZEN_OOF")
    return reg, bundle, oof, prior


# ------------------------------- betting views --------------------------------
def _prepare_rows(oof: pd.DataFrame, incumbent: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "physical_game_id", "Season", "Game_Date", "Team_Norm", "Opponent_Norm",
        "Spread_Value", "Current_Total", "actual_margin", "actual_total",
        "pbp_margin_pred", "pbp_total_pred",
    ]
    missing = set(keep) - set(oof.columns)
    if missing:
        raise RuntimeError("NFL_PBP_DIAGNOSTIC_OOF_COLUMNS_MISSING " + str(sorted(missing)))
    ccols = [
        "physical_game_id", "core_margin_pred", "core_total_pred", "h2h_prob",
        "H2H_label", "H2H_close_novig_reference", "Week_Number", "Is_Home",
        "Is_Away", "Is_Neutral", "Is_Division_Game", "Spread_Value", "Current_Total",
        "actual_margin", "actual_total",
    ]
    ccols = [c for c in ccols if c in incumbent.columns]
    x = oof[keep].copy().merge(incumbent[ccols], on="physical_game_id", how="inner", suffixes=("", "_inc"), validate="one_to_one")
    if len(x) != len(oof):
        raise RuntimeError(f"NFL_PBP_DIAGNOSTIC_INCUMBENT_MATCH_MISMATCH oof={len(oof)} merged={len(x)}")

    sp = _num(x["Spread_Value"])
    total = _num(x["Current_Total"])
    am = _num(x["actual_margin"])
    at = _num(x["actual_total"])
    pm = _num(x["pbp_margin_pred"])
    cm = _num(x["core_margin_pred"])
    pt = _num(x["pbp_total_pred"])
    ct = _num(x["core_total_pred"])

    x["spread_market_margin"] = am + sp
    x["pbp_spread_edge"] = pm + sp
    x["core_spread_edge"] = cm + sp
    x["pbp_spread_side"] = np.sign(x["pbp_spread_edge"])
    x["core_spread_side"] = np.sign(x["core_spread_edge"])
    x["pbp_spread_selection_margin"] = x["pbp_spread_side"] * x["spread_market_margin"]
    x["core_spread_selection_margin"] = x["core_spread_side"] * x["spread_market_margin"]

    x["total_market_margin"] = at - total
    x["pbp_total_edge"] = pt - total
    x["core_total_edge"] = ct - total
    x["pbp_total_side"] = np.sign(x["pbp_total_edge"])
    x["core_total_side"] = np.sign(x["core_total_edge"])
    x["pbp_total_selection_margin"] = x["pbp_total_side"] * x["total_market_margin"]
    x["core_total_selection_margin"] = x["core_total_side"] * x["total_market_margin"]

    x["pbp_core_margin_delta"] = pm - cm
    x["core_margin_error"] = am - cm
    x["pbp_margin_error"] = am - pm
    x["core_total_error"] = at - ct
    x["pbp_total_error"] = at - pt
    x["spread_agree"] = (x["pbp_spread_side"] == x["core_spread_side"]) & x["pbp_spread_side"].ne(0) & x["core_spread_side"].ne(0)
    x["total_agree"] = (x["pbp_total_side"] == x["core_total_side"]) & x["pbp_total_side"].ne(0) & x["core_total_side"].ne(0)
    return x


def _edge_bucket_table(x: pd.DataFrame, market: str, model: str) -> dict:
    if market == "SPREADS":
        edge_col = f"{model}_spread_edge"
        sel_col = f"{model}_spread_selection_margin"
    else:
        edge_col = f"{model}_total_edge"
        sel_col = f"{model}_total_selection_margin"
    edge = _num(x[edge_col])
    bucket = pd.cut(edge.abs(), bins=EDGE_BINS, labels=EDGE_LABELS, right=False)
    out = {}
    for lab in EDGE_LABELS:
        m = bucket.astype(str).eq(lab)
        out[lab] = _side_record(x.loc[m, sel_col])
        out[lab]["mean_abs_edge"] = _round(edge.loc[m].abs().mean()) if m.any() else None
    # Sign-split prevents a large edge bucket from hiding directional asymmetry.
    signs = {}
    for label, sm in (("POSITIVE", edge.gt(0)), ("NEGATIVE", edge.lt(0))):
        signs[label] = _side_record(x.loc[sm, sel_col])
        signs[label]["mean_abs_edge"] = _round(edge.loc[sm].abs().mean()) if sm.any() else None
    return {"overall": _side_record(x[sel_col]), "edge_buckets": out, "edge_signs": signs}


def _spread_independence(x: pd.DataFrame) -> dict:
    p = _num(x.pbp_margin_pred)
    c = _num(x.core_margin_pred)
    actual = _num(x.actual_margin)
    pdiff = p - c
    core_err = actual - c
    valid = np.isfinite(pdiff) & np.isfinite(core_err) & np.abs(pdiff) > 1e-9
    correct = (np.sign(pdiff[valid]) == np.sign(core_err[valid]))
    slope = None
    if valid.sum() >= 10 and float(np.var(pdiff[valid])) > 1e-12:
        slope = float(np.cov(pdiff[valid], core_err[valid], ddof=0)[0, 1] / np.var(pdiff[valid]))
    return {
        "prediction_corr": _finite_corr(p, c),
        "edge_corr": _finite_corr(x.pbp_spread_edge, x.core_spread_edge),
        "residual_corr": _finite_corr(x.pbp_margin_error, x.core_margin_error),
        "direction_agreement_rate": _round(x.spread_agree.mean()),
        "direction_disagreement_rate": _round((~x.spread_agree & x.pbp_spread_side.ne(0) & x.core_spread_side.ne(0)).mean()),
        "pbp_deviation_points_toward_core_error_rate": _round(np.mean(correct)) if len(correct) else None,
        "pbp_minus_core_vs_core_error_corr": _finite_corr(pdiff, core_err),
        "diagnostic_correction_slope": _round(slope) if slope is not None else None,
    }


def _disagreement_table(x: pd.DataFrame) -> dict:
    ps = _num(x.pbp_spread_side)
    cs = _num(x.core_spread_side)
    cats = {
        "AGREE_TEAM": ps.eq(1) & cs.eq(1),
        "AGREE_OPPONENT": ps.eq(-1) & cs.eq(-1),
        "PBP_TEAM_CORE_OPPONENT": ps.eq(1) & cs.eq(-1),
        "PBP_OPPONENT_CORE_TEAM": ps.eq(-1) & cs.eq(1),
        "AGREE_ALL": x.spread_agree,
        "DISAGREE_ALL": (~x.spread_agree) & ps.ne(0) & cs.ne(0),
    }
    out = {}
    for name, m in cats.items():
        out[name] = {
            "n_games": int(m.sum()),
            "pbp_selection": _side_record(x.loc[m, "pbp_spread_selection_margin"]),
            "core_selection": _side_record(x.loc[m, "core_spread_selection_margin"]),
            "mean_pbp_abs_edge": _round(_num(x.loc[m, "pbp_spread_edge"]).abs().mean()) if m.any() else None,
            "mean_core_abs_edge": _round(_num(x.loc[m, "core_spread_edge"]).abs().mean()) if m.any() else None,
        }
    return out


def _confirmation_layers(x: pd.DataFrame, market: str) -> dict:
    if market == "SPREADS":
        pe, ce = "pbp_spread_edge", "core_spread_edge"
        psel, csel = "pbp_spread_selection_margin", "core_spread_selection_margin"
        agree = x.spread_agree
    else:
        pe, ce = "pbp_total_edge", "core_total_edge"
        psel, csel = "pbp_total_selection_margin", "core_total_selection_margin"
        agree = x.total_agree
    pabs = _num(x[pe]).abs(); cabs = _num(x[ce]).abs()
    out = {
        "all_agreement": {
            "n_games": int(agree.sum()),
            "core_selection": _side_record(x.loc[agree, csel]),
            "pbp_selection": _side_record(x.loc[agree, psel]),
        },
        "all_opposition": {
            "n_games": int((~agree).sum()),
            "core_selection": _side_record(x.loc[~agree, csel]),
            "pbp_selection": _side_record(x.loc[~agree, psel]),
        },
    }
    for threshold in (1.0, 2.0, 3.0, 4.0, 6.0):
        strong = pabs.ge(threshold)
        for label, mask in (("CONFIRMS", strong & agree), ("OPPOSES", strong & ~agree)):
            out[f"PBP_{threshold:g}+_{label}"] = {
                "n_games": int(mask.sum()),
                "core_selection": _side_record(x.loc[mask, csel]),
                "pbp_selection": _side_record(x.loc[mask, psel]),
                "mean_pbp_abs_edge": _round(pabs.loc[mask].mean()) if mask.any() else None,
                "mean_core_abs_edge": _round(cabs.loc[mask].mean()) if mask.any() else None,
            }
    # Explicit "incumbent uncertain / PBP strong" lane.
    for threshold in (2.0, 3.0, 4.0):
        mask = cabs.lt(1.0) & pabs.ge(threshold)
        out[f"CORE_LT1__PBP_{threshold:g}+"] = {
            "n_games": int(mask.sum()),
            "pbp_selection": _side_record(x.loc[mask, psel]),
            "core_selection": _side_record(x.loc[mask, csel]),
        }
    return out


def _season_stability(x: pd.DataFrame, market: str, model: str = "pbp") -> dict:
    sel = f"{model}_spread_selection_margin" if market == "SPREADS" else f"{model}_total_selection_margin"
    out = {}
    for sy, p in x.groupby("Season"):
        out[str(int(sy))] = _side_record(p[sel])
    return out


def _spread_regimes(x: pd.DataFrame) -> dict:
    edge = _num(x.pbp_spread_edge)
    selected_spread = np.where(edge.gt(0), _num(x.Spread_Value), -_num(x.Spread_Value))
    week = _num(x.get("Week_Number", pd.Series(np.nan, index=x.index)))
    abs_spread = _num(x.Spread_Value).abs()
    neutral = _num(x.get("Is_Neutral", pd.Series(0, index=x.index))).eq(1)
    selected_home = (~neutral) & edge.gt(0)
    selected_away = (~neutral) & edge.lt(0)
    masks = {
        "SELECTED_FAVORITE": pd.Series(selected_spread, index=x.index).lt(0),
        "SELECTED_DOG": pd.Series(selected_spread, index=x.index).gt(0),
        "SELECTED_PICKEM": pd.Series(selected_spread, index=x.index).eq(0),
        "SELECTED_HOME": selected_home,
        "SELECTED_AWAY": selected_away,
        "NEUTRAL_SITE": neutral,
        "MARKET_SPREAD_0_3": abs_spread.lt(3),
        "MARKET_SPREAD_3_7": abs_spread.ge(3) & abs_spread.lt(7),
        "MARKET_SPREAD_7_PLUS": abs_spread.ge(7),
        "WEEKS_1_4": week.le(4),
        "WEEKS_5_12": week.ge(5) & week.le(12),
        "WEEKS_13_PLUS": week.ge(13),
    }
    return {k: _side_record(x.loc[m, "pbp_spread_selection_margin"]) for k, m in masks.items()}


def _total_regimes(x: pd.DataFrame) -> dict:
    total = _num(x.Current_Total)
    week = _num(x.get("Week_Number", pd.Series(np.nan, index=x.index)))
    masks = {
        "CLOSING_TOTAL_LT42": total.lt(42),
        "CLOSING_TOTAL_42_46": total.ge(42) & total.lt(46),
        "CLOSING_TOTAL_46_50": total.ge(46) & total.lt(50),
        "CLOSING_TOTAL_50_PLUS": total.ge(50),
        "WEEKS_1_4": week.le(4),
        "WEEKS_5_12": week.ge(5) & week.le(12),
        "WEEKS_13_PLUS": week.ge(13),
    }
    return {k: _side_record(x.loc[m, "pbp_total_selection_margin"]) for k, m in masks.items()}


def _h2h_diagnostic(x: pd.DataFrame) -> dict:
    actual = _num(x.actual_margin)
    y = np.where(actual > 0, 1.0, np.where(actual < 0, 0.0, np.nan))
    pscore = _num(x.pbp_margin_pred)
    cscore = _num(x.core_margin_pred)
    hprob = _num(x.get("h2h_prob", pd.Series(np.nan, index=x.index)))
    valid = np.isfinite(y)

    def direction_rec(score):
        s = np.asarray(score, float)
        m = valid & np.isfinite(s) & (np.abs(s) > 1e-9)
        correct = np.where(s[m] > 0, y[m] == 1, y[m] == 0)
        w = int(correct.sum()); n = int(len(correct))
        return {"n": n, "wins": w, "losses": n - w, "accuracy": round(w / n, 6) if n else None, "wilson95": _wilson(w, n)}

    def auc(score):
        s = np.asarray(score, float); m = valid & np.isfinite(s)
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            return None
        return round(float(roc_auc_score(y[m], s[m])), 6)

    pbp = direction_rec(pscore); core = direction_rec(cscore)
    hp = direction_rec(hprob - .5) if np.isfinite(hprob).sum() else {"n": 0, "wins": 0, "losses": 0, "accuracy": None, "wilson95": [None, None]}
    disagree = np.sign(pscore) != np.sign(hprob - .5)
    disagree &= np.isfinite(pscore) & np.isfinite(hprob) & valid
    pbp_dis = direction_rec(np.where(disagree, pscore, np.nan))
    inc_dis = direction_rec(np.where(disagree, hprob - .5, np.nan))
    return {
        "pbp_margin_direction": {**pbp, "auc_using_margin_score": auc(pscore)},
        "core_margin_direction": {**core, "auc_using_margin_score": auc(cscore)},
        "incumbent_h2h_probability": {**hp, "auc": auc(hprob)},
        "pbp_vs_incumbent_h2h_score_corr": _finite_corr(pscore, hprob),
        "pbp_vs_incumbent_h2h_disagreement_n": int(disagree.sum()),
        "on_h2h_disagreement": {"pbp_direction": pbp_dis, "incumbent_h2h_direction": inc_dis},
        "note": "PBP CORE2 has no frozen probability calibrator; H2H is directional/AUC attribution only. No diagnostic probability model is fit here.",
    }


# -------------------------- frozen feature attribution -------------------------
def _feature_family(feature: str) -> str:
    f = feature.lower()
    if "min_prior_games" in f:
        return "EXPERIENCE"
    if "primary_qb" in f:
        return "QB_HISTORY"
    if "opponent_adjusted" in f:
        return "OPPONENT_ADJUSTED_EPA"
    if "explosive" in f:
        return "EXPLOSIVES"
    if "turnover" in f:
        return "TURNOVERS"
    if "sack" in f:
        return "SACKS"
    if "red_zone" in f:
        return "RED_ZONE"
    if "success" in f:
        return "SUCCESS_RATE"
    if "rush_epa" in f:
        return "RUSH_EPA"
    if "pass_epa" in f or "neutral_pass" in f or "cpoe" in f:
        return "PASSING"
    if "off_epa_play" in f or "def_allow_off_epa_play" in f:
        return "OVERALL_EPA"
    return "OTHER"


def _frozen_model_reliance(bundle: Mapping[str, Any], model_matrix: pd.DataFrame) -> dict:
    features = list(bundle.get("features") or [])
    if features != list((bundle.get("metadata") or {}).get("features") or features):
        raise RuntimeError("NFL_PBP_DIAGNOSTIC_BUNDLE_FEATURE_LIST_MISMATCH")
    missing = set(features) - set(model_matrix.columns)
    if missing:
        raise RuntimeError("NFL_PBP_DIAGNOSTIC_MODEL_MATRIX_FEATURES_MISSING " + str(sorted(missing)[:20]))

    X = model_matrix[features]
    fam_ix = defaultdict(list)
    for j, f in enumerate(features):
        fam_ix[_feature_family(f)].append(j)

    out = {}
    for market, key in (("SPREADS", "margin_model"), ("TOTALS", "total_model")):
        pipe = bundle.get(key)
        if pipe is None:
            raise RuntimeError(f"NFL_PBP_DIAGNOSTIC_BUNDLE_MODEL_MISSING {key}")
        imp = pipe.named_steps.get("impute"); sc = pipe.named_steps.get("scale"); ridge = pipe.named_steps.get("ridge")
        if imp is None or sc is None or ridge is None:
            raise RuntimeError(f"NFL_PBP_DIAGNOSTIC_UNEXPECTED_PIPELINE {key}")
        Xt = sc.transform(imp.transform(X))
        coef = np.asarray(ridge.coef_, float).reshape(-1)
        if Xt.shape[1] != len(features) or len(coef) != len(features):
            raise RuntimeError(f"NFL_PBP_DIAGNOSTIC_COEFFICIENT_DIMENSION_MISMATCH {key}")
        total_abs = float(np.abs(coef).sum()) or 1.0
        fams = {}
        for fam, ix in sorted(fam_ix.items()):
            contrib = Xt[:, ix] @ coef[ix]
            fams[fam] = {
                "feature_count": int(len(ix)),
                "coefficient_abs_share": round(float(np.abs(coef[ix]).sum() / total_abs), 6),
                "mean_abs_linear_contribution": round(float(np.mean(np.abs(contrib))), 6),
                "p90_abs_linear_contribution": round(float(np.quantile(np.abs(contrib), .90)), 6),
            }
        ranked = sorted(fams, key=lambda f: (fams[f]["coefficient_abs_share"], fams[f]["mean_abs_linear_contribution"]), reverse=True)
        out[market] = {
            "families": fams,
            "ranked_by_frozen_model_reliance": ranked,
            "note": "Descriptive attribution of the already-frozen full-history Ridge coefficients; not held-out family performance and not a refit.",
        }
    return out


# ----------------------------- system interaction ------------------------------
def _load_frozen_system_report(storage_client, bucket_name: str) -> tuple[dict, dict] | tuple[None, None]:
    try:
        reg = json.loads(_download_bytes(storage_client, bucket_name, f"{FROZEN_SYSTEM_PREFIX}/system_registry.json"))
        report = json.loads(_download_bytes(storage_client, bucket_name, f"{FROZEN_SYSTEM_PREFIX}/system_lab_report.json"))
    except Exception:
        return None, None
    stated = str(reg.get("registry_sha256") or "")
    computed = _registry_payload_hash(reg)
    if stated != FROZEN_SYSTEM_REGISTRY_SHA256 or computed != FROZEN_SYSTEM_REGISTRY_SHA256:
        raise RuntimeError("NFL_PBP_DIAGNOSTIC_SYSTEM_REGISTRY_MISMATCH")
    return reg, report


def _system_interaction(*, bq_client, storage_client, bucket_name: str, diag_rows: pd.DataFrame, log_func=print) -> dict:
    reg, report = _load_frozen_system_report(storage_client, bucket_name)
    if reg is None or report is None:
        return {"status": "FROZEN_SYSTEM_ARTIFACT_NOT_AVAILABLE", "production_authority": 0}

    # Import the exact frozen system-lab logic only to apply already-frozen rule
    # definitions to 2017-2025 rows. No search/miner/discovery function is run.
    import nfl_system_lab_v1 as sl
    if getattr(sl, "SOURCE_TAG", "") != "nfl-system-lab-v1-research-v2.0-ncaaf-miner-v3-methodology-20261001":
        raise RuntimeError("NFL_PBP_DIAGNOSTIC_STALE_SYSTEM_LAB_MODULE")
    view_cols = {f.name for f in bq_client.get_table(sl.VIEW).schema}
    side = bq_client.query(sl.build_intelligence_query(view_cols)).to_dataframe(create_bqstorage_client=False)
    if int(pd.to_numeric(side.Season, errors="coerce").max()) > 2025:
        raise RuntimeError("NFL_PBP_DIAGNOSTIC_SYSTEM_2026_QUERY_LEAK")
    state = sl.prepare_system_state(side)

    plays = []
    # Documented Big Al systems are applied directly from the frozen implementation.
    _, bigal_plays = sl._bigal_systems(state)
    if bigal_plays is not None and len(bigal_plays):
        for _, r in bigal_plays.iterrows():
            plays.append({"physical_game_id": str(r.get("physical_game_id")), "bet_team": str(r.get("bet_team")), "system_id": str(r.get("system_id")), "family": "BIG_AL"})

    # Apply only the exact already-frozen promising spread hypotheses. This is
    # evaluation/attribution, not a fresh search. Correlated variants are later
    # collapsed to one family vote.
    promising = (((report.get("miner") or {}).get("spreads") or {}).get("promising_rules") or [])
    atom_by = {a["name"]: a for a in sl._spread_atoms(state)}
    for rule in promising:
        conds = list(rule.get("conditions") or [])
        if not conds or any(c not in atom_by for c in conds):
            continue
        mask = np.ones(len(state), bool)
        for c in conds:
            mask &= np.asarray(atom_by[c]["mask"], bool)
        ix, _ = sl._candidate_rows_side(state, mask)
        direction = str(rule.get("direction") or "PLAY_ON").upper()
        rid = "__".join(conds)
        for j in ix:
            rr = state.iloc[int(j)]
            bet_team = str(rr.get("Team_Norm")) if direction == "PLAY_ON" else str(rr.get("Opponent_Norm"))
            plays.append({"physical_game_id": str(rr.get("physical_game_id")), "bet_team": bet_team, "system_id": rid, "family": "FROZEN_ROLE_FLIP_FAMILY"})

    if not plays:
        return {"status": "NO_HISTORICAL_SYSTEM_PLAYS_TO_INTERSECT", "frozen_system_registry_sha256": FROZEN_SYSTEM_REGISTRY_SHA256, "production_authority": 0}
    p = pd.DataFrame(plays).drop_duplicates(["physical_game_id", "bet_team", "system_id", "family"])
    base = diag_rows[["physical_game_id", "Team_Norm", "Opponent_Norm", "pbp_spread_edge", "spread_market_margin"]].copy()
    z = p.merge(base, on="physical_game_id", how="inner", validate="many_to_one")
    z["system_side"] = np.where(z.bet_team.astype(str).str.lower().eq(z.Team_Norm.astype(str).str.lower()), 1.0,
                         np.where(z.bet_team.astype(str).str.lower().eq(z.Opponent_Norm.astype(str).str.lower()), -1.0, np.nan))
    z = z.loc[np.isfinite(z.system_side)].copy()
    z["pbp_side"] = np.sign(_num(z.pbp_spread_edge))
    z["pbp_confirms"] = z.pbp_side.eq(z.system_side) & z.pbp_side.ne(0)
    z["system_selection_margin"] = z.system_side * _num(z.spread_market_margin)

    def group_report(q: pd.DataFrame) -> dict:
        conf = q.pbp_confirms
        opp = (~conf) & q.pbp_side.ne(0)
        return {
            "all": _side_record(q.system_selection_margin),
            "pbp_confirms": _side_record(q.loc[conf, "system_selection_margin"]),
            "pbp_opposes": _side_record(q.loc[opp, "system_selection_margin"]),
            "pbp_neutral": _side_record(q.loc[q.pbp_side.eq(0), "system_selection_margin"]),
        }

    # Collapse correlated mined variants to a single family-game-team observation.
    family = z.loc[z.family.eq("FROZEN_ROLE_FLIP_FAMILY")].drop_duplicates(["physical_game_id", "bet_team"]).copy()
    out = {
        "status": "FROZEN_SYSTEM_INTERACTION_COMPLETE",
        "frozen_system_registry_sha256": FROZEN_SYSTEM_REGISTRY_SHA256,
        "big_al": {},
        "frozen_role_flip_family": group_report(family) if len(family) else {"all": _side_record([])},
        "variant_independence_policy": "CORRELATED_PROMISING_VARIANTS_COLLAPSED_TO_ONE_FAMILY_GAME_TEAM_OBSERVATION",
        "production_authority": 0,
    }
    for sid, q in z.loc[z.family.eq("BIG_AL")].groupby("system_id"):
        out["big_al"][str(sid)] = group_report(q)
    return out


# ----------------------------- decision classification -------------------------
def _rate(d: Mapping[str, Any]) -> float | None:
    v = d.get("hit_rate") if isinstance(d, Mapping) else None
    return float(v) if v is not None else None


def _classify(report: Mapping[str, Any]) -> dict:
    spread = report["spread_signal"]
    s_over = spread["pbp"]["overall"]
    seasons = report["regime_stability"]["spreads_by_season"]
    season_good = sum(1 for v in seasons.values() if (v.get("hit_rate") or 0) > .5)
    hr = _rate(s_over) or 0.0
    n = int(s_over.get("n") or 0)
    if n >= 1000 and hr >= .54 and season_good >= 4:
        standalone = "STRONG"
    elif n >= 1000 and hr >= .525 and season_good >= 3:
        standalone = "MODERATE"
    elif n >= 800 and hr >= .505:
        standalone = "WEAK"
    else:
        standalone = "NONE"

    ind = report["spread_independence"]
    corr = abs(float(ind.get("prediction_corr") or 1.0))
    dis = float(ind.get("direction_disagreement_rate") or 0.0)
    if corr < .65 and dis >= .30:
        independence = "HIGH"
    elif corr < .80 and dis >= .20:
        independence = "MEDIUM"
    else:
        independence = "LOW"

    conf = report["spread_confirmation"]
    a = conf.get("PBP_3+_CONFIRMS", {}).get("core_selection", {})
    o = conf.get("PBP_3+_OPPOSES", {}).get("core_selection", {})
    ar, orr = _rate(a), _rate(o)
    confirmation = bool((a.get("n") or 0) >= 100 and (o.get("n") or 0) >= 100 and ar is not None and orr is not None and ar - orr >= .025)
    contradiction = bool((o.get("n") or 0) >= 100 and orr is not None and orr < .50)

    # Require a non-trivial 3+ edge population and a historical hit-rate signal.
    b3 = spread["pbp"]["edge_buckets"].get("3-4", {})
    b4 = spread["pbp"]["edge_buckets"].get("4-6", {})
    b6 = spread["pbp"]["edge_buckets"].get("6+", {})
    nn = sum(int(z.get("n") or 0) for z in (b3, b4, b6))
    ww = sum(int(z.get("wins") or 0) for z in (b3, b4, b6))
    high_edge_rate = ww / nn if nn else None
    market_residual = bool(nn >= 200 and high_edge_rate is not None and high_edge_rate >= .53)

    total = report["total_signal"]["pbp"]["overall"]
    total_value = bool((total.get("n") or 0) >= 1000 and (_rate(total) or 0) >= .525)

    h = report["h2h"]
    hd = h.get("on_h2h_disagreement", {})
    ph = hd.get("pbp_direction", {}); ih = hd.get("incumbent_h2h_direction", {})
    h2h_value = bool((ph.get("n") or 0) >= 150 and ph.get("accuracy") is not None and ih.get("accuracy") is not None and ph["accuracy"] - ih["accuracy"] >= .02)

    if standalone in ("STRONG", "MODERATE") and independence in ("HIGH", "MEDIUM") and market_residual:
        role = "INDEPENDENT_BRAIN"
    elif confirmation or contradiction:
        role = "CONFIRMATION_BRAIN"
    elif total_value:
        role = "TOTALS_BRAIN"
    elif h2h_value:
        role = "H2H_DIRECTIONAL_BRAIN"
    else:
        role = "RESEARCH_ONLY"

    return {
        "FROZEN_ARTIFACT_VALID": True,
        "SPREAD": {
            "STANDALONE_SIGNAL": standalone,
            "INDEPENDENCE_FROM_CORE": independence,
            "MARKET_RESIDUAL_SIGNAL": "YES" if market_residual else "NO",
            "CONFIRMATION_VALUE": "YES" if confirmation else "NO",
            "CONTRADICTION_VALUE": "YES" if contradiction else "NO",
            "HIGH_EDGE_3_PLUS_N": nn,
            "HIGH_EDGE_3_PLUS_HIT_RATE": round(high_edge_rate, 6) if high_edge_rate is not None else None,
        },
        "H2H": {"INCREMENTAL_DIRECTIONAL_VALUE": "YES" if h2h_value else "NO", "PROBABILITY_CALIBRATION_TESTED": False},
        "TOTALS": {"INCREMENTAL_VALUE": "YES" if total_value else "NO"},
        "RECOMMENDED_ROLE": role,
        "PRODUCTION_AUTHORITY": 0,
        "FROZEN_ARTIFACT_CHANGED": False,
        "AUTOMATIC_PROMOTION": False,
        "classification_note": "Diagnostic heuristic only. Historical closing-line results are retrospective evidence, not prospective proof or executable CLV.",
    }


# -------------------------------- main runner ----------------------------------
def run_nfl_pbp_attribution_v1(*, bq_client, storage_client, bucket_name="sharp-models", audit_report=None, log_func=print) -> dict:
    if not isinstance(audit_report, dict) or audit_report.get("status") != "READY_FOR_OFFLINE_CHALLENGER_SANDBOX":
        raise RuntimeError("NFL_PBP_DIAGNOSTIC_AUDIT_NOT_GREEN")
    assert_contract()
    registry, bundle, oof, prior = _load_and_verify_frozen_pbp(storage_client, bucket_name)
    log_func("[NFL-RESEARCH-V2-PBP-DIAG-PREFLIGHT] " + json.dumps({
        "status": "PASS", "source_tag": SOURCE_TAG,
        "frozen_pbp_registry_sha256": FROZEN_PBP_REGISTRY_SHA256,
        "frozen_status": "NFL_RESEARCH_V2_PBP_FOUNDATION_FROZEN_FOR_PROSPECTIVE_SHADOW",
        "oof_games": int(len(oof)), "year_2026_queried": False,
        "pbp_refit": False, "production_authority": 0,
    }, sort_keys=True))

    # Protected historical source, still 2017-2025 only. This regenerates the
    # same incumbent OOF baseline; it does NOT refit or alter the frozen PBP artifact.
    view_cols = {f.name for f in bq_client.get_table(VIEW).schema}
    side = bq_client.query(build_readonly_query(view_cols)).to_dataframe(create_bqstorage_client=False)
    if int(pd.to_numeric(side.Season, errors="coerce").max()) > 2025:
        raise RuntimeError("NFL_PBP_DIAGNOSTIC_2026_QUERY_LEAK")
    incumbent = _generate_oof_core(side)
    rows = _prepare_rows(oof, incumbent)

    spread_signal = {
        "pbp": _edge_bucket_table(rows, "SPREADS", "pbp"),
        "incumbent_core": _edge_bucket_table(rows, "SPREADS", "core"),
        "retrospective_line_note": "Spread_Value is a historical closing reference used for settlement/attribution, not an as-of executable quote.",
    }
    total_signal = {
        "pbp": _edge_bucket_table(rows, "TOTALS", "pbp"),
        "incumbent_core": _edge_bucket_table(rows, "TOTALS", "core"),
        "retrospective_line_note": "Current_Total is a historical closing reference used for settlement/attribution, not an as-of executable quote.",
    }
    spread_independence = _spread_independence(rows)
    disagreement = _disagreement_table(rows)
    spread_confirmation = _confirmation_layers(rows, "SPREADS")
    total_confirmation = _confirmation_layers(rows, "TOTALS")
    h2h = _h2h_diagnostic(rows)

    # Rebuild only the predictor matrix from the stored frozen prior-context artifact.
    # No PBP download and no model fit occurs.
    games = physical_games(side)
    matched, match_audit = match_pbp_to_authoritative(games, prior)
    model_matrix, reconstructed_features = build_model_matrix(matched)
    if list(reconstructed_features) != list(bundle.get("features") or []):
        raise RuntimeError("NFL_PBP_DIAGNOSTIC_RECONSTRUCTED_FEATURE_CONTRACT_MISMATCH")
    feature_attribution = _frozen_model_reliance(bundle, model_matrix)

    systems = _system_interaction(
        bq_client=bq_client, storage_client=storage_client, bucket_name=bucket_name,
        diag_rows=rows, log_func=log_func,
    )

    report = {
        "status": STATUS,
        "source_tag": SOURCE_TAG,
        "contract_sha256": contract_hash(),
        "frozen_pbp_registry_sha256": FROZEN_PBP_REGISTRY_SHA256,
        "frozen_system_registry_sha256": FROZEN_SYSTEM_REGISTRY_SHA256,
        "frozen_artifact_status": "VERIFIED_UNCHANGED",
        "pbp_refit": False,
        "pbp_tuning": False,
        "automatic_promotion": False,
        "production_authority": 0,
        "year_2026_queried": False,
        "oof_games": int(len(rows)),
        "seasons": sorted(int(s) for s in pd.to_numeric(rows.Season, errors="coerce").dropna().unique()),
        "spread_signal": spread_signal,
        "spread_independence": spread_independence,
        "spread_disagreement": disagreement,
        "spread_confirmation": spread_confirmation,
        "total_signal": total_signal,
        "total_confirmation": total_confirmation,
        "h2h": h2h,
        "regime_stability": {
            "spreads_by_season": _season_stability(rows, "SPREADS", "pbp"),
            "totals_by_season": _season_stability(rows, "TOTALS", "pbp"),
            "spread_regimes": _spread_regimes(rows),
            "total_regimes": _total_regimes(rows),
        },
        "feature_attribution": feature_attribution,
        "stored_prior_context_match_audit": match_audit,
        "systems_interaction": systems,
        "notes": [
            "No frozen PBP model is retrained or mutated.",
            "No 2026 data is queried or used in historical attribution.",
            "Feature-family attribution describes frozen-model reliance; it is not a family-removal refit.",
            "Historical closing lines are retrospective reference labels and are not claimed as executable prices or CLV.",
        ],
    }
    report["classification"] = _classify(report)

    diagnostic_registry = {
        "source_tag": SOURCE_TAG,
        "contract_sha256": contract_hash(),
        "frozen_pbp_registry_sha256": FROZEN_PBP_REGISTRY_SHA256,
        "frozen_system_registry_sha256": FROZEN_SYSTEM_REGISTRY_SHA256,
        "inputs": {
            "pbp_oof_predictions": f"gs://{bucket_name}/{FROZEN_PBP_PREFIX}/pbp_oof_predictions.csv.gz",
            "pbp_prior_context": f"gs://{bucket_name}/{FROZEN_PBP_PREFIX}/pbp_prior_context.csv.gz",
            "pbp_model_bundle": f"gs://{bucket_name}/{FROZEN_PBP_PREFIX}/pbp_core2_bundle.joblib",
        },
        "method": "FROZEN_OOF_ATTRIBUTION_DISAGREEMENT_RESIDUAL_CONFIRMATION_REGIME_SYSTEM_INTERACTION",
        "pbp_refit": False,
        "pbp_tuning": False,
        "model_artifact_written": False,
        "year_2026_queried": False,
        "production_authority": 0,
        "automatic_promotion": False,
    }
    # Deterministic immutable identity: reruns with the same inputs/method point
    # to the same evidence package.
    hash_payload = dict(diagnostic_registry)
    diag_sha = hashlib.sha256(json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    diagnostic_registry["diagnostic_registry_sha256"] = diag_sha
    report["diagnostic_registry_sha256"] = diag_sha
    prefix = f"nfl-research/v2_0/pbp_diagnostic/{diag_sha[:16]}"

    csv_cols = [
        "physical_game_id", "Season", "Game_Date", "Team_Norm", "Opponent_Norm",
        "Spread_Value", "Current_Total", "actual_margin", "actual_total",
        "pbp_margin_pred", "core_margin_pred", "pbp_spread_edge", "core_spread_edge",
        "pbp_spread_selection_margin", "core_spread_selection_margin", "spread_agree",
        "pbp_total_pred", "core_total_pred", "pbp_total_edge", "core_total_edge",
        "pbp_total_selection_margin", "core_total_selection_margin", "total_agree",
        "h2h_prob", "H2H_label", "Week_Number", "Is_Home", "Is_Neutral",
    ]
    csv_cols = [c for c in csv_cols if c in rows.columns]
    raw_csv = rows[csv_cols].to_csv(index=False).encode()
    gz = gzip.compress(raw_csv, compresslevel=9)

    clean_report = _sanitize_json(report)
    clean_reg = _sanitize_json(diagnostic_registry)
    arts = {
        "registry": _upload_immutable(storage_client, bucket_name, f"{prefix}/diagnostic_registry.json", _safe_json(clean_reg), "application/json"),
        "report": _upload_immutable(storage_client, bucket_name, f"{prefix}/pbp_attribution_report.json", _safe_json(clean_report), "application/json"),
        "diagnostic_rows": _upload_immutable(storage_client, bucket_name, f"{prefix}/pbp_diagnostic_rows.csv.gz", gz, "application/gzip"),
    }
    report["artifacts"] = arts

    log_func("[NFL-RESEARCH-V2-PBP-DIAG-INDEPENDENCE] " + json.dumps(_sanitize_json(spread_independence), sort_keys=True))
    log_func("[NFL-RESEARCH-V2-PBP-DIAG-SPREAD] " + json.dumps(_sanitize_json(spread_signal), sort_keys=True))
    log_func("[NFL-RESEARCH-V2-PBP-DIAG-CONFIRMATION] " + json.dumps(_sanitize_json(spread_confirmation), sort_keys=True))
    log_func("[NFL-RESEARCH-V2-PBP-DIAG-TOTALS] " + json.dumps(_sanitize_json(total_signal), sort_keys=True))
    log_func("[NFL-RESEARCH-V2-PBP-DIAG-H2H] " + json.dumps(_sanitize_json(h2h), sort_keys=True))
    log_func("[NFL-RESEARCH-V2-PBP-DIAG-SYSTEMS] " + json.dumps(_sanitize_json(systems), sort_keys=True))
    log_func("[NFL-RESEARCH-V2-PBP-DIAG-CLASSIFICATION] " + json.dumps(_sanitize_json(report["classification"]), sort_keys=True))
    log_func("[NFL-RESEARCH-V2-PBP-DIAG-CONTRACT] " + json.dumps({
        "status": STATUS,
        "diagnostic_registry_sha256": diag_sha,
        "frozen_pbp_registry_sha256": FROZEN_PBP_REGISTRY_SHA256,
        "frozen_artifact_changed": False,
        "pbp_refit": False,
        "year_2026_queried": False,
        "production_authority": 0,
        "artifacts": arts,
    }, sort_keys=True, default=str))
    return report


# -------------------------------- smoke tests ----------------------------------
def self_test() -> dict:
    x = pd.DataFrame({
        "physical_game_id": ["a", "b", "c", "d"], "Season": [2025]*4,
        "Game_Date": ["2025-09-01"]*4, "Team_Norm": ["A"]*4, "Opponent_Norm": ["B"]*4,
        "Spread_Value": [-3, 3, -1, 1], "Current_Total": [44, 44, 44, 44],
        "actual_margin": [7, -7, 3, -3], "actual_total": [50, 38, 46, 42],
        "pbp_margin_pred": [5, -5, 4, -4], "pbp_total_pred": [48, 40, 46, 42],
    })
    inc = pd.DataFrame({
        "physical_game_id": ["a", "b", "c", "d"], "core_margin_pred": [4, -4, -2, 2],
        "core_total_pred": [47, 41, 43, 45], "h2h_prob": [.65, .35, .45, .55],
        "H2H_label": [1, 0, 1, 0], "H2H_close_novig_reference": [.6, .4, .48, .52],
        "Week_Number": [1,2,3,4], "Is_Home": [1]*4, "Is_Away": [0]*4, "Is_Neutral": [0]*4,
        "Is_Division_Game": [0]*4, "Spread_Value": [-3, 3, -1, 1], "Current_Total": [44]*4,
        "actual_margin": [7,-7,3,-3], "actual_total": [50,38,46,42],
    })
    d = _prepare_rows(x, inc)
    if len(d) != 4 or not np.isfinite(d.pbp_spread_edge).all():
        raise AssertionError("PREPARE_ROWS_FAILED")
    if _edge_bucket_table(d, "SPREADS", "pbp")["overall"]["n"] != 4:
        raise AssertionError("EDGE_TABLE_FAILED")
    if _disagreement_table(d)["DISAGREE_ALL"]["n_games"] <= 0:
        raise AssertionError("DISAGREEMENT_TEST_FAILED")
    return {"status": "PASS", "rows": 4, "production_authority": 0, "pbp_refit": False}
