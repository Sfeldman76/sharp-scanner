"""Auditable 2013-2025 NCAAF multi-brain research (shadow mode; no bet authority).

Independent historical system validation uses the *entire* available game history.
The CORE comparison is a separate matched cohort; unavailable CORE games never
remove a game from system validation. Unknown system coverage is never 0 triggers.
No production code, thresholds, or bets are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import itertools
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

VERSION = "NCAAF_BRAIN_FULL_HISTORY_SHADOW_V2_20261023"
BRAINS = ("STAT", "BIG_AL", "PATHI", "SYSTEM_MINER", "PT")
YEAR_FIRST, YEAR_LAST = 2013, 2025
DISCOVERY_LAST = 2021
SELECTION_YEARS = (2022, 2023)
CONFIRM_YEARS = (2024, 2025)
REQUIRED_GAME = {"season", "game_id", "kickoff_utc", "actual_home_margin"}
REQUIRED_SIGNALS = {"season", "game_id", "brain", "system_id", "side", "feature_asof_utc", "evidence_family"}
REQUIRED_COVERAGE = {"season", "brain", "status", "eligible_games", "feature_rows", "source_hash", "point_in_time_certified"}
REQUIRED_CORE = {"season", "game_id", "model_home_margin", "prediction_at_utc"}
REQUIRED_ELIGIBILITY = {"season", "game_id", "brain", "evaluation_status", "feature_asof_utc"}
REQUIRED_UNIVERSE = {"season", "expected_games", "game_ids_sha256", "source", "source_snapshot_utc"}
REQUIRED_RULE_EVAL = {"season", "game_id", "brain", "system_id", "evaluation_status", "feature_asof_utc", "rule_fit_max_season", "rule_version_sha256"}


@dataclass(frozen=True)
class RunConfig:
    core_min_edge: float = 2.0
    min_system_confirmation_bets: int = 80
    min_confirmation_seasons: int = 2
    research_min_year: int = YEAR_FIRST
    research_max_year: int = YEAR_LAST


def _require(df: pd.DataFrame, names: set[str], label: str) -> None:
    absent = names - set(df.columns)
    if absent:
        raise ValueError(f"{label}: required columns absent: {sorted(absent)}")


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, dtype={"game_id": "string", "system_id": "string"}, low_memory=False)


def _time(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x, utc=True, errors="coerce")


def _to_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype("string").str.strip().str.lower().isin({"true", "1", "yes", "certified"})


def _as_num(df: pd.DataFrame, key: str) -> pd.Series:
    if key not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[key], errors="coerce")


def _validate_years(df: pd.DataFrame, label: str, cfg: RunConfig) -> pd.DataFrame:
    x = df.copy()
    x["season"] = pd.to_numeric(x["season"], errors="raise").astype(int)
    outside = x[~x["season"].between(cfg.research_min_year, cfg.research_max_year)]
    if not outside.empty:
        raise ValueError(f"{label}: rows outside sealed {cfg.research_min_year}-{cfg.research_max_year} window: {sorted(outside.season.unique().tolist())}")
    if "game_id" in x:
        x["game_id"] = x["game_id"].astype("string").str.strip()
    if "game_id" in x and (x["game_id"].isna() | x["game_id"].eq("")).any():
        raise ValueError(f"{label}: empty game_id")
    return x


def _unique(df: pd.DataFrame, keys: list[str], label: str) -> None:
    if df.duplicated(keys).any():
        bad = df.loc[df.duplicated(keys, keep=False), keys].head(6).to_dict("records")
        raise ValueError(f"{label}: duplicate identities on {keys}: {bad}")


def _price_certified(games: pd.DataFrame) -> pd.Series:
    # A valid timestamp is necessary, NOT sufficient: provider provenance must
    # explicitly assert a genuine archived pregame quote (not imputed opener).
    if not {"home_spread", "quote_at_utc", "quote_source", "quote_snapshot_id", "quote_vintage_certified"}.issubset(games.columns):
        return pd.Series(False, index=games.index)
    when = _time(games.quote_at_utc)
    approved = _to_bool(games.quote_vintage_certified)
    has_snapshot = games.quote_snapshot_id.fillna("").astype(str).str.strip().ne("")
    return (approved & has_snapshot & when.notna() & (when < games.kickoff_utc)
            & games.quote_source.fillna("").astype(str).str.strip().ne("")
            & _as_num(games, "home_spread").notna())


def prepare_games(frame: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    _require(frame, REQUIRED_GAME, "games")
    g = _validate_years(frame, "games", cfg)
    _unique(g, ["season", "game_id"], "games")
    g["kickoff_utc"] = _time(g.kickoff_utc)
    if g.kickoff_utc.isna().any():
        raise ValueError("games: missing/invalid kickoff_utc")
    g["actual_home_margin"] = _as_num(g, "actual_home_margin")
    if g.actual_home_margin.isna().any():
        raise ValueError("games: missing final score margin")
    g["quote_at_utc"] = _time(g.quote_at_utc) if "quote_at_utc" in g else pd.NaT
    g["home_spread"] = _as_num(g, "home_spread")
    g["market_line_type"] = np.where(g.home_spread.notna(),"OPEN", "MISSING")
    if "close_home_spread" in g.columns:
        fallback = g.home_spread.isna() & _as_num(g,"close_home_spread").notna()
        g.loc[fallback,"home_spread"] = _as_num(g,"close_home_spread").loc[fallback]
        g.loc[fallback,"market_line_type"] = "CLOSE_FALLBACK_DESCRIPTIVE_ONLY"
    g["price_vintage_verified"] = _price_certified(g) & g.market_line_type.eq("OPEN")
    g["home_ats_delta"] = g.actual_home_margin + g.home_spread
    return g



def universe_audit(games: pd.DataFrame, manifest: pd.DataFrame | None,
                   cfg: RunConfig) -> pd.DataFrame:
    """Independent CFBD physical-game manifest, including IDs, must match.

    The manifest should be computed against the raw CFBD archive, NOT by copying
    this candidate modeling frame. Without it, completeness is UNKNOWN.
    """
    n=games.groupby("season").size()
    h=(games.groupby("season")["game_id"].apply(lambda ids:
        hashlib.sha256("\n".join(sorted(ids.astype(str))).encode()).hexdigest()))
    years=pd.DataFrame({"season":list(range(cfg.research_min_year,cfg.research_max_year+1))})
    years["observed_games"]=years.season.map(n).fillna(0).astype(int)
    years["observed_ids_sha256"]=years.season.map(h).fillna("")
    if manifest is None:
        years["expected_games"]=np.nan
        years["game_ids_sha256"]=""
        years["source"]=""
        years["source_snapshot_utc"]=pd.NaT
        years["universe_verified"]=False
        return years
    _require(manifest,REQUIRED_UNIVERSE,"universe_manifest")
    m=manifest.copy()
    m["season"]=pd.to_numeric(m.season,errors="raise").astype(int)
    _unique(m,["season"],"universe_manifest")
    if not m.season.between(cfg.research_min_year,cfg.research_max_year).all():
        raise ValueError("universe_manifest: outside 2013-2025 window")
    m["expected_games"]=pd.to_numeric(m.expected_games,errors="raise")
    years=years.merge(m,on="season",how="left",validate="one_to_one")
    years["universe_verified"]=(years.expected_games.eq(years.observed_games)
       & years.game_ids_sha256.eq(years.observed_ids_sha256)
       & years.source.fillna("").str.strip().ne("")
       & _time(years.source_snapshot_utc).notna())
    return years


def prepare_coverage(frame: pd.DataFrame, g: pd.DataFrame, cfg: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require(frame, REQUIRED_COVERAGE, "coverage")
    c = _validate_years(frame, "coverage", cfg)
    c["brain"] = c.brain.astype(str).str.upper().str.strip()
    if (~c.brain.isin(BRAINS)).any():
        raise ValueError(f"coverage: unrecognized brains: {sorted(set(c.brain)-set(BRAINS))}")
    _unique(c, ["season", "brain"], "coverage")
    c["status"] = c.status.astype(str).str.upper().str.strip()
    if (~c.status.isin(["COMPLETE", "PARTIAL", "UNAVAILABLE"])).any():
        raise ValueError("coverage: invalid status")
    c["eligible_games"] = pd.to_numeric(c.eligible_games, errors="raise").astype(int)
    c["feature_rows"] = pd.to_numeric(c.feature_rows, errors="raise").astype(int)
    if (c[["eligible_games", "feature_rows"]] < 0).any().any():
        raise ValueError("coverage: negative eligibility/features")
    c["point_in_time_certified"] = _to_bool(c.point_in_time_certified)
    c["source_hash"] = c.source_hash.fillna("").astype(str).str.strip()
    # Full 13*5 matrix, including explicit UNKNOWN rows; missing != zero triggers.
    matrix = pd.MultiIndex.from_product(
        [range(cfg.research_min_year, cfg.research_max_year+1), BRAINS],
        names=["season", "brain"]).to_frame(index=False)
    c = matrix.merge(c, on=["season", "brain"], how="left", validate="one_to_one")
    c["status"] = c.status.fillna("UNKNOWN")
    c["point_in_time_certified"] = _to_bool(c.point_in_time_certified)
    c["source_hash"] = c.source_hash.fillna("")
    c["coverage_complete"] = (c.status.eq("COMPLETE") & c.point_in_time_certified &
                               c.source_hash.str.fullmatch(r"[a-fA-F0-9]{64}") & c.eligible_games.gt(0) &
                               c.feature_rows.ge(c.eligible_games))
    count = g.groupby("season").size().rename("universe_games")
    c = c.join(count, on="season")
    if (c.eligible_games.fillna(0) > c.universe_games).any():
        raise ValueError("coverage: eligible_games exceeds canonical games universe")
    summary = (c.groupby("brain", as_index=False)
               .agg(years_total=("season", "size"), years_complete=("coverage_complete", "sum"),
                    eligible_games_reported=("eligible_games", "sum"),
                    feature_rows_reported=("feature_rows", "sum")))
    summary["full_history_certified"] = summary.years_complete.eq(cfg.research_max_year-cfg.research_min_year+1)
    return c, summary



def prepare_eligibility(frame: pd.DataFrame | None, games: pd.DataFrame,
                        coverage: pd.DataFrame, cfg: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One explicit evaluated/ineligible/unknown record per game per brain.

    Season coverage ALONE never establishes that a silent brain evaluated a game.
    This prevents missing historical Big Al/Pathi/PT records being interpreted as
    a negative signal. Event rows must be a subset of EVALUATED rows.
    """
    matrix = games[["season", "game_id", "kickoff_utc"]].merge(
        pd.DataFrame({"brain": BRAINS}), how="cross")
    if frame is None:
        e = matrix.copy()
        e["evaluation_status"] = "UNKNOWN"
        e["feature_asof_utc"] = pd.NaT
    else:
        _require(frame, REQUIRED_ELIGIBILITY, "eligibility")
        supplied = _validate_years(frame, "eligibility", cfg)
        supplied["brain"] = supplied.brain.astype(str).str.upper().str.strip()
        supplied["evaluation_status"] = supplied.evaluation_status.astype(str).str.upper().str.strip()
        if not supplied.brain.isin(BRAINS).all() or not supplied.evaluation_status.isin(
                ["EVALUATED", "INELIGIBLE", "UNKNOWN"]).all():
            raise ValueError("eligibility: invalid brain or evaluation_status")
        _unique(supplied,["season","game_id","brain"],"eligibility")
        supplied["feature_asof_utc"] = _time(supplied.feature_asof_utc)
        e=matrix.merge(supplied[["season","game_id","brain","evaluation_status","feature_asof_utc"]],
                       on=["season","game_id","brain"],how="left",validate="one_to_one")
        present = supplied.merge(matrix[["season","game_id","brain"]],
                                 on=["season","game_id","brain"],how="left",indicator=True)
        if present._merge.ne("both").any():
            raise ValueError("eligibility: orphan game or unrecognized season")
        e["evaluation_status"] = e.evaluation_status.fillna("UNKNOWN")
    e["evaluation_verified"] = (e.evaluation_status.eq("INELIGIBLE") |
        (e.evaluation_status.eq("EVALUATED") & e.feature_asof_utc.notna() &
         (e.feature_asof_utc < e.kickoff_utc)))
    counts=(e.groupby(["season","brain"],as_index=False).agg(
        games_accounted=("evaluation_verified","sum"),
        eligible_replayed=("evaluation_status",lambda x:int(x.eq("EVALUATED").sum())),
        universe_games=("game_id","size")))
    counts["eligibility_complete"] = counts.games_accounted.eq(counts.universe_games)
    c=coverage.merge(counts,on=["season","brain"],how="left",validate="one_to_one",
                     suffixes=("", "_eligibility"))
    c["coverage_complete"] = (c.coverage_complete & c.eligibility_complete &
                                c.eligible_games.eq(c.eligible_replayed))
    e=e.merge(c[["season","brain","coverage_complete"]],on=["season","brain"],
              how="left",validate="many_to_one")
    e["game_brain_verified"] = e.evaluation_verified & e.coverage_complete
    return e,c



def prepare_rule_evaluations(frame: pd.DataFrame | None, eligibility: pd.DataFrame,
                             cfg: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audits full per-game nonfiring denominators for *each* historical rule.

    A partial rule log never certifies non-firing games. A complete rule log must
    account for each brain-eligible game with EVALUATED or INELIGIBLE.
    """
    cols=["season","game_id","brain","system_id","evaluation_status","feature_asof_utc",
          "rule_fit_max_season","rule_version_sha256","rule_game_verified"]
    aggcols=["season","brain","system_id","rule_year_complete","rule_games_accounted",
             "brain_games_eligible","rule_evaluated_games","non_firing_denominator_known"]
    if frame is None:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=aggcols)
    _require(frame,REQUIRED_RULE_EVAL,"rule_evaluations")
    r=_validate_years(frame,"rule_evaluations",cfg)
    r["brain"]=r.brain.astype(str).str.upper().str.strip()
    r["evaluation_status"]=r.evaluation_status.astype(str).str.upper().str.strip()
    r["system_id"]=r.system_id.astype("string").str.strip()
    if (~r.brain.isin(BRAINS)).any() or (~r.evaluation_status.isin(["EVALUATED","INELIGIBLE","UNKNOWN"])).any():
        raise ValueError("rule_evaluations: invalid brain/status")
    if r.system_id.isna().any() or r.system_id.eq("").any():
        raise ValueError("rule_evaluations: missing system_id")
    _unique(r,["season","game_id","brain","system_id"],"rule_evaluations")
    r["feature_asof_utc"]=_time(r.feature_asof_utc)
    r["rule_fit_max_season"]=_as_num(r,"rule_fit_max_season")
    r["rule_version_sha256"]=r.rule_version_sha256.fillna("").astype(str).str.strip()
    r=r.merge(eligibility[["season","game_id","brain","evaluation_status","kickoff_utc", "game_brain_verified"]],
              on=["season","game_id","brain"], how="left",validate="many_to_one",indicator=True,
              suffixes=("", "_brain"))
    if r._merge.ne("both").any() or r.evaluation_status_brain.ne("EVALUATED").any():
        raise ValueError("rule_evaluations: record outside a brain-eligible game")
    r=r.drop(columns="_merge")
    r["rule_game_verified"]=(r.game_brain_verified &
        r.evaluation_status.isin(["EVALUATED","INELIGIBLE"]) &
        r.feature_asof_utc.notna() & r.feature_asof_utc.lt(r.kickoff_utc) &
        r.rule_fit_max_season.lt(r.season) &
        r.rule_version_sha256.str.fullmatch(r"[a-fA-F0-9]{64}"))
    eligible=(eligibility.loc[eligibility.evaluation_status.eq("EVALUATED")]
              .groupby(["season","brain"]).size().rename("brain_games_eligible"))
    a=(r.groupby(["season","brain","system_id"],as_index=False).agg(
        rule_games_accounted=("game_id","size"),
        rule_evaluated_games=("evaluation_status",lambda z:int(z.eq("EVALUATED").sum())),
        valid_records=("rule_game_verified","sum"),
        unique_rule_versions=("rule_version_sha256","nunique")))
    a=a.join(eligible,on=["season","brain"])
    # The no-fire denominator must correspond to one frozen rule definition,
    # not a mixture of revised rules selected after seeing game outcomes.
    a["rule_year_complete"]=(a.valid_records.eq(a.rule_games_accounted) &
                              a.rule_games_accounted.eq(a.brain_games_eligible) &
                              a.unique_rule_versions.eq(1))
    a["non_firing_denominator_known"]=a.rule_year_complete
    r=r.merge(a[["season","brain","system_id","rule_year_complete"]],
              on=["season","brain","system_id"],how="left",validate="many_to_one")
    r["rule_game_verified"] &= r.rule_year_complete
    return r,a[aggcols]


def prepare_signals(frame: pd.DataFrame, games: pd.DataFrame, coverage: pd.DataFrame,
                    eligibility: pd.DataFrame, rule_evaluations: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    _require(frame, REQUIRED_SIGNALS, "signals")
    s = _validate_years(frame, "signals", cfg)
    s["brain"] = s.brain.astype("string").str.upper().str.strip()
    s["side"] = s.side.astype("string").str.upper().str.strip()
    s["system_id"] = s.system_id.astype("string").str.strip()
    s["evidence_family"] = s.evidence_family.astype("string").str.upper().str.strip()
    if (~s.brain.isin(BRAINS)).any() or (~s.side.isin(["HOME", "AWAY"])).any():
        raise ValueError("signals: invalid brain or side")
    if s.system_id.isna().any() or s.system_id.eq("").any() or s.evidence_family.isna().any() or s.evidence_family.eq("").any():
        raise ValueError("signals: require system_id and evidence_family")
    _unique(s, ["season", "game_id", "brain", "system_id"], "signals")
    s["feature_asof_utc"] = _time(s.feature_asof_utc)
    if "rule_fit_max_season" not in s:
        s["rule_fit_max_season"] = np.nan
    s["rule_fit_max_season"] = _as_num(s, "rule_fit_max_season")
    # A fired rule must attest the *same exact recipe* as the full no-fire
    # evaluation ledger. Missing/mismatched hashes are retained for diagnostics
    # but can never become independently certified supporters.
    if "rule_version_sha256" not in s:
        s["rule_version_sha256"] = ""
    s["rule_version_sha256"] = s.rule_version_sha256.fillna("").astype(str).str.strip()
    # System behavior discovered after the game may be described, but must NOT
    # be counted as independently validated live-period rule evidence.
    s = s.merge(games[["season", "game_id", "kickoff_utc", "actual_home_margin", "home_spread",
                       "home_ats_delta", "price_vintage_verified", "market_line_type"]],
                on=["season", "game_id"], how="left", validate="many_to_one", indicator=True)
    if s._merge.ne("both").any():
        raise ValueError("signals: orphan game_id or season mismatch")
    s = s.drop(columns="_merge")
    s = s.merge(coverage[["season", "brain", "coverage_complete", "status"]],
                on=["season", "brain"], how="left", validate="many_to_one")
    s = s.merge(eligibility[["season", "game_id", "brain", "evaluation_status", "game_brain_verified"]],
                on=["season", "game_id", "brain"], how="left", validate="many_to_one")
    if s.evaluation_status.ne("EVALUATED").any():
        raise ValueError("signals: a triggered rule belongs to an UNKNOWN/INELIGIBLE brain-game")
    s["feature_vintage_verified"] = (s.feature_asof_utc.notna() &
                                     (s.feature_asof_utc < s.kickoff_utc))
    s["rule_prior_only"] = s.rule_fit_max_season.notna() & s.rule_fit_max_season.lt(s.season)
    if rule_evaluations.empty:
        s["rule_event_verified"] = False
    else:
        s=s.merge(rule_evaluations[["season","game_id","brain","system_id",
                                    "evaluation_status","rule_game_verified",
                                    "feature_asof_utc","rule_fit_max_season",
                                    "rule_version_sha256"]],
                  on=["season","game_id","brain","system_id"],how="left",
                  validate="many_to_one",suffixes=("", "_rule"))
        if s.evaluation_status_rule.eq("INELIGIBLE").any():
            raise ValueError("signals: triggered rule declared INELIGIBLE on same game")
        s["rule_event_verified"]=(s.evaluation_status_rule.eq("EVALUATED") &
                                   _to_bool(s.rule_game_verified) &
                                   s.feature_asof_utc.eq(s.feature_asof_utc_rule) &
                                   s.rule_fit_max_season.eq(s.rule_fit_max_season_rule) &
                                   s.rule_version_sha256.str.fullmatch(r"[a-fA-F0-9]{64}") &
                                   s.rule_version_sha256.eq(s.rule_version_sha256_rule))
    s["system_verified"] = (s.feature_vintage_verified & s.rule_prior_only &
                            s.game_brain_verified.fillna(False) & s.rule_event_verified)
    # Grade, including unverified quotes, is descriptive. A clean system rule
    # and a certified bookmaker price are necessary for a qualifying test.
    home = s.home_ats_delta
    s["ats_delta"] = np.where(s.side.eq("HOME"), home, -home)
    s["ats_result"] = np.select([s.ats_delta.gt(0), s.ats_delta.lt(0), s.ats_delta.eq(0)],
                                ["WIN", "LOSS", "PUSH"], default="NO_LINE")
    s["ats_qualifies"] = s.system_verified & s.price_vintage_verified & s.ats_result.ne("NO_LINE")
    # Full-game historical straight-up directional grading is available even
    # where no archival bookmaker line survives. Never treat SU as ATS.
    home_margin = s.actual_home_margin
    s["su_delta"] = np.where(s.side.eq("HOME"), home_margin, -home_margin)
    s["su_result"] = np.select([s.su_delta.gt(0), s.su_delta.lt(0), s.su_delta.eq(0)],
                                ["WIN", "LOSS", "PUSH"], default="NO_SCORE")
    s["su_qualifies"] = s.system_verified & s.su_result.ne("NO_SCORE")
    return s


def prepare_core(frame: pd.DataFrame, games: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    _require(frame, REQUIRED_CORE, "core")
    c = _validate_years(frame, "core", cfg)
    _unique(c, ["season", "game_id"], "core")
    c["prediction_at_utc"] = _time(c.prediction_at_utc)
    c["model_home_margin"] = _as_num(c, "model_home_margin")
    c = c.merge(games[["season", "game_id", "kickoff_utc", "actual_home_margin", "home_spread",
                       "home_ats_delta", "price_vintage_verified", "market_line_type", "quote_at_utc"]],
                on=["season", "game_id"], how="left", validate="one_to_one", indicator=True)
    if c._merge.ne("both").any():
        raise ValueError("core: orphan game_id or season mismatch")
    c = c.drop(columns="_merge")
    c["prediction_asof_valid"] = (c.prediction_at_utc.notna() &
                                  c.prediction_at_utc.lt(c.kickoff_utc))
    # A CORE forecast is not independently historical merely because the
    # prediction timestamp is earlier than the game. Require the selected
    # model recipe, trained fold and feature provenance as of that timestamp.
    evidence = {"model_recipe_sha256", "recipe_selected_at_utc", "training_end_utc",
                "pregame_features_verified", "true_oos", "model_id"}
    if evidence.issubset(c.columns):
        recipe_time=_time(c.recipe_selected_at_utc)
        trained_time=_time(c.training_end_utc)
        recipe_hash=c.model_recipe_sha256.fillna("").astype(str).str.strip()
        same_model=c.model_id.astype(str).eq("NONLINEAR_HGB")
        c["core_vintage_verified"]=(c.prediction_asof_valid & same_model &
            recipe_hash.str.fullmatch(r"[a-fA-F0-9]{64}") &
            _to_bool(c.pregame_features_verified) & _to_bool(c.true_oos) &
            recipe_time.notna() & recipe_time.le(c.prediction_at_utc) &
            trained_time.notna() & trained_time.lt(c.prediction_at_utc) &
            trained_time.lt(c.kickoff_utc) &
            (~c.season.isin(CONFIRM_YEARS) | recipe_time.lt(pd.Timestamp("2024-01-01", tz="UTC"))))
    else:
        c["core_vintage_verified"]=False
    # Decision time is when BOTH a cached pregame model forecast and the
    # actual offered quote existed; it need not equal model-fit timestamp.
    # The game result and later closing quotes are not decision inputs.
    quote_time=_time(c.quote_at_utc)
    c["decision_at_utc"] = pd.concat([c.prediction_at_utc,quote_time],axis=1).max(axis=1)
    c["quote_available_at_decision"]=(c.price_vintage_verified &
         quote_time.notna() & c.decision_at_utc.lt(c.kickoff_utc))
    # Legacy diagnostic column retained for compatibility, not certification.
    c["quote_known_at_prediction"]=(quote_time.notna() & quote_time.le(c.prediction_at_utc))
    c["core_edge_signed"] = c.model_home_margin + c.home_spread
    c["core_edge_abs"] = c.core_edge_signed.abs()
    c["core_side"] = np.where(c.core_edge_signed.gt(0), "HOME", "AWAY")
    c["core_candidate"] = (c.core_edge_abs.ge(cfg.core_min_edge) &
                           c.prediction_asof_valid & c.model_home_margin.notna() & c.home_spread.notna())
    c["core_ats_delta"] = np.where(c.core_side.eq("HOME"), c.home_ats_delta, -c.home_ats_delta)
    c["core_result"] = np.select([c.core_ats_delta.gt(0), c.core_ats_delta.lt(0), c.core_ats_delta.eq(0)],
                                 ["WIN", "LOSS", "PUSH"], default="NO_LINE")
    return c


def _rate(rows: pd.DataFrame, result_col: str="ats_result") -> dict:
    r = rows[result_col].astype(str)
    wins, losses, pushes = int(r.eq("WIN").sum()), int(r.eq("LOSS").sum()), int(r.eq("PUSH").sum())
    n = wins + losses
    win_rate = wins/n if n else None
    p = float(binomtest(wins, n, 0.5, alternative="greater").pvalue) if n else None
    return {"events": len(rows), "wins": wins, "losses": losses, "pushes": pushes,
            "graded": n, "win_rate": win_rate, "p_one_sided_vs_half": p}


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    idx = [i for i,p in enumerate(p_values) if p is not None and np.isfinite(p)]
    by = sorted(idx, key=lambda i: p_values[i])
    n=len(by)
    result=[None]*len(p_values)
    prev=0.0
    for rank,i in enumerate(by):
        prev=max(prev, min(1.0, p_values[i]*(n-rank)))
        result[i]=prev
    return result


def per_system_reports(signals: pd.DataFrame, cfg: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All available 2013-2025 event rows remain in the ledger.

    Rules are never *selected* using the confirmation years. 2024/25 report
    is a locked, inspection-only test, not promotion or rule discovery.
    """
    columns=["brain", "system_id", "evidence_family", "phase", "first_year", "last_year", "years_with_events",
             "events", "wins", "losses", "pushes", "graded", "win_rate", "p_one_sided_vs_half",
             "p_holm", "price_vintage_verified_events", "system_verified_events", "su_graded", "su_wins", "su_losses", "su_win_rate", "qualified", "authority"]
    out=[]
    for (brain,sys,family), d in signals.groupby(["brain", "system_id", "evidence_family"],dropna=False):
        for phase, condition in (("full_history", d.season.between(cfg.research_min_year,cfg.research_max_year)),
                                 ("discovery", d.season.le(DISCOVERY_LAST)),
                                 ("selection", d.season.isin(SELECTION_YEARS)),
                                 ("confirmation", d.season.isin(CONFIRM_YEARS))):
            t=d.loc[condition]
            if t.empty: continue
            # Only events with point-in-time-certified inputs/price in inference.
            grade=t.loc[t.ats_qualifies]
            metrics=_rate(grade)
            su=_rate(t.loc[t.su_qualifies],"su_result")
            out.append({"brain":brain,"system_id":sys,"evidence_family":family,"phase":phase,
                        "first_year":int(t.season.min()),"last_year":int(t.season.max()),
                        "years_with_events":int(t.season.nunique()),"events":len(t),
                        "price_vintage_verified_events":int(t.price_vintage_verified.sum()),
                        "system_verified_events":int(t.system_verified.sum()),
                        "su_graded":su["graded"],"su_wins":su["wins"],"su_losses":su["losses"],
                        "su_win_rate":su["win_rate"],
                        **{k:v for k,v in metrics.items() if k!="events"},"p_holm":None,
                        "qualified":False,"authority":0})
    reports=pd.DataFrame(out,columns=columns)
    if reports.empty:
        return reports, pd.DataFrame()
    # Holm within an evaluation phase over all tested systems, not cherry-picked.
    for phase,ix in reports.groupby("phase").groups.items():
        adjusted=holm_adjust(reports.loc[ix,"p_one_sided_vs_half"].tolist())
        reports.loc[ix,"p_holm"] = pd.Series(adjusted,index=ix,dtype=object)
    # No authority auto-granted. Eligibility is descriptive even if qualified.
    confirm=reports.loc[reports.phase.eq("confirmation")].copy()
    if not confirm.empty:
        complete_years=(signals.loc[signals.season.isin(CONFIRM_YEARS)]
                        .groupby(["brain", "system_id", "evidence_family"])
                        .apply(lambda d: int(d.loc[d.ats_qualifies,"season"].nunique()),include_groups=False)
                        .rename("qualified_seasons").reset_index())
        confirm=confirm.merge(complete_years,on=["brain","system_id","evidence_family"],how="left")
        confirm["meets_evidence_floor"] = (confirm.graded.ge(cfg.min_system_confirmation_bets)
                                            & confirm.qualified_seasons.ge(cfg.min_confirmation_seasons))
        # Prospective promotion remains disabled, irrespective of results.
        reports=reports.merge(confirm[["brain","system_id","evidence_family","meets_evidence_floor"]],
                              on=["brain","system_id","evidence_family"],how="left")
    return reports, confirm


def combined_core_view(core: pd.DataFrame, signals: pd.DataFrame, eligibility: pd.DataFrame,
                       cfg: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Core-matched cohort only; independent system history is never trimmed."""
    c=core.loc[core.core_candidate].copy()
    if c.empty:
        return c, pd.DataFrame()
    # A game missing any brain's coverage CANNOT be mistaken for 0 signals.
    wide=(eligibility.groupby(["season","game_id"])["game_brain_verified"].all())
    c["all_brains_covered"] = [bool(wide.get((int(y),str(id)),False)) for y,id in zip(c.season,c.game_id)]
    lookup={(int(season),str(game_id)):group for (season,game_id),group in signals.groupby(["season","game_id"])}
    details=[]
    for _, row in c.iterrows():
        d=lookup.get((int(row.season),str(row.game_id)))
        if d is None or d.empty:
            supporters=opponents=set()
            certified_support=certified_oppose=set()
            families=[]
            supporting=[]
            opposing=[]
        else:
            support=d.loc[d.side.eq(row.core_side)]
            oppose=d.loc[d.side.ne(row.core_side)]
            # Evidence independence: repeated Miner systems / related brain
            # calculations belong to one family and count once.
            supporters=set(support.evidence_family.dropna())
            opponents=set(oppose.evidence_family.dropna())
            certified_support=set(support.loc[support.ats_qualifies,"evidence_family"].dropna())
            certified_oppose=set(oppose.loc[oppose.ats_qualifies,"evidence_family"].dropna())
            families=sorted(set(d.evidence_family.dropna()))
            supporting=sorted(supporters)
            opposing=sorted(opponents)
        if not bool(row.all_brains_covered):
            classification="COVERAGE_UNKNOWN"
        elif opponents:
            classification="VETO_CONFLICT" if supporters else "VETO_OPPOSITION"
        elif len(supporters)>=2:
            classification="MULTI_SUPPORT"
        elif len(supporters)==1:
            classification="SINGLE_SUPPORT"
        else:
            classification="NO_SIGNAL"
        # 'certified' is even stricter: every data family used has pregame
        # provenance, verified quote, and rule trained only on prior seasons.
        certification=(bool(row.core_vintage_verified) and bool(row.quote_available_at_decision) and bool(row.all_brains_covered)
                       and bool(d is None or d.system_verified.all())
                       and bool(d is None or d.ats_qualifies.all()))
        details.append({"classification":classification,"support_families":len(supporters),
                        "opposition_families":len(opponents),"certified_support_families":len(certified_support),
                        "certified_opposition_families":len(certified_oppose),
                        "families_seen":"|".join(families),"supporting_families":"|".join(supporting),
                        "opposing_families":"|".join(opposing),"research_certification":certification})
    c=pd.concat([c.reset_index(drop=True),pd.DataFrame(details)],axis=1)
    c["edge_bucket"] = pd.cut(c.core_edge_abs,[0,2,4,6,8,np.inf],include_lowest=True,
                              labels=["0-<2","2-<4","4-<6","6-<8","8+"])
    c["role"] = np.where(((c.home_spread < 0)&c.core_side.eq("HOME")) |
                         ((c.home_spread > 0)&c.core_side.eq("AWAY")),"FAVORITE",
                         np.where(c.home_spread.eq(0),"PICKEM","UNDERDOG"))
    rows=[]
    for label, d in list(c.groupby("classification",dropna=False)) + [("CORE_BASELINE",c)]:
        stats=_rate(d, "core_result")
        confirmed=d.loc[d.research_certification]
        verified=_rate(confirmed,"core_result")
        rows.append({"cohort":label,**stats,"certified_events":len(confirmed),
                     "certified_wins":verified["wins"],"certified_losses":verified["losses"],
                     "certified_win_rate":verified["win_rate"],"authority":0})
    return c,pd.DataFrame(rows)


def _cohort_split(core_matched: pd.DataFrame) -> pd.DataFrame:
    if core_matched.empty: return pd.DataFrame()
    rows=[]
    for dims in [["season","classification"],["classification","edge_bucket"],["classification","role"]]:
        for keys, t in core_matched.groupby(dims,dropna=False,observed=True):
            x=dict(zip(dims,keys if isinstance(keys,tuple) else (keys,)))
            rows.append({"dimension":"|".join(dims),**x,**_rate(t,"core_result")})
    return pd.DataFrame(rows)



def combination_report(matched: pd.DataFrame, min_selection_n: int = 20) -> pd.DataFrame:
    """Discover support-family pairs on 2022-23, inspect locked 2024-25 ONLY.

    Pairs are defined on certified pregame signals without conflicting signals.
    No use of 2024/25 targets to generate/select pair definitions. The output
    never enables production promotion, even if a sample passes a p-value.
    """
    fields=["family_a","family_b","selection_n","selection_wins","selection_losses",
            "selection_win_rate","selection_vs_core_baseline_pp","selection_p_raw",
            "selection_p_holm","selected_before_confirmation","confirmation_n",
            "confirmation_wins","confirmation_losses","confirmation_win_rate",
            "confirmation_vs_core_baseline_pp","authority"]
    if matched.empty:
        return pd.DataFrame(columns=fields)
    d=matched.loc[matched.research_certification &
                  matched.opposition_families.eq(0) &
                  matched.season.isin((*SELECTION_YEARS,*CONFIRM_YEARS))].copy()
    if d.empty:
        return pd.DataFrame(columns=fields)
    d["pairs"]=d.supporting_families.map(lambda x:
        list(itertools.combinations(sorted(set(filter(None,str(x).split("|")))),2)))
    all_selection= d.loc[d.season.isin(SELECTION_YEARS)]
    all_confirmation=d.loc[d.season.isin(CONFIRM_YEARS)]
    sel_baseline=_rate(all_selection,"core_result")["win_rate"]
    conf_baseline=_rate(all_confirmation,"core_result")["win_rate"]
    possible=sorted({pair for arr in all_selection.pairs for pair in arr})
    out=[]
    for pair in possible:
        train=all_selection.loc[all_selection.pairs.map(lambda x:pair in x)]
        conf=all_confirmation.loc[all_confirmation.pairs.map(lambda x:pair in x)]
        a=_rate(train,"core_result")
        b=_rate(conf,"core_result")
        out.append(dict(family_a=pair[0],family_b=pair[1],
           selection_n=a["graded"],selection_wins=a["wins"],selection_losses=a["losses"],
           selection_win_rate=a["win_rate"],selection_vs_core_baseline_pp=(
               100*(a["win_rate"]-sel_baseline) if a["win_rate"] is not None and sel_baseline is not None else None),
           selection_p_raw=a["p_one_sided_vs_half"],selection_p_holm=None,
           selected_before_confirmation=False,
           confirmation_n=b["graded"],confirmation_wins=b["wins"],confirmation_losses=b["losses"],
           confirmation_win_rate=b["win_rate"],confirmation_vs_core_baseline_pp=(
               100*(b["win_rate"]-conf_baseline) if b["win_rate"] is not None and conf_baseline is not None else None),
           authority=0))
    result=pd.DataFrame(out,columns=fields)
    if result.empty: return result
    result["selection_p_holm"] = holm_adjust(result.selection_p_raw.tolist())
    result["selected_before_confirmation"] = (result.selection_n.ge(min_selection_n) &
        result.selection_p_holm.notna() & result.selection_p_holm.le(0.10) &
        result.selection_vs_core_baseline_pp.gt(0))
    # Confirmation results are only interpretable for preselected pairs.
    # Keep other pairs visible but never treat their confirmation stats as OOS.
    return result


def per_year_systems(signals: pd.DataFrame, coverage: pd.DataFrame,
                     rule_audit: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """Explicit season cells per known system; no-fire is not presumed verified.

    A rule-level evaluation manifest is additionally necessary before claiming
    a particular rule was checked on every no-trigger game.
    """
    rows=[]
    audited={(int(z.season),str(z.brain),str(z.system_id)):z
             for z in rule_audit.itertuples(index=False)} if not rule_audit.empty else {}
    # Enumerate rules from the full evaluation manifest, even rules that
    # NEVER FIRED in thirteen seasons. Ignoring nonfiring rules would create
    # survivorship bias and falsely shrink the tested-system universe.
    known = signals[["brain","system_id","evidence_family"]].drop_duplicates().copy()
    if not rule_audit.empty:
        extra=rule_audit[["brain","system_id"]].drop_duplicates().merge(
            known,on=["brain","system_id"],how="left")
        extra["evidence_family"]=extra.evidence_family.fillna("UNKNOWN_RULE_FAMILY")
        known=pd.concat([known,extra],ignore_index=True).drop_duplicates()
    for brain,sys,family in known.itertuples(index=False,name=None):
        events=signals.loc[signals.brain.eq(brain) & signals.system_id.eq(sys) &
                           signals.evidence_family.eq(family)]
        cover=coverage.loc[coverage.brain.eq(brain)].set_index("season")
        for year in range(cfg.research_min_year,cfg.research_max_year+1):
            e=events.loc[events.season.eq(year)]
            x=e.loc[e.ats_qualifies]
            met=_rate(x)
            complete=bool(cover.loc[year,"coverage_complete"])
            xrule=audited.get((year,str(brain),str(sys)))
            rule_complete=bool(xrule.rule_year_complete) if xrule is not None else False
            rows.append({"season":year,"brain":brain,"system_id":sys,"evidence_family":family,
                         "brain_full_replay":complete,"rule_replay_verified":rule_complete,
                         "rule_games_checked":int(xrule.rule_games_accounted) if xrule is not None else None,
                         "rule_evaluated_games":int(xrule.rule_evaluated_games) if xrule is not None else None,
                         "fired_events_observed":len(e),
                         "fired_events_certified":len(x),
                         "no_fire_count_known":rule_complete,
                         "observed_no_fire_count":(int(xrule.rule_evaluated_games)-len(e)) if rule_complete else None,
                         "rule_level_evaluation_manifest":"VERIFIED" if rule_complete else "MISSING_OR_PARTIAL",
                         **{k:v for k,v in met.items() if k!="events"},"authority":0})
    return pd.DataFrame(rows)


def report(games: pd.DataFrame, signals: pd.DataFrame, coverage: pd.DataFrame,
           core: pd.DataFrame, cfg: RunConfig, eligibility: pd.DataFrame | None = None,
           universe_manifest: pd.DataFrame | None = None,
           rule_evaluations: pd.DataFrame | None = None) -> dict[str,Any]:
    g=prepare_games(games,cfg)
    audit=universe_audit(g,universe_manifest,cfg)
    cv,_=prepare_coverage(coverage,g,cfg)
    cv=cv.merge(audit[["season","universe_verified"]],on="season",how="left",validate="many_to_one")
    cv["coverage_complete"] &= cv.universe_verified
    elig,cv=prepare_eligibility(eligibility,g,cv,cfg)
    coverage_summary=(cv.groupby("brain",as_index=False).agg(
        years_total=("season","size"),years_complete=("coverage_complete","sum"),
        eligible_games_reported=("eligible_games","sum"),
        feature_rows_reported=("feature_rows","sum")))
    coverage_summary["full_history_certified"] = coverage_summary.years_complete.eq(
        cfg.research_max_year-cfg.research_min_year+1)
    replays,rule_audit=prepare_rule_evaluations(rule_evaluations,elig,cfg)
    s=prepare_signals(signals,g,cv,elig,replays,cfg)
    c=prepare_core(core,g,cfg)
    system_summary, confirmation=per_system_reports(s,cfg)
    matched,cohort=combined_core_view(c,s,elig,cfg)
    combos=combination_report(matched)
    year_systems=per_year_systems(s,cv,rule_audit,cfg)
    stats={"version":VERSION,"mode":"SHADOW_ONLY", "bet_authority":0,
           "system_history_window":[cfg.research_min_year,cfg.research_max_year],
           "locked_confirmation_seasons":list(CONFIRM_YEARS),
           "2026_outcomes":"SEALED", "production_mutation":False,
           "universe_games":len(g),"universe_years":int(g.season.nunique()),
           "universe_archive_years_verified":int(audit.universe_verified.sum()),
           "core_predictions":len(c),"core_candidate_games":len(matched),
           "all_brains_full_coverage":bool(coverage_summary.full_history_certified.all()),
           "brain_events":len(s),"brain_events_system_vintage_verified":int(s.system_verified.sum()),
           "rule_years_certified":int(rule_audit.rule_year_complete.sum()) if not rule_audit.empty else 0,
           "rule_year_cells_missing_or_partial":int((~year_systems.rule_replay_verified).sum()) if not year_systems.empty else 0,
           "brain_events_ats_qualifying":int(s.ats_qualifies.sum()),
           "brain_events_su_qualifying":int(s.su_qualifies.sum()),
           "market_quotes_certified":int(g.price_vintage_verified.sum()),
           "close_fallback_descriptive_games":int(g.market_line_type.eq("CLOSE_FALLBACK_DESCRIPTIVE_ONLY").sum()),
           "core_games_full_brain_coverage":int(matched.all_brains_covered.sum()) if not matched.empty else 0,
           "errors":[],"warnings":[]}
    if not audit.universe_verified.all():
        stats["warnings"].append("CANONICAL_CFBD_GAME_UNIVERSE_NOT_INDEPENDENTLY_VERIFIED")
    if not stats["all_brains_full_coverage"]:
        stats["warnings"].append("INCOMPLETE_FULL_HISTORY_BRAIN_COVERAGE__MISSING_NEVER_EQUALS_ZERO_TRIGGERS")
    if not s.ats_qualifies.any():
        stats["warnings"].append("NO_CERTIFIED_POINT_IN_TIME_BRAIN_AND_PRICE_EVENTS__EXPLORATORY_ONLY")
    if rule_audit.empty or not rule_audit.rule_year_complete.all():
        stats["warnings"].append("RULE_LEVEL_NO_FIRE_DENOMINATORS_REQUIRE_COMPLETE_RULE_EVALUATIONS")
    if not year_systems.empty and (~year_systems.rule_replay_verified).any():
        stats["warnings"].append("SOME_SYSTEM_YEAR_CELLS_LACK_COMPLETE_RULE_REPLAY__FULL_HISTORY_UNVERIFIED")
    if not g.price_vintage_verified.all():
        stats["warnings"].append("SOME_MARKET_LINES_UNVERIFIED__NO_SPORTSBOOK_RETURN_CLAIMS")
    stats["core_predictions_vintage_verified"] = int(c.core_vintage_verified.sum())
    stats["core_candidates_vintage_verified"] = int(c.loc[c.core_candidate,"core_vintage_verified"].sum())
    if bool(c.core_candidate.any()) and not bool(c.loc[c.core_candidate,"core_vintage_verified"].all()):
        stats["warnings"].append("CORE_PREDECISION_MODEL_RECIPE_UNVERIFIED__NO_CERTIFIED_COMBINED_BETS")
    result={"status":stats,"games":g,"universe_audit":audit,"coverage_matrix":cv,"coverage_by_brain":coverage_summary,
            "brain_game_eligibility":elig,"rule_evaluations":replays,"rule_year_audit":rule_audit,
            "signals_ledger":s,"systems_by_period":system_summary,
            "systems_by_year":year_systems,"brain_pairs":combos,
            "holdout_review":confirmation,"core_matched":matched,
            "combined_cohorts":cohort,"combined_splits":_cohort_split(matched)}
    return result


def write_output(result: dict[str,Any], outdir: str | Path) -> Path:
    p=Path(outdir);p.mkdir(parents=True,exist_ok=True)
    for name,frame in result.items():
        if isinstance(frame,pd.DataFrame):
            frame.to_csv(p/f"{name}.csv",index=False)
    # hashes bind status to reproducible artifacts; never falsely mark verified
    digests={n:hashlib.sha256((p/f"{n}.csv").read_bytes()).hexdigest()
             for n,v in result.items() if isinstance(v,pd.DataFrame)}
    status=dict(result["status"]);status["artifact_sha256"]=digests
    (p/"status.json").write_text(json.dumps(status,indent=2,default=str)+"\n")
    return p


def main() -> int:
    a=argparse.ArgumentParser(description=VERSION)
    for source in ("games","core","signals","coverage"):
        a.add_argument(f"--{source}",required=True)
    a.add_argument("--eligibility",help="Per-game brain eligibility matrix (required for coverage certification)")
    a.add_argument("--universe-manifest",help="Independently generated physical CFBD game identity manifest")
    a.add_argument("--rule-evaluations",help="Per-rule per-game complete nonfiring replay ledger")
    a.add_argument("--out",required=True)
    a.add_argument("--core-min-edge",type=float,default=2.0)
    a.add_argument("--min-confirmation-bets",type=int,default=80)
    args=a.parse_args()
    config=RunConfig(core_min_edge=args.core_min_edge,min_system_confirmation_bets=args.min_confirmation_bets)
    result=report(_read(args.games),_read(args.signals),_read(args.coverage),_read(args.core),config,
                  _read(args.eligibility) if args.eligibility else None,
                  _read(args.universe_manifest) if args.universe_manifest else None,
                  _read(args.rule_evaluations) if args.rule_evaluations else None)
    p=write_output(result,args.out)
    print(json.dumps({**result["status"],"output":str(p)},indent=2,default=str))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
