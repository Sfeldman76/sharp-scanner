"""Source-aware, *retrospective* NCAAF brain replay. Research only.

Unlike the certified ncaaf_brain_full_history contract, this module reconstructs
fixed research hypotheses *today* against historic features. It cannot prove the
rules, prices, forecasts or archived PT files were available on historical dates.
It never supplies certified inputs to the strict validator and never promotes bets.

Coverage is per-rule/per-season. Every eligible/nonfire/unknown decision is explicit.
The immutable CFBD archive feature game IDs form the universe independent of the
legacy STAT/market training view; games without those joins remain in the universe.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
import re
from typing import Any

import numpy as np
import pandas as pd

VERSION = "NCAAF_BRAIN_RETROSPECTIVE_REPLAY_V1_20261024"
MIN_YEAR, MAX_YEAR = 2013, 2025
PT_YEARS = range(2022, 2026)
SPLITS = {**{y:"DISCOVERY" for y in range(2013, 2022)},
          2022:"SELECTION", 2023:"SELECTION", 2024:"CONFIRMATION", 2025:"CONFIRMATION"}
# This is a post-2025 rule definition, so even 2024-25 is NOT an independent
# holdout against these newly proposed hypotheses. No split can certify it.
RULE_DEFINED_AFTER_HOLDOUT = True
CONTEXT_COLUMNS = ["Season","Source_Game_ID","Game_Date","Game_Start","Team","Opponent", "Is_Home",
   "SU_Margin","Team_Score","Opponent_Score","Opening_Spread","Spread_Value", "Opening_Total",
   "Team_Game_Number", "Prev_SU_Loss", "Prev_SU_Win", "Prev_ATS_Loss", "Prev_ATS_Win",
   "WinPct_Prior_System", "ATS_WinPct_Prior_System", "Opp_WinPct_Prior_System",
   "Revenge_Flag_Current", "Opp_Revenge_Flag_Current","Is_Conference_Game",
   "Pathi_FB_On_Key_3","Pathi_FB_On_Key_7","Pathi_FB_On_Key_10","Pathi_FB_On_Key_14",
   "Pathi_FB_Crossed_Key_Toward_Team","Pathi_FB_Crossed_Key_Away_From_Team",
   "Pathi_FB_Moved_Onto_Key","Pathi_FB_Moved_Off_Key","Pathi_FB_Moved_Through_Key"]
CFBD_COLUMNS = ["Season","Game_ID","Game_Date","Team","Opponent","Is_Home",
   "CFBD_Games_Prior","CFBD_Season_Net_PPA_Prior","CFBD_Matchup_Net_PPA_Adv",
   "CFBD_Matchup_Net_Success_Adv", "CFBD_Recent3_Off_PPA_Prior", "CFBD_Season_Off_PPA_Prior"]

@dataclass(frozen=True)
class Rule:
    brain: str
    name: str
    source: str
    fields: tuple[str, ...]
    # The target is the side for which ALL predicates are expressed: HOME/AWAY
    target: str
    criteria: tuple[tuple[str, str, float], ...]
    notes: str

# Small *explicit* research hypotheses, NOT a reproduction of the 191 approved
# or published Miner mechanisms. Thresholds were not frozen before the games.
RULES = (
    Rule("STAT","HOME_OFF_SU_LOSS","CONTEXT",("Prev_SU_Loss",),"HOME",(("Prev_SU_Loss","eq",1),),"Previous game straight-up loss"),
    Rule("STAT","HOME_OFF_ATS_LOSS","CONTEXT",("Prev_ATS_Loss",),"HOME",(("Prev_ATS_Loss","eq",1),),"Previous game ATS loss; historical ATS inputs may be source-vintage unverified"),
    Rule("STAT","HOME_RECENT_WINPCT_LOW","CONTEXT",("WinPct_Prior_System",),"HOME",(("WinPct_Prior_System","le",0.4),),"Prior record <= .400"),
    Rule("STAT","HOME_REVENGE","CONTEXT",("Revenge_Flag_Current",),"HOME",(("Revenge_Flag_Current","eq",1),),"Revenge condition based on past opponents"),
    Rule("SYSTEM_MINER","CFBD_HOME_NET_PPA_ADV","CFBD",("CFBD_Games_Prior","CFBD_Matchup_Net_PPA_Adv"),"HOME",(("CFBD_Games_Prior","ge",3),("CFBD_Matchup_Net_PPA_Adv","ge",0.08)),"CFBD prior-game matchup PPA research hypothesis"),
    Rule("SYSTEM_MINER","CFBD_HOME_NET_SUCCESS_ADV","CFBD",("CFBD_Games_Prior","CFBD_Matchup_Net_Success_Adv"),"HOME",(("CFBD_Games_Prior","ge",3),("CFBD_Matchup_Net_Success_Adv","ge",0.05)),"CFBD prior-game matchup success research hypothesis"),
    Rule("SYSTEM_MINER","CFBD_HOME_NET_PPA_DISADV","CFBD",("CFBD_Games_Prior","CFBD_Matchup_Net_PPA_Adv"),"AWAY",(("CFBD_Games_Prior","ge",3),("CFBD_Matchup_Net_PPA_Adv","le",-0.08)),"Opposing-side CFBD PPA research hypothesis"),
    Rule("PATHI","OPEN_HOME_KEY_3","CONTEXT",("Opening_Spread",),"HOME",(("Opening_Spread","abs_eq",3),),"Opening line exactly +/-3; does not imply directional edge"),
    Rule("PATHI","OPEN_HOME_KEY_7","CONTEXT",("Opening_Spread",),"HOME",(("Opening_Spread","abs_eq",7),),"Opening line exactly +/-7; does not imply directional edge"),
    Rule("PATHI","OPEN_HOME_KEY_10","CONTEXT",("Opening_Spread",),"HOME",(("Opening_Spread","abs_eq",10),),"Opening line exactly +/-10; does not imply directional edge"),
)
# PATHI on-key predicates and other descriptive non-directional context are not
# allowed to act as confirmation votes; only evaluate timing/coverage/ATS slices.
NON_DIRECTIONAL = {r.name for r in RULES if r.brain == "PATHI"}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _str(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip()


def _key(s: pd.Series) -> pd.Series:
    return _str(s).str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def _home_mask(s: pd.Series) -> pd.Series:
    return _str(s).str.lower().isin(("1", "1.0", "true", "yes"))


def _checked_source(source: pd.DataFrame, key: str, cols: list[str]) -> pd.DataFrame:
    if not isinstance(source, pd.DataFrame):
        raise ValueError(f"REPLAY_SOURCE_MISSING:{key}")
    absent = set(cols) - set(source.columns)
    if absent:
        raise ValueError(f"REPLAY_SOURCE_SCHEMA_MISMATCH:{key}:{','.join(sorted(absent))}")
    x = source[cols].copy()
    years = _num(x["Season"])
    return x[years.between(MIN_YEAR, MAX_YEAR)].copy()


def _stage_cfbd(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    x = _checked_source(frame, "cfbd", CFBD_COLUMNS)
    x["season"] = _num(x.Season).astype(int)
    x["game_id"] = _str(x.Game_ID)
    x["kickoff_utc"] = _dates(x.Game_Date)
    x["home_team_key"] = _key(x.Team)
    x["away_team_key"] = _key(x.Opponent)
    all_sides = x[x.game_id.ne("")].copy()
    side_counts = all_sides.groupby(["season","game_id"]).size()
    duplicate_or_single_sides = int(side_counts.ne(2).sum())
    x = x[_home_mask(x.Is_Home)].copy()
    x = x[x.game_id.ne("") & x.kickoff_utc.notna() & x.home_team_key.ne("") & x.away_team_key.ne("")].copy()
    if x.duplicated(["season", "game_id"]).any():
        raise ValueError("CFBD_DUPLICATE_HOME_GAME_ID")
    x["game_date_key"] = x.kickoff_utc.dt.strftime("%Y-%m-%d")
    # Independent universe: CFBD games cannot be dropped for lack of STAT/CORE.
    x = x.rename(columns={col:"src_"+col for col in CFBD_COLUMNS if col.startswith("CFBD_")})
    cols = ["season", "game_id", "kickoff_utc", "game_date_key", "home_team_key", "away_team_key"]
    universe = x[cols].copy()
    manifest = universe.groupby("season").agg(expected_games=("game_id", "size")).reset_index()
    manifest["game_ids_sha256"] = manifest.season.map(universe.groupby("season").game_id.apply(
        lambda ids:hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()))
    manifest["source"] = "BIGQUERY_CFBD_ARCHIVE_RESEARCH_V1__GAME_IDENTITY_RECONSTRUCTED"
    manifest["independent_of_context"] = True
    manifest["historical_source_vintage_certified"] = False
    return universe, x, {"source":"ncaaf_cfbd_archive_research_v1", "expected_by_year":dict(zip(manifest.season.astype(str),manifest.expected_games.astype(int))), "games":len(universe), "source_games_without_exactly_two_sides":duplicate_or_single_sides,"manifest":manifest}


def _stage_context(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x = _checked_source(frame, "context", CONTEXT_COLUMNS)
    x["season"] = _num(x.Season).astype(int)
    x["source_game_id"] = _str(x.Source_Game_ID)
    x["ctx_home_key"] = _key(x.Team)
    x["ctx_away_key"] = _key(x.Opponent)
    x["ctx_kickoff"] = _dates(x.Game_Date)
    x = x[_home_mask(x.Is_Home) & x.source_game_id.ne("") & x.ctx_kickoff.notna()].copy()
    if x.duplicated(["season", "source_game_id"]).any():
        raise ValueError("CONTEXT_DUPLICATE_HOME_GAME_ID")
    # No implicit reinterpretation of a 2026-ingested source as a 2013 snapshot.
    x["context_source_vintage_certified"] = False
    return x, {"home_rows":len(x), "historical_vintage_certified":False}


def _attach_context(universe: pd.DataFrame, context: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    c = context.copy()
    c["ctx_game_date_key"] = c.ctx_kickoff.dt.strftime("%Y-%m-%d")
    # First exact season/game-id/teams; fallback exactly same date+home+away,
    # ONLY where both the CFBD and context keys are unique. No fuzzy matching.
    ids = universe.merge(c, left_on=["season","game_id","home_team_key","away_team_key"],
                         right_on=["season","source_game_id","ctx_home_key","ctx_away_key"],
                         how="left", validate="one_to_one",indicator=True)
    direct = ids._merge.eq("both")
    missing = ids.loc[~direct,universe.columns]
    used = set(zip(ids.loc[direct,"season"].astype(int), ids.loc[direct,"source_game_id"].astype(str)))
    rem = c.loc[~pd.Series(list(zip(c.season.astype(int), c.source_game_id.astype(str))), index=c.index).isin(used)].copy()
    left_keys = ["season","game_date_key","home_team_key","away_team_key"]
    right_keys = ["season","ctx_game_date_key","ctx_home_key","ctx_away_key"]
    # Duplicate alternatives are ambiguous, fail closed (not arbitrary drop_duplicates).
    left_unique = missing.loc[~missing.duplicated(left_keys,keep=False)]
    right_unique = rem.loc[~rem.duplicated(right_keys,keep=False)]
    by_date = left_unique.merge(right_unique,left_on=left_keys,right_on=right_keys,
                                how="left",validate="one_to_one",indicator=True)
    by_date = by_date[by_date._merge.eq("both")].drop(columns="_merge")
    direct_full = ids.loc[direct].drop(columns="_merge")
    # direct and fallback share right-hand column set; data stays aligned.
    all_matched = pd.concat([direct_full,by_date],ignore_index=True)
    if all_matched.duplicated(["season","game_id"]).any():
        raise ValueError("CONTEXT_DOUBLE_ASSIGNED_CFBD_GAME")
    # Use a stable physical CFBD key; preserve every universe member.
    b = all_matched.drop(columns=[col for col in universe.columns if col not in ("season","game_id")])
    result = universe.merge(b,on=["season","game_id"],how="left",validate="one_to_one")
    result["has_context"] = result.source_game_id.notna()
    result["actual_home_margin"] = _num(result.SU_Margin) if "SU_Margin" in result else np.nan
    if "Team_Score" in result:
        alt=_num(result.Team_Score)-_num(result.Opponent_Score)
        result["actual_home_margin"] = result.actual_home_margin.where(result.actual_home_margin.notna(),alt)
    result["open_home_spread"] = _num(result.Opening_Spread)
    result["open_line_source_vintage_certified"] = False
    return result, {"matched_context_games":int(result.has_context.sum()),
                    "unmatched_cfbd_games":int((~result.has_context).sum()),
                    "matched_by_strict_id_team":int(direct.sum()),
                    "matched_by_exact_date_teams":len(by_date),
                    "game_identity_ambiguous_or_unmatched":int((~result.has_context).sum())}


def _compare(v: pd.Series, op: str, target: float) -> pd.Series:
    if op == "eq": return v.eq(target)
    if op == "ge": return v.ge(target)
    if op == "le": return v.le(target)
    if op == "abs_eq": return v.abs().sub(target).abs().lt(1e-8)
    raise ValueError("INVALID_RULE_OPERATOR:"+op)


def replay_rules(games: pd.DataFrame, cfbd: pd.DataFrame, rules: tuple[Rule,...]=RULES) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Emit one EVALUATED_FIRE/NO_FIRE/UNKNOWN record for every rule/game.

    A false predicate is never inferred from a null; history on the full CFBD
    universe is preserved even where STAT/Pathi data is absent.
    """
    base = games[["season","game_id","kickoff_utc","actual_home_margin","open_home_spread"]].copy()
    if len(cfbd):
        fields = ["season","game_id"]+[c for c in cfbd if c.startswith("src_CFBD_")]
        # src_ columns already prefixed to avoid accidental same-name confusion.
        base=base.merge(cfbd[fields],on=["season","game_id"],how="left",validate="one_to_one")
    for col in set(x for rule in rules if rule.source=="CONTEXT" for x in rule.fields):
        base[col] = games[col] if col in games else np.nan
    outcomes=[]
    for rule in rules:
        vals=[_num(base["src_"+f if rule.source=="CFBD" else f]) if ("src_"+f if rule.source=="CFBD" else f) in base else pd.Series(np.nan,index=base.index) for f in rule.fields]
        known = pd.Series(True,index=base.index)
        for v in vals: known &= v.notna()
        fire = known.copy()
        for field,operator,threshold in rule.criteria:
            lookup = "src_"+field if rule.source=="CFBD" else field
            fire &= _compare(_num(base[lookup]),operator,threshold)
        record=base[["season","game_id","kickoff_utc","actual_home_margin","open_home_spread"]].copy()
        record["brain"]=rule.brain
        record["system_id"]="HYP_"+rule.name
        record["direction"] = rule.target if rule.name not in NON_DIRECTIONAL else "NONE"
        record["evaluation_status"] = np.where(~known,"UNKNOWN",np.where(fire,"FIRE","NO_FIRE"))
        record["evidence_type"]="POST_HOC_RETROSPECTIVE_HYPOTHESIS"
        record["rule_version_sha256"]=_hash({"rule":rule.__dict__,"version":VERSION})
        record["source_vintage_certified"]=False
        record["rule_frozen_before_game"]=False
        record["market_vintage_certified"]=False
        record["split"] = record.season.map(SPLITS)
        record["outcome_available"] = record.actual_home_margin.notna()
        record["descriptive_ats_result"] = np.nan
        avail=fire & record.outcome_available & record.open_home_spread.notna() & record.direction.ne("NONE")
        chosen=np.where(record.direction.eq("HOME"),1,-1)
        delta=(_num(record.actual_home_margin)+_num(record.open_home_spread))*chosen
        record.loc[avail,"descriptive_ats_result"] = np.sign(delta[avail])
        outcomes.append(record)
    ledger=pd.concat(outcomes,ignore_index=True)
    summaries=(ledger.groupby(["brain","system_id","season","split","evaluation_status"],dropna=False)
       .size().rename("games").reset_index())
    return ledger,summaries


def pt_archive_inputs(storage_client, bucket_name: str, *, parse_func=None) -> tuple[pd.DataFrame,dict]:
    """Read original GCS raw-year blobs; do not claim uploaded-at == published-at.

    Reuse the repository's strict named-header PT parser so we cannot accidentally
    treat an unnamed column or missing PI as a source; no exact-five requirement.
    """
    frames=[];audit={"files":{},"years_present":[],"historical_published_at_verified":False}
    for year in PT_YEARS:
        key=f"research/ncaaf/external/prediction_tracker/raw/ncaa{year}.csv"
        blob=storage_client.bucket(bucket_name).blob(key)
        if not blob.exists():
            audit["files"][str(year)]={"status":"NOT_PRESENT"};continue
        try:
            payload=blob.download_as_bytes()
            if parse_func is None:
                from ncaaf_research_v2 import _pt_parse_csv
                parse_func=_pt_parse_csv
            parsed,meta=parse_func(payload,year,log_func=lambda *_: None)
            if parsed.empty:
                audit["files"][str(year)]={"status":"PARSE_EMPTY_OR_INVALID", "detail":meta.get("status")};continue
            parsed=parsed.copy()
            parsed["_pt_archive_sha256"]=hashlib.sha256(payload).hexdigest()
            parsed["_pt_retro_from_gcs"] = True
            frames.append(parsed)
            audit["files"][str(year)]={"status":"RECONSTRUCTED_RETROSPECTIVE", "rows":len(parsed), "sha256":hashlib.sha256(payload).hexdigest(),"original_publication_time_verified":False}
            audit["years_present"].append(year)
        except Exception as exc:
            audit["files"][str(year)]={"status":"READ_OR_PARSE_FAILED","error":str(exc)[:200]}
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(),audit


def attach_pt(games: pd.DataFrame, parsed: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    """Explicit HOME/AWAY name + local calendar date; unambiguous match only.

    Historical UTC offsets may put CFBD games on different dates. Without an
    independently verified local date source, mismatches remain unmatched.
    """
    if parsed.empty:
        return pd.DataFrame(columns=["season","game_id","brain","system_id","direction","evidence_type","market_vintage_certified"]),{"matched_games":0,"unmatched_pt_rows":0}
    p=parsed.copy()
    necessary={"season","home_key","away_key","game_date"}
    if not necessary.issubset(p):raise ValueError("PT_NORMALIZED_SCHEMA_NOT_MATCHABLE")
    p["date_key"] = p.game_date.astype("string").fillna("").str[:10]
    p["home_norm"] = _key(p.home_key)
    p["away_norm"] = _key(p.away_key)
    p=p.loc[p.date_key.str.fullmatch(r"\d{4}-\d{2}-\d{2}") & p.home_norm.ne("") & p.away_norm.ne("")].copy()
    keys=["season","date_key","home_norm","away_norm"]
    p=p.loc[~p.duplicated(keys,keep=False)]
    u=games[["season","game_id","game_date_key","home_team_key","away_team_key","open_home_spread"]].copy()
    u.columns=["season","game_id","date_key","home_norm","away_norm","open_home_spread"]
    u=u.loc[~u.duplicated(keys,keep=False)]
    x=u.merge(p,on=keys,how="inner",validate="one_to_one")
    # PT predictor-vs-spread direction is exploratory: orientations of raw
    # source-native numeric predictions are not certified across years.
    forecast=_num(x["prediction_avg_home"]) if "prediction_avg_home" in x else pd.Series(np.nan,index=x.index)
    openline=_num(x.open_home_spread)
    # sign of forecast + line is a candidate relative-to-market indication,
    # NOT a certified source-side or dated bet, even where source has numbers.
    edge=forecast+openline
    x["brain"]="PT"
    x["system_id"]="PT_AVERAGE_VS_CONTEXT_OPEN__RETROSPECTIVE"
    x["direction"]=np.where(edge>0,"HOME",np.where(edge<0,"AWAY","NONE"))
    x["evaluation_status"]=np.where(edge.notna(),"FIRE","UNKNOWN")
    x["evidence_type"]="PT_ARCHIVE_UPLOAD_2026__PREGAME_PUBLICATION_UNVERIFIED"
    x["market_vintage_certified"]=False
    x["source_vintage_certified"]=False
    x["rule_frozen_before_game"]=False
    x["split"]=x.season.map(SPLITS)
    # PT values are kept as unverified diagnostics; must not enter certified
    # CORE-support counts and are never represented as original dated picks.
    x["pt_forecast_numeric"]=forecast
    x["pt_context_edge_numeric"]=edge
    keep=["season","game_id","brain","system_id","direction","evaluation_status","evidence_type","market_vintage_certified","source_vintage_certified","rule_frozen_before_game","split","pt_forecast_numeric","pt_context_edge_numeric"]
    return x[keep].copy(),{"matched_games":len(x),"unmatched_pt_rows":int(len(parsed)-len(x)),"date_and_team_exact":True,"prediction_direction_is_diagnostic_not_certified":True}


def compare_to_core(games: pd.DataFrame, signals: pd.DataFrame, core: pd.DataFrame | None) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    """Same-game baseline vs independent brain hypothesis combinations.

    Counts and descriptive ATS grade are *not* certified historic profitability:
    the V5.5 recipe, brain rules and bookmaker quote vintages are unverified.
    """
    if not isinstance(core,pd.DataFrame) or core.empty:
        return pd.DataFrame(),pd.DataFrame(),{"matched_core_games":0,"reason":"RETROSPECTIVE_V55_REPLAY_UNAVAILABLE"}
    required={"season","game_id","model_home_margin"}
    if not required.issubset(core):return pd.DataFrame(),pd.DataFrame(),{"matched_core_games":0,"reason":"INVALID_CORE_SCHEMA"}
    c=core[["season","game_id","model_home_margin"]].copy()
    c["game_id"] = _str(c.game_id);c["season"] = _num(c.season).astype("Int64")
    if c.duplicated(["season","game_id"]).any():raise ValueError("CORE_DUPLICATE_REPLAY_GAME")
    g=games[["season","game_id","actual_home_margin","open_home_spread"]].merge(c,on=["season","game_id"],how="inner",validate="one_to_one")
    g["edge_home"] = _num(g.model_home_margin)+_num(g.open_home_spread)
    g=g[g.edge_home.abs().ge(2.0) & g.actual_home_margin.notna()].copy()
    if g.empty:return g,pd.DataFrame(),{"matched_core_games":0,"reason":"NO_TWO_POINT_CORE_EDGES_WITH_OUTCOMES"}
    g["core_side"]=np.where(g.edge_home>0,"HOME","AWAY")
    g["core_ats_delta"]=np.where(g.core_side.eq("HOME"),1,-1)*(_num(g.actual_home_margin)+_num(g.open_home_spread))
    g["core_ats_sign"]=np.sign(g.core_ats_delta)
    # Only directional signals can support. Group by independent brain, not by
    # the number of heavily-correlated mechanisms within that same family.
    f=signals.loc[(signals.evaluation_status=="FIRE") & signals.direction.isin(("HOME","AWAY"))].copy()
    f=f[f.brain.isin(("STAT","SYSTEM_MINER","PT"))]
    joined=g.merge(f[["season","game_id","brain","direction"]],on=["season","game_id"],how="left")
    joined["support"] = joined.direction.eq(joined.core_side)
    joined["oppose"] = joined.direction.notna() & joined.direction.ne(joined.core_side)
    agg=joined.groupby(["season","game_id"],as_index=False).agg(
        supporters=("brain",lambda v: "|".join(sorted(set(v[joined.loc[v.index,"support"]].dropna())))),
        conflicts=("brain",lambda v: "|".join(sorted(set(v[joined.loc[v.index,"oppose"]].dropna())))))
    g=g.merge(agg,on=["season","game_id"],how="left",validate="one_to_one")
    g["support_family_count"]=g.supporters.map(lambda z:len(str(z).split("|")) if isinstance(z,str) and z else 0)
    g["conflict_family_count"]=g.conflicts.map(lambda z:len(str(z).split("|")) if isinstance(z,str) and z else 0)
    g["split"]=g.season.map(SPLITS)
    g["cohort"] = np.select([g.conflict_family_count.gt(0),g.support_family_count.ge(2),g.support_family_count.eq(1)],
        ["ANY_BRAIN_CONFLICT__VETO_HYPOTHESIS","TWO_PLUS_FAMILIES__STRONG_HYPOTHESIS","ONE_FAMILY_SUPPORT"],
        default="CORE_ONLY_NO_BRAIN_SUPPORT")
    g["retrospective_only"] = True
    g["bet_authority"] = 0
    g["original_price_vintage_certified"] = False
    g["core_recipe_frozen_before_game"] = False
    g["brain_rules_frozen_before_game"] = False
    summary=g.groupby(["split","cohort"],dropna=False).agg(
        candidate_games=("game_id","size"),wins=("core_ats_sign",lambda v:int(v.eq(1).sum())),
        losses=("core_ats_sign",lambda v:int(v.eq(-1).sum())),pushes=("core_ats_sign",lambda v:int(v.eq(0).sum())),
        avg_abs_model_edge=("edge_home",lambda v:float(v.abs().mean()))).reset_index()
    summary["nonpush_ats_win_rate"] = summary.wins.div(summary.wins+summary.losses).replace([np.inf,-np.inf],np.nan)
    summary["causal_incremental_edge_certified"] = False
    return g,summary,{"matched_core_games":len(g),"correlated_rules_counted_as_one_family":True,
                      "replay_not_independent_holdout":True,"ats_quote_vintage_certified":False}



def independent_brain_core_diagnostics(core_games: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Matched CORE outcomes conditioned on each independent brain's signal.

    No unverified archive absence becomes a known NO_FIRE. A single brain with
    contradictory systems is explicitly INTERNAL_CONFLICT rather than one vote.
    These descriptive subgroup rates are NOT estimates of incremental ROI.
    """
    cols=["brain","season","split","relation","games","wins","losses","pushes","nonpush_ats_win_rate","incremental_edge_certified"]
    if core_games.empty:
        return pd.DataFrame(columns=cols)
    assert not core_games.duplicated(["season","game_id"]).any()
    evidence={}
    if not signals.empty:
        for (year,gid,brain),grp in signals.groupby(["season","game_id","brain"]):
            fire=grp[(grp.evaluation_status=="FIRE") & grp.direction.isin(["HOME","AWAY"])]
            if not fire.empty:
                evidence[(year,str(gid),brain)] = set(fire.direction)
            elif grp.evaluation_status.eq("UNKNOWN").any():
                # Partial rule knowledge does not certify a brain-level nonfire.
                evidence[(year,str(gid),brain)] = None
            elif grp.evaluation_status.eq("NO_FIRE").any():
                evidence[(year,str(gid),brain)] = set()
            else:
                evidence[(year,str(gid),brain)] = None
    rows=[]
    for record in core_games.itertuples(index=False):
        for brain in ("STAT","SYSTEM_MINER","PATHI","PT","BIG_AL"):
            observed=evidence.get((int(record.season),str(record.game_id),brain))
            if brain=="PATHI": relation="CONTEXT_ONLY_NO_DIRECTIONAL_VOTE"
            elif observed is None: relation="UNKNOWN"
            elif len(observed)>1: relation="INTERNAL_CONFLICT"
            elif len(observed)==0: relation="KNOWN_NO_FIRE"
            else: relation="SUPPORT" if record.core_side in observed else "OPPOSE"
            rows.append({"brain":brain,"season":int(record.season),"split":record.split,
                         "relation":relation,"core_ats_sign":record.core_ats_sign})
    x=pd.DataFrame(rows)
    out=x.groupby(["brain","season","split","relation"],dropna=False).agg(
        games=("core_ats_sign","size"),wins=("core_ats_sign",lambda z:int(z.eq(1).sum())),
        losses=("core_ats_sign",lambda z:int(z.eq(-1).sum())),
        pushes=("core_ats_sign",lambda z:int(z.eq(0).sum()))).reset_index()
    out["nonpush_ats_win_rate"]=out.wins.div(out.wins+out.losses)
    out["incremental_edge_certified"]=False
    return out[cols]


def reconstruct(*, cfbd: pd.DataFrame, context: pd.DataFrame,
                pt: pd.DataFrame | None=None, core: pd.DataFrame | None=None) -> dict[str,Any]:
    universe, cfbd_home, meta_cfbd=_stage_cfbd(cfbd)
    ctx,meta_ctx=_stage_context(context)
    games,join_meta=_attach_context(universe,ctx)
    rule_evals,rule_coverage=replay_rules(games,cfbd_home)
    pt_evidence,pt_meta=attach_pt(games,pt if isinstance(pt,pd.DataFrame) else pd.DataFrame())
    all_signals = pd.concat([rule_evals,pt_evidence], ignore_index=True,sort=False) if not pt_evidence.empty else rule_evals.copy()
    core_matched,comparison,core_meta=compare_to_core(games,all_signals,core)
    by_brain_core = independent_brain_core_diagnostics(core_matched, all_signals)
    # Per-game evaluated/unknown coverage for source families without dated
    # independent original picks: UNKNOWN is never treated as a non-fire.
    family_eligibility=[]
    for brain in ("BIG_AL","PT"):
        e=universe[["season","game_id"]].copy()
        e["brain"]=brain
        e["evaluation_status"]="UNKNOWN"
        e["reason"]="NO_ORIGINAL_DATED_BIG_AL_PICKS" if brain=="BIG_AL" else "NO_MATCHED_HISTORICAL_PT_RECORD_OR_VINTAGE"
        if brain=="PT" and not pt_evidence.empty:
            present=pd.MultiIndex.from_frame(pt_evidence[["season","game_id"]])
            match=pd.MultiIndex.from_frame(e[["season","game_id"]]).isin(present)
            e.loc[match,"evaluation_status"]="RETROSPECTIVE_ONLY"
            e.loc[match,"reason"]="EXACT_PT_ARCHIVE_MATCH__ORIGINAL_PUBLICATION_UNVERIFIED"
        e["point_in_time_certified"]=False
        family_eligibility.append(e)
    family_eligibility=pd.concat(family_eligibility,ignore_index=True)
    f=rule_evals[(rule_evals.evaluation_status=="FIRE") &
                 rule_evals.direction.isin(("HOME","AWAY")) &
                 rule_evals.descriptive_ats_result.notna()].copy()
    if not f.empty:
        descriptive=f.groupby(["brain","system_id","season","split"],dropna=False).agg(
            fired_with_descriptive_price=("game_id","size"),
            wins=("descriptive_ats_result",lambda x:int(x.eq(1).sum())),
            losses=("descriptive_ats_result",lambda x:int(x.eq(-1).sum())),
            pushes=("descriptive_ats_result",lambda x:int(x.eq(0).sum()))).reset_index()
        descriptive["nonpush_win_rate"]=descriptive.wins.div(descriptive.wins+descriptive.losses)
    else:
        descriptive=pd.DataFrame(columns=["brain","system_id","season","split","fired_with_descriptive_price","wins","losses","pushes","nonpush_win_rate"])
    descriptive["original_quote_timestamp_certified"]=False
    descriptive["rule_frozen_before_game"]=False
    manifest=meta_cfbd.pop("manifest")
    rule_status=rule_evals.groupby(["brain","evaluation_status"]).size().unstack(fill_value=0).to_dict("index")
    audit={"status":"RETROSPECTIVE_DIAGNOSTICS_COMPLETE_NO_CERTIFIED_BET_AUTHORITY",
           "version":VERSION,"historical_window":"2013-2025", "bet_authority":0,
           "2026_selection_influence":0,"production_mutation":False,
           "source_coverage":{"cfbd":meta_cfbd,"context":meta_ctx,"join":join_meta,"pt":pt_meta,"core":core_meta},
           "rule_counts_by_brain":rule_status,"rule_definition_postdates_confirmation_period":True,
           "independent_2024_2025_holdout_certified":False,
           "original_pregame_quote_timestamp_verified":False,
           "strict_historical_signal_vintage_certified":False,
           "big_al_pick_history":"UNAVAILABLE__CONTEXT_IS_NOT_PICKS",
           "incumbent_miner_191_mechanisms_replayed":False,
           "frozen_2026_production_untouched":True}
    return {"audit":audit,"cfbd_universe_manifest":manifest,
            "game_join_coverage":games[["season","game_id","game_date_key","home_team_key","away_team_key","has_context","actual_home_margin","open_home_spread"]],
            "rule_evaluations_retrospective":rule_evals,"rule_coverage_by_year":rule_coverage,
            "rule_descriptive_ats_by_year":descriptive,
            "uncovered_brain_eligibility":family_eligibility,
            "pt_evidence_retrospective":pt_evidence,"core_matched_game_edges":core_matched,
            "core_incremental_exploratory":comparison,
            "core_independent_brain_diagnostics":by_brain_core}


def load_bigquery_source_frames(project="sharplogger",dataset="sharp_data") -> tuple[pd.DataFrame,pd.DataFrame]:
    """Bounded, read-only and fixed-column BigQuery queries. No POSTGAME_CFBD inputs."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*",project) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",dataset):
        raise ValueError("INVALID_BIGQUERY_PROJECT_OR_DATASET")
    from google.cloud import bigquery
    client=bigquery.Client(project=project)
    queries=[]
    for table,cols in (("ncaaf_cfbd_archive_research_v1", CFBD_COLUMNS),
                       ("ncaaf_historical_game_side_context", CONTEXT_COLUMNS)):
        full=f"{project}.{dataset}.{table}"
        actual={c.name for c in client.get_table(full).schema}
        missing=set(cols)-actual
        if missing:raise ValueError("BIGQUERY_SOURCE_COLUMN_MISSING:"+table+":"+",".join(sorted(missing)))
        fields=",".join("`"+x+"`" for x in cols)
        sql=f"SELECT {fields} FROM `{full}` WHERE Season BETWEEN @start_season AND @end_season"
        cfg=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("start_season","INT64",MIN_YEAR),
              bigquery.ScalarQueryParameter("end_season","INT64",MAX_YEAR)])
        queries.append(client.query(sql,job_config=cfg).to_dataframe(create_bqstorage_client=False))
    return queries[0], queries[1]


def run_heavy_retrospective(*,storage_client,bucket_name="sharp-models",core=None,
                             source_frames=None,log_func=print,project="sharplogger",dataset="sharp_data") -> dict[str,Any]:
    """Run only as a research sidecar; let the caller publish immutable artifacts."""
    outcome={"status":"HOLD","reason":"REPLAY_SOURCE_NOT_AVAILABLE", "bet_authority":0,
             "2026_selection_influence":0,"production_mutation":False,"version":VERSION}
    try:
        if source_frames is None:
            cfbd,context=load_bigquery_source_frames(project=project,dataset=dataset)
        else:
            cfbd,context=source_frames
        pt,pt_gcs=pt_archive_inputs(storage_client,bucket_name)
        result=reconstruct(cfbd=cfbd,context=context,pt=pt,core=core)
        result["audit"]["pt_gcs"] = pt_gcs
        log_func("[NCAAF-HIST-REPLAY] status=RETROSPECTIVE_DIAGNOSTICS_COMPLETE "
                 f"cfbd_games={len(result['game_join_coverage'])} "
                 f"context_matched={int(result['game_join_coverage'].has_context.sum())} "
                 f"rule_eval_rows={len(result['rule_evaluations_retrospective'])} "
                 f"core_matched={len(result['core_matched_game_edges'])} authority=0")
        return result
    except Exception as exc:
        outcome["reason"]="SOURCE_EXTRACTION_OR_REPLAY_FAILED"
        outcome["error_type"]=type(exc).__name__
        outcome["error"] = str(exc)[:600]
        log_func("[NCAAF-HIST-REPLAY] status=HOLD type="+type(exc).__name__+" detail="+str(exc)[:300])
        return {"audit":outcome}
