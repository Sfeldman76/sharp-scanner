"""NFL Research V2.0 play-by-play statistical foundation.

Purpose
-------
Add a genuinely new, market-blind information source to NFL research rather than
re-running the existing box-score model with another algorithm.  Historical
nflverse play-by-play is aggregated to team-game football-efficiency measures,
then converted to strictly prior-only rolling context before any prediction is
made.

Critical contracts
------------------
* Development is 2017-2025 only.  2026 is never downloaded or queried.
* nflverse PBP final scores are NOT authoritative outcomes.  Existing audited
  BigDataBall history supplies scores and historical market reference labels.
* Same-game PBP is never allowed into same-game predictors.
* Primary QB identity/performance observed in the current game's PBP is outcome
  metadata only.  Only lagged/prior QB performance may enter a predictor.
* The first challenger is a single fixed regularized Ridge model.  This is a new
  information test, not another model tournament.
* Incumbent CORE remains protected and is compared, never overwritten.
* Production authority is always zero.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nfl_feature_audit_v1 import VIEW
from nfl_challenger_v1 import build_readonly_query, physical_games
from nfl_intelligence_v1 import _generate_oof_core
from nfl_research_v2_contract import assert_contract, contract_hash

SOURCE_TAG = "nfl-pbp-foundation-v1-research-v2.0-20261001"
DATA_SOURCE = "nflverse/nflverse-data parsed play-by-play"
PBP_URL_TEMPLATE = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv"
SEASONS = tuple(range(2017, 2026))
VALIDATION_SEASONS = (2021, 2022, 2023, 2024, 2025)
SEALED_SEASON = 2026
RIDGE_ALPHA = 25.0
PRODUCTION_AUTHORITY = 0
STATUS = "NFL_RESEARCH_V2_PBP_FOUNDATION_FROZEN_FOR_PROSPECTIVE_SHADOW"

# Only columns used by the aggregation are loaded.  The downloader first reads
# the header and intersects with this list so schema evolution fails explicitly
# only for truly essential identity columns.
PBP_WANTED_COLUMNS = (
    "season", "season_type", "week", "game_id", "game_date", "home_team", "away_team",
    "posteam", "defteam", "play_id", "drive", "qtr", "down", "play_type",
    "epa", "success", "pass_attempt", "rush_attempt", "qb_dropback", "qb_scramble",
    "sack", "interception", "fumble_lost", "first_down", "third_down_converted",
    "fourth_down_converted", "yardline_100", "touchdown", "yards_gained", "air_yards",
    "yards_after_catch", "cpoe", "qb_epa", "passer_player_id", "passer_player_name", "wp",
    "game_seconds_remaining", "sp", "qb_kneel", "qb_spike",
)
PBP_REQUIRED_COLUMNS = ("season", "season_type", "game_id", "game_date", "home_team", "away_team", "posteam", "defteam", "epa")

# Franchise-level canonical codes. Historical relocations intentionally map to
# the same modern franchise code so PBP and source naming can match robustly.
_TEAM_ALIASES = {
    "ari":"ARI","arizona":"ARI","arizonacardinals":"ARI","cardinals":"ARI",
    "atl":"ATL","atlanta":"ATL","atlantafalcons":"ATL","falcons":"ATL",
    "bal":"BAL","baltimore":"BAL","baltimoreravens":"BAL","ravens":"BAL",
    "buf":"BUF","buffalo":"BUF","buffalobills":"BUF","bills":"BUF",
    "car":"CAR","carolina":"CAR","carolinapanthers":"CAR","panthers":"CAR",
    "chi":"CHI","chicago":"CHI","chicagobears":"CHI","bears":"CHI",
    "cin":"CIN","cincinnati":"CIN","cincinnatibengals":"CIN","bengals":"CIN",
    "cle":"CLE","cleveland":"CLE","clevelandbrowns":"CLE","browns":"CLE",
    "dal":"DAL","dallas":"DAL","dallascowboys":"DAL","cowboys":"DAL",
    "den":"DEN","denver":"DEN","denverbroncos":"DEN","broncos":"DEN",
    "det":"DET","detroit":"DET","detroitlions":"DET","lions":"DET",
    "gb":"GB","gnb":"GB","greenbay":"GB","greenbaypackers":"GB","packers":"GB",
    "hou":"HOU","houston":"HOU","houstontexans":"HOU","texans":"HOU",
    "ind":"IND","indianapolis":"IND","indianapoliscolts":"IND","colts":"IND",
    "jax":"JAX","jac":"JAX","jacksonville":"JAX","jacksonvillejaguars":"JAX","jaguars":"JAX",
    "kc":"KC","kan":"KC","kansascity":"KC","kansascitychiefs":"KC","chiefs":"KC",
    "lv":"LV","oak":"LV","lasvegas":"LV","lasvegasraiders":"LV","oaklandraiders":"LV","raiders":"LV",
    "lac":"LAC","sd":"LAC","sandiegochargers":"LAC","losangeleschargers":"LAC","chargers":"LAC",
    "la":"LA","lar":"LA","stl":"LA","losangelesrams":"LA","stlouisrams":"LA","rams":"LA",
    "mia":"MIA","miami":"MIA","miamidolphins":"MIA","dolphins":"MIA",
    "min":"MIN","minnesota":"MIN","minnesotavikings":"MIN","vikings":"MIN",
    "ne":"NE","nwe":"NE","newengland":"NE","newenglandpatriots":"NE","patriots":"NE",
    "no":"NO","nor":"NO","neworleans":"NO","neworleanssaints":"NO","saints":"NO",
    "nyg":"NYG","newyorkgiants":"NYG","giants":"NYG",
    "nyj":"NYJ","newyorkjets":"NYJ","jets":"NYJ",
    "phi":"PHI","philadelphia":"PHI","philadelphiaeagles":"PHI","eagles":"PHI",
    "pit":"PIT","pittsburgh":"PIT","pittsburghsteelers":"PIT","steelers":"PIT",
    "sea":"SEA","seattle":"SEA","seattleseahawks":"SEA","seahawks":"SEA",
    "sf":"SF","sfo":"SF","sanfrancisco":"SF","sanfrancisco49ers":"SF","49ers":"SF",
    "tb":"TB","tam":"TB","tampabay":"TB","tampabaybuccaneers":"TB","buccaneers":"TB","bucs":"TB",
    "ten":"TEN","tennessee":"TEN","tennesseetitans":"TEN","titans":"TEN",
    "was":"WAS","wsh":"WAS","washington":"WAS","washingtoncommanders":"WAS","washingtonfootballteam":"WAS","washingtonredskins":"WAS","commanders":"WAS","redskins":"WAS",
}


_NICKNAME_CODE = {
    "cardinals":"ARI","falcons":"ATL","ravens":"BAL","bills":"BUF","panthers":"CAR","bears":"CHI",
    "bengals":"CIN","browns":"CLE","cowboys":"DAL","broncos":"DEN","lions":"DET","packers":"GB",
    "texans":"HOU","colts":"IND","jaguars":"JAX","chiefs":"KC","raiders":"LV","chargers":"LAC",
    "rams":"LA","dolphins":"MIA","vikings":"MIN","patriots":"NE","saints":"NO","giants":"NYG",
    "jets":"NYJ","eagles":"PHI","steelers":"PIT","seahawks":"SEA","49ers":"SF","buccaneers":"TB",
    "titans":"TEN","commanders":"WAS","redskins":"WAS"
}

def canonical_team(value) -> str:
    s = re.sub(r"[^a-z0-9]", "", str(value or "").lower())
    if s in _TEAM_ALIASES:
        return _TEAM_ALIASES[s]
    for nickname, code in _NICKNAME_CODE.items():
        if s.endswith(nickname):
            return code
    return str(value or "").upper().strip()


def _safe_num(s: pd.Series, default=np.nan) -> pd.Series:
    if s is None:
        return pd.Series(default)
    return pd.to_numeric(s, errors="coerce")


def _bool_num(d: pd.DataFrame, col: str) -> pd.Series:
    if col not in d:
        return pd.Series(0.0, index=d.index)
    return pd.to_numeric(d[col], errors="coerce").fillna(0.0)


def _read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _download_one_season(season: int, cache_dir: Path, log_func=print) -> Path:
    if season == SEALED_SEASON or season not in SEASONS:
        raise RuntimeError(f"NFL_PBP_SEALED_OR_UNEXPECTED_SEASON_{season}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"play_by_play_{season}.csv"
    if path.exists() and path.stat().st_size > 1024:
        return path
    url = PBP_URL_TEMPLATE.format(season=season)
    tmp = path.with_suffix(".csv.part")
    log_func(f"[NFL-RESEARCH-V2-PBP-DOWNLOAD] season={season} source=nflverse url={url}")
    req = urllib.request.Request(url, headers={"User-Agent":"sharp-scanner-nfl-research-v2/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024*1024)
    if tmp.stat().st_size < 1024:
        raise RuntimeError(f"NFL_PBP_DOWNLOAD_TOO_SMALL season={season}")
    tmp.replace(path)
    return path


def load_nflverse_pbp(seasons: Iterable[int] = SEASONS, *, cache_dir: str | Path | None = None, log_func=print) -> pd.DataFrame:
    years = tuple(int(x) for x in seasons)
    if not years or max(years) > 2025 or 2026 in years:
        raise RuntimeError("NFL_PBP_2026_DATA_FORBIDDEN")
    cdir = Path(cache_dir or os.getenv("NFL_PBP_CACHE_DIR", "/tmp/nfl_research_v2_pbp"))
    pieces=[]
    for sy in years:
        p=_download_one_season(sy,cdir,log_func=log_func)
        header=_read_header(p)
        missing=set(PBP_REQUIRED_COLUMNS)-set(header)
        if missing:
            raise RuntimeError(f"NFL_PBP_REQUIRED_COLUMNS_MISSING season={sy} missing={sorted(missing)}")
        use=[c for c in PBP_WANTED_COLUMNS if c in header]
        x=pd.read_csv(p,usecols=use,low_memory=False)
        if "season" not in x or int(pd.to_numeric(x["season"],errors="coerce").dropna().max())>2025:
            raise RuntimeError(f"NFL_PBP_SEALED_SEASON_LEAK season_file={sy}")
        pieces.append(x)
        log_func(f"[NFL-RESEARCH-V2-PBP-LOADED] season={sy} rows={len(x)} columns={len(x.columns)}")
    out=pd.concat(pieces,ignore_index=True)
    return out


def _mean_mask(v: pd.Series, mask: pd.Series) -> float:
    q=pd.to_numeric(v,errors="coerce")[mask]
    return float(q.mean()) if q.notna().any() else math.nan


def aggregate_pbp_game_team(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PBP into one offensive team row per physical game.

    The output contains same-game outcomes/efficiency.  It is not itself a
    predictor table. `build_prior_context` performs the mandatory one-game shift.
    """
    d=pbp.copy()
    if d.empty: raise RuntimeError("NFL_PBP_EMPTY")
    d["season"]=pd.to_numeric(d["season"],errors="coerce")
    if d["season"].dropna().gt(2025).any(): raise RuntimeError("NFL_PBP_2026_DATA_LEAK")
    d=d.loc[d.get("season_type","").astype(str).isin(["REG","POST"])].copy()
    d=d.loc[d["posteam"].notna() & d["defteam"].notna()].copy()
    d["team_code"]=d["posteam"].map(canonical_team); d["opp_code"]=d["defteam"].map(canonical_team)
    d["game_date_norm"]=pd.to_datetime(d["game_date"],errors="coerce").dt.strftime("%Y-%m-%d")
    epa=pd.to_numeric(d["epa"],errors="coerce")
    drop=_bool_num(d,"qb_dropback").eq(1) | _bool_num(d,"pass_attempt").eq(1) | _bool_num(d,"sack").eq(1)
    rush=_bool_num(d,"rush_attempt").eq(1) & ~_bool_num(d,"qb_kneel").eq(1)
    scrim=(drop|rush) & epa.notna() & ~_bool_num(d,"qb_spike").eq(1)
    d["_scrim"]=scrim
    d["_drop"]=drop & epa.notna(); d["_rush"]=rush & epa.notna()
    down=pd.to_numeric(d.get("down"),errors="coerce")
    d["_early"]=scrim & down.isin([1,2])
    wp=pd.to_numeric(d.get("wp"),errors="coerce")
    qtr=pd.to_numeric(d.get("qtr"),errors="coerce")
    d["_neutral"]=scrim & qtr.le(3) & wp.between(.20,.80)
    rz=pd.to_numeric(d.get("yardline_100"),errors="coerce").le(20)
    d["_rz"]=scrim & rz
    yg=pd.to_numeric(d.get("yards_gained"),errors="coerce")
    d["_explosive"]=((_bool_num(d,"pass_attempt").eq(1)&yg.ge(20)) | (rush&yg.ge(10))) & scrim
    d["_turnover"]=( _bool_num(d,"interception").eq(1) | _bool_num(d,"fumble_lost").eq(1) ) & scrim
    d["_third"]=down.eq(3) & scrim; d["_fourth"]=down.eq(4) & scrim

    records=[]
    for (sy,gid,gd,team,opp),p in d.groupby(["season","game_id","game_date_norm","team_code","opp_code"],dropna=False,sort=False):
        s=p["_scrim"].fillna(False); dr=p["_drop"].fillna(False); ru=p["_rush"].fillna(False)
        early=p["_early"].fillna(False); neu=p["_neutral"].fillna(False); rzmask=p["_rz"].fillna(False)
        third=p["_third"].fillna(False); fourth=p["_fourth"].fillna(False)
        rec={
            "Season":int(sy),"pbp_game_id":str(gid),"game_date_norm":str(gd),"team_code":str(team),"opp_code":str(opp),
            "pbp_plays":int(s.sum()),"pbp_drives":int(pd.to_numeric(p.get("drive"),errors="coerce").dropna().nunique()) if "drive" in p else 0,
            "off_epa_play":_mean_mask(p["epa"],s),
            "off_success_rate":_mean_mask(p.get("success",pd.Series(np.nan,index=p.index)),s),
            "pass_epa_dropback":_mean_mask(p["epa"],dr),
            "rush_epa":_mean_mask(p["epa"],ru),
            "early_down_epa":_mean_mask(p["epa"],early),
            "early_down_success":_mean_mask(p.get("success",pd.Series(np.nan,index=p.index)),early),
            "explosive_rate":float(p.loc[s,"_explosive"].mean()) if s.any() else math.nan,
            "turnover_play_rate":float(p.loc[s,"_turnover"].mean()) if s.any() else math.nan,
            "yards_per_play":_mean_mask(p.get("yards_gained",pd.Series(np.nan,index=p.index)),s),
            "first_down_rate":_mean_mask(p.get("first_down",pd.Series(np.nan,index=p.index)),s),
            "sack_rate":float(_bool_num(p,"sack")[dr].mean()) if dr.any() else math.nan,
            "third_down_conv_rate":_mean_mask(p.get("third_down_converted",pd.Series(np.nan,index=p.index)),third),
            "fourth_down_conv_rate":_mean_mask(p.get("fourth_down_converted",pd.Series(np.nan,index=p.index)),fourth),
            "red_zone_epa":_mean_mask(p["epa"],rzmask),
            "red_zone_td_rate":_mean_mask(p.get("touchdown",pd.Series(np.nan,index=p.index)),rzmask),
            "neutral_pass_rate":float(dr[neu].mean()) if neu.any() else math.nan,
            "cpoe":_mean_mask(p.get("cpoe",pd.Series(np.nan,index=p.index)),dr),
            "air_yards_per_dropback":_mean_mask(p.get("air_yards",pd.Series(np.nan,index=p.index)),dr),
            "yac_per_dropback":_mean_mask(p.get("yards_after_catch",pd.Series(np.nan,index=p.index)),dr),
        }
        rec["plays_per_drive"]=(rec["pbp_plays"]/rec["pbp_drives"]) if rec["pbp_drives"] else math.nan
        # Same-game primary QB is metadata only.  Its values are shifted before
        # the model sees them; current-game QB identity never becomes a feature.
        if "passer_player_id" in p.columns:
            q=p.loc[dr & p["passer_player_id"].notna()].copy()
            if not q.empty:
                counts=q.groupby(["passer_player_id","passer_player_name"],dropna=False).size().sort_values(ascending=False)
                qid,qname=counts.index[0]
                qm=q["passer_player_id"].astype(str).eq(str(qid))
                rec["primary_qb_id_outcome_metadata"]=str(qid)
                rec["primary_qb_name_outcome_metadata"]=str(qname)
                rec["primary_qb_epa_game"]=_mean_mask(q.get("qb_epa",q["epa"]),qm)
                rec["primary_qb_cpoe_game"]=_mean_mask(q.get("cpoe",pd.Series(np.nan,index=q.index)),qm)
            else:
                rec.update({"primary_qb_id_outcome_metadata":None,"primary_qb_name_outcome_metadata":None,"primary_qb_epa_game":math.nan,"primary_qb_cpoe_game":math.nan})
        records.append(rec)
    out=pd.DataFrame.from_records(records)
    if out.empty: raise RuntimeError("NFL_PBP_NO_GAME_TEAM_ROWS")
    key=["Season","pbp_game_id","team_code"]
    if out.duplicated(key).any(): raise RuntimeError("NFL_PBP_DUPLICATE_GAME_TEAM")
    return out.sort_values(["Season","game_date_norm","pbp_game_id","team_code"]).reset_index(drop=True)


BASE_METRICS=(
    "off_epa_play","off_success_rate","pass_epa_dropback","rush_epa","early_down_epa","early_down_success",
    "explosive_rate","turnover_play_rate","yards_per_play","first_down_rate","sack_rate","third_down_conv_rate",
    "red_zone_epa","red_zone_td_rate","neutral_pass_rate","cpoe","air_yards_per_dropback","yac_per_dropback",
    "plays_per_drive",
)


def add_defensive_allowed(game_team: pd.DataFrame) -> pd.DataFrame:
    d=game_team.copy()
    opp=d[["Season","pbp_game_id","team_code",*BASE_METRICS]].copy()
    opp=opp.rename(columns={"team_code":"opp_code_join",**{c:"def_allow_"+c for c in BASE_METRICS}})
    x=d.merge(opp,left_on=["Season","pbp_game_id","opp_code"],right_on=["Season","pbp_game_id","opp_code_join"],how="left",validate="one_to_one")
    return x.drop(columns=["opp_code_join"])


CONTEXT_RAW_METRICS=tuple(BASE_METRICS)+tuple("def_allow_"+c for c in BASE_METRICS)+("primary_qb_epa_game","primary_qb_cpoe_game")


def build_prior_context(game_team: pd.DataFrame) -> pd.DataFrame:
    """Create strictly prior-only rolling features.

    Every feature is computed from shifted prior games.  This function is the
    central leakage boundary for the PBP lane.
    """
    d=add_defensive_allowed(game_team)
    d=d.sort_values(["Season","team_code","game_date_norm","pbp_game_id"],kind="mergesort").copy()
    grp=d.groupby(["Season","team_code"],sort=False,dropna=False)
    d["team_game_number_prior"]=grp.cumcount().astype(float)
    ctx_cols={}
    for c in CONTEXT_RAW_METRICS:
        if c not in d: continue
        s=pd.to_numeric(d[c],errors="coerce")
        shifted=s.groupby([d["Season"],d["team_code"]],sort=False).shift(1)
        ctx_cols[c+"__prior1"]=shifted
        ctx_cols[c+"__l3"]=shifted.groupby([d["Season"],d["team_code"]],sort=False).rolling(3,min_periods=1).mean().reset_index(level=[0,1],drop=True)
        ctx_cols[c+"__l5"]=shifted.groupby([d["Season"],d["team_code"]],sort=False).rolling(5,min_periods=1).mean().reset_index(level=[0,1],drop=True)
        # Expanding mean of shifted values = season-to-date prior only.
        ctx_cols[c+"__season_prior"]=shifted.groupby([d["Season"],d["team_code"]],sort=False).expanding(min_periods=1).mean().reset_index(level=[0,1],drop=True)
    if ctx_cols:
        d=pd.concat([d,pd.DataFrame(ctx_cols,index=d.index)],axis=1)
    # Previous primary QB identity is retained for future starter-parity research,
    # but is deliberately not numeric/model input in V2.0.
    if "primary_qb_id_outcome_metadata" in d:
        d["prior_primary_qb_id"]=grp["primary_qb_id_outcome_metadata"].shift(1)
    # Opponent-adjusted EPA game residual uses only opponent state known before
    # this game; the resulting game residual itself is then only available to
    # later games when shifted below.
    lookup=d[["Season","pbp_game_id","team_code","def_allow_off_epa_play__season_prior"]].rename(columns={"team_code":"opp_lookup","def_allow_off_epa_play__season_prior":"opp_def_epa_prior"})
    d=d.merge(lookup,left_on=["Season","pbp_game_id","opp_code"],right_on=["Season","pbp_game_id","opp_lookup"],how="left",validate="one_to_one")
    d["opponent_adjusted_off_epa_game"]=pd.to_numeric(d["off_epa_play"],errors="coerce")-pd.to_numeric(d["opp_def_epa_prior"],errors="coerce")
    d=d.sort_values(["Season","team_code","game_date_norm","pbp_game_id"],kind="mergesort")
    g2=d.groupby(["Season","team_code"],sort=False,dropna=False)
    shifted=pd.to_numeric(d["opponent_adjusted_off_epa_game"],errors="coerce").groupby([d["Season"],d["team_code"]],sort=False).shift(1)
    d["opponent_adjusted_off_epa__l5"]=shifted.groupby([d["Season"],d["team_code"]],sort=False).rolling(5,min_periods=1).mean().reset_index(level=[0,1],drop=True)
    d["opponent_adjusted_off_epa__season_prior"]=shifted.groupby([d["Season"],d["team_code"]],sort=False).expanding(min_periods=1).mean().reset_index(level=[0,1],drop=True)
    return d.sort_values(["Season","game_date_norm","pbp_game_id","team_code"]).reset_index(drop=True)


def _authority_game_keys(games: pd.DataFrame) -> pd.DataFrame:
    a=games.copy()
    a["game_date_norm"]=pd.to_datetime(a["Game_Date"],errors="coerce").dt.strftime("%Y-%m-%d")
    a["team_code"]=a["Team_Norm"].map(canonical_team); a["opp_code"]=a["Opponent_Norm"].map(canonical_team)
    a["pair_key"]=a.apply(lambda r:"|".join(sorted([str(r.team_code),str(r.opp_code)])),axis=1)
    return a


def match_pbp_to_authoritative(games: pd.DataFrame, pbp_context: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    """Match one oriented authoritative physical game to both PBP team contexts."""
    a=_authority_game_keys(games)
    p=pbp_context.copy()
    p["pair_key"]=p.apply(lambda r:"|".join(sorted([str(r.team_code),str(r.opp_code)])),axis=1)
    # First collapse PBP to one game id per date/franchise pair.
    pkey=p[["Season","game_date_norm","pair_key","pbp_game_id"]].drop_duplicates()
    dup=pkey.duplicated(["Season","game_date_norm","pair_key"],keep=False)
    if dup.any():
        # Do not guess ambiguous pair/date matches.
        bad=pkey.loc[dup,["Season","game_date_norm","pair_key","pbp_game_id"]].head(10).to_dict("records")
        raise RuntimeError("NFL_PBP_AMBIGUOUS_PAIR_DATE "+json.dumps(bad))
    a=a.merge(pkey,on=["Season","game_date_norm","pair_key"],how="left",validate="many_to_one")
    tctx=p.add_prefix("team_")
    octx=p.add_prefix("opp_")
    a=a.merge(tctx,left_on=["Season","pbp_game_id","team_code"],right_on=["team_Season","team_pbp_game_id","team_team_code"],how="left",validate="one_to_one")
    a=a.merge(octx,left_on=["Season","pbp_game_id","opp_code"],right_on=["opp_Season","opp_pbp_game_id","opp_team_code"],how="left",validate="one_to_one")
    matched=a["pbp_game_id"].notna()
    audit={
        "authoritative_games":int(len(a)),"matched_games":int(matched.sum()),"unmatched_games":int((~matched).sum()),
        "match_rate":round(float(matched.mean()),6) if len(a) else 0.0,
        "unmatched_sample":a.loc[~matched,["Season","game_date_norm","Team_Norm","Opponent_Norm","team_code","opp_code"]].head(15).to_dict("records"),
    }
    if audit["match_rate"] < .95:
        raise RuntimeError("NFL_PBP_AUTHORITY_MATCH_RATE_TOO_LOW "+json.dumps(audit))
    return a.loc[matched].copy(),audit


PBP_MODEL_L5_BASES=(
    "off_epa_play","def_allow_off_epa_play","off_success_rate","def_allow_off_success_rate",
    "pass_epa_dropback","def_allow_pass_epa_dropback","rush_epa","def_allow_rush_epa",
    "early_down_epa","def_allow_early_down_epa","explosive_rate","def_allow_explosive_rate",
    "turnover_play_rate","def_allow_turnover_play_rate","sack_rate","def_allow_sack_rate",
    "red_zone_td_rate","def_allow_red_zone_td_rate","neutral_pass_rate","cpoe",
    "primary_qb_epa_game","primary_qb_cpoe_game","opponent_adjusted_off_epa",
)
PBP_MODEL_SEASON_BASES=(
    "off_epa_play","def_allow_off_epa_play","off_success_rate","def_allow_off_success_rate",
    "pass_epa_dropback","def_allow_pass_epa_dropback","rush_epa","def_allow_rush_epa",
    "primary_qb_epa_game","opponent_adjusted_off_epa",
)


def build_model_matrix(matched: pd.DataFrame) -> tuple[pd.DataFrame,list[str]]:
    """Create a curated symmetric PBP feature matrix.

    This is intentionally not a 300-column feature dump.  The first PBP challenger
    uses last-five context for a compact set of football-efficiency concepts plus
    season-to-date priors for the most structural EPA/success/QB measures.
    """
    x=matched.copy()
    wanted=[]
    for base in PBP_MODEL_L5_BASES:
        wanted.append(base+"__l5")
    for base in PBP_MODEL_SEASON_BASES:
        wanted.append(base+"__season_prior")
    newcols={}; features=[]
    for raw in wanted:
        tc="team_"+raw; oc="opp_"+raw
        if tc not in x or oc not in x:
            continue
        base=re.sub(r"[^A-Za-z0-9_]+","_",raw)
        diff="pbp_diff__"+base; summ="pbp_sum__"+base
        newcols[diff]=pd.to_numeric(x[tc],errors="coerce")-pd.to_numeric(x[oc],errors="coerce")
        newcols[summ]=pd.to_numeric(x[tc],errors="coerce")+pd.to_numeric(x[oc],errors="coerce")
        features.extend([diff,summ])
    if "team_team_game_number_prior" in x and "opp_team_game_number_prior" in x:
        newcols["pbp_min_prior_games"]=np.minimum(pd.to_numeric(x["team_team_game_number_prior"],errors="coerce"),pd.to_numeric(x["opp_team_game_number_prior"],errors="coerce"))
        features.append("pbp_min_prior_games")
    if newcols:
        x=pd.concat([x,pd.DataFrame(newcols,index=x.index)],axis=1)
    forbidden=[f for f in features if "outcome_metadata" in f or f.endswith("_game") or "actual" in f.lower() or "score" in f.lower()]
    if forbidden: raise RuntimeError("NFL_PBP_FORBIDDEN_MODEL_FEATURES "+str(forbidden))
    features=sorted(set(features))
    # Globally empty columns carry no information and are excluded before folds;
    # this is based only on predictor availability, never outcomes.
    features=[f for f in features if pd.to_numeric(x[f],errors="coerce").notna().any()]
    if len(features)<30: raise RuntimeError(f"NFL_PBP_TOO_FEW_MODEL_FEATURES count={len(features)}")
    return x,features


def _pipeline() -> Pipeline:
    return Pipeline([
        ("impute",SimpleImputer(strategy="median",add_indicator=False)),
        ("scale",StandardScaler()),
        ("ridge",Ridge(alpha=RIDGE_ALPHA)),
    ])


def _metrics(y,p) -> dict:
    y=np.asarray(y,float); p=np.asarray(p,float); m=np.isfinite(y)&np.isfinite(p)
    if not m.any(): return {"n":0,"mae":None,"rmse":None,"corr":None}
    yy=y[m]; pp=p[m]; err=yy-pp
    corr=float(np.corrcoef(yy,pp)[0,1]) if len(yy)>2 and np.std(yy)>0 and np.std(pp)>0 else math.nan
    return {"n":int(len(yy)),"mae":round(float(np.mean(np.abs(err))),6),"rmse":round(float(np.sqrt(np.mean(err**2))),6),"corr":round(corr,6) if math.isfinite(corr) else None}


def run_oof_pbp_challenger(model_df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame,dict,dict]:
    d=model_df.copy()
    d["actual_margin"]=pd.to_numeric(d["Team_Score"],errors="coerce")-pd.to_numeric(d["Opponent_Score"],errors="coerce")
    d["actual_total"]=pd.to_numeric(d["Team_Score"],errors="coerce")+pd.to_numeric(d["Opponent_Score"],errors="coerce")
    pieces=[]; fold_report={}
    for vy in VALIDATION_SEASONS:
        tr=d.loc[pd.to_numeric(d.Season,errors="coerce").lt(vy)].copy(); va=d.loc[pd.to_numeric(d.Season,errors="coerce").eq(vy)].copy()
        if len(tr)<900 or len(va)<240: raise RuntimeError(f"NFL_PBP_INSUFFICIENT_FOLD_{vy} train={len(tr)} val={len(va)}")
        part=va[["physical_game_id","Season","Game_Date","Team_Norm","Opponent_Norm","Spread_Value","Current_Total","actual_margin","actual_total"]].copy()
        for target,label in (("actual_margin","pbp_margin_pred"),("actual_total","pbp_total_pred")):
            pipe=_pipeline(); mask=tr[target].notna()
            pipe.fit(tr.loc[mask,features],tr.loc[mask,target].astype(float))
            part[label]=pipe.predict(va[features])
        pieces.append(part)
        fold_report[str(vy)]={"train_games":int(len(tr)),"validation_games":int(len(va)),"margin":_metrics(part.actual_margin,part.pbp_margin_pred),"total":_metrics(part.actual_total,part.pbp_total_pred)}
    oof=pd.concat(pieces,ignore_index=True)
    if oof.physical_game_id.duplicated().any(): raise RuntimeError("NFL_PBP_DUPLICATE_OOF_GAME")
    final={}
    for target,label in (("actual_margin","margin_model"),("actual_total","total_model")):
        pipe=_pipeline(); mask=d[target].notna(); pipe.fit(d.loc[mask,features],d.loc[mask,target].astype(float)); final[label]=pipe
    return oof,fold_report,final


def _comparison(oof: pd.DataFrame, incumbent_oof: pd.DataFrame) -> dict:
    c=incumbent_oof[["physical_game_id","core_margin_pred","core_total_pred","actual_margin","actual_total","Spread_Value","Current_Total"]].copy()
    x=oof.merge(c,on="physical_game_id",how="inner",suffixes=("","_core"),validate="one_to_one")
    x["market_margin_reference"]=-pd.to_numeric(x.Spread_Value,errors="coerce")
    x["market_total_reference"]=pd.to_numeric(x.Current_Total,errors="coerce")
    x["blend_margin_pred"]=(pd.to_numeric(x.pbp_margin_pred,errors="coerce")+pd.to_numeric(x.core_margin_pred,errors="coerce"))/2
    x["blend_total_pred"]=(pd.to_numeric(x.pbp_total_pred,errors="coerce")+pd.to_numeric(x.core_total_pred,errors="coerce"))/2
    out={"spreads":{},"totals":{},"by_season":{}}
    for name,col in (("PBP_CORE2","pbp_margin_pred"),("INCUMBENT_CORE","core_margin_pred"),("MARKET_REFERENCE","market_margin_reference"),("INCUMBENT50_PBP50_DIAGNOSTIC","blend_margin_pred")):
        out["spreads"][name]=_metrics(x.actual_margin,x[col])
    for name,col in (("PBP_CORE2","pbp_total_pred"),("INCUMBENT_CORE","core_total_pred"),("MARKET_REFERENCE","market_total_reference"),("INCUMBENT50_PBP50_DIAGNOSTIC","blend_total_pred")):
        out["totals"][name]=_metrics(x.actual_total,x[col])
    for sy,p in x.groupby("Season"):
        out["by_season"][str(int(sy))]={
            "pbp_margin":_metrics(p.actual_margin,p.pbp_margin_pred),"incumbent_margin":_metrics(p.actual_margin,p.core_margin_pred),"market_margin":_metrics(p.actual_margin,p.market_margin_reference),
            "pbp_total":_metrics(p.actual_total,p.pbp_total_pred),"incumbent_total":_metrics(p.actual_total,p.core_total_pred),"market_total":_metrics(p.actual_total,p.market_total_reference),
        }
    return out


def _upload_immutable(storage_client,bucket_name,name,data:bytes,content_type:str):
    from google.api_core.exceptions import PreconditionFailed
    blob=storage_client.bucket(bucket_name).blob(name)
    try:
        blob.upload_from_string(data,content_type=content_type,if_generation_match=0)
        return {"uri":f"gs://{bucket_name}/{name}","created":True}
    except PreconditionFailed:
        return {"uri":f"gs://{bucket_name}/{name}","created":False,"reason":"ALREADY_EXISTS_IMMUTABLE"}


def _gzip_csv(df:pd.DataFrame)->bytes:
    raw=df.to_csv(index=False).encode()
    return gzip.compress(raw,compresslevel=6)


def run_nfl_pbp_foundation_v1(*,bq_client,storage_client,bucket_name="sharp-models",audit_report=None,log_func=print,cache_dir=None):
    if not isinstance(audit_report,dict) or audit_report.get("status")!="READY_FOR_OFFLINE_CHALLENGER_SANDBOX":
        raise RuntimeError("NFL_RESEARCH_V2_PBP_AUDIT_NOT_GREEN")
    contract=assert_contract()
    view_cols={f.name for f in bq_client.get_table(VIEW).schema}
    side=bq_client.query(build_readonly_query(view_cols)).to_dataframe(create_bqstorage_client=False)
    if int(pd.to_numeric(side.Season,errors="coerce").max())>2025: raise RuntimeError("NFL_RESEARCH_V2_PBP_2026_QUERY_LEAK")
    games=physical_games(side)
    log_func(f"[NFL-RESEARCH-V2-PBP-PREFLIGHT] status=PASS source_tag={SOURCE_TAG} data_through=2025 year_2026_queried=FALSE production_authority=0 contract_sha256={contract_hash()}")

    pbp=load_nflverse_pbp(SEASONS,cache_dir=cache_dir,log_func=log_func)
    game_team=aggregate_pbp_game_team(pbp)
    context=build_prior_context(game_team)
    matched,audit=match_pbp_to_authoritative(games,context)
    model_df,features=build_model_matrix(matched)
    oof,folds,models=run_oof_pbp_challenger(model_df,features)
    incumbent=_generate_oof_core(side)
    comparison=_comparison(oof,incumbent)

    registry={
        "source_tag":SOURCE_TAG,"contract_sha256":contract_hash(),"data_source":DATA_SOURCE,
        "source_url_template":PBP_URL_TEMPLATE,"seasons":list(SEASONS),"year_2026_queried":False,
        "model":"PBP_CORE2_FIXED_RIDGE","ridge_alpha":RIDGE_ALPHA,"feature_count":len(features),"features":features,"feature_policy":"CURATED_PBP_LAST5_PLUS_STRUCTURAL_SEASON_PRIOR_SYMMETRIC_DIFF_SUM",
        "current_game_qb_predictor":False,"same_game_pbp_predictor":False,"authoritative_outcome_source":"existing_nfl_history",
        "incumbent_core_status":"PROTECTED_BASELINE","production_authority":0,"automatic_promotion":False,
        "prospective_clock":"NFL_RESEARCH_V2_PBP_POST_DEPLOYMENT_CLOCK",
    }
    reg_sha=hashlib.sha256(json.dumps(registry,sort_keys=True,separators=(",",":")).encode()).hexdigest(); registry["registry_sha256"]=reg_sha
    bundle={"metadata":registry,"margin_model":models["margin_model"],"total_model":models["total_model"],"features":features}
    report={
        "status":STATUS,"source_tag":SOURCE_TAG,"registry":registry,"match_audit":audit,"folds":folds,"comparison":comparison,
        "pbp_rows":int(len(pbp)),"pbp_game_team_rows":int(len(game_team)),"matched_authoritative_games":int(len(model_df)),
        "production_authority":0,"year_2026_queried":False,"ncaaf":"UNCHANGED","legacy_nfl":"UNCHANGED",
        "notes":["PBP final scores are not authoritative.","Same-game PBP and current-game observed QB are excluded from predictors.","Incumbent CORE is not replaced regardless of this research result."],
    }
    prefix=f"nfl-research/v2_0/pbp/{reg_sha[:16]}"
    bio=io.BytesIO(); joblib.dump(bundle,bio,compress=3)
    arts={
        "model_bundle":_upload_immutable(storage_client,bucket_name,f"{prefix}/pbp_core2_bundle.joblib",bio.getvalue(),"application/octet-stream"),
        "registry":_upload_immutable(storage_client,bucket_name,f"{prefix}/registry.json",json.dumps(registry,sort_keys=True,indent=2).encode(),"application/json"),
        "report":_upload_immutable(storage_client,bucket_name,f"{prefix}/research_report.json",json.dumps(report,sort_keys=True,default=str).encode(),"application/json"),
        "game_team_features":_upload_immutable(storage_client,bucket_name,f"{prefix}/pbp_game_team.csv.gz",_gzip_csv(game_team),"application/gzip"),
        "prior_context":_upload_immutable(storage_client,bucket_name,f"{prefix}/pbp_prior_context.csv.gz",_gzip_csv(context),"application/gzip"),
        "oof_predictions":_upload_immutable(storage_client,bucket_name,f"{prefix}/pbp_oof_predictions.csv.gz",_gzip_csv(oof),"application/gzip"),
    }
    result={**report,"artifacts":arts}
    log_func("[NFL-RESEARCH-V2-PBP-MATCH] "+json.dumps(audit,sort_keys=True,default=str))
    log_func("[NFL-RESEARCH-V2-PBP-COMPARISON] "+json.dumps(comparison,sort_keys=True,default=str))
    log_func("[NFL-RESEARCH-V2-PBP-CONTRACT] "+json.dumps({"status":STATUS,"registry_sha256":reg_sha,"feature_count":len(features),"matched_games":len(model_df),"year_2026_queried":False,"production_authority":0,"ncaaf":"UNCHANGED","legacy_nfl":"UNCHANGED","artifacts":arts},sort_keys=True,default=str))
    return result


# ------------------------------- synthetic tests -------------------------------
def _synthetic_pbp_for_tests() -> pd.DataFrame:
    rows=[]
    for sy in range(2017,2026):
        for wk in range(1,5):
            gid=f"{sy}_{wk:02d}_AAA_BBB"; gd=f"{sy}-09-{wk+1:02d}"
            for team,opp in (("NE","NYJ"),("NYJ","NE")):
                for j in range(12):
                    rows.append({"season":sy,"season_type":"REG","week":wk,"game_id":gid,"game_date":gd,"home_team":"NE","away_team":"NYJ","posteam":team,"defteam":opp,"play_id":j+1,"drive":j//4+1,"qtr":1+j//4,"down":1+j%4,"play_type":"pass" if j%2==0 else "run","epa":(.1 if team=="NE" else -.05)+.01*j,"success":1 if j%3 else 0,"pass_attempt":1 if j%2==0 else 0,"rush_attempt":1 if j%2 else 0,"qb_dropback":1 if j%2==0 else 0,"sack":0,"interception":0,"fumble_lost":0,"first_down":1 if j%4==0 else 0,"third_down_converted":1 if j%4==2 else 0,"fourth_down_converted":0,"yardline_100":50-j,"touchdown":0,"yards_gained":5+j%3,"air_yards":7 if j%2==0 else np.nan,"yards_after_catch":3 if j%2==0 else np.nan,"cpoe":2 if j%2==0 else np.nan,"qb_epa":.1,"passer_player_id":team+"QB","passer_player_name":team+" QB","wp":.5,"qb_kneel":0,"qb_spike":0})
    return pd.DataFrame(rows)


def self_test() -> dict:
    p=_synthetic_pbp_for_tests(); g=aggregate_pbp_game_team(p); c=build_prior_context(g)
    # Mutating one current game must not change that game's own prior features.
    key=(2019,"2019_03_AAA_BBB","NE")
    before=c.loc[(c.Season==key[0])&(c.pbp_game_id==key[1])&(c.team_code==key[2]),"off_epa_play__l3"].iloc[0]
    p2=p.copy(); m=(p2.season==2019)&(p2.game_id=="2019_03_AAA_BBB")&(p2.posteam=="NE"); p2.loc[m,"epa"]=99
    c2=build_prior_context(aggregate_pbp_game_team(p2)); after=c2.loc[(c2.Season==key[0])&(c2.pbp_game_id==key[1])&(c2.team_code==key[2]),"off_epa_play__l3"].iloc[0]
    if not (pd.isna(before) and pd.isna(after)) and abs(float(before)-float(after))>1e-12:
        raise AssertionError("SAME_GAME_PBP_LEAK_TEST_FAILED")
    if c.Season.max()>2025: raise AssertionError("SEALED_YEAR_TEST_FAILED")
    return {"status":"PASS","same_game_mutation_leakage":0,"game_team_rows":len(g),"context_rows":len(c)}
