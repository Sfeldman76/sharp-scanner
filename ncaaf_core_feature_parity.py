"""Research-only parity and population audit of frozen NCAAF V1 predictors.

Never changes Production V1, game data or source statistics. Compares feature
names to the read-only frozen production source (AST, not import), measures
non-null seasonal population and numeric variability, and reports actual folds.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CONTRACT = "NCAAF_RESEARCH_FROZEN_CORE_FEATURE_PARITY_20261017_V1"
REFERENCE = {
    "spread": (
        "Context_Intercept", "Diff_RawRecent3_Off_YPP",
        "B_RawSeason_GameAdj_Def_Rush_YPA",
    ),
    "totals": ("Context_Intercept", "A_RawRecent3_Def_Rush_YPA_Allowed"),
    "h2h": (
        "H2H2_Prior_Meetings", "H2H2_Prior_Margin_Current_Orientation",
        "H2H2_Prior_Total", "H2H2_Days_Since", "H2H2_Same_Home_Role",
        "H2H2_Recency_Margin", "H2H2_Recency_Total",
    ),
}
# These are a frozen STAT *family*, not a frozen list of specific fields.
# Its exact selected field names are learned at production refit and are not
# available from the production source's fixed backbone tuple constants.
STAT_FAMILY = "rushing"


def frozen_reference_from_source(source_file: str | Path | None = None) -> dict[str, Any]:
    path = Path(source_file) if source_file is not None else Path(__file__).with_name("ncaaf_production_v1.py")
    output={"source":str(path), "source_present":path.is_file(), "status":"UNVERIFIED_NO_SOURCE", "drift":{}}
    if not path.is_file():
        return output
    try:
        tree=ast.parse(path.read_text(encoding="utf8"))
        found={}
        for node in tree.body:
            if not isinstance(node, (ast.Assign,ast.AnnAssign)):
                continue
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t,ast.Name) and t.id in ("PROD_SPREAD_FEATURES","PROD_TOTAL_FEATURES","PROD_H2H_FEATURES"):
                    found[t.id]=tuple(ast.literal_eval(node.value))
        mapping={"spread":"PROD_SPREAD_FEATURES","totals":"PROD_TOTAL_FEATURES","h2h":"PROD_H2H_FEATURES"}
        drift={k:{"reference":list(REFERENCE[k]),"source":list(found.get(v,()))} for k,v in mapping.items() if found.get(v)!=REFERENCE[k]}
        output.update({"status":"MATCH" if not drift else "CONTRACT_DRIFT", "drift":drift})
    except (ValueError,TypeError,SyntaxError,UnicodeError,OSError) as exc:
        output.update({"status":"SOURCE_PARSE_ERROR","error":f"{type(exc).__name__}:{exc}"})
    return output


def required_research_features() -> tuple[str,...]:
    return tuple(dict.fromkeys(x for family in ("spread","totals","h2h") for x in REFERENCE[family]))


def _finite(s: pd.Series) -> pd.Series:
    v=pd.to_numeric(s,errors="coerce")
    return v.replace([np.inf,-np.inf],np.nan)


def _season_profile(frame:pd.DataFrame, name:str, season:int) -> dict[str, Any]:
    part=frame.loc[_finite(frame["Season"]).eq(season)]
    n=int(len(part))
    if name=="Context_Intercept":
        return {"season":season,"rows":n,"observed":n,"coverage":1.0 if n else None,
                "unique":1,"status":"DERIVED_CONSTANT","zeros":0}
    if name not in part:
        return {"season":season,"rows":n,"observed":0,"coverage":0.0 if n else None,
                "unique":0,"status":"COLUMN_MISSING","zeros":0}
    v=_finite(part[name]).dropna()
    count=int(v.size)
    item={"season":season,"rows":n,"observed":count,"coverage":round(count/n,4) if n else None,
          "unique":int(v.nunique()),"zeros":int(v.eq(0).sum())}
    if count:
        item.update({"median":round(float(v.median()),5),
                     "p10":round(float(v.quantile(.1)),5),"p90":round(float(v.quantile(.9)),5),
                     "std":round(float(v.std(ddof=0)),5)})
    if not n: status="NO_GAMES"
    elif count==0: status="NO_OBSERVATIONS"
    elif v.nunique()<2: status="CONSTANT_OR_ONE_OBSERVATION"
    elif count/n<.5: status="SPARSE_OBSERVED"
    else: status="OBSERVED"
    item["status"]=status
    return item


def audit_core_parity(frame:pd.DataFrame, selected:list[str],
                     fold_features:dict[int,list[str]]|None=None, *,
                     source_file:str|Path|None=None) -> dict[str,Any]:
    if not isinstance(frame,pd.DataFrame) or "Season" not in frame:
        raise ValueError("Season required for NCAAF source parity")
    selected=set(selected)
    year_values=_finite(frame["Season"]).dropna().astype(int)
    years=sorted(y for y in year_values.unique().tolist() if 2013<=y<=2025)
    actual_folds=fold_features or {}
    allrequired=required_research_features()
    detail={}
    for name in allrequired:
        populations=[_season_profile(frame,name,y) for y in years]
        historic=[x for x in populations if x["season"]<=2021]
        modern=[x for x in populations if x["season"]>=2022]
        def weighted(rows):
            denominator=sum(x["rows"] for x in rows)
            return round(sum(x["observed"] for x in rows)/denominator,4) if denominator else None
        detail[name]={"frozen_families":[k for k,v in REFERENCE.items() if name in v],
                      "present_in_frame":name in frame or name=="Context_Intercept",
                      "selected_for_foundation":name in selected,
                      "historic_2013_2021_coverage":weighted(historic),
                      "modern_2022_2025_coverage":weighted(modern),
                      "by_season":populations,
                      "folds_using_feature":[int(y) for y,v in sorted(actual_folds.items()) if name in v]}
    # Historical and contemporary populations should be *reported*, not silently
    # equated. Missing values are never manufactured to improve audit coverage.
    absent=[n for n,r in detail.items() if not r["present_in_frame"]]
    omitted=[n for n,r in detail.items() if r["present_in_frame"] and not r["selected_for_foundation"]]
    warnings=[]
    for name,d in detail.items():
        if name=="Context_Intercept":continue
        if d["historic_2013_2021_coverage"] is not None and d["historic_2013_2021_coverage"] < .5:
            warnings.append({"feature":name,"issue":"HISTORIC_COVERAGE_BELOW_50_PCT","coverage":d["historic_2013_2021_coverage"]})
        if d["modern_2022_2025_coverage"] is not None and d["modern_2022_2025_coverage"] < .5:
            warnings.append({"feature":name,"issue":"MODERN_COVERAGE_BELOW_50_PCT","coverage":d["modern_2022_2025_coverage"]})
    return {"contract":CONTRACT,"read_only":True,"production_mutation":False,
            "production_contract":frozen_reference_from_source(source_file),
            "years":years,"required_fixed_count":len(allrequired),
            "present_in_frame":len(allrequired)-len(absent),
            "selected_for_foundation":sum(n in selected for n in allrequired),
            "absent_columns":absent,"unselected_present_columns":omitted,
            "missing_frozen_stat_rushing_exact_selection":"PRODUCTION_RESEARCH_CACHE_NEEDED_TO_COMPARE_DYNAMIC_COLS",
            "production_population_comparison":"NOT_VERIFIED_SEPARATE_PRODUCTION_TRAINING_FRAME_NOT_SUPPLIED",
            "comparison_scope":"EXACT_FROZEN_SOURCE_FEATURE_NAMES_AND_RESEARCH_OBSERVED_COVERAGE_BY_SEASON",
            "fields":detail,"coverage_warnings":warnings,
            "status":"PARITY_REVIEW_REQUIRED" if absent or omitted or warnings else "FROZEN_FEATURES_COVERED_IN_RESEARCH"}
