"""Read-only checks at the *actual* NCAAF research/Heavy Research input boundary.

Never edits DataFrames or changes model selection, fitting, thresholds or authority.
The audit runs inside run_ncaaf_research_v2 immediately after historical input
construction. It also writes a separate, independently visible research report.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging

LOG=logging.getLogger('ncaaf_v55_research_health')
PREFIX='research/ncaaf/v55/pipeline_health_v1'


def _now(v=None):
    t=v or datetime.now(timezone.utc)
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def _check(out, code, status, why, **metrics):
    assert status in {'PASS','WARNING','FAIL','UNKNOWN'}
    out['checks'].append({'code':code,'status':status,'reason':why,'metrics':metrics})


def _summarize(out):
    counts=Counter(x['status'] for x in out['checks'])
    out['check_counts']={v:int(counts[v]) for v in ('PASS','WARNING','FAIL','UNKNOWN')}
    out['status']=next((v for v in ('FAIL','WARNING','UNKNOWN','PASS') if counts[v]),'UNKNOWN')
    out['failure_codes']=[x['code'] for x in out['checks'] if x['status']=='FAIL']
    out['warning_codes']=[x['code'] for x in out['checks'] if x['status']=='WARNING']
    out['unknown_codes']=[x['code'] for x in out['checks'] if x['status']=='UNKNOWN']
    return out


def _find(frame, names):
    return next((col for col in names if col in frame.columns),None)


def _safe_int(v):
    try:return int(v)
    except (ValueError,TypeError,OverflowError):return None


def _json(obj):
    return json.dumps(obj,sort_keys=True,default=str,separators=(',',':'),allow_nan=False)


def audit_research_frames(*, games, miner_games, seasons, oof_margin, oof_total,
                          candidate_feature_cols, source_tag='', now=None,
                          storage_client=None, bucket_name='sharp-models', persist=True):
    """Checks the exact frozen <=2025 row-aligned frames passed to research.

    Not a surrogate archive-table check. Only explicitly verified provenance can
    certify feature vintages. Missing feature coverage does not become a zero.
    """
    import numpy as np
    import pandas as pd

    t=_now(now)
    out={'schema_version':'V55_RESEARCH_INPUT_HEALTH_V1_20261101',
         'created_at_utc':t.isoformat(),'source_tag':str(source_tag),
         'scope':'RESEARCH_ACTUAL_INPUT_FRAMES_OBSERVABILITY_ONLY',
         'production_authority':0,'frames_mutated':False,
         'historical_feature_completeness_certified':False,
         'pregame_feature_vintage_certified':False,'checks':[]}
    if not isinstance(games,pd.DataFrame) or not isinstance(miner_games,pd.DataFrame):
        _check(out,'RESEARCH_FRAMES_PRESENT','FAIL','Research frames unavailable')
        return _finish(out,storage_client,bucket_name,persist)
    n=len(games)
    sy=pd.to_numeric(pd.Series(seasons).reset_index(drop=True),errors='coerce')
    arrays={'miner_rows':len(miner_games),'season_rows':len(sy),
            'oof_margin_rows':len(oof_margin),'oof_total_rows':len(oof_total)}
    aligned=bool(n>0 and all(k==n for k in arrays.values()))
    _check(out,'RESEARCH_FRAME_ROW_ALIGNMENT','PASS' if aligned else 'FAIL',
           'CORE, Miner, seasons, and out-of-fold predictions share identical row counts',
           core_rows=n,**arrays)
    if not aligned:return _finish(out,storage_client,bucket_name,persist)

    sy_arr=sy.to_numpy(dtype=float)
    finite=np.isfinite(sy_arr)
    years=sorted({_safe_int(x) for x in sy_arr[finite]})
    prospective=int((sy_arr>=2026).sum())
    unassigned=int((~finite).sum())
    _check(out,'RESEARCH_HISTORICAL_SEASON_SEAL','FAIL' if prospective or unassigned else 'PASS',
           'Actual fit/discovery frame must exclude prospective 2026+ and unassigned seasons',
           year_min=years[0] if years else None,year_max=years[-1] if years else None,
           seasons=years,prospective_rows=prospective,unassigned_season_rows=unassigned)
    _check(out,'RESEARCH_SEASON_COVERAGE','WARNING' if years and years[0]>2013 else 'PASS' if years else 'UNKNOWN',
           'Historical input years observed; earlier archived seasons may legitimately be excluded by research eligibility',
           earliest_input_year=years[0] if years else None,expected_archive_start=2013)

    features=list(dict.fromkeys(str(x) for x in (candidate_feature_cols or []) if str(x).strip()))
    missing=[f for f in features if f not in games.columns]
    _check(out,'RESEARCH_FEATURE_REGISTRY_JOIN','FAIL' if missing else 'PASS' if features else 'UNKNOWN',
           'Candidate feature registry resolved against the actual CORE/STAT research input',
           registry_features=len(features),missing_count=len(missing),missing_features=missing[:30])
    coverage={}
    sparse=[]
    for c in features:
        if c not in games.columns:continue
        series=pd.to_numeric(games[c],errors='coerce')
        valid=np.isfinite(series.to_numpy(dtype=float))
        by_season={str(y):round(float(valid[sy_arr==y].mean()),4) for y in years if (sy_arr==y).any()}
        pct=float(valid.mean()) if n else 0.0
        if pct<0.50:sparse.append({'feature':c,'finite_fraction':round(pct,4)})
        coverage[c]=by_season
    _check(out,'RESEARCH_FEATURE_NONNULL_COVERAGE','WARNING' if sparse else 'PASS' if features and not missing else 'UNKNOWN',
           'Per-season numeric feature availability (legitimate sparse features are warnings, not imputed)',
           sparse_features=sparse[:30],per_season_finite_coverage=coverage)
    # Source-family columns are inspected in the *enriched Miner matrix*.
    # An absent optional brain is UNKNOWN, not proof of disconnection; a field
    # populated only for some games is normal and must not be zero-filled.
    family_tags={
        'MINER':('miner_','system_miner_','ncaaf_rv22_'),
        'PATHI':('pathi_','expert_pathi_'),
        'BIG_AL':('bigal_','big_al_','expert_bigal_'),
        'PT':('ncaaf_pt_','prediction_tracker_','ptidx_','_pt_meta_'),
        'STAT':('stat_','ncaaf_stat_','sparse_stat_'),
        'MARKET_RICH':('market_rich_','opening_spread','line_move','sportsbook_'),
        'CFBD':('cfbd_','away_travel_','venue_elevation','altitude_gain','roster_','personnel_'),
    }
    for family,tags in family_tags.items():
        field_list=[c for c in miner_games.columns if any(tag in str(c).lower() for tag in tags)]
        with_values=[]
        for c in field_list:
            series=miner_games[c]
            valid=series.notna() & series.astype(str).str.strip().ne('')
            if bool(valid.any()):with_values.append(c)
        _check(out,f'RESEARCH_{family}_FEATURE_FLOW',
               'PASS' if with_values else 'WARNING' if field_list else 'UNKNOWN',
               'Observed optional family columns in enriched historical Miner research frame',
               available_columns=len(field_list),populated_columns=len(with_values),
               example_columns=[str(x) for x in field_list[:8]])
    # Test the *same ordered sides* in both model research matrices.
    match_cols=[c for c in ('Season','Game_ID','Game_Date','Team_Norm','Opponent_Norm','Is_Home')
                if c in games.columns and c in miner_games.columns]
    mismatches={}
    for c in match_cols:
        a=games[c].reset_index(drop=True).fillna('<MISSING>').astype(str)
        b=miner_games[c].reset_index(drop=True).fillna('<MISSING>').astype(str)
        d=int((a!=b).sum())
        if d:mismatches[c]=d
    _check(out,'RESEARCH_CORE_MINER_ROW_KEYS','FAIL' if mismatches else 'PASS' if match_cols else 'UNKNOWN',
           'CORE research and Miner enriched rows must preserve original game-side order',
           compared_key_fields=match_cols,mismatched_fields=mismatches)
    side_cols=[c for c in ('Season','Game_ID','Team_Norm','Opponent_Norm','Is_Home') if c in games.columns]
    if 'Game_ID' in side_cols and 'Team_Norm' in side_cols:
        df=games[side_cols].fillna('<MISSING>').astype(str)
        duplicates=int(df.duplicated(keep='first').sum())
        _check(out,'RESEARCH_DUPLICATE_GAME_SIDES','FAIL' if duplicates else 'PASS',
               'Duplicate same-game same-side rows; two opposing sides per game are valid',
               duplicate_side_rows=duplicates,side_key_fields=side_cols)
    else:
        _check(out,'RESEARCH_DUPLICATE_GAME_SIDES','UNKNOWN',
               'Game and side key columns are insufficient for a reliable duplication check',
               available_key_fields=side_cols)

    missing_oof={}
    for name,values in (('margin',oof_margin),('total',oof_total)):
        try:
            arr=np.asarray(values,dtype=float)
            n_valid=int(np.isfinite(arr).sum())
            missing_oof[name]={'finite':n_valid,'missing':int(n-n_valid)}
        except (TypeError,ValueError):
            missing_oof[name]={'finite':0,'missing':n}
    _check(out,'RESEARCH_OOF_PREDICTION_COVERAGE',
           'WARNING' if any(x['missing'] for x in missing_oof.values()) else 'PASS',
           'Missing OOF values are reported, never filled; early seasons may lack a trained predecessor',
           oof=missing_oof)

    # We cannot certify as-of correctness from column *names* or today's data.
    # Only reject obviously future quote timestamps when both timestamp fields are present.
    game_time=_find(games,('Game_Start_UTC','Kickoff_UTC'))
    quote_time=_find(games,('Source_Snapshot_UTC','Snapshot_Timestamp','Quote_Timestamp_UTC'))
    if game_time and quote_time:
        kick=pd.to_datetime(games[game_time],utc=True,errors='coerce')
        quote=pd.to_datetime(games[quote_time],utc=True,errors='coerce')
        both=kick.notna()&quote.notna()
        late=int(((quote>=kick)&both).sum())
        _check(out,'RESEARCH_QUOTE_PREGAME_CLOCK','FAIL' if late else 'PASS' if int(both.sum()) else 'UNKNOWN',
               'Observed quote-clock order, not a complete feature-vintage certification',
               timed_rows=int(both.sum()),late_or_at_kickoff=late,game_field=game_time,quote_field=quote_time)
    else:
        _check(out,'RESEARCH_QUOTE_PREGAME_CLOCK','UNKNOWN',
               'Quote and actual kickoff clocks not both in research input; cannot infer from game date',
               kickoff_field=game_time,quote_field=quote_time)
    _check(out,'RESEARCH_ALL_FEATURE_VINTAGES','UNKNOWN',
           'Row-level source-as-of evidence for every feature is not supplied; anti-leakage is not certified by chronology alone')
    out['row_summary']={'rows':n,'seasons':years,'feature_count':len(features),
                        'oof_coverage':missing_oof}
    return _finish(out,storage_client,bucket_name,persist)


def _finish(report,storage_client,bucket_name,persist):
    _summarize(report)
    if persist:
        try:
            if storage_client is None:
                from google.cloud import storage
                storage_client=storage.Client()
            b=storage_client.bucket(bucket_name)
            stamp=report['created_at_utc'].replace('-','').replace(':','').replace('+00:00','Z').replace('.','')
            body=_json(report)
            sha=hashlib.sha256(body.encode()).hexdigest()
            path=f'{PREFIX}/research_runs/{stamp}_{sha[:16]}.json'
            b.blob(path).upload_from_string(body,content_type='application/json',if_generation_match=0)
            b.blob(f'{PREFIX}/latest_research.json').upload_from_string(body,content_type='application/json')
            report['report_uri']=f'gs://{bucket_name}/{path}'
        except Exception as e:
            _check(report,'RESEARCH_AUDIT_REPORT_WRITE','FAIL',
                   'Cannot publish read-only research audit',exception=type(e).__name__)
            _summarize(report)
    LOG.info('[NCAAF-V55-RESEARCH-PIPELINE-HEALTH] %s',_json({
        'status':report['status'],'check_counts':report['check_counts'],
        'failure_codes':report['failure_codes'],'warning_codes':report['warning_codes'],
        'unknown_codes':report['unknown_codes'],'report_uri':report.get('report_uri'),
        'production_authority':0}))
    return report
