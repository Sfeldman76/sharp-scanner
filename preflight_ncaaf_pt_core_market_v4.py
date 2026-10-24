"""Read-only, *single-pass* source preflight for NCAAF PT alias and CORE market gaps.

Cloud Shell: python3 preflight_ncaaf_pt_core_market_v4.py --out ~/pt_core_market_v4.json
No BigQuery writes, no model refit, no production mutation; no authority grant.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd
from ncaaf_historical_brain_replay_v1 import (
    load_bigquery_source_frames, _stage_cfbd, _stage_context, _attach_context,
    pt_archive_inputs, attach_pt, compare_to_core)
from preflight_ncaaf_replay_identity import _core_from_gcs


def market_alternatives(games: pd.DataFrame,core: pd.DataFrame)->dict:
    """Report possible fallback coverage without *using* a fallback in scoring."""
    needed={'season','game_id','open_home_spread','actual_home_margin'}
    if not needed.issubset(games):
        raise ValueError('CORE_MARKET_PREFLIGHT_GAME_SCHEMA_MISSING')
    if not {'season','game_id','model_home_margin'}.issubset(core):
        raise ValueError('CORE_MARKET_PREFLIGHT_CORE_SCHEMA_MISSING')
    cf=core[['season','game_id','model_home_margin']].copy()
    cf['season']=pd.to_numeric(cf.season,errors='coerce').astype('Int64')
    cf['game_id']=cf.game_id.astype('string')
    gm=games[['season','game_id','open_home_spread','actual_home_margin']+
             (['Spread_Value'] if 'Spread_Value' in games else [])].copy()
    gm['season']=pd.to_numeric(gm.season,errors='coerce').astype('Int64')
    gm['game_id']=gm.game_id.astype('string')
    merged=cf.merge(gm,on=['season','game_id'],how='left',indicator=True,validate='one_to_one')
    open_line=pd.to_numeric(merged.open_home_spread,errors='coerce')
    alt=pd.to_numeric(merged['Spread_Value'],errors='coerce') if 'Spread_Value' in merged else pd.Series(float('nan'),index=merged.index)
    actual=pd.to_numeric(merged.actual_home_margin,errors='coerce')
    pred=pd.to_numeric(merged.model_home_margin,errors='coerce')
    missing_open=open_line.isna()
    fallback_only=missing_open & alt.notna()
    any_market=open_line.notna() | alt.notna()
    fallback_market=open_line.where(open_line.notna(),alt)
    # Not a second backtest: compare descriptive sample coverage only.
    rows=[]
    for year, g in merged.groupby('season'):
        idx=g.index
        rows.append({'season':int(year),
          'core_games':int(len(idx)),
          'cfbd_id_matched':int(merged.loc[idx,'_merge'].eq('both').sum()),
          'open_present':int(open_line.loc[idx].notna().sum()),
          'open_missing_fallback_Spread_Value_present':int(fallback_only.loc[idx].sum()),
          'market_either_present':int(any_market.loc[idx].sum()),
          'both_market_fields_missing':int((~any_market.loc[idx]).sum()),
          'outcome_present':int(actual.loc[idx].notna().sum()),
          'eligible_open_edge_ge2':int((open_line.loc[idx].notna() & actual.loc[idx].notna() &
               (pred.loc[idx]+open_line.loc[idx]).abs().ge(2)).sum()),
          'potential_eligible_with_Spread_Value_fallback_edge_ge2':int((fallback_market.loc[idx].notna() & actual.loc[idx].notna() &
               (pred.loc[idx]+fallback_market.loc[idx]).abs().ge(2)).sum()),
        })
    return {'per_season':rows,
       'fallback_is_not_enabled_in_comparisons':True,
       'Spread_Value_quote_type_or_original_timestamp_verified':False,
       'fallback_vintage_certified':False,
       'interpretation':'Research coverage diagnostic only; do NOT equate Spread_Value with original opening price or a contemporaneously tradable quote.'}


def inspect(project,dataset,bucket):
    cfbd,context,aliases=load_bigquery_source_frames(project=project,dataset=dataset)
    u,_,_= _stage_cfbd(cfbd)
    ctx,_=_stage_context(context)
    games,join=_attach_context(u,ctx,aliases=aliases)
    from google.cloud import storage
    storage_client=storage.Client(project=project)
    pt,pt_source=pt_archive_inputs(storage_client,bucket)
    signals,pt_audit=attach_pt(games,pt,aliases=aliases)
    core,receipt=_core_from_gcs(bucket)
    compared,_,attrition=compare_to_core(games,signals,core)
    return {
      'status':'NCAAF_READ_ONLY_PT_CORE_MARKET_V4',
      'generated_utc':datetime.now(timezone.utc).isoformat(),
      'production_mutation':False,'bet_authority':0,
      'cfbd_games':len(games),'context_matched':int(games.has_context.sum()),
      'pt':pt_audit,'pt_archives':pt_source,
      'core':attrition,'core_provenance':receipt,
      'core_eligible_research_comparisons':len(compared),
      'market_alternatives':market_alternatives(games,core),
      'historical_publication_vintage_certified':False,
      'no_historical_rule_promotion':True,
    }


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',default='sharplogger')
    ap.add_argument('--dataset',default='sharp_data')
    ap.add_argument('--bucket',default='sharp-models')
    ap.add_argument('--out',default='pt_core_market_v4.json')
    args=ap.parse_args()
    report=inspect(args.project,args.dataset,args.bucket)
    Path(args.out).write_text(json.dumps(report,indent=2,default=str))
    print('PT:',report['pt'].get('matched_games'),'/',report['pt'].get('pt_source_rows'))
    print('PT unresolved teams:',report['pt'].get('unresolved_pt_team_names_ranked',[])[:12])
    print('PT reverse orientation:',report['pt'].get('reversed_orientation_matches',0))
    print('CORE matched:',report['core'].get('core_identity_matched_games'))
    print('CORE edge eligible:',report['core'].get('eligible_core_comparisons'))
    print('Fallback market by season:',[(r['season'],r['open_missing_fallback_Spread_Value_present']) for r in report['market_alternatives']['per_season']])
    print('Saved:',str(Path(args.out).resolve()))
