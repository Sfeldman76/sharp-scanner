"""Display-only integration of the one executable Miner library into the live evaluator.

Never emits NCAAF_RV22_Miner_Votes or changes V55_Bet_Authority; original
qualified-event contract is the only pathway to betting eligibility.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import logging
import os
from ncaaf_v55_executable_miner_library import (CURRENT,PROTOCOL,
                                                  validate_library,evaluate_research_frame,MARKETS)


def read_library(*, storage_client=None,bucket_name=None):
    from google.cloud import storage
    cli=storage_client or storage.Client()
    bucket=bucket_name or os.getenv('MODEL_BUCKET','sharp-models')
    b=cli.bucket(bucket)
    pointer=json.loads(b.blob(CURRENT).download_as_bytes())
    if pointer.get('schema')!=PROTOCOL or pointer.get('bet_authority')!=0:
        raise RuntimeError('MINER_CURRENT_POINTER_SCHEMA_OR_AUTHORITY_INVALID')
    uri=pointer.get('latest_uri','')
    expected='gs://'+bucket+'/research/ncaaf/v55/unified_executable_miner_library_v2/runs/'
    if not uri.startswith(expected) or not uri.endswith('/library.json'):
        raise RuntimeError('MINER_INVALID_IMMUTABLE_SOURCE')
    raw=b.blob(uri[len('gs://'+bucket+'/'):]).download_as_bytes()
    if hashlib.sha256(raw).hexdigest()!=pointer.get('sha256'):
        raise RuntimeError('MINER_IMMUTABLE_LIBRARY_SHA256_MISMATCH')
    obj=json.loads(raw)
    validate_library(obj)
    if obj['rule_count']!=pointer.get('rule_count'):
        raise RuntimeError('MINER_CURRENT_POINTER_COUNT_MISMATCH')
    if obj['market_counts']!=pointer.get('market_counts'):
        raise RuntimeError('MINER_CURRENT_POINTER_MARKET_COUNTS_MISMATCH')
    return obj,uri


def _norm_name(x):
    return ' '.join(str(x or '').lower().strip().split())


def _market_source_frame(out, enriched, keycol, market):
    """Select actual market rows; NEVER infer H2H/totals prices from spreads.

    For side markets the representative must be the HOME quote; totals require
    the OVER quote. The rule's PLAY_ON/FADE direction then has unambiguous
    orientation. Strictly research-only: first observed eligible quote per game.
    """
    import pandas as pd
    if market=='spreads':
        # The canonical enriched home-side game frame already used by research.
        if 'Market' in enriched.columns:
            sp=enriched['Market'].fillna('').astype(str).str.lower().eq('spreads')
            if not sp.all():
                # The upstream context picked a fallback when no spread exists;
                # such rows cannot be used for spread price-dependent rules.
                enriched=enriched.loc[sp].copy()
        return enriched
    if 'Market' not in out.columns:
        return out.iloc[0:0].copy()
    frame=out.loc[out['Market'].fillna('').astype(str).str.lower().eq(market)].copy()
    if frame.empty:return frame
    if keycol not in frame:return frame.iloc[0:0].copy()
    chosen=[]
    for _,group in frame.groupby(keycol,sort=False,dropna=False):
        # Selecting canonical source perspective is not choosing a bet.
        # Take the latest *authentic pregame* quote, never a synthetic spread
        # substitute or a postgame snapshot. Deterministic source tie-breaks.
        choices=[]
        def field(row, *names):
            for name in names:
                v=row.get(name)
                if v is not None and not isinstance(v,(dict,list)) and pd.notna(v) and str(v).strip():
                    return str(v)
            return ''
        for _,r in group.iterrows():
            home=_norm_name(field(r,'Home_Team_Norm','Home_Team'))
            outcome=_norm_name(field(r,'Outcome_Norm','Outcome'))
            if not ((market=='totals' and outcome=='over') or
                    (market=='h2h' and home and outcome==home)):
                continue
            kickoff=pd.to_datetime(r.get('Game_Start'),utc=True,errors='coerce')
            stamp=pd.to_datetime(r.get('Snapshot_Timestamp'),utc=True,errors='coerce')
            if pd.isna(kickoff) or pd.isna(stamp) or stamp>=kickoff:
                continue
            choices.append((stamp.isoformat(),field(r,'Bookmaker','Book'),
                            field(r,'Value'),r))
        if choices:
            _,_,_,candidate=sorted(choices,key=lambda x:(x[0],x[1],x[2]))[-1]
            q=candidate.copy()
            q['_rv22_game_id_key']=candidate[keycol]
            chosen.append(q)
    if not chosen:return frame.iloc[0:0].copy()
    return pd.DataFrame(chosen).reset_index(drop=True)


def attach_shadow_live_library(rows, enriched_game_rows, keycol, *, dashboard_module=None,
                               library=None,storage_client=None,bucket_name=None):
    """Run frozen rules market by market, never crossing a quote/market boundary.

    Spreads use upstream enriched canonical-home context. H2H and totals use
    their OWN actual live rows, their OWN price, and their OWN source clock.
    Without that source row the rule is unevaluable; no synthetic market.
    """
    out=rows.copy()
    col='NCAAF_V55_Unified_Miner_Signals_JSON'
    out[col]='[]'
    out['NCAAF_V55_Unified_Miner_Live_Status']='HOLD_NO_LIBRARY'
    out['NCAAF_V55_Unified_Miner_Executable_Rules']=0
    if out.empty or enriched_game_rows is None or enriched_game_rows.empty:
        return out,{'status':'NO_LIVE_ROWS','authority_changed':False}
    if keycol not in out or '_rv22_game_id_key' not in enriched_game_rows:
        raise ValueError('LIVE_KEY_CONTRACT_MISSING')
    if library is None:library,uri=read_library(storage_client=storage_client,bucket_name=bucket_name)
    else:uri='INJECTED_TEST_LIBRARY'
    validate_library(library)
    if 'Market' in out.columns:
        original_market=out['Market'].fillna('').astype(str).str.lower().str.strip()
    else:
        # Backward compatibility only: old spread-only test/source frames.
        original_market=['spreads']*len(out)
    by_game_market=defaultdict(list)
    audit={'status':'PASS_RESEARCH_ONLY','immutable_uri':uri,'rule_count':library['rule_count'],
           'market_counts':library['market_counts'],'by_market':{},
           'fired':0,'unknown':0,'nonfired':0,'pregame_clock_unverified':0,
           'authority_events':0,'authority_changed':False}
    for market in MARKETS:
        if not library['market_counts'][market]:
            audit['by_market'][market]={'status':'NO_FROZEN_RULES_FOR_MARKET','source_rows':0,'fired':0}
            continue
        frame=_market_source_frame(out,enriched_game_rows,keycol,market)
        if frame.empty:
            audit['by_market'][market]={'status':'NO_VERIFIED_MARKET_ORIENTED_SOURCE_ROWS','source_rows':0,'fired':0}
            continue
        # Every row must be the source's real timestamp and a real game kickoff.
        if 'Game_Start' not in frame or 'Snapshot_Timestamp' not in frame:
            audit['by_market'][market]={'status':'MARKET_SOURCE_CLOCK_FIELDS_UNAVAILABLE',
                                       'source_rows':len(frame),'fired':0}
            continue
        fired,stats=evaluate_research_frame(library,frame,
                                           dashboard_module=dashboard_module,market=market)
        for event in fired:
            key=frame.iloc[event['row_index']]['_rv22_game_id_key']
            by_game_market[(str(key),market)].append({k:event[k] for k in
                ('rule_id','mechanism_id','market','direction','target_orientation','evaluation_state')})
        stats['status']='RESEARCH_ONLY_EVALUATED'
        stats['source_rows']=int(len(frame))
        audit['by_market'][market]=stats
        for k in ('fired','unknown','nonfired','pregame_clock_unverified','authority_events'):
            audit[k]+=stats.get(k,0)
    out[col]=[json.dumps(by_game_market.get((str(k),str(m)),[]),sort_keys=True)
              for k,m in zip(out[keycol],original_market)]
    out['NCAAF_V55_Unified_Miner_Live_Status']='RESEARCH_ONLY_MARKET_SPECIFIC_RULES'
    out['NCAAF_V55_Unified_Miner_Executable_Rules']=int(library['rule_count'])
    logging.info('[NCAAF-V55-UNIFIED-MINER-LIVE] '+json.dumps(audit,sort_keys=True))
    return out,audit
