"""Restore *real observed* pregame PT and static CFBD inputs to the NCAAF Miner.

No guessed values, fake zeroes, new rule qualifications, model refit, or authority
changes. Existing frozen rule/underlying-family gates remain the sole authority.
"""
from __future__ import annotations
import re
from datetime import datetime, timezone
import numpy as np
import pandas as pd

# Exact school-key canonicalization shared with V5.5's VERIFIED price bridge.
# Never use fuzzy closest-team guessing for bet-authority pregame evidence.
def school_key(value):
    try:
        from ncaaf_v55_board import _norm_team
        return _norm_team(value)
    except ImportError:
        # Research-only image without the V5.5 board: use deterministic,
        # conservative canonicalization without inventing a fuzzy match.
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


STATIC_CFBD = (
    'CFBD_QB_Returning_Flag', 'CFBD_Portal_In_Count', 'CFBD_Portal_Out_Count',
    'CFBD_Portal_Net_Count', 'CFBD_Portal_In_QB_Count', 'CFBD_Portal_Out_QB_Count',
    'CFBD_Portal_In_4StarPlus', 'CFBD_Portal_Out_4StarPlus',
    'CFBD_Portal_In_QB_Prior_TotalPPA', 'CFBD_Roster_ALL_Retention',
    'CFBD_Returning_Pct_PPA', 'CFBD_Talent_Rank',
)


def restore_cfbd_static(cdf, cfbd_table, *, team_key, asof=None):
    """Carry *season-static* values forward from strictly earlier same-season games.

    Never backfill mutable shifted game statistics or game-result-dependent fields.
    Ambiguous prior team-season values are not considered authoritative.
    """
    out = cdf.copy()
    diag = {'status':'NO_SOURCE', 'rows':len(out), 'filled_cells':0,
            'matched_games':0,'source_static_fields':[], 'team_season_prior_matches':0, 'ambiguous_static_fields':0}
    if not isinstance(cfbd_table,pd.DataFrame) or cfbd_table.empty or out.empty:
        return out,diag
    fields = [k for k in STATIC_CFBD if k in cfbd_table]
    if not fields or 'Season' not in cfbd_table or 'Game_Date' not in cfbd_table:
        return out,diag
    if not any(k in cfbd_table for k in ('Team_Key','Team')):
        return out,diag
    right=cfbd_table.copy()
    right['_sy']=pd.to_numeric(right['Season'],errors='coerce')
    right['_date']=pd.to_datetime(right['Game_Date'],errors='coerce',utc=True)
    rc='Team_Key' if 'Team_Key' in right else 'Team'
    right['_team']=right[rc].map(school_key)
    right=right.loc[right['_sy'].notna() & right['_date'].notna() & right['_team'].ne('')].copy()
    if right.empty: return out,diag
    for k in fields: right[k]=pd.to_numeric(right[k],errors='coerce')
    # Detect changing fields only in observations PRECEDING the upcoming game.
    # Future rows must neither supply values nor decide whether they are safe.
    season=pd.to_numeric(out.get('Season',pd.Series(np.nan,index=out.index)),errors='coerce')
    date=pd.to_datetime(out.get('Game_Date',out.get('Game_Start',pd.Series(pd.NaT,index=out.index))),errors='coerce',utc=True)
    nm=out.get('Team_Norm',out.get('Home_Team_Norm',out.get('Home_Team',pd.Series('',index=out.index)))).map(school_key)
    now=pd.Timestamp(asof if asof is not None else datetime.now(timezone.utc))
    if now.tzinfo is None: now=now.tz_localize('UTC')
    filled=0; rows=0; prior_matches=0; ambiguous=0
    new_vals={k:pd.to_numeric(out[k],errors='coerce').copy() if k in out else pd.Series(np.nan,index=out.index,dtype=float) for k in fields}
    for i in out.index:
        if pd.isna(season.loc[i]) or pd.isna(date.loc[i]) or date.loc[i]<=now or not nm.loc[i]:
            continue
        candidates=right.loc[(right['_sy']==int(season.loc[i]))
                             & right['_team'].eq(nm.loc[i])
                             & right['_date'].lt(date.loc[i])
                             & right['_date'].le(now)]
        if candidates.empty:
            continue
        prior_matches += 1
        did=False
        for k in fields:
            if pd.notna(new_vals[k].loc[i]):
                continue
            values=candidates[k].dropna()
            if values.empty:
                continue
            # A purported season-static field must agree across all earlier
            # observations. Never pick an arbitrary row when it changes.
            if values.nunique()>1:
                ambiguous += 1
                continue
            new_vals[k].loc[i]=float(values.iloc[-1]);filled+=1;did=True
        rows+=int(did)
    for k,v in new_vals.items(): out[k]=v
    diag={'status':'PASS','rows':len(out),'filled_cells':filled,'matched_games':rows,
          'source_static_fields':fields,'team_season_prior_matches':prior_matches, 'ambiguous_static_fields':ambiguous, 'mutable_game_priors_carried':False,
          'source':'SAME_SEASON_EARLIER_CFBD_GAME_ONLY'}
    return out,diag


def restore_pt_index_features(cdf, exact_pair_pt, *, home_keys, away_keys,
                              snapshot_updated_utc, source_cluster,
                              meta_constituents, asof=None):
    """Match exact live games and reproduce historical PT index atoms.

    `exact_pair_pt` is the unique pair-key current.csv frame already alias-mapped
    by _pt_attach_current_live_frame; duplicate keys must be removed upstream.
    Snapshot updated timestamp and source season/date are required to prevent
    a postgame or wrong-season observation becoming a qualified Miner trigger.
    """
    out=cdf.copy()
    d={'status':'NO_VALID_SNAPSHOT','matched_games':0,'source_indexes':0,
       'usable_rows':0,'restored_atoms':0,'source':'PT_CURRENT_BLOB_EXACT_PAIR', 'pairs_found':0, 'missing_game_date_matches':0, 'rejected_missing_open':0, 'rejected_stale_blob':0}
    if out.empty or not isinstance(exact_pair_pt,pd.DataFrame) or exact_pair_pt.empty:
        return out,d
    snap=pd.to_datetime(snapshot_updated_utc,errors='coerce',utc=True)
    if pd.isna(snap): return out,d
    current=pd.to_datetime(asof if asof is not None else datetime.now(timezone.utc),utc=True)
    if snap>current+pd.Timedelta(minutes=5): return out,d
    # A current snapshot older than 14 days cannot establish actionable
    # contemporary PT evidence; request the upstream PT refresh instead.
    if snap < current-pd.Timedelta(days=14):
        return out,{**d,'status':'STALE_PT_CURRENT_BLOB','rejected_stale_blob':len(out)}
    cols=sorted(k for k in exact_pair_pt if k.startswith('ptidx__'))
    if not cols:return out,{**d,'status':'NO_SOURCE_INDEX_COLUMNS'}
    n=len(out); ids=[k[len('ptidx__'):] for k in cols]
    book_spread=pd.to_numeric(out.get('Consensus_Open_Spread',out.get('Opening_Spread',pd.Series(np.nan,index=out.index))),errors='coerce').to_numpy(float)
    market=-book_spread
    starts=pd.to_datetime(out.get('Game_Start',out.get('Game_Date',pd.Series(pd.NaT,index=out.index))),errors='coerce',utc=True)
    seasons=pd.to_numeric(out.get('Season',pd.Series(np.nan,index=out.index)),errors='coerce')
    indexes=np.full((n,len(ids)),np.nan)
    matched=0; pairs_found=0; no_open=0; missing_date_matches=0
    for i,(h,a) in enumerate(zip(home_keys,away_keys)):
        key=str(h)+'|'+str(a)
        if key not in exact_pair_pt.index:
            continue
        pairs_found += 1
        if not np.isfinite(market[i]):
            no_open += 1
            continue
        row=exact_pair_pt.loc[key]
        if isinstance(row,pd.DataFrame):
            continue
        kickoff=starts.iloc[i]
        if pd.isna(kickoff) or kickoff<=snap or kickoff<=current:continue
        # Do not reuse a prior-season or wrong-day prediction for this game.
        if 'season' not in row or pd.isna(seasons.iloc[i]):continue
        src_sy=pd.to_numeric(row.get('season'),errors='coerce')
        if pd.isna(src_sy) or int(src_sy)!=int(seasons.iloc[i]):continue
        src_date=pd.to_datetime(row.get('game_date'),errors='coerce',utc=True)
        if pd.isna(src_date):
            # Real 2026 current.csv has NO populated game_date. For unique
            # same-season, exactly oriented, pregame matchup rows, verified
            # kickoff in sportsbook data supplies the date identity. No
            # date is invented or written into the external source.
            missing_date_matches += 1
        else:
            event_days={kickoff.date(),kickoff.tz_convert('America/New_York').date()}
            if src_date.date() not in event_days:
                continue
        for j,k in enumerate(cols):
            v=pd.to_numeric(row.get(k),errors='coerce')
            if pd.notna(v) and np.isfinite(float(v)):
                indexes[i,j]=float(v)
        matched+=1
    valid=np.isfinite(indexes);count=valid.sum(axis=1).astype(float)
    edge=indexes-market[:,None]
    new={}
    for j,k in enumerate(ids):
        if not valid[:,j].any(): continue  # Missing provider is not an evaluable atom.
        slug=re.sub('[^A-Z0-9]+','_',k.upper())
        new[f'_V218_PTIDX_{slug}_EDGE_POINTS']=edge[:,j]
        new[f'_V218_PTIDX_{slug}_MARGIN_TEAM']=indexes[:,j]
    new['_V218_PTIDX_AVAILABLE_COUNT']=count
    new['_V218_PTIDX_OPP_AGREE_FRAC']=np.divide(np.sum(valid & (edge<0),axis=1),count,out=np.full(n,np.nan),where=count>0)
    new['_V218_PTIDX_TEAM_AGREE_FRAC']=np.divide(np.sum(valid & (edge>0),axis=1),count,out=np.full(n,np.nan),where=count>0)
    new['_V218_PTIDX_STRONG_OPP_FRAC']=np.divide(np.sum(valid & (edge<=-2),axis=1),count,out=np.full(n,np.nan),where=count>0)
    new['_V218_PTIDX_STRONG_TEAM_FRAC']=np.divide(np.sum(valid & (edge>=2),axis=1),count,out=np.full(n,np.nan),where=count>0)
    means=np.full(n,np.nan);med=np.full(n,np.nan); std=np.full(n,np.nan);iqr=np.full(n,np.nan)
    for i in range(n):
        vals=indexes[i,valid[i]]
        if len(vals):means[i]=np.mean(vals);med[i]=np.median(vals)
        if len(vals)>=2:std[i]=np.std(vals);iqr[i]=np.percentile(vals,75)-np.percentile(vals,25)
    new['_V218_PTIDX_MEDIAN_EDGE_POINTS']=med-market
    new['_V218_PTIDX_MEAN_EDGE_POINTS']=means-market
    new['_V218_PTIDX_STD']=std
    new['_V218_PTIDX_IQR']=iqr
    clusters={}
    for j,k in enumerate(ids): clusters.setdefault(source_cluster(k),[]).append(j)
    clnames=sorted(clusters)
    clmat=np.full((n,len(clnames)),np.nan)
    for j,k in enumerate(clnames):
        inds=clusters[k]
        for i in range(n):
            vv=indexes[i,inds]; vv=vv[np.isfinite(vv)]
            if len(vv):clmat[i,j]=np.mean(vv)  # historical Miner contract
    # Full index cluster counts (historical contract).
    cl_count=np.isfinite(clmat).sum(axis=1).astype(float)
    cl_edge=clmat-market[:,None]
    new['_V218_PTIDX_CLUSTER_COUNT']=cl_count
    new['_V218_PTIDX_CLUSTER_OPP_AGREE_FRAC']=np.divide(np.sum(np.isfinite(cl_edge)&(cl_edge<0),axis=1),cl_count,out=np.full(n,np.nan),where=cl_count>0)
    new['_V218_PTIDX_CLUSTER_TEAM_AGREE_FRAC']=np.divide(np.sum(np.isfinite(cl_edge)&(cl_edge>0),axis=1),cl_count,out=np.full(n,np.nan),where=cl_count>0)
    clmed=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan for v in clmat])
    new['_V218_PTIDX_CLUSTER_MEDIAN_EDGE_POINTS']=clmed-market
    excluded={source_cluster(k) for k in meta_constituents}
    keep=[j for j,k in enumerate(ids) if source_cluster(k) not in excluded]
    exidx=indexes[:,keep]
    excount=np.isfinite(exidx).sum(axis=1).astype(float)
    exedge=exidx-market[:,None]
    new['_V2182_PT_EXMETA_AVAILABLE_COUNT']=excount
    new['_V2182_PT_EXMETA_OPP_AGREE_FRAC']=np.divide(np.sum(np.isfinite(exedge)&(exedge<0),axis=1),excount,out=np.full(n,np.nan),where=excount>0)
    new['_V2182_PT_EXMETA_TEAM_AGREE_FRAC']=np.divide(np.sum(np.isfinite(exedge)&(exedge>0),axis=1),excount,out=np.full(n,np.nan),where=excount>0)
    new['_V2182_PT_EXMETA_STRONG_OPP_FRAC']=np.divide(np.sum(np.isfinite(exedge)&(exedge<=-2),axis=1),excount,out=np.full(n,np.nan),where=excount>0)
    new['_V2182_PT_EXMETA_STRONG_TEAM_FRAC']=np.divide(np.sum(np.isfinite(exedge)&(exedge>=2),axis=1),excount,out=np.full(n,np.nan),where=excount>0)
    exmed=np.full(n,np.nan); exmean=np.full(n,np.nan); exstd=np.full(n,np.nan); exiqr=np.full(n,np.nan)
    for i in range(n):
        v=exidx[i,np.isfinite(exidx[i])]
        if len(v):exmed[i]=np.median(v);exmean[i]=np.mean(v)
        if len(v)>=2:exstd[i]=np.std(v);exiqr[i]=np.percentile(v,75)-np.percentile(v,25)
    new['_V2182_PT_EXMETA_MEDIAN_EDGE_POINTS']=exmed-market
    new['_V2182_PT_EXMETA_MEAN_EDGE_POINTS']=exmean-market
    new['_V2182_PT_EXMETA_STD']=exstd
    new['_V2182_PT_EXMETA_IQR']=exiqr
    eclnames=[j for j,k in enumerate(clnames) if k not in excluded]
    ecl=clmat[:,eclnames]
    ecn=np.isfinite(ecl).sum(axis=1).astype(float)
    eedge=ecl-market[:,None]
    new['_V2182_PT_EXMETA_CLUSTER_COUNT']=ecn
    new['_V2182_PT_EXMETA_CLUSTER_OPP_AGREE_FRAC']=np.divide(np.sum(np.isfinite(eedge)&(eedge<0),axis=1),ecn,out=np.full(n,np.nan),where=ecn>0)
    new['_V2182_PT_EXMETA_CLUSTER_TEAM_AGREE_FRAC']=np.divide(np.sum(np.isfinite(eedge)&(eedge>0),axis=1),ecn,out=np.full(n,np.nan),where=ecn>0)
    emed=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan for v in ecl])
    new['_V2182_PT_EXMETA_CLUSTER_MEDIAN_EDGE_POINTS']=emed-market
    # Aggregate rules have a frozen >=8-source gate. Do not publish their
    # atom columns if zero games satisfy that gate: diagnostics must continue
    # reporting them as MISSING_LIVE_ATOMS rather than falsely evaluable.
    if not (count>=8).any():
        new={k:v for k,v in new.items() if not (k.startswith('_V218_PTIDX_') and
             k not in {f'_V218_PTIDX_{re.sub("[^A-Z0-9]+","_",s.upper())}_{kind}'
                       for s in ids for kind in ('EDGE_POINTS','MARGIN_TEAM')})}
    if not (cl_count>=8).any():
        new={k:v for k,v in new.items() if not k.startswith('_V218_PTIDX_CLUSTER_')}
    if not (excount>=8).any():
        new={k:v for k,v in new.items() if not k.startswith('_V2182_PT_EXMETA_')}
    elif not (ecn>=8).any():
        new={k:v for k,v in new.items() if not k.startswith('_V2182_PT_EXMETA_CLUSTER_')}
    if new:
        # A single join avoids fragmenting the large Miner feature frame.
        out=pd.concat([out.drop(columns=list(new),errors='ignore'),pd.DataFrame(new,index=out.index)],axis=1)
    return out,{'status':'PASS','matched_games':matched,'source_indexes':len(cols),
                'usable_rows':int((count>0).sum()),'restored_atoms':len(new), 'pairs_found':pairs_found, 'missing_game_date_matches':missing_date_matches, 'rejected_missing_open':no_open, 'snapshot_utc':snap.isoformat(),
                'index_8plus_games':int((count>=8).sum()),
                'exmeta_8plus_games':int((excount>=8).sum()),
                'source':'GCS_PT_CURRENT_WITH_SEASON_DATE_TIMESTAMP'}
