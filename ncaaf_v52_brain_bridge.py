"""V5.2: conservative bridge from existing NCAAF system data to V4 game-level edges.

No rules are discovered here. Nothing in this module grants betting authority.
Source-level provenance is *independently verified* only when a captured original GCS
object was actually created before kickoff and its exact bytes match the declared SHA256.
Rebuilt views and retrospective game features remain research context, not pregame votes.
"""
from __future__ import annotations
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from io import StringIO, BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = 'NCAAF_V52_BRAIN_BRIDGE_20261020_V1'
FAMILIES = ('BIG_AL', 'PATHI', 'PREDICTION_TRACKER', 'SYSTEM_MINER', 'STATE_BOUNCE_BACK')
KEY = ('Season', 'Game_ID')
IDENTIFIER = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
FQ_TABLE = re.compile(r'^[A-Za-z0-9_-]+\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$')


def namekey(x: Any) -> str:
    return re.sub(r'[^a-z0-9]', '', str(x).lower())


def _ts(x):
    return pd.to_datetime(x, errors='coerce', utc=True)


def _string(x):
    return '' if x is None or pd.isna(x) else str(x).strip()


def _valid_hash(x):
    return bool(re.fullmatch('[a-fA-F0-9]{64}', _string(x)))


def normalize_records(raw: pd.DataFrame, spec: dict, edges: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Map one explicit source schema; reject uncertain game or side matches.

    spec: {table, format: long|wide, mapping: {semantic_name: source_col},
           family: optional constant, system_columns: {source_column: family} for wide}
    Long records use a stated signal Side (TEAM/OPPONENT), or a selected source
    team name that is matched to the V4 Team/Opponent. Wide flags need side_team.
    """
    source = str(spec.get('table', 'LOCAL_FILE'))
    fmt = str(spec.get('format', 'long')).lower()
    mp = spec.get('mapping') or {}
    missing_cols = sorted(set(mp.values()) - set(raw.columns))
    if missing_cols:
        raise ValueError(f'V52_SOURCE_MAPPING_MISSING_COLUMNS {source}: {missing_cols}')
    e = edges[['Season','Game_ID','Game_Date','Team','Opponent']].drop_duplicates(['Season','Game_ID']).copy()
    e['Season'] = pd.to_numeric(e.Season, errors='raise').astype(int)
    e['Game_ID'] = e.Game_ID.astype(str).str.strip()
    e['Date_Key'] = pd.to_datetime(e.Game_Date, utc=True).dt.strftime('%Y-%m-%d')
    e['T_Key'] = e.Team.map(namekey)
    e['O_Key'] = e.Opponent.map(namekey)
    if e.duplicated(['Season','Game_ID']).any():
        raise ValueError('V52_EDGE_LEDGER_NONUNIQUE_GAMES')

    mapped = pd.DataFrame(index=raw.index)
    for semantic, col in mp.items():
        mapped[semantic] = raw[col]
    if 'family' not in mapped and spec.get('family'):
        mapped['family'] = str(spec['family']).upper()
    if 'season' not in mapped:
        raise ValueError('V52_SOURCE_MISSING_SEASON')
    mapped['season'] = pd.to_numeric(mapped['season'], errors='coerce')
    mapped = mapped[mapped.season.isin(e.Season.unique())].copy()
    mapped['season'] = mapped.season.astype(int)
    mapped['source_table'] = source

    if fmt == 'wide':
        systems = spec.get('system_columns') or {}
        if not systems: raise ValueError('V52_WIDE_SOURCE_REQUIRES_EXPLICIT_SYSTEM_COLUMNS')
        chunks = []
        for col, family in systems.items():
            if col not in raw.columns: raise ValueError('V52_WIDE_MISSING_FLAG:'+col)
            if family not in FAMILIES: raise ValueError('V52_UNKNOWN_FAMILY:'+str(family))
            active = pd.to_numeric(raw.loc[mapped.index, col], errors='coerce').eq(1)
            if active.any():
                part = mapped.loc[active].copy()
                part['system_id'] = col
                part['family'] = family
                chunks.append(part)
        mapped = pd.concat(chunks, ignore_index=True) if chunks else mapped.iloc[:0].copy()
    elif fmt == 'long':
        if 'system_id' not in mapped or 'family' not in mapped:
            raise ValueError('V52_LONG_SOURCE_REQUIRES_FAMILY_AND_SYSTEM_ID')
    else:
        raise ValueError('V52_UNKNOWN_SOURCE_FORMAT')
    if len(mapped) == 0:
        return pd.DataFrame(), {'source':source,'status':'NO_TRIGGER_ROWS','input_rows':len(raw),'active_records':0}
    mapped['family'] = mapped.family.astype(str).str.upper().str.strip()
    if (~mapped.family.isin(FAMILIES)).any():
        raise ValueError('V52_UNKNOWN_FAMILY_IN_RECORDS')
    mapped['system_id'] = mapped.system_id.astype(str).str.strip()
    mapped = mapped[~mapped.system_id.isin(['', 'nan', 'None'])].copy()
    mapped['Season'] = mapped.season
    if 'game_id' in mapped:
        mapped['Game_ID'] = mapped.game_id.astype(str).str.strip()
        mapped = mapped.merge(e, on=['Season','Game_ID'],how='left',validate='many_to_one',indicator=True)
    else:
        if 'game_date' not in mapped or not {'team','opponent'}.issubset(mapped):
            raise ValueError('V52_TEAM_MATCH_REQUIRES_GAME_DATE_TEAM_OPPONENT')
        mapped['Date_Key'] = _ts(mapped.game_date).dt.strftime('%Y-%m-%d')
        mapped['T_Key'] = mapped.team.map(namekey)
        mapped['O_Key'] = mapped.opponent.map(namekey)
        # A timestamp day can differ from the local game day; do not guess.
        # Match only exact date plus unordered team pair, and reject ambiguous collisions.
        mapped['Pair_Key'] = mapped.apply(lambda r: ':'.join(sorted([r.T_Key,r.O_Key])),axis=1)
        e['Pair_Key'] = e.apply(lambda r: ':'.join(sorted([r.T_Key,r.O_Key])),axis=1)
        amb = e.duplicated(['Season','Date_Key','Pair_Key'],keep=False)
        if amb.any():
            e=e.loc[~amb].copy()
        mapped = mapped.merge(e.drop(columns=['T_Key','O_Key']),on=['Season','Date_Key','Pair_Key'],how='left',validate='many_to_one',indicator=True)
    rejected_match=int((mapped._merge!='both').sum())
    mapped=mapped[mapped._merge=='both'].copy().drop(columns=['_merge'])
    if not len(mapped):
        return pd.DataFrame(),{'source':source,'status':'ZERO_GAME_MATCH','input_rows':len(raw), 'active_records':0,'rejected_matches':rejected_match}
    if 'side' in mapped:
        mapped['Side'] = mapped.side.astype(str).str.upper().str.strip()
    elif 'side_team' in mapped:
        s = mapped.side_team.map(namekey)
        mapped['Side'] = np.where(s==mapped.Team.map(namekey),'TEAM',np.where(s==mapped.Opponent.map(namekey),'OPPONENT','INVALID'))
    else:
        raise ValueError('V52_SOURCE_NEEDS_SIDE_OR_SIDE_TEAM')
    rejected_side=int((~mapped.Side.isin(('TEAM','OPPONENT'))).sum())
    mapped=mapped[mapped.Side.isin(('TEAM','OPPONENT'))].copy()
    if 'team' in mapped and 'opponent' in mapped:
        pair_match = (mapped.team.map(namekey)==mapped.Team.map(namekey)) & (mapped.opponent.map(namekey)==mapped.Opponent.map(namekey))
        pair_swap = (mapped.team.map(namekey)==mapped.Opponent.map(namekey)) & (mapped.opponent.map(namekey)==mapped.Team.map(namekey))
        # A source may store the opposite orientation; its stated side_team was resolved above.
        ok=pair_match | pair_swap
        rejected_side+=int((~ok).sum());mapped=mapped[ok].copy()
    for c in ('snapshot_utc','kickoff_utc','rule_declared_utc','source_gcs_uri','source_artifact_sha256'):
        if c not in mapped: mapped[c] = None
    for c in ('snapshot_utc','kickoff_utc','rule_declared_utc'):
        mapped[c] = _ts(mapped[c])
    mapped['time_claim_valid'] = (mapped.snapshot_utc.notna() & mapped.kickoff_utc.notna() &
          (mapped.snapshot_utc<mapped.kickoff_utc) & (mapped.rule_declared_utc.notna()) &
          (mapped.rule_declared_utc<mapped.kickoff_utc))
    mapped['preseason_declared'] = mapped.time_claim_valid & (mapped.rule_declared_utc < pd.to_datetime(mapped.Season.astype(str)+'-01-01',utc=True))
    mapped['source_hash_valid'] = mapped.source_artifact_sha256.map(_valid_hash)
    mapped['vintage_class'] = np.where(mapped.preseason_declared & mapped.source_hash_valid,
       'TIMESTAMP_CLAIM_ONLY', 'RETROSPECTIVE_OR_UNVERIFIED')
    mapped['archive_verified'] = False
    result=mapped[['Season','Game_ID','family','system_id','Side','source_table','snapshot_utc',
                   'kickoff_utc','rule_declared_utc','source_gcs_uri','source_artifact_sha256',
                   'time_claim_valid','preseason_declared','source_hash_valid','vintage_class',
                   'archive_verified']].rename(columns={'family':'Family','system_id':'System_ID'}).copy()
    # A system may have several historical quotes/snapshots. Retain a single
    # *earliest* eligible signal, do not use the later result to choose a side.
    result = result.sort_values(['Season','Game_ID','Family','System_ID','snapshot_utc'],na_position='last')
    contradictions=result.groupby(['Season','Game_ID','Family','System_ID']).Side.nunique()
    conflicting_keys=contradictions[contradictions>1].index
    if len(conflicting_keys):
        bad_idx=pd.MultiIndex.from_frame(result[['Season','Game_ID','Family','System_ID']]).isin(conflicting_keys)
        result=result[~bad_idx].copy()
    result=result.drop_duplicates(['Season','Game_ID','Family','System_ID'],keep='first')
    audit={'source':source,'status':'CANDIDATES_CONTEXT_ONLY_UNTIL_ARCHIVE_VERIFIED',
           'input_rows':int(len(raw)),'matched_active_records':len(result),
           'rejected_game_matches':rejected_match,'rejected_side_or_pair':rejected_side,
           'conflicting_system_games':len(conflicting_keys),
           'preseason_timestamp_claims':int(result.preseason_declared.sum()),
           'independent_archive_proof':0,
           'family_counts':{x:int((result.Family==x).sum()) for x in FAMILIES}}
    return result,audit


def verify_gcs_originals(candidate: pd.DataFrame, storage_client, *, max_bytes=8_000_000, max_objects=200):
    """Validate a truly *original* pregame GCS CSV file, including row membership.

    A hash of an old unrelated blob is not evidence that a system fired. Thus
    original CSV bytes must contain the exact Season/Game_ID/Family/System_ID/
    Side/Snapshot_UTC/Kickoff_UTC/Rule_Declared_UTC record as well as match SHA256.
    Unknown original layouts fail closed and can be mapped explicitly in a
    later audited converter; they must not receive fake archive proof.
    """
    out=candidate.copy()
    out['archive_verified']=False
    verification=[]
    if out.empty:return out,verification
    keys=out[['source_gcs_uri','source_artifact_sha256']].fillna('').drop_duplicates()
    if len(keys)>max_objects:
        return out,[{'status':'TOO_MANY_OBJECTS_FOR_AUTOMATIC_VERIFICATION','objects':int(len(keys))}]
    required={'Season','Game_ID','Family','System_ID','Side','Snapshot_UTC','Kickoff_UTC','Rule_Declared_UTC'}
    for r in keys.itertuples(index=False):
        uri=_string(r.source_gcs_uri);declared=_string(r.source_artifact_sha256).lower()
        if not uri.startswith('gs://') or not _valid_hash(declared):continue
        try:
            prefix=uri[5:];bucket_name,obj=prefix.split('/',1)
            if bucket_name!='sharp-models':
                verification.append({'uri':uri,'status':'BUCKET_NOT_ALLOWLISTED'});continue
            blob=storage_client.bucket(bucket_name).blob(obj)
            blob.reload()
            length=int(blob.size or 0)
            if length>max_bytes or length<1:
                verification.append({'uri':uri,'status':'OBJECT_SIZE_NOT_VERIFIABLE','bytes':length});continue
            payload=blob.download_as_bytes()
            actual=hashlib.sha256(payload).hexdigest()
            created=pd.Timestamp(blob.time_created)
            created=created.tz_localize('UTC') if created.tzinfo is None else created.tz_convert('UTC')
            match=(out.source_gcs_uri.fillna('')==uri)&(out.source_artifact_sha256.fillna('').str.lower()==declared)
            if actual!=declared:
                verification.append({'uri':uri,'status':'HASH_MISMATCH','matching_claim_rows':int(match.sum())});continue
            try:
                archived=pd.read_csv(BytesIO(payload),dtype=str)
            except Exception:
                verification.append({'uri':uri,'status':'UNPARSABLE_ORIGINAL_RECORD'});continue
            if not required.issubset(archived.columns):
                verification.append({'uri':uri,'status':'ARCHIVED_RECORD_SCHEMA_UNSUPPORTED',
                    'missing':sorted(required-set(archived.columns))});continue
            ar=archived.copy()
            ar['Season']=pd.to_numeric(ar.Season,errors='coerce')
            for name in ('Game_ID','Family','System_ID','Side'):
                ar[name]=ar[name].astype(str).str.strip()
            for name in ('Snapshot_UTC','Kickoff_UTC','Rule_Declared_UTC'):
                ar[name]=pd.to_datetime(ar[name],utc=True,errors='coerce')
            verified_count=0
            for i,v in out.loc[match].iterrows():
                if not v.preseason_declared or not v.time_claim_valid or not created<v.kickoff_utc:
                    continue
                found=(ar.Season.eq(int(v.Season)) & ar.Game_ID.eq(str(v.Game_ID)) &
                       ar.Family.eq(str(v.Family)) & ar.System_ID.eq(str(v.System_ID)) &
                       ar.Side.eq(str(v.Side)) & ar.Snapshot_UTC.eq(v.snapshot_utc) &
                       ar.Kickoff_UTC.eq(v.kickoff_utc) & ar.Rule_Declared_UTC.eq(v.rule_declared_utc))
                if found.any():
                    out.at[i,'archive_verified']=True;verified_count+=1
            verification.append({'uri':uri,'status':'ORIGINAL_ARCHIVE_ROWS_CHECKED',
                'gcs_created_utc':created.isoformat(),'hash_match':True,
                'matching_claim_rows':int(match.sum()),'verified_pregame_rows':verified_count})
        except Exception as ex:
            verification.append({'uri':uri,'status':'VERIFICATION_ERROR','message':str(ex)[:240]})
    out['vintage_class']=np.where(out.archive_verified,'ARCHIVED_PREGAME_VERIFIED',out.vintage_class)
    return out,verification


def attribute(edges: pd.DataFrame, candidates: pd.DataFrame):
    """Research-only matched cohort audit; system flags do not automatically vote.
    Independent confirmations require independent GCS archive verification.
    Nonverified contexts are catalogued separately, never treated as failures.
    """
    e=edges.copy()
    needed={'Season','Game_ID','Model','Side','Result','Edge_Bin','Spread_Bin','Selection_Type'}
    if needed-set(e):raise ValueError('V52_EDGE_LEDGER_MISSING:'+str(sorted(needed-set(e))))
    if e.duplicated(['Season','Game_ID','Model']).any():raise ValueError('V52_DUPLICATE_MODEL_GAME')
    if len(candidates):
        c=candidates.copy()
        valid=c.archive_verified.astype(bool)&c.preseason_declared.astype(bool)&c.time_claim_valid.astype(bool)
        eligible=c[valid].copy()
    else:
        c=pd.DataFrame(columns=['Season','Game_ID','Family','System_ID','Side','archive_verified'])
        eligible=c.copy()
    for fam in FAMILIES:e[f'Brain_{fam}']='MISSING'
    system_out=[]
    for family in FAMILIES:
        sub=eligible[eligible.Family==family]
        if len(sub):
            for (season,gid),sides in sub.groupby(['Season','Game_ID']).Side:
                uniq=set(sides)
                relation='CONFLICT' if len(uniq)>1 else ('CONFIRM' if next(iter(uniq))=='TEAM' else 'OPPOSE')
                idx=(e.Season==season)&(e.Game_ID.astype(str)==str(gid))
                if relation=='CONFLICT':e.loc[idx,f'Brain_{family}']='CONFLICT'
                else:e.loc[idx,f'Brain_{family}']=np.where(e.loc[idx,'Side']==next(iter(uniq)),'CONFIRM','OPPOSE')
        for (model,period),g in e.groupby(['Model','Period'],observed=True):
            status_col=f'Brain_{family}'
            stats={'Family':family,'Model':model,'Period':period,'verified_system_games':len(sub),
              'confirm':_wl(g[g[status_col]=='CONFIRM']),
              'oppose':_wl(g[g[status_col]=='OPPOSE']),
              'conflict':_wl(g[g[status_col]=='CONFLICT']),
              'missing':_wl(g[g[status_col]=='MISSING'])}
            system_out.append(stats)
    system_details=[]
    for (family,sys_id), rows in eligible.groupby(['Family','System_ID'], observed=True):
        merged=e.merge(rows[['Season','Game_ID','Side']].rename(columns={'Side':'Signal_Side'}),
                       on=['Season','Game_ID'],how='inner',validate='many_to_one')
        merged['Relation']=np.where(merged.Side==merged.Signal_Side,'CONFIRM','OPPOSE')
        for (model,period),group in merged.groupby(['Model','Period'],observed=True):
            system_details.append({'Family':family,'System_ID':str(sys_id),'Model':model,'Period':period,
                'confirm':_wl(group[group.Relation=='CONFIRM']),
                'oppose':_wl(group[group.Relation=='OPPOSE']),
                'authority':'NONE__RESEARCH_ONLY'})
    # Separate retrospective pattern research from *verified* pregame signals.
    # This supports hypothesis discovery but is NEVER part of bet authority or
    # the "verified" confirmation statistics above.
    retrospective_systems=[]
    for (family,sys_id),rows in c.groupby(['Family','System_ID'],observed=True):
        if not len(rows):continue
        merged=e.merge(rows[['Season','Game_ID','Side','archive_verified']].rename(columns={'Side':'Context_Side'}),
                       on=['Season','Game_ID'],how='inner',validate='many_to_one')
        merged['Context_Relation']=np.where(merged.Side==merged.Context_Side,'CONFIRM','OPPOSE')
        for (model,period),group in merged.groupby(['Model','Period'],observed=True):
            retrospective_systems.append({
               'Family':family,'System_ID':str(sys_id),'Model':model,'Period':period,
               'context_rows':len(group),
               'archive_verified_n':int(group.archive_verified.sum()),
               'confirm':_wl(group[group.Context_Relation=='CONFIRM']),
               'oppose':_wl(group[group.Context_Relation=='OPPOSE']),
               'status':'DESCRIPTIVE_ONLY__NOT_HISTORICAL_VOTES',
               'warning':'Retrospective systems may have been discovered using these very outcomes; no edge inference or authority.'
            })
    context={f:{'raw_candidate_records':int((c.Family==f).sum()),
      'independently_verified_records':int((eligible.Family==f).sum())} for f in FAMILIES}
    total_valid=int(len(eligible))
    report={'version':VERSION,'status':'ATTRIBUTION_RESEARCH_ONLY' if total_valid else 'BLOCKED_NO_INDEPENDENTLY_VERIFIED_BRAIN_HISTORY',
       'games':int(e[['Season','Game_ID']].drop_duplicates().shape[0]),
       'edge_model_rows':len(e),'brain_candidates':len(c),'verified_system_game_records':total_valid,
       'family_coverage':context,'family_attribution':system_out,'system_attribution':system_details,
       'retrospective_context_system_research':retrospective_systems,
       'provenance_warning':'Claimed snapshot timestamps, restored feature views, and retrospective mined rules are not evidence of pregame rule existence.',
       'source_vintage_certified':False,'roi':None,'bet_authority':0,'production_modified':False}
    return report,e


def _wl(part: pd.DataFrame):
    w=int((part.Result=='WIN').sum());l=int((part.Result=='LOSS').sum());p=int((part.Result=='PUSH').sum())
    return {'n':len(part),'wins':w,'losses':l,'pushes':p,'hit_rate':w/(w+l) if w+l else None}


def fingerprint(raw: pd.DataFrame):
    return hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest()
