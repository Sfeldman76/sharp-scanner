#!/usr/bin/env python3
"""Research-only coordinator for the existing CFBD archive preview loader.

Does not collect from the CFBD API, change the legacy CFBD feature warehouse, or
change frozen NCAAF Production V1. Builds annual previews with the EXISTING
ncaaf_cfbd_warehouse_loader_v2.py, then optionally publishes an immutable
BigQuery build and separate historical/shadow views.

Usage (Cloud Shell):
  python3 ncaaf_cfbd_archive_research_build_v1.py --resume --publish
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PROJECT = 'sharplogger'
DATASET = 'sharp_data'
LEGACY = f'{PROJECT}.{DATASET}.ncaaf_cfbd_miner_features_v1'
RESEARCH_VIEW = f'{PROJECT}.{DATASET}.ncaaf_cfbd_archive_research_v1'
SHADOW_VIEW = f'{PROJECT}.{DATASET}.ncaaf_cfbd_archive_shadow_2026_v1'


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1024*1024), b''):
            h.update(buf)
    return h.hexdigest()


def validate_year(df, year: int, quality: dict) -> dict:
    import pandas as pd
    keys = ['Season', 'Game_ID', 'Team_Key']
    if not set(keys).issubset(df.columns):
        raise RuntimeError(f'{year}: missing keys {set(keys)-set(df.columns)}')
    if df[keys].isna().any().any():
        raise RuntimeError(f'{year}: empty keys')
    if set(pd.to_numeric(df['Season'],errors='raise').astype(int)) != {year}:
        raise RuntimeError(f'{year}: contaminated by other seasons')
    if df.duplicated(keys).any():
        raise RuntimeError(f'{year}: duplicate game/team keys')
    side_count=df.groupby(['Season','Game_ID'])['Team_Key'].nunique()
    if side_count.empty or not side_count.eq(2).all() or len(df)!=2*len(side_count):
        raise RuntimeError(f'{year}: game-side parity failed')
    if quality.get('status')!='OFFLINE_PREVIEW_ONLY_NOT_CERTIFIED_FOR_BQ_PROMOTION':
        raise RuntimeError(f'{year}: unrecognized source quality status')
    if quality.get('rows')!=len(df):
        raise RuntimeError(f'{year}: CSV and report disagree on row counts')
    advanced=int(df['CFBD_Season_Net_PPA_Prior'].notna().sum())
    if year<=2025 and not advanced:
        raise RuntimeError(f'{year}: no prior-game advanced CFBD observations; research coverage not verified')
    family_quality=(quality.get('season_quality') or [{}])[0]
    return {'season':year, 'games':int(len(side_count)), 'rows':int(len(df)),
            'advanced_prior':advanced,
            'missing_completed_fbs_family':family_quality.get('missing_completed_fbs_family',{}),
            'current_game_observations_are_NOT_model_features':True,
            'source_vintage_certified':False}


def build_previews(args, outdir: Path, loader: Path, archive: Path):
    import pandas as pd
    manifest=archive/'manifest.jsonl'
    if not manifest.is_file():
        raise RuntimeError(f'Archive manifest missing: {manifest}')
    fingerprint = file_sha(loader) + ':' + file_sha(Path(__file__)) + ':' + file_sha(manifest)
    reports=[]
    for year in range(args.start, args.end+1):
        csv=outdir/f'cfbd_{year}_archive_research.csv'
        quality=csv.with_suffix('.quality.json')
        stamp=outdir/f'cfbd_{year}_archive_research.fingerprint'
        reusable=(args.resume and csv.is_file() and quality.is_file() and
                  stamp.is_file() and stamp.read_text().strip()==fingerprint)
        if not reusable:
            for p in (stamp,):
                p.unlink(missing_ok=True)
            print(f'[CFBD-ARCHIVE-BUILD] season={year} building',flush=True)
            cmd=[sys.executable,str(loader),'--archive-preview',
                 '--archive-root',str(archive),'--seasons',str(year),
                 '--history-seasons','','--local-output',str(csv)]
            with (outdir/f'cfbd_{year}_archive_research.log').open('w') as log:
                proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            if proc.returncode:
                print((outdir/f'cfbd_{year}_archive_research.log').read_text()[-6000:])
                raise RuntimeError(f'{year}: existing archive preview builder failed; no BigQuery publication')
        df=pd.read_csv(csv,low_memory=False)
        report=json.loads(quality.read_text())
        check=validate_year(df,year,report)
        if not reusable:
            stamp.write_text(fingerprint+'\n')
        reports.append(check)
        print(f"[CFBD-ARCHIVE-BUILD] season={year} status=PASS games={check['games']} rows={check['rows']} advanced={check['advanced_prior']}",flush=True)
        del df
    (outdir/'archive_research_build_audit.json').write_text(json.dumps({
        'status':'LOCAL_VALIDATED_NOT_PREGAME_VINTAGE_CERTIFIED',
        'loader_sha256':file_sha(loader),'manifest_sha256':file_sha(manifest),
        'seasons':reports,'legacy_table_modified':False,'production_modified':False,
        'warnings':['Current-game CFBD_Game_* fields exist ONLY as postgame source observations for strict pregame reconstruction.',
                    'Static roster/talent/returning release timestamps are not certified and should not enter model selection.',
                    '2026 is prospective/shadow only; no 2026 outcomes may be used for historical discovery or selection.',
                    'Historical opening-spread coverage begins in 2021; no earlier market lines are synthesized.',
                    'New research views contain only pregame-safe columns; raw observation build retains postgame data for audit.']
    },indent=2)+'\n')
    return reports


def _archive_safe_pregame_feature(c):
    """Same conservative contract enforced in patched CORE and Miner readers."""
    if not str(c).startswith("CFBD_"):
        return False
    x=str(c)[5:]
    if x.startswith("Opp_"):
        x=x[4:]
    if x in ("Games_Prior", "Conference_Games_Prior", "Conference_Game_Streak_Prior"):
        return True
    if x in (
        "Matchup_Net_PPA_Adv", "Matchup_Net_Success_Adv", "Matchup_Net_Explosiveness_Adv",
        "Matchup_EarlyDown_PPA_Adv", "Matchup_PPA_Per_Drive_Adv", "Matchup_Scoring_Opp_Rate_Adv"):
        return True
    if not x.endswith("_Prior"):
        return False
    if x.startswith(("Season_CFBD_Game_", "Recent3_CFBD_Game_", "Recent5_CFBD_Game_", "PrevSeason_CFBD_Game_")):
        return True
    if x.startswith(("Season_Off_", "Season_Def_", "Recent3_Off_", "Recent3_Def_",
                     "Recent5_Off_", "Recent5_Def_", "PrevSeason_Off_", "PrevSeason_Def_")):
        return True
    return x.startswith(("Season_Net_", "Recent3_Net_", "Recent5_Net_"))


def infer_archive_schema(outdir, reports, reference_schema, bigquery):
    """Preserve ALL archive columns; never silently truncate to old warehouse schema.

    Use old schema only for columns with unchanged names, not for field coverage.
    The preview CSVs themselves supply the independent feature universe.
    """
    import pandas as pd
    legacy={f.name:f for f in reference_schema}
    columns=[]
    for rec in reports:
        csv=outdir/f"cfbd_{rec['season']}_archive_research.csv"
        hdr=pd.read_csv(csv,nrows=0).columns.tolist()
        if len(hdr)!=len(set(hdr)):
            raise RuntimeError(f"Duplicate preview CSV columns: {csv}")
        for col in hdr:
            if col not in columns:
                columns.append(col)
    safe_cols=[c for c in columns if _archive_safe_pregame_feature(c)]
    if len(safe_cols)<10 or "CFBD_Season_Net_PPA_Prior" not in safe_cols:
        raise RuntimeError(f"Archive pregame gate unexpectedly sparse safe_count={len(safe_cols)}")
    extras=[c for c in columns if c not in legacy]
    # All years participate in inference. Restrict to a modest bounded sample
    # per season; conversion subsequently validates every non-null value.
    evidence={c:set() for c in extras}
    for rec in reports:
        csv=outdir/f"cfbd_{rec['season']}_archive_research.csv"
        sample=pd.read_csv(csv,nrows=3000,dtype=str,keep_default_na=False)
        for c in extras:
            if c not in sample: continue
            vals=sample[c].astype(str).str.strip()
            vals=vals.loc[~vals.str.lower().isin(("","nan","none","null","<na>"))]
            if vals.empty:continue
            if vals.str.lower().isin(("true","false")).all():
                evidence[c].add("bool")
            elif pd.to_numeric(vals,errors="coerce").notna().all():
                evidence[c].add("numeric")
            else:
                evidence[c].add("string")
    schema=[]
    for c in columns:
        if c in legacy:
            src=legacy[c]
            schema.append(bigquery.SchemaField(c,src.field_type,mode="NULLABLE"))
        else:
            e=evidence[c]
            if "string" in e or len(e)>1:
                typ="STRING"
            elif e=={"bool"}:
                typ="BOOLEAN"
            else:
                typ="FLOAT64" if c.startswith("CFBD_") else "STRING"
            schema.append(bigquery.SchemaField(c,typ,mode="NULLABLE"))
    return schema,safe_cols


def convert_columns(df, schema):
    import pandas as pd
    wanted=[f.name for f in schema]
    missing=set(wanted)-set(df.columns)
    if missing:
        for c in missing: df[c]=None  # A family unavailable in a specific season is NULL.
    out=df[wanted].copy()
    for field in schema:
        name=field.name;kind=field.field_type.upper()
        if kind in ('INTEGER','INT64','FLOAT','FLOAT64','NUMERIC','BIGNUMERIC'):
            source=out[name]
            numeric=pd.to_numeric(source,errors='coerce')
            bad=source.notna() & numeric.isna()
            if bad.any():
                raise RuntimeError(f"Archive numeric column {name}: {int(bad.sum())} unexpected nonnumeric values")
            if kind in ('INTEGER','INT64'):
                out[name]=numeric.astype('Int64')
            else:
                out[name]=numeric.astype('float64')
        elif kind=='TIMESTAMP':
            out[name]=pd.to_datetime(out[name],errors='raise',utc=True)
        elif kind=='DATETIME':
            out[name]=pd.to_datetime(out[name],errors='raise').dt.tz_localize(None)
        elif kind=='DATE':
            out[name]=pd.to_datetime(out[name],errors='raise').dt.date
        elif kind=='BOOLEAN':
            z=out[name].astype('string').str.strip().str.lower()
            bad=z.notna() & ~z.isin(('true','false','1','0'))
            if bad.any():raise RuntimeError(f"Unexpected BOOLEAN values in {name}")
            out[name]=z.map({'true':True,'false':False,'1':True,'0':False}).astype('boolean')
        elif kind=='STRING':
            out[name]=out[name].astype('string')
        else:
            raise RuntimeError(f'Unsupported schema field: {name} ({kind})')
    return out


def publish(args, outdir, reports):
    from google.cloud import bigquery
    from google.api_core.exceptions import NotFound
    import pandas as pd
    client=bigquery.Client(project=PROJECT)
    live=client.get_table(LEGACY) # Reference field TYPES, never reference feature VALUES.
    schema,safe_cols=infer_archive_schema(outdir,reports,live.schema,bigquery)
    cols=[f.name for f in schema]
    key_cols=[k for k in ('Season','Game_ID','Game_Date','Week','Team','Opponent','Team_Key','Opponent_Key','Is_Home') if k in cols]
    for required in ('Season','Game_ID','Team_Key','Opponent_Key','Is_Home'):
        if required not in key_cols:
            raise RuntimeError(f"Missing essential research key: {required}")
    view_cols=list(dict.fromkeys(key_cols+safe_cols))
    # Do not expose final-score, same-game CFBD_Game_* or uncertified release-time
    # personnel/talent columns through either view. They remain available ONLY in
    # the immutable raw-observation build, for explicit audited research.
    if any(c.startswith(('CFBD_Game_','CFBD_Opp_Game_')) for c in view_cols):
        raise RuntimeError('Unsafe game-outcome field in pregame view')
    build_id=args.build_id or datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    if not re.fullmatch(r'[A-Za-z0-9_]{6,50}',build_id):
        raise RuntimeError('Bad build-id (letters/digits/underscores only)')
    dest=f'{PROJECT}.{DATASET}.ncaaf_cfbd_archive_research_build_{build_id}'
    # All preflight checks happen before creating the first BigQuery asset.
    for name in (dest, RESEARCH_VIEW, SHADOW_VIEW):
        try:
            client.get_table(name)
        except NotFound:
            continue
        raise RuntimeError(f'Existing destination {name}; refusing to replace')
    from google.cloud.bigquery import LoadJobConfig
    total=0
    for i,rec in enumerate(reports):
        year=rec['season']
        csv=outdir/f'cfbd_{year}_archive_research.csv'
        data=pd.read_csv(csv,low_memory=False)
        validate_year(data,year,json.loads(csv.with_suffix('.quality.json').read_text()))
        frame=convert_columns(data,schema)
        conf=LoadJobConfig(schema=schema,
            write_disposition='WRITE_EMPTY' if i==0 else 'WRITE_APPEND',
            create_disposition='CREATE_IF_NEEDED')
        print(f'[CFBD-ARCHIVE-PUBLISH] year={year} rows={len(frame)} cols={len(schema)} immutable_table={dest}',flush=True)
        client.load_table_from_dataframe(frame,dest,job_config=conf).result()
        total+=len(frame)
        del data,frame
    table=client.get_table(dest)
    if table.num_rows!=total:
        raise RuntimeError(f'Readback row-count mismatch expected={total}, actual={table.num_rows}; views NOT created')
    check=client.query(f"SELECT Season, COUNT(*) AS n, COUNT(DISTINCT Game_ID) AS games FROM `{dest}` GROUP BY Season ORDER BY Season").result()
    actual={int(r.Season):(int(r.n),int(r.games)) for r in check}
    for rec in reports:
        if actual.get(rec['season']) != (rec['rows'],rec['games']):
            raise RuntimeError(f"Season {rec['season']} readback mismatch; views NOT created")
    table.description=('Full archive-v2 preview, immutable research observation source. '
                       'Raw contemporaneous CFBD_Game_* are POSTGAME observations, not pregame predictors. '
                       'Public historical research views expose strict prior-only features. Static vintage NOT certified.')
    table.labels={'research_only':'true','certified':'false','source':'archive_v2'}
    client.update_table(table,['description','labels'])
    if args.start!=2013 or args.end!=2026:
        print(f'[CFBD-ARCHIVE-PUBLISH] TABLE_ONLY: expected full 2013-2026 for stable views, got {args.start}-{args.end}')
        return dest
    columns_sql=', '.join('`'+c+'`' for c in view_cols)
    for name, query, desc in [
        (RESEARCH_VIEW,f'SELECT {columns_sql} FROM `{dest}` WHERE Season BETWEEN 2013 AND 2025',
         '2013-2025 research; strictly prior-game-derived features only; not production-certified.'),
        (SHADOW_VIEW,f'SELECT {columns_sql} FROM `{dest}` WHERE Season = 2026',
         '2026 shadow; strictly pregame features only; no outcomes or production authority.')]:
        v=bigquery.Table(name)
        v.view_query=query;v.description=desc
        v.labels={'research_only':'true','certified':'false','source':'archive_v2'}
        client.create_table(v)
        print(f'[CFBD-ARCHIVE-VIEW] {name} exposed_pregame_features={len(safe_cols)}',flush=True)
    print(f'[CFBD-ARCHIVE-DONE] rows={total} all_archive_columns={len(schema)} safe_view_features={len(safe_cols)} immutable_table={dest} legacy_unchanged=TRUE',flush=True)
    return dest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    home=Path.home()
    p.add_argument('--repo',type=Path,default=home/'sharp-scanner')
    p.add_argument('--archive-root',type=Path,default=home/'cfbd_rebuild/cfbd_archive_v2/cfbd_archive')
    p.add_argument('--output-dir',type=Path,default=home/'cfbd_rebuild/cfbd_archive_v2/archive_research_build')
    p.add_argument('--start',type=int,default=2013)
    p.add_argument('--end',type=int,default=2026)
    p.add_argument('--resume',action='store_true',help='Reuse year previews only if loader and archive manifest hashes match')
    p.add_argument('--publish',action='store_true',help='Explicit permission to WRITE only new immutable BQ research table and new research views')
    p.add_argument('--build-id',help='Optional unique suffix for immutable BigQuery build table')
    a=p.parse_args()
    if a.start<2013 or a.end>2026 or a.start>a.end:
        p.error('Supported build years: 2013..2026')
    outdir=a.output_dir.expanduser().resolve();outdir.mkdir(parents=True,exist_ok=True)
    loader=a.repo.expanduser().resolve()/'ncaaf_cfbd_warehouse_loader_v2.py'
    if not loader.is_file():raise RuntimeError(f'Missing patched loader {loader}')
    reports=build_previews(a,outdir,loader,a.archive_root.expanduser().resolve())
    print(f'[CFBD-ARCHIVE-LOCAL] status=PASS seasons={len(reports)} total_rows={sum(r["rows"] for r in reports)}',flush=True)
    if a.publish:publish(a,outdir,reports)
    else:print('[CFBD-ARCHIVE-LOCAL] BigQuery unchanged; pass --publish to create research-only build and views',flush=True)

if __name__=='__main__':main()
