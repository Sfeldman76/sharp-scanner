"""Independent CFBD raw source archive. Never mutates existing NCAAF data tables.

By default this module only inventories/plans; network retrieval is an EXPLICIT CLI action.
The source payload is immutable and hashes and retrieval times travel with the bytes.
API retrieval time is NOT a historical availability timestamp for the source facts.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

BASE_URL = 'https://api.collegefootballdata.com'
# Info validated against current /api/games, /api/drives, /api/plays, /api/stats docs.
ENDPOINTS = {
    'games': {'path': '/games', 'grain': 'physical_game', 'min_year': 1869, 'parameter': 'season'},
    'team_box': {'path': '/games/teams', 'grain': 'game_team_stat', 'min_year': 2004, 'parameter': 'week'},
    'player_box': {'path': '/games/players', 'grain': 'game_player_stat', 'min_year': 2004, 'parameter': 'week'},
    'drives': {'path': '/drives', 'grain': 'game_possession', 'min_year': 2001, 'parameter': 'season'},
    'plays': {'path': '/plays', 'grain': 'game_play', 'min_year': 2001, 'parameter': 'week'},
    'advanced': {'path': '/stats/game/advanced', 'grain': 'game_team', 'min_year': 2001, 'parameter': 'season'},
    'havoc': {'path': '/stats/game/havoc', 'grain': 'game_team', 'min_year': 2004, 'parameter': 'season'},
}
RAW_FAMILIES = tuple(ENDPOINTS)


def stable_payload_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False)+'\n').encode('utf-8')


def request_key(family: str, params: dict) -> str:
    if family not in ENDPOINTS:
        raise ValueError('Unknown family: '+family)
    raw = stable_payload_bytes({'family': family, 'params': params})
    return hashlib.sha256(raw).hexdigest()[:20]


def archive_response(root: Path | str, family: str, params: dict, payload: Any,
                     retrieved_at: str | None = None) -> dict:
    """Never replace prior source evidence; a revision gets a new content-addressed file."""
    if family not in ENDPOINTS:
        raise ValueError('Invalid source family')
    if not isinstance(payload, list):
        raise ValueError('Expected JSON array from CFBD endpoint')
    root = Path(root)
    sha = hashlib.sha256(stable_payload_bytes(payload)).hexdigest()
    key = request_key(family, params)
    target = root / family / f'{key}_{sha[:16]}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    content = stable_payload_bytes(payload)
    if target.exists():
        if target.read_bytes() != content:
            raise AssertionError('Immutable source file collision')
        existed = True
    else:
        target.write_bytes(content)
        existed = False
    meta = target.with_suffix('.meta.json')
    if not meta.exists():
        stamp = retrieved_at or datetime.now(timezone.utc).isoformat()
        meta.write_text(json.dumps({
            'family': family, 'endpoint': ENDPOINTS[family]['path'],
            'grain': ENDPOINTS[family]['grain'], 'parameters': params,
            'request_key': key, 'payload_sha256': sha, 'row_count': len(payload),
            'retrieved_at': stamp,
            'historical_pre_kickoff_vintage_verified': False,
            'data_asof_rule': 'completed_game_outcome_only_before_future_kickoff',
        }, indent=2) + '\n')
    return {'path': str(target), 'rows': len(payload), 'sha256': sha,
            'previously_archived': existed}


def _stored_request(root: Path, family: str, params: dict) -> bool:
    return bool(list((root / family).glob(request_key(family, params)+'_*.json')))


def derive_weeks(game_rows: Iterable[dict], year: int) -> list[tuple[str, int]]:
    pairs = set()
    for g in game_rows:
        if int(g.get('season') or 0) != int(year):
            continue
        w = g.get('week')
        if w is None:
            continue
        kind = str(g.get('seasonType') or 'regular').lower()
        if kind not in ('regular','postseason'):
            continue
        pairs.add((kind, int(w)))
    return sorted(pairs)


def request_plan(root: Path | str, years: Iterable[int], families: Iterable[str],
                 games_by_year: dict[int, list[dict]] | None = None) -> list[dict]:
    root = Path(root)
    selected = list(families)
    for family in selected:
        if family not in ENDPOINTS:
            raise ValueError('Unknown family: '+family)
    plans=[]
    for year in sorted(set(int(y) for y in years)):
        for family in selected:
            spec=ENDPOINTS[family]
            if year < spec['min_year']:
                continue
            if spec['parameter']=='season':
                configs=[{'year':year,'seasonType':'both'}]
            else:
                weeks=derive_weeks((games_by_year or {}).get(year,[]),year)
                if not weeks:
                    plans.append({'family':family,'year':year,'status':'NEEDS_GAMES_FIRST',
                                  'reason':'Week-constrained API; derive actual weeks from the games source'})
                    continue
                configs=[{'year':year,'seasonType':kind,'week':week} for kind,week in weeks]
            for params in configs:
                plans.append({'family':family,'year':year,'endpoint':spec['path'],
                              'parameters':params, 'status':'CACHED' if _stored_request(root,family,params) else 'MISSING'})
    return plans


def inventory_archive(root: Path | str, years: Iterable[int] | None=None) -> dict:
    root=Path(root)
    filtered=set(map(int,years)) if years is not None else None
    out={'families': {},'files':0,'rows':0,'issues':[],'season_family_rows':{}}
    for family in RAW_FAMILIES:
        records=[]
        for path in sorted((root/family).glob('*.meta.json')):
            try:
                meta=json.loads(path.read_text())
                yr=int(meta['parameters']['year'])
                if filtered is not None and yr not in filtered: continue
                raw=Path(str(path)[:-len('.meta.json')]+'.json')
                if not raw.exists():
                    out['issues'].append(f'NO_PAYLOAD:{path}')
                    continue
                digest=hashlib.sha256(raw.read_bytes()).hexdigest()
                if digest != meta['payload_sha256']:
                    out['issues'].append(f'HASH_MISMATCH:{raw}')
                    continue
                records.append(meta)
                skey=str(yr)
                out['season_family_rows'].setdefault(skey,{}).setdefault(family,0)
                out['season_family_rows'][skey][family]+=meta['row_count']
            except (OSError,ValueError,KeyError,TypeError) as exc:
                out['issues'].append(f'INVALID_META:{path}:{exc}')
        out['families'][family]={'source_files':len(records),
                                 'source_rows':sum(x['row_count'] for x in records),
                                 'years':sorted({int(x['parameters']['year']) for x in records})}
        out['files']+=len(records); out['rows']+=sum(x['row_count'] for x in records)
    return out


def load_family(root:Path|str, family:str, years:Iterable[int]|None=None) -> list[dict]:
    """Read cached raw API data; de-duplicate *same request*, NEVER cross-source keys."""
    if family not in ENDPOINTS: raise ValueError('unknown family')
    root=Path(root); scope=set(map(int,years)) if years is not None else None
    files_by_request={}
    for meta_path in (root/family).glob('*.meta.json'):
        m=json.loads(meta_path.read_text())
        if scope is not None and int(m['parameters']['year']) not in scope: continue
        files_by_request.setdefault(m['request_key'],[]).append(meta_path)
    result=[]
    for key, paths in sorted(files_by_request.items()):
        # Multiple versions of same request reflect corrections; refuse silent mixing.
        fingerprints={json.loads(p.read_text())['payload_sha256'] for p in paths}
        if len(fingerprints)>1:
            raise ValueError(f'Conflicting snapshots for {family} request {key}; explicit vintage required')
        p=paths[0]; m=json.loads(p.read_text()); data=Path(str(p)[:-len('.meta.json')]+'.json').read_bytes()
        if hashlib.sha256(data).hexdigest()!=m['payload_sha256']:
            raise ValueError(f'Broken raw archive checksum: {p}')
        result.extend(json.loads(data))
    return result


def fetch_plan(root:Path|str, plans:list[dict], *, api_key:str, max_requests:int=1) -> list[dict]:
    """Explicit, limited retrieval; never invoked by building/models/reporting."""
    if not api_key: raise ValueError('CFBD_API_KEY is required to fetch sources')
    import requests
    pending=[p for p in plans if p['status']=='MISSING']
    if len(pending)>max_requests:
        raise ValueError(f'{len(pending)} requests are pending. --max-requests={max_requests} would be exceeded; no calls sent.')
    outcome=[]
    for p in pending:
        r=requests.get(BASE_URL+p['endpoint'], params=p['parameters'],
                       headers={'Authorization':'Bearer '+api_key},timeout=70)
        r.raise_for_status()
        outcome.append(archive_response(root,p['family'],p['parameters'],r.json()))
    return outcome
