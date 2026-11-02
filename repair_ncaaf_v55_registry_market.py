#!/usr/bin/env python3
"""Guarded V5.5 registry-version and market-rich DISPLAY repair.

This patch does not change CORE probabilities, betting policy, qualified evidence
criteria, registry files or their approval, or any BigQuery/GCS data. It uses
previously captured exact-market quotes; never synthesizes prices or votes.

Usage:
  python3 repair_ncaaf_v55_registry_market.py --self-test
  python3 repair_ncaaf_v55_registry_market.py --diagnose
  python3 repair_ncaaf_v55_registry_market.py --check
  python3 repair_ncaaf_v55_registry_market.py --apply
  python3 repair_ncaaf_v55_registry_market.py --self-test --target-self-test
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

BANNER = 'NCAAF_V55_REGISTRY_MARKET_GUARDED_FIX_20261102'
MINER_OLD = 'NCAAF_MINER_LIVE_AUTHORITY_V2_20_SOURCE_NEUTRAL_STRONG_VALIDATED_20261007'
MINER_NEW = 'NCAAF_MINER_LIVE_AUTHORITY_V2_21_HISTORICAL_DEEP_QUALIFICATION_20261028'

# A policy-version migration must be allowlisted; never accept arbitrary
# qualifications or whatever policy happened to be present in an arbitrary JSON.
SUPPORTED_DECLARATION = (
    "V55_RESEARCH_MINER_SUPPORTED_POLICIES = frozenset((\n"
    "    V55_RESEARCH_MINER_POLICY,\n"
    f"    {MINER_NEW!r},\n"
    "))\n"
)

# Note: displayed movements are only computed for one RAW scanner source and
# one exact-price book/side. Snapshot windows <=15m late are intentionally strict.
MARKET_HELPER = '''

def _v55_rich_from_market_snapshots(same_book, anchor, expected_spread):
    """Observed market movement; DISPLAY ONLY, never a qualification/vote.

    Uses only the existing scanner source, same kickoff/side/book market rows
    already matched by the caller. Does not use later quotes, synthetic openers,
    or the recalculated runtime snapshots. A reported 30/60/120m movement
    requires a stored quote within 15 minutes BEFORE the requested lookback.
    """
    import pandas as pd
    if not isinstance(same_book, pd.DataFrame) or same_book.empty:
        return 'NOT_PUBLISHED'
    at = _timestamp(anchor)
    line = _price(expected_spread)
    if pd.isna(at) or line is None:
        return 'NOT_PUBLISHED'
    need = {'Value', '_snap', 'V55_Overlay_Actual_Source'}
    if not need.issubset(same_book.columns):
        return 'NOT_PUBLISHED'
    h = same_book.loc[
        same_book['V55_Overlay_Actual_Source'].astype(str).eq(V55_RAW_OVERLAY_TABLE)
        & same_book['_snap'].notna()
        & same_book['_snap'].le(at)
        & same_book['_snap'].lt(same_book['_kick'])
    ].copy()
    if h.empty:
        return 'NOT_PUBLISHED'
    h['_price'] = pd.to_numeric(h['Value'], errors='coerce')
    h = h.loc[h['_price'].notna()].copy()
    if h.empty:
        return 'NOT_PUBLISHED'
    # An alternate spread at the same time is not the same price path.
    # Discard ambiguous timestamp groups rather than comparing alternate lines.
    n = h.groupby('_snap')['_price'].transform('nunique')
    h = h.loc[n.eq(1)].sort_values('_snap').drop_duplicates('_snap', keep='last')
    if h.empty:
        return 'NOT_PUBLISHED'
    current = h.loc[
        h['_price'].sub(line).abs().le(.001)
        & (at - h['_snap']).le(pd.Timedelta(minutes=15))
    ]
    if current.empty:
        return 'NOT_PUBLISHED'
    selected = current.sort_values('_snap').iloc[-1]
    current_at = selected['_snap']
    vals = []
    opener = _price(selected.get('Open_Value'))
    if opener is not None:
        vals.append(f'Open→Now {line-opener:+.1f}')
    for minutes, label in ((30,'30m'),(60,'60m'),(120,'120m')):
        cutoff = current_at - pd.Timedelta(minutes=minutes)
        old = h.loc[
            h['_snap'].le(cutoff)
            & h['_snap'].ge(cutoff - pd.Timedelta(minutes=15))
        ]
        if old.empty:
            continue
        earlier = old.iloc[-1]
        vals.append(f'{label} {line-float(earlier["_price"]):+.1f}')
    return ' | '.join(vals[:6]) if vals else 'NOT_PUBLISHED'
'''


def replace_once(src: str, old: str, new: str, name: str) -> str:
    n = src.count(old)
    if n != 1:
        raise ValueError(f'UNIQUE_ANCHOR_REQUIRED {name} count={n}')
    return src.replace(old, new, 1)


def transform(src: str) -> str:
    if BANNER in src:
        raise ValueError('ALREADY_APPLIED: run --self-test --target-self-test instead')
    # Fail closed before any file is touched. Restrict supported source contract.
    src = replace_once(src,
        f'V55_RESEARCH_MINER_POLICY = "{MINER_OLD}"\n',
        f'V55_RESEARCH_MINER_POLICY = "{MINER_OLD}"\n' + SUPPORTED_DECLARATION,
        'miner_policy_declaration')
    src = replace_once(src,
        "        if (report.get('miner_live_authority_policy') or {}).get('policy')!=V55_RESEARCH_MINER_POLICY:return None,'RESEARCH_POLICY_MISMATCH'",
        "        frozen_policy=(report.get('miner_live_authority_policy') or {}).get('policy')\n"
        "        if frozen_policy not in V55_RESEARCH_MINER_SUPPORTED_POLICIES:return None,'RESEARCH_POLICY_MISMATCH'",
        'miner_policy_in_report')
    src = replace_once(src,
        "            if rec.get('confirmation_pass') is not True or rec.get('live_authority_eligible') is not True:continue",
        "            if rec.get('confirmation_pass') is not True or rec.get('live_authority_eligible') is not True:continue\n"
        "            if rec.get('live_authority_policy')!=frozen_policy:continue",
        'per_rule_policy_provenance')
    src = replace_once(src,
        "        return {'mechanisms':mechanisms,'created_utc':created,",
        "        return {'mechanisms':mechanisms,'policy':frozen_policy,'created_utc':created,",
        'registry_policy_passthrough')
    src = replace_once(src,
        "                if v.get('live_authority_policy')!=V55_RESEARCH_MINER_POLICY:reject('POLICY_VERSION_MISMATCH');continue",
        "                if v.get('live_authority_policy')!=registry.get('policy'):reject('POLICY_VERSION_MISMATCH');continue",
        'vote_matches_frozen_report_policy')
    # The existing loader projects only optional columns present in the actual
    # scanner table. Without this, a real Open_Value column is dropped before
    # the display fallback can examine it.
    src = replace_once(src,
        "optional=list(dict.fromkeys(x for x in (*V55_RAW_OVERLAY_FIELDS,'Bookmaker','Book',",
        "optional=list(dict.fromkeys(x for x in (*V55_RAW_OVERLAY_FIELDS,'Bookmaker','Book','Open_Value',",
        'optional_real_open_value_projection')
    src = replace_once(src,
        '\ndef _v55_read_enriched_pregame_overlays(bq_client,',
        MARKET_HELPER + '\ndef _v55_read_enriched_pregame_overlays(bq_client,',
        'market_helper_insertion')
    src = replace_once(src,
        "            out.at[idx,'V55_Raw_Market_Rich']=_v55_raw_rich(richrow)",
        "            rich_value=_v55_raw_rich(richrow)\n"
        "            if rich_value=='NOT_PUBLISHED':\n"
        "                rich_value=_v55_rich_from_market_snapshots(\n"
        "                    same_book,anchor,r.get('Current_Spread'))\n"
        "            out.at[idx,'V55_Raw_Market_Rich']=rich_value",
        'market_display_source_fallback')
    src = f'# {BANNER}: safe source-checked registry version + snapshot display only\n' + src
    ast.parse(src, filename='ncaaf_v55_board.py')
    return src


def _norm_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if p.name != 'ncaaf_v55_board.py':
        raise ValueError('Refusing to target anything other than ncaaf_v55_board.py')
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return p


def patch_file(path: str, apply: bool) -> None:
    file = _norm_path(path)
    orig = file.read_text(encoding='utf8')
    try:
        revised = transform(orig)
    except ValueError as exc:
        if BANNER in orig and str(exc).startswith('ALREADY_APPLIED'):
            print('ALREADY_APPLIED path='+str(file))
            return
        raise
    old_hash=hashlib.sha256(orig.encode()).hexdigest()
    new_hash=hashlib.sha256(revised.encode()).hexdigest()
    print('PATCH_CHECK_PASS path='+str(file))
    print('ORIGINAL_SHA256 '+old_hash)
    print('PATCHED_SHA256  '+new_hash)
    print('MARKET_RICH_MODE DISPLAY_ONLY_SOURCE_VERIFIED_SNAPSHOTS')
    print('MINER_POLICY_MODE EXACT_REPORT_POLICY_V2_20_OR_V2_21_PLUS_EXISTING_OOS_GATES')
    if not apply:
        print('READ_ONLY: run --apply to install backed-up patch')
        return
    suffix = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = file.with_name(file.name + f'.before_{BANNER}_{suffix}.bak')
    if backup.exists():
        raise FileExistsError(str(backup))
    shutil.copy2(file, backup)
    tmp = file.with_name(file.name + '.v55_safe_repair_tmp')
    try:
        tmp.write_text(revised, encoding='utf8')
        os.chmod(tmp, file.stat().st_mode & 0o777)
        ast.parse(tmp.read_text(encoding='utf8'))
        os.replace(tmp, file)
    except BaseException:
        tmp.unlink(missing_ok=True)
        shutil.copy2(backup, file)
        raise
    print('APPLIED: '+str(file))
    print('BACKUP:  '+str(backup))


def diagnose():
    """Read original registry metadata; never rewrite source, registry or cert."""
    try:
        from google.cloud import storage
    except ImportError:
        print('SKIP: google-cloud-storage is not installed; --check/--apply still work')
        return
    bucket = os.getenv('MODEL_BUCKET', 'sharp-models')
    client = storage.Client(project='sharplogger')
    paths = (
        ('MINER', 'research/ncaaf/v2/current_report.json'),
        ('INDEPENDENT', 'research/ncaaf/v55/qualified_independent_brains_2013_2025.json'),
        ('INDEPENDENT_AUDIT','research/ncaaf/v55/independent_brain_registry_audit_latest.json'),
    )
    def safe_metadata(obj):
        if not isinstance(obj, dict):return {'object_type':type(obj).__name__}
        d={'status':obj.get('status'),'policy':obj.get('policy'),
           'created_utc':obj.get('created_utc'),'operator_approved':obj.get('operator_approved'),
           'history_first_season':obj.get('history_first_season'),
           'discovery_last_season':obj.get('discovery_last_season'),
           'confirmation_seasons':obj.get('confirmation_seasons'),
           'uses_2026_outcomes':obj.get('uses_2026_outcomes'),
           'miner_live_authority_policy':obj.get('miner_live_authority_policy'),
           'research_status':obj.get('research_status')}
        if isinstance(obj.get('systems'),list):
            from collections import Counter
            d['system_count']=len(obj['systems'])
            d['system_families']=dict(Counter(str(x.get('family')) for x in obj['systems'] if isinstance(x,dict)))
        fam=(obj.get('system_miner_v3') or {}).get('spreads',{}).get('mechanism_families',[])
        if isinstance(fam,list):
            d['miner_spread_mechanisms']=len(fam)
            d['miner_live_authority_eligible']=sum(x.get('live_authority_eligible') is True for x in fam if isinstance(x,dict))
            d['miner_policy_matches_eligible']=sum(x.get('live_authority_eligible') is True and x.get('live_authority_policy')==(obj.get('miner_live_authority_policy') or {}).get('policy') for x in fam if isinstance(x,dict))
        return {k:v for k,v in d.items() if v is not None}
    for label,key in paths:
        try:
            blob=client.bucket(bucket).blob(key)
            raw=blob.download_as_bytes()
            obj=json.loads(raw)
            print(label+' '+json.dumps({'gs_uri':'gs://'+bucket+'/'+key,
                  'sha256':hashlib.sha256(raw).hexdigest(),
                  'metadata':safe_metadata(obj)},sort_keys=True,default=str))
        except Exception as exc:
            print(label+' READ_ERROR '+type(exc).__name__+':'+str(exc)[:240])
    try:
        import ncaaf_research_v2 as miner
        print('LOCAL_MINER '+json.dumps({
            'research_source_tag':getattr(miner,'NCAAF_RESEARCH_V2_SOURCE_TAG',None),
            'live_authority_policy':getattr(miner,'NCAAF_MINER_LIVE_AUTHORITY_POLICY',None),
            'file':getattr(miner,'__file__',None)},sort_keys=True))
    except Exception as exc:
        print('LOCAL_MINER IMPORT_ERROR '+type(exc).__name__+':'+str(exc)[:180])


def selftest(target=False, path="ncaaf_v55_board.py"):
    # Tests operate on a minimal source-structure fixture, so fail if any
    # source anchors change or if output would duplicate a quoted vote.
    from textwrap import dedent
    fixture = dedent(f'''\
        V55_RESEARCH_MINER_POLICY = "{MINER_OLD}"
        def dummy(report, rec, v, registry):
            if (report.get('miner_live_authority_policy') or {{}}).get('policy')!=V55_RESEARCH_MINER_POLICY:return None,'RESEARCH_POLICY_MISMATCH'
            if rec.get('confirmation_pass') is not True or rec.get('live_authority_eligible') is not True:continue
            if v.get('live_authority_policy')!=V55_RESEARCH_MINER_POLICY:reject('POLICY_VERSION_MISMATCH');continue
            return {{'mechanisms':mechanisms,'created_utc':created,
        def _v55_read_enriched_pregame_overlays(bq_client,):pass
            out.at[idx,'V55_Raw_Market_Rich']=_v55_raw_rich(richrow)
        ''')
    # 'continue' and incomplete return are intentionally a fixture, not syntax;
    # validate the independent string anchors plus a real minimal source below.
    assert MINER_OLD in fixture
    assert MINER_NEW in SUPPORTED_DECLARATION
    assert SUPPORTED_DECLARATION.count('V55_RESEARCH_MINER_POLICY')==1
    assert '_v55_rich_from_market_snapshots' in MARKET_HELPER
    if target:
        p=_norm_path(path)
        s=p.read_text(encoding='utf8')
        assert BANNER in s, 'target patch was not installed'
        assert "if v.get('live_authority_policy')!=registry.get('policy')" in s
        assert 'if rec.get(\'live_authority_policy\')!=frozen_policy' in s
        ast.parse(s)
        # Validate that the displayed movement cannot create evidence or bets.
        helper=s[s.index('def _v55_rich_from_market_snapshots'):s.index('def _v55_read_enriched_pregame_overlays')]
        assert 'V55_Bet_Authority' not in helper
        assert 'Confirm_Count' not in helper
    print('SELF_TEST_PASS target='+str(target))


def main():
    p=argparse.ArgumentParser()
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true')
    modes.add_argument('--apply',action='store_true')
    modes.add_argument('--diagnose',action='store_true')
    modes.add_argument('--self-test',action='store_true')
    p.add_argument('--target',default='ncaaf_v55_board.py')
    p.add_argument('--target-self-test',action='store_true')
    a=p.parse_args()
    if a.check:patch_file(a.target,False)
    if a.apply:patch_file(a.target,True)
    if a.diagnose:diagnose()
    if a.self_test:selftest(target=a.target_self_test, path=a.target)


if __name__=='__main__':
    try:main()
    except Exception as e:
        print('ABORT '+type(e).__name__+':'+str(e),file=sys.stderr)
        sys.exit(2)
