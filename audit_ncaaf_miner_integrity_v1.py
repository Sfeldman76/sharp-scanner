"""Read-only NCAAF System Miner integrity and transfer audit.

No training, production mutation, cloud writes, or bet authority. Reads the
current V2 Miner report (GCS by default, or --report for offline audit) and
optionally the combined-brains replay audit. This is a manifest-level audit,
NOT an actual-game confluence backtest or proof of historical timestamp vintage.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pathlib
import re
from typing import Any

DEFAULT_REPORT_BLOB = 'research/ncaaf/v2/current_report.json'
PROTECTED_PREFIXES = ('CORE_OOF_', 'SPEC_', 'CFBD_WEATHER_')


def _number(x: Any) -> float | None:
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _load(path: str | None, bucket: str, blob: str):
    if path:
        return json.loads(pathlib.Path(path).read_text(encoding='utf-8')), f'local:{path}'
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise RuntimeError('Install google-cloud-storage, or use --report local.json') from exc
    uri=f'gs://{bucket}/{blob}'
    return json.loads(storage.Client().bucket(bucket).blob(blob).download_as_bytes()), uri


def audit(report: dict[str, Any], replay: dict[str, Any] | None = None, source_code: str | None = None):
    miners=report.get('system_miner_v3') or {}
    rows=[]
    for market, obj in miners.items():
        if not isinstance(obj,dict):
            continue
        for m in obj.get('mechanism_families') or []:
            if not isinstance(m, dict):
                continue
            conditions=[str(x) for x in (m.get('representative_conditions') or [])]
            status=str(m.get('incumbent_revalidation_status') or '')
            held=status in {'LEGACY_CONTRACT_HOLD', 'HOLD_NOT_EVALUABLE'}
            live=bool(m.get('live_authority_eligible'))
            confirmed=bool(m.get('confirmation_pass'))
            conf_by_season=((m.get('attribution') or {}).get('confirmation_seasons') or [])
            years={int(z.get('season')) for z in conf_by_season if z.get('season') is not None}
            mkey=str(m.get('evidence_family_key') or m.get('mechanism_id') or '')
            flagged=[]
            if held: flagged.append('GRANDFATHERED_CONTRACT' if status=='LEGACY_CONTRACT_HOLD' else 'INPUT_UNAVAILABLE')
            if live and status=='HOLD_NOT_EVALUABLE': flagged.append('LIVE_WHILE_INPUT_MISSING')
            if live and any(c.startswith(PROTECTED_PREFIXES) for c in conditions): flagged.append('LIVE_RESTRICTED_SOURCE')
            if confirmed and years and not {2024,2025}.issubset(years): flagged.append('CONFIRMED_WITH_INCOMPLETE_YEAR_BREAKDOWN')
            if not conditions: flagged.append('NO_EXACT_RULE_CONDITIONS')
            if not mkey: flagged.append('NO_EVIDENCE_FAMILY_KEY')
            if str(m.get('meta_constituent_overlap_guard','PASS'))=='FAIL': flagged.append('META_SELF_CONFIRMATION')
            rows.append({
                'market':str(market),'mechanism_id':str(m.get('mechanism_id') or ''),
                'direction':str(m.get('direction') or ''), 'evidence_family_key':mkey,
                'rule':' AND '.join(conditions),'condition_count':len(conditions),
                'discovery_n':m.get('discovery_n'), 'discovery_rate':_number(m.get('discovery_rate')),
                'confirmation_n':m.get('confirmation_n'), 'confirmation_rate':_number(m.get('confirmation_rate')),
                'confirmed':confirmed, 'live_authority_eligible':live, 'revalidation_status':status or 'NOT_SPECIFIED',
                'available_confirmation_years':','.join(str(x) for x in sorted(years)),
                'flags':';'.join(flagged),
            })
    by_market={}
    for market in sorted(set(x['market'] for x in rows)|set(miners)):
        sub=[x for x in rows if x['market']==market]
        keys={x['evidence_family_key'] for x in sub if x['live_authority_eligible']}
        by_market[market]={
            'mechanism_families':len(sub),
            'confirmed':sum(x['confirmed'] for x in sub),
            'live_eligible':sum(x['live_authority_eligible'] for x in sub),
            'distinct_live_evidence_keys':len(keys),
            'legacy_contract_holds':sum(x['revalidation_status']=='LEGACY_CONTRACT_HOLD' for x in sub),
            'unevaluable_holds':sum(x['revalidation_status']=='HOLD_NOT_EVALUABLE' for x in sub),
            'flagged':sum(bool(x['flags']) for x in sub),
            'atoms':(miners.get(market) or {}).get('atoms'),
            'tested':(miners.get(market) or {}).get('tested_hypotheses'),
        }
    sigs=collections.defaultdict(list)
    evidence=collections.defaultdict(set)
    for x in rows:
        k=(x['market'],x['direction'],x['rule'])
        sigs[k].append(x['mechanism_id'])
        if x['live_authority_eligible']:
            evidence[(x['market'],x['evidence_family_key'])].add(x['direction'])
    dup=[{'market':k[0], 'direction':k[1], 'rule':k[2], 'ids':v} for k,v in sigs.items() if len(v)>1]
    opposing=[{'market':k[0],'evidence_family_key':k[1],'directions':sorted(v)} for k,v in evidence.items() if len(v)>1]
    lib=report.get('system_library_inventory') or {}
    results=report.get('system_results') or {}
    results_by_market={}
    for market,d in (results.get('markets') or {}).items():
        results_by_market[market]={
            'confirmed_mechanisms_in_pool':d.get('confirmed_mechanisms'),
            'any_n':(d.get('any_confirmed_system') or {}).get('n'),
            'any_rate':(d.get('any_confirmed_system') or {}).get('hit_rate'),
            'multi_n':(d.get('two_plus_independent') or {}).get('n'),
            'multi_rate':(d.get('two_plus_independent') or {}).get('hit_rate'),
            'conflict_rows':d.get('conflict_rows'),
            'WARNING':'Confluence is evaluated on same 2024-2025 data used for rule confirmation; diagnostic only',
        }
    replay=replay or {}
    replay_miner=replay.get('incumbent_miner_191_mechanisms_replayed')
    notes=[]
    if replay_miner is False:
        notes.append('COMBINED_BRAINS_REPLAY_SKIPS_EXACT_INCUMBENT_MINER_MECHANISMS')
    if source_code:
        if re.search(r'rep\s*=\s*max\(members,\s*key\s*=\s*lambda\s+z:\s*\(z\["confirmation_pass"\]',source_code):
            notes.append('MECHANISM_REPRESENTATIVE_SELECTED_USING_CONFIRMATION_OUTCOMES')
        elif ('rep=max(members' in source_code):
            notes.append('MECHANISM_REPRESENTATIVE_SELECTION_PRESENT__REVIEW_FOR_CONFIRMATION_LEAKAGE')
        if 'if _inc_status=="LEGACY_CONTRACT_HOLD":' in source_code:
            notes.append('INCUMBENT_CONTRACT_HOLD_CAN_PRESERVE_PRIOR_LIVE_AUTHORITY')
    context=report.get('cfbd_context_bridge') or {}
    historical=report.get('research_health') or {}
    output={
        'status':'READ_ONLY_MANIFEST_AUDIT_COMPLETE' if rows else 'NO_MECHANISM_RECORDS_FOUND',
        'production_mutation':False,'bet_authority':0,
        'scope':'EXACT_MECHANISM_MANIFEST_AND_AGGREGATE_OUTCOMES__NOT_GAME_LEVEL_REPLAY',
        'report_source_tag':report.get('source_tag'),
        'total_mechanisms':len(rows),'confirmed_mechanisms':sum(x['confirmed'] for x in rows),
        'live_eligible_mechanisms':sum(x['live_authority_eligible'] for x in rows),
        'by_market':by_market,'confluence_2024_2025':results_by_market,
        'same_exact_rule_multiple_mechanisms':dup[:50],
        'live_evidence_family_bidirectional_presence':opposing[:50],
        'incumbent_audit':{k:v for k,v in (report.get('incumbent_system_library_audit') or {}).items() if k!='details'},
        'library_reported':{k:v for k,v in lib.items() if k not in ('records','systems')},
        'source_bridge':{k:context.get(k) for k in ('status','source_rows','miner_rows','matched_rows','historical_rows','advanced_ready','returning_ready','roster_ready','portal_ready','talent_ready') if k in context},
        'research_health_status':historical.get('status'),
        'replay_exact_miner_library_replayed':replay_miner,
        'replay_rule_definition_postdates_confirmation_period':replay.get('rule_definition_postdates_confirmation_period'),
        'code_and_replay_warnings':notes,
        'critical_caveats':[
            'No mechanism-level paired CORE outcome rows supplied; cannot calculate incremental CORE benefit from this report alone.',
            'Confirmation-set rule selection / representative choice can bias apparent 2024-2025 hit rates.',
            'Pre-game market quote timestamp and vintage are not established by a published Miner report.',
            'Multiple mechanism rules may share team, matchup, source, and outcomes; correlated votes must be collapsed.',
        ],
    }
    return output,rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',help='Local current_report.json; if omitted read from GCS')
    p.add_argument('--replay-audit',help='Optional combined-brains retrospective_replay/audit.json')
    p.add_argument('--source',help='Optional local ncaaf_research_v2.py for static contract checks')
    p.add_argument('--bucket',default='sharp-models')
    p.add_argument('--blob',default=DEFAULT_REPORT_BLOB)
    p.add_argument('--out-dir',default='.')
    a=p.parse_args(argv)
    report,source=_load(a.report,a.bucket,a.blob)
    replay=json.loads(pathlib.Path(a.replay_audit).read_text()) if a.replay_audit else None
    src=pathlib.Path(a.source).read_text() if a.source else None
    audit_result,records=audit(report,replay,src)
    audit_result['read_source']=source
    output=pathlib.Path(a.out_dir).expanduser(); output.mkdir(parents=True,exist_ok=True)
    (output/'miner_integrity_audit.json').write_text(json.dumps(audit_result,indent=2,allow_nan=False,default=str)+'\n')
    cols=['market','mechanism_id','direction','evidence_family_key','rule','condition_count','discovery_n','discovery_rate',
          'confirmation_n','confirmation_rate','confirmed','live_authority_eligible','revalidation_status','available_confirmation_years','flags']
    with (output/'miner_exact_mechanisms.csv').open('w',newline='',encoding='utf-8') as fp:
        wr=csv.DictWriter(fp,fieldnames=cols);wr.writeheader();wr.writerows(records)
    print(json.dumps({'status':audit_result['status'],'mechs':len(records),'confirmed':audit_result['confirmed_mechanisms'],
                      'live_eligible':audit_result['live_eligible_mechanisms'],'by_market':audit_result['by_market'],
                      'warnings':audit_result['code_and_replay_warnings'],'outputs':str(output)},indent=2))


if __name__=='__main__': main()
