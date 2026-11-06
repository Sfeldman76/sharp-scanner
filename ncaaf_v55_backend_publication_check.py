#!/usr/bin/env python3
"""Read-only latest NCAAF V5.5 publication diagnostics; no scoring/writes."""
import json

TABLE = 'sharplogger.sharp_data.ncaaf_v55_live_board'


def inspect(audit):
    miner=audit.get('v55_unified_miner_v2_live_executor') or {}
    live=audit.get('v55_live_system_evaluation') or {}
    return {
        'published_at':audit.get('published_at_utc'),
        'unified_backend_bridge_deployed':bool(miner),
        'miner_rule_count':audit.get('v55_unified_miner_v2_rule_count'),
        'quote_atom_rule_evaluable':miner.get('quote_atom_rules_evaluable'),
        'quote_atom_rules_missing_sources':miner.get('rules_with_missing_atom_providers'),
        'source_atoms_unavailable':miner.get('unavailable_atoms'),
        'quote_atom_live_firings':miner.get('fired_rule_ids'),
        'quote_atom_research_events':audit.get('v55_unified_quote_atom_research_events'),
        'legacy_live_miner_status':live.get('miner'),
        'legacy_live_miner_vote_rows':live.get('miner_vote_rows'),
        'legacy_live_miner_report_status':live.get('miner_report_status'),
        'upstream_verified_event_rows':audit.get('v55_unified_upstream_event_rows'),
        'research_brains':audit.get('v55_unified_live_research_by_brain'),
        'qualified_brains':audit.get('v55_unified_live_qualified_by_brain'),
        'certified_combinations':audit.get('v55_unified_certified_combinations'),
        'authority_evaluable':audit.get('v55_authority_evaluable'),
        'authorized_bets':audit.get('v55_authorized_bet_rows'),
        'hold_reasons':audit.get('v55_unified_hold_reasons'),
    }


def main():
    from google.cloud import bigquery
    client=bigquery.Client(project='sharplogger')
    rows=list(client.query(f'''SELECT snapshot_id, published_at, audit_json
        FROM `{TABLE}` WHERE record_kind='COMPLETE'
        ORDER BY published_at DESC LIMIT 1''').result())
    if not rows:raise SystemExit('NO_COMPLETE_NCAAF_V55_PUBLICATION')
    record=rows[0]
    result=inspect(json.loads(record.audit_json))
    result['snapshot_id']=str(record.snapshot_id)
    result['published_at']=str(record.published_at)
    print(json.dumps(result,indent=2,sort_keys=True,default=str))
    if not result['unified_backend_bridge_deployed']:
        print('\nHOLD: New engine was not executed in the backend publisher image. Uploading GitHub source alone does not redeploy Cloud Run.')
    elif not result['certified_combinations']:
        print('\nHOLD: No accepted independent original-vintage combination certificates; no authorized bets can be issued.')

if __name__=='__main__':main()
