"""Read-only 2026 prospective expert-brain performance ledger.

Loads IMMUTABLE pregame observations and separately settled final scores from
GCS. Aggregates one first-observed pregame signal per physical game/system.
2026 results MUST NOT select/qualify the 2013-2025 frozen V5.5 registry.

Cloud Shell: python3 analyze_v55_prospective_brain_evidence.py --days 21
"""
import argparse
import hashlib
import json
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta

BUCKET='sharp-models'
PREFIX='research/ncaaf/v55/prospective_brain_evidence_v1'


def _parsed(v):
    from datetime import datetime,timezone
    if not isinstance(v,str) or not v:return None
    try:
        dt=datetime.fromisoformat(v.replace('Z','+00:00'))
        return dt.astimezone(timezone.utc) if dt.tzinfo else None
    except (ValueError,TypeError):return None


def grade_immutable_evidence(observations, outcomes):
    """Outcome grades for research only; no pricing or selection mutations."""
    seen={};audit=Counter()
    for sig in observations:
        fam=str(sig.get('family') or '').strip().upper()
        gid=str(sig.get('physical_game_id') or '').strip()
        side=str(sig.get('target') or '').strip().casefold()
        name=str(sig.get('signal_id') or '').strip()
        if not gid or not fam or not name or not side or fam not in ('MINER','BIG_AL','PATHI','PT','STAT'):
            audit['skipped_non_directional_or_id_missing']+=1;continue
        if sig.get('market')!='spreads' or sig.get('game_id_verified') is not True:
            audit['skipped_market_or_identity']+=1;continue
        game_dt=_parsed(sig.get('kickoff_utc'))
        source_dt=_parsed(sig.get('source_snapshot_utc'))
        capture_dt=_parsed(sig.get('captured_at_utc'))
        if game_dt is None or source_dt is None or capture_dt is None or not source_dt<=capture_dt<game_dt:
            audit['skipped_unverified_pregame']+=1;continue
        if sig.get('original_quote_line_verified') is not True:
            audit['skipped_unverified_line']+=1;continue
        home=str(sig.get('home_team') or '').casefold().strip()
        away=str(sig.get('away_team') or '').casefold().strip()
        if side not in (home,away) or home==away:
            audit['skipped_direction_mismatch']+=1;continue
        line=sig.get('original_book_line')
        try:line=float(line)
        except (ValueError,TypeError):audit['skipped_bad_spread']+=1;continue
        if not -100<=line<=100:
            audit['skipped_bad_spread']+=1;continue
        # One prospective observation per physical game / source / specific
        # system / side; choose EARLIEST receipt rather than outcome-dependent
        # selection of the most favorable historical quote.
        key=(gid,fam,name,side)
        if key not in seen or capture_dt<seen[key][0]:
            seen[key]=(capture_dt,{**sig,'_line':line,'_side_is_home':side==home})
    detail=[];pending=set()
    for (gid,fam,sysid,side),(_,sig) in seen.items():
        final=outcomes.get(gid)
        if final is None:
            pending.add(gid);continue
        try:
            h=float(final['home_score']);a=float(final['away_score'])
        except (KeyError,ValueError,TypeError):
            audit['skipped_invalid_outcome']+=1;continue
        if str(final.get('home_team_at_capture') or '').casefold().strip()!=str(sig.get('home_team') or '').casefold().strip():
            audit['skipped_score_orientation_mismatch']+=1;continue
        if str(final.get('away_team_at_capture') or '').casefold().strip()!=str(sig.get('away_team') or '').casefold().strip():
            audit['skipped_score_orientation_mismatch']+=1;continue
        margin=(h-a) if sig['_side_is_home'] else (a-h)
        spread_result=round(margin+sig['_line'],5)
        label='WIN' if spread_result>0 else 'LOSS' if spread_result<0 else 'PUSH'
        detail.append({'physical_game_id':gid,'brain':fam,'system_id':sysid,
                       'grade':label,'observed_book':sig.get('sportsbook'),
                       'observation_side':side,'observed_spread':sig['_line'],
                       'first_captured_utc':sig['captured_at_utc']})
    by=defaultdict(Counter)
    for item in detail:by[item['brain']][item['grade']]+=1
    return {'status':'RESEARCH_ONLY_NO_2026_PROMOTION',
            'directional_unique_pregame_system_events':len(seen),
            'graded_events':len(detail),
            'pending_game_ids':len(pending),
            'graded_by_brain':{brain:dict(counts) for brain,counts in sorted(by.items())},
            'skips':dict(audit),
            'production_authority':0, 'uses_2026_outcomes_for_selection':False,
            'grade_scope':'SPREAD_PREDICTION_SIDE_FIRST_VERIFIED_PREGAME_QUOTE',
            'dedup_policy':'FIRST_CAPTURE_PER_GAME_SYSTEM_SIDE',
            'notes':'Correlated events in the same game are NOT independent bets. No ROI is implied by ATS grades.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days',type=int,default=21)
    args=parser.parse_args()
    from google.cloud import storage
    days=max(1,min(90,args.days))
    client=storage.Client();bucket=client.bucket(BUCKET)
    now=datetime.now(timezone.utc)
    observations=[];dates=[]
    for day in range(days+1):
        date=(now-timedelta(days=day)).strftime('%Y-%m-%d')
        dates.append(date)
        for blob in client.list_blobs(BUCKET,prefix=f'{PREFIX}/observations/{date}/'):
            try:
                data=json.loads(blob.download_as_bytes())
                if data.get('policy')=='NCAAF-ORIGINAL-PREGAME-OBSERVATIONS-V1-20261026':
                    observations.extend(data.get('observations') or [])
            except (ValueError,TypeError):continue
    ids=sorted({str(x.get('physical_game_id') or '') for x in observations if x.get('physical_game_id')})
    outcomes={}
    for gid in ids:
        key=hashlib.sha256(gid.encode()).hexdigest()[:32]
        try:
            raw=bucket.blob(f'{PREFIX}/outcomes/{key}.json').download_as_bytes()
            data=json.loads(raw)
            if str(data.get('physical_game_id'))==gid:outcomes[gid]=data
        except Exception:pass
    report=grade_immutable_evidence(observations,outcomes)
    report['observation_rows_read']=len(observations)
    report['outcome_records_read']=len(outcomes)
    report['analysis_days']=days
    print(json.dumps(report,indent=2,sort_keys=True))
    print('PROSPECTIVE_RESEARCH_AUDIT_COMPLETE_READ_ONLY')

if __name__=='__main__':main()
