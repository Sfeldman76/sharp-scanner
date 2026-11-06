"""NCAAF V5.5: sealed, original-pregame, holdout *betting-return* certificates.

This is NOT a calibrated prediction model. It validates realized returns from
predeclared directions, quotes, independent signals, and immutable score grades.
It does not create missing as-of source evidence, certify retrospectively
constructed events, promote research-only historical artifacts, or place bets.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "V55-VALIDATED-BETTING-EDGE-R1"
MANIFEST = "V55-EDGE-PREDECLARED-SELECTION-R1"
SOURCE_AUDIT = "V55-ORIGINAL-PREGAME-CAPTURE-AUDIT-R1"
STATUS = "PASS_INDEPENDENT_ORIGINAL_PREGAME_EDGE_OOS"
BRAINS = {"CORE", "SYSTEM_MINER", "PATHI", "BIG_AL", "STAT", "PREDICTION_TRACKER", "MARKET_RICH"}
REQUIRED = {
    "season", "game_id", "kickoff_utc", "home_score", "away_score", "side", "spread", "odds", "book",
    "source_family", "system_id", "evidence_family_key", "signal_observed_at_utc",
    "quote_observed_at_utc", "feature_asof_utc", "signal_capture_ref", "quote_capture_ref",
    "original_pregame_signal_verified", "original_pregame_quote_verified", "feature_vintage_verified",
}


def _dt(s):
    try:
        v = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return v.astimezone(timezone.utc) if v.tzinfo is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def _true(s):
    return s is True or str(s).strip().lower() in {"true", "1", "yes"}


def _num(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def net_win(odds):
    o = _num(odds)
    return None if o is None or -100 < o < 100 else o / 100 if o > 0 else 100 / -o


def profit(result, odds):
    net = net_win(odds)
    if net is None: return None
    return net if result == "WIN" else (-1.0 if result == "LOSS" else 0.0 if result == "PUSH" else None)


def _family(s):
    f = str(s or "").upper().strip()
    return "PREDICTION_TRACKER" if f == "PT" else f


def component(family, system_id):
    return _family(family) + "|" + str(system_id).strip()


def _sha_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def _canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _critical_normal(raw):
    if not REQUIRED.issubset(raw):
        raise ValueError("MISSING_CERTIFICATION_COLUMNS:" + ",".join(sorted(REQUIRED-set(raw))))
    f = _family(raw["source_family"])
    if f not in BRAINS or not str(raw["system_id"]).strip() or not str(raw["evidence_family_key"]).strip():
        raise ValueError("INVALID_BRAIN_SYSTEM_DEPENDENCY")
    if not all(_true(raw[k]) for k in ("original_pregame_signal_verified", "original_pregame_quote_verified", "feature_vintage_verified")):
        raise ValueError("UNCERTIFIED_SOURCE_CAPTURE")
    if not raw["signal_capture_ref"] or not raw["quote_capture_ref"]:
        raise ValueError("ORIGINAL_CAPTURE_REF_MISSING")
    kick, obs, quote, feat = (_dt(raw[k]) for k in ("kickoff_utc", "signal_observed_at_utc", "quote_observed_at_utc", "feature_asof_utc"))
    if any(x is None for x in (kick, obs, quote, feat)) or not feat <= obs <= quote < kick or (quote-obs).total_seconds() > 5400:
        raise ValueError("INVALID_ORIGINAL_PREGAME_ASOF")
    side = str(raw["side"]).upper().strip()
    if side not in ("HOME", "AWAY"):
        raise ValueError("SIDE_MUST_BE_HOME_OR_AWAY")
    line, odds = _num(raw["spread"]), _num(raw["odds"])
    home, away = _num(raw["home_score"]), _num(raw["away_score"])
    if line is None or net_win(odds) is None or home is None or away is None or home < 0 or away < 0 or int(home) != home or int(away) != away:
        raise ValueError("INVALID_EXECUTABLE_PRICE_OR_FINAL_SCORE")
    year = int(raw["season"])
    if year < 2013 or year > 2026 or kick.year not in (year, year+1):
        raise ValueError("INVALID_SEASON_OR_KICKOFF")
    margin = (home-away if side == "HOME" else away-home) + line
    result = "WIN" if margin > 0.00001 else "LOSS" if margin < -0.00001 else "PUSH"
    return {"game_id":str(raw["game_id"]).strip(), "season":year,
            "kick":kick,"obs":obs,"quote":quote,"family":f,"sid":str(raw["system_id"]).strip(),
            "component_id":component(f,raw["system_id"]), "dep":str(raw["evidence_family_key"]).strip().upper(),
            "side":side,"line":line,"odds":odds,"book":str(raw["book"]).lower().strip(),
            "capture_ref":str(raw["signal_capture_ref"]).strip(),"quote_ref":str(raw["quote_capture_ref"]).strip(),
            "result":result,"net_profit":profit(result,odds), "home":int(home), "away":int(away)}


def _tcrit(confidence, n):
    # A Student t critical value is used, with Bonferroni control across the
    # entire sealed hypothesis universe. Never silently substitute a normal z.
    from scipy.stats import t
    return float(t.ppf(1-confidence, n-1))


def reward_stats(counts, odds, *, alpha=0.05):
    """Reprice observed WIN/LOSS/PUSH outcomes at a given American price.

    Confidence bound measures OOS net *bet returns*; no P(win) is published.
    """
    wins,losses,pushes = (int(counts[k]) for k in ("wins","losses","pushes"))
    n = wins+losses+pushes
    net = net_win(odds)
    if net is None or n<2 or not 0<alpha<0.5: return None
    roi = (wins*net - losses)/n
    var = (wins*(net-roi)**2 + losses*(-1-roi)**2 + pushes*roi**2)/(n-1)
    se = math.sqrt(max(0.0,var)/n)
    return {"roi":roi,"roi_lower":roi-_tcrit(alpha,n)*se,"n":n,"wins":wins,"losses":losses,"pushes":pushes}


def _validate_manifest(manifest):
    if not isinstance(manifest,dict) or manifest.get("schema")!=MANIFEST or not isinstance(manifest.get("selections"),list):
        raise ValueError("MISSING_PREDECLARED_SELECTION_MANIFEST")
    specs=manifest["selections"]
    if not specs: raise ValueError("EMPTY_PREDECLARED_SELECTIONS")
    m=int(manifest.get("hypotheses_total") or 0)
    if m<len(specs) or m<1:
        raise ValueError("INCOMPLETE_MULTIPLE_TESTING_UNIVERSE")
    seen=set()
    for s in specs:
        sid=str(s.get("selection_id") or "").strip()
        comp=s.get("components")
        if not sid or sid in seen or not isinstance(comp,list) or not comp:
            raise ValueError("DUPLICATE_OR_EMPTY_SELECTION")
        seen.add(sid)
        keys=[]; deps=[]
        for c in comp:
            family=_family(c.get("family"))
            name=str(c.get("system_id") or "").strip()
            dep=str(c.get("evidence_family_key") or "").strip().upper()
            if family not in BRAINS or not name or not dep:
                raise ValueError("INVALID_PREDECLARED_COMPONENT")
            keys.append(component(family,name));deps.append(dep)
        if len(set(keys))!=len(keys) or len(set(deps))!=len(deps):
            raise ValueError("DUPLICATED_OR_NONINDEPENDENT_COMPONENTS")
        if str(s.get("market"))!="spreads" or not _dt(s.get("frozen_at_utc")):
            raise ValueError("UNFROZEN_OR_UNSUPPORTED_SELECTION")
        if int(s.get("selection_end_season") or 0)<2013:
            raise ValueError("MISSING_SELECTION_END_SEASON")
    return specs,m


def certify(predeclared_manifest, original_capture_audit, historical_rows, *, source_sha256, created_utc=None,
            min_holdout_games=100, min_holdout_year_games=25, min_roi=0.02):
    """Return a sealed PASS certificate document or granular NO_CERTIFICATES audit.

    Source attestations must originate with a verified *external* pregame
    capture audit. This function checks a SHA binding; it cannot independently
    prove an upstream operator's attestation was truthful.
    """
    specs, universe=_validate_manifest(predeclared_manifest)
    aud=original_capture_audit
    if not isinstance(aud,dict) or aud.get('schema')!=SOURCE_AUDIT or aud.get('status')!='PASS':
        raise ValueError('ORIGINAL_CAPTURE_AUDIT_NOT_PASSED')
    if (aud.get('bet_record_sha256')!=source_sha256 or aud.get('selection_manifest_sha256')!=_sha_bytes(_canonical(predeclared_manifest))
        or not all(_true(aud.get(k)) for k in
        ('original_signal_capture_verified','original_price_capture_verified',
         'feature_vintage_verified','final_score_identity_verified','no_retroactive_signal_reconstruction',
         'frozen_holdout_universe_complete','all_signal_occurrences_accounted_for',
         'predeclared_selection_manifest_verified','predeclared_hypothesis_universe_verified'))):
        raise ValueError('ORIGINAL_CAPTURE_PROVENANCE_NOT_ATTESTED_OR_HASH_MISMATCH')
    rows=[]; rejects=defaultdict(int)
    for r in historical_rows:
        try:
            row=_critical_normal(r)
            if not row['game_id'] or not row['book']:
                raise ValueError('MISSING_GAME_ID_OR_BOOK')
            rows.append(row)
        except (ValueError,TypeError) as exc:
            rejects[str(exc)]+=1
    # Conflicting game outcomes or kickoff identities are fatal, not droppable.
    game_meta={}
    for r in rows:
        k=r['game_id']; v=(r['home'],r['away'],r['kick'])
        if k in game_meta and game_meta[k]!=v:
            raise ValueError('CONFLICTING_FINALS_OR_KICKOFF_FOR_GAME:'+k)
        game_meta[k]=v
    alpha = .05/universe
    certs=[];reasons={}
    # Invalid source observations may selectively hide losing bets. Never
    # certify a subset when any supplied original-capture row failed the
    # provenance/clock/score gate. Publish diagnostic rejection counts only.
    if rejects:
        return {"schema":SCHEMA,"status":"SOURCE_INTEGRITY_HOLD",
            "created_utc":created_utc or datetime.now(timezone.utc).isoformat(),
            "certificates":[],"accepted_selections":0,"tested_selections":len(specs),
            "invalid_source_rows":dict(rejects),"rejected_selections":{},
            "historical_evidence_sha256":source_sha256,
            "selection_manifest_sha256":_sha_bytes(_canonical(predeclared_manifest)),
            "original_pregame_capture_certified":False,
            "2026_excluded_from_selection":True,
            "policy":"VALIDATED_ROI_NOT_PREDICTED_WIN_PROBABILITY"}
    for s in specs:
        sid=str(s['selection_id']); freeze=_dt(s['frozen_at_utc'])
        end=int(s['selection_end_season'])
        years={int(y) for y in s.get('holdout_seasons',[])}
        if not years or min(years)<=end:
            reasons[sid]={'holds':['HOLDOUT_YEARS_NOT_AFTER_SELECTION'], 'oos_games':0}
            continue
        # No mid-holdout freeze may drop early losing games from the replay.
        # An exact frozen date and predeclared holdout-start proof are needed.
        if freeze >= datetime(min(years),8,1,tzinfo=timezone.utc):
            reasons[sid]={'holds':['FREEZE_AFTER_PREDECLARED_HOLDOUT_BEGAN'], 'oos_games':0}
            continue
        components={component(x['family'],x['system_id']):str(x['evidence_family_key']).strip().upper() for x in s['components']}
        target=set(components)
        # Historical validation must be later than rule selection, not merely
        # a 2024/25 replay of a rule selected on those seasons.
        group=defaultdict(dict)
        for r in rows:
            if r['component_id'] not in target or r['dep']!=components.get(r['component_id']):continue
            if r['season']<=end or r['obs']<=freeze:continue
            if r['quote']<=freeze:continue
            if r['season'] not in years:continue
            key=(r['game_id'],r['side'],r['book'],r['line'],r['odds'])
            past=group[key].get(r['component_id'])
            if past is None or r['quote']<past['quote']:
                group[key][r['component_id']]=r
        complete=[]
        for key, same_price in group.items():
            if set(same_price)!=target:continue
            record=next(iter(same_price.values()))
            if any(x['result']!=record['result'] for x in same_price.values()):
                raise ValueError('CONFLICTING_RESULT_IN_COMBINATION')
            complete.append(record)
        # Never select profitable quote ex post. Earliest certified same-game
        # quote wins, then lexicographically stable bookmaker/side order.
        by_game={}
        for r in sorted(complete,key=lambda x:(x['quote'],x['book'],x['side'])):
            by_game.setdefault(r['game_id'],r)
        chosen=list(by_game.values()); n=len(chosen)
        yearly=defaultdict(list)
        for r in chosen:yearly[r['season']].append(r)
        stats={str(y):{'n':len(v),'roi':sum(x['net_profit'] for x in v)/len(v)} for y,v in yearly.items()}
        fail=[]
        if n<min_holdout_games:fail.append('INSUFFICIENT_INDEPENDENT_GAME_SAMPLE')
        if (len(yearly)<2 and not (len(yearly)==1 and list(yearly)==[2026] and n>=150)):
            fail.append('INSUFFICIENT_SEASON_STABILITY')
        if any(len(v)<min_holdout_year_games or sum(x['net_profit'] for x in v)<=0 for v in yearly.values()):
            fail.append('SEASON_SAMPLE_OR_ROI_FAILED')
        if not chosen:fail.append('NO_MATCHED_PREGAME_BETS')
        profits=[x['net_profit'] for x in chosen]
        roi=sum(profits)/n if n else 0.0
        if roi<min_roi:fail.append('REALIZED_HOLDOUT_ROI_BELOW_THRESHOLD')
        if n>1:
            var=sum((p-roi)**2 for p in profits)/(n-1)
            lower=roi-_tcrit(alpha,n)*math.sqrt(var/n)
        else:lower=None
        if lower is None or lower<=0:fail.append('MULTIPLICITY_ADJUSTED_ROI_LOWER_BOUND_NONPOSITIVE')
        if fail:
            reasons[sid]={'holds':fail,'oos_games':n,'oos_roi':round(roi,6),
                          'oos_roi_lower':round(lower,6) if lower is not None else None,
                          'by_season':stats}
            continue
        counts={x:sum(r['result']==x for r in chosen) for x in ('WIN','LOSS','PUSH')}
        spread_min=min(x['line'] for x in chosen)
        spread_max=max(x['line'] for x in chosen)
        odds_min=min(x['odds'] for x in chosen)
        odds_max=max(x['odds'] for x in chosen)
        c={"selection_id":sid,"market":"spreads",
           "component_ids":sorted(target),"component_dependencies":{k:components[k] for k in sorted(target)},
           "frozen_at_utc":s['frozen_at_utc'],"selection_end_season":end,
           "holdout_seasons":sorted(yearly),"oos_unique_games":n,
           "wins":counts['WIN'],"losses":counts['LOSS'],"pushes":counts['PUSH'],
           "oos_realized_roi":roi,"oos_roi_lower_adjusted":lower,
           "by_season":stats,"allowed_books":sorted({x['book'] for x in chosen}),
           "validated_spread_min":spread_min,"validated_spread_max":spread_max,
           "validated_odds_min":odds_min,"validated_odds_max":odds_max,
           "tested_hypotheses_total":universe,"selection_manifest_sha256":_sha_bytes(_canonical(predeclared_manifest)),
           "historical_evidence_sha256":source_sha256,"original_quote_provenance_verified":True,
           "original_signal_provenance_verified":True,"frozen_before_oos":True,
           "confidence_method":"ONE_SIDED_STUDENT_T_BONFERRONI_FULL_PREDECLARED_UNIVERSE",
           "certificate_id":_sha_bytes(_canonical([sid,source_sha256,sorted(target),n,counts,round(roi,9)]))[:32]}
        certs.append(c)
    created=created_utc or datetime.now(timezone.utc).isoformat()
    result={"schema":SCHEMA,"status":STATUS if certs else 'NO_VALIDATED_EDGE_CERTIFICATES',
        "created_utc":created,"selection_manifest_sha256":_sha_bytes(_canonical(predeclared_manifest)),
        "historical_evidence_sha256":source_sha256,"original_capture_audit_sha256":_sha_bytes(_canonical(aud)),
        "original_pregame_capture_certified":True,"2026_excluded_from_selection":True,
        "tested_hypotheses_total":universe,"market":"spreads",
        "certificates":certs,"certificates_sha256":_sha_bytes(_canonical(certs)),
        "rejected_selections":reasons,"invalid_source_rows":dict(rejects),
        "accepted_selections":len(certs),"tested_selections":len(specs),"input_event_rows":len(rows),
        "policy":"VALIDATED_ROI_NOT_PREDICTED_WIN_PROBABILITY"}
    return result


def load_edge_certificates(doc, *, now=None):
    """Fail closed on malformed or re-labeled certificate documents."""
    if not isinstance(doc,dict) or doc.get('schema')!=SCHEMA or doc.get('status')!=STATUS:
        return {}
    if doc.get('certificates_sha256')!=_sha_bytes(_canonical(doc.get('certificates',[]))):
        return {}
    if doc.get('invalid_source_rows'):return {}
    if doc.get('original_pregame_capture_certified') is not True or doc.get('2026_excluded_from_selection') is not True:
        return {}
    created=_dt(doc.get('created_utc'))
    at=_dt(now) if now is not None else datetime.now(timezone.utc)
    if created is None or created>at:return {}
    universe=int(doc.get('tested_hypotheses_total') or 0)
    if universe<1:return {}
    found={}
    for c in doc.get('certificates',[]):
        keys=c.get('component_ids') or []
        if not isinstance(keys,list) or not keys or keys!=sorted(set(keys)) or any('|' not in x for x in keys):continue
        if c.get('market')!='spreads' or not c.get('frozen_before_oos') is True:continue
        if c.get('original_quote_provenance_verified') is not True or c.get('original_signal_provenance_verified') is not True:continue
        n=int(c.get('oos_unique_games') or 0)
        w=int(c.get('wins') or 0);l=int(c.get('losses') or 0);p=int(c.get('pushes') or 0)
        if n<100 or w+l+p!=n or c.get('tested_hypotheses_total')!=universe:continue
        if c.get('historical_evidence_sha256')!=doc.get('historical_evidence_sha256') or c.get('selection_manifest_sha256')!=doc.get('selection_manifest_sha256'):continue
        if not _dt(c.get('frozen_at_utc')) or _dt(c['frozen_at_utc'])>created:continue
        if not isinstance(c.get('holdout_seasons'),list) or not c['holdout_seasons'] or min(c['holdout_seasons'])<=int(c.get('selection_end_season') or 3000):continue
        if _num(c.get('oos_realized_roi')) is None or _num(c.get('oos_realized_roi'))<.02:continue
        if _num(c.get('oos_roi_lower_adjusted')) is None or _num(c.get('oos_roi_lower_adjusted'))<=0:continue
        deps=c.get('component_dependencies') or {}
        if not isinstance(deps,dict) or set(deps)!=set(keys) or len(set(deps.values()))!=len(deps):continue
        if not c.get('allowed_books') or not c.get('certificate_id'):continue
        found[tuple(keys)]=c
    return found


def current_edge(c, odds, spread, book):
    """Reprice historical WIN/LOSS/PUSH at the executable quote (not a win-prob model)."""
    if str(book).strip().lower() not in c['allowed_books'] or net_win(odds) is None or _num(spread) is None:
        return None
    if not (_num(c['validated_spread_min'])<=float(spread)<=_num(c['validated_spread_max'])):
        return None
    # Live price can be better than the observed best; never extrapolate to a
    # worse price than any quoted during validation.
    if float(odds)<_num(c['validated_odds_min']):return None
    alpha=.05/int(c['tested_hypotheses_total'])
    rep=reward_stats({'wins':c['wins'],'losses':c['losses'],'pushes':c['pushes']},odds,alpha=alpha)
    if rep is None:return None
    rep['historical_realized_roi']=float(c['oos_realized_roi'])
    return rep


def build_from_files(manifest_path, audit_path, csv_path):
    manifest=json.loads(Path(manifest_path).read_text())
    audit=json.loads(Path(audit_path).read_text())
    b=Path(csv_path).read_bytes()
    with Path(csv_path).open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    return certify(manifest,audit,rows,source_sha256=_sha_bytes(b))


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description='Certify sealed OOS *betting return* edges; no probability model')
    p.add_argument('--manifest',required=True)
    p.add_argument('--capture-audit',required=True)
    p.add_argument('--historical-bets',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args(argv)
    doc=build_from_files(args.manifest,args.capture_audit,args.historical_bets)
    Path(args.out).write_text(json.dumps(doc,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'status':doc['status'],'accepted_selections':doc['accepted_selections'],
        'tested_selections':doc['tested_selections'],'rejected_selections':doc['rejected_selections'],
        'invalid_source_rows':doc['invalid_source_rows']},indent=2,sort_keys=True))
    return 0 if doc['status']==STATUS else 2

if __name__=='__main__':
    raise SystemExit(main())
