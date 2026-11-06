"""NCAAF V5.5 single-engine candidate and certified betting authority.

Read-only research sources are NEVER automatically promoted. Seven brains can
originate a game-side candidate, including without CORE. The only writer is the
existing V5.5 board publisher. No legacy incumbent Miner library is admitted.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone

BRAINS = ("CORE", "STAT", "SYSTEM_MINER", "PATHI", "BIG_AL", "PREDICTION_TRACKER", "MARKET_RICH")
POLICY_ID = "V55-UNIFIED-SEVEN-BRAIN-VALIDATED-BETTING-EDGE-20261106-R1"
POLICY_STATUS = "OOS_VALIDATED_BETTING_EDGE_AUTHORITY"
MIN_EV = .02
MAX_BETS = 3
MAX_QUOTE_MIN = 90
BET_MAX_QUOTE_MIN = 15


def _dt(v):
    try:
        import pandas as pd
        t = pd.to_datetime(v, utc=True, errors="coerce")
        return None if pd.isna(t) else t.to_pydatetime()
    except Exception:
        return None


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _str(v):
    if v is None: return ""
    s = str(v).strip()
    return "" if s.lower() in ("none", "nan", "nat") else s


def _team(v):
    # Reuse the project's canonical team aliases; never fuzzy-match ourselves.
    import ncaaf_v55_board as board
    return board._norm_team(v)


def _prob_net(odds):
    p = _num(odds)
    if p is None or p == 0 or (-100 < p < 100): return None
    return p / 100 if p >= 100 else -100 / p


def _truth(v):
    return v is True or _str(v).lower() in ("true", "1", "yes")


def verify_miner_library(doc):
    """Prove latest unified Miner V2 loaded, preserving 345 rule IDs.

    The publication explicitly marks rules as RESEARCH_ONLY: that property is
    intentionally retained. Rules require actual 2026 firings and certified
    vintage/OOS proof before they become qualified votes.
    """
    expected = {"spreads": 116, "h2h": 120, "totals": 109}
    if not isinstance(doc, dict) or doc.get("schema") != "V55-EXECUTABLE-UNIFIED-FULL-HISTORY-2013-2025-R2":
        raise RuntimeError("LATEST_UNIFIED_MINER_V2_NOT_LOADED")
    rules = doc.get("rules") or []
    if doc.get("rule_count") != 345 or len(rules) != 345 or doc.get("market_counts") != expected:
        raise RuntimeError("MINER_V2_345_RULE_INTEGRITY_FAILURE")
    ids = [_str(x.get("rule_id")) for x in rules]
    if not all(ids) or len(set(ids)) != 345 or any(_truth(x.get("legacy_incumbent")) for x in rules):
        raise RuntimeError("MINER_V2_DUPLICATE_OR_LEGACY_RULE")
    if any(not _truth(x.get("rule_definition_executable")) for x in rules):
        raise RuntimeError("MINER_V2_NON_EXECUTABLE_RULE")
    # Arm all 345 frozen definitions for LIVE observation, not automatic bets.
    # Historical certification fields remain authoritative and unchanged.
    # Use runtime copies: never rewrite the pinned, hash-checked GCS library.
    return {
        x["rule_id"]: {**x, "live_tracking_enabled": True,
                        "live_tracking_state": "LIVE_MONITORED"}
        for x in rules
    }


def _event_identity(raw):
    family = _str(raw.get("Source_Family")).upper()
    if family == "PT": family = "PREDICTION_TRACKER"
    sid = _str(raw.get("Signal_Family") or raw.get("Rule_ID") or raw.get("system_id"))
    dep = _str(raw.get("Evidence_Family_Key") or raw.get("source_dependency") or family).upper()
    return family, sid, dep


def _event_valid(raw, row, team, now, miner_rules):
    """Apply original-quote, direction and evidence rules to each live event.

    A raw 2026 signal can originate RESEARCH, but the certified flag is a
    separate, deliberately strict test. No vote derives from text overlays.
    """
    f, sid, dep = _event_identity(raw)
    if f not in BRAINS or f == "CORE" or not sid: return None
    kick = _dt(row.get("Game_Start_UTC"))
    obs = _dt(raw.get("Observed_At_UTC"))
    if not kick or not obs or not obs <= now < kick: return None
    if _str(raw.get("Game_ID")) != _str(row.get("Game_ID")): return None
    if _dt(raw.get("Game_Start_UTC")) != kick: return None
    sig_team = _team(raw.get("Signal_Side"))
    (home, away), ok = __import__("ncaaf_v55_board")._matchup_teams(row)
    if not ok or sig_team not in (home, away): return None
    if not _truth(raw.get("Direction_Verified")): return None
    book = _str(raw.get("Observed_Book")).lower()
    quote_book = _str(row.get("Backend_Quote_Book")).lower()
    spread = _num(raw.get("Observed_Spread"))
    chosen = _num(row.get("Current_Spread"))
    # Upstream frozen-rule validators in ncaaf_v55_board.py return a normalized
    # event WITHOUT Observed_Book / Observed_Spread.  The previous engine
    # rejected every such event here. Its builder has already checked exact
    # original game, book, side, spread and pregame timestamp. Admit that
    # upstream-vetted form only when all its certification markers agree;
    # unvetted external/research evidence cannot use this fallback.
    upstream_verified = (
        _truth(raw.get('Registry_Verified')) and
        _str(raw.get('Qualification_Status')) == 'QUALIFIED_OOS' and
        _str(raw.get('OOS_Status')) == 'PASS' and
        _truth(raw.get('Direction_Verified')) and
        _str(raw.get('Provenance_Status')) in
            ('PRE_GAME_VERIFIED','PRE_GAME_VERIFIED_LIVE_RECALCULATED')
    )
    edge_price_attested = bool(book and spread is not None)
    if not book and spread is None and upstream_verified:
        book, spread = quote_book, chosen
    # Same game, bookmaker and absolute line; no mismatched quote promotion.
    if not book or book != quote_book or spread is None or chosen is None or abs(abs(spread)-abs(chosen)) > .001:
        return None
    q = _dt(row.get("Backend_Quote_Timestamp"))
    if not q or q > now or q >= kick: return None
    provenance = _str(raw.get("Provenance_Status"))
    if provenance == "PRE_GAME_VERIFIED":
        if obs > q: return None
    elif provenance == "PRE_GAME_VERIFIED_LIVE_RECALCULATED":
        original = _dt(raw.get("Observed_Quote_UTC"))
        if not original or not (original <= q <= obs <= now) or (q-original).total_seconds() > 900 or (obs-q).total_seconds() > 900 or (now-obs).total_seconds() > 300:
            return None
    else:
        return None
    qualified = _dt(raw.get("Qualified_At_UTC"))
    valid_cert = (_str(raw.get("Qualification_Status")) == "QUALIFIED_OOS" and
        _str(raw.get("OOS_Status")) == "PASS" and qualified is not None and qualified <= obs and qualified < kick and
        _truth(raw.get("Registry_Verified")))
    if f == "SYSTEM_MINER":
        # New frozen V2 library ONLY. Old incumbent/legacy IDs do not even
        # enter candidate discovery, independent of their historical labels.
        rule = miner_rules.get(sid)
        if rule is None: return None
        valid_cert = valid_cert and rule is not None and _truth(rule.get("historical_oos_certified")) and _truth(rule.get("live_authority_eligible"))
        if not rule or rule.get("market") != "spreads":
            valid_cert = False
    elif f == "MARKET_RICH":
        # Raw bookmaker context isn't independent proof of historical skill.
        valid_cert = valid_cert and _truth(raw.get("Original_Pregame_Quote_Verified"))
    return {"family": f, "system_id": sid, "dependency": dep,
            "team": sig_team, "certified": bool(valid_cert),
            "edge_price_attested": edge_price_attested,
            "observed_at": obs.isoformat()}


def normalize_prospective_shadow_events(events):
    """Translate only independently quote-attested discovery events to unified input.

    Historical qualification is deliberately NOT asserted. Validation is repeated
    by _event_valid against the current publisher's exact game, side and quote.
    """
    import pandas as pd
    if not isinstance(events, pd.DataFrame) or events.empty:
        return pd.DataFrame()
    required = {"Game_ID", "Game_Start_UTC", "Source_Family", "Signal_Family",
                "Signal_Side", "Observed_At_UTC", "Observed_Quote_UTC",
                "Observed_Book", "Observed_Spread"}
    if not required.issubset(events.columns):
        return pd.DataFrame()
    out = events.copy()
    out["Direction_Verified"] = True
    out["Provenance_Status"] = "PRE_GAME_VERIFIED_LIVE_RECALCULATED"
    out["Registry_Verified"] = False
    out["Qualification_Status"] = "RESEARCH_ONLY"
    out["OOS_Status"] = "NOT_CERTIFIED"
    out["Qualified_At_UTC"] = None
    out["Evidence_Family_Key"] = out.apply(
        lambda x: _str(x.get('Evidence_Family_Key')) or
        ('RESEARCH_SOURCE::' + _str(x.get('Source_Family')).upper()), axis=1)
    return out


def _independent_keys(evidence):
    """Count independent evidence components, not mechanisms or raw families.

    Ordinary brains each receive one vote. Miner systems receive a separate
    vote only for independently frozen underlying family keys. Any reused
    evidence-family key across sources joins those sources into one component.
    """
    if not evidence: return ()
    parents = list(range(len(evidence)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(i,j):
        a,b=root(i),root(j)
        if a!=b:parents[b]=a
    seen_dep={};seen_source={}
    for idx,e in enumerate(evidence):
        dep=e["dependency"];family=e["family"]
        source=(family+"::"+dep) if family=="SYSTEM_MINER" else family
        if dep in seen_dep:union(idx,seen_dep[dep])
        if source in seen_source:union(idx,seen_source[source])
        seen_dep[dep]=idx;seen_source[source]=idx
    components={}
    for idx,e in enumerate(evidence):
        components.setdefault(root(idx),[]).append(e["dependency"])
    return tuple(sorted(min(keys) for keys in components.values()))


def _certs(doc, *, now=None):
    """Only sealed *betting return* certificates, not predictive probabilities."""
    from ncaaf_v55_validated_edge import load_edge_certificates
    return load_edge_certificates(doc, now=now)


def _edge_matches(cert, evs, now):
    """An OOS certificate binds exact brain/system/dependency and predates live signal."""
    from ncaaf_v55_validated_edge import component
    selected = {}
    frozen = _dt(cert.get('frozen_at_utc'))
    if not frozen or frozen > now: return None
    expected = cert.get('component_dependencies') or {}
    for e in evs:
        k = component(e['family'], e['system_id'])
        obs = _dt(e.get('observed_at'))
        if k in expected and e['dependency'] == expected[k] and obs and frozen <= obs and (
                e['family']=='CORE' or e.get('edge_price_attested') is True):
            selected[k] = e
    return [selected[k] for k in cert['component_ids']] if len(selected)==len(cert['component_ids']) else None


def evaluate_v55_unified(board, events, *, miner_library=None, certification=None, now=None, live_feature_rows=None):
    """Return one V5.5 board row/game and its single unified authority audit.

    The existing V5.5 publisher remains the ONLY authority publisher.
    Independent brains may originate even when CORE has no selected team.
    Current implementation is SPREADS; H2H/totals Miner rules remain loaded
    and reported for their corresponding independently priced market engine.
    """
    import pandas as pd
    import ncaaf_v55_board as legacy
    from ncaaf_v55_any_brain_candidates import build_two_sided_scan

    if not isinstance(board, pd.DataFrame): raise TypeError("V55_BOARD_NOT_DATAFRAME")
    now = _dt(now) if now is not None else datetime.now(timezone.utc)
    if not now: raise ValueError("BAD_EVALUATION_CLOCK")
    miner_rules = verify_miner_library(miner_library) if miner_library is not None else {}
    certs = _certs(certification, now=now)
    signals = events.to_dict("records") if isinstance(events, pd.DataFrame) else []
    # Clones BOTH sides from the exact paired executable quote, not from a
    # CORE-selected-team lookup.
    sides = build_two_sided_scan(board, now=now)
    # Execute only those new frozen Miner predicates that can be evaluated
    # honestly from current original same-book prices. CFBD/PT/Pathi atoms
    # require a separate observed-at pregame feature provider and remain held;
    # no rule receives a fabricated certificate or authority permission.
    from ncaaf_v55_live_frozen_miner_bridge import evaluate_quote_atoms
    new_miner_events, new_miner_audit = evaluate_quote_atoms(sides, miner_rules, now=now)
    if not new_miner_events.empty:
        signals.extend(new_miner_events.to_dict('records'))
    # Full atom-catalog execution uses the EXACT frozen 2013-25 atom producer.
    # It is research only until original-vintage OOS certificates exist.
    from ncaaf_v55_live_feature_handoff import evaluate_verified_live_features
    feature_events, feature_audit = evaluate_verified_live_features(
        sides, live_feature_rows, miner_library, now=now)
    if not feature_events.empty:
        signals.extend(feature_events.to_dict('records'))
    # A rule can fire through both quote atoms and full-feature provider.
    # Count it once per game/side/price so duplicate evaluators never increase
    # evidence strength or change certification outcomes.
    unique = {}
    for ev in signals:
        key = (_str(ev.get('Game_ID')), _str(ev.get('Source_Family')).upper(),
               _str(ev.get('Signal_Family')), _team(ev.get('Signal_Side')),
               _str(ev.get('Observed_Book')).lower(), _num(ev.get('Observed_Spread')))
        current = unique.get(key)
        if current is None or (_truth(ev.get('Registry_Verified')) and
                               not _truth(current.get('Registry_Verified'))):
            unique[key] = ev
    signals = list(unique.values())
    out = board.copy()
    for key, val in {
        "V55_Bet_Authority": 0, "V55_Bet_Decision": "PASS",
        "V55_Authority_State": None, "V55_Authority_Policy_Id": None,
        "V55_Authority_Hold_Reason": "NO_VALIDATED_BETTING_EDGE",
        "V55_Bet_Key": None, "V55_Policy_EV": None, "V55_Policy_Win_Prob": None,
        "V55_Policy_Push_Prob": None, "V55_Policy_Lose_Prob": None,
        "V55_Policy_Edge": None, "V55_Policy_Model_Edge_Points": None,
        "V55_Policy_Quote_Age_Min": None, "V55_Policy_Stake_Units": None,
        "V55_Qualified_Evidence_Detail_JSON": "[]", "Confirm_Count": 0,
        "Oppose_Count": 0, "V55_Independent_Support_Keys": None,
        "V55_Independent_Conflict_Keys": None, "V55_Active_System_Names": None,
        "V55_Unified_Candidates_JSON": "[]",
        "V55_Validated_Edge_ROI": None,
        "V55_Validated_Edge_Lower_ROI": None,
        "V55_Validated_Edge_Historical_ROI": None,
        "V55_Validated_Edge_Sample": None,
        "V55_Validated_Edge_Certificate_ID": None,
        "V55_Unified_Brains_Seen": None, "V55_Unified_Research_State": "NO_PAIRED_PRICE",
    }.items(): out[key] = val
    audit = {
        "v55_authority_policy_id": POLICY_ID,
        "v55_authority_policy_status": POLICY_STATUS,
        "v55_authority_probability_basis": "NOT_USED__VALIDATED_OUT_OF_SAMPLE_BETTING_RETURNS",
        "v55_edge_certification_schema": "V55-VALIDATED-BETTING-EDGE-R1",
        "v55_edge_certification_method": "ATTESTED_REALIZED_OOS_ROI_REPRICED_AT_LIVE_BOOKMAKER_ODDS",
        "v55_edge_certification_rejected": len((certification or {}).get('rejected_selections',{})) if isinstance(certification,dict) else 0,
        "v55_authorized_bet_rows": 0, "v55_strong_bet_rows": 0,
        "v55_candidate_rows": 0, "v55_authority_evaluable": 0,
        "v55_bet_authority": 0,
        "v55_policy_status": "NO_VALIDATED_BETTING_EDGE_BETS",
        "v55_decision_ladder": "ANY_BRAIN_ORIGIN__ATTESTED_OOS_RETURN_CERTIFICATE__CONFLICT_VETO__LIVE_REPRICED_ROI",
        "v55_automatic_wager_execution": False,
        "v55_v1_selector_used": False,
        "v55_research_historical_seasons": "2013-2025_PRE_2026_ONLY",
        "v55_unified_miner_v2_rule_count": len(miner_rules),
        # LIVE is an observation/execution state, not an OOS certification.
        "v55_unified_miner_v2_execution_mode": "LIVE_MONITORING" if miner_rules else "NO_FROZEN_LIBRARY",
        "v55_unified_miner_v2_live_tracking_rules": sum(
            r.get("live_tracking_enabled") is True for r in miner_rules.values()),
        "v55_unified_miner_v2_live_bet_eligible_rules": sum(
            _truth(r.get("historical_oos_certified")) and
            _truth(r.get("live_authority_eligible")) for r in miner_rules.values()),
        "v55_unified_miner_v2_live_markets": dict(
            spreads=sum(r.get("market")=="spreads" for r in miner_rules.values()),
            h2h=sum(r.get("market")=="h2h" for r in miner_rules.values()),
            totals=sum(r.get("market")=="totals" for r in miner_rules.values())),
        "v55_unified_miner_v2_market_counts": {"spreads":116, "h2h":120, "totals":109} if miner_rules else {},
        "v55_unified_miner_legacy_admitted": False,
        "v55_unified_certified_combinations": len(certs),  # compatibility: includes single-brain edge certificates
        "v55_validated_edge_certificates": len(certs),
        "v55_validated_edge_single_brain_certificates": sum(len(c)==1 for c in certs),
        "v55_validated_edge_multi_brain_certificates": sum(len(c)>=2 for c in certs),
        "v55_unified_input_qualified_event_rows": len(signals),
        "v55_unified_side_rows": len(sides),
        "v55_unified_h2h_totals_authority": 0,
        "v55_unified_candidate_sides": 0,
        "v55_unified_brain_origin_games": 0,
        "v55_unified_hold_reasons": defaultdict(int),
        "v55_unified_live_qualified_by_brain": {},
        "v55_unified_live_research_by_brain": {},
        "v55_unified_new_miner_v2_rule_ids_fired": 0,
        "v55_unified_miner_v2_live_executor": new_miner_audit,
        "v55_unified_full_atom_executor": feature_audit,
        "v55_unified_combined_signal_rows_deduplicated": len(signals),
        "v55_unified_full_atom_research_events": len(feature_events),
        "v55_unified_certification_state": (
            "ACCEPTED_COMBINATION_CERTIFICATES" if certs
            else "MISSING_INDEPENDENT_ORIGINAL_PREGAME_VALIDATED_BETTING_EDGE_CERTIFICATES"),
        "v55_unified_upstream_event_rows": max(0, len(signals)-len(new_miner_events)-len(feature_events)),
        "v55_unified_quote_atom_research_events": len(new_miner_events),
        "v55_unified_miner_v2_evidence_state": (
            "LIVE_ACTIVE_OOS_CERTIFIED_SOME_RULES" if any(
                _truth(r.get("historical_oos_certified")) and
                _truth(r.get("live_authority_eligible")) for r in miner_rules.values())
            else "LIVE_ACTIVE_RESEARCH_SIGNALS_NO_OOS_CERTIFICATION"
            if miner_rules else "FROZEN_LIBRARY_UNAVAILABLE"),
    }
    qualified_by_brain = defaultdict(set)
    research_by_brain = defaultdict(set)
    fired_new_miner_ids = set()
    candidates = []
    side_results = defaultdict(list)
    by_gid = defaultdict(list)
    for ev in signals: by_gid[_str(ev.get("Game_ID"))].append(ev)
    for _, row in sides.iterrows():
        gid = _str(row.get("Game_ID")); selected = _team(row.get("Selected_Team"))
        kick, q = _dt(row.get("Game_Start_UTC")), _dt(row.get("Backend_Quote_Timestamp"))
        quote_age = (now-q).total_seconds()/60 if q else None
        spread, odds = _num(row.get("Current_Spread")), _num(row.get("Odds_American"))
        net = _prob_net(odds)
        if not gid or not selected or not kick or not q or not q <= now < kick or not kick <= now + pd.Timedelta(days=7) or quote_age is None or quote_age > MAX_QUOTE_MIN or net is None or spread is None:
            audit["v55_unified_hold_reasons"]["INVALID_LIVE_PAIRED_PRICE"] += 1
            continue
        evs = []
        for raw in by_gid.get(gid, []):
            r = _event_valid(raw, row, selected, now, miner_rules)
            if r is not None:
                # Old registry fields are not decision authority; only the
                # new validated-ROI manifest may certify a live signal.
                evs.append({**r, 'certified': False})
                fingerprint = (gid, r['system_id'], r['team'], r['observed_at'])
                research_by_brain[r['family']].add(fingerprint)
                if r['family'] == 'SYSTEM_MINER': fired_new_miner_ids.add(r['system_id'])
        side = _str(row.get('V55_Brain_Scan_Side'))
        home_away, oriented = legacy._matchup_teams(row)
        if oriented:
            # Preserve independent CORE opposition even when another brain
            # originated the candidate on the opposite side.
            for role, team in zip(('HOME','AWAY'),home_away):
                raw_core_ev = _num(row.get('V55_Home_Raw_EV' if role=='HOME' else 'V55_Away_Raw_EV'))
                if raw_core_ev is not None and raw_core_ev>0:
                    evs.append({'family':'CORE','system_id':'CORE_V55',
                                'dependency':'CORE_V55','team':team,'certified':False,
                                'edge_price_attested':True,
                        'observed_at':q.isoformat()})
        all_origin = sorted({e['family'] for e in evs if e['team']==selected})
        same = [e for e in evs if e['team']==selected]
        opposite = [e for e in evs if e['team']!=selected]
        # Iterate over predeclared single or combined betting edges. We do
        # NOT fit/estimate a new predictive win probability from live evidence.
        matching = []
        opposed = []
        from ncaaf_v55_validated_edge import current_edge
        for cert in certs.values():
            other = _edge_matches(cert, opposite, now)
            if other:
                opposed.append((cert,other))
            chosen = _edge_matches(cert, same, now)
            if chosen:
                returns = current_edge(cert, odds, spread, row.get('Backend_Quote_Book'))
                if returns:
                    matching.append((cert,chosen,returns))
        # A live research observation is *not* counted as an independent
        # qualified vote without the original-vintage certificate.
        supported = {(e['family'],e['system_id']) for _, ss, _ in matching for e in ss}
        keys = sorted({e['dependency'] for e in same if (e['family'],e['system_id']) in supported})
        opp_keys = sorted({e['dependency'] for _, ss in opposed for e in ss})
        for e in same:
            if (e['family'],e['system_id']) in supported:
                qualified_by_brain[e['family']].add((gid,e['system_id'],e['team'],e['observed_at']))
        snapshot = {'game_id':gid,'team':selected,'side':side,'origins':all_origin,
            'qualified_family_keys':keys,'opposing_family_keys':opp_keys,
            'research_system_ids':sorted({e['system_id'] for e in same if (e['family'],e['system_id']) not in supported})[:30],
            'certified_system_ids':sorted({e['system_id'] for e in same if (e['family'],e['system_id']) in supported})[:30],
            'raw_signal_count':len(evs),'certified_signal_count':len(supported),
            'price_book':_str(row.get('Backend_Quote_Book')),
            'price_spread':spread,'price_odds':odds,'quote_utc':q.isoformat(),
            'bet_authority':0}
        side_results[gid].append(snapshot)
        if all_origin:audit['v55_unified_candidate_sides']+=1
        if not all_origin:continue
        if opposed:
            snapshot['hold']='CERTIFIED_OPPOSITION_VETO'
            audit['v55_unified_hold_reasons']['CERTIFIED_OPPOSITION_VETO']+=1
            continue
        if not matching:
            snapshot['hold']='NO_MATCHED_VALIDATED_BETTING_EDGE_FOR_LIVE_SYSTEMS_AND_PRICE'
            audit['v55_unified_hold_reasons']['NO_MATCHED_VALIDATED_BETTING_EDGE_FOR_LIVE_SYSTEMS_AND_PRICE']+=1
            continue
        if quote_age > BET_MAX_QUOTE_MIN:
            snapshot['hold']='PRICE_TOO_OLD_FOR_BET_AUTHORITY'
            audit['v55_unified_hold_reasons']['PRICE_TOO_OLD_FOR_BET_AUTHORITY']+=1
            continue
        audit['v55_authority_evaluable']+=1
        # Selection is by conservative ROI lower bound, not by forecast p(win).
        matching.sort(key=lambda x:(-x[2]['roi_lower'],-x[2]['roi'],-len(x[1]),x[0]['certificate_id']))
        certificate, support, returns = matching[0]
        ev, min_ev = returns['roi'], returns['roi_lower']
        cert_keys = tuple(e['dependency'] for e in support)
        snapshot.update({'estimated_ev':ev,'conservative_ev':min_ev,
            'validated_historical_roi':returns['historical_realized_roi'],
            'certificate_id':certificate['certificate_id'],
            'oos_games':certificate['oos_unique_games'],
            'validated_roi_not_probability':True})
        if ev < MIN_EV or min_ev<=0:
            snapshot['hold']='LIVE_PRICE_REPRICED_EDGE_NOT_POSITIVE'
            audit['v55_unified_hold_reasons']['LIVE_PRICE_REPRICED_EDGE_NOT_POSITIVE']+=1
            continue
        candidates.append((min_ev,ev,len(set(cert_keys)),gid,selected,side,spread,odds,quote_age,
            certificate,support,tuple(sorted(set(cert_keys))),tuple(opp_keys)))
    # Only one side of each physical game; cap to the existing V55 top 3.
    candidates.sort(key=lambda x:(-x[0],-x[1],-x[2],x[3],x[4]))
    # Retain the established operator emergency stop even after replacing the
    # previous authority selector. OFF blocks *every* recommendation.
    authority_enabled = os.getenv("NCAAF_V55_BET_AUTHORITY_MODE", "USER_DIRECTED_PROVISIONAL").strip().upper() != "OFF"
    audit["v55_unified_operator_authority_enabled"] = authority_enabled
    seen = set()
    for c in (candidates if authority_enabled else []):
        min_ev,ev,nkeys,gid,selected,side,spread,odds,age,certificate,support,keys,opp_keys = c
        if gid in seen: continue
        seen.add(gid)
        if audit["v55_authorized_bet_rows"] >= MAX_BETS: break
        matches = out.index[out["Game_ID"].astype(str).eq(gid)].tolist()
        if len(matches) != 1: raise RuntimeError("UNIFIED_DUPLICATED_OR_MISSING_PHYSICAL_GAME:"+gid)
        idx = matches[0]
        # Replace the previous CORE-only selected side with the winning ANY-
        # BRAIN side, using its own certified same-book market quote.
        quote_fields = ("Backend_Home_" if side == "HOME" else "Backend_Away_")
        out.at[idx,"Selected_Team"] = selected
        out.at[idx,"V55_Research_Selected_Team"] = selected
        out.at[idx,"V55_Research_Selected_Side"] = side
        out.at[idx,"Current_Spread"] = spread
        out.at[idx,"Odds_American"] = odds
        out.at[idx,"Book"] = out.at[idx,"Backend_Quote_Book"]
        out.at[idx,"V55_Policy_EV"] = ev
        out.at[idx,"V55_Policy_Edge"] = min_ev # conservative out-of-sample ROI, not a probability difference
        out.at[idx,"V55_Policy_Win_Prob"] = None
        out.at[idx,"V55_Policy_Push_Prob"] = None
        out.at[idx,"V55_Policy_Lose_Prob"] = None
        out.at[idx,"V55_Validated_Edge_ROI"] = ev
        out.at[idx,"V55_Validated_Edge_Lower_ROI"] = min_ev
        out.at[idx,"V55_Validated_Edge_Historical_ROI"] = certificate['oos_realized_roi']
        out.at[idx,"V55_Validated_Edge_Sample"] = certificate['oos_unique_games']
        out.at[idx,"V55_Validated_Edge_Certificate_ID"] = certificate['certificate_id']
        out.at[idx,"V55_Policy_Model_Edge_Points"] = None  # not fabricated from CORE
        out.at[idx,"V55_Policy_Quote_Age_Min"] = age
        out.at[idx,"V55_Policy_Stake_Units"] = .5 if nkeys >= 2 else .25
        out.at[idx,"V55_Bet_Authority"] = 1
        out.at[idx,"V55_Bet_Decision"] = "STRONG BET" if nkeys >= 2 else "BET"
        out.at[idx,"V55_Authority_State"] = POLICY_STATUS
        out.at[idx,"V55_Authority_Policy_Id"] = POLICY_ID
        out.at[idx,"V55_Authority_Hold_Reason"] = None
        out.at[idx,"Confirm_Count"] = nkeys
        out.at[idx,"Oppose_Count"] = 0
        out.at[idx,"V55_Independent_Support_Keys"] = "|".join(keys)
        out.at[idx,"V55_Independent_Conflict_Keys"] = ""
        out.at[idx,"V55_Active_System_Names"] = "|".join(sorted(set(e["family"] for e in support)))
        out.at[idx,"V55_Qualified_Evidence_Detail_JSON"] = json.dumps([{
            "source":e["family"], "system_id":e["system_id"], "direction":e["team"],
            "evidence_family_key":e["dependency"], "observed_utc":e["observed_at"],
            "historical_source":"VALIDATED_PREGAME_BETTING_EDGE", "certificate_id":certificate["certificate_id"]}
            for e in support[:20]],sort_keys=True)
        key = "|".join(("NCAAF","V55","SPREADS",gid,_str(out.at[idx,"Game_Start_UTC"]), selected))
        out.at[idx,"V55_Bet_Key"] = hashlib.sha256(key.encode("utf-8")).hexdigest()[:28]
        audit["v55_authorized_bet_rows"] += 1
        audit["v55_strong_bet_rows"] += int(nkeys >= 2)
    for idx, r in out.iterrows():
        gid = _str(r.get("Game_ID"))
        opportunities = side_results.get(gid, [])
        out.at[idx,"V55_Unified_Candidates_JSON"] = json.dumps(opportunities,sort_keys=True)
        origins = sorted({f for s in opportunities for f in s["origins"]})
        out.at[idx,"V55_Unified_Brains_Seen"] = "|".join(origins) if origins else None
        out.at[idx,"V55_Unified_Research_State"] = "EVALUATED_BOTH_SIDES" if opportunities else "NO_ELIGIBLE_PAIRED_SIDE"
        audit["v55_unified_brain_origin_games"] += bool(origins)
        if r.get("V55_Bet_Authority") != 1 and origins:
            out.at[idx,"V55_Bet_Decision"] = "CANDIDATE"
            # Preserve the actual blocking reason from the evaluated game side.
            # A valid research signal is not a certified wager.
            holds = {x.get('hold') for x in opportunities if x.get('origins') and x.get('hold')}
            priority = ('CERTIFIED_OPPOSITION_VETO','LIVE_PRICE_REPRICED_EDGE_NOT_POSITIVE',
                        'NO_MATCHED_VALIDATED_BETTING_EDGE_FOR_LIVE_SYSTEMS_AND_PRICE',
                        'PRICE_TOO_OLD_FOR_BET_AUTHORITY')
            out.at[idx,"V55_Authority_Hold_Reason"] = ('|'.join(h for h in priority if h in holds)
                if holds else 'UNCERTIFIED_OR_NO_POSITIVE_CROSS_BRAIN_EV')
            audit["v55_candidate_rows"] += 1
    audit["v55_bet_authority"] = int(audit["v55_authorized_bet_rows"] > 0)
    audit["v55_policy_status"] = ("DISABLED_BY_OPERATOR" if not authority_enabled
                                 else "VALIDATED_EDGE_BETS_PUBLISHED" if audit["v55_bet_authority"]
                                 else "NO_VALIDATED_BETTING_EDGE_BETS")
    audit["v55_unified_hold_reasons"] = dict(audit["v55_unified_hold_reasons"])
    audit["v55_unified_live_qualified_by_brain"] = {x:len(v) for x,v in sorted(qualified_by_brain.items())}
    audit["v55_unified_live_research_by_brain"] = {x:len(v) for x,v in sorted(research_by_brain.items())}
    audit["v55_unified_new_miner_v2_rule_ids_fired"] = len(fired_new_miner_ids)
    return out, audit


def load_gcs_artifacts(storage_client, bucket_name):
    """Independently load pinned Miner library and validated ROI certificates.

    Missing/unavailable Miner must not prevent a certified PT, PATHI, BIG_AL,
    STAT, MARKET_RICH or CORE edge from receiving its own authority check.
    Missing edge certificates never turn legacy probabilistic evidence into
    betting authority.
    """
    import logging
    bucket = storage_client.bucket(bucket_name)
    cert = None
    try:
        cert_blob = bucket.blob("research/ncaaf/v55/unified_combined_authority/validated_edge_certificates_latest.json")
        cert = json.loads(cert_blob.download_as_text()) if cert_blob.exists() else None
    except Exception as exc:
        logging.warning('[NCAAF-V55-EDGE-LOAD] certificate_unavailable=%s:%s',type(exc).__name__,str(exc)[:160])
    lib = None
    try:
        pointer = json.loads(bucket.blob(
            "research/ncaaf/v55/unified_executable_miner_library_v2/current_library.json").download_as_text())
        if pointer.get("rule_count") != 345 or pointer.get("bet_authority") != 0:
            raise RuntimeError("MINER_V2_POINTER_INTEGRITY_FAILURE")
        uri = _str(pointer.get("latest_uri"))
        root = "gs://"+bucket_name+"/research/ncaaf/v55/unified_executable_miner_library_v2/"
        if not uri.startswith(root) or not uri.endswith("/library.json"):
            raise RuntimeError("MINER_V2_URI_NOT_IN_AUTHORIZED_PREFIX")
        payload = bucket.blob(uri[len("gs://"+bucket_name+"/"):]).download_as_bytes()
        if hashlib.sha256(payload).hexdigest() != _str(pointer.get("sha256")):
            raise RuntimeError("MINER_V2_LIBRARY_SHA256_MISMATCH")
        lib = json.loads(payload)
        verify_miner_library(lib)
    except Exception as exc:
        logging.warning('[NCAAF-V55-EDGE-LOAD] miner_unavailable=%s:%s; non-Miner edge authority remains independent',
                        type(exc).__name__,str(exc)[:160])
    return lib, cert
