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
POLICY_ID = "V55-UNIFIED-SEVEN-BRAIN-CERTIFIED-20261104-R1"
POLICY_STATUS = "CERTIFIED_COMBINED_BRAIN_AUTHORITY"
MIN_EV = .02
MAX_BETS = 3
MAX_QUOTE_MIN = 90


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
    return {x["rule_id"]: x for x in rules}


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
            "observed_at": obs.isoformat()}


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


def _certs(doc):
    """Only accept independently sealed 2024-25 OOS combination certificates."""
    if not isinstance(doc, dict) or doc.get("schema") != "V55-COMBINED-OOS-CERTIFICATE-R1": return {}
    if doc.get("status") != "PASS_CERTIFIED_ORIGINAL_PREGAME_VINTAGE": return {}
    if not _truth(doc.get("original_pregame_vintage_certified")) or not _truth(doc.get("2026_excluded_from_selection")):
        return {}
    found = {}
    for c in doc.get("combinations") or []:
        if c.get("market") != "spreads" or not _truth(c.get("frozen_before_oos")):
            continue
        if not _truth(c.get("out_of_sample_pass")) or not _truth(c.get("original_quote_provenance_verified")):
            continue
        n = _num(c.get("oos_unique_games")); prob = _num(c.get("prob_win"))
        low = _num(c.get("prob_win_lower_95")); push = _num(c.get("prob_push"))
        if push is None: push = 0.0
        if n is None or n < 100 or prob is None or low is None or not 0 <= low <= prob <= 1-push:
            continue
        keys = tuple(sorted(set(_str(x) for x in (c.get("component_keys") or []) if _str(x))))
        if not keys or keys in found: continue
        found[keys] = {"n": int(n), "p": prob, "lower": low, "push": push,
                       "certificate_id": _str(c.get("certificate_id")),
                       "source_sha": _str(c.get("source_sha256"))}
    return found


def evaluate_v55_unified(board, events, *, miner_library=None, certification=None, now=None):
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
    certs = _certs(certification)
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
    out = board.copy()
    for key, val in {
        "V55_Bet_Authority": 0, "V55_Bet_Decision": "PASS",
        "V55_Authority_State": None, "V55_Authority_Policy_Id": None,
        "V55_Authority_Hold_Reason": "NO_CERTIFIED_COMBINED_EDGE",
        "V55_Bet_Key": None, "V55_Policy_EV": None, "V55_Policy_Win_Prob": None,
        "V55_Policy_Push_Prob": None, "V55_Policy_Lose_Prob": None,
        "V55_Policy_Edge": None, "V55_Policy_Model_Edge_Points": None,
        "V55_Policy_Quote_Age_Min": None, "V55_Policy_Stake_Units": None,
        "V55_Qualified_Evidence_Detail_JSON": "[]", "Confirm_Count": 0,
        "Oppose_Count": 0, "V55_Independent_Support_Keys": None,
        "V55_Independent_Conflict_Keys": None, "V55_Active_System_Names": None,
        "V55_Unified_Candidates_JSON": "[]",
        "V55_Unified_Brains_Seen": None, "V55_Unified_Research_State": "NO_PAIRED_PRICE",
    }.items(): out[key] = val
    audit = {
        "v55_authority_policy_id": POLICY_ID,
        "v55_authority_policy_status": POLICY_STATUS,
        "v55_authority_probability_basis": "V55_UNIFIED_CERTIFIED_CROSS_BRAIN_OOS",
        "v55_authorized_bet_rows": 0, "v55_strong_bet_rows": 0,
        "v55_candidate_rows": 0, "v55_authority_evaluable": 0,
        "v55_bet_authority": 0,
        "v55_policy_status": "NO_CERTIFIED_COMBINED_BETS",
        "v55_decision_ladder": "ANY_QUALIFIED_BRAIN_ORIGIN__INDEPENDENCE_CONFLICT_VETO__CERTIFIED_PRICE_EV",
        "v55_automatic_wager_execution": False,
        "v55_v1_selector_used": False,
        "v55_research_historical_seasons": "2013-2025_PRE_2026_ONLY",
        "v55_unified_miner_v2_rule_count": len(miner_rules),
        "v55_unified_miner_v2_market_counts": {"spreads":116, "h2h":120, "totals":109} if miner_rules else {},
        "v55_unified_miner_legacy_admitted": False,
        "v55_unified_certified_combinations": len(certs),
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
        "v55_unified_upstream_event_rows": len(signals)-len(new_miner_events),
        "v55_unified_quote_atom_research_events": len(new_miner_events),
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
                evs.append(r)
                fingerprint = (gid, r["system_id"],r["team"],r["observed_at"])
                (qualified_by_brain if r["certified"] else research_by_brain)[r["family"]].add(fingerprint)
                if r["family"] == "SYSTEM_MINER": fired_new_miner_ids.add(r["system_id"])
        # CORE may originate on EITHER side, but its raw uncalibrated margin
        # is a candidate only; it can never create a certified combo itself.
        side = _str(row.get("V55_Brain_Scan_Side"))
        raw_core_ev = _num(row.get("V55_Home_Raw_EV" if side == "HOME" else "V55_Away_Raw_EV"))
        core_origin = raw_core_ev is not None and raw_core_ev > 0
        (home, away), ok = legacy._matchup_teams(row)
        support = [e for e in evs if e["team"] == selected and e["certified"]]
        oppose = [e for e in evs if e["team"] != selected and e["certified"]]
        all_origin = sorted({e["family"] for e in evs if e["team"] == selected} | ({"CORE"} if core_origin else set()))
        # Collapses duplicate mechanisms by source-family + dependency key.
        # Different systems sharing one dependency are one support vote.
        keys = _independent_keys(support)
        opp_keys = _independent_keys(oppose)
        snapshot = {"game_id":gid,"team":selected,"side":side,"origins":all_origin,
            "qualified_family_keys":list(keys),"opposing_family_keys":list(opp_keys),
            "research_system_ids":sorted({e['system_id'] for e in evs if not e['certified'] and e['team']==selected})[:30],
            "certified_system_ids":sorted({e['system_id'] for e in support})[:30],
            "raw_signal_count":len(evs),"certified_signal_count":len(support),
            "price_book":_str(row.get("Backend_Quote_Book")),
            "price_spread":spread,"price_odds":odds,
            "quote_utc":q.isoformat(),"bet_authority":0}
        side_results[gid].append(snapshot)
        if all_origin: audit["v55_unified_candidate_sides"] += 1
        if not all_origin: continue
        if not support:
            snapshot['hold'] = 'NO_CERTIFIED_DIRECTIONAL_BRAIN'
            audit["v55_unified_hold_reasons"]["NO_CERTIFIED_DIRECTIONAL_BRAIN"] += 1; continue
        if oppose:
            snapshot["hold"] = "CERTIFIED_OPPOSITION_VETO"
            audit["v55_unified_hold_reasons"]["CERTIFIED_OPPOSITION_VETO"] += 1; continue
        certificate = certs.get(keys)
        if certificate is None:
            snapshot["hold"] = "MISSING_MATCHED_COMBINATION_OOS_CERTIFICATE"
            audit["v55_unified_hold_reasons"]["MISSING_MATCHED_COMBINATION_OOS_CERTIFICATE"] += 1; continue
        audit["v55_authority_evaluable"] += 1
        p, low, push = certificate["p"], certificate["lower"], certificate["push"]
        ev = p*net-(1-p-push)
        min_ev = low*net-(1-low-push)
        snapshot.update({"estimated_ev":ev,"conservative_ev":min_ev,
            "certificate_id":certificate["certificate_id"],"oos_games":certificate["n"]})
        if ev < MIN_EV or min_ev <= 0:
            snapshot["hold"] = "NO_POSITIVE_CERTIFIED_NET_VALUE"
            audit["v55_unified_hold_reasons"]["NO_POSITIVE_CERTIFIED_NET_VALUE"] += 1; continue
        candidates.append((min_ev,ev,len(keys),gid,selected,side,spread,odds,quote_age,certificate,
            support,keys,opp_keys))
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
        out.at[idx,"V55_Policy_Edge"] = ev/(1+_prob_net(odds))
        out.at[idx,"V55_Policy_Win_Prob"] = certificate["p"]
        out.at[idx,"V55_Policy_Push_Prob"] = certificate["push"]
        out.at[idx,"V55_Policy_Lose_Prob"] = 1-certificate["p"]-certificate["push"]
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
            "historical_source":"FROZEN_V55_REGISTRY_MATCH", "certificate_id":certificate["certificate_id"]}
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
            priority = ('CERTIFIED_OPPOSITION_VETO','MISSING_MATCHED_COMBINATION_OOS_CERTIFICATE',
                        'NO_POSITIVE_CERTIFIED_NET_VALUE','NO_CERTIFIED_DIRECTIONAL_BRAIN')
            out.at[idx,"V55_Authority_Hold_Reason"] = ('|'.join(h for h in priority if h in holds)
                if holds else 'UNCERTIFIED_OR_NO_POSITIVE_CROSS_BRAIN_EV')
            audit["v55_candidate_rows"] += 1
    audit["v55_bet_authority"] = int(audit["v55_authorized_bet_rows"] > 0)
    audit["v55_policy_status"] = ("DISABLED_BY_OPERATOR" if not authority_enabled
                                 else "CERTIFIED_UNIFIED_BETS_PUBLISHED" if audit["v55_bet_authority"]
                                 else "NO_CERTIFIED_COMBINED_BETS")
    audit["v55_unified_hold_reasons"] = dict(audit["v55_unified_hold_reasons"])
    audit["v55_unified_live_qualified_by_brain"] = {x:len(v) for x,v in sorted(qualified_by_brain.items())}
    audit["v55_unified_live_research_by_brain"] = {x:len(v) for x,v in sorted(research_by_brain.items())}
    audit["v55_unified_new_miner_v2_rule_ids_fired"] = len(fired_new_miner_ids)
    return out, audit


def load_gcs_artifacts(storage_client, bucket_name):
    """Load pinned Miner rules and optional separately certified combo file."""
    bucket = storage_client.bucket(bucket_name)
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
    cert_blob = bucket.blob("research/ncaaf/v55/unified_combined_authority/certified_combinations_latest.json")
    cert = json.loads(cert_blob.download_as_text()) if cert_blob.exists() else None
    return lib, cert
