"""Read-only NCAAF V5.5 unified candidate UI — no model inference or bet promotion.

Consumes the *published* V5.5 weekly DataFrame and its publisher audit. Every
candidate game/side is independently visible even without a CORE Selected_Team.
Only backend-approved, fresh-quoted bets can appear as BET or STRONG BET.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import Any

SIGNAL_FIELDS = (
    "V55_Unified_Candidates_JSON", "V55_Unified_Research_State", "V55_Unified_Brains_Seen"
)
ACTIONS = ("BET", "STRONG BET")


def _str(x: Any) -> str:
    if x is None: return ""
    s = str(x).strip()
    return "" if s.lower() in ("nan", "none", "null", "nat", "<na>") else s


def _number(x):
    try:
        v = float(str(x).replace("%", "").replace("pts", "").strip())
        return v if __import__("math").isfinite(v) else None
    except (TypeError, ValueError): return None


def _candidates(x):
    if isinstance(x, list): val=x
    elif isinstance(x, str) and x.strip():
        try: val=json.loads(x)
        except (ValueError, TypeError): return [], "MALFORMED_CANDIDATES_JSON"
    else: return [], None
    if not isinstance(val, list): return [], "CANDIDATES_NOT_LIST"
    return [q for q in val if isinstance(q, dict)], None


def _utc(x):
    import pandas as pd
    try: return pd.to_datetime(x, utc=True, errors="coerce")
    except Exception: return pd.NaT


def build_unified_ui_rows(weekly, *, now=None):
    """Return presentation rows and diagnostics. Never calculate betting authority.

    Bet actions are accepted ONLY from publisher fields on the physical game row
    with a matching side, valid original price and a FRESH_ quote. A research
    candidate can never inherit an unrelated CORE selection's authorization.
    """
    import pandas as pd
    if not isinstance(weekly, pd.DataFrame): raise TypeError("WEEKLY_MUST_BE_DATAFRAME")
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None: now=now.tz_localize("UTC")
    else: now=now.tz_convert("UTC")
    records=[]
    errors=[]
    seen=set()
    stats={"upcoming_games":0,"games_with_unified_candidate_field":0,
           "candidate_sides":0,"backend_authorized_games":0,
           "displayed_authorized_bets":0,"priced_candidate_sides":0,
           "missing_backend_unified_fields":False}
    has_col="V55_Unified_Candidates_JSON" in weekly.columns
    stats["missing_backend_unified_fields"]=not has_col
    if not has_col: return pd.DataFrame(), stats, errors
    for idx, row in weekly.iterrows():
        start=_utc(row.get("Game_Start_UTC"))
        if pd.notna(start) and start<=now: continue
        stats["upcoming_games"]+=1
        gid=_str(row.get("Game_ID"))
        matchup=_str(row.get("Matchup"))
        candidates,err=_candidates(row.get("V55_Unified_Candidates_JSON"))
        if err: errors.append({"game":gid or matchup,"reason":err})
        if candidates: stats["games_with_unified_candidate_field"]+=1
        quote_state=_str(row.get("Quote_State"))
        fresh=quote_state.startswith("FRESH_")
        published_action=_str(row.get("V55_Bet_Decision")).upper()
        published_side=_str(row.get("Selected_Team"))
        publisher_auth=_str(row.get("V55_Bet_Authority")) in ("1", "1.0", "True", "true")
        if publisher_auth and published_action in ACTIONS:
            stats["backend_authorized_games"]+=1
        # The publisher's two-sided JSON is the source of research candidates.
        # If the backend approves a bet not represented in candidate JSON,
        # still render the backend bet with an explicit missing-evidence warning.
        if publisher_auth and published_action in ACTIONS and not any(
                _str(c.get("team")).casefold()==published_side.casefold() for c in candidates):
            if published_side:
                candidates.append({"team":published_side,
                                   "side":_str(row.get("V55_Research_Selected_Side")),
                                   "origins":[],"hold":"PUBLISHED_BET_MISSING_UNIFIED_CANDIDATE_DETAIL",
                                   "price_spread":row.get("Current_Spread"),
                                   "price_odds":row.get("Odds_American"),
                                   "price_book":row.get("Backend_Quote_Book")})
                errors.append({"game":gid or matchup,"reason":"PUBLISHED_BET_MISSING_UNIFIED_CANDIDATE_DETAIL"})
        for c in candidates:
            team=_str(c.get("team")); side=_str(c.get("side"))
            if not team: errors.append({"game":gid,"reason":"CANDIDATE_MISSING_TEAM"});continue
            key=(gid or matchup,team.casefold(),side.casefold())
            if key in seen: continue
            seen.add(key)
            matched_auth=bool(publisher_auth and published_action in ACTIONS and
                              team.casefold()==published_side.casefold())
            spread=_number(c.get("price_spread"))
            odds=_number(c.get("price_odds"))
            if matched_auth:
                # The authoritative selected-side price is published at game level.
                spread=_number(row.get("Current_Spread"))
                odds=_number(row.get("Odds_American"))
            book=_str(c.get("price_book")) or _str(row.get("Backend_Quote_Book"))
            usable_quote=bool(fresh and spread is not None and odds is not None and book)
            authorized=bool(matched_auth and usable_quote)
            if authorized: stats["displayed_authorized_bets"]+=1
            if usable_quote: stats["priced_candidate_sides"]+=1
            origins=c.get("origins") or []
            if not isinstance(origins, list): origins=[_str(origins)]
            support=c.get("qualified_family_keys") or []
            oppose=c.get("opposing_family_keys") or []
            if not isinstance(support,list):support=[_str(support)]
            if not isinstance(oppose,list):oppose=[_str(oppose)]
            hold=_str(c.get("hold")) or _str(row.get("V55_Authority_Hold_Reason"))
            action=published_action if authorized else "CANDIDATE — NOT AUTHORIZED"
            if not usable_quote:
                action="NO VERIFIED CURRENT PRICE" if not authorized else action
                hold= "MISSING_OR_STALE_SIDE_PRICE" + ("; "+hold if hold else "")
            if matched_auth and not authorized:
                errors.append({"game":gid or matchup,"reason":"BACKEND_BET_WITHOUT_VALID_FRESH_SIDE_QUOTE"})
            ev=_number(row.get("V55_Policy_EV")) if authorized else _number(c.get("estimated_ev"))
            records.append({
                "Game Time (UTC)":start.isoformat() if pd.notna(start) else "—",
                "Matchup":matchup or "—", "Market":"SPREADS",
                "Candidate Team":team,"Side":side or "—",
                "Spread":spread,"Odds":odds,"Book":book or "—",
                "Originating Brains":" | ".join(sorted(set(map(_str, origins)))) or "—",
                "Qualified Support":" | ".join(map(_str,support)) or "—",
                "Conflicts":" | ".join(map(_str,oppose)) or "—",
                "Published Decision":action,
                "Authority": "BACKEND AUTHORIZED" if authorized else "RESEARCH ONLY",
                "EV (published/research)":ev,
                "Hold / Reason": "—" if authorized else (hold or "NO_CERTIFIED_BET_AUTHORITY"),
                "Quote State":quote_state or "—",
                "Game ID":gid,
            })
            stats["candidate_sides"]+=1
    frame=pd.DataFrame.from_records(records)
    if not frame.empty:
        frame["_priority"]=frame["Authority"].eq("BACKEND AUTHORIZED").astype(int)
        frame=frame.sort_values(["_priority","Game Time (UTC)","Matchup"],ascending=[False,True,True],kind="stable").drop(columns="_priority")
    return frame,stats,errors


def render_unified_weekly_board(weekly,audit,st):
    """Render published V55 two-sided opportunities, return True if handled."""
    import pandas as pd
    frame,stats,errors=build_unified_ui_rows(weekly)
    if stats["missing_backend_unified_fields"]:
        st.warning("The published backend board lacks V55_Unified_Candidates_JSON. "
                   "The unified candidate publisher is not connected to this UI snapshot. "
                   "The legacy board is shown below; it cannot display independent Miner origins.")
        return False
    st.markdown("### Unified V5.5 betting board — all originating brains")
    st.caption("Current/upcoming publisher records; one row per independently originated game side. "
               "BET / STRONG BET appears only with published backend authority and a fresh matching-side quote. "
               "Research candidates are not authorized wagers.")
    a,b,c,d=st.columns(4)
    a.metric("Unified candidate sides",stats["candidate_sides"])
    b.metric("Games with candidates",stats["games_with_unified_candidate_field"])
    c.metric("Backend authorized bets",stats["backend_authorized_games"])
    d.metric("Usable displayed bets",stats["displayed_authorized_bets"])
    if not frame.empty:
        st.dataframe(frame,use_container_width=True,hide_index=True)
    else:
        st.info("No upcoming two-sided unified candidates in this backend publication.")
    if errors:
        with st.expander("Publication mismatches / diagnostics",expanded=False):
            st.dataframe(pd.DataFrame(errors),use_container_width=True,hide_index=True)
    if not stats["displayed_authorized_bets"]:
        st.info("No authorized wager is currently published. Candidate signals remain visible "
                "with pricing and certification hold reasons. This UI does not grant bet authority.")
    return True
