"""Display-only V5.5 NCAAF betting board from the verified backend publication.

No Miner rerun, no model scoring, no odds fetching and no authority promotion.
The complete published BigQuery snapshot is verified by ncaaf_v55_board.
"""
from __future__ import annotations
import math


def _string(value):
    if value is None:
        return ""
    try:
        import pandas as pd
        empty = pd.isna(value)
        if isinstance(empty, bool) and empty:
            return ""
    except (ValueError, TypeError):
        pass
    value = str(value).strip()
    return "" if value.lower() in {"nan", "none", "null", "nat", "<na>"} else value


def _float(value):
    try:
        n = float(_string(value).replace("%", ""))
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def _auth(value):
    return _string(value).lower() in {"1", "1.0", "true"}


def published_game_rows(weekly, *, now=None):
    """One existing backend game row per game. Never invent a side or a bet."""
    import pandas as pd
    if not isinstance(weekly, pd.DataFrame):
        raise TypeError("WEEKLY_MUST_BE_DATAFRAME")
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tz is None else now.tz_convert("UTC")
    out = []
    for _, r in weekly.iterrows():
        kick = pd.to_datetime(r.get("Game_Start_UTC"), errors="coerce", utc=True)
        if pd.notna(kick) and kick <= now:
            continue
        quote = _string(r.get("Quote_State"))
        fresh = quote.startswith("FRESH_")
        book = _string(r.get("Backend_Quote_Book"))
        selected = _string(r.get("Selected_Team"))
        if selected.upper() in {"NO CORE PICK", "NONE", "—"}:
            selected = ""
        original_decision = _string(r.get("V55_Bet_Decision"))
        original_auth = _auth(r.get("V55_Bet_Authority"))
        spread = _float(r.get("Current_Spread"))
        odds = _float(r.get("Odds_American"))
        valid_bet = (original_auth and original_decision in {"BET", "STRONG BET"}
                     and fresh and bool(selected) and spread is not None
                     and odds is not None and (odds >= 100 or odds <= -100)
                     and bool(book))
        if valid_bet:
            action = original_decision
        elif original_auth and original_decision in {"BET", "STRONG BET"}:
            action = "BLOCKED — PUBLISHED BET NOT PRICE-VERIFIED"
        elif original_decision in {"BET", "STRONG BET"}:
            action = "BLOCKED — NO PUBLISHED BET AUTHORITY"
        else:
            action = original_decision or "NO BET PUBLISHED"
        match = _string(r.get("Matchup"))
        halves = match.split(" @ ", 1)
        away, home = halves if len(halves) == 2 else ("", "")
        home_line = _float(r.get("Backend_Home_Spread"))
        away_line = _float(r.get("Backend_Away_Spread"))
        home_odds = _float(r.get("Backend_Home_Odds_American"))
        away_odds = _float(r.get("Backend_Away_Odds_American"))
        def line_label(team, line, price):
            if not team or line is None:
                return "—"
            return (f"{team} {line:+g}" + (f" ({price:+.0f})" if price is not None else ""))
        ctx = _string(r.get("V55_Unified_Brains_Seen")) or _string(r.get("V55_Active_System_Names"))
        out.append({
            "Game Time (ET)": (kick.tz_convert("America/New_York").strftime("%a %b %d %I:%M %p")
                               if pd.notna(kick) else "—"),
            "Matchup": match or "—",
            "Decision": action,
            "Published Pick": selected or "NO SIDE PUBLISHED",
            "Selected Spread": spread,
            "Selected Odds": odds,
            "Home Quote": line_label(home, home_line, home_odds),
            "Away Quote": line_label(away, away_line, away_odds),
            "Sportsbook": book or "—",
            "Policy EV": _float(r.get("V55_Policy_EV")) if valid_bet else None,
            "Authority": "BACKEND AUTHORIZED" if valid_bet else "NOT AUTHORIZED",
            "Brain Evidence": ctx or "NOT PUBLISHED",
            "Miner Evidence": _string(r.get("V55_Raw_Miner")) or _string(r.get("SYSTEM_MINER")) or "—",
            "Qualified Support": _string(r.get("Confirm_Families")) or "—",
            "Conflicts": _string(r.get("Oppose_Families")) or "—",
            "Hold Reason": "—" if valid_bet else (_string(r.get("V55_Authority_Hold_Reason")) or "NOT AUTHORIZED BY BACKEND"),
            "Quote State": quote or "NOT PUBLISHED",
            "Game ID": _string(r.get("Game_ID")),
            "_sort": 0 if valid_bet else 1,
            "_kick": kick,
        })
    ans = pd.DataFrame(out)
    if len(ans):
        ans = ans.sort_values(["_sort", "_kick"], na_position="last", kind="stable")
        ans = ans.drop(columns=["_sort", "_kick"]).reset_index(drop=True)
    return ans


def render_published_v55_board(st):
    """Render V5.5 at the *existing* NCAAF dashboard entry point.

    Missing publications are errors, not an excuse to run V1 predictions.
    """
    st.subheader("NCAAF V5.5 — Published Betting Board")
    st.caption("Backend-published actions and prices only. Research candidates are not bets.")
    try:
        from ncaaf_v55_board import read_published_latest
        weekly, audit = read_published_latest()
    except Exception as exc:
        st.error("Unable to read the verified V5.5 backend publication. "
                 "The existing fast V1 board has not been substituted.")
        st.code(f"{type(exc).__name__}: {str(exc)[:600]}")
        return

    try:
        import pandas as pd
        if audit.get("status") != "COMPLETE_V55_INDEPENDENT_WEEKLY_SHADOW" or audit.get("production_v1_probabilities_used") is not False:
            st.error("V5.5 publication provenance failed; no betting decisions displayed.")
            st.json({"status": audit.get("status"),
                     "production_v1_probabilities_used": audit.get("production_v1_probabilities_used")})
            return
        if not isinstance(weekly, pd.DataFrame):
            raise TypeError("PUBLISHED_WEEKLY_NOT_DATAFRAME")
        cols = st.columns(4)
        cols[0].metric("Published games", len(weekly))
        cols[1].metric("Fresh priced", int(weekly.get("Quote_State", pd.Series(dtype=str)).fillna("").astype(str).str.startswith("FRESH_").sum()))
        cols[2].metric("Authorized bets", int(audit.get("v55_authorized_bet_rows") or 0))
        cols[3].metric("Published candidates", int(audit.get("v55_candidate_rows") or 0))
        st.caption(f"Published: {audit.get('published_at_utc','—')} · publication: "
                   f"{audit.get('publication_status','—')} · "
                   f"snapshot: {audit.get('snapshot_id','—')}")
        if audit.get("publication_status") == "STALE_MARKET":
            st.error("Backend market snapshot is stale. No prices or bets are current.")

        has_unified = "V55_Unified_Candidates_JSON" in weekly.columns
        candidate_rows = 0
        if has_unified:
            try:
                from ncaaf_v55_unified_ui import build_unified_ui_rows
                _, unified_stats, _ = build_unified_ui_rows(weekly)
                candidate_rows = int(unified_stats.get("candidate_sides", 0))
            except Exception:
                candidate_rows = 0
        if has_unified and candidate_rows:
            try:
                from ncaaf_v55_unified_ui import render_unified_weekly_board
                rendered = render_unified_weekly_board(weekly, audit, st)
            except Exception as exc:
                rendered = False
                st.warning(f"Unified detail renderer unavailable: {type(exc).__name__}: {str(exc)[:220]}")
            if rendered:
                with st.expander("All published game prices and backend decisions", expanded=False):
                    frame = published_game_rows(weekly)
                    st.dataframe(frame, hide_index=True, use_container_width=True) if not frame.empty else st.info("No upcoming games")
                return
        st.warning(("The backend publication has no independently originated candidate records. "
                    if has_unified else "The backend publication lacks V55_Unified_Candidates_JSON. ")
                   + "Showing the actual two-sided sportsbook board and all published brain/hold fields. "
                   "Independent Miner-led candidate sides cannot be invented by the dashboard; "
                   "the backend must publish their candidate records.")
        frame = published_game_rows(weekly)
        if frame.empty:
            st.info("No upcoming V5.5 backend games currently published.")
        else:
            st.dataframe(frame, hide_index=True, use_container_width=True)
        with st.expander("Backend source / coverage audit", expanded=False):
            st.json({k: audit.get(k) for k in (
                "v55_unified_candidate_sides", "v55_unified_artifact_hold",
                "v55_unified_miner_v2_rule_count", "miner_live_bridge",
                "other_brain_registry_status", "v55_authority_policy_status",
                "v55_authorized_bet_rows", "priced_games")})
    except Exception as exc:
        st.error("V5.5 publication was read, but the display could not be built.")
        st.code(f"{type(exc).__name__}: {str(exc)[:600]}")
