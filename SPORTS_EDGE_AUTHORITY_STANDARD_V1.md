# Sports Edge Authority Standard V1

This is the reusable betting-decision architecture for every sport in the project.
It was extracted from the NCAAF Production V1 architecture after the NFL generic
second-stage Betting Engine V1 failed its historical authority gate.

## Permanent design rules

1. **Predict first, authorize bets separately.** Frozen target-specific models estimate fair value. They never become betting authority merely because they predict outcomes better than a baseline.
2. **Research conditional edge.** Edge research asks when model/market disagreement, statistics, systems, and market behavior are reliable—not whether one giant classifier can relearn the entire sport.
3. **One mechanism family = one vote.** Threshold variants, aliases, nested rules, and correlated systems are collapsed before authority is counted.
4. **Discovery chooses; confirmation validates.** The representative/threshold/direction is fixed using discovery data. Confirmation cannot re-optimize it.
5. **Independent mechanisms may strengthen; correlated variants may not.** A stronger action requires genuinely different evidence families and separate confirmation of the multi-mechanism state.
6. **Conflicts fail closed.** Opposing validated mechanisms produce `PASS — CONFLICT`.
7. **MODEL ONLY is a valid result.** A market is never forced to produce bets.
8. **Prospective evidence stays separate.** Historical confirmation can open conditional authority; live prospective performance is tracked independently and controls later promotion/governance.
9. **Production is frozen; research is continuous.** Research challengers may evolve. Production changes only through an explicit promotion contract.
10. **Sport adapters change; the framework does not.** Football, basketball, baseball, tennis, etc. may use different statistics/systems/market microstructure, but these authority rules remain shared.

## Standard action vocabulary

- `MODEL ONLY` — prediction exists; no validated edge family is active.
- `PLAY` — one validated independent mechanism (or multiple without validated escalation) supports one direction.
- `STRONG PLAY` — at least two independent validated mechanisms agree **and** the sport/market's multi-mechanism confirmation gate passed.
- `PASS — CONFLICT` — validated mechanisms disagree.
- `EDGE — NO EXEC QUOTE` — authority exists but an executable/current market quote is unavailable.

## NCAAF precedent

NCAAF Production V1 already follows this structure: separate frozen Spread/H2H/Totals models; a frozen Spread edge-family selector; family-collapsed Totals systems; H2H model-only when no repeatable edge passed; independent-mechanism escalation; conflict = pass.

## NFL V2 adaptation

NFL Edge Authority V2 keeps Betting Engine V1 as a benchmark/shadow and tests interpretable mechanism families: FAIR_VALUE, STAT_SELECTOR, SYSTEM, and MARKET_CONFIRMATION. 2021–2023 is discovery, 2024–2025 is confirmation, and 2026 remains prospective.
