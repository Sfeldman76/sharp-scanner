# Sports Edge Authority Standard V1.1

This is the reusable betting-decision architecture for every sport in the project. It was extracted from the NCAAF Production V1 architecture after the NFL generic second-stage Betting Engine V1 failed its historical authority gate.

## Permanent design rules

1. **Predict first, authorize bets separately.** Frozen target-specific models estimate fair value. They never become betting authority merely because they predict outcomes better than a baseline.
2. **Research conditional edge.** Edge research asks when model/market disagreement, statistics, systems, and market behavior are reliable—not whether one giant classifier can relearn the entire sport.
3. **Rich research, small production.** Broad statistical/feature discovery belongs in the research lane. Production backbones remain compact, frozen, and live-reproducible.
4. **STAT is a selector of shared edge, not an automatic CORE rewrite.** Statistical families may predict market residual/error and identify when a frozen fair-value disagreement is trustworthy. Discovery fixes family/model/strength rules; confirmation only validates them.
5. **One mechanism family = one vote.** Threshold variants, aliases, nested rules, and correlated systems are collapsed before authority is counted.
6. **Dependency is audited, not assumed.** Parent/child systems, rules sharing defining conditions, or high-overlap families share one `independence_key` unless a predeclared incrementality test demonstrates that the overlap adds distinct forward value.
7. **Discovery chooses; confirmation validates.** The representative/threshold/direction is fixed using discovery data. Confirmation cannot re-optimize it.
8. **Independent mechanisms may strengthen; correlated variants may not.** A stronger action requires genuinely different independence keys and separate confirmation of the multi-mechanism state.
9. **Conflicts fail closed.** Opposing validated mechanisms—including disagreement inside one dependency cluster—produce `PASS — CONFLICT`.
10. **MODEL ONLY is a valid result.** A market is never forced to produce bets.
11. **Prospective evidence stays separate.** Historical confirmation can qualify a fixed candidate; live prospective performance is tracked independently and controls later governance/promotion.
12. **Production is frozen; research is continuous.** Research challengers may evolve. Production changes only through an explicit promotion contract.
13. **Sport adapters change; the framework does not.** Football, basketball, baseball, tennis, etc. may use different statistics/systems/market microstructure, but these authority rules remain shared.

## Standard action vocabulary

- `MODEL ONLY` — prediction exists; no validated edge family is active.
- `PLAY` — one validated independent mechanism (or multiple correlated families collapsed to one mechanism) supports one direction.
- `STRONG PLAY` — at least two independent validated mechanisms agree **and** the sport/market's multi-mechanism confirmation gate passed.
- `PASS — CONFLICT` — validated mechanisms disagree.
- `EDGE — NO EXEC QUOTE` — authority exists but an executable/current market quote is unavailable.

## Standard selector pattern

A sport may have many research-side statistical families. The preferred sequence is:

`frozen fair value -> model/market gap -> prior-only STAT market-error model -> discovery-only family/threshold selection -> untouched confirmation -> one STAT independence mechanism -> prospective tracking`

Multiple validated STAT families do not become multiple votes unless a future explicit dependency standard proves they are independent. The default is one STAT mechanism per market.

## NCAAF precedent

NCAAF Production V1 follows this structure: separate frozen Spread/H2H/Totals models; a frozen Spread statistical/edge selector; family-collapsed Totals systems; H2H model-only when no repeatable edge passed; independent-mechanism escalation; conflict = pass. Broad feature discovery remains research-only while production stays compact.

## NFL V2.3 adaptation

NFL Edge Authority V2.3 keeps Betting Engine V1 as a failed benchmark/shadow and uses four interpretable classes: `FAIR_VALUE`, `STAT_SELECTOR`, `SYSTEM`, and `MARKET_CONFIRMATION`.

For new STAT selectors, 2021–2023 is discovery, 2024–2025 is confirmation, and 2026 is prospective only. Rich NFL statistical families predict market residual reliability rather than rewriting CORE. All validated STAT selectors collapse to one STAT independence key per market.

Direct System Miner families retain their natural 2017–2022 discovery and 2023–2025 frozen validation windows. A parent/child and overlap audit assigns system independence keys before any `STRONG PLAY` escalation is allowed.
