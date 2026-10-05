# NCAAF V2.2 — Advanced Miner + CORE-First Bet Authority

## Files to replace
Replace these six files together from this bundle:

- `ncaaf_research_v2.py`
- `ncaaf_production_v1.py`
- `ncaaf_production_ledger_v1.py`
- `utils.py`
- `train_job.py`
- `sharp_line_dashboard.py`

Do not mix these with older copies.

## Deployment sequence
1. Replace all six files.
2. Rebuild/redeploy the dashboard/scanner and training job from the same source set.
3. In Model maintenance select **NCAAF Research Update — V2.2** and run it once.
4. Do **not** run **NCAAF Production — Publish Approved Contract** merely for this release. The frozen Production V1 probability artifact is intentionally unchanged.
5. When the V2.2 research job finishes, the normal scanner automatically reads the frozen qualified Miner registry and evaluates current pregame triggers. The report cache retries immediately when no V2.2 artifact exists and otherwise refreshes on a short TTL.

## Bet Authority V2
The frozen Spread / H2H / Totals model probabilities remain separate and unchanged.

For Spread and Totals:

- CORE candidate gate: model edge over break-even >= 2 percentage points AND live expected value >= +2%.
- CORE below gate -> `PASS`.
- CORE qualifies but has no independent confirming evidence -> `CANDIDATE`.
- CORE + 1 clean independent supporting evidence family -> `BET`.
- Spread CORE + 2+ clean independent supporting evidence families -> `STRONG BET`.
- Totals never escalates to `STRONG BET` from multi-system overlap; the prior prospective evidence did not validate that escalation.
- Mixed support/conflict -> `CANDIDATE` / caution.
- 2+ independent conflicts with no support -> `PASS — CONFLICT`.
- Systems may not create a wager without CORE, reverse CORE, or rewrite CORE probability.
- H2H remains `MODEL ONLY`.

Evidence lanes include the existing frozen STAT / Pathi / Big Al / promoted system evidence plus confirmed V2.2 Miner mechanism families.

## System Miner V4
Research source tag:
`ncaaf-research-v2.2-advanced-miner-live-overlay-20261005`

Research contract:
- Discovery: through 2023 only.
- Confirmation: 2024 AND 2025.
- 2026+: prospective only; never used for search, family selection, direction selection, or qualification.
- Search depth up to 5 condition families, beam width 64.
- Symmetric 1/2/3-game SU and ATS sequences.
- Prior SU / ATS magnitude and streak state.
- Conference and conference-pair context.
- Rivalry, revenge and H2H history.
- Role changes and team price history.
- Rest / season timing.
- Team / coach identity where historical support permits.
- Market path, sharp-soft, and key-cross context when leakage-safe historical fields exist.
- FDR + season robustness + remove-best-season controls.
- Dependency / near-duplicate collapse to one vote per mechanism family.
- Unknown live atoms fail closed (`NOT EVALUABLE` / no vote), never guessed.

## Live Miner bridge
The V2.2 report publishes frozen confirmed mechanism definitions. The scanner evaluates only those definitions against the current pregame state. Current-season outcomes can satisfy a prior-state trigger but cannot qualify or rerank a system.

Dashboard diagnostics show:
- qualified Miner families
- live-evaluable Miner families
- triggers fired
- supports CORE
- conflicts CORE

## New dashboard
The NCAAF board now shows:
- `CANDIDATE / BET / STRONG BET / PASS`
- CORE probability, edge and EV
- support/conflict source and family attribution
- Pathi state
- Miner state
- triggered systems
- full Bet Authority decision-reason table
- historical confirmed Miner pools
- 2+ independent Miner confluence
- best individual Miner systems ranked with Wilson lower bound
- best Miner systems when participating in independent-family confluence

## Prospective ledger
New policy records use:
`ncaaf-production-v2-core-candidate-bet-authority-20261005`

and a new model-instance prefix:
`NCAAF_PROD_BETAUTH_V2__...`

This segregates the new decision policy from prior Production V1 pick history without deleting or rewriting old records.

Only `BET`, `STRONG BET`, historical `PLAY`, and historical `STRONG PLAY` are treated as wager actions in ROI reporting. `CANDIDATE` is recorded for prospective audit but is not treated as a wager.

## Validation performed
- All six Python files compile.
- Research V2.2 self-test passes.
- Live Miner frozen-rule trigger test passes.
- Bet Authority synthetic ladder passes:
  - CORE only -> CANDIDATE
  - CORE + one Miner -> BET
  - CORE + Miner + Pathi -> STRONG BET
  - CORE + one conflict -> CANDIDATE
  - CORE + two independent conflicts -> PASS — CONFLICT
  - CORE below gate -> PASS
  - Totals + two supports -> BET, not STRONG BET
- Ledger action mapping passes.
