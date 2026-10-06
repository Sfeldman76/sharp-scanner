# NCAAF Engine V2.9 — Expert / Model Atom Bridge

This release extends the protected NCAAF research path while leaving **Production V1 frozen**. It does not create a new Miner, new global model, or new market backend. It connects intelligence that previously ran largely in parallel and lets the **existing Miner** test whether those sources add incremental value together.

## Main change

The existing Miner can now research combinations involving:

- deterministic Pathi football systems;
- base Big Al NCAAF systems;
- the existing conference / rivalry / H2H / timing / team-memory / ATS / SU / spread / total / market-path atoms;
- season-forward OOF incumbent CORE state from the protected CORE challenger;
- season-forward OOF specialist state for discovery-selected specialists;
- richer key-number journey context.

Examples of questions the same Miner can now test are:

- `PATHI dog hook above 3 + same conference + game 7+`;
- `BIG AL CF2 + road`;
- `PATHI rule + incumbent CORE support`;
- `BIG AL rule + specialist/CORE divergence`;
- `CORE strong edge + Pathi support + opponent off ATS loss`.

All combinations still pass through the existing discovery, FDR, 2024/2025 confirmation, dependency collapse, parent/child lineage, shrinkage, and prospective framework.

## 1. Pathi atoms

Directional Pathi concepts already reconstructed in the historical frame are exposed to the Miner as one `EXPERT_PATHI` family. Because the Miner allows only one atom per family inside a rule, it cannot stack several correlated Pathi flags and count them as independent evidence. A separate `PATHI_MULTI_2PLUS` atom allows multiplicity itself to be tested without double-counting the individual flags.

Market-key events that are context rather than directional recommendations remain context atoms rather than being assigned artificial W/L direction.

## 2. Big Al atoms

The Miner receives the **base** Big Al NCAAF hypotheses:

- CF1 Week 2 home off 42+ win;
- CF2 late-season revenge dog;
- CF3 fade 19+ favorite after upset loss.

Hand-tightened children such as CF2 Away are **not** inserted as primitive Miner atoms. The Miner can instead test `CF2 + ROAD` itself. Existing parent/child lineage can then determine whether the added ROAD condition genuinely improves the parent out of sample.

The direct published-system W/L table still grades CF2 Away separately for source-system tracking.

## 3. OOF incumbent CORE state bridge

The CORE challenger now publishes an incumbent CORE state into the Miner frame using only season-forward predictions:

- 2023 predicted from 2022;
- 2024 predicted from seasons before 2024;
- 2025 predicted from seasons before 2025;
- no 2026 state is used for selection.

Miner atoms include strong positive/negative CORE edge and large absolute CORE edge. These atoms are explicitly labeled research CORE state.

## 4. OOF specialist-state bridge

Specialists that survive the 2023 conditional-attribution discovery screen can also publish season-forward state into the Miner frame. For each bridged specialist the Miner may test:

- specialist edge toward the current team;
- specialist edge toward the opponent;
- strong CORE/specialist agreement;
- strong CORE/specialist conflict;
- specialist-vs-CORE divergence using a cutoff frozen from 2023 discovery.

This does **not** feed confirmation outcomes back into the feature definition.

## 5. Research-only authority guard

Any Miner mechanism containing `CORE_OOF_*` or `SPEC_*` atoms is research-only in V2.9. Even if it confirms historically, it cannot become a live Bet Authority family yet because there is not yet a proven like-for-like live scorer for those OOF state fields.

Pathi and Big Al atoms remain deterministic pregame rules and therefore remain live-evaluable under the existing Miner authority policy if they independently clear all existing qualification gates.

## 6. Heavy Research execution order

Heavy Research now intentionally runs:

1. historical / OOF cache;
2. CORE Challenger and OOF CORE/specialist bridge publication;
3. the existing Miner with the expanded atom catalog.

This ordering is required so the Miner sees leakage-safe CORE/specialist states during discovery and 2024/2025 confirmation.

## 7. Existing research preserved

V2.9 retains rather than replaces:

- PLAY_ON vs FADE evaluation;
- discovery FDR controls;
- 2022-2023 discovery only;
- both 2024 and 2025 confirmation;
- dependency/mechanism collapse;
- parent/child incremental lineage;
- conference and conference-pair context;
- rivalry/revenge/H2H context;
- 1/2/3-game SU/ATS sequence and team-memory atoms;
- role-change and timing/rest context;
- market movement, sharp/soft divergence, key crossing and persistence where historical data exists;
- price-aware H2H validation;
- published Pathi / Big Al W/L attribution;
- shrinkage / Wilson diagnostics;
- Miner threshold-neighborhood diagnostics;
- sealed 2026 prospective tracking;
- no automatic production promotion.

## What intentionally did NOT change

- frozen NCAAF Spread Production V1 model;
- frozen H2H Production V1 model;
- frozen Totals Production V1 model;
- Production V1 probability artifact;
- frozen STAT selector;
- 2% CORE edge + 2% live EV candidate gates;
- STRONG_VALIDATED Miner live-authority threshold;
- Bet Authority architecture;
- Utils / Move Master live market backend;
- production ledger identity / physical-game dedupe;
- 2026 outcomes remain excluded from research selection;
- no automatic promotion.

The following production/backend files are unchanged from V2.8:

- `ncaaf_production_v1.py`
- `ncaaf_production_ledger_v1.py`
- `utils.py`

## Files to replace

Replace these four files:

1. `ncaaf_core_challenger_v2.py`
2. `ncaaf_research_v2.py`
3. `sharp_line_dashboard.py`
4. `train_job.py`

The ZIP also contains the synchronized unchanged production/backend files as a full repository snapshot.

## After deployment

Run **NCAAF Research — Heavy Challenger Search** once.

In the log, look for:

- `[NCAAF-CORE-V24-MINER-BRIDGE]` — OOF CORE/specialist state publication;
- `[NCAAF-RV25-ATOM-BRIDGE]` — Pathi / Big Al / CORE / specialist atom counts by market;
- normal Miner mechanism lines and lineage diagnostics;
- `[NCAAF-RV25-CONTRACT]` — final research contract including bridge mechanism count.

Do **not** run Production Publish for this research patch.
