# NCAAF 13-season Historical-to-Market CORE Research (18 Oct 2026)

## Scope and non-promises
This is a **research-only** GitHub-ready Python update. It reuses 2013–2025 NCAAF scored history (10,396 physical games in the October 18 Heavy log) and strictly chronological market-blind first-stage forecasts. The code has been tested offline with synthetic games. **Actual BigQuery research results have NOT been run in this environment.** A new Google Cloud Run Heavy execution is required to measure gains. No prospective ROI is promised.

**Do not modify:** `ncaaf_production_v1.py`, frozen production backbones, `utils.py`, bet authority, `nfl_engine.py` or live-scanner logic. Do not infer new original opening quotes from closing quotes. The 2026 season is sealed from research selection.

## GitHub deployment
Upload into the same directory as your existing `ncaaf_core_challenger_v2.py` and `ncaaf_research_v2.py`. **The `ncaaf_research_v2.py` Big Al integration from the October 18 previous package is NOT replaced by this package.**

| File | Action | Purpose |
|---|---|---|
| `ncaaf_core_challenger_v2.py` | REPLACE matching current 2.4.x | Wires historical score OOF, margin Core bridge and the three-head comparative scorecard into existing Heavy; preserves preexisting Core and Miner research |
| `ncaaf_full_history_foundation.py` | REPLACE | Adds ephemeral margin/total/H2H OOF sink, with source fingerprint and verified market-blind flag; no GCS dump of OOF predictions |
| `ncaaf_historical_core_bridge.py` | ADD | Matches independent margin predictions, genuine opening spreads, and the existing incumbent Core recipe on exactly the same games; pre-2024 research nomination only |
| `ncaaf_historical_multitarget_core.py` | ADD | Margin/Totals/H2H historical football vs original opening market vs market-aware model scorecard; market-blind opening-quote forecast diagnostic |
| `ncaaf_hierarchical_research.py` | REPLACE only if still at October 17 version | October 18 known-compatible hierarchy module; do NOT overwrite later refinements |
| `ncaaf_historical_signal_discovery.py` | REPLACE only if still at October 17 version | October 18 modern CFBD eligible-selection fix; do NOT overwrite later refinements |
| `ncaaf_combined_signal_challenger.py` | ADD if absent | October 18 parent-child historical statistical family tests |

The three dependencies in the bottom rows are included for a complete compatible research installation. If the October 18 `NCAAF_Research_Fixes_GitHub_20261018.zip` was already deployed, those files can be left unchanged, provided hashes match. `ncaaf_core_feature_parity.py` is NOT replaced. The tests live under `tests/` and are optional for deployment.

**IMPORTANT source-version caveat:** This integration was constructed from the project-local October 18 in-progress `ncaaf_core_challenger_v2.py`, based on the `ncaaf-core-challenger-v2.4-expert-model-atom-bridge-20261006` source (version 2.4.1). If GitHub contains later unreviewed changes to this file, merge the new import/call/report key rather than blindly overwrite.

## Evaluation design

1. **Score foundation**: 2013–2025, no market inputs, season-forward out-of-fold expected margin, total points and H2H probability. Older games without openers are fully retained.
2. **Market sources**: original `Consensus_Open_Spread` plus sign-checked `Market_Open_Margin`, and original `Consensus_Open_Total` (optionally cross-checked against `Market_Open_Total`). Each head can independently be source-limited. The benchmark quote must be genuine and original; closing quotes do not qualify. Data earlier than 2021 are kept for score-only research; no unaudited historical openers are imputed.
3. **Core margin bridge**: match game-level football OOF, true opening, and the existing Core research incumbent; report paired year-specific RMSE/MAE versus both baselines, with 2022/2023 selection only.
4. **Multi-head comparison**: Margin and totals market-only baselines are original prices. H2H is a market-SPREAD-derived probability calibrated using prior seasons' straight-up outcomes; this is *not* observed bookmaker moneyline probability. Evaluate Brier/log loss.
5. **Market-forecast diagnostic**: predict opening spread/total from prior years' market-blind football OOF, without seeing the validation game's quote. This target is the market expectation, not the final score.
6. **Selection/confirmation**: 2022–2023 nominate only when both selection years improve on market; 2024–2025 are retrospective stress tests already examined in earlier research, not pristine holds. 2026 is prospective-only and no 2026 outcomes are used here.
7. **No betting authority**: no automatic promotions, no overlay votes, no CLV/ROI invented, no probability cover bet recommendation without a separate audited distribution/odds layer.

## What to look for in Cloud Run Logs

- `[NCAAF-FULL-HISTORY-CONTRACT]`: expect the existing ~10,396 historical score games and ~9,642 OOF predictions per head, subject to real data refresh.
- `[NCAAF-HISTORICAL-CORE-DUAL-TARGET]`: market-blind margin prediction against actual result and separately original opening-price prediction.
- `[NCAAF-HISTORICAL-CORE-BRIDGE-FOLD]`: incumbent Core versus market and margin challenger on identical games. If no verified opener for prior training year, `SOURCE_LIMITED` is expected.
- `[NCAAF-MULTIHEAD-FOLD]`: independent football, market-only, and blend metrics for margin/total/H2H by 2022–2025; `SOURCE_LIMITED` if prior-year opening data is insufficient.
- `[NCAAF-MULTIHEAD-HISTORICAL-MARKET]`: total eligible open prices by head and candidate(s) selected strictly before 2024.
- `[NCAAF-CORE-V22-CONTRACT]`: existing unchanged keep/incumbent decision. Check `production_authority=0` and `production_mutated=FALSE`.

The updated `research/ncaaf/core_challenger/v2/current_report.json` will include `full_history_score_foundation`, `historical_to_core_bridge`, and `multihead_historical_market_research` as **separate shadow sections**. None enters the production ranking.

### Known market-source limitation
The October 18 historical log does not independently verify that `Consensus_Open_Total` is populated in the Core research frame. If it is missing, the Totals score-only forecast still runs, but the market-priced Totals comparison reports `SOURCE_LIMITED`. The next evidence-based step is to expose that original opening-total column **only if auditable and truly available at the selected timestamp**. Do NOT re-label a closing total as an opening total.

### Expected execution and test status

Offline command from the package directory:

```sh
pytest -q tests/test_multihead_history_market.py tests/test_historical_core_bridge.py
```

27 synthetic offline tests passed on Python 3.13/scikit-learn. Tests verify 13-season score-only coverage, original-opening sign checks, missing-opening fail-closed behavior, unmatched total source independent fallback, 2026 seal, first-stage row fingerprint, pre-2024 selection immutability, matched comparisons, and no production authority. **They do not prove a real-world improvement**.
