# CFBD-native program: independent research gates

The path is **source evidence -> temporal correctness -> model falsification -> conditional discovery -> market increment -> prospective confirmation**, not "fix a run until it says PASS".

| Gate | Scientific question | Evidence required | Status |
|---|---|---|---|
| 0. Universe | Which CFBD source families actually have rich historical game-level observations per 2013–2025 year, conference and matchup? | BigQuery physical table/schema inventory; immutable response payload audits; completeness matrix | NOT EXECUTED LIVE |
| 1. Fidelity | Can we reconcile official game scores with team boxes, drive points and play sequences on representative 2013, 2015, 2022, 2025 games? | 2 sides per game, exact IDs, score reconciliation, conflicting/missing evidence separated | TOOLING BUILT, DATA TEST PENDING |
| 2. Temporal validity | Would a forecaster at the time of each game have known all features? | Mutable future/current-game canary tests; historical PPA/vintage audit; verified source publication date rules | CANARY PASS ON SYNTHETIC, LIVE PENDING |
| 3. Honest football-only baseline | Does original-grain data improve prediction versus scores alone? | Same games, year-forward OOF, family ablations and matched uncertainty | EXPERIMENT BUILT, LIVE PENDING |
| 4. Information architecture | Do play/drive sequence structure and dynamic opponent adjustment outperform static averages? | Paired scoring-component, hierarchical, uncertainty experiments, out-of-fold | DESIGN ONLY |
| 5. Historical transport | Are older rules/measurements transferable, and when are 13 years better than 2/5? | Era breaks, actual 2/5/full time windows, separate state reset, missingness controls | TRAINING-WINDOW TEST BUILT, EXTENSIONS PENDING |
| 6. New phenomena | Which nonlinear interactions improve independent error, robustly beyond familiar legacy features? | Predeclared and hypothesis-mined interactions, multiple-tests correction, stability and coverage | FUTURE CHALLENGER |
| 7. Market increment | Which football insights improve forecast beyond genuine timestamped openers? | Certified opener audits; market + OOF football residual model; paired CI | STRICT READ-ONLY DIAGNOSTIC BUILT |
| 8. Prospective qualification | Do findings generalize to future games? | Pre-registered configurations and untouched forward 2026+ results; no auto promotion | NOT STARTED |

**First requested artifacts from user's environment:** BigQuery audit results (`sql/01_existing_warehouse_read_only_audit.sql`) and any genuine CFBD API response archive exports already available. Do not bulk redownload before checking reuse and API quota.
