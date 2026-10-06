# NCAAF V2.5 — Expert/Specialist CORE Research

This release keeps the frozen Production V1 models and current V2.2.1 Bet Authority unchanged. It expands only the protected NCAAF Spread CORE challenger inside **NCAAF Research — Heavy Challenger Search**.

## Why this exists

The first compact challenger search showed that ordinary feature reshuffling did not beat the incumbent two-stat market-residual CORE in either 2024 or 2025. V2.5 therefore tests genuinely different football information rather than just adding more of the same box-score variables.

## Expert CORE lanes

The challenger now builds and tests:

1. **POWER_SOS_CONTEXT**
   - Sequential pregame-only team power
   - Team and opponent strength of schedule
   - Conference member strength
   - Cross-conference residual strength
   - Home/away/neutral role
   - Rest differential
   - Season maturity / games played
   - Power-rating disagreement with the opening market

2. **STRUCTURED_STATS**
   - Opponent-adjusted efficiency
   - Passing and rushing efficiency
   - Turnover/takeaway pressure
   - First-down / conversion proxy
   - Tempo and play mix
   - Offense-vs-defense matchup features
   - Any EPA / success-rate / explosiveness / havoc / line-yards / field-position style numeric fields already present in the processed research frame

3. **EXPERT_COMBINED**
   - Power/SOS/conference context + the strongest discovery-selected football statistics + preregistered matchup/trend interactions

4. **PROGRAM_HIERARCHY**
   - Regularized team, opponent, conference and opponent-conference categorical effects
   - Compact numeric football/strength context
   - This is designed to learn persistent program/conference strength without hard-coding school rankings.

5. **INCUMBENT_PLUS_SPECIALIST**
   - A frozen discovery-selected blend of the current Production V1 CORE and the strongest specialist challenger
   - Allows incremental specialist information to help without forcing a full CORE replacement.

## Validation contract

- 2022: initial training
- 2023: feature/model/blend selection only
- 2024: untouched confirmation
- 2025: untouched confirmation
- 2026+: sealed and never queried by CORE research

A challenger is not promotion-eligible unless it beats the incumbent RMSE in **both 2024 and 2025**, improves pooled RMSE, does not worsen pooled MAE, and then survives paired-bootstrap review. Even a passing challenger remains research-only and must be frozen into prospective shadow before any manual production promotion.

## Production safety

- Production V1 CORE: unchanged
- Production probabilities: unchanged
- NCAAF Bet Authority: unchanged
- Miner authority: unchanged
- Weekly workflow: unchanged
- Automatic model promotion: disabled
- Automatic policy promotion: disabled

No Pathi, Big Al, Miner, closing line, live line movement, or 2026 result can enter this CORE challenger.

## Operator workflows

The UI remains the same two-workflow model:

- **NCAAF Production — Weekly Update** — normal weekly production refresh. No research/refit/promotion.
- **NCAAF Research — Heavy Challenger Search** — runs STAT/System Miner research and this new expert CORE challenger.

## Deploy

From NCAAF V2.4:

1. Add `ncaaf_core_challenger_v2.py`.
2. Replace `train_job.py`.
3. Replace `sharp_line_dashboard.py`.
4. Remove the retired `ncaaf_core_challenger_v1.py` after the new deployment is confirmed.

The full bundle also includes the unchanged current `ncaaf_research_v2.py`, `ncaaf_production_v1.py`, `ncaaf_production_ledger_v1.py`, and `utils.py` for a complete synchronized snapshot.

Then run **NCAAF Research — Heavy Challenger Search** once and review the CORE Challenger V2 section/log. Do not publish a new production contract unless a challenger later passes the full promotion process and is explicitly approved.
