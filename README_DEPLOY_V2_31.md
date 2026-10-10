# NCAAF Research V2.31 — Deep Objective Context Completion

## Purpose
V2.31 completes the objective System Miner vocabulary exposed by the newest Big Al examples. It does **not** make Big Al the betting authority and does not hard-code the observed teams or 2026 outcomes into systems.

V2.31 builds on V2.30's H2H/revenge, rivalry, travel/time-zone, road-sequence, rest and market-transition context and adds four source-neutral families:

1. `RESULT_QUALITY_REGRESSION`
2. `SCHEDULE_RESUME_QUALITY`
3. `RECENT_VS_SEASON`
4. `MATCHUP_DIFFERENTIAL`

Production V1 remains frozen and separate.

## Files to deploy
Replace these together in the existing `sharp_job_build` source directory:

1. `ncaaf_research_v2.py`
2. `train_job.py`
3. `big_al_observations.jsonl`

The travel bootstrap does **not** need to be rerun. The persistent BigQuery travel tables have already been populated and repaired with coordinate-based timezone fallback.

## New deep context bridge
The Heavy job now reconstructs the following from the validated NCAAF team-side historical source before System Miner search.

### Result quality / regression
Prior-game objective state includes:
- total-yardage margin;
- yards-per-play margin;
- first-down margin;
- turnover margin;
- explicit non-offensive TD margin where source fields exist;
- win despite negative underlying statistics;
- loss despite positive underlying statistics;
- bounded deceptive-win/deceptive-loss scores;
- turnover/non-offensive-score assistance or adversity.

Representative Miner atoms include:
- `TEAM_OFF_WIN_NEG_YARDS`
- `TEAM_OFF_WIN_NEG_YPP`
- `TEAM_OFF_WIN_TURNOVER_EDGE_2PLUS`
- `TEAM_OFF_DECEPTIVE_WIN_2PLUS`
- `TEAM_OFF_RESULT_OVERPERFORMANCE`
- `TEAM_OFF_LOSS_POS_YPP`
- `TEAM_OFF_RESULT_UNDERPERFORMANCE`
- opponent-mirror variants.

### Schedule / resume quality
Pregame schedule context is reconstructed chronologically using each prior opponent's quality **as it was known before that prior game**.

Fields include:
- prior opponents' win percentage;
- prior opponents' net YPP;
- opponent-adjusted game net YPP;
- rolling SOS/resume quality;
- record-versus-underlying-quality disagreement.

Representative atoms include:
- `TEAM_SOS_ADV_10P`
- `OPP_SOS_ADV_10P`
- `TEAM_OPPADJ_NETYPP_ADV_075`
- `OPP_OPPADJ_NETYPP_ADV_075`
- `TEAM_STRONG_RECORD_WEAK_SOS`
- `OPP_STRONG_RECORD_WEAK_SOS`
- `TEAM_STRONG_RECORD_WEAK_UNDERLYING`
- `OPP_STRONG_RECORD_WEAK_UNDERLYING`.

### Recent vs season
For each team and opponent, V2.31 reconstructs prior-only season-to-date, Recent 3 and selected Recent 5 metrics.

Metrics include:
- offensive and defensive PPG;
- offensive YPP and YPP allowed;
- rush YPA and rush YPA allowed;
- pass YPA and pass YPA allowed;
- net YPP;
- turnover margin;
- sack-allowed rate;
- defensive sack rate.

Representative atoms include:
- `TEAM_RECENT3_OFF_YPP_UP_050`
- `TEAM_RECENT3_NET_YPP_UP_075`
- `TEAM_RECENT3_DEF_YPP_IMPROVED_050`
- `TEAM_RECENT3_RUSH_YPA_UP_075`
- `TEAM_RECENT3_PASS_YPA_UP_100`
- `TEAM_RECENT3_AND5_NET_YPP_UP`
- opponent-mirror variants.

### Matchup differential
V2.31 creates direct offense-vs-opponent-defense states rather than asking Miner to infer them from isolated columns.

Representative atoms include:
- `TEAM_RUSH_MATCHUP_EDGE_075PLUS`
- `TEAM_PASS_MATCHUP_EDGE_100PLUS`
- `TEAM_YPP_MATCHUP_EDGE_050PLUS`
- `TEAM_SCORING_MATCHUP_EDGE_5PLUS`
- `TEAM_MULTI_MATCHUP_EDGE_2PLUS`
- `TEAM_MULTI_MATCHUP_EDGE_3PLUS`
- `TEAM_PASS_PROTECTION_STRESS`
- opponent-mirror variants.

## Leakage contract
All deep context is written before the current game's result enters rolling state.

- Discovery: 2022-2023 only.
- Confirmation: 2024 and 2025.
- 2026+: prospective/live trigger context only.
- An exact historical row with a missing pregame value remains missing.
- Latest-state fallback is allowed only for 2026+ prospective games with no completed exact game row.
- Unmatched 2022-2025 rows fail closed; season-end state cannot backfill them.
- Closing-line data is not used as a fake historical current quote.
- Weather and injuries are not added.

## System authority / family caps
The normal System Miner statistical contract is unchanged. The standard new-system discovery minimum remains 100 observations.

Each new context family collapses to a maximum of one independent family vote:
- `NCAAF_RESULT_QUALITY_FAMILY`
- `NCAAF_SCHEDULE_RESUME_FAMILY`
- `NCAAF_RECENT_VS_SEASON_FAMILY`
- `NCAAF_MATCHUP_DIFFERENTIAL_FAMILY`

The existing caps remain for H2H, rivalry, travel, State/Bounceback and external-rating families. Correlated variants cannot manufacture multiple independent votes.

## Local historical validation
The supplied 2022-2025 matched team-side files were used for a full context smoke test:

- team-side source rows: **7,308**;
- physical games: **3,654**;
- 2022: 895 games;
- 2023: 906 games;
- 2024: 919 games;
- 2025: 934 games;
- exact HOME-side deep-context matches: **3,654 / 3,654**;
- exact ROAD-side deep-context matches: **3,654 / 3,654**;
- result-quality ready: **3,298**;
- Recent-3 ready: **3,298**;
- schedule/resume ready: **2,907**;
- matchup ready: **3,085**.

Historical support-floor catalog on the combined frame contains **121 atoms** without the legacy dashboard extras. New-family admitted atom counts:
- Result quality/regression: 11;
- Schedule/resume quality: 16;
- Recent-vs-season: 22;
- Matchup differential: 16.

The special-atom audit found **zero source-unavailable expected atoms** across all four new families. Sparse rules remain diagnostic-only when they do not meet the frozen 100-game discovery requirement.

Season-opener leakage checks returned zero non-null values for season-to-date/recent/previous-game fields, and a deliberately unmatched 2022 historical row remained null rather than receiving latest-state fallback.

`ncaaf_research_v2.py` self-test and Python compilation both pass.

## Deploy / run
After replacing the three deploy files:

```bash
cd ~/sharp_job_build

python3 -m py_compile ncaaf_research_v2.py train_job.py

grep -n "ncaaf-research-v2.31.0-deep-objective-context-completion-20261010" \
  ncaaf_research_v2.py train_job.py
```

Rebuild/redeploy the existing `sharp-train-job` using the same deployment command already used for the current job image.

Then run only:

**NCAAF Research — Heavy Challenger Search**

Do **not** run Production Publish for this research change.

## Heavy-run acceptance markers
Review these markers in the next log:

- `[NCAAF-RV231-DEPLOY-PREFLIGHT]`
- `[NCAAF-DEEP-CONTEXT-V231-BRIDGE]`
- `[NCAAF-SPECIAL-ATOM-AUDIT]` for all four new families
- `[NCAAF-RV231-FAMILY-CAPS]`
- `[NCAAF-INCUMBENT-LIBRARY]`
- `[NCAAF-SYSTEM-LIBRARY-SUMMARY]`
- `[NCAAF-RV231-CONTRACT]`

Expected safeguards:
- `outcomes_2026_selection=FALSE`;
- `production_authority=0` for research construction;
- `automatic_promotion=FALSE`;
- persistent incumbents preserved/revalidated under the existing V2.29+ lifecycle;
- no change to frozen Production V1 model probabilities.
