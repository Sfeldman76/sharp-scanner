# NFL + NCAAF PT Incremental Value V2 — No PT Margin UI

This package adds a formal **Prediction Tracker incremental-information / residual challenger** to both sports while keeping the frozen production models unchanged.

## Replace these three repository files

Copy these into the repository root, replacing the current versions:

- `nfl_engine.py` — NFL Engine **V3.13.0**; embeds the new PT incremental component as component 33.
- `ncaaf_research_v2.py` — NCAAF Research **V2.19.0**.
- `train_job.py` — merged launcher containing both the current NCAAF routes and the NFL V3.13 changes. Use this copy so deploying one sport does not roll back the other sport's job routing.

`patch_dashboard_remove_pt_ui_both.py` removes all Prediction Tracker margin display columns from both NFL and NCAAF. It is idempotent and supports the older four-column NCAAF PT UI as well as the newer single `PT Margin vs Market` UI.

## What changed

For both sports, PT is no longer judged mainly by whether an individual rating looks good by itself. The research lane now asks the stronger question:

> Does PT add forecast information after frozen Production V1 has already made its prediction?

The tested architectures are:

1. `BASELINE` — frozen Production V1.
2. `BASE_CALIBRATION` — a no-PT residual recalibration control.
3. `PT_STANDALONE` — cluster-balanced PT consensus.
4. `SIMPLE_BLEND` — constrained Production/PT blend.
5. `PT_RESIDUAL` — regularized PT residual correction.
6. `PT_CONDITIONAL_RESIDUAL` — regularized PT residual correction with agreement/disagreement and dispersion structure.

The explicit `BASE_CALIBRATION` control is important. A PT challenger cannot claim incremental value just because Production V1 itself could be recalibrated. PT must beat **both** frozen Production V1 and the best no-PT control on untouched validation data.

Pathi / Big Al / existing Miner interactions remain in the separate PT external Miner. They are not standalone inputs to the formal residual challenger, which avoids falsely crediting PT for signal that actually comes from an existing system.

## Protected time splits

### NFL
- Fit residual models: **2021**
- Select architecture / hyperparameters: **2022**
- Untouched validation: **2023–2025**
- **2026 is prospective-only**
- Runs on both Spreads and Totals

### NCAAF
- Fit residual models: **2022**
- Select architecture / hyperparameters: **2023**
- Untouched validation: **2024–2025**
- **2026 is prospective-only**
- Spread PT only; there is not currently a NCAAF PT totals feed

## Promotion rule

Nothing is automatically promoted. `strict_incremental_signal=true` requires the selected PT architecture to beat:

- frozen Production V1 by at least 0.05 points of validation MAE,
- the best no-PT calibration control by at least 0.025 points of validation MAE,
- both controls on RMSE,
- both controls on Brier score when probabilistic evaluation is available,
- the controls across the required validation seasons,
- and both bootstrap MAE-gain confidence intervals must remain above zero.

Even when this gate passes, the result remains research/shadow with `production_authority=0` and `automatic_promotion=false`.

## Run after deployment

Run these two heavy jobs once:

1. **NFL Research — Heavy Challenger Search**
2. **NCAAF Heavy Challenger Research**

Do not run Production Publish merely for this change.

### NFL markers to look for

- `[NFL-PT-INCREMENTAL-REPLAY]`
- `[NFL-PT-INCREMENTAL]` for SPREADS
- `[NFL-PT-INCREMENTAL]` for TOTALS
- `[NFL-PT-INCREMENTAL-CONTRACT]`
- `[NFL-HEAVY-V313-OPERATOR-CONTRACT]`

### NCAAF markers to look for

- `[NCAAF-RV219-DEPLOY-PREFLIGHT]`
- `[NCAAF-PT-INCREMENTAL]`
- `[NCAAF-RV219-CONTRACT]`
- `[NCAAF-HEAVY-RUN]`

A result such as `BASELINE_NO_PT` or `BASE_CALIBRATION_NO_PT` is a valid research result. It means PT has not demonstrated incremental value beyond the relevant control and should remain backend research/shadow information.

## UI

Prediction Tracker is now **backend-only** on the NFL and NCAAF production boards. Run:

```bash
python patch_dashboard_remove_pt_ui_both.py sharp_line_dashboard.py
```

This removes:

- `PT Margin vs Market`
- `PT Meta Home Margin`
- `PT Meta vs Market`
- `PT External Consensus Home Margin`
- `PT External Consensus vs Market`

It also removes the NFL live-PT display helper that existed only to populate the UI. It does **not** remove PT from the Miner, external-family research, residual challenger, artifacts, or Heavy Research jobs.

## Files that are *not* added to the repository

There is no separate `nfl_pt_incremental_v1.py` to maintain in production. It is embedded inside `nfl_engine.py`, preserving the consolidated NFL repository design.
