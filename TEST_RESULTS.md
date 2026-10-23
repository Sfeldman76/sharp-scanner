# Local release validation — October 23, 2026

## Passed

- Python 3.11 syntax compilation: all 66 deployed Python modules.
- Static internal-import closure and 19 exact-local loader modules: no missing first-party modules.
- Route parity: dashboard `ncaaf_v55_heavy` is recognized by `train_job.py`.
- Existing V5.5 exact-model historical replay API remains present.
- On-demand V5.5 historical margin prediction viewer exists; checks GCS model, path and output SHA-256 digests.
- API credential lookup migrated to `ODDS_API_KEY` in three runtime source files; original literal not present in release Python source.
- Three structural unit tests executed against **the real historical replay function**, injecting a deterministic V4 test stub: prior-year fold cutoffs, sealed 2026, and missing required columns.
- Deploy profile includes Streamlit, training-job, and FastAPI Dockerfiles and original requirements.

## Not verified here

- Live Cloud Run execution, real CFBD history rows, GCS write/read permissions, pregame sportsbook vintage, or ATS win rates.
- Real `NONLINEAR_HGB` sklearn fits on full historical CFBD input; this environment does not contain the private BigQuery archive.
- Absence of all possible reflection/dynamic references outside examined entry points.
- Independent validation of the recipe's retrospective 2024–2025 model selection.

## Safety

The deployed application previously embedded the same API secret in multiple Python modules. Configure the replacement `ODDS_API_KEY` environment/Secret Manager binding **before deployment**, and rotate the old exposed credential. The original ZIP contains `.git` history; do not publish it with a deployment package.
