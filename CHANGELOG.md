# V2.17 changes

1. Prediction Tracker is sparse/optional, never a schedule authority.
2. Historical PT bridge now attaches available individual systems even when META is unavailable.
3. Missing PT rows remain NaN external context; no NCAA game is dropped or neutralized.
4. META stays exact five-of-five with the original published weights and raw weight sum.
5. Consensus/dispersion atoms are gated to five available components.
6. Diagnostics split listed / partial / full-five / no-PT coverage.
7. Removed mandatory cloud feeder/Oxylabs preflight and network refresh calls from `train_job.py`.
8. Manual GCS is the normal contract; direct web fallback remains disabled by default.
