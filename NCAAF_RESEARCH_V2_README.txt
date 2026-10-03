NCAAF RESEARCH V2 — MINER V3 + ORTHOGONAL STAT
Build: 2026-10-03

PURPOSE
- Keep NCAAF Production V1 frozen.
- Transfer the stronger NFL research discipline back into NCAAF.
- Research-only: zero automatic production authority.

NEW WORKFLOW
Dashboard -> NCAAF -> Train which market? ->
  NCAAF Research V2 — Miner + Orthogonal STAT

WHAT IT DOES
1. Rebuilds the existing leakage-safe NCAAF historical/STAT research caches.
2. Runs orthogonal residual STAT families against the frozen OOF incumbent.
3. Runs System Miner V3 with richer market/path, role, revenge/H2H, timing,
   schedule, conference/team/coach, and model-state concepts when fields exist.
4. Uses discovery only through 2023.
5. Uses 2024-2025 only as untouched confirmation.
6. Seals 2026+ for prospective shadow; it cannot select systems or thresholds.
7. Collapses highly overlapping system variants into independent mechanism families.
8. Publishes research/ncaaf/v2/current_report.json and current_bundle.pkl.
9. Adds a lazy Research V2 panel to the NCAAF UI.
10. Tags live games with Research V2 system triggers as SHADOW only; these tags
    cannot change Production V1 probabilities, PLAY/PASS, or authority.

FILES
- ncaaf_research_v2.py                 NEW
- train_job.py                          UPDATED
- sharp_line_dashboard.py               UPDATED
- ncaaf_production_v1.py                UNCHANGED baseline supplied by user
- ncaaf_production_ledger_v1.py         UNCHANGED baseline supplied by user
- utils.py                              UNCHANGED shared Utils/Move Master backend

DO NOT DELETE YET
The legacy NCAAF edge-research modules are still used by the existing
"NCAAF Legacy Edge Research" and Production V1 publication routes. Do not remove
those files until Research V2 has run successfully and we explicitly collapse or
retire that legacy route.

FIRST RUN
Select:
  NCAAF Research V2 — Miner + Orthogonal STAT
Then send the Cloud Run log for review before changing any production contract.

SAFETY CONTRACT
- Production V1 stays frozen.
- Production contract is not mutated.
- Production authority is always 0 in Research V2 artifacts.
- Current market remains owned by Utils -> sharp_moves_master.
- Enriched market context remains Utils -> moves_with_features_merged.
