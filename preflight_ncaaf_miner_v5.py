"""Fail-fast dependency verification for Unified NCAAF Miner V5 Cloud Run image.

Use before the Cloud Build upload and inside Dockerfile.job after COPY . . .
Does not access GCS, mutate production, or trigger research.
"""
from pathlib import Path
import ast
import importlib
import sys
import unittest

ROOT = Path(__file__).resolve().parent
REQUIRED = {
    "ncaaf_miner_permanent_integrity_v4": (
        "repair_recycled_ids", "research_scenarios", "collapse_live_source_votes"),
    "ncaaf_miner_exact_path_shadow_v1": ("extract_exact_path_shadow", "publish_shadow"),
    "ncaaf_miner_identity_crosswalk_v2": ("build_crosswalk", "attach_verified_identity"),
    "ncaaf_unified_history_miner_v5": ("run_full_history", "load_current_v55_core"),
}

def main():
    errors = []
    for module_name, functions in REQUIRED.items():
        path = ROOT / (module_name + ".py")
        if not path.is_file():
            errors.append(f"MISSING_FILE {path}")
            continue
        try:
            module = importlib.import_module(module_name)
            for function in functions:
                if not callable(getattr(module, function, None)):
                    errors.append(f"MISSING_CALLABLE {module_name}.{function}")
        except Exception as exc:
            errors.append(f"IMPORT_ERROR {module_name}: {type(exc).__name__}: {exc}")
    entrypoint = ROOT / "ncaaf_research_v2.py"
    if not entrypoint.is_file():
        errors.append(f"MISSING_FILE {entrypoint}")
    else:
        try:
            ast.parse(entrypoint.read_text(encoding="utf8"), filename=str(entrypoint))
            source = entrypoint.read_text(encoding="utf8")
            for marker in ("from ncaaf_miner_permanent_integrity_v4 import",
                           "from ncaaf_unified_history_miner_v5 import",
                           "[NCAAF-UNIFIED-HISTORY-MINER-V5]"):
                if marker not in source:
                    errors.append(f"MISSING_V5_INTEGRATION {marker}")
        except Exception as exc:
            errors.append(f"RESEARCH_SOURCE_ERROR {type(exc).__name__}: {exc}")
    # Run the same populated-CORE regression that failed in the October 25
    # Heavy image. A successful import alone would not have caught shadowed
    # outcome-array variables; build must reject that defect before deployment.
    test_dir = ROOT / 'research_validation_tests' / 'tests'
    if test_dir.is_dir() and not errors:
        for filename in ('test_ncaaf_miner_permanent_integrity_v4.py',
                         'test_ncaaf_unified_history_miner_v5.py'):
            suite=unittest.defaultTestLoader.discover(str(test_dir),pattern=filename)
            result=unittest.TextTestRunner(verbosity=0).run(suite)
            if not result.wasSuccessful() or result.testsRun<1:
                errors.append('BUILD_TIME_TEST_FAILURE '+filename)
    else:
        errors.append('MISSING_TEST_SUITE '+str(test_dir)) if not errors else None
    if errors:
        for error in errors:
            print("[NCAAF-MINER-V5-DEPLOY-PREFLIGHT] FAIL", error, flush=True)
        return 1
    print("[NCAAF-MINER-V5-DEPLOY-PREFLIGHT] PASS modules=4 regression_suites=2 v5_wiring=TRUE",flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
