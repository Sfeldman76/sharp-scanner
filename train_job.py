# train_job.py — NFL V1.9.3 corrected orchestration wrapper
#
# This file intentionally wraps the CURRENT October-1 train job rather than
# reconstructing it. Keep the current job beside this file as:
#
#     train_job_v1_9_2_base.py
#
# Behavior:
#   1. Run the current/base train job unchanged.
#   2. For SPORT=NFL only, continue into NFL Research V1.9.3 after the base
#      job's successful return / SystemExit(0).
#   3. All non-NFL behavior remains entirely inside the base train job.
#
# Safety:
#   - V1.9.3 production_authority remains 0.
#   - V1.9.3 verifies the V1.9.2 ledger health before research execution.
#   - NCAAF code is not reimplemented here.
#   - Legacy NFL artifacts are not published or replaced by this wrapper.

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path


BASE_FILENAME = os.getenv(
    "TRAIN_JOB_BASE_FILE",
    "train_job_v1_9_2_base.py",
)

V193_ENABLED = str(
    os.getenv("NFL_RESEARCH_V1_9_3", "1")
).strip().lower() not in {"0", "false", "no", "off"}


def _log(msg: str) -> None:
    print(str(msg), flush=True)


def _load_base_train_job():
    here = Path(__file__).resolve().parent
    path = (here / BASE_FILENAME).resolve()

    if not path.exists():
        raise RuntimeError(
            "[NFL-V1.9.3-WRAPPER] BASE_TRAIN_JOB_MISSING "
            f"path={path}. Preserve the current train_job.py as "
            "train_job_v1_9_2_base.py before using this wrapper."
        )

    if path == Path(__file__).resolve():
        raise RuntimeError(
            "[NFL-V1.9.3-WRAPPER] RECURSIVE_BASE_PATH_BLOCKED"
        )

    spec = importlib.util.spec_from_file_location(
        "train_job_v1_9_2_base",
        str(path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"[NFL-V1.9.3-WRAPPER] BASE_IMPORT_SPEC_FAILED path={path}"
        )

    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_job_v1_9_2_base"] = mod
    spec.loader.exec_module(mod)

    if not callable(getattr(mod, "main", None)):
        raise RuntimeError(
            "[NFL-V1.9.3-WRAPPER] BASE_MAIN_MISSING "
            f"path={path}"
        )

    return mod, path


def _dashboard_module_after_base():
    # The base job normally loads the exact deployment-local dashboard into
    # sys.modules. Reuse that object if available so V1.9.3 sees the same code.
    mod = sys.modules.get("sharp_line_dashboard")
    if mod is not None:
        return mod

    try:
        return importlib.import_module("sharp_line_dashboard")
    except Exception as exc:
        raise RuntimeError(
            "[NFL-V1.9.3-WRAPPER] DASHBOARD_IMPORT_FAILED "
            f"error={exc}"
        ) from exc


def main():
    sport = str(os.getenv("SPORT", "NBA")).upper().strip()

    base, base_path = _load_base_train_job()

    _log(
        "[NFL-V1.9.3-WRAPPER] "
        f"status=BASE_START sport={sport} base={base_path.name}"
    )

    base_exit_zero = False

    try:
        base.main()
        base_exit_zero = True
    except SystemExit as exc:
        code = exc.code
        if code in (None, 0):
            base_exit_zero = True
            _log(
                "[NFL-V1.9.3-WRAPPER] "
                "status=BASE_SYSTEM_EXIT_ZERO intercepted=TRUE"
            )
        else:
            raise

    if not base_exit_zero:
        raise RuntimeError(
            "[NFL-V1.9.3-WRAPPER] BASE_DID_NOT_COMPLETE_SUCCESSFULLY"
        )

    # Every non-NFL path ends here. This is deliberate: NCAAF and all other
    # sports remain exactly the behavior of the preserved base job.
    if sport != "NFL":
        _log(
            "[NFL-V1.9.3-WRAPPER] "
            f"status=BASE_COMPLETE sport={sport} v1_9_3=NOT_APPLICABLE"
        )
        return

    if not V193_ENABLED:
        _log(
            "[NFL-V1.9.3-WRAPPER] "
            "status=BASE_COMPLETE sport=NFL v1_9_3=DISABLED"
        )
        return

    _log(
        "[NFL-V1.9.3-TRAIN-JOB] "
        "status=START after_v1_9_2=TRUE "
        "production_authority=0 legacy_nfl=UNCHANGED ncaaf=UNCHANGED"
    )

    from nfl_research_execution_v1_9_3 import run_after_v192_foundation

    dashboard_module = _dashboard_module_after_base()

    result = run_after_v192_foundation(
        dashboard_module=dashboard_module,
        log_func=_log,
    )

    if str(result.get("status")) != "RESEARCH_EXECUTION_COMPLETE":
        raise RuntimeError(
            "[NFL-V1.9.3-TRAIN-JOB] UNEXPECTED_FINAL_STATUS "
            f"status={result.get('status')!r}"
        )

    _log(
        "[NFL-V1.9.3-TRAIN-JOB] "
        f"status={result.get('status')} "
        "production_authority=0 publication=FALSE "
        "legacy_nfl=UNCHANGED ncaaf=UNCHANGED"
    )


if __name__ == "__main__":
    main()
