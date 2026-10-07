#!/usr/bin/env python3
"""Residential-side Prediction Tracker -> GCS feeder.

Run this from a normal residential machine/network.  It preserves exact source
bytes, validates that anti-bot/challenge pages are never uploaded as good data,
and writes both current raw files plus immutable snapshots to GCS.  The NCAAF
Heavy/Weekly jobs consume GCS first and therefore do not depend on Prediction
Tracker being reachable from Cloud Run.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

PT_BASE_URL = "https://www.thepredictiontracker.com"
PT_ARCHIVE_PAGE_URL = PT_BASE_URL + "/ncaaarchive.html"
PT_LIVE_PAGE_URL = PT_BASE_URL + "/predncaa.php"
PT_LIVE_CSV_URL = PT_BASE_URL + "/ncaapredictions.csv"
PT_RAW_PREFIX = "research/ncaaf/external/prediction_tracker/raw"
PT_SNAPSHOT_PREFIX = "research/ncaaf/external/prediction_tracker/snapshots"
PT_MANIFEST_BLOB = "research/ncaaf/external/prediction_tracker/feeder_manifest.json"
EXPECTED_LIVE_SYSTEMS = ("ESPN FPI", "Pi-Ratings Bias", "Dokter", "Keeper", "Pigskin Index")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36")


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_utc(x: Optional[dt.datetime] = None) -> str:
    x = x or now_utc()
    return x.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def stamp_utc(x: Optional[dt.datetime] = None) -> str:
    x = x or now_utc()
    return x.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def challenge_payload(raw: bytes) -> bool:
    if not raw:
        return True
    head = raw[:15000].decode("utf-8", errors="ignore").lower()
    markers = (
        "title: just a moment", "<title>just a moment", "cf-chl-", "challenge-platform",
        "cloudflare ray id", "enable javascript and cookies", "checking your browser",
    )
    return any(m in head for m in markers)


def _header_cells(raw: bytes) -> list[str]:
    text = raw.decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
    for line in text.split("\n"):
        if not line.strip():
            continue
        try:
            row = next(csv.reader([line]))
        except Exception:
            continue
        return [str(x).strip() for x in row]
    return []


def validate_csv(raw: bytes) -> tuple[bool, dict]:
    if challenge_payload(raw):
        return False, {"reason": "challenge_page"}
    headers = _header_cells(raw)
    norm = [h.lower().strip() for h in headers]
    home = any(x in {"home", "home team"} for x in norm)
    road = any(x in {"road", "away", "visitor", "away team"} for x in norm)
    if not (home and road):
        return False, {"reason": "missing_home_road_headers", "headers": headers[:40]}
    line_like = sum(1 for h in norm if h.startswith("line") or "rating" in h or "prediction" in h)
    if len(headers) < 5:
        return False, {"reason": "header_too_short", "headers": headers}
    return True, {"headers": headers, "line_like_columns": line_like}


def validate_live_html(raw: bytes) -> tuple[bool, dict]:
    if challenge_payload(raw):
        return False, {"reason": "challenge_page"}
    text = raw.decode("utf-8", errors="ignore")
    low = text.lower()
    missing = [x for x in EXPECTED_LIVE_SYSTEMS if x.lower() not in low]
    if missing:
        return False, {"reason": "missing_named_systems", "missing": missing}
    if "home" not in low or ("road" not in low and "away" not in low):
        return False, {"reason": "missing_home_road_identity"}
    return True, {"named_systems": list(EXPECTED_LIVE_SYSTEMS)}


@dataclass
class FetchResult:
    url: str
    method: str
    bytes: int
    sha256: str
    fetched_utc: str
    validation: dict


def requests_fetch(url: str, parent: str, timeout: int) -> bytes:
    import requests
    s = requests.Session()
    headers = {
        "User-Agent": UA,
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    try:
        s.get(parent, headers={**headers, "Accept": "text/html,application/xhtml+xml,*/*;q=0.8", "Referer": PT_BASE_URL + "/"}, timeout=timeout)
    except Exception:
        pass
    accept = "text/csv,text/plain,*/*" if url.lower().endswith(".csv") else "text/html,application/xhtml+xml,*/*;q=0.8"
    r = s.get(url, headers={**headers, "Accept": accept, "Referer": parent}, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.content


def playwright_fetch(url: str, parent: str, timeout: int, headed: bool = False) -> bytes:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise RuntimeError("Playwright unavailable; install with: pip install playwright && playwright install chromium") from exc
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headed)
        ctx = browser.new_context(user_agent=UA, locale="en-US")
        page = ctx.new_page()
        page.goto(parent, wait_until="domcontentloaded", timeout=timeout * 1000)
        # Give a residential browser session time to clear any Cloudflare JS
        # challenge before the cookie-sharing API request is attempted.
        deadline = time.time() + min(max(timeout, 10), 45)
        while time.time() < deadline:
            try:
                title = (page.title() or "").lower()
                html = page.content().encode("utf-8")
                if "just a moment" not in title and not challenge_payload(html):
                    break
            except Exception:
                pass
            page.wait_for_timeout(1000)

        def _api_get() -> bytes:
            resp = ctx.request.get(url, headers={
                "Referer": parent,
                "Accept": "text/csv,text/plain,*/*" if url.lower().endswith(".csv") else "text/html,*/*",
            }, timeout=timeout * 1000)
            if not resp.ok:
                raise RuntimeError(f"browser_api_http_{resp.status}")
            return resp.body()

        body = _api_get()
        if challenge_payload(body):
            # Refresh the browser page once to let any newly-issued challenge
            # complete, then retry through the cookie-sharing request context.
            try:
                page.goto(parent, wait_until="domcontentloaded", timeout=timeout * 1000)
                page.wait_for_timeout(5000)
            except Exception:
                pass
            body = _api_get()
        browser.close()
        return body


def fetch_validated(url: str, parent: str, kind: str, timeout: int, headed: bool) -> tuple[bytes, FetchResult]:
    errors = []
    for method in ("REQUESTS", "PLAYWRIGHT"):
        try:
            raw = requests_fetch(url, parent, timeout) if method == "REQUESTS" else playwright_fetch(url, parent, timeout, headed=headed)
            ok, diag = validate_csv(raw) if kind == "csv" else validate_live_html(raw)
            if not ok:
                raise RuntimeError(f"validation_failed:{diag}")
            sha = hashlib.sha256(raw).hexdigest()
            return raw, FetchResult(url=url, method=method, bytes=len(raw), sha256=sha, fetched_utc=iso_utc(), validation=diag)
        except Exception as exc:
            errors.append(f"{method}:{type(exc).__name__}:{exc}")
    raise RuntimeError(" | ".join(errors))


def blob_exists(bucket, path: str) -> bool:
    try:
        return bool(bucket.blob(path).exists())
    except Exception:
        return False


def upload_exact(bucket, raw: bytes, raw_path: str, snapshot_path: str, content_type: str, result: FetchResult, dry_run: bool) -> dict:
    if not dry_run:
        bucket.blob(raw_path).upload_from_string(raw, content_type=content_type)
        bucket.blob(snapshot_path).upload_from_string(raw, content_type=content_type)
    return {
        **asdict(result),
        "raw_blob": f"gs://{bucket.name}/{raw_path}",
        "snapshot_blob": f"gs://{bucket.name}/{snapshot_path}",
        "dry_run": bool(dry_run),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Prediction Tracker residential feeder -> GCS")
    ap.add_argument("--bucket", default="sharp-models")
    ap.add_argument("--refresh-history", action="store_true", help="Re-download 2022-2025 even if a raw GCS blob already exists")
    ap.add_argument("--current-only", action="store_true", help="Skip completed historical seasons")
    ap.add_argument("--headed", action="store_true", help="Use a visible Chromium window for the Playwright fallback")
    ap.add_argument("--timeout", type=int, default=35)
    ap.add_argument("--dry-run", action="store_true", help="Fetch and validate, but do not upload")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        good = b'Home,Road,line,lineespn,lindokter\nAlpha,Beta,3,4,5\n'
        bad = b'<html><title>Just a moment...</title>Checking your browser</html>'
        ok1, _ = validate_csv(good)
        ok2, _ = validate_csv(bad)
        ok3, _ = validate_live_html(b'<html>Home Road ESPN FPI Pi-Ratings Bias Dokter Keeper Pigskin Index</html>')
        print(json.dumps({"status": "PASS" if ok1 and not ok2 and ok3 else "FAIL", "csv_good": ok1, "challenge_rejected": not ok2, "live_names": ok3}, indent=2))
        return 0 if ok1 and not ok2 and ok3 else 1

    try:
        from google.cloud import storage
    except Exception as exc:
        print("google-cloud-storage is required: pip install google-cloud-storage", file=sys.stderr)
        return 2

    client = storage.Client()
    bucket = client.bucket(args.bucket)
    run_ts = now_utc()
    stamp = stamp_utc(run_ts)
    sources = {}
    failures = {}

    jobs = []
    if not args.current_only:
        for season in range(2022, 2026):
            raw_path = f"{PT_RAW_PREFIX}/ncaa{season}.csv"
            if args.refresh_history or not blob_exists(bucket, raw_path):
                jobs.append((f"ncaa{season}", PT_BASE_URL + f"/ncaa{season}.csv", PT_ARCHIVE_PAGE_URL, "csv", raw_path, "text/csv"))
            else:
                sources[f"ncaa{season}"] = {"status": "SKIPPED_EXISTING", "raw_blob": f"gs://{args.bucket}/{raw_path}"}

    # Current archive + live CSV + named live page are refreshed every run.
    jobs.extend([
        ("ncaa2026", PT_BASE_URL + "/ncaa2026.csv", PT_ARCHIVE_PAGE_URL, "csv", f"{PT_RAW_PREFIX}/ncaa2026.csv", "text/csv"),
        ("ncaapredictions", PT_LIVE_CSV_URL, PT_LIVE_PAGE_URL, "csv", f"{PT_RAW_PREFIX}/ncaapredictions.csv", "text/csv"),
        ("predncaa_live_page", PT_LIVE_PAGE_URL, PT_BASE_URL + "/", "html", f"{PT_RAW_PREFIX}/predncaa_live_page.txt", "text/html"),
    ])

    for name, url, parent, kind, raw_path, ctype in jobs:
        try:
            raw, result = fetch_validated(url, parent, kind, args.timeout, args.headed)
            snap_path = f"{PT_SNAPSHOT_PREFIX}/{stamp}/{raw_path.rsplit('/',1)[-1]}"
            rec = upload_exact(bucket, raw, raw_path, snap_path, ctype, result, args.dry_run)
            rec["status"] = "PASS"
            sources[name] = rec
            print(f"[PT-FEEDER] {name} PASS method={result.method} bytes={result.bytes} sha256={result.sha256[:16]} raw={rec['raw_blob']}")
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}:{exc}"
            sources[name] = {"status": "FAILED", "url": url, "error": failures[name]}
            print(f"[PT-FEEDER] {name} FAILED {failures[name]}", file=sys.stderr)

    manifest = {
        "status": "PASS" if not failures else ("PARTIAL" if any(v.get("status") == "PASS" for v in sources.values()) else "FAILED"),
        "updated_utc": iso_utc(run_ts),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "network_role": "RESIDENTIAL_FEEDER",
        "bucket": args.bucket,
        "snapshot_prefix": f"gs://{args.bucket}/{PT_SNAPSHOT_PREFIX}/{stamp}/",
        "sources": sources,
        "failures": failures,
        "policy": {
            "preserve_exact_source_bytes": True,
            "reject_challenge_pages": True,
            "do_not_overwrite_last_good_on_failure": True,
            "model_side_contract": "GCS_FIRST",
        },
    }
    if not args.dry_run:
        bucket.blob(PT_MANIFEST_BLOB).upload_from_string(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"), content_type="application/json")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
