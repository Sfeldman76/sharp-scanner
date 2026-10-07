#!/usr/bin/env python3
"""Cloud-only Prediction Tracker -> GCS feeder.

Designed to run INSIDE the existing Google Cloud Run training job.  It never
requires Python, gcloud, or a browser on the operator's PC.  Requests are routed
through a web-unblocker/residential proxy service because Prediction Tracker
rejects ordinary datacenter egress.

Supported configuration:
  1) Generic proxy URL:
       PT_UNBLOCKER_PROXY_URL=http://user:pass@host:port
  2) Oxylabs Web Unblocker:
       OXYLABS_USERNAME=...
       OXYLABS_PASSWORD=...
       OXYLABS_ENDPOINT=unblock.oxylabs.io:60000   (optional)

Credentials should be injected into Cloud Run from Secret Manager, never stored
in this file.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Optional
from urllib.parse import quote

SOURCE_TAG = "prediction-tracker-cloud-feeder-v1.0-web-unblocker-20261007"
PT_BASE_URL = "https://www.thepredictiontracker.com"
PT_ARCHIVE_PAGE_URL = PT_BASE_URL + "/ncaaarchive.html"
PT_LIVE_PAGE_URL = PT_BASE_URL + "/predncaa.php"
PT_LIVE_CSV_URL = PT_BASE_URL + "/ncaapredictions.csv"
PT_RAW_PREFIX = "research/ncaaf/external/prediction_tracker/raw"
PT_SNAPSHOT_PREFIX = "research/ncaaf/external/prediction_tracker/snapshots"
PT_MANIFEST_BLOB = "research/ncaaf/external/prediction_tracker/feeder_manifest.json"
EXPECTED_LIVE_SYSTEMS = ("ESPN FPI", "Pi-Ratings Bias", "Dokter", "Keeper", "Pigskin Index")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36"


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(x: Optional[dt.datetime] = None) -> str:
    return (x or _now()).isoformat().replace("+00:00", "Z")


def _stamp(x: Optional[dt.datetime] = None) -> str:
    return (x or _now()).strftime("%Y%m%dT%H%M%SZ")


def challenge_payload(raw: bytes) -> bool:
    if not raw:
        return True
    head = raw[:20000].decode("utf-8", errors="ignore").lower()
    markers = (
        "title: just a moment", "<title>just a moment", "cf-chl-",
        "challenge-platform", "cloudflare ray id", "enable javascript and cookies",
        "checking your browser", "attention required! | cloudflare",
    )
    return any(m in head for m in markers)


def _header_cells(raw: bytes) -> list[str]:
    text = raw.decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
    for line in text.split("\n"):
        if line.strip():
            try:
                return [str(x).strip() for x in next(csv.reader([line]))]
            except Exception:
                return []
    return []


def validate_csv(raw: bytes) -> tuple[bool, dict[str, Any]]:
    if challenge_payload(raw):
        return False, {"reason": "challenge_page"}
    headers = _header_cells(raw)
    norm = [h.lower().strip() for h in headers]
    home = any(x in {"home", "home team", "hometeam"} for x in norm)
    road = any(x in {"road", "away", "visitor", "away team", "visitor team"} for x in norm)
    if not (home and road):
        return False, {"reason": "missing_home_road_headers", "headers": headers[:40]}
    if len(headers) < 5:
        return False, {"reason": "header_too_short", "headers": headers}
    return True, {"headers": headers[:80]}


def validate_live_html(raw: bytes) -> tuple[bool, dict[str, Any]]:
    if challenge_payload(raw):
        return False, {"reason": "challenge_page"}
    low = raw.decode("utf-8", errors="ignore").lower()
    missing = [x for x in EXPECTED_LIVE_SYSTEMS if x.lower() not in low]
    if missing:
        return False, {"reason": "missing_named_systems", "missing": missing}
    if "home" not in low or ("road" not in low and "away" not in low):
        return False, {"reason": "missing_home_road_identity"}
    return True, {"named_systems": list(EXPECTED_LIVE_SYSTEMS)}


def _proxy_config() -> dict[str, Any]:
    generic = str(os.getenv("PT_UNBLOCKER_PROXY_URL", "")).strip()
    if generic:
        return {"status": "READY", "provider": "GENERIC", "proxy_url": generic}

    user = str(os.getenv("OXYLABS_USERNAME", "")).strip()
    password = str(os.getenv("OXYLABS_PASSWORD", "")).strip()
    if user and password:
        endpoint = str(os.getenv("OXYLABS_ENDPOINT", "unblock.oxylabs.io:60000")).strip()
        proxy = f"http://{quote(user, safe='')}:{quote(password, safe='')}@{endpoint}"
        return {"status": "READY", "provider": "OXYLABS", "proxy_url": proxy}
    return {
        "status": "CONFIG_MISSING",
        "provider": None,
        "required": ["PT_UNBLOCKER_PROXY_URL", "or OXYLABS_USERNAME + OXYLABS_PASSWORD"],
    }


def _fetch(url: str, *, kind: str, proxy_cfg: dict[str, Any], timeout: int = 90) -> tuple[bytes, dict[str, Any]]:
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    proxy = proxy_cfg["proxy_url"]
    headers = {
        "User-Agent": UA,
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    if proxy_cfg.get("provider") == "OXYLABS":
        headers["x-oxylabs-geo-location"] = str(os.getenv("PT_UNBLOCKER_GEO", "United States"))
        if kind == "html":
            headers["X-Oxylabs-Render"] = "html"
    headers["Accept"] = "text/csv,text/plain,*/*" if kind == "csv" else "text/html,application/xhtml+xml,*/*;q=0.8"

    r = requests.get(
        url,
        headers=headers,
        proxies={"http": proxy, "https": proxy},
        timeout=timeout,
        allow_redirects=True,
        verify=False,
    )
    r.raise_for_status()
    raw = r.content
    ok, diag = validate_csv(raw) if kind == "csv" else validate_live_html(raw)
    if not ok:
        raise RuntimeError(f"validation_failed:{diag}")
    return raw, {"status": "PASS", "http_status": int(r.status_code), "validation": diag}


def _existing_valid(bucket, path: str, kind: str) -> bool:
    try:
        blob = bucket.blob(path)
        if not blob.exists():
            return False
        raw = blob.download_as_bytes()
        ok, _ = validate_csv(raw) if kind == "csv" else validate_live_html(raw)
        return bool(ok)
    except Exception:
        return False


@dataclass
class SourceResult:
    name: str
    url: str
    status: str
    provider: str
    bytes: int = 0
    sha256: str = ""
    fetched_utc: str = ""
    raw_blob: str = ""
    snapshot_blob: str = ""
    error: str = ""
    validation: Optional[dict[str, Any]] = None


def run_prediction_tracker_cloud_feeder(*, storage_client, bucket_name: str = "sharp-models", include_history: bool = True,
                                        force_history: bool = False, timeout: int = 90, log_func=print) -> dict[str, Any]:
    cfg = _proxy_config()
    if cfg.get("status") != "READY":
        out = {"status": "CONFIG_MISSING", "source_tag": SOURCE_TAG, "provider": None, "production_authority": 0}
        log_func("[NCAAF-PT-CLOUD-FEEDER] status=CONFIG_MISSING required=PT_UNBLOCKER_PROXY_URL_or_OXYLABS_credentials production_authority=0")
        return out

    bucket = storage_client.bucket(bucket_name)
    ts = _now(); stamp = _stamp(ts)
    jobs: list[tuple[str, str, str, str, str]] = []
    if include_history:
        for season in range(2022, 2026):
            jobs.append((f"ncaa{season}", PT_BASE_URL + f"/ncaa{season}.csv", "csv", f"{PT_RAW_PREFIX}/ncaa{season}.csv", "text/csv"))
    jobs.extend([
        ("ncaa2026", PT_BASE_URL + "/ncaa2026.csv", "csv", f"{PT_RAW_PREFIX}/ncaa2026.csv", "text/csv"),
        ("ncaapredictions", PT_LIVE_CSV_URL, "csv", f"{PT_RAW_PREFIX}/ncaapredictions.csv", "text/csv"),
        ("predncaa_live_page", PT_LIVE_PAGE_URL, "html", f"{PT_RAW_PREFIX}/predncaa_live_page.txt", "text/html"),
    ])

    sources: dict[str, Any] = {}; failures: dict[str, str] = {}
    for name, url, kind, raw_path, content_type in jobs:
        historical_completed = name in {"ncaa2022", "ncaa2023", "ncaa2024", "ncaa2025"}
        if historical_completed and not force_history and _existing_valid(bucket, raw_path, kind):
            rec = SourceResult(name=name, url=url, status="SKIPPED_VALID_EXISTING", provider=str(cfg["provider"]), raw_blob=f"gs://{bucket_name}/{raw_path}")
            sources[name] = asdict(rec)
            log_func(f"[NCAAF-PT-CLOUD-FEEDER] source={name} status=SKIPPED_VALID_EXISTING path=gs://{bucket_name}/{raw_path} authority=0")
            continue
        try:
            raw, diag = _fetch(url, kind=kind, proxy_cfg=cfg, timeout=timeout)
            sha = hashlib.sha256(raw).hexdigest()
            snap_path = f"{PT_SNAPSHOT_PREFIX}/{stamp}/{raw_path.rsplit('/', 1)[-1]}"
            bucket.blob(raw_path).upload_from_string(raw, content_type=content_type)
            bucket.blob(snap_path).upload_from_string(raw, content_type=content_type)
            rec = SourceResult(
                name=name, url=url, status="PASS", provider=str(cfg["provider"]), bytes=len(raw), sha256=sha,
                fetched_utc=_iso(), raw_blob=f"gs://{bucket_name}/{raw_path}", snapshot_blob=f"gs://{bucket_name}/{snap_path}",
                validation=diag.get("validation"),
            )
            sources[name] = asdict(rec)
            log_func(f"[NCAAF-PT-CLOUD-FEEDER] source={name} status=PASS provider={cfg['provider']} bytes={len(raw)} sha={sha[:16]} authority=0")
        except Exception as exc:
            err = f"{type(exc).__name__}:{exc}"
            failures[name] = err
            sources[name] = asdict(SourceResult(name=name, url=url, status="FAILED", provider=str(cfg["provider"]), error=err))
            log_func(f"[NCAAF-PT-CLOUD-FEEDER] source={name} status=FAILED provider={cfg['provider']} error={err} authority=0")

    pass_count = sum(1 for v in sources.values() if v.get("status") in {"PASS", "SKIPPED_VALID_EXISTING"})
    refreshed_count = sum(1 for v in sources.values() if v.get("status") == "PASS")
    status = "PASS" if not failures else ("PARTIAL" if pass_count else "FAILED")
    manifest = {
        "status": status,
        "source_tag": SOURCE_TAG,
        "updated_utc": _iso(ts),
        "network_role": "CLOUD_WEB_UNBLOCKER",
        "provider": cfg.get("provider"),
        "bucket": bucket_name,
        "source_count": len(sources),
        "refreshed_count": refreshed_count,
        "pass_count": pass_count,
        "snapshot_prefix": f"gs://{bucket_name}/{PT_SNAPSHOT_PREFIX}/{stamp}/",
        "sources": sources,
        "failures": failures,
        "policy": {
            "cloud_only": True,
            "preserve_exact_source_bytes": True,
            "reject_challenge_pages": True,
            "do_not_overwrite_last_good_on_failure": True,
            "model_side_contract": "GCS_FIRST",
            "production_authority": 0,
        },
    }
    bucket.blob(PT_MANIFEST_BLOB).upload_from_string(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"), content_type="application/json")
    log_func(f"[NCAAF-PT-CLOUD-FEEDER] status={status} provider={cfg.get('provider')} passed={pass_count}/{len(sources)} refreshed={refreshed_count} failures={len(failures)} manifest=gs://{bucket_name}/{PT_MANIFEST_BLOB} authority=0")
    return manifest


def self_test() -> dict[str, Any]:
    good_csv = b"Home,Road,line,lineespn,linedokter\nAlpha,Beta,3,4,5\n"
    bad = b"<html><title>Just a moment...</title>Checking your browser</html>"
    live = b"<html>Home Road ESPN FPI Pi-Ratings Bias Dokter Keeper Pigskin Index</html>"
    a, _ = validate_csv(good_csv); b, _ = validate_csv(bad); c, _ = validate_live_html(live)
    return {"status": "PASS" if a and not b and c else "FAIL", "csv_good": a, "challenge_rejected": not b, "live_names": c, "source_tag": SOURCE_TAG}


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2, sort_keys=True))
