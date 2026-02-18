# progress.py
import json
import time
import random
from dataclasses import dataclass


@dataclass
class ProgressEvent:
    ts: float
    stage: str
    msg: str
    pct: float | None = None
    meta: dict | None = None


class ProgressWriter:
    """
    Writes progress events to GCS without mutating the same object repeatedly.

    Given gs://bucket/path/run.jsonl, this writer stores:
      - gs://bucket/path/run/events/<ts>-<rand>.json   (one event per object)
      - gs://bucket/path/run/latest.json              (optional pointer)
    """

    def __init__(self, gcs_uri: str, gcs_client, *, write_latest: bool = True, max_retries: int = 8):
        self.gcs_uri = gcs_uri
        self.gcs = gcs_client
        self.write_latest = write_latest
        self.max_retries = max_retries

        self._bucket, self._base_prefix = self._parse_base_prefix(gcs_uri)

    def _parse_gcs_uri(self, uri: str):
        assert uri.startswith("gs://")
        rest = uri[5:]
        b, p = rest.split("/", 1)
        return b, p

    def _parse_base_prefix(self, uri: str):
        b, p = self._parse_gcs_uri(uri)
        # If user passed .../something.jsonl, use prefix without suffix for folder-like layout
        # e.g. train-progress/.../cbee9242.jsonl -> train-progress/.../cbee9242
        if p.endswith(".jsonl"):
            p = p[:-5]
        return b, p.rstrip("/")

    def _upload_with_retry(self, blob, data: bytes, content_type: str):
        # Exponential backoff + jitter for transient 429/503s
        for attempt in range(self.max_retries):
            try:
                blob.upload_from_string(data, content_type=content_type)
                return
            except Exception as e:
                # Only retry common transient GCS failures
                msg = str(e)
                is_retryable = ("429" in msg) or ("TooManyRequests" in msg) or ("503" in msg) or ("ServiceUnavailable" in msg)
                if not is_retryable or attempt == self.max_retries - 1:
                    raise
                backoff = (2 ** attempt) * 0.25
                jitter = random.uniform(0, 0.25)
                time.sleep(backoff + jitter)

    def emit(self, stage: str, msg: str, pct: float | None = None, meta: dict | None = None):
        ev = ProgressEvent(ts=time.time(), stage=stage, msg=msg, pct=pct, meta=meta)
        payload = json.dumps(ev.__dict__, default=str).encode("utf-8")

        bucket = self.gcs.bucket(self._bucket)

        # Unique object per event (no per-object mutation rate limit issues)
        ts_ms = int(ev.ts * 1000)
        rand = random.randint(100000, 999999)
        event_path = f"{self._base_prefix}/events/{ts_ms}-{rand}.json"
        event_blob = bucket.blob(event_path)
        self._upload_with_retry(event_blob, payload, content_type="application/json")

        # Optional: update a small latest pointer (still a single object mutation, but low-frequency)
        if self.write_latest:
            latest_path = f"{self._base_prefix}/latest.json"
            latest_blob = bucket.blob(latest_path)
            self._upload_with_retry(latest_blob, payload, content_type="application/json")
