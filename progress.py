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

    Given gs://bucket/path/run.jsonl (or .json), this writer stores:
      - gs://bucket/path/run/events/<ts>-<rand>.json
      - gs://bucket/path/run/latest.json
    """

    def __init__(
        self,
        gcs_uri: str,
        gcs_client,
        *,
        write_latest: bool = True,
        max_retries: int = 8,
        latest_min_interval_s: float = 1.0,  # throttle latest.json updates
    ):
        self.gcs_uri = gcs_uri
        self.gcs = gcs_client
        self.write_latest = write_latest
        self.max_retries = max_retries
        self.latest_min_interval_s = latest_min_interval_s
        self._last_latest_ts = 0.0

        self._bucket, self._base_prefix = self._parse_base_prefix(gcs_uri)

    def _parse_gcs_uri(self, uri: str):
        assert uri.startswith("gs://")
        rest = uri[5:]
        b, p = rest.split("/", 1)
        return b, p

    def _parse_base_prefix(self, uri: str):
        b, p = self._parse_gcs_uri(uri)
        p = p.rstrip("/")

        # ✅ handle both .jsonl and .json safely
        lower = p.lower()
        if lower.endswith(".jsonl"):
            p = p[:-len(".jsonl")]
        elif lower.endswith(".json"):
            p = p[:-len(".json")]

        return b, p.rstrip("/")

    def _upload_with_retry(self, blob, data: bytes, content_type: str):
        for attempt in range(self.max_retries):
            try:
                blob.upload_from_string(data, content_type=content_type)
                return
            except Exception as e:
                msg = str(e)
                is_retryable = (
                    ("429" in msg)
                    or ("TooManyRequests" in msg)
                    or ("503" in msg)
                    or ("ServiceUnavailable" in msg)
                )
                if not is_retryable or attempt == self.max_retries - 1:
                    raise
                backoff = (2 ** attempt) * 0.25
                jitter = random.uniform(0, 0.25)
                time.sleep(backoff + jitter)

    def emit(self, stage: str, msg: str, pct: float | None = None, meta: dict | None = None):
        ev = ProgressEvent(ts=time.time(), stage=stage, msg=msg, pct=pct, meta=meta)
        payload = json.dumps(ev.__dict__, default=str).encode("utf-8")

        bucket = self.gcs.bucket(self._bucket)

        # Unique object per event
        ts_ms = int(ev.ts * 1000)
        rand = random.randint(100000, 999999)
        event_path = f"{self._base_prefix}/events/{ts_ms}-{rand}.json"
        self._upload_with_retry(bucket.blob(event_path), payload, content_type="application/json")

        # Throttled latest pointer
        if self.write_latest:
            now = ev.ts
            if (now - self._last_latest_ts) >= self.latest_min_interval_s:
                latest_path = f"{self._base_prefix}/latest.json"
                self._upload_with_retry(bucket.blob(latest_path), payload, content_type="application/json")
                self._last_latest_ts = now
