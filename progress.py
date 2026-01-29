# progress.py
import json, time
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
    Writes JSONL events to gs://.../progress.jsonl
    """
    def __init__(self, gcs_uri: str, gcs_client):
        self.gcs_uri = gcs_uri
        self.gcs = gcs_client
        self._bucket, self._blob = self._parse_gcs_uri(gcs_uri)

    def _parse_gcs_uri(self, uri: str):
        assert uri.startswith("gs://")
        rest = uri[5:]
        b, p = rest.split("/", 1)
        return b, p

    def emit(self, stage: str, msg: str, pct: float | None = None, meta: dict | None = None):
        ev = ProgressEvent(ts=time.time(), stage=stage, msg=msg, pct=pct, meta=meta)
        line = json.dumps(ev.__dict__, default=str) + "\n"

        bucket = self.gcs.bucket(self._bucket)
        blob = bucket.blob(self._blob)

        # append semantics: download existing (small) + upload back
        # For long runs, better to write chunked files; this is minimal + works.
        existing = b""
        if blob.exists():
            existing = blob.download_as_bytes()
        blob.upload_from_string(existing + line.encode("utf-8"), content_type="application/jsonl")
