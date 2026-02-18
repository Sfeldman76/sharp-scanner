# streamlit_train_controller.py
import json
import time
import uuid
from typing import List, Dict, Tuple

import streamlit as st
from google.cloud import storage

import google.auth
from google.auth.transport.requests import AuthorizedSession


# -----------------------------
# GCS helpers
# -----------------------------
def _parse_gcs_uri(uri: str) -> Tuple[str, str]:
    assert uri.startswith("gs://")
    rest = uri[5:]
    bucket_name, path = rest.split("/", 1)
    return bucket_name, path


def _base_prefix_from_progress_uri(progress_uri: str) -> Tuple[str, str]:
    """
    Option A writer treats gs://bucket/path/run.jsonl as base prefix gs://bucket/path/run
    and stores:
      - base/latest.json
      - base/events/<ts>-<rand>.json
    """
    bucket, path = _parse_gcs_uri(progress_uri)
    if path.endswith(".jsonl"):
        path = path[:-5]
    return bucket, path.rstrip("/")


# -----------------------------
# Option A progress readers
# -----------------------------
def read_latest_event(gcs_client: storage.Client, progress_uri: str) -> Dict | None:
    bucket_name, base = _base_prefix_from_progress_uri(progress_uri)
    blob = gcs_client.bucket(bucket_name).blob(f"{base}/latest.json")
    if not blob.exists():
        return None
    try:
        return json.loads(blob.download_as_text())
    except Exception:
        return None


def read_recent_events(
    gcs_client: storage.Client,
    progress_uri: str,
    max_events: int = 200,
) -> List[Dict]:
    """
    Reads recent events from Option A event objects. Uses name sorting:
      .../events/<ts_ms>-<rand>.json
    so lexicographic order is chronological.
    """
    bucket_name, base = _base_prefix_from_progress_uri(progress_uri)
    prefix = f"{base}/events/"
    bucket = gcs_client.bucket(bucket_name)

    # List blobs under events/
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        return []

    # Sort by object name (chronological given our <ts_ms>-<rand>.json naming)
    blobs.sort(key=lambda b: b.name)

    # Take last N blobs only
    blobs = blobs[-max_events:]

    events: List[Dict] = []
    for b in blobs:
        try:
            events.append(json.loads(b.download_as_text()))
        except Exception:
            # Keep something visible in the log
            events.append({"stage": "parse_error", "msg": f"Could not parse {b.name}"})
    return events


# -----------------------------
# Legacy JSONL reader (fallback)
# -----------------------------
def read_progress_lines_legacy_jsonl(gcs_client, uri: str, max_lines: int = 200):
    bucket_name, path = _parse_gcs_uri(uri)
    blob = gcs_client.bucket(bucket_name).blob(path)
    if not blob.exists():
        return []
    data = blob.download_as_text()
    lines = [ln for ln in data.splitlines() if ln.strip()]
    return lines[-max_lines:]


# -----------------------------
# Cloud Run Jobs REST trigger
# -----------------------------
def start_job_with_rest(*, job_name: str, region: str, project_id: str, env: dict):
    """
    Trigger Cloud Run Job execution via REST and override container env vars.
    Works when Streamlit itself is running on Cloud Run (no gcloud needed).
    """
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(creds)

    url = (
        f"https://{region}-run.googleapis.com/apis/run.googleapis.com/v1/"
        f"namespaces/{project_id}/jobs/{job_name}:run"
    )

    env_list = [{"name": k, "value": str(v)} for k, v in env.items()]
    body = {"overrides": {"containerOverrides": [{"env": env_list}]}}
    resp = session.post(url, json=body, timeout=30)

    if resp.status_code >= 300:
        raise RuntimeError(f"Job run failed: {resp.status_code} {resp.text}")

    return resp.json()


# -----------------------------
# UI
# -----------------------------
st.title("📈 Training Controller (Cloud Run Job)")

# Config (adjust if needed)
PROJECT_ID = st.secrets.get("GCP_PROJECT_ID", "sharplogger")
REGION = st.secrets.get("RUN_REGION", "us-central1")
JOB_NAME = st.secrets.get("RUN_JOB_NAME", "sharp-train-job")
PROGRESS_BUCKET = st.secrets.get("PROGRESS_BUCKET", "sharp-models")

# Session state
st.session_state.setdefault("exec_id", None)
st.session_state.setdefault("progress_uri", None)

sport = st.selectbox("Sport", ["NBA", "NFL", "NCAAB", "WNBA", "NCAAF", "MLB"], index=0)
market = st.selectbox("Market", ["All", "spreads", "h2h", "totals"], index=0)
auto = st.checkbox("Auto-refresh", value=True)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🚀 Start Training Job"):
        exec_id = str(uuid.uuid4())[:8]

        # Keep the same env var shape you already use.
        # Option A writer will treat this as base prefix without ".jsonl".
        progress_uri = f"gs://{PROGRESS_BUCKET}/progress/{sport}_{market}_{exec_id}.jsonl"

        st.session_state.exec_id = exec_id
        st.session_state.progress_uri = progress_uri

        env = {
            "SPORT": sport,
            "MARKET": market,
            "PROGRESS_URI": progress_uri,
        }

        try:
            start_job_with_rest(
                job_name=JOB_NAME,
                region=REGION,
                project_id=PROJECT_ID,
                env=env,
            )
            st.success("Training job started 🚀")
        except Exception as e:
            st.error(f"Failed to start job: {e}")

with col2:
    st.write("")
    st.write("")
    st.caption("This triggers the Cloud Run Job via REST (works on Cloud Run; no gcloud needed).")

st.divider()

# -----------------------------
# Progress viewer
# -----------------------------
if not st.session_state.progress_uri:
    st.info("Start a job to see progress here.")
    st.stop()

progress_uri = st.session_state.progress_uri
st.caption(f"Progress target: {progress_uri}")

gcs = storage.Client()

# Prefer Option A (latest.json + events/)
latest = read_latest_event(gcs, progress_uri)
events = read_recent_events(gcs, progress_uri, max_events=200)

# If Option A has nothing yet, fall back to legacy JSONL (helps during transition)
if latest is None and not events:
    lines = read_progress_lines_legacy_jsonl(gcs, progress_uri, max_lines=200)
    if lines:
        parsed = []
        for ln in lines:
            try:
                parsed.append(json.loads(ln))
            except Exception:
                parsed.append({"stage": "parse_error", "msg": ln})
        events = parsed
        latest = parsed[-1] if parsed else None

if latest:
    pct = latest.get("pct")
    if pct is not None:
        try:
            st.progress(max(0.0, min(1.0, float(pct))))
        except Exception:
            pass

    stage = latest.get("stage")
    msg = latest.get("msg", "")

    if stage == "done":
        st.success("Job finished ✅")
    elif stage == "error":
        st.error(f"Job error: {msg}")
    else:
        st.info(msg)

if events:
    # Show last ~40 events as a readable JSON log
    tail = events[-40:]
    st.code("\n".join(json.dumps(e, default=str) for e in tail), language="json")
else:
    st.info("No progress events yet… (job may still be starting)")

# -----------------------------
# Refresh controls
# -----------------------------
if st.button("🔄 Refresh now"):
    st.rerun()

if auto:
    time.sleep(2)
    st.rerun()
