# streamlit_train_controller.py
import json
import time
import uuid
from typing import List, Dict, Tuple, Optional

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
      - base/events/<ts_ms>-<rand>.json
    """
    bucket, path = _parse_gcs_uri(progress_uri)
    if path.endswith(".jsonl"):
        path = path[:-5]
    return bucket, path.rstrip("/")


# -----------------------------
# Option A progress readers
# -----------------------------
def read_latest_event(gcs_client: storage.Client, progress_uri: str) -> Optional[Dict]:
    bucket_name, base = _base_prefix_from_progress_uri(progress_uri)
    blob = gcs_client.bucket(bucket_name).blob(f"{base}/latest.json")
    try:
        if not blob.exists():
            return None
        return json.loads(blob.download_as_text())
    except Exception:
        return None


def list_event_blob_names(
    gcs_client: storage.Client,
    progress_uri: str,
    *,
    limit: int = 80,
) -> List[str]:
    """
    List event object names under .../events/. We only need names (fast),
    then we can fetch a small tail for display.
    """
    bucket_name, base = _base_prefix_from_progress_uri(progress_uri)
    prefix = f"{base}/events/"
    bucket = gcs_client.bucket(bucket_name)
    try:
        blobs = list(bucket.list_blobs(prefix=prefix))
    except Exception:
        return []

    if not blobs:
        return []

    blobs.sort(key=lambda b: b.name)  # chronological due to ts_ms prefix
    blobs = blobs[-limit:]
    return [b.name for b in blobs]


def read_events_by_names(
    gcs_client: storage.Client,
    bucket_name: str,
    blob_names: List[str],
) -> List[Dict]:
    """
    Download + parse a small set of event JSON blobs.
    """
    bucket = gcs_client.bucket(bucket_name)
    out: List[Dict] = []
    for name in blob_names:
        try:
            txt = bucket.blob(name).download_as_text()
            out.append(json.loads(txt))
        except Exception:
            out.append({"stage": "parse_error", "msg": f"Could not parse {name}"})
    return out


# -----------------------------
# Legacy JSONL reader (fallback)
# -----------------------------
def read_progress_lines_legacy_jsonl(gcs_client: storage.Client, uri: str, max_lines: int = 200):
    bucket_name, path = _parse_gcs_uri(uri)
    blob = gcs_client.bucket(bucket_name).blob(path)
    try:
        if not blob.exists():
            return []
        data = blob.download_as_text()
        lines = [ln for ln in data.splitlines() if ln.strip()]
        return lines[-max_lines:]
    except Exception:
        return []


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

PROJECT_ID = st.secrets.get("GCP_PROJECT_ID", "sharplogger")
REGION = st.secrets.get("RUN_REGION", "us-central1")
JOB_NAME = st.secrets.get("RUN_JOB_NAME", "sharp-train-job")
PROGRESS_BUCKET = st.secrets.get("PROGRESS_BUCKET", "sharp-models")

st.session_state.setdefault("exec_id", None)
st.session_state.setdefault("progress_uri", None)

sport = st.selectbox("Sport", ["NBA", "NFL", "NCAAB", "WNBA", "NCAAF", "MLB"], index=0)
market = st.selectbox("Market", ["All", "spreads", "h2h", "totals"], index=0)
auto = st.checkbox("Auto-refresh", value=True)

with st.expander("⚙️ Display options", expanded=False):
    show_log = st.checkbox("Show event log (downloads recent events)", value=True)
    log_tail = st.slider("Log tail (events)", min_value=20, max_value=200, value=60, step=10)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🚀 Start Training Job"):
        exec_id = str(uuid.uuid4())[:8]
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

latest = read_latest_event(gcs, progress_uri)

# If Option A has nothing yet, fall back to legacy JSONL (transition-safe)
events: List[Dict] = []
if latest is None:
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

# ---- Render "latest" (fast path) ----
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
        st.caption("Tip: start another job to retrain a different sport/market.")
    elif stage == "error":
        st.error(f"Job error: {msg}")
    else:
        st.info(msg)
else:
    st.info("No progress yet… (job may still be starting)")

# ---- Render event tail (optional; avoids heavy work each refresh) ----
if show_log:
    bucket_name, _ = _parse_gcs_uri(progress_uri)

    # Cache the list of recent event blob names for 2 seconds to match your refresh cadence
    @st.cache_data(ttl=2.0, show_spinner=False)
    def _cached_event_names(uri: str, limit: int) -> List[str]:
        return list_event_blob_names(gcs, uri, limit=limit)

    names = _cached_event_names(progress_uri, log_tail)

    if names:
        # Download only the tail we want
        tail_events = read_events_by_names(gcs, bucket_name, names[-log_tail:])
        st.code("\n".join(json.dumps(e, default=str) for e in tail_events[-40:]), language="json")
    else:
        # If we’re in legacy mode, show the legacy parsed events (already loaded)
        if events:
            st.code("\n".join(json.dumps(e, default=str) for e in events[-40:]), language="json")
        else:
            st.caption("No event objects found yet.")

# -----------------------------
# Refresh controls
# -----------------------------
if st.button("🔄 Refresh now"):
    st.rerun()

if auto:
    time.sleep(2)
    st.rerun()
