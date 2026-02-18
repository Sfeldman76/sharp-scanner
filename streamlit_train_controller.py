# streamlit_train_controller.py
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
    bucket, path = _parse_gcs_uri(progress_uri)
    if path.lower().endswith(".jsonl"):
        path = path[: -len(".jsonl")]
    return bucket, path.strip("/")


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


def list_event_blob_names(gcs_client: storage.Client, progress_uri: str, *, limit: int = 120) -> List[str]:
    bucket_name, base = _base_prefix_from_progress_uri(progress_uri)
    prefix = f"{base}/events/"
    bucket = gcs_client.bucket(bucket_name)
    try:
        blobs = list(bucket.list_blobs(prefix=prefix))
    except Exception:
        return []
    if not blobs:
        return []
    blobs.sort(key=lambda b: b.name)  # chronological
    return [b.name for b in blobs[-limit:]]


def read_events_by_names(gcs_client: storage.Client, bucket_name: str, blob_names: List[str]) -> List[Dict]:
    bucket = gcs_client.bucket(bucket_name)
    out: List[Dict] = []
    for name in blob_names:
        try:
            out.append(json.loads(bucket.blob(name).download_as_text()))
        except Exception:
            out.append({"stage": "parse_error", "msg": f"Could not parse {name}"})
    return out


# -----------------------------
# Cloud Run Jobs REST trigger
# -----------------------------
def start_job_with_rest(*, job_name: str, region: str, project_id: str, env: dict):
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
refresh_s = st.slider("Refresh interval (seconds)", 1, 10, 2)

with st.expander("⚙️ Display options", expanded=False):
    show_log = st.checkbox("Show event log", value=True)
    log_tail = st.slider("Log tail (events)", min_value=20, max_value=200, value=60, step=10)

# Show a tick so you can SEE reruns happening
st.caption(f"UI tick: {time.strftime('%H:%M:%S')}")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🚀 Start Training Job"):
        exec_id = str(uuid.uuid4())[:8]

        # ✅ Match your job/log convention
        progress_uri = f"gs://{PROGRESS_BUCKET}/train-progress/{sport}/{market}/{exec_id}.jsonl"

        st.session_state.exec_id = exec_id
        st.session_state.progress_uri = progress_uri

        env = {"SPORT": sport, "MARKET": market, "PROGRESS_URI": progress_uri}
        try:
            start_job_with_rest(job_name=JOB_NAME, region=REGION, project_id=PROJECT_ID, env=env)
            st.success("Training job started 🚀")
        except Exception as e:
            st.error(f"Failed to start job: {e}")

with col2:
    st.caption("Triggers the Cloud Run Job via REST (works on Cloud Run; no gcloud needed).")

st.divider()

# -----------------------------
# Progress viewer
# -----------------------------
if not st.session_state.progress_uri:
    st.info("Start a job to see progress here.")
else:
    progress_uri = st.session_state.progress_uri
    bucket_name, base = _base_prefix_from_progress_uri(progress_uri)

    st.caption(f"Progress target: {progress_uri}")
    st.caption(f"Polling latest: gs://{bucket_name}/{base}/latest.json")
    st.caption(f"Polling events: gs://{bucket_name}/{base}/events/")

    gcs = storage.Client()
    latest = read_latest_event(gcs, progress_uri)

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
    else:
        st.warning("No latest.json yet… (job may still be starting)")

    if show_log:
        names = list_event_blob_names(gcs, progress_uri, limit=log_tail)
        if names:
            tail_events = read_events_by_names(gcs, bucket_name, names[-log_tail:])
            st.code("\n".join(json.dumps(e, default=str) for e in tail_events[-40:]), language="json")
        else:
            st.caption("No event objects found yet.")


# -----------------------------
# Refresh controls
# -----------------------------
if st.button("🔄 Refresh now"):
    st.rerun()

# ✅ Safe-ish auto refresh without sleep-looping the app forever:
# Use query params to trigger reruns (Streamlit supports this reliably).
if auto:
    # This causes the browser to request the page again after N seconds.
    st.markdown(
        f"""
        <script>
        setTimeout(function() {{
            window.location.reload();
        }}, {refresh_s * 1000});
        </script>
        """,
        unsafe_allow_html=True,
    )
