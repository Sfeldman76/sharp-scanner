# streamlit_train_controller.py
import json
import time
import uuid

import streamlit as st
from google.cloud import storage

import google.auth
from google.auth.transport.requests import AuthorizedSession


# -----------------------------
# GCS progress reader
# -----------------------------
def read_progress_lines(gcs_client, uri: str, max_lines: int = 200):
    assert uri.startswith("gs://")
    rest = uri[5:]
    bucket_name, path = rest.split("/", 1)
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

sport = st.selectbox("Sport", ["NBA", "NFL", "NCAAB", "WNBA", "NCAAF","MLB"], index=0)
market = st.selectbox("Market", ["All", "spreads", "h2h", "totals"], index=0)
auto = st.checkbox("Auto-refresh", value=True)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🚀 Start Training Job"):
        exec_id = str(uuid.uuid4())[:8]
        progress_uri = f"gs://{PROGRESS_BUCKET}/progress/{sport}_{market}_{exec_id}.jsonl"

        st.session_state.exec_id = exec_id
        st.session_state.progress_uri = progress_uri

        # Env vars the job will read
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

st.caption(f"Progress file: {st.session_state.progress_uri}")

gcs = storage.Client()
lines = read_progress_lines(gcs, st.session_state.progress_uri, max_lines=200)

if lines:
    # Parse JSONL safely
    events = []
    for ln in lines:
        try:
            events.append(json.loads(ln))
        except Exception:
            events.append({"stage": "parse_error", "msg": ln})

    last = events[-1]
    pct = last.get("pct")

    if pct is not None:
        try:
            st.progress(max(0.0, min(1.0, float(pct))))
        except Exception:
            pass

    st.code("\n".join(lines[-40:]), language="json")

    stage = last.get("stage")
    if stage == "done":
        st.success("Job finished ✅")
        st.caption("Tip: start another job to retrain a different sport/market.")
    elif stage == "error":
        st.error(f"Job error: {last.get('msg')}")
else:
    st.info("No progress events yet… (job may still be starting)")

# -----------------------------
# Refresh controls
# -----------------------------
if st.button("🔄 Refresh now"):
    st.rerun()

# Auto-refresh without blocking loops (lightweight)
if auto:
    time.sleep(2)
    st.rerun()
