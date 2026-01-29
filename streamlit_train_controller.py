# streamlit_train_controller.py
import json
import time
import uuid
import shutil
import subprocess

import streamlit as st
from google.cloud import storage


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


def start_job_with_gcloud(job_name: str, region: str, sport: str, market: str, progress_uri: str):
    # Hard fail early if gcloud isn't available (common on Cloud Run containers)
    if shutil.which("gcloud") is None:
        raise RuntimeError("gcloud not found in this environment. If Streamlit runs on Cloud Run, use REST trigger instead.")

    subprocess.Popen([
        "gcloud", "run", "jobs", "execute", job_name,
        "--region", region,
        "--update-env-vars", f"SPORT={sport},MARKET={market},PROGRESS_URI={progress_uri}",
    ])


st.title("📈 Training Controller (Cloud Run Job)")

# session state
st.session_state.setdefault("exec_id", None)
st.session_state.setdefault("progress_uri", None)

sport = st.selectbox("Sport", ["NBA", "NFL", "NCAAB", "WNBA"], index=0)
market = st.selectbox("Market", ["All", "spreads", "h2h", "totals"], index=0)
auto = st.checkbox("Auto-refresh", value=True)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🚀 Start Training Job"):
        exec_id = str(uuid.uuid4())[:8]
        progress_uri = f"gs://sharp-models/progress/{sport}_{market}_{exec_id}.jsonl"

        st.session_state.exec_id = exec_id
        st.session_state.progress_uri = progress_uri

        try:
            start_job_with_gcloud(
                job_name="sharp-train-job",
                region="us-central1",
                sport=sport,
                market=market,
                progress_uri=progress_uri,
            )
            st.success("Training job started 🚀")
        except Exception as e:
            st.error(f"Failed to start job: {e}")

with col2:
    st.write("")
    st.write("")
    st.caption("Tip: If this is deployed on Cloud Run, gcloud won’t exist. Use REST trigger.")

st.divider()

if not st.session_state.progress_uri:
    st.info("Start a job to see progress here.")
    st.stop()

st.caption(f"Progress file: {st.session_state.progress_uri}")

gcs = storage.Client()
lines = read_progress_lines(gcs, st.session_state.progress_uri, max_lines=200)

if lines:
    events = [json.loads(ln) for ln in lines]
    last = events[-1]
    pct = last.get("pct")

    if pct is not None:
        st.progress(float(pct))

    st.code("\n".join(lines[-40:]), language="json")

    if last.get("stage") == "done":
        st.success("Job finished ✅")
    elif last.get("stage") == "error":
        st.error(f"Job error: {last.get('msg')}")
else:
    st.info("No progress events yet… (job may still be starting)")

# Auto-refresh without blocking loops
if auto:
    time.sleep(2)
    st.rerun()

if st.button("🔄 Refresh now"):
    st.rerun()
