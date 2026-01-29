# streamlit_train_controller.py
import json, time, subprocess, uuid
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

st.title("📈 Training Controller (Cloud Run Job)")

if "exec_id" not in st.session_state:
    st.session_state.exec_id = None
if "progress_uri" not in st.session_state:
    st.session_state.progress_uri = None

sport = st.selectbox("Sport", ["NBA","NFL","NCAAB","WNBA"], index=0)
market = st.selectbox("Market", ["All","spreads","h2h","totals"], index=0)

col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Start Training Job"):
        exec_id = str(uuid.uuid4())[:8]
        progress_uri = f"gs://sharp-models/progress/{sport}_{market}_{exec_id}.jsonl"

        st.session_state.exec_id = exec_id
        st.session_state.progress_uri = progress_uri

        # Trigger job (simple CLI way). Better is REST, but this is fine to start.
        cmd = [
            "gcloud", "run", "jobs", "execute", "sharp-train-job",
            "--region=us-central1",
            "--update-env-vars", f"SPORT={sport},MARKET={market},PROGRESS_URI={progress_uri}",
        ]
        out = subprocess.run(cmd, capture_output=True, text=True)
        st.code(out.stdout or out.stderr)

with col2:
    auto = st.checkbox("Auto-refresh", value=True)

st.divider()

if st.session_state.progress_uri:
    st.caption(f"Progress file: {st.session_state.progress_uri}")

    gcs = storage.Client()
    box = st.empty()

    # A “bounded” poll loop: keeps the page alive without running forever
    # (Streamlit will rerun on interactions; this just helps show progress now.)
    polls = 30 if auto else 1  # ~30*2s = 60s window per run
    for _ in range(polls):
        lines = read_progress_lines(gcs, st.session_state.progress_uri, max_lines=200)
        if lines:
            events = [json.loads(ln) for ln in lines]
            last = events[-1]
            pct = last.get("pct")
            if pct is not None:
                st.progress(float(pct))
            box.code("\n".join(lines[-40:]), language="json")

            if last.get("stage") == "done":
                st.success("Job finished ✅")
                break
        else:
            box.info("No progress events yet…")

        if not auto:
            break
        time.sleep(2)

    st.button("🔄 Refresh now")
else:
    st.info("Start a job to see progress here.")
