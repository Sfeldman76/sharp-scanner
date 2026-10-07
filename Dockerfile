FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements_weekly_uploader.txt .
RUN pip install -r requirements_weekly_uploader.txt \
    && pip install "google-cloud-storage>=2.16,<4"

COPY ncaaf_weekly_core.py .
COPY ncaaf_weekly_uploader.py .
COPY ncaaf_context_rebuild.sql .

CMD exec streamlit run ncaaf_weekly_uploader.py \
    --server.address=0.0.0.0 \
    --server.port=${PORT:-8080} \
    --server.headless=true \
    --browser.gatherUsageStats=false
