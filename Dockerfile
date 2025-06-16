# Use the official lightweight Python image.
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY . .

# === Streamlit or FastAPI Entrypoint Config ===
# Use these as needed per deployment:

# Uncomment this line for Streamlit app:
# CMD ["streamlit", "run", "sharp_line_dashboard.py", "--server.port=8080", "--server.enableCORS=false"]

# Uncomment this line for FastAPI trigger:
# CMD ["uvicorn", "sharp_trigger_service:app", "--host", "0.0.0.0", "--port", "8080"]
