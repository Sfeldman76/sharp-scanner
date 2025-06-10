# ✅ Use the latest stable Python 3.10 base image
FROM python:3.10

# ✅ Set working directory
WORKDIR /app

# ✅ Copy project files into container
COPY . .

# ✅ Install system dependencies (needed for PyDrive2, some Google auth libs)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ✅ Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# ✅ Set environment variables (fallback or default use)
ENV GDRIVE_CREDS_PATH=/secrets/drive/drive_service_account.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/secrets/oauth/credentials.json

# ✅ Expose the Streamlit default port
EXPOSE 8501

# ✅ Run the Streamlit app
CMD ["streamlit", "run", "sharp_line_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
