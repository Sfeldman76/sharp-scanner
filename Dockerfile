FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "sharp_line_dashboard.py", "--server.port=8501", "--server.enableCORS=false"]
