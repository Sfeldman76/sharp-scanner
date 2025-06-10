FROM python:3.10

WORKDIR /app
COPY . .


RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "sharp_line_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

