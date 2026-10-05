FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "uvicorn[standard]==0.34.2" "gunicorn==23.0.0"
COPY backend ./backend
COPY data ./data
COPY frontend/dist ./frontend/dist
EXPOSE 8000
ENV APP_ENV=production \
    DATABASE_URL=sqlite:////data/athlete_lens.db \
    MODEL_DIR=/tmp/athlete_lens_models
RUN mkdir -p /data
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
# One worker keeps SQLite writes simple; scale with threads/instances + Postgres for bigger squads.
CMD ["gunicorn", "backend.app.main:app", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--workers", "1", "--timeout", "90"]
