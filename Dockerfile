FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY data ./data
COPY frontend/dist ./frontend/dist
EXPOSE 8000
ENV DATABASE_URL=sqlite:////tmp/athlete_lens.db
ENV MODEL_DIR=/tmp/athlete_lens_models
CMD ["gunicorn", "backend.app.main:app", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--workers", "2", "--timeout", "90"]
