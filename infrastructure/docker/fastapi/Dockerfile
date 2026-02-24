FROM python:3.10-slim

# ── Dépendances système ────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Utilisateur non-root ───────────────────────────────────────────
RUN useradd -m -u 1000 app
WORKDIR /app

# ── Dépendances Python ─────────────────────────────────────────────
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn[standard]==0.29.0 \
    pydantic==2.6.4 \
    scikit-learn==1.3.2 \
    pandas==2.1.4 \
    numpy==1.26.3 \
    mlflow==2.9.2 \
    boto3==1.28.85 \
    prometheus-client==0.20.0 \
    python-dotenv==1.0.0

# ── Code de l'application ──────────────────────────────────────────
COPY serving/      /app/serving/
COPY training/     /app/training/
COPY data/schemas/ /app/data/schemas/
COPY models/       /app/models/

# Fichier vide pour les imports relatifs
RUN touch /app/serving/__init__.py \
          /app/serving/fastapi/__init__.py \
          /app/training/__init__.py \
          /app/training/models/__init__.py

USER app

# ── Variables d'environnement par défaut ──────────────────────────
ENV PYTHONPATH=/app \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MODEL_NAME=naive-bayes-recommender \
    MODEL_STAGE=Production \
    LOG_LEVEL=info \
    PORT=8000

EXPOSE 8000

# ── Healthcheck ────────────────────────────────────────────────────
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Démarrage ─────────────────────────────────────────────────────
CMD uvicorn serving.fastapi.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --log-level ${LOG_LEVEL} \
    --workers 1
