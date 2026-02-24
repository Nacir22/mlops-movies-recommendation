FROM python:3.10-slim

RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    boto3==1.28.85 \
    psycopg2-binary==2.9.9

EXPOSE 5000

# Healthcheck intégré
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
    CMD curl -f http://localhost:5000/health || exit 1
