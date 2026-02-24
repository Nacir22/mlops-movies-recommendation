FROM apache/airflow:2.8.1-python3.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
USER airflow

# Dépendances supplémentaires pour nos DAGs
RUN pip install --no-cache-dir \
    scikit-learn==1.3.2 \
    pandas==2.1.4 \
    numpy==1.26.3 \
    boto3==1.28.85 \
    mlflow==2.9.2 \
    feast==0.35.0 \
    redis==5.0.1 \
    python-dotenv==1.0.0

# Variables Airflow par défaut
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__CORE__EXECUTOR=LocalExecutor \
    AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True
