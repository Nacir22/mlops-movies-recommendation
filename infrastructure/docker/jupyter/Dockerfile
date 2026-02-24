FROM jupyter/scipy-notebook:python-3.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    boto3==1.28.85 \
    scikit-learn==1.3.2 \
    pyarrow==14.0.1 \
    python-dotenv==1.0.0

EXPOSE 8888
