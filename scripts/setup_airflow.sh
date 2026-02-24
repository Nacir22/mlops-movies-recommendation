#!/bin/bash
# scripts/setup_airflow.sh
# Initialisation d'Airflow pour le projet MovieLens MLOps
# Usage : bash scripts/setup_airflow.sh

set -e
echo "🌬️  Configuration Airflow — MovieLens MLOps"
echo "============================================="

# 1. Vérifier Docker Compose
if ! docker compose version &>/dev/null; then
    echo "❌ Docker Compose non disponible"
    exit 1
fi

# 2. Démarrer la base de données (PostgreSQL doit être up)
echo ""
echo "📦 Vérification de PostgreSQL..."
docker compose ps postgres | grep -q "running" \
    && echo "   ✅ PostgreSQL déjà démarré" \
    || { docker compose up postgres -d; sleep 10; echo "   ✅ PostgreSQL démarré"; }

# 3. Initialiser Airflow (crée les tables + user admin)
echo ""
echo "🔧 Initialisation Airflow (db migrate + user admin)..."
docker compose --profile airflow run --rm airflow-init
echo "   ✅ Airflow initialisé (admin/admin)"

# 4. Démarrer webserver + scheduler
echo ""
echo "🚀 Démarrage Airflow (webserver + scheduler)..."
docker compose --profile airflow up airflow-webserver airflow-scheduler -d

# 5. Attendre que le webserver soit prêt
echo ""
echo "⏳ Attente du webserver Airflow (max 60s)..."
for i in $(seq 1 12); do
    if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
        echo "   ✅ Airflow accessible : http://localhost:8080"
        break
    fi
    echo "   Tentative $i/12..."
    sleep 5
done

# 6. Créer les connexions Airflow via CLI
echo ""
echo "🔌 Création des connexions Airflow..."

# Connexion MinIO (S3)
docker compose --profile airflow exec airflow-webserver \
    airflow connections add minio_s3 \
    --conn-type aws \
    --conn-host "${MINIO_ENDPOINT:-http://minio:9000}" \
    --conn-login "${MINIO_ACCESS_KEY:-minioadmin}" \
    --conn-password "${MINIO_SECRET_KEY:-minioadmin}" 2>/dev/null \
    && echo "   ✅ Connexion minio_s3 créée" \
    || echo "   ℹ️  Connexion minio_s3 déjà existante"

# Connexion Redis
docker compose --profile airflow exec airflow-webserver \
    airflow connections add redis_default \
    --conn-type redis \
    --conn-host "${FEAST_REDIS_HOST:-redis}" \
    --conn-port 6379 2>/dev/null \
    && echo "   ✅ Connexion redis_default créée" \
    || echo "   ℹ️  Connexion redis_default déjà existante"

# 7. Vérifier les DAGs
echo ""
echo "📋 DAGs disponibles :"
docker compose --profile airflow exec airflow-webserver \
    airflow dags list 2>/dev/null | grep movielens || echo "   (en cours de chargement...)"

echo ""
echo "🎉 Airflow configuré !"
echo ""
echo "   UI     : http://localhost:8080  (admin/admin)"
echo "   DAGs   : movielens_ingestion | feast_materialization | trigger_kubeflow_training"
echo ""
echo "Déclencher un DAG manuellement :"
echo "   docker compose --profile airflow exec airflow-webserver \\"
echo "       airflow dags trigger movielens_ingestion"
