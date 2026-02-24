#!/bin/bash
# scripts/setup_mlflow.sh
# Vérifie la connexion MLflow et crée l'expérience initiale
# Usage : bash scripts/setup_mlflow.sh

set -e

MLFLOW_URI=${MLFLOW_TRACKING_URI:-http://localhost:5000}
EXPERIMENT=${MLFLOW_EXPERIMENT_NAME:-movielens-recommender}

echo "🔍 Vérification MLflow — $MLFLOW_URI"

# Attendre que MLflow soit prêt
MAX_RETRIES=10
for i in $(seq 1 $MAX_RETRIES); do
    if curl -sf "$MLFLOW_URI/health" > /dev/null 2>&1; then
        echo "✅ MLflow accessible"
        break
    fi
    echo "   Tentative $i/$MAX_RETRIES — attente 5s..."
    sleep 5
    if [ $i -eq $MAX_RETRIES ]; then
        echo "❌ MLflow non accessible après $MAX_RETRIES tentatives"
        exit 1
    fi
done

# Créer l'expérience via l'API REST
echo "📊 Création de l'expérience : $EXPERIMENT"
curl -sf -X POST "$MLFLOW_URI/api/2.0/mlflow/experiments/create" \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"$EXPERIMENT\"}" \
    && echo "✅ Expérience créée : $EXPERIMENT" \
    || echo "ℹ️  Expérience déjà existante (normal)"

echo ""
echo "🎉 MLflow prêt — UI : $MLFLOW_URI"
