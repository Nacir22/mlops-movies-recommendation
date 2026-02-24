#!/bin/bash
# scripts/setup_serving.sh
# Démarrage de l'API FastAPI MovieLens
# Usage : bash scripts/setup_serving.sh [--local | --docker | --k8s]

set -e
MODE=${1:-"--docker"}

echo "🌐 Déploiement API MovieLens — mode : $MODE"
echo "============================================="

case "$MODE" in

  --local)
    # Test local sans Docker (stdlib Python, pas de FastAPI requis)
    echo ""
    echo "🧪 Lancement du serveur de test local..."
    cd "$(dirname "$0")/.."
    python3 serving/fastapi/test_server.py --test \
        && echo "✅ Tous les tests API passent" \
        || { echo "❌ Échec des tests"; exit 1; }
    echo ""
    echo "Pour démarrer le serveur interactif :"
    echo "  python serving/fastapi/test_server.py"
    ;;

  --fastapi)
    # Démarrage avec FastAPI/uvicorn (si installé)
    echo ""
    echo "🚀 Démarrage avec uvicorn..."
    cd "$(dirname "$0")/.."
    pip install fastapi uvicorn pydantic -q 2>/dev/null || true
    uvicorn serving.fastapi.main:app \
        --host 0.0.0.0 --port 8000 --reload \
        --log-level info
    ;;

  --docker)
    # Démarrage via Docker Compose
    echo ""
    echo "🐳 Démarrage du service FastAPI via Docker..."
    docker compose up fastapi -d
    echo ""
    echo "⏳ Attente du healthcheck..."
    for i in $(seq 1 20); do
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ API disponible : http://localhost:8000"
            echo "   Docs Swagger   : http://localhost:8000/docs"
            echo "   Métriques      : http://localhost:8000/metrics"
            break
        fi
        echo "   Tentative $i/20..."
        sleep 3
    done
    ;;

  --k8s)
    # Déploiement Kubernetes
    echo ""
    echo "☸️  Déploiement Kubernetes..."
    kubectl create namespace mlops --dry-run=client -o yaml | kubectl apply -f -
    kubectl apply -f infrastructure/kubernetes/fastapi/
    kubectl apply -f infrastructure/kubernetes/monitoring/

    echo ""
    echo "⏳ Attente du rollout..."
    kubectl rollout status deployment/movielens-api -n mlops --timeout=120s
    echo ""
    kubectl get pods -n mlops -l app=movielens-api
    ;;

  *)
    echo "Usage : $0 [--local | --fastapi | --docker | --k8s]"
    exit 1
    ;;
esac
