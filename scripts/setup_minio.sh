#!/bin/bash
# scripts/setup_minio.sh
# Vérifie et initialise les buckets MinIO manuellement
# Usage : bash scripts/setup_minio.sh

set -e

MINIO_ENDPOINT=${MINIO_ENDPOINT:-http://localhost:9000}
MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY:-minioadmin}
MINIO_SECRET_KEY=${MINIO_SECRET_KEY:-minioadmin}

echo "🪣  Configuration MinIO — $MINIO_ENDPOINT"

# Vérifier que mc (MinIO Client) est disponible
if ! command -v mc &> /dev/null; then
    echo "❌ mc non installé. Installer avec :"
    echo "   curl https://dl.min.io/client/mc/release/linux-amd64/mc -o /usr/local/bin/mc"
    echo "   chmod +x /usr/local/bin/mc"
    exit 1
fi

# Configurer l'alias
mc alias set local "$MINIO_ENDPOINT" "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY"

# Créer les buckets
for bucket in mlflow-artifacts raw-data processed-data; do
    mc mb --ignore-existing "local/$bucket"
    echo "✅ Bucket créé : $bucket"
done

echo ""
echo "🎉 MinIO configuré — console : http://localhost:9001"
