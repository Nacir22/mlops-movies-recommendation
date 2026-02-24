# 🎬 MLOps MovieLens — Période 1 : Fondations & Infrastructure locale

## Objectif

Avoir une stack locale fonctionnelle avec **PostgreSQL, MinIO, MLflow et Jupyter**,
connectés ensemble et prêts pour les expérimentations de la Période 2.

---

## Structure des fichiers créés

```
mlops-movies/
│
├── .env.example                          ← Variables d'environnement (template)
├── .gitignore
├── pyproject.toml                        ← Dépendances Python
├── docker-compose.yaml                   ← Stack complète Période 1
│
├── infrastructure/
│   └── docker/
│       ├── mlflow/Dockerfile             ← Image MLflow + boto3 + psycopg2
│       └── jupyter/Dockerfile            ← Image Jupyter + scikit-learn + mlflow
│
├── scripts/
│   ├── init_postgres.sql                 ← Init BDD au démarrage
│   ├── setup_minio.sh                    ← Création des buckets MinIO
│   └── setup_mlflow.sh                   ← Vérification + création expérience
│
└── data/
    ├── raw/
    │   ├── movies.dat                    ← MovieID::Title::Genres
    │   ├── users.dat                     ← UserID::Gender::Age::Occupation::Zip
    │   └── ratings.dat                   ← UserID::MovieID::Rating::Timestamp
    └── schemas/
        └── data_schema.py                ← Validation des fichiers .dat
```

---

## Ordre de démarrage

### 1. Copier et remplir le fichier d'environnement

```bash
cp .env.example .env
# Éditer .env si besoin (les valeurs par défaut fonctionnent en local)
```

### 2. Démarrer l'infra de base

```bash
# PostgreSQL + MinIO + Redis
docker compose up postgres minio -d

# Attendre que les healthchecks passent (~30s)
docker compose ps
```

### 3. Créer les buckets MinIO

```bash
# Automatique via minio-init
docker compose up minio-init

# OU manuellement
bash scripts/setup_minio.sh
```

### 4. Démarrer MLflow

```bash
docker compose up mlflow -d

# Vérifier
curl http://localhost:5000/health
# → {"status": "OK"}
```

### 5. Créer l'expérience MLflow

```bash
bash scripts/setup_mlflow.sh
```

### 6. Démarrer Jupyter

```bash
docker compose up jupyter -d

# Récupérer le token
docker logs mlops_jupyter 2>&1 | grep "token="
```

---

## Vérifications

| Service    | URL                       | Credentials          |
|------------|---------------------------|----------------------|
| MLflow UI  | http://localhost:5000      | —                    |
| MinIO UI   | http://localhost:9001      | minioadmin/minioadmin|
| Jupyter    | http://localhost:8888      | token dans les logs  |
| PostgreSQL | localhost:5432             | mlflow/mlflow_password|

### Test de connexion MLflow depuis Jupyter

```python
import mlflow
import os

mlflow.set_tracking_uri("http://mlflow:5000")   # depuis Jupyter (réseau Docker)
# mlflow.set_tracking_uri("http://localhost:5000") # depuis local

mlflow.set_experiment("movielens-recommender")

with mlflow.start_run(run_name="test_periode_1"):
    mlflow.log_param("test", "periode_1")
    mlflow.log_metric("dummy_metric", 0.99)
    print("✅ Run loggé dans MLflow !")
```

### Test de stockage MinIO depuis Python

```python
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
)

# Lister les buckets
buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
print("Buckets :", buckets)
# → ['mlflow-artifacts', 'raw-data', 'processed-data']
```

---

## Valider les données MovieLens

```bash
# Vérifier les fichiers .dat
python data/schemas/data_schema.py
```

Sortie attendue :
```
✅ movies.dat  : 102 valides, 0 invalides
✅ users.dat   : 500 valides, 0 invalides
✅ ratings.dat : ~31800 valides, 0 invalides
```

> **Note :** Les fichiers `.dat` fournis sont synthétiques et reproduisent
> fidèlement le format MovieLens 1M. Remplacez-les par les vrais fichiers
> téléchargés depuis https://grouplens.org/datasets/movielens/1m/
> sans modifier aucun autre fichier du projet.

---

## Arrêter la stack

```bash
docker compose down          # Arrêter sans supprimer les volumes
docker compose down -v       # Arrêter ET supprimer les données
```

---

## Prochaine étape → Période 2

Exploration des données et premières expériences MLflow dans Jupyter.
