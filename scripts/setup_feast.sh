#!/bin/bash
# scripts/setup_feast.sh
# Initialisation du Feature Store Feast pour MovieLens
# Usage : bash scripts/setup_feast.sh

set -e
echo "🌾 Configuration Feast — MovieLens Recommender"
echo "================================================"

# 1. Vérifier Python
python3 --version

# 2. Générer les fichiers de features offline (si absents)
echo ""
echo "📂 Génération des features offline..."
if [ ! -f "data/feast/offline/user_features.csv" ]; then
    python3 -c "
import sys; sys.path.insert(0, '.')
from training.data_loader import load_movies, load_users, load_ratings
import pandas as pd, numpy as np, os
from datetime import datetime, timedelta

os.makedirs('data/feast/offline', exist_ok=True)
os.makedirs('data/feast/online', exist_ok=True)

movies  = load_movies()
users   = load_users()
ratings = load_ratings()

user_stats = ratings.groupby('UserID').agg(
    user_avg_rating=('Rating','mean'),
    user_n_ratings=('Rating','count'),
    user_std_rating=('Rating','std')
).reset_index().fillna(0)

user_feat = users[['UserID','Gender_enc','Age','Occupation']].merge(user_stats, on='UserID')
user_feat.columns = [c.lower() for c in user_feat.columns]
user_feat = user_feat.rename(columns={'userid':'user_id','gender_enc':'gender_enc'})
user_feat['event_timestamp'] = datetime(2024,1,1)
user_feat['created'] = datetime(2024,1,1)
user_feat.to_csv('data/feast/offline/user_features.csv', index=False)
print('  ✅ user_features.csv')

genre_cols = [c for c in movies.columns if c not in ['MovieID','Title','Genres','Year']]
movie_stats = ratings.groupby('MovieID').agg(
    movie_avg_rating=('Rating','mean'),
    movie_n_ratings=('Rating','count')
).reset_index()
movie_feat = pd.concat([movies[['MovieID','Year']], movies[genre_cols]], axis=1)
movie_feat = movie_feat.merge(movie_stats, on='MovieID')
movie_feat.columns = [c.lower().replace(\"'\",\"\").replace('-','_') for c in movie_feat.columns]
movie_feat = movie_feat.rename(columns={'movieid':'movie_id'})
movie_feat['event_timestamp'] = datetime(2024,1,1)
movie_feat['created'] = datetime(2024,1,1)
movie_feat.to_csv('data/feast/offline/movie_features.csv', index=False)
print('  ✅ movie_features.csv')
"
else
    echo "  ✅ Fichiers offline déjà présents"
fi

# 3. Tenter feast apply (si Feast installé)
echo ""
echo "🔧 Tentative feast apply..."
if command -v feast &> /dev/null; then
    cd feast/
    feast apply
    cd ..
    echo "  ✅ feast apply réussi"
else
    echo "  ℹ️  Feast non installé — utilisation du simulateur local"
    echo "     Installer avec : pip install feast redis"
fi

# 4. Matérialisation
echo ""
echo "🔄 Matérialisation offline → online..."
python3 feast/feast_materialize.py --validate

echo ""
echo "🎉 Feature Store configuré et validé !"
echo "   Offline : data/feast/offline/"
echo "   Online  : data/feast/online/ (snapshots JSON)"
echo ""
echo "Prochaine étape : python training/train_with_feast.py"
