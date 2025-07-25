# 📊 Interface Streamlit – Stack Overflow Tag Predictor  
*Par Agnès Regaud – Projet 5 – OpenClassrooms – Parcours Ingénieur Machine Learning*

Cette application Streamlit permet d'interagir avec l'API Flask et de visualiser les prédictions de tags pour des questions Stack Overflow.

🎯 Objectif

- Tester les prédictions de tags sur des exemples réels ou saisis manuellement
- Visualiser les tags proposés par les modèles supervisé (CatBoost) et non supervisé (NMF)
- Afficher les métriques d’évaluation clés (couverture, précision@3)

📁 Contenu du répertoire

- main.py : script principal Streamlit
- config.json : fichier contenant l’URL de l’API Flask (versionné, utilisé par Streamlit Cloud)
- environment.yml : dépendances conda
- Dockerfile : configuration de l’image Docker
- utils.py : fonctions utilitaires (normalisation, métriques…)
- update_config.py : script Python pour mettre à jour automatiquement l’URL de l’API dans config.json après chaque déploiement AWS
- test_data.csv : exemples de questions Stack Overflow

▶️ Lancement local

conda env create -f environment.yml
conda activate streamlit_env
streamlit run main.py

Crée un fichier .env (non versionné) à la racine si tu veux personnaliser l’URL de l’API pour tes tests locaux :

API_URL=http://localhost:5001/predict
DOCKERIZED=0

▶️ Lancement avec Docker:

docker build -t app_streamlit .
docker run -p 8501:8501 --env-file .env app_streamlit

.env doit être présent en local, mais non versionné.

ou :

docker-compose up --build  

▶️ Déploiement sur Streamlit Cloud

Le fichier config.json versionné contient l’URL publique de l’API Flask (exposée sur AWS ECS Fargate).

Après chaque redéploiement de l’API Flask, lance le script :

export AWS_PROFILE=local-docker-user
python3 update_config.py

→ Cela mettra à jour config.json, commitera et poussera la bonne URL.

Streamlit Cloud rebuildera automatiquement avec la nouvelle configuration.

💡 Fonctionnalités

- Saisie ou sélection d’une question Stack Overflow
- Affichage des tags prédits par CatBoost (supervisé) et NMF (non supervisé)
- Visualisation des scores associés
- Affichage des tags d’origine et filtrés
- Présentation claire des métriques : couverture, précision@3

Note :

- .env est utile seulement pour l’usage local/Docker.
- Pour le Cloud, tout passe par config.json (versionné, MAJ automatique via script).
