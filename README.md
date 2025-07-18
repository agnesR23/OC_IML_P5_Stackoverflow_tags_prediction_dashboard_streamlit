# 📊 Interface Streamlit – Stack Overflow Tag Predictor  
*Par Agnès Regaud – Projet 5 – OpenClassrooms – Parcours Ingénieur Machine Learning*

Cette application Streamlit permet d'interagir avec l'API Flask et de visualiser les prédictions de tags pour des questions Stack Overflow.

## 🎯 Objectif

Fournir une interface simple et conviviale pour :

- Tester les prédictions de tags sur des exemples réels ou saisis manuellement

- Visualiser clairement les tags proposés par les modèles supervisé (CatBoost) et non supervisé (NMF)

- Afficher les métriques d’évaluation clés (couverture, précision@3)

## 📁 Contenu du répertoire

- `main.py` : script principal de l’app Streamlit
- `.env` : fichier contenant l’URL de l’API Flask : non versionné 
- `environment.yml` : dépendances conda de l’interface
- `Dockerfile` : configuration pour créer l’image Docker de l’application
- `utils.py` : fonctions utilitaires partagées (normalisation, métriques, etc.)
- `README.md` : ce fichier

## ▶️ Lancement local

```bash
conda env create -f environment.yml
conda activate streamlit_env
streamlit run main.py

Important : Avant de lancer l’application, créez un fichier `.env` à la racine du répertoire avec le contenu suivant à adapter si besoin :
API dockérisée en local :
API_URL=http://flask_app:5001/predict
DOCKERIZED=1
ou en local simple :
API_URL=http://localhost:5001/predict  
DOCKERIZED=0

## ▶️ Lancement avec Docker :
docker build -t app_streamlit
docker run -p 8501:8501 --env-file .env app_streamlit
Et dans .env : 
API_URL=http://flask_app:5001/predict
DOCKERIZED=1

ou avec Docker-compose :
docker-compose up --build  

## ▶️ Cas déploiement Streamlit Community par exemple (API Flask sur cloud) :
Dans .env :
API_URL=https://votre-api-flask-sur-le-cloud/predict  
STREAMLIT_CLOUD=1  


💡 Fonctionnalités
- Envoi d’une question (titre + description) à l’API Flask

- Affichage des tags prédits par deux modèles : CatBoost (supervisé) et NMF (non supervisé)

- Visualisation des scores associés aux tags lorsque disponibles

- Comparaison des tags d’origine (réels) et des tags retenus (filtrés)

- Présentation claire des métriques de qualité des prédictions (couverture, précision@3)