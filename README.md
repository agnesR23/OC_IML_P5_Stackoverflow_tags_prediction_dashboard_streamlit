# 📊 Interface Streamlit – Stack Overflow Tag Predictor  
*Projet 5 – OpenClassrooms – Parcours Ingénieur Machine Learning*

Cette application Streamlit permet d'interagir avec l'API Flask et de visualiser les prédictions de tags pour des questions Stack Overflow.

## 🎯 Objectif

Offrir une interface simple à l’utilisateur pour tester les prédictions, visualiser les tags proposés, et éventuellement soumettre ses propres textes.

## 📁 Contenu du répertoire

- `main.py` : script principal de l’app Streamlit
- `.env` : fichier contenant l’URL de l’API Flask (modifiable)
- `environment.yml` : dépendances conda de l’interface
- `Dockerfile` : image Docker de l’interface
- `README.md` : ce fichier

## ▶️ Lancement local

```bash
conda env create -f environment.yml
conda activate streamlit_env
streamlit run main.py

#Vérifie que le fichier .env contient l’URL correcte de l’API Flask:
API_URL=http://localhost:5001

🐳 Docker
docker build -t app_streamlit .
docker run -p 8501:8501 --env-file .env app_streamlit

💡 Fonctionnalités
Envoi d’une question (titre + corps)

Appel à l’API Flask

Affichage des tags prédits

Visualisation des scores ou probas si dispo