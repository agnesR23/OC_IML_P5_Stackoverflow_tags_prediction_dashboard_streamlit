FROM continuumio/miniconda3

WORKDIR /app

# Copier le fichier environment.yml dans le conteneur
COPY environment.yml .

# Créer l'environnement conda avec le nom 'streamlit_env'
RUN conda env create -f environment.yml

# Copier le reste des fichiers de l'app dans le conteneur
COPY . .

# Utiliser le shell conda pour activer l'environnement puis lancer streamlit
SHELL ["conda", "run", "-n", "streamlit_env", "/bin/bash", "-c"]

# Exposer le port par défaut de streamlit
EXPOSE 8501

# Commande pour lancer l'application streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
