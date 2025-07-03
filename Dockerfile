FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml

COPY . .

EXPOSE 8501

# Commande pour lancer streamlit dans l'environnement conda 'streamlit_env'
CMD ["conda", "run", "-n", "streamlit_env", "streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
