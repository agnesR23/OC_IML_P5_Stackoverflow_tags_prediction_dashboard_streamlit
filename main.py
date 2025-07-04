import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import ast
from utils import (
    coverage_score_true_pred,
    precision_at_k_true_pred,
    coverage_score,
    precision_at_k,
)

# --------- Config ---------
st.set_page_config(
    page_title="Prédiction de tags Stack Overflow",
    page_icon=":mag_right:",
    layout="wide",
    initial_sidebar_state="expanded"
)
# URL de l'API Flask:
if os.getenv("STREAMLIT_CLOUD", "0") == "1":
    API_URL = "https://ton-api-flask-sur-aws-ou-autre/predict"
elif os.getenv("DOCKERIZED", "0") == "1":
    API_URL = "http://flask_app:5001/predict"
else:
    API_URL = "http://localhost:5001/predict"


DATA_PATH = "test_data.csv"
DEFAULT_THRESHOLD = 0.5
NUM_EXAMPLES = 5  # pour afficher 5 exemples du test_data

# --------- Chargement des données ---------
@st.cache_data
def load_test_data():
    return pd.read_csv(DATA_PATH)

df_test = load_test_data()

# --------- Fonction pour appeler l'API ---------
def call_api_predict(title, body, threshold, model_type, true_tags=None):
    payload = {
        "title": title,
        "body": body,
        "threshold": threshold,
        "model_type": model_type
    }
    if true_tags is not None:
        payload["true_tags"] = true_tags
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("error", "Erreur inconnue")}
    except Exception as e:
        return {"error": str(e)}

# --------- Titre et description ---------
st.title("🔍 Prédiction automatique de tags Stack Overflow")
st.markdown("""
Ce dashboard, réalisé par **Agnès Regaud**, est le **5ᵉ projet** de la filière **Ingénieur Machine Learning** d'OpenClassrooms.

Il permet de prédire automatiquement les tags les plus pertinents pour une question Stack Overflow, à partir de son **titre** et de sa **description**.

---

## 🔍 Objectif
Aider les utilisateurs à mieux taguer leurs questions techniques sur Stack Overflow, en s’appuyant sur deux approches :
- ✅ **Modèle supervisé (CatBoost)** : entraîné sur des données labellisées pour apprendre les relations entre texte et tags.
- 🧪 **Modèle non supervisé (NMF)** : extrait des thématiques dominantes pour suggérer des tags probables, même sans apprentissage supervisé.

---
            
## 📊 Définitions des métriques d’évaluation :

- **Couverture** : proportion de questions où au moins un tag correct est prédit.  
- **Précision@3** : proportion des 3 premiers tags prédits qui sont corrects.

---
            
## ✨ Fonctionnalités
Ce dashboard propose deux modes d’utilisation :

### 📌 Partie 1 – Exemples prédéfinis
Testez la prédiction de tags sur des exemples réels de questions Stack Overflow. Cela permet de comparer les performances entre les deux modèles.

### ✍️ Partie 2 – Saisie manuelle
Soumettez votre propre question (titre + description) pour obtenir une prédiction personnalisée des tags.

---

""")


st.subheader("📌 Partie 1 – Exemples issus du jeu de test : choisissez avec la barre de gauche")

# --------- Affichage des exemples ---------

def render_tags_as_badges(tag_list):
    badges = " ".join([
        f"<span style='background-color:#e8ddff; border-radius:12px; padding:6px 12px; margin:4px; display:inline-block; color:#3c0066; font-weight:500;'>{tag}</span>"
        for tag in tag_list
    ])
    return f"<div style='font-size:18px; line-height:2.2;'>{badges}</div>"

def render_tags_simple(label, tags):
    if isinstance(tags, str):
        tags = tags.strip("[]").replace("'", "").split(", ")
    badges = " ".join([
        f"<span style='color:#2f4f2f; font-weight:600; padding:4px 10px; border-radius:12px; background-color:#b6d7a8; margin:4px 6px 4px 0; display:inline-block;'>{tag}</span>"
        for tag in tags if tag
    ])
    return f"""
    <div style='font-size:16px; font-weight:bold; color:#000; margin-bottom:4px;'>{label} :</div>
    <div>{badges}</div>
    """


def render_text_area_custom(text, height=150):
    html = f"""
    <div style="
        white-space: pre-wrap;
        background-color: #f5f7fa;  /* fond très pâle, légèrement bleu-gris */
        color: #2c3e50;  /* texte foncé mais doux, bleu foncé */
        padding: 10px;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        height: {height}px;
        overflow-y: auto;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 15px;
        line-height: 1.4;
    ">{text}</div>
    """
    return html


st.sidebar.markdown("""
    <h2>📚 Exemples prédéfinis</h2>
    <p style="font-size:14px; line-height:1.4;">
        Choisissez un exemple dans la liste pour afficher ses détails, son titre, sa description, et les tags prédits par les modèles.
    </p>
    """, unsafe_allow_html=True)
ex_index = st.sidebar.slider("Choisissez un exemple", 0, len(df_test)-1, 0)


i = ex_index

st.markdown(f"### Exemple {i+1}")

    
st.markdown("**Titre :**")
st.markdown(render_text_area_custom(df_test.loc[i, "Title"], height=40), unsafe_allow_html=True)
st.markdown("**Question :**")
st.markdown(render_text_area_custom(df_test.loc[i, "Body"], height=110), unsafe_allow_html=True)


st.markdown(render_tags_simple("🗂️ Tous les tags d'origine (TagsList)", df_test.loc[i, 'TagsList']), unsafe_allow_html=True)
st.markdown(render_tags_simple("✅ Tags retenus (FilteredTags)", df_test.loc[i, 'FilteredTags']), unsafe_allow_html=True)


true_tags_raw = df_test.loc[i, "FilteredTags"]
try:
    true_tags = ast.literal_eval(true_tags_raw) if isinstance(true_tags_raw, str) else true_tags_raw
except Exception:
    true_tags = []


# Prédiction CatBoost
res_catboost = call_api_predict(
    title=df_test.loc[i, "Title"],
    body=df_test.loc[i, "Body"],
    threshold=DEFAULT_THRESHOLD,
    model_type="catboost",
    true_tags=true_tags
)
if "error" not in res_catboost:
    with st.expander("📎 Afficher les prédictions CatBoost"):
        st.markdown("<div style='font-size:18px; font-weight:bold; color:#000; margin-bottom:8px;'>Tags prédits (CatBoost) et métriques</div>", unsafe_allow_html=True)
        tags_cat = res_catboost["predicted_tags"]
        st.markdown(render_tags_as_badges(tags_cat), unsafe_allow_html=True)
        coverage_cat = res_catboost.get("coverage", None)
        precision_cat = res_catboost.get("precision_at_3", None)
        if coverage_cat is not None and precision_cat is not None:
            st.markdown(f"📊 Couverture : {coverage_cat:.2f} | Precision@3 : {precision_cat:.2f}")


else:
    st.error(f"Erreur CatBoost: {res_catboost['error']}")

# Prédiction NMF
res_nmf = call_api_predict(
    title=df_test.loc[i, "Title"],
    body=df_test.loc[i, "Body"],
    threshold=DEFAULT_THRESHOLD,
    model_type="nmf"
)
if "error" not in res_nmf:
    with st.expander("📎 Afficher les prédictions"):
        st.markdown("<div style='font-size:18px; font-weight:bold; color:#000; margin-bottom:8px;'>Tags prédits (NMF) et métriques</div>", unsafe_allow_html=True)
        tags_nmf = res_nmf["predicted_tags"]
        st.markdown(render_tags_as_badges(tags_nmf), unsafe_allow_html=True)
        coverage_nmf = coverage_score_true_pred([df_test.loc[i, "FilteredTags"]], [tags_nmf])
        precision_nmf = precision_at_k_true_pred([df_test.loc[i, "FilteredTags"]], [tags_nmf], k=3)
        st.markdown(f"📊 Couverture : {coverage_nmf:.2f} | Precision@3 : {precision_nmf:.2f}")
else:
    st.error(f"Erreur NMF: {res_nmf['error']}")

st.divider()

# --------- Entrée manuelle ---------


st.subheader("✍️ Partie 2 – Tester votre propre question")

with st.form("manual_input"):
    title = st.text_input("Titre de la question")
    body = st.text_area("Contenu de la question")
    threshold = st.slider("Seuil de prédiction", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01)
    submitted = st.form_submit_button("Prédire les tags")

    if submitted:
        if not title or not body:
            st.warning("Veuillez remplir le titre et le corps.")
        else:
            # Prédiction CatBoost
            res_catboost = call_api_predict(title=title, body=body, threshold=threshold, model_type="catboost")
            if "error" not in res_catboost:
                scores_cat = res_catboost.get("scores", {})
                # Filtrage et tri des tags par score décroissant
                sorted_scores_cat = dict(sorted(scores_cat.items(), key=lambda x: x[1], reverse=True))
                filtered_tags_cat = [tag for tag, score in sorted_scores_cat.items() if score >= threshold]
                if filtered_tags_cat:
                    st.markdown(f"✅ Tags prédits (CatBoost, seuil ≥ {threshold}) :", unsafe_allow_html=True)
                    st.markdown(render_tags_as_badges(filtered_tags_cat), unsafe_allow_html=True)
                else:
                    st.markdown(
                                f"<div style='background-color:#f9d342; color:#5a3e00; padding:10px; border-radius:6px; font-weight:bold;'>"
                                f"⚠️ Aucun tag prédit (CatBoost) avec un seuil ≥ {threshold}."
                                "</div>", unsafe_allow_html=True
                                )                  
                st.markdown("**📊 Scores associés (CatBoost)**")
                st.json(sorted_scores_cat)
            else:
                st.error(f"Erreur CatBoost : {res_catboost['error']}")

            # Prédiction NMF
            res_nmf = call_api_predict(title=title, body=body, threshold=threshold, model_type="nmf")
            if "error" not in res_nmf:
                scores_nmf = res_nmf.get("scores", {})
                sorted_scores_nmf = dict(sorted(scores_nmf.items(), key=lambda x: x[1], reverse=True))
                filtered_tags_nmf = [tag for tag, score in sorted_scores_nmf.items() if score >= threshold]
                if filtered_tags_nmf:
                    st.markdown(f"✅ Tags prédits (NMF, seuil ≥ {threshold}) :", unsafe_allow_html=True)
                    st.markdown(render_tags_as_badges(filtered_tags_nmf), unsafe_allow_html=True)
                else:
                    st.markdown(
                            "<div style='background-color:#f9d342; color:#5a3e00; padding:10px; border-radius:6px; font-weight:bold;'>"
                            "⚠️ Aucun tag prédit (NMF) avec un seuil ≥ {threshold}."
                            "</div>", unsafe_allow_html=True
                        )                
                if sorted_scores_nmf:
                    st.markdown("**📊 Scores associés (NMF)**")
                    st.json(sorted_scores_nmf)
            else:
                st.error(f"Erreur NMF : {res_nmf['error']}")
