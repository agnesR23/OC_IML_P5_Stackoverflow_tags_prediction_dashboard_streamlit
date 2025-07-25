import re
import pandas as pd
import numpy as np
import json

from sklearn.metrics import f1_score, hamming_loss, jaccard_score

def load_config(config_path="config/config.json"):
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

# =============================================================
# Fonction de normalisation du texte
# =============================================================

languages_frameworks = [
    # Langages
    "Python", "python3", "python 3",
    "Java", "Java SE", "Java EE",
    "JavaScript", "JS",
    "TypeScript", "TS",
    "C", "C99",
    "C++", "C plus plus", "cpp",
    "C#", "C sharp", "csharp",
    "Go", "Golang",
    "Ruby",
    "PHP",
    "Swift",
    "Kotlin",
    "Rust",
    "Scala",
    "Perl",
    "Objective-C", "Objective C",
    "R",
    "MATLAB",
    "SQL", "T-SQL", "PL/SQL", "MySQL", "PostgreSQL", "SQLite", "SQL Server",

    # Frameworks & bibliothèques
    "React", "React.js", "ReactJS",
    "Angular", "AngularJS",
    "Vue", "Vue.js", "VueJS",
    "Next.js", "NextJS",
    "Nuxt.js", "NuxtJS",
    "Node.js", "NodeJS",
    "Express", "Express.js",
    "Django",
    "Flask",
    "Spring", "Spring Boot",
    "Ruby on Rails", "Rails",
    "Laravel",
    "Symfony",
    "Bootstrap",
    "jQuery",
    "ASP.NET", "ASP.NET Core", "ASP",
    "TensorFlow", "Tensorflow",
    "PyTorch", "Pytorch",
    "Scikit-learn", "scikit learn",
    "pandas",
    "NumPy", "numpy",
    "OpenCV",
    "Keras",
    "XGBoost", "xgboost",
    "LightGBM", "lightgbm",
    "Transformers",
    "FastAPI",
    "BeautifulSoup", "Beautiful Soup",
    "Selenium",
    "Hadoop",
    "Spark", "PySpark",
    "Airflow",
    "Docker",
    "Kubernetes", "k8s",
    "Terraform",
    "Ansible",
    "Jenkins",
    "Git", "GitHub", "GitLab",
    "CI/CD",
]



def normalize_text(text, languages_frameworks):
    """Normalise le texte tout en préservant certains langages et frameworks.

    Args:
        text (str): Le texte à normaliser.
        languages_frameworks (list): Liste de langages et frameworks à préserver.

    Returns:
        str: Le texte normalisé avec les langages et frameworks préservés.
    """
    preserved_words = {}

    for word in languages_frameworks:
        # Regex pour correspondance exacte du mot
        pattern = r'\b' + re.escape(word) + r'\b'
        placeholder = f"__preserve_{len(preserved_words)}__"
        text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)
        preserved_words[placeholder] = word  # on garde la version originale

    # Nettoyage standard sans toucher aux placeholders
    text = re.sub(r'[^a-zA-Z0-9\s_.]+', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()

    # Restauration des mots préservés (en minuscule si tu veux)
    for placeholder, original_word in preserved_words.items():
        text = text.replace(placeholder, original_word.lower())  # .lower() optionnel

    return text

# =============================================================
# Fonctions d’évaluation multilabel classiques (binarisées)
# Utilisées dans l’approche supervisée (e.g. CatBoost, LogisticRegression)
# =============================================================



def compute_metrics(y_true, y_pred_probs, thresholds=[0.5], k=3, model_name=None, approach=None):
    results = []
    for threshold in thresholds:
        y_pred = (y_pred_probs >= threshold).astype(int)
        f1_micro = f1_score(y_true, y_pred, average="micro")
        f1_macro = f1_score(y_true, y_pred, average="macro")
        h_loss = hamming_loss(y_true, y_pred)
        cov = coverage_score(y_true, y_pred)
        prec_k = precision_at_k(y_true, y_pred_probs, k=k)
        jac = jaccard_score_multilabel(y_true, y_pred)

        metrics_dict ={
            "ModelName": model_name,
            "Threshold": threshold,
            "F1_micro": f1_micro,
            "F1_macro": f1_macro,
            "HammingLoss": h_loss,
            "Coverage": cov,
            f"Precision@{k}": prec_k,
            "Jaccard Score": jac
            }
        if approach is not None:
            metrics_dict["Approach"] = approach
        results.append(metrics_dict)
    return pd.DataFrame(results)

def coverage_score(y_true, y_pred):
    """Taux d’exemples où au moins un tag vrai est prédit."""
    correct = 0
    for i in range(len(y_true)):
        true_labels = set(np.where(y_true[i])[0])
        pred_labels = set(np.where(y_pred[i])[0])
        if true_labels & pred_labels:
            correct += 1
    return correct / len(y_true)

def precision_at_k(y_true, y_pred, k=3):
    """Precision@k : proportion de vrais tags parmi les k meilleurs prédits."""
    precisions = []
    for i in range(len(y_true)):
        pred_top_k = np.argsort(y_pred[i])[::-1][:k]
        true_indices = np.where(y_true[i])[0]
        intersect = len(set(pred_top_k) & set(true_indices))
        precisions.append(intersect / k)
    return np.mean(precisions)


def jaccard_score_multilabel(y_true, y_pred):
    """Calcule le Jaccard Score pour la classification multilabel."""
    return jaccard_score(y_true, y_pred, average='macro')

# =============================================================
# Fonctions d’évaluation avec listes de tags (non binarisées)
# Utilisées dans l’approche non supervisée (e.g. NMF, LDA)
# =============================================================


def coverage_score_true_pred(y_true, y_pred):
    """Taux d’exemples où au moins un tag vrai est prédit."""
    correct = 0
    for true_tags, pred_tags in zip(y_true, y_pred):
        if set(true_tags) & set(pred_tags):
            correct += 1
    return correct / len(y_true)

def precision_at_k_true_pred(y_true, y_pred, k=3):
    """Precision@k : proportion de vrais tags parmi les k premiers prédits."""
    precisions = []
    for true_tags, pred_tags in zip(y_true, y_pred):
        pred_top_k = pred_tags[:k]
        intersect = len(set(pred_top_k) & set(true_tags))
        precisions.append(intersect / k)
    return sum(precisions) / len(precisions)

def f1_at_k(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def compute_row_scores(true_tags, pred_tags, k=3):
    pred_top_k = pred_tags[:k]
    precision = len(set(pred_top_k) & set(true_tags)) / k
    recall = len(set(pred_top_k) & set(true_tags)) / len(true_tags) if true_tags else 0.0
    f1 = f1_at_k(precision, recall)
    return precision, recall, f1