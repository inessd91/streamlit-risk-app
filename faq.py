# faq.py
import numpy as np
from dotenv import load_dotenv
import os

from langchain_openai import OpenAIEmbeddings

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"

embeddings_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ------------------------------
# FAQ locale
# ------------------------------
FAQ_LOCAL = {
    "prime": {
        "keywords": ["prime", "tarif", "coût"],
        "text": (
            "La prime finale correspond au montant à payer pour être couvert par l’assurance. "
            "Elle est calculée à partir de la prime théorique issue du modèle de risque, "
            "ajustée selon la classe de risque du client et soumise à un plancher tarifaire."
        ),
        "embedding": None
    },
    "risque": {
        "keywords": ["risque", "défaut", "score"],
        "text": (
            "Le risque représente la propension du client à faire défaut sur ses engagements financiers. "
            "Il est estimé à partir des données financières, comportementales et déclaratives."
        ),
        "embedding": None
    },
    "decision": {
        "keywords": ["décision", "acceptation", "refus"],
        "text": (
            "La décision proposée dépend de la classe de risque du client et peut être standard, "
            "ajustée ou nécessiter une analyse approfondie. "
            "La décision finale est validée par un analyste humain."
        ),
        "embedding": None
    }
}

# ------------------------------
# Calcul des embeddings au démarrage
# ------------------------------
def init_faq_embeddings():
    for theme, data in FAQ_LOCAL.items():
        if data["embedding"] is None:
            vec = embeddings_model.embed_query(data["text"])
            data["embedding"] = np.array(vec)

init_faq_embeddings()

# ------------------------------
# Matching FAQ basé sur embeddings
# ------------------------------
def match_faq(question: str, max_themes: int = 2):
    """Retourne les thèmes FAQ les plus pertinents pour une question"""

    question_vec = np.array(
        embeddings_model.embed_query(question)
    )

    scores = {}
    for theme, data in FAQ_LOCAL.items():
        faq_vec = data["embedding"]
        cos_sim = np.dot(question_vec, faq_vec) / (
            np.linalg.norm(question_vec) * np.linalg.norm(faq_vec)
        )
        scores[theme] = cos_sim

    matched = [
        t for t, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if s > 0.6
    ]

    # Fallback mots-clés
    if not matched:
        q_lower = question.lower()
        for theme, data in FAQ_LOCAL.items():
            for kw in data["keywords"]:
                if kw in q_lower:
                    matched.append(theme)
                    break

    return matched[:max_themes]

# ------------------------------
# Génération de la réponse FAQ
# ------------------------------
def get_faq_response(theme: str, shap_pos=None, shap_neg=None):
    data = FAQ_LOCAL.get(theme)
    if not data:
        return "Information non disponible."

    shap_pos = shap_pos or []
    shap_neg = shap_neg or []

    text = data["text"]

    if shap_pos or shap_neg:
        text += "\n\n**Facteurs influents pour ce client :**"
        if shap_pos:
            text += "\n- 🔺 Augmentant le risque : " + ", ".join(shap_pos)
        if shap_neg:
            text += "\n- 🔻 Réduisant le risque : " + ", ".join(shap_neg)

    text += "\n\n*La décision finale relève d’un analyste humain.*"
    return text
