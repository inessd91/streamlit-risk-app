# faq.py
import numpy as np
from dotenv import load_dotenv
import os

from langchain_openai import OpenAIEmbeddings

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"

# Client embeddings créé à la demande uniquement si OPENAI_API_KEY est défini
_embeddings_model = None
_faq_embeddings_initialized = False


def _get_embeddings():
    """Retourne le client OpenAI Embeddings ou None si pas de clé API."""
    global _embeddings_model
    if _embeddings_model is not None:
        return _embeddings_model
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    _embeddings_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
    )
    return _embeddings_model


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
        "embedding": None,
    },
    "risque": {
        "keywords": ["risque", "défaut", "score"],
        "text": (
            "Le risque représente la propension du client à faire défaut sur ses engagements financiers. "
            "Il est estimé à partir des données financières, comportementales et déclaratives."
        ),
        "embedding": None,
    },
    "decision": {
        "keywords": ["décision", "acceptation", "refus"],
        "text": (
            "La décision proposée dépend de la classe de risque du client et peut être standard, "
            "ajustée ou nécessiter une analyse approfondie. "
            "La décision finale est validée par un analyste humain."
        ),
        "embedding": None,
    },
}


def _init_faq_embeddings():
    """Précalcule les embeddings FAQ si une clé OpenAI est disponible."""
    global _faq_embeddings_initialized
    if _faq_embeddings_initialized:
        return
    emb = _get_embeddings()
    if emb is None:
        _faq_embeddings_initialized = True
        return
    for _, data in FAQ_LOCAL.items():
        if data["embedding"] is None:
            vec = emb.embed_query(data["text"])
            data["embedding"] = np.array(vec)
    _faq_embeddings_initialized = True


def _match_faq_keywords(question: str, max_themes: int):
    """Matching par mots-clés uniquement (sans API)."""
    matched = []
    q_lower = question.lower()
    for theme, data in FAQ_LOCAL.items():
        for kw in data["keywords"]:
            if kw in q_lower:
                matched.append(theme)
                break
    return matched[:max_themes]


# ------------------------------
# Matching FAQ : embeddings si clé présente, sinon mots-clés
# ------------------------------
def match_faq(question: str, max_themes: int = 2):
    """Retourne les thèmes FAQ les plus pertinents pour une question."""

    emb = _get_embeddings()
    if emb is None:
        return _match_faq_keywords(question, max_themes)

    _init_faq_embeddings()

    # Si toujours pas d’embeddings thématiques (ex. échec silencieux), fallback mots-clés
    if any(data["embedding"] is None for data in FAQ_LOCAL.values()):
        return _match_faq_keywords(question, max_themes)

    question_vec = np.array(emb.embed_query(question))
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

    if not matched:
        matched = _match_faq_keywords(question, max_themes)

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