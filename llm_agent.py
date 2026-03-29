# llm_agent.py
# Variante "safe" : ne crée pas le client OpenAI au moment de l'import.
# Comme ça, l'app Streamlit ne plante pas si `OPENAI_API_KEY` n'est pas défini
# (elle pourra alors utiliser le fallback FAQ).

from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

MODEL = "gpt-4o-mini"

# Prompt système : cadre métier, ton, garde-fous (ne remplace pas l’analyste humain).
_SYSTEM_PROMPT = """Tu es un assistant métier pour un service d’assurance crédit en France.

Rôle :
- Expliquer le risque, la tarification indicative et la décision **proposée** à partir des données fournies.
- Rédiger, si demandé, des courriers ou synthèses **à titre d’aide à la rédaction**, jamais comme engagement contractuel.

Règles strictes :
- Réponds en français, ton professionnel, clair, institutionnel.
- Ne fais aucune promesse d’acceptation/refus définitif : la décision finale relève toujours d’un analyste humain.
- N’invente aucune donnée, aucun montant, aucun seuil ou réglementation non présents dans le contexte.
- Si une information manque pour répondre, dis-le en une phrase et propose ce qui peut être dit avec les données disponibles.
- Ne donne pas de conseil juridique personnalisé ; reste factuel par rapport au contexte métier fourni.
- Pour les courriers : style formel, à la troisième personne pour désigner le client, sans signature d’engagement."""

_INTENT_HINTS = (
    (
        ("courrier", "lettre", "mail", "courriel", "notification", "mise en demeure"),
        "COURRIER",
        "Rédige un courrier formel, professionnel, structuré (objet/synthèse si pertinent, corps, formule de politesse). "
        "Ne prends aucune décision finale ; explique la situation et les éléments à partir du contexte.",
    ),
    (
        ("résumé", "resume", "synthèse", "synthese", "profil", "vue d'ensemble"),
        "RÉSUMÉ",
        "Fais une synthèse courte du profil : risque, classe, prime indicative, décision proposée, "
        "puis 2 facteurs SHAP les plus pertinents (sans jargon technique inutile).",
    ),
    (
        ("prime", "tarif", "coût", "montant", "plancher", "plafond"),
        "PRIME",
        "Concentre-toi sur la prime finale et la logique de calcul (sans inventer de formules). "
        "Rappelle les bornes si mentionnées dans le contexte.",
    ),
    (
        ("risque", "shap", "facteur", "explication", "impact"),
        "RISQUE",
        "Concentre-toi sur le score de risque, la classe et l’interprétation des facteurs listés (SHAP). "
        "Reste factuel.",
    ),
    (
        ("décision", "decision", "acceptation", "refus", "étude", "approfondie"),
        "DÉCISION",
        "Concentre-toi sur la décision proposée et ce qu’elle implique en termes de suivi, sans statuer définitivement.",
    ),
)


def _infer_intent(question: str) -> tuple[str, str]:
    """Retourne (code_intent, instructions_spécifiques)."""
    q = (question or "").lower()
    for keywords, code, hint in _INTENT_HINTS:
        if any(k in q for k in keywords):
            return code, hint
    return "GÉNÉRAL", "Réponds de façon directe à la question, en t’appuyant uniquement sur le contexte client."


def _get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return ChatOpenAI(
        model=MODEL,
        temperature=0,
        api_key=api_key,
    )


def call_llm(prompt: str) -> str:
    llm = _get_llm()
    if llm is None:
        raise RuntimeError(
            "OPENAI_API_KEY manquant. Renseigne la clé dans le fichier `.env` "
            "à la racine du projet (ou désactive `use_llm` pour utiliser le fallback FAQ)."
        )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def build_llm_prompt(
    question: str,
    result: dict,
    shap_pos: list,
    shap_neg: list,
    chat_history: list = None,
) -> str:
    """Prompt utilisateur structuré : contexte, intention, contraintes, format."""

    hist_text = ""
    if chat_history:
        hist_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-8:]])

    top_pos = ", ".join(shap_pos) if shap_pos else "aucun"
    top_neg = ", ".join(shap_neg) if shap_neg else "aucun"
    intent_code, intent_instructions = _infer_intent(question)

    return f"""## Contexte métier (données fournies — ne pas extrapoler)
- Score de risque (probabilité indicative) : {result['score_risque']:.3f}
- Classe de risque : {result['classe_risque']}
- Prime finale (indicative) : {result['prime_finale']} €
- Décision proposée (indicative) : {result['decision']}
- Facteurs augmentant le risque (SHAP) : {top_pos}
- Facteurs réduisant le risque (SHAP) : {top_neg}

## Historique conversationnel (récent)
{hist_text if hist_text else "Aucun."}

## Intention détectée (pour guider le ton)
- Code : {intent_code}
- Consigne : {intent_instructions}

## Question de l’utilisateur
{question}

## Consignes de réponse
1. Réponds **uniquement** à la question ; ne parle pas de sujets non demandés (ex. si la question porte sur le risque, ne détaille pas la prime sauf si nécessaire pour répondre).
2. Longueur : **court** pour risque/prime/décision (environ 4–8 lignes utiles) ; **plus développé** pour courrier ou résumé (jusqu’à ~12 lignes ou 3 courts paragraphes).
3. Format : Markdown simple (titres **optionnels**, listes à puces si ça clarifie).
4. Termine toujours par une phrase du type : « La décision finale relève d’un analyste humain. » (ou équivalent professionnel).
5. Si la question est ambiguë, commence par une courte clarification de ce que tu comprends, puis réponds.

---

## Réponse attendue
"""
