# llm_agent.py
from dotenv import load_dotenv
import os

# LangChain 
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

MODEL = "gpt-4o-mini"

llm = ChatOpenAI(
    model_name=MODEL,                
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")  
)

def call_llm(prompt: str) -> str:
    """Appel LLM"""
    messages = [
        SystemMessage(content="Tu es un assistant métier expert en assurance crédit, factuel et institutionnel."),
        HumanMessage(content=prompt)
    ]
    # invoke
    response = llm.invoke(messages)
    return response.content.strip()


def build_llm_prompt(
    question: str,
    result: dict,
    shap_pos: list,
    shap_neg: list,
    chat_history: list = None
) -> str:
    """Prompt structuré Rôle / Contexte / Tâche / Contraintes"""

    hist_text = ""
    if chat_history:
        hist_text = "\n".join(
            [f"{m['role']}: {m['content']}" for m in chat_history[-5:]]
        )

    top_pos = ", ".join(shap_pos) if shap_pos else "aucun"
    top_neg = ", ".join(shap_neg) if shap_neg else "aucun"

    return f"""
RÔLE :
Tu es un assistant métier expert en assurance crédit, factuel et institutionnel.

CONTEXTE :
Historique de conversation récent :
{hist_text if hist_text else "Aucun historique disponible."}

Données client :
- Score de risque : {result['score_risque']:.3f}
- Classe de risque : {result['classe_risque']}
- Prime finale : {result['prime_finale']} €
- Décision proposée : {result['decision']}
- Facteurs augmentant le risque : {top_pos}
- Facteurs réduisant le risque : {top_neg}

TÂCHE :
Répond uniquement à la question posée par l'utilisateur.
Comprends automatiquement le sujet (risque, prime, décision, courrier, résumé, etc.) sans utiliser de mot-clé.
Ne mélange jamais les sujets : si la question porte sur le risque, ne parle pas de prime ni de décision.
Si la question demande un résumé du client, synthétise les données et les facteurs de risque.
Si la question demande un courrier :  rédige un courrier formel, professionnel, institutionnel , adresse le client à la troisième personne , explique la décision et les éléments clés et ne prends aucune décision finale.

CONTRAINTES :
- Réponse concise : max 6 lignes pour risque/primes/decision, max 10 lignes pour courrier/résumé
- Factuel, institutionnel, neutre
- Ne jamais mélanger les sujets
- Ne rien inventer
- Mentionne toujours que la décision finale relève d’un analyste humain
- Utilise les données du client uniquement

QUESTION :
{question}

RÉPONSE :
"""