# assistant_hybride.py
from llm_agent import call_llm, build_llm_prompt
from faq import match_faq, get_faq_response

def assistant_hybride(question: str, result: dict, shap_pos: list, shap_neg: list, chat_history: list = None, use_llm: bool = True) -> str:
    """
    Assistant hybride : LLM intelligent + FAQ fallback
    Historique conversationnel utilisé pour contexte.
    """
    
    if chat_history is None:
        chat_history = []

    # --- LLM ---
    if use_llm:
        prompt = build_llm_prompt(question, result, shap_pos, shap_neg, chat_history)
        llm_response = call_llm(prompt)
        chat_history.append({"role": "Utilisateur", "content": question})
        chat_history.append({"role": "Assistant", "content": llm_response})
        return llm_response

    # --- FAQ fallback ---
    matched_themes = match_faq(question)
    if matched_themes:
        faq_responses = [get_faq_response(theme, shap_pos, shap_neg) for theme in matched_themes]
        response = "\n\n".join(faq_responses)
        chat_history.append({"role": "Utilisateur", "content": question})
        chat_history.append({"role": "Assistant", "content": response})
        return response

    # --- Réponse par défaut ---
    default_response = "Je ne peux pas répondre précisément à cette question. Merci de reformuler ou préciser votre demande."
    chat_history.append({"role": "Utilisateur", "content": question})
    chat_history.append({"role": "Assistant", "content": default_response})
    return default_response

def summarize_client(chat_history: list, result: dict, shap_pos: list, shap_neg: list) -> str:
    """Résumé synthétique du client à la demande"""
    question = "Fais un résumé synthétique du client"
    prompt = build_llm_prompt(question, result, shap_pos, shap_neg, chat_history)
    summary = call_llm(prompt)
    chat_history.append({"role": "Utilisateur", "content": question})
    chat_history.append({"role": "Assistant", "content": summary})
    return summary
