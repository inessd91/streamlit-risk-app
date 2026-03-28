# ========================================================
# IMPORTS
# ========================================================
import numpy as np
import streamlit as st
import pandas as pd

import shap
import matplotlib.pyplot as plt
import hashlib, json
import xgboost as xgb

from risk_explain import explain_risk_from_shap, FEATURE_LABELS
from assistant_hybride import assistant_hybride, summarize_client
import streamlit.components.v1 as components


# ============================================================
# CONFIG STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Évaluation du Profil de Risque Client",
    layout="wide"
)

st.title("Évaluation du Profil de Risque Client")
st.write("""
Cette application évalue le risque financier d'un client et propose une tarification adaptée.
Elle fournit également une décision métier, des explications SHAP et un assistant.
""")


# ============================================================
# MENU
# ============================================================

page = st.sidebar.radio(
    "Menu",
    ["Profil de risque", "Tarification & Décision", "Assistant métier"]
)


# ============================================================
# SAISIE CLIENT
# ============================================================

st.sidebar.header("Saisie des données client")

with st.sidebar.expander("Informations personnelles", expanded=True):
    age = st.slider("Âge", 18, 100, 40)

with st.sidebar.expander("Situation financière", expanded=True):
    revenu_par_incident = st.number_input("Revenu par incident (€)", 0, 200000, 30000)
    dette_totale = st.number_input("Dette totale (€)", 0, 500000, 20000)
    charges_totales = st.number_input("Charges mensuelles (€)", 0, 10000, 1500)
    ratio_dette_revenu = st.slider("Ratio dette / revenu", 0.0, 5.0, 0.3)

with st.sidebar.expander("Crédit & Assurance", expanded=True):
    historique_credit = st.slider("Historique de crédit", 0, 100, 70)
    score_credit = st.slider("Score crédit", 0, 1000, 650)
    assurance_sur_revenu = st.slider("Assurance sur revenu (%)", 0, 100, 30)
    montant_assurance = st.number_input("Montant de l'assurance (€)", 0, 100000, 10000)


payload = {
    "age": age,
    "revenu_par_incident": revenu_par_incident,
    "assurance_sur_revenu": assurance_sur_revenu,
    "historique_credit": historique_credit,
    "dette_totale": dette_totale,
    "charges_totales": charges_totales,
    "score_credit": score_credit,
    "montant_assurance": montant_assurance,
    "ratio_dette_revenu": ratio_dette_revenu
}


selected_features = [
    "revenu_par_incident", "assurance_sur_revenu", "historique_credit",
    "dette_totale", "charges_totales", "score_credit",
    "montant_assurance", "ratio_dette_revenu", "age"
]


# ============================================================
# CHARGEMENT MODELE (CLEAN + SAFE)
# ============================================================

import numpy as np

@st.cache_resource
def load_artifacts():
    # Charger booster XGBoost (safe)
    booster = xgb.Booster()
    booster.load_model("xgb_booster.json")

    # Charger scaler en numpy (safe)
    params = np.load("preprocessor_params.npz")
    mean = params["mean"]
    scale = params["scale"]

    return mean, scale, booster


mean, scale, xgb_local = load_artifacts()

# ============================================================
# MOTEUR METIER
# ============================================================

SEUILS_RISQUE = [0.2, 0.6]
PRIME_MINIMALE = 500
PRIME_MAXIMALE = 5000

COEFS = {
    "Risque faible": 1.0,
    "Risque moyen": 1.2,
    "Risque élevé": 1.4
}

DECISIONS = {
    "Risque faible": "🟢 Acceptation standard",
    "Risque moyen": "🟠 Acceptation avec surprime",
    "Risque élevé": "🔴 Étude approfondie"
}


def predict(payload):
    X = pd.DataFrame([payload])[selected_features]

    # preprocessing SAFE (sans sklearn)
    X_values = X.values.astype(float)
    X_proc = (X_values - mean) / scale

    # prediction
    dmat = xgb.DMatrix(X_proc)
    score_risque = float(xgb_local.predict(dmat)[0])

    # logique métier
    if score_risque < 0.3:
        classe = "Risque faible"
        coef = 1.0
        decision = "Accepté"
    elif score_risque < 0.6:
        classe = "Risque moyen"
        coef = 1.2
        decision = "Accepté avec réserve"
    else:
        classe = "Risque élevé"
        coef = 1.5
        decision = "Refus ou étude approfondie"

    prime_theorique = score_risque * montant_assurance * 1.1
    prime_finale = max(min(prime_theorique * coef, PRIME_MAXIMALE), PRIME_MINIMALE)

    return {
    "score_risque": score_risque,
    "classe_risque": classe,
    "decision": decision,
    "prime_theorique": float(prime_theorique),
    "prime_finale": float(prime_finale),
    "prime_minimale": PRIME_MINIMALE,
    "prime_maximale": PRIME_MAXIMALE
    }

result = predict(payload)

# ============================================================
# SIGNATURE CLIENT & MEMOIRE CONVERSATIONNELLE
# ============================================================

def client_signature(payload: dict) -> str:
    """Retourne un hash unique du client selon ses données"""
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()

current_client_signature = client_signature(payload)

if "client_signature" not in st.session_state:
    st.session_state.client_signature = current_client_signature
    st.session_state.chat_history = []
    st.session_state.user_question = ""
    st.session_state.use_llm = True

elif st.session_state.client_signature != current_client_signature:
    # Nouveau client → reset mémoire
    st.session_state.client_signature = current_client_signature
    st.session_state.chat_history = []
    st.session_state.user_question = ""
    st.session_state.use_llm = True

# ============================================================
# SHAP
# ============================================================

X_input = pd.DataFrame([payload])[selected_features]
X_values = X_input.values.astype(float)
X_processed = (X_values - mean) / scale
X_processed_df = pd.DataFrame(X_processed, columns=selected_features)

explainer = shap.TreeExplainer(xgb_local)
shap_values = explainer(X_processed_df)

shap_details = explain_risk_from_shap(shap_values, selected_features)

# ============================================================
# PAGES
# ============================================================

# --------------------
# PAGE PROFIL DE RISQUE
# --------------------
if page == "Profil de risque":
    st.markdown("### Profil de risque du client")
    col1, col2, col3 = st.columns(3)
    col1.metric("Score de risque", f"{result['score_risque']:.3f}")
    col2.metric("Classe de risque", result["classe_risque"])
    col3.metric(
        "Niveau de vigilance",
        "Faible" if result["classe_risque"] == "Risque faible"
        else "Modéré" if result["classe_risque"] == "Risque moyen"
        else "Élevé"
    )

    if result["classe_risque"] == "Risque faible":
        st.success("Profil financier sain. Aucun facteur de risque majeur détecté.")
    elif result["classe_risque"] == "Risque moyen":
        st.warning("Profil globalement équilibré, avec certains points de vigilance.")
    else:
        st.error("Profil à risque. Plusieurs facteurs contribuent à une exposition élevée.")

    st.markdown("### Principaux facteurs explicatifs du risque")
    shap_df = pd.DataFrame({
        "feature": selected_features,
        "impact": shap_values.values[0]
    })
    shap_df["impact_abs"] = shap_df["impact"].abs()
    top_features = shap_df.sort_values("impact_abs", ascending=False).head(5)
    top_features["label"] = top_features["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in top_features["impact"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top_features["label"], top_features["impact"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Impact sur le score de risque")
    ax.set_title("Variables les plus influentes")
    plt.gca().invert_yaxis()
    st.pyplot(fig)
    st.caption("🔴 Augmente le risque  |  🟢 Réduit le risque — comparaison par rapport à un client moyen du portefeuille.")
    


# ------------------------------
# PAGE TARIFICATION & DECISION
# ------------------------------
elif page == "Tarification & Décision":
    st.header("Tarification et décision du client")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Prime théorique", f"{int(result['prime_theorique'])} €")
    col2.metric("Plancher tarifaire", f"{result['prime_minimale']} €")
    col3.metric("Plafond tarifaire", f"{PRIME_MAXIMALE} €")
    col4.metric("Prime finale", f"{int(result['prime_finale'])} €")

    st.markdown(
        "La **prime finale** est calculée à partir de la prime théorique, "
        "ajustée selon le coefficient lié à la classe de risque, "
        "en respectant le plancher et le plafond tarifaire."
    )
    st.markdown("---")

    st.subheader("Décision proposée")
    if result["classe_risque"] == "Risque faible":
        st.success(f"{result['decision']} – Acceptation standard. Aucun ajustement requis.")
    elif result["classe_risque"] == "Risque moyen":
        st.warning(f"{result['decision']} – Acceptation avec ajustement tarifaire. Surveillance recommandée.")
    else:
        st.error(f"{result['decision']} – Étude approfondie requise avant acceptation.")

# ------------------------------
# PAGE ASSISTANT METIER
# ------------------------------
elif page == "Assistant métier":
    st.subheader("Assistant métier")
    st.caption(
        "Posez votre question métier concernant le risque, la prime, la décision, ou demandez un résumé/courrier du client."
    )

    
    # Initialisation session_state
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "use_llm" not in st.session_state:
        st.session_state.use_llm = True

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""

    if "submit_question" not in st.session_state:
        st.session_state.submit_question = False

  
    # Checkbox LLM
  
    st.checkbox(
        "Afficher des explications détaillées du modèle",
        key="use_llm"
    )

    
    # Fonction appelée à l'entrée 
    
    def on_enter():
        question = st.session_state.question_input.strip()
        if question:
            st.session_state.pending_question = question
            st.session_state.submit_question = True

    
    # Champ texte pour question
    
    st.text_input(
        "💬 Votre question",
        key="question_input",
        value="",
        on_change=on_enter
    )
    
    # Bouton génération courrier client
    
    if st.button("✉️ Générer un courrier client"):
        st.session_state.pending_question = (
            "Courrier formel et professionnel à destination du client. "
        )
        st.session_state.submit_question = True

  
    # Préparer SHAP
    
    shap_pos = [
        f"{e['label']} (impact {e['impact']})"
        for e in shap_details if "augmente" in e["direction"]
    ]
    shap_neg = [
        f"{e['label']} (impact {e['impact']})"
        for e in shap_details if "réduit" in e["direction"]
    ]

    
    # Appel assistant si question saisie
   
    if st.session_state.submit_question:
        question = st.session_state.pending_question
        response = assistant_hybride(
            question=question,
            result=result,
            shap_pos=shap_pos,
            shap_neg=shap_neg,
            chat_history=st.session_state.chat_history,
            use_llm=st.session_state.use_llm
        )

        
        # Vérifier doublons avant ajout
        
        last_question = st.session_state.chat_history[-2]["content"] if len(st.session_state.chat_history) >= 2 else None
        last_response = st.session_state.chat_history[-1]["content"] if len(st.session_state.chat_history) >= 1 else None

        if question != last_question:
            st.session_state.chat_history.append({"role": "Utilisateur", "content": question})

        if response != last_response:
            st.session_state.chat_history.append({"role": "Assistant", "content": response})

        # Reset champ texte
        st.session_state.pending_question = ""
        st.session_state.submit_question = False

    
    # Affichage type chat avec scroll
    
    chat_container = st.container()
    chat_html = "<div style='max-height:500px; overflow-y:auto; border:1px solid #ddd; padding:10px;'>"
    for msg in st.session_state.chat_history:
        if msg["role"] == "Utilisateur":
            chat_html += f"<div style='text-align:right; margin:4px 0;'>💬 {msg['content']}</div>"
        else:
            chat_html += f"<div style='text-align:left; margin:4px 0;'>🤖 {msg['content']}</div>"
    chat_html += "</div>"

    chat_container.markdown(chat_html, unsafe_allow_html=True)

 