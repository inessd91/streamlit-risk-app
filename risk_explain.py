FEATURE_LABELS = {
    "score_credit": "Score de crédit",
    "ratio_dette_revenu": "Ratio dette / revenu",
    "charges_totales": "Charges mensuelles",
    "dette_totale": "Dette totale",
    "revenu_par_incident": "Revenu par incident",
    "historique_credit": "Historique de crédit",
    "assurance_sur_revenu": "Assurance sur revenu",
    "montant_assurance": "Montant assuré",
    "age": "Âge"
}

def explain_risk_from_shap(shap_values, feature_names, top_k=4):
    values = shap_values.values[0]
    impacts = list(zip(feature_names, values))
    impacts = sorted(impacts, key=lambda x: abs(x[1]), reverse=True)

    detailed = []
    MIN_SHAP = 0.003
    for feat, val in impacts[:top_k]:
        if abs(val) < MIN_SHAP: continue
        label = FEATURE_LABELS.get(feat, feat)
        direction = "augmente le risque" if val > 0 else "réduit le risque"
        impact = round(val, 3)
        detailed.append({"label": label, "direction": direction, "impact": impact})
    return detailed
