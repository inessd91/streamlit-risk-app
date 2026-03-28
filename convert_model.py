"""
convert_model.py
================
À exécuter UNE SEULE FOIS en local (Python 3.9) pour re-exporter le modèle
dans un format compatible avec n'importe quelle version Python / XGBoost.

    python convert_model.py

Produit :
  - xgb_booster.json      → booster XGBoost natif (indépendant de Python)
  - preprocessor.joblib   → pipeline de prétraitement sklearn seul
"""

import warnings
import joblib
from pathlib import Path

BASE = Path(__file__).resolve().parent

# ── Charger le pipeline complet ───────────────────────────────────────────────
pipeline_path = BASE / "xgboost_final_model.joblib"
if not pipeline_path.exists():
    pipeline_path = BASE / "notebooks" / "xgboost_final_model.joblib"

print(f"Chargement depuis : {pipeline_path}")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    pipeline = joblib.load(str(pipeline_path))

preprocess = pipeline.named_steps["preprocess"]
xgb_model  = pipeline.named_steps["model"]
booster    = xgb_model.get_booster()

# ── Sauvegarder le booster en JSON natif XGBoost ─────────────────────────────
booster_path = BASE / "xgb_booster.json"
booster.save_model(str(booster_path))
print(f"Booster sauvegardé  : {booster_path}")

# ── Sauvegarder le préprocesseur seul ─────────────────────────────────────────
preprocess_path = BASE / "preprocessor.joblib"
joblib.dump(preprocess, str(preprocess_path), compress=3)
print(f"Préprocesseur sauvegardé : {preprocess_path}")

print("\nConversion terminée. Ajoutez xgb_booster.json et preprocessor.joblib au repo GitHub.")