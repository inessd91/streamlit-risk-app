"""
convert_model.py
================
À exécuter UNE SEULE FOIS en local (Python 3.9) pour re-exporter le modèle
dans un format compatible avec n'importe quelle version Python / XGBoost.

    python convert_model.py

Produit :
  - xgb_booster.json      → booster XGBoost natif (indépendant de Python)
  - preprocessor_params.npz   
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
print("\n===== PREPROCESS STRUCTURE =====")
print(preprocess)
print("================================\n")

xgb_model  = pipeline.named_steps["model"]
booster    = xgb_model.get_booster()

# ── Sauvegarder le booster en JSON natif XGBoost ─────────────────────────────
booster_path = BASE / "xgb_booster.json"
booster.save_model(str(booster_path))
print(f"Booster sauvegardé  : {booster_path}")

# ── Sauvegarder le préprocesseur seul ─────────────────────────────────────────
import numpy as np

# récupérer le scaler interne
scaler = preprocess.named_transformers_["num"]

np.savez(
    BASE / "preprocessor_params.npz",
    mean=scaler.mean_,
    scale=scaler.scale_
)

print(" Scaler sauvegardé : preprocessor_params.npz")

print("\nConversion terminée. Ajoutez xgb_booster.json et preprocessor_params.npz au repo GitHub.")