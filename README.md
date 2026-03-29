# Insurance Risk Scoring App

Cette application Streamlit permet d’évaluer le risque financier d’un client et de proposer une tarification indicative à partir d’un modèle de Machine Learning.

**Application en ligne :**   
https://insurance-risk-scoring.streamlit.app/ 

## Objectif du projet

L’objectif est de reproduire un cas d’usage concret en assurance crédit :  
estimer le risque d’impayé d’un client, proposer une décision métier (acceptation, réserve, étude) et calculer une prime adaptée.

L’application a été conçue comme un outil d’aide à la décision, destiné à des utilisateurs métier, avec un accent particulier sur la transparence et l’interprétabilité des résultats.

## Fonctionnement général

L’application repose sur trois briques principales :
- un modèle de scoring basé sur XGBoost
- un moteur de tarification basé sur la perte attendue
- un module d’explicabilité utilisant SHAP

À cela s’ajoute un assistant métier permettant d’interroger le modèle et de mieux comprendre les résultats.  

## Modèle de scoring

Le modèle est un classifieur binaire qui estime la probabilité de défaut d’un client.

Cette probabilité est utilisée comme un score de risque continu, permettant :
- de segmenter les clients,
- d’ajuster la tarification,
- d’éviter une logique trop binaire (accepté / refusé).

## Assistant métier

L’application intègre un assistant conversationnel hybride :
- une FAQ locale, rapide et stable, pour répondre aux questions simples (score, prime, décision),
- un LLM optionnel, activable uniquement en environnement contrôlé, pour fournir des explications plus détaillées et contextualisées.

## Problèmes rencontrés lors du déploiement

Lors du passage en production sur Streamlit Cloud, plusieurs difficultés sont apparues :
- incompatibilités entre versions de scikit-learn (erreurs de type AttributeError)
- impossibilité de recharger certains objets sérialisés (ColumnTransformer)
- dépendance forte à l’environnement Python d’entraînement

Ces problèmes rendaient le modèle instable et difficilement déployable.

## Solution technique mise en place

Pour garantir la stabilité du modèle, une approche plus robuste a été adoptée :
- export du modèle XGBoost au format natif (xgb_booster.json)
- reconstruction du preprocessing sans scikit-learn, à partir des paramètres du StandardScaler sauvegardés en .npz

Cette solution permet :
- d’éviter les problèmes de compatibilité entre versions,
- de rendre le modèle indépendant de scikit-learn,
- de simplifier le chargement en production.

## Sécurisation de l’assistant LLM

L’utilisation d’un modèle de langage en environnement public pose un risque de coût lié aux appels API.

Pour cette raison, plusieurs mécanismes ont été mis en place :
- désactivation globale du LLM en mode démonstration (LLM_ENABLED = False)
- séparation entre l’interface utilisateur (checkbox) et l’activation réelle
- limitation du nombre de requêtes
- fallback automatique vers la FAQ locale

Ainsi :
- le LLM ne peut pas être activé par un utilisateur externe,
- aucun appel API n’est effectué en mode public,
- l’application reste entièrement fonctionnelle sans dépendance externe.

## Déploiement

L’application est déployée sur Streamlit Cloud.

Le modèle est chargé localement, ce qui garantit :
- une latence faible,
- une indépendance vis-à-vis d’API externes pour le scoring.

## MLOps et traçabilité

Le développement du modèle (entraînement, suivi, validation) est réalisé dans un environnement séparé :
- suivi des expériences avec MLflow
- validation des données avec Deepchecks
- versionnage via DagsHub

Cette séparation permet de garder une architecture claire entre la partie data science (expérimentation) et la partie application (déploiement)

## Lancer en local
    pip install -r requirements.txt

    streamlit run app.py

## Limites
- jeu de données de taille limitée
- score utilisé comme indicateur relatif (non calibré)
- seuils de risque définis de manière métier

Les résultats doivent être interprétés comme une aide à la décision, et non comme une décision automatique.

## Perspectives
- amélioration du matching FAQ (embeddings plus avancés)
- intégration d’un module RAG (documents métier)
- calibration probabiliste du modèle
- industrialisation complète via API et pipeline CI/CD

## Auteur

Projet réalisé dans le cadre d’un stage en Data Analytics, avec pour objectif de concevoir une solution complète de scoring intégrant modélisation, explicabilité, MLOps et interface décisionnelle.
