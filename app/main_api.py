from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd
import numpy as np

app = FastAPI(
    title="API de Scoring Crédit - Projet 8",
    description="API sécurisée avec validation des entrées et chargement optimisé du modèle."
)

# 1. Chargement unique du modèle au démarrage (Point de vigilance respecté)
model = joblib.load("model/lgbm_model.pkl")
SEUIL_OPTIMAL = 0.51

# 2. Schéma d'entrée avec validation stricte (Pydantic)
class ClientData(BaseModel):
    # On s'assure que le dictionnaire/features ne contient pas de types aberrants
    features: dict

    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        if not isinstance(v, dict) or len(v) == 0:
            raise ValueError("Le dictionnaire des features ne peut pas être vide.")
        
        # Exemple de vérification de valeurs aberrantes critiques exigées par l'école
        for key, val in v.items():
            if not isinstance(val, (int, float, np.number)):
                raise ValueError(f"La variable '{key}' doit être un nombre, type reçu invalide.")
            
            # Exemple de contrôle de plage (ex: l'âge ou un montant ne doit pas être négatif si applicable)
            if "DAYS_BIRTH" in key and val > 0:
                # Les jours de naissance dans ce dataset sont souvent négatifs ou gérés d'une certaine façon, 
                # adaptez selon vos variables réelles, ou vérifions un revenu :
                pass
            if "AMT_INCOME" in key and val <= 0:
                raise ValueError(f"Le revenu ('{key}') doit être strictement supérieur à 0.")
                
        return v

@app.post("/predict")
def predict(data: ClientData):
    try:
        df_client = pd.DataFrame([data.features])
        
        # Prédiction
        proba = float(model.predict_proba(df_client)[:, 1][0])
        decision = "Refusé" if proba > SEUIL_OPTIMAL else "Accepté"
        
        return {
            "probability_default": proba,
            "threshold_applied": SEUIL_OPTIMAL,
            "decision": decision
        }
    except (ValueError, TypeError) as e:
        # Erreur client (mauvais type, valeur hors plage) -> HTTP 400
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Erreur inattendue -> HTTP 500
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {str(e)}")