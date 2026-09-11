from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd
import numpy as np

app = FastAPI(
    title="API de Scoring Crédit - Projet 8",
    description="API sécurisée avec validation des entrées et alignement automatique des features."
)

# 1. Chargement unique du modèle au démarrage
model = joblib.load("model/lgbm_model.pkl")
SEUIL_OPTIMAL = 0.51

# Récupération automatique des colonnes attendues par le modèle LightGBM
if hasattr(model, "feature_name_"):
    EXPECTED_COLUMNS = model.feature_name_
elif hasattr(model, "feature_names_in_"):
    EXPECTED_COLUMNS = model.feature_names_in_
else:
    EXPECTED_COLUMNS = []

# 2. Schéma d'entrée avec validation stricte (Pydantic)
class ClientData(BaseModel):
    features: dict

    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        if not isinstance(v, dict) or len(v) == 0:
            raise ValueError("Le dictionnaire des features ne peut pas être vide.")
        
        for key, val in v.items():
            if not isinstance(val, (int, float, np.number)):
                raise ValueError(f"La variable '{key}' doit être un nombre, type reçu invalide.")
            
            if "AMT_INCOME" in key and val <= 0:
                raise ValueError(f"Le revenu ('{key}') doit être strictement supérieur à 0.")
                
        return v

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de Scoring Crédit. Ajoutez /docs à l'URL pour accéder à l'interface Swagger."}

@app.post("/predict")
def predict(data: ClientData):
    try:
        # Transformation du dictionnaire entrant en DataFrame
        input_df = pd.DataFrame([data.features])
        
        # Alignement automatique : création d'un DataFrame complet aux dimensions du modèle (rempli de 0 par défaut)
        if len(EXPECTED_COLUMNS) > 0:
            full_df = pd.DataFrame(0.0, index=[0], columns=EXPECTED_COLUMNS)
            for col in input_df.columns:
                if col in full_df.columns:
                    full_df[col] = input_df[col].values
            df_to_predict = full_df
        else:
            df_to_predict = input_df

        # Prédiction via le modèle LightGBM
        proba = float(model.predict_proba(df_to_predict)[:, 1][0])
        decision = "Refusé" if proba > SEUIL_OPTIMAL else "Accepté"
        
        return {
            "probability_default": proba,
            "threshold_applied": SEUIL_OPTIMAL,
            "decision": decision
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {str(e)}")