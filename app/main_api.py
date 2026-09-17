import time
import json
import logging
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd
import numpy as np

# === CONFIGURATION MLOps : Stockage des logs de production (PoC) ===
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("api_logs.jsonl"), # Sauvegarde sur le disque
        logging.StreamHandler()                # Affichage console
    ]
)

app = FastAPI(
    title="API de Scoring Crédit - Projet 8",
    description="API avec schéma simplifié, validation, alignement automatique et logging de production."
)

model = joblib.load("model/lgbm_model.pkl")
SEUIL_OPTIMAL = 0.51

if hasattr(model, "feature_name_"):
    EXPECTED_COLUMNS = model.feature_name_
elif hasattr(model, "feature_names_in_"):
    EXPECTED_COLUMNS = model.feature_names_in_
else:
    EXPECTED_COLUMNS = []

@app.middleware("http")
async def log_requests_and_performance(request: Request, call_next):
    start_time = time.time()
    
    body_dict = {}
    if request.method == "POST" and "/predict" in request.url.path:
        body_bytes = await request.body()
        if body_bytes:
            try:
                body_dict = json.loads(body_bytes.decode("utf-8"))
            except Exception:
                pass
            async def receive():
                return {"type": "http.request", "body": body_bytes}
            request._receive = receive

    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000

    log_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": round(process_time, 2),
        "request_payload": body_dict
    }
    
    # Enregistre le log au format JSON dans api_logs.jsonl via le middleware
    logging.info(json.dumps(log_data))
    return response

class ClientData(BaseModel):
    age: int = Field(..., description="Âge du client en années", example=35)
    is_female: int = Field(..., description="Genre (1 pour Femme, 0 pour Homme)", example=1)
    AMT_INCOME_TOTAL: float = Field(..., description="Revenu total du client", example=120000.0)
    AMT_CREDIT: float = Field(..., description="Montant du crédit demandé", example=150000.0)
    EXT_SOURCE_2: float = Field(0.5, description="Score externe 2 (0 à 1)", example=0.7)
    EXT_SOURCE_3: float = Field(0.5, description="Score externe 3 (0 à 1)", example=0.6)

    @field_validator('AMT_INCOME_TOTAL')
    @classmethod
    def validate_income(cls, v):
        if v <= 0:
            raise ValueError("Le revenu total doit être strictement supérieur à 0.")
        return v

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de Scoring Crédit. Ajoutez /docs à l'URL pour accéder à l'interface Swagger."}

@app.get("/drift", response_class=HTMLResponse)
def get_drift_report():
    report_path = "data_drift_report.html"
    if not os.path.exists(report_path):
        return "<h3>Rapport de dérive indisponible. Générez-le d'abord avec drift_analysis.py.</h3>"
    with open(report_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/drift/download")
def download_drift_report():
    report_path = "data_drift_report.html"
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Rapport de dérive introuvable.")
    return FileResponse(
        path=report_path,
        filename="rapport_data_drift.html",
        media_type="text/html"
    )

@app.post("/predict")
def predict(data: ClientData):
    try:
        features_dict = {
            "AMT_INCOME_TOTAL": data.AMT_INCOME_TOTAL,
            "AMT_CREDIT": data.AMT_CREDIT,
            "EXT_SOURCE_2": data.EXT_SOURCE_2,
            "EXT_SOURCE_3": data.EXT_SOURCE_3,
            "DAYS_BIRTH": -int(data.age) * 365
        }
        
        if data.is_female == 1:
            features_dict["CODE_GENDER_F"] = 1
        else:
            features_dict["CODE_GENDER_M"] = 1

        input_df = pd.DataFrame([features_dict])
        
        if len(EXPECTED_COLUMNS) > 0:
            full_df = pd.DataFrame(0.0, index=[0], columns=EXPECTED_COLUMNS)
            for col in input_df.columns:
                if col in full_df.columns:
                    full_df[col] = input_df[col].values
            df_to_predict = full_df
        else:
            df_to_predict = input_df

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