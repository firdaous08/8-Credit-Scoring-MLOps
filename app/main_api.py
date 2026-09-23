import json
import os
import time
import joblib
import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Credit Scoring API - High Performance ONNX",
    version="2.0",
    description="API de prédiction de risque de crédit optimisée via ONNX Runtime avec monitoring et détection de drift intégrés."
)

# --- CONFIGURATION ET CHARGEMENT DU MODÈLE OPTIMISÉ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

ONNX_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "lgbm_model.onnx")
PKL_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "lgbm_model.pkl")
LOG_FILE = os.path.join(PROJECT_ROOT, "api_logs.jsonl")
DRIFT_REPORT_PATH = os.path.join(PROJECT_ROOT, "data_drift_report.html")

# Récupération de la liste ordonnée des colonnes d'entraînement
ref_model = joblib.load(PKL_MODEL_PATH)
if hasattr(ref_model, "feature_name_"):
    EXPECTED_COLUMNS = list(ref_model.feature_name_)
elif hasattr(ref_model, "feature_names_in_"):
    EXPECTED_COLUMNS = list(ref_model.feature_names_in_)
else:
    EXPECTED_COLUMNS = []

NUM_FEATURES = len(EXPECTED_COLUMNS)
COL_TO_IDX = {col: i for i, col in enumerate(EXPECTED_COLUMNS)}

# Initialisation de la session ONNX Runtime
onnx_session = rt.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
onnx_input_name = onnx_session.get_inputs()[0].name

# Seuil métier d'octroi de crédit
BEST_THRESHOLD = 0.51


# --- SCHÉMA DE VALIDATION DES ENTRÉES (PYDANTIC) ---
class ClientData(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Âge du client (entre 18 et 100 ans)")
    is_female: int = Field(..., ge=0, le=1, description="1 si femme, 0 si homme")
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Revenu total annuel du client")
    AMT_CREDIT: float = Field(..., gt=0, description="Montant du crédit sollicité")
    EXT_SOURCE_2: float = Field(..., ge=0.0, le=1.0, description="Score externe normalisé 2")
    EXT_SOURCE_3: float = Field(..., ge=0.0, le=1.0, description="Score externe normalisé 3")


# --- MIDDLEWARE DE MONITORING ET AUDIT EN PRODUCTION ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    payload = None

    if request.method == "POST" and request.url.path == "/predict":
        try:
            body_bytes = await request.body()
            if body_bytes:
                payload = json.loads(body_bytes.decode("utf-8"))

            async def receive():
                return {"type": "http.request", "body": body_bytes}

            request._receive = receive
        except Exception:
            payload = None

    response = await call_next(request)
    latency_ms = round((time.time() - start_time) * 1000, 2)

    if request.url.path == "/predict":
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "request_payload": payload
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    return response


# --- ROUTES DE L'API ---

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Credit Scoring API is online",
        "engine": "ONNX Runtime High Performance",
        "version": "2.0"
    }


@app.post("/predict")
def predict(client: ClientData):
    # Construction vectorisée directe (NumPy float32) pour éliminer le surcoût Pandas
    vec = np.zeros((1, NUM_FEATURES), dtype=np.float32)

    mapping = {
        "AMT_INCOME_TOTAL": client.AMT_INCOME_TOTAL,
        "AMT_CREDIT": client.AMT_CREDIT,
        "EXT_SOURCE_2": client.EXT_SOURCE_2,
        "EXT_SOURCE_3": client.EXT_SOURCE_3,
        "DAYS_BIRTH": -float(client.age) * 365.0,
        "CODE_GENDER_F": 1.0 if client.is_female == 1 else 0.0,
        "CODE_GENDER_M": 0.0 if client.is_female == 1 else 1.0
    }

    for col_name, val in mapping.items():
        if col_name in COL_TO_IDX:
            vec[0, COL_TO_IDX[col_name]] = val

    # Exécution de l'inférence via le runtime ONNX
    outputs = onnx_session.run(None, {onnx_input_name: vec})
    proba = float(outputs[1][0][1])
    decision = "Prêt Refusé" if proba >= BEST_THRESHOLD else "Prêt Accordé"

    return {
        "status": "success",
        "default_probability": round(proba, 4),
        "decision": decision
    }


@app.get("/drift")
def get_drift_report():
    if not os.path.exists(DRIFT_REPORT_PATH):
        raise HTTPException(
            status_code=404,
            detail="Le rapport de Data Drift n'a pas encore été généré sur le serveur."
        )
    return FileResponse(DRIFT_REPORT_PATH, media_type="text/html")