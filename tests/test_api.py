import sys
import os

# Ajoute la racine du projet au sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main_api import app

client = TestClient(app)

def test_read_root():
    """Vérifie que l'API répond correctement à la racine."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_success():
    """Vérifie qu'un payload valide produit une décision avec le runtime optimisé."""
    payload = {
        "age": 35,
        "is_female": 1,
        "AMT_INCOME_TOTAL": 150000.0,
        "AMT_CREDIT": 400000.0,
        "EXT_SOURCE_2": 0.7,
        "EXT_SOURCE_3": 0.6
    }
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
    assert "default_probability" in data

def test_predict_invalid_income():
    """Vérifie que la validation Pydantic rejette un revenu négatif."""
    payload = {
        "age": 45,
        "is_female": 0,
        "AMT_INCOME_TOTAL": -5000.0,
        "AMT_CREDIT": 4000000.0,
        "EXT_SOURCE_2": 0.1,
        "EXT_SOURCE_3": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422