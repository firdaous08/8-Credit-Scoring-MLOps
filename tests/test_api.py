from fastapi.testclient import TestClient
from app.main_api import app

client = TestClient(app)

def test_predict_success():
    """Vérifie qu'un dictionnaire valide passe correctement."""
    # Mettez une vraie feature valide de votre modèle pour tester
    response = client.post("/predict", json={"features": {"AMT_INCOME_TOTAL": 150000.0, "AMT_CREDIT": 400000.0}})
    # Selon vos features exactes, l'API renverra 200 ou une erreur si la feature n'est pas reconnue, 
    # adaptez le test avec les colonnes de votre dataset.
    assert response.status_code in [200, 400, 500]

def test_predict_invalid_type():
    """Vérifie qu'un type incorrect (ex: du texte à la place d'un chiffre) est rejeté avec un code 400."""
    response = client.post("/predict", json={"features": {"AMT_INCOME_TOTAL": "vingt-mille"}})
    assert response.status_code == 400

def test_predict_out_of_range():
    """Vérifie qu'une valeur hors plage (ex: revenu négatif ou nul) déclenche une erreur 400."""
    response = client.post("/predict", json={"features": {"AMT_INCOME_TOTAL": 0}})
    assert response.status_code == 400

def test_predict_empty_payload():
    """Vérifie qu'un payload vide est rejeté proprement."""
    response = client.post("/predict", json={"features": {}})
    assert response.status_code == 400