from fastapi.testclient import TestClient
from main import app
import pytest
import os

client = TestClient(app)


def test_model_file_exists():
    """Test de la présence du fichier model.pkl"""
    assert os.path.exists("data/model.pkl"), "Le fichier model.pkl n'existe pas dans le dossier data/"

def test_predict_endpoint():
    """Test de l'endpoint /predict avec des données valides"""
    test_input = {
        "Nombre_pieces_principales": 4.0,
        "Surface_reelle_bati": 100.0
    }
    response = client.post("/predict", json=test_input)
    assert response.status_code == 200
    assert "Prix_estime" in response.json()
    assert isinstance(response.json()["Prix_estime"], float)

def test_predict_invalid_input():
    """Test de l'endpoint /predict avec des données invalides"""
    test_input = {
        "Nombre_pieces_principales": "invalid",
        "Surface_reelle_bati": 100.0
    }
    response = client.post("/predict", json=test_input)
    assert response.status_code == 422

def test_home_endpoint():
    """Test de l'endpoint / qui retourne la page HTML"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_metrics_endpoint():
    """Test de l'endpoint /metrics qui retourne les métriques"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "calls" in response.json()
    assert isinstance(response.json()["calls"], list)

