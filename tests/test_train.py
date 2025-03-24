import pytest
import pandas as pd
import os
import joblib


def test_model_training():
    """Test que le modèle est bien entraîné et sauvegardé"""
    # Vérifier que le fichier model.pkl existe
    assert os.path.exists("data/model.pkl"), "Le fichier model.pkl n'existe pas"
    
    # Charger le modèle et vérifier qu'il peut faire des prédictions
    model = joblib.load("data/model.pkl")
    test_data = pd.DataFrame({
        'Nombre pieces principales': [4.0],
        'Surface reelle bati': [100.0]
    })
    prediction = model.predict(test_data)
    assert isinstance(prediction[0], float), "La prédiction n'est pas un nombre flottant"

def test_model_predictions():
    """Test que les prédictions du modèle sont cohérentes"""
    model = joblib.load("data/model.pkl")
    
    # Test avec différentes valeurs
    small_house = pd.DataFrame({
        'Nombre pieces principales': [2.0],
        'Surface reelle bati': [30.0]
    })
    big_house = pd.DataFrame({
        'Nombre pieces principales': [6.0],
        'Surface reelle bati': [200.0]
    })
    
    small_pred = model.predict(small_house)[0]
    big_pred = model.predict(big_house)[0]
    
    # Vérifier que le prix d'une grande maison est supérieur à celui d'une petite
    assert big_pred > small_pred, "La prédiction n'est pas cohérente: une grande maison devrait coûter plus cher qu'une petite"
