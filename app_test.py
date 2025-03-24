import requests
from pydantic import BaseModel
import random 
import time , pandas as pd


# URL de l'API
url = "http://127.0.0.1:8000/predict"


class PredictionInput(BaseModel):
    Nombre_pieces_principales: float
    Surface_reelle_bati: float

# Nombre d'appels à effectuer
nombre_appels = 1000

for i in range(nombre_appels):
    try:
        time.sleep(random.randint(1, 3))
        # Créer une instance de PredictionInput
        input_data = PredictionInput(Nombre_pieces_principales=random.randint(0, 14), Surface_reelle_bati=random.randint(30, 200))
        # Effectuer une requête POST avec les données en JSON
        response = requests.post(url, json=input_data.dict())
        
        # Vérifier le statut de la réponse
        if response.status_code == 200:
            print(f"Appel {i + 1}: Succès - Réponse: {response.json()}")
        else:
            print(f"Appel {i + 1}: Échec - Code de statut: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Appel {i + 1}: Erreur - {e}")
