from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import pickle
from datetime import datetime
import pandas as pd
import joblib
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time
# Supprimer les métriques existantes avant de les redéfinir
from prometheus_client import REGISTRY
collectors = list(REGISTRY._collector_to_names.keys())
for collector in collectors:
    REGISTRY.unregister(collector)

# Métriques Prometheus
PREDICTION_REQUEST_COUNT = Counter(
    'prediction_request_total',
    'Nombre total de requêtes de prédiction'
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Temps de traitement des prédictions'
)

PREDICTION_VALUE = Histogram(
    'prediction_value_euros',
    'Distribution des valeurs prédites',
    buckets=[100000, 200000, 300000, 400000, 500000, 1000000, float("inf")]
)

# Chargement du modèle
model = joblib.load(r"data/model.pkl")

app = FastAPI()

# Middleware pour mesurer le temps de réponse
class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        PREDICTION_LATENCY.observe(duration)
        return response

app.add_middleware(PrometheusMiddleware)

# Stockage en mémoire des appels pour monitoring
calls_data = []  # Chaque élément sera un dictionnaire {"timestamp": datetime, "anxiety_score": float}

class PredictionInput(BaseModel):
    Nombre_pieces_principales: float
    Surface_reelle_bati: float

@app.post("/predict")
async def predict(input: PredictionInput):
    PREDICTION_REQUEST_COUNT.inc()
    
    value = pd.DataFrame([[input.Nombre_pieces_principales, input.Surface_reelle_bati]], 
                        columns=['Nombre pieces principales', 'Surface reelle bati'])
    
    prediction = model.predict(value)
    score = float(prediction[0])
    
    # Enregistrer la valeur prédite dans l'histogramme
    PREDICTION_VALUE.observe(score)
    
    # Enregistrement de l'appel avec l'heure et le score prédit
    calls_data.append({"timestamp": datetime.now(), "prix_estime": score})
    return {"Prix_estime": score}

# Endpoint Prometheus pour exposer les métriques brutes
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# IHM de prédiction
@app.get("/", response_class=HTMLResponse)
async def home():
    with open('templates/index.html', 'r') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Endpoint pour récupérer les métriques d'appels
@app.get("/metrics", response_class=JSONResponse)
async def metrics():
    data = [{"timestamp": call["timestamp"].isoformat(), "prix_estime": call["prix_estime"]} for call in calls_data]
    return {"calls": data}


# Dashboard Prometheus
@app.get("/prometheus", response_class=HTMLResponse)
async def prometheus_dashboard():
    with open('templates/prometheus_dashboard.html', 'r') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Lancer l'application via uvicorn (exécutable directement)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
