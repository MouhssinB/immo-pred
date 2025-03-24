# Utiliser une image Python officielle comme image de base
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .
COPY main.py .
COPY boucle_appel.py .
COPY templates/ templates/
COPY data/ data/
COPY tests/ tests/
COPY prometheus.yml .


# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exécuter les tests avec pytest
RUN python -m pytest tests/test_train.py -v

# Exposer le port sur lequel l'application s'exécute
EXPOSE 8000

# Commande pour démarrer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 
