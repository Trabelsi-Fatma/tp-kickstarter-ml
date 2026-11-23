FROM python:3.12-slim

WORKDIR /app

# Les dépendances d'abord : l'étape est mise en cache tant que le fichier ne change pas
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Le code de l'API et le modèle entraîné
COPY app/ app/
COPY models/ models/

EXPOSE 8000

# Render fournit le port dans la variable PORT ; en local on utilise 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
