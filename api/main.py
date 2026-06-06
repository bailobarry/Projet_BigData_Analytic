"""
API pour piloter des runs LLM sur les données du challenge ELOQUENT @ CLEF 2026.
Lancement :
    uvicorn api.main:app --reload --port 8000
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

# Charger les variables d'environnement depuis .env
load_dotenv()

app = FastAPI(
    title="ELOQUENT – Cultural Robustness & Diversity",
    description=(
        "API pour piloter des runs LLM sur les données du challenge "
        "ELOQUENT @ CLEF 2026."
    ),
    version="0.1.0",
)

# Autoriser les requêtes depuis le frontend Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "ELOQUENT Cultural Robustness & Diversity"}

