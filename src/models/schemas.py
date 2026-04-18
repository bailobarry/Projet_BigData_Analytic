"""
Schémas de données pour les fichiers JSONL du challenge ELOQUENT.

- PromptItem  : une entrée du fichier d'entrée  (id + prompt)
- ResultItem  : une entrée du fichier de sortie  (id + prompt + answer)
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PromptItem(BaseModel):
    """Représente une ligne d'un fichier JSONL d'entrée."""

    id: str = Field(..., description="Identifiant unique du prompt (ex: '1' ou '1-5')")
    prompt: str = Field(..., description="Texte de la question à envoyer au LLM")


class ResultItem(BaseModel):
    """Représente une ligne d'un fichier JSONL de sortie (avec la réponse)."""

    id: str = Field(..., description="Identifiant unique (doit correspondre au prompt)")
    prompt: str = Field(..., description="Texte original de la question")
    answer: str = Field(..., description="Réponse générée par le LLM")

