"""
Provider **Google Gemini 2.0 Flash** – via l'endpoint compatible OpenAI.

Gemini 2.0 Flash (par Google) :
- Modèle très puissant, excellent multilingue (EN, FR, DE, ES, RU)
- Gratuit via Google AI Studio (15 req/min)

Clé API : https://aistudio.google.com/apikey
Endpoint : https://generativelanguage.googleapis.com/v1beta/openai/
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import openai as _openai_pkg

from src.models.config import GenerationConfig, RunConfig
from src.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """
    Fournisseur LLM pour Google Gemini, via le SDK ``openai``
    et l'endpoint compatible OpenAI Chat Completions de Google.
    """

    def __init__(self, config: RunConfig) -> None:
        # Récupère la clé API depuis la variable d'environnement GEMINI_API_KEY
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Variable d'environnement 'GEMINI_API_KEY' non définie. "
                "Ajoutez-la dans votre fichier .env."
            )

        self._model = config.provider.model
        self._client = _openai_pkg.OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

        logger.info("GeminiProvider initialisé : model=%s", self._model)

    # ── Propriétés ──────────────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_id(self) -> str:
        return self._model

    # ── Génération ──────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        generation: GenerationConfig,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Appelle l'API Google Gemini et retourne la réponse texte."""

        # Construction des messages (chaque question = session indépendante)
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Paramètres de la requête
        kwargs: dict = dict(
            model=self._model,
            messages=messages,
            temperature=generation.temperature,
            max_tokens=generation.max_tokens,
            top_p=generation.top_p,
        )

        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        return (content or "").strip()
