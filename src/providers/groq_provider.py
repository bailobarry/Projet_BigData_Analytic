"""Provider Groq (OpenAI-compatible) simple avec une seule clé API.

Variable d'environnement requise:
- GROQ_API_KEY: clé API unique
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import openai as openai_pkg
from openai import RateLimitError

from src.models.config import GenerationConfig, RunConfig
from src.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """Provider Groq simple avec une seule clé API."""

    def __init__(self, config: RunConfig) -> None:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "Clé API Groq manquante. Configurez GROQ_API_KEY dans votre fichier .env"
            )

        self._model = config.provider.model
        self._client = openai_pkg.OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self._last_request_time = 0.0
        self._min_interval = 2.0  # Délai minimum entre 2 requêtes (30 req/min = 2s)

        logger.info("GroqProvider initialisé : model=%s", self._model)

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_id(self) -> str:
        return self._model

    def generate(
        self,
        prompt: str,
        generation: GenerationConfig,
        system_prompt: Optional[str] = None,
    ) -> str:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": generation.temperature,
            "max_tokens": generation.max_tokens,
            "top_p": generation.top_p,
        }

        # Respect du rate limit : attendre si nécessaire
        now = time.monotonic()
        time_since_last = now - self._last_request_time
        if time_since_last < self._min_interval:
            time.sleep(self._min_interval - time_since_last)

        max_attempts = 3
        last_error: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                self._last_request_time = time.monotonic()
                return (response.choices[0].message.content or "").strip()

            except RateLimitError as exc:
                last_error = exc
                wait_time = min(60.0, 5.0 * attempt)
                logger.warning(
                    "Rate limit atteint (tentative %d/%d), attente de %.1fs",
                    attempt, max_attempts, wait_time
                )
                if attempt < max_attempts:
                    time.sleep(wait_time)
                continue

            except Exception as exc:
                last_error = exc
                logger.error("Erreur API Groq (tentative %d/%d): %s", attempt, max_attempts, exc)
                if attempt < max_attempts:
                    time.sleep(2.0)
                continue

        if last_error:
            raise last_error
        raise RuntimeError("Génération Groq échouée après toutes les tentatives.")
