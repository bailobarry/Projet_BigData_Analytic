"""
Provider **Groq – Llama 3.3 70B Versatile** – via l'endpoint compatible OpenAI.

Llama 3.3 70B Versatile (Meta, via Groq) :
- 70 milliards de paramètres, excellent multilingue (EN, FR, DE, ES, IT)
- Inférence ultra-rapide grâce aux puces LPU de Groq
- Gratuit via Groq Cloud (30 req/min, 14 400 req/jour)

Clé API  : https://console.groq.com/keys
Endpoint : https://api.groq.com/openai/v1

Gestion des quotas :
- Délai fixe de 60/30 = 2s entre chaque requête pour respecter la limite de 30 req/min
- Retry automatique avec backoff exponentiel sur erreur 429 résiduelle
"""

from __future__ import annotations

import os
import logging
import time
from typing import Optional

import openai as _openai_pkg
from openai import RateLimitError

from src.models.config import GenerationConfig, RunConfig
from src.providers.base import LLMProvider

logger = logging.getLogger(__name__)

# Délai fixe entre chaque requête : 60s / 30 req = 2.0s
_REQUESTS_PER_MINUTE = 30
_DELAY = 60 / _REQUESTS_PER_MINUTE  # 2.0 secondes


class GroqProvider(LLMProvider):
    """
    Fournisseur LLM pour Groq (Llama 3.3 70B), via le SDK ``openai``
    et l'endpoint compatible OpenAI Chat Completions de Groq.

    Gestion du rate-limit :
      - Délai fixe de 2s après chaque requête (= 30 req/min max)
      - Retry avec backoff exponentiel si erreur 429 malgré le délai
    """

    def __init__(self, config: RunConfig) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Variable d'environnement 'GROQ_API_KEY' non définie. "
                "Ajoutez-la dans votre fichier .env. "
                "Obtenez une clé gratuite sur : https://console.groq.com/keys"
            )

        self._model = config.provider.model
        self._client = _openai_pkg.OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        logger.info(
            "GroqProvider initialisé : model=%s | délai=%gs entre requêtes (%d req/min)",
            self._model,
            _DELAY,
            _REQUESTS_PER_MINUTE,
        )

    # ── Propriétés ──────────────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return "groq"

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
        """
        Appelle l'API Groq et retourne la réponse texte.

        Après chaque requête réussie, attend _DELAY secondes pour
        respecter la limite de 30 req/min du free tier.
        En cas de 429, retry avec backoff exponentiel (sécurité).
        """

        # Construction des messages
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = dict(
            model=self._model,
            messages=messages,
            temperature=generation.temperature,
            max_tokens=generation.max_tokens,
            top_p=generation.top_p,
        )

        # Retry avec backoff exponentiel sur erreur 429
        retry_delay = 30.0  # délai initial si 429 : 30s → 60s → 120s
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                content: str = response.choices[0].message.content or ""

                # Délai fixe après chaque requête réussie : 60 / 30 = 2s
                time.sleep(_DELAY)

                return content.strip()

            except RateLimitError as exc:
                if attempt == max_retries:
                    raise
                logger.warning(
                    "Erreur 429 (tentative %d/%d). Attente de %.0fs… (erreur : %s)",
                    attempt,
                    max_retries,
                    retry_delay,
                    exc,
                )
                time.sleep(retry_delay)
                retry_delay *= 2  # backoff : 30s → 60s → 120s

        # Ce point est théoriquement inatteignable (la boucle raise au dernier essai)
        raise RuntimeError("generate() : toutes les tentatives ont échoué sans exception.")

