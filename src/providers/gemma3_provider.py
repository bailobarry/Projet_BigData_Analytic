import logging
from typing import Optional

import ollama as _ollama_pkg

from src.models.config import GenerationConfig, RunConfig
from src.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class Gemma3Provider(LLMProvider):

    def __init__(self, config: RunConfig) -> None:
        prov = config.provider
        self._model = prov.model
        self._host = prov.ollama_host or "http://localhost:11434"

        # Client Ollama
        self._client = _ollama_pkg.Client(host=self._host)

        logger.info(
            "Gemma3Provider initialisé : model=%s  host=%s",
            self._model,
            self._host,
        )

    # Propriétés

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_id(self) -> str:
        return self._model

    # Génération

    def generate(
        self,
        prompt: str,
        generation: GenerationConfig,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Interroge Gemma 3 via Ollama et retourne la réponse texte."""

        # Construction des messages (session indépendante)
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Options de génération Ollama
        options: dict = {
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "num_predict": generation.max_tokens,
        }
        if generation.seed is not None:
            options["seed"] = generation.seed

        response = self._client.chat(
            model=self._model,
            messages=messages,
            options=options,
        )

        # ollama ≥ 0.4 retourne un objet ChatResponse
        if hasattr(response, "message"):
            content = response.message.content
        else:
            content = response.get("message", {}).get("content", "")
        return (content or "").strip()

