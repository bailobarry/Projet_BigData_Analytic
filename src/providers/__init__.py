"""
Providers LLM – factory et implémentations.

Deux providers disponibles :
- GeminiProvider       → Google Gemini 2.0 Flash (API cloud)
- MistralNemoProvider  → Mistral-Nemo 12B (local via Ollama)

Utilisation :
    from src.providers import create_provider
    provider = create_provider(run_config)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.config import RunConfig

from src.providers.base import LLMProvider  # noqa: F401


def create_provider(config: "RunConfig") -> LLMProvider:
    """Factory : instancie le bon provider à partir de la configuration."""
    provider_type = config.provider.type.lower()

    if provider_type == "gemini":
        from src.providers.gemini_provider import GeminiProvider
        return GeminiProvider(config)

    if provider_type == "ollama":
        from src.providers.mistral_nemo_provider import MistralNemoProvider
        return MistralNemoProvider(config)

    raise ValueError(
        f"Provider inconnu : '{provider_type}'. "
        f"Valeurs acceptées : 'gemini', 'ollama'."
    )
