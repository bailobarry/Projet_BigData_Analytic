"""
Providers LLM – factory et implémentations.

Deux providers disponibles :
- GroqProvider   → Groq Cloud (Llama 3.3 70B, API cloud, gratuit)
- Gemma3Provider → Google Gemma 3 12B (local via Ollama)

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

    if provider_type == "groq":
        from src.providers.groq_provider import GroqProvider
        return GroqProvider(config)

    if provider_type == "ollama":
        from src.providers.gemma3_provider import Gemma3Provider
        return Gemma3Provider(config)

    raise ValueError(
        f"Provider inconnu : '{provider_type}'. "
        f"Valeurs acceptées : 'groq', 'ollama'."
    )
