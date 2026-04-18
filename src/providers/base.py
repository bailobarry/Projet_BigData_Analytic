"""
Classe abstraite ``LLMProvider`` – contrat pour tous les fournisseurs LLM.

Tout nouveau provider (API, local, …) doit hériter de ``LLMProvider``
et implémenter ``generate()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.models.config import GenerationConfig


class LLMProvider(ABC):
    """Interface commune pour interroger un modèle de langage."""

    # ── Propriétés ──────────────────────────────────────────────────────

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Nom lisible du provider (ex: 'groq', 'ollama')."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Identifiant du modèle utilisé (ex: 'llama-3.3-70b-versatile')."""
        ...

    # ── Génération ──────────────────────────────────────────────────────

    @abstractmethod
    def generate(
        self,
        prompt: str,
        generation: "GenerationConfig",
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Génère une réponse textuelle pour le *prompt* donné.

        Parameters
        ----------
        prompt : str
            Le texte de la question (issu du fichier JSONL).
        generation : GenerationConfig
            Paramètres de génération (température, max_tokens, …).
        system_prompt : str | None
            Consigne système optionnelle (``None`` pour la baseline).

        Returns
        -------
        str
            Le texte de la réponse du LLM.
        """
        ...

