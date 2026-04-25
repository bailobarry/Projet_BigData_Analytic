"""
Configuration centralisée d'un run.

LLM retenus :
  - Google Gemini 2.0 Flash (API)
  - Mistral-Nemo 12B (Ollama local)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Sous-modèles ────────────────────────────────────────────────────────────


class ProviderConfig(BaseModel):
    """Paramètres de connexion au provider LLM.

    Deux providers supportés :
      - type='gemini'  → Google Gemini via API (nécessite GEMINI_API_KEY)
      - type='ollama'  → Mistral-Nemo 12B en local via Ollama
    """

    type: Literal["gemini", "ollama"] = Field(
        ...,
        description="Type de provider : 'gemini' (API Google) ou 'ollama' (Mistral-Nemo local)",
    )
    model: Literal["gemini-2.0-flash", "mistral-nemo"] = Field(
        ...,
        description="Nom du modèle : 'gemini-2.0-flash' ou 'mistral-nemo'",
    )
    ollama_host: str = Field(
        "http://localhost:11434",
        description="Hôte Ollama (utilisé uniquement quand type='ollama')",
    )


class GenerationConfig(BaseModel):
    """Paramètres de génération du LLM."""

    temperature: float = Field(0.0, description="Température (0 = déterministe)")
    max_tokens: int = Field(256, description="Nombre max de tokens de la réponse")
    top_p: float = Field(1.0, description="Top-p / nucleus sampling")
    seed: Optional[int] = Field(42, description="Seed pour reproductibilité")


class PipelineConfig(BaseModel):
    """Paramètres du pipeline d'exécution."""

    input_dir: str = Field("data/input", description="Répertoire des fichiers JSONL d'entrée")
    output_dir: str = Field("data/output", description="Répertoire racine de sortie")
    languages: List[str] = Field(
        default=["en", "fr", "de", "es", "it"],
        description="Codes langues à traiter",
    )
    dataset_types: List[str] = Field(
        default=["specific", "unspecific"],
        description="Types de dataset : 'specific' et/ou 'unspecific'",
    )
    request_delay: float = Field(
        2.1,
        description="Délai (sec) entre requêtes API pour respecter le rate-limit",
    )
    system_prompt: Optional[str] = Field(
        None,
        description="System prompt optionnel (None = baseline vanilla)",
    )
    prompt_template: Optional[str] = Field(
        None,
        description="Template de reformulation (None = prompt brut). "
                    "Utilisera {prompt} comme placeholder.",
    )


# ── Configuration principale ────────────────────────────────────────────────


class RunConfig(BaseModel):
    """Configuration complète d'un run"""

    run_id: str = Field(
        default_factory=lambda: f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}",
        description="Identifiant unique du run",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Date/heure de création (ISO 8601)",
    )
    description: str = Field(
        "",
        description="Description libre du run (ex: 'baseline vanilla')",
    )
    provider: ProviderConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    # ── Méthodes utilitaires ────────────────────────────────────────────

    def input_files(self) -> List[str]:
        """Retourne la liste des chemins de fichiers JSONL d'entrée."""
        files = []
        for lang in self.pipeline.languages:
            for ds_type in self.pipeline.dataset_types:
                filename = f"{lang}_{ds_type}.jsonl"
                filepath = str(Path(self.pipeline.input_dir) / filename)
                files.append(filepath)
        return files

    def output_path(self) -> Path:
        """Retourne le répertoire de sortie pour ce run."""
        return Path(self.pipeline.output_dir) / self.run_id

    def save(self, directory: Optional[Path] = None) -> Path:
        """Sauvegarde la configuration en JSON dans le répertoire donné."""
        if directory is None:
            directory = self.output_path()
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / "config.json"
        filepath.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return filepath

    @classmethod
    def from_file(cls, path: str | Path) -> "RunConfig":
        """Charge une configuration depuis un fichier JSON."""
        raw = Path(path).read_text(encoding="utf-8")
        return cls.model_validate_json(raw)


