"""
Routes de l'API – lancer des runs, lister les résultats, etc.

Ces endpoints seront consommés par l'interface Streamlit (Lot B).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.models.config import RunConfig
from src.pipelines.runner import run_pipeline

router = APIRouter(tags=["runs"])

# Stockage en mémoire de l'état des runs en cours
_run_status: dict[str, dict] = {}


# ── Schémas de requête/réponse ──────────────────────────────────────────────


class RunResponse(BaseModel):
    run_id: str
    status: str
    message: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    summary: Optional[dict] = None


# ── Tâche d'arrière-plan ────────────────────────────────────────────────────


def _execute_run(config: RunConfig) -> None:
    """Exécute le pipeline en arrière-plan."""
    _run_status[config.run_id] = {"status": "running"}
    try:
        summary = run_pipeline(config)
        _run_status[config.run_id] = {"status": "completed", "summary": summary}
    except Exception as exc:
        _run_status[config.run_id] = {"status": "failed", "error": str(exc)}


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.post("/runs", response_model=RunResponse)
async def create_run(config: RunConfig, background_tasks: BackgroundTasks):
    """Lance un nouveau run en arrière-plan."""
    _run_status[config.run_id] = {"status": "queued"}
    background_tasks.add_task(_execute_run, config)
    return RunResponse(
        run_id=config.run_id,
        status="queued",
        message=f"Run '{config.run_id}' lancé en arrière-plan.",
    )


@router.get("/runs", response_model=list[dict])
async def list_runs():
    """Liste tous les runs passés (depuis configs/runs/)."""
    runs_dir = Path("configs/runs")
    runs: list[dict] = []
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir(), reverse=True):
            config_file = run_dir / "config.json" if run_dir.is_dir() else None
            if config_file and config_file.exists():
                data = json.loads(config_file.read_text(encoding="utf-8"))
                summary_file = Path("data/output") / data.get("run_id", "") / "run_summary.json"
                summary = None
                if summary_file.exists():
                    summary = json.loads(summary_file.read_text(encoding="utf-8"))
                runs.append({
                    "run_id": data.get("run_id"),
                    "description": data.get("description", ""),
                    "provider": data.get("provider", {}).get("provider_label"),
                    "model": data.get("provider", {}).get("model"),
                    "created_at": data.get("created_at"),
                    "summary": summary,
                })
    return runs


@router.get("/runs/{run_id}/status", response_model=RunStatusResponse)
async def get_run_status(run_id: str):
    """Retourne le statut d'un run (en cours ou terminé)."""
    # Vérifier d'abord en mémoire (run en cours)
    if run_id in _run_status:
        info = _run_status[run_id]
        return RunStatusResponse(
            run_id=run_id,
            status=info.get("status", "unknown"),
            summary=info.get("summary"),
        )

    # Vérifier les résultats sur disque
    summary_file = Path("data/output") / run_id / "run_summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        return RunStatusResponse(run_id=run_id, status="completed", summary=summary)

    raise HTTPException(status_code=404, detail=f"Run '{run_id}' introuvable.")


@router.get("/runs/{run_id}/results/{filename}")
async def get_run_results(run_id: str, filename: str):
    """Retourne le contenu d'un fichier de résultats JSONL."""
    result_file = Path("data/output") / run_id / filename
    if not result_file.exists():
        raise HTTPException(status_code=404, detail=f"Fichier '{filename}' introuvable pour le run '{run_id}'.")

    results = []
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


@router.get("/providers")
async def list_providers():
    """Retourne les providers et modèles recommandés."""
    return {
        "providers": [
            {
                "type": "openai_compatible",
                "label": "google",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "api_key_env": "GEMINI_API_KEY",
                "models": ["gemini-2.0-flash"],
            },
            {
                "type": "ollama",
                "label": "ollama",
                "models": ["mistral-nemo"],
            },
        ],
        "languages": ["en", "fr", "de", "es", "ru"],
        "dataset_types": ["specific", "unspecific"],
    }

