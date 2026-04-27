"""
Routes de l'API – lancer des runs, lister les résultats, etc.

Ces endpoints seront consommés par l'interface Streamlit (Lot B).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from src.models.config import RunConfig
from src.pipelines.runner import run_pipeline

router = APIRouter(tags=["runs"])

_run_status: dict[str, dict] = {}

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

# ── Schémas ──────────────────────────────────────────────────────────────────

class RunResponse(BaseModel):
    run_id: str
    status: str
    message: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    summary: Optional[dict] = None

# ── Helpers SSE ───────────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    """Formate un dict en événement SSE valide."""
    return f"data: {json.dumps(data)}\n\n"


async def _wait_for_run(run_id: str) -> bool:
    """Attend que le run soit enregistré dans _run_status."""
    for _ in range(20):
        if run_id in _run_status:
            return True
        await asyncio.sleep(0.2)
    return False


async def _wait_for_log_file(log_file: Path) -> None:
    """Attend que le fichier de log soit créé sur disque."""
    for _ in range(50):
        if log_file.exists():
            return
        await asyncio.sleep(0.2)


def _read_new_logs(log_file: Path, last_size: int) -> tuple[list[str], int]:
    """Lit les lignes ajoutées au fichier de log depuis la dernière lecture."""
    if not log_file.exists():
        return [], last_size
    current_size = log_file.stat().st_size
    if current_size <= last_size:
        return [], last_size
    with open(log_file, "r", encoding="utf-8") as f:
        f.seek(last_size)
        new_logs = [l for l in f.read().splitlines() if l.strip()]
    return new_logs, current_size

# ── Tâche d'arrière-plan ─────────────────────────────────────────────────────

def _execute_run(config: RunConfig) -> None:
    """Exécute le pipeline en arrière-plan."""
    _run_status[config.run_id] = {"status": "running"}
    try:
        summary = run_pipeline(config)
        _run_status[config.run_id] = {"status": "completed", "summary": summary}
    except Exception as exc:
        _run_status[config.run_id] = {"status": "failed", "error": str(exc)}

# ── Endpoints : runs ──────────────────────────────────────────────────────────

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
                    "provider": data.get("provider", {}).get("type"),
                    "model": data.get("provider", {}).get("model"),
                    "created_at": data.get("created_at"),
                    "summary": summary,
                })
    return runs


@router.get("/runs/{run_id}/status", response_model=RunStatusResponse)
async def get_run_status(run_id: str):
    """Retourne le statut d'un run (en cours ou terminé)."""
    if run_id in _run_status:
        info = _run_status[run_id]
        return RunStatusResponse(
            run_id=run_id,
            status=info.get("status", "unknown"),
            summary=info.get("summary"),
        )
    summary_file = Path("data/output") / run_id / "run_summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        return RunStatusResponse(run_id=run_id, status="completed", summary=summary)
    raise HTTPException(status_code=404, detail=f"Run '{run_id}' introuvable.")


@router.get("/runs/{run_id}/stream")
async def stream_run_progress(run_id: str):
    """Stream SSE des logs d'un run."""

    async def event_generator() -> AsyncGenerator[str, None]:
        log_file = Path("data/output") / run_id / "run.log"
        last_size = 0

        if not await _wait_for_run(run_id):
            yield _sse({"status": "error", "detail": f"Run '{run_id}' introuvable."})
            return

        await _wait_for_log_file(log_file)

        while True:
            info = _run_status.get(run_id, {})
            status = info.get("status", "unknown")
            new_logs, last_size = _read_new_logs(log_file, last_size)

            if status in ("completed", "failed"):
                if new_logs:
                    yield _sse({"status": "running", "new_logs": new_logs})
                    await asyncio.sleep(0)
                yield _sse({"status": status, "summary": info.get("summary"), "error": info.get("error")})
                return

            yield _sse({"status": status, "new_logs": new_logs})
            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── Endpoints : fichiers ──────────────────────────────────────────────────────

@router.get("/runs/{run_id}/files")
async def list_run_files(run_id: str):
    """Liste les fichiers de résultats disponibles pour un run."""
    output_dir = Path("data/output") / run_id
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' introuvable.")
    files = [
        f.name for f in output_dir.iterdir()
        if f.is_file() and f.suffix in (".jsonl", ".json") and f.name != "config.json"
    ]
    return {"files": sorted(files)}


@router.get("/runs/{run_id}/results/{filename}")
async def get_run_results(run_id: str, filename: str):
    """Retourne le contenu brut d'un fichier de résultats."""
    result_file = Path("data/output") / run_id / filename
    if not result_file.exists():
        raise HTTPException(status_code=404, detail=f"Fichier '{filename}' introuvable.")
    return FileResponse(path=result_file, filename=filename, media_type="application/octet-stream")

# ── Endpoints : config ────────────────────────────────────────────────────────

@router.get("/providers")
async def list_providers():
    """Retourne les providers, modèles et autres éléments de configuration."""
    return json.loads((CONFIGS_DIR / "providers.json").read_text(encoding="utf-8"))
