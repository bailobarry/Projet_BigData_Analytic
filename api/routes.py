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

_run_status:      dict[str, dict] = {}
_analysis_status: dict[str, dict] = {}
_cancel_flags:    dict[str, bool] = {}   # clés : run_id ou "analysis_<run_id>"

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


class AnalysisRequest(BaseModel):
    methods: list[str] = ["quantitative", "semantic", "qualitative", "llm_judge"]
    sample_size: int = 10
    run_specific_id: Optional[str] = None


class CompareRequest(BaseModel):
    run_id_a: str
    run_id_b: str
    methods: list[str] = ["quantitative", "semantic"]
    sample_size: int = 10

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

# ── Tâche d'arrière-plan – pipeline ─────────────────────────────────────────

def _execute_run(config: RunConfig) -> None:
    """Exécute le pipeline en arrière-plan."""
    # Vérification cancel avant démarrage (run mis en file d'attente puis annulé)
    if _cancel_flags.get(config.run_id):
        _run_status[config.run_id] = {"status": "cancelled"}
        return
    _run_status[config.run_id] = {"status": "running"}
    try:
        summary = run_pipeline(config)
        if _cancel_flags.get(config.run_id):
            _run_status[config.run_id] = {"status": "cancelled"}
        else:
            _run_status[config.run_id] = {"status": "completed", "summary": summary}
    except Exception as exc:
        _run_status[config.run_id] = {"status": "failed", "error": str(exc)}


# ── Tâche d'arrière-plan – analyse ───────────────────────────────────────────

def _execute_analysis(run_id: str, request: AnalysisRequest) -> None:
    """Exécute les analyses (quantitative / sémantique / LLM Judge) en arrière-plan."""
    import json as _json
    from pathlib import Path as _P

    flag_key = f"analysis_{run_id}"
    _analysis_status[run_id] = {"status": "running", "steps_done": [], "current": ""}

    def _cancelled() -> bool:
        return _cancel_flags.get(flag_key, False)

    def _mark_cancelled() -> None:
        _analysis_status[run_id] = {
            "status": "cancelled",
            "steps_done": _analysis_status[run_id].get("steps_done", []),
        }

    try:
        results: dict = {}
        # Références aux per_prompt pour que le qualitative en bénéficie si semantic tourne aussi
        _div_per_prompt = None
        _rob_per_prompt = None

        # ── 1. Quantitatif ────────────────────────────────────────────────
        if "quantitative" in request.methods:
            if _cancelled():
                _mark_cancelled(); return
            _analysis_status[run_id]["current"] = "quantitative"
            from src.analysis.quantitative import generate_report as _quant
            report = _quant(run_id, save=True)
            results["quantitative"] = report
            _analysis_status[run_id]["steps_done"].append("quantitative")

        # ── 2. Sémantique – Diversité ─────────────────────────────────────
        if "semantic" in request.methods:
            if _cancelled():
                _mark_cancelled(); return
            _analysis_status[run_id]["current"] = "semantic_diversity"
            from src.analysis.semantic import diversity_score as _div
            div = _div(run_id, sample_size=request.sample_size)
            _div_per_prompt = div.get("per_prompt")
            div_save = {k: v for k, v in div.items() if k != "per_prompt"}
            (_P("data/output") / run_id / "analysis_diversity.json").write_text(
                _json.dumps(div_save, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            results["diversity"] = div_save
            _analysis_status[run_id]["steps_done"].append("semantic_diversity")

            # ── 2b. Sémantique – Robustesse ───────────────────────────────
            if _cancelled():
                _mark_cancelled(); return
            rob_run = request.run_specific_id or run_id
            _analysis_status[run_id]["current"] = "semantic_robustness"
            try:
                from src.analysis.semantic import robustness_score as _rob
                rob = _rob(rob_run, sample_size=request.sample_size)
                _rob_per_prompt = rob.get("per_prompt")
                rob_save = {k: v for k, v in rob.items() if k != "per_prompt"}
                (_P("data/output") / rob_run / "analysis_robustness.json").write_text(
                    _json.dumps(rob_save, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                results["robustness"] = rob_save
                results["robustness_run_id"] = rob_run
                _analysis_status[run_id]["steps_done"].append("semantic_robustness")
            except ValueError as exc:
                results["robustness_error"] = str(exc)

        # ── 3. Analyse qualitative (cas extrêmes + typologies) ────────────
        if "qualitative" in request.methods:
            if _cancelled():
                _mark_cancelled(); return
            _analysis_status[run_id]["current"] = "qualitative"
            try:
                from src.analysis.qualitative import generate_qualitative_report as _qual
                qual_report = _qual(
                    run_id,
                    diversity_per_prompt=_div_per_prompt,
                    robustness_per_prompt=_rob_per_prompt,
                    save=True,
                )
                results["qualitative"] = qual_report
                _analysis_status[run_id]["steps_done"].append("qualitative")
            except Exception as exc_q:
                results["qualitative_error"] = str(exc_q)

        # ── 5. LLM Judge ─────────────────────────────────────────────────
        if "llm_judge" in request.methods:
            if _cancelled():
                _mark_cancelled(); return
            _analysis_status[run_id]["current"] = "llm_judge_diversity"
            from src.analysis.llm_judge import LLMJudge, evaluate_diversity, evaluate_robustness
            judge = LLMJudge()
            div_r = evaluate_diversity(run_id, sample_size=request.sample_size, judge=judge)
            results["llm_judge_diversity"] = div_r
            _analysis_status[run_id]["steps_done"].append("llm_judge_diversity")

            if _cancelled():
                _mark_cancelled(); return
            rob_run = request.run_specific_id or run_id
            _analysis_status[run_id]["current"] = "llm_judge_robustness"
            try:
                rob_r = evaluate_robustness(rob_run, sample_size=request.sample_size, judge=judge)
                results["llm_judge_robustness"] = rob_r
                _analysis_status[run_id]["steps_done"].append("llm_judge_robustness")
            except Exception as exc:
                results["llm_judge_robustness_error"] = str(exc)

        _analysis_status[run_id] = {
            "status": "completed",
            "steps_done": _analysis_status[run_id]["steps_done"],
            "results": results,
        }

    except Exception as exc:
        _analysis_status[run_id] = {
            "status": "failed",
            "error": str(exc),
            "steps_done": _analysis_status.get(run_id, {}).get("steps_done", []),
        }

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
    """Liste tous les runs passés (depuis configs/runs/) avec leur statut."""
    runs_dir = Path("configs/runs")
    runs: list[dict] = []
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir(), reverse=True):
            config_file = run_dir / "config.json" if run_dir.is_dir() else None
            if config_file and config_file.exists():
                data = json.loads(config_file.read_text(encoding="utf-8"))
                rid = data.get("run_id", "")
                output_dir = Path("data/output") / rid
                summary_file = output_dir / "run_summary.json"
                summary = None
                if summary_file.exists():
                    summary = json.loads(summary_file.read_text(encoding="utf-8"))

                # Statut
                if summary:
                    status = "completed"
                elif rid in _run_status:
                    status = _run_status[rid].get("status", "running")
                else:
                    status = "interrupted"

                # Compte de prompts déjà traités
                prompts_done = 0
                if output_dir.exists():
                    for f in output_dir.glob("*.jsonl"):
                        try:
                            prompts_done += sum(1 for _ in open(f, encoding="utf-8"))
                        except Exception:
                            pass

                runs.append({
                    "run_id": rid,
                    "description": data.get("description", ""),
                    "provider": data.get("provider", {}).get("type"),
                    "model": data.get("provider", {}).get("model"),
                    "languages": data.get("pipeline", {}).get("languages", []),
                    "dataset_types": data.get("pipeline", {}).get("dataset_types", []),
                    "created_at": data.get("created_at"),
                    "status": status,
                    "prompts_done": prompts_done,
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

            if status in ("completed", "failed", "cancelled"):
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

@router.post("/runs/{run_id}/resume", response_model=RunResponse)
async def resume_run(run_id: str, background_tasks: BackgroundTasks):
    """Reprend un run interrompu depuis le point où il s'était arrêté."""
    # Chercher la config sauvegardée
    config_file = Path("data/output") / run_id / "config.json"
    if not config_file.exists():
        config_file = Path("configs/runs") / run_id / "config.json"
    if not config_file.exists():
        raise HTTPException(status_code=404, detail=f"Configuration du run '{run_id}' introuvable.")

    data = json.loads(config_file.read_text(encoding="utf-8"))
    config = RunConfig(**data)

    _run_status[run_id] = {"status": "queued"}
    background_tasks.add_task(_execute_run, config)
    return RunResponse(
        run_id=run_id,
        status="queued",
        message=f"Run '{run_id}' repris depuis le point d'arrêt.",
    )


@router.post("/runs/{run_id}/analyse")
async def start_analysis(run_id: str, request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Lance les analyses (quantitative / sémantique / LLM Judge) en arrière-plan."""
    output_dir = Path("data/output") / run_id
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' introuvable.")
    _analysis_status[run_id] = {"status": "queued", "steps_done": [], "current": ""}
    background_tasks.add_task(_execute_analysis, run_id, request)
    return {"run_id": run_id, "status": "queued", "methods": request.methods}


@router.get("/runs/{run_id}/analyse/stream")
async def stream_analysis(run_id: str):
    """Stream SSE de la progression de l'analyse."""

    async def event_generator() -> AsyncGenerator[str, None]:
        for _ in range(600):  # timeout 10 min max
            info = _analysis_status.get(run_id, {"status": "waiting"})
            status = info.get("status", "waiting")
            payload = {
                "status": status,
                "current": info.get("current", ""),
                "steps_done": info.get("steps_done", []),
                "error": info.get("error", ""),
            }
            if status in ("completed", "cancelled"):
                payload["results"] = info.get("results", {})
            yield _sse(payload)
            if status in ("completed", "failed", "cancelled"):
                return
            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/runs/{run_id}/analyse/results")
async def get_analysis_results(run_id: str):
    """Retourne les résultats d'analyse sauvegardés pour un run."""
    output_dir = Path("data/output") / run_id
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' introuvable.")
    results = {}
    for f in sorted(output_dir.glob("analysis_*.json")):
        results[f.name] = json.loads(f.read_text(encoding="utf-8"))
    return results

@router.get("/runs/{run_id}/files")
async def list_run_files(run_id: str):
    """Liste les fichiers de résultats disponibles pour un run."""
    output_dir = Path("data/output") / run_id
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' introuvable.")
    files = [
        f.name for f in output_dir.iterdir()
        if f.is_file() and (f.suffix == ".jsonl" or f.name == "submission_metadata.json")
    ]
    return {"files": sorted(files)}


@router.get("/runs/{run_id}/results/{filename}")
async def get_run_results(run_id: str, filename: str):
    """Retourne le contenu brut d'un fichier de résultats."""
    result_file = Path("data/output") / run_id / filename
    if not result_file.exists():
        raise HTTPException(status_code=404, detail=f"Fichier '{filename}' introuvable.")
    return FileResponse(path=result_file, filename=filename, media_type="application/octet-stream")


# ── Endpoint : comparaison de deux runs ──────────────────────────────────────

@router.post("/runs/compare")
async def compare_two_runs(request: CompareRequest):
    """
    Compare deux runs (baseline vs variante) sur les métriques quantitatives et/ou sémantiques.

    Retourne les deltas de longueur, taux d'erreurs, diversité, robustesse et score combiné.
    """
    output_dir = "data/output"

    for rid in [request.run_id_a, request.run_id_b]:
        if not (Path(output_dir) / rid).exists():
            raise HTTPException(status_code=404, detail=f"Run '{rid}' introuvable dans data/output.")

    result: dict = {
        "run_a": request.run_id_a,
        "run_b": request.run_id_b,
    }

    # ── Quantitatif ──────────────────────────────────────────────────────────
    if "quantitative" in request.methods:
        try:
            from src.analysis.quantitative import compare_runs as _cmp_quant
            result["quantitative"] = _cmp_quant(
                request.run_id_a, request.run_id_b, output_dir
            )
        except Exception as exc:
            result["quantitative_error"] = str(exc)

    # ── Sémantique ────────────────────────────────────────────────────────────
    if "semantic" in request.methods:
        try:
            from src.analysis.semantic import compare_runs_semantic as _cmp_sem
            result["semantic"] = _cmp_sem(
                request.run_id_a,
                request.run_id_b,
                output_dir,
                sample_size=request.sample_size,
            )
        except Exception as exc:
            result["semantic_error"] = str(exc)

    return result

# ── Endpoints : config ────────────────────────────────────────────────────────

@router.get("/providers")
async def list_providers():
    """Retourne les providers, modèles et autres éléments de configuration."""
    return json.loads((CONFIGS_DIR / "providers.json").read_text(encoding="utf-8"))


# ── Endpoints : annulation ────────────────────────────────────────────────────

@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    """Annule une expérience en cours (queued ou running)."""
    _cancel_flags[run_id] = True
    if run_id in _run_status:
        if _run_status[run_id].get("status") in ("running", "queued"):
            _run_status[run_id]["status"] = "cancelled"
    return {"run_id": run_id, "status": "cancelled"}


@router.post("/runs/{run_id}/analyse/cancel")
async def cancel_analysis(run_id: str):
    """Annule une analyse en cours."""
    _cancel_flags[f"analysis_{run_id}"] = True
    if run_id in _analysis_status:
        if _analysis_status[run_id].get("status") in ("running", "queued"):
            _analysis_status[run_id]["status"] = "cancelled"
    return {"run_id": run_id, "status": "cancelled"}

