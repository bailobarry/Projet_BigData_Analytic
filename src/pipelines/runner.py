"""
Pipeline principal – exécute un *run* complet.

Workflow :
1. Charge la configuration (``RunConfig``)
2. Sauvegarde la config dans ``configs/runs/`` et ``data/output/{run_id}/``
3. Instancie le provider LLM via la factory
4. Itère sur chaque fichier JSONL d'entrée
5. Pour chaque prompt, interroge le LLM et écrit la réponse en streaming
6. Produit un résumé JSON du run (``run_summary.json``)

Les erreurs (timeouts, quotas, …) sont journalisées mais ne bloquent
pas le pipeline : un champ ``answer`` avec ``"ERROR: ..."`` est écrit.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import jsonlines

from src.models.config import RunConfig
from src.models.schemas import PromptItem, ResultItem
from src.pipelines.logs import setup_logging
from src.promptings.system_prompt import apply_prompt_template, get_system_prompt
from src.providers import create_provider

logger = logging.getLogger("pipeline")


# ── Callback de progression (pour Lot B – UI) ──────────────────────────────

ProgressCallback = Optional[Callable[[str, int, int, int, int], None]]
"""
Signature : callback(file_name, file_index, total_files, prompt_index, total_prompts)
"""


# ── Fonctions utilitaires ───────────────────────────────────────────────────


def _load_prompts(filepath: str) -> list[PromptItem]:
    """Charge les prompts depuis un fichier JSONL."""
    items: list[PromptItem] = []
    with jsonlines.open(filepath, mode="r") as reader:
        for obj in reader:
            items.append(PromptItem(**obj))
    return items


def _already_processed_ids(output_file: Path) -> set[str]:
    """
    Retourne les IDs déjà traités dans le fichier de sortie (reprise).
    Permet de reprendre un run interrompu sans tout recommencer.
    """
    done: set[str] = set()
    if output_file.exists():
        with jsonlines.open(str(output_file), mode="r") as reader:
            for obj in reader:
                done.add(obj.get("id", ""))
    return done


# ── Pipeline principal ──────────────────────────────────────────────────────


def run_pipeline(
    config: RunConfig,
    progress_cb: ProgressCallback = None,
) -> dict:
    """
    Exécute un run complet à partir de la configuration donnée.

    Parameters
    ----------
    config : RunConfig
        Configuration complète du run.
    progress_cb : callable | None
        Callback optionnel de progression (pour le Lot B – UI).

    Returns
    -------
    dict
        Résumé du run (nombre de prompts, erreurs, durée, …).
    """
    # ── 1. Initialisation ───────────────────────────────────────────────
    run_logger = setup_logging(config.run_id, config.pipeline.output_dir)
    run_logger.info("═" * 60)
    run_logger.info("DÉBUT DU RUN : %s", config.run_id)
    run_logger.info("Description  : %s", config.description or "(baseline)")
    run_logger.info("Provider     : %s / %s", config.provider.type, config.provider.model)
    run_logger.info("Température  : %s  |  Seed : %s", config.generation.temperature, config.generation.seed)
    run_logger.info("═" * 60)

    # ── 2. Sauvegarde de la configuration ───────────────────────────────
    output_dir = config.output_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir)

    # Copie dans configs/runs/ pour accès rapide
    runs_dir = Path("configs/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    config.save(runs_dir / config.run_id)

    run_logger.info("Configuration sauvegardée dans %s", output_dir / "config.json")

    # ── 3. Instanciation du provider ────────────────────────────────────
    provider = create_provider(config)
    run_logger.info("Provider instancié : %s (%s)", provider.provider_name, provider.model_id)


    # ── 4. Itération sur les fichiers ───────────────────────────────────
    input_files = config.input_files()
    total_files = len(input_files)
    total_prompts_processed = 0
    total_errors = 0
    start_time = time.time()

    for file_idx, input_file in enumerate(input_files, start=1):
        input_path = Path(input_file)
        if not input_path.exists():
            run_logger.warning("Fichier introuvable, ignoré : %s", input_file)
            continue

        filename = input_path.name
        # On récupère les 2 premiers caractères 
        current_lang = filename[:2] 
        system_prompt = get_system_prompt(config.pipeline.system_prompt, lang=current_lang)
        if system_prompt:
            run_logger.info("System prompt actif : '%s…'", system_prompt[:80])
        else:
            run_logger.info("Pas de system prompt (mode baseline vanilla)")
        output_file = output_dir / filename
        run_logger.info("─" * 40)
        run_logger.info("[%d/%d] Traitement de %s", file_idx, total_files, filename)

        # Charger les prompts
        prompts = _load_prompts(input_file)
        total_in_file = len(prompts)
        run_logger.info("  %d prompts chargés", total_in_file)

        # IDs déjà traités (reprise)
        done_ids = _already_processed_ids(output_file)
        if done_ids:
            run_logger.info("  %d prompts déjà traités (reprise)", len(done_ids))

        # Ouvrir le fichier de sortie en mode append
        with jsonlines.open(str(output_file), mode="a") as writer:
            for prompt_idx, item in enumerate(prompts, start=1):
                # Reprise : sauter les prompts déjà traités
                if item.id in done_ids:
                    continue

                # Appliquer le template de prompt (baseline = aucun changement)
                final_prompt = apply_prompt_template(
                    item.prompt, config.pipeline.prompt_template
                )

                # Interroger le LLM
                try:
                    answer = provider.generate(
                        prompt=final_prompt,
                        generation=config.generation,
                        system_prompt=system_prompt,
                    )
                except Exception as exc:
                    answer = f"ERROR: {type(exc).__name__}: {exc}"
                    total_errors += 1
                    run_logger.error(
                        "  ERREUR prompt id=%s : %s", item.id, answer
                    )

                # Écrire le résultat
                result = ResultItem(id=item.id, prompt=item.prompt, answer=answer)
                writer.write(result.model_dump())
                total_prompts_processed += 1

                # Log de progression
                if prompt_idx % 50 == 0 or prompt_idx == total_in_file:
                    run_logger.info(
                        "  Progression : %d/%d  (erreurs: %d)",
                        prompt_idx,
                        total_in_file,
                        total_errors,
                    )

                # Callback de progression (pour le Lot B)
                if progress_cb:
                    progress_cb(filename, file_idx, total_files, prompt_idx, total_in_file)

                # Rate limiting
                if config.pipeline.request_delay > 0:
                    time.sleep(config.pipeline.request_delay)

    # ── 6. Résumé du run ────────────────────────────────────────────────
    elapsed = time.time() - start_time
    summary = {
        "run_id": config.run_id,
        "provider": config.provider.type,
        "model": config.provider.model,
        "total_prompts": total_prompts_processed,
        "total_errors": total_errors,
        "duration_seconds": round(elapsed, 2),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }

    # Sauvegarder le résumé
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    run_logger.info("═" * 60)
    run_logger.info("FIN DU RUN : %s", config.run_id)
    run_logger.info("  Prompts traités : %d", total_prompts_processed)
    run_logger.info("  Erreurs         : %d", total_errors)
    run_logger.info("  Durée           : %.1f s", elapsed)
    run_logger.info("  Résultats dans  : %s", output_dir)
    run_logger.info("═" * 60)

    return summary

