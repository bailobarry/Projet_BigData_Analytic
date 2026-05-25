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
from dotenv import load_dotenv

from src.models.config import RunConfig
from src.models.schemas import PromptItem, ResultItem
from src.pipelines.logs import setup_logging
from src.promptings.system_prompt import apply_full_reformulation, get_strategy_elements
from src.providers import create_provider

load_dotenv()
# Logger module (messages hors run – ex: imports). Chaque run utilise
# son propre logger isolé retourné par setup_logging(run_id).
logger = logging.getLogger(__name__)

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


def _repair_jsonl_file(output_file: Path) -> int:
    """
    Répare un fichier JSONL potentiellement corrompu après une interruption.

    Stratégie :
    - Lit chaque ligne brute du fichier.
    - Tente de la parser en JSON.
    - Conserve uniquement les lignes valides.
    - Réécrit le fichier avec seulement ces lignes valides.

    Returns
    -------
    int
        Nombre de lignes corrompues supprimées.
    """
    if not output_file.exists():
        return 0

    valid_lines: list[str] = []
    removed = 0

    with open(str(output_file), "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line.strip():
                continue  # Ignorer les lignes vides
            try:
                json.loads(line)
                valid_lines.append(line)
            except json.JSONDecodeError:
                removed += 1
                logger.warning(
                    "Ligne JSONL corrompue ignorée (reprise) : %s…", line[:80]
                )

    if removed > 0:
        # Réécrire le fichier avec uniquement les lignes valides
        with open(str(output_file), "w", encoding="utf-8") as f:
            for line in valid_lines:
                f.write(line + "\n")
        logger.info(
            "Fichier réparé : %d ligne(s) corrompue(s) supprimée(s) dans %s",
            removed,
            output_file.name,
        )

    return removed


def _already_processed_ids(output_file: Path) -> set[str]:
    """
    Retourne les IDs déjà traités dans le fichier de sortie (reprise).
    Répare automatiquement le fichier s'il contient des lignes corrompues
    (interruption brutale en cours d'écriture).
    """
    done: set[str] = set()
    if not output_file.exists():
        return done

    # Réparer d'abord les éventuelles lignes corrompues
    _repair_jsonl_file(output_file)

    # Lecture sécurisée ligne par ligne
    with open(str(output_file), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(obj.get("id", ""))
            except json.JSONDecodeError:
                pass  # Ne devrait pas arriver après la réparation, précaution défensive

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
        strategy_pack = get_strategy_elements(config.pipeline.system_prompt, lang=current_lang)
        
        system_prompt = strategy_pack["system"]
        prefix = strategy_pack["prefix"]
        suffix = strategy_pack["suffix"]
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

        # Nombre de prompts réellement à traiter dans ce run
        remaining = total_in_file - len(done_ids)
        run_logger.info("  %d prompts à traiter", remaining)

        # Ouvrir le fichier de sortie en mode append
        with jsonlines.open(str(output_file), mode="a") as writer:
            processed_in_file = 0  # compte uniquement les prompts traités dans CE run

            for item in prompts:
                # Reprise : sauter les prompts déjà traités
                if item.id in done_ids:
                    continue

                # Appliquer le template de prompt (baseline = aucun changement)
                final_prompt = apply_full_reformulation(item.prompt, prefix=prefix, suffix=suffix)

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
                processed_in_file += 1

                # Log de progression toutes les 50 requêtes ou à la fin
                if processed_in_file % 50 == 0 or processed_in_file == remaining:
                    run_logger.info(
                        "  Progression : %d/%d  (erreurs: %d)",
                        processed_in_file,
                        remaining,
                        total_errors,
                    )

                # Callback de progression (pour le Lot B)
                if progress_cb:
                    progress_cb(filename, file_idx, total_files, processed_in_file, remaining)

                # Note : le rate-limiting est géré directement par le provider
                # (fenêtre glissante 15 req/min pour Gemini, aucune limite pour Ollama)

    # ── 5. Génération du ZIP ────────────────────────────────────────────────
    run_logger.info(" Résultats dans : %s", output_dir)
    from src.export.challenge_export import export_submission
    submission_zip = export_submission(run_id=config.run_id, team_name="Master MIAGE Toulouse", )
    run_logger.info(" Submission ZIP généré : %s", submission_zip)

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

