from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

from src.models.config import RunConfig


def build_metadata(run_id: str, team_name: str = "Master MIAGE Toulouse") -> dict:
    """
    Génère le submission_metadata.json à partir d'un run_id.
    """

    # Charger la config depuis data/output/{run_id}/config.json
    config_path = Path("data/output") / run_id / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config introuvable pour le run '{run_id}' : {config_path}"
        )

    config = RunConfig.from_file(config_path)

    languages_configs = {}

    if config.pipeline.system_prompt:
        from src.promptings.system_prompt import get_strategy_elements

        langs = (
            config.pipeline.languages
            if isinstance(config.pipeline.languages, list)
            else [config.pipeline.languages]
        )

        for lang in langs:
            strategy_pack = get_strategy_elements(
                config.pipeline.system_prompt, lang=lang
            )
            languages_configs[lang] = {
                "system_prompt": strategy_pack.get("system"),
                "prompt_prefix": strategy_pack.get("prefix"),
                "prompt_suffix": strategy_pack.get("suffix"),
            }

    return {
        "team": team_name,
        "system": "ELOQUENT-Cultural-Pipeline",
        "model": config.provider.model,
        "submissionid": config.run_id,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "label": "eloquent-2026-cultural",
        "languages": config.pipeline.languages,
        "modifications": {
            "templates_by_language": languages_configs,
            "generation_params": {
                "do_sample": config.generation.temperature > 0.0,
                "temperature": config.generation.temperature,
                "max_new_tokens": config.generation.max_tokens,
                "top_p": config.generation.top_p,
                "seed": config.generation.seed,
            },
            "notes": config.description,
        },
    }


def export_submission(
    run_id: str,
    team_name: str = "Master MIAGE Toulouse",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Génère le submission.zip complet prêt à soumettre.

    Structure produite :
        submission.zip
        ├── submission_metadata.json
        ├── en_specific.jsonl
        ├── en_unspecific.jsonl
        ├── fr_specific.jsonl
        └── ... .
    """

    # Charger la config
    config_path = Path("data/output") / run_id / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config introuvable pour le run '{run_id}' : {config_path}"
        )

    config = RunConfig.from_file(config_path)

    # Dossier de sortie du run
    run_output_dir = Path("data/output") / run_id

    # Collecter uniquement les fichiers JSONL du challenge
    jsonl_files = [
        f for f in run_output_dir.glob("*.jsonl")
        if f.name.endswith("_specific.jsonl")
        or f.name.endswith("_unspecific.jsonl")
    ]

    if not jsonl_files:
        raise FileNotFoundError(
            f"Aucun fichier JSONL trouvé dans {run_output_dir}. "
            "Le run a-t-il été exécuté ?"
        )

    # Générer le metadata
    metadata = build_metadata(run_id, team_name)

    # Sauvegarder aussi le metadata dans le dossier du run
    metadata_path = run_output_dir / "submission_metadata.json"

    metadata_json = json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False,
    )

    metadata_path.write_text(
        metadata_json,
        encoding="utf-8",
    )

    # Construire le ZIP en mémoire
    buffer = BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:

        # submission_metadata.json
        zf.writestr(
            "submission_metadata.json",
            json.dumps(metadata, indent=2, ensure_ascii=False),
        )

        # Ajouter les fichiers JSONL
        for jsonl_file in sorted(jsonl_files):
            zf.write(jsonl_file, arcname=jsonl_file.name)

    # Sauvegarder le zip
    if output_path is None:
        output_path = run_output_dir / "submission.zip"

    output_path.write_bytes(buffer.getvalue())

    return output_path