#!/usr/bin/env python
"""
run_baseline.py – Script CLI pour lancer un run depuis un fichier de config.

Usage :
    # Baseline Groq Llama 3.3 70B (API cloud, recommandé)
    python run_baseline.py

    # Baseline Gemma 3 12B via Ollama (local)
    python run_baseline.py --config configs/baseline_ollama.json

    # Seulement les fichiers unspecific (test rapide)
    python run_baseline.py --types unspecific

    # Seulement certaines langues
    python run_baseline.py --languages fr en
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from src.models.config import RunConfig
from src.pipelines.runner import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lancer un run ELOQUENT Cultural Robustness & Diversity",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_groq.json",
        help="Chemin vers le fichier de configuration JSON (défaut: configs/baseline_groq.json)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Langues à traiter (ex: --languages fr en). Défaut: toutes.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        choices=["specific", "unspecific"],
        help="Types de dataset (ex: --types unspecific). Défaut: tous.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identifiant personnalisé pour le run.",
    )
    args = parser.parse_args()

    # Charger les variables d'environnement depuis .env
    load_dotenv()

    # Charger la configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERREUR : Fichier de configuration introuvable : {config_path}")
        sys.exit(1)

    config = RunConfig.from_file(config_path)

    # Surcharger les langues/types si spécifiés
    if args.languages:
        config.pipeline.languages = args.languages
    if args.types:
        config.pipeline.dataset_types = args.types
    if args.run_id:
        config.run_id = args.run_id

    # Afficher la config
    print("-" * 60)
    print("ELOQUENT – Cultural Robustness & Diversity")
    print("-" * 60)
    print(f"  Config     : {config_path}")
    print(f"  Run ID     : {config.run_id}")
    print(f"  Provider   : {config.provider.type} / {config.provider.model}")
    print(f"  Langues    : {config.pipeline.languages}")
    print(f"  Datasets   : {config.pipeline.dataset_types}")
    print(f"  Temp       : {config.generation.temperature}")
    print(f"  Seed       : {config.generation.seed}")
    print(f"  Max tokens : {config.generation.max_tokens}")
    print("-" * 60)

    # Lancer le pipeline
    summary = run_pipeline(config)

    print("\nRun terminé !")
    print(f"   Prompts traités : {summary['total_prompts']}")
    print(f"   Erreurs         : {summary['total_errors']}")
    print(f"   Durée           : {summary['duration_seconds']}s")
    print(f"   Résultats dans  : data/output/{config.run_id}/")


if __name__ == "__main__":
    main()

