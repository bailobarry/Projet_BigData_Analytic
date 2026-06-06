"""
Configuration du logging pour le pipeline.

Deux handlers :
- **Console** (StreamHandler)  : messages ≥ INFO,
- **Fichier**  (FileHandler)   : messages ≥ DEBUG, écrit dans
  ``data/output/{run_id}/run.log``.

Isolation par run :
  Chaque run obtient un logger nommé ``pipeline.<run_id>`` afin que
  plusieurs runs simultanés (ex: via l'API FastAPI) n'interfèrent pas
  entre eux et écrivent chacun dans leur propre fichier de log.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(run_id: str, output_dir: str = "data/output") -> logging.Logger:
    log_dir = Path(output_dir) / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"

    # Logger UNIQUE par run → pas de singleton partagé entre runs concurrents
    logger = logging.getLogger(f"pipeline.{run_id}")
    logger.setLevel(logging.DEBUG)

    # Ne pas propager vers le logger racine (évite les doublons en console)
    logger.propagate = False

    # Éviter les doublons si setup_logging est appelé deux fois avec le même run_id
    if logger.handlers:
        return logger

    # Format
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s pipeline – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Handler fichier (mode append pour permettre la reprise)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info("Logging initialisé – fichier : %s", log_file)
    return logger

