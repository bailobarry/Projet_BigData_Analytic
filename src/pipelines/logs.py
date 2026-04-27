"""
Configuration du logging pour le pipeline.

Deux handlers :
- **Console** (StreamHandler)  : messages ≥ INFO,
- **Fichier**  (FileHandler)   : messages ≥ DEBUG, écrit dans
  ``data/output/{run_id}/run.log``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(run_id: str, output_dir: str = "data/output") -> logging.Logger:
    """
    Configure et retourne le logger principal du run.

    Parameters
    ----------
    run_id : str
        Identifiant du run (utilisé pour le nom du fichier de log).
    output_dir : str
        Répertoire racine de sortie.

    Returns
    -------
    logging.Logger
        Logger configuré nommé ``pipeline``.
    """
    log_dir = Path(output_dir) / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"

    # Logger principal
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)

    # Éviter les doublons si la fonction est appelée plusieurs fois
    if logger.handlers:
        logger.handlers.clear()

    # Format
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Handler fichier
    file_handler = logging.FileHandler(log_file, encoding="utf-8", delay=False)
    file_handler.stream = open(log_file, "a", encoding="utf-8", buffering=1)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info("Logging initialisé – fichier : %s", log_file)
    return logger

