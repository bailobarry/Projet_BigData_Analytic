from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ["en", "fr", "de", "es", "it"]


def detect_available_datasets(
    run_id_a: str,
    run_id_b: str,
    output_dir: str = "data/output",
    languages: Optional[list[str]] = None,
) -> dict:
    """
    Détecte les types de datasets disponibles pour la comparaison de deux runs.
    
    Cette fonction analyse les fichiers présents dans les deux runs et détermine :
    - Quels types de datasets sont disponibles (unspecific, specific, ou les deux)
    - Quelles langues sont communes aux deux runs
    
    Parameters
    ----------
    run_id_a : str
        Premier run à comparer.
    run_id_b : str
        Second run à comparer.
    output_dir : str
        Répertoire contenant les résultats des runs.
    languages : list[str] | None
        Langues à vérifier (défaut: toutes les langues supportées).
    """
    if languages is None:
        languages = SUPPORTED_LANGUAGES
    
    run_a_path = Path(output_dir) / run_id_a
    run_b_path = Path(output_dir) / run_id_b
    
    # Vérifier quels fichiers existent pour chaque run
    unspecific_files_a = [lang for lang in languages if (run_a_path / f"{lang}_unspecific.jsonl").exists()]
    specific_files_a = [lang for lang in languages if (run_a_path / f"{lang}_specific.jsonl").exists()]
    
    unspecific_files_b = [lang for lang in languages if (run_b_path / f"{lang}_unspecific.jsonl").exists()]
    specific_files_b = [lang for lang in languages if (run_b_path / f"{lang}_specific.jsonl").exists()]
    
    # Déterminer les langues communes
    common_unspecific = sorted(set(unspecific_files_a) & set(unspecific_files_b))
    common_specific = sorted(set(specific_files_a) & set(specific_files_b))
    
    # Déterminer les types disponibles
    has_unspecific = len(common_unspecific) > 0
    has_specific = len(common_specific) > 0
    
    # Types disponibles par run
    run_a_types = []
    if unspecific_files_a:
        run_a_types.append("unspecific")
    if specific_files_a:
        run_a_types.append("specific")
    
    run_b_types = []
    if unspecific_files_b:
        run_b_types.append("unspecific")
    if specific_files_b:
        run_b_types.append("specific")
    
    result = {
        "has_unspecific": has_unspecific,
        "has_specific": has_specific,
        "unspecific_langs": common_unspecific,
        "specific_langs": common_specific,
        "run_a_types": run_a_types,
        "run_b_types": run_b_types,
        "summary": generate_summary_dataset(has_unspecific, has_specific, common_unspecific, common_specific),
    }
    
    logger.info(
        "Détection datasets : Run A (%s) | Run B (%s) | Communs : %s",
        ", ".join(run_a_types) if run_a_types else "aucun",
        ", ".join(run_b_types) if run_b_types else "aucun",
        f"unspecific ({len(common_unspecific)} langs), specific ({len(common_specific)} langs)"
    )
    
    return result


def generate_summary_dataset(
    has_unspecific: bool,
    has_specific: bool,
    unspecific_langs: list[str],
    specific_langs: list[str],
) -> str:
    """Génère un résumé textuel de la détection."""
    parts = []
    
    if has_unspecific:
        parts.append(f"unspecific ({len(unspecific_langs)} langues communes)")
    
    if has_specific:
        parts.append(f"specific ({len(specific_langs)} langues communes)")
    
    if not parts:
        return "Aucun type de dataset commun trouvé"
    
    return f"Datasets disponibles : {' et '.join(parts)}"


def get_dataset_type_for_llm_judge(
    run_id_a: str,
    run_id_b: str,
    output_dir: str = "data/output",
    languages: Optional[list[str]] = None,
) -> str:
    """
    Détermine automatiquement le type de dataset pour le LLM Judge.
    
    Stratégie simple :
    - Si un seul type est disponible : retourne ce type
    - Si les deux types sont disponibles : lève une exception pour forcer le choix manuel
    - Si aucun type n'est disponible : lève une exception
    
    Parameters
    ----------
    run_id_a : str
        Premier run.
    run_id_b : str
        Second run.
    output_dir : str
        Répertoire des résultats.
    languages : list[str] | None
        Langues à vérifier.
    
    Returns
    -------
    str
        Type de dataset ('unspecific' ou 'specific').
    """
    detection = detect_available_datasets(run_id_a, run_id_b, output_dir, languages)
    
    has_both = detection["has_unspecific"] and detection["has_specific"]
    
    if has_both:
        error_msg = (
            f"Les deux types de datasets sont disponibles pour les runs '{run_id_a}' et '{run_id_b}'. "
        )
        raise ValueError(error_msg)
    elif detection["has_unspecific"]:
        return "unspecific"
    elif detection["has_specific"]:
        return "specific"
    else:
        error_msg = (
            f"Impossible de comparer les runs '{run_id_a}' et '{run_id_b}' : "
            f"aucun type de dataset commun trouvé. "
            f"Run A : {', '.join(detection['run_a_types']) if detection['run_a_types'] else 'aucun fichier'} | "
            f"Run B : {', '.join(detection['run_b_types']) if detection['run_b_types'] else 'aucun fichier'}"
        )
        raise ValueError(error_msg)

