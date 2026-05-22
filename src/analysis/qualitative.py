"""
Analyse qualitative des résultats d'un run.

Ce module complète les analyses quantitative et sémantique en fournissant :

1. **Sélection de cas extrêmes** (select_extreme_cases)
   À partir des scores per_prompt issus de semantic.py, identifie les questions
   les plus intéressantes à examiner manuellement :
   - Cas très variables (forte diversité → le modèle s'adapte bien)
   - Cas incohérents (faible robustesse → le modèle est instable)
   - Cas problématiques (toutes langues confondues donnent la même réponse)

2. **Détection de non-conformité à la consigne** (detect_instruction_violations)
   Les prompts du challenge demandent tous "répondez en une seule phrase" /
   "Answer in one sentence". Cette fonction détecte les réponses qui ignorent
   la consigne (trop longues) ou sont suspicieusement courtes (généricité).

3. **Typologie d'erreurs** (classify_errors)
   Pour chaque réponse, attribue une ou plusieurs étiquettes :
   - "generic"       : réponse trop courte ou vague (< 5 mots utiles)
   - "non_compliant" : réponse de plus de 3 phrases (non-respect de la consigne)
   - "repetitive"    : réponse identique ou quasi-identique à celle d'une autre langue
   - "error"         : erreur pipeline (préfixe ERROR:)
   - "ok"            : réponse conforme

4. **Rapport qualitatif complet** (generate_qualitative_report)
   Combine sélection de cas extrêmes, typologies et statistiques d'erreurs
   dans un rapport JSON sauvegardé dans le dossier du run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import jsonlines

logger = logging.getLogger(__name__)

# ─── Constantes ──────────────────────────────────────────────────────────────

_DEFAULT_OUTPUT_DIR = "data/output"
_SUPPORTED_LANGUAGES = ["en", "fr", "de", "es", "it"]

# Seuils pour la classification
_GENERIC_THRESHOLD_WORDS = 4      # Moins de 4 mots → générique
_NON_COMPLIANT_SENTENCES = 3      # Plus de 3 phrases → non-respect de la consigne
_REPETITION_SIM_THRESHOLD = 0.97  # Similarité cosinus > 0.97 → répétition


# ─── Chargement des données ──────────────────────────────────────────────────


def _load_results_by_language(
    run_id: str,
    dataset_type: str,
    output_dir: str,
    languages: list[str],
) -> dict[str, dict[str, dict]]:
    """
    Charge les réponses d'un type de dataset pour toutes les langues.

    Returns
    -------
    dict[str, dict[str, dict]]
        { "fr": {"1": {"answer": "...", "prompt": "..."}}, "en": {...}, ... }
    """
    data: dict[str, dict[str, dict]] = {}
    run_path = Path(output_dir) / run_id

    for lang in languages:
        filepath = run_path / f"{lang}_{dataset_type}.jsonl"
        if not filepath.exists():
            continue
        lang_data: dict[str, dict] = {}
        with jsonlines.open(str(filepath), mode="r") as reader:
            for obj in reader:
                lang_data[obj["id"]] = {
                    "answer": obj.get("answer", ""),
                    "prompt": obj.get("prompt", ""),
                }
        data[lang] = lang_data
    return data


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _count_sentences(text: str) -> int:
    """Compte le nombre de phrases terminées par . ! ou ?"""
    import re
    return len([s for s in re.split(r"[.!?]+", text.strip()) if s.strip()])


def _count_words(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def _is_error(answer: str) -> bool:
    return answer.strip().startswith("ERROR:")


# ─── Sélection de cas extrêmes ───────────────────────────────────────────────


def select_extreme_cases(
    per_prompt_scores: dict[str, dict],
    top_n: int = 5,
    metric_label: str = "score",
) -> dict:
    """
    Sélectionne les cas extrêmes à partir des scores per_prompt de semantic.py.

    Utilisé après diversity_score() ou robustness_score() pour identifier
    les questions les plus intéressantes à analyser manuellement.

    Parameters
    ----------
    per_prompt_scores : dict[str, dict]
        Dictionnaire {prompt_id: {"score": float, ...}} issu de diversity_score
        ou robustness_score.
    top_n : int
        Nombre de cas à retenir dans chaque catégorie (défaut: 5).
    metric_label : str
        Nom de la métrique dans le dict (défaut: "score").

    Returns
    -------
    dict
        {
            "top_highest": [{"id": "42", "score": 0.89}, ...],  # très diverse / très robuste
            "top_lowest":  [{"id": "7",  "score": 0.03}, ...],  # peu diverse / peu robuste
            "median":      [{"id": "55", "score": 0.44}, ...]   # cas typiques
        }
    """
    scored = sorted(
        [{"id": k, "score": v.get(metric_label, 0.0)} for k, v in per_prompt_scores.items()],
        key=lambda x: x["score"],
    )

    if not scored:
        return {"top_highest": [], "top_lowest": [], "median": []}

    n = len(scored)
    mid_start = max(0, n // 2 - top_n // 2)

    return {
        "top_highest": list(reversed(scored[-top_n:])),  # les plus élevés
        "top_lowest":  scored[:top_n],                   # les plus bas
        "median":      scored[mid_start: mid_start + top_n],  # cas médians
    }


# ─── Détection de non-conformité à la consigne ───────────────────────────────


def detect_instruction_violations(
    run_id: str,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    languages: Optional[list[str]] = None,
    dataset_types: Optional[list[str]] = None,
) -> dict:
    """
    Détecte les réponses qui ne respectent pas la consigne "Répondez en une seule phrase".

    Toutes les questions du challenge demandent une réponse en une seule phrase.
    Cette fonction identifie :
    - Les réponses avec plus de 2 phrases (non_compliant)
    - Les réponses de moins de 5 mots (generic / trop vague)
    - Les erreurs pipeline

    Parameters
    ----------
    run_id : str
        Identifiant du run.
    output_dir : str
        Répertoire racine des résultats.
    languages : list[str] | None
        Langues à analyser (défaut : toutes).
    dataset_types : list[str] | None
        Types de dataset à analyser (défaut : ["unspecific", "specific"]).

    Returns
    -------
    dict
        {
            "total_responses": 500,
            "violations": {
                "non_compliant": 23,    # plus de 2 phrases
                "generic":       15,    # moins de 5 mots
                "error":          2
            },
            "violation_rate": 0.08,
            "by_file": {
                "fr_unspecific": {"non_compliant": 5, "generic": 3, "error": 0, "total": 101},
                ...
            },
            "examples": {
                "non_compliant": [{"file": "fr_unspecific", "id": "7", "answer": "..."}],
                "generic":   [...],
            }
        }
    """
    if languages is None:
        languages = _SUPPORTED_LANGUAGES
    if dataset_types is None:
        dataset_types = ["unspecific", "specific"]

    run_path = Path(output_dir) / run_id
    if not run_path.exists():
        raise FileNotFoundError(f"Run introuvable : {run_path}")

    total = 0
    violations: dict[str, int] = {"non_compliant": 0, "generic": 0, "error": 0}
    by_file: dict[str, dict] = {}
    examples: dict[str, list] = {"non_compliant": [], "generic": []}

    for dtype in dataset_types:
        for lang in languages:
            filepath = run_path / f"{lang}_{dtype}.jsonl"
            if not filepath.exists():
                continue

            file_key = f"{lang}_{dtype}"
            by_file[file_key] = {"non_compliant": 0, "generic": 0, "error": 0, "total": 0}

            with jsonlines.open(str(filepath), mode="r") as reader:
                for obj in reader:
                    answer = obj.get("answer", "")
                    total += 1
                    by_file[file_key]["total"] += 1

                    if _is_error(answer):
                        violations["error"] += 1
                        by_file[file_key]["error"] += 1
                    elif _count_words(answer) < _GENERIC_THRESHOLD_WORDS:
                        violations["generic"] += 1
                        by_file[file_key]["generic"] += 1
                        if len(examples["generic"]) < 5:
                            examples["generic"].append({
                                "file": file_key, "id": obj.get("id"), "answer": answer
                            })
                    elif _count_sentences(answer) > _NON_COMPLIANT_SENTENCES:
                        violations["non_compliant"] += 1
                        by_file[file_key]["non_compliant"] += 1
                        if len(examples["non_compliant"]) < 5:
                            examples["non_compliant"].append({
                                "file": file_key, "id": obj.get("id"),
                                "answer": answer[:300]
                            })

    total_violations = sum(violations.values())
    violation_rate = round(total_violations / total, 4) if total else 0.0

    return {
        "run_id": run_id,
        "total_responses": total,
        "violations": violations,
        "total_violations": total_violations,
        "violation_rate": violation_rate,
        "by_file": by_file,
        "examples": examples,
    }


# ─── Typologie d'erreurs ─────────────────────────────────────────────────────


def classify_errors(
    run_id: str,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    languages: Optional[list[str]] = None,
    dataset_types: Optional[list[str]] = None,
) -> dict:
    """
    Attribue une étiquette d'erreur à chaque réponse et produit une distribution.

    Étiquettes (non exclusives) :
    - ``"ok"``            : réponse conforme (une phrase, contenu substantiel)
    - ``"error"``         : erreur pipeline (préfixe ERROR:)
    - ``"generic"``       : réponse trop courte/vague (< 5 mots), généricité excessive
    - ``"non_compliant"`` : dépasse 2 phrases (non-respect de la consigne)
    - ``"empty"``         : réponse vide ou blanche

    Parameters
    ----------
    run_id : str
        Identifiant du run.
    output_dir : str
        Répertoire racine des résultats.
    languages : list[str] | None
    dataset_types : list[str] | None

    Returns
    -------
    dict
        {
            "run_id": "...",
            "total": 500,
            "distribution": {
                "ok": 450, "error": 5, "generic": 30, "non_compliant": 10, "empty": 5
            },
            "rates": {
                "ok": 0.900, "error": 0.010, "generic": 0.060, ...
            },
            "by_file": {
                "fr_unspecific": {"ok": 95, "generic": 4, ...},
                ...
            },
            "problematic_examples": [
                {
                    "id": "13", "file": "fr_unspecific",
                    "label": "generic",
                    "answer": "Sois honnête.",
                    "n_words": 2
                },
                ...
            ]
        }
    """
    if languages is None:
        languages = _SUPPORTED_LANGUAGES
    if dataset_types is None:
        dataset_types = ["unspecific", "specific"]

    run_path = Path(output_dir) / run_id
    if not run_path.exists():
        raise FileNotFoundError(f"Run introuvable : {run_path}")

    labels = ["ok", "error", "generic", "non_compliant", "empty"]
    distribution: dict[str, int] = {k: 0 for k in labels}
    by_file: dict[str, dict] = {}
    problematic: list[dict] = []
    total = 0

    for dtype in dataset_types:
        for lang in languages:
            filepath = run_path / f"{lang}_{dtype}.jsonl"
            if not filepath.exists():
                continue

            file_key = f"{lang}_{dtype}"
            by_file[file_key] = {k: 0 for k in labels}

            with jsonlines.open(str(filepath), mode="r") as reader:
                for obj in reader:
                    answer = obj.get("answer", "").strip()
                    total += 1

                    if not answer:
                        label = "empty"
                    elif _is_error(answer):
                        label = "error"
                    elif _count_words(answer) < _GENERIC_THRESHOLD_WORDS:
                        label = "generic"
                    elif _count_sentences(answer) > _NON_COMPLIANT_SENTENCES:
                        label = "non_compliant"
                    else:
                        label = "ok"

                    distribution[label] += 1
                    by_file[file_key][label] += 1

                    if label != "ok" and len(problematic) < 20:
                        problematic.append({
                            "id":      obj.get("id"),
                            "file":    file_key,
                            "label":   label,
                            "answer":  answer[:200],
                            "n_words": _count_words(answer),
                        })

    rates = {
        k: round(v / total, 4) if total else 0.0
        for k, v in distribution.items()
    }

    return {
        "run_id": run_id,
        "total": total,
        "distribution": distribution,
        "rates": rates,
        "by_file": by_file,
        "problematic_examples": problematic,
    }


# ─── Analyse par catégorie thématique ────────────────────────────────────────

# Mapping manuel des IDs (1–101) vers les catégories thématiques du challenge.
# Basé sur les questions en anglais (en_unspecific.jsonl).
# Catégories : food, social_life, work_education, family, social_norms
_CATEGORY_MAP: dict[str, list[int]] = {
    "food": [1, 10, 11, 24, 25, 86, 87],
    "family": [2, 3, 4, 5, 29, 83, 88],
    "social_life": [7, 8, 9, 12, 13, 14, 15, 23, 26, 27, 28, 30, 84, 95, 96, 101],
    "work_education": [6, 22, 85, 89, 90, 93, 94],
    "social_norms": [
        16, 17, 18, 19, 20, 21, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
        73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 91, 92, 97, 98, 99, 100,
    ],
}


def get_category(prompt_id: str) -> str:
    """Retourne la catégorie thématique d'une question à partir de son ID."""
    # Les IDs specific sont du type "1-5" → base = "1"
    base_id = int(prompt_id.split("-")[0])
    for category, ids in _CATEGORY_MAP.items():
        if base_id in ids:
            return category
    return "other"


def analyze_by_category(
    per_prompt_scores: dict[str, dict],
    metric_label: str = "score",
) -> dict:
    """
    Regroupe les scores per_prompt par catégorie thématique.

    Parameters
    ----------
    per_prompt_scores : dict[str, dict]
        Dictionnaire {prompt_id: {"score": float}} issu de diversity_score()
        ou robustness_score().
    metric_label : str
        Clé du score dans le dict (défaut: "score").

    Returns
    -------
    dict
        {
            "food":           {"avg": 0.23, "std": 0.07, "n": 7},
            "family":         {"avg": 0.18, "std": 0.05, "n": 8},
            "social_life":    {"avg": 0.31, "std": 0.10, "n": 16},
            "work_education": {"avg": 0.19, "std": 0.06, "n": 7},
            "social_norms":   {"avg": 0.25, "std": 0.09, "n": 64},
            "other":          {"avg": 0.20, "std": 0.08, "n": 3}
        }
    """
    import numpy as np

    categories: dict[str, list[float]] = {k: [] for k in _CATEGORY_MAP}
    categories["other"] = []

    for prompt_id, data in per_prompt_scores.items():
        score = data.get(metric_label, 0.0)
        cat = get_category(prompt_id)
        categories.setdefault(cat, []).append(score)

    result: dict[str, dict] = {}
    for cat, scores in categories.items():
        if scores:
            result[cat] = {
                "avg": round(float(np.mean(scores)), 4),
                "std": round(float(np.std(scores)), 4),
                "n":   len(scores),
                "min": round(float(np.min(scores)), 4),
                "max": round(float(np.max(scores)), 4),
            }
    return result


# ─── Rapport qualitatif complet ──────────────────────────────────────────────


def generate_qualitative_report(
    run_id: str,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    languages: Optional[list[str]] = None,
    diversity_per_prompt: Optional[dict] = None,
    robustness_per_prompt: Optional[dict] = None,
    top_n: int = 5,
    save: bool = True,
) -> dict:
    """
    Génère un rapport d'analyse qualitative complet pour un run.

    Combine :
    - Détection de non-conformité à la consigne
    - Typologie d'erreurs avec distribution
    - Sélection de cas extrêmes (si per_prompt scores fournis)
    - Analyse par catégorie thématique (si per_prompt scores fournis)

    Parameters
    ----------
    run_id : str
        Identifiant du run.
    output_dir : str
        Répertoire racine des résultats.
    languages : list[str] | None
        Langues à analyser.
    diversity_per_prompt : dict | None
        Dict {id: {"score": float}} issu de diversity_score()["per_prompt"].
    robustness_per_prompt : dict | None
        Dict {id: {"score": float}} issu de robustness_score()["per_prompt"].
    top_n : int
        Nombre de cas extrêmes à sélectionner dans chaque catégorie.
    save : bool
        Si True, sauvegarde le rapport en JSON.

    Returns
    -------
    dict
        Rapport qualitatif complet.
    """
    report: dict = {
        "run_id": run_id,
        "analysis_type": "qualitative",
    }

    # 1. Détection de non-conformité
    logger.info("Analyse qualitative – détection de non-conformité...")
    try:
        report["instruction_violations"] = detect_instruction_violations(
            run_id, output_dir, languages
        )
    except Exception as exc:
        logger.warning("Erreur détection non-conformité : %s", exc)
        report["instruction_violations"] = {"error": str(exc)}

    # 2. Typologie d'erreurs
    logger.info("Analyse qualitative – classification des erreurs...")
    try:
        report["error_typology"] = classify_errors(run_id, output_dir, languages)
    except Exception as exc:
        logger.warning("Erreur classification : %s", exc)
        report["error_typology"] = {"error": str(exc)}

    # 3. Cas extrêmes à partir des scores sémantiques
    if diversity_per_prompt:
        report["diversity_extreme_cases"] = select_extreme_cases(
            diversity_per_prompt, top_n=top_n
        )
        report["diversity_by_category"] = analyze_by_category(diversity_per_prompt)

    if robustness_per_prompt:
        report["robustness_extreme_cases"] = select_extreme_cases(
            robustness_per_prompt, top_n=top_n
        )
        report["robustness_by_category"] = analyze_by_category(robustness_per_prompt)

    if save:
        # Sauvegarde sans les exemples détaillés pour limiter la taille
        report_light = {k: v for k, v in report.items()
                        if k not in ("diversity_extreme_cases", "robustness_extreme_cases")}
        # Inclure les cas extrêmes mais sans les exemples de réponses complètes
        if "diversity_extreme_cases" in report:
            report_light["diversity_extreme_cases"] = report["diversity_extreme_cases"]
        if "robustness_extreme_cases" in report:
            report_light["robustness_extreme_cases"] = report["robustness_extreme_cases"]

        report_path = Path(output_dir) / run_id / "analysis_qualitative.json"
        report_path.write_text(
            json.dumps(report_light, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Rapport qualitatif sauvegardé : %s", report_path)

    return report

