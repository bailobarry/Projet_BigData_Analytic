"""
Analyse quantitative des résultats d'un run.

Ce module calcule des statistiques simples sur les réponses produites
par le pipeline ELOQUENT. Il ne nécessite pas de modèle d'embedding :
tout se calcule directement sur le texte brut.

Fonctions principales
---------------------
load_run_results(run_id)
    Charge tous les fichiers JSONL de résultats d'un run.

compute_basic_stats(run_id)
    Calcule longueur moyenne, taux de vides et taux d'erreurs
    par langue et par type de dataset.

compare_runs(run_id_a, run_id_b)
    Compare deux runs prompt par prompt (baseline vs variante).

generate_report(run_id)
    Génère un rapport complet au format JSON dans le dossier du run.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonlines


# ── Constantes ──────────────────────────────────────────────────────────────

_DEFAULT_OUTPUT_DIR = "data/output"
_SUPPORTED_LANGUAGES = ["en", "fr", "de", "es", "it"]
_SUPPORTED_TYPES = ["unspecific", "specific"]


# ── Chargement ──────────────────────────────────────────────────────────────


def load_run_results(
    run_id: str,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
) -> dict[str, list[dict]]:
    """
    Charge tous les fichiers JSONL de résultats d'un run.

    Parameters
    ----------
    run_id : str
        Identifiant du run (ex: "run_20260519_054344_513e07").
    output_dir : str
        Répertoire racine contenant les résultats (défaut: "data/output").

    Returns
    -------
    dict[str, list[dict]]
        Dictionnaire { "fr_unspecific": [...], "en_specific": [...], ... }
        où chaque valeur est la liste des résultats (id, prompt, answer).
    """
    run_path = Path(output_dir) / run_id
    if not run_path.exists():
        raise FileNotFoundError(f"Répertoire du run introuvable : {run_path}")

    results: dict[str, list[dict]] = {}
    for jsonl_file in sorted(run_path.glob("*.jsonl")):
        key = jsonl_file.stem  # ex: "fr_unspecific"
        items: list[dict] = []
        with jsonlines.open(str(jsonl_file), mode="r") as reader:
            for obj in reader:
                items.append(obj)
        results[key] = items

    if not results:
        raise ValueError(f"Aucun fichier JSONL trouvé dans : {run_path}")

    return results


# ── Helpers privés ───────────────────────────────────────────────────────────


def _count_words(text: str) -> int:
    """Compte le nombre de mots dans un texte."""
    return len(text.split()) if text.strip() else 0


def _is_empty(answer: str) -> bool:
    """Retourne True si la réponse contient moins de 3 mots (hors erreur)."""
    return (not _is_error(answer)) and _count_words(answer.strip()) < 3


def _is_error(answer: str) -> bool:
    """Retourne True si la réponse est une erreur pipeline (préfixe ERROR:)."""
    return answer.strip().startswith("ERROR:")


# ── Statistiques de base ─────────────────────────────────────────────────────


def compute_basic_stats(
    run_id: str,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
) -> dict:
    """
    Calcule des statistiques quantitatives sur toutes les réponses d'un run.

    Pour chaque fichier (langue × type) calcule :
    - Nombre total de réponses
    - Longueur moyenne en mots et en caractères
    - Taux de réponses vides (< 3 mots)
    - Taux d'erreurs pipeline ("ERROR: …")

    Returns
    -------
    dict
        Statistiques par clé ``"lang_type"`` + clé ``"global"`` agrégée.

    Exemple
    -------
    {
        "fr_unspecific": {
            "total": 101,
            "avg_words": 25.3,
            "avg_chars": 142.1,
            "empty_rate": 0.0,
            "error_rate": 0.02
        },
        ...
        "global": { ... }
    }
    """
    all_results = load_run_results(run_id, output_dir)
    stats: dict[str, dict] = {}

    all_words: list[int] = []
    all_chars: list[int] = []
    total_all = empty_all = error_all = 0

    for key, items in sorted(all_results.items()):
        word_counts: list[int] = []
        char_counts: list[int] = []
        empty_count = error_count = 0

        for item in items:
            answer = item.get("answer", "")
            word_counts.append(_count_words(answer))
            char_counts.append(len(answer))
            if _is_error(answer):
                error_count += 1
            elif _is_empty(answer):
                empty_count += 1

        n = len(items)
        stats[key] = {
            "total":      n,
            "avg_words":  round(sum(word_counts) / n, 2) if n else 0,
            "avg_chars":  round(sum(char_counts) / n, 2) if n else 0,
            "min_words":  min(word_counts) if word_counts else 0,
            "max_words":  max(word_counts) if word_counts else 0,
            "empty_rate": round(empty_count / n, 4) if n else 0.0,
            "error_rate": round(error_count / n, 4) if n else 0.0,
        }

        all_words.extend(word_counts)
        all_chars.extend(char_counts)
        total_all += n
        empty_all += empty_count
        error_all += error_count

    stats["global"] = {
        "total":      total_all,
        "avg_words":  round(sum(all_words) / total_all, 2) if total_all else 0,
        "avg_chars":  round(sum(all_chars) / total_all, 2) if total_all else 0,
        "min_words":  min(all_words) if all_words else 0,
        "max_words":  max(all_words) if all_words else 0,
        "empty_rate": round(empty_all / total_all, 4) if total_all else 0.0,
        "error_rate": round(error_all / total_all, 4) if total_all else 0.0,
    }

    return stats


# ── Comparaison de deux runs ─────────────────────────────────────────────────


def compare_runs(
    run_id_a: str,
    run_id_b: str,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
) -> dict:
    """
    Compare deux runs sur les métriques quantitatives.

    Ne calcule PAS de similarité sémantique (c'est le rôle de semantic.py).
    Mesure uniquement les différences de longueur et de taux d'erreurs.

    Parameters
    ----------
    run_id_a : str
        Run de référence (ex: baseline).
    run_id_b : str
        Run à comparer (ex: variante empathetic_synthesis).

    Returns
    -------
    dict
        {
            "run_a": "...", "run_b": "...",
            "files": {
                "fr_unspecific": {
                    "common_ids": 101,
                    "only_in_a": 0,
                    "only_in_b": 0,
                    "avg_words_a": 25.3,
                    "avg_words_b": 31.7,
                    "delta_avg_words": 6.4,
                    "error_rate_a": 0.0,
                    "error_rate_b": 0.0,
                    "delta_error_rate": 0.0
                },
                ...
            }
        }
    """
    results_a = load_run_results(run_id_a, output_dir)
    results_b = load_run_results(run_id_b, output_dir)

    comparison: dict = {
        "run_a": run_id_a,
        "run_b": run_id_b,
        "files": {},
    }

    for key in sorted(set(results_a.keys()) | set(results_b.keys())):
        items_a = results_a.get(key, [])
        items_b = results_b.get(key, [])

        ids_a = {item["id"] for item in items_a}
        ids_b = {item["id"] for item in items_b}

        words_a = [_count_words(i["answer"]) for i in items_a]
        words_b = [_count_words(i["answer"]) for i in items_b]

        errors_a = sum(1 for i in items_a if _is_error(i["answer"]))
        errors_b = sum(1 for i in items_b if _is_error(i["answer"]))

        avg_a = round(sum(words_a) / len(words_a), 2) if words_a else 0
        avg_b = round(sum(words_b) / len(words_b), 2) if words_b else 0
        err_rate_a = round(errors_a / len(items_a), 4) if items_a else 0.0
        err_rate_b = round(errors_b / len(items_b), 4) if items_b else 0.0

        comparison["files"][key] = {
            "common_ids": len(ids_a & ids_b),
            "only_in_a": len(ids_a - ids_b),
            "only_in_b": len(ids_b - ids_a),
            "avg_words_a": avg_a,
            "avg_words_b": avg_b,
            "delta_avg_words": round(avg_b - avg_a, 2),
            "error_rate_a": err_rate_a,
            "error_rate_b": err_rate_b,
            "delta_error_rate": round(err_rate_b - err_rate_a, 4),
        }

    return comparison


# ── Rapport complet ──────────────────────────────────────────────────────────


def generate_report(
    run_id: str,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    save: bool = True,
) -> dict:
    """
    Génère un rapport d'analyse quantitative complet pour un run.

    Sauvegarde optionnellement le résultat dans
    ``data/output/{run_id}/analysis_quantitative.json``.

    Parameters
    ----------
    run_id : str
        Identifiant du run.
    output_dir : str
        Répertoire racine des résultats.
    save : bool
        Si True (défaut), sauvegarde le rapport en JSON.

    Returns
    -------
    dict
        Rapport complet (statistiques par fichier + stats globales).
    """
    stats = compute_basic_stats(run_id, output_dir)

    report = {
        "run_id": run_id,
        "analysis_type": "quantitative",
        "stats_by_file": stats,
    }

    if save:
        report_path = Path(output_dir) / run_id / "analysis_quantitative.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return report

