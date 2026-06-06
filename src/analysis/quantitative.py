"""
Analyse quantitative des résultats d'un run.

Ce module calcule des statistiques simples sur les réponses produites
par le pipeline ELOQUENT. Il ne nécessite pas de modèle d'embedding :
tout se calcule directement sur le texte brut.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonlines


# ── Constantes ──────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = "data/output"
SUPPORTED_LANGUAGES = ["en", "fr", "de", "es", "it"]
SUPPORTED_TYPES = ["unspecific", "specific"]


# ── Chargement ──────────────────────────────────────────────────────────────


def load_run_results(
    run_id: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, list[dict]]:
    """
    Charge tous les fichiers JSONL de résultats d'un run.
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


def count_words(text: str) -> int:
    """Compte le nombre de mots dans un texte."""
    return len(text.split()) if text.strip() else 0

def is_empty(answer: str) -> bool:
    """Retourne True si la réponse contient moins de 3 mots."""
    return (not is_error(answer)) and count_words(answer.strip()) < 3

def is_error(answer: str) -> bool:
    """Retourne True si la réponse est une erreur pipeline (préfixe ERROR:)."""
    return answer.strip().startswith("ERROR:")


# Statistiques de base
def compute_basic_stats(
    run_id: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict:
    """
    Calcule des statistiques quantitatives sur toutes les réponses d'un run.

    Pour chaque fichier (langue × type) calcule :
    - Nombre total de réponses
    - Longueur moyenne en mots et en caractères
    - Taux de réponses vides (< 3 mots)
    - Taux d'erreurs pipeline ("ERROR: …")
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
            word_counts.append(count_words(answer))
            char_counts.append(len(answer))
            if is_error(answer):
                error_count += 1
            elif is_empty(answer):
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


# Comparaison de deux runs
def compare_runs(
    run_id_a: str,
    run_id_b: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
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

        words_a = [count_words(i["answer"]) for i in items_a]
        words_b = [count_words(i["answer"]) for i in items_b]

        errors_a = sum(1 for i in items_a if is_error(i["answer"]))
        errors_b = sum(1 for i in items_b if is_error(i["answer"]))

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


# Fusion de deux rapports quantitatifs
def merge_stats(stats_a: dict, stats_b: dict) -> dict:
    """
    Fusionne les statistiques de deux runs en un seul dict.

    Les clés ``"lang_type"`` des deux dicts sont combinées.
    La clé ``"global"`` est recalculée à partir de l'ensemble des fichiers.

    Utilisé pour afficher à la fois les fichiers *unspecific* et *specific*
    lorsque les deux types sont répartis sur deux runs distincts.
    """
    merged: dict[str, dict] = {}

    # Fusionner toutes les clés sauf "global"
    for key, val in stats_a.items():
        if key != "global":
            merged[key] = val
    for key, val in stats_b.items():
        if key != "global" and key not in merged:
            merged[key] = val

    # Recalculer "global" à partir de l'ensemble des fichiers fusionnés
    total_all  = 0
    sum_words  = 0.0
    sum_chars  = 0.0
    min_words  = None
    max_words  = None
    empty_all  = 0
    error_all  = 0

    for s in merged.values():
        n = s.get("total", 0)
        if n == 0:
            continue
        total_all  += n
        sum_words  += s.get("avg_words", 0) * n
        sum_chars  += s.get("avg_chars", 0) * n
        empty_all  += round(s.get("empty_rate", 0) * n)
        error_all  += round(s.get("error_rate", 0) * n)
        fw = s.get("min_words", 0)
        lw = s.get("max_words", 0)
        if min_words is None or fw < min_words:
            min_words = fw
        if max_words is None or lw > max_words:
            max_words = lw

    merged["global"] = {
        "total":      total_all,
        "avg_words":  round(sum_words / total_all, 2) if total_all else 0,
        "avg_chars":  round(sum_chars / total_all, 2) if total_all else 0,
        "min_words":  min_words or 0,
        "max_words":  max_words or 0,
        "empty_rate": round(empty_all / total_all, 4) if total_all else 0.0,
        "error_rate": round(error_all / total_all, 4) if total_all else 0.0,
    }

    return merged


# Rapport complet
def generate_report(
    run_id: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    save: bool = True,
) -> dict:
    """
    Génère un rapport d'analyse quantitative complet pour un run.

    Parameters
    ----------
    run_id : str
        Identifiant du run.
    output_dir : str
        Répertoire racine des résultats.
    save : bool
        Si True (défaut), sauvegarde le rapport en JSON.
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

