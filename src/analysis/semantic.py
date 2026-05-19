"""
Analyse sémantique des résultats d'un run via embeddings multilingues.

Ce module mesure deux propriétés clés définies par le challenge ELOQUENT :

1. **Score de Diversité Culturelle** (sur les fichiers ``*_unspecific``)
   Pour chaque question posée dans les 5 langues, les réponses devraient
   idéalement varier selon la culture de la langue.  Un score élevé indique
   que le modèle produit des réponses culturellement distinctes selon la
   langue, ce qui est souhaité.

2. **Score de Robustesse Culturelle** (sur les fichiers ``*_specific``)
   Pour chaque question posée avec un contexte culturel explicite dans les
   5 langues, les réponses devraient rester cohérentes sur le fond.
   Un score élevé indique que le modèle n'est pas déstabilisé par le
   contexte culturel et maintient une réponse de qualité stable.

3. **Score Combiné** : moyenne harmonique des deux scores ci-dessus.

Modèle utilisé
--------------
``paraphrase-multilingual-MiniLM-L12-v2`` (sentence-transformers)
Modèle léger (≈ 120 Mo), entraîné sur 50+ langues, idéal pour comparer
des réponses dans EN, FR, DE, ES, IT.

Fonctions principales
---------------------
load_model(model_name)
    Charge (et met en cache) le modèle d'embedding.

diversity_score(run_id, ...)
    Calcule le score de diversité kulturelle en mesurant la dispersion
    cosinus des réponses inter-langues pour les questions *unspecific*.

robustness_score(run_id, ...)
    Calcule le score de robustesse culturelle en mesurant la similarité
    cosinus des réponses inter-langues pour les questions *specific*.

combined_score(run_id, ...)
    Calcule et retourne les deux scores + la moyenne harmonique.

generate_report(run_id, ...)
    Génère un rapport complet sauvegardé en JSON dans le dossier du run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Constantes ──────────────────────────────────────────────────────────────

_DEFAULT_OUTPUT_DIR = "data/output"
_DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_SUPPORTED_LANGUAGES = ["en", "fr", "de", "es", "it"]

# Cache global du modèle pour éviter de le recharger à chaque appel
_model_cache: dict[str, object] = {}


# ─── Chargement du modèle ────────────────────────────────────────────────────


def load_model(model_name: str = _DEFAULT_MODEL):
    """
    Charge le modèle SentenceTransformer et le met en cache.

    Le modèle est téléchargé automatiquement la première fois (≈ 120 Mo).
    Les appels suivants utilisent le cache mémoire.

    Parameters
    ----------
    model_name : str
        Nom Hugging Face du modèle (défaut: ``paraphrase-multilingual-MiniLM-L12-v2``).

    Returns
    -------
    SentenceTransformer
        Instance du modèle prête à encoder.
    """
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer  # import lazy
        logger.info("Chargement du modèle d'embedding : %s", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
        logger.info("Modèle chargé.")
    return _model_cache[model_name]


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similarité cosinus entre deux vecteurs 1-D."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _pairwise_cosine_similarity(embeddings: np.ndarray) -> list[float]:
    """
    Calcule toutes les similarités cosinus paires pour une matrice
    d'embeddings (N × D).  Retourne une liste plate des N*(N-1)/2 valeurs.
    """
    n = len(embeddings)
    similarities: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            similarities.append(_cosine_similarity(embeddings[i], embeddings[j]))
    return similarities


def _dispersion(similarities: list[float]) -> float:
    """
    Diversité = 1 - similarité_moyenne.
    Proche de 1 → réponses très différentes (diversité forte).
    Proche de 0 → réponses très similaires (faible diversité).
    """
    if not similarities:
        return 0.0
    return round(1.0 - float(np.mean(similarities)), 4)


def _cohesion(similarities: list[float]) -> float:
    """
    Robustesse = similarité_moyenne.
    Proche de 1 → réponses stables malgré les contextes culturels.
    Proche de 0 → réponses très instables.
    """
    if not similarities:
        return 0.0
    return round(float(np.mean(similarities)), 4)


def _load_results_by_language(
    run_id: str,
    dataset_type: str,
    output_dir: str,
    languages: list[str],
) -> dict[str, dict[str, str]]:
    """
    Charge les réponses d'un type de dataset pour toutes les langues.

    Returns
    -------
    dict[str, dict[str, str]]
        { "fr": {"1": "réponse...", "2": "..."}, "en": {...}, ... }
    """
    import jsonlines as jl

    data: dict[str, dict[str, str]] = {}
    run_path = Path(output_dir) / run_id

    for lang in languages:
        filepath = run_path / f"{lang}_{dataset_type}.jsonl"
        if not filepath.exists():
            logger.warning("Fichier absent : %s", filepath)
            continue
        lang_data: dict[str, str] = {}
        with jl.open(str(filepath), mode="r") as reader:
            for obj in reader:
                answer = obj.get("answer", "")
                # Ignorer les erreurs pipeline dans les calculs sémantiques
                if not answer.strip().startswith("ERROR:") and answer.strip():
                    lang_data[obj["id"]] = answer
        data[lang] = lang_data

    return data


# ─── Score de Diversité ───────────────────────────────────────────────────────


def diversity_score(
    run_id: str,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    languages: Optional[list[str]] = None,
    model_name: str = _DEFAULT_MODEL,
    sample_size: Optional[int] = None,
) -> dict:
    """
    Calcule le score de diversité culturelle sur les questions *unspecific*.

    Pour chaque question (même ID), récupère les réponses dans toutes les
    langues disponibles, encode les textes, puis mesure la **dispersion**
    cosinus inter-langues :

        diversité_Q = 1 − sim_cosinus_moyenne(réponses_toutes_langues)

    Un score élevé signifie que le modèle adapte ses réponses selon la
    langue / culture, ce qui est souhaité (objectif du challenge).

    Parameters
    ----------
    run_id : str
        Identifiant du run.
    output_dir : str
        Répertoire racine des résultats.
    languages : list[str] | None
        Langues à inclure (défaut : toutes les 5 langues).
    model_name : str
        Modèle SentenceTransformer à utiliser.
    sample_size : int | None
        Si défini, limite l'analyse aux N premiers IDs (pour aller plus vite).

    Returns
    -------
    dict
        {
            "metric": "cultural_diversity",
            "dataset_type": "unspecific",
            "score": 0.23,            # moyenne sur tous les prompts
            "score_std": 0.08,        # écart-type
            "n_prompts": 101,         # nombre de questions évaluées
            "n_languages": 5,
            "per_prompt": {           # optionnel si sample_size actif
                "1": {"score": 0.21, "languages": ["en","fr","de","es","it"]},
                ...
            }
        }
    """
    if languages is None:
        languages = _SUPPORTED_LANGUAGES

    model = load_model(model_name)
    data_by_lang = _load_results_by_language(run_id, "unspecific", output_dir, languages)

    if not data_by_lang:
        raise ValueError(f"Aucun fichier unspecific chargé pour le run : {run_id}")

    # Intersection des IDs présents dans toutes les langues chargées
    available_langs = list(data_by_lang.keys())
    common_ids = set.intersection(*[set(data_by_lang[l].keys()) for l in available_langs])
    common_ids_sorted = sorted(common_ids)[:sample_size]  # limit si demandé

    logger.info(
        "Diversité – %d prompts × %d langues à encoder...",
        len(common_ids_sorted),
        len(available_langs),
    )

    per_prompt_scores: dict[str, dict] = {}
    all_scores: list[float] = []

    for prompt_id in common_ids_sorted:
        texts = [data_by_lang[lang][prompt_id] for lang in available_langs]
        embeddings = model.encode(texts, convert_to_numpy=True)
        sims = _pairwise_cosine_similarity(embeddings)
        score = _dispersion(sims)
        per_prompt_scores[prompt_id] = {
            "score": score,
            "languages": available_langs,
        }
        all_scores.append(score)

    global_score = round(float(np.mean(all_scores)), 4) if all_scores else 0.0
    global_std = round(float(np.std(all_scores)), 4) if all_scores else 0.0

    return {
        "metric": "cultural_diversity",
        "dataset_type": "unspecific",
        "model": model_name,
        "score": global_score,
        "score_std": global_std,
        "n_prompts": len(all_scores),
        "n_languages": len(available_langs),
        "languages": available_langs,
        "per_prompt": per_prompt_scores,
    }


# ─── Score de Robustesse ──────────────────────────────────────────────────────


def robustness_score(
    run_id: str,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    languages: Optional[list[str]] = None,
    model_name: str = _DEFAULT_MODEL,
    sample_size: Optional[int] = None,
) -> dict:
    """
    Calcule le score de robustesse culturelle sur les questions *specific*.

    Pour chaque question (même ID), récupère les réponses dans toutes les
    langues disponibles, encode les textes, puis mesure la **cohésion**
    cosinus inter-langues :

        robustesse_Q = sim_cosinus_moyenne(réponses_toutes_langues)

    Un score élevé signifie que le modèle donne des réponses cohérentes
    sur le fond même quand le *contexte culturel* change.

    Parameters
    ----------
    run_id : str
        Identifiant du run.
    output_dir : str
        Répertoire racine des résultats.
    languages : list[str] | None
        Langues à inclure (défaut : toutes les 5 langues).
    model_name : str
        Modèle SentenceTransformer à utiliser.
    sample_size : int | None
        Si défini, limite l'analyse aux N premiers IDs.

    Returns
    -------
    dict
        {
            "metric": "cultural_robustness",
            "dataset_type": "specific",
            "score": 0.74,
            "score_std": 0.11,
            "n_prompts": 101,
            "n_languages": 5,
            "per_prompt": { "1": {"score": 0.78, ...}, ... }
        }
    """
    if languages is None:
        languages = _SUPPORTED_LANGUAGES

    model = load_model(model_name)
    data_by_lang = _load_results_by_language(run_id, "specific", output_dir, languages)

    if not data_by_lang:
        raise ValueError(f"Aucun fichier specific chargé pour le run : {run_id}")

    available_langs = list(data_by_lang.keys())
    common_ids = set.intersection(*[set(data_by_lang[l].keys()) for l in available_langs])
    common_ids_sorted = sorted(common_ids)[:sample_size]

    logger.info(
        "Robustesse – %d prompts × %d langues à encoder...",
        len(common_ids_sorted),
        len(available_langs),
    )

    per_prompt_scores: dict[str, dict] = {}
    all_scores: list[float] = []

    for prompt_id in common_ids_sorted:
        texts = [data_by_lang[lang][prompt_id] for lang in available_langs]
        embeddings = model.encode(texts, convert_to_numpy=True)
        sims = _pairwise_cosine_similarity(embeddings)
        score = _cohesion(sims)
        per_prompt_scores[prompt_id] = {
            "score": score,
            "languages": available_langs,
        }
        all_scores.append(score)

    global_score = round(float(np.mean(all_scores)), 4) if all_scores else 0.0
    global_std = round(float(np.std(all_scores)), 4) if all_scores else 0.0

    return {
        "metric": "cultural_robustness",
        "dataset_type": "specific",
        "model": model_name,
        "score": global_score,
        "score_std": global_std,
        "n_prompts": len(all_scores),
        "n_languages": len(available_langs),
        "languages": available_langs,
        "per_prompt": per_prompt_scores,
    }


# ─── Score Combiné ────────────────────────────────────────────────────────────


def combined_score(
    run_id: str,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    languages: Optional[list[str]] = None,
    model_name: str = _DEFAULT_MODEL,
    sample_size: Optional[int] = None,
) -> dict:
    """
    Calcule les deux scores sémantiques et leur moyenne harmonique.

    La **moyenne harmonique** pénalise les déséquilibres : un score
    combiné élevé exige à la fois diversité ET robustesse.

        H = 2 × (D × R) / (D + R)

    Parameters
    ----------
    run_id : str
        Identifiant du run.
    output_dir, languages, model_name, sample_size
        Idem que ``diversity_score`` et ``robustness_score``.

    Returns
    -------
    dict
        {
            "run_id": "...",
            "diversity": { ... résultat complet diversity_score ... },
            "robustness": { ... résultat complet robustness_score ... },
            "combined_score": 0.44,   # moyenne harmonique
            "interpretation": "..."   # texte explicatif
        }
    """
    div = diversity_score(run_id, output_dir, languages, model_name, sample_size)
    rob = robustness_score(run_id, output_dir, languages, model_name, sample_size)

    d = div["score"]
    r = rob["score"]

    if d + r > 0:
        harmonic = round(2 * d * r / (d + r), 4)
    else:
        harmonic = 0.0

    # Interprétation qualitative
    if harmonic >= 0.60:
        interpretation = "Excellent : réponses très diversifiées culturellement et robustes."
    elif harmonic >= 0.45:
        interpretation = "Bon : bonne diversité culturelle avec une robustesse satisfaisante."
    elif harmonic >= 0.30:
        interpretation = "Moyen : des améliorations sont possibles sur la diversité ou la robustesse."
    else:
        interpretation = "Faible : les réponses manquent soit de diversité culturelle, soit de robustesse."

    return {
        "run_id": run_id,
        "diversity": div,
        "robustness": rob,
        "combined_score": harmonic,
        "interpretation": interpretation,
    }


# ─── Rapport complet ──────────────────────────────────────────────────────────


def generate_report(
    run_id: str,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    languages: Optional[list[str]] = None,
    model_name: str = _DEFAULT_MODEL,
    sample_size: Optional[int] = None,
    save: bool = True,
) -> dict:
    """
    Génère un rapport d'analyse sémantique complet pour un run.

    Calcule diversité, robustesse et score combiné, puis sauvegarde
    le résultat dans ``data/output/{run_id}/analysis_semantic.json``.

    Les résultats ``per_prompt`` sont exclus du fichier sauvegardé pour
    limiter la taille (ils sont conservés dans le dict retourné).

    Parameters
    ----------
    run_id : str
        Identifiant du run.
    output_dir : str
        Répertoire racine des résultats.
    languages : list[str] | None
        Langues à inclure.
    model_name : str
        Modèle d'embedding à utiliser.
    sample_size : int | None
        Limite le nombre de prompts analysés (utile pour tests rapides).
    save : bool
        Si True (défaut), sauvegarde le rapport JSON.

    Returns
    -------
    dict
        Rapport complet incluant les détails par prompt.
    """
    report = combined_score(run_id, output_dir, languages, model_name, sample_size)
    report["analysis_type"] = "semantic"

    if save:
        # Version allégée sans les détails par prompt (peut être très grand)
        report_light = {
            "run_id": report["run_id"],
            "analysis_type": "semantic",
            "model": model_name,
            "diversity_score": report["diversity"]["score"],
            "diversity_std": report["diversity"]["score_std"],
            "diversity_n_prompts": report["diversity"]["n_prompts"],
            "robustness_score": report["robustness"]["score"],
            "robustness_std": report["robustness"]["score_std"],
            "robustness_n_prompts": report["robustness"]["n_prompts"],
            "combined_score": report["combined_score"],
            "interpretation": report["interpretation"],
            "languages": report["diversity"]["languages"],
        }
        report_path = Path(output_dir) / run_id / "analysis_semantic.json"
        report_path.write_text(
            json.dumps(report_light, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Rapport sémantique sauvegardé : %s", report_path)

    return report

