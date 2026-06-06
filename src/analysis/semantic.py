"""
Analyse sémantique des résultats d'un run via embeddings.

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
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Constantes

DEFAULT_OUTPUT_DIR = "data/output"
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
SUPPORTED_LANGUAGES = ["en", "fr", "de", "es", "it"]

# Cache global du modèle pour éviter de le recharger à chaque appel
model_cache: dict[str, object] = {}


# Chargement du modèle


def load_model(model_name: str = DEFAULT_MODEL):
    """
    Charge le modèle SentenceTransformer et le met en cache.

    Le modèle est téléchargé automatiquement la première fois (≈ 120 Mo).
    Les appels suivants utilisent le cache mémoire.

    Parameters
    ----------
    model_name : str
        Nom Hugging Face du modèle (défaut: ``paraphrase-multilingual-MiniLM-L12-v2``).
    """
    if model_name not in model_cache:
        from sentence_transformers import SentenceTransformer  # import lazy
        logger.info("Chargement du modèle d'embedding : %s", model_name)
        model_cache[model_name] = SentenceTransformer(model_name)
        logger.info("Modèle chargé.")
    return model_cache[model_name]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similarité cosinus entre deux vecteurs 1-D."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def pairwisecosine_similarity(embeddings: np.ndarray) -> list[float]:
    """
    Calcule toutes les similarités cosinus paires pour une matrice
    d'embeddings (N × D).  Retourne une liste plate des N*(N-1)/2 valeurs.
    """
    n = len(embeddings)
    similarities: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            similarities.append(cosine_similarity(embeddings[i], embeddings[j]))
    return similarities


def pairwise_cosine_labeled(
    embeddings: np.ndarray, labels: list[str]
) -> dict[str, float]:
    """
    Retourne la similarité cosinus pour chaque paire de langues.
    Clé : "lang_a-lang_b", valeur : similarité cosinus.
    """
    n = len(embeddings)
    result: dict[str, float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = f"{labels[i]}-{labels[j]}"
            result[key] = round(cosine_similarity(embeddings[i], embeddings[j]), 4)
    return result


def dispersion(similarities: list[float]) -> float:
    """
    Diversité = 1 - similarité_moyenne.
    Proche de 1 -> réponses très différentes (diversité forte).
    Proche de 0 -> réponses très similaires (faible diversité).
    """
    if not similarities:
        return 0.0
    return round(1.0 - float(np.mean(similarities)), 4)


def cohesion(similarities: list[float]) -> float:
    """
    Robustesse = similarité_moyenne.
    Proche de 1 -> réponses stables malgré les contextes culturels.
    Proche de 0 -> réponses très instables.
    """
    if not similarities:
        return 0.0
    return round(float(np.mean(similarities)), 4)


def load_results_by_language_sem(
    run_id: str,
    dataset_type: str,
    output_dir: str,
    languages: list[str],
) -> dict[str, dict[str, str]]:
    """
    Charge les réponses d'un type de dataset pour toutes les langues.
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


# Score de Diversité
def diversity_score(
    run_id: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    languages: Optional[list[str]] = None,
    model_name: str = DEFAULT_MODEL,
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
    """
    if languages is None:
        languages = SUPPORTED_LANGUAGES

    model = load_model(model_name)
    data_by_lang = load_results_by_language_sem(run_id, "unspecific", output_dir, languages)

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
    # Accumulateur pour les scores par paire de langues
    pair_sims_accum: dict[str, list[float]] = {}

    for prompt_id in common_ids_sorted:
        texts = [data_by_lang[lang][prompt_id] for lang in available_langs]
        embeddings = model.encode(texts, convert_to_numpy=True)
        sims = pairwisecosine_similarity(embeddings)
        score = dispersion(sims)
        # Scores labellisés par paire
        labeled = pairwise_cosine_labeled(embeddings, available_langs)
        per_prompt_scores[prompt_id] = {
            "score": score,
            "languages": available_langs,
        }
        all_scores.append(score)
        for pair_key, sim_val in labeled.items():
            pair_sims_accum.setdefault(pair_key, []).append(sim_val)

    global_score  = round(float(np.mean(all_scores)), 4) if all_scores else 0.0
    global_std    = round(float(np.std(all_scores)),  4) if all_scores else 0.0
    global_min    = round(float(np.min(all_scores)),  4) if all_scores else 0.0
    global_max    = round(float(np.max(all_scores)),  4) if all_scores else 0.0
    global_median = round(float(np.median(all_scores)), 4) if all_scores else 0.0

    # Moyenne de divergence par paire (1 - sim)  -> paire la plus diverse en tête
    per_language_pair = {
        pair: round(1.0 - float(np.mean(vals)), 4)
        for pair, vals in pair_sims_accum.items()
    }

    return {
        "metric": "cultural_diversity",
        "dataset_type": "unspecific",
        "model": model_name,
        "score": global_score,
        "score_std": global_std,
        "score_min": global_min,
        "score_max": global_max,
        "score_median": global_median,
        "n_prompts": len(all_scores),
        "n_languages": len(available_langs),
        "languages": available_langs,
        "per_language_pair_diversity": per_language_pair,
        "per_prompt": per_prompt_scores,
    }


# Score de Robustesse
def robustness_score(
    run_id: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    languages: Optional[list[str]] = None,
    model_name: str = DEFAULT_MODEL,
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
    """
    if languages is None:
        languages = SUPPORTED_LANGUAGES

    model = load_model(model_name)
    data_by_lang = load_results_by_language_sem(run_id, "specific", output_dir, languages)

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
    pair_sims_accum: dict[str, list[float]] = {}

    for prompt_id in common_ids_sorted:
        texts = [data_by_lang[lang][prompt_id] for lang in available_langs]
        embeddings = model.encode(texts, convert_to_numpy=True)
        sims = pairwisecosine_similarity(embeddings)
        score = cohesion(sims)
        labeled = pairwise_cosine_labeled(embeddings, available_langs)
        per_prompt_scores[prompt_id] = {
            "score": score,
            "languages": available_langs,
        }
        all_scores.append(score)
        for pair_key, sim_val in labeled.items():
            pair_sims_accum.setdefault(pair_key, []).append(sim_val)

    global_score  = round(float(np.mean(all_scores)), 4) if all_scores else 0.0
    global_std    = round(float(np.std(all_scores)),  4) if all_scores else 0.0
    global_min    = round(float(np.min(all_scores)),  4) if all_scores else 0.0
    global_max    = round(float(np.max(all_scores)),  4) if all_scores else 0.0
    global_median = round(float(np.median(all_scores)), 4) if all_scores else 0.0

    # Robustesse par paire = similarité moyenne entre les deux langues
    per_language_pair = {
        pair: round(float(np.mean(vals)), 4)
        for pair, vals in pair_sims_accum.items()
    }

    return {
        "metric": "cultural_robustness",
        "dataset_type": "specific",
        "model": model_name,
        "score": global_score,
        "score_std": global_std,
        "score_min": global_min,
        "score_max": global_max,
        "score_median": global_median,
        "n_prompts": len(all_scores),
        "n_languages": len(available_langs),
        "languages": available_langs,
        "per_language_pair_robustness": per_language_pair,
        "per_prompt": per_prompt_scores,
    }


# Score Combiné
def combined_score(
    run_id: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    languages: Optional[list[str]] = None,
    model_name: str = DEFAULT_MODEL,
    sample_size: Optional[int] = None,
) -> dict:
    """
    Calcule les deux scores sémantiques et leur moyenne harmonique.

    La **moyenne harmonique** pénalise les déséquilibres : un score
    combiné élevé exige à la fois diversité ET robustesse.

        H = 2 × (D × R) / (D + R)
    """
    div = diversity_score(run_id, output_dir, languages, model_name, sample_size)
    rob = robustness_score(run_id, output_dir, languages, model_name, sample_size)

    d = div["score"]
    r = rob["score"]

    # Score combiné officiel challenge : produit simple D × R
    product = round(d * r, 4)

    # Score harmonique (alternatif, pénalise les déséquilibres)
    harmonic = round(2 * d * r / (d + r), 4) if (d + r) > 0 else 0.0

    # Interprétation basée sur le produit (méthode officielle du challenge)
    if product >= 0.36:
        interpretation = "Excellent : réponses très diversifiées culturellement et robustes."
    elif product >= 0.20:
        interpretation = "Bon : bonne diversité culturelle avec une robustesse satisfaisante."
    elif product >= 0.09:
        interpretation = "Moyen : des améliorations sont possibles sur la diversité ou la robustesse."
    else:
        interpretation = "Faible : les réponses manquent soit de diversité culturelle, soit de robustesse."

    return {
        "run_id": run_id,
        "diversity": div,
        "robustness": rob,
        "combined_score": product,          # méthode officielle challenge : D × R
        "combined_score_harmonic": harmonic, # alternative : pénalise les déséquilibres
        "interpretation": interpretation,
    }


# Rapport complet
def generate_report(
    run_id: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    languages: Optional[list[str]] = None,
    model_name: str = DEFAULT_MODEL,
    sample_size: Optional[int] = None,
    save: bool = True,
) -> dict:
    """
    Génère un rapport d'analyse sémantique complet pour un run.

    Calcule diversité, robustesse et score combiné, puis sauvegarde
    le résultat dans ``data/output/{run_id}/analysis_semantic.json``.

    Les résultats ``per_prompt`` sont exclus du fichier sauvegardé pour
    limiter la taille (ils sont conservés dans le dict retourné).
    """
    report = combined_score(run_id, output_dir, languages, model_name, sample_size)
    report["analysis_type"] = "semantic"

    if save:
        # Version allégée sans les détails par prompt (peut être très grand)
        report_light = {
            "run_id": report["run_id"],
            "analysis_type": "semantic",
            "model": model_name,
            # Diversité
            "diversity_score":    report["diversity"]["score"],
            "diversity_std":      report["diversity"]["score_std"],
            "diversity_min":      report["diversity"]["score_min"],
            "diversity_max":      report["diversity"]["score_max"],
            "diversity_median":   report["diversity"]["score_median"],
            "diversity_n_prompts": report["diversity"]["n_prompts"],
            "diversity_per_language_pair": report["diversity"].get("per_language_pair_diversity", {}),
            # Robustesse
            "robustness_score":   report["robustness"]["score"],
            "robustness_std":     report["robustness"]["score_std"],
            "robustness_min":     report["robustness"]["score_min"],
            "robustness_max":     report["robustness"]["score_max"],
            "robustness_median":  report["robustness"]["score_median"],
            "robustness_n_prompts": report["robustness"]["n_prompts"],
            "robustness_per_language_pair": report["robustness"].get("per_language_pair_robustness", {}),
            # Combiné
            "combined_score":     report["combined_score"],
            "combined_score_harmonic": report.get("combined_score_harmonic"),
            "interpretation":     report["interpretation"],
            "languages":          report["diversity"]["languages"],
        }
        report_path = Path(output_dir) / run_id / "analysis_semantic.json"
        report_path.write_text(
            json.dumps(report_light, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Rapport sémantique sauvegardé : %s", report_path)

    return report




# Comparaison sémantique entre deux runs
def compare_runs_semantic(
    run_id_a: str,
    run_id_b: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    languages: Optional[list[str]] = None,
    model_name: str = DEFAULT_MODEL,
    sample_size: Optional[int] = None,
) -> dict:
    """
    Compare les scores sémantiques (diversité + robustesse) de deux runs.

    Parameters
    ----------
    run_id_a : str
        Run de référence (ex: baseline).
    run_id_b : str
        Run à comparer (ex: variante cultural_expert).
    output_dir : str
        Répertoire racine des résultats.
    languages : list[str] | None
        Langues à inclure.
    model_name : str
        Modèle d'embedding.
    sample_size : int | None
        Limite le nombre de prompts analysés.
    """
    if languages is None:
        languages = SUPPORTED_LANGUAGES

    logger.info("Comparaison sémantique : %s vs %s", run_id_a, run_id_b)

    # Diversité (unspecific)
    try:
        div_a = diversity_score(run_id_a, output_dir, languages, model_name, sample_size)
        da = div_a["score"]
    except ValueError:
        da = None

    try:
        div_b = diversity_score(run_id_b, output_dir, languages, model_name, sample_size)
        db = div_b["score"]
    except ValueError:
        db = None

    # Robustesse (specific)
    try:
        rob_a = robustness_score(run_id_a, output_dir, languages, model_name, sample_size)
        ra = rob_a["score"]
    except ValueError:
        ra = None

    try:
        rob_b = robustness_score(run_id_b, output_dir, languages, model_name, sample_size)
        rb = rob_b["score"]
    except ValueError:
        rb = None

    # Scores combinés
    ca = round(da * ra, 4) if (da is not None and ra is not None) else None
    cb = round(db * rb, 4) if (db is not None and rb is not None) else None

    def delta(a, b):
        if a is None or b is None:
            return None
        return round(b - a, 4)

    delta_div = delta(da, db)
    delta_rob = delta(ra, rb)
    delta_com = delta(ca, cb)

    # Verdict textuel basé sur les deltas (amélioration ou dégradation)
    parts = []
    if delta_div is not None:
        parts.append(
            f"diversité {'améliorée' if delta_div > 0 else 'dégradée'} "
            f"({'+'if delta_div>0 else ''}{delta_div:.4f})"
        )
    if delta_rob is not None:
        parts.append(
            f"robustesse {'améliorée' if delta_rob > 0 else 'dégradée'} "
            f"({'+'if delta_rob>0 else ''}{delta_rob:.4f})"
        )
    verdict = f"Run B vs A : {', '.join(parts)}." if parts else "Comparaison incomplète."

    return {
        "run_a": run_id_a,
        "run_b": run_id_b,
        "diversity": {
            "score_a": da,
            "score_b": db,
            "delta": delta_div,
            "improved": (delta_div > 0) if delta_div is not None else None,
        },
        "robustness": {
            "score_a": ra,
            "score_b": rb,
            "delta": delta_rob,
            "improved": (delta_rob > 0) if delta_rob is not None else None,
        },
        "combined": {
            "score_a": ca,
            "score_b": cb,
            "delta": delta_com,
            "improved": (delta_com > 0) if delta_com is not None else None,
        },
        "verdict": verdict,
    }


