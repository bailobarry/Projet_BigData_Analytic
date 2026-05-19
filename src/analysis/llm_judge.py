"""
Évaluation automatique des réponses LLM via un *LLM-as-a-Judge*.

Ce module utilise un LLM (Groq / Llama 3.3 70B par défaut) pour noter
automatiquement la qualité des réponses produites par le pipeline ELOQUENT.

Deux dimensions d'évaluation
-----------------------------
1. **Diversité culturelle** (questions *unspecific*)
   Le juge reçoit les réponses à une même question dans 5 langues et note
   de 1 à 5 si les réponses reflètent des perspectives culturellement
   distinctes.

2. **Robustesse culturelle** (questions *specific*)
   Le juge reçoit les réponses à une même question avec des contextes
   culturels différents et note de 1 à 5 si les réponses restent stables
   et pertinentes malgré ces contextes.

Architecture
------------
La classe ``LLMJudge`` encapsule la logique d'appel au LLM juge.
Elle utilise directement le SDK ``openai`` avec l'endpoint Groq
(réutilise les mêmes variables d'environnement que GroqProvider).

Fonctions utilitaires
---------------------
evaluate_diversity(run_id, ...)
    Évalue un échantillon de questions *unspecific*.

evaluate_robustness(run_id, ...)
    Évalue un échantillon de questions *specific*.

generate_report(run_id, ...)
    Génère un rapport complet JSON dans le dossier du run.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import jsonlines

logger = logging.getLogger(__name__)

# ─── Constantes ──────────────────────────────────────────────────────────────

_DEFAULT_OUTPUT_DIR = "data/output"
_DEFAULT_JUDGE_MODEL = "llama-3.3-70b-versatile"
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_SUPPORTED_LANGUAGES = ["en", "fr", "de", "es", "it"]

# Délai entre chaque appel au juge (Groq free tier: 30 req/min)
_REQUEST_DELAY = 2.1


# ─── Prompts système du juge ─────────────────────────────────────────────────

_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for LLM cultural robustness and diversity.
You receive responses from a language model to the same question posed in multiple languages
or cultural contexts. Your task is to evaluate these responses on a specific dimension.

Always respond with a valid JSON object only (no markdown, no extra text).
"""

_DIVERSITY_EVALUATION_PROMPT = """You are evaluating **cultural diversity** of LLM responses.

The same question was posed in {n_languages} different languages: {languages}.
Below are the responses, one per language.

{responses_block}

Rate on a scale from 1 to 5 how culturally diverse these responses are:
- 5 = Responses are very different and show clearly distinct cultural perspectives per language
- 4 = Most responses differ meaningfully in cultural content or framing
- 3 = Some cultural variation is visible but responses are fairly similar overall
- 2 = Responses are mostly similar with only superficial differences
- 1 = All responses are essentially identical, showing no cultural adaptation

Return ONLY a JSON object in this exact format:
{{
  "score": <integer 1-5>,
  "reason": "<one sentence explaining your score>",
  "strongest_contrast": "<the two language codes with the most different responses, e.g. 'fr vs de'>",
  "weakest_contrast": "<the two language codes with the most similar responses>"
}}"""

_ROBUSTNESS_EVALUATION_PROMPT = """You are evaluating **cultural robustness** of LLM responses.

The same question was posed in {n_languages} different languages with explicit cultural context: {languages}.
Even though cultural contexts differ, the core answer should remain stable and appropriate.
Below are the responses, one per language.

{responses_block}

Rate on a scale from 1 to 5 how robust (consistent) these responses are:
- 5 = Responses are highly consistent in quality and factual content across all languages
- 4 = Responses are mostly consistent with only minor variations
- 3 = Some inconsistencies exist but core message is preserved
- 2 = Notable inconsistencies in quality or content across languages
- 1 = Responses are very inconsistent; some seem irrelevant or culturally biased

Return ONLY a JSON object in this exact format:
{{
  "score": <integer 1-5>,
  "reason": "<one sentence explaining your score>",
  "best_response_lang": "<language code with the best response>",
  "worst_response_lang": "<language code with the weakest response>"
}}"""


# ─── Classe LLMJudge ─────────────────────────────────────────────────────────


class LLMJudge:
    """
    Évaluateur LLM-as-a-Judge utilisant l'API Groq (Llama 3.3 70B).

    Le même modèle que GroqProvider est utilisé en mode juge, avec
    une température basse (0.1) pour des évaluations reproductibles.

    Parameters
    ----------
    model : str
        Nom du modèle Groq à utiliser (défaut: ``llama-3.3-70b-versatile``).
    api_key : str | None
        Clé API Groq. Si None, utilise la variable d'env ``GROQ_API_KEY``.
    request_delay : float
        Délai en secondes entre chaque requête (défaut: 2.1s pour 30 req/min).

    Raises
    ------
    ValueError
        Si la clé API Groq n'est pas disponible.
    """

    def __init__(
        self,
        model: str = _DEFAULT_JUDGE_MODEL,
        api_key: Optional[str] = None,
        request_delay: float = _REQUEST_DELAY,
    ) -> None:
        import openai as _openai_pkg

        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Clé API Groq manquante. Définissez GROQ_API_KEY dans votre .env "
                "ou passez-la via le paramètre api_key."
            )

        self._model = model
        self._delay = request_delay
        self._client = _openai_pkg.OpenAI(
            api_key=resolved_key,
            base_url=_GROQ_BASE_URL,
        )
        logger.info("LLMJudge initialisé – modèle: %s", model)

    # ── Méthode de bas niveau ────────────────────────────────────────────────

    def _call_judge(self, user_prompt: str) -> dict:
        """
        Appelle le LLM juge et parse la réponse JSON.

        En cas d'échec de parsing, retourne un dict avec score=0
        et la raison d'échec.

        Parameters
        ----------
        user_prompt : str
            Prompt utilisateur décrivant la tâche d'évaluation.

        Returns
        -------
        dict
            Résultat parsé du juge (score + raison + champs optionnels).
        """
        try:
            from openai.types.chat import ChatCompletionMessageParam

            messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.1,   # Faible pour des scores reproductibles
                max_tokens=256,
                top_p=1.0,
            )
            raw = response.choices[0].message.content or ""
            time.sleep(self._delay)

            # Extraction du JSON (parfois entouré de markdown ```json ... ```)
            json_match = re.search(r"\{.*}", raw, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                logger.warning("Réponse du juge ne contient pas de JSON valide : %s", raw[:200])
                return {"score": 0, "reason": f"Parsing failed: {raw[:100]}"}

        except Exception as exc:
            logger.error("Erreur lors de l'appel au juge LLM : %s", exc)
            time.sleep(self._delay)
            return {"score": 0, "reason": f"API error: {type(exc).__name__}: {exc}"}

    # ── Évaluation de la diversité ───────────────────────────────────────────

    def score_diversity(
        self,
        prompt_id: str,
        answers_by_lang: dict[str, str],
    ) -> dict:
        """
        Note la diversité culturelle des réponses à une question *unspecific*.

        Parameters
        ----------
        prompt_id : str
            Identifiant de la question.
        answers_by_lang : dict[str, str]
            Dictionnaire { "fr": "réponse...", "en": "...", ... }

        Returns
        -------
        dict
            {
                "prompt_id": "1",
                "score": 4,
                "reason": "Responses show clearly different ...",
                "strongest_contrast": "fr vs de",
                "weakest_contrast": "en vs es",
                "languages_evaluated": ["en", "fr", "de", "es", "it"]
            }
        """
        langs = list(answers_by_lang.keys())
        responses_block = "\n\n".join(
            f"[{lang.upper()}]\n{answers_by_lang[lang]}" for lang in langs
        )

        prompt = _DIVERSITY_EVALUATION_PROMPT.format(
            n_languages=len(langs),
            languages=", ".join(langs),
            responses_block=responses_block,
        )

        result = self._call_judge(prompt)
        result["prompt_id"] = prompt_id
        result["languages_evaluated"] = langs
        return result

    # ── Évaluation de la robustesse ──────────────────────────────────────────

    def score_robustness(
        self,
        prompt_id: str,
        answers_by_lang: dict[str, str],
    ) -> dict:
        """
        Note la robustesse culturelle des réponses à une question *specific*.

        Parameters
        ----------
        prompt_id : str
            Identifiant de la question.
        answers_by_lang : dict[str, str]
            Dictionnaire { "fr": "réponse...", "en": "...", ... }

        Returns
        -------
        dict
            {
                "prompt_id": "1",
                "score": 3,
                "reason": "Responses are roughly consistent but ...",
                "best_response_lang": "en",
                "worst_response_lang": "it",
                "languages_evaluated": ["en", "fr", "de", "es", "it"]
            }
        """
        langs = list(answers_by_lang.keys())
        responses_block = "\n\n".join(
            f"[{lang.upper()}]\n{answers_by_lang[lang]}" for lang in langs
        )

        prompt = _ROBUSTNESS_EVALUATION_PROMPT.format(
            n_languages=len(langs),
            languages=", ".join(langs),
            responses_block=responses_block,
        )

        result = self._call_judge(prompt)
        result["prompt_id"] = prompt_id
        result["languages_evaluated"] = langs
        return result


# ─── Fonctions utilitaires de haut niveau ────────────────────────────────────


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
    data: dict[str, dict[str, str]] = {}
    run_path = Path(output_dir) / run_id

    for lang in languages:
        filepath = run_path / f"{lang}_{dataset_type}.jsonl"
        if not filepath.exists():
            logger.warning("Fichier absent : %s", filepath)
            continue
        lang_data: dict[str, str] = {}
        with jsonlines.open(str(filepath), mode="r") as reader:
            for obj in reader:
                answer = obj.get("answer", "")
                if not answer.strip().startswith("ERROR:") and answer.strip():
                    lang_data[obj["id"]] = answer
        data[lang] = lang_data

    return data


def evaluate_diversity(
    run_id: str,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    languages: Optional[list[str]] = None,
    sample_size: int = 10,
    judge: Optional[LLMJudge] = None,
) -> dict:
    """
    Évalue la diversité culturelle d'un run sur un échantillon de questions.

    Sélectionne ``sample_size`` questions parmi les *unspecific*, envoie
    les réponses multi-langues au LLM juge et agrège les scores.

    Parameters
    ----------
    run_id : str
        Identifiant du run à évaluer.
    output_dir : str
        Répertoire racine des résultats.
    languages : list[str] | None
        Langues à inclure (défaut : toutes les 5 langues).
    sample_size : int
        Nombre de questions à évaluer (défaut: 10).
        Attention : chaque question = 1 appel API au juge.
    judge : LLMJudge | None
        Instance de juge à réutiliser (utile pour partager la même connexion).
        Si None, une nouvelle instance est créée.

    Returns
    -------
    dict
        {
            "metric": "llm_judge_diversity",
            "run_id": "...",
            "sample_size": 10,
            "avg_score": 3.6,
            "score_distribution": {"1": 0, "2": 1, "3": 3, "4": 4, "5": 2},
            "evaluations": [ {...}, ... ]
        }
    """
    if languages is None:
        languages = _SUPPORTED_LANGUAGES

    if judge is None:
        judge = LLMJudge()

    data_by_lang = _load_results_by_language(run_id, "unspecific", output_dir, languages)
    if not data_by_lang:
        raise ValueError(f"Aucun fichier unspecific chargé pour le run : {run_id}")

    available_langs = list(data_by_lang.keys())
    common_ids = sorted(
        set.intersection(*[set(data_by_lang[l].keys()) for l in available_langs])
    )

    # Sélection de l'échantillon : premiers IDs (pour reproductibilité)
    sample_ids = common_ids[:sample_size]

    logger.info(
        "LLM Judge – Diversité : évaluation de %d questions (%d langues)...",
        len(sample_ids),
        len(available_langs),
    )

    evaluations: list[dict] = []
    valid_scores: list[int] = []

    for prompt_id in sample_ids:
        answers = {lang: data_by_lang[lang][prompt_id] for lang in available_langs}
        result = judge.score_diversity(prompt_id, answers)
        evaluations.append(result)
        score = result.get("score", 0)
        if isinstance(score, int) and 1 <= score <= 5:
            valid_scores.append(score)
        logger.info("  ID %s → score diversité = %s", prompt_id, score)

    # Distribution des scores
    distribution = {str(s): sum(1 for x in valid_scores if x == s) for s in range(1, 6)}
    avg_score = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else 0.0

    return {
        "metric": "llm_judge_diversity",
        "run_id": run_id,
        "sample_size": len(sample_ids),
        "n_valid_scores": len(valid_scores),
        "avg_score": avg_score,
        "score_distribution": distribution,
        "languages_evaluated": available_langs,
        "evaluations": evaluations,
    }


def evaluate_robustness(
    run_id: str,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    languages: Optional[list[str]] = None,
    sample_size: int = 10,
    judge: Optional[LLMJudge] = None,
) -> dict:
    """
    Évalue la robustesse culturelle d'un run sur un échantillon de questions.

    Sélectionne ``sample_size`` questions parmi les *specific*, envoie
    les réponses multi-langues au LLM juge et agrège les scores.

    Parameters
    ----------
    run_id : str
        Identifiant du run à évaluer.
    output_dir : str
        Répertoire racine des résultats.
    languages : list[str] | None
        Langues à inclure (défaut : toutes les 5 langues).
    sample_size : int
        Nombre de questions à évaluer (défaut: 10).
    judge : LLMJudge | None
        Instance de juge à réutiliser.

    Returns
    -------
    dict
        {
            "metric": "llm_judge_robustness",
            "run_id": "...",
            "sample_size": 10,
            "avg_score": 4.1,
            "score_distribution": {"1": 0, "2": 0, "3": 2, "4": 5, "5": 3},
            "evaluations": [ {...}, ... ]
        }
    """
    if languages is None:
        languages = _SUPPORTED_LANGUAGES

    if judge is None:
        judge = LLMJudge()

    data_by_lang = _load_results_by_language(run_id, "specific", output_dir, languages)
    if not data_by_lang:
        raise ValueError(f"Aucun fichier specific chargé pour le run : {run_id}")

    available_langs = list(data_by_lang.keys())
    common_ids = sorted(
        set.intersection(*[set(data_by_lang[l].keys()) for l in available_langs])
    )

    sample_ids = common_ids[:sample_size]

    logger.info(
        "LLM Judge – Robustesse : évaluation de %d questions (%d langues)...",
        len(sample_ids),
        len(available_langs),
    )

    evaluations: list[dict] = []
    valid_scores: list[int] = []

    for prompt_id in sample_ids:
        answers = {lang: data_by_lang[lang][prompt_id] for lang in available_langs}
        result = judge.score_robustness(prompt_id, answers)
        evaluations.append(result)
        score = result.get("score", 0)
        if isinstance(score, int) and 1 <= score <= 5:
            valid_scores.append(score)
        logger.info("  ID %s → score robustesse = %s", prompt_id, score)

    distribution = {str(s): sum(1 for x in valid_scores if x == s) for s in range(1, 6)}
    avg_score = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else 0.0

    return {
        "metric": "llm_judge_robustness",
        "run_id": run_id,
        "sample_size": len(sample_ids),
        "n_valid_scores": len(valid_scores),
        "avg_score": avg_score,
        "score_distribution": distribution,
        "languages_evaluated": available_langs,
        "evaluations": evaluations,
    }


# ─── Rapport complet ──────────────────────────────────────────────────────────


def generate_report(
    run_id: str,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    languages: Optional[list[str]] = None,
    sample_size: int = 10,
    save: bool = True,
) -> dict:
    """
    Génère un rapport d'évaluation LLM-as-a-Judge complet pour un run.

    Évalue diversité et robustesse sur un échantillon, agrège les scores,
    et sauvegarde le rapport dans ``data/output/{run_id}/analysis_llm_judge.json``.

    Un seul ``LLMJudge`` est instancié et partagé entre les deux évaluations
    pour économiser les connexions et respecter le rate limit.

    Parameters
    ----------
    run_id : str
        Identifiant du run.
    output_dir : str
        Répertoire racine des résultats.
    languages : list[str] | None
        Langues à inclure.
    sample_size : int
        Nombre de questions évaluées par dimension (défaut: 10).
        Total : 2 × sample_size appels à l'API du juge.
    save : bool
        Si True (défaut), sauvegarde le rapport JSON.

    Returns
    -------
    dict
        Rapport complet avec les détails de chaque évaluation.

    Warning
    -------
    Cette fonction consomme des crédits API Groq (2 × sample_size requêtes).
    Avec sample_size=10, comptez environ 40-50 secondes d'exécution.
    """
    # Instancier le juge une seule fois et le partager
    judge = LLMJudge()

    div_result = evaluate_diversity(
        run_id, output_dir, languages, sample_size, judge=judge
    )
    rob_result = evaluate_robustness(
        run_id, output_dir, languages, sample_size, judge=judge
    )

    # Score global : moyenne simple des deux scores moyens
    avg_div = div_result["avg_score"]
    avg_rob = rob_result["avg_score"]
    global_avg = round((avg_div + avg_rob) / 2, 2) if (avg_div + avg_rob) > 0 else 0.0

    # Interprétation qualitative
    if global_avg >= 4.0:
        interpretation = "Excellent : le modèle montre une forte diversité et robustesse culturelle."
    elif global_avg >= 3.0:
        interpretation = "Satisfaisant : bonne performance globale avec des axes d'amélioration."
    elif global_avg >= 2.0:
        interpretation = "Insuffisant : le modèle échoue sur la diversité ou la robustesse culturelle."
    else:
        interpretation = "Très faible : réponses inadaptées ou incohérentes sur le plan culturel."

    report = {
        "run_id": run_id,
        "analysis_type": "llm_judge",
        "judge_model": judge._model,
        "sample_size_per_dimension": sample_size,
        "diversity": div_result,
        "robustness": rob_result,
        "global_avg_score": global_avg,
        "interpretation": interpretation,
    }

    if save:
        report_path = Path(output_dir) / run_id / "analysis_llm_judge.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Rapport LLM Judge sauvegardé : %s", report_path)

    return report


