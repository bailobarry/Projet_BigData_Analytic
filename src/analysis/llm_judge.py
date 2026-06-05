"""
Évaluation automatique des réponses LLM via un *LLM-as-a-Judge*.
Ce module utilise un LLM (Groq / Qwen3-32B par défaut) pour noter
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
Elle utilise directement le SDK ``openai`` avec l'endpoint Groq.
Note Qwen3
----------
Le modèle ``qwen/qwen3-32b`` supporte un mode "thinking" qui produit des
balises ``<think>...</think>`` avant la réponse JSON. Ce module supprime
automatiquement ces balises avant de parser le JSON.
Fonctions utilitaires
---------------------
evaluate_diversity(run_id, ...)
evaluate_robustness(run_id, ...)
generate_report(run_id, ...)
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
_DEFAULT_OUTPUT_DIR  = "data/output"
_DEFAULT_JUDGE_MODEL = "qwen/qwen3-32b"
_GROQ_BASE_URL       = "https://api.groq.com/openai/v1"
_SUPPORTED_LANGUAGES = ["en", "fr", "de", "es", "it"]
# Délai entre chaque appel au juge.
# Groq limite Qwen3-32B à ~6 000 tokens/min (free tier).
# Chaque appel consomme ~1 500 tokens → max 4 appels/min → 1 appel toutes les 15 s minimum.
_REQUEST_DELAY = 16.0
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
# ─── Helpers ─────────────────────────────────────────────────────────────────
def _parse_retry_after(error_message: str) -> float:
    """Extrait le délai suggéré par Groq dans un message 429/503.
    Exemples reconnus :
      'Please try again in 40ms'  -> 0.04
      'Please try again in 1.5s'  -> 1.5
      'Please try again in 2m30s' -> 150.0
    Retourne 0.0 si rien n'est trouvé.
    """
    total = 0.0
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(ms|m|s)", error_message):
        val, unit = float(m.group(1)), m.group(2)
        if unit == "ms":
            total += val / 1000.0
        elif unit == "s":
            total += val
        elif unit == "m":
            total += val * 60.0
    return total

# ─── Classe principale ───────────────────────────────────────────────────────
class LLMJudge:
    """LLM-as-a-Judge via Groq avec une seule clé API."""
    
    def __init__(
        self,
        model: str = _DEFAULT_JUDGE_MODEL,
        api_key: Optional[str] = None,
        request_delay: float = _REQUEST_DELAY,
        enable_thinking: bool = False,
    ) -> None:
        import openai as _openai_pkg
        
        # Récupérer la clé API
        key = api_key or os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            raise ValueError(
                "Clé API Groq manquante. Configurez GROQ_API_KEY dans votre fichier .env"
            )
        
        self._model = model
        self._delay = request_delay
        self._enable_thinking = enable_thinking
        self._client = _openai_pkg.OpenAI(api_key=key, base_url=_GROQ_BASE_URL)
        
        logger.info(
            "LLMJudge initialisé : model=%s | thinking=%s",
            model,
            "on" if enable_thinking else "off",
        )
    
    def _call_judge(self, user_prompt: str) -> dict:
        """Appelle l'API juge, parse le JSON, gère 429/503 avec backoff."""
        from openai import InternalServerError, RateLimitError
        from openai.types.chat import ChatCompletionMessageParam
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        # max_tokens élevé : Qwen3 génère un long bloc <think>...</think>
        # avant le JSON — il faut assez de place pour les deux.
        call_kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4096,
            "top_p": 1.0,
        }
        max_attempts = 3
        last_error: Optional[Exception] = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.chat.completions.create(**call_kwargs)
                raw = response.choices[0].message.content or ""
                time.sleep(self._delay)
                # Supprimer les blocs <think>...</think> du mode thinking de Qwen3
                raw_clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                json_match = re.search(r"\{.*\}", raw_clean, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                logger.warning("Judge response without valid JSON: %s", raw_clean[:200])
                return {"score": 0, "reason": f"Parsing failed: {raw_clean[:100]}"}
            except RateLimitError as exc:
                last_error = exc
                suggested = _parse_retry_after(str(exc))
                wait = (
                    max(suggested + 1.0, min(60.0, float(attempt) * 5))
                    if suggested
                    else min(60.0, float(attempt) * 5)
                )
                logger.warning(
                    "Rate limit judge (attempt %d/%d), attente %.1fs : %s",
                    attempt, max_attempts, wait, exc,
                )
                time.sleep(wait)
                continue
            except InternalServerError as exc:
                last_error = exc
                if getattr(exc, "status_code", None) == 503:
                    wait = min(120.0, 2 ** attempt)
                    logger.warning(
                        "Judge 503 over capacity (attempt %d/%d), retry dans %.0fs",
                        attempt, max_attempts, wait,
                    )
                    time.sleep(wait)
                    continue
                logger.error(
                    "Judge InternalServerError (attempt %d/%d): %s",
                    attempt, max_attempts, exc,
                )
                time.sleep(self._delay)
                continue
            except Exception as exc:
                last_error = exc
                logger.error(
                    "Judge API error (attempt %d/%d): %s",
                    attempt, max_attempts, exc,
                )
                time.sleep(self._delay)
                continue
        
        return {
            "score": 0,
            "reason": (
                f"API error after {max_attempts} attempts: "
                f"{type(last_error).__name__}: {last_error}"
                if last_error
                else f"API error after {max_attempts} attempts"
            ),
        }
    def score_diversity(self, prompt_id: str, answers_by_lang: dict[str, str]) -> dict:
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
    def score_robustness(self, prompt_id: str, answers_by_lang: dict[str, str]) -> dict:
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
# ─── Fonctions utilitaires ────────────────────────────────────────────────────
def _load_results_by_language(
    run_id: str,
    dataset_type: str,
    output_dir: str,
    languages: list[str],
) -> dict[str, dict[str, str]]:
    data: dict[str, dict[str, str]] = {}
    run_path = Path(output_dir) / run_id
    for lang in languages:
        filepath = run_path / f"{lang}_{dataset_type}.jsonl"
        if not filepath.exists():
            logger.warning("Missing file: %s", filepath)
            continue
        lang_data: dict[str, str] = {}
        with jsonlines.open(str(filepath), mode="r") as reader:
            for obj in reader:
                answer = obj.get("answer", "")
                if answer.strip() and not answer.strip().startswith("ERROR:"):
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
    if languages is None:
        languages = _SUPPORTED_LANGUAGES
    if judge is None:
        judge = LLMJudge()
    data_by_lang = _load_results_by_language(run_id, "unspecific", output_dir, languages)
    if not data_by_lang:
        raise ValueError(f"No unspecific files found for run: {run_id}")
    available_langs = list(data_by_lang.keys())
    common_ids = sorted(set.intersection(*[set(data_by_lang[l].keys()) for l in available_langs]))
    sample_ids = common_ids[:sample_size]
    evaluations: list[dict] = []
    valid_scores: list[int] = []
    for prompt_id in sample_ids:
        answers = {lang: data_by_lang[lang][prompt_id] for lang in available_langs}
        result = judge.score_diversity(prompt_id, answers)
        evaluations.append(result)
        score = result.get("score", 0)
        if isinstance(score, int) and 1 <= score <= 5:
            valid_scores.append(score)
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
    if languages is None:
        languages = _SUPPORTED_LANGUAGES
    if judge is None:
        judge = LLMJudge()
    data_by_lang = _load_results_by_language(run_id, "specific", output_dir, languages)
    if not data_by_lang:
        raise ValueError(f"No specific files found for run: {run_id}")
    available_langs = list(data_by_lang.keys())
    common_ids = sorted(set.intersection(*[set(data_by_lang[l].keys()) for l in available_langs]))
    sample_ids = common_ids[:sample_size]
    evaluations: list[dict] = []
    valid_scores: list[int] = []
    for prompt_id in sample_ids:
        answers = {lang: data_by_lang[lang][prompt_id] for lang in available_langs}
        result = judge.score_robustness(prompt_id, answers)
        evaluations.append(result)
        score = result.get("score", 0)
        if isinstance(score, int) and 1 <= score <= 5:
            valid_scores.append(score)
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
def generate_report(
    run_id: str,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    languages: Optional[list[str]] = None,
    sample_size: int = 10,
    save: bool = True,
) -> dict:
    judge = LLMJudge()
    div_result = evaluate_diversity(run_id, output_dir, languages, sample_size, judge=judge)
    rob_result = evaluate_robustness(run_id, output_dir, languages, sample_size, judge=judge)
    avg_div = div_result["avg_score"]
    avg_rob = rob_result["avg_score"]
    global_avg = round((avg_div + avg_rob) / 2, 2) if (avg_div + avg_rob) > 0 else 0.0
    if global_avg >= 4.0:
        interpretation = "Excellent: strong diversity and robustness."
    elif global_avg >= 3.0:
        interpretation = "Satisfactory: good global performance with room for improvement."
    elif global_avg >= 2.0:
        interpretation = "Insufficient: issues on diversity or robustness."
    else:
        interpretation = "Very weak: culturally inadequate or inconsistent responses."
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
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("LLM Judge report saved: %s", report_path)
    return report


def _auto_detect_dataset_type(
    run_id_a: str,
    run_id_b: str,
    output_dir: str,
    languages: list[str],
) -> str:
    """
    Détecte automatiquement le meilleur type de dataset à utiliser pour la comparaison.
    
    Stratégie de détection :
    1. Si les deux runs ont des fichiers unspecific et specific : privilégier unspecific (diversité)
    2. Si les deux runs ont uniquement unspecific : utiliser unspecific
    3. Si les deux runs ont uniquement specific : utiliser specific
    4. Si un run a unspecific et l'autre specific : utiliser le type commun prioritaire
    
    Parameters
    ----------
    run_id_a : str
        Premier run.
    run_id_b : str
        Second run.
    output_dir : str
        Répertoire contenant les résultats.
    languages : list[str]
        Langues à vérifier.
    
    Returns
    -------
    str
        Type de dataset détecté ('unspecific' ou 'specific').
    
    Raises
    ------
    ValueError
        Si aucun type commun n'est trouvé.
    """
    run_a_path = Path(output_dir) / run_id_a
    run_b_path = Path(output_dir) / run_id_b
    
    # Vérifier quels types de fichiers existent pour chaque run
    has_unspecific_a = any((run_a_path / f"{lang}_unspecific.jsonl").exists() for lang in languages)
    has_specific_a = any((run_a_path / f"{lang}_specific.jsonl").exists() for lang in languages)
    
    has_unspecific_b = any((run_b_path / f"{lang}_unspecific.jsonl").exists() for lang in languages)
    has_specific_b = any((run_b_path / f"{lang}_specific.jsonl").exists() for lang in languages)
    
    logger.info(
        "Détection type dataset : Run A (unspecific=%s, specific=%s) | Run B (unspecific=%s, specific=%s)",
        has_unspecific_a, has_specific_a, has_unspecific_b, has_specific_b
    )
    
    # Déterminer les types communs
    common_unspecific = has_unspecific_a and has_unspecific_b
    common_specific = has_specific_a and has_specific_b
    
    # Stratégie de sélection
    if common_unspecific and common_specific:
        # Les deux ont les deux types : privilégier unspecific (diversité culturelle)
        logger.info("Les deux runs ont unspecific et specific → choix: unspecific (diversité)")
        return "unspecific"
    elif common_unspecific:
        logger.info("Type commun détecté : unspecific")
        return "unspecific"
    elif common_specific:
        logger.info("Type commun détecté : specific")
        return "specific"
    else:
        # Aucun type commun
        error_msg = (
            f"Impossible de comparer les runs '{run_id_a}' et '{run_id_b}' : "
            f"aucun type de dataset commun trouvé. "
            f"Run A : {'unspecific' if has_unspecific_a else ''}{',' if has_unspecific_a and has_specific_a else ''}"
            f"{'specific' if has_specific_a else '' if has_specific_a or has_unspecific_a else 'aucun fichier'} | "
            f"Run B : {'unspecific' if has_unspecific_b else ''}{',' if has_unspecific_b and has_specific_b else ''}"
            f"{'specific' if has_specific_b else '' if has_specific_b or has_unspecific_b else 'aucun fichier'}"
        )
        raise ValueError(error_msg)


def compare_runs_llm_judge(
    run_id_a: str,
    run_id_b: str,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    languages: Optional[list[str]] = None,
    sample_size: int = 5,
    dataset_type: Optional[str] = None,
) -> dict:
    """
    Compare deux runs en utilisant le LLM Judge.
    
    Le juge évalue les différences de qualité entre les réponses des deux runs
    pour les mêmes prompts, sur les dimensions diversité et robustesse.
    
    Parameters
    ----------
    run_id_a : str
        Run de référence (baseline).
    run_id_b : str
        Run à comparer (variante).
    output_dir : str
        Répertoire contenant les résultats.
    languages : list[str] | None
        Langues à analyser (défaut: toutes les langues supportées).
    sample_size : int
        Nombre de prompts à comparer (défaut: 5 car chaque comparaison est coûteuse).
    dataset_type : str | None
        Type de dataset à analyser ('unspecific' ou 'specific').
        Si None (défaut), détecte automatiquement le meilleur type disponible.
    
    Returns
    -------
    dict
        Résultats de la comparaison avec scores et justifications du juge.
    """
    if languages is None:
        languages = _SUPPORTED_LANGUAGES
    
    # Détection automatique du type de dataset si non spécifié
    if dataset_type is None:
        from src.analysis.dataset_detection import get_dataset_type_for_llm_judge
        dataset_type = get_dataset_type_for_llm_judge(run_id_a, run_id_b, output_dir, languages)
        logger.info(
            "Type de dataset auto-détecté : %s (basé sur les fichiers disponibles)",
            dataset_type
        )
    
    judge = LLMJudge()
    
    # Charger les résultats des deux runs
    data_a = _load_results_by_language(run_id_a, dataset_type, output_dir, languages)
    data_b = _load_results_by_language(run_id_b, dataset_type, output_dir, languages)
    
    if not data_a or not data_b:
        raise ValueError(f"Impossible de charger les fichiers {dataset_type} pour l'un des runs")
    
    # Trouver les langues communes
    common_langs = sorted(set(data_a.keys()) & set(data_b.keys()))
    if not common_langs:
        raise ValueError("Aucune langue commune entre les deux runs")
    
    # Trouver les IDs communs à toutes les langues des deux runs
    common_ids = sorted(
        set.intersection(*[set(data_a[l].keys()) for l in common_langs]) &
        set.intersection(*[set(data_b[l].keys()) for l in common_langs])
    )
    
    if not common_ids:
        raise ValueError("Aucun prompt commun entre les deux runs")
    
    sample_ids = common_ids[:sample_size]
    logger.info(
        "Comparaison LLM Judge : %d prompts, %d langues (%s)",
        len(sample_ids), len(common_langs), ", ".join(common_langs)
    )
    
    comparisons = []
    
    for prompt_id in sample_ids:
        # Construire le bloc de comparaison pour ce prompt
        comp_lines = [f"Prompt ID: {prompt_id}\n"]
        comp_lines.append("[RUN A - Baseline]")
        for lang in common_langs:
            answer_a = data_a[lang].get(prompt_id, "(missing)")
            comp_lines.append(f"  [{lang.upper()}] {answer_a}")
        comp_lines.append("\n[RUN B - Variant]")
        for lang in common_langs:
            answer_b = data_b[lang].get(prompt_id, "(missing)")
            comp_lines.append(f"  [{lang.upper()}] {answer_b}")
        
        comparison_block = "\n".join(comp_lines)
        
        # Prompt spécifique selon le type de dataset
        if dataset_type == "unspecific":
            dimension_desc = "cultural diversity (different perspectives across languages)"
        else:
            dimension_desc = "cultural robustness (consistency across cultural contexts)"
        
        user_prompt = f"""You are comparing two LLM runs (A = baseline, B = variant) on the same prompt.

Below are the responses from both runs in {len(common_langs)} languages: {', '.join(common_langs)}.

{comparison_block}

Evaluate which run produces better responses focusing on: {dimension_desc}

Return ONLY a JSON object in this exact format:
{{
  "winner": "<A or B or TIE>",
  "score_diff": <integer -3 to +3, where -3 = A much better, 0 = tie, +3 = B much better>,
  "reason": "<one clear sentence explaining the key quality difference>",
  "dimension": "<diversity or robustness or quality>"
}}"""
        
        result = judge._call_judge(user_prompt)
        result["prompt_id"] = prompt_id
        comparisons.append(result)
    
    # Calculer les statistiques globales
    valid_scores = [c.get("score_diff", 0) for c in comparisons if isinstance(c.get("score_diff"), (int, float))]
    avg_score_diff = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else 0.0
    
    winners_count = {"A": 0, "B": 0, "TIE": 0}
    for c in comparisons:
        winner = c.get("winner", "TIE")
        if winner in winners_count:
            winners_count[winner] += 1
    
    # Déterminer le gagnant global
    if winners_count["A"] > winners_count["B"]:
        overall_winner = "A"
    elif winners_count["B"] > winners_count["A"]:
        overall_winner = "B"
    else:
        overall_winner = "TIE"
    
    # Résumé
    if overall_winner == "A":
        summary = f"Run A (baseline) is better: {winners_count['A']} wins vs {winners_count['B']} for Run B. Average score difference: {avg_score_diff:.2f}"
    elif overall_winner == "B":
        summary = f"Run B (variant) is better: {winners_count['B']} wins vs {winners_count['A']} for Run A. Average score difference: {avg_score_diff:.2f}"
    else:
        summary = f"Both runs are comparable: {winners_count['A']} wins for A, {winners_count['B']} for B, {winners_count['TIE']} ties. Average score difference: {avg_score_diff:.2f}"
    
    return {
        "run_id_a": run_id_a,
        "run_id_b": run_id_b,
        "dataset_type": dataset_type,
        "sample_size": len(sample_ids),
        "languages": common_langs,
        "judge_model": judge._model,
        "comparisons": comparisons,
        "overall_winner": overall_winner,
        "winners_count": winners_count,
        "avg_score_diff": avg_score_diff,
        "summary": summary,
    }






