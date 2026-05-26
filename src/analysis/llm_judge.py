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
balises ``{_..._}`` avant la réponse. Ce module désactive ce mode
par défaut (``thinking: disabled``) pour des évaluations rapides et
économiques. Passez ``enable_thinking=True`` pour activer le raisonnement.

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
_DEFAULT_JUDGE_MODEL = "qwen/qwen3-32b"   # Remplace llama-3.3-70b-versatile
_GROQ_BASE_URL       = "https://api.groq.com/openai/v1"
_SUPPORTED_LANGUAGES = ["en", "fr", "de", "es", "it"]

# Délai entre chaque appel au juge (Groq free tier ~30 req/min)
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
    Évaluateur LLM-as-a-Judge utilisant l'API Groq (Qwen3-32B par défaut).

    Le modèle Qwen3-32B est utilisé avec une température basse (0.1)
    pour des évaluations reproductibles. Le mode "thinking" est désactivé
    par défaut pour réduire la latence et la consommation de tokens.

    Parameters
    ----------
    model : str
        Nom du modèle Groq à utiliser (défaut: ``qwen/qwen3-32b``).
    api_key : str | None
        Clé API Groq. Si None, utilise la variable d'env ``GROQ_API_KEY``.
    request_delay : float
        Délai en secondes entre chaque requête (défaut: 2.1s pour 30 req/min).
    enable_thinking : bool
        Active le mode raisonnement Qwen3 (balises ändig).
        Défaut False : réponse directe, plus rapide, moins de tokens.
        Mettre True pour une évaluation plus approfondie (~3× plus lent).

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
        enable_thinking: bool = False,
    ) -> None:
        import openai as _openai_pkg

        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Clé API Groq manquante. Définissez GROQ_API_KEY dans votre .env "
                "ou passez-la via le paramètre api_key."
            )

        self._model           = model
        self._delay           = request_delay
        self._enable_thinking = enable_thinking
        self._client = _openai_pkg.OpenAI(
            api_key=resolved_key,
            base_url=_GROQ_BASE_URL,
        )
        logger.info(
            "LLMJudge initialisé – modèle: %s | thinking: %s",
            model, "activé" if enable_thinking else "désactivé",
        )

    # ── Méthode de bas niveau ────────────────────────────────────────────────

    def _call_judge(self, user_prompt: str) -> dict:
        """
        Appelle le LLM juge et parse la réponse JSON.

        Gère automatiquement les balises ändig...
