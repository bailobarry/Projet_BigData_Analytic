"""
Stratégies de prompting et de reformulation.

Ce module fournit les fonctions utilisées par le pipeline pour :
1. Obtenir un *system prompt* optionnel (consigne globale),
2. Transformer le prompt utilisateur avant envoi au LLM.

**Baseline** : aucun system prompt, aucune transformation du prompt.

Le **Lot C** ajoutera ici des stratégies supplémentaires (ex: role-play,
chain-of-thought, paraphrase contrôlée, etc.).
"""

from __future__ import annotations

from typing import Optional


# ── Registre des stratégies de system prompt ────────────────────────────────

_SYSTEM_PROMPTS: dict[str, dict[str, str]] = {
    # VARIANTE 1 : Neutralité (Objectif Robustesse)
    "neutral": {
        "en": "You are a helpful assistant. Provide a neutral and concise answer.",
        "fr": "Vous êtes un assistant utile. Fournissez une réponse neutre et concise.",
        "de": "Sie sind ein hilfreicher Assistent. Geben Sie eine neutrale und knappe Antwort.",
        "es": "Eres un asistente útil. Proporciona una respuesta neutra y concisa.",
        "it": "Sei un assistente utile. Fornisci una risposta neutrale e concisa."
    },
    
    # VARIANTE 2 : Expert Culturel (Objectif Diversité)
    "cultural_expert": {
        "en": "You are a local cultural expert. Answer the question based strictly on local customs and traditions.",
        "fr": "Vous êtes un expert culturel local. Répondez à la question en vous basant strictement sur les coutumes et traditions locales.",
        "de": "Sie sind ein lokaler Kulturexperte. Beantworten Sie die Frage ausschließlich auf der Grundlage lokaler Bräuche und Traditionen.",
        "es": "Eres un experto cultural local. Responde a la pregunta basándote estrictamente en las costumbres y tradiciones locales.",
        "it": "Sei un esperto culturale locale. Rispondi alla domanda basandoti rigorosamente sui costumi e le tradizioni locali."
    },

    # VARIANTE 3 : Format Court (Objectif Contrainte de Style)
    "short_form": {
        "en": "Answer in exactly one short sentence. Do not repeat the question.",
        "fr": "Répondez en une seule phrase courte. Ne répétez pas la question.",
        "de": "Antworten Sie in genau einem kurzen Satz. Wiederholen Sie die Frage nicht.",
        "es": "Responde en exactamente una oración corta. No repitas la pregunta.",
        "it": "Rispondi in una sola frase breve. Non ripetere la domanda."
    }
}


def get_system_prompt(strategy: Optional[str] = None, lang: str = "en") -> Optional[str]:
    """
    Retourne le system prompt correspondant à la stratégie demandée.

    Parameters
    ----------
    strategy : str | None
        Nom de la stratégie. ``None`` → baseline (pas de system prompt).

    Returns
    -------
    str | None
        Le system prompt, ou ``None`` pour la baseline vanilla.
    """
    if strategy is None:
        return None
        
    strategy_dict = _SYSTEM_PROMPTS.get(strategy)
    if not strategy_dict:
        raise ValueError(f"Stratégie inconnue : {strategy}")
    
    # On renvoie la langue demandée, sinon l'anglais par défaut
    return strategy_dict.get(lang, strategy_dict["en"])


# ── Registre des transformations de prompt ──────────────────────────────────


def apply_prompt_template(
    prompt: str,
    template: Optional[str] = None,
) -> str:
 
    if template is None:
        return prompt.strip()
    
    try:
        return template.format(prompt=prompt)
    except KeyError:
        return f"{template} {prompt}"

