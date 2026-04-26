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

_SYSTEM_PROMPTS: dict[str, str] = {
    # Variante 1 : Consigne de neutralité
    "neutral": "You are a helpful assistant. Provide a neutral and concise answer in one sentence.",
    
    # Variante 2 : Rôle d'expert
    "cultural_expert": "You are a local cultural expert. Answer the question based strictly on local customs and traditions of the region mentioned.",
    
    # Variante 3 : Contrainte stricte de format
    "short_form": "Answer in exactly one short sentence. Do not repeat the question or provide preamble."
}


def get_system_prompt(strategy: Optional[str] = None) -> Optional[str]:
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
    if strategy not in _SYSTEM_PROMPTS:
        raise ValueError(
            f"Stratégie de system prompt inconnue : '{strategy}'. "
            f"Stratégies disponibles : {list(_SYSTEM_PROMPTS.keys())}"
        )
    return _SYSTEM_PROMPTS[strategy]


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

