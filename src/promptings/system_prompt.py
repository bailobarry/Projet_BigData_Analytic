from __future__ import annotations
from typing import Optional

# ── Registre des stratégies complètes (Lot C) ───────────────────────────

_STRATEGIES: dict[str, dict[str, dict[str, str]]] = {
    "cultural_expert": {
        "fr": {
            "system": "Vous êtes un expert de la culture locale, des traditions et de l'histoire régionale.",
            "prefix": "En tant qu'expert local, veuillez répondre à la question suivante :",
            "suffix": "Répondez en vous basant exclusivement sur les coutumes locales."
        },
        "en": {
            "system": "You are an expert in local culture, traditions, and regional history.",
            "prefix": "As a local expert, please answer the following question:",
            "suffix": "Answer based exclusively on local customs."
        },
        "de": {
            "system": "Sie sind Experte für lokale Kultur, Traditionen und Regionalgeschichte.",
            "prefix": "Als lokaler Experte beantworten Sie bitte die folgende Frage:",
            "suffix": "Antworten Sie ausschließlich auf der Grundlage lokaler Bräuche."
        },
        "es": {
            "system": "Eres un experto en cultura local, tradiciones e historia regional.",
            "prefix": "Como experto local, por favor responda a la siguiente pregunta:",
            "suffix": "Responda basándose exclusivamente en las costumbres locales."
        },
        "it": {
            "system": "Sei un esperto di cultura locale, tradizioni e storia regionale.",
            "prefix": "In qualità di esperto locale, rispondi alla seguente domanda:",
            "suffix": "Rispondi basandoti esclusivamente sulle usanze locali."
        }
    },
    "empathetic_synthesis": {
        "fr": {
            "system": "Vous êtes un conseiller attentif qui comprend l'humain derrière chaque question.",
            "prefix": "Prenez en compte la sensibilité de la situation suivante :",
            "suffix": "Donnez un conseil bienveillant et juste."
        },
        "en": {
            "system": "You are a thoughtful advisor who understands the human element behind every question.",
            "prefix": "Consider the sensitive nature of the following situation:",
            "suffix": "Give a kind and fair piece of advice."
        },
        "de": {
            "system": "Sie sind ein aufmerksamer Berater, der das Menschliche hinter jeder Frage versteht.",
            "prefix": "Berücksichtigen Sie die Sensibilität der folgenden Situation:",
            "suffix": "Geben Sie einen wohlwollenden und fairen Rat."
        },
        "es": {
            "system": "Eres un asesor atento que comprende el factor humano detrás de cada pregunta.",
            "prefix": "Tenga en cuenta la sensibilidad de la siguiente situación:",
            "suffix": "Dé un consejo amable y justo."
        },
        "it": {
            "system": "Sei un consulente premuroso che comprende l'elemento umano dietro ogni domanda.",
            "prefix": "Considera la natura sensibile della seguente situazione:",
            "suffix": "Dai un consiglio gentile e giusto."
        }
    }
}


def get_strategy_elements(strategy_name: Optional[str], lang: str = "en") -> dict[str, str]:
    """
    Récupère le pack complet (system, prefix, suffix) pour la stratégie et la langue données.
    """
    default_pack = {"system": "", "prefix": "", "suffix": ""}
    
    if not strategy_name or strategy_name not in _STRATEGIES:
        return default_pack
        
    return _STRATEGIES[strategy_name].get(lang, _STRATEGIES[strategy_name]["en"])

def apply_full_reformulation(prompt: str, prefix: str = "", suffix: str = "") -> str:
    """
    Combine les éléments pour créer le prompt utilisateur final.
    """
    parts = []
    if prefix: parts.append(prefix)
    parts.append(prompt)
    if suffix: parts.append(suffix)
    return " ".join(parts).strip()