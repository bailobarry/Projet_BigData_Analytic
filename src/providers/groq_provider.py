"""Provider Groq (OpenAI-compatible) avec rotation multi-cles.

Variables supportees:
- GROQ_API_KEY: une cle unique
- GROQ_API_KEYS: liste de cles (prioritaire si renseignee)
- GROQ_RPM_PER_KEY: limite locale req/min appliquee par cle

Comportement:
- rotation round-robin entre les cles
- limitation locale par cle
- cooldown progressif d'une cle en cas de 429
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import openai as _openai_pkg
from openai import RateLimitError

from src.models.config import GenerationConfig, RunConfig
from src.providers.base import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class _KeySlot:
    api_key: str
    client: _openai_pkg.OpenAI
    last_used_at: float = 0.0
    cooldown_until: float = 0.0
    consecutive_429: int = 0


def _parse_groq_keys() -> list[str]:
    """Parse GROQ_API_KEYS (priority) or fallback to GROQ_API_KEY."""
    raw_multi = os.environ.get("GROQ_API_KEYS", "").strip()
    if raw_multi:
        chunks = raw_multi.replace(";", ",").replace("\n", ",").split(",")
        keys = [c.strip() for c in chunks if c.strip()]
        if keys:
            return keys

    single = os.environ.get("GROQ_API_KEY", "").strip()
    return [single] if single else []


def _mask_key(value: str) -> str:
    if len(value) < 10:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


class GroqProvider(LLMProvider):
    """Provider Groq avec rotation multi-cles et cooldown sur 429."""

    def __init__(self, config: RunConfig) -> None:
        keys = _parse_groq_keys()
        if not keys:
            raise ValueError(
                "Aucune cle Groq definie. Configurez GROQ_API_KEY (1 cle) "
                "ou GROQ_API_KEYS (liste de cles)."
            )

        rpm_env = os.environ.get("GROQ_RPM_PER_KEY", "30").strip()
        try:
            self._rpm_per_key = max(float(rpm_env), 1.0)
        except ValueError:
            self._rpm_per_key = 30.0
        self._min_interval = 60.0 / self._rpm_per_key

        self._model = config.provider.model
        self._slots: list[_KeySlot] = [
            _KeySlot(
                api_key=key,
                client=_openai_pkg.OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1"),
            )
            for key in keys
        ]
        self._next_idx = 0

        logger.info(
            "GroqProvider initialise : model=%s | cles=%d | rpm/cle=%.0f",
            self._model,
            len(self._slots),
            self._rpm_per_key,
        )
        logger.debug("Cles chargees: %s", ", ".join(_mask_key(s.api_key) for s in self._slots))

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_id(self) -> str:
        return self._model

    def _pick_slot_idx(self) -> int:
        now = time.monotonic()
        n = len(self._slots)

        earliest_idx = 0
        earliest_cooldown = float("inf")

        for off in range(n):
            idx = (self._next_idx + off) % n
            slot = self._slots[idx]
            if slot.cooldown_until <= now:
                return idx
            if slot.cooldown_until < earliest_cooldown:
                earliest_cooldown = slot.cooldown_until
                earliest_idx = idx

        sleep_for = max(earliest_cooldown - now, 0.0)
        if sleep_for > 0:
            time.sleep(sleep_for)
        return earliest_idx

    def generate(
        self,
        prompt: str,
        generation: GenerationConfig,
        system_prompt: Optional[str] = None,
    ) -> str:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": generation.temperature,
            "max_tokens": generation.max_tokens,
            "top_p": generation.top_p,
        }

        max_attempts = max(3, len(self._slots) * 3)
        last_429: Optional[RateLimitError] = None

        for _ in range(max_attempts):
            idx = self._pick_slot_idx()
            slot = self._slots[idx]

            now = time.monotonic()
            wait_interval = (slot.last_used_at + self._min_interval) - now
            if wait_interval > 0:
                time.sleep(wait_interval)

            try:
                response = slot.client.chat.completions.create(**kwargs)
                slot.last_used_at = time.monotonic()
                slot.consecutive_429 = 0
                slot.cooldown_until = 0.0
                self._next_idx = (idx + 1) % len(self._slots)
                return (response.choices[0].message.content or "").strip()

            except RateLimitError as exc:
                last_429 = exc
                slot.last_used_at = time.monotonic()
                slot.consecutive_429 += 1
                cooldown = min(120.0, float(2 ** slot.consecutive_429))
                slot.cooldown_until = time.monotonic() + cooldown
                self._next_idx = (idx + 1) % len(self._slots)
                logger.warning(
                    "429 sur cle %s -> cooldown %.0fs (consecutive_429=%d)",
                    _mask_key(slot.api_key),
                    cooldown,
                    slot.consecutive_429,
                )

        if last_429 is not None:
            raise last_429
        raise RuntimeError("Generation Groq echouee sans erreur exploitable.")
