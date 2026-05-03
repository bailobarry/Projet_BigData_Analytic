"""
Tests unitaires pour le Lot A – Pipeline backend.

Lancez avec : pytest tests/ -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models.config import (
    GenerationConfig,
    PipelineConfig,
    ProviderConfig,
    RunConfig,
)
from src.models.schemas import PromptItem, ResultItem
from src.promptings.system_prompt import apply_prompt_template, get_system_prompt
from src.providers.base import LLMProvider


# ── Tests des schémas ───────────────────────────────────────────────────────


class TestSchemas:
    def test_prompt_item(self):
        item = PromptItem(id="1", prompt="What to serve?")
        assert item.id == "1"
        assert item.prompt == "What to serve?"

    def test_result_item(self):
        item = ResultItem(id="1", prompt="What to serve?", answer="Pancakes.")
        assert item.answer == "Pancakes."
        d = item.model_dump()
        assert d == {"id": "1", "prompt": "What to serve?", "answer": "Pancakes."}

    def test_prompt_item_from_dict(self):
        data = {"id": "1-5", "prompt": "We live in France..."}
        item = PromptItem(**data)
        assert item.id == "1-5"


# ── Tests de la configuration ───────────────────────────────────────────────


class TestConfig:
    def test_generation_defaults(self):
        gen = GenerationConfig()
        assert gen.temperature == 0.0
        assert gen.max_tokens == 256
        assert gen.seed == 42

    def test_provider_config_groq(self):
        prov = ProviderConfig(type="groq", model="llama-3.3-70b-versatile")
        assert prov.type == "groq"
        assert prov.model == "llama-3.3-70b-versatile"

    def test_provider_config_ollama(self):
        prov = ProviderConfig(type="ollama", model="gemma3:12b")
        assert prov.type == "ollama"
        assert prov.model == "gemma3:12b"
        assert prov.ollama_host == "http://localhost:11434"

    def test_run_config_from_file(self, tmp_path):
        config_data = {
            "run_id": "test_run",
            "provider": {
                "type": "groq",
                "model": "llama-3.3-70b-versatile",
            },
        }
        config_file = tmp_path / "test_config.json"
        config_file.write_text(json.dumps(config_data))

        config = RunConfig.from_file(config_file)
        assert config.run_id == "test_run"
        assert config.provider.model == "llama-3.3-70b-versatile"
        assert config.generation.temperature == 0.0  # default

    def test_input_files(self):
        config = RunConfig(
            provider=ProviderConfig(type="ollama", model="gemma3:12b"),
            pipeline=PipelineConfig(
                languages=["en", "fr"],
                dataset_types=["unspecific"],
            ),
        )
        files = config.input_files()
        assert len(files) == 2
        assert "en_unspecific.jsonl" in files[0]
        assert "fr_unspecific.jsonl" in files[1]

    def test_save_and_load(self, tmp_path):
        config = RunConfig(
            run_id="save_test",
            provider=ProviderConfig(type="ollama", model="gemma3:12b"),
        )
        saved = config.save(tmp_path)
        assert saved.exists()

        loaded = RunConfig.from_file(saved)
        assert loaded.run_id == "save_test"
        assert loaded.provider.model == "gemma3:12b"


# ── Tests du prompting ──────────────────────────────────────────────────────


class TestPrompting:
    def test_baseline_no_system_prompt(self):
        assert get_system_prompt(None) is None

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="inconnue"):
            get_system_prompt("nonexistent_strategy")

    def test_apply_template_none(self):
        prompt = "What to eat?"
        assert apply_prompt_template(prompt, None) == prompt

    def test_apply_template_with_placeholder(self):
        prompt = "What to eat?"
        result = apply_prompt_template(prompt, "Answer this: {prompt}")
        assert result == "Answer this: What to eat?"


# ── Tests du provider (abstraction) ────────────────────────────────────────


class TestProviderAbstraction:
    def test_cannot_instantiate_abstract(self):
        """Vérifier qu'on ne peut pas instancier LLMProvider directement."""
        with pytest.raises(TypeError):
            LLMProvider()

    def test_concrete_provider(self):
        """Vérifier qu'un provider concret fonctionne."""

        class MockProvider(LLMProvider):
            @property
            def provider_name(self) -> str:
                return "mock"

            @property
            def model_id(self) -> str:
                return "mock-model"

            def generate(self, prompt, generation, system_prompt=None):
                return f"Mock response to: {prompt}"

        provider = MockProvider()
        gen = GenerationConfig()
        result = provider.generate("Hello", gen)
        assert result == "Mock response to: Hello"
        assert provider.provider_name == "mock"
        assert provider.model_id == "mock-model"


# ── Tests de la factory ─────────────────────────────────────────────────────


class TestFactory:
    def test_unknown_provider_raises(self):
        from src.providers import create_provider

        # Utiliser model_construct pour contourner la validation Pydantic
        # et tester directement la factory avec un type inconnu
        config = RunConfig.model_construct(
            run_id="test",
            provider=ProviderConfig.model_construct(type="unknown", model="x"),
            generation=GenerationConfig(),
            pipeline=PipelineConfig(),
        )
        with pytest.raises(ValueError, match="Provider inconnu"):
            create_provider(config)


# ── Tests du chargement de fichiers baseline ────────────────────────────────


class TestBaselineConfig:
    def test_load_baseline_json(self):
        baseline = Path("configs/baseline_groq.json")
        if baseline.exists():
            config = RunConfig.from_file(baseline)
            assert config.run_id == "baseline_groq"
            assert config.generation.temperature == 0.0
            assert config.generation.seed == 42
            assert len(config.pipeline.languages) == 5
            assert config.pipeline.system_prompt is None
            assert config.pipeline.prompt_template is None

    def test_load_baseline_ollama(self):
        baseline = Path("configs/baseline_ollama.json")
        if baseline.exists():
            config = RunConfig.from_file(baseline)
            assert config.provider.type == "ollama"
            assert config.provider.model == "gemma3:12b"

