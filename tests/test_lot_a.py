"""
Tests unitaires pour le Lot A – Pipeline backend.

Ce fichier vérifie que tous les composants du pipeline fonctionnent
correctement de manière isolée (sans appel réel aux LLMs).

Organisation :
  - TestSchemas           : validation des modèles de données (entrée / sortie)
  - TestConfig            : validation de la configuration (RunConfig, ProviderConfig…)
  - TestPrompting         : stratégies de prompting et reformulation des prompts
  - TestProviderAbstraction: contrat de la classe abstraite LLMProvider
  - TestFactory           : factory create_provider()
  - TestBaselineConfig    : chargement des fichiers de config JSON réels

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
from src.promptings.system_prompt import apply_full_reformulation, get_strategy_elements
from src.providers.base import LLMProvider


# ── Tests des schémas ───────────────────────────────────────────────────────
# Ces tests vérifient que PromptItem et ResultItem (les modèles Pydantic
# qui représentent une ligne JSONL) se construisent et se sérialisent
# correctement.

class TestSchemas:
    def test_prompt_item(self):
        """Un PromptItem doit stocker correctement l'id et le prompt."""
        item = PromptItem(id="1", prompt="What to serve?")
        assert item.id == "1"
        assert item.prompt == "What to serve?"

    def test_result_item(self):
        """
        Un ResultItem doit stocker id, prompt et answer.
        model_dump() doit produire un dict avec exactement ces 3 clés
        (c'est ce qui sera écrit dans le fichier JSONL de sortie).
        """
        item = ResultItem(id="1", prompt="What to serve?", answer="Pancakes.")
        assert item.answer == "Pancakes."
        d = item.model_dump()
        assert d == {"id": "1", "prompt": "What to serve?", "answer": "Pancakes."}

    def test_prompt_item_from_dict(self):
        """
        On doit pouvoir construire un PromptItem depuis un dict Python,
        comme on le fait lors de la lecture d'un fichier JSONL avec jsonlines.
        """
        data = {"id": "1-5", "prompt": "We live in France..."}
        item = PromptItem(**data)
        assert item.id == "1-5"


# ── Tests de la configuration ───────────────────────────────────────────────
# Ces tests vérifient que les classes de configuration (RunConfig,
# ProviderConfig, GenerationConfig, PipelineConfig) fonctionnent bien :
# valeurs par défaut, chargement/sauvegarde JSON, calcul des fichiers d'entrée.

class TestConfig:
    def test_generation_defaults(self):
        """
        GenerationConfig doit avoir des valeurs par défaut cohérentes :
        température 0 (déterministe), 256 tokens max, seed 42.
        """
        gen = GenerationConfig()
        assert gen.temperature == 0.0
        assert gen.max_tokens == 256
        assert gen.seed == 42

    def test_provider_config_groq(self):
        """ProviderConfig doit accepter le type 'groq' avec le bon modèle."""
        prov = ProviderConfig(type="groq", model="llama-3.3-70b-versatile")
        assert prov.type == "groq"
        assert prov.model == "llama-3.3-70b-versatile"

    def test_provider_config_ollama(self):
        """
        ProviderConfig de type 'ollama' doit avoir un host par défaut
        pointant vers localhost:11434 (port standard d'Ollama).
        """
        prov = ProviderConfig(type="ollama", model="gemma3:12b")
        assert prov.type == "ollama"
        assert prov.model == "gemma3:12b"
        assert prov.ollama_host == "http://localhost:11434"

    def test_run_config_from_file(self, tmp_path):
        """
        RunConfig.from_file() doit charger un fichier JSON et appliquer
        les valeurs par défaut pour les champs non renseignés (ex: temperature).
        """
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
        assert config.generation.temperature == 0.0  # valeur par défaut

    def test_input_files(self):
        """
        input_files() doit retourner la liste des chemins JSONL à traiter,
        en croisant les langues et les types de dataset.
        Ex : 2 langues × 1 type = 2 fichiers.
        """
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
        """
        config.save() doit écrire un fichier config.json lisible,
        et from_file() doit recharger exactement la même configuration.
        """
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
# Ces tests vérifient le module src/promptings/system_prompt.py qui gère
# les 3 stratégies de prompting (cultural_expert, neutral, empathetic_synthesis)
# et la reformulation des prompts (ajout du prefix et du suffix).

class TestPrompting:
    def test_baseline_returns_empty_pack(self):
        """
        En mode baseline (strategy=None), get_strategy_elements() retourne
        un pack vide : pas de system prompt, pas de prefix, pas de suffix.
        C'est le comportement voulu pour la baseline "vanilla".
        """
        pack = get_strategy_elements(None)
        assert pack["system"] == ""
        assert pack["prefix"] == ""
        assert pack["suffix"] == ""

    def test_unknown_strategy_returns_default(self):
        """
        Une stratégie inconnue ne lève pas d'exception : elle retourne
        silencieusement le pack vide (comportement tolérant).
        """
        pack = get_strategy_elements("nonexistent_strategy")
        assert pack["system"] == ""
        assert pack["prefix"] == ""
        assert pack["suffix"] == ""

    def test_known_strategy_en(self):
        """
        La stratégie 'cultural_expert' en anglais doit retourner un pack
        complet avec un system prompt, un prefix et un suffix non vides.
        """
        pack = get_strategy_elements("cultural_expert", lang="en")
        assert len(pack["system"]) > 0
        assert len(pack["prefix"]) > 0
        assert len(pack["suffix"]) > 0

    def test_known_strategy_fr(self):
        """La stratégie 'neutral' doit avoir un system prompt en français."""
        pack = get_strategy_elements("neutral", lang="fr")
        assert len(pack["system"]) > 0

    def test_known_strategy_fallback_to_en(self):
        """
        Si la langue demandée n'est pas définie pour la stratégie,
        le système doit utiliser la version anglaise comme fallback.
        """
        pack_en = get_strategy_elements("cultural_expert", lang="en")
        pack_xx = get_strategy_elements("cultural_expert", lang="xx")
        assert pack_en["system"] == pack_xx["system"]

    def test_all_languages_covered(self):
        """
        Les 5 langues du projet (EN, FR, DE, ES, IT) doivent toutes être
        définies pour chacune des 3 stratégies. Garantit qu'aucune
        combinaison langue × stratégie ne retourne un system prompt vide.
        """
        for strategy in ("cultural_expert", "neutral", "empathetic_synthesis"):
            for lang in ("en", "fr", "de", "es", "it"):
                pack = get_strategy_elements(strategy, lang=lang)
                assert pack["system"] != "", f"system vide pour {strategy}/{lang}"

    def test_apply_reformulation_no_prefix_no_suffix(self):
        """
        Sans prefix ni suffix (mode baseline), apply_full_reformulation()
        retourne le prompt original sans modification.
        """
        prompt = "What to eat?"
        assert apply_full_reformulation(prompt) == prompt

    def test_apply_reformulation_with_prefix(self):
        """
        Avec un prefix, le prompt final = 'prefix + espace + prompt'.
        Le prefix est typiquement une instruction ajoutée avant la question.
        """
        result = apply_full_reformulation("What to eat?", prefix="Answer this:")
        assert result == "Answer this: What to eat?"

    def test_apply_reformulation_with_suffix(self):
        """
        Avec un suffix, le prompt final = 'prompt + espace + suffix'.
        Le suffix est typiquement une consigne de style ajoutée après la question.
        """
        result = apply_full_reformulation("What to eat?", suffix="Be concise.")
        assert result == "What to eat? Be concise."

    def test_apply_reformulation_with_prefix_and_suffix(self):
        """
        Avec prefix ET suffix, les trois parties sont assemblées dans l'ordre :
        'prefix + espace + prompt + espace + suffix'.
        C'est le cas nominal des variantes de prompting (Lot C).
        """
        result = apply_full_reformulation(
            "What to eat?",
            prefix="Please answer:",
            suffix="Keep it short.",
        )
        assert result == "Please answer: What to eat? Keep it short."


# ── Tests du provider (abstraction) ────────────────────────────────────────
# Ces tests vérifient que la classe abstraite LLMProvider impose bien
# son contrat : on ne peut pas l'instancier directement, et un provider
# concret doit implémenter provider_name, model_id et generate().

class TestProviderAbstraction:
    def test_cannot_instantiate_abstract(self):
        """
        LLMProvider est une classe abstraite (ABC).
        Tenter de l'instancier directement doit lever une TypeError.
        """
        with pytest.raises(TypeError):
            LLMProvider()

    def test_concrete_provider(self):
        """
        Un provider concret (ici un mock) doit pouvoir être instancié
        et sa méthode generate() doit retourner une chaîne de caractères.
        Ce test valide que le contrat de la classe abstraite est respecté.
        """

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
# Ces tests vérifient que create_provider() retourne le bon provider
# selon le type spécifié dans la configuration.

class TestFactory:
    def test_unknown_provider_raises(self):
        """
        Passer un type de provider inconnu à create_provider() doit lever
        une ValueError avec un message explicite.
        On utilise model_construct() pour contourner la validation Pydantic
        et forcer un type invalide directement dans la factory.
        """
        from src.providers import create_provider

        config = RunConfig.model_construct(
            run_id="test",
            provider=ProviderConfig.model_construct(type="unknown", model="x"),
            generation=GenerationConfig(),
            pipeline=PipelineConfig(),
        )
        with pytest.raises(ValueError, match="Provider inconnu"):
            create_provider(config)


# ── Tests du chargement de fichiers baseline ────────────────────────────────
# Ces tests chargent les vrais fichiers JSON de configuration pour vérifier
# qu'ils sont valides et cohérents avec ce qu'on attend.

class TestBaselineConfig:
    def test_load_baseline_json(self):
        """
        Le fichier configs/baseline_groq.json doit être valide et contenir
        les paramètres attendus pour un run baseline :
        - température 0 (déterministe)
        - seed 42 (reproductibilité)
        - 5 langues (en, fr, de, es, it)
        - aucun system prompt (mode vanilla)
        """
        baseline = Path("configs/baseline_groq.json")
        if baseline.exists():
            config = RunConfig.from_file(baseline)
            assert config.run_id == "baseline_groq"
            assert config.generation.temperature == 0.0
            assert config.generation.seed == 42
            assert len(config.pipeline.languages) == 5
            assert config.pipeline.system_prompt is None

    def test_load_baseline_ollama(self):
        """
        Le fichier configs/baseline_ollama.json doit pointer sur le bon
        provider (ollama) et le bon modèle (gemma3:12b).
        """
        baseline = Path("configs/baseline_ollama.json")
        if baseline.exists():
            config = RunConfig.from_file(baseline)
            assert config.provider.type == "ollama"
            assert config.provider.model == "gemma3:12b"
