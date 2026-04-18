# 📚 Guide Complet pour Réécrire le Lot A de Zéro

Ce document explique **chaque ligne de code** pour que vous puissiez tout réécrire sans aide.

---

## TABLE DES MATIÈRES

1. [Concepts Python essentiels](#1-concepts-python-essentiels)
2. [Fichier schemas.py](#2-fichier-schemaspy)
3. [Fichier config.py](#3-fichier-configpy)
4. [Fichier base.py](#4-fichier-basepy)
5. [Fichier __init__.py (providers)](#5-fichier-initpy-providers)
6. [Fichier gemini_provider.py](#6-fichier-gemini_providerpy)
7. [Fichier mistral_nemo_provider.py](#7-fichier-mistral_nemo_providerpy)
8. [Fichier system_prompt.py](#8-fichier-system_promptpy)
9. [Fichier logs.py](#9-fichier-logspy)
10. [Fichier runner.py](#10-fichier-runnerpy)
11. [Fichier routes.py (API)](#11-fichier-routespy)
12. [Fichier run_baseline.py](#12-fichier-run_baselinepy)

---

## 1. CONCEPTS PYTHON ESSENTIELS

Avant de comprendre le code, voici les concepts utilisés :

### 1.1 Pydantic (BaseModel)
```python
from pydantic import BaseModel, Field

class Personne(BaseModel):
    nom: str                           # Champ obligatoire de type string
    age: int = 25                      # Champ optionnel avec valeur par défaut
    email: str = Field(..., description="Email")  # ... = obligatoire
```
- `BaseModel` = classe qui valide automatiquement les données
- `Field(...)` = champ obligatoire avec métadonnées
- `Field(default_value)` = champ optionnel

### 1.2 Classes abstraites (ABC)
```python
from abc import ABC, abstractmethod

class Animal(ABC):           # ABC = Abstract Base Class
    @abstractmethod          # Cette méthode DOIT être implémentée par les enfants
    def parler(self) -> str:
        ...                  # ... = pas d'implémentation ici

class Chien(Animal):         # Hérite de Animal
    def parler(self) -> str: # DOIT implémenter parler()
        return "Wouf!"
```

### 1.3 Décorateur @property
```python
class Voiture:
    def __init__(self):
        self._marque = "Toyota"
    
    @property                    # Transforme une méthode en attribut
    def marque(self) -> str:
        return self._marque

v = Voiture()
print(v.marque)  # Pas de parenthèses ! Comme un attribut
```

### 1.4 Type hints
```python
def saluer(nom: str) -> str:           # Prend un str, retourne un str
    return f"Bonjour {nom}"

from typing import Optional, List
def foo(x: Optional[str] = None):      # str ou None, défaut = None
    pass
def bar(items: List[str]):             # Liste de strings
    pass
```

### 1.5 Import conditionnel (TYPE_CHECKING)
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # Ce bloc ne s'exécute JAMAIS
    from module import ClasseLourde     # Sert uniquement pour les annotations

def foo(x: "ClasseLourde"):             # Les guillemets car pas importé vraiment
    pass
```

---

## 2. FICHIER `schemas.py`

**📍 Chemin** : `src/models/schemas.py`

### Pourquoi ce fichier ?

Sans ce fichier, vous manipuleriez des dictionnaires Python partout :
```python
# SANS Pydantic (mauvais) :
item = {"id": "1", "prompt": "Que manger ?"}
# Problème : rien n'empêche d'écrire item["idd"] (faute de frappe)
# ou d'oublier un champ, ou de mettre un entier au lieu d'un string
```

Avec Pydantic, Python **vérifie automatiquement** que les données sont correctes :
```python
# AVEC Pydantic (bon) :
item = PromptItem(id="1", prompt="Que manger ?")  # ✅ OK
item = PromptItem(id=123, prompt="Que manger ?")  # ❌ Erreur : id doit être str
item = PromptItem(id="1")  # ❌ Erreur : prompt manquant
```

### À quoi ça sert ?

1. **Valider les données d'entrée** : quand on lit un fichier JSONL, on s'assure que chaque ligne a bien `id` et `prompt`
2. **Structurer les données de sortie** : quand on écrit, on est sûr d'avoir `id`, `prompt` et `answer`
3. **Documenter le format** : en lisant la classe, on sait exactement quels champs existent

### Comment c'est utilisé ?

```python
# Dans runner.py - LECTURE :
with jsonlines.open("data/input/fr_unspecific.jsonl") as reader:
    for obj in reader:  # obj = {"id": "1", "prompt": "..."}
        item = PromptItem(**obj)  # Valide et convertit en objet
        print(item.id)  # Accès propre aux attributs

# Dans runner.py - ÉCRITURE :
result = ResultItem(id="1", prompt="...", answer="...")
writer.write(result.model_dump())  # Convertit en dict pour écrire
```

### Ce qui se passe sans ce fichier

Si vous n'aviez pas ces classes :
- Vous risquez des erreurs silencieuses (champ manquant découvert trop tard)
- Le code est moins lisible (on ne sait pas quels champs existent)
- Pas d'autocomplétion dans l'IDE

**But** : Définir la structure des données (questions et réponses).

```python
"""
Docstring : explique ce que fait le fichier.
"""

from __future__ import annotations
# Permet d'utiliser les annotations de type comme "str | None" (Python 3.10+)

from pydantic import BaseModel, Field
# BaseModel = classe de base Pydantic pour la validation
# Field = pour ajouter des métadonnées aux champs


class PromptItem(BaseModel):
    """Une question du fichier d'entrée."""
    
    id: str = Field(..., description="Identifiant unique")
    # ... = obligatoire
    # Le type est str
    # description = documentation
    
    prompt: str = Field(..., description="Texte de la question")
    # Pareil : champ obligatoire de type str


class ResultItem(BaseModel):
    """Une question + sa réponse (fichier de sortie)."""
    
    id: str = Field(..., description="Identifiant")
    prompt: str = Field(..., description="Question originale")
    answer: str = Field(..., description="Réponse du LLM")
    # 3 champs : on garde la question + on ajoute la réponse
```

**Comment l'utiliser :**
```python
# Créer un objet
item = PromptItem(id="1", prompt="Que manger ?")

# Convertir en dictionnaire
d = item.model_dump()  # {"id": "1", "prompt": "Que manger ?"}

# Créer depuis un dictionnaire
data = {"id": "2", "prompt": "Bonjour"}
item2 = PromptItem(**data)  # ** = décompresse le dict
```

---

## 3. FICHIER `config.py`

**📍 Chemin** : `src/models/config.py`

### Pourquoi ce fichier ?

Imaginez que vous lancez une expérience avec ces paramètres :
- Modèle : Gemini 2.0 Flash
- Température : 0.0
- Langues : français seulement

3 mois plus tard, votre prof vous demande : *"Comment avez-vous obtenu ces résultats ?"*

**Sans config sauvegardée** : vous ne vous souvenez plus des paramètres → impossible de reproduire.

**Avec RunConfig** : chaque run sauvegarde automatiquement un `config.json` → vous pouvez tout refaire à l'identique.

### À quoi ça sert ?

1. **Centraliser tous les paramètres** : au lieu d'avoir des variables éparpillées partout, tout est dans UN objet
2. **Garantir la reproductibilité** : la config est sauvegardée avec les résultats
3. **Faciliter les variantes** : pour tester un autre modèle, on change juste le JSON
4. **Valider les paramètres** : Pydantic vérifie que temperature est un float, que languages est une liste, etc.

### Comment c'est utilisé ?

```python
# 1. CHARGER une config existante
config = RunConfig.from_file("configs/baseline_gimini.json")

# 2. ACCÉDER aux paramètres
print(config.provider.model)          # "gemini-2.0-flash"
print(config.generation.temperature)  # 0.0
print(config.pipeline.languages)      # ["en", "fr", "de", "es", "ru"]

# 3. LISTER les fichiers à traiter
files = config.input_files()
# → ["data/input/en_specific.jsonl", "data/input/en_unspecific.jsonl", ...]

# 4. SAUVEGARDER (automatique dans le pipeline)
config.save()  # Écrit dans data/output/{run_id}/config.json

# 5. PASSER au pipeline
summary = run_pipeline(config)
```

### Pourquoi 4 classes imbriquées ?

```
RunConfig                    # La config complète
├── ProviderConfig           # QUEL LLM utiliser (Gemini ou Mistral-Nemo)
├── GenerationConfig         # COMMENT générer (température, longueur)
└── PipelineConfig           # QUOI traiter (langues, fichiers, délai)
```

Ça permet de :
- Regrouper les paramètres par thème
- Avoir des valeurs par défaut sensées pour chaque groupe
- Faciliter la lecture du JSON

**But** : Stocker TOUS les paramètres d'un run pour pouvoir le reproduire.

```python
from __future__ import annotations

from datetime import datetime, timezone
# datetime = manipuler les dates
# timezone.utc = fuseau horaire UTC

from pathlib import Path
# Path = manipuler les chemins de fichiers de façon portable

from typing import List, Optional
# List = liste typée
# Optional = peut être None

from uuid import uuid4
# uuid4() = génère un identifiant unique aléatoire

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
# SOUS-MODÈLE 1 : Configuration du provider (quel LLM utiliser)
# ═══════════════════════════════════════════════════════════════

class ProviderConfig(BaseModel):
    """Comment se connecter au LLM."""
    
    type: str = Field(...)
    # "openai_compatible" pour Gemini
    # "ollama" pour Mistral-Nemo local
    
    model: str = Field(...)
    # Nom du modèle : "gemini-2.0-flash" ou "mistral-nemo"
    
    base_url: Optional[str] = Field(None)
    # URL de l'API, None si pas besoin
    # Ex: "https://generativelanguage.googleapis.com/v1beta/openai/"
    
    api_key_env: Optional[str] = Field(None)
    # Nom de la variable d'environnement contenant la clé
    # Ex: "GEMINI_API_KEY" (pas la clé elle-même !)
    
    provider_label: str = Field("custom")
    # Nom lisible : "google", "ollama"
    
    ollama_host: Optional[str] = Field("http://localhost:11434")
    # Adresse du serveur Ollama (par défaut : local)


# ═══════════════════════════════════════════════════════════════
# SOUS-MODÈLE 2 : Paramètres de génération du texte
# ═══════════════════════════════════════════════════════════════

class GenerationConfig(BaseModel):
    """Comment le LLM génère sa réponse."""
    
    temperature: float = Field(0.0)
    # 0.0 = toujours la même réponse (déterministe)
    # 1.0 = réponses variées (créatif)
    
    max_tokens: int = Field(256)
    # Longueur maximale de la réponse en tokens (~= mots)
    
    top_p: float = Field(1.0)
    # Nucleus sampling (1.0 = tout le vocabulaire)
    
    seed: Optional[int] = Field(42)
    # Graine aléatoire pour reproductibilité
    # Avec le même seed, même question = même réponse


# ═══════════════════════════════════════════════════════════════
# SOUS-MODÈLE 3 : Paramètres du pipeline
# ═══════════════════════════════════════════════════════════════

class PipelineConfig(BaseModel):
    """Quels fichiers traiter et comment."""
    
    input_dir: str = Field("data/input")
    # Où sont les fichiers de questions
    
    output_dir: str = Field("data/output")
    # Où écrire les résultats
    
    languages: List[str] = Field(default=["en", "fr", "de", "es", "ru"])
    # Les 5 langues à traiter
    
    dataset_types: List[str] = Field(default=["specific", "unspecific"])
    # Les 2 types de datasets
    
    request_delay: float = Field(2.1)
    # Pause entre chaque appel API (en secondes)
    # Évite de dépasser le quota
    
    system_prompt: Optional[str] = Field(None)
    # Consigne système (None = baseline sans consigne)
    
    prompt_template: Optional[str] = Field(None)
    # Template de reformulation (None = question telle quelle)


# ═══════════════════════════════════════════════════════════════
# MODÈLE PRINCIPAL : La configuration complète d'un run
# ═══════════════════════════════════════════════════════════════

class RunConfig(BaseModel):
    """Tout ce qui définit un run."""
    
    run_id: str = Field(
        default_factory=lambda: f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    )
    # default_factory = fonction appelée pour créer la valeur par défaut
    # Génère un ID comme : "run_20260412_143052_a1b2c3"
    #   - datetime.now(timezone.utc) = date/heure actuelle en UTC
    #   - strftime('%Y%m%d_%H%M%S') = formatte en "20260412_143052"
    #   - uuid4().hex[:6] = 6 premiers caractères d'un UUID aléatoire
    
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # Date de création au format ISO : "2026-04-12T14:30:52+00:00"
    
    description: str = Field("")
    # Description libre du run
    
    provider: ProviderConfig
    # PAS de valeur par défaut = OBLIGATOIRE
    
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    # Valeur par défaut = un GenerationConfig avec ses propres défauts
    
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    # Idem
    
    # ─────────────────────────────────────────────────────────
    # MÉTHODES UTILITAIRES
    # ─────────────────────────────────────────────────────────
    
    def input_files(self) -> List[str]:
        """Liste les fichiers JSONL à traiter."""
        files = []
        for lang in self.pipeline.languages:        # ["en", "fr", ...]
            for ds_type in self.pipeline.dataset_types:  # ["specific", "unspecific"]
                filename = f"{lang}_{ds_type}.jsonl"     # "en_specific.jsonl"
                filepath = str(Path(self.pipeline.input_dir) / filename)
                # Path(...) / filename = concatène les chemins proprement
                files.append(filepath)
        return files
        # Retourne : ["data/input/en_specific.jsonl", "data/input/en_unspecific.jsonl", ...]
    
    def output_path(self) -> Path:
        """Retourne le dossier de sortie pour ce run."""
        return Path(self.pipeline.output_dir) / self.run_id
        # Ex: Path("data/output/baseline_gemini_flash")
    
    def save(self, directory: Optional[Path] = None) -> Path:
        """Sauvegarde la config en JSON."""
        if directory is None:
            directory = self.output_path()
        
        directory.mkdir(parents=True, exist_ok=True)
        # Crée le dossier et tous ses parents si nécessaire
        # exist_ok=True = pas d'erreur si existe déjà
        
        filepath = directory / "config.json"
        filepath.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        # model_dump_json() = convertit l'objet en JSON string
        # indent=2 = JSON indenté avec 2 espaces
        
        return filepath
    
    @classmethod
    def from_file(cls, path: str | Path) -> "RunConfig":
        """Charge une config depuis un fichier JSON."""
        # @classmethod = méthode de classe (pas d'instance nécessaire)
        # cls = la classe elle-même (RunConfig)
        
        raw = Path(path).read_text(encoding="utf-8")
        # Lit le contenu du fichier
        
        return cls.model_validate_json(raw)
        # Parse le JSON et valide contre le schéma Pydantic
```

---

## 4. FICHIER `base.py`

**📍 Chemin** : `src/providers/base.py`

### Pourquoi ce fichier ?

Imaginez que vous écrivez directement dans le pipeline :

```python
# SANS abstraction (mauvais) :
if config.provider.type == "gemini":
    client = OpenAI(api_key=..., base_url=...)
    response = client.chat.completions.create(model=..., messages=...)
    answer = response.choices[0].message.content
elif config.provider.type == "ollama":
    client = ollama.Client(host=...)
    response = client.chat(model=..., messages=...)
    answer = response.message.content
# Et si on ajoute un 3ème provider ? Un 4ème ? Le code devient un chaos de if/elif
```

Avec une classe abstraite :

```python
# AVEC abstraction (bon) :
provider = create_provider(config)  # Crée le bon type automatiquement
answer = provider.generate(prompt, config.generation)
# Le pipeline ne sait PAS et ne SE SOUCIE PAS de quel provider c'est
# Il appelle juste .generate() et ça marche
```

### À quoi ça sert ?

1. **Définir un contrat** : tout provider DOIT avoir `generate()`, `provider_name`, `model_id`
2. **Permettre le polymorphisme** : le pipeline traite tous les providers de la même façon
3. **Faciliter l'ajout de nouveaux providers** : il suffit de créer une classe qui hérite de `LLMProvider`
4. **Éviter les erreurs** : si vous oubliez d'implémenter `generate()`, Python refuse de créer l'objet

### Comment c'est utilisé ?

```python
# Dans __init__.py (factory) :
def create_provider(config) -> LLMProvider:  # Retourne le TYPE ABSTRAIT
    if config.provider.type == "openai_compatible":
        return GeminiProvider(config)  # Mais c'est un type CONCRET
    if config.provider.type == "ollama":
        return MistralNemoProvider(config)

# Dans runner.py (pipeline) :
provider = create_provider(config)  # type: LLMProvider
# Le pipeline ne connaît que LLMProvider, pas GeminiProvider
answer = provider.generate(prompt, config.generation)
# Ça marche car GeminiProvider a implémenté generate()
```

### Ce qui se passe si on n'a pas cette abstraction

```python
# Erreur si on oublie d'implémenter generate() :
class MauvaisProvider(LLMProvider):
    @property
    def provider_name(self): return "test"
    @property
    def model_id(self): return "test"
    # Oups, j'ai oublié generate() !

provider = MauvaisProvider()
# TypeError: Can't instantiate abstract class MauvaisProvider
#            with abstract method generate
```

Python vous **force** à implémenter toutes les méthodes abstraites.

**But** : Définir le contrat (interface) que tout provider LLM doit respecter.

```python
from __future__ import annotations

from abc import ABC, abstractmethod
# ABC = Abstract Base Class (classe qu'on ne peut pas instancier directement)
# abstractmethod = décorateur pour les méthodes à implémenter obligatoirement

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.models.config import GenerationConfig
    # Import uniquement pour les annotations de type (pas exécuté)


class LLMProvider(ABC):
    """Interface commune pour tous les providers LLM."""
    
    # ─────────────────────────────────────────────────────────
    # PROPRIÉTÉS ABSTRAITES
    # ─────────────────────────────────────────────────────────
    
    @property                    # Accessible comme un attribut : provider.provider_name
    @abstractmethod              # DOIT être implémenté par les sous-classes
    def provider_name(self) -> str:
        """Retourne le nom du provider (ex: 'google')."""
        ...                      # ... = pas d'implémentation ici
    
    @property
    @abstractmethod
    def model_id(self) -> str:
        """Retourne le nom du modèle (ex: 'gemini-2.0-flash')."""
        ...
    
    # ─────────────────────────────────────────────────────────
    # MÉTHODE ABSTRAITE PRINCIPALE
    # ─────────────────────────────────────────────────────────
    
    @abstractmethod
    def generate(
        self,
        prompt: str,                              # La question
        generation: "GenerationConfig",           # Paramètres (temp, max_tokens...)
        system_prompt: Optional[str] = None,      # Consigne système optionnelle
    ) -> str:                                     # Retourne la réponse texte
        """
        Envoie une question au LLM et retourne sa réponse.
        """
        ...
```

**Pourquoi une classe abstraite ?**
- On ne peut PAS faire `provider = LLMProvider()` → erreur
- On DOIT créer une sous-classe qui implémente toutes les méthodes abstraites
- Le pipeline appelle `provider.generate()` sans savoir quel provider c'est

---

## 5. FICHIER `__init__.py` (providers)

**📍 Chemin** : `src/providers/__init__.py`

### Pourquoi ce fichier ?

Ce fichier a **2 rôles** :

**Rôle 1 : Transformer le dossier en package Python**
```python
# Sans __init__.py :
from src.providers.base import LLMProvider  # ❌ Erreur : 'providers' n'est pas un module

# Avec __init__.py (même vide) :
from src.providers.base import LLMProvider  # ✅ OK
```

**Rôle 2 : Fournir une fonction "usine" (factory)**
```python
# Sans factory, le code appelant doit connaître tous les providers :
if config.provider.type == "openai_compatible":
    from src.providers.gemini_provider import GeminiProvider
    provider = GeminiProvider(config)
elif config.provider.type == "ollama":
    from src.providers.mistral_nemo_provider import MistralNemoProvider
    provider = MistralNemoProvider(config)
# Répété partout où on crée un provider !

# Avec factory, c'est simple :
from src.providers import create_provider
provider = create_provider(config)  # Une seule ligne
```

### À quoi ça sert ?

1. **Encapsuler la logique de création** : le code appelant ne sait pas comment créer un provider
2. **Point d'entrée unique** : tout passe par `create_provider()`
3. **Import paresseux** : les providers sont importés seulement quand nécessaire (pas au démarrage)
4. **Faciliter les modifications** : pour ajouter un provider, on modifie UN fichier

### Comment c'est utilisé ?

```python
# Dans runner.py :
from src.providers import create_provider

config = RunConfig.from_file("configs/baseline_gimini.json")
provider = create_provider(config)
# → Retourne GeminiProvider si type="openai_compatible"
# → Retourne MistralNemoProvider si type="ollama"

# Le pipeline ne sait pas quel provider c'est, et s'en fiche
answer = provider.generate(prompt, config.generation)
```

### Pourquoi les imports sont à l'intérieur des if ?

```python
if provider_type == "openai_compatible":
    from src.providers.gemini_provider import GeminiProvider  # Import ICI
    return GeminiProvider(config)
```

C'est un **import paresseux (lazy import)** :
- Le module `gemini_provider` n'est chargé que si on utilise Gemini
- Avantage : si Ollama n'est pas installé, ça ne plante pas quand on utilise Gemini
- Gain de temps au démarrage

**But** : Fournir une fonction "usine" qui crée le bon provider selon la config.

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.config import RunConfig

from src.providers.base import LLMProvider  # noqa: F401
# noqa: F401 = dit au linter "je sais que j'importe sans utiliser, c'est voulu"
# On l'importe pour qu'il soit accessible depuis `from src.providers import LLMProvider`


def create_provider(config: "RunConfig") -> LLMProvider:
    """
    Factory pattern : crée le bon provider selon la configuration.
    """
    provider_type = config.provider.type.lower()
    # .lower() = convertit en minuscules pour être flexible
    
    if provider_type == "openai_compatible":
        # Import local (pas au début) = chargé seulement si nécessaire
        from src.providers.gemini_provider import GeminiProvider
        return GeminiProvider(config)
    
    if provider_type == "ollama":
        from src.providers.mistral_nemo_provider import MistralNemoProvider
        return MistralNemoProvider(config)
    
    # Si on arrive ici, le type est inconnu
    raise ValueError(
        f"Provider inconnu : '{provider_type}'. "
        f"Valeurs acceptées : 'openai_compatible', 'ollama'."
    )
```

**Usage :**
```python
from src.providers import create_provider

config = RunConfig.from_file("configs/baseline_gimini.json")
provider = create_provider(config)  # → GeminiProvider ou MistralNemoProvider
answer = provider.generate("Que manger ?", config.generation)
```

---

## 6. FICHIER `gemini_provider.py`

**📍 Chemin** : `src/providers/gemini_provider.py`

### Pourquoi ce fichier ?

C'est l'**implémentation concrète** du contrat `LLMProvider` pour Google Gemini.

Il fait le pont entre :
- Votre code Python (qui appelle `provider.generate("Que manger ?")`)
- L'API de Google (qui reçoit des requêtes HTTP et renvoie du JSON)

### À quoi ça sert ?

1. **Encapsuler la complexité de l'API Google** : le pipeline appelle juste `.generate()`, ce fichier gère tout le reste
2. **Gérer l'authentification** : lit la clé API depuis les variables d'environnement
3. **Construire les requêtes** : formate les messages au format attendu par l'API
4. **Parser les réponses** : extrait le texte de la réponse JSON

### Comment c'est utilisé ?

```python
# Créé par la factory :
provider = create_provider(config)  # → GeminiProvider

# Utilisé par le pipeline :
answer = provider.generate(
    prompt="Que servir au petit-déjeuner ?",
    generation=config.generation,  # temp=0, max_tokens=256, seed=42
    system_prompt=None  # Pas de consigne (baseline)
)
# → "Servez des tartines avec du beurre et de la confiture."
```

### Ce qui se passe en coulisse

Quand vous appelez `provider.generate("Que manger ?")` :

```
1. Construction du message :
   messages = [{"role": "user", "content": "Que manger ?"}]

2. Envoi HTTP à Google :
   POST https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
   Headers: Authorization: Bearer <GEMINI_API_KEY>
   Body: {"model": "gemini-2.0-flash", "messages": [...], "temperature": 0}

3. Réception de la réponse :
   {"choices": [{"message": {"content": "Des tartines..."}}]}

4. Extraction du texte :
   return "Des tartines..."
```

### Pourquoi utiliser le SDK OpenAI pour Google ?

Google a créé un endpoint **compatible avec l'API OpenAI**. Ça veut dire :
- Même format de requêtes/réponses qu'OpenAI
- On peut utiliser le SDK `openai` (très bien maintenu, beaucoup de docs)
- Pas besoin d'apprendre un nouveau SDK

```python
# Au lieu d'utiliser le SDK Google :
import google.generativeai as genai  # SDK spécifique Google

# On utilise le SDK OpenAI avec l'URL de Google :
import openai
client = openai.OpenAI(
    api_key="...",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
# Même code que si on parlait à OpenAI !
```

**But** : Implémenter le provider pour Google Gemini (via API cloud).

```python
from __future__ import annotations

import os
# os.environ = dictionnaire des variables d'environnement

import logging
# logging = système de logs de Python

from typing import Optional

import openai as _openai_pkg
# Le SDK officiel OpenAI (fonctionne aussi avec Gemini)
# _openai_pkg = on le renomme pour éviter confusion avec un éventuel module local

from src.models.config import GenerationConfig, RunConfig
from src.providers.base import LLMProvider

logger = logging.getLogger(__name__)
# __name__ = nom du module courant ("src.providers.gemini_provider")
# Crée un logger nommé pour ce fichier


class GeminiProvider(LLMProvider):
    """Provider pour Google Gemini via l'API compatible OpenAI."""
    
    def __init__(self, config: RunConfig) -> None:
        """
        Constructeur : appelé quand on fait GeminiProvider(config).
        """
        prov = config.provider  # Raccourci
        
        # ─────────────────────────────────────────────────────
        # RÉCUPÉRER LA CLÉ API
        # ─────────────────────────────────────────────────────
        api_key: str | None = None
        
        if prov.api_key_env:
            # prov.api_key_env = "GEMINI_API_KEY" (le NOM de la variable)
            api_key = os.environ.get(prov.api_key_env)
            # os.environ.get("GEMINI_API_KEY") retourne la VALEUR de la variable
            # ou None si elle n'existe pas
            
            if not api_key:
                raise ValueError(
                    f"Variable d'environnement '{prov.api_key_env}' non définie."
                )
        
        # ─────────────────────────────────────────────────────
        # STOCKER LES ATTRIBUTS
        # ─────────────────────────────────────────────────────
        self._model = prov.model          # "gemini-2.0-flash"
        self._label = prov.provider_label # "google"
        
        # ─────────────────────────────────────────────────────
        # CRÉER LE CLIENT OPENAI
        # ─────────────────────────────────────────────────────
        self._client = _openai_pkg.OpenAI(
            api_key=api_key or "dummy",   # La clé API
            base_url=prov.base_url,       # L'URL de l'API Gemini
        )
        # Ce client utilise le protocole OpenAI mais parle à Google
        
        logger.info("GeminiProvider initialisé : model=%s", self._model)
    
    # ─────────────────────────────────────────────────────────
    # PROPRIÉTÉS (implémentent les @abstractmethod de LLMProvider)
    # ─────────────────────────────────────────────────────────
    
    @property
    def provider_name(self) -> str:
        return self._label   # "google"
    
    @property
    def model_id(self) -> str:
        return self._model   # "gemini-2.0-flash"
    
    # ─────────────────────────────────────────────────────────
    # MÉTHODE GENERATE (cœur du provider)
    # ─────────────────────────────────────────────────────────
    
    def generate(
        self,
        prompt: str,
        generation: GenerationConfig,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Envoie une question à Gemini et retourne la réponse."""
        
        # 1. CONSTRUIRE LES MESSAGES
        # ─────────────────────────────────────────────────────
        messages: list[dict] = []
        
        if system_prompt:
            # Si on a une consigne système, on l'ajoute en premier
            messages.append({"role": "system", "content": system_prompt})
        
        # La question de l'utilisateur
        messages.append({"role": "user", "content": prompt})
        
        # Exemple de messages :
        # [
        #   {"role": "system", "content": "Sois concis."},
        #   {"role": "user", "content": "Que manger au petit-déj ?"}
        # ]
        
        # 2. PRÉPARER LES PARAMÈTRES
        # ─────────────────────────────────────────────────────
        kwargs: dict = dict(
            model=self._model,                      # "gemini-2.0-flash"
            messages=messages,                      # La conversation
            temperature=generation.temperature,    # 0.0
            max_tokens=generation.max_tokens,      # 256
            top_p=generation.top_p,                # 1.0
        )
        
        if generation.seed is not None:
            kwargs["seed"] = generation.seed       # 42
        # Le seed n'est pas supporté par tous les modèles
        
        # 3. APPELER L'API
        # ─────────────────────────────────────────────────────
        response = self._client.chat.completions.create(**kwargs)
        # ** = décompresse le dict en arguments nommés
        # Équivalent à : .create(model=..., messages=..., temperature=..., ...)
        
        # 4. EXTRAIRE LA RÉPONSE
        # ─────────────────────────────────────────────────────
        content = response.choices[0].message.content
        # response.choices = liste des réponses (on en demande 1)
        # .message.content = le texte de la réponse
        
        return (content or "").strip()
        # or "" = si content est None, utiliser ""
        # .strip() = supprimer les espaces au début/fin
```

---

## 7. FICHIER `mistral_nemo_provider.py`

**📍 Chemin** : `src/providers/mistral_nemo_provider.py`

### Pourquoi ce fichier ?

C'est l'**implémentation concrète** du contrat `LLMProvider` pour Mistral-Nemo via Ollama.

**Différence avec Gemini** :
- Gemini = API cloud (serveur de Google sur internet)
- Mistral-Nemo = modèle local (tourne sur VOTRE ordinateur via Ollama)

### À quoi ça sert ?

1. **Permettre l'exécution hors-ligne** : pas besoin d'internet
2. **Aucune limite de requêtes** : c'est votre PC, pas de quota
3. **Gratuit illimité** : pas de clé API, pas de facturation
4. **Comparer les modèles** : le Lot D peut comparer Gemini (cloud) vs Mistral-Nemo (local)

### Comment c'est utilisé ?

```python
# Créé par la factory :
config = RunConfig.from_file("configs/baseline_ollama.json")
provider = create_provider(config)  # → MistralNemoProvider

# Utilisé par le pipeline (identique à Gemini !) :
answer = provider.generate(
    prompt="Que servir au petit-déjeuner ?",
    generation=config.generation,
    system_prompt=None
)
# → "Je vous suggère des croissants avec du café."
```

### Ce qui se passe en coulisse

```
1. Ollama doit tourner sur votre PC (service en arrière-plan)
   - Il écoute sur http://localhost:11434

2. Quand vous appelez generate() :
   POST http://localhost:11434/api/chat
   Body: {"model": "mistral-nemo", "messages": [...], "options": {"temperature": 0}}

3. Ollama charge le modèle en RAM (~7.5 Go) si pas déjà fait

4. Le modèle génère la réponse localement (GPU ou CPU)

5. Ollama renvoie la réponse
```

### Différences avec le provider Gemini

| Aspect | GeminiProvider | MistralNemoProvider |
|--------|----------------|---------------------|
| SDK | `openai` | `ollama` |
| Authentification | Clé API | Aucune |
| URL | Internet (Google) | localhost:11434 |
| Paramètre longueur | `max_tokens` | `num_predict` |
| Réseau | Requis | Pas requis |

### Prérequis

Avant d'utiliser ce provider :
```bash
# 1. Installer Ollama (https://ollama.com)
# 2. Télécharger le modèle :
ollama pull mistral-nemo
# 3. Ollama doit tourner (se lance automatiquement après installation)
```

**But** : Implémenter le provider pour Mistral-Nemo (via Ollama local).

```python
from __future__ import annotations

import logging
from typing import Optional

import ollama as _ollama_pkg
# Bibliothèque Python pour parler au serveur Ollama

from src.models.config import GenerationConfig, RunConfig
from src.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class MistralNemoProvider(LLMProvider):
    """Provider pour Mistral-Nemo 12B via Ollama (local)."""
    
    def __init__(self, config: RunConfig) -> None:
        prov = config.provider
        
        self._model = prov.model          # "mistral-nemo"
        self._label = prov.provider_label or "ollama"
        self._host = prov.ollama_host or "http://localhost:11434"
        # Ollama tourne sur le port 11434 par défaut
        
        # Créer le client Ollama
        self._client = _ollama_pkg.Client(host=self._host)
        
        logger.info("MistralNemoProvider initialisé : model=%s", self._model)
    
    @property
    def provider_name(self) -> str:
        return self._label
    
    @property
    def model_id(self) -> str:
        return self._model
    
    def generate(
        self,
        prompt: str,
        generation: GenerationConfig,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Interroge Mistral-Nemo via Ollama."""
        
        # 1. CONSTRUIRE LES MESSAGES (même format que OpenAI)
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # 2. OPTIONS OLLAMA (noms légèrement différents)
        options: dict = {
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "num_predict": generation.max_tokens,  # Ollama dit "num_predict" pas "max_tokens"
        }
        if generation.seed is not None:
            options["seed"] = generation.seed
        
        # 3. APPELER OLLAMA
        response = self._client.chat(
            model=self._model,
            messages=messages,
            options=options,
        )
        
        # 4. EXTRAIRE LA RÉPONSE
        # Ollama >= 0.4 retourne un objet, les anciennes versions un dict
        if hasattr(response, "message"):
            content = response.message.content  # Nouvelle API
        else:
            content = response.get("message", {}).get("content", "")  # Ancienne API
        
        return (content or "").strip()
```

---

## 8. FICHIER `system_prompt.py`

**📍 Chemin** : `src/promptings/system_prompt.py`

### Pourquoi ce fichier ?

La **baseline** du projet envoie les questions telles quelles au LLM. Mais le **Lot C** doit tester des variantes :

- *"Et si on ajoutait une consigne système ?"*
- *"Et si on reformulait les questions ?"*

Ce fichier est le **point d'extension** pour le Lot C.

### À quoi ça sert ?

**1. System prompts** = consigne globale donnée au LLM avant la question

```
SANS system prompt (baseline) :
Utilisateur : "Que servir au petit-déjeuner ?"
LLM : "Voici quelques suggestions : des œufs brouillés, du pain grillé..."

AVEC system prompt "Sois concis" :
Système : "Tu es un assistant. Réponds en une seule phrase."
Utilisateur : "Que servir au petit-déjeuner ?"
LLM : "Servez des tartines avec de la confiture."
```

**2. Prompt templates** = reformulation de la question

```
SANS template (baseline) :
Prompt envoyé : "Que servir au petit-déjeuner ?"

AVEC template "Réponds en 1 mot : {prompt}" :
Prompt envoyé : "Réponds en 1 mot : Que servir au petit-déjeuner ?"
```

### Comment c'est utilisé ?

```python
# Dans runner.py :
from src.promptings.system_prompt import get_system_prompt, apply_prompt_template

# 1. Récupérer le system prompt (depuis la config)
system_prompt = get_system_prompt(config.pipeline.system_prompt)
# Si config.pipeline.system_prompt = None → retourne None (baseline)
# Si config.pipeline.system_prompt = "neutral" → retourne le texte correspondant

# 2. Reformuler le prompt (depuis la config)
final_prompt = apply_prompt_template(item.prompt, config.pipeline.prompt_template)
# Si template = None → retourne le prompt tel quel
# Si template = "Réponds en 1 phrase : {prompt}" → applique le template

# 3. Envoyer au LLM
answer = provider.generate(final_prompt, config.generation, system_prompt)
```

### Comment le Lot C l'utilise

**Étape 1 : Ajouter une stratégie** dans `system_prompt.py` :
```python
_SYSTEM_PROMPTS = {
    "neutral": "You are a helpful assistant. Be concise.",
    "expert": "You are a cultural expert. Consider local customs.",
}
```

**Étape 2 : Créer une config** `configs/variante_expert.json` :
```json
{
  "pipeline": {
    "system_prompt": "expert",
    "prompt_template": null
  }
}
```

**Étape 3 : Lancer** :
```bash
python run_baseline.py --config configs/variante_expert.json
```

### Pourquoi séparer dans un fichier ?

- **Séparation des responsabilités** : le pipeline ne sait pas quelles stratégies existent
- **Facile à étendre** : le Lot C modifie UN fichier, sans toucher au pipeline
- **Traçable** : le nom de la stratégie est sauvegardé dans `config.json`

**But** : Gérer les stratégies de prompting (system prompts et templates).

```python
from __future__ import annotations
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# REGISTRE DES SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════

_SYSTEM_PROMPTS: dict[str, str] = {
    # Clé = nom de la stratégie
    # Valeur = texte du system prompt
    
    # Exemples pour le Lot C :
    # "neutral": "You are a helpful assistant. Answer concisely.",
    # "cultural_expert": "You are a local cultural expert.",
}


def get_system_prompt(strategy: Optional[str] = None) -> Optional[str]:
    """
    Retourne le system prompt pour la stratégie donnée.
    
    Args:
        strategy: Nom de la stratégie, ou None pour la baseline
    
    Returns:
        Le texte du system prompt, ou None (baseline)
    """
    if strategy is None:
        return None  # Baseline = pas de system prompt
    
    if strategy not in _SYSTEM_PROMPTS:
        raise ValueError(
            f"Stratégie inconnue : '{strategy}'. "
            f"Disponibles : {list(_SYSTEM_PROMPTS.keys())}"
        )
    
    return _SYSTEM_PROMPTS[strategy]


# ═══════════════════════════════════════════════════════════════
# TRANSFORMATION DU PROMPT
# ═══════════════════════════════════════════════════════════════

def apply_prompt_template(
    prompt: str,
    template: Optional[str] = None,
) -> str:
    """
    Applique un template au prompt.
    
    Args:
        prompt: La question originale
        template: Template avec {prompt} comme placeholder, ou None
    
    Returns:
        Le prompt transformé, ou le prompt original si template=None
    """
    if template is None:
        return prompt  # Baseline = pas de transformation
    
    return template.format(prompt=prompt)
    # Ex: "Réponds en 1 phrase : {prompt}".format(prompt="Que manger ?")
    #   → "Réponds en 1 phrase : Que manger ?"
```

---

## 9. FICHIER `logs.py`

**📍 Chemin** : `src/pipelines/logs.py`

### Pourquoi ce fichier ?

Un run complet prend **plusieurs heures** (21 000 questions × 4 secondes = ~24h).

Sans logging :
- Vous ne savez pas où en est le run
- Si ça plante, vous ne savez pas pourquoi
- Vous ne pouvez pas débugger après coup

Avec logging :
```
[2026-04-12 14:30:01] INFO  DÉBUT DU RUN : baseline_gemini_flash
[2026-04-12 14:30:02] INFO  [1/10] Traitement de en_specific.jsonl
[2026-04-12 14:30:02] INFO    4140 prompts chargés
[2026-04-12 14:35:00] INFO    Progression : 50/4140  (erreurs: 0)
[2026-04-12 14:35:03] ERROR   ERREUR id=42 : Timeout après 30s
[2026-04-12 15:30:00] INFO    Progression : 500/4140  (erreurs: 1)
```

### À quoi ça sert ?

1. **Suivre la progression en direct** : messages affichés dans le terminal
2. **Garder une trace permanente** : tout est écrit dans `run.log`
3. **Débugger les erreurs** : on sait exactement quelle question a échoué
4. **Analyser après coup** : le fichier `run.log` est conservé avec les résultats

### Comment c'est utilisé ?

```python
# Au début du pipeline :
from src.pipelines.logs import setup_logging

logger = setup_logging(config.run_id, config.pipeline.output_dir)
# Crée le fichier data/output/baseline_gemini_flash/run.log

# Pendant le run :
logger.info("Traitement de %s", filename)      # INFO = information normale
logger.warning("Fichier introuvable : %s", f)  # WARNING = attention
logger.error("ERREUR id=%s : %s", id, exc)     # ERROR = problème
logger.debug("Détails techniques : %s", data)  # DEBUG = verbeux (fichier seulement)
```

### Pourquoi 2 sorties (console + fichier) ?

| Sortie | Niveau | Pourquoi |
|--------|--------|----------|
| Console | INFO et plus | Pour suivre le run en direct sans être noyé |
| Fichier | DEBUG et plus | Pour avoir TOUS les détails pour le débogage |

```python
# Dans le fichier seulement (DEBUG) :
logger.debug("Réponse brute du LLM : %s", response)

# Dans le fichier ET la console (INFO) :
logger.info("Progression : %d/%d", current, total)
```

### Ce qui est écrit dans run.log

```
[2026-04-12 14:30:01] INFO     pipeline – DÉBUT DU RUN : baseline_gemini_flash
[2026-04-12 14:30:01] INFO     pipeline – Provider : google (gemini-2.0-flash)
[2026-04-12 14:30:02] INFO     pipeline – [1/10] Traitement de en_specific.jsonl
[2026-04-12 14:30:02] DEBUG    pipeline – 4140 prompts chargés
[2026-04-12 14:30:05] DEBUG    pipeline – Prompt id=1 : "What to serve..."
[2026-04-12 14:30:06] DEBUG    pipeline – Réponse : "Serve pancakes..."
...
[2026-04-12 18:30:00] INFO     pipeline – FIN DU RUN – 21061 prompts, 3 erreurs
```

**But** : Configurer le système de logging (console + fichier).

```python
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(run_id: str, output_dir: str = "data/output") -> logging.Logger:
    """
    Configure le logging pour un run.
    
    Args:
        run_id: Identifiant du run (pour nommer le fichier)
        output_dir: Dossier racine de sortie
    
    Returns:
        Le logger configuré
    """
    # 1. CRÉER LE DOSSIER ET LE FICHIER DE LOG
    log_dir = Path(output_dir) / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"
    
    # 2. CRÉER LE LOGGER
    logger = logging.getLogger("pipeline")
    # Nom du logger = "pipeline" (on peut avoir plusieurs loggers nommés)
    
    logger.setLevel(logging.DEBUG)
    # DEBUG = niveau le plus verbeux (capture tout)
    
    # Éviter les doublons si appelé plusieurs fois
    if logger.handlers:
        logger.handlers.clear()
    
    # 3. FORMAT DES MESSAGES
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # %(asctime)s = date/heure
    # %(levelname)-8s = niveau (INFO, ERROR...) aligné sur 8 caractères
    # %(name)s = nom du logger
    # %(message)s = le message
    
    # 4. HANDLER CONSOLE (affiche à l'écran)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)   # Affiche INFO et plus grave
    console.setFormatter(fmt)
    logger.addHandler(console)
    
    # 5. HANDLER FICHIER (écrit dans run.log)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # Écrit TOUT (même DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    
    logger.info("Logging initialisé – fichier : %s", log_file)
    
    return logger
```

---

## 10. FICHIER `runner.py`

**📍 Chemin** : `src/pipelines/runner.py`

### Pourquoi ce fichier ?

C'est le **chef d'orchestre** du projet. Il coordonne tout :
- Lecture des fichiers JSONL
- Création du provider LLM
- Envoi des questions une par une
- Écriture des réponses
- Gestion des erreurs
- Logging
- Sauvegarde de la configuration

Sans ce fichier, vous devriez écrire tout ça à la main à chaque fois.

### À quoi ça sert ?

1. **Automatiser le processus complet** : une seule fonction `run_pipeline(config)` fait tout
2. **Gérer la reprise après crash** : si le programme s'arrête, il reprend où il en était
3. **Respecter les quotas API** : pause entre chaque requête
4. **Ne jamais crasher** : les erreurs sont capturées et enregistrées
5. **Produire des résultats propres** : fichiers JSONL + config + logs + résumé

### Comment c'est utilisé ?

```python
# Usage simple (depuis run_baseline.py) :
from src.models.config import RunConfig
from src.pipelines.runner import run_pipeline

config = RunConfig.from_file("configs/baseline_gimini.json")
summary = run_pipeline(config)

print(summary)
# {
#   "run_id": "baseline_gemini_flash",
#   "total_prompts": 21061,
#   "total_errors": 3,
#   "duration_seconds": 86400
# }
```

### Ce que le pipeline produit

```
data/output/baseline_gemini_flash/
├── config.json           # Copie exacte de la configuration utilisée
├── run.log               # Journal détaillé de l'exécution
├── run_summary.json      # Résumé (durée, erreurs, etc.)
├── en_specific.jsonl     # Réponses pour l'anglais (specific)
├── en_unspecific.jsonl   # Réponses pour l'anglais (unspecific)
├── fr_specific.jsonl     # Réponses pour le français (specific)
├── fr_unspecific.jsonl   # ...
├── de_specific.jsonl
├── de_unspecific.jsonl
├── es_specific.jsonl
├── es_unspecific.jsonl
├── ru_specific.jsonl
└── ru_unspecific.jsonl
```

### La fonctionnalité de reprise

Si le programme plante au milieu (coupure de courant, erreur réseau...) :

```python
# Quand on relance, le pipeline :
# 1. Ouvre le fichier de sortie existant
# 2. Lit les IDs déjà traités
done_ids = _already_processed_ids(output_file)  # {"1", "2", "3", ...}

# 3. Saute les questions déjà faites
for item in prompts:
    if item.id in done_ids:
        continue  # Déjà fait, on passe
    # Sinon, on traite normalement
```

**Exemple** : vous avez traité 5000 questions, le PC s'éteint. Vous relancez → le pipeline reprend à la question 5001.

### Le callback de progression (pour le Lot B)

```python
def run_pipeline(config, progress_cb=None):
    # ...
    if progress_cb:
        progress_cb(filename, file_idx, total_files, prompt_idx, total_prompts)
```

Le Lot B peut passer une fonction callback pour afficher une barre de progression :

```python
def ma_progression(filename, file_idx, total_files, prompt_idx, total_prompts):
    percent = (prompt_idx / total_prompts) * 100
    print(f"Fichier {file_idx}/{total_files} – {percent:.1f}%")

run_pipeline(config, progress_cb=ma_progression)
```

**But** : Orchestrer tout le pipeline (lecture → LLM → écriture).

```python
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import jsonlines
# Bibliothèque pour lire/écrire des fichiers JSONL facilement

from src.models.config import RunConfig
from src.models.schemas import PromptItem, ResultItem
from src.pipelines.logs import setup_logging
from src.promptings.system_prompt import apply_prompt_template, get_system_prompt
from src.providers import create_provider

logger = logging.getLogger("pipeline")


# Type pour le callback de progression (utilisé par le Lot B)
ProgressCallback = Optional[Callable[[str, int, int, int, int], None]]
# Callable[[args...], return_type] = une fonction avec cette signature


def _load_prompts(filepath: str) -> list[PromptItem]:
    """Charge les questions depuis un fichier JSONL."""
    items: list[PromptItem] = []
    
    with jsonlines.open(filepath, mode="r") as reader:
        # Ouvre le fichier JSONL en lecture
        for obj in reader:
            # obj = un dictionnaire Python (une ligne du fichier)
            items.append(PromptItem(**obj))
            # ** décompresse le dict en arguments nommés
    
    return items


def _already_processed_ids(output_file: Path) -> set[str]:
    """
    Retourne les IDs déjà traités (pour la reprise après crash).
    """
    done: set[str] = set()
    
    if output_file.exists():
        with jsonlines.open(str(output_file), mode="r") as reader:
            for obj in reader:
                done.add(obj.get("id", ""))
    
    return done


def run_pipeline(
    config: RunConfig,
    progress_cb: ProgressCallback = None,
) -> dict:
    """
    Exécute un run complet.
    
    Args:
        config: Configuration du run
        progress_cb: Callback optionnel pour suivre la progression
    
    Returns:
        Résumé du run (dict)
    """
    
    # ═══════════════════════════════════════════════════════════
    # 1. INITIALISATION
    # ═══════════════════════════════════════════════════════════
    
    run_logger = setup_logging(config.run_id, config.pipeline.output_dir)
    run_logger.info("DÉBUT DU RUN : %s", config.run_id)
    
    # ═══════════════════════════════════════════════════════════
    # 2. SAUVEGARDE DE LA CONFIGURATION
    # ═══════════════════════════════════════════════════════════
    
    output_dir = config.output_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir)
    
    # Copie aussi dans configs/runs/
    runs_dir = Path("configs/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    config.save(runs_dir / config.run_id)
    
    # ═══════════════════════════════════════════════════════════
    # 3. CRÉER LE PROVIDER
    # ═══════════════════════════════════════════════════════════
    
    provider = create_provider(config)
    run_logger.info("Provider : %s (%s)", provider.provider_name, provider.model_id)
    
    # ═══════════════════════════════════════════════════════════
    # 4. RÉSOUDRE LE PROMPTING
    # ═══════════════════════════════════════════════════════════
    
    system_prompt = get_system_prompt(config.pipeline.system_prompt)
    # None pour la baseline
    
    # ═══════════════════════════════════════════════════════════
    # 5. BOUCLE SUR LES FICHIERS
    # ═══════════════════════════════════════════════════════════
    
    input_files = config.input_files()
    total_files = len(input_files)
    total_prompts_processed = 0
    total_errors = 0
    start_time = time.time()
    
    for file_idx, input_file in enumerate(input_files, start=1):
        # enumerate(..., start=1) = commence à 1 au lieu de 0
        
        input_path = Path(input_file)
        if not input_path.exists():
            run_logger.warning("Fichier introuvable : %s", input_file)
            continue
        
        filename = input_path.name  # "fr_unspecific.jsonl"
        output_file = output_dir / filename
        
        run_logger.info("[%d/%d] Traitement de %s", file_idx, total_files, filename)
        
        # Charger les questions
        prompts = _load_prompts(input_file)
        total_in_file = len(prompts)
        
        # Vérifier ce qui est déjà fait (pour la reprise)
        done_ids = _already_processed_ids(output_file)
        
        # Ouvrir le fichier de sortie en mode APPEND (ajout)
        with jsonlines.open(str(output_file), mode="a") as writer:
            
            for prompt_idx, item in enumerate(prompts, start=1):
                
                # Sauter les prompts déjà traités
                if item.id in done_ids:
                    continue
                
                # Appliquer le template (baseline = rien)
                final_prompt = apply_prompt_template(
                    item.prompt,
                    config.pipeline.prompt_template
                )
                
                # ───────────────────────────────────────────────
                # APPELER LE LLM
                # ───────────────────────────────────────────────
                try:
                    answer = provider.generate(
                        prompt=final_prompt,
                        generation=config.generation,
                        system_prompt=system_prompt,
                    )
                except Exception as exc:
                    # En cas d'erreur, ne pas crasher
                    answer = f"ERROR: {type(exc).__name__}: {exc}"
                    total_errors += 1
                    run_logger.error("ERREUR id=%s : %s", item.id, answer)
                
                # ───────────────────────────────────────────────
                # ÉCRIRE LE RÉSULTAT
                # ───────────────────────────────────────────────
                result = ResultItem(id=item.id, prompt=item.prompt, answer=answer)
                writer.write(result.model_dump())
                # model_dump() = convertit en dict
                # writer.write() = écrit une ligne JSONL
                
                total_prompts_processed += 1
                
                # Log progression tous les 50 prompts
                if prompt_idx % 50 == 0 or prompt_idx == total_in_file:
                    run_logger.info("  Progression : %d/%d", prompt_idx, total_in_file)
                
                # Callback pour le Lot B
                if progress_cb:
                    progress_cb(filename, file_idx, total_files, prompt_idx, total_in_file)
                
                # Rate limiting
                if config.pipeline.request_delay > 0:
                    time.sleep(config.pipeline.request_delay)
    
    # ═══════════════════════════════════════════════════════════
    # 6. RÉSUMÉ
    # ═══════════════════════════════════════════════════════════
    
    elapsed = time.time() - start_time
    
    summary = {
        "run_id": config.run_id,
        "provider": config.provider.provider_label,
        "model": config.provider.model,
        "total_prompts": total_prompts_processed,
        "total_errors": total_errors,
        "duration_seconds": round(elapsed, 2),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    
    # Sauvegarder le résumé
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    run_logger.info("FIN DU RUN – %d prompts, %d erreurs, %.1fs", 
                    total_prompts_processed, total_errors, elapsed)
    
    return summary
```

---

## 11. FICHIER `routes.py` (API)

**📍 Chemin** : `api/routes.py`

### Pourquoi ce fichier ?

Le Lot B doit créer une **interface web** (Streamlit) pour piloter le pipeline. Mais Streamlit ne peut pas importer directement le code Python du pipeline (problèmes de process, de threads...).

**Solution** : une API HTTP entre les deux.

```
┌─────────────────┐         HTTP          ┌─────────────────┐
│   Streamlit     │ ←──────────────────→  │   FastAPI       │
│   (Lot B)       │   POST /api/runs      │   (routes.py)   │
│   Interface     │   GET /api/status     │   Backend       │
└─────────────────┘                       └────────┬────────┘
                                                   │
                                                   ↓
                                          ┌─────────────────┐
                                          │   runner.py     │
                                          │   Pipeline      │
                                          └─────────────────┘
```

### À quoi ça sert ?

1. **Découpler l'interface du backend** : Streamlit et le pipeline sont indépendants
2. **Permettre les runs en arrière-plan** : l'interface ne bloque pas pendant le run
3. **Exposer une API REST** : standard, facile à tester, documenté automatiquement
4. **Faciliter le travail du Lot B** : ils appellent juste des URLs

### Comment c'est utilisé ?

**Par le Lot B (Streamlit)** :
```python
import requests

# Lancer un run
config = {"run_id": "test", "provider": {...}, ...}
response = requests.post("http://localhost:8000/api/runs", json=config)
print(response.json())  # {"run_id": "test", "status": "queued", ...}

# Vérifier le statut
response = requests.get("http://localhost:8000/api/runs/test/status")
print(response.json())  # {"run_id": "test", "status": "running", ...}

# Récupérer les résultats
response = requests.get("http://localhost:8000/api/runs/test/results/fr_unspecific.jsonl")
print(response.json())  # [{"id": "1", "prompt": "...", "answer": "..."}, ...]
```

### Les 5 endpoints

| Endpoint | Méthode | Description | Exemple de réponse |
|----------|---------|-------------|-------------------|
| `/api/runs` | POST | Lancer un run | `{"status": "queued"}` |
| `/api/runs` | GET | Lister les runs | `[{"run_id": "...", ...}]` |
| `/api/runs/{id}/status` | GET | Statut d'un run | `{"status": "completed"}` |
| `/api/runs/{id}/results/{file}` | GET | Contenu d'un JSONL | `[{...}, {...}]` |
| `/api/providers` | GET | LLMs disponibles | `{"providers": [...]}` |

### Pourquoi les runs en arrière-plan ?

Un run complet prend des heures. Si on faisait :
```python
@router.post("/runs")
async def create_run(config):
    run_pipeline(config)  # ← BLOQUE pendant des heures !
    return {"status": "done"}
```

L'utilisateur attendrait des heures sans réponse. À la place :

```python
@router.post("/runs")
async def create_run(config, background_tasks: BackgroundTasks):
    background_tasks.add_task(_execute_run, config)  # Lance en arrière-plan
    return {"status": "queued"}  # Répond immédiatement
```

L'interface peut ensuite interroger `/status` régulièrement pour savoir où en est le run.

### Documentation automatique

FastAPI génère automatiquement une documentation interactive :

```
http://localhost:8000/docs
```

Vous pouvez tester tous les endpoints directement depuis le navigateur.

**But** : Exposer des endpoints HTTP pour piloter le pipeline.

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.models.config import RunConfig
from src.pipelines.runner import run_pipeline

router = APIRouter(tags=["runs"])
# APIRouter = groupe de routes
# tags = catégorie dans la doc Swagger

# Stockage en mémoire de l'état des runs
_run_status: dict[str, dict] = {}


# ═══════════════════════════════════════════════════════════════
# SCHÉMAS DE RÉPONSE
# ═══════════════════════════════════════════════════════════════

class RunResponse(BaseModel):
    run_id: str
    status: str
    message: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    summary: Optional[dict] = None


# ═══════════════════════════════════════════════════════════════
# TÂCHE EN ARRIÈRE-PLAN
# ═══════════════════════════════════════════════════════════════

def _execute_run(config: RunConfig) -> None:
    """Exécute le pipeline (appelé en arrière-plan)."""
    _run_status[config.run_id] = {"status": "running"}
    try:
        summary = run_pipeline(config)
        _run_status[config.run_id] = {"status": "completed", "summary": summary}
    except Exception as exc:
        _run_status[config.run_id] = {"status": "failed", "error": str(exc)}


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@router.post("/runs", response_model=RunResponse)
async def create_run(config: RunConfig, background_tasks: BackgroundTasks):
    """
    POST /api/runs – Lance un nouveau run.
    
    Le run s'exécute en arrière-plan (ne bloque pas).
    """
    _run_status[config.run_id] = {"status": "queued"}
    background_tasks.add_task(_execute_run, config)
    # add_task = lance la fonction après avoir répondu au client
    
    return RunResponse(
        run_id=config.run_id,
        status="queued",
        message=f"Run '{config.run_id}' lancé.",
    )


@router.get("/runs")
async def list_runs():
    """GET /api/runs – Liste tous les runs passés."""
    runs_dir = Path("configs/runs")
    runs = []
    
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir(), reverse=True):
            config_file = run_dir / "config.json"
            if config_file.exists():
                data = json.loads(config_file.read_text())
                runs.append({
                    "run_id": data.get("run_id"),
                    "description": data.get("description"),
                    "provider": data.get("provider", {}).get("provider_label"),
                    "model": data.get("provider", {}).get("model"),
                })
    
    return runs


@router.get("/runs/{run_id}/status", response_model=RunStatusResponse)
async def get_run_status(run_id: str):
    """GET /api/runs/{id}/status – Statut d'un run."""
    
    # Vérifier en mémoire (run en cours)
    if run_id in _run_status:
        info = _run_status[run_id]
        return RunStatusResponse(
            run_id=run_id,
            status=info.get("status"),
            summary=info.get("summary"),
        )
    
    # Vérifier sur disque (run terminé)
    summary_file = Path("data/output") / run_id / "run_summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        return RunStatusResponse(run_id=run_id, status="completed", summary=summary)
    
    raise HTTPException(status_code=404, detail=f"Run '{run_id}' introuvable.")


@router.get("/providers")
async def list_providers():
    """GET /api/providers – Liste les providers disponibles."""
    return {
        "providers": [
            {"type": "openai_compatible", "label": "google", "models": ["gemini-2.0-flash"]},
            {"type": "ollama", "label": "ollama", "models": ["mistral-nemo"]},
        ],
        "languages": ["en", "fr", "de", "es", "ru"],
    }
```

---

## 12. FICHIER `run_baseline.py`

**📍 Chemin** : `run_baseline.py` (racine du projet)

### Pourquoi ce fichier ?

C'est le **point d'entrée utilisateur**. Au lieu de lancer Python et taper :

```python
>>> from src.models.config import RunConfig
>>> from src.pipelines.runner import run_pipeline
>>> from dotenv import load_dotenv
>>> load_dotenv()
>>> config = RunConfig.from_file("configs/baseline_gimini.json")
>>> run_pipeline(config)
```

Vous tapez simplement :
```bash
python run_baseline.py
```

### À quoi ça sert ?

1. **Simplifier le lancement** : une seule commande
2. **Charger les variables d'environnement** : lit automatiquement le fichier `.env`
3. **Permettre des options CLI** : `--languages`, `--types`, `--config`, `--run-id`
4. **Afficher un résumé clair** : avant et après le run

### Comment c'est utilisé ?

```bash
# Lancer la baseline par défaut (Gemini)
python run_baseline.py

# Utiliser Mistral-Nemo (local)
python run_baseline.py --config configs/baseline_ollama.json

# Test rapide : seulement français, seulement unspecific
python run_baseline.py --languages fr --types unspecific

# Combiner les options
python run_baseline.py --config configs/baseline_ollama.json --languages fr de --types unspecific --run-id mon_test
```

### Ce qui s'affiche

```
============================================================
Run ID    : baseline_gemini_flash
Provider  : google / gemini-2.0-flash
Langues   : ['en', 'fr', 'de', 'es', 'ru']
Types     : ['specific', 'unspecific']
============================================================

[2026-04-12 14:30:01] INFO  DÉBUT DU RUN...
[2026-04-12 14:30:02] INFO  [1/10] Traitement de en_specific.jsonl
...

✅ Terminé ! 21061 prompts traités.
```

### Les arguments disponibles

| Argument | Description | Exemple |
|----------|-------------|---------|
| `--config` | Chemin du fichier de config | `--config configs/baseline_ollama.json` |
| `--languages` | Langues à traiter (espace séparés) | `--languages fr en de` |
| `--types` | Types de dataset | `--types unspecific` |
| `--run-id` | Identifiant personnalisé | `--run-id test_01` |

### Pourquoi `sys.path.insert(0, ...)` ?

```python
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
```

Quand vous lancez `python run_baseline.py`, Python ne connaît pas le dossier `src/`. Cette ligne ajoute la racine du projet au `PYTHONPATH`, ce qui permet :

```python
from src.models.config import RunConfig  # ✅ Fonctionne maintenant
```

### Pourquoi `if __name__ == "__main__"` ?

```python
if __name__ == "__main__":
    main()
```

Ce bloc ne s'exécute **que si on lance le script directement** :
- `python run_baseline.py` → `__name__ == "__main__"` → `main()` s'exécute
- `from run_baseline import main` → `__name__ == "run_baseline"` → `main()` ne s'exécute PAS

Ça permet d'importer des fonctions du script sans déclencher le run.

**But** : Script qu'on lance depuis le terminal.

```python
#!/usr/bin/env python
# ↑ Shebang : permet de faire ./run_baseline.py sur Linux

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
# Sans ça, Python ne trouverait pas "src.models", etc.

from dotenv import load_dotenv
# Charge les variables d'environnement depuis le fichier .env

from src.models.config import RunConfig
from src.pipelines.runner import run_pipeline


def main() -> None:
    # ═══════════════════════════════════════════════════════════
    # PARSING DES ARGUMENTS
    # ═══════════════════════════════════════════════════════════
    
    parser = argparse.ArgumentParser(
        description="Lancer un run ELOQUENT"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_gimini.json",
        help="Chemin vers la config JSON",
    )
    
    parser.add_argument(
        "--languages",
        nargs="+",           # Accepte plusieurs valeurs : --languages fr en
        default=None,
        help="Langues à traiter",
    )
    
    parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        choices=["specific", "unspecific"],
        help="Types de dataset",
    )
    
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identifiant personnalisé",
    )
    
    args = parser.parse_args()
    
    # ═══════════════════════════════════════════════════════════
    # CHARGER L'ENVIRONNEMENT
    # ═══════════════════════════════════════════════════════════
    
    load_dotenv()
    # Lit le fichier .env et ajoute les variables à os.environ
    
    # ═══════════════════════════════════════════════════════════
    # CHARGER LA CONFIGURATION
    # ═══════════════════════════════════════════════════════════
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERREUR : {config_path} introuvable")
        sys.exit(1)
    
    config = RunConfig.from_file(config_path)
    
    # Surcharger avec les arguments CLI
    if args.languages:
        config.pipeline.languages = args.languages
    if args.types:
        config.pipeline.dataset_types = args.types
    if args.run_id:
        config.run_id = args.run_id
    
    # ═══════════════════════════════════════════════════════════
    # AFFICHER LA CONFIG
    # ═══════════════════════════════════════════════════════════
    
    print("=" * 60)
    print(f"Run ID    : {config.run_id}")
    print(f"Provider  : {config.provider.provider_label} / {config.provider.model}")
    print(f"Langues   : {config.pipeline.languages}")
    print(f"Types     : {config.pipeline.dataset_types}")
    print("=" * 60)
    
    # ═══════════════════════════════════════════════════════════
    # LANCER LE PIPELINE
    # ═══════════════════════════════════════════════════════════
    
    summary = run_pipeline(config)
    
    print(f"\n✅ Terminé ! {summary['total_prompts']} prompts traités.")


if __name__ == "__main__":
    main()
    # Ce bloc ne s'exécute que si on lance le script directement
    # (pas si on l'importe)
```

---

## RÉSUMÉ : ORDRE D'IMPLÉMENTATION

Pour tout réécrire de zéro, suivez cet ordre :

1. **`src/models/schemas.py`** – Les structures de données (5 min)
2. **`src/models/config.py`** – La configuration (15 min)
3. **`src/providers/base.py`** – L'interface abstraite (5 min)
4. **`src/providers/gemini_provider.py`** – Le provider Gemini (10 min)
5. **`src/providers/mistral_nemo_provider.py`** – Le provider Ollama (10 min)
6. **`src/providers/__init__.py`** – La factory (5 min)
7. **`src/promptings/system_prompt.py`** – Le prompting (5 min)
8. **`src/pipelines/logs.py`** – Le logging (5 min)
9. **`src/pipelines/runner.py`** – Le pipeline (20 min)
10. **`api/routes.py`** – L'API (15 min)
11. **`run_baseline.py`** – Le script CLI (10 min)

**Total estimé : ~1h30** si vous comprenez bien chaque concept.

