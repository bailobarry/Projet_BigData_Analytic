# 📖 Explication complète du Lot A – État actuel + Guide pour vos collègues

---

## 1. VUE D'ENSEMBLE : Ce qui existe maintenant

```
Projet_BigData_Analytic/
│
├── src/providers/                    ← LES 2 LLMs
│   ├── __init__.py                   ← Usine qui crée le bon provider
│   ├── base.py                       ← Contrat (classe abstraite)
│   ├── gemini_provider.py            ← Google Gemini 2.0 Flash (API)
│   └── mistral_nemo_provider.py      ← Mistral-Nemo 12B (local)
│
├── src/models/                       ← STRUCTURE DES DONNÉES
│   ├── schemas.py                    ← Format des questions/réponses
│   └── config.py                     ← Configuration d'un run
│
├── src/pipelines/                    ← LE MOTEUR
│   ├── runner.py                     ← Pipeline principal (lit → interroge → écrit)
│   └── logs.py                       ← Journal d'exécution
│
├── src/promptings/                   ← STRATÉGIES DE PROMPTING (pour Lot C)
│   └── system_prompt.py              ← Baseline = aucun prompt, extensible
│
├── src/analysis/                     ← ANALYSE (pour Lot D)
│   ├── quantitative.py               ← (vide, prêt à remplir)
│   └── semantic.py                   ← (vide, prêt à remplir)
│
├── src/export/                       ← EXPORT (pour Lot E)
│   └── challenge_export.py           ← (vide, prêt à remplir)
│
├── api/                              ← API WEB (pour Lot B)
│   ├── main.py                       ← Serveur FastAPI
│   └── routes.py                     ← 5 endpoints REST
│
├── configs/                          ← CONFIGURATIONS
│   ├── baseline.json                 ← Gemini 2.0 Flash
│   └── baseline_ollama.json          ← Mistral-Nemo 12B
│
├── client/app.py                     ← (vide, pour Lot B)
├── run_baseline.py                   ← Script CLI pour lancer un run
├── tests/test_lot_a.py               ← 17 tests unitaires
├── .env.example                      ← Template de la clé API
└── README.md                         ← Documentation complète
```

---

## 2. FICHIER PAR FICHIER : Ce que fait chaque fichier

---

### 📄 `src/models/schemas.py` – Les moules à données

Définit la **forme exacte** des données qui circulent dans le système :

```python
class PromptItem:        # Ce qu'on LIT dans les fichiers d'entrée
    id: str              # "1" ou "1-5"
    prompt: str          # "What to serve for breakfast?"

class ResultItem:        # Ce qu'on ÉCRIT dans les fichiers de sortie
    id: str              
    prompt: str          
    answer: str          # "Serve pancakes with maple syrup."
```

**Qui en a besoin ?**
- Le pipeline (`runner.py`) les utilise pour lire/écrire les JSONL
- Le **Lot D** les utilisera pour charger les résultats et les analyser

---

### 📄 `src/models/config.py` – La carte d'identité d'un run

Un **run** = une exécution complète (ex: "envoyer toutes les questions à Gemini").

La config dit **tout** sur un run pour qu'on puisse le **reproduire** :

```
RunConfig
├── run_id: "baseline_gemini_flash"      ← Nom unique
├── description: "Baseline vanilla..."   ← Description humaine
├── provider: ProviderConfig             ← QUEL LLM
│   ├── type: "openai_compatible"        ← Gemini ou Ollama
│   ├── model: "gemini-2.0-flash"       ← Nom du modèle
│   ├── base_url: "https://..."          ← URL de l'API
│   └── api_key_env: "GEMINI_API_KEY"   ← Nom de la variable d'env
├── generation: GenerationConfig         ← COMMENT GÉNÉRER
│   ├── temperature: 0.0                 ← 0 = déterministe
│   ├── max_tokens: 256                  ← Longueur max réponse
│   ├── top_p: 1.0                       
│   └── seed: 42                         ← Reproductibilité
└── pipeline: PipelineConfig             ← QUOI TRAITER
    ├── languages: ["en","fr","de","es","ru"]
    ├── dataset_types: ["specific","unspecific"]
    ├── request_delay: 4.1               ← Pause entre requêtes (rate-limit)
    ├── system_prompt: null              ← Pas de consigne (baseline)
    └── prompt_template: null            ← Pas de reformulation (baseline)
```

**Méthodes utiles :**
- `config.input_files()` → liste les 10 fichiers JSONL à traiter
- `config.output_path()` → `data/output/baseline_gemini_flash/`
- `config.save(dossier)` → sauvegarde en JSON
- `RunConfig.from_file("configs/baseline.json")` → charge depuis un fichier

**Qui en a besoin ?**
- **Tout le monde.** C'est le contrat commun entre tous les lots.
- Le **Lot B** enverra une `RunConfig` via l'API pour lancer un run
- Le **Lot C** ajoutera `system_prompt` et `prompt_template` dans la config
- Le **Lot D** lira la config sauvegardée pour savoir quels paramètres ont été utilisés

---

### 📄 `src/providers/base.py` – Le contrat des LLMs

Classe abstraite = un **contrat** que tout LLM doit respecter :

```python
class LLMProvider(ABC):
    provider_name → str    # "google" ou "ollama"
    model_id → str         # "gemini-2.0-flash" ou "mistral-nemo"
    generate(prompt, generation, system_prompt) → str  # Question → Réponse
```

Le pipeline appelle TOUJOURS `provider.generate(prompt)`. Il ne sait **jamais** si c'est Gemini ou Mistral-Nemo derrière. C'est le **polymorphisme**.

---

### 📄 `src/providers/gemini_provider.py` – Google Gemini (API)

```
Quand on appelle generate("Que servir au petit-déj ?") :

1. Construit les messages :
   [{"role": "user", "content": "Que servir au petit-déj ?"}]

2. Envoie à Google via le SDK openai :
   POST https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
   {model: "gemini-2.0-flash", messages: [...], temperature: 0.0, seed: 42}

3. Reçoit la réponse et la retourne :
   "Servez des tartines avec du beurre et de la confiture."
```

---

### 📄 `src/providers/mistral_nemo_provider.py` – Mistral-Nemo (local)

```
Même chose, mais en local via Ollama :

1. Construit les messages (identique)

2. Envoie à Ollama sur votre PC :
   POST http://localhost:11434/api/chat
   {model: "mistral-nemo", messages: [...], options: {temperature: 0.0, seed: 42}}

3. Reçoit la réponse et la retourne
```

---

### 📄 `src/providers/__init__.py` – L'usine à providers

```python
def create_provider(config):
    if config.provider.type == "openai_compatible":
        return GeminiProvider(config)       # → Google Gemini
    if config.provider.type == "ollama":
        return MistralNemoProvider(config)  # → Mistral-Nemo local
```

Le pipeline dit "donne-moi un provider", et l'usine regarde la config pour créer le bon.

---

### 📄 `src/promptings/system_prompt.py` – Les stratégies de prompting

Deux fonctions :

**1. `get_system_prompt(strategy)`** – Récupère un system prompt
```python
get_system_prompt(None)          → None  (baseline : pas de consigne)
get_system_prompt("neutral")     → "You are a helpful assistant..."  (Lot C)
```

**2. `apply_prompt_template(prompt, template)`** – Reformule le prompt
```python
apply_prompt_template("Que servir ?", None)
→ "Que servir ?"  (baseline : question telle quelle)

apply_prompt_template("Que servir ?", "Réponds en 1 phrase : {prompt}")
→ "Réponds en 1 phrase : Que servir ?"  (variante Lot C)
```

**Pour le Lot C :** il suffit d'ajouter des entrées dans le dictionnaire `_SYSTEM_PROMPTS` :
```python
_SYSTEM_PROMPTS = {
    "neutral": "You are a helpful assistant. Answer concisely.",
    "cultural_expert": "You are a local cultural expert...",
}
```

---

### 📄 `src/pipelines/logs.py` – Le journal de bord

Crée un logger qui écrit à 2 endroits en même temps :
- **Écran** (console) : pour voir la progression en direct
- **Fichier** `data/output/{run_id}/run.log` : pour relire après

Exemple de ce qu'on voit :
```
[2026-04-12 14:30:01] INFO  DÉBUT DU RUN : baseline_gemini_flash
[2026-04-12 14:30:02] INFO  [1/10] Traitement de en_specific.jsonl
[2026-04-12 14:30:02] INFO    4140 prompts chargés
[2026-04-12 14:35:00] INFO    Progression : 50/4140  (erreurs: 0)
[2026-04-12 14:35:03] ERROR   ERREUR prompt id=42 : ERROR: Timeout
```

---

### 📄 `src/pipelines/runner.py` – LE CŒUR (le plus important)

C'est le fichier qui orchestre tout. Voici son déroulement :

```
run_pipeline(config)
│
├── 1. PRÉPARER
│   ├── Configurer le logging (console + fichier)
│   ├── Sauvegarder la config dans data/output/{run_id}/config.json
│   └── Sauvegarder une copie dans configs/runs/{run_id}/config.json
│
├── 2. CRÉER LE PROVIDER
│   └── create_provider(config) → GeminiProvider ou MistralNemoProvider
│
├── 3. RÉSOUDRE LE PROMPTING
│   ├── get_system_prompt(config.pipeline.system_prompt)
│   │   → None pour la baseline
│   └── Le prompt_template est aussi lu depuis la config
│
├── 4. BOUCLE SUR LES 10 FICHIERS
│   │
│   ├── Fichier 1 : en_specific.jsonl (4140 questions)
│   ├── Fichier 2 : en_unspecific.jsonl (101 questions)
│   ├── ... (8 autres fichiers)
│   │
│   └── Pour CHAQUE fichier :
│       ├── Charger toutes les questions
│       ├── Vérifier s'il y a des résultats déjà faits (reprise)
│       │
│       └── Pour CHAQUE question :
│           ├── Appliquer le template (baseline = rien)
│           ├── Envoyer au LLM : provider.generate(prompt)
│           ├── Si ERREUR → écrire "ERROR: timeout" et continuer
│           ├── Écrire {"id":"1","prompt":"...","answer":"..."} dans le JSONL
│           ├── Logger la progression tous les 50 prompts
│           ├── Appeler le callback de progression (pour Lot B)
│           └── Attendre 4.1s (rate-limit Gemini)
│
└── 5. RÉSUMER
    └── Écrire data/output/{run_id}/run_summary.json :
        {
          "total_prompts": 21061,
          "total_errors": 3,
          "duration_seconds": 86400
        }
```

**Points importants :**
- **Reprise** : si le programme plante, relancez-le — il saute les questions déjà traitées
- **Callback** : `progress_cb(filename, file_idx, total_files, prompt_idx, total_prompts)` — c'est un "crochet" que le Lot B utilisera pour afficher une barre de progression
- **Erreurs** : jamais de crash — les erreurs sont logguées et écrites comme `"ERROR: ..."` dans le champ `answer`

---

### 📄 `api/main.py` + `api/routes.py` – L'API Web

Serveur FastAPI que le **Lot B** consommera depuis Streamlit.

**5 endpoints :**

| Endpoint | Méthode | Ce qu'il fait |
|----------|---------|---------------|
| `POST /api/runs` | Lancer un run | Reçoit une `RunConfig` JSON, lance le pipeline en arrière-plan |
| `GET /api/runs` | Lister les runs | Lit tous les `configs/runs/*/config.json` |
| `GET /api/runs/{id}/status` | Statut d'un run | "running", "completed", ou "failed" |
| `GET /api/runs/{id}/results/{file}` | Résultats | Retourne le contenu d'un JSONL de sortie |
| `GET /api/providers` | Modèles dispo | Retourne la liste des 2 LLMs et leurs configs |

**Exemple :** pour lancer un run depuis Streamlit :
```python
import requests
config = {...}  # RunConfig en JSON
requests.post("http://localhost:8000/api/runs", json=config)
```

---

### 📄 `run_baseline.py` – Le script qu'on tape dans le terminal

Point d'entrée CLI simple :
```bash
python run_baseline.py                                    # Gemini (défaut)
python run_baseline.py --config configs/baseline_ollama.json  # Mistral-Nemo
python run_baseline.py --languages fr --types unspecific   # Test rapide
```

Il charge la config, charge les variables d'env (.env), et appelle `run_pipeline(config)`.

---

### 📄 `configs/baseline.json` + `baseline_ollama.json` – Les 2 configs

Fichiers JSON qui décrivent exactement comment lancer chaque baseline :
- `baseline.json` → Google Gemini 2.0 Flash, temp=0, seed=42, 5 langues
- `baseline_ollama.json` → Mistral-Nemo 12B, temp=0, seed=42, 5 langues

---

## 3. CE QUI SE PASSE QUAND ON LANCE UN RUN

```
AVANT :                              APRÈS :
data/input/                          data/output/baseline_gemini_flash/
├── en_specific.jsonl                ├── config.json          ← Config exacte
├── en_unspecific.jsonl              ├── run.log              ← Journal détaillé
├── fr_specific.jsonl                ├── run_summary.json     ← Résumé
├── fr_unspecific.jsonl              ├── en_specific.jsonl    ← Réponses EN
├── de_specific.jsonl                ├── en_unspecific.jsonl
├── de_unspecific.jsonl              ├── fr_specific.jsonl    ← Réponses FR
├── es_specific.jsonl                ├── fr_unspecific.jsonl
├── es_unspecific.jsonl              ├── de_specific.jsonl    ← Réponses DE
├── ru_specific.jsonl                ├── de_unspecific.jsonl
└── ru_unspecific.jsonl              ├── es_specific.jsonl    ← Réponses ES
                                     ├── es_unspecific.jsonl
                                     ├── ru_specific.jsonl    ← Réponses RU
                                     └── ru_unspecific.jsonl
```

Chaque fichier de sortie a le même format que l'entrée, PLUS le champ `answer` :
```json
{"id": "1", "prompt": "Que servir au petit-déj ?", "answer": "Des tartines avec du beurre."}
```

---

## 4. 🔗 GUIDE POUR CHAQUE COLLÈGUE

---

### 👩‍💻 Collègue B – Interface Streamlit

**Son travail :** Créer une interface web dans `client/app.py` pour piloter le pipeline.

**Ce qui est prêt pour elle :**

| Besoin | Déjà fourni |
|--------|-------------|
| Lancer un run | `POST /api/runs` avec une `RunConfig` JSON |
| Choisir le LLM | `GET /api/providers` retourne les 2 modèles |
| Voir la progression | Le callback `progress_cb` dans `runner.py` |
| Voir le statut | `GET /api/runs/{id}/status` → "running"/"completed" |
| Voir les résultats | `GET /api/runs/{id}/results/fr_unspecific.jsonl` |
| Lister les runs passés | `GET /api/runs` |

**Comment démarrer :**
```python
# client/app.py
import streamlit as st
import requests

API = "http://localhost:8000/api"

# Lister les providers
providers = requests.get(f"{API}/providers").json()

# Choisir un LLM
provider = st.selectbox("Modèle", ["Google Gemini", "Mistral-Nemo"])

# Lancer un run
if st.button("Lancer"):
    config = {... }  # Construire la RunConfig
    requests.post(f"{API}/runs", json=config)

# Suivre le statut
status = requests.get(f"{API}/runs/{run_id}/status").json()
st.write(f"Statut : {status['status']}")
```

---

### 👨‍💻 Collègue C – Variantes de prompting

**Son travail :** Tester si les réponses changent quand on reformule les questions ou ajoute des consignes.

**Ce qui est prêt pour lui :**

| Besoin | Déjà fourni |
|--------|-------------|
| Ajouter un system prompt | Modifier `_SYSTEM_PROMPTS` dans `system_prompt.py` |
| Ajouter un template | Utiliser `prompt_template` dans la config JSON |
| Traçabilité | La config est sauvegardée automatiquement avec les résultats |
| Lancer une variante | Créer un nouveau `.json` dans `configs/` |

**Comment ajouter une variante :**

**Étape 1 :** Ajouter une stratégie dans `src/promptings/system_prompt.py` :
```python
_SYSTEM_PROMPTS = {
    "neutral": "You are a helpful assistant. Give a concise, neutral answer.",
    "cultural_expert": "You are a cultural expert. Consider local customs.",
}
```

**Étape 2 :** Créer un fichier config `configs/variante_neutral.json` :
```json
{
  "run_id": "variante_neutral_gemini",
  "description": "Variante avec system prompt neutral",
  "provider": { ... même chose que baseline ... },
  "pipeline": {
    "system_prompt": "neutral",
    "prompt_template": null
  }
}
```

**Étape 3 :** Lancer : `python run_baseline.py --config configs/variante_neutral.json`

**Pour un template de reformulation :**
```json
{
  "pipeline": {
    "system_prompt": null,
    "prompt_template": "Answer briefly and neutrally: {prompt}"
  }
}
```
Le `{prompt}` sera remplacé par la question originale.

---

### 👩‍💻 Collègue D – Analyse quantitative et qualitative

**Son travail :** Comparer les résultats entre modèles, langues, et variantes.

**Ce qui est prêt pour elle :**

| Besoin | Déjà fourni |
|--------|-------------|
| Données structurées | Fichiers JSONL dans `data/output/{run_id}/` |
| Format standardisé | `{"id": "...", "prompt": "...", "answer": "..."}` |
| Savoir quel modèle a été utilisé | `config.json` sauvegardé avec chaque run |
| Savoir s'il y a eu des erreurs | `run_summary.json` + `run.log` |
| Comparer 2 runs | Même IDs dans chaque fichier → facile à joindre |

**Comment charger les résultats :**
```python
# src/analysis/quantitative.py
import pandas as pd
import jsonlines

def load_results(run_id: str, filename: str) -> pd.DataFrame:
    """Charge un fichier de résultats en DataFrame."""
    path = f"data/output/{run_id}/{filename}"
    items = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            items.append(obj)
    return pd.DataFrame(items)

# Charger les résultats Gemini et Mistral-Nemo
gemini = load_results("baseline_gemini_flash", "fr_unspecific.jsonl")
mistral = load_results("baseline_mistral_nemo", "fr_unspecific.jsonl")

# Comparer les longueurs de réponses
gemini["answer_len"] = gemini["answer"].str.len()
mistral["answer_len"] = mistral["answer"].str.len()

# Joindre sur l'ID pour comparer question par question
comparison = gemini.merge(mistral, on="id", suffixes=("_gemini", "_mistral"))
```

**Analyses possibles :**
- Longueur moyenne des réponses par langue
- Taux d'erreurs par modèle
- Diversité des réponses (unspecific) : les réponses diffèrent-elles entre langues ?
- Robustesse (specific) : les réponses sont-elles cohérentes quand le pays est fixé ?
- Embeddings + cosine similarity pour mesurer la similarité sémantique

---

## 5. RÉSUMÉ VISUEL DU FLUX

```
                    ┌─────────────────┐
                    │  configs/*.json  │  ← Quelle config utiliser
                    └────────┬────────┘
                             │
                             ▼
┌──────────┐    ┌───────────────────────┐    ┌──────────────────┐
│  data/   │    │                       │    │   data/output/   │
│  input/  │───▶│    runner.py          │───▶│   {run_id}/      │
│  *.jsonl │    │    (pipeline)         │    │   *.jsonl        │
└──────────┘    │                       │    │   config.json    │
                │  ┌─────────────────┐  │    │   run.log        │
                │  │ system_prompt.py│  │    │   run_summary    │
                │  │ (prompting)     │  │    └──────────────────┘
                │  └─────────────────┘  │             │
                │                       │             │
                │  ┌─────────────────┐  │             ▼
                │  │ gemini_provider │  │    ┌──────────────────┐
                │  │ ou              │  │    │  Lot D : analyse │
                │  │ mistral_nemo    │  │    │  quantitative.py │
                │  └─────────────────┘  │    │  semantic.py     │
                └───────────────────────┘    └──────────────────┘
                         ▲
                         │
                ┌────────┴────────┐
                │   api/routes.py │  ← Lot B : Streamlit appelle l'API
                │   POST /runs    │
                └─────────────────┘
```

