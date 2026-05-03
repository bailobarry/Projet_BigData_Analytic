# 🌍 ELOQUENT – Cultural Robustness & Diversity

> Application multi-LLM pour le challenge **ELOQUENT @ CLEF 2026** –  
> Évaluation de la robustesse culturelle et de la diversité des réponses de LLMs.

## 📋 Description

Ce projet interroge plusieurs modèles de langage (LLM) avec des questions multilingues
(anglais, français, allemand, espagnol, italien) et compare les réponses pour évaluer :

- **Cultural Diversity** (questions *unspecific*) : les réponses varient-elles selon la langue ?
- **Cultural Robustness** (questions *specific*) : les réponses sont-elles cohérentes quand le contexte culturel est fixé ?

## 🏗️ Architecture du projet

```
├── api/                       # Backend FastAPI
│   ├── main.py                # Point d'entrée de l'API
│   └── routes.py              # Endpoints REST (runs, résultats)
│
├── client/                    # Interface Streamlit (Lot B)
│   └── app.py
│
├── src/
│   ├── providers/             # Abstraction LLM + implémentations
│   │   ├── base.py            # Classe abstraite LLMProvider
│   │   ├── groq_provider.py      # Groq – Llama 3.3 70B (API cloud)
│   │   └── gemma3_provider.py    # Google Gemma 3 12B (local via Ollama)
│   │
│   ├── pipelines/             # Exécution des runs
│   │   ├── runner.py          # Pipeline principal
│   │   └── logs.py            # Configuration du logging
│   │
│   ├── models/                # Schémas de données et config
│   │   ├── schemas.py         # PromptItem, ResultItem
│   │   └── config.py          # RunConfig (Pydantic)
│   │
│   ├── promptings/            # Stratégies de prompting (Lot C)
│   │   └── system_prompt.py   # System prompts + templates
│   │
│   ├── analysis/              # Analyses quantitative/sémantique (Lot D)
│   │   ├── quantitative.py
│   │   └── semantic.py
│   │
│   └── export/                # Export challenge (Lot E)
│       └── challenge_export.py
│
│   ├── configs/                   # Fichiers de configuration
│   ├── baseline_groq.json         # Config baseline (Groq – Llama 3.3 70B)
│   ├── baseline_ollama.json       # Config baseline (Google Gemma 3 12B via Ollama)
│   └── runs/                  # Configs sauvegardées par run
│
├── data/
│   ├── input/                 # Fichiers JSONL du challenge (5 langues × 2 types)
│   └── output/                # Résultats des runs
│
├── docs/                      # Documentation de référence
├── run_baseline.py            # Script CLI pour lancer un run
├── requirements.txt           # Dépendances Python
└── .env.example               # Template clé API Google Gemini
```

## 🚀 Installation

### 1. Cloner le projet et créer un environnement virtuel

```bash
git clone <url-du-repo>
cd Projet_BigData_Analytic
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Configurer la clé API Groq

```bash
# Copier le template
cp .env.example .env

# Éditer .env et renseigner votre clé :
# GROQ_API_KEY : https://console.groq.com/keys (gratuit)
```

### 4. Installer Ollama pour le modèle local

```bash
# Installer Ollama : https://ollama.com/download
# Puis télécharger le modèle (~8 Go, nécessite 16 Go de RAM) :
ollama pull gemma3:12b
```

## ⚡ Utilisation

### Lancer une baseline (CLI)

```bash
# Baseline avec Groq Llama 3.3 70B (API cloud, recommandé)
python run_baseline.py --config configs/baseline_groq.json

# Baseline avec Google Gemma 3 12B (local via Ollama)
python run_baseline.py --config configs/baseline_ollama.json

# Test rapide : seulement les fichiers unspecific en français
python run_baseline.py --languages fr --types unspecific

# Avec un run ID personnalisé
python run_baseline.py --run-id test_rapide_01 --languages en --types unspecific
```

### Lancer l'API backend

```bash
uvicorn api.main:app --reload --port 8000
# Documentation Swagger : http://localhost:8000/docs
```

### Endpoints API principaux

| Méthode | Endpoint                        | Description                       |
|---------|---------------------------------|-----------------------------------|
| POST    | `/api/runs`                     | Lancer un nouveau run             |
| GET     | `/api/runs`                     | Lister tous les runs              |
| GET     | `/api/runs/{id}/status`         | Statut d'un run                   |
| GET     | `/api/runs/{id}/results/{file}` | Résultats JSONL d'un run          |
| GET     | `/api/providers`                | Providers et modèles disponibles  |

## 🤖 Modèles LLM utilisés (gratuits)

| | API (Cloud) | Local |
|---|---|---|
| **Service** | Groq Cloud | Ollama |
| **Modèle** | Llama 3.3 70B Versatile | Gemma 3 12B |
| **Paramètres** | 70 milliards | 12 milliards |
| **Multilingue** | ★★★★★ | ★★★★★ |
| **RAM nécessaire** | 0 (cloud) | ~8 Go |
| **Coût** | Gratuit (30 req/min) | Gratuit (illimité) |
| **Config** | `configs/baseline_groq.json` | `configs/baseline_ollama.json` |

## 📁 Format des données

### Entrée (fichiers JSONL)

```json
{"id": "1", "prompt": "Qu'est-ce je peux servir à mon enfant pour le petit-déjauner? Répondez en une seule phrase."}
{"id": "1-5", "prompt": "What to serve my kid for breakfast? We live in France... Answer in one sentence."}
```

### Sortie (fichiers JSONL avec réponse)

```json
{"id": "1", "prompt": "...", "answer": "Servez-lui des tartines de pain avec du beurre et de la confiture."}
```

## 📐 Reproductibilité

Chaque run sauvegarde automatiquement :
- `config.json` – la configuration complète utilisée
- `run_summary.json` – résumé (durée, erreurs, etc.)
- `run.log` – journal détaillé de l'exécution
- Les fichiers JSONL de résultats

Le tout dans `data/output/{run_id}/`.

## 👥 Répartition des lots

| Lot   | Responsable | Description                                |
|-------|-------------|--------------------------------------------|
| **A** | Backend     | Pipeline, providers LLM, config, baseline  |
| **B** | Interface   | Streamlit UI, lancement, visualisation     |
| **C** | Variantes   | Prompting, reformulation, plugins          |
| **D** | Analyse     | Métriques, embeddings, rapport qualitatif  |
| **E** | Rapport     | Export challenge, documentation, article   |

