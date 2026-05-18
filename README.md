# ELOQUENT – Cultural Robustness & Diversity

> Application multi-LLM pour le challenge **ELOQUENT @ CLEF 2026** –  
> Évaluation de la robustesse culturelle et de la diversité des réponses de LLMs.

---

## Description

Ce projet interroge des modèles de langage (LLM) avec des questions multilingues
(anglais, français, allemand, espagnol, italien) et compare leurs réponses pour évaluer :

- **Cultural Diversity** (questions *unspecific*) : les réponses varient-elles selon la langue de la question ?
- **Cultural Robustness** (questions *specific*) : les réponses sont-elles cohérentes quand le contexte culturel est explicitement fixé ?

---

## Architecture du projet

```
Projet_BigData_Analytic/
│
├── run_baseline.py            # Script CLI pour lancer un run
├── requirements.txt           # Dépendances Python
├── .env                       # Clés API (non versionné)
├── .env.example               # Template des variables d'environnement
│
├── configs/
│   ├── baseline_groq.json         # Config baseline – Groq Llama 3.3 70B (API)
│   ├── baseline_ollama.json       # Config baseline – Gemma 3 12B (local)
│   ├── variant_expert_ollama.json # Variante : rôle d'expert culturel
│   ├── variant_neutral_ollama.json# Variante : réponse neutre et factuelle
│   ├── variant_short_ollama.json  # Variante : réponse courte
│   ├── providers.json             # Catalogue providers / langues / variantes
│   └── runs/                      # Configs sauvegardées automatiquement par run
│
├── data/
│   ├── input/                 # 10 fichiers JSONL (5 langues × 2 types)
│   │   ├── en_specific.jsonl
│   │   ├── en_unspecific.jsonl
│   │   └── ...
│   └── output/                # Résultats classés par run_id/
│       └── {run_id}/
│           ├── config.json        # Config complète du run
│           ├── run.log            # Journal détaillé
│           ├── run_summary.json   # Résumé (durée, erreurs…)
│           ├── submission.zip     # Archive prête pour le challenge
│           └── *.jsonl            # Fichiers de réponses
│
├── src/
│   ├── models/
│   │   ├── config.py          # RunConfig, ProviderConfig, GenerationConfig, PipelineConfig
│   │   └── schemas.py         # PromptItem, ResultItem
│   │
│   ├── providers/
│   │   ├── base.py            # Classe abstraite LLMProvider
│   │   ├── __init__.py        # Factory create_provider()
│   │   ├── groq_provider.py   # Groq – Llama 3.3 70B (API cloud)
│   │   └── gemma3_provider.py # Google Gemma 3 12B (local via Ollama)
│   │
│   ├── pipelines/
│   │   ├── runner.py          # Pipeline principal run_pipeline()
│   │   └── logs.py            # Logger isolé par run (pas de mélange multi-runs)
│   │
│   ├── promptings/
│   │   └── system_prompt.py   # 3 stratégies de prompting multilingues
│   │
│   ├── analysis/
│   │   ├── quantitative.py    # Métriques quantitatives (Lot D)
│   │   └── semantic.py        # Analyse sémantique / embeddings (Lot D)
│   │
│   └── export/
│       └── challenge_export.py# Génère le submission.zip pour CLEF 2026
│
├── api/
│   ├── main.py                # Application FastAPI (CORS, routing)
│   └── routes.py              # Endpoints REST + streaming SSE
│
├── client/
│   └── app.py                 # Interface Streamlit (Lot B)
│
├── docs/                      # Documentation de référence (sujet, explications)
└── tests/
    └── test_lot_a.py          # Tests unitaires du pipeline backend
```

---

## Installation

### 1. Cloner le projet et créer un environnement virtuel

```bash
git clone <url-du-repo>
cd Projet_BigData_Analytic

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Configurer les variables d'environnement

```bash
# Copier le template
cp .env.example .env
```

Puis éditer `.env` :

```dotenv
# Clé API Groq (gratuit) : https://console.groq.com/keys
GROQ_API_KEY=ta_cle_groq_ici
```

### 4. (Optionnel) Installer Ollama pour le modèle local

```bash
# Télécharger Ollama : https://ollama.com/download
# Puis télécharger le modèle (~8 Go, nécessite 16 Go de RAM) :
ollama pull gemma3:12b
```

---

## Utilisation

### Lancer une baseline (CLI)

```bash
# Baseline Groq – Llama 3.3 70B
python run_baseline.py

# Baseline Gemma 3 12B via Ollama (local)
python run_baseline.py --config configs/baseline_ollama.json

# Exemple de test rapide : seulement le français, fichiers unspecific
python run_baseline.py --languages fr --types unspecific

# Tester plusieurs langues simultanément : français, anglais, allemand, etc. avec les deux types de questions
python run_baseline.py --languages fr en de --types specific unspecific

# Relancer un run interrompu (même run_id) pour reprendre là où il s'est arrêté
python run_baseline.py --run-id mon_test_01 --languages en --types unspecific
```

> **Reprise automatique** : si un run est interrompu, le relancer avec le même `--run-id` reprend là où il s'est arrêté sans retraiter les prompts déjà faits.

### Lancer l'API backend

```bash
uvicorn api.main:app --reload --port 8000
```

Documentation Swagger interactive : **http://localhost:8000/docs**

---

## Endpoints API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/api/runs` | Lancer un nouveau run en arrière-plan |
| `GET` | `/api/runs` | Lister tous les runs passés |
| `GET` | `/api/runs/{id}/status` | Statut d'un run (queued / running / completed / failed) |
| `GET` | `/api/runs/{id}/stream` | **Streaming SSE** – logs en temps réel |
| `GET` | `/api/runs/{id}/files` | Lister les fichiers de résultats d'un run |
| `GET` | `/api/runs/{id}/results/{filename}` | Télécharger un fichier de résultats JSONL |
| `GET` | `/api/providers` | Providers, modèles, langues et variantes disponibles |

---

## Modèles LLM utilisés (100% gratuits)

| | **API Cloud** | **Local** |
|---|---|---|
| **Service** | Groq Cloud | Ollama |
| **Modèle** | Llama 3.3 70B Versatile | Google Gemma 3 12B |
| **Paramètres** | 70 milliards | 12 milliards |
| **Date de sortie** | Décembre 2024 | Mars 2025 |
| **Multilingue EU** | ★★★★★ | ★★★★★ |
| **RAM nécessaire** | 0 (cloud) | ~8 Go |
| **Limite** | 30 req/min, 14 400 req/jour | Illimité |
| **Coût** | Gratuit | Gratuit |
| **Config** | `configs/baseline_groq.json` | `configs/baseline_ollama.json` |

---

## Stratégies de prompting

Trois stratégies sont disponibles en plus de la baseline (aucun prompt système) :

| Stratégie | Clé config | Description |
|---|---|---|
| Baseline | `null` | Aucun system prompt, prompt brut |
| Expert culturel | `"cultural_expert"` | Rôle d'expert en culture locale et traditions |
| Neutre et factuel | `"neutral"` | Réponses objectives sans opinion ni biais |
| Conseiller empathique | `"empathetic_synthesis"` | Conseiller attentif à l'humain derrière la question |

Chaque stratégie est déclinée dans les **5 langues** (EN, FR, DE, ES, IT) avec un `system`, un `prefix` et un `suffix` adaptés.

---

## Format des données

### Entrée (fichiers JSONL dans `data/input/`)

```json
{"id": "1", "prompt": "Qu'est-ce que je peux servir à mon enfant pour le petit-déjeuner ? Répondez en une seule phrase."}
{"id": "1-5", "prompt": "What to serve my kid for breakfast? We live in France. Answer in one sentence."}
```

### Sortie (fichiers JSONL dans `data/output/{run_id}/`)

```json
{"id": "1", "prompt": "Qu'est-ce que...", "answer": "Servez-lui des tartines de pain avec du beurre et de la confiture."}
```

---

## Reproductibilité

Chaque run sauvegarde automatiquement dans `data/output/{run_id}/` :

| Fichier | Contenu |
|---|---|
| `config.json` | Configuration complète du run (provider, langues, température…) |
| `run.log` | Journal détaillé horodaté de toute l'exécution |
| `run_summary.json` | Résumé : total prompts, erreurs, durée, date de fin |
| `{lang}_{type}.jsonl` | Réponses du LLM pour chaque combinaison langue × type |
| `submission.zip` | Archive prête à soumettre au challenge CLEF 2026 |

La configuration est également copiée dans `configs/runs/{run_id}/` pour un accès rapide.

---

## Tests

```bash
# Lancer tous les tests
pytest tests/ -v

# Lancer uniquement les tests du pipeline backend
pytest tests/test_lot_a.py -v
```

---

## Variables d'environnement

| Variable | Obligatoire | Description |
|---|---|---|
| `GROQ_API_KEY` | Pour Groq | Clé API Groq : https://console.groq.com/keys |
