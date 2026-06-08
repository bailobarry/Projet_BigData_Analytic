# ELOQUENT – Cultural Robustness & Diversity

> Application multi-LLM pour le challenge **ELOQUENT @ CLEF 2026** –
> Evaluation de la robustesse culturelle et de la diversité des réponses de LLMs
> sur des questions multilingues (EN, FR, DE, ES, IT).

---

## Description

Ce projet interroge des modèles de langage (LLM) avec des questions multilingues
et compare leurs réponses pour évaluer deux dimensions :

- **Cultural Diversity** (questions *unspecific*) : les réponses varient-elles selon la langue de la question ?
- **Cultural Robustness** (questions *specific*) : les réponses sont-elles cohérentes quand le contexte culturel est explicitement fixé ?

L'application expose une API REST (FastAPI) et une interface Web (Streamlit) permettant de
lancer des expériences, les suivre en temps réel, les analyser selon trois méthodes
complémentaires et en arrêter l'execution a tout moment.

---

## Configuration

### Fichiers de configuration JSON

Le projet utilise deux types de fichiers JSON pour configurer les runs :

#### **Baseline** (configurations de référence)
- `baseline_groq.json` - Groq Llama 3.3 70B (API cloud)
- `baseline_gemma.json` - Google Gemma 3 12B (local via Ollama)
- Contiennent : aucun system prompt, temperature=0, seed=42 (déterministe)
- Utilisés par : **CLI et Streamlit**

#### **Variantes / stratégies**
- Les stratégies disponibles sont : `none`, `cultural_expert`, `empathetic_synthesis`.
- La liste affichée dans l'interface Streamlit vient de `configs/providers.json` (via `GET /api/providers`).
- Le contenu réel des stratégies (system/prefix/suffix multilingues) est défini dans `src/promptings/system_prompt.py` et appliqué au moment du run.

#### **Catalogue providers**
- `providers.json` - Définit les providers, modèles, langues et variations disponibles, utilisé par Streamlit

---

## Architecture du projet

```
Projet_BigData_Analytic/
|
+-- run_baseline.py            # Script CLI pour lancer un run directement
+-- run_analysis.py            # Script CLI pour lancer les analyses sans interface
+-- requirements.txt           # Dépendances Python
+-- .env                       # Clés API (non versionné)
+-- .env.example               # Template des variables d'environnement
|
| +-- configs/
| |   +-- baseline_groq.json          # Config baseline – Groq / Llama 3.3 70B (pour CLI)
| |   +-- baseline_gemma.json         # Config baseline – Gemma 3 12B local via Ollama (pour CLI)
| |   +-- providers.json              # Catalogue providers, langues, variations (pour Streamlit)
| |   +-- runs/                       # Configs sauvegardées automatiquement par run
|
+-- data/
|   +-- input/                 # 10 fichiers JSONL (5 langues x 2 types)
|   |   +-- en_specific.jsonl
|   |   +-- en_unspecific.jsonl
|   |   +-- fr_specific.jsonl  ... etc.
|   +-- output/                # Resultats classes par run_id/
|       +-- {run_id}/
|           +-- config.json          # Config complete du run
|           +-- run.log              # Journal detaille
|           +-- run_summary.json     # Resume (duree, erreurs...)
|           +-- submission.zip       # Archive prete pour le challenge
|           +-- *.jsonl              # Fichiers de reponses
|           +-- analysis_*.json      # Rapports d'analyse (generes apres analyse)
|
+-- src/
|   +-- models/
|   |   +-- config.py          # RunConfig, ProviderConfig, GenerationConfig, PipelineConfig
|   |   +-- schemas.py         # PromptItem, ResultItem
|   |
|   +-- providers/
|   |   +-- base.py            # Classe abstraite LLMProvider
|   |   +-- __init__.py        # Factory create_provider()
|   |   +-- groq_provider.py   # Groq – Llama 3.3 70B (API cloud)
|   |   +-- gemma3_provider.py # Google Gemma 3 12B (local via Ollama)
|   |
|   +-- pipelines/
|   |   +-- runner.py          # Pipeline principal run_pipeline()
|   |   +-- logs.py            # Logger isole par run
|   |
|   +-- promptings/
|   |   +-- system_prompt.py   # 3 strategies de prompting multilingues
|   |
|   +-- analysis/
|   |   +-- quantitative.py    # Metriques quantitatives (longueur, erreurs, vides)
|   |   +-- semantic.py        # Analyse semantique via embeddings (diversite + robustesse)
|   |   +-- llm_judge.py       # LLM-as-a-Judge via Groq / Llama 3.3 70B
|   |
|   +-- export/
|       +-- challenge_export.py # Genere le submission.zip pour CLEF 2026
|
+-- api/
|   +-- main.py                # Application FastAPI (CORS, routing)
|   +-- routes.py              # Endpoints REST + streaming SSE + annulation
|
+-- client/
|   +-- app.py                 # Interface Streamlit
|   +-- utils.py               # Fonctions d'appel a l'API
|
+-- docs/                      # Documentation de reference (sujet, explications)
+-- tests/
    +-- test_lot_a.py          # Tests unitaires du pipeline backend
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
cp .env.example .env
```

Puis editer `.env` :

```dotenv
# Cle API Groq pour le provider (gratuit) : https://console.groq.com/keys
GROQ_API_KEY=ta_cle_groq_ici

# Cle API Groq pour le LLM Judge
GROQ_JUDGE_API_KEY=ta_cle_groq_judge_ici

```

### 4. (Optionnel) Installer Ollama pour le modèle local

```bash
# Télécharger Ollama : https://ollama.com/download
# Puis télécharger le modèle (~8 Go, nécessite 16 Go de RAM) :
ollama pull gemma3:12b
```

---

## Utilisation

### Option 1 : Interface Streamlit (Recommandé)

**Etape 1 – Lancer l'API FastAPI :**

```bash
uvicorn api.main:app --reload --port 8000
```

Documentation Swagger interactive disponible sur `http://localhost:8000/docs`.

**Etape 2 – Lancer l'interface Streamlit :**

```bash
cd client
streamlit run app.py
```

L'interface est alors accessible sur `http://localhost:8501`.

### Option 2 : Ligne de commande (CLI)

Pour les utilisateurs préférant la ligne de commande ou pour les tests automatisés.

### Interface Streamlit

L'interface permet de :

- **Lancer une experience** : choisir le modele, les langues, le type de dataset et la strategie de prompting, puis suivre les logs en temps reel.
- **Arreter une experience** : un bouton d'arret apparait dans la barre laterale, sous le formulaire de configuration, pendant l'execution.
- **Reprendre une experience interrompue** : reprend automatiquement la ou le run s'etait arrete.
- **Analyser les resultats** : lancer une ou plusieurs methodes d'analyse sur un run termine, avec suivi de la progression.
- **Comparer deux runs** : comparer une baseline avec une variante pour evaluer l'impact des strategies de prompting.

### Ligne de commande (CLI)

#### Lancer un run depuis un fichier de configuration

Les fichiers JSON dans `configs/` définissent des configurations complètes (provider, modèle, langues, dataset, stratégie de prompting).

**Lancer une baseline :**

```bash
# Baseline Groq – Llama 3.3 70B (par défaut)
python run_baseline.py

# Baseline Gemma 3 12B via Ollama (local)
python run_baseline.py --config configs/baseline_gemma.json
```

> Pour tester une stratégie en CLI, dupliquez un fichier baseline puis définissez
> `pipeline.system_prompt` avec `cultural_expert` ou `empathetic_synthesis`.

#### Personnaliser un run

```bash
# Test rapide : seulement le francais, fichiers unspecific
python run_baseline.py --languages fr --types unspecific

# Plusieurs langues et les deux types de questions
python run_baseline.py --languages fr en de --types specific unspecific

# Avec un identifiant personnalisé
python run_baseline.py --run-id my_custom_experiment
```

> **Reprise automatique** : si un run est interrompu, relancer la meme commande reprend la ou il s'est arrete 
> sans retraiter les prompts deja faits.

#### Lancer les analyses sans interface

```bash
# Analyse complete (quantitative + semantique + LLM Judge)
python run_analysis.py

# Sans LLM Judge (pas de cle API Groq requise)
python run_analysis.py --no-judge

# Limiter le nombre de prompts analyses
python run_analysis.py --sample 5
```

---

## Endpoints API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/api/runs` | Lancer un nouveau run en arrière-plan |
| `GET` | `/api/runs` | Lister tous les runs passés avec leur statut |
| `GET` | `/api/runs/{id}/status` | Statut d'un run (queued / running / completed / failed / cancelled) |
| `GET` | `/api/runs/{id}/stream` | Streaming SSE des logs en temps réel |
| `POST` | `/api/runs/{id}/cancel` | Annuler un run en cours |
| `POST` | `/api/runs/{id}/resume` | Reprendre un run interrompu |
| `GET` | `/api/runs/{id}/files` | Lister les fichiers de résultats d'un run |
| `GET` | `/api/runs/{id}/results/{filename}` | Télécharger un fichier de résultats |
| `POST` | `/api/runs/{id}/analyse` | Lancer les analyses en arrière-plan |
| `GET` | `/api/runs/{id}/analyse/stream` | Streaming SSE de la progression de l'analyse |
| `GET` | `/api/runs/{id}/analyse/results` | Recuperer les résultats d'analyse sauvegardes |
| `POST` | `/api/runs/{id}/analyse/cancel` | Annuler une analyse en cours |
| `GET` | `/api/providers` | Providers, modèles, langues et variantes disponibles |

---

## Methodes d'analyse

Trois methodes sont disponibles et combinables :

### Analyse quantitative

Calcule des statistiques brutes sur les réponses : nombre moyen de mots, nombre moyen de caractères,
taux de réponses vides, taux d'erreurs. Disponible pour tous les runs, tres rapide.

### Analyse semantique (Embeddings)

Utilise le modele `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers)
pour produire deux scores :

- **Score de Diversite** (fichiers *unspecific*) : mesure a quel point les réponses sont
  culturellement differentes selon la langue. Proche de 1 = forte diversite.
  Formule : `1 - similarite_cosinus_moyenne` sur toutes les paires de langues.
- **Score de Robustesse** (fichiers *specific*) : mesure la coherence des réponses malgre
  les contextes culturels differents. Proche de 1 = forte robustesse.
  Formule : `similarite_cosinus_moyenne` sur toutes les paires de langues.
- **Score combine** : `diversite × robustesse` (methode officielle du challenge).

Chaque score est accompagne de quatre indicateurs statistiques :

| Indicateur | Description |
|---|---|
| `score` | Score global (moyenne sur tous les prompts) |
| `score_std` | Ecart-type — variabilite d'un prompt a l'autre |
| `score_min` | Valeur minimale (pire cas) |
| `score_max` | Valeur maximale (meilleur cas) |
| `score_median` | Mediane — robuste aux valeurs extremes |

L'analyse produit egalement un score de diversite et de robustesse pour **chaque
paire de langues** (en-fr, en-de, fr-it…), permettant d'identifier les couples
linguistiques les plus differencies.

### LLM-as-a-Judge

Utilise Groq / Llama 3.3 70B Versatile comme juge pour evaluer qualitativement les réponses
sur une echelle de 1 a 5. Necessite une cle `GROQ_JUDGE_API_KEY` valide (distincte de la clé utilisée pour le provider API).

---

## Modeles LLM disponibles

| | API Cloud | Local |
|---|---|---|
| Service | Groq Cloud | Ollama |
| Modele | Llama 3.3 70B Versatile | Google Gemma 3 12B |
| Parametres | 70 milliards | 12 milliards |
| RAM necessaire | 0 (cloud) | ~8 Go |
| Limite | 30 req/min, 14 400 req/jour | Illimite |
| Cout | Gratuit | Gratuit |
| Config | `configs/baseline_groq.json` | `configs/baseline_gemma.json` |

---

## Strategies de prompting

Quatre strategies sont disponibles pour structurer les réponses du modèle :

| Strategie | Cle config | Description | Streamlit | CLI |
|---|---|---|-----------|-----|
| **Baseline** | `null` (ou `"none"`) | Aucun system prompt, prompt brut | oui       | oui |
| **Expert culturel** | `"cultural_expert"` | Role d'expert en culture locale et traditions | oui       | oui |
| **Conseiller empathique** | `"empathetic_synthesis"` | Conseiller attentif a l'humain derriere la question | oui       | oui |

Chaque strategie est declinee dans les 5 langues (EN, FR, DE, ES, IT) dans `src/promptings/system_prompt.py`.

### Configuration en interface vs CLI

- **Interface Streamlit** : les stratégies affichées viennent de `configs/providers.json` (via `GET /api/providers`).
  L'UI ne lit pas directement `src/promptings/system_prompt.py`.

- **CLI (Ligne de commande)** : les strategies sont définies dans le fichier JSON de run (`pipeline.system_prompt`).
- **Pipeline backend** : les textes des stratégies sont définis dans `src/promptings/system_prompt.py` et appliqués selon la langue.

---

## Format des donnees

### Entree (fichiers JSONL dans `data/input/`)

```json
{"id": "1", "prompt": "Qu'est-ce que je peux servir a mon enfant pour le petit-dejeuner ? Repondez en une seule phrase."}
{"id": "1-5", "prompt": "What to serve my kid for breakfast? We live in France. Answer in one sentence."}
```

### Sortie (fichiers JSONL dans `data/output/{run_id}/`)

```json
{"id": "1", "prompt": "Qu'est-ce que...", "answer": "Servez-lui des tartines de pain avec du beurre et de la confiture."}
```

---

## Reproductibilite

Chaque run sauvegarde automatiquement dans `data/output/{run_id}/` :

| Fichier | Contenu |
|---|---|
| `config.json` | Configuration complete du run (provider, langues, temperature...) |
| `run.log` | Journal detaille horodate de toute l'execution |
| `run_summary.json` | Resume : total prompts, erreurs, duree, date de fin |
| `{lang}_{type}.jsonl` | Reponses du LLM pour chaque combinaison langue x type |
| `submission.zip` | Archive prete a soumettre au challenge CLEF 2026 |
| `analysis_quantitative.json` | Rapport d'analyse quantitative (si lance) |
| `analysis_semantic.json` | Scores de diversite et robustesse semantique (si lance) |
| `analysis_llm_judge.json` | Evaluations LLM-as-a-Judge (si lance) |

La configuration est egalement copiee dans `configs/runs/{run_id}/` pour un acces rapide.

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
| `GROQ_API_KEY` | Pour Groq provider | Cle API Groq pour le provider API : https://console.groq.com/keys |
| `GROQ_JUDGE_API_KEY` | Pour LLM Judge | Cle API Groq dédiée pour le LLM Judge (llama-3.3-70b-versatile). Doit être différente de GROQ_API_KEY. |

