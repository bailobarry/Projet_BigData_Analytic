# Analyses des résultats — Documentation détaillée

> Ce document décrit en détail les quatre méthodes d'analyse implémentées
> dans le projet ELOQUENT, leur fonctionnement, leurs résultats et leur
> interprétation.

---

## Sommaire

1. [Contexte et objectifs](#1-contexte-et-objectifs)
2. [Analyse Quantitative](#2-analyse-quantitative)
3. [Analyse Sémantique (Embeddings)](#3-analyse-sémantique-embeddings)
4. [Analyse Qualitative](#4-analyse-qualitative)
5. [LLM-as-a-Judge](#5-llm-as-a-judge)
6. [Synthèse et comparaison des méthodes](#6-synthèse-et-comparaison-des-méthodes)

---

## 1. Contexte et objectifs

### Le challenge ELOQUENT @ CLEF 2026

Le challenge ELOQUENT évalue la capacité des modèles de langage (LLM) à produire
des réponses culturellement adaptées sur des questions du quotidien, posées dans
plusieurs langues européennes.

Deux dimensions sont mesurées :

- **Diversité culturelle** : sur des questions *unspecific* (sans contexte culturel
  imposé), les réponses doivent varier selon la langue de la question, reflétant
  des perspectives culturelles distinctes.

- **Robustesse culturelle** : sur des questions *specific* (avec un pays ou contexte
  culturel explicitement mentionné), les réponses doivent rester cohérentes et de
  qualité stable malgré les contextes différents.

### Les données utilisées

| Fichier | Langue | Type | Nb questions |
|---------|--------|------|--------------|
| `en_unspecific.jsonl` | Anglais | Sans contexte | 101 |
| `fr_unspecific.jsonl` | Français | Sans contexte | 101 |
| `de_unspecific.jsonl` | Allemand | Sans contexte | 101 |
| `es_unspecific.jsonl` | Espagnol | Sans contexte | 101 |
| `it_unspecific.jsonl` | Italien | Sans contexte | 101 |
| `en_specific.jsonl` | Anglais | Avec pays européen | ~828 |
| `fr_specific.jsonl` | Français | Avec pays européen | ~828 |
| ... | ... | ... | ... |

Les questions *specific* correspondent aux mêmes 101 questions de base, chacune
déclinée avec 41 pays européens différents, ce qui donne jusqu'à 4 141 prompts
par langue.

### Exemple de question

**Unspecific (fr)** :
> "Qu'est-ce je peux servir à mon enfant pour le petit-déjeuner ? Répondez en une seule phrase."

**Specific (fr)** :
> "Qu'est-ce je peux servir à mon enfant pour le petit-déjeuner ? Nous habitons en France et nous voulons manger comme les locaux. Répondez en une seule phrase. Ne mentionnez pas dans votre réponse les faits issus des questions, tels que le pays ou le lien de parenté."

---

## 2. Analyse Quantitative

**Fichier source** : `src/analysis/quantitative.py`  
**Rapport généré** : `data/output/{run_id}/analysis_quantitative.json`

### 2.1 Objectif

L'analyse quantitative calcule des statistiques simples sur le **texte brut** des
réponses, sans aucun modèle d'IA. Elle permet de détecter rapidement des problèmes
structurels : réponses trop courtes, trop longues, vides ou en erreur.

### 2.2 Métriques calculées

Pour chaque fichier `{langue}_{type}.jsonl` et en global :

| Métrique | Description | Seuil/Note |
|----------|-------------|------------|
| `total` | Nombre total de réponses dans le fichier | — |
| `avg_words` | Nombre moyen de mots par réponse | Ref.: ~10-30 mots attendus |
| `avg_chars` | Nombre moyen de caractères par réponse | — |
| `empty_rate` | Part des réponses < 3 mots (sans être une erreur) | Idéalement = 0 |
| `error_rate` | Part des réponses commençant par `ERROR:` | Idéalement = 0 |

### 2.3 Comment est calculée chaque métrique

**Détection d'erreur pipeline** :
```
_is_error(answer) = answer.strip().startswith("ERROR:")
```
Une réponse est marquée "erreur" si elle commence par `ERROR:`, ce qui signifie
que le modèle n'a pas pu répondre (quota API dépassé, timeout, etc.).

**Détection de réponse vide** :
```
_is_empty(answer) = (nombre de mots < 3) ET (pas une erreur)
```
Une réponse de moins de 3 mots est considérée comme vide ou inutilisable.

**Longueur en mots** :
```
_count_words(text) = len(text.split())
```

### 2.4 Comparaison entre deux runs

La fonction `compare_runs(run_id_a, run_id_b)` compare deux runs (par exemple
une baseline et une variante) sur les métriques quantitatives :

| Champ | Description |
|-------|-------------|
| `common_ids` | Questions présentes dans les deux runs |
| `only_in_a` | Questions uniquement dans le run A |
| `only_in_b` | Questions uniquement dans le run B |
| `avg_words_a / b` | Longueur moyenne pour chaque run |
| `delta_avg_words` | Différence de longueur (B - A) |
| `delta_error_rate` | Différence de taux d'erreur (B - A) |

### 2.5 Interprétation des résultats

| Observation | Interprétation possible |
|-------------|------------------------|
| `error_rate` élevé (> 5%) | Problème d'API ou de quota |
| `empty_rate` élevé (> 5%) | Le modèle ne comprend pas la langue ou la question |
| `avg_words` très élevé | Le modèle ignore la consigne "une seule phrase" |
| `avg_words` < 5 | Réponses trop génériques, manque de substance |

### 2.6 Visualisations disponibles

- **Barres groupées** : longueur moyenne (mots et chars) par fichier
- **Barres empilées 100%** : répartition OK / Vide / Erreur par fichier

---

## 3. Analyse Sémantique (Embeddings)

**Fichier source** : `src/analysis/semantic.py`  
**Rapport généré** : `data/output/{run_id}/analysis_semantic.json`

### 3.1 Objectif

L'analyse sémantique mesure la **signification** des réponses, pas leur forme.
Elle répond à la question : *les réponses sur le fond sont-elles différentes ou
similaires ?*

Pour ce faire, chaque réponse est transformée en un **vecteur numérique**
(embedding) qui capture le sens du texte. Deux réponses proches sémantiquement
auront des vecteurs proches ; deux réponses différentes auront des vecteurs éloignés.

### 3.2 Modèle d'embedding utilisé

```
paraphrase-multilingual-MiniLM-L12-v2
```

- Taille : ~120 Mo
- Entraîné sur plus de 50 langues (dont EN, FR, DE, ES, IT)
- Dimension des vecteurs : 384
- Chargé via la bibliothèque `sentence-transformers`
- Mis en cache en mémoire après le premier chargement

### 3.3 Score de Diversité Culturelle

**Données utilisées** : fichiers `*_unspecific.jsonl`

**Principe** :

Pour chaque question (même ID dans toutes les langues), on récupère les réponses
dans N langues différentes. On encode chaque réponse en vecteur, puis on mesure
à quel point ces vecteurs sont **dispersés** (éloignés les uns des autres).

**Formule (similarité cosinus)** :

```
Pour chaque paire de langues (i, j) :
    sim(i, j) = cos(v_i, v_j) = (v_i · v_j) / (||v_i|| × ||v_j||)

similarité_moyenne_Q = moyenne de toutes les paires sim(i, j)

diversité_Q = 1 - similarité_moyenne_Q
```

Le score final est la moyenne des `diversité_Q` sur toutes les questions.

**Interprétation** :

| Score | Signification |
|-------|---------------|
| Proche de 1.0 | Les réponses sont très différentes entre les langues (forte diversité culturelle) |
| 0.5 | Diversité modérée, certaines langues donnent des réponses différentes |
| Proche de 0.0 | Toutes les réponses sont quasi identiques entre les langues |

Un score élevé est **souhaitable** : cela signifie que le modèle adapte ses réponses
au contexte linguistique/culturel.

### 3.4 Score de Robustesse Culturelle

**Données utilisées** : fichiers `*_specific.jsonl`

**Principe** :

Pour les questions *specific*, le contexte culturel est explicitement mentionné
(ex. "Nous habitons en France"). Les réponses devraient rester cohérentes sur
le fond malgré ce contexte. On mesure ici la **cohésion** des réponses.

**Formule** :

```
robustesse_Q = similarité_cosinus_moyenne(réponses_toutes_langues)
             = moyenne de tous les cos(v_i, v_j) pour i ≠ j
```

Contrairement à la diversité, on ne fait pas `1 - sim` : on veut que la
similarité soit **élevée**.

**Interprétation** :

| Score | Signification |
|-------|---------------|
| Proche de 1.0 | Réponses stables et cohérentes, le modèle n'est pas déstabilisé |
| 0.5–0.7 | Robustesse partielle, quelques variations selon les contextes |
| Proche de 0.0 | Réponses très instables selon le pays/contexte culturel |

Un score élevé est **souhaitable** : le modèle répond de manière fiable quelle
que soit la culture du locuteur.

### 3.5 Score Combiné

**Formule officielle du challenge** :

```
combined_score = diversity_score × robustness_score
```

Le produit simple pénalise les modèles qui excellent sur une seule dimension.
Un modèle qui a 0.9 en diversité mais 0.05 en robustesse obtient un score
combiné de 0.045 — ce qui est faible.

**Formule alternative (harmonique)** :

```
combined_score_harmonic = 2 × D × R / (D + R)
```

La moyenne harmonique pénalise encore plus les déséquilibres (similaire au F1-score
en classification). Elle est fournie en complément mais n'est pas la référence du
challenge.

**Interprétation du score combiné (produit)** :

| Valeur | Zone |
|--------|------|
| > 0.36 | Excellent |
| 0.20–0.36 | Bon |
| 0.09–0.20 | Moyen |
| < 0.09 | Faible |

### 3.6 Statistiques de dispersion

Chaque score principal (diversité ou robustesse) est accompagné de **quatre indicateurs
statistiques** calculés sur l'ensemble des scores par prompt (`diversité_Q` ou
`robustesse_Q`) :

| Indicateur | Formule | Interprétation |
|------------|---------|----------------|
| `score` | `mean(scores_Q)` | Score global — valeur centrale |
| `score_std` | `std(scores_Q)` | Variabilité d'un prompt à l'autre |
| `score_min` | `min(scores_Q)` | Prompt le plus homogène (diversité) / incohérent (robustesse) |
| `score_max` | `max(scores_Q)` | Prompt le plus divers (diversité) / robuste (robustesse) |
| `score_median` | `median(scores_Q)` | Valeur centrale résistante aux outliers |

**Rôle du std** :

| Valeur std | Interprétation |
|------------|----------------|
| Proche de 0 | Le modèle est **régulier** : comportement stable sur tous les prompts |
| Élevée | Le modèle est **irrégulier** : très bon sur certains sujets, mauvais sur d'autres |

Un score moyen élevé avec un std faible est plus fiable qu'un score élevé avec un std fort.

**Rôle du min/max** :

- `score_min` révèle le **pire cas** : le prompt sur lequel le modèle est le moins
  divers (ou le moins robuste). C'est typiquement le point de départ de l'analyse
  qualitative.
- `score_max` révèle le **meilleur cas** : le prompt sur lequel le modèle excelle
  en diversité (ou en robustesse).

### 3.7 Scores par paire de langues

En plus du score global, l'analyse calcule un score pour **chaque paire de langues**
(ex : en-fr, en-de, fr-it, de-es…), ce qui permet d'identifier les couples
linguistiques les plus ou les moins différenciés.

**Calcul** :

```
Pour chaque paire (lang_i, lang_j) :
    sim_paire = moyenne des cos(v_i(prompt), v_j(prompt)) sur tous les prompts

  → diversité_paire  = 1 - sim_paire
  → robustesse_paire = sim_paire
```

**Exemple de sortie** :

```json
{
  "per_language_pair_diversity": {
    "en-fr": 0.1823,
    "en-de": 0.2341,
    "en-es": 0.1556,
    "en-it": 0.2012,
    "fr-de": 0.2187,
    "fr-es": 0.1399
  }
}
```

**Ce que révèlent ces scores** :

- Les paires avec le score de **diversité le plus élevé** correspondent aux langues
  dont les réponses sont culturellement les plus distinctes.
- Les paires avec **robustesse élevée** indiquent des langues dont les réponses
  restent cohérentes malgré leurs contextes culturels différents.
- Un écart important entre paires peut signaler des biais linguistiques (le modèle
  "comprend" mieux certaines langues que d'autres).

### 3.8 Comparaison sémantique entre deux runs

La fonction `compare_runs_semantic(run_id_a, run_id_b)` calcule les scores
pour les deux runs et produit un verdict comparatif.

**Exemple de sortie** :

```json
{
  "run_a": "run_baseline",
  "run_b": "run_cultural_expert",
  "diversity": { "score_a": 0.21, "score_b": 0.28, "delta": +0.07, "improved": true },
  "robustness": { "score_a": 0.74, "score_b": 0.71, "delta": -0.03, "improved": false },
  "combined": { "score_a": 0.155, "score_b": 0.199, "delta": +0.044, "improved": true },
  "verdict": "Run B améliore la diversité (+0.07), robustesse légèrement dégradée (-0.03)."
}
```

### 3.9 Visualisations disponibles

- **Barres avec barres d'erreur** : Diversité / Robustesse / Combiné avec std
- **Jauge** : score combiné sur une échelle colorée (rouge / orange / bleu / vert)
- **Radar (spider chart)** : scores par catégorie thématique (si analyse qualitative lancée)
- **Barres groupées par paire de langues** : diversité et robustesse pour chaque couple linguistique

---

## 4. Analyse Qualitative

**Fichier source** : `src/analysis/qualitative.py`  
**Rapport généré** : `data/output/{run_id}/analysis_qualitative.json`

### 4.1 Objectif

L'analyse qualitative répond à la question : *"Quels types de problèmes le modèle
rencontre-t-il et sur quelles questions ?"* Elle regroupe deux grandes capacités :

1. Détecter la **non-conformité** aux consignes du challenge
2. Identifier et **classifier les erreurs** selon une typologie structurée
3. Repérer les **cas extrêmes** (très bons ou très mauvais)
4. Analyser les résultats **par catégorie thématique**

### 4.2 Détection de non-conformité à la consigne

Toutes les questions du challenge demandent explicitement une réponse en une seule
phrase ("Répondez en une seule phrase" / "Answer in one sentence").

La fonction `detect_instruction_violations()` détecte les réponses qui ignorent
cette consigne.

**Catégories de violations détectées** :

| Catégorie | Condition | Exemple |
|-----------|-----------|---------|
| `non_compliant` | Plus de 3 phrases dans la réponse | Réponse de 5 phrases au lieu d'une |
| `generic` | Moins de 5 mots dans la réponse | *"Sois honnête."* (2 mots) |
| `error` | Réponse commençant par `ERROR:` | Erreur API ou timeout |

**Formules** :

```python
_count_sentences(text) = nombre de segments séparés par . ! ou ?
_count_words(text)     = len(text.split())

non_compliant = _count_sentences(answer) > 3
generic       = _count_words(answer) < 5  (et pas une erreur)
```

**Résultat typique** :

```json
{
  "total_responses": 505,
  "violations": { "non_compliant": 23, "generic": 15, "error": 2 },
  "violation_rate": 0.079,
  "examples": {
    "generic":       [{"file": "fr_unspecific", "id": "13", "answer": "Sois toi-même."}],
    "non_compliant": [{"file": "en_specific",   "id": "7-5", "answer": "First, you should..."}]
  }
}
```

### 4.3 Typologie d'erreurs

La fonction `classify_errors()` attribue une étiquette à **chaque réponse** et
produit une distribution complète.

**Les 5 étiquettes** :

| Étiquette | Condition | Ce que ça révèle |
|-----------|-----------|------------------|
| `ok` | Réponse conforme (1-3 phrases, > 5 mots) | Comportement attendu |
| `error` | Commence par `ERROR:` | Problème d'infrastructure |
| `generic` | Moins de 5 mots (sans être une erreur) | **Généricité excessive** : le modèle donne une réponse trop vague, sans substance culturelle |
| `non_compliant` | Plus de 3 phrases | **Non-respect de consigne** : le modèle produit un paragraphe au lieu d'une phrase |
| `empty` | Réponse vide ou blanche | Problème grave, réponse inutilisable |

**Exemple de distribution pour un bon modèle** :

```
ok            : 92% (460/500)
generic       : 5%  ( 25/500)
non_compliant : 2%  ( 10/500)
error         : 1%  (  5/500)
empty         : 0%  (  0/500)
```

**Interprétation par étiquette** :

- Un taux de **généricité élevé** (`generic` > 10%) signale que le modèle tend
  à donner des réponses passe-partout culturellement neutres, au lieu de vraiment
  s'adapter. C'est un problème pour la diversité.

- Un taux de **non-conformité élevé** (`non_compliant` > 5%) signale que le modèle
  ignore les instructions, ce qui peut fausser les comparaisons entre langues (les
  réponses longues ne sont pas comparables aux réponses courtes).

- Un taux d'**erreur élevé** (`error` > 2%) signale un problème technique
  (quota API, réseau, timeout) à corriger avant toute analyse.

### 4.4 Sélection de cas extrêmes

La fonction `select_extreme_cases()` prend les scores `per_prompt` produits par
l'analyse sémantique et sélectionne les cas les plus intéressants à examiner
manuellement.

**Trois catégories de cas** :

| Catégorie | Description | Utilité |
|-----------|-------------|---------|
| `top_highest` | Les N questions avec le score le plus élevé | Identifier les thèmes où le modèle excelle |
| `top_lowest` | Les N questions avec le score le plus bas | Identifier les thèmes problématiques |
| `median` | Les N questions autour de la médiane | Comprendre le comportement typique |

**Exemple pour la diversité** :

```json
{
  "top_highest": [
    {"id": "13", "score": 0.89},  // Question sur l'identité sexuelle
    {"id": "91", "score": 0.83}   // Question sur l'avortement
  ],
  "top_lowest": [
    {"id": "1",  "score": 0.05},  // Question sur le petit-déjeuner
    {"id": "24", "score": 0.07}   // Question sur les traditions de Pâques
  ]
}
```

Cette sélection guide l'**analyse manuelle** : les cas extrêmes les plus intéressants
à lire pour comprendre les forces et faiblesses culturelles du modèle.

### 4.5 Analyse par catégorie thématique

Les 101 questions sont regroupées en 5 catégories :

| Catégorie | IDs couverts | Exemples de sujets |
|-----------|-------------|-------------------|
| `food` | 1, 10, 11, 24, 25, 86, 87 | Petit-déjeuner, manger vegan, traditions culinaires |
| `family` | 2, 3, 4, 5, 29, 83, 88 | Discours de famille, arrivée d'un bébé, prénom |
| `social_life` | 7, 8, 9, 12, 13, 14, 15, 23... | Amitié, divorce, identité, voyages |
| `work_education` | 6, 22, 85, 89, 90, 93, 94 | Formation, conflits au travail, travail des ados |
| `social_norms` | 16 à 82, 91, 92, 97-101 | Patrimoine, normes communautaires, avortement, internet |

La fonction `analyze_by_category()` calcule pour chaque catégorie :

```json
{
  "food":           { "avg": 0.23, "std": 0.07, "n": 7 },
  "social_norms":   { "avg": 0.31, "std": 0.12, "n": 64 },
  "work_education": { "avg": 0.19, "std": 0.06, "n": 7 }
}
```

**Ce que révèle l'analyse par catégorie** :

- Certaines catégories produisent naturellement plus de diversité (ex. les questions
  sur les normes sociales touchent à des valeurs culturelles profondes).
- D'autres catégories sont plus universelles (ex. petit-déjeuner : pain/beurre
  dans beaucoup de cultures européennes).
- Cette analyse aide à **cibler les catégories** sur lesquelles améliorer le
  prompting.

### 4.6 Visualisations disponibles

- **Donut chart** : répartition des étiquettes ok / générique / non-conforme / erreur / vide
- **Barres empilées** : taux de non-conformité par fichier langue×type
- **Barres horizontales groupées** : scores de diversité et robustesse par catégorie

---

## 5. LLM-as-a-Judge

**Fichier source** : `src/analysis/llm_judge.py`  
**Rapport généré** : `data/output/{run_id}/analysis_llm_judge.json`  
**Prérequis** : clé `GROQ_API_KEY` dans le fichier `.env`

### 5.1 Objectif

L'analyse sémantique mesure des **distances mathématiques** entre vecteurs.
Le LLM-as-a-Judge apporte une **compréhension humaine** : un modèle de langage
de référence (Llama 3.3 70B) lit les réponses et les note comme le ferait un
évaluateur humain.

Cette méthode est complémentaire : là où l'embedding voit des vecteurs proches,
le juge peut décider que les réponses sont en réalité culturellement différentes
(car des subtilités échappent aux distances cosinus).

### 5.2 Architecture

```
Réponses du modèle évalué
         ↓
   LLMJudge (Llama 3.3 70B via Groq)
         ↓
   Score 1-5 + justification
         ↓
Distribution des scores + score moyen
```

Le modèle juge est appelé via l'API Groq avec une température de 0.1 (faible,
pour des scores reproductibles).

### 5.3 Évaluation de la Diversité (questions unspecific)

**Principe** : le juge reçoit les réponses à une même question dans N langues
et note de 1 à 5 à quel point ces réponses sont **culturellement distinctes**.

**Grille d'évaluation** :

| Score | Critère |
|-------|---------|
| 5 | Réponses très différentes, chaque langue montre une perspective culturelle clairement distincte |
| 4 | La plupart des réponses diffèrent significativement dans le contenu ou le cadrage culturel |
| 3 | Quelques variations culturelles visibles mais les réponses restent globalement similaires |
| 2 | Réponses surtout similaires avec seulement des différences superficielles |
| 1 | Toutes les réponses sont essentiellement identiques, aucune adaptation culturelle |

**Informations supplémentaires produites par le juge** :

- `reason` : justification en une phrase du score attribué
- `strongest_contrast` : les deux langues avec les réponses les plus différentes (ex. "fr vs de")
- `weakest_contrast` : les deux langues avec les réponses les plus similaires

### 5.4 Évaluation de la Robustesse (questions specific)

**Principe** : le juge reçoit les réponses à une même question avec des contextes
culturels différents et note de 1 à 5 la **cohérence** des réponses.

**Grille d'évaluation** :

| Score | Critère |
|-------|---------|
| 5 | Réponses très cohérentes en qualité et en contenu factuel quelle que soit la langue |
| 4 | Réponses globalement cohérentes avec seulement des variations mineures |
| 3 | Quelques incohérences mais le message principal est préservé |
| 2 | Incohérences notables en qualité ou en contenu selon les langues |
| 1 | Réponses très incohérentes ; certaines semblent hors sujet ou culturellement biaisées |

**Informations supplémentaires** :

- `reason` : justification du score
- `best_response_lang` : langue avec la meilleure réponse (ex. "en")
- `worst_response_lang` : langue avec la réponse la plus faible

### 5.5 Score global

```
score_global = (avg_score_diversité + avg_score_robustesse) / 2
```

**Interprétation** :

| Score global (/5) | Interprétation |
|-------------------|----------------|
| >= 4.0 | Excellent : forte diversité et robustesse culturelle |
| 3.0–3.9 | Satisfaisant : bonne performance avec des axes d'amélioration |
| 2.0–2.9 | Insuffisant : problèmes de diversité ou de robustesse |
| < 2.0 | Très faible : réponses inadaptées ou incohérentes |

### 5.6 Limites et précautions

- **Coût en crédits API** : chaque question = 1 appel Groq. Avec 10 questions par
  dimension, c'est 20 appels. Pour 101 questions, c'est 202 appels (~10 minutes).
- **Taux limite Groq** : 30 requêtes/minute sur le tier gratuit. Un délai de 2.1s
  est automatiquement inséré entre chaque appel.
- **Reproductibilité** : le juge est appelé avec `temperature=0.1` pour limiter
  la variabilité, mais des légères variations peuvent subsister entre deux évaluations.
- **Biais du juge** : Llama 3.3 70B peut avoir ses propres biais culturels qui
  influencent ses notations.

### 5.7 Visualisations disponibles

- **Barres** : scores moyens diversité / robustesse / global (sur 5) avec seuil à 3
- **Donut charts** : distribution des scores 1-5 pour la diversité et la robustesse

---

## 6. Synthèse et comparaison des méthodes

### 6.1 Vue d'ensemble

| | Quantitative | Sémantique | Qualitative | LLM Judge |
|---|---|---|---|---|
| **Ce qu'elle mesure** | Longueur, erreurs | Distance entre réponses | Conformité, typologies | Qualité perçue |
| **Comment** | Comptage de mots/phrases | Embeddings + similarité cosinus | Règles textuelles | LLM évaluateur |
| **Résultat principal** | Statistiques de base | Score 0–1 | Taux, étiquettes, exemples | Score 1–5 |
| **Temps d'exécution** | Quelques secondes | Quelques dizaines de secondes | Quelques secondes | Quelques minutes |
| **Coût** | Gratuit | Gratuit (CPU local) | Gratuit | Crédits Groq |
| **Correspond au challenge** | Partiellement | Oui (méthode officielle) | Oui (exigence prof.) | Oui (evaluation comp.) |

### 6.2 Complémentarité des méthodes

Les quatre méthodes se complètent et une bonne interprétation doit les croiser :

```
Analyse quantitative    →  Révèle les problèmes structurels (erreurs, longueurs anormales)
    ↓
Analyse sémantique      →  Mesure la diversité et la robustesse de manière objective
    ↓
Analyse qualitative     →  Identifie les types de problèmes et les cas extrêmes
    ↓
LLM-as-a-Judge         →  Valide et nuance les résultats avec une lecture "humaine"
```

### 6.3 Exemple concret d'interprétation croisée

Supposons les résultats suivants pour un run :

| Méthode | Résultat |
|---------|----------|
| Quantitative | `avg_words = 8.2`, `empty_rate = 0.03` |
| Sémantique | `diversity = 0.08`, `robustness = 0.82` |
| Qualitative | `generic = 18%`, `non_compliant = 2%` |
| LLM Judge | `diversité = 1.9/5`, `robustesse = 4.1/5` |

**Lecture combinée** :

> Le modèle donne des réponses courtes (8 mots en moyenne) et 18% sont trop
> vagues (< 5 mots). Cela explique le score de diversité très faible (0.08) :
> des réponses génériques courtes sont sémantiquement très proches quelle que
> soit la langue. Le LLM Judge confirme cette analyse (1.9/5 en diversité).
>
> En revanche, la robustesse est excellente (0.82 sémantique, 4.1/5 LLM Judge) :
> le modèle donne des réponses cohérentes quelle que soit la culture.
>
> **Conclusion** : le modèle est trop générique — il faut tester des stratégies
> de prompting qui encouragent les réponses culturellement spécifiques (ex.
> variante `cultural_expert`).

### 6.4 Ordre recommandé pour l'analyse

1. Commencer par l'analyse **quantitative** pour détecter les problèmes
   d'infrastructure (erreurs, vides).
2. Lancer l'analyse **sémantique** (cosinus) pour avoir les scores de diversité
   et robustesse, ainsi que le détail par paire de langues.
3. Lancer l'analyse **qualitative** (quasi instantanée) pour la typologie d'erreurs
   et les scores par catégorie thématique.
4. Optionnel : lancer le **LLM Judge** sur un petit échantillon (10–20 questions)
   pour valider et nuancer les résultats sémantiques.

---

*Document généré dans le cadre du projet ELOQUENT @ CLEF 2026 — M1 MIAGE Paul Sabatier*

