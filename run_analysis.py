"""
Script de test complet – Lot D (Analyse des résultats)
=======================================================

Lance les 3 méthodes d'analyse sur les meilleurs runs disponibles :

  1. Analyse Quantitative  → statistiques brutes (longueur, erreurs, vides)
  2. Analyse Sémantique    → scores de diversité et robustesse via embeddings
  3. LLM-as-a-Judge        → évaluation qualitative par Groq / Llama 3.3 70B

Runs utilisés
-------------
- llama_empathetic_unspecific : 5 langues *unspecific* (EN, FR, DE, ES, IT)
  → Idéal pour : stats quantitatives + diversité sémantique + juge diversité

- gemma_baseline_specifique : 3 langues *specific* (DE, EN, FR)
  → Idéal pour : robustesse sémantique + juge robustesse

Usage
-----
    python run_analysis.py            # mode complet (LLM Judge inclus)
    python run_analysis.py --no-judge # sans LLM Judge (pas de clé API requise)
    python run_analysis.py --sample 5 # limiter le nombre de prompts sémantiques
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import os

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_analysis")

# ── Constantes ────────────────────────────────────────────────────────────────

RUN_UNSPECIFIC = "llama_empathetic_unspecific"   # 5 langues unspecific
RUN_SPECIFIC   = "gemma_baseline_specifique"   # 3 langues specific
OUTPUT_DIR     = "data/output"

SEPARATOR = "=" * 70


# ── Helpers d'affichage ───────────────────────────────────────────────────────


def _print_section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def _print_json(data: dict, indent: int = 2) -> None:
    print(json.dumps(data, indent=indent, ensure_ascii=False))


def _print_summary(label: str, value) -> None:
    print(f"  ► {label:<40} {value}")


# ── 1. Analyse Quantitative ───────────────────────────────────────────────────


def run_quantitative(run_id: str) -> None:
    _print_section(f"MÉTHODE 1 – Analyse Quantitative  [{run_id}]")
    print(f"  Run : {run_id}")
    print(f"  Calcul des statistiques de longueur, erreurs et vides...\n")

    try:
        from src.analysis.quantitative import generate_report

        report = generate_report(run_id, output_dir=OUTPUT_DIR, save=True)
        stats = report["stats_by_file"]

        # Affichage par fichier
        print(f"  {'Fichier':<25} {'Total':>6} {'Moy. mots':>10} {'Moy. cars':>10} "
              f"{'Min':>5} {'Max':>5} {'Vides':>8} {'Erreurs':>8}")
        print(f"  {'-'*25} {'-'*6} {'-'*10} {'-'*10} {'-'*5} {'-'*5} {'-'*8} {'-'*8}")

        for key, s in stats.items():
            if key == "global":
                continue
            print(
                f"  {key:<25} {s['total']:>6} {s['avg_words']:>10.1f} {s['avg_chars']:>10.1f} "
                f"{s['min_words']:>5} {s['max_words']:>5} {s['empty_rate']:>8.2%} {s['error_rate']:>8.2%}"
            )

        g = stats["global"]
        print(f"\n  {'GLOBAL':<25} {g['total']:>6} {g['avg_words']:>10.1f} {g['avg_chars']:>10.1f} "
              f"{g['min_words']:>5} {g['max_words']:>5} {g['empty_rate']:>8.2%} {g['error_rate']:>8.2%}")

        print(f"\n  ✔ Rapport sauvegardé : data/output/{run_id}/analysis_quantitative.json")

    except Exception as exc:
        logger.error("Erreur analyse quantitative : %s", exc)
        print(f"\n  ✘ ERREUR : {exc}")


# ── 2. Analyse Sémantique ─────────────────────────────────────────────────────


def run_semantic(run_unspecific: str, run_specific: str, sample_size: int | None) -> None:
    _print_section(f"MÉTHODE 2 – Analyse Sémantique (Embeddings)")
    print(f"  Modèle : paraphrase-multilingual-MiniLM-L12-v2")
    if sample_size:
        print(f"  Limite : {sample_size} prompts par calcul (mode rapide)")
    print()

    from src.analysis import semantic

    # --- 2a. Score de Diversité (unspecific) ---
    print(f"  [2a] Score de Diversité Culturelle  [{run_unspecific}]")
    print(f"       Fichiers : *_unspecific  ({5} langues EN/FR/DE/ES/IT)")
    try:
        div = semantic.diversity_score(
            run_unspecific,
            output_dir=OUTPUT_DIR,
            sample_size=sample_size,
        )
        print(f"\n       Score diversité   : {div['score']:.4f}  (std: {div['score_std']:.4f})")
        print(f"       Prompts analysés  : {div['n_prompts']}")
        print(f"       Langues           : {', '.join(div['languages'])}")
        print(f"\n       Interprétation :")
        print(f"         • Plus le score est proche de 1.0, plus les réponses")
        print(f"           sont culturellement DISTINCTES entre les langues (✓ souhaité).")
        print(f"         • Score {div['score']:.2f} → ", end="")
        if div['score'] >= 0.30:
            print("bonne diversité culturelle détectée.")
        elif div['score'] >= 0.15:
            print("diversité modérée, le modèle s'adapte partiellement.")
        else:
            print("faible diversité, réponses très similaires entre langues.")

        # Sauvegarde individuelle du score diversité
        import json
        from pathlib import Path
        div_path = Path(OUTPUT_DIR) / run_unspecific / "analysis_diversity.json"
        div_save = {k: v for k, v in div.items() if k != "per_prompt"}
        div_path.write_text(json.dumps(div_save, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n       ✔ Rapport sauvegardé : data/output/{run_unspecific}/analysis_diversity.json")

    except Exception as exc:
        logger.error("Erreur diversité sémantique : %s", exc)
        print(f"\n       ✘ ERREUR : {exc}")

    # --- 2b. Score de Robustesse (specific) ---
    print(f"\n  [2b] Score de Robustesse Culturelle  [{run_specific}]")
    print(f"       Fichiers : *_specific  (DE, EN, FR)")
    try:
        rob = semantic.robustness_score(
            run_specific,
            output_dir=OUTPUT_DIR,
            sample_size=sample_size,
        )
        print(f"\n       Score robustesse  : {rob['score']:.4f}  (std: {rob['score_std']:.4f})")
        print(f"       Prompts analysés  : {rob['n_prompts']}")
        print(f"       Langues           : {', '.join(rob['languages'])}")
        print(f"\n       Interprétation :")
        print(f"         • Plus le score est proche de 1.0, plus les réponses")
        print(f"           sont COHÉRENTES malgré le contexte culturel (✓ souhaité).")
        print(f"         • Score {rob['score']:.2f} → ", end="")
        if rob['score'] >= 0.70:
            print("excellente robustesse, réponses très stables.")
        elif rob['score'] >= 0.50:
            print("bonne robustesse, cohérence satisfaisante.")
        else:
            print("robustesse insuffisante, réponses instables selon les langues.")

        # Sauvegarde
        rob_path = Path(OUTPUT_DIR) / run_specific / "analysis_robustness.json"
        rob_save = {k: v for k, v in rob.items() if k != "per_prompt"}
        rob_path.write_text(json.dumps(rob_save, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n       ✔ Rapport sauvegardé : data/output/{run_specific}/analysis_robustness.json")

    except Exception as exc:
        logger.error("Erreur robustesse sémantique : %s", exc)
        print(f"\n       ✘ ERREUR : {exc}")

    # --- 2c. Score combiné (si les deux ont fonctionné) ---
    print(f"\n  [2c] Résumé Scores Sémantiques")
    try:
        div_score = div["score"] if "div" in dir() else None
        rob_score = rob["score"] if "rob" in dir() else None
        if div_score is not None and rob_score is not None:
            if div_score + rob_score > 0:
                harmonic = round(2 * div_score * rob_score / (div_score + rob_score), 4)
            else:
                harmonic = 0.0
            print(f"       Score diversité       : {div_score:.4f}")
            print(f"       Score robustesse      : {rob_score:.4f}")
            print(f"       Score combiné (harm.) : {harmonic:.4f}")
    except Exception:
        pass


# ── 3. LLM-as-a-Judge ────────────────────────────────────────────────────────


def run_llm_judge(run_unspecific: str, run_specific: str, sample_size: int) -> None:
    _print_section(f"MÉTHODE 3 – LLM-as-a-Judge (Groq / Llama 3.3 70B)")

    # Vérification clé API
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        # Tenter de charger depuis .env
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("GROQ_API_KEY")
        except ImportError:
            pass

    if not api_key:
        print("  ✘ Clé GROQ_API_KEY non trouvée. Ajoutez-la dans votre .env")
        print("    Exemple : GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx")
        return

    print(f"  ✔ Clé API Groq détectée")
    print(f"  Échantillon : {sample_size} questions par dimension")
    print(f"  Attention   : {2 * sample_size} appels API (~{2 * sample_size * 2.5:.0f} secondes)")
    print()

    from src.analysis import llm_judge

    # --- 3a. Diversité (unspecific) ---
    print(f"  [3a] Évaluation Diversité Culturelle  [{run_unspecific}]")
    judge = None
    div_result = None
    try:
        judge = llm_judge.LLMJudge()
        div_result = llm_judge.evaluate_diversity(
            run_unspecific,
            output_dir=OUTPUT_DIR,
            sample_size=sample_size,
            judge=judge,
        )
        print(f"\n       Score moyen  : {div_result['avg_score']:.2f} / 5.0")
        print(f"       Évaluations  : {div_result['n_valid_scores']}/{div_result['sample_size']} valides")
        dist = div_result['score_distribution']
        print(f"       Distribution : " + "  ".join(f"★{k}:{v}" for k, v in dist.items()))
        print(f"\n       Détail des évaluations :")
        for ev in div_result["evaluations"][:3]:   # Afficher les 3 premiers
            print(f"         ID {ev.get('prompt_id')}: score={ev.get('score')} – {ev.get('reason','')[:80]}")
        if len(div_result["evaluations"]) > 3:
            print(f"         ... ({len(div_result['evaluations']) - 3} autres évaluations)")

    except Exception as exc:
        logger.error("Erreur LLM Judge diversité : %s", exc)
        print(f"\n       ✘ ERREUR : {exc}")

    # --- 3b. Robustesse (specific) ---
    print(f"\n  [3b] Évaluation Robustesse Culturelle  [{run_specific}]")
    rob_result = None
    try:
        if judge is None:
            judge = llm_judge.LLMJudge()
        rob_result = llm_judge.evaluate_robustness(
            run_specific,
            output_dir=OUTPUT_DIR,
            sample_size=sample_size,
            judge=judge,
        )
        print(f"\n       Score moyen  : {rob_result['avg_score']:.2f} / 5.0")
        print(f"       Évaluations  : {rob_result['n_valid_scores']}/{rob_result['sample_size']} valides")
        dist = rob_result['score_distribution']
        print(f"       Distribution : " + "  ".join(f"★{k}:{v}" for k, v in dist.items()))
        print(f"\n       Détail des évaluations :")
        for ev in rob_result["evaluations"][:3]:
            print(f"         ID {ev.get('prompt_id')}: score={ev.get('score')} – {ev.get('reason','')[:80]}")
        if len(rob_result["evaluations"]) > 3:
            print(f"         ... ({len(rob_result['evaluations']) - 3} autres évaluations)")

    except Exception as exc:
        logger.error("Erreur LLM Judge robustesse : %s", exc)
        print(f"\n       ✘ ERREUR : {exc}")

    # --- 3c. Score global ---
    print(f"\n  [3c] Score Global LLM Judge")
    if div_result and rob_result:
        avg_div = div_result["avg_score"]
        avg_rob = rob_result["avg_score"]
        global_avg = round((avg_div + avg_rob) / 2, 2) if (avg_div + avg_rob) > 0 else 0.0
        print(f"       Score diversité   : {avg_div:.2f} / 5.0")
        print(f"       Score robustesse  : {avg_rob:.2f} / 5.0")
        print(f"       Score global      : {global_avg:.2f} / 5.0")

        if global_avg >= 4.0:
            interp = "Excellent : forte diversité culturelle et robustesse."
        elif global_avg >= 3.0:
            interp = "Satisfaisant : bonne performance avec des axes d'amélioration."
        elif global_avg >= 2.0:
            interp = "Insuffisant : problèmes de diversité ou de robustesse."
        else:
            interp = "Très faible : réponses inadaptées sur le plan culturel."
        print(f"       Interprétation    : {interp}")

        # Sauvegarde rapport combiné
        import json
        from pathlib import Path
        full_report = {
            "run_unspecific": run_unspecific,
            "run_specific": run_specific,
            "analysis_type": "llm_judge",
            "judge_model": judge._model,
            "sample_size": sample_size,
            "diversity": div_result,
            "robustness": rob_result,
            "global_avg_score": global_avg,
            "interpretation": interp,
        }
        report_path = Path(OUTPUT_DIR) / run_unspecific / "analysis_llm_judge.json"
        report_path.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n       ✔ Rapport sauvegardé : data/output/{run_unspecific}/analysis_llm_judge.json")


# ── Point d'entrée ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test complet Lot D – Analyse des résultats ELOQUENT"
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Ignorer le LLM-as-a-Judge (pas de clé API nécessaire)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        metavar="N",
        help="Nombre de prompts à analyser pour sémantique/juge (défaut: 10)",
    )
    parser.add_argument(
        "--run-unspecific",
        default=RUN_UNSPECIFIC,
        help=f"Run ID avec fichiers unspecific (défaut: {RUN_UNSPECIFIC})",
    )
    parser.add_argument(
        "--run-specific",
        default=RUN_SPECIFIC,
        help=f"Run ID avec fichiers specific (défaut: {RUN_SPECIFIC})",
    )
    args = parser.parse_args()

    print(SEPARATOR)
    print("  ELOQUENT – Test Complet Lot D (Analyse des résultats)")
    print(SEPARATOR)
    print(f"  Run unspecific : {args.run_unspecific}")
    print(f"  Run specific   : {args.run_specific}")
    print(f"  Échantillon    : {args.sample} prompts")
    print(f"  LLM Judge      : {'DÉSACTIVÉ (--no-judge)' if args.no_judge else 'ACTIVÉ'}")

    # Chargement des variables d'environnement
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("Variables d'environnement chargées depuis .env")
    except ImportError:
        pass

    # ── 1. Quantitatif ────────────────────────────────────────────────────────
    run_quantitative(args.run_unspecific)
    if args.run_specific != args.run_unspecific:
        run_quantitative(args.run_specific)

    # ── 2. Sémantique ─────────────────────────────────────────────────────────
    run_semantic(
        args.run_unspecific,
        args.run_specific,
        sample_size=args.sample if args.sample > 0 else None,
    )

    # ── 3. LLM Judge ─────────────────────────────────────────────────────────
    if not args.no_judge:
        run_llm_judge(
            args.run_unspecific,
            args.run_specific,
            sample_size=args.sample,
        )

    # ── Résumé final ──────────────────────────────────────────────────────────
    _print_section("RÉSUMÉ FINAL")
    print(f"  ✔ Analyse quantitative  : data/output/{args.run_unspecific}/analysis_quantitative.json")
    if args.run_specific != args.run_unspecific:
        print(f"                            data/output/{args.run_specific}/analysis_quantitative.json")
    print(f"  ✔ Analyse sémantique    : data/output/{args.run_unspecific}/analysis_diversity.json")
    print(f"  ✔ Analyse sémantique    : data/output/{args.run_unspecific}/analysis_diversity.json")
    print(f"                            data/output/{args.run_specific}/analysis_robustness.json")
    if not args.no_judge:
        print(f"  ✔ LLM Judge             : data/output/{args.run_unspecific}/analysis_llm_judge.json")
    print()
    print("  Tous les rapports sont au format JSON et prêts pour le Lot C (visualisation).")
    print(SEPARATOR)


if __name__ == "__main__":
    main()

