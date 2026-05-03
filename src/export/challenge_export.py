"""
Export de soumission ELOQUENT.
Construis automatiquement le submission_metadata.json.
"""

from __future__ import annotations
import json
import zipfile
import argparse
from pathlib import Path



# Lecture des fichiers du run


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_run_artifacts(run_id: str) -> tuple[dict, dict]:
    """Retourne (config, summary) depuis data/output/{run_id}/"""
    base = Path("data/output") / run_id
    if not base.exists():
        raise FileNotFoundError(
            f"Dossier de run introuvable : {base}\n"
            f"Lance d'abord : python run_baseline.py --run-id {run_id}"
        )
    config = _read_json(base / "config.json")

    summary_path = base / "run_summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {}

    return config, summary



# Construction du metadata
def build_metadata(run_id: str, team: str, submission_id: str | None = None) -> dict:
    """
    Construit le submission_metadata.json à partir des fichiers
    sauvegardés automatiquement par le pipeline.

    """
    config, summary = load_run_artifacts(run_id)

    # ── Provider / modèle (depuis provider{}) ─────────────────────
    provider = config.get("provider", {})
    model    = provider.get("model", "unknown")          # "gemini-2.0-flash" ou "mistral-nemo"
    ptype    = provider.get("type", "unknown")           # "gemini" ou "ollama"

    # ── Langues (depuis pipeline{}) ───────────────────────────────
    pipeline  = config.get("pipeline", {})
    languages = pipeline.get("languages", [])            # ["en","fr","de","es","ru"]

    # ── Génération (depuis generation{}) ──────────────────────────
    gen         = config.get("generation", {})
    temperature = gen.get("temperature", 0.0)
    max_tokens  = gen.get("max_tokens", 256)
    seed        = gen.get("seed", 42)

    # ── Prompting (depuis pipeline{}) ─────────────────────────────
    # system_prompt  → None en baseline vanilla
    # prompt_template → None en baseline vanilla
    system_prompt   = pipeline.get("system_prompt") or ""
    prompt_template = pipeline.get("prompt_template") or ""

    # ── Date : depuis finished_at du summary ──────────────────────
    finished_at = summary.get("finished_at", config.get("created_at", ""))
    run_date    = finished_at[:10] if finished_at else "unknown"

    # ── Notes : depuis description du config ──────────────────────
    notes = (
        config.get("description")
        or f"Run automatique — modèle : {model}, provider : {ptype}"
    )

    # ── submissionid : argument CLI ou run_id ─────────────────────
    sub_id = submission_id or config.get("run_id", run_id)

    return {
        "team"        : team,
        "system"      : f"{ptype}-{model}",              # ex: "gemini-gemini-2.0-flash"
        "model"       : model,
        "submissionid": sub_id,
        "date"        : run_date,
        "label"       : "eloquent-2026-cultural",
        "languages"   : languages,
        "modifications": {
            "system_prompt"         : system_prompt,
            "prompt_prefix_english" : "",
            "prompt_suffix_english" : prompt_template,
            "generation_params"     : {
                "do_sample"     : temperature > 0,
                "max_new_tokens": max_tokens,
                "seed"          : seed,
                "temperature"   : temperature,
            },
            "notes": notes,
        },
    }


# ─────────────────────────────────────────────
# Génération du ZIP
# ─────────────────────────────────────────────

def generate_submission(
    run_id       : str,
    team         : str,
    submission_id: str | None = None,
    output_dir   : str = "data/submission",
) -> Path:
    """
    Génère submission_{sub_id}.zip dans output_dir.
    Lit tout depuis data/output/{run_id}/.
    """
    run_output = Path("data/output") / run_id
    submit_dir = Path(output_dir)
    submit_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Lecture du run : {run_output}")

    # 1. Construire et sauvegarder le metadata
    metadata      = build_metadata(run_id, team, submission_id)
    metadata_path = submit_dir / "submission_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("✅ submission_metadata.json généré")
    print(f"   modèle     : {metadata['model']}")
    print(f"   langues    : {metadata['languages']}")
    print(f"   sub id     : {metadata['submissionid']}")
    print(f"   date       : {metadata['date']}")
    print(f"   do_sample  : {metadata['modifications']['generation_params']['do_sample']}")
    print(f"   max_tokens : {metadata['modifications']['generation_params']['max_new_tokens']}")
    print(f"   sys_prompt : '{metadata['modifications']['system_prompt'] or '(baseline vanilla)'}'")

    # 2. Créer le ZIP
    sub_id   = metadata["submissionid"]
    zip_path = submit_dir / f"submission_{sub_id}.zip"
    missing, added = [], []

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # metadata en premier (obligatoire)
        zf.write(metadata_path, "submission_metadata.json")

        # fichiers JSONL de résultats
        for lang in metadata["languages"]:
            for dtype in ["specific", "unspecific"]:
                jsonl_file = run_output / f"{lang}_{dtype}.jsonl"
                if jsonl_file.exists():
                    zf.write(jsonl_file, f"{lang}_{dtype}.jsonl")
                    added.append(f"{lang}_{dtype}.jsonl")
                else:
                    missing.append(str(jsonl_file))

    # 3. Rapport
    print(f"\n📦 ZIP : {zip_path}")
    print(f"   ✅ {len(added)} fichiers ajoutés : {added}")

    if missing:
        print(f"\n   ⚠️  {len(missing)} fichiers MANQUANTS :")
        for f in missing:
            print(f"      ❌ {f}")
        print("   → Lance le run complet avant de soumettre !")
    else:
        print("\n   🎉 Prêt à soumettre !")

    return zip_path


# ─────────────────────────────────────────────
# Endpoint API (utilisé par routes.py)
# ─────────────────────────────────────────────

def get_submission_metadata(run_id: str, team: str, submission_id: str | None = None) -> dict:
    """
    Appelable directement depuis routes.py pour exposer
    le metadata via l'API FastAPI.

    Exemple dans routes.py :
        from src.export.challenge_export import get_submission_metadata

        @router.get("/runs/{run_id}/metadata")
        async def get_metadata(run_id: str, team: str):
            return get_submission_metadata(run_id, team)
    """
    return build_metadata(run_id, team, submission_id)


# ─────────────────────────────────────────────
# Point d'entrée CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Générer le ZIP de soumission ELOQUENT"
    )
    parser.add_argument("--run-id",        required=True,
                        help="ID du run (ex: run_20260501_123456_abc123)")
    parser.add_argument("--team",          required=True,
                        help="Nom de l'équipe")
    parser.add_argument("--submission-id", default=None,
                        help="ID de soumission custom (défaut = run_id)")
    parser.add_argument("--output-dir",    default="data/submission",
                        help="Dossier de sortie du ZIP")
    args = parser.parse_args()

    generate_submission(
        run_id        = args.run_id,
        team          = args.team,
        submission_id = args.submission_id,
        output_dir    = args.output_dir,
    )