import zipfile
from io import BytesIO

import httpx

API_URL = "http://127.0.0.1:8000/api"


def get_all_configs() -> dict:
    """GET /providers — retourne providers, langues, strategies, etc."""
    response = httpx.get(f"{API_URL}/providers")
    response.raise_for_status()
    return response.json()


def run_experience(
        provider: str,
        model: str,
        languages: list[str],
        dataset_types: list[str],
        variation: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        description: str = "",
) -> str:
    """POST /runs — lance un run et retourne le run_id."""
    payload = {
        "description": description,
        "provider": {"type": provider, "model": model},
        "generation": {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p, "seed": 42},
        "pipeline": {
            "languages": languages,
            "dataset_types": dataset_types,
            "system_prompt": variation,
        },
    }
    response = httpx.post(f"{API_URL}/runs", json=payload)
    response.raise_for_status()
    return response.json()["run_id"]


def get_run_results(run_id: str) -> dict[str, bytes]:
    """Récupère dynamiquement tous les fichiers de résultats du run via l'API."""
    # 1. Lister les fichiers disponibles
    response = httpx.get(f"{API_URL}/runs/{run_id}/files")
    response.raise_for_status()
    filenames = response.json()["files"]

    # 2. Récupérer chaque fichier
    results = {}
    for filename in filenames:
        r = httpx.get(f"{API_URL}/runs/{run_id}/results/{filename}")
        if r.status_code == 404:
            continue
        r.raise_for_status()
        results[filename] = r.content
    return results


def download_submission_zip(run_id: str) -> bytes:
    """GET /runs/{run_id}/results/submission.zip — retourne le zip brut."""
    r = httpx.get(f"{API_URL}/runs/{run_id}/results/submission.zip")
    r.raise_for_status()
    return r.content