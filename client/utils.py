import httpx

API_URL = "http://127.0.0.1:8000/api"


def get_all_configs() -> dict:
    """GET /providers — retourne providers, langues, strategies, etc."""
    response = httpx.get(f"{API_URL}/providers")
    response.raise_for_status()
    return response.json()


def list_runs() -> list[dict]:
    """GET /runs — retourne la liste de tous les runs avec statut."""
    response = httpx.get(f"{API_URL}/runs")
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


def resume_run(run_id: str) -> str:
    """POST /runs/{run_id}/resume — reprend un run interrompu."""
    response = httpx.post(f"{API_URL}/runs/{run_id}/resume")
    response.raise_for_status()
    return response.json()["run_id"]


def start_analysis(
        run_id: str,
        methods: list[str],
        sample_size: int = 10,
        run_specific_id: str | None = None,
) -> dict:
    """POST /runs/{run_id}/analyse — lance les analyses en arrière-plan."""
    payload = {
        "methods": methods,
        "sample_size": sample_size,
        "run_specific_id": run_specific_id,
    }
    response = httpx.post(f"{API_URL}/runs/{run_id}/analyse", json=payload)
    response.raise_for_status()
    return response.json()


def get_analysis_results(run_id: str) -> dict:
    """GET /runs/{run_id}/analyse/results — retourne les JSON d'analyse sauvegardés."""
    response = httpx.get(f"{API_URL}/runs/{run_id}/analyse/results")
    response.raise_for_status()
    return response.json()


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
    """GET /runs/{run_id}/results/submission.zip — retourne le zip."""
    r = httpx.get(f"{API_URL}/runs/{run_id}/results/submission.zip")
    r.raise_for_status()
    return r.content


def cancel_run(run_id: str) -> dict:
    """POST /runs/{run_id}/cancel — annule une expérience en cours."""
    response = httpx.post(f"{API_URL}/runs/{run_id}/cancel")
    response.raise_for_status()
    return response.json()


def cancel_analysis(run_id: str) -> dict:
    """POST /runs/{run_id}/analyse/cancel — annule une analyse en cours."""
    response = httpx.post(f"{API_URL}/runs/{run_id}/analyse/cancel")
    response.raise_for_status()
    return response.json()


def compare_runs(
        run_id_a: str,
        run_id_b: str,
        methods: list[str],
        sample_size: int = 10,
        dataset_type: str | None = None,
) -> dict:
    """POST /runs/compare — compare deux runs."""
    payload = {
        "run_id_a": run_id_a,
        "run_id_b": run_id_b,
        "methods": methods,
        "sample_size": sample_size,
    }
    if dataset_type is not None:
        payload["dataset_type"] = dataset_type
    response = httpx.post(f"{API_URL}/runs/compare", json=payload, timeout=300.0)
    response.raise_for_status()
    return response.json()

