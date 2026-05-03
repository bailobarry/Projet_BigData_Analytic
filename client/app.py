import json

import httpx
import streamlit as st

import utils

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cultural Robustness and Diversity",
    layout="wide",
)

# ── Data loading ──────────────────────────────────────────────────────────────

try:
    CONFIGS = utils.get_all_configs()
except httpx.ConnectError:
    st.error("L'API n'est pas accessible. Vérifiez que le serveur est bien lancé sur `http://127.0.0.1:8000`.")
    st.stop()

PROVIDERS = [p["label"] for p in CONFIGS["providers"]]
MODELS_BY_PROVIDER = {p["label"]: p["models"] for p in CONFIGS["providers"]}
LANGUAGES = CONFIGS["languages"]
DATASET_TYPES = CONFIGS["dataset_types"]
VARIATION = [s["name"] for s in CONFIGS["variations"]]
VARIATION_DETAILS = [s["description"] for s in CONFIGS["variations"]]

model_options = []
model_captions = []
for provider in PROVIDERS:
    for model in MODELS_BY_PROVIDER[provider]:
        model_options.append(model)
        model_captions.append(f"via {provider}")

# ── Helpers UI ────────────────────────────────────────────────────────────────

@st.fragment
def download_submission_zip_button(data_bytes):
    st.download_button(
        label="Télécharger submission.zip",
        data=data_bytes,
        file_name="submission.zip",
        mime="application/zip",
    )


def display_results(results: dict):
    st.subheader("Résultats")
    json_files  = sorted((f, c) for f, c in results.items() if f.endswith(".json"))
    jsonl_files = sorted((f, c) for f, c in results.items() if f.endswith(".jsonl"))

    for filename, content in json_files + jsonl_files:
        decoded = content.decode("utf-8")
        if filename.endswith(".json"):
            with st.expander(filename, expanded=False):
                st.json(json.loads(decoded))
        else:
            rows = [json.loads(line) for line in decoded.splitlines() if line.strip()]
            with st.expander(f"{filename} ({len(rows)} lignes)", expanded=False):
                st.table(rows)

    zip_bytes = utils.download_submission_zip(run_id)
    download_submission_zip_button(zip_bytes)

# ── Form ──────────────────────────────────────────────────────────────────────

with st.sidebar:

    st.title("Cultural Robustness and Diversity (ELOQUENT @ CLEF 2026)")

    st.header("Lancer une expérience : ")

    with st.form("experience"):

        description = st.text_input(label="Entrer une description rapide :")

        model = st.radio(
            label="Choisir un modèle :",
            options=model_options,
            captions=model_captions,
            horizontal=True,
        )
        provider = next(p for p in PROVIDERS if model in MODELS_BY_PROVIDER[p])

        languages = st.pills(
            label="Choisir les langues : ",
            options=LANGUAGES,
            default=LANGUAGES,
            selection_mode="multi",
        )

        dataset_types = st.pills(
            label="Choisir le type de dataset :",
            options=DATASET_TYPES,
            default=[DATASET_TYPES[0]],
            selection_mode="multi",
        )

        variation = st.radio(
            label="Choisir la stratégie :",
            options=VARIATION,
            captions=VARIATION_DETAILS,
            horizontal=True,
        )
        if variation == "none":
            variation = None

        col1, col2, col3 = st.columns(3)
        with col1:
            temperature = st.number_input(label="Temperature :", value=1.0, min_value=0.0, max_value=2.0, step=0.1)
        with col2:
            max_tokens = st.number_input(label="max_tokens :", value=256, min_value=1, max_value=8192, step=128)
        with col3:
            top_p = st.number_input(label="top_p :", value=1.0, min_value=0.0, max_value=1.0, step=0.05)

        submitted = st.form_submit_button(label="Lancer l'expérience")

# ── Exécution ─────────────────────────────────────────────────────────────────

results_placeholder = st.empty()
results_placeholder.info("Lancer une expérience pour que les résultats s'affichent ici.")

if submitted:
    results_placeholder.empty()
    with results_placeholder.container():

        if not languages or not dataset_types:
            st.error("Sélectionner au moins une langue et un type de dataset.")
            st.stop()

        try:
            run_id = utils.run_experience(
                provider=provider,
                model=model,
                languages=languages,
                dataset_types=dataset_types,
                variation=variation,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                description=description,
            )
        except Exception as e:
            st.error(f"Impossible de lancer le run : {e}")
            st.stop()

        final_status = "unknown"

        with st.spinner("Expérience lancée...", show_time=True):
            log_lines = []
            log_box = st.empty()
            status_box = st.empty()

            with httpx.Client(timeout=None) as client:
                with client.stream("GET", f"{utils.API_URL}/runs/{run_id}/stream") as r:
                    for line in r.iter_lines():
                        if not line.startswith("data:"):
                            continue

                        event = json.loads(line[5:].strip())
                        final_status = event.get("status", final_status)

                        for log_line in event.get("new_logs", []):
                            log_lines.append(log_line)
                            log_box.code("\n".join(log_lines), language=None)

                        if final_status == "completed":
                            s = event.get("summary", {})
                            total_prompts = s.get('total_prompts')
                            total_errors = s.get('total_errors')
                            duration_seconds = s.get('duration_seconds')
                            if total_errors == 0:
                                status_box.success(
                                    f"Terminé : {total_prompts} prompts, "
                                    f"{total_errors} erreurs, "
                                    f"{duration_seconds}s"
                                )
                            else:
                                status_box.error(
                                    f"Terminé : {total_prompts} prompts, "
                                    f"{total_errors} erreurs, "
                                    f"{duration_seconds}s"
                                )
                            break

                        elif final_status == "failed":
                            status_box.error(f"Échec : {event.get('error')}")
                            break

# ── Résultats ─────────────────────────────────────────────────────────────────

        if final_status == "completed":
            try:
                results = utils.get_run_results(run_id)
            except Exception as e:
                st.error(f"Impossible de charger les résultats : {e}")
                st.stop()

            display_results(results)
