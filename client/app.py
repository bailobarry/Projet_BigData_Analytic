"""
Interface Streamlit – ELOQUENT Cultural Robustness & Diversity
==============================================================

4 onglets :
  Nouvelle expérience  – lancer un run LLM
  Reprendre            – relancer un run interrompu depuis le point d'arrêt
  Analyser             – lancer les 3 méthodes d'analyse sur un run
  Historique           – consulter tous les runs passés
"""

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

PROVIDERS          = [p["label"] for p in CONFIGS["providers"]]
MODELS_BY_PROVIDER = {p["label"]: p["models"] for p in CONFIGS["providers"]}
LANGUAGES          = CONFIGS["languages"]
DATASET_TYPES      = CONFIGS["dataset_types"]
VARIATION          = [s["name"] for s in CONFIGS["variations"]]
VARIATION_DETAILS  = [s["description"] for s in CONFIGS["variations"]]

model_options, model_captions = [], []
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


def display_results(run_id: str, results: dict):
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


def display_analysis(results: dict):
    """Affiche les résultats des 3 méthodes d'analyse."""
    quant    = results.get("quantitative")
    div_data = results.get("diversity")
    rob_data = results.get("robustness")
    llm_div  = results.get("llm_judge_diversity")
    llm_rob  = results.get("llm_judge_robustness")

    # ── Quantitatif ──────────────────────────────────────────────────────
    if quant:
        st.subheader("Analyse quantitative")
        stats = quant.get("stats_by_file", {})
        g = stats.get("global", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total prompts",     g.get("total", "–"))
        c2.metric("Moy. mots/réponse", f"{g.get('avg_words', 0):.1f}")
        c3.metric("Taux d'erreurs",    f"{g.get('error_rate', 0):.1%}")
        c4.metric("Taux de vides",     f"{g.get('empty_rate', 0):.1%}")
        rows = [
            {
                "Fichier":    k,
                "Réponses":   s.get("total"),
                "Moy. mots":  s.get("avg_words"),
                "Moy. chars": s.get("avg_chars"),
                "Min mots":   s.get("min_words"),
                "Max mots":   s.get("max_words"),
                "Vides":      f"{s.get('empty_rate', 0):.1%}",
                "Erreurs":    f"{s.get('error_rate', 0):.1%}",
            }
            for k, s in stats.items() if k != "global"
        ]
        if rows:
            st.dataframe(rows, use_container_width=True)

    # ── Sémantique ────────────────────────────────────────────────────────
    if div_data or rob_data:
        st.subheader("Analyse sémantique (Embeddings)")
        c1, c2, c3 = st.columns(3)
        with c1:
            if div_data:
                st.metric("Score de Diversité", f"{div_data.get('score', 0):.4f}",
                          help="Proche de 1 = réponses culturellement très distinctes entre les langues")
                st.caption(f"std : {div_data.get('score_std', 0):.4f} · {div_data.get('n_prompts')} prompts · {', '.join(div_data.get('languages', []))}")
        with c2:
            if rob_data:
                st.metric("Score de Robustesse", f"{rob_data.get('score', 0):.4f}",
                          help="Proche de 1 = réponses stables malgré les contextes culturels")
                st.caption(f"std : {rob_data.get('score_std', 0):.4f} · {rob_data.get('n_prompts')} prompts · {', '.join(rob_data.get('languages', []))}")
            else:
                st.info("Pas de score de robustesse.\nSélectionnez un run avec des fichiers *specific*.")
        with c3:
            if div_data and rob_data:
                d, r = div_data.get("score", 0), rob_data.get("score", 0)
                harmonic = round(2 * d * r / (d + r), 4) if (d + r) > 0 else 0.0
                st.metric("Score Combiné (Harmonique)", f"{harmonic:.4f}",
                          help="Moyenne harmonique diversité × robustesse")
        if results.get("robustness_error"):
            st.info(f"Robustesse ignorée : {results['robustness_error']}")

    # ── LLM Judge ─────────────────────────────────────────────────────────
    if llm_div or llm_rob:
        st.subheader("LLM-as-a-Judge (Llama 3.3 70B) — scores sur 5")
        c1, c2, c3 = st.columns(3)
        with c1:
            if llm_div:
                avg = llm_div.get("avg_score", 0)
                st.metric("Diversité", f"{avg:.2f} / 5")
                dist = llm_div.get("score_distribution", {})
                st.caption("  ".join(f"★{k}:{v}" for k, v in dist.items()))
        with c2:
            if llm_rob:
                avg = llm_rob.get("avg_score", 0)
                st.metric("Robustesse", f"{avg:.2f} / 5")
                dist = llm_rob.get("score_distribution", {})
                st.caption("  ".join(f"★{k}:{v}" for k, v in dist.items()))
        with c3:
            if llm_div and llm_rob:
                g_avg = round((llm_div.get("avg_score", 0) + llm_rob.get("avg_score", 0)) / 2, 2)
                st.metric("Score global", f"{g_avg:.2f} / 5")
        if llm_div and llm_div.get("evaluations"):
            with st.expander("Détail évaluations Diversité", expanded=False):
                st.dataframe(llm_div["evaluations"], use_container_width=True)
        if llm_rob and llm_rob.get("evaluations"):
            with st.expander("Détail évaluations Robustesse", expanded=False):
                st.dataframe(llm_rob["evaluations"], use_container_width=True)


def show_analysis_form(preselected_run_id: str | None = None):
    """
    Affiche le formulaire d'analyse.
    Si preselected_run_id est fourni, ce run est présélectionné.
    Sinon, l'utilisateur choisit parmi tous les runs terminés.
    """
    try:
        all_runs_list  = utils.list_runs()
        completed_list = [r for r in all_runs_list if r.get("status") == "completed"]
    except Exception:
        completed_list = []

    if not completed_list:
        st.info("Aucun run terminé disponible. Lancez d'abord une expérience.")
        return

    # Sélection du run principal à analyser
    run_ids    = [r["run_id"] for r in completed_list]
    run_labels = [
        f"{r['run_id']}  —  {r.get('description') or '(sans description)'}  —  {r.get('prompts_done', 0)} prompts"
        for r in completed_list
    ]
    default_idx = run_ids.index(preselected_run_id) if preselected_run_id in run_ids else 0

    selected_main = st.selectbox(
        "Run à analyser :",
        options=run_ids,
        format_func=lambda rid: run_labels[run_ids.index(rid)],
        index=default_idx,
        key="analysis_main_run",
    )

    with st.form("form_analyse"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Méthodes d'analyse :**")
            do_quant    = st.checkbox("Analyse quantitative", value=True,  help="Statistiques de base — rapide")
            do_semantic = st.checkbox("Analyse sémantique",   value=True,  help="Embeddings — quelques dizaines de secondes")
            do_judge    = st.checkbox("LLM-as-a-Judge",       value=False, help="Groq/Llama — consomme des crédits API")
        with col_b:
            _help_sample = (
                "Nombre de questions analysées.\n\n"
                "• unspecific : max 101 questions/langue\n"
                "• specific   : jusqu'à 4140 prompts au total\n\n"
                "LLM Judge : chaque question = 1 appel API (~2s). "
                "À 4140, comptez ~2h30 et beaucoup de crédits Groq."
            )
            st.markdown("**Taille échantillon (sémantique & juge) :**", help=_help_sample)
            _num_col, _slider_col = st.columns([1, 2])
            with _num_col:
                _sample_input = st.number_input(
                    "Valeur",
                    min_value=3,
                    max_value=4140,
                    value=st.session_state.get("sample_size_val", 10),
                    step=1,
                    label_visibility="collapsed",
                    key="sample_size_num",
                )
            st.session_state["sample_size_val"] = _sample_input
            spec_options  = ["— Même run —"] + run_ids
            selected_spec = st.selectbox(
                "Run secondaire pour la robustesse (fichiers *specific*) :",
                spec_options,
                help="Sélectionnez un run avec des fichiers *_specific.jsonl si différent du run principal.",
            )
        submit_analyse = st.form_submit_button("Analyser", type="primary")

    if submit_analyse:
        methods = []
        if do_quant:    methods.append("quantitative")
        if do_semantic: methods.append("semantic")
        if do_judge:    methods.append("llm_judge")

        if not methods:
            st.warning("Sélectionnez au moins une méthode.")
            return

        spec_run = None if selected_spec == "— Même run —" else selected_spec
        try:
            utils.start_analysis(
                run_id=selected_main,
                methods=methods,
                sample_size=_sample_input,
                run_specific_id=spec_run,
            )
        except Exception as e:
            st.error(f"Impossible de lancer l'analyse : {e}")
            return

        # ── Étape 1 : stocker l'état et recharger pour afficher le bouton Arrêter ──
        st.session_state["stream_analysis_data"] = {
            "run_id":          selected_main,
            "methods":         methods,
            "sample_size":     _sample_input,
            "run_specific_id": spec_run,
        }
        st.rerun()


# ── Lecture de l'état de streaming AVANT le sidebar ──────────────────────────
# (pour que les boutons d'arrêt soient rendus dès le début du script)

stream_run_id       = st.session_state.get("stream_run_id")        # run en cours
stream_analysis_data = st.session_state.get("stream_analysis_data")  # analyse en cours

# ── Sidebar ───────────────────────────────────────────────────────────────────

resume_submitted = False
resume_run_id    = None

with st.sidebar:

    st.title("Cultural Robustness and Diversity (ELOQUENT @ CLEF 2026)")

    # ── Lancer une expérience ─────────────────────────────────────────────

    st.header("Lancer une expérience :")

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
            temperature = st.number_input(label="Temperature :", value=0.0, min_value=0.0, max_value=2.0, step=0.1)
        with col2:
            max_tokens = st.number_input(label="max_tokens :", value=256, min_value=1, max_value=8192, step=128)
        with col3:
            top_p = st.number_input(label="top_p :", value=1.0, min_value=0.0, max_value=1.0, step=0.05)

        submitted = st.form_submit_button(label="Lancer l'expérience")

    # ── Arrêt d'une expérience en cours (juste sous le formulaire) ────────

    if stream_run_id:
        st.warning(f"**Expérience en cours**\n\n`{stream_run_id}`")
        if st.button("Arrêter l'expérience", use_container_width=True, type="secondary",
                     key="btn_cancel_run"):
            try:
                utils.cancel_run(stream_run_id)
            except Exception:
                pass
            st.session_state.pop("stream_run_id", None)
            stream_run_id = None
            st.session_state["show_cancelled_run"] = True

    # ── Reprendre un run interrompu ───────────────────────────────────────

    st.markdown("---")
    st.header("Reprendre une expérience")

    try:
        all_runs    = utils.list_runs()
        interrupted = [r for r in all_runs if r.get("status") == "interrupted"]
    except Exception:
        interrupted = []

    if not interrupted:
        st.info("Aucun run interrompu.")
    else:
        resume_options = {
            f"{r['run_id'][:26]}… ({r.get('prompts_done', 0)} prompts traités)": r["run_id"]
            for r in interrupted
        }
        selected_label = st.selectbox("Choisir le run à reprendre :", list(resume_options.keys()))
        resume_run_id  = resume_options[selected_label]

        sel = next(r for r in interrupted if r["run_id"] == resume_run_id)
        st.caption(
            f"Modèle : {sel.get('model', '–')} · "
            f"Langues : {', '.join(sel.get('languages', []))} · "
            f"Date : {(sel.get('created_at') or '')[:16].replace('T', ' ')}"
        )
        if sel.get("description"):
            st.caption(f"Description : {sel['description']}")

        resume_submitted = st.button("Reprendre ce run", use_container_width=True, type="primary")


# ── Zone principale ───────────────────────────────────────────────────────────

results_placeholder = st.empty()
results_placeholder.info("Lancer une expérience pour que les résultats s'affichent ici.")

last_run_id = None  # run_id du dernier run terminé (pour pré-sélectionner l'analyse)

# ── Étape 1 – Nouveau run : lancement + rerun ─────────────────────────────────

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

        st.session_state["stream_run_id"] = run_id
        st.rerun()  # → recharge la page : le bouton "Arrêter" apparaît dans le sidebar

# ── Étape 2 – Streaming du run (après le rerun) ───────────────────────────────

elif stream_run_id:
    results_placeholder.empty()
    with results_placeholder.container():

        def _stream_run_logs(run_id: str) -> str:
            """Affiche les logs SSE en temps réel. Retourne le statut final."""
            log_lines    = []
            log_box      = st.empty()
            status_box   = st.empty()
            final_status = "unknown"

            with httpx.Client(timeout=None) as client:
                with client.stream("GET", f"{utils.API_URL}/runs/{run_id}/stream") as r:
                    for line in r.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        event        = json.loads(line[5:].strip())
                        final_status = event.get("status", final_status)

                        for log_line in event.get("new_logs", []):
                            log_lines.append(log_line)
                            log_box.code("\n".join(log_lines), language=None)

                        if final_status == "completed":
                            s = event.get("summary", {}) or {}
                            total   = s.get("total_prompts")
                            errors  = s.get("total_errors")
                            elapsed = s.get("duration_seconds")
                            if errors == 0:
                                status_box.success(f"Terminé : {total} prompts, {errors} erreurs, {elapsed}s")
                            else:
                                status_box.error(f"Terminé : {total} prompts, {errors} erreurs, {elapsed}s")
                            break
                        elif final_status == "failed":
                            status_box.error(f"Échec : {event.get('error')}")
                            break
                        elif final_status == "cancelled":
                            status_box.warning("Expérience arrêtée")
                            break

            return final_status

        with st.spinner("Expérience en cours...", show_time=True):
            final_status = _stream_run_logs(stream_run_id)

        st.session_state.pop("stream_run_id", None)

        if final_status == "completed":
            try:
                results = utils.get_run_results(stream_run_id)
            except Exception as e:
                st.error(f"Impossible de charger les résultats : {e}")
                st.stop()
            display_results(stream_run_id, results)
            last_run_id = stream_run_id

# ── Reprise d'un run interrompu : lancement + rerun ───────────────────────────

elif resume_submitted and resume_run_id:
    results_placeholder.empty()
    with results_placeholder.container():

        try:
            utils.resume_run(resume_run_id)
        except Exception as e:
            st.error(f"Impossible de reprendre le run : {e}")
            st.stop()

        st.session_state["stream_run_id"] = resume_run_id
        st.rerun()

# ── Message d'annulation d'expérience ────────────────────────────────────────

elif st.session_state.pop("show_cancelled_run", False):
    results_placeholder.empty()
    with results_placeholder.container():
        st.warning("Expérience arrêtée")

# ── Section Analyse – toujours visible ───────────────────────────────────────

st.markdown("---")
st.header("Analyser les résultats")

# ── 1. Formulaire de configuration (toujours affiché en PREMIER) ─────────────

show_analysis_form(preselected_run_id=last_run_id)

# ── 2. Progression + bouton Arrêter + résultats (EN DESSOUS de la config) ────

if stream_analysis_data:
    _analysis_run_id = stream_analysis_data.get("run_id", "")

    st.markdown("---")
    st.subheader(f"Progression de l'analyse — `{_analysis_run_id}`")

    # Bouton rendu AVANT la boucle bloquante → reste cliquable pendant le streaming
    if st.button("Arrêter l'analyse", type="secondary", key="btn_stop_analysis_inline"):
        try:
            utils.cancel_analysis(_analysis_run_id)
        except Exception:
            pass
        st.session_state.pop("stream_analysis_data", None)
        st.session_state["show_cancelled_analysis"] = True
        st.rerun()

    _steps_labels = {
        "quantitative":        "Analyse quantitative",
        "semantic_diversity":  "Calcul embeddings — diversité",
        "semantic_robustness": "Calcul embeddings — robustesse",
        "llm_judge_diversity": "LLM Judge — diversité",
        "llm_judge_robustness":"LLM Judge — robustesse",
    }
    progress_box  = st.empty()
    final_results = None

    with httpx.Client(timeout=None) as client:
        with client.stream("GET", f"{utils.API_URL}/runs/{_analysis_run_id}/analyse/stream") as r:
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                event      = json.loads(line[5:].strip())
                status     = event.get("status", "")
                steps_done = event.get("steps_done", [])
                current    = event.get("current", "")

                lines_md = "\n".join(f"- {_steps_labels.get(s, s)}" for s in steps_done)
                if current and status == "running":
                    lines_md += f"\n- {_steps_labels.get(current, current)} *(en cours…)*"
                progress_box.markdown(lines_md or "*Démarrage…*")

                if status == "completed":
                    final_results = event.get("results", {})
                    progress_box.success("Analyse terminée.")
                    break
                elif status == "cancelled":
                    progress_box.warning("Analyse arrêtée")
                    break
                elif status == "failed":
                    progress_box.error(f"Échec : {event.get('error')}")
                    break

    st.session_state.pop("stream_analysis_data", None)

    # Résultats affichés EN DESSOUS de la progression
    if final_results:
        st.markdown("---")
        st.subheader("Résultats de l'analyse")
        display_analysis(final_results)

# Message d'annulation (affiché sous le formulaire)
if st.session_state.pop("show_cancelled_analysis", False):
    st.warning("Analyse arrêtée")
