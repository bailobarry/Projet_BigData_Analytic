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
import plotly.graph_objects as go
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


def _auto_df(data, max_height: int = 600, min_row_height: int = 36, **kwargs):
    """
    Affiche un tableau Streamlit dont la hauteur s'adapte automatiquement
    au nombre de lignes, et dont les colonnes de texte long se réajustent.

    Parameters
    ----------
    data : list[dict] | pd.DataFrame
        Données à afficher.
    max_height : int
        Hauteur maximale en pixels (défaut 600).
    min_row_height : int
        Hauteur estimée par ligne en pixels (défaut 36).
    """
    import pandas as pd

    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data

    # Hauteur auto : header (38px) + n lignes * hauteur_ligne, plafonné à max_height
    n_rows = len(df)
    height = min(38 + n_rows * min_row_height, max_height)

    # Colonnes de texte long → largeur "large" pour éviter le découpage
    _wide_cols = {"answer", "reason", "Réponse", "Raison", "Fichier",
                  "strongest_contrast", "weakest_contrast",
                  "best_response_lang", "worst_response_lang"}
    col_config = {}
    for col in df.columns:
        if col in _wide_cols or df[col].dtype == object and df[col].str.len().mean() > 40:
            col_config[col] = st.column_config.TextColumn(col, width="large")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=height,
        column_config=col_config if col_config else None,
        **kwargs,
    )


def display_analysis(results: dict):
    """Affiche les résultats des 3 méthodes d'analyse."""
    quant    = results.get("quantitative")
    div_data = results.get("diversity")
    rob_data = results.get("robustness")
    qual     = results.get("qualitative")
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
            _auto_df(rows)

    # ── Sémantique ────────────────────────────────────────────────────────
    if div_data or rob_data:
        st.subheader("Analyse sémantique (Embeddings)")
        c1, c2, c3 = st.columns(3)
        with c1:
            if div_data:
                st.metric("Score de Diversité", f"{div_data.get('score', 0):.4f}",
                          help="Proche de 1 = réponses culturellement très distinctes entre les langues")
                st.caption(
                    f"std : {div_data.get('score_std', 0):.4f} · "
                    f"min : {div_data.get('score_min', 0):.4f} · "
                    f"max : {div_data.get('score_max', 0):.4f} · "
                    f"médiane : {div_data.get('score_median', 0):.4f}"
                )
                st.caption(f"{div_data.get('n_prompts')} prompts · {', '.join(div_data.get('languages', []))}")
        with c2:
            if rob_data:
                st.metric("Score de Robustesse", f"{rob_data.get('score', 0):.4f}",
                          help="Proche de 1 = réponses stables malgré les contextes culturels")
                st.caption(
                    f"std : {rob_data.get('score_std', 0):.4f} · "
                    f"min : {rob_data.get('score_min', 0):.4f} · "
                    f"max : {rob_data.get('score_max', 0):.4f} · "
                    f"médiane : {rob_data.get('score_median', 0):.4f}"
                )
                st.caption(f"{rob_data.get('n_prompts')} prompts · {', '.join(rob_data.get('languages', []))}")
            else:
                st.info("Pas de score de robustesse.\nSélectionnez un run avec des fichiers *specific*.")
        with c3:
            if div_data and rob_data:
                d, r = div_data.get("score", 0), rob_data.get("score", 0)
                harmonic = round(2 * d * r / (d + r), 4) if (d + r) > 0 else 0.0
                product  = round(d * r, 4)
                st.metric("Score Combiné (D × R)", f"{product:.4f}",
                          help="Méthode officielle du challenge : diversité × robustesse")
                st.caption(f"Harmonique : {harmonic:.4f}")
        if results.get("robustness_error"):
            st.info(f"Robustesse ignorée : {results['robustness_error']}")

        # Diversité par paire de langues
        pair_div = div_data.get("per_language_pair_diversity") if div_data else None
        pair_rob = rob_data.get("per_language_pair_robustness") if rob_data else None
        if pair_div or pair_rob:
            with st.expander("Détail par paire de langues", expanded=False):
                pair_rows = []
                all_pairs = sorted(set(list(pair_div or {}) + list(pair_rob or {})))
                for pair in all_pairs:
                    row = {"Paire": pair}
                    if pair_div:
                        row["Diversité"] = f"{pair_div.get(pair, 0):.4f}"
                    if pair_rob:
                        row["Robustesse"] = f"{pair_rob.get(pair, 0):.4f}"
                    pair_rows.append(row)
                if pair_rows:
                    _auto_df(pair_rows)

    # ── Analyse qualitative ───────────────────────────────────────────────
    if qual:
        st.subheader("Analyse qualitative")

        # Conformité à la consigne
        violations = qual.get("instruction_violations", {})
        typology   = qual.get("error_typology", {})

        c1, c2, c3 = st.columns(3)
        with c1:
            rate = violations.get("violation_rate", 0)
            st.metric("Taux de non-conformité", f"{rate:.1%}",
                      help="Réponses ne respectant pas 'répondez en une seule phrase'")
        with c2:
            if typology:
                rates = typology.get("rates", {})
                st.metric("Généricité (< 5 mots)", f"{rates.get('generic', 0):.1%}",
                          help="Réponses trop courtes / trop vagues")
        with c3:
            if typology:
                rates = typology.get("rates", {})
                st.metric("Taux OK", f"{rates.get('ok', 0):.1%}")

        # Distribution des étiquettes d'erreurs
        if typology and typology.get("distribution"):
            with st.expander("Distribution des types de réponses", expanded=False):
                dist = typology["distribution"]
                total = typology.get("total", 1)
                rows = [
                    {"Type": k, "Nombre": v, "Taux": f"{v/total:.1%}"}
                    for k, v in dist.items()
                ]
                _auto_df(rows)

        # Cas extrêmes par catégorie (diversité)
        div_cat = qual.get("diversity_by_category")
        rob_cat = qual.get("robustness_by_category")
        if div_cat or rob_cat:
            with st.expander("Scores par catégorie thématique", expanded=False):
                cat_rows = []
                categories = sorted(set(list(div_cat or {}) + list(rob_cat or {})))
                for cat in categories:
                    row = {"Catégorie": cat}
                    if div_cat and cat in div_cat:
                        row["Diversité moy."] = f"{div_cat[cat]['avg']:.4f}"
                        row["n"] = div_cat[cat]["n"]
                    if rob_cat and cat in rob_cat:
                        row["Robustesse moy."] = f"{rob_cat[cat]['avg']:.4f}"
                    cat_rows.append(row)
                if cat_rows:
                    _auto_df(cat_rows)

        # Exemples problématiques
        if typology and typology.get("problematic_examples"):
            with st.expander("Exemples problématiques (non-conformité, généricité)", expanded=False):
                _auto_df(typology["problematic_examples"], min_row_height=64)

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
                _auto_df(llm_div["evaluations"], min_row_height=52)
        if llm_rob and llm_rob.get("evaluations"):
            with st.expander("Détail évaluations Robustesse", expanded=False):
                _auto_df(llm_rob["evaluations"], min_row_height=52)


def display_charts(results: dict):
    """
    Génère les graphiques interactifs (Plotly) adaptés aux résultats d'analyse.

    Un onglet par méthode d'analyse présente les visuels les plus appropriés :
    - Quantitative  : longueurs et taux d'erreurs/vides par fichier
    - Semantique    : scores de diversité et robustesse + radar par catégorie
    - LLM Judge     : distributions de scores (1-5) et comparaison
    - Qualitative   : typologie d'erreurs et conformité à la consigne
    """
    quant   = results.get("quantitative")
    div     = results.get("diversity")
    rob     = results.get("robustness")
    llm_div = results.get("llm_judge_diversity")
    llm_rob = results.get("llm_judge_robustness")
    qual    = results.get("qualitative")

    # Déterminer quels onglets afficher
    tab_labels = []
    if quant:
        tab_labels.append("Quantitative")
    if div or rob:
        tab_labels.append("Semantique")
    if llm_div or llm_rob:
        tab_labels.append("LLM Judge")
    if qual:
        tab_labels.append("Qualitative")

    if not tab_labels:
        st.info("Aucun résultat disponible pour générer des graphiques.")
        return

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    # ── Onglet Quantitative ───────────────────────────────────────────────
    if quant and tab_idx < len(tabs):
        with tabs[tab_idx]:
            stats = quant.get("stats_by_file", {})
            files = [k for k in stats if k != "global"]

            if files:
                # Graphique 1 : longueur moyenne par fichier
                avg_words = [stats[f]["avg_words"] for f in files]
                avg_chars = [stats[f]["avg_chars"] for f in files]

                fig_len = go.Figure()
                fig_len.add_trace(go.Bar(
                    name="Moy. mots",
                    x=files, y=avg_words,
                    marker_color="#4C9BE8",
                    text=[f"{v:.1f}" for v in avg_words],
                    textposition="outside",
                ))
                fig_len.add_trace(go.Bar(
                    name="Moy. chars / 10",
                    x=files, y=[v / 10 for v in avg_chars],
                    marker_color="#A8D8EA",
                    text=[f"{v:.0f}" for v in avg_chars],
                    textposition="outside",
                ))
                fig_len.update_layout(
                    title="Longueur moyenne des réponses par fichier",
                    barmode="group",
                    xaxis_title="Fichier (langue_type)",
                    yaxis_title="Nombre de mots (chars ÷ 10)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    height=400,
                )
                st.plotly_chart(fig_len, use_container_width=True)

                # Graphique 2 : taux d'erreurs et de vides (barres empilées %)
                error_rates = [stats[f]["error_rate"] * 100 for f in files]
                empty_rates = [stats[f]["empty_rate"] * 100 for f in files]
                ok_rates    = [max(0, 100 - e - v) for e, v in zip(error_rates, empty_rates)]

                fig_qual = go.Figure()
                fig_qual.add_trace(go.Bar(name="OK", x=files, y=ok_rates,
                                          marker_color="#52b788"))
                fig_qual.add_trace(go.Bar(name="Vide", x=files, y=empty_rates,
                                          marker_color="#f4a261"))
                fig_qual.add_trace(go.Bar(name="Erreur", x=files, y=error_rates,
                                          marker_color="#e63946"))
                fig_qual.update_layout(
                    title="Qualité des réponses par fichier (%)",
                    barmode="stack",
                    xaxis_title="Fichier (langue_type)",
                    yaxis_title="Pourcentage (%)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    yaxis=dict(range=[0, 102]),
                    height=400,
                )
                st.plotly_chart(fig_qual, use_container_width=True)

        tab_idx += 1

    # ── Onglet Sémantique ─────────────────────────────────────────────────
    if (div or rob) and tab_idx < len(tabs):
        with tabs[tab_idx]:
            d_score = div.get("score", 0) if div else 0
            d_std   = div.get("score_std", 0) if div else 0
            r_score = rob.get("score", 0) if rob else 0
            r_std   = rob.get("score_std", 0) if rob else 0
            product = round(d_score * r_score, 4)
            harmonic = round(2 * d_score * r_score / (d_score + r_score), 4) \
                       if (d_score + r_score) > 0 else 0

            # Graphique 1 : barres diversité / robustesse / combiné avec écart-type
            labels  = ["Diversité", "Robustesse", "Combiné (D×R)"]
            scores  = [d_score, r_score, product]
            errors  = [d_std, r_std, 0]
            colors  = ["#457b9d", "#e76f51", "#2a9d8f"]

            fig_sem = go.Figure()
            fig_sem.add_trace(go.Bar(
                x=labels, y=scores,
                error_y=dict(type="data", array=errors, visible=True),
                marker_color=colors,
                text=[f"{v:.4f}" for v in scores],
                textposition="outside",
                name="Score",
            ))
            fig_sem.update_layout(
                title="Scores sémantiques (embeddings multilingues)",
                yaxis=dict(title="Score", range=[0, 1.05]),
                xaxis_title="Métrique",
                height=400,
                showlegend=False,
            )
            # Ligne de référence harmonic
            fig_sem.add_hline(y=harmonic, line_dash="dot", line_color="gray",
                              annotation_text=f"Harmonique = {harmonic:.4f}",
                              annotation_position="right")
            st.plotly_chart(fig_sem, use_container_width=True)

            # Graphique 2 : jauge du score combiné
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=product,
                number={"valueformat": ".4f"},
                title={"text": "Score Combiné (D × R) — méthode officielle challenge"},
                delta={"reference": 0.20, "valueformat": ".4f"},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": "#2a9d8f"},
                    "steps": [
                        {"range": [0, 0.09],  "color": "#e63946"},
                        {"range": [0.09, 0.20], "color": "#f4a261"},
                        {"range": [0.20, 0.36], "color": "#a8dadc"},
                        {"range": [0.36, 1],   "color": "#52b788"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": product,
                    },
                },
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.caption("Zones : rouge < 0.09 | orange 0.09–0.20 | bleu 0.20–0.36 | vert > 0.36")

            # Graphique 3 : radar par catégorie (si qualitative disponible)
            if qual:
                div_cat = qual.get("diversity_by_category", {})
                rob_cat = qual.get("robustness_by_category", {})
                if div_cat and rob_cat:
                    categories = [c for c in div_cat if c in rob_cat]
                    if len(categories) >= 3:
                        theta = categories + [categories[0]]  # fermer le polygone
                        div_vals = [div_cat[c]["avg"] for c in categories] + \
                                   [div_cat[categories[0]]["avg"]]
                        rob_vals = [rob_cat[c]["avg"] for c in categories] + \
                                   [rob_cat[categories[0]]["avg"]]

                        fig_radar = go.Figure()
                        fig_radar.add_trace(go.Scatterpolar(
                            r=div_vals, theta=theta,
                            fill="toself", name="Diversité",
                            line_color="#457b9d",
                        ))
                        fig_radar.add_trace(go.Scatterpolar(
                            r=rob_vals, theta=theta,
                            fill="toself", name="Robustesse",
                            line_color="#e76f51",
                        ))
                        fig_radar.update_layout(
                            title="Scores par catégorie thématique (radar)",
                            polar=dict(radialaxis=dict(range=[0, 1], visible=True)),
                            legend=dict(orientation="h"),
                            height=450,
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

            # Graphique 4 : Diversité et robustesse par paire de langues
            pair_div_g = div.get("per_language_pair_diversity") if div else None
            pair_rob_g = rob.get("per_language_pair_robustness") if rob else None
            if pair_div_g or pair_rob_g:
                all_pairs_g = sorted(set(list(pair_div_g or {}) + list(pair_rob_g or {})))
                fig_pairs = go.Figure()
                if pair_div_g:
                    fig_pairs.add_trace(go.Bar(
                        name="Diversité (1−sim)",
                        x=all_pairs_g,
                        y=[pair_div_g.get(p, 0) for p in all_pairs_g],
                        marker_color="#457b9d",
                    ))
                if pair_rob_g:
                    fig_pairs.add_trace(go.Bar(
                        name="Robustesse (sim)",
                        x=all_pairs_g,
                        y=[pair_rob_g.get(p, 0) for p in all_pairs_g],
                        marker_color="#e76f51",
                    ))
                fig_pairs.update_layout(
                    title="Diversité et robustesse par paire de langues",
                    barmode="group",
                    yaxis=dict(title="Score", range=[0, 1]),
                    xaxis=dict(title="Paire de langues"),
                    height=400,
                    legend=dict(orientation="h"),
                )
                st.plotly_chart(fig_pairs, use_container_width=True)

        tab_idx += 1

    # ── Onglet LLM Judge ──────────────────────────────────────────────────
    if (llm_div or llm_rob) and tab_idx < len(tabs):
        with tabs[tab_idx]:
            # Graphique 1 : scores moyens diversité vs robustesse
            judge_labels, judge_scores, judge_colors = [], [], []
            if llm_div:
                judge_labels.append("Diversité")
                judge_scores.append(llm_div.get("avg_score", 0))
                judge_colors.append("#457b9d")
            if llm_rob:
                judge_labels.append("Robustesse")
                judge_scores.append(llm_rob.get("avg_score", 0))
                judge_colors.append("#e76f51")
            if llm_div and llm_rob:
                g = round((llm_div["avg_score"] + llm_rob["avg_score"]) / 2, 2)
                judge_labels.append("Moyenne globale")
                judge_scores.append(g)
                judge_colors.append("#2a9d8f")

            fig_judge = go.Figure(go.Bar(
                x=judge_labels, y=judge_scores,
                marker_color=judge_colors,
                text=[f"{v:.2f}/5" for v in judge_scores],
                textposition="outside",
            ))
            fig_judge.update_layout(
                title="Scores LLM-as-a-Judge (sur 5)",
                yaxis=dict(title="Score moyen", range=[0, 5.5]),
                height=380,
                showlegend=False,
            )
            # Ligne de référence à 3/5
            fig_judge.add_hline(y=3, line_dash="dash", line_color="gray",
                                annotation_text="Seuil satisfaisant (3/5)",
                                annotation_position="right")
            st.plotly_chart(fig_judge, use_container_width=True)

            # Graphiques 2-3 : distributions des scores 1-5 (pie charts)
            col1, col2 = st.columns(2)
            if llm_div and llm_div.get("score_distribution"):
                dist = llm_div["score_distribution"]
                labels_d = [f"★{k}" for k in dist]
                values_d = list(dist.values())
                fig_pie_d = go.Figure(go.Pie(
                    labels=labels_d, values=values_d,
                    hole=0.4,
                    marker_colors=["#e63946", "#f4a261", "#e9c46a", "#a8dadc", "#52b788"],
                    textinfo="label+percent",
                ))
                fig_pie_d.update_layout(
                    title="Distribution scores — Diversité",
                    showlegend=False,
                    height=320,
                )
                col1.plotly_chart(fig_pie_d, use_container_width=True)

            if llm_rob and llm_rob.get("score_distribution"):
                dist = llm_rob["score_distribution"]
                labels_r = [f"★{k}" for k in dist]
                values_r = list(dist.values())
                fig_pie_r = go.Figure(go.Pie(
                    labels=labels_r, values=values_r,
                    hole=0.4,
                    marker_colors=["#e63946", "#f4a261", "#e9c46a", "#a8dadc", "#52b788"],
                    textinfo="label+percent",
                ))
                fig_pie_r.update_layout(
                    title="Distribution scores — Robustesse",
                    showlegend=False,
                    height=320,
                )
                col2.plotly_chart(fig_pie_r, use_container_width=True)

        tab_idx += 1

    # ── Onglet Qualitative ────────────────────────────────────────────────
    if qual and tab_idx < len(tabs):
        with tabs[tab_idx]:
            typology = qual.get("error_typology", {})
            violations = qual.get("instruction_violations", {})

            # Graphique 1 : distribution de la typologie d'erreurs (donut)
            if typology and typology.get("distribution"):
                dist = typology["distribution"]
                color_map = {
                    "ok":            "#52b788",
                    "generic":       "#f4a261",
                    "non_compliant": "#e9c46a",
                    "error":         "#e63946",
                    "empty":         "#adb5bd",
                }
                labels_t = list(dist.keys())
                values_t = list(dist.values())
                colors_t = [color_map.get(k, "#888") for k in labels_t]

                fig_typo = go.Figure(go.Pie(
                    labels=labels_t,
                    values=values_t,
                    hole=0.45,
                    marker_colors=colors_t,
                    textinfo="label+percent+value",
                ))
                fig_typo.update_layout(
                    title="Répartition des types de réponses (typologie d'erreurs)",
                    height=380,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                )
                st.plotly_chart(fig_typo, use_container_width=True)

            # Graphique 2 : taux de non-conformité par fichier
            by_file = violations.get("by_file", {})
            if by_file:
                files_q = list(by_file.keys())
                nc_vals = [by_file[f].get("non_compliant", 0) for f in files_q]
                gen_vals = [by_file[f].get("generic", 0) for f in files_q]
                err_vals = [by_file[f].get("error", 0) for f in files_q]
                tot_vals = [by_file[f].get("total", 1) for f in files_q]

                # Normaliser en pourcentage
                nc_pct  = [round(n / t * 100, 1) if t else 0 for n, t in zip(nc_vals, tot_vals)]
                gen_pct = [round(g / t * 100, 1) if t else 0 for g, t in zip(gen_vals, tot_vals)]
                err_pct = [round(e / t * 100, 1) if t else 0 for e, t in zip(err_vals, tot_vals)]

                fig_viol = go.Figure()
                fig_viol.add_trace(go.Bar(name="Non-conforme (>2 phrases)",
                                          x=files_q, y=nc_pct,
                                          marker_color="#e9c46a"))
                fig_viol.add_trace(go.Bar(name="Générique (<5 mots)",
                                          x=files_q, y=gen_pct,
                                          marker_color="#f4a261"))
                fig_viol.add_trace(go.Bar(name="Erreur pipeline",
                                          x=files_q, y=err_pct,
                                          marker_color="#e63946"))
                fig_viol.update_layout(
                    title="Taux de non-conformité par fichier (%)",
                    barmode="stack",
                    xaxis_title="Fichier (langue_type)",
                    yaxis_title="% des réponses",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    height=420,
                )
                st.plotly_chart(fig_viol, use_container_width=True)

            # Graphique 3 : scores par catégorie (horizontal bar chart)
            div_cat = qual.get("diversity_by_category", {})
            rob_cat = qual.get("robustness_by_category", {})
            if div_cat:
                categories = sorted(div_cat.keys())
                div_avgs = [div_cat[c]["avg"] for c in categories]
                rob_avgs = [rob_cat[c]["avg"] if (rob_cat and c in rob_cat) else 0
                            for c in categories]

                fig_cat = go.Figure()
                fig_cat.add_trace(go.Bar(
                    y=categories, x=div_avgs,
                    orientation="h", name="Diversité",
                    marker_color="#457b9d",
                    text=[f"{v:.4f}" for v in div_avgs],
                    textposition="outside",
                    error_x=dict(
                        type="data",
                        array=[div_cat[c].get("std", 0) for c in categories],
                        visible=True,
                    ),
                ))
                if any(rob_avgs):
                    fig_cat.add_trace(go.Bar(
                        y=categories, x=rob_avgs,
                        orientation="h", name="Robustesse",
                        marker_color="#e76f51",
                        text=[f"{v:.4f}" for v in rob_avgs],
                        textposition="outside",
                    ))
                fig_cat.update_layout(
                    title="Scores moyens par catégorie thématique",
                    barmode="group",
                    xaxis=dict(title="Score", range=[0, 1]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    height=380,
                )
                st.plotly_chart(fig_cat, use_container_width=True)

        tab_idx += 1


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
            do_qual     = st.checkbox("Analyse qualitative",  value=True,  help="Cas extrêmes, typologie d'erreurs, scores par catégorie")
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
        if do_qual:     methods.append("qualitative")
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
        "quantitative":              "Analyse quantitative",
        "semantic_diversity":        "Calcul embeddings — diversité",
        "semantic_robustness":       "Calcul embeddings — robustesse",
        "qualitative":               "Analyse qualitatives",
        "llm_judge_diversity":       "LLM Judge — diversité",
        "llm_judge_robustness":      "LLM Judge — robustesse",
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

    # Sauvegarder les résultats dans la session pour affichage persistant
    if final_results:
        st.session_state["last_analysis_results"] = final_results
        st.session_state["show_charts"] = False  # revenir aux résultats textuels par défaut

# Message d'annulation (affiché sous le formulaire)
if st.session_state.pop("show_cancelled_analysis", False):
    st.warning("Analyse arrêtée par l'utilisateur.")

# ── Résultats + bouton "Générer / Masquer les graphiques" ────────────────────

_last_results = st.session_state.get("last_analysis_results")
if _last_results:
    st.markdown("---")
    st.subheader("Résultats de l'analyse")

    # ── Boutons toujours visibles ─────────────────────────────────────────
    col_btn, col_clear = st.columns([2, 1])
    with col_btn:
        _show_charts = st.session_state.get("show_charts", False)
        _charts_label = "Masquer les graphiques" if _show_charts else "Générer les graphiques"
        if st.button(_charts_label, type="primary", key="btn_show_charts",
                     use_container_width=True):
            st.session_state["show_charts"] = not _show_charts
            st.rerun()
    with col_clear:
        if st.button("Effacer les résultats", type="secondary", key="btn_clear_results",
                     use_container_width=True):
            st.session_state.pop("last_analysis_results", None)
            st.session_state.pop("show_charts", None)
            st.rerun()

    st.markdown("")  # espace visuel

    # ── Affichage : graphiques OU résultats textuels (pas les deux) ───────
    if st.session_state.get("show_charts", False):
        display_charts(_last_results)
    else:
        display_analysis(_last_results)


# ── Section Comparaison ───────────────────────────────────────────────────────

st.markdown("---")
st.header("Comparer deux runs")
st.caption(
    "Sélectionnez deux runs pour comparer leurs statistiques quantitatives "
    "(longueur, taux d'erreurs) et/ou leurs scores sémantiques (diversité, robustesse, combiné)."
)

try:
    _all_runs_cmp = utils.list_runs()
    _done_runs_cmp = [r for r in _all_runs_cmp if r.get("status") == "completed"]
except Exception:
    _done_runs_cmp = []

if len(_done_runs_cmp) < 2:
    st.info("Il faut au moins 2 runs terminés pour effectuer une comparaison.")
else:
    _run_ids_cmp = [r["run_id"] for r in _done_runs_cmp]

    def _run_label(rid: str) -> str:
        r = next((x for x in _done_runs_cmp if x["run_id"] == rid), {})
        desc = r.get("description") or "(sans description)"
        model = r.get("model") or "?"
        variant = r.get("summary", {}) or {}
        return f"{rid[:26]}… | {model} | {desc}"

    col_run_a, col_run_b = st.columns(2)
    with col_run_a:
        _cmp_run_a = st.selectbox(
            "Run A (référence / baseline) :",
            _run_ids_cmp,
            format_func=_run_label,
            key="cmp_run_a",
        )
    with col_run_b:
        _cmp_run_b = st.selectbox(
            "Run B (variante à comparer) :",
            _run_ids_cmp,
            format_func=_run_label,
            index=min(1, len(_run_ids_cmp) - 1),
            key="cmp_run_b",
        )

    with st.form("form_compare"):
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            _cmp_do_quant = st.checkbox("Quantitatif", value=True,
                                        help="Longueur, taux d'erreurs — rapide")
        with col_m2:
            _cmp_do_sem = st.checkbox("Sémantique", value=True,
                                      help="Diversité & robustesse par embeddings")
        with col_m3:
            _cmp_sample = st.number_input(
                "Échantillon sémantique :", min_value=3, max_value=4140,
                value=10, step=1,
                help="Nombre de prompts pour le calcul sémantique",
                key="cmp_sample",
            )
        _cmp_submit = st.form_submit_button("Comparer", type="primary")

    if _cmp_submit:
        if _cmp_run_a == _cmp_run_b:
            st.warning("Choisissez deux runs différents.")
        else:
            _cmp_methods = []
            if _cmp_do_quant:
                _cmp_methods.append("quantitative")
            if _cmp_do_sem:
                _cmp_methods.append("semantic")

            if not _cmp_methods:
                st.warning("Sélectionnez au moins une méthode de comparaison.")
            else:
                with st.spinner("Comparaison en cours…", show_time=True):
                    try:
                        _cmp_result = utils.compare_runs(
                            _cmp_run_a, _cmp_run_b,
                            methods=_cmp_methods,
                            sample_size=int(_cmp_sample),
                        )
                        st.session_state["last_compare_result"] = _cmp_result
                    except Exception as _e:
                        st.error(f"Erreur lors de la comparaison : {_e}")

# Affichage des résultats de comparaison (persistant)
_cmp_res = st.session_state.get("last_compare_result")
if _cmp_res:
    _cmp_a = _cmp_res.get("run_a", "A")
    _cmp_b = _cmp_res.get("run_b", "B")

    col_clr, _ = st.columns([1, 3])
    with col_clr:
        if st.button("Effacer la comparaison", key="btn_clear_cmp", type="secondary"):
            st.session_state.pop("last_compare_result", None)
            st.rerun()

    st.subheader(f"Résultats : `{_cmp_a[:20]}…` vs `{_cmp_b[:20]}…`")

    # ── Comparaison Quantitative ──────────────────────────────────────────
    _cmp_quant = _cmp_res.get("quantitative")
    if _cmp_quant and not _cmp_res.get("quantitative_error"):
        st.markdown("#### Comparaison quantitative")
        _files_cmp = _cmp_quant.get("files", {})
        if _files_cmp:
            _rows_cmp = []
            for _fk, _fv in sorted(_files_cmp.items()):
                _delta_w = _fv.get("delta_avg_words", 0)
                _delta_e = _fv.get("delta_error_rate", 0)
                _rows_cmp.append({
                    "Fichier":             _fk,
                    "Moy. mots A":         _fv.get("avg_words_a"),
                    "Moy. mots B":         _fv.get("avg_words_b"),
                    "delta mots":              f"{'+'if _delta_w>=0 else ''}{_delta_w:.1f}",
                    "Taux erreurs A":      f"{_fv.get('error_rate_a', 0):.1%}",
                    "Taux erreurs B":      f"{_fv.get('error_rate_b', 0):.1%}",
                    "delta erreurs":           f"{'+'if _delta_e>=0 else ''}{_delta_e:.2%}",
                    "IDs communs":         _fv.get("common_ids"),
                })
            _auto_df(_rows_cmp)

            # Graphique : delta mots par fichier
            import plotly.graph_objects as go
            _fnames = [r["Fichier"] for r in _rows_cmp]
            _deltas_w = [_files_cmp[f]["delta_avg_words"] for f in _fnames]
            _colors_delta = ["#52b788" if d >= 0 else "#e63946" for d in _deltas_w]
            _fig_delta = go.Figure(go.Bar(
                x=_fnames, y=_deltas_w,
                marker_color=_colors_delta,
                text=[f"{'+'if d>=0 else ''}{d:.1f}" for d in _deltas_w],
                textposition="outside",
            ))
            _fig_delta.add_hline(y=0, line_color="gray", line_dash="dot")
            _fig_delta.update_layout(
                title="delta Longueur moyenne (B − A) par fichier",
                yaxis_title="delta mots",
                xaxis_title="Fichier (langue_type)",
                height=380,
                showlegend=False,
            )
            st.plotly_chart(_fig_delta, use_container_width=True)

    elif _cmp_res.get("quantitative_error"):
        st.warning(f"Comparaison quantitative échouée : {_cmp_res['quantitative_error']}")

    # ── Comparaison Sémantique ────────────────────────────────────────────
    _cmp_sem = _cmp_res.get("semantic")
    if _cmp_sem and not _cmp_res.get("semantic_error"):
        st.markdown("#### Comparaison sémantique (embeddings)")

        _div_cmp  = _cmp_sem.get("diversity", {})
        _rob_cmp  = _cmp_sem.get("robustness", {})
        _comb_cmp = _cmp_sem.get("combined", {})

        _cols_sem = st.columns(3)
        with _cols_sem[0]:
            _da = _div_cmp.get("score_a")
            _db = _div_cmp.get("score_b")
            _dd = _div_cmp.get("delta")
            if _da is not None and _db is not None:
                st.metric(
                    "Diversité (unspecific)",
                    f"B = {_db:.4f}",
                    delta=f"{_dd:+.4f}" if _dd is not None else None,
                    help=f"A (référence) = {_da:.4f}"
                )
        with _cols_sem[1]:
            _ra = _rob_cmp.get("score_a")
            _rb = _rob_cmp.get("score_b")
            _rd = _rob_cmp.get("delta")
            if _ra is not None and _rb is not None:
                st.metric(
                    "Robustesse (specific)",
                    f"B = {_rb:.4f}",
                    delta=f"{_rd:+.4f}" if _rd is not None else None,
                    help=f"A (référence) = {_ra:.4f}"
                )
        with _cols_sem[2]:
            _ca = _comb_cmp.get("score_a")
            _cb = _comb_cmp.get("score_b")
            _cd = _comb_cmp.get("delta")
            if _ca is not None and _cb is not None:
                st.metric(
                    "Score combiné D×R",
                    f"B = {_cb:.4f}",
                    delta=f"{_cd:+.4f}" if _cd is not None else None,
                    help=f"A (référence) = {_ca:.4f}"
                )

        st.info(f" **Resultats :** {_cmp_sem.get('Resultat', '–')}")

        # Graphique : barres comparatives A vs B
        if any(x is not None for x in [_da, _ra]):
            import plotly.graph_objects as go
            _metrics_lbl = []
            _scores_a_list = []
            _scores_b_list = []
            if _da is not None and _db is not None:
                _metrics_lbl.append("Diversité")
                _scores_a_list.append(_da)
                _scores_b_list.append(_db)
            if _ra is not None and _rb is not None:
                _metrics_lbl.append("Robustesse")
                _scores_a_list.append(_ra)
                _scores_b_list.append(_rb)
            if _ca is not None and _cb is not None:
                _metrics_lbl.append("Combiné D×R")
                _scores_a_list.append(_ca)
                _scores_b_list.append(_cb)

            _fig_cmp_sem = go.Figure()
            _fig_cmp_sem.add_trace(go.Bar(
                name=f"Run A (baseline)",
                x=_metrics_lbl, y=_scores_a_list,
                marker_color="#457b9d",
                text=[f"{v:.4f}" for v in _scores_a_list],
                textposition="outside",
            ))
            _fig_cmp_sem.add_trace(go.Bar(
                name=f"Run B (variante)",
                x=_metrics_lbl, y=_scores_b_list,
                marker_color="#e76f51",
                text=[f"{v:.4f}" for v in _scores_b_list],
                textposition="outside",
            ))
            _fig_cmp_sem.update_layout(
                title="Comparaison sémantique : Run A vs Run B",
                barmode="group",
                yaxis=dict(title="Score", range=[0, 1.1]),
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(_fig_cmp_sem, use_container_width=True)

    elif _cmp_res.get("semantic_error"):
        st.warning(f"Comparaison sémantique échouée : {_cmp_res['semantic_error']}")


