import time
import streamlit as st

from client.utils import *

API_URL = "http://127.0.0.1:8000/"

MODELS, MODELS_DETAILS = get_models()
LANGUAGES, LANGUAGES_DETAILS = get_languages()
DATASET_TYPES, DATASET_TYPES_DETAILS = get_dataset_types()
STRATEGIES, STRATEGIES_DETAILS = get_strategies()

st.set_page_config(
    page_title="Cultural Robustness and Diversity",
    # layout="wide",
)

st.title("Cultural Robustness and Diversity (ELOQUENT @ CLEF 2026)")

with st.form("experience"):
    st.write("Personnaliser les paramètres de l'expérience")

    # 1. Choix du modèle
    model = st.radio(
        label="Choisir un modèle :",
        options=MODELS,
        captions=MODELS_DETAILS,
        horizontal=True,
    )

    # 2. Choix des langues
    languages = st.pills(
        label="Choisir les langues : ",
        options=LANGUAGES,
        default=LANGUAGES,
        selection_mode="multi",
        required=True,
    )

    # 3. Choix du type de dataset
    type_dataset = st.radio(
        label="Choisir le type de dataset :",
        options=DATASET_TYPES,
        captions=DATASET_TYPES_DETAILS,
        horizontal=True,
    )

    # 4. Choix des autres paramètres
    # TODO: add
    # Parameters: longueur, temperature, déterminisme...

    # 5. Choix de la stratégie
    strategy = st.radio(
        label="Choisir la stratégie :",
        options=STRATEGIES,
        captions=STRATEGIES_DETAILS,
        horizontal=True,
    )

    # TODO: bouton grisé si cliqué ou formulaire incomplet
    submitted = st.form_submit_button(
        label="Lancer l'expérience",
        on_click=run_experience(model, languages, type_dataset, strategy),  # TODO: add parameters
    )

if submitted:
    with st.spinner("Expérience lancée...", show_time=True):
        time.sleep(5)  # TODO: wait until response from experience

# TODO: si succès -> message de réussite (+ résultats), sinon, message d'erreur
# st.success("This is a success message!", icon="✅")
# st.error("This is an error", icon="🚨")

# TODO: si succès -> bouton pour télécharger les fichiers
(st.button(
    label="Télécharger *submission.zip*",
    on_click=download_submission_package()
))
