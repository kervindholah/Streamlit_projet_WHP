
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="World Happiness Report",
    page_icon="👋",
)

st.write("# Projet analyse du bonheur")

# st.write("## Contexte")

# st.sidebar.success("Select a demo above.")

st.markdown(
    """
    ## Contexte
    Le World Happinness Report est une initiative importante qui vise à mesurer et à comprendre le 
    bonheur dans le monde. Les données collectées dans ce rapport permettent aux chercheurs de mieux 
    comprendre les facteurs qui contribuent au bonheur et de proposer des solutions pour améliorer la 
    qualité de vie des individus.
    Il s'agit d'une enquête annuelle menée par l'Organisation des Nations Unies pour évaluer le niveau 
    de bonheur dans les pays du monde entier. Le rapport est basé sur des enquêtes auprès des citoyens 
    des pays participants, ainsi que sur des données économiques et sociales.
   
    ## Objectifs
    Notre premier objectif est de proposer une analyse du bonheur sur Terre en répondant à la problématique suivante :
    - ###### Quels sont les facteurs qui influencent le score du bonheur ?
   
   Notre second objectif est de mettre en place un algorithme de prédiction du score du bonheur
   
    ### Biais possibles
    - Biais culturel
    - Biais de sélection
    - Biais lié à la méthode de collecte de données
    - Biais temporel
"""
)


st.markdown(
    """
    ## Collecte des données
    Nous avons à notre disposition deux jeux de données librement accessible sur Kaggle.
    - world-happiness-report.csv
    - world-happiness-report-2021.csv
    Ces deux jeux de données regroupent des données de 2005 à 2021.

    Après avoir concaténé et nettoyé le dataset, le jeu de données compte 2098 lignes pour 10 colonnes :
    """
)

fp = "projet_world_happiness_pour_machine_learning.csv"
df = pd.read_csv(fp)

st.dataframe(df)

st.markdown(
    """
    ## Explication des variables
    - Life_Ladder : Il s’agit de la réponse moyenne nationale à la question sur l’évaluation de la vie. C'est le score du bonheur mesuré sur une échelle de 0 à 10
    - Country_name : Nom du pays
    - Regional_indicator : Région du monde auquel appartient le pays
    - year : Année de sondage
    - Log_GDP_per_capita : PIB par habitant en parité de pouvoir d’achat
    - Social_support : Mesure du soutien social perçu, mesurée sur une échelle de 0 à 1
    - Healthy_life_expectancy_at_birth : L'espérance de vie en bonne santé mesurée en années
    - Freedom_to_make_life_choices : Mesure de la liberté perçue pour faire des choix de vie, mesurée sur une échelle de 0 à 1
    - Generosity : Mesure de la générosité perçue, mesurée sur une échelle de 0 à 1
    - Perceptions_of_corruption : Mesure de la perception de la corruption dans le pays, mesurée sur une échelle de 0 à 1

    """
)


