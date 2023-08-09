import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV


st.title("Application de l’algorithme")


model_path = "GDBregressor.joblib"
GBR_model = load(model_path)

fp = "projet_world_happiness_pour_machine_learning.csv"
df = pd.read_csv(fp)

# Style personnalisé pour la colonne "Life_Ladder"
def highlight_target(s):
    if s.name == "Life_Ladder":
        return ['background-color: yellow'] * len(s)
    else:
        return [''] * len(s)


# Affichage du dataframe
# Affichage du DataFrame avec style personnalisé
st.dataframe(df.style.apply(highlight_target, axis=0))

# Options de tri
sort_options = ['Ordre croissant', 'Ordre décroissant']


# Liste déroulante pour choisir le Country_name
country_name = st.selectbox('Choisir un Country_name:', df['Country_name'].unique())

# Affichage du dataframe correspondant au Country_name choisi
filtered_df = df[df['Country_name'] == country_name]

# Affichage du DataFrame filtré avec style personnalisé
st.dataframe(filtered_df.style.apply(highlight_target, axis=0))


# -----------------------------------------------------------

# intéractivité

st.write("Choisissez un pays pour pouvoir changer les paramètres :")
selected_country = st.selectbox('Country_name', df['Country_name'].unique())

# Filtrer le DataFrame pour obtenir la ligne correspondante à l'année 2021 ou la plus élevée si non disponible
filtered_row = df[df['Country_name'] == selected_country].nlargest(1, 'year')
score = filtered_row["Life_Ladder"]
filtered_row = filtered_row.drop(['Life_Ladder'], axis=1)

selected_index = filtered_row.index[0]  # Obtenir l'index de la ligne filtrée

# Obtenir la ligne correspondante
X = df.drop(['Life_Ladder', 'year', 'Country_name', 'Regional_indicator'], axis=1)
df_row = X[X.index == selected_index]

# Afficher la ligne correspondante
st.write(df_row)

# Affichage de la valeur de "Life_Ladder" séparée
st.write("Valeur de Life_Ladder : ", score)

# Séparer les caractéristiques en deux groupes
features_slider = df_row.columns[:6]

if 'Log_GDP_per_capita' in features_slider:
    current_value1 = df_row['Log_GDP_per_capita']
    new_value1 = st.slider("Entrez une nouvelle valeur pour Log_GDP_per_capita", min_value=0.0, max_value=20.0, value=float(current_value1))
    df_row['Log_GDP_per_capita'] = new_value1

if 'Social_support' in features_slider:
    current_value2 = df_row['Social_support']
    new_value2 = st.slider("Entrez une nouvelle valeur pour Social_support", min_value=0.0, max_value=1.0, value=float(current_value2))
    df_row['Social_support'] = new_value2

if 'Healthy_life_expectancy_at_birth' in features_slider:
    current_value3 = df_row['Healthy_life_expectancy_at_birth']
    new_value3 = st.slider("Entrez une nouvelle valeur pour Healthy_life_expectancy_at_birth", min_value=20.0, max_value=100.0, value=float(current_value3))
    df_row['Healthy_life_expectancy_at_birth'] = new_value3

if 'Freedom_to_make_life_choices' in features_slider:
    current_value4 = df_row['Freedom_to_make_life_choices']
    new_value4 = st.slider("Entrez une nouvelle valeur pour Freedom_to_make_life_choices", min_value=0.0, max_value=1.0, value=float(current_value4))
    df_row['Freedom_to_make_life_choices'] = new_value4

if 'Generosity' in features_slider:
    current_value5 = df_row['Generosity']
    new_value5 = st.slider("Entrez une nouvelle valeur pour Generosity", min_value=-1.0, max_value=1.0, value=float(current_value5))
    df_row['Generosity'] = new_value5

if 'Perceptions_of_corruption' in features_slider:
    current_value6 = df_row['Perceptions_of_corruption']
    new_value6 = st.slider("Entrez une nouvelle valeur pour Perceptions_of_corruption", min_value=0.0, max_value=1.0, value=float(current_value6))
    df_row['Perceptions_of_corruption'] = new_value6

# Afficher la ligne correspondante mise à jour
df_row_modified = pd.DataFrame(df_row)
st.write(df_row_modified)

# Afficher un message pour rafraîchir la page
st.info("Rafraîchissez la page pour récupérer les valeurs d'origine")

# Afficher un message pour obtenir la prédiction
st.info("Cliquer sur 'prédiction' pour obtenir le nouveau score du bonheur")

# Afficher le bouton "Prédiction"
ok = st.button("Prédiction")

index_b = df_row_modified.index[0]

if ok:
    X.loc[index_b] = df_row_modified.iloc[0]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(X)
    choice_scaled = scaler.transform(df_row_modified)

    prediction = GBR_model.predict(choice_scaled)

#    prediction = model.predict(choice_scaled)
    st.write("Résultat de la prédiction de Life Ladder :", prediction)


