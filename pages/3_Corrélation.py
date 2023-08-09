import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

fp = "projet_world_happiness_concatenation.csv"
df = pd.read_csv(fp)

st.title("Étude des corrélations entre les différentes variables")

# st.markdown(""" """)

st.markdown(
    """
    Pour mesurer la corrélation entre les deux variables, on s'appuiera sur le coefficient de corrélation de Pearson.
    Le coefficient de corrélation de Pearson est une formule qui permet de quantifier la relation linéaire entre deux 
    variables : le coefficient est un réel entre -1 et 1 avec :
    - 1 les variables sont corrélées
    - 0 les variables sont décorrélées
    - -1 les variables sont corrélées négativement
    """)

st.markdown(
    """
    #### Heatmap
    Nous affichons une HeatMap (fonction graphique de Seaborn permettant d’afficher un tableau, colorisé en fonction 
    des résultats du test de corrélation de Pearson ainsi que le coefficient de corrélation).
    """)
df_numeric = df.drop(["Regional_indicator","Country_name","year"], axis=1)
plt.figure(figsize=(10,6))
sns.heatmap(df_numeric.corr(), annot=True, linewidths = 2, linecolor="Yellow", cmap="YlGnBu")
st.pyplot(plt)

st.markdown(
    """
   Les corrélations les plus importantes avec le score du bonheur : 
   - Log_GDP_per_capita (PIB par habitant) --> 0.79
   - Social_support (le soutien social) --> 0.71
   - Healthy_life_expectancy_at_birth (l'espérance de vie en bonne santé) --> 0.75
    """)


st.markdown(
    """
   #### Représentation visuelle des corrélations entre les variables
    """)
fig = plt.figure(figsize=(12, 5))
sns.set_theme()
sns.regplot(data=df, x="Log_GDP_per_capita", y="Life_Ladder",
            line_kws={"color": "red"}, scatter_kws={"s": 10, "alpha": 0.5, "color": "blue"})
plt.title("Corrélation entre le score du bonheur et le PIB par habitant")
plt.xlabel("Log_GDP_per_capita")
plt.ylabel("Score de bonheur")
st.pyplot(fig)

fig = plt.figure(figsize=(12, 5))
sns.set_theme()
sns.regplot(data=df, x="Social_support", y="Life_Ladder",
            line_kws={"color": "red"}, scatter_kws={"s": 10, "alpha": 0.5, "color": "blue"})
plt.title("Corrélation entre le score du bonheur et le le soutien social")
plt.xlabel("Social_support")
plt.ylabel("Score de bonheur")
st.pyplot(fig)

fig = plt.figure(figsize=(12, 5))
sns.set_theme()
sns.regplot(data=df, x="Healthy_life_expectancy_at_birth", y="Life_Ladder",
            line_kws={"color": "red"}, scatter_kws={"s": 10, "alpha": 0.5, "color": "blue"})
plt.title("Corrélation entre le score du bonheur et l'espérance de vie en bonne santé")
plt.xlabel("Healthy_life_expectancy_at_birth")
plt.ylabel("Score de bonheur")
st.pyplot(fig)

fig = plt.figure(figsize=(12, 5))
sns.set_theme()
sns.regplot(data=df, x="Freedom_to_make_life_choices", y="Life_Ladder",
            line_kws={"color": "red"}, scatter_kws={"s": 10, "alpha": 0.5, "color": "blue"})
plt.title("Corrélation entre le score du bonheur et la liberté de faire des choix")
plt.xlabel("Freedom_to_make_life_choices")
plt.ylabel("Score de bonheur")
st.pyplot(fig)

fig = plt.figure(figsize=(12, 5))
sns.set_theme()
sns.regplot(data=df, x="Generosity", y="Life_Ladder",
            line_kws={"color": "red"}, scatter_kws={"s": 10, "alpha": 0.5, "color": "blue"})
plt.title("Corrélation entre le score du bonheur et la générosité")
plt.xlabel("Generosity")
plt.ylabel("Score de bonheur")
st.pyplot(fig)

fig = plt.figure(figsize=(12, 5))
sns.set_theme()
sns.regplot(data=df, x="Perceptions_of_corruption", y="Life_Ladder",
            line_kws={"color": "red"}, scatter_kws={"s": 10, "alpha": 0.5, "color": "blue"})
plt.title("Corrélation entre le score du bonheur et la perception de la corruption")
plt.xlabel("Perceptions_of_corruption")
plt.ylabel("Score de bonheur")
st.pyplot(fig)


st.markdown(
    """
    Nous pouvons constater visuellement qu’effectivement les corrélations sont importantes entre la variable cible et 
    les variables « Log_GDP_per_capita », « Social_support » et « Healthy_life_expectancy_at_birth ». Ces relations 
    sont linéaires. 
    Nous optons donc pour l’étape de modélisation pour un algorithme d’apprentissage supervisé et pour un modèle de 
    Régression Linéaire.
    """
)



