import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title("Analyse exploratoire et datavisualisation")

st.markdown(
    """
    #### La variable cible : "Life_Ladder" ou "score du bonheur"
    Le dataset contient les scores du bonheur de 2005 à 2021. La répartition géographique est sur 10 régions avec 166 pays au total.

    Les 166 pays ne sont pas présents sur toutes les années. Concernant les régions, 7 régions sont représentées en 2005 puis à partir de 
    2006 nous retrouvons les 10 régions.
    """
)

st.markdown(
    """
    #### Evolution du score du bonheur dans le monde de 2005 à 2021
    """
)

fp = "projet_world_happiness_concatenation.csv"
df = pd.read_csv(fp)

df_mean_score_per_year = df.groupby("year", as_index=False)["Life_Ladder"].mean()
fig = plt.figure(figsize=(10, 5))
fig = px.line(df_mean_score_per_year, x="year", y="Life_Ladder")
fig.update_layout(title_text="Evolution de la moyenne du score mondial de 2005 à 2021", width=800)
st.plotly_chart(fig)

st.markdown(
    """
    #### Comparez l'évolution du score du bonheur entre pays et par rapport à la moyenne mondiale
    """
)

# 2è graphique permettant de choisir des pays et de comparer
# Sélection des pays via une liste déroulante
selected_countries = st.multiselect("Choisissez un ou plusieurs pays", df["Country_name"].unique())
# Filtrer le DataFrame en fonction des pays sélectionnés
filtered_df = df[df["Country_name"].isin(selected_countries)]
# Créer le graphique interactif avec Plotly Express
color_map = {"Life_Ladder": "red"}  # Couleur rouge pour la courbe mondiale
for country in selected_countries:
    color_map[country] = px.colors.qualitative.Set1[len(color_map) % len(px.colors.qualitative.Set1)]
# Créer le graphique interactif avec Plotly Express
fig = px.line(filtered_df, x="year", y="Life_Ladder", color="Country_name", color_discrete_map=color_map)
fig.add_trace(px.line(df_mean_score_per_year, x="year", y="Life_Ladder", name="Moyenne mondiale").data[0])  # Ajouter la courbe mondiale
fig.update_layout(title_text="Evolution des scores du bonheur de 2005 à 2021", width=800)
st.plotly_chart(fig)




st.markdown(
    """
    Trois nouvelles régions sont entrées dans le World Happiness report en 2006 : 
    « Common wealth of Independent States », « Sub-Saharan Africa » et « South Asia ».

    Ces 3 régions sont arrivées dans le dataset avec un score du bonheur faible :
    - Commonwealth of Independent States : 4.83
    - Sub-Saharan Africa : 4.07
    - South Asia : 5.23
    Ces pays à faibles scores du bonheur sont également des pays à faible PIB par habitant. 
    Cette relation entre le PIB par habitant et le score du bonheur sera à analyser par la suite !


    """
)

st.markdown(
    """
    #### Représentation sur carte des pays présents par année
    """
)

fig = px.choropleth(df.sort_values("year"),
                    locations="Country_name",
                    color="Life_Ladder",
                    locationmode="country names",
                    animation_frame="year",
                    color_continuous_scale="Viridis")
fig.update_layout(title="Life Ladder par pays", title_x=0.5)
fig.update_layout(width=1000, height=700)
st.plotly_chart(fig)

st.markdown(
    """
    #### PIB et score du bonheur
    """
)

df_mean_per_region = df.groupby("Regional_indicator", as_index=False)[["Life_Ladder", "Log_GDP_per_capita"]].mean()
df_mean_per_region = df_mean_per_region.sort_values(by="Life_Ladder", ascending=False)
df_mean_per_region_pib = df_mean_per_region.sort_values(by="Log_GDP_per_capita", ascending=False)
st.table(df_mean_per_region.set_index("Regional_indicator"))



st.markdown("Le classement des régions par PIB par habitant correspond au classement des régions par score du bonheur.")

st.markdown("Dans la continuité de notre analyse et pour suivre la tendance que nous avons observé qui est la relation "
            "entre le PIB par habitants et le score du bonheur, il semble intéressant de suivre l’évolution du score du "
            "bonheur selon le PIB par habitant.")

df_mean_pib_per_year_region = df.groupby(["year", "Regional_indicator"],
                                         as_index=False)[["Life_Ladder", "Log_GDP_per_capita"]].mean()
df_mean_pib_per_year_region["quartiles_PIB"] = pd.qcut(df_mean_pib_per_year_region["Log_GDP_per_capita"],
                                                       q=4, labels=False)

fig = plt.figure(figsize=(12, 8))
sns.set_theme()
palette = {0: "blue", 1: "red", 2: "black", 3: "green"}
sns.lineplot(data=df_mean_pib_per_year_region, x="year", y="Life_Ladder", hue="quartiles_PIB",
             ci=None, linewidth=2.5, palette=palette)
plt.title("Evolution de la moyenne du score par niveau de PIB de 2005 à 2021")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', labels=['q1', 'q2', 'q3', 'q4'])
st.pyplot(fig)

st.markdown(
    """
    Les scores de bonheur varient légèrement, mais ils sont généralement associés à une plage de scores correspondant 
    au niveau de PIB par habitant. 
    """
)



st.markdown(
    """
    #### Boxplot
    """
)
df_mean_country = df.groupby(["Country_name", "Regional_indicator"],
                             as_index=False)["Life_Ladder"].mean().sort_values(by="Life_Ladder", ascending=False)

fig = plt.figure(figsize=(15, 8))
sns.boxplot(data=df_mean_country, x="Regional_indicator", y="Life_Ladder")
plt.xticks(rotation=90)
plt.ylim(3, 8)
plt.grid(axis='y', linestyle='--')

mean_global = df_mean_country.Life_Ladder.mean()
plt.axhline(y=mean_global, color='r', linestyle='--', label="Moyenne générale du Score du bonheur")

plt.legend(loc="upper right")
st.pyplot(fig)

st.markdown(
    """
    Sur ce diagramme en boîte on peut s’apercevoir que deux régions sont largement en tête dans le classement 
    du bonheur « Western Europe » et « North America and ANZ ». Pour ces deux régions, tous les pays ont un score 
    du bonheur supérieur à la moyenne mondiale.

    Les scores les plus bas se trouvent dans les régions « Sub-Saharan Africa » et « South Asia » où tous les pays ont 
    un Life_Ladder inférieur à la moyenne mondiale. 

    Quasiment 75% des pays de « Sub-Saharan Africa » sont en-dessous de 4.5 

    Les scores les plus hétérogènes se situent dans la région « Middle East and North Africa ».
    """
)

st.markdown(
    """
    #### Classement des pays selon le score du bonheur : Top 10 et bottom 10
    """
)
df_mean_per_country = df.groupby("Country_name", as_index=False)["Life_Ladder"].mean().sort_values(by="Life_Ladder",
                                                                                                   ascending=False)

# Les 10 meilleurs
top_10 = df_mean_per_country.head(10)
fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
sns.set_theme()
sns.barplot(ax=axs[0], data=top_10, x="Country_name", y="Life_Ladder", color="b")
axs[0].set_title("Les 10 meilleurs scores")
axs[0].set_xlabel("Pays")
axs[0].set_ylabel("Score")
axs[0].set_ylim([0, 8])
axs[0].xaxis.set_tick_params(rotation=75)

# Les 10 derniers
bottom_10 = df_mean_per_country.tail(10)
sns.barplot(ax=axs[1], data=bottom_10, x="Country_name", y="Life_Ladder", color="b")
axs[1].set_title("Les 10 derniers scores")
axs[1].set_xlabel("Pays")
axs[1].set_ylabel("Score")
axs[1].set_ylim([0, 8])
axs[1].xaxis.set_tick_params(rotation=75)

# Ajuster la mise en page pour éviter que les graphiques se chevauchent
plt.tight_layout()

# Afficher les graphiques dans Streamlit
st.pyplot(fig)

st.markdown(
    """
    Tous les pays du top 10 sont soit des pays de « Wester Europe » soit des pays de « North America and ANZ ». Le 
    dernier pays du classement est de « South Asia » et 6 pays du bottom 10 sont des pays de « Sub-Saharan Africa ».
    """
)










