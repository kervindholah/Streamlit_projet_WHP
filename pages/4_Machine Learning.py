import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image

fp = "projet_world_happiness_pour_machine_learning.csv"
df = pd.read_csv(fp)

st.title("Choix du modèle de Machine Learning ")

st.markdown(
    """
    ### Feature Scaling
    Toutes nos valeurs sont quantitatives mais elles ne sont pas à la même échelle. C’est pourquoi nous devons 
    normaliser les données avant d’entraîner nos algorithmes de Machine Learning.
    La normalisation des données est une étape essentielle dans la préparation des données pour la mise en place d’un 
    algorithme de machine learning. Elle permet de rendre les variables comparables, d’améliorer la convergence de 
    l’algorithme, d’éviter les erreurs numériques et de faciliter l’interprétation des résultats. Elle est importante 
    pour garantir des résultats précis et fiables.
    
    ### Normalisation ou Standardisation ?
    La Normalisation consiste à borner toutes les valeurs entre 0 et 1. 
    
    La Standardisation consiste à retrancher à chaque valeur la valeur moyenne de la variable et de la diviser par 
    son écart type.
    
    Avec la Normalisation, les prédictions ne pourront pas être inférieures à la valeur minimum actuelle et supérieures 
    à la valeur maximale actuelle. C’est pourquoi nous devons choisir d’utiliser la STANDARDISATION (Z-Score). Nous 
    utiliserons la méthode StandardScaler.
    
    Cette étape sera effectuée après la séparation de notre jeu de données en jeu d'entraînement et jeu de test, 
    pour éviter les fuites d’informations.
    
    Concernant l’étape de Machine Learning, nous devrons utiliser plusieurs modèles, les évaluer, les améliorer et 
    étudier les résultats en utilisant les métriques les plus appropriées pour choisir le modèle le plus performant.

    """)

st.markdown(
    """
    #### Nous avons décidé de ne pas utiliser l'analyse de séries chrnonologique (Time Series)
    Les données du World Happiness Report ne sont pas strictement temporelles dans le sens traditionnel des séries 
    chronologiques. Bien que les données soient associées à différentes années, l'aspect temporel n'est pas l'aspect 
    central de l'analyse. L'objectif est de prédire le bonheur en fonction des caractéristiques socio-économiques et 
    culturelles, ce qui peut être mieux capturé par des modèles de régression plutôt que par des modèles de séries 
    chronologiques.
    """)

st.markdown(
    """
    #### Test de plusieurs modèles de Régression Linéaire
    ##### Encodage de la variable "Regional_indicator" / Jeux de Test --> année 2021
    """)

data = {
    "Modèles": ["LinearRegression", "Lasso", "Ridge", "DecisionTreeRegressor", "RandomForestRegressor"],
    "MSE train": ["0.259414", "1.287871", "0.259414", "0.000000", "0.021873"],
    "MSE test": ["0.241572", "1.147823", "0.241518", "0.093775", "0.056748"],
    "RMSE train": ["0.509327", "1.134844", "0.509327", "0.000000", "0.147895"],
    "RMSE test": ["0.491500", "1.071365", "0.491445", "0.306227", "0.238219"],
    "R2 train": ["0.798572", "0.000000", "0.798571", "1.000000", "0.983016"],
    "R2 test": ["0.789125", "0.001966", "0.789173", "0.918142", "0.950463"],
    "MAE train": ["0.389878", "0.945533", "0.389878", "0.000000", "0.108240"],
    "MAE test": ["0.377233", "0.877528", "0.377214", "0.191034", "0.159895"]
}
resultats1 = pd.DataFrame(data)
st.table(resultats1)


col1, col2, col3 = st.columns(3)
with col1:
    image = Image.open("learning curve - linear.png")
    st.image(image, width=250)
with col2:
    image2 = Image.open("learning curve - lasso.png")
    st.image(image2, width=250)
with col3:
    image3 = Image.open("learning curve - ridge.png")
    st.image(image3, width=250)



col1, col2 = st.columns(2)
with col1:
    image4 = Image.open("learning curve - decision.png")
    st.image(image4, width=250)
with col2:
    image5 = Image.open("learning curve - random.png")
    st.image(image5, width=250)


st.markdown(
    """
    ##### Nous avons également utiliser le Boosting et le Bagging. Deux techniques d'ensemble utilisées pour améliorer 
    les performances des modèles prédictifs en combinant les prédictions de plusieurs modèles individuels. Le boosting 
    construit des modèles successifs en mettant l'accent sur les erreurs de prédiction des modèles précédents, 
    tandis que le bagging entraîne des modèles indépendants sur des sous-ensembles aléatoires des données 
    d'entraînement.
    
    Les résultats n'ont pas été satisfaisant.

    """)

st.markdown(
    """
    #### Choix de l'algorithme : Ridge
    Le choix se porte sur l'algorithme de la régression Ridge en se basant sur les métriques R2 et MAE et les courbes 
    d'apprentissage (learning curves) où on peut observer la stabilité et la cohérence entre le score d'entraînement et 
    le score de validation.
    
    Le coefficient de détermination R2 de 0.789173 indique que le modèle explique environ 79% de la variance totale de 
    la variable cible. Cela signifie que les variables explicatives incluses dans le modèle expliquent une grande 
    partie de la variation observée dans les valeurs réelles de la variable cible. Une valeur de R2 élevée suggère 
    une bonne adéquation du modèle aux données.
    
    L'erreur absolue moyenne (MAE) est une mesure de la performance du modèle qui quantifie l'erreur moyenne entre les 
    prédictions du modèle et les valeurs réelles. Une MAE de 0.377214 signifie qu' en moyenne, les prédictions du 
    modèle diffèrent de 0.377214 unités de la valeur réelle.

    """)

st.markdown(
    """
    ##### Ajustement des hyperparamètres
    En utilisant une technique appelée recherche par grille (GridSearch) 
    """)


data_ = {
    "Modèles": ["Ridge"],
    "MSE test": ["0.241518"],
    "MSE test best params": ["0.241100"],
    "RMSE test": ["0.491445"],
    "RMSE test best params": ["0.491019"],
    "R2 test": ["0.789173"],
    "R2 test best params": ["0.789537"],
    "MAE test": ["0.377214"],
    "MAE test best params": ["0.377063"]
}
resultats2 = pd.DataFrame(data_)
st.table(resultats2)


st.markdown(
    """
     L'ajustement des hyperparamètres n'a pas entraîné d'augmentation des performances du modèle.
    """)

st.markdown(
    """
    ##### Feature Importance
    Mesure utilisée en apprentissage automatique pour évaluer l'influence relative de chaque caractéristique 
    (ou variable) sur les prédictions d'un modèle.
    
    """)



image_feature_importance = Image.open("feature importance.png")
st.image(image_feature_importance)


st.markdown(
    """
    #### MODIFICATION DE LA MÉTHODE DE MACHINE LEARNING SELON L'ÉTUDE DES RÉSULTATS OBTENUS
    Nous avons adapté notre méthodologie car nous avons observé deux problèmes :
    - Les régions en tant que variables explicatives introduisent des biais et des incohérences.
    -  L’utilisation de l’année 2021 en tant que jeu de données de test ne permet pas d’avoir suffisamment de données car nous nous retrouvons avec un test size d’à peine 15%.
    
    ###### Nous décidons donc de ne pas retenir les régions en tant que variables explicatives pour notre modèle de Machine Learning.
    ###### Et de ne pas séparer l’année 2021 de l’ensemble du dataset afin de permettre à notre modèle de prédire de manière plus robuste les scores du bonheur sur de nouvelles données.

    """)

st.markdown(
    """
    Nous avons entraînés et testés les mêmes modèles mais nous n'avons pas obtenus de résultats satisfaisants.
    
    Les résultats étaient moins bons et sur le DecisionTreeRegressor et le RandomForestRegressor nous avons observé 
    de l'overfitting.
    Les algorithmes testés sont soit en situation d'overfitting soit en manque de stabilité ( pour le Ridge regression 
    et le Linear regression, les performances chutes à partir d'un training size de 1200)
    """)

image_learning_curve_ridge2 = Image.open("learning curve - ridge2.png")
st.image(image_learning_curve_ridge2)

st.markdown(
    """
    ##### Pour palier à ces soucis, nous avons testé des algorithmes d'ensemble de Bagging et de Bossting
    """)


data2 = {
    "Modèles": ["BaggingRegressor", "GradientBoostingRegressor", "AdaBoostRegressor"],
    "MSE train": ["0.035862", "0.158285", "0.261139"],
    "MSE test": ["0.185621", "0.224129", "0.279545"],
    "RMSE train": ["0.189374", "0.397851", "0.511017"],
    "RMSE test": ["0.430838", "0.473423", "0.528720"],
    "R2 train": ["0.971630", "0.874783", "0.793417"],
    "R2 test": ["0.852624", "0.822050", "0.778052"],
    "MAE train": ["0.132524", "0.306091", "0.429175"],
    "MAE test": ["0.330822", "0.362469", "0.431310"]
}
resultats_data2 = pd.DataFrame(data2)
st.table(resultats_data2)


col1, col2, col3 = st.columns(3)
with col1:
    image_bagging = Image.open("learning curve - baggingregressor.png")
    st.image(image_bagging, width=250)
with col2:
    image_gradient = Image.open("learning curve - gradientboosting.png")
    st.image(image_gradient, width=250)
with col3:
    image_ada = Image.open("learning curve - adaboost.png")
    st.image(image_ada, width=250)


st.markdown(
    """
    #### Le choix se porte sur l'algorithme GradientBoostingRegressor
    En se basant sur les métriques R2 et MAE et les courbes d'apprentissage (learning curves) où on peut observer 
    la stabilité et la cohérence entre le score d'entraînement et le score de validation.
    - Le coefficient de détermination R2 de 0.82 indique que le modèle explique environ 85% de la variance totale de la variable cible.
    - L'erreur absolue moyenne (MAE) de 0.36 signifie que, en moyenne, les prédictions du modèle diffèrent de 0.329074 unités de la valeur réelle.
    """)


st.markdown(
    """
    ##### Ajustement des hyperparamètres
    L'ajustement des hyperparamètres nous permet d'améliorer les performances de l'algorithme. Nous passons d'un R2 de 
    0.82 à 0.84 et d'une MAE de 0.361853 à 0.33
    
    """)

data3 = {
    "Modèles": ["GradientBoostingRegressor"],
    "MSE test": ["0.224129"],
    "MSE best_params": ["0.195384"],
    "RMSE test": ["0.473423"],
    "RMSE best_params": ["0.442022"],
    "R2 test": ["0.822050"],
    "R2 best_params": ["0.844872"],
    "MAE test": ["0.362469"],
    "MAE best_params": ["0.329985"],
}
resultats_data3 = pd.DataFrame(data3)
st.table(resultats_data3)

image_learning_curve_gradient_bestparams = Image.open("learning curve - gradientboosting_bestparams.png")
st.image(image_learning_curve_gradient_bestparams, width=500)

st.markdown(
    """
    ##### Feature Importance
    """)
image_feature_importance2 = Image.open("feature importance2.png")
st.image(image_feature_importance2)


