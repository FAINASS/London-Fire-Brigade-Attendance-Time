# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import pandas as pd
pd.set_option('display.max_colwidth', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder,MinMaxScaler,OrdinalEncoder,RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

from PIL import Image


#Configurer l'affichage en mode Wide
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title = "Temps de Réponse de la Brigade des Pompiers de Londres")


## Supprimer l'espace vide en haut de la page
st.markdown("""
<style>

.block-container
{
    padding-top: 1rem;
    padding-bottom: 5rem;
    margin-top: 0rem;
}

</style>
""", unsafe_allow_html=True)

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(https://www.kenistonha.co.uk/wp-content/uploads/2017/06/London-fire-brigade-2-992x561.png);
                background-repeat: no-repeat;
                background-size: 250px 125px;
                padding-top: 80px;
                background-position: 40px 20px;
                margin-top: 4px;

            }
        """,
        unsafe_allow_html=True,
    )


def main():
    
    add_logo()

    st.header("🏋️ Modélisation")
    st.write("L'objectif de cette étape est de développer un modèle de Machine Learning pour répondre à l'objectif initial.")
    
    
    df = load_data("df_EnrichiModelisation.csv")

    all_years = df["YearOfTheCall"].unique().tolist()
    
    min_year = min(all_years)
    max_year = max(all_years)
    
    expander = st.sidebar.expander("**CHOISIR UNE PÉRIODE**",expanded=False)
    start_year, end_year = expander.slider("Date of Call", min_year, max_year, (2022, 2022))
    
    df = df[(df['YearOfTheCall'] >= start_year) & (df['YearOfTheCall'] <= end_year)]

    st.subheader(" ")
    st.subheader("0. Choix d'un modèle")
    model_type = st.selectbox("Choisir un modèle :", ['Ridge','XGBRegressor'])
    
    if model_type == "XGBRegressor" :
        df = df[["IncidentGroupType", "BoroughName","WardName","HourOfCall","PropertyType","DeployedFromStationName","Distance","NumStationsWithPumpsAttending",
                 "LatitudeIncident","LongitudeIncident","LatitudeStation","LongitudeStation","SecondPumpArrivingDeployedFromStation","AttendanceTime"]]

        with st.expander("Explications", expanded=True):
                st.markdown("""
                    Le XGBRegressor est une méthode de prédiction basée sur le principe du gradient boosting. 
                    
                    C'est comme si vous aviez une équipe de chercheurs (les arbres de décision) qui travaillent ensemble pour résoudre un problème complexe (la prédiction). Chaque chercheur apporte sa propre expertise et ses propres idées, et ensemble, ils arrivent à une solution plus précise et plus robuste qu'un seul chercheur ne pourrait le faire.
                    
                    Dans le cas du XGBRegressor, chaque "chercheur" est un arbre de décision. Le modèle commence par un seul arbre, puis ajoute progressivement d'autres arbres pour corriger les erreurs faites par les arbres précédents. C'est ce qu'on appelle le gradient boosting.
                    
                    Le XGBRegressor est particulièrement efficace lorsque vous avez beaucoup de données et de nombreuses variables. 
                    
                    Cependant, il faut noter que son temps de traitement est généralement plus long comparé au modèle Ridge.
                    """)
                        
    else :
        
        df = df[["Distance","DeployedFromStationName","WardName","LongitudeStation","LongitudeIncident","ResourceCode","BoroughName","WeekOfTheCall","MonthOfTheCall",
    "Region","MomentOfTheDay","PropertyType","AttendanceTime"]]

        with st.expander("Explications", expanded=True):
            st.markdown("""
                La régression Ridge est une méthode utilisée en statistiques pour prédire des données. 
                
                Elle fonctionne un peu comme une recette de cuisine : on a plusieurs ingrédients (les données d'entrée) et on veut obtenir un plat délicieux (la prédiction). 
                
                Cependant, parfois, certains ingrédients peuvent prendre le dessus et gâcher le plat. Pour éviter cela, la régression Ridge pénalise les ingrédients trop dominants, c'est-à-dire qu'elle réduit leur importance dans la recette. 
                
                Le but est d'obtenir un plat (une prédiction) qui est un bon équilibre de tous les ingrédients (données d'entrée), plutôt que d'être dominé par un ou deux ingrédients. """)
 

    # Séparation des features (X) et de la variable cible (y)
    X = df.drop('AttendanceTime', axis=1)
    y = df['AttendanceTime']
    
    # Séparation des données d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)
    
    st.subheader(" ")
    
########################################################################################################################################################################################################################## 
    st.subheader("1. Pré-traitement des données")
    
    my_expander = st.sidebar.expander("**PRÉTRAITER LES DONNÉES**",expanded=False)
    my_expander2 = st.sidebar.expander("**RÉGLER LES HYPERPARAMÈTRES**",expanded=True)
    
    encoder_type = my_expander.selectbox("Type d'encodeur", ['OneHotEncoder','OrdinalEncoder'])
    
    scaler_type = my_expander.selectbox('Type de normalisateur', ['StandardScaler','MinMaxScaler','RobustScaler'])
    
    st.markdown(f"<div style='text-align: left; color: black; background-color: #ff9800; padding: 10px; border-radius: 5px;'>⚠️ Sur la période de {start_year} à {end_year}, vous avez pré-traité les données avec un {encoder_type} et un {scaler_type}.</div>", unsafe_allow_html=True)
    st.subheader(" ")
    
    
    numeric_features = make_column_selector(dtype_include=np.number)
    categorical_features = make_column_selector(dtype_exclude=np.number)
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
        ('scaler', eval(scaler_type)())
    ])
    

    if encoder_type == 'OneHotEncoder':
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant',fill_value='missing')),
            ('encoder', eval(encoder_type)(drop='first', sparse_output=True, handle_unknown='ignore'))
        ])
    
    else:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant',fill_value='missing')),
            ('encoder', eval(encoder_type)(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
    

    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', categorical_transformer, categorical_features),
            ('numeric', numeric_transformer, numeric_features) 
        ])
    

    array_transformed = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(array_transformed, "toarray"):

        df_transformed = pd.DataFrame(array_transformed.toarray(), columns=feature_names)
    
    else:

        df_transformed = pd.DataFrame(array_transformed, columns=feature_names)
    
    st.dataframe(df_transformed.head(7))

    with st.expander("Comprendre le prétraitement des données", expanded=False):
        
            st.markdown("""
                Les algorithmes de machine learning peuvent nécessiter que toutes les variables soient numériques. Le prétraitement permet de transformer les données pour qu’elles soient compatibles avec ces exigences.
                ######   
                ##### Choix d’un encodeur (pour les variables catégorielles)
                - Ordinal Encoder : 
                c’est donner un numéro à chaque élément d’une liste. Par exemple, dans une liste d’interventions, l’incendie serait le numéro 1, le sauvetage le numéro 2.
                    
                - One-Hot Encoder :
                pour ce type d’encodeur, une colonne est créée pour chaque type d’intervention. Si l’intervention est présente, un 1 est mis dans cette colonne, si elle n’est pas présente, un 0 est mis. 
                Par exemple, pour "Incendie", on aurait : Incendie (1) Sauvetage (0) soit 10.

                ######   
                
                ##### Choix d'un normalisateur (pour les variables numériques)
                - MinMaxScaler :
                imaginez que vous avez une échelle de 1 à 100, mais vos données sont comprises entre 200 et 300. Le MinMaxScaler va simplement "recalibrer" vos données pour qu’elles soient comprises entre 1 et 100.
                
                - StandardScaler :
                imaginez que vous mesurez la taille de personnes en centimètres et en pouces. Il serait difficile de les comparer directement car elles sont sur des échelles différentes. Le StandardScaler convertit toutes les mesures dans une "langue commune" (comme convertir toutes les mesures en centimètres), ce qui rend la comparaison plus facile.

                - RobustScaler :
                imaginez que vous mesurez le poids de plusieurs objets, dont un éléphant. Le poids de l’éléphant est une valeur aberrante et peut fausser vos résultats. Le RobustScaler réduit l’impact de ces valeurs aberrantes en se concentrant sur la majorité des données, ce qui donne une meilleure représentation générale.
              """)

    
########################################################################################################################################################################################################################## 
    st.subheader("2. Performances de votre modèle")
    st.subheader(" ")
    
    if model_type == 'XGBRegressor':
       colsample_bytree = my_expander2.slider('Colsample bytree', min_value=0.1, max_value=1.0, value=0.7746831999163204)
       learning_rate = my_expander2.slider('Learning rate', min_value=0.1, max_value=1.0, value=0.0624207548570334)
       max_depth = my_expander2.slider('Max Depth', min_value=1, max_value=12, value=1)
       min_child_weight = my_expander2.slider('Min child weight', min_value=1, max_value=5, value=1)
       n_estimators = my_expander2.slider('N_estimators', min_value=100, max_value=1200, value=300)

       model = XGBRegressor(colsample_bytree =colsample_bytree,  
       learning_rate = learning_rate,  
       max_depth = max_depth,  
       min_child_weight = min_child_weight, 
       n_estimators = n_estimators, 
       random_state=0)
        
       with st.expander("Comprendre le réglage des hyperparamètres", expanded=False):
            
                st.markdown("""
                    Le réglage des hyperparamètres peut aider à améliorer les performances de votre modèle.
                    - **Colsample_bytree** : pensez à cela comme à un tirage au sort parmi vos caractéristiques (taille, poids, âge, etc.) 
                    pour faire une prédiction. Ce paramètre détermine combien de ces caractéristiques sont sélectionnées pour le tirage.

                    - Learning_rate : c’est la vitesse à laquelle votre modèle apprend. Un rythme d’apprentissage plus lent signifie 
                    que votre modèle prend son temps pour apprendre, ce qui peut réduire les erreurs. 
                    Un rythme plus rapide signifie que votre modèle apprend rapidement, mais il peut être plus sujet aux erreurs.

                    - Max_depth : c’est comme choisir le nombre de questions que vous voulez poser avant de faire une prédiction. 
                    Plus vous posez de questions, plus vous obtenez de détails, mais il y a aussi un risque de se perdre dans les détails.

                    - Min_child_weight : c’est comme décider combien d’informations vous avez besoin avant de poser une nouvelle question. 
                    Plus vous avez besoin d’informations, moins vous posez de questions, ce qui peut rendre votre modèle plus simple et 
                    plus facile à comprendre.

                    - N_estimators : c’est le nombre de fois que vous voulez répéter le processus d’apprentissage. 
                    Plus vous répétez, plus vous pouvez apprendre, mais cela peut aussi prendre plus de temps et rendre le processus plus complexe.
                  """)
    
    else:
        alpha = my_expander2.slider('Alpha', min_value=1.0, max_value=50.0, value=9.372353071731432)
        solver = my_expander2.selectbox('Solver', ['auto', 'lsqr', 'sparse_cg', 'sag'])
        fit_intercept = my_expander2.checkbox('fit_intercept', value=True)
        model = Ridge(alpha=alpha, solver=solver, fit_intercept=fit_intercept)
        
        with st.expander("Comprendre le réglage des hyperparamètres", expanded=False):
            
                st.markdown("""
                    Le réglage des hyperparamètres peut aider à améliorer les performances de votre modèle.
                    - Alpha : c’est comme un gardien qui contrôle la complexité de votre modèle. Un alpha élevé rend le modèle plus simple, 
                    tandis qu’un alpha faible le rend plus complexe, ce qui peut conduire à un surapprentissage si la valeur est trop faible.

                    - Solver : c’est l’algorithme utilisé pour effectuer la régression. Selon le type de données, certains solveurs peuvent être 
                    plus efficaces que d’autres. Par exemple, en réglant le solver sur “auto”, le modèle choisira lui-même le solver le plus adapté.

                    - Fit_intercept : c’est le terme constant de la régression linéaire. Si vous cochez la case fit_intercept, 
                    le modèle essaiera de trouver la meilleure valeur pour l’ordonnée à l’origine. En revanche, si vous ne cochez pas la case 
                    fit_intercept, le modèle supposera que les données sont déjà centrées et n’effectuera pas de calcul pour l’ordonnée à l’origine.
                  """)


    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('estimator', model)
    ])

    
    # Entraînement du modèle sur les données d'entraînement
    model_pipeline.fit(X_train, y_train)
    
    # Prédiction sur les données d'entraînement et de test
    y_train_pred = model_pipeline.predict(X_train)
    y_test_pred = model_pipeline.predict(X_test)
    
    # Calcul des métriques d'évaluation
    Train_score = round(model_pipeline.score(X_train, y_train), 2)
    Train_RMSE= round(mean_squared_error(y_train, y_train_pred, squared=False), 3)
    Test_score= round(model_pipeline .score(X_test, y_test), 2)
    Test_RMSE= round(mean_squared_error(y_test, y_test_pred, squared=False), 3)
    
    data_score_after = pd.DataFrame({
    'Model': [str(model_pipeline["estimator"]).split("(")[0]],
    'R² Train': [Train_score],
    'R² Test': [Test_score],
    'Train RMSE': [Train_RMSE],
    'Test RMSE': [Test_RMSE] })
            
    st.write(data_score_after)
    
    
    with st.expander("Explications", expanded=False):
        st.markdown("""
        #### Métriques d'évaluation
        Nous avons utilisé 2 métriques pour évaluer les performances des modèles :
        
        1. **Coefficient de détermination (R²)** : 
            - Indique la proportion de la variance de la variable cible qui est prévisible à partir des variables indépendantes. 
            - Il varie de 0 à 1, où 1 indique que le modèle explique parfaitement la variabilité des données.
        
        ######   
        
        2. **Erreur quadratique moyenne racine (RMSE)** : 
            - C'est la racine carrée de la moyenne des carrés des différences entre les valeurs prédites et les valeurs réelles. 
            - Elle a la même unité que la variable cible, ce qui la rend facilement interprétable. 
            - Elle varie de 0 à 1, où 1 indique que le modèle explique parfaitement la variabilité des données.
            
        Ces métriques sont calculées sur les données d'entraînement (Train) et les données de test (Test).
        
        ######   

        #### Risque d'overfitting
        Il est important de noter que lors de l’évaluation d'un modèle, nous devons être conscient du risque d’overfitting. 
        L’overfitting se produit lorsque le modèle apprend “par cœur” les données d’entraînement, au point qu’il ne peut pas généraliser efficacement face à de nouvelles données. 
        Un modèle overfitted peut sembler offrir une précision élevée lorsqu’il est appliqué aux données d’entraînement (R² Train élevé), 
        mais sa performance sera amoindrie en production sur de nouvelles données (R² Test faible). 
        Par conséquent, il est crucial de surveiller ces 2 métriques.
        """)
    
    st.subheader(" ")
    
########################################################################################################################################################################################################################## 
    st.subheader("3. Visualisation graphique des prédictions")
    
    
    # Sélection de la plage de données
    début, fin = st.slider('Sélectionnez une plage de données', min_value=0, max_value=500, value=(0, 50))
    x_ax = range(len(y_test))[début:fin]
    
    # Création du graphique
    fig, ax = plt.subplots(figsize=(16,6.5))
    
    ax.plot(x_ax, y_test[début:fin], label="Valeur réelle", c="blue", linewidth=1, marker="o", markersize=6)
    ax.plot(x_ax, y_test_pred[début:fin], label="Valeur prédite", c="orange", linewidth=2, marker="o", markersize=6)
    
    ax.set_title("Prédictions avec le modèle : " + str(model_pipeline.named_steps['estimator']).split("(")[0], fontsize=14)
    ax.legend(loc='best')
    
    ax.grid(visible=True, linewidth=0.5)
    
    st.pyplot(fig)

    def convert_to_min_sec(value):
        minutes = int(abs(value))
        seconds = int((abs(value) - minutes) * 60)
        return f"{'-' if value < 0 else ''}{minutes} min {seconds} sec"
    
    st.title(" ")

########################################################################################################################################################################################################################## 
    st.subheader("4. Analyse des résidus")
    st.write(" ")
    
    # Prédiction du modèle
    y_test_pred_residu = model_pipeline.predict(X_test)
    
    # Calcul des résidus
    residus = y_test_pred_residu - y_test
    
    # Calcul des quantiles
    quantiles = np.quantile(residus, [0.10, 0.90])
    
    sns.relplot(x=y_test, y=residus, alpha=0.05, height=5, aspect=50/22)
    plt.plot([0, y_test.max()], [0, 0], 'r--')
    plt.plot([0, y_test.max()], [quantiles[0], quantiles[0]], 'y--', label="80% des résidus présents dans cet interquartile")
    plt.plot([0, y_test.max()], [quantiles[1], quantiles[1]], 'y--')
    
    # Ajout des annotations
    plt.text(y_test.max(), quantiles[0], f'{quantiles[0]:.2f}', verticalalignment='bottom', horizontalalignment='right')
    plt.text(y_test.max(), quantiles[1], f'{quantiles[1]:.2f}', verticalalignment='bottom', horizontalalignment='right')
    
    plt.xlabel("y_test")
    plt.ylabel("test_résidus")
    plt.legend()
    st.pyplot()
    
    with st.expander("Explications",expanded=True):
        st.write("""
        Ce graphique est une représentation visuelle de l'analyse des résidus. Voici quelques points clés :
        - La concentration dense de points bleus indique la distribution des résidus.
        - La ligne pointillée rouge représente le point de référence où il n’y a aucun résidu
        - Les lignes jaunes représentent l'interquatile de 80% des résidus 
        """)
        st.write(f"""Avec votre modèle, dans 80% des cas, l'erreur des prédictions se situe entre {convert_to_min_sec(quantiles[0])} et {convert_to_min_sec(quantiles[1])}.
                 """)
        
    st.title(" ")

########################################################################################################################################################################################################################## 
    st.subheader("5. Interprétabilité de votre modèle - Features importances")
    st.write(" ")
    
    
    # Calcul des importances des variables explicatives à l'aide de la permutation
    feature_importances = permutation_importance(model_pipeline, X_test, y_test, n_repeats=1, random_state=42,scoring="r2")
    
    # Création d'un DataFrame avec les noms des variables explicatives et leurs importances
    importances_df = pd.DataFrame({
        "Features": X.columns,
        "Importance": feature_importances.importances_mean
    })
    
    # Tri des variables explicatives par ordre d'importance croissante
    importances_df = importances_df.sort_values(by="Importance", ascending=True).tail(10)
    
    # Tracé d'un graphique à barres horizontales
    fig, ax = plt.subplots(figsize=(16,8))
    plt.grid(visible=True, linewidth=0.5)
    plt.barh(y=importances_df["Features"], width=importances_df["Importance"], color="skyblue")
    plt.xlabel("Average impact on model output")
    plt.ylabel("Features")
    
    # Augmentation de la taille des étiquettes des axes
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel("Average impact on model output", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.title(str(model_pipeline["estimator"]).split("(")[0], fontsize=16)
    
    st.pyplot(plt)
    with st.expander("Explications",expanded=True):
        st.write("""
        Ce graphique illustre comment les différentes variables affectent les prédictions du modèle. Plus la barre est longue, plus l’impact de cette caractéristique est important. 
        
        Vous pouvez y voir les dix variables les plus influentes et leur effet sur les prédictions.
        """)

    st.title(" ")
    st.write("---")
    
########################################################################################################################################################################################################################## 
    st.subheader("6. [Modèle retenu] - XGBRegressor")    
    st.write(" ")

    image_shap = Image.open('processus_modelisation.png')
    st.image(image_shap,use_column_width=True)
    st.write("Ce modèle a été entraîné sur des données allant de 2015 à 2022.")
    
    st.write("")
    st.write("Voici les performances atteintes :")  
    score = pd.DataFrame([ 0.53, 0.5, 1.27, 1.32], index=['R² Train','R² Test', 'RMSE Train', 'RMSE Test']).T
    st.dataframe(score)
    st.write("")
    
    st.write("Features Importances :")
    image_features = Image.open('model_featuresImportances.png')
    st.image(image_features,use_column_width=True)
    with st.expander("Explications",expanded=True):
         st.write("""
                Quelques points clés :
                - La variable qui a le plus grand impact sur le modèle est "Distance".
                - DeployedFromStationName et WardName ont également un impact significatif, bien que moins que "Distance".
                - Les autres variables comme LongitudeIncident, LatitudeIncident, HourOfCall, etc., ont un impact moindre sur le modèle.
                """)
    st.header("")
    
    st.write("Analyse des résidus :")
    image_residus = Image.open('Analyse_résidus.png')
    st.image(image_residus,use_column_width=True)
    with st.expander("Explications",expanded=True):
         st.write("""
         Dans 80% des cas, l'erreur des prédictions de notre modèle se situe entre -1 min 28 sec et 1 min 22 sec. 
         
         Cela signifie que lorsque le modèle prédit 6 minutes, il existe une probabilité de 80 % que la valeur réelle se situe entre 4 minutes 38 secondes et 7 minutes 28 secondes.
        """)  
    st.header("")
    
    st.write("Interprétabilité globale (impact des variables sur plusieurs prédictions) :")
    image_shap = Image.open('model_Shap.png')
    st.image(image_shap,use_column_width=True)
    with st.expander("Explications",expanded=True):
         st.write("""
             Ce graphique SHAP illustre comment les différentes variables influencent les prédictions d’un modèle. 
             Les points rouges correspondent à des valeurs élevées de la variable, tandis que les points bleus correspondent à des valeurs basses. 
    
             Voici ce que nous pouvons déduire de notre modèle :
              -  Plus la distance est courte, plus le temps de réponse est rapide.
              -  Lorsque la variable SecondPumpArrivingDeployedFromStation a la valeur "No Second pump deployed", le temps de réponse est plus long que s’il y avait un second camion de pompiers déployé.      
        """)  
        
if __name__ == "__main__":
    main()
