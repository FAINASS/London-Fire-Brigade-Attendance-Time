# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import pandas as pd


#Configurer l'affichage en mode Wide
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title = "Temps de Réponse de la Brigade des Pompiers de Londres")

## Supprimer l'espace vide en haut de la page
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 2rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)


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
    
    st.header("🚒 La London Fire Brigade")

    st.write(" ")
    
    st.markdown("""
    ##### La LFB en quelques chiffres
    """)
    
    df1 = pd.DataFrame({
        'Statistiques': ['Rang mondial', 'Nombre total d\'employés', 'Nombre de sapeurs-pompiers professionnels', 'Nombre de casernes de pompiers'],
        'Valeurs': ['5ème', '5 992', '5 096', '103']
    })
    st.dataframe(df1)
    st.write("")
    
    st.write("La LFB s’est fixée deux objectifs majeurs :")
    
    df2 = pd.DataFrame({
        'Objectifs': ['Arriver sur un lieu d\'un incident', 'Envoyer une seconde équipe en assistance'],
        'Temps': ['6 minutes (360 sec) en moyenne', '8 minutes (480 sec) après le signalement']
    })
    
    st.dataframe(df2)
    
    st.write(" ")
    st.header("🎯 Enjeux du projet")

    st.write(" ")
    st.markdown("""
    # Prédiction du temps de réponse des sapeurs-pompiers de Londres 🚒🔥

    Nous utilisons une décennie de données, de **2009 à 2023**, pour prédire le temps de réponse des sapeurs-pompiers de Londres. Notre objectif est ambitieux mais réalisable :

    - Nous visons un **coefficient de détermination (R²) supérieur à 70%**. Cela signifie que notre modèle expliquerait plus de 70% de la variabilité dans les temps de réponse.
    - Nous cherchons à obtenir une **erreur quadratique moyenne (RMSE) inférieure à 1 minute**. Cela signifie que notre modèle prédit les temps de réponse avec une précision moyenne d'une minute.

    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
