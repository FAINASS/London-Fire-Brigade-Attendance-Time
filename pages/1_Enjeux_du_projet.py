# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd

# Configuration de la page de l'application Streamlit
st.set_page_config(
    layout="wide",  # Layout en mode large
    initial_sidebar_state="expanded",  # Barre latérale initialement déployée
    page_title = "Temps de Réponse de la Brigade des Pompiers de Londres"  # Titre de la page
)

# Suppression de l'espace vide en haut de la page
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

# Fonction pour ajouter le logo
def add_logo():
    # Utilisation de HTML et CSS pour ajouter une image de fond à la barre latérale
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
    
    add_logo()  # Ajout du logo à la barre latérale
    
    st.header("🚒 La London Fire Brigade")  # Titre de la section

    st.write(" ")  
    
    st.markdown(""" ##### La LFB en quelques chiffres""")  
    
    # Création d'un DataFrame avec des statistiques sur la London Fire Brigade
    df1 = pd.DataFrame({
        'Statistiques': ['Rang mondial', 'Nombre total d\'employés', 'Nombre de sapeurs-pompiers professionnels', 'Nombre de casernes de pompiers'],
        'Valeurs': ['5ème', '5 992', '5 096', '103']
    })
    st.dataframe(df1) 
    st.write("")  
    
    st.write("La LFB s’est fixée deux objectifs majeurs :")  
    
    # Création d'un DataFrame avec les objectifs de la London Fire Brigade
    df2 = pd.DataFrame({
        'Objectifs': ['Arriver sur un lieu d\'un incident', 'Envoyer une seconde équipe en assistance'],
        'Temps': ['6 minutes (360 sec) en moyenne', '8 minutes (480 sec) après le signalement']
    })
    
    st.dataframe(df2) 
    
    st.write(" ") 
    
############################################################################################################################################################################################################################
    
    st.header("🎯 Enjeux du projet")  

    st.write(" ")  
    
    # Description des enjeux du projet
    st.markdown("""
    Notre objectif est de prédire le temps de réponse des sapeurs-pompiers de Londres à partir des données de 2009 à 2023. 
    Nous visons un coefficient de détermination (R²) > à 70% et nous cherchons à obtenir une erreur quadratique moyenne (RMSE) < à 1 minute.
    """, unsafe_allow_html=True)

# Exécution de la fonction principale
if __name__ == "__main__":
    main()
