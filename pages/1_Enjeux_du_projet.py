# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import pandas as pd
from PIL import Image


# Configuration de l'affichage en mode Wide
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title = "Temps de Réponse de la Brigade des Pompiers de Londres")

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

    st.subheader(" ")

    image_shap = Image.open('LFB_Chiffres.png')
    st.image(image_shap, width=900)
    
#################################################################################################################################################################################################################
    st.subheader(" ")
    st.header("🎯 Enjeux du projet")

    st.write(" ")
    st.markdown("""
    Notre objectif est de prédire le temps de réponse des sapeurs-pompiers de Londres à partir des données de 2009 à 2023.

    - Nous visons un coefficient de détermination (R²) supérieur à 70%. Cela signifie que notre modèle expliquerait plus de 70% de la variabilité dans les temps de réponse.
    - Nous cherchons à obtenir une erreur quadratique moyenne (RMSE) inférieure à 2 minutes. Cela signifie que notre modèle prédit les temps de réponse avec une précision moyenne de deux minutes.

    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
