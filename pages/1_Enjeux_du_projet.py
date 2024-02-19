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
        
    st.markdown("""
    La brigade des pompiers de Londres est le service d'incendie et de sauvetage le plus actif du Royaume-Uni. 
    
    #### La LFB en quelques chiffres
    """)
    
    df1 = pd.DataFrame({
        'Statistiques': ['Rang mondial', 'Nombre total d\'employés', 'Nombre de sapeurs-pompiers professionnels', 'Nombre de casernes de pompiers'],
        'Valeurs': ['5ème', '5 992', '5 096', '103']
    })
    
    st.dataframe(df1)
    
    st.subheader("Objectifs 2023 - 2029")
    
    st.write("La LFB s’est fixée deux objectifs majeurs :")
    
    df2 = pd.DataFrame({
        'Objectifs': ['Arrivée sur un lieu d\'un incident', 'Envoi d\'une seconde équipe en assistance'],
        'Temps': ['6 minutes (360 sec) en moyenne', '8 minutes (480 sec) après le signalement']
    })
    
    st.dataframe(df2)
    
    st.write(" ")
    st.header("🎯 Enjeux du projet")
        
    st.markdown("""
    Notre objectif est de prédire le temps de réponse des sapeurs-pompiers de Londres à partir des données de 2009 à 2023. 
    Nous visons un coefficient de détermination (R²) supérieur à 70% et nous cherchons à obtenir une erreur quadratique moyenne (RMSE) inférieure à 1 minute.
    """, unsafe_allow_html=True)
        
    st.write(" ")
    st.header("🔍 Démarche utilisée")
        
    etapes = ["Collecte des données", "Exploratory Data Analysis", "Modélisation","Test et validation","Conclusion"]
    emojis = ['📥', '📊', '🏋️', '✅']
        
    timeline = ' ➡️ '.join(f'{emoji} {etape}' for emoji, etape in zip(emojis, etapes))
        
    st.write(timeline)

if __name__ == "__main__":
    main()
