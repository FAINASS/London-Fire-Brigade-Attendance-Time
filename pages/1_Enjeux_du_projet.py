# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st


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

    #### Statistiques

    | Statistiques | Valeurs |
    | --- | --- |
    | Rang mondial | <span style='color:lightblue'>5ème</span> |
    | Nombre total d'employés | <span style='color:lightblue'>5 992</span> |
    | Nombre de sapeurs-pompiers professionnels | <span style='color:lightblue'>5 096</span> |
    | Nombre de casernes de pompiers | <span style='color:lightblue'>103</span> |

    <br>
    Le temps de réponse des casernes lors d’une urgence est un facteur majeur pour la limitation des dégâts à la fois physique et matériel. 

    <br>
    <br>
    #### Objectifs 
    
    La LFB s’est fixée deux objectifs majeurs :
    
    | Objectifs | Temps |
    | --- | --- |
    | Arrivée sur un lieu d'un incident | <span style='color:lightblue'>6 minutes (360 sec)</span> en moyenne |
    | Envoi d'une seconde équipe en assistance | <span style='color:lightblue'>8 minutes (480 sec)</span> après le signalement |
    """, unsafe_allow_html=True)
    
    st.write(" ")
    st.header("🎯 Enjeux du projet")
        
    st.markdown("""
    Notre objectif est de prédire le temps de réponse des sapeurs-pompiers de Londres à partir des données de 2009 à 2023. 
    
    Nous viserons un coefficient de détermination (R²) > à 70% et une erreur quadratique moyenne (RMSE) < à 1 minute.
    """)
        
    st.write(" ")
    st.header("🔍 Démarche utilisée")
        
    etapes = ["Collecte des données", "Exploratory Data Analysis", "Modélisation","Test et validation","Conclusion"]
    emojis = ['📥', '📊', '🏋️', '✅']
        
    timeline = ' ➡️ '.join(f'{emoji} {etape}' for emoji, etape in zip(emojis, etapes))
        
    st.write(timeline)

if __name__ == "__main__":
    main()
