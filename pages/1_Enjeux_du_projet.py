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
    
    C'est le <span style='color:lightblue'>cinquième</span> plus grand corps de sapeurs-pompiers dans le monde avec <span style='color:lightblue'>5 992 employés</span> dont <span style='color:lightblue'>5 096 sapeurs-pompiers professionnels</span>.
    
    Créée il y a presque deux siècles, la London Fire Brigade (LFB) est composée de <span style='color:lightblue'>103 casernes de pompiers</span> et d’une brigade fluviale.
    
    Le temps de réponse des casernes lors d’une urgence est un facteur majeur pour la limitation des dégâts à la fois physique et matériel. La LFB s’est donc fixé comme deux objectifs majeurs :
    
    - arriver sur un lieu d'un incident dans un temps inférieur à <span style='color:lightblue'>6 minutes (360 sec)</span> en moyenne
            
    - et d'envoyer une seconde équipe en assistance, si nécessaire, dans les <span style='color:lightblue'> 8 premières minutes (480 sec)</span> après le signalement.
            
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
