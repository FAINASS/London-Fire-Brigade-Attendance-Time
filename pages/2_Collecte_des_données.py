# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import pandas as pd
pd.set_option('display.max_colwidth', None)

#Configurer l'affichage en mode Wide
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title = "Temps de Réponse de la Brigade des Pompiers de Londres")


@st.cache_data
def load_data(file):
    data = pd.read_csv(file,nrows=15)
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
    st.header("📥 Collecte des données")


    titres_onglets = ['Incidents', 'Mobilisations']
    onglet1, onglet2 = st.tabs(titres_onglets)
      
    
    with onglet1:
        st.markdown("""
        Le premier jeu de données fourni contient les détails de chaque incident traité depuis janvier 2009. Des informations sont fournies sur la date et le lieu de l'incident ainsi que sur le type d'incident traité.    
        """, unsafe_allow_html=True)
        
        st.markdown("[Source des données](https://data.london.gov.uk/dataset/london-fire-brigade-incident-records)")
        st.write(" ") 
        st.write("Le dataset des incidents comporte 1 580 629 lignes et {} colonnes.".format(incident.shape[1]))
        
        incident = load_data("LFB Incident data.csv")
        st.write(incident)
       
        
    with onglet2:
        st.markdown("""
        Le second jeu de données contient les détails de chaque camion de pompiers envoyé sur les lieux d'un incident depuis janvier 2009. Des informations sont fournies sur l'appareil mobilisé, son lieu de déploiement et les heures d'arrivée sur les lieux de l'incident.             
        """, unsafe_allow_html=True)
        
        st.markdown("[Source des données](https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records)")

        st.write(" ") 
        st.write("Le dataset des mobilisations comporte 2 167 042 lignes et {} colonnes.".format(mobilisation.shape[1]))
        mobilisation = load_data("LFB Mobilisation data.csv")
        st.write(mobilisation)
       
      

if __name__ == "__main__":
    main()
