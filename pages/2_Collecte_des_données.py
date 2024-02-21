# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
pd.set_option('display.max_colwidth', None)  # Configuration de pandas pour afficher la totalité du contenu des cellules

# Configuration de la page de l'application Streamlit
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",  
    page_title = "Temps de Réponse de la Brigade des Pompiers de Londres"  

# Fonction pour charger les données
@st.cache_data
def load_data(file):
    data = pd.read_csv(file,nrows=9)  # Lecture des 9 premières lignes du fichier CSV
    return data

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

# Suppression de l'espace vide en haut de la page
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 5rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

def main():
    
    add_logo() 
    
    st.header("📥 Collecte des données")  

    #Création des deux onglets
    titres_onglets = ['Incidents', 'Mobilisations']  
    onglet1, onglet2 = st.tabs(titres_onglets)  
    
    with onglet1:  # Contenu de l'onglet 'Incidents'
        # Description du premier jeu de données
        st.markdown("""
        Le premier jeu de données fourni contient les détails de chaque incident traité depuis janvier 2009. Des informations sont fournies sur la date et le lieu de l'incident ainsi que sur le type d'incident traité.    
        """, unsafe_allow_html=True)
        
        st.write(" ")  
        
        incident = load_data("LFB Incident data.csv")  
        # Affichage du nombre de lignes et de colonnes du DataFrame
        st.write(f"Il comporte 1 580 629 lignes et {incident.shape[1]} colonnes.")
        st.write("")  
        
        # Lien vers la source des données
        st.markdown("[Source des données](https://data.london.gov.uk/dataset/london-fire-brigade-incident-records)")
        st.write(incident)  

 ###################################################################################################################################################################################################################
    
    with onglet2:  # Contenu de l'onglet 'Mobilisations'
        # Description du second jeu de données
        st.markdown("""
        Le second jeu de données contient les détails de chaque camion de pompiers envoyé sur les lieux d'un incident depuis janvier 2009. Des informations sont fournies sur l'appareil mobilisé, son lieu de déploiement et les heures d'arrivée sur les lieux de l'incident.             
        """, unsafe_allow_html=True)

        st.write(" ")  
        
        mobilisation = load_data("LFB Mobilisation data.csv") 
        # Affichage du nombre de lignes et de colonnes du DataFrame
        st.write(f"Il comporte 2 167 042 lignes et {mobilisation.shape[1]} colonnes.")
        st.write(" ")  
        
        # Lien vers la source des données
        st.markdown("[Source des données](https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records)")
        st.write(mobilisation)  
        
# Exécution de la fonction principale
if __name__ == "__main__":
    main()
