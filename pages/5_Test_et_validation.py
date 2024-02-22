# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import folium
import branca
import time


pd.set_option('display.max_colwidth', None)

#Configurer l'affichage en mode Wide
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title = "Temps de Réponse de la Brigade des Pompiers de Londres")


@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

@st.cache_resource
def load_model():
    loaded_model = load('model.joblib')
    return loaded_model

def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return np.round(res, 2)

## Supprimer l'espace vide en haut de la page
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
    
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    
    add_logo()
    
    st.header("✅ Test et validation")
    st.write("L'objectif de cette étape est d'utiliser notre modèle pour prédire le temps de réponse de la Brigade des Pompiers de Londres.")
    st.write(" ")

    df = load_data("df_Predictions.csv")
    
    st.subheader("0. Incident à prédire")
    selected_columns = df[["IncidentGroupType", "PropertyType", "BoroughName", "WardName", "DeployedFromStationName","Distance",
                           "HourOfCall", "NumStationsWithPumpsAttending","SecondPumpArrivingDeployedFromStation", "AttendanceTime"]]
    
    

    placeholder = st.empty()
    
    # Vérifier si 'incident' est déjà dans session_state
    if 'incident' not in st.session_state:
        st.session_state['incident'] = selected_columns.iloc[88530]

    if st.button('Générer un autre incident'):
        random_index = selected_columns.sample(n=1).index[0]
        st.session_state['incident'] = selected_columns.iloc[random_index]
        placeholder.table(st.session_state['incident'].to_frame())

    st.write ("")
    st.write ("Pour obtenir la prédiction, il suffit de cliquer sur le bouton 'Prédire' en bas de la page.")
 

if __name__ == "__main__":
    main()
