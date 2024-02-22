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

    placeholder.table(st.session_state['incident'].to_frame())

    if st.button('Générer un autre incident'):
        random_index = selected_columns.sample(n=1).index[0]
        st.session_state['incident'] = selected_columns.iloc[random_index]
    
    
        st.write ("")
        st.write ("Pour obtenir la prédiction, il suffit de cliquer sur le bouton 'Prédire' en bas de la page.")
        
        st.markdown("---")
        
        st.subheader("1. Type d'incident")
        col1, col2 = st.columns(2)
        
        IncidentGroupType = sorted(df['IncidentGroupType'].unique().tolist())
        selected_incidents = col1.selectbox("Catégorie d'incident:", IncidentGroupType, index=IncidentGroupType.index(st.session_state['incident']['IncidentGroupType']), disabled=True)
        
        propertyType = sorted(df['PropertyType'].unique().tolist())
        selected_property = col2.selectbox("Type d'emplacement:", propertyType, index=propertyType.index(st.session_state['incident']['PropertyType']))
        
        st.subheader(" ")
        st.subheader("2. Géolocalisation")
        col3, col4, col5 = st.columns(3)
        
        boroughs = sorted(df['BoroughName'].unique().tolist())
        selected_boroughs = col3.selectbox("Arrondissement:", boroughs, index=boroughs.index(st.session_state['incident']['BoroughName']), disabled=True)
        df_filtreBoroughs = df[df['BoroughName'] == selected_boroughs]
        
        wards = sorted(df_filtreBoroughs['WardName'].unique().tolist())
        selected_wards = col4.selectbox("Quartier:", wards, index=wards.index(st.session_state['incident']['WardName']))
        
        station = sorted(df_filtreBoroughs['DeployedFromStationName'].unique().tolist())
        selected_station = col5.selectbox("Première caserne déployée:", station, index=station.index(st.session_state['incident']['DeployedFromStationName']))
            


if __name__ == "__main__":
    main()
