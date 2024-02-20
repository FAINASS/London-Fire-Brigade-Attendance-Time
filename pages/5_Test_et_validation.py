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
    
    st.header("✅ Test et validation")
    st.write("L'objectif de cette étape est d'utiliser notre modèle pour prédire le temps de réponse de la Brigade des Pompiers de Londres.")
    st.write(" ")

    df = load_data("df_Predictions.csv")
    
    st.subheader("0. Incident à prédire")
    selected_columns = df[["IncidentGroupType", "PropertyType", "BoroughName", "WardName", "DeployedFromStationName","Distance",
                           "HourOfCall", "NumStationsWithPumpsAttending","SecondPumpArrivingDeployedFromStation", "AttendanceTime"]]
    
    df = df.drop(df[(df['NumStationsWithPumpsAttending'] > 1) & (df['SecondPumpArrivingDeployedFromStation'] != "No Second pump deployed")].index)

    placeholder = st.empty()
    
    # Vérifier si 'incident' est déjà dans session_state
    if 'incident' not in st.session_state:
        st.session_state['incident'] = selected_columns.iloc[88530]
    
    placeholder.table(st.session_state['incident'].to_frame())
    
    if st.button('Générer un autre incident'):
        random_index = selected_columns.sample(n=1).index[0]
        st.session_state['incident'] = selected_columns.iloc[random_index]
        placeholder.table(st.session_state['incident'].to_frame())
        
    st.markdown("---")
    
    st.subheader("1. Type d'incident")
    col1, col2 = st.columns(2) 
    
    IncidentGroupType = sorted(df['IncidentGroupType'].unique().tolist())
    selected_incidents = col1.selectbox("Catégorie d'incident:", IncidentGroupType, index=IncidentGroupType.index(st.session_state['incident']['IncidentGroupType']),disabled=True)
    df_filtreIncidents = df[df['IncidentGroupType'] == selected_incidents]
    
    
    propertyType = sorted(df['PropertyType'].unique().tolist())
    selected_property = col2.selectbox("Type d'emplacement:", propertyType, index=propertyType.index(st.session_state['incident']['PropertyType']),disabled=True)
    
    
    st.subheader(" ")
    st.subheader("2. Géolocalisation")
    col3, col4,col5 = st.columns(3) 
    
    boroughs = sorted(df['BoroughName'].unique().tolist())
    selected_boroughs = col3.selectbox("Arrondissement:", boroughs, index=boroughs.index(st.session_state['incident']['BoroughName']),disabled=True)
    df_filtreBoroughs = df[df['BoroughName'] == selected_boroughs]
    
    wards = sorted(df_filtreBoroughs['WardName'].unique().tolist())
    selected_wards= col4.selectbox("Quartier:", wards, index=wards.index(st.session_state['incident']['WardName']),disabled=True)
    df_filtreWards = df_filtreBoroughs[df_filtreBoroughs['WardName'] == selected_wards]
    
    station = sorted(df_filtreWards['DeployedFromStationName'].unique().tolist())
    selected_station= col5.selectbox("Première caserne déployée:", station, index=station.index(st.session_state['incident']['DeployedFromStationName']))
    
 
    ###############################################################################################################################################

    station_data = df[df['DeployedFromStationName'] == selected_station]
    lat_station = station_data['LatitudeStation'].median()
    lon_station = station_data['LongitudeStation'].median()
    
    ward_data = df[(df['WardName'] == selected_wards) & (df['DeployedFromStationName'] == selected_station)]
    lat_ward = ward_data['LatitudeIncident'].median()
    lon_ward = ward_data['LongitudeIncident'].median()
    
    
    st.title(" ")
    st.markdown("Légende : 🔴 Lieu de l'incident 🔵 Caserne déployée")

    df_filtered = df[df['DeployedFromStationName'].isin(station)]

    distances = df_filtered.apply(lambda row: haversine_distance(lat_ward, lon_ward, row['LatitudeStation'], row['LongitudeStation']), axis=1)
    min_distance_index = distances.idxmin()
    
    nearest_station = df_filtered.loc[min_distance_index, 'DeployedFromStationName']
    
    m = folium.Map(location=[lat_ward, lon_ward], zoom_start=10,zoom_control=False,scrollWheelZoom=False,dragging=False)
    
    folium.Marker(
        location=[lat_ward, lon_ward],
        icon=folium.Icon(color="red"),
    ).add_to(m)
    
    folium.Marker(
        location=[lat_station, lon_station],
        icon=folium.Icon(color="blue"),
    ).add_to(m)
    
    html = branca.element.Figure()
    html.add_child(m)
    
    st.components.v1.html(html.render(), width=950, height=310)
    
    if selected_station != nearest_station:
        st.write(f"La station la plus proche est {nearest_station}.")
        
 
    ###############################################################################################################################################
    
    st.write(" ")
    st.subheader("3. Intervention")
    
    col6,col7,col8 = st.columns(3)

    Hour = list(np.arange(0.0, 24.0, 1.0))
    selectedHour = col6.selectbox("Heure de l'appel:", Hour, index=Hour.index(st.session_state['incident']['HourOfCall']))
     
    NumPump = list(np.arange(1.0,21.0,1.0))
    selected_NumPump = col7.selectbox("Nombre de caserne engagée:", NumPump, index=NumPump.index(st.session_state['incident']['NumStationsWithPumpsAttending']))
    
    if selected_NumPump == 1:
        secondPump = sorted(df['SecondPumpArrivingDeployedFromStation'].unique().tolist())
        selected_secondPump = col8.selectbox("Deuxième caserne déployée:", secondPump, index=secondPump.index("No Second pump deployed"))
    
    else:  
        secondPump = sorted(df['SecondPumpArrivingDeployedFromStation'].unique().tolist())
        selected_secondPump = col8.selectbox("Deuxième caserne déployée:", secondPump, index=secondPump.index(st.session_state['incident']['SecondPumpArrivingDeployedFromStation']))
    ###############################################################################################################################################

    st.write(" ")
    
    if st.button('Prédire'):
        
        # Charger le modèle
        model = load_model()
        
        # Créer un DataFrame avec les données d'entrée
         
        X = pd.DataFrame({
            "IncidentGroupType" : [selected_incidents],
            "BoroughName": [selected_boroughs],
            "WardName" : [selected_wards],
            "HourOfCall": [selectedHour],
            "PropertyType": [selected_property],
            "DeployedFromStationName": [selected_station],
            "NumStationsWithPumpsAttending": [selected_NumPump],
            "LatitudeIncident": [lat_ward],
            "LatitudeStation" : [lat_station],
            "LongitudeIncident": [lon_ward],
            "LongitudeStation" : [lon_station],
            'SecondPumpArrivingDeployedFromStation' : [selected_secondPump]
        })
        
        X['HourOfCall'] = X['HourOfCall'].astype(float)
        X['NumStationsWithPumpsAttending'] = X['NumStationsWithPumpsAttending'].astype(float)
        
        # Ajout de la colonne 'Distance'
        X["Distance"] = X.apply(lambda row: haversine_distance(lat_ward, lon_ward, lat_station,lon_station), axis=1)
        

    try:
        st.header(" ")
        prediction = model.predict(X)[0]
        
        # Créer un emplacement vide
        counter = st.empty()
        # Boucle pour mettre à jour le compteur
        for i in np.arange(0, prediction, 0.1): 
            minutes, secondes = divmod(i * 60, 60)
            counter.markdown(f"<h3 style='text-align: center; color: White;'>Le temps de réponse estimé est {prediction:.2f} soit : <span style='color: Orange;'>{int(minutes)}</span> minute(s) et <span style='color: Orange;'>{int(secondes)}</span> seconde(s).</h3>", unsafe_allow_html=True)
            time.sleep(0.05)
        
    except UnboundLocalError:
        st.write('')
if __name__ == "__main__":
    main()
