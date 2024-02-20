# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import pandas as pd
import matplotlib.patches as mpatches
pd.set_option('display.max_colwidth', None)
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import plotly.express as px
from scipy.stats import spearmanr
from PIL import Image


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
def load_json():
    london_borough = json.load(open('london_boroughs.json',"r"))
    return london_borough

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

## Supprimer l'espace vide en haut de la page sans affecter les selectbox
st.markdown("""
<style>
    .block-container:not(.stSelectbox) {
        padding-top: 0;  /* Ajustez cette valeur pour supprimer la marge supérieure */
        padding-bottom: 0rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    
    add_logo()
    st.header("📊 Exploratory Data Analysis")
    st.write("L'objectif de cette étape est de comprendre au maximum les données dont on dispose pour définir une stratégie de modélisation.")
    
    st.write(" ")
    
   
    ############################################################################################################################################### 

    df = load_data("df_NettoyageOK.csv")
    

    df["BoroughName"].replace(to_replace =["Kingston upon thames","Waltham forest","Richmond upon thames" ,"Hammersmith and fulham","Kensington and chelsea",
                                                 "Tower hamlets","Barking and dagenham","City of london"],value=["Kingston upon Thames","Waltham Forest",
                                                                                                                 "Richmond upon Thames","Hammersmith and Fulham",
                                                                                                                 "Kensington and Chelsea","Tower Hamlets","Barking and Dagenham","City of London"],inplace=True)
    my_expander = st.sidebar.expander("**FILTRER LES DONNÉES**",expanded=True)
    
    borough = ["Tous"] + sorted(df["BoroughName"].unique().tolist())
    
    selected_boroughs = my_expander.selectbox('Borough Name', borough, index=borough.index("Tous"))
    
    df['DateOfCall'] = pd.to_datetime(df['DateOfCall'])
    
    min_year = df['DateOfCall'].dt.year.min()
    max_year = df['DateOfCall'].dt.year.max()
    
    start_year, end_year = my_expander.slider("Date Of Call", min_year, max_year, (2021, 2022))
    

    if selected_boroughs == "Tous":
        df = df[(df['DateOfCall'].dt.year >= start_year) & (df['DateOfCall'].dt.year <= end_year)]
    else:
        df = df[(df['BoroughName'] == selected_boroughs) & (df['DateOfCall'].dt.year >= start_year) & (df['DateOfCall'].dt.year <= end_year)]
    
    
    st.subheader("0. Préparation des données")
    
    st.write("Après avoir fusionné et réorganisé les deux jeux de données, nous avons obtenu le jeu de données suivant.")
    
    st.write(" ")
    
    if selected_boroughs == "Tous":
        st.markdown(f"<div style='text-align: left; color: black; background-color: #ff9800; padding: 10px; border-radius: 5px;'>⚠️ Vous avez appliqué un filtre sur tous les arrondissements. Les données affichées couvrent la période de {start_year} à {end_year}.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left; color: black; background-color: #ff9800; padding: 10px; border-radius: 5px;'>⚠️ Vous avez appliqué un filtre sur {selected_boroughs}. Les données affichées sont pour la période de {start_year} à {end_year}.</div>", unsafe_allow_html=True)
    
    st.write(" ")
    
    df = df.reset_index(drop=True)
    st.write(df.head(8))

    st.subheader(" ")
    
    ############################################################################################################################################### 
    st.subheader("1. Informations générales")
    
    nombre_de_lignes = df.shape[0]
    nombre_de_variables_numeriques = df.select_dtypes(include=[np.number]).shape[1]
    nombre_de_variables_categorielles = df.select_dtypes(include=['object']).shape[1]
    nombre_de_doublons = df.duplicated().sum()
    nombre_de_valeurs_manquantes = df.isna().sum().sum()
    
    ## Séparateur de milliers avec un espace
    nombre_de_lignes = "{:,}".format(nombre_de_lignes).replace(",", " ")
    
    data = {"Nombre d'incidents": [nombre_de_lignes],
            'Nombre de variables qualitatives': [nombre_de_variables_categorielles],
            'Nombre de variables quantitatives': [nombre_de_variables_numeriques],
            'Nombre de doublons': [nombre_de_doublons],
            'Nombre de valeurs manquantes': [nombre_de_valeurs_manquantes]}
    
    df_stats = pd.DataFrame(data)
    df_stats = df_stats.style.set_properties(**{'text-align': 'left'})
    df_stats_html = df_stats.to_html()
    st.markdown(df_stats_html, unsafe_allow_html=True)
    
    st.title(" ")
    st.write(" ")
    
   ############################################################################################################################################### 
    
    st.subheader("2. Distribution des variables qualitatives (TOP 5)")
    
    def split_int_cat (data):
        int_var = data.select_dtypes(include = np.number)
        cat_var = data.select_dtypes(exclude=np.number)
        return int_var, cat_var

    int_var = split_int_cat(df)[0] # Sélectionne les variables numériques
    cat_var = split_int_cat(df)[1] # Sélectionne les variables catégorielles
    
    
    col_var = list(cat_var.columns[1:])
    col_var.append("DateOfCall")
    
    selected_col = st.selectbox('Sélectionnez une variable',col_var)
    
    fig,ax= plt.subplots(figsize=(16,5))
    
    ax = df[selected_col].value_counts()[:5].plot(kind="bar",color=["skyblue"])
    ax.set_title(selected_col)
    
    if selected_col =="PropertyType" or selected_col =="AddressQualifier" :
        plt.xticks(rotation=25)
    
    else:
        plt.xticks(rotation=0)

    for c in ax.containers:
        labels = [f'{h/df[selected_col].count()*100:0.1f}%' if (h := v.get_height()) > 0 else '' for v in c]
        ax.bar_label(c,labels=labels, label_type='edge')
    
    st.pyplot(fig)
    
    st.header(" ")
    
    
    # Calcul de la fréquence des retards par catégorie
    x = df.DelayCodeDescription.value_counts(normalize=True).drop(["No Delay"]).index
    y = df.DelayCodeDescription.value_counts(normalize=True).drop(["No Delay"]).values
    
    # Création du graphique
    fig = go.Figure(data=[go.Pie(labels=x, values=y, hole=.3, hoverinfo='none')])
    
    # Mise à jour du layout
    fig.update_layout(
        title_text="Répartition des causes de retard (DelayCodeDescription)",
        title_font=dict(color='black'),
        title_x=0.05,
        autosize=False,
        width=800,  
        height=480,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        paper_bgcolor="white",
        legend=dict(font=dict(color='black'))
    )
    
    # Affichage du graphique
    st.plotly_chart(fig,use_container_width=True)
    
    with st.expander("Explications",expanded=True):
        st.write("""
        - En grande partie, les retards sont dus à la circulation routière et aux travaux. 
        """)
        
    st.subheader(" ")
    ############################################################################################################################################### 
    
    st.subheader("3. Distribution des variables quantitatives")
    col_int = int_var.columns
    
    col = st.selectbox('Sélectionnez une variable', col_int,index=0)
    plt.figure(figsize=(16,4))
    meanprops={"marker":"x","markerfacecolor":"red", "markeredgecolor":"black"}
    sns.boxplot(x=df[col], showmeans=True, meanprops=meanprops, color="skyblue")
    
    st.pyplot(plt)
    st.write("La moyenne de la variable {} est {}".format(col,df[col].mean().round(2)))
        
    st.header(" ")
    
    ###############################################################################################################################################
    
    # Calcul Q1 et Q3
    Q1 = df["AttendanceTime"].quantile(0.25)
    Q3 = df["AttendanceTime"].quantile(0.75)
    
    # Calcul IQR (Inter Quartile Range)
    IQR = Q3 - Q1
    
    # Définition des limites pour les outliers
    borne_inf = Q1 - 1.5 * IQR
    borne_sup = Q3 + 1.5 * IQR
    
    # Filtrage des valeurs qui sont dans l'intervalle [borne_inf, borne_sup]
    df = df[(df["AttendanceTime"] >= borne_inf) & (df["AttendanceTime"] <= borne_sup)]   
    
    variable = "AttendanceTime"
    st.markdown(f"<div style='text-align: center; color: black; background-color: antiquewhite; padding: 10px; border-radius: 5px;'>⚠️ Avant de procéder aux analyses suivantes, nous avons éliminé les valeurs aberrantes de la variable {variable}.</div>", unsafe_allow_html=True)    
    
    st.title(" ")
    st.write("")
    
    ############################################################################################################################################### 
    st.markdown("<h5>L’objectif de la LFB est d’assurer l’arrivée des premiers secours en moins de <span style='color: red; font-weight: bold;'>6 minutes</span>. Les graphiques suivants nous permettront de contrôler si cet objectif est atteint.</h5>", unsafe_allow_html=True)
    
    st.write("--------------------------------------------------------------------------------------------------------------")
    st.subheader("4. Répartition des valeurs de la variable cible")
    
   # Création des intervalles
    bins = pd.cut(df["AttendanceTime"], 10)
    
    colors = ['lightcoral' if (interval.right > 6) else 'skyblue' for interval in bins.cat.categories]
    
    fig, ax = plt.subplots(figsize=(16,6))
    ax = bins.value_counts(sort=False).plot(kind='bar', color=colors)
    
    total_percentage = 0
    red_percentage = 0
    
    for c in ax.containers:
        heights = [v.get_height() for v in c]
        percentages = [h/df["AttendanceTime"].count()*100 for h in heights]
        total_percentage += sum(percentages)
        red_percentage += sum(p for p, color in zip(percentages, colors) if color == 'lightcoral')
    
    for c in ax.containers:
        labels = [f'{p:0.1f}%' if (p := h/df["AttendanceTime"].count()*100) > 0 else '' for h in heights]
        ax.bar_label(c, labels=labels, label_type='edge')
    
    plt.xticks(rotation=0)

    plt.title("Répartition des valeurs de la variable AttendanceTime", fontsize=14)

    red_patch = mpatches.Patch(color='lightcoral', label='Valeurs au-delà des objectifs de la LFB')
    plt.legend(handles=[red_patch])
    
    st.pyplot(fig)
    
    st.write(f"Dans près de {red_percentage/total_percentage*100:.2f} % des interventions, la LFB ne parvient pas à respecter son objectif d’une arrivée en moins de 6 minutes.")

    st.subheader(" ")
    ############################################################################################################################################### 
    
    st.subheader("5. Temps de réponse en fonction des variables qualitatives")
    columns = cat_var.drop(["DateOfCall"],axis=1).columns
    
    selected_col = st.selectbox('Sélectionnez une variable',columns,index=11)
    
    font = {
            'color':  'k',
            'weight': 'bold',
            'size': 30,
            }
    
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,6),sharey=True)
    
    if len(df.groupby(selected_col)["AttendanceTime"].median().sort_values()) > 3 :
    
        df.groupby(selected_col)["AttendanceTime"].median().sort_values()[:3].plot(kind="bar",ax=ax1,title= str(selected_col) + " : Plus RAPIDE",rot=5,color=["yellowgreen","lightgreen","palegreen"])
        ax1.bar_label(ax1.containers[0],fmt='%.2f', padding=3)
    
        df.groupby(selected_col)["AttendanceTime"].median().sort_values()[-3:].plot(kind="bar",ax=ax2,title= str(selected_col) +" : Plus LONG",rot=5,color=["pink","indianred","crimson"])
        ax2.bar_label(ax2.containers[0],fmt='%.2f', padding=3)
        
        # Ajout d'une ligne en pointillé rouge pour le seuil supérieur du temps de réponse sur les deux graphiques
        ax1.axhline(y=6, color='r', linestyle='--',  linewidth=1,label='Seuil supérieur')
        ax2.axhline(y=6, color='r', linestyle='--',  linewidth=1,label='Seuil supérieur')
    
    else :
    
        df.groupby(selected_col)["AttendanceTime"].median().sort_values()[:3].plot(kind="bar",ax=ax1,title= str(selected_col) + " : Plus RAPIDE",rot=5,color=["yellowgreen","lightgreen","palegreen"])
        ax1.bar_label(ax1.containers[0],fmt='%.2f', padding=3)
    
        ax2.text(x= 0.25, y=2.5, s="No More data",fontdict=font)
    
    st.pyplot(plt)
    
    ############################################################################################################################################### 
    
    st.header(" ")
    st.write ("Ajout d'une nouvelle variable : SAISON")
    
    df = df.reset_index()
    df["DateOfCall"]= pd.to_datetime(df["DateOfCall"])
    
    Saison = {
       "Spring" : [3,4,5],
        "Summer" : [6,7,8],
        "Autumn" : [9, 10,11],
        "Winter" : [12, 1,2]}
    
    # Fonction pour faire du mapping
    def argcontains(item):
        for i, v in Saison.items():
            if item in v:
                return i
    
    # Extraction du jour et le mois de la colonne "DateOfCall"
    df["DayOfTheCall"] = df["DateOfCall"].dt.dayofweek
    df["MonthOfTheCall"] = df["DateOfCall"].dt.month.astype("O")
    
    # Récupération de la saison
    df["Saison"]= df["MonthOfTheCall"].map(argcontains)
    
    medians = df.groupby('Saison')['AttendanceTime'].median()
    
    highest_season = medians.idxmax()
    lowest_season = medians.idxmin()
    
    palette = {season: 'lightcoral' if season == highest_season else 'limegreen' if season == lowest_season else 'lavender' for season in df['Saison'].unique()}
    
    
    plt.figure(figsize=(16,6))

    for season in df['Saison'].unique():
        subset = df[df['Saison'] == season]
        if season == highest_season or season == lowest_season:
            sns.lineplot(data=subset, x="DayOfTheCall", y="AttendanceTime", hue="Saison", errorbar=None, marker=".", estimator=np.median, palette=palette, linewidth=3)
        else:
            sns.lineplot(data=subset, x="DayOfTheCall", y="AttendanceTime", hue="Saison", errorbar=None, marker=".", estimator=np.median, palette=palette)
    
    #plt.axhline(y=6, color='r', linestyle='--',  linewidth=1,label='Seuil supérieur')
    min_median = medians.min()
    max_median = medians.max()
    plt.ylim(min_median - 1, max_median + 1)
    
    plt.title("AttendanceTime en fonction du JOUR ET SAISON",fontsize=14)
    plt.xticks([0,1,2,3,4,5,6], ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.legend(fontsize=12)
    st.pyplot(plt)
    plt.tight_layout()
    
    ############################################################################################################################################### 
    
    st.header(" ")
    st.write("Ajout d'une nouvelle variable : Moment de la journée")
    
    labels = ["Late Night", "Early Morning", "Morning", "Afternoon", "Early Evening", "Late Evening"]
    df["MomentOfTheDay"] = pd.cut(df["HourOfCall"], bins = [-1, 3, 6, 12, 18, 21, 24], labels = labels,right=True)
    
    medians = df.groupby('MomentOfTheDay')['AttendanceTime'].median()
    
    highest_Hour= medians.idxmax()
    lowest_Hour = medians.idxmin()
    
    palette = {MomentOfTheDay: 'lightcoral' if MomentOfTheDay == highest_Hour else 'limegreen' if MomentOfTheDay == lowest_Hour else 'lavender' for MomentOfTheDay in df['MomentOfTheDay'].unique()}
    
    fig,ax = plt.subplots(figsize=(16,6))
    barplot = sns.barplot(data=df, x="MomentOfTheDay",y="AttendanceTime",estimator='median',palette=palette)
    ax.set_title("AttendanceTime en fonction de MomentOfTheDay",fontsize=14)
    
    for container in barplot.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
    
    plt.axhline(y=6, color='r', linestyle='--',  linewidth=1,label='Seuil supérieur')
    plt.legend()
    st.pyplot(plt)
    plt.tight_layout()
    
    with st.expander("Explications"):
        st.write("""
        - Late Night : 00:00 à 02:59,
        - Early Morning : 03:00 à 05:59,
        - Morning : 06:00 à 11:59,
        - Afternoon :  12:00 à 17:59,
        - Early Evening : 18:00 à 20:59
        - Late Evening : 21:00 à 23:59.
        """)
        
    st.header(" ")
    
    ############################################################################################################################################### 
    
    st.subheader("6. Temps de réponse en fonction de la zone géographique")
    
    st.write(" ")
   
    st.write ("AttendanceTime en fonction de l'arrondissement")
    data = load_data("df_SampleDistanceOK.csv")
    
    data["BoroughName"].replace(to_replace =["Kingston Upon Thames","Waltham forest","Richmond Upon Thames" ,"Hammersmith And Fulham","Kensington And Chelsea",
                                                 "Tower Hamletss","Barking And Dagenham","City Of London"],value=["Kingston upon Thames","Waltham Forest",
                                                                                                                 "Richmond upon Thames","Hammersmith and Fulham","Kensington and Chelsea",
                                                                                                                 "Tower Hamlets","Barking and Dagenham","City of London"],inplace=True)
                                                                                                                                                                                                                             
    data["DateOfCall"] = pd.to_datetime(data["DateOfCall"])
    
    if selected_boroughs == "Tous":
        data = data[(data['DateOfCall'].dt.year >= start_year) & (data['DateOfCall'].dt.year <= end_year)]
    else:
        data = data[(data['BoroughName'] == selected_boroughs) & (data['DateOfCall'].dt.year >= start_year) & (data['DateOfCall'].dt.year <= end_year)]
    
    dataset_carto = data[["BoroughName","AttendanceTime","LatitudeStation","LongitudeStation","LatitudeIncident","LongitudeIncident","DeployedFromStationName"]]
    
    sample_size = min(200000, len(dataset_carto))
    dataset_carto = dataset_carto.sample(sample_size)
    
    london_borough = json.load(open('london_boroughs.json',"r"))
    
    df_carto_Incident = dataset_carto.groupby("BoroughName").agg({"AttendanceTime": np.median,
                                              "LatitudeIncident": np.median,
                                              "LongitudeIncident":np.median}).reset_index()
    
    df_carto_Incident["BoroughName"].replace(to_replace =["Kingston Upon Thames","Waltham Forestt","Richmond Upon Thames" ,"Hammersmith And Fulham","Kensington And Chelsea",
                                                 "Tower Hamletss","Barking And Dagenham","City Of London"],value=["Kingston upon Thames","Waltham Forest",
                                                                                                                 "Richmond upon Thames","Hammersmith and Fulham",
                                                                                                                 "Kensington and Chelsea","Tower Hamlets","Barking and Dagenham","City of London"],inplace=True)
                                                                                                                                                                                                                             # Initialisation d'un dictionnaire vide pour stocker les correspondances entre les noms et les identifiants des arrondissements
    borough_name_map = {}
    
    for feature in london_borough["features"]:
    
      try :
        feature["id"] = feature["properties"]["id"]

        borough_name_map[feature["properties"]["name"]]=feature["id"]
    
      except :
        print("error")
    

    df_carto_Incident["id"]=df_carto_Incident["BoroughName"].apply(lambda x : borough_name_map[x])
    
    fig = px.choropleth_mapbox(df_carto_Incident,locations="id", # Identifiant de l'emplacement
                        geojson=london_borough, # Données géographiques
                        color="AttendanceTime", # Données de couleur
                        mapbox_style="carto-positron", # Style de la carte
                        hover_name="BoroughName", # Nom de la zone survolée
                        color_continuous_scale="Temps",# Échelle de couleur
                        range_color=[3.8, 5.8])
    
    # Mettre à jour la carte pour qu'elle s'adapte aux emplacements des incidents
    fig.update_geos(fitbounds="locations", visible=False)
    
    # Mettre à jour la mise en page de la carte
    fig.update_layout(title='London Borough Attendance Time')
    fig.update_layout(mapbox_zoom=8.7, mapbox_center = {"lat": 51.490065, "lon": -0.119092})
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    

    st.plotly_chart(fig, use_container_width=True)
    
    ###############################################################################################################################################   
    
    if selected_boroughs == "Tous" : 
        st.write(" ")
        st.write ("Ajout d'une nouvelle variable : RÉGION")
        
     
        Region = {
            "Center": ["Camden","Islington","Westminster","Lambeth","Southwark","Kensington and Chelsea","City of London"],
            "East" : ["Hackney","Waltham Forest","Redbridge","Tower Hamlets","Lewisham","Greenwich","Bexley","Barking and Dagenham","Havering","Newham"],
            "North" : ["Barnet","Enfield","Haringey"],
            "South" : ["Wandsworth","Kingston upon Thames","Merton","Sutton","Croydon","Bromley"],
            "West" :["Hammersmith and Fulham","Brent","Ealing","Richmond upon Thames","Hounslow","Harrow","Hillingdon"]
        }
         
        def argcontains(item):
             for i, v in Region.items():
                 if item in v:
                     return i
         
        df["Region"]= df["BoroughName"].map(argcontains)
         
        medians = df.groupby('Region')['AttendanceTime'].median()
         
        highest_region = medians.idxmax()
        lowest_region = medians.idxmin()
         
        palette = {region: 'lightcoral' if region == highest_region else 'limegreen' if region == lowest_region else 'lavender' for region in df['Region'].unique()}
         
        fig,ax = plt.subplots(figsize=(16,6))
        barplot = sns.barplot(data=df, x="Region",y="AttendanceTime",estimator=np.median,palette=palette)
        ax.set_title("AttendanceTime en fonction de la Région",fontsize=14)
        
        for container in barplot.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)
        
        plt.axhline(y=6, color='r', linestyle='--',  linewidth=1,label='Seuil supérieur')
        plt.legend()
        st.pyplot(plt)
        plt.tight_layout()
    
    image = Image.open('carte_londres.png')
    
    expander = st.expander("Explications",expanded=True)
    
    with expander:
         st.image(image, caption='Carte de Londres', width=300,use_column_width=False)
         
    st.header(" ")
    
    ############################################################################################################################################### 
    
    st.subheader("7. Temps de réponse en fonction des variables quantitatives ")
    
    st.write(" ")
    st.write("Analyse des corrélations")
    df.drop(["index"],axis=1,inplace=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    df["MonthOfTheCall"] = df["MonthOfTheCall"].astype(int)
    # Calcul de la corrélation de 'AttendanceTime' avec toutes les autres colonnes
    corr = df.corrwith(df['AttendanceTime'], method = "spearman",numeric_only=True)
    
    corr_df = pd.DataFrame(corr, columns=['AttendanceTime'])
    corr_df = corr_df.sort_values(by='AttendanceTime', ascending=False)
    sns.heatmap(corr_df, cmap="PiYG", cbar=False,annot=True)
    
    st.pyplot(fig)
    
    with st.expander("Explications"):
        st.write("""
         - AttendanceTime : Temps total pour répondre à un appel (= TurnoutTime + TravelTime)
         - TurnoutTime : Temps de préparation et de départ des pompiers.
         - TravelTime : Temps de trajet jusqu’à l’incident.
        """)


    st.header(" ")
    
    ###############################################################################################################################################     
    st.write("Évolution du temps de réponse en fonction de l'année")

    df["DateOfCall"]=pd.to_datetime(df["DateOfCall"])
    df.set_index("DateOfCall",inplace=True)

    fig,ax=plt.subplots(figsize=(16,6),sharex=True)
    ax.set_title("AttendanceTime en fonction de DateOfCall",fontsize=14)
    
    df.loc["2009":"2022"]["AttendanceTime"].resample("M").median().plot(kind="line",label="Median",linewidth=2,marker="o")
    
    # Calculer les valeurs min et max des médianes
    medians = df.loc["2009":"2022"]["AttendanceTime"].resample("M").median()
    min_median = medians.min()
    max_median = medians.max()
    
    # Ajuster les limites de l'axe des y
    plt.ylim(min_median - 1, max_median + 1)
    
    plt.axhline(y=6, color='r', linestyle='--',  linewidth=1,label='Seuil supérieur')

    plt.legend(loc="upper right")
    
    st.pyplot(fig)
    plt.tight_layout()
    
    with st.expander("Explications"):
        st.write("""
        - Janvier 2014 : La LFB ferme dix casernes et retire 14 camions de pompiers.
        """)

    ############################################################################################################################################### 
    
    st.write(" ")
    
    st.write("Ajout d'une nouvelle variable : Distance entre le lieu de l'incident et la Caserne")
    
    data_dist = data.sample(500)
    fig, ax = plt.subplots(figsize=(16,6))
    sns.regplot(data=data_dist, x="Distance", y="AttendanceTime", line_kws={"color":"red"})
    
    # Calcul du coefficient de corrélation
    corr, _ = spearmanr(data_dist["Distance"], data_dist["AttendanceTime"])
    ax.set_title("Coefficient de corrélation = {:.2f}".format(corr), fontsize=14)
    st.pyplot(fig)
    
if __name__ == "__main__":
    main()
