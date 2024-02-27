# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import matplotlib.pyplot as plt


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

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    
    add_logo()
    st.header("🏁 Conclusion")    
    
    st.markdown("""
    Notre objectif était de prédire le temps de réponse des sapeurs-pompiers de Londres à partir des données de 2009 à 2023. Nous avons visé un 
    coefficient de détermination (R²) > à 70% et une erreur quadratique moyenne (RMSE) < à 1.5 minutes.
    """)
    
    st.write(" ")
    st.markdown("""
   Le 9 janvier 2014, dix casernes de pompiers de Londres ont été fermées dans le cadre du Cinquième Plan de Sécurité de Londres (LSP5). 
   Pour éviter toute influence sur nos résultats, nous avons décidé de supprimer toutes les données antérieures à 2014. 
   Ainsi, les modèles seront entraînés uniquement sur les données recueillies entre 2015 et 2022. 
   Nous avons décidé d’exclure les données de 2023, car l’année n’était pas encore terminée au moment de la collecte des données.
    """)
    
    st.write(" ") 
    
    st.header("🎯 Résultats obtenus")
    st.write(" ")
    
    objectif_R = 70
    atteint_R = 50  
    
    objectif_RMSE = 1.5
    atteint_RMSE= 1.32
    
    
    categories = ['']
    valeurs = [atteint_R]
    
    valeurs_RMSE = [atteint_RMSE]
    
    fig, ax = plt.subplots(1,2,figsize=(16,5))
    
    ax[0].bar(categories, valeurs, color='orange')
    ax[0].plot([-0.5, 0.5], [objectif_R, objectif_R], color='green')
    ax[0].set_ylim([0, 100])
    ax[0].set_title("Coefficient de détermination (R²)")
    # Ajout des annotations
    ax[0].text(0, atteint_R/2, str(atteint_R)+"%", ha='center', va='bottom', color='white',size=14)
    ax[0].text(0, objectif_R, f'Objectif: {objectif_R}%', ha='center', va='bottom', color='green')
    
    ax[1].bar(categories, valeurs_RMSE, color='green')
    ax[1].plot([-0.5, 0.5], [objectif_RMSE, objectif_RMSE], color='green')
    ax[1].set_title("Erreur quadratique moyenne (RMSE)")
    # Ajout des annotations
    ax[1].text(0, atteint_RMSE/2, str(atteint_RMSE)+" minutes", ha='center', va='bottom', color='white',size=14)
    ax[1].text(0, objectif_RMSE, f'Objectif: {objectif_RMSE} minutes', ha='center', va='bottom', color='green')
    
    st.pyplot(fig)
    
    st.write(" ")
    
    # R²
    st.markdown("#### - Coefficient de détermination (R²)")
    st.write("""
    Notre modèle peut expliquer environ 50% des variations dans le temps de réponse. 
    Cela signifie que près de la moitié des facteurs qui influencent le temps de réponse sont pris en compte dans notre modèle.
    """)
    
    # RMSE
    st.markdown("#### - Erreur quadratique moyenne (RMSE)")
    st.write("""
      Quand notre modèle commet une erreur, elle est généralement d’environ 1 minute et 19 secondes (1.32 minutes).
      Donc, si la prédiction est que les pompiers arriveront en 6 minutes, 
      ils seront sur les lieux au plus tard en 7 minutes et 19 secondes.
    """)
    
    # Conclusion
    st.markdown("#### - Conclusion")
    st.write("""
    En résumé, notre modèle a une performance modérée. 
    Il capture une partie des facteurs qui influencent le temps de réponse, mais il reste encore beaucoup d'informations non expliquées. 
    De plus, l'erreur moyenne est assez importante par rapport au temps de réponse moyen. 
    Il serait donc bénéfique d'explorer d'autres facteurs ou méthodes pour améliorer la précision du modèle.
    """)
    
    st.subheader(" ")
    st.header("🔭 Perspectives")
    st.write(" ")

    st.write("Notre modèle pourrait être amélioré en intégrant des données supplémentaires et en modifiant notre approche.")
    
    # Intégration de données météorologiques
    st.write("""
    - En envisageant d'intégrer de nouvelles données, telles que la température, la pluviométrie, la couverture nuageuse 
    et la visibilité, notre modèle pourrait être en mesure de mieux comprendre certaines relations et ainsi faire des prédictions 
    plus précises.
    """)
    
    # Transformation du problème en un problème de classification
    st.write("""
    - Une autre amélioration potentielle pourrait consister à transformer notre problème en un problème de classification. 
    Par exemple, si notre objectif principal est de déterminer si le temps de réponse des pompiers sera supérieur ou inférieur à 6 minutes,
    un modèle de classification pourrait être plus approprié. Cette approche pourrait également rendre notre modèle plus facile à interpréter. 
    Au lieu de prédire un temps de réponse continu, nous pourrions simplement classer les temps de réponse comme étant soit 
    dans la classe 0 (inférieur à 6 minutes), soit dans la classe 1 (supérieur à 6 minutes).
    """)

    st.write("""
    Cependant, il est important de souligner que chaque modification apportée à notre modèle doit être soigneusement testée et validée 
    pour s'assurer qu'elle améliore réellement les performances.
    """)
    
if __name__ == "__main__":
    main()
