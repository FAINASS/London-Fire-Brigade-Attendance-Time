# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Importation des bibliothèques nécessaires
import streamlit as st
from PIL import Image

# Configuration de la page de l'application Streamlit
st.set_page_config(
    layout="wide",  # Layout en mode large
    initial_sidebar_state="expanded",  # Barre latérale initialement déployée
    page_title = "Temps de Réponse de la Brigade des Pompiers de Londres"  # Titre de la page
)

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

# Fonction principale
def main():
    # Ajout du logo à la barre latérale
    add_logo()
    
    # Ouverture et redimensionnement de l'image
    image = Image.open('LFB_illustration.png')
    new_image = image.resize((600, 290))

    # Affichage de l'image
    st.image(new_image,use_column_width=True)
    
    # Ajout de titres en utilisant du HTML
    st.markdown("<h2 style='text-align: center; color: white;'>Temps de Réponse de la Brigade des Pompiers de Londres</h2>",
                unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: white;'>Auteurs: Falonne Landrine MEFOTIE -  Faiz NASSER ALI </h5>",
                unsafe_allow_html=True)
    
# Exécution de la fonction principale
if __name__ == "__main__":
    main()
