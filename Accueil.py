# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
from PIL import Image


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
    image = Image.open('LFB_illustration.png')
    new_image = image.resize((600, 290))

    st.image(new_image,use_column_width=True)
    
    st.markdown("<h2 style='text-align: center; color: white;'>Temps de Réponse de la Brigade des Pompiers de Londres</h2>",
                unsafe_allow_html=True)
    
    
    st.markdown("<h5 style='text-align: center; color: white;'>Auteurs: Falonne Landrine MEFOTIE -  Faiz NASSER ALI </h5>",
                unsafe_allow_html=True)
    


if __name__ == "__main__":
    main()