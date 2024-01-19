import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer
import plotly_express as px
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)
pd.set_option('display.max_columns', None)
st.set_page_config(layout= "wide")
st.title(" :red[Classification And Prediction]")
st.write("")

def datafr():
    df= pd.read_csv("C:/Users/admin/Desktop/Final_Project_1/Classi_predict_data.csv")
    return df

df= datafr()

with st.sidebar:
    select= option_menu("Main Menu", ["Classification and Prediction", "Data Exploration"])
    #logoimage = Image.open("C:/Users/admin/Desktop/Airbnb_Analysis_project/airbnblogo_new.jpg")
    #st.image(logoimage)

if select== "Classification and Prediction":
    tab1,tab2=st.tabs(["***Geological Visualization***","***Data Visualization***"])   

    with tab1:
        st.title("Geological Visualization of Country browsing products")
        data_plt = df
        fig = px.scatter_mapbox(data_plt,
                    lat ="geoNetwork_latitude",
                        lon = "geoNetwork_longitude",
                        hover_data = ["geoNetwork_region"] )
        fig.update_layout(
            mapbox_style = "carto-positron",
            width=800,height=800)
        st.plotly_chart(fig)
    with tab2:
        st.write("Data visualization")         
        data_device = df["device_operatingSystem"].value_counts()
        print(data_device)
        color = sns.color_palette("husl",len(data_device))
        plt.figure(figsize=(14,7))
        data_device.plot(kind="bar",color=color)
        st.pyplot()

           



