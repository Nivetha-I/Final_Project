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
from sklearn.preprocessing import LabelEncoder


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
    tab1,tab2,tab3,tab4=st.tabs(["***Geological Visualization***","***Data Visualization***","***Data Correlation***","***Calculations***"])   

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
        col1,col2 = st.columns(2)
        col3,col4 = st.columns(2)
        def ploting(col_name):
            data_device = df[col_name].value_counts()
            st.write(data_device)
            color = sns.color_palette("husl",len(data_device))
            plt.figure(figsize=(10,7))
            data_device.plot(kind="bar",color=color)
            st.pyplot()
        with col1:
            st.write("Different Types of Operating System used for browsing")         
            ploting("device_operatingSystem")
        
        with col2:
            st.write("Customers converted")
            ploting("has_converted")
        with col3:
            fig_pie_1= px.pie(data_frame=df, names= "geoNetwork_region", values="avg_session_time", hover_data= "has_converted",
                    width=1000,title="Region where browsers converted to customers",hole=0.2, color_discrete_sequence= px.colors.sequential.Magenta_r)
            st.plotly_chart(fig_pie_1)
        #fig_pie_1      
    with tab3:  
       
        df_new = pd.DataFrame(df)

        lab_enc = LabelEncoder()

        for i in df_new.select_dtypes(["object"]).columns:
            df_new[i] = lab_enc.fit_transform(df_new[i])   

        corr_data = df_new.corr()
        plt.figure(figsize=(52,52))
        sns.heatmap(corr_data,annot=True,cmap="coolwarm")
        st.pyplot()
    with tab4:

        # Load your dataset
        df = pd.read_csv("Classi_predict_data.csv")

        # Create a header for the app title
        st.header("Customer Conversion Predictor")

        # Create multiselect widgets for the input data
        click_count = st.multiselect("Select click count", df["count_hit"].unique())
        session_count = st.multiselect("Select session count", df["count_session"].unique())
        device = st.multiselect("Select device", df["device_deviceCategory"].unique())
                    
        # Create a button to display the result
        button = st.button("Display result")

        # Create a placeholder for the result
        result_placeholder = st.empty()

        # Check if the button is clicked
        if button:
            # Filter the dataset based on the user input
            filtered_df = df[(df["count_hit"].isin(click_count)) & (df["count_session"].isin(session_count)) & (df["device_deviceCategory"].isin(device))]
            
            # Check if the filtered DataFrame is empty
            if filtered_df.empty:
                # Display an error message
                result_placeholder.error("No matching data found. Please try another input.")
            else:
                # Access the value of the target variable 'has_converted'
                result = filtered_df['has_converted'].iloc[0]
                 # Display the result
                result_placeholder.write(result)
                if result ==0:
                    st.write("Oops!, User Not Converted into Customer")
                else:
                    st.write("User Converted into Customer!")    
               
                

