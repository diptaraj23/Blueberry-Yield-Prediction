from copyreg import pickle
import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np 
import joblib

data = pd.read_csv("WildBlueberryPollinationSimulationData.csv")
data = data.drop(["Row#"],axis=1)

st.title("------Blueberry Yield Prediction-------")

nav = st.sidebar.radio("Go to:",["About the Project","About the Dataset", "Prediction"])

if (nav == "About the Project"):

    st.header("About the Project")

    text1 = """In this Project we are using a Dataset (refer to \"About the Dataset\" tab) 
    to predict the Blueberry yeild production. The project is inspired from the journal paper
    [Wild blueberry yield prediction using a combination of computer simulation
and machine learning algorithms](https://www.sciencedirect.com/science/article/abs/pii/S016816992031156X)"""
    
    text2 = "We have deployed only a Fine-Tuned Random Forest Model for this purpose."
    
    text3 = """ You can get the whole project on Github at [Blueberry-Yield-Prediction]
    (https://github.com/diptaraj23/Blueberry-Yield-Prediction)"""

    st.write(text1)
    st.text(text2)
    st.write(text3)

if (nav == "About the Dataset"):

    st.header("About the Dataset")

    if st.checkbox("Show the Dataset"):
        st.dataframe(data)

    if st.checkbox("Know about the Data"):
        data_description = data.describe()
        st.table(data_description)
    
    if st.checkbox("Visualize the Data"):
        st.write("Line Chart")
        st.line_chart(data)
        st.write("Area Chart")
        st.area_chart(data)
        st.write("Bar Chart")
        st.bar_chart(data)

    


if (nav == "Prediction"):
    st.header("Predict Blueberry Yield")

    text5 = """After analyzing the data we have kept only the important columns
    for building the model that is necessary for prediction"""

    st.write(text5)

    val_clonesize = st.text_input("clonesize","0")
    val_osmia = st.text_input("osmia","0")
    val_AverageOfUpperTRange = st.text_input("AverageOfUpperTRange","0")
    val_AverageOfLowerTRange = st.text_input("AverageOfLowerTRange","0")
    val_AverageRainingDays = st.text_input("AverageRainingDays","0")
    val_fruitset = st.text_input("fruitset","0")
    val_fruitmass = st.text_input("fruitmass","0")
    val_seeds = st.text_input("seeds","0")
    
    attributes = np.array([float(val_clonesize),
    float(val_osmia),
    float(val_AverageOfUpperTRange),
    float(val_AverageOfLowerTRange),
    float(val_AverageRainingDays),
    float(val_fruitset),
    float(val_fruitmass),
    float(val_seeds)]).reshape(1,-1)

    loaded_model = joblib.load('rf_bbry_tuned_model.pkl')
    pred = loaded_model.predict(attributes)[0]

    if st.button("Predict"):
        st.success(f"Predictied Yield is : {(pred)}")

