import streamlit as st
import pandas as pd
import joblib

model=joblib.load("KNN_heart.pkl")
scaler=joblib.load("scaler.pkl")
expected_columns=joblib.load("columns.pkl")


st.title("Heart Stroke Prediction")
st.markdown("Provide the Following Details")


age=st.slider("Age",18,100,40)
sex=st.selectbox("Sex",["Male","Female"])

chest_pain=st.selectbox("Chest Pain Type",["ATA","NAP","TA","ASY"])

resting_bp=st.number_input("Resting Blood Pressure (mm HG)",80,200,120)
cholesterol=st.number_input("Cholesterol (mg/dL)",100,600,200)
fasting_bs=st.selectbox("Fasting Blood Sugar > 120 mg/dL",[0,1])



