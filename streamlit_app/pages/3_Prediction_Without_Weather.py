import streamlit as st
import pandas as pd
import plotly.express as px
from utils.utils import load_data, load_catboost_model
from utils.feature_engineering import add_features

st.title("Consumption Prediction – Without Weather")

uploaded = st.file_uploader("Upload data for prediction", type=["csv"])

if uploaded:
    df = load_data(uploaded)
    df_feat = add_features(df)

    model = load_catboost_model("models/catboost_no_weather.cbm")

    weather_cols = [
        'WindDir_Mean_10m','WindDir_Max_10m','Humidity_Mean','Humidity_Min',
        'Humidity_Max','Pressure_Mean','Pressure_Min','Precipitation',
        'Pressure_Max','Solar_Radiation_24h','Temp_Mean','Temp_Min',
        'Temp_Max','WindSpeed_Mean_10m','WindSpeed_Max_10m'
    ]

    features = [c for c in df_feat.columns if c not in ["FECHA", "CONSUM_DIARI"] + weather_cols]

    df_feat["PRED"] = model.predict(df_feat[features])

    st.subheader("Prediction vs Actual")
    fig = px.line(df_feat, x="FECHA", y=["CONSUM_DIARI", "PRED"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction Data")
    st.dataframe(df_feat[["FECHA", "CONSUM_DIARI", "PRED"]].tail())

else:
    st.info("Upload your dataset to run the prediction.")
