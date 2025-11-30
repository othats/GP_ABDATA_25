import streamlit as st
import plotly.express as px
from utils.utils import load_data, load_catboost_model
from utils.feature_engineering import add_features

st.title("Consumption Prediction – With Weather")

uploaded = st.file_uploader("Upload data for prediction", type=["csv"])

if uploaded:
    df = load_data(uploaded)

    df_feat = add_features(df)

    model = load_catboost_model("models/catboost_weather.cbm")

    features = [c for c in df_feat.columns if c not in ["FECHA", "CONSUM_DIARI"]]
    df_feat["PRED"] = model.predict(df_feat[features])

    st.subheader("Prediction vs Actual")
    fig = px.line(df_feat, x="FECHA", y=["CONSUM_DIARI", "PRED"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction Data")
    st.dataframe(df_feat[["FECHA", "CONSUM_DIARI", "PRED"]].tail())

else:
    st.info("Upload your dataset to run the prediction.")
