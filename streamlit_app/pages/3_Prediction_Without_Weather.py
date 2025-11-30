import streamlit as st
import pandas as pd
import plotly.express as px
from utils.utils import load_data, load_catboost_model
from utils.feature_engineering import add_features

import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


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


    # Errors
    df_feat['error'] = df_feat['PRED'] - df_feat['CONSUM_DIARI']
    df_feat['abs_error'] = df_feat['error'].abs()

    # Metrics
    mae = mean_absolute_error(df_feat["CONSUM_DIARI"], df_feat["PRED"])
    rmse = root_mean_squared_error(df_feat["CONSUM_DIARI"], df_feat["PRED"])
    r2 = r2_score(df_feat["CONSUM_DIARI"], df_feat["PRED"])

    st.subheader("Model Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")

    st.write("""

             The MAE of {mae:.2f} indicates that on average, the model's daily water consumption predictions deviate from actual values by this amount. 
             The RMSE of {rmse:.2f} shows that larger errors are present, as it penalizes bigger deviations more heavily.

             """.format(mae=mae, rmse=rmse))
    
    st.metric("R²", f"{r2:.2f}")
    
    st.write("""

             The R² is {r2:.2f}, indicating the model captures overall trends quite well. However, the MAE and RMSE show that some days have large deviations, 
             suggesting there may be occasional overfitting or spikes in consumption not captured by the features.

             """.format(r2=r2))
    
    st.subheader("Predicted vs Actual Water Consumption")
    fig = px.scatter(df_feat, x='CONSUM_DIARI', y='PRED',
                 labels={'CONSUM_DIARI': 'Actual', 'PRED': 'Predicted'},
                 title="Predicted vs Actual")

    st.plotly_chart(fig, width="stretch")

    top_errors = df_feat.nlargest(10, 'abs_error')

    st.subheader("Top 10 Days with Highest Errors")
    fig_bar = px.bar(top_errors, x='FECHA', y='abs_error', title="Top Error Days",
                    color='abs_error', color_continuous_scale='reds')
    st.plotly_chart(fig_bar, width="stretch")



else:
    st.info("Upload your dataset to run the prediction.")
