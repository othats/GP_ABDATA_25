import streamlit as st
import plotly.express as px
from utils.utils import load_data, load_catboost_model
from utils.feature_engineering import add_features
import numpy as np
import matplotlib.pyplot as plt
from catboost import Pool
import pandas as pd

st.title("Consumption Prediction - With Weather")

uploaded = st.file_uploader("Upload data for prediction", type=["csv"])

if uploaded:
    df = load_data(uploaded)

    df_feat = add_features(df)

    model = load_catboost_model("models/catboost_weather.cbm")

    features = [c for c in df_feat.columns if c not in ["FECHA", "CONSUM_DIARI"]]
    shap_values = model.get_feature_importance(
        data=Pool(df_feat[features]),
        type="ShapValues"
    )

    # The last column is the expected value, drop it
    shap_vals = shap_values[:, :-1]

    # Mean absolute SHAP per feature
    mean_shap = np.abs(shap_vals).mean(axis=0)

    # Sort features by impact
    sorted_idx = np.argsort(mean_shap)[::-1]

    # Prepare DataFrame for Plotly
    shap_df = (
        pd.DataFrame({
            "feature": features,
            "mean_abs_shap": mean_shap
        })
        .sort_values("mean_abs_shap", ascending=False)
    )

    # Plotly bar chart
    fig = px.bar(
        shap_df,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        title="Mean |SHAP Value| per Feature",
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed"),  # largest on top
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload your dataset to run the prediction.")
