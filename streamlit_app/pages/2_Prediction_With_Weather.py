import streamlit as st
import plotly.express as px
from utils.utils import load_data, load_catboost_model
from utils.feature_engineering import add_features
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import numpy as np
import plotly.express as px



st.title("Consumption Prediction - With Weather")

uploaded = st.file_uploader("Upload data for prediction", type=["csv"])

if uploaded:
    df = load_data(uploaded)

    df_feat = add_features(df)

    model = load_catboost_model("models/catboost_weather.cbm")

    features = [c for c in df_feat.columns if c not in ["FECHA", "CONSUM_DIARI"]]
    df_feat["PRED"] = model.predict(df_feat[features])
    df_feat['error'] = df_feat['PRED'] - df_feat['CONSUM_DIARI']


    st.subheader("Prediction vs Actual")
    fig = px.line(df_feat, x="FECHA", y=["CONSUM_DIARI", "PRED"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction Data")
    st.dataframe(df_feat[["FECHA", "CONSUM_DIARI", "PRED"]].tail())

    mae = mean_absolute_error(df_feat["CONSUM_DIARI"], df_feat["PRED"])
    rmse = root_mean_squared_error(df_feat["CONSUM_DIARI"], df_feat["PRED"])
    r2 = r2_score(df_feat["CONSUM_DIARI"], df_feat["PRED"])


    st.subheader("Metrics")

    st.metric("MAE", f"{mae:.2f}")

    st.metric("RMSE", f"{rmse:.2f}")

    st.write("""

             The MAE of {mae:.2f} indicates that on average, the model's daily water consumption predictions deviate from actual values by this amount. 
             The RMSE of {rmse:.2f} shows that larger errors are present, as it penalizes bigger deviations more heavily.

             """.format(mae=mae, rmse=rmse))

    st.metric("R²", f"{r2:.2f}")


    st.write("""

             The R² is 0.99, indicating the model captures overall trends very well. However, the MAE and RMSE show that some days have large deviations, 
             suggesting there may be occasional overfitting or spikes in consumption not captured by the features.

             """)

    st.subheader("Predicted vs Actual Water Consumption with colored Errors")
    threshold = df_feat['error'].abs().mean() + 2 * df_feat['error'].std()

    fig = px.scatter(df_feat, x='CONSUM_DIARI', y='PRED',
                    color=df_feat['error'].abs() > threshold,
                    labels={'color':'High Error'},
                    title='Predicted vs Actual Water Consumption')
    st.plotly_chart(fig, width="stretch")

    df_feat['abs_error'] = df_feat['error'].abs()
    top_errors = df_feat.nlargest(10, 'abs_error')

    st.subheader("Top 10 Days with Highest Prediction Errors")
    fig = px.bar(top_errors, x='FECHA', y='abs_error', 
                labels={'abs_error':'Absolute Error'}, 
                color='abs_error', color_continuous_scale='reds')
    st.plotly_chart(fig, width="stretch")



else:
    st.info("Upload your dataset to run the prediction.")
