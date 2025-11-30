import streamlit as st
import plotly.express as px
from utils.utils import *
from utils.feature_engineering import add_features
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_squared_error
import numpy as np
import plotly.express as px
import pandas as pd

st.title("Consumption Prediction - Without Weather")

X_val, y_val = load_test_data()

df_feat = pd.concat([X_val[col_names], y_val], axis=1, ignore_index=True)
df_feat.columns = col_names + ['CONSUM_DIARI']

model = load_catboost_model("models/catboost_no_weather.cbm")

# features = [c for c in df_feat.columns if c not in ["FECHA", "CONSUM_DIARI"]]
df_feat['PRED'] = model.predict(X_val)

df_feat['FECHA'] = df_feat.apply(lambda x: f"{x['year']}-{x['month']}-{x['day']}", axis=1)

st.subheader("Prediction vs Actual")
fig = px.line(df_feat, x="FECHA", y=["CONSUM_DIARI", "PRED"])
st.plotly_chart(fig, use_container_width=True)

st.subheader("Prediction Data")
st.dataframe(df_feat[["FECHA", "CONSUM_DIARI", "PRED"]].tail())

mae = mean_absolute_error(df_feat["CONSUM_DIARI"], df_feat["PRED"])
rmse = root_mean_squared_error(df_feat["CONSUM_DIARI"], df_feat["PRED"])
r2 = r2_score(df_feat["CONSUM_DIARI"], df_feat["PRED"])
smape_value = smape(df_feat["CONSUM_DIARI"], df_feat['PRED'])

df_feat['abs_error'] = np.abs(df_feat["CONSUM_DIARI"] - df_feat["PRED"])

st.subheader("📊 Model Performance Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", f"{mae:.2f}", help="Average error magnitude – lower is better.")

with col2:
    st.metric("RMSE", f"{rmse:.2f}", help="Penalizes large errors more than MAE.")

with col3:
    st.metric("R²", f"{r2:.3f}", help="Explains how much variance in consumption is captured.")

# Detailed explanation panel
st.markdown("""
### 🔍 What These Metrics Mean

#### **📘 Mean Absolute Error (MAE): {mae:.2f}**
This tells us **how far off, on average**, the model's daily water consumption predictions are from the actual recorded values.  
- A lower MAE means more consistent accuracy.
- MAE treats all errors equally, so it’s good for understanding *typical* prediction gaps.

#### **📗 Root Mean Squared Error (RMSE): {rmse:.2f}**
RMSE also measures error but gives **more weight to large mistakes**.  
- The fact that RMSE is higher than MAE suggests **occasional large spikes in prediction error**, likely due to sudden changes in water consumption that the model didn't anticipate.

#### **📘 R² Score: {r2:.3f}**
An R² near **1.0** means:
- The model explains almost all of the variation in daily water consumption.
- It captures underlying patterns and trends very effectively.

However:
- High R² does *not* guarantee perfect performance day-to-day.
- The disparity between R² and RMSE/MAE indicates **some days behave unpredictably** or that the model may be **slightly overfitting smooth patterns while missing sharp fluctuations**.
""".format(mae=mae, rmse=rmse, r2=r2))

st.subheader("📉 Error Distribution")


fig = px.histogram(
    df_feat,
    x="abs_error",
    nbins=30,
    labels={"abs_error": "Prediction Error"}
)

fig.update_layout(
    xaxis_title="Prediction Error",
    yaxis_title="Frequency",
    bargap=0.1
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Predicted vs Actual Water Consumption with colored Errors")
# threshold = df_feat['error'].abs().mean() + 2 * df_feat['error'].std()
threshold = df_feat["abs_error"].mean() + 2*df_feat["abs_error"].std()

fig = px.scatter(df_feat, x='CONSUM_DIARI', y='PRED',
                color=df_feat["abs_error"] > threshold,
                labels={'color':'High Error'},
                title='Predicted vs Actual Water Consumption',
                trendline='ols')
fig.data = fig.data[0:3]
st.plotly_chart(fig, width="stretch")

top_errors = df_feat.nlargest(10, 'abs_error')

st.subheader("Top 10 Days with Highest Prediction Errors")
fig = px.bar(top_errors, x='FECHA', y='abs_error', 
            labels={'abs_error':'Absolute Error'}, 
            color='abs_error', color_continuous_scale='reds')
st.plotly_chart(fig, width="stretch")