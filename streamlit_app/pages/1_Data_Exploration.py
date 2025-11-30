import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from utils.utils import load_consum_parquet

st.title("Exploratory Data Analysis - Aggregated Consumption Data")

# -------------------------------
# CACHE LOADING & AGGREGATION
# -------------------------------
@st.cache_data
def load_and_aggregate():
    df = load_consum_parquet()
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")

    # Daily total consumption
    daily_total = df.groupby("FECHA")["CONSUMO_REAL"].sum().reset_index()

    # Daily average by usage type
    daily_usage = df.groupby(["FECHA", "US_AIGUA_GEST"])["CONSUMO_REAL"].mean().reset_index()

    # Daily average by census section
    daily_census = df.groupby(["FECHA", "SECCIO_CENSAL"])["CONSUMO_REAL"].mean().reset_index()

    # Sampled dataset for heavy categorical plots
    sampled = df.sample(n=min(500_000, len(df)), random_state=42)

    return df, daily_total, daily_usage, daily_census, sampled

df, daily_total, daily_usage, daily_census, sampled = load_and_aggregate()

st.write(f"Total rows in raw dataset: {df.shape[0]:,}")

# -------------------------------
# DATA TYPES & MISSING VALUES
# -------------------------------
st.header("Data Types and Missing Values")
dtypes_df = pd.DataFrame({
    "column": df.columns,
    "dtype": df.dtypes.values,
    "missing_count": df.isna().sum().values,
    "missing_percent": (df.isna().mean() * 100).values
})
st.dataframe(dtypes_df)

# -------------------------------
# NUMERIC SUMMARY
# -------------------------------
st.header("Numeric Columns Summary")
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
st.dataframe(df[numeric_cols].describe().T)

# -------------------------------
# CATEGORICAL SUMMARY
# -------------------------------
st.header("Categorical Columns Summary")
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
st.dataframe(df[cat_cols].describe().T)

# -------------------------------
# TIME SERIES ANALYSIS
# -------------------------------
st.header("Daily Consumption Trends")

st.subheader("Total Daily Consumption")
fig = px.line(daily_total, x="FECHA", y="CONSUMO_REAL",
              labels={"CONSUMO_REAL": "Total Consumption (L)"})
st.plotly_chart(fig, use_container_width=True)

st.subheader("Average Daily Consumption by Usage Type")
fig = px.line(daily_usage, x="FECHA", y="CONSUMO_REAL", color="US_AIGUA_GEST",
              labels={"CONSUMO_REAL": "Average Consumption (L)"})
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# DISTRIBUTION ANALYSIS
# -------------------------------
st.header("Consumption Distribution")

st.subheader("Distribution of CONSUMO_REAL (sampled)")
fig = px.histogram(sampled, x="CONSUMO_REAL", nbins=80)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Boxplot by Usage Type (sampled)")
if "US_AIGUA_GEST" in sampled.columns:
    fig = px.box(sampled, x="US_AIGUA_GEST", y="CONSUMO_REAL", points="suspectedoutliers")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# CENSUS SECTION ANALYSIS
# -------------------------------
st.header("Census Section Analysis")

st.subheader("Average Consumption per Census Section (sampled)")
sample_census = sampled.groupby("SECCIO_CENSAL")["CONSUMO_REAL"].mean().reset_index()
fig = px.histogram(sample_census, x="CONSUMO_REAL", nbins=50)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# CORRELATION ANALYSIS
# -------------------------------
st.header("Correlation Heatmap (Numeric Columns)")

if len(numeric_cols) > 1:
    corr = df[numeric_cols].sample(n=min(500_000, len(df)), random_state=42).corr()
    fig, ax = plt.subplots(figsize=(8,4))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    st.pyplot(fig)
else:
    st.write("Not enough numeric columns for correlation analysis.")
