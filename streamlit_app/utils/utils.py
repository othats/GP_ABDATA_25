import pandas as pd
import os
from catboost import CatBoostRegressor
import streamlit as st

data_path = "data/"
current_path = 'streamlit_app/'

@st.cache_resource
def load_data(path):
    return pd.read_csv(path, parse_dates=["FECHA"])

@st.cache_resource
def load_catboost_model(path):
    model = CatBoostRegressor()
    model.load_model(os.path.join(current_path, path))
    return model

@st.cache_resource
def load_consum_parquet():
    path = os.path.join(data_path, 'parquet/full/consum.parquet')
    return pd.read_parquet(path)
