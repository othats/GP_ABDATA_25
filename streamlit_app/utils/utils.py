import pandas as pd
import os
from catboost import CatBoostRegressor
import streamlit as st
import pickle
import numpy as np

data_path = "data/"
current_path = 'streamlit_app/'

weather_col_names = ['WindDir_Mean_10m', 'WindDir_Max_10m', 
    'Humidity_Mean', 'Humidity_Min', 'Humidity_Max', 
    'Pressure_Mean', 'Pressure_Min', 'Precipitation', 
    'Pressure_Max', 'Solar_Radiation_24h', 'Temp_Mean', 
    'Temp_Min', 'Temp_Max', 'Temp_App', 'WindSpeed_Mean_10m', 'WindSpeed_Max_10m', 'dry', 'dry_streak']

col_names = [
    'month',
    'dayofweek',
    'day',
    'weekofyear',
    'year',
    'is_weekend',
    'is_holiday',
    'is_summer',
    'CONSUMO_LAG1',
    'CONSUMO_LAG3',
    'CONSUMO_LAG7',
    'CONSUMO_LAG14',
    'CONSUMO_LAG30',
    'CONSUMO_ROLL7',
    'CONSUMO_STDROLL7',
    'CONSUMO_ROLL14',
    'CONSUMO_ROLL30',
    'CONSUMO_STDROLL30',
    'DIFF1',
    'DIFF7']

@st.cache_resource
def load_test_data():
    with open(os.path.join(data_path, 'consumption_weather_test.pkl'), 'rb') as fp:
        (X_val, y_val) = pickle.load(fp)
    return X_val, y_val

@st.cache_resource
def load_catboost_model(path):
    model = CatBoostRegressor()
    model.load_model(os.path.join(current_path, path))
    return model

@st.cache_resource
def load_consum_parquet():
    path = os.path.join(data_path, 'parquet/full/consum.parquet')
    return pd.read_parquet(path)

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))