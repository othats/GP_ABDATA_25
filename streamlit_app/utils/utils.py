import pandas as pd
import os
from catboost import CatBoostRegressor

data_path = "data/"
current_path = 'streamlit_app/'

def load_data(path):
    return pd.read_csv(path, parse_dates=["FECHA"])

def load_catboost_model(path):
    model = CatBoostRegressor()
    model.load_model(os.path.join(current_path, path))
    return model

def load_consum_parquet():
    path = os.path.join(data_path, 'parquet/full/consum.parquet')
    return pd.read_parquet(path)
