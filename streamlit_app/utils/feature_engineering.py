import pandas as pd

def add_features(df):
    df = df.copy()

    lista_festivos = [
        # 2021
        "2021-01-01",  
        "2021-01-06", 
        "2021-04-02", 
        "2021-04-05",  
        "2021-05-01",  
        "2021-05-24",  
        "2021-06-24",  
        "2021-08-15",  
        "2021-09-11",  
        "2021-09-24",  
        "2021-10-12",  
        "2021-11-01",  
        "2021-12-06",  
        "2021-12-08",  
        "2021-12-25",  
        "2021-12-26",  

        # 2022
        "2022-01-01",  
        "2022-01-06",  
        "2022-04-15",  
        "2022-04-18",  
        "2022-06-06",  
        "2022-06-24",  
        "2022-08-15",  
        "2022-09-24",  
        "2022-10-12",  
        "2022-11-01",  
        "2022-12-06",  
        "2022-12-08",  
        "2022-12-25",  
        "2022-12-26",  

        # 2023
        "2023-01-06",  
        "2023-04-07",  
        "2023-04-10",
        "2023-06-05",  
        "2023-06-24",  
        "2023-08-15",  
        "2023-09-11",  
        "2023-09-25",  
        "2023-10-12",  
        "2023-11-01",  
        "2023-12-06",  
        "2023-12-08",  
        "2023-12-25",  
        "2023-12-26",  

        # 2024
        "2024-01-01",
        "2024-01-06",  
        "2024-03-29",  
        "2024-04-01",  
        "2024-05-01",  
        "2024-05-20",  
        "2024-06-24",  
        "2024-08-15",  
        "2024-09-11",  
        "2024-09-24",  
        "2024-10-12",  
        "2024-11-01",  
        "2024-12-06",  
        "2024-12-25",  
        "2024-12-26",  
    ]
    lista_festivos = pd.to_datetime(lista_festivos)
    
    # Apply same transformations
    df["Temp_App"] = df["Temp_Mean"] + 0.33*df["Humidity_Mean"]/100 - 0.70*df["WindSpeed_Mean_10m"] - 4
    df['month'] = df['FECHA'].dt.month
    df['dayofweek'] = df['FECHA'].dt.dayofweek
    df['day'] = df['FECHA'].dt.day
    df["weekofyear"] = df["FECHA"].dt.isocalendar().week.astype(int)
    df['year'] = df['FECHA'].dt.year
    df['is_weekend'] = df['dayofweek'] >= 5
    df["is_holiday"] = df["FECHA"].isin(lista_festivos).astype(int)
    df["is_summer"] = df["FECHA"].dt.month.isin([6,7,8]).astype(int)

    for lag in [1,3,7,14,30]:
        df[f"CONSUMO_LAG{lag}"] = df["CONSUM_DIARI"].shift(lag)

    for roll in [7,14,30]:
        df[f"CONSUMO_ROLL{roll}"] = df["CONSUM_DIARI"].rolling(window=roll).mean()
        if roll != 14:
            df[f"CONSUMO_STDROLL{roll}"] = df["CONSUM_DIARI"].rolling(window=roll).std()

    df["DIFF1"] = df["CONSUM_DIARI"].diff()
    df["DIFF7"] = df["CONSUM_DIARI"] - df["CONSUMO_ROLL7"]

    df["dry"] = (df["Precipitation"] == 0).astype(int)
    df["dry_streak"] = df["dry"].groupby((df["dry"] != df["dry"].shift()).cumsum()).cumsum()

    return df
