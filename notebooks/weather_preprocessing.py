import os

import pandas as pd

data_path = '../data'

df = pd.read_csv(os.path.join(data_path, '2023_MeteoCat_Detall_Estacions.csv'))

df.columns = [c.strip().upper().replace("Ò","O") for c in df.columns]

df = df[["DATA_LECTURA","CODI_ESTACIO","ACRONIM","VALOR"]]

df_wide = df.pivot_table(index=["DATA_LECTURA","CODI_ESTACIO"], values="VALOR", columns = "ACRONIM", aggfunc='mean').reset_index()

df_wide.columns.name = None

df_wide = df_wide.sort_values(["DATA_LECTURA", "CODI_ESTACIO"])

rename_dict = {
    "TM": "Temp_Mean",
    "TX": "Temp_Max",
    "TN": "Temp_Min",
    "HRM": "Humidity_Mean",
    "HRX": "Humidity_Max",
    "HRN": "Humidity_Min",
    "PM": "Pressure_Mean",
    "PX": "Pressure_Max",
    "PN": "Pressure_Min",
    "PPT": "Precipitation",
    "RS24h": "Solar_Radiation_24h",
    "VVM10": "WindSpeed_Mean_10m",
    "DVM10": "WindDir_Mean_10m",
    "VVX10": "WindSpeed_Max_10m",
    "DVVX10": "WindDir_Max_10m"
}

df_wide = df_wide.rename(columns=rename_dict)
df_wide.to_csv(os.path.join(data_path, 'Barcelona_weather_2023_clean.csv'), index=False)

print('File written and exiting program!')