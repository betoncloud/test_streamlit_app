# SINGLE CELL: DTW Analog Years from Real CSVs in /data folder
import streamlit as st
import pandas as pd
import numpy as np
from tslearn.metrics import dtw
from sklearn.preprocessing import StandardScaler
import os

st.set_page_config(page_title="SequentialGroup - Analog Years", layout="wide")
st.title("🔍 DTW Analog Year Analysis (by Variable)")
st.markdown("""
Top 5 most similar years based on **ndvi**, **temperature**, and **rainfall** — shown as tables.  
Data loaded from `ndvi.csv`, `temperature.csv`, `rainfall.csv`
""")

try:
       

    df_ndvi = pd.read_csv("ndvi.csv")
    df_temp = pd.read_csv("temperature.csv")
    df_rain = pd.read_csv("rainfall.csv")

        # Standardize column names
    df_ndvi = df_ndvi.rename(columns={'year': 'year', 'month': 'month', 'WoredaCode': 'WoredaCode'})
    df_temp = df_temp.rename(columns={'Year': 'year', 'Month': 'month', 'WoredaCode': 'WoredaCode'})
    df_rain = df_rain.rename(columns={'year': 'year', 'month': 'month', 'WoredaCode': 'WoredaCode'})

        # Select and merge
    df = pd.merge(df_ndvi[['ymw',  'WoredaCode', 'ndvi']],
                      df_temp[['ymw',  'WoredaCode', 'Avg Temp']],
                      on=['ymw'], how='outer')

    df = pd.merge(df,
                      df_rain[['ymw', 'WoredaCode', 'rainfall']],
                      on=['ymw'], how='outer')

     
    df.head()
    

except Exception as e:
        st.error(f"❌ Error loading or merging data: {e}")
        st.stop()
