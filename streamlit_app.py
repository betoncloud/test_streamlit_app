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
Data loaded from `data/ndvi.csv`, `data/temperature.csv`, `data/rainfall.csv`
""")

# -------------------------------
# Load Real Data from data/ folder
# -------------------------------
@st.cache_data
def load_data():
    try:
        ndvi_path = os.path.join("data", "ndvi.csv")
        temp_path = os.path.join("data", "temperature.csv")
        rain_path = os.path.join("data", "rainfall.csv")

        if not all(os.path.exists(p) for p in [ndvi_path, temp_path, rain_path]):
            missing = [p for p in [ndvi_path, temp_path, rain_path] if not os.path.exists(p)]
            st.error(f"❌ Missing file(s): {missing}")
            st.stop()

        df_ndvi = pd.read_csv(ndvi_path)
        df_temp = pd.read_csv(temp_path)
        df_rain = pd.read_csv(rain_path)

        # Standardize column names
        df_ndvi = df_ndvi.rename(columns={'year': 'year', 'month': 'month', 'WoredaCode': 'WoredaCode'})
        df_temp = df_temp.rename(columns={'Year': 'year', 'Month': 'month', 'WoredaCode': 'WoredaCode'})
        df_rain = df_rain.rename(columns={'year': 'year', 'month': 'month', 'WoredaCode': 'WoredaCode'})

        # Select and merge
        df = pd.merge(df_ndvi[['year', 'month', 'WoredaCode', 'ndvi']],
                      df_temp[['year', 'month', 'WoredaCode', 'Avg Temp']],
                      on=['year', 'month', 'WoredaCode'], how='outer')

        df = pd.merge(df,
                      df_rain[['year', 'month', 'WoredaCode', 'rainfall']],
                      on=['year', 'month', 'WoredaCode'], how='outer')

        # Convert year/month to int
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['month'] = pd.to_numeric(df['month'], errors='coerce')
        df.dropna(subset=['year', 'month', 'WoredaCode'], inplace=True)
        df['year'] = df['year'].astype(int)
        df['month'] = df['month'].astype(int)

        return df.reset_index(drop=True)

    except Exception as e:
        st.error(f"❌ Error loading or merging data: {e}")
        st.stop()

df = load_data()

# -------------------------------
# User Inputs
# -------------------------------
woreda_list = sorted(df['WoredaCode'].unique())
selected_woreda = st.selectbox("Select Woreda Code:", woreda_list)

all_years = sorted(df['year'].unique())
min_year, max_year = int(all_years[0]), int(all_years[-1])
target_year = st.number_input(
    "Target Year (can be future)",
    min_value=min_year - 10,
    max_value=2050,
    value=min(2023, max_year + 1),
    step=1
)
run_analysis = st.button("🚀 Find Analog Years")

# -------------------------------
# Helper: Build time series {year: array(12)} for a variable
# -------------------------------
def build_series_dict(data, col):
    series_dict = {}
    for yr in data['year'].unique():
        subset = data[data['year'] == yr].sort_values('month')
        if len(subset) < 12:
            placeholder = pd.DataFrame({'month': range(1, 13)})
            merged = placeholder.merge(subset, on='month', how='left')
            merged[col] = pd.to_numeric(merged[col], errors='coerce')
            merged[col] = merged[col].interpolate(method='linear').ffill().bfill()
        else:
            merged = subset.copy()
        seq = merged[col].values[:12]  # Ensure length 12
        if len(seq) == 12 and not np.isnan(seq).all():
            series_dict[yr] = seq
    return series_dict

# -------------------------------
# Extrapolate Future Year Using Linear Trend
# -------------------------------
def extrapolate_var(df_grouped, col, target_yr, lookback=5):
    recent_years = [y for y in range(target_yr - lookback, target_yr) if y in df_grouped['year'].values]
    if len(recent_years) < 2:
        raise ValueError(f"Not enough past data to extrapolate {col}")

    matrices = []
    for y in recent_years:
        s = df_grouped[df_grouped['year'] == y].sort_values('month')[col]
        s = pd.DataFrame({'month': range(1, 13)}).merge(s, on='month', how='left')[col]
        s = s.interpolate().ffill().bfill().values
        if len(s) == 12:
            matrices.append(s)

    if len(matrices) < 2:
        raise ValueError(f"Not enough valid sequences to extrapolate {col}")

    mats = np.array(matrices)
    trend = np.diff(mats, axis=0).mean(axis=0)
    pred = mats[-1] + trend * (target_yr - recent_years[-1])
    return pred

# -------------------------------
# Main Analysis
# -------------------------------
if run_analysis:
    df_w = df[df['WoredaCode'] == selected_woreda].copy()
    available_years = df_w['year'].unique()
    ref_years = [y for y in available_years if y != target_year]

    if len(ref_years) < 1:
        st.warning("Not enough other years in dataset to compare.")
    else:
        for var in ['ndvi', 'Avg Temp', 'rainfall']:
            st.subheader(f"🌾 Analog Years Based on: **{var}**")

            # Build sequence dictionary
            all_seq_dict = build_series_dict(df_w, var)
            ref_seq_dict = {y: seq for y, seq in all_seq_dict.items() if y in ref_years}

            if len(ref_seq_dict) == 0:
                st.warning(f"No complete reference years found for {var}.")
                continue

            # Scale using reference data only
            scaler = StandardScaler()
            all_vals = np.vstack(list(ref_seq_dict.values()))  # (N, 12)
            scaler.fit(all_vals.T.reshape(-1, 1))
            scaled_ref = {y: scaler.transform(seq.reshape(-1, 1)).flatten() for y, seq in ref_seq_dict.items()}

            # Get or extrapolate target
            if target_year in all_seq_dict:
                raw_target = all_seq_dict[target_year]
                status = f"Using observed data for {target_year}"
            else:
                try:
                    raw_target = extrapolate_var(df_w, var, target_year)
                    status = f"🔮 Predicted {var} for {target_year} using trend"
                except Exception as e:
                    st.warning(f"Cannot predict {var}: {e}")
                    continue

            scaled_target = scaler.transform(raw_target.reshape(-1, 1)).flatten()

            # Compute DTW distances
            dtw_scores = {}
            for year, seq in scaled_ref.items():
                dist = dtw(scaled_target, seq)
                dtw_scores[year] = round(dist, 3)

            # Sort by similarity (lowest DTW = most similar)
            sorted_analogs = sorted(dtw_scores.items(), key=lambda x: x[1])
            top_5 = sorted_analogs[:5]

            # Create result table
            result_df = pd.DataFrame(top_5, columns=['Analog Year', 'DTW Distance'])
            result_df.index = ['1st', '2nd', '3rd', '4th', '5th']

            # Display
            st.write(f"**{status}**")
            st.dataframe(result_df.style.format({"DTW Distance": "{:.3f}"}), use_container_width=True)

            # Optional: Show pattern comparison
            with st.expander(f"📊 View {var} Pattern Comparison"):
                months = np.arange(1, 13)
                plot_data = pd.DataFrame({'Month': months})
                plot_data[f"{target_year} (Target)"] = scaled_target
                for yr, _ in top_5:
                    plot_data[f"{yr}"] = scaled_ref[yr]
                st.line_chart(plot_data.set_index('Month'))

else:
    st.info("""
    👆 Please:
    1. Select a **Woreda Code**
    2. Enter a **Target Year** (e.g., 2023 or 2026)
    3. Click **'Find Analog Years'**
    
    Make sure you have:
    - `data/ndvi.csv`
    - `data/temperature.csv`
    - `data/rainfall.csv`
    """)

st.markdown("---")
st.caption("✅ DTW • Table Output • Real CSV Input • No Bar Charts • Streamlit App")