# SINGLE CELL: DTW Analog Years for NDVI, Temp, Rainfall Separately
import streamlit as st
import pandas as pd
import numpy as np
from tslearn.metrics import dtw
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="SequentialGroup", layout="wide")
st.title("📈 DTW Analog Years – Separate Variables (NDVI, Temp, Rain)")
st.markdown("Top 5 most similar years based on **NDVI**, **Temperature**, and **Rainfall** separately.")

# -------------------------------
# Simulate Data (Replace with pd.read_csv(...) in production)
# -------------------------------
@st.cache_data
def simulate_data():
    np.random.seed(42)
    woredas = [1001]
    years = list(range(2000, 2024))
    months = list(range(1, 13))
    data = []
    for w in woredas:
        lat, lon = 9.0 + np.random.rand()*2, 38.0 + np.random.rand()*2
        for y in years:
            for m in months:
                # Seasonal + noise
                ndvi = 0.3 + 0.4 * np.sin(m * np.pi / 6) + np.random.normal(0, 0.05)
                temp = 20 + 5 * np.sin(m * np.pi / 6) + np.random.normal(0, 1)
                rain = max(0, 80 * np.cos(m * np.pi / 6) + np.random.normal(0, 20))
                data.append([y, m, w, ndvi, temp, rain, lat, lon])
    df = pd.DataFrame(data, columns=['year','month','WoredaCode','NDVI','Avg Temp','Rainfall','Latitude','Longitude'])
    return df

df = simulate_data()
woreda_list = sorted(df['WoredaCode'].unique())
selected_woreda = st.selectbox("Select Woreda Code:", woreda_list)
target_year = st.number_input("Target Year (can be future)", min_value=1990, max_value=2050, value=2023)
run_analysis = st.button("🔍 Find Analog Years")

# -------------------------------
# Helper: Build monthly sequence per year
# -------------------------------
def build_series_dict(data, col):
    series_dict = {}
    for yr in data['year'].unique():
        subset = data[data['year'] == yr].sort_values('month')
        if len(subset) < 12:
            placeholder = pd.DataFrame({'month': range(1,13)})
            subset = placeholder.merge(subset, on='month', how='left').fillna(method='ffill').fillna(0)
        seq = subset[col].values  # (12,)
        if len(seq) == 12:
            series_dict[yr] = seq
    return series_dict

# -------------------------------
# Extrapolate future year using linear trend
# -------------------------------
def extrapolate_var(df_grouped, col, target_yr, lookback=5):
    recent = [y for y in range(target_yr - lookback, target_yr) if y in df_grouped['year'].values]
    if len(recent) < 2:
        raise ValueError(f"Not enough data to extrapolate {col}")
    mats = np.array([df_grouped[df_grouped['year']==y].sort_values('month')[col].values 
                     for y in recent])  # (T, 12)
    trend = np.diff(mats, axis=0).mean(axis=0)
    pred = mats[-1] + trend * (target_yr - recent[-1])
    return pred

# -------------------------------
# Main Logic
# -------------------------------
if run_analysis:
    df_w = df[df['WoredaCode'] == selected_woreda].copy()
    available_years = df_w['year'].unique()

    if len(available_years) < 2:
        st.error("Not enough data.")
    else:
        ref_years = [y for y in available_years if y != target_year]
        if not ref_years:
            st.warning("Only one year available.")
            ref_years = [y for y in available_years if y != target_year]

        results = {}

        for var in ['NDVI', 'Avg Temp', 'Rainfall']:
            st.subheader(f"🌾 Top 5 Analog Years Based on {var}")

            # Get all sequences
            seq_dict = build_series_dict(df_w, var)

            # Remove target if used later
            ref_seq_dict = {y: seq for y, seq in seq_dict.items() if y in ref_years}

            # Scale
            scaler = StandardScaler()
            all_vals = np.vstack(list(ref_seq_dict.values()))  # (N, 12) → (N*1, 12)
            scaler.fit(all_vals.T.reshape(-1, 1))  # scale over all values

            scaled_ref = {y: scaler.transform(seq.reshape(-1, 1)).flatten() for y, seq in ref_seq_dict.items()}

            # Target sequence
            if target_year in seq_dict:
                target_seq_raw = seq_dict[target_year]
                st.caption(f"Using observed {var} data for {target_year}.")
            else:
                try:
                    target_seq_raw = extrapolate_var(df_w, var, target_year)
                    st.caption(f"🔮 {var}: Predicted for {target_year} using trend.")
                except:
                    st.error(f"❌ Cannot predict {var} for {target_year}")
                    continue

            scaled_target = scaler.transform(target_seq_raw.reshape(-1, 1)).flatten()

            # Compute DTW distances
            dtw_scores = {}
            for yr, seq in scaled_ref.items():
                dist = dtw(scaled_target, seq)
                dtw_scores[yr] = dist

            # Sort by similarity: lowest DTW = most similar
            sorted_scores = sorted(dtw_scores.items(), key=lambda x: x[1])
            top_5 = sorted_scores[:5]
            top_years = [t[0] for t in top_5]
            dtw_dists = [t[1] for t in top_5]

            # Invert distance for bar height: higher bar = more similar
            max_dist = max(dtw_dists) + 1e-6
            inverted_sim = [max_dist - d for d in dtw_dists]

            # Create bar chart
            fig = go.Figure(go.Bar(
                x=[str(y) for y in top_years],
                y=inverted_sim,
                text=[f"DTW: {d:.2f}" for d in dtw_dists],
                textposition="auto",
                marker_color="lightseagreen"
            ))
            fig.update_layout(
                title=f"Top 5 Analog Years — {var}",
                xaxis_title="Year",
                yaxis_title="Similarity Score (Inverted DTW)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Select Woreda and Target Year, then click 'Find Analog Years'.")

st.markdown("---")
st.caption("✅ Dynamic Time Warping • Separate Variable Analysis • Streamlit App")