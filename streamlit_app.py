# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from tslearn.metrics import dtw
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Analog Year Finder", layout="wide")
st.title("🌍 DTW-Based Analog Year Analysis")
st.markdown("Find climatically similar years using NDVI, Temperature, and Rainfall.")

# -------------------------------
# Load Data (Placeholder - Replace with your actual loading logic)
# -------------------------------
@st.cache_data
def load_data():
    # Simulate sample data (replace this with real CSV reads)
    np.random.seed(42)
    woredas = [1001, 1002, 1003]
    years = list(range(2000, 2024))
    months = list(range(1, 13))
    
    data = []
    for w in woredas:
        for y in years:
            for m in months:
                ndvi = 0.3 + 0.4 * np.sin(m * np.pi / 6) + np.random.normal(0, 0.05)
                temp = 20 + 5 * np.sin(m * np.pi / 6) + np.random.normal(0, 1)
                rain = max(0, 80 * np.cos(m * np.pi / 6) + np.random.normal(0, 20))
                data.append([f"{y}{m:02d}1", y, m, w, ndvi, 0, 0, 0, {}, 
                             temp, temp-5, temp+5, rain])
    
    df_ndvi = pd.DataFrame(data, columns=[
        'ymw', 'year', 'month', 'WoredaCode', 'NDVI', 'QA_PIXEL', 'vim', 'viq', '.geo',
        'Avg Temp', 'Min Temp', 'Max Temp', 'Rainfall'
    ])
    return df_ndvi[['year', 'month', 'WoredaCode', 'NDVI', 'Avg Temp', 'Rainfall']]

# Uncomment below and comment simulation when you have real files
# df_ndvi = pd.read_csv("NDVI.csv")
# df_temp = pd.read_csv("Temperature.csv")
# df_rain = pd.read_csv("Rainfall.csv")
# Then merge them...

df = load_data()

# -------------------------------
# User Inputs
# -------------------------------
st.sidebar.header("Select Parameters")
woreda_list = sorted(df['WoredaCode'].unique())
selected_woreda = st.sidebar.selectbox("Select Woreda Code:", woreda_list)

all_years = sorted(df['year'].unique())
min_year, max_year = int(all_years[0]), int(all_years[-1])

target_year = st.sidebar.number_input(
    "Target Year (can be future)", 
    min_value=min_year - 10, 
    max_value=2050, 
    value=2023,
    step=1
)

analyze_button = st.sidebar.button("Find Analog Years")

# -------------------------------
# Helper: Extract & Build Time Series
# -------------------------------
def build_yearly_matrix(df_sub, year_range):
    """Returns dict of {year: (12, 3) matrix} """
    matrix_dict = {}
    for yr in year_range:
        subset = df_sub[df_sub['year'] == yr].sort_values('month')
        if len(subset) < 12:
            # Pad missing months with NaN (will interpolate later)
            placeholder = pd.DataFrame({'month': range(1,13)})
            subset = placeholder.merge(subset, on='month', how='left')
        
        # Fill missing values via interpolation or forward fill
        subset[['NDVI', 'Avg Temp', 'Rainfall']] = subset[['NDVI', 'Avg Temp', 'Rainfall']].interpolate(method='linear').ffill().bfill()
        
        seq = subset[['NDVI', 'Avg Temp', 'Rainfall']].values  # Shape: (12, 3)
        if seq.shape[0] == 12:
            matrix_dict[yr] = seq
    return matrix_dict

# -------------------------------
# Impute Future Year (if needed)
# -------------------------------
def extrapolate_year(df_sub, target_yr):
    """Simple linear trend extrapolation across last N years"""
    recent_years = range(target_yr - 5, target_yr)
    valid_recent = [y for y in recent_years if y in df_sub['year'].values]
    
    if len(valid_recent) < 2:
        raise ValueError("Not enough past data to extrapolate.")
    
    matrices = []
    for yr in valid_recent:
        s = df_sub[df_sub['year'] == yr].sort_values('month')[['NDVI', 'Avg Temp', 'Rainfall']]
        if len(s) < 12:
            s = pd.DataFrame({'month': range(1,13)}).merge(s, on='month', how='left').interpolate().ffill().bfill()
        matrices.append(s.values)  # (12,3)

    mats = np.array(matrices)  # (T, 12, 3)
    trends = np.diff(mats, axis=0).mean(axis=0)  # Average monthly trend (12,3)
    base = mats[-1]  # Last observed year
    pred = base + trends * (target_yr - valid_recent[-1])  # Extrapolated (12,3)
    return pred

# -------------------------------
# Main Logic
# -------------------------------
if analyze_button:
    with st.spinner(f"Finding analog years for Woreda {selected_woreda}, Target Year {target_year}..."):
        # Filter data for selected woreda
        df_w = df[df['WoredaCode'] == selected_woreda].copy()

        if df_w.empty:
            st.error("No data found for selected Woreda.")
        else:
            # Get available years
            available_years = df_w['year'].unique()
            
            # Prepare reference years (exclude target if not in dataset, but allow it as query)
            ref_years = [y for y in available_years if y != target_year]
            if target_year in available_years:
                ref_years = [y for y in available_years if y != target_year]
            else:
                ref_years = available_years.copy()

            # Build reference sequences
            ref_sequences = build_yearly_matrix(df_w, ref_years)

            if not ref_sequences:
                st.error("Not enough complete years to compare.")
            else:
                # Prepare scaler
                scaler = StandardScaler()
                
                # Collect all reference data for fitting scaler
                all_ref_data = np.vstack(list(ref_sequences.values()))  # (N*12, 3)
                scaler.fit(all_ref_data)

                # Scale all reference sequences
                scaled_ref_seqs = {
                    yr: scaler.transform(seq) for yr, seq in ref_sequences.items()
                }

                # Create or extrapolate target sequence
                if target_year in available_years:
                    target_seq = ref_sequences[target_year]
                else:
                    try:
                        target_seq = extrapolate_year(df_w, target_year)
                        st.info(f"Target year {target_year} is not in dataset. Using linear extrapolation from recent trends.")
                    except Exception as e:
                        st.error(f"Cannot extrapolate target year: {e}")
                        st.stop()

                # Scale target sequence using same scaler
                scaled_target = scaler.transform(target_seq)

                # Compute DTW distances
                dtw_distances = {}
                for yr, seq in scaled_ref_seqs.items():
                    dist = dtw(scaled_target, seq)
                    dtw_distances[yr] = dist

                # Sort by similarity (lowest DTW = most similar)
                sorted_dists = sorted(dtw_distances.items(), key=lambda x: x[1])
                top_5 = sorted_dists[:5]

                top_years = [t[0] for t in top_5]
                top_scores = [t[1] for t in top_5]

                # Invert scores for bar height (higher = more similar), optional
                max_score = max(top_scores) + 1e-6
                inverse_sim = [max_score - s for s in top_scores]  # Just for visual appeal

                # -------------------------------
                # Plot Top 5 Analog Years
                # -------------------------------
                fig_bar = go.Figure(go.Bar(
                    x=top_years,
                    y=inverse_sim,
                    text=[f"{s:.2f}" for s in top_scores],
                    textposition="auto",
                    marker_color="skyblue"
                ))
                fig_bar.update_layout(
                    title=f"Top 5 Analog Years to {target_year} (Woreda {selected_woreda})",
                    xaxis_title="Year",
                    yaxis_title="Similarity Score (inverted DTW)",
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # -------------------------------
                # Optional: Compare Time Series Patterns
                # -------------------------------
                st.subheader("Time Series Comparison")

                # Reconstruct long format for plotting
                months = np.arange(1, 13)
                df_plot = pd.DataFrame()

                # Add target
                tgt_df = pd.DataFrame({
                    'Month': months,
                    'Value': scaled_target.mean(axis=1),  # Combined index
                    'Year': str(target_year),
                    'Variable': 'Composite Index'
                })

                for yr in top_years:
                    seq = scaled_ref_seqs[yr]
                    tmp_df = pd.DataFrame({
                        'Month': months,
                        'Value': seq.mean(axis=1),
                        'Year': str(yr),
                        'Variable': 'Composite Index'
                    })
                    tgt_df = pd.concat([tgt_df, tmp_df], ignore_index=True)

                tgt_df = pd.concat([tgt_df, pd.DataFrame({
                    'Month': months,
                    'Value': scaled_target.mean(axis=1),
                    'Year': str(target_year),
                    'Variable': 'Composite Index'
                })], ignore_index=True)

                fig_line = px.line(tgt_df, x='Month', y='Value', color='Year', 
                                   title="Monthly Composite Pattern Comparison",
                                   labels={'Value': 'Scaled Composite (NDVI + Temp + Rain)'})
                st.plotly_chart(fig_line, use_container_width=True)

                # Show raw DTW scores
                st.write("**Raw DTW Distances (lower = more similar):**")
                score_df = pd.DataFrame(top_5, columns=['Year', 'DTW Distance'])
                st.dataframe(score_df)

else:
    st.info("👈 Please select parameters and click 'Find Analog Years'.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, tslearn, and DTW.")