# streamlit_analog_year.py
# Run with: streamlit run streamlit_analog_year.py

import streamlit as st
import pandas as pd
import numpy as np
from tslearn.metrics import dtw
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="🌍 Analog Year Finder", layout="wide")
st.title("🌾 DTW-Based Analog Year Analysis")
st.markdown("""
Find climatically similar years using **NDVI, Temperature, and Rainfall**.
Supports historical and future target years.
""")

# -------------------------------
# Step 1: Load & Merge Data from CSVs
# -------------------------------
@st.cache_data
def load_and_merge_data():
    """
    Replace these paths with your actual file paths.
    This function simulates loading if files don't exist.
    """
    try:
        # 🔽 UNCOMMENT BELOW AND UPDATE PATHS WHEN YOU HAVE REAL FILES
        # df_ndvi = pd.read_csv("NDVI.csv")
        # df_temp = pd.read_csv("Temperature.csv")
        # df_rain = pd.read_csv("Rainfall.csv")

        # Simulate realistic data (REMOVE THIS BLOCK when using real files)
        np.random.seed(42)
        woredas = [1001, 1002, 1003, 1004]
        years = list(range(2000, 2024))
        months = list(range(1, 13))

        data = []
        for w in woredas:
            lat = 8 + np.random.rand() * 6   # Ethiopia: ~3–15°N
            lon = 35 + np.random.rand() * 7  # ~33–48°E
            for y in years:
                for m in months:
                    ndvi = 0.3 + 0.4 * np.sin(m * np.pi / 6) + np.random.normal(0, 0.05)
                    temp = 20 + 5 * np.sin(m * np.pi / 6) + np.random.normal(0, 1)
                    rain = max(0, 80 * np.cos(m * np.pi / 6) + np.random.normal(0, 20))
                    data.append([
                        f"{y}{m:02d}1", y, m, w,
                        ndvi, 0, 0, 0, {},  # QA_PIXEL etc.
                        temp, temp - 5, temp + 5,
                        lat, lon,
                        f"{y}-{m:02d}-15", rain
                    ])

        df = pd.DataFrame(data, columns=[
            'ymw', 'year', 'month', 'WoredaCode',
            'NDVI', 'QA_PIXEL', 'vim', 'viq', '.geo',
            'Avg Temp', 'Min Temp', 'Max Temp',
            'Latitude', 'Longitude',
            'Date', 'Rainfall'
        ])
        return df[
            ['year', 'month', 'WoredaCode', 'NDVI', 'Avg Temp', 'Rainfall', 'Latitude', 'Longitude']
        ].copy()

    except Exception as e:
        st.warning(f"Using simulated data. Add NDVI.csv, Temperature.csv, Rainfall.csv to use real data. Error: {e}")
        return load_and_merge_data()  # retry with simulation


df = load_and_merge_data()

# -------------------------------
# Extract Centroids for Mapping
# -------------------------------
@st.cache_data
def get_centroids(_df):
    return _df.groupby('WoredaCode')[['Latitude', 'Longitude']].mean().reset_index()

centroids = get_centroids(df)

# -------------------------------
# User Inputs Sidebar
# -------------------------------
st.sidebar.header("🔍 Select Parameters")
woreda_list = sorted(df['WoredaCode'].unique())
selected_woreda = st.sidebar.selectbox("Select Woreda Code:", woreda_list)

# Allow future years
all_years = sorted(df['year'].unique())
min_year, max_year = int(all_years[0]), int(all_years[-1])

target_year = st.sidebar.number_input(
    "🎯 Target Year (historical or future)",
    min_value=min_year - 10,
    max_value=2050,
    value=2023,
    step=1
)

analyze_button = st.sidebar.button("🔍 Find Analog Years")

# -------------------------------
# Show Map of Woredas
# -------------------------------
if len(centroids) > 0:
    fig_map = px.scatter_mapbox(
        centroids,
        lat='Latitude',
        lon='Longitude',
        hover_name='WoredaCode',
        zoom=5,
        center={"lat": 9.5, "lon": 38.5},  # Center on Ethiopia
        title="📍 Woreda Locations",
        height=300
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    st.sidebar.plotly_chart(fig_map, use_container_width=True)

# -------------------------------
# Build Monthly Multivariate Sequence per Year
# -------------------------------
def build_yearly_matrix(df_grouped, year_range):
    matrix_dict = {}
    for yr in year_range:
        subset = df_grouped[df_grouped['year'] == yr].sort_values('month')
        if len(subset) < 12:
            placeholder = pd.DataFrame({'month': range(1, 13)})
            subset = placeholder.merge(subset, on='month', how='left')

        # Interpolate and fill missing values
        for col in ['NDVI', 'Avg Temp', 'Rainfall']:
            if col in subset.columns:
                subset[col] = subset[col].interpolate(method='linear').ffill().bfill()
        subset[['NDVI', 'Avg Temp', 'Rainfall']] = subset[['NDVI', 'Avg Temp', 'Rainfall']].fillna(0)

        seq = subset[['NDVI', 'Avg Temp', 'Rainfall']].values  # (12, 3)
        if seq.shape == (12, 3):
            matrix_dict[yr] = seq
    return matrix_dict

# -------------------------------
# Extrapolate Future Year Using Linear Trend
# -------------------------------
def extrapolate_future_year(df_grouped, target_yr, lookback=5):
    recent_years = [y for y in range(target_yr - lookback, target_yr) if y in df_grouped['year'].values]
    if len(recent_years) < 2:
        raise ValueError("Not enough past data to extrapolate.")

    matrices = []
    for yr in recent_years:
        s = df_grouped[df_grouped['year'] == yr].sort_values('month')
        s = pd.DataFrame({'month': range(1, 13)}).merge(s, on='month', how='left')
        for col in ['NDVI', 'Avg Temp', 'Rainfall']:
            s[col] = s[col].interpolate().ffill().bfill()
        matrices.append(s[['NDVI', 'Avg Temp', 'Rainfall']].values)

    mats = np.array(matrices)  # (T, 12, 3)
    trends = np.diff(mats, axis=0).mean(axis=0)  # (12, 3)
    base = mats[-1]
    steps = target_yr - recent_years[-1]
    pred = base + trends * steps
    return pred

# -------------------------------
# Main Execution
# -------------------------------
if analyze_button:
    df_w = df[df['WoredaCode'] == selected_woreda].copy()

    if df_w.empty:
        st.error("❌ No data found for selected Woreda.")
    else:
        available_years = df_w['year'].unique()
        ref_years = [y for y in available_years if y != target_year]
        if not ref_years:
            st.error("Not enough other years to compare.")
        else:
            # Build reference sequences
            ref_sequences = build_yearly_matrix(df_w, ref_years)

            if not ref_sequences:
                st.error("❌ Could not build time series for reference years (missing too much data).")
            else:
                # Scale all variables together
                scaler = StandardScaler()
                all_data = np.vstack(list(ref_sequences.values()))  # (N*12, 3)
                scaler.fit(all_data)

                scaled_ref_seqs = {yr: scaler.transform(seq) for yr, seq in ref_sequences.items()}

                # Handle target year
                if target_year in available_years:
                    target_seq = ref_sequences[target_year]
                    st.info(f"📊 Using observed data for target year {target_year}.")
                else:
                    try:
                        target_seq = extrapolate_future_year(df_w, target_year)
                        st.info(f"🔮 Target year {target_year} is not in dataset. Using trend-based extrapolation.")
                    except Exception as e:
                        st.error(f"❌ Cannot extrapolate target year: {e}")
                        st.stop()

                scaled_target = scaler.transform(target_seq)

                # Compute DTW distances
                dtw_distances = {}
                for yr, seq in scaled_ref_seqs.items():
                    dist = dtw(scaled_target, seq)
                    dtw_distances[yr] = dist

                # Rank top 5 most similar (lowest DTW)
                sorted_dists = sorted(dtw_distances.items(), key=lambda x: x[1])
                top_5 = sorted_dists[:5]

                top_years = [t[0] for t in top_5]
                top_scores = [t[1] for t in top_5]

                # Invert scores for visual appeal (higher bar = more similar)
                inv_scores = [max(top_scores) + 1 - s for s in top_scores]

                # -------------------------------
                # Bar Chart: Top 5 Analog Years
                # -------------------------------
                fig_bar = go.Figure(go.Bar(
                    x=[str(y) for y in top_years],
                    y=inv_scores,
                    text=[f"DTW: {s:.2f}" for s in top_scores],
                    textposition="auto",
                    marker=dict(color=inv_scores, colorscale="Blues", showscale=False)
                ))
                fig_bar.update_layout(
                    title=f"🏆 Top 5 Analog Years to {target_year} (Woreda {selected_woreda})",
                    xaxis_title="Year",
                    yaxis_title="Similarity (Inverted DTW Distance)",
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # -------------------------------
                # Line Plot: Pattern Comparison
                # -------------------------------
                st.subheader("📈 Seasonal Pattern Comparison")

                months = np.arange(1, 13)
                lines_df = pd.DataFrame()

                # Add target year
                tgt_df = pd.DataFrame({
                    'Month': months,
                    'Value': scaled_target.mean(axis=1),
                    'Year': str(target_year)
                })
                lines_df = pd.concat([lines_df, tgt_df], ignore_index=True)

                # Add top analogs
                for yr in top_years:
                    seq = scaled_ref_seqs[yr]
                    tmp_df = pd.DataFrame({
                        'Month': months,
                        'Value': seq.mean(axis=1),
                        'Year': str(yr)
                    })
                    lines_df = pd.concat([lines_df, tmp_df], ignore_index=True)

                fig_line = px.line(
                    lines_df,
                    x='Month',
                    y='Value',
                    color='Year',
                    title="Monthly Composite Pattern (Scaled NDVI + Temp + Rainfall)",
                    labels={'Value': 'Average Scaled Value'}
                )
                fig_line.add_vrect(x0=1, x1=12, col=None, row=None, annotation_text="Month", fillcolor="gray", opacity=0.1)
                st.plotly_chart(fig_line, use_container_width=True)

                # -------------------------------
                # Raw DTW Table
                # -------------------------------
                st.write("### 📊 Raw DTW Distances (Lower = More Similar)")
                result_df = pd.DataFrame(top_5, columns=['Analog Year', 'DTW Distance'])
                result_df['Rank'] = result_df.index + 1
                result_df = result_df[['Rank', 'Analog Year', 'DTW Distance']]
                st.dataframe(result_df.style.format({"DTW Distance": "{:.3f}"}), use_container_width=True)

                # -------------------------------
                # Download Button
                # -------------------------------
                @st.experimental_memo
                def convert_df(csv_df):
                    return csv_df.to_csv(index=False).encode('utf-8')

                csv = convert_df(result_df)
                st.download_button(
                    label="💾 Download Results as CSV",
                    data=csv,
                    file_name=f"analog_years_{selected_woreda}_{target_year}.csv",
                    mime="text/csv"
                )

else:
    st.info("👈 Please select a Woreda and Target Year, then click **'Find Analog Years'**.")
    st.image("https://i.imgur.com/ZYHX2rY.gif", width=300, caption="Example NDVI seasonal cycle")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, tslearn, Plotly • Dynamic Time Warping for Climate Analogy")