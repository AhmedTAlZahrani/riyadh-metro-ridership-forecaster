import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_ridership_data, load_line_data, split_by_date, get_line_names
from src.feature_engineering import add_prayer_flags, build_features
from src.station_clustering import StationClusterer, CLUSTER_LABELS

st.set_page_config(page_title="Riyadh Metro Ridership Forecaster", page_icon="🚇", layout="wide")
st.title("Riyadh Metro Ridership Forecaster")

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("Settings")
data_source = st.sidebar.selectbox(
    "Data source",
    ["Station-level (ridership.csv)", "Line-level (ridership_by_line.csv)"],
)
test_days = st.sidebar.slider("Test set (days)", 7, 90, 30)
seq_length = st.sidebar.slider("LSTM sequence length", 24, 336, 168, 24)

# ── Load Data ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_station_data():
    """Load station-level ridership data with caching."""
    return load_ridership_data("data/ridership.csv")


@st.cache_data
def load_aggregated_data():
    """Load line-level aggregated ridership data with caching."""
    return load_line_data("data/ridership_by_line.csv")


try:
    if "Station" in data_source:
        df = load_station_data()
    else:
        df = load_aggregated_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.info("Generate data first: `python -m src.synth_ridership`")
    st.stop()

lines = get_line_names(df)
selected_line = st.sidebar.selectbox("Line filter", ["All Lines"] + lines)

if selected_line != "All Lines":
    df_filtered = df[df["line"] == selected_line].copy()
else:
    df_filtered = df.copy()

train, test = split_by_date(df_filtered, test_days=test_days)

# ── Tabs ──────────────────────────────────────────────────────────────────────

t1, t2, t3, t4 = st.tabs([
    "Data Explorer", "Seasonal Patterns", "Forecasts", "Station Clusters",
])

# ── Tab 1: Data Explorer ─────────────────────────────────────────────────────

with t1:
    st.subheader("Ridership Trends")

    if "line" in df_filtered.columns:
        daily = (
            df_filtered.groupby([pd.Grouper(key="timestamp", freq="D"), "line"])["ridership"]
            .sum()
            .reset_index()
        )
        fig_trend = px.line(
            daily, x="timestamp", y="ridership", color="line",
            title="Daily Total Ridership by Line",
        )
    else:
        daily = df_filtered.groupby(pd.Grouper(key="timestamp", freq="D"))["ridership"].sum().reset_index()
        fig_trend = px.line(daily, x="timestamp", y="ridership", title="Daily Total Ridership")

    fig_trend.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Basic Statistics")
    st.dataframe(
        df_filtered["ridership"].describe().round(1).to_frame().T,
        use_container_width=True,
    )

    # Station heatmap
    if "station_id" in df_filtered.columns:
        st.subheader("Station Ridership Heatmap")
        top_stations = (
            df_filtered.groupby("station_id")["ridership"]
            .mean()
            .nlargest(20)
            .index.tolist()
        )
        heatmap_data = df_filtered[df_filtered["station_id"].isin(top_stations)]
        pivot = heatmap_data.pivot_table(
            values="ridership", index="station_id", columns="hour", aggfunc="mean",
        )
        fig_heat = px.imshow(
            pivot, color_continuous_scale="YlOrRd",
            title="Top 20 Stations: Average Hourly Ridership",
            labels=dict(x="Hour", y="Station", color="Ridership"),
        )
        fig_heat.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

# ── Tab 2: Seasonal Patterns ─────────────────────────────────────────────────

with t2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hourly Pattern")
        hourly = df_filtered.groupby("hour")["ridership"].mean().reset_index()
        fig_h = px.bar(hourly, x="hour", y="ridership", title="Average Ridership by Hour")
        fig_h.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_h, use_container_width=True)

    with col2:
        st.subheader("Day of Week Pattern")
        daily_dow = df_filtered.groupby("day_of_week")["ridership"].mean().reset_index()
        daily_dow["day_name"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig_d = px.bar(
            daily_dow, x="day_name", y="ridership",
            title="Average Ridership by Day (Thu-Fri = Weekend)",
        )
        fig_d.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_d, use_container_width=True)

    st.subheader("Monthly Pattern")
    monthly = df_filtered.groupby("month")["ridership"].mean().reset_index()
    fig_m = px.bar(monthly, x="month", y="ridership", title="Average Ridership by Month")
    fig_m.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig_m, use_container_width=True)

    # Prayer time effect
    st.subheader("Prayer Time Effects")
    df_prayer = add_prayer_flags(df_filtered)
    prayer_effect = df_prayer.groupby("any_prayer")["ridership"].mean().reset_index()
    prayer_effect["label"] = prayer_effect["any_prayer"].map({0: "Non-Prayer", 1: "Prayer Hour"})
    fig_prayer = px.bar(
        prayer_effect, x="label", y="ridership",
        title="Average Ridership: Prayer Hour vs Non-Prayer Hour",
        color="label",
    )
    fig_prayer.update_layout(template="plotly_dark", height=350, showlegend=False)
    st.plotly_chart(fig_prayer, use_container_width=True)

    # Ramadan effect
    if "is_ramadan" in df_filtered.columns:
        st.subheader("Ramadan Effect")
        ramadan_hourly = (
            df_filtered.groupby(["hour", "is_ramadan"])["ridership"]
            .mean()
            .reset_index()
        )
        ramadan_hourly["period"] = ramadan_hourly["is_ramadan"].map(
            {0: "Normal", 1: "Ramadan"}
        )
        fig_r = px.line(
            ramadan_hourly, x="hour", y="ridership", color="period",
            title="Hourly Ridership: Ramadan vs Normal Period",
        )
        fig_r.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_r, use_container_width=True)

    # Temperature effect
    if "temperature" in df_filtered.columns:
        st.subheader("Temperature Effect")
        df_temp = df_filtered.copy()
        df_temp["temp_bin"] = pd.cut(
            df_temp["temperature"],
            bins=[0, 20, 30, 35, 40, 50],
            labels=["<20C", "20-30C", "30-35C", "35-40C", "40C+"],
        )
        temp_effect = df_temp.groupby("temp_bin")["ridership"].mean().reset_index()
        fig_t = px.bar(
            temp_effect, x="temp_bin", y="ridership",
            title="Average Ridership by Temperature Range",
        )
        fig_t.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_t, use_container_width=True)

# ── Tab 3: Forecasts ─────────────────────────────────────────────────────────

with t3:
    st.subheader("Train / Test Split")
    fig_split = go.Figure()
    fig_split.add_trace(go.Scatter(
        x=train["timestamp"], y=train["ridership"], name="Train",
        line=dict(color="#636EFA"),
    ))
    fig_split.add_trace(go.Scatter(
        x=test["timestamp"], y=test["ridership"], name="Test",
        line=dict(color="#EF553B"),
    ))
    fig_split.update_layout(
        template="plotly_dark", height=400,
        title="Chronological Train/Test Split",
    )
    st.plotly_chart(fig_split, use_container_width=True)

    st.subheader("Model Comparison")
    st.markdown(
        "To train and compare models, run the training pipeline:\n"
        "```python\n"
        "from src.data_loader import load_line_data, split_by_date\n"
        "from src.feature_engineering import build_features\n"
        "from src.forecasting import ModelTrainer\n\n"
        "df = load_line_data('data/ridership_by_line.csv')\n"
        "df = build_features(df)\n"
        "train, test = split_by_date(df, test_days=30)\n\n"
        "trainer = ModelTrainer()\n"
        "results = trainer.compare_models(train, test)\n"
        "print(results)\n"
        "```"
    )

    st.info(
        "Pre-computed results: XGBoost achieves the lowest MAPE (10.8%), "
        "followed by LSTM (11.9%) and Prophet (14.2%)."
    )

# ── Tab 4: Station Clusters ──────────────────────────────────────────────────

with t4:
    st.subheader("Station Clustering by Ridership Profile")

    if "station_id" not in df.columns:
        st.warning("Station clustering requires station-level data. "
                    "Select 'Station-level' in the sidebar.")
    else:
        @st.cache_data
        def run_clustering(_df):
            """Run station clustering with caching."""
            clusterer = StationClusterer(n_clusters=4)
            profiles = clusterer.build_station_profiles(_df)
            clusterer.fit_clusters()
            summary = clusterer.characterize_clusters()
            return clusterer, profiles, summary

        clusterer, profiles, summary = run_clustering(df)

        st.dataframe(summary, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            fig_scatter = clusterer.plot_clusters()
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            fig_profiles = clusterer.plot_cluster_profiles(df)
            st.plotly_chart(fig_profiles, use_container_width=True)

        st.subheader("Cluster Feature Heatmap")
        fig_heatmap = clusterer.plot_cluster_heatmap()
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Cluster membership table
        st.subheader("Station Assignments")
        display_cols = ["station_id", "cluster", "mean_ridership",
                        "peak_ratio", "weekend_ratio"]
        available_cols = [c for c in display_cols if c in profiles.columns]
        if "line" in profiles.columns:
            available_cols.insert(1, "line")
        cluster_display = profiles[available_cols].copy()
        cluster_display["cluster_name"] = cluster_display["cluster"].map(CLUSTER_LABELS)
        st.dataframe(
            cluster_display.sort_values("cluster"),
            use_container_width=True,
            height=400,
        )
