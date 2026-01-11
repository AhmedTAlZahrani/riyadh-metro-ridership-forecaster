import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


N_CLUSTERS = 4
CLUSTER_LABELS = {
    0: "Business District",
    1: "Residential",
    2: "Interchange Hub",
    3: "Airport / Special",
}

RANDOM_SEED = 42


class StationClusterer:
    """Cluster metro stations by ridership profile.

    Uses K-means clustering on aggregated ridership features to identify
    station archetypes: business, residential, interchange, and airport.

    Parameters
    ----------
    n_clusters : int
        Number of clusters for K-means.
    """

    def __init__(self, n_clusters=N_CLUSTERS):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.station_profiles = None
        self.cluster_labels = None

    def build_station_profiles(self, df):
        """Aggregate hourly ridership into station-level profiles.

        Creates features per station including average ridership by
        time period, peak ratios, weekend effects, and variability.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with station_id, ridership, hour, day_of_week.

        Returns
        -------
        pandas.DataFrame
            DataFrame with one row per station and profile features.
        """
        print("Building station profiles...")

        profiles = df.groupby("station_id").agg(
            mean_ridership=("ridership", "mean"),
            std_ridership=("ridership", "std"),
            max_ridership=("ridership", "max"),
            total_ridership=("ridership", "sum"),
        ).reset_index()

        morning = df[df["hour"].between(6, 9)].groupby("station_id")["ridership"].mean()
        midday = df[df["hour"].between(10, 15)].groupby("station_id")["ridership"].mean()
        evening = df[df["hour"].between(16, 19)].groupby("station_id")["ridership"].mean()
        night = df[df["hour"].isin(list(range(0, 6)) + list(range(20, 24)))].groupby("station_id")["ridership"].mean()

        profiles["morning_avg"] = profiles["station_id"].map(morning).fillna(0)
        profiles["midday_avg"] = profiles["station_id"].map(midday).fillna(0)
        profiles["evening_avg"] = profiles["station_id"].map(evening).fillna(0)
        profiles["night_avg"] = profiles["station_id"].map(night).fillna(0)

        profiles["peak_ratio"] = (
            (profiles["morning_avg"] + profiles["evening_avg"])
            / (profiles["midday_avg"] + 1)
        )

        weekend = df[df["day_of_week"].isin([3, 4])].groupby("station_id")["ridership"].mean()
        weekday = df[~df["day_of_week"].isin([3, 4])].groupby("station_id")["ridership"].mean()
        profiles["weekend_ratio"] = (
            profiles["station_id"].map(weekend).fillna(0)
            / (profiles["station_id"].map(weekday).fillna(1))
        )

        profiles["cv"] = profiles["std_ridership"] / (profiles["mean_ridership"] + 1)

        if "line" in df.columns:
            line_map = df.groupby("station_id")["line"].first()
            profiles["line"] = profiles["station_id"].map(line_map)

        if "station_type" in df.columns:
            type_map = df.groupby("station_id")["station_type"].first()
            profiles["station_type"] = profiles["station_id"].map(type_map)

        self.station_profiles = profiles
        print(f"  Built profiles for {len(profiles)} stations")
        return profiles

    def fit_clusters(self, profiles=None):
        """Fit K-means clustering on station profiles.

        Parameters
        ----------
        profiles : pandas.DataFrame or None
            Station profiles DataFrame. Uses stored profiles
            if not provided.

        Returns
        -------
        numpy.ndarray
            Array of cluster labels for each station.
        """
        profiles = profiles if profiles is not None else self.station_profiles
        if profiles is None:
            raise ValueError("No station profiles available. Run build_station_profiles first.")

        feature_cols = [
            "mean_ridership", "std_ridership", "morning_avg", "midday_avg",
            "evening_avg", "night_avg", "peak_ratio", "weekend_ratio", "cv",
        ]

        X = profiles[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=RANDOM_SEED,
            n_init=10,
        )
        labels = self.kmeans.fit_predict(X_scaled)

        profiles["cluster"] = labels
        self.station_profiles = profiles
        self.cluster_labels = labels

        print(f"\nCluster distribution:")
        for cluster_id in range(self.n_clusters):
            count = (labels == cluster_id).sum()
            label = CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")
            print(f"  {label}: {count} stations")

        return labels

    def characterize_clusters(self):
        """Generate descriptive statistics for each cluster.

        Returns
        -------
        pandas.DataFrame
            DataFrame with cluster-level summary statistics.
        """
        if self.station_profiles is None or "cluster" not in self.station_profiles.columns:
            raise ValueError("Run fit_clusters first")

        summary = self.station_profiles.groupby("cluster").agg(
            num_stations=("station_id", "count"),
            avg_ridership=("mean_ridership", "mean"),
            avg_morning=("morning_avg", "mean"),
            avg_evening=("evening_avg", "mean"),
            avg_peak_ratio=("peak_ratio", "mean"),
            avg_weekend_ratio=("weekend_ratio", "mean"),
        ).round(1)

        summary["label"] = summary.index.map(CLUSTER_LABELS)
        summary = summary.reset_index()

        print("\nCluster Characteristics:")
        print(summary.to_string(index=False))
        return summary

    def plot_clusters(self):
        """Create scatter plot of stations colored by cluster.

        Uses morning and evening average ridership as axes to
        visualize the separation between station archetypes.

        Returns
        -------
        plotly.graph_objects.Figure
            Plotly Figure with cluster scatter plot.
        """
        if self.station_profiles is None or "cluster" not in self.station_profiles.columns:
            raise ValueError("Run fit_clusters first")

        df = self.station_profiles.copy()
        df["cluster_name"] = df["cluster"].map(CLUSTER_LABELS)

        fig = px.scatter(
            df,
            x="morning_avg",
            y="evening_avg",
            color="cluster_name",
            hover_data=["station_id", "mean_ridership", "peak_ratio"],
            title="Station Clusters by Ridership Profile",
            labels={
                "morning_avg": "Morning Avg Ridership",
                "evening_avg": "Evening Avg Ridership",
                "cluster_name": "Cluster",
            },
        )
        fig.update_layout(template="plotly_dark", height=500)
        return fig

    def plot_cluster_profiles(self, df):
        """Plot average hourly ridership pattern for each cluster.

        Parameters
        ----------
        df : pandas.DataFrame
            Original hourly ridership DataFrame with station_id.

        Returns
        -------
        plotly.graph_objects.Figure
            Plotly Figure with hourly patterns by cluster.
        """
        if self.station_profiles is None or "cluster" not in self.station_profiles.columns:
            raise ValueError("Run fit_clusters first")

        station_cluster = self.station_profiles.set_index("station_id")["cluster"]
        merged = df.copy()
        merged["cluster"] = merged["station_id"].map(station_cluster)
        merged["cluster_name"] = merged["cluster"].map(CLUSTER_LABELS)

        hourly = (
            merged.groupby(["cluster_name", "hour"])["ridership"]
            .mean()
            .reset_index()
        )

        fig = px.line(
            hourly,
            x="hour",
            y="ridership",
            color="cluster_name",
            title="Hourly Ridership Pattern by Cluster",
            labels={
                "hour": "Hour of Day",
                "ridership": "Average Ridership",
                "cluster_name": "Cluster",
            },
        )
        fig.update_layout(template="plotly_dark", height=450)
        return fig

    def plot_cluster_heatmap(self):
        """Create heatmap of cluster feature centroids.

        Returns
        -------
        plotly.graph_objects.Figure
            Plotly Figure with normalized feature heatmap.
        """
        if self.kmeans is None:
            raise ValueError("Run fit_clusters first")

        feature_cols = [
            "mean_ridership", "std_ridership", "morning_avg", "midday_avg",
            "evening_avg", "night_avg", "peak_ratio", "weekend_ratio", "cv",
        ]

        centroids = self.kmeans.cluster_centers_
        cluster_names = [CLUSTER_LABELS.get(i, f"Cluster {i}")
                         for i in range(self.n_clusters)]

        fig = px.imshow(
            centroids,
            x=feature_cols,
            y=cluster_names,
            color_continuous_scale="RdBu_r",
            title="Cluster Feature Centroids (Standardized)",
            labels=dict(color="Value"),
        )
        fig.update_layout(template="plotly_dark", height=400)
        return fig
