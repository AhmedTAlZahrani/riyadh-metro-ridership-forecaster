import pandas as pd
import numpy as np
from pathlib import Path


REQUIRED_COLUMNS = ["timestamp", "line", "ridership"]

VALID_LINES = [
    "Line 1 (Blue)", "Line 2 (Green)", "Line 3 (Orange)",
    "Line 4 (Yellow)", "Line 5 (Purple)", "Line 6 (Gold)",
]


def load_ridership_data(path="data/ridership.csv"):
    """Load and validate the Riyadh Metro ridership dataset.

    Parses timestamps, validates required columns, fills missing
    hourly intervals with zero ridership, and adds time features.

    Parameters
    ----------
    path : str
        Path to the ridership CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with validated and enriched ridership data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. Run synth_ridership.py first."
        )

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["line", "timestamp"]).reset_index(drop=True)

    df = _fill_missing_intervals(df)

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)

    print(f"Loaded {len(df):,} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Lines: {df['line'].nunique()} | Date range: {df['timestamp'].dt.date.nunique()} days")

    return df


def _fill_missing_intervals(df):
    """Fill missing hourly intervals with interpolated ridership.

    For each group (line + station_id if present, or just line),
    creates a complete hourly date range and fills missing timestamps
    via linear interpolation.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with timestamp and line columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with complete hourly coverage per group.
    """
    filled_frames = []

    if "station_id" in df.columns:
        group_cols = ["line", "station_id"]
    else:
        group_cols = ["line"]

    for group_key, group_df in df.groupby(group_cols):
        group_df = group_df.copy()
        full_range = pd.date_range(
            start=group_df["timestamp"].min(),
            end=group_df["timestamp"].max(),
            freq="h",
        )

        group_df = group_df.set_index("timestamp").reindex(full_range)
        group_df.index.name = "timestamp"

        if isinstance(group_key, str):
            group_df["line"] = group_key
        else:
            for col, val in zip(group_cols, group_key):
                group_df[col] = val

        numeric_cols = group_df.select_dtypes(include=[np.number]).columns
        group_df[numeric_cols] = group_df[numeric_cols].interpolate(method="linear")

        cat_cols = group_df.select_dtypes(include=["object", "category"]).columns
        group_df[cat_cols] = group_df[cat_cols].ffill()

        group_df = group_df.reset_index()
        filled_frames.append(group_df)

    result = pd.concat(filled_frames, ignore_index=True)
    gaps = len(result) - len(df)
    if gaps > 0:
        print(f"  Filled {gaps} missing hourly intervals")

    return result


def load_line_data(path="data/ridership_by_line.csv"):
    """Load line-level aggregated ridership data.

    Parameters
    ----------
    path : str
        Path to the line-level CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with line-level hourly ridership.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Line data not found: {path}. Run synth_ridership.py first."
        )

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["line", "timestamp"]).reset_index(drop=True)

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    print(f"Loaded {len(df):,} line-level records")
    return df


def split_by_date(df, test_days=30):
    """Split data chronologically into train and test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with timestamp column.
    test_days : int
        Number of days to reserve for testing.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        Train and test DataFrames.
    """
    cutoff = df["timestamp"].max() - pd.Timedelta(days=test_days)
    train = df[df["timestamp"] <= cutoff].copy()
    test = df[df["timestamp"] > cutoff].copy()
    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows")
    return train, test


def get_line_names(df):
    """Return sorted list of unique line names in the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with line column.

    Returns
    -------
    list of str
        Line name strings.
    """
    return sorted(df["line"].unique().tolist())
