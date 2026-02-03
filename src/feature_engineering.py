import numpy as np
import pandas as pd


# Approximate prayer times by month (Riyadh)
PRAYER_HOURS = {
    "fajr":    [5, 5, 5, 4, 4, 3, 4, 4, 4, 5, 5, 5],
    "dhuhr":   [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    "asr":     [15, 15, 15, 15, 15, 16, 16, 15, 15, 15, 15, 15],
    "maghrib": [17, 18, 18, 18, 19, 19, 19, 18, 18, 17, 17, 17],
    "isha":    [19, 19, 19, 20, 20, 21, 21, 20, 20, 19, 19, 19],
}

# Fixed Saudi holidays (month, day)
SAUDI_HOLIDAYS = [
    (2, 22),   # Founding Day
    (9, 23),   # National Day
]

# Average monthly temperatures (Celsius)
MONTHLY_TEMPS = [15, 18, 22, 28, 34, 39, 42, 42, 38, 32, 24, 17]

# Default configuration
DEFAULT_LAGS = [1, 24, 168]
DEFAULT_ROLLING_WINDOWS = [24, 168]


def add_cyclical_features(df):
    """Encode hour, day_of_week, and month as sin/cos pairs.

    Cyclical encoding preserves the circular nature of time
    (e.g., hour 23 is close to hour 0, December is close to January).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with hour, day_of_week, and month columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with sin/cos encoded time features.
    """
    df = df.copy()

    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    if "day_of_week" in df.columns:
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_lag_features(df, column="ridership", lags=None):
    """Add lag features for the target column.

    Creates lagged versions of the ridership column at 1 hour,
    24 hours (1 day), and 168 hours (1 week) offsets.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with the target column.
    column : str
        Name of the column to lag.
    lags : list of int or None
        Lag offsets to use. Defaults to [1, 24, 168].

    Returns
    -------
    pandas.DataFrame
        DataFrame with lag columns added.
    """
    df = df.copy()
    lags = lags or DEFAULT_LAGS
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df


def add_prayer_flags(df):
    """Add binary flags for each of the five daily prayer times.

    Each prayer reduces transit ridership for approximately one hour.
    The prayer hour varies by month due to solar position changes.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with hour and month columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with prayer flag columns added.
    """
    # FIXME: date parsing slow on large datasets
    df = df.copy()

    for prayer, hours_by_month in PRAYER_HOURS.items():
        month_to_hour = {m + 1: h for m, h in enumerate(hours_by_month)}
        df[f"prayer_{prayer}"] = df.apply(
            lambda row: 1 if row.get("hour") == month_to_hour.get(row.get("month"), -1) else 0,
            axis=1,
        )

    df["any_prayer"] = (
        df[[f"prayer_{p}" for p in PRAYER_HOURS]].max(axis=1)
    )
    return df


def _is_eid_fitr(ts):
    """Check if timestamp falls in approximate Eid al-Fitr period.

    Parameters
    ----------
    ts : pandas.Timestamp
        Timestamp to check.

    Returns
    -------
    bool
        True if the date is within Eid al-Fitr.
    """
    year = ts.year
    if year == 2023:
        return pd.Timestamp("2023-04-21") <= ts <= pd.Timestamp("2023-04-23")
    elif year == 2024:
        return pd.Timestamp("2024-04-10") <= ts <= pd.Timestamp("2024-04-12")
    elif year == 2025:
        return pd.Timestamp("2025-03-30") <= ts <= pd.Timestamp("2025-04-01")
    return False


def _is_eid_adha(ts):
    """Check if timestamp falls in approximate Eid al-Adha period.

    Parameters
    ----------
    ts : pandas.Timestamp
        Timestamp to check.

    Returns
    -------
    bool
        True if the date is within Eid al-Adha.
    """
    year = ts.year
    if year == 2023:
        return pd.Timestamp("2023-06-28") <= ts <= pd.Timestamp("2023-07-02")
    elif year == 2024:
        return pd.Timestamp("2024-06-16") <= ts <= pd.Timestamp("2024-06-20")
    elif year == 2025:
        return pd.Timestamp("2025-06-06") <= ts <= pd.Timestamp("2025-06-10")
    return False


def add_holiday_flags(df):
    """Add Saudi holiday indicator features.

    Includes fixed-date holidays (Founding Day, National Day) and
    approximate Eid holidays.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with timestamp column.

    Returns
    -------
    pandas.DataFrame
        DataFrame with holiday flag columns added.
    """
    df = df.copy()

    df["is_founding_day"] = (
        (df["timestamp"].dt.month == 2) & (df["timestamp"].dt.day == 22)
    ).astype(int)

    df["is_national_day"] = (
        (df["timestamp"].dt.month == 9) & (df["timestamp"].dt.day == 23)
    ).astype(int)

    df["is_eid_fitr"] = df["timestamp"].apply(_is_eid_fitr).astype(int)
    df["is_eid_adha"] = df["timestamp"].apply(_is_eid_adha).astype(int)

    df["is_holiday"] = (
        df[["is_founding_day", "is_national_day", "is_eid_fitr", "is_eid_adha"]].max(axis=1)
    )

    return df


def add_rolling_features(df, column="ridership", windows=None):
    """Add rolling mean and standard deviation features.

    Computes rolling statistics over 24-hour and 168-hour (1 week)
    windows to capture short and medium-term trends.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with the target column.
    column : str
        Name of the column.
    windows : list of int or None
        Rolling window sizes. Defaults to [24, 168].

    Returns
    -------
    pandas.DataFrame
        DataFrame with rolling feature columns added.
    """
    df = df.copy()
    windows = windows or DEFAULT_ROLLING_WINDOWS
    for window in windows:
        df[f"{column}_roll_mean_{window}"] = df[column].rolling(window).mean()
        df[f"{column}_roll_std_{window}"] = df[column].rolling(window).std()
    return df


def add_weekend_flag(df):
    """Add Saudi weekend flag (Thursday-Friday).

    In Saudi Arabia, the weekend is Thursday and Friday, unlike the
    Saturday-Sunday weekend used in Western countries.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with day_of_week column.

    Returns
    -------
    pandas.DataFrame
        DataFrame with is_weekend column.
    """
    df = df.copy()
    if "day_of_week" in df.columns:
        df["is_weekend"] = df["day_of_week"].isin([3, 4]).astype(int)
    return df


def add_temperature_features(df):
    """Add temperature interaction features.

    Creates features capturing the relationship between temperature
    and ridership. Extreme heat in Riyadh drives transit use up
    as people prefer air-conditioned metro over driving.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with temperature and month columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with temperature features added.
    """
    df = df.copy()

    if "temperature" in df.columns:
        df["temp_squared"] = df["temperature"] ** 2
        df["is_extreme_heat"] = (df["temperature"] >= 40).astype(int)
    elif "month" in df.columns:
        temps = df["month"].map(
            {m + 1: t for m, t in enumerate(MONTHLY_TEMPS)}
        )
        df["temperature"] = temps
        df["temp_squared"] = temps ** 2
        df["is_extreme_heat"] = (temps >= 40).astype(int)

    if "ridership" in df.columns and "temperature" in df.columns:
        df["temp_ridership_interaction"] = df["temperature"] * df["ridership"]

    return df


def build_features(df, lags=None, rolling_windows=None):
    """Apply all feature engineering steps sequentially.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with timestamp, ridership, and time columns.
    lags : list of int or None
        Lag offsets for the target column.
    rolling_windows : list of int or None
        Window sizes for rolling statistics.

    Returns
    -------
    pandas.DataFrame
        DataFrame with all engineered features, NaN rows dropped.
    """
    df = add_prayer_flags(df)
    df = add_holiday_flags(df)
    df = add_cyclical_features(df)
    df = add_lag_features(df, lags=lags)
    df = add_rolling_features(df, windows=rolling_windows)
    df = add_weekend_flag(df)
    df = add_temperature_features(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  Features added: {len(df.columns)} columns, {len(df):,} rows")
    return df


def get_feature_names(lags=None, rolling_windows=None):
    """Return list of feature names created by the engineering pipeline.

    Parameters
    ----------
    lags : list of int or None
        Lag offsets used.
    rolling_windows : list of int or None
        Rolling window sizes used.

    Returns
    -------
    list of str
        Feature name strings.
    """
    lags = lags or DEFAULT_LAGS
    rolling_windows = rolling_windows or DEFAULT_ROLLING_WINDOWS

    prayer_features = [f"prayer_{p}" for p in PRAYER_HOURS] + ["any_prayer"]
    holiday_features = [
        "is_founding_day", "is_national_day",
        "is_eid_fitr", "is_eid_adha", "is_holiday",
    ]
    cyclical_features = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos",
    ]
    lag_features = [f"ridership_lag_{lag}" for lag in lags]
    rolling_features = []
    for w in rolling_windows:
        rolling_features.extend([
            f"ridership_roll_mean_{w}",
            f"ridership_roll_std_{w}",
        ])
    temp_features = [
        "temp_squared", "is_extreme_heat", "temp_ridership_interaction",
    ]
    weekend_features = ["is_weekend"]

    return (prayer_features + holiday_features + cyclical_features
            + lag_features + rolling_features + temp_features
            + weekend_features)

