import numpy as np
import pandas as pd
import pytest

from src.forecasting import mape, ModelTrainer, FORECAST_FEATURES, TARGET
from src.feature_engineering import (
    add_cyclical_features,
    add_lag_features,
    add_rolling_features,
    add_weekend_flag,
    add_prayer_flags,
    add_holiday_flags,
    add_temperature_features,
    build_features,
    get_feature_names,
    DEFAULT_LAGS,
    DEFAULT_ROLLING_WINDOWS,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def hourly_df():
    """Create a small hourly DataFrame spanning two weeks.

    Returns
    -------
    pandas.DataFrame
        DataFrame with timestamp, ridership, hour, day_of_week, month,
        and temperature columns.
    """
    rng = np.random.RandomState(42)
    timestamps = pd.date_range("2024-06-01", periods=336, freq="h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "ridership": rng.poisson(500, size=336).astype(float),
        "hour": timestamps.hour,
        "day_of_week": timestamps.dayofweek,
        "month": timestamps.month,
        "temperature": 38.0 + rng.normal(0, 3, size=336),
    })


@pytest.fixture
def train_test_dfs(hourly_df):
    """Build engineered features and split into train/test.

    Parameters
    ----------
    hourly_df : pandas.DataFrame
        Raw hourly DataFrame from the fixture above.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        Train and test DataFrames with all engineered features.
    """
    df = build_features(hourly_df)
    split = int(len(df) * 0.8)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


# ── MAPE function ────────────────────────────────────────────────────────────


class TestMape:
    """Tests for the standalone mape helper."""

    def test_perfect_prediction(self):
        """MAPE of identical arrays should be zero."""
        y = np.array([100.0, 200.0, 300.0])
        assert mape(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        """MAPE for a known deviation should match hand calculation."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        # (10/100 + 20/200) / 2 * 100 = 10.0
        assert mape(y_true, y_pred) == pytest.approx(10.0)

    def test_ignores_zero_actuals(self):
        """Rows where y_true == 0 should be excluded from MAPE."""
        y_true = np.array([0.0, 100.0, 200.0])
        y_pred = np.array([5.0, 110.0, 180.0])
        expected = mape(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
        assert mape(y_true, y_pred) == pytest.approx(expected)


# ── Cyclical encoding ────────────────────────────────────────────────────────


class TestCyclicalFeatures:
    """Tests for add_cyclical_features output shapes and value ranges."""

    def test_adds_six_columns(self, hourly_df):
        """Should add hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos."""
        result = add_cyclical_features(hourly_df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                     "month_sin", "month_cos"]:
            assert col in result.columns

    def test_values_bounded(self, hourly_df):
        """All sin/cos values must be in [-1, 1]."""
        result = add_cyclical_features(hourly_df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                     "month_sin", "month_cos"]:
            assert result[col].min() >= -1.0 - 1e-9
            assert result[col].max() <= 1.0 + 1e-9

    def test_hour_zero_cos_equals_one(self):
        """At hour 0 the cosine encoding should be 1.0."""
        df = pd.DataFrame({"hour": [0], "day_of_week": [0], "month": [1]})
        result = add_cyclical_features(df)
        assert result["hour_cos"].iloc[0] == pytest.approx(1.0)

    def test_does_not_mutate_input(self, hourly_df):
        """Original DataFrame should not be modified."""
        cols_before = set(hourly_df.columns)
        add_cyclical_features(hourly_df)
        assert set(hourly_df.columns) == cols_before


# ── Lag features ─────────────────────────────────────────────────────────────


class TestLagFeatures:
    """Tests for add_lag_features."""

    def test_default_lag_columns(self, hourly_df):
        """Should create ridership_lag_1, _lag_24, _lag_168 by default."""
        result = add_lag_features(hourly_df)
        for lag in DEFAULT_LAGS:
            assert f"ridership_lag_{lag}" in result.columns

    def test_lag_1_value(self, hourly_df):
        """Lag-1 value at row i should equal ridership at row i-1."""
        result = add_lag_features(hourly_df, lags=[1])
        assert result["ridership_lag_1"].iloc[1] == hourly_df["ridership"].iloc[0]

    def test_custom_lags(self, hourly_df):
        """Custom lag list should produce only those columns."""
        result = add_lag_features(hourly_df, lags=[2, 5])
        assert "ridership_lag_2" in result.columns
        assert "ridership_lag_5" in result.columns
        assert "ridership_lag_1" not in result.columns


# ── Rolling features ─────────────────────────────────────────────────────────


class TestRollingFeatures:
    """Tests for add_rolling_features."""

    def test_default_rolling_columns(self, hourly_df):
        """Should create rolling mean and std for windows 24 and 168."""
        result = add_rolling_features(hourly_df)
        for w in DEFAULT_ROLLING_WINDOWS:
            assert f"ridership_roll_mean_{w}" in result.columns
            assert f"ridership_roll_std_{w}" in result.columns

    def test_rolling_mean_value(self, hourly_df):
        """Rolling mean at row 24 should match manual calculation."""
        result = add_rolling_features(hourly_df, windows=[24])
        manual = hourly_df["ridership"].iloc[:24].mean()
        assert result["ridership_roll_mean_24"].iloc[23] == pytest.approx(manual)


# ── Weekend flag ─────────────────────────────────────────────────────────────


class TestWeekendFlag:
    """Tests for the Saudi weekend (Thu=3, Fri=4) flag."""

    def test_thursday_friday_are_weekend(self):
        """Thursday and Friday should be flagged as weekend."""
        df = pd.DataFrame({"day_of_week": [3, 4]})
        result = add_weekend_flag(df)
        assert result["is_weekend"].tolist() == [1, 1]

    def test_other_days_are_weekday(self):
        """Days 0-2 and 5-6 should not be weekend."""
        df = pd.DataFrame({"day_of_week": [0, 1, 2, 5, 6]})
        result = add_weekend_flag(df)
        assert result["is_weekend"].sum() == 0


# ── Prayer flags ─────────────────────────────────────────────────────────────


class TestPrayerFlags:
    """Tests for add_prayer_flags."""

    def test_any_prayer_column_exists(self, hourly_df):
        """The aggregate any_prayer column should be created."""
        result = add_prayer_flags(hourly_df)
        assert "any_prayer" in result.columns

    def test_any_prayer_is_binary(self, hourly_df):
        """any_prayer should only contain 0 or 1."""
        result = add_prayer_flags(hourly_df)
        assert set(result["any_prayer"].unique()).issubset({0, 1})


# ── Temperature features ────────────────────────────────────────────────────


class TestTemperatureFeatures:
    """Tests for add_temperature_features."""

    def test_temp_squared(self, hourly_df):
        """temp_squared should equal temperature ** 2."""
        result = add_temperature_features(hourly_df)
        expected = hourly_df["temperature"] ** 2
        np.testing.assert_array_almost_equal(
            result["temp_squared"].values, expected.values
        )

    def test_extreme_heat_threshold(self):
        """Temperatures >= 40 should flag is_extreme_heat = 1."""
        df = pd.DataFrame({"temperature": [39.9, 40.0, 45.0]})
        result = add_temperature_features(df)
        assert result["is_extreme_heat"].tolist() == [0, 1, 1]


# ── Holiday flags ────────────────────────────────────────────────────────────


class TestHolidayFlags:
    """Tests for add_holiday_flags."""

    def test_founding_day(self):
        """Feb 22 should be flagged as a holiday."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-02-22 12:00:00"]),
            "hour": [12],
            "month": [2],
        })
        result = add_holiday_flags(df)
        assert result["is_founding_day"].iloc[0] == 1
        assert result["is_holiday"].iloc[0] == 1

    def test_non_holiday(self):
        """A regular day should have is_holiday = 0."""
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-05-15 10:00:00"]),
            "hour": [10],
            "month": [5],
        })
        result = add_holiday_flags(df)
        assert result["is_holiday"].iloc[0] == 0


# ── build_features pipeline ─────────────────────────────────────────────────


class TestBuildFeatures:
    """Tests for the full build_features pipeline."""

    def test_no_nans_in_output(self, hourly_df):
        """build_features should drop all NaN rows."""
        result = build_features(hourly_df)
        assert result.isna().sum().sum() == 0

    def test_output_has_cyclical_and_lag_columns(self, hourly_df):
        """Output should contain cyclical and lag feature columns."""
        result = build_features(hourly_df)
        assert "hour_sin" in result.columns
        assert "ridership_lag_1" in result.columns
        assert "any_prayer" in result.columns

    def test_row_count_reduced(self, hourly_df):
        """Dropping NaN rows means output is shorter than input."""
        result = build_features(hourly_df)
        assert len(result) < len(hourly_df)


# ── get_feature_names ────────────────────────────────────────────────────────


class TestGetFeatureNames:
    """Tests for get_feature_names utility."""

    def test_returns_list_of_strings(self):
        """Should return a non-empty list of strings."""
        names = get_feature_names()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)
        assert len(names) > 0

    def test_contains_cyclical(self):
        """Output should include cyclical encoding names."""
        names = get_feature_names()
        assert "hour_sin" in names
        assert "month_cos" in names


# ── ModelTrainer ─────────────────────────────────────────────────────────────


class TestModelTrainer:
    """Tests for ModelTrainer initialisation and helpers."""

    def test_init_creates_output_dir(self, tmp_path):
        """Constructor should create the output directory."""
        out = tmp_path / "model_out"
        trainer = ModelTrainer(output_dir=str(out))
        assert out.exists()
        assert trainer.models == {}
        assert trainer.results == {}

    def test_compute_metrics_keys(self):
        """_compute_metrics should return RMSE, MAE, MAPE keys."""
        y = np.array([100.0, 200.0, 300.0])
        metrics = ModelTrainer._compute_metrics(y, y)
        assert set(metrics.keys()) == {"RMSE", "MAE", "MAPE"}

    def test_compute_metrics_perfect(self):
        """Perfect predictions should yield zero for all metrics."""
        y = np.array([100.0, 200.0, 300.0])
        metrics = ModelTrainer._compute_metrics(y, y)
        assert metrics["RMSE"] == 0.0
        assert metrics["MAE"] == 0.0
        assert metrics["MAPE"] == 0.0

    def test_create_sequences_shape(self):
        """_create_sequences should produce correct array shapes."""
        data = np.arange(20).reshape(-1, 1).astype(float)
        seq_len = 5
        X, y = ModelTrainer._create_sequences(data, seq_len)
        assert X.shape == (15, 5, 1)
        assert y.shape == (15,)

    def test_create_sequences_values(self):
        """First sequence should match data[0:seq_len], target = data[seq_len]."""
        data = np.arange(10).reshape(-1, 1).astype(float)
        X, y = ModelTrainer._create_sequences(data, 3)
        np.testing.assert_array_equal(X[0].flatten(), [0, 1, 2])
        assert y[0] == 3.0

    def test_create_sequences_empty_for_short_data(self):
        """If data length <= seq_length, sequences should be empty."""
        data = np.arange(5).reshape(-1, 1).astype(float)
        X, y = ModelTrainer._create_sequences(data, 5)
        assert len(X) == 0
        assert len(y) == 0

    def test_comparison_table_after_xgboost(self, train_test_dfs):
        """comparison_table should include XGBoost after training it."""
        train_df, test_df = train_test_dfs
        trainer = ModelTrainer()
        trainer.train_xgboost(train_df, test_df)
        table = trainer.comparison_table()
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 1
        assert "XGBoost" in table["Model"].values

    def test_train_xgboost_runs(self, train_test_dfs):
        """XGBoost training should complete and populate results."""
        train_df, test_df = train_test_dfs
        trainer = ModelTrainer()
        metrics = trainer.train_xgboost(train_df, test_df)
        assert "RMSE" in metrics
        assert "XGBoost" in trainer.models
        assert "XGBoost" in trainer.predictions
        assert len(trainer.predictions["XGBoost"]) == len(test_df)

    def test_train_xgboost_no_features_raises(self):
        """Passing a DataFrame without any valid features should raise."""
        trainer = ModelTrainer()
        df = pd.DataFrame({"ridership": [1, 2, 3], "unrelated": [4, 5, 6]})
        with pytest.raises(ValueError, match="No valid features"):
            trainer.train_xgboost(df, df, features=["nonexistent_col"])

    def test_save_model_unknown(self, capsys):
        """Saving an unknown model name should print a warning."""
        trainer = ModelTrainer()
        trainer.save_model("FakeModel")
        captured = capsys.readouterr()
        assert "not found" in captured.out


# ── FORECAST_FEATURES constant ───────────────────────────────────────────────


class TestForecastFeatures:
    """Sanity checks on the FORECAST_FEATURES list."""

    def test_target_not_in_features(self):
        """The target column should not appear in the feature list."""
        assert TARGET not in FORECAST_FEATURES

    def test_cyclical_pairs_present(self):
        """Cyclical features should appear in sin/cos pairs."""
        for prefix in ["hour", "dow", "month"]:
            assert f"{prefix}_sin" in FORECAST_FEATURES
            assert f"{prefix}_cos" in FORECAST_FEATURES
