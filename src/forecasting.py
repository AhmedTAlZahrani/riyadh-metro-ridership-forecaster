import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor


FORECAST_FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "any_prayer", "is_holiday",
    "ridership_lag_1", "ridership_lag_24", "ridership_lag_168",
    "ridership_roll_mean_24", "ridership_roll_std_24",
    "ridership_roll_mean_168", "ridership_roll_std_168",
    "temperature", "temp_squared", "is_extreme_heat",
]

TARGET = "ridership"


def mape(y_true, y_pred):
    """Compute Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : array-like
        Array of actual values.
    y_pred : array-like
        Array of predicted values.

    Returns
    -------
    float
        MAPE as a percentage.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


class ModelTrainer:
    """Train and compare forecasting models for metro ridership.

    Supports Prophet, XGBoost, and LSTM models with a unified
    comparison framework using RMSE, MAE, and MAPE metrics.

    Parameters
    ----------
    output_dir : str
        Directory path for saving trained model artifacts.
    """

    def __init__(self, output_dir="models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.results = {}
        self.predictions = {}

    def train_prophet(self, train_df, test_df):
        """Train a Prophet model on ridership data.

        Prophet handles seasonality and holiday effects natively.
        The model is configured with daily and weekly seasonality
        components appropriate for metro transit patterns.

        Parameters
        ----------
        train_df : pandas.DataFrame
            Training DataFrame with timestamp and ridership.
        test_df : pandas.DataFrame
            Test DataFrame for evaluation.

        Returns
        -------
        dict
            Dict with RMSE, MAE, and MAPE metrics.
        """
        from prophet import Prophet

        print("Training Prophet model...")

        prophet_train = train_df[["timestamp", "ridership"]].copy()
        prophet_train.columns = ["ds", "y"]

        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
        )

        model.add_country_holidays(country_name="SA")
        model.fit(prophet_train)

        future = model.make_future_dataframe(
            periods=len(test_df), freq="h", include_history=False
        )
        forecast = model.predict(future)

        y_pred = forecast["yhat"].values[:len(test_df)]
        y_pred = np.maximum(y_pred, 0)
        y_true = test_df["ridership"].values[:len(y_pred)]

        metrics = self._compute_metrics(y_true, y_pred)
        self.models["Prophet"] = model
        self.predictions["Prophet"] = y_pred
        self.results["Prophet"] = metrics

        print(f"  Prophet -- RMSE: {metrics['RMSE']:.1f}, "
              f"MAE: {metrics['MAE']:.1f}, MAPE: {metrics['MAPE']:.1f}%")

        return metrics

    def train_xgboost(self, train_df, test_df, features=None):
        """Train an XGBoost regression model.

        Uses gradient boosted trees with features engineered from
        the time series. Typically performs best on tabular features
        with lag and rolling statistics.

        Parameters
        ----------
        train_df : pandas.DataFrame
            Training DataFrame with engineered features.
        test_df : pandas.DataFrame
            Test DataFrame for evaluation.
        features : list of str or None
            Feature column names to use.

        Returns
        -------
        dict
            Dict with RMSE, MAE, and MAPE metrics.
        """
        print("Training XGBoost model...")

        features = features or FORECAST_FEATURES
        available = [f for f in features if f in train_df.columns]

        if not available:
            raise ValueError("No valid features found in training data")

        X_train = train_df[available].values
        y_train = train_df[TARGET].values
        X_test = test_df[available].values
        y_true = test_df[TARGET].values

        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_true)],
            verbose=False,
        )

        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)

        metrics = self._compute_metrics(y_true, y_pred)
        self.models["XGBoost"] = model
        self.predictions["XGBoost"] = y_pred
        self.results["XGBoost"] = metrics

        importance = pd.DataFrame({
            "feature": available,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        print(f"  XGBoost -- RMSE: {metrics['RMSE']:.1f}, "
              f"MAE: {metrics['MAE']:.1f}, MAPE: {metrics['MAPE']:.1f}%")
        print(f"  Top features: {importance.head(5)['feature'].tolist()}")

        return metrics

    def train_lstm(self, train_df, test_df, seq_length=168, epochs=50,
                   batch_size=64):
        """Train a two-layer LSTM neural network.

        Uses a sliding window approach with configurable sequence length.
        The architecture consists of two LSTM layers with dropout
        regularization followed by dense output layers.

        Parameters
        ----------
        train_df : pandas.DataFrame
            Training DataFrame with ridership column.
        test_df : pandas.DataFrame
            Test DataFrame for evaluation.
        seq_length : int
            Number of timesteps in each input sequence.
        epochs : int
            Number of training epochs.
        batch_size : int
            Training batch size.

        Returns
        -------
        dict
            Dict with RMSE, MAE, and MAPE metrics.
        """
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import MinMaxScaler

        print("Training LSTM model...")

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(
            train_df[["ridership"]].values
        )
        test_scaled = scaler.transform(
            test_df[["ridership"]].values
        )

        X_train, y_train = self._create_sequences(train_scaled, seq_length)
        X_test, y_test = self._create_sequences(test_scaled, seq_length)

        if len(X_train) == 0 or len(X_test) == 0:
            print("  Warning: Not enough data for LSTM sequences")
            return {"RMSE": float("inf"), "MAE": float("inf"), "MAPE": float("inf")}

        model = Sequential([
            LSTM(128, return_sequences=True,
                 input_shape=(seq_length, 1)),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ])

        model.compile(optimizer="adam", loss="mse")

        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0,
        )

        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        y_pred = np.maximum(y_pred, 0)

        metrics = self._compute_metrics(y_true, y_pred)
        self.models["LSTM"] = model
        self.predictions["LSTM"] = y_pred
        self.results["LSTM"] = metrics

        print(f"  LSTM -- RMSE: {metrics['RMSE']:.1f}, "
              f"MAE: {metrics['MAE']:.1f}, MAPE: {metrics['MAPE']:.1f}%")

        return metrics

    def compare_models(self, train_df, test_df, features=None,
                       seq_length=168, epochs=50):
        """Train and compare all three forecasting models.

        Parameters
        ----------
        train_df : pandas.DataFrame
            Training DataFrame with engineered features.
        test_df : pandas.DataFrame
            Test DataFrame for evaluation.
        features : list of str or None
            Feature columns for XGBoost.
        seq_length : int
            LSTM sequence length.
        epochs : int
            LSTM training epochs.

        Returns
        -------
        pandas.DataFrame
            DataFrame with comparison metrics sorted by MAPE.
        """
        print("\n=== Model Comparison ===\n")

        self.train_prophet(train_df, test_df)
        self.train_xgboost(train_df, test_df, features)
        self.train_lstm(train_df, test_df, seq_length, epochs)

        comparison = self.comparison_table()
        print(f"\n{comparison.to_string(index=False)}")

        return comparison

    def comparison_table(self):
        """Return metrics as a sorted DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with Model, RMSE, MAE, MAPE columns.
        """
        rows = [{"Model": name, **metrics}
                for name, metrics in self.results.items()]
        return pd.DataFrame(rows).sort_values("MAPE")

    def save_model(self, model_name, filename=None):
        """Save a trained model to disk.

        Parameters
        ----------
        model_name : str
            Name of the model (Prophet, XGBoost, LSTM).
        filename : str or None
            Output filename without extension.
        """
        model = self.models.get(model_name)
        if model is None:
            print(f"Model '{model_name}' not found")
            return

        filename = filename or model_name.lower()

        if model_name == "LSTM":
            path = self.output_dir / f"{filename}.keras"
            model.save(path)
        else:
            path = self.output_dir / f"{filename}.pkl"
            joblib.dump(model, path)

        print(f"Saved {model_name} to {path}")

    @staticmethod
    def _compute_metrics(y_true, y_pred):
        """Compute RMSE, MAE, and MAPE.

        Parameters
        ----------
        y_true : array-like
            Array of actual values.
        y_pred : array-like
            Array of predicted values.

        Returns
        -------
        dict
            Dict with RMSE, MAE, and MAPE values.
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape_val = mape(y_true, y_pred)

        return {
            "RMSE": round(rmse, 1),
            "MAE": round(mae, 1),
            "MAPE": round(mape_val, 1),
        }

    @staticmethod
    def _create_sequences(data, seq_length):
        """Create sliding window sequences for LSTM training.

        Parameters
        ----------
        data : numpy.ndarray
            Scaled numpy array of shape (n_samples, 1).
        seq_length : int
            Number of timesteps per sequence.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            X array of shape (n_sequences, seq_length, 1) and y array.
        """
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i - seq_length:i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
