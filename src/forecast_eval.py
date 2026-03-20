import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error


PEAK_HOURS = [7, 8, 9, 16, 17, 18]
OFF_PEAK_HOURS = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23]


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


class ModelEvaluator:
    """Evaluate ridership forecasting model performance.

    Provides per-line metrics, peak vs off-peak comparison,
    and interactive forecast visualizations.
    """

    @staticmethod
    def compute_metrics(y_true, y_pred):
        """Compute RMSE, MAE, and MAPE for a set of predictions.

        Parameters
        ----------
        y_true : array-like
            Array of actual ridership values.
        y_pred : array-like
            Array of predicted ridership values.

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
    def per_line_metrics(test_df, predictions, line_col="line"):
        """Compute evaluation metrics for each metro line separately.

        Parameters
        ----------
        test_df : pandas.DataFrame
            Test DataFrame with actual ridership and line labels.
        predictions : array-like
            Array of predicted ridership values.
        line_col : str
            Name of the line identifier column.

        Returns
        -------
        pandas.DataFrame
            DataFrame with RMSE, MAE, MAPE for each line.
        """
        df = test_df.copy()
        df["predicted"] = predictions[:len(df)]

        rows = []
        for line in sorted(df[line_col].unique()):
            line_data = df[df[line_col] == line]
            y_true = line_data["ridership"].values
            y_pred = line_data["predicted"].values

            metrics = ModelEvaluator.compute_metrics(y_true, y_pred)
            metrics["Line"] = line
            rows.append(metrics)

        result = pd.DataFrame(rows)[["Line", "RMSE", "MAE", "MAPE"]]
        print("\nPer-Line Metrics:")
        print(result.to_string(index=False))
        return result

    @staticmethod
    def peak_vs_offpeak(test_df, predictions):
        """Compare model accuracy during peak vs off-peak hours.

        Peak hours are defined as morning (7-9) and evening (16-18)
        commute periods.

        Parameters
        ----------
        test_df : pandas.DataFrame
            Test DataFrame with hour column.
        predictions : array-like
            Array of predicted ridership values.

        Returns
        -------
        pandas.DataFrame
            DataFrame with metrics for peak and off-peak periods.
        """
        df = test_df.copy()
        df["predicted"] = predictions[:len(df)]

        peak_mask = df["hour"].isin(PEAK_HOURS)
        offpeak_mask = df["hour"].isin(OFF_PEAK_HOURS)

        peak_metrics = ModelEvaluator.compute_metrics(
            df.loc[peak_mask, "ridership"].values,
            df.loc[peak_mask, "predicted"].values,
        )
        peak_metrics["Period"] = "Peak (7-9, 16-18)"

        offpeak_metrics = ModelEvaluator.compute_metrics(
            df.loc[offpeak_mask, "ridership"].values,
            df.loc[offpeak_mask, "predicted"].values,
        )
        offpeak_metrics["Period"] = "Off-Peak"

        result = pd.DataFrame([peak_metrics, offpeak_metrics])[
            ["Period", "RMSE", "MAE", "MAPE"]
        ]
        print("\nPeak vs Off-Peak Accuracy:")
        print(result.to_string(index=False))
        return result

    @staticmethod
    def plot_forecast(test_df, predictions, model_name="Model", line=None):
        """Create interactive forecast vs actual plot.

        Parameters
        ----------
        test_df : pandas.DataFrame
            Test DataFrame with timestamp and ridership.
        predictions : array-like
            Array of predicted ridership values.
        model_name : str
            Name of the model for the title.
        line : str or None
            Optional line name to filter data.

        Returns
        -------
        plotly.graph_objects.Figure
            Plotly Figure with actual vs predicted overlay.
        """
        df = test_df.copy()
        df["predicted"] = predictions[:len(df)]

        if line is not None:
            df = df[df["line"] == line]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["ridership"],
            name="Actual", line=dict(color="white", width=1.5),
        ))

        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["predicted"],
            name=model_name, line=dict(color="#636EFA", width=1.5),
        ))

        title = f"Forecast vs Actual: {model_name}"
        if line:
            title += f" ({line})"

        fig.update_layout(
            template="plotly_dark",
            height=450,
            title=title,
            xaxis_title="Time",
            yaxis_title="Ridership",
        )
        return fig

    @staticmethod
    def plot_model_comparison(test_df, all_predictions):
        """Overlay predictions from all models against actual values.

        Parameters
        ----------
        test_df : pandas.DataFrame
            Test DataFrame with timestamp and ridership.
        all_predictions : dict
            Dict mapping model names to prediction arrays.

        Returns
        -------
        plotly.graph_objects.Figure
            Plotly Figure with all models overlaid.
        """
        fig = go.Figure()

        timestamps = test_df["timestamp"].values
        fig.add_trace(go.Scatter(
            x=timestamps, y=test_df["ridership"].values,
            name="Actual", line=dict(color="white", width=2),
        ))

        colors = {
            "Prophet": "#EF553B",
            "XGBoost": "#636EFA",
            "LSTM": "#00CC96",
        }

        for name, pred in all_predictions.items():
            n = min(len(pred), len(timestamps))
            fig.add_trace(go.Scatter(
                x=timestamps[:n], y=pred[:n],
                name=name, line=dict(color=colors.get(name, "gray")),
            ))

        fig.update_layout(
            template="plotly_dark",
            height=500,
            title="Forecast Comparison: All Models vs Actual",
            xaxis_title="Time",
            yaxis_title="Ridership",
        )
        return fig

    @staticmethod
    def plot_error_distribution(y_true, y_pred, model_name="Model"):
        """Plot distribution of forecast errors.

        Parameters
        ----------
        y_true : array-like
            Array of actual values.
        y_pred : array-like
            Array of predicted values.
        model_name : str
            Name for the chart title.

        Returns
        -------
        plotly.graph_objects.Figure
            Plotly Figure with error histogram.
        """
        errors = np.array(y_true) - np.array(y_pred)

        fig = px.histogram(
            x=errors, nbins=50,
            title=f"Forecast Error Distribution: {model_name}",
            labels={"x": "Error (Actual - Predicted)", "y": "Count"},
        )
        fig.update_layout(template="plotly_dark", height=400)
        return fig

    @staticmethod
    def save_metrics(metrics, path="output/metrics.json"):
        """Save evaluation metrics to a JSON file.

        Parameters
        ----------
        metrics : dict or list
            Dict or list of dicts with metric values.
        path : str
            Output file path.
        """
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"Metrics saved to {output}")
