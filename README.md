# Riyadh Metro Ridership Forecaster

Predicting passenger volumes for urban rail systems requires models that
capture both universal transit demand patterns and city-specific temporal
effects. This project develops and evaluates ridership forecasting models for
the Riyadh Metro network (6 lines, 85 stations), incorporating Saudi-specific
factors: five daily prayer times that temporarily reduce demand, the
Thursday-Friday weekend, Ramadan schedule shifts, Hajj-period travel
reductions, Eid holidays, and temperature-driven mode substitution during
extreme summer heat.

## Methodology

Synthetic hourly ridership data is generated using Poisson sampling with
multiplicative effects from station type (interchange, business, residential,
airport), hourly demand curves, prayer-time reductions, weekend patterns,
Ramadan behavioral shifts, and temperature-based multipliers. Feature
engineering produces cyclical time encodings, lag variables at 1/24/168-hour
offsets, rolling statistics, prayer flags, holiday indicators, and
temperature interaction terms.

Three models are compared:

**Prophet.** Configured with daily, weekly, and yearly seasonality plus Saudi
country holidays. Captures trend changepoints automatically.

**XGBoost.** Gradient boosted trees trained on the full engineered feature set.
Benefits from explicit lag and rolling features; the 24-hour lag is
consistently the strongest predictor.

**LSTM.** Two-layer architecture (128 + 64 units) with dropout, trained on
168-hour sliding windows using Adam with early stopping.

## Results

On a 30-day holdout test, XGBoost achieved the lowest error (MAPE 10.8%),
followed by LSTM (11.9%) and Prophet (14.2%). The lag-24 feature dominates
XGBoost's importance ranking, confirming that same-hour-yesterday ridership
is the single best predictor. Prayer-time flags and the weekend indicator
rank in the top five, validating the importance of Saudi-specific temporal
modeling. Station clustering via K-means identifies four distinct archetypes
(business, residential, interchange, airport) with markedly different
demand profiles.

## Quick Start

```bash
pip install -r requirements.txt
python -m src.synth_ridership
streamlit run app.py
```

```python
from src.data_loader import load_line_data, split_by_date
from src.feature_engineering import build_features
from src.forecasting import ModelTrainer

df = load_line_data("data/ridership_by_line.csv")
df = build_features(df)
train, test = split_by_date(df, test_days=30)

trainer = ModelTrainer()
results = trainer.compare_models(train, test)
```

## Project Structure

```
riyadh-metro-ridership-forecaster/
├── src/
│   ├── __init__.py
│   ├── synth_ridership.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── forecasting.py
│   ├── forecast_eval.py
│   └── station_clustering.py
├── app.py
├── requirements.txt
└── LICENSE
```

## License

Apache License 2.0 -- see [LICENSE](LICENSE) for details.
