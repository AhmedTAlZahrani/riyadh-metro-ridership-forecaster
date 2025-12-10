import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


# ── Metro Network Configuration ──────────────────────────────────────────────

LINES = {
    "Line 1 (Blue)": {
        "corridor": "King Fahd Road",
        "num_stations": 22,
        "base_ridership": 850,
        "station_types": {
            "interchange": [0, 5, 10, 15, 21],
            "business": [3, 4, 6, 7, 8, 9, 11, 12],
            "residential": [1, 2, 13, 14, 16, 17, 18, 19, 20],
        },
    },
    "Line 2 (Green)": {
        "corridor": "Olaya Street",
        "num_stations": 13,
        "base_ridership": 720,
        "station_types": {
            "interchange": [0, 6, 12],
            "business": [3, 4, 5, 7, 8],
            "residential": [1, 2, 9, 10, 11],
        },
    },
    "Line 3 (Orange)": {
        "corridor": "Eastern corridor",
        "num_stations": 19,
        "base_ridership": 680,
        "station_types": {
            "interchange": [0, 9, 18],
            "business": [4, 5, 6, 7, 8, 10, 11],
            "residential": [1, 2, 3, 12, 13, 14, 15, 16, 17],
        },
    },
    "Line 4 (Yellow)": {
        "corridor": "King Abdullah Road",
        "num_stations": 11,
        "base_ridership": 580,
        "station_types": {
            "interchange": [0, 5, 10],
            "business": [2, 3, 4, 6, 7],
            "residential": [1, 8, 9],
        },
    },
    "Line 5 (Purple)": {
        "corridor": "Airport connector",
        "num_stations": 7,
        "base_ridership": 420,
        "station_types": {
            "interchange": [0],
            "airport": [6],
            "business": [2, 3, 4],
            "residential": [1, 5],
        },
    },
    "Line 6 (Gold)": {
        "corridor": "Diplomatic Quarter",
        "num_stations": 13,
        "base_ridership": 550,
        "station_types": {
            "interchange": [0, 6, 12],
            "business": [3, 4, 5, 7, 8, 9],
            "residential": [1, 2, 10, 11],
        },
    },
}

# Approximate prayer times by month (hour of day) for Riyadh
PRAYER_TIMES = {
    "fajr":    [5, 5, 5, 4, 4, 3, 4, 4, 4, 5, 5, 5],
    "dhuhr":   [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    "asr":     [15, 15, 15, 15, 15, 16, 16, 15, 15, 15, 15, 15],
    "maghrib": [17, 18, 18, 18, 19, 19, 19, 18, 18, 17, 17, 17],
    "isha":    [19, 19, 19, 20, 20, 21, 21, 20, 20, 19, 19, 19],
}

# Average monthly temperatures in Riyadh (Celsius)
MONTHLY_TEMPS = [15, 18, 22, 28, 34, 39, 42, 42, 38, 32, 24, 17]

# Saudi public holidays (month, day) -- fixed-date ones
FIXED_HOLIDAYS = [
    (2, 22),   # Founding Day
    (9, 23),   # National Day
]

RANDOM_SEED = 42


def _get_station_type(line_config, station_idx):
    """Determine the type of a station by its index.

    Parameters
    ----------
    line_config : dict
        Dictionary with station_types mapping.
    station_idx : int
        Index of the station within the line.

    Returns
    -------
    str
        Station type string (interchange, business, residential, airport).
    """
    for stype, indices in line_config["station_types"].items():
        if station_idx in indices:
            return stype
    return "residential"


def _station_type_multiplier(station_type, hour):
    """Calculate ridership multiplier based on station type and hour.

    Different station types have distinct ridership patterns throughout
    the day. Interchange stations are consistently high, business stations
    peak during commute hours, residential stations have reverse commute
    patterns, and airport stations align with flight schedules.

    Parameters
    ----------
    station_type : str
        One of interchange, business, residential, airport.
    hour : int
        Hour of the day (0-23).

    Returns
    -------
    float
        Multiplier for base ridership.
    """
    if station_type == "interchange":
        return 1.8
    elif station_type == "business":
        if hour in [7, 8, 9, 16, 17, 18]:
            return 1.6
        elif 10 <= hour <= 15:
            return 1.2
        else:
            return 0.5
    elif station_type == "airport":
        if hour in [5, 6, 7, 14, 15, 22, 23]:
            return 1.5
        elif 8 <= hour <= 13:
            return 1.0
        else:
            return 0.7
    else:  # residential
        if hour in [6, 7, 8]:
            return 1.3
        elif hour in [16, 17, 18]:
            return 1.4
        elif 22 <= hour or hour <= 4:
            return 0.2
        else:
            return 0.7


def _hourly_pattern(hour):
    """Base hourly demand curve for Riyadh Metro.

    Parameters
    ----------
    hour : int
        Hour of the day (0-23).

    Returns
    -------
    float
        Multiplier representing hourly demand shape.
    """
    pattern = {
        0: 0.10, 1: 0.05, 2: 0.03, 3: 0.02, 4: 0.03, 5: 0.15,
        6: 0.40, 7: 0.80, 8: 1.00, 9: 0.85, 10: 0.65, 11: 0.60,
        12: 0.55, 13: 0.60, 14: 0.65, 15: 0.70, 16: 0.90, 17: 1.00,
        18: 0.85, 19: 0.65, 20: 0.50, 21: 0.40, 22: 0.30, 23: 0.18,
    }
    return pattern.get(hour, 0.3)


def _is_prayer_time(hour, month):
    """Check if a given hour overlaps with any prayer time.

    Parameters
    ----------
    hour : int
        Hour of the day (0-23).
    month : int
        Month (1-12).

    Returns
    -------
    float
        Reduction factor (1.0 = no reduction, lower = more reduction).
    """
    month_idx = month - 1
    for prayer, hours_by_month in PRAYER_TIMES.items():
        prayer_hour = hours_by_month[month_idx]
        if hour == prayer_hour:
            if prayer == "fajr":
                return 0.85
            elif prayer in ("dhuhr", "asr"):
                return 0.75
            elif prayer == "maghrib":
                return 0.80
            else:
                return 0.80
    return 1.0


def _is_ramadan(date):
    """Approximate Ramadan period for 2024-2025.

    Uses approximate Gregorian dates since the Islamic calendar shifts
    annually. This is a simplification for synthetic data generation.

    Parameters
    ----------
    date : datetime
        Date to check.

    Returns
    -------
    bool
        True if date falls in approximate Ramadan.
    """
    year = date.year
    if year == 2024:
        start = datetime(2024, 3, 11)
        end = datetime(2024, 4, 9)
    elif year == 2025:
        start = datetime(2025, 2, 28)
        end = datetime(2025, 3, 30)
    else:
        start = datetime(year, 3, 1)
        end = datetime(year, 3, 30)
    return start <= date <= end


def _is_hajj_period(date):
    """Approximate Hajj period for 2024-2025.

    Parameters
    ----------
    date : datetime
        Date to check.

    Returns
    -------
    bool
        True if date falls in approximate Hajj period.
    """
    year = date.year
    if year == 2024:
        start = datetime(2024, 6, 7)
        end = datetime(2024, 6, 19)
    elif year == 2025:
        start = datetime(2025, 5, 27)
        end = datetime(2025, 6, 8)
    else:
        start = datetime(year, 6, 1)
        end = datetime(year, 6, 15)
    return start <= date <= end


def _is_eid(date):
    """Approximate Eid al-Fitr and Eid al-Adha holidays.

    Parameters
    ----------
    date : datetime
        Date to check.

    Returns
    -------
    bool
        True if date falls in an Eid holiday.
    """
    year = date.year
    if year == 2024:
        eid_fitr = (datetime(2024, 4, 10), datetime(2024, 4, 12))
        eid_adha = (datetime(2024, 6, 16), datetime(2024, 6, 20))
    elif year == 2025:
        eid_fitr = (datetime(2025, 3, 30), datetime(2025, 4, 1))
        eid_adha = (datetime(2025, 6, 6), datetime(2025, 6, 10))
    else:
        eid_fitr = (datetime(year, 4, 10), datetime(year, 4, 12))
        eid_adha = (datetime(year, 6, 16), datetime(year, 6, 20))

    return (eid_fitr[0] <= date <= eid_fitr[1]) or (eid_adha[0] <= date <= eid_adha[1])


def _is_school_period(date):
    """Check if date falls within school term.

    Parameters
    ----------
    date : datetime
        Date to check.

    Returns
    -------
    bool
        True if schools are in session.
    """
    month = date.month
    if month in [7, 8]:
        return False
    if month == 6 and date.day > 20:
        return False
    return True


def _temperature_effect(month):
    """Compute temperature-driven ridership multiplier.

    In Riyadh, extreme summer heat (45C+) drives more people to use
    air-conditioned metro instead of walking or driving.

    Parameters
    ----------
    month : int
        Month (1-12).

    Returns
    -------
    float
        Multiplier (higher in summer months).
    """
    temp = MONTHLY_TEMPS[month - 1]
    if temp >= 40:
        return 1.25
    elif temp >= 35:
        return 1.15
    elif temp >= 30:
        return 1.05
    elif temp <= 18:
        return 0.90
    return 1.0


def generate_ridership_data(start_date="2023-01-01", periods_days=730,
                            output_path="data/ridership.csv"):
    """Generate synthetic hourly ridership data for all Riyadh Metro lines.

    Creates realistic ridership patterns incorporating Saudi-specific
    temporal effects including prayer times, Ramadan, Hajj, Eid holidays,
    school calendar, extreme summer heat, and the Thursday-Friday weekend.

    Parameters
    ----------
    start_date : str
        Start date string in YYYY-MM-DD format.
    periods_days : int
        Number of days to generate.
    output_path : str
        Path to save the output CSV.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: timestamp, line, station_id, station_type,
        ridership, temperature, is_weekend, is_holiday, is_ramadan.
    """
    np.random.seed(RANDOM_SEED)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(start_date)
    total_hours = periods_days * 24
    timestamps = pd.date_range(start=start, periods=total_hours, freq="h")

    records = []
    total_stations = sum(cfg["num_stations"] for cfg in LINES.values())

    print(f"Generating {periods_days} days of hourly data for {total_stations} stations...")
    print(f"Date range: {timestamps[0]} to {timestamps[-1]}")

    for line_name, config in LINES.items():
        print(f"  Processing {line_name} ({config['num_stations']} stations)...")

        for station_idx in range(config["num_stations"]):
            station_type = _get_station_type(config, station_idx)
            station_id = f"{line_name[:6]}_{station_idx:02d}"

            for ts in timestamps:
                hour = ts.hour
                month = ts.month
                day_of_week = ts.dayofweek

                base = config["base_ridership"]
                hourly_mult = _hourly_pattern(hour)
                type_mult = _station_type_multiplier(station_type, hour)

                # Saudi weekend (Thursday=3, Friday=4)
                is_weekend = 1 if day_of_week in [3, 4] else 0
                if is_weekend:
                    if station_type == "business":
                        weekend_mult = 0.35
                    elif station_type == "residential":
                        weekend_mult = 1.1
                    else:
                        weekend_mult = 0.7
                else:
                    weekend_mult = 1.0

                prayer_mult = _is_prayer_time(hour, month)
                temp_mult = _temperature_effect(month)
                temperature = MONTHLY_TEMPS[month - 1] + np.random.normal(0, 3)

                is_ramadan = _is_ramadan(ts.to_pydatetime())
                if is_ramadan:
                    if hour >= 21 or hour <= 1:
                        ramadan_mult = 1.6
                    elif 5 <= hour <= 10:
                        ramadan_mult = 0.4
                    else:
                        ramadan_mult = 0.8
                else:
                    ramadan_mult = 1.0

                if _is_hajj_period(ts.to_pydatetime()):
                    hajj_mult = 0.85
                else:
                    hajj_mult = 1.0

                is_holiday = _is_eid(ts.to_pydatetime())
                if (month, ts.day) in FIXED_HOLIDAYS:
                    is_holiday = True
                if is_holiday:
                    holiday_mult = 0.5
                else:
                    holiday_mult = 1.0

                if not _is_school_period(ts.to_pydatetime()):
                    school_mult = 0.80
                else:
                    school_mult = 1.0

                rate = (base * hourly_mult * type_mult * weekend_mult
                        * prayer_mult * temp_mult * ramadan_mult * hajj_mult
                        * holiday_mult * school_mult)

                ridership = np.random.poisson(max(rate, 1))

                records.append({
                    "timestamp": ts,
                    "line": line_name,
                    "station_id": station_id,
                    "station_type": station_type,
                    "ridership": ridership,
                    "temperature": round(temperature, 1),
                    "is_weekend": is_weekend,
                    "is_holiday": int(is_holiday),
                    "is_ramadan": int(is_ramadan),
                })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    print(f"\nGenerated {len(df):,} records")
    print(f"Saved to {output_path}")
    print(f"\nRidership statistics:")
    print(df.groupby("line")["ridership"].describe().round(1))

    return df


def generate_line_level_data(output_path="data/ridership_by_line.csv"):
    """Generate aggregated line-level hourly ridership data.

    Aggregates station-level data to line totals for simpler modeling.
    If the full station data does not exist, generates it first.

    Parameters
    ----------
    output_path : str
        Path to save the aggregated CSV.

    Returns
    -------
    pandas.DataFrame
        DataFrame with hourly ridership totals per line.
    """
    station_path = Path("data/ridership.csv")
    if station_path.exists():
        print("Loading existing station-level data...")
        df = pd.read_csv(station_path, parse_dates=["timestamp"])
    else:
        print("Generating station-level data first...")
        df = generate_ridership_data()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    line_df = (
        df.groupby(["timestamp", "line"])
        .agg(
            ridership=("ridership", "sum"),
            avg_temperature=("temperature", "mean"),
            is_weekend=("is_weekend", "first"),
            is_holiday=("is_holiday", "first"),
            is_ramadan=("is_ramadan", "first"),
        )
        .reset_index()
    )

    line_df.to_csv(output_path, index=False)
    print(f"\nAggregated to {len(line_df):,} line-level records")
    print(f"Saved to {output_path}")

    return line_df


if __name__ == "__main__":
    generate_ridership_data()
    generate_line_level_data()
