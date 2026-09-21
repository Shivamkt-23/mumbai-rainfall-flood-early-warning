import pandas as pd

def prepare_features(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    defaults = {
        "temperature_c": 28.0,
        "humidity_pct": 75.0,
        "pressure_hpa": 1005.0,
        "wind_speed_ms": 3.0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce").fillna(0).clip(lower=0)

    df["rain_3h"] = df["rainfall_mm"].rolling(3, min_periods=1).sum()
    df["rain_6h"] = df["rainfall_mm"].rolling(6, min_periods=1).sum()
    df["rain_24h"] = df["rainfall_mm"].rolling(24, min_periods=1).sum()
    df["next_hour_rain"] = df["rainfall_mm"].shift(-1)

    feature_cols = [
        "rainfall_mm", "rain_3h", "rain_6h", "rain_24h",
        "temperature_c", "humidity_pct", "pressure_hpa", "wind_speed_ms"
    ]
    return df.dropna(subset=feature_cols + ["next_hour_rain"]).copy(), feature_cols
