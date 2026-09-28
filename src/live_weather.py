# ============================================================
# LIVE MUMBAI WEATHER
# Open-Meteo + ML Feature Preparation
# ============================================================

from datetime import datetime
import time
from typing import Dict

import numpy as np
import pandas as pd
import requests


# ============================================================
# MUMBAI LOCATION
# ============================================================

MUMBAI_LATITUDE = 19.0760
MUMBAI_LONGITUDE = 72.8777

OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
)


LIVE_WEATHER_CACHE_TTL = 10 * 60

_LIVE_WEATHER_CACHE = None
_LIVE_WEATHER_CACHE_TIME = 0.0


# ============================================================
# FETCH LIVE WEATHER
# ============================================================

def fetch_live_mumbai_weather() -> Dict:
    """
    Fetch recent hourly Mumbai weather and prepare
    the latest row for the ML model.

    The returned dictionary contains:

        latest
            Latest usable model row.

        hourly
            Full recent hourly dataset for charts.

        fetched_at
            Local fetch timestamp.
    """

    # --------------------------------------------------------
    # Request enough history for the 24-hour lag feature.
    #
    # We keep the historical rows for the chart.
    # We do NOT drop them before creating the chart.
    # --------------------------------------------------------

    params = {

        "latitude":
            MUMBAI_LATITUDE,

        "longitude":
            MUMBAI_LONGITUDE,

        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "dew_point_2m,"
            "surface_pressure,"
            "rain,"
            "wind_speed_10m,"
            "wind_direction_10m"
        ),

        # 24 previous hours + current hour
        "past_hours": 24,

        # One future hour is useful for live context
        "forecast_hours": 1,

        "timezone":
            "Asia/Kolkata",

        "wind_speed_unit":
            "ms",

        "temperature_unit":
            "celsius",

        "precipitation_unit":
            "mm",
    }


    # --------------------------------------------------------
    # API REQUEST
    # --------------------------------------------------------

    global _LIVE_WEATHER_CACHE
    global _LIVE_WEATHER_CACHE_TIME

    now = time.monotonic()

    if (
        _LIVE_WEATHER_CACHE is not None
        and now - _LIVE_WEATHER_CACHE_TIME
        < LIVE_WEATHER_CACHE_TTL
    ):
        payload = _LIVE_WEATHER_CACHE

    else:
        last_error = None

        for attempt in range(3):
            try:
                response = requests.get(
                    OPEN_METEO_URL,
                    params=params,
                    timeout=15,
                )

                if response.status_code == 429:
                    retry_after = response.headers.get(
                        "Retry-After"
                    )

                    try:
                        wait_seconds = float(retry_after)
                    except (TypeError, ValueError):
                        wait_seconds = 2.0 * (attempt + 1)

                    wait_seconds = min(
                        max(wait_seconds, 1.0),
                        10.0,
                    )

                    last_error = RuntimeError(
                        f"Open-Meteo rate limit (HTTP 429). "
                        f"Retrying in {wait_seconds:.1f}s."
                    )

                    if attempt < 2:
                        time.sleep(wait_seconds)
                        continue

                    raise last_error

                response.raise_for_status()

                payload = response.json()

                _LIVE_WEATHER_CACHE = payload
                _LIVE_WEATHER_CACHE_TIME = time.monotonic()

                break

            except requests.RequestException as exc:
                last_error = exc

                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue

                raise RuntimeError(
                    f"Open-Meteo request failed: {exc}"
                ) from exc


    if "hourly" not in payload:

        raise ValueError(
            "Open-Meteo response does not contain "
            "hourly weather data."
        )


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    hourly = payload["hourly"]


    weather_df = pd.DataFrame(
        hourly
    )


    # ========================================================
    # DATETIME
    # ========================================================

    weather_df["datetime"] = (
        pd.to_datetime(
            weather_df["time"]
        )
    )


    weather_df = (
        weather_df
        .drop(
            columns=["time"]
        )
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # RENAME COLUMNS
    # ========================================================

    weather_df = weather_df.rename(
        columns={

            "temperature_2m":
                "temperature_c",

            "relative_humidity_2m":
                "humidity_pct",

            "dew_point_2m":
                "dewpoint_c",

            "surface_pressure":
                "pressure_hpa",

            "rain":
                "rainfall_mm",

            "wind_speed_10m":
                "wind_speed_ms",

            "wind_direction_10m":
                "wind_direction_deg",
        }
    )


    # ========================================================
    # WIND COMPONENTS
    # ========================================================

    direction_rad = np.deg2rad(
        weather_df[
            "wind_direction_deg"
        ]
    )


    speed = weather_df[
        "wind_speed_ms"
    ]


    weather_df["u10_ms"] = (
        -speed * np.sin(
            direction_rad
        )
    )


    weather_df["v10_ms"] = (
        -speed * np.cos(
            direction_rad
        )
    )


    # ========================================================
    # RAINFALL ACCUMULATIONS
    # ========================================================

    weather_df["rain_3h"] = (
        weather_df[
            "rainfall_mm"
        ]
        .rolling(
            window=3,
            min_periods=1
        )
        .sum()
    )


    weather_df["rain_6h"] = (
        weather_df[
            "rainfall_mm"
        ]
        .rolling(
            window=6,
            min_periods=1
        )
        .sum()
    )


    weather_df["rain_24h"] = (
        weather_df[
            "rainfall_mm"
        ]
        .rolling(
            window=24,
            min_periods=1
        )
        .sum()
    )


    # ========================================================
    # RAINFALL LAGS
    # ========================================================

    lag_hours = [
        1,
        2,
        3,
        6,
        12,
        24,
    ]


    for lag in lag_hours:

        weather_df[
            f"rainfall_lag_{lag}h"
        ] = (
            weather_df[
                "rainfall_mm"
            ]
            .shift(lag)
        )


    # ========================================================
    # TIME FEATURES
    # ========================================================

    weather_df["month"] = (
        weather_df[
            "datetime"
        ].dt.month
    )


    weather_df["hour"] = (
        weather_df[
            "datetime"
        ].dt.hour
    )


    weather_df["day_of_year"] = (
        weather_df[
            "datetime"
        ].dt.dayofyear
    )


    weather_df["hour_sin"] = np.sin(
        2
        * np.pi
        * weather_df["hour"]
        / 24
    )


    weather_df["hour_cos"] = np.cos(
        2
        * np.pi
        * weather_df["hour"]
        / 24
    )


    weather_df["doy_sin"] = np.sin(
        2
        * np.pi
        * weather_df[
            "day_of_year"
        ]
        / 365.25
    )


    weather_df["doy_cos"] = np.cos(
        2
        * np.pi
        * weather_df[
            "day_of_year"
        ]
        / 365.25
    )


    # ========================================================
    # FIND USABLE MODEL ROW
    # ========================================================
    #
    # The chart needs ALL rows.
    #
    # The model only needs the latest row with complete
    # 1h/2h/3h/6h/12h/24h lag information.
    #
    # Therefore we DO NOT drop rows from weather_df.
    # We only select the latest complete row.
    # ========================================================

    model_features = [

        "temperature_c",
        "humidity_pct",
        "dewpoint_c",
        "pressure_hpa",

        "rainfall_mm",

        "wind_speed_ms",
        "u10_ms",
        "v10_ms",

        "rain_3h",
        "rain_6h",
        "rain_24h",

        "rainfall_lag_1h",
        "rainfall_lag_2h",
        "rainfall_lag_3h",
        "rainfall_lag_6h",
        "rainfall_lag_12h",
        "rainfall_lag_24h",

        "month",
        "hour",
        "day_of_year",

        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ]


    usable = weather_df.dropna(
        subset=model_features
    )


    if usable.empty:

        raise ValueError(
            "Not enough historical hourly data "
            "to create the required ML features."
        )


    # ========================================================
    # LATEST MODEL ROW
    # ========================================================

    latest = (
        usable
        .iloc[-1]
        .copy()
    )


    # ========================================================
    # FETCH TIMESTAMP
    # ========================================================

    fetched_at = datetime.now()


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "latest":
            latest,

        "hourly":
            weather_df,

        "fetched_at":
            fetched_at,
    }