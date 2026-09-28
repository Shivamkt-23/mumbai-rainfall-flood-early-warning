# ============================================================
# MUMBAI NWP FORECAST
# ECMWF IFS HRES via Open-Meteo
# ============================================================

from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
import requests


# ============================================================
# MUMBAI LOCATION
# ============================================================

MUMBAI_LATITUDE = 19.0760
MUMBAI_LONGITUDE = 72.8777

ECMWF_URL = "https://api.open-meteo.com/v1/ecmwf"

TIMEZONE = "Asia/Kolkata"

MODEL_NAME = "ECMWF IFS HRES 9 km"


# ============================================================
# FETCH NWP FORECAST
# ============================================================

def fetch_mumbai_nwp_forecast(
    forecast_hours: int = 48,
) -> Dict:
    """
    Fetch the latest ECMWF IFS HRES forecast for Mumbai.

    Parameters
    ----------
    forecast_hours:
        Number of future hourly forecast rows to return.

    Returns
    -------
    dict
        forecast:
            Clean future hourly forecast DataFrame.

        fetched_at:
            Local timestamp when the request was made.

        model:
            Forecast model name.

        source:
            Forecast data source.

        latitude:
            Mumbai latitude.

        longitude:
            Mumbai longitude.
    """

    # --------------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------------

    if forecast_hours < 1:
        raise ValueError(
            "forecast_hours must be at least 1."
        )

    if forecast_hours > 72:
        raise ValueError(
            "forecast_hours cannot be greater than 72."
        )


    # ========================================================
    # API PARAMETERS
    # ========================================================

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
            "precipitation,"
            "rain,"
            "wind_speed_10m,"
            "wind_direction_10m"
        ),

        # Request enough forecast days so that we always
        # have sufficient future hourly rows.
        "forecast_days":
            3,

        "timezone":
            TIMEZONE,

        "wind_speed_unit":
            "ms",

        "temperature_unit":
            "celsius",

        "precipitation_unit":
            "mm",
    }


    # ========================================================
    # API REQUEST
    # ========================================================

    fetched_at = datetime.now()

    response = requests.get(
        ECMWF_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()


    # ========================================================
    # VALIDATE RESPONSE
    # ========================================================

    if "hourly" not in payload:
        raise ValueError(
            "ECMWF Open-Meteo response does not contain "
            "hourly forecast data."
        )

    hourly = payload["hourly"]


    required_columns = [
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "surface_pressure",
        "precipitation",
        "rain",
        "wind_speed_10m",
        "wind_direction_10m",
    ]


    missing_columns = [
        column
        for column in required_columns
        if column not in hourly
    ]


    if missing_columns:
        raise ValueError(
            "Missing expected ECMWF forecast fields: "
            + ", ".join(missing_columns)
        )


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    forecast_df = pd.DataFrame(hourly)


    # ========================================================
    # DATETIME
    # ========================================================

    forecast_df["datetime"] = pd.to_datetime(
        forecast_df["time"]
    )


    forecast_df = (
        forecast_df
        .drop(columns=["time"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )


    # ========================================================
    # RENAME VARIABLES
    # ========================================================

    forecast_df = forecast_df.rename(
        columns={

            "temperature_2m":
                "forecast_temperature_c",

            "relative_humidity_2m":
                "forecast_humidity_pct",

            "dew_point_2m":
                "forecast_dewpoint_c",

            "surface_pressure":
                "forecast_pressure_hpa",

            "precipitation":
                "forecast_precipitation_mm",

            "rain":
                "forecast_rain_mm",

            "wind_speed_10m":
                "forecast_wind_speed_ms",

            "wind_direction_10m":
                "forecast_wind_direction_deg",
        }
    )


    # ========================================================
    # WIND COMPONENTS
    # ========================================================

    direction_rad = np.deg2rad(
        forecast_df[
            "forecast_wind_direction_deg"
        ]
    )


    speed = forecast_df[
        "forecast_wind_speed_ms"
    ]


    forecast_df["forecast_u10_ms"] = (
        -speed * np.sin(direction_rad)
    )


    forecast_df["forecast_v10_ms"] = (
        -speed * np.cos(direction_rad)
    )


    # ========================================================
    # CURRENT LOCAL HOUR
    # ========================================================

    now_local = (
        datetime.now()
        .replace(
            minute=0,
            second=0,
            microsecond=0,
        )
    )


    end_time = (
        now_local
        + timedelta(
            hours=forecast_hours
        )
    )


    # ========================================================
    # KEEP FUTURE FORECAST ONLY
    # ========================================================

    forecast_df = forecast_df[
        (forecast_df["datetime"] >= now_local)
        &
        (forecast_df["datetime"] < end_time)
    ].copy()


    if forecast_df.empty:
        raise ValueError(
            "No future ECMWF forecast rows were returned."
        )


    # ========================================================
    # RESET INDEX
    # ========================================================

    forecast_df = (
        forecast_df
        .sort_values("datetime")
        .reset_index(drop=True)
    )


    # ========================================================
    # RAINFALL SERIES
    # ========================================================

    rain = (
        forecast_df[
            "forecast_precipitation_mm"
        ]
        .fillna(0.0)
        .astype(float)
    )


    # ========================================================
    # FORWARD-LOOKING FORECAST ACCUMULATIONS
    # ========================================================
    #
    # These are intentionally forward-looking.
    #
    # Example:
    #
    # At 19:00:
    #   next 3h =
    #   19:00 + 20:00 + 21:00
    #
    # At 20:00:
    #   next 3h =
    #   20:00 + 21:00 + 22:00
    #
    # This is appropriate for an early-warning system.
    # ========================================================

    rain_values = (
        rain.to_numpy(
            dtype=float
        )
    )


    def forward_sum(
        values: np.ndarray,
        hours: int,
    ) -> np.ndarray:
        """
        Calculate rainfall accumulation from each
        forecast timestamp forward.

        At the end of the available forecast period,
        only the remaining available hours are summed.
        """

        result = np.zeros(
            len(values),
            dtype=float,
        )

        for i in range(len(values)):

            end_index = min(
                i + hours,
                len(values),
            )

            result[i] = (
                values[i:end_index]
                .sum()
            )

        return result


    forecast_df[
        "forecast_next_1h_mm"
    ] = forward_sum(
        rain_values,
        1,
    )


    forecast_df[
        "forecast_next_3h_mm"
    ] = forward_sum(
        rain_values,
        3,
    )


    forecast_df[
        "forecast_next_6h_mm"
    ] = forward_sum(
        rain_values,
        6,
    )


    forecast_df[
        "forecast_next_24h_mm"
    ] = forward_sum(
        rain_values,
        24,
    )


    # ========================================================
    # FORECAST EXTREMES
    # ========================================================

    max_hourly_rain = float(
        rain.max()
    )


    max_rain_index = rain.idxmax()


    peak_time = forecast_df.loc[
        max_rain_index,
        "datetime",
    ]


    forecast_df[
        "forecast_max_hourly_rain_mm"
    ] = max_hourly_rain


    forecast_df[
        "forecast_peak_time"
    ] = peak_time


    # ========================================================
    # METADATA
    # ========================================================

    forecast_df[
        "forecast_model"
    ] = MODEL_NAME


    forecast_df[
        "forecast_source"
    ] = "Open-Meteo ECMWF API"


    # ========================================================
    # FINAL COLUMN ORDER
    # ========================================================

    column_order = [

        "datetime",

        # Immediate forecast precipitation
        "forecast_precipitation_mm",

        "forecast_rain_mm",

        # Atmospheric variables
        "forecast_temperature_c",

        "forecast_humidity_pct",

        "forecast_dewpoint_c",

        "forecast_pressure_hpa",

        # Wind
        "forecast_wind_speed_ms",

        "forecast_wind_direction_deg",

        "forecast_u10_ms",

        "forecast_v10_ms",

        # Forward rainfall accumulation
        "forecast_next_1h_mm",

        "forecast_next_3h_mm",

        "forecast_next_6h_mm",

        "forecast_next_24h_mm",

        # Forecast extremes
        "forecast_max_hourly_rain_mm",

        "forecast_peak_time",

        # Metadata
        "forecast_model",

        "forecast_source",
    ]


    forecast_df = forecast_df[
        column_order
    ].reset_index(drop=True)


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "forecast":
            forecast_df,

        "fetched_at":
            fetched_at,

        "model":
            MODEL_NAME,

        "source":
            "Open-Meteo ECMWF API",

        "latitude":
            MUMBAI_LATITUDE,

        "longitude":
            MUMBAI_LONGITUDE,
    }


# ============================================================
# SIMPLE LOCAL TEST
# ============================================================

if __name__ == "__main__":

    result = fetch_mumbai_nwp_forecast(
        forecast_hours=48
    )


    forecast = result[
        "forecast"
    ]


    print()

    print(
        "=" * 80
    )

    print(
        "MUMBAI NWP FORECAST"
    )

    print(
        "=" * 80
    )


    print(
        f"Model   : {result['model']}"
    )


    print(
        f"Source  : {result['source']}"
    )


    print(
        f"Fetched : {result['fetched_at']}"
    )


    print(
        f"Rows    : {len(forecast)}"
    )


    print(
        f"Start   : "
        f"{forecast['datetime'].min()}"
    )


    print(
        f"End     : "
        f"{forecast['datetime'].max()}"
    )


    print()

    print(
        "FORECAST SUMMARY"
    )

    print(
        "-" * 80
    )


    print(
        f"Next 1h rainfall : "
        f"{forecast.iloc[0]['forecast_next_1h_mm']:.2f} mm"
    )


    print(
        f"Next 3h rainfall : "
        f"{forecast.iloc[0]['forecast_next_3h_mm']:.2f} mm"
    )


    print(
        f"Next 6h rainfall : "
        f"{forecast.iloc[0]['forecast_next_6h_mm']:.2f} mm"
    )


    print(
        f"Next 24h rainfall: "
        f"{forecast.iloc[0]['forecast_next_24h_mm']:.2f} mm"
    )


    print(
        f"Maximum hourly  : "
        f"{forecast['forecast_max_hourly_rain_mm'].iloc[0]:.2f} mm"
    )


    print(
        f"Peak rainfall at: "
        f"{forecast['forecast_peak_time'].iloc[0]}"
    )


    print()

    print(
        "FIRST 10 FORECAST ROWS"
    )

    print(
        "-" * 80
    )


    display_columns = [

        "datetime",

        "forecast_precipitation_mm",

        "forecast_next_3h_mm",

        "forecast_next_6h_mm",

        "forecast_next_24h_mm",

        "forecast_temperature_c",

        "forecast_pressure_hpa",

        "forecast_wind_speed_ms",
    ]


    print(
        forecast[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


    print()

    print(
        "=" * 80
    )