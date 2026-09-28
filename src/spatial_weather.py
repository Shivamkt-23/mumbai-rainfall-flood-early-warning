# ============================================================

# MUMBAI SPATIAL WEATHER

# Live Open-Meteo weather for Mumbai monitoring locations

# ============================================================



from __future__ import annotations



from typing import Any, Dict, List



import math

import requests

import numpy as np

import pandas as pd





# ============================================================

# CONFIGURATION

# ============================================================



OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"



REQUEST_TIMEOUT = 20



# Nine monitoring locations used by the prototype.

MUMBAI_LOCATIONS: List[Dict[str, Any]] = [

    {

        "name": "Borivali",

        "latitude": 19.2307,

        "longitude": 72.8567,

    },

    {

        "name": "Andheri",

        "latitude": 19.1197,

        "longitude": 72.8468,

    },

    {

        "name": "Powai",

        "latitude": 19.1176,

        "longitude": 72.9060,

    },

    {

        "name": "Bandra",

        "latitude": 19.0607,

        "longitude": 72.8362,

    },

    {

        "name": "Central Mumbai",

        "latitude": 18.9894,

        "longitude": 72.8291,

    },

    {

        "name": "Thane",

        "latitude": 19.2183,

        "longitude": 72.9781,

    },

    {

        "name": "Worli",

        "latitude": 19.0178,

        "longitude": 72.8173,

    },

    {

        "name": "Dadar",

        "latitude": 19.0178,

        "longitude": 72.8478,

    },

    {

        "name": "Sion",

        "latitude": 19.0465,

        "longitude": 72.8636,

    },

]





# ============================================================

# REQUIRED MODEL FEATURES

# ============================================================



MODEL_FEATURES = [

    "rainfall_mm",

    "temperature_c",

    "dewpoint_c",

    "pressure_hpa",

    "u10_ms",

    "v10_ms",

    "wind_speed_ms",

    "humidity_pct",

    "rain_3h",

    "rain_6h",

    "rain_24h",

    "rainfall_lag_1h",

    "rainfall_lag_2h",

    "rainfall_lag_3h",

    "rainfall_lag_6h",

    "rainfall_lag_12h",

    "rainfall_lag_24h",

    "year",

    "month",

    "day",

    "hour",

    "day_of_year",

    "hour_sin",

    "hour_cos",

    "doy_sin",

    "doy_cos",

]





# ============================================================

# HTTP SESSION

# ============================================================



SESSION = requests.Session()



SESSION.headers.update(

    {

        "User-Agent": (

            "Mumbai-Rainfall-Flood-Early-Warning/"

            "1.0"

        )

    }

)





# ============================================================

# HELPERS

# ============================================================



def _safe_float(

    value: Any,

    default: float = 0.0,

) -> float:

    """Convert a value safely to float."""



    try:

        if value is None:

            return default



        result = float(value)



        if not math.isfinite(result):

            return default



        return result



    except (TypeError, ValueError):

        return default





def _clip_nonnegative(

    value: Any,

) -> float:

    """Return a finite non-negative float."""



    return max(

        0.0,

        _safe_float(value),

    )





def _calculate_uv_components(

    wind_speed_ms: float,

    wind_direction_deg: float,

) -> tuple[float, float]:

    """

    Convert wind speed + meteorological wind direction

    into approximate U10/V10 components.



    Meteorological convention:

        direction = direction wind is coming FROM.



    U component:

        positive = eastward



    V component:

        positive = northward

    """



    speed = _safe_float(

        wind_speed_ms

    )



    direction = _safe_float(

        wind_direction_deg

    )



    radians = math.radians(

        direction

    )



    u10 = -speed * math.sin(

        radians

    )



    v10 = -speed * math.cos(

        radians

    )



    return (

        float(u10),

        float(v10),

    )





# ============================================================

# FETCH ONE LOCATION

# ============================================================



def fetch_location_weather(

    latitude: float,

    longitude: float,

) -> Dict[str, Any]:

    """

    Fetch approximately two days of hourly weather data

    for one Mumbai location.

    """



    params = {

        "latitude": latitude,

        "longitude": longitude,

        "hourly": (

            "temperature_2m,"

            "relative_humidity_2m,"

            "dew_point_2m,"

            "surface_pressure,"

            "precipitation,"

            "wind_speed_10m,"

            "wind_direction_10m"

        ),

        "past_days": 2,

        "forecast_days": 1,

        "timezone": "Asia/Kolkata",

    }



    response = SESSION.get(

        OPEN_METEO_URL,

        params=params,

        timeout=REQUEST_TIMEOUT,

    )



    response.raise_for_status()



    data = response.json()



    if "hourly" not in data:

        raise RuntimeError(

            "Open-Meteo response does not contain hourly data."

        )



    return data





# ============================================================

# CONVERT API RESPONSE TO HOURLY DATAFRAME

# ============================================================



def _hourly_dataframe_from_response(

    weather_data: Dict[str, Any],

) -> pd.DataFrame:

    """Convert Open-Meteo hourly JSON to DataFrame."""



    hourly = weather_data.get(

        "hourly",

        {},

    )



    if not hourly:

        raise RuntimeError(

            "No hourly weather data returned."

        )



    times = hourly.get(

        "time",

        [],

    )



    if not times:

        raise RuntimeError(

            "Open-Meteo returned no hourly timestamps."

        )



    df = pd.DataFrame(

        {

            "datetime": times,



            "temperature_c": hourly.get(

                "temperature_2m",

                [],

            ),



            "humidity_pct": hourly.get(

                "relative_humidity_2m",

                [],

            ),



            "dewpoint_c": hourly.get(

                "dew_point_2m",

                [],

            ),



            "pressure_hpa": hourly.get(

                "surface_pressure",

                [],

            ),



            "rainfall_mm": hourly.get(

                "precipitation",

                [],

            ),



            "wind_speed_ms": hourly.get(

                "wind_speed_10m",

                [],

            ),



            "wind_direction_deg": hourly.get(

                "wind_direction_10m",

                [],

            ),

        }

    )



    # --------------------------------------------------------

    # Datetime

    # --------------------------------------------------------



    df["datetime"] = pd.to_datetime(

        df["datetime"],

        errors="coerce",

    )



    # --------------------------------------------------------

    # Numeric conversion

    # --------------------------------------------------------



    numeric_columns = [

        "temperature_c",

        "humidity_pct",

        "dewpoint_c",

        "pressure_hpa",

        "rainfall_mm",

        "wind_speed_ms",

        "wind_direction_deg",

    ]



    for column in numeric_columns:



        df[column] = pd.to_numeric(

            df[column],

            errors="coerce",

        )



    # --------------------------------------------------------

    # Invalid values

    # --------------------------------------------------------



    df[numeric_columns] = (

        df[numeric_columns]

        .replace(

            [

                np.inf,

                -np.inf,

            ],

            np.nan,

        )

        .fillna(0.0)

    )



    # --------------------------------------------------------

    # Physical clipping

    # --------------------------------------------------------



    df["rainfall_mm"] = df[

        "rainfall_mm"

    ].clip(

        lower=0.0

    )



    df["wind_speed_ms"] = df[

        "wind_speed_ms"

    ].clip(

        lower=0.0

    )



    df["humidity_pct"] = df[

        "humidity_pct"

    ].clip(

        lower=0.0,

        upper=100.0,

    )



    # --------------------------------------------------------

    # U10 / V10

    # --------------------------------------------------------



    uv_values = df.apply(

        lambda row: _calculate_uv_components(

            row["wind_speed_ms"],

            row["wind_direction_deg"],

        ),

        axis=1,

    )



    df["u10_ms"] = uv_values.apply(

        lambda x: x[0]

    )



    df["v10_ms"] = uv_values.apply(

        lambda x: x[1]

    )



    # --------------------------------------------------------

    # Sort chronologically

    # --------------------------------------------------------



    df = df.sort_values(

        "datetime"

    ).reset_index(

        drop=True

    )



    return df





# ============================================================

# BUILD RAINFALL FEATURES

# ============================================================



def _build_rainfall_features(

    df: pd.DataFrame,

) -> pd.DataFrame:

    """

    Build rolling rainfall totals and rainfall lag features.

    """



    work = df.copy()



    work = work.sort_values(

        "datetime"

    ).reset_index(

        drop=True

    )



    # --------------------------------------------------------

    # Rainfall lags

    # --------------------------------------------------------



    work[

        "rainfall_lag_1h"

    ] = work[

        "rainfall_mm"

    ].shift(

        1

    )



    work[

        "rainfall_lag_2h"

    ] = work[

        "rainfall_mm"

    ].shift(

        2

    )



    work[

        "rainfall_lag_3h"

    ] = work[

        "rainfall_mm"

    ].shift(

        3

    )



    work[

        "rainfall_lag_6h"

    ] = work[

        "rainfall_mm"

    ].shift(

        6

    )



    work[

        "rainfall_lag_12h"

    ] = work[

        "rainfall_mm"

    ].shift(

        12

    )



    work[

        "rainfall_lag_24h"

    ] = work[

        "rainfall_mm"

    ].shift(

        24

    )



    # --------------------------------------------------------

    # Rolling rainfall

    # --------------------------------------------------------



    work[

        "rain_3h"

    ] = (

        work[

            "rainfall_mm"

        ]

        .rolling(

            window=3,

            min_periods=1,

        )

        .sum()

    )



    work[

        "rain_6h"

    ] = (

        work[

            "rainfall_mm"

        ]

        .rolling(

            window=6,

            min_periods=1,

        )

        .sum()

    )



    work[

        "rain_24h"

    ] = (

        work[

            "rainfall_mm"

        ]

        .rolling(

            window=24,

            min_periods=1,

        )

        .sum()

    )



    # --------------------------------------------------------

    # Fill lag values safely

    # --------------------------------------------------------



    lag_columns = [

        "rainfall_lag_1h",

        "rainfall_lag_2h",

        "rainfall_lag_3h",

        "rainfall_lag_6h",

        "rainfall_lag_12h",

        "rainfall_lag_24h",

    ]



    for column in lag_columns:



        work[column] = (

            work[column]

            .fillna(

                0.0

            )

        )



    return work





# ============================================================

# BUILD TIME FEATURES

# ============================================================



def _build_time_features(

    df: pd.DataFrame,

) -> pd.DataFrame:

    """Build calendar and cyclical time features."""



    work = df.copy()



    work["year"] = (

        work["datetime"]

        .dt.year

    )



    work["month"] = (

        work["datetime"]

        .dt.month

    )



    work["day"] = (

        work["datetime"]

        .dt.day

    )



    work["hour"] = (

        work["datetime"]

        .dt.hour

    )



    work["day_of_year"] = (

        work["datetime"]

        .dt.dayofyear

    )



    # --------------------------------------------------------

    # Cyclical hour

    # --------------------------------------------------------



    work["hour_sin"] = np.sin(

        2.0

        * np.pi

        * work["hour"]

        / 24.0

    )



    work["hour_cos"] = np.cos(

        2.0

        * np.pi

        * work["hour"]

        / 24.0

    )



    # --------------------------------------------------------

    # Cyclical day of year

    # --------------------------------------------------------



    work["doy_sin"] = np.sin(

        2.0

        * np.pi

        * work["day_of_year"]

        / 365.0

    )



    work["doy_cos"] = np.cos(

        2.0

        * np.pi

        * work["day_of_year"]

        / 365.0

    )



    return work





# ============================================================

# EXTRACT LATEST LOCATION VALUES

# ============================================================



def _extract_location_values(

    weather_data: Dict[str, Any],

) -> Dict[str, Any]:

    """

    Convert the latest Open-Meteo observation into the

    flat dictionary expected by the application.

    """



    hourly_df = _hourly_dataframe_from_response(

        weather_data

    )



    hourly_df = _build_rainfall_features(

        hourly_df

    )



    hourly_df = _build_time_features(

        hourly_df

    )



    if hourly_df.empty:

        raise RuntimeError(

            "Hourly weather dataframe is empty."

        )



    now_local = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)

    available_now = hourly_df[hourly_df["datetime"] <= now_local]

    if available_now.empty:
        latest = hourly_df.iloc[-1]
    else:
        latest = available_now.iloc[-1]



    return {

        "datetime": latest[

            "datetime"

        ],



        "temperature_c": _safe_float(

            latest["temperature_c"]

        ),



        "humidity_pct": _safe_float(

            latest["humidity_pct"]

        ),



        "dewpoint_c": _safe_float(

            latest["dewpoint_c"]

        ),



        "pressure_hpa": _safe_float(

            latest["pressure_hpa"]

        ),



        "rainfall_mm": _clip_nonnegative(

            latest["rainfall_mm"]

        ),



        "wind_speed_ms": _clip_nonnegative(

            latest["wind_speed_ms"]

        ),



        "u10_ms": _safe_float(

            latest["u10_ms"]

        ),



        "v10_ms": _safe_float(

            latest["v10_ms"]

        ),



        "rain_3h": _clip_nonnegative(

            latest["rain_3h"]

        ),



        "rain_6h": _clip_nonnegative(

            latest["rain_6h"]

        ),



        "rain_24h": _clip_nonnegative(

            latest["rain_24h"]

        ),



        "rainfall_lag_1h": _clip_nonnegative(

            latest["rainfall_lag_1h"]

        ),



        "rainfall_lag_2h": _clip_nonnegative(

            latest["rainfall_lag_2h"]

        ),



        "rainfall_lag_3h": _clip_nonnegative(

            latest["rainfall_lag_3h"]

        ),



        "rainfall_lag_6h": _clip_nonnegative(

            latest["rainfall_lag_6h"]

        ),



        "rainfall_lag_12h": _clip_nonnegative(

            latest["rainfall_lag_12h"]

        ),



        "rainfall_lag_24h": _clip_nonnegative(

            latest["rainfall_lag_24h"]

        ),



        "year": int(

            _safe_float(

                latest["year"]

            )

        ),



        "month": int(

            _safe_float(

                latest["month"]

            )

        ),



        "day": int(

            _safe_float(

                latest["day"]

            )

        ),



        "hour": int(

            _safe_float(

                latest["hour"]

            )

        ),



        "day_of_year": int(

            _safe_float(

                latest["day_of_year"]

            )

        ),



        "hour_sin": _safe_float(

            latest["hour_sin"]

        ),



        "hour_cos": _safe_float(

            latest["hour_cos"]

        ),



        "doy_sin": _safe_float(

            latest["doy_sin"]

        ),



        "doy_cos": _safe_float(

            latest["doy_cos"]

        ),

    }





# ============================================================

# FETCH ALL MUMBAI LOCATIONS

# ============================================================



def fetch_all_locations_with_history() -> List[Dict[str, Any]]:
    """Fetch live weather and engineered hourly history for multisource inference."""

    results: List[Dict[str, Any]] = []

    for location in MUMBAI_LOCATIONS:
        try:
            weather_data = fetch_location_weather(
                latitude=location["latitude"],
                longitude=location["longitude"],
            )

            hourly = _hourly_dataframe_from_response(weather_data)
            hourly = _build_rainfall_features(hourly)
            hourly = _build_time_features(hourly)
            hourly = hourly.sort_values("datetime").reset_index(drop=True)

            values = _extract_location_values(weather_data)

            results.append({
                **location,
                **values,
                "hourly": hourly,
                "api_status": "connected",
            })

        except Exception as exc:
            results.append({
                **location,
                "datetime": None,
                "temperature_c": 0.0,
                "humidity_pct": 0.0,
                "dewpoint_c": 0.0,
                "pressure_hpa": 0.0,
                "rainfall_mm": 0.0,
                "wind_speed_ms": 0.0,
                "u10_ms": 0.0,
                "v10_ms": 0.0,
                "rain_3h": 0.0,
                "rain_6h": 0.0,
                "rain_24h": 0.0,
                "rainfall_lag_1h": 0.0,
                "rainfall_lag_2h": 0.0,
                "rainfall_lag_3h": 0.0,
                "rainfall_lag_6h": 0.0,
                "rainfall_lag_12h": 0.0,
                "rainfall_lag_24h": 0.0,
                "year": 0,
                "month": 0,
                "day": 0,
                "hour": 0,
                "day_of_year": 0,
                "hour_sin": 0.0,
                "hour_cos": 0.0,
                "doy_sin": 0.0,
                "doy_cos": 0.0,
                "hourly": None,
                "api_status": f"error: {exc}",
            })

    return results


def fetch_all_locations() -> List[Dict[str, Any]]:

    """

    Fetch live weather independently for all monitoring

    locations.

    """



    results: List[Dict[str, Any]] = []



    for location in MUMBAI_LOCATIONS:



        try:



            weather_data = fetch_location_weather(

                latitude=location["latitude"],

                longitude=location["longitude"],

            )



            values = _extract_location_values(

                weather_data

            )



            result = {

                **location,

                **values,

                "api_status": "connected",

            }



        except Exception as exc:



            result = {

                **location,



                "datetime": None,



                "temperature_c": 0.0,

                "humidity_pct": 0.0,

                "dewpoint_c": 0.0,

                "pressure_hpa": 0.0,



                "rainfall_mm": 0.0,



                "wind_speed_ms": 0.0,

                "u10_ms": 0.0,

                "v10_ms": 0.0,



                "rain_3h": 0.0,

                "rain_6h": 0.0,

                "rain_24h": 0.0,



                "rainfall_lag_1h": 0.0,

                "rainfall_lag_2h": 0.0,

                "rainfall_lag_3h": 0.0,

                "rainfall_lag_6h": 0.0,

                "rainfall_lag_12h": 0.0,

                "rainfall_lag_24h": 0.0,



                "year": 0,

                "month": 0,

                "day": 0,

                "hour": 0,

                "day_of_year": 0,



                "hour_sin": 0.0,

                "hour_cos": 0.0,

                "doy_sin": 0.0,

                "doy_cos": 0.0,



                "api_status": (

                    f"error: {exc}"

                ),

            }



        results.append(

            result

        )



    return results





# ============================================================

# MUMBAI SPATIAL WEATHER

# ============================================================



def fetch_mumbai_spatial_weather() -> pd.DataFrame:

    """

    Fetch all Mumbai locations and return a DataFrame

    ready for the application's spatial ML pipeline.

    """



    locations = fetch_all_locations()



    if not locations:

        raise RuntimeError(

            "No Mumbai weather locations were returned."

        )



    spatial = pd.DataFrame(

        locations

    )



    # --------------------------------------------------------

    # Ensure location column exists

    # --------------------------------------------------------



    if (

        "location" not in spatial.columns

        and "name" in spatial.columns

    ):



        spatial = spatial.rename(

            columns={

                "name": "location"

            }

        )



    # --------------------------------------------------------

    # Required location columns

    # --------------------------------------------------------



    required_columns = [

        "location",

        "latitude",

        "longitude",

        "datetime",

        *MODEL_FEATURES,

        "api_status",

    ]



    # --------------------------------------------------------

    # Create missing columns safely

    # --------------------------------------------------------



    for column in required_columns:



        if column not in spatial.columns:



            if column == "location":



                spatial[column] = "Unknown"



            elif column == "datetime":



                spatial[column] = pd.NaT



            elif column == "api_status":



                spatial[column] = "unknown"



            else:



                spatial[column] = 0.0



    # --------------------------------------------------------

    # Numeric columns

    # --------------------------------------------------------



    numeric_columns = [

        column

        for column in required_columns

        if column not in {

            "location",

            "datetime",

            "api_status",

        }

    ]



    for column in numeric_columns:



        spatial[column] = pd.to_numeric(

            spatial[column],

            errors="coerce",

        )



    spatial[numeric_columns] = (

        spatial[numeric_columns]

        .replace(

            [

                np.inf,

                -np.inf,

            ],

            np.nan,

        )

        .fillna(0.0)

    )



    # --------------------------------------------------------

    # Datetime

    # --------------------------------------------------------



    spatial["datetime"] = pd.to_datetime(

        spatial["datetime"],

        errors="coerce",

    )



    # --------------------------------------------------------

    # Keep rainfall non-negative

    # --------------------------------------------------------



    rainfall_columns = [

        "rainfall_mm",

        "rain_3h",

        "rain_6h",

        "rain_24h",

        "rainfall_lag_1h",

        "rainfall_lag_2h",

        "rainfall_lag_3h",

        "rainfall_lag_6h",

        "rainfall_lag_12h",

        "rainfall_lag_24h",

    ]



    for column in rainfall_columns:



        if column in spatial.columns:



            spatial[column] = spatial[

                column

            ].clip(

                lower=0.0

            )



    # --------------------------------------------------------

    # Final column ordering

    # --------------------------------------------------------



    ordered_columns = [

        "location",

        "latitude",

        "longitude",

        "datetime",

        *MODEL_FEATURES,

        "api_status",

    ]



    # Remove duplicates while preserving order.

    ordered_columns = list(

        dict.fromkeys(

            ordered_columns

        )

    )



    return spatial[

        ordered_columns

    ].copy()





# ============================================================

# FETCH HOURLY MUMBAI WEATHER

# ============================================================



def fetch_mumbai_hourly_weather() -> pd.DataFrame:

    """

    Fetch hourly weather for the central Mumbai location.



    This is useful for the application's 24-hour rainfall

    chart on the live-weather page.

    """



    center = MUMBAI_LOCATIONS[4]



    weather_data = fetch_location_weather(

        latitude=center["latitude"],

        longitude=center["longitude"],

    )



    df = _hourly_dataframe_from_response(

        weather_data

    )



    df = _build_rainfall_features(

        df

    )



    return df





# ============================================================

# QUICK CONNECTION TEST

# ============================================================



def test_connection() -> bool:

    """Return True when Open-Meteo is reachable."""



    try:



        test_location = MUMBAI_LOCATIONS[4]



        fetch_location_weather(

            latitude=test_location["latitude"],

            longitude=test_location["longitude"],

        )



        return True



    except Exception:



        return False





# ============================================================

# QUICK TEST

# ============================================================



if __name__ == "__main__":



    print(

        "=" * 70

    )



    print(

        "MUMBAI SPATIAL WEATHER TEST"

    )



    print(

        "=" * 70

    )



    locations = fetch_all_locations()



    for location in locations:



        print()



        print(

            f"Location: "

            f"{location['name']}"

        )



        print(

            f"Rainfall: "

            f"{location['rainfall_mm']:.2f} mm"

        )



        print(

            f"Rain 3h: "

            f"{location['rain_3h']:.2f} mm"

        )



        print(

            f"Rain 6h: "

            f"{location['rain_6h']:.2f} mm"

        )



        print(

            f"Rain 24h: "

            f"{location['rain_24h']:.2f} mm"

        )



        print(

            f"Temperature: "

            f"{location['temperature_c']:.2f} °C"

        )



        print(

            f"Humidity: "

            f"{location['humidity_pct']:.1f}%"

        )



        print(

            f"Wind speed: "

            f"{location['wind_speed_ms']:.2f} m/s"

        )



        print(

            f"U10: "

            f"{location['u10_ms']:.2f} m/s"

        )



        print(

            f"V10: "

            f"{location['v10_ms']:.2f} m/s"

        )



        print(

            f"Status: "

            f"{location['api_status']}"

        )



    print()



    print(

        "=" * 70

    )



    print(

        "BUILDING SPATIAL DATAFRAME"

    )



    print(

        "=" * 70

    )



    try:



        spatial_df = (

            fetch_mumbai_spatial_weather()

        )



        print()



        print(

            f"Rows: "

            f"{len(spatial_df)}"

        )



        print(

            f"Columns: "

            f"{len(spatial_df.columns)}"

        )



        print()



        print(

            "Columns:"

        )



        for column in spatial_df.columns:



            print(

                f"  - {column}"

            )



        print()



        print(

            spatial_df[

                [

                    "location",

                    "rainfall_mm",

                    "rain_3h",

                    "rain_6h",

                    "rain_24h",

                    "temperature_c",

                    "humidity_pct",

                    "wind_speed_ms",

                ]

            ].to_string(

                index=False

            )

        )



    except Exception as exc:



        print()



        print(

            "Spatial dataframe error:"

        )



        print(

            exc

        )