from datetime import timezone

import h5py
import numpy as np
import pandas as pd

from imerg_mumbai import (
    login_earthdata,
    find_granule,
    get_download_url,
    download_granule,
)


# ============================================================
# CONFIGURATION
# ============================================================

ERA5_FILE = (
    "data/processed/"
    "mumbai_era5_2010_2025_base.csv"
)

TARGET_LAT = 19.0
TARGET_LON = 73.0

# Same recent events used in the previous validation
EVENTS = [
    "2024-07-13 08:00:00+00:00",
    "2024-07-19 20:00:00+00:00",
    "2024-08-24 05:00:00+00:00",
    "2024-09-25 12:00:00+00:00",
    "2024-09-25 18:00:00+00:00",
    "2025-06-18 20:00:00+00:00",
    "2025-08-18 10:00:00+00:00",
    "2025-09-22 08:00:00+00:00",
    "2025-09-28 06:00:00+00:00",
    "2025-09-28 12:00:00+00:00",
]


# ============================================================
# LOAD ERA5
# ============================================================

print("Loading ERA5...")

df = pd.read_csv(
    ERA5_FILE,
    parse_dates=["datetime"]
)

df["datetime"] = pd.to_datetime(
    df["datetime"],
    utc=True
)


# ============================================================
# EXTRACT ONE IMERG HALF-HOUR SPATIAL FIELD
# ============================================================

def get_imerg_half_hour_spatial(start_time):
    """
    Return center, 3x3 mean and 3x3 maximum rainfall depth
    for one IMERG 30-minute granule.
    """

    granule = find_granule(start_time)

    url = get_download_url(granule)

    local_file = download_granule(url)

    with h5py.File(local_file, "r") as h5:

        lat = h5["Grid/lat"][...]
        lon = h5["Grid/lon"][...]

        precipitation = h5[
            "Grid/precipitation"
        ]

        # ----------------------------------------------------
        # Find nearest Mumbai pixel
        # ----------------------------------------------------

        lat_idx = int(
            np.abs(
                lat - TARGET_LAT
            ).argmin()
        )

        lon_idx = int(
            np.abs(
                lon - TARGET_LON
            ).argmin()
        )

        # ----------------------------------------------------
        # 3x3 neighborhood
        # ----------------------------------------------------

        lat_start = max(
            0,
            lat_idx - 1
        )

        lat_end = min(
            len(lat),
            lat_idx + 2
        )

        lon_start = max(
            0,
            lon_idx - 1
        )

        lon_end = min(
            len(lon),
            lon_idx + 2
        )

        values = np.asarray(
            precipitation[
                0,
                lon_start:lon_end,
                lat_start:lat_end
            ],
            dtype=float
        )

        # ----------------------------------------------------
        # Missing values
        # ----------------------------------------------------

        fill_value = (
            precipitation.attrs.get(
                "_FillValue",
                -9999.9
            )
        )

        values[
            np.isclose(
                values,
                float(fill_value),
                atol=0.01
            )
        ] = np.nan

        # ----------------------------------------------------
        # Center pixel
        # ----------------------------------------------------

        center = values[
            lat_idx - lat_start,
            lon_idx - lon_start
        ]

        # ----------------------------------------------------
        # Convert rate to 30-minute depth
        # ----------------------------------------------------

        center_depth = center * 0.5

        mean_depth = (
            np.nanmean(values)
            * 0.5
        )

        max_depth = (
            np.nanmax(values)
            * 0.5
        )

        return {
            "center_mm": float(center_depth),
            "mean_3x3_mm": float(mean_depth),
            "max_3x3_mm": float(max_depth),
        }


# ============================================================
# LOGIN
# ============================================================

print(
    "\nLogging into NASA Earthdata..."
)

login_earthdata()


# ============================================================
# VALIDATION LOOP
# ============================================================

results = []

for i, event_string in enumerate(
    EVENTS,
    start=1
):

    era5_time = pd.Timestamp(
        event_string
    )

    imerg_start = (
        era5_time
        - pd.Timedelta(hours=1)
    ).to_pydatetime()

    imerg_start = (
        imerg_start.astimezone(
            timezone.utc
        )
    )

    era5_row = df[
        df["datetime"] == era5_time
    ]

    if era5_row.empty:
        print(
            f"Event {i}: ERA5 record missing."
        )
        continue

    era5_rain = float(
        era5_row.iloc[0]["rainfall_mm"]
    )

    print()
    print("=" * 70)
    print(
        f"EVENT {i}/{len(EVENTS)}"
    )

    print(
        "ERA5 timestamp:",
        era5_time
    )

    print(
        "IMERG hour:",
        imerg_start
    )

    # --------------------------------------------------------
    # First half hour
    # --------------------------------------------------------

    first = get_imerg_half_hour_spatial(
        imerg_start
    )

    # --------------------------------------------------------
    # Second half hour
    # --------------------------------------------------------

    second = get_imerg_half_hour_spatial(
        imerg_start
        + pd.Timedelta(minutes=30)
    )

    # --------------------------------------------------------
    # Hourly totals
    # --------------------------------------------------------

    center_hourly = (
        first["center_mm"]
        + second["center_mm"]
    )

    mean_hourly = (
        first["mean_3x3_mm"]
        + second["mean_3x3_mm"]
    )

    max_hourly = (
        first["max_3x3_mm"]
        + second["max_3x3_mm"]
    )

    print(
        "ERA5:",
        round(era5_rain, 3),
        "mm"
    )

    print(
        "IMERG center:",
        round(center_hourly, 3),
        "mm"
    )

    print(
        "IMERG 3x3 mean:",
        round(mean_hourly, 3),
        "mm"
    )

    print(
        "IMERG 3x3 max:",
        round(max_hourly, 3),
        "mm"
    )

    results.append(
        {
            "era5_timestamp": era5_time,
            "era5_rain_mm": era5_rain,
            "imerg_center_mm": center_hourly,
            "imerg_mean_3x3_mm": mean_hourly,
            "imerg_max_3x3_mm": max_hourly,
        }
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    errors = (
        predicted - actual
    )

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )

    bias = np.mean(
        errors
    )

    correlation = (
        np.corrcoef(
            actual,
            predicted
        )[0, 1]
        if len(actual) > 1
        else np.nan
    )

    return (
        mae,
        rmse,
        bias,
        correlation
    )


# ============================================================
# CALCULATE CENTER METRICS
# ============================================================

center_metrics = calculate_metrics(
    results_df["era5_rain_mm"],
    results_df["imerg_center_mm"]
)


# ============================================================
# CALCULATE 3x3 MEAN METRICS
# ============================================================

mean_metrics = calculate_metrics(
    results_df["era5_rain_mm"],
    results_df["imerg_mean_3x3_mm"]
)


# ============================================================
# CALCULATE 3x3 MAX METRICS
# ============================================================

max_metrics = calculate_metrics(
    results_df["era5_rain_mm"],
    results_df["imerg_max_3x3_mm"]
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    "data/processed/"
    "imerg_era5_spatial_validation.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 75)
print("IMERG SPATIAL VALIDATION")
print("=" * 75)

print()
print("CENTER PIXEL")
print(
    "MAE:",
    round(center_metrics[0], 4),
    "mm"
)
print(
    "RMSE:",
    round(center_metrics[1], 4),
    "mm"
)
print(
    "Bias:",
    round(center_metrics[2], 4),
    "mm"
)
print(
    "Correlation:",
    round(center_metrics[3], 4)
)

print()
print("3x3 MEAN")
print(
    "MAE:",
    round(mean_metrics[0], 4),
    "mm"
)
print(
    "RMSE:",
    round(mean_metrics[1], 4),
    "mm"
)
print(
    "Bias:",
    round(mean_metrics[2], 4),
    "mm"
)
print(
    "Correlation:",
    round(mean_metrics[3], 4)
)

print()
print("3x3 MAX")
print(
    "MAE:",
    round(max_metrics[0], 4),
    "mm"
)
print(
    "RMSE:",
    round(max_metrics[1], 4),
    "mm"
)
print(
    "Bias:",
    round(max_metrics[2], 4),
    "mm"
)
print(
    "Correlation:",
    round(max_metrics[3], 4)
)

print()
print(
    "Saved:",
    output_file
)

print("=" * 75)