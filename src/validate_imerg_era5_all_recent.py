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


ERA5_FILE = (
    "data/processed/"
    "mumbai_era5_2010_2025_base.csv"
)

TARGET_LAT = 19.0
TARGET_LON = 73.0

MIN_RAIN_MM = 5.0


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
# ALL RECENT RAINY HOURS
# ============================================================

recent = df[
    (df["datetime"] >= "2024-01-01")
    & (df["datetime"] < "2025-10-01")
    & (df["rainfall_mm"] >= MIN_RAIN_MM)
].copy()

recent = recent.sort_values(
    "datetime"
)

print(
    "Recent rainy hours:",
    len(recent)
)


# ============================================================
# IMERG 30-MINUTE SPATIAL EXTRACTION
# ============================================================

def get_half_hour(
    start_time
):

    granule = find_granule(
        start_time
    )

    url = get_download_url(
        granule
    )

    local_file = download_granule(
        url
    )

    with h5py.File(
        local_file,
        "r"
    ) as h5:

        lat = h5["Grid/lat"][...]
        lon = h5["Grid/lon"][...]

        precipitation = h5[
            "Grid/precipitation"
        ]

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

        center = values[
            lat_idx - lat_start,
            lon_idx - lon_start
        ]

        mean_3x3 = np.nanmean(
            values
        )

        max_3x3 = np.nanmax(
            values
        )

        return {
            "center_mm": float(center * 0.5),
            "mean_3x3_mm": float(mean_3x3 * 0.5),
            "max_3x3_mm": float(max_3x3 * 0.5),
        }


# ============================================================
# LOGIN
# ============================================================

print(
    "\nLogging into NASA Earthdata..."
)

login_earthdata()


# ============================================================
# VALIDATION
# ============================================================

results = []

for i, (_, row) in enumerate(
    recent.iterrows(),
    start=1
):

    era5_time = row["datetime"]

    imerg_start = (
        era5_time
        - pd.Timedelta(hours=1)
    ).to_pydatetime()

    imerg_start = (
        imerg_start.astimezone(
            timezone.utc
        )
    )

    print()
    print("=" * 70)
    print(
        f"EVENT {i}/{len(recent)}"
    )

    print(
        "ERA5:",
        era5_time
    )

    try:

        first = get_half_hour(
            imerg_start
        )

        second = get_half_hour(
            imerg_start
            + pd.Timedelta(minutes=30)
        )

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

        results.append(
            {
                "datetime": era5_time,
                "era5_mm": float(
                    row["rainfall_mm"]
                ),
                "imerg_center_mm":
                    center_hourly,
                "imerg_mean_3x3_mm":
                    mean_hourly,
                "imerg_max_3x3_mm":
                    max_hourly,
            }
        )

        print(
            "ERA5:",
            round(
                float(row["rainfall_mm"]),
                3
            )
        )

        print(
            "IMERG center:",
            round(
                center_hourly,
                3
            )
        )

        print(
            "IMERG 3x3 mean:",
            round(
                mean_hourly,
                3
            )
        )

        print(
            "IMERG 3x3 max:",
            round(
                max_hourly,
                3
            )
        )

    except Exception as e:

        print(
            "ERROR:",
            e
        )


# ============================================================
# DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


if results_df.empty:

    raise RuntimeError(
        "No validation results produced."
    )


# ============================================================
# METRICS
# ============================================================

def metrics(
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

    error = (
        predicted - actual
    )

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    bias = np.mean(
        error
    )

    correlation = np.corrcoef(
        actual,
        predicted
    )[0, 1]

    return (
        mae,
        rmse,
        bias,
        correlation
    )


center = metrics(
    results_df["era5_mm"],
    results_df["imerg_center_mm"]
)

mean_3x3 = metrics(
    results_df["era5_mm"],
    results_df["imerg_mean_3x3_mm"]
)

max_3x3 = metrics(
    results_df["era5_mm"],
    results_df["imerg_max_3x3_mm"]
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    "data/processed/"
    "imerg_era5_all_recent_validation.csv"
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
print("ALL RECENT RAINY HOURS — IMERG vs ERA5")
print("=" * 75)

print(
    "Completed:",
    len(results_df)
)

print()
print("CENTER PIXEL")

print(
    "MAE:",
    round(center[0], 4),
    "mm"
)

print(
    "RMSE:",
    round(center[1], 4),
    "mm"
)

print(
    "Bias:",
    round(center[2], 4),
    "mm"
)

print(
    "Correlation:",
    round(center[3], 4)
)

print()
print("3x3 MEAN")

print(
    "MAE:",
    round(mean_3x3[0], 4),
    "mm"
)

print(
    "RMSE:",
    round(mean_3x3[1], 4),
    "mm"
)

print(
    "Bias:",
    round(mean_3x3[2], 4),
    "mm"
)

print(
    "Correlation:",
    round(mean_3x3[3], 4)
)

print()
print("3x3 MAX")

print(
    "MAE:",
    round(max_3x3[0], 4),
    "mm"
)

print(
    "RMSE:",
    round(max_3x3[1], 4),
    "mm"
)

print(
    "Bias:",
    round(max_3x3[2], 4),
    "mm"
)

print(
    "Correlation:",
    round(max_3x3[3], 4)
)

print()
print(
    "Saved:",
    output_file
)

print("=" * 75)