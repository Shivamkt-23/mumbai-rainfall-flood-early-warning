from pathlib import Path

import earthaccess
import h5py
import numpy as np
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ERA5_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mumbai_era5_2010_2025_ml_ready.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "GPM_IMERG"
    / "comparison_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# TEST TIME
# =========================================================

TEST_START = "2024-09-25T17:00:00"
TEST_END = "2024-09-25T17:59:59"
ERA5_TIMESTAMP = "2024-09-25T18:00:00"

TARGET_LAT = 19.0
TARGET_LON = 73.0


# =========================================================
# AUTHENTICATION
# =========================================================

print("Authenticating with NASA Earthdata...")

auth = earthaccess.login(
    strategy="environment"
)

if not auth.authenticated:
    raise RuntimeError(
        "NASA Earthdata authentication failed."
    )

print("Authentication successful.")


# =========================================================
# SEARCH IMERG
# =========================================================

print("\nSearching IMERG granules...")

results = earthaccess.search_data(
    concept_id="C2723754847-GES_DISC",
    temporal=(
        TEST_START,
        TEST_END,
    ),
    bounding_box=(
        72.7,
        18.8,
        73.1,
        19.3,
    ),
    count=2,
)

if len(results) != 2:
    raise RuntimeError(
        f"Expected 2 IMERG granules, found {len(results)}"
    )

print(
    f"Found {len(results)} granules."
)


# =========================================================
# DOWNLOAD
# =========================================================

print("\nDownloading IMERG granules...")

files = earthaccess.download(
    results,
    local_path=str(OUTPUT_DIR),
    threads=1,
)

print(
    f"Downloaded {len(files)} files."
)


# =========================================================
# PARSE FILENAME TIME
# =========================================================

def parse_time(file_path):

    filename = Path(file_path).name

    stamp = filename.split(".")[4]

    date_part = (
        stamp.split("-S")[0]
    )

    time_part = (
        stamp
        .split("-S")[1]
        .split("-E")[0]
    )

    return pd.to_datetime(
        date_part + time_part,
        format="%Y%m%d%H%M%S",
        utc=True,
    )


# =========================================================
# EXTRACT IMERG PIXEL
# =========================================================

def extract_pixel(file_path):

    with h5py.File(
        file_path,
        "r"
    ) as hdf:

        lat = hdf[
            "/Grid/lat"
        ][:]

        lon = hdf[
            "/Grid/lon"
        ][:]

        lat_index = np.abs(
            lat - TARGET_LAT
        ).argmin()

        lon_index = np.abs(
            lon - TARGET_LON
        ).argmin()

        rate = float(
            hdf[
                "/Grid/precipitation"
            ][
                0,
                lon_index,
                lat_index,
            ]
        )

        if rate <= -9999:
            rate = np.nan

        selected_lat = float(
            lat[lat_index]
        )

        selected_lon = float(
            lon[lon_index]
        )

    return (
        rate,
        selected_lat,
        selected_lon,
    )


# =========================================================
# BUILD IMERG DATAFRAME
# =========================================================

rows = []

for file_path in files:

    rate, lat, lon = extract_pixel(
        file_path
    )

    start_time = parse_time(
        file_path
    )

    rows.append(
        {
            "datetime": start_time,
            "file": Path(file_path).name,
            "lat": lat,
            "lon": lon,
            "rate_mm_per_hour": rate,
            "depth_mm": (
                rate * 0.5
                if not np.isnan(rate)
                else np.nan
            ),
        }
    )


imerg = pd.DataFrame(
    rows
)

imerg = (
    imerg
    .sort_values("datetime")
    .reset_index(drop=True)
)


# =========================================================
# HOURLY IMERG
# =========================================================

imerg["hour"] = (
    imerg["datetime"]
    .dt.floor("h")
)

imerg_hourly = (
    imerg
    .groupby("hour", as_index=False)[
        "depth_mm"
    ]
    .sum()
    .rename(
        columns={
            "hour": "datetime",
            "depth_mm":
                "imerg_hourly_rain_mm",
        }
    )
)


# =========================================================
# LOAD ERA5
# =========================================================

era5 = pd.read_csv(
    ERA5_PATH
)

era5["datetime"] = pd.to_datetime(
    era5["datetime"],
    errors="coerce",
    utc=True,
)

era5_row = era5[
    era5["datetime"]
    == pd.Timestamp(
        ERA5_TIMESTAMP,
        tz="UTC"
    )
]
if era5_row.empty:
    raise RuntimeError(
        "ERA5 row for test time not found."
    )

era5_rain = float(
    era5_row.iloc[0]["rainfall_mm"]
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 70)
print("30-MINUTE IMERG VALUES")
print("=" * 70)

print(
    imerg[
        [
            "datetime",
            "lat",
            "lon",
            "rate_mm_per_hour",
            "depth_mm",
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 70)
print("HOURLY COMPARISON")
print("=" * 70)

print(
    "Hourly interval:",
    TEST_START,
    "to",
    TEST_END,
)

print(
    "ERA5 validity time:",
    ERA5_TIMESTAMP,
)
print(
    f"ERA5 rainfall : {era5_rain:.6f} mm"
)

print(
    f"IMERG rainfall: "
    f"{imerg_hourly.iloc[0]['imerg_hourly_rain_mm']:.6f} mm"
)

print(
    f"Difference    : "
    f"{imerg_hourly.iloc[0]['imerg_hourly_rain_mm'] - era5_rain:.6f} mm"
)

print("=" * 70)