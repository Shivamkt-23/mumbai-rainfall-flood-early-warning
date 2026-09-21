from pathlib import Path

import earthaccess
import h5py
import numpy as np
import pandas as pd


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
# SEARCH 4 HALF-HOUR GRANULES
# =========================================================

print("\nSearching IMERG granules...")

results = earthaccess.search_data(
    concept_id="C2723754847-GES_DISC",
    temporal=(
        "2024-09-01T00:00:00",
        "2024-09-01T01:59:59",
    ),
    bounding_box=(
        72.7,
        18.8,
        73.1,
        19.3,
    ),
    count=4,
)

if len(results) != 4:
    raise RuntimeError(
        f"Expected 4 granules, found {len(results)}"
    )

print(
    f"Found {len(results)} granules."
)


# =========================================================
# DOWNLOAD
# =========================================================

OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "GPM_IMERG"
    / "test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("\nDownloading test granules...")

files = earthaccess.download(
    results,
    local_path=str(OUTPUT_DIR),
    threads=1,
)

print(
    f"Downloaded {len(files)} files."
)


# =========================================================
# TARGET LOCATION
# =========================================================

TARGET_LAT = 19.0
TARGET_LON = 73.0


# =========================================================
# PARSE IMERG FILENAME TIME
# =========================================================

def parse_time(file_path):

    filename = Path(file_path).name

    # Example:
    #
    # 3B-HHR.MS.MRG.3IMERG.
    # 20240901-S000000-E002959.0000.V07B.HDF5
    #
    # Extract:
    # 20240901
    # 000000

    stamp = filename.split(".")[4]

    date_part = stamp.split("-S")[0]

    time_part = (
        stamp
        .split("-S")[1]
        .split("-E")[0]
    )

    timestamp = (
        date_part
        + time_part
    )

    return pd.to_datetime(
        timestamp,
        format="%Y%m%d%H%M%S",
        utc=True,
    )


# =========================================================
# EXTRACT ONE GRANULE
# =========================================================

def extract_mumbai_value(file_path):

    with h5py.File(file_path, "r") as hdf:

        lat = hdf["/Grid/lat"][:]
        lon = hdf["/Grid/lon"][:]

        lat_index = np.abs(
            lat - TARGET_LAT
        ).argmin()

        lon_index = np.abs(
            lon - TARGET_LON
        ).argmin()

        value = float(
            hdf["/Grid/precipitation"][
                0,
                lon_index,
                lat_index,
            ]
        )

        # IMERG missing-value handling
        if value <= -9999:
            value = np.nan

    return value


# =========================================================
# EXTRACT ALL GRANULES
# =========================================================

rows = []

for file_path in sorted(files):

    file_path = Path(file_path)

    rate = extract_mumbai_value(
        file_path
    )

    timestamp = parse_time(
        file_path
    )

    rows.append(
        {
            "datetime": timestamp,
            "file": file_path.name,
            "precip_rate_mm_per_hour": rate,
            "rain_depth_mm": (
                rate * 0.5
                if not np.isnan(rate)
                else np.nan
            ),
        }
    )


df = pd.DataFrame(
    rows
)

df = (
    df.sort_values("datetime")
    .reset_index(drop=True)
)


# =========================================================
# 30-MINUTE RESULTS
# =========================================================

print("\n" + "=" * 70)
print("IMERG 30-MINUTE RAINFALL")
print("=" * 70)

print(
    df[
        [
            "datetime",
            "precip_rate_mm_per_hour",
            "rain_depth_mm",
        ]
    ].to_string(index=False)
)


# =========================================================
# HOURLY AGGREGATION
# =========================================================

df["hour"] = (
    df["datetime"]
    .dt.floor("h")
)

hourly = (
    df.groupby("hour", as_index=False)[
        "rain_depth_mm"
    ]
    .sum()
    .rename(
        columns={
            "hour": "datetime",
            "rain_depth_mm":
                "imerg_hourly_rain_mm",
        }
    )
)


print("\n" + "=" * 70)
print("IMERG HOURLY RAINFALL")
print("=" * 70)

print(
    hourly.to_string(index=False)
)