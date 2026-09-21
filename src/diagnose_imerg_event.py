from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from imerg_mumbai import (
    login_earthdata,
    find_granule,
    get_download_url,
    download_granule,
)


# ============================================================
# EVENT
# ============================================================

# ERA5 timestamp = 2011-06-10 00:00 UTC
# Corresponding IMERG hour = 2011-06-09 23:00 -> 00:00 UTC

START_TIME = datetime(
    2011,
    6,
    9,
    23,
    0,
    tzinfo=timezone.utc,
)


TARGET_LAT = 19.0
TARGET_LON = 73.0


# ============================================================
# LOGIN
# ============================================================

print("Logging into NASA Earthdata...")

login_earthdata()


# ============================================================
# FIND TWO IMERG GRANULES
# ============================================================

granule_times = [
    START_TIME,
    START_TIME.replace(minute=30)
]


files = []

for t in granule_times:

    print()
    print("=" * 60)
    print("Searching IMERG:", t)
    print("=" * 60)

    granule = find_granule(t)

    url = get_download_url(granule)

    local_file = download_granule(url)

    files.append(local_file)

    print("File:", local_file)


# ============================================================
# PROCESS EACH HALF-HOUR
# ============================================================

hourly_depths = []

for file_path in files:

    print()
    print("=" * 60)
    print("FILE:", file_path.name)
    print("=" * 60)

    with h5py.File(file_path, "r") as h5:

        lat = h5["Grid/lat"][...]
        lon = h5["Grid/lon"][...]

        precipitation = h5[
            "Grid/precipitation"
        ]

        time_values = h5[
            "Grid/time"
        ][...]

        # ----------------------------------------------------
        # Nearest Mumbai pixel
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

        print(
            "Mumbai pixel:",
            float(lat[lat_idx]),
            float(lon[lon_idx])
        )

        print(
            "Latitude index:",
            lat_idx
        )

        print(
            "Longitude index:",
            lon_idx
        )

        # ----------------------------------------------------
        # Time metadata
        # ----------------------------------------------------

        print(
            "\nGrid time value:"
        )

        print(time_values)

        print(
            "Time attributes:"
        )

        print(
            h5["Grid/time"].attrs
        )

        # ----------------------------------------------------
        # 3x3 neighbourhood
        #
        # 3 pixels in each direction around Mumbai
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

        neighborhood = precipitation[
            0,
            lon_start:lon_end,
            lat_start:lat_end
        ]

        neighborhood = np.asarray(
            neighborhood,
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

        neighborhood[
            np.isclose(
                neighborhood,
                fill_value,
                atol=0.01
            )
        ] = np.nan

        # ----------------------------------------------------
        # Print neighbourhood
        # ----------------------------------------------------

        print(
            "\n3x3 IMERG rainfall-rate neighbourhood"
        )

        print(
            "Shape:",
            neighborhood.shape
        )

        print(
            neighborhood
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        center_value = neighborhood[
            lon_idx - lon_start,
            lat_idx - lat_start
        ]

        neighborhood_mean = np.nanmean(
            neighborhood
        )

        neighborhood_max = np.nanmax(
            neighborhood
        )

        print(
            "\nCenter pixel rate:",
            center_value,
            "mm/hr"
        )

        print(
            "3x3 mean rate:",
            neighborhood_mean,
            "mm/hr"
        )

        print(
            "3x3 max rate:",
            neighborhood_max,
            "mm/hr"
        )

        # Convert rate to 30-min depth
        center_depth = (
            center_value * 0.5
        )

        mean_depth = (
            neighborhood_mean * 0.5
        )

        max_depth = (
            neighborhood_max * 0.5
        )

        print(
            "\nCenter 30-min depth:",
            center_depth,
            "mm"
        )

        print(
            "3x3 mean 30-min depth:",
            mean_depth,
            "mm"
        )

        print(
            "3x3 max 30-min depth:",
            max_depth,
            "mm"
        )

        hourly_depths.append(
            {
                "center": center_depth,
                "mean": mean_depth,
                "max": max_depth,
            }
        )


# ============================================================
# HOURLY TOTALS
# ============================================================

center_hourly = sum(
    x["center"]
    for x in hourly_depths
)

mean_hourly = sum(
    x["mean"]
    for x in hourly_depths
)

max_hourly = sum(
    x["max"]
    for x in hourly_depths
)


print()
print("=" * 60)
print("EVENT SUMMARY")
print("=" * 60)

print(
    "Center-pixel hourly rainfall:",
    center_hourly,
    "mm"
)

print(
    "3x3 mean hourly rainfall:",
    mean_hourly,
    "mm"
)

print(
    "3x3 max hourly rainfall:",
    max_hourly,
    "mm"
)

print()
print(
    "ERA5 hourly rainfall:",
    "24.909 mm"
)

print("=" * 60)