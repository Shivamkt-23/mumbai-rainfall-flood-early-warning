from datetime import datetime, timezone, timedelta

import h5py
import pandas as pd

from imerg_mumbai import (
    login_earthdata,
    find_granule,
    get_download_url,
    download_granule,
)


# ============================================================
# CONFIG
# ============================================================

ERA5_FILE = (
    "data/processed/"
    "mumbai_era5_2010_2025_base.csv"
)

IMERG_TIME = datetime(
    2011,
    6,
    9,
    23,
    0,
    tzinfo=timezone.utc
)


# ============================================================
# ERA5
# ============================================================

print("=" * 70)
print("ERA5 TIME CHECK")
print("=" * 70)

df = pd.read_csv(
    ERA5_FILE,
    parse_dates=["datetime"]
)

df["datetime"] = pd.to_datetime(
    df["datetime"],
    utc=True
)

start = pd.Timestamp(
    "2011-06-09 21:00:00",
    tz="UTC"
)

end = pd.Timestamp(
    "2011-06-10 02:00:00",
    tz="UTC"
)

window = df[
    (df["datetime"] >= start)
    & (df["datetime"] <= end)
][
    [
        "datetime",
        "rainfall_mm"
    ]
]

print(window.to_string(index=False))


# ============================================================
# IMERG
# ============================================================

print()
print("=" * 70)
print("IMERG TIME METADATA")
print("=" * 70)

login_earthdata()

granule = find_granule(
    IMERG_TIME
)

url = get_download_url(
    granule
)

file_path = download_granule(
    url
)

print("File:")
print(file_path)

with h5py.File(
    file_path,
    "r"
) as h5:

    time_ds = h5[
        "Grid/time"
    ]

    print()
    print("Raw Grid/time value:")
    print(time_ds[:])

    print()
    print("Grid/time attributes:")

    for key, value in time_ds.attrs.items():
        print(
            repr(key),
            ":",
            repr(value)
        )


print()
print("=" * 70)
print("DONE")
print("=" * 70)