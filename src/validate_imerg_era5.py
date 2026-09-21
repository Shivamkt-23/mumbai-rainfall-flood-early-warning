from datetime import datetime, timezone

import pandas as pd

from imerg_hourly import get_hourly_imerg_mumbai, login_earthdata


# ============================================================
# CONFIGURATION
# ============================================================

ERA5_FILE = (
    "data/processed/"
    "mumbai_era5_2010_2025_base.csv"
)

TEST_TIME = datetime(
    2024,
    9,
    25,
    17,
    0,
    tzinfo=timezone.utc
)


# ============================================================
# LOAD ERA5
# ============================================================

print("Loading ERA5 data...")

df = pd.read_csv(
    ERA5_FILE,
    parse_dates=["datetime"]
)

df["datetime"] = pd.to_datetime(
    df["datetime"],
    utc=True
)


# ============================================================
# FIND ERA5 RECORD
# ============================================================

era5_time = (
    TEST_TIME
    + pd.Timedelta(hours=1)
)

era5_row = df[
    df["datetime"] == era5_time
]

if era5_row.empty:
    raise RuntimeError(
        f"ERA5 record not found for {era5_time}"
    )

era5_rain = float(
    era5_row.iloc[0]["rainfall_mm"]
)


# ============================================================
# GET IMERG HOURLY RAINFALL
# ============================================================

print("Logging into NASA Earthdata...")

login_earthdata()

print(
    "\nGetting IMERG rainfall..."
)

imerg = get_hourly_imerg_mumbai(
    TEST_TIME
)

imerg_rain = float(
    imerg["hourly_rainfall_mm"]
)


# ============================================================
# COMPARISON
# ============================================================

difference = (
    imerg_rain
    - era5_rain
)

absolute_difference = abs(
    difference
)

percentage_difference = (
    absolute_difference
    / max(era5_rain, 1e-6)
) * 100


# ============================================================
# PRINT RESULT
# ============================================================

print()
print("=" * 65)
print("IMERG vs ERA5 VALIDATION")
print("=" * 65)

print(
    "IMERG interval:",
    TEST_TIME.isoformat(),
    "to",
    (
        TEST_TIME
        + pd.Timedelta(hours=1)
    ).isoformat()
)

print()

print(
    "IMERG rainfall:",
    round(imerg_rain, 4),
    "mm"
)

print(
    "ERA5 rainfall:",
    round(era5_rain, 4),
    "mm"
)

print(
    "Difference:",
    round(difference, 4),
    "mm"
)

print(
    "Absolute difference:",
    round(
        absolute_difference,
        4
    ),
    "mm"
)

print(
    "Percentage difference:",
    round(
        percentage_difference,
        2
    ),
    "%"
)

print("=" * 65)