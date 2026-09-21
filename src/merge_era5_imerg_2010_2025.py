#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
ERA5_PATH = BASE_DIR / "data" / "processed" / "mumbai_era5_2010_2025_ml_ready.csv"
IMERG_PATH = BASE_DIR / "data" / "processed" / "imerg_2010_2025_daily.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "mumbai_era5_imerg_multisource_2010_2025.csv"

def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing columns: {missing}")

print("=" * 78)
print("ERA5 + NASA IMERG 2010-2025 MULTISOURCE MERGE")
print("=" * 78)
print("ERA5 :", ERA5_PATH)
print("IMERG:", IMERG_PATH)
print("OUT  :", OUTPUT_PATH)

era5 = pd.read_csv(ERA5_PATH, parse_dates=["datetime"])
imerg = pd.read_csv(IMERG_PATH, parse_dates=["date"])

require_columns(era5, ["datetime", "next_hour_rain", "rainfall_mm"], "ERA5")
require_columns(
    imerg,
    ["date", "precip_mm_day", "randomError", "precipitation_cnt",
     "precipitation_cnt_cond", "probabilityLiquidPrecipitation", "lat", "lon"],
    "IMERG",
)

era5 = era5.sort_values("datetime").reset_index(drop=True)
imerg = imerg.sort_values("date").reset_index(drop=True)

if era5["datetime"].duplicated().any():
    raise RuntimeError("ERA5 contains duplicate datetimes.")
if imerg["date"].duplicated().any():
    raise RuntimeError("IMERG contains duplicate dates.")

era5["date"] = era5["datetime"].dt.normalize()

# Quality ratio for each completed IMERG day.
cnt = imerg["precipitation_cnt"].replace(0, np.nan)
imerg["quality_ratio"] = imerg["precipitation_cnt_cond"] / cnt

# IMPORTANT: shift by one completed day before using satellite data
# with an hourly prediction at date D. This prevents future leakage.
f = pd.DataFrame({"date": imerg["date"]})
rain = imerg["precip_mm_day"]

f["imerg_prev_day_mm"] = rain.shift(1)
f["imerg_3day_sum_mm"] = rain.shift(1).rolling(3, min_periods=1).sum()
f["imerg_7day_sum_mm"] = rain.shift(1).rolling(7, min_periods=1).sum()
f["imerg_3day_max_mm"] = rain.shift(1).rolling(3, min_periods=1).max()
f["imerg_7day_max_mm"] = rain.shift(1).rolling(7, min_periods=1).max()
f["imerg_prev_day_random_error"] = imerg["randomError"].shift(1)
f["imerg_prev_day_quality_ratio"] = imerg["quality_ratio"].shift(1)
f["imerg_prev_day_liquid_probability"] = imerg["probabilityLiquidPrecipitation"].shift(1)
f["imerg_prev_day_count"] = imerg["precipitation_cnt"].shift(1)
f["imerg_prev_day_cond_count"] = imerg["precipitation_cnt_cond"].shift(1)

# Provenance: fixed IMERG grid point.
f["imerg_lat"] = imerg["lat"]
f["imerg_lon"] = imerg["lon"]

merged = era5.merge(f, on="date", how="left", validate="many_to_one", sort=False)
merged = merged.sort_values("datetime").reset_index(drop=True)

if len(merged) != len(era5):
    raise RuntimeError(f"Merge changed ERA5 row count: {len(era5)} -> {len(merged)}")

sat_cols = [
    "imerg_prev_day_mm", "imerg_3day_sum_mm", "imerg_7day_sum_mm",
    "imerg_3day_max_mm", "imerg_7day_max_mm",
    "imerg_prev_day_random_error", "imerg_prev_day_quality_ratio",
    "imerg_prev_day_liquid_probability", "imerg_prev_day_count",
    "imerg_prev_day_cond_count",
]

print()
print("ERA5 rows :", len(era5))
print("IMERG rows:", len(imerg))
print("Merged    :", len(merged))
print("Datetime  :", merged["datetime"].min(), "->", merged["datetime"].max())
print()

for c in sat_cols:
    print(f"{c:40s} missing={merged[c].isna().sum():,}")

# The first ERA5 row is 2010-01-02, so previous-day IMERG should exist.
# Since IMERG starts on 2010-01-01, this first row should be populated.
first_date = merged["date"].min()
first = merged.loc[merged["date"] == first_date, "imerg_prev_day_mm"]
if first.isna().all():
    raise RuntimeError(
        "Unexpected missing IMERG previous-day feature on the first ERA5 date."
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUTPUT_PATH, index=False)

print()
print("Saved:", OUTPUT_PATH)
print("Shape:", merged.shape)
print()
print("First merged row:")
print(
    merged.loc[
        merged["datetime"].idxmin(),
        ["datetime", "rainfall_mm", "next_hour_rain",
         "imerg_prev_day_mm", "imerg_3day_sum_mm", "imerg_7day_sum_mm"]
    ].to_string()
)

valid = merged[merged["imerg_prev_day_mm"].notna()]
if not valid.empty:
    print()
    print("First row with valid IMERG history:")
    print(
        valid.iloc[0][
            ["datetime", "rainfall_mm", "next_hour_rain",
             "imerg_prev_day_mm", "imerg_3day_sum_mm",
             "imerg_7day_sum_mm", "imerg_prev_day_random_error",
             "imerg_prev_day_quality_ratio"]
        ].to_string()
    )

print()
print("=" * 78)
print("MERGE COMPLETE")
print("=" * 78)
