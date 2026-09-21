from datetime import timezone

import numpy as np
import pandas as pd

from imerg_hourly import (
    get_hourly_imerg_mumbai,
    login_earthdata
)


# ============================================================
# CONFIGURATION
# ============================================================

ERA5_FILE = (
    "data/processed/"
    "mumbai_era5_2010_2025_base.csv"
)

N_EVENTS = 10
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
# RESTRICT TO RECENT PERIOD
# ============================================================

recent = df[
    (df["datetime"] >= "2024-01-01")
    & (df["datetime"] < "2025-10-01")
    & (df["rainfall_mm"] >= MIN_RAIN_MM)
].copy()

print(
    "Recent rainy hours found:",
    len(recent)
)


# ============================================================
# SELECT WELL-SPACED EVENTS
# ============================================================

recent = recent.sort_values(
    "rainfall_mm",
    ascending=False
)

selected = []

for _, row in recent.iterrows():

    t = row["datetime"]

    too_close = any(
        abs(
            (t - old_t).total_seconds()
        ) < 6 * 3600
        for old_t in selected
    )

    if not too_close:
        selected.append(t)

    if len(selected) >= N_EVENTS:
        break


selected = sorted(selected)

print("\nSelected events:")

for t in selected:
    print(t)


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

for i, era5_time in enumerate(
    selected,
    start=1
):

    print()
    print("=" * 70)
    print(
        f"EVENT {i}/{len(selected)}"
    )
    print(
        "ERA5 timestamp:",
        era5_time
    )

    # ERA5 timestamp T represents
    # precipitation ending at T.
    # Therefore IMERG hour is T-1h to T.

    imerg_start = (
        era5_time
        - pd.Timedelta(hours=1)
    ).to_pydatetime()

    imerg_start = imerg_start.astimezone(
        timezone.utc
    )

    era5_row = df[
        df["datetime"] == era5_time
    ]

    if era5_row.empty:
        print("ERA5 record missing.")
        continue

    era5_rain = float(
        era5_row.iloc[0]["rainfall_mm"]
    )

    print(
        "Getting IMERG:",
        imerg_start
    )

    try:

        imerg_result = (
            get_hourly_imerg_mumbai(
                imerg_start
            )
        )

        imerg_rain = float(
            imerg_result[
                "hourly_rainfall_mm"
            ]
        )

        difference = (
            imerg_rain
            - era5_rain
        )

        results.append(
            {
                "era5_timestamp": era5_time,
                "imerg_start": imerg_start,
                "era5_rain_mm": era5_rain,
                "imerg_rain_mm": imerg_rain,
                "difference_mm": difference
            }
        )

        print(
            "ERA5 :",
            round(era5_rain, 3),
            "mm"
        )

        print(
            "IMERG:",
            round(imerg_rain, 3),
            "mm"
        )

        print(
            "Diff :",
            round(difference, 3),
            "mm"
        )

    except Exception as e:

        print(
            "ERROR:",
            e
        )


# ============================================================
# METRICS
# ============================================================

results_df = pd.DataFrame(
    results
)

if results_df.empty:
    raise RuntimeError(
        "No validation events completed."
    )

actual = results_df[
    "era5_rain_mm"
].to_numpy()

predicted = results_df[
    "imerg_rain_mm"
].to_numpy()

errors = (
    predicted - actual
)

mae = np.mean(
    np.abs(errors)
)

rmse = np.sqrt(
    np.mean(errors ** 2)
)

bias = np.mean(
    errors
)

correlation = (
    np.corrcoef(
        actual,
        predicted
    )[0, 1]
    if len(results_df) > 1
    else np.nan
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    "data/processed/"
    "imerg_era5_recent_validation.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 70)
print(
    "RECENT IMERG vs ERA5 VALIDATION"
)
print("=" * 70)

print(
    "Events completed:",
    len(results_df)
)

print(
    "MAE:",
    round(mae, 4),
    "mm"
)

print(
    "RMSE:",
    round(rmse, 4),
    "mm"
)

print(
    "Bias:",
    round(bias, 4),
    "mm"
)

print(
    "Correlation:",
    round(correlation, 4)
)

print()
print(
    "Saved:",
    output_file
)

print("=" * 70)