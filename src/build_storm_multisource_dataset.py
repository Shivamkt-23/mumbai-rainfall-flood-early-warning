from datetime import timezone
from pathlib import Path
import os

import numpy as np
import pandas as pd

from imerg_features import (
    login_earthdata,
    get_hourly_imerg_features,
)


# ============================================================
# CONFIGURATION
# ============================================================

ERA5_FILE = (
    "data/processed/"
    "mumbai_era5_2010_2025_base.csv"
)

WINDOW_FILE = (
    "data/processed/"
    "mumbai_storm_windows_v1.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "mumbai_storm_multisource_v1.csv"
)

CHECKPOINT_FILE = (
    "data/processed/"
    "mumbai_storm_multisource_v1_checkpoint.csv"
)


# ============================================================
# LOAD ERA5
# ============================================================

print("Loading ERA5...")

era5 = pd.read_csv(
    ERA5_FILE,
    parse_dates=["datetime"]
)

era5["datetime"] = pd.to_datetime(
    era5["datetime"],
    utc=True
)

era5 = (
    era5
    .sort_values("datetime")
    .reset_index(drop=True)
)


# ============================================================
# REQUIRED ERA5 COLUMNS
# ============================================================

required_columns = [
    "datetime",
    "latitude_center",
    "longitude_center",
    "rainfall_mm",
    "temperature_c",
    "dewpoint_c",
    "pressure_hpa",
    "u10_ms",
    "v10_ms",
]

missing_columns = [
    col
    for col in required_columns
    if col not in era5.columns
]

if missing_columns:
    raise RuntimeError(
        "Missing ERA5 columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# ERA5 LOOKUP
# ============================================================

era5_lookup = (
    era5
    .set_index("datetime")
)


# ============================================================
# LOAD STORM WINDOWS
# ============================================================

print(
    "\nLoading storm windows..."
)

windows = pd.read_csv(
    WINDOW_FILE,
    parse_dates=[
        "requested_start",
        "requested_end",
        "actual_start",
        "actual_end",
    ]
)

for col in [
    "requested_start",
    "requested_end",
    "actual_start",
    "actual_end",
]:

    windows[col] = pd.to_datetime(
        windows[col],
        utc=True
    )


# ============================================================
# GENERATE ALL HOURLY TIMESTAMPS
# ============================================================

all_times = []

for _, window in windows.iterrows():

    start = window["actual_start"]
    end = window["actual_end"]

    timestamps = pd.date_range(
        start=start,
        end=end,
        freq="1h",
        tz="UTC"
    )

    for timestamp in timestamps:

        all_times.append(
            (
                int(window["window_id"]),
                timestamp
            )
        )


# Remove duplicates while preserving order
seen = set()

unique_times = []

for window_id, timestamp in all_times:

    key = timestamp

    if key not in seen:

        seen.add(key)

        unique_times.append(
            (
                window_id,
                timestamp
            )
        )


print(
    "Total continuous hours:",
    len(unique_times)
)


# ============================================================
# CHECKPOINT / RESUME
# ============================================================

completed = {}

if os.path.exists(
    CHECKPOINT_FILE
):

    print(
        "\nCheckpoint found:"
    )

    print(
        CHECKPOINT_FILE
    )

    checkpoint = pd.read_csv(
        CHECKPOINT_FILE,
        parse_dates=["datetime"]
    )

    checkpoint["datetime"] = pd.to_datetime(
        checkpoint["datetime"],
        utc=True
    )

    for _, row in checkpoint.iterrows():

        completed[
            row["datetime"]
        ] = row.to_dict()

    print(
        "Checkpoint rows:",
        len(completed)
    )

else:

    print(
        "\nNo checkpoint found."
    )


# ============================================================
# LOGIN
# ============================================================

print(
    "\nLogging into NASA Earthdata..."
)

login_earthdata()


# ============================================================
# HELPER: RAINFALL ACCUMULATIONS
# ============================================================

def rainfall_accumulations(
    model_time
):

    results = {}

    for name, hours in [
        ("rain_3h", 3),
        ("rain_6h", 6),
        ("rain_24h", 24),
    ]:

        start = (
            model_time
            - pd.Timedelta(
                hours=hours - 1
            )
        )

        expected = pd.date_range(
            start=start,
            end=model_time,
            freq="1h",
            tz="UTC"
        )

        history = era5_lookup.reindex(
            expected
        )

        rainfall = history[
            "rainfall_mm"
        ]

        if len(rainfall) != hours:
            results[name] = np.nan
            continue

        if rainfall.isna().any():
            results[name] = np.nan
            continue

        results[name] = float(
            rainfall.sum()
        )

    return results


# ============================================================
# CHECKPOINT SAVE FUNCTION
# ============================================================

def save_checkpoint(
    rows
):

    if not rows:
        return

    checkpoint_df = pd.DataFrame(
        rows
    )

    checkpoint_df = (
        checkpoint_df
        .sort_values("datetime")
        .drop_duplicates(
            subset=["datetime"],
            keep="last"
        )
    )

    checkpoint_df.to_csv(
        CHECKPOINT_FILE,
        index=False
    )


# ============================================================
# START WITH EXISTING CHECKPOINT ROWS
# ============================================================

rows = list(
    completed.values()
)


# ============================================================
# MAIN BUILD LOOP
# ============================================================

for index, (
    window_id,
    model_time
) in enumerate(
    unique_times,
    start=1
):

    # --------------------------------------------------------
    # Skip completed row
    # --------------------------------------------------------

    if model_time in completed:

        print(
            f"[{index}/{len(unique_times)}] "
            f"Already completed:",
            model_time
        )

        continue

    print()
    print("=" * 75)

    print(
        f"PROCESSING "
        f"{index}/{len(unique_times)}"
    )

    print(
        "Window ID:",
        window_id
    )

    print(
        "Model time:",
        model_time
    )

    # --------------------------------------------------------
    # ERA5 current row
    # --------------------------------------------------------

    if model_time not in era5_lookup.index:

        print(
            "ERA5 current row missing. Skipping."
        )

        continue

    current = era5_lookup.loc[
        model_time
    ]

    # --------------------------------------------------------
    # Next-hour target
    # --------------------------------------------------------

    target_time = (
        model_time
        + pd.Timedelta(hours=1)
    )

    if target_time not in era5_lookup.index:

        print(
            "Next-hour target unavailable. Skipping."
        )

        continue

    next_hour_rain = float(
        era5_lookup.loc[
            target_time,
            "rainfall_mm"
        ]
    )

    # --------------------------------------------------------
    # Historical rainfall accumulations
    # --------------------------------------------------------

    accumulation = (
        rainfall_accumulations(
            model_time
        )
    )

    if any(
        pd.isna(value)
        for value in accumulation.values()
    ):

        print(
            "Incomplete ERA5 history. Skipping."
        )

        continue

    # --------------------------------------------------------
    # IMERG observed hour
    #
    # T-1h → T
    # --------------------------------------------------------

    imerg_start = (
        model_time
        - pd.Timedelta(hours=1)
    ).to_pydatetime()

    imerg_start = (
        imerg_start
        .astimezone(timezone.utc)
    )

    print(
        "IMERG start:",
        imerg_start
    )

    try:

        imerg = (
            get_hourly_imerg_features(
                imerg_start
            )
        )

    except Exception as e:

        print(
            "IMERG extraction failed:"
        )

        print(
            repr(e)
        )

        print(
            "Row will be retried next run."
        )

        continue

    # --------------------------------------------------------
    # Build row
    # --------------------------------------------------------

    row = {

        "window_id":
            window_id,

        "datetime":
            model_time,

        # -------------------------------
        # Location
        # -------------------------------

        "latitude":
            float(
                current[
                    "latitude_center"
                ]
            ),

        "longitude":
            float(
                current[
                    "longitude_center"
                ]
            ),

        # -------------------------------
        # ERA5 current weather
        # -------------------------------

        "era5_rainfall_mm":
            float(
                current[
                    "rainfall_mm"
                ]
            ),

        "temperature_c":
            float(
                current[
                    "temperature_c"
                ]
            ),

        "dewpoint_c":
            float(
                current[
                    "dewpoint_c"
                ]
            ),

        "pressure_hpa":
            float(
                current[
                    "pressure_hpa"
                ]
            ),

        "u10_ms":
            float(
                current[
                    "u10_ms"
                ]
            ),

        "v10_ms":
            float(
                current[
                    "v10_ms"
                ]
            ),

        # -------------------------------
        # ERA5 accumulation
        # -------------------------------

        "rain_3h":
            accumulation[
                "rain_3h"
            ],

        "rain_6h":
            accumulation[
                "rain_6h"
            ],

        "rain_24h":
            accumulation[
                "rain_24h"
            ],

        # -------------------------------
        # IMERG rainfall
        # -------------------------------

        "imerg_center_hourly_mm":
            float(
                imerg[
                    "imerg_center_hourly_mm"
                ]
            ),

        "imerg_mean_3x3_hourly_mm":
            float(
                imerg[
                    "imerg_mean_3x3_hourly_mm"
                ]
            ),

        "imerg_max_3x3_hourly_mm":
            float(
                imerg[
                    "imerg_max_3x3_hourly_mm"
                ]
            ),

        # -------------------------------
        # IMERG quality
        # -------------------------------

        "imerg_quality_center":
            float(
                imerg[
                    "imerg_quality_center"
                ]
            ),

        "imerg_quality_mean_3x3":
            float(
                imerg[
                    "imerg_quality_mean_3x3"
                ]
            ),

        "imerg_quality_min_3x3":
            float(
                imerg[
                    "imerg_quality_min_3x3"
                ]
            ),

        # -------------------------------
        # IMERG uncertainty
        # -------------------------------

        "imerg_random_error_center":
            float(
                imerg[
                    "imerg_random_error_center"
                ]
            ),

        "imerg_random_error_mean_3x3":
            float(
                imerg[
                    "imerg_random_error_mean_3x3"
                ]
            ),

        "imerg_random_error_max_3x3":
            float(
                imerg[
                    "imerg_random_error_max_3x3"
                ]
            ),

        # -------------------------------
        # Target
        # -------------------------------

        "next_hour_rain":
            next_hour_rain,
    }

    rows.append(
        row
    )

    completed[
        model_time
    ] = row

    # --------------------------------------------------------
    # SAVE CHECKPOINT AFTER EVERY ROW
    # --------------------------------------------------------

    save_checkpoint(
        rows
    )

    print(
        "Row saved."
    )

    print(
        "Current completed rows:",
        len(rows)
    )


# ============================================================
# FINAL DATASET
# ============================================================

result = pd.DataFrame(
    rows
)


result["datetime"] = pd.to_datetime(
    result["datetime"],
    utc=True
)


result = (
    result
    .sort_values("datetime")
    .drop_duplicates(
        subset=["datetime"],
        keep="last"
    )
    .reset_index(drop=True)
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 75)
print("FINAL STORM MULTI-SOURCE DATASET")
print("=" * 75)

print(
    "Rows:",
    len(result)
)

print(
    "Columns:",
    len(result.columns)
)

print(
    "Duplicate timestamps:",
    result["datetime"].duplicated().sum()
)

print(
    "\nMissing values:"
)

print(
    result.isna().sum()
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_features = [
    "era5_rainfall_mm",
    "temperature_c",
    "dewpoint_c",
    "pressure_hpa",
    "u10_ms",
    "v10_ms",
    "rain_3h",
    "rain_6h",
    "rain_24h",
    "imerg_center_hourly_mm",
    "imerg_mean_3x3_hourly_mm",
    "imerg_max_3x3_hourly_mm",
    "imerg_quality_center",
    "imerg_quality_mean_3x3",
    "imerg_quality_min_3x3",
    "imerg_random_error_center",
    "imerg_random_error_mean_3x3",
    "imerg_random_error_max_3x3",
    "next_hour_rain",
]


missing_required = [
    col
    for col in required_features
    if col not in result.columns
]

if missing_required:

    raise RuntimeError(
        "Missing required columns: "
        + ", ".join(
            missing_required
        )
    )


# ============================================================
# SAVE FINAL DATASET
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print(
    "Saved final dataset:"
)

print(
    OUTPUT_FILE
)

print(
    "=" * 75
)