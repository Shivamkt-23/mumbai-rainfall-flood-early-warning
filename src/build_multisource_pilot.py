from datetime import timezone

import numpy as np
import pandas as pd

from imerg_features import (
    login_earthdata,
    get_hourly_imerg_features,
)


# ============================================================
# FILES
# ============================================================

ERA5_FILE = (
    "data/processed/"
    "mumbai_era5_2010_2025_base.csv"
)

VALIDATION_FILE = (
    "data/processed/"
    "imerg_era5_all_recent_validation.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "mumbai_multisource_pilot_v1.csv"
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

era5 = era5.sort_values(
    "datetime"
).reset_index(
    drop=True
)


# ============================================================
# CHECK REQUIRED ERA5 COLUMNS
# ============================================================

required_era5_columns = [
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
    for col in required_era5_columns
    if col not in era5.columns
]

if missing_columns:

    raise RuntimeError(
        "ERA5 dataset is missing required columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# LOAD VALIDATED IMERG EVENT TIMES
# ============================================================

print(
    "Loading validated IMERG event times..."
)

events = pd.read_csv(
    VALIDATION_FILE,
    parse_dates=["datetime"]
)

events["datetime"] = pd.to_datetime(
    events["datetime"],
    utc=True
)

events = (
    events
    .sort_values("datetime")
    .drop_duplicates(
        subset=["datetime"]
    )
    .reset_index(
        drop=True
    )
)

print(
    "Events available:",
    len(events)
)


# ============================================================
# BUILD ERA5 LOOKUP
# ============================================================

era5_lookup = era5.set_index(
    "datetime"
)


# ============================================================
# HELPER: CONTIGUOUS RAINFALL ACCUMULATIONS
# ============================================================

def get_rainfall_accumulations(
    model_time
):
    """
    Calculate 3h, 6h and 24h rainfall accumulation
    ending at model_time.

    The function requires every hourly timestamp to be
    present. This prevents a large dataset gap from being
    accidentally treated as continuous rainfall history.
    """

    windows = {
        "rain_3h": 3,
        "rain_6h": 6,
        "rain_24h": 24,
    }

    results = {}

    for feature_name, hours in windows.items():

        start_time = (
            model_time
            - pd.Timedelta(
                hours=hours - 1
            )
        )

        expected_times = pd.date_range(
            start=start_time,
            end=model_time,
            freq="1h",
            tz="UTC"
        )

        available = era5_lookup.reindex(
            expected_times
        )

        rainfall = available[
            "rainfall_mm"
        ]

        # Require a complete hourly sequence.
        if len(rainfall) != hours:
            results[feature_name] = np.nan
            continue

        if rainfall.isna().any():
            results[feature_name] = np.nan
            continue

        results[feature_name] = float(
            rainfall.sum()
        )

    return results


# ============================================================
# EARTHDATA LOGIN
# ============================================================

print(
    "\nLogging into NASA Earthdata..."
)

login_earthdata()


# ============================================================
# BUILD MULTI-SOURCE DATASET
# ============================================================

rows = []

for event_number, (_, event) in enumerate(
    events.iterrows(),
    start=1
):

    model_time = pd.Timestamp(
        event["datetime"]
    )

    print()
    print(
        "=" * 70
    )

    print(
        f"Building row "
        f"{event_number}/{len(events)}"
    )

    print(
        "Model time:",
        model_time
    )

    # ========================================================
    # 1. ERA5 CURRENT ROW
    # ========================================================

    if model_time not in era5_lookup.index:

        print(
            "ERA5 current row missing. Skipping."
        )

        continue

    current = era5_lookup.loc[
        model_time
    ]

    # ========================================================
    # 2. NEXT-HOUR TARGET
    # ========================================================

    target_time = (
        model_time
        + pd.Timedelta(hours=1)
    )

    if target_time not in era5_lookup.index:

        print(
            "Next-hour target missing. Skipping."
        )

        continue

    next_hour_rain = float(
        era5_lookup.loc[
            target_time,
            "rainfall_mm"
        ]
    )

    # ========================================================
    # 3. ERA5 RAINFALL ACCUMULATIONS
    # ========================================================

    accumulation = (
        get_rainfall_accumulations(
            model_time
        )
    )

    print(
        "ERA5 3h rainfall:",
        accumulation["rain_3h"]
    )

    print(
        "ERA5 6h rainfall:",
        accumulation["rain_6h"]
    )

    print(
        "ERA5 24h rainfall:",
        accumulation["rain_24h"]
    )

    # ========================================================
    # 4. IMERG OBSERVATION
    #
    # Model time T uses the observed IMERG hour:
    #
    # T-1:00 → T
    # ========================================================

    imerg_start = (
        model_time
        - pd.Timedelta(hours=1)
    ).to_pydatetime()

    imerg_start = (
        imerg_start.astimezone(
            timezone.utc
        )
    )

    print(
        "IMERG:",
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
            "IMERG extraction failed:",
            e
        )

        continue

    # ========================================================
    # 5. BUILD FEATURE ROW
    # ========================================================

    feature_row = {

        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------

        "datetime":
            model_time,

        # ----------------------------------------------------
        # Spatial location
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # ERA5 / NWP variables
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Historical rainfall accumulation
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # IMERG rainfall
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # IMERG quality
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # IMERG uncertainty
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Prediction target
        # ----------------------------------------------------

        "next_hour_rain":
            next_hour_rain,
    }

    rows.append(
        feature_row
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

result = pd.DataFrame(
    rows
)


# ============================================================
# SORT
# ============================================================

result = (
    result
    .sort_values("datetime")
    .reset_index(drop=True)
)


# ============================================================
# VALIDATION
# ============================================================

print()
print(
    "=" * 75
)

print(
    "MULTI-SOURCE DATASET VALIDATION"
)

print(
    "=" * 75
)

print(
    "Rows:",
    len(result)
)

print(
    "Columns:",
    len(result.columns)
)


print(
    "\nColumn names:"
)

for column in result.columns:
    print(
        " -",
        column
    )


# ============================================================
# MISSING VALUE CHECK
# ============================================================

print(
    "\nMissing values:"
)

missing = (
    result
    .isna()
    .sum()
)

print(
    missing
)


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicate_count = (
    result["datetime"]
    .duplicated()
    .sum()
)

print(
    "\nDuplicate timestamps:",
    duplicate_count
)


# ============================================================
# BASIC RANGE CHECKS
# ============================================================

print(
    "\nBasic statistics:"
)

numeric_columns = [
    col
    for col in result.columns
    if col != "datetime"
]

print(
    result[
        numeric_columns
    ].describe()
    .transpose()
)


# ============================================================
# FIRST ROWS
# ============================================================

print(
    "\nFirst rows:"
)

print(
    result.head()
    .to_string(
        index=False
    )
)


# ============================================================
# ENSURE REQUIRED FEATURES EXIST
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
        "Missing required feature columns: "
        + ", ".join(missing_required)
    )


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print(
    "Saved:"
)

print(
    OUTPUT_FILE
)

print(
    "=" * 75
)