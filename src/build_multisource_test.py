from datetime import timezone

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

TEST_TIME = pd.Timestamp(
    "2024-09-25 18:00:00+00:00"
)


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
# GET ERA5 ROW AT MODEL TIME T
# ============================================================

era5_rows = df[
    df["datetime"] == TEST_TIME
]

if era5_rows.empty:
    raise RuntimeError(
        f"No ERA5 row found for {TEST_TIME}"
    )

era5_row = (
    era5_rows
    .iloc[0]
    .to_dict()
)


# ============================================================
# GET TARGET
#
# next_hour_rain is not stored in the base dataset,
# so calculate it from the next ERA5 hourly record.
# ============================================================

next_time = (
    TEST_TIME
    + pd.Timedelta(hours=1)
)

next_rows = df[
    df["datetime"] == next_time
]

if next_rows.empty:
    raise RuntimeError(
        f"No next-hour ERA5 row found for {next_time}"
    )

target_next_hour_rain = float(
    next_rows.iloc[0]["rainfall_mm"]
)


# ============================================================
# IMERG ALIGNMENT
#
# IMERG observation used at model time T represents:
#
# T-1 hour → T
#
# Therefore request IMERG beginning at T-1 hour.
# ============================================================

imerg_start = (
    TEST_TIME
    - pd.Timedelta(hours=1)
).to_pydatetime()

imerg_start = (
    imerg_start.astimezone(
        timezone.utc
    )
)


# ============================================================
# LOGIN
# ============================================================

print(
    "\nLogging into NASA Earthdata..."
)

login_earthdata()


# ============================================================
# EXTRACT IMERG FEATURES
# ============================================================

print(
    "\nExtracting IMERG features..."
)

imerg_features = (
    get_hourly_imerg_features(
        imerg_start
    )
)


# ============================================================
# BUILD MULTI-SOURCE ROW
# ============================================================

multisource = {

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    "datetime":
        TEST_TIME.isoformat(),

    # --------------------------------------------------------
    # ERA5
    # --------------------------------------------------------

    "era5_rainfall_mm":
        float(
            era5_row["rainfall_mm"]
        ),

    "temperature_c":
        float(
            era5_row["temperature_c"]
        ),

    "dewpoint_c":
        float(
            era5_row["dewpoint_c"]
        ),

    "pressure_hpa":
        float(
            era5_row["pressure_hpa"]
        ),

    "u10_ms":
        float(
            era5_row["u10_ms"]
        ),

    "v10_ms":
        float(
            era5_row["v10_ms"]
        ),

    # --------------------------------------------------------
    # IMERG rainfall
    # --------------------------------------------------------

    "imerg_center_hourly_mm":
        imerg_features[
            "imerg_center_hourly_mm"
        ],

    "imerg_mean_3x3_hourly_mm":
        imerg_features[
            "imerg_mean_3x3_hourly_mm"
        ],

    "imerg_max_3x3_hourly_mm":
        imerg_features[
            "imerg_max_3x3_hourly_mm"
        ],

    # --------------------------------------------------------
    # IMERG quality
    # --------------------------------------------------------

    "imerg_quality_center":
        imerg_features[
            "imerg_quality_center"
        ],

    "imerg_quality_mean_3x3":
        imerg_features[
            "imerg_quality_mean_3x3"
        ],

    "imerg_quality_min_3x3":
        imerg_features[
            "imerg_quality_min_3x3"
        ],

    # --------------------------------------------------------
    # IMERG uncertainty
    # --------------------------------------------------------

    "imerg_random_error_center":
        imerg_features[
            "imerg_random_error_center"
        ],

    "imerg_random_error_mean_3x3":
        imerg_features[
            "imerg_random_error_mean_3x3"
        ],

    "imerg_random_error_max_3x3":
        imerg_features[
            "imerg_random_error_max_3x3"
        ],

    # --------------------------------------------------------
    # Prediction target
    # --------------------------------------------------------

    "next_hour_rain":
        target_next_hour_rain,
}


# ============================================================
# DISPLAY
# ============================================================

result = pd.DataFrame(
    [multisource]
)

print()
print("=" * 75)
print("MULTI-SOURCE FEATURE ROW")
print("=" * 75)

print(
    result.to_string(
        index=False
    )
)

print("=" * 75)


# ============================================================
# SAVE TEST ROW
# ============================================================

output_file = (
    "data/processed/"
    "multisource_feature_test.csv"
)

result.to_csv(
    output_file,
    index=False
)

print()
print(
    "Saved:",
    output_file
)