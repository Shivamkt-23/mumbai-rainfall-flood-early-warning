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

# Use only a manageable number of rainy events for validation
N_EVENTS = 10

# Minimum ERA5 rainfall for an event
MIN_RAIN_MM = 5.0


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

# Keep only reasonably rainy hours
rainy = df[
    df["rainfall_mm"] >= MIN_RAIN_MM
].copy()

print(
    "Rainy ERA5 hours found:",
    len(rainy)
)


# ============================================================
# SELECT EVENTS
#
# Sort by rainfall and then enforce a minimum spacing so that
# we don't test ten adjacent hours from the same storm burst.
# ============================================================

rainy = rainy.sort_values(
    "rainfall_mm",
    ascending=False
)

selected = []

for _, row in rainy.iterrows():

    current_time = row["datetime"]

    too_close = False

    for selected_time in selected:

        difference_hours = abs(
            (
                current_time
                - selected_time
            ).total_seconds()
            / 3600
        )

        if difference_hours < 6:
            too_close = True
            break

    if not too_close:

        selected.append(
            current_time
        )

    if len(selected) >= N_EVENTS:
        break


selected = sorted(selected)

print(
    "\nSelected validation events:"
)

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
# VALIDATION LOOP
# ============================================================

results = []

for i, era5_time in enumerate(
    selected,
    start=1
):

    print()
    print(
        "=" * 70
    )

    print(
        f"Event {i}/{len(selected)}"
    )

    print(
        "ERA5 timestamp:",
        era5_time
    )

    # --------------------------------------------------------
    # Important temporal alignment:
    #
    # ERA5 rainfall at time T represents the previous hour.
    #
    # Therefore compare ERA5 T against IMERG:
    #
    # T-1:00 → T
    # --------------------------------------------------------

    imerg_start = (
        era5_time
        - pd.Timedelta(hours=1)
    ).to_pydatetime()

    imerg_start = imerg_start.astimezone(
        timezone.utc
    )

    try:

        # ----------------------------------------------------
        # ERA5 rainfall
        # ----------------------------------------------------

        era5_row = df[
            df["datetime"]
            == era5_time
        ]

        if era5_row.empty:

            print(
                "ERA5 record unavailable. Skipping."
            )

            continue

        era5_rain = float(
            era5_row.iloc[0][
                "rainfall_mm"
            ]
        )

        # ----------------------------------------------------
        # IMERG rainfall
        # ----------------------------------------------------

        print(
            "Getting IMERG:",
            imerg_start
        )

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

        # ----------------------------------------------------
        # Metrics for this event
        # ----------------------------------------------------

        difference = (
            imerg_rain
            - era5_rain
        )

        absolute_difference = abs(
            difference
        )

        print(
            "ERA5 :",
            round(
                era5_rain,
                3
            ),
            "mm"
        )

        print(
            "IMERG:",
            round(
                imerg_rain,
                3
            ),
            "mm"
        )

        print(
            "Diff :",
            round(
                difference,
                3
            ),
            "mm"
        )

        results.append(
            {
                "era5_timestamp": era5_time,
                "imerg_start": imerg_start,
                "era5_rain_mm": era5_rain,
                "imerg_rain_mm": imerg_rain,
                "difference_mm": difference,
                "absolute_difference_mm":
                    absolute_difference
            }
        )

    except Exception as e:

        print(
            "ERROR:",
            e
        )


# ============================================================
# CHECK RESULTS
# ============================================================

if not results:

    raise RuntimeError(
        "No validation events completed."
    )


results_df = pd.DataFrame(
    results
)


# ============================================================
# OVERALL METRICS
# ============================================================

actual = results_df[
    "era5_rain_mm"
].to_numpy()

predicted = results_df[
    "imerg_rain_mm"
].to_numpy()

errors = (
    predicted
    - actual
)

mae = np.mean(
    np.abs(errors)
)

rmse = np.sqrt(
    np.mean(
        errors ** 2
    )
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
# SAVE RESULTS
# ============================================================

output_file = (
    "data/processed/"
    "imerg_era5_validation.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("IMERG vs ERA5 MULTI-EVENT VALIDATION")
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
    "Bias (IMERG - ERA5):",
    round(bias, 4),
    "mm"
)

print(
    "Correlation:",
    round(correlation, 4)
)

print()
print(
    "Results saved to:"
)

print(output_file)

print("=" * 70)