from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

EVENT_FILE = (
    "data/processed/"
    "imerg_era5_all_recent_validation.csv"
)

ERA5_FILE = (
    "data/processed/"
    "mumbai_era5_2010_2025_base.csv"
)

OUTPUT_FILE = (
    "data/processed/"
    "mumbai_storm_windows_v1.csv"
)

# Hours before and after each detected event
WINDOW_BEFORE_HOURS = 24
WINDOW_AFTER_HOURS = 24

# Approximate IMERG file size from the files we've downloaded
AVERAGE_IMERG_FILE_MB = 7.8


# ============================================================
# LOAD EVENT TIMES
# ============================================================

print("Loading validation events...")

events = pd.read_csv(
    EVENT_FILE,
    parse_dates=["datetime"]
)

events["datetime"] = pd.to_datetime(
    events["datetime"],
    utc=True
)

event_times = sorted(
    events["datetime"]
    .drop_duplicates()
)

print(
    "Events:",
    len(event_times)
)


# ============================================================
# CREATE RAW WINDOWS
# ============================================================

raw_windows = []

for event_time in event_times:

    start = (
        event_time
        - pd.Timedelta(
            hours=WINDOW_BEFORE_HOURS
        )
    )

    end = (
        event_time
        + pd.Timedelta(
            hours=WINDOW_AFTER_HOURS
        )
    )

    raw_windows.append(
        (
            start,
            end
        )
    )


# ============================================================
# MERGE OVERLAPPING WINDOWS
# ============================================================

raw_windows.sort(
    key=lambda x: x[0]
)

merged_windows = []

for start, end in raw_windows:

    if not merged_windows:

        merged_windows.append(
            [start, end]
        )

        continue

    previous_start, previous_end = (
        merged_windows[-1]
    )

    # Merge overlapping OR touching windows
    if start <= previous_end + pd.Timedelta(
        hours=1
    ):

        merged_windows[-1][1] = max(
            previous_end,
            end
        )

    else:

        merged_windows.append(
            [start, end]
        )


# ============================================================
# LOAD ERA5
# ============================================================

print(
    "\nLoading ERA5 timestamps..."
)

era5 = pd.read_csv(
    ERA5_FILE,
    usecols=["datetime"],
    parse_dates=["datetime"]
)

era5["datetime"] = pd.to_datetime(
    era5["datetime"],
    utc=True
)

era5_times = set(
    era5["datetime"]
)


# ============================================================
# BUILD HOURLY WINDOWS
# ============================================================

rows = []

total_hours = 0

for window_id, (
    start,
    end
) in enumerate(
    merged_windows,
    start=1
):

    timestamps = pd.date_range(
        start=start,
        end=end,
        freq="1h",
        tz="UTC"
    )

    print()
    print("=" * 70)
    print(
        f"WINDOW {window_id}"
    )
    print(
        "Start:",
        start
    )
    print(
        "End:",
        end
    )
    print(
        "Candidate hours:",
        len(timestamps)
    )

    available = [
        t
        for t in timestamps
        if t in era5_times
    ]

    missing = (
        len(timestamps)
        - len(available)
    )

    print(
        "ERA5 hours available:",
        len(available)
    )

    print(
        "ERA5 hours missing:",
        missing
    )

    if available:

        actual_start = min(
            available
        )

        actual_end = max(
            available
        )

    else:

        actual_start = None
        actual_end = None

    rows.append(
        {
            "window_id": window_id,
            "requested_start": start,
            "requested_end": end,
            "candidate_hours": len(timestamps),
            "era5_available_hours": len(
                available
            ),
            "era5_missing_hours": missing,
            "actual_start": actual_start,
            "actual_end": actual_end,
        }
    )

    total_hours += len(
        available
    )


# ============================================================
# SAVE WINDOW SUMMARY
# ============================================================

windows_df = pd.DataFrame(
    rows
)

windows_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# ESTIMATE IMERG REQUIREMENT
# ============================================================

# Every hourly observation requires:
# 2 half-hour IMERG granules

total_half_hour_granules = (
    total_hours * 2
)

estimated_storage_mb = (
    total_half_hour_granules
    * AVERAGE_IMERG_FILE_MB
)

estimated_storage_gb = (
    estimated_storage_mb
    / 1024
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 75)
print("STORM WINDOW SUMMARY")
print("=" * 75)

print(
    "Original events:",
    len(event_times)
)

print(
    "Merged storm windows:",
    len(merged_windows)
)

print(
    "Total continuous ERA5 hours:",
    total_hours
)

print(
    "Estimated IMERG half-hour files:",
    total_half_hour_granules
)

print(
    "Estimated IMERG storage:",
    round(
        estimated_storage_gb,
        2
    ),
    "GB"
)

print()
print(
    "Window summary saved to:"
)

print(
    OUTPUT_FILE
)

print("=" * 75)