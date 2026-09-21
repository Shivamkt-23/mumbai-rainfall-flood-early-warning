# ============================================================
# IMERG 2010 REMOTE SUBSET TEST
#
# Source:
#   NASA GPM IMERG Final V07
#
# Access:
#   ERDDAP remote mirror
#
# Purpose:
#   Test whether we can remotely retrieve only the Mumbai
#   rainfall grid point for January 2010 without downloading
#   the full global IMERG files.
#
# If this succeeds, we will extend the same method to
# 2010-2025 and build the final ERA5 + IMERG dataset.
# ============================================================

from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

ERDDAP_BASE = (
    "https://erddap.aoml.noaa.gov/hdb/erddap/griddap"
)

DATASET = "precipitation_2010"


# Mumbai nearest IMERG grid point
MUMBAI_LAT = 19.05
MUMBAI_LON = 72.95


# Test period:
# One day only first.
START = datetime(
    2010,
    1,
    1,
    0,
    0,
    tzinfo=timezone.utc,
)

END = datetime(
    2010,
    1,
    2,
    0,
    0,
    tzinfo=timezone.utc,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "imerg_subsets"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_FILE = (
    OUTPUT_DIR
    / "imerg_2010_january_test.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("NASA GPM IMERG REMOTE SUBSET TEST")
print("=" * 70)

print()
print("Dataset:")
print(DATASET)

print()
print("Period:")
print(
    START.isoformat(),
    "→",
    END.isoformat()
)

print()
print("Mumbai:")
print(
    f"Latitude : {MUMBAI_LAT}"
)

print(
    f"Longitude: {MUMBAI_LON}"
)


# ============================================================
# STEP 1 — DATASET INFO
# ============================================================

print()
print("=" * 70)
print("STEP 1 — CHECKING REMOTE DATASET")
print("=" * 70)


info_url = (
    f"https://erddap.aoml.noaa.gov/"
    f"hdb/erddap/info/{DATASET}/index.json"
)


print()
print("Requesting dataset metadata...")

info_response = requests.get(
    info_url,
    timeout=60,
)

print(
    "HTTP status:",
    info_response.status_code
)


info_response.raise_for_status()

info_json = info_response.json()


print(
    "Dataset metadata received."
)


# ============================================================
# STEP 2 — EXTRACT DIMENSION INFORMATION
# ============================================================

print()
print("=" * 70)
print("STEP 2 — READING DIMENSIONS")
print("=" * 70)


table = pd.DataFrame(
    info_json.get(
        "table",
        {}
    ).get(
        "rows",
        []
    ),
    columns=(
        info_json.get(
            "table",
            {}
        ).get(
            "columnNames",
            []
        )
    ),
)


print()
print(
    "Metadata rows:",
    len(table)
)


if not table.empty:

    print()
    print(
        table.to_string(
            index=False
        )
    )


# ============================================================
# STEP 3 — ERDDAP QUERY
# ============================================================

print()
print("=" * 70)
print("STEP 3 — BUILDING REMOTE QUERY")
print("=" * 70)


start_text = START.strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)

end_text = END.strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)


# ERDDAP grid query:
#
# time
# latitude
# longitude
#
# We request only one spatial point.
#
# The exact dimension order is validated against the returned
# CSV rather than downloading the global product.

query = (
    f"precipitation"
    f"[(%s):1:(%s)]"
    f"[(%s):1:(%s)]"
    f"[(%s):1:(%s)]"
    % (
        start_text,
        end_text,
        MUMBAI_LAT,
        MUMBAI_LAT,
        MUMBAI_LON,
        MUMBAI_LON,
    )
)


query_url = (
    f"{ERDDAP_BASE}/"
    f"{DATASET}.csv?"
    f"{query}"
)


print()
print("Remote query prepared.")

print()
print(
    "Querying only the Mumbai point."
)


# ============================================================
# STEP 4 — DOWNLOAD REMOTE SUBSET
# ============================================================

print()
print("=" * 70)
print("STEP 4 — DOWNLOADING REMOTE SUBSET")
print("=" * 70)


response = requests.get(
    query_url,
    timeout=180,
)


print()
print(
    "HTTP status:",
    response.status_code
)


if not response.ok:

    print()
    print(
        "Remote server returned an error:"
    )

    print(
        response.text[:3000]
    )

    response.raise_for_status()


print(
    "Remote subset downloaded."
)


# ============================================================
# STEP 5 — READ CSV
# ============================================================

print()
print("=" * 70)
print("STEP 5 — READING SUBSET")
print("=" * 70)


from io import StringIO


subset = pd.read_csv(
    StringIO(
        response.text
    )
)


print()
print(
    "Rows:",
    len(subset)
)

print(
    "Columns:",
    len(subset.columns)
)


print()
print(
    "Columns:"
)

for column in subset.columns:

    print(
        f" - {column}"
    )


print()
print(
    "First rows:"
)

print(
    subset.head(
        10
    ).to_string(
        index=False
    )
)


# ============================================================
# STEP 6 — BASIC VALIDATION
# ============================================================

print()
print("=" * 70)
print("STEP 6 — VALIDATING RESULT")
print("=" * 70)


if subset.empty:

    raise RuntimeError(
        "The remote IMERG subset returned zero rows."
    )


# Look for rainfall variable
rain_candidates = [
    column
    for column in subset.columns
    if "precip" in column.lower()
]


if not rain_candidates:

    raise RuntimeError(
        "No precipitation column was found "
        "in the returned subset."
    )


rain_column = (
    rain_candidates[0]
)


print()
print(
    "Rainfall column:",
    rain_column
)


subset[rain_column] = pd.to_numeric(
    subset[rain_column],
    errors="coerce",
)


print()
print(
    "Missing rainfall values:",
    int(
        subset[rain_column]
        .isna()
        .sum()
    )
)


print()
print(
    "Rainfall statistics:"
)

print(
    subset[rain_column]
    .describe()
)


# ============================================================
# STEP 7 — SAVE
# ============================================================

print()
print("=" * 70)
print("STEP 7 — SAVING TEST FILE")
print("=" * 70)


subset.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print(
    "Saved:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)

print()
print(
    "The January 2010 remote IMERG extraction succeeded."
)

print()
print(
    "Next step:"
)

print(
    "Scale this same remote-query approach to "
    "2010-2025 and merge it with the existing "
    "ERA5 2010-2025 dataset."
)