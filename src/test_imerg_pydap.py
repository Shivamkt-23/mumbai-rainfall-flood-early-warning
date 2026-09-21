import os
from pathlib import Path

import h5py
import numpy as np
import earthaccess
import requests


# ============================================================
# CONFIGURATION
# ============================================================

CONCEPT_ID = "C2723754847-GES_DISC"

# One IMERG half-hour period for testing
START_TIME = "2024-09-25T17:00:00Z"
END_TIME = "2024-09-25T18:00:00Z"

# Mumbai target
TARGET_LAT = 19.0
TARGET_LON = 73.0

# Local storage
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOWNLOAD_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "GPM_IMERG"
)

DOWNLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. LOGIN
# ============================================================

print("Logging into NASA Earthdata...")

earthaccess.login(
    strategy="environment"
)

print("Earthdata login successful.")


# ============================================================
# 2. SEARCH IMERG GRANULE
# ============================================================

print("\nSearching for IMERG granule...")

granules = earthaccess.search_data(
    concept_id=CONCEPT_ID,
    temporal=(
        START_TIME,
        END_TIME
    ),
    count=1
)

print(
    "Granules found:",
    len(granules)
)

if not granules:
    raise RuntimeError(
        "No IMERG granule found."
    )

granule = granules[0]

print("\nGranule:")
print(granule)


# ============================================================
# 3. GET DIRECT HTTPS DOWNLOAD URL
# ============================================================

download_url = None

for link in granule.get(
    "umm",
    {}
).get(
    "RelatedUrls",
    []
):

    if link.get(
        "Type"
    ) == "GET DATA":

        download_url = link.get(
            "URL"
        )

        break


if download_url is None:

    # Fallback to granule.data
    data_urls = getattr(
        granule,
        "data",
        None
    )

    if data_urls:
        download_url = data_urls[0]


if download_url is None:

    raise RuntimeError(
        "Could not find IMERG download URL."
    )


print("\nDownload URL:")
print(download_url)


# ============================================================
# 4. LOCAL FILE NAME
# ============================================================

filename = download_url.split("/")[-1]

local_file = (
    DOWNLOAD_DIR
    / filename
)

print("\nLocal file:")
print(local_file)


# ============================================================
# 5. DOWNLOAD ONLY IF NOT ALREADY PRESENT
# ============================================================

if local_file.exists():

    print(
        "\nFile already exists."
    )

else:

    print(
        "\nDownloading IMERG granule..."
    )

    session = (
        earthaccess
        .get_requests_https_session()
    )

    response = session.get(
        download_url,
        stream=True,
        timeout=120
    )

    response.raise_for_status()

    total_bytes = 0

    with open(
        local_file,
        "wb"
    ) as f:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if chunk:

                f.write(chunk)

                total_bytes += len(
                    chunk
                )

    print(
        "Downloaded:",
        round(
            total_bytes / (1024 * 1024),
            2
        ),
        "MB"
    )


# ============================================================
# 6. OPEN HDF5 FILE
# ============================================================

print(
    "\nOpening IMERG HDF5 file..."
)

with h5py.File(
    local_file,
    "r"
) as h5:

    # --------------------------------------------------------
    # Read coordinates
    # --------------------------------------------------------

    lat_values = h5[
        "Grid/lat"
    ][...]

    lon_values = h5[
        "Grid/lon"
    ][...]

    # --------------------------------------------------------
    # Read precipitation
    # --------------------------------------------------------

    precipitation = h5[
        "Grid/precipitation"
    ]


    print(
        "\nDataset information:"
    )

    print(
        "Latitude shape:",
        lat_values.shape
    )

    print(
        "Longitude shape:",
        lon_values.shape
    )

    print(
        "Precipitation shape:",
        precipitation.shape
    )


    # --------------------------------------------------------
    # Find nearest Mumbai pixel
    # --------------------------------------------------------

    lat_idx = int(
        np.abs(
            lat_values
            - TARGET_LAT
        ).argmin()
    )

    lon_idx = int(
        np.abs(
            lon_values
            - TARGET_LON
        ).argmin()
    )


    nearest_lat = float(
        lat_values[
            lat_idx
        ]
    )

    nearest_lon = float(
        lon_values[
            lon_idx
        ]
    )


    print(
        "\nRequested location:"
    )

    print(
        "Latitude:",
        TARGET_LAT
    )

    print(
        "Longitude:",
        TARGET_LON
    )


    print(
        "\nNearest IMERG pixel:"
    )

    print(
        "Latitude index:",
        lat_idx
    )

    print(
        "Latitude:",
        nearest_lat
    )

    print(
        "Longitude index:",
        lon_idx
    )

    print(
        "Longitude:",
        nearest_lon
    )


    # --------------------------------------------------------
    # IMERG dimensions:
    #
    # time, lon, lat
    #
    # Extract ONLY:
    # one time
    # one longitude
    # one latitude
    # --------------------------------------------------------

    print(
        "\nExtracting Mumbai precipitation..."
    )

    mumbai_value = precipitation[
        0,
        lon_idx,
        lat_idx
    ]

    mumbai_value = float(
        mumbai_value
    )


    # --------------------------------------------------------
    # Read metadata
    # --------------------------------------------------------

    units = (
        precipitation.attrs
        .get(
            "Units",
            b"mm/hr"
        )
    )

    if isinstance(
        units,
        bytes
    ):
        units = units.decode()


    fill_value = (
        precipitation.attrs
        .get(
            "_FillValue",
            -9999.9
        )
    )

    if isinstance(
        fill_value,
        np.ndarray
    ):
        fill_value = float(
            fill_value
        )


# ============================================================
# 7. CHECK MISSING VALUE
# ============================================================

is_missing = np.isclose(
    mumbai_value,
    float(fill_value),
    atol=0.01
)


# ============================================================
# 8. FINAL RESULT
# ============================================================

print("\n")
print("=" * 60)
print("MUMBAI IMERG RESULT")
print("=" * 60)

print(
    "Requested latitude :",
    TARGET_LAT
)

print(
    "Requested longitude:",
    TARGET_LON
)

print(
    "IMERG latitude     :",
    nearest_lat
)

print(
    "IMERG longitude    :",
    nearest_lon
)

print(
    "Latitude index     :",
    lat_idx
)

print(
    "Longitude index    :",
    lon_idx
)

print(
    "Precipitation      :",
    mumbai_value
)

print(
    "Units              :",
    units
)

print("=" * 60)


if is_missing:

    print(
        "\nWARNING:"
    )

    print(
        "IMERG returned a missing-value flag."
    )

else:

    print(
        "\nSUCCESS:"
    )

    print(
        "Valid IMERG precipitation value."
    )