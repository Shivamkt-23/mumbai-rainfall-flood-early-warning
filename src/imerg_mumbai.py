import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import earthaccess
import h5py
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

CONCEPT_ID = "C2723754847-GES_DISC"

TARGET_LAT = 19.0
TARGET_LON = 73.0

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
# LOGIN
# ============================================================

def login_earthdata():
    """
    Login using EARTHDATA_TOKEN from the environment.
    """

    earthaccess.login(
        strategy="environment"
    )


# ============================================================
# FIND IMERG GRANULE
# ============================================================

def find_granule(start_time):
    """
    Find the IMERG half-hour granule covering start_time.

    start_time:
        timezone-aware datetime in UTC
    """

    if start_time.tzinfo is None:
        raise ValueError(
            "start_time must be timezone-aware."
        )

    start_time = start_time.astimezone(
        timezone.utc
    )

    end_time = (
        start_time
        + timedelta(minutes=30)
    )

    granules = earthaccess.search_data(
        concept_id=CONCEPT_ID,
        temporal=(
            start_time.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            end_time.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        ),
        count=5
    )

    if not granules:
        raise RuntimeError(
            f"No IMERG granule found for {start_time}"
        )

    # Prefer the granule whose beginning time
    # exactly matches the requested time.
    requested = start_time

    for granule in granules:

        temporal = (
            granule
            .get("umm", {})
            .get("TemporalExtent", {})
        )

        # Metadata layout can vary, so this is
        # intentionally defensive.
        beginning = None

        try:
            beginning = (
                temporal
                ["RangeDateTime"]
                ["BeginningDateTime"]
            )
        except Exception:
            pass

        if beginning:

            beginning_dt = datetime.fromisoformat(
                beginning.replace(
                    "Z",
                    "+00:00"
                )
            )

            if beginning_dt == requested:
                return granule

    # If exact matching is unavailable,
    # use the first returned granule.
    return granules[0]


# ============================================================
# GET DOWNLOAD URL
# ============================================================

def get_download_url(granule):
    """
    Extract the authenticated HTTPS download URL.
    """

    for link in granule.get(
        "umm",
        {}
    ).get(
        "RelatedUrls",
        []
    ):

        if link.get("Type") == "GET DATA":

            return link.get("URL")

    # Fallback
    data_urls = getattr(
        granule,
        "data",
        None
    )

    if data_urls:
        return data_urls[0]

    raise RuntimeError(
        "Could not find IMERG download URL."
    )


# ============================================================
# DOWNLOAD GRANULE
# ============================================================

def download_granule(
    download_url
):
    """
    Download IMERG granule if it is not already cached.
    """

    filename = download_url.split("/")[-1]

    local_file = (
        DOWNLOAD_DIR
        / filename
    )

    if local_file.exists():
        return local_file

    print(
        f"Downloading {filename}..."
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

    with open(
        local_file,
        "wb"
    ) as f:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if chunk:
                f.write(chunk)

    return local_file


# ============================================================
# EXTRACT MUMBAI PIXEL
# ============================================================

def extract_mumbai_rainfall(
    local_file
):
    """
    Extract nearest IMERG pixel to Mumbai.

    Returns:
        rainfall_rate_mm_hr
        rainfall_depth_mm
        latitude
        longitude
    """

    with h5py.File(
        local_file,
        "r"
    ) as h5:

        lat_values = h5[
            "Grid/lat"
        ][...]

        lon_values = h5[
            "Grid/lon"
        ][...]

        precipitation = h5[
            "Grid/precipitation"
        ]

        # --------------------------------------------
        # Find nearest pixel
        # --------------------------------------------

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

        # --------------------------------------------
        # IMERG dimensions:
        #
        # time, lon, lat
        # --------------------------------------------

        rainfall_rate = float(
            precipitation[
                0,
                lon_idx,
                lat_idx
            ]
        )

        # --------------------------------------------
        # Fill value
        # --------------------------------------------

        fill_value = (
            precipitation
            .attrs
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

        # --------------------------------------------
        # Missing-value check
        # --------------------------------------------

        if np.isclose(
            rainfall_rate,
            float(fill_value),
            atol=0.01
        ):
            rainfall_rate = np.nan

        # --------------------------------------------
        # Half-hour rainfall depth
        #
        # Rate is mm/hr
        # Product interval is 30 min
        # --------------------------------------------

        if np.isnan(
            rainfall_rate
        ):

            rainfall_depth = np.nan

        else:

            rainfall_depth = (
                rainfall_rate
                * 0.5
            )

    return {
        "latitude": nearest_lat,
        "longitude": nearest_lon,
        "rainfall_rate_mm_hr": rainfall_rate,
        "rainfall_depth_mm": rainfall_depth,
        "latitude_index": lat_idx,
        "longitude_index": lon_idx,
    }


# ============================================================
# MAIN REUSABLE FUNCTION
# ============================================================

def get_imerg_mumbai(
    start_time
):
    """
    Get IMERG rainfall for Mumbai for one 30-minute period.

    Example:

        start_time = datetime(
            2024,
            9,
            25,
            17,
            0,
            tzinfo=timezone.utc
        )

    Returns a dictionary containing:
        timestamp
        rainfall_rate_mm_hr
        rainfall_depth_mm
        latitude
        longitude
    """

    if start_time.tzinfo is None:

        raise ValueError(
            "start_time must include UTC timezone."
        )

    start_time = start_time.astimezone(
        timezone.utc
    )

    # --------------------------------------------------------
    # Find granule
    # --------------------------------------------------------

    granule = find_granule(
        start_time
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    download_url = get_download_url(
        granule
    )

    local_file = download_granule(
        download_url
    )

    # --------------------------------------------------------
    # Extract Mumbai
    # --------------------------------------------------------

    result = extract_mumbai_rainfall(
        local_file
    )

    # --------------------------------------------------------
    # Add timestamp and source
    # --------------------------------------------------------

    result["timestamp"] = (
        start_time.isoformat()
    )

    result["source"] = (
        "NASA GPM IMERG Final V07"
    )

    result["file"] = (
        str(local_file)
    )

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    login_earthdata()

    test_time = datetime(
        2024,
        9,
        25,
        17,
        0,
        tzinfo=timezone.utc
    )

    result = get_imerg_mumbai(
        test_time
    )

    print()
    print("=" * 60)
    print("IMERG MUMBAI EXTRACTION")
    print("=" * 60)

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )

    print("=" * 60)