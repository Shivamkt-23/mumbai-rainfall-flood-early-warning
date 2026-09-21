from datetime import datetime, timezone, timedelta

import h5py
import numpy as np

from imerg_mumbai import (
    login_earthdata,
    find_granule,
    get_download_url,
    download_granule,
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_LAT = 19.0
TARGET_LON = 73.0


# ============================================================
# HELPER: CLEAN FILL VALUES
# ============================================================

def clean_fill_values(
    values,
    dataset
):
    """
    Convert IMERG fill values to NaN.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    fill_value = dataset.attrs.get(
        "_FillValue",
        None
    )

    if fill_value is not None:

        if isinstance(
            fill_value,
            np.ndarray
        ):
            fill_value = float(
                fill_value
            )

        values[
            np.isclose(
                values,
                float(fill_value),
                atol=0.01
            )
        ] = np.nan

    return values


# ============================================================
# FIND NEAREST MUMBAI PIXEL
# ============================================================

def get_mumbai_indices(
    lat,
    lon
):
    """
    Return nearest IMERG grid indices for Mumbai.
    """

    lat_idx = int(
        np.abs(
            lat - TARGET_LAT
        ).argmin()
    )

    lon_idx = int(
        np.abs(
            lon - TARGET_LON
        ).argmin()
    )

    return (
        lat_idx,
        lon_idx
    )


# ============================================================
# EXTRACT FEATURES FROM ONE HALF-HOUR GRANULE
# ============================================================

def extract_half_hour_features(
    start_time
):
    """
    Extract satellite rainfall and uncertainty features
    from one IMERG 30-minute granule.

    Returns rainfall in mm for the 30-minute interval.
    """

    if start_time.tzinfo is None:
        raise ValueError(
            "start_time must be timezone-aware."
        )

    start_time = start_time.astimezone(
        timezone.utc
    )

    # --------------------------------------------------------
    # Find and download granule
    # --------------------------------------------------------

    granule = find_granule(
        start_time
    )

    download_url = get_download_url(
        granule
    )

    local_file = download_granule(
        download_url
    )

    # --------------------------------------------------------
    # Open HDF5
    # --------------------------------------------------------

    with h5py.File(
        local_file,
        "r"
    ) as h5:

        lat = h5[
            "Grid/lat"
        ][...]

        lon = h5[
            "Grid/lon"
        ][...]

        precipitation = h5[
            "Grid/precipitation"
        ]

        quality = h5[
            "Grid/precipitationQualityIndex"
        ]

        random_error = h5[
            "Grid/randomError"
        ]

        # ----------------------------------------------------
        # Mumbai pixel
        # ----------------------------------------------------

        lat_idx, lon_idx = (
            get_mumbai_indices(
                lat,
                lon
            )
        )

        nearest_lat = float(
            lat[lat_idx]
        )

        nearest_lon = float(
            lon[lon_idx]
        )

        # ----------------------------------------------------
        # 3x3 neighborhood
        # ----------------------------------------------------

        lat_start = max(
            0,
            lat_idx - 1
        )

        lat_end = min(
            len(lat),
            lat_idx + 2
        )

        lon_start = max(
            0,
            lon_idx - 1
        )

        lon_end = min(
            len(lon),
            lon_idx + 2
        )

        # ----------------------------------------------------
        # Rainfall
        #
        # IMERG dimensions:
        # time, lon, lat
        #
        # Units:
        # mm/hr
        #
        # Convert to 30-minute depth:
        # rate * 0.5
        # ----------------------------------------------------

        rain_values = precipitation[
            0,
            lon_start:lon_end,
            lat_start:lat_end
        ]

        rain_values = clean_fill_values(
            rain_values,
            precipitation
        )

        # ----------------------------------------------------
        # Quality index
        # ----------------------------------------------------

        quality_values = quality[
            0,
            lon_start:lon_end,
            lat_start:lat_end
        ]

        quality_values = clean_fill_values(
            quality_values,
            quality
        )

        # ----------------------------------------------------
        # Random error
        # ----------------------------------------------------

        error_values = random_error[
            0,
            lon_start:lon_end,
            lat_start:lat_end
        ]

        error_values = clean_fill_values(
            error_values,
            random_error
        )

        # ----------------------------------------------------
        # Center pixel positions
        # ----------------------------------------------------

        center_row = (
            lat_idx
            - lat_start
        )

        center_col = (
            lon_idx
            - lon_start
        )

        # ----------------------------------------------------
        # Center rainfall
        # ----------------------------------------------------

        center_rain_rate = float(
            rain_values[
                center_col,
                center_row
            ]
        )

        center_rain_depth = (
            center_rain_rate
            * 0.5
            if not np.isnan(
                center_rain_rate
            )
            else np.nan
        )

        # ----------------------------------------------------
        # 3x3 rainfall statistics
        # ----------------------------------------------------

        rain_mean_rate = (
            np.nanmean(
                rain_values
            )
        )

        rain_max_rate = (
            np.nanmax(
                rain_values
            )
        )

        rain_mean_depth = (
            rain_mean_rate
            * 0.5
        )

        rain_max_depth = (
            rain_max_rate
            * 0.5
        )

        # ----------------------------------------------------
        # Center quality
        # ----------------------------------------------------

        center_quality = float(
            quality_values[
                center_col,
                center_row
            ]
        )

        # ----------------------------------------------------
        # 3x3 quality statistics
        # ----------------------------------------------------

        quality_mean = (
            np.nanmean(
                quality_values
            )
        )

        quality_min = (
            np.nanmin(
                quality_values
            )
        )

        # ----------------------------------------------------
        # Center random error
        # ----------------------------------------------------

        center_random_error = float(
            error_values[
                center_col,
                center_row
            ]
        )

        # ----------------------------------------------------
        # 3x3 random-error statistics
        # ----------------------------------------------------

        random_error_mean = (
            np.nanmean(
                error_values
            )
        )

        random_error_max = (
            np.nanmax(
                error_values
            )
        )

    return {
        "timestamp": start_time.isoformat(),

        "latitude": nearest_lat,
        "longitude": nearest_lon,

        # -------------------------
        # Rainfall
        # -------------------------

        "imerg_center_30m_mm":
            center_rain_depth,

        "imerg_mean_3x3_30m_mm":
            rain_mean_depth,

        "imerg_max_3x3_30m_mm":
            rain_max_depth,

        # -------------------------
        # Quality
        # -------------------------

        "imerg_quality_center":
            center_quality,

        "imerg_quality_mean_3x3":
            quality_mean,

        "imerg_quality_min_3x3":
            quality_min,

        # -------------------------
        # Error
        # -------------------------

        "imerg_random_error_center":
            center_random_error,

        "imerg_random_error_mean_3x3":
            random_error_mean,

        "imerg_random_error_max_3x3":
            random_error_max,

        "source":
            "NASA GPM IMERG Final V07",

        "file":
            str(local_file)
    }


# ============================================================
# HOURLY IMERG FEATURES
# ============================================================

def get_hourly_imerg_features(
    hour_start
):
    """
    Combine two IMERG half-hour granules into one
    hourly satellite feature record.
    """

    if hour_start.tzinfo is None:
        raise ValueError(
            "hour_start must be timezone-aware."
        )

    hour_start = hour_start.astimezone(
        timezone.utc
    )

    # --------------------------------------------------------
    # First half-hour
    # --------------------------------------------------------

    first = extract_half_hour_features(
        hour_start
    )

    # --------------------------------------------------------
    # Second half-hour
    # --------------------------------------------------------

    second = extract_half_hour_features(
        hour_start
        + timedelta(minutes=30)
    )

    # --------------------------------------------------------
    # Hourly rainfall
    # --------------------------------------------------------

    hourly_center = (
        first["imerg_center_30m_mm"]
        + second["imerg_center_30m_mm"]
    )

    hourly_mean = (
        first["imerg_mean_3x3_30m_mm"]
        + second["imerg_mean_3x3_30m_mm"]
    )

    hourly_max = (
        first["imerg_max_3x3_30m_mm"]
        + second["imerg_max_3x3_30m_mm"]
    )

    # --------------------------------------------------------
    # Quality aggregation
    #
    # Mean over the two half-hours.
    # --------------------------------------------------------

    quality_center = np.nanmean([
        first["imerg_quality_center"],
        second["imerg_quality_center"]
    ])

    quality_mean = np.nanmean([
        first["imerg_quality_mean_3x3"],
        second["imerg_quality_mean_3x3"]
    ])

    quality_min = np.nanmin([
        first["imerg_quality_min_3x3"],
        second["imerg_quality_min_3x3"]
    ])

    # --------------------------------------------------------
    # Random error aggregation
    # --------------------------------------------------------

    random_error_center = np.nanmean([
        first["imerg_random_error_center"],
        second["imerg_random_error_center"]
    ])

    random_error_mean = np.nanmean([
        first["imerg_random_error_mean_3x3"],
        second["imerg_random_error_mean_3x3"]
    ])

    random_error_max = np.nanmax([
        first["imerg_random_error_max_3x3"],
        second["imerg_random_error_max_3x3"]
    ])

    return {
        "timestamp":
            hour_start.isoformat(),

        "latitude":
            first["latitude"],

        "longitude":
            first["longitude"],

        # -------------------------
        # Hourly rainfall
        # -------------------------

        "imerg_center_hourly_mm":
            hourly_center,

        "imerg_mean_3x3_hourly_mm":
            hourly_mean,

        "imerg_max_3x3_hourly_mm":
            hourly_max,

        # -------------------------
        # Quality
        # -------------------------

        "imerg_quality_center":
            quality_center,

        "imerg_quality_mean_3x3":
            quality_mean,

        "imerg_quality_min_3x3":
            quality_min,

        # -------------------------
        # Random error
        # -------------------------

        "imerg_random_error_center":
            random_error_center,

        "imerg_random_error_mean_3x3":
            random_error_mean,

        "imerg_random_error_max_3x3":
            random_error_max,

        "source":
            "NASA GPM IMERG Final V07"
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Logging into NASA Earthdata..."
    )

    login_earthdata()

    test_time = datetime(
        2024,
        9,
        25,
        17,
        0,
        tzinfo=timezone.utc
    )

    print()
    print(
        "Extracting hourly IMERG features..."
    )

    result = get_hourly_imerg_features(
        test_time
    )

    print()
    print("=" * 70)
    print("IMERG SATELLITE FEATURE BLOCK")
    print("=" * 70)

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )

    print("=" * 70)