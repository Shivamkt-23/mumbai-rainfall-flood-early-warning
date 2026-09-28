from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


BASE_DIR = Path(__file__).resolve().parents[1]

DEM_PATH = (
    BASE_DIR
    / "data"
    / "dem"
    / "mumbai_dem.tif"
)


def add_elevation_to_spatial(
    spatial: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add SRTM elevation to each spatial monitoring location.

    The DEM is used only as a spatial terrain layer.
    It is NOT added to the trained rainfall model features.

    Returns:
        spatial dataframe with:
            elevation_m
            elevation_source
    """

    result = spatial.copy()

    if not DEM_PATH.exists():
        raise FileNotFoundError(
            f"DEM file not found: {DEM_PATH}"
        )

    elevations = []
    sources = []

    with rasterio.open(DEM_PATH) as dem:

        coordinates = [
            (
                float(row["longitude"]),
                float(row["latitude"]),
            )
            for _, row in result.iterrows()
        ]

        sampled_values = dem.sample(
            coordinates,
            masked=True,
        )

        for sample in sampled_values:

            value = sample[0]

            if np.ma.is_masked(value):

                elevations.append(np.nan)
                sources.append("DEM NoData")

                continue

            value = float(value)

            if not np.isfinite(value):

                elevations.append(np.nan)
                sources.append("DEM NoData")

                continue

            if (
                dem.nodata is not None
                and np.isclose(
                    value,
                    float(dem.nodata),
                )
            ):

                elevations.append(np.nan)
                sources.append("DEM NoData")

                continue

            elevations.append(value)
            sources.append("SRTM 1 Arc-Second")

    result["elevation_m"] = elevations
    result["elevation_source"] = sources

    return result