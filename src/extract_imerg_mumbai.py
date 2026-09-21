from pathlib import Path

import h5py
import numpy as np


# =========================================================
# PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "GPM_IMERG"
    / "3B-HHR.MS.MRG.3IMERG.20240901-S000000-E002959.0000.V07B.HDF5"
)


TARGET_LAT = 19.0
TARGET_LON = 73.0


# =========================================================
# MAIN
# =========================================================

def main():

    print("Opening IMERG file...")

    with h5py.File(FILE_PATH, "r") as hdf:

        # -------------------------------------------------
        # Read coordinates
        # -------------------------------------------------

        lat = hdf["/Grid/lat"][:]
        lon = hdf["/Grid/lon"][:]

        # -------------------------------------------------
        # Find nearest grid cell
        # -------------------------------------------------

        lat_index = np.abs(
            lat - TARGET_LAT
        ).argmin()

        lon_index = np.abs(
            lon - TARGET_LON
        ).argmin()

        selected_lat = float(
            lat[lat_index]
        )

        selected_lon = float(
            lon[lon_index]
        )

        # -------------------------------------------------
        # Extract precipitation
        # -------------------------------------------------

        precipitation = hdf[
            "/Grid/precipitation"
        ][0]

        value = float(
            precipitation[
                lon_index,
                lat_index
            ]
        )

        # -------------------------------------------------
        # Missing-value information
        # -------------------------------------------------

        dataset = hdf[
            "/Grid/precipitation"
        ]

        print("\n" + "=" * 60)
        print("IMERG MUMBAI EXTRACTION")
        print("=" * 60)

        print(
            "\nRequested location:"
        )

        print(
            f"Latitude : {TARGET_LAT}"
        )

        print(
            f"Longitude: {TARGET_LON}"
        )

        print(
            "\nNearest IMERG grid cell:"
        )

        print(
            f"Latitude : {selected_lat}"
        )

        print(
            f"Longitude: {selected_lon}"
        )

        print(
            f"\nArray indices:"
        )

        print(
            f"Latitude index : {lat_index}"
        )

        print(
            f"Longitude index: {lon_index}"
        )

        print(
            "\nIMERG precipitation rate:"
        )

        print(
            f"{value:.4f} mm/hour"
        )

        print(
            "\nDataset attributes:"
        )

        for key, val in dataset.attrs.items():

            print(
                f"{key}: {val}"
            )

        print(
            "\n" + "=" * 60
        )


if __name__ == "__main__":
    main()