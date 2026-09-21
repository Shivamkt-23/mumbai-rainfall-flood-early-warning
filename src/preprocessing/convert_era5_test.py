from pathlib import Path
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ERA5_2010_2019"
    / "reanalysis-era5-single-levels-timeseries-sfc340llkm4.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ERA5_2010_2019"
    / "era5_2010_test_converted.csv"
)


# =========================================================
# CONVERSION
# =========================================================

def convert_era5(df):

    # Datetime
    df["datetime"] = pd.to_datetime(
        df["valid_time"],
        errors="coerce"
    )

    # ERA5 units → project units
    df["temperature_c"] = df["t2m"] - 273.15
    df["dewpoint_c"] = df["d2m"] - 273.15
    df["pressure_hpa"] = df["msl"] / 100.0

    # ERA5 total precipitation is metres.
    # Convert metres → millimetres.
    df["rainfall_mm"] = df["tp"] * 1000.0

    # Wind is already m/s
    df["u10_ms"] = df["u10"]
    df["v10_ms"] = df["v10"]

    # Location
    df["latitude_center"] = df["latitude"]
    df["longitude_center"] = df["longitude"]

    # Keep only the columns needed by our project
    result = df[
        [
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
    ].copy()

    # Sort
    result = (
        result
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return result


# =========================================================
# MAIN
# =========================================================

def main():

    print("\nLoading ERA5 test file...")

    df = pd.read_csv(INPUT_PATH)

    print(
        "Raw shape:",
        df.shape
    )

    converted = convert_era5(df)

    print(
        "Converted shape:",
        converted.shape
    )

    print("\nConverted columns:")

    print(
        converted.columns.tolist()
    )

    print("\nFirst 5 rows:")

    print(
        converted.head().to_string(index=False)
    )

    print("\nData types:")

    print(
        converted.dtypes
    )

    converted.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSaved to:")

    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()