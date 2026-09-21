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
    / "full"
    / "reanalysis-era5-single-levels-timeseries-sfcehulg9ij.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ERA5_2010_2019"
    / "era5_2010_2019_converted.csv"
)


# =========================================================
# CONVERSION
# =========================================================

def convert_era5(df):

    # -----------------------------------------------------
    # Datetime
    # -----------------------------------------------------

    df["datetime"] = pd.to_datetime(
        df["valid_time"],
        errors="coerce"
    )

    # -----------------------------------------------------
    # Unit conversions
    # -----------------------------------------------------

    # Kelvin → Celsius
    df["temperature_c"] = (
        df["t2m"] - 273.15
    )

    df["dewpoint_c"] = (
        df["d2m"] - 273.15
    )

    # Pa → hPa
    df["pressure_hpa"] = (
        df["msl"] / 100.0
    )

    # metres → millimetres
    df["rainfall_mm"] = (
        df["tp"] * 1000.0
    )

    # Wind already in m/s
    df["u10_ms"] = df["u10"]
    df["v10_ms"] = df["v10"]

    # -----------------------------------------------------
    # Location
    # -----------------------------------------------------

    df["latitude_center"] = df["latitude"]
    df["longitude_center"] = df["longitude"]

    # -----------------------------------------------------
    # Select project columns
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Sort
    # -----------------------------------------------------

    result = (
        result
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return result


# =========================================================
# VALIDATION
# =========================================================

def validate(df):

    print("\n" + "=" * 55)
    print("CONVERTED DATASET VALIDATION")
    print("=" * 55)

    print(
        "\nShape:",
        df.shape
    )

    print(
        "\nDate range:"
    )

    print(
        df["datetime"].min(),
        "->",
        df["datetime"].max()
    )

    print(
        "\nMissing values:"
    )

    print(
        df.isna().sum()
    )

    # -----------------------------------------------------
    # Duplicate timestamps
    # -----------------------------------------------------

    duplicates = (
        df["datetime"]
        .duplicated()
        .sum()
    )

    print(
        "\nDuplicate timestamps:",
        duplicates
    )

    # -----------------------------------------------------
    # Time continuity
    # -----------------------------------------------------

    time_diff = (
        df["datetime"]
        .diff()
        .dropna()
    )

    expected = pd.Timedelta(hours=1)

    invalid_gaps = (
        time_diff != expected
    ).sum()

    print(
        "Non-1-hour gaps:",
        invalid_gaps
    )

    # -----------------------------------------------------
    # Spatial coverage
    # -----------------------------------------------------

    locations = (
        df[
            [
                "latitude_center",
                "longitude_center",
            ]
        ]
        .drop_duplicates()
    )

    print(
        "\nUnique locations:",
        len(locations)
    )

    print(locations)

    # -----------------------------------------------------
    # Rainfall
    # -----------------------------------------------------

    print(
        "\nRainfall statistics:"
    )

    print(
        df["rainfall_mm"]
        .describe()
    )

    print(
        "\n" + "=" * 55
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("\nLoading full 2010–2019 ERA5 file...")

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        "Raw shape:",
        df.shape
    )

    converted = convert_era5(
        df
    )

    validate(
        converted
    )

    converted.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()