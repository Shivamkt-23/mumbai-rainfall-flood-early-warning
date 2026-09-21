from pathlib import Path
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OLD_DATA = (
    PROJECT_ROOT
    / "data"
    / "mumbai_era5_ml_ready_cleaned.csv"
)

NEW_DATA = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ERA5_2010_2019"
    / "era5_2010_2019_converted.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mumbai_era5_2010_2025_base.csv"
)


# =========================================================
# BASE COLUMNS
# =========================================================

BASE_COLUMNS = [
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


# =========================================================
# LOAD OLD DATA
# =========================================================

def load_old_data():

    print("\nLoading existing 2020–2025 dataset...")

    df = pd.read_csv(
        OLD_DATA
    )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    missing_columns = [
        column
        for column in BASE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns in existing dataset: "
            + ", ".join(missing_columns)
        )

    df = df[
        BASE_COLUMNS
    ].copy()

    return df


# =========================================================
# LOAD NEW DATA
# =========================================================

def load_new_data():

    print(
        "Loading 2010–2019 converted dataset..."
    )

    df = pd.read_csv(
        NEW_DATA
    )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    missing_columns = [
        column
        for column in BASE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns in 2010–2019 dataset: "
            + ", ".join(missing_columns)
        )

    df = df[
        BASE_COLUMNS
    ].copy()

    return df


# =========================================================
# VALIDATION
# =========================================================

def validate_merged(df):

    print("\n" + "=" * 60)
    print("MERGED DATASET VALIDATION")
    print("=" * 60)

    print(
        "\nRows:",
        len(df)
    )

    print(
        "Date range:",
        df["datetime"].min(),
        "->",
        df["datetime"].max()
    )

    # -----------------------------------------------------
    # Missing values
    # -----------------------------------------------------

    print("\nMissing values:")

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
    # Time gaps
    # -----------------------------------------------------

    ordered = (
        df.sort_values("datetime")
        .reset_index(drop=True)
    )

    time_diff = (
        ordered["datetime"]
        .diff()
        .dropna()
    )

    gap_counts = (
        time_diff
        .value_counts()
        .sort_index(ascending=False)
    )

    print("\nLargest time gaps:")

    print(
        gap_counts.head(10)
    )

    non_hourly = (
        (time_diff != pd.Timedelta(hours=1))
        .sum()
    )

    print(
        "\nNon-1-hour gaps:",
        non_hourly
    )

    # -----------------------------------------------------
    # Spatial coverage
    # -----------------------------------------------------

    locations = (
        ordered[
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

    print(
        locations
    )

    print(
        "\n" + "=" * 60
    )


# =========================================================
# MAIN
# =========================================================

def main():

    old_df = load_old_data()

    new_df = load_new_data()

    print(
        "\n2010–2019 rows:",
        len(new_df)
    )

    print(
        "Existing rows:",
        len(old_df)
    )

    # -----------------------------------------------------
    # Combine
    # -----------------------------------------------------

    merged = pd.concat(
        [
            new_df,
            old_df,
        ],
        ignore_index=True
    )

    # -----------------------------------------------------
    # Sort chronologically
    # -----------------------------------------------------

    merged = (
        merged
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Remove duplicate timestamps
    # -----------------------------------------------------

    before = len(merged)

    merged = (
        merged
        .drop_duplicates(
            subset=["datetime"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    removed = (
        before - len(merged)
    )

    print(
        "\nDuplicate rows removed:",
        removed
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    merged.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # -----------------------------------------------------
    # Validate
    # -----------------------------------------------------

    validate_merged(
        merged
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()