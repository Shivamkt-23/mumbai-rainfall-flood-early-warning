from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# DATA PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "mumbai_era5_ml_ready_cleaned.csv"
)


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> None:

    print("\n" + "=" * 50)
    print("DATASET VALIDATION")
    print("=" * 50)

    # -----------------------------------------------------
    # 1. BASIC INFORMATION
    # -----------------------------------------------------

    print("\n[1] BASIC INFORMATION")

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print(
        "Date range:",
        df["datetime"].min(),
        "->",
        df["datetime"].max()
    )

    # -----------------------------------------------------
    # 2. MISSING VALUES
    # -----------------------------------------------------

    print("\n[2] MISSING VALUES")

    missing = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    print(missing.head(15))

    # -----------------------------------------------------
    # 3. DUPLICATE TIMESTAMPS
    # -----------------------------------------------------

    print("\n[3] DUPLICATE TIMESTAMPS")

    duplicate_count = (
        df["datetime"]
        .duplicated()
        .sum()
    )

    print("Duplicate timestamps:", duplicate_count)

    # -----------------------------------------------------
    # 4. TIME GAPS
    # -----------------------------------------------------

    print("\n[4] TIME GAPS")

    ordered = (
        df.sort_values("datetime")
        .reset_index(drop=True)
    )

    time_diff = (
        ordered["datetime"]
        .diff()
    )

    print(
        time_diff
        .value_counts()
        .sort_index(ascending=False)
        .head(15)
    )

    # -----------------------------------------------------
    # 5. INVALID NEXT-HOUR TARGETS
    # -----------------------------------------------------

    print("\n[5] NEXT-HOUR TARGET VALIDATION")

    expected_next_time = (
        ordered["datetime"]
        + pd.Timedelta(hours=1)
    )

    actual_next_time = (
        ordered["datetime"]
        .shift(-1)
    )

    valid_next_hour = (
        actual_next_time == expected_next_time
    )

    invalid_boundaries = (
        (~valid_next_hour)
        .iloc[:-1]
        .sum()
    )

    print(
        "Invalid next-hour boundaries:",
        invalid_boundaries
    )

    # -----------------------------------------------------
    # 6. SPATIAL COVERAGE
    # -----------------------------------------------------

    print("\n[6] SPATIAL COVERAGE")

    spatial_columns = {
        "latitude_center",
        "longitude_center",
    }

    if spatial_columns.issubset(df.columns):

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
            "Unique spatial points:",
            len(locations)
        )

        print(locations)

    else:

        print(
            "Latitude/longitude columns not found."
        )

    # -----------------------------------------------------
    # 7. TARGET INFORMATION
    # -----------------------------------------------------

    print("\n[7] TARGET")

    if "next_hour_rain" in df.columns:

        print(
            df["next_hour_rain"]
            .describe()
        )

    else:

        print(
            "next_hour_rain column not found."
        )

    # -----------------------------------------------------
    # DONE
    # -----------------------------------------------------

    print("\n" + "=" * 50)
    print("VALIDATION COMPLETE")
    print("=" * 50 + "\n")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("Loading:", DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    if df["datetime"].isna().any():

        print(
            "WARNING: Invalid datetime values found."
        )

    validate_dataset(df)


if __name__ == "__main__":
    main()