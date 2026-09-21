from pathlib import Path
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "mumbai_era5_ml_ready_cleaned.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "mumbai_era5_baseline_v1.csv"
)


# =========================================================
# BUILD DATASET
# =========================================================

def build_dataset():

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_PATH)

    # -----------------------------------------------------
    # Datetime
    # -----------------------------------------------------

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    # Remove invalid timestamps
    df = df.dropna(
        subset=["datetime"]
    )

    # Sort chronologically
    df = (
        df.sort_values("datetime")
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Check duplicates
    # -----------------------------------------------------

    duplicate_count = (
        df["datetime"]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:

        print(
            f"Removing {duplicate_count} duplicate timestamps."
        )

        df = (
            df.drop_duplicates(
                subset=["datetime"],
                keep="first"
            )
            .reset_index(drop=True)
        )

    # -----------------------------------------------------
    # Rebuild next-hour target
    # -----------------------------------------------------

    next_timestamp = (
        df["datetime"].shift(-1)
    )

    next_rainfall = (
        df["rainfall_mm"].shift(-1)
    )

    # A valid target exists ONLY when the next
    # observation is exactly one hour later.

    continuous = (
        next_timestamp - df["datetime"]
        == pd.Timedelta(hours=1)
    )

    df["next_hour_rain"] = (
        next_rainfall.where(continuous)
    )

    # -----------------------------------------------------
    # Count invalid targets
    # -----------------------------------------------------

    invalid_target_count = (
        df["next_hour_rain"].isna().sum()
    )

    print(
        "Rows before cleaning:",
        len(df)
    )

    print(
        "Invalid next-hour targets:",
        invalid_target_count
    )

    # -----------------------------------------------------
    # Remove rows without valid target
    # -----------------------------------------------------

    df = (
        df.dropna(
            subset=["next_hour_rain"]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nRows after cleaning:",
        len(df)
    )

    print(
        "Saved to:",
        OUTPUT_PATH
    )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    remaining_next_timestamp = (
        df["datetime"].shift(-1)
    )

    remaining_continuous = (
        remaining_next_timestamp - df["datetime"]
        == pd.Timedelta(hours=1)
    )

    # Last row has no next row, so exclude it.
    invalid_remaining = (
        (~remaining_continuous)
        .iloc[:-1]
        .sum()
    )

    print(
        "Remaining invalid boundaries:",
        invalid_remaining
    )

    print("\nDataset created successfully.")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    build_dataset()