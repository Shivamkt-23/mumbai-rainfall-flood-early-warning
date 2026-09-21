from pathlib import Path
import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mumbai_era5_baseline_v1.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mumbai_era5_baseline_v1_features.csv"
)


# =========================================================
# FEATURE BUILDING
# =========================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:

    df = (
        df.sort_values("datetime")
        .reset_index(drop=True)
        .copy()
    )

    # -----------------------------------------------------
    # Identify continuous hourly blocks
    # -----------------------------------------------------

    time_diff = df["datetime"].diff()

    new_block = (
        time_diff != pd.Timedelta(hours=1)
    )

    # First row starts the first block
    new_block.iloc[0] = True

    df["block_id"] = new_block.cumsum()

    # -----------------------------------------------------
    # Build lag features INSIDE each continuous block
    # -----------------------------------------------------

    for lag in [1, 2, 3, 6, 12, 24]:

        df[f"rainfall_lag_{lag}h"] = (
            df.groupby("block_id")["rainfall_mm"]
            .shift(lag)
        )

    # -----------------------------------------------------
    # Rolling rainfall
    #
    # Includes the current hour because current rainfall
    # is known when predicting the next hour.
    # -----------------------------------------------------

    for window in [3, 6, 24]:

        df[f"rain_{window}h"] = (
            df.groupby("block_id")["rainfall_mm"]
            .transform(
                lambda s: s.rolling(
                    window=window,
                    min_periods=window
                ).sum()
            )
        )

    # -----------------------------------------------------
    # Time features
    # -----------------------------------------------------

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["day_of_year"] = df["datetime"].dt.dayofyear

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["doy_sin"] = np.sin(
        2 * np.pi * df["day_of_year"] / 365
    )

    df["doy_cos"] = np.cos(
        2 * np.pi * df["day_of_year"] / 365
    )

    # -----------------------------------------------------
    # Remove rows that don't have enough history
    # for the 24-hour features.
    # -----------------------------------------------------

    required_feature_columns = [
        "rainfall_lag_1h",
        "rainfall_lag_2h",
        "rainfall_lag_3h",
        "rainfall_lag_6h",
        "rainfall_lag_12h",
        "rainfall_lag_24h",
        "rain_3h",
        "rain_6h",
        "rain_24h",
    ]

    before = len(df)

    df = (
        df.dropna(
            subset=required_feature_columns
        )
        .reset_index(drop=True)
    )

    removed = before - len(df)

    print(
        "Rows removed because of insufficient "
        f"history: {removed}"
    )

    # -----------------------------------------------------
    # Remove helper column
    # -----------------------------------------------------

    df = df.drop(columns=["block_id"])

    return df


# =========================================================
# MAIN
# =========================================================

def main():

    print("\nLoading baseline dataset...")

    df = pd.read_csv(INPUT_PATH)

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    print(
        "Rows before feature generation:",
        len(df)
    )

    df = build_features(df)

    print(
        "Rows after feature generation:",
        len(df)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_PATH
    )

    print("\nFeature columns:")

    print(
        df.columns.tolist()
    )


if __name__ == "__main__":
    main()