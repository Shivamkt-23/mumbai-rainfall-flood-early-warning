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
    / "mumbai_era5_2010_2025_base.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mumbai_era5_2010_2025_ml_ready.csv"
)


# =========================================================
# BUILD FEATURES
# =========================================================

def build_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:

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

    # First row starts first block
    new_block.iloc[0] = True

    df["block_id"] = new_block.cumsum()

    print(
        "Continuous blocks:",
        df["block_id"].nunique()
    )

    # -----------------------------------------------------
    # NEXT-HOUR TARGET
    # -----------------------------------------------------

    df["next_hour_rain"] = (
        df.groupby("block_id")["rainfall_mm"]
        .shift(-1)
    )

    # -----------------------------------------------------
    # LAG FEATURES
    # -----------------------------------------------------

    for lag in [1, 2, 3, 6, 12, 24]:

        df[f"rainfall_lag_{lag}h"] = (
            df.groupby("block_id")["rainfall_mm"]
            .shift(lag)
        )

    # -----------------------------------------------------
    # ROLLING RAINFALL
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
    # TIME FEATURES
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
    # Required ML columns
    # -----------------------------------------------------

    required_columns = [
        "next_hour_rain",

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

    # -----------------------------------------------------
    # Remove rows without enough history / target
    # -----------------------------------------------------

    before = len(df)

    df = (
        df.dropna(
            subset=required_columns
        )
        .reset_index(drop=True)
    )

    removed = before - len(df)

    print(
        "Rows removed:",
        removed
    )

    # -----------------------------------------------------
    # Remove helper
    # -----------------------------------------------------

    df = df.drop(
        columns=["block_id"]
    )

    return df


# =========================================================
# VALIDATION
# =========================================================

def validate(df):

    print("\n" + "=" * 60)
    print("2010–2025 ML DATASET VALIDATION")
    print("=" * 60)

    print(
        "\nShape:",
        df.shape
    )

    print(
        "\nDate range:",
        df["datetime"].min(),
        "->",
        df["datetime"].max()
    )

    print("\nMissing values:")

    print(
        df.isna().sum()
    )

    # -----------------------------------------------------
    # Target
    # -----------------------------------------------------

    print(
        "\nTarget statistics:"
    )

    print(
        df["next_hour_rain"]
        .describe()
    )

    # -----------------------------------------------------
    # Timestamp duplicates
    # -----------------------------------------------------

    print(
        "\nDuplicate timestamps:",
        df["datetime"].duplicated().sum()
    )

    # -----------------------------------------------------
    # Spatial locations
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
        "\nUnique spatial points:",
        len(locations)
    )

    print(locations)

    print(
        "\n" + "=" * 60
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "\nLoading merged 2010–2025 dataset..."
    )

    df = pd.read_csv(
        INPUT_PATH
    )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    print(
        "Input rows:",
        len(df)
    )

    df = build_ml_dataset(
        df
    )

    validate(
        df
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


if __name__ == "__main__":
    main()