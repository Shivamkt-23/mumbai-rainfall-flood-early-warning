from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mumbai_era5_2010_2025_ml_ready.csv"
)


def main():

    df = pd.read_csv(DATA_PATH)

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    # -----------------------------------------------------
    # Find rainfall periods in 2024
    # -----------------------------------------------------

    rainy = (
        df[
            (df["datetime"] >= "2024-01-01")
            & (df["datetime"] < "2025-01-01")
            & (df["rainfall_mm"] > 1.0)
        ]
        .sort_values(
            "rainfall_mm",
            ascending=False
        )
        .head(20)
    )

    print("\n" + "=" * 70)
    print("TOP 2024 RAINFALL PERIODS")
    print("=" * 70)

    print(
        rainy[
            [
                "datetime",
                "rainfall_mm"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()