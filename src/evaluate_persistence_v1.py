from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =========================================================
# PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mumbai_era5_baseline_v1_features.csv"
)


# =========================================================
# MAIN
# =========================================================

def main():

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    df = (
        df.sort_values("datetime")
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Same 80/20 chronological split as Random Forest
    # -----------------------------------------------------

    split = int(
        len(df) * 0.80
    )

    test = (
        df.iloc[split:]
        .copy()
    )

    # -----------------------------------------------------
    # Persistence prediction
    #
    # Prediction for next hour =
    # current observed rainfall
    # -----------------------------------------------------

    actual = test["next_hour_rain"].to_numpy()

    predicted = test["rainfall_mm"].to_numpy()

    predicted = np.maximum(
        predicted,
        0.0
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print("\n" + "=" * 50)
    print("PERSISTENCE BASELINE")
    print("=" * 50)

    print(
        f"MAE  : {mae:.6f} mm"
    )

    print(
        f"RMSE : {rmse:.6f} mm"
    )

    print(
        f"R²   : {r2:.6f}"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()