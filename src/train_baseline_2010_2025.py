from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mumbai_era5_2010_2025_ml_ready.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_PATH = (
    MODEL_DIR
    / "rainfall_model_2010_2025.pkl"
)

FEATURES_PATH = (
    MODEL_DIR
    / "rainfall_features_2010_2025.pkl"
)

METRICS_PATH = (
    MODEL_DIR
    / "baseline_2010_2025_metrics.json"
)


# =========================================================
# FEATURES
# =========================================================

FEATURE_COLUMNS = [
    "rainfall_mm",
    "temperature_c",
    "dewpoint_c",
    "pressure_hpa",
    "u10_ms",
    "v10_ms",
    "rain_3h",
    "rain_6h",
    "rain_24h",
    "rainfall_lag_1h",
    "rainfall_lag_2h",
    "rainfall_lag_3h",
    "rainfall_lag_6h",
    "rainfall_lag_12h",
    "rainfall_lag_24h",
    "year",
    "month",
    "day",
    "hour",
    "day_of_year",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]

TARGET_COLUMN = "next_hour_rain"


# =========================================================
# MAIN
# =========================================================

def main():

    print("\nLoading 2010–2025 ML dataset...")

    df = pd.read_csv(DATA_PATH)

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
    )

    df = (
        df.sort_values("datetime")
        .reset_index(drop=True)
    )

    print("Rows:", len(df))

    # -----------------------------------------------------
    # Validate columns
    # -----------------------------------------------------

    required = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    # -----------------------------------------------------
    # Remove missing values
    # -----------------------------------------------------

    df = (
        df.dropna(
            subset=required
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Chronological 80/20 split
    # -----------------------------------------------------

    split = int(
        len(df) * 0.80
    )

    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    print("\nTRAIN / TEST")
    print("-" * 50)

    print(
        "Training rows:",
        len(train_df)
    )

    print(
        "Testing rows:",
        len(test_df)
    )

    print(
        "Training period:",
        train_df["datetime"].min(),
        "->",
        train_df["datetime"].max()
    )

    print(
        "Testing period:",
        test_df["datetime"].min(),
        "->",
        test_df["datetime"].max()
    )

    # -----------------------------------------------------
    # X / y
    # -----------------------------------------------------

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    # -----------------------------------------------------
    # Same model as V1
    # -----------------------------------------------------

    print("\nTraining Random Forest...")

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    predictions = model.predict(
        X_test
    )

    predictions = np.maximum(
        predictions,
        0.0
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print("\n" + "=" * 55)
    print("EXPANDED ERA5 BASELINE RESULTS")
    print("=" * 55)

    print(
        f"MAE  : {mae:.6f} mm"
    )

    print(
        f"RMSE : {rmse:.6f} mm"
    )

    print(
        f"R²   : {r2:.6f}"
    )

    print("=" * 55)

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    joblib.dump(
        FEATURE_COLUMNS,
        FEATURES_PATH
    )

    metrics = {
        "version": "ERA5_2010_2025_BASELINE",
        "model": "RandomForestRegressor",
        "n_estimators": 150,
        "max_depth": 12,
        "rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_start": str(
            train_df["datetime"].min()
        ),
        "train_end": str(
            train_df["datetime"].max()
        ),
        "test_start": str(
            test_df["datetime"].min()
        ),
        "test_end": str(
            test_df["datetime"].max()
        ),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(
        "\nSaved model:",
        MODEL_PATH
    )

    print(
        "Saved metrics:",
        METRICS_PATH
    )


if __name__ == "__main__":
    main()