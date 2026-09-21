import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = (
    "data/processed/"
    "mumbai_multisource_pilot_v1.csv"
)

TARGET = "next_hour_rain"

TEST_FRACTION = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading multi-source pilot dataset...")

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["datetime"]
)

df["datetime"] = pd.to_datetime(
    df["datetime"],
    utc=True
)

df = (
    df
    .sort_values("datetime")
    .reset_index(drop=True)
)

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)


# ============================================================
# FEATURE COLUMNS
# ============================================================

feature_cols = [
    # ERA5
    "era5_rainfall_mm",
    "temperature_c",
    "dewpoint_c",
    "pressure_hpa",
    "u10_ms",
    "v10_ms",

    # Rainfall history
    "rain_3h",
    "rain_6h",
    "rain_24h",

    # IMERG rainfall
    "imerg_center_hourly_mm",
    "imerg_mean_3x3_hourly_mm",
    "imerg_max_3x3_hourly_mm",

    # IMERG quality
    "imerg_quality_center",
    "imerg_quality_mean_3x3",
    "imerg_quality_min_3x3",

    # IMERG uncertainty
    "imerg_random_error_center",
    "imerg_random_error_mean_3x3",
    "imerg_random_error_max_3x3",
]


# ============================================================
# VALIDATE COLUMNS
# ============================================================

missing_features = [
    col
    for col in feature_cols
    if col not in df.columns
]

if missing_features:
    raise RuntimeError(
        "Missing feature columns: "
        + ", ".join(missing_features)
    )

if TARGET not in df.columns:
    raise RuntimeError(
        f"Target column '{TARGET}' not found."
    )


# ============================================================
# CHECK MISSING VALUES
# ============================================================

missing_count = (
    df[
        feature_cols + [TARGET]
    ]
    .isna()
    .sum()
    .sum()
)

if missing_count != 0:
    raise RuntimeError(
        f"Found {missing_count} missing values."
    )


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_idx = int(
    len(df)
    * (1 - TEST_FRACTION)
)

train = df.iloc[
    :split_idx
].copy()

test = df.iloc[
    split_idx:
].copy()


print()
print("=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print(
    "Train rows:",
    len(train)
)

print(
    "Test rows:",
    len(test)
)

print(
    "Train period:",
    train["datetime"].iloc[0],
    "→",
    train["datetime"].iloc[-1]
)

print(
    "Test period:",
    test["datetime"].iloc[0],
    "→",
    test["datetime"].iloc[-1]
)


X_train = train[
    feature_cols
]

y_train = train[
    TARGET
]

X_test = test[
    feature_cols
]

y_test = test[
    TARGET
]


# ============================================================
# METRIC FUNCTION
# ============================================================

def evaluate_model(
    name,
    model,
    X_test,
    y_test,
):

    predictions = model.predict(
        X_test
    )

    predictions = np.maximum(
        predictions,
        0
    )

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

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        "MAE :",
        round(mae, 6)
    )

    print(
        "RMSE:",
        round(rmse, 6)
    )

    print(
        "R²  :",
        round(r2, 6)
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "predictions": predictions,
    }


# ============================================================
# 1. RANDOM FOREST
# ============================================================

print()
print(
    "Training Random Forest..."
)

rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
)

rf.fit(
    X_train,
    y_train
)

rf_results = evaluate_model(
    "RANDOM FOREST — ERA5 + IMERG",
    rf,
    X_test,
    y_test,
)


# ============================================================
# SAVE RF
# ============================================================

rf_path = (
    MODEL_DIR
    / "multisource_pilot_rf.pkl"
)

joblib.dump(
    {
        "model": rf,
        "features": feature_cols,
    },
    rf_path
)

print(
    "\nRF saved:",
    rf_path
)


# ============================================================
# 2. XGBOOST
# ============================================================

print()
print(
    "Training XGBoost..."
)

xgb = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
)

xgb.fit(
    X_train,
    y_train
)

xgb_results = evaluate_model(
    "XGBOOST — ERA5 + IMERG",
    xgb,
    X_test,
    y_test,
)


# ============================================================
# SAVE XGBOOST
# ============================================================

xgb_path = (
    MODEL_DIR
    / "multisource_pilot_xgboost.json"
)

xgb.save_model(
    xgb_path
)

print(
    "\nXGBoost saved:",
    xgb_path
)


# ============================================================
# FEATURE IMPORTANCE — XGBOOST
# ============================================================

importance = pd.DataFrame(
    {
        "feature": feature_cols,
        "importance": (
            xgb.feature_importances_
        ),
    }
)

importance = (
    importance
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)

importance_path = (
    MODEL_DIR
    / "multisource_pilot_xgb_feature_importance.csv"
)

importance.to_csv(
    importance_path,
    index=False
)


print()
print(
    "=" * 70
)

print(
    "TOP XGBOOST FEATURES"
)

print(
    importance.head(10).to_string(
        index=False
    )
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

predictions = test[
    [
        "datetime",
        TARGET,
    ]
].copy()

predictions[
    "rf_prediction"
] = rf_results[
    "predictions"
]

predictions[
    "xgb_prediction"
] = xgb_results[
    "predictions"
]

prediction_path = (
    "data/processed/"
    "multisource_pilot_predictions.csv"
)

predictions.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "dataset": "mumbai_multisource_pilot_v1",
    "rows": int(len(df)),
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "target": TARGET,
    "features": feature_cols,

    "random_forest": {
        "MAE": rf_results["MAE"],
        "RMSE": rf_results["RMSE"],
        "R2": rf_results["R2"],
    },

    "xgboost": {
        "MAE": xgb_results["MAE"],
        "RMSE": xgb_results["RMSE"],
        "R2": xgb_results["R2"],
    },

    "status": (
        "PILOT_ONLY — "
        "not final model evaluation"
    ),
}


summary_path = (
    MODEL_DIR
    / "multisource_pilot_model_comparison.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    f"RF      → "
    f"MAE={rf_results['MAE']:.6f}, "
    f"RMSE={rf_results['RMSE']:.6f}, "
    f"R²={rf_results['R2']:.6f}"
)

print(
    f"XGBoost → "
    f"MAE={xgb_results['MAE']:.6f}, "
    f"RMSE={xgb_results['RMSE']:.6f}, "
    f"R²={xgb_results['R2']:.6f}"
)

print()
print(
    "Prediction file:",
    prediction_path
)

print(
    "Summary file:",
    summary_path
)

print("=" * 70)