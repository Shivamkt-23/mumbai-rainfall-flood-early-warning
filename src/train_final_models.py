from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "mumbai_storm_multisource_v1.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])

df = df.sort_values("datetime").reset_index(drop=True)

print("=" * 70)
print("FINAL RF vs XGBOOST MODEL COMPARISON")
print("=" * 70)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
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

TARGET = "next_hour_rain"


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [f for f in FEATURES if f not in df.columns]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_idx = int(len(df) * 0.80)

train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print("\n" + "=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print(f"Train rows: {len(train_df)}")
print(f"Test rows : {len(test_df)}")

print(f"\nTrain period:")
print(f"  {train_df['datetime'].min()} → {train_df['datetime'].max()}")

print(f"\nTest period:")
print(f"  {test_df['datetime'].min()} → {test_df['datetime'].max()}")


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = np.maximum(rf.predict(X_test), 0)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)


# ============================================================
# XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

xgb = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train, y_train)

xgb_pred = np.maximum(xgb.predict(X_test), 0)

xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)


# ============================================================
# HEAVY-RAIN EVALUATION
# ============================================================

# We evaluate heavy-rain prediction separately.
# Threshold is 5 mm in the next hour.

HEAVY_THRESHOLD = 5.0

heavy_mask = y_test >= HEAVY_THRESHOLD

if heavy_mask.sum() > 0:

    rf_heavy_mae = mean_absolute_error(
        y_test[heavy_mask],
        rf_pred[heavy_mask]
    )

    rf_heavy_rmse = np.sqrt(
        mean_squared_error(
            y_test[heavy_mask],
            rf_pred[heavy_mask]
        )
    )

    xgb_heavy_mae = mean_absolute_error(
        y_test[heavy_mask],
        xgb_pred[heavy_mask]
    )

    xgb_heavy_rmse = np.sqrt(
        mean_squared_error(
            y_test[heavy_mask],
            xgb_pred[heavy_mask]
        )
    )

else:
    rf_heavy_mae = None
    rf_heavy_rmse = None
    xgb_heavy_mae = None
    xgb_heavy_rmse = None


# ============================================================
# RESULTS
# ============================================================

results = {
    "dataset": {
        "rows": len(df),
        "features": len(FEATURES),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_start": str(train_df["datetime"].min()),
        "train_end": str(train_df["datetime"].max()),
        "test_start": str(test_df["datetime"].min()),
        "test_end": str(test_df["datetime"].max())
    },

    "random_forest": {
        "MAE": float(rf_mae),
        "RMSE": float(rf_rmse),
        "R2": float(rf_r2),
        "heavy_rain_MAE": (
            None if rf_heavy_mae is None else float(rf_heavy_mae)
        ),
        "heavy_rain_RMSE": (
            None if rf_heavy_rmse is None else float(rf_heavy_rmse)
        )
    },

    "xgboost": {
        "MAE": float(xgb_mae),
        "RMSE": float(xgb_rmse),
        "R2": float(xgb_r2),
        "heavy_rain_MAE": (
            None if xgb_heavy_mae is None else float(xgb_heavy_mae)
        ),
        "heavy_rain_RMSE": (
            None if xgb_heavy_rmse is None else float(xgb_heavy_rmse)
        )
    },

    "heavy_rain_threshold_mm": HEAVY_THRESHOLD,
    "heavy_rain_test_samples": int(heavy_mask.sum())
}


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL RESULTS")
print("=" * 70)

print("\nRandom Forest")
print(f"MAE  : {rf_mae:.6f}")
print(f"RMSE : {rf_rmse:.6f}")
print(f"R²   : {rf_r2:.6f}")

print("\nXGBoost")
print(f"MAE  : {xgb_mae:.6f}")
print(f"RMSE : {xgb_rmse:.6f}")
print(f"R²   : {xgb_r2:.6f}")

print("\nHeavy-rain test samples:", heavy_mask.sum())

if heavy_mask.sum() > 0:
    print("\nHeavy Rain Performance")

    print("\nRandom Forest")
    print(f"MAE  : {rf_heavy_mae:.6f}")
    print(f"RMSE : {rf_heavy_rmse:.6f}")

    print("\nXGBoost")
    print(f"MAE  : {xgb_heavy_mae:.6f}")
    print(f"RMSE : {xgb_heavy_rmse:.6f}")


# ============================================================
# DETERMINE WINNER
# ============================================================

# Primary criterion: RMSE
# Secondary criterion: MAE

if (
    xgb_rmse < rf_rmse
    or (
        np.isclose(xgb_rmse, rf_rmse)
        and xgb_mae < rf_mae
    )
):
    winner = "XGBoost"
else:
    winner = "Random Forest"

results["selected_model"] = winner


print("\n" + "=" * 70)
print("SELECTED MODEL")
print("=" * 70)

print(winner)


# ============================================================
# SAVE MODELS
# ============================================================

with open(MODEL_DIR / "final_random_forest.pkl", "wb") as f:
    pickle.dump(rf, f)

xgb.save_model(MODEL_DIR / "final_xgboost.json")


# ============================================================
# SAVE METRICS
# ============================================================

with open(MODEL_DIR / "final_model_comparison.json", "w") as f:
    json.dump(results, f, indent=2)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

rf_importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

rf_importance.to_csv(
    MODEL_DIR / "final_rf_feature_importance.csv",
    index=False
)


xgb_importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": xgb.feature_importances_
}).sort_values("importance", ascending=False)

xgb_importance.to_csv(
    MODEL_DIR / "final_xgb_feature_importance.csv",
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

predictions = test_df[
    ["datetime", "window_id", TARGET]
].copy()

predictions["rf_prediction"] = rf_pred
predictions["xgb_prediction"] = xgb_pred

predictions.to_csv(
    OUTPUT_DIR / "final_model_test_predictions.csv",
    index=False
)


print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(MODEL_DIR / "final_random_forest.pkl")
print(MODEL_DIR / "final_xgboost.json")
print(MODEL_DIR / "final_model_comparison.json")
print(MODEL_DIR / "final_rf_feature_importance.csv")
print(MODEL_DIR / "final_xgb_feature_importance.csv")
print(OUTPUT_DIR / "final_model_test_predictions.csv")

print("\nDone.")