#!/usr/bin/env python3
"""
train_final_multisource_2010_2025.py

Final historical model training for the Mumbai flood/rainfall prototype.

Input:
    data/processed/mumbai_era5_imerg_multisource_2010_2025.csv

Target:
    next_hour_rain

Chronological split:
    TRAIN = 2010-01-02 through 2024-12-31
    TEST  = 2025-01-01 through 2025-09-30

Models:
    1) ERA5-only Random Forest
    2) ERA5 + IMERG Random Forest
    3) ERA5-only XGBoost
    4) ERA5 + IMERG XGBoost

Purpose:
    Fairly measure whether the historical IMERG features add value
    over the ERA5-only model on the same 2025 holdout period.

Outputs:
    models/final_multisource_random_forest.pkl
    models/final_multisource_xgboost.json
    models/final_training_metrics.json
    models/final_multisource_test_predictions.csv
    models/final_multisource_feature_importance.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise SystemExit(
        "xgboost is not installed. Run:\n"
        "python -m pip install -U xgboost"
    ) from exc


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "mumbai_era5_imerg_multisource_2010_2025.csv"
)

MODEL_DIR = BASE_DIR / "models"

RF_PATH = MODEL_DIR / "final_multisource_random_forest.pkl"
XGB_PATH = MODEL_DIR / "final_multisource_xgboost.json"
METRICS_PATH = MODEL_DIR / "final_training_metrics.json"
PREDICTIONS_PATH = MODEL_DIR / "final_multisource_test_predictions.csv"
IMPORTANCE_PATH = MODEL_DIR / "final_multisource_feature_importance.csv"


# ============================================================
# CONFIG
# ============================================================

TARGET = "next_hour_rain"
HEAVY_THRESHOLD_MM = 5.0

TRAIN_END = pd.Timestamp("2024-12-31 23:59:59")
TEST_START = pd.Timestamp("2025-01-01 00:00:00")
TEST_END = pd.Timestamp("2025-09-30 23:59:59")

# ERA5 columns that must never be model inputs.
NON_FEATURE_COLUMNS = {
    "datetime",
    "date",
    TARGET,
}

# Constant coordinates are kept out of the model.
CONSTANT_COLUMNS = {
    "latitude_center",
    "longitude_center",
    "imerg_lat",
    "imerg_lon",
}


# ============================================================
# HELPERS
# ============================================================

def banner(text: str) -> None:
    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE_mm": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_mm": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def heavy_metrics(y_true, y_pred) -> dict:
    mask = np.asarray(y_true) >= HEAVY_THRESHOLD_MM

    count = int(mask.sum())

    if count == 0:
        return {
            "count": 0,
            "MAE_mm": None,
            "RMSE_mm": None,
        }

    return {
        "count": count,
        "MAE_mm": float(mean_absolute_error(
            np.asarray(y_true)[mask],
            np.asarray(y_pred)[mask],
        )),
        "RMSE_mm": float(np.sqrt(mean_squared_error(
            np.asarray(y_true)[mask],
            np.asarray(y_pred)[mask],
        ))),
    }


def prepare_features(df: pd.DataFrame, include_imerg: bool):
    columns = []

    for col in df.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        if col in CONSTANT_COLUMNS:
            continue
        if not include_imerg and col.startswith("imerg_"):
            continue
        columns.append(col)

    # Target must never appear in X.
    if TARGET in columns:
        columns.remove(TARGET)

    X = df[columns].copy()
    return X, columns


def sanitize_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df[columns].copy()

    for col in columns:
        out[col] = pd.to_numeric(
            out[col],
            errors="coerce",
        )

    if out.isna().any().any():
        missing = {
            col: int(out[col].isna().sum())
            for col in columns
            if out[col].isna().any()
        }
        raise RuntimeError(
            f"Missing/non-numeric model features detected: {missing}"
        )

    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    banner("FINAL 2010-2025 MULTISOURCE MODEL TRAINING")

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Multisource dataset not found:\n{DATA_PATH}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    banner("1. LOAD DATA")

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["datetime", "date"],
    )

    required = {
        "datetime",
        "date",
        TARGET,
        "imerg_prev_day_mm",
        "imerg_3day_sum_mm",
        "imerg_7day_sum_mm",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Required columns missing: {missing}"
        )

    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"Rows       : {len(df):,}")
    print(f"Start      : {df['datetime'].min()}")
    print(f"End        : {df['datetime'].max()}")
    print(f"Columns    : {len(df.columns)}")

    # Drop rows with missing target only.
    before = len(df)
    df = df.dropna(subset=[TARGET]).copy()

    if len(df) != before:
        print(
            f"Dropped {before - len(df)} rows with missing target."
        )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    banner("2. CHRONOLOGICAL TRAIN / TEST SPLIT")

    train = df[
        (df["datetime"] <= TRAIN_END)
    ].copy()

    test = df[
        (df["datetime"] >= TEST_START)
        & (df["datetime"] <= TEST_END)
    ].copy()

    if train.empty:
        raise RuntimeError("Training set is empty.")

    if test.empty:
        raise RuntimeError("2025 test set is empty.")

    print(
        f"Train rows : {len(train):,} | "
        f"{train['datetime'].min()} -> {train['datetime'].max()}"
    )
    print(
        f"Test rows  : {len(test):,} | "
        f"{test['datetime'].min()} -> {test['datetime'].max()}"
    )

    # --------------------------------------------------------
    # Feature sets
    # --------------------------------------------------------

    X_train_era5, era5_features = prepare_features(
        train,
        include_imerg=False,
    )

    X_test_era5, _ = prepare_features(
        test,
        include_imerg=False,
    )

    X_train_multi, multi_features = prepare_features(
        train,
        include_imerg=True,
    )

    X_test_multi, _ = prepare_features(
        test,
        include_imerg=True,
    )

    X_train_era5 = sanitize_numeric(
        X_train_era5,
        era5_features,
    )
    X_test_era5 = sanitize_numeric(
        X_test_era5,
        era5_features,
    )

    X_train_multi = sanitize_numeric(
        X_train_multi,
        multi_features,
    )
    X_test_multi = sanitize_numeric(
        X_test_multi,
        multi_features,
    )

    y_train = pd.to_numeric(
        train[TARGET],
        errors="coerce",
    ).to_numpy()

    y_test = pd.to_numeric(
        test[TARGET],
        errors="coerce",
    ).to_numpy()

    if np.isnan(y_train).any() or np.isnan(y_test).any():
        raise RuntimeError("Target contains NaN after numeric conversion.")

    print()
    print(f"ERA5-only features : {len(era5_features)}")
    print(f"Multisource features: {len(multi_features)}")
    print(f"Target              : {TARGET}")

    # --------------------------------------------------------
    # Model 1: ERA5-only Random Forest
    # --------------------------------------------------------

    banner("3. ERA5-ONLY RANDOM FOREST")

    rf_era5 = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=1,
    )

    rf_era5.fit(
        X_train_era5,
        y_train,
    )

    pred_rf_era5 = np.clip(
        rf_era5.predict(X_test_era5),
        0,
        None,
    )

    rf_era5_metrics = regression_metrics(
        y_test,
        pred_rf_era5,
    )

    rf_era5_heavy = heavy_metrics(
        y_test,
        pred_rf_era5,
    )

    print(rf_era5_metrics)
    print("Heavy:", rf_era5_heavy)

    # --------------------------------------------------------
    # Model 2: ERA5 + IMERG Random Forest
    # --------------------------------------------------------

    banner("4. ERA5 + IMERG RANDOM FOREST")

    rf_multi = RandomForestRegressor(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=1,
    )

    rf_multi.fit(
        X_train_multi,
        y_train,
    )

    pred_rf_multi = np.clip(
        rf_multi.predict(X_test_multi),
        0,
        None,
    )

    rf_multi_metrics = regression_metrics(
        y_test,
        pred_rf_multi,
    )

    rf_multi_heavy = heavy_metrics(
        y_test,
        pred_rf_multi,
    )

    print(rf_multi_metrics)
    print("Heavy:", rf_multi_heavy)

    # --------------------------------------------------------
    # Model 3: ERA5-only XGBoost
    # --------------------------------------------------------

    banner("5. ERA5-ONLY XGBOOST")

    xgb_era5 = XGBRegressor(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )

    xgb_era5.fit(
        X_train_era5,
        y_train,
    )

    pred_xgb_era5 = np.clip(
        xgb_era5.predict(X_test_era5),
        0,
        None,
    )

    xgb_era5_metrics = regression_metrics(
        y_test,
        pred_xgb_era5,
    )

    xgb_era5_heavy = heavy_metrics(
        y_test,
        pred_xgb_era5,
    )

    print(xgb_era5_metrics)
    print("Heavy:", xgb_era5_heavy)

    # --------------------------------------------------------
    # Model 4: ERA5 + IMERG XGBoost
    # --------------------------------------------------------

    banner("6. ERA5 + IMERG XGBOOST")

    xgb_multi = XGBRegressor(
        n_estimators=700,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )

    xgb_multi.fit(
        X_train_multi,
        y_train,
    )

    pred_xgb_multi = np.clip(
        xgb_multi.predict(X_test_multi),
        0,
        None,
    )

    xgb_multi_metrics = regression_metrics(
        y_test,
        pred_xgb_multi,
    )

    xgb_multi_heavy = heavy_metrics(
        y_test,
        pred_xgb_multi,
    )

    print(xgb_multi_metrics)
    print("Heavy:", xgb_multi_heavy)

    # --------------------------------------------------------
    # Save prediction table
    # --------------------------------------------------------

    banner("7. SAVE PREDICTIONS")

    predictions = pd.DataFrame({
        "datetime": test["datetime"].to_numpy(),
        "actual_next_hour_rain": y_test,
        "rf_era5_prediction": pred_rf_era5,
        "rf_multisource_prediction": pred_rf_multi,
        "xgb_era5_prediction": pred_xgb_era5,
        "xgb_multisource_prediction": pred_xgb_multi,
    })

    predictions["actual_heavy"] = (
        predictions["actual_next_hour_rain"]
        >= HEAVY_THRESHOLD_MM
    ).astype(int)

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print(f"Saved: {PREDICTIONS_PATH}")

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    banner("8. SAVE MULTISOURCE FEATURE IMPORTANCE")

    rf_importance = pd.DataFrame({
        "feature": multi_features,
        "rf_importance": rf_multi.feature_importances_,
        "xgb_importance": xgb_multi.feature_importances_,
    }).sort_values(
        "rf_importance",
        ascending=False,
    )

    rf_importance.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    print(f"Saved: {IMPORTANCE_PATH}")

    print()
    print("Top multisource features:")
    print(
        rf_importance.head(20).to_string(index=False)
    )

    # --------------------------------------------------------
    # Save model artifacts
    # --------------------------------------------------------

    banner("9. SAVE FINAL MODEL ARTIFACTS")

    # Save the multisource RF as a normal joblib artifact.
    joblib.dump(
        rf_multi,
        RF_PATH,
    )

    xgb_multi.save_model(
        XGB_PATH,
    )

    print(f"RF : {RF_PATH}")
    print(f"XGB: {XGB_PATH}")

    # --------------------------------------------------------
    # Metrics JSON
    # --------------------------------------------------------

    metrics = {
        "dataset": {
            "path": str(DATA_PATH),
            "rows": int(len(df)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_start": str(train["datetime"].min()),
            "train_end": str(train["datetime"].max()),
            "test_start": str(test["datetime"].min()),
            "test_end": str(test["datetime"].max()),
        },
        "target": TARGET,
        "heavy_threshold_mm": HEAVY_THRESHOLD_MM,
        "feature_counts": {
            "era5_only": len(era5_features),
            "multisource": len(multi_features),
            "added_imerg_features": len(
                set(multi_features) - set(era5_features)
            ),
        },
        "models": {
            "era5_random_forest": {
                "overall": rf_era5_metrics,
                "heavy": rf_era5_heavy,
            },
            "multisource_random_forest": {
                "overall": rf_multi_metrics,
                "heavy": rf_multi_heavy,
            },
            "era5_xgboost": {
                "overall": xgb_era5_metrics,
                "heavy": xgb_era5_heavy,
            },
            "multisource_xgboost": {
                "overall": xgb_multi_metrics,
                "heavy": xgb_multi_heavy,
            },
        },
    }

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Metrics: {METRICS_PATH}")

    # --------------------------------------------------------
    # Final console summary
    # --------------------------------------------------------

    banner("10. FINAL TEST SUMMARY")

    rows = [
        ("ERA5 RF", rf_era5_metrics, rf_era5_heavy),
        ("ERA5 + IMERG RF", rf_multi_metrics, rf_multi_heavy),
        ("ERA5 XGB", xgb_era5_metrics, xgb_era5_heavy),
        ("ERA5 + IMERG XGB", xgb_multi_metrics, xgb_multi_heavy),
    ]

    print(
        f"{'Model':24s} "
        f"{'MAE':>10s} "
        f"{'RMSE':>10s} "
        f"{'R2':>10s} "
        f"{'Heavy N':>10s} "
        f"{'Heavy MAE':>12s}"
    )

    for name, overall, heavy in rows:
        heavy_mae = heavy["MAE_mm"]
        heavy_mae_text = (
            f"{heavy_mae:.4f}"
            if heavy_mae is not None
            else "N/A"
        )

        print(
            f"{name:24s} "
            f"{overall['MAE_mm']:10.4f} "
            f"{overall['RMSE_mm']:10.4f} "
            f"{overall['R2']:10.4f} "
            f"{heavy['count']:10d} "
            f"{heavy_mae_text:>12s}"
        )

    print()
    print("=" * 80)
    print("FINAL MODEL TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
